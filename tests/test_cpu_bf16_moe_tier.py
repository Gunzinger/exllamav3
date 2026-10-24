"""Plan 07 §Design 3 unit tests: CPU prefill MoE tier — bucketed per-expert BF16 GEMM
(model-free).

Covers:
  (a) tier-vs-stock parity on synthetic MoE shapes: engineered per-expert counts straddling
      EXL3_CPU_PREFILL_MIN_M (7/8 at MIN_M 8, re-gated at MIN_M 9/7), multi-member and
      single-member power-of-2 buckets, cold-only rows bit-identical to the stock kernel
      (the tail reproduces the un-tiered call exactly), full-tensor class checks vs stock and
      vs fp64 truth (tier closer to truth than stock, same class contract as the linear tier).
  (b) bucketing correctness: row permutation preserved — the tier's expert-major gather /
      scatter reproduces the reference routing sum for every row (covered by (a)'s fp64 gate).
  (c) activation mirrors vs a literal transcription of moe_mul1.cpp apply_activation
      (silu/gelu/swiglu_oai/relu2 x limit 0 / nonzero).
  (d) dispatch contract: no expert at MIN_M -> False (caller runs the stock kernel; decode
      rows < MIN_M never enter).
  (e) info, not a gate: dequant ms for one qwen-shaped expert matrix (2560x640, K=2) — the
      per-chunk dequant-vs-GEMM decision number behind the cache bypass for routed experts.

Usage:
  TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD cpu-trial/.venv/bin/python tests/test_cpu_bf16_moe_tier.py
"""
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.block_sparse_mlp_cpu import _moe_tier_forward, _moe_tier_activation
from tests.test_cpu_bf16_prefill import make_linear, dequant_weight_fp32

MUL1_MULT = 0x83DCD12D
G = torch.Generator().manual_seed(0x70C)

K = 4
DIM_K, DIM_N = 256, 512
E = 6
THREADS = 4


# --------------------------------------------------------------------------- fixtures

class _FakeMoe:
    """Duck-typed stand-in for BlockSparseMLP_CPU's tier-relevant surface."""

    def __init__(self, gated = True):
        self.gated = gated
        self.activation_fn = "silu" if gated else "relu2"
        self.act_limit = 0.0
        self.cpu_layer_handle = None
        self.cpu_threads = THREADS
        self.num_local_experts = E
        self.gates = [None] * E
        self.ups = [None] * E
        self.downs = [None] * E


def _make_linear(seed, k = DIM_K, n = DIM_N, rate = K):
    trellis, suh, svh, Kf, swz = make_linear(k = k, n = n, K = rate, seed = seed)
    lin = LinearEXL3(None, k, n, suh = suh, svh = svh, trellis = trellis,
                     mul1 = torch.tensor(MUL1_MULT, dtype = torch.uint32).view(torch.int),
                     key = f"e{seed}")
    lin.cpu_repack()          # native -> band-8 packed, cpu_swz = True
    return lin


def make_module(gated = True):
    m = _FakeMoe(gated)
    lists = [m.gates, m.ups, m.downs] if gated else [m.ups, m.downs]
    for pi, ls in enumerate(lists):
        for e in range(E):
            # gate/up: hidden -> interm; down: interm -> hidden (the kernel's shape contract)
            ls[e] = _make_linear(seed = 1000 * (pi + 1) + e + (0 if gated else 500),
                                 k = DIM_K if ls is not m.downs else DIM_N,
                                 n = DIM_N if ls is not m.downs else DIM_K)
    def tensors(ls, attr):
        # entries are bare LinearEXL3 (the wrapper's .inner) — same getattr rule as the module
        return [getattr(getattr(ls[e], "inner", ls[e]), attr).contiguous() for e in range(E)]
    gate_t = ([tensors(m.gates, a) for a in ("trellis", "suh", "svh")]
              if gated else ([], [], []))
    up_t = [tensors(m.ups, a) for a in ("trellis", "suh", "svh")]
    down_t = [tensors(m.downs, a) for a in ("trellis", "suh", "svh")]
    swz = 1 if all(getattr(l, "inner", l).cpu_swz for ls in lists for l in ls) else 0
    m.cpu_layer_handle = ext.exl3_moe_cpu_make_layer(
        *gate_t, *up_t, *down_t, [], [], [],
        {"silu": 0, "relu2": 2}[m.activation_fn], 0.0, swz)
    return m


def make_routing(counts, min_m = 8):
    """[rows, topk] long selection + half weights realizing the given per-expert assignment
    counts exactly. Rows whose slots are all cold experts (count < min_m) are packed FIRST
    whenever the cold pool allows, so the tail's bit-identity check has a subject; the kernel
    and the tier see identical assignments either way."""
    topk = 3
    g = torch.Generator().manual_seed(0x71)
    cold = [e for e, c in enumerate(counts) if c < min_m]
    flat: list = []
    # all-cold rows: consume the cold pool in topk chunks first
    while sum(counts[e] for e in cold) >= topk:
        taken = 0
        for e in cold:
            if taken == topk:
                break
            if counts[e] > 0:
                flat.append(e)
                counts[e] -= 1
                taken += 1
    # remaining rows: round-robin the still-open experts so rows mix hot and cold slots
    while any(c > 0 for c in counts):
        for e in range(len(counts)):
            if counts[e] > 0:
                flat.append(e)
                counts[e] -= 1
    # no shuffle: the all-cold rows must survive reshaping intact (weights are seeded instead)
    flat = torch.tensor(flat, dtype = torch.long)
    rows = flat.numel() // topk
    sel = flat[: rows * topk].view(rows, topk).contiguous()
    w = (torch.rand(rows, topk, generator = g) * 0.5 + 0.5).half().contiguous()
    return sel, w


def moe_reference_fp64(module, x, sel, w, limit = 0.0):
    """fp64 truth: integer-exact dequant per expert + fp64 routing sum with the documented
    activation semantics (the same oracle order as tests/test_cpu_bf16_prefill.py)."""
    out = torch.zeros(x.shape[0], module.downs[0].out_features, dtype = torch.float64)
    flat = sel.reshape(-1)
    wf = w.reshape(-1).double()
    for i in range(flat.numel()):
        e = int(flat[i])
        if e < 0:
            continue
        t = i // sel.shape[1]
        xx = x[t]
        if module.gated:
            gW, uW, dW = (dequant_weight_fp32(l[e].trellis, l[e].suh,
                                              l[e].svh,
                                              grp = 8 if l[e].cpu_swz else 0,
                                              dtype = torch.float64).t()
                          for l in (module.gates, module.ups, module.downs))
            gv = gW @ xx
            uv = uW @ xx
            lim = limit if limit != 0.0 else float("inf")
            av = torch.clamp(gv / (1.0 + torch.exp((-gv).clamp(max = 80.0))), max = lim) \
                * uv.clamp(-lim, lim)
        else:
            uW, dW = (dequant_weight_fp32(l[e].trellis, l[e].suh, l[e].svh,
                                          grp = 8 if l[e].cpu_swz else 0,
                                          dtype = torch.float64).t()
                      for l in (module.ups, module.downs))
            av = (uW @ xx).clamp(min = 0.0) ** 2
        out[t] += wf[i] * (dW @ av)
    return out


# --------------------------------------------------------------------------- (a) + (b)

def run_tier_vs_stock(module, sel, w, x, min_m, label):
    os.environ["EXL3_CPU_PREFILL_MIN_M"] = str(min_m)
    try:
        # fp64 truth first: the oracle's barrier-heavy python loop is sensitive to the
        # thread-state the kernel/tier calls leave behind (quantum-quantized waits on this
        # host), and correctness must not depend on that ordering
        counts = torch.bincount(sel.reshape(-1), minlength = module.num_local_experts).tolist()
        hot = [e for e, c in enumerate(counts) if c >= min_m]
        ref = moe_reference_fp64(module, x.double(), sel, w) if hot else None

        out_stock = torch.empty(x.shape[0], module.downs[0].out_features, dtype = torch.float)
        ext.exl3_moe_cpu_forward(module.cpu_layer_handle, x, sel, w, out_stock, THREADS)
        out_tier = torch.zeros_like(out_stock)
        handled = _moe_tier_forward(module, x, sel, w, out_tier)

        assert handled == bool(hot), f"{label}: tier engagement {handled} != hot {bool(hot)}"
        if not handled:
            print(f"    {label}: no hot experts -> stock fallback (asserted)")
            return None

        # cold-only rows: the tail kernel computed them from the masked selection — exactly
        # the un-tiered call's arithmetic, so they must be BIT-identical
        is_hot = torch.zeros(module.num_local_experts, dtype = torch.bool)
        is_hot[torch.tensor(hot, dtype = torch.long)] = True
        row_hot = is_hot[sel].any(dim = 1)
        cold_rows = (~row_hot).nonzero(as_tuple = True)[0]
        if cold_rows.numel():
            assert torch.equal(out_tier[cold_rows], out_stock[cold_rows]), \
                f"{label}: cold-only rows not bit-identical to the stock kernel"
        else:
            print(f"    {label}: (no all-cold row in fixture — tail bit-identity checked "
                  f"in the other configs)")

        # fp64 truth + class gates (same contract as the linear tier); `ref` was computed
        # before the kernel/tier calls (see note above)
        scale = ref.abs().max().item()
        rel_stock = ((out_stock.double() - ref).abs().max() / scale).item()
        rel_tier = ((out_tier.double() - ref).abs().max() / scale).item()
        flips = (out_stock.argmax(-1) != out_tier.argmax(-1)).sum().item()
        print(f"    {label}: rows {x.shape[0]}, hot {hot}, "
              f"rel_stock {rel_stock:.5f} (int8 class), rel_tier {rel_tier:.5f} (bf16 class), "
              f"argmax flips {flips}/{x.shape[0]}, cold rows {cold_rows.numel()} bit-exact")
        assert rel_tier <= 1e-2, f"{label}: tier {rel_tier:.5f} out of bf16 class"
        assert rel_tier <= rel_stock + 2e-3, \
            f"{label}: tier farther from truth ({rel_tier:.5f}) than stock ({rel_stock:.5f})"
        return ref
    finally:
        os.environ.pop("EXL3_CPU_PREFILL_MIN_M", None)


def test_tier_parity_and_buckets():
    print("(a)+(b) tier-vs-stock parity, bucket shapes, row-permutation preservation")
    # counts straddling MIN_M=8 with multi-member (16) and single-member (8, 64) buckets;
    # experts 0 and 5 stay cold (3 assignments each); 48 assignments / topk 3 / 16 rows
    module = make_module(gated = True)
    sel, w = make_routing([3, 8, 9, 15, 10, 3])
    x = (0.5 + 0.1 * torch.randn(sel.shape[0], DIM_K, generator = G)).half().contiguous()
    run_tier_vs_stock(module, sel, w, x, min_m = 8,
                      label = "MIN_M=8 (8|9,15,10 -> buckets 8|16x3)")
    # re-gate at MIN_M=9: the count-8 expert drops to the tail; bucket 16 unchanged
    run_tier_vs_stock(module, sel, w, x, min_m = 9, label = "MIN_M=9 (count-8 expert demoted)")
    # straddle from the other side: a count of exactly 7 stays cold at MIN_M 8, promotes at 7
    sel7, w7 = make_routing([7, 8, 9, 15, 6, 3])
    x7 = (0.5 + 0.1 * torch.randn(sel7.shape[0], DIM_K, generator = G)).half().contiguous()
    run_tier_vs_stock(module, sel7, w7, x7, min_m = 8, label = "count 7 vs 8 straddle (7 cold)")
    run_tier_vs_stock(module, sel7, w7, x7, min_m = 7, label = "MIN_M=7 (count-7 expert promoted)")
    # one hot expert alone in its bucket
    run_tier_vs_stock(module, sel7, w7, x7, min_m = 15, label = "MIN_M=15 (single hot expert)")


def test_gateless_all_hot():
    print("(a+) gateless relu2 module, all-hot routing in one bucket")
    module = make_module(gated = False)
    sel, w = make_routing([8, 8, 8, 8, 8, 8])
    x = (0.5 + 0.1 * torch.randn(sel.shape[0], DIM_K, generator = G)).half().contiguous()
    run_tier_vs_stock(module, sel, w, x, min_m = 8,
                      label = "gateless relu2, all-hot bucket 8")


# --------------------------------------------------------------------------- (c)

def _kernel_act_reference(activation, g, u, limit):
    """Literal transcription of moe_mul1.cpp apply_activation (the spec)."""
    gf, uf, outf = g.tolist(), u.tolist(), torch.empty_like(g).flatten()
    lim = limit if limit != 0.0 else float("inf")
    exp_max = 80.0
    for i, (gv, uv) in enumerate(zip(gf, uf)):
        if activation == 0:
            av = min(gv / (1.0 + math.exp(min(-gv, exp_max))), lim)
            outf[i] = av * max(-lim, min(lim, uv))
        elif activation == 1:
            cdf = 0.5 * (1.0 + math.erf(gv * 0.70710678))
            outf[i] = min(gv * cdf, lim) * max(-lim, min(lim, uv))
        elif activation == 3:
            gv2 = min(gv, limit)
            uv2 = max(-limit, min(limit, uv))
            outf[i] = (uv2 + 1.0) * gv2 / (1.0 + math.exp(min(-1.702 * gv2, exp_max)))
        else:
            z = uv if uv > 0.0 else 0.0
            outf[i] = z * z
    return outf.view_as(g)


def test_activation_mirrors():
    print("(c) activation mirrors vs the kernel transcription")
    g = torch.randn(257, generator = G) * 3
    u = torch.randn(257, generator = G) * 3
    for act, name in ((0, "silu"), (1, "gelu"), (3, "swiglu_oai"), (2, "relu2")):
        for limit in (0.0, 7.0):
            if act == 3 and limit == 0.0:
                continue   # degenerate config (kernel clamps everything to 0); never shipped
            mine = _moe_tier_activation(act, g, u, limit)
            ref = _kernel_act_reference(act, g, u, limit)
            err = (mine - ref).abs().max().item()
            print(f"    {name:<11} limit {limit:<4} max-abs {err:.3e}")
            assert err <= 2e-5 * ref.abs().max().item() + 1e-6, \
                f"{name} limit {limit}: mirror out of fp32 class ({err:.3e})"


# --------------------------------------------------------------------------- (d)

def test_dispatch_contract():
    print("(d) dispatch: no expert at MIN_M -> stock fallback; decode rows never qualify")
    module = make_module(gated = True)
    sel, w = make_routing([2, 3, 2, 3, 2, 3])   # 15 assignments / topk 3 / 5 rows
    x = (0.5 + 0.1 * torch.randn(sel.shape[0], DIM_K, generator = G)).half().contiguous()
    os.environ["EXL3_CPU_PREFILL_MIN_M"] = "32"
    try:
        out = torch.zeros(sel.shape[0], DIM_N, dtype = torch.float)
        assert _moe_tier_forward(module, x, sel, w, out) is False, \
            "rows below MIN_M must report not-handled (caller then runs the stock kernel)"
    finally:
        os.environ.pop("EXL3_CPU_PREFILL_MIN_M", None)
    print("    5-row (decode-class) routing: not handled, as the exact gate requires")


# --------------------------------------------------------------------------- (e)

def test_dequant_cost():
    print("(e) dequant cost per qwen-shaped expert matrix (info, not a gate)")
    big = _make_linear(seed = 99, k = 2560, n = 640, rate = 2)
    t0 = time.perf_counter()
    for _ in range(4):
        w = ext.exl3_cpu_dequant_bf16_avx512(big.trellis, big.suh, big.svh, big.K, big.cpu_swz)
    per = (time.perf_counter() - t0) / 4 * 1e3
    mb = big.trellis.numel() * 2 / 2 ** 20
    chunk_s = per * 3 * 512 * 47 / 1e3
    print(f"    2560x640 K=2 expert matrix: {per:.2f} ms/dequant ({mb / (per / 1e3):.0f} MiB/s "
          f"packed) -> full routed set per 512-token chunk (512 exp x 3 mats x 47 layers) "
          f"~ {chunk_s:.1f} s")
    assert w.dtype == torch.bfloat16


def main():
    test_tier_parity_and_buckets()
    test_gateless_all_hot()
    test_activation_mirrors()
    test_dispatch_contract()
    test_dequant_cost()
    print("PASS: tests/test_cpu_bf16_moe_tier.py")


if __name__ == "__main__":
    main()
