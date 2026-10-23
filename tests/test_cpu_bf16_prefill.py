"""Plan 07 Option 1 unit tests: CPU prefill tier — dequant-to-BF16 + torch.matmul (model-free).

Covers:
  (a) ext.exl3_cpu_dequant_bf16 parity vs the pure-torch reference dequant (the literal port of
      the moe_mul1.cpp scalar decode used by tests over in cpu-trial, i.e. an independent oracle)
      at integer and half-integer K, native and band-8 packed layouts. Acceptance: within one
      BF16 ulp "class" — the reference and the C++ path accumulate the Hadamard passes in
      different fp32 orders, so a handful of elements can sit one rounding boundary apart.
  (b) tier dispatch: M < EXL3_CPU_PREFILL_MIN_M (and TIER=off) hits the stock mul1 binding
      exactly; M >= min takes the dequant+matmul path; the batched decode path defers to the
      per-linear forwards while the tier engages.
  (c) old-vs-new linear outputs at M=64/128: measured max-abs / relative deltas (the stock path
      carries int8 activation-quantization noise, the tier carries bf16 weight rounding) and
      argmax stability over the rows.

Usage:
  .venv-agnostic; from the worktree root with the CPU venv:
    TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD cpu-trial/.venv/bin/python tests/test_cpu_bf16_prefill.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3, cpu_prefill_torch_forward
from exllamav3.modules.quant import exl3 as exl3_mod

MUL1_MULT = 0x83DCD12D
K_INV = torch.tensor([0x1EEE], dtype = torch.uint16).view(torch.float16).float().item()

G = torch.Generator().manual_seed(0x07)


# --------------------------------------------------------------------------- reference

def tc_perm_inv() -> torch.Tensor:
    p = torch.empty(256, dtype = torch.long)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        for i, (r, c) in enumerate([(r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                                    (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8)]):
            p[t * 8 + i] = r * 16 + c
    inv = torch.empty(256, dtype = torch.long)
    inv[p] = torch.arange(256)
    return inv


def had128() -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < 128:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def dequant_weight_fp32(trellis, suh, svh, grp = 0, dtype = torch.float32) -> torch.Tensor:
    """Reference W [in, out] (mul1 codebook, any K incl. +0.5) — independent oracle.
    grp: band-8 packed trellis is undone to native tile order first. dtype: fp64 isolates the
    integer-exact decode (states -> codebook) from fp32 butterfly accumulation order."""
    if grp:
        kt, nt, words = trellis.shape
        native = torch.empty_like(trellis)
        native.view(kt, nt // grp, grp, words).copy_(
            trellis.view(nt // grp, kt, grp, words).permute(1, 0, 2, 3))
        trellis = native
    kt, nt, w = trellis.shape
    hb = (w % 16) == 8
    rate2 = 2 * (w // 16) + hb
    words32 = 4 * rate2
    u = trellis.view(torch.uint16).to(torch.int64).reshape(kt, nt, words32, 2)
    u32 = u[..., 0] | (u[..., 1] << 16)
    t = torch.arange(256)
    b1 = (((t + 1) * rate2) >> 1) + 128 * rate2
    b0 = b1 - 16
    shift = ((b1 - 1) // 32 + 1) * 32 - b1
    merged = (u32[:, :, (b0 // 32) % words32] << 32) | u32[:, :, ((b1 - 1) // 32) % words32]
    states = (merged >> shift) & 0xFFFF
    prod = (states * MUL1_MULT) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + (prod >> 24)
    val = (bsum.float() - 510.0) * K_INV
    val = val.to(dtype)
    what = val[:, :, tc_perm_inv()].reshape(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)
    k, n = what.shape
    H = had128().to(dtype)
    W = torch.empty(k, n, dtype = dtype)
    suhf, svhf = suh.to(dtype), svh.to(dtype)
    for kb in range(0, k, 128):
        for nb in range(0, n, 128):
            W[kb:kb + 128, nb:nb + 128] = (H @ what[kb:kb + 128, nb:nb + 128] @ H) \
                * (suhf[kb:kb + 128, None] * svhf[None, nb:nb + 128]) / 128.0
    return W


# --------------------------------------------------------------------------- fixtures

def make_linear(k = 256, n = 512, K = 4, swizzled = False, seed = 7):
    """Valid runtime trellis bytes: banded rates are group-8 packed (the loader repacks at first
    use, and make_matrix re-derives the flag from the rate), so the "native" arm sets
    EXL3_MOE_CPU_SWIZZLE=0 around creation. words = 16*bits (+8 at half-integer K)."""
    g = torch.Generator().manual_seed(seed)
    kt, nt = k // 16, n // 16
    hb = not float(K).is_integer()
    words = 16 * int(K) + (8 if hb else 0)
    trellis = torch.randint(0, 2 ** 16, (kt, nt, words), dtype = torch.int32, generator = g).to(torch.int16)
    suh = (torch.randn(k, generator = g) * 0.25).half()
    svh = (torch.randn(n, generator = g) * 0.25).half()
    old = os.environ.get("EXL3_MOE_CPU_SWIZZLE")
    try:
        # the group query is env-blind (the loader applies EXL3_MOE_CPU_SWIZZLE at repack time),
        # so the fixture gates it here exactly like the loader would
        grp = 0 if old == "0" else int(ext.exl3_moe_cpu_swizzle_group(float(K)))
    except Exception:
        grp = 0
    assert grp in (0, 8), f"planar layout (group {grp}) out of tier scope"
    swz = bool(grp) and swizzled
    if grp and swizzled:
        trellis = trellis.view(nt // grp, kt, grp, words).permute(1, 0, 2, 3).reshape(kt, nt, words).contiguous()
    return trellis, suh, svh, float(K), swz


def bf16_ulp(ref: torch.Tensor) -> torch.Tensor:
    a = ref.abs().double()
    ulp = torch.where(a > 0, torch.exp2(torch.floor(torch.log2(a)) - 7), torch.full_like(a, 1e-30))
    return ulp.float()


# --------------------------------------------------------------------------- (a)

def test_dequant_parity():
    print("(a) dequant-to-BF16 parity vs fp32 reference oracle")
    worst = 0.0
    for K in (1, 2, 4, 8, 2.5, 3.5):
        for swizzle_on in (False, True):
            # the dequant re-derives the packed layout from the rate under the live
            # EXL3_MOE_CPU_SWIZZLE, so bytes and call must see the same env
            os.environ["EXL3_MOE_CPU_SWIZZLE"] = "1" if swizzle_on else "0"
            try:
                trellis, suh, svh, Kf, swz = make_linear(K = K, swizzled = swizzle_on, seed = int(K * 10) + swizzle_on)
                if not exl3_mod.cpu_prefill_supported(type("_I", (), {"K": Kf})()):
                    print(f"    K={K:<4} swz={int(swz)}  SKIP (planar layout, stock path keeps this rate)")
                    continue
                w = ext.exl3_cpu_dequant_bf16(trellis, suh, svh, Kf, swz)
            finally:
                os.environ.pop("EXL3_MOE_CPU_SWIZZLE", None)
            assert w.dtype == torch.bfloat16 and w.shape == (512, 256)
            ref = dequant_weight_fp32(trellis, suh, svh, grp = 8 if swz else 0).t()   # [out, in]
            err = (w.float() - ref).abs()
            ratio = (err / bf16_ulp(ref)).max().item()
            worst = max(worst, ratio)
            # 1-ulp-class gate with an explicit fp32 cross-order noise floor: the C++ and the
            # torch oracle accumulate the 128-point Hadamards in different orders, so elements
            # where the oracle cancels to (near-)zero can differ by ~1e-8 absolute while being
            # meaningless in relative terms (measured: 2/131072 elements at K=1, all |ref|<=3e-8,
            # err <= 4e-9). Every element with |ref| > 1e-6 must be strictly within 1 ulp; the
            # absolute floor may be used by at most 0.1 % of elements.
            strict = err <= 1.05 * bf16_ulp(ref)
            floor = err <= 1e-7
            n_floor = int((~strict).sum())
            print(f"    K={K:<4} swz={int(swz)}  max-abs {err.max().item():.3e}  max dist {ratio:.3f} bf16-ulp"
                  f"  ({n_floor} elems on noise floor)")
            assert (strict | floor).all(), f"K={K} swz={swz}: element exceeds 1-ulp class AND the 1e-7 floor"
            assert n_floor <= max(1, err.numel() // 1000), f"K={K} swz={swz}: {n_floor} elements need the floor (>0.1%)"
    print(f"    worst: {worst:.3f} bf16-ulp  (<= 1-ulp class incl. fp32 accumulation-order noise)")


# --------------------------------------------------------------------------- (b)

def test_tier_dispatch():
    print("(b) tier dispatch")
    trellis, suh, svh, Kf, swz = make_linear(K = 4)
    lin = LinearEXL3(None, 256, 512, suh = suh, svh = svh, trellis = trellis,
                     mul1 = torch.tensor(MUL1_MULT, dtype = torch.uint32).view(torch.int), key = "test")
    lin.cpu_repack()
    params = {}

    calls = {"single": 0, "batch": 0}
    orig_single = ext.exl3_cpu_linear_forward
    orig_batch = ext.exl3_cpu_linear_forward_batch
    ext.exl3_cpu_linear_forward = lambda *a, **k: (calls.__setitem__("single", calls["single"] + 1),
                                                   orig_single(*a, **k))[1]
    ext.exl3_cpu_linear_forward_batch = lambda *a, **k: (calls.__setitem__("batch", calls["batch"] + 1),
                                                         orig_batch(*a, **k))[1]
    try:
        # tier off (default): M >= min still takes the stock path
        os.environ.pop("EXL3_CPU_PREFILL_TIER", None)
        x64 = torch.randn(64, 256, generator = G).half()
        y = lin.forward(x64, params)
        assert calls["single"] == 1 and calls["batch"] == 0, "tier off must hit the stock path"
        y_stock = y.clone()

        # tier on, M < min: stock path
        os.environ["EXL3_CPU_PREFILL_TIER"] = "torch"
        os.environ["EXL3_CPU_PREFILL_MIN_M"] = "32"
        x8 = torch.randn(8, 256, generator = G).half()
        ref8 = orig_single(lin.trellis, suh, svh, Kf, None, x8, False, lin.cpu_swz)
        y8 = lin.forward(x8, params)
        assert calls["single"] == 2, "M < min must hit the stock path"
        assert torch.equal(y8, ref8.view_as(y8)), "M < min must be byte-identical to the stock binding"

        # tier on, M >= min: new path, stock binding untouched
        y64 = lin.forward(x64, params)
        assert calls["single"] == 2, "M >= min must NOT hit the stock binding"
        d = (y64.float() - y_stock.float()).abs().max().item()
        print(f"    M=64 tier-vs-stock max-abs {d:.5f} (bf16-rounding class)")
        assert y64.dtype == torch.half

        # batched decode path defers while the tier engages
        calls["batch"] = 0
        lin2 = LinearEXL3(None, 256, 512, suh = suh, svh = svh, trellis = trellis,
                          mul1 = torch.tensor(MUL1_MULT, dtype = torch.uint32).view(torch.int), key = "test2")
        ys = exl3_mod.cpu_linear_forward_batch([lin2, lin2], x64, {})
        assert calls["batch"] == 0, "tier active: batched decode path must defer to per-linear forwards"
        assert torch.equal(ys[0], ys[1]) and ys[0].shape == (64, 512)
    finally:
        ext.exl3_cpu_linear_forward = orig_single
        ext.exl3_cpu_linear_forward_batch = orig_batch
        os.environ.pop("EXL3_CPU_PREFILL_TIER", None)
        os.environ.pop("EXL3_CPU_PREFILL_MIN_M", None)


# --------------------------------------------------------------------------- (c)

def test_old_vs_new():
    print("(c) old (mul1 GEMV, int8-x quant) and new (bf16 dequant + matmul) vs fp64 truth, M=64/128")
    for m in (64, 128):
        trellis, suh, svh, Kf, swz = make_linear(K = 4, seed = 100 + m)
        lin = LinearEXL3(None, 256, 512, suh = suh, svh = svh, trellis = trellis,
                         mul1 = torch.tensor(MUL1_MULT, dtype = torch.uint32).view(torch.int), key = f"m{m}")
        lin.cpu_repack()
        os.environ["EXL3_CPU_PREFILL_TIER"] = "torch"
        try:
            # well-conditioned activations (post-RMS-norm-like): keeps the old path's int8
            # quantization noise small so the comparison isolates rounding-class differences
            x = (0.5 + 0.1 * torch.randn(m, 256, generator = G)).half()
            y_old = ext.exl3_cpu_linear_forward(lin.trellis, suh, svh, Kf, None, x, True, lin.cpu_swz)
            y_new = cpu_prefill_torch_forward(lin, x, torch.float)
            # fp64 ground truth: integer-exact decode + fp64 butterflies; both paths quantize
            # the SAME half-precision activations, so the reference isolates weight/quant class
            w64 = dequant_weight_fp32(trellis, suh, svh, grp = 8 if swz else 0, dtype = torch.float64)
            y_ref = (x.double() @ w64).float()
            scale = y_ref.abs().max().item()
            rel_old = (y_old - y_ref).abs().max().item() / scale
            rel_new = (y_new - y_ref).abs().max().item() / scale
            err = (y_new - y_old).abs()
            rms = (err.pow(2).mean().sqrt() / y_ref.pow(2).mean().sqrt()).item()
            flips = (y_old.argmax(-1) != y_new.argmax(-1)).sum().item()
            print(f"    M={m:<4} rel_old {rel_old:.5f} (int8 class)  rel_new {rel_new:.5f} (bf16 class)"
                  f"  rms-ratio {rms:.5f}  argmax flips {flips}/{m}")
            # classes: old = int8 activation quant (step ~ group-max/127, errors can accumulate
            # coherently -> measured ~1.8e-2 worst-case here; sanity bound 2.5e-2 guards the
            # oracle, not the kernel); new = bf16 weights AND activations (2^-8) with fp32
            # accumulation -> gated at 1e-2, and never farther from the truth than the old path
            assert rel_old <= 2.5e-2, f"M={m}: old path {rel_old:.5f} out of int8 class (oracle broken?)"
            assert rel_new <= 1e-2, f"M={m}: new path {rel_new:.5f} exceeds bf16 class"
            assert rel_new <= rel_old + 2e-3, f"M={m}: new path farther from truth than old"
            assert flips <= max(1, m // 64), f"M={m}: {flips} argmax flips (not argmax-stable class)"
        finally:
            os.environ.pop("EXL3_CPU_PREFILL_TIER", None)


def main():
    test_dequant_parity()
    test_tier_dispatch()
    test_old_vs_new()
    print("PASS: tests/test_cpu_bf16_prefill.py")


if __name__ == "__main__":
    main()
