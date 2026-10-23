# Unit tests for the reconstruct-once FP16 shadow cache (branch opt/recon-cache, plan 04).
#
# No model load, no GPU kernels: real LinearEXL3 instances are built from small synthetic packed
# tensors on CPU, and the extension entry points reconstruct_hgemm uses (reconstruct /
# reconstruct_had_slice / hgemm_recon / had_r_128) are monkeypatched with deterministic
# torch equivalents (originals restored after each test). This validates the LRU manager, the
# basis handling, the decode gate and the exclusion rules; dequant parity of the REAL kernels is
# covered by the deferred GPU parity gate (see RESUME.md).
#
# Run:  PYTHONPATH=<worktree> python tests/test_recon_cache.py

import os
import sys

import torch

os.environ.setdefault("EXL3_RECON_CACHE_MIB", "4")  # default-on for the process; tests set budgets explicitly

from exllamav3.modules.quant import exl3 as exl3_mod
from exllamav3.modules.quant import recon_cache as rc_mod
from exllamav3.util.cuda_flags import recon_cache_budget_bytes, RECON_CACHE_MIB_ENV

rc = rc_mod.recon_cache


def det_fill(shape, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
    return torch.randn(shape, generator = g).half()


def make_lin(in_f, out_f, key = None, seed = 1, K = 4):
    trellis = torch.randint(-32768, 32768, (2, 16, 16 * K), dtype = torch.int16)
    return exl3_mod.LinearEXL3(
        None, in_f, out_f,
        suh = torch.ones(in_f, dtype = torch.half),
        svh = torch.ones(out_f, dtype = torch.half),
        trellis = trellis,
        key = key,
    ), trellis


class _MockExt:
    """Deterministic stand-ins for the extension kernels reconstruct_hgemm calls."""

    def __init__(self):
        self.counts = {"recon": 0, "had_slice": 0, "hgemm": 0, "had": 0}

    def install(self):
        self._saved = {n: getattr(exl3_mod.ext, n) for n in
                       ("reconstruct", "reconstruct_slice", "reconstruct_had_slice", "hgemm_recon", "had_r_128")}
        exl3_mod.ext.reconstruct = self.reconstruct
        exl3_mod.ext.reconstruct_slice = self.reconstruct_slice
        exl3_mod.ext.reconstruct_had_slice = self.reconstruct_had_slice
        exl3_mod.ext.hgemm_recon = self.hgemm_recon
        exl3_mod.ext.had_r_128 = self.had_r_128

    def restore(self):
        for n, fn in self._saved.items():
            setattr(exl3_mod.ext, n, fn)

    def reconstruct(self, w, trellis, K, mcg, mul1):
        self.counts["recon"] += 1
        w.copy_(det_fill(w.shape, int(trellis.sum().item()) + int(K)))

    def reconstruct_slice(self, w, trellis, K, mcg, mul1, n_start):
        self.counts["recon"] += 1
        w.copy_(det_fill(w.shape, int(trellis.sum().item()) + int(K) + int(n_start)))

    def reconstruct_had_slice(self, w, trellis, suh, svh, K, mcg, mul1, n_start):
        self.counts["had_slice"] += 1
        w.copy_(det_fill(w.shape, int(trellis.sum().item()) + 1000 + int(n_start)))

    def hgemm_recon(self, xh, w, y_):
        self.counts["hgemm"] += 1
        y_.copy_(torch.matmul(xh.float(), w.float()).to(y_.dtype))

    def had_r_128(self, x, out, suh, svh, scale):
        self.counts["had"] += 0  # identity is deterministic; x and out may alias
        if out.data_ptr() != x.data_ptr():
            out.copy_(x)


mock = _MockExt()


def fresh_cache(budget_bytes):
    rc.clear()
    rc.set_budget_bytes(budget_bytes)
    mock.counts = dict.fromkeys(mock.counts, 0)


# --- tests ------------------------------------------------------------------------------------------------


def test_env_knob_parsing():
    old = os.environ.get(RECON_CACHE_MIB_ENV)
    try:
        os.environ[RECON_CACHE_MIB_ENV] = "0"
        assert recon_cache_budget_bytes() == 0
        os.environ[RECON_CACHE_MIB_ENV] = "2048"
        assert recon_cache_budget_bytes() == 2048 << 20
        os.environ[RECON_CACHE_MIB_ENV] = "garbage"
        assert recon_cache_budget_bytes() == 0
        os.environ[RECON_CACHE_MIB_ENV] = "-5"
        assert recon_cache_budget_bytes() == 0
    finally:
        if old is None:
            os.environ.pop(RECON_CACHE_MIB_ENV, None)
        else:
            os.environ[RECON_CACHE_MIB_ENV] = old
    print("PASS env knob parsing (default 0 = off)")


def test_plain_parity_reuse_and_off_parity():
    # Same packed contents twice: cache-on instance (reuse on second call, byte-identical y) and
    # a cache-off instance (budget 0) must produce bitwise-equal outputs.
    lin_on, trellis = make_lin(256, 256, key = "model.blk0.q_a", seed = 7)
    lin_off, _ = make_lin(256, 256, key = "model.blk0.q_a", seed = 7)
    lin_off.trellis = trellis.clone()
    fresh_cache(1 << 20)
    x = torch.randn(512, 256).half()  # rows 512: > 144 (cache tier), < 1024 (plain basis)

    y1 = lin_on.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 1, "first call must materialize exactly once"
    y2 = lin_on.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 1, "second call must reuse the shadow (no reconstruct)"
    assert mock.counts["hgemm"] == 2
    assert torch.equal(y1, y2), "cached second call must be byte-identical"
    assert len(rc) == 1 and rc.total_bytes == 256 * 256 * 2

    rc.set_budget_bytes(0)  # cache off
    y_off = lin_off.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 2
    assert torch.equal(y1, y_off), "cached output must be bitwise-equal to reconstruct-per-call"
    print("PASS plain path: parity vs cache-off + second-call reuse (byte-identical)")


def test_fused_basis_reuse():
    # rows >= 1024 on 128-divisible dims takes the fused-Had path; the shadow stores the
    # original-basis dequant and reuse replays hgemm on it, byte-identically.
    lin, _ = make_lin(256, 256, key = "model.blk0.o", seed = 11)
    fresh_cache(1 << 20)
    assert lin._fused_reconstruct is None
    x = torch.randn(1024, 256).half()
    y1 = lin.reconstruct_hgemm(x, None)
    assert lin._fused_reconstruct is True and mock.counts["had_slice"] == 1
    y2 = lin.reconstruct_hgemm(x, None)
    assert mock.counts["had_slice"] == 1, "fused shadow must be materialized once"
    assert torch.equal(y1, y2)

    # A plain-basis call on the same layer must not consume the fused-basis shadow
    x_small = torch.randn(512, 256).half()
    y3 = lin.reconstruct_hgemm(x_small, None)
    assert mock.counts["recon"] == 1, "plain-basis call materializes its own basis"
    y4 = lin.reconstruct_hgemm(x_small, None)
    assert mock.counts["recon"] == 1 and torch.equal(y3, y4)
    assert len(rc) == 2, "both bases may coexist under the budget"
    print("PASS fused path: reuse + per-basis shadows")


def test_eviction_lru():
    lin_a, _ = make_lin(256, 256, key = "model.blk0.a", seed = 21)
    lin_b, _ = make_lin(256, 256, key = "model.blk0.b", seed = 22)
    shadow_bytes = 256 * 256 * 2
    fresh_cache(int(1.5 * shadow_bytes))  # fits one shadow; inserting B evicts A (LRU)
    x = torch.randn(512, 256).half()

    lin_a.reconstruct_hgemm(x, None)
    lin_b.reconstruct_hgemm(x, None)
    assert rc.total_bytes <= rc.budget_bytes, "budget must be enforced"
    assert len(rc) == 1, "oldest shadow evicted"

    lin_b.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 2, "B (recent) still cached"
    lin_a.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 3, "A was evicted and re-materializes"
    # A is now the most recent; B should be the one evicted next
    lin_a.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 3
    lin_b.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 4
    print("PASS eviction: LRU order under a tiny budget, total never over budget")


def test_decode_gate_never_caches():
    lin, _ = make_lin(256, 256, key = "model.blk0.k", seed = 31)
    fresh_cache(1 << 20)
    x = torch.randn(64, 256).half()  # rows 64 <= 144: decode tier
    lin.reconstruct_hgemm(x, None)
    lin.reconstruct_hgemm(x, None)
    assert mock.counts["recon"] == 2, "decode-tier calls must never consult or fill the cache"
    assert len(rc) == 0
    print("PASS decode gate: rows <= 144 never touches the cache")


def test_expert_exclusion_and_slicing():
    assert not rc.key_cacheable("model.layers.0.mlp.experts.3.down")
    assert not rc.key_cacheable("model.layers.0.mlp.experts.127.gate_up")
    assert rc.key_cacheable("model.layers.0.mlp.shared_experts.gate_up")  # dense, in scope
    assert not rc.key_cacheable(None), "unknown keys are conservative"

    lin_x, _ = make_lin(256, 256, key = "model.layers.0.mlp.experts.3.down", seed = 41)
    fresh_cache(1 << 20)
    x = torch.randn(512, 256).half()
    lin_x.reconstruct_hgemm(x, None)
    assert len(rc) == 0, "routed-expert tensors are excluded"

    # lm_head-class shape (N > MAX_RECONSTRUCT_SLICE_N): sliced loop keeps per-call reconstruct,
    # never cached (small enough here to stay under the tensor budget for a unit test)
    lin_head, _ = make_lin(128, 32896, key = "model.lm_head", seed = 42)
    x_head = torch.randn(512, 128).half()
    lin_head.reconstruct_hgemm(x_head, None)
    lin_head.reconstruct_hgemm(x_head, None)
    assert len(rc) == 0, "sliced-N path is not cached"
    print("PASS exclusions: routed experts + sliced-N (lm_head-class) never cached")


def test_unload_and_clear_drop():
    lin, _ = make_lin(256, 256, key = "model.blk0.v", seed = 51)
    fresh_cache(1 << 20)
    x = torch.randn(512, 256).half()
    lin.reconstruct_hgemm(x, None)
    assert len(rc) == 1
    lin.unload()
    assert len(rc) == 0, "LinearEXL3.unload drops the layer's shadows"
    lin.reconstruct_hgemm(x, None)
    assert len(rc) == 1
    rc.clear()
    assert len(rc) == 0 and rc.total_bytes == 0, "model-unload clear() drops everything"
    print("PASS invalidation: per-layer unload + global clear")


def test_budget_shrink():
    lin_a, _ = make_lin(256, 256, key = "model.blk0.a", seed = 61)
    lin_b, _ = make_lin(256, 256, key = "model.blk0.b", seed = 62)
    fresh_cache(4 << 20)
    x = torch.randn(512, 256).half()
    lin_a.reconstruct_hgemm(x, None)
    lin_b.reconstruct_hgemm(x, None)
    assert len(rc) == 2
    rc.set_budget_bytes(256 * 256 * 2)  # shrink to one shadow
    assert rc.total_bytes <= rc.budget_bytes
    print("PASS budget shrink re-evicts to fit")


if __name__ == "__main__":
    mock.install()
    try:
        test_env_knob_parsing()
        test_plain_parity_reuse_and_off_parity()
        test_fused_basis_reuse()
        test_eviction_lru()
        test_decode_gate_never_caches()
        test_expert_exclusion_and_slicing()
        test_unload_and_clear_drop()
        test_budget_shrink()
    finally:
        mock.restore()
        rc.clear()
        rc.set_budget_bytes(0)
    print("ALL RECON-CACHE UNIT TESTS PASSED")
