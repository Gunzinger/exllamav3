"""Plan 07 Option 2 (dequant tier) unit tests: AVX-512 dequant-to-BF16 twin (model-free).

exl3_cpu_dequant_bf16_avx512 (cpu/recon_bf16_avx512.cpp) must produce output EQUAL to the
landed scalar binding exl3_cpu_dequant_bf16: the state decode is integer-exact and the
butterflies replicate the scalar radix-2 stage order, so the expected result is BIT-IDENTICAL
(the 1-bf16-ulp allowance is only a fallback class, reported if ever hit).

Covers:
  (a) parity vs the scalar binding across K in {1, 2, 4, 8, 2.5, 3.5} x {native, band-8}
      (fixtures reused from tests/test_cpu_bf16_prefill.py), plus one wider multi-block case
      (k=512, n=256, K=2.5, band-8) exercising more than two k-blocks per row strip.
  (b) the scalar fallback path still works (it is the parity reference; also directly
      re-checked against the pure-torch fp32 oracle here).
  (c) timing: cold dequant MB/s (output bf16 bytes / wall time), scalar vs AVX-512, on
      4096x4096 K=2.5 -- the acceptance case (>= 8x the scalar, i.e. >= ~400 MB/s).

Usage (own extension cache so concurrent agents' builds never race):
  cd <worktree root>
  MAX_JOBS=8 TORCH_EXTENSIONS_DIR=$PWD/.torch_ext_avx PYTHONPATH=$PWD \
    /home/daniel/development/gunzi/local-llm/cpu-trial/.venv/bin/python tests/test_recon_bf16_avx512.py

RESUME.md (status log; append the next step if interrupted):
  [2026-10-24] DONE — kernel landed (bb87f5bf + c749113b): bit-identical to the scalar binding
  at every K x layout (13 cases incl. a 512x256 multi-block band-8 arm); cold 4096x4096 K=2.5
  ~4.2 GB/s vs ~90 MB/s scalar (48x; acceptance was 8x / 400 MB/s). Not yet done: python-side
  tier dispatch does not call the avx512 binding yet (the other agent owns exl3.py); the
  fp32_out variant of the scalar binding has no avx512 twin (by design — bf16-out kernel).
  Debugging lore if this ever breaks: (1) SWAR byte-sum needs (s16 & 0xFFFF) + (s16 >> 16),
  the unmasked add silently rides the high pair-sum at 2^16; (2) the 16x16 transpose's stage k
  must pair EVERY m with bit k clear against m ^ 2^k (stepping i by 2^(k+1) skips half the
  pairs); (3) the JIT CPU build has no -fopenmp, so at::parallel_for silently runs serial —
  use the file's rb_parallel_for (std::thread) helper for block parallelism.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from exllamav3.ext import exllamav3_ext as ext
from test_cpu_bf16_prefill import make_linear, dequant_weight_fp32, bf16_ulp


def run_pair(K, swizzled, k = 256, n = 512, seed = None, ref = False):
    """Both bindings on the same fixture; env handled exactly like test_cpu_bf16_prefill."""
    os.environ["EXL3_MOE_CPU_SWIZZLE"] = "1" if swizzled else "0"
    try:
        s = seed if seed is not None else int(K * 10) + swizzled
        trellis, suh, svh, Kf, swz = make_linear(k = k, n = n, K = K, swizzled = swizzled, seed = s)
        w_s = ext.exl3_cpu_dequant_bf16(trellis, suh, svh, Kf, swz)
        w_v = ext.exl3_cpu_dequant_bf16_avx512(trellis, suh, svh, Kf, swz)
        ref_w = dequant_weight_fp32(trellis, suh, svh, grp = 8 if swz else 0).t() if ref else None
    finally:
        os.environ.pop("EXL3_MOE_CPU_SWIZZLE", None)
    return w_s, w_v, ref_w


def check_parity(K, swizzled, k = 256, n = 512, seed = None, ref = False):
    w_s, w_v, ref_w = run_pair(K, swizzled, k, n, seed, ref)
    assert w_v.dtype == torch.bfloat16 and w_v.shape == (n, k), (w_v.dtype, w_v.shape)
    same = torch.equal(w_s.view(torch.uint16), w_v.view(torch.uint16))
    if same:
        msg = "bit-identical"
    else:
        ndiff = int((w_s.view(torch.uint16) != w_v.view(torch.uint16)).sum())
        err = (w_v.float() - w_s.float()).abs()
        r = ref_w if ref_w is not None else w_s.float()
        dist = (err / bf16_ulp(r)).max().item()
        msg = f"NOT bit-identical: {ndiff}/{w_s.numel()} elems, max dist {dist:.2f} bf16-ulp"
        assert dist <= 1.05, f"K={K} swz={swizzled}: exceeds 1-ulp class ({dist})"
    # scalar path alive (it produced the reference) and oracle agreement when asked
    if ref_w is not None:
        err = (w_s.float() - ref_w).abs()
        ok = (err <= 1.05 * bf16_ulp(ref_w)) | (err <= 1e-7)
        assert ok.all(), f"K={K} swz={swizzled}: scalar binding off the oracle"
    print(f"    K={K:<4} swz={int(swizzled)}  {k}x{n}  {msg}")
    return same


def test_parity():
    print("(a) AVX-512 dequant vs scalar binding")
    all_same = True
    for K in (1, 2, 4, 8, 2.5, 3.5):
        for sw in (False, True):
            all_same &= check_parity(K, sw, ref = (K in (1, 4, 2.5)))
    # wider multi-block strip: 4 k-blocks per row strip, band-8 tile permutation
    all_same &= check_parity(2.5, True, k = 512, n = 256, seed = 99)
    print(f"    -> {'BIT-IDENTICAL everywhere' if all_same else 'within 1-ulp class (see above)'}")


def test_fallback_intact():
    print("(b) scalar fallback path intact")
    trellis, suh, svh, Kf, swz = make_linear(K = 3, seed = 33)
    w = ext.exl3_cpu_dequant_bf16(trellis, suh, svh, Kf, swz)
    assert w.dtype == torch.bfloat16 and w.shape == (512, 256)
    print("    scalar binding OK (K=3 native)")


def bench_one(fn, trellis, suh, svh, Kf, swz, reps):
    fn(trellis, suh, svh, Kf, swz)      # warmup
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(trellis, suh, svh, Kf, swz)
        best = min(best, time.perf_counter() - t0)
    return best


def test_timing():
    print("(c) cold dequant timing, 4096x4096 K=2.5 (MB/s = output bf16 bytes / wall time)")
    n = k = 4096
    for swizzled in (True, False):
        os.environ["EXL3_MOE_CPU_SWIZZLE"] = "1" if swizzled else "0"
        try:
            trellis, suh, svh, Kf, swz = make_linear(k = k, n = n, K = 2.5, swizzled = swizzled, seed = 4096)
        finally:
            os.environ.pop("EXL3_MOE_CPU_SWIZZLE", None)
        mib = n * k * 2 / (1024 * 1024)
        t_s = bench_one(ext.exl3_cpu_dequant_bf16, trellis, suh, svh, Kf, swz, 2)
        t_v = bench_one(ext.exl3_cpu_dequant_bf16_avx512, trellis, suh, svh, Kf, swz, 5)
        mbs_s, mbs_v = mib / t_s, mib / t_v
        layout = "band-8" if swizzled else "native "
        print(f"    {layout}  scalar {mbs_s:8.1f} MB/s ({t_s:.3f} s)   "
              f"avx512 {mbs_v:8.1f} MB/s ({t_v:.3f} s)   speedup {mbs_v / mbs_s:6.1f}x"
              f"   [threads={torch.get_num_threads()}]")
        if swizzled:
            assert mbs_v >= 8.0 * mbs_s, f"acceptance: avx512 {mbs_v:.0f} MB/s < 8x scalar {mbs_s:.0f}"
            assert mbs_v >= 400.0, f"acceptance: {mbs_v:.0f} MB/s < 400 MB/s floor"


def main():
    test_parity()
    test_fallback_intact()
    test_timing()
    print("PASS: tests/test_recon_bf16_avx512.py")


if __name__ == "__main__":
    main()
