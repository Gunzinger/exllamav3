"""Plan 07 micro bench: stock mul1 batch-M GEMV vs bf16-dequant+matmul tier (Option 1).

Model-free; run from the worktree root with the CPU venv:
  TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD cpu-trial/.venv/bin/python \
      tests/micro_cpu_bf16.py
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import torch
from test_cpu_bf16_prefill import make_linear, MUL1_MULT
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3, cpu_prefill_torch_forward
from exllamav3.modules.quant import exl3 as exl3_mod

G = torch.Generator().manual_seed(0x7)
SHAPES = [(4096, 4096, 2.5), (4096, 8192, 2.0)]   # (k_in, n_out, K): proj-class and wide-class


def timed(fn, reps=3):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    torch.set_grad_enabled(False)
    print(f"torch threads: {torch.get_num_threads()}")
    for k, n, K in SHAPES:
        trellis, suh, svh, Kf, swz = make_linear(k=k, n=n, K=K, seed=11)
        lin = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis,
                         mul1=torch.tensor(MUL1_MULT, dtype=torch.uint32).view(torch.int), key=f"m{k}x{n}")
        lin.cpu_repack()
        tre_bytes = trellis.numel() * 2
        w_bytes = k * n * 2
        print(f"\nshape k={k} n={n} K={Kf}  trellis {tre_bytes/2**20:.0f} MiB  bf16 W {w_bytes/2**20:.0f} MiB")
        print(f"{'M':>5} {'stock ms':>9} {'tier-cold ms':>12} {'tier-warm ms':>12} "
              f"{'speedup warm':>12} {'warm GF/s':>9} {'warm GB/s':>9}")
        for m in (128, 512, 2048):
            x = (0.5 + 0.1 * torch.randn(m, k, generator=G)).half()
            t_stock, y0 = timed(lambda: ext.exl3_cpu_linear_forward(
                lin.trellis, suh, svh, Kf, None, x, True, lin.cpu_swz))
            os.environ["EXL3_CPU_PREFILL_TIER"] = "torch"
            os.environ["EXL3_CPU_PREFILL_CACHE_MIB"] = "0"
            exl3_mod._cpu_bf16_cache = None
            t_cold, _ = timed(lambda: cpu_prefill_torch_forward(lin, x, torch.float))
            os.environ["EXL3_CPU_PREFILL_CACHE_MIB"] = "65536"
            exl3_mod._cpu_bf16_cache = None
            cpu_prefill_torch_forward(lin, x, torch.float)      # materialize + cache
            t_warm, y1 = timed(lambda: cpu_prefill_torch_forward(lin, x, torch.float))
            os.environ.pop("EXL3_CPU_PREFILL_TIER", None)
            os.environ.pop("EXL3_CPU_PREFILL_CACHE_MIB", None)
            gf = 2 * m * k * n / t_warm / 1e9
            gbs = (m * k * 2 + w_bytes + m * n * 4) / t_warm / 2**30
            print(f"{m:>5} {t_stock*1e3:>9.1f} {t_cold*1e3:>12.1f} {t_warm*1e3:>12.1f} "
                  f"{t_stock/t_warm:>11.2f}x {gf:>9.1f} {gbs:>9.1f}")


if __name__ == "__main__":
    main()
