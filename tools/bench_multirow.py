#!/usr/bin/env python3
"""bench_multirow.py — plan 03 go/no-go microbench: multi-row GEMM tiles vs the M=16 shapes.

Times ext.exl3_gemm with a FORCED shape index on real 2.05bpw tensors of
qwen3.8-flash-next-exl3 (layers.3.mlp.shared_expert.{gate,down}_proj + the indexer
qk projection), sweeping M across the direct-GEMM band [16, 144]. Forcing skips the
GEMV and autotune paths entirely (force_shape_idx > 0), so every shape can be timed
from ONE process without cross-contamination of the autotune cache.

Run from the worktree root with the worktree ext FIRST on PYTHONPATH (see
RESUME.md / tools/jit_build_wt.py):
  PYTHONPATH=$TORCH_EXTENSIONS_DIR/exllamav3_ext:$PWD python tools/bench_multirow.py
"""
import argparse
import os
import sys
import time

import torch
from safetensors import safe_open

MDIR = os.environ.get(
    "BENCH_MDIR",
    "/home/daniel/development/gunzi/local-llm/models/qwen3.8-flash-next-exl3/2.05bpw",
)
SHARD = "model-00001-of-00005.safetensors"
PFX = "model.language_model.layers.3"
TENSORS = {
    # name -> (key, expected N or None); K and dims come from the checkpoint (the model
    # mixes bitrates per tensor family: shared_expert is 4.05bpw-class, routed 2.05)
    "sh_gate": (f"{PFX}.mlp.shared_expert.gate_proj", None),
    "sh_down": (f"{PFX}.mlp.shared_expert.down_proj", None),
    "idx_qk": (f"{PFX}.self_attn.indexer.index_qk_proj", None),
}
M_SWEEP = [16, 17, 24, 32, 33, 48, 64, 100, 128, 144]
ITERS = 50
WARMUP = 10


def load_bc(prefix, K_known, N_known):
    from exllamav3.ext import exllamav3_ext as ext
    with safe_open(f"{MDIR}/{SHARD}", framework="pt") as f:
        trellis = f.get_tensor(f"{prefix}.trellis").cuda()
        suh = f.get_tensor(f"{prefix}.suh").cuda().half()
        svh = f.get_tensor(f"{prefix}.svh").cuda().half()
        mul1 = f.get_tensor(f"{prefix}.mul1").cuda()
    K = int(trellis.shape[-1] // 16)
    in_f, out_f = trellis.shape[0] * 16, trellis.shape[1] * 16
    if N_known is not None:
        assert out_f == N_known, (out_f, N_known)
    x0 = torch.zeros(1, in_f, dtype=torch.half, device="cuda")
    bc = ext.BC_LinearEXL3(trellis, suh, svh, K, None, False, bool(mul1.item()), x0)
    return ext, bc, trellis, suh, svh, bool(mul1.item()), in_f, out_f, K


def time_forced(ext, trellis, suh, svh, mul1, K, in_f, out_f, m, shape):
    A = torch.randn(m, in_f, dtype=torch.half, device="cuda") * 0.5
    C = torch.empty(m, out_f, dtype=torch.half, device="cuda")
    xh = torch.empty_like(A)
    mcg = False
    for _ in range(WARMUP):
        ext.exl3_gemm(A, trellis, C, suh, xh, svh, shape, mcg, mul1, 0)
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(ITERS):
        start.record()
        ext.exl3_gemm(A, trellis, C, suh, xh, svh, shape, mcg, mul1, 0)
        stop.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1,2,3,4,5,6,7", help="shape indices to time")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    shapes = [int(s) for s in args.shapes.split(",")]
    torch.set_grad_enabled(False)

    from exllamav3.ext import exllamav3_ext as ext
    print(f"# ext: {ext.__file__}", flush=True)
    n_shapes = ext.exl3_gemm_num_kernel_shapes()
    print(f"# num shapes: {n_shapes}", flush=True)

    rows = []
    for name, (key, Nk) in TENSORS.items():
        ext_, bc, trellis, suh, svh, mul1, in_f, out_f, K = load_bc(key, None, Nk)
        print(f"\n== {name}: K={K} in={in_f} out={out_f} mul1={mul1} "
              f"(trellis {tuple(trellis.shape)})", flush=True)
        hdr = f"{'M':>5} |" + "".join(f"   shape{s:>2} " for s in shapes) + "|  best old -> best new   speedup"
        print(hdr, flush=True)
        for m in M_SWEEP:
            times = {}
            for s in shapes:
                if s > n_shapes:
                    continue
                if not ext_.exl3_gemm_shape_compat(s, m, in_f, out_f, K, False):
                    continue
                try:
                    times[s] = time_forced(ext_, trellis, suh, svh, mul1, K, in_f, out_f, m, s)
                except RuntimeError as e:
                    print(f"  shape {s} m {m}: {e}", file=sys.stderr)
                    times[s] = float("nan")
            old = {s: t for s, t in times.items() if s <= 4 and t == t}
            new = {s: t for s, t in times.items() if s >= 5 and t == t}
            bo = min(old.values()) if old else float("nan")
            bn = min(new.values()) if new else float("nan")
            cells = "".join(f" {times.get(s, float('nan')):9.3f}" for s in shapes)
            sp = bo / bn if bn == bn and bo == bo and bn > 0 else float("nan")
            print(f"{m:>5} |{cells} | {bo:9.3f} -> {bn:9.3f}   {sp:6.2f}x", flush=True)
            rows.append((name, m, times, bo, bn, sp))

    if args.out:
        with open(args.out, "w") as f:
            f.write("tensor,m,shape,ms\n")
            for name, m, times, bo, bn, sp in rows:
                for s, t in times.items():
                    f.write(f"{name},{m},{s},{t:.6f}\n")
        print(f"\n# wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
