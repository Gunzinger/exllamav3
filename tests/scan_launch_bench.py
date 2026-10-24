"""opt06 scan-level launch microbench: kernels per vendored fla chunk-scan call
at B=1 vs B=4 (T=2048, H=16, K=V=128 — qwen GDN geometry).

The serving generator never issues multi-row GDN scans (job.prefill runs one
sequence per forward, tests/launch_gdn.py evidence), so this isolates what the
EXL3_GDN_SCAN_BATCHED patch would save IF a multi-row caller existed: the old
per-row loop launches the scan-kernel group once per row (B x), the batched pass
folds the rows into the kernel grid (1 x). Main-thread CUDA work only — the
torch profiler sees device events fine here (unlike Generator background
threads). Counts: chrome-trace cat=="kernel" events.

  PYTHONPATH=$PWD .venv-python tests/scan_launch_bench.py \
      --out /tmp/opencode/scan_launch_bench.json
"""
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _install_triton_counter(counter):
    """Count triton kernel launches by name (this kineto build does not trace
    triton's launch API — see tests/launch_gdn.py)."""
    from triton.runtime.jit import JITFunction

    orig = JITFunction.run

    def wrapped(self, *args, **kwargs):
        counter[self.fn.__name__] += 1
        return orig(self, *args, **kwargs)

    JITFunction.run = wrapped
    return orig


def count_kernels(prof):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json", dir="/tmp/opencode")
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    os.unlink(path)
    kernels = Counter()
    for e in trace.get("traceEvents", []):
        if e.get("cat") in ("kernel", "gpu_kernel", "Kernel"):
            kernels[e["name"]] += 1
    return kernels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--HV", type=int, default=48)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only-B", type=int, default=None,
                   help="run ONE combo (this B, this arm) — for external profilers")
    p.add_argument("--only-arm", choices=["old", "new"], default=None)
    p.add_argument("--out", default="/tmp/opencode/scan_launch_bench.json")
    args = p.parse_args()

    import torch
    from torch.profiler import profile, ProfilerActivity
    import exllamav3.modules.gated_delta_net_fn.gated_delta_rule as gdr
    from exllamav3.vendor.fla import chunk_gated_delta_rule

    dev = torch.device("cuda:0")
    results = {}
    combos = [(1, "old"), (1, "new"), (4, "old"), (4, "new")]
    if args.only_B:
        combos = [(args.only_B, args.only_arm)]
    for B, arm in combos:
        batched = arm == "new"
        q = torch.randn(B, args.T, args.H, args.K, device=dev, dtype=torch.half)
        k = torch.randn(B, args.T, args.H, args.K, device=dev, dtype=torch.half)
        v = torch.randn(B, args.T, args.HV, args.V, device=dev, dtype=torch.half)
        g = torch.randn(B, args.T, args.HV, device=dev, dtype=torch.float)
        beta = torch.rand(B, args.T, args.HV, device=dev, dtype=torch.half)
        pool = torch.zeros(2 * B, 1, args.HV, args.K, args.V, device=dev,
                           dtype=torch.float16)

        def one_pass():
            rows = list(range(B))
            gdr._chunked_scan_rows(chunk_gated_delta_rule, q, k, v, g, beta,
                                   pool, rows, rows, False, True)

        gdr._scan_batched = batched
        for _ in range(2):  # warmup + autotune
            one_pass()
        torch.cuda.synchronize()

        tri_counter = Counter()
        _install_triton_counter(tri_counter)

        tri_totals = []
        tri_names = Counter()
        for _ in range(args.reps):
            tri_counter.clear()
            one_pass()
            torch.cuda.synchronize()
            tri_totals.append(sum(tri_counter.values()))
            tri_names.update(tri_counter)
        results[f"B{B}_{arm}"] = {
            "batched": batched, "triton_per_pass": tri_totals,
            "triton_names": dict(tri_names),
        }
        print(f"[scan] B={B} arm={arm}: triton kernels/pass={tri_totals} "
              f"(distinct={len(tri_names)})", flush=True)

    if args.only_B:
        print(f"[single] B={args.only_B} arm={args.only_arm} done")
        return

    for B in (1, 4):
        o = results[f"B{B}_old"]["triton_per_pass"]
        n = results[f"B{B}_new"]["triton_per_pass"]
        if not o or not n:
            continue
        print(f"[summary] B={B}: old={o} new={n} "
              f"delta={100.0 * (n[0] - o[0]) / o[0]:+.1f}%")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"T": args.T, "H": args.H, "HV": args.HV, "K": args.K,
                   "V": args.V, "detail": results}, f, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
