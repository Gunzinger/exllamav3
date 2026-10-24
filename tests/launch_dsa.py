"""opt05 launch-count evidence: kernel launches of ONE 2048-token dense prefill chunk
(max_new_tokens=1 -> exactly one prefill forward, no decode step), arms
  base   EXL3_DSA_LAZY_PLANES=0 EXL3_DSA_BATCH_UPPROJ=0
  lazy   EXL3_DSA_LAZY_PLANES=1 (part A: k-pool upkeep chain deferred)
  upproj EXL3_DSA_LAZY_PLANES=1 EXL3_DSA_BATCH_UPPROJ=1 (part B step 1: one GEMM)

Counting technique (opt06's, adapted): torch.profiler chrome trace counts
cat=="kernel" events (torch ops + extension kernels); triton launches are counted by
wrapping triton.runtime.jit.JITFunction.run (this kineto build does not trace the
hipLaunchKernel*ExC path). The arm knobs are IMPORT-TIME constants but the module
globals are read at call time, so all three arms run in ONE process against one
loaded model with `mla_attn._dsa_lazy_planes` / `mla_attn._dsa_batch_upproj` flipped
between passes (counting only — parity arms use the real env knobs in separate
processes). Each pass uses a distinct salted prompt so the in-process prompt cache
never serves a repeat; a warmup pass per arm settles triton autotune and (upproj)
the lazily built _w_kv_b_cat before counting.

  PYTHONPATH=$PWD .venv-python tests/launch_dsa.py \
      --model-dir models/GLM-5.3-Flash-EXL3-Yamz/mix --out /tmp/opencode/launch_dsa.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ids(tokenizer, target_len, salt):
    """One exact-length prompt, distinct per salt (busts the in-process prompt cache)."""
    base = ("The harbor authority published a revised schedule for the winter season, "
            "and every ferry operator had to refile paperwork before the first storm "
            "reached the coast. " * 4)
    ids = tokenizer.encode(base, add_bos = True).flatten().tolist()
    pad = tokenizer.encode(" Cargo manifests, inspection stamps, and berth assignments "
                           "arrived in a slow stream of faxes and phone calls.",
                           add_bos = False).flatten().tolist()
    k = salt * 2654435761 % (1 << 31)
    while len(ids) < target_len:
        k = (k * 1103515245 + 12345) & 0x7FFFFFFF
        ids.append(pad[k % len(pad)])
    import torch
    return torch.tensor([ids[:target_len]], dtype = torch.long)


def _run_job(gen, tokenizer, ids):
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature = 0.0, top_p = 1.0, top_k = 0)
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = sampler,
              stop_conditions = [tokenizer.eos_token_id], identifier = 0)
    gen.enqueue(job)
    n = 0
    for result in gen.iterate():
        if result.get("identifier") is None:
            continue
        n += 1
    return n


def _install_triton_counter(counter):
    """Count triton kernel launches by name (kineto misses triton's launch path)."""
    from triton.runtime.jit import JITFunction
    orig = JITFunction.run

    def wrapped(self, *args, **kwargs):
        counter[self.fn.__name__] += 1
        return orig(self, *args, **kwargs)

    JITFunction.run = wrapped
    return orig


def _kineto_kernels(prof, tmpdir):
    import tempfile
    fd, path = tempfile.mkstemp(suffix = ".json", dir = tmpdir)
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    os.unlink(path)
    kernels, dur = Counter(), Counter()
    for e in trace.get("traceEvents", []):
        if e.get("cat") in ("kernel", "gpu_kernel", "Kernel"):
            kernels[e["name"]] += 1
            dur[e["name"]] += e.get("dur", 0)
    return kernels, dur


ARMS = (("base", False, False), ("lazy", True, False), ("upproj", True, True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required = True)
    p.add_argument("--prompt-tokens", type = int, default = 2048)
    p.add_argument("--out", default = "/tmp/opencode/launch_dsa.json")
    args = p.parse_args()

    import torch
    import exllamav3.modules.mla_attn as mla
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator

    ctx = (args.prompt_tokens + 511) // 256 * 256 + 256
    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = ctx, max_batch_size = 1, max_history = 0)
    gen = Generator(model, cache, tokenizer)   # default max_chunk_size = 2048
    model.load(torch.device("cuda:0"))
    print("[load] done", flush = True)

    triton_counter = Counter()
    _install_triton_counter(triton_counter)
    from torch.profiler import profile, ProfilerActivity

    results = {}
    for i, (name, lazy, up) in enumerate(ARMS):
        mla._dsa_lazy_planes = lazy
        mla._dsa_batch_upproj = up
        # warmup: autotune + (upproj) _w_kv_b_cat build must not pollute the count
        _run_job(gen, tokenizer, _ids(tokenizer, args.prompt_tokens, 1000 + i))
        torch.cuda.synchronize()
        triton_counter.clear()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities = [ProfilerActivity.CUDA]) as prof:
            _run_job(gen, tokenizer, _ids(tokenizer, args.prompt_tokens, i))
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        kernels, dur = _kineto_kernels(prof, os.path.dirname(args.out) or ".")
        tri = dict(triton_counter)
        total = sum(kernels.values()) + sum(tri.values())
        results[name] = {
            "lazy": lazy, "batch_upproj": up,
            "total_launches": total,
            "kineto_kernels": sum(kernels.values()),
            "triton_launches": sum(tri.values()),
            "triton_names": tri,
            "kernel_names": dict(kernels),
            "kernel_us_by_name": dict(dur),
            "wall_s": round(wall, 4),
        }
        print(f"[count] arm={name}: total={total} (kineto={sum(kernels.values())} "
              f"triton={sum(tri.values())}) wall={wall * 1e3:.1f} ms", flush = True)

    mla._dsa_lazy_planes = True
    mla._dsa_batch_upproj = False

    summary = {}
    b, l, u = (results[n]["total_launches"] for n, _, _ in ARMS)
    summary["lazy_vs_base"] = {"base": b, "lazy": l, "delta": l - b,
                               "delta_pct": round(100.0 * (l - b) / b, 2)}
    summary["upproj_vs_lazy"] = {"lazy": l, "upproj": u, "delta": u - l,
                                 "delta_pct": round(100.0 * (u - l) / l, 2)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok = True)
    with open(args.out, "w") as f:
        json.dump({"prompt_tokens": args.prompt_tokens, "summary": summary,
                   "detail": results}, f, indent = 1)
    print(f"[summary] {json.dumps(summary)}")
    print(f"[out] {args.out}")
    model.unload()


if __name__ == "__main__":
    main()
