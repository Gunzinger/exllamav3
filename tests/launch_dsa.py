"""opt05 launch-count evidence: ONE 2048-token dense prefill (complete job:
prefill chunk + tail forward + sample), arms
  base   EXL3_DSA_LAZY_PLANES=0 EXL3_DSA_BATCH_UPPROJ=0
  lazy   EXL3_DSA_LAZY_PLANES=1 (part A: k-pool upkeep chain deferred)
  upproj EXL3_DSA_LAZY_PLANES=1 EXL3_DSA_BATCH_UPPROJ=1 (part B step 1: one GEMM)

Counting technique: torch-profiler kineto does not deliver device events in model
processes on this stack (cat=="kernel" works in a bare matmul process but is empty
once exllamav3 loads — root cause not pursued; rocprofv3 hangs per plan-06), so the
counters are direct python-level wrappers, all installed identically for every arm:
  - triton launches by name   : triton.runtime.jit.JITFunction.run wrapper (opt06
    technique; kineto does not trace triton's hipLaunchKernel*ExC path here either)
  - up-projection GEMMs       : torch.mm wrapper -> call count + CUDA-event us
  - pool-plane upkeep         : MLAttention._update_pool_plane wrapper -> calls +
    CUDA-event us (its torch-op chain is ~10 launches/layer from source: arange,
    index gather, softmax, mul, sum, to-half, fill_ + the triton plane append)
  - prefill chunk boundaries  : MLAttention._attend wrapper (seqlen > 1), proving
    every pass ran the full chunk and tail forward
The arm knobs are IMPORT-TIME constants but the module globals are read at call
time, so all three arms run in ONE process with `mla_attn._dsa_lazy_planes` /
`mla_attn._dsa_batch_upproj` flipped between passes (counting only — parity arms
use the real env knobs in separate processes). Each pass runs a COMPLETE job
(drain loops gen.iterate() until the job reports eos/tokens — a single iterate()
drain exits mid-job and leaks its tail into the next pass), with distinct salted
prompts so the in-process prompt cache never serves a repeat; a warmup pass per
arm settles triton autotune and (upproj) the lazily built _w_kv_b_cat.

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


class _Counters:
    triton = Counter()
    mm_calls = 0
    mm_us = 0.0
    pool_calls = 0
    pool_us = 0.0
    spans = []


def _install(c):
    from triton.runtime.jit import JITFunction
    orig_run = JITFunction.run

    def wrapped_run(self, *args, **kwargs):
        c.triton[self.fn.__name__] += 1
        return orig_run(self, *args, **kwargs)

    JITFunction.run = wrapped_run

    import torch
    orig_mm = torch.mm

    def wrapped_mm(*args, **kwargs):
        # count only: per-call event timing (elapsed_time on in-flight events)
        # deadlocks on this ROCm stack
        c.mm_calls += 1
        return orig_mm(*args, **kwargs)

    torch.mm = wrapped_mm
    return orig_run, orig_mm


def _install_model_hooks(c):
    import torch
    import exllamav3.modules.mla_attn as mla

    orig_upd = mla.MLAttention._update_pool_plane

    def wrapped_upd(self, *a, **kwargs):
        # count only (async-kernel timing per call deadlocks on this stack; the
        # chain's cost is read from the per-pass wall delta instead)
        c.pool_calls += 1
        return orig_upd(self, *a, **kwargs)

    mla.MLAttention._update_pool_plane = wrapped_upd

    orig_attend = mla.MLAttention._attend

    def wrapped_attend(self, x, bsz, seqlen, params, *a, **kw):
        if seqlen > 1:
            cs = params.get("cache_seqlens")
            h = cs.flatten().tolist() if isinstance(cs, torch.Tensor) else [0]
            c.spans.append((int(h[0]), int(h[0]) + seqlen))
        return orig_attend(self, x, bsz, seqlen, params, *a, **kw)

    mla.MLAttention._attend = wrapped_attend


def _run_job(gen, tokenizer, ids):
    """Enqueue ONE max_new_tokens=1 job and drain until it completes (a single
    iterate() drain exits mid-job: the prefill chunk runs in the first drain but the
    tail forward + sample only land in a later one)."""
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature = 0.0, top_p = 1.0, top_k = 0)
    job = Job(input_ids = ids, max_new_tokens = 1, sampler = sampler,
              stop_conditions = [tokenizer.eos_token_id], identifier = 0)
    gen.enqueue(job)
    got = False
    while not got:
        for result in gen.iterate():
            if result.get("identifier") is None:
                continue
            if result.get("eos") or result.get("token_ids"):
                got = True
    return got


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

    _install_model_hooks(_Counters)
    _install(_Counters)

    results = {}
    for i, (name, lazy, up) in enumerate(ARMS):
        mla._dsa_lazy_planes = lazy
        mla._dsa_batch_upproj = up
        # warmup: autotune + (upproj) _w_kv_b_cat build must not pollute the count
        _run_job(gen, tokenizer, _ids(tokenizer, args.prompt_tokens, 1000 + i))
        torch.cuda.synchronize()
        c = _Counters
        c.triton.clear(); c.mm_calls = 0; c.mm_us = 0.0
        c.pool_calls = 0; c.pool_us = 0.0; c.spans.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_job(gen, tokenizer, _ids(tokenizer, args.prompt_tokens, i))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        results[name] = {
            "lazy": lazy, "batch_upproj": up,
            "wall_s": round(wall, 4),
            "prefill_spans": c.spans,
            "triton_launches": sum(c.triton.values()),
            "triton_names": dict(c.triton),
            "torch_mm_calls": c.mm_calls,
            "torch_mm_us_total": round(c.mm_us, 1),
            "pool_plane_updates": c.pool_calls,
            "pool_plane_us_total": round(c.pool_us, 1),
        }
        print(f"[count] arm={name}: triton={sum(c.triton.values())} mm={c.mm_calls} "
              f"pool_upd={c.pool_calls} wall={wall * 1e3:.1f} ms spans={c.spans}",
              flush = True)

    mla._dsa_lazy_planes = True
    mla._dsa_batch_upproj = False

    summary = {}
    b, l, u = (results[n] for n, _, _ in ARMS)
    summary["lazy_vs_base"] = {
        "triton": l["triton_launches"] - b["triton_launches"],
        "pool_updates": f"{b['pool_plane_updates']} -> {l['pool_plane_updates']}",
        "mm_calls": f"{b['torch_mm_calls']} -> {l['torch_mm_calls']}",
        "wall_s": f"{b['wall_s']:.3f} -> {l['wall_s']:.3f} "
                  f"({100.0 * (l['wall_s'] - b['wall_s']) / b['wall_s']:+.1f} %)",
    }
    summary["upproj_vs_lazy"] = {
        "mm_calls": f"{l['torch_mm_calls']} -> {u['torch_mm_calls']}",
        "wall_s": f"{l['wall_s']:.3f} -> {u['wall_s']:.3f} "
                  f"({100.0 * (u['wall_s'] - l['wall_s']) / l['wall_s']:+.1f} %)",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok = True)
    with open(args.out, "w") as f:
        json.dump({"prompt_tokens": args.prompt_tokens, "summary": summary,
                   "detail": results}, f, indent = 1)
    print(f"[summary] {json.dumps(summary, indent = 1)}")
    print(f"[out] {args.out}")
    model.unload()


if __name__ == "__main__":
    main()
