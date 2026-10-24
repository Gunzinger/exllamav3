"""opt06 launch-count evidence: CUDA kernel launches of one 2048-token prefill
(max_new_tokens=1), EXL3_GDN_SCAN_BATCHED old vs new.

Both arms run in ONE process against the same loaded model: the knob is an
import-time constant, but _chunked_scan_rows reads the module global
gated_delta_rule._scan_batched at call time, so flipping it between passes is an
exact A/B (counting only — parity arms use the real env knob in separate
processes). Kernel counts come from torch.profiler chrome traces
(cat == "kernel"), which also gives per-kernel-name breakdowns; the substitution
is documented because classic `rocprof` is not shipped in ROCm 10.0.0
(rocprofv3 is available as a cross-check).

At bsz=1 both paths run one scan pass per layer, so the launch premise (plan 06:
per-row python loop) can only show at bsz>1 — measured at bsz=1 AND bsz=4
(concurrent equal-length Jobs, the generator's ubatch path).

  PYTHONPATH=$PWD .venv-python tests/launch_gdn.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw \
      --prompt-tokens 2048 --out /tmp/opencode/launch_gdn.json
"""
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _job_ids(qbench_path, target_len, bsz, salt=0):
    """(bsz, target_len) ids; each batch row built from a DIFFERENT qbench row so
    no job (or pass — warmup vs measured use different salts) shares a prompt
    with another: the Generator's in-process prompt cache would otherwise serve
    repeats and mask the prefill (plan-04 lesson)."""
    import torch
    with open(qbench_path) as f:
        rows = [r["input_ids"] for r in json.load(f)["rows"]]
    out = []
    for b in range(bsz):
        base = rows[(salt + b) % len(rows)]
        ids = base[:]
        while len(ids) < target_len:
            ids.extend(base[1:])
        out.append(ids[:target_len])
    return torch.tensor(out, dtype=torch.long)


def _run_jobs(gen, tokenizer, ids, max_new=1):
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    jobs = [Job(input_ids=ids[b:b + 1], max_new_tokens=max_new, sampler=sampler,
                stop_conditions=[tokenizer.eos_token_id], identifier=b)
            for b in range(ids.shape[0])]
    for j in jobs:
        gen.enqueue(j)
    n = 0
    done = set()
    while len(done) < len(jobs):
        for result in gen.iterate():
            r = result.get("identifier")
            if r is None:
                continue
            if result.get("eos") or result.get("token_ids"):
                if r not in done:
                    done.add(r)
                    n += 1
    return n


def _install_triton_counter(counter):
    """Count triton kernel launches by name. Needed because this kineto build
    does not trace triton's launch API (hipLaunchKernel*ExC path): torch-op and
    extension kernels appear as cat=="kernel" events, triton ones do not."""
    from triton.runtime.jit import JITFunction

    orig = JITFunction.run

    def wrapped(self, *args, **kwargs):
        counter[self.fn.__name__] += 1
        return orig(self, *args, **kwargs)

    JITFunction.run = wrapped
    return orig


def count_kernels(prof, trace_path):
    import tempfile
    from collections import Counter
    fd, path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(trace_path))
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    os.unlink(path)
    cats = Counter(e.get("cat") for e in trace.get("traceEvents", []))
    kernels = {}
    for e in trace.get("traceEvents", []):
        if e.get("cat") in ("kernel", "gpu_kernel", "Kernel"):
            name = e["name"]
            kernels[name] = kernels.get(name, 0) + 1
    if not kernels:
        print(f"[warn] no kernel events in trace; categories: {dict(cats)}",
              flush=True)
    return kernels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-tokens", type=int, default=2048)
    p.add_argument("--ctx", type=int, default=None)
    p.add_argument("--qbench", default=os.path.join(
        os.path.dirname(ROOT), "models", "qwen3.8-flash-next-exl3",
        "qbench_prompts.json"))
    p.add_argument("--arm", choices=["old", "new"], default=None,
                   help="run ONE pass only (this arm) — for external profilers "
                        "(rocprofv3 wraps the whole process); no in-process A/B")
    p.add_argument("--bsz", type=int, default=4, help="batch size in --arm mode")
    p.add_argument("--skip-pass", action="store_true",
                   help="baseline: load only, no jobs (subtract from arm runs "
                        "to get pass-only launch counts)")
    p.add_argument("--out", default="/tmp/opencode/launch_gdn.json")
    args = p.parse_args()

    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator
    import exllamav3.modules.gated_delta_net_fn.gated_delta_rule as gdr

    if args.ctx is None:
        # concurrent jobs each need room for their own prompt + generation span
        need = 4 * (args.prompt_tokens + 16)
        args.ctx = (need + 255) // 256 * 256  # page-size multiple of 256

    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    # ONE cache + generator, created BEFORE model.load (exllamav3 rule); per-job
    # states start cleared (fresh GDNState at position 0), so every pass is
    # well-defined. 4 slots cover the bsz=4 passes; bsz=1 uses one slot.
    cache = Cache(model, max_num_tokens=args.ctx, max_batch_size=4, max_history=0)
    gen = Generator(model, cache, tokenizer)
    model.load(torch.device("cuda:0"))
    print("[load] done", flush=True)

    from torch.profiler import profile, ProfilerActivity

    # hook: record the batch shape the scan actually saw per call
    scan_calls = []
    orig = gdr._chunked_scan_rows

    def wrapped(fn, q, k, v, g, beta, recurrent_state, rows_out, rows_in,
                history, save_state):
        scan_calls.append(tuple(q.shape))
        return orig(fn, q, k, v, g, beta, recurrent_state, rows_out, rows_in,
                    history, save_state)

    gdr._chunked_scan_rows = wrapped

    triton_counter = Counter()
    _install_triton_counter(triton_counter)

    results = {}
    pass_index = 0
    if args.skip_pass:
        # baseline for external-profiler runs: model load, no jobs
        print("[baseline] load only, no passes", flush=True)
        model.unload()
        return
    bszs = (args.bsz,) if args.arm else (1, 4)
    arms = ((args.arm, args.arm == "new"),) if args.arm else \
        (("old", False), ("new", True))
    for bsz in bszs:
        for arm, batched in arms:
            # distinct prompts for warmup (salt 100+) and measured pass (salt
            # pass_index), and across batch rows — bust the prompt cache
            gdr._scan_batched = batched

            # warmup at the same shape: triton autotune must not pollute counts
            scan_calls.clear()
            triton_counter.clear()
            _run_jobs(gen, tokenizer, _job_ids(args.qbench, args.prompt_tokens,
                                               bsz, salt=100 + pass_index))
            torch.cuda.synchronize()
            tri_warm = sum(triton_counter.values())

            scan_calls.clear()
            triton_counter.clear()
            ids = _job_ids(args.qbench, args.prompt_tokens, bsz,
                           salt=pass_index)
            pass_index += 1
            if args.arm:
                # external-profiler mode: count offline from the profiler's own
                # trace; here just run the pass (autotune warmup already done)
                _run_jobs(gen, tokenizer, ids)
                torch.cuda.synchronize()
                print(f"[single] bsz={bsz} arm={arm} done "
                      f"(scan_calls={len(scan_calls)} shapes={sorted(set(scan_calls))})",
                      flush=True)
                continue
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _run_jobs(gen, tokenizer, ids)
                torch.cuda.synchronize()
            kernels = count_kernels(prof, args.out)
            calls = list(scan_calls)
            tri = dict(triton_counter)
            total = sum(kernels.values()) + sum(tri.values())
            results[f"bsz{bsz}_{arm}"] = {
                "batched": batched,
                "total_kernels": total,
                "torch_kernels": sum(kernels.values()),
                "triton_launches": sum(tri.values()),
                "triton_names": tri,
                "triton_warmup": tri_warm,
                "scan_calls": len(calls),
                "scan_shapes": sorted(set(calls)),
                "kernels": kernels,
            }
            print(f"[count] bsz={bsz} arm={arm}: total={total} "
                  f"(torch={sum(kernels.values())} triton={sum(tri.values())} "
                  f"warmup_tri={tri_warm}) scan_calls={len(calls)} "
                  f"shapes={sorted(set(calls))}", flush=True)

    gdr._chunked_scan_rows = orig
    gdr._scan_batched = True

    if args.arm:
        model.unload()
        return

    summary = {}
    for bsz in (1, 4):
        o = results[f"bsz{bsz}_old"]["total_kernels"]
        n = results[f"bsz{bsz}_new"]["total_kernels"]
        summary[f"bsz{bsz}"] = {"old": o, "new": n,
                                "delta_pct": round(100.0 * (n - o) / o, 2)}
        print(f"[summary] bsz={bsz}: old={o} new={n} "
              f"delta={100.0 * (n - o) / o:+.2f}%")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"prompt_tokens": args.prompt_tokens,
                   "summary": summary, "detail": results}, f, indent=1)
    print(f"[out] {args.out}")
    model.unload()


if __name__ == "__main__":
    main()
