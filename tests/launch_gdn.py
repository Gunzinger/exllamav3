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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _tile_ids(qbench_path, target_len, bsz):
    with open(qbench_path) as f:
        base = list(json.load(f)["rows"][0]["input_ids"])
    ids = base[:]
    while len(ids) < target_len:
        ids.extend(base[1:])
    ids = torch.tensor([ids[:target_len]], dtype=torch.long)
    return ids.repeat(bsz, 1)


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


def count_kernels(prof, trace_path):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(trace_path))
    os.close(fd)
    prof.export_chrome_trace(path)
    with open(path) as f:
        trace = json.load(f)
    os.unlink(path)
    kernels = {}
    for e in trace.get("traceEvents", []):
        if e.get("cat") in ("kernel", "gpu_kernel"):
            name = e["name"]
            kernels[name] = kernels.get(name, 0) + 1
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
                   help="run ONE pass only (this arm, this bsz) — for external "
                        "profilers (rocprofv3); no in-process A/B, no summary")
    p.add_argument("--out", default="/tmp/opencode/launch_gdn.json")
    args = p.parse_args()

    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator
    import exllamav3.modules.gated_delta_net_fn.gated_delta_rule as gdr

    if args.ctx is None:
        # concurrent jobs each need room for their own prompt + generation span
        args.ctx = 4 * (args.prompt_tokens + 16) if args.prompt_tokens >= 1024 \
            else args.prompt_tokens + 64

    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
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

    results = {}
    bszs = (4,) if args.arm else (1, 4)
    arms = ((args.arm, args.arm == "new"),) if args.arm else \
        (("old", False), ("new", True))
    for bsz in bszs:
        for arm, batched in arms:
            cache = Cache(model, max_num_tokens=args.ctx, max_batch_size=bsz,
                          max_history=0)
            gen = Generator(model, cache, tokenizer)
            ids = _tile_ids(args.qbench, args.prompt_tokens, bsz)
            gdr._scan_batched = batched

            # warmup at the same shape: triton autotune must not pollute counts
            scan_calls.clear()
            _run_jobs(gen, tokenizer, ids)
            torch.cuda.synchronize()

            scan_calls.clear()
            if args.arm:
                # external-profiler mode: count offline from the profiler's own
                # trace; here just run the pass (autotune warmup already done)
                _run_jobs(gen, tokenizer, ids)
                torch.cuda.synchronize()
                print(f"[single] bsz={bsz} arm={arm} done "
                      f"(scan_calls={len(scan_calls)} shapes={sorted(set(scan_calls))})",
                      flush=True)
                del gen, cache
                continue
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _run_jobs(gen, tokenizer, ids)
                torch.cuda.synchronize()
            kernels = count_kernels(prof, args.out)
            calls = list(scan_calls)
            total = sum(kernels.values())
            results[f"bsz{bsz}_{arm}"] = {
                "batched": batched,
                "total_kernels": total,
                "scan_calls": len(calls),
                "scan_shapes": sorted(set(calls)),
                "kernels": kernels,
            }
            print(f"[count] bsz={bsz} arm={arm}: total={total} "
                  f"scan_calls={len(calls)} shapes={sorted(set(calls))}",
                  flush=True)
            del gen, cache

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
