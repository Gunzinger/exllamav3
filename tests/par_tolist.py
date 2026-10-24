"""opt02 parity gate: greedy temperature=0 outputs, EXL3_MOE_DEVICE_COUNT=0 vs 1.

EXL3_MOE_DEVICE_COUNT is an IMPORT-TIME constant (util/cuda_flags.py), so the two
arms are two separate process runs of this driver with the same worktree build;
the JSON outputs are then diffed (byte-identical token lists required).

  # arm A (old path)
  EXL3_MOE_DEVICE_COUNT=0 PYTHONPATH=$PWD .venv-python tests/par_tolist.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw --mode qbench \
      --out /tmp/opencode/par_tolist_qwen_old.json
  # arm B (new path, knob default ON)
  PYTHONPATH=$PWD .venv-python tests/par_tolist.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw --mode qbench \
      --out /tmp/opencode/par_tolist_qwen_new.json
  # verdict (no model load):
  PYTHONPATH=$PWD .venv-python tests/par_tolist.py \
      --compare /tmp/opencode/par_tolist_qwen_old.json /tmp/opencode/par_tolist_qwen_new.json

Cache is created BEFORE model.load(); one Generator per process; greedy
ComboSampler(temperature=0) — same pattern as opt-cpu-bf16/tests/par_qbench.py,
on the cuda device because the opt02 change is CUDA-path-only (device-gated).
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NATURAL_PROMPTS = [
    "Explain the difference between TCP and UDP to a network engineer, covering "
    "reliability, ordering, and typical use cases for each protocol.",
    "Write a short poem about autumn rain on a quiet town, in four stanzas.",
    "Summarize the main causes of the 2008 financial crisis in five bullet points, "
    "then add one sentence on what changed afterwards.",
    "What is 17 * 23? Show your reasoning step by step, then state the final answer.",
    "Describe how a refrigerator works, from the compressor cycle to why the inside "
    "stays cold. Keep it under 300 words but be technically precise.",
]


def run_generation(model_dir, mode, rows, max_new, ctx, qbench_path):
    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
    from exllamav3.generator.sampler.presets import ComboSampler

    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=ctx, max_batch_size=1, max_history=0)
    gen = Generator(model, cache, tokenizer)
    t0 = time.perf_counter()
    model.load(torch.device("cuda:0"))
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)

    if mode == "qbench":
        with open(qbench_path) as f:
            # rows carry pre-tokenized plain lists — Job wants cpu tensors
            prompts = [torch.tensor([r["input_ids"]], dtype=torch.long)
                       for r in json.load(f)["rows"]]
    else:
        prompts = [tokenizer.encode(p, add_bos=True, encode_special_tokens=True)
                   for p in NATURAL_PROMPTS]
    prompts = prompts[:rows]

    # keep room for the generated span inside the cache
    prompt_budget = ctx - max_new - 16

    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    out = {}
    for i, ids in enumerate(prompts):
        if len(ids) > prompt_budget:
            ids = ids[-prompt_budget:]
        job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler,
                  stop_conditions=[tokenizer.eos_token_id], identifier=i)
        gen.enqueue(job)
        toks = []
        done = False
        while not done:
            for result in gen.iterate():
                if result.get("identifier") is None:
                    continue
                if result.get("token_ids"):
                    toks.extend(int(t) for t in result["token_ids"])
                if result.get("eos"):
                    done = True
        toks = toks[:max_new]
        out[f"row{i}"] = {"in_len": int(ids.numel()), "tokens": toks}
        print(f"[gen] row{i}: in={int(ids.numel())} out={len(toks)} "
              f"head={toks[:8]}", flush=True)

    model.unload()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir")
    p.add_argument("--mode", default="qbench", choices=["qbench", "natural"])
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--qbench", default=os.path.join(
        os.path.dirname(ROOT), "models", "qwen3.8-flash-next-exl3",
        "qbench_prompts.json"))
    p.add_argument("--out", default="/tmp/opencode/par_tolist_out.json")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="diff two result JSONs; no model load")
    args = p.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        print(f"A: model={a['model_dir']} knob={a['knob']} rows={len(a['generations'])}")
        print(f"B: model={b['model_dir']} knob={b['knob']} rows={len(b['generations'])}")
        if set(a["generations"]) != set(b["generations"]):
            print("VERDICT: FAIL — row sets differ")
            sys.exit(1)
        ndiff = 0
        ntok = 0
        for k in sorted(a["generations"]):
            ta, tb = a["generations"][k]["tokens"], b["generations"][k]["tokens"]
            ntok += max(len(ta), len(tb))
            if ta != tb:
                ndiff += 1
                first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                             min(len(ta), len(tb)))
                print(f"  DIFF {k}: len {len(ta)} vs {len(tb)}, first flip @{first} "
                      f"({ta[first:first+4] if first < len(ta) else 'EOF'} vs "
                      f"{tb[first:first+4] if first < len(tb) else 'EOF'})")
        ha = hashlib.sha256(json.dumps(a["generations"], sort_keys=True).encode()).hexdigest()[:16]
        hb = hashlib.sha256(json.dumps(b["generations"], sort_keys=True).encode()).hexdigest()[:16]
        print(f"stream sha A={ha} B={hb}; total tokens={ntok}; rows with diffs={ndiff}")
        print(f"VERDICT: {'PASS — byte-identical' if ndiff == 0 else 'FAIL — DIVERGENCE'}")
        sys.exit(0 if ndiff == 0 else 1)

    if not args.model_dir:
        p.error("--model-dir required (or use --compare)")

    result = {
        "model_dir": args.model_dir,
        "mode": args.mode,
        "max_new": args.max_new,
        # None means the knob was unset → default ON (the new path)
        "knob": os.environ.get("EXL3_MOE_DEVICE_COUNT", "unset(default=1)"),
        "generations": run_generation(args.model_dir, args.mode, args.rows,
                                      args.max_new, args.ctx, args.qbench),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[out] {args.out} (knob={result['knob']})")


if __name__ == "__main__":
    main()
