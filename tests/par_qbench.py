"""Plan 07 parity: greedy qbench rows, tier off vs tier torch, one model process.

Metric per plan §Verification 1: argmax-stable outputs — token flips per 256 generated tokens
(plan rule: <= 2 flips / 512 tokens documented). Env knobs are read live per forward
(util/cuda_flags), so both arms run in one process; the off arm never consults the tier.

  cd /home/daniel/development/gunzi/local-llm/opt-cpu-bf16
  TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD cpu-trial/.venv/bin/python \
      tests/par_qbench.py [--rows 3] [--tokens 256]
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EXL3_CPU_TEXT_ONLY", "1")
os.environ.setdefault("EXL3_FUSED_SAMPLER", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(ROOT, ".torch_ext"))

import torch

MODEL_DIR = "/home/daniel/development/gunzi/local-llm/models/qwen3.8-flash-next-exl3/2.05bpw"
QBENCH = "/home/daniel/development/gunzi/local-llm/models/qwen3.8-flash-next-exl3/qbench_prompts.json"


def run_arm(model, tokenizer, ids, max_new, tier, cache_mib, ctx):
    from exllamav3 import Cache, Generator, Job
    from exllamav3.generator.sampler.presets import ComboSampler
    if tier:
        os.environ["EXL3_CPU_PREFILL_TIER"] = "torch"
        os.environ["EXL3_CPU_PREFILL_CACHE_MIB"] = str(cache_mib)
    else:
        os.environ.pop("EXL3_CPU_PREFILL_TIER", None)
        os.environ.pop("EXL3_CPU_PREFILL_CACHE_MIB", None)
    cache = Cache(model, max_num_tokens=ctx, max_batch_size=1, max_history=0)
    gen = Generator(model, cache, tokenizer)
    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler,
              stop_conditions=[tokenizer.eos_token_id], identifier=1)
    gen.enqueue(job)
    toks = []
    done = False
    while not done:
        for result in gen.iterate():
            if result.get("identifier") is None:
                continue
            if result.get("token_ids"):
                toks.extend(result["token_ids"])
            if result.get("eos"):
                done = True
    return toks[:max_new]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--out", default="/tmp/opencode/par_qbench.pt")
    args = p.parse_args()

    torch.set_num_threads(12)
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "10")

    from exllamav3 import Config, Model, Tokenizer
    t0 = time.perf_counter()
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    model.load(torch.device("cpu"))
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)

    with open(QBENCH) as f:
        rows = json.load(f)["rows"]

    results = {}
    total_flips = 0
    total_toks = 0
    for r in range(args.rows):
        ids = torch.tensor(rows[r]["input_ids"], dtype=torch.long)[None, : args.ctx - args.tokens - 8]
        t_ref = run_arm(model, tokenizer, ids, args.tokens, tier=False, cache_mib=0, ctx=args.ctx)
        t_new = run_arm(model, tokenizer, ids, args.tokens, tier=True, cache_mib=65536, ctx=args.ctx)
        n = min(len(t_ref), len(t_new))
        diffs = [i for i in range(n) if t_ref[i] != t_new[i]]
        first = diffs[0] if diffs else -1
        flips = len(diffs)
        total_flips += flips
        total_toks += n
        print(f"[row {r}] {n} tokens, flips {flips} ({flips / max(n, 1) * 1000:.1f}/1000), "
              f"first diff at {first}", flush=True)
        results[f"row{r}"] = {"ref": t_ref, "new": t_new, "flips": flips, "first_diff": first}
    print(f"[total] {total_flips} flips / {total_toks} tokens "
          f"({total_flips / max(total_toks, 1) * 512:.2f} per 512; plan rule <= 2)")
    torch.save(results, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
