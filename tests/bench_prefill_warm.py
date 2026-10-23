"""Plan 07 warm-cache prefill bench: consecutive prefill-only rounds, one model process.

Measures the tier's steady-state (serving) profile: round 1 pays the cold dequant per tensor,
rounds 2+ hit the BF16 shadow cache (EXL3_CPU_PREFILL_CACHE_MIB). Also runs a tier-off
reference arm in the same process (env read live per forward).

  cd /home/daniel/development/gunzi/local-llm/opt-cpu-bf16
  TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD cpu-trial/.venv/bin/python \
      tests/bench_prefill_warm.py [--tokens 512] [--rounds 4]
"""
import argparse, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EXL3_CPU_TEXT_ONLY", "1")
os.environ.setdefault("EXL3_FUSED_SAMPLER", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(ROOT, ".torch_ext"))

import torch

MODEL_DIR = "/home/daniel/development/gunzi/local-llm/models/qwen3.8-flash-next-exl3/2.05bpw"
TEXT = ("The history of computing machinery begins with mechanical aids and unfolds through "
        "electromechanical relays, vacuum tubes, transistors, and integrated circuits. ")

WORDS = TEXT.split()


def prefill_once(model, tokenizer, ctx, n_tokens, seed):
    from exllamav3 import Cache, Generator, Job
    g = torch.Generator().manual_seed(seed)
    words = [WORDS[i % len(WORDS)] for i in torch.randint(0, len(WORDS), (n_tokens // 2 + 2,), generator=g)]
    ids = tokenizer.encode(" ".join(words), add_bos=True, encode_special_tokens=True)[:, :n_tokens]
    cache = Cache(model, max_num_tokens=ctx, max_batch_size=1, max_history=0)
    gen = Generator(model, cache, tokenizer)
    job = Job(input_ids=ids, max_new_tokens=1, identifier=1)   # prefill-only
    gen.enqueue(job)
    t0 = time.perf_counter()
    done = False
    while not done:
        for res in gen.iterate():
            if res.get("identifier") is not None:
                done = True
    return time.perf_counter() - t0, ids.shape[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--ctx", type=int, default=4096)
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

    for arm, tier in (("tier-off", False), ("tier-torch", True)):
        if tier:
            os.environ["EXL3_CPU_PREFILL_TIER"] = "torch"
            os.environ["EXL3_CPU_PREFILL_CACHE_MIB"] = "65536"
        else:
            os.environ.pop("EXL3_CPU_PREFILL_TIER", None)
            os.environ.pop("EXL3_CPU_PREFILL_CACHE_MIB", None)
        times = []
        for r in range(args.rounds):
            dt, n = prefill_once(model, tokenizer, args.ctx, args.tokens, seed=100 + r)
            times.append(dt)
            print(f"[{arm}] round {r}: {n} tokens in {dt:.2f}s -> {n / dt:.2f} tok/s", flush=True)
        steady = args.tokens / min(times[1:])
        print(f"[{arm}] steady (min of rounds 1+): {steady:.2f} tok/s | "
              f"cold round: {args.tokens / times[0]:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
