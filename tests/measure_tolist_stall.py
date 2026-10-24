"""opt02: quantify the REMAINING per-layer count-readback stall on a prefill chunk.

The staged pinned ring (EXL3_MOE_DEVICE_COUNT=1) removes the pageable D2H but keeps
the host-side dependency (event sync) because three consumers still need exact host
counts. This driver measures how long the host actually blocks per prefill chunk:

  knob=0: total time in torch.Tensor.tolist() during the chunk (the old drain)
  knob=1: total time in _PinnedCountRing.read_counts() (event sync wait)

plus the number of calls (expect one per MoE layer per chunk) and the chunk wall time.
Two process runs (the knob is import-time), one 2048-token prompt, max_new_tokens=1
so exactly one prefill chunk runs.

  EXL3_MOE_DEVICE_COUNT=0 PYTHONPATH=$PWD .venv-python tests/measure_tolist_stall.py \
      --model-dir models/GLM-5.3-Flash-EXL3-Yamz/mix
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PROMPT_A = (
    "The industrial revolution reshaped Europe between 1760 and 1840. Steam power "
    "replaced water and muscle, iron replaced wood, and factories replaced workshops. "
    "Railways collapsed distance; telegraph collapsed time. Rural populations moved to "
    "cities faster than housing could follow, and the social contract had to be "
    "rewritten around wage labour. " * 40
)  # ~2900 tokens raw; cropped to --tokens below
PROMPT_B = (
    "Coral reefs occupy less than one percent of the ocean floor yet host a quarter of "
    "all marine species. Polyps build limestone skeletons; zooxanthellae farm sunlight "
    "inside their tissues. Bleaching breaks that symbiosis when water stays too warm. "
    "Reefs buffer shorelines, feed hundreds of millions of people, and generate "
    "billions in tourism revenue each year. " * 40
)  # disjoint content → zero shared prefix with PROMPT_A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--out", default="/tmp/opencode/tolist_stall.json")
    args = p.parse_args()

    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
    from exllamav3.generator.sampler.presets import ComboSampler
    import exllamav3.modules.block_sparse_mlp as bsm

    stats = {"calls": 0, "total_s": 0.0, "worst_s": 0.0}

    if os.environ.get("EXL3_MOE_DEVICE_COUNT", "1") != "0":
        orig = bsm._PinnedCountRing.read_counts
        def timed(self, counts):
            t0 = time.perf_counter()
            r = orig(self, counts)
            dt = time.perf_counter() - t0
            stats["calls"] += 1
            stats["total_s"] += dt
            stats["worst_s"] = max(stats["worst_s"], dt)
            return r
        bsm._PinnedCountRing.read_counts = timed
        what = "ring.read_counts (event sync)"
    else:
        orig_tostring = torch.Tensor.tolist
        def timed(self, *a, **k):
            t0 = time.perf_counter()
            r = orig_tostring(self, *a, **k)
            dt = time.perf_counter() - t0
            stats["calls"] += 1
            stats["total_s"] += dt
            stats["worst_s"] = max(stats["worst_s"], dt)
            return r
        torch.Tensor.tolist = timed
        what = "Tensor.tolist (pageable drain)"

    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.ctx, max_batch_size=1, max_history=0)
    gen = Generator(model, cache, tokenizer)
    model.load(torch.device("cuda:0"))

    # warm one chunk first (ring alloc, autotune), reset stats, then measure.
    # Fully disjoint prompts per rep so neither the KV/prompt prefix cache nor
    # first-touch autotune pollutes the measured pass.
    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    prompts = [PROMPT_A, PROMPT_B]
    for rep in range(2):
        ids = tokenizer.encode(prompts[rep], add_bos=True, encode_special_tokens=True)
        ids = ids[:, :args.tokens]
        stats.update(calls=0, total_s=0.0, worst_s=0.0)
        job = Job(input_ids=ids, max_new_tokens=1, sampler=sampler, identifier=rep)
        gen.enqueue(job)
        t0 = time.perf_counter()
        for _ in gen.iterate():
            pass
        wall = time.perf_counter() - t0
        if rep == 0:
            print(f"[warm] wall={wall:.3f}s (discarded)", flush=True)
            continue
        result = {
            "model_dir": args.model_dir,
            "knob": os.environ.get("EXL3_MOE_DEVICE_COUNT", "unset(default=1)"),
            "prompt_tokens": int(ids.numel()),
            "chunk_wall_s": wall,
            "stall_what": what,
            "stall_calls": stats["calls"],
            "stall_total_s": stats["total_s"],
            "stall_worst_ms": stats["worst_s"] * 1e3,
            "stall_pct_of_chunk": 100.0 * stats["total_s"] / wall,
        }
        print(json.dumps(result, indent=2), flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f)

    model.unload()


if __name__ == "__main__":
    main()
