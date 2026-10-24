"""opt04 parity gate: greedy temperature=0 outputs, EXL3_RECON_CACHE_MIB off vs on.

Plan 04 gate: greedy outputs must be BYTE-IDENTICAL with the reconstruct-once FP16
shadow cache disabled vs enabled — including the warm-cache steady state (the first
cached pass materializes the shadows; later passes must replay them without any
divergence).

Arm switching: EXL3_RECON_CACHE_MIB is a *runtime* knob (util/cuda_flags.py reads it
at call time), but ReconCache caches the parsed budget after its first configure() —
the sanctioned re-read seam is recon_cache.configure(), which re-reads the env and
evicts back to fit (that is also the documented shrink path). This driver therefore
flips the env AND calls configure() between passes, all within one process/model load
(a fresh process would start with a cold cache again, defeating the warm-cache check):

  pass 1  env unset + configure()   -> arm "off"    (cache disabled, never consults)
  pass 2  env =4096 + configure()   -> arm "on-cold" (prefill materializes shadows)
  pass 3  env =4096 + configure()   -> arm "on-warm" (every dense projection is a hit)

recon_cache.stats() is printed per pass: pass 2 must show entries > 0 (else the cache
never engaged and the gate is vacuous -> FAIL), pass 3 must show the same entries.

Usage (mirrors opt-tolist/tests/par_tolist.py):
  PYTHONPATH=$PWD .venv-python tests/par_reconcache.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw --mode qbench \
      --out /tmp/opencode/par_reconcache_qwen.json
  PYTHONPATH=$PWD .venv-python tests/par_reconcache.py --compare A.json B.json   # optional

The JSON records all three passes; the driver itself diffs off vs on-cold and off vs
on-warm and prints the verdict (exit 1 on any divergence).
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

# GLM yamz natural prompts (plan 04 parity gate: 4 natural prompts)
NATURAL_PROMPTS = [
    "Explain the difference between TCP and UDP to a network engineer, covering "
    "reliability, ordering, and typical use cases for each protocol.",
    "Write a short poem about autumn rain on a quiet town, in four stanzas.",
    "Summarize the main causes of the 2008 financial crisis in five bullet points, "
    "then add one sentence on what changed afterwards.",
    "What is 17 * 23? Show your reasoning step by step, then state the final answer.",
]

# Long/medium natural prompts prepended to every mode so the parity gate actually
# exercises the cached tier: the first lesson of the 2026-10-24 run is that short
# prompts (GLM natural prompts encode to 18-28 tokens) never reach the rows > 144
# tier and the gate passes vacuously (cache_engaged=False). Rows coverage wanted:
#   long  (~1500 words -> >2 chunks of >=1024 rows)  -> FUSED-Had basis (basis=1),
#        the tier every real prefill chunk uses;
#   medium (~250 words -> one 144 < rows < 1024 chunk) -> PLAIN rotated basis
#        (basis=0);
#   the short prompts + the 384-token decode steps cover rows <= 144 (GEMV tier,
#        must never touch the cache).
_WORDS = ("time person year way day thing man world life hand part child eye woman "
          "place work week case point government company number group problem fact "
          "system program question study business issue side kind head house service "
          "friend father power hour game line end member law car city community name "
          "president team minute idea kid body information back parent face others "
          "level office door health person art war history party result change morning "
          "reason research girl moment air teacher force education speed river cloud "
          "stone forest engine window garden market price value quality measure shadow "
          "signal current detail pattern feature process structure material surface "
          "balance journey machine silence flight harbour version instance committee "
          "proposal timeline dependency benchmark scheduler latency throughput provision")


def _long_text(n_words: int, seed: int) -> str:
    import random
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))

KNOB = "EXL3_RECON_CACHE_MIB"


def set_knob(mib: int | None):
    """Flip the runtime knob and force the sanctioned re-read (configure())."""
    if mib is None:
        os.environ.pop(KNOB, None)
    else:
        os.environ[KNOB] = str(mib)
    from exllamav3.modules.quant.recon_cache import recon_cache
    recon_cache.configure()  # re-reads env, evicts back to fit
    return recon_cache.stats()


def run_pass(model, tokenizer, prompts, max_new, ctx, label):
    import torch
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job, Cache, Generator

    # FRESH Cache + Generator per pass: the Generator keeps resumable KV pages keyed
    # by prompt hash, so a second pass over identical prompts would be served from the
    # *prompt* cache and never re-prefill (measured: on-cold row0 27.1 s vs off
    # 34.7 s — no prefill happened at all). The recon shadows live in the GLOBAL
    # recon_cache LRU, not in the Generator, so recreating Cache/Generator yields a
    # genuine cold-prefill pass 2 and a warm-shadow pass 3 without changing what the
    # gate compares. (Cache tensors are allocated post-load via cache.alloc().)
    cache = Cache(model, max_num_tokens=ctx, max_batch_size=1, max_history=0)
    cache.alloc(torch.device("cuda:0"))
    gen = Generator(model, cache, tokenizer)

    prompt_budget = ctx - max_new - 16
    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    out = {}
    for i, ids in enumerate(prompts):
        if len(ids) > prompt_budget:
            ids = ids[-prompt_budget:]
        t0 = time.perf_counter()
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
        print(f"  [{label}] row{i}: in={int(ids.numel())} out={len(toks)} "
              f"head={toks[:6]} ({time.perf_counter() - t0:.1f}s)", flush=True)
    del gen
    cache.free()
    del cache
    return out


def run_generation(model_dir, mode, rows, max_new, ctx, qbench_path):
    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator
    from exllamav3.modules.quant.recon_cache import recon_cache

    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    t0 = time.perf_counter()
    model.load(torch.device("cuda:0"))
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)

    if mode == "qbench":
        with open(qbench_path) as f:
            prompts = [torch.tensor([r["input_ids"]], dtype=torch.long)
                       for r in json.load(f)["rows"]]
    else:
        prompts = [tokenizer.encode(p, add_bos=True, encode_special_tokens=True)
                   for p in NATURAL_PROMPTS]
    prompts = prompts[:rows]
    # fused-path + plain-path coverage (see comment at _long_text); prepend so they
    # are the FIRST rows of every pass
    extra = [tokenizer.encode(_long_text(1500, 911), add_bos=True,
                              encode_special_tokens=True),
             tokenizer.encode(_long_text(250, 912), add_bos=True,
                              encode_special_tokens=True)]
    prompts = [e if torch.is_tensor(e) else torch.tensor([e], dtype=torch.long)
               for e in extra] + prompts

    stats_off = set_knob(None)
    print(f"[knob] off  stats={stats_off}", flush=True)
    p_off = run_pass(model, tokenizer, prompts, max_new, ctx, "off")
    # flip the knob on; pass 2 re-prefills cold (fresh Cache) and materializes shadows
    stats_cold0 = set_knob(4096)
    p_cold = run_pass(model, tokenizer, prompts, max_new, ctx, "on-cold")
    stats_cold = recon_cache.stats()
    print(f"[knob] on   stats before={stats_cold0} after={stats_cold}", flush=True)
    p_warm = run_pass(model, tokenizer, prompts, max_new, ctx, "on-warm")
    stats_warm = recon_cache.stats()
    print(f"[knob] on   stats after warm pass={stats_warm}", flush=True)

    model.unload()

    engaged = stats_cold["entries"] > 0
    verdict = {"off_vs_on-cold": p_off == p_cold, "off_vs_on-warm": p_off == p_warm,
               "cache_engaged": engaged}
    ok = all(verdict.values())
    print(f"VERDICT: {'PASS — byte-identical (off == on-cold == on-warm), cache engaged '
          if ok else 'FAIL'} {verdict}", flush=True)
    return {
        "generations": {"off": p_off, "on-cold": p_cold, "on-warm": p_warm},
        "cache_stats": {"after_cold": stats_cold, "after_warm": stats_warm},
        "verdict": verdict, "pass": ok,
    }


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
    p.add_argument("--out", default="/tmp/opencode/par_reconcache_out.json")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="diff two result JSONs; no model load")
    args = p.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        ha = hashlib.sha256(json.dumps(a["generations"], sort_keys=True).encode()).hexdigest()[:16]
        hb = hashlib.sha256(json.dumps(b["generations"], sort_keys=True).encode()).hexdigest()[:16]
        print(f"stream sha A={ha} B={hb}")
        print(f"VERDICT: {'PASS — byte-identical' if ha == hb else 'FAIL — DIVERGENCE'}")
        sys.exit(0 if ha == hb else 1)

    if not args.model_dir:
        p.error("--model-dir required (or use --compare)")

    result = {
        "model_dir": args.model_dir,
        "mode": args.mode,
        "max_new": args.max_new,
        "knob_passes": ["off(unset)", "on-cold(4096)", "on-warm(4096)"],
        **run_generation(args.model_dir, args.mode, args.rows, args.max_new,
                         args.ctx, args.qbench),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[out] {args.out}")
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
