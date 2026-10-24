"""opt04 hit/timing probe: do fresh-prefill chunks actually HIT the recon shadow
cache, and what fraction of prefill time does the eliminated reconstruct cost?

Answers the two questions the bench arms cannot separate:
  1. hit/miss counts of ReconCache.lookup/store on REAL kernels, fresh prefill,
     cache warm from a previous identical prefill (global LRU, fresh Cache per run
     so the prompt cache cannot mask the prefill);
  2. wall time spent INSIDE ext.reconstruct*/reconstruct_had_slice during a prefill
     (the exact cost a cache hit eliminates) vs total prefill wall time.

Run (GPU, ~2 min on qwen 2.05):
  PYTHONPATH=$PWD .venv-python tests/probe_reconcache_hits.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("EXL3_RECON_CACHE_MIB", "4096")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--prompt-words", type=int, default=600)
    args = ap.parse_args()

    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant import recon_cache as rcmod
    from tests.par_reconcache import _long_text

    # ---- instrumentation --------------------------------------------------------
    stats = {"lookup": 0, "hit": 0, "store": 0, "recon_calls": 0, "recon_s": 0.0,
             "hgemm_calls": 0, "hgemm_s": 0.0}
    rc = rcmod.recon_cache
    _orig_lookup, _orig_store = rc.lookup, rc.store

    def lookup(lin, basis):
        stats["lookup"] += 1
        w, hit = _orig_lookup(lin, basis)
        if hit:
            stats["hit"] += 1
        return w, hit

    def store(lin, basis, tensor):
        ok = _orig_store(lin, basis, tensor)
        if ok:
            stats["store"] += 1
            if stats["store"] <= 3:
                stats.setdefault("store_keys", []).append(
                    (lin.key, lin.trellis.data_ptr(), id(lin.trellis), basis,
                     tensor.data_ptr()))
        return ok

    def lookup_dbg(lin, basis):
        key = rc._key(lin, basis)
        entry = rc._entries.get(key)
        if entry is not None and stats["lookup"] < 3:
            stats.setdefault("lookup_dbg", []).append(
                (lin.key, key, entry.trellis is lin.trellis,
                 entry.in_features == lin.in_features,
                 entry.out_features == lin.out_features,
                 entry.trellis.data_ptr() == lin.trellis.data_ptr()))
        return lookup(lin, basis)

    rc.lookup, rc.store = lookup_dbg, store

    _orig_recon = ext.reconstruct
    _orig_had = ext.reconstruct_had_slice
    _orig_hgemm = ext.hgemm_recon

    def reconstruct(w, *a, **k):
        t0 = time.perf_counter()
        r = _orig_recon(w, *a, **k)
        stats["recon_calls"] += 1
        stats["recon_s"] += time.perf_counter() - t0
        return r

    def had_slice(w, *a, **k):
        t0 = time.perf_counter()
        r = _orig_had(w, *a, **k)
        stats["recon_calls"] += 1
        stats["recon_s"] += time.perf_counter() - t0
        return r

    def hgemm(xh, w, y):
        t0 = time.perf_counter()
        r = _orig_hgemm(xh, w, y)
        stats["hgemm_calls"] += 1
        stats["hgemm_s"] += time.perf_counter() - t0
        return r

    ext.reconstruct = reconstruct
    ext.reconstruct_had_slice = had_slice
    ext.hgemm_recon = hgemm

    # ---- model + two genuinely-fresh prefill runs -------------------------------
    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    model.load(torch.device("cuda:0"))
    ids = tokenizer.encode(_long_text(args.prompt_words, 931), add_bos=True,
                           encode_special_tokens=True)
    print(f"prompt tokens: {int(ids.numel())}", flush=True)

    def one_run(tag):
        for k in ("lookup", "hit", "store", "recon_calls", "recon_s",
                  "hgemm_calls", "hgemm_s"):
            stats[k] = 0
        cache = Cache(model, max_num_tokens=args.ctx, max_batch_size=1, max_history=0)
        dev = torch.device("cuda:0")
        for cl in cache.layers.values():
            cl.alloc(dev)
        for rl in cache.recurrent_layers.values():
            rl.alloc(dev)
        cache.initialized = True
        gen = Generator(model, cache, tokenizer)
        from exllamav3.generator.sampler.presets import ComboSampler
        from exllamav3 import Job
        t0 = time.perf_counter()
        job = Job(input_ids=ids, max_new_tokens=args.max_new,
                  sampler=ComboSampler(temperature=0.0), identifier=0)
        gen.enqueue(job)
        err = None
        for r in gen.iterate():
            if r.get("error"):
                err = r["error"]
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if err:
            print(f"[{tag}] JOB ERROR: {err}", flush=True)
        print(f"[{tag}] wall {wall:.2f}s  lookup {stats['lookup']}  hit {stats['hit']}  "
              f"store {stats['store']}  recon {stats['recon_calls']} calls / "
              f"{stats['recon_s'] * 1e3:.1f} ms  hgemm {stats['hgemm_calls']} calls / "
              f"{stats['hgemm_s']:.2f} s  lru {rc.stats()}", flush=True)
        if stats.get("store_keys"):
            print(f"[{tag}] store_keys[:3] (key, trellis_ptr, trellis_id, basis, w_ptr): "
                  f"{stats['store_keys']}", flush=True)
        if stats.get("lookup_dbg"):
            print(f"[{tag}] lookup_dbg[:3] (key, entry_is_lin_trellis, in_ok, out_ok, "
                  f"ptr_ok): {stats['lookup_dbg']}", flush=True)
        del gen
        for cl in cache.layers.values():
            cl.free()
        for rl in cache.recurrent_layers.values():
            rl.free()
        cache.initialized = False
        del cache

    one_run("run1 (cold shadows)")
    one_run("run2 (warm shadows)")
    model.unload()


if __name__ == "__main__":
    main()
