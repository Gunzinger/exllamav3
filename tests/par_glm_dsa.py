"""opt05 parity gate (plan 05): greedy temperature=0 outputs on GLM-5.3 DSA models,
EXL3_DSA_LAZY_PLANES / EXL3_DSA_BATCH_UPPROJ arms.

Both knobs are IMPORT-TIME constants (modules/mla_attn.py), so each arm is a separate
process run of this driver; the JSON outputs are then diffed (byte-identical token
lists required). Arms:
  baseline : EXL3_DSA_LAZY_PLANES=0 EXL3_DSA_BATCH_UPPROJ=0
  lazy     : EXL3_DSA_LAZY_PLANES=1 (default)
  upproj   : EXL3_DSA_LAZY_PLANES=1 EXL3_DSA_BATCH_UPPROJ=1

Context regimes (each row = exact-length prompt + greedy span):
  1024  pure dense (prompt <= index_topk 2048, decode stays under the window)
  3000  dense->sparse crossing INSIDE the run (prefill chunk 2 is sparse -> the lazy
        arm's one-shot bulk pool rebuild must match the baseline's incremental upkeep)
  8192  delivered as 4 x 2048-token prefill chunks (the Generator's default
        max_chunk_size=2048; verified per-run by the prefill chunk log below)
  1024+1200  decode crossing: dense prefill, sparse decode -> exercises the
        bc_mla_step seqlen=0 arrears bulk rebuild before the first captured step

Chunking/crossing evidence: MLAttention._attend is wrapped (class level, both arms
identically) and every PREFILL call (seqlen > 1) logs its [host, host+seqlen) span;
spans past 2048 mark the sparse regime.

  # arm A (baseline)
  EXL3_DSA_LAZY_PLANES=0 EXL3_DSA_BATCH_UPPROJ=0 PYTHONPATH=$PWD .venv-python \
      tests/par_glm_dsa.py --model-dir models/GLM-5.3-Flash-EXL3-Yamz/mix \
      --out /tmp/opencode/par_glm_base.json
  # arm B (default-on lazy planes)
  PYTHONPATH=$PWD .venv-python tests/par_glm_dsa.py --model-dir <same> \
      --out /tmp/opencode/par_glm_lazy.json
  # verdict (no model load)
  .venv-python tests/par_glm_dsa.py --compare /tmp/opencode/par_glm_base.json \
      /tmp/opencode/par_glm_lazy.json

Cache is created BEFORE model.load(); one Generator per process; jobs run
sequentially on max_batch_size=1; prompts are exact-length salted constructions
(distinct per regime AND row, so the in-process prompt cache never serves a repeat).
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

# 4 rotated natural prompts; every (regime, row) prompt is padded/truncated to the
# exact regime length from one of these + filler, salted by (regime, row)
NATURAL_PROMPTS = [
    "Explain the difference between TCP and UDP to a network engineer, covering "
    "reliability, ordering, and typical use cases for each protocol.",
    "Write a short essay about how public libraries have changed in the digital "
    "age, touching on funding, community programs, and digital lending.",
    "Summarize the main causes of the 2008 financial crisis in five bullet points, "
    "then add one sentence on what changed in banking regulation afterwards.",
    "Describe how a modern rice cooker works, from the temperature sensor to the "
    "fuzzy-logic cycle, and why the rice rests after cooking.",
]
FILLER = (
    " The weather turned mild by late morning, and the delivery schedule for the "
    "warehouse shifted accordingly; nobody expected the order to arrive before "
    "Thursday, though the manifest listed it as pending since Monday. "
    "Engineers reviewed the calibration data twice, found the drift acceptable, "
    "and signed off on the batch with a short note about the humidity bias."
)


def exact_ids(tokenizer, n_tokens, salt_label, row):
    """Deterministic token list of EXACTLY n_tokens tokens; salted by (salt_label, row)
    so distinct regimes never share a prompt (even equal lengths)."""
    # deterministic regime rotation (never hash(): randomized per process)
    r_sel = int.from_bytes(hashlib.sha256(salt_label.encode()).digest()[:4], "big")
    base = NATURAL_PROMPTS[(row + r_sel) % len(NATURAL_PROMPTS)]
    enc = lambda s, bos: tokenizer.encode(s, add_bos = bos, encode_special_tokens = bos) \
        .flatten().tolist()
    ids = enc(base, True)
    fill = enc(FILLER * (1 + row + len(salt_label)), False)
    salt = hashlib.sha256(f"{salt_label}:{row}".encode()).digest()
    ids = ids + [fill[(salt[i % len(salt)] * 31 + i * 7) % len(fill)]
                 for i in range(len(salt) * 4)]
    stream = enc(FILLER, False)
    k = int.from_bytes(salt[:4], "big")
    while len(ids) < n_tokens:
        ids.append(stream[(k := (k * 1103515245 + 12345) & 0x7FFFFFFF) % len(stream)])
    return ids[:n_tokens]


def run_generation(model_dir, regimes, rows, max_new, ctx, decode_new):
    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
    from exllamav3.generator.sampler.presets import ComboSampler
    import exllamav3.modules.mla_attn as mla

    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    # BEFORE model.load (exllamav3 rule); room for the largest prompt + its span
    cache = Cache(model, max_num_tokens = ctx, max_batch_size = 1, max_history = 0)
    gen = Generator(model, cache, tokenizer)

    # prefill chunk / crossing evidence: log every prefill-sized _attend span
    orig_attend = mla.MLAttention._attend
    spans = []

    def attend(self, x, bsz, seqlen, params, *a, **kw):
        if seqlen > 1:
            hs = params.get("_mla_host_seqlens")
            if hs is None:
                cs = params.get("cache_seqlens")
                hs = cs.tolist() if isinstance(cs, torch.Tensor) and cs.device.type == "cpu" \
                    else [0] * bsz
            spans.append((int(hs[0]), int(hs[0]) + int(seqlen)))
        return orig_attend(self, x, bsz, seqlen, params, *a, **kw)

    mla.MLAttention._attend = attend

    t0 = time.perf_counter()
    model.load(torch.device("cuda:0"))
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush = True)

    sampler = ComboSampler(temperature = 0.0, top_p = 1.0, top_k = 0)
    out = {}
    for regime in regimes:
        r_new = decode_new if "+" in regime else max_new
        r_len = int(regime.split("+")[0])
        n_rows = 1 if "+" in regime else rows
        for row in range(n_rows):
            ids = torch.tensor([exact_ids(tokenizer, r_len, regime, row)], dtype = torch.long)
            spans.clear()
            t1 = time.perf_counter()
            job = Job(input_ids = ids, max_new_tokens = r_new, sampler = sampler,
                      stop_conditions = [tokenizer.eos_token_id],
                      identifier = f"{regime}:{row}")
            gen.enqueue(job)
            toks, done = [], False
            while not done:
                for result in gen.iterate():
                    if result.get("identifier") is None:
                        continue
                    if result.get("token_ids"):
                        toks.extend(int(t) for t in result["token_ids"])
                    if result.get("eos"):
                        done = True
            toks = toks[:r_new]
            key = f"reg{regime}_row{row}"
            out[key] = {
                "in_len": int(ids.numel()), "new": r_new, "tokens": toks,
                "prefill_spans": spans[:16],
            }
            sp = " ".join(f"[{a}->{b}]{'S' if b > 2048 else 'd'}" for a, b in spans[:16])
            print(f"[gen] {key}: in={ids.numel()} out={len(toks)} "
                  f"chunks={len(spans)} {sp} ({time.perf_counter() - t1:.1f}s)",
                  flush = True)

    model.unload()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir")
    p.add_argument("--regimes", default = "1024,3000,8192,1024+1200",
                   help = "comma list of exact prompt lengths; N+M = dense prefill N "
                          "with M decode tokens (decode-crossing regime)")
    p.add_argument("--rows", type = int, default = 4)
    p.add_argument("--max-new", type = int, default = 512)
    p.add_argument("--decode-new", type = int, default = 1200,
                   help = "generated span for the N+M decode-crossing regime")
    p.add_argument("--ctx", type = int, default = 8960)
    p.add_argument("--out", default = "/tmp/opencode/par_glm_dsa_out.json")
    p.add_argument("--compare", nargs = 2, metavar = ("A", "B"),
                   help = "diff two result JSONs; no model load")
    args = p.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        print(f"A: model={a['model_dir']} knob={a['knob']}")
        print(f"B: model={b['model_dir']} knob={b['knob']}")
        if set(a["generations"]) != set(b["generations"]):
            print("VERDICT: FAIL — regime/row sets differ")
            sys.exit(1)
        ndiff = 0
        by_regime = {}
        for k in sorted(a["generations"]):
            ta = a["generations"][k]["tokens"]
            tb = b["generations"][k]["tokens"]
            regime = k.split("_")[0]
            stat = by_regime.setdefault(regime, {"rows": 0, "diff": 0})
            stat["rows"] += 1
            if ta != tb:
                ndiff += 1
                stat["diff"] += 1
                first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                             min(len(ta), len(tb)))
                print(f"  DIFF {k}: len {len(ta)} vs {len(tb)}, first flip @{first}")
        for regime, stat in sorted(by_regime.items()):
            print(f"  {regime}: {stat['rows'] - stat['diff']}/{stat['rows']} rows identical")
        ntok = sum(max(len(a["generations"][k]["tokens"]), len(b["generations"][k]["tokens"]))
                   for k in a["generations"])
        print(f"total tokens={ntok}; rows with diffs={ndiff}")
        print(f"VERDICT: {'PASS — zero divergence' if ndiff == 0 else 'FAIL — DIVERGENCE'}")
        sys.exit(0 if ndiff == 0 else 1)

    if not args.model_dir:
        p.error("--model-dir required (or use --compare)")

    import exllamav3.modules.mla_attn as mla
    regimes = args.regimes.split(",")
    result = {
        "model_dir": args.model_dir,
        "regimes": regimes,
        "max_new": args.max_new,
        "knob": f"LAZY_PLANES={mla._dsa_lazy_planes:d} BATCH_UPPROJ={mla._dsa_batch_upproj:d}",
        "generations": run_generation(args.model_dir, regimes, args.rows,
                                      args.max_new, args.ctx, args.decode_new),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok = True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[out] {args.out} (knob={result['knob']})")


if __name__ == "__main__":
    main()
