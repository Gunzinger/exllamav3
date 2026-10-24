"""opt06 parity gate: greedy temperature=0 outputs AND GDN recurrent states,
EXL3_GDN_SCAN_BATCHED=0 vs =1 (default ON).

EXL3_GDN_SCAN_BATCHED is an IMPORT-TIME constant (gated_delta_rule.py:255), so the
two arms are two separate process runs of this driver with the same worktree build;
results are then diffed with --compare (tokens) / --compare-state (recurrent pool).

  # arm A (old per-row loop) — qbench token parity AND 512/2048-state capture in one load
  EXL3_GDN_SCAN_BATCHED=0 PYTHONPATH=$PWD .venv-python tests/par_gdn.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw --mode all \
      --out /tmp/opencode/par_gdn_205_old
  # arm B (new batched path, knob default ON)
  PYTHONPATH=$PWD .venv-python tests/par_gdn.py \
      --model-dir models/qwen3.8-flash-next-exl3/2.05bpw --mode all \
      --out /tmp/opencode/par_gdn_205_new
  # verdicts (no model load)
  PYTHONPATH=$PWD .venv-python tests/par_gdn.py --compare A.json B.json
  PYTHONPATH=$PWD .venv-python tests/par_gdn.py --compare-state A_512.pt B_512.pt

State capture: for each --state-lens length one job with max_new_tokens=1 runs on
the pre-load cache/generator (each Job starts a fresh GDNState at position 0,
which clears the slot's pool rows, so every prefill starts from the zero state).
A hook around
gated_delta_rule._chunked_scan_rows snapshots every recurrent-layer pool right
after the prefill calls — decode runs the fused step path, which never enters
_chunked_scan_rows, so the snapshot IS the post-prefill state (plan 06
§Implementation.4). Each layer dumps pool rows 0..1 (base + scratch) plus the conv
ring row 0, to a torch-save file per prompt length.

Driver pattern follows opt-tolist/tests/par_tolist.py: Cache BEFORE model.load(),
greedy ComboSampler(temperature=0), one model load per process.
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


def _run_job(gen, tokenizer, ids, max_new=1):
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    job = Job(input_ids=ids.clone(), max_new_tokens=max_new, sampler=sampler,
              stop_conditions=[tokenizer.eos_token_id], identifier=0)
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
    return toks[:max_new]


# pool id -> (pool dtype, rows 0..1 clone on cpu); updated after every chunk-scan
# call. Decode (fused step path) never calls _chunked_scan_rows, so after a
# max_new_tokens=1 job these hold the post-prefill states of every recurrent layer.
_STATE_SNAP = {}


def _install_state_hook():
    import exllamav3.modules.gated_delta_net_fn.gated_delta_rule as gdr

    orig = gdr._chunked_scan_rows

    def wrapped(fn, q, k, v, g, beta, recurrent_state, rows_out, rows_in,
                history, save_state):
        out = orig(fn, q, k, v, g, beta, recurrent_state, rows_out, rows_in,
                   history, save_state)
        if recurrent_state is not None:
            _STATE_SNAP[id(recurrent_state)] = (
                recurrent_state.dtype,
                recurrent_state[:2, 0].detach().to("cpu", copy=True),
            )
        return out

    gdr._chunked_scan_rows = wrapped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir")
    p.add_argument("--mode", default="all", choices=["all", "qbench", "state"])
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--row-picks", type=int, nargs="+", default=None,
                   help="qbench row indices (default: first --rows rows)")
    p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--state-lens", type=int, nargs="+", default=[512, 2048])
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--qbench", default=os.path.join(
        os.path.dirname(ROOT), "models", "qwen3.8-flash-next-exl3",
        "qbench_prompts.json"))
    p.add_argument("--out", default="/tmp/opencode/par_gdn_out",
                   help="token JSON base name; state files get _<len>.pt suffixes")
    args = p.parse_args()

    do_tokens = args.mode in ("all", "qbench")
    do_state = args.mode in ("all", "state")
    if not (do_tokens or do_state):
        p.error("nothing to do")
    if not args.model_dir:
        p.error("--model-dir required")

    import torch
    from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
    from exllamav3.generator.sampler.presets import ComboSampler

    knob = os.environ.get("EXL3_GDN_SCAN_BATCHED", "unset(default=1)")

    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    # token-parity cache first (pattern: Cache BEFORE model.load())
    cache = Cache(model, max_num_tokens=args.ctx, max_batch_size=1, max_history=0)
    gen = Generator(model, cache, tokenizer)
    t0 = time.perf_counter()
    model.load(torch.device("cuda:0"))
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)

    sampler = ComboSampler(temperature=0.0, top_p=1.0, top_k=0)
    prompt_budget = args.ctx - args.max_new - 16
    out_path = args.out if args.out.endswith(".json") else args.out + ".json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if do_tokens:
        with open(args.qbench) as f:
            qrows = json.load(f)["rows"]
        picks = args.row_picks if args.row_picks else list(range(args.rows))
        generations = {}
        for i in picks:
            ids = torch.tensor([qrows[i]["input_ids"]], dtype=torch.long)
            if ids.numel() > prompt_budget:
                ids = ids[:, -prompt_budget:]
            toks = _run_job(gen, tokenizer, ids, args.max_new)
            generations[f"row{i}"] = {"in_len": int(ids.numel()), "tokens": toks}
            print(f"[gen] row{i}: in={int(ids.numel())} out={len(toks)} "
                  f"head={toks[:8]}", flush=True)
        result = {"model_dir": args.model_dir, "mode": args.mode,
                  "max_new": args.max_new, "knob": knob,
                  "generations": generations}
        with open(out_path, "w") as f:
            json.dump(result, f)
        print(f"[out] {out_path} (knob={knob})", flush=True)

    if do_state:
        _install_state_hook()
        with open(args.qbench) as f:
            base_ids = list(json.load(f)["rows"][0]["input_ids"])
        for target_len in args.state_lens:
            # the pre-load cache/generator is reused: each Job starts a fresh
            # GDNState at position 0, which clears the slot's pool rows
            # (GDNLayerState.clear), so the prefill starts from the zero state
            ids = base_ids[:]
            while len(ids) < target_len:
                ids.extend(base_ids[1:])  # skip the BOS on repeats
            ids = torch.tensor([ids[:target_len]], dtype=torch.long)
            _STATE_SNAP.clear()
            toks = _run_job(gen, tokenizer, ids, 1)

            layers = {}
            for n, (instance, l) in enumerate(
                    cache.get_all_recurrent_layers().items()):
                if not hasattr(l, "recurrent_state"):
                    continue  # PLE etc. — a different scan path, untouched here
                dtype, rows = _STATE_SNAP[id(l.recurrent_state)]
                layers[n] = {
                    "instance": str(instance),
                    "cls": type(l).__name__,
                    "pool_dtype": str(dtype).replace("torch.", ""),
                    "recurrent": rows,
                    "conv": l.conv_state[:1, :, :l.module.conv_kernel_size]
                        .detach().to("cpu", copy=True),
                }
            payload = {"model_dir": args.model_dir, "prompt_len": int(target_len),
                       "knob": knob, "tokens": toks, "layers": layers}
            spath = f"{args.out}_{target_len}.pt"
            torch.save(payload, spath)
            print(f"[state] len={target_len}: {len(layers)} layers -> {spath}, "
                  f"token={toks}", flush=True)
            _STATE_SNAP.clear()

    model.unload()


def compare_tokens(a_path, b_path):
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
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
            print(f"  DIFF {k}: len {len(ta)} vs {len(tb)}, first flip @{first}")
    ha = hashlib.sha256(json.dumps(a["generations"], sort_keys=True).encode()).hexdigest()[:16]
    hb = hashlib.sha256(json.dumps(b["generations"], sort_keys=True).encode()).hexdigest()[:16]
    print(f"stream sha A={ha} B={hb}; total tokens={ntok}; rows with diffs={ndiff}")
    print(f"VERDICT: {'PASS — byte-identical' if ndiff == 0 else 'FAIL — DIVERGENCE'}")
    sys.exit(0 if ndiff == 0 else 1)


def compare_states(a_path, b_path):
    import torch

    GATE = {"float": 1e-3, "half": 5e-3}
    a = torch.load(a_path, weights_only=False)
    b = torch.load(b_path, weights_only=False)
    assert a["prompt_len"] == b["prompt_len"], "prompt lengths differ"
    print(f"A: knob={a['knob']} len={a['prompt_len']} tokens={a['tokens']}")
    print(f"B: knob={b['knob']} len={b['prompt_len']} tokens={b['tokens']}")
    token_match = a["tokens"] == b["tokens"]
    assert set(a["layers"]) == set(b["layers"]), "layer sets differ"
    worst_rec = worst_conv = 0.0
    nexact = 0
    nfail = 0
    for idx in sorted(a["layers"]):
        la, lb = a["layers"][idx], b["layers"][idx]
        assert la["pool_dtype"] == lb["pool_dtype"], f"layer {idx}: pool dtype differs"
        gate = GATE[la["pool_dtype"]]
        rec = (la["recurrent"].float() - lb["recurrent"].float()).abs().max().item()
        conv = (la["conv"].float() - lb["conv"].float()).abs().max().item()
        worst_rec = max(worst_rec, rec)
        worst_conv = max(worst_conv, conv)
        if max(rec, conv) == 0.0:
            nexact += 1
        if max(rec, conv) > gate:
            nfail += 1
            print(f"  OVER-GATE layer {idx} ({la.get('cls', '?')}, "
                  f"{la['pool_dtype']}): recurrent={rec:.3e} conv={conv:.3e} "
                  f"gate={gate:g}")
    print(f"layers={len(a['layers'])} bit-identical={nexact} "
          f"worst_recurrent={worst_rec:.3e} worst_conv={worst_conv:.3e} "
          f"over-gate={nfail}")
    print(f"first-token match: {token_match}")
    ok = nfail == 0 and token_match
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--compare" in argv and "--compare-state" not in argv:
        i = argv.index("--compare")
        compare_tokens(argv[i + 1], argv[i + 2])
    elif "--compare-state" in argv:
        i = argv.index("--compare-state")
        compare_states(argv[i + 1], argv[i + 2])
    else:
        main()
