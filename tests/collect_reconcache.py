#!/usr/bin/env python3
"""Collect opt04 bench rows (arm-labeled) into a summary table for PLAN-opt-reconcache.md.

Scans benchmarks/<ts>-<mode>/ artifact dirs created after --since, groups by LABEL
(parsed from meta.txt), prints per-arm prefill tok/s (mean±spread over reps),
decode tok/s, GTT peaks, and joins the warm-resend JSONs from /tmp/opencode/reconcache.
Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--benchmarks", default="/home/daniel/development/gunzi/local-llm/benchmarks")
AP.add_argument("--evidence", default="/tmp/opencode/reconcache")
AP.add_argument("--since", default="20261024-053200",
                help="only artifact dirs with timestamp >= this (poisoned first run excluded)")
AP.add_argument("--tag", default="", help="filter warm-resend files by tag prefix")
args = AP.parse_args()

arms: dict[str, dict] = {}

for d in sorted(glob.glob(os.path.join(args.benchmarks, "*-prefill")) +
                glob.glob(os.path.join(args.benchmarks, "*-latency"))):
    ts = os.path.basename(d).split("-")[0] + "-" + os.path.basename(d).split("-")[1]
    if ts < args.since:
        continue
    meta = open(os.path.join(d, "meta.txt")).read() if os.path.exists(os.path.join(d, "meta.txt")) else ""
    m = re.search(r"LABEL=(\S+)", meta)
    label = m.group(1) if m else "?"
    mode = "prefill" if d.endswith("-prefill") else "latency"
    rpt = os.path.join(d, "report.txt")
    if not os.path.exists(rpt):
        continue
    txt = open(rpt).read()
    a = arms.setdefault(label, {"prefill": [], "latency": []})
    if mode == "prefill":
        rows = re.findall(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                          txt, re.M)
        gt = re.search(r"GTT peak ([\d.]+) MiB \(start ([\d.]+)\)", txt)
        a["prefill"].append({"ts": ts, "rows": [
            {"target": int(r[0]), "actual": int(r[1]), "cached": int(r[2]),
             "prefill_s": float(r[3]), "tok_s": float(r[4])} for r in rows],
            "gtt_peak": float(gt.group(1)) if gt else None})
    else:
        dec = re.search(r"decode\s+([\d.]+) tok/s", txt)
        itl = re.search(r"ITL\s+mean\s+([\d.]+) ms\s+p50", txt)
        gt = re.search(r"GTT peak ([\d.]+) MiB \(start ([\d.]+)\)", txt)
        a["latency"].append({"ts": ts, "decode_tok_s": float(dec.group(1)) if dec else None,
                             "itl_ms": float(itl.group(1)) if itl else None,
                             "gtt_peak": float(gt.group(1)) if gt else None})

for label in sorted(arms):
    a = arms[label]
    print(f"\n== {label} ==")
    agg: dict[int, list[float]] = {}
    for run in a["prefill"]:
        cached_bad = [r["target"] for r in run["rows"] if r["cached"] > 0]
        if cached_bad:
            print(f"  {run['ts']}: PROMPT-CACHE HIT on sizes {cached_bad} — row POISONED")
        for r in run["rows"]:
            agg.setdefault(r["target"], []).append(r["tok_s"])
        if run["gtt_peak"]:
            print(f"  {run['ts']}: GTT peak {run['gtt_peak']:.0f} MiB")
    for tgt in sorted(agg):
        v = agg[tgt]
        print(f"  pp{tgt}: {statistics.fmean(v):7.1f} tok/s  (n={len(v)}, "
              f"spread {min(v):.1f}–{max(v):.1f})")
    for run in a["latency"]:
        print(f"  {run['ts']}: decode {run['decode_tok_s']} tok/s, ITL {run['itl_ms']} ms, "
              f"GTT peak {run['gtt_peak']:.0f} MiB")

print("\n== warm-resend ==")
for f in sorted(glob.glob(os.path.join(args.evidence, f"{args.tag}*-warmresend.json"))):
    d = json.load(open(f))
    print(f"\n{os.path.basename(f)}  (knob={d['knob']})")
    for k in ("full", "resend", "suffix"):
        r = d[k]
        print(f"  {k:7s}: actual={r['prompt_tokens']} cached={r['cached_tokens']} "
              f"prefill={r['time_prefill_s']}s uncached_tok_s={r['uncached_tok_s']}")
