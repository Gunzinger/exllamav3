# PLAN-opt-tolist — MoE expert-count drain removal (opt02, branch `opt/tolist-counts`)

**VERIFICATION COMPLETE 2026-10-24. Verdict: parity PASS, perf acceptance FAIL — staged
version gains nothing (the drain was never a bottleneck) and must not be merged
default-ON. Path B (device-side scan) = NO-GO: measured stall ceiling ~0.1 % of prefill.**

## Implementation state

| Piece | Status | Notes |
|---|---|---|
| Env gate `EXL3_MOE_DEVICE_COUNT` (cuda_flags.py, default ON) | landed | own EOF block, stdlib-only; **import-time constant** (two-process A/B required, not live-flippable) |
| Pinned-ring count readback (batch path only, device-gated) | landed | `_PinnedCountRing.read_counts` — async D2H + **event sync (host still blocks; values identical to tolist)** |
| Pinned-ring slot-table H2D (FUSED_DET tables) | landed | `_PinnedCountRing.upload_tables` |
| Worst-case-grid fused launch `run_fused(-1)` | landed (non-mtile branch) | M-tile tiering untouched; this host MOE_MTILE=False |
| True sync removal (device-side tier plan / Path B scan) | NOT started | **measured pointless, see verdict** |
| CPU/MoeCpuHost branch | untouched by design | existing offload-mode gates |
| bsz≤8 decode / CUDA-graph branch (`run_bszN`) | untouched by design | — |
| Unit tests (model-free) | landed, 5/5 PASS re-verified 2026-10-24 | `tests/test_moe_device_counts.py` |
| Parity driver | landed `tests/par_tolist.py` (6ddfad72) | two-process greedy A/B + `--compare` verdict |
| Stall-measurement driver | landed `tests/measure_tolist_stall.py` (44ed3e5f) | per-chunk readback host-block timing, disjoint prompts per rep (prefix cache defeated) |

Branch head at verification: `44ed3e5f` (off cpu-only `bc731a5e`).

## Parity gate — PASS (zero divergence, byte-identical)

Two process runs per model, same worktree build, greedy `ComboSampler(temperature=0)`,
GPU (cuda:0), Cache before `model.load()`. Compare via `tests/par_tolist.py --compare`.

| Model | Knob | Prompts | Tokens compared | Divergence | Verdict |
|---|---|---|---|---|---|
| qwen3.8-flash 2.05bpw | 0 vs 1(default) | qbench rows 0–3 (in 36–408 tok), 384 new each | 1534 | 0 rows, 0 flips | **PASS** |
| GLM-5.3-Flash-EXL3-Yamz mix | 0 vs 1(default) | 5 varied natural prompts, 320 new each | 1600 | 0 rows, identical stream sha | **PASS** |

## Benches — same-window paired A/B (2026-10-24 03:44–04:39)

Both arms served from the worktree (`PYTHONPATH=…/opt-tolist` inherited by the server,
verified via `/proc/<pid>/environ` per arm; knob exists only in the worktree, so a delta
is only possible with branch code loaded). qwen `MODEL_REV=2.05bpw`, GLM `MODEL_REV=mix`;
CTX=32768 KV=8; REPS=3; PROMPT_STYLE=real; per arm: fresh server → warm-up discard at
4096 → prefill sweep → latency. Labels: `tolist-base` (knob=0) / `tolist-fix` (default ON);
discarded first attempt labels `tolist-warm-discard` (8192, contaminated — see anomalies),
`tolist-base2/fix2/fix-first/base-second/fix3` (GLM 512-only re-samples).

| Model | Metric | base (knob=0) | fix (knob=1) | Δ |
|---|---|---|---|---|
| qwen 2.05 | pp512 tok/s | 171.6 | 170.7 | −0.5 % |
| qwen 2.05 | pp2048 tok/s | 175.4 | 174.9 | −0.3 % |
| qwen 2.05 | pp8192 tok/s | 141.4 | 141.1 | −0.2 % |
| qwen 2.05 | decode tok/s | 12.62 | 12.63 | +0.1 % (neutral) |
| GLM yamz | pp2048 tok/s | 73.8 | 74.6 | +1.1 % (noise) |
| GLM yamz | pp8192 tok/s | 52.9 | 53.0 | +0.2 % (noise) |
| GLM yamz | decode tok/s | 8.37 | 8.38 | neutral |
| GLM yamz | pp512 tok/s | 81.5 / 81.8 / 81.3 (stable) | 75.4 / 74.3 / 81.2 / 81.8 | **bimodal** — see anomalies |

## Remaining-drain quantification (Path B go/no-go input)

`tests/measure_tolist_stall.py`, one full 2048-token prefill chunk, steady state
(warm pass discards autotune), disjoint prompts per rep, host-block timed inside
`Tensor.tolist` (knob=0) / `_PinnedCountRing.read_counts` (knob=1):

| Model | old drain (tolist) | staged ring (event sync) | share of 3.5 s chunk |
|---|---|---|---|
| GLM yamz | 78 calls, **1.10 ms** total, worst 0.077 ms | 41 calls, **2.71 ms**, worst 0.245 ms | 0.031 % / 0.078 % |
| qwen 2.05 | 88 calls, **1.24 ms** total, worst 0.089 ms | 47 calls, **2.49 ms**, worst 0.189 ms | (0.07 % / 0.14 % of chunk) |

**Path B (device-side exclusive scan, C++) = NO-GO.** The entire host-side count
dependency costs 1–3 ms per 2048-token chunk — 0.03–0.14 % of chunk wall — two orders of
magnitude below the 3 % acceptance bar. The D2H of `E+1` ints completes behind routing
work the GPU is still executing, so the launch stream is never starved; plan 02's
"serializes the stream, expect 3–8 %" premise is refuted by direct measurement. Removing
the sync entirely (device-side plan/tables/slicing) can recover at most ~0.1 % prefill and
would spend significant C++ complexity. Do not build it for this reason.

## Acceptance verdict (plan 02 §Acceptance)

| Criterion | Result |
|---|---|
| Zero greedy-output diffs, both models | **PASS** |
| ≥ 3 % pp2048 on qwen 2.05 | **FAIL** (−0.3 %, noise) |
| GLM pp2048 ≥ + 3 % (42 drains) | **FAIL** (+1.1 %, noise) |
| No decode regression > 2 % | **PASS** (neutral both models) |

**Overall: REJECT for merge.** The staged version has no benefit and one unexplained
hazard (below). Recommended disposition: keep `opt/tolist-counts` unmerged (rollback is
trivial — the knob exists only on the branch; production `cpu-only` is unaffected), or at
most re-land as default-OFF documentation of the staging pattern. The pinned-ring
machinery is correct (unit 5/5, parity clean) but solves a problem this host doesn't have.

## Anomalies / lessons

1. **GLM@512 fix-arm bimodality**: fix samples 75.4, 74.3, then 81.2 (order-swapped pair)
   and 81.8 (third independent sample); base is stable at 81.3–81.8. The two slow samples
   were both fix-second in base-first pairs, but the swapped pair and later samples came
   out equal — not a stable regression, not order either. Un-isolated between the
   worst-case-grid launch and the ring H2D (not separable via env; the knob bundles both).
   Needs a dedicated A/A study only if 512-heavy GLM serving ever matters.
2. **Discarded row**: the first `tolist-base` prefill row (03:46:21, pp_tok_s=25894) is
   contaminated — the warm-up ran at 8192 and collided with the measured 8192 seed →
   server prompt-cache hit (`cached=9472`, 0.12 s "prefill"). Warm-up size moved to 4096;
   every reported row re-measured after that change. Lesson: warm-up benches must use a
   size outside the measured set on this server (in-process prompt cache).
3. **Possible big prefill lever spotted (out of scope, unverified)**: in the stall-driver
   context (KV=fp16, CTX=8192, no MTP) a GLM 2048-chunk prefills in ~3.5 s (~575 tok/s)
   vs 31.9 s (73.8 tok/s) on the KV=8 server — a ~7–8× gap that reproduces on qwen
   (1.9 s vs 13.7 s). Contexts differ (KV bits, CTX, prompt style), so not same-window —
   but if even half the gap is the KV=8 prefill dequant path, that dwarfs every plan-02
   hypothesis. Worth a same-window KV=fp16-vs-8 serving A/B as a follow-up plan (note:
   GLM fp16 KV at CTX=32768 does not fit GTT; use KV=fp16 at smaller CTX or KV=12/16
   if supported).
4. First-touch/autotune dominates the first prefill after load (21–29 s warm vs 3.5 s
   steady state) — any micro-benchmark of prefill must discard the first pass.

## Reporting discipline (done)

- Bench rows: `benchmarks/results.csv` labels `tolist-base`, `tolist-fix` (+ discarded /
  re-sample labels listed above); artifacts under `benchmarks/20261024-*-prefill|latency/`.
- (No `bench_metrics.csv` twin: that mandate applies to `cpu-trial/bench_decode.py` CPU
  rows; the serving bench (`65_bench.sh`) records into `benchmarks/results.csv` only.)
- INDEX dispatch-table row 02 + dispatch log updated in `amd_zen_sogs`.
