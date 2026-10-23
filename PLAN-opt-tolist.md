# PLAN-opt-tolist — MoE expert-count drain removal (opt02, branch `opt/tolist-counts`)

Skeleton; fill tables after the deferred parity gate + benches (commands in RESUME.md).

## Implementation state

| Piece | Status | Notes |
|---|---|---|
| Env gate `EXL3_MOE_DEVICE_COUNT` (cuda_flags.py, default ON) | landed | own EOF block, stdlib-only |
| Pinned-ring count readback (batch path only, device-gated) | landed | `_PinnedCountRing.read_counts` |
| Pinned-ring slot-table H2D (FUSED_DET tables) | landed | `_PinnedCountRing.upload_tables` |
| Worst-case-grid fused launch `run_fused(-1)` | landed (non-mtile branch) | M-tile tiering untouched |
| True sync removal (device-side tier plan / Path B scan) | NOT started | see §Next steps |
| CPU/MoeCpuHost branch | untouched by design | existing offload-mode gates |
| bsz≤8 decode / CUDA-graph branch (`run_bszN`) | untouched by design | — |
| Unit tests (model-free) | landed, 5/5 PASS | `tests/test_moe_device_counts.py` |

## Parity gate (fill after run)

| Model | Knob | Prompts | Divergence | Verdict |
|---|---|---|---|---|
| qwen3.8-flash 2.05bpw | 0 vs 1 | qbench_prompts.json (greedy, 128–512 tok) | — | — |
| GLM-5.3-Flash-EXL3-Yamz mix | 0 vs 1 | 4 natural prompts | — | — |

## Benches (fill after run; every row needs its bench_metrics.csv twin)

| LABEL | MODE | tokens | tok/s old | tok/s new | Δ% | GiB/token Δ | J/token Δ |
|---|---|---|---|---|---|---|---|
| tolist-fix-old / tolist-fix | prefill | 512 | — | — | — | — | — |
| tolist-fix-old / tolist-fix | prefill | 2048 | — | — | — | — | — |
| tolist-fix-old / tolist-fix | prefill | 8192 | — | — | — | — | — |
| tolist-fix-old / tolist-fix | latency | decode | — | — | expect ≤2% | — | — |

Acceptance (from plan 02): zero parity diffs; ≥3% pp2048 on qwen 2.05; no decode regression >2%;
GLM pp2048 ≥ +3%.

## Next steps (Path B sketch, after this lands)

1. The sync survives because three host consumers need exact counts: `plan_groups`,
   host table layout (heavy experts present), per-expert DQ slicing. Device-side plan needs:
   exclusive scan over `expert_count` (check `histogram.cu` in ext sources when available —
   not present in this workspace), kernel-consumed offsets, device-built tables for the mixed
   fused+heavy case, and a count-free heavy tier (per-expert graph shapes are host-sized today).
2. Alternative cheap step: speculative all-fused launch (`run_fused(-1)` + device tables sized
   A = tokens×top_k) with a per-chunk accumulated `any_heavy` device flag read once at chunk end;
   on mispredict, re-run the chunk on the old path. Correct because deterministic; needs a
   recompute hook in the chunk loop (model-level).
3. Measure the launch-grid delta (`-1` vs exact hint) on an mtile-capable build before merging
   the worst-case-grid branch there.

## Reporting discipline

After benches: fill tables above, append `LABEL=tolist-fix` rows to `benchmarks/results.csv`
(plan) / `bench_history.csv` + `bench_metrics.csv` (workspace), mark done in
`plans/INDEX.md` upstream of this plan file.
