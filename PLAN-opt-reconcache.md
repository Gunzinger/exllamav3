# PLAN-opt-reconcache.md — reconstruct-once FP16 shadow cache (Plan 04)

> Skeleton per dispatch; design landed on `opt/recon-cache`, results tables to be filled by the
> deferred benches (see RESUME.md for exact commands). Rollback: `EXL3_RECON_CACHE_MIB=0` (default).

## Design (as implemented)

- Per-`LinearEXL3` lazily-materialized FP16 shadow for the `rows > 144` tier; global
  GTT-budgeted LRU (`modules/quant/recon_cache.py`) keyed by trellis identity + basis.
- Env `EXL3_RECON_CACHE_MIB` (via `util/cuda_flags.py`, own block; 0 = off = default).
- MoE routed experts (`experts.<idx>` keys) excluded; `shared_experts` in scope; sliced-N
  (lm_head-class, N > 32768) not cached; decode tier never consults.
- Fused-Had path (rows ≥ 1024): shadow stores the fused kernel's own original-basis output;
  hit replays identical hgemm → byte-identical. No plain-path restriction needed.

## Results (to fill)

| model | budget MiB | pp2048 | pp8192 | pp32768 | warm-resend pp8192 | tok/s (decode) | GTT peak Δ |
|---|---|---|---|---|---|---|---|
| qwen 2.05 | 0 (ctrl) | | | | | | |
| qwen 2.05 | 2048 | | | | | | |
| qwen 2.05 | 4096 | | | | | | |
| qwen 2.05 | 8192 | | | | | | |
| GLM Yamz  | 0 (ctrl) | | | | | | |
| GLM Yamz  | 4096 | | | | | | |

## Parity (to fill)

- qbench greedy + GLM prompts: token-identical cache off vs on: [ ]
- Unit tests (mocked kernels): PASS (8/8)

## Acceptance checklist

- [ ] ≥8 % pp8192, ≥12 % pp32768 on ≥1 model at a fitting budget
- [ ] decode within ±2 %
- [ ] GTT peak delta ≤ budget + 5 %
