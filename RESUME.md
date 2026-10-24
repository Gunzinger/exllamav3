# RESUME — opt/multirow-tiles (plan 03: multi-row GEMM tiles for the ROCm/RDNA kernel map)

**STATUS: COMPLETE — GO (see `/home/daniel/development/gunzi/local-llm/PLAN-opt-multirow.md`
for the full report).** Branch `opt/multirow-tiles`, all work committed. No server left running;
main tree untouched.

Worktree: `/home/daniel/development/gunzi/local-llm/opt-multirow` (branch `opt/multirow-tiles`,
forked from `cpu-only` @ bc731a5e). Plan authority:
`/home/daniel/development/amd_zen_sogs/inference-analysis/plans/03-rdna-multirow-tiles.md`.

## Audit findings (task 1) — final

- (a) **Native WMMA is active on gfx1150** — device probe with the build's exact flags:
  `__GFX11__` defined, `EXL3_WMMA_EMULATED` gate false (`rocm/rdna_wmma_emu.cuh:8`). No pivot.
- (b) LDS table (64 kB budget, `exl3_gemm_smem_bytes` ROCm branch): stock 1-4 = 13.0-62.8 kB;
  new 5 (32×16×128 s4) = 34.5 kB @ b=8, 6 (48×16×128 s4) = 45.5 kB, 7 (32×16×256 s3) = 53.0 kB —
  all bitrates fit; the gfx1150 128 kB/WGP gate was never needed. multiProcessorCount = 8 WGPs;
  grids ≤ 8 blocks = 1/WGP → `EXL3_GRID_SYNC` safe (cf15c60 machinery inherited, OCC_MAX unset).
- **The RDNA inner was already M-general in code** (all loops over TILEBLOCKS_M, partial tiles
  predicated); only `static_assert(TILESIZE_M == 16)` + a stale comment guarded it. Landed
  change = shape table + guard {16,32,48} + selection + tests — not a new kernel.
- Selection: the **timing autotuner** is the real selector on both backends (CUDA's static
  scorer never returns its shapes 5-7 either); new shapes become candidates automatically.
  `COOP_AUTOTUNE_VERSION` bumped 4→5: tuned records are candidate-set specific, and the stale
  `~/.cache/exllamav3/autotune` disk cache made the tuner pick shape 1 over 6 at m=100 (2×
  mispick) until resalted. Overrides are mixed into the autotune hashes (disjoint arm records).

## Landed

- Shapes 5/6/7 for USE_ROCM (`quant/exl3_kernel_map.cuh`); guard lift
  (`rocm/quant/exl3_gemm_inner_rdna.cuh`); `EXL3_GEMM_SHAPE` (force index) / `EXL3_GEMM_SHAPE_M`
  (row-tile class filter over autotuner candidates, 16 = exact old table) with empty-set
  fallback (`quant/exl3_gemm.cu`, also mgemm + smem lowering); scorer M-term
  (`quant/exl3_kernel_map.cu`).
- Tests `tests/kernels/gemm/test_exl3_gemm_multirow.py`: 15/15 green (band M×N×codebooks vs
  fp64 + real 2.05 bpw tensors). Full kernels+graph suite: **4953 passed / 0 failed**.
- Harnesses: `tools/bench_multirow.py`, `tests/par_multirow.py`, `tools/jit_build_wt.py`.

## Results (details + tables in PLAN-opt-multirow.md)

- µbench (forced shapes, real tensors): N=640 family (old table tile-starved) **1.63-2.39×**
  at M ∈ [32,144]; wide-N family +19-31 % (shape 7). Autotuner harvests per (m,k,n) → no
  regression anywhere.
- e2E qwen 2.05: pp512/2048 ±5 % noise (**zero band exposure**: chunk 2048, reconstruct >144 —
  the plan's pp512 premise mis-read the chunking); tail-chunk prompts ±0.5 %; **fully in-band
  prompt (131 tok) +17.1 %**; decode neutral (+1.5 %).
- Parity re-classed with evidence: same config = byte-identical (determinism); **two old-table
  shape-1 configs with different grids diverge 4/4 rows** — greedy output was never stable
  across grid changes (split-K fp32 regrouping); numerics class proven vs fp64. Zero-divergence
  across shape tables is not an achievable invariant for ANY shape change.

## Landmines (for any follow-up)

- Kernel-source worktrees MUST use `tools/jit_build_wt.py` + PYTHONPATH-first ext dir; the
  editable install otherwise silently imports the MAIN tree's prebuilt `.so` (verified via
  `/proc/<pid>/maps` for servers).
- Env overrides are read once per process → one arm per process.
- `EXLLAMAV3_TUNE_CACHE` / version bump: any future shape-table edit must bump
  `COOP_AUTOTUNE_VERSION` again.
- 65_bench: PROMPT_STYLE=real inflates prompt length (96→167 via chat template) — synthetic
  style + small targets to hit the band; repeated seed+size across runs = server prompt-cache
  hits (warmup runs poison the next measured run at the same sizes).
