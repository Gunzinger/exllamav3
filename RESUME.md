# RESUME — opt/multirow-tiles (plan 03: multi-row GEMM tiles for the ROCm/RDNA kernel map)

Worktree: `/home/daniel/development/gunzi/local-llm/opt-multirow` (branch `opt/multirow-tiles`,
forked from `cpu-only` @ bc731a5e). Plan authority:
`/home/daniel/development/amd_zen_sogs/inference-analysis/plans/03-rdna-multirow-tiles.md`
(= the dispatch's "03-rdma…", filename differs). Report goes to
`/home/daniel/development/gunzi/local-llm/PLAN-opt-multirow.md` (local-llm ROOT).

## Environment (every shell)

```bash
source /home/daniel/development/gunzi/local-llm/env-rocm10.sh
cd /home/daniel/development/gunzi/local-llm/opt-multirow
export MAX_JOBS=16 TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD
PY=/home/daniel/development/gunzi/local-llm/.venv/bin/python
```

## AUDIT FINDINGS (task 1) — complete, verified in source 2026-10-24

### (a) WMMA vs emulation on gfx1150 → native WMMA is active (no pivot needed)

- Gate: `rocm/rdna_wmma_emu.cuh:8-9` — `EXL3_WMMA_EMULATED` is defined only when
  `__HIP_DEVICE_COMPILE__ && !__GFX11__ && !__GFX12__`. gfx1150 is a gfx11 target (clang
  defines `__GFX11__`), so the emulator is compiled out and
  `rocm/rdna_wmma.cuh` uses the native `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`.
- The RDNA GEMM inner (`rocm/quant/exl3_gemm_inner_rdna.cuh:62`) includes `../rdna_wmma.cuh`,
  so the EXL3 GEMM mainloop rides native WMMA. Runtime confirmation still to do post-build:
  (i) preprocessor probe with the exact hipcc flags, (ii) EXL3_DEBUG_AUTOTUNE + µbench
  throughput sanity (~native TFLOP/s, not ~1/10 of it), (iii) optional ISA grep of the built
  code object for v_wmma encoding.
- Structure of the RDNA inner (read in full): 3-stage software pipeline, **synchronous**
  global→LDS copies (no cp.async on RDNA — header note 4), B dequant → LDS transpose →
  WMMA `load_matrix_b`, split-K via global locks, `EXL3_GRID_SYNC()` device barrier in the
  wrapper (`quant/exl3_gemm_kernel.cuh`) after the input Hadamard and between M-tile chunks.

### (b) LDS per workgroup, current shapes (derived with the file's own accounting,
`exl3_gemm_smem_bytes()` ROCm branch in `quant/exl3_kernel_map.cuh:171-200`; the kernel's
static_assert asserts host/device agree, so this IS the launch request modulo DevCtx cap).

Formula (bytes): `SH_STAGES*(2*sh_a + 2*sh_b) + 2*sh_b_dq + 4*sh_c` where
`sh_a = TILESIZE_M*(TILESIZE_K+8)` halfs, `sh_b = (K/16)*(N/16)*(16*bits)` uint16,
`sh_b_dq = 8*(K/16)*16*18` halfs, `sh_c = max(0, N*M)` floats (shmem_out_had=true bound;
the GEMM passes true, MoE false).

| shape (M,K,N,stages) | b=2 | b=4 | b=8 | note |
|---|---|---|---|---|
| 1 (16,16,128,4)  | 13.3 kB | 14.3 kB | 18.4 kB | not selected by scorer |
| 2 (16,16,256,3)  | 25.8 kB | 28.8 kB | 41.0 kB | |
| 3 (16,16,384,3)  | 35.3 kB | 39.8 kB | 53.4 kB | |
| 4 (16,16,512,3)  | 44.8 kB | 50.8 kB | 62.8 kB | b=8 within 11 kB of the 64 kB cap — why stages=3 |
| **5R (32,16,128,4)** | **28.5 kB** | **30.5 kB** | **35.3 kB** | proposed; fits at all bitrates, no __GFX11__ 128 kB LDS needed |
| (48,16,128,4)    | 42.8 kB | 45.8 kB | 52.3 kB | 6R candidate, fits too |
| (64,16,128,4)    | 57.0 kB | 61.0 kB | 69.3 kB | b=8 overflows 64 kB → S=3 for 8R, or gate deeper LDS |

Method note: exact per-shape bytes are also computable at runtime via
`ext.exl3_gemm_shape_smem(idx, K, half_k)` (not exposed to python; the constexpr in the
header is the single source). Occupancy: the launch path clamps RDNA grids to 1 block/WGP
(EXL3_RDNA_OCC_MAX default 1, `quant/exl3_gemm.cu:345-352`) because the kernel grid-syncs;
leave unset. `EXL3_DEBUG_AUTOTUNE=1` prints occ_api per candidate for the runtime numbers.

### Key structural discovery (changes the shape of the work)

`rocm/quant/exl3_gemm_inner_rdna.cuh` is **already M-general in code**: every loop
(`load_a` predication, `frag_a[FRAG_STAGES][TILEBLOCKS_M]`, `frag_c[TILEBLOCKS_M][…]`,
`matmul`, `read/write_sum_gl`, `write_sum_tile_sh`, `output_had_sh_gl`) iterates
TILEBLOCKS_M = TILESIZE_M/16 and predicates on the runtime row count. The only M=16 guard
is `static_assert(TILESIZE_M == 16)` at line 137 + the comment "the 32/64-row MoE tiles are
NOT ported; the static_assert is the guard". So this plan is: lift the assert, add the
shape(s) to the table, verify, measure — NOT a new kernel. The B-fragment decode amortizes
across TILEBLOCKS_M rows inside `load_frags` (frag_b shared over m) — exactly the CUDA
shapes 5-7 mechanism.

### Selection logic (how shapes are really picked)

- Runtime default = **timing autotuner** (`CoopKernelAutotuner`, `quant/exl3_gemm.cu:310-377`):
  candidates = every `exl3_gemm_shape_compat` shape, timed, cached by
  `gemm_autotune_hash(MAX(size_m,2), size_k, size_n, K, …)` (+ RDNA salt). New table shapes
  become candidates automatically. The M-rule in `exl3_gemm_shape_compat`
  (`exl3_kernel_map.cu:189`: wide tile skipped when `size_m <= tilesize_m/2`) keeps M≤16 on
  16-row shapes.
- On CUDA, `select_gemm_shape` (static scorer) NEVER returns 5-7 either — the autotuner is
  the selector there too; the scorer only fires in the `force_num_sms` path.
- `ext.exl3_gemm` binding takes `force_shape_idx` positionally (arg 7) — the µbench and the
  shape tests can pin shapes without any env machinery.
- E2E A/B override: adding `EXL3_GEMM_SHAPE` (index) / `EXL3_GEMM_SHAPE_M` (tile M) env,
  honored as force_shape_idx in `exl3_gemm_gr` (skips GEMV + autotuner; TORCH_CHECKed
  divisibility/smem; grid = num_sms ≤ 1 block/WGP so grid-sync co-residency holds).

### CRITICAL build hazard (differs from plans 02/04/05/06 — all python-side)

The .venv editable install puts the MAIN checkout root on sys.path, so
`import exllamav3_ext` resolves to the MAIN tree's prebuilt HIP .so and
`exllamav3/ext.py` prefers it (`is_precompiled_extension_available`). For kernel-source
changes this silently benches the wrong build. Fix in this worktree:
`tools/jit_build_wt.py` scrubs the main-tree sys.path entries before importing → JIT-builds
this worktree's sources into `$TORCH_EXTENSIONS_DIR`; afterwards put the output directory
FIRST on PYTHONPATH. Server verification: grep `/proc/<pid>/maps` for the .so path.

## Status / next steps

- [x] Audit (a) WMMA native on gfx1150 — no pivot. (b) LDS table above.
- [x] Worktree + isolated JIT build path (tools/jit_build_wt.py)
- [ ] Stock build green (in progress — long, ccache warm)
- [ ] Baseline test suite (tests/kernels at minimum) green with stock build
- [ ] Shape 5R (32,16,128,4,3) landed behind stock LDS accounting; static_assert lifted
- [ ] Selection: autotuner candidacy (automatic) + env override + scorer M-bonus
- [ ] Correctness: extended shape tests, M ∈ {16,17,31,32,33,64,100,144} × N ∈ {128,2560,4096}
- [ ] µbench go/no-go ≥1.5× at M ∈ [32,144] (real 2.05bpw tensor)
- [ ] E2E (only if µbench passes) + greedy parity
- [ ] Report PLAN-opt-multirow.md (local-llm root), INDEX row, final commit

## Landmines

- NEVER let a bench import the main tree's .so (see hazard above); check
  `ext.exllamav3_ext.__file__` startswith the worktree/.torch_ext.
- The autotuner caches by (size_m, k, n, K, …) — env overrides must be set before the
  first launch of a process; one arm per process.
- `wt_run_tests.py` requires TORCH_EXTENSIONS_DIR inside the worktree (it asserts).
- ollama runs in background: ±5% e2e noise; serialized GPU benches, same-window A/B,
  discard first run after each server start.
