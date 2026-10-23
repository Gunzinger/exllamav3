# PLAN-cpu-bf16-prefill — CPU prefill BF16/AVX-512 tier for M ≥ 32 (Plan 07, Option 1)

Status: implementation + unit gates green; E2E + parity gates in flight (this file is the live
report; final location per plan: `cpu-trial/PLAN-cpu-bf16-prefill.md` at merge time).

Branch `opt/cpu-bf16-prefill` off `cpu-only` @bc731a5e. Scope this tranche: **Option 1 only**
(dequant-to-BF16 + `torch.matmul`), non-MoE linears only, `EXL3_CPU_PREFILL_TIER=off|torch`
(default **off** — opt-in until gates pass), `EXL3_CPU_PREFILL_MIN_M` (default 32),
`EXL3_CPU_PREFILL_CACHE_MIB` (default 0 = dequant per call; >0 = per-tensor BF16 shadow cache).
Option 2 (hand AVX-512 VDPBF16PS GEMM) and the MoE grouped tier are explicit follow-ups.

## Design as landed

- `exllamav3_ext/cpu/recon_bf16.{cpp,h}`: scalar dequant of one mul1 matrix to BF16
  (`exl3_cpu_dequant_bf16`): integer-exact state decode (same tables as `moe_mul1.cpp`),
  k-major tensor-core tile-slot permutation `g_tc_inv[k_off*16 + n_off]`, native `[k/16, n/16]`
  and band-8 `[n/8][k][n%8]` packed layouts, Hadamards FIRST and diagonal scales strictly AFTER
  (`W = (H @ what @ H) · (suh ⊗ svh) / 128` — the kernel-equivalent oracle order), fp32→BF16 RNE.
  AVX2-planar (band-2) rates are excluded (TORCH_CHECK + python-side `cpu_prefill_supported`).
- `modules/quant/exl3.py`: tier dispatch in the CPU arm of `LinearEXL3.forward` (M ≥ MIN_M,
  supported rate) + `cpu_prefill_torch_forward` (shadow cache, `torch.matmul` bf16 → fp32, bias);
  the batched decode path defers to per-linear forwards whenever the tier would engage.
- `util/cuda_flags.py`: the three knobs in their own commented block.

## Bugs found & fixed during verification (all in the interrupted first draft)

1. **Scales before the butterflies** — a diagonal does not commute with the Hadamard; the
   kernel-equivalent W applies suh/svh after both passes (oracle `cpu-trial/tests/test_linear_cpu.py`).
2. **Transposed tile-slot permutation** — the weight at (n_off, k_off) lives in slot
   `g_tc_inv[k_off*16 + n_off]` (k-major), not row-major.
3. Native trellis indexing is `[kt][nt]` (row walk of `moe_mul1.cpp`), `mat.swz` is a 0/1 band
   flag and the band group is re-derived from the rate (`exl3_moe_cpu_swizzle_group`).
4. Batched-path entry dereferenced `linears[0].inner` on bare `LinearEXL3` entries → `getattr`.

## Unit gates — `tests/test_cpu_bf16_prefill.py` (model-free) — PASS

- (a) dequant parity vs the fp32 oracle at K ∈ {1,2,4,8,2.5,3.5} × {native, band-8}: 1-ulp class
  with an explicit fp32 cross-order noise floor (measured: ≤5 of 131072 elements per config at
  oracle exact-cancellation points, all |ref| ≤ 3e-8, err ≤ 4e-9; every |ref| > 1e-6 element
  strictly ≤ 1.05 bf16-ulp).
- (b) tier dispatch: M < MIN_M and TIER=off hit the stock mul1 binding; M ≥ MIN_M takes the tier.
- (c) fp64-truth gate at M=64/128: new path rel ≤ 4.9e-3 (bf16 class, gated ≤ 1e-2) and closer
  to truth than the stock int8-quant GEMV (rel_old 0.7–1.8e-2); argmax flips 0/64, 2/128.

## Micro bench — `tests/micro_cpu_bf16.py` (torch threads 12, this host)

| shape (k×n, K) | M | stock ms | tier-cold | tier-warm | speedup | warm GF/s |
|---|---|---|---|---|---|---|
| 4096×4096, 2.5 | 128 | 9.0 | 6.1 | 6.6 | 1.36× | 650 |
| 4096×4096, 2.5 | 512 | 37.3 | 13.4 | 12.4 | **3.01×** | 1384 |
| 4096×4096, 2.5 | 2048 | 149.6 | 51.6 | 50.5 | **2.97×** | 1362 |
| 4096×8192, 2.0 | 128 | 18.3 | 8.8 | 7.7 | 2.40× | 1121 |
| 4096×8192, 2.0 | 512 | 75.0 | 24.5 | 24.2 | **3.09×** | 1418 |
| 4096×8192, 2.0 | 2048 | 215.5 | 102.2 | 96.0 | 2.25× | 1432 |

Gate "≥3× at M=128/512/2048": met at M=512 (both shapes) and ~met at 2048 narrow (2.97×);
2.25× at 2048 wide and 1.36× at 128 narrow miss the letter of the gate — dequant amortization
and matmul efficiency at small M bound Option 1 (~1.4 TF/s ≈ 37 % of Zen5 VDPBF16PS peak).
Dequant cost itself ≈ 1 ms per 32 MiB W (≈37 GB/s single-thread scalar) — Option 2's SIMD
dequant + GEMM is the follow-up lever (plan §Design Option 2, triggered: >20 % on the table).

## E2E prefill A/B — `bench_decode.py` (metrics twin on, GPU server stopped, AC)

Rows `p07-tier-{off,torch}-{128,512,2048}` in `bench_history.csv` + `bench_metrics.csv`:
(pending — table filled when the matrix completes)

## Parity gates

(pending) self_golden record(tier off) / compare(tier torch), qwen 2.05; 256-token greedy on 3
qbench rows × {off, torch} — flip counts vs the plan's argmax-stability rule (≤2 flips / 512 tok,
else fp32-GEMM fallback decision).

## Open items

- Option 2 (AVX-512 VDPBF16PS GEMM + SIMD dequant-to-bf16) — follow-up tranche.
- MoE grouped tier (bucketed experts) — second env per plan, after linear path proven.
- Thread discipline: tier currently rides torch's default threads (12); EXL3_MOE_CPU_THREADS
  interplay untested.
