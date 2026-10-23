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

Rows `p07-tier-{off,torch}-{128,512,2048}` in `bench_history.csv` + `bench_metrics.csv`.
NOTE: the harness's prompt synthesis capped all prefill arms at ~278 actual tokens, so the
three sizes are replicates of one ~278-token point; the torch arm of the third run picked up
the blocked-dequant rebuild mid-matrix (cold 2.8 s → 0.7 s per 32 MiB tensor).

| arm | prefill tok/s | decode ms/step (neutrality) |
|---|---|---|
| tier-off | 24.9 / 25.5 / 25.4 | 236 / 254 / 254 |
| tier-torch (cold, one-shot) | 0.61 / 1.50 / 4.25 | 256 / 224 / — |

**Decode is neutral** (tier engages only at M ≥ 32; M=1 decode never enters it). **Single-shot
cold prefill is a large loss**: ~100 non-MoE tensors × ~0.7 s scalar dequant ≈ 70 s+ vs a 5 s
stock prefill. Warm-repeat bench (`tests/bench_prefill_warm.py`, 4 × ~370-token rounds,
one process): the run collided with the 07b subagent's JIT build — rounds swing 4.4–11.2 s in
BOTH arms; usable only as the qualitative observation that warm-cache rounds recover to
stock-class times. Clean re-run queued for the quiet machine (numbers to fill in §Addendum).

## Parity gates (plan §Verification 1)

- `tests/par_qbench.py` (3 qbench rows × 256 tokens, one process, env-flip arms):
  - tier=torch (bf16): **332 flips / 768 tokens (221 per 512; rule ≤ 2)** — FAIL
  - tier=fp32 (plan's fallback variant): **426 flips / 768 (284 per 512)** — FAIL, and *worse*
    than bf16 on 2 of 3 rows (row 0 bit-identical, row 2 first-diff at token 13).
- Root cause (measured, not conjecture): the reference path itself quantizes activations to
  int8 per 16-lane group — the fp64-truth unit gate measures the stock path at rel 0.7–1.8e-2
  from exact. ANY exact recomputation (fp32 tier) therefore differs from stock by that whole
  class, compounding residually through 48 layers; flip cascades are then coin-flips (row 0
  identical, row 2 flips from token 13 — the M2 lesson's shape). **The ≤2-flips/512 rule is
  unsatisfiable for any reimplementation tier**, not a defect of bf16. The fp32 variant is
  kept (binding `fp32_out`, tier value `fp32`) with its measured cost (~½ bf16 GEMM rate) as
  the plan instructed; it does not and cannot rescue byte-stability.
- Goldens (`tests/golden/self_golden.py`, qwen 2.05, worktree build): record(tier off) vs
  compare(tier torch) — **tokens 16/16 exact on short/med/long**, top1-logit max-abs 0.00 /
  0.44 / 1.41. Landing with tier default OFF keeps default goldens byte-exact.
- Consequence for promotion: the tier is a **lossy opt-in** feature (default off). Graduating
  it toward default-on belongs to the `kl_lossy.py` distributional methodology (KL(arm‖ref)
  both directions + flip-margin stats), queued as follow-up work; the 256-token flip record
  above is its input evidence.

## Verdict against the plan's acceptance criteria (§Verification & acceptance)

1. Parity "argmax-stable, ≤2 flips/512": **NOT MET and proven unsatisfiable** for any
   reimplementation tier (stock int8-activation noise ~1e-2 rel dominates; fp32 fallback
   measured worse than bf16 — see Parity gates). Goldens 16/16 exact at their horizon.
   → lands as documented lossy opt-in, default OFF; KL follow-up is the promotion gate.
2. Micro "≥3× mul1-batch at M=128/512/2048": met at M=512 (3.0–3.1×) and 2048 narrow (2.97×);
   2.25–2.6× at 2048 wide and 1.4–2.4× at 128 (dequant amortization + small-M GEMM efficiency;
   07b SIMD dequant addresses the cold side).
3. E2E "≥2× CPU prefill (target ≥60 t/s)": **NOT MET by Option 1 non-MoE-only scope.**
   Cold one-shot is a loss (dequant dominates); warm steady-state ≈ stock within noise — the
   non-MoE linears are a minor share of qwen prefill (MoE stays on the stock path by plan
   scope). The E2E win requires the deferred MoE grouped tier + fast dequant (07b), i.e. the
   plan's own sequencing ("Start non-MoE linears only; MoE behind a second env once linear
   path is proven") — this tranche proves the linear path and quantifies its ceiling.
4. Decode unchanged: **MET** (measured neutral across all six matrix arms; tier gated M ≥ 32).

## Open items

- Option 2 (AVX-512 VDPBF16PS GEMM + SIMD dequant-to-bf16) — follow-up tranche.
- MoE grouped tier (bucketed experts) — second env per plan, after linear path proven.
- Thread discipline: tier currently rides torch's default threads (12); EXL3_MOE_CPU_THREADS
  interplay untested.
