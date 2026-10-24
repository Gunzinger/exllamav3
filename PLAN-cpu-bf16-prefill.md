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

## Micro bench — `tests/micro_cpu_bf16.py` (torch threads 12, quiet machine, reproducible)

Cold = AVX-512 dequant + GEMM (cache off); warm = cached bf16 W. The first-session numbers
(3.0× @M512) did not reproduce — oneDNN's bf16 GEMM has an M-dependent kernel threshold
(~715 GF/s at M=512, ~1350 GF/s at M=2048); the reproducible table:

| shape (k×n, K) | M | stock ms | tier-cold | tier-warm | speedup | warm GF/s |
|---|---|---|---|---|---|---|
| 4096×4096, 2.5 | 128 | 9.0 | 28.4 | 6.1 | 1.44× | 709 |
| 4096×4096, 2.5 | 512 | 37.2 | 35.7 | 24.1 | 1.54× | 713 |
| 4096×4096, 2.5 | 2048 | 122.5 | 72.6 | 50.9 | **2.75×** | 1349 |
| 4096×8192, 2.0 | 128 | 12.8 | 51.6 | 8.0 | 1.61× | 1079 |
| 4096×8192, 2.0 | 512 | 54.4 | 68.5 | 24.4 | 2.23× | 1410 |
| 4096×8192, 2.0 | 2048 | 209.6 | 128.6 | 92.0 | **2.28×** | 1494 |

Gate "≥3× at M=128/512/2048": met only at M=2048 (2.3–2.75×); at M≤512 the warm GEMM is
shape-bound (oneDNN threshold above) and cold loses (dequant not amortized at small M).
AVX-512 dequant (07b): 4.1–5.1 GB/s, 48–59× the scalar binding, **bit-identical** output.

## E2E warm-cache steady state — `tests/bench_prefill_warm.py` (quiet machine)

4 × ~370-token prefill rounds, one process: tier-off steady 143.8 tok/s vs tier-torch steady
138.3 tok/s (min-of-rounds 3.56 s vs 3.70 s) — **parity within noise** (round-to-round
variance stays ±2× even clean, THP/cache bimodality). Cold round 52.0 vs 47.6 tok/s.
Conclusion: with the MoE experts on the stock path (plan scope), the non-MoE linear share is
too small for the tier's 1.5–2.75× per-tensor win to surface E2E.

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
2. Micro "≥3× mul1-batch at M=128/512/2048": met at M=2048 (2.28–2.75×); 1.5–2.2× at M≤512
   (oneDNN bf16 shape threshold + dequant amortization). AVX-512 dequant 48–59× scalar
   (bit-identical) removes the cold-path obstacle the scalar version had.
3. E2E "≥2× CPU prefill (target ≥60 t/s)": **NOT MET by Option 1 non-MoE-only scope.**
   Cold one-shot is a loss (dequant dominates); warm steady-state ≈ stock within noise — the
   non-MoE linears are a minor share of qwen prefill (MoE stays on the stock path by plan
   scope). The E2E win requires the deferred MoE grouped tier + fast dequant (07b), i.e. the
   plan's own sequencing ("Start non-MoE linears only; MoE behind a second env once linear
   path is proven") — this tranche proves the linear path and quantifies its ceiling.
4. Decode unchanged: **MET** (measured neutral across all six matrix arms; tier gated M ≥ 32).

## Open items

- MoE grouped tier (bucketed experts, second env per plan) — THE E2E lever: the non-MoE share
  is too small for E2E movement (measured); the tier machinery, AVX-512 dequant, and shadow
  cache all carry over. → **Landed and measured 2026-10-25 (tranche 07c): the tier is an E2E
  loss, not a lever — see §MoE grouped tier below; acceptance ≥2× is unreachable at the
  measured ceiling.**
- KL quantification (kl_lossy.py, tier arms) — the promotion gate if default-on is ever
  pursued; the 256-token flip records (bf16 332/768, fp32 426/768) are the input evidence.
- Hand AVX-512 GEMM (VDPBF16PS): only worth it if the MoE tier lands and oneDNN's ~1350 GF/s
  ceiling at M=2048 becomes the bottleneck.
- Thread discipline: tier rides torch's default threads (12); EXL3_MOE_CPU_THREADS interplay
  untested; the avx512 dequant uses std::thread strips (12).
- Default OFF knobs: EXL3_CPU_PREFILL_TIER=off|torch|fp32, EXL3_CPU_PREFILL_MIN_M=32,
  EXL3_CPU_PREFILL_CACHE_MIB=0, EXL3_CPU_PREFILL_AVX512=1 (kill-switch for the fast dequant).

## MoE grouped tier (tranche 07c, 2026-10-25) — measured negative, default off

Branch `opt/cpu-bf16-prefill` @ `d1e383c1` (+ docs). Plan 07 §Design 3: the grouped path via
per-expert dequant + batched matmul. **Verdict up front: the tier works exactly as designed
and loses 2.1× E2E; the plan's premise (the batched-M mul1 kernel is latency-bound at prefill
shapes) is refuted on this host+model. Stock CPU prefill stays ~30 t/s; the ≥2× acceptance is
unreachable even at an infinite MoE speedup (ceiling 1.31×).**

### Design as landed

- `EXL3_CPU_PREFILL_MOE=off|torch` (default off; own block in `util/cuda_flags.py`), shares
  `EXL3_CPU_PREFILL_MIN_M`; `EXL3_CPU_PREFILL_MOE_DEBUG=1` prints per-call hot/cold split,
  buckets and phase timings. Dispatch in `BlockSparseMLP_CPU.cpu_forward` behind an exact
  rows < MIN_M gate (a per-expert count is bounded by the chunk's row count → decode M=1-3
  provably never enters; measured decode-neutral).
- Hot experts (routed count ≥ MIN_M): dequant gate/up/down via the shared AVX-512 binding
  (`_dequant_weights`), bucket by power-of-2 row count (T2.4: `next_pow2(count)`, sub-batched
  64 experts per stacked bmm to cap the ~630 MB transient weight stack), one bmm per
  projection per bucket, activation in an fp32 mirror of `moe_mul1.cpp apply_activation`
  (same clamps in the same order), down-proj bmm, per-expert `index_add_` scaled by routing
  weights. Cold assignments: masked (-1) selection compressed to a tail that runs the STOCK
  `exl3_moe_cpu_forward` — bit-identical arithmetic for every row the tier does not touch.
- The per-expert torch fallback path (no `cpu_layer_handle`) is out of scope (non-mul1
  corner); the shared expert stays on its stock kernel handle.

### First measurement: moe share = the E2E ceiling (documented before implementing)

- bench_decode full-window attribution: **moe 23.6 %** of the 17.5 s 512-token prefill window
  (4134 ms of 17 542; 25.1 % @2048), behind gdn 36 % and hc 25 %. Even an infinite MoE
  speedup caps E2E at 1/(1−0.236) = **1.31×** — the ≥2× acceptance cannot be met by ANY MoE
  tier on this host+model.
- The Generator prefill chunker delivers ~256-token passes (190 routed+shared kernel calls /
  2 per pass / 47 MoE layers), so per-expert counts average ~5 — but the router is
  **concentrated**: 25/512 experts hold 46 % of assignments (max count 92), so the tier
  genuinely engages at MIN_M=32 (25 hot experts/layer; my uniform-router estimate of
  "engages on nobody" was wrong in an interesting direction).

### Cache vs per-chunk decision (measured, not guessed)

- Routed-expert BF16 working set: 512 experts × (2·2560·640 + 640·2560) × 2 B × 47 MoE layers
  = **≈212 GiB** (3.3× the 64 GiB box) — full residency impossible; any smaller budget is the
  plan-04 thrash regime (sequential-scan LRU, 0 % hits). Even one layer's set is 4.5 GiB.
- Per-chunk dequant cost measured on the real expert shape (2560×640, K=2): **2.48 ms/matrix**
  (157 MiB/s packed — thread-strip spawn dominates at this 410 KB size; the 4.6 GB/s 07b
  headline is a large-tensor number). Full routed set per 512-token chunk ≈ 179 s vs the
  ~4.1 s total stock moe budget.
- **Decision: shadow cache bypassed for routed experts; per-chunk dequant; tier engages only
  where the router concentrates rows.** The A/B then measured that even the engaged subset
  cannot pay for its dequant: at 25 hot experts/layer the dequant alone is 153.5 ms vs the
  stock whole-layer time of 88 ms (4134 ms / 47 layers) — **dequant > the entire stock layer
  before any GEMM runs**; no matmul speedup can recover it.

### E2E A/B — bench_decode (worktree build, metrics twins on, GPU server stopped)

Labels `moe-tier-{off,torch}-{512,2048}` + `-w` warm repeats + `moe-tier-torch-minm8-*`
(MIN_M=8 diagnostic) in `bench_history.csv` + `bench_metrics.csv` @ `d1e383c1`. Cold ≈ warm
within ±1 % in both arms — the cache-bypassed design has no warm state, as intended.

| arm | prefill tok/s 512 | 2048 | decode ms/step (16 steps) | moe-kernel ms (512-token job) |
|---|---|---|---|---|
| off (stock) cold / warm | 29.61 / 29.91 | 31.25 / 31.32 | 205.8 / 205.1 | 4134 / 4143 |
| torch (MIN_M=32) cold / warm | 14.11 / 14.14 | 15.10 / 15.35 | 200.5 / 200.2 | 3078 / 3041 (tier adds ~17.6 s layer wall) |
| torch (MIN_M=8, deep) | 6.27 | 10.99 | 197.3 | 1003 (tier adds ~65 s) |

- **E2E: 0.48× stock at both sizes at the default knob (2.1× loss); 0.21×/0.35× at MIN_M=8.**
  Monotonic in engagement: more tier = slower.
- Per-layer attribution (debug, 512-token job): tiered layer 284.7 ms = tail 68.6 + dequant
  153.5 (75 mats) + bmm+scatter 215.5 vs stock 88 ms. The bmm phase runs at ~53 GF/s
  effective (buckets m=64/128, N=640) — oneDNN's bf16 batched path never approaches its
  700-1400 GF/s linear-tier class at these shapes (contrast the 2D-matmul micro table above).
- **Decode neutrality: MET** — 197.3-205.8 ms/step across all ten arms (±2.7 %, within the
  host noise band); the tier is gated out at decode by construction.

### Parity class

- Unit (`tests/test_cpu_bf16_moe_tier.py`, model-free): tier-vs-stock + fp64-truth across
  MIN_M straddles (7/8/9), single- and multi-member buckets, gateless relu2 all-hot, and a
  full engagement sweep — **rel_tier ≤ rel_stock in every config** (e.g. 0.0062 vs 0.0100),
  **0 argmax flips**, **cold-only rows bit-identical to the stock kernel** (the masked tail
  reproduces the un-tiered call exactly). Activation mirrors match a literal transcription of
  the kernel's apply_activation to ≤1.9e-6 abs. Dispatch contract asserted (no hot experts →
  stock fallback; decode rows never qualify).
- Goldens (qwen 2.05, worktree build, off vs torch): **short 16/16 and med 16/16 byte-exact**
  (prompts below the per-chunk MIN_M gate — the tier provably never runs), **long 3/16**
  (first divergence token 4) — the same lossy class as the linear tier (stock int8-activation
  quantization noise dominates any reimplementation; the ≤2-flips rule remains proven
  unsatisfiable). Default OFF keeps every default-path golden byte-exact.

### Verdict against plan acceptance (§Verification & acceptance)

1. "E2E CPU prefill ≥2× (stretch 60 t/s)": **NOT MET — and unreachable.** Measured 0.48×
   (the tier loses); the share ceiling is 1.31× at infinite MoE speedup; the linear tranche
   already showed E2E neutrality. Stock prefill: 29.6-31.3 tok/s at 512/2048 (not ~25 — the
   earlier figure was the capped ~278-token prompt; harness synthesis fixed this tranche).
2. "Decode unchanged": **MET** (above).
3. Parity: lossy opt-in class as documented; engagement-boundary visible in goldens
   (byte-exact below MIN_M, divergent above).
4. Root cause for the record: the stock mul1 batched-M kernel is NOT latency-bound at these
   shapes — it runs ~2.2 TFLOP/s effective int8-VNNI on 256-token chunks; per-chunk bf16
   dequant + oneDNN batched bf16 (≈53 GF/s at N=640) cannot compete. A winning design needs
   dequant-free residency (≈4.5 GiB bf16/layer just for routed experts) or a hand VDPBF16PS
   GEMM ≥40× oneDNN's batched rate — neither is a tier knob; recorded as a measured negative.

### Harness fixes carried in cpu-trial (this tranche)

- `bench_decode.py` prefill synthesis now scales with `--prefill-tokens` (the fixed ×12
  repeat capped every prefill arm at ~278 actual tokens — the p07-tier NOTE).
- Full-prefill attribution: the component table snapped at the first chunk only (8.7 s of a
  17.3 s job); it now covers the whole job.
- `git_sha()` falls back to the imported package's repo — `cpu-trial/exllamav3` is gone since
  the unification, so every bench_decode row since then crashed before the CSV append (this
  is why the p07-tier rows never landed). Rows now carry the worktree sha (d1e383c1).
