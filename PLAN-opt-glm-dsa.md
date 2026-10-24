# PLAN-opt-glm-dsa — GLM-5.3-Flash prefill: lazy indexer planes + up-projection batching

Status: **VERIFICATION COMPLETE 2026-10-24 — parity PASS everywhere, acceptance FAIL
(serving delta ≈ 0 % both parts) → NO-GO for merge.** Branch `opt/glm-dsa-prefill`
(worktree `opt-glm-dsa/`) stays unmerged. Head @ `a6376d26`.

## Planes-reconstructability answer (summary, from the dev session)

- k-pool POOL plane: **derivable from cached idx keys** — `_update_pool_plane`
  (mla_attn.py:632-663 @ cpu-only) is a pure function of the cached packed
  `[k || gate]` plane rows + constant APE; complete pools are immutable. → deferred
  upkeep in dense mode + one-shot bulk rebuild on the first sparse-mode step is exact.
- Per-token IDX plane: **NOT derivable** — `_indexer_keys` consumes the transient
  block input `x` (idx_wk GEMM, gate F.linear); the append must run every chunk.
- Third consumer: the graph-captured decode block maintains pool planes itself
  (bc_mla.py) → arrears honored in `bc_mla_step` (seqlen=0 bulk rebuild before the
  captured step; fires on the first decode step after ANY deferred window).

## Changes (landed on the branch)

| part | gate | default | what |
|---|---|---|---|
| A | `EXL3_DSA_LAZY_PLANES` | 1 (on branch) | defer k-pool upkeep while context ≤ index_topk; bulk rebuild `[0, total//P)` on first sparse step / first captured decode step (prefill `_attend`, `bc_mla_step`, `copy_page` propagation); pre-allocated device-resident offset scalar replaces the pageable `torch.tensor([first_pool])` H2D |
| B1 | `EXL3_DSA_BATCH_UPPROJ` | 0 | kv_b halves cat'd on N dim → one GEMM/tile into merged buffer; `_mla_prefill_mha_kernel` takes runtime k/v row strides (gate-off = byte-identical strides) |
| B2 | `EXL3_DSA_FUSED_UPPROJ` | reserved | NOT implemented (scaffold only) |

## Verification 1 — parity (hard gate): PASS, zero divergence everywhere

Driver `tests/par_glm_dsa.py` (greedy temp=0, `ComboSampler(temperature=0)`, cache
created before `model.load()`, one Generator, sequential jobs, exact-length salted
prompts — distinct first pages per (regime,row) so the in-process prompt cache never
serves a repeat; prefill chunk spans logged per run as engagement evidence). Arms are
separate process runs (both knobs are import-time constants). Raw JSONs under
`tests/results/`; raw logs `/tmp/opencode/par_glm_*.log` (ephemeral).

GLM yamz (`models/GLM-5.3-Flash-EXL3-Yamz/mix`), `LAZY_PLANES=1` vs `=0`:

| regime | prefill chunking (observed, per _attend spans) | verdict |
|---|---|---|
| (a) 1024 ×4 rows, +512 gen | `[0->768]d` + `[768->1023]d` (last-token holdback), all dense | 4/4 identical |
| (b) 3000 ×4 rows | `[0->2048]d` (exactly at the window) → `[2048->2816]S` — **crossing inside the run** | 4/4 identical |
| (c) 8192 ×4 rows | `[0->2048]d, [2048->4096]S, [4096->6144]S, [6144->7936]S, [7936->8191]S` — 2048-token chunks (Generator `max_chunk_size=2048`, page-floor aligned; documented mechanism, not fed manually) | 4/4 identical |
| (d) 1024 + 1200 gen (extra: decode crossing) | dense prefill; decode crosses 2048 → `bc_mla_step` seqlen=0 bulk rebuild before the captured step | 1/1 identical |

Total 7344 tokens, 0 diffs — `VERDICT: PASS — zero divergence`.

`LAZY_PLANES=1 + BATCH_UPPROJ=1` vs baseline (yamz, regimes a/b/c × 2 rows): 6/6
identical (3072 tokens). Second arch variant `models/glm-5.3-flash-exl3/2.05bpw`,
full stack vs baseline, regime (a) × 4 rows: 4/4 identical (2048 tokens).

Synthetic harness `test_dsa_lazy_planes.py` re-run this session: PASS (pool planes
bit-identical, 1637 pools, prefill + decode crossings; MHA outputs max-abs 0.0).

## Verification 2 — launch/traffic evidence

`tests/launch_dsa.py` — ONE complete 2048-token dense prefill job per arm
(prefill chunk `[0->1792)` + tail `[1792->2047)` = 22 `_attend` forwards = 11 DSA
layers × 2; drain loops `gen.iterate()` until the job completes — a single drain
exits mid-job and leaks its tail into the next pass). Counters are direct
python-level wrappers: triton via `JITFunction.run` (kineto does not trace triton's
hipLaunchKernel*ExC path here), `torch.mm` count, `_update_pool_plane` count,
`_attend` spans. NOTE: torch-profiler kineto delivers **no device events in model
processes** on this stack (`cat=="kernel"` works in a bare matmul process, is empty
once exllamav3 loads; root cause unpursued; `rocprofv3` hangs per plan-06) — the
pool chain's ~9-10 torch-op launches/layer are therefore counted by source audit,
cross-checked by the visible triton appends.

| arm (2048-tok job, in-process A/B) | triton | `_mla_plane_update` | pool upkeeps | torch.mm | wall |
|---|---|---|---|---|---|
| base (LAZY=0, UP=0) | 756 | 44 (4/layer: idx+pool × chunk+tail) | 22 incr | 44 (2/tile) | 24.95 s |
| lazy (LAZY=1) | 745 (**−11**) | 33 | 11 (= the one-shot bc arrears bulk) | 44 | 24.90 s (−0.2 %) |
| upproj (LAZY=1, UP=1) | 745 | 33 | 11 | **22** (halved) | 23.86 s (−4.2 %) |

- Part A: plan's "~190 fewer launches" → measured −11 triton appends + audited
  ~−110 torch-op launches per dense chunk (10 ops × 11 layers: arange, gathers,
  softmax, mul, sum, to-half, fill_). Wall −0.2 %: ≈0.3-0.5 ms of submission
  against a ~25 s chunk (0.002 %).
- Part B: −22 `torch.mm` per job (2→1 per tile × 22 tiles) ✓ as planned; the
  in-process −4.2 % wall carries an ordering bias (base ran first/coldest in the
  process) and did **not** reproduce in the serving bench (below).
- upproj residency: GTT peak 98329 vs 98133 MiB (base) = +195 MiB ≈ the predicted
  176 MiB `_w_kv_b_cat` (16 MiB × 11) + allocator slack.

## Verification 3 — serving bench (GPU-exclusive, serialized, one server per arm)

`MODEL=glm5.3-flash-yamz MODEL_REV=mix CTX=65536 KV=8`, PYTHONPATH → worktree
(propagates through `nohup env` in 60_serve.sh; verified via
`/proc/<pid>/environ` — sys.path puts PYTHONPATH ahead of the editable .pth, so the
server imported the worktree package). Protocol: discard first bench per server
start (mandatory GLM cold-start rule; at off-measurement size 4096), then 3 reps of
`MODE=prefill PROMPT_TOKENS=512,2048,8192,32768`, then `MODE=latency REPS=3`.
**PROMPT_STYLE=real was NOT usable**: bench_serve's real prompts are fixed 31-43
words (~80 tokens) and cannot hit the sweep sizes — synthetic (the plan's own
command) governs; noted as a dispatch deviation. REPS does not apply to prefill
mode (1 request/size/invocation) → reps are repeated invocations with **distinct
seeds per rep** (SEED passthrough added to 65_bench.sh; first lazy-seed repeat was
served by the server's in-process prompt cache at 3502 tok/s — that row discarded,
plan-04/06 lesson re-learned at the invocation level). TIMEOUT passthrough added
(pp32768 = ~770 s > the 600 s default). Rows: `benchmarks/results.csv`, labels
`dsa-base`, `dsa-lazy`, `dsa-upproj`, `dsa-{base,lazy}-drift`.

Median tok/s (3 reps per size, one server window per arm):

| arm | pp512 | pp2048 | pp8192 | pp32768 | decode (latency) |
|---|---|---|---|---|---|
| dsa-base | 82.2 | 72.7 | 53.2 | 49.1 | 8.41 tok/s |
| dsa-lazy | 80.4 | 71.4 | 52.2 | 48.5 | 8.44 |
| dsa-upproj | 81.8 | 71.6 | 52.5 | 48.5 | 8.43 |

The naive arm-vs-arm deficit (−1..−2 %, one-signed) is **intra-session drift**: a
paired late-window A/B (fresh servers, same protocol, same fresh seed 5234, ~25 min
apart) gives

| size | base-drift | lazy-drift | Δ |
|---|---|---|---|
| pp512 | 69.9 | 74.7 | +6.9 % (noisiest size, single rep) |
| pp2048 | 72.0 | 72.5 | +0.7 % |
| pp8192 | 52.8 | 52.8 | 0.0 % |
| pp32768 | 48.7 | 49.0 | +0.6 % |

→ true serving deltas are ≈ 0 % (±1 %) for both parts; decode flat (±0.4 %).

## Acceptance verdict

| criterion | target | measured | verdict |
|---|---|---|---|
| Parity, 3 regimes + arch variant | zero divergence | 0 diffs in 12 464 greedy tokens across 5 arm-pairs | **PASS** |
| Part A dense regime | ≥ 10 % pp512/pp2048 | ≈ 0 % (same-window +0.7 % pp2048) | **FAIL** |
| Long-ctx regression | no > 3 % at pp8192/32768 | 0.0 % / +0.6 % (same-window) | PASS |
| Part B alone | ≥ 5 % dense regime | ≈ 0 % serving (in-process −4.2 % carries ordering bias) | **FAIL** |
| Decode neutrality | flat | 8.41 / 8.44 / 8.43 tok/s | PASS |

## Per-part attribution (why the premises don't convert)

- **Part A**: the deferred chain is real (~121 launches/chunk: ~110 tiny torch ops
  + 11 triton appends) but worth ~0.3-0.5 ms of submission against 24-32 s chunks —
  the kernels are µs-scale and fully hidden behind the quant-GEMM stream. This is
  plan-02's ceiling argument verbatim: the launch stream is never starved on this
  host, so launch-count reductions have no time value at serving level.
- **Part B**: the traffic model over-counted — the split k/v buffers write the same
  bytes as the merged one; the only saved read is the gathered latent tile
  (1.8 MiB/tile/layer), and each half-GEMM (512→8192 per tile) was already wide
  enough for rocBLAS. Launch halving (44→22) is real; time effect ≈ 0.

## Go/no-go

**NO-GO for merge** on perf grounds (acceptance FAIL; branch stays unmerged).
Part A is bit-exact with zero delta and a live decode path (8.44 tok/s) — harmless
if a merge ever happens for other reasons (e.g. the pageable-H2D removal or a
launch-bound serving stack), but it must not land carrying a perf claim.
`EXL3_DSA_BATCH_UPPROJ` stays opt-in and off. Rollback: env gates off; branch
unmerged. Bench-protocol by-products kept regardless: `TIMEOUT` + `SEED` knobs in
`scripts/65_bench.sh` (documented above), `tests/par_glm_dsa.py` +
`tests/launch_dsa.py` as reusable DSA-parity/launch-evidence drivers.
