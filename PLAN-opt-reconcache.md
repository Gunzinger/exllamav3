# PLAN-opt-reconcache.md — reconstruct-once FP16 shadow cache (Plan 04)

> Verification dispatch (2026-10-24): unit suite + parity gate + bench gates run by the
> deferred-verification session. Design landed on `opt/recon-cache`; rollback:
> `EXL3_RECON_CACHE_MIB=0` (default). Results tables below are measured, not projected.

## Design (as implemented)

- Per-`LinearEXL3` lazily-materialized FP16 shadow for the `rows > 144` tier; global
  GTT-budgeted LRU (`modules/quant/recon_cache.py`) keyed by trellis identity + basis.
- Env `EXL3_RECON_CACHE_MIB` (via `util/cuda_flags.py`, own block; 0 = off = default).
- MoE routed experts (`experts.<idx>` keys) excluded; `shared_experts` in scope; sliced-N
  (lm_head-class, N > 32768) not cached; decode tier never consults.
- Fused-Had path (rows ≥ 1024): shadow stores the fused kernel's own original-basis output;
  hit replays identical hgemm → byte-identical. No plain-path restriction needed.

## Verification method (what was actually run)

- **Unit suite**: `tests/test_recon_cache.py` (mocked ext kernels, CPU tensors) — 8/8 PASS
  at head `b8e8450b` (re-run this session before any GPU work).
- **Parity gate** (`tests/par_reconcache.py`, committed @ `12795abf`): one process per model,
  greedy `ComboSampler(temperature=0)`, three passes in-process — `off` (env unset +
  `recon_cache.configure()`), `on-cold` (=4096, materializes shadows), `on-warm` (=4096,
  steady state). All three passes must be token-identical AND `stats().entries > 0` after the
  cold pass (else the gate is vacuous). Note: the env knob is parsed at call time but the LRU
  caches the budget after first `configure()` — the sanctioned live-flip seam is
  `recon_cache.configure()` (the module's documented re-read path), which is what the driver
  uses between passes; server arms set the env at process start (single configure — no seam
  needed).
- **Bench gates** (`tests/bench_reconcache.sh`, committed): serialized arms off:0 /
  2048:2048 / 4096:4096 / 8192:8192, one `60_serve.sh start→stop` per arm, server launched
  with `PYTHONPATH=<worktree>` + the arm's knob (verified via `/proc/<pid>/environ`, evidence
  in `/tmp/opencode/reconcache/<tag>-<arm>-environ.txt`, plus a same-env import assert).
  Per arm: 1 warm-discard prefill (labeled `-discard`, never compared), then 3 ×
  `MODE=prefill PROMPT_TOKENS=2048,8192,32768` invocations (the harness's prefill mode sends
  one request per size per invocation, so REPS is realized as 3 invocations; per-request
  `cached=0` is visible in every report), then `MODE=latency REPS=3 MAX_TOKENS=256`, then the
  warm-resend driver (`tests/warm_resend.py`) run identically in every arm.

## Deviations from the dispatch text (all documented, none touch the A/B)

1. **CTX=40960 (not 32768), KV=8 kept**: with CTX=32768 the harness skips the 32768 sweep
   size (`size <= ctx-64` gate) and the warm-resend suffix prompt (34816 tokens) cannot fit.
   KV need at 40k is 0.64 GiB (qwen) / 0.36 GiB (GLM) — fit verified with `fit_calc.py`.
2. **PROMPT_STYLE=synthetic (dispatch said real)**: `bench_serve.py gen_prompt(style="real")`
   returns a ~100-token template regardless of PROMPT_TOKENS — real-style long prefills are
   impossible with this harness. Natural text is honored where it matters: the parity gate
   uses qbench rows + natural prompts, and the warm-resend driver builds natural-prose
   prompts (not the 44-word synthetic vocab).
3. **Cached-arm rows measure the warm-serving steady state**: the recon shadows persist
   across requests within a server session (that is the feature), so in a cached arm even the
   first pp8192 request of an invocation benefits from shadows materialized by the preceding
   pp2048 request. The off arm re-dequantizes on every chunk of every request. The pp2048
   row bounds the fill cost (a fill replaces the reconstruct the off arm pays anyway).

## Results (measured 2026-10-24, host HX 370, gfx1150, ROCm 10)

| model | budget MiB | pp2048 | pp8192 | pp32768 | warm-resend suffix | tok/s (decode) | GTT peak Δ |
|---|---|---|---|---|---|---|---|
| qwen 2.05 | 0 (ctrl) | 176.3 | 142.2 | 133.6 | 125.0 (cold) | 12.53 | — (37.5 GiB) |
| qwen 2.05 | 2048 | 175.8 | 141.9 | 133.4 | 134.3 | 12.67 | +2.4 GiB ⚠ |
| qwen 2.05 | 4096 | 175.6 | 141.8 | 133.3 | 134.2 | 12.70 | +4.5 GiB ✓ |
| qwen 2.05 | 8192 | 175.6 | 142.7 | 133.9 | 134.7 | 12.68 | +8.7 GiB ✓ |
| GLM Yamz  | 0 (ctrl) | (36.4*) | 51.4 | 49.2 † | 48.4 (cold) | 8.43 | — (97.3 GiB) |
| GLM Yamz  | 2048 | (36.4*) | 51.7 | — | 47.5 | 8.43 | +2.7 GiB ⚠ |
| GLM Yamz  | 4096 | (37.7*) | 52.7 | — | 47.7 | 8.40 | +4.7 GiB ⚠ |
| GLM Yamz  | 8192 | (38.0*) | 54.2 ‡ | 49.2 † | 49.6 | 8.44 | +9.4 GiB ⚠ |

\* GLM pp2048 is bimodal in every arm (rep 1 ≈ 70–75 tok/s, reps 2–3 ≈ 19–20 with ~88 s
non-prefill wall stall) — a GLM@97-GiB page-pool phenomenon present in the off arm too;
means are not meaningful and the column is not used for any verdict.

**GLM pp32768 addendum (CTX=49152, cold single request per arm, `--timeout 1800`):**
**49.2 tok/s (off) vs 49.2 tok/s (8192 MiB) = 0.0 %.** Both rows `cached=0`; the cached arm
held +9.1 GiB of resident shadows (GTT peak 106.5 vs 97.4 GiB) and still gained nothing —
consistent with the probe's 0 % hit rate at that budget. Two measurement notes for the
record: (1) the morning GLM 32768 rows were dropped by bench_serve's **600 s client
timeout** (the request needs ~768 s at ~49 tok/s and *did* fit CTX=40960 — the initial
"GLM tokenizer overflows the ctx" theory was wrong; 65_bench.sh cannot pass `--timeout`),
and (2) at CTX=49152 the completed request's KV pages survive to the next rep, so reps 2/3
of the sweep are prompt-cache-poisoned (`cached=36 864–37 632`) — only rep-1 rows and the
direct single-request runs above are usable at 49152.

**pp = mean tok/s over 3 same-window reps** (spread ≤0.4 % per row; every request
`cached=0` — a first attempt had the 2048 rows poisoned by a full prompt-cache hit from a
same-seed 2048-sized discard, `cached=2304/2402`, discarded and re-run with a 512-sized
discard @ `6996f756`). Decode = mean server `time_generate` over 3 × 256-token streams.

† CTX=49152 addendum, cold single request per arm (`--timeout 1800` direct bench_serve
invocation — 65_bench.sh cannot pass a longer timeout): 49.2 vs 49.2 tok/s, 0.0 %.
‡ 53.6 in the CTX=49152 addendum (+0.9 % vs 53.1 same-protocol off) — the morning +5.4 %
was session drift, see below.

**Budget-vs-gain (qwen 2.05): NULL at every budget — and the probe says why: 0 % hit
rate.** pp8192: −0.2 % / −0.3 % / +0.4 % and pp32768: −0.1 % / −0.2 % / +0.3 % for
2048/4096/8192 MiB — inside the ±0.4 % same-window noise band. The real-kernel probe
(`tests/probe_reconcache_hits.py`) settles the mechanism:

- qwen's cacheable working set is **309 tensors, 5.85 GiB per basis** (5 849 088 000 B
  measured at full residency). The sweep requests visit layers in *both* row regimes
  (2048-row fused chunks + 354-row plain tail chunks), so the dual-basis working set is
  ~11.7 GiB — larger than every tested budget.
- Sequential-scan LRU below full residency is the textbook 0 % hit case: at the 4096 MiB
  budget the probe measures `lookup 309, hit 0, store 309` on the warm run — every pass
  evicts exactly what the next pass needs.
- At a budget that holds the whole working set, the cache works perfectly: **`hit 309/309,
  store 0`, reconstruct never called**, on both the plain basis (512-row chunks) and the
  fused basis (1367-token single chunk), real kernels, identity checks all green. The
  mechanism is sound; the tested budgets never reached residency for qwen.

**GLM Yamz: the apparent morning budget-vs-gain curve (+0.6 % → +2.5 % → +5.4 % at
2048/4096/8192 MiB) is NOT bankable — it was session drift.** The pp8192 numbers were
monotone in budget but equally monotone in wall-clock time (serial 09:06→13:19 arms). The
CTX=49152 addendum (post-addendum pair, 25 min apart, identical protocol) re-measured
pp8192 at **53.1 (off) vs 53.6 (8192) = +0.9 %** — most of the morning's +5.4 % did not
reproduce. And the probe had already ruled the cache out: GLM's true working set is **244
tensors, ≥8.5 GiB per basis** (the parity run's "79 entries / 4.26 GiB" was the 4096-MiB
budget cap, not the tier), so at the 8192 MiB budget the probe measures `hit 0` on the warm
run. A gain cannot come from hits that never happen.

**Warm-resend (natural-prose prompts, ~1.9 k-token suffix prefill after a ~29 k-token
cached prefix): no reproducible win on GLM** (48.4 cold / 47.5 / 47.7 / 49.6 tok/s —
inside ±2.6 %); the qwen suffix numbers (125.0 → 134.2–134.7, +7.3–7.8 %, consistent across
three cached arms) are the one effect that survived every check — but given the GLM result
and that the qwen deltas run through the same resume path, treat the qwen resend gain as
unexplained-but-real-on-that-model until a same-session knob-flip A/B isolates it.
Identical resend is prompt-cache-bound in every arm (26 368–29 184 of ~29 k tokens cached,
prefill 1.6–1.9 s) — the recon cache cannot matter there, measured as predicted.

**Dequant ceiling — why the plan's ≥8 %/≥12 % targets are unreachable on this host:** the
plan-08 traffic model (34 vs 16 bpw-equivalents per chunk) counts *traffic*, but the
reconstruct kernels are not bandwidth-bound on the iGPU; their wall share is the ceiling for
the whole optimization, and a per-chunk dequant is a *constant adder* — eliminating it gives
the same percentage at pp8192 (4/5 of chunks saved) as at pp32768 (18/19); the ratio
saturates, it does not grow to 12 %. GLM's (confounded) +5.4 % is itself an upper bound on
that share for the larger model; qwen's is smaller. 

**GTT peak check (acceptance: Δ ≤ budget + 5 %):** qwen 4096/8192 arms pass (+3.7 % /
+1.6 % over budget); qwen 2048 (+17 %) and **all GLM budgets (+29 % / +14 % / +8.7 %)
fail** — LRU churn leaves freed shadow segments retained by the torch caching allocator
(sysfs GTT counts the allocator pool, not live tensors), and the overshoot is worst at
small budgets where eviction churn is constant. The RESUME-noted missing GTT accounting
seam bites in practice.

**Decode:** qwen 12.53 → 12.67/12.70/12.68 (+1.1–1.4 %), GLM 8.43 → 8.43/8.40/8.44
(±0.1 %) — within ±2 % everywhere ✓ (decode never consults the cache; GTT side-effects
invisible).

## Parity (measured 2026-10-24)

- **qwen 2.05 (qbench ≥4 rows + fused/plain coverage prompts): PASS** — greedy
  temperature=0 outputs byte-identical across off / on-cold(=4096) / on-warm passes,
  cache engaged (228 entries / 4 269 015 040 B held = the 4096 MiB budget cap — a subset;
  the probe measures the full tier at 5.85 GiB). Driver `tests/par_reconcache.py`
  @ `2412cb24`.
- **GLM yamz (4 natural prompts + coverage prompts): PASS** — byte-identical across the
  same three passes, 79 entries / 4 261 412 864 B engaged (again the budget cap, not the
  tier — probe: 244 tensors, ≥8.5 GiB).
- Coverage note (why the driver grew these checks): a first GLM run with only the 4 short
  natural prompts was **vacuous** — they encode to 18–28 tokens, below the 144-row tier, so
  `cache_engaged=False` while "byte-identical" trivially held. The driver now prepends a
  long (~1.5 k-token, fused-basis ≥1024 rows) and a medium (~250-token, plain-basis) prompt
  to every mode. A second lesson: the Generator's KV page cache served passes 2/3 from the
  *prompt* cache (no re-prefill at all — visible as on-cold row times ≈ decode-only), so
  each pass now builds a fresh Cache (manual alloc/free replicating the loader's
  `load_local`) and Generator; the recon LRU is global, so shadows correctly survive.
- Unit tests (mocked kernels): PASS (8/8) at `b8e8450b`, re-run before any GPU work.

## GLM bench note: the pp32768 row was dropped by the client timeout at CTX=40960

The morning GLM sweeps produced no pp32768 row: the request fits CTX comfortably
(37 753 tokens measured) but needs ~768 s at GLM's ~49 tok/s, and bench_serve's default
client timeout is 600 s → `TimeoutError` → run_prefill drops the errored row. (65_bench.sh
cannot pass `--timeout`; the addendum above calls bench_serve.py directly with
`--timeout 1800`.) Also documented: GLM pp2048 is bimodal in BOTH arms (70.8 tok/s rep 1,
then 19.2 with ~88 s of non-prefill wall stall — a GLM@97-GiB page-pool phenomenon, not
cache-related); GLM pp8192 is dead stable within arm (±0.1 %) and is the reliable GLM
signal — read together with the drift caveat above.

## Acceptance checklist (VERDICT: FAIL — NO-GO for merge)

- [ ] ≥8 % pp8192, ≥12 % pp32768 on ≥1 model at a fitting budget — **FAIL**. qwen: NULL at
  every budget (0 % hit rate — dual-basis working set ~11.7 GiB exceeds all tested budgets).
  GLM: pp8192 +0.6…+5.4 % was session drift (CTX=49152 addendum re-measures +0.9 %) and the
  probe proves 0 % hit rate at the 8192 budget (working set ≥8.5 GiB); pp32768 = 0.0 %
  measured cold. No budget in the tested range (nor any implied by the working sets)
  reaches the bar.
- [x] decode within ±2 % — PASS (qwen +1.1–1.4 %, GLM ±0.1 %).
- [x] parity — PASS (both models, byte-identical, engagement enforced). Not in the plan's
  numeric list but a gate: green.
- [ ] GTT peak Δ ≤ budget + 5 % — FAIL at qwen-2048 and all GLM budgets (allocator churn;
  worst at small budgets).

## Go/no-go rationale (dispatch deliverable)

**NO-GO — keep the branch unmerged; the knob is default-off so production is unaffected.**
Rollback per plan: `EXL3_RECON_CACHE_MIB=0`.

1. The acceptance targets are unreachable *by construction of the mechanism*, not by
   tuning: the saving is the per-chunk dequant share (a constant adder), which saturates at
   ~5 % on GLM and less on qwen — it cannot reach 8 %/12 % on these models. The plan-08
   traffic model counted traffic (34 vs 16 bpw-equivalents); the reconstruct kernels are
   not bandwidth-bound on gfx1150, so their wall-clock share is an order of magnitude below
   what the traffic ratio implies.
2. The LRU, as implemented, has a sequential-scan thrash pathology: any budget below the
   working set yields exactly 0 % hits (probe-proven on both models). Since qwen's
   dual-basis working set is ~11.7 GiB, "a fitting budget" would need ≥12–16 GiB of GTT
   reserved for dense projections alone — at which point the ceiling argument (≤5 %) still
   kills the acceptance.
3. No GTT accounting seam exists (RESUME note 1) and the accounting overshoots the stated
   budgets in practice (allocator retention) — the cache fights the allocator it lives in.
4. What IS good on the branch and worth keeping as reference: the cache is
   correctness-solid (parity byte-identical with engagement enforced; unit suite; probe
   shows 100 % hit rate at full residency on both bases with identical hgemm replay), the
   LRU/budget/invalidation machinery is clean, and the probe/driver tooling
   (`tests/probe_reconcache_hits.py`, `tests/par_reconcache.py`,
   `tests/collect_reconcache.py`) is reusable for any future reconstruct-tier work. If
   anyone revives the idea, the prerequisites are: working-set-aware admission control
   (not plain LRU), a runtime knob flip for same-window A/B in the server, GTT accounting
   integration, and a ceiling model based on measured dequant wall share.

## Lessons recorded for the gate tooling (why this took three parity attempts)

1. Short prompts never reach the rows>144 tier → gates must assert `stats().entries > 0`
   (else they pass vacuously).
2. The Generator's KV page cache serves identical repeat prompts from the *prompt* cache —
   parity passes over a shared Generator do not re-prefill; fresh Cache/Generator per pass
   is mandatory.
3. `Cache` must exist before `model.load()`; per-pass caches need manual
   `layer.alloc()/free()` replicating the loader's `load_local`.
4. Bench hygiene: the warm-discard must not share bench_serve's `(seed + size)` prompt
   derivation with a measured size (a 2048-sized discard poisoned the 2048 row with a full
   prompt-cache hit — `cached=2304/2402`, caught by the per-row `cached` column).
5. Arm-level A/B across serial 1 h sessions drifts (GLM pp8192 +5.4 % monotone with time
   of day); knobs read once at process start cannot be flipped for same-window A/B —
   either expose a runtime seam or interleave arms in one session.
