# RESUME — Plan 04, reconstruct-once FP16 shadow cache (`opt/recon-cache`)

## Status: code complete, unit-tested (mocked), parity + benches DEFERRED (no network / no model
load / no GPU server allowed in this window; host shutdown imminent).

Branch `opt/recon-cache` (off `cpu-only` @ bc731a5e), python-only changes — no extension rebuild
needed or performed.

## What landed

1. **`exllamav3/modules/quant/recon_cache.py`** (new) — global GTT-budgeted LRU manager:
   - keyed by `(device_index, trellis.data_ptr(), basis)`; entry holds a strong ref to the
     trellis (key can't be recycled while cached; shadow validity tied to its source);
   - `basis`: 0 = plain rotated-basis dequant (`_reconstruct_inner`), 1 = fused original-basis
     dequant (`reconstruct_had_slice`, full `svh`, `n_start=0`). A layer active in both row
     regimes may hold both; budget covers the total;
   - opt-in budget `EXL3_RECON_CACHE_MIB` (0 = off, default); LRU eviction on store and on
     budget shrink; declines to cache a single tensor larger than the whole budget;
   - routed-expert exclusion: keys matching `experts.<idx>` (BlockSparseMLP routed experts).
     `shared_experts.*` stays cacheable (dense projection, in scope per plan Problem §);
   - `total_bytes` / `stats()` accounting accessors; `clear()` / `drop_linear()` invalidation.
2. **`exllamav3/modules/quant/exl3.py`** — `reconstruct_hgemm` consults the shadow only on
   `rows > AUTO_RECONSTRUCT_THRESHOLD` (144) **and** only on the single-shot N path
   (`out_features <= MAX_RECONSTRUCT_SLICE_N` = 32768). The sliced loop (lm_head-class shapes,
   N > 32768) keeps per-call reconstruct — shadow column slices would need per-call contiguous
   copies, and those shapes are exactly the exclusion the plan suggests. A hit replays the
   identical hgemm (+ surrounding `had_r_128`) sequence on identical contents → byte-identical
   outputs. `LinearEXL3.unload()` drops the layer's shadows.
3. **`exllamav3/util/cuda_flags.py`** — self-contained clearly-commented block at the end of the
   file: `RECON_CACHE_MIB_ENV` + `recon_cache_budget_bytes()` (read at call time; garbage/negative
   → 0). Safe to reconcile with the concurrent `EXL3_MOE_DEVICE_COUNT` branch: the block is
   append-only, no existing lines touched.
4. **`exllamav3/model/model.py`** — `Exl3Model.unload()` calls `recon_cache.clear()` (belt and
   braces over per-module unload).
5. **`tests/test_recon_cache.py`** — 8 unit tests, all passing.

## Verified (this window)

- Import recipe: `PYTHONPATH=<worktree> .venv python -c "import exllamav3"` resolves INTO the
  worktree (no symlink needed; no JIT rebuild triggered — plain import, <60 s).
- Unit tests, one-liner:
  ```
  cd /home/daniel/development/gunzi/local-llm/opt-recon-cache && PYTHONPATH=$PWD /home/daniel/development/gunzi/local-llm/.venv/bin/python tests/test_recon_cache.py
  ```
  → `ALL RECON-CACHE UNIT TESTS PASSED`. Coverage: env-knob parsing; plain-path parity vs
  cache-off (bitwise) + second-call reuse (bitwise, reconstruct runs exactly once); fused-path
  reuse + per-basis coexistence; LRU eviction under a tiny budget (order + budget enforcement);
  decode gate (rows ≤ 144 never consults/fills); routed-expert + sliced-N exclusion; per-layer
  unload + global clear; budget shrink.
- **These use real `LinearEXL3` instances from small synthetic CPU tensors with the ext kernels
  monkeypatched** (deterministic torch stand-ins for `reconstruct(_slice)` / `reconstruct_had_slice`
  / `hgemm_recon` / `had_r_128`), per the task's fallback allowance. Real-kernel dequant parity is
  NOT yet exercised (see Unverified).

## Unverified (deferred — GPU, model load, benches)

- Real-kernel dequant parity: shadow vs per-call `reconstruct` on actual packed tensors
  (the mocks stand in for the kernels; argument identity and call sequences are verified, the
  kernels' outputs are not — though a hit replays identical kernels on identical contents, so
  byte-identity is expected by construction).
- Parity gate (greedy outputs, plan-02 procedure on qbench + GLM prompts), cache on vs off.
- All benches (GPU-exclusive): prefill 2048/8192/32768, warm resend, decode latency, GTT peak.

## Deferred commands (run in order; one 64 GiB model process at a time, serialize)

Env knob: `EXL3_RECON_CACHE_MIB=<mib>` (try 2048, 4096, 8192). Kill switch/rollback: unset (=0).

```bash
# 0. Unit tests (no GPU, no model) — must pass first
cd /home/daniel/development/gunzi/local-llm/opt-recon-cache && PYTHONPATH=$PWD \
  /home/daniel/development/gunzi/local-llm/.venv/bin/python tests/test_recon_cache.py

# 1. Parity, plan-02 procedure (greedy decode, cache OFF vs ON), qwen 2.05bpw then GLM Yamz:
#    run the plan-02 golden/greedy harness twice, differing only in EXL3_RECON_CACHE_MIB
#    (0 vs 4096), MODE=prefill-style greedy prompts on qbench + GLM prompts; outputs must be
#    token-identical. (Exact harness invocation per plan-02 run log; NOTE the dispatch's
#    LABEL/MODE/PROMPT_TOKENS env convention belongs to the bench wrapper of this dispatch
#    series — resolve against that wrapper, bench_decode.py itself takes --label/--prompt-tokens.)

# 2. Prefill benches (per model: qwen3.8-flash-next-exl3/2.05bpw, cpu-trial/GLM-5.3-Flash-EXL3-Yamz)
#    LABEL=reconcache-0    MODE=prefill PROMPT_TOKENS=2048,8192,32768   # control (cache off)
#    LABEL=reconcache-2048 MODE=prefill PROMPT_TOKENS=2048,8192,32768 EXL3_RECON_CACHE_MIB=2048
#    LABEL=reconcache-4096 MODE=prefill PROMPT_TOKENS=2048,8192,32768 EXL3_RECON_CACHE_MIB=4096
#    LABEL=reconcache-8192 MODE=prefill PROMPT_TOKENS=2048,8192,32768 EXL3_RECON_CACHE_MIB=8192
#    # warm resend (prompt-cache scenario): re-send the identical prompt in the same process /
#    # same LABEL=reconcache-4096 window and record the second-pass pp times
# 3. Decode check (must be within ±2%): LABEL=reconcache-4096 MODE=latency EXL3_RECON_CACHE_MIB=4096
# 4. GTT peak: 50_smoke.sh with EXL3_RECON_CACHE_MIB=4096 → peak delta ≤ budget + 5% (no leak)
```

Acceptance (plan): ≥8 % pp8192 and ≥12 % pp32768 on ≥1 model at a fitting budget; decode ±2 %;
GTT peak delta ≤ budget + 5 %.

## Open questions / notes for the dispatcher

1. **GTT accounting seam (plan step 3) — the seam named in the plan does not exist in this tree**:
   `mem_info_gtt_used` matches nothing; `use_per_device` budgets are resolved once at load time
   (`model_ls.py` / `model_tp.py` via `set_memory_fraction_use`) and shadows materialize lazily
   *after* load, so there is no runtime subtraction point. Best-effort implemented instead: the
   LRU budget IS the cap (hard-eviction above `EXL3_RECON_CACHE_MIB`) + `recon_cache.total_bytes`
   accessor for future accounting integration. If the autosplit path later gains a runtime GTT
   hook, subtract `recon_cache.total_bytes` there.
2. **Fused-Hadamard handling (plan Design §4)**: a shadow CAN serve the fused path — the shadow
   stores the fused kernel's own output basis (`reconstruct_had_slice` with full `svh`,
   `n_start=0`), and a hit replays the identical `hgemm_recon(x, shadow, y)` sequence, so no
   basis conversion and no restricted-to-plain fallback was needed. Caveat kept from the plan:
   `lm_head`-class shapes (N > 32768, the sliced loop) are not cached at all. EXL3_NO_FUSED_
   RECONSTRUCT=1 interplay is automatic (basis follows the active path per call).
3. **Rows-regime duality**: a layer seeing both 144 < rows < 1024 (plain) and rows ≥ 1024 (fused)
   may materialize two shadows (one per basis). Budget-holding cost is visible via `stats()`;
   if that's undesirable in practice, add a "single basis per layer" policy knob later.
4. Concurrency note: this branch only APPENDS to `util/cuda_flags.py` (own block); reconcile
   trivially with the `EXL3_MOE_DEVICE_COUNT` branch.
5. `PLAN-opt-reconcache.md` (skeleton) is in THIS worktree root per dispatch instruction; move/copy
   to local-llm root only if the series convention requires it after results land.
