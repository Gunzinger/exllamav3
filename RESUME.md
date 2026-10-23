# RESUME — opt/gdn-batch (plan 06: batch the GDN chunk-scan row loop)

Branch `opt/gdn-batch` in worktree `/home/daniel/development/gunzi/local-llm/opt-gdn-batch`
(off `cpu-only` @ bc731a5e). All work committed; deferred items need GPU + model runs.

## Loop-shape finding (the plan's a/b/c question)

**(a) batch rows.** The outer python loop is per batch row; each iteration slices
`q[i:i+1]` and calls the fla chunk kernel with B=1:

- GDN: `exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py:394` (pre-change)
  `for i, (so, si) in enumerate(zip(rows_out, rows_in)):` → `_chunked_scan(chunk_gated_delta_rule, q[i:i+1], …)`
- KDA: same file `:342`, identical shape for `chunk_kda`.

The sub-chunk loop at `:264` (`_chunked_scan`, `EXL3_GDN_SUB_CHUNK`) is a **sequence**
split nested inside each row call — not the outer loop — and stays as-is (bit-identical
state carry). The KDA-vs-GDN split (`channelwise_g` branches) is untouched.

**Vendored fla kernels already accept B>1** (no re-vendor needed): entry points
`exllamav3/vendor/fla/__init__.py:29` (`chunk_gated_delta_rule`, `[B, T, H, K]`,
`initial_state: [B, HV, K, V]`) and `:71` (`chunk_kda`); grids are `(…, B*HV)` /
`(NT, B*H)`: `chunk_delta_h.py:368`, `chunk_o.py:162`, `cumsum.py:154/192`.

## What landed (commit 15469b25)

- `gated_delta_rule.py`: new `EXL3_GDN_SCAN_BATCHED` (default **ON**, proven by the unit
  test; `=0` restores the per-row loop exactly), `_slot_index` helper (cached arange for
  the default slots, no per-layer H2D), `_chunked_scan_rows` shared by GDN and KDA.
  Batched pass = gather `recurrent_state[rows_in, 0]` (advanced indexing → copy, so the
  scan never aliases the pool) → one `_chunked_scan` over (B, T, H, D) → scatter write-back
  to `rows_out` in pool dtype. `history=True` (spec pass, rows_in ≠ rows_out) and
  `rows_in != rows_out` keep the sequential per-row loop (a row could read a slot another
  row writes; sequential semantics preserved). Decode fused-step path untouched.
- `test_gdn_rowbatch.py` (worktree root): old-vs-new **outputs AND final recurrent state**,
  bsz 1 and 4, T 128/256, `EXL3_GDN_SUB_CHUNK` 64 vs whole-pass, fp32/fp16 pools,
  permuted slot rows, KDA variant, history sanity — **30/30 pass**: GDN bit-identical
  (0.00e+00 everywhere), KDA bsz>1 ≤ 2.44e-4 out / 3.89e-5 state (gates 5e-3 / 1e-3).

Run: `PYTHONPATH=$PWD /home/daniel/development/gunzi/local-llm/.venv/bin/python test_gdn_rowbatch.py`
(from the worktree root; import resolves into the worktree, no JIT rebuild, no symlink needed).

## Verified / unverified

- Verified: import resolves in worktree; unit test green (GPU, small synthetic tensors).
- Unverified (deferred, GPU-exclusive + model runs): model parity, launch-count, perf.

## Deferred commands (run from `/home/daniel/development/gunzi/local-llm/opt-gdn-batch`; models at `models/qwen3.8-flash-next-exl3/{2.05,6.05}bpw`; serialize benches, one 64 GiB process at a time)

The dispatcher's env-var bench style (`LABEL=… MODE=… PROMPT_TOKENS=…`) belongs to its
wrapper; the direct-harness equivalents are given below (`cpu-trial/bench_decode.py --label
--prefill-tokens/--prompt-tokens`; it writes `bench_history.csv` + `bench_metrics.csv` twin).

1. Parity + recurrent state (qwen 2.05 AND 6.05, 512 and 2048 tokens):
   - `tests/golden/self_golden.py compare` (reference: record with `EXL3_GDN_SCAN_BATCHED=0`
     first if not present) with `PYTHONPATH=/home/daniel/development/gunzi/local-llm/opt-gdn-batch`,
     greedy, med+long prompts — token-exact or explicitly reclassified.
   - Recurrent-state comparison at 512/2048: rerun `test_gdn_rowbatch.py` with T=512/2048
     (edit `T in (512, 2048)` in `main()`), plus on-model check via
     `cpu-trial/parity_gdn_step.py` pattern if it covers prefill states; else extend the
     golden hook to dump state per plan §Implementation.4.
2. Launch-count (plan acceptance: ≥30 % fewer launches on a 2048 chunk):
   `rocprof --stats -o old -- env EXL3_GDN_SCAN_BATCHED=0 …` vs same with default, one
   512/2048-token prefill pass; compare kernel-launch counts per layer ×36.
3. Bench: `LABEL=gdn-batch MODE=prefill PROMPT_TOKENS=512,2048,8192` (env style) ≡
   `python cpu-trial/bench_decode.py --model-dir models/qwen3.8-flash-next-exl3/2.05bpw
   --label gdn-batch --prefill-tokens 512|2048|8192 --prompt-tokens … --steps 0` (prefill rows),
   `MODE=latency` ≡ default decode rows (`--steps 64`), `MODE=throughput CONCURRENCY=4` ≡
   `--steps 64` with the concurrency/ubatch setting of that wrapper (bsz=4 is where the
   batched path pays). Expect 3–8 % pp on prefill; decode must be flat (fused step path).

## Open questions

- KDA bsz>1 shows ≤2.4e-4 diff (autotune picks a different tile when B>1?) — within gates,
  watch the model parity gate on 6.05 (KDA layers?) for drift.
- `recurrent_slots_scan` permutations other than arange now cost one tiny H2D
  (`torch.tensor(rows)`) per call — only when the param overrides slots; default slots use the
  cached arange. Negligible, but measurable in the launch-count pass.
- The 6–10 python dispatches/layer outside the scan (projections → fused_op → conv → scan →
  norm → o_proj) are untouched — separate, larger win, needs its own plan.
- Buffer pre-allocation (plan §Problem last paragraph) NOT done (time); the batched path
  already removes one of the two per-row fp32 state copies per call (gather is one copy for
  all rows vs one per row).
