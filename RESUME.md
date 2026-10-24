# RESUME — opt05: GLM-5.3 DSA prefill (lazy k-pool planes + batched up-projection)

**COMPLETE 2026-10-24 — verification gates run: parity PASS everywhere, acceptance
FAIL (serving delta ≈ 0 %) → NO-GO for merge. Full results, per-part attribution
and verdict: `PLAN-opt-glm-dsa.md`. Raw artifacts: `tests/results/`.**

Branch `opt/glm-dsa-prefill` (worktree `opt-glm-dsa/`, off `cpu-only` @ bc731a5e).
Commits: `4aa9c428` (code), `2e008d7e` (harness + default-on). Written under a ~30-min
host deadline: parity gate + benches DEFERRED (see §5), everything else landed.

## 1. Planes-reconstructability finding (the make-or-break question)

**Answer: the k-pool plane IS derivable from cached idx keys (plan branch 1, for the
pool plane); the per-token idx-key plane itself is NOT derivable later and its append
must continue every chunk.** So Part A = "skip the POOL upkeep chain while dense +
bulk-build the pool plane on the first sparse-mode step", not "skip everything".

Evidence (file:line against `cpu-only` @ bc731a5e, i.e. the main checkout; worktree
lines are shifted ~+25 after the edits):

- `_indexer_keys` consumes the TRANSIENT block input `x`: `idx_wk.forward(x)` at
  `modules/mla_attn.py:524`, gate `F.linear(x, self.idx_kpool_gate)` at `:533`. `x` is
  never cached → packed `[k || gate]` rows cannot be recomputed for past tokens; they
  must land in the idx plane every chunk (`:899-902`, comment at `:895-897` already says
  the cached path must fill the plane while dense).
- `_update_pool_plane` (`mla_attn.py:632-663`) is a PURE FUNCTION of the cached idx
  plane + constant module weight: reads `plane = idx_layer.get_idx()` (`:641`), gathers
  members `flat[rows]` (`:653-655`), `probs = softmax(gates + ape)` (`:657`),
  `pool_keys = (probs*keys).sum(1).to(half)` (`:658`). No other history input.
- The sparse scorer for kpool models reads ONLY the pool plane:
  `_indexer_topk_kpool(..., pool_plane = idx_layer.get_pool())` at `mla_attn.py:908-914`,
  scoring at `:731-735`; incomplete tail pools are appended as RAW indices (`:692-704`)
  and attended through the latent cache — no plane state involved. (Non-kpool DSA,
  GLM-5.2-style, reads the idx plane directly at `:920` — but that plane is appended
  unconditionally anyway, so nothing is deferrable there.)
- Complete pools are immutable (docstring `:636-638`) → incremental and bulk builds
  produce the same values for the same members; `update_pool_direct`
  (`cache/mla.py:149-154`) is a plain paged append at an offset — the offset is the only
  difference between "incremental" and "bulk", and it is a free int.
- **Hidden third consumer found**: the graph-captured decode block maintains the pool
  plane ITSELF — `bc_mla.py:397-403` compiles `_dsa_pool_update_kernel` into the capture
  (pool plane wired at `:148-170`). A dense-window skip must therefore be honored before
  the captured step too, or the plane keeps a hole the sparse selection reads as zero
  keys. Handled in `bc_mla_step` (seqlen=0 bulk rebuild, same arithmetic).

What dense mode still pays after Part A (NOT deferrable): idx_wk GEMM + k_norm LN +
rope + gate GEMM + cat + 1st `mla_plane_append` (~6-7 launches). What it skips: the
kpool torch chain (arange/gather/view/softmax/mul-sum/to-half) + 2nd `mla_plane_append`
+ the `torch.tensor([first_pool])` pageable H2D (~10-11 launches of the ~17/layer/chunk).
For GLM yamz (11 full-indexer DSA layers, index_topk 2048, kpool 4): ~110-120 fewer
launches per dense chunk.

## 2. What landed (worktree `opt-glm-dsa/`)

- **Part A** (`EXL3_DSA_LAZY_PLANES`, default **1**): dense-mode defer + one-shot bulk
  rebuild in `_update_pool_plane(..., bulk=True)` (slabbed, 2048 pools/slab, bounds the
  fp32 member transient at long context). Flag `_dsa_pool_lazy` lives on the cache
  layer; honored at BOTH entry paths (`_attend` upkeep block, `bc_mla_step` before the
  captured step) and propagated by `copy_page` (both `CacheLayer_MLA_*` classes).
  The `torch.tensor([first_pool], device=…)` pageable H2D (`mla_attn.py:660` old) is
  replaced by a pre-allocated device-resident scalar + stream-ordered `fill_` — no H2D
  at all. (Plan asked for pinned; rejected deliberately: a pinned staging buffer +
  `copy_(non_blocking=True)` races the CPU refill against an in-flight copy from a
  previous chunk submission across forwards. Device fill_ is race-free, same cost class.)
- **Part B step 1** (`EXL3_DSA_BATCH_UPPROJ`, default 0 — opt-in until benches): kv_b
  halves concatenated once on the N dim (`_w_kv_b_cat`, 16 MiB/layer at GLM dims,
  built lazily in `_mha_form`), ONE `torch.mm` per tile into a merged buffer,
  `_mla_prefill_mha_kernel` gains runtime `k_row_stride`/`v_row_stride` and reads the
  column-half views. Gate-off path passes the contiguous strides → bit-identical to
  the old hard-coded strides.
- **Part B step 2** (`EXL3_DSA_FUSED_UPPROJ`): NOT attempted (per instructions). The
  gate is read and documented as reserved/no-op — scaffold only.
- Decode path otherwise untouched (bc graph, flash-decode, dispatch kernels byte-equal;
  the only decode-visible change is the bc_mla_step arrears check, which is a no-op
  when the flag is unset, and the pool-upkeep scalar shared with prefill).

## 3. Verified (this session, GPU, synthetic only — no model loads)

`test_dsa_lazy_planes.py` (worktree root):
- Part A: lazy vs incremental arms share ONE module (same APE) and identical packed idx
  rows fed through the REAL `mla_plane_append`/`update_pool_direct`; final pool planes
  `torch.equal` (bit-identical) for (1) prefill crossing (dense 2048 crossed in chunk 3)
  and (2) decode crossing (bc-style seqlen=0 rebuild mid-stream), 1637 pools.
- Part B: two-GEMM vs batched `w_kv_cat`, single-tile AND multi-tile (running softmax
  state + causal masking): attention outputs max-abs 0.0.
- Caveat: bit-identity is on THIS stack (gfx1150, torch 2.13+rocm10); fp32 op kernel
  selection by shape could differ elsewhere → the model-level parity gate still governs.

## 4. NOT verified / deferred

- 3-regime greedy parity on GLM yamz (hard gate, plan §Verification 1) — deferred.
- Launch-count evidence (`rocprof --stats`), prefill/latency benches — deferred.
- `EXL3_DSA_BATCH_UPPROJ` default flip — deferred until step-1 benches.
- Part B step 2 (fused up-projection in the attention kernel) — not attempted.

## 5. Exact deferred commands

Unit harness (rerun anytime, ~40 s, no model):
```
cd /home/daniel/development/gunzi/local-llm/opt-glm-dsa && \
PYTHONPATH=. EXL3_DSA_LAZY_PLANES=1 EXL3_DSA_BATCH_UPPROJ=1 ../.venv/bin/python test_dsa_lazy_planes.py
```

Parity (3 regimes, GLM yamz; needs the GPU free, one model process at a time).
`cpu-trial/tests/golden/self_golden.py` covers fixed short/med/long prompts only (16
tokens, `--model-dir`, honors PYTHONPATH) — it does NOT take prompt-length/chunking
args, so:
1. Short/med/long token-exactness via the existing harness (quick confidence check):
```
cd /home/daniel/development/gunzi/local-llm && \
PYTHONPATH=/home/daniel/development/gunzi/local-llm/opt-glm-dsa EXL3_CPU_TEXT_ONLY=1 \
  .venv/bin/python cpu-trial/tests/golden/self_golden.py record \
  --model-dir models/GLM-5.3-Flash-EXL3-Yamz/mix --out /tmp/opencode/opt05_ref.pt   # baseline: gate-off tree (main checkout)
# then same with PYTHONPATH=.../opt-glm-dsa → /tmp/opencode/opt05_new.pt, then compare
```
   (record the REFERENCE from the main checkout with `EXL3_DSA_LAZY_PLANES=0
   EXL3_DSA_BATCH_UPPROJ=0`, the ARM from the worktree with the gates on.)
2. The plan's exact 3 regimes (prompt 1024 pure-dense / 3000 crossing / 8192 in
   2048-token chunks, ×512 greedy tokens, zero divergence) need prompt-length +
   chunked-prefill knobs — small driver to write when the host is back: clone
   self_golden.py's record loop, parameterize `ids = tokenizer.encode(prompt)` by
   padding to N tokens and feed the 8192 case through the generator's chunked-prefill
   path (2048-token chunks). Same A/B: gate-off reference vs gate-on arm, both on the
   SAME model dir + device.

Bench (server-based, `scripts/65_bench.sh`; GPU-exclusive window):
```
cd /home/daniel/development/gunzi/local-llm
MODEL=glm5.3-flash-yamz CTX=65536 KV=8 bash scripts/60_serve.sh          # with the worktree importable
LABEL=dsa-lazy MODE=prefill PROMPT_TOKENS=512,2048,8192,32768 bash scripts/65_bench.sh
LABEL=dsa-lazy MODE=latency bash scripts/65_bench.sh                     # decode neutrality
# A/B: baseline rows first (gates off), then LABEL=dsa-lazy[-batchupproj] with gates on
```
Acceptance (plan): ≥10 % pp512/pp2048 dense-regime gain, no >3 % regression at
pp8192/pp32768, latency mode flat.

## 6. Open questions

- Is the copy_page flag propagation sufficient for branch/lookahead copies that
  straddle the dense window (rare; bulk rebuild is idempotent so worst case is wasted
  work, not wrongness — but untested)?
- The 16 MiB/layer resident `_w_kv_b_cat` (176 MiB across 11 layers on a UMA iGPU) —
  measure whether the single-GEMM shape win beats the residency cost at pp8192/pp32768
  before flipping the default.
- `_mha_block_m` smem probing cache key includes block_m/block_n/num_stages but not the
  new strides — fine (strides are runtime args), noted for awareness.
- Does the bulk rebuild's slab of 2048 pools need tuning for 32k+ contexts (transient
  is 8 MiB fp32 members — probably not)?
