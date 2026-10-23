# opt02 — Remove the per-MoE-layer `expert_count.tolist()` drain (branch `opt/tolist-counts`)

Status: **partial land, unit-verified, E2E UNVERIFIED** (host window forbade models/benches).
Commits: `369903e3` (code), `30608781` (tests), + docs.

## What landed (all gated by `EXL3_MOE_DEVICE_COUNT`, default ON, `0` = old path)

1. **Env knob** in `exllamav3/util/cuda_flags.py` — own commented block at the END of the file
   (stdlib-only read; setup.py safe). Consumed by `modules/block_sparse_mlp.py`.
2. **Pinned-ring staging** (`_PinnedCountRing`, 4 slots, per-module lazy alloc, freed in `unload()`):
   - count readback: async D2H into pinned + event sync (values identical to `.tolist()`);
   - FUSED_DET slot tables: async H2D from pinned (replaces per-layer pageable `host_to_device`
     at old `:1227`; ring owns the pinned source, so HIP-safe). Slot reuse is fenced by the
     readback event (stream order) — reasoning in the class docstring.
3. **Worst-case-grid fused launch**: when the knob is ON and M-tile banding doesn't apply, the
   `run_fused(len(counts))` num_active hint is replaced by `run_fused(-1)` (unknown counts) —
   same contract the all-fused fast path already uses; M-tile tiered launches (mtile_ok) unchanged.

## Kernel-contract verification (honest finding)

`exllamav3_ext/exl3_moe*` sources are NOT in this workspace (only the prebuilt
`exllamav3_ext.cpython-313-x86_64-linux-gnu.so`; no `exl3_moe*.cu` anywhere under local-llm).
Evidence used instead: (a) existing `run_fused(-1)` fast path + comment "num_active -1 = unknown,
kernel launches at max concurrency" (block_sparse_mlp.py); (b) tiered launches rely on the kernel
filtering experts by DEVICE-side counts against [count_lo, count_hi]; (c) analysis note
`notes/prefill-path-qwen.md` §13.1 ("it supports num_active = -1 if the tier plan moved on-device").

## What could NOT be removed python-only (Path A limit — IMPORTANT for next steps)

The tolist **sync itself remains** (staged, not eliminated): the batch tier plan consumes exact
host counts in three places the fused kernel cannot cover — (1) batched-recon `plan_groups`
(moe_batch_recon), (2) FUSED_DET host table layout `base/kind/n_slots` when heavy experts exist,
(3) the per-expert DQ/torch loop slices `token_sorted[start:end]`. Removing the sync entirely
needs the device-side tier plan (Path B: device-side exclusive scan / kernel-consumed offsets),
or a speculative all-fused launch with heavy-fallback (design sketched in PLAN-opt-tolist.md §Next).

Untouched as required: decode/bsz≤8 branch (`run_bszN`, ~:1115/:1391 region), CPU/MoeCpuHost
offload branches (`block_sparse_mlp_cpu.py` tolists stay gated by their offload-mode checks),
CUDA-path behavior byte-identical when knob is `0`.

## Unit test (runs WITHOUT a model; GPU optional, tiny tensors)

```bash
cd /home/daniel/development/gunzi/local-llm/opt-tolist
PYTHONPATH=. /home/daniel/development/gunzi/local-llm/.venv/bin/python tests/test_moe_device_counts.py
```
5/5 PASS (ring readback == tolist across wrap-arounds; upload roundtrip; knob plumbing; host-vs-device
slot-table layout invariant; launch-site presence). pytest-compatible.

## DEFERRED — exact commands to run after this window

```bash
# 0. sanity import from the worktree
cd /home/daniel/development/gunzi/local-llm/opt-tolist
PYTHONPATH=. /home/daniel/development/gunzi/local-llm/.venv/bin/python -c \
  "import exllamav3; print(exllamav3.__file__)"   # must point into opt-tolist/

# 1. PARITY GATE (plan §Verification, hard requirement; zero greedy divergence on both models)
#    old path baseline:
EXL3_MOE_DEVICE_COUNT=0 MODEL=models/qwen3.8-flash-next-exl3/2.05bpw MODEL_DIR=models/qwen3.8-flash-next-exl3 \
  bash scripts/50_smoke.sh          # temperature=0, qbench_prompts.json, 128–512 new tokens
#    new path:
EXL3_MOE_DEVICE_COUNT=1 MODEL=models/qwen3.8-flash-next-exl3/2.05bpw MODEL_DIR=models/qwen3.8-flash-next-exl3 \
  bash scripts/50_smoke.sh
#    then GLM: models/GLM-5.3-Flash-EXL3-Yamz/mix, 4 varied natural prompts, both knob settings

# 2. BENCH (GPU-exclusive, after plan-01 baseline exists; append rows LABEL=tolist-fix)
LABEL="tolist-fix" MODE=prefill PROMPT_TOKENS=512,2048,8192 bash scripts/65_bench.sh
LABEL="tolist-fix-old" EXL3_MOE_DEVICE_COUNT=0 MODE=prefill PROMPT_TOKENS=512,2048,8192 bash scripts/65_bench.sh
MODE=latency   # decode regression check (expect neutral; decode path untouched)

# 3. bench row discipline (AGENTS.md): every row → bench_history.csv + bench_metrics.csv twin
```

## Open questions / risks

- `run_fused(-1)` vs `len(counts)` grid size on mtile-capable builds unmeasured (this host has
  MOE_MTILE=False, so the tiered branch never ran anyway); if benches regress, keep only the
  pinned staging under the knob and hard-code the hint back.
- Pinned ring costs (E+1)·8·2·RING bytes per module — negligible, but allocates per-layer modules
  on first batched forward; watch fragmentation on the 64 GiB serving config.
- Parity assumption: identical count VALUES ⇒ identical tier plan ⇒ bit-identical outputs. The
  pinned copy is a plain D2H of the same tensor, so no fp reordering is involved, but the gate
  above is still mandatory before merge.
- Path B (true sync removal) design notes live in PLAN-opt-tolist.md §Next steps.

## Merge notes

- `util/cuda_flags.py`: knob appended as its own block at EOF to minimize conflict with the
  concurrent EXL3_RECON_CACHE_MIB agent — trivial textual reconcile either way.
- Do NOT merge before the parity gate + benches above have been run and recorded.
