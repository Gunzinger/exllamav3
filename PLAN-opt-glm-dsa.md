# PLAN-opt-glm-dsa — GLM-5.3-Flash prefill: lazy indexer planes + up-projection batching

Status: implemented on branch `opt/glm-dsa-prefill` (worktree `opt-glm-dsa/`); synthetic
parity bit-exact; model-level 3-regime parity + benches DEFERRED. See `RESUME.md` for
the full state, verified/unverified split and exact deferred commands.

## Planes-reconstructability answer (summary)

- k-pool POOL plane: **derivable from cached idx keys** — `_update_pool_plane`
  (mla_attn.py:632-663 @ cpu-only) is a pure function of the cached packed
  `[k || gate]` plane rows + constant APE; complete pools are immutable. → deferred
  upkeep in dense mode + one-shot bulk rebuild on the first sparse-mode step is exact.
- Per-token IDX plane: **NOT derivable** — `_indexer_keys` consumes the transient
  block input `x` (idx_wk GEMM :524, gate F.linear :533); the append must run every
  chunk (it is also cheap relative to the pool chain).
- Third consumer: the graph-captured decode block maintains pool planes itself
  (bc_mla.py:397-403) → arrears honored in `bc_mla_step` (seqlen=0 bulk rebuild).

## Changes

| part | gate | default | what |
|---|---|---|---|
| A | `EXL3_DSA_LAZY_PLANES` | 1 (bit-exact synthetic state equality; model parity deferred) | defer k-pool upkeep while context ≤ index_topk; bulk rebuild `[0, total//P)` on first sparse step (prefill `_attend`, captured decode `bc_mla_step`, `copy_page` propagation); pre-allocated device-resident offset scalar replaces the pageable `torch.tensor([first_pool])` H2D |
| B1 | `EXL3_DSA_BATCH_UPPROJ` | 0 (pending benches) | kv_b halves cat'd on N dim → one GEMM/tile into merged buffer; `_mla_prefill_mha_kernel` takes runtime k/v row strides |
| B2 | `EXL3_DSA_FUSED_UPPROJ` | reserved | NOT implemented (scaffold only) |

Expected effect at GLM yamz (11 DSA layers, topk 2048, kpool 4): ~110-120 fewer
launches per dense chunk (A), −11 GEMM launches + merged-buffer traffic (B1).

## Verification

1. Synthetic harness (this session, PASSED, bit-exact):
   `PYTHONPATH=. ../.venv/bin/python test_dsa_lazy_planes.py`
2. Deferred: 3-regime greedy parity (1024 / 3000 crossing / 8192-chunked × 512, zero
   divergence) on `models/GLM-5.3-Flash-EXL3-Yamz/mix`; second arch-variant pass on
   `models/glm-5.3-flash-exl3/2.05bpw`.
3. Deferred: `rocprof --stats` launch-count evidence on one 2048-token chunk.
4. Deferred bench (GPU-exclusive): `LABEL=dsa-lazy[-batchupproj] MODE=prefill
   PROMPT_TOKENS=512,2048,8192,32768` + `MODE=latency`; acceptance ≥10 % dense regime,
   no >3 % regression at pp8192+, decode flat.

## Rollback

`EXL3_DSA_LAZY_PLANES=0 EXL3_DSA_BATCH_UPPROJ=0`; branch stays unmerged until §2 passes.
