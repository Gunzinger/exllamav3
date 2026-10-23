# Plan 06 — Batch the GDN chunk-scan row loop (Qwen prefill) — report

Status: **code landed on `opt/gdn-batch` (worktree `opt-gdn-batch/`), benches deferred** —
see `opt-gdn-batch/RESUME.md` for the loop-shape finding, unit-test evidence and the exact
deferred commands. This file is the skeleton to fill with parity/bench tables when the
GPU-exclusive runs happen.

## Loop-shape finding

(a) batch rows — `gated_delta_rule.py:394` (GDN) / `:342` (KDA) pre-change; nested sub-chunk
loop at `:264` is a sequence split (`EXL3_GDN_SUB_CHUNK`), untouched. Vendored fla kernels
already grid over B (`chunk_delta_h.py:368` `(BV, N*HV)`, `chunk_o.py:162`, `cumsum.py:154`) —
no re-vendor needed.

## Diff summary

- `EXL3_GDN_SCAN_BATCHED` (default ON; `0` = old per-row loop), `_slot_index`, `_chunked_scan_rows`
  shared by GDN + KDA; gather/scatter of pool rows; `history` keeps sequential semantics.
- `test_gdn_rowbatch.py`: 30/30 — GDN bit-identical, KDA bsz>1 ≤ 2.44e-4 out / 3.89e-5 state
  (gates 5e-3/1e-3), outputs AND final recurrent state compared.

## Parity (deferred)

| model | prompt len | result |
|---|---|---|
| qwen 2.05 | 512 / 2048 | TODO |
| qwen 6.05 | 512 / 2048 | TODO |

## Launch count (deferred, ≥30 % target at 2048)

TODO (rocprof --stats old vs new).

## Bench (deferred)

| row | 2.05 pp | notes |
|---|---|---|
| prefill 512 / 2048 / 8192 | TODO | expect 3–8 % |
| latency (decode) | TODO | must be flat |
| throughput CONCURRENCY=4 | TODO | batched path pays here |

## Rollback

`EXL3_GDN_SCAN_BATCHED=0`; branch unmerged.
