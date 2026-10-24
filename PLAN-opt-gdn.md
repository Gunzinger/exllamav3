# Plan 06 — Batch the GDN chunk-scan row loop (Qwen prefill) — report

Status: **verified 2026-10-24 — parity green everywhere, launch/perf premise refuted
(NO-GO for merge)**. Code landed on `opt/gdn-batch` (worktree `opt-gdn-batch/`, off
`cpu-only` @ bc731a5e); unit 30/30; model parity byte-identical + bit-identical
recurrent states on qwen 2.05 **and** 6.05; serving-level launch delta **0.00 %**
because the Generator never issues multi-row GDN scans. Verification subagent ran the
deferred gates GPU-exclusive and serialized; not merged.

## Loop-shape finding

(a) batch rows — `gated_delta_rule.py:394` (GDN) / `:342` (KDA) pre-change; nested sub-chunk
loop at `:264` is a sequence split (`EXL3_GDN_SUB_CHUNK`, default 2048 = whole-pass), untouched.
Vendored fla kernels already grid over B (`chunk_delta_h.py:368` `(BV, N*HV)`, `chunk_o.py:162`,
`cumsum.py:154`) — no re-vendor needed.

**Architecture finding (decisive for acceptance):** the serving Generator never produces a
multi-row GDN scan. Prefill is `Generator.iterate → for job in active_jobs: job.prefill()`
(`generator.py:686`), and `job.prefill` forwards **one sequence chunk at a time** —
`model.forward(input_ids = prefill_ids, …)` with `prefill_ids = seq.sequence_ids.torch_slice(...)`
(`job.py:1572`), shape `(1, T)`. Concurrent jobs are batched only at DECODE (fused step path,
untouched by this change). Hook evidence: with 4 concurrent 2048-token jobs, every
`_chunked_scan_rows` call saw `B=1` (288 calls, shapes `(1, 1792, 16, 128)` + `(1, 255, 16, 128)`;
prefill chunks are page-granular, `max_chunk_size`=2048 → 1792+255 tail). The per-row python
loop the plan targeted never loops over more than one row in production.

## Diff summary

- `EXL3_GDN_SCAN_BATCHED` (default ON; `0` = old per-row loop), `_slot_index`, `_chunked_scan_rows`
  shared by GDN + KDA; gather/scatter of pool rows; `history` (spec pass) and `rows_in != rows_out`
  keep sequential semantics; decode fused-step path untouched.
- `test_gdn_rowbatch.py`: 30/30 — GDN bit-identical, KDA bsz>1 ≤ 2.44e-4 out / 3.89e-5 state
  (gates 5e-3/1e-3), outputs AND final recurrent state compared.

## Gate 1 — parity (hard): PASS, both models

Driver `tests/par_gdn.py` (committed): qbench greedy `temperature=0` tokens + post-prefill
recurrent-state capture (hook around `_chunked_scan_rows`; decode never enters it, so the
snapshot is exactly the post-prefill pool) in one model load; arms are separate processes
(knob is import-time, `gated_delta_rule.py:255`).

- Token parity: 4 qbench rows (84/408/1906/1624-token prompts, in+384 out tokens):
  - qwen 2.05: `EXL3_GDN_SCAN_BATCHED=0` vs default — **byte-identical**, sha
    `38966a2f490e279f` = `38966a2f490e279f`, 1534 tokens, 0 divergences.
  - qwen 6.05: **byte-identical**, sha `f9821c9f402cb588` = `f9821c9f402cb588`, 1536 tokens.
- Recurrent state (gates ≤1e-3 fp32 / ≤5e-3 fp16 class; pool dtype fp16 here), fixed tiled
  prompt, first generated token compared too:

| model | prompt | layers compared | bit-identical | worst max-abs (recurrent / conv) | verdict |
|---|---|---|---|---|---|
| qwen 2.05 | 512 | 36 GDN | 36/36 | 0.00e+00 / 0.00e+00 | PASS |
| qwen 2.05 | 2048 | 36 GDN | 36/36 | 0.00e+00 / 0.00e+00 | PASS |
| qwen 6.05 | 512 | 36 GDN | 36/36 | 0.00e+00 / 0.00e+00 | PASS |
| qwen 6.05 | 2048 | 36 GDN | 36/36 | 0.00e+00 / 0.00e+00 | PASS |

Bit-identity is expected: at B=1 both paths call the same `_chunked_scan` on the same
`(1, T, H, D)` view with the same fp32 state copy; the batched path only reorders the
gather/scatter around it (and the unit test shows GDN is bit-exact at B=4 too).

## Gate 2 — launch count: 0.00 % serving-level (acceptance ≥30 %: FAIL); −75 % only in a synthetic multi-row caller

Tooling notes (documented substitutions): classic `rocprof --stats` is not shipped in
ROCm 10.0.0 (only `rocprofv3`), and `rocprofv3 --kernel-trace` **hangs the python process at
startup** on this gfx1150 stack (10-min timeout, no output; retry killed). torch.profiler
kineto does not trace triton launches on this stack (hipBLAS/extension kernels appear as
`cat:"kernel"`, triton's launch API is invisible — verified with mixed-op probes). Substitutes
(committed): `tests/launch_gdn.py` counts triton kernel launches exactly by wrapping
`triton.runtime.jit.JITFunction.run`, plus torch-profiler for the rest; `tests/scan_launch_bench.py`
drives the vendored kernel directly. Non-triton launches (ext/torch) are identical across arms
and unaffected by the diff region.

Serving path, full 2048-token prefill through the Generator (qwen 2.05, triton launches):

| scenario | old | new | delta |
|---|---|---|---|
| bsz=1 (one 2048-token job, chunks 1792+255) | 732 | 732 | **+0.00 %** |
| bsz=4 (4 concurrent 2048-token jobs) | 2928 | 2928 | **+0.00 %** |

At bsz=1 the old loop has exactly one iteration — nothing to batch. At bsz=4 the generator
still runs 288 single-row scan calls (4 jobs × 72 chunks/layers), so the batched path never
engages either. Scan-level microbench (`tests/scan_launch_bench.py`, vendored
`chunk_gated_delta_rule`, T=2048 H=16 HV=48 K=V=128, whole-pass): B=1: 7→7 kernels/pass
(+0.0 %); B=4: 28→7 (**−75 %**). The 30 % launch reduction the plan priced in is real **only
for a direct multi-row caller, which no production path exercises** — the same
premise-refuted shape as plans 02/04, this time one level up: the python row loop is not
the multi-row fan-out site because prefill batching does not exist in the Generator.

## Gate 3 — bench (qwen 2.05, CTX=32768 KV=8, 3 paired same-window reps per size)

Protocol: each rep starts the server fresh per arm (PYTHONPATH=worktree inherited through
`60_serve.sh`'s `nohup env`, verified in `/proc/<pid>/environ`; base arm
`EXL3_GDN_SCAN_BATCHED=0`, batch arm unset), per-arm labeled discard at 4096 (no
seed+size collision with measured 512/2048/8192), then one prefill invocation per rep
(prefill mode = 1 request/size/invocation, so reps = invocations); latency `REPS=3` and
one throughput row per arm after a fresh start + discard. Rep spread within an arm is
±0.5 % — the window was quiet. Rows in `benchmarks/results.csv` (labels `gdn-base`,
`gdn-batch`, `*-discard`). Note: `PROMPT_STYLE=real` applies to latency/throughput only —
`run_prefill` uses seeded synthetic prompts by design (`gen_prompt(target, seed+target)`);
identical in both arms, so the A/B stays valid.

| row | gdn-base (knob=0) | gdn-batch (default) | delta |
|---|---|---|---|
| prefill 512 tok/s (mean of 3) | 171.2 | 172.0 | **+0.46 %** |
| prefill 2048 tok/s (mean of 3) | 175.0 | 175.7 | **+0.37 %** |
| prefill 8192 tok/s (mean of 3) | 141.5 | 141.9 | **+0.23 %** |
| latency decode tok/s (REPS=3) | 12.57 | 12.76 | +1.5 % (noise) |
| throughput C=4 MAX_TOKENS=512 agg tok/s | 21.59 | 22.56 | +4.5 % (single row, <5 % noise rule) |

## Acceptance verdict

1. Parity: **PASS** (byte-identical tokens, bit-identical states, both models).
2. Launch count ≥30 % on a 2048 chunk: **FAIL** (0.00 % serving-level — no multi-row scan
   exists in production; premise refuted at the Generator architecture level).
3. 3–8 % pp gain: **FAIL** (+0.2…0.5 %, within rep noise; consistent with 0.00 % launch
   delta and bit-identical work at B=1).
4. Decode regression >2 %: **PASS** (fused step path untouched; +1.5 % = noise).
5. Throughput regression: **PASS** (none; +4.5 % single-row is under the 5 % noise rule,
   and no mechanism exists for a real gain — prefill is single-row, decode untouched).

**Go/No-Go: NO-GO for merge as a perf win.** The landed default-ON batched path is safe
(bit-identical, zero launch/perf delta at serving) and is the right shape if a future
Generator batches prefill across jobs (`job.prefill` is the seam; the scan-level −75 %
launch fold at B=4 is real); until then it is dead code at serving time. Recommendation:
keep the branch unmerged (or merge purely as groundwork with the plan marked
premise-refuted); the actual prefill lever in this code is the 6–10 python dispatches/layer
outside the scan (RESUME.md open question), which needs its own plan.

## Rollback

`EXL3_GDN_SCAN_BATCHED=0`; branch `opt/gdn-batch` unmerged.
