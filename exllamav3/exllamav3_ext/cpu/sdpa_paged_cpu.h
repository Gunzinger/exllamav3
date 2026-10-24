#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// Fused CPU paged-cache attention decode kernel (work package C7). One call replaces the torch
// path's per-sequence page gather + chunked SDPA (modules/attention_fn/torch.py,
// fn_torch_sdpa_cpu_paged -> _torch_paged_gather_sdpa) for decode-class query lengths: the KV
// pages are read in place through the block table (no gather copy) and the step's new K/V rows
// come from the separate k_new/v_new tensors, positioned at cache_seqlens..cache_seqlens+Q-1.
// Flash-decode math per (b, qi, h) head-task: query row qi attends causally to keys
// 0..cache_seqlens[b]+qi, GQA maps q head h to kv head h/(H/HK), scores are scaled by `scale`
// and reduced with an online softmax (running max, running sum, rescaled fp32 output
// accumulator), and the output is rounded to fp16 exactly once on store. fp16 loads, fp32 FMA
// accumulation throughout. Head tasks (B*Q*H) run on the exported moe worker pool
// (exl3_moe_cpu_pool_run; EXL3_CPU_SDPA_THREADS, default 6, 1 = serial, never exceeds
// the pool's current size; an unspawned pool runs serially). Per-head math is worker-
// independent, so any worker count gives bit-identical output. Requires
// AVX-512F/BW/VL/DQ + FMA + F16C (sdpa_paged_cpu_has_avx512); the Python side keeps the torch
// gather path when it reports false or the tensors are ineligible (head_dim not %16, dtypes
// not fp16, q_len > 8, non-contiguous). The cache write-back stays in Python.
//   q:             (B, Q, H, D) fp16, Q in 1..8, D a multiple of 16 up to 512, H up to 64
//   k_new, v_new:  (B, Q, HK, D) fp16, rows treated as appended at cache_seqlens
//   k_cache/v_cache: (P, page, HK, D) fp16
//   block_table:   (B, max_pages) int32/int64
//   cache_seqlens: (B,) int32/int64, valid keys in the cache BEFORE this step's new rows
//   out:           (B, Q, H, D) fp16
void sdpa_paged_cpu
(
    const at::Tensor& q,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    double scale,
    at::Tensor& out
);

// True when the AVX-512 tier this kernel needs is present at runtime.
bool sdpa_paged_cpu_has_avx512();
