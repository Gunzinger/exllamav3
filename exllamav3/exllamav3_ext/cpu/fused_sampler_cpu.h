#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// CPU port of the fused terminal sampling step (generator/sampling_fused.cu, work package B7).
// One call per token computes the collapsed stack [Temperature?] [MinP?] [TopK?] [TopP?] (Sample |
// Argmax) directly in logit space:
//   p_i >= min_p * p_max                    <=>  l_i >= M + ln(min_p),  M = max(l)
//   softmax(l)^(1/T) renormalized           <=>  softmax(l / T)
//   categorical sample from softmax(l / T)  ==   argmax(l_i / T + g_i),  g_i ~ Gumbel(0, 1)
// Top-K/top-P keep a top segment of the logit ordering, so the kept set is one threshold: the
// CUDA kernel finds it with a deterministic fixed-point histogram plus sub-bucket refinement;
// here the boundary bucket's elements are gathered and resolved exactly (a few hundred keys),
// which matches the eager sort-based truncation's kept set also at fp32 tie granularity (the
// CUDA kernel is exact only finer than fp16 ULP and diverges from eager on far-tail and tie
// edges). Same modes/filters as the CUDA entry; the additive half mask / packed bitmask and the
// dim scan bound have the same semantics (-inf outside), workspace/histogram are accepted for
// call parity and unused. Gumbel noise is splitmix64(rand_u32, index) -- NOT the CUDA Philox
// stream, so draws match the CUDA path distributionally, not bit-wise (the eager CPU fallback in
// SS_Sample is seeded from the same rand_u32 for the parity tests).
// Single-threaded by design (bsz x dim is a few hundred us of streaming work); the binding
// releases the GIL. Modes: 0 greedy argmax, 1 gumbel sample, 2 + min-P, 3 + top-K/top-P filters
// (F_TOPK 1 | F_TOPP 2 | F_MINP 4). Argmax ties prefer the lowest index (sequential-scan order).
//   logits: (bsz, dim2) half|float, contiguous, dim2 >= dim
//   mask:   additive half (1|bsz, >= dim), or none; bits: packed u32, bit clear = masked
//   sample: (bsz, 1) long out
void fused_sampler_cpu
(
    const at::Tensor& logits,
    const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& bits,
    at::Tensor& sample,
    const at::Tensor& workspace,
    int64_t dim,
    double inv_temp,
    double minp_log,
    int64_t rand_u32,
    int64_t mode,
    int64_t filters,
    int64_t top_k,
    double top_p,
    double inv_temp_filter,
    const std::optional<at::Tensor>& histogram
);
