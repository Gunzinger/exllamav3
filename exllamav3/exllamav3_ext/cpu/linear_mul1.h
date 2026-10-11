#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// Standalone CPU forward for one mul1-quantized EXL3 dense linear: y = x @ Wᵀ (+ bias), where W is
// the dequantized weight (diag(suh) · H128 · What · H128 · diag(svh) per 128-block). Runs the
// cpu/moe_mul1.cpp kernels over the shared worker pool; the activation-side Hadamard/int8
// quantization and the output transform are fused exactly as in the expert path (fp32
// accumulation, single final rounding).
//   trellis: [in/16, out/16, 16K (+8 at half-integer K)] int16, CPU, linear tile layout
//   suh/svh: half [in_features] / [out_features]
//   K:        bitrate (integer or half-integer), cross-checked against the tile width
//   bias:     half [out_features], optional
//   x:        [M, in_features] half, CPU, contiguous
//   swizzled: trellis bytes are in the packed layout (exl3_moe_cpu_swizzle_group) the loader
//             repacked at load; re-derived from the rate on the C++ side either way
// Returns [M, out_features] fp32 when want_fp32_out, else fp16.
at::Tensor exl3_cpu_linear_forward
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& x,
    bool want_fp32_out,
    bool swizzled
);

// Same computation for a batch of same-input matrices (one projection group per pool round set
// instead of one call per matrix). The per-matrix lists must be the same length; biases optional
// per matrix; out dtype per matrix (fp32, else fp16).
std::vector<at::Tensor> exl3_cpu_linear_forward_batch
(
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<double>& K,
    const std::vector<std::optional<at::Tensor>>& bias,
    const at::Tensor& x,
    const std::vector<bool>& want_fp32_out,
    const std::vector<bool>& swizzled
);
