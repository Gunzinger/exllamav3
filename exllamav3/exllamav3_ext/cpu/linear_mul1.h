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
// Returns [M, out_features] fp32 when want_fp32_out, else fp16.
at::Tensor exl3_cpu_linear_forward
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& x,
    bool want_fp32_out
);
