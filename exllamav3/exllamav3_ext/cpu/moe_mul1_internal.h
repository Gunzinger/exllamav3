#pragma once

#include "moe_mul1.h"

// Internal cross-TU surface of moe_mul1.cpp for cpu/linear_mul1.cpp (the standalone dense-linear
// forward). Not part of the module's public API (moe_mul1.h); the expert path is unchanged.

// Matrix descriptor from a trellis [k/16, n/16, 16K (+8 at half-integer rates)] + suh/svh (+ optional
// bias) in the checkpoint's linear tile layout (no swizzled repack). Validates as make_matrix does.
MoeCpuMatrix exl3_cpu_mul1_make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias
);

// One matrix end to end on the moe_mul1 kernels: per-row suh/Hadamard-128 transform + int8
// quantization, fused-decode GEMV, svh/Hadamard-128 output transform, bias add. Same arithmetic as
// the expert path (fp32 accumulation, single rounding to the caller's dtype).
//   x:   [m, mat.k] half
//   out: [m, mat.n] fp32 (overwritten)
//   threads <= 0 selects the default (EXL3_CPU_LINEAR_THREADS / EXL3_MOE_CPU_THREADS / physical cores)
void exl3_cpu_mul1_matrix_forward(const MoeCpuMatrix& mat, const at::Half* x, int m, float* out, int threads);
