#pragma once

#include "moe_mul1.h"

// Internal cross-TU surface of moe_mul1.cpp for cpu/linear_mul1.cpp (the standalone dense-linear
// forward). Not part of the module's public API (moe_mul1.h); the expert path is unchanged.

// Matrix descriptor from a trellis [k/16, n/16, 16K (+8 at half-integer rates)] + suh/svh (+ optional
// bias). swizzled: the bytes are in the packed layout the loader repacked with the group
// exl3_moe_cpu_swizzle_group returns for this matrix's rate (the flag is re-derived from the rate
// either way). Validates as make_matrix does.
MoeCpuMatrix exl3_cpu_mul1_make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias,
    bool swizzled
);

// One matrix end to end on the moe_mul1 kernels: per-row suh/Hadamard-128 transform + int8
// quantization, fused-decode GEMV, svh/Hadamard-128 output transform, bias add. Same arithmetic as
// the expert path (fp32 accumulation, single rounding to the caller's dtype).
//   x:   [m, mat.k] half
//   out: [m, mat.n] fp32 (overwritten)
//   threads <= 0 selects the default (EXL3_CPU_LINEAR_THREADS / EXL3_MOE_CPU_THREADS / physical cores)
void exl3_cpu_mul1_matrix_forward(const MoeCpuMatrix& mat, const at::Half* x, int m, float* out, int threads);

// Same computation for a batch of same-input matrices (the projections of one layer: q/k/v(/g),
// qkv/z) in one pool round set instead of one per matrix; decode rows (m == 1) additionally fuse
// the gemv and output phases per worker. outs[i] receives [m, mats[i].n] fp32 (overwritten).
void exl3_cpu_mul1_batch_forward
(
    const std::vector<MoeCpuMatrix>& mats,
    const at::Half* x,
    int m,
    float* const* outs,
    int threads
);
