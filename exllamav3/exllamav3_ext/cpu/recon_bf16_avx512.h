#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// AVX-512 vectorized twin of exl3_cpu_dequant_bf16 (Plan 07, Option-2 clause for the DEQUANT
// only; the GEMM stays torch.matmul). Same contract, same arguments, same output [out, in]
// BF16 row-major -- and the same VALUES: the state decode is integer-exact and the butterflies
// replicate the scalar radix-2 stage order (pair (j, j^width) per stage), so the fp32
// accumulation order matches element-for-element and the result is bit-identical to the scalar
// binding, including the final fp32->BF16 round-to-nearest-even (integer conversion formula,
// identical to c10::BFloat16's scalar rounding).
//
// Threading: independent 128x128 blocks, parallelized over n-block strips with
// at::parallel_for (the ambient torch intra-op pool; torch.set_num_threads is the knob).
// Blocks write disjoint output rows, so the result does not depend on the thread count.
//
// Host gating: AVX-512 F + BW + BF16. On anything else the call raises (TORCH_CHECK) with a
// message pointing at the scalar binding; callers fall back to exl3_cpu_dequant_bf16.
at::Tensor exl3_cpu_dequant_bf16_avx512
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    bool swizzled
);
