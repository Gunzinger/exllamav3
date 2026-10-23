#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// Dequantize one mul1-quantized EXL3 dense matrix into a plain BF16 buffer (Plan 07, Option 1:
// CPU prefill tier). The packed trellis is decoded once (scalar reference decode of the tile bit
// ring, both Hadamard-128 passes, suh/svh scaling) and rounded once to BF16 with round-to-nearest-
// even; the result feeds a torch.matmul on the caller side. Zen 5 has no AVX512_FP16, so the fp32
// dequant values are narrowed with the scalar RNE conversion (u16 payload) — a vectorized
// narrow pass is deferred perf work and changes no values.
//   trellis: [in/16, out/16, 16K (+8 at half-integer K)] int16, CPU, linear tile layout or the
//            band-8 packed layout (loader repack); the AVX2 planar layout is rejected
//   suh/svh: half [in_features] / [out_features]
//   K:       bitrate (integer or half-integer), cross-checked against the tile width
//   swizzled: trellis bytes are in the packed layout (re-derived from the rate either way, as
//            everywhere else)
// Returns [out_features, in_features] BF16, contiguous, row-major (W, so y = x @ W.T).
// The dequant is single-threaded scalar code on purpose for this first landing: it is a one-time
// cost per (tensor, cache epoch) and correctness/parity is the gate; parallelizing over the
// worker pool is follow-up work under the usual EXL3_CPU_* knobs.
at::Tensor exl3_cpu_dequant_bf16
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    bool swizzled
);
