#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// Fused CPU decode step of the PLE layer (work package M2). One call per layer per token
// computes the whole forward_streams_cpu math (bsz = 1, seqlen = 1) without the torch ops,
// mirroring modules/ple.py (_short_conv_decode's shift-register conv included):
//   key/value = f32(half(fp32 GEMV(emb, w)))          fp16 weights read as f32, fp32 FMA
//   keyn/queryn = grouped RMS norm (eps, constant_scale, weight + constant_bias), fp32
//   gate   = <queryn, keyn> per stream * gate_scale;  s = sigmoid(signed_sqrt(gate))
//   gated  = s * value                                                    fp32
//   normed = half(gated * rsqrt(mean sq + eps) * (w + bias))              the new conv column
//   conv   = fp32 tap dot over [state | new] at columns k * dilation, y -> half, silu in
//            fp32 -> half (the _short_conv_decode rounding points)
//   delta  = gated + f32(conv out)                                        fp32
//   conv ring = state shifted by one with normed appended, updated in place; conv_stream is
//            the full padded column stream [state | new] the caller reduces to the next state
// Only the fp32 accumulation order (GEMV, gate dot, square sums) differs from torch's
// mm/bmm/mean -- a last-ulp difference the fp16 roundings almost always mask. Single-threaded
// like gdn_step_cpu's default. Requires AVX-512F/BW/VL/DQ + FMA + F16C
// (ple_step_cpu_has_avx512); callers fall back to forward_streams_cpu when it reports false
// or the tensors are ineligible.
//   streams:     (1, 1, H, D) float, contiguous
//   emb:         (1, 1, ple_dim) half, contiguous
//   key_w:       (ple_dim, H * D) half, contiguous (the fp16 Linear's matmul layout)
//   value_w:     (ple_dim, D) half, contiguous
//   norm_*_w:    (H * D) half, bfloat16 or float (the RMSNorm load dtype), contiguous
//   conv_w:      (H * D, 1, ksize) half, contiguous
//   conv_state:  (1, H * D, (ksize - 1) * dilation) half, updated in place, or nullopt
//   conv_stream: (1, H * D, state_len + 1) half out
//   delta:       (1, 1, H, D) float out
void ple_step_cpu
(
    const at::Tensor& streams,
    const at::Tensor& emb,
    const at::Tensor& key_w,
    const at::Tensor& value_w,
    const at::Tensor& norm_key_w,
    const at::Tensor& norm_query_w,
    const at::Tensor& norm_conv_w,
    double gate_scale,
    double eps,
    double constant_scale,
    double constant_bias,
    const at::Tensor& conv_w,
    const std::optional<at::Tensor>& conv_state,
    at::Tensor& conv_stream,
    at::Tensor& delta,
    int64_t conv_dilation
);

// True when the AVX-512 tier this kernel needs is present at runtime.
bool ple_step_cpu_has_avx512();
