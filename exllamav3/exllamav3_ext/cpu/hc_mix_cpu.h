#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// Fused CPU forward of the GatedResidual hyper-connection mix (work package B3). One call per
// site computes what the CUDA decode pair (gr_dots/gr_finalize in hc_mix.cu) and the torch
// reference (GatedResidual._mix_ref) compute, per row of the (R, H, D) fp32 stream stack:
//   dots[r, j, h] = <streams[r, h] * w[h], fn[j, h]>   row M = per-stream sum of squares
//   rmr[r, h]     = rsqrt(sumsq / D + rms_eps)
//   t[r, i]       = silu(sum_h rmr[r, h] * dots[r, i, h] / H)
//   post[r, h]    = 2 * sigmoid(sum_h' rmr[r, h'] * dots[r, LR + h, h'] / H)   (site form)
//   mixed[r, d]   = sum_h sigmoid(<t[r], up[h*D + d]>) * rmr[r, h] * w[h, d] * streams[r, h, d] / H
// The norm weight rides on the stream side of the dots (the CPU path has no predecessor-weighted
// stream copy), which is the same math by linearity, and is taken in fp32 (the reference's
// norm_w, incl +1) so the fp32-internal results match it to accumulation-order noise only; the
// CUDA kernel instead applies the half weight. All accumulation is fp32; fn/up are fp16 and
// converted in the inner loops. Two phases per call (dots, finalize) over a private floating
// spin-parked pool (no core pinning: the bursts must not preempt the moe pool's pinned workers,
// see the Pool note in the source) with one in-function barrier; workers are spawned with the
// process's load-time affinity mask (they would otherwise inherit the moe/linear pool's pinned
// caller and share its one core with the master), and joins that stay over 4 ms cap the pool at
// one worker (EXL3_HC_MIX_CPU_ADAPT=0 disables). Threads = EXL3_HC_MIX_CPU_THREADS, else a
// quarter of the logical CPU count (the bursts stop scaling
// well past ~6 workers). Requires AVX-512F/BW/VL + FMA + F16C (gr_mix_cpu_has_avx512); callers
// fall back to _mix_ref when it reports false or the tensors are ineligible.
//   streams: (R, H, D) float, CPU, contiguous, H = 4, R in [1, 8]
//   fn:      (M, H * D) half, or bfloat16 (dispatch on dtype: same math, fp32 accumulation;
//            bf16 tables halve the per-call table stream, python-side EXL3_HC_BF16 repack),
//            M = LR + (post ? H : 0)
//   up:      (H * D, LR) same dtype as fn, checkpoint orientation (not the CUDA kernel's repacked upt)
//   w:       (H * D) float norm weight (incl +1)
//   post:    (R, H) float out, or none (final-mixer form)
//   mixed:   (R, D) half out
void gr_mix_cpu
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& up,
    const at::Tensor& w,
    double rms_eps,
    const std::optional<at::Tensor>& post,
    at::Tensor& mixed
);

// True when the AVX-512 tier this kernel needs is present at runtime.
bool gr_mix_cpu_has_avx512();
