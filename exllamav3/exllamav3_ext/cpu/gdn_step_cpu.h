#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <optional>

// Fused CPU decode step of the split-projection GatedDeltaNet (work package B4). One call per
// layer per token computes the whole middle section between the qkv/z projection and the output
// projection, mirroring modules/gated_delta_net_fn/cpu.py (gdn_fused_op_2_cpu,
// causal_conv1d_update_cpu) and gated_delta_rule.py (torch_recurrent_gated_delta_rule_cpu at
// seqlen = 1) plus GatedRMSNorm._forward_cpu:
//   b = x @ wb (+ bb), a = x @ wa (+ ba)                    fp16 GEMVs, fp32 accumulation
//   beta = bf16(sigmoid(b) * beta_scale)
//   g    = -softplus(a + dt_bias, thr 20) * exp(a_log)      fp32
//   mixed = bf16(silu(conv([state | qkv])))                  state window <- last kdim inputs
//   scan  = one token of the grouped delta rule              q/k l2norm eps 1e-6, undecayed
//           state readback, state <- state * decay + k u^T, out = bf16(scale q^T state)
//   out   = bf16(scan out) -> rmsnorm -> weight (+ bias) -> gate(z) -> fp16
// All accumulation fp32; state rows and conv ring rows are updated in place. Three phases
// (gates+conv, scan, norm) over a private floating spin-parked pool (hc_mix_cpu.cpp Pool) with
// in-function barriers; threads = EXL3_GDN_STEP_CPU_THREADS, else 4. Requires
// AVX-512F/BW/VL/DQ + FMA + F16C (gdn_step_cpu_has_avx512); callers fall back to the torch
// path when it reports false or the tensors are ineligible.
//   x:             (bsz, 1, hidden) half
//   qkv:           (bsz, 1, fdim_qkv) half, fdim_qkv = 2 * nk * dk + nv * dv
//   z:             (bsz, 1, nv, dv) float or half (z_is_half)
//   conv_state:    (mb, fdim_qkv, kdim + hist) bf16, updated in place (rows conv_slots)
//   conv_w:        (kdim, fdim_qkv) float, tap-major
//   conv_b:        (fdim_qkv,) float, or nullopt
//   wb/wa:         (hidden, nv) half, or nullopt (bb/ba: (nv,) half biases)
//   dt_bias/a_log: (nv,) float
//   recurrent_state: (rows, 1, nv, dk, dv) float or half, updated in place (rows scan_slots);
//                    fp16 pools (recurrent_dtype = half, the default since 02446529) convert on
//                    the load and round RNE once per step on the store; the scan math is fp32
//   conv_slots / scan_slots: (bsz,) int32 or int64 pool rows
//   norm_w:        (dv,) float
//   out:           (bsz, nv * dv) half
void gdn_step_cpu
(
    const at::Tensor& x,                // (bsz, 1, hidden) fp16, layer input
    const at::Tensor& qkv,              // (bsz, 1, fdim_qkv) fp16, split qkv projection output
    const at::Tensor& z,                // (bsz, 1, nv, dv) fp32 or fp16, output gate
    const at::Tensor& conv_state,       // (max_batch, fdim_qkv, kdim + hist) bf16, updated in place
    const at::Tensor& conv_w,           // (kdim, fdim_qkv) fp32
    const std::optional<at::Tensor>& conv_b,    // (fdim_qkv,) fp32 or nullopt
    const std::optional<at::Tensor>& wb,        // (hidden, nv) fp16 b projection, or nullopt
    const std::optional<at::Tensor>& bb,        // (nv,) fp16 b bias, or nullopt
    const std::optional<at::Tensor>& wa,        // (hidden, nv) fp16 a projection, or nullopt
    const std::optional<at::Tensor>& ba,        // (nv,) fp16 a bias, or nullopt
    const at::Tensor& dt_bias,          // (nv,) fp32
    const at::Tensor& a_log,            // (nv,) fp32
    const at::Tensor& recurrent_state,  // (rows, 1, nv, dk, dv) fp32, scan rows updated in place
    const at::Tensor& conv_slots,       // (bsz,) int conv ring rows
    const at::Tensor& scan_slots,       // (bsz,) int recurrent state rows
    const at::Tensor& norm_w,           // (dv,) fp32
    at::Tensor& out,                    // (bsz, nv * dv) half, gated norm output (o_proj input)
    double beta_scale,
    double rms_eps,
    double norm_bias_add,
    bool gate_sigmoid,
    bool z_is_half,
    int num_threads
);

// True when the AVX-512 tier this kernel needs is present at runtime.
bool gdn_step_cpu_has_avx512();
