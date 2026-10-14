#pragma once

#include <ATen/Tensor.h>

// Fused CPU router for the "std" routing type (work package C5): one C++ call per layer replaces
// _routing_std_torch's ~10 torch ops. Mirrors its math exactly (modules/block_sparse_mlp_routing.py,
// the established parity reference):
//   logits = (y.float() @ gate.float()).half()      fp16 GEMV, fp32 accumulation, result rounded to half
//   l = logits.float(); top = l.topk(k)             largest, sorted; tie order must match torch
//   e = (top.values - max).exp(); w = e / (e.sum() + 1e-20); w.half()
// The fp16 rounding of the logits before top-k/softmax is part of the contract. Single-threaded
// (small op; tiny ops lose under pool contention), AVX-512 dot kernels with runtime ISA detection
// (router_std_cpu_has_avx512); callers fall back to the torch path when it reports false.
//   gate_t:           (E, K) fp16, contiguous (RoutingCFG.gate_tensor.T.contiguous())
//   y:                (M, K) fp16, contiguous router input
//   selected_experts: (M, k) int64 out
//   routing_weights:  (M, k) fp16 out
void router_std_cpu
(
    const at::Tensor& gate_t,
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
);

// True when the AVX-512 tier this kernel needs is present at runtime.
bool router_std_cpu_has_avx512();
