// CPU-only build of exllamav3_ext (util/cuda_flags.py, cpu_extension_sources): registers the
// subset of the module that is pure C++ -- the safetensors loader (stloader), the Hadamard
// matrices (hadamard.cpp), the EXL3 mul1 CPU MoE kernels (cpu/moe_mul1.cpp) and the SAM suffix
// automaton (sam.cpp). Everything GPU-only keeps its Python-side entry points absent, except the
// HAS_* capability flags, which are defined false so the package imports and the modules fall back
// or fail per call instead of at import. m.def lines mirror bindings.cpp; keep them in sync.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stloader.h"
#include "hadamard.h"
#include "cpu/moe_mul1.h"
#include "cpu/hc_mix_cpu.h"
#include "cpu/gdn_step_cpu.h"
#include "cpu/sdpa_paged_cpu.h"
#include "cpu/ple_step_cpu.h"
#include "cpu/fused_sampler_cpu.h"
#include "cpu/router_cpu.h"

#include "ngram.cuh"
#include "sam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // GPU-only kernel families, compiled out of this build; Python checks these before taking a
    // path that needs them (mirrors bindings.cpp, all false here)
    m.attr("HAS_GR_MIX_TILED") = false;         // gr_mix_tiled, gr_mix_tiled_slices
    m.attr("HAS_DET_GEMM") = false;             // routing_gemm_det, det_quant_weight, det_math_test
    m.attr("HAS_HGEMM_F16ACC") = false;         // hgemm_f16acc, hgemm_f16acc_status
    m.attr("HAS_DFLASH2") = false;              // dflash2_dynconv, dflash2_selector_walk, dflash2_topk

    // Marks this as the reduced CPU-only module (no CUDA/HIP toolchain was involved in building it)
    m.attr("EXLLAMA_CPU_BUILD") = true;

    m.def("stloader_read", &stloader_read, "stloader_read");
    m.def("stloader_open_file", &stloader_open_file, "stloader_open_file");
    m.def("stloader_close_file", &stloader_close_file, "stloader_close_file");
    py::class_<TensorLoadJob>(m, "TensorLoadJob")
        .def(py::init<std::vector<uintptr_t>, size_t, size_t, uintptr_t, size_t, bool, bool, bool, int>());
    m.def("stloader_deferred_cpu", &stloader_deferred_cpu, py::arg("jobs"));
    // stloader_deferred_cuda is not registered: the CUDA engine is stubbed out in this build
    // (cpu_compat/), and stloader_deferred_batch routes to it only for rows flagged
    // STLOADER_FLAG_CUDA, which a CPU-only load never produces
    m.def("stloader_deferred_batch", &stloader_deferred_batch, py::arg("file_handles"), py::arg("loads"), py::arg("max_chunk_size"));

    m.def("had_paley", &had_paley, "had_paley");
    m.def("had_paley2", &had_paley2, "had_paley2");

    m.def("exl3_moe_cpu_make_layer", &exl3_moe_cpu_make_layer, "exl3_moe_cpu_make_layer");
    m.def("exl3_moe_cpu_free_layer", &exl3_moe_cpu_free_layer, "exl3_moe_cpu_free_layer");
    m.def("exl3_moe_cpu_forward", &exl3_moe_cpu_forward, "exl3_moe_cpu_forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("exl3_moe_cpu_has_avx2", &exl3_moe_cpu_has_avx2, "exl3_moe_cpu_has_avx2");
    m.def("exl3_moe_cpu_has_avx512_bw", &exl3_moe_cpu_has_avx512_bw, "exl3_moe_cpu_has_avx512_bw");
    m.def("exl3_moe_cpu_has_avx512_vnni", &exl3_moe_cpu_has_avx512_vnni, "exl3_moe_cpu_has_avx512_vnni");
    m.def("exl3_moe_cpu_has_avx512_vbmi", &exl3_moe_cpu_has_avx512_vbmi, "exl3_moe_cpu_has_avx512_vbmi");
    m.def("exl3_moe_cpu_set_prof", &exl3_moe_cpu_set_prof, "exl3_moe_cpu_set_prof");
    m.def("exl3_moe_cpu_pool_prime", &exl3_moe_cpu_pool_prime, "exl3_moe_cpu_pool_prime");
    m.def("exl3_moe_cpu_pool_stress", &exl3_moe_cpu_pool_stress, "exl3_moe_cpu_pool_stress");
    m.def("exl3_moe_cpu_core_order", &exl3_moe_cpu_core_order, "exl3_moe_cpu_core_order");
    // exl3_moe_flag_write/flag_wait/worker_run (cpu/moe_handoff.cu) are not registered: they drive
    // the CUDA-stream handoff, which the CPU-only path bypasses. set_memops only configures that
    // handoff's wait primitive, and moe_cpu_host.py calls it at import, so it exists as a no-op
    m.def("exl3_moe_cpu_set_memops", [](bool) {}, "exl3_moe_cpu_set_memops");
    m.def("exl3_moe_cpu_swizzle_group", &exl3_moe_cpu_swizzle_group, "exl3_moe_cpu_swizzle_group");
    m.def("exl3_moe_cpu_planar_layout", &exl3_moe_cpu_planar_layout, "exl3_moe_cpu_planar_layout");

    // GPU-only kernels whose function objects Python binds at import time (modules/
    // moe_batch_recon.py builds its activation dispatch from them): registered as stubs with the
    // real signatures (activation.cuh) so the module imports; a call fails cleanly
    auto act_stub = [](const char* name)
    {
        TORCH_CHECK(false, name, " is a CUDA kernel and is not part of the CPU-only build");
    };
    m.def("silu_mul", [act_stub](const at::Tensor&, const at::Tensor&, at::Tensor&, float) { act_stub("silu_mul"); }, "silu_mul");
    m.def("gelu_mul", [act_stub](const at::Tensor&, const at::Tensor&, at::Tensor&, float) { act_stub("gelu_mul"); }, "gelu_mul");
    m.def("silu_oai_mul", [act_stub](const at::Tensor&, const at::Tensor&, at::Tensor&, float) { act_stub("silu_oai_mul"); }, "silu_oai_mul");
    m.def("relu2_mul", [act_stub](const at::Tensor&, const at::Tensor&, at::Tensor&, float) { act_stub("relu2_mul"); }, "relu2_mul");
    m.def("relu_mul", [act_stub](const at::Tensor&, const at::Tensor&, at::Tensor&, float) { act_stub("relu_mul"); }, "relu_mul");

    // [agent B] exl3_cpu_linear_forward lands here

    m.def("gr_mix_cpu", &gr_mix_cpu, "gr_mix_cpu",
          py::call_guard<py::gil_scoped_release>());
    m.def("gr_mix_cpu_has_avx512", &gr_mix_cpu_has_avx512, "gr_mix_cpu_has_avx512");

    m.def("fused_sampler_cpu", &fused_sampler_cpu, "fused_sampler_cpu",
          py::call_guard<py::gil_scoped_release>());
    m.attr("FUSED_SAMPLER_MAX_BLOCKS") = 128;      // unused by the CPU kernel, call parity
    m.attr("FUSED_SAMPLER_HIST_STRIDE") = 2 * 1024 * 12 + 64;

    m.def("gdn_step_cpu", &gdn_step_cpu, "gdn_step_cpu",
          py::call_guard<py::gil_scoped_release>());
    m.def("gdn_step_cpu_has_avx512", &gdn_step_cpu_has_avx512, "gdn_step_cpu_has_avx512");

    m.def("sdpa_paged_cpu", &sdpa_paged_cpu, "sdpa_paged_cpu",
          py::arg("q"), py::arg("k_new"), py::arg("v_new"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("block_table"), py::arg("cache_seqlens"),
          py::arg("scale"), py::arg("out"),
          py::call_guard<py::gil_scoped_release>());
    m.def("sdpa_paged_cpu_has_avx512", &sdpa_paged_cpu_has_avx512, "sdpa_paged_cpu_has_avx512");

    m.def("ple_step_cpu", &ple_step_cpu, "ple_step_cpu",
          py::call_guard<py::gil_scoped_release>());
    m.def("ple_step_cpu_has_avx512", &ple_step_cpu_has_avx512, "ple_step_cpu_has_avx512");

    m.def("router_std_cpu", &router_std_cpu, "router_std_cpu",
          py::call_guard<py::gil_scoped_release>());
    m.def("router_std_cpu_has_avx512", &router_std_cpu_has_avx512, "router_std_cpu_has_avx512");

    m.def("ngram_hash_cpu", &ngram_hash_cpu, "ngram_hash_cpu");
#ifndef _WIN32
    m.def("ngram_gather_cpu", &ngram_gather_cpu, "ngram_gather_cpu");
#endif
    m.def("ngram_dequant", &ngram_dequant, "ngram_dequant");
    #include "cpu/linear_mul1.h"
    m.def("exl3_cpu_linear_forward", &exl3_cpu_linear_forward, "exl3_cpu_linear_forward",
          py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("K"), py::arg("bias"),
          py::arg("x"), py::arg("want_fp32_out"), py::arg("swizzled") = false,
          py::call_guard<py::gil_scoped_release>());
    m.def("exl3_cpu_linear_forward_batch", &exl3_cpu_linear_forward_batch,
          "exl3_cpu_linear_forward_batch",
          py::arg("trellis"), py::arg("suh"), py::arg("svh"), py::arg("K"), py::arg("bias"),
          py::arg("x"), py::arg("want_fp32_out"), py::arg("swizzled"),
          py::call_guard<py::gil_scoped_release>());

    #include "sam_bc.h"
}
