#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "linear_mul1.h"
#include "moe_mul1_internal.h"
#include <torch/extension.h>

#include <cmath>

at::Tensor exl3_cpu_linear_forward
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& x,
    bool want_fp32_out,
    bool swizzled
)
{
    TORCH_CHECK(trellis.device().is_cpu() && suh.device().is_cpu() && svh.device().is_cpu()
                    && x.device().is_cpu(),
                "exl3_cpu_linear_forward: tensors must be on CPU");
    TORCH_CHECK(trellis.scalar_type() == at::kShort && trellis.is_contiguous(),
                "trellis must be contiguous CPU int16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf && suh.is_contiguous()
                    && svh.scalar_type() == at::kHalf && svh.is_contiguous(),
                "suh/svh must be contiguous CPU half");
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "x must be contiguous CPU half [M, in_features]");
    if (bias)
        TORCH_CHECK(bias->device().is_cpu() && bias->scalar_type() == at::kHalf
                        && bias->is_contiguous(),
                    "bias must be contiguous CPU half");

    MoeCpuMatrix mat = exl3_cpu_mul1_make_matrix(trellis, suh, svh, bias ? &*bias : nullptr, swizzled);
    const double K_tile = mat.bits + (mat.hb ? 0.5 : 0.0);
    TORCH_CHECK(std::fabs(K - K_tile) < 1e-6,
                "K (", K, ") does not match the trellis tile width (", K_tile, ")");
    TORCH_CHECK(x.size(1) == mat.k, "x in_features (", x.size(1), ") does not match trellis (", mat.k, ")");
    TORCH_CHECK(suh.numel() == mat.k && svh.numel() == mat.n, "suh/svh length mismatch");

    const int64_t m = x.size(0);
    at::Tensor out = at::empty({m, mat.n}, x.options().dtype(at::kFloat));
    exl3_cpu_mul1_matrix_forward(mat, reinterpret_cast<const at::Half*>(x.data_ptr()),
                                 static_cast<int>(m), out.data_ptr<float>(), 0);
    if (want_fp32_out) return out;
    return out.to(at::kHalf);
}

std::vector<at::Tensor> exl3_cpu_linear_forward_batch
(
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<double>& K,
    const std::vector<std::optional<at::Tensor>>& bias,
    const at::Tensor& x,
    const std::vector<bool>& want_fp32_out,
    const std::vector<bool>& swizzled
)
{
    const int nm = static_cast<int>(trellis.size());
    TORCH_CHECK(nm > 0 && static_cast<int>(suh.size()) == nm && static_cast<int>(svh.size()) == nm
                    && static_cast<int>(K.size()) == nm && static_cast<int>(bias.size()) == nm
                    && static_cast<int>(want_fp32_out.size()) == nm
                    && static_cast<int>(swizzled.size()) == nm,
                "exl3_cpu_linear_forward_batch: per-matrix argument count mismatch");
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kHalf && x.is_contiguous()
                    && x.dim() == 2,
                "x must be contiguous CPU half [M, in_features]");
    const int64_t m = x.size(0);

    std::vector<MoeCpuMatrix> mats;
    std::vector<at::Tensor> outs;
    std::vector<float*> outp;
    mats.reserve(nm); outs.reserve(nm); outp.reserve(nm);
    for (int i = 0; i < nm; ++i)
    {
        const at::Tensor& t = trellis[i];
        const at::Tensor& su = suh[i];
        const at::Tensor& sv = svh[i];
        TORCH_CHECK(t.device().is_cpu() && su.device().is_cpu() && sv.device().is_cpu(),
                    "matrix ", i, ": tensors must be on CPU");
        TORCH_CHECK(t.scalar_type() == at::kShort && t.is_contiguous(),
                    "matrix ", i, ": trellis must be contiguous CPU int16");
        TORCH_CHECK(su.scalar_type() == at::kHalf && su.is_contiguous()
                        && sv.scalar_type() == at::kHalf && sv.is_contiguous(),
                    "matrix ", i, ": suh/svh must be contiguous CPU half");
        if (bias[i])
            TORCH_CHECK(bias[i]->device().is_cpu() && bias[i]->scalar_type() == at::kHalf
                            && bias[i]->is_contiguous(),
                        "matrix ", i, ": bias must be contiguous CPU half");
        MoeCpuMatrix mat = exl3_cpu_mul1_make_matrix(t, su, sv, bias[i] ? &*bias[i] : nullptr, swizzled[i]);
        const double K_tile = mat.bits + (mat.hb ? 0.5 : 0.0);
        TORCH_CHECK(std::fabs(K[i] - K_tile) < 1e-6,
                    "matrix ", i, ": K (", K[i], ") does not match the trellis tile width (", K_tile, ")");
        TORCH_CHECK(x.size(1) == mat.k,
                    "matrix ", i, ": x in_features (", x.size(1), ") does not match trellis (", mat.k, ")");
        TORCH_CHECK(su.numel() == mat.k && sv.numel() == mat.n, "matrix ", i, ": suh/svh length mismatch");
        outs.emplace_back(at::empty({m, mat.n}, x.options().dtype(at::kFloat)));
        outp.push_back(outs.back().data_ptr<float>());
        mats.push_back(mat);
    }

    exl3_cpu_mul1_batch_forward(mats, reinterpret_cast<const at::Half*>(x.data_ptr()),
                                static_cast<int>(m), outp.data(), 0);
    for (int i = 0; i < nm; ++i)
        if (!want_fp32_out[i]) outs[i] = outs[i].to(at::kHalf);
    return outs;
}

#else  // !x86: stubbed like moe_mul1 (AVX-only kernels)
#include "linear_mul1.h"
#include <torch/extension.h>

at::Tensor exl3_cpu_linear_forward
(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    const std::optional<at::Tensor>&,
    const at::Tensor&,
    bool,
    bool
)
{
    TORCH_CHECK(false, "exl3_cpu_linear_forward is not available on this CPU architecture (x86 AVX2/AVX-512 only)");
    return {};
}

std::vector<at::Tensor> exl3_cpu_linear_forward_batch
(
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<double>&,
    const std::vector<std::optional<at::Tensor>>&,
    const at::Tensor&,
    const std::vector<bool>&,
    const std::vector<bool>&
)
{
    TORCH_CHECK(false, "exl3_cpu_linear_forward_batch is not available on this CPU architecture (x86 AVX2/AVX-512 only)");
    return {};
}
#endif
