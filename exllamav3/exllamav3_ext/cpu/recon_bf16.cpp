// Dequantize-to-BF16 for one mul1 EXL3 dense matrix (Plan 07 Option 1: CPU prefill tier). See
// recon_bf16.h for the contract. The format tables below are exact copies of moe_mul1.cpp's
// (pure integer math — a duplicate cannot diverge), and the transform order mirrors the fused
// kernel: weights carry suh before the k-side Hadamard and svh before the n-side pass, with one
// HAD_SCALE per pass, so the fp32 dequant equals the kernel's implied weight matrix up to the
// activation-side int8 quantization the fused path performs and this path does not.
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "recon_bf16.h"
#include "moe_mul1_internal.h"
#include <torch/extension.h>

#include <array>
#include <cmath>
#include <cstring>

namespace {

constexpr uint32_t R_MUL1_MULT = 0x83DCD12Du;
constexpr float R_HAD_SCALE = 0.088388347648f;   // 1/sqrt(128), once per Hadamard pass

constexpr int rate_k2(int bits, bool hb) { return 2 * bits + (hb ? 1 : 0); }
constexpr int tile_words32(int bits, bool hb) { return 4 * rate_k2(bits, hb); }
constexpr int state_end(int bits, bool hb, int t)
{
    return (((t + 1) * rate_k2(bits, hb)) >> 1) + 128 * rate_k2(bits, hb);
}

// Matches tensor-core permutation baked into the EXL3 tile storage format (inverse table)
constexpr std::array<uint16_t, 256> make_tc_perm_inv()
{
    std::array<uint16_t, 256> perm{};
    std::array<uint16_t, 256> inv{};
    for (int t = 0; t < 32; ++t)
    {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        perm[t * 8 + 0] = static_cast<uint16_t>(r0 * 16 + c0);
        perm[t * 8 + 1] = static_cast<uint16_t>(r1 * 16 + c0);
        perm[t * 8 + 2] = static_cast<uint16_t>(r2 * 16 + c0);
        perm[t * 8 + 3] = static_cast<uint16_t>(r3 * 16 + c0);
        perm[t * 8 + 4] = static_cast<uint16_t>(r0 * 16 + c1);
        perm[t * 8 + 5] = static_cast<uint16_t>(r1 * 16 + c1);
        perm[t * 8 + 6] = static_cast<uint16_t>(r2 * 16 + c1);
        perm[t * 8 + 7] = static_cast<uint16_t>(r3 * 16 + c1);
    }
    for (int i = 0; i < 256; ++i) inv[perm[i]] = static_cast<uint16_t>(i);
    return inv;
}

const std::array<uint16_t, 256> g_tc_inv = make_tc_perm_inv();

inline uint32_t load_u32_(const uint16_t* ptr, int index)
{
    uint32_t v;
    std::memcpy(&v, ptr + index * 2, sizeof(v));
    return v;
}

// One 16-bit state of tile slot t from the tile's bit ring (scalar reference of the fused
// kernel's dword-gather decode; integer-exact)
template <int bits, bool hb>
inline uint16_t decode_state_scalar(const uint16_t* packed, int t_offset)
{
    constexpr int words32 = tile_words32(bits, hb);
    const int b1 = state_end(bits, hb, t_offset);
    const int b0 = b1 - 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const uint64_t merged = (static_cast<uint64_t>(load_u32_(packed, (b0 / 32) % words32)) << 32) |
                            load_u32_(packed, ((b1 - 1) / 32) % words32);
    return static_cast<uint16_t>(merged >> shift);
}

// The mul1 codebook is affine in the byte-sum of state * MUL1_MULT
inline float decode_mul1_scalar(uint16_t state)
{
    const uint32_t x = static_cast<uint32_t>(state) * R_MUL1_MULT;
    const int sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    static const float k_inv = static_cast<float>(c10::Half(uint16_t(0x1eee), c10::Half::from_bits()));
    return (static_cast<float>(sum) - 510.0f) * k_inv;
}

// In-place 128-wide Hadamard butterfly over elements at stride `stride` starting at v (the
// n-side pass walks a column of a row-major [n, k] buffer)
inline void hadamard_128_stride(float* v, int stride)
{
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
            for (int i = 0; i < width; ++i)
            {
                float* pa = v + (base + i) * stride;
                float* pb = v + (base + width + i) * stride;
                const float a = *pa;
                const float b = *pb;
                *pa = a + b;
                *pb = a - b;
            }
}

inline float half_to_float(at::Half h) { return static_cast<float>(h); }

template <int bits, bool hb>
void dequant_impl(const MoeCpuMatrix& mat, float* w, int g)
{
    constexpr int words = 8 * rate_k2(bits, hb);
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    const size_t ring = words;

    // Decode: tile (kt, nt) covers out-rows nt*16..+16 and in-cols kt*16..+16; weight (row, col)
    // lives in tile slot g_tc_inv[row * 16 + col]. The packed layout permutes tiles within
    // g-groups: stored index (nt/g) * tiles_k * g + kt * g + nt % g.
    for (int nt = 0; nt < tiles_n; ++nt)
        for (int kt = 0; kt < tiles_k; ++kt)
        {
            const size_t stored = g
                ? static_cast<size_t>(nt / g) * tiles_k * g + static_cast<size_t>(kt) * g + (nt % g)
                : static_cast<size_t>(kt) * tiles_n + nt;   // native: trellis is [k/16, n/16, words]
            const uint16_t* tile = mat.trellis + stored * words;
            float* base = w + static_cast<size_t>(nt * 16) * mat.k + static_cast<size_t>(kt) * 16;
            for (int row = 0; row < 16; ++row)
            {
                float* out_row = base + static_cast<size_t>(row) * mat.k;
                for (int col = 0; col < 16; ++col)
                {
                    const int t = g_tc_inv[row * 16 + col];
                    out_row[col] = decode_mul1_scalar(decode_state_scalar<bits, hb>(tile, t))
                                   * half_to_float(mat.suh[kt * 16 + col]);
                }
            }
        }

    // k-side Hadamard per 128-block, scale once (mirrors the input-side x transform: values
    // carry suh before the butterfly, HAD_SCALE after)
    for (int k0 = 0; k0 < mat.k; k0 += 128)
        for (int n = 0; n < mat.n; ++n)
        {
            float* v = w + static_cast<size_t>(n) * mat.k + k0;
            hadamard_128_stride(v, 1);
            for (int i = 0; i < 128; ++i) v[i] *= R_HAD_SCALE;
        }

    // svh into every row before the n-side pass
    for (int n = 0; n < mat.n; ++n)
    {
        const float sv = half_to_float(mat.svh[n]);
        float* row = w + static_cast<size_t>(n) * mat.k;
        for (int k = 0; k < mat.k; ++k) row[k] *= sv;
    }

    // n-side Hadamard per 128-block (output-side transform), scale once
    for (int n0 = 0; n0 < mat.n; n0 += 128)
        for (int k = 0; k < mat.k; ++k)
        {
            float* v = w + static_cast<size_t>(n0) * mat.k + k;
            hadamard_128_stride(v, mat.k);
            for (int i = 0; i < 128; ++i) v[static_cast<size_t>(i) * mat.k] *= R_HAD_SCALE;
        }
}

}  // namespace

at::Tensor exl3_cpu_dequant_bf16
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    bool swizzled
)
{
    TORCH_CHECK(trellis.device().is_cpu() && suh.device().is_cpu() && svh.device().is_cpu(),
                "exl3_cpu_dequant_bf16: tensors must be on CPU");
    TORCH_CHECK(trellis.scalar_type() == at::kShort && trellis.is_contiguous(),
                "trellis must be contiguous CPU int16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf && suh.is_contiguous()
                    && svh.scalar_type() == at::kHalf && svh.is_contiguous(),
                "suh/svh must be contiguous CPU half");

    MoeCpuMatrix mat = exl3_cpu_mul1_make_matrix(trellis, suh, svh, nullptr, swizzled);
    const double K_tile = mat.bits + (mat.hb ? 0.5 : 0.0);
    TORCH_CHECK(std::fabs(K - K_tile) < 1e-6,
                "K (", K, ") does not match the trellis tile width (", K_tile, ")");
    // mat.swz is the caller-ANDed band flag (0/1), not the group: the group for this rate decides
    // the tile permutation, and the AVX2 group (2) implies the planar dword repack, which this
    // scalar reference decode does not implement
    TORCH_CHECK(!exl3_moe_cpu_planar_layout(K_tile),
                "exl3_cpu_dequant_bf16: the AVX2 planar packed layout (band-2 rates on AVX2-only "
                "hosts) is not supported; this tier targets the native/band-8 layouts");
    const int grp = mat.swz ? exl3_moe_cpu_swizzle_group(K_tile) : 0;
    TORCH_CHECK(mat.k % 128 == 0 && mat.n % 128 == 0,
                "exl3_cpu_dequant_bf16: in/out features must be multiples of 128 (Hadamard blocks), got ",
                mat.k, "x", mat.n);
    TORCH_CHECK(suh.numel() == mat.k && svh.numel() == mat.n, "suh/svh length mismatch");

    at::Tensor wf = at::empty({mat.n, mat.k},
                              at::TensorOptions().dtype(at::kFloat).device(trellis.device()));
    float* wp = wf.data_ptr<float>();

    switch (mat.bits)
    {
#define R_BF16_CASE(B)                                                    \
        case B:                                                           \
            if (mat.hb) dequant_impl<B, true>(mat, wp, grp);              \
            else dequant_impl<B, false>(mat, wp, grp);                    \
            break;
        R_BF16_CASE(1) R_BF16_CASE(2) R_BF16_CASE(3) R_BF16_CASE(4)
        R_BF16_CASE(5) R_BF16_CASE(6) R_BF16_CASE(7) R_BF16_CASE(8)
#undef R_BF16_CASE
        default: TORCH_CHECK(false, "exl3_cpu_dequant_bf16: unsupported bitrate ", mat.bits);
    }

    // fp32 dequant -> BF16, round-to-nearest-even (c10::BFloat16's scalar conversion), emitted
    // as u16 payloads — Zen 5 has no AVX512_FP16 and the vectorized narrow is deferred perf work
    at::Tensor out = at::empty({mat.n, mat.k},
                               at::TensorOptions().dtype(at::kBFloat16).device(trellis.device()));
    const int64_t total = mat.n * mat.k;
    c10::BFloat16* op = reinterpret_cast<c10::BFloat16*>(out.data_ptr());
    for (int64_t i = 0; i < total; ++i) op[i] = c10::BFloat16(wp[i]);
    return out;
}

#else  // !x86: the mul1 format is only decoded by the x86 kernels
#include "recon_bf16.h"
#include <torch/extension.h>

at::Tensor exl3_cpu_dequant_bf16
(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    bool
)
{
    TORCH_CHECK(false, "exl3_cpu_dequant_bf16 is not available on this CPU architecture (x86 AVX2/AVX-512 only)");
    return {};
}
#endif
