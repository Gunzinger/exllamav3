#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "ple_step_cpu.h"
#include <c10/util/Half.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

// Fused PLE decode step on CPU (ple_step_cpu.h): the whole forward_streams_cpu math for one
// token -- key/value GEMVs over the fp16 projection weights, the grouped key/query norms, the
// signed-sqrt gate, the half-rounded normed conv column, the dilated short conv over
// [state | new] and the delta sum -- with every fp16 rounding point of the torch path in
// place; only the fp32 accumulation order differs (last-ulp class). Single-threaded like
// gdn_step_cpu's default.

#define PLE_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))

// -------------------------------------------------------------------------------------------
//   Primitives (gdn_step_cpu.cpp's)
// -------------------------------------------------------------------------------------------

PLE_TARGET_AVX512
inline __m512 ple_exp512(__m512 d)
{
    // Sleef u10 expf (sleefsimdsp.c xexpf) -- the routine torch's fp32 elementwise
    // exp/sigmoid/silu dispatch to on AVX-512, so the gate sigmoid and conv silu round
    // identically: q = rint(d log2 e); s = d - q ln2 (hi/lo split, fma form); degree-6 poly;
    // u * 2^(q>>1) * 2^(q - (q>>1)); zero below -104, infinity above 100
    const __m512 rln2 = _mm512_set1_ps(1.442695040888963407359924681001892137426645954152985934135449406931f);
    const __m512 nl2u = _mm512_set1_ps(-0.693145751953125f);
    const __m512 nl2l = _mm512_set1_ps(-1.428606765330187045e-06f);
    __m512i q = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(d, rln2),
                                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m512 qf = _mm512_cvtepi32_ps(q);
    __m512 s = _mm512_fmadd_ps(qf, nl2u, d);
    s = _mm512_fmadd_ps(qf, nl2l, s);
    __m512 u = _mm512_set1_ps(0.000198527617612853646278381f);
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(0.00139304355252534151077271f));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(0.00833336077630519866943359f));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(0.0416664853692054748535156f));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(0.166666671633720397949219f));
    u = _mm512_fmadd_ps(u, s, _mm512_set1_ps(0.5f));
    u = _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_fmadd_ps(_mm512_mul_ps(s, s), u, s));
    __m512i qh = _mm512_srai_epi32(q, 1);
    __m512 u1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(qh, _mm512_set1_epi32(0x7f)), 23));
    __m512 u2 = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_add_epi32(_mm512_sub_epi32(q, qh), _mm512_set1_epi32(0x7f)), 23));
    u = _mm512_mul_ps(_mm512_mul_ps(u, u1), u2);
    // d < -104 -> 0, d > 100 -> +inf
    u = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(d, _mm512_set1_ps(-104.0f), _CMP_GE_OQ), u);
    u = _mm512_mask_mov_ps(u, _mm512_cmp_ps_mask(d, _mm512_set1_ps(100.0f), _CMP_GT_OQ),
                           _mm512_set1_ps(__builtin_inff()));
    return u;
}

PLE_TARGET_AVX512
inline __m512 ple_sigmoid512(__m512 x)
{
    return _mm512_div_ps(_mm512_set1_ps(1.0f),
                         _mm512_add_ps(_mm512_set1_ps(1.0f), ple_exp512(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

PLE_TARGET_AVX512
inline __m512 ple_silu512(__m512 x)
{
    return _mm512_mul_ps(x, ple_sigmoid512(x));
}

inline float ple_bf16_to_f32(uint16_t h)
{
    uint32_t u = ((uint32_t) h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

PLE_TARGET_AVX512
inline void ple_half_store(at::Half* dst, __m512 v)
{
    // f32 -> fp16 RNE per lane (vcvtps2ph), packed halves
    _mm256_storeu_si256((__m256i*) dst, _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

PLE_TARGET_AVX512
inline float ple_sumsq(const float* x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(x + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i)
        s = std::fma(x[i], x[i], s);
    return s;
}

// out[j] = f32(half(sum_r x[r] * f32(w[r, j]))): the fp16 matrix is streamed once, the fp32
// accumulators live in L1 one column tile at a time, and the result is rounded to fp16 and
// widened (the .to(torch.half) after the reference's fp32 GEMV). cols must be a multiple of 16
PLE_TARGET_AVX512
static void ple_gemv_h16(const uint16_t* w, const float* x, int rows, int cols, float* out)
{
    constexpr int TILE = 512;
    alignas(64) float acc[TILE];
    for (int j0 = 0; j0 < cols; j0 += TILE)
    {
        const int tn = std::min(cols - j0, TILE);
        std::memset(acc, 0, (size_t) tn * sizeof(float));
        for (int r = 0; r < rows; ++r)
        {
            const uint16_t* wr = w + (size_t) r * cols + j0;
            const __m512 xv = _mm512_set1_ps(x[r]);
            for (int j = 0; j < tn; j += 16)
                _mm512_storeu_ps(acc + j, _mm512_fmadd_ps(xv,
                    _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (wr + j))),
                    _mm512_loadu_ps(acc + j)));
        }
        for (int j = 0; j < tn; j += 16)
            _mm512_storeu_ps(out + j0 + j,
                _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_loadu_ps(acc + j), _MM_FROUND_TO_NEAREST_INT)));
    }
}

// -------------------------------------------------------------------------------------------
//   Kernel
// -------------------------------------------------------------------------------------------

struct PleCtx
{
    // inputs
    const float* streams;          // (H, D) fp32
    const uint16_t* emb;           // (ple_dim,) fp16
    const uint16_t* key_w;         // (ple_dim, H * D) fp16
    const uint16_t* value_w;       // (ple_dim, D) fp16
    const void* nk_w;              // (H * D) fp16/bf16/fp32 norm weights
    const void* nq_w;
    const void* nc_w;
    int norm_dt;                   // 0 = fp16, 1 = bf16, 2 = fp32
    const uint16_t* conv_w;        // (H * D, ksize) fp16, channel-major
    uint16_t* conv_state;          // (H * D, state_len) fp16, updated in place, or null
    uint16_t* conv_stream;         // (H * D, state_len + 1) fp16 out
    float* delta;                  // (H, D) fp32 out
    // scratch
    float* embf;
    float* key;
    float* value;
    float* keyn;
    float* queryn;
    float* gated;
    float* nk;                     // norm weights widened to fp32, constant bias folded in
    float* nq;
    float* nc;
    at::Half* normed;              // half-rounded new conv column
    // dims / scalars
    int H, D, E, K, dil, win;
    float gate_scale, eps, scale, bias;
};

PLE_TARGET_AVX512
static void ple_widen(const void* src, int dt, int n, float bias, float* dst)
{
    if (dt == 2)
    {
        const float* s = (const float*) src;
        for (int i = 0; i < n; ++i) dst[i] = s[i] + bias;
    }
    else
    {
        const uint16_t* s = (const uint16_t*) src;
        for (int i = 0; i < n; ++i)
            dst[i] = (dt == 0 ? _cvtsh_ss(s[i]) : ple_bf16_to_f32(s[i])) + bias;
    }
}

PLE_TARGET_AVX512
static void ple_step_run(PleCtx& c)
{
    const int H = c.H, D = c.D, hc = H * D;
    const float eps = c.eps, scale = c.scale;
    int i = 0;
    for (; i + 16 <= c.E; i += 16)
        _mm512_storeu_ps(c.embf + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (c.emb + i))));
    for (; i < c.E; ++i) c.embf[i] = _cvtsh_ss(c.emb[i]);
    ple_gemv_h16(c.key_w, c.embf, c.E, hc, c.key);
    ple_gemv_h16(c.value_w, c.embf, c.E, D, c.value);
    ple_widen(c.nk_w, c.norm_dt, hc, c.bias, c.nk);
    ple_widen(c.nq_w, c.norm_dt, hc, c.bias, c.nq);
    ple_widen(c.nc_w, c.norm_dt, hc, c.bias, c.nc);

    for (int h = 0; h < H; ++h)
    {
        const float* kf = c.key + (size_t) h * D;
        const float* qf = c.streams + (size_t) h * D;
        const float rk = 1.0f / std::sqrt(ple_sumsq(kf, D) / (float) D + eps) * scale;
        const float rq = 1.0f / std::sqrt(ple_sumsq(qf, D) / (float) D + eps) * scale;
        float* kn = c.keyn + (size_t) h * D;
        float* qn = c.queryn + (size_t) h * D;
        int d = 0;
        for (; d + 16 <= D; d += 16)
        {
            const __m512 kk = _mm512_mul_ps(_mm512_loadu_ps(kf + d), _mm512_set1_ps(rk));
            const __m512 qq = _mm512_mul_ps(_mm512_loadu_ps(qf + d), _mm512_set1_ps(rq));
            _mm512_storeu_ps(kn + d, _mm512_mul_ps(kk, _mm512_loadu_ps(c.nk + (size_t) h * D + d)));
            _mm512_storeu_ps(qn + d, _mm512_mul_ps(qq, _mm512_loadu_ps(c.nq + (size_t) h * D + d)));
        }
        for (; d < D; ++d)
        {
            kn[d] = (kf[d] * rk) * c.nk[(size_t) h * D + d];
            qn[d] = (qf[d] * rq) * c.nq[(size_t) h * D + d];
        }
        __m512 gacc = _mm512_setzero_ps();
        d = 0;
        for (; d + 16 <= D; d += 16)
            gacc = _mm512_fmadd_ps(_mm512_loadu_ps(qn + d), _mm512_loadu_ps(kn + d), gacc);
        float gate = _mm512_reduce_add_ps(gacc);
        for (; d < D; ++d)
            gate = std::fma(qn[d], kn[d], gate);
        const float g = gate * c.gate_scale;
        const float a = std::sqrt(std::fmax(std::fabs(g), 1e-6f));
        const float ss = g > 0.0f ? a : (g < 0.0f ? -a : 0.0f);
        const float s = _mm512_cvtss_f32(ple_sigmoid512(_mm512_set1_ps(ss)));
        const __m512 sv = _mm512_set1_ps(s);
        float* gt = c.gated + (size_t) h * D;
        d = 0;
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(gt + d, _mm512_mul_ps(sv, _mm512_loadu_ps(c.value + d)));
        for (; d < D; ++d)
            gt[d] = s * c.value[d];
    }

    for (int h = 0; h < H; ++h)
    {
        const float* nf = c.gated + (size_t) h * D;
        const float rn = 1.0f / std::sqrt(ple_sumsq(nf, D) / (float) D + eps) * scale;
        at::Half* np = c.normed + (size_t) h * D;
        int d = 0;
        for (; d + 16 <= D; d += 16)
            ple_half_store(np + d, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(nf + d), _mm512_set1_ps(rn)),
                                                 _mm512_loadu_ps(c.nc + (size_t) h * D + d)));
        for (; d < D; ++d)
            np[d] = at::Half((nf[d] * rn) * c.nc[(size_t) h * D + d]);
    }

    // conv column stream [state | new column], dilated tap dot, silu, delta
    const int K = c.K, dil = c.dil, win = c.win, stride = win + 1;
    const uint16_t* normed16 = (const uint16_t*) c.normed;
    for (int j = 0; j < hc; ++j)
    {
        uint16_t* row = c.conv_stream + (size_t) j * stride;
        if (c.conv_state)
            std::memcpy(row, c.conv_state + (size_t) j * win, (size_t) win * sizeof(uint16_t));
        else
            std::memset(row, 0, (size_t) win * sizeof(uint16_t));
        row[win] = normed16[j];
    }
    {
        int j0 = 0;
        for (; j0 + 16 <= hc; j0 += 16)
        {
            // (xt * w).sum(-1): fp32 products added in tap order, then the single round to
            // half; silu reads the half-rounded y and its result rounds to half again
            alignas(64) float tv[16], wv[16];
            for (int u = 0; u < 16; ++u)
            {
                tv[u] = _cvtsh_ss(c.conv_stream[(size_t) (j0 + u) * stride]);
                wv[u] = _cvtsh_ss(c.conv_w[(size_t) (j0 + u) * K]);
            }
            __m512 y = _mm512_mul_ps(_mm512_loadu_ps(tv), _mm512_loadu_ps(wv));
            for (int k = 1; k < K; ++k)
            {
                for (int u = 0; u < 16; ++u)
                {
                    tv[u] = _cvtsh_ss(c.conv_stream[(size_t) (j0 + u) * stride + k * dil]);
                    wv[u] = _cvtsh_ss(c.conv_w[(size_t) (j0 + u) * K + k]);
                }
                y = _mm512_add_ps(y, _mm512_mul_ps(_mm512_loadu_ps(tv), _mm512_loadu_ps(wv)));
            }
            const __m512 yh = _mm512_cvtph_ps(_mm512_cvtps_ph(y, _MM_FROUND_TO_NEAREST_INT));
            const __m512 co = _mm512_cvtph_ps(_mm512_cvtps_ph(ple_silu512(yh), _MM_FROUND_TO_NEAREST_INT));
            _mm512_storeu_ps(c.delta + j0, _mm512_add_ps(_mm512_loadu_ps(c.gated + j0), co));
        }
        for (int j = j0; j < hc; ++j)
        {
            float y = _cvtsh_ss(c.conv_stream[(size_t) j * stride]) * _cvtsh_ss(c.conv_w[(size_t) j * K]);
            for (int k = 1; k < K; ++k)
                y += _cvtsh_ss(c.conv_stream[(size_t) j * stride + k * dil]) * _cvtsh_ss(c.conv_w[(size_t) j * K + k]);
            const float yh = _cvtsh_ss((uint16_t) _cvtss_sh(y, _MM_FROUND_TO_NEAREST_INT));
            const float sv = _mm512_cvtss_f32(ple_silu512(_mm512_set1_ps(yh)));
            c.delta[j] = c.gated[j] + _cvtsh_ss((uint16_t) _cvtss_sh(sv, _MM_FROUND_TO_NEAREST_INT));
        }
    }

    // conv ring in place: drop the oldest column, append the new one (xt[:, :, 1:])
    if (c.conv_state)
    {
        for (int j = 0; j < hc; ++j)
        {
            uint16_t* row = c.conv_state + (size_t) j * win;
            if (win > 1)
                std::memmove(row, row + 1, (size_t) (win - 1) * sizeof(uint16_t));
            if (win > 0)
                row[win - 1] = normed16[j];
        }
    }
}

// -------------------------------------------------------------------------------------------
//   Entry
// -------------------------------------------------------------------------------------------

bool ple_step_cpu_has_avx512()
{
    static const bool v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
                          __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return v;
}

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
)
{
    TORCH_CHECK(ple_step_cpu_has_avx512(), "ple_step_cpu: AVX-512 not available");
    TORCH_CHECK(streams.is_cpu() && emb.is_cpu() && key_w.is_cpu() && value_w.is_cpu() &&
                norm_key_w.is_cpu() && norm_query_w.is_cpu() && norm_conv_w.is_cpu() &&
                conv_w.is_cpu() && conv_stream.is_cpu() && delta.is_cpu(),
                "ple_step_cpu: tensors must be on CPU");
    TORCH_CHECK(streams.scalar_type() == at::kFloat && streams.is_contiguous() && streams.dim() == 4 &&
                streams.size(0) == 1 && streams.size(1) == 1,
                "streams must be contiguous CPU float (1, 1, H, D)");
    const int H = (int) streams.size(2);
    const int D = (int) streams.size(3);
    const int hc = H * D;
    TORCH_CHECK(H >= 1 && D >= 16 && D % 16 == 0, "ple_step_cpu: D must be a multiple of 16");
    TORCH_CHECK(emb.scalar_type() == at::kHalf && emb.is_contiguous() && emb.dim() == 3 &&
                emb.size(0) == 1 && emb.size(1) == 1,
                "emb must be contiguous CPU half (1, 1, ple_dim)");
    const int E = (int) emb.size(2);
    TORCH_CHECK(E >= 1, "ple_step_cpu: empty embedding row");
    TORCH_CHECK(key_w.scalar_type() == at::kHalf && key_w.is_contiguous() && key_w.dim() == 2 &&
                key_w.size(0) == E && key_w.size(1) == hc,
                "key_w must be contiguous CPU half (ple_dim, H * D)");
    TORCH_CHECK(value_w.scalar_type() == at::kHalf && value_w.is_contiguous() && value_w.dim() == 2 &&
                value_w.size(0) == E && value_w.size(1) == D,
                "value_w must be contiguous CPU half (ple_dim, D)");
    for (const at::Tensor* w : {&norm_key_w, &norm_query_w, &norm_conv_w})
        TORCH_CHECK((w->scalar_type() == at::kHalf || w->scalar_type() == at::kBFloat16 ||
                     w->scalar_type() == at::kFloat) && w->is_contiguous() && w->numel() == hc,
                    "norm weights must be contiguous CPU half/bfloat16/float of H * D");
    const int norm_dt = norm_key_w.scalar_type() == at::kHalf ? 0 :
                        (norm_key_w.scalar_type() == at::kBFloat16 ? 1 : 2);
    TORCH_CHECK(conv_w.scalar_type() == at::kHalf && conv_w.is_contiguous() && conv_w.dim() == 3 &&
                conv_w.size(0) == hc && conv_w.size(1) == 1 && conv_w.size(2) >= 1,
                "conv_w must be contiguous CPU half (H * D, 1, ksize)");
    const int K = (int) conv_w.size(2);
    TORCH_CHECK(conv_dilation >= 1, "ple_step_cpu: conv_dilation must be >= 1");
    const int win = (K - 1) * (int) conv_dilation;
    TORCH_CHECK(conv_stream.scalar_type() == at::kHalf && conv_stream.is_contiguous() &&
                conv_stream.dim() == 3 && conv_stream.size(0) == 1 &&
                conv_stream.size(1) == hc && conv_stream.size(2) == win + 1,
                "conv_stream must be contiguous CPU half (1, H * D, state_len + 1)");
    if (conv_state)
        TORCH_CHECK(conv_state->scalar_type() == at::kHalf && conv_state->is_contiguous() &&
                    conv_state->dim() == 3 && conv_state->size(0) == 1 &&
                    conv_state->size(1) == hc && conv_state->size(2) == win,
                    "conv_state must be contiguous CPU half (1, H * D, state_len)");
    TORCH_CHECK(delta.scalar_type() == at::kFloat && delta.is_contiguous() && delta.dim() == 4 &&
                delta.size(0) == 1 && delta.size(1) == 1 && delta.size(2) == H && delta.size(3) == D,
                "delta must be contiguous CPU float (1, 1, H, D)");

    auto fopts = streams.options();
    at::Tensor embf = at::empty({E}, fopts);
    at::Tensor key = at::empty({hc}, fopts);
    at::Tensor value = at::empty({D}, fopts);
    at::Tensor keyn = at::empty({hc}, fopts);
    at::Tensor queryn = at::empty({hc}, fopts);
    at::Tensor gated = at::empty({hc}, fopts);
    at::Tensor nk = at::empty({hc}, fopts);
    at::Tensor nq = at::empty({hc}, fopts);
    at::Tensor nc = at::empty({hc}, fopts);
    at::Tensor normed = at::empty({hc}, emb.options());

    PleCtx c;
    c.streams = streams.data_ptr<float>();
    c.emb = reinterpret_cast<const uint16_t*>(emb.data_ptr<at::Half>());
    c.key_w = reinterpret_cast<const uint16_t*>(key_w.data_ptr<at::Half>());
    c.value_w = reinterpret_cast<const uint16_t*>(value_w.data_ptr<at::Half>());
    c.nk_w = norm_key_w.const_data_ptr();
    c.nq_w = norm_query_w.const_data_ptr();
    c.nc_w = norm_conv_w.const_data_ptr();
    c.norm_dt = norm_dt;
    c.conv_w = reinterpret_cast<const uint16_t*>(conv_w.data_ptr<at::Half>());
    c.conv_state = conv_state ? reinterpret_cast<uint16_t*>(conv_state->data_ptr<at::Half>()) : nullptr;
    c.conv_stream = reinterpret_cast<uint16_t*>(conv_stream.data_ptr<at::Half>());
    c.delta = delta.data_ptr<float>();
    c.embf = embf.data_ptr<float>();
    c.key = key.data_ptr<float>();
    c.value = value.data_ptr<float>();
    c.keyn = keyn.data_ptr<float>();
    c.queryn = queryn.data_ptr<float>();
    c.gated = gated.data_ptr<float>();
    c.nk = nk.data_ptr<float>();
    c.nq = nq.data_ptr<float>();
    c.nc = nc.data_ptr<float>();
    c.normed = normed.data_ptr<at::Half>();
    c.H = H;
    c.D = D;
    c.E = E;
    c.K = K;
    c.dil = (int) conv_dilation;
    c.win = win;
    c.gate_scale = (float) gate_scale;
    c.eps = (float) eps;
    c.scale = (float) constant_scale;
    c.bias = (float) constant_bias;
    ple_step_run(c);
}

#else  // !x86
#include "ple_step_cpu.h"
#include <torch/extension.h>

void ple_step_cpu
(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    double, double, double, double,
    const at::Tensor&, const std::optional<at::Tensor>&, at::Tensor&, at::Tensor&, int64_t
)
{
    TORCH_CHECK(false, "ple_step_cpu is not available on this CPU architecture (x86 AVX-512 only)");
}

bool ple_step_cpu_has_avx512()
{
    return false;
}
#endif
