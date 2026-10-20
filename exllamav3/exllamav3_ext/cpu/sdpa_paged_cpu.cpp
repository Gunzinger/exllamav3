// Fused paged-attention decode kernel (sdpa_paged_cpu.h). x86-64 build; other platforms get
// stubs that report no AVX-512 so the Python side keeps the torch gather path.

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "sdpa_paged_cpu.h"
#include "cpu/moe_mul1.h"
#include <torch/extension.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <immintrin.h>

#define SDPA_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))

namespace {

constexpr int SDPA_MAX_D = 512;
constexpr int SDPA_TILE = 16;

constexpr float SDPA_NEG_INF = -std::numeric_limits<float>::infinity();

// Sleef u10 expf (sleefsimdsp.c xexpf) -- the routine torch's fp32 elementwise exp dispatches
// to on AVX-512 (copied from gdn_step_cpu.cpp, gdn_exp512), so the softmax exponentials round
// identically to the torch path
SDPA_TARGET_AVX512 inline __m512 sdpa_exp512(__m512 d)
{
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

SDPA_TARGET_AVX512 inline float sdpa_exp1(float x)
{
    alignas(64) float t[16];
    _mm512_store_ps(t, sdpa_exp512(_mm512_set1_ps(x)));
    return t[0];
}

SDPA_TARGET_AVX512 inline float sdpa_hsum16(__m512 v)
{
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    s4 = _mm_add_ps(s4, _mm_movehdup_ps(s4));
    s4 = _mm_add_ss(s4, _mm_movehl_ps(s4, s4));
    return _mm_cvtss_f32(s4);
}

struct SdpaCtx
{
    const uint16_t* q;             // (B, Q, H, D) fp16
    const uint16_t* k_new;         // (B, Q, HK, D) fp16
    const uint16_t* v_new;
    const uint16_t* k_cache;       // (P, page, HK, D) fp16
    const uint16_t* v_cache;
    const void* bt;                // (B, max_pages) int32 or int64
    const void* cs;                // (B,) int32 or int64
    int bt_is64, cs_is64;
    at::Half* out;                 // (B, Q, H, D) fp16
    int64_t max_pages;
    int B, Q, H, HK, D, page;
    float scale;
};

inline int64_t sdpa_bt(const SdpaCtx& c, int64_t b, int64_t i)
{
    return c.bt_is64 ? static_cast<const int64_t*>(c.bt)[b * c.max_pages + i]
                     : (int64_t) static_cast<const int32_t*>(c.bt)[b * c.max_pages + i];
}

inline int64_t sdpa_cs(const SdpaCtx& c, int b)
{
    return c.cs_is64 ? static_cast<const int64_t*>(c.cs)[b]
                     : (int64_t) static_cast<const int32_t*>(c.cs)[b];
}

// One 16-key tile of one contiguous-row segment (stride row_stride halves): scores against the
// fp32 query row, then the flash recurrence -- running max m, running sum l, output accumulator
// acc rescaled by exp(m - m_new) and accumulated with p_j v_j. Rows past n are masked to -inf
// (their p is exactly 0) and their load addresses clamped in-range
SDPA_TARGET_AVX512 inline void sdpa_tile
(
    const uint16_t* __restrict__ kb,
    const uint16_t* __restrict__ vb,
    int64_t row_stride,
    int n,
    const float* __restrict__ qf,
    float* __restrict__ acc,
    int d16,
    float scale,
    float& m,
    float& l
)
{
    alignas(64) float s[SDPA_TILE];
    {
        __m512 sv[SDPA_TILE];
        for (int j = 0; j < SDPA_TILE; ++j) sv[j] = _mm512_setzero_ps();
        for (int d = 0; d < d16; ++d)
        {
            const __m512 qv = _mm512_load_ps(qf + 16 * d);
            for (int j = 0; j < SDPA_TILE; ++j)
            {
                const int64_t jj = j < n ? j : n - 1;
                __m512 kv = _mm512_cvtph_ps(_mm256_loadu_si256(
                    (const __m256i*) (const void*) (kb + jj * row_stride + 16 * d)));
                sv[j] = _mm512_fmadd_ps(kv, qv, sv[j]);
            }
        }
        for (int j = 0; j < SDPA_TILE; ++j)
            s[j] = j < n ? sdpa_hsum16(sv[j]) * scale : SDPA_NEG_INF;
    }
    float tm = s[0];
    for (int j = 1; j < SDPA_TILE; ++j) tm = tm < s[j] ? s[j] : tm;
    const float m_new = m < tm ? tm : m;
    const __m512 pv = sdpa_exp512(_mm512_sub_ps(_mm512_load_ps(s), _mm512_set1_ps(m_new)));
    alignas(64) float p[SDPA_TILE];
    _mm512_store_ps(p, pv);
    const float alpha = sdpa_exp1(m - m_new);
    for (int d = 0; d < d16; ++d)
    {
        __m512 a = _mm512_load_ps(acc + 16 * d);
        a = _mm512_mul_ps(a, _mm512_set1_ps(alpha));
        for (int j = 0; j < n; ++j)
        {
            __m512 vv = _mm512_cvtph_ps(_mm256_loadu_si256(
                (const __m256i*) (const void*) (vb + j * row_stride + 16 * d)));
            a = _mm512_fmadd_ps(_mm512_set1_ps(p[j]), vv, a);
        }
        _mm512_store_ps(acc + 16 * d, a);
    }
    l = l * alpha + _mm512_reduce_add_ps(pv);
    m = m_new;
}

// Head tasks (b, qi, h), h innermost so a worker's contiguous chunk shares kv rows: query row
// qi over keys [0, cache_seqlen + qi], cache keys walked page by page in place, the newest
// qi + 1 keys from k_new/v_new
SDPA_TARGET_AVX512 static void sdpa_range(const SdpaCtx& c, int worker, int nw)
{
    const int group = c.H / c.HK;
    const int64_t row_hk = (int64_t) c.HK * c.D;   // halves between consecutive key rows
    const int d16 = c.D / 16;
    const int tasks = c.B * c.Q * c.H;
    const int t0 = (int) ((int64_t) tasks * worker / nw);
    const int t1 = (int) ((int64_t) tasks * (worker + 1) / nw);
    if (t0 >= t1) return;
    alignas(64) static thread_local float qf[SDPA_MAX_D];
    alignas(64) static thread_local float acc[SDPA_MAX_D];
    for (int t = t0; t < t1; ++t)
    {
        const int b = t / (c.Q * c.H);
        const int qi = (t / c.H) % c.Q;
        const int h = t % c.H;
        const int kh = h / group;
        const int64_t cs = sdpa_cs(c, b);
        const uint16_t* qp = c.q + (((int64_t) b * c.Q + qi) * c.H + h) * c.D;
        for (int d = 0; d < d16; ++d)
        {
            _mm512_store_ps(qf + 16 * d, _mm512_cvtph_ps(_mm256_loadu_si256(
                (const __m256i*) (const void*) (qp + 16 * d))));
            _mm512_store_ps(acc + 16 * d, _mm512_setzero_ps());
        }
        float m = SDPA_NEG_INF;
        float l = 0.0f;
        int64_t pos = 0;
        while (pos < cs)
        {
            const int64_t page_idx = pos / c.page;
            const int off = (int) (pos - page_idx * c.page);
            const int n = (int) std::min<int64_t>(c.page - off, cs - pos);
            const int64_t phys = sdpa_bt(c, b, page_idx);
            const int64_t base = (phys * c.page + off) * row_hk + (int64_t) kh * c.D;
            for (int i = 0; i < n; i += SDPA_TILE)
                sdpa_tile(c.k_cache + base + (int64_t) i * row_hk,
                          c.v_cache + base + (int64_t) i * row_hk,
                          row_hk, std::min(SDPA_TILE, n - i), qf, acc, d16, c.scale, m, l);
            pos += n;
        }
        const int64_t nbase = (int64_t) b * c.Q * row_hk + (int64_t) kh * c.D;
        const int n = qi + 1;
        for (int i = 0; i < n; i += SDPA_TILE)
            sdpa_tile(c.k_new + nbase + (int64_t) i * row_hk,
                      c.v_new + nbase + (int64_t) i * row_hk,
                      row_hk, std::min(SDPA_TILE, n - i), qf, acc, d16, c.scale, m, l);
        const float inv_l = 1.0f / l;
        at::Half* op = c.out + (((int64_t) b * c.Q + qi) * c.H + h) * c.D;
        for (int d = 0; d < d16; ++d)
        {
            __m512 o = _mm512_mul_ps(_mm512_load_ps(acc + 16 * d), _mm512_set1_ps(inv_l));
            _mm256_storeu_si256((__m256i*) (void*) (op + 16 * d),
                                _mm512_cvtps_ph(o, _MM_FROUND_TO_NEAREST_INT));
        }
    }
}

void sdpa_pool_fn(void* vctx, int worker, int nw)
{
    sdpa_range(*static_cast<SdpaCtx*>(vctx), worker, nw);
}

int sdpa_paged_cpu_threads()
{
    // The KV stream saturates on a slice of the moe pool; exl3_moe_cpu_pool_run never exceeds
    // the pool's current size and an unspawned pool runs serially. 1 = serial
    static const int v = [] {
        int t = 6;
        const char* e = std::getenv("EXL3_SDPA_PAGED_CPU_THREADS");
        if (e && *e) t = std::atoi(e);
        return std::max(1, t);
    }();
    return v;
}

} // namespace

bool sdpa_paged_cpu_has_avx512()
{
    static const bool v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
                          __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return v;
}

void sdpa_paged_cpu
(
    const at::Tensor& q,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    double scale,
    at::Tensor& out
)
{
    TORCH_CHECK(sdpa_paged_cpu_has_avx512(), "sdpa_paged_cpu: AVX-512 not available");
    TORCH_CHECK(q.is_cpu() && k_new.is_cpu() && v_new.is_cpu() && k_cache.is_cpu() &&
                v_cache.is_cpu() && block_table.is_cpu() && cache_seqlens.is_cpu() &&
                out.is_cpu(), "sdpa_paged_cpu: tensors must be on CPU");
    TORCH_CHECK(q.scalar_type() == at::kHalf && q.is_contiguous() && q.dim() == 4,
                "q must be contiguous CPU half (B, Q, H, D)");
    TORCH_CHECK(k_new.scalar_type() == at::kHalf && k_new.is_contiguous() && k_new.dim() == 4,
                "k_new must be contiguous CPU half (B, Q, HK, D)");
    TORCH_CHECK(v_new.scalar_type() == at::kHalf && v_new.is_contiguous() && v_new.dim() == 4,
                "v_new must be contiguous CPU half (B, Q, HK, D)");
    TORCH_CHECK(k_cache.scalar_type() == at::kHalf && k_cache.is_contiguous() && k_cache.dim() == 4,
                "k_cache must be contiguous CPU half (P, page, HK, D)");
    TORCH_CHECK(v_cache.scalar_type() == at::kHalf && v_cache.is_contiguous() && v_cache.dim() == 4,
                "v_cache must be contiguous CPU half (P, page, HK, D)");
    const int B = (int) q.size(0);
    const int Q = (int) q.size(1);
    const int H = (int) q.size(2);
    const int D = (int) q.size(3);
    TORCH_CHECK(Q >= 1 && Q <= 8, "sdpa_paged_cpu: decode-class Q (1..8) only");
    TORCH_CHECK(H >= 1 && H <= 64, "sdpa_paged_cpu: H up to 64 supported");
    TORCH_CHECK(D >= 16 && D <= SDPA_MAX_D && D % 16 == 0,
                "sdpa_paged_cpu: D must be a multiple of 16 up to 512");
    const int HK = (int) k_new.size(2);
    TORCH_CHECK(HK >= 1 && H % HK == 0, "H must be a multiple of HK");
    TORCH_CHECK(k_new.size(0) == B && k_new.size(1) == Q && k_new.size(3) == D &&
                v_new.size(0) == B && v_new.size(1) == Q && v_new.size(2) == HK &&
                v_new.size(3) == D, "k_new/v_new shape mismatch");
    TORCH_CHECK(k_cache.size(2) == HK && k_cache.size(3) == D &&
                v_cache.size(0) == k_cache.size(0) && v_cache.size(1) == k_cache.size(1) &&
                v_cache.size(2) == HK && v_cache.size(3) == D, "cache shape mismatch");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B && block_table.is_contiguous() &&
                (block_table.scalar_type() == at::kInt || block_table.scalar_type() == at::kLong),
                "block_table must be contiguous (B, max_pages) int32/int64");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.numel() == B && cache_seqlens.is_contiguous() &&
                (cache_seqlens.scalar_type() == at::kInt || cache_seqlens.scalar_type() == at::kLong),
                "cache_seqlens must be contiguous (B,) int32/int64");
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous() && out.dim() == 4 &&
                out.size(0) == B && out.size(1) == Q && out.size(2) == H && out.size(3) == D,
                "out must be contiguous CPU half (B, Q, H, D)");

    SdpaCtx c;
    c.q = reinterpret_cast<const uint16_t*>(q.data_ptr<at::Half>());
    c.k_new = reinterpret_cast<const uint16_t*>(k_new.data_ptr<at::Half>());
    c.v_new = reinterpret_cast<const uint16_t*>(v_new.data_ptr<at::Half>());
    c.k_cache = reinterpret_cast<const uint16_t*>(k_cache.data_ptr<at::Half>());
    c.v_cache = reinterpret_cast<const uint16_t*>(v_cache.data_ptr<at::Half>());
    c.bt = block_table.data_ptr();
    c.cs = cache_seqlens.data_ptr();
    c.bt_is64 = block_table.scalar_type() == at::kLong ? 1 : 0;
    c.cs_is64 = cache_seqlens.scalar_type() == at::kLong ? 1 : 0;
    c.out = out.data_ptr<at::Half>();
    c.max_pages = block_table.size(1);
    c.B = B;
    c.Q = Q;
    c.H = H;
    c.HK = HK;
    c.D = D;
    c.page = (int) k_cache.size(1);
    TORCH_CHECK(c.page >= 1, "cache page size must be positive");
    c.scale = (float) scale;
    const int64_t num_phys = k_cache.size(0);
    for (int b = 0; b < B; ++b)
    {
        const int64_t cs = sdpa_cs(c, b);
        TORCH_CHECK(cs >= 0, "cache_seqlens must be non-negative");
        const int64_t need = (cs + Q + c.page - 1) / c.page;
        TORCH_CHECK(need <= c.max_pages, "block_table too short for cache_seqlens + q_len");
        for (int64_t i = 0; i < need; ++i)
        {
            const int64_t phys = sdpa_bt(c, b, i);
            TORCH_CHECK(phys >= 0 && phys < num_phys, "block_table entry out of range");
        }
    }
    exl3_moe_cpu_pool_run(&sdpa_pool_fn, &c, sdpa_paged_cpu_threads());
}

#else  // !x86

#include "sdpa_paged_cpu.h"
#include <torch/extension.h>

void sdpa_paged_cpu
(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, double, at::Tensor&
)
{
    TORCH_CHECK(false, "sdpa_paged_cpu is x86-only");
}

bool sdpa_paged_cpu_has_avx512()
{
    return false;
}

#endif
