#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "gdn_step_cpu.h"
#include <c10/util/Half.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <immintrin.h>

#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Fused GDN decode step on CPU (gdn_step_cpu.h): the split-projection GatedDeltaNet middle
// section for one token -- b/a projections (fp16 GEMVs), beta/g gates, depthwise short conv
// update + SiLU, one-token grouped delta-rule scan (in-kernel q/k l2norm, undecayed state
// readback: gdn.cu cuda_recurrent_gated_delta_rule_kernel semantics) and the gated RMS norm.
// fp32 accumulation; math mirrors modules/gated_delta_net_fn/cpu.py + gated_delta_rule.py
// (torch_recurrent_gated_delta_rule_cpu, seqlen = 1). Runtime ISA detection with a Python-side
// fallback when the tier is absent, and a persistent spin-parked worker pool (hc_mix_cpu.cpp
// Pool). State rows and the conv ring are addressed in place; both stay contiguous per row.

#define GDN_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))

inline void gdn_pause()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// -------------------------------------------------------------------------------------------
//   Thread pool (persistent, spin-parked; hc_mix_cpu.cpp Pool)
// -------------------------------------------------------------------------------------------

typedef void (*GdnPoolFn)(void* ctx, int worker, int num_workers);

struct GdnPool
{
    std::atomic<uint64_t> dispatch{0};
    std::atomic<uint64_t> done{0};
    std::atomic<GdnPoolFn> fn{nullptr};
    void* ctx = nullptr;
    uint64_t generation = 0;
    int spawned = 0;
    int num_workers = 1;
    static constexpr int DISPATCH_NW_BITS = 16;
    static int dispatch_nw(uint64_t d) { return (int) (d & ((1ull << DISPATCH_NW_BITS) - 1)); }

    void worker_loop(int idx)
    {
        uint64_t seen = 0;
        int idle = 0;
        while (true)
        {
            const uint64_t g = dispatch.load(std::memory_order_acquire);
            if (g == seen)
            {
                if (++idle < 512) { gdn_pause(); continue; }
#ifdef __linux__
                const uint64_t cmp = seen;
                syscall(SYS_futex, (void*) &dispatch, FUTEX_WAIT_PRIVATE, cmp, nullptr, nullptr, 0);
#else
                std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
                continue;
            }
            idle = 0;
            seen = g;
            const int nw = dispatch_nw(g);
            if (idx < nw)
            {
                fn.load(std::memory_order_relaxed)(ctx, idx, nw);
                done.fetch_add(1, std::memory_order_release);
            }
        }
    }

    void ensure(int n)
    {
        while (spawned < n - 1)
        {
            std::thread(&GdnPool::worker_loop, this, spawned + 1).detach();
            ++spawned;
        }
        num_workers = n;
    }

    void run(GdnPoolFn f, void* c, int n_req = 0)
    {
        int n = num_workers;
        if (n_req > 0 && n_req < n) n = n_req;
        if (n <= 1) { f(c, 0, 1); return; }
        ctx = c;
        fn.store(f, std::memory_order_relaxed);
        const uint64_t d0 = done.load(std::memory_order_acquire);
        ++generation;
        dispatch.store((generation << DISPATCH_NW_BITS) | (uint64_t) n, std::memory_order_release);
#ifdef __linux__
        syscall(SYS_futex, (void*) &dispatch, FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr, nullptr, 0);
#endif
        f(c, 0, n);
        while (static_cast<int64_t>(done.load(std::memory_order_acquire) - d0) < n - 1)
            gdn_pause();
    }
};

GdnPool g_gdn_pool;
std::mutex g_gdn_pool_mutex;

inline int gdn_step_cpu_threads()
{
    // Explicit env, else single-threaded: the step runs 36x per token between the moe pool's
    // bursts, and waking parked workers under the moe pool + system load measured far worse
    // than the single-threaded sweep (EXL3_GDN_STEP_CPU_THREADS=2+: 2.6 tok/s vs 7.0)
    static const int v = [] {
        int t = 0;
        const char* e = std::getenv("EXL3_GDN_STEP_CPU_THREADS");
        if (e) t = std::atoi(e);
        if (t <= 0)
            t = 1;
        return std::max(1, t);
    }();
    return v;
}

// -------------------------------------------------------------------------------------------
//   Primitives
// -------------------------------------------------------------------------------------------

GDN_TARGET_AVX512
inline __m512 gdn_exp512(__m512 d)
{
    // Sleef u10 expf (sleefsimdsp.c xexpf) -- the routine torch's fp32 elementwise
    // exp/sigmoid/silu dispatch to on AVX-512, so the conv/norm activations round identically:
    // q = rint(d log2 e); s = d - q ln2 (hi/lo split, fma form); degree-6 poly; u * 2^(q>>1) *
    // 2^(q - (q>>1)); zero below -104, infinity above 100
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

GDN_TARGET_AVX512
inline __m512 gdn_sigmoid512(__m512 x)
{
    return _mm512_div_ps(_mm512_set1_ps(1.0f),
                         _mm512_add_ps(_mm512_set1_ps(1.0f), gdn_exp512(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

GDN_TARGET_AVX512
inline __m512 gdn_silu512(__m512 x)
{
    return _mm512_mul_ps(x, gdn_sigmoid512(x));
}

inline float gdn_sigmoid1(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float gdn_silu1(float x)
{
    return x * gdn_sigmoid1(x);
}

// float -> bf16, round to nearest even (torch .to(bfloat16))
inline uint16_t gdn_f32_to_bf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

inline float gdn_bf16_to_f32(uint16_t h)
{
    uint32_t u = ((uint32_t) h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// bf16 halves -> f32 (bit shift, not the fp16 cvt)
GDN_TARGET_AVX512
inline __m512 gdn_bf16_load(const uint16_t* p)
{
    return _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) p)), 16));
}

GDN_TARGET_AVX512
inline void gdn_bf16_store(uint16_t* dst, __m512 v)
{
    // f32 -> bf16 RNE per lane, packed halves (no AVX512-BF16 dependency: vpmovdw truncates)
    __m512i u = _mm512_castps_si512(v);
    __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                                                     _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
    _mm256_storeu_si256((__m256i*) dst, _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16)));
}

GDN_TARGET_AVX512
inline void gdn_half_store(at::Half* dst, __m512 v)
{
    // f32 -> fp16 RNE per lane (vcvtps2ph), packed halves
    _mm256_storeu_si256((__m256i*) dst, _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

GDN_TARGET_AVX512
inline float gdn_sumsq(const float* x, int n)
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

GDN_TARGET_AVX512
inline float gdn_sumsq_hp(const uint16_t* x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
    {
        __m512 v = gdn_bf16_load(x + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i)
    {
        float v = gdn_bf16_to_f32(x[i]);
        s = std::fma(v, v, s);
    }
    return s;
}

// -------------------------------------------------------------------------------------------
//   Kernels
// -------------------------------------------------------------------------------------------

#define GDN_MAX_QK 512
#define GDN_MAX_DV 512

struct GdnCtx
{
    // inputs
    const uint16_t* x;             // (bsz, hidden) fp16, layer input
    const uint16_t* qkv;           // (bsz, fdim) fp16, split qkv projection
    const float* z32;              // (bsz, nv, dv) fp32 gate, or
    const uint16_t* z16;           // (bsz, nv, dv) fp16 gate
    uint16_t* conv_state;          // (mb, fdim, kdim + hist) bf16, updated in place
    int64_t conv_row_stride;       // halves per conv ring row
    const float* conv_w;           // (kdim, fdim) fp32, tap-major
    const float* conv_b;           // (fdim,) fp32, nullable
    const uint16_t* wb;            // (hidden, nv) fp16, nullable
    const uint16_t* bb;            // (nv,) fp16, nullable
    const uint16_t* wa;            // (hidden, nv) fp16, nullable
    const uint16_t* ba;            // (nv,) fp16, nullable
    const float* dt_bias;          // (nv,) fp32
    const float* a_log;            // (nv,) fp32
    float* recurrent_state;        // (rows, nv, dk, dv) fp32, scan rows updated in place
    int64_t state_row_stride;      // floats per pool row
    const float* norm_w;           // (dv,) fp32
    at::Half* out;                 // (bsz, nv * dv) half, gated norm output
    // scratch
    uint16_t* mixed;               // (bsz, fdim) bf16, conv output
    uint16_t* beta;                // (bsz, nv) bf16
    float* g;                      // (bsz, nv) fp32
    uint16_t* cao;                 // (bsz, nv * dv) bf16, scan output
    // dims / scalars
    int bsz, hidden, fdim, kdim, nv, nk, dk, dv;
    float beta_scale, rms_eps, norm_bias;
    int gate_sigmoid;
    int z_is_half;
    // slots (pool rows)
    std::vector<int64_t> conv_slot, scan_slot;
    std::atomic<int> bar1, bar2;
};

static inline float softplus20(float x)
{
    // torch F.softplus(x, threshold = 20): x > 20 ? x : log1p(exp(x))
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

// gates: b = x @ wb (+ bb), beta = bf16(sigmoid(b) * beta_scale); a = x @ wa (+ ba),
// g = -softplus(a + dt_bias, thr 20) * exp(a_log)
GDN_TARGET_AVX512
static void gdn_gates_range(GdnCtx& c, int worker, int nw)
{
    if (c.wb == nullptr || c.wa == nullptr) return;
    const int nv = c.nv, hidden = c.hidden;
    const int j0 = (int) ((int64_t) nv * worker / nw);
    const int j1 = (int) ((int64_t) nv * (worker + 1) / nw);
    if (j0 >= j1) return;
    alignas(64) float accb[512], acca[512];
    for (int j = j0; j < j1; ++j) { accb[j] = 0.0f; acca[j] = 0.0f; }
    for (int b = 0; b < c.bsz; ++b)
    {
        const uint16_t* xr = c.x + (size_t) b * hidden;
        for (int h = 0; h < hidden; ++h)
        {
            const float xh = _cvtsh_ss(xr[h]);
            const uint16_t* wrb = c.wb + (size_t) h * nv;
            const uint16_t* wra = c.wa + (size_t) h * nv;
            const __m512 xb = _mm512_set1_ps(xh);
            int j = j0;
            for (; j + 16 <= j1; j += 16)
            {
                __m512 vb = _mm512_fmadd_ps(xb, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (wrb + j))), _mm512_loadu_ps(accb + j));
                __m512 va = _mm512_fmadd_ps(xb, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (wra + j))), _mm512_loadu_ps(acca + j));
                _mm512_storeu_ps(accb + j, vb);
                _mm512_storeu_ps(acca + j, va);
            }
            for (; j < j1; ++j)
            {
                accb[j] = std::fma(xh, _cvtsh_ss(wrb[j]), accb[j]);
                acca[j] = std::fma(xh, _cvtsh_ss(wra[j]), acca[j]);
            }
        }
        for (int j = j0; j < j1; ++j)
        {
            float bv = accb[j] + (c.bb ? _cvtsh_ss(c.bb[j]) : 0.0f);
            float av = acca[j] + (c.ba ? _cvtsh_ss(c.ba[j]) : 0.0f);
            c.beta[(size_t) b * nv + j] = gdn_f32_to_bf16(gdn_sigmoid1(bv) * c.beta_scale);
            c.g[(size_t) b * nv + j] = -softplus20(av + c.dt_bias[j]) * std::exp(c.a_log[j]);
        }
        // accumulators are shared across rows: re-zero for the next row
        for (int j = j0; j < j1; ++j) { accb[j] = 0.0f; acca[j] = 0.0f; }
    }
}

// depthwise short conv over [state window | token] + SiLU into mixed; the state window shifts by
// one and keeps the raw token (causal_conv1d_update_cpu, history = False, out position 1 of 2)
GDN_TARGET_AVX512
static void gdn_conv_range(GdnCtx& c, int worker, int nw)
{
    const int K = c.kdim, fdim = c.fdim;
    const int c0 = (int) ((int64_t) fdim * worker / nw);
    const int c1 = (int) ((int64_t) fdim * (worker + 1) / nw);
    for (int b = 0; b < c.bsz; ++b)
    {
        uint16_t* cs = c.conv_state + (size_t) c.conv_slot[b] * c.conv_row_stride;
        const uint16_t* qr = c.qkv + (size_t) b * fdim;
        uint16_t* mr = c.mixed + (size_t) b * fdim;
        int ci = c0;
        for (; ci + 16 <= c1; ci += 16)
        {
            __m512 y = c.conv_b ? _mm512_loadu_ps(c.conv_b + ci) : _mm512_setzero_ps();
            for (int k = 0; k < K - 1; ++k)
            {
                float s[16];
                for (int u = 0; u < 16; ++u)
                    s[u] = gdn_bf16_to_f32(cs[(size_t) (ci + u) * c.conv_row_stride + k + 1]);
                y = _mm512_fmadd_ps(_mm512_loadu_ps(s), _mm512_loadu_ps(c.conv_w + (size_t) k * fdim + ci), y);
            }
            // the torch path rounds the fp16 projection to bf16 before the conv (the conv
            // input is the channel-major bf16 mixed_qkv) and the state keeps those bf16 values
            __m512i xw = _mm512_castps_si512(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (qr + ci))));
            __m512i xr = _mm512_add_epi32(xw, _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                              _mm512_and_si512(_mm512_srli_epi32(xw, 16), _mm512_set1_epi32(1))));
            __m512 xb = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(_mm512_cvtepi32_epi16(_mm512_srli_epi32(xr, 16))), 16));
            y = _mm512_fmadd_ps(xb, _mm512_loadu_ps(c.conv_w + (size_t) (K - 1) * fdim + ci), y);
            y = gdn_silu512(y);
            gdn_bf16_store(mr + ci, y);
            for (int u = 0; u < 16; ++u)
            {
                uint16_t* w = cs + (size_t) (ci + u) * c.conv_row_stride;
                for (int k = 0; k < K - 1; ++k) w[k] = w[k + 1];
                w[K - 1] = gdn_f32_to_bf16(_cvtsh_ss(qr[ci + u]));
            }
        }
        for (; ci < c1; ++ci)
        {
            uint16_t* w = cs + (size_t) ci * c.conv_row_stride;
            const uint16_t xbits = gdn_f32_to_bf16(_cvtsh_ss(qr[ci]));
            const float xb = gdn_bf16_to_f32(xbits);
            float y = c.conv_b ? c.conv_b[ci] : 0.0f;
            for (int k = 0; k < K - 1; ++k)
                y = std::fma(gdn_bf16_to_f32(w[k + 1]), c.conv_w[(size_t) k * fdim + ci], y);
            y = std::fma(xb, c.conv_w[(size_t) (K - 1) * fdim + ci], y);
            mr[ci] = gdn_f32_to_bf16(gdn_silu1(y));
            for (int k = 0; k < K - 1; ++k) w[k] = w[k + 1];
            w[K - 1] = xbits;
        }
    }
}

// one-token grouped delta-rule scan: in-kernel q/k l2norm (eps 1e-6), undecayed state readback,
// u = beta (v - decay k^T state), state <- state * decay + k u^T, out = bf16(scale q^T state)
GDN_TARGET_AVX512
static void gdn_scan_range(GdnCtx& c, int worker, int nw)
{
    const int nv = c.nv, dk = c.dk, dv = c.dv;
    const int group = nv / c.nk;
    const float scale = 1.0f / std::sqrt((float) dk);
    const int h0 = (int) ((int64_t) nv * worker / nw);
    const int h1 = (int) ((int64_t) nv * (worker + 1) / nw);
    if (h0 >= h1) return;
    static thread_local float kf[GDN_MAX_QK], qf[GDN_MAX_QK];
    alignas(64) static thread_local float kvm[GDN_MAX_DV], u[GDN_MAX_DV];
    for (int b = 0; b < c.bsz; ++b)
    {
        const uint16_t* mr = c.mixed + (size_t) b * c.fdim;
        const uint16_t* qr = mr;
        const uint16_t* kr = mr + c.nk * dk;
        const uint16_t* vr = mr + 2 * c.nk * dk;
        float* st = c.recurrent_state + (size_t) c.scan_slot[b] * c.state_row_stride;
        for (int i = h0; i < h1; ++i)
        {
            const int kh = i / group;
            const uint16_t* kp = kr + (size_t) kh * dk;
            const uint16_t* qp = qr + (size_t) kh * dk;
            for (int d = 0; d < dk; ++d) { kf[d] = gdn_bf16_to_f32(kp[d]); qf[d] = gdn_bf16_to_f32(qp[d]); }
            float kinv = 1.0f / std::sqrt(gdn_sumsq(kf, dk) + 1e-6f);
            float qinv = 1.0f / std::sqrt(gdn_sumsq(qf, dk) + 1e-6f);
            for (int d = 0; d < dk; ++d) { kf[d] *= kinv; qf[d] *= qinv; }
            const float decay = std::exp(c.g[(size_t) b * nv + i]);
            const float betaf = gdn_bf16_to_f32(c.beta[(size_t) b * nv + i]);
            float* s = st + (size_t) i * dk * dv;
            const uint16_t* vp = vr + (size_t) i * dv;
            uint16_t* op = c.cao + (size_t) (b * nv + i) * dv;
            const __m512 dcy = _mm512_set1_ps(decay), bcy = _mm512_set1_ps(betaf);
            int v = 0;
            for (; v + 16 <= dv; v += 16)
            {
                // kv_mem = k^T state (undecayed readback)
                __m512 acc = _mm512_setzero_ps();
                for (int d = 0; d < dk; ++d)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(s + (size_t) d * dv + v), _mm512_set1_ps(kf[d]), acc);
                _mm512_storeu_ps(kvm + v, acc);
                // u = beta (v - decay kv_mem)
                __m512 vv = gdn_bf16_load(vp + v);
                __m512 uv = _mm512_mul_ps(_mm512_fnmadd_ps(_mm512_loadu_ps(kvm + v), dcy, vv), bcy);
                _mm512_storeu_ps(u + v, uv);
                // state <- state * decay + k u^T ; out = scale q^T state (updated state)
                __m512 oac = _mm512_setzero_ps();
                for (int d = 0; d < dk; ++d)
                {
                    __m512 sv = _mm512_mul_ps(_mm512_loadu_ps(s + (size_t) d * dv + v), dcy);
                    sv = _mm512_fmadd_ps(_mm512_set1_ps(kf[d]), uv, sv);
                    _mm512_storeu_ps(s + (size_t) d * dv + v, sv);
                    oac = _mm512_fmadd_ps(_mm512_set1_ps(qf[d]), sv, oac);
                }
                gdn_bf16_store(op + v, _mm512_mul_ps(oac, _mm512_set1_ps(scale)));
            }
            for (; v < dv; ++v)
            {
                float acc = 0.0f;
                for (int d = 0; d < dk; ++d) acc = std::fma(s[(size_t) d * dv + v], kf[d], acc);
                const float uv = (gdn_bf16_to_f32(vp[v]) - acc * decay) * betaf;
                float oac = 0.0f;
                for (int d = 0; d < dk; ++d)
                {
                    float& sd = s[(size_t) d * dv + v];
                    sd = sd * decay + kf[d] * uv;
                    oac = std::fma(qf[d], sd, oac);
                }
                op[v] = gdn_f32_to_bf16(oac * scale);
            }
        }
    }
}

// gated RMS norm over the scan output: h = bf16(scan out) -> f32, * rsqrt(mean h^2 + eps),
// * (weight + bias), * gate(z), -> fp16 (GatedRMSNorm._forward_cpu, not gate_first)
GDN_TARGET_AVX512
static void gdn_norm_range(GdnCtx& c, int worker, int nw)
{
    const int nv = c.nv, dv = c.dv;
    const int rows = c.bsz * nv;
    const int r0 = (int) ((int64_t) rows * worker / nw);
    const int r1 = (int) ((int64_t) rows * (worker + 1) / nw);
    if (r0 >= r1) return;
    const __m512 eps = _mm512_set1_ps(c.rms_eps);
    const __m512 bias = _mm512_set1_ps(c.norm_bias);
    for (int r = r0; r < r1; ++r)
    {
        const int b = r / nv;
        const int i = r - b * nv;
        const uint16_t* hp = c.cao + (size_t) r * dv;
        const float rmf = 1.0f / std::sqrt(gdn_sumsq_hp(hp, dv) / (float) dv + c.rms_eps);
        const float* zp32 = c.z32 ? c.z32 + (size_t) r * dv : nullptr;
        const uint16_t* zp16 = c.z16 ? c.z16 + (size_t) r * dv : nullptr;
        at::Half* op = c.out + (size_t) r * dv;
        int d = 0;
        for (; d + 16 <= dv; d += 16)
        {
            __m512 hv = gdn_bf16_load(hp + d);
            hv = _mm512_mul_ps(hv, _mm512_set1_ps(rmf));
            hv = _mm512_mul_ps(hv, _mm512_add_ps(_mm512_loadu_ps(c.norm_w + d), bias));
            __m512 gv = c.z32 ? _mm512_loadu_ps(zp32 + d)
                              : _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (zp16 + d)));
            __m512 gated = c.gate_sigmoid ? gdn_sigmoid512(gv) : gdn_silu512(gv);
            gdn_half_store(op + d, _mm512_mul_ps(hv, gated));
        }
        for (; d < dv; ++d)
        {
            float v = gdn_bf16_to_f32(hp[d]) * rmf * (c.norm_w[d] + c.norm_bias);
            float gv = c.z32 ? zp32[d] : _cvtsh_ss(zp16[d]);
            float gate = c.gate_sigmoid ? gdn_sigmoid1(gv) : gdn_silu1(gv);
            op[d] = at::Half(v * gate);
        }
    }
}

static void gdn_pool_fn(void* vctx, int worker, int nw)
{
    GdnCtx& c = *static_cast<GdnCtx*>(vctx);
    gdn_gates_range(c, worker, nw);
    gdn_conv_range(c, worker, nw);
    c.bar1.fetch_add(1, std::memory_order_acq_rel);
    while (c.bar1.load(std::memory_order_acquire) < nw) gdn_pause();
    gdn_scan_range(c, worker, nw);
    c.bar2.fetch_add(1, std::memory_order_acq_rel);
    while (c.bar2.load(std::memory_order_acquire) < nw) gdn_pause();
    gdn_norm_range(c, worker, nw);
}

// -------------------------------------------------------------------------------------------
//   Entry
// -------------------------------------------------------------------------------------------




bool gdn_step_cpu_has_avx512()
{
    static const bool v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
                          __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return v;
}

void gdn_step_cpu
(
    const at::Tensor& x,
    const at::Tensor& qkv,
    const at::Tensor& z,
    const at::Tensor& conv_state,
    const at::Tensor& conv_w,
    const std::optional<at::Tensor>& conv_b,
    const std::optional<at::Tensor>& wb,
    const std::optional<at::Tensor>& bb,
    const std::optional<at::Tensor>& wa,
    const std::optional<at::Tensor>& ba,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    const at::Tensor& recurrent_state,
    const at::Tensor& conv_slots,
    const at::Tensor& scan_slots,
    const at::Tensor& norm_w,
    at::Tensor& out,
    double beta_scale,
    double rms_eps,
    double norm_bias_add,
    bool gate_sigmoid,
    bool z_is_half,
    int num_threads
)
{
    TORCH_CHECK(gdn_step_cpu_has_avx512(), "gdn_step_cpu: AVX-512 not available");
    TORCH_CHECK(x.is_cpu() && qkv.is_cpu() && z.is_cpu() && conv_state.is_cpu() &&
                recurrent_state.is_cpu() && out.is_cpu(), "gdn_step_cpu: tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 3 && x.size(1) == 1,
                "x must be contiguous CPU half (bsz, 1, hidden)");
    TORCH_CHECK(qkv.scalar_type() == at::kHalf && qkv.is_contiguous() && qkv.dim() == 3 && qkv.size(1) == 1,
                "qkv must be contiguous CPU half (bsz, 1, fdim)");
    const int bsz = (int) x.size(0);
    const int hidden = (int) x.size(2);
    const int fdim = (int) qkv.size(2);
    TORCH_CHECK(z.is_contiguous() && z.dim() == 4 && z.size(0) == bsz && z.size(1) == 1,
                "z must be contiguous (bsz, 1, nv, dv)");
    TORCH_CHECK(z.scalar_type() == (z_is_half ? at::kHalf : at::kFloat), "z dtype mismatch");
    const int nv = (int) z.size(2);
    const int dv = (int) z.size(3);
    TORCH_CHECK(nv >= 1 && nv <= 512 && dv >= 1 && dv <= GDN_MAX_DV, "gdn_step_cpu: nv, dv <= 512 supported");
    TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && conv_state.is_contiguous() && conv_state.dim() == 3,
                "conv_state must be contiguous CPU bf16 (mb, fdim, kdim + hist)");
    TORCH_CHECK(conv_state.size(1) == fdim, "conv_state fdim mismatch");
    TORCH_CHECK(conv_w.scalar_type() == at::kFloat && conv_w.is_contiguous() && conv_w.dim() == 2 &&
                conv_w.size(1) == fdim, "conv_w must be contiguous CPU float (kdim, fdim)");
    const int kdim = (int) conv_w.size(0);
    TORCH_CHECK(kdim >= 1 && conv_state.size(2) >= kdim, "conv_state window narrower than the conv kernel");
    if (conv_b) TORCH_CHECK(conv_b->scalar_type() == at::kFloat && conv_b->is_contiguous() &&
                            conv_b->numel() == fdim, "conv_b must be contiguous CPU float (fdim,)");
    if (wb) TORCH_CHECK(wb->scalar_type() == at::kHalf && wb->is_contiguous() && wb->dim() == 2 &&
                        wb->size(0) == hidden && wb->size(1) == nv, "wb must be contiguous CPU half (hidden, nv)");
    if (bb) TORCH_CHECK(bb->scalar_type() == at::kHalf && bb->is_contiguous() && bb->numel() == nv,
                        "bb must be contiguous CPU half (nv,)");
    if (wa) TORCH_CHECK(wa->scalar_type() == at::kHalf && wa->is_contiguous() && wa->dim() == 2 &&
                        wa->size(0) == hidden && wa->size(1) == nv, "wa must be contiguous CPU half (hidden, nv)");
    if (ba) TORCH_CHECK(ba->scalar_type() == at::kHalf && ba->is_contiguous() && ba->numel() == nv,
                        "ba must be contiguous CPU half (nv,)");
    TORCH_CHECK(dt_bias.scalar_type() == at::kFloat && dt_bias.is_contiguous() && dt_bias.numel() == nv,
                "dt_bias must be contiguous CPU float (nv,)");
    TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.is_contiguous() && a_log.numel() == nv,
                "a_log must be contiguous CPU float (nv,)");
    TORCH_CHECK(recurrent_state.scalar_type() == at::kFloat && recurrent_state.is_contiguous() &&
                recurrent_state.dim() == 5 && recurrent_state.size(1) == 1,
                "recurrent_state must be contiguous CPU float (rows, 1, nv, dk, dv)");
    TORCH_CHECK(recurrent_state.size(2) == nv && recurrent_state.size(4) == dv,
                "recurrent_state shape mismatch");
    const int dk = (int) recurrent_state.size(3);
    TORCH_CHECK(dk >= 1 && dk <= GDN_MAX_QK, "gdn_step_cpu: dk <= 512 supported");
    TORCH_CHECK(norm_w.scalar_type() == at::kFloat && norm_w.is_contiguous() && norm_w.numel() == dv,
                "norm_w must be contiguous CPU float (dv,)");
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous() &&
                out.numel() == (int64_t) bsz * nv * dv, "out must be contiguous CPU half (bsz, nv * dv)");
    TORCH_CHECK(conv_slots.numel() == bsz && scan_slots.numel() == bsz, "slots must have bsz entries");

    auto slot_of = [](const at::Tensor& t, int i) -> int64_t {
        return t.scalar_type() == at::kLong ? (int64_t) t.data_ptr<int64_t>()[i] : (int64_t) t.data_ptr<int>()[i];
    };

    // Scratch: conv output, beta, g, scan output. Fixed small shapes per call; allocated here to
    // keep the binding surface flat (one call per layer per decode step)
    auto bf16opts = qkv.options().dtype(at::kBFloat16);
    at::Tensor mixed = at::empty({bsz, fdim}, bf16opts);
    at::Tensor beta = at::empty({bsz, nv}, bf16opts);
    at::Tensor g = at::empty({bsz, nv}, qkv.options().dtype(at::kFloat));
    at::Tensor cao = at::empty({(int64_t) bsz * nv * dv}, bf16opts);

    GdnCtx c;
    c.x = reinterpret_cast<const uint16_t*>(x.data_ptr<at::Half>());
    c.qkv = reinterpret_cast<const uint16_t*>(qkv.data_ptr<at::Half>());
    c.z32 = z_is_half ? nullptr : z.data_ptr<float>();
    c.z16 = z_is_half ? reinterpret_cast<const uint16_t*>(z.data_ptr<at::Half>()) : nullptr;
    c.conv_state = reinterpret_cast<uint16_t*>(conv_state.data_ptr<at::BFloat16>());
    c.conv_row_stride = conv_state.size(2);
    c.conv_w = conv_w.data_ptr<float>();
    c.conv_b = conv_b ? conv_b->data_ptr<float>() : nullptr;
    c.wb = wb ? reinterpret_cast<const uint16_t*>(wb->data_ptr<at::Half>()) : nullptr;
    c.bb = bb ? reinterpret_cast<const uint16_t*>(bb->data_ptr<at::Half>()) : nullptr;
    c.wa = wa ? reinterpret_cast<const uint16_t*>(wa->data_ptr<at::Half>()) : nullptr;
    c.ba = ba ? reinterpret_cast<const uint16_t*>(ba->data_ptr<at::Half>()) : nullptr;
    c.dt_bias = dt_bias.data_ptr<float>();
    c.a_log = a_log.data_ptr<float>();
    c.recurrent_state = recurrent_state.data_ptr<float>();
    c.state_row_stride = recurrent_state.stride(0);
    c.norm_w = norm_w.data_ptr<float>();
    c.out = out.data_ptr<at::Half>();
    c.mixed = reinterpret_cast<uint16_t*>(mixed.data_ptr<at::BFloat16>());
    c.beta = reinterpret_cast<uint16_t*>(beta.data_ptr<at::BFloat16>());
    c.g = g.data_ptr<float>();
    c.cao = reinterpret_cast<uint16_t*>(cao.data_ptr<at::BFloat16>());
    c.bsz = bsz;
    c.hidden = hidden;
    c.fdim = fdim;
    c.kdim = kdim;
    c.nv = nv;
    c.dk = dk;
    c.dv = dv;
    // nk from the qkv split: fdim = 2 * nk * dk + nv * dv
    int nk = -1;
    for (int t = 1; t <= nv; ++t)
        if (2 * t * dk + nv * dv == fdim) { nk = t; break; }
    TORCH_CHECK(nk > 0, "gdn_step_cpu: fdim does not decompose as 2 * nk * dk + nv * dv");
    c.nk = nk;
    c.beta_scale = (float) beta_scale;
    c.rms_eps = (float) rms_eps;
    c.norm_bias = (float) norm_bias_add;
    c.gate_sigmoid = gate_sigmoid ? 1 : 0;
    c.z_is_half = z_is_half ? 1 : 0;
    c.conv_slot.resize(bsz);
    c.scan_slot.resize(bsz);
    for (int b = 0; b < bsz; ++b)
    {
        c.conv_slot[b] = slot_of(conv_slots, b);
        c.scan_slot[b] = slot_of(scan_slots, b);
        TORCH_CHECK(c.conv_slot[b] >= 0 && c.conv_slot[b] < conv_state.size(0),
                    "conv slot out of range");
        TORCH_CHECK(c.scan_slot[b] >= 0 && c.scan_slot[b] < recurrent_state.size(0),
                    "scan slot out of range");
    }
    c.bar1.store(0, std::memory_order_relaxed);
    c.bar2.store(0, std::memory_order_relaxed);

    int nw = num_threads > 0 ? std::min(num_threads, gdn_step_cpu_threads()) : gdn_step_cpu_threads();
    nw = std::max(1, nw);
    std::lock_guard<std::mutex> lock(g_gdn_pool_mutex);
    g_gdn_pool.ensure(nw);
    g_gdn_pool.run(&gdn_pool_fn, &c, nw);


}

#else  // !x86
#include "gdn_step_cpu.h"
#include <torch/extension.h>

void gdn_step_cpu
(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    double, double, double, bool, bool, int
)
{
    TORCH_CHECK(false, "gdn_step_cpu is not available on this CPU architecture (x86 AVX-512 only)");
}

bool gdn_step_cpu_has_avx512()
{
    return false;
}
#endif
