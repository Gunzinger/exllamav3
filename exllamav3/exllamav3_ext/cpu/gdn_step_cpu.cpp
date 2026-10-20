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

inline int gdn_chunk_scan_threads()
{
    // Prefill-sized work (the scan is ~5 MFLOP per chunk per head); a small fixed pool keeps
    // the phase barriers tight (EXL3_GDN_CHUNK_SCAN_THREADS)
    static const int v = [] {
        int t = 0;
        const char* e = std::getenv("EXL3_GDN_CHUNK_SCAN_THREADS");
        if (e) t = std::atoi(e);
        if (t <= 0)
            t = 4;
        return std::max(1, t);
    }();
    return v;
}

inline int gdn_chunk_scan_barriers()
{
    static const int v = [] {
        const char* e = std::getenv("EXL3_GDN_CHUNK_SCAN_BARRIERS");
        return (e && std::atoi(e) != 0) ? 1 : 0;
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
inline float gdn_half_to_f32(uint16_t h)
{
    return _cvtsh_ss(h);
}

GDN_TARGET_AVX512
inline uint16_t gdn_f32_to_half(float v)
{
    return (uint16_t) _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT);
}

// 16 fp16 -> one f32 vector in a single vcvtph2ps (256-bit source, 16 lanes)
GDN_TARGET_AVX512
inline __m512 gdn_half_load16(const uint16_t* p)
{
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) p));
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
    float* recurrent_state;        // (rows, nv, dk, dv) fp32 view, scan rows updated in place
    uint16_t* recurrent_state_h;   // same pool as fp16 (recurrent_dtype = half since 02446529)
    int state_half;                // pool storage dtype: 0 = fp32, 1 = fp16
    int64_t state_row_stride;      // elements per pool row (dtype-independent)
    const float* norm_w;           // (dv,) fp32
    at::Half* out;                 // (rows, nv * dv) half, gated norm output
    // scratch
    uint16_t* mixed;               // (rows, fdim) bf16, conv output
    uint16_t* beta;                // (rows, nv) bf16
    float* g;                      // (rows, nv) fp32
    uint16_t* cao;                 // (rows, nv * dv) bf16, scan output
    float* wscr;                   // chunk-scan worker scratch, wscr_stride floats per worker
    size_t wscr_stride;
    int barriers;                  // chunk pool: 1 = legacy two-barrier phase schedule
    int64_t conv_slot_stride;      // conv_state plane stride
    // dims / scalars
    int rows;                      // token rows: bsz (decode) or bsz * T (chunk scan)
    int T;                         // tokens per batch row (chunk scan)
    int chunk;                     // scan chunk length C (chunk scan)
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
static void gdn_gates_heads(GdnCtx& c, int j0, int j1)
{
    if (c.wb == nullptr || c.wa == nullptr || j0 >= j1) return;
    const int nv = c.nv, hidden = c.hidden;
    alignas(64) float accb[512], acca[512];
    for (int j = j0; j < j1; ++j) { accb[j] = 0.0f; acca[j] = 0.0f; }
    for (int b = 0; b < c.rows; ++b)
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

GDN_TARGET_AVX512
static void gdn_gates_range(GdnCtx& c, int worker, int nw)
{
    const int j0 = (int) ((int64_t) c.nv * worker / nw);
    const int j1 = (int) ((int64_t) c.nv * (worker + 1) / nw);
    gdn_gates_heads(c, j0, j1);
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
        uint16_t* cs = c.conv_state + (size_t) c.conv_slot[b] * c.conv_slot_stride;
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
// u = beta (v - decay k^T state), state <- state * decay + k u^T, out = bf16(scale q^T state).
// SH: the state pool's storage dtype (fp16 since recurrent_dtype defaulted to half in 02446529).
// The scan math is fp32 either way; the fp16 pool converts on the load and rounds RNE once per
// step on the store, matching the torch fallback's per-pass rounding (gated_delta_rule.py)
template <bool SH>
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
        float* st;
        uint16_t* sth;
        if constexpr (SH) sth = c.recurrent_state_h + (size_t) c.scan_slot[b] * c.state_row_stride;
        else st = c.recurrent_state + (size_t) c.scan_slot[b] * c.state_row_stride;
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
            const size_t soff = (size_t) i * dk * dv;
            const uint16_t* vp = vr + (size_t) i * dv;
            uint16_t* op = c.cao + (size_t) (b * nv + i) * dv;
            const __m512 dcy = _mm512_set1_ps(decay), bcy = _mm512_set1_ps(betaf);
            int v = 0;
            for (; v + 16 <= dv; v += 16)
            {
                // kv_mem = k^T state (undecayed readback)
                __m512 acc = _mm512_setzero_ps();
                for (int d = 0; d < dk; ++d)
                {
                    __m512 sv;
                    if constexpr (SH) sv = gdn_half_load16(sth + soff + (size_t) d * dv + v);
                    else sv = _mm512_loadu_ps(st + soff + (size_t) d * dv + v);
                    acc = _mm512_fmadd_ps(sv, _mm512_set1_ps(kf[d]), acc);
                }
                _mm512_storeu_ps(kvm + v, acc);
                // u = beta (v - decay kv_mem)
                __m512 vv = gdn_bf16_load(vp + v);
                __m512 uv = _mm512_mul_ps(_mm512_fnmadd_ps(_mm512_loadu_ps(kvm + v), dcy, vv), bcy);
                _mm512_storeu_ps(u + v, uv);
                // state <- state * decay + k u^T ; out = scale q^T state (updated state)
                __m512 oac = _mm512_setzero_ps();
                for (int d = 0; d < dk; ++d)
                {
                    __m512 sv;
                    if constexpr (SH) sv = gdn_half_load16(sth + soff + (size_t) d * dv + v);
                    else sv = _mm512_loadu_ps(st + soff + (size_t) d * dv + v);
                    sv = _mm512_fmadd_ps(_mm512_set1_ps(kf[d]), uv, _mm512_mul_ps(sv, dcy));
                    if constexpr (SH) gdn_half_store((at::Half*) (sth + soff + (size_t) d * dv + v), sv);
                    else _mm512_storeu_ps(st + soff + (size_t) d * dv + v, sv);
                    oac = _mm512_fmadd_ps(_mm512_set1_ps(qf[d]), sv, oac);
                }
                gdn_bf16_store(op + v, _mm512_mul_ps(oac, _mm512_set1_ps(scale)));
            }
            for (; v < dv; ++v)
            {
                float acc = 0.0f;
                for (int d = 0; d < dk; ++d)
                {
                    float sd;
                    if constexpr (SH) sd = gdn_half_to_f32(sth[soff + (size_t) d * dv + v]);
                    else sd = st[soff + (size_t) d * dv + v];
                    acc = std::fma(sd, kf[d], acc);
                }
                const float uv = (gdn_bf16_to_f32(vp[v]) - acc * decay) * betaf;
                float oac = 0.0f;
                for (int d = 0; d < dk; ++d)
                {
                    const size_t off = soff + (size_t) d * dv + v;
                    if constexpr (SH)
                    {
                        float sd = gdn_half_to_f32(sth[off]);
                        sd = sd * decay + kf[d] * uv;
                        sth[off] = gdn_f32_to_half(sd);
                        oac = std::fma(qf[d], sd, oac);
                    }
                    else
                    {
                        float& sd = st[off];
                        sd = sd * decay + kf[d] * uv;
                        oac = std::fma(qf[d], sd, oac);
                    }
                }
                op[v] = gdn_f32_to_bf16(oac * scale);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
//   Prefill chunk scan (gdn_chunk_scan_cpu): the split-projection GDN middle section over T
//   tokens in one call -- b/a gates + full-sequence conv + the chunked WY delta rule +
//   gated norm -- mirroring modules/gated_delta_net_fn/cpu.py (gdn_fused_op_2_cpu,
//   causal_conv1d_update_cpu), gated_delta_rule.py (torch_chunk_gated_delta_rule_cpu) and
//   GatedRMSNorm._forward_cpu. No history rows: the state row is read at t = 0 and written
//   back once at t = T-1.
// -------------------------------------------------------------------------------------------

GDN_TARGET_AVX512
inline float gdn_dot(const float* a, const float* b, int n)
{
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i)
        s = std::fma(a[i], b[i], s);
    return s;
}

GDN_TARGET_AVX512
inline void gdn_l2norm_row(const uint16_t* p, float* o, int n)
{
    const float inv = 1.0f / std::sqrt(gdn_sumsq_hp(p, n) + 1e-6f);
    const __m512 iv = _mm512_set1_ps(inv);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(o + i, _mm512_mul_ps(gdn_bf16_load(p + i), iv));
    for (; i < n; ++i)
        o[i] = gdn_bf16_to_f32(p[i]) * inv;
}

// Depthwise causal conv over [state window | qkv_0..T-1] + SiLU into mixed, one token at a
// time, the window walking forward in registers; ends holding the last kdim raw
// (bf16-rounded) tokens, which is what causal_conv1d_update_cpu stores
// (history = False, channel-major bf16 token pre-round as in gdn_conv_range)
GDN_TARGET_AVX512
static void gdn_conv_seq_channels(GdnCtx& c, int c0, int c1)
{
    const int K = c.kdim, fdim = c.fdim;
    TORCH_CHECK(K >= 1 && K <= 16, "gdn_chunk_scan_cpu: kdim <= 16 supported");
    for (int b = 0; b < c.bsz; ++b)
    {
        uint16_t* cs = c.conv_state + (size_t) c.conv_slot[b] * c.conv_slot_stride;
        const uint16_t* qr = c.qkv + (size_t) b * c.T * fdim;
        uint16_t* mr = c.mixed + (size_t) b * c.T * fdim;
        int ci = c0;
        for (; ci + 16 <= c1; ci += 16)
        {
            alignas(64) __m512 win[16];
            for (int k = 0; k < K; ++k)
            {
                alignas(64) float s[16];
                for (int u = 0; u < 16; ++u)
                    s[u] = gdn_bf16_to_f32(cs[(size_t) (ci + u) * c.conv_row_stride + k]);
                win[k] = _mm512_load_ps(s);
            }
            for (int t = 0; t < c.T; ++t)
            {
                __m512 y = c.conv_b ? _mm512_loadu_ps(c.conv_b + ci) : _mm512_setzero_ps();
                for (int k = 0; k < K - 1; ++k)
                    y = _mm512_fmadd_ps(win[k + 1], _mm512_loadu_ps(c.conv_w + (size_t) k * fdim + ci), y);
                // the torch path rounds the fp16 projection to bf16 before the conv and the
                // state keeps those bf16 values (gdn_conv_range)
                __m512i xw = _mm512_castps_si512(_mm512_cvtph_ps(_mm256_loadu_si256(
                    (const __m256i*) (qr + (size_t) t * fdim + ci))));
                __m512i xr = _mm512_add_epi32(xw, _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                                  _mm512_and_si512(_mm512_srli_epi32(xw, 16), _mm512_set1_epi32(1))));
                __m512 xb = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm512_cvtepi32_epi16(_mm512_srli_epi32(xr, 16))), 16));
                y = _mm512_fmadd_ps(xb, _mm512_loadu_ps(c.conv_w + (size_t) (K - 1) * fdim + ci), y);
                gdn_bf16_store(mr + (size_t) t * fdim + ci, gdn_silu512(y));
                for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
                win[K - 1] = xb;
            }
            for (int k = 0; k < K; ++k)
            {
                alignas(64) float s[16];
                _mm512_store_ps(s, win[k]);
                for (int u = 0; u < 16; ++u)
                    cs[(size_t) (ci + u) * c.conv_row_stride + k] = gdn_f32_to_bf16(s[u]);
            }
        }
        for (; ci < c1; ++ci)
        {
            float win[16];
            for (int k = 0; k < K; ++k)
                win[k] = gdn_bf16_to_f32(cs[(size_t) ci * c.conv_row_stride + k]);
            for (int t = 0; t < c.T; ++t)
            {
                const uint16_t xbits = gdn_f32_to_bf16(_cvtsh_ss(qr[(size_t) t * fdim + ci]));
                const float xb = gdn_bf16_to_f32(xbits);
                float y = c.conv_b ? c.conv_b[ci] : 0.0f;
                for (int k = 0; k < K - 1; ++k)
                    y = std::fma(win[k + 1], c.conv_w[(size_t) k * fdim + ci], y);
                y = std::fma(xb, c.conv_w[(size_t) (K - 1) * fdim + ci], y);
                mr[(size_t) t * fdim + ci] = gdn_f32_to_bf16(gdn_silu1(y));
                for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
                win[K - 1] = xb;
            }
            for (int k = 0; k < K; ++k)
                cs[(size_t) ci * c.conv_row_stride + k] = gdn_f32_to_bf16(win[k]);
        }
    }
}

GDN_TARGET_AVX512
static void gdn_conv_seq_range(GdnCtx& c, int worker, int nw)
{
    const int fdim = c.fdim;
    gdn_conv_seq_channels(c, (int) ((int64_t) fdim * worker / nw),
                          (int) ((int64_t) fdim * (worker + 1) / nw));
}

// the worker's own channels: the k/q blocks of its k-heads plus its v-heads' block, so every
// mixed channel is produced by exactly one worker before that worker's scan reads it
GDN_TARGET_AVX512
static void gdn_conv_seq_parts(GdnCtx& c, int h0, int h1)
{
    const int group = c.nv / c.nk;
    const int k0 = h0 / group, k1 = h1 / group;
    const int qoff = c.nk * c.dk, voff = 2 * c.nk * c.dk;
    if (k1 > k0)
    {
        gdn_conv_seq_channels(c, k0 * c.dk, k1 * c.dk);
        gdn_conv_seq_channels(c, qoff + k0 * c.dk, qoff + k1 * c.dk);
    }
    gdn_conv_seq_channels(c, voff + h0 * c.dv, voff + h1 * c.dv);
}

// T-token grouped delta-rule scan, WY chunk form, mirroring torch_chunk_gated_delta_rule_cpu
// (gated_delta_rule.py:85): within a chunk of n tokens, G = cumsum(g), D = G_i - G_j,
// A = tril_strict((k_i . k_j) e^D beta_i), X = (I + A)^-1 by forward substitution (unit lower
// triangular), w = X (beta e^G k), u = X (beta v), v_new = u - w h,
// o_i = e^{G_i} (q_i h) + sum_{j<=i} (q_i . k_j) e^{D_ij} v_new_j, out = bf16(scale o), and the
// chunk closes with h <- h e^{G_last} + sum_i k_i (v_new_i e^{G_last - G_i}). q/k l2norm
// (eps 1e-6) on the bf16 conv output, one k-head shared by its v-head group, as in the decode
// kernel. h lives fp32 in the worker's scratch: read from the pool row (fp16 pools convert on
// the load) once at t = 0, written back (round RNE) once at t = T-1.
template <bool SH>
GDN_TARGET_AVX512
static void gdn_chunk_scan_range(GdnCtx& c, int worker, int h0, int h1)
{
    const int nv = c.nv, dk = c.dk, dv = c.dv, T = c.T, C = c.chunk;
    const int group = nv / c.nk;
    const float scale = 1.0f / std::sqrt((float) dk);
    if (h0 >= h1) return;
    float* W = c.wscr + (size_t) worker * c.wscr_stride;
    float* hloc_grp = W; W += (size_t) group * dk * dv;
    float* kn_ = W; W += (size_t) C * dk;
    float* qn_ = W; W += (size_t) C * dk;
    float* w_ = W; W += (size_t) C * dk;
    float* u_ = W; W += (size_t) C * dv;
    float* vnr = W; W += (size_t) C * dv;
    float* o1 = W; W += (size_t) C * dv;
    float* vv = W; W += (size_t) C * dv;
    float* ktk = W; W += (size_t) C * C;
    float* qk = W; W += (size_t) C * C;
    float* ee = W; W += (size_t) C * C;
    float* xx = W; W += (size_t) C * C;
    float* aa = W; W += (size_t) C * C;
    float* Gc = W; W += C;
    float* eG = W; W += C;
    float* bc = W;
    for (int b = 0; b < c.bsz; ++b)
    {
        const uint16_t* mr = c.mixed + (size_t) b * T * c.fdim;
        const uint16_t* qr = mr;
        const uint16_t* kr = mr + c.nk * dk;
        const uint16_t* vr = mr + 2 * c.nk * dk;
        float* st;
        uint16_t* sth;
        if constexpr (SH) sth = c.recurrent_state_h + (size_t) c.scan_slot[b] * c.state_row_stride;
        else st = c.recurrent_state + (size_t) c.scan_slot[b] * c.state_row_stride;
        // one k-head group at a time: its q/k l2norm rows are filled once per chunk right
        // before the group's heads consume them, and the heads' fp32 state lives in group
        // scratch for the whole pass (fp32 slot round-trips between chunks are exact)
        const int k0 = h0 / group, k1 = (h1 - 1) / group + 1;
        for (int kh = k0; kh < k1; ++kh)
        {
            const int i_lo = std::max(h0, kh * group);
            const int i_hi = std::min(h1, kh * group + group);
            if (i_lo >= i_hi) continue;
            for (int i = i_lo; i < i_hi; ++i)
            {
                const size_t soff = (size_t) i * dk * dv;
                float* hloc = hloc_grp + (size_t) (i - i_lo) * dk * dv;
                if constexpr (SH)
                {
                    for (int e = 0; e + 16 <= dk * dv; e += 16)
                        _mm512_storeu_ps(hloc + e, gdn_half_load16(sth + soff + e));
                    for (int e = dk * dv & ~15; e < dk * dv; ++e)
                        hloc[e] = gdn_half_to_f32(sth[soff + e]);
                }
                else
                {
                    for (int e = 0; e + 16 <= dk * dv; e += 16)
                        _mm512_storeu_ps(hloc + e, _mm512_loadu_ps(st + soff + e));
                    for (int e = dk * dv & ~15; e < dk * dv; ++e)
                        hloc[e] = st[soff + e];
                }
            }
            for (int t0 = 0; t0 < T; t0 += C)
            {
                const int n = std::min(C, T - t0);
                for (int j = 0; j < n; ++j)
                {
                    const int t = t0 + j;
                    gdn_l2norm_row(kr + (size_t) t * c.fdim + (size_t) kh * dk, kn_ + (size_t) j * dk, dk);
                    gdn_l2norm_row(qr + (size_t) t * c.fdim + (size_t) kh * dk, qn_ + (size_t) j * dk, dk);
                }
                for (int i = i_lo; i < i_hi; ++i)
                {
                    float* hloc = hloc_grp + (size_t) (i - i_lo) * dk * dv;
                    for (int j = 0; j < n; ++j)
                    {
                        const int t = t0 + j;
                        Gc[j] = (j ? Gc[j - 1] : 0.0f) + c.g[(size_t) (b * T + t) * nv + i];
                        eG[j] = std::exp(Gc[j]);
                        bc[j] = gdn_bf16_to_f32(c.beta[(size_t) (b * T + t) * nv + i]);
                    }
                for (int a = 0; a < n; ++a)
                {
                    const float* kna = kn_ + (size_t) a * dk;
                    for (int j = 0; j <= a; ++j)
                    {
                        ktk[(size_t) a * n + j] = gdn_dot(kna, kn_ + (size_t) j * dk, dk);
                        qk[(size_t) a * n + j] = gdn_dot(qn_ + (size_t) a * dk, kn_ + (size_t) j * dk, dk);
                    }
                    const float ga = Gc[a];
                    const __m512 gav = _mm512_set1_ps(ga);
                    int j = 0;
                    for (; j + 16 <= a + 1; j += 16)
                        _mm512_storeu_ps(ee + (size_t) a * n + j,
                                         gdn_exp512(_mm512_sub_ps(gav, _mm512_loadu_ps(Gc + j))));
                    for (; j <= a; ++j)
                        ee[(size_t) a * n + j] = std::exp(ga - Gc[j]);
                    for (int u = 0; u < a; ++u)
                        aa[(size_t) a * n + u] = ktk[(size_t) a * n + u] * ee[(size_t) a * n + u] * bc[a];
                }
                for (int a = 0; a < n; ++a)
                {
                    // xa[j] = -sum_{m=j..a-1} aa[a][m] xx[m][j]: accumulate across j in vector
                    // lanes, m ascending so every lane's FMA chain matches the scalar order,
                    // negate once at the store (same -0.0/+0.0 behavior as the scalar -s)
                    float* xa = xx + (size_t) a * n;
                    const float* aaa = aa + (size_t) a * n;
                    const int nb = (a + 15) >> 4;
                    __m512 acc[4];
                    for (int jb = 0; jb < nb; ++jb) acc[jb] = _mm512_setzero_ps();
                    for (int m = 0; m < a; ++m)
                    {
                        const __m512 av = _mm512_set1_ps(aaa[m]);
                        const float* xm = xx + (size_t) m * n;
                        int jb = 0;
                        for (; jb < (m >> 4); ++jb)
                            acc[jb] = _mm512_fmadd_ps(av, _mm512_loadu_ps(xm + (jb << 4)), acc[jb]);
                        const __mmask16 km = (__mmask16) ((1u << ((m & 15) + 1)) - 1u);
                        acc[jb] = _mm512_fmadd_ps(av, _mm512_maskz_loadu_ps(km, xm + (jb << 4)), acc[jb]);
                    }
                    const __m512 sn = _mm512_set1_ps(-0.0f);
                    int jb = 0;
                    for (; (jb << 4) + 16 <= a; ++jb)
                        _mm512_storeu_ps(xa + (jb << 4), _mm512_xor_ps(acc[jb], sn));
                    if ((jb << 4) < a)
                        _mm512_mask_storeu_ps(xa + (jb << 4),
                                              (__mmask16) ((1u << (a - (jb << 4))) - 1u),
                                              _mm512_xor_ps(acc[jb], sn));
                    xa[a] = 1.0f;
                }
                for (int a = 0; a < n; ++a)
                {
                    // wa/ua accumulate in zmm registers (8 lanes-wide tiles of 128 elements,
                    // j ascending per element = the original order), one store per tile
                    float* wa = w_ + (size_t) a * dk;
                    float* ua = u_ + (size_t) a * dv;
                    for (int d = 0; d < dk; ++d) wa[d] = 0.0f;
                    for (int v = 0; v < dv; ++v) ua[v] = 0.0f;
                    const float* xa = xx + (size_t) a * n;
                    for (int d0 = 0; d0 < dk; d0 += 128)
                    {
                        const int d1 = std::min(dk, d0 + 128);
                        const int fb = (d1 - d0) >> 4;
                        __m512 acc[8];
                        for (int bl = 0; bl < fb; ++bl) acc[bl] = _mm512_setzero_ps();
                        for (int j = 0; j <= a; ++j)
                        {
                            const float bge = xa[j] * (bc[j] * eG[j]);
                            const __m512 bgev = _mm512_set1_ps(bge);
                            const float* knj = kn_ + (size_t) j * dk + d0;
                            int bl = 0;
                            for (; bl < fb; ++bl)
                                acc[bl] = _mm512_fmadd_ps(bgev, _mm512_loadu_ps(knj + (bl << 4)), acc[bl]);
                            for (int d = d0 + (fb << 4); d < d1; ++d)
                                wa[d] = std::fma(bge, knj[d - d0], wa[d]);
                        }
                        for (int bl = 0; bl < fb; ++bl)
                            _mm512_storeu_ps(wa + d0 + (bl << 4), acc[bl]);
                    }
                    for (int v0 = 0; v0 < dv; v0 += 128)
                    {
                        const int v1 = std::min(dv, v0 + 128);
                        const int fb = (v1 - v0) >> 4;
                        __m512 acc[8];
                        for (int bl = 0; bl < fb; ++bl) acc[bl] = _mm512_setzero_ps();
                        for (int j = 0; j <= a; ++j)
                        {
                            const float xbj = xa[j] * bc[j];
                            const __m512 xbjv = _mm512_set1_ps(xbj);
                            const uint16_t* vp = vr + (size_t) (t0 + j) * c.fdim + (size_t) i * dv + v0;
                            int bl = 0;
                            for (; bl < fb; ++bl)
                                acc[bl] = _mm512_fmadd_ps(xbjv, gdn_bf16_load(vp + (bl << 4)), acc[bl]);
                            for (int v = v0 + (fb << 4); v < v1; ++v)
                                ua[v] = std::fma(xbj, gdn_bf16_to_f32(vp[v - v0]), ua[v]);
                        }
                        for (int bl = 0; bl < fb; ++bl)
                            _mm512_storeu_ps(ua + v0 + (bl << 4), acc[bl]);
                    }
                }
                for (int a = 0; a < n; ++a)
                {
                    const float* wa = w_ + (size_t) a * dk;
                    const float* qa = qn_ + (size_t) a * dk;
                    float* va = vnr + (size_t) a * dv;
                    float* oa = o1 + (size_t) a * dv;
                    int v = 0;
                    for (; v + 16 <= dv; v += 16)
                    {
                        __m512 oac = _mm512_setzero_ps();
                        __m512 uu = _mm512_loadu_ps(u_ + (size_t) a * dv + v);
                        for (int d = 0; d < dk; ++d)
                        {
                            const __m512 hv = _mm512_loadu_ps(hloc + (size_t) d * dv + v);
                            oac = _mm512_fmadd_ps(_mm512_set1_ps(qa[d]), hv, oac);
                            uu = _mm512_fnmadd_ps(_mm512_set1_ps(wa[d]), hv, uu);
                        }
                        _mm512_storeu_ps(va + v, uu);
                        _mm512_storeu_ps(oa + v, oac);
                    }
                    for (; v < dv; ++v)
                    {
                        float oac = 0.0f;
                        float uu = u_[(size_t) a * dv + v];
                        for (int d = 0; d < dk; ++d)
                        {
                            const float hv = hloc[(size_t) d * dv + v];
                            oac = std::fma(qa[d], hv, oac);
                            uu = std::fma(-wa[d], hv, uu);
                        }
                        va[v] = uu;
                        oa[v] = oac;
                    }
                }
                for (int a = 0; a < n; ++a)
                {
                    const int t = t0 + a;
                    uint16_t* op = c.cao + ((size_t) (b * T + t) * nv + i) * dv;
                    const float* qka = qk + (size_t) a * n;
                    const float* eea = ee + (size_t) a * n;
                    int v = 0;
                    for (; v + 16 <= dv; v += 16)
                    {
                        __m512 o2 = _mm512_setzero_ps();
                        for (int j = 0; j <= a; ++j)
                            o2 = _mm512_fmadd_ps(_mm512_set1_ps(qka[j] * eea[j]),
                                                 _mm512_loadu_ps(vnr + (size_t) j * dv + v), o2);
                        __m512 o = _mm512_fmadd_ps(_mm512_set1_ps(eG[a]),
                                                   _mm512_loadu_ps(o1 + (size_t) a * dv + v), o2);
                        gdn_bf16_store(op + v, _mm512_mul_ps(o, _mm512_set1_ps(scale)));
                    }
                    for (; v < dv; ++v)
                    {
                        float o2 = 0.0f;
                        for (int j = 0; j <= a; ++j)
                            o2 = std::fma(qka[j] * eea[j], vnr[(size_t) j * dv + v], o2);
                        op[v] = gdn_f32_to_bf16(std::fma(eG[a], o1[(size_t) a * dv + v], o2) * scale);
                    }
                }
                const float eGl = std::exp(Gc[n - 1]);
                {
                    const __m512 eGlv = _mm512_set1_ps(eGl);
                    for (int d = 0; d < dk; ++d)
                    {
                        int v = 0;
                        for (; v + 16 <= dv; v += 16)
                            _mm512_storeu_ps(hloc + (size_t) d * dv + v,
                                             _mm512_mul_ps(_mm512_loadu_ps(hloc + (size_t) d * dv + v), eGlv));
                        for (; v < dv; ++v) hloc[(size_t) d * dv + v] *= eGl;
                    }
                }
                for (int a = 0; a < n; ++a)
                {
                    const float sfa = std::exp(Gc[n - 1] - Gc[a]);
                    const __m512 sfv = _mm512_set1_ps(sfa);
                    float* va = vv + (size_t) a * dv;
                    const float* va0 = vnr + (size_t) a * dv;
                    int v = 0;
                    for (; v + 16 <= dv; v += 16)
                        _mm512_storeu_ps(va + v, _mm512_mul_ps(sfv, _mm512_loadu_ps(va0 + v)));
                    for (; v < dv; ++v) va[v] = sfa * va0[v];
                }
                for (int a = 0; a < n; ++a)
                {
                    const float* kna = kn_ + (size_t) a * dk;
                    const float* va = vv + (size_t) a * dv;
                    for (int d = 0; d < dk; ++d)
                    {
                        const __m512 kv = _mm512_set1_ps(kna[d]);
                        int v = 0;
                        for (; v + 16 <= dv; v += 16)
                            _mm512_storeu_ps(hloc + (size_t) d * dv + v,
                                             _mm512_fmadd_ps(kv, _mm512_loadu_ps(va + v),
                                                             _mm512_loadu_ps(hloc + (size_t) d * dv + v)));
                        for (; v < dv; ++v)
                            hloc[(size_t) d * dv + v] = std::fma(kna[d], va[v], hloc[(size_t) d * dv + v]);
                    }
                }
                if (t0 + n == T)
                {
                    const size_t soff = (size_t) i * dk * dv;
                    if constexpr (SH)
                    {
                        for (int e = 0; e + 16 <= dk * dv; e += 16)
                            gdn_half_store((at::Half*) (sth + soff + e), _mm512_loadu_ps(hloc + e));
                        for (int e = dk * dv & ~15; e < dk * dv; ++e)
                            sth[soff + e] = gdn_f32_to_half(hloc[e]);
                    }
                    else
                    {
                        for (int e = 0; e + 16 <= dk * dv; e += 16)
                            _mm512_storeu_ps(st + soff + e, _mm512_loadu_ps(hloc + e));
                        for (int e = dk * dv & ~15; e < dk * dv; ++e)
                            st[soff + e] = hloc[e];
                    }
                }
            }
        }
    }
    }
}

// gated RMS norm over the scan output: h = bf16(scan out) -> f32, * rsqrt(mean h^2 + eps),
// * (weight + bias), * gate(z), -> fp16 (GatedRMSNorm._forward_cpu, not gate_first)
GDN_TARGET_AVX512
static void gdn_norm_row(GdnCtx& c, int r)
{
    const int nv = c.nv, dv = c.dv;
    const __m512 eps = _mm512_set1_ps(c.rms_eps);
    const __m512 bias = _mm512_set1_ps(c.norm_bias);
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

GDN_TARGET_AVX512
static void gdn_norm_range(GdnCtx& c, int worker, int nw)
{
    const int rows = c.rows * c.nv;
    const int r0 = (int) ((int64_t) rows * worker / nw);
    const int r1 = (int) ((int64_t) rows * (worker + 1) / nw);
    for (int r = r0; r < r1; ++r) gdn_norm_row(c, r);
}

static void gdn_norm_heads(GdnCtx& c, int h0, int h1)
{
    for (int tr = 0; tr < c.rows; ++tr)
        for (int i = h0; i < h1; ++i)
            gdn_norm_row(c, tr * c.nv + i);
}

static void gdn_pool_fn(void* vctx, int worker, int nw)
{
    GdnCtx& c = *static_cast<GdnCtx*>(vctx);
    gdn_gates_range(c, worker, nw);
    gdn_conv_range(c, worker, nw);
    c.bar1.fetch_add(1, std::memory_order_acq_rel);
    while (c.bar1.load(std::memory_order_acquire) < nw) gdn_pause();
    if (c.state_half) gdn_scan_range<true>(c, worker, nw);
    else gdn_scan_range<false>(c, worker, nw);
    c.bar2.fetch_add(1, std::memory_order_acq_rel);
    while (c.bar2.load(std::memory_order_acquire) < nw) gdn_pause();
    gdn_norm_range(c, worker, nw);
}

// k-head-group partition: each worker owns whole k-head groups, so its v-heads' gate/norm
// work, its k-heads' conv channel blocks and its scan rows all pair up with no cross-worker
// dependency (v-heads of one k-head never span workers)
static inline void gdn_chunk_head_range(const GdnCtx& c, int worker, int nw, int& h0, int& h1)
{
    const int group = c.nv / c.nk;
    const int k0 = (int) ((int64_t) c.nk * worker / nw);
    const int k1 = (int) ((int64_t) c.nk * (worker + 1) / nw);
    h0 = k0 * group;
    h1 = k1 * group;
}

// gates | conv, then the chunk scan (q/k l2norm in-chunk), then the norm epilogue;
// default head-partitioned barrier-free, EXL3_GDN_CHUNK_SCAN_BARRIERS=1 restores the
// two-barrier phase schedule with the plain head/channel partitions
static void gdn_chunk_pool_fn(void* vctx, int worker, int nw)
{
    GdnCtx& c = *static_cast<GdnCtx*>(vctx);
    if (c.barriers)
    {
        gdn_gates_range(c, worker, nw);
        gdn_conv_seq_range(c, worker, nw);
        c.bar1.fetch_add(1, std::memory_order_acq_rel);
        while (c.bar1.load(std::memory_order_acquire) < nw) gdn_pause();
        const int h0 = (int) ((int64_t) c.nv * worker / nw);
        const int h1 = (int) ((int64_t) c.nv * (worker + 1) / nw);
        if (c.state_half) gdn_chunk_scan_range<true>(c, worker, h0, h1);
        else gdn_chunk_scan_range<false>(c, worker, h0, h1);
        c.bar2.fetch_add(1, std::memory_order_acq_rel);
        while (c.bar2.load(std::memory_order_acquire) < nw) gdn_pause();
        gdn_norm_range(c, worker, nw);
        return;
    }
    int h0, h1;
    gdn_chunk_head_range(c, worker, nw, h0, h1);
    if (h0 >= h1) return;
    gdn_gates_heads(c, h0, h1);
    gdn_conv_seq_parts(c, h0, h1);
    if (c.state_half) gdn_chunk_scan_range<true>(c, worker, h0, h1);
    else gdn_chunk_scan_range<false>(c, worker, h0, h1);
    gdn_norm_heads(c, h0, h1);
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
    TORCH_CHECK((recurrent_state.scalar_type() == at::kFloat || recurrent_state.scalar_type() == at::kHalf) &&
                recurrent_state.is_contiguous() &&
                recurrent_state.dim() == 5 && recurrent_state.size(1) == 1,
                "recurrent_state must be contiguous CPU float or half (rows, 1, nv, dk, dv)");
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
    c.state_half = recurrent_state.scalar_type() == at::kHalf ? 1 : 0;
    c.recurrent_state = c.state_half ? nullptr : recurrent_state.data_ptr<float>();
    c.recurrent_state_h = c.state_half ? reinterpret_cast<uint16_t*>(recurrent_state.data_ptr<at::Half>()) : nullptr;
    c.state_row_stride = recurrent_state.stride(0);
    c.norm_w = norm_w.data_ptr<float>();
    c.out = out.data_ptr<at::Half>();
    c.mixed = reinterpret_cast<uint16_t*>(mixed.data_ptr<at::BFloat16>());
    c.beta = reinterpret_cast<uint16_t*>(beta.data_ptr<at::BFloat16>());
    c.g = g.data_ptr<float>();
    c.cao = reinterpret_cast<uint16_t*>(cao.data_ptr<at::BFloat16>());
    c.bsz = bsz;
    c.rows = bsz;
    c.T = 1;
    c.chunk = 0;
    c.conv_slot_stride = conv_state.stride(0);
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

constexpr int GDN_SCAN_CHUNK = 64;   // the torch chunk reference's default chunk_size
static_assert(GDN_SCAN_CHUNK <= 64); // forward substitution keeps 4 vector accumulators per row

void gdn_chunk_scan_cpu
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
    TORCH_CHECK(gdn_step_cpu_has_avx512(), "gdn_chunk_scan_cpu: AVX-512 not available");
    TORCH_CHECK(x.is_cpu() && qkv.is_cpu() && z.is_cpu() && conv_state.is_cpu() &&
                recurrent_state.is_cpu() && out.is_cpu(), "gdn_chunk_scan_cpu: tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 3,
                "x must be contiguous CPU half (bsz, T, hidden)");
    TORCH_CHECK(qkv.scalar_type() == at::kHalf && qkv.is_contiguous() && qkv.dim() == 3 &&
                qkv.size(0) == x.size(0) && qkv.size(1) == x.size(1),
                "qkv must be contiguous CPU half (bsz, T, fdim)");
    const int bsz = (int) x.size(0);
    const int T = (int) x.size(1);
    const int hidden = (int) x.size(2);
    const int fdim = (int) qkv.size(2);
    TORCH_CHECK(T >= 1, "gdn_chunk_scan_cpu: T >= 1 required");
    TORCH_CHECK(z.is_contiguous() && z.dim() == 4 && z.size(0) == bsz && z.size(1) == T,
                "z must be contiguous (bsz, T, nv, dv)");
    TORCH_CHECK(z.scalar_type() == (z_is_half ? at::kHalf : at::kFloat), "z dtype mismatch");
    const int nv = (int) z.size(2);
    const int dv = (int) z.size(3);
    TORCH_CHECK(nv >= 1 && nv <= 512 && dv >= 1 && dv <= GDN_MAX_DV, "gdn_chunk_scan_cpu: nv, dv <= 512 supported");
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
    TORCH_CHECK((recurrent_state.scalar_type() == at::kFloat || recurrent_state.scalar_type() == at::kHalf) &&
                recurrent_state.is_contiguous() &&
                recurrent_state.dim() == 5 && recurrent_state.size(1) == 1,
                "recurrent_state must be contiguous CPU float or half (rows, 1, nv, dk, dv)");
    TORCH_CHECK(recurrent_state.size(2) == nv && recurrent_state.size(4) == dv,
                "recurrent_state shape mismatch");
    const int dk = (int) recurrent_state.size(3);
    TORCH_CHECK(dk >= 1 && dk <= GDN_MAX_QK, "gdn_chunk_scan_cpu: dk <= 512 supported");
    TORCH_CHECK(norm_w.scalar_type() == at::kFloat && norm_w.is_contiguous() && norm_w.numel() == dv,
                "norm_w must be contiguous CPU float (dv,)");
    TORCH_CHECK(out.scalar_type() == at::kHalf && out.is_contiguous() &&
                out.numel() == (int64_t) bsz * T * nv * dv, "out must be contiguous CPU half (bsz, T, nv * dv)");
    TORCH_CHECK(conv_slots.numel() == bsz && scan_slots.numel() == bsz, "slots must have bsz entries");

    auto slot_of = [](const at::Tensor& t, int i) -> int64_t {
        return t.scalar_type() == at::kLong ? (int64_t) t.data_ptr<int64_t>()[i] : (int64_t) t.data_ptr<int>()[i];
    };

    const int rows = bsz * T;
    const int C = GDN_SCAN_CHUNK;
    int nk = -1;
    for (int t = 1; t <= nv; ++t)
        if (2 * t * dk + nv * dv == fdim) { nk = t; break; }
    TORCH_CHECK(nk > 0, "gdn_chunk_scan_cpu: fdim does not decompose as 2 * nk * dk + nv * dv");
    int nw = num_threads > 0 ? std::min(num_threads, gdn_chunk_scan_threads()) : gdn_chunk_scan_threads();
    nw = std::max(1, nw);
    const int group = nv / nk;
    const size_t wscr_stride = (size_t) group * dk * dv + (size_t) 3 * C * dk +
                               (size_t) 4 * C * dv + (size_t) 5 * C * C + (size_t) 3 * C;
    at::Tensor wscr = at::empty({(int64_t) nw * wscr_stride}, qkv.options().dtype(at::kFloat));

    auto bf16opts = qkv.options().dtype(at::kBFloat16);
    at::Tensor mixed = at::empty({rows, fdim}, bf16opts);
    at::Tensor beta = at::empty({rows, nv}, bf16opts);
    at::Tensor g = at::empty({rows, nv}, qkv.options().dtype(at::kFloat));
    at::Tensor cao = at::empty({(int64_t) rows * nv * dv}, bf16opts);

    GdnCtx c;
    c.x = reinterpret_cast<const uint16_t*>(x.data_ptr<at::Half>());
    c.qkv = reinterpret_cast<const uint16_t*>(qkv.data_ptr<at::Half>());
    c.z32 = z_is_half ? nullptr : z.data_ptr<float>();
    c.z16 = z_is_half ? reinterpret_cast<const uint16_t*>(z.data_ptr<at::Half>()) : nullptr;
    c.conv_state = reinterpret_cast<uint16_t*>(conv_state.data_ptr<at::BFloat16>());
    c.conv_row_stride = conv_state.size(2);
    c.conv_slot_stride = conv_state.stride(0);
    c.conv_w = conv_w.data_ptr<float>();
    c.conv_b = conv_b ? conv_b->data_ptr<float>() : nullptr;
    c.wb = wb ? reinterpret_cast<const uint16_t*>(wb->data_ptr<at::Half>()) : nullptr;
    c.bb = bb ? reinterpret_cast<const uint16_t*>(bb->data_ptr<at::Half>()) : nullptr;
    c.wa = wa ? reinterpret_cast<const uint16_t*>(wa->data_ptr<at::Half>()) : nullptr;
    c.ba = ba ? reinterpret_cast<const uint16_t*>(ba->data_ptr<at::Half>()) : nullptr;
    c.dt_bias = dt_bias.data_ptr<float>();
    c.a_log = a_log.data_ptr<float>();
    c.state_half = recurrent_state.scalar_type() == at::kHalf ? 1 : 0;
    c.recurrent_state = c.state_half ? nullptr : recurrent_state.data_ptr<float>();
    c.recurrent_state_h = c.state_half ? reinterpret_cast<uint16_t*>(recurrent_state.data_ptr<at::Half>()) : nullptr;
    c.state_row_stride = recurrent_state.stride(0);
    c.norm_w = norm_w.data_ptr<float>();
    c.out = reinterpret_cast<at::Half*>(out.data_ptr<at::Half>());
    c.mixed = reinterpret_cast<uint16_t*>(mixed.data_ptr<at::BFloat16>());
    c.beta = reinterpret_cast<uint16_t*>(beta.data_ptr<at::BFloat16>());
    c.g = g.data_ptr<float>();
    c.cao = reinterpret_cast<uint16_t*>(cao.data_ptr<at::BFloat16>());
    c.wscr = wscr.data_ptr<float>();
    c.wscr_stride = wscr_stride;
    c.barriers = gdn_chunk_scan_barriers();
    c.bsz = bsz;
    c.rows = rows;
    c.T = T;
    c.chunk = C;
    c.hidden = hidden;
    c.fdim = fdim;
    c.kdim = kdim;
    c.nv = nv;
    c.nk = nk;
    c.dk = dk;
    c.dv = dv;
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

    std::lock_guard<std::mutex> lock(g_gdn_pool_mutex);
    g_gdn_pool.ensure(nw);
    g_gdn_pool.run(&gdn_chunk_pool_fn, &c, nw);
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

void gdn_chunk_scan_cpu
(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
    double, double, double, bool, bool, int
)
{
    TORCH_CHECK(false, "gdn_chunk_scan_cpu is not available on this CPU architecture (x86 AVX-512 only)");
}

bool gdn_step_cpu_has_avx512()
{
    return false;
}
#endif
