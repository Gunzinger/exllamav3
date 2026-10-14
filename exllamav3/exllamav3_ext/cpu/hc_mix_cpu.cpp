#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "hc_mix_cpu.h"
#include <c10/util/Half.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <immintrin.h>

#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

inline uint64_t hc_now_ns()
{
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Process affinity as it stood when the extension loaded, before any pool pinned the caller
// (see Pool::ensure). Function-local statics would capture a pinned mask; namespace-scope
// dynamic init runs at dlopen on the loading (main, unpinned) thread
#ifdef __linux__
struct HcAffinity
{
    cpu_set_t set;
    bool have;
    HcAffinity()
    {
        CPU_ZERO(&set);
        have = sched_getaffinity(0, sizeof(set), &set) == 0
               && CPU_COUNT(&set) > 0;
    }
};
static HcAffinity g_hc_worker_affinity;
#endif

// Fused GatedResidual hyper-connection mix on CPU (hc_mix_cpu.h). Structure mirrors the row
// kernels of moe_mul1.cpp: AVX-512 dot products over fp16 tables against the fp32 stream stack,
// fp32 accumulation, runtime ISA detection with a Python-side fallback when the tier is absent,
// and a persistent spin-parked worker pool (master participates as worker 0).

#define HC_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))

inline void cpu_pause()
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// -------------------------------------------------------------------------------------------
//   Thread pool (persistent, spin-parked; moe_mul1.cpp Pool, Linux-only pinning)
// -------------------------------------------------------------------------------------------

typedef void (*PoolFn)(void* ctx, int worker, int num_workers);

// Workers float (no pinning, unlike the moe pool): the mixes run in ~300 us bursts ~3 ms apart,
// on cores the moe pool's pinned workers keep busy between hc calls. A pinned hc worker woken
// under a busy moe worker waits out the CFS wakeup granularity (~ms, and evicts it), which
// measured worse than running the whole mix single-threaded; a floating worker is placed on an
// idle LP the moment it wakes. Floating only works if the mask the workers inherit at spawn is
// the full process set: Pool::ensure restores that mask (see HcAffinity), because the spawning
// thread is by then the moe/linear pool's pinned worker 0 whenever a prefill moe kernel ran
// first -- an inherited {core0} mask parks every hc worker and the master on one core and turns
// each ~0.5 ms mix into a ~20 ms timeslice (13x decode stall, sticky for the process lifetime).

struct Pool
{
    void worker_loop(int idx)
    {
        uint64_t seen = 0;
        int idle = 0;
        while (true)
        {
            const uint64_t g = dispatch.load(std::memory_order_acquire);
            if (g == seen)
            {
                // Short spin window, then park: unlike the moe pool (whose calls are back to
                // back), hc calls are ~3 ms apart with whole moe/torch phases in between, so a
                // long spin only steals SMT/core time from the other pools' workers. Parking
                // is a futex wait on the dispatch word (Linux) so the next dispatch's single
                // FUTEX_WAKE revives every worker in parallel; the fallback polls a short nap
                if (++idle < 512) { cpu_pause(); continue; }
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
            std::thread t(&Pool::worker_loop, this, spawned + 1);
#ifdef __linux__
            // Workers are detached and reused for the life of the process, and inherit the
            // spawning thread's affinity mask. By the first hc call that thread is usually the
            // moe/linear pool's pinned worker 0 (any prefill that runs a moe kernel pins the
            // caller to its core), so an inherited mask would confine every hc worker to that
            // one core with the master -- the ~0.5 ms mix turns into a ~20 ms futex-wake
            // timeslice (measured: 0.5 -> 7.1 tok/s decode, sticky for the process lifetime).
            // Restore the process's original mask, captured at ext load before any pool pins
            // anything; empty on failure means leave the inheritance alone.
            if (g_hc_worker_affinity.have)
                pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &g_hc_worker_affinity.set);
#endif
            t.detach();
            ++spawned;
        }
        if (n > num_workers) n_eff = n;   // first config or a raised env target: cap follows
        num_workers = n;
        n_eff = std::min(n_eff, n);
        static const bool adapt_v = [] {
            const char* e = std::getenv("EXL3_HC_MIX_CPU_ADAPT");
            return !(e && *e == '0');
        }();
        adapt_cap = adapt_v;
    }

    void run(PoolFn f, void* c, int n_req = 0)
    {
        int n = num_workers;
        if (n_req > 0 && n_req < n) n = n_req;
        if (n > n_eff) n = n_eff;
        if (n <= 1) { f(c, 0, 1); return; }
        ctx = c;
        fn.store(f, std::memory_order_relaxed);
        const uint64_t d0 = done.load(std::memory_order_acquire);
        const uint64_t t0 = hc_now_ns();
        ++generation;
        dispatch.store((generation << DISPATCH_NW_BITS) | (uint64_t) n, std::memory_order_release);
#ifdef __linux__
        syscall(SYS_futex, (void*) &dispatch, FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr, nullptr, 0);
#endif
        f(c, 0, n);
        while (static_cast<int64_t>(done.load(std::memory_order_acquire) - d0) < n - 1)
            cpu_pause();
        // One-way safety net against a pathologically slow pool: a healthy parked pool rejoins
        // in <= ~1 ms (measured 0.44 - 0.5 ms at 6 workers here, <= ~1 ms under load) while a
        // broken one (workers confined to one busy core, oversized pool against the scheduler)
        // sits at tens of ms, where single-threading wins. Eight joins over 4 ms (a step's
        // worth, counted after a 32-call spawn grace) cap the pool at one worker for the rest
        // of the process; EXL3_HC_MIX_CPU_ADAPT=0 disables the cap
        ++joins;
        if (joins > 32 && hc_now_ns() - t0 > 4000000 && ++slow_joins >= 8 && adapt_cap) n_eff = 1;
    }

    std::atomic<uint64_t> dispatch{0};
    std::atomic<uint64_t> done{0};
    std::atomic<PoolFn> fn{nullptr};
    void* ctx = nullptr;
    uint64_t generation = 0;
    int spawned = 0;
    int num_workers = 1;
    int n_eff = 1;
    uint64_t joins = 0;
    int slow_joins = 0;
    bool adapt_cap = true;
    static constexpr int DISPATCH_NW_BITS = 16;
    static int dispatch_nw(uint64_t d) { return (int) (d & ((1ull << DISPATCH_NW_BITS) - 1)); }
};

Pool g_pool;
std::mutex g_pool_mutex;

inline int hc_cpu_threads()
{
    // Explicit env, else a small default: the mixes are ~13 MB streaming bursts ~3 ms apart,
    // and beyond ~6 workers the wake/scheduler cost (and scatter across slow cores) outgrows
    // the bandwidth gain (measured: hc 45 ms/step at 6, 75 at 8, 205 at 12 on the bench model)
    static const int v = [] {
        int t = 0;
        const char* e = std::getenv("EXL3_HC_MIX_CPU_THREADS");
        if (e) t = std::atoi(e);
        if (t <= 0)
            t = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 4);
        return std::max(1, t);
    }();
    return v;
}

// -------------------------------------------------------------------------------------------
//   Kernels
// -------------------------------------------------------------------------------------------

#define GR_MAX_R 8

struct GrCtx
{
    const float* streams;    // (R, H, D)
    const uint16_t* fn;      // (M, H * D)
    const uint16_t* up;      // (H * D, LR)
    const float* w;          // (H * D)
    float* dots;             // (R, M + 1, H)
    float* post;             // (R, H), nullable
    at::Half* mixed;         // (R, D)
    float* t;                // (R, LR) workspace
    int R, H, D, LR, M;
    float rms_eps;
    std::atomic<int> bar;
};

inline float gr_sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// 16-lane fp16/bf16 -> fp32 convert; bf16 is the high half of an fp32 (plain 16-bit shift),
// exact, no avx512bf16 needed. Same latency class as vcvtph2ps on Zen 5
template <bool BF16>
HC_TARGET_AVX512
inline __m512 gr_cvt16_ps(const __m256i v)
{
    if (BF16) return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
    return _mm512_cvtph_ps(v);
}

// dots[j, h] = <streams * w, fn[j, h]> over one stream row
template <bool BF16>
HC_TARGET_AVX512
inline float gr_dot_w(const float* s, const float* w, const uint16_t* f, int D)
{
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < D; d += 16)
    {
        __m512 sv = _mm512_loadu_ps(s + d);
        __m512 wv = _mm512_loadu_ps(w + d);
        __m512 fv = gr_cvt16_ps<BF16>(_mm256_loadu_si256((const __m256i*) (f + d)));
        acc = _mm512_fmadd_ps(_mm512_mul_ps(sv, wv), fv, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

HC_TARGET_AVX512
inline float gr_sumsq(const float* s, int D)
{
    __m512 acc = _mm512_setzero_ps();
    for (int d = 0; d < D; d += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(s + d), _mm512_loadu_ps(s + d), acc);
    return _mm512_reduce_add_ps(acc);
}

// <t, up row> over one low rank
template <bool BF16>
HC_TARGET_AVX512
inline float gr_dot_t(const float* t, const uint16_t* u, int LR)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < LR; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(t + i),
                              gr_cvt16_ps<BF16>(_mm256_loadu_si256((const __m256i*) (u + i))), acc);
    return _mm512_reduce_add_ps(acc);
}

template <bool BF16>
static void gr_dots_range(const GrCtx& c, int i0, int i1)
{
    const int mh = c.M + 1;
    for (int it = i0; it < i1; ++it)
    {
        const int r = it / mh;
        const int j = it - r * mh;
        float* out = c.dots + (size_t) it * c.H;
        if (j < c.M)
        {
            const uint16_t* frow = c.fn + (size_t) j * (c.H * c.D);
            for (int h = 0; h < c.H; ++h)
                out[h] = gr_dot_w<BF16>(c.streams + ((size_t) r * c.H + h) * c.D,
                                        c.w + (size_t) h * c.D,
                                        frow + (size_t) h * c.D, c.D);
        }
        else
        {
            for (int h = 0; h < c.H; ++h)
                out[h] = gr_sumsq(c.streams + ((size_t) r * c.H + h) * c.D, c.D);
        }
    }
}

template <bool BF16>
static void gr_finalize_range(const GrCtx& c, int worker, int d0, int d1, float inv_h)
{
    const int H = c.H, D = c.D, LR = c.LR, R = c.R;
    float rmr[GR_MAX_R * 4];
    for (int r = 0; r < R; ++r)
    {
        const float* dr = c.dots + (size_t) r * (c.M + 1) * H;
        for (int h = 0; h < H; ++h)
            rmr[r * H + h] = 1.0f / std::sqrt(dr[(size_t) c.M * H + h] / (float) D + c.rms_eps);
    }
    for (int r = 0; r < R; ++r)
    {
        const float* dr = c.dots + (size_t) r * (c.M + 1) * H;
        float* tr = c.t + (size_t) r * LR;
        for (int i = 0; i < LR; ++i)
        {
            float v = 0.0f;
            for (int h = 0; h < H; ++h)
                v = std::fma(rmr[r * H + h], dr[(size_t) i * H + h], v);
            v *= inv_h;
            tr[i] = v * gr_sigmoid(v);
        }
        if (c.post && worker == 0)
            for (int h = 0; h < H; ++h)
            {
                float v = 0.0f;
                for (int h2 = 0; h2 < H; ++h2)
                    v = std::fma(rmr[r * H + h2], dr[(size_t) (LR + h) * H + h2], v);
                c.post[(size_t) r * H + h] = 2.0f * gr_sigmoid(v * inv_h);
            }
    }
    for (int d = d0; d < d1; ++d)
    {
        float o[GR_MAX_R];
        for (int r = 0; r < R; ++r) o[r] = 0.0f;
        for (int h = 0; h < H; ++h)
        {
            const uint16_t* u = c.up + ((size_t) h * D + d) * LR;
            const float wd = c.w[(size_t) h * D + d];
            for (int r = 0; r < R; ++r)
            {
                const float g = gr_dot_t<BF16>(c.t + (size_t) r * LR, u, LR);
                o[r] += gr_sigmoid(g) * rmr[r * H + h] * inv_h * wd
                        * c.streams[((size_t) r * H + h) * D + d];
            }
        }
        for (int r = 0; r < R; ++r)
            c.mixed[(size_t) r * D + d] = at::Half(o[r]);
    }
}

template <bool BF16>
static void gr_pool_fn(void* vctx, int worker, int nw)
{
    GrCtx& c = *static_cast<GrCtx*>(vctx);
    const int total_a = c.R * (c.M + 1);
    gr_dots_range<BF16>(c, (int) ((int64_t) total_a * worker / nw),
                     (int) ((int64_t) total_a * (worker + 1) / nw));
    c.bar.fetch_add(1, std::memory_order_acq_rel);
    while (c.bar.load(std::memory_order_acquire) < nw)
        cpu_pause();
    const int D = c.D;
    gr_finalize_range<BF16>(c, worker,
                            (int) ((int64_t) D * worker / nw),
                            (int) ((int64_t) D * (worker + 1) / nw),
                            1.0f / (float) c.H);
}

// -------------------------------------------------------------------------------------------
//   Entry
// -------------------------------------------------------------------------------------------

bool gr_mix_cpu_has_avx512()
{
    static const bool v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("fma") &&
                          __builtin_cpu_supports("f16c");
    return v;
}

void gr_mix_cpu
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& up,
    const at::Tensor& w,
    double rms_eps,
    const std::optional<at::Tensor>& post,
    at::Tensor& mixed
)
{
    TORCH_CHECK(gr_mix_cpu_has_avx512(), "gr_mix_cpu: AVX-512 not available");
    TORCH_CHECK(streams.device().is_cpu() && fn.device().is_cpu() && up.device().is_cpu()
                    && w.device().is_cpu() && mixed.device().is_cpu(),
                "gr_mix_cpu: tensors must be on CPU");
    TORCH_CHECK(streams.scalar_type() == at::kFloat && streams.is_contiguous() && streams.dim() == 3,
                "streams must be contiguous CPU float (R, H, D)");
    // fn/up: fp16 (default) or both bf16 (EXL3_HC_BF16) -- dispatch below
    const bool bf16 = fn.scalar_type() == at::kBFloat16;
    TORCH_CHECK((fn.scalar_type() == at::kHalf || bf16) && fn.is_contiguous() && fn.dim() == 2,
                "fn must be contiguous CPU half or bfloat16 (M, H * D)");
    TORCH_CHECK(up.scalar_type() == fn.scalar_type() && up.is_contiguous() && up.dim() == 2,
                "up must be contiguous CPU, same dtype as fn (H * D, LR)");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.numel() == streams.size(1) * streams.size(2),
                "w must be contiguous CPU float (H * D)");
    const int R = streams.size(0);
    const int H = streams.size(1);
    const int D = streams.size(2);
    const int M = fn.size(0);
    const int LR = up.size(1);
    TORCH_CHECK(H == 4, "gr_mix_cpu: H = 4 only");
    TORCH_CHECK(R >= 1 && R <= GR_MAX_R, "gr_mix_cpu: R in [1, 8]");
    TORCH_CHECK(D % 16 == 0 && LR % 16 == 0, "gr_mix_cpu: D and LR must be multiples of 16");
    TORCH_CHECK(fn.size(1) == H * D && up.size(0) == H * D && w.numel() == H * D, "gr_mix_cpu: dims");
    float* post_p = nullptr;
    if (post)
    {
        TORCH_CHECK(post->scalar_type() == at::kFloat && post->is_contiguous()
                        && post->dim() == 2 && post->size(0) == R && post->size(1) == H,
                    "post must be contiguous CPU float (R, H)");
        post_p = post->data_ptr<float>();
    }
    TORCH_CHECK(mixed.scalar_type() == at::kHalf && mixed.is_contiguous() && mixed.dim() == 2
                    && mixed.size(0) == R && mixed.size(1) == D,
                "mixed must be contiguous CPU half (R, D)");

    at::Tensor dots = at::empty({R, M + 1, H}, streams.options());
    std::vector<float> t((size_t) R * LR);
    GrCtx c;
    c.streams = streams.data_ptr<float>();
    c.fn = reinterpret_cast<const uint16_t*>(fn.const_data_ptr());
    c.up = reinterpret_cast<const uint16_t*>(up.const_data_ptr());
    c.w = w.data_ptr<float>();
    c.dots = dots.data_ptr<float>();
    c.post = post_p;
    c.mixed = mixed.data_ptr<at::Half>();
    c.t = t.data();
    c.R = R; c.H = H; c.D = D; c.LR = LR; c.M = M;
    c.rms_eps = (float) rms_eps;

    const int nw = std::min(hc_cpu_threads(), std::max(1, std::min(R * (M + 1), D)));
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.ensure(nw);
    c.bar.store(0, std::memory_order_relaxed);
    g_pool.run(bf16 ? &gr_pool_fn<true> : &gr_pool_fn<false>, &c, nw);
}

#else  // !x86
#include "hc_mix_cpu.h"
#include <torch/extension.h>

void gr_mix_cpu
(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    const std::optional<at::Tensor>&,
    at::Tensor&
)
{
    TORCH_CHECK(false, "gr_mix_cpu is not available on this CPU architecture (x86 AVX-512 only)");
}

bool gr_mix_cpu_has_avx512()
{
    return false;
}
#endif
