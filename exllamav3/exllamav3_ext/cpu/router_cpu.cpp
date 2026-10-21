// Fused CPU router (router_cpu.h). x86-64 build; other platforms get stubs that report no
// AVX-512 so the Python side keeps the torch path.

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "router_cpu.h"
#include "cpu/moe_mul1.h"
#include <torch/extension.h>
#include <c10/util/Half.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <immintrin.h>

#define ROUTER_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,fma,f16c")))

namespace {

constexpr int ROUTER_MAX_TOPK = 32;

// torch.topk(largest = True, sorted = True) rank order on CPU: higher value first, and among
// equal values the lower index first (verified empirically against torch.topk on crafted
// tied inputs; see tests in the C5 smoke script). Selection and sort below use this comparator.
inline bool router_rank_above(float v1, int64_t i1, float v2, int64_t i2)
{
    if (v1 != v2) return v1 > v2;
    return i1 < i2;
}

ROUTER_TARGET_AVX512 inline float router_hsum16(__m512 v)
{
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    s4 = _mm_add_ps(s4, _mm_movehdup_ps(s4));
    s4 = _mm_add_ss(s4, _mm_movehl_ps(s4, s4));
    return _mm_cvtss_f32(s4);
}

// One y row against every gate row: 4 expert rows per pass, the y chunk converted to fp32 once
// (vcvtph2ps) and FMA'd into 4 fp32 accumulators. The fp32 result is rounded to fp16 and back,
// which the reference path does by storing the matmul output as half.
ROUTER_TARGET_AVX512 inline void router_logits_row(
    const c10::Half* __restrict__ y,
    const c10::Half* __restrict__ gate,   // (E, K) row-major
    int64_t E, int64_t K,
    float* __restrict__ l,
    int64_t e0, int64_t e1)
{
    for (int64_t e = e0; e < e1; e += 4)
    {
        const int64_t rows = E - e < 4 ? E - e : 4;
        const c10::Half* g[4] =
        {
            gate + e * K,
            gate + (e + 1 < E ? e + 1 : e) * K,
            gate + (e + 2 < E ? e + 2 : e) * K,
            gate + (e + 3 < E ? e + 3 : e) * K,
        };
        __m512 acc[4] =
        {
            _mm512_setzero_ps(), _mm512_setzero_ps(),
            _mm512_setzero_ps(), _mm512_setzero_ps(),
        };
        for (int64_t k = 0; k < K; k += 16)
        {
            __m512 yv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (const void*) (y + k)));
            for (int r = 0; r < 4; r++)
            {
                __m512 gv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (const void*) (g[r] + k)));
                acc[r] = _mm512_fmadd_ps(gv, yv, acc[r]);
            }
        }
        for (int r = 0; r < rows; r++)
        {
            float d = router_hsum16(acc[r]);
            // round to half (nearest-even, as .half() does) and back, then top-k/softmax in fp32
            l[e + r] = _cvtsh_ss(_cvtss_sh(d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }
}

// M4: logits phase on the moe worker pool (pinned, hot right before the expert phase). Each
// worker takes a contiguous expert-row chunk of every y row; the per-row math is the serial
// kernel's unchanged loop, so logits are bit-identical, and the top-k/tie-guard/softmax in
// router_std_cpu stays the serial code reading the completed l[]
struct RouterLogitsCtx
{
    const c10::Half* y;
    const c10::Half* gate;
    int64_t M, E, K;
    float* l;
};

void router_logits_pool_fn(void* vctx, int worker, int nw)
{
    RouterLogitsCtx& c = *static_cast<RouterLogitsCtx*>(vctx);
    const int64_t e0 = c.E * worker / nw;
    const int64_t e1 = c.E * (worker + 1) / nw;
    if (e0 >= e1) return;
    for (int64_t mo = 0; mo < c.M; ++mo)
        router_logits_row(c.y + mo * c.K, c.gate, c.E, c.K, c.l + mo * c.E, e0, e1);
}

int router_cpu_threads()
{
    // 1 = serial. The gate matrix is a once-per-step DRAM stream (~2.5 MB at 512x2560) a single
    // core streams at ~19 GB/s; an expert-row split on the moe pool measured 148.6/137.3/54.5/
    // 84.3 us/layer at 1/4/6/8 workers on the bench host, so 6 is the default. Never exceeds
    // the pool's current size
    static const int v = [] {
        int t = 6;
        const char* e = std::getenv("EXL3_CPU_ROUTER_THREADS");
        if (e && *e) t = std::atoi(e);
        return std::max(1, t);
    }();
    return v;
}

} // namespace

bool router_std_cpu
(
    const at::Tensor& gate_t,
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    TORCH_CHECK(router_std_cpu_has_avx512(), "router_std_cpu needs AVX-512F/BW/VL/DQ+FMA+F16C");
    TORCH_CHECK(gate_t.dim() == 2 && gate_t.scalar_type() == at::kHalf && gate_t.is_contiguous(),
                "gate_t must be (E, K) contiguous fp16");
    TORCH_CHECK(y.dim() == 2 && y.scalar_type() == at::kHalf && y.is_contiguous(),
                "y must be (M, K) contiguous fp16");
    const int64_t E = gate_t.size(0);
    const int64_t K = gate_t.size(1);
    const int64_t M = y.size(0);
    const int64_t TK = selected_experts.size(1);
    TORCH_CHECK(y.size(1) == K, "y/gate width mismatch");
    TORCH_CHECK(E > 0 && TK > 0 && TK <= ROUTER_MAX_TOPK && TK <= E, "bad top-k");
    TORCH_CHECK(selected_experts.dim() == 2 && selected_experts.size(0) == M &&
                selected_experts.scalar_type() == at::kLong && selected_experts.is_contiguous(),
                "selected_experts must be (M, k) contiguous int64");
    TORCH_CHECK(routing_weights.dim() == 2 && routing_weights.size(0) == M &&
                routing_weights.size(1) == TK && routing_weights.scalar_type() == at::kHalf &&
                routing_weights.is_contiguous(), "routing_weights must be (M, k) contiguous fp16");

    const c10::Half* gate_p = reinterpret_cast<const c10::Half*>(gate_t.data_ptr());
    const c10::Half* y_p = reinterpret_cast<const c10::Half*>(y.data_ptr());
    int64_t* sel_p = selected_experts.data_ptr<int64_t>();
    c10::Half* w_p = reinterpret_cast<c10::Half*>(routing_weights.data_ptr());

    std::vector<float> l((size_t) M * E);
    const int rt = router_cpu_threads();
    if (rt > 1)
    {
        RouterLogitsCtx rl{y_p, gate_p, M, E, K, l.data()};
        exl3_moe_cpu_pool_run(&router_logits_pool_fn, &rl, rt);
    }
    else
    {
        for (int64_t m = 0; m < M; m++)
            router_logits_row(y_p + m * K, gate_p, E, K, l.data() + m * E, 0, E);
    }
    for (int64_t m = 0; m < M; m++)
    {
        const float* lm = l.data() + m * E;

        // Top-k as a best-first sorted insert list (k <= 32, E in L1): a candidate enters only
        // if it beats the current worst; equal values resolve to the lower index.
        float tv[ROUTER_MAX_TOPK];
        int64_t ti[ROUTER_MAX_TOPK];
        int n = 0;
        for (int64_t e = 0; e < E; e++)
        {
            const float v = lm[e];
            if (n == TK && !router_rank_above(v, e, tv[TK - 1], ti[TK - 1])) continue;
            int j = n < TK ? n++ : (int) TK - 1;
            tv[j] = v; ti[j] = e;
            while (j > 0 && router_rank_above(tv[j], ti[j], tv[j - 1], ti[j - 1]))
            {
                std::swap(tv[j], tv[j - 1]);
                std::swap(ti[j], ti[j - 1]);
                j--;
            }
        }

        // Tie guard: any equal-value pair inside the selection reorders equal-weight expert
        // contributions in the moe kernel (fp sum order), and a rank-k/k+1 boundary tie can pick
        // a different subset under torch's radix topk. Either way the caller redoes the batch on
        // the torch path for exact parity; untied rows (the common case) take the fast path.
        bool tied = false;
        for (int j = 0; j + 1 < n; j++) if (tv[j] == tv[j + 1]) { tied = true; break; }
        if (!tied && n == TK && TK < E)
        {
            float bnd = -std::numeric_limits<float>::infinity();
            for (int64_t e = 0; e < E; e++)
            {
                bool in = false;
                for (int j = 0; j < n; j++) if (ti[j] == e) { in = true; break; }
                if (!in && lm[e] > bnd) bnd = lm[e];
            }
            if (bnd == tv[TK - 1]) tied = true;
        }
        if (tied) return false;

        // Softmax over the selection (norm_topk_prob): max is tv[0] (sorted descending)
        float es[ROUTER_MAX_TOPK];
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            es[j] = std::exp(tv[j] - tv[0]);
            sum += es[j];
        }
        const float denom = sum + 1e-20f;
        for (int j = 0; j < n; j++)
        {
            sel_p[m * TK + j] = ti[j];
            w_p[m * TK + j] = c10::Half(es[j] / denom);
        }
    }
    return true;
}

bool router_std_cpu_has_avx512()
{
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

#else  // !x86

#include "router_cpu.h"
#include <torch/extension.h>

bool router_std_cpu(const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&)
{
    TORCH_CHECK(false, "router_std_cpu is x86-only");
    return false;
}

bool router_std_cpu_has_avx512() { return false; }

#endif
