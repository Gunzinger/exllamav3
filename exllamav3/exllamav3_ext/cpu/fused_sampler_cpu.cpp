#include "fused_sampler_cpu.h"
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Fused terminal sampler, CPU (see fused_sampler_cpu.h). Mirrors the modes and filter semantics
// of generator/sampling_fused.cu; the threshold select gathers the boundary histogram bucket and
// resolves it exactly instead of re-histogramming sub-buckets.

namespace {

constexpr float FS_NEG_INF = -std::numeric_limits<float>::infinity();
constexpr int FS_NB = 4096;                        // coarse buckets over FS_HIST_RANGE nats
constexpr float FS_HIST_RANGE = 32.0f;
constexpr double FS_MASS_SCALE = 1099511627776.0;  // 2^40 fixed-point exp mass

inline float fs_gumbel(float x)
{
    x = std::fmin(x, 0.99999994f);
    return -std::log(std::fmax(-std::log(std::fmax(x, 1e-20f)), 1e-20f));
}

// Per-index uniform in [0, 1), splitmix64 over (rand_u32, index) -- a documented divergence from
// the CUDA kernel's curand Philox stream (sampling is random; parity is distributional, and the
// eager CPU fallback in SS_Sample is seeded from the same rand_u32 for tests)
inline uint64_t fs_splitmix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline float fs_uniform(uint32_t random, int i)
{
    uint64_t z = fs_splitmix((((uint64_t) random << 32) ^ (uint64_t) (uint32_t) i * 0x9E3779B97F4A7C15ull));
    return (float) ((double) (z >> 11) * (1.0 / 9007199254740992.0));
}

// One row's mask application and read
template <typename T>
struct FSRow
{
    const T* base;
    int stride;
    const at::Half* row_mask;
    const uint32_t* row_bits;

    inline float at(int i) const
    {
        float v = (float) base[i];
        if (row_mask) v += (float) row_mask[i];
        else if (row_bits && !((row_bits[i >> 5] >> (i & 31)) & 1)) v = FS_NEG_INF;
        return v;
    }
};

struct FSElem
{
    float key;
    int idx;
};

inline bool fs_elem_gt(const FSElem& a, const FSElem& b)
{
    return a.key != b.key ? a.key > b.key : a.idx < b.idx;
}

// Bucket of one logit under the shared binning expression (all passes use this exact arithmetic,
// so the partition is identical everywhere)
inline int fs_bin(float x, float m, float bucket_scale)
{
    float db = (m - x) * bucket_scale;
    return std::min((int) db, FS_NB - 1);
}

// Fixed-point mass contribution of one element (deterministic accumulation order, like CUDA)
inline unsigned long long fs_mass(float x, float m, float inv_tf)
{
    return (unsigned long long) (std::exp((double) ((x - m) * inv_tf)) * FS_MASS_SCALE + 0.5);
}

}

void fused_sampler_cpu
(
    const at::Tensor& logits,
    const std::optional<at::Tensor>& mask,
    const std::optional<at::Tensor>& bits,
    at::Tensor& sample,
    const at::Tensor& workspace,
    int64_t dim,
    double inv_temp,
    double minp_log,
    int64_t rand_u32,
    int64_t mode,
    int64_t filters,
    int64_t top_k,
    double top_p,
    double inv_temp_filter,
    const std::optional<at::Tensor>& histogram
)
{
    TORCH_CHECK(logits.device().is_cpu(), "fused_sampler_cpu: tensors must be on CPU");
    TORCH_CHECK(logits.scalar_type() == at::kHalf || logits.scalar_type() == at::kFloat,
                "fused_sampler_cpu: logits must be half or float");
    TORCH_CHECK(logits.is_contiguous() && logits.dim() == 2, "fused_sampler_cpu: logits (bsz, dim)");
    const int bsz = (int) logits.size(0);
    const int dim2 = (int) logits.size(1);
    const int size = std::min((int) dim, dim2);
    TORCH_CHECK(!mask || (mask->scalar_type() == at::kHalf && mask->is_contiguous()),
                "fused_sampler_cpu: mask must be contiguous half");
    TORCH_CHECK(!bits || (bits->scalar_type() == at::kInt && bits->is_contiguous()),
                "fused_sampler_cpu: bits must be contiguous int32");
    TORCH_CHECK(sample.scalar_type() == at::kLong && sample.is_contiguous()
                    && sample.numel() == bsz,
                "fused_sampler_cpu: sample (bsz, 1) long");
    TORCH_CHECK(mode >= 0 && mode <= 3, "fused_sampler_cpu: mode");
    (void) workspace; (void) histogram;   // call parity with the CUDA entry

    const bool mask_per_row = mask.has_value() && mask->size(0) == bsz && bsz > 1;
    const bool bits_per_row = bits.has_value() && bits->size(0) == bsz && bsz > 1;
    const int mask_stride = mask.has_value() ? (int) mask->size(-1) : 0;
    const int bits_words = bits.has_value() ? (int) bits->size(-1) : 0;

    const bool f_minp = mode == 2 || (mode == 3 && (filters & 4));
    const bool f_topk = mode == 3 && (filters & 1) && top_k > 0;
    const bool f_topp = mode == 3 && (filters & 2) && top_p < 1.0;
    const float inv_t = (float) inv_temp;
    const float inv_tf = (float) inv_temp_filter;
    const float bucket_scale = (float) FS_NB / FS_HIST_RANGE * inv_tf;
    const uint32_t rnd = (uint32_t) rand_u32;

    std::vector<FSElem> gathered;
    std::vector<unsigned long long> mass;
    std::vector<unsigned int> cnt;

    for (int row = 0; row < bsz; ++row)
    {
        FSRow<float> rf{nullptr, 0, nullptr, nullptr};
        FSRow<at::Half> rh{nullptr, 0, nullptr, nullptr};
        const bool half = logits.scalar_type() == at::kHalf;
        if (half)
        {
            rh.base = logits.data_ptr<at::Half>() + (size_t) row * dim2;
            rh.row_mask = mask.has_value()
                ? mask->data_ptr<at::Half>() + (mask_per_row ? (size_t) row * mask_stride : 0)
                : nullptr;
            rh.row_bits = bits.has_value()
                ? (const uint32_t*) bits->data_ptr<int>() + (bits_per_row ? (size_t) row * bits_words : 0)
                : nullptr;
        }
        else
        {
            rf.base = logits.data_ptr<float>() + (size_t) row * dim2;
            rf.row_mask = rh.row_mask;
            rf.row_bits = rh.row_bits;
        }
        auto read = [&](int i) -> float { return half ? rh.at(i) : rf.at(i); };

        // Pass 1: masked max
        float m = FS_NEG_INF;
        for (int i = 0; i < size; ++i) m = std::fmax(m, read(i));

        int64_t out = 0;

        if (mode == 0)
        {
            // Greedy: temperature/min-P never move the winner; lowest index wins ties
            float best = FS_NEG_INF;
            int best_i = -1;
            for (int i = 0; i < size; ++i)
            {
                float x = read(i);
                if (x > best) { best = x; best_i = i; }
            }
            out = best_i;
        }
        else
        {
            const float minp_threshold = f_minp ? m + (float) minp_log : FS_NEG_INF;

            // Kept-set bound: bucket bb and key rule (bound_gt ? key > bound_key : key >=
            // bound_key); bb = -1 keeps the whole (min-P-filtered) set
            int bb = -1;
            float bound_key = 0.0f;
            bool bound_gt = false;

            if (f_topk || f_topp)
            {
                // Pass 2: deterministic fixed-point histogram (counts for K, exp mass for P)
                mass.assign(FS_NB, 0ull);
                cnt.assign(FS_NB, 0u);
                for (int i = 0; i < size; ++i)
                {
                    float x = read(i);
                    if (x == FS_NEG_INF || x < minp_threshold) continue;
                    int b = fs_bin(x, m, bucket_scale);
                    cnt[b] += 1;
                    mass[b] += fs_mass(x, m, inv_tf);
                }

                // Top-K: count walk locates the boundary bucket kb; `rem` elements remain in it
                int kb = FS_NB;
                int rem = 0;
                if (f_topk)
                {
                    rem = (int) top_k;
                    for (int b = 0; b < FS_NB; ++b)
                    {
                        if (cnt[b] >= rem) { kb = b; break; }
                        rem -= cnt[b];
                    }
                }

                // Gather + exact resolve of the K boundary bucket (few elements at decode scale)
                if (kb < FS_NB)
                {
                    gathered.clear();
                    for (int i = 0; i < size; ++i)
                    {
                        float x = read(i);
                        if (x == FS_NEG_INF || x < minp_threshold) continue;
                        if (fs_bin(x, m, bucket_scale) == kb) gathered.push_back({x, i});
                    }
                    std::sort(gathered.begin(), gathered.end(), fs_elem_gt);
                }

                // Kept mass for top-P: full buckets below kb + kept elements inside kb
                unsigned long long mass_kept = 0;
                float k_cutoff = FS_NEG_INF;
                for (int b = 0; b < kb && b < FS_NB; ++b) mass_kept += mass[b];
                if (kb < FS_NB)
                {
                    if ((size_t) rem <= gathered.size())
                    {
                        // The rem-th largest of the bucket is the cutoff; exact ties all kept
                        k_cutoff = gathered[(size_t) std::max(rem - 1, 0)].key;
                        for (const FSElem& e : gathered)
                            if (e.key >= k_cutoff) mass_kept += fs_mass(e.key, m, inv_tf);
                    }
                }

                if (!f_topp)
                {
                    if (kb < FS_NB && k_cutoff > FS_NEG_INF)
                    {
                        bb = kb;
                        bound_key = k_cutoff;
                        bound_gt = false;
                    }
                }
                else
                {
                    // top-P target over the kept set; find the crossing in key-desc order. Kept
                    // = largest prefix with mass <= target; the crossing token and its exact
                    // ties are dropped (SS_Fused's rule)
                    unsigned long long target = (unsigned long long) ((double) mass_kept * top_p);
                    target = std::max(target, 1ull);
                    unsigned long long cum = 0;
                    int bp = FS_NB;
                    for (int b = 0; b < FS_NB; ++b)
                    {
                        unsigned long long mb = mass[b];
                        if (b == kb)
                        {
                            mb = 0;
                            for (const FSElem& e : gathered)
                                if (e.key >= k_cutoff) mb += fs_mass(e.key, m, inv_tf);
                        }
                        if (cum + mb > target) { bp = b; break; }
                        cum += mb;
                    }
                    if (bp < FS_NB)
                    {
                        // Exact crossing inside bp over the kept subset
                        if (bp != kb)
                        {
                            gathered.clear();
                            for (int i = 0; i < size; ++i)
                            {
                                float x = read(i);
                                if (x == FS_NEG_INF || x < minp_threshold) continue;
                                if (fs_bin(x, m, bucket_scale) == bp) gathered.push_back({x, i});
                            }
                            std::sort(gathered.begin(), gathered.end(), fs_elem_gt);
                        }
                        float ck = FS_NEG_INF;
                        unsigned long long cum2 = cum;
                        for (const FSElem& e : gathered)
                        {
                            if (bp == kb && e.key < k_cutoff) continue;   // not kept for K
                            cum2 += fs_mass(e.key, m, inv_tf);
                            if (cum2 > target) { ck = e.key; break; }
                        }
                        if (ck > FS_NEG_INF)
                        {
                            bb = bp;
                            bound_key = ck;
                            bound_gt = true;
                        }
                    }
                }
            }

            // Pass 3: gumbel-max over the kept set, argmax lowest-index tie-break
            float best = FS_NEG_INF;
            int best_i = -1;
            for (int i = 0; i < size; ++i)
            {
                float x = read(i);
                if (x == FS_NEG_INF || x < minp_threshold) continue;
                if (bb >= 0)
                {
                    int b = fs_bin(x, m, bucket_scale);
                    if (b > bb) continue;
                    if (b == bb && !(bound_gt ? x > bound_key : x >= bound_key)) continue;
                }
                float score = x * inv_t + fs_gumbel(fs_uniform(rnd, i));
                if (score > best) { best = score; best_i = i; }
            }
            if (best_i < 0)
            {
                // Degenerate row (everything filtered): masked argmax
                for (int i = 0; i < size; ++i)
                {
                    float x = read(i);
                    if (x > best) { best = x; best_i = i; }
                }
            }
            out = best_i;
        }

        sample[row] = out;
    }
}
