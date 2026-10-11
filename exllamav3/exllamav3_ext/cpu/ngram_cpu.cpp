// CPU-only build: host ports of the n-gram embedding fast path (see ngram.cu for the CUDA
// original and the algorithm notes). ngram_hash_cpu and the Linux ngram_gather_cpu are pure
// host code and are transcribed verbatim; ngram_dequant's ring decode + Walsh-Hadamard rotate
// runs scalar over rows (U is the unique-row count of one gather: tens at decode, thousands at
// prefill), with the exact fp16 rounding sequence of the GPU kernel preserved bitwise.
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <c10/util/Half.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <algorithm>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include "util.h"

namespace py = pybind11;

#define MAX_RING_DIM 256
#define MUL1 0x83DCD12Du

int64_t ngram_hash_cpu
(
    const at::Tensor& ids,           // (bsz, ctx + seq) int64 CPU
    int64_t seq_len,
    const at::Tensor& multipliers,   // (ngram_size) int64 CPU
    const at::Tensor& offsets,       // (num_heads) int64 CPU
    const at::Tensor& sizes,         // (num_heads) int64 CPU
    int64_t heads_per_ngram,
    int64_t eos_token,
    at::Tensor uids,                 // (cap) int64 CPU out
    at::Tensor inverse,              // (bsz * seq * num_heads) int64 CPU out
    at::Tensor heads                 // (cap) int32 CPU out
)
{
    TORCH_CHECK(ids.device().is_cpu() && ids.dtype() == at::kLong && ids.is_contiguous(),
                "ngram_hash_cpu: ids must be contiguous int64 CPU");
    int64_t bsz = ids.size(0);
    int64_t T = ids.size(1);
    int64_t ctx = T - seq_len;
    int64_t ngram_size = multipliers.numel();
    int64_t H = offsets.numel();
    TORCH_CHECK(ctx >= 0 && H == (ngram_size - 1) * heads_per_ngram, "ngram_hash_cpu: dims");
    int64_t n = bsz * seq_len * H;
    TORCH_CHECK(uids.numel() >= n && inverse.numel() >= n && heads.numel() >= n,
                "ngram_hash_cpu: output buffers too small");

    const int64_t* idp = (const int64_t*) ids.data_ptr();
    const int64_t* mult = (const int64_t*) multipliers.data_ptr();
    const int64_t* offs = (const int64_t*) offsets.data_ptr();
    const int64_t* szs = (const int64_t*) sizes.data_ptr();
    int64_t* uids_p = (int64_t*) uids.data_ptr();
    int64_t* inv_p = (int64_t*) inverse.data_ptr();
    int32_t* heads_p = (int32_t*) heads.data_ptr();

    py::gil_scoped_release release;

    struct HK { int64_t hash; int64_t idx; };
    std::vector<HK> hk((size_t) n);

    for (int64_t b = 0; b < bsz; ++b)
    {
        const int64_t* row = idp + b * T;
        int64_t prev_eos = -1;
        for (int64_t p = 0; p < T; ++p)
        {
            if (p > 0 && row[p - 1] == eos_token) prev_eos = p - 1;
            if (p >= ctx)
            {
                int64_t seg_start = prev_eos + 1;
                uint64_t mixed = (uint64_t) row[p] * (uint64_t) mult[0];
                for (int64_t s = 1; s < ngram_size; ++s)
                {
                    int64_t src = (p - s >= seg_start && p - s >= 0) ? row[p - s] : eos_token;
                    mixed ^= (uint64_t) src * (uint64_t) mult[s];
                    int64_t lo = (s - 1) * heads_per_ngram;
                    for (int64_t h = lo; h < lo + heads_per_ngram; ++h)
                    {
                        int64_t m = (int64_t) mixed % szs[h];
                        if (m < 0) m += szs[h];
                        int64_t idx = (b * seq_len + (p - ctx)) * H + h;
                        hk[(size_t) idx] = { m + offs[h], idx };
                    }
                }
            }
        }
    }

    std::sort(hk.begin(), hk.end(), [](const HK& a, const HK& b) { return a.hash < b.hash; });

    int64_t u = -1;
    int64_t last = -1;
    for (int64_t i = 0; i < n; ++i)
    {
        if (u < 0 || hk[(size_t) i].hash != last)
        {
            last = hk[(size_t) i].hash;
            uids_p[++u] = last;
            int64_t h = (int64_t) (std::upper_bound(offs, offs + H, last) - offs) - 1;
            heads_p[u] = (int32_t) h;
        }
        inv_p[hk[(size_t) i].idx] = u;
    }
    return u + 1;
}

#ifndef _WIN32

namespace
{

struct GatherCtx
{
    int fd;
    int64_t base_offset;
    int64_t row_bytes;
    int64_t uid_base;
    const int64_t* up;
    uint8_t* op;
    std::atomic<bool> failed { false };
};

struct GatherPool
{
    std::mutex mx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    GatherCtx* ctx = nullptr;
    std::vector<std::pair<int64_t, int64_t>> tasks;
    size_t next = 0;
    size_t pending = 0;
    bool started = false;
    std::mutex call_mx;

    static void read_span(GatherCtx* c, int64_t i0, int64_t i1)
    {
        int64_t i = i0;
        while (i < i1)
        {
            int64_t j = i + 1;
            while (j < i1 && c->up[j] == c->up[j - 1] + 1) ++j;
            int64_t off = c->base_offset + (c->up[i] - c->uid_base) * c->row_bytes;
            int64_t nbytes = (j - i) * c->row_bytes;
            int64_t pos = 0;
            while (pos < nbytes)
            {
                ssize_t got = pread(c->fd, c->op + i * c->row_bytes + pos,
                                    (size_t) (nbytes - pos), (off_t) (off + pos));
                if (got <= 0) { c->failed.store(true); return; }
                pos += got;
            }
            i = j;
        }
    }

    void worker()
    {
        std::unique_lock<std::mutex> lk(mx);
        while (true)
        {
            cv_work.wait(lk, [&] { return next < tasks.size(); });
            auto span = tasks[next++];
            GatherCtx* c = ctx;
            lk.unlock();
            read_span(c, span.first, span.second);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
    }

    void run(GatherCtx* c, std::vector<std::pair<int64_t, int64_t>>& new_tasks)
    {
        std::lock_guard<std::mutex> call_lock(call_mx);
        std::unique_lock<std::mutex> lk(mx);
        if (!started)
        {
            started = true;
            int n = (int) std::min(32u, std::max(4u, std::thread::hardware_concurrency()));
            for (int i = 0; i < n; ++i)
                std::thread(&GatherPool::worker, this).detach();
        }
        ctx = c;
        tasks.swap(new_tasks);
        next = 0;
        pending = tasks.size();
        cv_work.notify_all();
        while (next < tasks.size())
        {
            auto span = tasks[next++];
            lk.unlock();
            read_span(c, span.first, span.second);
            lk.lock();
            if (--pending == 0)
                cv_done.notify_all();
        }
        cv_done.wait(lk, [&] { return pending == 0; });
        tasks.clear();
        ctx = nullptr;
    }
};

GatherPool* gather_pool()
{
    static GatherPool* pool = new GatherPool();
    return pool;
}

}  // namespace

void ngram_gather_cpu
(
    int64_t fd,
    int64_t base_offset,
    int64_t row_bytes,
    const at::Tensor& uids,
    int64_t uid_base,
    at::Tensor out
)
{
    TORCH_CHECK(uids.device().is_cpu() && uids.dtype() == at::kLong && uids.is_contiguous(),
                "ngram_gather_cpu: uids must be contiguous int64 CPU");
    TORCH_CHECK(out.device().is_cpu() && out.is_contiguous(), "ngram_gather_cpu: bad out");
    int64_t U = uids.numel();
    TORCH_CHECK(out.size(0) >= U && out.size(1) * out.element_size() == row_bytes,
                "ngram_gather_cpu: out shape");
    if (!U) return;
    const int64_t* up = (const int64_t*) uids.data_ptr();
    uint8_t* op = (uint8_t*) out.data_ptr();

    py::gil_scoped_release release;

    GatherCtx ctx { (int) fd, base_offset, row_bytes, uid_base, up, op };

    std::vector<std::pair<int64_t, int64_t>> runs;
    runs.reserve(64);
    for (int64_t i = 0; i < U; )
    {
        int64_t j = i + 1;
        while (j < U && up[j] == up[j - 1] + 1) ++j;
        runs.emplace_back(i, j - i);
        i = j;
    }
    size_t n_runs = runs.size();

    if (n_runs == 1)
    {
        GatherPool::read_span(&ctx, 0, U);
        TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: short read");
        return;
    }

    std::vector<std::pair<int64_t, int64_t>> tasks;
    size_t group = (n_runs + 63) / 64;
    tasks.reserve((n_runs + group - 1) / group);
    for (size_t r = 0; r < n_runs; r += group)
    {
        size_t r1 = std::min(n_runs, r + group);
        tasks.emplace_back(runs[r].first, runs[r1 - 1].first + runs[r1 - 1].second);
    }
    gather_pool()->run(&ctx, tasks);
    TORCH_CHECK(!ctx.failed.load(), "ngram_gather_cpu: short read");
}

#endif  // ^ Linux (pread)

static float half_bits_f(uint16_t bits)
{
    c10::Half h;
    memcpy(&h, &bits, sizeof(h));
    return (float) h;
}

void ngram_dequant
(
    const at::Tensor& packed,                   // (U, 1 + dim * K / 16) int16 CPU
    int64_t K,
    const c10::optional<at::Tensor>& heads,     // (U) int32 CPU
    const c10::optional<at::Tensor>& aux,       // (num_heads, dim) half CPU
    at::Tensor out,                             // (U, dim) half CPU
    bool rotate
)
{
    TORCH_CHECK(packed.device().is_cpu() && out.device().is_cpu(), "ngram_dequant: CPU tensors");
    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "ngram_dequant: contiguous inputs required");
    int64_t U = packed.size(0);
    int words = (int) packed.size(1);
    int dim = (int) out.size(1);
    TORCH_CHECK(1 <= K && K <= 8 && words == 1 + dim * (int) K / 16, "ngram_dequant: bad K/words");
    TORCH_CHECK(dim <= MAX_RING_DIM && dim % 16 == 0 && out.size(0) >= U, "ngram_dequant: shapes");
    TORCH_CHECK(!rotate || (dim & (dim - 1)) == 0, "ngram_dequant: rotation needs a power of two");
    if (heads)
    {
        TORCH_CHECK_DTYPE((*heads), kInt);
        TORCH_CHECK(aux && heads->is_contiguous() && heads->numel() >= U, "ngram_dequant: heads");
    }
    if (aux)
    {
        TORCH_CHECK_DTYPE((*aux), kHalf);
        TORCH_CHECK(aux->is_contiguous() && aux->dim() == 2 && aux->size(1) == dim, "ngram_dequant: aux shape");
    }
    if (!U) return;

    const uint16_t* pk = (const uint16_t*) packed.data_ptr();
    const int32_t* hp = heads ? (const int32_t*) heads->data_ptr() : nullptr;
    const c10::Half* ap = aux ? (const c10::Half*) aux->data_ptr() : nullptr;
    c10::Half* op = (c10::Half*) out.data_ptr();
    int num_heads = aux ? (int) aux->size(0) : 1;

    const float k_inv = half_bits_f(0x1eee);
    const float k_bias = half_bits_f(0xc931);
    std::vector<float> sf(dim);

    py::gil_scoped_release release;

    for (int64_t r = 0; r < U; ++r)
    {
        const uint16_t* sw = pk + (size_t) r * words;
        float scale = half_bits_f(sw[0]);

        for (int i = 0; i < dim; ++i)
        {
            uint32_t state = 0;
            for (int m = 0; m < 16; ++m)
            {
                int pos = i - m / (int) K;
                if (pos < 0) pos += dim;
                int sb = pos * (int) K + m % (int) K;
                uint32_t bit = (sw[1 + (sb >> 4)] >> (sb & 15)) & 1;
                state |= bit << m;
            }
            uint32_t prod = state * MUL1;
            float h = 1024.0f + (float) ((prod & 0xff) + ((prod >> 8) & 0xff) +
                                         ((prod >> 16) & 0xff) + ((prod >> 24) & 0xff));
            float v = (float) (c10::Half) (h * k_inv + k_bias) * scale;

            float a = 0.0f;
            if (ap)
            {
                int head = hp ? hp[r] : (int) (r % num_heads);
                a = (float) ap[(size_t) head * dim + i];
            }

            if (rotate) { sf[i] = v; (void) a; }
            else sf[i] = v + a;
        }

        if (rotate)
        {
            for (int s = 1; s < dim; s <<= 1)
                for (int i = 0; i < dim; ++i)
                {
                    float lo = sf[i & ~s];
                    float hi = sf[i | s];
                    sf[i] = (i & s) ? lo - hi : lo + hi;
                }
            float inv_sqrt_d = 1.0f / std::sqrt((float) dim);
            for (int i = 0; i < dim; ++i)
            {
                float v = sf[i] * inv_sqrt_d;
                if (ap) v *= (float) ap[(size_t) (hp ? hp[r] : (int) (r % num_heads)) * dim + i];
                op[(size_t) r * dim + i] = (c10::Half) v;
            }
        }
        else
            for (int i = 0; i < dim; ++i)
                op[(size_t) r * dim + i] = (c10::Half) sf[i];
    }
}
