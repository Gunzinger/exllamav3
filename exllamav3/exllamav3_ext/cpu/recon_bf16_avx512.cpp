// AVX-512 dequantize-to-BF16 for one mul1 EXL3 dense matrix (Plan 07 Option 2, dequant only).
// See recon_bf16_avx512.h for the contract. This file shares the scalar reference's format
// tables and math (recon_bf16.cpp) but executes it 16 lanes wide:
//
//   - state decode: integer-exact. Per tile, the u32 bit ring is mirrored once into a small
//     extended buffer (ring | ring | head, constants per rate); each tile row's 16 slot states
//     then come out of ONE unaligned 16-word window load + two in-register permutes and
//     per-lane variable shifts (the (w[F] << 32 | w[L]) merged-word trick of the scalar
//     decode, folded into srlv/sllv). The slot permutation is the closed form
//     slot(n, k) = 32*(n&7) + 8*((k>>1)&3) + (k&1) + 2*((k>>3)&1) + 4*(n>>3)
//     == g_tc_inv[k*16 + n] (verified against recon_bf16.cpp's constexpr inverse table).
//   - mul1 codebook: byte-sum via the exact (b0+b1)+(b2+b3) SWAR reduce; the affine float
//     conversion uses the scalar's exact op order ((float)sum - 510) * k_inv.
//   - Hadamards: the same radix-2 stage sequence as the scalar (width 1,2,4,...,64, pairing
//     element j with j^width), so every fp32 add/sub happens between the same operand pair in
//     the same stage order -> bit-identical accumulation. The n-side pass runs on the
// in-register transpose of the block (transposing is exact), then transposes back.
//   - scales: 1/sqrt(128) after each butterfly pass and (w*svh)*suh after both, exactly like
//     the scalar; fp32->BF16 uses the integer RNE formula c10::BFloat16 itself uses.
//
// Result: bit-identical output to exl3_cpu_dequant_bf16 (asserted by tests/test_recon_bf16_avx512.py).
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include "recon_bf16_avx512.h"
#include "moe_mul1_internal.h"
#include <torch/extension.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__GNUC__)
#define RB16_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
#else
#define RB16_TARGET
#endif

namespace {

constexpr uint32_t R_MUL1_MULT = 0x83DCD12Du;
constexpr float R_HAD_SCALE = 0.088388347648f;   // 1/sqrt(128), once per Hadamard pass

inline float rb_half_to_float(at::Half h) { return static_cast<float>(h); }

// mul1 codebook scale (c10::Half 0x1EEE -> float), computed once at load: a function-local
// static would add an init-guard check to every decode row
static const float g_rb_k_inv = static_cast<float>(c10::Half(uint16_t(0x1eee), c10::Half::from_bits()));

// ------------------------------------------------------------------ constexpr tables

struct RbSel { int32_t v[16]; };

constexpr RbSel rb_xor_sel(int w)
{
    RbSel s{};
    for (int j = 0; j < 16; ++j) s.v[j] = j ^ w;
    return s;
}

// Hadamard intra-vector stage selectors (stage w pairs lane j with lane j ^ w, w = 1,2,4,8)
constexpr RbSel g_sel1 = rb_xor_sel(1);
constexpr RbSel g_sel2 = rb_xor_sel(2);
constexpr RbSel g_sel4 = rb_xor_sel(4);
constexpr RbSel g_sel8 = rb_xor_sel(8);

// 16x16 transpose exchange selectors. Stage k exchanges zmm-index bit k with lane-index bit k
// (elements where the two bits differ swap; composing k = 0..3 is the full transpose). For the
// zmm with bit k clear: lanes with the lane bit clear keep their element, lanes with the lane
// bit set pull the partner's lane ^ (1 << k) -- and mirrored for the bit-set zmm.
constexpr RbSel rb_ts_lo(int k)
{
    RbSel s{};
    const int b = 1 << k;
    for (int j = 0; j < 16; ++j) s.v[j] = (j & b) ? 16 + (j ^ b) : j;
    return s;
}
constexpr RbSel rb_ts_hi(int k)
{
    RbSel s{};
    const int b = 1 << k;
    for (int j = 0; j < 16; ++j) s.v[j] = (j & b) ? 16 + j : (j ^ b);
    return s;
}
constexpr std::array<RbSel, 4> g_ts_lo = { rb_ts_lo(0), rb_ts_lo(1), rb_ts_lo(2), rb_ts_lo(3) };
constexpr std::array<RbSel, 4> g_ts_hi = { rb_ts_hi(0), rb_ts_hi(1), rb_ts_hi(2), rb_ts_hi(3) };

// Per-rate decode tables. One tile row (n = tile n-offset) reads 16 slot states; slot
// s = 32*(n&7) + 8*((k>>1)&3) + (k&1) + 2*((k>>3)&1) + 4*(n>>3) lives at ring bits
// [b1-16, b1) with b1 = ((s+1)*rate2 >> 1) + 128*rate2; the scalar merges ring words
// w[b0/32] << 32 | w[(b1-1)/32] and shifts right by 31 - ((b1-1) & 31). A row's 16 slots
// span <= 27 consecutive slot indices -> <= 8 consecutive ring words (m(s) is monotone in
// s), so instead of gathering per lane, both words are selected with in-register permutes
// from ONE 16-word window of a per-tile extended ring copy; the window base and the
// per-lane selectors/shifts are compile-time constants per (rate, row).
template <int bits, bool hb>
struct RbTabs
{
    static constexpr int rate2 = 2 * bits + (hb ? 1 : 0);
    static constexpr int words32 = 4 * rate2;

    static constexpr int rb_slot(int n, int k)
    {
        return 32 * (n & 7) + 8 * ((k >> 1) & 3) + (k & 1) + 2 * ((k >> 3) & 1) + 4 * (n >> 3);
    }
    static constexpr int rb_b1(int s) { return (((s + 1) * rate2) >> 1) + 128 * rate2; }

    struct Row { int32_t base; int32_t sel_lo[16]; int32_t sel_hi[16]; int32_t sh[16]; };

    static constexpr Row make_row(int n)
    {
        Row r{};
        const int base = (rb_b1(rb_slot(n, 0)) - 1) / 32 - 1;    // window base (hi may need L-1)
        r.base = base;
        for (int k = 0; k < 16; ++k)
        {
            const int b1 = rb_b1(rb_slot(n, k));
            r.sel_lo[k] = (b1 - 1) / 32 - base;     // word holding the window's last bit
            r.sel_hi[k] = (b1 - 16) / 32 - base;    // word holding the window's first bit
            r.sh[k] = 31 - ((b1 - 1) & 31);
        }
        return r;
    }

    static constexpr std::array<Row, 16> make_rows()
    {
        std::array<Row, 16> a{};
        for (int n = 0; n < 16; ++n) a[n] = make_row(n);
        return a;
    }
    static constexpr std::array<Row, 16> rows = make_rows();

    // Extended-ring length: slot indices wrap the ring exactly twice, so the highest row's
    // window reaches word (b1(slot 228)-1)/32 + 16; ext = ring | ring | head mirrors that
    static constexpr int ext_words = (rb_b1(rb_slot(15, 0)) - 1) / 32 + 17;
};

// ------------------------------------------------------------------ intrinsic helpers

// 16 slot states of one tile row -> 16 mul1 codebook floats (integer-exact, then the scalar's
// exact affine float conversion)
RB16_TARGET inline __m512 rb_decode_row(const uint32_t* ext, const int32_t* base, const int32_t* sel_lo, const int32_t* sel_hi, const int32_t* sh)
{
    const __m512i win = _mm512_loadu_si512(ext + *base);
    const __m512i w_lo = _mm512_permutexvar_epi32(_mm512_loadu_si512(sel_lo), win);
    const __m512i w_hi = _mm512_permutexvar_epi32(_mm512_loadu_si512(sel_hi), win);
    const __m512i vsh = _mm512_loadu_si512(sh);
    // state = ((w[L] >> sh) | (w[F] << (32 - sh))) & 0xFFFF; sh = 0 -> shift count 32 -> 0
    const __m512i inv = _mm512_sub_epi32(_mm512_set1_epi32(32), vsh);
    __m512i st = _mm512_or_si512(_mm512_srlv_epi32(w_lo, vsh), _mm512_sllv_epi32(w_hi, inv));
    st = _mm512_and_si512(st, _mm512_set1_epi32(0xFFFF));
    // codebook: byte-sum of state * MUL1_MULT ((b0+b1)+(b2+b3), SWAR-exact), affine to float
    const __m512i x = _mm512_mullo_epi32(st, _mm512_set1_epi32(static_cast<int32_t>(R_MUL1_MULT)));
    const __m512i msk = _mm512_set1_epi32(0x00FF00FF);
    const __m512i s16 = _mm512_add_epi32(_mm512_and_si512(x, msk), _mm512_and_si512(_mm512_srli_epi32(x, 8), msk));
    // s16 packs (b0+b1) in bits 0..8 and (b2+b3) in bits 16..24: the pair-sum halves must be
    // masked apart before the final add, else the high half rides along scaled by 2^16
    const __m512i sum = _mm512_add_epi32(_mm512_and_si512(s16, _mm512_set1_epi32(0xFFFF)),
                                         _mm512_srli_epi32(s16, 16));
    __m512 v = _mm512_cvtepi32_ps(sum);
    v = _mm512_sub_ps(v, _mm512_set1_ps(510.0f));
    return _mm512_mul_ps(v, _mm512_set1_ps(g_rb_k_inv));
}

// One intra-vector Hadamard stage: t = partner lane's value; lanes with the stage bit clear get
// z+t (a+b), lanes with it set get t-z (a-b) -- the scalar's pair update, vectorized
RB16_TARGET inline __m512 rb_bfly_intra(__m512 z, const int32_t* sel, __mmask16 m)
{
    const __m512 t = _mm512_permutexvar_ps(_mm512_loadu_si512(sel), z);
    const __m512 s = _mm512_add_ps(z, t);
    return _mm512_mask_sub_ps(s, m, t, z);
}

// One cross-vector Hadamard stage: a holds the bit-clear half (a' = a+b), b the bit-set half
// (b' = a-b) -- same operand pairing as the scalar's (base+i, base+width+i)
RB16_TARGET inline void rb_bfly_cross(__m512& a, __m512& b)
{
    const __m512 s = _mm512_add_ps(a, b);
    b = _mm512_sub_ps(a, b);
    a = s;
}

// Full 128-point Hadamard over one contiguous row + the 1/sqrt(128) scale, replicating the
// scalar's stage order exactly (width 1, 2, 4, ..., 64)
RB16_TARGET inline void rb_had_scale_row(float* row)
{
    __m512 z[8];
    for (int i = 0; i < 8; ++i) z[i] = _mm512_load_ps(row + i * 16);
    for (int i = 0; i < 8; ++i) z[i] = rb_bfly_intra(z[i], g_sel1.v, 0xAAAA);
    for (int i = 0; i < 8; ++i) z[i] = rb_bfly_intra(z[i], g_sel2.v, 0xCCCC);
    for (int i = 0; i < 8; ++i) z[i] = rb_bfly_intra(z[i], g_sel4.v, 0xF0F0);
    for (int i = 0; i < 8; ++i) z[i] = rb_bfly_intra(z[i], g_sel8.v, 0xFF00);
    for (int i = 0; i < 8; i += 2) rb_bfly_cross(z[i], z[i + 1]);                       // width 16
    for (int i = 0; i < 8; i += 4)                                                      // width 32
    {
        rb_bfly_cross(z[i], z[i + 2]);
        rb_bfly_cross(z[i + 1], z[i + 3]);
    }
    rb_bfly_cross(z[0], z[4]);                                                          // width 64
    rb_bfly_cross(z[1], z[5]);
    rb_bfly_cross(z[2], z[6]);
    rb_bfly_cross(z[3], z[7]);
    const __m512 sc = _mm512_set1_ps(R_HAD_SCALE);
    for (int i = 0; i < 8; ++i) _mm512_store_ps(row + i * 16, _mm512_mul_ps(z[i], sc));
}

// In-register 16x16 float transpose (16 zmms of one row each in, one column each out)
RB16_TARGET inline void rb_transpose16(__m512* a)
{
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 16; ++i)
        {
            const int b = 1 << k;
            if (i & b) continue;        // pair every m with bit k clear against m ^ b
            const int j = i + b;
            const __m512 x = _mm512_permutex2var_ps(a[i], _mm512_loadu_si512(g_ts_lo[k].v), a[j]);
            const __m512 y = _mm512_permutex2var_ps(a[i], _mm512_loadu_si512(g_ts_hi[k].v), a[j]);
            a[i] = x;
            a[j] = y;
        }
}

// 128x128 fp32 transpose, 16x16 tile at a time (src/dst 64-byte aligned, offsets are
// multiples of 16 floats)
RB16_TARGET inline void rb_transpose_128(const float* src, float* dst)
{
    for (int r = 0; r < 128; r += 16)
        for (int c = 0; c < 128; c += 16)
        {
            __m512 a[16];
            for (int i = 0; i < 16; ++i) a[i] = _mm512_load_ps(src + (r + i) * 128 + c);
            rb_transpose16(a);
            for (int i = 0; i < 16; ++i) _mm512_store_ps(dst + (c + i) * 128 + r, a[i]);
        }
}

// fp32 -> BF16 round-to-nearest-even, packed halves: the integer rounding formula from
// c10::BFloat16's scalar conversion (no AVX512-BF16 instruction dependency)
RB16_TARGET inline void rb_bf16_store(uint16_t* dst, __m512 v)
{
    const __m512i u = _mm512_castps_si512(v);
    const __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                                                           _mm512_and_si512(_mm512_srli_epi32(u, 16),
                                                                            _mm512_set1_epi32(1))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16)));
}

// ------------------------------------------------------------------ block kernel

// Rows [n_begin, n_end) (multiples of 128), all k: decode each 128x128 block into an L2-resident
// buffer, run both Hadamard passes in-cache (the n-side on the transposed block), apply the
// diagonal scales and stream the block out as BF16. Identical block math to recon_bf16.cpp's
// dequant_impl -- same values in, same fp32 op order, same rounding out.
template <int bits, bool hb>
RB16_TARGET void dequant_impl_avx
(
    const MoeCpuMatrix& mat,
    uint16_t* out,
    int n_begin,
    int n_end,
    int g
)
{
    using Tabs = RbTabs<bits, hb>;
    constexpr int words = 8 * Tabs::rate2;      // u16 per tile
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;

    alignas(64) float blk[128 * 128];
    alignas(64) float blk2[128 * 128];
    for (int n0 = n_begin; n0 < n_end; n0 += 128)
        for (int k0 = 0; k0 < mat.k; k0 += 128)
        {
            // decode the block's 64 tiles into blk ([n][k] row-major)
            for (int nt = 0; nt < 8; ++nt)
                for (int kt = 0; kt < 8; ++kt)
                {
                    const int ntl = n0 / 16 + nt;
                    const int ktl = k0 / 16 + kt;
                    const size_t stored = g
                        ? static_cast<size_t>(ntl / g) * tiles_k * g + static_cast<size_t>(ktl) * g + (ntl % g)
                        : static_cast<size_t>(ktl) * tiles_n + ntl;    // native: [k/16, n/16, words]
                    // per-tile extended ring (ring | ring | head): the rows' 16-word windows
                    // index it directly, no per-lane mod
                    const uint32_t* ring = reinterpret_cast<const uint32_t*>(mat.trellis + stored * words);
                    uint32_t ext[Tabs::ext_words];
                    for (int i = 0; i < Tabs::words32; ++i) ext[i] = ring[i];
                    for (int i = Tabs::words32; i < Tabs::ext_words; ++i) ext[i] = ext[i - Tabs::words32];
                    float* base = blk + static_cast<size_t>(nt) * 16 * 128 + kt * 16;
                    for (int row = 0; row < 16; ++row)
                    {
                        const auto& tb = Tabs::rows[row];
                        _mm512_store_ps(base + static_cast<size_t>(row) * 128,
                                        rb_decode_row(ext, &tb.base, tb.sel_lo, tb.sel_hi, tb.sh));
                    }
                }
            // k-side Hadamard per block row (contiguous), scale once
            for (int r = 0; r < 128; ++r) rb_had_scale_row(blk + r * 128);
            // n-side Hadamard per block column, on the in-register transpose (identical
            // element pairing and stage order to the scalar's strided column walk)
            rb_transpose_128(blk, blk2);
            for (int r = 0; r < 128; ++r) rb_had_scale_row(blk2 + r * 128);
            rb_transpose_128(blk2, blk);
            // diagonal scales strictly after both butterflies, fused into the BF16 stream-out
            __m512 su[8];
            for (int c = 0; c < 8; ++c)
                su[c] = _mm512_cvtph_ps(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(mat.suh + k0 + c * 16)));
            for (int r = 0; r < 128; ++r)
            {
                const __m512 sv = _mm512_set1_ps(rb_half_to_float(mat.svh[n0 + r]));
                uint16_t* orow = out + static_cast<size_t>(n0 + r) * mat.k + k0;
                const float* brow = blk + r * 128;
                for (int c = 0; c < 8; ++c)
                    rb_bf16_store(orow + c * 16,
                                  _mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(brow + c * 16), sv), su[c]));
            }
        }
}

inline bool rb_host_ok()
{
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512bf16");
#else
    return false;
#endif
}

// Block-strip parallelism over std::thread. The JIT CPU build compiles without -fopenmp, and
// the inline at::parallel_for from <ATen/ParallelOpenMP.h> then silently drops its omp pragmas
// and runs serial, so the strip loop needs its own workers. Thread cap = at::get_num_threads()
// (torch's intra-op setting: OMP_NUM_THREADS / torch.set_num_threads). Items are independent
// 128-row strips, so the split is deterministic for any thread count. The task lambda must
// not throw (a worker exception would terminate the process); the kernel is noexcept.
template <class F>
inline void rb_parallel_for(int n_items, F&& fn)
{
    int nt = at::get_num_threads();
    if (nt > n_items) nt = n_items;
    if (nt < 2 || n_items < 2)
    {
        fn(0, n_items);
        return;
    }
    const int chunk = (n_items + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(nt - 1);
    for (int w = 1; w < nt; ++w)
    {
        const int b = w * chunk;
        const int e = std::min(n_items, b + chunk);
        if (b >= e) break;
        workers.emplace_back([&fn, b, e]() { fn(b, e); });
    }
    fn(0, std::min(n_items, chunk));
    for (auto& t : workers) t.join();
}

}  // namespace

at::Tensor exl3_cpu_dequant_bf16_avx512
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    double K,
    bool swizzled
)
{
    TORCH_CHECK(rb_host_ok(),
                "exl3_cpu_dequant_bf16_avx512 requires AVX-512 F/BW/BF16; this host lacks it -- "
                "fall back to exl3_cpu_dequant_bf16 (scalar)");
    TORCH_CHECK(trellis.device().is_cpu() && suh.device().is_cpu() && svh.device().is_cpu(),
                "exl3_cpu_dequant_bf16_avx512: tensors must be on CPU");
    TORCH_CHECK(trellis.scalar_type() == at::kShort && trellis.is_contiguous(),
                "trellis must be contiguous CPU int16");
    TORCH_CHECK(suh.scalar_type() == at::kHalf && suh.is_contiguous()
                    && svh.scalar_type() == at::kHalf && svh.is_contiguous(),
                "suh/svh must be contiguous CPU half");

    MoeCpuMatrix mat = exl3_cpu_mul1_make_matrix(trellis, suh, svh, nullptr, swizzled);
    const double K_tile = mat.bits + (mat.hb ? 0.5 : 0.0);
    TORCH_CHECK(std::fabs(K - K_tile) < 1e-6,
                "K (", K, ") does not match the trellis tile width (", K_tile, ")");
    // Same-layout rules as the scalar binding: band-2 planar (AVX2-only hosts) is out of scope
    TORCH_CHECK(!exl3_moe_cpu_planar_layout(K_tile),
                "exl3_cpu_dequant_bf16_avx512: the AVX2 planar packed layout (band-2 rates on "
                "AVX2-only hosts) is not supported; this tier targets the native/band-8 layouts");
    const int grp = mat.swz ? exl3_moe_cpu_swizzle_group(K_tile) : 0;
    TORCH_CHECK(mat.k % 128 == 0 && mat.n % 128 == 0,
                "exl3_cpu_dequant_bf16_avx512: in/out features must be multiples of 128 "
                "(Hadamard blocks), got ", mat.k, "x", mat.n);
    TORCH_CHECK(suh.numel() == mat.k && svh.numel() == mat.n, "suh/svh length mismatch");

    at::Tensor out = at::empty({mat.n, mat.k},
                               at::TensorOptions().dtype(at::kBFloat16).device(trellis.device()));
    uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr());

    switch (mat.bits)
    {
#define R_BF16X_CASE(B)                                                          \
        case B:                                                                  \
            if (mat.hb)                                                          \
                rb_parallel_for(mat.n / 128, [&](int b, int e)                   \
                { dequant_impl_avx<B, true>(mat, op, b * 128, e * 128, grp); }); \
            else                                                                 \
                rb_parallel_for(mat.n / 128, [&](int b, int e)                   \
                { dequant_impl_avx<B, false>(mat, op, b * 128, e * 128, grp); });\
            break;
        R_BF16X_CASE(1) R_BF16X_CASE(2) R_BF16X_CASE(3) R_BF16X_CASE(4)
        R_BF16X_CASE(5) R_BF16X_CASE(6) R_BF16X_CASE(7) R_BF16X_CASE(8)
#undef R_BF16X_CASE
        default: TORCH_CHECK(false, "exl3_cpu_dequant_bf16_avx512: unsupported bitrate ", mat.bits);
    }
    return out;
}

#else  // !x86: the mul1 format is only decoded by the x86 kernels
#include "recon_bf16_avx512.h"
#include <torch/extension.h>

at::Tensor exl3_cpu_dequant_bf16_avx512
(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    bool
)
{
    TORCH_CHECK(false, "exl3_cpu_dequant_bf16_avx512 is not available on this CPU architecture "
                       "(x86 AVX-512 only)");
    return {};
}
#endif
