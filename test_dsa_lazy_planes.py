"""Synthetic parity harness for plan 05 (GLM-5.3 DSA prefill), gates OFF vs ON.

Part A (EXL3_DSA_LAZY_PLANES): drives a dense->sparse crossing on synthetic packed
[k || gate] idx-plane rows and proves the lazy arm's final POOL-PLANE STATE is
bit-identical (torch.equal, not allclose) to the incremental arm's:
  1. prefill-crossing   -- deferred dense chunks, one bulk rebuild on the first
                           sparse chunk, incremental afterwards
  2. decode-crossing    -- deferred dense chunks, then the seqlen=0 bulk rebuild
                           bc_mla_step performs before its first captured step,
                           incremental afterwards
The idx plane is filled through the REAL cache-layer append path in both arms, so
the comparison exercises update_pool_direct/mla_plane_append end to end.

Part B (EXL3_DSA_BATCH_UPPROJ): runs mla_attn_triton_prefill_mha with the two-GEMM
path vs the batched w_kv_cat path on tiny shapes (single-tile AND multi-tile with
running softmax state + causal masking) and compares attention outputs.

Run (from the worktree root):
  PYTHONPATH=. ../.venv/bin/python test_dsa_lazy_planes.py
"""
import sys
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.mla import CacheLayer_MLA_fp16
from exllamav3.modules.mla_attn import MLAttention
from exllamav3.modules.attention_fn.mla_triton import (
    mla_attn_triton_prefill_mha, mla_plane_append,
)

DEV = torch.device("cuda")
D_I = 128          # index_head_dim (GLM-5.3)
P = 4              # index_kpool (GLM-5.3)
DENSE_WIN = 2048   # index_topk (GLM-5.3 yamz)


class FakeAttn:
    """The attributes CacheLayer_MLA_fp16 and _update_pool_plane read off MLAttention."""
    def __init__(self, dev):
        self.kv_lora_rank = 64
        self.qk_rope_head_dim = 16
        self.index_head_dim = D_I
        self.index_kpool = P
        self.idx_plane_dim = 2 * D_I   # packed [k || gate]
        # module-side state used by the unbound method
        self.idx_kpool_ape = torch.randn(P, D_I, device = dev, dtype = torch.float) * 0.5
        self._dsa_pool_offs = None


def bind_update_pool_plane(attn):
    """MLAttention._update_pool_plane bound to the fake module (no model needed)."""
    return MLAttention._update_pool_plane.__get__(attn)


def make_layer(dev, max_tokens, attn):
    layer = CacheLayer_MLA_fp16(None, attn, 0, max_tokens)
    layer.alloc(dev)
    return layer


def feed_chunk(layer, block_table, cache_seqlens, rows, host_len):
    """Append packed idx rows for [host_len, host_len + len) via the real cache path,
    mirroring _attend: update_idx_direct then (optionally) pool upkeep."""
    bsz, length, _ = rows.shape
    mla_plane_append(rows.reshape(bsz, length, 2 * D_I), layer.get_idx(),
                     block_table, cache_seqlens)
    cache_seqlens += length


def run_arm(dev, chunks, lazy, decode_crossing, attn):
    """Both arms see identical packed rows and the SAME module (APE/offset state);
    returns (idx_plane, pool_plane)."""
    total = sum(chunks)
    pages = -(-total // PAGE_SIZE)
    layer = make_layer(dev, pages * PAGE_SIZE, attn)
    upd = bind_update_pool_plane(attn)
    torch.manual_seed(1234)
    rows_all = (torch.randn(1, total, 2 * D_I, device = dev) * 0.3).to(torch.half)
    block_table = torch.arange(pages, dtype = torch.int32, device = dev).view(1, pages)
    cache_seqlens = torch.zeros((1,), dtype = torch.int32, device = dev)
    host = 0
    lazy_pending = False
    for i, length in enumerate(chunks):
        feed_chunk(layer, block_table, cache_seqlens,
                   rows_all[:, host:host + length], host)
        sparse = host + length > DENSE_WIN
        if not lazy:
            upd(layer, block_table, [host], length)
        else:
            if decode_crossing:
                # emulate bc_mla_step: while dense defer; the crossing "step" bulk-builds
                # pools complete BEFORE it appends (seqlen=0), then incremental resumes
                if lazy_pending and (sparse or i == len(chunks) - 1):
                    upd(layer, block_table, [host], 0, bulk = True)
                    lazy_pending = False
                    upd(layer, block_table, [host], length)
                elif sparse:
                    upd(layer, block_table, [host], length)
                else:
                    lazy_pending = True
            else:
                # emulate _attend: defer while dense; first sparse chunk bulk-builds
                # [0, total//P) INCLUSIVE of its own pools, then incremental resumes
                if not sparse:
                    lazy_pending = True
                elif lazy_pending:
                    upd(layer, block_table, [host], length, bulk = True)
                    lazy_pending = False
                else:
                    upd(layer, block_table, [host], length)
        host += length
    assert not lazy_pending
    return layer.get_idx().clone(), layer.get_pool().clone(), layer


def test_part_a(dev):
    # 1. prefill crossing: 2048 dense window crossed inside chunk 3
    chunks = [1024, 1024, 2000, 1500, 1000]
    attn = FakeAttn(dev)
    idx_o, pool_o, _ = run_arm(dev, chunks, lazy = False, decode_crossing = False, attn = attn)
    idx_n, pool_n, _ = run_arm(dev, chunks, lazy = True, decode_crossing = False, attn = attn)
    assert torch.equal(idx_o, idx_n), "idx planes diverged"
    assert torch.equal(pool_o, pool_n), "pool planes not bit-identical after bulk rebuild"
    n_pools = sum(chunks) // P
    nz = pool_o.view(-1, D_I)[: n_pools].abs().sum(dim = 1)
    assert (nz > 0).all(), "reference arm left unwritten pools -- harness is broken"

    # 2. decode crossing: dense prefill, then the seqlen=0 bc-style rebuild mid-stream
    chunks = [1500, 500, 64, 2000, 1000]
    attn = FakeAttn(dev)
    idx_o, pool_o, _ = run_arm(dev, chunks, lazy = False, decode_crossing = False, attn = attn)
    idx_n, pool_n, _ = run_arm(dev, chunks, lazy = True, decode_crossing = True, attn = attn)
    assert torch.equal(idx_o, idx_n), "idx planes diverged (decode crossing)"
    assert torch.equal(pool_o, pool_n), "pool planes not bit-identical (decode crossing)"

    print(f"part A OK: pool planes bit-identical ({n_pools} pools, prefill + decode crossings)")


def test_part_b(dev):
    torch.manual_seed(7)
    H, D_c, D_n, D_v, D_r = 4, 64, 32, 32, 16
    q_len, total = 96, 352   # past 256 + chunk 96 -> multi-tile at tile_size 128
    pages = -(-total // PAGE_SIZE)
    w_uk = (torch.randn(D_c, H * D_n, device = dev) * 0.05).to(torch.half)
    w_uv = (torch.randn(D_c, H * D_v, device = dev) * 0.05).to(torch.half)
    w_kv_cat = torch.cat((w_uk, w_uv), dim = 1).contiguous()
    q = (torch.randn(q_len, H, D_n + D_r, device = dev) * 0.5).to(torch.half)
    ckv = (torch.randn(pages, PAGE_SIZE, 1, D_c, device = dev) * 0.4).to(torch.half)
    kpe = (torch.randn(pages, PAGE_SIZE, 1, D_r, device = dev) * 0.4).to(torch.half)
    bt = torch.arange(pages, dtype = torch.int32, device = dev).view(1, pages)
    host = [total - q_len]
    kw = dict(q_len = q_len, v_head_dim = D_v, qk_nope_head_dim = D_n,
              softmax_scale = (D_n + D_r) ** -0.5, pre_appended_len = q_len)
    for tile_size, tag in ((4096, "single-tile"), (128, "multi-tile")):
        o_ref = mla_attn_triton_prefill_mha(
            q, w_uk, w_uv, ckv, kpe, bt, host, 1, tile_size = tile_size, **kw)
        o_new = mla_attn_triton_prefill_mha(
            q, w_uk, w_uv, ckv, kpe, bt, host, 1, tile_size = tile_size,
            w_kv_cat = w_kv_cat, **kw)
        diff = (o_ref.float() - o_new.float()).abs()
        assert torch.isfinite(o_new).all()
        assert diff.max() <= 2e-2, f"{tag}: batched up-proj diverged (max {diff.max():.4f})"
        print(f"part B OK: {tag} attention outputs match (max-abs {diff.max():.2e}, "
              f"mean {diff.mean():.2e})")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA unavailable: synthetic parity needs the GPU"); sys.exit(1)
    dev = torch.cuda.current_device()
    test_part_a(dev)
    test_part_b(dev)
    print("ALL SYNTHETIC PARITY CHECKS PASSED")
