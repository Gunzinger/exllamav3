"""
Multi-row GEMM tiles on the RDNA/ROCm shape table (plan 03): the first shape with
TILESIZE_M > 16 (index 5 on both backends; ROCm gains 32x16x128 where CUDA already has
CUDA-table shapes 5-7). What is under test:

- the shape exists, is returned by exl3_gemm_num_kernel_shapes, and its row-tile predicate
  is the same upward-closed rule the 16-row shapes follow (excluded only while
  size_m <= TILESIZE_M / 2: a wider tile must replace at least two 16-row passes)
- forced runs at every M in the direct-GEMM band (16..144, including values that do not
  divide the row tile) match the independently decoded fp64 reference within the suite
  tolerance, i.e. the numerics class of the M=16 shapes (fp16 in, fp32 accumulation)
- the same holds on real 2.05 bpw checkpoint tensors (mul1 codebook, hidden 2560 /
  moe-intermediate 640 shapes of qwen3.8-flash-next), where the new tile must also agree
  with the best 16-row tile on the same input

Tolerance: the shared suite bound (relative RMS < 2.5e-3, max < 4e-3 vs fp64; see
test_exl3_gemm_shapes.py).
"""

import os

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.exl3 import codebook_flags, generator, rand_scale

REAL_MODEL_DIR = os.environ.get(
    "EXL3_TEST_REAL_MODEL",
    "/home/daniel/development/gunzi/local-llm/models/qwen3.8-flash-next-exl3/2.05bpw",
)
M_BAND = (16, 17, 31, 32, 33, 64, 100, 144)
N_COLS = (128, 2560, 4096)


def _n():
    return ext.exl3_gemm_num_kernel_shapes()


def _multirow_shape():
    """First shape whose row tile exceeds 16 (5 on both tables), or None"""
    for s in range(1, _n() + 1):
        if next(m for m in range(1, 1025) if ext.exl3_gemm_shape_compat(s, m, 1536, 1536, 4)) > 1:
            return s
    return None


def _tiles(s):
    tk = next(k for k in range(16, 1024, 16) if ext.exl3_gemm_shape_compat(s, 128, k, 3072, 4))
    tn = next(n for n in range(128, 4096, 128) if ext.exl3_gemm_shape_compat(s, 128, 3072, n, 4))
    m_min = next(m for m in range(1, 1025) if ext.exl3_gemm_shape_compat(s, m, 3072, 3072, 4))
    return tk, tn, m_min


def _gemm(A, tr, suh, svh, n, s, mcg, mul1):
    C = torch.empty((A.shape[0], n), dtype = torch.half, device = A.device)
    r = ext.exl3_gemm(A, tr, C, suh, torch.empty_like(A), svh, s, mcg, mul1, 0)
    return r, C


def _assert_close(C, ref):
    e = C.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@pytest.mark.rocm_only
def test_multirow_shape_exists():
    s = _multirow_shape()
    assert s is not None, "no TILESIZE_M > 16 shape in the table"
    tk, tn, m_min = _tiles(s)
    assert tn % 128 == 0
    # upward-closed in m, excluded while the tile would not replace two 16-row passes
    assert m_min > 1
    for m in range(1, m_min):
        assert not ext.exl3_gemm_shape_compat(s, m, 1536, 1536, 4)
    for m in (m_min, m_min * 4, 144):
        assert ext.exl3_gemm_shape_compat(s, m, 1536, 1536, 4)


def _cases():
    s = _multirow_shape()
    if s is None:
        return []
    # 3inst at 4 bits (the dominant class) and mul1 at 2.5 bits (the 2.05 bpw class)
    return [(s, 4, "3inst"), (s, 2.5, "mul1")]


@pytest.mark.parametrize("s, K, cb", _cases())
@pytest.mark.parametrize("n", N_COLS)
@pytest.mark.parametrize("k", (640, 2560))  # real model dims, divisible by every k tile
@torch.inference_mode()
def test_multirow_band_vs_reference(device, s, K, cb, n, k):
    half = not float(K).is_integer()
    mcg, mul1 = codebook_flags(cb)
    seed = s * 100 + int(K * 2)
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)) \
        .reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g), rand_scale(n, g)
    tr = torch.from_numpy(p.copy()).to(device)
    W = tref.dequant(p, suh, svh, K, cb)
    ran = 0
    for m in M_BAND:
        if not ext.exl3_gemm_shape_compat(s, m, k, n, int(K), half):
            continue
        A = (torch.randn(m, k, generator = g) * 0.5).half()
        r, C = _gemm(A.to(device), tr, suh.to(device), svh.to(device), n, s, mcg, mul1)
        assert r == s, f"forced shape {s} reported {r}"
        _assert_close(C, A.double() @ W)
        ran += 1
    assert ran > 0, "no compatible row count in the band"


@pytest.mark.rocm_only
@pytest.mark.parametrize("proj, k, n", (
    ("mlp.shared_expert.gate_proj", 2560, 640),
    ("mlp.shared_expert.down_proj", 640, 2560),
))
@torch.inference_mode()
def test_multirow_real_checkpoint(device, proj, k, n):
    """Real 2.05 bpw tensors: the multirow tile stays in the M=16 numerics class and
    agrees with the best 16-row tile on the same inputs"""
    from safetensors import safe_open
    s_new = _multirow_shape()
    # best 16-row shape that actually divides this projection's n (640 admits only N=128 tiles)
    s_old = next(
        s for s in range(4, 0, -1)
        if ext.exl3_gemm_shape_compat(s, 32, k, n, 2)
    )
    prefix = "model.language_model.layers.3." + proj
    shard = os.path.join(REAL_MODEL_DIR, "model-00001-of-00005.safetensors")
    if not os.path.exists(shard):
        pytest.skip("real 2.05 bpw checkpoint not available")
    with safe_open(shard, framework = "pt") as f:
        tr = f.get_tensor(prefix + ".trellis").to(device)
        suh = f.get_tensor(prefix + ".suh").to(device).half()
        svh = f.get_tensor(prefix + ".svh").to(device).half()
        mul1 = bool(f.get_tensor(prefix + ".mul1").item())
    assert tr.shape[0] * 16 == k and tr.shape[1] * 16 == n
    g = generator(42)
    for m in (17, 32, 64, 100, 144):
        A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
        _, C_new = _gemm(A, tr, suh, svh, n, s_new, False, mul1)
        _, C_old = _gemm(A, tr, suh, svh, n, s_old, False, mul1)
        d = (C_new.double() - C_old.double()).abs().max().item()
        scale = C_old.double().abs().max().item()
        assert d / scale < 4e-3, f"m={m}: max abs diff {d:.3e} (rel {d / scale:.2e})"
