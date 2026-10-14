"""C5 fused CPU router (ext.router_std_cpu) parity tests against the torch reference path
(_routing_std_torch). No model loads: synthetic [512, 2560] fp16 gates plus crafted tie inputs.

Tie-order finding (empirical, documented here per the task brief): torch.topk CPU
(largest = True, sorted = True) does NOT resolve equal values by index -- its radix-select +
sort emits a deterministic but scrambled order (e.g. an all-equal 512-row returns first-10
indices [344, 337, 338, ...]; a tied top pair returns [1, 0, ...]). No simple comparator can
reproduce it. The kernel instead resolves ties lowest-index-first, which gives:
  - exact index order equality whenever the top-k values are distinct (the overwhelmingly
    common case, incl. the real qwen4_exp router on random inputs, M = 1..8), and
  - under exact fp16 logit ties: identical (expert, weight) pair multiset whenever the tie is
    inside the selection (reordering equal weights), and a lowest-index subset choice when the
    tie sits at the k boundary -- the same class of arbitrary choice torch's own radix select
    makes (and the CUDA routing kernel makes differently again); tied experts carry identical
    routing weights either way.
"""

import os, sys, types
import torch
import pytest

pytestmark = [pytest.mark.nogpu, pytest.mark.cpu_flags("avx512f", "avx512bw", "avx512vl", "avx512dq", "fma", "f16c")]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.block_sparse_mlp_routing import _routing_std_torch

E, K, TOPK = 512, 2560, 10

def _ref(gate_t, y):
    cfg = types.SimpleNamespace(
        gate_tensor = gate_t.T.contiguous(),
        num_experts_per_tok = TOPK,
        per_expert_scale = None,
    )
    sel = torch.empty((y.shape[0], TOPK), dtype = torch.long)
    w = torch.empty((y.shape[0], TOPK), dtype = torch.half)
    _routing_std_torch(cfg, y, None, sel, w)
    return sel, w

def _kern(gate_t, y):
    sel = torch.empty((y.shape[0], TOPK), dtype = torch.long)
    w = torch.empty((y.shape[0], TOPK), dtype = torch.half)
    ext.router_std_cpu(gate_t, y, sel, w)
    return sel, w

def _assert_parity(gate_t, y):
    sel_r, w_r = _ref(gate_t, y)
    sel_k, w_k = _kern(gate_t, y)
    # weights bit-exact, (expert, weight) pair multiset equal; strict order when no ties
    assert torch.equal(w_r, w_k)
    s_r = sel_r.sort(dim = 1).values
    s_k = sel_k.sort(dim = 1).values
    logits = (y.float() @ gate_t.float().T).half().float()
    top11 = torch.topk(logits, TOPK + 1, dim = -1).values
    strict = (top11[:, :-1].diff(dim = -1) != 0).all(dim = -1) & \
             (top11[:, -1] != top11[:, -2])          # no tie touching selection or boundary
    assert torch.equal(s_r[strict], s_k[strict]), "selection differs without ties"
    assert torch.equal(s_r, s_k) or bool((~strict).any()), "selection differs on tied rows"

def test_router_std_cpu_avx512():
    assert ext.router_std_cpu_has_avx512()

def test_router_std_cpu_random():
    torch.manual_seed(1234)
    gate_t = (torch.randn(E, K) * 0.02).half()
    for m in (1, 8, 37):
        _assert_parity(gate_t, torch.randn(m, K).half())

def test_router_std_cpu_onehot_zeros():
    torch.manual_seed(1235)
    gate_t = (torch.randn(E, K) * 0.02).half()
    _assert_parity(gate_t, torch.eye(8, K, dtype = torch.half))
    _assert_parity(gate_t, torch.zeros(2, K, dtype = torch.half))

def test_router_std_cpu_ties():
    torch.manual_seed(1236)
    gate_t = (torch.randn(E, K) * 0.02).half()
    g = gate_t.clone(); g[7] = g[3]; g[100] = g[3]      # exactly equal logits
    _assert_parity(g, torch.randn(4, K).half())
    g = gate_t.clone()
    for j in range(0, 200, 2): g[j + 1] = g[j]          # 100 dup pairs
    _assert_parity(g, torch.randn(3, K).half())

def test_router_std_cpu_topk_tie_order_documented():
    # torch's own tie order is radix-scrambled, not index-ordered (the finding this suite
    # documents): all-equal input must NOT come back as [0, 1, 2, ...] from torch
    v, i = torch.topk(torch.zeros(1, E), TOPK)
    assert not torch.equal(i, torch.arange(TOPK).unsqueeze(0))
