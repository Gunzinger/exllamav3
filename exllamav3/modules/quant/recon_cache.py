"""
Reconstruct-once FP16 shadow cache for LinearEXL3 (branch opt/recon-cache, plan 04).

Every LinearEXL3 call on the rows > 144 tier (all dense/attention projections during prefill)
re-runs the whole-matrix dequantize (reconstruct) before the rocBLAS hgemm. For an 8192-token
prompt split into chunks, every projection's weights are dequantized once per chunk, and every
warm prompt resend dequantizes again. This module lets a layer materialize that dequantized FP16
matrix once into a "shadow" buffer and reuse it on later calls.

Design contract (plan 04):

- Opt-in via EXL3_RECON_CACHE_MIB (budget in MiB; 0 = off, the default). The budget is a global
  GTT cap enforced by LRU eviction here; there is no per-model split.
- One global LRU keyed by the *identity of the packed trellis tensor* plus a basis flag, not by
  layer name: two model instances over the same checkpoint share the shadow, and the same tensor
  can carry two shadows (plain and fused-Hadamard basis, see below).
- Shadows are pure functions of the packed checkpoint tensors, so they never go stale within a
  model instance; they are dropped on model unload (model.unload -> clear(), and per-layer
  LinearEXL3.unload -> drop_linear()).
- MoE routed experts are excluded (see key_cacheable): too large to shadow, and they run the
  fused MoE tier rather than this path.
- Only the rows > 144 tier consults the cache (decode's batch-1 GEMV path never touches it), and
  only the single-shot N path (out_features <= MAX_RECONSTRUCT_SLICE_N) is cached.

Basis flag: reconstruct_hgemm has two weight bases. The plain path reconstructs the
*rotated-basis* weights (input had_r_128 on x, output had_r_128 on y around the hgemm); the
fused-Hadamard path (rows >= 1024, 128-divisible dims) reconstructs *original-basis* weights with
both Hadamards folded into the reconstruct kernel. A shadow stores whichever basis its
materializing call used, and a hit replays the exact same surrounding op sequence on the same
contents, so cached outputs are byte-identical to reconstruct-per-call. A layer that sees both
row regimes may hold both bases; the LRU budget covers the total.

Accounting seam note: the plan asked to subtract shadow bytes where `mem_info_gtt_used` /
`use_per_device` budgets are checked, but no such runtime GTT accounting hook exists in this tree
(used budgets are resolved once at load time in model_ls.py/model_tp.py via
set_memory_fraction_use, and shadows materialize lazily after load). The best-effort seam is
therefore the budget enforcement here plus the total_bytes() accessor below for future
integration; documented in RESUME.md.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict

import torch

from ...util.cuda_flags import recon_cache_budget_bytes

# Routed-expert keys look like "...mlp.experts.<idx>.gate_up" (block_sparse_mlp.py builds them as
# f"{key}.{key_gate/up/down}" with {expert_idx} substituted). Shared experts
# ("...shared_experts.gate_up") are dense projections and stay in scope: the digit after the
# segment is what separates them.
_EXPERT_KEY_RE = re.compile(r"experts\.\d")


def key_cacheable(key: str | None) -> bool:
    """False for MoE routed-expert tensors (excluded per plan); unknown/None keys are conservative."""
    if not key:
        return False
    return _EXPERT_KEY_RE.search(key) is None


class _ShadowEntry:
    __slots__ = ("tensor", "bytes", "trellis", "in_features", "out_features", "device_index", "basis", "owner_key")

    def __init__(self, tensor: torch.Tensor, trellis: torch.Tensor, in_features: int, out_features: int,
                 device_index: int | None, basis: int, owner_key: str | None):
        self.tensor = tensor
        self.bytes = tensor.numel() * tensor.element_size()
        # Strong reference: the shadow is only meaningful while its packed source lives, and
        # holding the trellis also keeps its data_ptr (the LRU key) from being recycled.
        self.trellis = trellis
        self.in_features = in_features
        self.out_features = out_features
        self.device_index = device_index
        self.basis = basis
        self.owner_key = owner_key


class ReconCache:
    """Global GTT-budgeted LRU of per-layer FP16 shadow matrices."""

    def __init__(self):
        self._entries: OrderedDict[tuple, _ShadowEntry] = OrderedDict()
        self._total = 0
        self._budget = -1  # unresolved; configure() reads the env on first use

    # -- configuration ----------------------------------------------------------------------------------

    def configure(self) -> None:
        """Re-read EXL3_RECON_CACHE_MIB (via util/cuda_flags.py) and drop back to fit if shrunken."""
        self._budget = recon_cache_budget_bytes()
        self._evict_to(self._budget)

    def set_budget_bytes(self, budget: int) -> None:
        """Direct budget override (tests); a negative/zero budget disables the cache."""
        self._budget = max(0, int(budget))
        self._evict_to(self._budget)

    def enabled(self) -> bool:
        if self._budget < 0:
            self.configure()
        return self._budget > 0

    @property
    def budget_bytes(self) -> int:
        if self._budget < 0:
            self.configure()
        return self._budget

    # -- inspection --------------------------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return self._total

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict:
        return {"entries": len(self._entries), "total_bytes": self._total, "budget_bytes": self.budget_bytes}

    # -- cacheability -------------------------------------------------------------------------------------

    def key_cacheable(self, key: str | None) -> bool:
        return key_cacheable(key)

    # -- core LRU ------------------------------------------------------------------------------------------

    def _key(self, lin, basis: int) -> tuple:
        device_index = lin.trellis.device.index if lin.trellis.device.type == "cuda" else -1
        return (device_index, lin.trellis.data_ptr(), int(basis))

    def _evict_to(self, budget: int) -> None:
        while self._total > budget and self._entries:
            _, entry = self._entries.popitem(last=False)  # evict least-recently-used
            self._total -= entry.bytes

    def lookup(self, lin, basis: int):
        """Returns (tensor, hit). On hit the LRU entry is refreshed and the shadow is ready for
        hgemm. On miss returns (None, False); the caller materializes a fresh weight matrix and
        may offer it back via store()."""
        if not self.enabled():
            return None, False
        key = self._key(lin, basis)
        entry = self._entries.get(key)
        if entry is None:
            return None, False
        # Identity belt-and-braces: a data_ptr collision can only come from a recycled allocation
        # with mismatched shape; treat it as a miss and drop the stale entry.
        if (entry.in_features != lin.in_features or entry.out_features != lin.out_features
                or entry.trellis is not lin.trellis):
            del self._entries[key]
            self._total -= entry.bytes
            return None, False
        self._entries.move_to_end(key)
        return entry.tensor, True

    def store(self, lin, basis: int, tensor: torch.Tensor) -> bool:
        """Offer a freshly materialized full-width shadow to the cache. Evicts LRU entries to fit;
        declines (returns False, caller's tensor is simply not retained) when the single tensor
        exceeds the whole budget."""
        if not self.enabled():
            return False
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self._budget:
            return False
        key = self._key(lin, basis)
        old = self._entries.get(key)
        if old is not None:
            self._total -= old.bytes
            del self._entries[key]
        self._evict_to(self._budget - nbytes)
        device_index = lin.trellis.device.index if lin.trellis.device.type == "cuda" else -1
        self._entries[key] = _ShadowEntry(
            tensor, lin.trellis, lin.in_features, lin.out_features, device_index, int(basis), lin.key)
        self._total += nbytes
        return True

    # -- invalidation -------------------------------------------------------------------------------------

    def drop_linear(self, lin) -> int:
        """Drop every shadow belonging to this layer's trellis (LinearEXL3.unload)."""
        dropped = 0
        for key in [k for k, e in self._entries.items() if e.trellis is lin.trellis]:
            self._total -= self._entries.pop(key).bytes
            dropped += 1
        return dropped

    def clear(self) -> int:
        """Drop everything (model.unload); returns freed bytes."""
        freed = self._total
        self._entries.clear()
        self._total = 0
        return freed


recon_cache = ReconCache()
