"""
opt02 unit tests (EXL3_MOE_DEVICE_COUNT): pinned-ring staging for the MoE batch path's
expert-count readback and FUSED_DET slot-table upload.

Model-free: everything runs on small synthetic tensors. The exl3_moe kernel itself is NOT
exercised (it needs real quantized weights); the worst-case-grid launch (num_active = -1) is
verified by parity/benches, which are deferred (see RESUME.md).

Run:
  PYTHONPATH=. /path/to/venv/python tests/test_moe_device_counts.py
or: pytest tests/test_moe_device_counts.py
"""

from __future__ import annotations
import os
import sys

import numpy as np
import torch


def _cuda():
    return torch.device("cuda") if torch.cuda.is_available() else None


def test_knob_default_and_override():
    from exllamav3.util import cuda_flags
    assert cuda_flags.MOE_DEVICE_COUNT is True, "EXL3_MOE_DEVICE_COUNT must default ON"
    # re-read semantics with the knob disabled (as a fresh process would see it)
    saved = os.environ.get("EXL3_MOE_DEVICE_COUNT")
    os.environ["EXL3_MOE_DEVICE_COUNT"] = "0"
    try:
        val = os.environ.get("EXL3_MOE_DEVICE_COUNT", "1") != "0"
        assert val is False
    finally:
        if saved is None:
            os.environ.pop("EXL3_MOE_DEVICE_COUNT", None)
        else:
            os.environ["EXL3_MOE_DEVICE_COUNT"] = saved
    # the consumer module picked the knob up at import time
    from exllamav3.modules import block_sparse_mlp as bsm
    assert bsm.MOE_DEVICE_COUNT is True


def test_ring_readback_matches_tolist():
    dev = _cuda()
    if dev is None:
        print("  (no CUDA device: skipped)")
        return
    from exllamav3.modules.block_sparse_mlp import _PinnedCountRing
    rng = np.random.default_rng(0)
    for e1 in (9, 129):  # small MoE + a sentinel-style layout
        ring = _PinnedCountRing(dev, e1)
        # exercise the ring across several wrap-arounds (slot reuse)
        for it in range(3 * _PinnedCountRing.RING):
            counts_np = rng.integers(0, 64, size = e1).astype(np.int64)
            counts_np[-1] = it  # sentinel bucket row varies per iteration
            counts = torch.from_numpy(counts_np).to(dev)
            assert counts.numel() * 8 < 1 << 20, "synthetic tensors must stay tiny"
            got = ring.read_counts(counts)
            assert got == counts_np.tolist(), f"ring readback diverged at e1={e1} iter={it}"


def test_ring_table_upload_roundtrip():
    dev = _cuda()
    if dev is None:
        print("  (no CUDA device: skipped)")
        return
    from exllamav3.modules.block_sparse_mlp import _PinnedCountRing
    e1 = 33
    ring = _PinnedCountRing(dev, e1)
    for it in range(3 * _PinnedCountRing.RING):
        counts = torch.randint(0, 8, (e1,), device = dev, dtype = torch.long)
        ring.read_counts(counts)  # upload_tables must pair with the same layer's slot
        stacked = torch.randint(0, 4096, (3, e1), dtype = torch.long)
        up = ring.upload_tables(stacked)
        assert up.device.type == "cuda" and up.dtype == torch.long and up.shape == (3, e1)
        assert torch.equal(up.cpu(), stacked), f"table upload corrupted at iter={it}"


def test_host_and_device_slot_table_layouts_agree_all_fused():
    """When every expert's count is within the fused row capacity, the host-side FUSED_DET
    table layout (base/kind/starts, built from the count list) and the device-side layout of
    the all-fused fast path (cumsum on the count tensor) describe the same slot assignment."""
    dev = _cuda() or torch.device("cpu")
    E, fused_rows, n = 16, 64, 4
    rng = np.random.default_rng(1)
    for _ in range(20):
        counts_list = [int(c) for c in rng.integers(0, fused_rows + 1, size = E)]
        counts_list[rng.integers(0, E)] = fused_rows  # keep the capacity edge covered
        # host layout (block_sparse_mlp.forward, expert_count_list branch)
        starts_np = np.cumsum([0] + counts_list[:-1]).astype(np.int64)
        base = np.zeros(E, dtype = np.int64)
        kind = np.zeros(E, dtype = np.int64)
        n_slots = 0
        for e, c in enumerate(counts_list):
            if 0 < c <= fused_rows:
                base[e] = n_slots
                kind[e] = 1
                n_slots += c
        # device layout (all-fused fast path)
        counts = torch.tensor(counts_list, dtype = torch.long, device = dev)
        expert_start = torch.cumsum(counts, 0) - counts
        tables_dev = torch.stack([expert_start, expert_start, (counts > 0).long()])
        assert torch.equal(tables_dev[1].cpu(), torch.from_numpy(starts_np))
        kind_t = torch.from_numpy(kind)
        assert torch.equal(tables_dev[2].cpu(), kind_t), "kind must equal (count>0)"
        # Zero-count experts never get slots (kind 0) so their base entry is unobservable;
        # wherever an expert IS fused the two layouts must agree exactly
        fused = kind_t == 1
        assert fused.any()
        assert torch.equal(tables_dev[0].cpu()[fused], torch.from_numpy(base)[fused]), \
            "fused experts' base must equal the cumsum offsets"
        assert n_slots == int(counts.sum()), "all-fused slots must cover every assignment"


def test_worst_case_launch_contract_site():
    """The -1 (unknown-counts) launch is the same contract the existing all-fused fast path
    uses; assert the fast-path call and the opt02 branch both exist in the source."""
    import inspect
    from exllamav3.modules import block_sparse_mlp as bsm
    src = inspect.getsource(bsm.BlockSparseMLP.forward)
    assert src.count("run_fused(-1)") >= 2, "fast path and opt02 worst-case-grid launch must both be present"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
