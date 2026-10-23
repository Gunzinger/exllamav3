# Unit test for the batched GDN/KDA prefill chunk scan (EXL3_GDN_SCAN_BATCHED, plan 06).
# Compares the batched path vs the original per-row python loop on synthetic shapes: outputs AND
# final recurrent state (max-abs <= 1e-3 for the fp32 state, matching the plan's tolerance).
# Run:
#   PYTHONPATH=$PWD /home/daniel/development/gunzi/local-llm/.venv/bin/python test_gdn_rowbatch.py
import os
import sys

import torch

from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdr

DEVICE = "cuda"
TOL_STATE = 1e-3   # fp32 recurrent state class
TOL_OUT = 5e-3     # bf16 output class


def run_case(bsz, T, HK, HV, K, V, pool_dtype, channelwise_g, history, slots, sub_chunk):
    torch.manual_seed(bsz * 100000 + T * 100 + HK + (17 if channelwise_g else 0))
    k_dim, v_dim = HK * K, HV * V

    def make_inputs():
        mixed_qkv = torch.randn(bsz, T, k_dim + k_dim + v_dim, device = DEVICE, dtype = torch.bfloat16) * 0.5
        beta = torch.rand(bsz, T, HV, device = DEVICE, dtype = torch.bfloat16) * 0.9 + 0.05
        if channelwise_g:
            g = -(torch.rand(bsz, T, HV, K, device = DEVICE, dtype = torch.float32) * 0.1)
        else:
            g = -(torch.rand(bsz, T, HV, device = DEVICE, dtype = torch.float32) * 0.1)
        return mixed_qkv, beta, g

    # one fixed input set shared by both paths
    inputs = make_inputs()

    def run(pool, rec_slots):
        mixed_qkv, beta, g = inputs
        return gdr.gated_delta_rule_fn(
            mixed_qkv, beta, g, pool, rec_slots,
            history = history, save_state = True,
            num_k_heads = HK, num_v_heads = HV, k_dim = k_dim, v_dim = v_dim,
            k_head_dim = K, v_head_dim = V,
            channelwise_g = channelwise_g,
        )

    nslots = max(slots) + 1
    pool_old = torch.randn(nslots, 1, HV, K, V, device = DEVICE, dtype = pool_dtype)
    pool_ref = pool_old.clone()

    old = gdr._scan_batched
    try:
        gdr._scan_batched = False
        out_ref = run(pool_ref, torch.tensor(slots, device = DEVICE))
        state_ref = pool_ref.clone()
        gdr._scan_batched = True
        out_new = run(pool_old, torch.tensor(slots, device = DEVICE))
        state_new = pool_old.clone()
    finally:
        gdr._scan_batched = old

    state_ref = state_ref[torch.tensor(slots, device = DEVICE), 0].float()
    state_new = state_new[torch.tensor(slots, device = DEVICE), 0].float()
    e_out = (out_ref.float() - out_new.float()).abs().max().item()
    e_state = (state_ref - state_new).abs().max().item()
    tag = (f"bsz={bsz} T={T} HK={HK} HV={HV} K={K} V={V} pool={pool_dtype} "
           f"{'KDA' if channelwise_g else 'GDN'} history={history} slots={slots} sub={sub_chunk}")
    assert e_out <= TOL_OUT, f"OUTPUT mismatch {e_out:.2e} > {TOL_OUT}: {tag}"
    assert e_state <= TOL_STATE, f"STATE mismatch {e_state:.2e} > {TOL_STATE}: {tag}"
    print(f"ok  {tag}  out_err={e_out:.2e} state_err={e_state:.2e}")


def main():
    assert torch.cuda.is_available(), "needs CUDA for the Triton fla kernels"
    cases = []
    for bsz in (1, 4):
        for T in (128, 256):
            for sub in (0, 64):  # 0 = default whole-pass, 64 = multi sub-range (EXL3_GDN_SUB_CHUNK)
                cases.append(dict(bsz = bsz, T = T, sub_chunk = sub))
    for c in cases:
        old_sub = gdr._scan_sub_range
        try:
            if c["sub_chunk"]:
                gdr._scan_sub_range = c["sub_chunk"]
            for pool_dtype in (torch.float32, torch.float16):
                run_case(HK = 2, HV = 4, K = 64, V = 64, pool_dtype = pool_dtype,
                         channelwise_g = False, history = False, slots = list(range(c["bsz"])), **c)
            # permuted slot rows, non-history (batched path with a non-arange slot list)
            if c["bsz"] > 1:
                run_case(HK = 2, HV = 4, K = 64, V = 64, pool_dtype = torch.float32,
                         channelwise_g = False, history = False,
                         slots = [2, 0, 3, 1][:c["bsz"]], **c)
            # KDA variant
            run_case(HK = 2, HV = 4, K = 64, V = 64, pool_dtype = torch.float32,
                     channelwise_g = True, history = False, slots = list(range(c["bsz"])), **c)
        finally:
            gdr._scan_sub_range = old_sub
    # history=True must produce identical results either way (both take the sequential loop)
    for c in cases[:2]:
        run_case(HK = 2, HV = 4, K = 64, V = 64, pool_dtype = torch.float32,
                 channelwise_g = False, history = True, slots = list(range(c["bsz"])), **c)
    print("ALL PASS")


if __name__ == "__main__":
    main()
