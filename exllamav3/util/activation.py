from __future__ import annotations
import torch
import torch.nn.functional as F
from ..ext import exllamav3_ext as ext

# act(x) * y -> z for the fused MLP/MoE epilogues; z is half as in the kernels. The ext calls
# move to the torch formulations whenever the tensors don't live on CUDA

def _clamped(r, y, act_limit):
    if act_limit:
        r = r.clamp(max = act_limit)
        y = y.clamp(-act_limit, act_limit)
    return r * y

def silu_mul(x, y, z, act_limit = 0.0):
    if x.device.type == "cuda":
        ext.silu_mul(x, y, z, act_limit)
        return
    z.copy_(_clamped(F.silu(x), y, act_limit))

def gelu_mul(x, y, z, act_limit = 0.0):
    if x.device.type == "cuda":
        ext.gelu_mul(x, y, z, act_limit)
        return
    z.copy_(_clamped(F.gelu(x, approximate = "tanh"), y, act_limit))

def relu2_mul(x, y, z, act_limit = 0.0):
    if x.device.type == "cuda":
        ext.relu2_mul(x, y, z, act_limit)
        return
    z.copy_(_clamped(torch.square(F.relu(x)), y, act_limit))

def relu_mul(x, y, z, act_limit = 0.0):
    if x.device.type == "cuda":
        ext.relu_mul(x, y, z, act_limit)
        return
    z.copy_(_clamped(F.relu(x), y, act_limit))

def silu_oai_mul(x, y, z, act_limit = 0.0):
    if x.device.type == "cuda":
        ext.silu_oai_mul(x, y, z, act_limit)
        return
    g = x.float()
    u = y.float()
    if act_limit:
        g = g.clamp(max = act_limit)
        u = u.clamp(-act_limit, act_limit)
    z.copy_((u + 1.0) * g * torch.sigmoid(1.702 * g))
