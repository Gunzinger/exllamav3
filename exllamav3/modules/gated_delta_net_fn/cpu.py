import torch
import torch.nn.functional as F

# CPU twins of the GDN CUDA/Triton ops (gdn.cu fused gate kernels, causal_conv1d_update) for
# hosts without a GPU toolchain. Formulas mirror the kernels exactly; see gdn.cu
# (gated_delta_net_fused_op_kernel:147-155, gated_delta_net_fused_op_2_kernel:264-266) and
# modules/gated_delta_net_fn/conv1d.py (_causal_conv1d_update_slotted_kernel)


def causal_conv1d_update_cpu(
    x: torch.Tensor,
    conv_state: torch.Tensor | None,
    slots: torch.Tensor | None,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    history: bool = False,
    token_major: bool = False,
) -> torch.Tensor:
    """
    Depthwise causal conv + SiLU over [conv_state prefix | x], per slot, in fp32 like the
    Triton kernel. x is (bsz, dim, seqlen) or token-major (bsz, seqlen, dim) in any float
    dtype; returns (bsz, seqlen, dim) bf16. Without history the state keeps the last
    conv_kernel_size inputs at the front; with history it becomes the right-aligned rolling
    buffer the rewind machinery reads (GDNLayerState.rewind)
    """
    if token_major:
        bsz, seqlen, dim = x.shape
        x = x.transpose(1, 2)
    else:
        bsz, dim, seqlen = x.shape
    cdim = conv1d_weight.shape[-1]
    state_size = conv_state.shape[-1] if conv_state is not None else cdim

    if conv_state is not None:
        idx = slots.long() if slots is not None else torch.arange(bsz)
        y = torch.cat([conv_state[idx, :, :cdim], x], dim = -1).float()
        if history:
            write_size = min(state_size, cdim + seqlen)
            conv_state[idx, :, -write_size:] = y[:, :, -write_size:].to(conv_state.dtype)
        else:
            conv_state[idx, :, :cdim] = y[:, :, -cdim:].to(conv_state.dtype)
    else:
        y = torch.cat([x.new_zeros(bsz, dim, cdim), x], dim = -1).float()

    y = F.conv1d(
        y,
        conv1d_weight.float().unsqueeze(1),
        conv1d_bias.float() if conv1d_bias is not None else None,
        groups = dim,
    )
    # First output position sees only zero-padded history, matching the kernel's src_t = s+k+1
    y = F.silu(y[:, :, 1:]).to(torch.bfloat16)
    return y.transpose(1, 2)


def gdn_fused_op_2_cpu(
    b: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    beta_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Twin of ext.gated_delta_net_fused_op_2 (split b/a projections): beta = sigmoid(b) *
    beta_scale (bf16), g = -softplus(a + dt_bias) * exp(a_log) (fp32)
    """
    beta = (torch.sigmoid(b.float()) * beta_scale).to(torch.bfloat16)
    g = -F.softplus(a.float() + dt_bias.float(), threshold = 20.0) * torch.exp(a_log.float())
    return beta, g


def gdn_fused_op_cpu(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
    beta_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Twin of ext.gated_delta_net_fused_op (fused qkvz projection, Qwen3-Next layout): unpacks
    the per-k-head [q | k | v(ng) | z(ng)] segments into the channel-major mixed_qkv
    (bsz, 2*nk*hk + nv*hv, seqlen) bf16 and z (bsz, seqlen, nv, hv) bf16, plus the beta/g
    gates from the packed b/a segments
    """
    bsz, seqlen, _ = qkvz.shape
    hk = k_head_dim
    hv = v_head_dim
    ng = num_v_heads // num_k_heads
    fseg = 2 * hk + 2 * ng * hv

    qkvz = qkvz.view(bsz, seqlen, num_k_heads, fseg)
    ba = ba.view(bsz, seqlen, num_k_heads, 2 * ng)

    q = qkvz[..., :hk].reshape(bsz, seqlen, -1)
    k = qkvz[..., hk:2 * hk].reshape(bsz, seqlen, -1)
    v = qkvz[..., 2 * hk:2 * hk + ng * hv].reshape(bsz, seqlen, -1)
    z = qkvz[..., 2 * hk + ng * hv:].reshape(bsz, seqlen, num_v_heads, hv)
    b = ba[..., :ng].reshape(bsz, seqlen, -1)
    a = ba[..., ng:].reshape(bsz, seqlen, -1)

    mixed_qkv = torch.cat([q, k, v], dim = -1).transpose(1, 2).to(torch.bfloat16).contiguous()
    beta, g = gdn_fused_op_2_cpu(b, a, dt_bias, a_log, beta_scale)
    return mixed_qkv, z.to(torch.bfloat16), beta, g
