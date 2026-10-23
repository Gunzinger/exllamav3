import os
import torch
from ...ext import exllamav3_ext as ext
from ...util.tensor import get_for_device, buffered_arange

# The vendored fla chunk kernels (exllamav3.vendor.fla) query the devices and Triton at import, so
# they are imported on first use rather than with the library


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + 1e-6)


def torch_recurrent_gated_delta_rule_cpu(
    mixed_qkv: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor | None,
    recurrent_slots: torch.Tensor | None,
    history: bool,
    save_state: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_dim: int,
    v_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    slots_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sequential grouped delta rule on CPU tensors, mirroring ext.cuda_recurrent_gated_delta_rule
    (gdn.cu: cuda_recurrent_gated_delta_rule_kernel: in-kernel q/k l2norm, per-v-head beta/g
    over the shared k-head's state, intermediate states into the history slots, output bf16)

    slots_in: base rows a speculative pass (history) reads its initial state from, writing the
    final state to the scratch rows in recurrent_slots; no intermediate states are stored (the
    rewind replays the staged scan inputs), matching the CUDA kernel's slots_in semantics
    """
    bsz, seqlen, _ = mixed_qkv.shape
    group = num_v_heads // num_k_heads
    scale = k_head_dim ** -0.5

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = _l2norm(q.float().view(bsz, seqlen, num_k_heads, k_head_dim)).repeat_interleave(group, dim = 2)
    k = _l2norm(k.float().view(bsz, seqlen, num_k_heads, k_head_dim)).repeat_interleave(group, dim = 2)
    v = v.view(bsz, seqlen, num_v_heads, v_head_dim).float()
    decay = g.float().exp()
    beta = beta.float()

    if recurrent_state is None:
        state = torch.zeros(bsz, num_v_heads, k_head_dim, v_head_dim, dtype = torch.float)
        idx = None
    else:
        idx = recurrent_slots.long() if recurrent_slots is not None else None
        in_idx = slots_in.long() if slots_in is not None else idx
        # .float(): the step math is fp32; the state pool may be fp16 (recurrent_dtype default since
        # 02446529) and is written back through the implicit cast of the index assignments below
        state = (recurrent_state[in_idx, 0] if in_idx is not None else recurrent_state[:bsz, 0]).clone().float()

    # The step math is expressed as batched GEMMs over (bsz * nv, dk, dv): the broadcast
    # elementwise form moves the fp32 state through ~5 full-size temporaries per token
    state3 = state.view(bsz * num_v_heads, k_head_dim, v_head_dim)
    out = torch.empty(bsz, seqlen, num_v_heads, v_head_dim, dtype = torch.bfloat16)
    for t in range(seqlen):
        kt = k[:, t].unsqueeze(-1)
        kv_mem = torch.matmul(kt.transpose(-1, -2), state)
        u = (v[:, t].unsqueeze(-2) - kv_mem * decay[:, t, :, None, None]) * beta[:, t, :, None, None]
        state3.mul_(decay[:, t].reshape(-1, 1, 1))
        state3.baddbmm_(kt.reshape(-1, k_head_dim, 1), u.reshape(-1, 1, v_head_dim))
        out[:, t] = (torch.matmul(q[:, t].unsqueeze(-2), state).squeeze(-2) * scale).to(torch.bfloat16)
        if history and slots_in is None and t < seqlen - 1 and recurrent_state is not None:
            if idx is not None:
                recurrent_state[idx, t + 1] = state.to(recurrent_state.dtype)
            else:
                recurrent_state[:bsz, t + 1] = state.to(recurrent_state.dtype)

    if save_state and recurrent_state is not None:
        if idx is not None:
            recurrent_state[idx, 0] = state.to(recurrent_state.dtype)
        else:
            recurrent_state[:bsz, 0] = state.to(recurrent_state.dtype)
    return out


def torch_chunk_gated_delta_rule_cpu(
    mixed_qkv: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor | None,
    recurrent_slots: torch.Tensor | None,
    save_state: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_dim: int,
    v_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunked delta rule on CPU tensors: WY form of the sequential recurrence, chunk-local decay
    cumsum G, A = tril(beta_i (k_i . k_j) e^{G_i - G_j}), state carried in fp32. Verified
    against torch_recurrent_gated_delta_rule (the decode-kernel semantics) to fp64 precision;
    the h-readback rows of w carry the e^G decay to the token's position, which is where the
    naive transcription of the fla kernels (w = (I+A)^-1 beta k) drifts ~5e-2
    """
    bsz, seqlen, _ = mixed_qkv.shape
    group = num_v_heads // num_k_heads
    scale = k_head_dim ** -0.5
    C = chunk_size

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = _l2norm(q.float().view(bsz, seqlen, num_k_heads, k_head_dim)).repeat_interleave(group, dim = 2)
    k = _l2norm(k.float().view(bsz, seqlen, num_k_heads, k_head_dim)).repeat_interleave(group, dim = 2)
    v = v.view(bsz, seqlen, num_v_heads, v_head_dim).float()
    beta = beta.float()
    g = g.float()

    out = torch.empty(bsz, seqlen, num_v_heads, v_head_dim, dtype = torch.bfloat16)

    for bi in range(bsz):
        slot = int(recurrent_slots[bi]) if recurrent_slots is not None else bi
        # .float(): fp32 chunk math against a possibly-fp16 state pool (recurrent_dtype default
        # since 02446529); the store at the end casts back through the index assignment
        h = recurrent_state[slot, 0].clone().float() if recurrent_state is not None \
            else torch.zeros(num_v_heads, k_head_dim, v_head_dim, dtype = torch.float)
        beta_b = beta[bi]
        g_b = g[bi]
        for t0 in range(0, seqlen, C):
            t1 = min(t0 + C, seqlen)
            c = t1 - t0
            qc = q[bi, t0:t1].transpose(0, 1)
            kc = k[bi, t0:t1].transpose(0, 1)
            vc = v[bi, t0:t1].transpose(0, 1)
            bc = beta_b[t0:t1].transpose(0, 1)
            G = g_b[t0:t1].transpose(0, 1).cumsum(dim = -1)
            D = G.unsqueeze(-1) - G.unsqueeze(-2)
            eye = torch.eye(c).expand(num_v_heads, c, c)

            strict = torch.ones(c, c, dtype = torch.bool).tril(-1)
            A = kc @ kc.transpose(-1, -2)
            A = A * D.masked_fill(~strict, float("-inf")).exp() * bc.unsqueeze(-1)
            t_inv = torch.linalg.solve_triangular(eye + A, eye, upper = False)

            w = t_inv @ ((bc * G.exp()).unsqueeze(-1) * kc)
            u = t_inv @ (bc.unsqueeze(-1) * vc)
            v_new = u - w @ h

            G_last = G[:, -1]
            incl = torch.ones(c, c, dtype = torch.bool).tril(0)
            o = (qc @ h) * G.exp()[:, :, None]
            o = o + (qc @ kc.transpose(-1, -2) *
                     D.masked_fill(~incl, float("-inf")).exp()) @ v_new
            out[bi, t0:t1] = (o * scale).transpose(0, 1).to(torch.bfloat16)

            h = h * G_last.exp()[:, None, None] + \
                kc.transpose(-1, -2) @ (v_new * (G_last[:, None] - G).exp()[:, :, None])
        if save_state and recurrent_state is not None:
            recurrent_state[slot, 0] = h.to(recurrent_state.dtype)
    return out

def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    """
    For reference, not used
    """

    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = 1 / torch.sqrt(
            (x * x).sum(dim = dim, keepdim = True)
            + eps
        )
        return x * inv_norm

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape

    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()

    for i in range(sequence_length):
        q_t = query[:, i, :]
        k_t = key[:, i, :]
        v_t = value[:, i, :]
        g_t = g[:, i, :].exp().unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)
        kv_mem = last_recurrent_state * k_t.unsqueeze(-1)
        kv_mem = kv_mem.sum(dim = -2)
        v_t = v_t - kv_mem * g_t
        upd = k_t.unsqueeze(-1) * v_t.unsqueeze(-2) * beta_t.unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t.unsqueeze(-1) + upd
        core_attn_out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2) * scale

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


def torch_recurrent_kda(q, k, v, g, beta, state):
    """Sequential per-channel-decay (KDA) delta rule, fp32, mirroring the GLM5.3/Kimi
    reference. q/k/v: (b, s, h, d); g: (b, s, h, d) log-decay per k-channel; beta: (b, s, h);
    state: (b, h, dk, dv), updated in place. Returns (b, s, h, dv) bf16."""
    b, s, h, dk = k.shape
    dv = v.shape[-1]
    q, k, v, g, beta = [t.float() for t in (q, k, v, g, beta)]

    def l2norm(x):
        return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + 1e-6)
    q = l2norm(q) * (dk ** -0.5)
    k = l2norm(k)

    out = torch.empty((b, s, h, dv), dtype = torch.float, device = v.device)
    for i in range(s):
        g_i = g[:, i].exp().unsqueeze(-1)                       # (b, h, dk, 1)
        state *= g_i
        kv_mem = (state * k[:, i].unsqueeze(-1)).sum(dim = -2)  # (b, h, dv)
        delta = (v[:, i] - kv_mem) * beta[:, i].unsqueeze(-1)
        state += k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q[:, i].unsqueeze(-1)).sum(dim = -2)
    return out.to(torch.bfloat16)


# Prefill sub-range for the fla chunk kernels. The chunked scan is run over consecutive
# sub-ranges of the sequence with the fp32 state carried between them, which is bit-identical
# to one pass over the whole range: the kernels' T-shaped temporaries (w, u, A, the per-64-token
# chunk states h, v_new, o: ~112 KB/token at 48 v-heads x 128) then scale with the sub-range
# instead of the chunk. 2048 is also the fastest point (-18% vs one 16k pass, -12% vs one 8k
# pass on a 5090: better program counts than a long pass, and 1024 already costs +38%)
_scan_sub_range = int(os.environ.get("EXL3_GDN_SUB_CHUNK", "2048"))

# Prefill chunk scan over all batch rows in one (B, T, H, D) fla pass instead of a per-row python
# loop. Rows are independent given their initial states, so this is exact whenever no row reads a
# slot another row of the same pass writes -- always true when rows_in == rows_out (every
# non-speculative pass; each row then only ever touches its own slot) and when the state pool is
# None. 0 restores the per-row loop everywhere.
_scan_batched = os.environ.get("EXL3_GDN_SCAN_BATCHED", "1") != "0"


def _slot_index(rows: list[int], device: torch.device) -> torch.Tensor:
    # Index tensor for a slot row list; the default 0..bsz-1 list avoids the per-call H2D copy
    if rows == list(range(len(rows))):
        return buffered_arange(len(rows), device)
    return torch.tensor(rows, dtype = torch.long, device = device)


def _chunked_scan(fn, q, k, v, g, beta, state, save_state):
    """
    fn (chunk_gated_delta_rule / chunk_kda) over one batch row in sub-ranges of the sequence.
    q, k: (1, T, H, K); v: (1, T, HV, V); state: (1, HV, K, V) fp32 or None. Returns the
    concatenated output (1, T, HV, V) in fn's output dtype and the final state (None unless
    save_state)
    """
    T = q.shape[1]
    sub = _scan_sub_range if _scan_sub_range > 0 else T
    if T <= sub:
        return fn(q, k, v, g = g, beta = beta, initial_state = state, output_final_state = save_state,
                  use_qk_l2norm_in_kernel = True)
    out = None
    for t0 in range(0, T, sub):
        t1 = min(t0 + sub, T)
        o, state = fn(q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g = g[:, t0:t1], beta = beta[:, t0:t1],
                      initial_state = state, output_final_state = True, use_qk_l2norm_in_kernel = True)
        if out is None:
            out = torch.empty((q.shape[0], T) + o.shape[2:], dtype = o.dtype, device = o.device)
        out[:, t0:t1] = o
    return out, state if save_state else None


def _chunked_scan_rows(fn, q, k, v, g, beta, recurrent_state, rows_out, rows_in, history, save_state):
    """
    fn (chunk_gated_delta_rule / chunk_kda) over every batch row. q, k: (B, T, H, K); v:
    (B, T, HV, V); recurrent_state: (S, 1, HV, K, V) pool or None; row i starts from pool row
    rows_in[i] and its final state is written back to pool row rows_out[i] (save_state). Returns
    the outputs concatenated on the batch dim, (B, T, HV, V).

    One batched kernel pass when no row can read a slot another row of the same pass writes: holds
    whenever rows_in == rows_out and not history (each row only touches its own slot), or when
    recurrent_state is None. The speculative pass (history, rows_in != rows_out) keeps the original
    sequential per-row loop; EXL3_GDN_SCAN_BATCHED=0 forces it everywhere.
    """
    if not _scan_batched or (recurrent_state is not None and (history or rows_in != rows_out)):
        core_attn_out = []
        for i, (so, si) in enumerate(zip(rows_out, rows_in)):
            # .float(): the chunk kernels take and return fp32; an fp16 pool is converted on the way in and
            # by the copy_ on the way out
            state = recurrent_state[si, 0].float().unsqueeze(0) if recurrent_state is not None else None
            core_attn, new_state = _chunked_scan(
                fn, q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], beta[i:i + 1],
                state, save_state,
            )
            if save_state and state is not None:
                recurrent_state[so, 0].copy_(new_state.squeeze(0))
            core_attn_out.append(core_attn)
        return torch.cat(core_attn_out, dim = 0)

    state = None
    if recurrent_state is not None:
        # Advanced indexing copies, so the scan never aliases the pool; .float() is a no-op on an
        # fp32 pool and the write-back casts back to the pool dtype
        state = recurrent_state[_slot_index(rows_in, recurrent_state.device), 0].float()
    core_attn, new_state = _chunked_scan(fn, q, k, v, g, beta, state, save_state)
    if save_state and state is not None:
        # index put does not cast: write back in the pool dtype (fp32 pool: no-op cast)
        recurrent_state[_slot_index(rows_out, recurrent_state.device), 0] = new_state.to(recurrent_state.dtype)
    return core_attn


def gated_delta_rule_fn(
    mixed_qkv: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_slots: torch.Tensor,
    history: bool,
    save_state: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_dim: int,
    v_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    params: dict = None,
    channelwise_g: bool = False,
    slots_in: torch.Tensor | None = None,
):
    if params is None:
        params = {}

    bsz, seqlen, _ = mixed_qkv.shape


    # CPU paths: the chunked scan for prefill (same shape split as the fla kernels below), the
    # sequential recurrence -- the decode math -- otherwise. KDA (channelwise decay) stays on
    # the CUDA kernels
    if mixed_qkv.device.type == "cpu" and not channelwise_g:
        if seqlen >= num_v_heads and not history:
            return torch_chunk_gated_delta_rule_cpu(
                mixed_qkv, beta, g, recurrent_state, recurrent_slots, save_state,
                num_k_heads, num_v_heads, k_dim, v_dim, k_head_dim, v_head_dim,
            )
        return torch_recurrent_gated_delta_rule_cpu(
            mixed_qkv, beta, g, recurrent_state, recurrent_slots, history, save_state,
            num_k_heads, num_v_heads, k_dim, v_dim, k_head_dim, v_head_dim,
            slots_in,
        )

    # KDA (per-k-channel decay, g shaped (b, s, h, dk)): fla chunk kernel for prefill, the
    # channelwise CUDA recurrent kernel (in-kernel q/k l2norm, history-capable) otherwise


    # recurrent_state rows: recurrent_slots are the rows the pass writes (and reads, except that
    # a speculative pass -- history -- reads each row's initial state from slots_in and leaves it
    # untouched; see GDNState). The chunk kernels take the initial and final state as separate
    # tensors, so they serve both cases; the CUDA kernel takes slots_in
    def chunked_rows():
        rows_out = get_for_device(params, "recurrent_slots_scan", "cpu", None)
        if rows_out is None:
            rows_out = buffered_arange(bsz, mixed_qkv.device)
        rows_in = get_for_device(params, "recurrent_slots_scan_in", "cpu", None) if history else None
        rows_out = rows_out.tolist()
        rows_in = rows_in.tolist() if rows_in is not None else rows_out
        return rows_out, rows_in

    # KDA (per-k-channel decay, g shaped (b, s, h, dk)): fla chunk kernel for long passes, the
    # channelwise CUDA recurrent kernel (in-kernel q/k l2norm) otherwise
    if channelwise_g:
        if seqlen >= num_v_heads:
            from ...vendor.fla import chunk_kda
            q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
            q = q.view(bsz, seqlen, -1, k_head_dim)
            k = k.view(bsz, seqlen, -1, k_head_dim)
            v = v.view(bsz, seqlen, -1, v_head_dim)

            rows_out, rows_in = chunked_rows()
            core_attn_out = _chunked_scan_rows(
                chunk_kda, q, k, v, g, beta,
                recurrent_state, rows_out, rows_in, history, save_state,
            )
            return core_attn_out.to(torch.bfloat16)

        core_attn_out = torch.empty(
            (bsz, seqlen, num_v_heads, v_head_dim),
            dtype = torch.bfloat16,
            device = mixed_qkv.device,
        )
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (bsz, 1, num_v_heads, k_head_dim, v_head_dim),
                dtype = torch.float,
                device = mixed_qkv.device
            )
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            recurrent_state,
            core_attn_out,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            recurrent_slots,
            history,
            slots_in,
        )
        return core_attn_out

    # Chunked rule
    if seqlen >= num_v_heads:
        from ...vendor.fla import chunk_gated_delta_rule

        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
        q = q.view(bsz, seqlen, -1, k_head_dim)
        k = k.view(bsz, seqlen, -1, k_head_dim)
        v = v.view(bsz, seqlen, -1, v_head_dim)

        # (Grouped attn supported in fla-core now)
        rows_out, rows_in = chunked_rows()
        core_attn_out = _chunked_scan_rows(
            chunk_gated_delta_rule, q, k, v, g, beta,
            recurrent_state, rows_out, rows_in, history, save_state,
        )
        return core_attn_out

    # Fused recurrent rule
    else:
        core_attn_out = torch.empty(
            (bsz, seqlen, num_v_heads, v_head_dim),
            dtype = torch.bfloat16,
            device = mixed_qkv.device,
        )

        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (bsz, 1, num_v_heads, k_head_dim, v_head_dim),
                dtype = torch.float,
                device = mixed_qkv.device
            )

        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv,
            g,
            beta,
            recurrent_state,
            core_attn_out,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            recurrent_slots,
            history,
            slots_in,
        )

    return core_attn_out
