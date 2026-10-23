from __future__ import annotations
import torch
from ...model.config import Config
from .exl3_lib.quantize import frac_k, preapply_had_l, preapply_had_r, had_k, had_n
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
import os
from ...util import profile_opt

AUTO_RECONSTRUCT_THRESHOLD = 144
MAX_RECONSTRUCT_SLICE_N = 32768
RECONSTRUCT_SLICE_GRANULARITY_N = 128

no_fused_reconstruct = os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0"

# Packed ("swizzled") trellis layout for the CPU mul1 kernels, same rule everywhere
# (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout; EXL3_MOE_CPU_SWIZZLE=0 rolls back):
# tile (kt, nt) moves to (nt/group, kt, nt%group), and with planar each tile's dwords are
# transposed (dword w to 8*(w%bits) + w/bits)
_cpu_swizzle = os.environ.get("EXL3_MOE_CPU_SWIZZLE", "1") != "0"

_mallopt_done = False


def _swizzle_mallopt():
    # The repack doubles each trellis tensor transiently; the freed native copy must go back to
    # the OS, not idle in the heap until the next same-size allocation (glibc's dynamic mmap
    # threshold rides every large free upward and retains GBs over a full-expert load). One
    # explicit mallopt freezes the threshold at its default: every >= 128 KiB block is mmap'd
    # and unmapped on free, so peak RSS stays ~one tensor above the resident set.
    # Applied lazily at the first CPU swizzle repack so a process that never touches the CPU
    # path (e.g. a pure GPU build) never gets its allocator reconfigured.
    global _mallopt_done
    if _mallopt_done or not (_cpu_swizzle and os.name == "posix"):
        return
    _mallopt_done = True
    import ctypes
    ctypes.CDLL(None).mallopt(-3, 128 << 10)   # M_MMAP_THRESHOLD

# 2 MiB pages for the CPU-resident trellis (TLB reach on the GEMV streams); best effort
_cpu_thp = os.name == "posix" and os.environ.get("EXL3_CPU_THP", "1") != "0"


# -------------------------------------------------------------------------------------------
# CPU prefill tier (Plan 07 Option 1, opt-in): at M >= EXL3_CPU_PREFILL_MIN_M the dense linear
# dequantizes its trellis once to a BF16 buffer (ext.exl3_cpu_dequant_bf16) and takes
# torch.matmul (oneDNN) instead of the decode-shape mul1 GEMV. Output dtype contract matches
# the mul1 path (fp32 accumulation + fp32 bias add, single rounding to the caller's dtype);
# values differ from the old path by the bf16 weight rounding where the old path instead has
# int8 activation-quantization noise — tolerance/argmax acceptance, PLAN-cpu-bf16-prefill.md.
# Env knobs live in util/cuda_flags.py (EXL3_CPU_PREFILL_TIER / _MIN_M / _CACHE_BYTES); "off"
# never reaches this helper, so the stock path stays byte-identical.
_cpu_bf16_cache = None   # weakref-keyed on the LinearEXL3: buffers die with their layer, and a
                         # cached entry can never outlive (and be reused for) another matrix


def cpu_prefill_supported(inner) -> bool:
    """The dequant reads the native or band-8 packed layout; rates that band to the AVX2 planar
    layout (K1 on AVX-512 hosts, integer rates on AVX2-only hosts) stay on the stock mul1 path."""
    return int(ext.exl3_moe_cpu_swizzle_group(float(inner.K))) in (0, 8)


def cpu_prefill_torch_forward(inner, x2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    from ...util import cuda_flags as _cf
    global _cpu_bf16_cache
    cache_bytes = _cf.cpu_prefill_cache_bytes()
    w = None
    if cache_bytes > 0:
        if _cpu_bf16_cache is None:
            import weakref
            _cpu_bf16_cache = weakref.WeakKeyDictionary()
        entry = _cpu_bf16_cache.get(inner)
        if entry is not None:
            # LRU refresh (pop+reinsert moves the key to the most-recent end); keys are
            # weakrefs so a model unload drops its whole entry set
            _cpu_bf16_cache[inner] = _cpu_bf16_cache.pop(inner)
            w = entry
    if w is None:
        # One dequant per (layer tensor, cache epoch); the padded tile columns beyond
        # out_features are dropped so the matmul only produces real outputs. tier "fp32"
        # keeps the weights fp32 (no bf16 rounding — the parity-fallback variant)
        from ...util import cuda_flags as _cf0
        _fp32 = _cf0.cpu_prefill_tier() == "fp32"
        w = ext.exl3_cpu_dequant_bf16(
            inner.trellis, inner.suh, inner.svh, inner.K, inner.cpu_swz, _fp32
        )[:inner.out_features].contiguous()
        if cache_bytes > 0 and w.numel() * w.element_size() <= cache_bytes:
            _cpu_bf16_cache[inner] = w
            # cumulative budget: evict least-recently-used whole tensors until we fit
            used = lambda: sum(v.numel() * v.element_size() for v in _cpu_bf16_cache.values())
            while used() > cache_bytes and len(_cpu_bf16_cache) > 1:
                _cpu_bf16_cache.pop(next(iter(_cpu_bf16_cache)))
    y = torch.matmul(x2d.to(w.dtype), w.t())
    y = y.float()
    if inner.bias is not None:
        y += inner.bias.float()
    return y if dtype == torch.float else y.to(torch.half)

if _cpu_thp:
    import ctypes as _ct

    _libc = _ct.CDLL(None, use_errno = True)
    _libc.madvise.argtypes = [_ct.c_void_p, _ct.c_size_t, _ct.c_int]

    def thp_advise(tensor: torch.Tensor, min_bytes: int = 4 << 20) -> None:
        try:
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes < min_bytes:
                return
            beg = tensor.data_ptr() & ~_ct.c_size_t(4095)
            end = (tensor.data_ptr() + nbytes + 4095) & ~_ct.c_size_t(4095)
            # Mark the range eligible; a THP=madvise host's khugepaged collapses it in the
            # background. MADV_COLLAPSE (6.1+) would do it now but is rejected on some kernels
            # for ordinary anonymous mappings (EINVAL), so it is only attempted
            if _libc.madvise(_ct.c_void_p(beg), _ct.c_size_t(end - beg), 14) != 0:   # MADV_HUGEPAGE
                return
            _libc.madvise(_ct.c_void_p(beg), _ct.c_size_t(end - beg), 25)            # MADV_COLLAPSE
        except Exception:
            pass
else:
    def thp_advise(tensor: torch.Tensor, min_bytes: int = 4 << 20) -> None:
        pass


def _swizzle_trellis_cpu(trellis: torch.Tensor, group: int, planar: bool) -> torch.Tensor:
    tk, tn, ps = trellis.shape
    dst = torch.empty_like(trellis)
    if planar:
        wd = ps // 2
        dst.view(torch.int32).view(tn // group, tk, group, wd // 8, 8).copy_(
            trellis.view(torch.int32).view(tk, tn // group, group, 8, wd // 8).permute(1, 0, 2, 4, 3))
    else:
        dst.view(tn // group, tk, group, ps).copy_(
            trellis.view(tk, tn // group, group, ps).permute(1, 0, 2, 3))
    return dst


def cpu_linear_forward_batch(linears, x, params, out_dtypes = None):
    """Batched CPU decode forward for same-input EXL3 linears (one projection group per
    worker-pool round set instead of one call per matrix). Returns the per-linear outputs with
    each linear's own shape; out_dtypes[i] (or the Linear's own resolution when None) decides
    fp32/fp16 per output. Falls back to the per-linear forward path whenever any linear would
    take anything but the plain EXL3 kernel semantics (override, padding, softcap, LoRA,
    scalers, trimmed outputs, non-CPU tensors)."""
    ovr = params.get("ovr") if isinstance(params, dict) else None
    k = x.shape[-1]
    # The prefill tier changes per-linear semantics at M >= min, so the batched decode path
    # defers to the per-linear forwards (which tier individually) whenever it would engage
    from ...util import cuda_flags as _cf
    _l0_inner = getattr(linears[0], "inner", linears[0])   # entries may be bare LinearEXL3
    _tier_active = (_cf.cpu_prefill_tier() != "off"
                    and x.numel() // k >= _cf.cpu_prefill_min_m()
                    and cpu_prefill_supported(_l0_inner))
    eligible = x.is_contiguous() and not _tier_active
    if eligible:
        for i, l in enumerate(linears):
            inner = getattr(l, "inner", None)
            if (
                not isinstance(inner, LinearEXL3) or
                not getattr(inner, "is_cpu", False) or
                not inner.mul1 or
                l.in_features != k or
                l.softcap or
                l.lora_a_tensors or
                l.pre_scale != 1.0 or
                l.post_scale != 1.0 or
                (l.out_features != getattr(l, "out_features_unpadded", l.out_features)) or
                (ovr is not None and l.key in ovr)
            ):
                eligible = False
                break
    if not eligible:
        return [
            l.forward(x, params, out_dtypes[i] if out_dtypes else None)
            for i, l in enumerate(linears)
        ]

    dts = []
    for i, l in enumerate(linears):
        dt = out_dtypes[i] if out_dtypes else None
        dts.append(dt or l.out_dtype or l.inner.default_out_dtype)
    # Packed-layout upgrade at first use (the loader's deferred fills may still be pending at
    # construction); all-or-nothing per group keeps the C++ swizzled flag uniform
    if not all(l.inner.cpu_repack() for l in linears):
        # Some matrix in the group has no packed form (or bytes already registered native in a
        # moe handle): the repacked and native members would disagree, so fall back per linear
        return [
            l.forward(x, params, out_dtypes[i] if out_dtypes else None)
            for i, l in enumerate(linears)
        ]
    ys = ext.exl3_cpu_linear_forward_batch(
        [l.inner.trellis for l in linears],
        [l.inner.suh for l in linears],
        [l.inner.svh for l in linears],
        [l.inner.K for l in linears],
        [l.inner.bias for l in linears],
        x.view(-1, k),
        [dt == torch.float for dt in dts],
        [l.inner.cpu_swz for l in linears],
    )
    out = []
    for l, dt, y in zip(linears, dts, ys):
        if y.dtype != dt:
            y = y.to(dt)
        out.append(y.view(x.shape[:-1] + (l.out_features,)))
    return out

class LinearEXL3:

    quant_type: str = "exl3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # Not a Module subclass, so the config-or-NullConfig default doesn't apply here; TP imports pass
        # config=None and forward() reads config.infer_params
        if config is None:
            from ...model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        # Bitrate from the tile width: 16 * K uint16 per tile, 16 * K + 8 for half-integer K (mul1 only)
        K = trellis.shape[-1] / 16
        self.K = int(K) if float(K).is_integer() else K
        self.frac = frac_k(self.K)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None
        assert self.frac is None or self.mul1, f"{key}: half-integer bitrate {self.K} requires the mul1 codebook"

        self._fused_reconstruct = None
        self.bsz1_xh_args = (self.trellis.device, (1, self.in_features), self.out_dtype)
        # BC (CUDA graph buffer collection) and the g_tensor_cache workspace it borrows are
        # CUDA-only; the CPU forward runs the moe_mul1 kernels directly
        self.is_cpu = self.trellis.device.type == "cpu"
        self.cpu_swz = False
        if self.is_cpu:
            self.bc = None
        else:
            # K is the bitrate (int, or float for the half-integer rates); the C++ side decomposes it
            self.bc = ext.BC_LinearEXL3(
                self.trellis,
                self.suh,
                self.svh,
                self.K,
                self.bias,
                self.mcg,
                self.mul1,
                g_tensor_cache.get(*self.bsz1_xh_args)
            )


    def unload(self):
        # g_tensor_cache.drop(*self.bsz1_xh_args)
        pass


    def cpu_can_swizzle(self) -> bool:
        """This linear's trellis has a packed-layout form the CPU kernels can take."""
        return (
            self.is_cpu and self.mul1 and _cpu_swizzle and
            bool(ext.exl3_moe_cpu_swizzle_group(float(self.K)))
        )


    def cpu_repack(self, inplace: bool = False) -> bool:
        """One-time move of the trellis bytes into the packed layout the banded CPU kernels
        stream sequentially (cpu_swz records that the bytes moved; make_matrix re-derives the
        flag from the rate either way). Must not run before the loader's deferred fills have
        landed — construction-time bytes are unfilled staging — so callers do this at first
        use. inplace: permute within the existing storage, keeping data_ptr (a registered
        moe_mul1 layer holds raw pointers into this tensor)."""
        if self.cpu_swz or not self.cpu_can_swizzle():
            return self.cpu_swz
        _swizzle_mallopt()
        Kf = float(self.K)
        grp = ext.exl3_moe_cpu_swizzle_group(Kf)
        pln = bool(ext.exl3_moe_cpu_planar_layout(Kf))
        swz = _swizzle_trellis_cpu(self.trellis, grp, pln)
        if inplace:
            self.trellis.copy_(swz)
        else:
            self.trellis = swz
        self.cpu_swz = True
        thp_advise(self.trellis)
        return True


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        # The EXL3 kernels read x as contiguous rows; a strided view (e.g. a head-group slice
        # of a wider tensor) would be silently misread as interleaved garbage. Producers are
        # responsible for contiguity (a silent copy here would hide a hot-path inefficiency
        # and break CUDA-graph address stability)
        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"

        if x.device.type == "cpu":
            if params.get("reconstruct"):
                raise NotImplementedError(f"LinearEXL3 {self.key}: reconstruct is not supported on CPU")
            assert self.mul1, \
                f"LinearEXL3 {self.key}: CPU forward supports the mul1 codebook only (mcg is unsupported)"
            self.cpu_repack()
            dtype = out_dtype or self.default_out_dtype
            x2d = x.view(-1, self.in_features)
            from ...util import cuda_flags as _cf
            if (_cf.cpu_prefill_tier() != "off" and x2d.shape[0] >= _cf.cpu_prefill_min_m()
                    and cpu_prefill_supported(self)):
                y = cpu_prefill_torch_forward(self, x2d, dtype)
            else:
                y = ext.exl3_cpu_linear_forward(
                    self.trellis, self.suh, self.svh, self.K, self.bias,
                    x2d, dtype == torch.float, self.cpu_swz
                )
            return y.view(x.shape[:-1] + (self.out_features,))

        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]
            if rows <= AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
                dtype = out_dtype or self.default_out_dtype
                return self.bc.run_alloc(x, self.out_features, dtype == torch.float)

        return self.reconstruct_hgemm(x, out_dtype)


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # (Only used for full reconstruct and loading old models, not during inference)
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        # NOT torch.where with CPU scalar tensors: that path misses the device guard when the
        # condition lives on a non-current device (observed on torch 2.11.0+cu130) — the kernel
        # launches on the current device's context, faults there, silently zero-fills the output
        # and leaves every other device in the process unusable. Map bool -> {-1, +1} arithmetically
        expanded = 1.0 - expanded.to(torch.float16) * 2.0
        return expanded.contiguous().to(device)


    def reconstruct_hgemm(self, x: torch.Tensor, out_dtype):

        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)

        y_ = y.view(rows, self.out_features)

        # Fused path: reconstruct emits ORIGINAL-basis weights (both Hadamards + sign
        # vectors folded into the memory-bound reconstruct kernel), so the gemm runs on the
        # raw input and the standalone input/output had_r_128 launches disappear (~14% of
        # long-chunk prefill GPU time). Requires 128-divisible dims (always true for EXL3
        # tensors: both sides are had-transformed at quant time)
        if self._fused_reconstruct is None:
            self._fused_reconstruct = (
                self.in_features % 128 == 0 and self.out_features % 128 == 0
                and not no_fused_reconstruct
            )

        # The fused kernel costs ~4x plain reconstruct (k*n-proportional) while the saved
        # had launches scale with rows*(k+n); breakeven is rows ~400-900 across shapes
        use_fused = self._fused_reconstruct and rows >= 1024

        if use_fused:
            xh = x
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
            else:
                self._reconstruct_inner(w, 0)
            ext.hgemm_recon(xh, w, y_)
        else:
            numel_ = self.in_features * MAX_RECONSTRUCT_SLICE_N
            w_ = torch.empty((numel_,), dtype = torch.half, device = self.trellis.device)
            for n_start in range(0, self.out_features, MAX_RECONSTRUCT_SLICE_N):
                n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, self.out_features)
                numel = self.in_features * (n_end - n_start)
                w = w_[:numel].view(self.in_features, n_end - n_start)
                if use_fused:
                    ext.reconstruct_had_slice(
                        w, self.trellis, self.suh, self.svh[n_start:], self.K, self.mcg, self.mul1, n_start)
                else:
                    self._reconstruct_inner(w, n_start)
                ext.hgemm_recon(xh, w, y_[:, n_start:n_end])

        if not use_fused:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)

        if self.bias is not None:
            y += self.bias
        return y


    def _reconstruct_inner(self, w: torch.Tensor, n_start: int):
        """Rotated-basis weight slice (in_features, w.shape[1]) from packed columns n_start.."""
        if n_start == 0 and w.shape[1] == self.out_features:
            ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        else:
            ext.reconstruct_slice(w, self.trellis, self.K, self.mcg, self.mul1, n_start)

    def get_inner_weight_tensor(self, n_offset: int = 0, n_features: int | None = None):
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        self._reconstruct_inner(w, 0)
        return w


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True) if first == 0 else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearEXL3.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        svh_ = []
        trellis_ = []
        bias_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out

            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            svh_.append(svh)
            trellis_.append(trellis)
            bias_.append(bias)

        svh = torch.cat(svh_, dim = 0)
        trellis = torch.cat(trellis_, dim = 1)
        bias = torch.cat(bias_, dim = 0) if bias_[0] is not None else None

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
