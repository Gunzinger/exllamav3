from .cpu import causal_conv1d_update_cpu, gdn_fused_op_cpu, gdn_fused_op_2_cpu

try:
    from .conv1d import causal_conv1d_update
except ModuleNotFoundError:
    # Hosts without triton (CPU-only builds): conv1d's Triton kernels cannot import, and the
    # convolution update runs its torch path instead; callers of the kernel entry point fail at
    # use time rather than failing the package import
    causal_conv1d_update = None

from .gated_delta_rule import gated_delta_rule_fn
