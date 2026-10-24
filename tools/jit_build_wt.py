#!/usr/bin/env python3
"""JIT-build THIS worktree's exllamav3_ext into $TORCH_EXTENSIONS_DIR.

Why this exists: the .venv's editable install drops the MAIN checkout's repo root on
sys.path, so `import exllamav3_ext` finds the main tree's prebuilt HIP .so and
exllamav3/ext.py line 39 (is_precompiled_extension_available) happily imports it —
the worktree's kernel sources would never be compiled. Every GPU plan that edits
kernel code must build its own extension; plans 02/04/05/06 changed python-side code
only, which is why they could run against the shared .so.

Usage (from the worktree root):
  source ../env-rocm10.sh
  export MAX_JOBS=16 TORCH_EXTENSIONS_DIR=$PWD/.torch_ext PYTHONPATH=$PWD
  python tools/jit_build_wt.py          # builds; prints the resulting .so path

Afterwards, put the printed directory FIRST on PYTHONPATH so `import exllamav3_ext`
resolves to this build, e.g.:
  export PYTHONPATH=$TORCH_EXTENSIONS_DIR/exllamav3_ext:$PWD
"""
import os
import sys

MAIN_ROOT = "/home/daniel/development/gunzi/local-llm/exllamav3"

# Scrub every sys.path entry that could expose the main tree's prebuilt .so, so
# is_precompiled_extension_available() is False and ext.py takes the JIT path.
scrubbed = []
for p in sys.path:
    real = os.path.realpath(p) if p and os.path.exists(p) else (p or "")
    if real.startswith(MAIN_ROOT):
        continue
    scrubbed.append(p)
sys.path[:] = scrubbed

import exllamav3  # noqa: E402  (worktree package via PYTHONPATH)
wt = os.path.dirname(os.path.abspath(exllamav3.__file__))
assert wt == os.path.realpath(os.path.join(os.getcwd(), "exllamav3")), exllamav3.__file__
print(f"[jit_build_wt] package: {exllamav3.__file__}", flush=True)

import exllamav3.ext as ext  # noqa: E402  (triggers the JIT build, minutes)
print(f"[jit_build_wt] extension: {ext.exllamav3_ext.__file__}", flush=True)
print(f"[jit_build_wt] add to PYTHONPATH ahead of $PWD: "
      f"{os.path.dirname(os.path.abspath(ext.exllamav3_ext.__file__))}", flush=True)
print(f"[jit_build_wt] shapes: {ext.exllamav3_ext.exl3_gemm_num_kernel_shapes()}", flush=True)
