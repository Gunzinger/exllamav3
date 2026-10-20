#!/usr/bin/env python
"""Run cpu-trial unit tests against THIS worktree (not the checked-out clone).

The test files sys.path.insert the main clone, but importing the worktree's exllamav3
package FIRST binds it in sys.modules, so every later import resolves here. Always set
TORCH_EXTENSIONS_DIR to a worktree-local dir so the JIT build never races the main tree.
Usage, from cpu-trial/:
  TORCH_EXTENSIONS_DIR=<abs wt>/.torch_ext .venv/bin/python wt-<x>/wt_run_tests.py tests/test_x.py [...]
"""
import os
import sys

WT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WT)
import exllamav3  # noqa: F401  (binds the worktree package)
print(f"[wt_run_tests] exllamav3 from {exllamav3.__file__}")
assert exllamav3.__file__.startswith(WT), "worktree binding failed"
assert os.environ.get("TORCH_EXTENSIONS_DIR", "").startswith(WT), \
    "set TORCH_EXTENSIONS_DIR to a worktree-local directory"

import pytest
sys.exit(pytest.main(["-q"] + sys.argv[1:]))
