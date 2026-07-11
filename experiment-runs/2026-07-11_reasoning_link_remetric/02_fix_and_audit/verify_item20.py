"""Throwaway LOCAL verification harness -- NOT part of the production tree.
Works around a genuine, PRE-EXISTING, UNRELATED bug discovered this session:
reasoning_link_probe.py's own CPU-fla-stub installer (_ensure_fla_stub) never
registers fla.ops.gated_delta_rule, so ANY import of reasoning_link_probe (or
reasoning_link_stage_minus1, which imports it) crashes at lm_pretrain_rd.py's
top-level `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` --
reproduced identically on the box under the chain's own documented
`REASONING_LINK_FORCE_CPU_STUB=1 python reasoning_link_probe.py --mode
selftest` invocation. This is UNRELATED to the squeeze_state_head fix (this
session never touches _ensure_fla_stub) and OUT OF SCOPE for the instrument
verification task -- reported to the coordinator, not silently fixed here.
This script pre-registers a minimal gated_delta_rule stub in sys.modules
purely so item 20 (and a couple of pure-CPU-tensor sanity items) can be
IMPORTED and RUN to validate the fix, without touching any committed file.
"""
import os
import sys
import types

sys.path.insert(0, "/Users/samuellarson/Experiments/learned-representations/matrix-thinking/deltanet_rd")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal pre-registration so lm_pretrain_rd.py's top-level import succeeds.
fla_gdr = types.ModuleType("fla.ops.gated_delta_rule")


def _stub_chunk_gated_delta_rule(*a, **k):
    raise RuntimeError("gated_delta_rule stub called -- not exercised by item 20/2/extra")


fla_gdr.chunk_gated_delta_rule = _stub_chunk_gated_delta_rule
sys.modules["fla.ops.gated_delta_rule"] = fla_gdr

import reasoning_link_stage_minus1 as s1  # noqa: E402

print("import OK, FLA_STUB_INSTALLED =", s1.rlp.FLA_STUB_INSTALLED)

s1.test_item_20_squeeze_state_head_layout_closed_form()
s1.test_item_2_hand_built_composition()
s1.test_extra_outcome_routing_gates()

print("\nALL THREE TARGETED CHECKS PASSED (item 20, item 2 regression sanity, extra outcome-routing gates)")
