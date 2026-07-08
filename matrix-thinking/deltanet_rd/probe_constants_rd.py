"""probe_constants_rd.py -- tiny, dependency-free home for the two frozen-
construction seeds HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1.1 pins for the Wave
-1 shared probe head, factored out of `probe_head_rd.py` so
`k_bindings_diagnostic_rd.py` (pure numpy, sec 1.3.1.5 -- "CPU-only, no
backbone forward pass, no training loop") can reuse the SAME
`PROBE_TARGET_SEED` value without importing `probe_head_rd.py`'s own heavy
transitive dependency chain (which pulls in `lm_pretrain_rd`/`model_rd`, and
therefore `fla`/the CPU stub, entirely unnecessary for a pure-numpy
diagnostic). No torch import here at all -- this module is a plain-Python
constants leaf.

`PROBE_TARGET_SEED` is deliberately DIFFERENT from `key_anchoring.py`'s
`ANCHOR_INIT_SEED` (20260705) -- sec 1.3.1.1's own "never share a seed
between the frozen KEY-bias table and this frozen VALUE-target table" rule;
a shared seed would silently correlate the two frozen structures and
confound the frozen-bias mechanism's own probe."""
from __future__ import annotations

PROBE_TARGET_SEED = 20260709
ANCHOR_INIT_SEED_REFERENCE = 20260705  # key_anchoring.py's own value, restated here ONLY so a
                                          # reader of this file can see at a glance the two frozen
                                          # seeds are provably distinct -- never imported for use.

assert PROBE_TARGET_SEED != ANCHOR_INIT_SEED_REFERENCE, (
    "PROBE_TARGET_SEED must stay distinct from key_anchoring.ANCHOR_INIT_SEED (sec 1.3.1.1's "
    "never-share-a-frozen-table-seed rule) -- this module-level assert fires at import time if "
    "that invariant is ever silently broken by an edit here.")
