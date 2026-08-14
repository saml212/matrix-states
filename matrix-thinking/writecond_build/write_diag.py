"""M7's repair: Band 2's emission point. As specified (D5.3 Band 2 gates on
L_key/||Z_sgd w||, but NOTHING in the pipeline logs either quantity at
eval -- attack R4's own finding: "as specified, the harvest cannot
evaluate its own second gate").

This module is the EMISSION FUNCTION (`compute_write_diag`), named per the
report's own field list: L_key, L_key_cstar, Zw_norm, Z_fro, Zw_ratio,
keys_cond, keys_null_gap. Zero marginal cost at eval_arm_at_hops's own
full_graft call site -- every tensor it needs (Z, keys_v, values_v) is
already in scope there.

WIRING (Stage-1 build, NOT authorized this round -- BUILD-PREPARABLE per
the design doc's own list): `eval_arm_at_hops` (ncr_lm_wave1_runner.py:934,
full_graft only, guarded by an `is_full_graft` flag the same way
compute_arm_losses already gates aux/ortho) should call
`compute_write_diag(Z, keys_v, values_v)` right after its own
`ncr_lm_forward_ablatable` call (runner:942) and fold the returned dict
into `out[f"h={h}"]` alongside `disc` (runner:949-952). This module ships
the tested FUNCTION; the one-line call-site edit into the pinned runner
itself is deferred to the Stage-1 build (editing an archived
experiment-runs/ file is out of this round's authorized scope -- design
doc `§A4-ADJUDICATION`, "Repo writes allowed: the design-doc append +
matrix-thinking/writecond_build/**").

No box dependency: pure torch. CPU-testable.
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from write_supervision_loss import band2_check, null_directions  # noqa: E402


def compute_write_diag(Z: torch.Tensor, keys_v: torch.Tensor, values_v: torch.Tensor,
                        K: int | None = None, eps: float = 1e-6) -> dict:
    """Returns the six named fields (M7), each a LIST of per-episode floats
    (matching the runner's own `default=str`/`json.dump` convention for
    per-batch-item arrays -- discriminability_metrics returns scalars
    reduced over the batch, but write_diag's own consumer, Band 2, needs
    the per-episode distribution to compute median/p90, so this emits the
    full per-episode vector, not a pre-reduced scalar; a harvest script
    reduces it exactly the way band2_check already demonstrates).
    """
    keys_v_d = keys_v.detach()
    b2 = band2_check(Z, keys_v, values_v, K=K)

    Kdim = keys_v_d.shape[1]
    _, S, _ = torch.linalg.svd(keys_v_d, full_matrices=True)
    keys_cond = (S[:, 0] / (S[:, -1] + eps))
    keys_null_gap = (S[:, -1] / (S[:, -2] + eps)) if Kdim >= 2 else torch.full_like(S[:, -1], float("nan"))

    return dict(
        L_key=b2["L_key_raw"].tolist(),
        L_key_cstar=b2["L_key_cstar"].tolist(),
        Zw_norm=b2["Zw_fro"].tolist(),
        Z_fro=b2["Z_fro"].tolist(),
        Zw_ratio=b2["Zw_ratio"].tolist(),
        keys_cond=keys_cond.tolist(),
        keys_null_gap=keys_null_gap.tolist(),
    )
