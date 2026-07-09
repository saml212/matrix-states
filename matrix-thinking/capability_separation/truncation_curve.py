"""CAPABILITY_SEPARATION_DESIGN.md S1.4.2/S1.5 M2 -- the post-hoc rank-k
truncation curve (S1.22 BA-F2 fix: this module had ZERO implementation at
build-audit time -- a pre-registered corroborating measurement with no code).

Given an UNCONSTRAINED-arm checkpoint (a model trained with no
`force_rank_k`), scores the SAME held-out eval-word sample (fixed
`base_seed`, so the curve is comparable across k, not confounded by
resampling) at k=1..d_state via readout.py's PRODUCTION pipeline -- reusing
its EXISTING `force_rank_k` threading (`GroupWordModel.encode` ->
`rank_utils.truncate_to_rank`, applied to the trained checkpoint's OWN
forward output; genuinely "post-hoc" since the checkpoint's WEIGHTS are
never retrained per k -- the same mechanism `force_rank_arms.py`'s 3-point
grid already uses, just swept over the FULL k=1..d_state range instead of 3
grid points).

Knee rule, pinned exactly per S1.5's M2 text: "Knee k* = smallest k with
acc(k) >= 0.9*acc(k=d_state)" -- Task D/E's own convention
(`chapter2/run_stage1.py`'s M2: "knee = smallest k reaching 0.9*acc(d)"),
ported to this design's `recovered_frac_90` field name and `d_state` bound
(this design's own primary threshold is tau=0.9, S1.5, not Task D Stage-1's
tau=0.99).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chapter2"))
from groups import D_STATE, GROUP_NAMES
from group_task import generating_set
from group_word_encoder import GroupWordModel, recovery_cosine
from rank_utils import truncate_to_rank
import readout

KNEE_METRIC = "recovered_frac_90"   # S1.5 M2's primary metric, tau=0.9 convention


def knee_from_curve(curve: dict, d_state: int, metric: str = KNEE_METRIC) -> int | None:
    """S1.5 M2's pinned knee rule: k* = smallest k in [1, d_state] with
    metric(k) >= 0.9 * metric(k=d_state) (Task D/E's `run_stage1.py`
    convention, "knee = smallest k reaching 0.9*acc(d)", ported to this
    design's field names). Returns None if no k clears the bar (should only
    happen if even k=d_state itself scores non-finite/degenerate)."""
    acc_d = curve[d_state][metric]
    for k in range(1, d_state + 1):
        if curve[k][metric] >= 0.9 * acc_d:
            return k
    return None


def truncation_curve(model, name: str, base_seed: int, device="cpu") -> dict:
    """S1.4.2/S1.5 M2: run readout.py's production pipeline at k=1..d_state
    on the SAME fixed eval-word sample (base_seed held constant across k --
    a comparable curve, not confounded by resampling), on an
    UNCONSTRAINED-arm model (the caller's responsibility: this function does
    not check how `model` was trained -- it only applies `force_rank_k`
    POST-HOC at eval time, exactly S1.5 M2's definition). Returns
    {'group', 'd_state', 'base_seed', 'curve': {k: {...scores}}, 'acc_d',
    'knee'}."""
    d_state = D_STATE[name]
    curve = {}
    for k in range(1, d_state + 1):
        scores = readout.run_subspace_restriction_pipeline(
            model, name, base_seed=base_seed, device=device, force_rank_k=k)
        curve[k] = dict(recovered_frac_90=scores["recovered_frac_90"], mean_cos=scores["mean_cos"],
                        restricted_effective_rank=scores["restricted_effective_rank"])
    knee = knee_from_curve(curve, d_state, metric=KNEE_METRIC)
    return dict(group=name, d_state=d_state, base_seed=base_seed, curve=curve,
               acc_d=curve[d_state][KNEE_METRIC], knee=knee)


# ---------------------------------------------------------------------------
# UNIT TEST (S1.22 BA-F2): synthetic Z of a KNOWN, EXACT planted rank --
# proves the knee-detection arithmetic itself has teeth (does the curve
# actually knee where a real rank boundary is?), independent of the full
# Procrustes-degauging pipeline (already exercised separately by
# gate1_synthetic_injection.py and readout.py's own tests).
# ---------------------------------------------------------------------------

def _test_truncation_curve_knees_at_planted_rank():
    print("=" * 88)
    print("UNIT TEST -- BA-F2: truncation curve knees at a KNOWN planted rank")
    print("=" * 88)
    torch.manual_seed(20260722)
    d_state, B, r_true = 7, 16, 4

    # Construct Z as a rank-r_true ORTHOGONAL PROJECTOR per matrix: Q (d_state x
    # r_true) has orthonormal columns (via QR of a random Gaussian), so
    # Z = Q @ Q.T is symmetric idempotent, rank EXACTLY r_true, with ALL r_true
    # nonzero singular values TIED at exactly 1.0. This tied spectrum is what
    # gives a genuinely SHARP knee (cos(rank-k truncation, Z) = sqrt(k/r_true)
    # exactly, a clean spectral-truncation identity for a tied-projector target)
    # -- an untied random-Gaussian construction (Z=A@A.T with generic singular
    # values) instead gives a SMOOTH curve that clears the 0.9x bar well before
    # k=r_true, since a few dominant directions already capture most of the
    # Frobenius energy -- verified to under-shoot the planted rank in an earlier
    # draft of this test; the tied-projector construction is what actually
    # isolates "does the knee land at the true rank boundary," not an artifact
    # of an arbitrary singular-value profile.
    A = torch.randn(B, d_state, r_true)
    Q, _ = torch.linalg.qr(A)
    Z_true = Q @ Q.transpose(-1, -2)
    ranks = torch.linalg.matrix_rank(Z_true)
    assert (ranks == r_true).all(), \
        f"construction bug: not every sample is exactly rank {r_true} ({ranks.tolist()})"
    print(f"  constructed {B} synthetic (d_state={d_state}) rank-{r_true} projector matrices "
          f"(tied singular values @ 1.0), verified rank == {r_true} for all {B}")

    curve = {}
    for k in range(1, d_state + 1):
        Z_k = truncate_to_rank(Z_true, k)
        cos = recovery_cosine(Z_k, Z_true)   # per-episode cosine vs the UNTRUNCATED planted target
        curve[k] = dict(mean_cos=float(cos.mean()), recovered_frac_90=float((cos > 0.9).float().mean()))
        print(f"  k={k}: mean_cos={curve[k]['mean_cos']:.6f}  recovered_frac_90={curve[k]['recovered_frac_90']:.4f}")

    knee = knee_from_curve(curve, d_state, metric="recovered_frac_90")
    print(f"\n  knee (recovered_frac_90, 0.9x rule) = {knee}  (expect exactly the planted rank {r_true})")
    assert knee == r_true, f"BA-F2 REGRESSION: curve kneed at k={knee}, expected the planted rank {r_true}"

    # sanity: k >= r_true must be LOSSLESS (best-rank-k of an already-rank-r_true
    # PSD matrix, for k>=r_true, reconstructs it EXACTLY -- standard spectral
    # truncation theory, not particular to this design); k < r_true must NOT be.
    for k in range(r_true, d_state + 1):
        assert curve[k]["mean_cos"] > 0.9999, \
            f"k={k}>=r_true={r_true} should reconstruct exactly, got mean_cos={curve[k]['mean_cos']:.6f}"
    assert curve[r_true - 1]["mean_cos"] < curve[r_true]["mean_cos"] - 1e-4, \
        "truncating below the planted rank should measurably hurt reconstruction fidelity"
    print(f"\nRESULT: knee correctly lands at the planted rank r_true={r_true}; k>=r_true is lossless, "
          f"k<r_true measurably degrades.\n")
    return knee


# ---------------------------------------------------------------------------
# SMOKE: exercises truncation_curve() through the REAL readout.py pipeline
# on an untrained model (shape/wiring sanity only -- an untrained model has
# no learned structure to knee correctly at, mirroring force_rank_arms.py's
# own untrained-model smoke convention).
# ---------------------------------------------------------------------------

def smoke(device="cpu"):
    print("=" * 88)
    print("  truncation_curve.py SMOKE -- M2 curve wiring through the real readout pipeline (S4)")
    print("=" * 88)
    torch.manual_seed(0)
    name = "S4"
    d_state, n_gens = D_STATE[name], len(generating_set(name))
    model = GroupWordModel(d_state, n_gens, L_max=16, h=32).to(device)

    result = truncation_curve(model, name, base_seed=9001, device=device)
    print(f"  group={result['group']}  d_state={result['d_state']}  acc_d={result['acc_d']:.4f}  "
         f"knee={result['knee']}")
    for k in sorted(result["curve"]):
        c = result["curve"][k]
        print(f"    k={k}: recovered_frac_90={c['recovered_frac_90']:.4f}  mean_cos={c['mean_cos']:.4f}")
    assert set(result["curve"].keys()) == set(range(1, d_state + 1)), \
        f"curve must cover k=1..d_state={d_state} exactly, got keys {sorted(result['curve'].keys())}"
    assert result["knee"] is None or 1 <= result["knee"] <= d_state
    print("\n" + "=" * 88 + "\n  truncation_curve.py SMOKE PASSED (wiring only -- untrained model, "
          "no CONFIRM claim)\n" + "=" * 88)


if __name__ == "__main__":
    _test_truncation_curve_knees_at_planted_rank()
    smoke("cuda" if torch.cuda.is_available() else "cpu")
