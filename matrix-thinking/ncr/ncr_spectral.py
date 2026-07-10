"""NCR spectral instruments (numpy + stdlib): the Axis-C locked decay-curve
machinery, the per-cell deep probe, and the S3.4 corrected trust rule as an
A-PRIORI SCREEN.

Lineage, imported VERBATIM (never ported-with-drift):

  chapter2/analyze_zdump.py  -- entity_subspace / block_decompose /
      synthetic_keys_from_pi / match_eigenvalues / effective_rank: the exact
      TASK_E_FINDINGS S9/S10 machinery (predicted-vs-measured within
      0.001-0.02 through h=21 is that code's own precedent).
  chapter2/test_trust_rule_negative.py -- trust_rule / linear_rule / op_norm:
      the DISCHARGED S3.8(c) implementation of the S3.4 sigma-form (verified
      against S5's simulation numbers, 29/29; see registry S6). Reusing it
      verbatim means the screen deployed here is bit-identical to the one the
      negative tests N1/N2 proved has teeth.

EXTENSION over the S9 lineage (documented, self-tested): predicted decay
curves must now reach h=1021+ (and the cost-probe depths), where literal
np.linalg.matrix_power overflows fp64 for c* > 1 (2.843^1021 ~ 10^463). The
per-squaring renormalized power (`_scaled_power`) computes the DIRECTION of
A^h exactly up to a strictly positive scalar -- the same cosine-invariance
argument as S3.1's read (a positive scalar cannot move a cosine) -- and the
reference Pi^h uses the exact single-cycle identity Pi^h == Pi^(h mod K)
(integer periodicity of the ideal cycle, no drift accumulation in the
reference). Both are cross-checked against the lineage's literal
matrix_power at h <= 21 in the self-test.

TRUST RULE ROLE (S3.4, mi3 disclosure of record): T(h) is an a-priori
screen -- it REJECTS adversarial/degenerate constructions and ranks seeds;
it REFUSES every decisive far-h point by construction (the honest price of
a true worst-case bound). Decisive-depth attribution rides ENTIRELY on the
Axis-C calibration-LOCKED exact per-seed curves; the fp64 shadow certifies
ROUNDING only, never leakage. Labels per (cell, h): RULE-TRUSTED (T <= tau)
/ SHADOW-VERIFIED (fp64 agreement) / UNTRUSTED -- flagged, never dropped.
Per S5 nit n3: the "4-10x conservative" figure quantifies T_lin ONLY and is
never computed or quoted against the sigma-form here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)

import analyze_zdump as az                    # noqa: E402 (S9/S10 lineage, verbatim)
import test_trust_rule_negative as ttrn       # noqa: E402 (discharged S3.8(c) rule, verbatim)

TAU = 0.2                                     # S3.4 trust threshold, unchanged
SHADOW_BAR = 0.005                            # S3.1: |cos_fp32 - cos_fp64| flag bar
AGREE_BAR = 5e-4                              # MA5: |cos_binexp - cos_loop|, h <= 125
AGREE_H_MAX = 125

# MA3 asymmetric-confidence classification for K=12 h*=57 (locked residuals)
K12_PREDICTED_HOLD_DELTA = 0.0079             # = 0.451/57
K12_STRADDLE_DELTA = 0.0311                   # = 1.772/57


# ---------------------------------------------------------------------------
# Scale-managed matrix powers (fp64, direction-exact)
# ---------------------------------------------------------------------------

def _scaled_power_apply(A: np.ndarray, x: np.ndarray, h: int) -> np.ndarray:
    """Direction of A^h x via square-and-multiply with per-squaring Frobenius
    renorm + per-multiply vector renorm (positive scalars only)."""
    assert h >= 0
    if h == 0:
        n = np.linalg.norm(x)
        return x / max(n, 1e-300)
    base = A.astype(np.float64)
    v = x.astype(np.float64)
    hh = h
    while hh > 0:
        if hh & 1:
            v = base @ v
            v = v / max(np.linalg.norm(v), 1e-300)
        hh >>= 1
        if hh > 0:
            base = base @ base
            base = base / max(np.linalg.norm(base, "fro"), 1e-300)
    return v


def predicted_cos_curve_far(A: np.ndarray, Pi: np.ndarray, keys: np.ndarray,
                            hops) -> dict:
    """S9(e) predicted decay curve, extended to arbitrary depth. Reference
    side uses the exact cycle periodicity Pi^h == Pi^(h mod K); trained side
    uses the renormalized power (direction-exact)."""
    K = keys.shape[0]
    out = {}
    for h in sorted(set(int(x) for x in hops)):
        h_ref = h % K
        Ph = np.linalg.matrix_power(Pi, h_ref) if h_ref > 0 else np.eye(Pi.shape[0])
        cs = []
        for i in range(K):
            pa = _scaled_power_apply(A, keys[i], h)
            pp = Ph @ keys[i]
            npv = np.linalg.norm(pp)
            if npv < 1e-12 or not np.all(np.isfinite(pa)):
                cs.append(0.0)
                continue
            cs.append(float(np.dot(pa, pp / npv)))
        out[h] = float(np.mean(cs))
    return out


# ---------------------------------------------------------------------------
# Per-cell deep probe (m4) + Axis-C lock content
# ---------------------------------------------------------------------------

def analyze_zdump_arrays(Z_list, z_ideal_list, hops) -> dict:
    """Restricted-operator analysis of a cell's Z-dump (lists of dxd arrays),
    returning the m4 deep-probe fields + per-example predicted decay curves
    over `hops`. Mirrors analyze_zdump.analyze_run's per-example loop but on
    in-memory arrays (that function reads a JSON file)."""
    per_example = []
    for Z_raw, Zi_raw in zip(Z_list, z_ideal_list):
        Z = np.asarray(Z_raw, dtype=np.float64)
        Zi = np.asarray(Zi_raw, dtype=np.float64)
        sub = az.entity_subspace(Zi)
        U, V, k_eff = sub["U"], sub["V"], sub["k_eff"]
        A, B, C, D = az.block_decompose(Z, U, V)
        Pi = U.T @ Zi @ U
        K = Pi.shape[0]

        c_star = float(np.sum(A * Pi) / max(np.sum(Pi * Pi), 1e-12))
        resid_scale = float(np.linalg.norm(A - c_star * Pi, "fro")
                            / max(np.linalg.norm(c_star * Pi, "fro"), 1e-12))
        roots = np.exp(2j * np.pi * np.arange(K) / K)
        ev = np.linalg.eigvals(A)
        ev_unit = ev / np.clip(np.abs(ev), 1e-12, None)
        _, phase_resid = az.match_eigenvalues(ev_unit, roots)

        keys, key_resid, imag_leak = az.synthetic_keys_from_pi(Pi)
        curve = predicted_cos_curve_far(A, Pi, keys, hops)

        per_example.append(dict(
            k_eff=int(k_eff),
            c_star=c_star,
            scale_corrected_residual=resid_scale,
            phase_resid_max=float(np.max(phase_resid)),
            phase_resid=[float(x) for x in phase_resid],
            normB=float(np.linalg.norm(B, "fro")),
            normC=float(np.linalg.norm(C, "fro")),
            normD=float(np.linalg.norm(D, "fro")),
            A_eff_rank=float(az.effective_rank(A)),
            synthetic_key_self_consistency=float(key_resid),
            synthetic_key_imag_leak=float(imag_leak),
            predicted_curve={str(h): c for h, c in curve.items()},
            blocks=dict(A=A.tolist(), B=B.tolist(), C=C.tolist(), D=D.tolist()),
        ))
    mean_curve = {}
    for h in hops:
        vals = [ex["predicted_curve"][str(int(h))] for ex in per_example]
        mean_curve[str(int(h))] = float(np.mean(vals))
    return dict(per_example=per_example, mean_predicted_curve=mean_curve,
                phase_resid_max_mean=float(np.mean(
                    [ex["phase_resid_max"] for ex in per_example])))


def k12_confidence_class(delta: float) -> str:
    """MA3 pre-registered asymmetric-confidence classification at h*=57."""
    if delta <= K12_PREDICTED_HOLD_DELTA:
        return "PREDICTED-HOLD"
    if delta <= K12_STRADDLE_DELTA:
        return "STRADDLE"
    return "PREDICTED-FAIL"


def write_axis_c_lock(path: str, cell_id: str, K: int, probe: dict) -> dict:
    """Write the Axis-C LOCK for one cell: predicted per-seed decay curves +
    locked residuals, hashed. MUST be called BEFORE any far-h behavioral
    eval for the cell runs (enforced by run_ncr's eval path, which refuses
    far-h points without a valid lock). K=12 cells additionally carry the
    MA3 classification from the locked residual."""
    content = dict(
        cell_id=cell_id, K=K,
        locked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mean_predicted_curve=probe["mean_predicted_curve"],
        per_example_curves=[ex["predicted_curve"] for ex in probe["per_example"]],
        phase_resid_max_per_example=[ex["phase_resid_max"] for ex in probe["per_example"]],
        phase_resid_max_mean=probe["phase_resid_max_mean"],
        c_star_per_example=[ex["c_star"] for ex in probe["per_example"]],
    )
    if K == 12:
        content["k12_confidence_class"] = k12_confidence_class(
            probe["phase_resid_max_mean"])
    blob = json.dumps(content, sort_keys=True).encode()
    content["lock_sha256"] = hashlib.sha256(blob).hexdigest()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(content, f, indent=1)
    os.replace(tmp, path)
    return content


def verify_axis_c_lock(path: str) -> dict:
    """Re-verify a lock file's own hash (tamper/corruption check)."""
    with open(path) as f:
        content = json.load(f)
    sha = content.pop("lock_sha256")
    blob = json.dumps(content, sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest() == sha, f"Axis-C lock hash mismatch: {path}"
    content["lock_sha256"] = sha
    return content


# ---------------------------------------------------------------------------
# S3.4 trust rule (a-priori screen) on measured blocks
# ---------------------------------------------------------------------------

def trust_screen(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                 hops) -> dict:
    """Evaluate the DISCHARGED sigma-form rule (ttrn.trust_rule, verbatim) on
    a cell's measured blocks at every grid h. Operator 2-norms throughout
    (the rule's pinned convention). Overflow -> inf -> UNTRUSTED (fail-safe,
    S5 item 5). B-feedback neglect disclosed per S3.4."""
    sig_A = np.linalg.svd(A, compute_uv=False)
    sigmax_A, sigmin_A = float(sig_A[0]), float(sig_A[-1])
    sigmin_A = max(sigmin_A, 1e-300)
    sigmax_D = float(np.linalg.svd(D, compute_uv=False)[0]) if D.size else 0.0
    a = sigmax_A / sigmin_A
    r = sigmax_D / sigmin_A
    base = ttrn.op_norm(C) / sigmin_A if C.size else 0.0
    out = dict(a=a, r=r, C_over_sigmin=base,
               b_feedback_neglect=float(
                   (np.linalg.norm(B, 2) * np.linalg.norm(C, 2)) / sigmin_A**2)
               if (B.size and C.size) else 0.0,
               tau=TAU, per_h={})
    m = max(a, r)
    for h in sorted(set(int(x) for x in hops)):
        try:
            T = ttrn.trust_rule(a, r, base, h)
            envelope = base * h * (m ** (h - 1))       # S3.4 sanity cross-check
        except OverflowError:
            T, envelope = math.inf, math.inf
        if not math.isfinite(T):
            T = math.inf
        # the envelope bounds both branches (S3.4); allow fp slop
        assert (not math.isfinite(T)) or T <= envelope * (1 + 1e-9) + 1e-12, (h, T, envelope)
        out["per_h"][str(h)] = dict(T=T if math.isfinite(T) else "inf",
                                    rule_trusted=bool(math.isfinite(T) and T <= TAU))
    return out


def trust_label(rule_trusted: bool, shadow_delta: float | None) -> str:
    """mi3 reporting label per (cell, h)."""
    if rule_trusted:
        return "RULE-TRUSTED"
    if shadow_delta is not None and abs(shadow_delta) <= SHADOW_BAR:
        return "SHADOW-VERIFIED"
    return "UNTRUSTED"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    K, d = 8, 16
    # exact ideal: cycle over a random orthonormal pool (matches task_e's own
    # construction: z_ideal = sum value_i key_i^T)
    pool, _ = np.linalg.qr(rng.standard_normal((d, d)))
    keys_raw = pool[:, :K].T
    vals_raw = np.roll(keys_raw, -1, axis=0)
    Zi = vals_raw.T @ keys_raw            # sum_i v_i k_i^T
    # trained stand-in: exact cycle * scalar + small C leakage + a healthy-
    # seed-like D block (sigma_max(D) slightly above sigma_min(A), the
    # measured r>1 regime -- without D the screen would trust deep h and the
    # refuse-deep assertion below would be vacuous)
    Z = 2.843 * Zi.copy()
    sub = az.entity_subspace(Zi)
    U, V = sub["U"], sub["V"]
    Z = Z + V @ (0.01 * rng.standard_normal((d - K, K))) @ U.T
    Z = Z + V @ (3.0 * np.eye(d - K)) @ V.T

    hops = (1, 2, 3, 5, 13, 21, 61, 125, 1021)
    probe = analyze_zdump_arrays([Z], [Zi], hops)
    ex = probe["per_example"][0]
    assert ex["k_eff"] == K, ex["k_eff"]
    assert abs(ex["c_star"] - 2.843) < 1e-6
    assert ex["phase_resid_max"] < 1e-6
    # predicted curve ~1.0 at every depth for a (near-)exact scaled cycle
    for h in hops:
        assert ex["predicted_curve"][str(h)] > 0.999, (h, ex["predicted_curve"][str(h)])
    print("  deep probe recovers c*=2.843 exact cycle; predicted curve ~1.0 to h=1021 "
          "(renormalized power stays finite where literal fp64 power would overflow)")

    # cross-check the far-power machinery against the S9 lineage's literal
    # matrix_power at h<=21 on a NON-trivial operator (one damped mode)
    A = U.T @ Z @ U
    A_damp = A.copy()
    evals, evecs = np.linalg.eig(A_damp)
    evals[0] *= 0.5
    A_damp = np.real(evecs @ np.diag(evals) @ np.linalg.inv(evecs))
    Pi = U.T @ Zi @ U
    skeys, _, _ = az.synthetic_keys_from_pi(Pi)
    lineage = az.predicted_cos_curve(A_damp, Pi, skeys, [1, 3, 7, 21])
    mine = predicted_cos_curve_far(A_damp, Pi, skeys, [1, 3, 7, 21])
    for h in (1, 3, 7, 21):
        assert abs(lineage[h] - mine[h]) < 1e-9, (h, lineage[h], mine[h])
    print("  far-power curve == lineage literal-matrix-power curve at h<=21 (<1e-9)")

    # trust screen: healthy-seed-like blocks trust shallow h, refuse deep h
    A8, B8, C8, D8 = az.block_decompose(Z, U, V)
    scr = trust_screen(A8, B8, C8, D8, (1, 21, 61))
    assert scr["per_h"]["1"]["rule_trusted"] is True, scr["per_h"]["1"]
    assert scr["per_h"]["61"]["rule_trusted"] is False  # S3.4: refuses h*
    # N2 cross-check against the DISCHARGED archive (deterministic case)
    arch_path = os.path.join(CHAPTER2, "test_trust_rule_negative", "results.json")
    with open(arch_path) as f:
        arch = json.load(f)
    A_n2 = ttrn.build_cycle(8)
    C_n2 = np.zeros((8, 8)); C_n2[1, 3] = 0.01
    D_n2 = np.zeros((8, 8)); D_n2[0, 1] = 100.0
    scr2 = trust_screen(A_n2, np.zeros((8, 8)), C_n2, D_n2, (21,))
    T_mine = scr2["per_h"]["21"]["T"]
    T_arch = arch["N2"]["T_sigma_21_corrected"]
    assert abs(T_mine - T_arch) / T_arch < 1e-9, (T_mine, T_arch)
    assert scr2["per_h"]["21"]["rule_trusted"] is False
    print(f"  trust screen reproduces the DISCHARGED N2 archive value "
          f"T_sigma(21)={T_arch:.4e} exactly and REJECTS it (wiring check vs "
          f"chapter2/test_trust_rule_negative/results.json)")

    # labels
    assert trust_label(True, None) == "RULE-TRUSTED"
    assert trust_label(False, 0.001) == "SHADOW-VERIFIED"
    assert trust_label(False, 0.01) == "UNTRUSTED"
    assert trust_label(False, None) == "UNTRUSTED"
    # MA3 classes
    assert k12_confidence_class(0.0044) == "PREDICTED-HOLD"
    assert k12_confidence_class(0.0099) == "STRADDLE"
    assert k12_confidence_class(0.0125) == "STRADDLE"
    assert k12_confidence_class(0.04) == "PREDICTED-FAIL"
    print("ncr_spectral self-test PASSED")


if __name__ == "__main__":
    _self_test()
