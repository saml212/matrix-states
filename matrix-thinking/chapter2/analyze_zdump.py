#!/usr/bin/env python3
"""Z-dump subspace analysis for Task E (2026-07-02).

Resolves the M1_E "rank-inflation complication" flagged in TASK_E_FINDINGS.md
section 4: whole-matrix effective/stable rank of converged K=8 operators is
~14.7-15.6 (near the full ambient dim d=16), not ~K=8, even though behavioral
recovery is exact through h=21. This script asks the sharper question: what
does the trained Z do on the K-dimensional entity subspace specifically, and
is the "extra" rank in the orthogonal complement actually load-bearing or
invisible junk?

For each run's Z_dump (4 eval examples: learned operator Z and the analytic
K-cycle target z_ideal, both d x d):

  1. Recover the K-dim entity subspace E from z_ideal's SVD (U = top-K left
     singular vectors). Verify dim(E) == K and that z_ideal's row space
     coincides with its column space (principal angle ~ 0) -- expected for a
     permutation cycle over an orthonormal key set.
  2. Block-decompose Z in the [E, E-perp] basis:
         A = U^T Z U    (K x K, restricted operator)
         B = U^T Z V    (K x (d-K), E-perp -> E leakage)
         C = V^T Z U    ((d-K) x K, E -> E-perp leakage)
         D = V^T Z V    ((d-K) x (d-K), complement dynamics)
     where V spans E-perp (the remaining left singular vectors of z_ideal;
     verified interchangeable with the right-singular-vector complement by
     the principal-angle check in step 1).
  3. Test the invariant-subspace hypothesis: A vs Pi := U^T z_ideal U (the
     exact K-cycle expressed in the U basis); eigenvalues of A vs the K-th
     roots of unity; effective rank of A. Report ||B||_F, ||C||_F and derive
     which one actually gates exact h-fold composition for queries living in
     E (see DERIVATION below) -- it is C, not B.
  4. Characterize D: Frobenius norm, spectral radius, effective/stable rank.
     Tests whether D is a structured or unconstrained/free leftover that the
     readout never sees (because C ~ 0 means a query starting in E never
     visits E-perp, so D's structure is invisible to the loss).
  5. For force-rank-7 runs: identify which eigenmode of A is missing/
     attenuated relative to the K-th roots of unity, and predict the full
     depth-decay curve cos(h) purely from A's own spectrum -- via literal
     matrix powers A^h, no access to the raw (unrecorded) key vectors needed,
     see DERIVATION -- compared against the run's own measured mean_cos /
     recovered_frac at the same hops (including the h=21 depth probe).

DERIVATION (why synthetic "keys" can be built from z_ideal's eigenvectors
alone, with no raw key vectors in the dump):
    z_ideal = K_mat @ P @ K_mat^T, where K_mat's K columns are the actual
    (unrecorded) orthonormal key vectors and P is the canonical K-cycle
    permutation matrix in K_mat's own frame (0/1 pattern). Let U be ANY
    orthonormal basis for span(K_mat) = E (e.g. from an SVD -- the specific
    rotation is arbitrary since z_ideal's top-K singular value is degenerate
    at exactly 1). Then K_mat = U R for some orthogonal K x K matrix R
    (R = U^T K_mat, unknown -- we never see K_mat directly). Pi := U^T z_ideal
    U = R P R^T: an R-conjugate of the canonical cycle, so Pi's eigenvectors
    are R w_j where w_j are P's DFT eigenvectors (P w_j = zeta^j w_j).
    A key vector e_i, expressed in the U frame, is a_i := U^T e_i = R delta_i
    (delta_i = i-th standard basis vector of K_mat's own frame). Decomposing
    a_i in Pi's eigenbasis {R w_j}: a_i = sum_j c_j (R w_j) implies delta_i =
    sum_j c_j w_j (multiply by R^T), i.e. c_j = conj(w_j)_i = (1/sqrt(K))
    zeta^{-ij} -- the ordinary DFT coefficient, independent of R. So: ANY key
    vector, expressed in ANY orthonormal E-basis U, decomposes into Pi's
    OWN eigenbasis with EXACTLY equal magnitude 1/sqrt(K) on every mode --
    a basis-independent consequence of the keys being a cyclically-permuted
    orthonormal set, not an assumption. This lets us reconstruct K synthetic
    "keys" a_i purely from Pi's numerically-computed eigenvectors (sorted by
    phase, assigned j=0..K-1) via the inverse-DFT formula, self-consistency
    checked by verifying Pi @ a_i ~= a_{(i+1) mod K}, and then apply the SAME
    a_i to the TRAINED A to get a fully data-driven, no-raw-keys-needed
    prediction of per-item cosine at any hop h.

Usage:
    PYTHONPATH=<numpy-site-packages> python3 analyze_zdump.py --dir <dir-of-run-jsons>

Self-contained: numpy + stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import warnings

import numpy as np

# Apple Accelerate (this machine's BLAS backend) emits spurious
# divide-by-zero/overflow/invalid RuntimeWarnings on some matmul shapes even
# when the actual output is fully finite -- a known Accelerate quirk, not a
# real numerical issue here (verified: outputs checked with np.isfinite
# below, and by hand against a NaN-free direct computation). Silenced so real
# problems aren't lost in the noise; every downstream block-decompose result
# is still explicitly asserted finite.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ---------------------------------------------------------------------------
# Hungarian (Kuhn-Munkres) matcher -- ported verbatim (algorithm, not code)
# from eigen_utils.py::_hungarian_min_cost so eigenvalue-matching here uses
# the same optimal-assignment convention as the project's C8 metric, not a
# greedy heuristic.
# ---------------------------------------------------------------------------
def _hungarian_min_cost(cost: list) -> list:
    n = len(cost)
    if n == 0:
        return []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [0] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assign[p[j] - 1] = j - 1
    return assign


def match_eigenvalues(ev: np.ndarray, targets: np.ndarray):
    """Optimal (Hungarian) assignment of ev (length-n complex) to targets
    (length-n complex). Returns (assign, residuals) with residuals[i] =
    |ev[i] - targets[assign[i]]|."""
    n = len(ev)
    cost = [[abs(ev[i] - targets[j]) for j in range(n)] for i in range(n)]
    assign = _hungarian_min_cost(cost)
    residuals = np.array([cost[i][assign[i]] for i in range(n)])
    return assign, residuals


# ---------------------------------------------------------------------------
# Rank metrics -- numpy port of rank_utils.py's torch versions (same
# formulas: effective_rank = exp(entropy of normalized singular spectrum),
# stable_rank = ||Z||_F^2 / ||Z||_2^2), so numbers here are directly
# comparable to the training-time M1_E/M2_E numbers already in the JSONs.
# ---------------------------------------------------------------------------
def effective_rank(M: np.ndarray) -> float:
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 1e-10, None)
    p = s / s.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def stable_rank(M: np.ndarray) -> float:
    s = np.linalg.svd(M, compute_uv=False)
    fro_sq = (s ** 2).sum()
    spec_sq = max(s[0] ** 2, 1e-20)
    return float(fro_sq / spec_sq)


# ---------------------------------------------------------------------------
# Subspace recovery + block decomposition
# ---------------------------------------------------------------------------
def entity_subspace(z_ideal: np.ndarray, tol_ratio: float = 1e-3):
    """SVD-based recovery of the K-dim entity subspace E from z_ideal.

    Returns dict with U (d x K, column-space basis), V (d x (d-K), column-
    space complement basis), Vrow (d x K, row-space basis, for the principal-
    angle check), k_eff, and the full singular spectrum s.
    """
    u, s, vt = np.linalg.svd(z_ideal, full_matrices=True)
    thresh = s[0] * tol_ratio
    k_eff = int(np.sum(s > thresh))
    U = u[:, :k_eff]
    V = u[:, k_eff:]
    Vrow = vt[:k_eff, :].T
    return dict(U=U, V=V, Vrow=Vrow, k_eff=k_eff, s=s)


def principal_angles_deg(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between two equal-ambient-dim subspaces
    given orthonormal bases. 0 deg for every angle == identical subspaces."""
    M = U1.T @ U2
    svals = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(svals))


def block_decompose(Z: np.ndarray, U: np.ndarray, V: np.ndarray):
    A = U.T @ Z @ U
    B = U.T @ Z @ V
    C = V.T @ Z @ U
    D = V.T @ Z @ V
    for name, M in (("A", A), ("B", B), ("C", C), ("D", D)):
        if not np.isfinite(M).all():
            raise FloatingPointError(f"block {name} contains non-finite values (real bug, not BLAS noise)")
    return A, B, C, D


# ---------------------------------------------------------------------------
# Synthetic "keys" from Pi's eigenbasis (see module DERIVATION above) +
# depth-decay curve prediction from A's own spectrum.
# ---------------------------------------------------------------------------
def synthetic_keys_from_pi(Pi: np.ndarray):
    """Build K real, orthonormal synthetic key vectors a_i (i=0..K-1) in the
    U-coordinate system purely from Pi's eigendecomposition, using the DFT
    relationship derived in the module docstring. Returns (keys (K,K) real
    array, rows = a_i; self_consistency_resid: mean ||Pi @ a_i - a_{i+1}||)."""
    K = Pi.shape[0]
    eigvals, eigvecs = np.linalg.eig(Pi)
    angles = np.mod(np.angle(eigvals), 2 * np.pi)
    order = np.argsort(angles)
    v = eigvecs[:, order]            # columns v_0..v_{K-1}, ascending phase (Pi v_m = zeta^m v_m)
    zeta = np.exp(2j * np.pi * np.arange(K) / K)
    keys = np.zeros((K, K), dtype=complex)
    # a_i = sum_m (1/sqrt(K)) zeta^{i m} v_m  -- NOT conj(zeta)^i: re-derived by
    # requiring Pi @ a_i == a_{(i+1) mod K} exactly (see module DERIVATION),
    # which fixes the sign given Pi's eigenvector sort convention above. Using
    # conj(zeta)**i here (the naive DFT-coefficient guess) was tried first and
    # produced a self-consistency residual of exactly sqrt(2) (orthogonal, not
    # ~0) on every run -- a clear sign error, not noise. Verify algebraically:
    # Pi a_i = sum_m (zeta^{im}/sqrt(K)) (Pi v_m) = sum_m (zeta^{(i+1)m}/sqrt(K)) v_m = a_{i+1}. QED.
    for i in range(K):
        c = (1.0 / np.sqrt(K)) * zeta ** i
        keys[i] = (v * c[np.newaxis, :]).sum(axis=1)
    imag_leak = float(np.abs(keys.imag).max())
    keys = keys.real
    # normalize (should already be unit-norm up to fp noise)
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    keys = keys / np.clip(norms, 1e-12, None)
    # self-consistency: Pi @ a_i ~= a_{(i+1) mod K}
    resids = []
    for i in range(K):
        pred = Pi @ keys[i]
        actual = keys[(i + 1) % K]
        # sign/rotation ambiguity is not expected here (Pi real, keys real by
        # construction), but guard with a sign flip just in case of a stray
        # global phase from the eig() convention.
        d_plus = np.linalg.norm(pred - actual)
        d_minus = np.linalg.norm(pred + actual)
        resids.append(min(d_plus, d_minus))
    return keys, float(np.mean(resids)), imag_leak


def predicted_cos_curve(A: np.ndarray, Pi: np.ndarray, keys: np.ndarray, hops):
    """cos(h) = mean over the K synthetic keys of cosine(A^h a_i, Pi^h a_i),
    computed via literal matrix powers (no eigen-shortcut), for each h in
    hops. Returns {h: predicted_mean_cos}."""
    out = {}
    K = keys.shape[0]
    Ah = np.eye(A.shape[0])
    Ph = np.eye(Pi.shape[0])
    h_prev = 0
    for h in sorted(hops):
        # advance both matrix powers from h_prev to h
        step = h - h_prev
        if step > 0:
            Ah = np.linalg.matrix_power(A, h) if h <= 30 else np.linalg.matrix_power(A, h)
            Ph = np.linalg.matrix_power(Pi, h)
        h_prev = h
        cs = []
        for i in range(K):
            pa = Ah @ keys[i]
            pp = Ph @ keys[i]
            na, npv = np.linalg.norm(pa), np.linalg.norm(pp)
            if na < 1e-12 or npv < 1e-12:
                cs.append(0.0)
                continue
            cs.append(float(np.dot(pa, pp) / (na * npv)))
        out[h] = float(np.mean(cs))
    return out


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------
def analyze_run(path: str, tol_ratio: float = 1e-3):
    with open(path) as f:
        d = json.load(f)
    if "Z_dump" not in d:
        return None
    K = d["K"]
    dim = d["d"]
    force_rank_k = d.get("force_rank_k")
    seed = d.get("seed")
    Zs = d["Z_dump"]["Z"]
    Zis = d["Z_dump"]["z_ideal"]
    n_ex = len(Zs)

    per_example = []
    for ex in range(n_ex):
        Z = np.array(Zs[ex], dtype=np.float64)
        Zi = np.array(Zis[ex], dtype=np.float64)
        sub = entity_subspace(Zi, tol_ratio=tol_ratio)
        U, V, Vrow, k_eff, s = sub["U"], sub["V"], sub["Vrow"], sub["k_eff"], sub["s"]
        pang = principal_angles_deg(U, Vrow)

        A, B, C, D = block_decompose(Z, U, V)
        Pi = U.T @ Zi @ U

        # A vs Pi (invariant-subspace / exact-cycle hypothesis).
        # Report BOTH the raw (magnitude-and-phase) residual and a
        # scale-corrected one: cosine similarity (what the model is actually
        # scored on) is invariant to a uniform isotropic rescaling of the
        # whole operator (Z -> cZ => Z^h u -> c^h (Zu), same direction), so a
        # trained A that is an exact permutation TIMES a scalar c != 1 would
        # still compose perfectly under the cosine criterion while showing a
        # large raw ||A-Pi|| residual that has nothing to do with structure.
        # c* = argmin_c ||A - c*Pi||_F = <A,Pi>_F / ||Pi||_F^2 (least squares).
        A_minus_Pi_rel = np.linalg.norm(A - Pi, "fro") / max(np.linalg.norm(Pi, "fro"), 1e-12)
        c_star = float(np.sum(A * Pi) / max(np.sum(Pi * Pi), 1e-12))
        A_minus_cPi_rel = np.linalg.norm(A - c_star * Pi, "fro") / max(np.linalg.norm(c_star * Pi, "fro"), 1e-12)

        # eigenvalues of A vs K-th roots of unity. The assignment itself must
        # be done on PHASE-normalized eigenvalues, not raw complex distance:
        # matching raw (unnormalized) eigenvalues against unit-magnitude
        # roots lets a large, cosine-irrelevant magnitude difference dominate
        # the Hungarian cost and scramble the phase-based pairing (verified:
        # doing it on raw values gave nonsensical near-uniform ~1.9 "phase"
        # residuals across most modes for runs where the isotropic-scale
        # story alone should have left 7/8 modes near-perfect). Normalize
        # eigenvalues to the unit circle FIRST, then match -- this is the
        # theoretically correct decoupling of magnitude (cosine-invisible)
        # from phase (composability-relevant, per Grazzi et al. 2411.12537).
        roots = np.exp(2j * np.pi * np.arange(K) / K)
        ev_A_raw = np.linalg.eigvals(A)
        ev_A_mag_raw = np.abs(ev_A_raw)
        ev_A_unit = ev_A_raw / np.clip(ev_A_mag_raw, 1e-12, None)
        assign, resid_A_phase_only = match_eigenvalues(ev_A_unit, roots)
        ev_A_matched = ev_A_raw[assign]
        ev_A_mag = ev_A_mag_raw[assign]
        resid_A = np.abs(ev_A_matched - roots)   # raw (magnitude+phase) residual, same assignment
        ev_Pi_raw = np.linalg.eigvals(Pi)
        ev_Pi_unit = ev_Pi_raw / np.clip(np.abs(ev_Pi_raw), 1e-12, None)
        assign_pi, resid_Pi = match_eigenvalues(ev_Pi_unit, roots)

        rec = dict(
            k_eff=k_eff,
            principal_angle_max_deg=float(pang.max()),
            A_minus_Pi_rel=A_minus_Pi_rel,
            c_star=c_star,
            A_minus_cPi_rel=A_minus_cPi_rel,
            A_eff_rank=effective_rank(A),
            A_stable_rank=stable_rank(A),
            eig_A_matched=ev_A_matched.tolist(),
            eig_A_mag=ev_A_mag.tolist(),
            eig_A_resid=resid_A.tolist(),
            eig_A_resid_max=float(resid_A.max()),
            eig_A_resid_argmax=int(np.argmax(resid_A)),
            eig_A_phase_resid=resid_A_phase_only.tolist(),
            eig_A_phase_resid_max=float(resid_A_phase_only.max()),
            eig_Pi_resid_max=float(resid_Pi.max()),
            normB=float(np.linalg.norm(B, "fro")),
            normC=float(np.linalg.norm(C, "fro")),
            normD=float(np.linalg.norm(D, "fro")),
            normZ=float(np.linalg.norm(Z, "fro")),
            D_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(D)))) if D.size else 0.0,
            D_eff_rank=effective_rank(D) if D.size else 0.0,
            D_stable_rank=stable_rank(D) if D.size else 0.0,
            # D "structured vs noise" test: condition number (top/bottom
            # singular value) of D itself, and of a random Gaussian matrix
            # scaled to the same Frobenius norm, as a reference point. D's
            # own condition number close to 1 (flat spectrum, a near-isometry)
            # is the opposite of what generic i.i.d. noise of the same norm
            # would show (a random Gaussian matrix's spectrum is much less
            # flat -- Marchenko-Pastur-like spread, condition number >> 1 at
            # this size) -- distinguishes "D is close to an unconstrained
            # near-orthogonal free direction" from "D is genuinely noisy."
            D_condition_number=(lambda dsv: float(dsv[0] / max(dsv[-1], 1e-12)))(
                np.linalg.svd(D, compute_uv=False)) if D.size else 0.0,
            svals_zideal=s.tolist(),
            A=A, Pi=Pi,  # kept for the depth-curve step below (not printed raw)
        )
        per_example.append(rec)

    out = dict(
        path=path,
        K=K,
        d=dim,
        force_rank_k=force_rank_k,
        seed=seed,
        n_skipped_steps=d.get("n_skipped_steps"),
        M2=d.get("M2_in_distribution", {}),
        M3=d.get("M3_held_out", {}),
        per_example=per_example,
        raw=d,
    )
    return out


def depth_curve_for_run(run: dict):
    """For each example, build synthetic keys from Pi and predict the full
    depth-decay curve from A's spectrum; average across examples."""
    hops = sorted(int(h) for h in list(run["M2"].keys()) + list(run["M3"].keys()))
    all_curves = []
    consistency = []
    for rec in run["per_example"]:
        A, Pi = rec["A"], rec["Pi"]
        keys, resid, imag_leak = synthetic_keys_from_pi(Pi)
        consistency.append((resid, imag_leak))
        curve = predicted_cos_curve(A, Pi, keys, hops)
        all_curves.append(curve)
    mean_curve = {h: float(np.mean([c[h] for c in all_curves])) for h in hops}
    return mean_curve, consistency


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def print_report(runs: list):
    print("=" * 100)
    print("TASK E Z-DUMP SUBSPACE ANALYSIS")
    print("=" * 100)

    for run in runs:
        name = os.path.basename(run["path"])
        print()
        print("-" * 100)
        tag = f"fr={run['force_rank_k']}" if run["force_rank_k"] else "frN (unconstrained)"
        print(f"{name}  [K={run['K']} d={run['d']} seed={run['seed']} {tag} "
              f"n_skipped_steps={run['n_skipped_steps']}]")
        print("-" * 100)

        n_ex = len(run["per_example"])
        ke = [r["k_eff"] for r in run["per_example"]]
        pang = [r["principal_angle_max_deg"] for r in run["per_example"]]
        apirel = [r["A_minus_Pi_rel"] for r in run["per_example"]]
        cstar = [r["c_star"] for r in run["per_example"]]
        acpirel = [r["A_minus_cPi_rel"] for r in run["per_example"]]
        aeff = [r["A_eff_rank"] for r in run["per_example"]]
        eigresid = [r["eig_A_resid_max"] for r in run["per_example"]]
        eigphaseresid = [r["eig_A_phase_resid_max"] for r in run["per_example"]]
        normB = [r["normB"] for r in run["per_example"]]
        normC = [r["normC"] for r in run["per_example"]]
        normD = [r["normD"] for r in run["per_example"]]
        normZ = [r["normZ"] for r in run["per_example"]]
        drho = [r["D_spectral_radius"] for r in run["per_example"]]
        deff = [r["D_eff_rank"] for r in run["per_example"]]
        dcond = [r["D_condition_number"] for r in run["per_example"]]

        print(f"  n_examples={n_ex}")
        print(f"  k_eff (should==K={run['K']}): {ke}")
        print(f"  max principal angle U(col) vs Vrow(row), deg (0=identical subspaces): "
              f"{[fmt(x,3) for x in pang]}")
        print(f"  ||A - Pi||_F / ||Pi||_F (restricted op vs exact cycle, RAW incl. magnitude): "
              f"{[fmt(x,5) for x in apirel]}  mean={fmt(np.mean(apirel),5)}")
        print(f"  best-fit isotropic scale c* = argmin_c||A-c*Pi|| (cosine-invisible DOF): "
              f"{[fmt(x,4) for x in cstar]}  mean={fmt(np.mean(cstar),4)}")
        print(f"  ||A - c*Pi||_F / ||c*Pi||_F (SCALE-CORRECTED: pure phase/structure residual): "
              f"{[fmt(x,5) for x in acpirel]}  mean={fmt(np.mean(acpirel),5)}")
        print(f"  effective_rank(A) (target K={run['K']}): "
              f"{[fmt(x,3) for x in aeff]}  mean={fmt(np.mean(aeff),3)}")
        print(f"  eig(A) vs {run['K']}th-roots-of-unity, max residual RAW (Hungarian-matched): "
              f"{[fmt(x,5) for x in eigresid]}  mean={fmt(np.mean(eigresid),5)}")
        print(f"  eig(A) vs {run['K']}th-roots-of-unity, max residual PHASE-ONLY (magnitude-normalized): "
              f"{[fmt(x,5) for x in eigphaseresid]}  mean={fmt(np.mean(eigphaseresid),5)}")
        print(f"  ||B||_F (E-perp->E leakage): {[fmt(x,4) for x in normB]}  mean={fmt(np.mean(normB),4)}")
        print(f"  ||C||_F (E->E-perp leakage): {[fmt(x,4) for x in normC]}  mean={fmt(np.mean(normC),4)}")
        print(f"  ||D||_F (complement/leftover): {[fmt(x,4) for x in normD]}  mean={fmt(np.mean(normD),4)}")
        print(f"  ||Z||_F (whole matrix, for scale): {[fmt(x,4) for x in normZ]}  mean={fmt(np.mean(normZ),4)}")
        print(f"  spectral_radius(D): {[fmt(x,4) for x in drho]}  mean={fmt(np.mean(drho),4)}")
        print(f"  effective_rank(D) (out of {run['d']-run['K']}): {[fmt(x,3) for x in deff]}  mean={fmt(np.mean(deff),3)}")
        print(f"  condition_number(D) (top/bottom svals; ~1=flat/near-isometry, >>1=peaked/noisy): "
              f"{[fmt(x,3) for x in dcond]}  mean={fmt(np.mean(dcond),3)}")

        # per-example eigenvalue residual breakdown (which mode is worst),
        # phase-only (magnitude-normalized) so a uniform scale doesn't hide
        # a genuinely missing/attenuated mode or falsely flag a healthy one.
        print(f"  per-example phase-only eigenmode residuals (sorted desc; a lone outlier = missing mode):")
        for i, r in enumerate(run["per_example"]):
            resids = r["eig_A_phase_resid"]
            mags = r["eig_A_mag"]
            order = np.argsort(resids)[::-1]
            print(f"    ex{i}: phase_resid(desc)={[round(resids[j],5) for j in order]}")
            print(f"          |eig| of same modes  ={[round(mags[j],4) for j in order]}")

        # depth curve, predicted vs measured
        curve, consistency = depth_curve_for_run(run)
        cres = [c[0] for c in consistency]
        cimag = [c[1] for c in consistency]
        print(f"  synthetic-key self-consistency ||Pi@a_i - a_(i+1)|| (validity of the")
        print(f"    no-raw-keys derivation; should be ~0): mean={fmt(np.mean(cres),6)}  "
              f"max_imag_leak={fmt(np.max(cimag),6)}")
        print(f"  {'hop':>5} {'eff_hop':>8} {'predicted_cos(h)':>18} {'measured_mean_cos':>18} "
              f"{'measured_recov@0.9':>20} {'delta(meas-pred)':>18}")
        for h in sorted(curve.keys()):
            entry = run["M2"].get(str(h)) or run["M3"].get(str(h))
            meas_cos = entry["mean_cos"] if entry else float("nan")
            meas_rec = entry["recovered_frac@0.9"] if entry else float("nan")
            eff_hop = entry["effective_hop"] if entry else (h % run["K"])
            pred = curve[h]
            print(f"  {h:>5} {eff_hop:>8} {fmt(pred,5):>18} {fmt(meas_cos,5):>18} "
                  f"{fmt(meas_rec,5):>20} {fmt(meas_cos-pred,5):>18}")

    print()
    print("=" * 100)
    print("SUMMARY TABLE (means across the 4 eval examples per run)")
    print("=" * 100)
    header = (f"{'run':<20} {'k_eff':>6} {'c*':>7} {'A-cPi_rel':>10} {'phase_resid':>12} "
              f"{'||B||':>8} {'||C||':>8} {'||D||':>8} {'rho(D)':>8} {'eff_rk(D)':>10} {'eff_rk(A)':>10} {'cond(D)':>9}")
    print(header)
    for run in runs:
        name = os.path.basename(run["path"]).replace("t1_matrix_permutation_", "").replace(".json", "")
        cstar = np.mean([r["c_star"] for r in run["per_example"]])
        acpirel = np.mean([r["A_minus_cPi_rel"] for r in run["per_example"]])
        eigphaseresid = np.mean([r["eig_A_phase_resid_max"] for r in run["per_example"]])
        normB = np.mean([r["normB"] for r in run["per_example"]])
        normC = np.mean([r["normC"] for r in run["per_example"]])
        normD = np.mean([r["normD"] for r in run["per_example"]])
        drho = np.mean([r["D_spectral_radius"] for r in run["per_example"]])
        deff = np.mean([r["D_eff_rank"] for r in run["per_example"]])
        aeff = np.mean([r["A_eff_rank"] for r in run["per_example"]])
        dcond = np.mean([r["D_condition_number"] for r in run["per_example"]])
        ke = run["per_example"][0]["k_eff"]
        print(f"{name:<20} {ke:>6} {fmt(cstar,3):>7} {fmt(acpirel,5):>10} {fmt(eigphaseresid,5):>12} "
              f"{fmt(normB,4):>8} {fmt(normC,4):>8} {fmt(normD,4):>8} {fmt(drho,4):>8} "
              f"{fmt(deff,3):>10} {fmt(aeff,3):>10} {fmt(dcond,3):>9}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="Directory of Task E run JSONs with embedded Z_dump")
    ap.add_argument("--tol-ratio", type=float, default=1e-3,
                     help="Singular-value threshold (relative to s[0]) for K_eff detection")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    runs = []
    for p in paths:
        r = analyze_run(p, tol_ratio=args.tol_ratio)
        if r is not None:
            runs.append(r)
    if not runs:
        print(f"No runs with Z_dump found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print_report(runs)


if __name__ == "__main__":
    main()
