#!/usr/bin/env python3
"""
NCR waterfall gate §3.8(c) — EXECUTED negative test for the corrected trust
rule T(h) = (||C||_2/sigma_min(A)) * (a^h - r^h)/(a - r),
a := sigma_max(A)/sigma_min(A), r := sigma_max(D)/sigma_min(A).

Implements BOTH pinned cases exactly per
matrix-thinking/NOVEL_ARCH_WATERFALL.md:
  - §3.4  the shared construction (numpy default_rng(20260709), d=16, K=8,
          A = 1.0*Pi exact 8-cycle, B=0, q=e0, probe depth h=21, operator
          2-norm convention) + N1 (amplifying near-normal D).
  - §3.9 MA1  N2 (non-normal nilpotent D = 100*E01, deterministic, no RNG).
  - §5  the verified expected numbers this script's output is cross-checked
        against (N1: T_lin(21)=0.2100, corrected T(21)=99.74 [more precisely
        99.7377]; N2: rho-based T(21)=0.0100, corrected T(21)=1.010e38,
        junk/signal=1.0 exactly, cosine=0.7071).

Three binding nits from §5 honored explicitly:
  (n1) N2's e3 occupancy appears at steps 3, 11, AND 19 -- uniqueness of the
       composed path comes from D^2=0 annihilating the j=4/j=12 injections,
       NOT from "C fires only once". Asserted at all three steps below.
  (n2) N1's old-rule ADMIT is scored as an explicit inequality against the
       s1-calibrated threshold (>0.37), since 0.21 > tau=0.2 read literally
       REJECTS -- both comparisons are printed.
  (n3) the "4-10x conservative" figure is NEVER quoted against the Rev-2
       (sigma-form) rule anywhere in this script's language -- it quantifies
       T_lin only and is not computed here at all.

EXITS NONZERO on any check miss (teeth). No early exit -- both cases run to
completion regardless of individual check outcomes; the failure is reported
in aggregate at the end.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# macOS Accelerate BLAS (the backend numpy links against on this box, per
# np.show_config()) emits spurious 'divide by zero'/'overflow'/'invalid
# value encountered in matmul' RuntimeWarnings on ordinary matmuls of these
# small (16x16) well-conditioned matrices -- a known benign Accelerate
# artifact, verified below (see the `assert not nan/inf` guards on every
# Z^h q result, and the fast-vs-naive-loop matrix_power cross-check) to NOT
# indicate any actual NaN/Inf/precision loss in the computed results.
warnings.filterwarnings(
    "ignore", message=".*encountered in matmul", category=RuntimeWarning
)

# ---------------------------------------------------------------------------
# Pinned constants (§3.4 shared construction; NOTHING here is guessed)
# ---------------------------------------------------------------------------
SEED = 20260709
K = 8            # E dimension (the cycle length)
D_TOTAL = 2 * K  # d = 16, E xor E-perp each dimension K
H = 21           # probe depth
TAU = 0.2        # trust threshold (§3.4: "Trust threshold tau = 0.2 unchanged")
# s1's own measured T_lin(21) under the OLD linear rule (§3.4 "Restated
# example values"): the loosest calibration that still admits known-good,
# measured-exact behavior. Nit (n2) requires N1's old-rule ADMIT verdict be
# scored explicitly against this, not just against the literal tau=0.2 floor.
S1_CALIBRATED_THRESHOLD = 0.37

OUT_DIR = Path(__file__).resolve().parent / "test_trust_rule_negative"
SCRIPT_PATH = Path(__file__).resolve()

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(msg)


def check(checks_list, name, cond, detail):
    cond = bool(cond)
    checks_list.append({"name": name, "passed": cond, "detail": detail})
    status = "PASS" if cond else "FAIL"
    log(f"[{status}] {name}: {detail}")
    return cond


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------
def build_cycle(k):
    """Exact k-cycle permutation matrix Pi: Pi @ e_i = e_{(i+1) mod k}.
    Orthogonal -> sigma_min = sigma_max = 1 (matches §3.4: 'A = 1.0*Pi
    (exact 8-cycle, sigma_min = sigma_max = 1)')."""
    A = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        A[(i + 1) % k, i] = 1.0
    return A


def assemble_Z(A, C, D, k):
    """Block lower-triangular Z = [[A, 0], [C, D]] in the [E, E-perp] basis
    (B = 0 per §3.4's shared construction)."""
    n = 2 * k
    Z = np.zeros((n, n), dtype=np.float64)
    Z[0:k, 0:k] = A
    Z[k:n, 0:k] = C
    Z[k:n, k:n] = D
    return Z


def op_norm(M):
    """Operator 2-norm (largest singular value) -- the rule's pinned norm
    convention (§3.4: 'every norm in this rule is the OPERATOR 2-norm')."""
    return float(np.linalg.norm(M, ord=2))


def spectral_radius(M):
    eigs = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigs)))


def matrix_power_verified(Z, h):
    """np.linalg.matrix_power (binary exponentiation) cross-checked against
    a naive h-fold matmul loop -- guards against the Accelerate-BLAS
    spurious-warning path silently corrupting a result (it does not, but we
    verify rather than assume)."""
    fast = np.linalg.matrix_power(Z, h)
    naive = np.eye(Z.shape[0])
    for _ in range(h):
        naive = naive @ Z
    max_diff = float(np.max(np.abs(fast - naive)))
    finite = bool(np.all(np.isfinite(fast)) and np.all(np.isfinite(naive)))
    return fast, max_diff, finite


def trust_rule(a, r, C_over_sigmin, h):
    """The §3.4 corrected-rule functional form (used both for the Rev-1
    rho-form, by passing r = rho(D)/sigma_min(A), and the Rev-2 sigma-form,
    by passing r = sigma_max(D)/sigma_min(A) -- the two 'old vs corrected'
    comparisons the negative tests exercise)."""
    if abs(a - r) < 1e-6:
        m = max(a, r)
        return C_over_sigmin * h * (m ** (h - 1))
    return C_over_sigmin * (a ** h - r ** h) / (a - r)


def linear_rule(C_over_sigmin, h):
    """The doubly-degenerate a=1,r=1 limit -- '§3.4: the Rev-1 linear rule
    is exactly this doubly-degenerate limit', T_lin(h) = h*||C||/sigma_min(A)."""
    return h * C_over_sigmin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = []
    log("=" * 78)
    log("NCR waterfall §3.8(c) negative test -- N1 (amplifying near-normal D)")
    log("                                    + N2 (non-normal nilpotent D)")
    log(f"seed={SEED}  K={K}  d={D_TOTAL}  h={H}  tau={TAU}")
    log("=" * 78)

    # --- shared construction: A = 1.0*Pi exact K-cycle, q = e0 -------------
    A = build_cycle(K)
    sigma_A = np.linalg.svd(A, compute_uv=False)
    sigma_min_A, sigma_max_A = float(sigma_A.min()), float(sigma_A.max())
    a = sigma_max_A / sigma_min_A
    check(checks, "shared: A is exact 8-cycle, sigma_min=sigma_max=1",
          np.isclose(sigma_min_A, 1.0) and np.isclose(sigma_max_A, 1.0),
          f"sigma_min(A)={sigma_min_A}, sigma_max(A)={sigma_max_A}, a={a}")

    q = np.zeros(D_TOTAL, dtype=np.float64)
    q[0] = 1.0  # q = e0

    log("")
    log("-" * 78)
    log("N1: amplifying near-normal D (C Gaussian rescaled to ||C||_2=0.01;")
    log("    D = 1.5*Q, Q = QR-orthogonalization of a Gaussian draw)")
    log("-" * 78)

    rng = np.random.default_rng(SEED)
    C_raw_n1 = rng.standard_normal((K, K))
    C_n1 = C_raw_n1 / op_norm(C_raw_n1) * 0.01
    Q_raw_n1 = rng.standard_normal((K, K))
    Q_n1, _ = np.linalg.qr(Q_raw_n1)
    D_n1 = 1.5 * Q_n1

    sigma_D_n1 = op_norm(D_n1)
    rho_D_n1 = spectral_radius(D_n1)
    r_n1 = sigma_D_n1 / sigma_min_A
    C_over_sigmin_n1 = op_norm(C_n1) / sigma_min_A

    check(checks, "N1 construction: ||C||_2 == 0.01 exactly",
          np.isclose(op_norm(C_n1), 0.01, atol=1e-9), f"{op_norm(C_n1)!r}")
    check(checks, "N1 construction: D=1.5*Q near-normal, rho(D)=sigma_max(D)=1.5",
          np.isclose(rho_D_n1, 1.5, atol=1e-6) and np.isclose(sigma_D_n1, 1.5, atol=1e-6),
          f"rho(D)={rho_D_n1}, sigma_max(D)={sigma_D_n1}")

    T_lin_n1 = linear_rule(C_over_sigmin_n1, H)
    T_corrected_n1 = trust_rule(a, r_n1, C_over_sigmin_n1, H)

    Z_n1 = assemble_Z(A, C_n1, D_n1, K)
    Z_n1_pow, n1_pow_diff, n1_pow_finite = matrix_power_verified(Z_n1, H)
    check(checks, "N1 Z^21 fast (binary-exp) matches naive h-fold loop, both finite",
          n1_pow_finite and n1_pow_diff < 1e-6,
          f"max|diff|={n1_pow_diff}, finite={n1_pow_finite} "
          "(Accelerate-BLAS spurious matmul RuntimeWarning verified benign)")
    Zh_q_n1 = Z_n1_pow @ q
    signal_n1 = float(np.linalg.norm(Zh_q_n1[:K]))
    junk_n1 = float(np.linalg.norm(Zh_q_n1[K:]))
    ratio_n1 = junk_n1 / signal_n1
    cosine_n1 = signal_n1 / float(np.linalg.norm(Zh_q_n1))
    recovery_n1 = 1.0 if cosine_n1 >= 0.9 else 0.0

    log(f"  T_lin(21)      = {T_lin_n1:.6f}")
    log(f"  T_corrected(21)= {T_corrected_n1:.6f}")
    log(f"  junk/signal    = {ratio_n1:.6f}   cosine={cosine_n1:.6f}  recovery@0.9={recovery_n1}")

    # (n2) explicit calibrated-threshold inequality, printed both ways
    old_rule_admits_calibrated = T_lin_n1 < S1_CALIBRATED_THRESHOLD
    old_rule_rejects_literal_tau = T_lin_n1 > TAU
    log(f"  [n2] old-rule vs s1-calibrated threshold: T_lin(21)={T_lin_n1:.4f} < "
        f"{S1_CALIBRATED_THRESHOLD} -> {'ADMIT' if old_rule_admits_calibrated else 'REJECT'}")
    log(f"  [n2] old-rule vs literal tau={TAU}: T_lin(21)={T_lin_n1:.4f} > {TAU} -> "
        f"{'REJECT (literal reading)' if old_rule_rejects_literal_tau else 'ADMIT'}")

    check(checks, "N1 T_lin(21) in [0.20, 0.22] (pinned pass criterion)",
          0.20 <= T_lin_n1 <= 0.22, f"{T_lin_n1}")
    check(checks, "N1 T_lin(21) matches §5 recorded 0.2100 (cross-check)",
          np.isclose(T_lin_n1, 0.2100, atol=1e-4), f"{T_lin_n1}")
    check(checks, "N1 (n2) old rule ADMITS under s1-calibrated threshold 0.37",
          old_rule_admits_calibrated, f"T_lin={T_lin_n1:.4f} < {S1_CALIBRATED_THRESHOLD}")
    check(checks, "N1 corrected T(21) > 10 (pinned pass criterion)",
          T_corrected_n1 > 10, f"{T_corrected_n1}")
    check(checks, "N1 corrected T(21) matches §5 recorded 99.74 (cross-check, precise 99.7377)",
          np.isclose(T_corrected_n1, 99.7377, atol=0.01), f"{T_corrected_n1}")
    check(checks, "N1 corrected rule REJECTS at tau=0.2 (T(21) > tau)",
          T_corrected_n1 > TAU, f"{T_corrected_n1} > {TAU}")
    check(checks, "N1 measured junk/signal > 1 (pinned pass criterion)",
          ratio_n1 > 1.0, f"{ratio_n1}")

    log("")
    log("-" * 78)
    log("N2: non-normal nilpotent D = 100*E01 (matrix unit e0 e1^T on the")
    log("    E-perp block); C = 0.01*e1^perp e3^T -- fully deterministic, no RNG")
    log("-" * 78)

    # e1^perp e3^T: matrix unit at (row=1 [E-perp-local index for f1],
    # col=3 [E-local index for e3]) in the 8x8 C block.
    C_n2 = np.zeros((K, K), dtype=np.float64)
    C_n2[1, 3] = 0.01
    # D = 100*E01: matrix unit at (row=0, col=1) in the 8x8 D block.
    D_n2 = np.zeros((K, K), dtype=np.float64)
    D_n2[0, 1] = 100.0

    sigma_D_n2 = op_norm(D_n2)
    rho_D_n2 = spectral_radius(D_n2)
    r_rho_n2 = rho_D_n2 / sigma_min_A
    r_sigma_n2 = sigma_D_n2 / sigma_min_A
    C_over_sigmin_n2 = op_norm(C_n2) / sigma_min_A

    check(checks, "N2 construction: ||C||_2 == 0.01 exactly",
          np.isclose(op_norm(C_n2), 0.01, atol=1e-12), f"{op_norm(C_n2)!r}")
    check(checks, "N2 construction: D nilpotent, rho(D)=0, sigma_max(D)=100",
          np.isclose(rho_D_n2, 0.0, atol=1e-9) and np.isclose(sigma_D_n2, 100.0, atol=1e-9),
          f"rho(D)={rho_D_n2}, sigma_max(D)={sigma_D_n2}")

    # (n1) occupancy check: A^s @ q occupies e3 (E-local index 3) at ALL of
    # s=3, 11, AND 19 -- NOT just once. This is the actual source of the
    # single surviving injection term, per §5's nit: uniqueness comes from
    # D^2=0 annihilating the j=4/j=12 injections, not from "C fires once".
    e3_local = np.zeros(K, dtype=np.float64)
    e3_local[3] = 1.0
    occupancy = {}
    for s in (3, 11, 19):
        As_q = np.linalg.matrix_power(A, s) @ q[:K]
        occupancy[s] = bool(np.allclose(As_q, e3_local))
    check(checks, "N2 (n1) e3 occupancy at steps 3, 11, AND 19 (all three)",
          all(occupancy.values()), f"occupancy={occupancy}")

    D2_n2 = D_n2 @ D_n2
    max_abs_D2 = float(np.max(np.abs(D2_n2)))
    check(checks, "N2 D^2 = 0 (nilpotent -- the actual annihilation mechanism)",
          np.allclose(D2_n2, 0.0, atol=1e-12), f"max|D^2|={max_abs_D2}")

    def injection_term(j):
        """D^(h-j) . C . A^(j-1) q -- the leakage injected at application j,
        propagated through the remaining h-j applications of D."""
        Aj1_q = np.linalg.matrix_power(A, j - 1) @ q[:K]
        Cx = C_n2 @ Aj1_q
        Dpow = np.linalg.matrix_power(D_n2, H - j)
        return Dpow @ Cx

    term_j4 = injection_term(4)   # s=3  (j-1=3), h-j=17 -> D^17, annihilated (>=2)
    term_j12 = injection_term(12)  # s=11 (j-1=11), h-j=9  -> D^9,  annihilated (>=2)
    term_j20 = injection_term(20)  # s=19 (j-1=19), h-j=1  -> D^1,  SURVIVES

    check(checks, "N2 j=4 injection term annihilated by D^17=0 (D^2=0 mechanism)",
          np.allclose(term_j4, 0.0, atol=1e-9), f"term={term_j4}")
    check(checks, "N2 j=12 injection term annihilated by D^9=0 (D^2=0 mechanism)",
          np.allclose(term_j12, 0.0, atol=1e-9), f"term={term_j12}")
    check(checks, "N2 j=20 injection term SURVIVES (D^1 . C . A^19 q)",
          not np.allclose(term_j20, 0.0, atol=1e-9), f"term={term_j20}")

    T_rho_n2 = trust_rule(a, r_rho_n2, C_over_sigmin_n2, H)       # OLD (Rev-1) rho-based rule
    T_sigma_n2 = trust_rule(a, r_sigma_n2, C_over_sigmin_n2, H)   # corrected (Rev-2) sigma-based rule

    Z_n2 = assemble_Z(A, C_n2, D_n2, K)
    Z_n2_pow, n2_pow_diff, n2_pow_finite = matrix_power_verified(Z_n2, H)
    check(checks, "N2 Z^21 fast (binary-exp) matches naive h-fold loop, both finite",
          n2_pow_finite and n2_pow_diff < 1e-6,
          f"max|diff|={n2_pow_diff}, finite={n2_pow_finite} "
          "(Accelerate-BLAS spurious matmul RuntimeWarning verified benign)")
    Zh_q_n2 = Z_n2_pow @ q
    signal_n2 = float(np.linalg.norm(Zh_q_n2[:K]))
    junk_n2 = float(np.linalg.norm(Zh_q_n2[K:]))
    ratio_n2 = junk_n2 / signal_n2
    cosine_n2 = signal_n2 / float(np.linalg.norm(Zh_q_n2))
    recovery_n2 = 1.0 if cosine_n2 >= 0.9 else 0.0

    log(f"  T_rho(21) [OLD, Rev-1]     = {T_rho_n2:.6f}")
    log(f"  T_sigma(21) [corrected]    = {T_sigma_n2:.6e}")
    log(f"  junk/signal = {ratio_n2:.6f}   cosine={cosine_n2:.6f}  recovery@0.9={recovery_n2}")

    # cross-check: analytic recursion (sum of the three surviving-or-not
    # injection terms) matches the direct brute-force Z^21 q computation.
    analytic_E_perp = term_j4 + term_j12 + term_j20
    diff_analytic = float(np.max(np.abs(analytic_E_perp - Zh_q_n2[K:])))
    check(checks, "N2 analytic recursion (3-term sum) matches direct Z^21 q E-perp component",
          diff_analytic < 1e-9, f"max|diff|={diff_analytic}")

    check(checks, "N2 rho-based (OLD) T(21) <= 0.011 (pinned pass criterion)",
          T_rho_n2 <= 0.011, f"{T_rho_n2}")
    check(checks, "N2 rho-based T(21) matches §5 recorded 0.0100 -- falsely ADMITS (cross-check)",
          np.isclose(T_rho_n2, 0.01, atol=1e-6), f"{T_rho_n2}")
    check(checks, "N2 rho-based rule falsely ADMITS at tau=0.2 (T(21) <= tau)",
          T_rho_n2 <= TAU, f"{T_rho_n2} <= {TAU}")
    check(checks, "N2 corrected (sigma) T(21) > 1e30 (pinned pass criterion)",
          T_sigma_n2 > 1e30, f"{T_sigma_n2:.6e}")
    check(checks, "N2 corrected T(21) matches §5 recorded 1.010e38 (cross-check)",
          np.isclose(T_sigma_n2, 1.010e38, rtol=0.01), f"{T_sigma_n2:.6e}")
    check(checks, "N2 corrected rule REJECTS at tau=0.2 (T(21) > tau)",
          T_sigma_n2 > TAU, f"{T_sigma_n2:.6e} > {TAU}")
    check(checks, "N2 measured junk/signal in [0.9, 1.1] (pinned pass criterion)",
          0.9 <= ratio_n2 <= 1.1, f"{ratio_n2}")
    check(checks, "N2 measured junk/signal == 1.0 exactly (deterministic construction, cross-check)",
          np.isclose(ratio_n2, 1.0, atol=1e-9), f"{ratio_n2}")
    check(checks, "N2 cosine matches §5 recorded 0.7071 = 1/sqrt(2) (cross-check)",
          np.isclose(cosine_n2, 1.0 / np.sqrt(2.0), atol=1e-6), f"{cosine_n2}")

    log("")
    log("=" * 78)
    n_pass = sum(1 for c in checks if c["passed"])
    n_total = len(checks)
    all_ok = n_pass == n_total
    log(f"RESULT: {n_pass}/{n_total} checks passed. Overall: {'PASS' if all_ok else 'FAIL'}")
    log("=" * 78)

    results = {
        "script": "matrix-thinking/chapter2/test_trust_rule_negative.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "K": K,
        "d_total": D_TOTAL,
        "h": H,
        "tau": TAU,
        "s1_calibrated_threshold": S1_CALIBRATED_THRESHOLD,
        "shared": {
            "sigma_min_A": sigma_min_A,
            "sigma_max_A": sigma_max_A,
            "a": a,
        },
        "N1": {
            "description": "amplifying near-normal D (C Gaussian ||C||_2=0.01, D=1.5*Q orthogonal)",
            "sigma_max_D": sigma_D_n1,
            "rho_D": rho_D_n1,
            "r": r_n1,
            "C_norm": op_norm(C_n1),
            "T_lin_21": T_lin_n1,
            "T_corrected_21": T_corrected_n1,
            "old_rule_admits_vs_s1_calibrated_0.37": old_rule_admits_calibrated,
            "old_rule_rejects_vs_literal_tau_0.2": old_rule_rejects_literal_tau,
            "corrected_rule_rejects_at_tau": bool(T_corrected_n1 > TAU),
            "junk_signal_ratio": ratio_n1,
            "cosine": cosine_n1,
            "recovery_at_0.9": recovery_n1,
            "expected_ref_S5": {"T_lin_21": 0.2100, "T_corrected_21": 99.7377},
        },
        "N2": {
            "description": "non-normal nilpotent D=100*E01, C=0.01*e1perp e3^T, deterministic (no RNG)",
            "sigma_max_D": sigma_D_n2,
            "rho_D": rho_D_n2,
            "r_rho": r_rho_n2,
            "r_sigma": r_sigma_n2,
            "C_norm": op_norm(C_n2),
            "e3_occupancy_steps_3_11_19": occupancy,
            "D_squared_is_zero": bool(np.allclose(D2_n2, 0.0, atol=1e-12)),
            "max_abs_D_squared": max_abs_D2,
            "injection_term_j4_norm": float(np.linalg.norm(term_j4)),
            "injection_term_j12_norm": float(np.linalg.norm(term_j12)),
            "injection_term_j20_norm": float(np.linalg.norm(term_j20)),
            "T_rho_21_OLD": T_rho_n2,
            "T_sigma_21_corrected": T_sigma_n2,
            "rho_rule_falsely_admits_at_tau": bool(T_rho_n2 <= TAU),
            "sigma_rule_rejects_at_tau": bool(T_sigma_n2 > TAU),
            "junk_signal_ratio": ratio_n2,
            "cosine": cosine_n2,
            "recovery_at_0.9": recovery_n2,
            "expected_ref_S5": {"T_rho_21": 0.0100, "T_sigma_21": 1.010e38,
                                "junk_signal_ratio": 1.0, "cosine": 0.7071},
        },
        "checks": checks,
        "n_checks_passed": n_pass,
        "n_checks_total": n_total,
        "overall_pass": all_ok,
    }

    return all_ok, results


if __name__ == "__main__":
    ok, results = main()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (OUT_DIR / "run.log").write_text("\n".join(_log_lines) + "\n")

    if not ok:
        sys.exit(1)
    sys.exit(0)
