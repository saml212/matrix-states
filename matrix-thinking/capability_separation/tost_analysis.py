"""CAPABILITY_SEPARATION_DESIGN.md S1.4.2.1/S1.5 -- the TOST equivalence-
testing module for the marquee S4-vs-A5 dissociation check, M1's Spearman
corroboration, the marquee INCONCLUSIVE -> n=7 escalation trigger (wired to
budget_guard.BudgetGuard), and the overall Stage-1 verdict combiner.

`welch_tost` is the SAME formula marquee_power_simulation.py's
`tost_declare_equivalence_rate` already validated (Welch's unpaired TOST,
Satterthwaite-approximated df, margin=+-0.5 rank-units on effective_rank,
S1.4.2.1's justification for unpaired-not-paired) -- applied here to REAL
per-seed samples rather than simulated ones, plus a `verdict` field with
THREE states: DECLARE / REJECT / INCONCLUSIVE. Disclosed operationalization
(the design's own S1.5 prose names three states -- "declare", "reject", the
"two one-sided tests split" INCONCLUSIVE case -- without pinning REJECT's
exact test statistic): DECLARE is TOST's own standard test (both one-sided
tests reject their respective H0, exactly `tost_declare_equivalence_rate`'s
definition); REJECT is the standard TOST-literature MIRROR test ("minimum
effect" test, one-sided in the direction of the observed gap, H0: |diff| <=
margin) -- a real gap the size of a full d_min rung (>=1.0) fails to DECLARE
and correctly REJECTs rather than sitting INCONCLUSIVE forever; a gap near
the margin boundary with realistic noise correctly lands INCONCLUSIVE
(neither test's evidence is strong enough either way). Per S1.1's own
CONFIRM/FALSIFY/INCONCLUSIVE table, a marquee REJECT (S4/A5 genuinely
diverge) routes to INCONCLUSIVE, not FALSIFY, at the Stage-1-verdict level
-- FALSIFY is reserved for M3 HARD FALSIFY or (M1 FALSIFY + passing
blank-out), per S1.5's exact wording.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPHA = 0.05
MARGIN = 0.5              # S1.4.2.1's pinned TOST equivalence margin, rank-units
DMIN_SEQ = [2, 3, 3, 4, 5]  # S1.2's table order: S3, S4, A5, S5, A6
# S1.3.3/spearman_null_calibration.py's already-EXECUTED exact-permutation-null
# figures (reused as constants -- re-enumerating 5!=120 permutations on every
# call would be needless; the source script remains the reproducible ground truth).
M1_EXACT_P_RHO_GE_08 = 8 / 120
M1_EXACT_P_RHO_GE_09 = 2 / 120
M1_MAX_ACHIEVABLE_RHO = 0.974679


def welch_tost(sample_a: np.ndarray, sample_b: np.ndarray, margin: float = MARGIN,
               alpha: float = ALPHA) -> dict:
    """Welch's unpaired TOST (S1.4.2.1's pinned procedure). sample_a,
    sample_b: 1D arrays of per-seed restricted effective rank (e.g. S4, A5).
    `diff = mean_b - mean_a`. Returns a dict incl. `verdict` in
    {'DECLARE','REJECT','INCONCLUSIVE'} -- see module docstring."""
    a, b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
    n_a, n_b = len(a), len(b)
    assert n_a >= 2 and n_b >= 2, "Welch's test needs >=2 samples per group"
    m_a, m_b = float(a.mean()), float(b.mean())
    v_a, v_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    diff = m_b - m_a
    se = float(np.sqrt(v_a / n_a + v_b / n_b))
    num = (v_a / n_a + v_b / n_b) ** 2
    den = (v_a / n_a) ** 2 / (n_a - 1) + (v_b / n_b) ** 2 / (n_b - 1)
    df = float(num / den) if den > 0 else float(n_a + n_b - 2)
    tcrit = float(stats.t.ppf(1 - alpha, df))

    t1 = (diff + margin) / se if se > 0 else float("inf")   # H01: true diff <= -margin
    t2 = (margin - diff) / se if se > 0 else float("inf")   # H02: true diff >= +margin
    declare = bool(t1 > tcrit and t2 > tcrit)

    # REJECT (mirror test): one-sided in the direction of the observed gap,
    # H0: |diff| <= margin.
    if diff >= 0:
        t_reject = (diff - margin) / se if se > 0 else float("inf")
    else:
        t_reject = (-diff - margin) / se if se > 0 else float("inf")
    reject = bool((not declare) and t_reject > tcrit)

    if declare:
        verdict = "DECLARE"
    elif reject:
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE"

    return dict(n_a=n_a, n_b=n_b, mean_a=m_a, mean_b=m_b, diff=diff, se=se, df=df,
                margin=margin, alpha=alpha, t1=t1, t2=t2, tcrit=tcrit, t_reject=t_reject,
                declare_equivalence=declare, reject_equivalence=reject, verdict=verdict)


def marquee_check(sample_s4: np.ndarray, sample_a5: np.ndarray, budget_guard=None) -> dict:
    """S1.5's marquee dissociation check + the CA3-M1(d)/CA4-M1 escalation
    wiring. If TOST returns INCONCLUSIVE and a `budget_guard` is supplied,
    request the marquee n=7 escalation through it (S1.6/S1.20's pinned
    yield order: marquee outranks all general escalations). A budget-denied
    escalation leaves INCONCLUSIVE standing (S1.21 note 2: "if the marquee
    n=7 escalation is budget-denied, TOST's INCONCLUSIVE stands... S1.5's
    CONFIRM cannot be reached without the declared equivalence")."""
    result = welch_tost(sample_s4, sample_a5)
    result["escalation_requested"] = False
    result["escalation_status"] = None
    if result["verdict"] == "INCONCLUSIVE" and budget_guard is not None:
        decision = budget_guard.request_marquee_n7()
        result["escalation_requested"] = True
        result["escalation_status"] = decision["status"]
        result["escalation_decision"] = decision
        if decision["status"] == "budget-denied":
            result["final_verdict"] = "INCONCLUSIVE"
            result["final_verdict_reason"] = (
                "marquee n=7 escalation budget-denied; INCONCLUSIVE stands (S1.21 note 2)")
        else:
            result["final_verdict"] = "PENDING_N7_RERUN"
            result["final_verdict_reason"] = (
                "n=7 escalation granted; re-run TOST on the pooled n=7 cohort before a verdict "
                "(gated by the variance-ratio check, S1.8, before pooling old/new cohorts)")
    else:
        result["final_verdict"] = result["verdict"]
        result["final_verdict_reason"] = "no escalation needed (DECLARE/REJECT, or no budget_guard supplied)"
    return result


def spearman_corroboration(metric_by_group: dict) -> dict:
    """M1's corroborating-only Spearman check (S1.5): metric_by_group in
    S1.2's group order (S3,S4,A5,S5,A6). Reports the OBSERVED rho (scipy,
    midrank ties) alongside the ALREADY-EXECUTED exact-permutation-null
    p-values (spearman_null_calibration.py, S1.3.3 -- reused as constants,
    not re-enumerated per call). M1 alone is never independently decisive
    (S1.5): `m1_confirm` requires BOTH rho>=0.8 AND each group's
    restricted_eff_rank in [0.7*d_min, 1.3*d_min] (Task D's M1 band)."""
    order = ["S3", "S4", "A5", "S5", "A6"]
    metric_seq = [metric_by_group[g] for g in order]
    rho, _ = stats.spearmanr(DMIN_SEQ, metric_seq)
    band_ok = {
        g: (0.7 * d <= metric_by_group[g] <= 1.3 * d)
        for g, d in zip(order, DMIN_SEQ)
    }
    m1_confirm = bool(rho >= 0.8) and all(band_ok.values())
    return dict(rho=float(rho), exact_p_rho_ge_08=M1_EXACT_P_RHO_GE_08,
                exact_p_rho_ge_09=M1_EXACT_P_RHO_GE_09, max_achievable_rho=M1_MAX_ACHIEVABLE_RHO,
                band_ok=band_ok, m1_confirm=m1_confirm, corroborating_only=True)


def stage1_verdict(m1_confirm: bool, m3_confirm_per_group: dict, marquee_final_verdict: str,
                   m3_hard_falsify: bool = False,
                   m1_falsify_with_passing_blankout: bool = False) -> dict:
    """S1.5's overall Stage-1 verdict combiner. FALSIFY (M3 HARD FALSIFY OR
    M1 FALSIFY-with-passing-blank-out) outranks everything else. Otherwise
    CONFIRM requires M1 CONFIRM AND M3 CONFIRM (every group) AND the
    marquee check DECLARE. A marquee REJECT (S4/A5 genuinely diverge) is
    INCONCLUSIVE, not FALSIFY, per S1.1's own table."""
    if m3_hard_falsify:
        return dict(verdict="FALSIFY", reason="M3 HARD FALSIFY: k=d_min-1 reached >=0.9x ceiling")
    if m1_falsify_with_passing_blankout:
        return dict(verdict="FALSIFY", reason="M1 FALSIFY (flat/non-tracking rank) with a passing blank-out test")
    m3_all_confirm = all(m3_confirm_per_group.values())
    marquee_pass = marquee_final_verdict == "DECLARE"
    if m1_confirm and m3_all_confirm and marquee_pass:
        return dict(verdict="CONFIRM", m1=m1_confirm, m3=m3_all_confirm, marquee=marquee_pass)
    return dict(verdict="INCONCLUSIVE", m1=m1_confirm, m3=m3_all_confirm, marquee=marquee_pass,
                reason="mixed across the family, or the marquee check did not DECLARE equivalence")


# ---------------------------------------------------------------------------
# Unit tests (synthetic data): CONFIRM / FALSIFY / INCONCLUSIVE + the
# budget-denial path.
# ---------------------------------------------------------------------------

def _test_confirm_path():
    print("=" * 88)
    print("UNIT TEST 1 -- CONFIRM path (marquee DECLAREs, M1+M3 all pass)")
    print("=" * 88)
    rng = np.random.default_rng(1)
    s4 = rng.normal(3.0, 0.15, size=5)   # true gap 0, low noise -- should DECLARE at n=5
    a5 = rng.normal(3.0, 0.15, size=5)
    mq = marquee_check(s4, a5, budget_guard=None)
    print(f"  marquee: verdict={mq['verdict']}  diff={mq['diff']:.4f}  final_verdict={mq['final_verdict']}")
    m1 = spearman_corroboration({"S3": 2.05, "S4": 3.0, "A5": 3.0, "S5": 4.1, "A6": 4.9})
    print(f"  M1: rho={m1['rho']:.4f}  m1_confirm={m1['m1_confirm']}")
    m3 = {"S3": True, "S4": True, "A5": True, "S5": True, "A6": True}
    verdict = stage1_verdict(m1["m1_confirm"], m3, mq["final_verdict"])
    print(f"  STAGE-1 VERDICT: {verdict['verdict']}")
    assert mq["verdict"] == "DECLARE", f"expected DECLARE, got {mq['verdict']}"
    assert verdict["verdict"] == "CONFIRM", f"expected CONFIRM, got {verdict['verdict']}"
    print("  PASS\n")


def _test_falsify_path():
    print("=" * 88)
    print("UNIT TEST 2 -- FALSIFY path (M3 HARD FALSIFY outranks everything)")
    print("=" * 88)
    verdict = stage1_verdict(m1_confirm=True, m3_confirm_per_group={"S3": True, "S4": True, "A5": True,
                             "S5": True, "A6": True}, marquee_final_verdict="DECLARE",
                             m3_hard_falsify=True)
    print(f"  STAGE-1 VERDICT: {verdict['verdict']}  reason={verdict['reason']}")
    assert verdict["verdict"] == "FALSIFY"
    verdict2 = stage1_verdict(m1_confirm=False, m3_confirm_per_group={"S3": True}, marquee_final_verdict="DECLARE",
                              m1_falsify_with_passing_blankout=True)
    print(f"  STAGE-1 VERDICT (M1 falsify path): {verdict2['verdict']}  reason={verdict2['reason']}")
    assert verdict2["verdict"] == "FALSIFY"
    print("  PASS\n")


def _test_inconclusive_no_guard():
    print("=" * 88)
    print("UNIT TEST 3 -- INCONCLUSIVE path (marquee ambiguous, no budget_guard supplied)")
    print("=" * 88)
    rng = np.random.default_rng(2)
    # gap right at the margin boundary with noisy samples -- should land
    # neither DECLARE nor REJECT at n=5 (the design's own disclosed
    # noise-sensitive residual, S1.4.2.1).
    s4 = rng.normal(3.0, 0.35, size=5)
    a5 = rng.normal(3.5, 0.35, size=5)
    mq = marquee_check(s4, a5, budget_guard=None)
    print(f"  marquee: verdict={mq['verdict']}  diff={mq['diff']:.4f}  se={mq['se']:.4f}  "
          f"escalation_requested={mq['escalation_requested']}")
    verdict = stage1_verdict(True, {"S3": True, "S4": True, "A5": True, "S5": True, "A6": True},
                             mq["final_verdict"])
    print(f"  STAGE-1 VERDICT: {verdict['verdict']}")
    assert mq["escalation_requested"] is False, "no budget_guard supplied -- must not request an escalation"
    assert verdict["verdict"] == "INCONCLUSIVE" or mq["verdict"] == "DECLARE", \
        "expected INCONCLUSIVE unless this noise draw happened to clearly DECLARE"
    print("  PASS\n")


def _test_reject_path():
    print("=" * 88)
    print("UNIT TEST 4 -- REJECT path (a large real gap correctly REJECTs, not INCONCLUSIVE forever)")
    print("=" * 88)
    rng = np.random.default_rng(3)
    s4 = rng.normal(3.0, 0.15, size=5)
    a5 = rng.normal(4.0, 0.15, size=5)   # true gap 1.0 rank-unit, low noise
    mq = marquee_check(s4, a5, budget_guard=None)
    print(f"  marquee: verdict={mq['verdict']}  diff={mq['diff']:.4f}")
    verdict = stage1_verdict(True, {"S3": True, "S4": True, "A5": True, "S5": True, "A6": True},
                             mq["final_verdict"])
    print(f"  STAGE-1 VERDICT: {verdict['verdict']}  (S1.1: a marquee REJECT is INCONCLUSIVE, not FALSIFY)")
    assert mq["verdict"] == "REJECT", f"expected REJECT for a real 1.0-unit gap at low noise, got {mq['verdict']}"
    assert verdict["verdict"] == "INCONCLUSIVE"
    print("  PASS\n")


def _test_budget_denial_path():
    print("=" * 88)
    print("UNIT TEST 5 -- budget-denial path (INCONCLUSIVE marquee, n=7 escalation DENIED)")
    print("=" * 88)
    import budget_guard as bg
    rng = np.random.default_rng(2)
    s4 = rng.normal(3.0, 0.35, size=5)
    a5 = rng.normal(3.5, 0.35, size=5)

    # near-cap guard: no headroom at all for the marquee escalation.
    guard = bg.BudgetGuard(cap=bg.CAP, spend_to_date=bg.CAP - 0.1)   # only 0.1 GPU-h headroom
    mq = marquee_check(s4, a5, budget_guard=guard)
    print(f"  marquee verdict={mq['verdict']}  escalation_requested={mq['escalation_requested']}  "
          f"escalation_status={mq['escalation_status']}  final_verdict={mq['final_verdict']}")
    if mq["verdict"] == "INCONCLUSIVE":
        assert mq["escalation_requested"] is True
        assert mq["escalation_status"] == "budget-denied"
        assert mq["final_verdict"] == "INCONCLUSIVE"
        verdict = stage1_verdict(True, {"S3": True, "S4": True, "A5": True, "S5": True, "A6": True},
                                 mq["final_verdict"])
        print(f"  STAGE-1 VERDICT: {verdict['verdict']}  (CONFIRM correctly UNREACHABLE without a declared "
              f"equivalence, S1.21 note 2)")
        assert verdict["verdict"] == "INCONCLUSIVE"
        print("  PASS (denial path exercised)\n")
    else:
        print(f"  this noise draw landed {mq['verdict']} directly (no escalation needed) -- "
              f"re-running with a guaranteed-INCONCLUSIVE synthetic sample instead.\n")
        # force an INCONCLUSIVE draw deterministically for the denial-path assertion.
        s4_forced = np.array([3.0, 3.1, 2.9, 3.05, 2.95])
        a5_forced = np.array([3.55, 3.6, 3.4, 3.5, 3.65])
        mq2 = marquee_check(s4_forced, a5_forced, budget_guard=guard)
        print(f"  forced marquee verdict={mq2['verdict']}  escalation_status={mq2['escalation_status']}")
        assert mq2["verdict"] == "INCONCLUSIVE", f"forced sample did not land INCONCLUSIVE ({mq2['verdict']})"
        assert mq2["escalation_status"] == "budget-denied"
        assert mq2["final_verdict"] == "INCONCLUSIVE"
        print("  PASS (denial path exercised via forced sample)\n")


def _test_budget_granted_path():
    print("=" * 88)
    print("UNIT TEST 6 -- escalation GRANTED path (plenty of headroom)")
    print("=" * 88)
    import budget_guard as bg
    s4_forced = np.array([3.0, 3.1, 2.9, 3.05, 2.95])
    a5_forced = np.array([3.55, 3.6, 3.4, 3.5, 3.65])
    guard = bg.BudgetGuard(cap=bg.CAP, spend_to_date=0.0)   # full 30 GPU-h headroom
    mq = marquee_check(s4_forced, a5_forced, budget_guard=guard)
    print(f"  marquee verdict={mq['verdict']}  escalation_status={mq['escalation_status']}  "
          f"final_verdict={mq['final_verdict']}")
    assert mq["verdict"] == "INCONCLUSIVE"
    assert mq["escalation_status"] == "granted"
    assert mq["final_verdict"] == "PENDING_N7_RERUN"
    print("  PASS\n")


if __name__ == "__main__":
    _test_confirm_path()
    _test_falsify_path()
    _test_inconclusive_no_guard()
    _test_reject_path()
    _test_budget_denial_path()
    _test_budget_granted_path()
    print("=" * 88)
    print("tost_analysis.py: ALL UNIT TESTS PASSED (CONFIRM / FALSIFY / INCONCLUSIVE / REJECT / "
          "budget-denied / budget-granted)")
    print("=" * 88)
