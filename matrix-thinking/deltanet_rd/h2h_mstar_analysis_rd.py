"""h2h_mstar_analysis_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.4.2's axis-2
`M*` DESCENDING fixed-sequence gatekeeping test (R3-F2), its straddle-
triggered sec 1.8 seed-extension hook, and the `CONFIRMED no-crossover` /
`INDETERMINATE` split. Pure statistics over CI dicts -- no torch, no GPU, no
backbone -- fully unit-testable with synthetic data, which is exactly how
this module's own smoke suite exercises it (round-4's own "all four edge
cases hand-walked" verdict, reproduced here as REAL, runnable test cases,
not merely narrated).

Procedure, pinned exactly (sec 1.4.2(a)):
  Walk M in DESCENDING order (32 -> 16 -> 8 -> 4 -> 2, the 5 ELIGIBLE grid
  points -- M=1 is floor-excluded, R3-F3). At each M, test H_M: "the
  contender's gap over (b-primary) at the primary horizon (H4) is >= 0.20"
  vs. the one-sided alternative "< 0.20", via the SAME paired `delta_ci_n`
  CI (sec 1.8). REJECT H_M (crossover established at this M -- the
  Transformer has caught up within margin) iff the CI sits ENTIRELY below
  0.20 (`ci_high < margin`). Continue to the next SMALLER eligible M only
  if the CURRENT M was rejected; STOP at the first M that is NOT rejected
  (either a CLEAN non-rejection, `ci_low > margin`, or a STRADDLE,
  `ci_low <= margin <= ci_high`). `M* = ` the smallest M in the resulting
  unbroken chain of rejections (informally: the smallest memory multiplier
  at which the Transformer still managed to close the gap) -- a LARGER M*
  is BETTER for the contender (the baseline needed more memory to catch
  up), matching the tier table's own WIN-if-M*>=8 direction.

Edge cases (sec 1.4.2(c), all four independently hand-walked by attack
round 4, reproduced here as runnable smoke items, not narrated):
  - Grid exhaustion (every eligible M rejected, even M=2) -> M*=2 -> LOSE.
  - A clean stop partway down (e.g. rejected at 32/16/8, clean stop at 4)
    -> M* = 8 (the last rejected M) -> WIN.
  - An IMMEDIATE clean non-rejection at M=32 (chain is empty) -> the M*=inf
    pathway -> WIN, but the STRONGEST `CONFIRMED no-crossover` sub-claim
    requires the FULL eligible grid tested and clean (never the
    cost-saving shortcut) -- `full_grid_tested` is checked explicitly, not
    assumed from an empty chain alone.
  - A STRADDLE at the deciding boundary M -> triggers sec 1.8's EXISTING
    seed-extension contingency for THAT SPECIFIC (task, M) cell only
    (`resolve_mstar_with_extension` below) -> `INDETERMINATE` if still
    unresolved after extension, never a WIN.

Run the smoke gate: python h2h_mstar_analysis_rd.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub   # noqa: E402

_STUB_INSTALLED = ensure_fla_stub()           # reasoning_link_probe's module-level import chain
                                              # reaches lm_pretrain_rd -> fla; the stub makes the
                                              # pure-math delta_ci_n import CPU-safe (no-op on box)
from reasoning_link_probe import delta_ci_n   # noqa: E402 -- only pure math is USED here

MARGIN_DEFAULT = 0.20
ELIGIBLE_M_DESCENDING = (32, 16, 8, 4, 2)   # R3-F3: M=1 floor-excluded, never in this grid


def compute_ci_from_raw_seed_values(contender_vals: list[float], baseline_vals: list[float]) -> dict:
    """Wraps `delta_ci_n` DIRECTLY on RAW per-seed arrays -- "analysis must
    read raw artifacts, never intermediate summaries" (this design's own
    binding instruction): a caller must pass the actual per-seed `acc_A`
    readings (sec 1.31.1's re-registration of the per-M gap statistic from
    the rf-era `recovered_frac@0.9` to acc_A, same 0.20 margin), never a
    pre-averaged number.

    `zero_seed_variance` disclosure (the sec 1.40 sigma=0-at-ceiling note's
    analog for THIS machinery): if every paired per-seed delta is identical
    (e.g. all seeds at exact ceiling acc_A=1.0 against identical baseline
    readings), the t-CI degenerates to zero width and classify_cell becomes
    a point-classification -- flagged for the harvest to disclose, never
    silently passed through."""
    ci = delta_ci_n(contender_vals, baseline_vals)
    ci["zero_seed_variance"] = len(set(ci["deltas"])) == 1
    return ci


def classify_cell(ci: dict, margin: float = MARGIN_DEFAULT) -> str:
    """'reject' (ci_high < margin: crossover established -- the CI sits
    entirely below the margin), 'clean_non_reject' (ci_low > margin:
    confidently NO crossover), or 'straddle' (margin lies inside
    [ci_low, ci_high])."""
    if ci["ci_high"] < margin:
        return "reject"
    if ci["ci_low"] > margin:
        return "clean_non_reject"
    return "straddle"


def descending_walk(cis_by_m: dict, margin: float = MARGIN_DEFAULT,
                    tie_equivalent_ms: frozenset | set = frozenset()) -> dict:
    """cis_by_m: {M: delta_ci_n-shaped dict}, covering a PREFIX of
    ELIGIBLE_M_DESCENDING starting at 32 (the cost-saving shortcut may omit
    smaller M's below a clean stop -- see the grid-endpoint rule, sec
    1.4.2). Returns {'chain': [M,...] in test order (every REJECTED M),
    'stop_m': the M the walk stopped at (None if the tested prefix was
    exhausted without stopping), 'stop_classification':
    'clean_non_reject'|'straddle'|None, 'skipped_tie_equivalent': [M,...]}.

    `tie_equivalent_ms` -- the sec 1.31.1 joint-NO-RECALL rule (Rev 5.1,
    sec 1.32 M3, part of the frozen MARGINS_FROZEN registration): a grid
    point where BOTH the contender's acc_A AND the capped transformer's
    acc_A sit at or below the demonstration bar is scored TIE-EQUIVALENT
    for the walk -- NEVER a LOSE for the contender, never used to set M* in
    either direction (never appended to the rejection chain, never a stop);
    the walk continues past it, exactly as the straddle rule treats an
    unresolved CI (it does not finalize a tier at that point). The caller
    (harvest) computes the joint-failure classification from the raw
    per-seed acc_A readings; this walk only honors it."""
    chain: list[int] = []
    skipped: list[int] = []
    stop_m, stop_class = None, None
    for m in ELIGIBLE_M_DESCENDING:
        if m in tie_equivalent_ms:
            skipped.append(m)
            continue   # joint-NO-RECALL: walk continues past, cell can never set M*
        if m not in cis_by_m:
            break   # cost-saving shortcut: untested smaller M's below a clean stop
        cls = classify_cell(cis_by_m[m], margin)
        if cls == "reject":
            chain.append(m)
            continue
        stop_m, stop_class = m, cls
        break
    return {"chain": chain, "stop_m": stop_m, "stop_classification": stop_class,
            "skipped_tie_equivalent": skipped}


def _tier_for_finite_m_star(m_star: int) -> str:
    assert m_star in ELIGIBLE_M_DESCENDING, f"m_star={m_star} not on the eligible grid {ELIGIBLE_M_DESCENDING}"
    if m_star <= 2:
        return "LOSE"
    if m_star == 4:
        return "TIE"
    return "WIN"   # m_star >= 8


def finalize_verdict(walk: dict, tested_full_grid: bool) -> dict:
    """Maps a `descending_walk` result onto sec 1.4.2's verdict tiers.
    `tested_full_grid`: True iff cis_by_m covered ALL 5 eligible M's (the
    strongest `CONFIRMED no-crossover` sub-claim's own requirement --
    checked explicitly here, never inferred from an empty chain alone)."""
    chain, stop_m, stop_class = walk["chain"], walk["stop_m"], walk["stop_classification"]
    skipped = walk.get("skipped_tie_equivalent", [])

    if stop_class == "straddle":
        return {"m_star": None, "tier": "INDETERMINATE", "stop_m": stop_m,
                "lower_bound_m_star": chain[-1] if chain else None, "chain": chain,
                "skipped_tie_equivalent": skipped}

    if not chain:
        if skipped and stop_m is None:
            # EVERY eligible point the walk saw was joint-NO-RECALL (sec 1.31.1's Rev 5.1 M3
            # rule) -- a joint task-learning/scale failure across the grid, NOT the M*=inf
            # strongest-win pathway (no rejection was ever tested and confirmed).
            return {"m_star": None, "tier": "TIE", "stop_m": None, "chain": chain,
                    "skipped_tie_equivalent": skipped, "confirmed_no_crossover": False,
                    "note": "joint-NO-RECALL at every walked grid point -- TIE-equivalent "
                            "(a task-learning/scale finding, never a memory-capacity verdict)"}
        # Immediate clean non-rejection at the very first tested M (32) -- the M*=inf pathway.
        return {"m_star": float("inf"), "tier": "WIN", "stop_m": stop_m, "chain": chain,
                "skipped_tie_equivalent": skipped,
                "confirmed_no_crossover": bool(tested_full_grid and stop_class == "clean_non_reject")}

    m_star = chain[-1]
    result = {"m_star": m_star, "tier": _tier_for_finite_m_star(m_star), "stop_m": stop_m,
              "chain": chain, "skipped_tie_equivalent": skipped, "confirmed_no_crossover": False}
    if stop_class is None:
        result["note"] = "every tested M was rejected -- crossover held down to the smallest tested M"
    return result


def run_mstar_procedure(cis_by_m: dict, margin: float = MARGIN_DEFAULT,
                        tie_equivalent_ms: frozenset | set = frozenset()) -> dict:
    """Top-level entry point: walk + finalize, in one call. A joint-NO-RECALL
    (tie-equivalent) grid point never counts toward -- nor against -- the
    full-grid requirement of the CONFIRMED-no-crossover sub-claim: that
    claim requires every eligible point individually confirmed CLEAN, and a
    joint-failure point is by definition not a clean non-rejection, so any
    skip disqualifies the strongest sub-claim (handled below by requiring
    an empty tie_equivalent_ms for full-grid status)."""
    walk = descending_walk(cis_by_m, margin, tie_equivalent_ms)
    tested_full_grid = (all(m in cis_by_m for m in ELIGIBLE_M_DESCENDING)
                        and not walk.get("skipped_tie_equivalent"))
    return finalize_verdict(walk, tested_full_grid)


def resolve_mstar_with_extension(cis_by_m_n3: dict, margin: float = MARGIN_DEFAULT,
                                  extended_ci_for_boundary: dict | None = None,
                                  tie_equivalent_ms: frozenset | set = frozenset()) -> dict:
    """Sec 1.8's straddle-triggered extension hook. Runs the n=3 walk; if
    it stops on a STRADDLE and an extended (n=9) CI for that EXACT
    boundary M is supplied, re-runs the walk with ONLY that cell's CI
    replaced (sec 1.8's own guarantee: "at most ONE cell per task can ever
    be the deciding boundary" -- never re-tests any OTHER cell, since the
    fixed-sequence structure stops testing beyond the first non-rejection
    by construction). If still a straddle (or no extension supplied),
    returns the n=3 verdict unresolved (INDETERMINATE)."""
    verdict = run_mstar_procedure(cis_by_m_n3, margin, tie_equivalent_ms)
    if verdict["tier"] != "INDETERMINATE":
        verdict["extension_used"] = False
        return verdict
    boundary_m = verdict["stop_m"]
    if extended_ci_for_boundary is None:
        verdict["extension_used"] = False
        return verdict
    cis_extended = dict(cis_by_m_n3)
    cis_extended[boundary_m] = extended_ci_for_boundary
    resolved = run_mstar_procedure(cis_extended, margin, tie_equivalent_ms)
    resolved["extension_used"] = True
    resolved["extended_at_m"] = boundary_m
    return resolved


# ---------------------------------------------------------------------------
# Smoke gate -- the four hand-walked edge cases (round 4's own verdict), reproduced as REAL,
# runnable synthetic-CI test cases, plus negative/structural tests.
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def _ci(low: float, high: float) -> dict:
    return {"ci_low": low, "ci_high": high, "mean": (low + high) / 2.0}


def smoke_1_grid_exhaustion_lose():
    """Every eligible M rejected, even M=2 (crossover holds all the way down)."""
    cis = {m: _ci(-0.05, 0.10) for m in ELIGIBLE_M_DESCENDING}   # ci_high=0.10 < 0.20 at every M
    v = run_mstar_procedure(cis)
    ok = v["m_star"] == 2 and v["tier"] == "LOSE" and v["chain"] == [32, 16, 8, 4, 2]
    _report("smoke 1: grid-exhaustion (every M rejected) -> M*=2 -> LOSE", ok, f"verdict={v}")


def smoke_2_stop_at_4_win_m_star_8():
    """Rejected at 32/16/8; clean stop at 4 -> M*=8 (the last rejected M) -> WIN."""
    cis = {32: _ci(-0.05, 0.10), 16: _ci(-0.03, 0.12), 8: _ci(0.02, 0.15),
           4: _ci(0.25, 0.35), 2: _ci(0.30, 0.40)}
    v = run_mstar_procedure(cis)
    ok = v["m_star"] == 8 and v["tier"] == "WIN" and v["stop_m"] == 4 and v["chain"] == [32, 16, 8]
    _report("smoke 2: stop-at-4 (clean non-reject) -> M*=8 -> WIN", ok, f"verdict={v}")


def smoke_3_immediate_nonreject_m_star_infinity_confirmed_requires_full_grid():
    """Immediate clean non-rejection at M=32 (chain empty). Two sub-cases:
    (a) FULL grid tested and clean -> CONFIRMED no-crossover; (b) only
    M=32 tested (cost-saving shortcut) -> WIN but NOT confirmed (the
    "full-grid requirement verified unambiguous" round-4 finding, checked
    directly here, not merely asserted)."""
    cis_full = {m: _ci(0.25, 0.35) for m in ELIGIBLE_M_DESCENDING}
    v_full = run_mstar_procedure(cis_full)
    full_ok = (v_full["m_star"] == float("inf") and v_full["tier"] == "WIN"
               and v_full["confirmed_no_crossover"] is True)

    cis_partial = {32: _ci(0.25, 0.35)}   # ONLY M=32 tested
    v_partial = run_mstar_procedure(cis_partial)
    partial_ok = (v_partial["m_star"] == float("inf") and v_partial["tier"] == "WIN"
                  and v_partial["confirmed_no_crossover"] is False)

    ok = full_ok and partial_ok
    _report("smoke 3: immediate non-reject -> M*=inf pathway; CONFIRMED-no-crossover requires the "
            "FULL grid, the shortcut alone does NOT earn it", ok,
            f"full={v_full} partial={v_partial}")


def smoke_4_boundary_straddle_triggers_extension():
    """Straddle at the boundary M -> INDETERMINATE, unresolved without an
    extension; resolved (to WIN, in this synthetic case) once an
    extended-seed CI for that EXACT M is supplied."""
    cis_n3 = {32: _ci(-0.05, 0.10), 16: _ci(-0.03, 0.12), 8: _ci(0.10, 0.28)}   # 8 straddles 0.20
    v_unresolved = run_mstar_procedure(cis_n3)
    unresolved_ok = (v_unresolved["tier"] == "INDETERMINATE" and v_unresolved["stop_m"] == 8
                      and v_unresolved["lower_bound_m_star"] == 16)

    v_no_ext = resolve_mstar_with_extension(cis_n3, extended_ci_for_boundary=None)
    no_ext_ok = v_no_ext["tier"] == "INDETERMINATE" and v_no_ext["extension_used"] is False

    extended_clean = _ci(0.22, 0.30)   # n=9 CI resolves clean_non_reject at M=8 (ci_low > 0.20)
    v_resolved = resolve_mstar_with_extension(cis_n3, extended_ci_for_boundary=extended_clean)
    resolved_ok = (v_resolved["tier"] == "WIN" and v_resolved["m_star"] == 16
                   and v_resolved["extension_used"] is True and v_resolved["extended_at_m"] == 8)

    still_straddling = _ci(0.15, 0.24)   # n=9 CI STILL straddles -> stays INDETERMINATE
    v_still = resolve_mstar_with_extension(cis_n3, extended_ci_for_boundary=still_straddling)
    still_ok = v_still["tier"] == "INDETERMINATE" and v_still["extension_used"] is True

    ok = unresolved_ok and no_ext_ok and resolved_ok and still_ok
    _report("smoke 4: boundary straddle -> INDETERMINATE -> resolved-by-extension (clean) OR "
            "stays INDETERMINATE (still straddling post-extension)", ok,
            f"unresolved={v_unresolved} resolved={v_resolved} still={v_still}")


def smoke_5_descending_order_enforced_negative_test():
    """cis_by_m keyed OUT of descending order must still be walked in the
    PINNED order (32 first) -- a dict has no inherent order guarantee this
    module should rely on; confirms `descending_walk` iterates
    ELIGIBLE_M_DESCENDING itself, never `cis_by_m`'s own key order."""
    cis = {2: _ci(0.30, 0.40), 4: _ci(0.25, 0.35), 8: _ci(0.02, 0.15),
           16: _ci(-0.03, 0.12), 32: _ci(-0.05, 0.10)}   # insertion order ASCENDING
    v = run_mstar_procedure(cis)
    ok = v["chain"] == [32, 16, 8] and v["stop_m"] == 4
    _report("smoke 5: descending walk order is PINNED (32 first), independent of dict insertion "
            "order", ok, f"chain={v['chain']} stop_m={v['stop_m']}")


def smoke_6_classify_cell_boundary_exactness_negative_test():
    """Exact-threshold structural checks (CLAUDE.md's own rule): ci_high
    EXACTLY at margin is NOT a reject (needs ci_high < margin, strict);
    ci_low EXACTLY at margin is NOT a clean_non_reject (needs ci_low >
    margin, strict) -- both land as 'straddle' at the exact boundary."""
    exact_high = classify_cell(_ci(0.05, 0.20))
    exact_low = classify_cell(_ci(0.20, 0.35))
    ok = exact_high == "straddle" and exact_low == "straddle"
    _report("smoke 6: classify_cell exact-margin boundary lands as 'straddle' on both sides "
            "(strict inequalities, not <=/>=)", ok, f"exact_high={exact_high} exact_low={exact_low}")


def smoke_8_joint_no_recall_skip_and_all_skip_tie():
    """The sec 1.31.1 joint-NO-RECALL rule (Rev 5.1 M3), two cases:
    (a) MID-WALK skip -- M=16 is tie-equivalent; the walk continues past
    it (never a stop, never in the chain, never M*): 32/8 reject, clean
    stop at 4 -> M*=8 -> WIN, skipped=[16]; the skip also DISQUALIFIES
    the CONFIRMED-no-crossover sub-claim by construction.
    (b) ALL-SKIP -- every eligible point joint-fails -> TIE (a joint
    task-learning/scale finding), NEVER the M*=inf strongest-win pathway
    an empty chain would otherwise trigger (negative teeth: the same
    empty-chain input WITHOUT the skips returns WIN, proving the guard,
    not the input, does the work)."""
    cis = {32: _ci(-0.05, 0.10), 16: _ci(0.25, 0.35), 8: _ci(0.02, 0.15),
           4: _ci(0.25, 0.35), 2: _ci(0.30, 0.40)}
    v = run_mstar_procedure(cis, tie_equivalent_ms={16})
    a_ok = (v["m_star"] == 8 and v["tier"] == "WIN" and v["chain"] == [32, 8]
            and v["stop_m"] == 4 and v["skipped_tie_equivalent"] == [16]
            and v["confirmed_no_crossover"] is False)

    cis_all = {m: _ci(0.25, 0.35) for m in ELIGIBLE_M_DESCENDING}
    v_all = run_mstar_procedure(cis_all, tie_equivalent_ms=set(ELIGIBLE_M_DESCENDING))
    b_ok = (v_all["tier"] == "TIE" and v_all["m_star"] is None
            and v_all["skipped_tie_equivalent"] == [32, 16, 8, 4, 2])
    v_noskip = run_mstar_procedure(cis_all)                     # same CIs, no skips -> WIN
    teeth_ok = v_noskip["tier"] == "WIN" and v_noskip["m_star"] == float("inf")

    ok = a_ok and b_ok and teeth_ok
    _report("smoke 8 (sec 1.31.1 joint-NO-RECALL): a tie-equivalent M is walked PAST (never "
            "chain/stop/M*, disqualifies CONFIRMED); all-points-joint-fail -> TIE, never the "
            "M*=inf WIN an empty chain alone would give (teeth run to completion)", ok,
            f"mid={v} all={v_all} noskip_tier={v_noskip['tier']}")


def smoke_9_zero_seed_variance_disclosed():
    """The sigma=0-at-ceiling analog for the M* machinery (sec 1.40's
    instrument note): identical paired deltas (all seeds at exact ceiling
    vs identical baseline reads) must be FLAGGED zero_seed_variance=True
    (zero-width CI disclosed, not silently point-classified); a normal
    varying-seed input must flag False."""
    degenerate = compute_ci_from_raw_seed_values([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    normal = compute_ci_from_raw_seed_values([1.0, 0.999, 0.998], [0.5, 0.5, 0.5])
    ok = (degenerate["zero_seed_variance"] is True
          and degenerate["ci_low"] == degenerate["ci_high"] == 0.5
          and normal["zero_seed_variance"] is False)
    _report("smoke 9 (sigma=0 analog): identical paired deltas -> zero_seed_variance=True with "
            "a zero-width CI (disclosed); varying deltas -> False", ok,
            f"degenerate=({degenerate['ci_low']:.4f},{degenerate['ci_high']:.4f},"
            f"flag={degenerate['zero_seed_variance']}) normal_flag={normal['zero_seed_variance']}")


def smoke_7_raw_seed_values_never_a_summary():
    """compute_ci_from_raw_seed_values must reject an obviously-summarized
    (length-1) input the same way delta_ci_n itself does (n>=2 required) --
    proves this wrapper doesn't silently accept a pre-averaged scalar."""
    raised = False
    try:
        compute_ci_from_raw_seed_values([0.5], [0.3])
        raise RuntimeError("NEGATIVE FAILED TO FAIL: n=1 (a disguised summary) did not raise")
    except AssertionError:
        raised = True
    real = compute_ci_from_raw_seed_values([0.5, 0.6, 0.55], [0.3, 0.25, 0.28])
    real_ok = "ci_low" in real and "ci_high" in real
    ok = raised and real_ok
    _report("smoke 7: compute_ci_from_raw_seed_values refuses a disguised n=1 summary; accepts "
            "real raw n=3 seed arrays", ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("h2h_mstar_analysis_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.4.2 M* smoke suite")
    print("=" * 70)
    smoke_1_grid_exhaustion_lose()
    smoke_2_stop_at_4_win_m_star_8()
    smoke_3_immediate_nonreject_m_star_infinity_confirmed_requires_full_grid()
    smoke_4_boundary_straddle_triggers_extension()
    smoke_5_descending_order_enforced_negative_test()
    smoke_6_classify_cell_boundary_exactness_negative_test()
    smoke_7_raw_seed_values_never_a_summary()
    smoke_8_joint_no_recall_skip_and_all_skip_tie()
    smoke_9_zero_seed_variance_disclosed()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
