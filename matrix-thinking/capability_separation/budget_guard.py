"""CAPABILITY_SEPARATION_DESIGN.md S1.6/S1.20 -- the 30 GPU-h hard-abort
wrapper for the Stage-1 dedicated ledger, WITH the CA4-M1 live
escalation-budget guard (S1.20's Rev 4 fix, S1.21-cleared): the hard-abort
re-checks actual-spend-to-date + projected cost BEFORE launching ANY
escalation-triggered cell (general escalation-to-5 OR the marquee n=7
trigger), not only once upfront against the base 58-cell sweep.

Pinned yield order if a projection would breach the cap (S1.6/S1.20, S1.21
tie-break note 1): the MARQUEE n=7 escalation outranks ALL general
escalations (it guards the design's central S4-vs-A5 dissociation claim,
S1.5 -- the one result this design cannot ship without); general
escalations are granted in a PINNED ORDER by ascending group |G| (S3 -> S4/A5
tied, S4-before-A5 canonical-name-order tie-break -> S5 -> A6, "cheapest
information first") until the cap is hit. Anything denied is reported with
an explicit `budget-denied` status, distinct from "resolved at base seed
count" or "skipped by design" (S1.6's harvest-provenance requirement).

This is a budget-ARBITRATION mechanism only (S1.21 note 2): it does not
change S1.4.2's escalation eligibility rules or S1.5's decision criteria. A
budget-denied marquee escalation leaves TOST's own INCONCLUSIVE verdict
standing -- S1.5's CONFIRM cannot be reached without a declared equivalence,
so a denial safely degrades to the pre-existing INCONCLUSIVE catch-all
rather than silently fabricating a verdict.

All figures below are DERIVED (not copied) from S1.4.2's cell table and
S1.6's cost table, and cross-checked against S1.19/S1.20/S1.21's own
independently-re-derived numbers (base 19.65, marquee +2.40 [8 cells],
general-max +12.60 [42 cells: 10/group for S3/S5/A6, 6/group flanking-only
for S4/A5], worst case 34.65 > 30 by 4.65) in `_self_test()` below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# S1.6's cost table, exact figures.
# ---------------------------------------------------------------------------

COST_PER_CELL = 0.3        # GPU-h/cell, S1.6's planning rate
BASE_SWEEP_CELLS = 58       # S1.4.2's cell count (10+14+14+10+10)
CONTINGENCY = 1.35          # rate-of-occurrence margin, S1.6 (unchanged by cell count)
BETA_SMOKE = 0.90           # S1.4.3/S1.6's beta-fla positive-control smoke (fixed, non-load-bearing)
BASE_RAW_TOTAL = round(BASE_SWEEP_CELLS * COST_PER_CELL + CONTINGENCY + BETA_SMOKE, 4)  # 19.65
CAP = 30.0                  # S1.6's Stage-1 dedicated ledger

# S1.4.2's per-group, per-cell-type base seed counts.
BASE_SEEDS = {
    "S3": {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "S4": {"unconstrained": 5, "k_dmin": 5, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "A5": {"unconstrained": 5, "k_dmin": 5, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "S5": {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "A6": {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
}
GROUP_SIZE = {"S3": 6, "S4": 24, "A5": 60, "S5": 120, "A6": 360}
# S1.21 note 1: |G| ascending, S4-before-A5 canonical-name-order tie-break (24 vs 60 are NOT
# literally equal, but S1.21 pins this order as the safe default regardless).
GROUP_ORDER_ASCENDING = ["S3", "S4", "A5", "S5", "A6"]

# S4/A5's unconstrained + k_dmin cells are ALREADY unconditional n=5 (CA3-M1(a)) --
# not eligible for the GENERAL escalation-to-5 trigger (there is nothing to escalate
# TO at n=5). Only their 2 flanking cell types remain general-escalation-eligible.
GENERAL_ESCALATION_ELIGIBLE = {
    "S3": ["unconstrained", "k_dmin", "k_dmin_minus_1", "k_dmin_plus_1"],
    "S4": ["k_dmin_minus_1", "k_dmin_plus_1"],
    "A5": ["k_dmin_minus_1", "k_dmin_plus_1"],
    "S5": ["unconstrained", "k_dmin", "k_dmin_minus_1", "k_dmin_plus_1"],
    "A6": ["unconstrained", "k_dmin", "k_dmin_minus_1", "k_dmin_plus_1"],
}
GENERAL_ESCALATION_TARGET_N = 5

MARQUEE_GROUPS = ("S4", "A5")
MARQUEE_CELL_TYPES = ("unconstrained", "k_dmin")
MARQUEE_N5, MARQUEE_N7 = 5, 7
MARQUEE_ESCALATION_COST = round(
    len(MARQUEE_GROUPS) * len(MARQUEE_CELL_TYPES) * (MARQUEE_N7 - MARQUEE_N5) * COST_PER_CELL, 4
)  # 2.4

GENERAL_ESCALATION_MAX_COST = round(sum(
    (GENERAL_ESCALATION_TARGET_N - BASE_SEEDS[g][ct]) * COST_PER_CELL
    for g in GROUP_ORDER_ASCENDING for ct in GENERAL_ESCALATION_ELIGIBLE[g]
), 4)  # 12.6

WORST_CASE_TOTAL = round(BASE_RAW_TOTAL + MARQUEE_ESCALATION_COST + GENERAL_ESCALATION_MAX_COST, 4)  # 34.65


class BaseSweepAbort(RuntimeError):
    """Raised by check_base_sweep_projection when the calibration cell's
    REAL per-cell rate projects the base 58-cell sweep over the cap (S1.7
    gate 2's timing pilot, mechanical enforced abort)."""


def check_base_sweep_projection(real_rate_per_cell: float) -> dict:
    """S1.7 gate 2: project the base 58-cell sweep from the calibration
    cell's REAL measured wall-clock rate (supersedes S1.6's 0.3 GPU-h/cell
    planning estimate). Hard-aborts (raises BaseSweepAbort) if the
    projection, plus the fixed contingency/beta-smoke rows, exceeds the 30
    GPU-h cap -- BEFORE the remaining 53-cell sweep launches."""
    projected = round(BASE_SWEEP_CELLS * real_rate_per_cell + CONTINGENCY + BETA_SMOKE, 4)
    ok = projected <= CAP
    result = dict(real_rate_per_cell=real_rate_per_cell, projected_total=projected, cap=CAP, ok=ok)
    if not ok:
        raise BaseSweepAbort(
            f"base-sweep projection {projected:.2f} GPU-h exceeds the {CAP:.1f} GPU-h cap "
            f"at real rate {real_rate_per_cell:.4f} GPU-h/cell -- HARD ABORT before spending "
            f"the remaining sweep. Re-scope the cell/seed grid before relaunch. {result}"
        )
    return result


@dataclass
class BudgetGuard:
    """CA4-M1's live escalation-budget guard. `spend_to_date` and
    `committed_escalations` are tracked separately so the projection
    formula (`spend_to_date + committed_escalations + candidate_cost`) is
    auditable at every decision -- matches S1.6's own worst-case arithmetic
    convention (actual spend + already-committed escalations + the
    newly-triggered escalation's cost)."""
    cap: float = CAP
    spend_to_date: float = 0.0
    committed_escalations: float = 0.0
    decisions: list = field(default_factory=list)

    def projected_total(self, candidate_cost: float) -> float:
        return round(self.spend_to_date + self.committed_escalations + candidate_cost, 4)

    def request_marquee_n7(self) -> dict:
        """The marquee n=7 escalation (S4+A5's unconstrained+k_dmin cells,
        5->7 seeds, 8 extra cells, 2.4 GPU-h). Outranks ALL general
        escalations -- always requested FIRST in run_escalation_pass."""
        cost = MARQUEE_ESCALATION_COST
        proj = self.projected_total(cost)
        granted = proj <= self.cap
        decision = dict(request="marquee_n7", cost=cost, projected_total=proj, cap=self.cap,
                        granted=granted, status="granted" if granted else "budget-denied")
        if granted:
            self.committed_escalations = round(self.committed_escalations + cost, 4)
        self.decisions.append(decision)
        return decision

    def request_general_escalation(self, group: str, cell_type: str) -> dict:
        """One (group, cell_type) general escalation-to-5 unit. Caller MUST
        offer requests in the pinned ascending-|G| order (this method grants
        or denies against the live projection only; it does not reorder)."""
        n_base = BASE_SEEDS[group][cell_type]
        extra_cells = GENERAL_ESCALATION_TARGET_N - n_base
        cost = round(extra_cells * COST_PER_CELL, 4)
        proj = self.projected_total(cost)
        granted = proj <= self.cap
        decision = dict(request=f"general_escalation:{group}:{cell_type}", cost=cost,
                        projected_total=proj, cap=self.cap, granted=granted,
                        status="granted" if granted else "budget-denied")
        if granted:
            self.committed_escalations = round(self.committed_escalations + cost, 4)
        self.decisions.append(decision)
        return decision

    @staticmethod
    def pinned_general_order() -> list:
        """S1.6/S1.21's pinned order: ascending |G|, S4-before-A5 tie-break;
        within a group, cell-type order is unconstrained by the design (no
        further tie-break is specified at that granularity) -- this
        implementation uses the grid's own declaration order, deterministic
        and stable across runs."""
        order = []
        for g in GROUP_ORDER_ASCENDING:
            for ct in GENERAL_ESCALATION_ELIGIBLE[g]:
                order.append((g, ct))
        return order

    def run_escalation_pass(self, marquee_triggered: bool, general_triggered_units: set) -> list:
        """S1.6/S1.20's pinned procedure for ONE escalation-check point:
        marquee FIRST (if triggered), THEN general escalations for
        TRIGGERED units only (S1.4.2's own eligibility rule -- this guard
        never requests an escalation nobody's ambiguity trigger actually
        fired), in the pinned ascending-|G| order. Returns the list of
        decisions made this pass (each already appended to self.decisions)."""
        results = []
        if marquee_triggered:
            results.append(self.request_marquee_n7())
        for g, ct in self.pinned_general_order():
            if (g, ct) in general_triggered_units:
                results.append(self.request_general_escalation(g, ct))
        return results

    def summary(self) -> dict:
        granted = [d for d in self.decisions if d["granted"]]
        denied = [d for d in self.decisions if not d["granted"]]
        return dict(n_granted=len(granted), n_denied=len(denied),
                    total_committed=self.committed_escalations,
                    final_projected_total=self.projected_total(0.0),
                    denied_requests=[d["request"] for d in denied])


# ---------------------------------------------------------------------------
# Self-test: the worst-case arithmetic, independently re-derived (S1.19-S1.21's
# own figures cross-checked, not copied).
# ---------------------------------------------------------------------------

def _self_test() -> None:
    print("=" * 88)
    print("budget_guard.py self-test -- worst-case arithmetic cross-check vs S1.19-S1.21")
    print("=" * 88)
    print(f"  BASE_RAW_TOTAL         = {BASE_RAW_TOTAL}  (expect 19.65)")
    print(f"  MARQUEE_ESCALATION_COST = {MARQUEE_ESCALATION_COST}  (expect 2.40, 8 cells)")
    print(f"  GENERAL_ESCALATION_MAX_COST = {GENERAL_ESCALATION_MAX_COST}  (expect 12.60, 42 cells)")
    print(f"  WORST_CASE_TOTAL       = {WORST_CASE_TOTAL}  (expect 34.65)")
    print(f"  WORST_CASE_TOTAL - CAP = {round(WORST_CASE_TOTAL - CAP, 4)}  (expect 4.65)")
    assert abs(BASE_RAW_TOTAL - 19.65) < 1e-6
    assert abs(MARQUEE_ESCALATION_COST - 2.40) < 1e-6
    assert abs(GENERAL_ESCALATION_MAX_COST - 12.60) < 1e-6
    assert abs(WORST_CASE_TOTAL - 34.65) < 1e-6
    assert abs((WORST_CASE_TOTAL - CAP) - 4.65) < 1e-6

    # per-group general-escalation cell counts (S1.21: S3/S5/A6=10 cells each; S4/A5=6 each).
    for g in ("S3", "S5", "A6"):
        cells = sum(GENERAL_ESCALATION_TARGET_N - BASE_SEEDS[g][ct] for ct in GENERAL_ESCALATION_ELIGIBLE[g])
        assert cells == 10, f"{g}: expected 10 general-escalation-eligible extra cells, got {cells}"
    for g in ("S4", "A5"):
        cells = sum(GENERAL_ESCALATION_TARGET_N - BASE_SEEDS[g][ct] for ct in GENERAL_ESCALATION_ELIGIBLE[g])
        assert cells == 6, f"{g}: expected 6 general-escalation-eligible extra cells (flanking only), got {cells}"
    print("  per-group general-escalation cell counts: S3/S5/A6=10 each, S4/A5=6 each (flanking only)  PASS")
    print("\nbudget_guard.py self-test PASSED.\n")


# ---------------------------------------------------------------------------
# NEGATIVE TEST (S1.12's build item): a simulated near-cap projection must
# deny the correct escalation(s) in the pinned order and log each denial
# with a `budget-denied` status, not silently skip or over-spend.
# ---------------------------------------------------------------------------

def _test_near_cap_denies_in_pinned_order():
    print("=" * 88)
    print("NEGATIVE TEST -- simulated near-cap projection denies escalations in pinned order")
    print("=" * 88)
    # Simulate: the base 58-cell sweep has ALREADY completed at spend =
    # BASE_RAW_TOTAL (19.65, only 10.35 GPU-h of headroom remains under the
    # 30 GPU-h cap), and EVERY escalation trigger fires simultaneously (the
    # literal S1.19 worst case: marquee TOST inconclusive on both S4/A5 AND
    # every general-escalation-eligible cell type is ambiguous).
    guard = BudgetGuard(cap=CAP, spend_to_date=BASE_RAW_TOTAL)
    all_general_units = set(guard.pinned_general_order())
    print(f"  spend_to_date={guard.spend_to_date}  cap={guard.cap}  headroom={guard.cap - guard.spend_to_date:.2f}")
    print(f"  triggering: marquee=True, general_units=ALL {len(all_general_units)} eligible units")

    decisions = guard.run_escalation_pass(marquee_triggered=True, general_triggered_units=all_general_units)

    print(f"\n  {'request':<32}{'cost':>7}{'proj_total':>12}{'status':>16}")
    print("  " + "-" * 67)
    for d in decisions:
        print(f"  {d['request']:<32}{d['cost']:>7.2f}{d['projected_total']:>12.2f}{d['status']:>16}")

    summ = guard.summary()
    print(f"\n  summary: n_granted={summ['n_granted']}  n_denied={summ['n_denied']}  "
          f"total_committed={summ['total_committed']:.2f}  final_projected_total={summ['final_projected_total']:.2f}")
    print(f"  denied requests: {summ['denied_requests']}")

    # Assertions with teeth:
    assert decisions[0]["request"] == "marquee_n7" and decisions[0]["granted"], \
        "marquee n=7 escalation must be requested FIRST and (at this headroom) GRANTED"
    assert summ["final_projected_total"] <= CAP, \
        f"guard over-spent: final projected total {summ['final_projected_total']} > cap {CAP}"
    assert summ["n_denied"] > 0, \
        "this stress scenario (headroom 10.35 < worst-case demand 15.0) must produce at least one denial"
    # A6 (last in the pinned order, largest |G|) must be entirely denied at this headroom.
    a6_decisions = [d for d in decisions if ":A6:" in d["request"]]
    assert all(not d["granted"] for d in a6_decisions), \
        "A6 (last in pinned order) should be fully denied before headroom is exhausted at this scenario"
    # S3 (first in the pinned order, smallest |G|) must be entirely granted.
    s3_decisions = [d for d in decisions if ":S3:" in d["request"]]
    assert all(d["granted"] for d in s3_decisions), \
        "S3 (first in pinned order, cheapest information) should be granted before the cap bites"
    # every denial carries the explicit 'budget-denied' status (not silently dropped).
    denied = [d for d in decisions if not d["granted"]]
    assert all(d["status"] == "budget-denied" for d in denied), \
        "every denied request must be logged with status='budget-denied'"

    print("\nRESULT: marquee granted first; general escalations granted in ascending-|G| pinned")
    print("order until the cap bites (S3 fully granted, A6 fully denied); every denial logged")
    print("with status='budget-denied'; final committed spend never exceeds the 30 GPU-h cap.")
    print("=" * 88)
    return decisions, summ


if __name__ == "__main__":
    _self_test()
    _test_near_cap_denies_in_pinned_order()
