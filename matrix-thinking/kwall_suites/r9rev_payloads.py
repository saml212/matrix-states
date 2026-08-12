#!/usr/bin/env python3
"""
Payload constructions for the Rev-9 revision agent's re-run of the
differential suite, faithfully reconstructed from the design doc's own
in-text descriptions (L-series "Reportable outcomes", A-series
"Adversarial JSONs from R7's own audit", B-series "adversarial
extensions" per §A8-ADJUDICATION's own K-disposition table and the R9
attack report's §1.1/§2.1 tables). The original vcheck_r8_rev.py /
drive_vcheck_r8_rev.py scripts are gone (ephemeral, prior sessions'
scratchpads, per KW10.1's own finding) -- this is an independent
re-derivation from the doc text, same as R9's own r9_indep.py was.

Every payload used in the SIX-FLIP delta check (B1,B1',B2,B2',B3-NEG,
B4,B3-AMENDED) and the L6 rebuild is reconstructed at FULL fidelity,
matching every field the design doc's own prose specifies. L1,L2,L3,
L4,L5,L7,L7',A1,A6,A6',B3-OLD-STYLE are included as regression/
sanity payloads. A2-A5,A7 and the exact remaining slots of the
original 24 are NOT reconstructed here (their prose lives outside the
8 edited items and none of Rev-9's edits touch the logic paths they
exercise) -- disclosed explicitly in the run summary, not silently
omitted.
"""

def mk_pairs_completed(pairs, arm="primary", elapsed=1.00):
    return [{"K": K, "seed": s, "arm": arm, "attempt_n": 1,
              "status": "COMPLETED", "elapsed_h": elapsed,
              "ceiling_charged": False} for (K, s) in pairs]


ALL12 = [(K, s) for K in (26, 28, 30) for s in (0, 1, 2, 3)]

PAYLOADS = {}

# ---- L1: COMPLETE, 12/12 canonical, strict, base case (legitimate) ----
ledger_L1 = mk_pairs_completed(ALL12, elapsed=1.00)
PAYLOADS["L1"] = dict(
    run_status="COMPLETE", realized=12.00, ledger=ledger_L1,
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=0, band={"interval_resolved_Ks": [], "incomplete_at_K": None},
    conditional=None, charged_vs_measured=None, trigger=None,
    expect_NEW="PASS", expect_OLD="PASS",
)

# ---- L2: INCOMPLETE-AT-K, 11 canonical, K=30 seed CRASHED both attempts ----
pairs_L2_completed = [(K, s) for (K, s) in ALL12 if not (K == 30 and s == 3)]
ledger_L2 = mk_pairs_completed(pairs_L2_completed, elapsed=1.00)
ledger_L2 += [
    {"K": 30, "seed": 3, "arm": "primary", "attempt_n": 1, "status": "CRASHED-RECOVERED",
     "elapsed_h": 1.20, "ceiling_charged": True},
    {"K": 30, "seed": 3, "arm": "primary", "attempt_n": 2, "status": "CRASHED-RECOVERED",
     "elapsed_h": 1.20, "ceiling_charged": True},
]
PAYLOADS["L2"] = dict(
    run_status="COMPLETE", realized=11 * 1.00 + 2.40, ledger=ledger_L2,
    primary_canonical=11, primary_canonical_by_K={26: 4, 28: 4, 30: 3},
    conditional_canonical=0, band={"interval_resolved_Ks": [], "incomplete_at_K": [30]},
    conditional=None, charged_vs_measured=None, trigger=None,
    expect_NEW="PASS", expect_OLD="PASS",
)

# ---- L3: COMPLETE-DEGRADED sub-case (i), 11 canonical, retry GATE-REFUSED ----
pairs_L3_completed = [(K, s) for (K, s) in ALL12 if not (K == 30 and s == 3)]
ledger_L3 = mk_pairs_completed(pairs_L3_completed, elapsed=1.00)
ledger_L3 += [
    {"K": 30, "seed": 3, "arm": "primary", "attempt_n": 1, "status": "CRASHED-RECOVERED",
     "elapsed_h": 1.20, "ceiling_charged": True},
    {"K": 30, "seed": 3, "arm": "primary", "attempt_n": 2, "status": "GATE-REFUSED",
     "elapsed_h": 0.0, "ceiling_charged": False},
]
PAYLOADS["L3"] = dict(
    run_status="COMPLETE-DEGRADED", realized=11 * 1.00 + 1.20, ledger=ledger_L3,
    primary_canonical=11, primary_canonical_by_K={26: 4, 28: 4, 30: 3},
    conditional_canonical=0, band={}, conditional=None,
    charged_vs_measured=None, trigger=None,
    expect_NEW="PASS", expect_OLD="PASS",
)

# ---- L4: COMPLETE-DEGRADED sub-case (ii)/(iii), 12/12 primary, conditional throttle ----
ledger_L4 = mk_pairs_completed(ALL12, elapsed=1.00)
ledger_L4 += [
    {"K": 26, "seed": 0, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False},
    {"K": 26, "seed": 1, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False},
    {"K": 26, "seed": 2, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False},
    {"K": 26, "seed": 3, "arm": "conditional", "attempt_n": 1, "status": "GATE-REFUSED",
     "elapsed_h": 0.0, "ceiling_charged": False},
]
PAYLOADS["L4"] = dict(
    run_status="COMPLETE-DEGRADED", realized=12.00, ledger=ledger_L4,  # primary-only (conditional rows tracked separately, not in the 15.50 cap)
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=3, band={}, conditional={"launched": True, "qualifier_band": None},
    charged_vs_measured=None, trigger=None,
    expect_NEW="PASS", expect_OLD="PASS",
)

# ---- L5: EXHAUSTED-BUDGET, realized=14.55, 9 canonical, 1 GATE-REFUSED ----
pairs_L5_completed = [(K, s) for (K, s) in ALL12][:9]
ledger_L5 = mk_pairs_completed(pairs_L5_completed, elapsed=1.35)  # 9*1.35=12.15
remaining = [p for p in ALL12 if p not in pairs_L5_completed]
ledger_L5 += [
    {"K": remaining[0][0], "seed": remaining[0][1], "arm": "primary", "attempt_n": 1,
     "status": "CRASHED-RECOVERED", "elapsed_h": 1.20, "ceiling_charged": True},
    {"K": remaining[1][0], "seed": remaining[1][1], "arm": "primary", "attempt_n": 1,
     "status": "CRASHED-RECOVERED", "elapsed_h": 1.20, "ceiling_charged": True},
    {"K": remaining[2][0], "seed": remaining[2][1], "arm": "primary", "attempt_n": 1,
     "status": "GATE-REFUSED", "elapsed_h": 0.0, "ceiling_charged": False},
]
PAYLOADS["L5"] = dict(
    run_status="EXHAUSTED-BUDGET", realized=12.15 + 2.40 + 0.0, ledger=ledger_L5,
    primary_canonical=9, primary_canonical_by_K=None,
    conditional_canonical=0, band={}, conditional=None,
    charged_vs_measured={"ceiling_charged_gpu_h": 2.40, "ceiling_charged_fraction": 2.40 / 14.55},
    trigger=None, expect_NEW="PASS", expect_OLD="PASS",
)

# ---- L6 (§R9 M1 REBUILT): EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE, self-consistent ----
pairs_L6_ceiling = ALL12[:10]
pairs_L6_completed = [ALL12[10]]
pairs_L6_refused = [ALL12[11]]
ledger_L6 = [
    {"K": K, "seed": s, "arm": "primary", "attempt_n": 1, "status": "CRASHED-RECOVERED",
     "elapsed_h": 1.20, "ceiling_charged": True} for (K, s) in pairs_L6_ceiling
]
ledger_L6 += mk_pairs_completed(pairs_L6_completed, elapsed=2.00)
ledger_L6 += [
    {"K": pairs_L6_refused[0][0], "seed": pairs_L6_refused[0][1], "arm": "primary",
     "attempt_n": 1, "status": "GATE-REFUSED", "elapsed_h": 0.0, "ceiling_charged": False}
]
PAYLOADS["L6"] = dict(
    run_status="EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", realized=12.00 + 2.00 + 0.0,
    ledger=ledger_L6, primary_canonical=1, primary_canonical_by_K=None,
    conditional_canonical=0, band={}, conditional=None,
    charged_vs_measured={"ceiling_charged_gpu_h": 12.00, "ceiling_charged_fraction": 12.00 / 14.00},
    trigger=None, expect_NEW="PASS", expect_OLD="PASS",  # J4 mirror clause predates Rev-8
)

# ---- L7: STOPPED-BY-OPERATOR -- excluded by U1, both OLD and NEW ----
PAYLOADS["L7"] = dict(
    run_status="STOPPED-BY-OPERATOR", realized=5.00, ledger=[],
    primary_canonical=0, primary_canonical_by_K=None, conditional_canonical=0,
    band={}, conditional=None, charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="FAIL",
)

# ---- L7': tie-break-min, COMPLETE strict, 12/12 canonical + full ledger ----
ledger_L7p = mk_pairs_completed(ALL12, elapsed=1.00)
PAYLOADS["L7'"] = dict(
    run_status="COMPLETE", realized=12.00, ledger=ledger_L7p,
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=0, band={"interval_resolved_Ks": [], "incomplete_at_K": None},
    conditional=None, charged_vs_measured=None,
    trigger={"K_trig": 26, "resolution": "tie-break-min",
             "resolution_detail": "candidates were [26, 28]"},
    expect_NEW="PASS", expect_OLD="PASS",
)

# ---- A1: ENHANCED no-op, COMPLETE otherwise, attempts=[] ----
PAYLOADS["A1"] = dict(
    run_status="COMPLETE", realized=0.0, ledger=[],
    primary_canonical=0, primary_canonical_by_K={26: 0, 28: 0, 30: 0},
    conditional_canonical=0, band={"interval_resolved_Ks": [], "incomplete_at_K": [26, 28, 30]},
    conditional=None, charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="FAIL",  # J1(a) fails both OLD (R7) and NEW
)

# ---- A6 / A6': the K4-rebuilt forced-fail negative, EXACT quotient declared ----
def build_A6_ledger():
    ledger = []
    for i in range(9):
        ledger.append({"K": 26, "seed": i % 4, "arm": "primary", "attempt_n": 1,
                        "status": "CRASHED-RECOVERED", "elapsed_h": 1.20,
                        "ceiling_charged": True, "_pairtag": f"cr{i}"})
    ledger.append({"K": 28, "seed": 0, "arm": "primary", "attempt_n": 1,
                    "status": "CRASHED-RECOVERED", "elapsed_h": 1.20, "ceiling_charged": True})
    ledger.append({"K": 28, "seed": 0, "arm": "primary", "attempt_n": 2,
                    "status": "CRASHED-RECOVERED", "elapsed_h": 1.20, "ceiling_charged": True})
    ledger.append({"K": 28, "seed": 1, "arm": "primary", "attempt_n": 1,
                    "status": "COMPLETED", "elapsed_h": 1.00, "ceiling_charged": False})
    ledger.append({"K": 28, "seed": 2, "arm": "primary", "attempt_n": 1,
                    "status": "GATE-REFUSED", "elapsed_h": 0.0, "ceiling_charged": False})
    # relabel the 9 single-attempt CRASHED-RECOVERED rows onto 9 distinct pairs
    distinct_pairs = [(30, s) for s in range(4)] + [(26, s) for s in range(4)] + [(28, 3)]
    for i, row in enumerate(ledger[:9]):
        row["K"], row["seed"] = distinct_pairs[i]
        del row["_pairtag"]
    return ledger

ledger_A6 = build_A6_ledger()
_A6_frac_exact = 13.20 / 14.20
PAYLOADS["A6"] = dict(
    run_status="EXHAUSTED-BUDGET", realized=14.20, ledger=ledger_A6,
    primary_canonical=1, primary_canonical_by_K=None, conditional_canonical=0,
    band={}, conditional=None,
    charged_vs_measured={"ceiling_charged_gpu_h": 13.20, "ceiling_charged_fraction": _A6_frac_exact},
    trigger=None, expect_NEW="FAIL", expect_OLD="FAIL",
)
PAYLOADS["A6-literal-0.9296"] = dict(PAYLOADS["A6"])
PAYLOADS["A6-literal-0.9296"]["charged_vs_measured"] = {
    "ceiling_charged_gpu_h": 13.20, "ceiling_charged_fraction": 0.9296}
PAYLOADS["A6-literal-0.9296"]["ledger"] = ledger_A6

PAYLOADS["A6'"] = dict(
    run_status="EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", realized=14.20, ledger=ledger_A6,
    primary_canonical=1, primary_canonical_by_K=None, conditional_canonical=0,
    band={}, conditional=None,
    charged_vs_measured={"ceiling_charged_gpu_h": 13.20, "ceiling_charged_fraction": _A6_frac_exact},
    trigger=None, expect_NEW="PASS", expect_OLD="PASS",
)
PAYLOADS["A6'-literal-0.9296"] = dict(PAYLOADS["A6'"])
PAYLOADS["A6'-literal-0.9296"]["charged_vs_measured"] = {
    "ceiling_charged_gpu_h": 13.20, "ceiling_charged_fraction": 0.9296}
PAYLOADS["A6'-literal-0.9296"]["ledger"] = ledger_A6

# ---- B1 / B1': zero-cost no-op via 12 GATE-REFUSED primary rows ----
ledger_B1 = [{"K": K, "seed": s, "arm": "primary", "attempt_n": 1, "status": "GATE-REFUSED",
              "elapsed_h": 0.0, "ceiling_charged": False} for (K, s) in ALL12]
PAYLOADS["B1"] = dict(
    run_status="COMPLETE", realized=0.0, ledger=ledger_B1,
    primary_canonical=0, primary_canonical_by_K={26: 0, 28: 0, 30: 0},
    conditional_canonical=0, band={"interval_resolved_Ks": [], "incomplete_at_K": [26, 28, 30]},
    conditional=None, charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="PASS",
)
PAYLOADS["B1'"] = dict(PAYLOADS["B1"])
PAYLOADS["B1'"]["run_status"] = "COMPLETE-DEGRADED"
PAYLOADS["B1'"]["ledger"] = ledger_B1
PAYLOADS["B1'"]["band"] = {}

# ---- B2 / B2': A6's real ledger, fraction/ccgh mis-declared ----
PAYLOADS["B2"] = dict(
    run_status="EXHAUSTED-BUDGET", realized=14.20, ledger=ledger_A6,
    primary_canonical=1, primary_canonical_by_K=None, conditional_canonical=0,
    band={}, conditional=None,
    charged_vs_measured={"ceiling_charged_gpu_h": 13.20, "ceiling_charged_fraction": 0.20},
    trigger=None, expect_NEW="FAIL", expect_OLD="PASS",
)
PAYLOADS["B2'"] = dict(
    run_status="EXHAUSTED-BUDGET", realized=14.20, ledger=ledger_A6,
    primary_canonical=1, primary_canonical_by_K=None, conditional_canonical=0,
    band={}, conditional=None,
    charged_vs_measured={"ceiling_charged_gpu_h": 2.84, "ceiling_charged_fraction": 0.20},
    trigger=None, expect_NEW="FAIL", expect_OLD="PASS",
)

# ---- B3-OLD-STYLE: 3/4 throttled, band STILL claimed (should FAIL both) ----
ledger_B3 = mk_pairs_completed(ALL12, elapsed=1.00)
ledger_B3 += [
    {"K": 26, "seed": 0, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False},
    {"K": 26, "seed": 1, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False},
    {"K": 26, "seed": 2, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False},
    {"K": 26, "seed": 3, "arm": "conditional", "attempt_n": 1, "status": "GATE-REFUSED",
     "elapsed_h": 0.0, "ceiling_charged": False},
]
PAYLOADS["B3-OLD-STYLE"] = dict(
    run_status="COMPLETE-DEGRADED", realized=12.00, ledger=ledger_B3,  # primary-only
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=3, band={},
    conditional={"launched": True, "qualifier_band": "SLOW-CONVERGENCE-AT-160K"},
    charged_vs_measured=None, trigger={"K_trig": 26},
    expect_NEW="FAIL", expect_OLD="FAIL",  # clause (a)/(b) predate Rev-8
)

# ---- B3-AMENDED: 3/4 throttled, band=null (the legitimate case, = L4) ----
PAYLOADS["B3-AMENDED"] = dict(PAYLOADS["L4"])
PAYLOADS["B3-AMENDED"]["expect_NEW"] = "PASS"
PAYLOADS["B3-AMENDED"]["expect_OLD"] = "PASS"  # validity_check verdict UNCHANGED (m1's point)

# ---- B3-NEG: 4/4 conditional completed, band still declared null ----
ledger_B3NEG = mk_pairs_completed(ALL12, elapsed=1.00)
ledger_B3NEG += [
    {"K": 26, "seed": s, "arm": "conditional", "attempt_n": 1, "status": "COMPLETED",
     "elapsed_h": 2.00, "ceiling_charged": False} for s in range(4)
]
PAYLOADS["B3-NEG"] = dict(
    run_status="COMPLETE-DEGRADED", realized=12.00, ledger=ledger_B3NEG,  # primary-only
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=4, band={},
    conditional={"launched": True, "qualifier_band": None},
    charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="PASS",
)
# NB: B3-NEG needs SOME throttle evidence for CD's own base clause to reach
# the mirror check at all (a primary GATE-REFUSED-free, fully-COMPLETED
# primary run reported as COMPLETE-DEGRADED still needs >=1 GATE-REFUSED/
# PERSISTENTLY-ABORTED row per the CD branch) -- mark one conditional row
# PERSISTENTLY-ABORTED-equivalent evidence via a primary GATE-REFUSED retry
# that was later promoted is over-complex; simplest: this payload's own
# base CD throttle-evidence clause would FAIL too pre-mirror, which is fine
# since it is UNREACHABLE either way (m1 tests the mirror in isolation) --
# retagged run_status COMPLETE (matches B4's own convention: complete
# conditional arm) to isolate the U7 mirror as the sole clause in play.
PAYLOADS["B3-NEG"]["run_status"] = "COMPLETE"
PAYLOADS["B3-NEG"]["band"] = {"interval_resolved_Ks": [], "incomplete_at_K": None}

# ---- B4: paid conditional arm silently absent from the ledger ----
PAYLOADS["B4"] = dict(
    run_status="COMPLETE", realized=12.00, ledger=mk_pairs_completed(ALL12, elapsed=1.00),
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=4, band={"interval_resolved_Ks": [], "incomplete_at_K": None},
    conditional={"launched": True, "qualifier_band": None},
    charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="PASS",
)

# ---- D1 / D1' (KW10.8 probes, m7 verification) ----
ledger_D1 = [{"K": K, "seed": s, "arm": "primary", "attempt_n": 1, "status": "GATE-REFUSED",
              "elapsed_h": 0.0, "ceiling_charged": False} for (K, s) in ALL12]
PAYLOADS["D1"] = dict(
    run_status="COMPLETE", realized=0.0, ledger=ledger_D1,
    primary_canonical=12, primary_canonical_by_K=None,  # 12 genuine files on disk (impossible in reality, but this is exactly the adversarial probe: canonical files exist, ledger doesn't reflect it honestly)
    conditional_canonical=0, band={"interval_resolved_Ks": [], "incomplete_at_K": None},
    conditional=None, charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="PASS",
)
PAYLOADS["D1'"] = dict(PAYLOADS["D1"])
PAYLOADS["D1'"]["ledger"] = []
PAYLOADS["D1'"]["realized"] = 0.0

# ---- D2 / D2' (KW10.4 probes, m3 verification) ----
PAYLOADS["D2"] = dict(
    run_status="COMPLETE", realized=12.00, ledger=mk_pairs_completed(ALL12, elapsed=1.00),
    primary_canonical=12, primary_canonical_by_K={26: 4, 28: 4, 30: 4},
    conditional_canonical=4,  # 4 conditional canonical files genuinely on disk
    band={"interval_resolved_Ks": [], "incomplete_at_K": None},
    conditional=None,  # report's own conditional block is null
    charged_vs_measured=None, trigger=None,
    expect_NEW="FAIL", expect_OLD="PASS",
)
PAYLOADS["D2'"] = dict(PAYLOADS["D2"])
PAYLOADS["D2'"]["conditional"] = {"launched": False, "qualifier_band": None}
