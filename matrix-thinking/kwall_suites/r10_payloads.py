#!/usr/bin/env python3
"""Round-10 payload reconstruction, transcribed from the design's in-text test
list (DRAFT-R9, :2529-2826).  Names follow the R8/R9 convention:
L1-L7,L7' = reportable outcomes; A1-A7,A6' = R5/R6/R7 adversarials;
B1..B4 = R8's eight extensions."""
from r10_vcheck import validity_check
from fractions import Fraction as F

PRIM = [(K, s) for K in (26, 28, 30) for s in range(4)]


def row(K, seed, arm, n, h, status, cc):
    return {"K": K, "seed": seed, "arm": arm, "attempt_n": n, "elapsed_h": h,
            "status": status, "d_override": K + 1, "ceiling_charged": cc}


def canon(pairs):
    return [{"K": K, "seed": s, "status": "COMPLETED"} for (K, s) in pairs]


def rep(run_status, attempts, realized, ccgh, frac, *, label="FRONTIER-AT-K*=26",
        irk=None, iak=None, resolution="unanimous", K_trig=26, conditional=None):
    return {"run_status": run_status,
            "band": {"label": label, "interval_resolved_Ks": [] if irk is None else irk,
                     "incomplete_at_K": iak},
            "trigger": {"resolution": resolution, "K_trig": K_trig},
            "smoke": {"d_override_smoke": "PASS"},
            "ledger": {"realized_gpu_h_final": realized, "attempts": attempts},
            "charged_vs_measured": {"ceiling_charged_gpu_h": ccgh,
                                    "ceiling_charged_fraction": frac},
            "conditional": conditional}


def disk(prim_pairs, n_cond=0):
    return {"primary_canonical": canon(prim_pairs),
            "cond_canonical": [{"status": "COMPLETED"} for _ in range(n_cond)]}


P = {}   # name -> (report, disk, expected_NEW)

# ---------------- L1: COMPLETE, 12/12 strict ----------------
att = [row(K, s, "primary", 1, "1.00", "COMPLETED", False) for (K, s) in PRIM]
P["L1"] = (rep("COMPLETE", att, "12.00", "0.00", "0.00"), disk(PRIM), "PASS")

# ---------------- L2: INCOMPLETE-AT-K (COMPLETE + incomplete_at_K=[30]) -------
c11 = [p for p in PRIM if p != (30, 3)]
att = [row(K, s, "primary", 1, "1.00", "COMPLETED", False) for (K, s) in c11]
att += [row(30, 3, "primary", 1, "1.20", "CRASHED-RECOVERED", True),
        row(30, 3, "primary", 2, "1.20", "CRASHED-RECOVERED", True)]
P["L2"] = (rep("COMPLETE", att, "13.40", "2.40", str(F("2.40") / F("13.40")),
               label="INCOMPLETE-AT-K", iak=[30]), disk(c11), "PASS")

# ---------------- L3: COMPLETE-DEGRADED sub-case (i) -------------------------
att = [row(K, s, "primary", 1, "1.00", "COMPLETED", False) for (K, s) in c11]
att += [row(30, 3, "primary", 1, "1.20", "CRASHED-RECOVERED", True),
        row(30, 3, "primary", 2, "0.00", "GATE-REFUSED", False)]
P["L3"] = (rep("COMPLETE-DEGRADED", att, "12.20", "1.20",
               str(F("1.20") / F("12.20"))), disk(c11), "PASS")

# ---------------- L4: COMPLETE-DEGRADED sub-case (ii)/(iii) ------------------
att = [row(K, s, "primary", 1, "0.60", "COMPLETED", False) for (K, s) in PRIM]
att += [row(26, s, "conditional", 1, "2.00", "COMPLETED", False) for s in range(3)]
att += [row(26, 3, "conditional", 1, "0.00", "GATE-REFUSED", False)]
P["L4"] = (rep("COMPLETE-DEGRADED", att, "13.20", "0.00", "0.00",
               conditional={"launched": True, "per_seed": [0, 1, 2],
                            "qualifier_band": None}),
           disk(PRIM, n_cond=3), "PASS")

# ---------------- L5: EXHAUSTED-BUDGET, realized 14.55, 9 canonical ----------
c9 = PRIM[:9]
att = [row(K, s, "primary", 1, "1.08", "COMPLETED", False) for (K, s) in c9[:8]]
att += [row(*c9[8], "primary", 1, "1.11", "COMPLETED", False)]
for (K, s) in PRIM[9:11]:
    att += [row(K, s, "primary", 1, "1.20", "CRASHED-RECOVERED", True),
            row(K, s, "primary", 2, "1.20", "CRASHED-RECOVERED", True)]
att += [row(30, 3, "primary", 1, "0.00", "GATE-REFUSED", False)]
P["L5"] = (rep("EXHAUSTED-BUDGET", att, "14.55", "4.80",
               str(F("4.80") / F("14.55"))), disk(c9), "PASS")

# ---------------- L6: EBSO — TRANSCRIBED VERBATIM from :2601-2617 ------------
# 10 single-attempt CRASHED-RECOVERED primary pairs (1.20 each = 12.00,
# ceiling-charged) + 1 COMPLETED pair (measured, elapsed_h=2.00, 1 canonical)
# + 1 GATE-REFUSED pair at attempt_n=1 (0.0).  realized=14.00, ccgh=12.00,
# fraction = 12.00/14.00 EXACTLY.
att = [row(K, s, "primary", 1, "1.20", "CRASHED-RECOVERED", True)
       for (K, s) in PRIM[:10]]
att += [row(30, 2, "primary", 1, "2.00", "COMPLETED", False),
        row(30, 3, "primary", 1, "0.00", "GATE-REFUSED", False)]
P["L6"] = (rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", att, "14.00", "12.00",
               str(F("12.00") / F("14.00"))), disk([(30, 2)]), "PASS")
# variant: the same payload with the ROUNDED 4-dp literal instead of the quotient
P["L6-literal0.8571"] = (rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", att, "14.00",
                             "12.00", "0.8571"), disk([(30, 2)]), "FAIL")

# ---------------- L7: STOPPED-BY-OPERATOR -----------------------------------
P["L7"] = (rep("STOPPED-BY-OPERATOR", [], "5.00", "0.00", "0.00"),
           disk([]), "FAIL")

# ---------------- L7': tie-break-min ----------------------------------------
att = [row(K, s, "primary", 1, "1.00", "COMPLETED", False) for (K, s) in PRIM]
P["L7'"] = (rep("COMPLETE", att, "12.00", "0.00", "0.00",
                resolution="tie-break-min"), disk(PRIM), "PASS")
P["L7'-fstring(pre-J2)"] = (rep("COMPLETE", att, "12.00", "0.00", "0.00",
                                resolution="tie-break-min, candidates were [26, 28]"),
                            disk(PRIM), "FAIL")

# ---------------- A1: audit-R5 no-op ----------------------------------------
P["A1"] = (rep("COMPLETE", [], "0.00", "0.00", "0.00"), disk([]), "FAIL")

# ---------------- A2: near-miss #1 (12 canonical, EB, realized 14.40) -------
att = [row(K, s, "primary", 1, "1.20", "COMPLETED", False) for (K, s) in PRIM]
P["A2"] = (rep("EXHAUSTED-BUDGET", att, "14.40", "0.00", "0.00"),
           disk(PRIM), "FAIL")

# ---------------- A3: near-miss #2 (STOPPED-BY-OPERATOR, no marker) ---------
P["A3"] = (rep("STOPPED-BY-OPERATOR", [], "0.00", "0.00", "0.00"), disk([]), "FAIL")

# ---------------- A4: near-miss #3 disk state mis-claimed as EB -------------
att = [row(K, s, "primary", 1, "0.60", "COMPLETED", False) for (K, s) in c11]
att += [row(30, 3, "primary", 1, "1.20", "CRASHED-RECOVERED", True),
        row(30, 3, "primary", 2, "1.20", "CRASHED-RECOVERED", True)]
P["A4"] = (rep("EXHAUSTED-BUDGET", att, "9.00", "2.40",
               str(F("2.40") / F("9.00"))), disk(c11), "FAIL")

# ---------------- A5: ENHANCED no-op (KW8.1) --------------------------------
P["A5"] = (rep("COMPLETE", [], "0.00", "0.00", "0.00", iak=[26, 28, 30],
               label="INCOMPLETE-AT-K"), disk([]), "FAIL")

# ---------------- A6 / A6' (the §R8 K4 rebuild) -----------------------------
a6 = [row(K, s, "primary", 1, "1.20", "CRASHED-RECOVERED", True) for (K, s) in PRIM[:9]]
a6 += [row(30, 1, "primary", 1, "1.20", "CRASHED-RECOVERED", True),
       row(30, 1, "primary", 2, "1.20", "CRASHED-RECOVERED", True),
       row(30, 2, "primary", 1, "1.00", "COMPLETED", False),
       row(30, 3, "primary", 1, "0.00", "GATE-REFUSED", False)]
A6FRAC = str(F("13.20") / F("14.20"))
P["A6"] = (rep("EXHAUSTED-BUDGET", a6, "14.20", "13.20", A6FRAC), disk([(30, 2)]), "FAIL")
P["A6'"] = (rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", a6, "14.20", "13.20", A6FRAC),
            disk([(30, 2)]), "PASS")
P["A6-literal0.9296"] = (rep("EXHAUSTED-BUDGET", a6, "14.20", "13.20", "0.9296"),
                         disk([(30, 2)]), "FAIL")
P["A6'-literal0.9296"] = (rep("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", a6, "14.20",
                              "13.20", "0.9296"), disk([(30, 2)]), "FAIL")

# ---------------- A7: fabricated conditional arm ----------------------------
att = [row(K, s, "primary", 1, "1.00", "COMPLETED", False) for (K, s) in PRIM]
P["A7"] = (rep("COMPLETE", att, "12.00", "0.00", "0.00",
               conditional={"launched": True, "per_seed": [],
                            "qualifier_band": "SLOW-CONVERGENCE-AT-160K"}),
           disk(PRIM, n_cond=0), "FAIL")

# ---------------- B1 / B1': zero-cost no-op via 12 GATE-REFUSED rows --------
gr12 = [row(K, s, "primary", 1, "0.00", "GATE-REFUSED", False) for (K, s) in PRIM]
P["B1"] = (rep("COMPLETE", gr12, "0.00", "0.00", "0.00", iak=[26, 28, 30],
               label="INCOMPLETE-AT-K"), disk([]), "FAIL")
P["B1'"] = (rep("COMPLETE-DEGRADED", gr12, "0.00", "0.00", "0.00"), disk([]), "FAIL")

# ---------------- B2 / B2': A6's ledger, charged_vs_measured mis-declared ---
P["B2"] = (rep("EXHAUSTED-BUDGET", a6, "14.20", "13.20", "0.20"), disk([(30, 2)]), "FAIL")
P["B2'"] = (rep("EXHAUSTED-BUDGET", a6, "14.20", "2.84", "0.20"), disk([(30, 2)]), "FAIL")

# ---------------- B3 family: conditional 3/4 and 4/4 ------------------------
b3att = [row(K, s, "primary", 1, "0.60", "COMPLETED", False) for (K, s) in PRIM]
b3att += [row(26, s, "conditional", 1, "2.00", "COMPLETED", False) for s in range(3)]
b3att += [row(26, 3, "conditional", 1, "0.00", "GATE-REFUSED", False)]
P["B3-OLD-STYLE"] = (rep("COMPLETE-DEGRADED", b3att, "13.20", "0.00", "0.00",
                         conditional={"launched": True, "per_seed": [0, 1, 2],
                                      "qualifier_band": "SLOW-CONVERGENCE-AT-160K"}),
                     disk(PRIM, n_cond=3), "FAIL")
P["B3-AMENDED"] = (rep("COMPLETE-DEGRADED", b3att, "13.20", "0.00", "0.00",
                       conditional={"launched": True, "per_seed": [0, 1, 2],
                                    "qualifier_band": None}),
                   disk(PRIM, n_cond=3), "PASS")
b3neg = [row(K, s, "primary", 1, "0.60", "COMPLETED", False) for (K, s) in PRIM]
b3neg += [row(26, s, "conditional", 1, "2.00", "COMPLETED", False) for s in range(4)]
P["B3-NEG"] = (rep("COMPLETE", b3neg, "15.20", "0.00", "0.00",
                   conditional={"launched": True, "per_seed": [0, 1, 2, 3],
                                "qualifier_band": None}),
               disk(PRIM, n_cond=4), "FAIL")

# ---------------- B4: paid conditional arm absent from the ledger -----------
b4 = [row(K, s, "primary", 1, "0.60", "COMPLETED", False) for (K, s) in PRIM]
P["B4"] = (rep("COMPLETE", b4, "7.20", "0.00", "0.00",
               conditional={"launched": True, "per_seed": [0, 1, 2, 3],
                            "qualifier_band": None}),
           disk(PRIM, n_cond=4), "FAIL")

# =================== DRIVE ==================================================
if __name__ == "__main__":
    flips, mism = [], []
    print("%-22s %-6s %-6s %-6s  %s" % ("payload", "OLD", "NEW", "want", "NEW failure reasons"))
    print("-" * 110)
    for name, (r, d, want) in P.items():
        fo = validity_check(r, d, "OLD")
        fn = validity_check(r, d, "NEW")
        vo = "PASS" if not fo else "FAIL"
        vn = "PASS" if not fn else "FAIL"
        if vn != want:
            mism.append(name)
        if vo != vn:
            flips.append(name)
        print("%-22s %-6s %-6s %-6s  %s" % (name, vo, vn, want, fn if fn else "[]"))
    print("-" * 110)
    print("FLIPS (OLD != NEW):", sorted(flips))
    print("count:", len(flips))
    print("MISMATCHES vs stated expectation:", mism if mism else "NONE")
    claimed = {"B1", "B1'", "B2", "B2'", "B3-NEG", "B4"}
    core = {f for f in flips if not f.endswith("literal0.8571")}
    print("set-equality vs claimed six-flip set:", core == claimed,
          "| extra:", sorted(core - claimed), "| missing:", sorted(claimed - core))
