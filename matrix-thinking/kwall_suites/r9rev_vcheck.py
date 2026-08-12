#!/usr/bin/env python3
"""
Rev-9 revision agent's independent transcription of validity_check,
written fresh from the CURRENT (post-Rev-9-edit)
matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md live-body text
(universal assertions 1-8 + the 5 per-run_status branches), per the
R9 report's own §0 METHOD (re-derive, don't trust cached results).

Two transcriptions:
  - validity_check_OLD  : pre-Rev-8 rules (through R7's J1-J7 only)
  - validity_check_NEW  : current text, i.e. through Rev-9's M1/m1-m7

Every payload is a dict with keys:
  run_status, realized, ledger (list of attempt-row dicts),
  primary_canonical (int, # files matching earlyln_K*_s*.json with
    status==COMPLETED on disk), conditional_canonical (int, # files
    in the conditional dir with status==COMPLETED), band (dict),
  conditional (dict or None), charged_vs_measured (dict or None),
  trigger (dict or None)

ledger rows: {"K":.., "seed":.., "arm": "primary"/"conditional",
  "attempt_n":.., "status":.., "elapsed_h":.., "ceiling_charged": bool}
"""

ALL_PRIMARY_PAIRS = [(K, s) for K in (26, 28, 30) for s in (0, 1, 2, 3)]


def completed_primary_pairs(ledger):
    return {(a["K"], a["seed"]) for a in ledger
            if a["arm"] == "primary" and a["status"] == "COMPLETED"}


def completed_conditional_pairs(ledger):
    return {(a["K"], a["seed"]) for a in ledger
            if a["arm"] == "conditional" and a["status"] == "COMPLETED"}


def primary_pairs_with_row(ledger):
    return {(a["K"], a["seed"]) for a in ledger if a["arm"] == "primary"}


# ---------------------------------------------------------------- NEW ----

def validity_check_NEW(p):
    f = []

    # Universal 1
    if p["run_status"] not in {"COMPLETE", "COMPLETE-DEGRADED",
                                "EXHAUSTED-BUDGET",
                                "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"}:
        f.append("U1-run_status-excluded")
        return False, f  # STOPPED-BY-OPERATOR etc never reach further checks

    # Universal 2
    if not (p["realized"] <= 15.50):
        f.append("U2-realized>15.50")

    # Universal 3
    ledger_sum = sum(a["elapsed_h"] for a in p["ledger"] if a["arm"] == "primary")
    if abs(p["realized"] - ledger_sum) > 1e-6:
        f.append(f"U3-bookkeeping({p['realized']:.4f}!={ledger_sum:.4f})")

    # Universal 6 (only the resolution-literal half is exercised by our payloads)
    trig = p.get("trigger")
    if trig is not None:
        if trig.get("resolution") not in {"unanimous", "tie-break-min",
                                           "TRIGGER-UNRESOLVED"}:
            f.append("U6-resolution-not-in-enum")

    # Universal 7 (K1 mirror + m3 Otherwise-arm addition)
    cond = p.get("conditional")
    qb = cond.get("qualifier_band") if cond else None
    if qb is not None:
        launched = cond.get("launched")
        n_cc = p["conditional_canonical"]
        n_cc_ledger = len(completed_conditional_pairs(p["ledger"]))
        ok_a = (launched is True and n_cc == 4 and n_cc_ledger == 4)
        ok_b = (launched is False and (p.get("trigger") or {}).get("K_trig") == 32)
        if not (ok_a or ok_b):
            f.append("U7-neither(a)nor(b)")
    elif cond is not None and cond.get("launched") is True:
        n_cc = p["conditional_canonical"]
        n_cc_ledger = len(completed_conditional_pairs(p["ledger"]))
        if n_cc != n_cc_ledger:
            f.append(f"U7-mirror:cond_canon{n_cc}!=ledgerCOMPLETED{n_cc_ledger}")
        elif not (n_cc < 4):
            f.append("U7-mirror:n_cond_canon<4-fails(4/4-with-null-band)")
    else:
        # Otherwise arm (§R9 m3): conditional is None, or qualifier_band is
        # None and launched is False/absent -- conditional canonical dir
        # must contain 0 COMPLETED files.
        n_cc = p["conditional_canonical"]
        if n_cc != 0:
            f.append(f"U7-otherwise:cond_canon{n_cc}!=0(m3)")

    # Universal 8 (§R8 K3, NEW)
    cvm = p.get("charged_vs_measured")
    if cvm is not None:
        ccgh_recomputed = sum(a["elapsed_h"] for a in p["ledger"]
                               if a.get("ceiling_charged"))
        if abs(cvm["ceiling_charged_gpu_h"] - ccgh_recomputed) > 1e-6:
            f.append(f"U8-ccgh({cvm['ceiling_charged_gpu_h']:.4f}!="
                      f"{ccgh_recomputed:.4f})")
        elif p["realized"] > 0:
            frac_recomputed = ccgh_recomputed / p["realized"]
            if abs(cvm["ceiling_charged_fraction"] - frac_recomputed) > 1e-6:
                f.append(f"U8-frac({cvm['ceiling_charged_fraction']:.6f}!="
                          f"{frac_recomputed:.6f})")

    # Per-run_status branch
    rs = p["run_status"]
    band = p.get("band", {})
    if rs == "COMPLETE":
        if band.get("interval_resolved_Ks") == [] and band.get("incomplete_at_K") is None:
            # STRICT branch
            if p["primary_canonical"] != 12:
                f.append("COMPLETE/strict:canonical!=12")
            # §R9 m7: J1(a)/(b) now apply to the strict branch too.
            if primary_pairs_with_row(p["ledger"]) != set(ALL_PRIMARY_PAIRS):
                f.append("COMPLETE/strict-m7-J1a:missing-pair-row")
            cpp = completed_primary_pairs(p["ledger"])
            if len(cpp) != p["primary_canonical"]:
                f.append(f"COMPLETE/strict-m7-J1b:canonical{p['primary_canonical']}"
                          f"!=COMPLETED{len(cpp)}")
        else:
            named = set(band.get("interval_resolved_Ks", []) or []) | \
                    set(band.get("incomplete_at_K", []) or [])
            for K in (26, 28, 30):
                cnt = p["primary_canonical_by_K"][K]
                if K in named and not (cnt < 4):
                    f.append(f"COMPLETE/otherwise:K{K}-named-but-count>=4")
                if K not in named and cnt != 4:
                    f.append(f"COMPLETE/otherwise:K{K}-unnamed-but-count!=4")
            if primary_pairs_with_row(p["ledger"]) != set(ALL_PRIMARY_PAIRS):
                f.append("COMPLETE/otherwise-J1a:missing-pair-row")
            cpp = completed_primary_pairs(p["ledger"])
            total_canon = sum(p["primary_canonical_by_K"].values())
            if total_canon != len(cpp):
                f.append("COMPLETE/otherwise-J1b:canonical!=COMPLETED")
            if len(cpp) < 1:
                f.append("COMPLETE/otherwise K2: 0 distinct COMPLETED primary pairs")
    elif rs == "COMPLETE-DEGRADED":
        if primary_pairs_with_row(p["ledger"]) != set(ALL_PRIMARY_PAIRS):
            f.append("CD:missing-pair-row")
        cpp = completed_primary_pairs(p["ledger"])
        if p["primary_canonical"] != len(cpp):
            f.append("CD:canonical!=COMPLETED")
        has_throttle_evidence = any(
            a["status"] == "GATE-REFUSED" or a.get("persistently_aborted")
            for a in p["ledger"])
        if not has_throttle_evidence:
            f.append("CD:no-throttle-evidence")
        if len(cpp) < 1:
            f.append("CD K2: 0 distinct COMPLETED primary pairs")
    elif rs == "EXHAUSTED-BUDGET":
        if not (p["realized"] > 13.80):
            f.append("EB:realized<=13.80")
        if not (p["primary_canonical"] < 12):
            f.append("EB:canonical>=12")
        if not any(a["arm"] == "primary" and a["attempt_n"] == 1 and
                   a["status"] == "GATE-REFUSED" for a in p["ledger"]):
            f.append("EB:no-first-attempt-GATE-REFUSED")
        cvm = p["charged_vs_measured"]
        if not (cvm["ceiling_charged_fraction"] <= 0.50):
            f.append("EB J4: ceiling_charged_fraction not <=0.50")
    elif rs == "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE":
        if not (p["realized"] > 13.80):
            f.append("EB/J4-base:realized<=13.80")
        if not (p["primary_canonical"] < 12):
            f.append("EB/J4-base:canonical>=12")
        if not any(a["arm"] == "primary" and a["attempt_n"] == 1 and
                   a["status"] == "GATE-REFUSED" for a in p["ledger"]):
            f.append("EB/J4-base:no-first-attempt-GATE-REFUSED")
        cvm = p["charged_vs_measured"]
        if not (cvm["ceiling_charged_fraction"] > 0.50):
            f.append("EB/J4-frac")

    return (len(f) == 0), f


# ---------------------------------------------------------------- OLD ----
# pre-Rev-8 (through R7's J1-J7 only): no K1 mirror-for-null-band branch,
# no K2 >=1-distinct-COMPLETED clause, no K3 universal assertion 8
# (ccgh/fraction trusted as self-reported), no m3/m7 (those are Rev-9
# additions layered on top of Rev-8, moot for a pre-Rev-8 baseline).

def validity_check_OLD(p):
    f = []

    if p["run_status"] not in {"COMPLETE", "COMPLETE-DEGRADED",
                                "EXHAUSTED-BUDGET",
                                "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"}:
        f.append("U1-run_status-excluded")
        return False, f

    if not (p["realized"] <= 15.50):
        f.append("U2-realized>15.50")

    ledger_sum = sum(a["elapsed_h"] for a in p["ledger"] if a["arm"] == "primary")
    if abs(p["realized"] - ledger_sum) > 1e-6:
        f.append(f"U3-bookkeeping({p['realized']:.4f}!={ledger_sum:.4f})")

    trig = p.get("trigger")
    if trig is not None:
        if trig.get("resolution") not in {"unanimous", "tie-break-min",
                                           "TRIGGER-UNRESOLVED"}:
            f.append("U6-resolution-not-in-enum")

    # Pre-K1 universal assertion 7 (R7 J5): fires ONLY when qualifier_band
    # is non-None. Silent (no constraint) when band is None, regardless of
    # `launched` or real disk evidence -- this silence is exactly KW9.1/
    # KW9.7's hole.
    cond = p.get("conditional")
    qb = cond.get("qualifier_band") if cond else None
    if qb is not None:
        launched = cond.get("launched")
        n_cc = p["conditional_canonical"]
        n_cc_ledger = len(completed_conditional_pairs(p["ledger"]))
        ok_a = (launched is True and n_cc == 4 and n_cc_ledger == 4)
        ok_b = (launched is False and (p.get("trigger") or {}).get("K_trig") == 32)
        if not (ok_a or ok_b):
            f.append("U7-neither(a)nor(b)")
    # else: silent, pre-K1.

    # No universal assertion 8 pre-Rev-8: ceiling_charged_fraction /
    # ceiling_charged_gpu_h are TRUSTED as self-reported, never recomputed.

    rs = p["run_status"]
    band = p.get("band", {})
    if rs == "COMPLETE":
        if band.get("interval_resolved_Ks") == [] and band.get("incomplete_at_K") is None:
            if p["primary_canonical"] != 12:
                f.append("COMPLETE/strict:canonical!=12")
            # pre-Rev-9: no ledger clause on the strict branch at all.
        else:
            named = set(band.get("interval_resolved_Ks", []) or []) | \
                    set(band.get("incomplete_at_K", []) or [])
            for K in (26, 28, 30):
                cnt = p["primary_canonical_by_K"][K]
                if K in named and not (cnt < 4):
                    f.append(f"COMPLETE/otherwise:K{K}-named-but-count>=4")
                if K not in named and cnt != 4:
                    f.append(f"COMPLETE/otherwise:K{K}-unnamed-but-count!=4")
            if primary_pairs_with_row(p["ledger"]) != set(ALL_PRIMARY_PAIRS):
                f.append("COMPLETE/otherwise-J1a:missing-pair-row")
            cpp = completed_primary_pairs(p["ledger"])
            total_canon = sum(p["primary_canonical_by_K"].values())
            if total_canon != len(cpp):
                f.append("COMPLETE/otherwise-J1b:canonical!=COMPLETED")
            # pre-Rev-8: NO K2 >=1-distinct-COMPLETED clause.
    elif rs == "COMPLETE-DEGRADED":
        if primary_pairs_with_row(p["ledger"]) != set(ALL_PRIMARY_PAIRS):
            f.append("CD:missing-pair-row")
        cpp = completed_primary_pairs(p["ledger"])
        if p["primary_canonical"] != len(cpp):
            f.append("CD:canonical!=COMPLETED")
        has_throttle_evidence = any(
            a["status"] == "GATE-REFUSED" or a.get("persistently_aborted")
            for a in p["ledger"])
        if not has_throttle_evidence:
            f.append("CD:no-throttle-evidence")
        # pre-Rev-8: NO K2 clause.
    elif rs == "EXHAUSTED-BUDGET":
        if not (p["realized"] > 13.80):
            f.append("EB:realized<=13.80")
        if not (p["primary_canonical"] < 12):
            f.append("EB:canonical>=12")
        if not any(a["arm"] == "primary" and a["attempt_n"] == 1 and
                   a["status"] == "GATE-REFUSED" for a in p["ledger"]):
            f.append("EB:no-first-attempt-GATE-REFUSED")
        cvm = p["charged_vs_measured"]
        if not (cvm["ceiling_charged_fraction"] <= 0.50):  # TRUSTED, not recomputed
            f.append("EB J4: ceiling_charged_fraction not <=0.50")
    elif rs == "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE":
        if not (p["realized"] > 13.80):
            f.append("EB/J4-base:realized<=13.80")
        if not (p["primary_canonical"] < 12):
            f.append("EB/J4-base:canonical>=12")
        if not any(a["arm"] == "primary" and a["attempt_n"] == 1 and
                   a["status"] == "GATE-REFUSED" for a in p["ledger"]):
            f.append("EB/J4-base:no-first-attempt-GATE-REFUSED")
        cvm = p["charged_vs_measured"]
        if not (cvm["ceiling_charged_fraction"] > 0.50):  # TRUSTED
            f.append("EB/J4-frac")

    return (len(f) == 0), f
