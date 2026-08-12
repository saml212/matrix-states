#!/usr/bin/env python3
"""
ROUND-10 independent transcription of the K-wall design's `validity_check`.

NEW  = current text, DRAFT-R9 (doc :2319-2527), transcribed by hand this round.
OLD  = pre-Rev-8 text (commit ad2bf48, :2254-2385), i.e. R7's J1-J7 only.

No prior round's harness was consulted or reused.
"""
from fractions import Fraction as F

PRIMARY_CEIL = F("1.20")
COND_CEIL = F("2.32")
BAND_LABELS = {"FRONTIER-AT-K*=24", "FRONTIER-AT-K*=26", "FRONTIER-AT-K*=28",
               "FRONTIER-AT-K*=30", "GRADUAL-DECAY", "NON-MONOTONE-UNRESOLVED",
               "INCOMPLETE-AT-K"}
RESOLUTIONS = {"unanimous", "tie-break-min", "TRIGGER-UNRESOLVED"}
ACCEPT = {"COMPLETE", "COMPLETE-DEGRADED", "EXHAUSTED-BUDGET",
          "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"}
EPS = F(1, 1000000)


def _prim_canon(disk):
    return [f for f in disk["primary_canonical"]]


def _prim_canon_count(disk):
    return len([f for f in disk["primary_canonical"] if f["status"] == "COMPLETED"])


def _prim_canon_count_K(disk, K):
    return len([f for f in disk["primary_canonical"]
                if f["K"] == K and f["status"] == "COMPLETED"])


def _cond_canon_completed(disk):
    return len([f for f in disk["cond_canonical"] if f["status"] == "COMPLETED"])


def _pairs(att, arm, status=None):
    return {(a["K"], a["seed"]) for a in att
            if a["arm"] == arm and (status is None or a["status"] == status)}


def validity_check(rep, disk, mode):
    """mode in {'OLD','NEW'} -> list of failure reasons ([] == PASS)."""
    f = []
    led = rep["ledger"]
    att = led["attempts"]
    realized = F(str(led["realized_gpu_h_final"]))

    # ---- universal 1
    if rep["run_status"] not in ACCEPT:
        f.append("U1: run_status %s not in accept-set" % rep["run_status"])
        return f                      # excluded outright; no branch fires
    # ---- universal 2
    if not realized <= F("15.50"):
        f.append("U2: realized %s > 15.50" % realized)
    # ---- universal 3
    tot = sum((F(str(a["elapsed_h"])) for a in att), F(0))
    if abs(realized - tot) > EPS:
        f.append("U3: realized %s != sum(elapsed_h) %s" % (realized, tot))
    # ---- universal 4
    if not all(a["K"] + 1 == a["d_override"] for a in att):
        f.append("U4: d_override != K+1 somewhere")
    # ---- universal 5
    if not all(v == "PASS" for v in rep["smoke"].values()):
        f.append("U5: smoke not all PASS")
    # ---- universal 6
    if rep["band"]["label"] not in BAND_LABELS:
        f.append("U6: band label %r" % rep["band"]["label"])
    if rep["trigger"]["resolution"] not in RESOLUTIONS:
        f.append("U6: trigger.resolution %r" % rep["trigger"]["resolution"])
    # ---- universal 7
    cond = rep["conditional"]
    n_cond_canon = _cond_canon_completed(disk)
    n_cond_led = len(_pairs(att, "conditional", "COMPLETED"))
    if cond is not None and cond["qualifier_band"] is not None:
        a_ok = (cond["launched"] is True and n_cond_canon == 4 and n_cond_led == 4)
        b_ok = (cond["launched"] is False and rep["trigger"]["K_trig"] == 32)
        if not (a_ok or b_ok):
            f.append("U7: qualifier_band with neither clause (a) nor (b)")
    elif (cond is not None and cond["qualifier_band"] is None
          and cond.get("launched") is True):
        if mode == "NEW":                       # R8 K1 mirror clause
            if n_cond_canon != n_cond_led:
                f.append("U7-mirror: cond_canon %d != ledger cond COMPLETED %d"
                         % (n_cond_canon, n_cond_led))
            if not n_cond_canon < 4:
                f.append("U7-mirror: cond_canon %d not < 4" % n_cond_canon)
    else:
        # conditional is None, or band None and launched False/absent
        if mode == "NEW":                       # R9 m3
            if n_cond_canon != 0:
                f.append("U7-otherwise: cond_canon %d != 0" % n_cond_canon)
    # ---- universal 8 (NEW only, R8 K3)
    if mode == "NEW":
        cvm = rep["charged_vs_measured"]
        ccgh_rec = sum((F(str(a["elapsed_h"])) for a in att if a["ceiling_charged"]), F(0))
        if abs(F(str(cvm["ceiling_charged_gpu_h"])) - ccgh_rec) > EPS:
            f.append("U8-ccgh: declared %s != recomputed %s"
                     % (cvm["ceiling_charged_gpu_h"], ccgh_rec))
        if realized > 0:
            frac_rec = ccgh_rec / realized
            if abs(F(str(cvm["ceiling_charged_fraction"])) - frac_rec) > EPS:
                f.append("U8-frac: declared %s != recomputed %s"
                         % (float(F(str(cvm['ceiling_charged_fraction']))), float(frac_rec)))

    # ---- per-run_status (exactly one branch)
    rs = rep["run_status"]
    band = rep["band"]
    if rs == "COMPLETE":
        if band["interval_resolved_Ks"] == [] and band["incomplete_at_K"] is None:
            # STRICT
            if _prim_canon_count(disk) != 12:
                f.append("COMPLETE/strict: primary canonical %d != 12"
                         % _prim_canon_count(disk))
            if mode == "NEW":                   # R9 m7
                missing = [(K, s) for K in (26, 28, 30) for s in range(4)
                           if (K, s) not in _pairs(att, "primary")]
                if missing:
                    f.append("COMPLETE/strict-m7-J1a: %d primary pairs with no row"
                             % len(missing))
                if _prim_canon_count(disk) != len(_pairs(att, "primary", "COMPLETED")):
                    f.append("COMPLETE/strict-m7-J1b: canonical %d != ledger COMPLETED pairs %d"
                             % (_prim_canon_count(disk),
                                len(_pairs(att, "primary", "COMPLETED"))))
        else:
            named = set(band["interval_resolved_Ks"] or []) | set(band["incomplete_at_K"] or [])
            for K in (26, 28, 30):
                c = _prim_canon_count_K(disk, K)
                if K in named:
                    if not c < 4:
                        f.append("COMPLETE/otherwise: K=%d count %d not <4" % (K, c))
                else:
                    if c != 4:
                        f.append("COMPLETE/otherwise: K=%d count %d != 4" % (K, c))
            missing = [(K, s) for K in (26, 28, 30) for s in range(4)
                       if (K, s) not in _pairs(att, "primary")]
            if missing:
                f.append("COMPLETE/otherwise-J1a: %d primary pairs with no row" % len(missing))
            if _prim_canon_count(disk) != len(_pairs(att, "primary", "COMPLETED")):
                f.append("COMPLETE/otherwise-J1b: canonical %d != ledger COMPLETED pairs %d"
                         % (_prim_canon_count(disk), len(_pairs(att, "primary", "COMPLETED"))))
            if mode == "NEW":                   # R8 K2
                if not len(_pairs(att, "primary", "COMPLETED")) >= 1:
                    f.append("COMPLETE/otherwise-K2: 0 COMPLETED primary pairs")
    elif rs == "COMPLETE-DEGRADED":
        missing = [(K, s) for K in (26, 28, 30) for s in range(4)
                   if (K, s) not in _pairs(att, "primary")]
        if missing:
            f.append("CD-J1a: %d primary pairs with no row" % len(missing))
        if _prim_canon_count(disk) != len(_pairs(att, "primary", "COMPLETED")):
            f.append("CD-J1b: canonical %d != ledger COMPLETED pairs %d"
                     % (_prim_canon_count(disk), len(_pairs(att, "primary", "COMPLETED"))))
        if not any(a["status"] in ("GATE-REFUSED", "PERSISTENTLY-ABORTED") for a in att):
            f.append("CD: no throttle evidence row")
        if mode == "NEW":                       # R8 K2
            if not len(_pairs(att, "primary", "COMPLETED")) >= 1:
                f.append("CD-K2: 0 COMPLETED primary pairs")
    elif rs in ("EXHAUSTED-BUDGET", "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"):
        if not realized > F("13.80"):
            f.append("EB base1: realized %s not >13.80" % realized)
        if not _prim_canon_count(disk) < 12:
            f.append("EB base2: primary canonical %d not <12" % _prim_canon_count(disk))
        if not any(a["arm"] == "primary" and a["attempt_n"] == 1
                   and a["status"] == "GATE-REFUSED" for a in att):
            f.append("EB base3: no primary attempt_n==1 GATE-REFUSED row")
        frac = F(str(rep["charged_vs_measured"]["ceiling_charged_fraction"]))
        if rs == "EXHAUSTED-BUDGET":
            if not frac <= F("0.50"):
                f.append("EB J4: ceiling_charged_fraction not <=0.50")
        else:
            if not frac > F("0.50"):
                f.append("EBSO J4-mirror: ceiling_charged_fraction not >0.50")
    return f
