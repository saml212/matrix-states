#!/usr/bin/env python3
"""Aggregate the depth-extension-across-K eval and apply the #2 bands.

Bands (EXPERIMENT_LOG 2026-08-22 #2 final block, commit 9bf2f40):
  (a) DRIFT-K-INDEPENDENT      = per-K drift (kappa@11sq - kappa@5sq) within
                                 +/-0.05 of the K=24 value at EVERY K.
  (b) ORDERING-AT-DEPTH-CONFIRMED  = stratified within-K exact permutation
                                 T >= 42/54 at 11 squarings (6 strata x 9
                                 frozen-vs-trainable pairs, ties count 1/2);
      ORDERING-AT-DEPTH-NEGLIGIBLE = below.

The permutation null is EXACT: within each stratum there are C(6,3)=20 ways to
label 3 of the 6 cells frozen, so the per-stratum T distribution is enumerated
exactly and the 6 strata are convolved. No sampling.

Within a K, all four rungs share one residue -> IDENTICAL GROUND TRUTH BY
CONSTRUCTION; spread across rungs is numerical squaring depth. Across K the
ground truth and chance (1/K) both differ, so only DRIFTS are compared across K.
"""
import glob
import itertools
import json
import os
import statistics
import sys
from collections import defaultdict

SQ = (5, 7, 9, 11)
KS_GRID = (12, 16, 20, 24, 28, 32)
REF_K = 24
TOL = 0.05
T_THRESHOLD = 42.0
N_PAIRS = 54


def acc_by_sq(rec):
    return {e["n_squarings"]: e["acc"] for e in rec["matched"]["P1b"]["per_hop"].values()}


def stratum_T(frozen, trainable):
    """Pairwise wins with ties at 1/2 (9 pairs when 3x3)."""
    t = 0.0
    for a in frozen:
        for b in trainable:
            t += 1.0 if a > b else (0.5 if a == b else 0.0)
    return t


def exact_null(vals_by_stratum):
    """Exact distribution of total T under within-stratum label permutation."""
    dist = {0.0: 1.0}
    for vals in vals_by_stratum:
        sub = defaultdict(float)
        idx = range(len(vals))
        combos = list(itertools.combinations(idx, len(vals) // 2))
        for c in combos:
            f = [vals[i] for i in c]
            t = [vals[i] for i in idx if i not in c]
            sub[stratum_T(f, t)] += 1.0 / len(combos)
        new = defaultdict(float)
        for a, pa in dist.items():
            for b, pb in sub.items():
                new[a + b] += pa * pb
        dist = dict(new)
    return dist


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*_depthext.json")))]
    print(f"loaded {len(recs)} cells")
    bad = [(r["tag"], r["self_check"], r["ckpt_step"]) for r in recs
           if r["self_check"] != "PASS" or r["ckpt_step"] != 20000]
    print("INVALID:", bad if bad else "none")
    print("all pools matched:", all(r["pools_matched"] for r in recs))
    print("all single-residue ladders:",
          all(len(set(r["ladder_diagnostics"]["residues"])) == 1 for r in recs))
    print("all squaring profiles == (5,7,9,11):",
          all(tuple(r["ladder_diagnostics"]["n_squarings"]) == SQ for r in recs))
    print("\nPER-K LADDERS (all rungs share r_fix -> SAME GROUND TRUTH BY CONSTRUCTION per K):")
    seen = {}
    for r in recs:
        seen.setdefault(r["K"], (r["depth_ladder"], r["ladder_diagnostics"]["r_fix"], r["chance"]))
    for k in KS_GRID:
        lad, rf, ch = seen[k]
        print(f"   K={k:2d}  r_fix={rf}  ladder={lad}  chance=1/K={ch:.4f}")

    g = defaultdict(list)   # (K, freeze) -> [rec]
    for r in recs:
        g[(r["K"], r["freeze_entity_adapter"])].append(r)

    # ---------- (a) drift table ------------------------------------------
    print("\n" + "=" * 104)
    print("(a) PER-K DEPTH-DRIFT -- P1b kappa, median over 3 seeds, matched pools")
    print("=" * 104)
    # compute ALL drifts first -- printing them inline made the K<24 rows show a
    # blank delta because the K=24 reference had not been computed yet
    drift, medians = {}, {}
    for fz in (True, False):
        for k in KS_GRID:
            rs = g[(k, fz)]
            medians[(k, fz)] = {s: statistics.median([acc_by_sq(r)[s] for r in rs]) for s in SQ}
            drift[(k, fz)] = statistics.median([acc_by_sq(r)[11] - acc_by_sq(r)[5] for r in rs])
    for fz, label in ((True, "FROZEN-contrastive (primary)"), (False, "TRAINABLE-contrastive (compB)")):
        print(f"\n{label}")
        print(f"   {'K':>3s} " + " ".join(f"{'@'+str(s)+'sq':>9s}" for s in SQ)
              + f" {'drift(11-5)':>12s} {'vs K=24':>9s}")
        ref = drift[(REF_K, fz)]
        for k in KS_GRID:
            med = medians[(k, fz)]
            dr = drift[(k, fz)]
            delta = "  (ref)" if k == REF_K else f"{dr-ref:+9.4f}"
            print(f"   {k:>3d} " + " ".join(f"{med[s]:9.4f}" for s in SQ)
                  + f" {dr:+12.4f} {delta:>9s}")
        worst = max((k for k in KS_GRID if k != REF_K), key=lambda k: abs(drift[(k, fz)] - ref))
        print(f"   closest call: K={worst} at {drift[(worst,fz)]-ref:+.4f} vs tolerance +/-{TOL}")
    # band (a)
    viol = []
    for fz in (True, False):
        ref = drift[(REF_K, fz)]
        for k in KS_GRID:
            if k == REF_K:
                continue
            if abs(drift[(k, fz)] - ref) > TOL:
                viol.append((("FROZEN" if fz else "TRAINABLE"), k,
                             round(drift[(k, fz)], 4), round(ref, 4),
                             round(drift[(k, fz)] - ref, 4)))
    print("\n" + "-" * 104)
    if viol:
        print(f"BAND (a) VERDICT: NOT DRIFT-K-INDEPENDENT -- {len(viol)} K/recipe cell(s) exceed "
              f"+/-{TOL} vs the K=24 reference:")
        for v in viol:
            print(f"    {v[0]:9s} K={v[1]:2d}: drift {v[2]:+.4f} vs K=24 {v[3]:+.4f} "
                  f"(delta {v[4]:+.4f})")
    else:
        print(f"BAND (a) VERDICT: DRIFT-K-INDEPENDENT -- every K within +/-{TOL} of the "
              f"K=24 drift, both recipes")

    # ---------- (b) stratified exact permutation at 11 squarings ----------
    print("\n" + "=" * 104)
    print("(b) STRATIFIED WITHIN-K ORDERING AT 11 SQUARINGS (frozen vs trainable, ties = 1/2)")
    print("=" * 104)
    print(f"   {'K':>3s} {'frozen kappa@11sq':>28s} {'trainable kappa@11sq':>28s} {'T_stratum':>10s}/9")
    T = 0.0
    strata_vals = []
    for k in KS_GRID:
        fv = sorted(acc_by_sq(r)[11] for r in g[(k, True)])
        tv = sorted(acc_by_sq(r)[11] for r in g[(k, False)])
        ts = stratum_T(fv, tv)
        T += ts
        strata_vals.append(fv + tv)
        print(f"   {k:>3d} {str([round(x,4) for x in fv]):>28s} "
              f"{str([round(x,4) for x in tv]):>28s} {ts:>10.1f}")
    print("-" * 104)
    losses = N_PAIRS - T
    print(f"   TOTAL T = {T:.1f}/{N_PAIRS}   (lower tail / trainable-wins-equivalent = {losses:.1f})")

    dist = exact_null(strata_vals)
    p_ge = sum(pr for t, pr in dist.items() if t >= T - 1e-9)
    mean_null = sum(t * pr for t, pr in dist.items())
    print(f"   exact stratified permutation null: mean T = {mean_null:.1f}, "
          f"P(T >= {T:.1f}) = {p_ge:.3e}")
    verdict = ("ORDERING-AT-DEPTH-CONFIRMED" if T >= T_THRESHOLD
               else "ORDERING-AT-DEPTH-NEGLIGIBLE")
    print(f"\nBAND (b) VERDICT: {verdict} -- T = {T:.1f}/{N_PAIRS} "
          f"(threshold >= {T_THRESHOLD:.0f}; lower tail {losses:.1f}, allowed <= {N_PAIRS-T_THRESHOLD:.0f})")
    print(f"   for reference, #2 Curve 3 measured T = 32/54 at 5 squarings (ORDERING-NEGLIGIBLE)")

    # T at every squaring count, to show where it opens
    print("\n   T by squaring count (same test, same strata):")
    for s in SQ:
        tt = 0.0
        for k in KS_GRID:
            fv = [acc_by_sq(r)[s] for r in g[(k, True)]]
            tv = [acc_by_sq(r)[s] for r in g[(k, False)]]
            tt += stratum_T(fv, tv)
        print(f"      {s:2d} squarings: T = {tt:5.1f}/{N_PAIRS}"
              + ("  <- pre-registered readout" if s == 11 else ""))

    # ---------- P0 wall spot-check ---------------------------------------
    print("\n" + "=" * 104)
    print("P0 (LEARNED-WRITE) WALL SPOT-CHECK at the deepest rung (11 squarings), per-K band 1/K +/-3sd")
    print("=" * 104)
    out = 0
    for k in KS_GRID:
        rs = g[(k, True)] + g[(k, False)]
        vals, band, ch = [], None, None
        for r in rs:
            e = list(r["matched"]["P0"]["per_hop"].values())[0]
            vals.append(e["acc"])
            band, ch = r["wall_band"], r["chance"]
            if not e["within_chance_3sd"]:
                out += 1
        print(f"   K={k:2d} chance={ch:.4f} band=[{band[0]:.4f},{band[1]:.4f}] "
              f"n={len(vals)} min={min(vals):.4f} max={max(vals):.4f}")
    print("-" * 104)
    print(f"P0 WALL: {'HOLDS -- all 36 readings within their per-K band' if out == 0 else f'{out} reading(s) OUT OF BAND'}")


if __name__ == "__main__":
    main()
