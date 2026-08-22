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
SWEEP_GRID = (12, 16, 20, 24, 28, 32)        # the six strata of #4
FRONTIER_GRID = (36, 40)                      # design sec 14 frontier extension
REF_K = 24
TOL = 0.05
# #4 band (6 strata) and the design sec 14.2 band (8 strata), threshold locked
# at build time BEFORE the frontier data existed.
BANDS = {6: dict(threshold=42.0, n_pairs=54, name="ORDERING-AT-DEPTH"),
         8: dict(threshold=53.0, n_pairs=72, name="ORDERING-ROBUST")}


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
    global KS_GRID
    KS_GRID = tuple(sorted({r["K"] for r in recs}))
    print(f"strata present: K = {list(KS_GRID)} ({len(KS_GRID)} strata)")
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
    def run_test(grid, squarings=11, verbose=True):
        T, strata = 0.0, []
        rows = []
        for k in grid:
            fv = sorted(acc_by_sq(r)[squarings] for r in g[(k, True)])
            tv = sorted(acc_by_sq(r)[squarings] for r in g[(k, False)])
            ts = stratum_T(fv, tv)
            T += ts
            strata.append(fv + tv)
            rows.append((k, fv, tv, ts))
        dist = exact_null(strata)
        p_ge = sum(pr for t, pr in dist.items() if t >= T - 1e-9)
        p_le = sum(pr for t, pr in dist.items() if t <= T + 1e-9)
        mean_null = sum(t * pr for t, pr in dist.items())
        p_two = min(1.0, 2 * min(p_ge, p_le))
        return dict(T=T, rows=rows, p_ge=p_ge, p_two=p_two, mean_null=mean_null, dist=dist)

    print("\n" + "=" * 104)
    print(f"(b) STRATIFIED WITHIN-K ORDERING AT {SQ[-1]} SQUARINGS (frozen vs trainable, ties = 1/2)")
    print("=" * 104)
    res8 = run_test(KS_GRID, 11)
    print(f"   {'K':>3s} {'frozen kappa@11sq':>28s} {'trainable kappa@11sq':>28s} {'T_stratum':>10s}/9")
    for k, fv, tv, ts in res8["rows"]:
        mark = "  <- frontier" if k in FRONTIER_GRID else ""
        print(f"   {k:>3d} {str([round(x,4) for x in fv]):>28s} "
              f"{str([round(x,4) for x in tv]):>28s} {ts:>10.1f}{mark}")
    print("-" * 104)
    band = BANDS[len(KS_GRID)]
    T, npairs, thr = res8["T"], band["n_pairs"], band["threshold"]
    losses = npairs - T
    print(f"   TOTAL T = {T:.1f}/{npairs}   (lower tail = {losses:.1f}, allowed <= {npairs-thr:.0f})")
    print(f"   exact stratified permutation null (enumerated C(6,3)=20 per stratum, convolved "
          f"over {len(KS_GRID)} strata):")
    print(f"      mean T = {res8['mean_null']:.1f}   P(T >= {T:.1f}) = {res8['p_ge']:.3e}   "
          f"exact two-sided p = {res8['p_two']:.3e}")
    verdict = (f"{band['name']}-CONFIRMED" if T >= thr else f"{band['name']}-NEGLIGIBLE")
    print(f"\nBAND (b) VERDICT: {verdict} -- T = {T:.1f}/{npairs} (threshold >= {thr:.0f})")

    # depth profile on the SAME strata (context, not verdict)
    print(f"\n   DEPTH PROFILE -- T by squaring count, same {len(KS_GRID)} strata:")
    for s in SQ:
        r = run_test(KS_GRID, s)
        print(f"      {s:2d} squarings: T = {r['T']:5.1f}/{npairs}  (two-sided p = {r['p_two']:.3e})"
              + ("  <- pre-registered readout" if s == 11 else "  (context)"))

    # continuity with #4's six-stratum result
    if set(SWEEP_GRID).issubset(set(KS_GRID)) and len(KS_GRID) > len(SWEEP_GRID):
        r6 = run_test(SWEEP_GRID, 11)
        b6 = BANDS[6]
        print(f"\n   CONTINUITY -- the original {len(SWEEP_GRID)} strata alone at 11 squarings: "
              f"T = {r6['T']:.1f}/{b6['n_pairs']} (threshold >= {b6['threshold']:.0f}, "
              f"{'clears' if r6['T'] >= b6['threshold'] else 'fails'}); unchanged from #4.")

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
    print(f"P0 WALL: {f'HOLDS -- all {len(recs)} readings within their per-K band' if out == 0 else f'{out} reading(s) OUT OF BAND'}")


if __name__ == "__main__":
    main()
