#!/usr/bin/env python3
"""Aggregate the deep-ladder matched-pool eval and apply the #7 bands.

Bands (EXPERIMENT_LOG 2026-08-21 #7, commit ad52dcf, applied mechanically):
  RESIDUAL-CONFIRMED  = frozen-vs-trainable median gap > 0.05 AND
                        Mann-Whitney p < 0.01 at ANY h >= 253
  RESIDUAL-NEGLIGIBLE = gap <= 0.05 at EVERY h
Any outcome satisfying neither is reported as BAND-INDETERMINATE with the raw
numbers, never forced into a band.

All four hops are == 13 (mod 24): SAME GROUND TRUTH by construction. Spread
across hops is numerical depth cost, not compositional information.
"""
import glob
import json
import os
import statistics
import sys

try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

M = "retrieval24_acc"
CHANCE = 1.0 / 24.0
HOPS = (61, 253, 1021, 4093)
SQUARINGS = {61: 5, 253: 7, 1021: 9, 4093: 11}
QUAD = {(True, "cosine"): "FROZEN-cosine (compA)",
        (True, "contrastive+cosine"): "FROZEN-contrastive (primary)",
        (False, "cosine"): "TRAINABLE-cosine (compD)",
        (False, "contrastive+cosine"): "TRAINABLE-contrastive (compB)"}
ORDER = ["FROZEN-contrastive (primary)", "FROZEN-cosine (compA)",
         "TRAINABLE-contrastive (compB)", "TRAINABLE-cosine (compD)"]


def v(r, h, regime="P1b"):
    return r[regime]["result"][f"h={h}"][M]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*_deepladder.json")))]
    print(f"loaded {len(recs)} cells")
    bad = [(r["tag"], r["self_check"], r["ckpt_step"]) for r in recs
           if r["self_check"] != "PASS" or r["ckpt_step"] != 20000]
    print("INVALID CELLS:", bad if bad else "none")
    nm = [r["tag"] for r in recs if not r.get("pools_matched")]
    print("UNMATCHED-POOL CELLS:", nm if nm else "none (all matched)")
    print(f"\nALL HOPS == 13 (mod 24) -- SAME GROUND TRUTH BY CONSTRUCTION. "
          f"Squarings: {SQUARINGS}\n")

    groups, ungrouped = {}, []
    for r in recs:
        cc = r.get("cell_config") or {}
        key = QUAD.get((r["freeze_entity_adapter"], cc.get("cfg_aux_loss_type")))
        if key is None:
            ungrouped.append(r)
            continue
        groups.setdefault(key, []).append(r)

    # --- quadrant table + drift curves -----------------------------------
    print("=" * 112)
    print("QUADRANT TABLE / DEPTH-DRIFT CURVES -- P1b (EXACT-WRITE), MATCHED pools, n=256, seed 90210")
    print("median [min-max] per quadrant per hop")
    print("=" * 112)
    print(f"{'quadrant':34s} {'n':>3s} " + " ".join(f"{'h='+str(h):>22s}" for h in HOPS))
    print("-" * 112)
    for key in ORDER:
        rs = groups.get(key, [])
        if not rs:
            continue
        cells = []
        for h in HOPS:
            vals = [v(r, h) for r in rs]
            cells.append(f"{statistics.median(vals):.4f} [{min(vals):.3f}-{max(vals):.3f}]")
        print(f"{key:34s} {len(rs):3d} " + " ".join(f"{c:>22s}" for c in cells))
    print("-" * 112)
    for key in ORDER:
        rs = groups.get(key, [])
        if not rs:
            continue
        means = {h: statistics.mean([v(r, h) for r in rs]) for h in HOPS}
        print(f"{key:34s} mean drift h61->h4093: {means[61]:.4f} -> {means[4093]:.4f} "
              f"({means[4093]-means[61]:+.4f})")
    if ungrouped:
        print(f"\nNOT IN THE 2x2 (flags predate the cell): "
              f"{[(r['tag'], r['ckpt_seed']) for r in ungrouped]}")
        for r in ungrouped:
            print("   " + r["tag"] + ": " + " ".join(f"h{h}={v(r,h):.4f}" for h in HOPS)
                  + f"  P0h61={v(r,61,'P0'):.4f}")
    print()

    # --- the #7 question: frozen vs trainable, per hop --------------------
    F = [r for k in ORDER[:2] for r in groups.get(k, [])]
    T = [r for k in ORDER[2:] for r in groups.get(k, [])]
    print("=" * 112)
    print(f"RESIDUAL ORDERING -- FROZEN (n={len(F)}) vs TRAINABLE (n={len(T)}), P1b, matched pools")
    print("=" * 112)
    print(f"{'hop':>6s} {'sq':>3s} {'frozen med':>11s} {'trainable med':>14s} {'gap':>9s} "
          f"{'MW p':>11s} {'gap>0.05':>9s} {'p<0.01':>7s} {'complete sep':>13s}")
    print("-" * 112)
    per_hop = {}
    for h in HOPS:
        fv = [v(r, h) for r in F]
        tv = [v(r, h) for r in T]
        gap = statistics.median(fv) - statistics.median(tv)
        if HAVE_SCIPY:
            _u, p = mannwhitneyu(fv, tv, alternative="two-sided")
        else:
            p = float("nan")
        comp = all(a > b for a in fv for b in tv)
        per_hop[h] = (gap, p)
        print(f"{h:6d} {SQUARINGS[h]:3d} {statistics.median(fv):11.4f} {statistics.median(tv):14.4f} "
              f"{gap:+9.4f} {p:11.3e} {str(gap>0.05):>9s} {str(p<0.01):>7s} {str(comp):>13s}")
    print("-" * 112)

    deep = [h for h in HOPS if h >= 253]
    confirmed = [h for h in deep if per_hop[h][0] > 0.05 and per_hop[h][1] < 0.01]
    negligible = all(per_hop[h][0] <= 0.05 for h in HOPS)
    if confirmed:
        verdict = (f"RESIDUAL-CONFIRMED -- gap > 0.05 AND MW p < 0.01 at h in {confirmed} "
                   f"(both conditions met at h >= 253)")
    elif negligible:
        verdict = "RESIDUAL-NEGLIGIBLE -- median gap <= 0.05 at EVERY hop"
    else:
        viol = {h: (round(per_hop[h][0], 4), f"{per_hop[h][1]:.2e}") for h in HOPS
                if per_hop[h][0] > 0.05}
        verdict = (f"BAND-INDETERMINATE -- gap exceeds 0.05 somewhere ({viol}) but no h >= 253 "
                   f"meets BOTH pinned conditions; neither band's definition is satisfied")
    print(f"BAND VERDICT: {verdict}")
    print()

    # --- P0 spot-check ----------------------------------------------------
    print("=" * 112)
    print(f"P0 (LEARNED-WRITE) SPOT-CHECK at h=61, matched pools -- chance = {CHANCE:.4f}")
    print("=" * 112)
    allv = []
    for key in ORDER:
        rs = groups.get(key, [])
        if not rs:
            continue
        vals = [v(r, 61, "P0") for r in rs]
        allv += vals
        print(f"{key:34s} n={len(vals):3d} min={min(vals):.4f} max={max(vals):.4f} "
              f"median={statistics.median(vals):.4f}")
    for r in ungrouped:
        allv.append(v(r, 61, "P0"))
    sd = (CHANCE * (1 - CHANCE) / 256) ** 0.5
    lo, hi = CHANCE - 3 * sd, CHANCE + 3 * sd
    print("-" * 112)
    print(f"{'ALL POOLED':34s} n={len(allv):3d} min={min(allv):.4f} max={max(allv):.4f}   "
          f"chance +/-3sd = [{lo:.4f}, {hi:.4f}]")
    print("P0 AT CHANCE:", "YES -- every reading inside chance +/- 3sd"
          if min(allv) >= lo and max(allv) <= hi else
          f"NO -- {sum(1 for x in allv if x < lo or x > hi)} reading(s) outside")


if __name__ == "__main__":
    main()
