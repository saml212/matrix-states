#!/usr/bin/env python3
"""RULE R-delta, EVALUATED -- NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.6.1 step 3.

Reads the SIX-RUNG 98M re-score (squarings {5,7,9,11,13,15}, produced by
depthext6_driver.py) and applies Rule R-delta (sec 5.2) MECHANICALLY. No
judgment is exercised between reading the re-score and writing delta_depth.

  kappa      = (acc - 1/K) / (1 - 1/K)
  H_c(s)     = 1 - median_seeds kappa_c(s)          over the 8 (K, recipe) cells
               at the FOUR PORTED K {16,24,32,40}
  delta*(s)  = 3rd-smallest H_c(s), rounded DOWN to the nearest 0.005
  admissible iff delta*(s) >= 0.05
  elect the SHALLOWEST admissible s in (11, 13, 15); delta_depth := delta*(s)
  if none admissible -> the per-K magnitude verdict is STRUCK and TEST-X is
  the sole improvement verdict (sec 4.6.1's contingency).

PARTIAL-LOSS LADDER (sec 4.6.1, verify-R2 MAJOR-4): 8 survivors -> as written;
6 or 7 -> the quantile is restated as the ceil(n/4)-th smallest and the missing
cells are NAMED; <=5, or ANY loss in the K=24 stratum -> the magnitude verdict
is struck. Implemented, not assumed: the survivor count is counted, never
hard-coded.

REPRODUCTION CROSS-CHECK. (5,7,9,11) is a strict prefix of the extended
profile and every pool/seed/n is unchanged, so the four archived accuracies
must reproduce EXACTLY. Any mismatch is reported and is a HALT condition --
it would mean the extension moved something it must not have.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

PORTED_K = (16, 24, 32, 40)
ALL_K = (12, 16, 20, 24, 28, 32, 36, 40)
CANDIDATE_S = (11, 13, 15)
HOUSE_FLOOR = 0.05
ROUND_TO = 0.005


def kappa(acc: float, k: int) -> float:
    return (acc - 1.0 / k) / (1.0 - 1.0 / k)


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def floor_to(x: float, q: float) -> float:
    return math.floor(x / q + 1e-9) * q


def load(dirpath: str) -> dict:
    """-> cells[(K, recipe)][seed][s] = acc"""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    meta: dict = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "*_depthext.json"))):
        d = json.load(open(p))
        if d.get("self_check") != "PASS":
            raise SystemExit(f"HALT: {p} self_check={d.get('self_check')} "
                             f"{d.get('self_check_defects')}")
        k = int(d["K"])
        recipe = "frozen" if d["freeze_entity_adapter"] else "trainable"
        seed = int(d["ckpt_seed"])
        for h, e in d["matched"]["P1b"]["per_hop"].items():
            cells[(k, recipe)][seed][int(e["n_squarings"])] = float(e["acc"])
        meta[(k, recipe, seed)] = dict(path=p, ladder=d["depth_ladder"],
                                       tag=d["tag"], ckpt=d["ckpt"])
    return cells, meta


def reproduction_check(new_dir: str, old_dir: str) -> dict:
    """The four archived rungs must reproduce EXACTLY."""
    old = {}
    for p in sorted(glob.glob(os.path.join(old_dir, "*_depthext.json"))):
        d = json.load(open(p))
        key = (int(d["K"]), bool(d["freeze_entity_adapter"]), int(d["ckpt_seed"]))
        old[key] = {int(e["n_squarings"]): float(e["acc"])
                    for e in d["matched"]["P1b"]["per_hop"].values()}
    n_cmp, mism = 0, []
    for p in sorted(glob.glob(os.path.join(new_dir, "*_depthext.json"))):
        d = json.load(open(p))
        key = (int(d["K"]), bool(d["freeze_entity_adapter"]), int(d["ckpt_seed"]))
        if key not in old:
            continue
        new = {int(e["n_squarings"]): float(e["acc"])
               for e in d["matched"]["P1b"]["per_hop"].values()}
        for s in (5, 7, 9, 11):
            n_cmp += 1
            if abs(new[s] - old[key][s]) > 0:
                mism.append(dict(cell=key, s=s, archived=old[key][s], rescore=new[s]))
    return {"n_compared": n_cmp, "n_mismatch": len(mism), "mismatches": mism[:20],
            "status": "EXACT REPRODUCTION" if not mism else "MISMATCH -- HALT"}


def strat_U(cells, s: int, ks) -> dict:
    """Within-K frozen-vs-trainable U over the 9 seed pairs, ties 1/2 (sec 5.3)."""
    out = {}
    for k in ks:
        fz = [kappa(cells[(k, "frozen")][sd][s], k) for sd in sorted(cells[(k, "frozen")])]
        tr = [kappa(cells[(k, "trainable")][sd][s], k) for sd in sorted(cells[(k, "trainable")])]
        u = 0.0
        for a in fz:
            for b in tr:
                u += 1.0 if a > b else (0.5 if a == b else 0.0)
        out[k] = u
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="dir of the six-rung re-score JSONs")
    ap.add_argument("--archived", default=None, help="dir of the four-rung wave of record")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells, meta = load(args.new)
    rec: dict = {"rule": "R-delta, NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 5.2",
                 "source": os.path.abspath(args.new),
                 "n_cells_loaded": len(cells), "n_records": len(meta)}

    if args.archived:
        rec["reproduction_check"] = reproduction_check(args.new, args.archived)

    # ---- the 8 (K, recipe) cells at the four ported K -----------------------
    survivors, missing = [], []
    for k in PORTED_K:
        for r in ("frozen", "trainable"):
            (survivors if (k, r) in cells and len(cells[(k, r)]) == 3
             else missing).append((k, r))
    rec["survivors"] = [f"K{k}_{r}" for k, r in survivors]
    rec["missing"] = [f"K{k}_{r}" for k, r in missing]
    n = len(survivors)
    k24_lost = any(k == 24 for k, _ in missing)

    # ---- per-cell kappa and headroom ---------------------------------------
    table = {}
    for k, r in survivors:
        per_seed = cells[(k, r)]
        row = {}
        for s in (5, 7, 9, 11, 13, 15):
            kaps = [kappa(per_seed[sd][s], k) for sd in sorted(per_seed)]
            row[s] = dict(per_seed_kappa=[round(x, 6) for x in kaps],
                          median_kappa=round(median(kaps), 6),
                          seed_range=round(max(kaps) - min(kaps), 6),
                          headroom=round(1.0 - median(kaps), 6))
        table[f"K{k}_{r}"] = row
    rec["per_cell"] = table

    # ---- Rule R-delta -------------------------------------------------------
    if n <= 5 or k24_lost:
        rec["rule_R_delta"] = {
            "verdict": "MAGNITUDE VERDICT STRUCK",
            "why": (f"survivors={n} (<=5) or a K=24 stratum loss ({k24_lost}); sec 4.6.1's "
                    f"partial-loss ladder strikes the per-K magnitude verdict and makes TEST-X "
                    f"the sole improvement verdict.")}
    else:
        quantile_idx = 3 if n == 8 else math.ceil(n / 4)     # sec 4.6.1's restatement
        per_s = {}
        for s in CANDIDATE_S:
            H = sorted((table[f"K{k}_{r}"][s]["headroom"], f"K{k}_{r}") for k, r in survivors)
            dstar = floor_to(H[quantile_idx - 1][0], ROUND_TO)
            reach = sum(1 for h, _ in H if h >= dstar)
            per_s[s] = dict(sorted_headroom=[[round(h, 6), c] for h, c in H],
                            quantile_index=quantile_idx,
                            raw=round(H[quantile_idx - 1][0], 6),
                            delta_star=round(dstar, 4),
                            admissible=bool(dstar >= HOUSE_FLOOR),
                            reachable_cells=reach, of=n,
                            reachable_frozen=sum(1 for h, c in H
                                                 if h >= dstar and c.endswith("_frozen")))
        elected = next((s for s in CANDIDATE_S if per_s[s]["admissible"]), None)
        rec["rule_R_delta"] = {
            "n_survivors": n, "quantile_index": quantile_idx, "house_floor": HOUSE_FLOOR,
            "per_s": per_s,
            "elected_s": elected,
            "delta_depth": (per_s[elected]["delta_star"] if elected is not None else None),
            "verdict": (f"s* = {elected}, delta_depth = {per_s[elected]['delta_star']}"
                        if elected is not None else
                        "NO ADMISSIBLE DEPTH -- magnitude verdict STRUCK, TEST-X sole "
                        "improvement verdict (sec 4.6.1 contingency)")}

        # teeth receipt: the rule must REJECT s = 11 on the data of record
        rec["rule_R_delta"]["teeth_receipt_s11_rejected"] = not per_s[11]["admissible"]

    # ---- sec 4.6.1 condition 2: the EXTENSION readings, never a retraction ---
    allk = [k for k in ALL_K if (k, "frozen") in cells and (k, "trainable") in cells]
    ext = {}
    for s in (5, 7, 9, 11, 13, 15):
        u8 = strat_U(cells, s, allk)
        u4 = {k: v for k, v in u8.items() if k in PORTED_K}
        ext[s] = dict(U_per_K_8strata=u8, T_8strata=round(sum(u8.values()), 1),
                      threshold_8strata="T >= 53/72",
                      U_per_K_4strata=u4, T_W_4strata=round(sum(u4.values()), 1),
                      partition_4strata="CONFIRMED T>31.5 / INDETERMINATE 29.5-31.5 / "
                                        "NEGLIGIBLE 6<T<29.5 / INVERTED T<=6")
    rec["ordering_extension"] = {
        "framing": ("sec 4.6.1 condition 2: the 13/15 readings EXTEND #8's 11-squaring verdict "
                    "of record and NEVER retract it. #8's T = 61.5/72 at 11 squarings is not "
                    "reopened by this re-score."),
        "per_s": ext}

    js = json.dumps(rec, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)
    print(js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
