#!/usr/bin/env python3
"""Build the deliverable 2x2 table from the *_poolmatched.json outputs.

Arm labels come from each cell's OWN recorded training config
(aux_loss_type x freeze_entity_adapter), never from the cell name.

Reports, per 2x2 quadrant:
  - median P1b@h61 under MATCHED pools, with every per-seed value listed
  - the same cells' P1b@h61 under pool seed 0 (the instrument of record)
  - the s0 anchors called out separately (pool seed 0 == trivially matched)
And the wall claim: does P0 stay at chance under matched pools (min/max)?
"""
import glob
import json
import os
import statistics
import sys

M = "retrieval24_acc"
CHANCE = 1.0 / 24.0
QUAD = {(True, "cosine"): "FROZEN-cosine (compA)",
        (True, "contrastive+cosine"): "FROZEN-contrastive (primary)",
        (False, "cosine"): "TRAINABLE-cosine (compD)",
        (False, "contrastive+cosine"): "TRAINABLE-contrastive (compB)"}


def get(rec, pool_seed, regime, h):
    b = rec["by_pool"].get(f"pool_seed={pool_seed}")
    if not b:
        return None
    return b[regime]["result"][f"h={h}"][M]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*_poolmatched.json"))):
        recs.append(json.load(open(p)))
    print(f"loaded {len(recs)} cells\n")

    bad = [r for r in recs if r["self_check"] != "PASS" or r["ckpt_step"] != 20000]
    if bad:
        print("!!! INVALID CELLS:", [(r["tag"], r["self_check"], r["ckpt_step"]) for r in bad], "\n")

    groups, ungrouped = {}, []
    for r in recs:
        cc = r.get("cell_config") or {}
        aux = cc.get("cfg_aux_loss_type")
        fz = r["freeze_entity_adapter"]
        key = QUAD.get((fz, aux))
        if key is None:
            ungrouped.append((r["tag"], fz, aux))
            continue
        groups.setdefault(key, []).append(r)

    print("=" * 108)
    print("2x2 TABLE -- P1b (EXACT-WRITE) retrieval24_acc @ h=61, n=256, eval seed 90210, all ckpt_step==20000")
    print("=" * 108)
    hdr = f"{'quadrant':34s} {'n':>3s} {'MATCHED median':>15s} {'seed-0 median':>14s} {'delta':>8s}"
    print(hdr)
    print("-" * 108)
    summary = {}
    for key in ["FROZEN-contrastive (primary)", "FROZEN-cosine (compA)",
                "TRAINABLE-contrastive (compB)", "TRAINABLE-cosine (compD)"]:
        rs = sorted(groups.get(key, []), key=lambda r: r["ckpt_seed"])
        if not rs:
            continue
        matched = [get(r, r["ckpt_seed"], "P1b", 61) for r in rs]
        ref = [get(r, 0, "P1b", 61) for r in rs]
        mm, mr = statistics.median(matched), statistics.median(ref)
        summary[key] = (rs, matched, ref, mm, mr)
        print(f"{key:34s} {len(rs):3d} {mm:15.4f} {mr:14.4f} {mm-mr:+8.4f}")
    print("-" * 108)
    if ungrouped:
        print(f"NOT IN THE 2x2 (config predates the aux_loss_type/freeze flags): {ungrouped}")
    print()

    for key, (rs, matched, ref, mm, mr) in summary.items():
        print(f"--- {key} --- per-seed P1b@h61 (ckpt_seed: matched / seed-0-pool)")
        line = "   " + "  ".join(
            f"s{r['ckpt_seed']}:{m:.3f}/{x:.3f}" + ("*" if r["trivially_matched"] else "")
            for r, m, x in zip(rs, matched, ref))
        print(line)
        anchors = [(r, m) for r, m in zip(rs, matched) if r["trivially_matched"]]
        if anchors:
            print(f"   * s0 ANCHOR (pool seed 0 == trivially matched): "
                  + ", ".join(f"s{r['ckpt_seed']}={m:.4f}" for r, m in anchors))
        print()

    # --- h=1 and h=3 for completeness ------------------------------------
    print("=" * 108)
    print("P1b median by hop (MATCHED pools)")
    print("=" * 108)
    print(f"{'quadrant':34s} {'h=1':>10s} {'h=3':>10s} {'h=61':>10s}   | seed-0 pool: h=1 / h=3 / h=61")
    for key, (rs, *_rest) in summary.items():
        mm = {h: statistics.median([get(r, r["ckpt_seed"], "P1b", h) for r in rs]) for h in (1, 3, 61)}
        rr = {h: statistics.median([get(r, 0, "P1b", h) for r in rs]) for h in (1, 3, 61)}
        print(f"{key:34s} {mm[1]:10.4f} {mm[3]:10.4f} {mm[61]:10.4f}   | "
              f"{rr[1]:.4f} / {rr[3]:.4f} / {rr[61]:.4f}")
    print()

    # --- the wall claim ---------------------------------------------------
    print("=" * 108)
    print(f"WALL CLAIM: does P0 (LEARNED-WRITE) stay at chance ({CHANCE:.4f}) under MATCHED pools?")
    print("=" * 108)
    allmin, allmax = 1.0, 0.0
    for key, (rs, *_rest) in summary.items():
        vals = [get(r, r["ckpt_seed"], "P0", h) for r in rs for h in (1, 3, 61)]
        lo, hi = min(vals), max(vals)
        allmin, allmax = min(allmin, lo), max(allmax, hi)
        worst = max(rs, key=lambda r: max(get(r, r["ckpt_seed"], "P0", h) for h in (1, 3, 61)))
        print(f"{key:34s} n_readings={len(vals):3d}  min={lo:.4f}  max={hi:.4f}  "
              f"(max cell: {worst['tag']})")
    print("-" * 108)
    print(f"{'ALL ARMS POOLED':34s} min={allmin:.4f}  max={allmax:.4f}  chance={CHANCE:.4f}")
    # binomial sd at n=256, p=1/24
    sd = (CHANCE * (1 - CHANCE) / 256) ** 0.5
    print(f"binomial sd at n=256, p=1/24: {sd:.4f}  -> chance +/- 3sd = "
          f"[{CHANCE-3*sd:.4f}, {CHANCE+3*sd:.4f}]")
    verdict = "YES -- every P0 reading is within chance +/- 3sd" if allmax <= CHANCE + 3 * sd and allmin >= CHANCE - 3 * sd else "NO -- at least one P0 reading is outside chance +/- 3sd"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
