#!/usr/bin/env python3
"""§1.36b razor seed-extension harvest (S4/A5/S5/A6 x seeds 1-3), mirroring
experiment-runs/2026-07-09_m3fix_s3ext/analyze_m3fix_s3ext_harvest.py.
Usage: python3 analyze_m3fix_ext4_harvest.py <seed0_dir> <ext_results_dir>
Pre-registered (CAPABILITY_SEPARATION_DESIGN.md §1.36b, commit 9063020, before launch):
  necessity: k=d_min-1 crosscheck xrec90 == 0.000 in every seed (else HALT: instrument leak);
  sufficiency: seed-mean over seeds 0-3 of k=d_min crosscheck xrec90 >= FIXED literal bar
  (0.9 x seed-0 anchor, from §1.36): S4 0.585, A5 0.630, S5 0.450, A6 0.585.
  Disclosed, never decisional: per-seed own-bar clears; self-referential bar (0.9 x 4-seed anchor mean).
"""
import json, glob, os, sys, statistics as st
GROUPS = {"S4": (3, 20000, 0.585), "A5": (3, 20000, 0.630), "S5": (4, 8000, 0.450), "A6": (5, 40000, 0.585)}
AMBIENT_TAX = 2
def expected_manifest(g, dmin, steps, seed):
    exp = {f"zero_pad__{g}__unconstrained__seed{seed}": ("zero_pad", None, "zero", steps, seed)}
    for lab, k in (("k_dmin_minus_1", dmin-1), ("k_dmin", dmin), ("k_dmin_plus_1", dmin+1)):
        exp[f"zero_pad__{g}__{lab}__seed{seed}"] = ("zero_pad", k, "zero", steps, seed)
    for lab, k in (("k_dmin_minus_1", dmin-1+AMBIENT_TAX), ("k_dmin", dmin+AMBIENT_TAX)):
        exp[f"tax_adjusted__{g}__{lab}__seed{seed}"] = ("tax_adjusted", k, "eye", steps, seed)
    return exp
def main():
    seed0_dir, ext_dir = sys.argv[1], sys.argv[2]
    cells = {}
    for f in glob.glob(os.path.join(seed0_dir, "*seed0.json")) + glob.glob(os.path.join(ext_dir, "*.json")):
        d = json.load(open(f)); cells[d["cell_id"]] = d
    problems = []; halted = False; summary = {}
    for g, (dmin, steps, bar) in GROUPS.items():
        expected = {}
        for seed in (0, 1, 2, 3): expected.update(expected_manifest(g, dmin, steps, seed))
        missing = [c for c in expected if c not in cells]
        if missing: problems.append(f"{g} MISSING {missing}")
        for cid, (variant, k, pad, st_pin, seed) in expected.items():
            d = cells.get(cid)
            if d is None: continue
            if d.get("steps_completed") != st_pin: problems.append(f"{cid}: steps {d.get('steps_completed')} != {st_pin}")
            if d.get("n_skipped_steps", 0) != 0: problems.append(f"{cid}: skipped steps {d.get('n_skipped_steps')}")
            if d.get("force_rank_k") != k: problems.append(f"{cid}: k {d.get('force_rank_k')} != {k}")
            if d.get("seed") != seed: problems.append(f"{cid}: seed {d.get('seed')} != {seed}")
        def get(seed, arm, key):
            return cells[f"zero_pad__{g}__{arm}__seed{seed}"][key]
        print(f"\n=== {g} (d_min={dmin}, steps {steps}, fixed bar {bar}) ===")
        print("seed  anchor  k=dmin-1  k=dmin  k=dmin+1  own-bar  clears   |  whole-matrix rank: anchor  k=dmin")
        kd, anch, nec_ok, own_clears = [], [], True, 0
        for seed in (0, 1, 2, 3):
            a = get(seed, "unconstrained", "crosscheck_recovered_frac_90")
            m1 = get(seed, "k_dmin_minus_1", "crosscheck_recovered_frac_90")
            m0 = get(seed, "k_dmin", "crosscheck_recovered_frac_90")
            p1 = get(seed, "k_dmin_plus_1", "crosscheck_recovered_frac_90")
            wa = get(seed, "unconstrained", "whole_matrix_effective_rank")
            wk = get(seed, "k_dmin", "whole_matrix_effective_rank")
            own = 0.9 * a; clr = m0 >= own; own_clears += clr
            if m1 != 0.0: nec_ok = False
            kd.append(m0); anch.append(a)
            print(f"{seed}     {a:.3f}   {m1:.3f}     {m0:.3f}   {p1:.3f}     {own:.3f}    {'yes' if clr else 'no '} ({m0-own:+.3f})  |  {wa:.3f}   {wk:.3f}")
        sm = st.mean(kd); sm_ext = st.mean(kd[1:]); self_bar = 0.9 * st.mean(anch)
        if not nec_ok: halted = True
        verdict = "CONFIRMED at n=4" if sm >= bar else "FAILS at n=4 (report as failing)"
        print(f"necessity (k=dmin-1 == 0.000 in all 4 seeds): {'OK' if nec_ok else 'VIOLATED -> HALT'}")
        print(f"seed-mean k=dmin (seeds 0-3) = {sm:.4f}  vs fixed bar {bar}  -> {verdict}")
        print(f"disclosed: extension-only mean {sm_ext:.4f}; own-bar clears {own_clears}/4; self-referential bar 0.9x4-seed-anchor-mean = {self_bar:.4f} ({'clears' if sm >= self_bar else 'misses'} by {sm-self_bar:+.4f})")
        print(f"disclosed: whole-matrix rank of anchors mean {st.mean(get(s,'unconstrained','whole_matrix_effective_rank') for s in range(4)):.3f} +- {st.pstdev([get(s,'unconstrained','whole_matrix_effective_rank') for s in range(4)]):.3f} vs d_min {dmin}")
        summary[g] = dict(seed_mean=sm, bar=bar, verdict=verdict, necessity_ok=nec_ok, own_clears=own_clears, self_bar=self_bar, k_dmin=kd, anchors=anch)
    print("\n=== A3 config-match problems ===", problems if problems else "none (48 cells: 4 groups x 4 seeds x 3... zero_pad 4/seed + tax 2/seed = 24 cells/group incl. seed0)")
    print("=== OVERALL ===", "HALT (necessity violated)" if halted else " ; ".join(f"{g}: {summary[g]['verdict']}" for g in GROUPS))
    json.dump(summary, open(os.path.join(ext_dir, "..", "EXT4_VERDICT.json"), "w"), indent=1)
main()
