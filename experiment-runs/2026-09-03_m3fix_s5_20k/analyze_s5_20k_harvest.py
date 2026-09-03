#!/usr/bin/env python3
"""§1.36c S5 20k re-test harvest. Pre-registered rule (CAPABILITY_SEPARATION_DESIGN.md §1.36c, commit 6fb7638,
before launch): necessity k=d_min-1 == 0.000 in all four 20k seeds; sufficiency = seed-mean over the four 20k
seeds of k=d_min crosscheck rec@0.9 >= 0.9 x seed-mean of the four 20k unconstrained anchors. One shot.
Usage: python3 analyze_s5_20k_harvest.py <results_dir>"""
import json, glob, os, sys, statistics as st
G, DMIN, STEPS, TAX = "S5", 4, 20000, 2
cells = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    d = json.load(open(f)); cells[d["cell_id"]] = d
problems = []
for seed in range(4):
    exp = {f"zero_pad__{G}__unconstrained__seed{seed}": None, f"zero_pad__{G}__k_dmin_minus_1__seed{seed}": DMIN-1,
           f"zero_pad__{G}__k_dmin__seed{seed}": DMIN, f"zero_pad__{G}__k_dmin_plus_1__seed{seed}": DMIN+1,
           f"tax_adjusted__{G}__k_dmin_minus_1__seed{seed}": DMIN-1+TAX, f"tax_adjusted__{G}__k_dmin__seed{seed}": DMIN+TAX}
    for cid, k in exp.items():
        d = cells.get(cid)
        if d is None: problems.append(f"MISSING {cid}"); continue
        if d.get("steps_completed") != STEPS: problems.append(f"{cid}: steps {d.get('steps_completed')} != {STEPS}")
        if d.get("n_skipped_steps", 0) != 0: problems.append(f"{cid}: skipped {d.get('n_skipped_steps')}")
        if d.get("force_rank_k") != k: problems.append(f"{cid}: k {d.get('force_rank_k')} != {k}")
        if d.get("seed") != seed: problems.append(f"{cid}: seed {d.get('seed')} != {seed}")
def get(seed, arm, key): return cells[f"zero_pad__{G}__{arm}__seed{seed}"][key]
print(f"=== {G} at {STEPS} steps (d_min={DMIN}); 8k reference: seed-mean k=d_min 0.4125 vs fixed bar 0.450 (FAIL) ===")
print("seed  anchor  k=dmin-1  k=dmin  k=dmin+1  own-bar clears  | gate1a min-val: anchor k=dmin | whole rank: anchor k=dmin")
kd, an, nec = [], [], True; own = 0
for s in range(4):
    a, m1, m0, p1 = (get(s, x, "crosscheck_recovered_frac_90") for x in ("unconstrained", "k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"))
    g_a = get(s, "unconstrained", "gate1a").get("min_val"); g_k = get(s, "k_dmin", "gate1a").get("min_val")
    wa, wk = get(s, "unconstrained", "whole_matrix_effective_rank"), get(s, "k_dmin", "whole_matrix_effective_rank")
    if m1 != 0.0: nec = False
    clr = m0 >= 0.9*a; own += clr; kd.append(m0); an.append(a)
    print(f"{s}     {a:.3f}   {m1:.3f}     {m0:.3f}   {p1:.3f}     {0.9*a:.3f}  {'yes' if clr else 'no '} ({m0-0.9*a:+.3f})  |  {g_a:.3f}  {g_k:.3f}  |  {wa:.3f}  {wk:.3f}")
sm, sa = st.mean(kd), st.mean(an); bar = 0.9*sa
print(f"necessity: {'OK (0.000 x4)' if nec else 'VIOLATED -> HALT'}")
print(f"seed-mean k=dmin = {sm:.4f}; seed-mean anchor = {sa:.4f}; decisional bar 0.9 x anchor-mean = {bar:.4f}")
print(f"VERDICT: {'S5 CONFIRMS at the 20k pin (n=4)' if sm >= bar else 'S5 FAILS at the 20k pin too'}  (margin {sm-bar:+.4f}); own-bar clears {own}/4; vs the 8k fixed literal 0.450: {'clears' if sm>=0.450 else 'misses'} ({sm-0.450:+.4f})")
print("A3 config-match:", problems if problems else "none (24 cells)")
json.dump({"seed_mean_k_dmin": sm, "seed_mean_anchor": sa, "bar": bar, "confirms": sm >= bar, "necessity_ok": nec, "own_clears": own, "k_dmin": kd, "anchors": an}, open(os.path.join(sys.argv[1], "..", "S5_20K_VERDICT.json"), "w"), indent=1)
