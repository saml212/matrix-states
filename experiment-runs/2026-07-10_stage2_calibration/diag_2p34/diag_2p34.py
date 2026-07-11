"""S2.34 diagnosis: M-D0 (per-depth TRAIN-support convergence profile) and
M-D2 (rank-vs-depth curve) executed against the committed 64-cell Stage-2
grid (62 sweep cells + 2 route_2p33 S5-extension cells), using
stage2_harvest.py's own m_d1_curve/m_d2_curve/own_d_train_ceiling
UNMODIFIED -- no new metrics invented, per the S2.34 charter.

Answers the three routed questions:
  (a) A6 n_h=2 vs n_h=4 -- budget artifact or architecture ceiling?
  (b) S5-seed-1 catastrophic non-generalizer vs its 4 healthy siblings.
  (c) A5 3/5-seeds-non-converged -- same question as (a).
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/Users/samuellarson/Experiments/learned-representations/matrix-thinking/capability_separation")
import stage2_harvest as H

BASE = "/Users/samuellarson/Experiments/learned-representations/experiment-runs/2026-07-10_stage2_calibration"

def load_all():
    cells = {}
    for sub in ("sweep_results", "route_2p33"):
        for path in sorted(glob.glob(os.path.join(BASE, sub, "*.json"))):
            fn = os.path.basename(path)
            if "harvest_report" in fn or "verdict" in fn or "s5ext_output" in fn:
                continue
            with open(path) as f:
                d = json.load(f)
            cid = d.get("cell_id") or os.path.splitext(fn)[0]
            cells[cid] = d
    return cells

def group_cells(cells, group, arm, n_h):
    return [c for cid, c in cells.items()
            if c["group"] == group and c["arm"] == arm and c["n_h"] == n_h]

def m_d0_table(cells_list, label):
    print(f"\n--- M-D0 per-depth TRAIN-support profile (PRIMARY lens, existing "
          f"m_d0_profile field, disclosure per S2.31a): {label} ({len(cells_list)} cells) ---")
    header = "seed | final_loss | " + " | ".join(f"D={d}(rf90/cos)" for d in range(1, H.D_TRAIN_MAX + 1))
    print(header)
    for c in sorted(cells_list, key=lambda c: c["seed"]):
        row_vals = []
        for d in range(1, H.D_TRAIN_MAX + 1):
            row = next(r for r in c["m_d0_profile"] if r["D"] == d)
            if row.get("excluded"):
                row_vals.append("EXCL")
            else:
                row_vals.append(f"{row['recovered_frac_90']:.2f}/{row['mean_cos']:+.2f}")
        print(f"seed{c['seed']} | {c['final_loss']:.2e} | " + " | ".join(row_vals))
    return cells_list


def m_d1_crosscheck_table(cells_list, label):
    """Crosscheck-lens M-D1 (S2.6's own already-computed
    `crosscheck_recovered_frac_90` field, D_test_results, D>=9) -- decisional
    per S2.31a for recovery numbers. Uses stage2_harvest.py's own private
    `_curve` helper unmodified, just pointed at the existing crosscheck
    field name (zero new metrics, zero new box compute -- the field is
    already committed in every D_test_results row)."""
    curve = H._curve(cells_list, "crosscheck_recovered_frac_90")
    print(f"\n--- M-D1 CROSSCHECK-lens accuracy-vs-depth (decisional per S2.31a): {label} ---")
    print("D | mean_xcheck_rf90 | std | n")
    for d, stats in sorted(curve.items()):
        print(f"{d} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['n']}")
    return curve

def m_d2_table(cells_list, label):
    curve = H.m_d2_curve(cells_list)
    print(f"\n--- M-D2 rank-vs-depth (restricted_effective_rank): {label} ---")
    print("D | mean_rank | std | n")
    for d, stats in sorted(curve.items()):
        print(f"{d} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['n']}")
    return curve

def m_d1_table(cells_list, label):
    curve = H.m_d1_curve(cells_list)
    print(f"\n--- M-D1 (context) accuracy-vs-depth: {label} ---")
    print("D | mean_recovered_frac_90 | std | n")
    for d, stats in sorted(curve.items()):
        print(f"{d} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['n']}")
    return curve

cells = load_all()
print(f"Loaded {len(cells)} cells total.")

print("\n" + "=" * 80)
print("QUESTION (a): A6 n_h staircase -- budget artifact or architecture ceiling?")
print("=" * 80)
a6_nh1 = group_cells(cells, "A6", "arm3_beta02", 1)
a6_nh2 = group_cells(cells, "A6", "arm3_beta02", 2)
a6_nh4 = group_cells(cells, "A6", "arm3_beta02", 4)
m_d0_table(a6_nh1, "A6 arm3 n_h=1 (3 seeds)")
m_d0_table(a6_nh2, "A6 arm3 n_h=2 (5 seeds, the never-pinned decisive config)")
m_d0_table(a6_nh4, "A6 arm3 n_h=4 (3 seeds, discharged decisive config)")
m_d2_table(a6_nh1, "A6 arm3 n_h=1")
m_d2_table(a6_nh2, "A6 arm3 n_h=2")
m_d2_table(a6_nh4, "A6 arm3 n_h=4")
m_d1_crosscheck_table(a6_nh2, "A6 arm3 n_h=2")
m_d1_crosscheck_table(a6_nh4, "A6 arm3 n_h=4")

print("\n" + "=" * 80)
print("QUESTION (b): S5 seed-1 catastrophic non-generalizer vs 4 healthy siblings")
print("=" * 80)
s5_nh4 = group_cells(cells, "S5", "arm3_beta02", 4)
m_d0_table(s5_nh4, "S5 arm3 n_h=4 (all 5 seeds, seed1=catastrophic)")
m_d2_table(s5_nh4, "S5 arm3 n_h=4 (all 5 seeds)")
m_d1_crosscheck_table(s5_nh4, "S5 arm3 n_h=4 (all 5 seeds)")
seed1 = [c for c in s5_nh4 if c["seed"] == 1]
healthy = [c for c in s5_nh4 if c["seed"] != 1]
m_d2_table(seed1, "S5 arm3 n_h=4 seed1 ONLY (catastrophic)")
m_d2_table(healthy, "S5 arm3 n_h=4 seeds {0,2,3,4} (healthy)")
m_d1_crosscheck_table(seed1, "S5 arm3 n_h=4 seed1 ONLY (catastrophic)")
m_d1_crosscheck_table(healthy, "S5 arm3 n_h=4 seeds {0,2,3,4} (healthy)")
print("\nfinal_loss by seed (S5 arm3 nh4):", {c["seed"]: c["final_loss"] for c in s5_nh4})

print("\n" + "=" * 80)
print("QUESTION (c): A5 3/5 seeds non-converged -- budget artifact or ceiling?")
print("=" * 80)
a5_arm2 = group_cells(cells, "A5", "arm2_beta01", 2)
a5_arm3 = group_cells(cells, "A5", "arm3_beta02", 2)
m_d0_table(a5_arm2, "A5 arm2 n_h=2 (5 seeds)")
m_d0_table(a5_arm3, "A5 arm3 n_h=2 (5 seeds)")
m_d2_table(a5_arm3, "A5 arm3 n_h=2 (5 seeds)")
m_d1_crosscheck_table(a5_arm3, "A5 arm3 n_h=2 (5 seeds)")
# Converged/non-converged split by final_loss, the registry's own established
# proxy (S2.31a ground 2: "non-converged cell ... final_loss 0.11-0.35" vs
# "the lenses disagree ONLY where final_loss ~= 1e-4"; S2.33 item 3: "the
# flagged three are all final_loss 0.077-0.253"). NOT by the primary-lens
# own_d_train_ceiling -- that lens is the documented broken instrument on
# converged composer cells (S2.31a ground 3, reproduced 4x), so a split
# keyed on it would misclassify every converged cell as non-converged.
# Threshold 1e-2: separates the observed bimodal distribution (converged
# cluster <= ~1.2e-2 vs non-converged cluster >= 6.9e-2) at the widest gap.
CONV_LOSS = 2e-2
for label, group_list in (("A5 arm3", a5_arm3), ("A5 arm2", a5_arm2)):
    conv = [c for c in group_list if c["final_loss"] < CONV_LOSS]
    nonconv = [c for c in group_list if c["final_loss"] >= CONV_LOSS]
    print(f"\n{label}: converged (final_loss < {CONV_LOSS}) seeds = {sorted(c['seed'] for c in conv)}, "
          f"non-converged seeds = {sorted(c['seed'] for c in nonconv)}")
    print("  final_loss by seed: " + ", ".join(
        f"seed{c['seed']}={c['final_loss']:.2e}" for c in sorted(group_list, key=lambda c: c['seed'])))
    if conv:
        m_d2_table(conv, f"{label} converged seeds")
        m_d1_crosscheck_table(conv, f"{label} converged seeds")
    if nonconv:
        m_d2_table(nonconv, f"{label} non-converged seeds")
        m_d1_crosscheck_table(nonconv, f"{label} non-converged seeds")

# Cross-group convergence context: full final_loss census, all 64 cells,
# to make the bimodality (and any budget-vs-ceiling reading) inspectable.
print("\n" + "=" * 80)
print("Full final_loss census (all 64 cells, sorted) -- bimodality check")
print("=" * 80)
for cid, c in sorted(cells.items(), key=lambda kv: kv[1]["final_loss"]):
    print(f"{c['final_loss']:.3e}  {cid}  steps={c['steps_completed']}")
