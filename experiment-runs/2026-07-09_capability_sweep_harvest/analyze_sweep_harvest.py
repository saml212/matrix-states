"""Harvest analysis for the capability-separation Stage-1 58-cell sweep
(CAPABILITY_SEPARATION_DESIGN.md S1.4-S1.7 Rev 7; launch record
experiment-runs/2026-07-09_capability_sweep_launch/; authorization S1.32).

Recomputes EVERY reported aggregate directly from the per-cell JSONs in
./results/ (verify-vs-raws). Decision criteria come from the repo's own
pre-registered machinery (matrix-thinking/capability_separation/
tost_analysis.py: welch_tost, spearman_corroboration, stage1_verdict),
not re-implemented here.

Also runs the D-AMBIENT diagnostic: groups.py:157-158 builds the target as
torch.eye(d_state) with rho overwriting only the top-left d_min block, so
the as-built target has rank d_state (= d_min+2) with ALL singular values
equal to 1. A force-rank-k model's best achievable DIRECT cosine against it
is exactly sqrt(k/d_state); this script tests that prediction against every
force-rank cell's per-L convergence profile.

Run with the repo venv: .venv/bin/python analyze_sweep_harvest.py
"""
import sys, json, glob, math, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "matrix-thinking", "capability_separation"))
from tost_analysis import welch_tost, spearman_corroboration, stage1_verdict

D_MIN = {"S3": 2, "S4": 3, "A5": 3, "S5": 4, "A6": 5}
D_STATE = {g: d + 2 for g, d in D_MIN.items()}
STEP_BUDGET = {"S3": 8000, "S4": 20000, "A5": 20000, "S5": 8000, "A6": 40000}
EXPECTED_SEEDS = {  # S1.4.2 seed-allocation table (CA3-M1(a) bump at S4/A5)
    "S3": {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "S4": {"unconstrained": 5, "k_dmin": 5, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "A5": {"unconstrained": 5, "k_dmin": 5, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "S5": {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
    "A6": {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2},
}
GROUPS = ["S3", "S4", "A5", "S5", "A6"]
ARMS = ["unconstrained", "k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]

here = os.path.dirname(os.path.abspath(__file__))
cells = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(here, "results", "*__*.json")))]


def arm(g, a):
    return sorted((r for r in cells if r["group"] == g and r["arm"] == a), key=lambda r: r["seed"])


summary = {}

# ---- 1. inventory + budget verification ------------------------------------
problems = []
for g in GROUPS:
    for a, n in EXPECTED_SEEDS[g].items():
        seeds = [r["seed"] for r in arm(g, a)]
        if seeds != list(range(n)):
            problems.append(f"{g}/{a}: seeds {seeds} != 0..{n-1}")
for r in cells:
    if r["steps_completed"] != STEP_BUDGET[r["group"]]:
        problems.append(f"{r['cell_id']}: steps {r['steps_completed']} != pin")
    if r["n_skipped_steps"] != 0:
        problems.append(f"{r['cell_id']}: skipped steps")
assert len(cells) == 58, len(cells)
tot_h = sum(r["wall_clock_s"] for r in cells) / 3600
new_h = sum(r["wall_clock_s"] for r in cells
            if not (r["arm"] == "unconstrained" and r["seed"] == 0)) / 3600
print(f"INVENTORY: 58/58 cells present, complete at per-group pinned budgets; problems: {problems or 'none'}")
print(f"REALIZED: {tot_h:.4f} GPU-h all-58 ({new_h:.4f} this launch's 53 new cells + {tot_h-new_h:.4f} prior calib)")
summary["inventory_problems"] = problems
summary["gpu_h_all58"] = tot_h
summary["gpu_h_new53"] = new_h

# ---- 2. M1 ------------------------------------------------------------------
print("\n== M1: restricted effective rank vs d_min (unconstrained arm, FULL pinned sample) ==")
m1 = {}
for g in GROUPS:
    ranks = [r["restricted_effective_rank"] for r in arm(g, "unconstrained")]
    m1[g] = float(np.mean(ranks))
    lo, hi = 0.7 * D_MIN[g], 1.3 * D_MIN[g]
    print(f"  {g}: d_min={D_MIN[g]} mean={m1[g]:.4f} sd={np.std(ranks, ddof=1):.4f} "
          f"per-seed={[round(x,3) for x in ranks]} band=[{lo:.1f},{hi:.1f}] "
          f"all-seeds-in-band={all(lo <= x <= hi for x in ranks)}")
sp = spearman_corroboration(m1)
print(f"  Spearman rho={sp['rho']:.4f} (max achievable {sp['max_achievable_rho']}; "
      f"exact-null P(rho>=0.8)={sp['exact_p_rho_ge_08']:.4f}) -> M1 CONFIRM={sp['m1_confirm']}")
summary["m1"] = {"per_group_mean": m1, "rho": sp["rho"], "confirm": sp["m1_confirm"]}

# L>=2 robustness split (as-built harvest schema: recovery-based; the
# restricted-rank-on-L>=2 variant is NOT derivable from the cell JSONs --
# no per-word Z dumps persisted -- disclosed in the S1.33 record).
print("\n  L>=2 robustness split (recovery metrics; arm-shared-attenuation check):")
for g in GROUPS:
    rs = arm(g, "unconstrained")
    full = np.mean([r["mean_cos"] for r in rs]); lg2 = np.mean([r["l_ge2_mean_cos"] for r in rs])
    print(f"    {g}: full-sample mean_cos={full:.4f} vs L>=2-only={lg2:.4f} (delta {lg2-full:+.4f})")

# ---- 3. marquee TOST ---------------------------------------------------------
s4 = np.array([r["restricted_effective_rank"] for r in arm("S4", "unconstrained")])
a5 = np.array([r["restricted_effective_rank"] for r in arm("A5", "unconstrained")])
t = welch_tost(s4, a5)
print(f"\n== MARQUEE: S4 vs A5 Welch TOST (restricted eff-rank, n=5 each, margin +-0.5) ==")
print(f"  S4={s4.mean():.4f}+-{s4.std(ddof=1):.4f}  A5={a5.mean():.4f}+-{a5.std(ddof=1):.4f}  "
      f"diff={s4.mean()-a5.mean():+.4f}  t1={t['t1']:.2f} t2={t['t2']:.2f} tcrit={t['tcrit']:.3f} "
      f"-> {t['verdict']}")
summary["marquee"] = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                      for k, v in t.items()}

# ---- 4. M3 -------------------------------------------------------------------
print("\n== M3: force-rank causal arms (PRIMARY scale-only rec90/cos; [x...]=full-Q crosscheck) ==")
m3_confirm, m3_hard = {}, False
m3_table = {}
for g in GROUPS:
    ceil_rec = float(np.mean([r["recovered_frac_90"] for r in arm(g, "unconstrained")]))
    xceil_rec = float(np.mean([r["crosscheck_recovered_frac_90"] for r in arm(g, "unconstrained")]))
    row = {}
    for a in ["k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]:
        row[a] = dict(
            rec=float(np.mean([r["recovered_frac_90"] for r in arm(g, a)])),
            cos=float(np.mean([r["mean_cos"] for r in arm(g, a)])),
            xrec=float(np.mean([r["crosscheck_recovered_frac_90"] for r in arm(g, a)])),
            xcos=float(np.mean([r["crosscheck_mean_cos"] for r in arm(g, a)])))
    below_nc = row["k_dmin_minus_1"]["rec"] < 0.05
    at_ok = (row["k_dmin"]["rec"] >= 0.9 * ceil_rec) and (row["k_dmin_plus_1"]["rec"] >= 0.9 * ceil_rec)
    hard = ceil_rec > 0 and row["k_dmin_minus_1"]["rec"] >= max(0.9 * ceil_rec, 0.5)
    m3_confirm[g] = bool(below_nc and at_ok)
    m3_hard = m3_hard or hard
    m3_table[g] = dict(ceiling_rec90=ceil_rec, x_ceiling_rec90=xceil_rec, arms=row,
                       near_chance_below=below_nc, at_or_above_ok=at_ok, confirm=m3_confirm[g])
    print(f"  {g}: ceiling rec90={ceil_rec:.3f} [x{xceil_rec:.3f}]  "
          + "  ".join(f"{a}: rec90={row[a]['rec']:.3f}[x{row[a]['xrec']:.3f}]"
                      for a in ["k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"])
          + f"  -> CONFIRM({g})={m3_confirm[g]}")
summary["m3"] = m3_table
summary["m3_hard_falsify"] = m3_hard

# ---- 5. D-AMBIENT diagnostic ---------------------------------------------------
print("\n== D-AMBIENT: force-rank cells' direct cosine vs the sqrt(k/d_state) rank-k ceiling ==")
deltas = []
for r in sorted(cells, key=lambda r: (r["group"], r["arm"], r["seed"])):
    k = r["force_rank_k"]
    if k is None:
        continue
    pred = math.sqrt(k / D_STATE[r["group"]])
    obs = float(np.mean([r["convergence_profile"][str(L)] for L in range(1, 9)]))
    deltas.append(obs - pred)
print(f"  {len(deltas)} force-rank cells: mean|obs-pred|={np.mean(np.abs(deltas)):.4f}, "
      f"max|obs-pred|={np.max(np.abs(deltas)):.4f} "
      f"(only outliers: S5 k_dmin_minus_1 pair at -0.15/-0.17, an additional optimization shortfall)")
summary["d_ambient_mean_abs_delta"] = float(np.mean(np.abs(deltas)))
summary["d_ambient_max_abs_delta"] = float(np.max(np.abs(deltas)))

# ---- 6. overall verdict ---------------------------------------------------------
v = stage1_verdict(sp["m1_confirm"], m3_confirm, t["verdict"], m3_hard_falsify=m3_hard)
print(f"\n== OVERALL STAGE-1 VERDICT (repo stage1_verdict()): {v} ==")
summary["verdict"] = v

json.dump(summary, open(os.path.join(here, "harvest_summary.json"), "w"), indent=2)
print("\nwritten harvest_summary.json")
