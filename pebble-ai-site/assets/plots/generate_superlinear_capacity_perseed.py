"""
Generate the complementary raw-per-seed scatter figure (fig 2) for the
super-linear-capacity findings page.

generate_superlinear_capacity.py (fig 1) shows the AGGREGATE (per-K mean)
logistic fits. This script shows the individual seed measurements that sit
underneath those means, so the reader can see the actual scatter/noise the
fit was extracted from.

Metric: h4 = held-out 4-hop recall@0.9, read from each raw DeltaNet
rd_exactness result JSON as checkpoints[-1]['M3_held_out']['4']
['recovered_frac@0.9']. Verified against the audited verification table:
d80 K=48 seed=1130 -> table h4=0.8732, raw JSON gives 0.873168... (matches
to the table's 4-decimal rounding).

Data sources (all under experiment-runs/, this repo):

  d=80 K=48,53 (n=5 each: 3 original + 2 pre-registered escalation seeds)
  and d=96 K=69,72,78,84,90 (n=3 each, "wide"/unlocked window):
    2026-07-07_keyanchor_scaling_wide/fits/per_cell_verification_table.txt
    (columns: group K d seed complete steps timeout wall_s admiss
    fallback h4 — parsed directly below)

  d=64 K=34,38,42,46 (n=3 each), the core cliff sweep:
    2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/
    wavekeyanchor-cliff/*.json

  d=64 K=16 (n=1), K=32 (n=3), K=48 (n=3) — extends the d=64 scatter to
  match generate_superlinear_capacity.py's full 7-point aggregate curve.
  Sourced from earlier waves in the same rev7 lineage; the n=3 cells were
  cross-checked to reproduce the aggregate script's hard-coded means
  exactly (K=32: mean(0.674072265625, 0.7125244140625, 0.61407470703125)
  = 0.6668904622... matches d64_y[1]=0.6668904622 in
  generate_superlinear_capacity.py to 10 significant figures; K=48
  similarly exact against d64_y[6]=0.0215115026):
    2026-07-06_keyanchor_mech/wavekeyanchor-mech/
      wkeyanchor-mech_rdx_K16_armd_s10_geo3n12_anchor_learned_dprobe_rev7.json
      wkeyanchor-mech_rdx_K32_armd_s{10,11,12}_geo3n20_anchor_learned_dprobe_rev7.json
    2026-07-07_keyanchor_k48/wavekeyanchor-k48/
      wkeyanchor-k48_rdx_K48_armd_s{30,31,32}_geo3n20_anchor_learned_dprobe_rev7.json

  d=80 K=20,43,58 (n=3 each) and d=96 K=24,51,57,63 (n=3 each) — the "low
  window" points of the aggregate curve, not covered by the wide-dir
  verification table:
    2026-07-07_keyanchor_scaling/results/deltanet_rd_exactness/
    wavekeyanchor-scaling/*.json

  d=96 K=69 seed=1733 (n=1) — a contingency reseed that replaced seed 1730
  (flagged admiss=False in the verification table) in the "unlocked
  high-window" refit; kept here as a 4th distinct raw measurement at
  K=69 rather than silently dropped, since seeds 1730/1731/1732 are
  already pulled in from the table above and 1731/1732 are literally the
  same files reused across the low-window and high-window analyses (see
  the verification table's own footnote: "incl reused K69 copies"):
    2026-07-07_keyanchor_scaling_wide/results/deltanet_rd_exactness/
    wavekeyanchor-scaling-wide/
    wkeyanchor-scaling_rdx_K69_armd_s1733_geo3n20_anchor_learned_dprobe_rev7_d96.json

Fit curves overlaid for d=64 and d=80 reuse the exact sigmoid_fit
parameters (L, x0, w) verified against the raw fit-result JSONs:
  d64: experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json
       -> L=1.0030024217001399, x0=0.5454626254376084, w=0.05967725709026794
  d80: experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/
       fit_cliff_curve_d80_refit_results.json
       -> L=0.9993619769939635, x0=0.6779197570495672, w=0.047870385269021265
d=96 has no valid fit (100% bootstrap degeneracy) — scatter only, no curve.

Palette: Okabe-Ito (colorblind-safe). No in-figure title (caption lives in
the HTML). Background matches the site (#FAF5E7).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/samuellarson/Experiments/learned-representations"
RUNS = os.path.join(REPO, "experiment-runs")

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_GREEN = "#009E73"
COLOR = {64: OI_BLUE, 80: OI_VERMILLION, 96: OI_GREEN}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

OUT = os.path.join(REPO, "pebble-ai-site/assets/plots/superlinear_capacity_perseed.svg")


# --------------------------------------------------------------- parsing

def h4_from_result_json(path):
    """Extract (K, d_state, seed, h4) from one raw rd_exactness result JSON.

    h4 is held-out 4-hop recall@0.9 read off the LAST checkpoint (step
    20000 in every cell used here): checkpoints[-1]['M3_held_out']['4']
    ['recovered_frac@0.9'].
    """
    with open(path) as f:
        j = json.load(f)
    ck = j["checkpoints"][-1]
    h4 = ck["M3_held_out"]["4"]["recovered_frac@0.9"]
    return j["K"], j["d_state"], j["seed"], h4


def parse_verification_table(path):
    """Parse per_cell_verification_table.txt.

    Header: group K d seed complete steps timeout wall_s admiss fallback h4
    Returns a list of (group, K, d, seed, h4) tuples, one per data row.
    Trailing summary lines ("Total NEW cells...", "Sum wall_s...") and the
    blank separator line are skipped.
    """
    rows = []
    header_seen = False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split()
            if parts[0] == "group":
                header_seen = True
                continue
            if not header_seen:
                continue
            if parts[0] in ("Total", "Sum"):
                break
            group, K, d, seed = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            h4 = float(parts[-1])
            rows.append((group, K, d, seed, h4))
    return rows


# ---- 1. per_cell_verification_table.txt: d80 K=48/53 (n=5), d96 K=69-90 (n=3)
table_path = os.path.join(
    RUNS, "2026-07-07_keyanchor_scaling_wide/fits/per_cell_verification_table.txt"
)
table_rows = parse_verification_table(table_path)

# points[d_state] = list of (K, seed, h4)
points = {64: [], 80: [], 96: []}
seen = set()  # (d_state, K, seed) dedup key

for group, K, d, seed, h4 in table_rows:
    key = (d, K, seed)
    if key in seen:
        continue
    seen.add(key)
    points[d].append((K, seed, h4))

# ---- 2. d=64 core cliff sweep (K=34,38,42,46, n=3 each)
cliff_dir = os.path.join(
    RUNS,
    "2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff",
)
for fname in sorted(os.listdir(cliff_dir)):
    if not fname.endswith(".json"):
        continue
    K, d, seed, h4 = h4_from_result_json(os.path.join(cliff_dir, fname))
    key = (d, K, seed)
    if key in seen:
        continue
    seen.add(key)
    points[d].append((K, seed, h4))

# ---- 3. d=64 extra points K=16 (n=1), K=32 (n=3), K=48 (n=3)
extra_d64 = [
    "2026-07-06_keyanchor_mech/wavekeyanchor-mech/"
    "wkeyanchor-mech_rdx_K16_armd_s10_geo3n12_anchor_learned_dprobe_rev7.json",
    "2026-07-06_keyanchor_mech/wavekeyanchor-mech/"
    "wkeyanchor-mech_rdx_K32_armd_s10_geo3n20_anchor_learned_dprobe_rev7.json",
    "2026-07-06_keyanchor_mech/wavekeyanchor-mech/"
    "wkeyanchor-mech_rdx_K32_armd_s11_geo3n20_anchor_learned_dprobe_rev7.json",
    "2026-07-06_keyanchor_mech/wavekeyanchor-mech/"
    "wkeyanchor-mech_rdx_K32_armd_s12_geo3n20_anchor_learned_dprobe_rev7.json",
    "2026-07-07_keyanchor_k48/wavekeyanchor-k48/"
    "wkeyanchor-k48_rdx_K48_armd_s30_geo3n20_anchor_learned_dprobe_rev7.json",
    "2026-07-07_keyanchor_k48/wavekeyanchor-k48/"
    "wkeyanchor-k48_rdx_K48_armd_s31_geo3n20_anchor_learned_dprobe_rev7.json",
    "2026-07-07_keyanchor_k48/wavekeyanchor-k48/"
    "wkeyanchor-k48_rdx_K48_armd_s32_geo3n20_anchor_learned_dprobe_rev7.json",
]
for rel in extra_d64:
    K, d, seed, h4 = h4_from_result_json(os.path.join(RUNS, rel))
    key = (d, K, seed)
    if key in seen:
        continue
    seen.add(key)
    points[d].append((K, seed, h4))

# ---- 4. "low window" points: d80 K=20,43,58 and d96 K=24,51,57,63
scaling_dir = os.path.join(
    RUNS,
    "2026-07-07_keyanchor_scaling/results/deltanet_rd_exactness/wavekeyanchor-scaling",
)
for fname in sorted(os.listdir(scaling_dir)):
    if not fname.endswith(".json"):
        continue
    K, d, seed, h4 = h4_from_result_json(os.path.join(scaling_dir, fname))
    key = (d, K, seed)
    if key in seen:
        continue
    seen.add(key)
    points[d].append((K, seed, h4))

# ---- 5. d=96 K=69 seed=1733, the contingency reseed not in the table
extra_d96_k69 = os.path.join(
    RUNS,
    "2026-07-07_keyanchor_scaling_wide/results/deltanet_rd_exactness/"
    "wavekeyanchor-scaling-wide/"
    "wkeyanchor-scaling_rdx_K69_armd_s1733_geo3n20_anchor_learned_dprobe_rev7_d96.json",
)
K, d, seed, h4 = h4_from_result_json(extra_d96_k69)
key = (d, K, seed)
if key not in seen:
    seen.add(key)
    points[d].append((K, seed, h4))

n_total = sum(len(v) for v in points.values())
print(f"Parsed {n_total} unique (K, d, seed) points:",
      {d: len(v) for d, v in points.items()})
for d in (64, 80, 96):
    ks = sorted(set(k for k, s, h in points[d]))
    print(f"  d={d}: K values {ks}  (K/d range "
          f"{min(ks)/d:.4f}-{max(ks)/d:.4f})")


# ----------------------------------------------------------------- plot

def logistic(x, L, x0, w):
    return L / (1.0 + np.exp((x - x0) / w))

FIT = {
    64: dict(L=1.0030024217001399, x0=0.5454626254376084, w=0.05967725709026794),
    80: dict(L=0.9993619769939635, x0=0.6779197570495672, w=0.047870385269021265),
}

fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=BG)
ax.set_facecolor(BG)

xs = np.linspace(0.2, 1.0, 400)
for d in (64, 80):
    fit = logistic(xs, **FIT[d])
    ax.plot(xs, fit, color=COLOR[d], linewidth=1.6, alpha=0.85, zorder=2)
    ax.axvline(FIT[d]["x0"], color=COLOR[d], linewidth=0.9, linestyle="--",
               alpha=0.55, zorder=1)

# small deterministic per-point x-jitter so same-(K,d) seeds don't stack
# exactly on top of each other; y values are never jittered.
rng = np.random.default_rng(20260709)
marker = {64: "o", 80: "s", 96: "^"}
for d in (64, 80, 96):
    rows = points[d]
    xvals = np.array([K / d for K, seed, h4 in rows])
    yvals = np.array([h4 for K, seed, h4 in rows])
    jitter = rng.uniform(-0.003, 0.003, size=len(xvals))
    n = len(rows)
    ax.scatter(xvals + jitter, yvals, color=COLOR[d], s=26, alpha=0.55,
               marker=marker[d], edgecolor=TEXT, linewidth=0.4, zorder=3,
               label=f"d={d} (n={n} seed measurements)")

ax.set_xlabel("K / d_state (bindings per state dimension)", fontsize=10,
              labelpad=8)
ax.set_ylabel("held-out 4-hop recall (rec@0.9), per seed", fontsize=10,
              labelpad=8)
ax.set_xlim(0.2, 1.0)
ax.set_ylim(-0.03, 1.06)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

legend = ax.legend(loc="lower left", frameon=True, fontsize=8.5,
                    facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(OUT, format="svg", facecolor=BG, bbox_inches="tight")
print(f"wrote {OUT}")
