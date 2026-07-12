"""
Generate the two figures for the rank-law findings page.

Figure 1 (rank_law_razor.svg): the causal razor step — basis-invariant
recovery (crosscheck rec@0.9, "xrec90") at forced rank k = d_min-1, d_min,
d_min+1 for each of the five finite groups, with the unconstrained anchor
as a dashed reference. S3 is the seed-extended group (n=4, seeds 0-3,
error bar = population sd); the other four groups are single-seed cells
per the fix-wave's flanking-cell convention.
Data sources (raw per-cell JSONs, corrected "zero_pad" target family only):
  experiment-runs/2026-07-09_m3fix_harvest/zero_pad__{S4,A5,S5,A6}__*__seed0.json
  experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S3__*__seed0.json
  experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__*__seed{1,2,3}.json

Figure 2 (rank_law_m1.svg): restricted effective rank of the unconstrained
arm vs d_min(G), per-seed points, mean +/- sd, with the pre-registered
[0.7, 1.3] * d_min band. Spearman rho = 0.9747 (tie-capped maximum given
the S4/A5 tie at d_min = 3).
Data source: experiment-runs/2026-07-09_capability_sweep_harvest/
             harvest_analysis_output.txt + harvest_summary.json
(the unconstrained arm was never rank-capped, so this leg is unaffected
by the D-AMB target-padding fix).

Palette: Okabe-Ito (colorblind-safe). No in-figure titles (captions live
in the HTML). Background matches the site (#FAF5E7).

RUNTIME VERIFICATION (added 2026-07, closing a provenance gap flagged by
site audit): figure 2's per-seed points, group means, and Spearman rho were
originally hand-transcribed from the cited raw archives. This block now
loads the actual per-seed cell JSONs and harvest_summary.json at generation
time and asserts every plotted number matches, before anything is drawn.
Figure 1's five-group razor points remain a cited (not runtime-reloaded)
transcription -- they come from 19 separate small per-cell JSONs across two
archive directories (one per group x rank-offset x seed) with no single
summary file; reproducing that fan-in here was judged not worth the added
script complexity given the numbers were independently re-verified against
the raw cells by two separate audit passes. See the file list in the
docstring above for exact provenance.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

OUT_DIR = "/Users/samuellarson/Experiments/learned-representations/pebble-ai-site/assets/plots"

groups = ["S3", "S4", "A5", "S5", "A6"]
d_min = [2, 3, 3, 4, 5]
solvable = [True, True, False, False, False]

# xrec90 per arm. S3 values are the 4-seed means; others are n=1 (seed 0).
xrec90_anchor = [0.6375, 0.650, 0.700, 0.500, 0.650]
xrec90_km1 = [0.0000, 0.000, 0.000, 0.000, 0.000]
xrec90_k = [0.5625, 0.800, 0.700, 0.600, 0.650]
xrec90_kp1 = [0.6625, 0.950, 0.750, 0.550, 0.700]
# S3-only population-sd error bars (n=4); None = n=1, no error bar drawn.
err_anchor = [0.0972, None, None, None, None]
err_k = [0.0739, None, None, None, None]
err_kp1 = [0.0821, None, None, None, None]

# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 5, figsize=(9.6, 3.2), facecolor=BG,
                         sharey=True)
for i, ax in enumerate(axes):
    ax.set_facecolor(BG)
    color = OI_BLUE if solvable[i] else OI_VERMILLION
    xs = [0, 1, 2]
    ys = [xrec90_km1[i], xrec90_k[i], xrec90_kp1[i]]
    ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=6,
            zorder=3)
    # error bars where multi-seed (S3 only)
    if err_k[i] is not None:
        ax.errorbar([1, 2], [xrec90_k[i], xrec90_kp1[i]],
                    yerr=[err_k[i], err_kp1[i]], fmt="none", ecolor=TEXT,
                    capsize=4, linewidth=1.1, zorder=4)
    # unconstrained anchor reference
    ax.axhline(xrec90_anchor[i], color=TEXT, linewidth=0.9,
               linestyle="--", alpha=0.55, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d_min[i]-1}", f"{d_min[i]}", f"{d_min[i]+1}"],
                       fontsize=9)
    ax.set_xlabel("forced rank k", fontsize=8.5)
    tag = "solvable" if solvable[i] else "non-solvable"
    ax.annotate(f"{groups[i]}\nd_min={d_min[i]}\n({tag})", (0.06, 0.97),
                xycoords="axes fraction", fontsize=8.5, va="top",
                color=color, fontweight="bold")
    ax.set_ylim(-0.06, 1.02)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)
axes[0].set_ylabel("crosscheck rec@0.9 (xrec90)", fontsize=9.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rank_law_razor.svg", format="svg", facecolor=BG,
            bbox_inches="tight")
plt.close(fig)
print("wrote rank_law_razor.svg")

# ---------------------------------------------------------------- figure 2
HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
HARVEST_DIR = REPO / "experiment-runs/2026-07-09_capability_sweep_harvest"

with open(HARVEST_DIR / "harvest_summary.json") as f:
    harvest = json.load(f)
print(f"loaded {HARVEST_DIR.relative_to(REPO)}/harvest_summary.json "
      f"(rho={harvest['m1']['rho']:.4f}, confirm={harvest['m1']['confirm']})")

rank_per_seed = {}
for g in groups:
    files = sorted((HARVEST_DIR / "results").glob(f"{g}__unconstrained__seed*.json"))
    assert files, f"no unconstrained-arm cell files found for group {g}"
    vals = []
    for fp in files:
        with open(fp) as f:
            vals.append(round(json.load(f)["restricted_effective_rank"], 3))
    rank_per_seed[g] = vals
print("loaded per-seed restricted_effective_rank from "
      f"{sum(len(v) for v in rank_per_seed.values())} raw cell JSONs across 5 groups")

# cross-check against the (documented) original transcription before trusting the plot
_expected_per_seed = {
    "S3": [1.808, 1.905, 1.919],
    "S4": [2.924, 2.848, 2.884, 2.809, 2.793],
    "A5": [2.882, 2.915, 2.776, 2.785, 2.804],
    "S5": [3.605, 3.652, 3.517],
    "A6": [4.709, 4.748, 4.750],
}
assert rank_per_seed == _expected_per_seed, rank_per_seed

rank_mean = [round(harvest["m1"]["per_group_mean"][g], 4) for g in groups]
# ddof=1 (sample std) to match the page's own prose convention ("Seed-level
# means: S3 1.877 +/- 0.060 (n=3)..." -- verified these are sample-std, not
# population-std, values); a prior version of this script used ddof=0,
# which silently understated the error bars relative to the adjacent text.
rank_sd = [round(float(np.std(rank_per_seed[g], ddof=1)), 4) for g in groups]
assert rank_mean == [1.8771, 2.8517, 2.8323, 3.5913, 4.7357], rank_mean
assert abs(harvest["m1"]["rho"] - 0.9746794344808963) < 1e-9
# small x-offsets so the tied S4/A5 pair at d_min=3 stays legible
x_offsets = {"S3": 0.0, "S4": -0.07, "A5": +0.07, "S5": 0.0, "A6": 0.0}
colors = {"S3": OI_BLUE, "S4": OI_BLUE, "A5": OI_VERMILLION,
          "S5": OI_VERMILLION, "A6": OI_VERMILLION}

fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=BG)
ax.set_facecolor(BG)

# pre-registered band [0.7, 1.3] * d_min
band_x = np.linspace(1.7, 5.3, 100)
ax.fill_between(band_x, 0.7 * band_x, 1.3 * band_x, color=OI_SKY,
                alpha=0.18, zorder=0)
ax.plot(band_x, band_x, color=TEXT, linewidth=0.9, linestyle=":",
        alpha=0.6, zorder=1)
ax.annotate("rank = d_min", (5.0, 5.0), fontsize=8.5, color=MUTED,
            rotation=33, xytext=(-30, 12), textcoords="offset points")
ax.annotate("pre-registered band\n[0.7, 1.3] x d_min", (1.85, 2.62),
            fontsize=8, color=MUTED)

for i, g in enumerate(groups):
    x = d_min[i] + x_offsets[g]
    seeds = rank_per_seed[g]
    ax.scatter([x] * len(seeds), seeds, s=20, facecolor=BG,
               edgecolor=colors[g], linewidth=1.1, zorder=4)
    ax.errorbar(x, rank_mean[i], yerr=rank_sd[i], fmt="D", markersize=7,
                color=colors[g], ecolor=TEXT, capsize=4, linewidth=1.1,
                markeredgecolor=TEXT, markeredgewidth=0.7, zorder=5)
    va = "bottom" if g != "A5" else "top"
    dy = 10 if g != "A5" else -12
    ax.annotate(f"{g} (n={len(seeds)})", (x, rank_mean[i]),
                textcoords="offset points", xytext=(8, dy), fontsize=8.5,
                color=colors[g], fontweight="bold", va=va)

ax.set_xlabel("d_min(G) — minimum faithful representation dimension",
              fontsize=10, labelpad=8)
ax.set_ylabel("restricted effective rank (unconstrained arm)",
              fontsize=10, labelpad=8)
ax.set_xlim(1.6, 5.6)
ax.set_ylim(1.2, 6.8)
ax.set_xticks([2, 3, 4, 5])
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

# legend for solvable / non-solvable
handles = [
    plt.Line2D([], [], marker="D", color=OI_BLUE, linestyle="none",
               markeredgecolor=TEXT, label="solvable group"),
    plt.Line2D([], [], marker="D", color=OI_VERMILLION, linestyle="none",
               markeredgecolor=TEXT, label="non-solvable group"),
]
legend = ax.legend(handles=handles, loc="upper left", frameon=True,
                   fontsize=8.5, facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rank_law_m1.svg", format="svg", facecolor=BG,
            bbox_inches="tight")
plt.close(fig)
print("wrote rank_law_m1.svg")
