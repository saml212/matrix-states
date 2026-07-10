"""
Generate the two figures for the write-geometry-attractor findings page.

Figure 1 (write_geometry_scale.svg): span_frac vs parameter count, the
4-point scale ladder (14M -> 1.31B), archived-4-corpus subset.
Data source: EXPERIMENT_LOG.md SCALE-TRANSFER Track C entry (lines ~5502-5507);
raw archives under experiment-runs/ (scale-ladder rungs 1-3 + mixcontrol).

Figure 2 (write_geometry_2x2.svg): the qk-norm x gating 2x2 at the 14M rung,
n=3 seeds per cell, same-corpus (openr1-mix-ext) layer-pooled
gram_deviation_mean. NOTE: this instrument reports gram-deviation, not
span_frac — the two figures use different metrics and are never overlaid.
Data sources:
  experiment-runs/2026-07-09_attrrob_2x2_harvest/box_results/AGGREGATE.json (n=1)
  experiment-runs/2026-07-09_attrrob_2x2_escalation_harvest/box_results/AGGREGATE.json
  experiment-runs/2026-07-09_attrrob_2x2_escalation_harvest/n3_recompute_summary.json

Palette: Okabe-Ito (colorblind-safe). No in-figure titles (captions live in
the HTML). Background matches the site (#FAF5E7).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Site background / text
BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_BLACK = "#000000"
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

# ---------------------------------------------------------------- figure 1
# Scale ladder: span_frac at 4 scales (EXPERIMENT_LOG.md, Track C).
params = np.array([14_048_896, 98e6, 392e6, 1_311_135_488])
span_frac = np.array([0.248, 0.344, 0.389, 0.455])
labels = ["14M", "98M", "392M", "1.31B"]

fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=BG)
ax.set_facecolor(BG)

ax.plot(params, span_frac, color=OI_VERMILLION, linewidth=2.2,
        marker="o", markersize=7, zorder=3)
for x, y, lab in zip(params, span_frac, labels):
    ax.annotate(f"{lab}\n{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8.5, color=TEXT)

ax.set_xscale("log")
ax.set_xlabel("parameters (log scale)", fontsize=10, labelpad=8)
ax.set_ylabel("span_frac (write-geometry collapse)", fontsize=10, labelpad=8)
ax.set_xlim(8e6, 3.2e9)
ax.set_ylim(0.20, 0.52)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/write_geometry_scale.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote write_geometry_scale.svg")

# ---------------------------------------------------------------- figure 2
# 2x2 at 14M, n=3 seeds per cell, gram_deviation_mean (openr1-mix-ext).
cells = [
    # (label, per-seed values, color)
    ("qk-norm ON\ngating OFF\n(baseline)", [19.069, 22.737, 18.777], OI_BLUE),
    ("qk-norm OFF\ngating OFF", [18.196, 20.873, 21.205], OI_SKY),
    ("qk-norm ON\ngating ON", [25.232, 25.862, 22.424], OI_VERMILLION),
    ("qk-norm OFF\ngating ON", [21.204, 20.800, 21.963], OI_ORANGE),
]

fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=BG)
ax.set_facecolor(BG)

xs = np.arange(len(cells))
for i, (label, seeds, color) in enumerate(cells):
    seeds = np.array(seeds)
    mean, sd = seeds.mean(), seeds.std(ddof=1)
    ax.bar(i, mean, width=0.62, color=color, edgecolor=TEXT,
           linewidth=1.0, zorder=2, alpha=0.85)
    ax.errorbar(i, mean, yerr=sd, color=TEXT, capsize=5,
                linewidth=1.2, zorder=4)
    # per-seed points
    jitter = np.array([-0.13, 0.0, 0.13])
    ax.scatter(i + jitter, seeds, color=TEXT, s=22, zorder=5,
               facecolor=BG, edgecolor=TEXT, linewidth=1.1)
    ax.annotate(f"{mean:.2f}", (i, mean), textcoords="offset points",
                xytext=(30, 4), fontsize=9, color=TEXT, fontweight="bold")

# baseline mean reference line
baseline_mean = np.mean(cells[0][1])
ax.axhline(baseline_mean, color=TEXT, linewidth=0.9, linestyle="--",
           alpha=0.6, zorder=1)
ax.annotate(f"baseline mean {baseline_mean:.2f}",
            (1.5, baseline_mean), fontsize=8, color=MUTED,
            va="bottom", ha="center", xytext=(0, 3),
            textcoords="offset points")

ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c in cells], fontsize=8.5)
ax.set_ylabel("gram_deviation_mean (openr1-mix-ext)", fontsize=10, labelpad=8)
ax.set_ylim(0, 29)
ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/write_geometry_2x2.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote write_geometry_2x2.svg")

# ---------------------------------------------------------------- figure 3
# (added 2026-07-10, fix-at-scale harvest)
# write_geometry_fixscale.svg: the fix-at-scale wave verdict. Left panel:
# the deployed per_token frozen-bias arm's post-blend span_frac delta vs the
# arm_off reference (positive = destabilizing = the 14M sign), with pinned
# 95% CIs, at 14M / 98M / 392M on both corpora. Right panel: the
# global-vector construction (the arm that STABILIZED at 14M): n=3 CI at
# 14M; n=1 exploratory probe (no CI, per the pre-registration) at 98M/392M.
# Data sources (both read at generation time, never hardcoded):
#   experiment-runs/2026-07-06_frozen_bias_rung1/results/frozen_bias_lm/
#     PHASE_D_FULL_REPORT.json          (md5 b12a9e376805ff91aa08c270df15b539)
#   experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json
#                                       (md5 f2f0aae84908c0db0a42b13c76a85158;
#      archive md5 manifest verified 132/132)
# Cross-scale magnitude caveat: 392M ran a reduced 20k-step budget vs 98M's
# Track-C-matched budget — the 98M->392M attenuation is confounded with
# token budget by design; within-scale readings are the registered claims.
import json
from pathlib import Path

_REPO = Path(OUT_DIR).resolve().parents[2]
with open(_REPO / "experiment-runs/2026-07-06_frozen_bias_rung1/results/"
                  "frozen_bias_lm/PHASE_D_FULL_REPORT.json") as f:
    _rung1 = json.load(f)
with open(_REPO / "experiment-runs/2026-07-10_fixscale_harvest/"
                  "fixscale_harvest_verdict.json") as f:
    _fx = json.load(f)

CORPORA = [("openr1-mix-ext", "openr1"), ("wikitext-mix-ext", "wikitext")]

# per_token rows: (label, delta, ci_lo, ci_hi)
pt_rows, gl_rows = [], []
for corpus, short in CORPORA:
    p = _rung1["primary_and_coprimary"]["post_blend_primary"][corpus]
    pt_rows.append((f"14M · {short}", p["mean_delta"], p["ci_lower"],
                    p["ci_upper"]))
    g = _rung1["arm2prime_vs_arm1double"][corpus]
    gl_rows.append((f"14M · {short}", g["mean_delta"], g["ci_lower"],
                    g["ci_upper"]))
for scale in ["98m", "392m"]:
    for corpus, short in CORPORA:
        c = _fx["scales"][scale]["corpora"][corpus]
        pt_rows.append((f"{scale.upper()} · {short}", c["PRIMARY_delta"],
                        c["PRIMARY_ci"][0], c["PRIMARY_ci"][1]))
        gl_rows.append((f"{scale.upper()} · {short}",
                        c["probe_exploratory_n1_no_CI"]
                        ["probe_delta_vs_arm_off_double_mean"], None, None))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 4.6), facecolor=BG,
                               sharey=True)
for ax, rows, color, panel in ((axL, pt_rows, OI_VERMILLION, "per_token"),
                               (axR, gl_rows, OI_GREEN, "global")):
    ax.set_facecolor(BG)
    ys = np.arange(len(rows))[::-1]
    for y, (label, d, lo, hi) in zip(ys, rows):
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color, linewidth=2.0, zorder=3)
            ax.plot([lo, lo], [y - 0.14, y + 0.14], color=color,
                    linewidth=2.0, zorder=3)
            ax.plot([hi, hi], [y - 0.14, y + 0.14], color=color,
                    linewidth=2.0, zorder=3)
            ax.plot(d, y, "o", color=color, markersize=7.5, zorder=4,
                    markeredgecolor=TEXT, markeredgewidth=0.6)
        else:
            ax.plot(d, y, "D", color=color, markersize=7.5, zorder=4,
                    markeredgecolor=TEXT, markeredgewidth=0.6)
        ax.annotate(f"{d:+.3f}", (d, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=TEXT,
                    fontweight="bold")
    ax.axvline(0.0, color=TEXT, linewidth=0.9, linestyle="--", alpha=0.7,
               zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(-0.70, 0.42)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.22,
            color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)

axL.set_xlabel("Δ span_frac (per_token − off)  ·  positive = destabilizing",
               fontsize=9, labelpad=8)
axR.set_xlabel("Δ span_frac (global-vector − off)  ·  negative = stabilizing",
               fontsize=9, labelpad=8)
axL.annotate("deployed per_token arm:\nsign persists at scale",
             (-0.67, 0.0), fontsize=8.5, color=OI_VERMILLION, ha="left",
             fontweight="bold")
axR.annotate("global-vector arm: the 14M\nstabilization decays to ≈0,\n"
             "sign flips at 392M · wikitext", (-0.67, 1.4), fontsize=8.5,
             color=OI_GREEN, ha="left", fontweight="bold")
axR.annotate("diamonds = n=1 exploratory\nprobe, no CI (pre-registered)",
             (0.38, 4.6), fontsize=7.5, color=MUTED, ha="right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/write_geometry_fixscale.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote write_geometry_fixscale.svg")
