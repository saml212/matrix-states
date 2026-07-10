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
