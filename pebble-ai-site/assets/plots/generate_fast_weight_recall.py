"""
Generate the two figures for the fast-weight-recall findings page.

Figure 1 (fast_weight_recall_sep.svg): two panels.
Left: episode-restricted K-way top-1 recall (acc_A) at K=32 for the three
matched arms (fast-weight contender, param-matched flat-vector ablation,
FLOP/data-matched transformer), with the chance line (1/32) and the
pre-registered demonstration bar (3x chance).
Right: the S0-necessity check — contender recall with both states intact,
with block-0 state (S0) zeroed, and with block-1 state (S1) zeroed.
Data sources:
  experiment-runs/2026-07-09_h2h_sweep_launch/round4_inputs/
    {contender,ablation,transformer}_task1_calib_round4.json
  (headline round-4 calibration cells; n=4096 queries/cell, single seed;
   round4_inputs lives in the project's full experiment archive)

Figure 2 (fast_weight_tap.svg): tap localization — fraction of held-out
queries whose ridge-decoded vector clears cosine threshold tau (rf@tau),
for four tap placements on the contender. Only the post-block-1
pre-LM-head hidden state is linearly decodable.
Data source:
  experiment-runs/2026-07-09_h2h_tap_localization/results/
    tap_localization_contender.json

Palette: Okabe-Ito (colorblind-safe). No in-figure titles (captions live
in the HTML). Background matches the site (#FAF5E7).
"""
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

CHANCE = 0.03125       # 1/K, K=32
DEMO_BAR = 0.09375     # 3x chance, pre-registered demonstration bar

# ---------------------------------------------------------------- figure 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2), facecolor=BG)

# Left panel: recall separation
arms = ["fast-weight\ncontender", "flat-vector\nablation\n(param-matched)",
        "transformer\n(FLOP/data-\nmatched)"]
accs = [0.9990234375, 0.044677734375, 0.029541015625]
colors = [OI_VERMILLION, OI_BLUE, OI_SKY]

ax1.set_facecolor(BG)
xs = np.arange(3)
ax1.bar(xs, accs, width=0.6, color=colors, edgecolor=TEXT, linewidth=1.0,
        zorder=3, alpha=0.9)
for x, a in zip(xs, accs):
    ax1.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=9, color=TEXT,
                 fontweight="bold")
ax1.axhline(CHANCE, color=TEXT, linewidth=0.9, linestyle=":", alpha=0.7,
            zorder=2)
ax1.axhline(DEMO_BAR, color=TEXT, linewidth=0.9, linestyle="--", alpha=0.7,
            zorder=2)
ax1.annotate("chance (1/32)", (2.45, CHANCE), fontsize=7.5, color=MUTED,
             va="bottom", ha="right")
ax1.annotate("demonstration bar (3x chance)", (2.45, DEMO_BAR), fontsize=7.5,
             color=MUTED, va="bottom", ha="right")
ax1.set_xticks(xs)
ax1.set_xticklabels(arms, fontsize=8)
ax1.set_ylabel("recall acc_A (K=32, top-1)", fontsize=9.5, labelpad=8)
ax1.set_ylim(0, 1.09)

# Right panel: S0-necessity (contender)
conds = ["both states\nintact", "S0 zeroed\n(block 0)", "S1 zeroed\n(block 1)"]
vals = [0.9990234375, 0.028564453125, 0.9990234375]
ax2.set_facecolor(BG)
ax2.bar(np.arange(3), vals, width=0.6, color=OI_VERMILLION, edgecolor=TEXT,
        linewidth=1.0, zorder=3, alpha=0.9)
for x, v in zip(np.arange(3), vals):
    ax2.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=9, color=TEXT,
                 fontweight="bold")
ax2.axhline(CHANCE, color=TEXT, linewidth=0.9, linestyle=":", alpha=0.7,
            zorder=2)
ax2.annotate("chance (1/32)", (2.45, CHANCE), fontsize=7.5, color=MUTED,
             va="bottom", ha="right")
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(conds, fontsize=8)
ax2.set_ylabel("contender recall acc_A", fontsize=9.5, labelpad=8)
ax2.set_ylim(0, 1.09)

for ax in (ax1, ax2):
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22,
            color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fast_weight_recall_sep.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote fast_weight_recall_sep.svg")

# ---------------------------------------------------------------- figure 2
taus = [0.5, 0.6, 0.7, 0.8, 0.9]
taps = [
    ("pre-LM-head hidden (post-block-1)",
     [0.99658203, 0.97875977, 0.93164062, 0.83227539, 0.67431641],
     OI_VERMILLION, "o", "-"),
    ("S1 @ true query (old registered tap)",
     [0.00341797, 0.0, 0.0, 0.0, 0.0], OI_BLUE, "s", "--"),
    ("S1 @ shallow query",
     [0.00317383, 0.00073242, 0.0, 0.0, 0.0], OI_SKY, "^", "--"),
    ("S0 @ q0 pathway (tap on S0 itself)",
     [0.0, 0.0, 0.0, 0.0, 0.0], OI_GREEN, "D", ":"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=BG)
ax.set_facecolor(BG)
for label, ys, color, marker, ls in taps:
    ax.plot(taus, ys, color=color, linewidth=2.0, marker=marker,
            markersize=6.5, linestyle=ls, label=label, zorder=3,
            markeredgecolor=TEXT, markeredgewidth=0.6)

ax.set_xlabel("cosine threshold tau", fontsize=10, labelpad=8)
ax.set_ylabel("recovered fraction rf@tau (ridge tap)", fontsize=10,
              labelpad=8)
ax.set_xticks(taus)
ax.set_ylim(-0.04, 1.06)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

legend = ax.legend(loc="center left", frameon=True, fontsize=8.5,
                   facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fast_weight_tap.svg", format="svg", facecolor=BG,
            bbox_inches="tight")
plt.close(fig)
print("wrote fast_weight_tap.svg")
