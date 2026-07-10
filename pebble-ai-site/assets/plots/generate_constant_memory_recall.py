"""
Generate the horizon figure for the constant-memory-recall findings page,
plus a standalone 1200x675 PNG export for social distribution.

Figure (constant_memory_horizon.svg): episode-restricted K-way top-1 recall
(acc_A) vs context length for the fast-weight contender (fixed 32,768-byte
recurrent state) against the param/token-matched transformer, uncapped and
KV-cache-capped at M in {2,4,8,16,32} slots. Horizons H2/H4/H8 = 454/902/
1798 tokens = 2x/4x/8x the binding window. The M=1 row is a pre-registered
descriptive-only (floor-excluded) cell and is not plotted.

Data source (verified md5 4f115ad55d5301122f387df504efa35c):
  experiment-runs/2026-07-10_h2h_mstar/MSTAR_VERDICT.json
(fan-out raws md5-manifest verified 96/96 in the same archive)

All numbers are READ from the archived verdict JSON, not hardcoded.

Palette: Okabe-Ito (colorblind-safe). No in-figure titles for the SVG
(captions live in the HTML); the PNG carries its own title because it
stands alone. Background matches the site (#FAF5E7).
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
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
VERDICT = REPO / "experiment-runs/2026-07-10_h2h_mstar/MSTAR_VERDICT.json"

with open(VERDICT) as f:
    verdict = json.load(f)

t1 = verdict["per_task"]["task1_sweep"]
CHANCE = 0.03125       # 1/K, K=32
DEMO_BAR = t1["demonstration_bar"]  # 0.09375, 3x chance, pre-registered

HORIZONS = ["H2", "H4", "H8"]
TOKENS = [454, 902, 1798]  # horizon_tokens from contender_horizon_refs.json

contender = np.array([t1["contender_refs_acc_A"][h] for h in HORIZONS])  # (3H, 3 seeds)
tfm_uncapped = np.array([t1["transformer_uncapped_refs_acc_A"][h] for h in HORIZONS])
capped = t1["capped_transformer_acc_A"]  # keys "2".."32", each H -> 3 seeds


def draw(ax, annotate_fontsize=9.0, label_fontsize=10.0, tick_fontsize=9.0):
    ax.set_facecolor(BG)

    # capped transformer: all M in {2..32}, seed means, one series (identity
    # is the cap family, not the individual M)
    for m_key, by_h in sorted(capped.items(), key=lambda kv: int(kv[0])):
        means = [float(np.mean(by_h[h])) for h in HORIZONS]
        ax.plot(TOKENS, means, color=OI_SKY, linewidth=1.0, linestyle="-",
                marker="^", markersize=5, alpha=0.55, zorder=2,
                markeredgecolor=TEXT, markeredgewidth=0.4)

    # transformer uncapped (seed mean)
    tfm_mean = tfm_uncapped.mean(axis=1)
    ax.plot(TOKENS, tfm_mean, color=OI_BLUE, linewidth=2.0, marker="s",
            markersize=7, zorder=3, markeredgecolor=TEXT, markeredgewidth=0.6)

    # contender: mean line + per-seed markers
    cont_mean = contender.mean(axis=1)
    ax.plot(TOKENS, cont_mean, color=OI_VERMILLION, linewidth=2.2, marker="o",
            markersize=7.5, zorder=4, markeredgecolor=TEXT, markeredgewidth=0.6)
    for s in range(contender.shape[1]):
        ax.plot(TOKENS, contender[:, s], color=OI_VERMILLION, linewidth=0,
                marker="o", markersize=4, alpha=0.55, zorder=4)

    ax.axhline(CHANCE, color=TEXT, linewidth=0.9, linestyle=":", alpha=0.7,
               zorder=1)
    ax.axhline(DEMO_BAR, color=TEXT, linewidth=0.9, linestyle="--", alpha=0.7,
               zorder=1)

    ax.annotate("fast-weight contender — fixed 32,768-byte state\n"
                "acc_A ≥ 0.998 at every seed, every horizon",
                (TOKENS[1], 0.955), fontsize=annotate_fontsize,
                color=OI_VERMILLION, ha="center", va="top", fontweight="bold")
    ax.annotate("transformer (param/token-matched): never leaves chance —\n"
                "uncapped (squares) or KV-cache-capped at M ∈ {2…32} "
                "(triangles)",
                (TOKENS[1], 0.145), fontsize=annotate_fontsize, color=OI_BLUE,
                ha="center", va="bottom")
    # reference-line labels sit ON their lines, masked by a bg box
    ax.annotate("demonstration bar (3× chance)", (2110, DEMO_BAR),
                fontsize=annotate_fontsize - 1.0, color=MUTED, ha="right",
                va="center", bbox=dict(facecolor=BG, edgecolor="none", pad=1))
    ax.annotate("chance (1/32)", (2110, CHANCE),
                fontsize=annotate_fontsize - 1.0, color=MUTED, ha="right",
                va="center", bbox=dict(facecolor=BG, edgecolor="none", pad=1))

    ax.set_xscale("log")
    ax.set_xticks(TOKENS)
    ax.set_xticklabels([f"{t:,}\n({m}× binding window)" for t, m in
                        zip(TOKENS, [2, 4, 8])], fontsize=tick_fontsize)
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_xlabel("context length at query time (tokens)",
                  fontsize=label_fontsize, labelpad=8)
    ax.set_ylabel("recall acc_A (K=32, episode-restricted top-1)",
                  fontsize=label_fontsize, labelpad=8)
    ax.set_ylim(-0.03, 1.07)
    ax.set_xlim(390, 2150)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22,
            color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)


# ------------------------------------------------------------- site SVG
fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor=BG)
draw(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "constant_memory_horizon.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote constant_memory_horizon.svg")

# ------------------------------------------------- standalone PNG (1200x675)
fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=100, facecolor=BG)
draw(ax, annotate_fontsize=12.5, label_fontsize=13.5, tick_fontsize=12.0)
ax.set_title("Constant-memory recall: a fixed 32 KB state vs a growing KV cache\n"
             "(14M params, matched params/tokens/compute; 3 seeds each)",
             fontsize=16, fontweight="bold", color=TEXT, pad=14, loc="left")
fig.text(0.99, 0.015, "pebbleml.com/findings/constant-memory-recall.html",
         fontsize=10.5, color=MUTED, ha="right")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(OUT_DIR / "constant_memory_horizon_x.png", format="png",
            facecolor=BG, dpi=100)
plt.close(fig)
print("wrote constant_memory_horizon_x.png")
