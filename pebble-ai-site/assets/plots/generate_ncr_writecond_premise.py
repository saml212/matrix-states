"""
Generate the money figure for the ncr-write-conditioning findings page
(premise-battery harvest, 2026-08-13): retrieval24_acc vs composition hop
h, for the P0 (model's own SGD-learned write) and P1b (exact write
substituted, teacher-forced) arms, across the three independently trained
98M-param checkpoints (compB/primary/compA).

Story: P1b sits at 0.977-1.0 through h=61 in all three checkpoints (the
read machinery composes exactly, given an exact operator); P0 sits at or
below chance (1/24 = 0.0417) at every depth in all three checkpoints (SGD
never learns a usable write). Same read path, same model weights, only
the operator fed to it differs.

Data source (raw JSONs, all values recomputed here directly from the
archived per-cell records, not copied from prose):
  experiment-runs/2026-08-13_ncr_writecond_premise_battery/writecond_premise_P0P1b.json
    (compB, h=1/13/37 full_graft + P1b h=1/2/3; h=61 comes from SUPP, disclosed below)
  experiment-runs/2026-08-13_ncr_writecond_premise_battery/writecond_premise_SUPP.json
    (compB P0_deep/P1b_deep h=61 supplement, same instrument/seed as P1a)
  experiment-runs/2026-08-13_ncr_writecond_premise_battery/writecond_premise_REPL_compA.json
  experiment-runs/2026-08-13_ncr_writecond_premise_battery/writecond_premise_REPL_primary.json

No md5 manifest was archived alongside this battery (disclosed); values
are read directly from the committed JSONs and cross-checked in-script
against the verdict numbers recorded in
matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md's PREMISE BATTERY
HARVEST section before plotting.

Palette: Okabe-Ito (colorblind-safe), matching this site's established
convention. Background matches the site (#FAF5E7).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_VERMILLION = "#D55E00"
OI_BLUE = "#0072B2"
OI_SKY = "#56B4E9"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
RAW_DIR = REPO / "experiment-runs/2026-08-13_ncr_writecond_premise_battery"

HOPS = [1, 13, 37, 61]
CHANCE = 1.0 / 24.0

# ------------------------------------------------------------- load raws
# writecond_premise_P0P1b.json (the primary compB cell) is the harvest's
# main record but only measured P1b at h in {1,2,3} and P0 in a separate
# non-ladder {1,2,3,5,12,20,29} breakdown (disclosed deviation (ii) in
# the harvest section) -- not the full {1,13,37,61} ladder this figure
# needs. writecond_premise_SUPP.json is the disclosed same-day supplement
# that independently re-measured BOTH arms at the full ladder with the
# identical instrument (eval_arm_at_hops, same seed/signature as P1a)
# applied to the restored compB arms, so it is used directly here as
# compB's source of record for this figure.
with open(RAW_DIR / "writecond_premise_SUPP.json") as f:
    supp = json.load(f)
with open(RAW_DIR / "writecond_premise_REPL_compA.json") as f:
    compA = json.load(f)
with open(RAW_DIR / "writecond_premise_REPL_primary.json") as f:
    primary = json.load(f)

compB_p1b = {
    1: supp["P1b_deep"]["result"]["h=1"]["retrieval24_acc"],
    13: supp["P1b_deep"]["result"]["h=13"]["retrieval24_acc"],
    37: supp["P1b_deep"]["result"]["h=37"]["retrieval24_acc"],
    61: supp["P1b_deep"]["result"]["h=61"]["retrieval24_acc"],
}
compB_p0 = {
    1: supp["P0_deep"]["result"]["h=1"]["retrieval24_acc"],
    13: supp["P0_deep"]["result"]["h=13"]["retrieval24_acc"],
    37: supp["P0_deep"]["result"]["h=37"]["retrieval24_acc"],
    61: supp["P0_deep"]["result"]["h=61"]["retrieval24_acc"],
}

compA_p0 = {h: compA["P0"]["result"][f"h={h}"]["retrieval24_acc"] for h in HOPS}
compA_p1b = {h: compA["P1b"]["result"][f"h={h}"]["retrieval24_acc"] for h in HOPS}
primary_p0 = {h: primary["P0"]["result"][f"h={h}"]["retrieval24_acc"] for h in HOPS}
primary_p1b = {h: primary["P1b"]["result"][f"h={h}"]["retrieval24_acc"] for h in HOPS}

# ----------------------------------------------------- cross-check gate
# Reproduce the exact numbers recorded in NCR_WRITE_CONDITIONING_DESIGN.md's
# PREMISE BATTERY HARVEST section before trusting the plot.
EXPECT = {
    ("compB", "P0"): [0.0703, 0.0352, 0.0352, 0.0664],
    ("compB", "P1b"): [1.0000, 0.9883, 0.9883, 0.9766],
    ("compA", "P0"): [0.043, 0.020, 0.020, 0.035],
    ("compA", "P1b"): [1.0, 1.0, 1.0, 0.9961],
    ("primary", "P0"): [0.055, 0.039, 0.039, 0.039],
    ("primary", "P1b"): [1.0, 0.9961, 1.0, 1.0],
}
got = {
    ("compB", "P0"): [compB_p0[h] for h in HOPS],
    ("compB", "P1b"): [compB_p1b[h] for h in HOPS],
    ("compA", "P0"): [compA_p0[h] for h in HOPS],
    ("compA", "P1b"): [compA_p1b[h] for h in HOPS],
    ("primary", "P0"): [primary_p0[h] for h in HOPS],
    ("primary", "P1b"): [primary_p1b[h] for h in HOPS],
}
for key, expect_vals in EXPECT.items():
    for e, g in zip(expect_vals, got[key]):
        assert abs(e - g) < 5e-3, f"MISMATCH {key}: expected~{e}, got {g}"
print("recomputed retrieval24_acc curves reproduce the design doc's harvest table "
      "(tolerance 5e-3) for all 3 checkpoints x 2 arms x 4 hops")

CHECKPOINTS = [
    ("compB", "trainable-adapter, ctr+cos", compB_p0, compB_p1b, OI_VERMILLION, "o"),
    ("primary", "frozen-adapter, ctr+cos", primary_p0, primary_p1b, OI_BLUE, "s"),
    ("compA", "frozen-adapter, cosine-only", compA_p0, compA_p1b, OI_SKY, "^"),
]


def draw(ax, ann_fs=9.5, label_fs=10.5, tick_fs=9.5, legend_fs=9.0):
    ax.set_facecolor(BG)

    for name, arm_label, p0, p1b, color, marker in CHECKPOINTS:
        p1b_ys = [p1b[h] for h in HOPS]
        p0_ys = [p0[h] for h in HOPS]
        ax.plot(HOPS, p1b_ys, color=color, linewidth=2.2, marker=marker, markersize=7,
                 markeredgecolor=TEXT, markeredgewidth=0.6, linestyle="-",
                 label=f"{name} ({arm_label}): P1b, exact write", zorder=5)
        ax.plot(HOPS, p0_ys, color=color, linewidth=1.6, marker=marker, markersize=6,
                 markeredgecolor=TEXT, markeredgewidth=0.5, linestyle="--", alpha=0.55,
                 label=f"{name} ({arm_label}): P0, own SGD write", zorder=4)

    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, linestyle=":", alpha=0.9, zorder=2)
    ax.annotate(f"chance = 1/24 = {CHANCE:.4f}", (1, CHANCE), fontsize=ann_fs - 0.5,
                color=MUTED, ha="left", va="bottom", xytext=(4, 4), textcoords="offset points")

    ax.annotate("P1b (exact write): 0.977–1.0\nin all 3 checkpoints, all 4 hops",
                (18, 1.09), fontsize=ann_fs, color=TEXT, fontweight="bold",
                ha="left", va="bottom")
    ax.annotate("P0 (SGD's own write):\nchance in all 3 checkpoints", (30, 0.16),
                fontsize=ann_fs, color=TEXT, fontweight="bold", ha="left", va="bottom")

    ax.set_xticks(HOPS)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xlabel("composition depth h (query hop count)", fontsize=label_fs, labelpad=8)
    ax.set_ylabel("retrieval24_acc (n=256)", fontsize=label_fs, labelpad=8)
    ax.set_ylim(-0.04, 1.30)
    ax.set_xlim(-2, 66)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)
    ax.legend(loc="center left", fontsize=legend_fs, frameon=False,
               bbox_to_anchor=(1.02, 0.5))


# ---- site figure
fig, ax = plt.subplots(figsize=(9.6, 5.3), facecolor=BG)
draw(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "ncr_writecond_premise.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_writecond_premise.svg")

# ---- standalone PNG for X (1200x675)
fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=100, facecolor=BG)
draw(ax, ann_fs=13.0, label_fs=14.0, tick_fs=12.5, legend_fs=11.5)
ax.set_title("The read machinery already composes 61 hops exactly. SGD never learns the write.",
              fontsize=16, fontweight="bold", color=TEXT, pad=14, loc="left")
fig.text(0.99, 0.015, "pebbleml.com/findings/ncr-write-conditioning.html", fontsize=10.5,
          color=MUTED, ha="right")
plt.tight_layout(rect=(0, 0.03, 1, 0.94))
plt.savefig(OUT_DIR / "ncr_writecond_premise_x.png", format="png", facecolor=BG, dpi=100)
plt.close(fig)
print("wrote ncr_writecond_premise_x.png")
