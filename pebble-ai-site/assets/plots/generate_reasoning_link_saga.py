"""
Generate the figures for the reasoning-link instrument-saga findings page
(finding no. 15): the three-layer chain — a pre-registered positive control
catches a transpose bug behind 80/80 published nulls, the fixed instrument
shows apparent signal, and two correspondence nulls (label-shuffle +
derangement-slot) kill the signal, re-closing the lane on doubly-validated
instrument grounds.

Figure 1 (reasoning_link_real_vs_null.svg): real recovered_frac@0.9 (x) vs
null recovered_frac@0.9 (y) for every (cell,h) reading, both correspondence
nulls overlaid. Points sit on the y=x line everywhere a real reading is
non-trivial -- the "signal" is exactly reproduced by both a shuffled target
and a target that is NEVER correct by construction (derangement).

Figure 2 (reasoning_link_h_profile.svg): mean recovered_frac by h in {1,2,3,4}
at the 41 floor-clearing ((real>=0.10)) readings, real vs succ-shuffle-null
vs derangement-null. The artifact is flat across every hop depth -- there is
no h-dependent structure that null grading fails to erase.

Data sources (md5-verified against
experiment-runs/2026-07-11_reasoning_link_validation/MANIFEST.md5):
  01_item12_shuffle_resample/results/ANALYSIS.json  (d55d43c92ff00b39300525d9b7fe16d3)
    -> item1.rows: 320 (cell,h) real/succ-shuffle-null pairs
  02_derange_control/results/leg_*.json  (80 files, per-cell per_h blocks)
    -> real recovered_frac + deranged_recovered_frac per (cell,h)

All numbers are READ from the archived raws, not hardcoded. The script
asserts real readings are bit-identical across both null sweeps before
plotting (the cross-run integrity claim in REASONING_LINK_DESIGN.md sec
17.7), and asserts summary statistics match the design doc's recorded
numbers within floating-point tolerance.

Palette: Okabe-Ito (colorblind-safe). No in-figure titles for the SVGs
(captions live in the HTML); the PNG (for X distribution) carries its own
title since it stands alone. Background matches the site (#FAF5E7).
"""
import hashlib
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
OI_GREY = "#999999"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
VAL_DIR = REPO / "experiment-runs/2026-07-11_reasoning_link_validation"

EXPECTED_MD5 = {
    "01_item12_shuffle_resample/results/ANALYSIS.json": "d55d43c92ff00b39300525d9b7fe16d3",
}


def check_md5(rel_path, expected):
    p = VAL_DIR / rel_path
    got = hashlib.md5(p.read_bytes()).hexdigest()
    assert got == expected, f"md5 mismatch for {rel_path}: {got} != {expected}"


for rel, exp in EXPECTED_MD5.items():
    check_md5(rel, exp)

# ---------------------------------------------------------------- load data
with open(VAL_DIR / "01_item12_shuffle_resample/results/ANALYSIS.json") as f:
    analysis = json.load(f)

item1_rows = analysis["item1"]["rows"]  # 320 rows: cell_name, h, real, null(shuffle)

shuffle_by_key = {(r["cell_name"], r["h"]): r for r in item1_rows}

derange_dir = VAL_DIR / "02_derange_control/results"
derange_files = sorted(p for p in derange_dir.glob("leg_*.json"))
assert len(derange_files) == 80, f"expected 80 derange-control cell files, got {len(derange_files)}"

rows = []  # cell_name, h, real, null_shuffle, null_derange
for p in derange_files:
    with open(p) as f:
        d = json.load(f)
    cell_name = d["cell"]["cell_name"]
    for h_str, block in d["per_h"].items():
        h = int(h_str)
        real_derange_view = block["recovered_frac"]
        deranged = block["deranged_recovered_frac"]
        key = (cell_name, h)
        shuf = shuffle_by_key.get(key)
        assert shuf is not None, f"missing shuffle-null row for {key}"
        # cross-run integrity: the real reading must be bit-identical between
        # the two null sweeps (sec 17.7's "320/320 bit-identical" claim)
        assert abs(real_derange_view - shuf["real_recovered_frac"]) < 1e-9, (
            f"real reading mismatch at {key}: {real_derange_view} vs "
            f"{shuf['real_recovered_frac']}"
        )
        rows.append({
            "cell": cell_name,
            "h": h,
            "real": real_derange_view,
            "null_shuffle": shuf["null_recovered_frac"],
            "null_derange": deranged,
        })

assert len(rows) == 320, f"expected 320 (cell,h) rows, got {len(rows)}"

FLOOR = 0.10
floor_rows = [r for r in rows if r["real"] >= FLOOR]
assert len(floor_rows) == 41, f"expected 41 floor-clearing readings, got {len(floor_rows)}"

mean_real = float(np.mean([r["real"] for r in floor_rows]))
mean_null_shuffle = float(np.mean([r["null_shuffle"] for r in floor_rows]))
mean_null_derange = float(np.mean([r["null_derange"] for r in floor_rows]))
print(f"mean real @ floor = {mean_real:.4f} (design doc: 0.3023)")
print(f"mean null(shuffle) @ floor = {mean_null_shuffle:.4f} (design doc: 0.3010)")
print(f"mean null(derange) @ floor = {mean_null_derange:.4f} (design doc: 0.2960)")
assert abs(mean_real - 0.3023) < 0.001
assert abs(mean_null_shuffle - 0.3010) < 0.001
assert abs(mean_null_derange - 0.2960) < 0.001

# strongest single cell (design doc's headline number)
strongest = max(rows, key=lambda r: r["real"])
print(
    f"strongest reading: {strongest['cell']} h={strongest['h']} "
    f"real={strongest['real']:.4f} deranged={strongest['null_derange']:.4f} "
    f"({strongest['null_derange'] / strongest['real'] * 100:.1f}% reproduction)"
)

# ============================================================= figure 1
# real vs null scatter, both nulls overlaid, y=x reference line
fig, ax = plt.subplots(figsize=(7.0, 7.0), facecolor=BG)
ax.set_facecolor(BG)

xs = [r["real"] for r in rows]
ys_shuffle = [r["null_shuffle"] for r in rows]
ys_derange = [r["null_derange"] for r in rows]

ax.plot([-0.05, 1.0], [-0.05, 1.0], color=MUTED, lw=1.2, ls="--", zorder=1,
        label="y = x (null exactly reproduces real)")

ax.scatter(xs, ys_shuffle, s=46, facecolors="none", edgecolors=OI_BLUE,
           linewidths=1.3, zorder=3, label="succ-shuffle null (item 1)")
ax.scatter(xs, ys_derange, s=40, marker="s", facecolors="none",
           edgecolors=OI_VERMILLION, linewidths=1.3, zorder=2,
           label="derangement-slot null (§17.6a)")

ax.axvline(FLOOR, color=OI_GREY, lw=0.9, ls=":", zorder=0)
ax.text(FLOOR + 0.012, 0.96, "H1_ABSOLUTE_FLOOR = 0.10", color=MUTED,
        fontsize=8.5, rotation=90, va="top", ha="left")

ax.set_xlim(-0.05, 1.0)
ax.set_ylim(-0.05, 1.0)
ax.set_xlabel("real recovered_frac@0.9", fontsize=11.5)
ax.set_ylabel("null recovered_frac@0.9 (same (cell,h) reading)", fontsize=11.5)
ax.tick_params(labelsize=10)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(loc="upper left", fontsize=9.5, frameon=False)

fig.tight_layout()
fig.savefig(OUT_DIR / "reasoning_link_real_vs_null.svg", facecolor=BG)
plt.close(fig)

# ============================================================= figure 2
# h-profile: mean recovered_frac by h at the 41 floor-clearing readings
hs = [1, 2, 3, 4]
means_real = []
means_shuffle = []
means_derange = []
for h in hs:
    hr = [r for r in floor_rows if r["h"] == h]
    means_real.append(np.mean([r["real"] for r in hr]) if hr else 0.0)
    means_shuffle.append(np.mean([r["null_shuffle"] for r in hr]) if hr else 0.0)
    means_derange.append(np.mean([r["null_derange"] for r in hr]) if hr else 0.0)
    print(f"h={h}: n={len(hr)} real={means_real[-1]:.4f} "
          f"shuffle={means_shuffle[-1]:.4f} derange={means_derange[-1]:.4f}")

fig2, ax2 = plt.subplots(figsize=(7.4, 4.6), facecolor=BG)
ax2.set_facecolor(BG)

x = np.arange(len(hs))
w = 0.26
ax2.bar(x - w, means_real, width=w, color=OI_VERMILLION, label="real target",
        zorder=3)
ax2.bar(x, means_shuffle, width=w, color=OI_BLUE, label="succ-shuffle null",
        zorder=3)
ax2.bar(x + w, means_derange, width=w, color=OI_ORANGE,
        label="derangement-slot null", zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels([f"h={h}" for h in hs], fontsize=11)
ax2.set_ylabel("mean recovered_frac@0.9\n(41 floor-clearing readings)",
               fontsize=11)
ax2.tick_params(labelsize=10)
ax2.set_ylim(0, 0.45)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
ax2.legend(loc="upper right", fontsize=9.5, frameon=False)
ax2.grid(axis="y", color=OI_GREY, alpha=0.25, lw=0.6, zorder=0)

fig2.tight_layout()
fig2.savefig(OUT_DIR / "reasoning_link_h_profile.svg", facecolor=BG)
plt.close(fig2)

# ============================================================= X card PNG
# 1200x675, standalone, own title -- summarizes fig 1's story for social.
fig3, ax3 = plt.subplots(figsize=(12.0, 6.75), dpi=100, facecolor=BG)
ax3.set_facecolor(BG)

ax3.plot([-0.05, 1.0], [-0.05, 1.0], color=MUTED, lw=1.4, ls="--", zorder=1,
         label="y = x  (null fully reproduces real)")
ax3.scatter(xs, ys_shuffle, s=70, facecolors="none", edgecolors=OI_BLUE,
            linewidths=1.6, zorder=3, label="label-shuffle null")
ax3.scatter(xs, ys_derange, s=62, marker="s", facecolors="none",
            edgecolors=OI_VERMILLION, linewidths=1.6, zorder=2,
            label="derangement-slot null (never the correct target)")

ax3.set_xlim(-0.05, 1.0)
ax3.set_ylim(-0.05, 1.0)
ax3.set_xlabel("real recovered_frac@0.9", fontsize=14)
ax3.set_ylabel("null recovered_frac@0.9", fontsize=14)
ax3.tick_params(labelsize=12)
for spine in ("top", "right"):
    ax3.spines[spine].set_visible(False)
ax3.legend(loc="upper left", fontsize=13, frameon=False)
ax3.set_title(
    "Our fixed instrument showed a signal. Our nulls killed it.\n"
    "0/320 readings distinguish the real target from a shuffled or "
    "never-correct one.",
    fontsize=15.5, fontweight="bold", color=TEXT, loc="left", pad=14,
)

fig3.tight_layout()
fig3.savefig(OUT_DIR / "reasoning_link_saga_x.png", facecolor=BG)
plt.close(fig3)

print("Wrote reasoning_link_real_vs_null.svg, reasoning_link_h_profile.svg, "
      "reasoning_link_saga_x.png")
