"""
Generate the two money figures for the native-composition-reads findings
page (NCR wave-1, K=8), plus a standalone 1200x675 PNG export for social
distribution.

Figures:
  ncr_depth_sweep_recovery.svg — median recovered_frac@0.9 (5 seeds) vs h
    for the three trained arms (ncr, fwm, loopedvec) on the log-spaced
    ladder 5..1021. h*=61 (pre-registered Axis-A comparison point) and the
    P2 pinned point h=29 are marked. Visualizes AXIS A = WIN / P2 = PASS.
  ncr_axis_b_timing.svg — median read wall-clock (s) vs h (log-log) for
    binexp (NCR's O(log h) repeated-squaring read), the three O(h) arms
    (fwm_recursive, loop, loopedvec_iter), and the disclosed O(1) control
    (cmlp_onehot). Visualizes AXIS B = WIN (20.9x @ h=1021).

Data source (raws, all md5-verified against the committed manifest before
use — see ASSERT_MANIFEST below):
  experiment-runs/2026-07-11_ncr_wave1/ncr_{ncr,fwm,loopedvec}_K8_s{0..4}.json
  experiment-runs/2026-07-11_ncr_wave1/wave1_verdict.json
  experiment-runs/2026-07-11_ncr_wave1/md5_manifest.txt

All recovery-curve numbers are RECOMPUTED from the per-seed cell JSONs
(median across 5 seeds at each ladder h), not copied from any prose
summary — this independently reproduces wave1_verdict.json's own
axis_a.arms.*.median_rec09_at_hstar (h=61) and predictions.p2.*.median_rec09
(h=29) bit-for-bit as a cross-check before plotting. Axis-B timing numbers
are read directly from wave1_verdict.json's axis_b.median_read_time_s.

Palette: Okabe-Ito (colorblind-safe; validated via dataviz skill's
validate_palette.js — ALL CHECKS PASS, contrast WARN discharged by direct
annotation labels + dark marker edges, matching this site's established
convention). Background matches the site (#FAF5E7).
"""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
RAW_DIR = REPO / "experiment-runs/2026-07-11_ncr_wave1"
MANIFEST = RAW_DIR / "md5_manifest.txt"
VERDICT = RAW_DIR / "wave1_verdict.json"

# ------------------------------------------------------------- md5 gate
manifest = {}
for line in MANIFEST.read_text().splitlines():
    if not line.strip():
        continue
    h, name = line.split(None, 1)
    manifest[name.strip()] = h

ARMS = ["ncr", "fwm", "loopedvec"]
SEEDS = [0, 1, 2, 3, 4]
cell_files = [f"ncr_{arm}_K8_s{s}.json" for arm in ARMS for s in SEEDS]
for fn in cell_files:
    fp = RAW_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(cell_files)}/{len(cell_files)} wave-1 cell JSONs against md5_manifest.txt")

with open(VERDICT) as f:
    verdict = json.load(f)

# ----------------------------------------------- recompute recovery curves
LADDER_H = [5, 13, 21, 29, 61, 125, 253, 509, 1021]


def load_ladder_rec09(arm):
    """Per-h, per-seed recovered_frac@0.9 on the ladder component, read
    straight from the archived cell JSONs (source of truth)."""
    out = {h: {} for h in LADDER_H}
    for s in SEEDS:
        with open(RAW_DIR / f"ncr_{arm}_K8_s{s}.json") as f:
            d = json.load(f)
        for p in d["eval"]["points"]:
            if p.get("component") == "ladder" and p["h"] in out:
                key = next(iter(p["reads"]))
                out[p["h"]][s] = p["reads"][key]["recovered_frac@0.9"]
    return out


curves = {arm: load_ladder_rec09(arm) for arm in ARMS}
medians = {
    arm: {h: statistics.median(curves[arm][h][s] for s in SEEDS) for h in LADDER_H}
    for arm in ARMS
}

# cross-check against the committed verdict of record before trusting the plot
assert abs(medians["ncr"][61] - verdict["axis_a"]["arms"]["ncr"]["median_rec09_at_hstar"]) < 1e-9
assert abs(medians["fwm"][61] - verdict["axis_a"]["arms"]["fwm"]["median_rec09_at_hstar"]) < 1e-9
assert abs(medians["loopedvec"][61] - verdict["axis_a"]["arms"]["loopedvec"]["median_rec09_at_hstar"]) < 1e-9
assert abs(medians["fwm"][29] - verdict["predictions"]["p2"]["fwm"]["median_rec09"]) < 1e-9
assert abs(medians["loopedvec"][29] - verdict["predictions"]["p2"]["loopedvec"]["median_rec09"]) < 1e-9
print("recomputed medians reproduce wave1_verdict.json bit-for-bit at h*=61 and the P2 pinned point h=29")

H_STAR = verdict["axis_a"]["h_star"]  # 61
P2_PIN = verdict["predictions"]["p2"]["pinned_point"]  # 29


def draw_recovery(ax, annotate_fontsize=9.5, label_fontsize=10.5, tick_fontsize=9.5):
    ax.set_facecolor(BG)

    style = {
        "ncr": dict(color=OI_VERMILLION, marker="o", label="NCR (O(log h) repeated-squaring read)"),
        "fwm": dict(color=OI_BLUE, marker="s", label="fast-weight memory baseline"),
        "loopedvec": dict(color=OI_SKY, marker="^", label="LoopedVec baseline"),
    }
    for arm in ARMS:
        ys = [medians[arm][h] for h in LADDER_H]
        st = style[arm]
        ax.plot(LADDER_H, ys, color=st["color"], linewidth=2.2, marker=st["marker"],
                 markersize=7, markeredgecolor=TEXT, markeredgewidth=0.6,
                 label=st["label"], zorder=4)
        # per-seed dots, lightly
        for s in SEEDS:
            seed_ys = [curves[arm][h][s] for h in LADDER_H]
            ax.plot(LADDER_H, seed_ys, color=st["color"], linewidth=0, marker=st["marker"],
                     markersize=3.2, alpha=0.35, zorder=3)

    ax.axvline(H_STAR, color=TEXT, linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)
    ax.annotate(f"h* = {H_STAR}\n(pre-registered O(h)-failure regime)",
                (H_STAR, 0.62), fontsize=annotate_fontsize - 0.5, color=TEXT,
                ha="left", va="center", xytext=(8, 0), textcoords="offset points")
    ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle=":", alpha=0.8, zorder=1)
    ax.annotate("P2 bar (below 0.5 at h=29)", (6, 0.5), fontsize=annotate_fontsize - 1.0,
                color=MUTED, ha="left", va="bottom")
    ax.axvline(P2_PIN, color=MUTED, linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)

    ax.annotate("NCR: median 1.0 at h*, 5/5 seeds hold",
                (95, 1.03), fontsize=annotate_fontsize, color=OI_VERMILLION,
                ha="left", va="bottom", fontweight="bold")
    ax.annotate("fwm: 0.158 at h*", (H_STAR, 0.158),
                fontsize=annotate_fontsize - 0.5, color=OI_BLUE, ha="left", va="top",
                xytext=(8, -6), textcoords="offset points")
    ax.annotate("loopedvec: 0.0\neverywhere on the ladder", (300, 0.06),
                fontsize=annotate_fontsize - 0.5, color=OI_SKY, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(LADDER_H)
    ax.get_xaxis().set_major_formatter(mticker.FixedFormatter([str(h) for h in LADDER_H]))
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xlabel("composition depth h (query hop count, log scale)", fontsize=label_fontsize, labelpad=8)
    ax.set_ylabel("median recovered_frac@0.9 (5 seeds)", fontsize=label_fontsize, labelpad=8)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xlim(4.3, 1200)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)
    ax.legend(loc="lower left", fontsize=annotate_fontsize - 0.5, frameon=False,
               bbox_to_anchor=(0.0, -0.02))


fig, ax = plt.subplots(figsize=(8.6, 5.3), facecolor=BG)
draw_recovery(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "ncr_depth_sweep_recovery.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_depth_sweep_recovery.svg")

# --------------------------------------------------------------- Axis B
axis_b = verdict["axis_b"]["median_read_time_s"]
COST_H = [61, 125, 253, 509, 1021, 1029, 16389, 131077, 1048581]

TIMING_STYLE = {
    "binexp": dict(color=OI_VERMILLION, marker="o", label="NCR: bin-exp O(log h) read", z=5, lw=2.4),
    "fwm_recursive": dict(color=OI_BLUE, marker="s", label="fwm: O(h) recursive read", z=4, lw=1.8),
    "loop": dict(color=OI_ORANGE, marker="D", label="loop: O(h) naive iteration", z=4, lw=1.8),
    "loopedvec_iter": dict(color=OI_SKY, marker="^", label="loopedvec: O(h) iterated map", z=4, lw=1.8),
    "cmlp_onehot": dict(color=MUTED, marker="x", label="C_MLP: O(1) one-hot (disclosed control)", z=3, lw=1.4),
}


def draw_timing(ax, annotate_fontsize=10.0, label_fontsize=11.0, tick_fontsize=10.0):
    ax.set_facecolor(BG)
    for key, st in TIMING_STYLE.items():
        ys = [axis_b[key][str(h)] for h in COST_H]
        dash = (0, (4, 2)) if key == "cmlp_onehot" else "-"
        ax.plot(COST_H, ys, color=st["color"], linewidth=st["lw"], marker=st["marker"],
                 markersize=6.5, markeredgecolor=TEXT, markeredgewidth=0.5,
                 linestyle=dash, label=st["label"], zorder=st["z"])

    ratio = verdict["axis_b"]["ncr_vs_best_oh_ratio_at_1021"]
    ax.annotate(f"{ratio:.1f}x faster at h=1,021", (1021, axis_b["binexp"]["1021"]),
                fontsize=annotate_fontsize, color=OI_VERMILLION, fontweight="bold",
                ha="left", va="bottom", xytext=(-70, 14), textcoords="offset points",
                bbox=dict(facecolor=BG, edgecolor="none", pad=2))

    gap_h = 1048581
    gap_ratio_lo = axis_b["fwm_recursive"][str(gap_h)] / axis_b["binexp"][str(gap_h)]
    gap_ratio_hi = axis_b["loopedvec_iter"][str(gap_h)] / axis_b["binexp"][str(gap_h)]
    ax.annotate(f"≈{gap_ratio_lo:,.0f}–{gap_ratio_hi:,.0f}x at h≈2^20",
                (gap_h, axis_b["binexp"][str(gap_h)]), fontsize=annotate_fontsize,
                color=OI_VERMILLION, fontweight="bold", ha="right", va="top",
                xytext=(-4, -10), textcoords="offset points",
                bbox=dict(facecolor=BG, edgecolor="none", pad=2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40, gap_h * 2.4)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xlabel("composition depth h (query hop count, log scale)", fontsize=label_fontsize, labelpad=8)
    ax.set_ylabel("median read wall-clock, seconds (log scale)", fontsize=label_fontsize, labelpad=8)
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)
    ax.legend(loc="upper left", fontsize=annotate_fontsize - 1.0, frameon=False)


fig, ax = plt.subplots(figsize=(8.6, 5.3), facecolor=BG)
draw_timing(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "ncr_axis_b_timing.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_axis_b_timing.svg")

# ------------------------------------------------- standalone PNG (1200x675)
fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=100, facecolor=BG)
draw_timing(ax, annotate_fontsize=13.0, label_fontsize=14.0, tick_fontsize=12.5)
ax.set_title("O(log h) reads: 20.9x faster at depth 1,021 — and it composes exactly (K=8)\n"
              "wave-1, 5 seeds/arm; win bar was 10x",
              fontsize=17, fontweight="bold", color=TEXT, pad=14, loc="left")
fig.text(0.99, 0.015, "pebbleml.com/findings/native-composition-reads.html",
          fontsize=10.5, color=MUTED, ha="right")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(OUT_DIR / "ncr_axis_b_timing_x.png", format="png", facecolor=BG, dpi=100)
plt.close(fig)
print("wrote ncr_axis_b_timing_x.png")
