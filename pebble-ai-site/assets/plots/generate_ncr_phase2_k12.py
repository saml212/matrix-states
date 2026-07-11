"""
Generate the K=12 cross-K recovery figure for the native-composition-reads
findings page (NCR wave-1+2, cross-K verdict of record).

Figure:
  ncr_phase2_k12_recovery.svg — median recovered_frac@0.9 (5 seeds) vs h
    for the three trained arms (ncr, fwm, loopedvec) on the K=12 ladder
    9..1533, with h*=57 (the K=12 pre-registered Axis-A comparison point,
    not itself a ladder rung — plotted from its own h_star eval component)
    marked. Visualizes AXIS A (K=12) = SEP-PARTIAL: NCR DEGRADED (median
    0.753, 2/5 seeds hold) vs both baselines FAIL (fwm 0.270, loopedvec
    0.0) at h*.

Data source (raws, all md5-verified against the committed manifest before
use — see the md5 gate below):
  experiment-runs/2026-07-11_ncr_phase2/ncr_{ncr,fwm,loopedvec}_K12_s{0..4}.json
  experiment-runs/2026-07-11_ncr_phase2/phase2_verdict.json
  experiment-runs/2026-07-11_ncr_phase2/md5_manifest.txt

All recovery-curve numbers are RECOMPUTED from the per-seed cell JSONs
(median across 5 seeds at each ladder h, and at h*=57 from the h_star
eval component), not copied from any prose summary — this independently
reproduces phase2_verdict.json's own axis_a.arms.*.median_rec09_at_hstar
bit-for-bit before plotting anything.

Palette: Okabe-Ito (colorblind-safe), same convention and same per-arm
colors as the K=8 figure (ncr_depth_sweep_recovery.svg) so the two
figures read as one system. Background matches the site (#FAF5E7).
"""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito — identical mapping to generate_ncr_wave1.py
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
RAW_DIR = REPO / "experiment-runs/2026-07-11_ncr_phase2"
MANIFEST = RAW_DIR / "md5_manifest.txt"
VERDICT = RAW_DIR / "phase2_verdict.json"

# ------------------------------------------------------------- md5 gate
manifest = {}
for line in MANIFEST.read_text().splitlines():
    if not line.strip():
        continue
    h, name = line.split(None, 1)
    manifest[name.strip()] = h

ARMS = ["ncr", "fwm", "loopedvec"]
SEEDS = [0, 1, 2, 3, 4]
cell_files = [f"ncr_{arm}_K12_s{s}.json" for arm in ARMS for s in SEEDS]
for fn in cell_files:
    fp = RAW_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(cell_files)}/{len(cell_files)} phase-2 (K=12) cell JSONs against md5_manifest.txt")

with open(VERDICT) as f:
    verdict = json.load(f)

# ----------------------------------------------- recompute recovery curves
LADDER_H = [9, 21, 45, 93, 189, 381, 765, 1533]
H_STAR = verdict["axis_a"]["h_star"]  # 57


def load_points(arm):
    """Per-h, per-seed recovered_frac@0.9: ladder rungs + the h_star point,
    read straight from the archived cell JSONs (source of truth)."""
    ladder = {h: {} for h in LADDER_H}
    hstar = {}
    for s in SEEDS:
        with open(RAW_DIR / f"ncr_{arm}_K12_s{s}.json") as f:
            d = json.load(f)
        for p in d["eval"]["points"]:
            key = next(iter(p["reads"]))
            if p.get("component") == "ladder" and p["h"] in ladder:
                ladder[p["h"]][s] = p["reads"][key]["recovered_frac@0.9"]
            elif p.get("component") == "h_star":
                hstar[s] = p["reads"][key]["recovered_frac@0.9"]
    return ladder, hstar


ladder_curves = {}
hstar_curves = {}
for arm in ARMS:
    ladder_curves[arm], hstar_curves[arm] = load_points(arm)

ladder_medians = {
    arm: {h: statistics.median(ladder_curves[arm][h][s] for s in SEEDS) for h in LADDER_H}
    for arm in ARMS
}
hstar_medians = {arm: statistics.median(hstar_curves[arm][s] for s in SEEDS) for arm in ARMS}

# cross-check against the committed verdict of record before trusting the plot
assert abs(hstar_medians["ncr"] - verdict["axis_a"]["arms"]["ncr"]["median_rec09_at_hstar"]) < 1e-9
assert abs(hstar_medians["fwm"] - verdict["axis_a"]["arms"]["fwm"]["median_rec09_at_hstar"]) < 1e-9
assert abs(hstar_medians["loopedvec"] - verdict["axis_a"]["arms"]["loopedvec"]["median_rec09_at_hstar"]) < 1e-9
p2 = verdict["predictions"]["p2"]
assert abs(ladder_medians["fwm"][45] - p2["fwm"]["median_rec09"]) < 1e-9
assert abs(ladder_medians["loopedvec"][45] - p2["loopedvec"]["median_rec09"]) < 1e-9
print("recomputed medians reproduce phase2_verdict.json bit-for-bit at h*=57 and the P2 pinned point h=45")

# per-seed h* values, for the figure's per-seed markers and the caption
ncr_hstar_per_seed = [hstar_curves["ncr"][s] for s in SEEDS]
print("per-seed NCR h*=57 recovered_frac@0.9:", ncr_hstar_per_seed)

P2_PIN = p2["pinned_point"]  # 45

# full x-grid for plotting: ladder rungs + h* inserted in sorted order
PLOT_H = sorted(set(LADDER_H) | {H_STAR})


def curve_with_hstar(arm):
    out = {}
    for h in PLOT_H:
        if h == H_STAR:
            out[h] = hstar_medians[arm]
        else:
            out[h] = ladder_medians[arm][h]
    return out


def seed_curve_with_hstar(arm, s):
    out = {}
    for h in PLOT_H:
        if h == H_STAR:
            out[h] = hstar_curves[arm][s]
        else:
            out[h] = ladder_curves[arm][h][s]
    return out


def draw_recovery_k12(ax, annotate_fontsize=9.5, label_fontsize=10.5, tick_fontsize=9.5):
    ax.set_facecolor(BG)

    style = {
        "ncr": dict(color=OI_VERMILLION, marker="o", label="NCR (O(log h) repeated-squaring read)"),
        "fwm": dict(color=OI_BLUE, marker="s", label="fast-weight memory baseline"),
        "loopedvec": dict(color=OI_SKY, marker="^", label="LoopedVec baseline"),
    }
    for arm in ARMS:
        med = curve_with_hstar(arm)
        ys = [med[h] for h in PLOT_H]
        st = style[arm]
        ax.plot(PLOT_H, ys, color=st["color"], linewidth=2.2, marker=st["marker"],
                 markersize=7, markeredgecolor=TEXT, markeredgewidth=0.6,
                 label=st["label"], zorder=4)
        for s in SEEDS:
            seed_ys = [seed_curve_with_hstar(arm, s)[h] for h in PLOT_H]
            ax.plot(PLOT_H, seed_ys, color=st["color"], linewidth=0, marker=st["marker"],
                     markersize=3.2, alpha=0.35, zorder=3)

    ax.axvline(H_STAR, color=TEXT, linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)
    ax.annotate(f"h* = {H_STAR}\n(K=12 pre-registered comparison point)",
                (H_STAR, 0.62), fontsize=annotate_fontsize - 0.5, color=TEXT,
                ha="left", va="center", xytext=(8, 0), textcoords="offset points")
    ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle=":", alpha=0.8, zorder=1)
    ax.annotate(f"P2 bar (below 0.5 at h={P2_PIN})", (10, 0.5), fontsize=annotate_fontsize - 1.0,
                color=MUTED, ha="left", va="bottom")
    ax.axvline(P2_PIN, color=MUTED, linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)

    ax.annotate("NCR: median 0.753 at h* (DEGRADED)\n2/5 seeds hold, 1 dead seed (s3)",
                (95, 0.80), fontsize=annotate_fontsize, color=OI_VERMILLION,
                ha="left", va="bottom", fontweight="bold")
    ax.annotate("fwm: 0.270 at h*", (H_STAR, hstar_medians["fwm"]),
                fontsize=annotate_fontsize - 0.5, color=OI_BLUE, ha="left", va="top",
                xytext=(8, -6), textcoords="offset points")
    ax.annotate("loopedvec: 0.0\neverywhere on the ladder", (300, 0.06),
                fontsize=annotate_fontsize - 0.5, color=OI_SKY, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(PLOT_H)
    ax.get_xaxis().set_major_formatter(mticker.FixedFormatter([str(h) for h in PLOT_H]))
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=tick_fontsize, labelrotation=0)
    ax.set_xlabel("composition depth h (query hop count, log scale), K=12", fontsize=label_fontsize, labelpad=8)
    ax.set_ylabel("median recovered_frac@0.9 (5 seeds)", fontsize=label_fontsize, labelpad=8)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xlim(7.5, 1800)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)
    ax.legend(loc="upper right", fontsize=annotate_fontsize - 0.5, frameon=False)


fig, ax = plt.subplots(figsize=(8.6, 5.3), facecolor=BG)
draw_recovery_k12(ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "ncr_phase2_k12_recovery.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_phase2_k12_recovery.svg")
