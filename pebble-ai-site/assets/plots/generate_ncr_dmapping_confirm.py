"""
Generate the d-mapping-convention figure for the ncr-operator-bank findings
page (finding no. 16), the section reporting the NEXT_LEVER probe wave
(NOVEL_ARCH_WATERFALL.md Sec 11.4, NCR_NEXT_LEVER_DESIGN.md a8e848d):
does convention-confound Story S2 (the K-scaling wall is largely an
artifact of the d=2K state-dimension mapping, not absolute K) beat Story
S1 (tight spare is worse) and the pure absolute-K-cliff story, when
Probe A (d=K+1, "tight-spare") is tried at both K=16 and K=24?

Figure (ncr_dmapping_confirm.svg): two panels sharing 4 grouped columns
{K16 d=32 (best-of-budget: 4x), K16 d=17 (1x), K24 d=48 (1x), K24 d=25
(1x)} -- the "dead column vs alive column" contrast.
  Left  -- Gate-1 convergence rate (n_converged / 4 seeds) per column.
           d=32/d=48 use vermillion (the conventional d=2K mapping,
           dead-or-nearly-dead at K=24, only reaches 4/4 at K=16 after
           4x budget); d=K+1 uses green (tight-spare, 4/4 at BOTH K on
           1x budget alone).
  Right -- per-seed far-depth recovered@0.9 at each column's own h*, dot
           plot, one dot per seed that individually converged at gate 1
           (K24 d=48 has zero converged seeds -> no dots). d=K+1 reaches
           the far-depth HOLD band (>=0.9, dashed line) in 2/4 K=16 seeds
           on 1x budget alone; no d=32 cell at ANY budget (1x/2x/4x) or
           anneal-shape tried across Sec 11.2/11.3/11.4 ever produced a
           rec@h* above 0.0001.

This is NOT "K-scaling solved" -- K=24 d=25's far-depth is real but weak
and highly seed-variable (fronts 21-189, sweep_min <=0.051 in all 4 -- not
plotted here, see the page's caveats). The figure documents the CONFIRM
verdict only: Gate-1 convergence and (at K=16) far-depth composition are
both far more reachable under the tight-spare mapping than under d=2K at
any budget/anneal tried.

Data sources (raw cell JSONs, md5-verified against each directory's own
committed manifest before use; convergence counts and rec@h* values
cross-checked bit-for-bit against each cell's own eval.points, never
trusted from a harvest summary alone):
  experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/
    earlyln_K{16,24}_s{0-3}.json         (Probe A, d=K+1, 1x budget)
  experiment-runs/2026-07-12_ncr_nextlever_wave/budget4x/
    earlyln_K16_s{0-3}.json              (Q1, d=32, 4x budget -- the best
                                           Gate-1 rate d=32 ever reaches)
  experiment-runs/2026-07-12_ncr_nextlever_wave/md5_manifest.txt
  experiment-runs/2026-07-11_ncr_earlyln_scale/
    earlyln_K24_s{0-3}.json, harvest_verdict.json  (d=48, 1x -- 0/4
                                           CONVERGED at every budget/anneal
                                           tried for K=24, unchanged)
  experiment-runs/2026-07-11_ncr_earlyln_scale/md5_manifest.txt

Palette: Okabe-Ito (colorblind-safe). No in-figure title (caption lives
in the HTML). Background matches the site (#FAF5E7).

Also writes ncr_dmapping_confirm_x.png -- a standalone 1200x675 PNG (X's
summary_large_image ratio) with an in-figure title and a footer URL, for
the social-post attachment. Same underlying panels, larger fonts; not
used on the site itself.
"""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_VERMILLION = "#D55E00"
OI_GREEN = "#009E73"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent

NEXTLEVER_DIR = REPO / "experiment-runs/2026-07-12_ncr_nextlever_wave"
NEXTLEVER_MANIFEST = NEXTLEVER_DIR / "md5_manifest.txt"

SCALE_DIR = REPO / "experiment-runs/2026-07-11_ncr_earlyln_scale"
SCALE_MANIFEST = SCALE_DIR / "md5_manifest.txt"

# ------------------------------------------------------------- md5 gates
# NEXTLEVER_MANIFEST is BSD `md5 -r` style: "MD5 (./path) = hash"
nextlever_manifest = {}
for line in NEXTLEVER_MANIFEST.read_text().splitlines():
    m = re.match(r"MD5 \((.+)\) = ([0-9a-f]+)", line.strip())
    if m:
        p = m.group(1)
        p = p[2:] if p.startswith("./") else p
        nextlever_manifest[p] = m.group(2)

nextlever_files = (
    [f"dratio/earlyln_K{k}_s{s}.json" for k in (16, 24) for s in range(4)]
    + [f"budget4x/earlyln_K16_s{s}.json" for s in range(4)]
)
for fn in nextlever_files:
    fp = NEXTLEVER_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = nextlever_manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(nextlever_files)} files from 2026-07-12_ncr_nextlever_wave")

# SCALE_MANIFEST is `md5sum` style: "hash  path"
scale_manifest = {}
for line in SCALE_MANIFEST.read_text().splitlines():
    parts = line.strip().split(None, 1)
    if len(parts) == 2:
        scale_manifest[parts[1]] = parts[0]

scale_files = [f"earlyln_K24_s{s}.json" for s in range(4)] + ["harvest_verdict.json"]
for fn in scale_files:
    fp = SCALE_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = scale_manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(scale_files)} files from 2026-07-11_ncr_earlyln_scale")


def load(path):
    with open(path) as f:
        return json.load(f)


# h* is K-specific and pre-registered (NOVEL_ARCH_WATERFALL.md Sec 11.2/11.4):
# K=16 -> h*=125, K=24 -> h*=189. This is NOT "the largest ladder rung" --
# the ladder continues past h* (K16 also has 253/509/1021/2045 claim points)
# and using the max would silently read the wrong (always-0-past-front) point.
H_STAR = {16: 125, 24: 189}


def cell_gate1_and_h_star_rec(cell, k):
    """Reconstruct Gate-1 convergence + rec@0.9 at the cell's own pre-registered
    h_star directly from eval.points -- never trusted from a summary field alone."""
    front_h = cell["eval"]["failure_front_h"]
    indist_recs = []
    h_star_rec = None
    h_star = H_STAR[k]
    for p in cell["eval"]["points"]:
        if p["component"] == "train_support" and p["h"] in (1, 2, 3):
            indist_recs.append(p["reads"]["binexp"]["recovered_frac@0.9"])
        if p["component"] == "ladder" and p["h"] == h_star:
            h_star_rec = p["reads"]["binexp"]["recovered_frac@0.9"]
    assert h_star_rec is not None, f"K={k}: no ladder point at h_star={h_star} in raw eval.points"
    indist_min = min(indist_recs) if indist_recs else 0.0
    converged = indist_min >= 0.9
    return converged, h_star_rec, front_h


# ---- K16 d=17 (Probe A, tight-spare, 1x) -- and K24 d=25 (Probe A, 1x)
dratio_cells = {(k, s): load(NEXTLEVER_DIR / f"dratio/earlyln_K{k}_s{s}.json") for k in (16, 24) for s in range(4)}
# ---- K16 d=32 (Q1, 4x budget -- the best Gate-1 rate d=32 ever reaches)
budget4x_cells = {s: load(NEXTLEVER_DIR / f"budget4x/earlyln_K16_s{s}.json") for s in range(4)}
# ---- K24 d=48 (1x, registry reference -- 0/4 CONVERGED at every budget/anneal tried)
k24_d48_cells = {s: load(SCALE_DIR / f"earlyln_K24_s{s}.json") for s in range(4)}

groups = {}
for label, k, cells in [
    ("K=16\nd=32\n(4x budget)", 16, budget4x_cells),
    ("K=16\nd=17\n(1x)", 16, {s: dratio_cells[(16, s)] for s in range(4)}),
    ("K=24\nd=48\n(1x)", 24, k24_d48_cells),
    ("K=24\nd=25\n(1x)", 24, {s: dratio_cells[(24, s)] for s in range(4)}),
]:
    n_conv = 0
    far_pts = []
    for s, c in cells.items():
        converged, h_star_rec, front_h = cell_gate1_and_h_star_rec(c, k)
        if converged:
            n_conv += 1
            far_pts.append(h_star_rec)
    groups[label] = {"n_conv": n_conv, "rate": n_conv / 4.0, "far_pts": far_pts}
    print(f"{label!r}: {n_conv}/4 converged, far@h* = {sorted(round(v, 4) for v in far_pts)}")

# cross-check against the values recorded in the registry (NOVEL_ARCH_WATERFALL.md
# Sec 11.4 Table 1/2 and Sec 11.2's K=24 table) -- hard assertion, not a soft print
assert groups["K=16\nd=32\n(4x budget)"]["n_conv"] == 4
assert groups["K=16\nd=17\n(1x)"]["n_conv"] == 4
assert groups["K=24\nd=48\n(1x)"]["n_conv"] == 0
assert groups["K=24\nd=25\n(1x)"]["n_conv"] == 4
assert all(v < 0.001 for v in groups["K=16\nd=32\n(4x budget)"]["far_pts"])
assert max(groups["K=16\nd=17\n(1x)"]["far_pts"]) > 0.9
print("cross-checked convergence counts + far-depth values against registry Sec 11.4 / Sec 11.2")

# ---------------------------------------------------------------- figure
labels = list(groups.keys())
x = list(range(len(labels)))
# vermillion = the conventional d=2K mapping (dead-or-budget-hungry);
# green = tight-spare d=K+1 (alive at 1x budget, both K)
bar_colors = [OI_VERMILLION, OI_GREEN, OI_VERMILLION, OI_GREEN]


def draw_panels(ax1, ax2, label_fs=9.5, tick_fs=8.7, marker_fs=7.8, dot_size=70):
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # ---- left panel: convergence rate
    rates = [groups[l]["rate"] for l in labels]
    ax1.bar(x, rates, color=bar_colors, edgecolor=TEXT, linewidth=1.0, width=0.62, zorder=3)
    for xi, l in zip(x, labels):
        n = groups[l]["n_conv"]
        ax1.text(xi, groups[l]["rate"] + 0.035, f"{n}/4", ha="center", va="bottom",
                  fontsize=label_fs + 0.5, fontweight="bold", color=TEXT)
    ax1.axhline(0.75, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    ax1.text(3.62, 0.70, "CONVERGED-\nROBUST bar", fontsize=marker_fs, color=MUTED, va="top", ha="left")
    ax1.set_ylim(0, 1.14)
    ax1.set_xlim(-0.5, 3.95)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=tick_fs)
    ax1.set_ylabel("gate 1: convergence rate (n / 4 seeds)", fontsize=label_fs)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT, zorder=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)

    # ---- right panel: far-depth recovered@h* per converged seed
    rng_jitter = [-0.09, -0.03, 0.03, 0.09]
    for xi, l in zip(x, labels):
        pts = groups[l]["far_pts"]
        color = bar_colors[x[labels.index(l)]]
        for j, y in enumerate(sorted(pts)):
            ax2.scatter(xi + rng_jitter[j % len(rng_jitter)], y, color=color,
                        edgecolor=TEXT, linewidth=0.9, s=dot_size, zorder=4)
        if not pts:
            ax2.text(xi, 0.05, "no converged\nseeds", ha="center", va="bottom", fontsize=marker_fs,
                      color=MUTED, style="italic")
    ax2.axhline(0.9, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    ax2.text(3.62, 0.85, "HOLD bar", fontsize=marker_fs, color=MUTED, va="top", ha="left")
    ax2.set_ylim(-0.06, 1.12)
    ax2.set_xlim(-0.5, 3.95)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=tick_fs)
    ax2.set_ylabel("gate 2: recovered@0.9 at own h* (CONVERGED seeds only)", fontsize=label_fs)
    ax2.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT, zorder=0)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)


# ---- site figure (no in-figure title; caption lives in the HTML)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.9), facecolor=BG)
draw_panels(ax1, ax2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ncr_dmapping_confirm.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_dmapping_confirm.svg")

# ---- standalone PNG for social (1200x675, X's summary_large_image ratio)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=100, facecolor=BG)
draw_panels(ax1, ax2, label_fs=12.0, tick_fs=10.8, marker_fs=10.0, dot_size=110)
fig.suptitle("Tight-spare (d=K+1) converges where d=2K needed 4x budget or never converged",
             fontsize=15.5, fontweight="bold", color=TEXT, x=0.02, ha="left", y=0.995)
fig.text(0.99, 0.015, "pebbleml.com/findings/ncr-operator-bank.html", fontsize=10.5,
          color=MUTED, ha="right")
plt.tight_layout(rect=(0, 0.03, 1, 0.94))
plt.savefig(f"{OUT_DIR}/ncr_dmapping_confirm_x.png", format="png", facecolor=BG, dpi=100)
plt.close(fig)
print("wrote ncr_dmapping_confirm_x.png")
