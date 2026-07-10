"""
Generate the two figures for the broken-lens findings page.

Figure 1 (broken_lens_dissociation.svg): the primary-vs-crosscheck readout
dissociation at the far (8x train) depth, all 62 sweep cells. Each point is
one cell: x = primary-lens recovered_frac@0.9 at D=64, y = crosscheck-lens
recovered_frac@0.9 at D=64, split by whether the cell reached train-support
convergence (final_loss <= 0.01) and by arm. The story: both lenses agree
(~0) on every non-converged cell; they contradict at 0-vs-1 magnitude
exactly on the converged contender cells.

Data source (verified md5 f26a769d5c263af224c91d39bd83710b):
  experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/
    crosscheck_lens_verdict_output.json   (flat_per_cell_table, 62 cells)

Figure 2 (broken_lens_endpoint.svg): the M-D3 endpoint under the discharged
pins (A6 decisive n_h=4 per the design's own calibration-pending clause; S5
extended to n=5 seeds), crosscheck lens decisional: per-group far-depth
recovered fraction for the contender vs the beta-range-restricted baseline
(Arm 2), with each group's own pre-registered 90%-of-own-ceiling bar.
Mechanical endpoint: INCONCLUSIVE (mixed pattern) — plotted as-is.

Data source (verified md5 a6b1a500d8a62bd1d8eb56618141796a):
  experiment-runs/2026-07-10_stage2_calibration/route_2p33/
    route_2p33_verdict_output.json   (verdict_discharged_pins_xcheck)

All numbers are READ from the archived verdict JSONs, not hardcoded.

Palette: Okabe-Ito (colorblind-safe). No in-figure titles (captions live
in the HTML). Background matches the site (#FAF5E7).
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

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
STAGE2 = REPO / "experiment-runs/2026-07-10_stage2_calibration"

with open(STAGE2 / "remetric_2p32/crosscheck_lens_verdict_output.json") as f:
    remetric = json.load(f)
with open(STAGE2 / "route_2p33/route_2p33_verdict_output.json") as f:
    route = json.load(f)

# Train-support convergence split. The grid is bimodal: converged cells
# read final_loss <= 0.012, non-converged >= 0.069 (registry sec 2.31's
# adjudication table) — any threshold in the gap gives the same split.
CONV_LOSS = 0.02

# ---------------------------------------------------------------- figure 1
cells = remetric["flat_per_cell_table"]
rng = np.random.default_rng(0)  # jitter only, cosmetic

groups = {
    # label, color, marker
    "arm3_conv": ("contender (Arm 3), converged", OI_VERMILLION, "o"),
    "arm3_nonconv": ("contender (Arm 3), not converged", OI_ORANGE, "o"),
    "arm2": ("baseline (Arm 2), any loss", OI_BLUE, "s"),
}
pts = {k: [] for k in groups}
for c in cells:
    x, y = c["primary_rf90_far64"], c["crosscheck_rf90_far64"]
    if c["arm"].startswith("arm2"):
        key = "arm2"
    elif c["final_loss"] <= CONV_LOSS:
        key = "arm3_conv"
    else:
        key = "arm3_nonconv"
    pts[key].append((x, y))

fig, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=BG)
ax.set_facecolor(BG)
ax.plot([0, 1], [0, 1], color=TEXT, linewidth=0.8, linestyle=":", alpha=0.5,
        zorder=1)
ax.annotate("lenses agree", (0.72, 0.68), fontsize=8.5, color=MUTED,
            rotation=38, ha="center")

for key, (label, color, marker) in groups.items():
    arr = np.array(pts[key])
    jx = np.clip(arr[:, 0] + rng.uniform(-0.014, 0.014, len(arr)), 0.0, 1.0)
    jy = np.clip(arr[:, 1] + rng.uniform(-0.014, 0.014, len(arr)), 0.0, 1.0)
    ax.scatter(jx, jy, s=52, color=color, marker=marker, alpha=0.8,
               edgecolor=TEXT, linewidth=0.6, zorder=3,
               label=f"{label} (n={len(arr)})")

n_dissoc = sum(1 for (x, y) in pts["arm3_conv"] if y - x >= 0.5)
ax.annotate(
    f"converged contender cells:\nprimary reads ≈0,\ncrosscheck reads ≈1\n"
    f"({n_dissoc} of {len(pts['arm3_conv'])} cells above the 0.5-gap line)",
    (0.10, 0.97), fontsize=9, color=OI_VERMILLION, ha="left", va="top",
    fontweight="bold")
ax.annotate("non-converged cells: both lenses ≈0,\nno disagreement anywhere",
            (0.97, 0.10), fontsize=9, color=MUTED, ha="right", va="bottom")

ax.set_xlabel("primary lens: recovered_frac@0.9 at 8× train depth (D=64)",
              fontsize=10, labelpad=8)
ax.set_ylabel("crosscheck lens: recovered_frac@0.9 at D=64",
              fontsize=10, labelpad=8)
ax.set_xlim(-0.06, 1.06)
ax.set_ylim(-0.06, 1.06)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)
legend = ax.legend(loc="center right", frameon=True, fontsize=8.5,
                   facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(OUT_DIR / "broken_lens_dissociation.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote broken_lens_dissociation.svg")

# ---------------------------------------------------------------- figure 2
per_group = route["verdict_discharged_pins_xcheck"]["per_group"]
order = ["S3", "S4", "A5", "S5", "A6"]

fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG)
ax.set_facecolor(BG)
xs = np.arange(len(order))
w = 0.34

cont = [per_group[g]["far_recovered_frac_90"] for g in order]
arm2 = [per_group[g]["arm2_far_recovered_frac_90"] for g in order]
bars_ = [per_group[g]["far_bar"] for g in order]

ax.bar(xs - w / 2, cont, width=w, color=OI_VERMILLION, edgecolor=TEXT,
       linewidth=1.0, alpha=0.9, zorder=3, label="contender (Arm 3)")
ax.bar(xs + w / 2, arm2, width=w, color=OI_BLUE, edgecolor=TEXT,
       linewidth=1.0, alpha=0.9, zorder=3,
       label="baseline (Arm 2, β∈[0,1])")

for x, b in zip(xs, bars_):
    ax.plot([x - w, x + 0.02], [b, b], color=TEXT, linewidth=1.6,
            linestyle="--", zorder=4)
for x, v in zip(xs, cont):
    ax.annotate(f"{v:.2f}", (x - w / 2, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9, color=TEXT,
                fontweight="bold")
for x, v in zip(xs, arm2):
    ax.annotate(f"{v:.2f}", (x + w / 2, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9, color=TEXT)

# separation verdict tags above each group pair
for x, g in zip(xs, order):
    sep = per_group[g]["separates_from_arm2"]
    gating = g in ("S5", "A6")
    tag = "separates" if sep else "no separation"
    if gating:
        tag += "\n(gating)"
    y_top = max(cont[list(order).index(g)], bars_[list(order).index(g)])
    ax.annotate(tag, (x, y_top + 0.075),
                fontsize=8, ha="center", va="bottom",
                color=OI_GREEN if sep else MUTED,
                fontweight="bold" if sep else "normal")

ax.annotate("dashes = each group's own bar: 90% of its D=8 ceiling",
            (-0.35, 1.235), fontsize=8.5, color=MUTED, ha="left")

ax.set_xticks(xs)
ax.set_xticklabels([f"{g} · n_h={per_group[g]['decisive_n_h']}"
                    for g in order], fontsize=9)
ax.set_ylabel("crosscheck recovered_frac@0.9 at D=64 (8× train)",
              fontsize=9.5, labelpad=8)
ax.set_ylim(0, 1.30)
ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)
legend = ax.legend(loc="center", bbox_to_anchor=(0.42, 0.55), frameon=True,
                   fontsize=8.5, facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(OUT_DIR / "broken_lens_endpoint.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote broken_lens_endpoint.svg")
