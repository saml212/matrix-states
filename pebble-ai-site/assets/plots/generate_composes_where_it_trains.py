"""Generate the two figures for the "composes-where-it-trains" findings page
(the S2.34/S2.35 diagnosis + promotion — the resolution chapter to
broken-lens.html's INCONCLUSIVE endpoint).

Both figures are computed DIRECTLY from committed raws:
  - test depths (D>=9): the per-cell sweep JSONs (`sweep_results/` +
    `route_2p33/`'s two S5-extension cells), via `stage2_harvest.py`'s own
    `m_d2_curve` / `_curve` helpers UNMODIFIED — the same harness
    diag_2p34.py used;
  - train-support depths (D<=8): the S2.34 box-side crosscheck M-D0 re-read
    (`diag_2p34/diag_2p34_md0_xcheck_output.json`, the artifact carrying the
    152/152 harness-fidelity teeth), aggregated over non-excluded rows.
No reimplementation, no hardcoded numbers.

Figure 1 (composes_a6_dissociation.svg): the A6 n_h staircase — crosscheck
recovery (train support + test depths) and restricted effective rank vs
depth, for n_h in {1, 2, 4}. The architecture dissociation: n_h=4 pins at
d_min(A6)=5 and reads crosscheck 1.00 flat from the shallowest evaluable
train depth out to 8x train (D=64); n_h=2 partially learns shallow
composition (0.60-0.75 at D=4) but never assembles the full rank-5
structure and decays with depth — the DEPTH-GRADED ceiling; n_h=1 never
gets off the floor.

Figure 2 (composes_s5_basin.svg): the S5 n_h=4 quintet split into the one
rank-deficient basin (seed 1) vs its four healthy siblings — same layout.
Seed 1 satisfies train loss (6.4e-3, unremarkable among the five) yet
never assembles the full rank-4 representation and decays INSIDE the train
support; final_loss alone cannot see this, M-D0+M-D2 jointly do.

Data sources (raw, read exactly as diag_2p34.py did):
  experiment-runs/2026-07-10_stage2_calibration/sweep_results/*.json
  experiment-runs/2026-07-10_stage2_calibration/route_2p33/S5__arm3_beta02__nh4__seed{3,4}.json
  experiment-runs/2026-07-10_stage2_calibration/diag_2p34/diag_2p34_md0_xcheck_output.json
Harness: matrix-thinking/capability_separation/stage2_harvest.py (m_d2_curve, _curve)

Palette: Okabe-Ito (colorblind-safe), matching every other figure on this
site. No in-figure titles (captions live in the HTML). Background matches
the site (#FAF5E7). Run with the repo .venv (needs numpy + matplotlib).
"""
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
STAGE2 = REPO / "experiment-runs/2026-07-10_stage2_calibration"

sys.path.insert(0, str(REPO / "matrix-thinking/capability_separation"))
import stage2_harvest as H

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


def load_all_cells():
    cells = {}
    for sub in ("sweep_results", "route_2p33"):
        for path in sorted(glob.glob(str(STAGE2 / sub / "*.json"))):
            fn = os.path.basename(path)
            if "harvest_report" in fn or "verdict" in fn or "s5ext_output" in fn:
                continue
            with open(path) as f:
                d = json.load(f)
            cid = d.get("cell_id") or os.path.splitext(fn)[0]
            cells[cid] = d
    return cells


def group_cells(cells, group, arm, n_h):
    return [c for c in cells.values()
            if c["group"] == group and c["arm"] == arm and c["n_h"] == n_h]


def md0_xcheck_curve(cell_list, md0_focus):
    """Aggregate the box-side crosscheck M-D0 per-depth profile (train
    support, D<=8) across a config's cells, skipping excluded rows.
    Sanity: every consumed row must carry the bit-identical-reproduction
    flag the 152/152 teeth asserted."""
    by_d = {}
    for c in cell_list:
        prof = md0_focus[c["cell_id"]]["profile"]
        for row in prof:
            if row.get("excluded"):
                continue
            assert row["reproduction_bit_identical"], c["cell_id"]
            by_d.setdefault(row["D"], []).append(row["crosscheck_rf90"])
    return {d: {"mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v)}
            for d, v in by_d.items()}


def combined_xcheck_curve(cell_list, md0_focus):
    """Train-support crosscheck (M-D0 re-read) + test-depth crosscheck
    (the committed D_test_results field, via stage2_harvest._curve)."""
    curve = md0_xcheck_curve(cell_list, md0_focus)
    curve.update(H._curve(cell_list, "crosscheck_recovered_frac_90"))
    return curve


def two_panel(curves, d_min, out_name, rec_legend_loc="center right",
              support_label_y=0.5):
    """curves: list of (label, color, marker, rank_curve, xcheck_curve)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)

    ax_rec, ax_rank = axes

    for label, color, marker, rank_curve, xcheck_curve in curves:
        ds = sorted(xcheck_curve.keys())
        means = [xcheck_curve[d]["mean"] for d in ds]
        stds = [xcheck_curve[d]["std"] for d in ds]
        ax_rec.plot(ds, means, color=color, marker=marker,
                    markersize=4.5, linewidth=1.8, label=label, zorder=3)
        lo = np.clip(np.array(means) - np.array(stds), 0, 1)
        hi = np.clip(np.array(means) + np.array(stds), 0, 1)
        ax_rec.fill_between(ds, lo, hi, color=color, alpha=0.15, zorder=2)

        ds_r = sorted(rank_curve.keys())
        rmeans = [rank_curve[d]["mean"] for d in ds_r]
        rstds = [rank_curve[d]["std"] for d in ds_r]
        ax_rank.plot(ds_r, rmeans, color=color, marker=marker,
                     markersize=4.5, linewidth=1.8, label=label, zorder=3)
        rlo = np.array(rmeans) - np.array(rstds)
        rhi = np.array(rmeans) + np.array(rstds)
        ax_rank.fill_between(ds_r, rlo, rhi, color=color, alpha=0.15, zorder=2)

    for ax in axes:
        ax.axvline(8.5, color=TEXT, linewidth=0.8, linestyle=":", alpha=0.6)
    ax_rec.text(8.15, support_label_y, "train support ends", rotation=90,
                fontsize=7.5, color=MUTED, ha="right", va="center")
    ax_rank.axhline(d_min, color=TEXT, linewidth=1.2, linestyle="--",
                    alpha=0.7, zorder=1)
    ax_rank.annotate(f"d_min = {d_min}", (0.98, d_min),
                     xycoords=("axes fraction", "data"),
                     fontsize=8.5, color=TEXT, ha="right", va="bottom",
                     fontweight="bold")

    ax_rec.set_ylabel("crosscheck recovered_frac@0.9 (mean ± 1σ)",
                      fontsize=9.5, labelpad=6)
    ax_rec.set_ylim(-0.05, 1.08)
    ax_rank.set_ylabel("restricted effective rank (mean ± 1σ)",
                       fontsize=9.5, labelpad=6)
    for ax, ticks in ((ax_rec, [2, 4, 8, 16, 32, 64]),
                      (ax_rank, [9, 16, 32, 64])):
        ax.set_xlabel("depth D (train support: D ≤ 8; far checkpoint: D = 64, 8× train)",
                      fontsize=9, labelpad=6)
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color(TEXT)

    leg1 = ax_rec.legend(loc=rec_legend_loc, frameon=True, fontsize=8.5,
                         facecolor=BG, edgecolor=TEXT)
    leg2 = ax_rank.legend(loc="lower left", frameon=True, fontsize=8.5,
                          facecolor=BG, edgecolor=TEXT)
    for leg in (leg1, leg2):
        leg.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, format="svg", facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_name}")


cells = load_all_cells()
assert len(cells) == 64, f"expected 64 cells, got {len(cells)}"

with open(STAGE2 / "diag_2p34/diag_2p34_md0_xcheck_output.json") as f:
    md0 = json.load(f)
assert md0["teeth"] == {"n_rows_evaluated": 152, "n_bit_identical": 152}
md0_focus = md0["focus_cells"]

# ------------------------------------------------------------- figure 1: A6
a6_nh1 = group_cells(cells, "A6", "arm3_beta02", 1)
a6_nh2 = group_cells(cells, "A6", "arm3_beta02", 2)
a6_nh4 = group_cells(cells, "A6", "arm3_beta02", 4)
assert len(a6_nh1) == 3 and len(a6_nh2) == 5 and len(a6_nh4) == 3

curves_a6 = [
    ("n_h=1 (3/3 fail, loss 0.21-0.40)", OI_SKY, "^",
     H.m_d2_curve(a6_nh1), combined_xcheck_curve(a6_nh1, md0_focus)),
    ("n_h=2 (5/5 fail, loss 0.07-0.29)", OI_ORANGE, "s",
     H.m_d2_curve(a6_nh2), combined_xcheck_curve(a6_nh2, md0_focus)),
    ("n_h=4 (3/3 converge, loss 3e-5-8e-5)", OI_VERMILLION, "o",
     H.m_d2_curve(a6_nh4), combined_xcheck_curve(a6_nh4, md0_focus)),
]
two_panel(curves_a6, d_min=5, out_name="composes_a6_dissociation.svg")

# ------------------------------------------------------------- figure 2: S5
s5_nh4_all = group_cells(cells, "S5", "arm3_beta02", 4)
assert len(s5_nh4_all) == 5
s5_seed1 = [c for c in s5_nh4_all if c["seed"] == 1]
s5_healthy = [c for c in s5_nh4_all if c["seed"] != 1]
assert len(s5_seed1) == 1 and len(s5_healthy) == 4

curves_s5 = [
    ("seeds {0,2,3,4} (healthy, loss 1e-3-1.2e-2)", OI_GREEN, "o",
     H.m_d2_curve(s5_healthy), combined_xcheck_curve(s5_healthy, md0_focus)),
    ("seed 1 (rank-deficient basin, loss 6.4e-3)", OI_PURPLE, "D",
     H.m_d2_curve(s5_seed1), combined_xcheck_curve(s5_seed1, md0_focus)),
]
two_panel(curves_s5, d_min=4, out_name="composes_s5_basin.svg",
          rec_legend_loc="lower center", support_label_y=0.15)
