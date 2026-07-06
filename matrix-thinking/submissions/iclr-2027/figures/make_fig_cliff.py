#!/usr/bin/env python3
"""Fig. cliff: the located capacity-cliff transition (candidate (d), d_state=64).

Standalone companion to `make_figures_v2.py` (style-matched, same conventions
-- fonts, colorblind-safe palette, savefig helper, "no fabricated numbers"
discipline) for the 2026-07-06 cliff-localization wave. Kept as its own
script rather than folded into `make_figures_v2.py`'s REQUIRED_PATHS list
because it reads a different archive tree
(`experiment-runs/2026-07-06_keyanchor_cliff/`) and is a strict *addition* to
the existing 11-figure set (fig11_capacity_curve.py's 3-point plot is
untouched) -- this figure supersedes fig11 as the headline capacity-law
figure once the paper is edited to cite it (NARRATIVE.md round 6, PAPER_
SPRINT_PLAN.md fig:cliff row).

Run from anywhere with a venv containing numpy/scipy/matplotlib:
    /path/to/figvenv/bin/python matrix-thinking/submissions/iclr-2027/figures/make_fig_cliff.py

Data provenance (every number traced to an archived file, never hand-typed):
  - 4 NEW points (K=34/38/42/46): per-seed h4 read directly from
    `experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/
    wavekeyanchor-cliff/wkeyanchor-k48_rdx_K{K}_armd_s*_geo3n20_anchor_
    learned_dprobe_rev7.json`, field
    `checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]`.
  - 2 ARCHIVED anchor points with their own per-seed arrays:
      K=32 (K/d=0.50): seeds {10,11,12} = 0.6741/0.7125/0.6141
        (KEY_ANCHORING_DESIGN.md sec 10.13.1 / sec 3667), archived at
        experiment-runs/2026-07-06_keyanchor_mech/wavekeyanchor-mech/
        wkeyanchor-mech_rdx_K32_armd_s1{0,1,2}_geo3n20_anchor_learned_
        dprobe_rev7.json.
      K=48 (K/d=0.75): seeds {30,31,32} = 0.02295/0.02287/0.01872
        (KEY_ANCHORING_DESIGN.md sec 11.12.1), archived at
        experiment-runs/2026-07-07_keyanchor_k48/wavekeyanchor-k48/
        wkeyanchor-k48_rdx_K48_armd_s3{0,1,2}_geo3n20_anchor_learned_
        dprobe_rev7.json.
  - K=16 (K/d=0.25): fixed saturation anchor, h4=1.000, no per-seed archive
    at this citation depth (KEY_ANCHORING_DESIGN.md sec 12.9's own table) --
    plotted as a single point, no scatter.
  - Sigmoid fit + bootstrap CI: `experiment-runs/2026-07-06_keyanchor_cliff/
    fit_cliff_curve_results.json` (`sigmoid_fit`, `bootstrap_ci`). Numbers
    verified against KEY_ANCHORING_DESIGN.md sec 12.9 before writing this
    script: x0=0.5455 (CI [0.5385,0.5513]), w=0.0597 (CI [0.0557,0.0642]),
    L=1.003.

Where NARRATIVE.md/KEY_ANCHORING_DESIGN.md quote a number, it is used ONLY
as a post-hoc sanity check (assertions below) -- never as a value fed
directly into the plot; every plotted number is read from a JSON on disk.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(FIG_DIR)                        # iclr-2027/
SUBMISSIONS_DIR = os.path.dirname(PAPER_DIR)                # submissions/
MT_DIR = os.path.dirname(SUBMISSIONS_DIR)                   # matrix-thinking/
REPO_ROOT = os.path.dirname(MT_DIR)                         # repo root

EXPRUNS = os.path.join(REPO_ROOT, "experiment-runs")

CLIFF_DIR = os.path.join(EXPRUNS, "2026-07-06_keyanchor_cliff")
CLIFF_RESULTS = os.path.join(CLIFF_DIR, "fit_cliff_curve_results.json")
CLIFF_CELLS_DIR = os.path.join(
    CLIFF_DIR, "results", "deltanet_rd_exactness", "wavekeyanchor-cliff")

KEYANCHOR_MECH_DIR = os.path.join(EXPRUNS, "2026-07-06_keyanchor_mech", "wavekeyanchor-mech")
KEYANCHOR_K48_DIR = os.path.join(EXPRUNS, "2026-07-07_keyanchor_k48", "wavekeyanchor-k48")

REQUIRED_PATHS = [CLIFF_RESULTS, CLIFF_CELLS_DIR, KEYANCHOR_MECH_DIR, KEYANCHOR_K48_DIR]
for p in REQUIRED_PATHS:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Expected archived data at {p} but it is missing. This script "
            "only plots real archived runs -- it does not fabricate data."
        )

D_STATE = 64  # this wave's fixed d_state (KEY_ANCHORING_DESIGN.md sec 12.2/12.9)

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe & Ito, 2008) -- identical to make_figures_v2.py
# ---------------------------------------------------------------------------
C_ORANGE = "#E69F00"
C_SKYBLUE = "#56B4E9"
C_GREEN = "#009E73"
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_PURPLE = "#CC79A7"
C_BLACK = "#000000"
C_GREY = "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,   # embed as real text, not paths
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def savefig(fig, name, formats=("pdf", "png")):
    """Write `name.<fmt>` for each fmt in `formats` into FIG_DIR."""
    for fmt in formats:
        out = os.path.join(FIG_DIR, f"{name}.{fmt}")
        kw = {"format": fmt, "bbox_inches": "tight"}
        if fmt == "png":
            kw["dpi"] = 150
        fig.savefig(out, **kw)
        print(f"wrote {out}")
    plt.close(fig)


def jload(path):
    with open(path) as f:
        return json.load(f)


def rec_h4(d):
    """h4 = recovered_frac@0.9 at the final checkpoint's held-out h=4 leg --
    identical field/path convention to make_figures_v2.py's rec_h()."""
    return d["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]


def h4_vals(base_dir, pattern):
    paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
    assert paths, f"no files for pattern {pattern!r} in {base_dir}"
    return [rec_h4(jload(p)) for p in paths]


def sigmoid(x, L, x0, w):
    return L / (1.0 + np.exp((x - x0) / w))


def main():
    fit = jload(CLIFF_RESULTS)

    # --- 3 archived anchor points, each with its own per-seed array ---
    # K=16: fixed saturation anchor, no per-seed archive at this citation
    # depth (KEY_ANCHORING_DESIGN.md sec 12.9's own table) -- single point.
    k16_h4 = [1.0]

    # K=32: seeds {10,11,12}, mechanism-tier wave (sec 10.13.1 / 3667/3870).
    k32_h4 = h4_vals(
        KEYANCHOR_MECH_DIR,
        "wkeyanchor-mech_rdx_K32_armd_s1*_geo3n20_anchor_learned_dprobe_rev7.json")
    assert len(k32_h4) == 3, f"expected 3 K=32 seeds, got {len(k32_h4)}"
    # Sanity check against the design doc's own quoted seed array (0.6741/
    # 0.7125/0.6141) -- used only as a post-hoc assertion, never fed to the plot.
    assert np.allclose(sorted(k32_h4), sorted([0.6741, 0.7125, 0.6141]), atol=2e-3), k32_h4

    # K=48: seeds {30,31,32}, K48 capacity-curve wave (sec 11.12.1).
    k48_h4 = h4_vals(
        KEYANCHOR_K48_DIR,
        "wkeyanchor-k48_rdx_K48_armd_s3*_geo3n20_anchor_learned_dprobe_rev7.json")
    assert len(k48_h4) == 3, f"expected 3 K=48 seeds, got {len(k48_h4)}"
    assert np.allclose(sorted(k48_h4), sorted([0.02295, 0.02287, 0.01872]), atol=2e-3), k48_h4

    # --- 4 new points this wave measured, each with its own per-seed array ---
    new_Ks = [34, 38, 42, 46]
    new_h4 = {}
    for K in new_Ks:
        vals = h4_vals(
            CLIFF_CELLS_DIR,
            f"wkeyanchor-k48_rdx_K{K}_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json")
        assert len(vals) == 3, f"expected 3 seeds at K={K}, got {len(vals)}"
        new_h4[K] = vals

    # --- assemble the 7-point curve (K/d, per-seed h4 arrays, label, archived?) ---
    points = [
        (16, k16_h4, "K16 (anchor)", True),
        (32, k32_h4, "K32 (archived)", True),
        (34, new_h4[34], "K34 (this wave)", False),
        (38, new_h4[38], "K38 (this wave)", False),
        (42, new_h4[42], "K42 (this wave)", False),
        (46, new_h4[46], "K46 (this wave)", False),
        (48, k48_h4, "K48 (archived)", True),
    ]
    xs_kd = [K / D_STATE for K, _, _, _ in points]
    means = [float(np.mean(vals)) for _, vals, _, _ in points]

    # Sanity-check the 7-point curve's means against fit_cliff_curve_results.json's
    # own curve_points (loaded, not retyped) and KEY_ANCHORING_DESIGN.md sec 12.9's
    # table -- post-hoc verification only.
    fit_x = fit["curve_points"]["x"]
    fit_h4 = fit["curve_points"]["h4"]
    fit_by_x = dict(zip(fit_x, fit_h4))
    for x, m in zip(xs_kd, means):
        if x in fit_by_x:
            assert abs(m - fit_by_x[x]) < 1e-3, (x, m, fit_by_x[x])

    # --- sigmoid fit + bootstrap CI, read directly from the archived JSON ---
    L = fit["sigmoid_fit"]["L"]
    x0 = fit["sigmoid_fit"]["x0"]
    w = fit["sigmoid_fit"]["w"]
    x0_lo = fit["bootstrap_ci"]["ci_x0"]["lo"]
    x0_hi = fit["bootstrap_ci"]["ci_x0"]["hi"]
    assert abs(x0 - 0.5455) < 1e-3, x0
    assert abs(w - 0.0597) < 1e-3, w
    assert abs(x0_lo - 0.5385) < 1e-3 and abs(x0_hi - 0.5513) < 1e-3, (x0_lo, x0_hi)

    xs_smooth = np.linspace(0.20, 0.80, 400)
    ys_smooth = sigmoid(xs_smooth, L, x0, w)

    # ---------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8))

    # x0 CI band (shaded vertical strip)
    ax.axvspan(x0_lo, x0_hi, color=C_PURPLE, alpha=0.18, zorder=1,
               label=f"95% CI on $x_0$ [{x0_lo:.4f}, {x0_hi:.4f}]")
    ax.axvline(x0, color=C_PURPLE, linestyle="--", linewidth=1.2, zorder=2,
               label=f"fitted $x_0{{=}}{x0:.4f}$")

    # fitted sigmoid curve
    ax.plot(xs_smooth, ys_smooth, color=C_BLUE, linewidth=1.6, zorder=3,
            label=rf"sigmoid fit ($w{{=}}{w:.4f}$)")

    # per-seed scatter + per-K means, archived anchors vs. this-wave points
    # visually distinguished by marker/color.
    for x, (K, vals, label, is_anchor), mean_v in zip(xs_kd, points, means):
        color = C_GREY if is_anchor else C_VERMILLION
        marker = "s" if is_anchor else "o"
        if len(vals) > 1:
            ax.scatter([x] * len(vals), vals, color=color, s=20, zorder=4,
                       edgecolor=C_BLACK, linewidth=0.4, marker=marker, alpha=0.85)
        ax.scatter([x], [mean_v], color=color, s=64, zorder=5, marker=marker,
                   edgecolor=C_BLACK, linewidth=0.8)

    # legend proxies for archived-anchor vs. this-wave marker distinction
    ax.scatter([], [], color=C_GREY, s=50, marker="s", edgecolor=C_BLACK,
               linewidth=0.8, label="archived anchors (K16/32/48)")
    ax.scatter([], [], color=C_VERMILLION, s=50, marker="o", edgecolor=C_BLACK,
               linewidth=0.8, label="this wave (K34/38/42/46)")

    ax.set_xlim(0.20, 0.80)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel(r"capacity ratio $K/d_{\mathrm{state}}$")
    ax.set_ylabel("recovered_frac@0.9 ($h{=}4$)")
    ax.set_title(
        "The capacity cliff, located: a measured transition point,\n"
        rf"not a bracket ($d_{{\mathrm{{state}}}}{{=}}64$, candidate (d))",
        fontsize=8.5)

    # twin top axis in raw K at d=64
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    k_ticks_kd = [0.25, 0.375, 0.5, 0.625, 0.75]
    ax_top.set_xticks(k_ticks_kd)
    ax_top.set_xticklabels([str(int(round(v * D_STATE))) for v in k_ticks_kd])
    ax_top.set_xlabel(r"$K$ (at $d_{\mathrm{state}}{=}64$)", fontsize=8)
    ax_top.tick_params(labelsize=7.5)

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=6.3)
    fig.tight_layout()
    savefig(fig, "fig_cliff")


if __name__ == "__main__":
    main()
    print("fig_cliff regenerated from archived JSONs.")
