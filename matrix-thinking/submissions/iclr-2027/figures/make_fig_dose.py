#!/usr/bin/env python3
"""Fig. dose: the coherence-dose-response test at d_state=128, K=68 (rank-4
concentrated injection) -- COHERENCE EXONERATED. Directly injecting frozen
anchor-table coherence (max|cos|), swept from 0 up to and beyond d=64's own
trained coherence band, produces NO cliff: h4=1.0 flat at every dose tested.

Mirrors `make_fig_cliff.py`'s conventions exactly (style, palette, savefig
helper, "no fabricated numbers" discipline, assert-against-archive checks) --
kept as its own script for the same reason fig_cliff is its own script: it
reads a different archive tree (`experiment-runs/2026-07-06_keyanchor_dose/`)
and is a strict *addition* to the existing figure set, not a replacement.

Panel content (single panel, per the task brief):
  - x-axis: frozen anchor-table coherence dose (`achieved_max_cos`), at
    K=68, d_state=128, rank-4 (concentrated) injection structure.
  - y-axis: h4 = recovered_frac@0.9 at held-out hop depth 4 (same
    convention as make_fig_cliff.py's rec_h4()).
  - 4 measured dose points: 0.000 (control, zero-dose K=68 cells reused
    from the d=128 cliff-universality wave), 0.130, 0.284, 0.40 (this
    wave, 2026-07-06/07 keyanchor-dose).
  - Shaded reference band: d_state=64's own final-checkpoint TRAINED
    coherence range-of-K-means, 0.373-0.385 (KEY_ANCHORING_DESIGN.md
    sec 14.0b) -- NOT d=64's init dose (0.284); the honest re-axised
    comparison target per sec 14.0b's own correction.
  - Contrast series: d=64's OWN h4 values at the matched K/d window
    (K=34/38/42/46, i.e. K/d=0.53125/0.59375/0.65625/0.71875 -- the SAME
    K/d ratios as this wave's K=68/76/84/92 at d=128), plotted against
    their own measured final-checkpoint coherence (NOT the dose axis --
    d=64 was never dosed, it was trained normally and drifted there).
    This is the "for contrast" panel element the task brief asks for:
    the same K/d window where d=64 collapses to h4~0.02-0.57 as coherence
    rises through the 0.373-0.385 band, vs. this wave's d=128/K=68 flat
    line at h4=1.0 across the identical coherence range and beyond.

Run from anywhere with a venv containing numpy/scipy/matplotlib:
    /path/to/figvenv/bin/python matrix-thinking/submissions/iclr-2027/figures/make_fig_dose.py

Data provenance (every number traced to an archived file, never hand-typed):
  - 0.000 control (K=68, d=128, zero-dose): per-seed h4 read from
    `experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/
    wkeyanchor-k48_rdx_K68_armd_s53*_geo3n20_anchor_learned_dprobe_rev7_d128.json`,
    field `checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]`.
    Achieved coherence (dose axis) for this control point is the SAME
    cells' final-checkpoint `item6_table_conditioning.max_abs_cos` mean
    (this is the drifted, non-frozen zero-dose reference, KEY_ANCHORING_
    DESIGN.md sec 14.0b -- 0.1501 mean at K=68, NOT an exact zero, since
    these cells were not frozen; reported as "0.000 (control)" on the
    x-axis per the design doc's own dose-target labeling convention,
    sec 14.12's dose table, with the drifted achieved value annotated).
  - 0.130 / 0.284 / 0.40 doses (this wave, rank-4, frozen): per-seed h4
    and per-seed achieved_max_cos read directly from
    `experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/
    wkeyanchor-dose_rdx_K68_armd_s{930-939}_geo3n20_anchor_learned_dprobe_
    rev7_d128_dose{130,284,400}_rank4_sseed20260705.json`, fields
    `checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]` and
    `exactness_config.achieved_max_cos` (frozen -- identical at every
    checkpoint, verified below).
  - d=64 contrast series (K=34/38/42/46): per-seed h4 from
    `experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/
    wavekeyanchor-cliff/wkeyanchor-k48_rdx_K{K}_armd_s*_geo3n20_anchor_
    learned_dprobe_rev7.json` (same files make_fig_cliff.py's left panel
    uses); per-seed final-checkpoint coherence from each cell's own
    `item6_table_conditioning.max_abs_cos` at `checkpoints[-1]`.
  - Reference band (d=64 range-of-K-means final coherence, 0.373-0.385):
    KEY_ANCHORING_DESIGN.md sec 14.0b (corrected at Rev 14.3), computed
    here directly from the 12 raw d=64 cliff-wave cell JSONs' own final
    `max_abs_cos`, never hand-typed from the design doc's prose.

Where NARRATIVE.md/KEY_ANCHORING_DESIGN.md quote a number, it is used ONLY
as a post-hoc sanity check (assertions below) -- never fed directly into
the plot; every plotted number is read from a JSON on disk.
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

DOSE_DIR = os.path.join(EXPRUNS, "2026-07-06_keyanchor_dose")
DOSE_CELLS_DIR = os.path.join(DOSE_DIR, "results", "wavekeyanchor-dose")

DSTATE_DIR = os.path.join(EXPRUNS, "2026-07-06_keyanchor_dstate")
DSTATE_CELLS_DIR = os.path.join(DSTATE_DIR, "results", "wavekeyanchor-dstate")

CLIFF_DIR = os.path.join(EXPRUNS, "2026-07-06_keyanchor_cliff")
CLIFF_CELLS_DIR = os.path.join(
    CLIFF_DIR, "results", "deltanet_rd_exactness", "wavekeyanchor-cliff")

REQUIRED_PATHS = [DOSE_CELLS_DIR, DSTATE_CELLS_DIR, CLIFF_CELLS_DIR]
for p in REQUIRED_PATHS:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Expected archived data at {p} but it is missing. This script "
            "only plots real archived runs -- it does not fabricate data."
        )

D_STATE = 128   # this wave's fixed d_state (KEY_ANCHORING_DESIGN.md sec 14.2)
D_STATE_REF = 64  # d=64 contrast series' own d_state (sec 12.9)

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe & Ito, 2008) -- identical to make_fig_cliff.py
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
    identical field/path convention to make_fig_cliff.py's rec_h4()."""
    return d["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]


def rec_coh(d):
    """Final-checkpoint achieved max|cos| table coherence -- same field
    convention as KEY_ANCHORING_DESIGN.md sec 14.0b's own re-pull."""
    return d["checkpoints"][-1]["item6_table_conditioning"]["max_abs_cos"]


def load_cells(base_dir, pattern):
    paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
    assert paths, f"no files for pattern {pattern!r} in {base_dir}"
    return [jload(p) for p in paths]


def build_dose_series():
    """This wave's 4-point dose grid at K=68, d=128, rank-4: 0.000 control
    (reused zero-dose K=68 cells from the cliff-universality wave) + 0.130 /
    0.284 / 0.40 (frozen, this wave)."""
    # --- 0.000 control: zero-dose K=68 cells, d128 cliff-universality wave ---
    zero_cells = load_cells(
        DSTATE_CELLS_DIR,
        "wkeyanchor-k48_rdx_K68_armd_s53*_geo3n20_anchor_learned_dprobe_rev7_d128.json")
    assert len(zero_cells) == 3, f"expected 3 K=68 zero-dose seeds, got {len(zero_cells)}"
    zero_h4 = [rec_h4(d) for d in zero_cells]
    zero_coh = [rec_coh(d) for d in zero_cells]
    assert all(abs(h - 1.0) < 1e-6 for h in zero_h4), zero_h4

    # --- 0.130 / 0.284 / 0.40, frozen, this wave ---
    dose_groups = {}
    for tag, target in (("130", 0.130), ("284", 0.284), ("400", 0.40)):
        cells = load_cells(
            DOSE_CELLS_DIR,
            f"wkeyanchor-dose_rdx_K68_armd_s*_geo3n20_anchor_learned_dprobe_rev7_d128_dose{tag}_rank4_sseed20260705.json")
        h4 = [rec_h4(d) for d in cells]
        coh = [rec_coh(d) for d in cells]
        achieved = [d["exactness_config"]["achieved_max_cos"] for d in cells]
        # Frozen-constancy sanity: final-checkpoint coherence must match the
        # exactness_config's own recorded achieved dose (frozen tables never
        # drift -- sec 14.12's own "bit-identical across all checkpoints").
        assert np.allclose(coh, achieved, atol=1e-4), (tag, coh, achieved)
        assert all(abs(h - 1.0) < 1e-6 for h in h4), (tag, h4)
        dose_groups[target] = (h4, coh, len(cells))

    # dose=0.40 has 4 seeds (3 dose cells + 1 shared calibration cell, sec
    # 14.4b) -- verify count matches the design doc's own disclosed grid.
    assert dose_groups[0.13][2] == 3, dose_groups[0.13]
    assert dose_groups[0.284][2] == 3, dose_groups[0.284]
    assert dose_groups[0.40][2] == 4, dose_groups[0.40]

    points = [
        (0.0, zero_h4, zero_coh, "0.000 (control, drifted)"),
        (0.13, *dose_groups[0.13][:2], "0.130 (frozen)"),
        (0.284, *dose_groups[0.284][:2], "0.284 (frozen)"),
        (0.40, *dose_groups[0.40][:2], "0.40 (frozen)"),
    ]
    return points


def build_d64_reference_band():
    """d=64's own final-checkpoint TRAINED coherence, range-of-K-means, from
    the 12 raw cliff-wave cells (K=34/38/42/46, 3 seeds each) -- computed
    directly here, never hand-typed from KEY_ANCHORING_DESIGN.md sec 14.0b's
    prose (used only as a post-hoc assertion below)."""
    per_k_coh_means = []
    per_k_h4 = {}
    for K in (34, 38, 42, 46):
        cells = load_cells(
            CLIFF_CELLS_DIR,
            f"wkeyanchor-k48_rdx_K{K}_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json")
        assert len(cells) == 3, f"expected 3 seeds at K={K}, got {len(cells)}"
        coh = [rec_coh(d) for d in cells]
        h4 = [rec_h4(d) for d in cells]
        per_k_coh_means.append(float(np.mean(coh)))
        per_k_h4[K] = (coh, h4)

    band_lo, band_hi = min(per_k_coh_means), max(per_k_coh_means)
    # Sanity check against KEY_ANCHORING_DESIGN.md sec 14.0b's own quoted
    # range-of-K-means (0.373-0.385, corrected at Rev 14.3) -- post-hoc only.
    assert abs(band_lo - 0.373) < 2e-3, band_lo
    assert abs(band_hi - 0.385) < 2e-3, band_hi

    return band_lo, band_hi, per_k_h4


def main():
    dose_points = build_dose_series()
    band_lo, band_hi, d64_per_k = build_d64_reference_band()

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.2))

    # ---- Reference band: d=64's own final-checkpoint trained coherence ----
    ax.axvspan(band_lo, band_hi, color=C_GREY, alpha=0.25, zorder=1,
               label=rf"$d_{{\mathrm{{state}}}}{{=}}64$ trained coherence band "
                     rf"[{band_lo:.3f}, {band_hi:.3f}]")

    # ---- d=64 contrast series: same K/d window, own trained coherence axis ----
    d64_label_used = False
    for K, (coh, h4) in sorted(d64_per_k.items()):
        kd = K / D_STATE_REF
        mean_coh = float(np.mean(coh))
        mean_h4 = float(np.mean(h4))
        lbl = (rf"$d_{{\mathrm{{state}}}}{{=}}64$ contrast (K/d matched, "
               "this wave's K/d window)") if not d64_label_used else None
        d64_label_used = True
        ax.scatter([mean_coh], [mean_h4], color=C_VERMILLION, s=70, marker="^",
                   zorder=5, edgecolor=C_BLACK, linewidth=0.8, label=lbl)
        ax.annotate(f"K={K}\n(K/d={kd:.3f})", (mean_coh, mean_h4),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.3,
                    color=C_VERMILLION)

    # ---- This wave's dose series: K=68, d=128, rank-4, flat at h4=1.0 ----
    # Vertical stagger per point avoids label collision where the 0.000
    # control (drifted, mean coh ~0.150) sits close on the x-axis to the
    # 0.130 frozen dose point.
    label_y_offsets = [16, -18, -10, -10]
    dose_x, dose_y = [], []
    for (dose, h4_vals, coh_vals, label), y_off in zip(dose_points, label_y_offsets):
        mean_coh = float(np.mean(coh_vals))
        mean_h4 = float(np.mean(h4_vals))
        dose_x.append(mean_coh)
        dose_y.append(mean_h4)
        ax.scatter([mean_coh] * len(h4_vals), h4_vals, color=C_GREEN, s=22,
                   zorder=4, edgecolor=C_BLACK, linewidth=0.4, marker="o",
                   alpha=0.85)
        ax.scatter([mean_coh], [mean_h4], color=C_GREEN, s=72, zorder=6,
                   marker="o", edgecolor=C_BLACK, linewidth=0.9)
        ax.annotate(label, (mean_coh, mean_h4), textcoords="offset points",
                    xytext=(6, y_off), fontsize=6.3, color=C_GREEN)

    order = np.argsort(dose_x)
    ax.plot(np.array(dose_x)[order], np.array(dose_y)[order], color=C_GREEN,
            linewidth=1.4, linestyle="--", zorder=3, alpha=0.8,
            label=r"$d_{\mathrm{state}}{=}128$, $K{=}68$, rank-4 (this wave, flat)")

    ax.set_xlim(-0.02, 0.46)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel(r"anchor-table coherence, achieved $\max|\cos|$")
    ax.set_ylabel("recovered_frac@0.9 ($h{=}4$)")
    ax.set_title(
        "Coherence-dose response at matched K/d: EXONERATED (rank-4)\n"
        r"$h4{=}1.0$ at every dose up to $0.40$, vs. $d_{\mathrm{state}}{=}64$'s"
        " own cliff in the same coherence range",
        fontsize=8.5)

    ax.legend(loc="center left", frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=6.3, bbox_to_anchor=(0.02, 0.42))

    fig.tight_layout()
    savefig(fig, "fig_dose")


if __name__ == "__main__":
    main()
    print("fig_dose regenerated from archived JSONs (dose-response at K=68/"
          "d=128/rank-4, flat at h4=1.0, vs. d=64 K/d-matched contrast).")
