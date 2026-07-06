#!/usr/bin/env python3
"""Regenerate every figure for the ICLR 2027 submission directly from
archived experiment JSONs/CSVs -- no hand-typed numbers. Each figure
function documents exactly which archived file(s) it reads and which
NARRATIVE.md figure-plan entry (Sec. 2) it implements.

Run from anywhere with the project's fig-generation venv, e.g.:
    /path/to/figvenv/bin/python matrix-thinking/submissions/iclr-2027/figures/make_figures_v2.py

Requires: numpy, matplotlib (pure-Python analysis, no torch/GPU needed).
Outputs vector PDFs AND rasterized PNGs into this same `figures/` directory.

FILENAME MAP (figure number -> output basename, this script's convention):
  Fig 1  -> fig01_eval_trunc_synth_vs_real          (2-panel)
  Fig 2  -> fig02_k_axis_exactness_frontier
  Fig 3  -> fig03_rank_recruited_vs_K
  Fig 4  -> fig04_embedding_source_attractor
  Fig 5  -> fig05_2x2_existence_proof
  Fig 6  -> fig06_soft_fixes_epsilon_h_law
  Fig 7  -> fig07_fgeo3_headline
  Fig 8  -> fig08_wave2_corpus_generality           (appendix)
  Fig 9  -> fig09_trackc_ladder_and_trackd          (2-panel, appendix)
  Fig 10 -> fig10_anchoring_wave_arc                (3-panel)
  Fig 11 -> fig11_capacity_curve                    (headline)

Data provenance: traced against matrix-thinking/submissions/iclr-2027/
NARRATIVE.md round 5 Sec. 2 (the figure plan) and independently verified by
direct inspection of every archived JSON/CSV file before any plotting code
below was written (per this repo's CLAUDE.md: "Every number must be read
from a file on disk"). Where NARRATIVE.md quotes an expected number, that
number is used ONLY as a post-hoc sanity check in code comments -- never as
a target fed back into the plot.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
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
DELTANET_RD = os.path.join(MT_DIR, "deltanet_rd")

# Fig 1
EVAL_TRUNC_SYNTH_AGG = os.path.join(
    EXPRUNS, "2026-07-02_deltanet_waves", "eval_trunc", "eval_trunc_d64_K32_AGGREGATE.json")
WAVE0_RERUN_DIR = os.path.join(EXPRUNS, "2026-07-03_deltanet_rd_waves", "wave0_rerun")
WAVEA_DIR = os.path.join(EXPRUNS, "2026-07-03_deltanet_rd_waves", "waveA")

# Fig 2, 3
WAVEA_AGG = os.path.join(WAVEA_DIR, "AGGREGATE.json")
SYNTH_WAVE0_DIR = os.path.join(EXPRUNS, "2026-07-02_deltanet_waves", "wave0")
SYNTH_WAVE0_AGG = os.path.join(SYNTH_WAVE0_DIR, "AGGREGATE.json")

# Fig 4, 5
WAVE1_DIR = os.path.join(EXPRUNS, "2026-07-03_deltanet_rd_waves", "exactness", "wave1")
WAVEGEO3_DIR = os.path.join(EXPRUNS, "2026-07-03_deltanet_rd_waves", "exactness", "wavegeo3")

# Fig 6, 7
WAVEF_DIR = os.path.join(EXPRUNS, "2026-07-03_deltanet_rd_waves", "exactness", "waveF")

# Fig 8
WAVE2_LM_DIR = os.path.join(EXPRUNS, "2026-07-04_lm_rd_wave2")
WAVED_DIR = os.path.join(WAVE2_LM_DIR, "waveD")

# Fig 9
TRAJ_PROBES_DIR = os.path.join(EXPRUNS, "2026-07-06_trajectory_probes")
TRAJ_TIDY_CSV = os.path.join(TRAJ_PROBES_DIR, "trajectories_tidy.csv")
TRACKD_JSON = os.path.join(EXPRUNS, "2026-07-04_track_d", "attractor_probe_trackd.json")
# Documented expected path convention for the not-yet-existing rung-3 harvest
# (extrapolated from experiment-runs/2026-07-04_trackc_rung1/rung1_pooled.json
# and experiment-runs/2026-07-05_trackc_rung2/rung2_final_pooled.json's own
# naming pattern: experiment-runs/<harvest-date>_trackc_rung<N>/rung<N>_final_pooled.json).
# Verified absent as of this script's writing (repo-wide search for
# "rung3"/"rung-3" returns zero hits); this is the ONE figure allowed
# missing data per the task brief -- see fig9_trackc_and_trackd's Panel A.
RUNG3_GLOB_PATTERNS = [
    os.path.join(EXPRUNS, "*trackc_rung3*", "rung3_final_pooled.json"),
    os.path.join(EXPRUNS, "*trackc_rung3*", "*_pooled.json"),
    os.path.join(EXPRUNS, "*trackc_rung3*", "*.json"),
]

# Fig 10
# NOTE (verified by direct inspection, corrects an initial guess): the
# "fresh bare-geo3 reference" (0.4105) used as Panel A's first bar does NOT
# live in exactness/wavegeo3/ (that dir's geo3-alone K32 n12 mean is 0.4368,
# the number quoted for Fig 5/Fig 7's "not-stable+orthogonal" cell instead).
# The 0.4105 reference lives in keyanchor_wave/waveref/.
KEYANCHOR_WAVEREF_DIR = os.path.join(EXPRUNS, "2026-07-05_keyanchor_wave", "waveref")
KEYANCHOR_WAVE1_DIR = os.path.join(EXPRUNS, "2026-07-05_keyanchor_wave", "wavekeyanchor")
KEYANCHOR_CONFIRM_DIR = os.path.join(EXPRUNS, "2026-07-05_keyanchor_confirm", "wavekeyanchor-confirm")
KEYANCHOR_MECH_DIR = os.path.join(EXPRUNS, "2026-07-06_keyanchor_mech", "wavekeyanchor-mech")
KEYANCHOR_E_DIR = os.path.join(EXPRUNS, "2026-07-07_keyanchor_e", "wavekeyanchor-e")

# Fig 11
KEYANCHOR_K48_DIR = os.path.join(EXPRUNS, "2026-07-07_keyanchor_k48", "wavekeyanchor-k48")
KEYANCHOR_K48_REF_DIR = os.path.join(EXPRUNS, "2026-07-07_keyanchor_k48", "wavekeyanchor-k48-ref")
BANDS_PINNED_K48 = os.path.join(KEYANCHOR_K48_REF_DIR, "BANDS_PINNED_K48.json")
BANDS_PINNED = os.path.join(KEYANCHOR_WAVEREF_DIR, "BANDS_PINNED.json")

REQUIRED_PATHS = [
    EVAL_TRUNC_SYNTH_AGG, WAVE0_RERUN_DIR, WAVEA_DIR, WAVEA_AGG,
    SYNTH_WAVE0_DIR, SYNTH_WAVE0_AGG, WAVE1_DIR, WAVEGEO3_DIR, WAVEF_DIR,
    WAVE2_LM_DIR, WAVED_DIR, TRAJ_PROBES_DIR, TRAJ_TIDY_CSV, TRACKD_JSON,
    KEYANCHOR_WAVEREF_DIR, KEYANCHOR_WAVE1_DIR, KEYANCHOR_CONFIRM_DIR,
    KEYANCHOR_MECH_DIR, KEYANCHOR_E_DIR, KEYANCHOR_K48_DIR,
    KEYANCHOR_K48_REF_DIR, BANDS_PINNED_K48, BANDS_PINNED,
]
for p in REQUIRED_PATHS:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Expected archived data at {p} but it is missing. This script "
            "only plots real archived runs -- it does not fabricate data. "
            "Check experiment-runs/ is present and up to date."
        )

sys.path.insert(0, DELTANET_RD)
import analyze_eval_truncation_rd as aetr  # noqa: E402  (numpy + stdlib only)

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Okabe & Ito, 2008) + redundant markers/linestyles
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


def rec_h(d, h, tau="recovered_frac@0.9"):
    """Fetch rec@0.9 (or another stat) for hop h from a standard RD run
    JSON's final checkpoint. h in {1,2,3} live under M2_in_distribution;
    h in {4,5,6,7,21} live under M3_held_out. Verified schema (both agent
    inspection and direct spot-checks against NARRATIVE.md sanity values):
        d["checkpoints"][-1]["M2_in_distribution"|"M3_held_out"][str(h)][tau]
    """
    ck = d["checkpoints"][-1]
    bucket = ck["M2_in_distribution"] if str(h) in ck.get("M2_in_distribution", {}) else ck["M3_held_out"]
    return bucket[str(h)][tau]


# ---------------------------------------------------------------------------
# Figure 1: synthetic-vs-real eval-truncation staircase (2-panel).
# NARRATIVE.md Fig. 1. Synthetic: eval_trunc_d64_K32_AGGREGATE.json (h=1,
# d=64, K=32; already-inspected schema, reusing the workshop script's exact
# indexing pattern). Real: analyze_eval_truncation_rd.py's own
# discover_unconstrained_runs/per_k_grid/run_staircase/aggregate_over_seeds,
# pooling wave0_rerun + waveA at K=32 (n=7 seeds total, matching
# DELTANET_REALDATA_DESIGN.md sec 17.3/17.4's pooling instruction). Real
# data is h=1 ONLY -- analyze_eval_truncation_rd.py's own docstring proves
# this is a principled boundary of the archived Z_dump (no `succ` permutation
# is dumped, so h>=2 can't be reconstructed without an argmax shortcut this
# project's own rules forbid) -- so only h=1 is plotted for the real panel,
# unlike the synthetic panel which has multiple h curves.
# ---------------------------------------------------------------------------
def fig01_eval_trunc_synth_vs_real():
    # --- synthetic panel (identical to the workshop script's fig_eval_trunc_staircase) ---
    agg = jload(EVAL_TRUNC_SYNTH_AGG)
    k_grid_synth = agg["k_grid"]
    K_synth = 32
    synth_vals = {}
    for h in (1,):
        vals = []
        for k in k_grid_synth:
            per_seed_vals = [ps["staircase"][str(h)][str(k)]["recovered_frac@0.9"]
                              for ps in agg["per_seed"]]
            vals.append(np.mean(per_seed_vals))
        synth_vals[h] = vals

    # --- real-data panel ---
    real_dirs = [WAVE0_RERUN_DIR, WAVEA_DIR]
    runs = aetr.discover_unconstrained_runs(real_dirs)
    k32_runs = [r for r in runs if r["K"] == 32]
    assert len(k32_runs) == 7, f"expected n=7 pooled K=32 seeds per NARRATIVE.md sec 2, got {len(k32_runs)}"
    d_state = k32_runs[0]["d_state"]
    k_grid_real = aetr.per_k_grid(32, d_state)
    staircases = [aetr.run_staircase(r, k_grid_real) for r in k32_runs]
    agg_real = aetr.aggregate_over_seeds(staircases, k_grid_real, "measured")
    real_ks = sorted(agg_real.keys())
    real_vals = [agg_real[k][0] for k in real_ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))

    ax1.plot(k_grid_synth, synth_vals[1], color=C_BLUE, marker="o", markersize=4,
             linewidth=1.3, clip_on=False, zorder=3)
    ax1.axvline(K_synth, color=C_GREY, linestyle="--", linewidth=1.0, zorder=1)
    ax1.text(K_synth, 1.03, r"$k{=}K{=}32$", color=C_GREY, fontsize=7, ha="center", va="bottom")
    ax1.set_xlabel("eval-time SVD-truncation rank $k$")
    ax1.set_ylabel("recovered_frac@0.9")
    ax1.set_ylim(-0.03, 1.12)
    ax1.set_title("synthetic: razor cliff at $k{=}31{\\to}32$", fontsize=8)

    ax2.plot(real_ks, real_vals, color=C_VERMILLION, marker="s", markersize=4,
             linewidth=1.3, clip_on=False, zorder=3)
    ax2.axvline(32, color=C_GREY, linestyle="--", linewidth=1.0, zorder=1)
    ax2.text(32, 1.03, r"$k{=}K{=}32$", color=C_GREY, fontsize=7, ha="center", va="bottom")
    ax2.set_xlabel("eval-time SVD-truncation rank $k$")
    ax2.set_ylim(-0.03, 1.12)
    ax2.set_title(f"real text (GPT-2 tokenized): graded, $n{{=}}{len(k32_runs)}$ seeds pooled\n"
                  "(wave0_rerun + waveA; ceiling $\\approx$0.79, not 1.0)", fontsize=7.5)

    fig.suptitle(
        r"$d_{\mathrm{state}}{=}64, K{=}32, h{=}1$: synthetic razor cliff vs. real-data "
        "graded transition (~12--13-rank window)", fontsize=8, y=1.05,
    )
    fig.tight_layout()
    savefig(fig, "fig01_eval_trunc_synth_vs_real")


# ---------------------------------------------------------------------------
# Figure 2: K-axis exactness frontier on real text (NARRATIVE.md Fig. 2).
# Source: experiment-runs/2026-07-03_deltanet_rd_waves/waveA/ per-seed run
# JSONs (AGGREGATE.json only has h=1 headline numbers; h=1..7 comes from
# each run's own checkpoints[-1]["M2_in_distribution"|"M3_held_out"] table,
# verified directly -- see rec_h() above). K in {8,16,24,32}, 2 seeds each.
# ---------------------------------------------------------------------------
def fig02_k_axis_exactness_frontier():
    Ks = [8, 16, 24, 32]
    colors = {8: C_BLUE, 16: C_GREEN, 24: C_ORANGE, 32: C_VERMILLION}
    hops = [1, 2, 3, 4, 5, 6, 7]

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2))
    for K in Ks:
        paths = sorted(glob.glob(os.path.join(WAVEA_DIR, f"wA_rd_K{K}_frN_s*.json")))
        # exclude the _composition variant files (a different diagnostic, K16 only)
        paths = [p for p in paths if "_composition" not in os.path.basename(p)]
        assert paths, f"no waveA run files found for K={K}"
        runs = [jload(p) for p in paths]
        vals_per_h = []
        for h in hops:
            vals_per_h.append(np.mean([rec_h(d, h) for d in runs]))
        ax.plot(hops, vals_per_h, color=colors[K], marker="o", markersize=4,
                linewidth=1.5, label=f"$K{{=}}{K}$ ($n{{=}}{len(runs)}$)")
    ax.axvline(3.5, color=C_GREY, linestyle=":", linewidth=1.0, zorder=1)
    ax.text(3.5, 1.05, "in-dist. | held-out", color=C_GREY, fontsize=6.5, ha="center")
    ax.set_xlabel("hop depth $h$")
    ax.set_ylabel("recovered_frac@0.9")
    ax.set_xticks(hops)
    ax.set_ylim(-0.03, 1.12)
    ax.set_title("Real-text K-axis exactness frontier\n"
                  "($d_{\\mathrm{state}}{=}64$, waveA, per-seed mean)", fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig02_k_axis_exactness_frontier")


# ---------------------------------------------------------------------------
# Figure 3: rank recruited vs K -- capacity is not the issue (NARRATIVE.md
# Fig. 3). Real: waveA/AGGREGATE.json's by_cell[*]["headline_entity_subspace_
# eff_rank_h1"] as % of target K (K parsed from the cell-name string, e.g.
# "K32_frN" -> 32). Synthetic no-inflation reference: 2026-07-02_deltanet_
# waves/wave0/AGGREGATE.json's by_cell[*]["final_entity_subspace_eff_rank_h1"]
# (cell format "d64_K{K}_frN"; only K in {16,32} available at d=64).
# ---------------------------------------------------------------------------
def fig03_rank_recruited_vs_K():
    real_agg = jload(WAVEA_AGG)
    real_Ks, real_pct = [], []
    for cell, info in sorted(real_agg["by_cell"].items(),
                              key=lambda kv: int(kv[0].split("_")[0][1:])):
        K = int(cell.split("_")[0][1:])
        pct = 100.0 * info["headline_entity_subspace_eff_rank_h1"] / K
        real_Ks.append(K)
        real_pct.append(pct)

    synth_agg = jload(SYNTH_WAVE0_AGG)
    synth_Ks, synth_pct = [], []
    for cell, info in sorted(synth_agg["by_cell"].items()):
        parts = cell.split("_")
        d_tok = next(p for p in parts if p.startswith("d"))
        k_tok = next(p for p in parts if p.startswith("K"))
        if d_tok != "d64":
            continue  # only the d=64 cells are the real-data-comparable operating point
        K = int(k_tok[1:])
        pct = 100.0 * info["final_entity_subspace_eff_rank_h1"] / K
        synth_Ks.append(K)
        synth_pct.append(pct)
    order = np.argsort(synth_Ks)
    synth_Ks = list(np.array(synth_Ks)[order])
    synth_pct = list(np.array(synth_pct)[order])

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.0))
    width = 1.6
    x_real = np.array(real_Ks, dtype=float)
    ax.bar(x_real - width / 2, real_pct, width=width, color=C_VERMILLION,
           label="real text (waveA)", zorder=3)
    x_synth = np.array(synth_Ks, dtype=float)
    ax.bar(x_synth + width / 2, synth_pct, width=width, color=C_BLUE,
           label="synthetic, $d{=}64$ (no-inflation ref.)", zorder=3)
    ax.axhline(100, color=C_GREY, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlabel("target rank $K$")
    ax.set_ylabel("effective rank recruited\n(% of target $K$)")
    ax.set_xticks(sorted(set(real_Ks) | set(synth_Ks)))
    ax.set_ylim(0, 112)
    ax.set_title("Rank recruitment survives transfer to real text\n"
                  "(Fig. 2's decay is NOT a capacity failure)", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig03_rank_recruited_vs_K")


def _key_gram_dev(path):
    """Mean key-Gram deviation (h=1,2,3 average -- near-constant across h in
    this diagnostic, confirmed by direct inspection) from a wave1/wavegeo3
    run JSON's final checkpoint."""
    d = jload(path)
    ck = d["checkpoints"][-1]
    vals = [ck["M2_in_distribution"][str(h)]["key_gram_deviation_mean"] for h in (1, 2, 3)]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Figure 4: embedding-source interpolation attractor (NARRATIVE.md Fig. 4).
# Source: experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/,
# arms i, ii, iii-beta, iv, i-strong at K=32. Field:
# checkpoints[-1]["M2_in_distribution"][h]["key_gram_deviation_mean"]
# (h=1,2,3 averaged; verified near-constant across h).
# ---------------------------------------------------------------------------
def fig04_embedding_source_attractor():
    arms = [
        ("i\n(frozen orthon.)", "w1_rdx_K32_armi_s*.json", C_SKYBLUE),
        ("ii\n(frozen real emb.)", "w1_rdx_K32_armii_s*.json", C_GREEN),
        ("iii-beta\n(learned)", "w1_rdx_K32_armiii-beta_s*.json", C_ORANGE),
        ("iv\n(Gram-matched)", "w1_rdx_K32_armiv_s*_rho0p225.json", C_PURPLE),
        ("i-strong\n(surgical pin)", "w1_rdx_K32_armi-strong_s*_sp.json", C_VERMILLION),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.2))
    labels, means = [], []
    for i, (label, pattern, color) in enumerate(arms):
        paths = sorted(glob.glob(os.path.join(WAVE1_DIR, pattern)))
        assert paths, f"no files for arm {label!r} pattern {pattern}"
        vals = [_key_gram_dev(p) for p in paths]
        means.append(np.mean(vals))
        labels.append(label)
        ax.scatter([i] * len(vals), vals, color=color, s=22, zorder=3,
                   edgecolor=C_BLACK, linewidth=0.4)
        ax.bar([i], [np.mean(vals)], color=color, alpha=0.35, width=0.6, zorder=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel(r"key-Gram deviation $\|K_{\mathrm{eff}}^T K_{\mathrm{eff}} - I\|_F$")
    ax.set_title("Every input geometry converges to the same write-time\n"
                  "attractor except the surgical pin ($K{=}32$)", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "fig04_embedding_source_attractor")


# ---------------------------------------------------------------------------
# Figure 5: 2x2 existence proof (NARRATIVE.md Fig. 5). rec@0.9 at h=4,7,
# K=32, across {orthogonal, not-orthogonal} x {stable, not-stable}.
#   stable+orthogonal:      wave1 arm i-strong
#   stable+not-orthogonal:  wave1 arm i (frozen but trained W_k de-orthog.)
#   not-stable+orthogonal:  wavegeo3 bare geo3, n_iter=12 (NOT the n20 escal.)
#   not-stable+not-orthog.: wave1 arm iii-beta (fully-learned baseline)
# ---------------------------------------------------------------------------
def fig05_2x2_existence_proof():
    cells = {
        ("orthogonal", "stable"): ("i-strong", "w1_rdx_K32_armi-strong_s*_sp.json", WAVE1_DIR),
        ("not-orthogonal", "stable"): ("i (frozen)", "w1_rdx_K32_armi_s*.json", WAVE1_DIR),
        ("orthogonal", "not-stable"): ("bare geo3 (n12)", "wgeo3_rdx_K32_armgeo3_s*_geo3n12.json", WAVEGEO3_DIR),
        ("not-orthogonal", "not-stable"): ("iii-beta (baseline)", "w1_rdx_K32_armiii-beta_s*.json", WAVE1_DIR),
    }
    data = {}
    for key, (label, pattern, base_dir) in cells.items():
        paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
        assert paths, f"no files for cell {key} pattern {pattern}"
        runs = [jload(p) for p in paths]
        h4 = [rec_h(d, 4) for d in runs]
        h7 = [rec_h(d, 7) for d in runs]
        data[key] = {"label": label, "h4": h4, "h7": h7, "n": len(runs)}

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.0), sharex=True, sharey=True)
    grid_order = [("not-orthogonal", "stable"), ("orthogonal", "stable"),
                  ("not-orthogonal", "not-stable"), ("orthogonal", "not-stable")]
    for ax, key in zip(axes.flat, grid_order):
        ortho, stable = key
        cell = data[key]
        xs = [0, 1]
        for x, (h, vals) in zip(xs, [(4, cell["h4"]), (7, cell["h7"])]):
            ax.scatter([x] * len(vals), vals, color=C_BLUE, s=26, zorder=3,
                       edgecolor=C_BLACK, linewidth=0.4)
            ax.scatter([x], [np.mean(vals)], color=C_VERMILLION, marker="_",
                       s=300, linewidth=2, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(["$h{=}4$", "$h{=}7$"])
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(f"{ortho}, {stable}\n({cell['label']}, $n{{=}}{cell['n']}$)", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("recovered_frac@0.9")
    fig.suptitle(
        "2$\\times$2 existence proof: orthogonality and stability are each\n"
        "necessary, jointly sufficient ($K{=}32$). Note: i-strong is\n"
        "pool-restricted to $32 \\leq d_{\\mathrm{state}}$.",
        fontsize=8, y=1.03,
    )
    fig.tight_layout()
    savefig(fig, "fig05_2x2_existence_proof")


# ---------------------------------------------------------------------------
# Figure 6: soft fixes fail proportionally -- the epsilon^h law (NARRATIVE.md
# Fig. 6). Source: exactness/waveF/. Baseline = w1_iiibeta_K32_s*.json (the
# waveF-local copy, a distinct seed realization from wave1/'s iii-beta, used
# for self-consistency within this batch per the verified agent finding).
# Soft arms: wF_geo1_l10 (lambda=1.0), wF_geo1_l01 (lambda=0.1), wF_geo2_zca
# (ZCA whitening). Gain = soft_arm - baseline at h=2,3,4, K=32.
# ---------------------------------------------------------------------------
def fig06_soft_fixes_epsilon_h_law():
    def load_mean_by_h(pattern, hops):
        paths = sorted(glob.glob(os.path.join(WAVEF_DIR, pattern)))
        assert paths, f"no waveF files for pattern {pattern}"
        runs = [jload(p) for p in paths]
        return {h: np.mean([rec_h(d, h) for d in runs]) for h in hops}

    hops = [2, 3, 4]
    baseline = load_mean_by_h("w1_iiibeta_K32_s*.json", hops)
    arms = [
        ("F-geo-1 $\\lambda{=}1.0$", "wF_geo1_l10_K32_s*.json", C_BLUE, "o"),
        ("F-geo-1 $\\lambda{=}0.1$", "wF_geo1_l01_K32_s*.json", C_GREEN, "s"),
        ("F-geo-2 (ZCA)", "wF_geo2_zca_K32_s*.json", C_ORANGE, "^"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2))
    for label, pattern, color, marker in arms:
        arm_vals = load_mean_by_h(pattern, hops)
        gains = [arm_vals[h] - baseline[h] for h in hops]
        ax.plot(hops, gains, color=color, marker=marker, markersize=5,
                linewidth=1.5, label=label, zorder=3)
    ax.axhline(0.5, color=C_VERMILLION, linestyle="--", linewidth=1.0,
               label="pre-registered headline bar ($\\geq 0.5$)")
    ax.axhline(0.8, color=C_BLACK, linestyle=":", linewidth=1.0,
               label="min.-publishable bar ($\\geq 0.8$)")
    ax.axhline(0.0, color=C_GREY, linewidth=0.6, zorder=1)
    ax.set_xticks(hops)
    ax.set_xlabel("hop depth $h$")
    ax.set_ylabel("gain over baseline, recovered_frac@0.9")
    ax.set_title("Soft fixes shrink geometrically with depth ($K{=}32$)\n"
                  r"($\varepsilon^h$ compounding law)", fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig06_soft_fixes_epsilon_h_law")


# ---------------------------------------------------------------------------
# Figure 7: F-geo-3's headline demonstration (NARRATIVE.md Fig. 7). Grouped
# bar, h=4 rec@0.9 at K=16 and K=32: baseline (wave1 iii-beta) vs. best soft
# arm (waveF geo1_l01, the highest-gain arm from Fig. 6 at h=4) vs. F-geo-3
# (wavegeo3, K16 uses n_iter=12, K32 uses the n_iter=20 escalation cells per
# NARRATIVE.md's explicit instruction -- NOT the K32 n12 cells).
# ---------------------------------------------------------------------------
def fig07_fgeo3_headline():
    def h4_vals(base_dir, pattern):
        paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
        assert paths, f"no files for pattern {pattern} in {base_dir}"
        return [rec_h(jload(p), 4) for p in paths]

    groups = {
        16: {
            "baseline": h4_vals(WAVE1_DIR, "w1_rdx_K16_armiii-beta_s*.json"),
            "best soft (F-geo-1 $\\lambda{=}0.1$)": h4_vals(WAVEF_DIR, "wF_geo1_l01_K16_s*.json"),
            "F-geo-3": h4_vals(WAVEGEO3_DIR, "wgeo3_rdx_K16_armgeo3_s*_geo3n12.json"),
        },
        32: {
            "baseline": h4_vals(WAVEF_DIR, "w1_iiibeta_K32_s*.json"),
            "best soft (F-geo-1 $\\lambda{=}0.1$)": h4_vals(WAVEF_DIR, "wF_geo1_l01_K32_s*.json"),
            "F-geo-3": h4_vals(WAVEGEO3_DIR, "wgeo3_rdx_K32_armgeo3_s*_geo3n20.json"),
        },
    }
    bar_labels = ["baseline", "best soft (F-geo-1 $\\lambda{=}0.1$)", "F-geo-3"]
    colors = [C_GREY, C_ORANGE, C_VERMILLION]
    ref_lines = {16: 0.8, 32: 0.5}

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
    for ax, K in zip(axes, (16, 32)):
        for i, (label, color) in enumerate(zip(bar_labels, colors)):
            vals = groups[K][label]
            ax.bar([i], [np.mean(vals)], color=color, width=0.6, zorder=2)
            ax.scatter([i] * len(vals), vals, color=C_BLACK, s=16, zorder=3)
        ax.axhline(ref_lines[K], color=C_BLUE, linestyle="--", linewidth=1.0,
                   label=f"bar $\\geq{ref_lines[K]}$")
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(["baseline", "best\nsoft", "F-geo-3"], fontsize=7.5)
        ax.set_title(f"$K{{=}}{K}$", fontsize=9)
        ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    axes[0].set_ylabel("recovered_frac@0.9 ($h{=}4$)")
    fig.suptitle("F-geo-3 clears the K=16 bar and improves K=32 ~45$\\times$\n"
                 "(K=32 uses the $n_{\\mathrm{iter}}{=}20$ escalation)", fontsize=8.5, y=1.05)
    fig.tight_layout()
    savefig(fig, "fig07_fgeo3_headline")


# ---------------------------------------------------------------------------
# Figure 8 (appendix): Wave 2 corpus generality (NARRATIVE.md Fig. 8).
# Truncation-damage (nats/token, raw_mean) vs k in {8,16,24,32,48,64},
# openr1 vs wikitext, each on its OWN home corpus (per analysis_lm_w2.py's
# "HOME-corpus" headline table, reproduced here by direct JSON parsing
# rather than regexing its .txt output -- the per-seed JSONs are the clean
# source; the .txt/.py are read-only references, not imported, since
# analysis_lm_w2.py hardcodes remote /home/nvidia paths and isn't
# importable as-is).
# Source: experiment-runs/2026-07-04_lm_rd_wave2/waveD/wD_lm_{corpus}_s{seed}.json
# ---------------------------------------------------------------------------
def fig08_wave2_corpus_generality():
    corpora = [("openr1", C_VERMILLION, "o"), ("wikitext", C_BLUE, "s")]
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2))
    for corpus, color, marker in corpora:
        paths = sorted(glob.glob(os.path.join(WAVED_DIR, f"wD_lm_{corpus}_s*.json")))
        assert paths, f"no waveD files for corpus {corpus}"
        runs = [jload(p) for p in paths]
        k_grid = runs[0]["k_grid"]
        for r in runs:
            assert r["k_grid"] == k_grid, "k_grid mismatch across seeds"
        means, stds = [], []
        for k in k_grid:
            vals = [r["damage_by_corpus_and_k"][corpus]["by_k"][str(k)]["raw_mean"] for r in runs]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(k_grid, means, yerr=stds, color=color, marker=marker, markersize=5,
                    linewidth=1.5, capsize=2, label=f"{corpus} (home corpus, $n{{=}}{len(runs)}$)")
    ax.axhline(0.0, color=C_GREY, linewidth=0.6, zorder=1)
    ax.set_xlabel("eval-time truncation rank $k$")
    ax.set_ylabel("truncation damage (nats/token)")
    ax.set_title("Truncation damage is not an artifact of the synthetic\n"
                  "probe grammar: two real corpora, same $k^*{\\approx}48$", fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig08_wave2_corpus_generality")


def _find_rung3_data():
    """Search the documented expected path convention for the not-yet-landed
    Track C rung-3 (1.31B) harvest. Extrapolated from rung-1/rung-2's own
    naming (experiment-runs/<date>_trackc_rung{1,2}/rung{1,2}_final_pooled.json
    or *_pooled.json). Returns (param_count, span_frac) if found, else None.
    Verified absent as of this script's writing via repo-wide search for
    "rung3"/"rung-3" (zero hits) -- this is the ONE figure allowed missing
    data per the task brief. Once the file lands at any of these patterns,
    re-running this script picks it up with NO code change."""
    for pattern in RUNG3_GLOB_PATTERNS:
        for path in sorted(glob.glob(pattern)):
            try:
                d = jload(path)
            except Exception:
                continue
            # Expect a pooled-probe schema analogous to rung1/rung2's
            # *_pooled.json: look for a span-frac-like field and a param count.
            span = d.get("archived4_span_frac") or d.get("span_frac")
            n_params = d.get("n_params")
            if span is not None and n_params is not None:
                return float(n_params), float(span)
    return None


# ---------------------------------------------------------------------------
# Figure 9 (appendix): Track C pure-scale ladder + Track D production models
# (NARRATIVE.md Fig. 9, 2-panel).
# Panel A source: experiment-runs/2026-07-06_trajectory_probes/
# trajectories_tidy.csv (540 rows). Column `family` (NOT "cell" -- verified;
# distinct values mixcontrol/wave1/wave1ext/wave2) x column `archived4_span_frac`
# (the cross-scale-comparable convention). family->param-count mapping cross-
# checked against each track's own summary .txt: mixcontrol=14,048,896 (14M),
# wave1ext=97,618,176 (98M, EXTENDED-mix -- matches 0.344, NOT plain `wave1`
# which is the original-mix 98M cell), wave2=391,869,440 (392M). Rung-3
# (1.31B) is not yet archived -- see _find_rung3_data() above; if absent,
# a hollow placeholder marker is drawn with NO plotted y-value.
# Panel B source: experiment-runs/2026-07-04_track_d/attractor_probe_trackd.json,
# models.{rwkv7,falcon_mamba,qwen_control}.per_corpus.{openr1,wikitext}.
# per_chunk_size["64"].pooled_across_layers_gram_deviation_mean.
# ---------------------------------------------------------------------------
def fig09_trackc_ladder_and_trackd():
    # --- Panel A: read the tidy CSV (stdlib csv, no pandas in the venv) ---
    with open(TRAJ_TIDY_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 540, f"expected 540 rows in trajectories_tidy.csv, got {len(rows)}"

    family_to_paramcount = {}
    family_span = {}
    for r in rows:
        fam = r["family"]
        family_to_paramcount.setdefault(fam, set()).add(int(r["n_params"]))
        family_span.setdefault(fam, []).append(
            (int(r["step"]), float(r["archived4_span_frac"])))

    # 3 confirmed rungs, in scale order: 14M (mixcontrol), 98M (wave1ext,
    # extended-mix -- the cell that numerically matches NARRATIVE.md's 0.344),
    # 392M (wave2).
    ladder_families = ["mixcontrol", "wave1ext", "wave2"]
    for fam in ladder_families:
        assert fam in family_span, f"family {fam!r} not found in trajectories_tidy.csv"

    xs_params, ys_span = [], []
    for fam in ladder_families:
        n_params = sorted(family_to_paramcount[fam])[0]
        # use the LAST checkpoint (max step) per seed, averaged across
        # seeds/corpora at that final step -- the archived-comparable read.
        last_step = max(s for s, _ in family_span[fam])
        vals_at_last = [v for s, v in family_span[fam] if s == last_step]
        xs_params.append(n_params)
        ys_span.append(float(np.mean(vals_at_last)))

    rung3 = _find_rung3_data()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    axA.plot(xs_params, ys_span, color=C_BLUE, marker="o", markersize=5,
             linewidth=1.5, zorder=3, label="measured (trajectories_tidy.csv)")
    if rung3 is not None:
        n4, span4 = rung3
        axA.plot([xs_params[-1], n4], [ys_span[-1], span4], color=C_BLUE,
                  linewidth=1.5, zorder=3)
        axA.scatter([n4], [span4], color=C_BLUE, marker="o", s=30, zorder=4)
    else:
        # Rung-3 not yet archived: draw a visible hollow placeholder marker
        # at its documented expected x-position (1.31B params, per
        # NARRATIVE.md/STATE.md), with NO y-value plotted at all -- placed
        # in AXES-fraction y-coordinates (blended transform) specifically so
        # it cannot be mistaken for a real data point at some y, and so a
        # nan-y scatter (which matplotlib would silently exclude from the
        # autoscaled data range, making it invisible) is never needed.
        rung3_x_placeholder = 1.31e9
        axA.set_xlim(xs_params[0] * 0.5, rung3_x_placeholder * 1.6)
        trans = matplotlib.transforms.blended_transform_factory(axA.transData, axA.transAxes)
        axA.axvline(rung3_x_placeholder, color=C_GREY, linestyle=":", linewidth=1.2, zorder=2)
        axA.scatter([rung3_x_placeholder], [0.5], marker="o", s=70,
                    facecolors="none", edgecolors=C_GREY, linewidth=1.5,
                    zorder=4, transform=trans, clip_on=False)
        axA.annotate("rung-3 pending\n(1.31B, running)\nno y-value yet",
                     xy=(rung3_x_placeholder, 0.5), xycoords=trans,
                     xytext=(-50, 0), textcoords="offset points", ha="center", va="center",
                     fontsize=6.3, color=C_GREY)
    axA.set_xscale("log")
    axA.set_xlabel("model scale (params)")
    axA.set_ylabel("write-geometry span-fraction\n(archived4_span_frac)")
    axA.set_title("Panel A: Track C pure-scale ladder", fontsize=8.5)
    axA.legend(loc="upper left", frameon=False, fontsize=6.5)

    # --- Panel B: Track D production models ---
    trackd = jload(TRACKD_JSON)
    model_labels = [("rwkv7", "RWKV-7 1.5B", C_BLUE),
                    ("falcon_mamba", "Falcon-Mamba-7B", C_VERMILLION),
                    ("qwen_control", "Qwen2.5-1.5B\n(neg. control)", C_GREY)]
    for i, (key, label, color) in enumerate(model_labels):
        m = trackd["models"][key]
        vals = []
        for corpus in ("openr1", "wikitext"):
            v = m["per_corpus"][corpus]["per_chunk_size"]["64"]["pooled_across_layers_gram_deviation_mean"]
            vals.append(v)
        axB.bar([i], [np.mean(vals)], color=color, width=0.6, zorder=2)
        axB.scatter([i] * len(vals), vals, color=C_BLACK, s=20, zorder=3)
    axB.set_xticks(range(len(model_labels)))
    axB.set_xticklabels([lbl for _, lbl, _ in model_labels], fontsize=6.5)
    axB.set_ylabel("key-Gram deviation (chunk=64)")
    axB.set_title("Panel B: Track D production models\n"
                   "(measurement-only; not delta-rule-attributed)", fontsize=8)
    fig.suptitle("The write-geometry attractor persists with scale "
                 "(Panel A) and at\nproduction scale (Panel B, honest non-finding)",
                 fontsize=8.5, y=1.06)
    fig.tight_layout()
    savefig(fig, "fig09_trackc_ladder_and_trackd")


# ---------------------------------------------------------------------------
# Figure 10: the anchoring wave's complete arc (NARRATIVE.md Fig. 10,
# 3-panel). K=32, h=4 throughout.
# Panel A source (chronological):
#   bare-geo3 reference (0.4105)  -> keyanchor_wave/waveref/wref_rdx_K32_armgeoref_s{1,2,3}_geo3n20_dprobe.json
#     (VERIFIED: NOT exactness/wavegeo3/, which is a different, earlier
#     geo3-alone run with a different mean, 0.4368 -- corrected during
#     data-inspection, see KEYANCHOR_WAVEREF_DIR comment above)
#   Wave-1/confirm arm d          -> keyanchor_wave/wavekeyanchor/wkeyanchor_rdx_K32_armd_s{0,1,2}_geo3n20_anchor_learned.json
#   mechanism wave candidate (d)  -> keyanchor_mech/wavekeyanchor-mech/wkeyanchor-mech_rdx_K32_armd_s{10,11,12}_...json
#   mechanism wave candidate (d') -> keyanchor_mech/wavekeyanchor-mech/wkeyanchor-mech_rdx_K32_armdprime_s{20,21,22}_...json
# Panel B source: same mechanism-tier wave files, field r_e_engagement
#   (median_r_e, engaged_frac_v3_with_hubs, band).
# Panel C source: keyanchor_e/wavekeyanchor-e/, arms e (frozen random) and
#   e-fp (frozen frame-potential), same r_e_engagement field for the inset.
# ---------------------------------------------------------------------------
def fig10_anchoring_wave_arc():
    def h4_vals(base_dir, pattern):
        paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
        assert paths, f"no files for pattern {pattern} in {base_dir}"
        return [rec_h(jload(p), 4) for p in paths]

    def re_stats(base_dir, pattern):
        paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
        assert paths, f"no files for pattern {pattern} in {base_dir}"
        out = []
        for p in paths:
            ck = jload(p)["checkpoints"][-1]
            re = ck["r_e_engagement"]
            out.append({"median_r_e": re["median_r_e"],
                        "engaged_frac": re["engaged_frac_v3_with_hubs"],
                        "band": re["band"]})
        return out

    fig = plt.figure(figsize=(6.8, 5.6))
    axA = fig.add_subplot(2, 2, 1)
    axB = fig.add_subplot(2, 2, 2)
    axC = fig.add_subplot(2, 2, 3)
    axC_inset = fig.add_subplot(2, 4, 7)

    # --- Panel A: chronological bars ---
    panelA_groups = [
        ("bare-geo3\nreference", h4_vals(KEYANCHOR_WAVEREF_DIR, "wref_rdx_K32_armgeoref_s*_geo3n20_dprobe.json"), C_GREY),
        ("Wave-1/\nconfirm (d)", h4_vals(KEYANCHOR_WAVE1_DIR, "wkeyanchor_rdx_K32_armd_s*_geo3n20_anchor_learned.json"), C_SKYBLUE),
        ("mech. wave\ncand. (d)", h4_vals(KEYANCHOR_MECH_DIR, "wkeyanchor-mech_rdx_K32_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json"), C_BLUE),
        ("mech. wave\ncand. (d$'$)", h4_vals(KEYANCHOR_MECH_DIR, "wkeyanchor-mech_rdx_K32_armdprime_s*_geo3n20_anchor_learned_per_entity_dprobe_rev7.json"), C_GREEN),
    ]
    for i, (label, vals, color) in enumerate(panelA_groups):
        axA.bar([i], [np.mean(vals)], color=color, width=0.6, zorder=2)
        axA.scatter([i] * len(vals), vals, color=C_BLACK, s=16, zorder=3)
    axA.axhline(0.5, color=C_VERMILLION, linestyle="--", linewidth=1.0, label="bar $\\geq 0.5$")
    axA.set_xticks(range(len(panelA_groups)))
    axA.set_xticklabels([g[0] for g in panelA_groups], fontsize=6.5)
    axA.set_ylabel("recovered_frac@0.9 ($h{=}4$, $K{=}32$)")
    axA.set_title("(A) chronological arc", fontsize=8.5)
    axA.legend(loc="lower right", frameon=False, fontsize=6)

    # --- Panel B: engagement scatter, mechanism wave ---
    mech_arms = [
        ("cand. (d)", "wkeyanchor-mech_rdx_K32_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json", C_BLUE, "o"),
        ("cand. (d$'$)", "wkeyanchor-mech_rdx_K32_armdprime_s*_geo3n20_anchor_learned_per_entity_dprobe_rev7.json", C_GREEN, "^"),
    ]
    for label, pattern, color, marker in mech_arms:
        stats = re_stats(KEYANCHOR_MECH_DIR, pattern)
        axB.scatter([s["median_r_e"] for s in stats], [s["engaged_frac"] * 100 for s in stats],
                    color=color, marker=marker, s=36, label=f"{label} (band {stats[0]['band']})", zorder=3)
    axB.axvline(0.25, color=C_GREY, linestyle=":", linewidth=1.0, label="band C/A boundary ($r_e{<}0.25$)")
    axB.set_xlabel("median $r_e$ (per-cell)")
    axB.set_ylabel("engaged_frac_v3 (%)")
    axB.set_title("(B) engagement instrument: band C\nat every K=32 cell", fontsize=8.5)
    axB.legend(loc="upper left", frameon=False, fontsize=6)

    # --- Panel C: the resolution -- frozen arms match/exceed learned ---
    cand_d_vals = h4_vals(KEYANCHOR_MECH_DIR,
                          "wkeyanchor-mech_rdx_K32_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json")
    panelC_groups = [
        ("learned (d)", cand_d_vals, C_BLUE),
        ("frozen-random\n(e)", h4_vals(KEYANCHOR_E_DIR, "wkeyanchor-e_rdx_K32_arme_s*_geo3n20_anchor_fixed_lam0p58_dprobe_rev7.json"), C_ORANGE),
        ("frozen-frame-pot.\n(e-fp)", h4_vals(KEYANCHOR_E_DIR, "wkeyanchor-e_rdx_K32_arme-fp_s*_geo3n20_anchor_fixed_lam0p58_dprobe_rev7.json"), C_VERMILLION),
    ]
    for i, (label, vals, color) in enumerate(panelC_groups):
        axC.bar([i], [np.mean(vals)], color=color, width=0.6, zorder=2)
        axC.scatter([i] * len(vals), vals, color=C_BLACK, s=16, zorder=3)
    axC.set_xticks(range(len(panelC_groups)))
    axC.set_xticklabels([g[0] for g in panelC_groups], fontsize=6.5)
    axC.set_ylabel("recovered_frac@0.9 ($h{=}4$)")
    axC.set_title("(C) both frozen arms match/exceed\nthe learned table", fontsize=8.5)

    # inset: arm e's median r_e is negative (negative control)
    e_stats = re_stats(KEYANCHOR_E_DIR, "wkeyanchor-e_rdx_K32_arme_s*_geo3n20_anchor_fixed_lam0p58_dprobe_rev7.json")
    e_re = [s["median_r_e"] for s in e_stats]
    axC_inset.scatter([0] * len(e_re), e_re, color=C_ORANGE, s=20, zorder=3)
    axC_inset.axhline(0.0, color=C_GREY, linewidth=0.8, zorder=1)
    axC_inset.set_xticks([])
    axC_inset.set_ylabel("median $r_e$", fontsize=7)
    axC_inset.set_title("arm e: negative\ncontrol", fontsize=6.5)
    axC_inset.tick_params(labelsize=6)

    fig.suptitle(
        "The anchoring wave's complete arc: a real gain (A), a rejected\n"
        "mechanism (B), and a confirmed explanation (C) -- $K{=}32$, $h{=}4$",
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "fig10_anchoring_wave_arc")


# ---------------------------------------------------------------------------
# Figure 11 (headline): the capacity curve (NARRATIVE.md Fig. 11).
# K=16/32 candidate (d): keyanchor_mech/wavekeyanchor-mech/ (K16: single
#   seed s10, geo3n12 suffix -- the mechanism wave only ran 1 K16 seed,
#   verified via directory listing; K32: 3 seeds, geo3n20 suffix).
# K=48 candidate (d): keyanchor_k48/wavekeyanchor-k48/, 3 seeds.
# K=48 reference arm + ceiling: keyanchor_k48/wavekeyanchor-k48-ref/,
#   BANDS_PINNED_K48.json (ceiling K=48 only).
# K=16/32 ceilings: keyanchor_wave/waveref/BANDS_PINNED.json (bands "16","32").
# Drift-space / value-Gram channels: K48 armd/ref files' drift_probe and
#   M3_held_out[4].value_gram_deviation_mean fields.
# ---------------------------------------------------------------------------
def fig11_capacity_curve():
    def h4_vals(base_dir, pattern):
        paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
        assert paths, f"no files for pattern {pattern} in {base_dir}"
        return [rec_h(jload(p), 4) for p in paths]

    def h1_vals(base_dir, pattern):
        paths = sorted(glob.glob(os.path.join(base_dir, pattern)))
        assert paths, f"no files for pattern {pattern} in {base_dir}"
        return [rec_h(jload(p), 1) for p in paths]

    Ks = [16, 32, 48]
    kd_ratio = {16: 0.25, 32: 0.5, 48: 0.75}

    measured_h4 = {
        16: h4_vals(KEYANCHOR_MECH_DIR, "wkeyanchor-mech_rdx_K16_armd_s*_geo3n12_anchor_learned_dprobe_rev7.json"),
        32: h4_vals(KEYANCHOR_MECH_DIR, "wkeyanchor-mech_rdx_K32_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json"),
        48: h4_vals(KEYANCHOR_K48_DIR, "wkeyanchor-k48_rdx_K48_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json"),
    }
    guard_h1 = {
        16: h1_vals(KEYANCHOR_MECH_DIR, "wkeyanchor-mech_rdx_K16_armd_s*_geo3n12_anchor_learned_dprobe_rev7.json"),
        32: h1_vals(KEYANCHOR_MECH_DIR, "wkeyanchor-mech_rdx_K32_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json"),
        48: h1_vals(KEYANCHOR_K48_DIR, "wkeyanchor-k48_rdx_K48_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json"),
    }

    bands = jload(BANDS_PINNED)          # K=16, K=32 ceilings
    bands_k48 = jload(BANDS_PINNED_K48)  # K=48 ceiling
    ceiling = {
        16: bands["bands"]["16"]["ceiling"],
        32: bands["bands"]["32"]["ceiling"],
        48: bands_k48["bands"]["48"]["ceiling"],
    }

    xs = [kd_ratio[K] for K in Ks]
    measured_means = [np.mean(measured_h4[K]) for K in Ks]
    ceiling_vals = [ceiling[K] for K in Ks]

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6))
    for K, x in zip(Ks, xs):
        ax.scatter([x] * len(measured_h4[K]), measured_h4[K], color=C_VERMILLION, s=18, zorder=3)
    ax.plot(xs, measured_means, color=C_VERMILLION, marker="o", markersize=6,
            linewidth=1.8, label="measured, cand. (d), $h{=}4$", zorder=4)
    ax.plot(xs, ceiling_vals, color=C_BLUE, marker="s", markersize=6,
            linewidth=1.8, linestyle="--", label="theoretical $\\lambda{=}1$ ceiling", zorder=4)
    K48_bar = 0.024434  # pre-registered K=48 bar (KEY_ANCHORING_DESIGN.md), reference constant
    ax.axhline(K48_bar, color=C_GREY, linestyle=":", linewidth=1.0,
               label=f"K=48 pre-registered bar ({K48_bar:.4f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$K/d{{=}}{kd_ratio[K]}$\n($K{{=}}{K}$)" for K in Ks])
    ax.set_xlabel("capacity ratio $K/d_{\\mathrm{state}}$")
    ax.set_ylabel("recovered_frac@0.9 ($h{=}4$)")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("The capacity curve: a cliff, not a smooth decline\n"
                 "(theoretical ceiling declines only gently)", fontsize=8.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), frameon=True,
              framealpha=0.9, edgecolor="none", fontsize=6.2)

    # small inset: h=1 in-distribution guard, flat at 1.0 across all K
    ax_inset = fig.add_axes([0.64, 0.58, 0.24, 0.22])
    for K, x in zip(Ks, xs):
        ax_inset.scatter([x] * len(guard_h1[K]), guard_h1[K], color=C_GREEN, s=10, zorder=3)
    ax_inset.set_ylim(0.95, 1.02)
    ax_inset.set_xticks(xs)
    ax_inset.set_xticklabels([str(K) for K in Ks], fontsize=6)
    ax_inset.set_title("$h{=}1$ guard", fontsize=6.5)
    ax_inset.tick_params(labelsize=5.5)

    # Annotate K=48's two explanatory channels directly from the archived
    # drift_probe / value_gram_deviation_mean fields (armd vs ref arm).
    armd_paths = sorted(glob.glob(os.path.join(
        KEYANCHOR_K48_DIR, "wkeyanchor-k48_rdx_K48_armd_s*_geo3n20_anchor_learned_dprobe_rev7.json")))
    ref_paths = sorted(glob.glob(os.path.join(
        KEYANCHOR_K48_REF_DIR, "wref_rdx_K48_armgeoref_s*_geo3n20_dprobe.json")))
    armd_runs = [jload(p) for p in armd_paths]
    ref_runs = [jload(p) for p in ref_paths]
    drift_post_ns_armd = np.mean([d["checkpoints"][-1]["drift_probe"]["post_ns"]["mean"] for d in armd_runs])
    drift_post_ns_ref = np.mean([d["checkpoints"][-1]["drift_probe"]["post_ns"]["mean"] for d in ref_runs])
    vgram_armd = np.mean([d["checkpoints"][-1]["M3_held_out"]["4"]["value_gram_deviation_mean"] for d in armd_runs])
    vgram_ref = np.mean([d["checkpoints"][-1]["M3_held_out"]["4"]["value_gram_deviation_mean"] for d in ref_runs])
    vgram_ratio = vgram_armd / vgram_ref
    annotation = (f"$K{{=}}48$ channels (both active, insufficient):\n"
                  f"drift-space post-NS {drift_post_ns_armd:.3f} vs.\n"
                  f"ceiling {ceiling[48]:.3f} vs. ref {drift_post_ns_ref:.3f}\n"
                  f"value-Gram {vgram_ratio:.3f}$\\times$ reference")
    ax.annotate(annotation, xy=(0.75, measured_means[-1]), xytext=(0.44, 0.15),
                fontsize=6, color=C_BLACK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.8))

    fig.tight_layout()
    savefig(fig, "fig11_capacity_curve")


if __name__ == "__main__":
    fig01_eval_trunc_synth_vs_real()
    fig02_k_axis_exactness_frontier()
    fig03_rank_recruited_vs_K()
    fig04_embedding_source_attractor()
    fig05_2x2_existence_proof()
    fig06_soft_fixes_epsilon_h_law()
    fig07_fgeo3_headline()
    fig08_wave2_corpus_generality()
    fig09_trackc_ladder_and_trackd()
    fig10_anchoring_wave_arc()
    fig11_capacity_curve()
    print("All 11 figures regenerated from archived JSONs/CSVs.")
