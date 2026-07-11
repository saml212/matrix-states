"""Generate every figure for the NeurReps 2026 EA ("rank recruitment under
teeth") from archived raw result JSONs — one versioned script, md5-asserted
sources.

Per the paper skill's figure discipline (templates/figure-gen.py.tmpl):
every plotted value is COMPUTED from a raw archived artifact whose md5 is
asserted before loading; no hand-entered numbers. If an archived file
changes, the build fails loudly. The Task E spectral prediction is computed
by importing matrix-thinking/chapter2/analyze_zdump.py (numpy + stdlib
only), never re-derived by hand.

Usage (from the repo root, local machine — archives are local):
    .venv/bin/python papers/rank-recruitment-ws/figures/figure_gen.py \
        --out papers/rank-recruitment-ws/figures --repo .

Figures:
    fig_forcerank.pdf — evidence row R2 (991-run Task D snapshot, M3 grid)
    fig_depth.pdf     — evidence rows R3/R4 (Task E zdump archive)
"""

import argparse
import hashlib
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Evidence sources (paths relative to the repo root) + recorded md5s.
# Computed from the archived artifacts on 2026-07-10 and duplicated in the
# brief's claims-to-evidence map (papers/rank-recruitment-ws/brief.md).
# ---------------------------------------------------------------------------

TASKD_AGG = "matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_latest.json"
ZDUMP = "experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump"

SOURCE_MD5 = {
    # --- R1/R2: 991-run Task D snapshot (M1 + M3 grids) ---
    TASKD_AGG: "c0a7d27e33a606d81e1babfc5d674edb",
    # --- R3: five unconstrained Task E seeds (per-hop recovery) ---
    f"{ZDUMP}/t1_matrix_permutation_K8_frN_s0.json": "5ecaaeb8fe649f209cd41c35ba95c082",
    f"{ZDUMP}/t1_matrix_permutation_K8_frN_s1.json": "20b615ad2ae1b33c734a06d4a4b2210f",
    f"{ZDUMP}/t1_matrix_permutation_K8_frN_s2.json": "7908556e99a475e489a237772df7eb02",
    f"{ZDUMP}/t1_matrix_permutation_K8_frN_s3.json": "eb5ee74e494854856296dccb54b5f50c",
    f"{ZDUMP}/t1_matrix_permutation_K8_frN_s4.json": "313eda0fe2c05fc9aa7e857da913a1c0",
    # --- R4: the one converged force-rank k=K-1 seed ---
    f"{ZDUMP}/t1_matrix_permutation_K8_fr7_s2.json": "ccfafb45699d022da8f93a49b9d4b793",
}

# Okabe & Ito (2008) colorblind-safe palette
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_GREEN = "#009E73"
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def load_checked(repo, relpath):
    """Load a JSON artifact only after asserting its recorded md5."""
    path = os.path.join(repo, relpath)
    digest = hashlib.md5(open(path, "rb").read()).hexdigest()
    expected = SOURCE_MD5[relpath]
    if digest != expected:
        raise AssertionError(
            f"md5 mismatch for {relpath}: got {digest}, recorded {expected}. "
            "The archived artifact changed; refusing to plot stale/mismatched data."
        )
    with open(path) as f:
        return json.load(f)


def per_hop(run_json, field):
    """Collect {hop: value} for a metric field across the two eval blocks."""
    out = {}
    for blk in ("M2_in_distribution", "M3_held_out"):
        for h, v in run_json.get(blk, {}).items():
            out[int(h)] = v[field]
    return dict(sorted(out.items()))


def fig_forcerank(repo, outdir):
    """R2: the causal force-rank staircase, three (d, K) cells."""
    agg = load_checked(repo, TASKD_AGG)
    assert agg.get("n_runs") == 991, "expected the 991-run pre-registered snapshot"
    m3 = agg["M3_recovered_frac@0.9_vs_forcerank"]

    cells = [
        ("d8_K4", 8, 4, C_BLUE, "o"),
        ("d16_K8", 16, 8, C_VERMILLION, "s"),
        ("d16_K12", 16, 12, C_GREEN, "^"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.38), sharey=True)
    for ax, (key, d, K, color, marker) in zip(axes, cells):
        curve = m3[key]
        ks = sorted(int(k) for k in curve)
        ax.plot(ks, [curve[str(k)] for k in ks], color=color, marker=marker,
                markersize=4, linewidth=1.3, clip_on=False, zorder=3)
        ax.axvline(K, color=C_GREY, linestyle="--", linewidth=1.0, zorder=1)
        ax.text(K, 1.04, r"$k{=}K$", color=C_GREY, fontsize=7, ha="center",
                va="bottom")
        ax.set_title(f"$d{{=}}{d}$, $K{{=}}{K}$", fontsize=8.5)
        ax.set_xlabel(r"train-time force-rank $k$")
        ax.set_xlim(1, d)
        ax.set_ylim(-0.03, 1.08)
    axes[0].set_ylabel(r"rec@0.9")
    fig.tight_layout()
    out = os.path.join(outdir, "fig_forcerank.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_depth(repo, outdir):
    """R3 + R4: spectral prediction vs measured decay; the depth contrast."""
    sys.path.insert(0, os.path.join(repo, "matrix-thinking", "chapter2"))
    import analyze_zdump as az  # noqa: E402

    fr7_rel = f"{ZDUMP}/t1_matrix_permutation_K8_fr7_s2.json"
    fr7 = load_checked(repo, fr7_rel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dead-seed batch entries emit matmul warnings
        run = az.analyze_run(os.path.join(repo, fr7_rel))
        predicted, _ = az.depth_curve_for_run(run)

    hops = sorted(per_hop(fr7, "mean_cos"))
    meas_cos = per_hop(fr7, "mean_cos")
    meas_rec = per_hop(fr7, "recovered_frac@0.9")

    seeds = {}
    for s in range(5):
        d = load_checked(repo, f"{ZDUMP}/t1_matrix_permutation_K8_frN_s{s}.json")
        seeds[s] = per_hop(d, "recovered_frac@0.9")
    converged = [1, 2, 3, 4]
    suff_mean = [float(np.mean([seeds[s][h] for s in converged])) for h in hops]
    outlier = [seeds[0][h] for h in hops]

    xpos = np.arange(len(hops))
    labels = [str(h) for h in hops]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 1.65))

    ax1.plot(xpos, [predicted[h] for h in hops], color=C_BLACK, marker="o",
             markersize=4, linewidth=1.3, label="predicted (eigenspectrum only)")
    ax1.plot(xpos, [meas_cos[h] for h in hops], color=C_VERMILLION, marker="s",
             markersize=4, linewidth=1.3, linestyle="--", label="measured")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(r"hop $h$")
    ax1.set_ylabel(r"mean $\cos(Z^h k_i,\ v_{\pi^h(i)})$")
    ax1.set_title(r"force-rank $k{=}7{=}K{-}1$: spectrum predicts decay",
                  fontsize=8)
    ax1.legend(loc="lower left", frameon=False)

    ax2.plot(xpos, suff_mean, color=C_BLUE, marker="^", markersize=4,
             linewidth=1.3, label="rank-sufficient (4/5 seeds)")
    ax2.plot(xpos, [meas_rec[h] for h in hops], color=C_VERMILLION, marker="s",
             markersize=4, linewidth=1.3, linestyle="--",
             label=r"force-rank $K{-}1$")
    ax2.plot(xpos, outlier, color=C_GREY, marker="x", markersize=4,
             linewidth=1.0, linestyle=":", label="still-transitioning seed")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel(r"hop $h$")
    ax2.set_ylabel(r"rec@0.9")
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_title("recovered fraction is the sharper depth probe", fontsize=8)
    ax2.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    out = os.path.join(outdir, "fig_depth.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="papers/rank-recruitment-ws/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_forcerank(args.repo, args.out)
    fig_depth(args.repo, args.out)
    print("All figures regenerated from md5-asserted archives.")


if __name__ == "__main__":
    main()
