"""Generate all figures for the ICML MI Workshop 2026 submission.

Outputs PDF files into this directory. Run from this directory:

    python generate_figures.py

Data sources:
  - experiment-runs/2026-04-17_round_pc/rank_evals/*.json (rank-k curves)
  - EXPERIMENT_LOG.md (3-seed decoupling, scale, depth, probe)

No emoji, no cute titles. Plain research figures.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RANK_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "..", "experiment-runs",
    "2026-04-17_round_pc", "rank_evals"
))


# ------------------------------------------------------------------
# Figure 1: rank-k curves for all 5 readouts
# ------------------------------------------------------------------
def fig1_rank_curves():
    ks = np.array([1, 2, 4, 8, 16])
    # flatten (Round 3 gamma=0 baseline) from EXPERIMENT_LOG.md
    flatten = np.array([76.80, 77.00, 76.80, 76.60, 76.60])  # %
    # load the four positive-control JSONs
    readouts = {}
    for name in ["bilinear", "bilinear_gelu", "svd_aug", "quadratic"]:
        with open(os.path.join(RANK_DIR, f"{name}_rankeval.json")) as f:
            d = json.load(f)
        accs = []
        for k in ks:
            accs.append(100.0 * d["results_by_k"][str(k)]["accuracy"])
        readouts[name] = np.array(accs)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    styles = {
        "flatten":       dict(marker="o", color="#1a1a1a",   label="flatten (linear baseline)"),
        "bilinear":      dict(marker="s", color="#2E5E8B",   label="bilinear (reparam.)"),
        "bilinear_gelu": dict(marker="D", color="#8B2E1F",   label="bilinear + GELU"),
        "svd_aug":       dict(marker="^", color="#2E8B57",   label="SVD-augmented"),
        "quadratic":     dict(marker="v", color="#8B6F2E",   label="quadratic"),
    }

    ax.plot(ks, flatten, linestyle="-", linewidth=1.2, **styles["flatten"])
    for name in ["bilinear", "bilinear_gelu", "svd_aug", "quadratic"]:
        ax.plot(ks, readouts[name], linestyle="-", linewidth=1.2, **styles[name])

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"Rank-$k$ SVD truncation of $Z$ at inference")
    ax.set_ylabel("ProsQA accuracy (%)")
    ax.set_ylim(75.0, 81.0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", frameon=False, ncol=1)

    fig.tight_layout()
    out = os.path.join(HERE, "fig1_rank_curves.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Figure 2: 3-seed rank-vs-accuracy scatter
# ------------------------------------------------------------------
def fig2_seed_decoupling():
    # From EXPERIMENT_LOG.md and PAPER_RESULTS_SUMMARY.md
    seeds = [1337, 42, 7]
    z_rank = np.array([12.9, 4.0, 12.0])
    acc    = np.array([80.47, 81.25, 82.81])

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.scatter(z_rank, acc, s=60, c="#8B2E1F", marker="o", zorder=3,
               edgecolor="#1a1a1a", linewidth=0.8)
    for s, zr, a in zip(seeds, z_rank, acc):
        ax.annotate(f"seed {s}", (zr, a), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)
    # reference band for mean +/- std
    m = acc.mean(); sd = acc.std(ddof=1)
    ax.axhspan(m - sd, m + sd, color="#8B2E1F", alpha=0.08, zorder=1,
               label=fr"mean $\pm$ std ({m:.2f}\% $\pm$ {sd:.2f}pp)")
    ax.axhline(m, color="#8B2E1F", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_xlabel(r"Final effective rank of $Z$ (exp of singular-value entropy)")
    ax.set_ylabel("ProsQA best accuracy (%)")
    ax.set_xlim(0, 16)
    ax.set_ylim(79.5, 84.0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig2_seed_decoupling.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Figure 3: scale sweep
# ------------------------------------------------------------------
def fig3_scale_sweep():
    models = ["GPT-2 small\n(124M)", "GPT-2 medium\n(355M)", "GPT-2 large\n(774M)"]
    vanilla = [81.77, 80.47, 68.75]
    matrix  = [80.47, 79.69, np.nan]  # large matrix-CODI pending (OOM x2)

    x = np.arange(len(models))
    w = 0.36

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    b1 = ax.bar(x - w/2, vanilla, width=w, color="#2E5E8B",
                edgecolor="#1a1a1a", linewidth=0.6, label="Vanilla SFT")
    matrix_plot = [v if not np.isnan(v) else 0 for v in matrix]
    b2 = ax.bar(x + w/2, matrix_plot, width=w, color="#8B2E1F",
                edgecolor="#1a1a1a", linewidth=0.6, label="Matrix-CODI")

    for rect, v in zip(b1, vanilla):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.4,
                f"{v:.2f}", ha="center", fontsize=7)
    for rect, v in zip(b2, matrix):
        if np.isnan(v):
            ax.text(rect.get_x() + rect.get_width()/2, 4.0,
                    "n/a", ha="center", fontsize=7, color="#5a5a5a",
                    style="italic")
        else:
            ax.text(rect.get_x() + rect.get_width()/2, v + 0.4,
                    f"{v:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("ProsQA best accuracy (%)")
    ax.set_ylim(0, 90)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig3_scale_sweep.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Figure 4: depth sweep (partial)
# ------------------------------------------------------------------
def fig4_depth_sweep():
    # R8 depth sweep n=6 banked, n=16/32/64 pending
    ns = [6]
    acc = [78.91]
    sft_ref = 81.77  # vanilla SFT no refinement

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(ns, acc, width=4, color="#8B2E1F",
           edgecolor="#1a1a1a", linewidth=0.6, label="Vanilla CODI (iterative latent refine)")
    ax.axhline(sft_ref, color="#2E5E8B", linestyle="--", linewidth=1.0,
               label=f"Vanilla SFT (no refinement) = {sft_ref:.2f}%")
    ax.text(6, 78.91 - 1.2, f"{78.91:.2f}%", ha="center", fontsize=8, color="white")
    ax.set_xlabel(r"Number of latent refinement steps $n$")
    ax.set_ylabel("ProsQA best accuracy (%)")
    ax.set_xticks([6, 16, 32, 64])
    ax.set_xticklabels(["6", "16\n(pending)", "32\n(pending)", "64\n(pending)"])
    ax.set_ylim(70, 85)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig4_depth_sweep.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Figure 5: linear probe AUC bar chart
# ------------------------------------------------------------------
def fig5_linear_probe():
    labels = [r"matrix $Z$ (all concat)", "vanilla GPT-2 hidden", "random GPT-2 hidden"]
    aucs   = [0.673, 0.846, 0.495]
    errs   = [0.030, 0.026, 0.031]
    colors = ["#8B2E1F", "#2E5E8B", "#5a5a5a"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(x, aucs, yerr=errs, width=0.6, color=colors,
           edgecolor="#1a1a1a", linewidth=0.6, capsize=4)
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.03, f"{a:.3f}", ha="center", fontsize=8)
    ax.axhline(0.896, color="#1a1a1a", linestyle="--", linewidth=0.8,
               label="pre-reg threshold 0.896")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("5-fold macro-averaged AUC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig5_linear_probe.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# Figure 6: negative-control overlay (vanilla SFT vs positive controls)
# ------------------------------------------------------------------
def fig6_negative_control():
    """Vanilla GPT-2 SFT (no matrix bottleneck, no Z) under the same
    rank-k truncation paradigm. Shows the probe alone produces a flat
    curve on a model with no Z. Per-seed accuracies from sec. 5.4
    Table; pooled mean overlaid."""
    ks = np.array([1, 2, 4, 8, 16])
    seed_1337 = np.array([79.80, 80.20, 80.00, 80.20, 80.00])
    seed_42   = np.array([79.00, 78.80, 78.60, 78.60, 79.00])
    seed_7    = np.array([78.00, 78.00, 78.00, 78.00, 78.20])
    pooled    = np.array([78.93, 79.00, 78.87, 78.93, 79.07])

    # for the overlay reference, load flatten and quadratic from the
    # positive-control side
    flatten = np.array([76.80, 77.00, 76.80, 76.60, 76.60])

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ks, seed_1337, marker="o", linestyle=":", linewidth=0.9,
            color="#a8a8a8", label="vanilla SFT seed 1337")
    ax.plot(ks, seed_42,   marker="s", linestyle=":", linewidth=0.9,
            color="#888888", label="vanilla SFT seed 42")
    ax.plot(ks, seed_7,    marker="^", linestyle=":", linewidth=0.9,
            color="#5a5a5a", label="vanilla SFT seed 7")
    ax.plot(ks, pooled, marker="D", linestyle="-", linewidth=1.6,
            color="#1a1a1a", label="vanilla SFT pooled mean")
    ax.plot(ks, flatten, marker="o", linestyle="-", linewidth=1.2,
            color="#8B2E1F", label="matrix-CODI flatten (reference)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"Rank-$k$ SVD truncation of fake $Z$ at inference")
    ax.set_ylabel("ProsQA accuracy (%)")
    ax.set_ylim(75.0, 81.0)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower left", frameon=False, ncol=1, fontsize=7)

    fig.tight_layout()
    out = os.path.join(HERE, "fig6_negative_control.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_rank_curves()
    fig2_seed_decoupling()
    fig3_scale_sweep()
    fig4_depth_sweep()
    fig5_linear_probe()
    fig6_negative_control()
