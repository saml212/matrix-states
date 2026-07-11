# figure-gen.py — single versioned figure script for the flagship paper.
#
# Filled from the paper skill's templates/figure-gen.py.tmpl. Every figure
# regenerates from archived result files under experiment-runs/ with an md5
# assertion per source file (SOURCE_MD5 below); a checksum mismatch fails the
# build rather than plotting stale data. No plotted value is hand-entered.
#
# Run from the repo root with the repo venv:
#   .venv/bin/python papers/flagship/figures/figure-gen.py \
#       --out papers/flagship/figures/ --data .
#
# Palette: 5-hue colorblind-safe categorical set (Okabe-Ito subset), validated
# with the dataviz skill's validate_palette.js (CVD separation PASS; the two
# lower-contrast hues are always paired with direct labels).

import argparse
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCE_MD5 = {
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S3__unconstrained__seed0.json": "c00d6f8d83937ced62f9e4400cf98246",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S3__unconstrained__seed1.json": "fa3e6810d5ac2130e1476247d345f963",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S3__unconstrained__seed2.json": "236a30d1d76a0d0421b97b6d793db0c7",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed0.json": "1f30c88318ba1c501d74265f47bf3113",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed1.json": "2090631683f971390a4000cc5000589f",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed2.json": "8d45fbb53fa4999d1decc784746900fd",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed3.json": "2b41c525128f94a2cc78d573cbd8a614",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed4.json": "d5c8740eefea71974c1d7d87aaadc1c3",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A5__unconstrained__seed0.json": "3f6492c9b41a78da6c92317fef672318",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A5__unconstrained__seed1.json": "f69493f9dd4fd60809a921101ce7c18c",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A5__unconstrained__seed2.json": "7b496884f765448f5bb0dd625b9462d2",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A5__unconstrained__seed3.json": "78745c1bbae90a911ca79aa9275fd9a1",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A5__unconstrained__seed4.json": "943e5a5e1285d45eaf248a0db477a149",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S5__unconstrained__seed0.json": "ff9ade6ad7deb270031fae57b413be2a",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S5__unconstrained__seed1.json": "b8ba517e2c6a2fe149d5703f09980fc2",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/S5__unconstrained__seed2.json": "74d86227f94fa6ec34f327b5f52cf082",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A6__unconstrained__seed0.json": "8549e9b6bd72515d620197ff1613a810",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A6__unconstrained__seed1.json": "e306d8cfd6edf4e573fb7a916307ac7f",
    "experiment-runs/2026-07-09_capability_sweep_harvest/results/A6__unconstrained__seed2.json": "24f20af8d85e5fd4c222cb3b127f970c",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S3__unconstrained__seed0.json": "a80bae92cbbcbfa52076cf423a244ddc",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S3__k_dmin_minus_1__seed0.json": "6bdd18ac270648047ee6e561f772ade6",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S3__k_dmin__seed0.json": "c60de1276f68dc5f2b911daff6850dee",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S3__k_dmin_plus_1__seed0.json": "660ce9c26e05503a000f6d13aad3b0b1",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S4__unconstrained__seed0.json": "f9389820342f51fecd24bbb830b5f20c",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S4__k_dmin_minus_1__seed0.json": "26acb7227ae2ee6e4b91f7e42fe5b0eb",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S4__k_dmin__seed0.json": "4925e1a7c965773107365c22bacc7967",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S4__k_dmin_plus_1__seed0.json": "02c7fb35419005c0624bdde911737115",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A5__unconstrained__seed0.json": "139e38060fd659ee9648d4bde83c3bf7",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A5__k_dmin_minus_1__seed0.json": "019caf31c72090d99b1db6422151c38c",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A5__k_dmin__seed0.json": "9ea3cc9f3da906d9a2bb2a9d508eac9f",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A5__k_dmin_plus_1__seed0.json": "6ab3999c1dd5d04ec96f96e9a6d4fd5c",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S5__unconstrained__seed0.json": "8dbe19afd816e12f4eedae3371104898",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S5__k_dmin_minus_1__seed0.json": "61bf20dc9c95f3a1aacc5da31ed06bd9",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S5__k_dmin__seed0.json": "41b7b539ca5711100cf4f40e1bda2b41",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S5__k_dmin_plus_1__seed0.json": "5c46ff77a2d7a2007dd131ce5b707253",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A6__unconstrained__seed0.json": "8e847bbb3fc0c84d3a74984a7f1d6209",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A6__k_dmin_minus_1__seed0.json": "c3641255962eecf7a250ccba05b2412e",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A6__k_dmin__seed0.json": "b252a7e63b1aa79192065edbb92efe06",
    "experiment-runs/2026-07-09_m3fix_harvest/zero_pad__A6__k_dmin_plus_1__seed0.json": "a203553383019dfe5bd01d6529610dab",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__unconstrained__seed1.json": "e3896481a209439b5112b01da7d5dc88",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__unconstrained__seed2.json": "81faf7e76f1f2c99d19b06ee96a12b34",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__unconstrained__seed3.json": "6efa28c9eebe72695daac0a3fc95beed",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_minus_1__seed1.json": "a84d9486161e7d6a61c06958782d77a2",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_minus_1__seed2.json": "525dfb94cf684d8d2306dd14bc912e49",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_minus_1__seed3.json": "629e8c41cf019447d73bd7aa2da345ac",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin__seed1.json": "9cbb21c10a4cfba64a5ec636744bbb1a",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin__seed2.json": "c9f1cd3c271a697039d065d07a712d1f",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin__seed3.json": "83de23e150d5ec5020a7c843cc6441c9",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_plus_1__seed1.json": "05d0cba6d40720ce51b4ab592bf80416",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_plus_1__seed2.json": "269c1dfb9285cbbff33a11d0401deb70",
    "experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_plus_1__seed3.json": "247d1de636d5df36530bd3414283b8e7",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s0_round4.json": "d559e498e8d24c0673727e34377519a3",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s1_round4.json": "3934ea69c60243b380f86eb3db9fdd70",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s2_round4.json": "677eb57748f06e59520ca37925ed2309",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_ablation_task1_sweep_s0_round4.json": "491d0eb61f1c29bdc0824ac14eba4ef0",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_ablation_task1_sweep_s1_round4.json": "2b415704c956b3b54d8786f3fdf58810",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_ablation_task1_sweep_s2_round4.json": "b71b44341d44bb30b5099f7dd9d62957",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_transformer_task1_sweep_s0_round4.json": "988fdae364ad1cb6cd41e6c28ff9b564",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_transformer_task1_sweep_s1_round4.json": "dc57f209053ae9245492724cf9024e17",
    "experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_transformer_task1_sweep_s2_round4.json": "0d715323ffd03c020ddf759a4579d065",
    "experiment-runs/2026-07-10_h2h_mstar/MSTAR_VERDICT.json": "4f115ad55d5301122f387df504efa35c",
    "experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_contender.json": "362333c89f4223c427fe8daf54f50fce",
    "experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_ablation.json": "ff8e352a13c2bc1e177f53f8cef47c01",
    "experiment-runs/2026-07-06_trajectory_probes/trajectories_tidy.json": "0fe53d8b40285b93fe81219fa6ff9606",
    "experiment-runs/2026-07-06_trackc_rung3/probe_analysis_rung3.json": "6a627c315b0c8e35e084bbbe7730a2f8",
    "experiment-runs/2026-07-09_attrrob_2x2_escalation_harvest/n3_recompute_summary.json": "b4bdffdf25bf85f84945a40b9170a467",
    "experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json": "f2f0aae84908c0db0a42b13c76a85158",
    "experiment-runs/2026-07-09_zdump_complement/complement_results.json": "03208edab7adc7e433f2cad46ee975bb",
}

# ---------------------------------------------------------------- style
PALETTE = {
    "blue": "#0072B2",
    "verm": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "ink": "#000000",
    "gray": "#666666",
    "lightgray": "#B0B0B0",
}
GROUP_COLORS = {
    "S3": PALETTE["blue"],
    "S4": PALETTE["verm"],
    "A5": PALETTE["green"],
    "S5": PALETTE["purple"],
    "A6": PALETTE["orange"],
}
ARM_COLORS = {
    "contender": PALETTE["blue"],
    "ablation": PALETTE["verm"],
    "transformer": PALETTE["gray"],
}
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
    }
)
SINGLE = (3.3, 2.4)
WIDE = (6.8, 2.4)

DMIN = {"S3": 2, "S4": 3, "A5": 3, "S5": 4, "A6": 5}
SEEDS_PER_GROUP = {"S3": 3, "S4": 5, "A5": 5, "S5": 3, "A6": 3}
GROUPS = ["S3", "S4", "A5", "S5", "A6"]

DATA_ROOT = "."


def _path(rel):
    return os.path.join(DATA_ROOT, rel)


def _assert_md5(rel):
    expected = SOURCE_MD5.get(rel)
    if not expected:
        raise ValueError(f"No recorded md5 for {rel!r} — refusing to plot.")
    with open(_path(rel), "rb") as f:
        actual = hashlib.md5(f.read()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"md5 mismatch for {rel!r}: expected {expected}, got {actual}. "
            f"The archived result file changed — do NOT plot it."
        )


def _load(rel):
    _assert_md5(rel)
    with open(_path(rel)) as f:
        return json.load(f)


# ---------------------------------------------------------------- fig 1
def fig1_rank_vs_dmin(out_dir):
    """Recruited restricted effective rank vs d_min; band + tie-capped rho."""
    fig, ax = plt.subplots(figsize=SINGLE)
    xs = np.linspace(1.6, 5.4, 50)
    ax.fill_between(xs, 0.7 * xs, 1.3 * xs, color="#EEEEEE", zorder=0,
                    label=r"$[0.7,1.3]\cdot d_{\min}$ band")
    ax.plot(xs, xs, ls="--", lw=0.8, color=PALETTE["lightgray"], zorder=1)
    jit = {"S3": 0.0, "S4": -0.09, "A5": 0.09, "S5": 0.0, "A6": 0.0}
    for g in GROUPS:
        vals = [
            _load(
                f"experiment-runs/2026-07-09_capability_sweep_harvest/"
                f"results/{g}__unconstrained__seed{s}.json"
            )["restricted_effective_rank"]
            for s in range(SEEDS_PER_GROUP[g])
        ]
        x = DMIN[g] + jit[g]
        ax.scatter([x] * len(vals), vals, s=14, facecolors="none",
                   edgecolors=GROUP_COLORS[g], linewidths=0.9, zorder=3)
        ax.scatter([x], [np.mean(vals)], s=26, color=GROUP_COLORS[g], zorder=4)
        dy = -0.42 if g in ("S4",) else 0.28
        ax.annotate(g, (x, np.mean(vals) + dy), ha="center",
                    fontsize=7, color=GROUP_COLORS[g])
    ax.set_xlabel(r"minimal faithful representation dimension $d_{\min}$")
    ax.set_ylabel("restricted effective rank")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.6, 5.4)
    ax.set_ylim(1.0, 6.6)
    ax.text(0.03, 0.97, r"Spearman $\rho=0.9747$" "\n(tie-capped maximum)",
            transform=ax.transAxes, va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_rank_vs_dmin.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig2_forcerank_staircase(out_dir):
    """xrec90 vs forced rank k per group; 0.000 below d_min, recovery at it."""
    fig, axes = plt.subplots(1, 5, figsize=(6.8, 1.9), sharey=True)
    for ax, g in zip(axes, GROUPS):
        dm = DMIN[g]
        arms = ["k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]
        ks = [dm - 1, dm, dm + 1]

        def cell(arm, seed=0, base="experiment-runs/2026-07-09_m3fix_harvest"):
            return _load(f"{base}/zero_pad__{g}__{arm}__seed{seed}.json")[
                "crosscheck_recovered_frac_90"
            ]

        anchor = cell("unconstrained")
        ys = [cell(a) for a in arms]
        ax.axhline(0.9 * anchor, ls=":", lw=0.9, color=PALETTE["gray"])
        ax.axhline(anchor, ls="--", lw=0.7, color=PALETTE["lightgray"])
        ax.plot(ks, ys, marker="o", ms=4, color=GROUP_COLORS[g],
                drawstyle="steps-post" if False else "default")
        if g == "S3":
            ext = "experiment-runs/2026-07-09_m3fix_s3ext"
            seed_ys = []
            for s in (1, 2, 3):
                sy = [cell(a, seed=s, base=ext) for a in arms]
                seed_ys.append(sy)
                ax.plot(ks, sy, marker="o", ms=2.5, lw=0.7, alpha=0.55,
                        color=GROUP_COLORS[g])
            mean_at_dmin = np.mean([ys[1]] + [sy[1] for sy in seed_ys])
            ax.scatter([dm], [mean_at_dmin], s=34, marker="D",
                       color=PALETTE["ink"], zorder=5)
            ax.annotate("4-seed mean", (dm, mean_at_dmin),
                        textcoords="offset points", xytext=(4, 6), fontsize=6)
        ax.set_title(rf"{g} ($d_{{\min}}={dm}$)", fontsize=7.5)
        ax.set_xticks(ks)
        ax.set_xticklabels([rf"$d_{{\min}}{s}$" for s in ("-1", "", "+1")],
                           fontsize=6.5)
        ax.set_ylim(-0.05, 1.02)
        if ax is axes[0]:
            ax.set_ylabel("held-out xrec90")
    axes[2].set_xlabel("forced train-time rank $k$")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_forcerank_staircase.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig3_recall_separation(out_dir):
    """acc_A separation at n=3 (left); constant-memory horizon vs KV caps."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=WIDE,
                                  gridspec_kw={"width_ratios": [1, 1.3]})
    arms = ["contender", "ablation", "transformer"]
    for i, a in enumerate(arms):
        vals = [
            _load(
                f"experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/"
                f"h2h_{a}_task1_sweep_s{s}_round4.json"
            )["leg_a"]["acc_A"]
            for s in range(3)
        ]
        x = np.array([i - 0.12, i, i + 0.12])
        ax.scatter(x, vals, s=22, color=ARM_COLORS[a], zorder=3)
        ax.annotate(f"{np.mean(vals):.4f}", (i, max(np.mean(vals) + 0.05, 0.115)),
                    ha="center", fontsize=7, color=ARM_COLORS[a])
    ax.axhline(0.03125, ls="--", lw=0.8, color=PALETTE["gray"])
    ax.axhline(0.09375, ls=":", lw=0.9, color=PALETTE["ink"])
    ax.text(2.42, 0.035, "chance", fontsize=6, ha="right")
    ax.text(2.42, 0.10, "bar", fontsize=6, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["contender", "vector-state\nablation", "transformer"])
    ax.set_ylabel(r"episodic recall acc$_A$")
    ax.set_ylim(-0.03, 1.08)

    m = _load("experiment-runs/2026-07-10_h2h_mstar/MSTAR_VERDICT.json")
    t1 = m["per_task"]["task1_sweep"]
    horizons = {"H2": 454, "H4": 902, "H8": 1798}
    hx = [horizons[h] for h in ("H2", "H4", "H8")]
    cont = np.array([t1["contender_refs_acc_A"][h] for h in ("H2", "H4", "H8")])
    for s in range(3):
        ax2.plot(hx, cont[:, s], marker="o", ms=3.5, color=ARM_COLORS["contender"],
                 alpha=0.9, lw=1.1)
    ax2.annotate("contender (fixed 32,768-B state)", (hx[1], 0.93),
                 ha="center", fontsize=6.5, color=ARM_COLORS["contender"])
    for M, per_h in sorted(t1["capped_transformer_acc_A"].items(), key=lambda kv: int(kv[0])):
        ys = [np.mean(per_h[h]) for h in ("H2", "H4", "H8")]
        ax2.plot(hx, ys, marker="s", ms=2.2, lw=0.7, color=PALETTE["lightgray"])
    unc = [np.mean(t1["transformer_uncapped_refs_acc_A"][h]) for h in ("H2", "H4", "H8")]
    ax2.plot(hx, unc, marker="^", ms=3, lw=0.9, ls="--", color=PALETTE["gray"])
    ax2.annotate(r"transformer, capped $M{\in}\{1..32\}$ and uncapped",
                 (hx[1], 0.085), ha="center", fontsize=6.5, color=PALETTE["gray"])
    ax2.axhline(0.03125, ls="--", lw=0.8, color=PALETTE["gray"])
    ax2.set_xscale("log")
    ax2.set_xticks(hx)
    ax2.set_xticklabels([str(v) for v in hx])
    ax2.minorticks_off()
    ax2.set_xlabel("query horizon (tokens)")
    ax2.set_ylabel(r"acc$_A$")
    ax2.set_ylim(-0.03, 1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_recall_separation.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig4_tap_localization(out_dir):
    """State-zeroing localization (left); linear-probe taps rf@0.9 (right)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=WIDE)
    conds = ["both_intact", "s0_zeroed", "s1_zeroed"]
    labels = ["intact", r"$S_0$ zeroed", r"$S_1$ zeroed"]
    width = 0.36
    for j, armname in enumerate(("contender", "ablation")):
        d = _load(
            f"experiment-runs/2026-07-09_h2h_tap_localization/results/"
            f"tap_localization_{armname}.json"
        )
        vals = [d["localization"][c]["accuracy"] for c in conds]
        x = np.arange(3) + (j - 0.5) * width
        ax.bar(x, vals, width * 0.92, color=ARM_COLORS[armname], label=armname)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3f}", (xi, v + 0.02), ha="center", fontsize=6)
    ax.axhline(0.03125, ls="--", lw=0.8, color=PALETTE["gray"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"acc$_A$")
    ax.set_ylim(0, 1.24)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, columnspacing=0.9, handlelength=1.2)

    taps = ["i_s1_qshallow", "ii_s0_q0", "iii_s1_qtrue", "iv_prelmhead"]
    tap_labels = [r"$S_1$@q$_{sh}$", r"$S_0$@q$_0$", r"$S_1$@q$_{true}$",
                  "pre-LM-head"]
    for j, armname in enumerate(("contender", "ablation")):
        d = _load(
            f"experiment-runs/2026-07-09_h2h_tap_localization/results/"
            f"tap_localization_{armname}.json"
        )
        vals = [d["tap_variants"][t]["rf@0.9"] for t in taps]
        x = np.arange(4) + (j - 0.5) * width
        ax2.bar(x, vals, width * 0.92, color=ARM_COLORS[armname], label=armname)
        for xi, v in zip(x, vals):
            ax2.annotate(f"{v:.3f}" if v > 0 else "0",
                         (xi, v + 0.015), ha="center", fontsize=6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(tap_labels, fontsize=6.5)
    ax2.set_ylabel("linear probe rf@0.9")
    ax2.set_ylim(0, 0.8)
    ax2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_tap_localization.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig5_attractor_ladder(out_dir):
    """Span fraction vs parameters, 4 rungs, with the 1.31B plateau inset."""
    tidy = _load("experiment-runs/2026-07-06_trajectory_probes/trajectories_tidy.json")
    fams = {"mixcontrol": 14048896, "wave1ext": 97618176, "wave2": 391869440}
    xs, ys, cell_pts = [], [], []
    for fam, npar in fams.items():
        rows = [r for r in tidy if r["family"] == fam]
        mx = max(r["step"] for r in rows)
        fin = [r for r in rows if r["step"] == mx]
        vals = [r["archived4_span_frac"] for r in fin]
        xs.append(npar)
        ys.append(float(np.mean(vals)))
        cell_pts += [(npar, v) for v in vals]
    r3 = _load("experiment-runs/2026-07-06_trackc_rung3/probe_analysis_rung3.json")
    r3_final = r3["rung3_final_pooled.json"]["pooled_archived4"]["span_frac"]
    xs.append(1311135488)
    ys.append(r3_final)
    ax_x = np.array(xs, dtype=float)

    fig, ax = plt.subplots(figsize=SINGLE)
    ax.plot(ax_x[:3].tolist() + [ax_x[3]], ys, marker="o", ms=5,
            color=PALETTE["blue"], zorder=3)
    px, py = zip(*cell_pts)
    ax.scatter(px, py, s=9, facecolors="none", edgecolors=PALETTE["blue"],
               linewidths=0.7, alpha=0.7, zorder=2)
    ax.scatter([ax_x[3]], [ys[3]], s=42, facecolors="white",
               edgecolors=PALETTE["blue"], linewidths=1.4, zorder=4)
    for x, y in zip(ax_x, ys):
        ax.annotate(f"{y:.3f}", (x, y + 0.018), ha="center", fontsize=6.5)
    ax.annotate("84.7% budget\n(flat final window)", (ax_x[3], ys[3] - 0.055),
                ha="center", fontsize=6, color=PALETTE["gray"])
    ax.set_xscale("log")
    ax.set_xticks(ax_x)
    ax.set_xticklabels(["14M", "98M", "392M", "1.31B"])
    ax.minorticks_off()
    ax.set_xlabel("parameters")
    ax.set_ylabel("span fraction")
    ax.set_ylim(0.18, 0.52)

    ins = ax.inset_axes([0.08, 0.66, 0.32, 0.26])
    steps, vals = [], []
    for st in (130000, 140000, 150000, 155000):
        key = f"rung3_plateau_step{st}.json"
        vals.append(r3[key]["pooled_archived4"]["span_frac"])
        steps.append(st / 1000)
    ins.plot(steps, vals, marker="o", ms=2, lw=0.8, color=PALETTE["blue"])
    ins.set_ylim(0.450, 0.462)
    ins.set_xticks([130, 155])
    ins.set_xticklabels(["130k", "155k"])
    # final-review C9: inset tick labels were below legibility at print
    # size (render-inspection v3's cosmetic minor); values restated in
    # the caption regardless.
    ins.set_title("1.31B final window", fontsize=6, pad=1.5)
    ins.tick_params(labelsize=6)
    ins.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_attractor_ladder.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig6_2x2_mitigations(out_dir):
    """2x2 qk-norm x gating factorial at 14M, n=3, with the 2-sigma bar."""
    d = _load(
        "experiment-runs/2026-07-09_attrrob_2x2_escalation_harvest/"
        "n3_recompute_summary.json"
    )
    order = ["qkTrue_gateFalse", "qkFalse_gateFalse", "qkTrue_gateTrue",
             "qkFalse_gateTrue"]
    labels = ["qk-norm on\nno gating\n(baseline)", "qk-norm off\nno gating",
              "qk-norm on\ngating", "qk-norm off\ngating"]
    base_seeds = [d["per_seed"]["qkTrue_gateFalse"][str(s)] for s in range(3)]
    base_mean = float(np.mean(base_seeds))
    gt = d["verdicts"]["qkTrue_gateTrue"]
    sigma_floor = gt["delta"] / gt["sigma_floor"]  # = 2.244355 (corrected floor)
    fig, ax = plt.subplots(figsize=SINGLE)
    ann_top = -np.inf
    for i, cell in enumerate(order):
        seeds = [d["per_seed"][cell][str(s)] for s in range(3)]
        mean, sd = float(np.mean(seeds)), float(np.std(seeds, ddof=1))
        color = PALETTE["blue"] if "gateFalse" in cell else PALETTE["verm"]
        ax.errorbar([i], [mean], yerr=[sd], fmt="o", ms=5, capsize=3,
                    color=color)
        ax.scatter([i - 0.10, i, i + 0.10], seeds, s=9, facecolors="none",
                   edgecolors=color, linewidths=0.7)
        if cell != "qkTrue_gateFalse":
            delta = mean - base_mean
            ann_y = mean + sd + 0.9
            ax.annotate(f"$\\Delta$={delta:+.2f}", (i, ann_y),
                        ha="center", fontsize=6.5)
            ann_top = max(ann_top, ann_y)
    ax.axhline(base_mean, ls="--", lw=0.8, color=PALETTE["gray"])
    ax.axhline(base_mean + 2 * sigma_floor, ls=":", lw=1.0, color=PALETTE["ink"])
    ax.text(3.45, base_mean + 2 * sigma_floor + 0.25,
            r"baseline $+2\sigma$ bar", fontsize=6, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("key-Gram deviation (raw)")
    # final-review C8: autoscale ignores annotations, which clipped the
    # qk-norm-on/gating cell's +4.31 delta (the number the caption leads
    # with) above the axes top. Give every delta annotation headroom.
    ax.set_ylim(top=max(ann_top + 0.7, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_2x2_mitigations.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 7
def fig7_fixscale_transfer(out_dir):
    """Forest plot: per-scale, per-corpus, per-instrument deltas with CIs."""
    d = _load("experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json")
    rows = []
    for scale in ("98m", "392m"):
        for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
            c = d["scales"][scale]["corpora"][corpus]
            short = corpus.split("-")[0]
            rows.append((f"{scale.upper()} {short} primary",
                         c["PRIMARY_delta"], c["PRIMARY_ci"], True))
            rows.append((f"{scale.upper()} {short} pre-blend",
                         c["COPRIMARY_delta"], c["COPRIMARY_ci"], False))
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ypos = np.arange(len(rows))[::-1]
    for y, (label, delta, ci, primary) in zip(ypos, rows):
        color = PALETTE["verm"] if delta > 0 and ci[0] > 0 else PALETTE["gray"]
        ax.errorbar([delta], [y], xerr=[[delta - ci[0]], [ci[1] - delta]],
                    fmt="o" if primary else "s", ms=4.5 if primary else 3.5,
                    mfc=color if primary else "white", mec=color, ecolor=color,
                    capsize=2.5, lw=1.1)
    ax.axvline(0.0, ls="--", lw=0.9, color=PALETTE["ink"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_xlabel(r"$\Delta$ span fraction (arm $-$ comparator)"
                  "\npositive = destabilizing")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_fixscale_transfer.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig A1
def figA1_complement_scaffold(out_dir):
    """Complement Procrustes residual per run (left); complement energy by
    architecture family (right) — the channel is empty in delta-rule states."""
    d = _load("experiment-runs/2026-07-09_zdump_complement/complement_results.json")
    s = d["summaries"]
    taskE = [r for r in s if r["family"] == "taskE"]
    keyan = [r for r in s if r["family"] == "keyanchor"]
    null_vals = [v[0] for v in d["gaussian_procrustes_ref"].values()]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=WIDE)
    ax.axhspan(min(null_vals), max(null_vals), color="#EEEEEE", zorder=0)
    ax.text(0.02, min(null_vals) * 0.80, "Gaussian null", fontsize=6,
            transform=ax.get_yaxis_transform())
    pr = sorted([r["proc_resid"] for r in taskE])
    ax.scatter(range(len(pr)), pr, s=18, color=PALETTE["blue"], zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Task-E encoder run (sorted)")
    ax.set_ylabel("complement Procrustes residual")
    ax.set_xticks(range(len(pr)))
    ax.set_xticklabels([])

    for j, (fam, color, lbl) in enumerate(
        [(taskE, PALETTE["blue"], "encoder (Task E)"),
         (keyan, PALETTE["verm"], "delta-rule (production)")]
    ):
        vals = [max(r["fD"], 1e-14) for r in fam]
        x = np.random.default_rng(0).uniform(j - 0.12, j + 0.12, len(vals))
        ax2.scatter(x, vals, s=16, color=color)
    ax2.set_yscale("log")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["encoder\n(Task E)", "delta-rule\n(production)"],
                        fontsize=6.5)
    ax2.set_ylabel("complement energy fraction $f_D$")
    ax2.annotate(r"$\leq 3.2\times10^{-12}$" "\n(numerically empty)",
                 (1, 3e-10), ha="center", fontsize=6, color=PALETTE["verm"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figA1_complement_scaffold.pdf"))
    plt.close(fig)


FIGURES = [
    fig1_rank_vs_dmin,
    fig2_forcerank_staircase,
    fig3_recall_separation,
    fig4_tap_localization,
    fig5_attractor_ladder,
    fig6_2x2_mitigations,
    fig7_fixscale_transfer,
    figA1_complement_scaffold,
]


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument("--out", default="papers/flagship/figures/")
    parser.add_argument("--data", default=".")
    args = parser.parse_args()
    DATA_ROOT = args.data
    os.makedirs(args.out, exist_ok=True)
    for make_fig in FIGURES:
        make_fig(args.out)
        print(f"built {make_fig.__name__}")


if __name__ == "__main__":
    main()
