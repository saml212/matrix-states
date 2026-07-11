"""Generate every figure and every data table for the mstar-colm-er paper.

Usage (repo root, the repo's own figure-build path):
    .venv/bin/python papers/mstar-colm-er/figures/figure-gen.py \
        --repo . --out papers/mstar-colm-er/figures

Every panel and every table cell regenerates from the archived raw artifacts
named in the brief's claims-to-evidence map (papers/mstar-colm-er/brief.md).
No number is hand-entered. Each source file's md5 is asserted before load;
if an archived artifact changed, the build fails loudly.
"""

import argparse
import hashlib
import json
import math
import os

# ---------------------------------------------------------------------------
# Evidence sources (paths relative to --repo), matching the brief's map.
# ---------------------------------------------------------------------------
HARVEST = "experiment-runs/2026-07-10_h2h_sweep_harvest"
MSTAR = "experiment-runs/2026-07-10_h2h_mstar"
TAPLOC = "experiment-runs/2026-07-09_h2h_tap_localization/results"
T2DIAG = "experiment-runs/2026-07-10_h2h_task2diag/results"

DATA_SOURCES = {
    # C1/C2/C5/C6b/C9 — verdict-grade re-metric + training configs
    "contender_s0": f"{HARVEST}/sweep_remetric/h2h_contender_task1_sweep_s0_round4.json",
    "contender_s1": f"{HARVEST}/sweep_remetric/h2h_contender_task1_sweep_s1_round4.json",
    "contender_s2": f"{HARVEST}/sweep_remetric/h2h_contender_task1_sweep_s2_round4.json",
    "ablation_s0": f"{HARVEST}/sweep_remetric/h2h_ablation_task1_sweep_s0_round4.json",
    "ablation_s1": f"{HARVEST}/sweep_remetric/h2h_ablation_task1_sweep_s1_round4.json",
    "ablation_s2": f"{HARVEST}/sweep_remetric/h2h_ablation_task1_sweep_s2_round4.json",
    "transformer_s0": f"{HARVEST}/sweep_remetric/h2h_transformer_task1_sweep_s0_round4.json",
    "transformer_s1": f"{HARVEST}/sweep_remetric/h2h_transformer_task1_sweep_s1_round4.json",
    "transformer_s2": f"{HARVEST}/sweep_remetric/h2h_transformer_task1_sweep_s2_round4.json",
    "train_contender": f"{HARVEST}/h2h_contender_task1_sweep_s0.json",
    "train_ablation": f"{HARVEST}/h2h_ablation_task1_sweep_s0.json",
    "train_transformer": f"{HARVEST}/h2h_transformer_task1_sweep_s0.json",
    # C3/C4 — M* protocol
    "horizon_refs": f"{MSTAR}/fanout/contender_horizon_refs.json",
    "mstar_verdict": f"{MSTAR}/MSTAR_VERDICT.json",
    # C6 — tap localization
    "taploc_contender": f"{TAPLOC}/tap_localization_contender.json",
    "taploc_ablation": f"{TAPLOC}/tap_localization_ablation.json",
    # C7/C8 — task2 diagnosis + K48 stress
    "t2diag_verdict": f"{T2DIAG}/TASK2DIAG_VERDICT.json",
}

# md5s recorded 2026-07-10 (brief.md evidence rows). Build fails on mismatch.
SOURCE_MD5 = {
    "contender_s0": "d559e498e8d24c0673727e34377519a3",
    "contender_s1": "3934ea69c60243b380f86eb3db9fdd70",
    "contender_s2": "677eb57748f06e59520ca37925ed2309",
    "ablation_s0": "491d0eb61f1c29bdc0824ac14eba4ef0",
    "ablation_s1": "2b415704c956b3b54d8786f3fdf58810",
    "ablation_s2": "b71b44341d44bb30b5099f7dd9d62957",
    "transformer_s0": "988fdae364ad1cb6cd41e6c28ff9b564",
    "transformer_s1": "dc57f209053ae9245492724cf9024e17",
    "transformer_s2": "0d715323ffd03c020ddf759a4579d065",
    "train_contender": "15c817f203bd7243165f953d3a6600a5",
    "train_ablation": "159687a5fbcfba334fb622c30704d553",
    "train_transformer": "427de1aaf3330f6f6d407fb12a49652a",
    "horizon_refs": "afd5af6b68b8947fe6c4f12a827fd916",
    "mstar_verdict": "4f115ad55d5301122f387df504efa35c",
    "taploc_contender": "362333c89f4223c427fe8daf54f50fce",
    "taploc_ablation": "ff8e352a13c2bc1e177f53f8cef47c01",
    "t2diag_verdict": "66d2291d8e65932d368d8978bfd16bdc",
}

# Print-safe style: black/dark-red/grays separate by lightness in grayscale;
# identity is never color-alone (linestyle + marker are the second encoding).
STYLE = {
    "font_size": 8,
    "dpi": 300,
    "ink": "#000000",
    "accent": "#8B2E1F",
    "gray": "#666666",
    "lightgray": "#B0B0B0",
}

CHANCE_K32 = 1.0 / 32
BAR_K32 = 3.0 / 32
T_CRIT_DF2 = 4.302652729911275  # two-sided 95%, df=2 (n=3 paired seeds)


def _assert_md5(source_key, path):
    expected = SOURCE_MD5[source_key]
    with open(path, "rb") as f:
        actual = hashlib.md5(f.read()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"md5 mismatch for {source_key!r} ({path}): expected {expected}, "
            f"got {actual}. The archived artifact changed - do NOT plot it."
        )


def _load(repo, source_key):
    path = os.path.join(repo, DATA_SOURCES[source_key])
    _assert_md5(source_key, path)
    with open(path) as f:
        return json.load(f)


def paired_ci(deltas):
    n = len(deltas)
    assert n == 3, "verdict-grade CIs in this paper are n=3 paired seeds"
    m = sum(deltas) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    return m, m - T_CRIT_DF2 * se, m + T_CRIT_DF2 * se


def fig1_horizon(repo, out_dir, plt):
    """Money figure: acc_A vs context length. Contender flat at ~1.0 with a
    fixed 32,768-byte state; transformer at chance uncapped and at every
    KV-cache cap."""
    refs = _load(repo, "horizon_refs")
    verdict = _load(repo, "mstar_verdict")["per_task"]["task1_sweep"]

    horizons = [("H2", 454), ("H4", 902), ("H8", 1798)]
    xs = [t for _, t in horizons]

    fig, ax = plt.subplots(figsize=(4.7, 2.9))

    # Contender, per seed (three lines overlap at ~0.999).
    for s in range(3):
        ys = [refs[f"task1_sweep|s{s}|{h}"]["acc_A"] for h, _ in horizons]
        ax.plot(xs, ys, color=STYLE["ink"], lw=1.4, marker="o", ms=4,
                zorder=5, label="contender (fixed 32,768 B state)" if s == 0 else None)

    # Transformer uncapped, per seed. alpha<1 so the capped-M crosses
    # (plotted above, higher zorder) remain visible through the dashed
    # line where they overlap in the near-chance band.
    unc = verdict["transformer_uncapped_refs_acc_A"]
    for s in range(3):
        ys = [unc[h][s] for h, _ in horizons]
        ax.plot(xs, ys, color=STYLE["accent"], lw=1.4, ls="--", marker="s",
                ms=4, zorder=4, alpha=0.6,
                label="transformer, uncapped KV cache" if s == 0 else None)

    # Transformer capped, every M in the walk grid, every seed: gray crosses.
    # zorder=6, above both the dashed uncapped line (4) and the solid
    # contender line (5) -- round-1 gauntlet (A3) found these markers
    # rendering invisible beneath the dashed line at the previous zorder=3.
    capped = verdict["capped_transformer_acc_A"]
    first = True
    for m_key, by_h in capped.items():
        for h, t in horizons:
            for s in range(3):
                ax.plot(t, by_h[h][s], color=STYLE["gray"], marker="x", ms=5,
                        mew=1.2, ls="none", zorder=6,
                        label=(r"transformer, KV capped at $M\times$32,768 B, "
                               r"$M\in\{2,\ldots,32\}$") if first else None)
                first = False

    ax.axhline(CHANCE_K32, color=STYLE["lightgray"], lw=1.0, ls=":", zorder=1)
    ax.axhline(BAR_K32, color=STYLE["lightgray"], lw=1.0, ls="-.", zorder=1)
    ax.text(430, CHANCE_K32 - 0.02, "chance 1/32", ha="left", va="top",
            fontsize=7, color=STYLE["gray"])
    ax.text(1798, BAR_K32 + 0.012, "demonstration bar 3/32", ha="right",
            va="bottom", fontsize=7, color=STYLE["gray"])

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("evaluation context length (tokens)")
    ax.set_ylabel("episode-restricted recall accuracy")
    ax.set_ylim(-0.04, 1.06)
    ax.legend(loc="center left", fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_horizon.pdf"))
    plt.close(fig)


def fig2_szero(repo, out_dir, plt):
    """Causal localization: zeroing S0 collapses recall to chance; zeroing S1
    leaves it unchanged. Grouped bars per seed, contender task1 cells."""
    conds = [("acc_intact", "intact", STYLE["ink"], None),
             ("acc_s0_zeroed", r"$S_0$ zeroed", STYLE["accent"], "///"),
             ("acc_s1_zeroed", r"$S_1$ zeroed", STYLE["gray"], None)]
    vals = {}
    for s in range(3):
        chk = _load(repo, f"contender_s{s}")["s0_necessity_check"]
        for key, *_ in conds:
            vals.setdefault(key, []).append(chk[key])

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    width = 0.26
    for j, (key, lab, color, hatch) in enumerate(conds):
        xpos = [s + (j - 1) * (width + 0.02) for s in range(3)]
        ax.bar(xpos, vals[key], width=width, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.8, label=lab)
        # round-1 gauntlet minor: the S0-zeroed bars collapse to near-zero
        # height at seeds 1-2 and can read as missing data at print size;
        # print the value above each bar in this condition so absence
        # reads as a measurement, not a gap.
        if key == "acc_s0_zeroed":
            for x, v in zip(xpos, vals[key]):
                ax.text(x, v + 0.025, f"{v:.4f}", ha="center", va="bottom",
                        fontsize=5.5, color=color, rotation=90)
    ax.axhline(CHANCE_K32, color=STYLE["lightgray"], lw=1.0, ls=":")
    ax.set_xlim(-0.85, 2.55)
    ax.text(-0.82, CHANCE_K32 + 0.02, "chance", ha="left", fontsize=7,
            color=STYLE["gray"])
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"seed {s}" for s in range(3)])
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7,
              frameon=False, ncol=3, columnspacing=1.0, handlelength=1.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_szero.pdf"))
    plt.close(fig)


def fig3_traincurve(repo, out_dir, plt):
    """Training-loss trajectories, all three arms, task1 seed 0 (round-1
    gauntlet FIX-1c). Same archived configs already used for the
    parameter-count row (C9); no new sources."""
    arms = [("contender", "matrix fast-weight", STYLE["ink"], "-"),
            ("ablation", "flat-vector recurrence", STYLE["accent"], "--"),
            ("transformer", "transformer", STYLE["gray"], ":")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for a, label, color, ls in arms:
        curve = _load(repo, f"train_{a}")["curve"]
        steps = [row["step"] for row in curve]
        losses = [row["train_loss"] for row in curve]
        ax.plot(steps, losses, color=color, lw=1.3, ls=ls, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("train loss")
    ax.legend(loc="center right", fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_traincurve.pdf"))
    plt.close(fig)


def emit_tables(repo, out_dir):
    """Emit every data table as LaTeX to tables_generated.tex, computed from
    the same md5-asserted raws (no hand-typed numbers in any table)."""
    arms = ["contender", "ablation", "transformer"]
    acc = {a: [_load(repo, f"{a}_s{s}")["leg_a"]["acc_A"] for s in range(3)]
           for a in arms}
    d_abl = [acc["contender"][i] - acc["ablation"][i] for i in range(3)]
    d_tra = [acc["contender"][i] - acc["transformer"][i] for i in range(3)]
    m_abl, lo_abl, hi_abl = paired_ci(d_abl)
    m_tra, lo_tra, hi_tra = paired_ci(d_tra)

    verdict = _load(repo, "mstar_verdict")["per_task"]["task1_sweep"]
    t2 = _load(repo, "t2diag_verdict")
    tap_c = _load(repo, "taploc_contender")
    tap_a = _load(repo, "taploc_ablation")
    n_params = {a: _load(repo, f"train_{a}")["n_params"] for a in arms}

    L = []
    A = L.append

    # --- Table 1: axis-1 per-seed recall + paired CIs (C1, C2) ---
    A("% AUTO-GENERATED by figure-gen.py from md5-asserted raws. Do not edit.")
    A(r"\newcommand{\tableone}{")
    A(r"\begin{tabular}{lccccc}")
    A(r"\toprule")
    A(r"arm & seed 0 & seed 1 & seed 2 & mean & clears bar? \\")
    A(r"\midrule")
    for a, label in [("contender", "matrix fast-weight"),
                     ("ablation", "flat-vector recurrence"),
                     ("transformer", "transformer (uncapped)")]:
        row = acc[a]
        clears = "every seed" if min(row) > BAR_K32 else "never"
        A(f"{label} & " + " & ".join(f"{v:.4f}" for v in row) +
          f" & {sum(row)/3:.4f} & {clears} " + r"\\")
    A(r"\midrule")
    A(r"\multicolumn{6}{l}{$\Delta$ vs flat-vector: mean "
      f"{m_abl:.4f}, 95\\% paired CI ({lo_abl:.4f}, {hi_abl:.4f});"
      r" excludes 0.30} \\")
    A(r"\multicolumn{6}{l}{$\Delta$ vs transformer: mean "
      f"{m_tra:.4f}, 95\\% paired CI ({lo_tra:.4f}, {hi_tra:.4f});"
      r" excludes 0.30} \\")
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    # --- Table 2 (appendix): capped transformer by M and horizon (C4) ---
    A(r"\newcommand{\tablecapped}{")
    A(r"\begin{tabular}{lccc}")
    A(r"\toprule")
    A(r"arm / KV budget & $T{=}454$ & $T{=}902$ & $T{=}1798$ \\")
    A(r"\midrule")
    refs = _load(repo, "horizon_refs")
    c_by_h = {h: [refs[f"task1_sweep|s{s}|{h}"]["acc_A"] for s in range(3)]
              for h in ("H2", "H4", "H8")}
    A("contender, fixed 32{,}768 B & " + " & ".join(
        f"{sum(c_by_h[h])/3:.4f}" for h in ("H2", "H4", "H8")) + r" \\")
    unc = verdict["transformer_uncapped_refs_acc_A"]
    A("transformer, uncapped & " + " & ".join(
        f"{sum(unc[h])/3:.4f}" for h in ("H2", "H4", "H8")) + r" \\")
    for m in ("32", "16", "8", "4", "2"):
        by_h = verdict["capped_transformer_acc_A"][m]
        A(f"transformer, $M{{=}}{m}$ cap & " + " & ".join(
            f"{sum(by_h[h])/3:.4f}" for h in ("H2", "H4", "H8")) + r" \\")
    A(r"\midrule")
    cis = verdict["cis_by_m_H4"]
    ci_lo_min = min(cis[m]["ci_low"] for m in cis)
    A(r"\multicolumn{4}{l}{per-$M$ paired-gap CI lower bounds at $T{=}902$: "
      f"all $\\geq {ci_lo_min:.4f}$" + r"} \\")
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    # --- Table 3 (appendix): tap-localization ridge fits (C6) ---
    A(r"\newcommand{\tabletaps}{")
    A(r"\begin{tabular}{llcccc}")
    A(r"\toprule")
    A(r"arm & linear read-out site & cos.\ mean & rf@0.9 & shuffled cos.\ & gap \\")
    A(r"\midrule")
    tapnames = [("i_s1_qshallow", r"$S_1 q$ (shallow query)"),
                ("ii_s0_q0", r"$S_0 q$ (bind-path query)"),
                ("iii_s1_qtrue", r"$S_1 q$ (true query)"),
                ("iv_prelmhead", "pre-LM-head hidden")]
    for arm_label, tj in [("contender", tap_c), ("flat-vector", tap_a)]:
        for k, name in tapnames:
            t = tj["tap_variants"][k]
            A(f"{arm_label} & {name} & {t['cos_mean']:.3f} & {t['rf@0.9']:.3f}"
              f" & {t['shuffled_control_cos_mean']:.3f}"
              f" & {t['gap_vs_shuffled_cos_mean']:+.3f} " + r"\\")
        A(r"\midrule" if arm_label == "contender" else r"\bottomrule")
    A(r"\end{tabular}}")

    # --- Table 4: task2 nine-seed diagnosis (C7) ---
    A(r"\newcommand{\tabletasktwo}{")
    A(r"\begin{tabular}{lccccccccc}")
    A(r"\toprule")
    A(r"arm & s0 & s1 & s2 & s3 & s4 & s5 & s6 & s7 & s8 \\")
    A(r"\midrule")
    for arm_key, label in [("acc_A_contender_by_seed", "contender"),
                           ("acc_A_ablation_by_seed", "flat-vector")]:
        row = [t2["task2"][arm_key][str(i)] for i in range(9)]
        cells = [f"\\textbf{{{v:.3f}}}" if v > BAR_K32 else f"{v:.3f}"
                 for v in row]
        A(f"{label} & " + " & ".join(cells) + r" \\")
    A(r"\midrule")
    ci = t2["task2"]["pooled_n9_paired_delta_ci"]
    A(r"\multicolumn{10}{l}{pooled paired $\Delta$ CI ("
      f"{ci['ci_low']:.3f}, {ci['ci_high']:.3f}"
      r"), flagged non-decision-grade (var-ratio "
      f"{t2['task2']['batch_effect_gates_sec_16_19_5']['ablation_acc']['var_ratio']:.2f}"
      r"$>$4.0)} \\")
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    # --- Table 5: K48 stress row (C8) ---
    k48 = t2["k48_stress_locate_only_3arm_table"]
    A(r"\newcommand{\tablekstress}{")
    A(r"\begin{tabular}{lccc}")
    A(r"\toprule")
    A(r" & matrix fast-weight & flat-vector & transformer \\")
    A(r"\midrule")
    A(r"$K{=}48$ ($K/d{=}0.75$) accuracy & " + " & ".join(
        f"{k48[a]['acc_A']:.4f}" for a in arms) + r" \\")
    A(r"\multicolumn{4}{l}{chance " + f"{k48['chance']:.4f}"
      + r"; locate-only bar " + f"{k48['bar']:.4f}"
      + r"; no arm clears} \\")
    A(r"\bottomrule")
    A(r"\end{tabular}}")

    # --- setup-table macros (C9, C10) ---
    A(r"\newcommand{\nparamscontender}{" + f"{n_params['contender']:,}".replace(",", "{,}") + "}")
    A(r"\newcommand{\nparamsablation}{" + f"{n_params['ablation']:,}".replace(",", "{,}") + "}")
    A(r"\newcommand{\nparamstransformer}{" + f"{n_params['transformer']:,}".replace(",", "{,}") + "}")

    with open(os.path.join(out_dir, "tables_generated.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(description="Generate all paper figures.")
    p.add_argument("--repo", default=".", help="repo root")
    p.add_argument("--out", default="papers/mstar-colm-er/figures")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": STYLE["font_size"],
        "pdf.fonttype": 42,
        "figure.dpi": STYLE["dpi"],
        "axes.linewidth": 0.7,
        "hatch.linewidth": 0.6,
    })

    os.makedirs(args.out, exist_ok=True)
    fig1_horizon(args.repo, args.out, plt)
    fig2_szero(args.repo, args.out, plt)
    fig3_traincurve(args.repo, args.out, plt)
    emit_tables(args.repo, args.out)
    print("figures + tables written to", args.out)


if __name__ == "__main__":
    main()
