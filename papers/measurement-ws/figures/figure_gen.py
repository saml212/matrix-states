#!/usr/bin/env python3
"""Figure generator for papers/measurement-ws (the instrument-methodology paper).

Every panel loads its data from an archived raw artifact named in the paper
brief's claims-to-evidence map and asserts the artifact's md5 before plotting
(paper-skill discipline: figures regenerate from raws, never from hand-entered
numbers; a changed artifact fails the build loudly).

Run from the repo root:
    .venv/bin/python papers/measurement-ws/figures/figure_gen.py \
        --out papers/measurement-ws/figures --repo .
"""

import argparse
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Source manifest: path (relative to repo root) -> required md5.
# These md5s were recorded 2026-07-10 against the committed archives and are
# also listed in papers/measurement-ws/brief.md (evidence rows X2 and W4).
# ---------------------------------------------------------------------------
SOURCE_MD5 = {
    "experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/"
    "crosscheck_lens_verdict_output.json": "f26a769d5c263af224c91d39bd83710b",
    "experiment-runs/2026-07-09_h2h_tap_localization/results/"
    "tap_localization_SUMMARY.json": "0e73ee283b7208db10af5588fe1c6713",
}


def load_verified(repo: str, relpath: str):
    """Load a JSON artifact only after its md5 matches the manifest."""
    path = os.path.join(repo, relpath)
    with open(path, "rb") as f:
        blob = f.read()
    digest = hashlib.md5(blob).hexdigest()
    want = SOURCE_MD5[relpath]
    assert digest == want, (
        f"md5 mismatch for {relpath}: got {digest}, manifest says {want}. "
        "The archived artifact changed; refusing to plot stale/mismatched data."
    )
    return json.loads(blob.decode("utf-8"))


# Colorblind-safe pair (Okabe-Ito blue / vermillion).
C_CONV = "#0072B2"
C_NONCONV = "#D55E00"
C_ABLATION = "#999999"


def fig1_lens_contradiction(repo: str, out: str) -> None:
    """62-cell scatter: primary vs crosscheck recovered_frac@0.9 at far depth.

    Data: the flat 62-cell table persisted by the S2.32 re-metric
    (crosscheck_lens_verdict_output.json). Convergence split at
    final_loss < 0.02, the natural gap in the observed losses (converged
    cells sit at <= 0.0117, non-converged at >= 0.069).
    """
    data = load_verified(
        repo,
        "experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/"
        "crosscheck_lens_verdict_output.json",
    )
    rows = data["flat_per_cell_table"]
    assert len(rows) == 62, f"expected 62 cells, got {len(rows)}"

    conv = [r for r in rows if r["final_loss"] < 0.02]
    nonconv = [r for r in rows if r["final_loss"] >= 0.02]

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot([0, 1], [0, 1], color="0.8", lw=0.8, zorder=1)
    ax.scatter(
        [r["primary_rf90_far64"] for r in nonconv],
        [r["crosscheck_rf90_far64"] for r in nonconv],
        s=26,
        facecolors="none",
        edgecolors=C_NONCONV,
        linewidths=1.1,
        label=f"not converged (final loss $\\geq$ 0.02), n={len(nonconv)}",
        zorder=2,
    )
    ax.scatter(
        [r["primary_rf90_far64"] for r in conv],
        [r["crosscheck_rf90_far64"] for r in conv],
        s=30,
        color=C_CONV,
        label=f"converged (final loss $<$ 0.02), n={len(conv)}",
        zorder=3,
    )
    ax.set_xlabel("primary lens: recovered fraction @0.9, depth 64")
    ax.set_ylabel("crosscheck lens: recovered\nfraction @0.9, depth 64")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center right", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_lens_contradiction.pdf"))
    plt.close(fig)

    # Consistency print for the caption's headline counts: the >=0.5-gap
    # count is computed over ALL rows so the "every contradicting cell is
    # a converged cell" caption claim is checked here, not assumed.
    off_diag_all = [
        r for r in rows if r["crosscheck_rf90_far64"] - r["primary_rf90_far64"] >= 0.5
    ]
    off_diag_conv = [r for r in off_diag_all if r["final_loss"] < 0.02]
    assert len(off_diag_all) == len(off_diag_conv), (
        "a non-converged cell shows a >=0.5 lens gap; caption claim is wrong"
    )
    print(
        f"[fig1] cells={len(rows)} converged={len(conv)} "
        f"contradiction(>=0.5 gap)={len(off_diag_all)} (all converged) "
        f"nonconv_max_xcheck={max(r['crosscheck_rf90_far64'] for r in nonconv):.2f}"
    )


def fig2_tap_localization(repo: str, out: str) -> None:
    """Two panels from the tap-localization run (h2h S1.30 Tables 1-2)."""
    data = load_verified(
        repo,
        "experiment-runs/2026-07-09_h2h_tap_localization/results/"
        "tap_localization_SUMMARY.json",
    )
    arms = ["contender", "ablation"]
    conditions = ["both_intact", "s0_zeroed", "s1_zeroed"]
    cond_labels = ["intact", "$S_0$ zeroed", "$S_1$ zeroed"]
    taps = ["i_s1_qshallow", "ii_s0_q0", "iii_s1_qtrue", "iv_prelmhead"]
    tap_labels = [
        "(i) $S_1$ tap\n(deployed)",
        "(ii) $S_0$ tap",
        "(iii) $S_1$, true\nquery path",
        "(iv) pre-LM-head\nhidden",
    ]
    chance = data["contender"]["chance_episode"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    width = 0.36
    for k, arm in enumerate(arms):
        acc = [data[arm]["localization"][c]["accuracy"] for c in conditions]
        xs = [i + (k - 0.5) * width for i in range(len(conditions))]
        ax1.bar(
            xs,
            acc,
            width=width,
            color=C_CONV if arm == "contender" else C_ABLATION,
            label=arm,
        )
    ax1.axhline(chance, color=C_NONCONV, lw=1.0, ls="--")
    ax1.text(2.52, chance + 0.02, "chance", color=C_NONCONV, fontsize=7, ha="right")
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(cond_labels, fontsize=8)
    ax1.set_ylabel("episode-restricted recall accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) causal state zeroing", fontsize=9)
    ax1.legend(fontsize=7, frameon=False, loc="upper right")

    for k, arm in enumerate(arms):
        rf = [data[arm]["tap_variants"][t]["rf@0.9"] for t in taps]
        xs = [i + (k - 0.5) * width for i in range(len(taps))]
        ax2.bar(
            xs,
            rf,
            width=width,
            color=C_CONV if arm == "contender" else C_ABLATION,
            label=arm,
        )
    ax2.set_xticks(range(len(taps)))
    ax2.set_xticklabels(tap_labels, fontsize=7)
    ax2.set_ylabel("ridge probe rf@0.9")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) linear decodability by tap site", fontsize=9)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_tap_localization.pdf"))
    plt.close(fig)

    print(
        "[fig2] contender intact/s0/s1 = "
        + "/".join(
            f"{data['contender']['localization'][c]['accuracy']:.4f}"
            for c in conditions
        )
        + f" | tap iv rf@0.9 contender={data['contender']['tap_variants']['iv_prelmhead']['rf@0.9']:.3f}"
        f" ablation={data['ablation']['tap_variants']['iv_prelmhead']['rf@0.9']:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig1_lens_contradiction(args.repo, args.out)
    fig2_tap_localization(args.repo, args.out)
    print("[figure_gen] done")


if __name__ == "__main__":
    main()
