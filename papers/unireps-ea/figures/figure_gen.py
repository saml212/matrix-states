"""Generate every figure for the UniReps 2026 EA ("dimension, not solvability")
from archived raw result JSONs — one versioned script, md5-asserted sources.

Per the paper skill's figure discipline: every plotted value is COMPUTED from
a raw archived artifact whose md5 is asserted before loading. The TOST inset's
mean difference and margin band are recomputed here from the same per-seed
JSONs the decisional analysis used (Welch TOST re-derived at brief time and
matching the archived harvest_summary.json exactly).

Usage (from the repo root, local machine — archives are local):
    .venv/bin/python papers/unireps-ea/figures/figure_gen.py \
        --out papers/unireps-ea/figures --repo .

Figures:
    fig1_convergence.pdf — evidence rows U1/U2 (19 unconstrained sweep cells)
    fig2_razor_step.pdf  — evidence rows U3/U4 (m3fix + s3ext zero-pad cells)
"""

import argparse
import glob
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "experiment-runs/2026-07-09_capability_sweep_harvest/results"
M3FIX = "experiment-runs/2026-07-09_m3fix_harvest"
S3EXT = "experiment-runs/2026-07-09_m3fix_s3ext"

# Recorded 2026-07-09 from the archived artifacts; duplicated in the brief's
# claims-to-evidence map and the reproducibility README.
SOURCE_MD5 = {
    f"{SWEEP}/A5__unconstrained__seed0.json": "3f6492c9b41a78da6c92317fef672318",
    f"{SWEEP}/A5__unconstrained__seed1.json": "f69493f9dd4fd60809a921101ce7c18c",
    f"{SWEEP}/A5__unconstrained__seed2.json": "7b496884f765448f5bb0dd625b9462d2",
    f"{SWEEP}/A5__unconstrained__seed3.json": "78745c1bbae90a911ca79aa9275fd9a1",
    f"{SWEEP}/A5__unconstrained__seed4.json": "943e5a5e1285d45eaf248a0db477a149",
    f"{SWEEP}/A6__unconstrained__seed0.json": "8549e9b6bd72515d620197ff1613a810",
    f"{SWEEP}/A6__unconstrained__seed1.json": "e306d8cfd6edf4e573fb7a916307ac7f",
    f"{SWEEP}/A6__unconstrained__seed2.json": "24f20af8d85e5fd4c222cb3b127f970c",
    f"{SWEEP}/S3__unconstrained__seed0.json": "c00d6f8d83937ced62f9e4400cf98246",
    f"{SWEEP}/S3__unconstrained__seed1.json": "fa3e6810d5ac2130e1476247d345f963",
    f"{SWEEP}/S3__unconstrained__seed2.json": "236a30d1d76a0d0421b97b6d793db0c7",
    f"{SWEEP}/S4__unconstrained__seed0.json": "1f30c88318ba1c501d74265f47bf3113",
    f"{SWEEP}/S4__unconstrained__seed1.json": "2090631683f971390a4000cc5000589f",
    f"{SWEEP}/S4__unconstrained__seed2.json": "8d45fbb53fa4999d1decc784746900fd",
    f"{SWEEP}/S4__unconstrained__seed3.json": "2b41c525128f94a2cc78d573cbd8a614",
    f"{SWEEP}/S4__unconstrained__seed4.json": "d5c8740eefea71974c1d7d87aaadc1c3",
    f"{SWEEP}/S5__unconstrained__seed0.json": "ff9ade6ad7deb270031fae57b413be2a",
    f"{SWEEP}/S5__unconstrained__seed1.json": "b8ba517e2c6a2fe149d5703f09980fc2",
    f"{SWEEP}/S5__unconstrained__seed2.json": "74d86227f94fa6ec34f327b5f52cf082",
    f"{M3FIX}/zero_pad__A5__k_dmin__seed0.json": "9ea3cc9f3da906d9a2bb2a9d508eac9f",
    f"{M3FIX}/zero_pad__A5__k_dmin_minus_1__seed0.json": "019caf31c72090d99b1db6422151c38c",
    f"{M3FIX}/zero_pad__A5__k_dmin_plus_1__seed0.json": "6ab3999c1dd5d04ec96f96e9a6d4fd5c",
    f"{M3FIX}/zero_pad__A5__unconstrained__seed0.json": "139e38060fd659ee9648d4bde83c3bf7",
    f"{M3FIX}/zero_pad__A6__k_dmin__seed0.json": "b252a7e63b1aa79192065edbb92efe06",
    f"{M3FIX}/zero_pad__A6__k_dmin_minus_1__seed0.json": "c3641255962eecf7a250ccba05b2412e",
    f"{M3FIX}/zero_pad__A6__k_dmin_plus_1__seed0.json": "a203553383019dfe5bd01d6529610dab",
    f"{M3FIX}/zero_pad__A6__unconstrained__seed0.json": "8e847bbb3fc0c84d3a74984a7f1d6209",
    f"{M3FIX}/zero_pad__S3__k_dmin__seed0.json": "c60de1276f68dc5f2b911daff6850dee",
    f"{M3FIX}/zero_pad__S3__k_dmin_minus_1__seed0.json": "6bdd18ac270648047ee6e561f772ade6",
    f"{M3FIX}/zero_pad__S3__k_dmin_plus_1__seed0.json": "660ce9c26e05503a000f6d13aad3b0b1",
    f"{M3FIX}/zero_pad__S3__unconstrained__seed0.json": "a80bae92cbbcbfa52076cf423a244ddc",
    f"{M3FIX}/zero_pad__S4__k_dmin__seed0.json": "4925e1a7c965773107365c22bacc7967",
    f"{M3FIX}/zero_pad__S4__k_dmin_minus_1__seed0.json": "26acb7227ae2ee6e4b91f7e42fe5b0eb",
    f"{M3FIX}/zero_pad__S4__k_dmin_plus_1__seed0.json": "02c7fb35419005c0624bdde911737115",
    f"{M3FIX}/zero_pad__S4__unconstrained__seed0.json": "f9389820342f51fecd24bbb830b5f20c",
    f"{M3FIX}/zero_pad__S5__k_dmin__seed0.json": "41b7b539ca5711100cf4f40e1bda2b41",
    f"{M3FIX}/zero_pad__S5__k_dmin_minus_1__seed0.json": "61bf20dc9c95f3a1aacc5da31ed06bd9",
    f"{M3FIX}/zero_pad__S5__k_dmin_plus_1__seed0.json": "5c46ff77a2d7a2007dd131ce5b707253",
    f"{M3FIX}/zero_pad__S5__unconstrained__seed0.json": "8dbe19afd816e12f4eedae3371104898",
    f"{S3EXT}/zero_pad__S3__k_dmin__seed1.json": "9cbb21c10a4cfba64a5ec636744bbb1a",
    f"{S3EXT}/zero_pad__S3__k_dmin__seed2.json": "c9f1cd3c271a697039d065d07a712d1f",
    f"{S3EXT}/zero_pad__S3__k_dmin__seed3.json": "83de23e150d5ec5020a7c843cc6441c9",
    f"{S3EXT}/zero_pad__S3__k_dmin_minus_1__seed1.json": "a84d9486161e7d6a61c06958782d77a2",
    f"{S3EXT}/zero_pad__S3__k_dmin_minus_1__seed2.json": "525dfb94cf684d8d2306dd14bc912e49",
    f"{S3EXT}/zero_pad__S3__k_dmin_minus_1__seed3.json": "629e8c41cf019447d73bd7aa2da345ac",
    f"{S3EXT}/zero_pad__S3__k_dmin_plus_1__seed1.json": "05d0cba6d40720ce51b4ab592bf80416",
    f"{S3EXT}/zero_pad__S3__k_dmin_plus_1__seed2.json": "269c1dfb9285cbbff33a11d0401deb70",
    f"{S3EXT}/zero_pad__S3__k_dmin_plus_1__seed3.json": "247d1de636d5df36530bd3414283b8e7",
    f"{S3EXT}/zero_pad__S3__unconstrained__seed1.json": "e3896481a209439b5112b01da7d5dc88",
    f"{S3EXT}/zero_pad__S3__unconstrained__seed2.json": "81faf7e76f1f2c99d19b06ee96a12b34",
    f"{S3EXT}/zero_pad__S3__unconstrained__seed3.json": "6efa28c9eebe72695daac0a3fc95beed",
}

GROUPS = ["S3", "S4", "A5", "S5", "A6"]
DMIN = {"S3": 2, "S4": 3, "A5": 3, "S5": 4, "A6": 5}
SOLVABLE = {"S3": True, "S4": True, "A5": False, "S5": False, "A6": False}

# Okabe-Ito subset, validated CVD-safe with the dataviz six-checks validator
# (2026-07-09); identity never color-alone (marker shapes + direct labels;
# per-panel titles in fig2).
COLOR = {"S3": "#0072B2", "S4": "#E69F00", "A5": "#009E73",
         "S5": "#CC79A7", "A6": "#D55E00"}
MARKER = {"S3": "o", "S4": "s", "A5": "^", "S5": "D", "A6": "v"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
})


def _load(repo: str, relpath: str) -> dict:
    expected = SOURCE_MD5.get(relpath)
    if not expected:
        raise ValueError(f"No recorded md5 for {relpath!r} — refuse to plot an "
                         f"unasserted source.")
    path = os.path.join(repo, relpath)
    with open(path, "rb") as f:
        blob = f.read()
    actual = hashlib.md5(blob).hexdigest()
    if actual != expected:
        raise ValueError(f"md5 mismatch for {relpath}: expected {expected}, got "
                         f"{actual}. Archived artifact changed — do NOT plot it.")
    return json.loads(blob)


def _ranks(repo, group):
    files = sorted(glob.glob(os.path.join(
        repo, SWEEP, f"{group}__unconstrained__seed*.json")))
    vals = []
    for f in files:
        d = _load(repo, os.path.relpath(f, repo))
        assert d["group"] == group and d["arm"] == "unconstrained"
        vals.append(d["restricted_effective_rank"])
    return vals


def fig1_convergence(repo, out_dir):
    """Evidence U1/U2: convergence to d_min + the marquee equivalence inset.

    Main panel: restricted effective rank of every unconstrained cell vs its
    group's d_min, with the pre-registered [0.7, 1.3] x d_min band and the
    identity line. Inset: S4 vs A5 per-seed values at matched d_min = 3 with
    the observed mean difference against the pre-registered +/-0.5 rank-unit
    TOST margin.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    band_x = [1.5, 5.5]
    ax.fill_between(band_x, [0.7 * x for x in band_x], [1.3 * x for x in band_x],
                    color="#000000", alpha=0.07, lw=0,
                    label="[0.7, 1.3] $\\cdot$ $d_{\\min}$ band")
    ax.plot(band_x, band_x, color="#555555", lw=0.8, ls="--",
            label="rank = $d_{\\min}$")
    jitter = {"S3": 0.0, "S4": -0.09, "A5": 0.09, "S5": 0.0, "A6": 0.0}
    ranks = {}
    for g in GROUPS:
        vals = _ranks(repo, g)
        ranks[g] = vals
        # White (not transparent) faces on the open markers: stacked seeds
        # occlude instead of blending, so the open/filled encoding stays
        # legible at print size (round-2 render-inspection finding).
        face = COLOR[g] if SOLVABLE[g] else "white"
        ax.scatter([DMIN[g] + jitter[g]] * len(vals), vals, s=30,
                   marker=MARKER[g], facecolors=face, edgecolors=COLOR[g],
                   linewidths=0.9, zorder=3)
        lab_y = max(vals) + 0.2 if g != "S4" else min(vals) - 0.32
        ax.annotate(g, (DMIN[g] + jitter[g] * 3.2, lab_y), color=COLOR[g],
                    ha="center", fontsize=7.5, fontweight="bold")
    ax.scatter([], [], s=30, marker="o", facecolors="#333333",
               edgecolors="#333333", label="solvable (filled)")
    ax.scatter([], [], s=30, marker="^", facecolors="white",
               edgecolors="#333333", linewidths=0.9,
               label="non-solvable (open)")
    ax.set_xlabel("minimal faithful real representation dimension $d_{\\min}$")
    ax.set_ylabel("restricted effective rank")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlim(1.6, 5.6)
    ax.set_ylim(1.2, 6.9)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              borderaxespad=0.1)

    # Marquee inset: recomputed from the same raws (mean diff vs TOST margin).
    # Placed low-right with clear whitespace from the S5 markers/label
    # (round-2 render-inspection v3 crowding finding).
    ins = ax.inset_axes([0.66, 0.05, 0.33, 0.27])
    s4, a5 = ranks["S4"], ranks["A5"]
    diff = sum(s4) / len(s4) - sum(a5) / len(a5)
    ins.axvspan(-0.5, 0.5, color="#000000", alpha=0.07, lw=0)
    ins.axvline(0.0, color="#555555", lw=0.6, ls="--")
    for v in [x - sum(a5) / len(a5) for x in s4]:
        ins.plot(v, 1.0, marker=MARKER["S4"], color=COLOR["S4"], ms=3.5,
                 alpha=0.8, ls="none")
    ins.axvline(diff, color="#333333", lw=1.2)
    ins.set_xlim(-0.62, 0.62)
    ins.set_yticks([])
    ins.set_xticks([-0.5, 0, 0.5])
    ins.tick_params(length=2, labelsize=6)
    ins.set_title("S4$-$A5 rank diff vs $\\pm$0.5\nTOST margin", fontsize=6,
                  pad=1.5)
    ins.grid(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out_dir, "fig1_convergence.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _razor_cell(repo, archive, group, arm, seed):
    d = _load(repo, f"{archive}/zero_pad__{group}__{arm}__seed{seed}.json")
    assert d["group"] == group and d["seed"] == seed and d["arm"] == arm
    return d["crosscheck_recovered_frac_90"]


def fig2_razor_step(repo, out_dir):
    """Evidence U3/U4: the causal razor — recovery step at k = d_min.

    Five small-multiple panels; x = force-rank k, y = crosscheck rec@0.9;
    dashed line = unconstrained anchor, dotted = pre-registered 0.9 x anchor
    bar (S3: the fixed seed-0 literal). S3 shows all four seeds.
    """
    arms = ["k_dmin_minus_1", "k_dmin", "k_dmin_plus_1"]
    fig, axes = plt.subplots(1, 5, figsize=(6.8, 1.6), sharey=True)
    for ax, g in zip(axes, GROUPS):
        dm = DMIN[g]
        xs = [dm - 1, dm, dm + 1]
        if g == "S3":
            seed_src = [(M3FIX, 0), (S3EXT, 1), (S3EXT, 2), (S3EXT, 3)]
            per_seed = []
            for src, s in seed_src:
                ys = [_razor_cell(repo, src, g, a, s) for a in arms]
                per_seed.append(ys)
                ax.plot(xs, ys, color=COLOR[g], alpha=0.35, lw=0.8, zorder=2)
            mean = [sum(v) / len(v) for v in zip(*per_seed)]
            anchor = _razor_cell(repo, M3FIX, g, "unconstrained", 0)
            ax.plot(xs, mean, color=COLOR[g], marker=MARKER[g], ms=4, lw=1.6,
                    zorder=3)
            ax.set_title(f"{g}  ($d_{{\\min}}$={dm}, 4 seeds)", pad=2)
        else:
            ys = [_razor_cell(repo, M3FIX, g, a, 0) for a in arms]
            anchor = _razor_cell(repo, M3FIX, g, "unconstrained", 0)
            ax.plot(xs, ys, color=COLOR[g], marker=MARKER[g], ms=4, lw=1.6,
                    zorder=3)
            ax.set_title(f"{g}  ($d_{{\\min}}$={dm})", pad=2)
        ax.axhline(anchor, color="#555555", ls="--", lw=0.8, zorder=1)
        # Coarse dot pitch so dotted reads distinctly from dashed at print
        # size (round-2 render-inspection v3 finding).
        ax.axhline(0.9 * anchor, color="#555555", ls=(0, (1, 2)), lw=1.0,
                   zorder=1)
        ax.axvline(dm, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(["$d_{\\min}$$-$1", "$d_{\\min}$", "$d_{\\min}$$+$1"])
        ax.set_ylim(-0.05, 1.0)
        ax.tick_params(length=2)
    axes[0].set_ylabel("exact recovery\n(rec@0.9)")
    axes[-1].plot([], [], color="#555555", ls="--", lw=0.8,
                  label="unconstrained anchor")
    axes[-1].plot([], [], color="#555555", ls=(0, (1, 2)), lw=1.0,
                  label="0.9 $\\times$ anchor bar")
    axes[-1].legend(loc="lower right", frameon=False, handlelength=1.6,
                    borderaxespad=0.1)
    fig.supxlabel("train-time force-rank $k$", y=-0.10, fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(out_dir, "fig2_razor_step.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="papers/unireps-ea/figures")
    p.add_argument("--repo", default=".")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig1_convergence(args.repo, args.out)
    fig2_razor_step(args.repo, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
