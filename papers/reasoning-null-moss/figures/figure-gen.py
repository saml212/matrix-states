#!/usr/bin/env python3
"""figure-gen.py — the single versioned figure script for reasoning-null-moss.

Every panel loads ONLY archived raw result files from experiment-runs/ and
asserts their md5s against the manifest below before plotting. No number is
hand-entered. If any archived file changed, the build fails loudly.

Figures (COLM single-column, 5.0 in wide, print PDF, no baked-in titles):
  fig1_validity_gates.pdf       — premise medians vs null p95, all instruments
  fig2_dissociation.pdf         — L_query falls while the geometric gate stays 0
  fig3_transient_replication.pdf — the n=3 transient and the n=12 batch gate

Palette: Okabe–Ito subset #0072B2/#D55E00/#009E73/#CC79A7 (validated
colorblind-safe, dataviz skill six-check validator, 2026-07-10; the #CC79A7
contrast WARN is relieved by direct labels on every series).
"""
import glob
import hashlib
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUNS = os.path.join(REPO, "experiment-runs")

# ---------------------------------------------------------------- manifest
# Aggregate manifest-md5 = md5 of the sorted "md5  relpath\n" line list over a
# file set (pins every member file byte-exactly); singleton files pinned
# directly. Recorded 2026-07-10 from the committed archives; these are the
# same identifiers as brief.md's claims-to-evidence map.
MANIFEST_SETS = {
    "phase1": ("2026-07-07_reasoning_link_phase1/results/leg_*.json",
               78, "3c885fbe49ada8cb62afb33f3f6faf60"),
    "rung3": ("2026-07-07_reasoning_link_rung3/results/*.json",
              2, "c83f4fbaed5e57ba40bde7cc57779b2a"),
    "phase1b": ("2026-07-07_phase1b_gate/results/stage0_natural_*.json",
                4, "f3ec8b2a89f19645f84bcaa385b4efd4"),
    "phase2_gates": ("2026-07-08_phase2_familiarization/ckpts/stage05_gate_*.json",
                     30, "85d381404dd89769eda382beacbee673"),
    "phase2_off": ("2026-07-08_phase2_familiarization/results/off_*.json",
                   6, "091c33c4786048cd3f94bb75496465ec"),
}
MANIFEST_FILES = {
    "2026-07-08_phase2b/results/trajectory_wikitext-mix-ext_phase2b.json":
        "5c727ac669aea02601790b4fb1dac8b4",
    "2026-07-08_phase2b_seedext/results/trajectory_seedext_wikitext_n12.json":
        "989f9997c50973cc299f25d89efff64f",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_set(name):
    pattern, n_expect, agg_expect = MANIFEST_SETS[name]
    files = sorted(glob.glob(os.path.join(RUNS, pattern)))
    assert len(files) == n_expect, f"{name}: {len(files)} files, expected {n_expect}"
    lines = "".join(f"{md5(p)}  {os.path.relpath(p, RUNS)}\n" for p in files)
    agg = hashlib.md5(lines.encode()).hexdigest()
    assert agg == agg_expect, f"{name}: manifest-md5 {agg} != recorded {agg_expect}"
    return [json.load(open(p)) for p in files]


def load_file(rel):
    p = os.path.join(RUNS, rel)
    got = md5(p)
    assert got == MANIFEST_FILES[rel], f"{rel}: md5 {got} != recorded {MANIFEST_FILES[rel]}"
    return json.load(open(p))


# ---------------------------------------------------------------- style
BLUE, VERM, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "figure.dpi": 200,
})
CKPTS = [250, 500, 1000, 2500, 5000]


def out(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- figure 1
def fig1():
    groups = {}  # label -> (color, marker, [(p95, median)] iii, [] iv)

    def add(label, color, marker, med3, p953, med4, p954):
        g = groups.setdefault(label, (color, marker, [], []))
        g[2].append((p953, med3))
        g[3].append((p954, med4))

    for d in load_set("phase1"):
        h1 = d["per_h"]["1"]
        rung = d.get("ckpt_config", {}).get("n_layers")
        label = "Phase 1 zero-shot (14M-392M)"
        add(label, BLUE, "o", h1["premise_iii_median"], h1["premise_iii_null_p95"],
            h1["premise_iv_median"], h1["premise_iv_null_p95"])
        # gates must fail everywhere: pass iff median > p95
        assert not h1["premise_iii_pass"] and not h1["premise_iv_pass"]
        for h in d["per_h"].values():
            assert h["recovered_frac"] == 0.0
    for d in load_set("rung3"):
        h1 = d["per_h"]["1"]
        add("Rung 3 zero-shot (1.31B)", VERM, "s",
            h1["premise_iii_median"], h1["premise_iii_null_p95"],
            h1["premise_iv_median"], h1["premise_iv_null_p95"])
        assert not h1["premise_iii_pass"] and not h1["premise_iv_pass"]
        for h in d["per_h"].values():
            assert h["recovered_frac"] == 0.0
    for d in load_set("phase1b"):
        h1 = d["per_h"]["1"]
        add("Phase 1b natural language", GREEN, "^",
            h1["premise_iii_median"], h1["premise_iii_null_p95"],
            h1["premise_iv_median"], h1["premise_iv_null_p95"])
        assert d["gate_result_h1_probe_valid"] is False
        for h in d["per_h"].values():
            assert h["recovered_frac"] == 0.0
    n_gate_zero = 0
    for d in load_set("phase2_gates"):
        h1 = d["per_h"]["1"]
        assert d["gate_pass"] is False and h1["recovered_frac"] == 0.0
        n_gate_zero += 1
        if d["checkpoint_step"] == 5000:
            add("Phase 2 familiarized (terminal)", PINK, "D",
                h1["premise_iii_median"], h1["premise_iii_null_p95"],
                h1["premise_iv_median"], h1["premise_iv_null_p95"])
    assert n_gate_zero == 30

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), sharey=True)
    for ax, idx, sub in zip(axes, (2, 3), ("(iii) query-key alignment",
                                           "(iv) key-value alignment")):
        lo, hi = -0.7, 0.9
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="0.55", zorder=1)
        for label, (color, marker, pts3, pts4) in groups.items():
            pts = pts3 if idx == 2 else pts4
            xs, ys = zip(*pts)
            assert all(y <= x for x, y in pts), f"{label}: a premise passed"
            ax.scatter(xs, ys, s=12, marker=marker, facecolors="none",
                       edgecolors=color, linewidths=0.9, label=label, zorder=2)
        ax.set_xlabel("shuffled-null p95")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(0.03, 0.96, sub, transform=ax.transAxes, va="top", fontsize=7.5)
        ax.text(0.60, 0.10, "fail region:\nmedian $\\leq$ null p95\n(every point)",
                transform=ax.transAxes, fontsize=6.5, color="0.45")
    axes[0].set_ylabel("same-entity median cosine")
    fig.subplots_adjust(wspace=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.14), handletextpad=0.1, columnspacing=0.8)
    out(fig, "fig1_validity_gates.pdf")


# ---------------------------------------------------------------- figure 2
def fig2():
    cells = load_set("phase2_off")
    gates = load_set("phase2_gates")
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.1))
    fig.subplots_adjust(wspace=0.42)
    axL, axR = axes
    for d in cells:
        by_step = {r["step"]: r["loss_query"] for r in d["trajectory"]
                   if r["step"] in CKPTS}
        ys = [by_step[c] for c in CKPTS]
        ratio = ys[-1] / ys[0]
        assert 0.53 < ratio < 0.79  # the 21.8-46.4% fall band
        corpus = "openr1" if "openr1" in d["corpus"] else "wikitext"
        color = BLUE if corpus == "openr1" else VERM
        axL.plot(CKPTS, ys, marker="o", ms=3, lw=1.2, color=color, alpha=0.85)
    axL.plot([], [], color=BLUE, lw=1.2, label="openr1 cells (3 seeds)")
    axL.plot([], [], color=VERM, lw=1.2, label="wikitext cells (3 seeds)")
    axL.set_xscale("log")
    axL.set_xticks(CKPTS)
    axL.set_xticklabels([str(c) for c in CKPTS])
    axL.set_xlabel("familiarization step")
    axL.set_ylabel(r"$L_\mathrm{query}$ (vocab-space CE)")
    axL.legend(frameon=False, loc="lower left", handlelength=1.4)

    # right panel: the geometric gate at the same 30 (cell, checkpoint) points
    xs, ys = [], []
    for d in gates:
        xs.append(d["checkpoint_step"])
        ys.append(d["per_h"]["1"]["recovered_frac"])
    assert all(y == 0.0 for y in ys) and len(ys) == 30
    axR.scatter(xs, ys, s=14, marker="x", color=GREEN, linewidths=1.0,
                label="all 30 readings = 0.0")
    axR.axhline(0.10, ls="--", lw=0.8, color="0.55")
    axR.text(260, 0.115, "h=1 absolute validity floor (0.10)", fontsize=6.5,
             color="0.35")
    axR.set_xscale("log")
    axR.set_xticks(CKPTS)
    axR.set_xticklabels([str(c) for c in CKPTS])
    axR.set_xlabel("familiarization step")
    axR.set_ylabel("recovered fraction @0.9")
    axR.set_ylim(-0.05, 1.0)
    axR.legend(frameon=False, loc="upper left", handlelength=1.0)
    out(fig, "fig2_dissociation.pdf")


# ---------------------------------------------------------------- figure 3
def fig3():
    tw = load_file("2026-07-08_phase2b/results/trajectory_wikitext-mix-ext_phase2b.json")
    pt = tw["per_arm"]["per_token"]
    assert pt["holds_by_c"] == {"250": False, "500": False, "1000": False,
                                "2500": True, "5000": False}
    t12 = load_file("2026-07-08_phase2b_seedext/results/trajectory_seedext_wikitext_n12.json")

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.45)
    axL, axR = axes

    for K, color, off in (("32", BLUE, 0.93), ("20", VERM, 1.075)):
        means, los, his = [], [], []
        for c in CKPTS:
            blk = pt["raw"][str(c)][f"delta_k{K}"]
            means.append(blk["mean"])
            los.append(blk["mean"] - blk["ci_low"])
            his.append(blk["ci_high"] - blk["mean"])
        xs = [c * off for c in CKPTS]
        axL.errorbar(xs, means, yerr=[los, his], fmt="o", ms=3, lw=1.0,
                     capsize=2, color=color, label=f"K={K}")
    blk = pt["raw"]["2500"]["delta_k32"]
    assert blk["ci_high"] < 0  # the transient: CI excludes zero, harm direction
    axL.annotate("c=2500, K=32:\nCI excludes 0\n(harm direction)",
                 xy=(2500 * 0.93, blk["ci_low"]), xytext=(300, -1.85),
                 fontsize=6.0, color="0.25",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="0.45"))
    axL.axhline(0, lw=0.8, color="0.55")
    axL.set_xscale("log")
    axL.set_xticks(CKPTS)
    axL.set_xticklabels([str(c) for c in CKPTS])
    axL.set_xlabel("familiarization step")
    axL.set_ylabel(r"$\Delta L_\mathrm{query}$ (off $-$ arm)")
    # legend above the axes: the c=5000 whisker caps reach the top of the
    # data region, so an in-axes legend collides with an error-bar cap
    axL.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01),
               ncol=2, handlelength=1.0, borderaxespad=0.0,
               columnspacing=1.2)

    # right: anchor cell per-seed deltas, two cohorts + naive pool
    anchor = t12["primary"]["2500"]["delta_k32"]
    old, new = anchor["old_cohort"], anchor["new_cohort"]
    assert anchor["batch_gate"]["flagged"] is True
    assert 4.4 < anchor["batch_gate"]["var_ratio"] < 4.5
    axR.scatter([0] * 3, old["deltas"], s=16, marker="o", facecolors="none",
                edgecolors=BLUE, linewidths=1.0)
    axR.scatter([1] * 9, new["deltas"], s=16, marker="s", facecolors="none",
                edgecolors=VERM, linewidths=1.0)
    for x, coh, color in ((0, old, BLUE), (1, new, VERM)):
        axR.errorbar([x + 0.18], [coh["mean"]],
                     yerr=[[coh["mean"] - coh["ci_low"]],
                           [coh["ci_high"] - coh["mean"]]],
                     fmt="_", ms=7, lw=1.3, capsize=3, color=color)
    # naive n=12 pool, recomputed from the 12 raw deltas (diagnostic only)
    pool = old["deltas"] + new["deltas"]
    n = len(pool)
    mean = sum(pool) / n
    sd = (sum((x - mean) ** 2 for x in pool) / (n - 1)) ** 0.5
    half = 2.201 * sd / n ** 0.5  # t(11, .975)
    axR.errorbar([0.5], [mean], yerr=[[half], [half]], fmt="_", ms=7, lw=1.1,
                 capsize=3, color="0.5", ls=":")
    axR.text(0.5, mean + half + 0.07, "naive n=12 pool\n(not decision-grade)",
             ha="center", va="bottom", fontsize=6.0, color="0.4")
    axR.set_ylim(-1.4, 1.05)
    assert mean - half < 0 < mean + half  # spans zero
    assert new["ci_low"] < 0 < new["ci_high"]  # new cohort spans zero
    assert old["ci_high"] < 0  # archived n=3 excludes zero
    axR.axhline(0, lw=0.8, color="0.55")
    axR.set_xticks([0, 0.5, 1])
    axR.set_xticklabels(["archived\nn=3", "", "new\nn=9"])
    axR.set_xlim(-0.4, 1.6)
    axR.set_ylabel(r"per-seed $\Delta$ at K=32, c=2500")
    out(fig, "fig3_transient_replication.pdf")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("all figures built; every source md5 asserted")
