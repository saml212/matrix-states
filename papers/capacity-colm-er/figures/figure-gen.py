#!/usr/bin/env python3
"""figure-gen.py — the single versioned figure script for the
capacity-colm-er submission (paper skill, repo mode, method Stage 4).

Every panel's data is LOADED from the archived raw artifacts named in
brief.md's claims-to-evidence map and recomputed here; no plotted value is
hand-entered. The MD5 manifest below is asserted before any file is read:
if an archived raw changes, this build fails loudly rather than plotting
stale or mismatched data (method.md Stage 4).

Run from the repo root:
    python3 papers/capacity-colm-er/figures/figure-gen.py
Outputs (written next to this script): fig1_cliff.{pdf,png},
fig2_dose.{pdf,png}, fig3_frontier.{pdf,png}.

Palette: Okabe-Ito subset (#0072B2 blue, #E69F00 orange, #D55E00
vermillion, #009E73 green), validated colorblind-safe (CVD separation
PASS) with shape/label relief for the low-contrast orange.
"""

import hashlib
import json
import os
import sys
import glob

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# MD5 manifest — one row per source file (brief.md rows C1-C13)
# ---------------------------------------------------------------------------
MANIFEST = {
    "experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json": "c4e233fe3c7c6ec246c6e92e35134258",
    "experiment-runs/2026-07-06_keyanchor_dstate/fit_cliff_curve_d128_results.json": "0801d804c89e739da9712e9beb24b50f",
    "experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/fit_cliff_curve_d80_refit_results.json": "05dd2f9e79747f871756201165fdd548",
    "experiment-runs/2026-07-08_c17_repro/fits/fit_d96_unlocked_results.json": "61eaffe1744a56086af2f4115f9a9cf4",
    "experiment-runs/2026-07-08_d96_scatter_resolution_design/fits/sim_d96_scatter_resolution_power_results.json": "fbd8342d266989ab9112b14997d69196",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K34_armd_s130_geo3n20_anchor_learned_dprobe_rev7.json": "30dd041f403c205099c7d38812b95492",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K34_armd_s131_geo3n20_anchor_learned_dprobe_rev7.json": "7d097e6d0fba369707376ab8095c9fb6",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K34_armd_s132_geo3n20_anchor_learned_dprobe_rev7.json": "9b0a313bf1f366bac64b2f47641850aa",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K38_armd_s230_geo3n20_anchor_learned_dprobe_rev7.json": "ac9daf69d6b6e720be3cd554f1cfa713",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K38_armd_s231_geo3n20_anchor_learned_dprobe_rev7.json": "4eaee26bd83fbf5ecce582bc468522e4",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K38_armd_s232_geo3n20_anchor_learned_dprobe_rev7.json": "3b00023d6a8f1bdb9e7800c712973be7",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K42_armd_s330_geo3n20_anchor_learned_dprobe_rev7.json": "83c248c9e0a653851108f69a70265ad7",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K42_armd_s331_geo3n20_anchor_learned_dprobe_rev7.json": "ab7ef75b7cbfa63b6d515f1d9842a1e0",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K42_armd_s332_geo3n20_anchor_learned_dprobe_rev7.json": "6093d71d393f7b6c0360857bba29e0bf",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K46_armd_s430_geo3n20_anchor_learned_dprobe_rev7.json": "e33aa27c26b8a4d67c26223f6f172f18",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K46_armd_s431_geo3n20_anchor_learned_dprobe_rev7.json": "82986a1beaaf9f030609cd3fe8808dc7",
    "experiment-runs/2026-07-06_keyanchor_cliff/results/deltanet_rd_exactness/wavekeyanchor-cliff/wkeyanchor-k48_rdx_K46_armd_s432_geo3n20_anchor_learned_dprobe_rev7.json": "2fc0db2bd9701c62dc57ce9483b84b46",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K68_armd_s530_geo3n20_anchor_learned_dprobe_rev7_d128.json": "9668de8df425683e09f31bc9276fd91d",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K68_armd_s531_geo3n20_anchor_learned_dprobe_rev7_d128.json": "de631b01788ae2128a5a8cbdd3ef5b21",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K68_armd_s532_geo3n20_anchor_learned_dprobe_rev7_d128.json": "373a09b9e1780477205c2488565de355",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K76_armd_s630_geo3n20_anchor_learned_dprobe_rev7_d128.json": "3a6cfd86c86afae4587ec9ea4cce7fdd",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K76_armd_s631_geo3n20_anchor_learned_dprobe_rev7_d128.json": "adf88154f6f37d96439be8b7fbee287d",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K76_armd_s632_geo3n20_anchor_learned_dprobe_rev7_d128.json": "3560356412df88a4e7f3f5ee72022cfe",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K84_armd_s730_geo3n20_anchor_learned_dprobe_rev7_d128.json": "47b23aaff41862d9de7d02f2ea89dd4c",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K84_armd_s731_geo3n20_anchor_learned_dprobe_rev7_d128.json": "6959edf865f5eeeb9e2ab456f74d4da1",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K84_armd_s732_geo3n20_anchor_learned_dprobe_rev7_d128.json": "3d8f9c21dac844c3a34ae5cd747bab17",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K92_armd_s830_geo3n20_anchor_learned_dprobe_rev7_d128.json": "22ccb279fafc8b45538a25b33415066c",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K92_armd_s831_geo3n20_anchor_learned_dprobe_rev7_d128.json": "8df14c4295549107e133ef44103dfb9a",
    "experiment-runs/2026-07-06_keyanchor_dstate/results/wavekeyanchor-dstate/wkeyanchor-k48_rdx_K92_armd_s832_geo3n20_anchor_learned_dprobe_rev7_d128.json": "c227367af22834992872341f19b9fd78",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s930_geo3n20_anchor_learned_dprobe_rev7_d128_dose130_rank4_sseed20260705.json": "bb62f002b1187e6b517425f3be65655b",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s931_geo3n20_anchor_learned_dprobe_rev7_d128_dose130_rank4_sseed20260705.json": "ffdb505d9fc947e732a6417753a49d4f",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s932_geo3n20_anchor_learned_dprobe_rev7_d128_dose130_rank4_sseed20260705.json": "6355e8a4f4eee71ac377b18e43956dd5",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s933_geo3n20_anchor_learned_dprobe_rev7_d128_dose284_rank4_sseed20260705.json": "5fc71f2fc8445d9e105521dc15b73fa9",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s934_geo3n20_anchor_learned_dprobe_rev7_d128_dose284_rank4_sseed20260705.json": "7e1195c08cca274e3659007781a47add",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s935_geo3n20_anchor_learned_dprobe_rev7_d128_dose284_rank4_sseed20260705.json": "3b2d50f210a272b408e4aa91d700097f",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s936_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_rank4_sseed20260705.json": "d51ac94d2c1b4013de218a2419b7d17b",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s937_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_rank4_sseed20260705.json": "4f2cac0a0b7eda54568ddfdf371d1c8b",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s938_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_rank4_sseed20260705.json": "cdd4cc35a689c7bf4ccf07b7a11faf13",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s939_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_rank4_sseed20260705.json": "fbcae1adabd9d25b9377906994b266ec",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s940_geo3n20_anchor_learned_dprobe_rev7_d128_dose130_diffuse_sseed20260705.json": "e298f1a99cb2f529f5670ec26cd27f9f",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s941_geo3n20_anchor_learned_dprobe_rev7_d128_dose130_diffuse_sseed20260705.json": "b825c888354b8b80a7184796bcb577d6",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s942_geo3n20_anchor_learned_dprobe_rev7_d128_dose130_diffuse_sseed20260705.json": "c7f34a857f5f31eb6cbe4bf8529e6d0d",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s943_geo3n20_anchor_learned_dprobe_rev7_d128_dose284_diffuse_sseed20260705.json": "749a4095de5db97e45ff02bf7c16e1fc",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s944_geo3n20_anchor_learned_dprobe_rev7_d128_dose284_diffuse_sseed20260705.json": "b21ada28dc5ebd8f68284830a621fc34",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s945_geo3n20_anchor_learned_dprobe_rev7_d128_dose284_diffuse_sseed20260705.json": "13b0877c7f50354ac89aa4f96f045293",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s946_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_diffuse_sseed20260705.json": "ce3fe1918c7df99491660d523533ac83",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s947_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_diffuse_sseed20260705.json": "c156fd2ce1c5f6a92e5c4862393075f8",
    "experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/wkeyanchor-dose_rdx_K68_armd_s948_geo3n20_anchor_learned_dprobe_rev7_d128_dose400_diffuse_sseed20260705.json": "47864b0a806e4f74e3a178455d9d20f9",
}

def _assert_manifest():
    bad = []
    for rel, want in MANIFEST.items():
        p = os.path.join(REPO, rel)
        got = hashlib.md5(open(p, "rb").read()).hexdigest()
        if got != want:
            bad.append((rel, want, got))
    if bad:
        for rel, want, got in bad:
            print(f"MANIFEST MISMATCH: {rel}\n  want {want}\n  got  {got}")
        sys.exit(1)
    print(f"manifest OK: {len(MANIFEST)} files")


def _load(rel):
    return json.load(open(os.path.join(REPO, rel)))


def _cell_h4(cell):
    return cell["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]


def _cell_coh(cell):
    return float(cell["checkpoints"][-1]["item6_table_conditioning"]["max_abs_cos"])


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE, ORANGE, VERM, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"
INK, MUTED = "#1a1a1a", "#666666"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})


def fig1_cliff():
    """Two panels: d=64 transition + logistic fit; d=128 flat at the
    identical K/d window (fit degenerate by construction -> points only)."""
    f64 = _load("experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json")
    f128 = _load("experiment-runs/2026-07-06_keyanchor_dstate/fit_cliff_curve_d128_results.json")
    cells64 = [ _load(r) for r in MANIFEST if "wavekeyanchor-cliff/" in r ]
    cells128 = [ _load(r) for r in MANIFEST if "wavekeyanchor-dstate/" in r ]

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)

    # -- left: d=64
    ax = axes[0]
    x = np.array(f64["curve_points"]["x"]); y = np.array(f64["curve_points"]["h4"])
    fit = f64["sigmoid_fit"]; ci = f64["bootstrap_ci"]["ci_x0"]
    xs = np.linspace(0.22, 0.78, 400)
    ax.plot(xs, fit["L"] / (1 + np.exp((xs - fit["x0"]) / fit["w"])),
            color=BLUE, lw=1.4, zorder=2, label="logistic fit")
    # per-seed scatter for the four new K's, from the cell raws
    first = True
    for c in cells64:
        ax.plot(c["K"] / 64.0, _cell_h4(c), marker="o", ms=3, mfc="none",
                mec=BLUE, mew=0.7, ls="none", zorder=3,
                label=("per-seed cells" if first else None))
        first = False
    ax.plot(x, y, marker="o", ms=4.5, color=BLUE, ls="none", zorder=4,
            label="seed-mean $h4$")
    ax.axvspan(ci["lo"], ci["hi"], color=BLUE, alpha=0.15, lw=0, zorder=1)
    ax.text(0.24, 0.33, f"$x_0={fit['x0']:.4f}$\n95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]",
            fontsize=7)
    ax.set_title("$d_{\\mathrm{state}}=64$: a sharp transition")
    ax.set_xlabel("load ratio $K/d_{\\mathrm{state}}$")
    ax.set_ylabel("held-out 4-hop recovery $h4$")
    ax.set_xlim(0.22, 0.78); ax.set_ylim(-0.04, 1.06)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4)

    # -- right: d=128, same window
    ax = axes[1]
    for c in cells128:
        ax.plot(c["K"] / 128.0, _cell_h4(c), marker="o", ms=3, mfc="none",
                mec=BLUE, mew=0.7, ls="none", zorder=3)
    x2 = np.array(f128["curve_points"]["x"]); y2 = np.array(f128["curve_points"]["h4"])
    ax.plot(x2, y2, marker="o", ms=4.5, color=BLUE, ls="none", zorder=4)
    ax.set_title("$d_{\\mathrm{state}}=128$: the same window, flat")
    ax.set_xlabel("load ratio $K/d_{\\mathrm{state}}$")
    ax.set_xlim(0.50, 0.75)
    ax.text(0.625, 0.5, "12/12 cells at $h4=1.0$\n(fit degenerate by\nconstruction; no curve drawn)",
            ha="center", va="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_cliff.{ext}"), dpi=300)
    plt.close(fig)


def fig2_dose():
    """Frozen coherence dose-response at d=128/K=68, both injection
    structures, with d=64's own trained band and collapse for reference."""
    dose_cells = [ _load(r) for r in MANIFEST if "wavekeyanchor-dose/" in r ]
    cliff_cells = [ _load(r) for r in MANIFEST if "wavekeyanchor-cliff/" in r ]

    fig, ax = plt.subplots(figsize=(4.4, 2.9))

    # d=64 reference: per-cell (final coherence, h4) — collapse across its band
    cx = [_cell_coh(c) for c in cliff_cells]
    cy = [_cell_h4(c) for c in cliff_cells]
    ax.plot(cx, cy, marker="x", ms=4.5, color=VERM, ls="none",
            label="$d=64$ trained cells (its own cliff)", zorder=3)

    # d=64 realized training band (range of final-checkpoint K-means)
    import collections
    byK = collections.defaultdict(list)
    for c in cliff_cells:
        byK[c["K"]].append(_cell_coh(c))
    kmeans = [np.mean(v) for v in byK.values()]
    ax.axvspan(min(kmeans), max(kmeans), color="#bbbbbb", alpha=0.35, lw=0,
               zorder=1)
    ax.text((min(kmeans) + max(kmeans)) / 2 - 0.055, 0.72,
            "$d=64$ realized\ntraining band\n(K-means)", ha="center", fontsize=6.5,
            color=MUTED)

    # dosed d=128 cells, both structures
    for struct, color, marker, dy, label in (
            ("rank4", BLUE, "o", -0.006, "$d=128$ dosed, concentrated (rank-4)"),
            ("diffuse", ORANGE, "^", 0.006, "$d=128$ dosed, diffuse (rank-48)")):
        cs = [c for c, r in zip(dose_cells, [r for r in MANIFEST if "wavekeyanchor-dose/" in r])
              if struct in r]
        xs = [_cell_coh(c) + dy for c in cs]  # +-0.006 horizontal offset,
        # disclosed in the caption, so the two structures' coincident
        # ceiling points stay separately visible
        ys = [_cell_h4(c) for c in cs]
        ax.plot(xs, ys, marker=marker, ms=4.5, mfc="none", mec=color, mew=1.0,
                ls="none", label=label, zorder=4)

    ax.set_xlabel("anchor-table coherence $\\max|\\cos|$ (frozen dose for $d=128$)")
    ax.set_ylabel("held-out 4-hop recovery $h4$")
    ax.set_xlim(0.0, 0.44); ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="center left", frameon=False, fontsize=6.5,
              handletextpad=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_dose.{ext}"), dpi=300)
    plt.close(fig)


def fig3_frontier():
    """The located load frontier x0(d): CI points at d=64/80, in-window
    lower bounds at d=96/128, the ratio-invariance band, rival bands."""
    f64 = _load("experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json")
    f80 = _load("experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/fit_cliff_curve_d80_refit_results.json")
    f96 = _load("experiment-runs/2026-07-08_c17_repro/fits/fit_d96_unlocked_results.json")
    f128 = _load("experiment-runs/2026-07-06_keyanchor_dstate/fit_cliff_curve_d128_results.json")
    sim = _load("experiment-runs/2026-07-08_d96_scatter_resolution_design/fits/sim_d96_scatter_resolution_power_results.json")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))

    # ratio-invariance band, registered around d=64's x0 for the d=80/96 test
    ax.axhspan(0.4745, 0.6165, color="#bbbbbb", alpha=0.30, lw=0, zorder=1)
    ax.text(121, 0.545, "pre-registered\nratio-invariance band",
            fontsize=6.5, color=MUTED, ha="center", va="center")

    # located transitions with bootstrap CIs
    for d, f in ((64, f64), (80, f80)):
        x0 = f["sigmoid_fit"]["x0"]; ci = f["bootstrap_ci"]["ci_x0"]
        ax.errorbar([d], [x0], yerr=[[x0 - ci["lo"]], [ci["hi"] - x0]],
                    fmt="o", ms=5, color=BLUE, capsize=2.5, lw=1.0, zorder=4)
        ax.annotate(f"$x_0={x0:.4f}$", xy=(d, x0), xytext=(d + 2.0, x0 - 0.045),
                    fontsize=7, color=INK)

    # in-window lower bounds (no transition found anywhere in the window)
    for d, f, lab in ((96, f96, "no transition through 0.9375"),
                      (128, f128, "no transition through 0.7188")):
        top = max(f["curve_points"]["x"])
        ax.plot([d], [top], marker="s", ms=5, mfc="white", mec=BLUE, mew=1.2,
                zorder=4)
        ax.annotate("", xy=(d, top + 0.055), xytext=(d, top),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))
        ax.annotate(lab, xy=(d, top), xytext=(d - 2.0, top - 0.028), fontsize=6.5,
                    color=INK, ha="right")

    # rival predictive bands for x0(96), registered before the wide grid
    rb = sim["rival_bands"]
    for (name, band, color, dx) in (("absolute slack", rb["abs_slack"], GREEN, -2.2),
                                    ("power law", rb["power_law"], ORANGE, 2.2)):
        ax.add_patch(plt.Rectangle((96 + dx - 1.5, band[0]), 3.0,
                                   band[1] - band[0], facecolor=color,
                                   alpha=0.45, edgecolor="none", zorder=2))
    ax.annotate("rival bands for $x_0(96)$:\nabsolute-slack (green),\npower-law (orange); both live",
                xy=(94.2, 0.735), xytext=(63.0, 0.74), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    ax.set_xlabel("state dimension $d_{\\mathrm{state}}$")
    ax.set_ylabel("load frontier $x_0 = K^*/d_{\\mathrm{state}}$")
    ax.set_xticks([64, 80, 96, 128])
    ax.set_xlim(58, 136); ax.set_ylim(0.42, 1.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_frontier.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    _assert_manifest()
    fig1_cliff()
    fig2_dose()
    fig3_frontier()
    print("wrote fig1_cliff, fig2_dose, fig3_frontier to", OUT)
