"""
Generate the K-scaling figure for the ncr-operator-bank findings page
(finding no. 16), the section reporting NOVEL_ARCH_WATERFALL.md §11/§11.2:
does the earlyln recipe (annealed inter-hop LayerNorm, the same fix that
recovered R=3 bank convergence in section 04 of this page) extend K=14's
single-relation success up the K axis?

Figure (ncr_earlyln_kscaling.svg): two panels sharing the K axis
{14, 15, 16, 24}.
  Left  -- convergence rate (n_converged / 4 seeds) per K, bar chart, with
           the pre-registered CONVERGED-ROBUST bar (>=3/4 = 0.75) marked.
  Right -- per-seed far-depth recovered@0.9 at each cell's own h*, dot
           plot, ONE dot per seed that actually CONVERGED at gate 1 (gate
           2 is scored only on converged cells, per the pre-registration --
           K=16 has 1 dot, K=24 has none), with the HOLD bar (0.9) marked.

This is the pooled verdict figure: K=14 (continuity) and K=15 both clear
both bars (SCALES); K=16 clears almost none of gate 1 and its lone
converged seed reads 0.0 at h* (front=13, the first ladder rung); K=24
clears neither bar. The trainability wall moves exactly one rung (K=14/15
dead->scales precedent was K=15 only under the PLAIN recipe per section 06
above -- earlyln's own contribution is rescuing K=15), then re-forms.

Data source (raw harvest + cell JSONs, md5-verified against the committed
manifest before use):
  experiment-runs/2026-07-11_ncr_earlyln_scale/harvest_verdict.json
    (per-K n_converged/rate/ladder_verdict, per-cell gate1/gate2 blocks)
  experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K{14,15,16,24}_s{0-3}.json
    (per-cell train.step / gate2.recovered_at_h_star cross-check)
  experiment-runs/2026-07-11_ncr_earlyln_scale/md5_manifest.txt

Palette: Okabe-Ito (colorblind-safe). No in-figure title (caption lives in
the HTML). Background matches the site (#FAF5E7).

Also writes ncr_earlyln_kscaling_x.png -- a standalone 1200x675 PNG (X's
summary_large_image ratio) with an in-figure title and a footer URL, for
the social-post attachment. Same underlying panels, larger fonts; not
used on the site itself.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_VERMILLION = "#D55E00"
OI_BLUE = "#0072B2"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_GREY = "#999999"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
RAW_DIR = REPO / "experiment-runs/2026-07-11_ncr_earlyln_scale"
MANIFEST = RAW_DIR / "md5_manifest.txt"
HARVEST = RAW_DIR / "harvest_verdict.json"

# ------------------------------------------------------------- md5 gate
manifest = {}
for line in MANIFEST.read_text().splitlines():
    if not line.strip():
        continue
    h, name = line.split(None, 1)
    manifest.setdefault(name.strip(), h)  # first occurrence wins (both copies match)

KS = [14, 15, 16, 24]
cell_files = [f"earlyln_K{k}_s{s}.json" for k in KS for s in range(4)] + ["harvest_verdict.json"]
for fn in cell_files:
    fp = RAW_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(cell_files)} files (16 cell JSONs + harvest_verdict.json)")

with open(HARVEST) as f:
    harvest = json.load(f)

cells = {}
for k in KS:
    for s in range(4):
        with open(RAW_DIR / f"earlyln_K{k}_s{s}.json") as f:
            cells[(k, s)] = json.load(f)

# ---------------------------------------------------------- cross-checks
# convergence rate and gate2 points must reproduce the raw cell JSONs
# bit-for-bit before anything is plotted -- the harvest file is derived,
# never trusted blind.
conv_rate = {}
far_depth_points = {}  # k -> list of recovered_at_h_star for CONVERGED seeds
for k in KS:
    pk = harvest["per_K"][str(k)]
    conv_rate[k] = pk["rate"]
    n_conv_harvest = pk["n_converged"]
    n_conv_raw = 0
    pts = []
    for s in range(4):
        c = cells[(k, s)]
        # gate1 verdict lives in harvest_verdict.json's per-cell block; the
        # cross-check below is on gate2 (far-depth), reconstructed from the
        # raw cell JSON's own eval.points rather than trusted from harvest.
        if pk["cells"][str(s)]["gate1"]["verdict"] == "CONVERGED":
            n_conv_raw += 1
            g2 = pk["cells"][str(s)].get("gate2")
            assert g2 is not None, f"K={k} s={s}: CONVERGED cell missing gate2 block"
            # cross-check against the raw cell's own eval.points ladder entry
            # at h_star (component="ladder", the pre-registered claim point;
            # "residue_sweep" duplicates the same h via a different sweep
            # pass and is not used here).
            h_star = g2["h_star"]
            raw_val = None
            for p in c["eval"]["points"]:
                if p["h"] == h_star and p["component"] == "ladder":
                    raw_val = p["reads"]["binexp"]["recovered_frac@0.9"]
                    break
            assert raw_val is not None, f"K={k} s={s}: no ladder point at h_star={h_star} in raw eval.points"
            assert abs(raw_val - g2["recovered_at_h_star"]) < 1e-9, (
                f"K={k} s={s}: raw ladder point at h_star={h_star} = {raw_val} != "
                f"harvest gate2.recovered_at_h_star={g2['recovered_at_h_star']}"
            )
            pts.append(g2["recovered_at_h_star"])
    assert n_conv_raw == n_conv_harvest, f"K={k}: raw CONVERGED count {n_conv_raw} != harvest n_converged {n_conv_harvest}"
    far_depth_points[k] = pts
print("convergence counts and far-depth recovered@h_star cross-checked bit-for-bit against raw cell JSONs for all 16 cells")

# ---------------------------------------------------------------- figure
bar_colors = {14: OI_GREY, 15: OI_GREEN, 16: OI_SKY, 24: OI_VERMILLION}
x = list(range(len(KS)))
labels = [f"K={k}" + (" (continuity)" if k == 14 else "") for k in KS]


def draw_panels(ax1, ax2, label_fs=9.5, tick_fs=9.5, marker_fs=7.8, dot_size=70):
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # ---- left panel: convergence rate
    rates = [conv_rate[k] for k in KS]
    ax1.bar(x, rates, color=[bar_colors[k] for k in KS], edgecolor=TEXT, linewidth=1.0, width=0.6, zorder=3)
    for xi, r, k in zip(x, rates, KS):
        n = harvest["per_K"][str(k)]["n_converged"]
        ax1.text(xi, r + 0.035, f"{n}/4", ha="center", va="bottom", fontsize=label_fs + 0.5,
                  fontweight="bold", color=TEXT)
    ax1.axhline(0.75, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    ax1.text(3.55, 0.70, "CONVERGED-\nROBUST bar", fontsize=marker_fs, color=MUTED, va="top", ha="left")
    ax1.set_ylim(0, 1.12)
    ax1.set_xlim(-0.5, 3.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=tick_fs)
    ax1.set_ylabel("gate 1: convergence rate (n / 4 seeds)", fontsize=label_fs)
    ax1.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT, zorder=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)

    # ---- right panel: far-depth recovered@h* per converged seed
    rng_jitter = [-0.09, -0.03, 0.03, 0.09]
    for xi, k in zip(x, KS):
        pts = far_depth_points[k]
        for j, y in enumerate(sorted(pts)):
            ax2.scatter(xi + rng_jitter[j % len(rng_jitter)], y, color=bar_colors[k],
                        edgecolor=TEXT, linewidth=0.9, s=dot_size, zorder=4)
        if not pts:
            ax2.text(xi, 0.05, "no converged\nseeds", ha="center", va="bottom", fontsize=marker_fs,
                      color=MUTED, style="italic")
    ax2.axhline(0.9, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    ax2.text(3.55, 0.85, "HOLD bar", fontsize=marker_fs, color=MUTED, va="top", ha="left")
    ax2.set_ylim(-0.06, 1.12)
    ax2.set_xlim(-0.5, 3.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=tick_fs)
    ax2.set_ylabel("gate 2: recovered@0.9 at own h* (CONVERGED seeds only)", fontsize=label_fs)
    ax2.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT, zorder=0)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(TEXT)


# ---- site figure (no in-figure title; caption lives in the HTML)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=BG)
draw_panels(ax1, ax2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ncr_earlyln_kscaling.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_earlyln_kscaling.svg")

# ---- standalone PNG for social (1200x675, X's summary_large_image ratio)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=100, facecolor=BG)
draw_panels(ax1, ax2, label_fs=12.0, tick_fs=11.5, marker_fs=10.0, dot_size=110)
fig.suptitle("The K-scaling wall moves exactly one rung under earlyln, then re-forms",
             fontsize=16.5, fontweight="bold", color=TEXT, x=0.02, ha="left", y=0.995)
fig.text(0.99, 0.015, "pebbleml.com/findings/ncr-operator-bank.html", fontsize=10.5,
          color=MUTED, ha="right")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig(f"{OUT_DIR}/ncr_earlyln_kscaling_x.png", format="png", facecolor=BG, dpi=100)
plt.close(fig)
print("wrote ncr_earlyln_kscaling_x.png")
