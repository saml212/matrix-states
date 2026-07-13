"""
Generate the closing figure for the ncr-operator-bank findings page
(finding no. 16), the section reporting the K-axis book closing
(NOVEL_ARCH_WATERFALL.md Sec 11.5's Q2 covariate result, cross-referenced
by Sec 11.6's K=32 budget-rescue dissociation probe that ends the K-axis
program; pre-registration NCR_MAPPING_LAW_DESIGN.md Sec 2.1).

This does NOT plot the K=32 budget-rescue probe's own numbers directly --
that result is a table (front pinned at 29 in all 12 cells across three
budgets; Gate-1 0/4 -> 1/4 -> 2/4, an ANOMALY-gated non-verdict), not a
correlation, and is reported as a table in the page prose instead. This
figure instead visualizes the ONE genuine positive that fell out of the
same wave: far-depth composition reliability is PREDICTABLE from a
write-residual metric the model already produces at ordinary convergence
time, at K=24/d=25 (the tight-spare mapping that CONFIRMED there),
extended to n=12 seeds (the original 4 from the next-lever wave's Probe A
plus 8 new seeds from the mapping-law wave's Q2 seed extension).

Figure (ncr_k32_closed.svg): two scatter panels sharing one x-axis (the
converged write residual, phase_resid_max_mean, log scale) across all 12
K=24/d=25 seeds.
  Left  -- x vs failure_front_h (the ladder rung composition survives to,
           of 21/45/93/189/... up to h*=189). Lower residual -> further
           front, visibly monotonic-ish; annotated with Spearman
           rho(delta, front).
  Right -- x vs sweep_min_recovered (the strict whole-sweep-window
           metric; HOLD requires >=0.9, dashed line). Same x-axis;
           annotated with Spearman rho(delta, sweep_min_rec).

Both correlations are recomputed here directly from the 12 raw cell
JSONs (scipy.stats.spearmanr), not copied from the registry, and then
cross-checked against NOVEL_ARCH_WATERFALL.md Sec 11.5's own cited
values (rho=-0.8771 front, rho=-0.8734 sweep_min_rec) as a hard
assertion with generous float tolerance.

Data sources (raw cell JSONs, md5-verified against each directory's own
committed manifest before use; delta/front/sweep_min_rec pulled directly
from deep_probe.phase_resid_max_mean / eval.failure_front_h /
eval.reducer_signature.sweep_min_recovered on every cell, never trusted
from a harvest summary alone):
  experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/
    earlyln_K24_s{0-3}.json                  (Probe A, d=K+1, 1x -- the
                                               original 4 seeds)
  experiment-runs/2026-07-12_ncr_nextlever_wave/md5_manifest.txt
  experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/
    earlyln_K24_s{4-11}.json                 (Q2 seed extension, same
                                               K=24/d=25/1x recipe, 8 new
                                               seeds)
  experiment-runs/2026-07-12_ncr_mappinglaw_wave1/md5_manifest.txt

Palette: Okabe-Ito (colorblind-safe). No in-figure title (caption lives
in the HTML). Background matches the site (#FAF5E7).

Also writes ncr_k32_closed_x.png -- a standalone 1200x675 PNG (X's
summary_large_image ratio) with an in-figure title and a footer URL, for
the social-post attachment. Same underlying panels, larger fonts; not
used on the site itself.
"""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent

NEXTLEVER_DIR = REPO / "experiment-runs/2026-07-12_ncr_nextlever_wave"
NEXTLEVER_MANIFEST = NEXTLEVER_DIR / "md5_manifest.txt"

MAPLAW_DIR = REPO / "experiment-runs/2026-07-12_ncr_mappinglaw_wave1"
MAPLAW_MANIFEST = MAPLAW_DIR / "md5_manifest.txt"

# ------------------------------------------------------------- md5 gates
# both manifests are BSD `md5 -r` style: "MD5 (./path) = hash"
def load_bsd_manifest(path):
    m = {}
    for line in path.read_text().splitlines():
        mm = re.match(r"MD5 \((.+)\) = ([0-9a-f]+)", line.strip())
        if mm:
            p = mm.group(1)
            p = p[2:] if p.startswith("./") else p
            m[p] = mm.group(2)
    return m


nextlever_manifest = load_bsd_manifest(NEXTLEVER_MANIFEST)
orig_files = [f"dratio/earlyln_K24_s{s}.json" for s in range(4)]
for fn in orig_files:
    fp = NEXTLEVER_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = nextlever_manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(orig_files)} files from 2026-07-12_ncr_nextlever_wave")

maplaw_manifest = load_bsd_manifest(MAPLAW_MANIFEST)
seedext_files = [f"q2_K24_seedext/earlyln_K24_s{s}.json" for s in range(4, 12)]
for fn in seedext_files:
    fp = MAPLAW_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = maplaw_manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
print(f"md5-verified {len(seedext_files)} files from 2026-07-12_ncr_mappinglaw_wave1")


def load(path):
    with open(path) as f:
        return json.load(f)


cells = (
    [load(NEXTLEVER_DIR / fn) for fn in orig_files]
    + [load(MAPLAW_DIR / fn) for fn in seedext_files]
)
assert len(cells) == 12, f"expected 12 K=24/d=25 seeds, got {len(cells)}"

deltas, fronts, sweeps, seeds = [], [], [], []
for c in cells:
    assert c["K"] == 24 and c["d"] == 25, f"unexpected (K,d): ({c['K']}, {c['d']})"
    deltas.append(float(c["deep_probe"]["phase_resid_max_mean"]))
    fronts.append(float(c["eval"]["failure_front_h"]))
    sweeps.append(float(c["eval"]["reducer_signature"]["sweep_min_recovered"]))
    seeds.append(c["seed"])

print(f"seeds present: {sorted(seeds)}")
assert sorted(seeds) == list(range(12)), "expected seeds 0-11 exactly once each"

rho_front, p_front = spearmanr(deltas, fronts)
rho_sweep, p_sweep = spearmanr(deltas, sweeps)
print(f"Spearman rho(delta, front) = {rho_front:.4f} (p={p_front:.4g})")
print(f"Spearman rho(delta, sweep_min_rec) = {rho_sweep:.4f} (p={p_sweep:.4g})")

# cross-check against NOVEL_ARCH_WATERFALL.md Sec 11.5's own cited values
assert abs(rho_front - (-0.8771)) < 0.01, f"rho_front mismatch vs registry: {rho_front}"
assert abs(rho_sweep - (-0.8734)) < 0.01, f"rho_sweep mismatch vs registry: {rho_sweep}"
print("cross-checked both Spearman rho values against registry Sec 11.5 (0.8771 / 0.8734)")

# ---------------------------------------------------------------- figure
H_STAR = 189


def draw_panels(ax1, ax2, label_fs=9.5, tick_fs=8.7, ann_fs=10.5, dot_size=75):
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_xscale("log")
        ax.set_xlabel("write residual $\\delta$ = phase_resid_max_mean (log scale)", fontsize=label_fs)
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color(TEXT)

    # ---- left: delta vs failure front
    ax1.scatter(deltas, fronts, color=OI_BLUE, edgecolor=TEXT, linewidth=0.9,
                s=dot_size, zorder=4)
    ax1.axhline(H_STAR, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    ax1.text(max(deltas) * 1.05, H_STAR - 4, "h* = 189", fontsize=ann_fs - 1.5,
              color=MUTED, va="top", ha="left")
    ax1.set_ylabel("failure front $h$ (ladder rung composition survives to)", fontsize=label_fs)
    ax1.set_ylim(0, 210)
    ax1.text(0.05, 0.06, f"Spearman ρ = {rho_front:.3f}\n(n=12)", transform=ax1.transAxes,
              fontsize=ann_fs, color=TEXT, fontweight="bold", va="bottom", ha="left",
              bbox=dict(facecolor=BG, edgecolor=TEXT, linewidth=0.8, boxstyle="round,pad=0.35"))

    # ---- right: delta vs sweep_min_recovered
    ax2.scatter(deltas, sweeps, color=OI_VERMILLION, edgecolor=TEXT, linewidth=0.9,
                s=dot_size, zorder=4)
    ax2.axhline(0.9, color=TEXT, linestyle="--", linewidth=1.1, alpha=0.6, zorder=2)
    ax2.text(max(deltas) * 1.05, 0.87, "HOLD bar", fontsize=ann_fs - 1.5, color=MUTED,
              va="top", ha="left")
    ax2.set_ylabel("sweep_min_recovered@0.9 (strict far-depth reliability)", fontsize=label_fs)
    ax2.set_ylim(-0.05, 1.08)
    ax2.text(0.05, 0.92, f"Spearman ρ = {rho_sweep:.3f}\n(n=12)", transform=ax2.transAxes,
              fontsize=ann_fs, color=TEXT, fontweight="bold", va="top", ha="left",
              bbox=dict(facecolor=BG, edgecolor=TEXT, linewidth=0.8, boxstyle="round,pad=0.35"))

    for ax in (ax1, ax2):
        ax.tick_params(axis="both", labelsize=tick_fs)


# ---- site figure (no in-figure title; caption lives in the HTML)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.9), facecolor=BG)
draw_panels(ax1, ax2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ncr_k32_closed.svg", format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote ncr_k32_closed.svg")

# ---- standalone PNG for social (1200x675, X's summary_large_image ratio)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=100, facecolor=BG)
draw_panels(ax1, ax2, label_fs=12.0, tick_fs=10.8, ann_fs=13.5, dot_size=120)
fig.suptitle("Far-depth composition is predictable from write residual alone (ρ ≈ -0.87, n=12)",
             fontsize=15.0, fontweight="bold", color=TEXT, x=0.02, ha="left", y=0.995)
fig.text(0.99, 0.015, "pebbleml.com/findings/ncr-operator-bank.html", fontsize=10.5,
          color=MUTED, ha="right")
plt.tight_layout(rect=(0, 0.03, 1, 0.94))
plt.savefig(f"{OUT_DIR}/ncr_k32_closed_x.png", format="png", facecolor=BG, dpi=100)
plt.close(fig)
print("wrote ncr_k32_closed_x.png")
