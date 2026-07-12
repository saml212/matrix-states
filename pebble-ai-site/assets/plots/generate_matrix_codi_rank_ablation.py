"""
Generate the rank-k projection ablation figure for "The gradient does not
see rank" (matrix-CODI rank-blindness, ICML 2026 MI workshop).

Figure (matrix_codi_rank_ablation.svg): accuracy vs. forced rank k in
{1, 2, 4, 8, 16} on 128 ProsQA test problems, for the four positive-control
readouts nonlinear in Z (bilinear, bilinear+GELU, SVD-augmented, quadratic
in ZZ^T) trained at gamma=0, seed 1337 (Section 06 / Table 2 / Figure 2 of
the findings page). All four curves are flat -- this is the headline
negative result the readout-Jacobian argument (Section 04) predicts.

Data source (raws, md5-printed below):
  experiment-runs/2026-04-17_round_pc/rank_evals/bilinear_rankeval.json
  experiment-runs/2026-04-17_round_pc/rank_evals/bilinear_gelu_rankeval.json
  experiment-runs/2026-04-17_round_pc/rank_evals/svd_aug_rankeval.json
  experiment-runs/2026-04-17_round_pc/rank_evals/quadratic_rankeval.json

Each file's "results_by_k" accuracy is RECOMPUTED here from the per-problem
"records" list (correct-count / n_problems) and cross-checked bit-for-bit
against the file's own stored "accuracy" field, and against the already-
published Table 2 / Figure 2 values on the same HTML page, before anything
is plotted.

NOTE on the fifth ("flatten") series in the original hand-drawn figure:
Table 2 on the page states a flatten baseline of 79.0% flat across all k.
No raw rank_projection_ablation-style JSON reproducing that exact 128-
problem eval (readout=flatten, seed=1337, gamma=0) could be found anywhere
under experiment-runs/ (checked 2026-04-17_round_pc, 2026-04-15_round_pc,
and the three sibling four-condition rank_projection_ablation.json files
in 2026-04-10/12_matrix_codi_round*). 79.0% is also not an exact multiple
of 1/128 (0.79 * 128 = 101.12, non-integer), unlike all four verified
series below, which land on exact n/128 fractions. This is flagged to the
human in the task report; the flatten value is plotted here ONLY as a
clearly-labeled unverified reference line sourced from the page's own
Table 2, not as a raw-data series.

Palette: Okabe-Ito (colorblind-safe). No in-figure title (caption lives in
the HTML). Background matches the site (#FAF5E7).
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent
RAW_DIR = REPO / "experiment-runs/2026-04-17_round_pc/rank_evals"

READOUTS = {
    "bilinear": {
        "file": "bilinear_rankeval.json",
        "label": "bilinear (linear in Z, control)",
        "color": OI_BLUE,
        "marker": "o",
    },
    "bilinear_gelu": {
        "file": "bilinear_gelu_rankeval.json",
        "label": "bilinear + GELU (nonlinear in Z)",
        "color": OI_VERMILLION,
        "marker": "s",
    },
    "svd_aug": {
        "file": "svd_aug_rankeval.json",
        "label": "SVD-augmented (exposes rank)",
        "color": OI_GREEN,
        "marker": "^",
    },
    "quadratic": {
        "file": "quadratic_rankeval.json",
        "label": "quadratic in Z Zᵀ",
        "color": OI_ORANGE,
        "marker": "D",
    },
}

K_VALUES = [1, 2, 4, 8, 16]

# ---------------------------------------------------------- load + md5 gate
loaded = {}
for name, spec in READOUTS.items():
    fp = RAW_DIR / spec["file"]
    raw = fp.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    print(f"md5 {md5}  {fp.relative_to(REPO)}")
    d = json.loads(raw)
    loaded[name] = d

# ------------------------------------------------ recompute + cross-check
# Table 2 / Figure 2 values as already published on the findings page
# (matrix-codi-rank-blindness.html, section 06), used only as a
# cross-check target -- never as a plotting source.
PUBLISHED_ACCURACY_PCT = {
    "bilinear":      [78.12, 78.91, 78.91, 78.12, 78.12],
    "bilinear_gelu": [78.91, 79.69, 79.69, 79.69, 79.69],
    "svd_aug":       [77.34, 78.12, 78.12, 77.34, 78.12],
    "quadratic":     [79.69, 79.69, 79.69, 79.69, 79.69],
}
PUBLISHED_SPEARMAN_P = {
    "bilinear": 0.63,
    "bilinear_gelu": 0.14,
    "svd_aug": 0.82,
    "quadratic": 0.46,
}

curves = {}
for name, d in loaded.items():
    accs = []
    for k in K_VALUES:
        cell = d["results_by_k"][str(k)]
        n = cell["n_problems"]
        assert n == 128, f"{name} k={k}: expected n_problems=128, got {n}"
        # recompute accuracy straight from the per-problem records, not the
        # file's own cached "accuracy" field
        recomputed = sum(1 for r in cell["records"] if r["correct"]) / n
        assert abs(recomputed - cell["accuracy"]) < 1e-12, (
            f"{name} k={k}: recomputed accuracy {recomputed} != stored {cell['accuracy']}"
        )
        accs.append(recomputed * 100.0)
    curves[name] = accs

    # cross-check against the already-published Table 2 / Figure 2 numbers
    # (rounded to 2 dp on the page)
    for k_idx, (got, published) in enumerate(zip(accs, PUBLISHED_ACCURACY_PCT[name])):
        assert abs(round(got, 2) - published) < 0.011, (
            f"{name} k={K_VALUES[k_idx]}: computed {got:.4f}% vs published {published}% -- MISMATCH"
        )
    p_val = loaded[name]["spearman"]["p_value"]
    assert abs(round(p_val, 2) - PUBLISHED_SPEARMAN_P[name]) < 0.011, (
        f"{name}: spearman p {p_val} vs published {PUBLISHED_SPEARMAN_P[name]} -- MISMATCH"
    )

print("all four readouts' recomputed accuracy-vs-k curves and Spearman "
      "p-values reproduce the published Table 2 / Figure 2 numbers exactly")

# unverified reference line -- see module docstring
FLATTEN_REFERENCE_PCT = 79.0

# --------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=BG)
ax.set_facecolor(BG)

ax.axhline(FLATTEN_REFERENCE_PCT, color=TEXT, linewidth=1.3,
           linestyle="--", alpha=0.6, zorder=1)
ax.annotate("flatten baseline, 79.0% (Table 2 reference --\nno raw JSON found, see caption)",
            (K_VALUES[0], FLATTEN_REFERENCE_PCT), fontsize=8, color=MUTED,
            ha="left", va="bottom", xytext=(2, 4), textcoords="offset points")

for name, spec in READOUTS.items():
    ax.plot(K_VALUES, curves[name], color=spec["color"], linewidth=2.0,
             marker=spec["marker"], markersize=6.5, markeredgecolor=TEXT,
             markeredgewidth=0.6, label=spec["label"], zorder=4)

ax.set_xscale("log", base=2)
ax.set_xticks(K_VALUES)
ax.get_xaxis().set_major_formatter(mticker.FixedFormatter([str(k) for k in K_VALUES]))
ax.minorticks_off()
ax.set_xlabel("forced rank k (SVD truncation at inference)", fontsize=10, labelpad=8)
ax.set_ylabel("ProsQA accuracy (%, n=128)", fontsize=10, labelpad=8)
ax.set_ylim(75.5, 80.8)
ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)
ax.legend(loc="lower left", fontsize=8.3, frameon=True, facecolor=BG,
          edgecolor=TEXT).get_frame().set_linewidth(1.0)

plt.tight_layout()
out_path = OUT_DIR / "matrix_codi_rank_ablation.svg"
plt.savefig(out_path, format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_path}")
