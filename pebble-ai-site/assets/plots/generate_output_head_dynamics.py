"""
Generate output_head_dynamics.svg for the output-head-dynamics findings page.

All three series below are loaded at runtime from the project's raw
experiment archives under experiment-runs/8xh100-session1/ -- no plotted
number in this script is hand-typed or fabricated.

  1. MultiProbeHead (bilinear output head), n=1, d=32, Run 12 (Round 2).
     Source: round2_matrix_results_FULL.json, training_curve entry for
     step=3000, evals.T8.ranks. This is the specific eval call the
     output-head-dynamics.html page cites (round2_full_train.log line
     125, "T= 8: PPL 72.4 | Rank [5.05, 5.45, 5.71, 5.86, 5.99, 6.06,
     6.11, 6.13] *BEST*"). Real per-iteration measurement, 8 points.
     (Note: the JSON's top-level "final_evals" block is a *different*,
     slightly later re-evaluation logged after training ends -- line 132,
     no *BEST* tag -- and does not match the numbers the HTML cites, so
     we deliberately select the step=3000 training_curve entry instead.)

  2. Vector-collapse output head (Run 10, Round 1, WikiText-103).
     Source: round1_matrix_results.json, final_evals.T8.ranks.
     IMPORTANT CORRECTION vs the April-2026 version of this script: the
     vector-collapse head is *not* the "model_type": "vector" run
     (round1_vector_*, which never measured rank -- its log prints
     "Rank [-]" at every step and round1_vector_results.json is empty).
     It is round1_matrix_script.py, whose forward pass does exactly the
     "vec = (collapse_W * M).sum(dim=-1); logits = output_head(vec)"
     operation the HTML describes as the vector-collapse head, on a
     matrix-native backbone that DOES log rank. Real per-iteration
     measurement exists (round1_matrix_train.log line 120, step 3000,
     "*BEST*") and is plotted here in full -- previously this script
     claimed "no per-iteration numbers were logged for this run" and drew
     a fabricated two-point illustrative line instead.

  3. 3D matrix-product attention (Run 21, d=16, OpenR1-Math).
     Source: exp_3d_attn_full_train.log, parsed at runtime for the final
     "T= 8: BPB ... | Rank [...] *BEST*" line (step 1000). Real
     per-iteration measurement exists here too -- the April-2026 version
     of this script claimed "intermediate points were not measured" for
     this run, which is false; only the two endpoints (2.75, 2.66) had
     previously been surfaced in EXPERIMENT_LOG.md/STATE.md prose.

No number in this figure is interpolated or invented. Palette is
Okabe-Ito (colorblind-safe). No in-figure title -- the caption lives in
the HTML <figcaption>.

Output: SVG at pebble-ai-site/assets/plots/output_head_dynamics.svg
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_ORANGE = "#E69F00"

RUN_DIR = "/Users/samuellarson/Experiments/learned-representations/experiment-runs/8xh100-session1"


def load_multiprobe_step3000():
    """Real per-iteration rank at the step-3000 (final) eval call
    (T8, *BEST*), round2_full_train.log line 125. This is the specific
    series the output-head-dynamics.html page's table and prose cite
    (5.05 -> 6.13)."""
    with open(f"{RUN_DIR}/round2_matrix_results_FULL.json") as f:
        text = f.read()
    data = json.loads(text[text.index("{"):])
    entry = [c for c in data["training_curve"] if c["step"] == 3000][0]
    return entry["evals"]["T8"]["ranks"]


def load_vector_collapse():
    """Real per-iteration rank, final eval (step 3000, T8, *BEST*),
    round1_matrix_train.log line 120. round1_matrix_script.py implements
    the vector-collapse readout the HTML describes."""
    with open(f"{RUN_DIR}/round1_matrix_results.json") as f:
        data = json.load(f)
    return data["final_evals"]["T8"]["ranks"]


def load_3d_matprod():
    """Real per-iteration rank, final eval (step 1000, T8, *BEST*),
    parsed from exp_3d_attn_full_train.log. Endpoints (2.75, 2.66) match
    the numbers previously reported in EXPERIMENT_LOG.md; the
    intermediate points were logged but not previously surfaced."""
    with open(f"{RUN_DIR}/exp_3d_attn_full_train.log") as f:
        log_text = f.read()
    matches = re.findall(
        r"T=\s*8: BPB [\d.]+ \| Rank \[([\d.,\s]+)\] \*BEST\*", log_text
    )
    if not matches:
        raise RuntimeError("no T=8 *BEST* eval line found in exp_3d_attn_full_train.log")
    return [float(x) for x in matches[-1].split(",")]


multiprobe = load_multiprobe_step3000()
vector_collapse = load_vector_collapse()
matprod_3d = load_3d_matprod()

iterations = list(range(1, 9))
assert len(multiprobe) == len(vector_collapse) == len(matprod_3d) == 8

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=BG)
ax.set_facecolor(BG)

# MultiProbeHead (measured at every iteration -- primary series)
ax.plot(iterations, multiprobe,
        color=OI_BLUE, linewidth=2.5, marker="o", markersize=6,
        label="MultiProbeHead (Run 12, step 3000, measured)", zorder=3)
for x, y in zip(iterations, multiprobe):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color=OI_BLUE, fontweight="bold")

# Vector-collapse head (measured, real per-iteration data)
ax.plot(iterations, vector_collapse,
        color=OI_VERMILLION, linewidth=1.8, marker="s", markersize=6,
        label="Vector-collapse (Run 10, measured)",
        zorder=2)

# 3D matrix-product attention (measured, real per-iteration data)
ax.plot(iterations, matprod_3d,
        color=OI_ORANGE, linewidth=1.5, marker="^", markersize=6,
        label="3D matrix-product (Run 21, measured)",
        zorder=1)

ax.set_xlabel("refinement iteration", fontsize=10, labelpad=8)
ax.set_ylabel("effective rank", fontsize=10, labelpad=8)
ax.set_xlim(0.5, 8.5)
ax.set_ylim(2.0, 6.8)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

legend = ax.legend(loc="center right", frameon=True, fontsize=8.0,
                   facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()

out = "/Users/samuellarson/Experiments/learned-representations/pebble-ai-site/assets/plots/output_head_dynamics.svg"
plt.savefig(out, format="svg", facecolor=BG, bbox_inches="tight")
print(f"wrote {out}")
