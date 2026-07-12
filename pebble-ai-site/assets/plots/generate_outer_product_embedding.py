"""
Generate outer_product_embedding.svg for the outer-product embedding finding.

Shows T=1 BPB (byte-per-byte loss) for matrix outer-product embedding vs
flat-vector baselines across the same three comparisons the findings page
describes (Run 22 param-matched, Round 2 Run 12 vs Run 13, Run 18
param-asymmetric). Unlike the April-2026 version of this script, no number
here is a hand-typed Python literal — every value is READ at plot time from
a real committed project file and printed with its md5 so a stale or edited
source is visible in the script's own stdout, not silently trusted.

Data sources (all repo-relative to this file's REPO root):

  - Round 2 (Run 12 vs Run 13): params and T=1 val_loss/val_ppl are parsed
    directly from the archived per-run result JSONs —
    experiment-runs/8xh100-session1/round2_matrix_results_FULL.json
    experiment-runs/8xh100-session1/loopformer_3000steps_results_FULL.json
    (both files carry a one-line "=== ... ===" banner before the JSON body;
    we slice from the first "{" rather than stripping a fixed prefix).
    T=1 BPB (2.12 / 4.29) is not itself in either JSON (no bytes-per-token
    field is recorded anywhere in the archive for this corpus), so it is
    read from the project's own published conversion in
    research/project-analysis-march2026.md's comparison table, and cross
    -checked against the JSONs' val_ppl (round(val_ppl) must equal the
    table's PPL) before being trusted.

  - Run 22 (param-matched, headline) and Run 18 (param-asymmetric,
    flagged): the raw run22/run18 script+result archives predate the
    2026-07-04 experiment-runs/ hybrid-archive policy and are not present
    on disk (checked: no run18/run22 directory under experiment-runs/ or
    the SSD mirror). The only surviving record of these two runs is
    EXPERIMENT_LOG.md (the project's committed, version-controlled run
    log). Params and T=1 BPB for both runs are parsed out of that file's
    "## Run 22" / "## Run 18" sections by regex rather than retyped, and
    Run 18's matrix-side param count (2.4M, stated in EXPERIMENT_LOG.md
    only as "10x our matrix model") is cross-checked against
    research/project-analysis-march2026.md, which states it directly
    ("the matrix model at 2.4M params").

Output: SVG at pebble-ai-site/assets/plots/outer_product_embedding.svg
"""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"
CODE_BG = "#f0e9d3"

# Okabe-Ito (colorblind-safe)
OI_VERMILLION = "#D55E00"
OI_BLUE = "#0072B2"

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent

ROUND2_MATRIX_JSON = REPO / "experiment-runs/8xh100-session1/round2_matrix_results_FULL.json"
ROUND2_LOOPFORMER_JSON = REPO / "experiment-runs/8xh100-session1/loopformer_3000steps_results_FULL.json"
EXPERIMENT_LOG = REPO / "EXPERIMENT_LOG.md"
PROJECT_ANALYSIS = REPO / "research/project-analysis-march2026.md"


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sanity_print(path: Path):
    h = md5_of(path)
    size = path.stat().st_size
    assert size > 0, f"{path} is empty — refusing to trust a zero-byte source file"
    print(f"source: {path.relative_to(REPO)}  md5={h}  {size} bytes")


for p in (ROUND2_MATRIX_JSON, ROUND2_LOOPFORMER_JSON, EXPERIMENT_LOG, PROJECT_ANALYSIS):
    sanity_print(p)

# --------------------------------------------------------------- Round 2
# Both FULL.json files have a one-line banner before the JSON body.
def load_banner_json(path: Path) -> dict:
    text = path.read_text()
    start = text.index("{")
    return json.loads(text[start:])


round2_matrix = load_banner_json(ROUND2_MATRIX_JSON)
round2_loopformer = load_banner_json(ROUND2_LOOPFORMER_JSON)

r2_matrix_params = round2_matrix["params"]
r2_flat_params = round2_loopformer["params"]
r2_matrix_t1_ppl = round2_matrix["final_evals"]["T1"]["val_ppl"]
r2_flat_t1_ppl = round2_loopformer["final_evals"]["L1"]["val_ppl"]
assert r2_matrix_params == 5155960, r2_matrix_params
assert r2_flat_params == 5330400, r2_flat_params
print(f"Round 2 (from JSON): matrix params={r2_matrix_params}, T=1 val_ppl={r2_matrix_t1_ppl:.2f}; "
      f"loopformer params={r2_flat_params}, L=1 val_ppl={r2_flat_t1_ppl:.2f}")

analysis_text = PROJECT_ANALYSIS.read_text()
m = re.search(
    r"Round 2 \(5\.16M params, MultiProbeHead\) \| PPL (\d+) / BPB ([\d.]+) \| "
    r"PPL ([\d,]+) / BPB ([\d.]+) \(LoopFormer\)",
    analysis_text,
)
assert m, "Round 2 comparison row not found in research/project-analysis-march2026.md"
table_matrix_ppl, r2_matrix_bpb, table_flat_ppl, r2_flat_bpb = (
    int(m.group(1)), float(m.group(2)), int(m.group(3).replace(",", "")), float(m.group(4))
)
r2_matrix_bpb_str, r2_flat_bpb_str = m.group(2), m.group(4)

# cross-check the BPB table's PPL against the independently loaded raw JSON
assert round(r2_matrix_t1_ppl) == table_matrix_ppl, (round(r2_matrix_t1_ppl), table_matrix_ppl)
assert round(r2_flat_t1_ppl) == table_flat_ppl, (round(r2_flat_t1_ppl), table_flat_ppl)
print(f"Round 2 BPB table cross-checked against raw JSON val_ppl: "
      f"matrix {table_matrix_ppl} PPL -> BPB {r2_matrix_bpb_str}, "
      f"loopformer {table_flat_ppl} PPL -> BPB {r2_flat_bpb_str}")

# ---------------------------------------------------------- Run 22 / Run 18
log_text = EXPERIMENT_LOG.read_text()


def section(text, heading):
    start = text.index(heading)
    end = text.find("\n## Run ", start + len(heading))
    if end == -1:
        end = len(text)
    return text[start:end]


run22 = section(log_text, "## Run 22: Param-Matched Ablation")
run22_matrix_part, run22_flat_part = run22.split("### Flat Model")

m = re.search(r"([\d,]+) params", run22_matrix_part)
r22_matrix_params = int(m.group(1).replace(",", ""))
m = re.search(r"T=1 BPB:\s*([\d.]+)", run22_matrix_part)
r22_matrix_bpb = float(m.group(1))

m = re.search(r"([\d,]+) params", run22_flat_part)
r22_flat_params = int(m.group(1).replace(",", ""))
m = re.search(r"T=1 BPB:\s*([\d.]+)", run22_flat_part)
r22_flat_bpb = float(m.group(1))

assert (r22_matrix_params, r22_matrix_bpb, r22_flat_params, r22_flat_bpb) == (
    2552788, 2.117, 5658428, 2.872
), "Run 22 parse mismatch vs EXPERIMENT_LOG.md — file may have changed"
print(f"Run 22 (from EXPERIMENT_LOG.md): matrix {r22_matrix_params} params, T=1 BPB {r22_matrix_bpb}; "
      f"flat {r22_flat_params} params, T=1 BPB {r22_flat_bpb}")

run18 = section(log_text, "## Run 18: Critical Ablation")
m = re.search(r"Params:\s*([\d.]+)M", run18)
r18_flat_params_m = float(m.group(1))
m = re.search(r"T=1 BPB:\s*(\d+\.\d+).*?T=1 BPB\s*(\d+\.\d+)", run18)
r18_flat_bpb, r18_matrix_bpb = float(m.group(1)), float(m.group(2))

# EXPERIMENT_LOG.md only states the flat model's param count directly
# ("24M ... 10x our matrix model"); cross-check the matrix-side 2.4M
# figure against the number research/project-analysis-march2026.md states
# explicitly, rather than deriving it from the "10x" prose alone.
m = re.search(
    r"matrix model at ([\d.]+)M params beats the flat-vector model at ([\d.]+)M params",
    analysis_text,
)
assert m, "Run 18 param cross-check sentence not found in project-analysis-march2026.md"
r18_matrix_params_m, r18_flat_params_m_check = float(m.group(1)), float(m.group(2))
assert r18_flat_params_m_check == r18_flat_params_m, (r18_flat_params_m_check, r18_flat_params_m)
assert abs(r18_flat_params_m / r18_matrix_params_m - 10.0) < 0.05

assert (r18_matrix_params_m, r18_flat_params_m, r18_matrix_bpb, r18_flat_bpb) == (
    2.4, 24.0, 2.18, 3.219
), "Run 18 parse mismatch vs EXPERIMENT_LOG.md / project-analysis-march2026.md"
print(f"Run 18 (from EXPERIMENT_LOG.md + project-analysis-march2026.md): "
      f"matrix {r18_matrix_params_m}M params, T=1 BPB {r18_matrix_bpb}; "
      f"flat {r18_flat_params_m}M params, T=1 BPB {r18_flat_bpb}")

# --------------------------------------------------------------- assemble
comparisons = [
    ("Run 22\nparam-matched*", r22_matrix_bpb, r22_flat_bpb,
     r22_matrix_params / 1e6, r22_flat_params / 1e6,
     f"flat has {r22_flat_params / r22_matrix_params:.1f}x more params", False),
    ("Round 2\n(Run 12 vs 13)", r2_matrix_bpb, r2_flat_bpb,
     r2_matrix_params / 1e6, r2_flat_params / 1e6,
     "tokens-matched", False),
    ("Run 18\nasymmetric", r18_matrix_bpb, r18_flat_bpb,
     r18_matrix_params_m, r18_flat_params_m,
     f"flat has {r18_flat_params_m / r18_matrix_params_m:.0f}x more params", True),
]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=BG)
ax.set_facecolor(BG)

n = len(comparisons)
bar_width = 0.36
x_positions = list(range(n))
matrix_x = [x - bar_width / 2 for x in x_positions]
baseline_x = [x + bar_width / 2 for x in x_positions]

matrix_bpbs = [c[1] for c in comparisons]
baseline_bpbs = [c[2] for c in comparisons]

bars_m = ax.bar(matrix_x, matrix_bpbs, bar_width,
                 color=OI_VERMILLION, edgecolor=TEXT, linewidth=1.0,
                 label="matrix (outer-product embedding)", zorder=3)

hatches = ["", "", "///"]
bars_b = ax.bar(baseline_x, baseline_bpbs, bar_width,
                 color=OI_BLUE, edgecolor=TEXT, linewidth=1.0,
                 hatch=hatches,
                 label="flat-vector baseline", zorder=3)

for bar, val in zip(bars_m, matrix_bpbs):
    ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, val),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9, color=OI_VERMILLION, fontweight="bold")

for bar, val in zip(bars_b, baseline_bpbs):
    ax.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, val),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9, color=OI_BLUE, fontweight="bold")

for i, c in enumerate(comparisons):
    m_params, b_params = c[3], c[4]
    ax.annotate(f"{m_params:.2f}M", (matrix_x[i], 0),
                textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=7.5, color=TEXT)
    ax.annotate(f"{b_params:.2f}M", (baseline_x[i], 0),
                textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=7.5, color=TEXT)

asym_idx = 2
ax.annotate(comparisons[asym_idx][5],
            (baseline_x[asym_idx], baseline_bpbs[asym_idx]),
            textcoords="offset points", xytext=(0, 20),
            ha="center", fontsize=7.5, color=TEXT, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", fc=CODE_BG, ec=TEXT, lw=0.8))

ax.set_xticks(x_positions)
ax.set_xticklabels([c[0] for c in comparisons], fontsize=9)
ax.set_ylabel("T=1 BPB (lower is better)", fontsize=10, labelpad=8)
ax.set_ylim(0, max(baseline_bpbs) * 1.16)
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend = ax.legend(loc="upper left", frameon=True, fontsize=8.5,
                    facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

fig.text(0.015, 0.015,
         "params (M) shown below each bar. * Run 22 flat baseline has more params than matrix. "
         "Run 18 hatched bar marks the param asymmetry. All values read from EXPERIMENT_LOG.md, "
         "research/project-analysis-march2026.md, and the archived Round 2 result JSONs at plot time.",
         fontsize=6.6, color=MUTED, style="italic")

plt.tight_layout(rect=(0, 0.04, 1, 1))

out = OUT_DIR / "outer_product_embedding.svg"
plt.savefig(out, format="svg", facecolor=BG, bbox_inches="tight")
print(f"wrote {out}")
