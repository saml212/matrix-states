"""
Generate two SVGs for the parameter-efficiency findings page.

FIGURE 1 (parameter_efficiency.svg) — same concept as the April-2026
version: parameters per projection at matrix dimension d=16 for four
projection families (Flatten->Linear, Kronecker K=8, Kronecker K=4,
RowThenCol bilinear).

Honesty note on sourcing (per the task's algebra-vs-data distinction):
these four numbers are NOT an experimental measurement — they are a
closed-form arithmetic function of d and K (params = d^4 for the
flattened Linear, 2*d^2 for the RowThenCol bilinear sandwich, 2*K*d^2
for a K-term Kronecker sum). There is no raw run to md5-verify because
no run produced these numbers; they are computed in this script from
the same formulas the architecture uses. What we DO verify against a
real committed file is that this script's computed values still match
the table published in research/matrix-native-projections.md (a project
research note, not a peer-reviewed source) — a regression check on the
documentation, not a data-provenance check on a measurement. Both that
file's md5 and its parsed table are printed so a future edit to either
the formula or the note is visible, not silently trusted.

FIGURE 2 (parameter_efficiency_configs.svg) — NEW content this page did
not previously have: eval BPB vs total params for the 5 real configs
(A-E) of the byte-level thought-interleaving sweep, read directly from
experiment-runs/8xh100-session1/sweep_all_results.json (5 concatenated
JSON objects, not a JSON array/JSONL — parsed with json.JSONDecoder.
raw_decode in a loop) and cross-checked against the independent
human-written sweep_all_summaries.txt in the same directory.

Palette: Okabe-Ito (colorblind-safe). No in-figure titles (captions
live in the HTML). Background matches the site (#FAF5E7).
"""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sanity_print(path: Path):
    size = path.stat().st_size
    assert size > 0, f"{path} is empty — refusing to trust a zero-byte source file"
    print(f"source: {path.relative_to(REPO)}  md5={md5_of(path)}  {size} bytes")


# =====================================================================
# FIGURE 1: parameters per projection at d=16 — closed-form arithmetic
# =====================================================================
D = 16

rows = [
    ("Flatten -> Linear\n(d^2 x d^2)", D ** 4, "baseline"),
    ("Kronecker K=8\n(8 x A_k @ M @ B_k)", 2 * 8 * D ** 2, None),
    ("Kronecker K=4\n(4 x A_k @ M @ B_k)", 2 * 4 * D ** 2, None),
    ("RowThenCol bilinear\nsilu(A @ M) @ B", 2 * D ** 2, None),
]
for i, (label, val, note) in enumerate(rows):
    if note is None:
        ratio = rows[0][1] / val
        rows[i] = (label, val, f"{ratio:.0f}x fewer")

# Cross-check against the committed research note (documentation
# regression check, not a data-provenance check — see docstring).
PROJECTIONS_NOTE = REPO / "research/matrix-native-projections.md"
sanity_print(PROJECTIONS_NOTE)
note_text = PROJECTIONS_NOTE.read_text()
published = {
    "Flatten→Linear": None,
    "Bilinear (K=1)": None,
    "Kronecker K=4": None,
    "Kronecker K=8": None,
}
for m in re.finditer(r"\|\s*\**([\w→() =]+?)\**\s*\|\s*\**([\d,]+)\**\s*\|\s*\**([\d,]+)\**\s*\|", note_text):
    label = m.group(1).strip()
    if label in published:
        published[label] = int(m.group(2).replace(",", ""))

assert published["Flatten→Linear"] == D ** 4, published
assert published["Bilinear (K=1)"] == 2 * D ** 2, published
assert published["Kronecker K=4"] == 2 * 4 * D ** 2, published
assert published["Kronecker K=8"] == 2 * 8 * D ** 2, published
print(f"figure 1: closed-form params at d={D} match research/matrix-native-projections.md's "
      f"published table: {published}")

labels = [r[0] for r in rows]
values = [r[1] for r in rows]
notes = [r[2] for r in rows]
colors = [MUTED, OI_ORANGE, OI_ORANGE, OI_VERMILLION]

fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=BG)
ax.set_facecolor(BG)

y_positions = list(range(len(rows)))
y_positions.reverse()

bars = ax.barh(
    y_positions, values,
    color=colors, edgecolor=TEXT, linewidth=1.0, height=0.62, zorder=3,
)

for bar, val, note in zip(bars, values, notes):
    y = bar.get_y() + bar.get_height() / 2
    ax.annotate(f"{val:,}", (val, y), xytext=(6, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9, color=TEXT, fontweight="bold")
    ax.annotate(note, (val, y), xytext=(6, -11), textcoords="offset points",
                va="center", ha="left", fontsize=7.5, color=MUTED, style="italic")

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xscale("log")
ax.set_xlim(200, 400_000)
ax.set_xlabel(f"parameters per projection (d = {D}, log scale)", fontsize=10, labelpad=8)

ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.015, 0.015,
    f"closed-form arithmetic at d={D} (params = d^4 flattened, 2*K*d^2 Kronecker-K, "
    "2*d^2 RowThenCol) — not an experimental measurement. Cross-checked against the "
    "published table in research/matrix-native-projections.md.",
    fontsize=6.6, color=MUTED, style="italic",
)

plt.tight_layout(rect=(0, 0.05, 1, 1))
out1 = OUT_DIR / "parameter_efficiency.svg"
plt.savefig(out1, format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out1}")

# =====================================================================
# FIGURE 2 (NEW): BPB vs params, 5-config sweep (A-E), real data
# =====================================================================
SWEEP_JSON = REPO / "experiment-runs/8xh100-session1/sweep_all_results.json"
SWEEP_SUMMARY = REPO / "experiment-runs/8xh100-session1/sweep_all_summaries.txt"
sanity_print(SWEEP_JSON)
sanity_print(SWEEP_SUMMARY)

# sweep_all_results.json is 5 concatenated `json.dumps(..., indent=2)`
# objects back to back (not a JSON array, not JSONL) — decode in a loop.
text = SWEEP_JSON.read_text()
dec = json.JSONDecoder()
configs = []
i, n = 0, len(text)
while i < n:
    while i < n and text[i] in " \t\n\r":
        i += 1
    if i >= n:
        break
    obj, end = dec.raw_decode(text, i)
    configs.append(obj)
    i = end
assert len(configs) == 5, f"expected 5 concatenated config objects, got {len(configs)}"

# cross-check every config against the independent human-written summary
# file (same archive, written by a different code path at run time).
summary_text = SWEEP_SUMMARY.read_text()
for c in configs:
    params_comma = f"{c['params']:,}"
    assert params_comma in summary_text, f"{c['name']}: params {params_comma} not found in summary"
    bpb_3dp = f"{c['bpb']:.3f}"
    assert bpb_3dp in summary_text, f"{c['name']}: BPB {bpb_3dp} not found in summary"
print(f"figure 2: {len(configs)}/5 configs cross-checked against sweep_all_summaries.txt "
      "(params + BPB substring match)")

configs.sort(key=lambda c: c["params"])
names = [c["name"].split("_")[0] for c in configs]  # A, B, C, D, E
params = [c["params"] for c in configs]
bpb = [c["bpb"] for c in configs]
no_thought_bpb = [c["no_thought_bpb"] for c in configs]
byte_rank = [c["ranks"]["byte_rank"] for c in configs]
n_thoughts = [c["config"]["n_thoughts"] for c in configs]

fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor=BG)
ax.set_facecolor(BG)

# dumbbell connector from no-thought baseline BPB down/up to the
# with-thinking BPB, per config (E has n_thoughts=0 so the two coincide).
for x, y0, y1 in zip(params, no_thought_bpb, bpb):
    ax.plot([x, x], [y0, y1], color=MUTED, linewidth=1.2, linestyle=":",
             alpha=0.7, zorder=2)

ax.scatter(params, no_thought_bpb, s=70, color=OI_SKY, edgecolor=TEXT,
           linewidth=0.8, zorder=3, label="no-thought baseline BPB")
ax.scatter(params, bpb, s=90, color=OI_VERMILLION, edgecolor=TEXT,
           linewidth=0.8, zorder=4, label="eval BPB (with thinking)")

for x, y, name, nt, br in zip(params, bpb, names, n_thoughts, byte_rank):
    ax.annotate(f"{name} (N={nt})\nbyte_rank {br:.2f}", (x, y),
                xytext=(9, 8), textcoords="offset points",
                fontsize=7.6, color=TEXT, ha="left", va="bottom")

ax.set_xlabel("total params", fontsize=10, labelpad=8)
ax.set_ylabel("eval BPB (lower is better)", fontsize=10, labelpad=8)
ax.set_xlim(min(params) * 0.9, max(params) * 1.28)
ax.set_ylim(min(bpb + no_thought_bpb) * 0.985, max(bpb + no_thought_bpb) * 1.03)

ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend = ax.legend(loc="upper right", frameon=True, fontsize=8.5,
                    facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

fig.text(
    0.015, 0.015,
    "5 configs (A-E), byte-level d=16 thought-interleaving sweep, 8xH100, 3000 steps each. "
    "Source: experiment-runs/8xh100-session1/sweep_all_results.json, cross-checked against "
    "sweep_all_summaries.txt. Config E (N=0, just more layers, no thinking) reaches the lowest "
    "absolute BPB of the five.",
    fontsize=6.6, color=MUTED, style="italic",
)

plt.tight_layout(rect=(0, 0.05, 1, 1))
out2 = OUT_DIR / "parameter_efficiency_configs.svg"
plt.savefig(out2, format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out2}")
