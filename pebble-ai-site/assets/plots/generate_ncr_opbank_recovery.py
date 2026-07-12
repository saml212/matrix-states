"""
Generate the money figure for the ncr-operator-bank findings page (finding
no. 16), section 04: "Diagnosis: an annealed LayerNorm recovers
convergence."

Figure (ncr_opbank_recovery.svg): train loss vs. optimizer step (log-y) for
all four single-seed convergence-diagnosis arms at the proven 256-batch /
80K-step budget — earlyln, baseline (control), warmup, curriculum. This is
the most honest available visualization of "recovery": all four arms use
the identical architecture and budget, and only earlyln's loss actually
leaves the neighborhood of 1.0 (a sharp phase transition at step
~8000-8500, then continued descent to 0.0052); the other three sit flat
in the [0.88, 1.02] band for the full 80K steps. This is a genuine
per-step (every 500 steps) train loss trajectory logged by the run itself
— not a derived or resampled curve.

Data source (raw per-cell JSONs, md5-verified against the committed
manifest before use):
  experiment-runs/2026-07-11_ncr_opbank_recover/ncropbank_recover_{earlyln,
    baseline,warmup,curriculum}.json  (each cell's train.loss_history)
  experiment-runs/2026-07-11_ncr_opbank_recover/recover_verdict.json
    (cross-check: each arm's train_final_loss must match the last
    loss_history point bit-for-bit)
  experiment-runs/2026-07-11_ncr_opbank_recover/md5_manifest.txt

Palette: Okabe-Ito (colorblind-safe). No in-figure title (caption lives in
the HTML). Background matches the site (#FAF5E7).
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
RAW_DIR = REPO / "experiment-runs/2026-07-11_ncr_opbank_recover"
MANIFEST = RAW_DIR / "md5_manifest.txt"
VERDICT = RAW_DIR / "recover_verdict.json"

# ------------------------------------------------------------- md5 gate
manifest = {}
for line in MANIFEST.read_text().splitlines():
    if not line.strip():
        continue
    h, name = line.split(None, 1)
    manifest[name.strip()] = h

ARMS = ["earlyln", "baseline", "warmup", "curriculum"]
cell_files = [f"ncropbank_recover_{arm}.json" for arm in ARMS]
for fn in cell_files:
    fp = RAW_DIR / fn
    actual = hashlib.md5(fp.read_bytes()).hexdigest()
    expected = manifest.get(fn)
    assert expected == actual, f"MD5 MISMATCH {fn}: manifest={expected} actual={actual}"
    print(f"md5-verified {fn}: {actual}")

with open(VERDICT) as f:
    verdict = json.load(f)

# ------------------------------------------------------- load loss curves
cells = {}
for arm in ARMS:
    with open(RAW_DIR / f"ncropbank_recover_{arm}.json") as f:
        cells[arm] = json.load(f)

loss_curves = {arm: cells[arm]["train"]["loss_history"] for arm in ARMS}

# cross-check: last loss_history point reproduces recover_verdict.json's
# train_final_loss bit-for-bit, for every arm, before anything is plotted.
for arm in ARMS:
    last_step, last_loss = loss_curves[arm][-1]
    ref = verdict["cells"][arm]["train_final_loss"]
    assert last_loss == ref, f"{arm}: last loss_history point {last_loss} != verdict train_final_loss {ref}"
print("recomputed final-step losses reproduce recover_verdict.json bit-for-bit for all 4 arms")

# also cross-check the other headline numbers plotted nowhere but quoted in
# the caption, so the figure and the prose can never silently diverge.
for arm in ARMS:
    v = verdict["cells"][arm]
    d = cells[arm]
    assert d["relation_id_swap"]["right_minus_wrong_gap"] == v["swap_gap"]
    lo = min(pr["A_eff_rank"][i] for pr in d["deep_probe_per_relation"].values() for i in range(4))
    hi = max(pr["A_eff_rank"][i] for pr in d["deep_probe_per_relation"].values() for i in range(4))
    assert abs(lo - v["A_eff_rank_min"]) < 1e-9 and abs(hi - v["A_eff_rank_max"]) < 1e-9
print("swap-gap and effective-rank figures cross-checked against recover_verdict.json bit-for-bit")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor=BG)
ax.set_facecolor(BG)

style = {
    "earlyln": dict(color=OI_VERMILLION, label="earlyln — RECOVERED (final loss 0.0052)"),
    "baseline": dict(color=OI_BLUE, label="baseline (control) — FAIL (final loss 0.884)"),
    "warmup": dict(color=OI_SKY, label="warmup — FAIL (final loss 0.982)"),
    "curriculum": dict(color=OI_PURPLE, label="curriculum — FAIL (final loss 0.986)"),
}

for arm in ARMS:
    steps = [p[0] for p in loss_curves[arm]]
    losses = [p[1] for p in loss_curves[arm]]
    st = style[arm]
    zorder = 5 if arm == "earlyln" else 3
    lw = 2.4 if arm == "earlyln" else 1.8
    ax.plot(steps, losses, color=st["color"], linewidth=lw, label=st["label"],
             zorder=zorder, alpha=0.95)

# mark earlyln's transition window (loss leaves the ~1.0 neighborhood
# between step 8000 and 8500 in the raw log)
ax.axvspan(8000, 8500, color=OI_VERMILLION, alpha=0.10, zorder=1)
ax.annotate("earlyln transitions\nhere (step 8000–8500)",
            xy=(8250, 0.09), xytext=(18500, 0.28),
            fontsize=9, color=OI_VERMILLION, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=OI_VERMILLION, linewidth=1.2),
            ha="left", va="center")

ax.set_yscale("log")
ax.set_xlim(0, 80000)
ax.set_ylim(0.0035, 1.15)
ax.set_xlabel("optimizer step (proven 256-batch / 80K-step budget)", fontsize=10.5, labelpad=8)
ax.set_ylabel("train loss (log scale)", fontsize=10.5, labelpad=8)
ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
ax.grid(True, which="minor", linestyle="-", linewidth=0.3, alpha=0.12, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

ax.tick_params(labelsize=9.5)
ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x/1000)}k" if x > 0 else "0")

legend = ax.legend(loc="upper right", frameon=True, fontsize=9.2,
                    facecolor=BG, edgecolor=TEXT, framealpha=0.96)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ncr_opbank_recovery.svg", format="svg", facecolor=BG,
            bbox_inches="tight")
plt.close(fig)
print("wrote ncr_opbank_recovery.svg")
