"""
Generate the fan-out heatmap figure (mstar_fanout_heatmap.svg) for the
constant-memory-recall findings page: a second, complementary figure to
constant_memory_horizon.svg (fig 1), drilling one level below the
MSTAR_VERDICT.json summary into the 90 raw per-cell fan-out JSONs.

Data source: 90 real per-cell eval JSONs, md5-verified against the archived
manifest:
  /Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-10_h2h_mstar/fanout/h2h_msweep_*.json
  (manifest: fanout/md5_manifest.txt, 90/90 cell entries verified here;
   96/96 total files in that manifest per the page's reproducibility section)

Sweep: M in {2,4,8,16,32} (KV-cache cap) x H in {H2,H4,H8} (454/902/1798
token horizons, 2x/4x/8x the binding window) x seed in {0,1,2} x
task in {task1_sweep, task2_sweep} = 90 cells. Each cell's acc_A is the
transformer baseline's episode-restricted K-way top-1 recall (K=32,
chance = 1/32 = 0.03125) at that (task, M, H, seed).

The figure is two small-multiple heatmaps (task1_sweep left, task2_sweep
right), x = M (log-spaced categories), y = horizon (token count at query
time), color = mean acc_A across the 3 seeds, annotated with the actual
per-cell numeric mean. Color scale is anchored to [0, demonstration_bar]
(3x chance, pre-registered) rather than to the tiny observed range, so the
color channel honestly shows "how far below the demonstration bar" instead
of amplifying chance-level noise into a false hot/cold pattern. Chance is
marked on the colorbar.

Palette: cividis (perceptually-uniform, colorblind-safe sequential
colormap, matplotlib built-in) -- an alternative to a custom Okabe-Ito
interpolation, chosen because a true sequential need is better served by a
purpose-built perceptually-uniform map than a 2-color interpolation between
categorical OI hues. No in-figure title (captions live in the HTML).
Background matches the site (#FAF5E7).
"""
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT_DIR = HERE.parent

FANOUT_DIR = Path("/Volumes/1TB_SSD/learned-representations/experiment-runs/"
                   "2026-07-10_h2h_mstar/fanout")
MANIFEST = FANOUT_DIR / "md5_manifest.txt"

# ----------------------------------------------------------------- md5 verify
expected = {}
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        h, name = line.split(None, 1)
        expected[name.strip()] = h.strip()

cell_files = sorted(FANOUT_DIR.glob("h2h_msweep_*.json"))
print(f"found {len(cell_files)} cell files matching h2h_msweep_*.json")

verified = 0
records = []
for fp in cell_files:
    raw = fp.read_bytes()
    got = hashlib.md5(raw).hexdigest()
    want = expected.get(fp.name)
    if want is None:
        raise SystemExit(f"FATAL: {fp.name} not in md5 manifest")
    if got != want:
        raise SystemExit(f"FATAL: md5 mismatch for {fp.name}: got {got}, want {want}")
    verified += 1
    records.append(json.loads(raw))

print(f"loaded {len(records)}/90 cell files, md5-verified {verified}/{len(records)} "
      f"against {MANIFEST.name}")
assert len(records) == 90, f"expected 90 fan-out cells, found {len(records)}"

# ------------------------------------------------------------- sanity summary
accs = np.array([r["acc_A"] for r in records])
chances = set(r["chance"] for r in records)
bars = set(r["demonstration_bar"] for r in records)
assert len(chances) == 1 and len(bars) == 1, "chance/demonstration_bar should be constant"
CHANCE = chances.pop()
DEMO_BAR = bars.pop()

print(f"acc_A across all 90 raw cells: min={accs.min():.4f} "
      f"mean={accs.mean():.4f} max={accs.max():.4f}")
print(f"chance = 1/K = {CHANCE:.5f}   demonstration_bar (3x chance) = {DEMO_BAR:.5f}")
n_above_bar = int((accs > DEMO_BAR).sum())
print(f"cells with acc_A > demonstration_bar: {n_above_bar}/90")

# ------------------------------------------------------------------- reshape
TASKS = ["task1_sweep", "task2_sweep"]
M_VALS = [2, 4, 8, 16, 32]
H_KEYS = ["H2", "H4", "H8"]

# horizon_tokens per H key, read from the data itself (not hardcoded)
h_tokens = {}
for r in records:
    h_tokens[r["horizon"]] = r["horizon_tokens"]
TOKEN_LABELS = [f"{h_tokens[h]:,} tok" for h in H_KEYS]

# grid[task][h_idx][m_idx] -> list of acc_A across seeds
grid_seeds = {t: defaultdict(list) for t in TASKS}
for r in records:
    grid_seeds[r["task"]][(r["horizon"], r["M"])].append(r["acc_A"])

grid_mean = {}
for t in TASKS:
    arr = np.zeros((len(H_KEYS), len(M_VALS)))
    for i, h in enumerate(H_KEYS):
        for j, m in enumerate(M_VALS):
            vals = grid_seeds[t][(h, m)]
            assert len(vals) == 3, f"expected 3 seeds for {t} {h} M={m}, got {len(vals)}"
            arr[i, j] = float(np.mean(vals))
    grid_mean[t] = arr
    print(f"{t}: per-(H,M) seed-mean acc_A min={arr.min():.4f} "
          f"mean={arr.mean():.4f} max={arr.max():.4f}")

TASK_TITLES = {
    "task1_sweep": "task 1 (single-hop recall)",
    "task2_sweep": "task 2 (multi-hop)",
}

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), facecolor=BG,
                          gridspec_kw={"wspace": 0.12})

vmin, vmax = 0.0, DEMO_BAR
cmap = matplotlib.colormaps["cividis"]

im = None
for ax, t in zip(axes, TASKS):
    ax.set_facecolor(BG)
    arr = grid_mean[t]
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                    origin="lower")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            # pick readable text color against the cividis fill
            frac = (v - vmin) / (vmax - vmin)
            txt_color = "#FAF5E7" if frac > 0.55 else "#1a1a1a"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontfamily="monospace")

    ax.set_xticks(range(len(M_VALS)))
    ax.set_xticklabels([str(m) for m in M_VALS], fontsize=9.5)
    ax.set_yticks(range(len(H_KEYS)))
    ax.set_yticklabels(TOKEN_LABELS, fontsize=9.5)
    ax.set_xlabel("KV-cache cap M (slots, log-spaced)", fontsize=10, labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # panel label, not a figure title: distinguishes the two side-by-side
    # heatmaps (there is no other way to tell them apart). House convention
    # keeps the descriptive/headline title out of the image entirely (it
    # lives in the HTML <figcaption>) -- this is a small in-axes annotation
    # analogous to the group labels in generate_rank_law.py's fig 1, not a
    # substitute for the caption.
    ax.annotate(TASK_TITLES[t], (0.5, 1.03), xycoords="axes fraction",
                fontsize=10.5, color=TEXT, ha="center", va="bottom",
                fontfamily="DejaVu Sans", fontweight="normal")

axes[0].set_ylabel("context length at query time", fontsize=10, labelpad=8)

cbar = fig.colorbar(im, ax=axes, fraction=0.055, pad=0.03)
cbar.set_label("recall acc_A (mean over 3 seeds)", fontsize=9.5, color=TEXT)
cbar.ax.tick_params(labelsize=8.5, color=TEXT)
cbar.outline.set_edgecolor(TEXT)
# mark chance level on the colorbar
chance_frac = (CHANCE - vmin) / (vmax - vmin)
cbar.ax.axhline(chance_frac, color=TEXT, linewidth=1.1, linestyle=":")
cbar.ax.text(1.55, chance_frac, "chance\n(1/32)", transform=cbar.ax.transAxes,
             fontsize=7.6, color=MUTED, ha="left", va="center")
cbar.ax.axhline(1.0, color=TEXT, linewidth=1.1, linestyle="--")
cbar.ax.text(1.55, 1.0, "demonstration\nbar (3x chance)",
             transform=cbar.ax.transAxes, fontsize=7.6, color=MUTED,
             ha="left", va="center")

plt.savefig(OUT_DIR / "mstar_fanout_heatmap.svg", format="svg",
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("wrote mstar_fanout_heatmap.svg")
