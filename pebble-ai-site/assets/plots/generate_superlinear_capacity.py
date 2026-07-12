"""
Generate the figure for the super-linear-capacity findings page.

Held-out 4-hop recall (rec@0.9, "h4") vs capacity ratio K/d_state for a
single-layer DeltaNet fast-weight memory at d_state = 64, 80, 96, with the
logistic fits for d=64 and d=80 (d=96 has no valid fit — 100% bootstrap
degeneracy; points shown as scatter only).

Data sources (raw fit JSONs, quoted verbatim):
  d=64:  experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json
         (L=1.0030, x0=0.5454626, w=0.0596773)
  d=80:  experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/
         fit_cliff_curve_d80_refit_results.json (n=5 at K=48/53;
         L=0.9994, x0=0.6779198, w=0.0478704)
  d=96:  experiment-runs/2026-07-07_keyanchor_scaling/fit_cliff_curve_d96_results.json
         (low window) and experiment-runs/2026-07-08_c17_repro/fits/
         fit_d96_unlocked_results.json (high window). No valid sigmoid fit.

Palette: Okabe-Ito (colorblind-safe). No in-figure titles (captions live in
the HTML). Background matches the site (#FAF5E7).

RUNTIME VERIFICATION (added 2026-07, closing a provenance gap flagged by
site audit): d=64 and d=80's curve points and sigmoid fit parameters
(L, x0, w) are now loaded live from their raw fit-result JSONs and asserted
against the original transcription before plotting. d=96 has no single
fit-result file (it's assembled from two disjoint measurement windows in
two separate archive directories, both cited above); those ten points
remain a cited, not runtime-reloaded, transcription -- there is no sigmoid
fit for d=96 to verify against in the first place (100% bootstrap
degeneracy is the finding). See pebble-ai-site/assets/plots/
generate_superlinear_capacity_perseed.py for the full raw per-seed scatter
underneath all three fits, including d=96, loaded entirely at runtime.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_GREEN = "#009E73"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = str(HERE.parent / "superlinear_capacity.svg")

# ------------------------------------------------------- load + verify raws
with open(REPO / "experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json") as f:
    fit64_raw = json.load(f)
with open(REPO / "experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/fit_cliff_curve_d80_refit_results.json") as f:
    fit80_raw = json.load(f)
print("loaded d=64 and d=80 sigmoid fits + curve points from raw fit-result JSONs")

d64_x = fit64_raw["curve_points"]["x"]
d64_y = fit64_raw["curve_points"]["h4"]
sf64 = fit64_raw["sigmoid_fit"]

d80_x = fit80_raw["curve_points"]["x"]
d80_y = fit80_raw["curve_points"]["h4"]
sf80 = fit80_raw["sigmoid_fit"]

# cross-check against the original (documented) transcription before trusting the plot
assert d64_x == [0.25, 0.5, 0.75, 0.53125, 0.59375, 0.65625, 0.71875]
assert [round(v, 6) for v in d64_y] == [1.0, 0.66689, 0.021512, 0.567593, 0.331603, 0.117668, 0.04345]
assert abs(sf64["x0"] - 0.5454626254376084) < 1e-9
assert abs(sf80["x0"] - 0.6779198) < 1e-6

# sort d64/d80 points by x for a clean left-to-right scatter/line reading
_o64 = sorted(range(len(d64_x)), key=lambda i: d64_x[i])
d64_x = [d64_x[i] for i in _o64]
d64_y = [d64_y[i] for i in _o64]

# d=96 measured points — two disjoint windows, no valid fit (100% bootstrap
# degeneracy); no single raw file to reload (see docstring), cited as before.
d96_x = [0.25, 0.53125, 0.59375, 0.65625, 0.71875,     # low window
         0.71875, 0.75, 0.8125, 0.875, 0.9375]          # high (unlocked) window
d96_y = [1.0, 1.0, 0.9973958532, 0.9804791411, 0.9839315613,
         0.9591542284, 0.9216127793, 0.9325754642, 0.9580930869, 1.0]

def logistic(x, L, x0, w):
    return L / (1.0 + np.exp((x - x0) / w))

xs = np.linspace(0.2, 1.0, 400)
fit64 = logistic(xs, sf64["L"], sf64["x0"], sf64["w"])
fit80 = logistic(xs, sf80["L"], sf80["x0"], sf80["w"])

fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=BG)
ax.set_facecolor(BG)

# fits
ax.plot(xs, fit64, color=OI_BLUE, linewidth=1.6, alpha=0.85, zorder=2)
ax.plot(xs, fit80, color=OI_VERMILLION, linewidth=1.6, alpha=0.85, zorder=2)

# measured points
ax.scatter(d64_x, d64_y, color=OI_BLUE, s=42, zorder=4, marker="o",
           edgecolor=TEXT, linewidth=0.7, label="d=64 (fit x0 = 0.5455)")
ax.scatter(d80_x, d80_y, color=OI_VERMILLION, s=46, zorder=4, marker="s",
           edgecolor=TEXT, linewidth=0.7, label="d=80 (fit x0 = 0.6779)")
ax.scatter(d96_x, d96_y, color=OI_GREEN, s=48, zorder=4, marker="^",
           edgecolor=TEXT, linewidth=0.7,
           label="d=96 (no valid fit — near ceiling)")

# x0 midpoints
ax.axvline(0.5455, color=OI_BLUE, linewidth=0.9, linestyle="--", alpha=0.6,
           zorder=1)
ax.axvline(0.6779, color=OI_VERMILLION, linewidth=0.9, linestyle="--",
           alpha=0.6, zorder=1)
ax.annotate("x0 = 0.5455", (0.5455, 0.045), fontsize=8.5, color=OI_BLUE,
            ha="right", va="bottom", rotation=90, xytext=(-4, 0),
            textcoords="offset points")
ax.annotate("x0 = 0.6779", (0.6779, 0.045), fontsize=8.5, color=OI_VERMILLION,
            ha="left", va="bottom", rotation=90, xytext=(4, 0),
            textcoords="offset points")

ax.set_xlabel("K / d_state (bindings per state dimension)", fontsize=10,
              labelpad=8)
ax.set_ylabel("held-out 4-hop recall (rec@0.9)", fontsize=10, labelpad=8)
ax.set_xlim(0.2, 1.0)
ax.set_ylim(-0.03, 1.06)
ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color=TEXT)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_color(TEXT)

legend = ax.legend(loc="lower left", frameon=True, fontsize=8.5,
                   facecolor=BG, edgecolor=TEXT)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
plt.savefig(OUT, format="svg", facecolor=BG, bbox_inches="tight")
print(f"wrote {OUT}")
