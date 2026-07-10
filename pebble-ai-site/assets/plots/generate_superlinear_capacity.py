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
"""
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

OUT = ("/Users/samuellarson/Experiments/learned-representations/"
       "pebble-ai-site/assets/plots/superlinear_capacity.svg")

# d=64 measured points (K/d, h4)
d64_x = [0.25, 0.5, 0.53125, 0.59375, 0.65625, 0.71875, 0.75]
d64_y = [1.0, 0.6668904622, 0.5675934553, 0.3316029410, 0.1176680326,
         0.0434499544, 0.0215115026]

# d=80 measured points (n=5 seeds at K=48 and K=53)
d80_x = [0.25, 0.5375, 0.6, 0.6625, 0.725]
d80_y = [1.0, 0.9424812198, 0.8459554195, 0.5712632775, 0.2762998343]

# d=96 measured points — two windows, no valid fit (100% bootstrap degeneracy)
d96_x = [0.25, 0.53125, 0.59375, 0.65625, 0.71875,     # low window
         0.71875, 0.75, 0.8125, 0.875, 0.9375]          # high (unlocked) window
d96_y = [1.0, 1.0, 0.9973958532, 0.9804791411, 0.9839315613,
         0.9591542284, 0.9216127793, 0.9325754642, 0.9580930869, 1.0]

def logistic(x, L, x0, w):
    return L / (1.0 + np.exp((x - x0) / w))

xs = np.linspace(0.2, 1.0, 400)
fit64 = logistic(xs, 1.0030, 0.5454626, 0.0596773)
fit80 = logistic(xs, 0.9994, 0.6779198, 0.0478704)

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
