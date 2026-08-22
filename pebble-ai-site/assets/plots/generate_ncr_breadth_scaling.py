"""
Generate the breadth-curve figure for findings/ncr-breadth-scaling.html
(finding no. 19), in two forms:

  ncr_breadth_scaling_x.png   — 1200x675 social/X card
  ncr_breadth_scaling_fig.svg — the inline <svg> body used in the page's
                                fig 1 (written to this directory; the page
                                embeds the same element markup)

Both are recomputed, every run, from the archived per-cell battery JSONs:

  experiment-runs/2026-08-22_kscaling_wave0/k32_wave0_*_kscaling.json
  experiment-runs/2026-08-22_kscaling_sweep/{sweep,anchor}_*_kscaling.json
  experiment-runs/2026-08-22_kscaling_frontier/frontier_*_kscaling.json

Fields read: matched.P1b.per_hop[*].acc  (role == "ladder_top") and
             matched.P0.per_hop["h=1"].acc.
Nothing is read from a harvest summary or from prose — house rule; a
previous pass caught three coordinator transcription errors this way.

Curves
  upper — P1b (EXACT teacher-forced operator substitution) chance-corrected
          accuracy kappa = (acc - 1/K)/(1 - 1/K) at the antipodal top rung,
          frozen-recipe median, per-seed values as dots.
  lower — the same checkpoints' own SGD-learned write (P0) at h=1, median
          over all six cells at that K.
  inset — every cell's P0 kappa at h=1 against the pre-registered wall band
          top (chance + 3 binomial sd at n=256), which itself falls with K.

The curve ends at K=40 for a CONSTRUCTION reason, not a capability one:
the antipodal probe needs a hop h in [32,63] with h == K/2 (mod K) at the
matched 5-squaring profile, which requires 3K/2 <= 63; derive_ladder(44)
raises. See EXPERIMENT_LOG.md 2026-08-22 #3/#7.
"""
import glob
import json
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "experiment-runs"

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#8B2E1F"   # P1b, exact write
BLUE = "#24607F"     # P0, learned write
PANEL = "#f0e9d3"

WAVES = (
    "2026-08-22_kscaling_wave0",
    "2026-08-22_kscaling_sweep",
    "2026-08-22_kscaling_frontier",
)


def kappa(acc, K):
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


def load():
    """Return {K: {...}} recomputed from raws."""
    cells = []
    for wave in WAVES:
        for f in sorted(glob.glob(str(RUNS / wave / "*_kscaling.json"))):
            if os.path.basename(f).startswith("remeasure_"):
                continue  # independent-draw re-reads, not curve-of-record
            d = json.load(open(f))
            if "matched" not in d:
                continue
            K = d["kscaling"]["K"]
            top = next(v for v in d["matched"]["P1b"]["per_hop"].values()
                       if v.get("role") == "ladder_top")
            cells.append(dict(
                K=K,
                frozen=bool(d["freeze_entity_adapter"]),
                seed=d["ckpt_seed"],
                k_top=kappa(top["acc"], K),
                h_top=top["h"],
                n_squarings=top["n_squarings"],
                k_p0h1=kappa(d["matched"]["P0"]["per_hop"]["h=1"]["acc"], K),
                band_top=d["wall_band"][1],
            ))

    out = {}
    for K in sorted({c["K"] for c in cells}):
        rs = [c for c in cells if c["K"] == K]
        assert len(rs) == 6, (K, len(rs))
        assert len({c["n_squarings"] for c in rs}) == 1
        assert rs[0]["n_squarings"] == 5, "matched squaring profile broken"
        fro = sorted((c["seed"], c["k_top"]) for c in rs if c["frozen"])
        out[K] = dict(
            h_top=rs[0]["h_top"],
            frozen_seeds=[v for _, v in fro],
            frozen_median=st.median([v for _, v in fro]),
            p0_all=[c["k_p0h1"] for c in rs],
            p0_median=st.median([c["k_p0h1"] for c in rs]),
            band_top_k=kappa(rs[0]["band_top"], K),
            n_above=sum(1 for c in rs if c["k_p0h1"] > kappa(c["band_top"], K)),
            k_top_min=min(c["k_top"] for c in rs),
        )
    return out


DATA = load()
KS = sorted(DATA)
KMIN_ALL = min(DATA[K]["k_top_min"] for K in KS)
NCELLS = 6 * len(KS)

# ───────────────────────── PNG (1200x675 X card) ─────────────────────────

fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=BG)
ax = fig.add_axes([0.115, 0.205, 0.80, 0.665], facecolor=BG)
plt.rcParams["font.family"] = "DejaVu Sans"

xs = list(range(len(KS)))
ax.set_xlim(-0.35, len(KS) - 0.65)
ax.set_ylim(-0.04, 1.05)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(TEXT)
ax.spines["bottom"].set_visible(False)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([f"{v:.1f}" for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)],
                   fontfamily="DejaVu Sans Mono", fontsize=10, color=MUTED)
ax.grid(axis="y", color="#d9cfb4", lw=0.8)
ax.set_axisbelow(True)
ax.set_xticks([])
ax.set_ylabel("chance-corrected accuracy  κ = (acc − 1/K)/(1 − 1/K)",
              fontsize=10.5, color=TEXT)

ax.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (2, 4)))
ax.text(len(KS) - 0.72, 0.012, "chance (κ = 0)", fontsize=9.5, color=MUTED,
        ha="right", fontfamily="DejaVu Sans Mono")

# upper curve: P1b exact write
top_med = [DATA[K]["frozen_median"] for K in KS]
for x, K in zip(xs, KS):
    ax.plot([x] * 3, DATA[K]["frozen_seeds"], "o", ms=4.5, color=ACCENT,
            alpha=0.40, zorder=3, mew=0)
ax.plot(xs, top_med, "-o", color=ACCENT, lw=2.4, ms=6, zorder=4)
ax.text(0.02, 1.028,
        f"P1b exact write — κ at the antipodal top rung "
        f"(min {KMIN_ALL:.4f} over all {NCELLS} cells)",
        fontsize=11.5, color=ACCENT, fontweight="bold")

# lower curve: P0 learned write at h=1
p0_med = [DATA[K]["p0_median"] for K in KS]
ax.plot(xs, p0_med, "-o", color=BLUE, lw=2.2, ms=5.5, zorder=4)
ax.text(0.02, 0.115, "P0 own learned write — κ at h=1, the shallowest train hop",
        fontsize=11.5, color=BLUE, fontweight="bold")

# the separation bracket
xb = len(KS) - 0.72
ax.plot([xb, xb], [top_med[-1], 0.0], color=TEXT, lw=1.1, clip_on=False)
for yv in (top_med[-1], 0.0):
    ax.plot([xb - 0.045, xb + 0.045], [yv, yv], color=TEXT, lw=1.1, clip_on=False)
ax.text(xb + 0.12, top_med[-1] / 2, "the separation", fontsize=10.5,
        color=TEXT, rotation=-90, va="center", ha="center")

# x labels
for x, K in zip(xs, KS):
    ax.text(x, -0.075, f"K={K}", ha="center", fontsize=11,
            fontfamily="DejaVu Sans Mono", color=TEXT, transform=ax.transData)
    ax.text(x, -0.115, f"d={K + 1}", ha="center", fontsize=9.5,
            fontfamily="DejaVu Sans Mono", color=MUTED, transform=ax.transData)
anchor_x = KS.index(24)
ax.plot([anchor_x], [-0.026], marker="^", ms=7, color=ACCENT, clip_on=False)

fig.text(0.5, 0.108,
         "binding breadth — the pair (K, d = K+1); 2 recipes × 3 seeds at every point",
         ha="center", fontsize=11, color=TEXT)
fig.text(0.5, 0.070,
         "K=24 = anchor: 6 checkpoints of the 58-cell record, re-scored on the derived ladder",
         ha="center", fontsize=9.5, color=ACCENT)
fig.text(0.5, 0.032,
         "curve ends at K=40 by CONSTRUCTION — K=44's antipodal probe needs 3K/2 ≤ 63 in the "
         "matched squaring band: a design limit, not a measured capability limit",
         ha="center", fontsize=9.5, color=ACCENT)
fig.text(0.028, 0.955,
         "Exact composition holds at ceiling from K=12 to K=40 — the learned-write wall widens",
         fontsize=16, fontweight="bold", color=TEXT)
fig.text(0.985, 0.908, "pebbleml.com/findings/ncr-breadth-scaling.html",
         ha="right", va="center", fontsize=9, color=MUTED)

# ── inset: the bottom of the kappa axis ──
ins = fig.add_axes([0.295, 0.375, 0.475, 0.375], facecolor=PANEL)
for s in ("top", "right", "bottom", "left"):
    ins.spines[s].set_color(TEXT)
    ins.spines[s].set_lw(1.0)
ins.set_xlim(-0.45, len(KS) - 0.55)
ins.set_ylim(-0.028, 0.098)
ins.set_yticks([0.0, 0.04, 0.08])
ins.set_yticklabels(["0.00", "0.04", "0.08"], fontsize=9,
                    fontfamily="DejaVu Sans Mono", color=MUTED)
ins.set_xticks(xs)
ins.set_xticklabels([str(K) for K in KS], fontsize=9,
                    fontfamily="DejaVu Sans Mono", color=MUTED)
ins.tick_params(length=2)
ins.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (1, 3)))
ins.text(-0.35, 0.088,
         "ZOOM — the bottom of the κ axis: the learned write against its own chance band",
         fontsize=9.5, color=TEXT, fontweight="bold", va="top")
ins.plot(xs, [DATA[K]["band_top_k"] for K in KS], color=TEXT, lw=1.3,
         ls=(0, (5, 3)))
ins.text(xs[3] + 0.1, DATA[KS[3]]["band_top_k"] + 0.006,
         "wall band top (chance + 3 sd)", fontsize=9, color=TEXT,
         fontfamily="DejaVu Sans Mono")
for x, K in zip(xs, KS):
    ins.plot([x] * 6, DATA[K]["p0_all"], "o", ms=3.6, color=BLUE, alpha=0.5, mew=0)
ins.plot(xs, p0_med, "-o", color=BLUE, lw=2.0, ms=5)
ins.text(-0.35, -0.024, f"K=12: {DATA[12]['n_above']}/6 cells above the band",
         fontsize=9, color=BLUE, fontweight="bold")
ins.text(len(KS) - 0.6, -0.024, "K≥16: every cell inside it", fontsize=9,
         color=MUTED, ha="right")

out_png = HERE / "ncr_breadth_scaling_x.png"
fig.savefig(out_png, facecolor=BG)
plt.close(fig)
print(f"wrote {out_png}")

# ───────────────────── SVG fragment for the page figure ─────────────────────
# Same geometry the page has used since the K=12→32 version: kappa axis maps
# 1.0 -> y=40 and 0.0 -> y=400; x spans 110..660.

X0, X1 = 110.0, 660.0
NP = len(KS)
XPOS = [X0 + i * (X1 - X0) / (NP - 1) for i in range(NP)]


def Y(k):
    return round(400.0 - 360.0 * k, 1)


IX0, IX1 = 288.0, 618.0
IXPOS = [IX0 + i * (IX1 - IX0) / (NP - 1) for i in range(NP)]


def IY(k):
    return round(263.5 - 1165.0 * k, 1)


L = []
a = L.append
a('<g stroke="#c9bfa4" stroke-width="1">')
for y in (328.0, 256.0, 184.0, 112.0, 40.0):
    a(f'<line x1="110.0" y1="{y}" x2="660.0" y2="{y}"/>')
a('</g>')
a('<line x1="110.0" y1="34.0" x2="110.0" y2="418.0" stroke="#1a1a1a" stroke-width="1.4"/>')
a('<line x1="110.0" y1="418.0" x2="660.0" y2="418.0" stroke="#1a1a1a" stroke-width="1.4"/>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="11.0" fill="#5a5a5a" text-anchor="end">')
for lab, y in (("1.0", 44.0), ("0.8", 116.0), ("0.6", 188.0), ("0.4", 260.0),
               ("0.2", 332.0), ("0.0", 404.0)):
    a(f'<text x="100.0" y="{y}">{lab}</text>')
a('</g>')
a('<text x="34.0" y="220.0" font-family="\'Space Grotesk\', sans-serif" font-size="12.0" '
  'fill="#1a1a1a" text-anchor="middle" transform="rotate(-90 34.0 220.0)">'
  'chance-corrected accuracy  κ = (acc − 1/K)/(1 − 1/K)</text>')
a('<line x1="110.0" y1="400.0" x2="660.0" y2="400.0" stroke="#5a5a5a" stroke-width="1.2" stroke-dasharray="2.0 4.0"/>')
a('<text x="656.0" y="413.0" font-family="\'JetBrains Mono\', monospace" font-size="10.5" '
  'fill="#5a5a5a" text-anchor="end">chance (κ = 0)</text>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="11.0" fill="#1a1a1a" text-anchor="middle">')
for x, K in zip(XPOS, KS):
    a(f'<text x="{x:.1f}" y="438.0">K={K}</text>')
a('</g>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="#5a5a5a" text-anchor="middle">')
for x, K in zip(XPOS, KS):
    a(f'<text x="{x:.1f}" y="452.0">d={K + 1}</text>')
a('</g>')
a('<text x="385.0" y="474.0" font-family="\'Space Grotesk\', sans-serif" font-size="11.5" '
  'fill="#1a1a1a" text-anchor="middle">binding breadth — the pair (K, d = K+1); '
  '2 recipes × 3 seeds at every point</text>')
# per-seed frozen dots
a('<g fill="#8B2E1F" opacity="0.45">')
for x, K in zip(XPOS, KS):
    for v in DATA[K]["frozen_seeds"]:
        a(f'<circle cx="{x:.1f}" cy="{Y(v)}" r="2.6"/>')
a('</g>')
a('<polyline points="' + " ".join(f'{x:.1f},{Y(DATA[K]["frozen_median"])}'
                                  for x, K in zip(XPOS, KS)) +
  '" fill="none" stroke="#8B2E1F" stroke-width="2.4"/>')
a('<g fill="#8B2E1F">')
for x, K in zip(XPOS, KS):
    a(f'<circle cx="{x:.1f}" cy="{Y(DATA[K]["frozen_median"])}" r="3.6"/>')
a('</g>')
a(f'<text x="116.0" y="28.0" font-family="\'Space Grotesk\', sans-serif" font-size="12.0" '
  f'fill="#8B2E1F" font-weight="700">P1b exact write — κ at the antipodal top rung '
  f'(min {KMIN_ALL:.4f} over all {NCELLS} cells)</text>')
a('<polyline points="' + " ".join(f'{x:.1f},{Y(DATA[K]["p0_median"])}'
                                  for x, K in zip(XPOS, KS)) +
  '" fill="none" stroke="#24607F" stroke-width="2.2"/>')
a('<g fill="#24607F">')
for x, K in zip(XPOS, KS):
    a(f'<circle cx="{x:.1f}" cy="{Y(DATA[K]["p0_median"])}" r="3.4"/>')
a('</g>')
a('<text x="118.0" y="366.0" font-family="\'Space Grotesk\', sans-serif" font-size="12.0" '
  'fill="#24607F" font-weight="700">P0 own learned write — κ at h=1, the shallowest train hop</text>')
ytop = Y(DATA[KS[-1]]["frozen_median"])
a(f'<line x1="682.0" y1="{ytop}" x2="682.0" y2="400.0" stroke="#1a1a1a" stroke-width="1.1"/>')
a(f'<line x1="678.0" y1="{ytop}" x2="686.0" y2="{ytop}" stroke="#1a1a1a" stroke-width="1.1"/>')
a('<line x1="678.0" y1="400.0" x2="686.0" y2="400.0" stroke="#1a1a1a" stroke-width="1.1"/>')
a('<text x="696.0" y="220.0" font-family="\'Space Grotesk\', sans-serif" font-size="11.0" '
  'fill="#1a1a1a" text-anchor="middle" transform="rotate(-90 696.0 220.0)">the separation</text>')
ax24 = XPOS[KS.index(24)]
a(f'<polygon points="{ax24:.1f},420.0 {ax24 - 5:.1f},428.0 {ax24 + 5:.1f},428.0" fill="#8B2E1F"/>')
a('<text x="385.0" y="490.0" font-family="\'Space Grotesk\', sans-serif" font-size="10.5" '
  'fill="#8B2E1F" text-anchor="middle">K=24 = anchor: 6 checkpoints of the 58-cell record, '
  're-scored on the derived ladder &nbsp;·&nbsp; curve ends at K=40 by construction (3K/2 ≤ 63)</text>')
# inset
a('<rect x="215.0" y="124.0" width="430.0" height="188.0" fill="#f0e9d3" stroke="#1a1a1a" stroke-width="1.0"/>')
a('<text x="227.0" y="142.0" font-family="\'Space Grotesk\', sans-serif" font-size="10.5" '
  'fill="#1a1a1a" font-weight="700">ZOOM — the bottom of the κ axis: the learned write '
  'against its own chance band</text>')
a(f'<line x1="{IX0 - 10:.1f}" y1="{IY(0.0)}" x2="{IX1 + 12:.1f}" y2="{IY(0.0)}" '
  'stroke="#5a5a5a" stroke-width="1.0" stroke-dasharray="2.0 4.0"/>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="#5a5a5a" text-anchor="end">')
for lab, k in (("0.08", 0.08), ("0.04", 0.04), ("0.00", 0.0), ("-0.02", -0.02)):
    a(f'<text x="274.0" y="{IY(k) + 3.3:.1f}">{lab}</text>')
a('</g>')
a('<polyline points="' + " ".join(f'{x:.1f},{IY(DATA[K]["band_top_k"])}'
                                  for x, K in zip(IXPOS, KS)) +
  '" fill="none" stroke="#1a1a1a" stroke-width="1.3" stroke-dasharray="5.0 3.0"/>')
a(f'<text x="{IXPOS[3] + 8:.1f}" y="{IY(DATA[KS[3]]["band_top_k"]) - 7:.1f}" '
  'font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="#1a1a1a">'
  'wall band top (chance + 3 sd)</text>')
a('<g fill="#24607F" opacity="0.5">')
for x, K in zip(IXPOS, KS):
    for v in DATA[K]["p0_all"]:
        a(f'<circle cx="{x:.1f}" cy="{IY(v)}" r="2.5"/>')
a('</g>')
a('<polyline points="' + " ".join(f'{x:.1f},{IY(DATA[K]["p0_median"])}'
                                  for x, K in zip(IXPOS, KS)) +
  '" fill="none" stroke="#24607F" stroke-width="2.0"/>')
a('<g fill="#24607F">')
for x, K in zip(IXPOS, KS):
    a(f'<circle cx="{x:.1f}" cy="{IY(DATA[K]["p0_median"])}" r="3.2"/>')
a('</g>')
a(f'<text x="{IX0:.1f}" y="286.2" font-family="\'Space Grotesk\', sans-serif" font-size="9.5" '
  f'fill="#24607F" font-weight="700">K=12: {DATA[12]["n_above"]}/6 cells above the band</text>')
a(f'<text x="{IX1 + 12:.1f}" y="286.2" font-family="\'Space Grotesk\', sans-serif" font-size="9.5" '
  'fill="#5a5a5a" text-anchor="end">K≥16: every cell inside it</text>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="#5a5a5a" text-anchor="middle">')
for x, K in zip(IXPOS, KS):
    a(f'<text x="{x:.1f}" y="304.0">{K}</text>')
a('</g>')

out_svg = HERE / "ncr_breadth_scaling_fig.svg"
out_svg.write_text("\n".join(L) + "\n")
print(f"wrote {out_svg}  ({len(L)} elements)")

print("\nrecomputed points (kappa):")
for K in KS:
    d = DATA[K]
    print(f"  K={K:3d} h_top={d['h_top']:3d}  P1b frozen median {d['frozen_median']:.4f} "
          f"seeds {['%.4f' % v for v in d['frozen_seeds']]}  |  P0 h=1 median {d['p0_median']:+.4f} "
          f"band_top {d['band_top_k']:.4f}  above {d['n_above']}/6")
print(f"  min top-rung kappa over all {NCELLS} cells: {KMIN_ALL:.4f}")
