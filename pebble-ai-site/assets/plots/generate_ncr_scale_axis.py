"""
Generate the scale-axis figure for findings/ncr-scale-axis.html (finding
no. 20), in two forms:

  ncr_scale_axis_x.png    — 1200x675 social/X card
  ncr_scale_axis_fig.svg  — the inline <svg> body used in the page's fig 1

Both are recomputed, every run, from the archived per-cell JSONs — never
from a harvest summary, never from prose. House rule: the two previous
passes each caught coordinator transcription errors this way.

  392M battery  experiment-runs/2026-08-22_scaleaxis_sweep/
                  {calib,sweep}_scaleaxis392m_*_kscaling.json      (24 cells)
  98M  battery  experiment-runs/2026-08-22_kscaling_sweep/         (K=16, K=24 anchor)
                experiment-runs/2026-08-22_kscaling_wave0/         (K=32)
                experiment-runs/2026-08-22_kscaling_frontier/      (K=40)
  392M depth    experiment-runs/2026-08-22_scaleaxis_stagec/
                  depthext6_392m_*_depthext.json                   (24 cells)
  98M  depth    experiment-runs/2026-08-22_scaleaxis_stagec/ref98m_depth/
                  depthext6_*_depthext.json                        (24 cells)

Fields read: matched.P1b.per_hop[*].acc, with role == "ladder_top" for the
breadth battery and n_squarings == 11 for the depth ladder.

  kappa = (acc - 1/K)/(1 - 1/K)        chance-corrected, design line 117

LEFT PANEL  — kappa at h_top (5 squarings, the antipodal top rung) against
              the pair (K, d=K+1), at both scales, both recipes. The frozen
              arms overlap at ceiling at both scales; the trainable arm
              falls away at 392M from K=32 on. That gap is the moat.
RIGHT PANEL — the within-scale freeze-ordering statistic T_W (frozen beats
              trainable, 4 strata x 9 cross-condition pairs, ties 1/2) at
              both readouts and both scales. The 31.5 line is §5.3.1's
              ORDERING-CONFIRMED threshold (strictly above the 30.5 the 98M
              reference reads); the raw 4-strata exact p<0.01 bar is 30.

The curve covers K=16..40 only; K=44's antipodal probe is
construction-impossible in the matched squaring band (needs 3K/2 <= 63,
derive_ladder(44) raises). See EXPERIMENT_LOG.md 2026-08-22 #3, #20, #21.
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
ACCENT = "#8B2E1F"   # frozen adapter (the shipped recipe)
BLUE = "#24607F"     # trainable adapter
PANEL = "#f0e9d3"

KS = [16, 24, 32, 40]


def kappa(acc, K):
    return (acc - 1.0 / K) / (1.0 - 1.0 / K)


def load_battery(patterns, label):
    """{(K, frozen, seed): kappa at the antipodal top rung}."""
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(str(pat))):
            if os.path.basename(p).startswith("remeasure_"):
                continue
            r = json.load(open(p))
            K = r.get("K") if r.get("K") is not None else r["kscaling"]["K"]
            if K not in KS or int(r.get("base_seed", 90210)) != 90210:
                continue
            assert r["self_check"] == "PASS", p
            assert r["ckpt_step"] == 20000, p
            assert r["kscaling"]["d_ncr"] == K + 1, p
            top = [e for e in r["matched"]["P1b"]["per_hop"].values()
                   if e["role"] == "ladder_top"]
            assert len(top) == 1, p
            assert top[0]["n_squarings"] == 5, p
            key = (K, bool(r["freeze_entity_adapter"]), int(r["ckpt_seed"]))
            assert key not in out, f"duplicate {key} at {p}"
            out[key] = kappa(top[0]["acc"], K)
    assert len(out) == 24, f"{label}: expected 24 battery cells, got {len(out)}"
    return out


def load_depth11(pattern, label):
    """{(K, frozen, seed): kappa at 11 squarings on the fixed-residue ladder}."""
    out = {}
    for p in sorted(glob.glob(str(pattern))):
        r = json.load(open(p))
        K = r["K"]
        if K not in KS or int(r.get("base_seed", 90210)) != 90210:
            continue
        assert r["self_check"] == "PASS", p
        assert r["ckpt_step"] == 20000, p
        assert r["squaring_profile"] == [5, 7, 9, 11, 13, 15], p
        e = [x for x in r["matched"]["P1b"]["per_hop"].values()
             if x["n_squarings"] == 11]
        assert len(e) == 1, p
        key = (K, bool(r["freeze_entity_adapter"]), int(r["ckpt_seed"]))
        out[key] = kappa(e[0]["acc"], K)
    assert len(out) == 24, f"{label}: expected 24 depth cells, got {len(out)}"
    return out


B392 = load_battery([RUNS / "2026-08-22_scaleaxis_sweep" / "*scaleaxis392m_*_kscaling.json"], "392M")
B98 = load_battery([RUNS / "2026-08-22_kscaling_sweep" / "sweep_kscaling_K16_*_kscaling.json",
                    RUNS / "2026-08-22_kscaling_sweep" / "anchor_mob_g3b31_*_kscaling.json",
                    RUNS / "2026-08-22_kscaling_wave0" / "k32_wave0_*_kscaling.json",
                    RUNS / "2026-08-22_kscaling_frontier" / "frontier_kscaling_K40_*_kscaling.json"], "98M")
D392 = load_depth11(RUNS / "2026-08-22_scaleaxis_stagec" / "depthext6_392m_*_depthext.json", "392M depth")
D98 = load_depth11(RUNS / "2026-08-22_scaleaxis_stagec" / "ref98m_depth" / "depthext6_*_depthext.json", "98M depth")


def seeds(d, K, fz):
    return [d[(K, fz, s)] for s in (0, 1, 2)]


def medians(d, fz):
    return [st.median(seeds(d, K, fz)) for K in KS]


def U(a, b):
    """cross-condition pairs where a exceeds b, ties 1/2 (9 pairs at 3v3)."""
    return sum(1.0 if x > y else (0.5 if x == y else 0.0) for x in a for y in b)


def T_W(d):
    return sum(U(seeds(d, K, True), seeds(d, K, False)) for K in KS)


TW = {("11sq", "392M"): T_W(D392), ("11sq", "98M"): T_W(D98),
      ("htop", "392M"): T_W(B392), ("htop", "98M"): T_W(B98)}

# the four verdict numbers this page leads with, asserted against the record
assert TW[("11sq", "392M")] == 36.0 and TW[("htop", "392M")] == 36.0, TW
assert TW[("11sq", "98M")] == 30.5 and TW[("htop", "98M")] == 25.0, TW

FROZ392, TRAIN392 = medians(B392, True), medians(B392, False)
FROZ98, TRAIN98 = medians(B98, True), medians(B98, False)
CAP_BAR = 0.90              # §6.1 Curve 1 capability bar, on kappa
CONFIRM_BAR = 31.5          # §5.3.1: ORDERING-CONFIRMED requires T_W > 31.5
EXACT_BAR = 30              # raw 4-strata exact two-sided p<0.01 bar

print("kappa @ h_top medians")
for lbl, row in (("frozen 392M", FROZ392), ("frozen  98M", FROZ98),
                 ("train  392M", TRAIN392), ("train   98M", TRAIN98)):
    print(f"  {lbl}: " + "  ".join(f"K={K}:{v:.4f}" for K, v in zip(KS, row)))
print("T_W:", {f"{a} {b}": v for (a, b), v in TW.items()})

# ───────────────────────── PNG (1200x675 X card) ─────────────────────────

plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=BG)

YLO, YHI = 0.74, 1.02
axL = fig.add_axes([0.068, 0.175, 0.535, 0.645], facecolor=BG)
axL.set_xlim(-0.30, len(KS) - 0.70)
axL.set_ylim(YLO, YHI)
for s in ("top", "right", "bottom"):
    axL.spines[s].set_visible(False)
axL.spines["left"].set_color(TEXT)
axL.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
axL.set_yticklabels([f"{v:.2f}" for v in (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)],
                    fontfamily="DejaVu Sans Mono", fontsize=10, color=MUTED)
axL.grid(axis="y", color="#d9cfb4", lw=0.8)
axL.set_axisbelow(True)
axL.set_xticks([])
axL.set_ylabel("κ = (acc − 1/K)/(1 − 1/K)  at the antipodal top rung",
               fontsize=10.5, color=TEXT)

xs = list(range(len(KS)))
axL.axhline(CAP_BAR, color=TEXT, lw=1.2, ls=(0, (5, 3)))
axL.text(-0.26, CAP_BAR + 0.007, "capability bar  κ ≥ 0.90",
         fontsize=9.5, color=TEXT, ha="left", fontfamily="DejaVu Sans Mono")

# 98M references — open markers, dashed
axL.plot(xs, FROZ98, "--o", color=ACCENT, lw=1.6, ms=6, mfc=BG, mew=1.6, zorder=3)
axL.plot(xs, TRAIN98, "--s", color=BLUE, lw=1.6, ms=5.5, mfc=BG, mew=1.6, zorder=3)
# 392M — filled, solid
axL.plot(xs, FROZ392, "-o", color=ACCENT, lw=2.6, ms=7, zorder=5)
for x, K in zip(xs, KS):
    axL.plot([x] * 3, seeds(B392, K, False), "s", ms=4.6, color=BLUE,
             alpha=0.42, mew=0, zorder=4)
axL.plot(xs, TRAIN392, "-s", color=BLUE, lw=2.6, ms=6.5, zorder=5)

# the moat
xg = len(KS) - 1
axL.annotate("", xy=(xg + 0.18, FROZ392[-1]), xytext=(xg + 0.18, TRAIN392[-1]),
             arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1.2))
axL.text(xg + 0.10, (FROZ392[-1] + TRAIN392[-1]) / 2,
         f"the moat at 392M\n{FROZ392[-1] - TRAIN392[-1]:.3f} κ at K=40",
         fontsize=9.5, color=TEXT, va="center", ha="right")

axL.text(0.02, 1.0125, "frozen adapter — 392M (solid) and 98M (open): both at ceiling",
         fontsize=10.5, color=ACCENT, fontweight="bold")
axL.text(0.02, 0.7575, "trainable adapter — 98M at ceiling (open), 392M falling from K=32 (solid)",
         fontsize=10.5, color=BLUE, fontweight="bold")

for x, K in zip(xs, KS):
    axL.text(x, YLO - 0.013, f"K={K}", ha="center", fontsize=10.5,
             fontfamily="DejaVu Sans Mono", color=TEXT)
    axL.text(x, YLO - 0.028, f"d={K + 1}", ha="center", fontsize=9,
             fontfamily="DejaVu Sans Mono", color=MUTED)
fig.text(0.335, 0.082, "binding breadth — the pair (K, d = K+1); 3 seeds per arm per scale",
         ha="center", fontsize=10.5, color=TEXT)
fig.text(0.335, 0.042,
         "P1b = EXACT teacher-forced operator substitution throughout — a read-path capability, not a learned write",
         ha="center", fontsize=9.5, color=ACCENT)

# ── right panel: the ordering statistic ──
axR = fig.add_axes([0.685, 0.175, 0.285, 0.645], facecolor=BG)
axR.set_xlim(-0.62, 3.62)
axR.set_ylim(0, 38)
for s in ("top", "right", "bottom"):
    axR.spines[s].set_visible(False)
axR.spines["left"].set_color(TEXT)
axR.set_yticks([0, 9, 18, 27, 36])
axR.set_yticklabels(["0", "9", "18", "27", "36"], fontfamily="DejaVu Sans Mono",
                    fontsize=10, color=MUTED)
axR.grid(axis="y", color="#d9cfb4", lw=0.8)
axR.set_axisbelow(True)
axR.set_xticks([])
axR.set_ylabel("T_W — frozen-beats-trainable pairs (of 36)", fontsize=10.5, color=TEXT)

bars = [(0.0, TW[("11sq", "98M")], "98M", False),
        (0.9, TW[("11sq", "392M")], "392M", True),
        (2.1, TW[("htop", "98M")], "98M", False),
        (3.0, TW[("htop", "392M")], "392M", True)]
for x, v, lab, filled in bars:
    axR.bar(x, v, width=0.72, color=ACCENT if filled else "#b9a98a",
            edgecolor=TEXT, lw=1.0, zorder=3)
    axR.text(x, v - 2.6, f"{v:g}", ha="center", fontsize=12,
             color=BG if filled else TEXT, fontweight="bold",
             fontfamily="DejaVu Sans Mono", zorder=4)
    axR.text(x, -1.6, lab, ha="center", fontsize=9.5, color=TEXT,
             fontfamily="DejaVu Sans Mono")
axR.axhline(CONFIRM_BAR, color=TEXT, lw=1.3, ls=(0, (5, 3)), zorder=5)
axR.text(-0.70, CONFIRM_BAR, "31.5", fontsize=9.5, color=TEXT, ha="right",
         va="center", fontfamily="DejaVu Sans Mono")
axR.text(0.45, -4.0, "11 squarings\n≈2,052 hops", ha="center", fontsize=9.5, color=MUTED)
axR.text(2.55, -4.0, "h_top\n5 squarings", ha="center", fontsize=9.5, color=MUTED)
fig.text(0.828, 0.046, "dashed: ORDERING-CONFIRMED needs T > 31.5",
         ha="center", fontsize=9.5, color=TEXT)
fig.text(0.828, 0.012, "perfect separation at 392M, both readouts",
         ha="center", fontsize=9.5, color=ACCENT)

fig.text(0.026, 0.952,
         "The capability separation survives 4× scale — and its moat widens",
         fontsize=17, fontweight="bold", color=TEXT)
fig.text(0.985, 0.905, "pebbleml.com/findings/ncr-scale-axis.html",
         ha="right", va="center", fontsize=9, color=MUTED)
fig.text(0.026, 0.905,
         "98M → 392M, four breadths ported, 24 training cells + 26 eval cells, 0 failures",
         fontsize=10, color=MUTED)

out_png = HERE / "ncr_scale_axis_x.png"
fig.savefig(out_png, facecolor=BG)
plt.close(fig)
print(f"wrote {out_png}")

# ───────────────────── SVG fragment for the page figure ─────────────────────

W, H = 840, 470
LX0, LX1 = 118.0, 418.0          # left panel x span (K=16 .. K=40)
LY0, LY1 = 44.0, 344.0           # left panel y span (kappa 1.02 .. 0.74)
RX0, RX1 = 545.0, 800.0          # right panel x span
RY0, RY1 = 60.0, 344.0           # right panel y span (T 36 .. 0)

XP = [LX0 + i * (LX1 - LX0) / (len(KS) - 1) for i in range(len(KS))]


def LY(k):
    return round(LY1 - (k - YLO) * (LY1 - LY0) / (YHI - YLO), 1)


def RY(t):
    return round(RY1 - t * (RY1 - RY0) / 36.0, 1)


def poly(pts):
    return " ".join(f"{x},{y}" for x, y in pts)


S = []
a = S.append
a(f'<svg viewBox="0 0 {W} {H}" width="100%" style="height:auto;display:block;" '
  f'role="img" aria-labelledby="fig1title fig1desc" xmlns="http://www.w3.org/2000/svg">')
a('<title id="fig1title">Exact-write capability across binding breadth at two model scales, '
  'and the freeze-ordering statistic at both scales</title>')
a('<desc id="fig1desc">Left panel: chance-corrected accuracy at the antipodal top rung, plotted '
  'against the pair K, d equals K plus one, for K equals 16, 24, 32 and 40. The frozen-adapter arm '
  'sits at ceiling, between 0.988 and 1.000, at both 98M and 392M parameters, the two curves '
  'overlapping. The trainable-adapter arm sits at ceiling at 98M but at 392M falls below the '
  'capability bar of 0.90 from K equals 32 onward, reaching 0.844 at K equals 40, with all three '
  'seeds shown as separate points. Right panel: the within-scale ordering statistic, counting how '
  'many of 36 frozen-versus-trainable pairs the frozen arm wins. At eleven squarings it reads 30.5 '
  'at 98M and 36 of 36 at 392M; at the five-squaring top rung it reads 25 at 98M and 36 of 36 at '
  '392M. A dashed line marks the pre-registered confirmation threshold of 31.5.</desc>')

# ── left panel frame ──
a('<g stroke="#c9bfa4" stroke-width="1">')
for k in (0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
    a(f'<line x1="{LX0 - 26}" y1="{LY(k)}" x2="{LX1 + 30}" y2="{LY(k)}"/>')
a('</g>')
a(f'<line x1="{LX0 - 26}" y1="{LY0 - 4}" x2="{LX0 - 26}" y2="{LY1 + 4}" stroke="{TEXT}" stroke-width="1.4"/>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="10.5" fill="#5a5a5a" text-anchor="end">')
for k in (0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
    a(f'<text x="{LX0 - 32}" y="{LY(k) + 3.6}">{k:.2f}</text>')
a('</g>')
a(f'<text x="30" y="{(LY0 + LY1) / 2}" font-family="\'Space Grotesk\', sans-serif" font-size="11.5" '
  f'fill="{TEXT}" text-anchor="middle" transform="rotate(-90 30 {(LY0 + LY1) / 2})">'
  f'κ at the antipodal top rung  (chance-corrected)</text>')

# capability bar
a(f'<line x1="{LX0 - 26}" y1="{LY(CAP_BAR)}" x2="{LX1 + 30}" y2="{LY(CAP_BAR)}" stroke="{TEXT}" '
  f'stroke-width="1.3" stroke-dasharray="5 3"/>')
a(f'<text x="{LX0 - 22}" y="{LY(CAP_BAR) - 7}" font-family="\'JetBrains Mono\', monospace" '
  f'font-size="9.5" fill="{TEXT}">capability bar  κ ≥ 0.90</text>')

# 98M references (open markers, dashed)
a(f'<polyline points="{poly([(x, LY(v)) for x, v in zip(XP, FROZ98)])}" fill="none" '
  f'stroke="{ACCENT}" stroke-width="1.6" stroke-dasharray="6 3"/>')
a(f'<g fill="{BG}" stroke="{ACCENT}" stroke-width="1.6">')
for x, v in zip(XP, FROZ98):
    a(f'<circle cx="{x}" cy="{LY(v)}" r="4.0"/>')
a('</g>')
a(f'<polyline points="{poly([(x, LY(v)) for x, v in zip(XP, TRAIN98)])}" fill="none" '
  f'stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="6 3"/>')
a(f'<g fill="{BG}" stroke="{BLUE}" stroke-width="1.6">')
for x, v in zip(XP, TRAIN98):
    a(f'<rect x="{x - 3.6}" y="{LY(v) - 3.6}" width="7.2" height="7.2"/>')
a('</g>')

# 392M per-seed trainable dots
a(f'<g fill="{BLUE}" opacity="0.42">')
for x, K in zip(XP, KS):
    for v in seeds(B392, K, False):
        a(f'<rect x="{x - 2.8}" y="{LY(v) - 2.8}" width="5.6" height="5.6"/>')
a('</g>')

# 392M curves
a(f'<polyline points="{poly([(x, LY(v)) for x, v in zip(XP, FROZ392)])}" fill="none" '
  f'stroke="{ACCENT}" stroke-width="2.6"/>')
a(f'<g fill="{ACCENT}">')
for x, v in zip(XP, FROZ392):
    a(f'<circle cx="{x}" cy="{LY(v)}" r="4.6"/>')
a('</g>')
a(f'<polyline points="{poly([(x, LY(v)) for x, v in zip(XP, TRAIN392)])}" fill="none" '
  f'stroke="{BLUE}" stroke-width="2.6"/>')
a(f'<g fill="{BLUE}">')
for x, v in zip(XP, TRAIN392):
    a(f'<rect x="{x - 4.2}" y="{LY(v) - 4.2}" width="8.4" height="8.4"/>')
a('</g>')

# the moat bracket at K=40
mx = XP[-1] + 16
a(f'<line x1="{mx}" y1="{LY(FROZ392[-1])}" x2="{mx}" y2="{LY(TRAIN392[-1])}" stroke="{TEXT}" stroke-width="1.2"/>')
for v in (FROZ392[-1], TRAIN392[-1]):
    a(f'<line x1="{mx - 4}" y1="{LY(v)}" x2="{mx + 4}" y2="{LY(v)}" stroke="{TEXT}" stroke-width="1.2"/>')
a(f'<text x="{mx - 8}" y="{(LY(FROZ392[-1]) + LY(TRAIN392[-1])) / 2 - 2}" text-anchor="end" '
  f'font-family="\'Space Grotesk\', sans-serif" font-size="10.5" fill="{TEXT}" '
  f'font-weight="700">the moat at 392M</text>')
a(f'<text x="{mx - 8}" y="{(LY(FROZ392[-1]) + LY(TRAIN392[-1])) / 2 + 12}" text-anchor="end" '
  f'font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="{TEXT}">'
  f'{FROZ392[-1] - TRAIN392[-1]:.3f} κ at K=40</text>')

# left legend + x labels
a(f'<text x="{LX0 - 26}" y="{LY0 - 26}" font-family="\'Space Grotesk\', sans-serif" font-size="11" '
  f'fill="{ACCENT}" font-weight="700">frozen adapter — 392M (filled) and 98M (open): both at ceiling</text>')
a(f'<text x="{LX0 - 26}" y="{LY0 - 10}" font-family="\'Space Grotesk\', sans-serif" font-size="11" '
  f'fill="{BLUE}" font-weight="700">trainable adapter — 98M at ceiling (open), 392M falls from K=32</text>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="11" fill="#1a1a1a" text-anchor="middle">')
for x, K in zip(XP, KS):
    a(f'<text x="{x}" y="{LY1 + 22}">K={K}</text>')
a('</g>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="#5a5a5a" text-anchor="middle">')
for x, K in zip(XP, KS):
    a(f'<text x="{x}" y="{LY1 + 36}">d={K + 1}</text>')
a('</g>')
a(f'<text x="{(LX0 + LX1) / 2}" y="{LY1 + 58}" font-family="\'Space Grotesk\', sans-serif" '
  f'font-size="11" fill="{TEXT}" text-anchor="middle">binding breadth — the pair (K, d = K+1); '
  f'3 seeds per arm per scale</text>')
a(f'<text x="{(LX0 + LX1) / 2}" y="{LY1 + 76}" font-family="\'Space Grotesk\', sans-serif" '
  f'font-size="10" fill="{ACCENT}" text-anchor="middle">P1b — EXACT teacher-forced operator '
  f'substitution, a read-path capability</text>')

# ── divider ──
a(f'<line x1="480" y1="{LY0 - 20}" x2="480" y2="{LY1 + 40}" stroke="#c9bfa4" stroke-width="1"/>')

# ── right panel ──
a('<g stroke="#c9bfa4" stroke-width="1">')
for t in (0, 9, 18, 27, 36):
    a(f'<line x1="{RX0 - 20}" y1="{RY(t)}" x2="{RX1}" y2="{RY(t)}"/>')
a('</g>')
a(f'<line x1="{RX0 - 20}" y1="{RY(36) - 6}" x2="{RX0 - 20}" y2="{RY(0)}" stroke="{TEXT}" stroke-width="1.4"/>')
a('<g font-family="\'JetBrains Mono\', monospace" font-size="10.5" fill="#5a5a5a" text-anchor="end">')
for t in (0, 9, 18, 27, 36):
    a(f'<text x="{RX0 - 26}" y="{RY(t) + 3.6}">{t}</text>')
a('</g>')
a(f'<text x="{RX0 - 46}" y="{(RY0 + RY1) / 2}" font-family="\'Space Grotesk\', sans-serif" '
  f'font-size="11.5" fill="{TEXT}" text-anchor="middle" '
  f'transform="rotate(-90 {RX0 - 46} {(RY0 + RY1) / 2})">T_W — frozen wins, of 36 pairs</text>')

BW = 46.0
GRP = [(RX0 + 8, TW[("11sq", "98M")], "98M", False),
       (RX0 + 8 + BW + 10, TW[("11sq", "392M")], "392M", True),
       (RX0 + 8 + 2 * (BW + 10) + 26, TW[("htop", "98M")], "98M", False),
       (RX0 + 8 + 3 * (BW + 10) + 26, TW[("htop", "392M")], "392M", True)]
for x, v, lab, filled in GRP:
    fill = ACCENT if filled else "#b9a98a"
    a(f'<rect x="{x}" y="{RY(v)}" width="{BW}" height="{RY(0) - RY(v):.1f}" fill="{fill}" '
      f'stroke="{TEXT}" stroke-width="1"/>')
    a(f'<text x="{x + BW / 2}" y="{RY(v) + 20}" font-family="\'JetBrains Mono\', monospace" '
      f'font-size="12" fill="{BG if filled else TEXT}" font-weight="700" '
      f'text-anchor="middle">{v:g}</text>')
    a(f'<text x="{x + BW / 2}" y="{RY(0) + 16}" font-family="\'JetBrains Mono\', monospace" '
      f'font-size="9.5" fill="{TEXT}" text-anchor="middle">{lab}</text>')
a(f'<line x1="{RX0 - 20}" y1="{RY(CONFIRM_BAR)}" x2="{RX1}" y2="{RY(CONFIRM_BAR)}" stroke="{TEXT}" '
  f'stroke-width="1.3" stroke-dasharray="5 3"/>')
a(f'<text x="{(GRP[1][0] + BW + GRP[2][0]) / 2}" y="{RY(CONFIRM_BAR) - 5}" '
  f'font-family="\'JetBrains Mono\', monospace" font-size="9.5" fill="{TEXT}" '
  f'text-anchor="middle">31.5</text>')
a(f'<text x="{(GRP[0][0] + GRP[1][0] + BW) / 2}" y="{RY(0) + 34}" '
  f'font-family="\'Space Grotesk\', sans-serif" font-size="10" fill="{MUTED}" text-anchor="middle">'
  f'11 squarings (≈2,052 hops)</text>')
a(f'<text x="{(GRP[2][0] + GRP[3][0] + BW) / 2}" y="{RY(0) + 34}" '
  f'font-family="\'Space Grotesk\', sans-serif" font-size="10" fill="{MUTED}" text-anchor="middle">'
  f'h_top (5 squarings)</text>')
a(f'<text x="{(RX0 + RX1) / 2}" y="{RY(0) + 58}" font-family="\'Space Grotesk\', sans-serif" '
  f'font-size="10" fill="{TEXT}" text-anchor="middle">dashed: ORDERING-CONFIRMED needs T &gt; 31.5</text>')
a(f'<text x="{(RX0 + RX1) / 2}" y="{RY(0) + 76}" font-family="\'Space Grotesk\', sans-serif" '
  f'font-size="10" fill="{ACCENT}" text-anchor="middle">perfect separation at 392M — exact p = 1.25×10⁻⁵, LOSO 4/4</text>')
a('</svg>')

out_svg = HERE / "ncr_scale_axis_fig.svg"
out_svg.write_text("\n".join(S) + "\n")
print(f"wrote {out_svg}")
