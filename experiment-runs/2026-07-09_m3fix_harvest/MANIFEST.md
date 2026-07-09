# 2026-07-09 M3 FIX-WAVE HARVEST (capability separation, Stage 1)

The §1.33-registered M3 fix wave (D-AMB ambient-identity tax removed):
30 cells n=1 — variant A `zero_pad` (5 groups × [unconstrained anchor +
k∈{d_min−1, d_min, d_min+1}]) + variant B `tax_adjusted` (5 × 2, raw k =
effective+2). Build b07d2b6 (§1.34), audit CLEARED (§1.35, C1 metric pin),
verdict recorded in `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` §1.36.

**VERDICT: CAUSAL-CONFIRM** — razor step at k=d_min in 4/5 groups incl.
both marquee members (S4, A5); k=d_min−1 xrec90 = 0.000 at all 5 groups;
S3 marginality trigger fired (0.450 vs 0.495) → seed extension routed.

## Contents

- `zero_pad__*.json`, `tax_adjusted__*.json` — 30 per-cell results
  (decisional metrics: `crosscheck_recovered_frac_90`,
  `crosscheck_mean_cos` per §1.35 C1; scale-only primary persisted as
  disclosed-diagnostic)
- `M3FIX_STAGE_DONE` — completion sentinel
- `m3fix_wave.log` — full supervisor/wave log (17:09:10→18:34:41 UTC,
  GPU 0, exit 0, zero tracebacks)
- `box_smoke_m3fix.log` — pre-launch 13-section box smoke
- `analyze_m3fix_harvest.py` — the exact analysis script (A3
  config-match vs an independent-literal manifest + the C1 table)
- `harvest_analysis_output.txt` — its output as run for §1.36
- `md5_manifest_local.txt` / `md5_manifest_box.txt` — 33/33 match

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_m3fix_harvest/`

Realized cost: 1.4235 GPU-h vs 1.3324 priced (+6.8%, eval overhead).
