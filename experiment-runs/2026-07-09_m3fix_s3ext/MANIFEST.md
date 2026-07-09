# 2026-07-09 M3FIX S3 SEED-PARAMETERIZATION EXTENSION (capability separation, Stage 1)

The §1.36-routed 3-seed extension of S3's variant-A/B cells (S3's
`k=d_min` decisive cell read 0.450 vs its 0.495 bar, |Δ|=0.045, inside
the pre-stated ±0.05 marginality trigger). Build `ccd7d39`
(`build_m3fix_manifest(seed)` + `--m3fix-seed`/`--m3fix-groups` on
`run_capability_sep.py`) — the prior extension attempt found
`build_m3fix_manifest()` hardcoded `seed=0` baked into the `cell_id`
f-strings with no `--seed` CLI flag; fixed and mutation-kill-proven
(reverting the fix made smoke fail on the intended assert).

Launch: seeds 1,2,3 × S3-only (18 cells, 4 variant-A + 2 variant-B per
seed), tmux `m3fix_s3ext`, GPU 0, `CAPABILITY_SEP_PI_SIGNOFF=1` citing
§1.36's routed trigger. Wave span 18:53:37→19:13:32 UTC (≈19.9 min),
supervisor exited 0.

**VERDICT: S3 CONFIRMED** — seed-mean (all 4: 0,1,2,3) of `k=d_min`'s
`crosscheck_recovered_frac_90` = 0.5625, clears the pre-registered
0.495 bar (extension-only seed-mean of 1,2,3 = 0.600, clears more
comfortably). `k=d_min−1` reads EXACTLY 0.000 in all 4 independent
seeds — the necessity leg has zero noise across the extension. Full
account: `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` §1.36a.

## Contents

- `zero_pad__S3__*__seed{1,2,3}.json`, `tax_adjusted__S3__*__seed{1,2,3}.json`
  — 18 per-cell results (decisional metrics: `crosscheck_recovered_frac_90`,
  `crosscheck_mean_cos` per §1.35's C1 pin, still in force)
- `M3FIX_S3EXT_STAGE_DONE` — completion sentinel
- `m3fix_s3ext_wave.log` — full supervisor/wave log, zero tracebacks
- `box_smoke_m3fix_s3ext.log` — pre-launch 13-section box smoke (incl.
  the two new seed-parameterization teeth sections)
- `m3fix_s3ext_supervisor.sh` — the exact launch script (self-healing,
  per-seed loop, mirrors `m3fix_supervisor.sh`'s house convention)
- `analyze_m3fix_s3ext_harvest.py` — the harvest analysis script (A3
  config-match vs an independent-literal 24-cell manifest + the
  per-seed C1 table + seed-mean verdict); reads this dir PLUS the
  original seed=0 S3 files from `../2026-07-09_m3fix_harvest/`
- `harvest_analysis_output.txt` — its output as run for §1.36a
- `md5_manifest_local.txt` / `md5_manifest_box.txt` — 21/21 match

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_m3fix_s3ext/`

Realized cost: 0.3283 GPU-h vs 0.3331 priced (−1.4%). Campaign ledger:
≈4.78 + 0.3283 = ≈5.11/30 GPU-h.
