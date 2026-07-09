# Attractor-Robustness 2×2 — n=3 escalation HARVEST

2026-07-09. Harvest of the 12-cell n=3 escalation (STATE.md campaign 3),
chained automatically behind the capability sweep on GPU 0 per
`experiment-runs/2026-07-09_capability_sweep_launch/MANIFEST.md`.
Completion marker `ESCALATION_STAGE_DONE` (box, 10:47 UTC) verified
before pull.

## Verdict (one line)

**NOT CONFIRMED at n=3 (threshold miss, direction-consistent trend).**
The n=1 gating-amplifies signal (+6.16 gram-dev, 2.75σ) shrank to a
**+4.31 mean delta (1.92σ_floor) at n=3 — below the pre-registered
2×2.244355 = 4.489 bar** (the runner's own `escalation.fire=false`
agrees, recomputed bit-exact). Direction HOLDS in all 3 paired seeds
(+6.16/+3.12/+3.65); exploratory Welch t (not pre-registered) p=0.062.
qk-norm stays **exonerated** (−0.10 = 0.05σ, mixed sign across seeds);
qk-off×gated +1.13 = 0.50σ, within noise. The n=1 caveats STAND; the
iclr-2027 angles-4/5 fold is NOT unblocked in its strong
("gating-amplifies confirmed") form — any fold must carry the n=3
trend-not-confirmed numbers.

## Numbers (same-corpus openr1-mix-ext, layer-pooled gram-deviation)

| combo | n=3 per-seed | mean ± sd | Δ vs baseline | σ_floor |
|---|---|---|---|---|
| qkTrue_gateFalse (baseline) | 19.069 / 22.737 / 18.777 | 20.194 ± 2.207 | — | — |
| qkTrue_gateTrue | 25.232 / 25.862 / 22.424 | 24.506 ± 1.830 | **+4.312** | **+1.92** |
| qkFalse_gateFalse | 18.196 / 20.873 / 21.205 | 20.091 ± 1.650 | −0.103 | −0.05 |
| qkFalse_gateTrue | 21.204 / 20.800 / 21.963 | 21.322 ± 0.590 | +1.128 | +0.50 |

Gated arms still train to lower loss in every seed (final train loss
3.515 vs 3.682 in the qk-on pair) — the "healthier on loss" pattern
persists directionally. rec@0.9 all-zero across 12 cells × h1-4
(PROBE-INVALID floor, NON-DECISIONAL as pre-registered). No failed
cells, no timeouts, zero skipped steps.

## Verify-vs-raws

- Per-seed values recomputed from the 12 raw `*_attractor_probe.json`
  files (`analyze_2x2_escalation.py` → `n3_analysis_output.txt`,
  `n3_recompute_summary.json`); all four combo means match the runner's
  `AGGREGATE.json` to <1e-6; the runner's `escalation.fire=false` and
  all three per-cell deltas match the recomputation exactly.
- Pull integrity: 63/63 files md5-match box (`md5_local.txt` vs
  `md5_box.txt`, zero mismatches). Checkpoints (644 MB, 12 × .pt) stay
  box-side; their md5s recorded in `checkpoints_md5_box.txt`.
- Budget: 8 new lm_pretrain legs sum to **1.9669 GPU-h**; full stage-2
  wall (incl. probe + rec@0.9 legs) 08:44:45→10:47 UTC ≈ **2.04 GPU-h**
  on GPU 0 — vs the 2.02 GPU-h incremental projection and 3.03 ceiling.

## Files

- `box_results/` — 12 cells × (lm.json/lm.log/attractor_probe.json/
  attractor_probe.log/rec_at_09.json) + `AGGREGATE.json` +
  `ESCALATION_STAGE_DONE` + `supervisor.log` (box copies)
- `analyze_2x2_escalation.py` → `n3_analysis_output.txt`, `n3_recompute_summary.json`
- `md5_local.txt`, `md5_box.txt`, `checkpoints_md5_box.txt`

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_attrrob_2x2_escalation_harvest/`.
