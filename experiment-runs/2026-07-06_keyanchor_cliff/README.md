# 2026-07-06 — Key-anchoring capacity-cliff localization wave (`keyanchor-cliff`)

## What

Localizes the K/d_state capacity cliff for candidate (d) (learned-λ key
anchoring) between the two previously-archived endpoints K=32 (K/d=0.50)
and K=48 (K/d=0.75), plus the K=16 saturation anchor (K/d=0.25). Four new
K's — **34, 38, 42, 46** (K/d = 0.53125, 0.59375, 0.65625, 0.71875) —
were run at 3 seeds each (12 mandatory cells, candidate (d) only; the
bare-geo3 reference arm was cut from this wave's scope, disclosed
tradeoff, see `KEY_ANCHORING_DESIGN.md` §12.2 item 2). A 6-point
`h4(x) = L / (1 + exp((x - x0)/w))` sigmoid was then fit
(`fit_cliff_curve.py`, `scipy.optimize.curve_fit`, bounds shared with the
pre-registered power simulation) with a 4,000-trial seed-level parametric
bootstrap CI over `x0`/`w`.

**Headline result:** `x0 = 0.5455` (95% bootstrap CI [0.5385, 0.5513],
width 0.0127), `w = 0.0597` (CI [0.0557, 0.0642]), `L = 1.003`,
`RSS = 0.00135`, 0/4000 degenerate bootstrap fits. Full verdict, curve
table, and interpretation: `matrix-thinking/KEY_ANCHORING_DESIGN.md` §12.9.

## When

Run on the Brev H100 box (`youthful-indigo-turkey`,
`/home/nvidia/chapter2/deltanet_rd`) on 2026-07-06. Stage 1 (K=38, K=42,
6 cells) launched first as a rate check; Stage 2 (K=34, K=46, 6 cells)
launched after Stage 1 cleared its dedicated 11.68 GPU-h sub-ceiling
(realized 1.6166 GPU-h, far under). Both stages ran on GPUs 2–7
(`KEYANCHOR_CLIFF_GPU_OFFSET=2`), 1 GPU per cell (`--per-gpu 1`).

## Where

- **This directory** (repo-tracked, all files ≤25MB so nothing was
  routed SSD-only per the hybrid archive policy,
  `experiment-runs/README.md`):
  - `results/deltanet_rd_exactness/wavekeyanchor-cliff/` — all 12 cell
    result JSONs (`wkeyanchor-k48_rdx_K{34,38,42,46}_armd_s*_...rev7.json`),
    `STAGE1_RATES_OK`, `ALL_DONE`, `PROGRESS.txt`, and the wave's own
    `logs/` subdirectory (per-cell training logs + `smoke.log`).
  - `fit_cliff_curve_results.json` — the fit script's output (headline
    numbers above, curve points, bootstrap CI, degenerate-fraction
    disclosure).
  - `logs/` (top-level) — the numbered pipeline logs `30`–`38`
    (power-sim, niter-check, threshold-derive, both smoke suites,
    Gate-2 construction test, Stage-1 launch, Stage-2 launch, fit run)
    plus `keyanchor_cliff_supervisor.log` (full supervisor session,
    includes the live budget-guard print: program-spent 55.83 + this
    wave's registered ceiling 23.3587 GPU-h = cumulative 79.1887 / 80
    GPU-h program ceiling).
  - `scripts/keyanchor_cliff_chain.sh`, `scripts/fit_cliff_curve.py` —
    the exact scripts run. **Byte-compared against the box's live
    copies at harvest time: zero drift** (`diff` exit 0 both files) —
    also identical to the versions already committed at
    `matrix-thinking/deltanet_rd/{keyanchor_cliff_chain.sh,fit_cliff_curve.py}`.
- **SSD mirror** (full superset, same tree):
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-06_keyanchor_cliff/`

## How to reproduce

```bash
# On the GPU box, from matrix-thinking/deltanet_rd/:
KEYANCHOR_CLIFF_GPUS=6 KEYANCHOR_CLIFF_GPU_OFFSET=2 bash keyanchor_cliff_chain.sh --stage 1
# after Stage 1 clears (writes STAGE1_RATES_OK):
KEYANCHOR_CLIFF_GPUS=6 KEYANCHOR_CLIFF_GPU_OFFSET=2 bash keyanchor_cliff_chain.sh --stage 2
# then fit the curve:
python3 fit_cliff_curve.py \
  --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-cliff \
  --n-boot 4000
```

Seeds: K34={130,131,132}, K38={230,231,232}, K42={330,331,332},
K46={430,431,432} (registered in the wave manifest,
`keyanchor_cliff_manifest()`).

## Verification performed at harvest (independent of the box's own fit)

Recomputed per-K mean `h4` directly from the 12 raw cell JSONs
(`checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]`, averaged
over each K's 3 seeds, matching `fit_cliff_curve.py`'s own
`load_k_mean_h4()` exactly) — **all 4 recomputed means matched
`fit_cliff_curve_results.json`'s `curve_points.h4` to full float
precision**:

| K | K/d | seeds' h4 | recomputed mean | curve_points.h4 (fit JSON) |
|---|---|---|---|---|
| 34 | 0.53125 | 0.5068, 0.6019, 0.5941 | 0.567593 | 0.5675934553146362 |
| 38 | 0.59375 | 0.3330, 0.3399, 0.3219 | 0.331603 | 0.3316029409567515 |
| 42 | 0.65625 | 0.1095, 0.1208, 0.1227 | 0.117668 | 0.1176680326461792 |
| 46 | 0.71875 | 0.0435, 0.0428, 0.0441 | 0.043450 | 0.04344995444019636 |

Realized GPU-h, summed from each cell JSON's own `wall_s` field (all 12
cells single-GPU, `--per-gpu 1`):

| Group | Cells | Sum wall_s | GPU-h |
|---|---|---|---|
| Stage 1 (K38+K42) | 6 | 5819.79s | 1.6166 (matches `STAGE1_RATES_OK` exactly) |
| Stage 2 (K34+K46) | 6 | 5629.37s | 1.5637 |
| **Total, 12 mandatory cells** | **12** | **11449.17s** | **3.1803** |

Calibration/smoke overhead (`sim_cliff_power.py`, niter-check,
threshold-derive, both smoke suites, Gate-2 construction test): all
CPU-only unit/fixture tests (synthetic tensors, `/tmp` sentinels), no
measurable GPU wall-clock — confirmed by inspecting each log; none
report a `wall_s`/step-timed training loop. Honest estimate: **~0 GPU-h
additional**, not merely omitted.

**Realized wave total: 3.1803 GPU-h**, against the registered
mandatory-only bracket-pessimistic 2× ceiling of **23.3587 GPU-h** (the
live budget-guard's own figure, `keyanchor_cliff_supervisor.log`) — the
wave used **13.6%** of its own worst-case ceiling; the pessimistic 2×
contention multiplier (applied because of the concurrent frozen-bias-LM
program sharing GPUs 2–7, §12.5) never materialized.
