# 2026-07-06 — Key-anchoring capacity-cliff, d_state=128 universality wave (`keyanchor-dstate`)

## What

Tests whether the K/d_state capacity cliff located at d=64 (`x0 = 0.5455`,
95% CI [0.5385, 0.5513], see `experiment-runs/2026-07-06_keyanchor_cliff/`)
is a universal property of the ratio K/d, or dimension-dependent. Re-runs
the SAME K/d window (K/d = 0.53125, 0.59375, 0.65625, 0.71875) at
d_state=128, using the equivalent K's for that ratio: **K = 68, 76, 84,
92**, 3 seeds each (12 mandatory cells, candidate (d) only).

**Headline result: NO CLIFF ANYWHERE IN THE MEASURED WINDOW at d=128.**
`h4 = 1.0` at all four K values, all 3 seeds each (12/12 cells). The
sigmoid fit is degenerate BY CONSTRUCTION on flat-at-ceiling data
(bootstrap `degenerate_frac = 1.0`, CI null) — this correctly triggers
the §12.4-registered degenerate-fit disclosure rule: the fit refuses to
localize a transition that is not present in the window. The point
estimate (`x0 = 0.898`, `w = 0.011`) is extrapolation garbage from a
degenerate fit and must NOT be quoted as a located cliff.

Per §13.0's registered outcome semantics this is **CONFIRM-SHIFTED
(strong form)**: not merely a shifted `x0` — the cliff exits the window
entirely. Capacity at d=128 exceeds K/d=0.71875, i.e. at least 92 of the
107 trainable entities recover perfectly through held-out 4-hop
composition, in the same K/d band that collapsed 0.667→0.022 at d=64.

Full verdict, verification table, and the disclosed-comparison-axis
interpretation: `matrix-thinking/KEY_ANCHORING_DESIGN.md` §13.10.

## When

Run on the Brev H100 box (`youthful-indigo-turkey`,
`/home/nvidia/chapter2/deltanet_rd`) on 2026-07-06. Pre-flight (zero/low
GPU cost): `keyanchor_dstate_niter_check.py --d-state 128` (Newton-Schulz
n_iter=20 sufficiency, confirmed via convergence check, wall_s=17.0s,
CPU-side) and `rev7_threshold_derive.py --d-state 128` (pure function of
{n_entities, d_state, alpha}, no data dependency, wall_s negligible) ran
first to produce d=128-specific pins, distinct from the d=64 pin file
(smoke 8 in `smoke_keyanchor_dstate.py` verifies the two pins are
byte-DIFFERENT). Calibration cell (K=68, seed=530) ran first and gated
the decision table (`keyanchor_dstate_chain.sh --stage calibration`);
realized rate 0.6410 GPU-h/cell routed to `PROCEED_FULL` (12 cells, no
descope). Full 12-cell grid then ran on GPUs 2–7
(`KEYANCHOR_DSTATE_GPU_OFFSET=2`), 1 GPU per cell (`--per-gpu 1`), across
3 chain-launch attempts (`keyanchor_dstate_chain_run{1,2,3}.log` — run 1
and run 2 exercised the smoke suite and calibration re-entrantly without
launching the full grid; run 3 is the one that actually launched and
completed all 12 training cells and wrote `ALL_DONE`). All 12 cells
finished at `steps=20000`, `complete=true`.

## Where

- **This directory** (repo-tracked, all files ≤25MB, nothing routed
  SSD-only per the hybrid archive policy, `experiment-runs/README.md`):
  - `results/wavekeyanchor-dstate/` — all 12 cell result JSONs
    (`wkeyanchor-k48_rdx_K{68,76,84,92}_armd_s*_geo3n20_anchor_learned_dprobe_rev7_d128.json`),
    `CALIBRATION_DONE`, `ALL_DONE`, `PROGRESS.txt`, and the wave's own
    `logs/` subdirectory (12 per-cell training logs + `smoke.log`).
  - `fit_cliff_curve_d128_results.json` — the fit script's output
    (curve points, degenerate sigmoid fit, degenerate-fraction
    disclosure).
  - `REV7_THRESHOLD_PINNED_D128.json` — the d=128-specific significance
    threshold pin (Bonferroni/BH/BY derivations, `sigma_chance =
    1/sqrt(128)` exactly), distinct from the d=64 pin.
  - `keyanchor_dstate_niter_check_results.json` — the pre-flight
    Newton-Schulz n_iter=20 sufficiency check at d=128 for K∈{68,76,84,92}.
  - `logs/` (top-level) — `41_keyanchor_dstate_niter_check.log`,
    `44_smoke_keyanchor_dstate.log`, `46_wave_keyanchor_dstate_calibration.log`,
    `47_wave_keyanchor_dstate_full.log`, plus the 3 chain-launch session
    logs `keyanchor_dstate_chain_run{1,2,3}.log`.
  - `scripts/` — `keyanchor_dstate_chain.sh`, `smoke_keyanchor_dstate.py`,
    `keyanchor_dstate_niter_check.py`, `fit_cliff_curve.py`,
    `rev7_threshold_derive.py` — the exact scripts run. **Byte-compared
    against the repo's already-committed copies at
    `matrix-thinking/deltanet_rd/{same names}` at harvest time: zero
    drift** (`diff` exit 0, all 5 files).
- **SSD mirror** (full superset, same tree):
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-06_keyanchor_dstate/`

## How to reproduce

```bash
# On the GPU box, from matrix-thinking/deltanet_rd/:
python3 keyanchor_dstate_niter_check.py --d-state 128 --out keyanchor_dstate_niter_check_results.json \
  --k-grid 68 76 84 92
python3 rev7_threshold_derive.py --d-state 128 --out REV7_THRESHOLD_PINNED_D128.json
KEYANCHOR_DSTATE_GPUS=6 KEYANCHOR_DSTATE_GPU_OFFSET=2 bash keyanchor_dstate_chain.sh --stage calibration
KEYANCHOR_DSTATE_GPUS=6 KEYANCHOR_DSTATE_GPU_OFFSET=2 bash keyanchor_dstate_chain.sh --stage full
python3 fit_cliff_curve.py \
  --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-dstate \
  --d-state 128 --k-grid 68 76 84 92 \
  --out fit_cliff_curve_d128_results.json --n-boot 4000
```

Seeds: K68={530,531,532}, K76={630,631,632}, K84={730,731,732},
K92={830,831,832} (registered in the wave manifest,
`keyanchor_dstate_manifest()`).

## Verification performed at harvest (independent of the box's own fit)

Recomputed per-cell `h4` directly from all 12 raw cell JSONs
(`checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]`) — **all 12
cells, all seeds: exactly 1.0**:

| K | K/d | seed 1 | seed 2 | seed 3 | curve_points.h4 (fit JSON) |
|---|---|---|---|---|---|
| 68 | 0.53125 | 1.0 | 1.0 | 1.0 | 1.0 |
| 76 | 0.59375 | 1.0 | 1.0 | 1.0 | 1.0 |
| 84 | 0.65625 | 1.0 | 1.0 | 1.0 | 1.0 |
| 92 | 0.71875 | 1.0 | 1.0 | 1.0 | 1.0 |

**Instrument-saturation check (not everything is saturated):** other
fields in the same JSONs show real, graded variation, ruling out a
broken/ceilinged readout:
- Hop depth 21 (`M3_held_out["21"]["recovered_frac@0.9"]`) — the
  far-extrapolation hop — is NOT uniformly 1.0: values range
  0.9900–1.0 across the 12 cells (e.g. K84 seed 731: 0.9963; K84 seed
  732: 0.9900), and `recovered_frac@0.99` at h=21 drops as low as
  0.8630 (K68 seed 530). `mean_cos` at h=21 (~0.994) is visibly lower
  than at h=4 (~0.9998).
- `pred_norm_mean` scales sensibly with hop depth (h4≈3.46 → h21≈469),
  as expected from compounding.
- `effective_rank_whole_mean` tracks K almost exactly at every K
  (K68→67.82, K76→75.8, K84→83.6, K92→91.7), i.e. the state is using
  essentially its full permitted rank, not saturating against d=128.
- Training losses are non-trivial and scale gently with K (final loss
  ≈0.0030 at K68 up to ≈0.0040 at K92), not floored at machine epsilon.
- `steps_completed=20000`, `complete=true`, `timed_out=false` for all 12
  cells — no early-exit artifact.

**Training-reality check:**

| K (seed range) | wall_s per seed | final loss (step 20000) |
|---|---|---|
| 68 (530/531/532) | 2307.7 / 2269.3 / 2233.1 s | 0.0030 / 0.0030 / 0.0030 |
| 76 (630/631/632) | 2176.1 / 2143.3 / 2167.6 s | 0.0033 / 0.0034 / 0.0033 |
| 84 (730/731/732) | 2227.8 / 2199.9 / 2215.1 s | 0.0037 / 0.0037 / 0.0037 |
| 92 (830/831/832) | 2122.0 / 2143.1 / 2121.8 s | 0.0040 / 0.0040 / 0.0040 |

Wall times run ~35–39 min/cell (not the placeholder 3600s/cell used only
inside the smoke suite's synthetic decision-table unit tests — those are
explicitly fake `/tmp` fixtures, never real training data; the real
calibration read used the real K68/seed530 cell's actual `wall_s`).

**Realized GPU-h**, summed from each cell JSON's own `wall_s` field (all
12 cells single-GPU, `--per-gpu 1`):

| Group | Cells | Sum wall_s | GPU-h |
|---|---|---|---|
| All 12 mandatory cells (incl. calibration cell K68/s530) | 12 | 26326.74s | 7.3130 |

Pre-flight overhead (`keyanchor_dstate_niter_check.py`,
`rev7_threshold_derive.py`, both smoke suites): CPU-only unit/fixture
tests plus one 17.0s Newton-Schulz convergence check and one
sub-second pure-function threshold derivation — no measurable GPU
wall-clock. `fit_cliff_curve.py`'s own run: `wall_s=15.43s`, CPU-only
(sigmoid fit + 4,000-trial bootstrap over already-produced scalars).
Honest estimate: **~0 GPU-h additional**, not merely omitted.

**Realized wave total: 7.3130 GPU-h**, against the registered
calibration-derived headroom of **20.99 GPU-h** (`decision.headroom_gpuh`
in `CALIBRATION_DONE`) — the wave used **34.8%** of its own approved
headroom. `decision.realized_rate_gpuh = 0.6410` GPU-h/cell (well under
all three escalation thresholds: `proceed_full<=1.7492`,
`option_b<=2.3322`, `option_c<=3.4983`), confirming `PROCEED_FULL`
(12 cells, no descope) was the correct mechanical branch.

## Process notes (launch-seam fixes across the 3 chain-run attempts)

1. **smoke-9 path** — `keyanchor_dstate_chain_run1.log` shows smoke 9
   (`read_wall_s_only` h4-blindness check) FAILING with "no real archived
   cell JSON found to test against" — this was expected/self-healing,
   not a code bug: smoke 9 needs a real prior cell JSON to test the
   blinding helper against, and none existed yet on the first attempt.
   By run 2/run 3, a real calibration-cell JSON existed and smoke 9
   passed cleanly (`EXECUTED against a real archived cell JSON, returns
   wall_s only, no M3_held_out content anywhere in its output`).
2. **Pin sha regeneration** — `rev7_threshold_derive.py --d-state 128`
   was re-run to (re)produce `REV7_THRESHOLD_PINNED_D128.json`;
   `script_sha256 = 2e3aaa2e64b27a71fa396bcaca3898d1999d745ccc76508e395b4e783bd302db`
   recorded in the run log, confirming the derivation is a pure function
   of its own source (no silent data dependency).
2. **Per-d pin selection** — the chain script explicitly writes/reads a
   `_D128`-suffixed pin file, kept separate from the d=64
   `REV7_THRESHOLD_PINNED.json`; smoke 8 verifies the two pins are
   byte-DIFFERENT and that `sigma_chance == 1/sqrt(128)` exactly in the
   d=128 pin, while `r_min_headline_band` stays fixed at 0.35 in both
   (a cross-d invariant, not re-derived per d).
