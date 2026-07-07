# 2026-07-07/08 — Key-anchoring capacity-cliff, x0(d) invariance wave (`keyanchor-scaling`)

## What

Tests whether the K/d_state capacity cliff located at d=64 (`x0=0.5455`,
95% CI [0.5385,0.5513], `w=0.0597`, see `experiment-runs/2026-07-06_keyanchor_cliff/`)
is INVARIANT IN RATIO TERMS as `d_state` grows from 64 to 80 to 96, while
HOLDING the same organic-coherence regime (`n_entities=107 > d_state`)
that produced the original d=64 measurement — the missing cell between
§12 (d=64, cliff) and §13 (d=128, no cliff, but a DIFFERENT, orthogonal-
table regime). Full design: `matrix-thinking/KEY_ANCHORING_SCALING_DRAFT.md`
§15; attack round + PARK/sign-off history: §15.18; harvest verdict: §15.19.

30 mandatory cells (candidate (d) only): d=80,
K∈{20(anchor),43,48,53,58}×3 seeds; d=96, K∈{24(anchor),51,57,63,69}×3
seeds. The 3 d=80/K=20 anchor cells hit a K=20-specific `H_extra` hop-
collision at config construction on first launch (`run1`); fixed with a
K-scoped `H_extra=(7,25)` override (§15.15 addendum) and re-run clean in
`run2` — every other cell's manifest/CLI/filename is byte-identical to
what it would have been without the fix.

**Headline result: WAVE VERDICT = AMBIGUOUS (mechanical, pre-registered,
§15.10)** — not a judgment call. d=96's fit is degenerate
(`degenerate_frac=98.5%`, far past the 10% pre-registered bar), because
h4 is flat near ceiling (0.98–1.0) across the ENTIRE tested K/d window at
d=96, mechanically identical in kind to §13.10's own d=128
"NO CLIFF ANYWHERE IN THE MEASURED WINDOW" result. Per §15.10 item 4's
own text, ANY d being AMBIGUOUS makes the whole wave AMBIGUOUS,
regardless of the other d's result.

**d=80 alone, however, is a clean, non-degenerate result and it REFUTES
ratio-invariance:** `x0(80) = 0.6756`, 95% CI `[0.6620, 0.6868]`,
`w=0.0521`, `degenerate_frac=0.0`. The CI excludes the pre-registered
invariance band `[0.4745, 0.6165]` entirely, on the high side (gap
0.0455) — `x0` drifted UPWARD by +0.130 in ratio terms (≈10.4 raw K) as
`d` grew from 64 to 80. This is far closer to the "absolute-slack" rival
prediction (`x0(80)≈0.636`) than to the ratio-invariant band center
(`0.5455`), and d=96's own descriptive (not statistically clean) decline
concentrates near the same rival's `x0(96)≈0.697` prediction.

One admissibility anomaly (first of its kind in the program's archive
history): `d=96, K=69, seed=1730` reads `geo3_admission.admissible=false`
(`checkpoint_fallback_seen=true`); every other admissibility field on
that cell is clean, and a sensitivity re-fit excluding it does not change
d=96's degenerate-fit outcome. Two K-groups at d=80 (K=48, K=53 — the two
groups straddling the fitted transition) trip the pre-registered
seed-contingency range trigger (§15.14, 3-seed h4 range >0.15).

Full verdict, per-cell verification table, rival comparison, and
disclosed critique of the mechanical AMBIGUOUS rule when applied to a
flat-not-noisy fit: `matrix-thinking/KEY_ANCHORING_SCALING_DRAFT.md`
§15.19.

## When

Run on the Brev H100 box (`youthful-indigo-turkey`,
`/home/nvidia/chapter2/deltanet_rd`) on 2026-07-07, under an explicit
`KEYANCHOR_SCALING_PI_SIGNOFF=1` past the §15.18 PARK gate (the wave was
parked pending REASONING-LINK Phase 1, which landed per `STATE.md`'s
2026-07-07 ~12:45 UTC snapshot). Kernel-safety gate (§15.2/§15.18
FATAL-2) re-verified at the full `T∈{128,224,448}` protocol before
launch (`results/smoke_dstate_kernel_result.json`): d=80/96 PASS at
every T, d=32 fails at T=128 exactly as historically predicted. Ran on
GPUs 2–7 (`KEYANCHOR_SCALING_GPU_OFFSET=2`), 1 GPU per cell
(`--per-gpu 1`), across 2 chain sessions
(`keyanchor_scaling_run{1,2}.log` — run 1: 25/28 attempted cells
succeeded, the 3 K=20 cells failed at config construction on the
hop-collision; run 2: fix applied, the 3 K=20 cells + any remainder
completed, `ALL_DONE` written, both `fit_cliff_curve_d{80,96}_results.json`
generated in-session). All 30 cells finished at `steps_completed=20000`,
`complete=true`, `timed_out=false`.

## Where

- **This directory** (repo-tracked, all files ≤25MB — largest raw
  4.38MB — nothing routed SSD-only per the hybrid archive policy,
  `experiment-runs/README.md`):
  - `results/deltanet_rd_exactness/wavekeyanchor-scaling/` — all 30 cell
    result JSONs (`wkeyanchor-scaling_rdx_K{20,24,43,48,51,53,57,58,63,69}
    _armd_s*_geo3n20_anchor_learned_dprobe_rev7[_hextra7_25]_d{80,96}.json`),
    `ALL_DONE`, `CALIBRATION_DONE`, `PROGRESS.txt`, and `logs/` (34
    per-cell logs — 30 successful + 3 pre-fix K=20 stub failures +
    `smoke.log`).
  - `logs/keyanchor_scaling_run{1,2}.log` — the two full chain sessions.
  - `fit_cliff_curve_d80_results.json` / `fit_cliff_curve_d96_results.json`
    — the box's own fit outputs, independently reproduced this session
    (see §15.19).
  - `REV7_THRESHOLD_PINNED_D80.json` / `_D96.json` — the two new
    d-specific significance-threshold pins.
  - `smoke_dstate_kernel_result.json` — the kernel-safety gate artifact
    (§15.2 item 1 / §15.18 FATAL-2's own re-run protocol).
  - `scripts/` — `fit_cliff_curve.py`, `sim_cliff_power.py`,
    `keyanchor_scaling_chain.sh`, `smoke_keyanchor_scaling.py` — the
    exact scripts run. **Byte-compared against the box's live copies AND
    the repo's already-committed copies at harvest time: zero drift**
    (`diff` exit 0 / sha256 match, all 4 files).
- **SSD mirror** (full superset, same tree):
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-07_keyanchor_scaling/`

## How to reproduce

```bash
# On the GPU box, from matrix-thinking/deltanet_rd/:
KEYANCHOR_SCALING_PI_SIGNOFF=1 KEYANCHOR_SCALING_GPUS=6 KEYANCHOR_SCALING_GPU_OFFSET=2 \
  bash keyanchor_scaling_chain.sh --stage calibration
KEYANCHOR_SCALING_PI_SIGNOFF=1 KEYANCHOR_SCALING_GPUS=6 KEYANCHOR_SCALING_GPU_OFFSET=2 \
  bash keyanchor_scaling_chain.sh --stage full

# Fit each d independently (neither 80 nor 96 is in ANCHORED_D_STATES=(64,),
# so no --k32-dir/--k48-dir):
python3 fit_cliff_curve.py \
  --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-scaling \
  --d-state 80 --k-grid 20 43 48 53 58 \
  --out fit_cliff_curve_d80_results.json --n-trials 4000
python3 fit_cliff_curve.py \
  --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-scaling \
  --d-state 96 --k-grid 24 51 57 63 69 \
  --out fit_cliff_curve_d96_results.json --n-trials 4000
```

Seeds (§15.15 item 5): d=80 K20={1020,1021,1022}, K43={1030,1031,1032},
K48={1130,1131,1132}, K53={1230,1231,1232}, K58={1330,1331,1332}; d=96
K24={1420,1421,1422}, K51={1430,1431,1432}, K57={1530,1531,1532},
K63={1630,1631,1632}, K69={1730,1731,1732}.

## Verification performed at harvest (independent of the box's own fit)

Recomputed every cell's `complete`/`steps_completed`/`timed_out`/seed-
table membership/`H_extra`/architecture-pin/`geo3_n_iter`/admissibility
flags/h1 guard/h4 directly from all 30 raw JSONs (never the box's own
printed summary). Re-ran `fit_cliff_curve.py` locally against the pulled
raws at both d's — reproduced the box's own committed
`fit_cliff_curve_d{80,96}_results.json` to full float precision
(`x0(80)=0.6756`, CI `[0.6620,0.6868]`, `degenerate_frac=0.0`;
`x0(96)=0.9000` bound-pinned, `degenerate_frac=0.9852`). Full per-cell
table, the one admissibility anomaly, the two seed-contingency triggers,
and the mechanical AMBIGUOUS verdict: `KEY_ANCHORING_SCALING_DRAFT.md`
§15.19.

**Realized GPU-h**, summed from each cell's own `wall_s` (all 30 cells
single-GPU, `--per-gpu 1`):

| Group | Cells | Sum wall_s | GPU-h |
|---|---|---|---|
| d=80 (K=20,43,48,53,58 × 3) | 15 | 19984.65s | 5.5513 |
| d=96 (K=24,51,57,63,69 × 3) | 15 | 22446.75s | 6.2352 |
| **All 30 mandatory cells** | **30** | **42431.40s** | **11.7865** |

**Realized wave total: 11.7865 GPU-h**, against the Tier-1 approved
ceiling `H_scaling=21 GPU-h` (mandatory-only 2× bracket: 20.956 GPU-h,
§15.12) — **56.2%** of the 2× ceiling, and **112.5%** of the 1× point
estimate (10.478 GPU-h) — the first wave in this program to land above
its own 1× point estimate (prior waves: 13.6%–87.0% of their own 1×/2×
brackets), still far under the 2× pessimistic tail. The 3 pre-fix K=20
failures cost ~0 GPU-h (failed at config construction, before any GPU
work started). Headroom remaining against the 21 GPU-h ceiling: **9.21
GPU-h** — comfortably covers the pre-registered seed-contingency
follow-up (queued, not yet run: +2 seeds at K=48/d=80 and K=53/d=80, ≈0.7
GPU-h at 2×; separately, §15.10's own data-quality follow-up for d=96).
