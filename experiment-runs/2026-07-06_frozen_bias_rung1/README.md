# 2026-07-06 — Frozen-bias LM rung-1 wave (`frozen-bias-lm`)

## What

Tests whether the geo3/key-anchoring testbed's "constancy suffices"
finding (a frozen, never-trained per-token key bias stabilizes
composition — `KEY_ANCHORING_DESIGN.md` candidate (e)) transplants to a
from-scratch 14M-param DeltaNet LM via a dense per-token frozen key bias
blended into `k_raw` at training time. Rung-1 (this wave)'s full 20-cell
mandatory training matrix: 2 corpora (openr1-mix-ext, wikitext-mix-ext) x
{off (Arm1, λ=0), per-token (Arm2, λ=0.58), global-vector (Arm2′,
λ=0.58)} x 3 seeds, plus a 1-seed λ∈{0.3, 0.8} mini-sweep on
openr1-mix-ext for per-token only (18 + 2 = 20 cells). Rung-2 is PARKED
(`FROZEN_BIAS_LM_DESIGN.md` §6.2/§8.1), not part of this wave's scope.

**Headline result — FOURTH OUTCOME, "sim-training divergence" (§1.3),
DESCRIPTIVE TIER ONLY (blind-pin fired post-training, see below):**

- **PRIMARY** (Arm2 − Arm1′ post-blend `span_frac`, pinned t(2,.975) CI):
  openr1 **+0.1955** [0.0937, 0.2973], wikitext **+0.2273** [0.0926,
  0.3621] — CI excludes zero in BOTH corpora but is **POSITIVE**, the
  opposite sign from every sim-family prediction (which predicted a
  *negative*, stabilizing effect).
- **CO-PRIMARY** (pre-blend `k_raw` geometry): openr1 **+0.1097**
  [0.0491, 0.1704], wikitext **+0.1345** [0.0070, 0.2621] — same
  direction as the primary, confirming the effect is training-mediated
  (not a static-blend artifact) and ruling out the §1.3 "post-blend-only
  win" suspicious-result pattern.
- **CONTROL** (Arm2′ − Arm1″, global-vector bias, same artifact-free
  footing): openr1 **−0.3319** [−0.6362, −0.0276], wikitext **−0.2308**
  [−0.2838, −0.1777] — NEGATIVE (stabilizing), the opposite direction
  from the per-token primary. §7.1a licensing (Arm2 must be more negative
  than Arm2′) FAILS.
- Cosine diagnostic (§1.3's own instrument): trained `k_raw` vs. frozen
  anchor `B[token_id]` cosines are ~0 in every arm, before and after
  training — no evidence of key-anchor alignment as the mechanism.
- Val-loss gate PASSES both arms, both corpora (no capability cost).
- λ-mini-sweep (n=1 per cell): +0.038 / +0.219 / +0.290 at λ=0.3 / 0.58 /
  0.8 — monotonically increasing, no sign flip.

Full verdict, the §1.3 fourth-outcome classification quoted verbatim,
and the descriptive-tier caveat: `matrix-thinking/FROZEN_BIAS_LM_DESIGN.md`
verdict section (appended 2026-07-06).

## When

Trained on the Brev H100 box (`youthful-indigo-turkey`,
`/home/nvidia/chapter2/deltanet_rd`) on 2026-07-06, GPU 2
(`FROZENBIAS_GPUS=1 FROZENBIAS_GPU_OFFSET=2`, single GPU per cell,
concurrent with the key-anchoring capacity-cliff wave on GPUs 2-7 sharing
the same physical box). Chain: `frozen_bias_chain.sh` (Phase A: 18 core
cells + 2 mini-sweep-off cells run across 4 supervised chain sessions,
`frozen_bias_chain_run1-4.log`) followed by a 6x-numbered measurement
pipeline (Phase A retrofit re-evals, Phase B `BANDS_PINNED` construction,
Phase C retrofit cross-arm evals, Phase D fit + cosine diagnostic).

## Where

- **This directory** (repo-tracked, all files ≤25MB — largest is
  ~875KB, total archive ~18MB, nothing routed SSD-only):
  - `results/frozen_bias_lm/` — all 20 training-cell result JSONs
    (`frozenbias_lm_{off,per_token,global}_lam*_{openr1,wikitext}-mix-ext_dm256_ds64_L2_s*.json`),
    all 46 `retrofit_*.json` re-eval JSONs, `BANDS_PINNED-FrozenBias.json`,
    `PHASE_D_FULL_REPORT.json`, the `estimation_*`/`fitinput_*`/
    `cosine_diagnostic_sec1_3.json` intermediate JSONs, `calibration.json`
    (timing constants), `rung1_calibration_summary.json`,
    `rung1_remaining19_summary.json`, and `results/frozen_bias_lm/logs/`
    (per-cell training logs, 20 + `smoke_pretrain.log`).
  - `logs/` (top-level) — numbered pipeline logs `50`-`54` (smoke tests,
    estimation self-test, calibration, remaining-19 launch) plus the
    `6x_phaseA/B/C/D_*.log` measurement-pipeline logs and
    `frozen_bias_chain_run1-4.log` (the 4 supervised training-chain
    sessions).
  - `build_bands_pin.py`, `build_fit_inputs_and_run.py`,
    `cosine_diagnostic.py` — the 3 driver scripts the measurement agent
    left on the box, pulled as-is (not re-verified byte-identical against
    a committed repo copy — these are one-off measurement scripts, not
    part of `matrix-thinking/deltanet_rd/`'s tracked source).
  - Checkpoints STAY on the box + `/data` — NOT archived here or on the
    SSD, per task scope.
- **SSD mirror** (full superset, same tree):
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-06_frozen_bias_rung1/`

## Verification

Independently recomputed the primary delta (Arm2 − Arm1′ post-blend
`span_frac`, openr1-mix-ext) from the raw `fitinput_arm2_post_blend.json`
/ `fitinput_arm1prime_post_blend.json` arrays using the pinned formula
(`mean_delta ± t(2,.975=4.303)·s/√3`, n=3): **mean_delta=0.1955009366341799,
CI=[0.0936538066504167, 0.2973480666179431]** — matches
`PHASE_D_FULL_REPORT.json` / `estimation_primary_arm2_vs_arm1prime.json`
to full float precision.

Realized training GPU-h: summed `wall_s` across all 20
`frozenbias_lm_*.json` cell files = 18175.744 s = **5.0488 GPU-h**
(all cells single-GPU, 20,000 steps each, tight 899-914s band, no
retries/crashes — 20/20 cells present). Plus ~1.6 GPU-h retrofit/
measurement eval (46 retrofit evals + cosine diagnostic + fit, all
short forward-pass-only jobs) and ~0.25 GPU-h calibration-run prior
(rung1 single-cell calibration + smoke suite) = **≈6.90 GPU-h realized**
for the whole wave, against the program's own 135 GPU-h ceiling
(~14.2 GPU-h committed at 2x contingency, rung-1-only).

## How to reproduce

```bash
ssh youthful-indigo-turkey
cd /home/nvidia/chapter2/deltanet_rd
FROZENBIAS_GPUS=1 FROZENBIAS_GPU_OFFSET=2 ./frozen_bias_chain.sh
python3 build_bands_pin.py
python3 build_fit_inputs_and_run.py
python3 cosine_diagnostic.py
```
