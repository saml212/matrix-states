#!/bin/bash
# frozen_bias_seedext_chain.sh -- REASONING_LINK_DESIGN.md sec 16.19.7.1 (Phase-2b SEED
# EXTENSION, Rev 3, DESIGN-CLEARED-FOR-BUILD after the round-4 verify) -- the Leg-A
# PRETRAINING-layer launch: 18 NEW frozen-bias LM cells (off/per_token x wikitext-mix-ext x
# seed in {3..11}) at the ORIGINAL rung-1 recipe, via the NEW additive `--wave rung1-seedext`
# manifest. FORKED from frozen_bias_chain.sh (that script stays the historical rung-1 record,
# untouched); same STAGE ORDER / contention-gate / pipefail discipline reused verbatim.
#
# NOT RUN by this build session -- build + CPU-verify only, no GPU work performed here.
#
# THE ONE STRUCTURAL DIFFERENCE from the original chain (sec 16.19.7.1, adjudicated and pinned):
# the original's post-calibration HUMAN-INSPECTION stop is REPLACED by TWO MECHANICAL gates --
# (a) the val-loss sanity-band gate [4.2426, 4.4426] (mean +- max(5*SD, 0.10) over the 3 archived
#     per_token/wikitext rung-1 terminal readings -- real numbers, not asserted), and
# (b) the timing gate (realized calibration wall_s vs the banked 908.7872 s/cell rate, projected
#     against the wave's registered 66.5 GPU-h ceiling, sec 16.19.6's calibration re-check) --
# both enforced aborts (frozen_bias_lm_sweep.py exits 6/8, propagated by set -e), unattended-
# launch compatible (CLAUDE.md's supervisor discipline cannot perform a visual inspection).
# Only after BOTH pass does the remaining-17 launch proceed -- no FROZENBIAS_LAUNCH_REMAINING
# human re-invocation in this fork.
#
# TWO contention gates on the calibration launch (sec 16.19.7.1):
#   1. the original K-anchoring Stage-1 sentinel (sec 8.2a, unchanged), and
#   2. a SECOND contention_gate() call pointed at the KEYANCHOR-SCALING sec 15.26 wave's own
#      completion sentinel (GPUs 2-7 are shared with that wave's K90 pool-margin diagnostic).
#      sec 15.26 has NOT yet registered a sentinel path -- SEC1526_SENTINEL below defaults to a
#      deliberately-nonexistent placeholder, so this chain REFUSES until either sec 15.26's build
#      registers one (export SEC1526_SENTINEL=<path>) or a human adds
#      --accept-sec1526-override to the calibration invocation (loud warning, never a default).
#      The two gates are INDEPENDENTLY wired: a present Stage-1 sentinel never satisfies the
#      second gate (negative-tested in phase2b_seedext_stage_minus1.py).
set -euo pipefail

# PI-signoff-style precondition (checked BEFORE cd so the refusal is locally negative-testable):
# FROZENBIAS_RUNG1_STEPS is REUSED UNCHANGED from the original chain -- a DIFFERENT step count
# would silently reintroduce a comparability confound between the 3 archived and 9 new per-arm
# seeds (sec 16.19.7.1). The realized rung-1 wave used 20000; set it explicitly, no placeholder.
: "${FROZENBIAS_RUNG1_STEPS:?FROZENBIAS_RUNG1_STEPS must be set (REUSED UNCHANGED from the realized rung-1 wave, 20000 -- sec 16.19.7.1, no silent placeholder)}"

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# sec 16.19.7.1's named env vars (FROZENBIAS_<PURPOSE> convention, mirrored from the original):
# GPUs 2-7, 6-way (sec 16.19.6's registered assignment); a SEPARATE out-dir from rung-1's own
# results/frozen_bias_lm (REQUIRED, not cosmetic: is_done_cell + the calibration-JSON write are
# scoped per out-dir -- a shared dir would risk colliding with rung-1's already-landed
# calibration.json); the SAME /data checkpoint tree (additive seed subdirs, the {arm}_{corpus}_
# s{seed} naming already disambiguates, sec 16.19.2's box-verified layout).
FROZENBIAS_SEEDEXT_GPUS=${FROZENBIAS_SEEDEXT_GPUS:-6}
FROZENBIAS_SEEDEXT_GPU_OFFSET=${FROZENBIAS_SEEDEXT_GPU_OFFSET:-2}
FROZENBIAS_SEEDEXT_OUT_DIR=${FROZENBIAS_SEEDEXT_OUT_DIR:-results/frozen_bias_lm_seedext}
FROZENBIAS_SEEDEXT_DATA_DIR=${FROZENBIAS_SEEDEXT_DATA_DIR:-/data/deltanet_rd_data}
FROZENBIAS_SEEDEXT_CKPT_BASE=${FROZENBIAS_SEEDEXT_CKPT_BASE:-/data/deltanet_rd_frozenbias_ckpts}
STAGE1_SENTINEL=${STAGE1_SENTINEL:-results/deltanet_rd_exactness/wavekeyanchor-cliff/STAGE1_RATES_OK}
# sec 15.26's own completion sentinel -- NOT yet registered by that wave (sec 16.19.9 item 12's
# disclosed forward dependency); the default matches frozen_bias_lm_sweep.SEC1526_SENTINEL_DEFAULT
# (deliberately nonexistent -> mechanical refusal until pointed at a real path or overridden).
SEC1526_SENTINEL=${SEC1526_SENTINEL:-results/UNREGISTERED_SEC1526_SENTINEL_SEE_REASONING_LINK_16.19.7.1}

mkdir -p "$FROZENBIAS_SEEDEXT_OUT_DIR"
mkdir -p "$FROZENBIAS_SEEDEXT_CKPT_BASE"

# Pre-launch CPU smoke gates (same three as the original chain; also re-run by _run_manifest).
$PY smoke_frozen_bias_wave_neg1.py 2>&1 | tee logs/60_seedext_smoke_frozen_bias_wave_neg1.log
$PY smoke_frozen_bias_lm.py 2>&1 | tee logs/61_seedext_smoke_frozen_bias_lm.log
$PY fit_frozenbias_estimation.py --self-test 2>&1 | tee logs/62_seedext_fit_selftest.log

# CALIBRATION CELL (per_token / wikitext-mix-ext / seed=3 / lambda=0.58 -- sec 16.19.7.1's
# registered choice, the FIRST of the 9 new seeds): launches FIRST AND ALONE. BOTH contention
# gates are checked inside the sweep's own --calibration-only path (mechanical sys.exit(4), not
# honor-system flags here).
$PY frozen_bias_lm_sweep.py --wave rung1-seedext --calibration-only \
    --rung1-steps "$FROZENBIAS_RUNG1_STEPS" \
    --out-dir "$FROZENBIAS_SEEDEXT_OUT_DIR" --data-dir "$FROZENBIAS_SEEDEXT_DATA_DIR" \
    --ckpt-base-dir "$FROZENBIAS_SEEDEXT_CKPT_BASE" \
    --gpus 1 --gpu-offset "$FROZENBIAS_SEEDEXT_GPU_OFFSET" \
    --stage1-sentinel "$STAGE1_SENTINEL" \
    --sec1526-sentinel "$SEC1526_SENTINEL" \
    2>&1 | tee logs/63_seedext_calibration.log

# MECHANICAL POST-CALIBRATION GATES (replace the original's human checkpoint, sec 16.19.7.1):
# both read the calibration cell's OWN result JSON; either failing exits nonzero and set -e
# halts the chain BEFORE the remaining 17 cells.
CALIB_JSON="$FROZENBIAS_SEEDEXT_OUT_DIR/frozenbias_lm_per_token_lam0p58_wikitext-mix-ext_dm256_ds64_L2_s3.json"
$PY frozen_bias_lm_sweep.py --seedext-band-check "$CALIB_JSON" \
    2>&1 | tee logs/64_seedext_valloss_band_gate.log
$PY frozen_bias_lm_sweep.py --seedext-timing-check "$CALIB_JSON" \
    2>&1 | tee logs/65_seedext_timing_gate.log

# REMAINING 17 CELLS -- 6-way on GPUs 2-7 (sec 16.19.6's registered assignment: 18 cells / 6 GPUs
# = 3 waves x ~908.8s/cell ~= 0.76h wall). Gated on the calibration cell's own completion
# (is_done_cell inside the sweep) + the two mechanical gates above; SAME --rung1-steps.
$PY frozen_bias_lm_sweep.py --wave rung1-seedext \
    --rung1-steps "$FROZENBIAS_RUNG1_STEPS" \
    --out-dir "$FROZENBIAS_SEEDEXT_OUT_DIR" --data-dir "$FROZENBIAS_SEEDEXT_DATA_DIR" \
    --ckpt-base-dir "$FROZENBIAS_SEEDEXT_CKPT_BASE" \
    --gpus "$FROZENBIAS_SEEDEXT_GPUS" --gpu-offset "$FROZENBIAS_SEEDEXT_GPU_OFFSET" \
    2>&1 | tee logs/66_seedext_remaining17.log

# Completion sentinel -- phase2b_seedext_chain.sh's own Leg-A completion gate reads this
# (mirrors the original chain's FROZENBIAS_RUNG1_CHAIN_DONE precedent).
touch "$FROZENBIAS_SEEDEXT_OUT_DIR/FROZENBIAS_SEEDEXT_CHAIN_DONE"
echo "FROZEN-BIAS SEEDEXT CHAIN COMPLETE: 18 Leg-A cells under $FROZENBIAS_SEEDEXT_CKPT_BASE."
