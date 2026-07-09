#!/usr/bin/env bash
# Self-healing supervisor: §1.36 S3 SEED-PARAMETERIZATION EXTENSION.
# GPU 0 ONLY -- never touches GPUs 3/4 (live fixscale_392m wave), GPU 6
# (live fixscale_98m_resume wave), or any other live session.
#
# Authorization: matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md §1.36
# (commit d048a7f) -- "S3 marginality trigger FIRED -> seed extension
# routed" (S3's decisive k=d_min cell reads 0.450 vs its 0.495 bar,
# |Delta|=0.045, inside the pre-stated +/-0.05 trigger). Build:
# matrix-thinking/capability_separation/run_capability_sep.py commit
# ccd7d39 (build_m3fix_manifest(seed)/--m3fix-seed/--m3fix-groups).
# CAPABILITY_SEP_PI_SIGNOFF=1 below cites §1.36's routed trigger.
#
# Pattern: CLAUDE.md's `while [ ! -f STOP ]; do <cmd>; sleep 15; done`
# self-healing loop, mirroring m3fix_supervisor.sh's house convention,
# extended to iterate seeds 1,2,3 SEQUENTIALLY (one GPU, minutes/seed --
# S3-only is 6 cells x 8000 steps = 48,000 step-cells/seed, ~0.11 GPU-h/seed
# at the measured Rev-7 rate). run_capability_sep.py is resume-safe
# (validity-checked output keyed by cell_id, which now includes the seed),
# so re-invoking after a crash costs nothing beyond the crashed cell's
# partial work. NO --steps override (would clobber the Rev-7 per-group
# STEP_BUDGET pins the m3fix manifest prices against). --m3fix-groups S3
# restricts each seed's run to S3's 6 cells (4 variant-A + 2 variant-B)
# instead of the full 30-cell both-variant manifest.
set -uo pipefail

CAP_DIR=/home/nvidia/chapter2/capability_separation
PY=/home/nvidia/tdenv/bin/python

cd "$CAP_DIR"
rm -f STOP_m3fix_s3ext
mkdir -p results_m3fix_s3ext
export CUDA_VISIBLE_DEVICES=0
export CAPABILITY_SEP_PI_SIGNOFF=1   # §1.36's routed S3 marginality-trigger extension

echo "[supervisor] === M3FIX S3 SEED EXTENSION START $(date -u) ===" | tee -a m3fix_s3ext_wave.log
for SEED in 1 2 3; do
  if [ -f STOP_m3fix_s3ext ]; then
    echo "[supervisor] STOP file present -- aborting before seed=$SEED $(date -u)" | tee -a m3fix_s3ext_wave.log
    break
  fi
  echo "[supervisor] --- seed=$SEED start $(date -u) ---" | tee -a m3fix_s3ext_wave.log
  while true; do
    "$PY" run_capability_sep.py --m3fix --m3fix-seed "$SEED" --m3fix-groups S3 \
      --device cuda --results-dir results_m3fix_s3ext/ 2>&1 | tee -a m3fix_s3ext_wave.log
    RC=${PIPESTATUS[0]}
    if [ "$RC" -eq 0 ]; then
      echo "[supervisor] seed=$SEED exited 0 -- DONE $(date -u)" | tee -a m3fix_s3ext_wave.log
      break
    else
      echo "[supervisor] seed=$SEED exited $RC -- retrying in 15s ($(date -u))" | tee -a m3fix_s3ext_wave.log
      sleep 15
    fi
  done
done
touch "$CAP_DIR/results_m3fix_s3ext/M3FIX_S3EXT_STAGE_DONE"
echo "[supervisor] === M3FIX S3 SEED EXTENSION COMPLETE $(date -u) ===" | tee -a "$CAP_DIR/m3fix_s3ext_wave.log"
