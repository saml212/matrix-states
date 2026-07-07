#!/bin/bash
# KEY_ANCHORING_SCALING_DRAFT.md sec 15 (attack-round-1 verdict RUN-AFTER-
# REASONING-LINK, wave PARKED, sec 15.18) -- keyanchor-scaling chain
# (cliff-location scaling law across d_state in {80,96}). NOT RUN by the
# build session that wrote it -- build + CPU-verify only, no GPU work
# performed anywhere in this build. The orchestrator runs this ONLY after
# (a) REASONING-LINK Phase 1 has landed (sec 15.18's own accepted
# sequencing decision -- this wave is explicitly PARKED, not merely
# "not yet launched") AND (b) a free GPU set is confirmed (check nvidia-smi
# FIRST, per house rule) AND (c) both PI-decisions sec 15's own header
# requires are on record: reopening the KEY_ANCHORING program (declared
# complete twice already, sec 15 header) AND a NEW sub-ledger allocation
# (Tier 1: 21 GPU-h, sec 15.12 -- this wave's own reserve CANNOT be funded
# from the exhausted 80 GPU-h KEY_ANCHORING ledger, which has only 1.773
# GPU-h left).
#
# FULL DEPENDENCY CLOSURE (every artifact/precondition this chain reads,
# stated once so nothing is silently assumed present):
#   - results/smoke_dstate_kernel_result.json  (sec 15.2/15.18 kernel-safety
#     gate artifact -- MUST exist, exit_code==0, verdict contains 'CLEARED',
#     t_sweep covers {128,224,448}, grid_pass['80']/['96'] all True at every
#     T. MECHANICALLY enforced below, TWICE: once as a standalone bash-level
#     check (belt), once again inside run_deltanet_rd_exactness_sweep.py's
#     own keyanchor_scaling_stage_gate (suspenders, sec 15.18 Q1's own
#     'gates-must-abort' requirement) -- neither check trusts the other.)
#   - REV7_THRESHOLD_PINNED_D80.json / _D96.json  (sec 15.5 -- ALREADY
#     GENERATED and committed by this build session; re-derived and
#     byte-diff-checked below as a defense-in-depth re-verification, not
#     assumed still valid from a prior session)
#   - key_anchoring.py's GATE2_N_ITER_BY_K  (extended this build session
#     with this wave's own genuinely-new K's -- 20/24/43/51/53/57/58/63/69
#     -- K=48 deliberately left untouched, sec 15.6)
#   - run_deltanet_rd_exactness_sweep.py's --wave keyanchor-scaling dispatch
#     (this build session's own FATAL-1 fix)
#   - smoke_keyanchor_scaling.py (this build session's own MAJOR-3 fix)
#   - model_rd.py's _SAFE_D_STATE / run_deltanet_rd.py's --d-state argparse
#     choices  -- EXTENDED to (64,80,96,128) / choices=[64,80,96,128] by
#     this build session (sec 15.2 item 3's own "only after BOTH new d's
#     pass" gate -- licensed strictly by the PASSING kernel-safety
#     artifact checked above, never added speculatively ahead of it). The
#     box's own working tree must have this same commit/diff present
#     before this chain's training cells can launch -- verify with
#     `git log -1 -- model_rd.py run_deltanet_rd.py` or an equivalent
#     working-tree check if the box's checkout could be stale relative to
#     this repo.
#
# RESUME-SAFETY (house [LEARN] rule -- skip-if-output-valid, not merely
# skip-if-exists): inherited for free from run_deltanet_rd_exactness_
# sweep.py's own is_done() (checks complete==True, steps_completed>=steps,
# AND every arm-identity field, never the filename alone) -- this chain
# performs NO separate resume bookkeeping of its own; re-running any line
# below after a partial/interrupted prior run is always safe.
#
# TMUX-READY (house [LEARN] rule): this script does NOT self-wrap in tmux
# (matching every prior *_chain.sh in this directory -- none of them do
# either) -- the ORCHESTRATOR launches it inside a named session so a local
# Claude Code session restart cannot kill it:
#   tmux new-session -d -s keyanchor-scaling "bash keyanchor_scaling_chain.sh 2>&1 | tee logs/00_chain.log"
# Every step below ALSO tees its own numbered log file independently, so
# the full record survives even if the outer tmux pane's own capture is
# lost.
#
# set -euo pipefail is REQUIRED (project hard rule: no `| tee` swallowing
# exits) -- under pipefail, `cmd | tee log` still propagates cmd's own
# exit code, so a failed smoke/gate/launch/stage-gate step stops this
# chain rather than silently continuing past a `tee`d failure.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# =============================================================================
# GATE 0 (BUDGET/PI-DECISION GUARD -- must be first, before any GPU-adjacent
# check even runs): sec 15's own header framing, restated as a MECHANICAL
# refusal, not prose a human could skip past. This is NOT the same override
# class as --accept-scaling-stage-override (which only ever bypasses the
# in-code calibration-completeness/abort-trigger gate) -- this is the
# ledger-reopening decision itself, recorded here as an explicit env-var
# sign-off so a bare `bash keyanchor_scaling_chain.sh` can never launch
# training by accident.
# =============================================================================
if [ "${KEYANCHOR_SCALING_PI_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: KEY_ANCHORING_SCALING_DRAFT.md sec 15 header -- this wave is PARKED" >&2
    echo "  (attack-round-1 verdict RUN-AFTER-REASONING-LINK, sec 15.18) and requires BOTH" >&2
    echo "  (a) an explicit PI decision to REOPEN the KEY_ANCHORING program (declared complete" >&2
    echo "  twice already) and (b) a NEW sub-ledger allocation (Tier 1: 21 GPU-h, sec 15.12 --" >&2
    echo "  the existing ledger's own 1.773 GPU-h reserve cannot fund even the cheapest" >&2
    echo "  mandatory cell). This chain refuses to proceed past this point without an explicit," >&2
    echo "  documented record of both decisions. Set KEYANCHOR_SCALING_PI_SIGNOFF=1 in the" >&2
    echo "  environment ONLY after both are on record (mirrors keyanchor_dose_chain.sh's own" >&2
    echo "  Stage-2 PI-signoff convention, sec 14.4 Option 1)." >&2
    exit 1
fi
echo "KEYANCHOR_SCALING_PI_SIGNOFF=1 recorded -- proceeding past the ledger/reopening guard." \
    | tee logs/60_pi_signoff.log

# =============================================================================
# GATE 1 (MECHANICAL kernel-safety gate, sec 15.2/15.18 Q1 -- the wave's
# own single highest-consequence risk). This is a STANDALONE bash-level
# check ("belt") that refuses BEFORE invoking the Python orchestrator at
# all -- run_deltanet_rd_exactness_sweep.py's own keyanchor_scaling_stage_gate
# performs the SAME check again independently ("suspenders", sec 15.18 Q1's
# own "gates-must-abort" requirement -- an ENFORCED tooling gate, not a
# documented prerequisite a human could skip). Reuses keyanchor_scaling_
# kernel_gate_check DIRECTLY (never a hand-copied twin of its own logic)
# since that module is fla-free and already proven importable in this
# environment by every keyanchor-* smoke gate.
# =============================================================================
KERNEL_GATE_JSON=results/smoke_dstate_kernel_result.json
if [ ! -f "$KERNEL_GATE_JSON" ]; then
    echo "ERROR: kernel-safety gate artifact missing at $KERNEL_GATE_JSON (sec 15.2 item 1) --" >&2
    echo "  re-run smoke_dstate_kernel.py to the FULL registered T in {128,224,448} protocol" >&2
    echo "  and commit a PASSING artifact before retrying this chain." >&2
    exit 1
fi
$PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx
r = rdx.keyanchor_scaling_kernel_gate_check()
print(f'KERNEL GATE: ok={r[\"ok\"]} reason={r[\"reason\"]!r} path={r[\"path\"]!r}')
sys.exit(0 if r['ok'] else 1)
" 2>&1 | tee logs/61_kernel_gate_check.log

# =============================================================================
# GPUs 0-1 run rung-3 (untouchable). GPUs 2-7 free but shared CONCURRENTLY
# with other programs on this box (sec 15.13's own pre-launch contention
# note) -- RE-CHECK nvidia-smi at actual launch time, not assumed from any
# prior session's own read; override via env vars once a real GPU set is
# confirmed free. Both d=80 and d=96 calibration cells run IN PARALLEL
# (sec 15.13 Stage 0 -- no cross-d dependency, unlike keyanchor-dstate's
# single-cell gate), so 2 GPUs minimum are needed for Stage 0 itself.
# =============================================================================
KEYANCHOR_SCALING_GPUS=${KEYANCHOR_SCALING_GPUS:-2}
KEYANCHOR_SCALING_GPU_OFFSET=${KEYANCHOR_SCALING_GPU_OFFSET:-2}

mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling

# =============================================================================
# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt). Zero-GPU items registered by sec 15.5/15.6/
# 15.7 that are NOT re-run automatically by main() (Wave -1 pre-registration
# artifacts, not per-launch gates) are run here explicitly, once, ahead of
# any training cell.
# =============================================================================
$PY rev7_threshold_derive.py --d-state 80 --out REV7_THRESHOLD_PINNED_D80.json \
    2>&1 | tee logs/62_rev7_threshold_derive_d80.log
$PY rev7_threshold_derive.py --d-state 96 --out REV7_THRESHOLD_PINNED_D96.json \
    2>&1 | tee logs/63_rev7_threshold_derive_d96.log
$PY smoke_key_anchoring.py 2>&1 | tee logs/64_smoke_key_anchoring_scaling.log
$PY smoke_keyanchor_scaling.py 2>&1 | tee logs/65_smoke_keyanchor_scaling.log
$PY gate2_construction_test.py 2>&1 | tee logs/66_gate2_construction_test_scaling.log

# =============================================================================
# STAGE: CALIBRATION (sec 15.13 Stage 0) -- BOTH d=80's K=43/seed=1030 AND
# d=96's K=51/seed=1430 cells, launched together, IN PARALLEL (no cross-d
# dependency -- unlike keyanchor-dstate's single-cell/single-d gate, this
# wave's own abort/re-price trigger is evaluated independently per d).
# Blinded per sec 15.13's own F9-inherited discipline: only wall_s is read
# by the stage gate below (never h4) before Stage 1's own 26 remaining
# mandatory cells launch.
# =============================================================================
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-scaling --scaling-stage calibration \
    --gpus "$KEYANCHOR_SCALING_GPUS" --gpu-offset "$KEYANCHOR_SCALING_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling \
    2>&1 | tee logs/67_wave_keyanchor_scaling_calibration.log

# CHECKPOINT (sec 15.13): each calibration cell's own completeness AND
# abort/re-price trigger (1.5x its own d's pessimistic-2x bracket) are
# re-checked by `--scaling-stage full` below. `set -e` stops this chain
# here if either cell overruns its own trigger (sys.exit(1), sec 15.13's
# own text: "halt, diagnose, re-price the FULL budget table" -- this wave
# does NOT unilaterally descope its own K-grid the way keyanchor-dstate's
# Option B/C branches do) -- do NOT re-run with
# --accept-scaling-stage-override without first diagnosing per sec
# 13.6.1 item 1's own precedent (nvidia-smi contention check first).
#
# STAGE: FULL (sec 15.13 Stage 1) -- the remaining 26 mandatory cells (30
# total minus the 2 already run as calibration). --include-scaling-gate1 is
# OFF by default (first cut under budget pressure, sec 15.14's own priority
# order) -- add it explicitly only after the mandatory grid completes and
# GPU headroom remains within the wave's own registered ceiling.
# =============================================================================
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-scaling --scaling-stage full \
    --gpus "$KEYANCHOR_SCALING_GPUS" --gpu-offset "$KEYANCHOR_SCALING_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling \
    2>&1 | tee logs/68_wave_keyanchor_scaling_full.log

# =============================================================================
# Real-data fits (sec 15.10) -- ANCHOR-FREE at both new d's (neither 80 nor
# 96 is in fit_cliff_curve.py's own ANCHORED_D_STATES=(64,), sec 15.8) --
# --k32-dir/--k48-dir OMITTED entirely (never passed at any d != 64,
# fit_cliff_curve.py's own argparse asserts this). --k-grid is REQUIRED --
# each d's OWN 5-point grid including the mandatory low-K anchor.
# =============================================================================
$PY fit_cliff_curve.py \
    --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-scaling \
    --d-state 80 --k-grid 20 43 48 53 58 \
    --n-trials 4000 --out fit_cliff_curve_d80_results.json \
    2>&1 | tee logs/69_fit_cliff_curve_d80.log

$PY fit_cliff_curve.py \
    --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-scaling \
    --d-state 96 --k-grid 24 51 57 63 69 \
    --n-trials 4000 --out fit_cliff_curve_d96_results.json \
    2>&1 | tee logs/70_fit_cliff_curve_d96.log

touch results/deltanet_rd_exactness/KEYANCHOR_SCALING_CHAIN_DONE
