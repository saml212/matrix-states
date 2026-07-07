#!/bin/bash
# KEY_ANCHORING_SCALING_DRAFT.md sec 15.20 Rev 1 (2026-07-07, post-attack-
# round-1) -- keyanchor-scaling-wide chain: d=96 wider-K cliff-hunting wave
# (K in {72,78,84,90} x 3 seeds, 12 new GPU cells) + d=80 seed escalation
# (K=48/53's already-registered sec 15.14 contingency seeds, 4 new GPU
# cells) + the fit_cliff_curve.py admissibility-filter fix (zero-GPU, built
# and regression-tested this session, no runtime step of its own beyond the
# fit invocations at the bottom of this chain). NOT RUN by the build session
# that wrote it -- build + CPU-verify only, no GPU/SSH work performed
# anywhere in this build. Rev 1 has NOT yet had its own independent audit
# pass (sec 15.20 header's own disclosure) -- do not launch this chain until
# that pass clears it.
#
# FULL DEPENDENCY CLOSURE (every artifact/precondition this chain reads,
# stated once so nothing is silently assumed present):
#   - results/smoke_dstate_kernel_result.json  (the ORIGINAL sec 15.2/15.18
#     kernel-safety gate artifact, T in {128,224,448} -- already committed by
#     the original keyanchor-scaling build; re-verified below, not assumed
#     still valid from a prior session).
#   - results/smoke_dstate_kernel_wide_result.json  (the NEW sec 15.20.1
#     supplementary kernel-safety gate artifact, T in {504,546,588,630} @
#     d=96 -- NOT YET GENERATED, box-only: run `python
#     smoke_dstate_kernel_wide.py` on the GPU box FIRST and commit a PASSING
#     artifact before this chain can proceed past GATE 1b below).
#   - results/keyanchor_scaling_wide_niter_result.json  (sec 15.20.6's own
#     design-doc Gate (b) -- GATE2_N_ITER_BY_D_K[96] n_iter-sufficiency at K
#     in {72,78,84,90} -- build-audit MAJOR-1 fix, 2026-07-07: ALREADY
#     GENERATED and committed, CPU-only/no-GPU, via
#     `../../.venv/bin/python keyanchor_scaling_wide_niter_check.py`; verdict
#     PASS, all four K's converge at n_iter=20, <0.5% rel change to n_iter=24
#     -- see GATE (b) below).
#   - results/keyanchor_scaling_wide_k69_copy_manifest.sha256  (sec 15.20.1's
#     ENFORCED sha256 gate, Gate (c) -- ALREADY GENERATED and committed this
#     build session, pinned against the ORIGINAL sec 15.19 archive; see
#     run_deltanet_rd_exactness_sweep.py's own KEYANCHOR_SCALING_WIDE_K69_
#     PINNED_SHA256_PATH).
#   - results/deltanet_rd_exactness/wavekeyanchor-scaling/  (the ORIGINAL
#     wave's own out_dir -- MUST already hold its 30 completed cells,
#     including the 3 K=69/d=96 result JSONs this chain copies from, AND is
#     where the d=80-escalation leg's own 4 new cells land, per sec 15.20.2's
#     own re-fit invocation).
#   - key_anchoring.py's GATE2_N_ITER_BY_K  (extended this build session with
#     K=72/78/90 -- K=84 deliberately left untouched, sec 15.20.1).
#   - run_deltanet_rd_exactness_sweep.py's --wave keyanchor-scaling-wide
#     dispatch (this build session's own new machinery, "mostly config
#     reuse" of the existing keyanchor-scaling functions per sec 15.20).
#   - smoke_keyanchor_scaling_wide.py (this build session's own new wide-
#     suite sibling to smoke_keyanchor_scaling.py -- BOTH suites run below,
#     sec 15.20.6 Stage -1 item 6).
#   - fit_cliff_curve.py's admissibility-filter fix (this build session,
#     sec 15.20.3 -- confirmed this session to reproduce degenerate_frac=
#     94.77% exactly on the archived sec 15.19 d=96 raws).
#   - Both PI-decision sign-off tokens on record (sec 15.20.5's own MAJOR-4
#     fix -- see GATE 0a/0b below): KEYANCHOR_SCALING_PI_SIGNOFF=1 (the
#     ORIGINAL sec 15 reopening decision) AND KEYANCHOR_SCALING_EXT_PI_
#     SIGNOFF=1 (the SECOND, DISTINCT +5.0 GPU-h sub-ledger-extension
#     decision, sec 15.20.5) -- NEVER OR'd together; a stale original-wave
#     token does NOT satisfy the second gate.
#
# RESUME-SAFETY (house [LEARN] rule -- skip-if-output-valid, not merely
# skip-if-exists): inherited for free from is_done() (checks complete==True,
# steps_completed>=steps, AND every arm-identity field) -- this chain
# performs NO separate resume bookkeeping of its own; re-running any line
# below after a partial/interrupted prior run is always safe. The K=69 copy
# +sha256 step is ALSO idempotent (shutil.copy2 overwrites; the sha256 check
# re-verifies from scratch every time, never trusting a stale sentinel).
#
# TMUX-READY (house [LEARN] rule): this script does NOT self-wrap in tmux
# (matches every prior *_chain.sh in this directory) -- the ORCHESTRATOR
# launches it inside a named session so a local Claude Code session restart
# cannot kill it:
#   tmux new-session -d -s keyanchor-scaling-wide "bash keyanchor_scaling_wide_chain.sh 2>&1 | tee logs/00_wide_chain.log"
# Every step below ALSO tees its own numbered log file independently.
#
# set -euo pipefail is REQUIRED (project hard rule: no `| tee` swallowing
# exits) -- under pipefail, `cmd | tee log` still propagates cmd's own exit
# code, so a failed smoke/gate/launch/stage-gate step stops this chain
# rather than silently continuing past a `tee`d failure.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# =============================================================================
# GATE 0a/0b (BOTH PI-DECISION GUARDS -- must be first, before any GPU-
# adjacent check even runs). GATE 0a mirrors keyanchor_scaling_chain.sh's own
# GATE 0 exactly (the ORIGINAL sec 15 reopening decision, still required).
# GATE 0b is a SECOND, DISTINCT guard (sec 15.20.5 MAJOR-4 fix) -- a stale
# KEYANCHOR_SCALING_PI_SIGNOFF=1 exported from launching the ORIGINAL wave
# must NOT satisfy this new +5.0 GPU-h sub-ledger-extension decision.
# =============================================================================
if [ "${KEYANCHOR_SCALING_PI_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: sec 15's own header -- this wave requires an explicit PI decision to REOPEN" >&2
    echo "  the KEY_ANCHORING program (declared complete twice already). Set" >&2
    echo "  KEYANCHOR_SCALING_PI_SIGNOFF=1 in the environment ONLY after that decision is on" >&2
    echo "  record (mirrors keyanchor_scaling_chain.sh's own GATE 0)." >&2
    exit 1
fi
echo "KEYANCHOR_SCALING_PI_SIGNOFF=1 recorded -- proceeding past the ORIGINAL reopening guard." \
    | tee logs/80_pi_signoff.log

if [ "${KEYANCHOR_SCALING_EXT_PI_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: sec 15.20.5's own SECOND, DISTINCT PI decision -- the +5.0 GPU-h sub-ledger" >&2
    echo "  extension (21 -> 26 GPU-h) this wave requires is a NEW ask, not covered by the" >&2
    echo "  ORIGINAL wave's own KEYANCHOR_SCALING_PI_SIGNOFF token (Rev 1 attack-round-1" >&2
    echo "  MAJOR-4 fix -- the two tokens are never OR'd together). Set" >&2
    echo "  KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 in the environment ONLY after this SECOND" >&2
    echo "  decision is separately on record." >&2
    exit 1
fi
echo "KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 recorded -- proceeding past the +5.0 GPU-h extension guard." \
    | tee logs/81_ext_pi_signoff.log

# =============================================================================
# GATE 1 (ORIGINAL kernel-safety gate, sec 15.2/15.18, T in {128,224,448}) +
# GATE 1b (NEW wide kernel-safety gate, sec 15.20.1, T in {504,546,588,630}
# @ d=96 -- K=90's own T_bind=630 is 41% beyond the ORIGINAL 448 ceiling).
# BOTH are STANDALONE bash-level checks ("belt") that refuse BEFORE invoking
# the Python orchestrator at all -- run_deltanet_rd_exactness_sweep.py's own
# keyanchor_scaling_wide_stage_gate performs BOTH checks again independently
# ("suspenders", sec 15.18 Q1's own "gates-must-abort" requirement carried
# forward to this wave).
# =============================================================================
KERNEL_GATE_JSON=results/smoke_dstate_kernel_result.json
if [ ! -f "$KERNEL_GATE_JSON" ]; then
    echo "ERROR: ORIGINAL kernel-safety gate artifact missing at $KERNEL_GATE_JSON (sec 15.2" >&2
    echo "  item 1) -- re-run smoke_dstate_kernel.py to the FULL {128,224,448} protocol first." >&2
    exit 1
fi
WIDE_KERNEL_GATE_JSON=results/smoke_dstate_kernel_wide_result.json
if [ ! -f "$WIDE_KERNEL_GATE_JSON" ]; then
    echo "ERROR: NEW wide kernel-safety gate artifact missing at $WIDE_KERNEL_GATE_JSON (sec" >&2
    echo "  15.20.1) -- run 'python smoke_dstate_kernel_wide.py' on this GPU box FIRST (T in" >&2
    echo "  {504,546,588,630} @ d_state=96, forward+backward) and commit a PASSING artifact." >&2
    echo "  This gate is NEVER bypassable -- K=90's own T_bind=630 is untested extrapolation" >&2
    echo "  beyond the ORIGINAL protocol's own 448 ceiling until this artifact CLEARs." >&2
    exit 1
fi
$PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx
r1 = rdx.keyanchor_scaling_kernel_gate_check()
print(f'ORIGINAL KERNEL GATE: ok={r1[\"ok\"]} reason={r1[\"reason\"]!r} path={r1[\"path\"]!r}')
r2 = rdx.keyanchor_scaling_wide_kernel_gate_check()
print(f'WIDE KERNEL GATE: ok={r2[\"ok\"]} reason={r2[\"reason\"]!r} path={r2[\"path\"]!r}')
sys.exit(0 if (r1['ok'] and r2['ok']) else 1)
" 2>&1 | tee logs/82_kernel_gates_check.log

# =============================================================================
# GATE (b) (sec 15.20.6's own design-doc Gate (b), build-audit MAJOR-1 fix,
# 2026-07-07): GATE2_N_ITER_BY_D_K[96] n_iter-sufficiency, K in
# {72,78,84,90} -- registered in the gate table from the start but never
# built until this fix (KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96] was set to
# n_iter=20 "by analogy only", never mechanically confirmed at d_state=96).
# keyanchor_scaling_wide_niter_check.py is CPU-only, pure torch (no fla/CUDA
# dependency, verified) -- runs locally, artifact committed to
# results/keyanchor_scaling_wide_niter_result.json. STANDALONE bash-level
# check ("belt") -- run_deltanet_rd_exactness_sweep.py's own
# keyanchor_scaling_wide_stage_gate performs the SAME check again
# independently ("suspenders") before either leg's cells launch.
# =============================================================================
NITER_GATE_JSON=results/keyanchor_scaling_wide_niter_result.json
if [ ! -f "$NITER_GATE_JSON" ]; then
    echo "ERROR: n_iter-sufficiency gate artifact missing at $NITER_GATE_JSON (sec 15.20.6 Gate" >&2
    echo "  (b), build-audit MAJOR-1 fix) -- run '../../.venv/bin/python" >&2
    echo "  keyanchor_scaling_wide_niter_check.py' (CPU-only, no GPU needed) and commit a PASSING" >&2
    echo "  artifact first. This gate is NEVER bypassable." >&2
    exit 1
fi
$PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx
r = rdx.keyanchor_scaling_wide_niter_gate_check()
print(f'N_ITER-SUFFICIENCY GATE (b): ok={r[\"ok\"]} reason={r[\"reason\"]!r} path={r[\"path\"]!r}')
sys.exit(0 if r['ok'] else 1)
" 2>&1 | tee logs/82b_niter_gate_check.log

# =============================================================================
# GATE (c) (ENFORCED sha256 gate, sec 15.20.1 Rev 1 attack-round-1 MAJOR-1
# fix): copy the 3 already-harvested K=69/d=96 result JSONs (seeds
# 1730/1731/1732, sec 15.19) from the ORIGINAL wave's own out_dir into THIS
# wave's own wavekeyanchor-scaling-wide/ output directory, then diff their
# freshly-computed sha256 against the PINNED manifest (generated ONCE, this
# build session, against the ORIGINAL sec 15.19 archive -- never regenerated
# from the copy itself, which would make the check tautological). STANDALONE
# bash-level check ("belt") -- run_deltanet_rd_exactness_sweep.py's own
# keyanchor_scaling_wide_stage_gate performs the SAME copy+verify again
# independently ("suspenders") before either leg's own calibration/full
# logic runs, so a direct python invocation bypassing this chain cannot skip
# it either.
# =============================================================================
WIDE_OUT_DIR=results/deltanet_rd_exactness/wavekeyanchor-scaling-wide
mkdir -p "$WIDE_OUT_DIR"
$PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx
copy_report = rdx.keyanchor_scaling_wide_copy_k69('$WIDE_OUT_DIR')
if not copy_report['ok']:
    print(f'COPY FAILED: {copy_report[\"reason\"]}')
    sys.exit(1)
print(f'copied {len(copy_report[\"copied\"])} K=69 files into $WIDE_OUT_DIR')
sha_gate = rdx.keyanchor_scaling_wide_k69_sha256_gate('$WIDE_OUT_DIR')
print(f'SHA256 GATE: ok={sha_gate[\"ok\"]} reason={sha_gate[\"reason\"]!r}')
sys.exit(0 if sha_gate['ok'] else 1)
" 2>&1 | tee logs/83_k69_sha256_gate.log

# =============================================================================
# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt). BOTH the ORIGINAL suite AND this wave's own
# sibling run (sec 15.20.6 Stage -1 item 6 -- the original wave's own
# manifest must stay independently re-verifiable, this wave's own new
# machinery gets its own dedicated coverage).
# =============================================================================
$PY smoke_key_anchoring.py 2>&1 | tee logs/84_smoke_key_anchoring_scaling_wide.log
$PY smoke_keyanchor_scaling.py 2>&1 | tee logs/85_smoke_keyanchor_scaling_original.log
$PY smoke_keyanchor_scaling_wide.py 2>&1 | tee logs/86_smoke_keyanchor_scaling_wide.log
$PY gate2_construction_test.py 2>&1 | tee logs/87_gate2_construction_test_scaling_wide.log

# =============================================================================
# LEG 1: d=96 WIDE GRID (sec 15.20.1) -- STAGE: CALIBRATION (sec 15.20.6
# Stage 0), ONE cell, K=72/seed=1740, blinded readout (wall_s only).
# =============================================================================
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-scaling-wide \
    --scaling-wide-leg d96-wide --scaling-wide-stage calibration \
    --gpus "${KEYANCHOR_SCALING_WIDE_GPUS:-1}" --gpu-offset "${KEYANCHOR_SCALING_WIDE_GPU_OFFSET:-2}" \
    --per-gpu 1 --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling-wide \
    2>&1 | tee logs/88_wave_keyanchor_scaling_wide_d96_calibration.log

# CHECKPOINT (sec 15.20.6): the calibration cell's own completeness AND
# abort/re-price trigger (1.5x the d=96 main-grid pessimistic-2x bracket,
# unchanged from sec 15.14, sec 15.20.6's own disclosure) are re-checked by
# --scaling-wide-stage full below. `set -e` stops this chain here if the
# calibration cell overruns its own trigger -- do NOT re-run with
# --accept-scaling-wide-stage-override without first diagnosing (nvidia-smi
# contention check first, sec 13.6.1 item 1's own precedent).
#
# LEG 1, STAGE: FULL (sec 15.20.6 Stage 1) -- the remaining 11 d=96-wide
# cells. --include-scaling-wide-gate1 is OFF by default (sec 15.20.5's own
# explicitly-not-requested cut-eligible layer).
# =============================================================================
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-scaling-wide \
    --scaling-wide-leg d96-wide --scaling-wide-stage full \
    --gpus "${KEYANCHOR_SCALING_WIDE_GPUS:-1}" --gpu-offset "${KEYANCHOR_SCALING_WIDE_GPU_OFFSET:-2}" \
    --per-gpu 1 --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling-wide \
    2>&1 | tee logs/89_wave_keyanchor_scaling_wide_d96_full.log

# =============================================================================
# LEG 2: d=80 SEED ESCALATION (sec 15.20.2) -- 4 cells, K in {48,53}, the
# ALREADY-RESERVED contingency seeds 1133/1134/1233/1234. No calibration
# cell needed (already-proven K's). Writes into the ORIGINAL wave's own
# out_dir (wavekeyanchor-scaling/, NOT wavekeyanchor-scaling-wide/) so
# fit_cliff_curve.py's re-fit below can read all 5 d=80 seeds per K in ONE
# --cliff-out-dir invocation (see run_deltanet_rd_exactness_sweep.py's own
# module note above keyanchor_scaling_wide_stage_gate for why).
# =============================================================================
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-scaling-wide \
    --scaling-wide-leg d80-escalation \
    --gpus "${KEYANCHOR_SCALING_WIDE_GPUS:-1}" --gpu-offset "${KEYANCHOR_SCALING_WIDE_GPU_OFFSET:-2}" \
    --per-gpu 1 --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling \
    2>&1 | tee logs/90_wave_keyanchor_scaling_wide_d80_escalation.log

# =============================================================================
# Real-data fits (sec 15.20.4's own discrimination test needs the d=96
# fit's x0/CI; sec 15.20.2's own re-fit needs the tightened d=80 CI).
#
# d=96 WIDE fit: K=69 (reused, 2 admissible seeds post-fix) + K in
# {72,78,84,90} (this wave's own new cells) -- reads wavekeyanchor-scaling-
# wide/ ONLY (where the K=69 copy landed via GATE (c) above, alongside the
# 12 freshly-launched wide cells).
# =============================================================================
$PY fit_cliff_curve.py \
    --cliff-out-dir "$WIDE_OUT_DIR" \
    --d-state 96 --k-grid 69 72 78 84 90 \
    --n-trials 4000 --out fit_cliff_curve_d96_wide_results.json \
    2>&1 | tee logs/91_fit_cliff_curve_d96_wide.log

# =============================================================================
# d=80 RE-FIT (sec 15.20.2): --cliff-out-dir points at the ORIGINAL wave's
# own directory (wavekeyanchor-scaling/, matching sec 15.20.2's own re-fit
# invocation verbatim) -- now holding 5 seeds each at K=48/53 (3 original +
# 2 escalation) and 3 seeds at K=20/43/58 (unchanged). --n-trials 4000
# unchanged; n=5 at K=48/53 is a property of the DATA now present in that
# directory, not a CLI flag (fit_cliff_curve.py's own load_k_mean_h4 reads
# every completed seed it finds for a given K, sec 15.20.2's own "not
# expected to change the already-clean REFUTE verdict... but run and
# reported, not assumed" framing).
# =============================================================================
$PY fit_cliff_curve.py \
    --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-scaling \
    --d-state 80 --k-grid 20 43 48 53 58 \
    --n-trials 4000 --out fit_cliff_curve_d80_refit_results.json \
    2>&1 | tee logs/92_fit_cliff_curve_d80_refit.log

touch results/deltanet_rd_exactness/KEYANCHOR_SCALING_WIDE_DONE
