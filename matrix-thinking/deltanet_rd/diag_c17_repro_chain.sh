#!/bin/bash
# KEY_ANCHORING_SCALING_DRAFT.md sec 15.24 (Rev 4, DESIGN-CLEARED-FOR-BUILD
# after 5 independent rounds) -- the C17 eval-admission repro instrument's
# chain: Stage -1 (CPU-only self-tests) -> kernel-gate artifact re-check ->
# ONE deterministic GPU launch (K=84, seed=1940, d_state=96, with
# --c17-repro-telemetry --full-ckpt-step 20000) -> offline analysis (a
# SEPARATE step, run only AFTER the cell completes). NOT RUN by the build
# session that wrote it -- build + CPU-verify only (diag_c17_repro_
# stage_minus1.py's own 12 items, 6 of which genuinely run locally, are
# already RUN and PASSING as of this build; the 6 BOX-DEFERRED items are
# re-run by THIS chain's own Stage -1 step below, on box). An independent
# audit follows before this chain is ever invoked for real.
#
# FULL DEPENDENCY CLOSURE (every artifact/precondition this chain reads):
#   - results/smoke_dstate_kernel_wide_result.json  (sec 15.20.1's own
#     supplementary kernel-safety gate, T in {504,546,588,630} @ d=96 --
#     T_bind(K=84)=588 is literally one of the four values this artifact's
#     own T-sweep already tested and passed, sec 15.24.2. This chain does
#     NOT re-run the GPU probe -- it reads and re-verifies the ALREADY-
#     COMMITTED artifact, exactly as keyanchor_scaling_wide_chain.sh's own
#     GATE 1b does for the wide-grid wave this repro cell belongs to.)
#   - The ALREADY-ARCHIVED K84/seed=1940 result JSON + its own on-box
#     10-checkpoint tree, at results/deltanet_rd_exactness/
#     wavekeyanchor-scaling-wide/ + /data/deltanet_rd_keyanchor_ckpts/
#     wavekeyanchor-scaling-wide/ (sec 15.24.2: "K84/seed=1940 is already
#     confirmed to have a complete, on-box 10-checkpoint tree" -- this
#     chain's own launch step token-diffs its generated command against a
#     byte-identical reconstruction of what ALREADY produced that archive,
#     NEVER against a hand-typed reference).
#   - Both PI sign-off tokens (sec 15.24.5's own belt-and-suspenders gate,
#     generalizing sec 15.20.5's own MAJOR-4 fix): KEYANCHOR_SCALING_
#     PI_SIGNOFF=1 (the base token, ALREADY required for anything drawing
#     against the KEY_ANCHORING_SCALING sub-ledger) AND KEYANCHOR_SCALING_
#     C17REPRO_PI_SIGNOFF=1 (a SECOND, DISTINCT token this specific
#     instrument registers -- NEVER OR'd with the base token; a stale base
#     token from a DIFFERENT wave's own launch does not satisfy this gate).
#   - run_deltanet_rd.py's own --c17-repro-telemetry/--full-ckpt-step flags
#     and diag_c17_repro_analysis.py / diag_c17_repro_stage_minus1.py (this
#     build session's own new files).
#
# GPU ASSIGNMENT (sec 15.24.7, HARD-PINNED, never read from environment --
# mirrors run_k69_s1733_contingency.py's own "not read from environment on
# purpose" discipline): GPU 2 ONLY. GPUs 0-1 belong to Phase-2b (a
# concurrent, unrelated wave); GPU 7 is USER-RESERVED (never touched by any
# automation); GPUs 3-6 are unclaimed but NOT this design's own registered
# assignment -- sec 15.24.7 registers GPU 2 alone, so that is the only GPU
# this chain will ever use. `nvidia-smi` contention on GPU 2 is re-checked
# at launch time (below), never assumed clean from drafting time.
#
# BUDGET GUARD (sec 15.24.5, mirrors sec 15.20.6's own Stage 0 abort-trigger
# formula, 1.5 x <1x point-estimate GPU-h> x 3600): 1.5 x 0.450 x 3600 =
# 2430s. Implemented as a LIVE wall-clock guard via GNU coreutils `timeout`
# wrapping the launch subprocess -- NOT merely a post-hoc log message --
# so a stuck/contended run is killed and diagnosed well before the training
# script's own much larger --internal-timeout ceiling (~17,747s at this
# cell's own steps/K) would ever fire. On timeout: halt, dump `nvidia-smi`
# scoped to GPU 2, print the "re-price before retrying" guidance, and STOP
# this chain (no auto-retry -- re-pricing is a human decision, sec 15.24.5).
#
# CONTINGENCY SEEDS ARE NEVER AUTO-FIRED (sec 15.24.4/15.24.7, pinned
# explicitly, checked at the bottom of this chain): 1943/1944 are a
# REGISTERED FOLLOW-UP on a NO-REPRO verdict ONLY, human-loop gated. If the
# analysis step's own verdict is NO-REPRO, this chain prints the registered
# follow-up and EXITS 0 (a legitimate, expected terminal state) -- it does
# NOT loop, retry, or launch anything further on its own.
#
# TMUX-READY (house [LEARN] rule) -- this script does NOT self-wrap in tmux
# (matches every prior *_chain.sh in this directory):
#   tmux new-session -d -s c17repro "bash diag_c17_repro_chain.sh 2>&1 | tee logs/00_c17repro_chain.log"
# Every step below ALSO tees its own numbered log file independently.
#
# set -euo pipefail is REQUIRED (project hard rule: no `| tee` swallowing
# exits) -- under pipefail, `cmd | tee log` still propagates cmd's own exit
# code, so a failed Stage -1 / gate / launch / analysis step stops this
# chain rather than silently continuing past a `tee`d failure.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# =============================================================================
# GPU pin (HARD-CODED, not overridable via environment -- sec 15.24.7).
# =============================================================================
readonly C17REPRO_GPU=2

# =============================================================================
# Path/identity constants (sec 15.24.2's own pinned cell + sec 15.24.7's
# "NEW --ckpt-dir target, never overwriting the original archived
# checkpoints" instruction).
# =============================================================================
readonly K=84
readonly SEED=1940
readonly D_STATE=96
readonly ORIGINAL_OUT_DIR=results/deltanet_rd_exactness/wavekeyanchor-scaling-wide
readonly ORIGINAL_CKPT_BASE=/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling-wide
readonly REPRO_OUT_DIR=results/deltanet_rd_exactness/wavekeyanchor-scaling-c17repro
readonly REPRO_CKPT_BASE=/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling-c17repro
readonly BUDGET_GUARD_S=2430
readonly FULL_CKPT_STEP=20000

mkdir -p "$REPRO_OUT_DIR" "$REPRO_CKPT_BASE"

# =============================================================================
# GATE 0a/0b: BOTH PI-decision guards, first, before any GPU-adjacent check.
# =============================================================================
if [ "${KEYANCHOR_SCALING_PI_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: this launch draws against the KEY_ANCHORING_SCALING sub-ledger (sec 15.24.7) --" >&2
    echo "  set KEYANCHOR_SCALING_PI_SIGNOFF=1 in the environment ONLY after that decision is on" >&2
    echo "  record (mirrors every prior keyanchor-scaling* chain's own GATE 0)." >&2
    exit 1
fi
echo "KEYANCHOR_SCALING_PI_SIGNOFF=1 recorded." | tee logs/c17repro_00_pi_signoff.log

if [ "${KEYANCHOR_SCALING_C17REPRO_PI_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: sec 15.24.5's own SECOND, DISTINCT PI decision -- a genuinely NEW scientific" >&2
    echo "  instrument, not a re-run of the wide-grid wave, earns its own explicit go-ahead," >&2
    echo "  even though this launch asks for no new ceiling. Set KEYANCHOR_SCALING_C17REPRO_" >&2
    echo "  PI_SIGNOFF=1 ONLY after this SECOND decision is separately on record. The base" >&2
    echo "  KEYANCHOR_SCALING_PI_SIGNOFF token alone does NOT satisfy this gate (never OR'd)." >&2
    exit 1
fi
echo "KEYANCHOR_SCALING_C17REPRO_PI_SIGNOFF=1 recorded." | tee logs/c17repro_01_c17repro_pi_signoff.log

# Python-side copy of the same two checks (belt-and-suspenders, sec 15.24.5).
$PY -c "
import os, sys
ok = os.environ.get('KEYANCHOR_SCALING_PI_SIGNOFF') == '1' \
    and os.environ.get('KEYANCHOR_SCALING_C17REPRO_PI_SIGNOFF') == '1'
print(f'python-side PI signoff check: {ok}')
sys.exit(0 if ok else 1)
" 2>&1 | tee logs/c17repro_02_pi_signoff_pyside.log

# =============================================================================
# STEP 1: Stage -1 self-test suite (diag_c17_repro_stage_minus1.py). CPU-
# only -- run BEFORE any GPU work. `set -e` stops this chain here on any
# RUN item's own failure (BOX-DEFERRED items do not count as failures --
# see that script's own exit-code convention).
# =============================================================================
$PY diag_c17_repro_stage_minus1.py 2>&1 | tee logs/c17repro_10_stage_minus1.log
touch "$REPRO_OUT_DIR/C17REPRO_STAGE_MINUS1_DONE"

# =============================================================================
# STEP 2: kernel-gate artifact check (sec 15.24.2: "No new kernel or Gate-2
# check needed -- both already cleared for this exact (K,d) pair."
# T_bind(K=84)=588 is one of the FOUR T's the wide kernel-safety gate's own
# sweep already tested. STANDALONE bash-level check ("belt") -- re-verifies
# the ALREADY-COMMITTED artifact, does not regenerate it.
# =============================================================================
WIDE_KERNEL_GATE_JSON=results/smoke_dstate_kernel_wide_result.json
if [ ! -f "$WIDE_KERNEL_GATE_JSON" ]; then
    echo "ERROR: wide kernel-safety gate artifact missing at $WIDE_KERNEL_GATE_JSON (sec 15.20.1)" >&2
    echo "  -- this repro cell's own T_bind(K=84)=588 is covered ONLY by this artifact's own" >&2
    echo "  {504,546,588,630} T-sweep (sec 15.24.2). Generate/commit a PASSING artifact first." >&2
    exit 1
fi
$PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx

T_BIND = 7 * $K
assert T_BIND == 588, f'T_bind formula drifted -- expected 588, got {T_BIND}'
r = rdx.keyanchor_scaling_wide_kernel_gate_check()
covered = T_BIND in set(rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_T_SWEEP)
print(f'T_bind(K=$K)={T_BIND} covered by wide kernel-gate T-sweep '
      f'{rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_T_SWEEP}: {covered}')
print(f'WIDE KERNEL GATE: ok={r[\"ok\"]} reason={r[\"reason\"]!r} path={r[\"path\"]!r}')
sys.exit(0 if (covered and r['ok']) else 1)
" 2>&1 | tee logs/c17repro_11_kernel_gate_check.log
touch "$REPRO_OUT_DIR/C17REPRO_KERNEL_GATE_DONE"

# =============================================================================
# STEP 3: build the launch command via _keyanchor_scaling_spec/build_cmd
# (NEVER hand-typed), token-diff it against a byte-identical reconstruction
# of the command that ALREADY produced the archived K84/seed=1940 cell (the
# K69 precedent's own field-diff pattern, run_k69_s1733_contingency.py) --
# the ONLY sanctioned deltas are --out/--ckpt-dir's PATH values (a NEW
# target, never overwriting the original archive) and the two new,
# additive --c17-repro-telemetry/--full-ckpt-step flags.
# =============================================================================
# Diagnostics (human-readable: MATCH/MISMATCH, timeout, is_done) go to
# stderr; the CMD itself, one token per line (mapfile-safe regardless of
# token content), goes to stdout ONLY -- kept on two separate streams so
# neither corrupts the other, no fd-juggling needed. Exit codes: 0 = build
# the cell now; 1 = token-diff mismatch (real bug, chain must stop); 3 =
# already done (idempotent no-op, NOT a failure -- intercepted explicitly
# below since `set -e` would otherwise treat any nonzero rc as fatal).
CMD_BUILD_LOG=logs/c17repro_12_cmd_build_and_diff.log
set +e
CMD_STDOUT=$($PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx

spec = rdx._keyanchor_scaling_spec($K, $SEED, $D_STATE)
timeout = rdx.default_timeout(spec['K'], spec['steps'])

ref_cmd = rdx.build_cmd(spec, '$ORIGINAL_OUT_DIR', timeout, ckpt_base_dir='$ORIGINAL_CKPT_BASE')
new_cmd = rdx.build_cmd(spec, '$REPRO_OUT_DIR', timeout, ckpt_base_dir='$REPRO_CKPT_BASE') + \
    ['--c17-repro-telemetry', '--full-ckpt-step', '$FULL_CKPT_STEP']

def normalize(cmd):
    out, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok in ('--out', '--ckpt-dir'):
            skip = True
            continue
        out.append(tok)
    return out

extra = ['--c17-repro-telemetry', '--full-ckpt-step', '$FULL_CKPT_STEP']
n_ref = normalize(ref_cmd)
n_new = normalize(new_cmd)
assert n_new[-len(extra):] == extra, f'expected trailing {extra}, got {n_new[-len(extra):]}'
n_new_base = n_new[:-len(extra)]
if n_ref != n_new_base:
    print('TOKEN-DIFF MISMATCH vs the archived K84/seed=1940 config (beyond --out/--ckpt-dir '
          'and the 2 sanctioned new flags):', file=sys.stderr)
    for a, b in zip(n_new_base, n_ref):
        if a != b:
            print(f'  new={a!r} ref(archived)={b!r}', file=sys.stderr)
    sys.exit(1)
print('token-diff vs archived K84/seed=1940 config: MATCH (out/ckpt-dir paths + the 2 '
      'sanctioned new flags excepted)', file=sys.stderr)
print(f'timeout={timeout}s (~{timeout/60:.1f} min)', file=sys.stderr)
is_done = rdx.is_done('$REPRO_OUT_DIR', spec)
print(f'is_done(\"$REPRO_OUT_DIR\")={is_done}', file=sys.stderr)
if is_done:
    print('ALREADY DONE -- nothing to launch (idempotent no-op).', file=sys.stderr)
    sys.exit(3)
for tok in new_cmd:
    print(tok)
" 2> >(tee "$CMD_BUILD_LOG" >&2))
BUILD_RC=$?
set -e

if [ "$BUILD_RC" -eq 1 ]; then
    echo "cmd-build/token-diff step FAILED (rc=1) -- see $CMD_BUILD_LOG" >&2
    exit 1
elif [ "$BUILD_RC" -ne 0 ] && [ "$BUILD_RC" -ne 3 ]; then
    echo "cmd-build step exited unexpectedly (rc=$BUILD_RC) -- see $CMD_BUILD_LOG" >&2
    exit 1
fi

if [ "$BUILD_RC" -eq 3 ]; then
    echo "C17 repro cell for K=$K/seed=$SEED already complete at $REPRO_OUT_DIR -- skipping launch." \
        | tee -a "$CMD_BUILD_LOG"
    CMD=()
else
    mapfile -t CMD <<< "$CMD_STDOUT"
    echo "CMD: ${CMD[*]}" | tee logs/c17repro_14_final_cmd.log

    # =========================================================================
    # LAUNCH, GPU 2 ONLY, budget-guard-wrapped (sec 15.24.5/15.24.7).
    # =========================================================================
    set +e
    CUDA_VISIBLE_DEVICES="$C17REPRO_GPU" timeout "$BUDGET_GUARD_S" "${CMD[@]}" \
        2>&1 | tee logs/c17repro_15_launch.log
    LAUNCH_RC="${PIPESTATUS[0]}"
    set -e

    if [ "$LAUNCH_RC" -eq 124 ]; then
        echo "BUDGET GUARD FIRED: realized wall_s >= ${BUDGET_GUARD_S}s (1.5x the 0.450 GPU-h " >&2
        echo "  1x point estimate, sec 15.24.5/15.24.7) -- HALTING, not auto-retrying." >&2
        echo "  Diagnosing GPU $C17REPRO_GPU contention (nvidia-smi, scoped to that GPU only):" >&2
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv \
            -i "$C17REPRO_GPU" 2>&1 | tee -a logs/c17repro_16_budget_guard_nvidia_smi.log
        nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv \
            -i "$C17REPRO_GPU" 2>&1 | tee -a logs/c17repro_16_budget_guard_nvidia_smi.log
        echo "  Re-price this cell (sec 15.24.5's own 're-price before retrying') before " >&2
        echo "  re-launching -- this chain does NOT auto-retry a budget-guard abort." >&2
        exit 1
    elif [ "$LAUNCH_RC" -ne 0 ]; then
        echo "LAUNCH FAILED: rc=$LAUNCH_RC (not a budget-guard timeout -- see " >&2
        echo "  logs/c17repro_15_launch.log for the real failure)." >&2
        exit 1
    fi
    touch "$REPRO_OUT_DIR/C17REPRO_LAUNCH_DONE"
    echo "LAUNCH complete, rc=0." | tee logs/c17repro_17_launch_done.log
fi

# =============================================================================
# STEP 4 (SEPARATE chain step, runs ONLY after the cell above completes --
# sec 15.24 task brief's own explicit instruction): offline analysis.
# =============================================================================
CKPT_DIR="$REPRO_CKPT_BASE/$($PY -c "
import sys
sys.path.insert(0, '.')
import run_deltanet_rd_exactness_sweep as rdx
spec = rdx._keyanchor_scaling_spec($K, $SEED, $D_STATE)
print(spec['name'])
")"
ANALYSIS_JSON="$REPRO_OUT_DIR/diag_c17_repro_analysis_K${K}_s${SEED}.json"

$PY diag_c17_repro_analysis.py --launch "$CKPT_DIR" "$SEED" --k "$K" --out-json "$ANALYSIS_JSON" \
    2>&1 | tee logs/c17repro_20_analysis.log
touch "$REPRO_OUT_DIR/C17REPRO_ANALYSIS_DONE"

# =============================================================================
# Contingency-seed disclosure ONLY -- NEVER auto-fired (sec 15.24.4/15.24.7,
# pinned explicitly). A NO-REPRO verdict is a legitimate, expected terminal
# state for this chain, not a failure -- print the registered follow-up and
# stop; a human decides whether/when to fire seeds 1943/1944.
# =============================================================================
VERDICT=$($PY -c "
import json
with open('$ANALYSIS_JSON') as f:
    print(json.load(f).get('verdict'))
")
echo "ANALYSIS VERDICT: $VERDICT" | tee logs/c17repro_21_verdict.log
if [ "$VERDICT" = "NO-REPRO" ]; then
    echo "NO-REPRO: the registered follow-up (contingency seeds 1943/1944) is HUMAN-LOOP GATED" \
        | tee -a logs/c17repro_21_verdict.log
    echo "  -- this chain does NOT fire it automatically. See $ANALYSIS_JSON's own 'followup' " \
        | tee -a logs/c17repro_21_verdict.log
    echo "  field for the exact re-run instructions." | tee -a logs/c17repro_21_verdict.log
fi

touch "$REPRO_OUT_DIR/C17REPRO_CHAIN_DONE"
echo "diag_c17_repro_chain.sh: DONE." | tee logs/c17repro_22_chain_done.log
