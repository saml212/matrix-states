#!/bin/bash
# REASONING_LINK_DESIGN.md sec 16.1.2 -- PHASE-1B, Stage-0-gate-only test
# (~0.01 GPU-h). This is NOT the full Phase-1b grid (sec 16.1.2's own
# "conditional (~0.4 GPU-h)" second stage, not built here, not launched by
# this script) -- it is exactly the single-cell calibration procedure sec 9
# Stage 0 already runs for the marker template, re-run on sec 16.1's
# natural-language surface form (Candidates A and B), on BOTH corpora:
#   - the WIKITEXT-mix-ext control-arm checkpoint (sec 16.1.2's own
#     registered fix -- this is the LICENSING cell, since natural-sentence
#     surface form has an actual referent in wikitext's own register, unlike
#     openr1's step-by-step math derivation text), and
#   - the OPENR1-mix-ext control-arm checkpoint, run alongside as the
#     REGISTERED, named "expected-null contrast" (sec 16.1.2/16.1.3 item
#     2b) -- if this cell ALSO clears the gate, that is a CONFOUND FLAG
#     (something generically easier about the new templates, unrelated to
#     distribution match), never a second confirmation. This script does
#     NOT gate on the openr1 cell's own verdict -- only wikitext licenses.
#
# Four gate cells total (2 templates x 2 corpora), per sec 16.1.2's own
# registration ("running BOTH candidates x 2 corpora (4 gate cells total)
# still costs ~=0.001-0.002 GPU-h"). ~0.01 GPU-h registered (conservative,
# includes CPU-only Stage -1 re-verification + build/debug margin, sec
# 16.1.2).
#
# GATE ENFORCEMENT (closes sec 15.2's DISCREPANCY finding for Phase 1's own
# chain, which computed gate_result_h1_probe_valid but never READ it before
# the grid launched): after BOTH wikitext cells run, this script calls
# reasoning_link_gate_enforce.py on each cell's own JSON and reads its REAL
# exit code -- if NEITHER candidate's wikitext cell clears the gate, this
# script's own final `exit 1` mechanically refuses (no "CLEARED" sentinel
# is written, and any caller chaining onto this script's own exit status
# also halts), exactly the enforcement Phase 1's own chain lacked. The
# openr1 cells are run and reported but never gate anything (sec 16.1.2's
# own "expected-null contrast" framing -- they inform the confound flag,
# they do not license).
#
# Three steps:
#   1. Stage -1 self-tests (ALL 18 items: sec 9's original 14 + this
#      feature's own items 15-18, sec 16.1) -- pure CPU, no GPU, no
#      checkpoints. Includes the gate-enforcement negative test (extra
#      check, wired into --mode selftest) -- a stale/broken enforcement
#      script is caught HERE, before any real cell runs.
#   2. Four stage0-natural cells: {A,B} x {wikitext-mix-ext,openr1-mix-ext},
#      K=32 (sec 9's own Stage-0 K), h in {1,2,3,4}, BLINDED console output
#      (reasoning_link_probe.py::blinded_console_summary -- identical
#      convention to the marker template's own Stage 0).
#   3. Gate enforcement on each WIKITEXT cell (the licensing corpus) via
#      reasoning_link_gate_enforce.py -- REFUSES (exits nonzero, halts this
#      script) if either candidate's wikitext cell is PROBE-INVALID.
#      Prints a final CLEARED/REFUSED summary, including the openr1
#      confound-flag check (sec 16.1.3 item 2b).
#
# ---------------------------------------------------------------------------
# CRITICAL dependency-closure list (mirrors reasoning_link_chain.sh's own
# list verbatim, PLUS the new reasoning_link_gate_enforce.py edge this
# script's own Step 3 introduces -- mech_stage2_chain.sh's own "[LEARN] from
# Stage 1: run 1 died on an unshipped dependency" -- restated here so THIS
# chain's deploy does not repeat it). Every file reasoning_link_probe.py /
# reasoning_link_stage_minus1.py / reasoning_link_gate_enforce.py import,
# TRANSITIVELY, at MODULE-IMPORT time -- verified by direct read of every
# file's own import lines this build session, not assumed:
#   reasoning_link_probe.py        (this program's own probe script)
#   reasoning_link_stage_minus1.py (this program's own Stage -1 script,
#                                    imports reasoning_link_probe)
#   reasoning_link_gate_enforce.py (THIS script's own Step 3 edge -- invoked
#                                    directly as a subprocess, `$PY
#                                    reasoning_link_gate_enforce.py <json>`,
#                                    not imported; stdlib-only (argparse,
#                                    json, os, subprocess, sys), so it adds
#                                    NO further transitive repo-file edges
#                                    of its own -- verified by direct read
#                                    of its own import lines this build
#                                    session. Also exercised by Step 1's
#                                    Stage -1 suite via
#                                    test_extra_gate_enforcement_negative_test,
#                                    which imports it directly -- a missing/
#                                    stale copy fails at Step 1, before any
#                                    real cell runs, same as every other
#                                    edge below.)
#   grammar_rd.py                  (reasoning_link_probe.py's own import:
#                                    build_entity_pools, DeltaNetRDTaskConfig,
#                                    sample_batch_rd, _iterate_permutation,
#                                    _permutation_graph, self_query_tokens,
#                                    EntityPools, load_gpt2_tokenizer)
#   lm_pretrain_rd.py               (reasoning_link_probe.py's own import:
#                                    DeltaNetLM, FROZEN_BIAS_ARM_MODES,
#                                    apply_frozen_bias_blend,
#                                    build_frozen_bias_table,
#                                    frozen_bias_global_vector,
#                                    _MIN_KERNEL_T, _SAFE_D_STATE)
#   model_rd.py                     (lm_pretrain_rd.py's own import:
#                                    _MIN_KERNEL_T, _SAFE_D_STATE,
#                                    _polar_via_eigh, newton_schulz_orthogonalize;
#                                    ALSO reasoning_link_probe.py's own
#                                    DIRECT import: gram_deviation)
#   rank_utils.py                   (lm_pretrain_rd.py's own import:
#                                    effective_rank, stable_rank; ALSO
#                                    model_rd.py's own import)
#   key_anchoring.py                (lm_pretrain_rd.py's own import:
#                                    random_unit_rows_init, ANCHOR_INIT_SEED;
#                                    ALSO model_rd.py's own import)
#   hard_selectivity_rd.py          (lm_pretrain_rd.py's own import --
#                                    several names, unused by this probe's
#                                    own cells but executed at
#                                    lm_pretrain_rd.py's own module-import
#                                    time regardless)
#   deltanet_core.py                (model_rd.py's own import: apply_state_power)
#   geo3_simulator.py                (key_anchoring.py's own import: newton_schulz)
#   rev7_threshold_derive.py        (key_anchoring.py's own import, "pure
#                                    Python, fla-free")
# PLUS: the real `fla` package (pip-installed on this box; this program's
# own CPU fla-stub is Stage -1-only and never installed when the real
# package import succeeds -- reasoning_link_probe.py's own module
# docstring), the `transformers` package + a cached/reachable 'gpt2'
# tokenizer (grammar_rd.py's own `load_gpt2_tokenizer`, already a
# pre-existing dependency of every sibling script in this dir that builds
# BIND/QUERY episodes), and the checkpoint root
# (/data/deltanet_rd_frozenbias_ckpts/) already present on this box per
# REASONING_LINK_DESIGN.md sec 0 (this script's own Leg-A-only scope --
# unlike reasoning_link_chain.sh, it never reads
# /data/lm_rd_trackc_ckpts/). If ANY of the repo-file list above is missing
# from the scp'd/rsync'd copy on this box, step 1 below fails at Python
# import time, before any GPU time is spent -- verify the deploy's file
# list against this comment BEFORE launching, not after a crash.
#
# Same `set -euo pipefail` discipline every sibling chain in this dir uses
# (mech_stage1_chain.sh, mech_stage2_chain.sh, keyanchor_confirm_chain.sh,
# reasoning_link_chain.sh) -- does NOT rely on `cmd 2>&1 | tee log &&
# next_cmd` masking a crashed command's real exit status behind `tee`'s own
# (successful) one. A PROBE-INVALID gate-enforce exit code stops this
# script immediately, no `cmd | tee` masking (tee never sits between the
# gate-enforce call and the pipeline's own exit status below).
# ---------------------------------------------------------------------------
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/reasoning_link_phase1b

REASONING_LINK_GPU=${REASONING_LINK_GPU:-3}
FROZEN_BIAS_CKPT_ROOT=${FROZEN_BIAS_CKPT_ROOT:-/data/deltanet_rd_frozenbias_ckpts}

# ---------------------------------------------------------------------------
# Step 1 -- Stage -1 self-tests (gate). Pure CPU, mirrors
# reasoning_link_chain.sh's own step 1 exactly (same CPU-stub-forcing
# rationale -- see that script's own comment).
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY reasoning_link_probe.py --mode selftest \
    2>&1 | tee logs/95_reasoning_link_phase1b_stage_minus1.log

# ---------------------------------------------------------------------------
# Step 2 -- four stage0-natural gate cells: {A,B} x {wikitext,openr1}.
# Checkpoint resolution reuses the EXISTING --family leg_a --arm off
# --corpus <corpus> resolver (reasoning_link_probe.py::leg_a_ckpt_path,
# sec 9's own pinned Leg-A control-arm family) -- the wikitext cell resolves
# to frozenbias_lm_off_lam0p00_wikitext-mix-ext_dm256_ds64_L2_s0 (final step
# 20000), per this feature's own BUILD brief.
# ---------------------------------------------------------------------------
TEMPLATES=(A B)
CORPORA=(wikitext-mix-ext openr1-mix-ext)

for template in "${TEMPLATES[@]}"; do
  for corpus in "${CORPORA[@]}"; do
    out="results/reasoning_link_phase1b/stage0_natural_${template}_${corpus}.json"
    log="logs/96_reasoning_link_phase1b_stage0_natural_${template}_${corpus}.log"
    CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
        --mode stage0-natural --template "$template" --family leg_a --arm off --corpus "$corpus" \
        --ckpt-base-dir "$FROZEN_BIAS_CKPT_ROOT" --k 32 --batch-size 16 --device cuda \
        --out "$out" \
        2>&1 | tee "$log"
  done
done

# ---------------------------------------------------------------------------
# Step 3 -- gate enforcement (closes sec 15.2's DISCREPANCY finding: Phase
# 1's own chain computed gate_result_h1_probe_valid but never READ it
# before the grid launched). Both candidates' WIKITEXT (licensing, sec
# 16.1.2) cells are READ via reasoning_link_gate_enforce.py's own REAL exit
# code -- deliberately inside `if` guards (bash's standard `set -e`
# exception), NOT an immediate hard-stop on the first PROBE-INVALID
# candidate, so BOTH candidates get a mechanical verdict before this script
# decides anything (sec 16.1.3 item 1's own framing: "both candidates fail
# identically" is itself the single most informative outcome, and cannot be
# observed if the script aborts after Candidate A alone). The OPENR1 cell
# is read too, but only for the sec 16.1.3 item 2b confound flag -- it never
# gates. The REFUSAL is mechanical and unconditional at the very end: if
# NEITHER candidate's wikitext cell cleared, this script's own `exit 1`
# below is what refuses (no full-grid launch is even attempted after this
# script, sec 16.1.2's own "conditional" framing) -- verified by the
# negative-test fixtures reasoning_link_gate_enforce.py --selftest runs
# (also wired into Step 1's Stage -1 suite).
# ---------------------------------------------------------------------------
ANY_WIKITEXT_CLEARED=0
for template in "${TEMPLATES[@]}"; do
  wikitext_json="results/reasoning_link_phase1b/stage0_natural_${template}_wikitext-mix-ext.json"
  openr1_json="results/reasoning_link_phase1b/stage0_natural_${template}_openr1-mix-ext.json"

  echo "=== Candidate $template: WIKITEXT (licensing) gate check ==="
  if $PY reasoning_link_gate_enforce.py "$wikitext_json"; then
    echo "Candidate $template: WIKITEXT cell CLEARED the Stage-0 gate."
    ANY_WIKITEXT_CLEARED=1
    # sec 16.1.3 item 2b -- check whether openr1 ALSO clears (confound flag,
    # never a second confirmation). Exit code inspected only, never re-gates.
    if $PY reasoning_link_gate_enforce.py "$openr1_json"; then
      echo "CONFOUND FLAG (sec 16.1.3 item 2b): Candidate $template's OPENR1 cell ALSO cleared the "
      echo "gate despite having no natural-template distributional referent -- this is a flag for a "
      echo "confound (something generically easier about Candidate $template, unrelated to "
      echo "distribution match), NOT a second confirmation. Audit before trusting the wikitext pass."
    else
      echo "Candidate $template's OPENR1 cell did NOT clear (expected-null behaved as expected, "
      echo "sec 16.1.2) -- no confound flag."
    fi
  else
    echo "Candidate $template: WIKITEXT cell did NOT clear the Stage-0 gate (PROBE-INVALID)."
  fi
done

if [ "$ANY_WIKITEXT_CLEARED" = "1" ]; then
  touch results/reasoning_link_phase1b/STAGE0_GATE_CLEARED
  echo "REASONING-LINK Phase-1b Stage-0-gate-only: CLEARED for at least one candidate -- the full "
  echo "Phase-1b grid (sec 16.1.2, ~0.4 GPU-h) MAY be considered by a PI/design-owner decision "
  echo "(sec 15.10's own discipline: a harvest does not self-launch the next stage)."
else
  touch results/reasoning_link_phase1b/STAGE0_GATE_REFUSED
  echo "REASONING-LINK Phase-1b Stage-0-gate-only: NEITHER candidate's wikitext cell cleared the "
  echo "gate -- per sec 16.1.3 item 1, this is the single most informative possible null (the "
  echo "failure is NOT attributable to surface form; both a structurally different template AND "
  echo "the original marker template fail the same way). Do NOT launch the full Phase-1b grid on "
  echo "this instrument. This mechanically promotes Path (ii) per sec 16.6's decision tree."
  exit 1
fi
