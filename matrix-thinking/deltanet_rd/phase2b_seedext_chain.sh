#!/bin/bash
# phase2b_seedext_chain.sh -- REASONING_LINK_DESIGN.md sec 16.19 (Rev 3, DESIGN-CLEARED-FOR-BUILD
# after the round-4 verify) Phase-2b SEED-EXTENSION familiarization+eval layer: Leg-A completion
# gate -> Stage -1 (CPU, incl. the seedext suite) -> real-kernel smoke x2 arms (off/per_token
# ONLY -- Option B trains no global cells) -> sha256 gates for every reused artifact -> seedext
# eval timing pilot -> 9 NEW OFF familiarization cells -> OFF-eval cache EXTENSION + the n=12
# wikitext-only FLOOR_PIN re-pin (BLIND: pinned before any per_token L_query is read, sec
# 16.16.6's sequencing, sec 16.19.9 item 4) -> wikitext floor gate -> 9 NEW per_token cells ->
# analyze_corpus_seedext harvest (the wave-specific driver, NEVER production analyze_corpus).
#
# FORKED from phase2b_chain.sh (sec 16.19.7's own registration; the original stays the historical
# n=3-build record, untouched) -- Stage -1 / smoke / sha256-manifest / budget machinery reused
# verbatim; the trajectory-analysis step drives analyze_corpus_seedext instead.
#
# NOT RUN by this build session -- build + CPU-verify only, no GPU work performed here.
#
# BUDGET (sec 16.19.6): raw ~=6.65 GPU-h total wave (Leg-A slice included, launched by
# frozen_bias_seedext_chain.sh); registered debug-tax bracket ~=33.3-66.5 GPU-h, ENFORCED ceiling
# 66.5 GPU-h -- budget_check() below aborts mechanically past it (sec 16.5 Constraint 1), so an
# overrun in THIS (familiarization/eval) slice is attributable separately from the Leg-A slice's
# own gates.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/phase2/ckpts results/phase2/gates

PHASE2_GPUS=${PHASE2_GPUS:-"0,1"}
N_GPUS=$(echo "$PHASE2_GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
IFS=',' read -ra GPU_LIST <<< "$PHASE2_GPUS"
# The 18 NEW Leg-A checkpoints live under the SAME /data tree the seedext pretraining chain wrote
# (sec 16.19.7.1's FROZENBIAS_SEEDEXT_CKPT_BASE default) -- additive seed subdirs.
FROZEN_BIAS_CKPT_ROOT=${FROZEN_BIAS_CKPT_ROOT:-/data/deltanet_rd_frozenbias_ckpts}
# The Leg-A layer's own out-dir + completion sentinel (frozen_bias_seedext_chain.sh writes it).
FROZENBIAS_SEEDEXT_OUT_DIR=${FROZENBIAS_SEEDEXT_OUT_DIR:-results/frozen_bias_lm_seedext}
CKPT_DIR=results/phase2/ckpts
STEPS=5000
CKPT_STEPS="250 500 1000 2500 5000"
CORPUS=wikitext-mix-ext                 # sec 16.19.4 Option B: ONE corpus
SEEDS=(3 4 5 6 7 8 9 10 11)            # the 9 NEW seeds
FROZEN_BIAS_LAMBDA=0.58
OFF_CACHE=results/phase2/off_lquery_cache-Phase2b.json
FLOOR_PINNED=results/phase2/FLOOR_PINNED-Phase2b.json
FLOOR_PINNED_N12=results/phase2/FLOOR_PINNED-Phase2b-n12-wikitext.json
ARCHIVED_TRAJ=results/phase2/trajectory_wikitext-mix-ext_phase2b.json
# NEW sha256 manifest for the 18 NEW Leg-A .pt files -- pinned ONCE on the box against the
# freshly-trained originals (a deploy-step obligation, sec 16.19.7's sha256 bullet; the repo's
# committed copy under experiment-runs/ is the source of truth, same convention as the original
# 30-checkpoint manifest).
SHA256_MANIFEST_SEEDEXT=results/phase2/gates/phase2b_seedext_lega_manifest.sha256
# Reused-ARCHIVE integrity pins (sec 16.19.5 item 5's two read-only source artifacts + the
# original floor pin the cache is verified against pre-extension). Hashed from the repo's own
# committed archive copies (experiment-runs/2026-07-08_phase2b/results/) at build time -- the
# box copies MUST match them exactly before this wave reads a single archived float.
ARCHIVED_TRAJ_SHA256=8f6313996f2ce1ec66b8458315c0d192b5d01f4ff57961c0437031c5686e9d02
ARCHIVED_OFF_CACHE_SHA256=56958f48508ebf338db24641c2ce07f14504c169628f245018013c5a3642ea8e
ARCHIVED_FLOOR_SHA256=e348699b4ddb17dbce68b77f66eeba45be41ec66ff228e65336f516c1c4b2bf6

BUDGET_CEILING_GPU_H=66.5   # sec 16.19.6's registered wave ceiling (this slice's own guard)

budget_check() {
  local elapsed_s=$SECONDS
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print(($elapsed_s * $N_GPUS) / 3600.0)")
  echo "[budget] elapsed=${elapsed_s}s  n_gpus=$N_GPUS  projected_gpu_h=${projected_gpu_h}  ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h (${projected_gpu_h}) exceeds the registered ceiling (${BUDGET_CEILING_GPU_H}, sec 16.19.6) -- halting mechanically." >&2
    touch results/phase2/PHASE2B_SEEDEXT_BUDGET_ABORTED
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# 2-way parallel cell launcher -- forked verbatim from phase2b_chain.sh's own run_cells_2way
# (same resume-safety, per-cell GPU pinning, fail-fast-on-either-half); the ONLY change is the
# seedext log prefix. Reused byte-for-byte rather than sourced (phase2b_chain.sh executes its own
# chain top-level immediately, same reason as its own fork note).
# ---------------------------------------------------------------------------
run_cells_2way() {
  local cells=("$@")
  local pending=()
  local arm corpus seed out
  for c in "${cells[@]}"; do
    IFS=':' read -r arm corpus seed <<< "$c"
    out="results/phase2/${arm}_${corpus}_s${seed}.json"
    if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('steps_completed')==$STEPS else 1)" 2>/dev/null; then
      echo "SKIP (already complete): $out"
      continue
    fi
    pending+=("$c")
  done

  local i=0
  while [ $i -lt ${#pending[@]} ]; do
    local pids=()
    local slot idx gpu log lam_tag init_ckpt
    for slot in 0 1; do
      idx=$((i + slot))
      [ $idx -ge ${#pending[@]} ] && break
      IFS=':' read -r arm corpus seed <<< "${pending[$idx]}"
      out="results/phase2/${arm}_${corpus}_s${seed}.json"
      log="logs/70_phase2b_seedext_${arm}_${corpus}_s${seed}.log"
      gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
      if [ "$arm" = "off" ]; then
        init_ckpt="$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_off_lam0p00_${corpus}_dm256_ds64_L2_s${seed}/lmC_${corpus}_dm256_ds64_L2_s${seed}_step20000.pt"
      else
        lam_tag="lam$($PY -c "print(f'{$FROZEN_BIAS_LAMBDA:.2f}'.replace('.', 'p'))")"
        init_ckpt="$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_${arm}_${lam_tag}_${corpus}_dm256_ds64_L2_s${seed}/lmC_${corpus}_dm256_ds64_L2_s${seed}_step20000.pt"
      fi
      echo "LAUNCH (gpu=$gpu): arm=$arm corpus=$corpus seed=$seed -> $out"
      ( CUDA_VISIBLE_DEVICES="$gpu" $PY phase2_familiarization_train.py \
            --init-checkpoint "$init_ckpt" --arm "$arm" --corpus "$corpus" --seed "$seed" \
            --steps $STEPS --ckpt-steps $CKPT_STEPS --ckpt-dir "$CKPT_DIR" --out "$out" \
            --device cuda 2>&1 | tee "$log" ) &
      pids+=($!)
    done
    local fail=0 pid
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
      echo "ERROR: at least one cell in this GPU-pair FAILED -- halting mechanically (see per-cell logs above)." >&2
      exit 1
    fi
    i=$((i + 2))
  done
}

# ---------------------------------------------------------------------------
# Step 0 -- Leg-A completion gate (sec 16.19.7: the familiarization layer is sequenced AFTER the
# pretraining layer, which produces this chain's own required inputs): the seedext pretraining
# chain's DONE sentinel must exist AND every one of the 18 new terminal checkpoints must be
# present on disk. Mechanical, loud refusal.
# ---------------------------------------------------------------------------
if [ ! -f "$FROZENBIAS_SEEDEXT_OUT_DIR/FROZENBIAS_SEEDEXT_CHAIN_DONE" ]; then
  echo "REFUSE: Leg-A completion sentinel $FROZENBIAS_SEEDEXT_OUT_DIR/FROZENBIAS_SEEDEXT_CHAIN_DONE not found -- frozen_bias_seedext_chain.sh (the pretraining layer, sec 16.19.7.1) must complete first." >&2
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  for armdir in "off_lam0p00" "per_token_lam0p58"; do
    ck="$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_${armdir}_${CORPUS}_dm256_ds64_L2_s${seed}/lmC_${CORPUS}_dm256_ds64_L2_s${seed}_step20000.pt"
    if [ ! -f "$ck" ]; then
      echo "REFUSE: missing new Leg-A terminal checkpoint $ck -- the 18-cell pretraining wave is incomplete." >&2
      exit 1
    fi
  done
done
echo "Leg-A completion gate: sentinel + all 18 terminal checkpoints present."

# ---------------------------------------------------------------------------
# Step 1 -- Stage -1 self-tests (CPU, gate): the existing suites verbatim PLUS the seedext suite
# (delta_ci_n/scipy cross-check, widened-keyspace enumerations, loader/guard negatives, the
# four-bucket behavioral test, manifest disjointness, band/timing gate fixtures).
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY reasoning_link_probe.py --mode selftest \
    2>&1 | tee logs/71_seedext_reasoning_link_stage_minus1.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY phase2_stage_minus1.py \
    2>&1 | tee logs/72_seedext_phase2_stage_minus1.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY phase2b_seedext_stage_minus1.py \
    2>&1 | tee logs/73_seedext_stage_minus1.log

# ---------------------------------------------------------------------------
# Step 1.5 -- REAL-KERNEL smoke gate, TWO arms (sec 16.19.4 Option B: off + per_token ONLY --
# this wave trains no global cells, so no global smoke is required or run).
# ---------------------------------------------------------------------------
for smoke_arm in off per_token; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2_smoke_gpu.py --arm "$smoke_arm" \
      --corpus "$CORPUS" \
      2>&1 | tee "logs/74_seedext_smoke_gpu_${smoke_arm}.log"
done
budget_check

# ---------------------------------------------------------------------------
# Step 2 -- sha256 gates for EVERY reused artifact (sec 16.19.7's sha256 bullet + sec 16.19.5
# item 5's two read-only archived sources):
#   (a) the 18 NEW Leg-A .pt files vs the seedext manifest (belt: sha256sum -c; suspenders:
#       phase2b_ckpt_reuse_gate.py's independent recomputation);
#   (b) the archived OFF-eval cache, original floor pin, and archived trajectory JSON vs the
#       build-time pins hashed from the repo's committed archive copies.
# ---------------------------------------------------------------------------
SHA256_MANIFEST_SEEDEXT_ABS="$(pwd)/$SHA256_MANIFEST_SEEDEXT"
( cd "$FROZEN_BIAS_CKPT_ROOT" && sha256sum -c "$SHA256_MANIFEST_SEEDEXT_ABS" ) 2>&1 | tee logs/75_seedext_sha256_belt.log \
  || { echo "REFUSE: bash-level sha256sum -c FAILED on the 18 new Leg-A checkpoints -- halting mechanically." >&2; touch results/phase2/PHASE2B_SEEDEXT_SHA256_GATE_REFUSED; exit 1; }
if ! $PY phase2b_ckpt_reuse_gate.py --ckpt-dir "$FROZEN_BIAS_CKPT_ROOT" --manifest "$SHA256_MANIFEST_SEEDEXT" \
    2>&1 | tee logs/76_seedext_sha256_suspenders.log; then
  echo "REFUSE: Python-level sha256 reuse gate FAILED on the 18 new Leg-A checkpoints -- halting mechanically." >&2
  touch results/phase2/PHASE2B_SEEDEXT_SHA256_GATE_REFUSED
  exit 1
fi
for pair in "$FLOOR_PINNED:$ARCHIVED_FLOOR_SHA256" "$ARCHIVED_TRAJ:$ARCHIVED_TRAJ_SHA256"; do
  IFS=':' read -r fpath fhash <<< "$pair"
  echo "$fhash  $fpath" | sha256sum -c - 2>&1 | tee -a logs/77_seedext_archived_artifact_sha256.log \
    || { echo "REFUSE: archived artifact $fpath does not match its build-time pin -- halting mechanically (never read a tampered/stale archive)." >&2; touch results/phase2/PHASE2B_SEEDEXT_SHA256_GATE_REFUSED; exit 1; }
done
# The OFF cache's own check is RESUME-AWARE: pre-extension it must match the archived pin
# byte-exactly; post-extension (a resumed run after step 5) its integrity is carried by the n12
# pin instead (validate_floor_pinned re-hashes it) -- either way a tampered cache refuses loudly.
if $PY -c "
import sys
sys.path.insert(0, '.')
import phase2b_off_cache as poc
sys.exit(0 if poc.validate_floor_pinned('$FLOOR_PINNED_N12') else 1)
" 2>/dev/null; then
  echo "OFF cache integrity: carried by the n12 pin ($FLOOR_PINNED_N12 validates the extended cache) -- resumed post-extension run." | tee -a logs/77_seedext_archived_artifact_sha256.log
else
  echo "$ARCHIVED_OFF_CACHE_SHA256  $OFF_CACHE" | sha256sum -c - 2>&1 | tee -a logs/77_seedext_archived_artifact_sha256.log \
    || { echo "REFUSE: $OFF_CACHE matches NEITHER the archived pre-extension pin NOR a validating n12 pin -- halting mechanically (never read a tampered/stale cache)." >&2; touch results/phase2/PHASE2B_SEEDEXT_SHA256_GATE_REFUSED; exit 1; }
fi
touch results/phase2/PHASE2B_SEEDEXT_SHA256_GATE_CLEARED

# ---------------------------------------------------------------------------
# Step 3 -- seedext eval timing pilot (sec 16.19.7 m3): ONE real pass on an ARCHIVED wikitext OFF
# familiarization checkpoint, projected over the 360 seedext passes against the 66.5 GPU-h
# ceiling. Must NOT assume the prior wave's warm rate (cold-Triton precedent).
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2b_off_cache.py --ckpt-dir "$CKPT_DIR" \
    --seedext-time-pilot --device cuda 2>&1 | tee logs/78_seedext_timing_pilot.log
budget_check

# ---------------------------------------------------------------------------
# Step 4 -- the 9 NEW OFF familiarization cells FIRST (blind-pin sequencing, sec 16.19.9 item 4:
# the n=12 OFF pin must be written BEFORE any per_token cell's own L_query could be read).
# ---------------------------------------------------------------------------
off_cells=()
for seed in "${SEEDS[@]}"; do off_cells+=("off:${CORPUS}:${seed}"); done
run_cells_2way "${off_cells[@]}"
budget_check

# ---------------------------------------------------------------------------
# Step 5 -- OFF-eval cache EXTENSION (180 additive keys, existing values byte-identical, verified
# against the ORIGINAL pin pre-extension) + FLOOR_PINNED-Phase2b-n12-wikitext.json re-pin over
# the 12 pooled wikitext OFF seeds (3 archived-via-cache + 9 new). openr1's pin untouched.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2b_off_cache.py --ckpt-dir "$CKPT_DIR" \
    --seedext-extend --out-cache "$OFF_CACHE" --original-floor "$FLOOR_PINNED" \
    --out-floor-n12 "$FLOOR_PINNED_N12" --device cuda \
    2>&1 | tee logs/79_seedext_cache_extend_repin.log
budget_check

# ---------------------------------------------------------------------------
# Step 6 -- wikitext floor gate at the NEW n=12 pin (mirrors the original chain's step 5 for the
# ONE in-scope corpus): FAMILIARIZATION-NULL refuses the per_token launch (sec 16.16.6's "at
# least one clears" convention, applied to this wave's single corpus). The n=3-vs-n=12 bucket
# comparison is a MANDATORY harvest disclosure (sec 16.19.7) -- both verdicts logged here.
# ---------------------------------------------------------------------------
ratio_n12=$($PY -c "
import json
with open('$FLOOR_PINNED_N12') as f:
    doc = json.load(f)
print(doc['floor_by_corpus']['$CORPUS']['pooled_ratio'])
")
floor_pin_n12=$($PY -c "
import json
with open('$FLOOR_PINNED_N12') as f:
    doc = json.load(f)
print(doc['floor_by_corpus']['$CORPUS']['floor_pin'])
")
echo "[floor-disclosure] wikitext n=3 pin (historical, FLOOR_PINNED-Phase2b.json) vs n=12 pin (this wave):"
$PY -c "
import json
with open('$FLOOR_PINNED') as f:
    old = json.load(f)['floor_by_corpus']['$CORPUS']
print(f\"  n=3:  pooled_ratio={old['pooled_ratio']:.4f} floor_pin={old['floor_pin']:.4f}\")
print(f\"  n=12: pooled_ratio=$ratio_n12 floor_pin=$floor_pin_n12\")
"
if $PY phase2b_floor_gate_enforce.py --ratio "$ratio_n12" --floor-pin "$floor_pin_n12" --corpus "$CORPUS"; then
  echo "OFF-floor gate (n=12 pin): corpus=$CORPUS NOT FAMILIARIZATION-NULL -- per_token launch proceeds."
else
  touch results/phase2/PHASE2B_SEEDEXT_FLOOR_GATE_REFUSED
  echo "REFUSE: $CORPUS landed FAMILIARIZATION-NULL at the n=12 OFF-floor gate -- the 9-cell" >&2
  echo "per_token launch is NOT licensed on this instrument (sec 16.5 Constraint 1; mirrors the" >&2
  echo "original chain's own refusal convention, scoped to this wave's single corpus)." >&2
  exit 1
fi
budget_check

# ---------------------------------------------------------------------------
# Step 7 -- the 9 NEW per_token familiarization cells (AFTER the blind pin above).
# ---------------------------------------------------------------------------
pt_cells=()
for seed in "${SEEDS[@]}"; do pt_cells+=("per_token:${CORPUS}:${seed}"); done
run_cells_2way "${pt_cells[@]}"
budget_check

# ---------------------------------------------------------------------------
# Step 8 -- the n=12 harvest: analyze_corpus_seedext (the wave-specific driver -- installs the
# whole-harvest-runtime live-eval guard at entry, sources seeds 0-2 via the archived-values
# loader/cache ONLY, seeds 3-11 via the guarded live path, applies the batch-effect pre-pooling
# gate, and classifies the anchor cell into the sec 16.19.8 MECE partition). --floor-pinned is
# the WAVE'S OWN n12 pin (its off_cache_sha256 records the EXTENDED cache).
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$PHASE2_GPUS $PY phase2_trajectory_analysis.py --seedext \
    --corpus "$CORPUS" --ckpt-dir "$CKPT_DIR" --off-cache "$OFF_CACHE" \
    --floor-pinned "$FLOOR_PINNED_N12" --archived-trajectory "$ARCHIVED_TRAJ" \
    --out results/phase2/trajectory_seedext_wikitext_n12.json \
    2>&1 | tee logs/80_seedext_harvest.log

# ---------------------------------------------------------------------------
# Step 9 -- final summary JSON.
# ---------------------------------------------------------------------------
$PY - <<PYEOF
import json, time
with open("results/phase2/trajectory_seedext_wikitext_n12.json") as f:
    d = json.load(f)
summary = {"design_ref": "REASONING_LINK_DESIGN.md sec 16.19 (Rev 3, DESIGN-CLEARED-FOR-BUILD, round-4 verify)",
           "program": "REASONING-LINK Phase 2b SEED EXTENSION (n=12 targeted transient confirmation)",
           "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "elapsed_s": $SECONDS, "n_gpus": $N_GPUS,
           "projected_gpu_h": ($SECONDS * $N_GPUS) / 3600.0,
           "corpus": d["corpus"], "arm": d["arm"], "anchor": d["anchor"]}
with open("results/phase2/PHASE2B_SEEDEXT_SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

touch results/phase2/PHASE2B_SEEDEXT_CHAIN_DONE
echo "PHASE-2B SEEDEXT CHAIN COMPLETE. See results/phase2/PHASE2B_SEEDEXT_SUMMARY.json"
