#!/usr/bin/env bash
# STAGE A0 -- pricing and port gate, BEFORE ANY TRAINING CELL EXISTS.
# NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.0.  Total ~0.5 GPU-h.
#
#   A0.1  build the port + the full B1-B8 smoke, incl. A0.2       ~0.05 GPU-h
#   A0.2  the MIN_KERNEL_T gate at d_state=128 (inside A0.1)      ~0.01 GPU-h
#   A0.3  FOUR solo phase0-timing probes:  392M K=24, 392M K=40,
#         98M K=24, 98M K=40                                       ~0.08 GPU-h
#   A0.4  EIGHT concurrent 392M probes, one per GPU -> R_8          ~0.35 GPU-h
#   A0.5  apply Rules P1-P4 (a0_rules.py)                          0
#
# THE 98M PROBES ARE FATAL-1's FIX AND ARE NOT OPTIONAL.  R1 defined
#   R = phase0-timing(392M) / <realized 98M wall-clock s/step>
# and asserted the two were "directly comparable".  They are not, and this
# repo's own archive measures the gap at 1.5500x
# (experiment-runs/2026-07-17_ncr_gate3_wave1/phase0_timing.json's 0.23075
# against the same-host/same-K/same-batch training cells' 0.148876):
# run_phase0_timing wraps EACH ARM in torch.cuda.synchronize() and sums, while
# the real loop never synchronizes per arm, and the probe's timer starts after
# build_task1_document.  With R = rho x 1.550 the R1 abort fires at EVERY rho in
# the design's own predicted band (3.48 -> 5.39 ... 4.00 -> 6.20) and clears
# only below rho = 3.226 -- i.e. Stage A0 would have HALTED THE WAVE WITH ZERO
# 392M CELLS RUN.  Taking BOTH sides from phase0-timing makes the 1.55x
# inflation cancel exactly, for ~0.02 GPU-h.
#
#   R(K) := phase0(392M, K).mean_s_per_step_both_arms_combined
#         / phase0( 98M, K).mean_s_per_step_both_arms_combined
#   R_8  := 392M 8-way / 392M solo          (already like-for-like; unaffected)
#
# The 98M probes run from the RETAINED KSCALING TREE (sec 3.5 requires both
# trees be kept), the 392M probes from the scaleaxis tree.  A cross-tree
# invocation is the failure that note exists to prevent.
#
# Usage:  run_stage_a0.sh solo        # A0.3, four probes, one GPU at a time
#         run_stage_a0.sh contended   # A0.4, eight concurrent 392M probes
#         run_stage_a0.sh rules       # A0.5
set -u
PY=/home/nvidia/tdenv/bin/python3
SAX=/home/nvidia/ncr_scaleaxis
KSC=/home/nvidia/ncr_kscaling
OUT=/ephemeral/scaleaxis/a0
PROBE_STEPS=${PROBE_STEPS:-60}
WARMUP=${WARMUP:-10}
mkdir -p "$OUT"

probe () {           # probe <tree> <scale> <K> <gpu> <tag>
  local tree="$1" scale="$2" k="$3" gpu="$4" tag="$5"
  local args=(--mode phase0-timing --device cuda --k "$k" --batch-size 32 \
              --warmup-steps "$WARMUP" --probe-steps "$PROBE_STEPS" \
              --target-steps 20000 --lr 3e-4 --seed 0 --out "$OUT/$tag.json")
  # --scale exists ONLY in the scaleaxis tree's runner (port patch R2); the
  # kscaling tree's runner would reject it, which is the tree separation
  # asserting itself.
  if [ "$tree" = "$SAX" ]; then args+=(--scale "$scale"); fi
  ( cd "$tree" && CUDA_VISIBLE_DEVICES="$gpu" NCR_K="$k" NCR_SCALE="$scale" \
      "$PY" ncr_lm_wave1_runner.py "${args[@]}" ) > "$OUT/$tag.log" 2>&1
  echo "  probe $tag exit=$? -> $OUT/$tag.json"
}

case "${1:-}" in
  solo)
    echo "=== A0.3: FOUR solo phase0-timing probes (the like-for-like R) ==="
    probe "$SAX" 392m 24 0 a0_solo_392m_K24
    probe "$SAX" 392m 40 0 a0_solo_392m_K40
    probe "$KSC" 98m  24 0 a0_solo_98m_K24
    probe "$KSC" 98m  40 0 a0_solo_98m_K40
    ;;
  contended)
    echo "=== A0.4: EIGHT concurrent 392M probes (homogeneous) -> R_8 ==="
    # Homogeneous by construction, one per GPU, all K=24 so R_8 is a pure
    # contention ratio against the K=24 solo probe and nothing else moves.
    for g in 0 1 2 3 4 5 6 7; do
      tmux new-session -d -s a0c_g$g \
        "bash $0 _one_contended $g"
    done
    echo "  8 sessions launched; poll with: tmux ls | grep a0c_"
    ;;
  _one_contended)
    probe "$SAX" 392m 24 "$2" "a0_8way_392m_K24_g$2"
    ;;
  rules)
    echo "=== A0.5: Rules P1-P4 ==="
    "$PY" "$SAX/a0_rules.py" --a0-dir "$OUT" \
        --gates "$SAX/gates_K24.json" --out "$OUT/A0_RULES.json"
    ;;
  *)
    echo "usage: $0 {solo|contended|rules}"; exit 2;;
esac
