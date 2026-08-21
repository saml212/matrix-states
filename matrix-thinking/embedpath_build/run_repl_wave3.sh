#!/usr/bin/env bash
# run_repl_wave3.sh -- D-F4 (NCR_EMBED_PATH_DESIGN.md DRAFT-R1 sec R1.5): adds
# compE / compE_adapter to the tag loop, a per-tag freeze map (both
# trainable-adapter targets => FZ=""), the /ephemeral/embed_path_ckpts root,
# the correct seed ranges, loud MISSING-CKPT, and re-score-not-skip (a stale
# eval record surviving a newer checkpoint is exactly what caused the
# #12/#13 stale-eval incident). Written against the box's ACTUAL current
# run_repl_wave2.sh (md5 dfba70bccd318074d95dbe698c40c77b, re-verified by the
# build agent 2026-08-21 -- unchanged since Rev-1's own re-verification),
# not any stale quoted snapshot.
set -u
cd /home/nvidia/ncr_writecond
export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-2}
OLD=/home/nvidia/ncr_g3b31_contrastive/results
MID=/ephemeral/reseed_ckpts
NEW=/ephemeral/embed_path_ckpts
declare -A SEEDS=( [compA]="1 24" [compB]="1 24" [compD]="1 24" [primary]="1 24" \
                    [compE]="1 8" [compE_adapter]="9 12" )
declare -A FZ=( [compA]="freeze" [compB]="" [compD]="" [primary]="freeze" \
                [compE]="" [compE_adapter]="" )
scored=0; missing=0; rescored=0
for tag in "${!SEEDS[@]}"; do
  read -r lo hi <<< "${SEEDS[$tag]}"
  PREFIX="mob_gembed_${tag}"; [ "$tag" = "compA" -o "$tag" = "compB" -o "$tag" = "compD" -o "$tag" = "primary" ] && PREFIX="mob_g3b31_${tag}"
  for s in $(seq "$lo" "$hi"); do
    NAME="${PREFIX}_s${s}"
    CK=""
    for cand in "$NEW/${NAME}_ckpts/${NAME}.ckpt.pt" "$MID/${NAME}_ckpts/${NAME}.ckpt.pt" "$OLD/${NAME}_ckpts/${NAME}.ckpt.pt"; do
      [ -f "$cand" ] && { CK="$cand"; break; }
    done
    RES="$OLD/${NAME}.json"
    OUT="/home/nvidia/ncr_writecond/results/writecond_premise_REPL_${tag}_s${s}.json"
    if [ -z "$CK" ]; then
      [ -f "$RES" ] && { echo "MISSING-CKPT ${tag}_s${s} (results JSON exists but no checkpoint in any of NEW/MID/OLD)"; missing=$((missing+1)); }
      continue
    fi
    if [ -f "$OUT" ] && [ "$CK" -ot "$OUT" ]; then
      continue                                    # ckpt not newer than the eval -- genuinely already scored
    fi
    [ -f "$OUT" ] && rescored=$((rescored+1))
    echo "=== scoring ${tag}_s${s} ($CK) ==="
    /home/nvidia/tdenv/bin/python3 pbe_repl "$CK" "${tag}_s${s}" "${FZ[$tag]}" && scored=$((scored+1)) || echo "FAILED ${tag}_s${s}"
  done
done
echo "SCORED=$scored RESCORED=$rescored MISSING=$missing"
# self-check (D-F4's own ask): FAIL LOUDLY if ANY expected arm/seed produced no output at all.
fail=0
for tag in compE compE_adapter; do          # G2: self-check scoped to this wave's arms (BUILD_AUDIT_R1)
  read -r lo hi <<< "${SEEDS[$tag]}"
  for s in $(seq "$lo" "$hi"); do
    OUT="/home/nvidia/ncr_writecond/results/writecond_premise_REPL_${tag}_s${s}.json"
    [ -f "$OUT" ] || { echo "SELF-CHECK FAIL: no output ever produced for ${tag}_s${s}"; fail=1; }
  done
done
[ "$fail" -eq 0 ] && echo "SELF-CHECK PASS: every expected arm/seed has an output"
echo REPL_WAVE3_DONE
