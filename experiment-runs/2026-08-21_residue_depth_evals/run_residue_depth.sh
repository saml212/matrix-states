#!/usr/bin/env bash
# Driver for the 2026-08-21 #2 pre-registration (EXPERIMENT_LOG.md, commit ee602df):
#   A = residue-space completion (15 unused residues, P1b exact-write)
#   B = physical-depth robustness at fixed residue 13 (h up to 4093)
# Eval-only, existing checkpoints. GPUs 2-7 ONLY (0-1 hold the PI's vLLM servers).
#
# Fail-loud contract (the silent-zero-scoring bug class has recurred 6 times):
# a per-arm nonzero exit is reported AND the trailing self-check independently
# re-reads every output JSON, so a missing arm can never be mistaken for a run.
set -u
cd /home/nvidia/ncr_writecond

GPU="${RD_GPU:-6}"
case "$GPU" in 2|3|4|5|6|7) ;; *) echo "REFUSING: GPU $GPU is not in the allowed set 2-7"; exit 1;; esac
export CUDA_VISIBLE_DEVICES="$GPU"

PY=/home/nvidia/tdenv/bin/python3
CKROOT=/ephemeral/reseed_ckpts
OUTDIR=/home/nvidia/ncr_writecond/results
TAGS="primary_s1 compA_s1 compB_s1"

echo "=== residue/depth eval wave on GPU $GPU ($(date -u +%FT%TZ)) ==="
rc_any=0
for tag in $TAGS; do
  NAME="mob_g3b31_${tag}"
  CK="$CKROOT/${NAME}_ckpts/${NAME}.ckpt.pt"
  if [ ! -f "$CK" ]; then
    echo "!!! MISSING-CKPT ${tag}: $CK does not exist -- NOT SCORED"
    rc_any=1
    continue
  fi
  echo "=== scoring ${tag} ($CK) ==="
  "$PY" residue_depth_eval.py --ckpt "$CK" --tag "$tag" --outdir "$OUTDIR"
  rc=$?
  [ "$rc" -ne 0 ] && { echo "!!! FAILED ${tag} (exit $rc)"; rc_any=1; }
done

# --- independent self-check over the written artifacts --------------------
echo "=== self-check ==="
"$PY" - "$OUTDIR" $TAGS <<'PYEOF'
import json, math, os, sys
outdir, tags = sys.argv[1], sys.argv[2:]
HOPS_A = (4,6,7,8,9,10,11,14,15,17,18,19,21,22,23)
HOPS_B = (13,61,253,1021,4093)
bad = []
for tag in tags:
    p = os.path.join(outdir, f"residue_depth_{tag}.json")
    if not os.path.exists(p):
        bad.append(f"{tag}: NO OUTPUT FILE EVER PRODUCED ({p})"); continue
    try:
        d = json.load(open(p))
    except Exception as e:
        bad.append(f"{tag}: output does not parse ({e!r})"); continue
    if d.get("ckpt_step") != 20000:
        bad.append(f"{tag}: ckpt_step={d.get('ckpt_step')!r}, pre-registration admits 20000 only")
    if d.get("self_check") != "PASS":
        bad.append(f"{tag}: in-script self_check={d.get('self_check')!r} {d.get('self_check_defects')}")
    for exp, hops in (("experiment_A", HOPS_A), ("experiment_B", HOPS_B)):
        for regime in ("P1b", "P0"):
            res = d.get(exp, {}).get(regime, {}).get("result", {})
            for h in hops:
                v = res.get(f"h={h}", {}).get("retrieval24_acc")
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    bad.append(f"{tag}/{exp}/{regime}/h={h}: retrieval24_acc={v!r}")
if bad:
    print(f"SELF-CHECK FAIL: {len(bad)} defect(s)")
    for b in bad: print("   -", b)
    sys.exit(1)
print("SELF-CHECK PASS: every expected arm x experiment x regime x hop produced a finite metric")
PYEOF
sc=$?
[ "$sc" -ne 0 ] && rc_any=1

echo "EXIT_STATUS=$rc_any"
echo RESIDUE_DEPTH_WAVE_DONE
exit "$rc_any"
