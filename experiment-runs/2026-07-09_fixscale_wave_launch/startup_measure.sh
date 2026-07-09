#!/bin/bash
# Measures startup-to-step-1000 cum_rate for a given scale, mirroring
# fixscale_wave.py's _budget_watchdog formula: cum_rate = elapsed_since_launch / last_logged_step.
set -u
SCALE="$1"   # 98m or 392m
cd /home/nvidia/chapter2/deltanet_rd
source /home/nvidia/tdenv/bin/activate

if [ "$SCALE" = "98m" ]; then
  DM=768; DS=64; NL=12
elif [ "$SCALE" = "392m" ]; then
  DM=1536; DS=128; NL=16
else
  echo "bad scale"; exit 1
fi

LOG=/data/fixscale_smoke_test/startup_${SCALE}.log
rm -f "$LOG"
T_LAUNCH=$(date +%s.%N)
echo "T_LAUNCH=$T_LAUNCH ($(date -u -d @${T_LAUNCH%.*} +%Y-%m-%dT%H:%M:%SZ))"
CUDA_VISIBLE_DEVICES=7 python lm_pretrain_rd.py \
  --corpus openr1-mix-ext --data-dir /data/deltanet_rd_data \
  --d-model $DM --d-state $DS --n-layers $NL --seq-len 512 --batch-size 32 \
  --steps 1000 --ckpt-every 1000 --seed 99 --internal-timeout 3600 \
  --frozen-bias-arm off --frozen-bias-lambda 0.58 \
  --out /data/fixscale_smoke_test/startup_${SCALE}.json \
  --ckpt-dir /data/fixscale_smoke_test/startup_${SCALE}_ckpt \
  > "$LOG" 2>&1 &
PID=$!

LAST_STEP=""
while kill -0 $PID 2>/dev/null; do
  sleep 3
  STEP=$(grep -oE '^  step +[0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  if [ -n "$STEP" ] && [ "$STEP" != "$LAST_STEP" ]; then
    NOW=$(date +%s.%N)
    ELAPSED=$(echo "$NOW - $T_LAUNCH" | bc)
    if [ "$STEP" -gt 0 ]; then
      CUMRATE=$(echo "scale=4; $ELAPSED / $STEP" | bc)
    else
      CUMRATE="n/a"
    fi
    echo "t=$NOW elapsed=${ELAPSED}s step=$STEP cum_rate=${CUMRATE}s/step"
    LAST_STEP="$STEP"
  fi
done
wait $PID
T_END=$(date +%s.%N)
TOTAL=$(echo "$T_END - $T_LAUNCH" | bc)
echo "PROCESS EXITED. total_wall=${TOTAL}s"
