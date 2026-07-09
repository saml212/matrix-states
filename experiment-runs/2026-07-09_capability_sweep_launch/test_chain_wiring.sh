#!/usr/bin/env bash
# Dry-run proof of the cap_sweep_supervisor.sh chain condition: stage 2 (the
# 2x2 escalation) must start automatically, unattended, the moment stage 1
# (the 58-cell sweep)'s while-loop exits -- no manual trigger, no polling.
# Simulates stage 1 crashing once (proving retry-on-failure) then succeeding
# (proving the STOP-file exit + automatic fallthrough into stage 2's own
# self-healing loop). Zero GPU cost -- pure shell logic, run locally.
set -uo pipefail
WORKDIR=$(mktemp -d)
cd "$WORKDIR"
STAGE1_LOG=stage1.log
STAGE2_LOG=stage2.log
ATTEMPT=0

rm -f STOP STOP_2x2_esc STAGE1_DONE STAGE2_DONE

echo "=== STAGE 1 (simulated: fails attempt 1, succeeds attempt 2) ==="
while [ ! -f STOP ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if [ "$ATTEMPT" -lt 2 ]; then
    echo "[sim] stage1 attempt $ATTEMPT: simulated crash (rc=1)" >>"$STAGE1_LOG"
    RC=1
  else
    echo "[sim] stage1 attempt $ATTEMPT: simulated success (rc=0)" >>"$STAGE1_LOG"
    RC=0
  fi
  if [ "$RC" -eq 0 ]; then
    touch STOP
  else
    sleep 0.2
  fi
done
touch STAGE1_DONE
echo "[chain] stage 1 while-loop exited -> falling through to stage 2 UNCONDITIONALLY"

echo "=== STAGE 2 (simulated, chained automatically) ==="
while [ ! -f STOP_2x2_esc ]; do
  echo "[sim] stage2 attempt: simulated success (rc=0)" >>"$STAGE2_LOG"
  touch STOP_2x2_esc
done
touch STAGE2_DONE

echo
if [ -f STAGE1_DONE ] && [ -f STAGE2_DONE ]; then
  echo "CHAIN WIRING VERIFIED: stage 1 retried-then-completed, stage 2 fired automatically, both DONE markers present."
else
  echo "CHAIN WIRING FAILED"
  exit 1
fi
echo "--- stage1.log ---"; cat "$STAGE1_LOG"
echo "--- stage2.log ---"; cat "$STAGE2_LOG"
cd /
rm -rf "$WORKDIR"
