#!/usr/bin/env bash
# Deploys the queue system (runner + docs + generated job specs) from this
# repo checkout to the box, then md5-verifies every deployed file landed
# byte-identical. Run LOCALLY (this is a scp wrapper, not a box script).
#
# Usage: bash deploy.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
NCR_LOCAL="$REPO_ROOT/matrix-thinking/ncr"
BOX="youthful-indigo-turkey"

echo "== ensuring box-side directories exist =="
ssh "$BOX" 'mkdir -p ~/queue/pending ~/queue/claimed ~/queue/completed ~/queue/failed ~/queue/logs'

echo "== copying runner + launcher + docs =="
scp -q "$HERE/queue_worker.sh" "$HERE/launch_workers.sh" "$HERE/QUEUE_README.md" "$BOX:~/queue/"

echo "== copying the K-ladder extension (ncr_task.py / ncr_earlyln_scale.py) --"
echo "   REQUIRED: Lane A/C job specs pass --K 20..256, which only exist in"
echo "   THIS session's additive GRIDS/GRID_SHAPES edit (audit FATAL F1 fix)."
scp -q "$NCR_LOCAL/ncr_task.py" "$NCR_LOCAL/ncr_earlyln_scale.py" "$BOX:~/ncr/"

echo "== copying job specs (jobs/pending/*.json) =="
scp -q "$HERE"/jobs/pending/*.json "$BOX:~/queue/pending/"

echo "== md5 verify: runner + launcher + docs =="
LOCAL_SUM=$(cd "$HERE" && (md5 -q queue_worker.sh launch_workers.sh QUEUE_README.md 2>/dev/null || md5sum queue_worker.sh launch_workers.sh QUEUE_README.md | awk '{print $1}'))
REMOTE_SUM=$(ssh "$BOX" 'cd ~/queue && (md5sum queue_worker.sh launch_workers.sh QUEUE_README.md | awk "{print \$1}")')
if [ "$(echo "$LOCAL_SUM" | tr -d '\n')" != "$(echo "$REMOTE_SUM" | tr -d '\n')" ]; then
  echo "MD5 MISMATCH on runner/docs -- deploy FAILED, do not launch." >&2
  echo "local:  $LOCAL_SUM" >&2
  echo "remote: $REMOTE_SUM" >&2
  exit 1
fi
echo "runner/launcher/docs md5-verified clean."

echo "== md5 verify: ncr_task.py / ncr_earlyln_scale.py =="
LOCAL_NCR_SUM=$(cd "$NCR_LOCAL" && (md5 -q ncr_task.py ncr_earlyln_scale.py 2>/dev/null || md5sum ncr_task.py ncr_earlyln_scale.py | awk '{print $1}'))
REMOTE_NCR_SUM=$(ssh "$BOX" 'cd ~/ncr && (md5sum ncr_task.py ncr_earlyln_scale.py | awk "{print \$1}")')
if [ "$(echo "$LOCAL_NCR_SUM" | tr -d '\n')" != "$(echo "$REMOTE_NCR_SUM" | tr -d '\n')" ]; then
  echo "MD5 MISMATCH on ncr_task.py/ncr_earlyln_scale.py -- deploy FAILED, do not launch." >&2
  echo "local:  $LOCAL_NCR_SUM" >&2
  echo "remote: $REMOTE_NCR_SUM" >&2
  exit 1
fi
echo "ncr_task.py/ncr_earlyln_scale.py md5-verified clean."

echo "== md5 verify: job specs (count + per-file digest) =="
LOCAL_N=$(ls "$HERE"/jobs/pending/*.json | wc -l | tr -d ' ')
REMOTE_N=$(ssh "$BOX" 'ls ~/queue/pending/*.json 2>/dev/null | wc -l' | tr -d ' ')
LOCAL_JOBSUM=$(cd "$HERE/jobs/pending" && (md5 -q *.json 2>/dev/null || md5sum *.json | awk '{print $1}') | sort | md5 -q 2>/dev/null || (cd "$HERE/jobs/pending" && md5sum *.json | awk '{print $1}' | sort | md5sum | awk '{print $1}'))
REMOTE_JOBSUM=$(ssh "$BOX" 'cd ~/queue/pending && md5sum *.json | awk "{print \$1}" | sort | md5sum | awk "{print \$1}"')
echo "local job count=$LOCAL_N remote job count=$REMOTE_N"
if [ "$LOCAL_N" != "$REMOTE_N" ]; then
  echo "JOB SPEC COUNT MISMATCH -- deploy FAILED, do not launch." >&2
  exit 1
fi
if [ "$LOCAL_JOBSUM" != "$REMOTE_JOBSUM" ]; then
  echo "JOB SPEC CONTENT MISMATCH (aggregate md5 differs) -- deploy FAILED, do not launch." >&2
  exit 1
fi
echo "job specs md5-verified clean ($LOCAL_N files)."

echo "== DEPLOY OK =="
