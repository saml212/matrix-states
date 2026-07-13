#!/usr/bin/env bash
# ON-BOX, CLAUDE-INDEPENDENT QUEUE WORKER. One instance per GPU. Dumb and
# robust by design (CLAUDE.md's perpetual-sweep-pattern hard rules): no
# network dependencies, no coordination beyond the filesystem, no cleverness.
#
# Contract (see QUEUE_README.md for the full writeup):
#   - PAUSE sentinel  ($QROOT/PAUSE)      -> stop claiming NEW jobs; running
#     job (if any) finishes normally. Delete PAUSE to resume.
#   - STOP sentinel   ($QROOT/STOP)       -> finish current job (if any),
#     then exit cleanly. Global kill switch for the whole system.
#   - Free-GPU gate: claim a job ONLY if this worker's assigned GPU shows
#     ZERO nvidia-smi compute-apps AND < 2 GiB memory.used. Polls every 60s
#     otherwise. This is deliberately a COMPUTE-APPS check first, not a bare
#     memory threshold -- the K-scaling NCR cells this box already runs use
#     well under 2 GiB (tiny models), so a memory-only gate would silently
#     stack a second job onto an already-busy GPU. compute-apps lists every
#     process actually holding a CUDA context, regardless of how little VRAM
#     it uses, so it is the reliable "is this GPU doing anything" signal.
#     This auto-yields to ANY other job on the box (filler sessions, the
#     priority §11 K-scaling sweep, or anything else) with zero coordination.
#   - Atomic claim: `mv` a job spec from pending/ to claimed/. `mv` within one
#     filesystem is atomic -- of N workers racing the same filename, exactly
#     one mv succeeds (the others get "No such file" and move on to the next
#     candidate). Claimed filenames are suffixed .gN so a crashed worker's
#     own stale claims are unambiguously its own to reclaim on restart.
#   - Validity-checked completion: after the job's cmd exits (success OR
#     failure), run its OWN validity_check command. That check, not the cmd's
#     exit code, decides completed/ vs failed/ (house rule: check output
#     validity, not existence/exit-code -- a cmd can exit 0 with a truncated
#     or malformed output).
#   - One bad job never wedges the worker: the outer while-loop always
#     continues regardless of a job's outcome; each job's own stdout/stderr
#     goes to its own dedicated log file.
#   - Jobs run as this worker's own child (`bash -c "$cmd"`, synchronous, no
#     backgrounding) so killing this worker's EXACT tmux session name kills
#     its in-flight job too (the intended preemption contract, PART 1(c)).
#
# NOT YET WIRED IN (2026-07-13, landmine 2 follow-up, deliberately deferred):
# a pre-claim reconciliation gate. `reconcile_specs.py --check-one <path>`
# (matrix-thinking/queue/reconcile_specs.py) can tell whether a claimed
# spec's `--internal-timeout`/`--steps` (or any other flag) still matches
# what `generate_jobs.py` would produce for that id TODAY -- job 200 burned
# ~44 GPU-hours running with a timeout the generator had already been fixed
# to no longer produce (see QUEUE_README.md's Reconciliation section). The
# narrow window between the atomic claim (mv pending/->claimed/, above) and
# `cmd` actually starting (below) is the ONE place a drift check could
# safely BLOCK (not just report): the file is claimed but `cmd` has not run
# yet, so refusing to launch costs zero GPU-hours, unlike a drift found
# once a job is already mid-flight (which can only ever be advisory --
# see reconcile_specs.py's own ADVISORY-VS-BLOCKING note). This was
# deliberately NOT wired into the live loop below in the same session that
# discovered it: `generate_jobs.py` is a LOCAL-only script (not deployed to
# this box -- see deploy.sh's own static-file list, which does not include
# it), so a real worker-side check would need either a small drift-manifest
# generated locally and deployed as a new static file, or `generate_jobs.py`
# itself deployed here and importable -- either is a genuine design/audit
# task of its own, not a same-session addition to a script actively
# supervising 8 live GPUs. Do this properly (its own design note + audit)
# before wiring it in, not as a rushed edit to this file.
#
# Usage: queue_worker.sh <gpu_id>
# Deployed inside the house supervisor pattern:
#   tmux new-session -d -s queue_worker_g<N> \
#     "while [ ! -f \$QROOT/STOP ]; do bash queue_worker.sh <N>; sleep 15; done"
set -u

GPU="${1:?usage: queue_worker.sh GPU_ID}"
QROOT="${QROOT:-$HOME/queue}"
PENDING="$QROOT/pending"
CLAIMED="$QROOT/claimed"
COMPLETED="$QROOT/completed"
FAILED="$QROOT/failed"
LOGS="$QROOT/logs"
mkdir -p "$PENDING" "$CLAIMED" "$COMPLETED" "$FAILED" "$LOGS"

WLOG="$QROOT/worker_g${GPU}.log"
log() { echo "[worker g$GPU] $(date -u +%FT%TZ) $*" >> "$WLOG"; }

log "worker (re)starting on GPU $GPU (pid $$)"

# --- Resume-safety: reclaim this GPU's own stale claims from a prior crash.
# A claim is suffixed .g<GPU> at claim time (below), so on a fresh process
# start we can tell OUR OWN interrupted claims apart from other workers'.
# Disclosed limitation (matches the project's own fixscale precedent, no
# compounding-resume logic): this moves the spec back to pending/ for a
# FROM-SCRATCH retry, not a resume-from-checkpoint. Cheap/short jobs are
# unaffected; a few long Lane-B jobs disclose this explicitly in their own
# job-spec notes.
for f in "$CLAIMED"/*".g${GPU}.json"; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  orig="${base%.g${GPU}.json}.json"
  log "reclaiming stale claim from a prior crash: $base -> pending/$orig"
  mv "$f" "$PENDING/$orig" 2>/dev/null
done

while true; do
  if [ -f "$QROOT/STOP" ]; then
    log "STOP seen, exiting cleanly"
    exit 0
  fi
  if [ -f "$QROOT/PAUSE" ]; then
    sleep 60
    continue
  fi

  # --- Free-GPU gate (compute-apps primary, memory-threshold fallback) ---
  napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | grep -c '[0-9]')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  napps="${napps:-1}"
  mem="${mem:-9999}"
  if [ "$napps" -gt 0 ] || [ "$mem" -ge 2048 ]; then
    sleep 60
    continue
  fi

  # --- Atomic claim: earliest filename (priority prefix) wins ---
  claimed=""
  for f in $(ls "$PENDING" 2>/dev/null | sort); do
    src="$PENDING/$f"
    dst="$CLAIMED/${f%.json}.g${GPU}.json"
    if mv "$src" "$dst" 2>/dev/null; then
      claimed="$dst"
      break
    fi
    # mv failed => another worker claimed it first (or a transient error);
    # either way, try the next candidate in this snapshot.
  done
  if [ -z "$claimed" ]; then
    sleep 15
    continue
  fi

  jobid="$(basename "$claimed" .json)"
  jobid="${jobid%.g${GPU}}"
  log "claimed $jobid"

  # --- Parse the job spec (python for robust JSON; never trust naive grep) ---
  cmd=$(python3 -c "import json; print(json.load(open('$claimed'))['cmd'])" 2>/dev/null)
  vcheck=$(python3 -c "import json; print(json.load(open('$claimed'))['validity_check'])" 2>/dev/null)
  outdir=$(python3 -c "import json; print(json.load(open('$claimed')).get('output_dir',''))" 2>/dev/null)

  if [ -z "$cmd" ] || [ -z "$vcheck" ]; then
    log "$jobid MALFORMED spec (missing cmd/validity_check) -- routing to failed/, not wedging"
    mv "$claimed" "$FAILED/$(basename "$claimed")" 2>/dev/null
    continue
  fi
  [ -n "$outdir" ] && mkdir -p "$outdir" 2>/dev/null

  jlog="$LOGS/${jobid}.log"
  {
    echo "=== job $jobid START $(date -u +%FT%TZ) on GPU $GPU ==="
    echo "cmd: $cmd"
    echo "validity_check: $vcheck"
  } > "$jlog"

  ( CUDA_VISIBLE_DEVICES="$GPU" bash -c "$cmd" ) >> "$jlog" 2>&1
  rc=$?
  echo "=== job $jobid cmd exit=$rc END $(date -u +%FT%TZ) ===" >> "$jlog"

  # --- Validity check decides completed/ vs failed/, NOT the cmd exit code ---
  if bash -c "$vcheck" >> "$jlog" 2>&1; then
    mv "$claimed" "$COMPLETED/$(basename "$claimed" .json | sed "s/\.g${GPU}\$//").json" 2>/dev/null
    echo "VALIDITY_CHECK: PASS" >> "$jlog"
    log "$jobid COMPLETED (validity check passed, cmd_exit=$rc)"
  else
    mv "$claimed" "$FAILED/$(basename "$claimed" .json | sed "s/\.g${GPU}\$//").json" 2>/dev/null
    echo "VALIDITY_CHECK: FAIL" >> "$jlog"
    log "$jobid FAILED (validity check failed, cmd_exit=$rc)"
  fi
  # loop unconditionally -- one bad job never wedges this worker
done
