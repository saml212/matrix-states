#!/usr/bin/env bash
# Deploys the queue system (runner + docs + generated job specs) from this
# repo checkout to the box, then md5-verifies every deployed file landed
# byte-identical. Run LOCALLY (this is a scp wrapper, not a box script).
#
# JOB-SPEC SYNC IS RESURRECTION-SAFE (fixed 2026-07-12, hardened across
# THREE independent audit rounds the same day -- six real bugs found and
# closed, see git history: 1 launch-blocking FATAL, 3 SERIOUS, 2 MINOR).
# The local jobs/pending/*.json tree is a FROZEN snapshot taken at
# generation time, while the box moves specs pending/ -> claimed/ ->
# completed/ (or failed/, or a parked_*/ re-gate dir) as workers run
# them. This script therefore NEVER blind-copies the whole local
# pending/ tree at the box. It:
#   1. resolves $REMOTE_QROOT to an ABSOLUTE remote path once, up front
#      (see the REMOTE_QROOT_ABS note below -- load-bearing, not cosmetic,
#      round-1-audit finding: a bare tilde-string default silently
#      enumerated NOTHING when passed through a heredoc positional param);
#   2. read-only enumerates every basename already present ANYWHERE on
#      the box (pending/claimed/completed/failed/parked_*/), NORMALIZING
#      claimed/, failed/, AND parked_*/ filenames -- queue_worker.sh:98
#      suffixes an in-flight claim "<name>.g<gpu>.json", and NOT every
#      code path that later moves a claimed file strips it back off (the
#      validity-check path does, queue_worker.sh:140/144; the
#      malformed-spec early-exit path does NOT, queue_worker.sh:122);
#      parked_*/ is a re-gate destination reachable by `mv` from ANY
#      state dir, including claimed/, so it can carry a suffixed name
#      too. Without normalizing all three, a currently-claimed, a
#      suffix-still-attached failed, or a suffix-still-attached parked
#      job would look like "not found anywhere" and get re-shipped into
#      pending/ -- exactly the resurrection bug this script exists to
#      prevent (round 1: would have hit the mid-flight 1.31B run via
#      claimed/; round 2: the same hole via failed/; round 3: the same
#      hole via parked_*/, reproduced live against a scratch fixture);
#   3. ships ONLY local specs whose basename is not present anywhere on
#      the box (genuinely new work);
#   4. stages shipped files in an isolated tmp dir, md5-verifies them
#      BEFORE placement, THEN does one final re-check
#      (completed/failed/claimed/parked_*, all suffix-tolerant) + atomic
#      no-clobber `mv` into pending/ in a single remote script (closing
#      the claim race to the tightest practical window). A genuine hard
#      placement failure (not a benign race-skip) makes the placement
#      script exit non-zero so the outer script surfaces it as FATAL
#      instead of silently printing "DEPLOY OK" (round-3 finding);
#   5. redeploys queue_worker.sh/launch_workers.sh/QUEUE_README.md/
#      ncr_task.py/ncr_earlyln_scale.py via STAGE, VERIFY, THEN
#      ATOMIC-RENAME (verify-before-publish, not verify-after) -- never a
#      direct in-place scp (round-2 audit: a direct scp truncates the
#      SAME inode a running worker or job may be mid-read/mid-exec of --
#      ncr_earlyln_scale.py is imported by every queued job's cmd, and
#      queue_worker.sh is re-exec'd by its own supervisor loop every
#      ~15s; a same-filesystem `mv` is atomic, so any in-flight reader
#      keeps the OLD complete inode until its next fresh open. Round 3
#      moved the md5 verification to run on the STAGED copy before the
#      rename, not the live copy after -- verify-after left a window
#      where a bad transfer could publish before being caught).
# A spec that's already claimed/completed/failed/parked on the box is
# NEVER re-shipped -- it is not "mid-flight or done" material for this
# script to touch, ever. If the box can't be reached or its queue state
# can't be enumerated, this script FAILS LOUDLY and copies nothing --
# it never falls back to a blind overwrite.
#
# REMOTE_QROOT_ABS (why this matters, don't remove it): bash tilde
# expansion is LEXICAL -- it only fires on a literal "~" at the START of
# a source-text word, never on a variable whose VALUE happens to start
# with "~". The default $REMOTE_QROOT is the literal string "~/queue".
# Interpolating that string directly into a remote COMMAND LINE (e.g.
# `ssh "$BOX" "mkdir -p $REMOTE_QROOT/pending"`) works, because the
# remote shell re-parses the whole line from scratch and sees a literal
# "~" at word-start. But passing the SAME string as a positional
# parameter to a `bash -s -- "$REMOTE_QROOT"` heredoc does NOT work --
# inside that heredoc, referencing "$1" or a variable assigned from it
# never re-triggers tilde expansion, so `"$QROOT/pending"` would resolve
# to a directory literally named "~" under $HOME, not $HOME/queue/pending
# -- silently enumerating nothing and, again, reintroducing the
# resurrection bug (verified against the live box during the audit that
# caught this). The fix: resolve the root to a concrete absolute path
# ONCE via `cd $REMOTE_QROOT && pwd` (interpolated, so it tilde-expands
# correctly), then use ONLY that resolved absolute path everywhere else
# (heredoc positional params, the staging dir, scp destinations). A
# resolved absolute path has no "~" left in it, so this hazard cannot
# recur downstream.
#
# Usage:
#   bash deploy.sh                  # normal deploy
#   DRY_RUN=1 bash deploy.sh        # show the ship/skip plan, copy nothing
#     (job-spec plan only -- a real run ALSO unconditionally re-deploys
#     queue_worker.sh/launch_workers.sh/QUEUE_README.md/ncr_task.py/
#     ncr_earlyln_scale.py, which are not job-spec state and are not
#     gated by this flag)
#   SKIP_STATIC=1 bash deploy.sh    # job-spec sync ONLY -- skip the
#     runner/launcher/docs/ncr redeploy entirely. Mainly for isolated
#     testing against a scratch REMOTE_QROOT without touching the live
#     ~/ncr/ (ncr_task.py/ncr_earlyln_scale.py are NOT parameterized by
#     REMOTE_QROOT -- they are genuinely shared, not per-queue-root state
#     -- so even a scratch-REMOTE_QROOT run would otherwise still write
#     the real ~/ncr/ files).
#   REMOTE_QROOT=queue_test bash deploy.sh   # point at a scratch remote
#     queue dir instead of the live ~/queue (for testing). A bare
#     relative name is safest (no leading "~" or "/") -- ssh's default
#     remote login shell starts in the box's own $HOME, so a relative
#     name resolves there unambiguously, and a leading "~" risks the
#     LOCAL-shell env-prefix tilde-expansion footgun described below.
#     (This script's own internal REMOTE_QROOT_ABS resolution, above,
#     protects the deploy logic either way -- this note is about the
#     value YOU type on the command line, not the script's internals.)
#     A LEADING "~" ON THE COMMAND LINE IS THE WRONG CHOICE: bash
#     tilde-expands an env-prefix assignment (`VAR=~/x cmd`) using the
#     LOCAL machine's $HOME before deploy.sh ever runs, silently handing
#     the script a local-looking absolute path to use as a REMOTE path
#     (verified empirically -- do not reintroduce this).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
NCR_LOCAL="$REPO_ROOT/matrix-thinking/ncr"
BOX="youthful-indigo-turkey"
REMOTE_QROOT="${REMOTE_QROOT:-~/queue}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_STATIC="${SKIP_STATIC:-0}"
SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes)

echo "== ensuring box-side directories exist (also doubles as the reachability check) =="
if ! ssh "${SSH_OPTS[@]}" "$BOX" "mkdir -p $REMOTE_QROOT/pending $REMOTE_QROOT/claimed $REMOTE_QROOT/completed $REMOTE_QROOT/failed $REMOTE_QROOT/logs"; then
  echo "FATAL: cannot reach $BOX (or cannot create $REMOTE_QROOT/{pending,claimed,completed,failed,logs}) -- refusing to do anything further. Nothing was copied." >&2
  exit 1
fi

echo "== resolving \$REMOTE_QROOT to an absolute remote path (see header comment -- avoids a tilde double-expansion hazard in the heredocs below) =="
REMOTE_QROOT_ABS="$(ssh "${SSH_OPTS[@]}" "$BOX" "cd $REMOTE_QROOT && pwd")" || {
  echo "FATAL: could not resolve $REMOTE_QROOT to an absolute path on $BOX -- refusing to do anything further. Nothing was copied." >&2
  exit 1
}
if [ -z "$REMOTE_QROOT_ABS" ]; then
  echo "FATAL: resolved remote queue root came back empty -- refusing to do anything further. Nothing was copied." >&2
  exit 1
fi
case "$REMOTE_QROOT_ABS" in
  /*) : ;;
  *)
    echo "FATAL: resolved remote queue root '$REMOTE_QROOT_ABS' is not an absolute path -- refusing to trust it. Nothing was copied." >&2
    exit 1
    ;;
esac
if [ "$(printf '%s' "$REMOTE_QROOT_ABS" | wc -l | tr -d ' ')" != "0" ]; then
  echo "FATAL: resolved remote queue root spans multiple lines (likely a polluted remote shell profile) -- refusing to trust it. Nothing was copied." >&2
  exit 1
fi
echo "   $REMOTE_QROOT -> $REMOTE_QROOT_ABS"

## ---------------------------------------------------------------------
## Job-spec plan: computed FIRST, before any copying, so DRY_RUN can
## short-circuit before anything live is touched.
## ---------------------------------------------------------------------

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

HAVE_LOCAL_SPECS=1
LOCAL_N=0
N_TO_SHIP=0
if ! ls "$HERE"/jobs/pending/*.json >/dev/null 2>&1; then
  HAVE_LOCAL_SPECS=0
  echo "== no local job specs in jobs/pending/ -- nothing to sync there. =="
else
  ls "$HERE"/jobs/pending/*.json | xargs -n1 basename | sort > "$WORKDIR/local_pending.txt"
  LOCAL_N=$(wc -l < "$WORKDIR/local_pending.txt" | tr -d ' ')

  echo "== read-only enumeration of EVERY basename already on the box"
  echo "   (pending/claimed/completed/failed/parked_*/) -- this decides what"
  echo "   is safe to ship. claimed/ filenames are normalized (worker claims"
  echo "   suffix '.g<gpu>.json' -- queue_worker.sh:98) so a currently-claimed"
  echo "   (mid-flight) job is correctly recognized as resolved. Never falls"
  echo "   back to a blind copy if this fails. =="
  if ! ssh "${SSH_OPTS[@]}" "$BOX" bash -s -- "$REMOTE_QROOT_ABS" <<'REMOTE_EOF' > "$WORKDIR/remote_state_raw.txt"
set -uo pipefail
QROOT="$1"
# Defense-in-depth (the sole backstop for the mode that caused the two
# resurrection bugs this script closed): if any of the four core state
# dirs are missing, something is badly wrong (wrong root, permissions,
# a half-migrated box) -- fail loudly rather than silently enumerate an
# empty/partial view that would make everything look "genuinely new".
for d in pending claimed completed failed; do
  if [ ! -d "$QROOT/$d" ]; then
    echo "ENUM-FATAL: $QROOT/$d does not exist" >&2
    exit 17
  fi
done
ls "$QROOT/pending"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null
true
ls "$QROOT/completed"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null
true
# failed/ is USUALLY unsuffixed (queue_worker.sh's validity-check path,
# lines 140/144, strips ".g<gpu>" before the mv) but NOT always: the
# malformed-spec early-exit path (queue_worker.sh:122, missing cmd/
# validity_check) moves the claimed file to failed/ via a bare
# `basename`, suffix intact. Normalize unconditionally -- harmless
# no-op on an already-bare name, and closes the exact class of bug the
# claimed/ normalization below exists for (found by a second audit
# round after the first claimed/ fix landed).
ls "$QROOT/failed"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed -E 's/\.g[0-9]+\.json$/.json/'
true
# claimed/ is suffixed .g<gpu>.json at claim time (queue_worker.sh's own
# atomic-claim mv, line 98) -- normalize back to the original spec
# basename so a currently-claimed job matches its unsuffixed local name.
ls "$QROOT/claimed"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed -E 's/\.g[0-9]+\.json$/.json/'
true
# parked_*/ is a re-gate destination reached by `mv` from ANY state dir
# (QUEUE_README.md's own "reversible -- mv back into pending/" contract
# does not restrict what can be parked FROM) -- a spec parked directly
# out of claimed/ would still carry its ".g<gpu>" suffix. Normalize here
# too (third instance of this class, found by a third audit round after
# the claimed/failed fixes landed -- reproduced live against a scratch
# fixture; not yet real on the box's actual parked_k24plus/, whose 34
# entries are all bare names today, but load-bearing on the very next
# re-gate that parks something out of claimed/).
for d in "$QROOT"/parked_*/; do
  [ -d "$d" ] || continue
  ls "$d"*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed -E 's/\.g[0-9]+\.json$/.json/'
  true
done
REMOTE_EOF
  then
    echo "FATAL: could not enumerate remote queue state (pending/claimed/completed/failed/parked_*) on $BOX -- refusing to ship ANY job specs. Nothing was copied. Never falling back to a blind overwrite." >&2
    exit 1
  fi
  sort "$WORKDIR/remote_state_raw.txt" > "$WORKDIR/remote_state.txt"
  REMOTE_STATE_N=$(wc -l < "$WORKDIR/remote_state.txt" | tr -d ' ')

  comm -23 "$WORKDIR/local_pending.txt" "$WORKDIR/remote_state.txt" > "$WORKDIR/to_ship.txt"
  comm -12 "$WORKDIR/local_pending.txt" "$WORKDIR/remote_state.txt" > "$WORKDIR/already_present.txt"
  N_TO_SHIP=$(wc -l < "$WORKDIR/to_ship.txt" | tr -d ' ')
  N_ALREADY=$(wc -l < "$WORKDIR/already_present.txt" | tr -d ' ')

  echo "local pending snapshot:        $LOCAL_N specs"
  echo "present anywhere on the box:   $REMOTE_STATE_N basenames (pending+claimed[normalized]+completed+failed+parked_*)"
  echo "already present -- SKIPPING (never resurrected, never overwritten): $N_ALREADY"
  echo "genuinely new -- safe to ship: $N_TO_SHIP"
  if [ "$N_ALREADY" -gt 0 ]; then
    echo "-- skipped (first 20) --"
    head -20 "$WORKDIR/already_present.txt"
  fi

  # Courtesy content-divergence check (non-blocking): if a skipped spec is
  # sitting in remote pending/ specifically (not claimed/completed/failed/
  # parked) with DIFFERENT content than the local copy, warn loudly. It is
  # still never overwritten (it could be claimed by a worker at any
  # moment) -- this is purely so a human notices a regenerated local spec
  # has silently diverged from the live one.
  if [ "$N_ALREADY" -gt 0 ]; then
    while IFS= read -r f; do
      remote_pending_md5=$(ssh -n "${SSH_OPTS[@]}" "$BOX" "md5sum '$REMOTE_QROOT_ABS/pending/$f' 2>/dev/null | awk '{print \$1}'")
      [ -n "$remote_pending_md5" ] || continue
      local_md5=$(cd "$HERE/jobs/pending" && (md5 -q "$f" 2>/dev/null || md5sum "$f" | awk '{print $1}'))
      if [ "$local_md5" != "$remote_pending_md5" ]; then
        echo "WARNING: $f is still pending on the box but its content DIFFERS from the local copy (local=$local_md5 remote=$remote_pending_md5) -- NOT overwritten (it could be claimed at any moment); reconcile by hand if this is unexpected." >&2
      fi
    done < "$WORKDIR/already_present.txt"
  fi
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "-- specs to ship (if any) --"
  if [ "$N_TO_SHIP" -gt 0 ]; then
    cat "$WORKDIR/to_ship.txt"
  else
    echo "(none)"
  fi
  echo "== DRY_RUN=1: the above is the exact job-spec ship/skip plan. A REAL"
  echo "   run would ALSO unconditionally re-deploy queue_worker.sh/"
  echo "   launch_workers.sh/QUEUE_README.md/ncr_task.py/ncr_earlyln_scale.py"
  echo "   (static files, not job-spec state -- not gated by DRY_RUN on a"
  echo "   real run). Nothing was copied by this DRY_RUN. =="
  exit 0
fi

if [ "$SKIP_STATIC" = "1" ]; then
  echo "== SKIP_STATIC=1: skipping runner/launcher/docs/ncr redeploy (job-spec sync only, for isolated testing). =="
else
  echo "== copying runner + launcher + docs (staged, then atomically renamed into"
  echo "   place). NEVER scp directly onto the live path here: queue_worker.sh is"
  echo "   re-exec'd by its own supervisor loop every ~15s, and a direct scp"
  echo "   truncates-then-rewrites the SAME inode in place -- a worker mid-read"
  echo "   (or the supervisor mid-respawn) could see a partial file. Staging +"
  echo "   'mv' (same-filesystem rename, atomic) means any in-flight reader keeps"
  echo "   the OLD complete inode until it next opens the file fresh. (Found by"
  echo "   the same audit round that caught the claimed/failed suffix bug.) =="
  STATIC_STAGE="$(ssh "${SSH_OPTS[@]}" "$BOX" "mktemp -d '$REMOTE_QROOT_ABS/.deploy_static_stage.XXXXXX'")" || {
    echo "FATAL: could not create a remote staging dir for runner/docs under $REMOTE_QROOT_ABS -- refusing to redeploy them. Nothing was touched." >&2
    exit 1
  }
  if [ -z "$STATIC_STAGE" ]; then
    echo "FATAL: runner/docs staging dir creation returned an empty path -- refusing to redeploy them." >&2
    exit 1
  fi
  if ! scp -q "$HERE/queue_worker.sh" "$HERE/launch_workers.sh" "$HERE/QUEUE_README.md" "$BOX:$STATIC_STAGE/"; then
    echo "FATAL: scp of runner/docs to staging dir failed -- refusing to touch the live copies. Staged (partial) files left at $STATIC_STAGE for inspection." >&2
    exit 1
  fi

  echo "== copying the K-ladder extension (ncr_task.py / ncr_earlyln_scale.py),"
  echo "   staged + atomically renamed (same truncate-in-place hazard as above --"
  echo "   ncr_earlyln_scale.py is imported by EVERY queued job's cmd) --"
  echo "   REQUIRED: Lane A/C job specs pass --K 20..256, which only exist in"
  echo "   THIS session's additive GRIDS/GRID_SHAPES edit (audit FATAL F1 fix)."
  NCR_STAGE="$(ssh "${SSH_OPTS[@]}" "$BOX" 'mktemp -d ~/ncr/.deploy_ncr_stage.XXXXXX')" || {
    echo "FATAL: could not create a remote staging dir under ~/ncr -- refusing to redeploy ncr_task.py/ncr_earlyln_scale.py. Nothing there was touched." >&2
    exit 1
  }
  if [ -z "$NCR_STAGE" ]; then
    echo "FATAL: ncr staging dir creation returned an empty path -- refusing to redeploy." >&2
    exit 1
  fi
  if ! scp -q "$NCR_LOCAL/ncr_task.py" "$NCR_LOCAL/ncr_earlyln_scale.py" "$BOX:$NCR_STAGE/"; then
    echo "FATAL: scp of ncr_task.py/ncr_earlyln_scale.py to staging dir failed -- refusing to touch the live copies. Staged (partial) files left at $NCR_STAGE for inspection." >&2
    exit 1
  fi

  # md5-verify BOTH staged sets BEFORE either atomic-rename publish step --
  # verify-before-publish, not verify-after (an exit-0-but-truncated scp
  # must never reach the live path even briefly; queue_worker.sh is
  # re-exec'd by its supervisor every ~15s and ncr_earlyln_scale.py is
  # imported by every queued job's cmd -- found by a third audit round).
  echo "== md5 verify: staged runner + launcher + docs (before publish) =="
  LOCAL_SUM=$(cd "$HERE" && (md5 -q queue_worker.sh launch_workers.sh QUEUE_README.md 2>/dev/null || md5sum queue_worker.sh launch_workers.sh QUEUE_README.md | awk '{print $1}'))
  REMOTE_SUM=$(ssh "${SSH_OPTS[@]}" "$BOX" "cd '$STATIC_STAGE' && (md5sum queue_worker.sh launch_workers.sh QUEUE_README.md | awk '{print \$1}')")
  if [ "$(echo "$LOCAL_SUM" | tr -d '\n')" != "$(echo "$REMOTE_SUM" | tr -d '\n')" ]; then
    echo "MD5 MISMATCH on staged runner/docs -- deploy FAILED, do not launch. Live copies NOT touched; bad staged files left at $STATIC_STAGE for inspection." >&2
    echo "local:  $LOCAL_SUM" >&2
    echo "remote: $REMOTE_SUM" >&2
    exit 1
  fi
  echo "staged runner/launcher/docs md5-verified clean."

  echo "== md5 verify: staged ncr_task.py / ncr_earlyln_scale.py (before publish) =="
  LOCAL_NCR_SUM=$(cd "$NCR_LOCAL" && (md5 -q ncr_task.py ncr_earlyln_scale.py 2>/dev/null || md5sum ncr_task.py ncr_earlyln_scale.py | awk '{print $1}'))
  REMOTE_NCR_SUM=$(ssh "${SSH_OPTS[@]}" "$BOX" "cd '$NCR_STAGE' && (md5sum ncr_task.py ncr_earlyln_scale.py | awk '{print \$1}')")
  if [ "$(echo "$LOCAL_NCR_SUM" | tr -d '\n')" != "$(echo "$REMOTE_NCR_SUM" | tr -d '\n')" ]; then
    echo "MD5 MISMATCH on staged ncr_task.py/ncr_earlyln_scale.py -- deploy FAILED, do not launch. Live copies NOT touched; bad staged files left at $NCR_STAGE for inspection." >&2
    echo "local:  $LOCAL_NCR_SUM" >&2
    echo "remote: $REMOTE_NCR_SUM" >&2
    exit 1
  fi
  echo "staged ncr_task.py/ncr_earlyln_scale.py md5-verified clean."

  # Both staged sets are now verified byte-identical to local -- ONLY now
  # publish, via same-filesystem atomic rename (never a direct scp onto
  # the live path -- see header comment).
  echo "== publishing verified runner + launcher + docs (atomic rename) =="
  if ! ssh "${SSH_OPTS[@]}" "$BOX" "mv -f '$STATIC_STAGE/queue_worker.sh' '$REMOTE_QROOT_ABS/queue_worker.sh' && mv -f '$STATIC_STAGE/launch_workers.sh' '$REMOTE_QROOT_ABS/launch_workers.sh' && mv -f '$STATIC_STAGE/QUEUE_README.md' '$REMOTE_QROOT_ABS/QUEUE_README.md' && rmdir '$STATIC_STAGE'"; then
    echo "FATAL: could not atomically move VERIFIED staged runner/docs into place -- each mv is atomic individually, but a partial set may have landed if this failed partway through; inspect $REMOTE_QROOT_ABS and $STATIC_STAGE by hand before trusting the live copies." >&2
    exit 1
  fi
  echo "runner/launcher/docs published."

  echo "== publishing verified ncr_task.py / ncr_earlyln_scale.py (atomic rename) =="
  if ! ssh "${SSH_OPTS[@]}" "$BOX" "mv -f '$NCR_STAGE/ncr_task.py' ~/ncr/ncr_task.py && mv -f '$NCR_STAGE/ncr_earlyln_scale.py' ~/ncr/ncr_earlyln_scale.py && rmdir '$NCR_STAGE'"; then
    echo "FATAL: could not atomically move VERIFIED staged ncr files into place -- each mv is atomic individually, but a partial set may have landed if this failed partway through; inspect ~/ncr and $NCR_STAGE by hand before trusting the live copies." >&2
    exit 1
  fi
  echo "ncr_task.py/ncr_earlyln_scale.py published."
fi

if [ "$HAVE_LOCAL_SPECS" -eq 0 ] || [ "$N_TO_SHIP" -eq 0 ]; then
  echo "== no new job specs to ship (already in sync). =="
  echo "== DEPLOY OK =="
  exit 0
fi

echo "-- specs to ship --"
cat "$WORKDIR/to_ship.txt"

echo "== staging $N_TO_SHIP new spec(s) in an isolated remote tmp dir (never writes to pending/ directly yet) =="
REMOTE_STAGE="$(ssh "${SSH_OPTS[@]}" "$BOX" "mktemp -d '$REMOTE_QROOT_ABS/.deploy_stage.XXXXXX'")" || {
  echo "FATAL: could not create a remote staging dir under $REMOTE_QROOT_ABS -- refusing to ship. Nothing was copied." >&2
  exit 1
}
if [ -z "$REMOTE_STAGE" ]; then
  echo "FATAL: remote staging dir creation returned an empty path -- refusing to ship. Nothing was copied." >&2
  exit 1
fi

SHIP_PATHS=()
while IFS= read -r f; do
  SHIP_PATHS+=("$HERE/jobs/pending/$f")
done < "$WORKDIR/to_ship.txt"

if ! scp -q "${SHIP_PATHS[@]}" "$BOX:$REMOTE_STAGE/"; then
  echo "FATAL: scp to staging dir failed -- refusing to move anything into pending/. Remote staging dir $REMOTE_STAGE left on the box for inspection; nothing in pending/claimed/completed/failed was touched." >&2
  exit 1
fi

echo "== md5 verify: staged (to-ship) job specs, per-file =="
MISMATCH=0
while IFS= read -r f; do
  local_md5=$(cd "$HERE/jobs/pending" && (md5 -q "$f" 2>/dev/null || md5sum "$f" | awk '{print $1}'))
  remote_md5=$(ssh -n "${SSH_OPTS[@]}" "$BOX" "md5sum '$REMOTE_STAGE/$f' 2>/dev/null | awk '{print \$1}'")
  if [ "$local_md5" != "$remote_md5" ]; then
    echo "MD5 MISMATCH: $f  local=$local_md5 remote=$remote_md5" >&2
    MISMATCH=1
  fi
done < "$WORKDIR/to_ship.txt"
if [ "$MISMATCH" -eq 1 ]; then
  echo "FATAL: job spec md5 mismatch on staged files -- deploy FAILED, do not launch. Staged files left at $REMOTE_STAGE (NOT pending/) for inspection; nothing in pending/claimed/completed/failed was touched." >&2
  exit 1
fi
echo "staged job specs md5-verified clean ($N_TO_SHIP files)."

echo "== final re-check + atomic no-clobber placement into pending/ (single remote script, tightest practical race window) =="
if ! ssh "${SSH_OPTS[@]}" "$BOX" bash -s -- "$REMOTE_QROOT_ABS" "$REMOTE_STAGE" > "$WORKDIR/placement.log" 2>&1 <<'REMOTE_EOF'
set -uo pipefail
QROOT="$1"
STAGE="$2"
placed=0
skipped=0
hard_errors=0
# resolved_elsewhere: true if $1 (a bare spec basename) already has a
# fate anywhere except pending/. completed/ is normally unsuffixed but
# checked both ways for symmetry/defense-in-depth; failed/ and claimed/
# are checked both ways because queue_worker.sh has more than one code
# path into each and not all of them strip the ".g<gpu>" claim suffix
# (see the enumeration script's own comment on this -- found by audit).
resolved_elsewhere() {
  b="$1"
  [ -e "$QROOT/completed/$b" ] && return 0
  compgen -G "$QROOT/completed/${b%.json}.g*.json" > /dev/null 2>&1 && return 0
  [ -e "$QROOT/failed/$b" ] && return 0
  compgen -G "$QROOT/failed/${b%.json}.g*.json" > /dev/null 2>&1 && return 0
  [ -e "$QROOT/claimed/$b" ] && return 0
  compgen -G "$QROOT/claimed/${b%.json}.g*.json" > /dev/null 2>&1 && return 0
  # parked_*/ can be reached by mv from any state dir (including claimed/),
  # so it can carry a suffixed name too -- check both forms.
  compgen -G "$QROOT"/parked_*/"$b" > /dev/null 2>&1 && return 0
  compgen -G "$QROOT"/parked_*/"${b%.json}".g*.json > /dev/null 2>&1 && return 0
  return 1
}
for f in "$STAGE"/*.json; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  if resolved_elsewhere "$base"; then
    echo "RACE-SKIP(resolved-on-box) $base -- gained a fate between enumeration and placement, NOT overwriting"
    skipped=$((skipped+1))
    continue
  fi
  # mv -n: no-clobber. If something claimed this exact name in pending/ in
  # the last few seconds, $f is left untouched in $STAGE (source survives).
  if mv -n "$f" "$QROOT/pending/$base" && [ ! -e "$f" ]; then
    echo "PLACED $base"
    placed=$((placed+1))
  elif [ -e "$QROOT/pending/$base" ]; then
    echo "RACE-SKIP(pending-collision) $base -- mv -n refused, name already exists in pending/, left staged for review (benign -- re-run to re-evaluate)"
    skipped=$((skipped+1))
  else
    # Genuinely NOT a collision (target name absent from pending/) and the
    # mv still didn't happen -- permissions, disk-full, or some other real
    # error, not the normal race. Counted separately so the outer script
    # can tell "some specs safely deferred" apart from "something is
    # actually broken" (found by a third audit round -- this branch used
    # to be indistinguishable from a benign race-skip and always exited 0).
    echo "HARD-ERROR(mv-failed) $base -- mv did not succeed and the target name is NOT in pending/ (permissions/disk/other real error, not a collision) -- left staged for review"
    hard_errors=$((hard_errors+1))
  fi
done
if rmdir "$STAGE" 2>/dev/null; then
  echo "stage dir removed (empty -- everything placed cleanly)"
else
  echo "NOTE: stage dir $STAGE left on box (non-empty -- race-skips present, or an error) -- inspect manually, do NOT delete pending/claimed/completed/failed to compensate"
fi
echo "SUMMARY placed=$placed skipped=$skipped hard_errors=$hard_errors"
[ "$hard_errors" -eq 0 ]
REMOTE_EOF
then
  echo "FATAL: remote placement script itself failed -- inspect $REMOTE_STAGE on $BOX by hand. Nothing further attempted automatically." >&2
  cat "$WORKDIR/placement.log" >&2
  exit 1
fi
cat "$WORKDIR/placement.log"
if grep -q "RACE-SKIP" "$WORKDIR/placement.log"; then
  echo "NOTE: one or more specs raced with the box between enumeration and placement and were correctly left un-shipped -- re-run this script to re-evaluate them (idempotent)." >&2
fi

echo "== DEPLOY OK =="
