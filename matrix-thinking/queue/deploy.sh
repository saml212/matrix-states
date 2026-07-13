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
#   bash deploy.sh                  # DRY_RUN=1 is the DEFAULT (see the
#     FILENAME-VS-ID LANDMINE note below) -- this alone only prints the
#     id-keyed ship/skip plan, copies nothing.
#   DRY_RUN=0 bash deploy.sh        # the REAL deploy -- ships genuinely-new
#     job specs (id-keyed) and unconditionally re-deploys queue_worker.sh/
#     launch_workers.sh/QUEUE_README.md/ncr_task.py/ncr_earlyln_scale.py,
#     which are not job-spec state and are not gated by DRY_RUN.
#   DRY_RUN=1 bash deploy.sh        # explicit dry run (identical to the
#     default -- spelled out for clarity in scripts/muscle memory)
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
#
# FILENAME-VS-ID LANDMINE (fixed 2026-07-13 -- the SECOND incarnation of
# the resurrection bug; the 2026-07-12 fix above closed BLIND-OVERWRITE,
# this closes FILENAME-KEYED DEDUP, a different failure mode of the same
# root cause). Confirmed LIVE on the box the same day this was fixed: a
# 2026-07-12 re-prioritization round (regate_2026-07-12.md S11.4) renamed
# two ALREADY-PENDING specs on the box for priority ordering --
# pending/235_laneB_392m_fulltoken_per_token_openr1-mix-ext_s3.json ->
# pending/029_..., and 236_... -> 030_... -- WITHOUT changing their JSON
# "id" field, which still reads "235_..."/"236_..." (basename and id are
# only the same string BY CONVENTION at generation time -- nothing
# enforces that stays true after an on-box rename). Both are now claimed,
# live, running 392M jobs (claimed/029_...g5.json, claimed/030_...g0.json,
# verified against the box the day this was fixed). The repo's local
# jobs/pending/ still has files literally named 235_....json and
# 236_....json (generate_jobs.py's out_path is always f"{id}.json" -- it
# has no knowledge of an on-box rename). The PRE-FIX dedup enumerated the
# box by FILENAME (normalizing only the claim-time ".g<N>" suffix), so it
# would see "029_...", "030_..." on the box and "235_...", "236_..."
# locally as DISJOINT names -- i.e. "not found anywhere" -- and stage
# 235_.../236_... to ship as "genuinely new", placing a SECOND, duplicate
# spec for two ALREADY-RUNNING 392M jobs into pending/, where an idle GPU
# would claim and duplicate ~31.4 GPU-h of already-in-flight compute.
# Filename-keyed dedup cannot see this: the rename is invisible to it by
# construction, no matter how thoroughly the suffix-stripping is audited.
#
# THE FIX: dedup is keyed on the job spec's own "id" JSON field, read from
# file CONTENT on both sides, NEVER on filename/basename. A rename on the
# box, or any local/remote basename mismatch, no longer matters -- the
# SAME id anywhere in pending/claimed/completed/failed/cancelled/
# parked_*/ is recognized as already-resolved regardless of what either
# copy is named. This also SIMPLIFIES the box-side enumeration: the old
# suffix-normalization heuristic (sed-stripping ".g<N>" off claimed/ and
# failed/ names, which needed 3 separate audit rounds to cover every code
# path that does or doesn't apply the suffix) is gone entirely -- the id
# lives in the file's own content, never in its name, so there is nothing
# left to normalize. cancelled/ (a state dir that exists on the box but
# predates this script's original 4-dir enumeration) is now scanned too,
# for the identical reason: a cancelled id must never be silently
# re-shippable under a new basename either.
#
# Any file in a state dir that looks like a job spec (".json" in its name)
# but fails to JSON-parse, or parses but has no "id" field, is now an
# ENUM-FATAL -- refuse to ship ANYTHING rather than enumerate a state view
# that could be silently missing a live job's id. Every already-resolved
# id found in claimed/ specifically is printed as its own loud
# "LIVE-SKIP(claimed)" line (distinct from the generic already-present
# count) so a human reading the plan sees the live-collision check ran and
# passed, not just a number. If a to-ship id is EVER found to collide with
# claimed/ at the final placement step (structurally should be
# impossible, since the plan-time check already excludes it) that is a
# "*** LIVE-COLLISION ***"-marked, script-failing (nonzero exit) event,
# never a routine race-skip line -- see the placement script below.
#
# DRY_RUN NOW DEFAULTS TO 1 (the opposite of before). This landmine's root
# cause, both times, was a real run happening without anyone having looked
# at the plan first; QUEUE_README.md's own "good practice" advice was
# already "DRY_RUN=1 first" -- making that the default (an explicit
# DRY_RUN=0 is now required to ship anything for real) costs one extra
# keystroke on a genuine deploy and removes an entire class of "just ran
# it" mistake.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
NCR_LOCAL="$REPO_ROOT/matrix-thinking/ncr"
BOX="youthful-indigo-turkey"
REMOTE_QROOT="${REMOTE_QROOT:-~/queue}"
DRY_RUN="${DRY_RUN:-1}"   # DEFAULT-ON as of 2026-07-13 -- set DRY_RUN=0 to ship for real
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
N_TO_SHIP=0
if ! ls "$HERE"/jobs/pending/*.json >/dev/null 2>&1; then
  HAVE_LOCAL_SPECS=0
  echo "== no local job specs in jobs/pending/ -- nothing to sync there. =="
else
  echo "== read-only enumeration of EVERY job spec already on the box"
  echo "   (pending/claimed/completed/failed/cancelled/parked_*/), keyed on"
  echo "   each file's own JSON \"id\" field -- NEVER on filename/basename"
  echo "   (see the FILENAME-VS-ID LANDMINE header note: a spec renamed on"
  echo "   the box for priority ordering is still correctly recognized)."
  echo "   Never falls back to a blind copy if this fails. =="
  if ! ssh "${SSH_OPTS[@]}" "$BOX" bash -s -- "$REMOTE_QROOT_ABS" <<'REMOTE_EOF' > "$WORKDIR/remote_ids_raw.txt"
set -uo pipefail
QROOT="$1"
# Defense-in-depth (the sole backstop for the mode that caused the
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
QROOT="$QROOT" python3 <<'PYEOF'
import json, os, sys

qroot = os.environ["QROOT"]

def scan_dir(dirpath, label, rows):
    if not os.path.isdir(dirpath):
        return
    for name in sorted(os.listdir(dirpath)):
        full = os.path.join(dirpath, name)
        if not os.path.isfile(full):
            continue
        if ".json" not in name:
            continue  # e.g. cancelled/README.md -- not a job spec, ignore
        try:
            with open(full) as fh:
                spec = json.load(fh)
        except Exception as e:
            print(f"ENUM-FATAL: could not parse {full} as JSON ({e})", file=sys.stderr)
            sys.exit(18)
        jid = spec.get("id")
        if not jid:
            print(f"ENUM-FATAL: {full} has no 'id' field -- cannot dedup safely by id", file=sys.stderr)
            sys.exit(19)
        rows.append((jid, label, name))

rows = []
for d in ("pending", "claimed", "completed", "failed", "cancelled"):
    scan_dir(os.path.join(qroot, d), d, rows)
for name in sorted(os.listdir(qroot)):
    if name.startswith("parked_"):
        p = os.path.join(qroot, name)
        if os.path.isdir(p):
            scan_dir(p, name, rows)

for jid, label, name in rows:
    print(f"{jid}\t{label}\t{name}")
PYEOF
REMOTE_EOF
  then
    echo "FATAL: could not enumerate remote queue state (pending/claimed/completed/failed/cancelled/parked_*) on $BOX -- refusing to ship ANY job specs. Nothing was copied. Never falling back to a blind overwrite or a filename-only check." >&2
    exit 1
  fi

  echo "== computing id-keyed local/remote diff (dedup key = each spec's own JSON \"id\" field, never filename) =="
  if ! python3 - "$HERE/jobs/pending" "$WORKDIR/remote_ids_raw.txt" "$WORKDIR" <<'PYEOF'
import json, os, sys

local_dir, remote_raw, workdir = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- local: read every jobs/pending/*.json's own "id" field. NEVER
# assume it matches the basename -- that assumption is exactly what the
# filename-vs-id landmine broke (an on-box rename leaves content/id
# unchanged but the basename diverges). ----
local = {}          # id -> basename
dup_local = {}
for name in sorted(os.listdir(local_dir)):
    if not name.endswith(".json"):
        continue
    full = os.path.join(local_dir, name)
    try:
        with open(full) as fh:
            spec = json.load(fh)
    except Exception as e:
        print(f"FATAL: could not parse local spec {full} as JSON ({e}) -- refusing to compute a dedup plan on an incomplete local view.", file=sys.stderr)
        sys.exit(21)
    jid = spec.get("id")
    if not jid:
        print(f"FATAL: local spec {full} has no 'id' field -- cannot dedup safely by id.", file=sys.stderr)
        sys.exit(22)
    if jid in local:
        dup_local.setdefault(jid, [local[jid]]).append(name)
    else:
        local[jid] = name

if dup_local:
    print("FATAL: duplicate id(s) within LOCAL jobs/pending/ -- generate_jobs.py must produce unique ids per file:", file=sys.stderr)
    for jid, names in dup_local.items():
        print(f"  id={jid}: {names}", file=sys.stderr)
    sys.exit(23)

# ---- remote: the (id, label, basename) TSV the box-side scan produced --
remote = {}          # id -> (label, basename)
dup_remote = {}
with open(remote_raw) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        jid, label, name = line.split("\t", 2)
        if jid in remote:
            dup_remote.setdefault(jid, [remote[jid]]).append((label, name))
        else:
            remote[jid] = (label, name)

if dup_remote:
    print("FATAL: the SAME id exists in more than one place on the box -- this is a corrupted/ambiguous queue state (e.g. the same job spec content copied into two state dirs). Refusing to guess which is authoritative:", file=sys.stderr)
    for jid, occ in dup_remote.items():
        print(f"  id={jid}: first={remote[jid]} also_in={occ}", file=sys.stderr)
    sys.exit(24)

to_ship = sorted(set(local) - set(remote))
already = sorted(set(local) & set(remote))
live_claimed = [j for j in already if remote[j][0] == "claimed"]
other_present = [j for j in already if remote[j][0] != "claimed"]

with open(os.path.join(workdir, "to_ship.tsv"), "w") as out:
    for jid in to_ship:
        out.write(f"{jid}\t{local[jid]}\n")

with open(os.path.join(workdir, "pending_divergence_check.tsv"), "w") as out:
    for jid in other_present:
        label, rname = remote[jid]
        if label == "pending":
            out.write(f"{jid}\t{local[jid]}\t{rname}\n")

print(f"local pending snapshot:        {len(local)} specs (by id)")
print(f"present anywhere on the box:   {len(remote)} ids (pending+claimed+completed+failed+cancelled+parked_*, ID-KEYED)")
print(f"already present -- SKIPPING (never resurrected, never overwritten): {len(already)}")
print(f"  of which LIVE (claimed/, running right now):                     {len(live_claimed)}")
print(f"genuinely new -- safe to ship:  {len(to_ship)}")
print("")
if live_claimed:
    print("== LIVE JOBS ON THE BOX RIGHT NOW (matched by id, not filename -- this is exactly the check a renamed-copy collision needs) ==")
    for jid in live_claimed:
        label, rname = remote[jid]
        lname = local[jid]
        note = "" if lname == rname else "   *** RENAMED ON BOX (local name != box name) -- id-keyed dedup is why this is still caught ***"
        print(f"  LIVE-SKIP(claimed) id={jid}  local_file={lname}  box_file={rname}{note}")
    print("")
if other_present:
    shown = other_present[:20]
    print(f"-- other already-present ids ({len(shown)} of {len(other_present)} shown) --")
    for jid in shown:
        label, rname = remote[jid]
        print(f"  SKIP({label}) id={jid}  local_file={local[jid]}  box_file={rname}")
    print("")
print("-- ids safe to ship (if any) --")
if to_ship:
    for jid in to_ship:
        print(f"  SHIP id={jid}  local_file={local[jid]}")
else:
    print("  (none)")
PYEOF
  then
    echo "FATAL: id-keyed local/remote diff failed (see above) -- refusing to ship ANY job specs. Nothing was copied. Never falling back to a blind overwrite or a filename-only check." >&2
    exit 1
  fi

  # Courtesy content-divergence check (non-blocking): if an id already
  # resolved specifically to remote pending/ (not claimed/completed/
  # failed/cancelled/parked) has DIFFERENT content than the local copy,
  # warn loudly. Still never overwritten (it could be claimed by a worker
  # at any moment) -- this is purely so a human notices a regenerated
  # local spec has silently diverged from the live one. Uses the id's OWN
  # remote basename (which may differ from the local basename -- exactly
  # the renamed-copy case), not an assumed same-name match.
  if [ -s "$WORKDIR/pending_divergence_check.tsv" ]; then
    echo "== courtesy check: content divergence on already-pending ids (non-blocking) =="
    while IFS=$'\t' read -r jid lname rname; do
      remote_pending_md5=$(ssh -n "${SSH_OPTS[@]}" "$BOX" "md5sum '$REMOTE_QROOT_ABS/pending/$rname' 2>/dev/null | awk '{print \$1}'")
      [ -n "$remote_pending_md5" ] || continue
      local_md5=$(cd "$HERE/jobs/pending" && (md5 -q "$lname" 2>/dev/null || md5sum "$lname" | awk '{print $1}'))
      if [ "$local_md5" != "$remote_pending_md5" ]; then
        echo "WARNING: id=$jid is still pending on the box (as '$rname'; local copy is '$lname') but its CONTENT DIFFERS (local=$local_md5 remote=$remote_pending_md5) -- NOT overwritten (it could be claimed at any moment); reconcile by hand if this is unexpected." >&2
      fi
    done < "$WORKDIR/pending_divergence_check.tsv"
  fi

  N_TO_SHIP=$(wc -l < "$WORKDIR/to_ship.tsv" 2>/dev/null | tr -d ' ')
  N_TO_SHIP="${N_TO_SHIP:-0}"
fi

if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "== DRY_RUN=1 (the DEFAULT as of 2026-07-13 -- pass DRY_RUN=0 to actually"
  echo "   ship). The plan printed above is the exact id-keyed ship/skip"
  echo "   decision; the LIVE JOBS section (if any ids are live) confirms"
  echo "   every currently-claimed id was matched by its internal id and"
  echo "   excluded, regardless of what it is named on the box today. A REAL"
  echo "   run (DRY_RUN=0) would ALSO unconditionally re-deploy"
  echo "   queue_worker.sh/launch_workers.sh/QUEUE_README.md/ncr_task.py/"
  echo "   ncr_earlyln_scale.py (static files, not job-spec state -- not"
  echo "   gated by DRY_RUN on a real run). Nothing was copied by this"
  echo "   DRY_RUN. =="
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
  echo "== no new job specs to ship (already in sync, id-keyed). =="
  echo "== DEPLOY OK =="
  exit 0
fi

echo "-- specs to ship (id -> local file) --"
cat "$WORKDIR/to_ship.tsv"

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
while IFS=$'\t' read -r jid bname; do
  SHIP_PATHS+=("$HERE/jobs/pending/$bname")
done < "$WORKDIR/to_ship.tsv"

if ! scp -q "${SHIP_PATHS[@]}" "$BOX:$REMOTE_STAGE/"; then
  echo "FATAL: scp to staging dir failed -- refusing to move anything into pending/. Remote staging dir $REMOTE_STAGE left on the box for inspection; nothing in pending/claimed/completed/failed/cancelled was touched." >&2
  exit 1
fi

echo "== md5 verify: staged (to-ship) job specs, per-file =="
MISMATCH=0
while IFS=$'\t' read -r jid bname; do
  local_md5=$(cd "$HERE/jobs/pending" && (md5 -q "$bname" 2>/dev/null || md5sum "$bname" | awk '{print $1}'))
  remote_md5=$(ssh -n "${SSH_OPTS[@]}" "$BOX" "md5sum '$REMOTE_STAGE/$bname' 2>/dev/null | awk '{print \$1}'")
  if [ "$local_md5" != "$remote_md5" ]; then
    echo "MD5 MISMATCH: id=$jid file=$bname  local=$local_md5 remote=$remote_md5" >&2
    MISMATCH=1
  fi
done < "$WORKDIR/to_ship.tsv"
if [ "$MISMATCH" -eq 1 ]; then
  echo "FATAL: job spec md5 mismatch on staged files -- deploy FAILED, do not launch. Staged files left at $REMOTE_STAGE (NOT pending/) for inspection; nothing in pending/claimed/completed/failed/cancelled was touched." >&2
  exit 1
fi
echo "staged job specs md5-verified clean ($N_TO_SHIP files)."

echo "== final id-keyed re-check + atomic no-clobber placement into pending/ (single remote script, tightest practical race window) =="
if ! ssh "${SSH_OPTS[@]}" "$BOX" bash -s -- "$REMOTE_QROOT_ABS" "$REMOTE_STAGE" <<'REMOTE_EOF' > "$WORKDIR/placement.log" 2>&1
set -uo pipefail
QROOT="$1"
STAGE="$2"
QROOT="$QROOT" STAGE="$STAGE" python3 <<'PYEOF'
import json, os, sys

qroot = os.environ["QROOT"]
stage = os.environ["STAGE"]

def scan_dir(dirpath, label, id_map, dup):
    if not os.path.isdir(dirpath):
        return
    for name in sorted(os.listdir(dirpath)):
        full = os.path.join(dirpath, name)
        if not os.path.isfile(full):
            continue
        if ".json" not in name:
            continue
        try:
            with open(full) as fh:
                spec = json.load(fh)
        except Exception as e:
            print(f"PLACEMENT-ENUM-FATAL: could not parse {full} as JSON ({e})")
            sys.exit(18)
        jid = spec.get("id")
        if not jid:
            print(f"PLACEMENT-ENUM-FATAL: {full} has no 'id' field")
            sys.exit(19)
        if jid in id_map:
            dup.setdefault(jid, [id_map[jid]]).append((label, name))
        else:
            id_map[jid] = (label, name)

# Fresh re-scan of CURRENT state, right before placement -- closes the
# race window between the earlier plan-time enumeration and now.
resolved = {}
dup = {}
for d in ("pending", "claimed", "completed", "failed", "cancelled"):
    scan_dir(os.path.join(qroot, d), d, resolved, dup)
for name in sorted(os.listdir(qroot)):
    if name.startswith("parked_"):
        p = os.path.join(qroot, name)
        if os.path.isdir(p):
            scan_dir(p, name, resolved, dup)

if dup:
    print("PLACEMENT-FATAL: duplicate id(s) appeared in more than one state dir during the race-recheck -- refusing to place anything until this is resolved by hand:")
    for jid, occ in dup.items():
        print(f"  id={jid}: first={resolved[jid]} also_in={occ}")
    sys.exit(20)

placed = 0
skipped = 0
live_collisions = 0
hard_errors = 0

staged = sorted(f for f in os.listdir(stage) if f.endswith(".json"))
for name in staged:
    full = os.path.join(stage, name)
    try:
        with open(full) as fh:
            spec = json.load(fh)
    except Exception as e:
        print(f"HARD-ERROR(unparseable-staged-file) {name}: {e} -- left staged for review")
        hard_errors += 1
        continue
    jid = spec.get("id")
    if not jid:
        print(f"HARD-ERROR(staged-file-no-id) {name} -- left staged for review")
        hard_errors += 1
        continue

    if jid in resolved:
        label, rname = resolved[jid]
        if label == "claimed":
            # Structurally should be impossible -- the plan-time check
            # already excludes any id present in claimed/ from the ship
            # list. If it fires anyway, something changed between plan and
            # placement (or the plan-time check itself regressed): this IS
            # the "would resurrect a live job" case, and it gets the
            # loudest marker this script has, not a routine race-skip
            # line -- and it fails the script's exit code (see below).
            print(f"*** LIVE-COLLISION *** id={jid} staged_file={name} is CURRENTLY CLAIMED on the box as {rname} -- REFUSING to place. This should never happen (the plan-time check already excludes claimed ids); investigate immediately.")
            live_collisions += 1
        else:
            print(f"RACE-SKIP({label}) id={jid} staged_file={name} -- gained a fate ({label}/{rname}) between enumeration and placement, NOT overwriting")
            skipped += 1
        continue

    dst = os.path.join(qroot, "pending", name)
    try:
        # Atomic no-clobber move on one filesystem: hard-link then unlink
        # the source. os.link() raises FileExistsError if dst already
        # exists (the same-name-different-id race the old `mv -n` handled)
        # -- at least as safe as `mv -n`, and additionally atomic w.r.t. a
        # concurrent os.link() of the same dst (only one call can win).
        os.link(full, dst)
        os.unlink(full)
        print(f"PLACED id={jid} file={name}")
        placed += 1
    except FileExistsError:
        print(f"RACE-SKIP(pending-collision) id={jid} file={name} -- a file with this exact name already exists in pending/ (name collision, not an id collision), left staged for review")
        skipped += 1
    except OSError as e:
        print(f"HARD-ERROR(link-failed) id={jid} file={name}: {e} -- left staged for review")
        hard_errors += 1

try:
    os.rmdir(stage)
    print("stage dir removed (empty -- everything placed cleanly)")
except OSError:
    print(f"NOTE: stage dir {stage} left on box (non-empty -- race-skips/collisions/errors present) -- inspect manually, do NOT delete pending/claimed/completed/failed/cancelled to compensate")

print(f"SUMMARY placed={placed} skipped={skipped} live_collisions={live_collisions} hard_errors={hard_errors}")
sys.exit(0 if (hard_errors == 0 and live_collisions == 0) else 1)
PYEOF
REMOTE_EOF
then
  echo "FATAL: remote placement script itself failed (hard_errors and/or a *** LIVE-COLLISION *** were found -- this is the 'would resurrect a live job' case made loud and blocking, see the log below) -- inspect $REMOTE_STAGE on $BOX by hand. Nothing further attempted automatically." >&2
  cat "$WORKDIR/placement.log" >&2
  exit 1
fi
cat "$WORKDIR/placement.log"
if grep -q "RACE-SKIP" "$WORKDIR/placement.log"; then
  echo "NOTE: one or more specs raced with the box between enumeration and placement and were correctly left un-shipped -- re-run this script to re-evaluate them (idempotent)." >&2
fi

echo "== DEPLOY OK =="
