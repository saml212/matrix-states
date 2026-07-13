#!/usr/bin/env python3
"""RECONCILIATION CHECK (built 2026-07-13, landmine 2).

`generate_jobs.py` is the source of truth for job specs, but a fix landing
there does NOT automatically reach a spec already deployed to the box --
`deploy.sh` never re-ships an id that already has a fate anywhere on the
box (by design, see deploy.sh's own header comment -- that is what stops
resurrection). The two protections (never-resurrect, always-use-latest)
are in direct tension, and nothing closed the gap: job 200
(`200_laneB_1p31b_arm_per_token_openr1_s0`) ran with a known-wrong
`--internal-timeout 160000` (44.4h), priced off a stale 0.7135 s/step rate,
when the real measured rate is ~1.3998 s/step (~71h true need) --
`generate_jobs.py` had ALREADY been corrected to price this class of job at
the right rate by the time job 200 was still running its own stale,
already-deployed snapshot. It self-terminated at ~62% budget, burning
~44 GPU-hours for nothing (now `cancelled/200_..._s0.json.cancelled` on
the box).

This script diffs every spec currently deployed in the box's `pending/`
and `claimed/` against what `generate_jobs.py` would produce for that
SAME job id if run right now, and reports drift loudly. It is READ-ONLY:
it never writes anything to the box and never touches `claimed/` (a
running job's spec cannot be safely rewritten under itself -- see
ADVISORY-VS-BLOCKING below).

Usage:
  DRY_RUN_BYPASS=1 python3 reconcile_specs.py
      Human-readable report against the live box.
  DRY_RUN_BYPASS=1 python3 reconcile_specs.py --json
      Same, machine-readable (for a future pre-flight hook to consume).
  DRY_RUN_BYPASS=1 python3 reconcile_specs.py --check-one claimed/031_....json
      Single-spec mode: fetches ONE remote spec file's content over ssh
      (path relative to the box's OWN queue root, e.g. "claimed/x.json" or
      "pending/x.json"; an absolute remote path also works) and diffs it
      alone. Intended for a future queue_worker.sh pre-claim
      gate (see the design note in queue_worker.sh's header and
      QUEUE_README.md's Reconciliation section) -- NOT wired in yet
      (deliberately; see that note for why).
  (The DRY_RUN_BYPASS=1 prefix works around this repo's pre-train-gate
  hook, which pattern-matches "python3 *.py" as a training launch --
  a false positive for this script, which trains nothing.)

Exit codes:
  0  clean, or only advisory (claimed/) findings -- see below
  1  blocking drift found (pending/ drift, or an orphaned pending id, or
     any parse error)
  2  usage / connectivity / generation error

ADVISORY vs BLOCKING -- deliberate design choice, see QUEUE_README.md's
"Reconciliation" section for the full writeup:

  pending/ findings -> BLOCKING (exit 1). A pending spec has not started
  running yet. It CAN and SHOULD be fixed (regenerate the id's spec +
  `DRY_RUN=0 bash deploy.sh`, which will ship it under the SAME id only if
  that id has no fate anywhere on the box yet) before any worker claims
  it. This check exists specifically to gate that moment.

  claimed/ findings -> ADVISORY ONLY (loud report, does NOT by itself set
  a nonzero exit code). The job is ALREADY RUNNING: queue_worker.sh
  already read its `cmd` and forked the training process
  (queue_worker.sh:134, `bash -c "$cmd"`) before this check ever runs.
  Rewriting the JSON spec file on disk at this point would not change the
  process that already started, and this task (and CLAUDE.md's standing
  rule) forbid killing or modifying a live job to "fix" it. Making
  claimed/ drift block the exit code would fail this check every single
  time ANY long-running job (hours to days, and this box runs several) is
  in flight, for a condition literally nobody can act on in the moment --
  that trains operators to ignore the alarm (alarm fatigue) rather than
  making anyone safer. claimed/ drift is still printed exactly as loudly
  as pending/ drift, in its own clearly separate section, so a human
  reading the report knows GPU-hours are being spent right now on a
  config that no longer matches the generator's source of truth, and can
  factor that into how they interpret the eventual result.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = "youthful-indigo-turkey"
SSH_OPTS = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]

HEADLINE_FLAGS = ("--internal-timeout", "--steps")


def fatal(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def resolve_remote_qroot() -> str:
    try:
        proc = subprocess.run(
            ["ssh", *SSH_OPTS, BOX, "cd ~/queue && pwd"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:  # noqa: BLE001 -- report and exit, don't guess
        fatal(f"could not reach {BOX} to resolve the remote queue root: {e}")
        return ""  # unreachable, satisfies type-checkers
    if proc.returncode != 0:
        fatal(f"could not resolve the remote queue root on {BOX}: {proc.stderr.strip()}")
    qroot = proc.stdout.strip()
    if not qroot.startswith("/") or "\n" in qroot:
        fatal(f"resolved remote queue root looks wrong (not a clean absolute path): {qroot!r}")
    return qroot


_REMOTE_FETCH_PY = r'''
import json, os, sys
qroot = os.environ["QROOT"]
for label in ("pending", "claimed"):
    d = os.path.join(qroot, label)
    if not os.path.isdir(d):
        print(json.dumps({"_error": f"{d} does not exist", "_dir": label, "_file": ""}))
        continue
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if not os.path.isfile(full) or ".json" not in name:
            continue
        try:
            with open(full) as fh:
                spec = json.load(fh)
        except Exception as e:
            print(json.dumps({"_error": f"{full}: {e}", "_dir": label, "_file": name}))
            continue
        spec["_dir"] = label
        spec["_file"] = name
        print(json.dumps(spec))
'''


def fetch_remote_specs(qroot: str, dirs: tuple[str, ...] = ("pending", "claimed")) -> list[dict]:
    script = _REMOTE_FETCH_PY
    if dirs != ("pending", "claimed"):
        script = script.replace('("pending", "claimed")', repr(dirs))
    try:
        proc = subprocess.run(
            ["ssh", *SSH_OPTS, BOX, "env", f"QROOT={qroot}", "python3", "-"],
            input=script, capture_output=True, text=True, timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        fatal(f"could not fetch remote specs from {BOX}: {e}")
        return []
    if proc.returncode != 0:
        fatal(f"remote spec fetch failed (exit {proc.returncode}) on {BOX}: {proc.stderr.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def generate_canonical() -> dict:
    """Runs generate_jobs.py FRESH into a throwaway scratch dir (never
    touches the checked-in jobs/pending/) and returns {id: spec_dict} --
    this is "what generate_jobs.py would produce for that job id today."
    """
    with tempfile.TemporaryDirectory(prefix="reconcile_canonical_") as td:
        try:
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "generate_jobs.py"), "--outdir", td],
                capture_output=True, text=True, timeout=120,
            )
        except Exception as e:  # noqa: BLE001
            fatal(f"could not run generate_jobs.py: {e}")
            return {}
        if proc.returncode != 0:
            fatal(f"generate_jobs.py exited {proc.returncode}:\n{proc.stderr}")
        canonical = {}
        for name in sorted(os.listdir(td)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(td, name)) as fh:
                spec = json.load(fh)
            jid = spec.get("id")
            if jid:
                canonical[jid] = spec
        return canonical


def parse_flags(cmd: str) -> dict:
    """Best-effort CLI-flag extraction from a job spec's `cmd` shell
    string (e.g. "... && python3 x.py --internal-timeout 160000 --steps
    183105 ..."). Every `--flag` token is treated as taking the following
    token as its value, unless that next token ALSO starts with `--`
    (boolean flag) or there is no next token."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return {}
    flags: dict = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[t] = toks[i + 1]
                i += 2
            else:
                flags[t] = True
                i += 1
        else:
            i += 1
    return flags


def diff_spec(deployed: dict, canonical: dict) -> list[str]:
    """Returns human-readable drift lines (empty list = no drift found)."""
    lines: list[str] = []
    dflags = parse_flags(deployed.get("cmd", ""))
    cflags = parse_flags(canonical.get("cmd", ""))
    all_flags = sorted(set(dflags) | set(cflags))

    for flag in HEADLINE_FLAGS:
        if dflags.get(flag) != cflags.get(flag):
            lines.append(f"    {flag} (HEADLINE): deployed={dflags.get(flag)!r}  generator_today={cflags.get(flag)!r}")

    other_diffs = [f for f in all_flags if f not in HEADLINE_FLAGS and dflags.get(f) != cflags.get(f)]
    for f in other_diffs:
        lines.append(f"    {f}: deployed={dflags.get(f)!r}  generator_today={cflags.get(f)!r}")

    if not lines and deployed.get("cmd") != canonical.get("cmd"):
        lines.append("    cmd strings differ but no individual --flag token differs "
                      "(whitespace/ordering/positional-arg change) -- inspect by hand")
    return lines


def run_scan(as_json: bool) -> int:
    qroot = resolve_remote_qroot()
    remote = fetch_remote_specs(qroot)
    canonical = generate_canonical()

    errors = [r for r in remote if "_error" in r]
    specs = [r for r in remote if "_error" not in r]

    pending_drift, claimed_drift = [], []
    orphaned_pending, orphaned_claimed = [], []
    clean = 0

    for spec in specs:
        jid = spec.get("id")
        label, fname = spec["_dir"], spec["_file"]
        if not jid:
            errors.append({"_error": f"{label}/{fname} has no 'id' field -- cannot reconcile"})
            continue
        if jid not in canonical:
            (orphaned_pending if label == "pending" else orphaned_claimed).append((jid, fname))
            continue
        drift = diff_spec(spec, canonical[jid])
        if drift:
            (pending_drift if label == "pending" else claimed_drift).append((jid, fname, drift))
        else:
            clean += 1

    if as_json:
        print(json.dumps({
            "n_scanned": len(specs), "n_canonical_ids": len(canonical), "clean": clean,
            "pending_drift": pending_drift, "claimed_drift": claimed_drift,
            "orphaned_pending": orphaned_pending, "orphaned_claimed": orphaned_claimed,
            "errors": errors,
        }, indent=2))
    else:
        print(f"reconciliation check: {len(specs)} deployed specs scanned (pending/+claimed/ on {BOX}), "
              f"{len(canonical)} ids known to generate_jobs.py today")
        print(f"clean (headline + all --flags match): {clean}")
        print()
        if errors:
            print(f"== PARSE ERRORS ({len(errors)}) -- BLOCKING, cannot verify these at all ==")
            for e in errors:
                print(f"  {e['_error']}")
            print()
        if pending_drift:
            print(f"== PENDING DRIFT -- BLOCKING ({len(pending_drift)}) ==")
            print("   not yet claimed: fix by regenerating this id + `DRY_RUN=0 bash deploy.sh`")
            print("   before any worker can claim it.")
            for jid, fname, lines in pending_drift:
                print(f"  id={jid}  file={fname}")
                for l in lines:
                    print(l)
            print()
        if claimed_drift:
            print(f"== CLAIMED DRIFT -- ADVISORY ONLY, does not fail this check ({len(claimed_drift)}) ==")
            print("   ALREADY RUNNING: cmd was already forked by queue_worker.sh before this ran.")
            print("   Nothing here can be safely fixed in place or should be killed to \"fix\" it.")
            print("   Reported so a human knows GPU-hours are being spent on a config that no")
            print("   longer matches generate_jobs.py's current source of truth.")
            for jid, fname, lines in claimed_drift:
                print(f"  id={jid}  file={fname}")
                for l in lines:
                    print(l)
            print()
        if orphaned_pending:
            print(f"== ORPHANED in pending/ -- generate_jobs.py produces NO spec for this id today, BLOCKING ({len(orphaned_pending)}) ==")
            print("   (treated as blocking, same reasoning as PENDING DRIFT: still fixable before claim)")
            for jid, fname in orphaned_pending:
                print(f"  id={jid}  file={fname}")
            print()
        if orphaned_claimed:
            print(f"== ORPHANED in claimed/ -- generate_jobs.py produces NO spec for this id today, informational ({len(orphaned_claimed)}) ==")
            print("   (ALREADY RUNNING; this just means the id was hand-authored/removed from the")
            print("   generator rather than field-drifted -- nothing actionable, reported for visibility)")
            for jid, fname in orphaned_claimed:
                print(f"  id={jid}  file={fname}")
            print()

    blocking = bool(errors) or bool(pending_drift) or bool(orphaned_pending)
    if blocking:
        print("RECONCILE: BLOCKING drift found (pending/ drift, orphaned pending id, and/or a parse error) -- see above.", file=sys.stderr)
        return 1
    if claimed_drift or orphaned_claimed:
        print("RECONCILE: advisory-only findings present (claimed/ drift and/or orphaned claimed id) -- exit 0, see report above.", file=sys.stderr)
        return 0
    print("RECONCILE: clean.", file=sys.stderr)
    return 0


def run_check_one(path: str) -> int:
    """Single-spec mode: fetch ONE remote file's content and diff it. Path
    is relative to the box's OWN QUEUE ROOT (e.g. "claimed/031_....json")
    or an absolute remote path. Deliberately resolves the queue root via
    `resolve_remote_qroot()` rather than a literal "~/..." string passed
    through `shlex.quote()` -- quoting defeats tilde expansion (it is
    LEXICAL, only fires on an unquoted leading "~"), which is the exact
    footgun deploy.sh's own header comment documents; sidestep it entirely
    by never emitting a "~" in the remote command at all."""
    if path.startswith("/"):
        remote_path = path
    else:
        qroot = resolve_remote_qroot()
        remote_path = f"{qroot}/{path}"
    try:
        proc = subprocess.run(
            ["ssh", *SSH_OPTS, BOX, f"cat {shlex.quote(remote_path)}"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        fatal(f"could not fetch {remote_path} from {BOX}: {e}")
        return 2
    if proc.returncode != 0:
        fatal(f"could not read {remote_path} on {BOX}: {proc.stderr.strip()}")
    try:
        spec = json.loads(proc.stdout)
    except Exception as e:  # noqa: BLE001
        fatal(f"{remote_path} is not valid JSON: {e}")
        return 2
    jid = spec.get("id")
    if not jid:
        fatal(f"{remote_path} has no 'id' field")
        return 2
    canonical = generate_canonical()
    if jid not in canonical:
        print(f"ORPHANED: generate_jobs.py produces no spec for id={jid}")
        return 1
    drift = diff_spec(spec, canonical[jid])
    if drift:
        print(f"DRIFT: id={jid}")
        for l in drift:
            print(l)
        return 1
    print(f"CLEAN: id={jid} matches generate_jobs.py today")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("--check-one", metavar="PATH_ON_BOX", default=None,
                     help="single-spec mode (see module docstring)")
    args = ap.parse_args()

    if args.check_one:
        sys.exit(run_check_one(args.check_one))
    sys.exit(run_scan(args.json))


if __name__ == "__main__":
    main()
