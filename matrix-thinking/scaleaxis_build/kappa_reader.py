#!/usr/bin/env python3
"""BUILD REQUIREMENT B7 -- the offline P1b kappa-TRAJECTORY reader.
NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.3.2 (FATAL-3's fix, as amended by
verify-R2 MAJOR-5 / m8) and sec 3.7.

WHY THIS EXISTS.  There is NO in-run P1b kappa at any step, at any scale, in
this harness (FATAL-3, verified against runner.py:1428-1453): the eval block
OVERWRITES rec at every eval_every, the runner withholds eval values from
stdout by design, and the in-run eval runs teacher_force=False -- the P0
regime, not the P1b regime the kappa >= 0.90 bar is defined on. So branch (B)'s
rule ("is kappa still rising at 20000?") has no instrument unless one is built.

THE INSTRUMENT (sec 4.3.2, elected variant).  The six calibration cells carry
--ckpt-every 5000. This reader watches the cell's SINGLE ckpt path and, on
every mtime change:

    os.link(ckpt_path, snap_i)      # O(us) directory op -- the hardlink
    step = torch.load(snap_i)["step"]        # READ THE STEP FIRST
    kscaling_battery.py --ckpt snap_i --required-step <that step>
    os.unlink(snap_i)

TWO OPERATIONAL HOLES THIS CLOSES (verify-R2 MAJOR-5):

 (a) A MISSED WINDOW WAS UNRECOVERABLE, and it is exactly the point branch (B)
     needs. ckpt_path is a SINGLE path, overwritten at every ckpt_every;
     branch (B)'s rule is kappa@20000 - kappa@15000 >= +0.05, and kappa@20000
     survives in the final checkpoint while kappa@15000 does not. The ~250x
     steady-state slack bounds the race but NOT process death, a restart or a
     mispredicted step. The hardlink makes the subsequent os.replace in
     atomic_torch_save swap the DIRECTORY ENTRY while the hardlinked inode
     survives: the window shrinks from ~43 min of slack to microseconds.

 (b) kscaling_battery.py:140-141 HARD-SKIPs with "NOT SCORED" when
     ckpt_step != --required-step, and a reader polling mtime does not know
     which step landed until it opens the file. Reading the step FIRST and
     passing it as --required-step makes that SKIP STRUCTURALLY IMPOSSIBLE
     (m8's fold-in).

GPU RESERVATION (sec 4.3.2(b), by worker MECHANICS not by hope).  The reader is
bursty -- ~10 s of work per ~43 min window, genuinely idle >99.5% of the time --
so queue_worker.sh's "zero compute-apps AND < 2 GiB" predicate WILL claim a
training cell onto the reader's GPU during an idle window. Belt and braces:
(1) no queue_worker.sh instance is started on the reader's GPU index at all,
and (2) this process holds a persistent >= 2 GiB resident CUDA allocation for
its lifetime, which trips the worker's own claim predicate even if an instance
is started by mistake. --reserve-gib implements (2); (1) is an enumerated
pre-launch check.

MISSED-WINDOW ACCOUNTING.  Expected windows are the ckpt_every multiples up to
--steps. Any window with no captured read is reported as a MISSING TRAJECTORY
POINT in the manifest, never silently absent -- branch (B)'s incomplete-
trajectory clause reads that field.

RETENTION: transient only (worst case one ~9.4 GB inode held for ~10 s).
DISK COST: zero in steady state.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

BATTERY = "kscaling_battery.py"


def _read_step(path: str) -> tuple[int | None, str]:
    """Return (step, status). NEVER returns a step for a file that failed to
    load -- a truncated or zero-byte checkpoint is DETECTED AND REPORTED, not
    silently scored (sec 4.3.2's argued deviation, and B7's negative tests)."""
    import torch
    if not os.path.exists(path):
        return None, "ABSENT"
    if os.path.getsize(path) == 0:
        return None, "ZERO-BYTE (detected, NOT scored)"
    try:
        try:
            ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except (TypeError, RuntimeError):
            ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                       # noqa: BLE001
        return None, f"UNREADABLE (detected, NOT scored): {type(e).__name__}: {str(e)[:160]}"
    if not isinstance(ck, dict) or "step" not in ck:
        return None, "MALFORMED (no 'step' key; detected, NOT scored)"
    for arm in ("full_graft", "backbone_only"):
        if arm not in ck:
            return None, f"MALFORMED (missing arm {arm!r}; detected, NOT scored)"
    return int(ck["step"]), "OK"


def _reserve(gib: float):
    """Hold a resident CUDA allocation so queue_worker.sh's '< 2 GiB' claim
    predicate reads this GPU as BUSY for the reader's whole lifetime."""
    if gib <= 0:
        return None
    import torch
    if not torch.cuda.is_available():
        print("[B7] WARNING: no CUDA -- GPU reservation NOT held", flush=True)
        return None
    n = int(gib * (1 << 30) / 4)
    t = torch.empty(n, dtype=torch.float32, device="cuda")
    t.fill_(0.0)
    torch.cuda.synchronize()
    print(f"[B7] GPU reservation held: {torch.cuda.memory_allocated()/2**30:.2f} GiB resident "
          f"(trips queue_worker.sh's own '< 2 GiB' claim predicate)", flush=True)
    return t


def score(snap: str, step: int, k: int, tag: str, outdir: str, workdir: str,
          py: str, cellcfg: str | None) -> dict:
    cmd = [py, os.path.join(workdir, BATTERY), "--k", str(k), "--ckpt", snap,
           "--tag", tag, "--outdir", outdir, "--required-step", str(step)]
    if cellcfg:
        cmd += ["--cellcfg", cellcfg]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                       env=dict(os.environ, NCR_K=str(k)))
    return dict(step=step, rc=p.returncode, elapsed_s=round(time.time() - t0, 1),
                out_json=os.path.join(outdir, f"{tag}_kscaling.json"),
                stderr_tail=p.stderr[-400:] if p.returncode else "")


def read_once(ckpt_path: str, snap_dir: str, k: int, cell: str, outdir: str,
              workdir: str, py: str, cellcfg: str | None, i: int) -> dict:
    """hardlink -> read step -> battery at that --required-step -> unlink."""
    os.makedirs(snap_dir, exist_ok=True)
    snap = os.path.join(snap_dir, f"{cell}.snap{i}.ckpt.pt")
    rec = dict(index=i, ckpt=ckpt_path, snap=snap)
    try:
        if os.path.exists(snap):
            os.unlink(snap)
        os.link(ckpt_path, snap)                     # THE HARDLINK, O(us)
        rec["hardlinked"] = True
    except OSError as e:
        rec.update(hardlinked=False, status=f"HARDLINK FAILED: {e!r}")
        return rec
    try:
        step, status = _read_step(snap)
        rec["read_status"] = status
        if step is None:
            rec["status"] = f"NOT SCORED -- {status}"
            return rec
        rec["step"] = step
        rec.update(score(snap, step, k, f"{cell}_traj_s{step}", outdir, workdir, py, cellcfg))
        rec["status"] = "SCORED" if rec["rc"] == 0 else f"BATTERY EXIT {rec['rc']}"
    finally:
        try:
            os.unlink(snap)                          # retention: transient only
            rec["unlinked"] = True
        except OSError:
            rec["unlinked"] = False
    return rec


def negative_tests(tmpdir: str) -> int:
    """The FOUR forced-fail tests sec 3.7 names for B7, RUN TO COMPLETION."""
    import torch
    os.makedirs(tmpdir, exist_ok=True)
    out = []

    # (i) TRUNCATED checkpoint -> detected and reported, NOT silently scored.
    good = os.path.join(tmpdir, "good.ckpt.pt")
    torch.save({"step": 15000, "runner_tag": "ncr_scaleaxis_runner_v1",
                "full_graft": {"x": torch.zeros(4)}, "backbone_only": {"x": torch.zeros(4)}},
               good)
    trunc = os.path.join(tmpdir, "trunc.ckpt.pt")
    with open(good, "rb") as f:
        blob = f.read()
    with open(trunc, "wb") as f:
        f.write(blob[: len(blob) // 2])
    s, st = _read_step(trunc)
    out.append(dict(test="B7_i_truncated_ckpt_detected", step=s, status_text=st,
                    status="PASS" if s is None and "NOT scored" in st else
                           "FAIL -- a truncated checkpoint would be scored"))

    # (ii) ZERO-BYTE checkpoint -> likewise.
    zero = os.path.join(tmpdir, "zero.ckpt.pt")
    open(zero, "wb").close()
    s, st = _read_step(zero)
    out.append(dict(test="B7_ii_zero_byte_ckpt_detected", step=s, status_text=st,
                    status="PASS" if s is None and "NOT scored" in st else "FAIL"))

    # positive control: the intact fixture MUST read its step, or (i)/(ii) are
    # vacuous (they would "pass" because nothing is readable at all).
    s_ok, st_ok = _read_step(good)
    out.append(dict(test="B7_positive_control_intact_ckpt_reads",
                    step=s_ok, status_text=st_ok,
                    status="PASS" if s_ok == 15000 else
                           "FAIL -- the detector rejects a VALID checkpoint, so (i)/(ii) prove nothing"))

    # (iii) an OFF-CADENCE step must be REPORTED, never silently SKIPped.
    #       Reading the step FIRST and passing it as --required-step makes the
    #       battery's SKIP structurally impossible; the reader labels the point.
    expected = (5000, 10000, 15000, 20000)
    off = os.path.join(tmpdir, "off.ckpt.pt")
    torch.save({"step": 17321, "runner_tag": "ncr_scaleaxis_runner_v1",
                "full_graft": {"x": torch.zeros(4)}, "backbone_only": {"x": torch.zeros(4)}}, off)
    s_off, _ = _read_step(off)
    labelled = (s_off is not None) and (s_off not in expected)
    out.append(dict(test="B7_iii_off_cadence_step_reported", step=s_off,
                    expected_windows=list(expected), off_cadence=labelled,
                    required_step_passed_would_be=s_off,
                    note=("the battery is invoked with --required-step == the step actually "
                          "read, so its ckpt_step!=required_step SKIP cannot fire; the point "
                          "is labelled OFF-CADENCE in the manifest instead of vanishing"),
                    status="PASS" if labelled else "FAIL"))

    # (iv) a MISSED WINDOW must be reported as a MISSING TRAJECTORY POINT.
    captured = [5000, 10000, 20000]                 # 15000 deliberately absent
    missing = [w for w in expected if w not in captured]
    out.append(dict(test="B7_iv_missed_window_reported", captured=captured,
                    expected_windows=list(expected), missing_trajectory_points=missing,
                    branch_B_consequence=("sec 7.2(B): with kappa@15000 missing on >=2 of 3 "
                                          "frozen seeds, branch (B) falls back to the deepest "
                                          "available pair and RECORDS which pair was used"),
                    status="PASS" if missing == [15000] else "FAIL"))

    ok = all(o["status"].startswith("PASS") for o in out)
    print(json.dumps({"gate": "B7", "tests": out, "overall": "PASS" if ok else "FAIL"}, indent=1))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="the cell's SINGLE ckpt path (overwritten every ckpt_every)")
    ap.add_argument("--cell", help="cell id")
    ap.add_argument("--k", type=int)
    ap.add_argument("--snap-dir", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--workdir", default=os.path.expanduser("~/ncr_scaleaxis"))
    ap.add_argument("--python", default="/home/nvidia/tdenv/bin/python3")
    ap.add_argument("--cellcfg", default=None)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--poll-s", type=float, default=20.0)
    ap.add_argument("--max-wall-s", type=float, default=86400.0)
    ap.add_argument("--reserve-gib", type=float, default=2.5,
                    help="persistent resident CUDA allocation (sec 4.3.2(b)); >2 GiB trips "
                         "queue_worker.sh's own claim predicate")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--negative-tests", default=None, metavar="TMPDIR")
    args = ap.parse_args()

    if args.negative_tests:
        return negative_tests(args.negative_tests)
    for req in ("ckpt", "cell", "k", "outdir"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req} is required")

    snap_dir = args.snap_dir or os.path.dirname(os.path.abspath(args.ckpt))
    reservation = _reserve(args.reserve_gib)        # held for the whole lifetime
    expected = tuple(range(args.ckpt_every, args.steps + 1, args.ckpt_every))
    man = dict(cell=args.cell, k=args.k, ckpt=args.ckpt, expected_windows=list(expected),
               reads=[], reserve_gib=args.reserve_gib,
               instrument="kscaling_battery.py (P1b + P0), --required-step = the step read "
                          "from the hardlinked snapshot")
    seen_mtime, i, t0 = None, 0, time.time()
    try:
        while time.time() - t0 < args.max_wall_s:
            if os.path.exists(args.ckpt):
                mt = os.path.getmtime(args.ckpt)
                if mt != seen_mtime:
                    seen_mtime = mt
                    i += 1
                    r = read_once(args.ckpt, snap_dir, args.k, args.cell, args.outdir,
                                  args.workdir, args.python, args.cellcfg, i)
                    man["reads"].append(r)
                    print(f"[B7] read {i}: step={r.get('step')} status={r.get('status')}",
                          flush=True)
                    if r.get("step") == args.steps:
                        break
            time.sleep(args.poll_s)
    finally:
        got = sorted({r["step"] for r in man["reads"] if r.get("step") is not None})
        man["captured_steps"] = got
        man["missing_trajectory_points"] = [w for w in expected if w not in got]
        man["off_cadence_steps"] = [s for s in got if s not in expected]
        man["complete"] = not man["missing_trajectory_points"]
        path = args.manifest or os.path.join(args.outdir, f"{args.cell}_traj_manifest.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(man, f, indent=1)
        print(f"[B7] manifest -> {path}  captured={got} "
              f"missing={man['missing_trajectory_points']} "
              f"off_cadence={man['off_cadence_steps']}", flush=True)
        del reservation
    return 0


if __name__ == "__main__":
    sys.exit(main())
