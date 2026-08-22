#!/usr/bin/env python3
"""BUILD REQUIREMENT B6 -- the CPU-only RATE BREAKER.
NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.6 (as corrected by verify-R2 MAJOR-7)
and sec 3.7.

WHAT IT WATCHES, AND WHY NOT THE RESULTS JSON.  R1 pointed this watcher at
`atomic_write_json(out_path, rec)` and claimed "blind discipline is preserved
by construction". It is not: that same `rec` carries `rec["arms"]` -- the full
two-arm eval result -- and `rec["attribution"]`, so any `json.load(out_path)`
MATERIALIZES EVERY PROTECTED VALUE IN THE WATCHER'S PROCESS. The blind would
then be preserved only by this file declining to print them: one stray debug
print, one traceback that reprs the parsed dict, and the blind breaks mid-run
on the calibration cells.

A provably-blind source already exists and is 40x faster.
`ncr_lm_wave1_runner.py:299` sets LOG_EVERY = 25 and :1403-1426 prints

  [{cell_id}] step {step}/{steps}  full_graft_loss=..  backbone_only_loss=..  lr=..  {elapsed:.0f}s..

carrying `step` and `elapsed` and NO eval metric -- the runner's own comment
classes the loss terms as "operational telemetry (liveness/divergence), never
an eval metric". Parsing that line is blind-safe in the literal, STRUCTURAL
sense, at a 25-step cadence instead of 1000.

THREE STRUCTURAL BLINDS, not three promises:
  1. The watcher's only readable input is a .log path. It REFUSES a path
     ending in .json, and contains no code that opens the results record.
  2. The regex captures EXACTLY three integers (step, steps, elapsed). The
     rest of the line is discarded before anything is stored -- the loss
     values are never bound to a name.
  3. Every line this process emits goes through _blind(), which HARD-FAILS on
     any eval-metric token. B6's negative test (ii) asserts that no
     eval-metric key ever appears in watcher output or logs.

THE RULE (sec 3.6, pinned): elapsed/step > 1.5 x calibrated_contended_s_per_step
on TWO CONSECUTIVE LOG_EVERY lines  =>  raise the cell's STOP file
(runner.py:1454-1458 saves a checkpoint and exits 3 cleanly). Fires at ~0.1% of
a cell instead of the --ceiling-gpuh backstop's ~150%. The two breakers have
distinct jobs and BOTH are adopted (sec 3.6, argued deviation (a), ratified).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

LOG_EVERY = 25          # runner.py:299 -- the cadence this watcher is keyed to
BREAKER_MULT = 1.5      # sec 3.6's pinned rule
CONSECUTIVE = 2         # two consecutive LOG_EVERY lines

# runner.py:1403-1426's format string, matched on the three fields we take and
# nothing else. `.*?` swallows the loss terms WITHOUT capturing them.
STEP_RE = re.compile(r"\]\s+step\s+(\d+)/(\d+)\s+.*?\s(\d+)s(?:\s|$)")

# B6 negative test (ii): no eval-metric token may EVER reach this process's
# output. Sourced from the actual key names in rec["arms"]/rec["attribution"].
BANNED_TOKENS = (
    "retrieval24", "recovered_frac", "margin_over_chance", "per_hop", "attribution",
    "arms", "kappa", "acc=", "accuracy", "primary_signal", "target_pairwise_cos",
    "discrimin", "cos_all", "teacher_force_check", "deep_gap", "P1b", "eval_result",
)


class BlindViolation(RuntimeError):
    pass


def _blind(s: str) -> str:
    low = s.lower()
    for t in BANNED_TOKENS:
        if t.lower() in low:
            raise BlindViolation(
                f"B6 BLIND VIOLATION: the watcher tried to emit a line containing the "
                f"eval-metric token {t!r}. Refusing. (line withheld)")
    return s


def emit(s: str) -> None:
    print(_blind(s), flush=True)


def parse_line(line: str):
    """-> (step, steps, elapsed_s) or None. Captures three integers; the loss
    terms in between are matched non-greedily and never bound."""
    m = STEP_RE.search(line)
    if not m:
        return None
    step, steps, elapsed = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if step <= 0:
        return None
    return step, steps, elapsed


def run(log_path: str, stop_file: str, calibrated_s_per_step: float,
        poll_s: float, max_wall_s: float, dry_run: bool) -> dict:
    if log_path.endswith(".json"):
        raise SystemExit(
            "B6 REFUSES a .json input. This watcher reads the runner's STDOUT LOG only; "
            "the results JSON carries rec['arms'] and rec['attribution'] and opening it "
            "would materialize every protected value in this process (verify-R2 MAJOR-7).")
    bar = BREAKER_MULT * calibrated_s_per_step
    emit(f"[B6] watching {log_path}")
    emit(f"[B6] calibrated_contended_s_per_step={calibrated_s_per_step:.6f} "
         f"bar={bar:.6f} (x{BREAKER_MULT}) consecutive={CONSECUTIVE} log_every={LOG_EVERY}")

    consec, last_step, t0, pos = 0, 0, time.time(), 0
    rec = dict(log=log_path, stop_file=stop_file, bar_s_per_step=bar,
               calibrated_s_per_step=calibrated_s_per_step, breaker_mult=BREAKER_MULT,
               consecutive_required=CONSECUTIVE, samples=[], tripped=False,
               stop_raised=False, dry_run=dry_run)
    while time.time() - t0 < max_wall_s:
        if os.path.exists(log_path):
            with open(log_path, "r", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.splitlines():
                p = parse_line(line)
                if p is None:
                    continue
                step, steps, elapsed = p
                if step <= last_step:
                    continue
                last_step = step
                rate = elapsed / step
                rec["samples"].append([step, elapsed, round(rate, 6)])
                over = rate > bar
                consec = consec + 1 if over else 0
                emit(f"[B6] step={step}/{steps} elapsed={elapsed}s rate={rate:.6f} "
                     f"over_bar={over} consecutive={consec}")
                if consec >= CONSECUTIVE:
                    rec["tripped"] = True
                    rec["tripped_at_step"] = step
                    rec["tripped_rate"] = rate
                    emit(f"[B6] BREAKER TRIPPED at step {step}: rate {rate:.6f} > bar "
                         f"{bar:.6f} on {CONSECUTIVE} consecutive LOG_EVERY lines")
                    if not dry_run:
                        os.makedirs(os.path.dirname(os.path.abspath(stop_file)) or ".",
                                    exist_ok=True)
                        with open(stop_file, "w") as f:
                            f.write(f"B6 rate breaker: step={step} rate={rate:.6f} > "
                                    f"{bar:.6f}\n")
                        rec["stop_raised"] = True
                        emit(f"[B6] STOP file raised: {stop_file} "
                             f"(runner saves a checkpoint and exits 3)")
                    return rec
                if step >= steps:
                    emit("[B6] cell reached its final step -- watcher exiting clean")
                    return rec
        time.sleep(poll_s)
    emit("[B6] max wall reached -- watcher exiting without tripping")
    return rec


def negative_tests(tmpdir: str) -> int:
    """The two forced-fail tests sec 3.7 names for B6, RUN TO COMPLETION."""
    os.makedirs(tmpdir, exist_ok=True)
    out = []

    # (i) a synthetic log with a DOUBLED elapsed must trip the breaker.
    good, bad = os.path.join(tmpdir, "good.log"), os.path.join(tmpdir, "bad.log")
    stop = os.path.join(tmpdir, "STOP")
    nominal = 0.60           # s/step
    def line(step, el):
        return (f"[cell_x] step {step}/20000  full_graft_loss=4.1234 "
                f"backbone_only_loss=4.5678  lr=3.00e-04  {int(el)}s\n")
    with open(good, "w") as f:
        for i in range(1, 9):
            f.write(line(i * LOG_EVERY, i * LOG_EVERY * nominal))
    with open(bad, "w") as f:
        for i in range(1, 9):
            mult = 2.0 if i >= 6 else 1.0        # doubled from the 6th line on
            f.write(line(i * LOG_EVERY, i * LOG_EVERY * nominal * mult))

    r_good = run(good, stop, nominal, poll_s=0.01, max_wall_s=3, dry_run=True)
    r_bad = run(bad, stop, nominal, poll_s=0.01, max_wall_s=3, dry_run=True)
    out.append(dict(test="B6_i_doubled_elapsed_trips",
                    nominal_log_tripped=r_good["tripped"],
                    doubled_log_tripped=r_bad["tripped"],
                    tripped_at_step=r_bad.get("tripped_at_step"),
                    status="PASS" if (r_bad["tripped"] and not r_good["tripped"]) else
                           "FAIL -- the breaker is vacuous or fires on a nominal log"))

    # (ii) NO eval-metric key may ever appear in watcher output or logs.
    #      Feed a log line carrying a real eval key and prove _blind() FIRES.
    poisoned = ("[cell_x] step 25/20000  full_graft_loss=4.1  backbone_only_loss=4.5  "
                "lr=3.00e-04  15s  retrieval24_acc=0.9958")
    fired, msg = False, ""
    try:
        _blind(f"[B6] echo: {poisoned}")
    except BlindViolation as e:
        fired, msg = True, str(e)[:200]
    # and prove the PARSER discards it: the parse must still yield only 3 ints
    p = parse_line(poisoned)
    leaked = p is not None and len(p) == 3 and all(isinstance(x, int) for x in p)
    out.append(dict(test="B6_ii_no_eval_metric_in_output",
                    blind_filter_fired=fired, blind_message=msg,
                    parser_returns_only_three_ints=leaked, parsed=list(p) if p else None,
                    refuses_json_input=True,
                    status="PASS" if (fired and leaked) else
                           "FAIL -- an eval metric could reach watcher output"))

    # (iii) belt: a .json input is REFUSED outright.
    refused = False
    try:
        run(os.path.join(tmpdir, "x.json"), stop, nominal, 0.01, 1, True)
    except SystemExit:
        refused = True
    out.append(dict(test="B6_iii_refuses_results_json_input", refused=refused,
                    status="PASS" if refused else "FAIL"))

    ok = all(o["status"].startswith("PASS") for o in out)
    print(json.dumps({"gate": "B6", "tests": out,
                      "overall": "PASS" if ok else "FAIL"}, indent=1))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="the runner's STDOUT log (NEVER the results JSON)")
    ap.add_argument("--stop-file")
    ap.add_argument("--calibrated-s-per-step", type=float,
                    help="contended s/step from Stage A0 (sec 4.0)")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--max-wall-s", type=float, default=86400.0)
    ap.add_argument("--dry-run", action="store_true", help="never raise the STOP file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--negative-tests", default=None, metavar="TMPDIR")
    args = ap.parse_args()

    if args.negative_tests:
        return negative_tests(args.negative_tests)

    for req in ("log", "stop_file", "calibrated_s_per_step"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_','-')} is required")
    rec = run(args.log, args.stop_file, args.calibrated_s_per_step,
              args.poll_s, args.max_wall_s, args.dry_run)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
