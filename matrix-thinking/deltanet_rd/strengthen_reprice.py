#!/usr/bin/env python
"""strengthen_reprice.py -- the sec 1.46 RE-PRICE RULE, standalone and
stdlib-only (an on-box watcher calls this with no project imports and no
torch, deliberately -- see the interface note below).

Reads the 0599 gating probe's raw + re-metric JSONs (300 steps of the
single most expensive STAGED cell, `h2h_strengthen_C2_lr1e-03_st60000_s0`)
and:
  1. computes r_C2 (measured s/step for C2) from the probe's own raw
     `wall_s`,
  2. derives r_C1 by log-space interpolation between the PINNED r_C0 and
     the MEASURED r_C2 (the exponent, 0.213, is
     log(block_flop_ratio(C1))/log(block_flop_ratio(C2)) from
     `h2h_strengthen_specs_gen.py`'s own model -- pinned here as a
     constant, not re-derived, so this script never needs that module),
  3. re-prices EVERY (capacity, steps) shape among the 27 STAGED cells
     (the `strengthen_specs_deferred/` cells are NOT re-priced or
     counted here -- they are not staged),
  4. checks r_C2 and the derived per-config rates against the
     PRE-REGISTERED kill thresholds, and
  5. prints a final `REPRICE: PASS` (exit 0) or `REPRICE: STOP <reason>`
     (exit 1) line an on-box watcher can act on mechanically.

INTERFACE (pinned, coordinator 2026-09-01):
  python3 strengthen_reprice.py --probe-dir <dir containing the 0599 raw
      JSON and remetric/ subdir> [--joblog <path to
      /home/nvidia/queue/logs/0599_h2h_strengthen_probe_C2.log>]

The 0599 spec (`strengthen_specs_probe/0599_h2h_strengthen_probe_C2.json`)
writes:
  raw:      <probe-dir>/h2h_strengthen_C2_lr1e-03_st60000_s0.json
  remetric: <probe-dir>/remetric/h2h_strengthen_C2_lr1e-03_st60000_s0_round4.json
`--probe-dir` is that spec's own `output_dir`
(`/home/nvidia/chapter2/deltanet_rd/results/h2h_rung1/strengthen_probe`
on box); these two filenames are pinned constants below
(`PROBE_RAW_FILENAME`/`PROBE_REMETRIC_FILENAME`), matching
`h2h_strengthen_specs_gen.PROBE_RAW_FILENAME`/`PROBE_REMETRIC_FILENAME`
byte-for-byte (kept in sync by hand -- this script deliberately does not
import that module; a selftest in `h2h_strengthen_rd.py` cross-checks the
two copies do not drift).

RE-PRICE RULE (verbatim, HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.46):
  r_C2 = probe raw wall_s / 300
  C2@60k = 60000*r_C2 + O,  C2@20k = 20000*r_C2 + O
  O = job-log START->END - train wall_s - remetric wall_s
  r_C1 = r_C0 * (r_C2/r_C0)^0.213, with r_C0 = 0.0472 s/step
  kill thresholds: C2@60k r>0.720, C2@20k r>0.900, C1@60k r>0.240,
                   C1@20k r>0.270, C0@60k r>0.180
  if r_C2 > 0.51: regenerate TIMEOUT_HOURS = 2x re-priced before staging

The verbatim rule prices ONLY the training-time term explicitly (steps*r
+ O); re-metric is NOT step-count-scaled (a fixed per-episode-set cost)
and is NOT free, so this script adds it back on top, scaled from the
probe's own measured re-metric `wall_s` by the SAME r_cap/r_C2 ratio used
for training -- a documented, disclosed completion of the verbatim rule,
not a silent deviation from it (see `_remetric_component` below). The
per-config totals and the grand total this script prints ALWAYS include
this addition; a reader who wants the bare verbatim-rule number can
recover it by subtracting the printed remetric-only line.

Exit code: 0 (REPRICE: PASS) or 1 (REPRICE: STOP <reason>).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

# --- pinned constants (RE-PRICE RULE, sec 1.46 -- verbatim numbers, never re-derived here) ---
R_C0 = 0.0472                      # s/step, pinned (fast-cluster-anchor-equivalent for C0)
C1_INTERP_EXPONENT = 0.213         # log(block_flop_ratio(C1)) / log(block_flop_ratio(C2)),
                                    # pinned as a constant so this script needs no project import
R_C2_REGEN_THRESHOLD = 0.51        # s/step -- above this, TIMEOUT_HOURS must be regenerated
                                    # (= 2x re-priced) before ANY staged spec is dispatched

# kill thresholds, s/step -- EXACTLY TIMEOUT_HOURS[(cap,steps)]*3600/steps for the pinned
# round-1 timeout table (h2h_strengthen_specs_gen.TIMEOUT_HOURS): the rate at which a cell's
# OWN currently-configured timeout would fire before it legitimately finishes.
KILL_THRESHOLDS = {
    ("C2", 60_000): 0.720,   # TIMEOUT_HOURS[C2,60000]=12h -> 12*3600/60000
    ("C2", 20_000): 0.900,   # TIMEOUT_HOURS[C2,20000]=5h  ->  5*3600/20000
    ("C1", 60_000): 0.240,   # TIMEOUT_HOURS[C1,60000]=4h  ->  4*3600/60000
    ("C1", 20_000): 0.270,   # TIMEOUT_HOURS[C1,20000]=1.5h-> 1.5*3600/20000
    ("C0", 60_000): 0.180,   # TIMEOUT_HOURS[C0,60000]=3h  ->  3*3600/60000
}

# the 27 STAGED (capacity, steps) -> cell-count shapes (strengthen_specs/, AFTER the round-2
# TRIM moved C2 x lr=3e-4 x 60,000 -- specs 0621-0623 -- to strengthen_specs_deferred/). Pinned
# here rather than imported (this script is stdlib-only, no project imports, no torch).
STAGED_CONFIG_CELL_COUNTS = {
    ("C0", 60_000): 6,   # lr in {1e-3, 3e-4} x 3 seeds
    ("C1", 20_000): 6,
    ("C1", 60_000): 6,
    ("C2", 20_000): 6,
    ("C2", 60_000): 3,   # lr=3e-4 x 60,000 deferred -- only lr=1e-3's 3 seeds remain staged
}
assert sum(STAGED_CONFIG_CELL_COUNTS.values()) == 27

PROBE_RAW_FILENAME = "h2h_strengthen_C2_lr1e-03_st60000_s0.json"
PROBE_REMETRIC_FILENAME = "h2h_strengthen_C2_lr1e-03_st60000_s0_round4.json"
PROBE_STEPS = 300

_JOBLOG_START_RE = re.compile(r"=== job \S+ START (\S+) on GPU \d+ ===")
_JOBLOG_END_RE = re.compile(r"=== job \S+ cmd exit=-?\d+ END (\S+) ===")
_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"   # queue_worker.sh's own `date -u +%FT%TZ` format


def _parse_iso(ts: str):
    return datetime.strptime(ts, _ISO_FMT).replace(tzinfo=timezone.utc)


def compute_overhead_from_joblog(joblog_path: str, train_wall_s: float,
                                  remetric_wall_s: float) -> tuple[float, str]:
    """O = job-log START->END - train wall_s - remetric wall_s. Returns (O, note). O defaults to
    0.0 with an explicit WARNING note (never a silent default) if the log is missing or its
    START/END markers cannot be parsed -- `queue_worker.sh`'s own per-job log format
    (`=== job <id> START <ISO8601> on GPU <N> ===` ... `=== job <id> cmd exit=<rc> END
    <ISO8601> ===`)."""
    if not joblog_path or not os.path.isfile(joblog_path):
        return 0.0, f"WARNING: no joblog at {joblog_path!r} -- O defaulted to 0.0 (not measured)"
    text = open(joblog_path).read()
    m_start, m_end = _JOBLOG_START_RE.search(text), _JOBLOG_END_RE.search(text)
    if not m_start or not m_end:
        return 0.0, ("WARNING: joblog found but START/END markers not both matched -- "
                     "O defaulted to 0.0 (not measured)")
    try:
        total = (_parse_iso(m_end.group(1)) - _parse_iso(m_start.group(1))).total_seconds()
    except ValueError as e:
        return 0.0, f"WARNING: joblog timestamps unparseable ({e}) -- O defaulted to 0.0"
    o = total - train_wall_s - remetric_wall_s
    return o, f"O={o:.2f}s (joblog total={total:.2f}s - train={train_wall_s:.2f}s - remetric={remetric_wall_s:.2f}s)"


def _remetric_component(r_cap: float, r_c2: float, probe_remetric_wall_s: float) -> float:
    """Re-metric is a fixed, forward-only, per-episode-set cost -- NOT step-count-scaled -- so it
    is scaled from the probe's own measured re-metric wall_s by the same r_cap/r_C2 ratio used
    for training (a documented completion of the verbatim RE-PRICE RULE, which prices training
    only; see the module docstring)."""
    return probe_remetric_wall_s * (r_cap / r_c2)


def reprice(probe_dir: str, joblog_path: str | None) -> dict:
    raw_path = os.path.join(probe_dir, PROBE_RAW_FILENAME)
    remetric_path = os.path.join(probe_dir, "remetric", PROBE_REMETRIC_FILENAME)
    # light integrity check: exactly the ONE expected remetric JSON should be in remetric/ --
    # a stray second file (e.g. a differently-named re-run) would silently go unread otherwise.
    remetric_glob = sorted(glob.glob(os.path.join(probe_dir, "remetric", "*_round4.json")))
    if remetric_glob and remetric_glob != [remetric_path]:
        print(f"WARNING: remetric/ contains unexpected file(s) beyond {PROBE_REMETRIC_FILENAME}: "
             f"{[os.path.basename(p) for p in remetric_glob]}")
    with open(raw_path) as f:
        raw = json.load(f)
    with open(remetric_path) as f:
        rem = json.load(f)
    if raw.get("step_count") != PROBE_STEPS:
        raise SystemExit(f"REPRICE: STOP probe raw step_count={raw.get('step_count')} != "
                         f"expected {PROBE_STEPS} -- wrong/stale probe artifact")

    train_wall_s = float(raw["wall_s"])
    remetric_wall_s = float(rem["wall_s"])
    r_c2 = train_wall_s / PROBE_STEPS
    o, o_note = compute_overhead_from_joblog(joblog_path, train_wall_s, remetric_wall_s)
    r_c1 = R_C0 * (r_c2 / R_C0) ** C1_INTERP_EXPONENT
    rates = {"C0": R_C0, "C1": r_c1, "C2": r_c2}

    per_config = {}
    total_gpu_h = 0.0
    for (cap, steps), n_cells in STAGED_CONFIG_CELL_COUNTS.items():
        r_cap = rates[cap]
        train_s = steps * r_cap + o
        remetric_s = _remetric_component(r_cap, r_c2, remetric_wall_s)
        per_cell_gpu_h = (train_s + remetric_s) / 3600.0
        config_gpu_h = n_cells * per_cell_gpu_h
        per_config[f"{cap}@{steps}"] = {
            "n_cells": n_cells, "r_s_per_step": r_cap, "train_s_per_cell": train_s,
            "remetric_s_per_cell": remetric_s, "gpu_h_per_cell": per_cell_gpu_h,
            "gpu_h_config_total": config_gpu_h,
        }
        total_gpu_h += config_gpu_h

    reasons = []
    if r_c2 > R_C2_REGEN_THRESHOLD:
        reasons.append(f"r_C2={r_c2:.4f} s/step > {R_C2_REGEN_THRESHOLD} -- regenerate "
                       "TIMEOUT_HOURS = 2x re-priced before staging")
    for (cap, steps), threshold in KILL_THRESHOLDS.items():
        r_cap = rates[cap]
        if r_cap > threshold:
            reasons.append(f"{cap}@{steps}: projected rate {r_cap:.4f} s/step exceeds its own "
                           f"timeout-implied ceiling {threshold} s/step -- the currently-set "
                           "timeout would kill this cell before it legitimately finishes")

    return {"r_C0": R_C0, "r_C1": r_c1, "r_C2": r_c2, "O": o, "O_note": o_note,
           "train_wall_s": train_wall_s, "remetric_wall_s": remetric_wall_s,
           "per_config": per_config, "total_gpu_h_27_staged": total_gpu_h,
           "stop_reasons": reasons, "pass": not reasons}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True,
                    help="dir containing the 0599 raw JSON and remetric/ subdir")
    ap.add_argument("--joblog", default=None,
                    help="path to /home/nvidia/queue/logs/0599_h2h_strengthen_probe_C2.log "
                         "(optional -- O defaults to 0.0 with a WARNING if omitted/unparseable)")
    args = ap.parse_args()

    try:
        result = reprice(args.probe_dir, args.joblog)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"REPRICE: STOP could not price the probe artifacts ({type(e).__name__}: {e})")
        return 1

    print(f"r_C0 = {result['r_C0']:.4f} s/step (pinned)")
    print(f"r_C1 = {result['r_C1']:.4f} s/step (interpolated, exponent={C1_INTERP_EXPONENT})")
    print(f"r_C2 = {result['r_C2']:.4f} s/step (measured: {result['train_wall_s']:.2f}s / "
         f"{PROBE_STEPS} steps)")
    print(result["O_note"])
    print("\nPer-config re-priced GPU-h (27 staged cells; re-metric added on top of the "
         "verbatim train-only formula, see module docstring):")
    for key, v in sorted(result["per_config"].items()):
        print(f"  {key}: n_cells={v['n_cells']} r={v['r_s_per_step']:.4f}s/step "
             f"train/cell={v['train_s_per_cell']:.1f}s remetric/cell={v['remetric_s_per_cell']:.1f}s "
             f"gpu_h/cell={v['gpu_h_per_cell']:.4f} config_total={v['gpu_h_config_total']:.4f}")
    print(f"\nTOTAL re-priced GPU-h (27 staged cells): {result['total_gpu_h_27_staged']:.3f}")

    if result["pass"]:
        print("\nREPRICE: PASS")
        return 0
    print("\nREPRICE: STOP " + "; ".join(result["stop_reasons"]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
