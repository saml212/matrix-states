"""reasoning_link_gate_enforce.py -- closes REASONING_LINK_DESIGN.md sec
15.2's own DISCREPANCY finding for the Phase-1b Stage-0-gate-only re-run
(sec 16.1.2): "the launch script did not enforce Stage 0's own registered
abort gate" -- `reasoning_link_chain.sh` computed `gate_result_h1_probe_
valid`/`marker_disagreement_flag` at Stage 0 but never READ either field
before letting the full grid launch (grepped directly at the time, zero
hits for either string in that script). This module is the fix for the
Phase-1b re-run: it mechanically READS a `--mode stage0-natural` JSON's own
gate fields and exits NONZERO (refuses) when the probe is PROBE-INVALID --
called from a chain script via a real subprocess exit-code check
(`set -euo pipefail`'s own enforcement, the SAME working pattern the
existing chain already uses for its wall-clock abort check, sec 10 abort
rule 1), never a narrated rule.

Registered gate (sec 8.4/9's h1 sanity floor, unchanged by sec 16.1):
`probe_valid` = null-relative pass AND absolute-0.10-backstop pass, BOTH
required. Reads `per_h["1"]["probe_valid"]` (or the equivalent int key
`per_h[1]`, since a caller may pass either a JSON-round-tripped dict [str
keys] or the in-memory dict [int keys] from reasoning_link_probe.py) --
never recomputes it, so this module can never silently drift from the
probe's own computed verdict.

Run standalone:
    python reasoning_link_gate_enforce.py results/reasoning_link/stage0_natural_A_wikitext.json
    python reasoning_link_gate_enforce.py --selftest   # negative test, run to completion
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def gate_verdict(d: dict) -> tuple[bool, str]:
    """Returns (probe_valid, reason). Mirrors sec 8.4/9's own h1-floor rule
    (BOTH null-relative and absolute conditions must pass) -- reads the
    ALREADY-COMPUTED per_h[1] fields a stage0-natural JSON carries (see
    reasoning_link_probe.py::run_stage0_natural / measure_cell_all_h /
    h1_sanity_floor), never recomputes them from raw numbers."""
    per_h = d.get("per_h", {})
    per_h1 = per_h.get("1", per_h.get(1))
    if per_h1 is None:
        return False, "no per_h[1] entry in JSON -- cannot evaluate the h1 sanity floor at all"
    if "probe_valid" not in per_h1:
        return False, "per_h[1] has no 'probe_valid' key -- JSON does not look like a stage0(-natural) cell"
    probe_valid = bool(per_h1["probe_valid"])
    null_rel = per_h1.get("null_relative_pass")
    abs_p = per_h1.get("absolute_pass")
    if not probe_valid:
        return False, f"h1 sanity floor FAILED (null_relative_pass={null_rel}, absolute_pass={abs_p})"
    return True, f"h1 sanity floor PASSED (null_relative_pass={null_rel}, absolute_pass={abs_p})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true",
                     help="run the negative-test fixture suite (no JSON path needed)")
    args = ap.parse_args(argv)

    if args.selftest:
        ok = _run_selftest()
        return 0 if ok else 1

    if not args.json_path:
        print("usage: reasoning_link_gate_enforce.py <stage0_natural.json> | --selftest", file=sys.stderr)
        return 2

    with open(args.json_path) as f:
        d = json.load(f)
    valid, reason = gate_verdict(d)
    label = d.get("template", "?")
    ckpt = d.get("ckpt_path", "?")
    print(f"[gate-enforce] template={label} ckpt={ckpt}: {reason}")
    if not valid:
        print(f"REFUSE: Phase-1b Stage-0 gate is PROBE-INVALID for {args.json_path} -- per sec 8.4's "
              f"own rule ('failure routes to probe-invalid, not to REFUTE'), this is NOT a licensed "
              f"launch. Halting mechanically (sec 15.2's own DISCREPANCY finding, closed).",
              file=sys.stderr)
        return 1
    print(f"PASS: Phase-1b Stage-0 gate is PROBE-VALID for {args.json_path}.")
    return 0


# ---------------------------------------------------------------------------
# Negative-test fixture suite. Run to completion (CLAUDE.md's own "run the
# negative test to completion, don't just write it" rule) -- both the pure
# `gate_verdict` function AND the subprocess-level exit-code behavior are
# checked, since sec 15.2's own finding was specifically that a COMPUTED
# gate never got READ/ACTED ON by the launching process -- a unit check on
# `gate_verdict` alone would not catch a regression where the CLI wrapper
# itself stopped calling it or stopped honoring its return value.
# ---------------------------------------------------------------------------

def _run_selftest() -> bool:
    ok = True

    valid_fixture = {"template": "A", "ckpt_path": "fake-valid.pt", "per_h": {"1": {
        "probe_valid": True, "null_relative_pass": True, "absolute_pass": True}}}
    invalid_both_fail = {"template": "A", "ckpt_path": "fake-invalid-both.pt", "per_h": {"1": {
        "probe_valid": False, "null_relative_pass": False, "absolute_pass": False}}}
    invalid_partial_fail = {"template": "B", "ckpt_path": "fake-invalid-partial.pt", "per_h": {"1": {
        "probe_valid": False, "null_relative_pass": True, "absolute_pass": False}}}
    missing_per_h1 = {"template": "A", "ckpt_path": "fake-missing.pt", "per_h": {}}

    v, _ = gate_verdict(valid_fixture)
    if v is not True:
        print(f"SELFTEST FAIL: valid fixture returned probe_valid={v} (expected True)")
        ok = False
    v, _ = gate_verdict(invalid_both_fail)
    if v is not False:
        print(f"SELFTEST FAIL: invalid (both conditions fail) fixture returned probe_valid={v} (expected False)")
        ok = False
    v, _ = gate_verdict(invalid_partial_fail)
    if v is not False:
        print(f"SELFTEST FAIL: invalid (partial fail -- probe_valid itself False) fixture "
              f"returned probe_valid={v} (expected False)")
        ok = False
    v, _ = gate_verdict(missing_per_h1)
    if v is not False:
        print(f"SELFTEST FAIL: missing-per_h[1] fixture returned probe_valid={v} (expected False, fail-closed)")
        ok = False

    # End-to-end negative test: write fixtures to REAL temp JSON files, invoke THIS module's own
    # main() via a real subprocess (the exact way a chain script would call it), and assert the
    # exit code -- proves the enforcement has TEETH at the process-boundary level, not merely in
    # the pure function (sec 15.2's own gap was exactly at that boundary: a computed value that
    # was never READ/ACTED ON by the launching process).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        this_file = os.path.abspath(__file__)
        paths = {}
        for name, fixture in (("valid", valid_fixture), ("invalid_both", invalid_both_fail),
                               ("invalid_partial", invalid_partial_fail), ("missing", missing_per_h1)):
            p = os.path.join(td, f"{name}.json")
            with open(p, "w") as f:
                json.dump(fixture, f)
            paths[name] = p

        r_valid = subprocess.run([sys.executable, this_file, paths["valid"]], capture_output=True, text=True)
        if r_valid.returncode != 0:
            print(f"SELFTEST FAIL: subprocess on VALID fixture exited {r_valid.returncode} (expected 0) -- "
                  f"stdout={r_valid.stdout!r} stderr={r_valid.stderr!r}")
            ok = False

        for name in ("invalid_both", "invalid_partial", "missing"):
            r = subprocess.run([sys.executable, this_file, paths[name]], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"SELFTEST FAIL: subprocess on {name!r} fixture exited 0 (expected nonzero) -- "
                      f"the gate has NO TEETH. stdout={r.stdout!r}")
                ok = False
            elif "REFUSE" not in r.stderr:
                print(f"SELFTEST FAIL: {name!r} fixture subprocess exited nonzero ({r.returncode}) but "
                      f"did not print the REFUSE message: stderr={r.stderr!r}")
                ok = False

    if ok:
        print("reasoning_link_gate_enforce --selftest: ALL CHECKS PASSED (pure-function positive + "
              "3 negative fixtures, PLUS subprocess-level exit-code proof for all 4 -- the gate has "
              "teeth at the process boundary sec 15.2's own finding identified as unguarded)")
    return ok


if __name__ == "__main__":
    sys.exit(main())
