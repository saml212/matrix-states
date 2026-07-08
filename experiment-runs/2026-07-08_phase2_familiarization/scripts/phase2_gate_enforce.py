"""phase2_gate_enforce.py -- REASONING_LINK_DESIGN.md sec 16.2.1's
per-checkpoint Stage-0.5-familiarized gate (MAJOR-4 / MAJOR-R3-3), and
sec 16.5 Constraint 1's registered "every gate needs a chain enforcement
branch (gates-must-abort)" requirement, applied here.

Mirrors `reasoning_link_gate_enforce.py`'s own pattern EXACTLY (same
mechanical READ-the-JSON-field-and-refuse discipline, same subprocess-
level negative-test proof) -- that file exists specifically because
REASONING_LINK_DESIGN.md sec 15.2's own DISCREPANCY finding showed a
COMPUTED gate value is not a passed gate unless something actually READS
it and acts on a real process-boundary exit code; this module closes the
exact same failure-mode class for Phase-2's own new per-checkpoint gate,
never re-litigating whether the pattern works (it is the audited,
box-proven reference this BUILD's own obligation (4) names explicitly).

Registered gate (sec 16.2.1's MAJOR-4/MAJOR-R3-3 paragraph): the
familiarized OFF-arm checkpoint's own premises (iii)/(iv) action-rule
gates AND the h1 sanity floor (sec 8.4) must ALL pass, at EVERY trajectory
checkpoint, before that checkpoint's arm-contrast (all 3 arms, that
corpus/seed) is trusted. Reads `per_h["1"]["premise_iii_pass"]`,
`per_h["1"]["premise_iv_pass"]`, `per_h["1"]["probe_valid"]` (or the
equivalent int key `per_h[1]`, mirroring reasoning_link_gate_enforce.py's
own str-vs-int-key tolerance for a JSON-round-tripped dict vs. an
in-memory one) -- NEVER recomputes them, so this module can never
silently drift from `reasoning_link_probe.premise_action_rule`'s /
`h1_sanity_floor`'s own computed verdict.

Run standalone:
    python phase2_gate_enforce.py results/phase2/stage05_gate_off_openr1-mix-ext_s0_c1000.json
    python phase2_gate_enforce.py --selftest   # negative test, run to completion
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def gate_verdict(d: dict) -> tuple[bool, str]:
    """Returns (gate_pass, reason). Mirrors `reasoning_link_gate_enforce.
    gate_verdict`'s own read-only-never-recompute discipline, extended to
    THIS gate's own three-part condition (premises (iii) AND (iv) AND the
    h1 sanity floor -- sec 16.2.1's own "premises (iii)/(iv) and the h1
    floor... are RE-MEASURED... if [either] fail... that checkpoint's
    arm-contrast is uninterpretable")."""
    per_h = d.get("per_h", {})
    per_h1 = per_h.get("1", per_h.get(1))
    if per_h1 is None:
        return False, "no per_h[1] entry in JSON -- cannot evaluate the Stage-0.5 gate at all"
    required = ("premise_iii_pass", "premise_iv_pass", "probe_valid")
    missing = [k for k in required if k not in per_h1]
    if missing:
        return False, f"per_h[1] is missing required field(s) {missing} -- JSON does not look like " \
                       f"a phase2 stage05-gate cell (compute_premises=True + null_seed required)"
    iii = bool(per_h1["premise_iii_pass"])
    iv = bool(per_h1["premise_iv_pass"])
    h1_floor = bool(per_h1["probe_valid"])
    gate_pass = iii and iv and h1_floor
    reason = (f"premise_iii_pass={iii} premise_iv_pass={iv} h1_probe_valid={h1_floor}")
    if not gate_pass:
        return False, f"Stage-0.5 gate FAILED ({reason})"
    return True, f"Stage-0.5 gate PASSED ({reason})"


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
        print("usage: phase2_gate_enforce.py <stage05_gate.json> | --selftest", file=sys.stderr)
        return 2

    with open(args.json_path) as f:
        d = json.load(f)
    gate_pass, reason = gate_verdict(d)
    step = d.get("checkpoint_step", d.get("step", "?"))
    ckpt = d.get("ckpt_path", "?")
    print(f"[phase2-gate-enforce] step={step} ckpt={ckpt}: {reason}")
    if not gate_pass:
        print(f"REFUSE: Phase-2 per-checkpoint Stage-0.5 gate FAILED for {args.json_path} -- per sec "
              f"16.5 Constraint 1's own gates-must-abort rule, this checkpoint's arm-contrast (all 3 "
              f"arms, that corpus/seed) is uninterpretable and MUST be excluded from any reading at "
              f"this step. Halting mechanically, not narrated.", file=sys.stderr)
        return 1
    print(f"PASS: Phase-2 per-checkpoint Stage-0.5 gate PASSED for {args.json_path}.")
    return 0


# ---------------------------------------------------------------------------
# Negative-test fixture suite -- run to completion (CLAUDE.md's "run the negative test to
# completion, don't just write it"), mirroring reasoning_link_gate_enforce._run_selftest's own
# pure-function + subprocess-level exit-code proof, extended to this gate's three-part condition.
# ---------------------------------------------------------------------------

def _run_selftest() -> bool:
    ok = True

    valid_fixture = {"checkpoint_step": 1000, "ckpt_path": "fake-valid.pt", "per_h": {"1": {
        "premise_iii_pass": True, "premise_iv_pass": True, "probe_valid": True}}}
    invalid_iii_fail = {"checkpoint_step": 1000, "ckpt_path": "fake-iii-fail.pt", "per_h": {"1": {
        "premise_iii_pass": False, "premise_iv_pass": True, "probe_valid": True}}}
    invalid_iv_fail = {"checkpoint_step": 1000, "ckpt_path": "fake-iv-fail.pt", "per_h": {"1": {
        "premise_iii_pass": True, "premise_iv_pass": False, "probe_valid": True}}}
    invalid_h1_fail = {"checkpoint_step": 1000, "ckpt_path": "fake-h1-fail.pt", "per_h": {"1": {
        "premise_iii_pass": True, "premise_iv_pass": True, "probe_valid": False}}}
    invalid_all_fail = {"checkpoint_step": 2500, "ckpt_path": "fake-all-fail.pt", "per_h": {"1": {
        "premise_iii_pass": False, "premise_iv_pass": False, "probe_valid": False}}}
    missing_per_h1 = {"checkpoint_step": 250, "ckpt_path": "fake-missing.pt", "per_h": {}}

    fixtures_expect_true = {"valid": valid_fixture}
    fixtures_expect_false = {
        "invalid_iii": invalid_iii_fail, "invalid_iv": invalid_iv_fail,
        "invalid_h1": invalid_h1_fail, "invalid_all": invalid_all_fail, "missing": missing_per_h1,
    }

    for name, fx in fixtures_expect_true.items():
        v, _ = gate_verdict(fx)
        if v is not True:
            print(f"SELFTEST FAIL: {name!r} fixture returned gate_pass={v} (expected True)")
            ok = False
    for name, fx in fixtures_expect_false.items():
        v, _ = gate_verdict(fx)
        if v is not False:
            print(f"SELFTEST FAIL: {name!r} fixture returned gate_pass={v} (expected False)")
            ok = False

    # End-to-end negative test: write fixtures to REAL temp JSON files, invoke THIS module's own
    # main() via a real subprocess (the exact way phase2_chain.sh would call it), and assert the
    # exit code -- proves the enforcement has TEETH at the process-boundary level.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        this_file = os.path.abspath(__file__)
        paths = {}
        for name, fixture in {**fixtures_expect_true, **fixtures_expect_false}.items():
            p = os.path.join(td, f"{name}.json")
            with open(p, "w") as f:
                json.dump(fixture, f)
            paths[name] = p

        r_valid = subprocess.run([sys.executable, this_file, paths["valid"]], capture_output=True, text=True)
        if r_valid.returncode != 0:
            print(f"SELFTEST FAIL: subprocess on VALID fixture exited {r_valid.returncode} (expected 0) -- "
                  f"stdout={r_valid.stdout!r} stderr={r_valid.stderr!r}")
            ok = False

        for name in fixtures_expect_false:
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
        print("phase2_gate_enforce --selftest: ALL CHECKS PASSED (pure-function positive + 5 negative "
              "fixtures [iii-fail/iv-fail/h1-fail/all-fail/missing], PLUS subprocess-level exit-code "
              "proof for all 6 -- the gate has teeth at the process boundary, sec 16.5 Constraint 1)")
    return ok


if __name__ == "__main__":
    sys.exit(main())
