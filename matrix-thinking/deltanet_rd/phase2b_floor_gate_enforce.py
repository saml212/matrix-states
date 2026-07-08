"""phase2b_floor_gate_enforce.py -- REASONING_LINK_DESIGN.md sec 16.16.6's
OFF-floor validity gate: replaces the retired per-checkpoint Stage-0.5 gate
(sec 16.16.3 item 2) with a SINGLE, upfront, per-corpus check on the OFF
arm's own readout-(B) `L_query` reproduction ratio, closing the sec
16.15.7 trichotomy gap formally (a uniform, substantial, sub-pin fall
mapped to none of the ORIGINAL 3-way rule -- this gate's own 3 buckets
close that gap by construction).

Mirrors `phase2_gate_enforce.py`'s own mechanical READ-and-refuse
discipline exactly (sec 16.5 Constraint 1's "every gate needs a chain
enforcement branch" requirement) -- pure function + subprocess-level
negative-test proof, same "gates must abort, never narrate" contract.

Rule (sec 16.16.6, exactly as specified):
    ratio := L_query(off, K=32, h in {1,2}, c=5000) / L_query(off, K=32, h in {1,2}, c=250)
    (mean-across-3-seeds-pooled per corpus)
  1. FLOOR-PASS          (ratio <= floor_pin)
  2. PARTIAL-BELOW-FLOOR  (floor_pin < ratio < 1.00)
  3. FAMILIARIZATION-NULL (ratio >= 1.00)

`floor_pin` is a REQUIRED argument -- read from `FLOOR_PINNED-Phase2b.json`
(written by `phase2b_off_cache.py`'s own chain-step-3 flow) by the CALLER,
never recomputed here and never a hardcoded constant (sec 16.16.6's own
MAJOR-3 fix: the Rev-0 `<=0.80` constant is demoted to a provisional
sanity-bound fixture ONLY, reused below solely as an EXAMPLE for the
self-test, never as the enforced pin).

Run standalone:
    python phase2b_floor_gate_enforce.py --ratio 0.75 --floor-pin 0.80
    python phase2b_floor_gate_enforce.py --selftest   # negative test, run to completion
"""
from __future__ import annotations

import argparse
import subprocess
import sys

FLOOR_PASS = "FLOOR-PASS"
PARTIAL_BELOW_FLOOR = "PARTIAL-BELOW-FLOOR"
FAMILIARIZATION_NULL = "FAMILIARIZATION-NULL"

# Rev-0's own knife-edge constant (sec 16.16.6's MAJOR-3 fix demotes this to a provisional sanity
# bound / self-test fixture value ONLY -- never the enforced pin, which is always read from
# FLOOR_PINNED-Phase2b.json by the caller).
PROVISIONAL_SANITY_BOUND = 0.80


def floor_verdict(ratio: float, floor_pin: float) -> str:
    """`floor_pin` REQUIRED (sec 16.16.6's own MAJOR-3 fix: never a hardcoded default) -- the
    3-way MECE bucket, closing the sec 16.15.7 trichotomy gap formally. Boundaries: `ratio ==
    floor_pin` -> FLOOR-PASS (`<=`); `ratio == 1.00` -> FAMILIARIZATION-NULL (`>=`)."""
    if ratio <= floor_pin:
        return FLOOR_PASS
    if ratio < 1.00:
        return PARTIAL_BELOW_FLOOR
    return FAMILIARIZATION_NULL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--floor-pin", type=float, default=None)
    ap.add_argument("--corpus", type=str, default=None, help="for the printed message only")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if _run_selftest() else 1

    if args.ratio is None or args.floor_pin is None:
        print("usage: phase2b_floor_gate_enforce.py --ratio <float> --floor-pin <float> "
              "[--corpus <name>] | --selftest", file=sys.stderr)
        return 2

    verdict = floor_verdict(args.ratio, args.floor_pin)
    corpus_tag = f" corpus={args.corpus}" if args.corpus else ""
    print(f"[phase2b-floor-gate]{corpus_tag} ratio={args.ratio:.4f} floor_pin={args.floor_pin:.4f}: "
          f"{verdict}")
    if verdict == FAMILIARIZATION_NULL:
        print(f"REFUSE (per-corpus): FAMILIARIZATION-NULL -- L_query did not fall at all (or rose) "
              f"for this corpus; its own arm-contrast is uninterpretable and MUST be excluded (sec "
              f"16.5 Constraint 1). Wave-level enforcement (at least one corpus must NOT be "
              f"FAMILIARIZATION-NULL) is the CALLER's own responsibility (mirrors the retired "
              f"Stage-0.5 gate's 'at least one clears' convention, phase2b_chain.sh).",
              file=sys.stderr)
        return 1
    print(f"PASS (per-corpus): {verdict} -- "
          + ("full CONFIRMATORY-tier hexachotomy classification proceeds." if verdict == FLOOR_PASS
             else "hexachotomy classification proceeds, but this corpus's own headline finding is "
                  "DEMOTED to DESCRIPTIVE TIER."))
    return 0


# ---------------------------------------------------------------------------
# Negative-test fixture suite -- run to completion, mirroring phase2_gate_enforce._run_selftest's
# own pure-function + subprocess-level exit-code proof. Positive + negative fixtures at an EXAMPLE
# floor_pin=0.80 (the demoted Rev-0 sanity-bound value, reused ONLY as a fixture here), PLUS a
# cross-pin fixture proving the function actually READS its floor_pin argument rather than
# silently reverting to a baked-in 0.80 (the "new gate needs a negative test" obligation the task
# itself names).
# ---------------------------------------------------------------------------

def _run_selftest() -> bool:
    ok = True

    # Pure-function fixtures at floor_pin=0.80.
    cases = [
        (0.75, 0.80, FLOOR_PASS),               # clean pass
        (0.90, 0.80, PARTIAL_BELOW_FLOOR),        # real fall, short of the floor
        (1.05, 0.80, FAMILIARIZATION_NULL),       # loss rose
        (0.80, 0.80, FLOOR_PASS),                 # boundary: ratio == floor_pin -> FLOOR-PASS ("<=")
        (1.00, 0.80, FAMILIARIZATION_NULL),       # boundary: ratio == 1.00 -> FAMILIARIZATION-NULL (">=")
        (0.999999, 0.80, PARTIAL_BELOW_FLOOR),    # just under 1.00 -> still PARTIAL, not NULL
    ]
    for ratio, floor_pin, expected in cases:
        got = floor_verdict(ratio, floor_pin)
        if got != expected:
            print(f"SELFTEST FAIL: floor_verdict(ratio={ratio}, floor_pin={floor_pin}) = {got!r}, "
                  f"expected {expected!r}")
            ok = False

    # Cross-pin fixture (proves floor_pin is actually READ, not a hidden 0.80 default): the SAME
    # ratio=0.75 reclassifies from FLOOR-PASS (floor_pin=0.80) to PARTIAL-BELOW-FLOOR (floor_pin=0.65).
    r_default_pin = floor_verdict(0.75, 0.80)
    r_lower_pin = floor_verdict(0.75, 0.65)
    if not (r_default_pin == FLOOR_PASS and r_lower_pin == PARTIAL_BELOW_FLOOR):
        print(f"SELFTEST FAIL: cross-pin fixture did not reclassify -- "
              f"floor_pin=0.80 -> {r_default_pin!r}, floor_pin=0.65 -> {r_lower_pin!r} "
              f"(expected {FLOOR_PASS!r} then {PARTIAL_BELOW_FLOOR!r}); floor_pin may not be "
              f"actually read")
        ok = False

    # End-to-end negative test: REAL subprocess, exit-code level, mirroring
    # phase2_gate_enforce._run_selftest's own discipline.
    this_file = __import__("os").path.abspath(__file__)
    r_pass = subprocess.run([sys.executable, this_file, "--ratio", "0.75", "--floor-pin", "0.80"],
                             capture_output=True, text=True)
    if r_pass.returncode != 0:
        print(f"SELFTEST FAIL: subprocess on FLOOR-PASS fixture exited {r_pass.returncode} "
              f"(expected 0) -- stdout={r_pass.stdout!r} stderr={r_pass.stderr!r}")
        ok = False

    r_partial = subprocess.run([sys.executable, this_file, "--ratio", "0.90", "--floor-pin", "0.80"],
                                capture_output=True, text=True)
    if r_partial.returncode != 0:
        print(f"SELFTEST FAIL: subprocess on PARTIAL-BELOW-FLOOR fixture exited "
              f"{r_partial.returncode} (expected 0, this bucket still proceeds) -- "
              f"stdout={r_partial.stdout!r} stderr={r_partial.stderr!r}")
        ok = False

    r_null = subprocess.run([sys.executable, this_file, "--ratio", "1.05", "--floor-pin", "0.80"],
                             capture_output=True, text=True)
    if r_null.returncode == 0:
        print(f"SELFTEST FAIL: subprocess on FAMILIARIZATION-NULL fixture exited 0 (expected "
              f"nonzero) -- the gate has NO TEETH. stdout={r_null.stdout!r}")
        ok = False
    elif "REFUSE" not in r_null.stderr:
        print(f"SELFTEST FAIL: FAMILIARIZATION-NULL fixture subprocess exited nonzero but did not "
              f"print the REFUSE message: stderr={r_null.stderr!r}")
        ok = False

    if ok:
        print("phase2b_floor_gate_enforce --selftest: ALL CHECKS PASSED (6 pure-function fixtures "
              "incl. both boundaries, 1 cross-pin reclassification fixture, PLUS subprocess-level "
              "exit-code proof for FLOOR-PASS/PARTIAL-BELOW-FLOOR [both proceed, exit 0] and "
              "FAMILIARIZATION-NULL [refuses, exit 1] -- the gate has teeth AND actually reads its "
              "own floor_pin argument, sec 16.5 Constraint 1)")
    return ok


if __name__ == "__main__":
    sys.exit(main())
