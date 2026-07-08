"""harvest_poolmargin_k84s1943_k90s2043.py -- KEY_ANCHORING_SCALING_DRAFT.md
sec 15.26.5 (Rev 2 + round-3 VERIFY-pass adopted fix m-b, c66b3f6/fd8c509/
16f3543, DESIGN-CLEARED-FOR-BUILD): reads both K90 pool-margin control
diagnostic cells' own result JSONs (K=84/seed=1943, K=90/seed=2043),
computes the pre-registered `shift`/`noise_shift`/`Delta_measured` triple
and applies the exact-threshold REAL/AMBIGUOUS/ARTIFACT outcome trigger
(with the MINOR-3 Delta_measured<=0 contingency routing), and writes a
single verdict JSON. All arithmetic lives in `compute_verdict()` below --
a pure function of 5 scalars, independently unit-testable without any
result JSON, GPU, or `fla` import (`--self-test` runs the registered
synthetic cases covering all 3 outcome buckets PLUS the Delta<=0
contingency, to completion, no GPU required).

Statistic (sec 15.26.5, pinned): h4 := M3_held_out['4']['recovered_frac@0.9']
(the SAME hop/tau `fit_cliff_curve.py` fits, sec 15.25.5).

shift (sec 15.26.5, pinned defn -- read directly against the design's own
text, not assumed by analogy): seed=1943's own `M3_held_out_pool_restricted`
h4 MINUS its own UNRESTRICTED `M3_held_out` h4, at the SAME (final)
checkpoint, same trained weights -- keeps its SIGN (a restriction that
makes recovery WORSE is still informative, sec 15.26.5's own explicit
"never treated as an artifact signal" note).

noise_shift (round-3 VERIFY-pass adopted fix m-b, 2026-07-08 -- a single
noise-repeat draw is only an n=1 sample of the null): the MAX of the two
independently drawn absolute deviations,
    noise_shift := max(|M3_held_out_noise_repeat h4 - M3_held_out h4|,
                        |M3_held_out_noise_repeat_2 h4 - M3_held_out h4|)
(conservative -- Rev 2's own single-draw formula, `|repeat - standard|`,
is the degenerate n=1 case of this same definition).

Delta_measured (sec 15.26.5 + MINOR-3 contingency, sec 15.26.10): K=90's
own real n=3 mean (1.0000, sec 15.26.1's table) MINUS seed=1943's own
freshly-measured UNRESTRICTED M3_held_out h4 -- re-pinned to seed=2043's
own fresh h4 if that fresh reading does not reproduce ceiling (h4 < 0.98
under the n_iter=28 admission gate), and the whole comparison routes to
AMBIGUOUS (never forcing a REAL/ARTIFACT call) if the fresh K90 reading
drops below K84's own mean -- i.e. the premise (K84 < K90) no longer holds.

Usage (build-audit MAJOR-1 fix, 2026-07-08: paths point at this diagnostic's OWN isolated
results/deltanet_rd_exactness/wavekeyanchor-scaling-poolmargin/ dir, matching
run_poolmargin_k84s1943_k90s2043.py's own OUT_DIR -- NEVER the shared, frozen
wavekeyanchor-scaling-wide/ dir the sec 15.26.1 per-K table's own cells live in):
  .venv/bin/python harvest_poolmargin_k84s1943_k90s2043.py \\
      --k84-json results/deltanet_rd_exactness/wavekeyanchor-scaling-poolmargin/<K84 cell>.json \\
      --k90-json results/deltanet_rd_exactness/wavekeyanchor-scaling-poolmargin/<K90 cell>.json \\
      --out verdict.json

  Local, GPU-free self-test (synthetic cases, no result JSON needed):
  DRY_RUN_BYPASS=1 .venv/bin/python harvest_poolmargin_k84s1943_k90s2043.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

H4_HOP = "4"           # sec 15.26.5: h4 = M3_held_out['4']['recovered_frac@0.9'] (JSON int-key stringification)
TAU_KEY = "recovered_frac@0.9"
K90_OLD_GRID_MEAN = 1.0000     # sec 15.26.1's table: K=90's own real n=3 mean, exact ceiling
K90_ADMISSION_FLOOR = 0.98     # sec 15.26.5 MINOR-3 contingency: "does NOT reproduce ceiling"

# build-audit MAJOR-1 fix (2026-07-08): the FROZEN sec 15.26.1 per-K table's own directory --
# this diagnostic's own result JSONs must NEVER be read from (or, by the wrapper's own OUT_DIR
# fix, written to) here. Refused loudly in harvest() below rather than silently harvesting off
# whatever the caller happens to point at.
FROZEN_WIDE_DIR_BASENAME = "wavekeyanchor-scaling-wide"


def _h4(entry: dict) -> float:
    return entry[H4_HOP][TAU_KEY]


def _final_checkpoint(result: dict) -> dict:
    ckpts = result.get("checkpoints") or []
    assert ckpts, "result JSON has no checkpoints -- cannot read a final h4"
    return ckpts[-1]


def compute_verdict(*, k84_standard: float, k84_restricted: float,
                     k84_noise_repeat: float, k84_noise_repeat_2: float,
                     k90_fresh: float | None) -> dict:
    """Pure function of 5 scalars (sec 15.26.5's own exact-threshold trigger + the round-3
    noise_shift max-of-two-draws adoption + the MINOR-3 Delta_measured contingency). Never
    reads a file -- independently unit-testable, no GPU/fla import required.

    k90_fresh=None models "seed=2043 never landed / degenerate cell" (sec 15.26.5's own
    "(degenerate cell)" trigger row) -- routed separately, below, before any REAL/ARTIFACT/
    AMBIGUOUS bucket is even considered."""
    if k90_fresh is None:
        return {"verdict": "DEGENERATE_CELL",
                "reason": "K=90/seed=2043 did not land an admissible reading (sec 15.26.5's own "
                          "'(degenerate cell)' trigger row) -- escalate per the SAME sec 15.23 "
                          "C17-exclusive-signature adjudication Rev 0 already registered, not "
                          "silently absorbed."}

    shift = k84_restricted - k84_standard   # SIGNED (sec 15.26.5: a worse-recovery restriction is still informative)
    noise_shift = max(abs(k84_noise_repeat - k84_standard), abs(k84_noise_repeat_2 - k84_standard))

    # MINOR-3 contingency (sec 15.26.10): Delta_measured's minuend re-pins to the FRESH K=90
    # reading if it doesn't reproduce the old grid's own ceiling; if the fresh K90 reading drops
    # BELOW K84's own mean (the premise this diagnostic exists to explain no longer holds), route
    # straight to AMBIGUOUS -- never forcing a REAL/ARTIFACT call on a premise that doesn't hold.
    delta_repinned = k90_fresh < K90_ADMISSION_FLOOR
    k90_for_delta = k90_fresh if delta_repinned else K90_OLD_GRID_MEAN
    delta_measured = k90_for_delta - k84_standard

    if delta_measured <= 0:
        return {"verdict": "AMBIGUOUS",
                "reason": f"Delta_measured={delta_measured:.6f} <= 0 -- the fresh K=90 reading "
                          f"({k90_fresh:.6f}) does not sit above K=84/seed=1943's own unrestricted "
                          f"mean ({k84_standard:.6f}); the K84<K90 premise this diagnostic exists "
                          f"to explain no longer holds under the fresh, matched-n_iter comparison "
                          f"-- routing directly to AMBIGUOUS (sec 15.26.5 MINOR-3 contingency), "
                          f"never forcing a REAL/ARTIFACT call, plus a registered follow-up naming "
                          f"the reversed direction as a new, disclosed open question.",
                "shift": shift, "noise_shift": noise_shift, "delta_measured": delta_measured,
                "delta_repinned": delta_repinned, "k90_for_delta": k90_for_delta}

    real_thresh = max(0.1 * delta_measured, noise_shift)
    artifact_thresh = max(0.5 * delta_measured, 3 * noise_shift)
    assert real_thresh <= artifact_thresh, (
        f"trigger totality VIOLATED: real_thresh={real_thresh} > artifact_thresh={artifact_thresh} "
        f"(delta_measured={delta_measured}, noise_shift={noise_shift}) -- sec 15.26.5's own "
        f"totality walk proves this cannot happen for any nonnegative noise_shift and "
        f"delta_measured > 0; a violation here means the trigger implementation itself has a bug")

    if shift >= artifact_thresh:
        verdict = "CEILING-IS-ARTIFACT"
    elif shift <= real_thresh:
        verdict = "CEILING-IS-REAL"
    else:
        verdict = "AMBIGUOUS"

    return {"verdict": verdict, "shift": shift, "noise_shift": noise_shift,
            "delta_measured": delta_measured, "delta_repinned": delta_repinned,
            "k90_for_delta": k90_for_delta, "real_thresh": real_thresh,
            "artifact_thresh": artifact_thresh,
            "reason": f"shift={shift:.6f} vs REAL_THRESH={real_thresh:.6f} / "
                      f"ARTIFACT_THRESH={artifact_thresh:.6f} (delta_measured={delta_measured:.6f}, "
                      f"noise_shift={noise_shift:.6f}, round-3 max-of-2-draws)"}


def _refuse_if_in_frozen_dir(label: str, path: str) -> None:
    """build-audit MAJOR-1 fix: refuses loudly if the given result JSON sits inside the FROZEN
    wavekeyanchor-scaling-wide/ dir -- this diagnostic's own cells belong exclusively under
    wavekeyanchor-scaling-poolmargin/ (the wrapper's own OUT_DIR). Catches a human/tool pointing
    this harvest at the wrong (shared) directory, e.g. by copy-pasting a pre-fix example path."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    assert parent != FROZEN_WIDE_DIR_BASENAME, (
        f"build-audit MAJOR-1: {label}_json path {path!r} sits inside the FROZEN "
        f"{FROZEN_WIDE_DIR_BASENAME!r} directory -- this diagnostic's own cells must be read from "
        f"wavekeyanchor-scaling-poolmargin/ only, never the shared frozen per-K table's own dir.")


def harvest(k84_json_path: str, k90_json_path: str) -> dict:
    _refuse_if_in_frozen_dir("k84", k84_json_path)
    _refuse_if_in_frozen_dir("k90", k90_json_path)
    with open(k84_json_path) as f:
        k84 = json.load(f)
    with open(k90_json_path) as f:
        k90 = json.load(f)

    ck84 = _final_checkpoint(k84)
    assert "M3_held_out_pool_restricted" in ck84 and "M3_held_out_noise_repeat" in ck84 and \
           "M3_held_out_noise_repeat_2" in ck84, (
        "K=84 result JSON's final checkpoint is missing one of the sec 15.26.2.2 additive eval "
        "keys -- was this cell launched with --m3-pool-restrict-n set?")
    k84_standard = _h4(ck84["M3_held_out"])
    k84_restricted = _h4(ck84["M3_held_out_pool_restricted"])
    k84_noise_repeat = _h4(ck84["M3_held_out_noise_repeat"])
    k84_noise_repeat_2 = _h4(ck84["M3_held_out_noise_repeat_2"])

    ga84 = (k84.get("geo3_admission") or {})
    ga90 = (k90.get("geo3_admission") or {})
    k84_admissible = ga84.get("admissible") is True
    k90_admissible = ga90.get("admissible") is True

    ck90 = _final_checkpoint(k90)
    k90_fresh = _h4(ck90["M3_held_out"]) if k90_admissible else None

    if not k84_admissible:
        result = {"verdict": "DEGENERATE_CELL",
                  "reason": "K=84/seed=1943 did not land an admissible reading (sec 15.26.5's own "
                            "'(degenerate cell)' trigger row) -- escalate per the SAME sec 15.23 "
                            "C17-exclusive-signature adjudication Rev 0 already registered."}
    else:
        result = compute_verdict(k84_standard=k84_standard, k84_restricted=k84_restricted,
                                  k84_noise_repeat=k84_noise_repeat,
                                  k84_noise_repeat_2=k84_noise_repeat_2, k90_fresh=k90_fresh)

    result["inputs"] = {
        "k84_standard_h4": k84_standard, "k84_restricted_h4": k84_restricted,
        "k84_noise_repeat_h4": k84_noise_repeat, "k84_noise_repeat_2_h4": k84_noise_repeat_2,
        "k90_fresh_h4": k90_fresh, "k84_admissible": k84_admissible, "k90_admissible": k90_admissible,
        "k84_json": k84_json_path, "k90_json": k90_json_path,
    }
    return result


# ---------------------------------------------------------------------------
# Self-test: synthetic cases covering all 3 outcome buckets + the
# Delta_measured<=0 contingency + the degenerate-cell routing. Run to
# completion, no GPU/file I/O required.
# ---------------------------------------------------------------------------

def _self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[self-test:{name}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
        if not cond:
            failures.append(name)

    # Case 1: CEILING-IS-REAL -- shift tiny relative to a clean Delta, noise_shift small.
    # k84_standard=0.90, K90_old_mean=1.00 -> Delta=0.10, real_thresh=max(0.01, noise)=0.01+
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.905, k84_noise_repeat=0.903,
                         k84_noise_repeat_2=0.898, k90_fresh=1.0)
    check("case-REAL", r["verdict"] == "CEILING-IS-REAL", r["reason"])

    # Case 2: CEILING-IS-ARTIFACT -- restriction recovers most of the gap.
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.99, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.899, k90_fresh=1.0)
    check("case-ARTIFACT", r["verdict"] == "CEILING-IS-ARTIFACT", r["reason"])

    # Case 3: AMBIGUOUS -- shift strictly between REAL_THRESH and ARTIFACT_THRESH.
    # Delta=0.10 -> real_thresh~0.01 (noise small), artifact_thresh=0.05; shift=0.03 lands between.
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.93, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.899, k90_fresh=1.0)
    check("case-AMBIGUOUS", r["verdict"] == "AMBIGUOUS", r["reason"])

    # Case 4: Delta_measured <= 0 contingency -- fresh K90 reading at/below K84's own mean.
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.93, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.899, k90_fresh=0.85)
    check("case-DELTA-LE-0-CONTINGENCY", r["verdict"] == "AMBIGUOUS" and r["delta_measured"] <= 0,
          r["reason"])
    # Boundary: delta_measured EXACTLY 0 must also route here (the <=0 check, not <0).
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.93, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.899, k90_fresh=0.90)
    check("case-DELTA-EXACTLY-0-CONTINGENCY",
          r["verdict"] == "AMBIGUOUS" and r["delta_measured"] == 0.0, r["reason"])

    # Case 5: degenerate cell (K=90 never landed / inadmissible) -- routed before any bucket.
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.93, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.899, k90_fresh=None)
    check("case-DEGENERATE-CELL", r["verdict"] == "DEGENERATE_CELL", r["reason"])

    # Case 6: MINOR-3 re-pin -- fresh K90 reading below 0.98 admission floor but still above K84's
    # mean -- Delta_measured re-pins to the FRESH reading (not the old grid's 1.0000 mean).
    r = compute_verdict(k84_standard=0.80, k84_restricted=0.82, k84_noise_repeat=0.801,
                         k84_noise_repeat_2=0.799, k90_fresh=0.95)
    check("case-REPIN-TO-FRESH-K90",
          r["delta_repinned"] is True and abs(r["k90_for_delta"] - 0.95) < 1e-9, r["reason"])

    # Case 7: no re-pin -- fresh K90 reading clears the 0.98 admission floor, Delta uses the
    # OLD grid's own 1.0000 mean (not the fresh reading).
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.905, k84_noise_repeat=0.903,
                         k84_noise_repeat_2=0.898, k90_fresh=0.999)
    check("case-NO-REPIN-USES-OLD-GRID-MEAN",
          r["delta_repinned"] is False and abs(r["k90_for_delta"] - 1.0) < 1e-9, r["reason"])

    # Case 8: noise_shift is the MAX of the two draws, not the mean and not just the first --
    # construct repeat_1 close to standard, repeat_2 far -- noise_shift must reflect repeat_2.
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.905, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.94, k90_fresh=1.0)
    check("case-NOISE-SHIFT-IS-MAX-OF-TWO", abs(r["noise_shift"] - 0.04) < 1e-9,
          f"noise_shift={r['noise_shift']} (expected max(|0.901-0.90|,|0.94-0.90|)=0.04)")

    # Case 9: negative shift (restriction made recovery WORSE) still routes correctly -- a
    # negative shift always falls at/under REAL_THRESH (>=0), never mis-signed into ARTIFACT.
    r = compute_verdict(k84_standard=0.90, k84_restricted=0.85, k84_noise_repeat=0.901,
                         k84_noise_repeat_2=0.899, k90_fresh=1.0)
    check("case-NEGATIVE-SHIFT-ROUTES-REAL", r["verdict"] == "CEILING-IS-REAL" and r["shift"] < 0,
          r["reason"])

    # Totality walk, numeric sweep (mirrors sec 15.26.5's own 200,001-point sweep, smaller here
    # since this is a build-time self-test, not the design's own independent verification pass):
    # REAL_THRESH <= ARTIFACT_THRESH for every noise_shift/delta_measured pair on a grid.
    violations = 0
    for i in range(0, 2001):
        noise_shift = i / 1000.0            # 0.0 .. 2.0
        for delta_measured in (0.01, 0.05, 0.1, 0.5, 1.0):
            real_thresh = max(0.1 * delta_measured, noise_shift)
            artifact_thresh = max(0.5 * delta_measured, 3 * noise_shift)
            if real_thresh > artifact_thresh:
                violations += 1
    check("totality-walk-numeric-sweep", violations == 0, f"{violations} violation(s) out of 10,005 grid points")

    # build-audit MAJOR-1 negative test: a path inside the FROZEN wavekeyanchor-scaling-wide/ dir
    # must refuse, never be silently harvested from.
    frozen_refused = False
    try:
        _refuse_if_in_frozen_dir("k84", "results/deltanet_rd_exactness/wavekeyanchor-scaling-wide/"
                                         "wkeyanchor-scaling-wide_rdx_K84_armd_s1940_geo3n20.json")
    except AssertionError:
        frozen_refused = True
    check("frozen-dir-read-guard-refuses[MAJOR-1]", frozen_refused)
    poolmargin_ok = False
    try:
        _refuse_if_in_frozen_dir("k84", "results/deltanet_rd_exactness/"
                                         "wavekeyanchor-scaling-poolmargin/"
                                         "wkeyanchor-scaling_rdx_K84_armd_s1943_geo3n28.json")
        poolmargin_ok = True
    except AssertionError:
        poolmargin_ok = False
    check("poolmargin-dir-read-guard-passes[MAJOR-1]", poolmargin_ok)

    print(f"\nSELF-TEST SUMMARY: {len(failures)} failure(s) out of self-test items run.", flush=True)
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    print("ALL SELF-TESTS PASSED.", flush=True)
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()

    ap = argparse.ArgumentParser()
    ap.add_argument("--k84-json", required=True)
    ap.add_argument("--k90-json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = harvest(args.k84_json, args.k90_json)
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
