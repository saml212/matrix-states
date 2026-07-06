"""fit_frozenbias_estimation.py -- FROZEN_BIAS_LM_DESIGN.md sec 7.1-real.1's
pinned ESTIMATION-mode readout: computes `mean_delta +/- t(2,0.975)*s/sqrt(3)`
(the SINGLE pinned CI formula, two-sided, n=3 seeds) for the Arm2-vs-Arm1'
post-blend primary and the pre-blend k_raw co-primary, per corpus, and
reports an estimation-mode result JSON (delta, CI, excludes-zero flag per
corpus) -- NOT a fixed-threshold binary CONFIRM/REFUTE (sec 7.1-real's own
demotion of the primary bar, ROUND-4).

Reused, not reimplemented: `pinned_ci_half_width`/`T_DIST_975_DF2`
(bands_pinned_frozenbias.py -- the SAME pinned-formula function the
blind-pin writer itself uses, so the CI here and the CI baked into
BANDS_PINNED-FrozenBias.json are, by construction, the same arithmetic).

This program's own pinned per-corpus REAL noise-floor stds (sec 7.1-real,
reproduced directly from `experiment-runs/2026-07-06_trajectory_probes/
reference_finals_archived.json`'s `mixcontrol` family this session --
verified to match the design doc's own quoted 0.021992/0.042838 to 6
decimal places): these are the PRE-LAUNCH power-check reference only; the
REAL launch-time `s` for Arm2-vs-Arm1' is measured FRESH from that
comparison's own actual per-seed deltas (this script's `derive_estimation`
takes the per-seed delta VALUES as its input, computing `s` from THEM
directly -- it does not import or fall back to the archived pre-launch
figures at all, since sec 7.1-real.1's own launch-time discipline requires
a fresh re-derivation, not a substitution).

Usage:
  python fit_frozenbias_estimation.py --arm2 <arm2_span_frac_per_seed.json> \\
      --arm1prime <arm1prime_span_frac_per_seed.json> --out results/.../estimation.json
  python fit_frozenbias_estimation.py --self-test   # negative tests, run to completion
"""
from __future__ import annotations

import argparse
import json
import sys

from bands_pinned_frozenbias import T_DIST_975_DF2, _mean_std, pinned_ci_half_width

N_SEEDS = 3

# sec 7.1-real.1's own pinned per-corpus thresholds -- reproduced here as a DOCUMENTED CROSS-CHECK
# constant (this script's own estimation-mode delta/CI computation does NOT depend on these at all;
# they are printed alongside the fresh result purely so a reader can see how the measured CI
# compares to the pre-launch power-check reference, sec 1.0/sec 9's own recap). Recomputed this
# session directly from experiment-runs/2026-07-06_trajectory_probes/reference_finals_archived.json
# (mixcontrol family, all7_span_frac, n=3 seeds/corpus) -- matches the design doc's own cited
# 0.0546/0.1064 to 4 decimal places.
PRELAUNCH_REFERENCE_THRESHOLDS = {
    "openr1-mix-ext": 0.0546,
    "wikitext-mix-ext": 0.1064,
}


def derive_estimation(per_seed_deltas: list[float], corpus: str, mechanism_direction: str = "negative") -> dict:
    """per_seed_deltas: [delta_seed0, delta_seed1, delta_seed2] where each delta is
    (measured_quantity(Arm2, seed) - measured_quantity(Arm1', seed)) -- i.e. this function is
    AGNOSTIC to whether the input is post-blend span_frac (the primary) or pre-blend k_raw
    span_frac (the co-primary); the caller supplies whichever delta family it wants estimated.

    mechanism_direction: "negative" (this design's own mechanism account, sec 7.1: a CONFIRM
    requires the CI to exclude zero in the negative direction -- Arm 2 below Arm 1') or "positive"
    (never expected here, but not hard-coded away -- kept as an explicit parameter rather than an
    assumption baked into the return value, so a fresh attack round can flip it without editing
    this function's own arithmetic)."""
    assert len(per_seed_deltas) == N_SEEDS, (
        f"corpus={corpus!r}: estimation requires exactly {N_SEEDS} per-seed deltas, got "
        f"{len(per_seed_deltas)}")
    mean_delta, s = _mean_std(per_seed_deltas)
    half_width = pinned_ci_half_width(s)
    ci_lower = mean_delta - half_width
    ci_upper = mean_delta + half_width
    excludes_zero = not (ci_lower <= 0.0 <= ci_upper)
    if mechanism_direction == "negative":
        confirm_direction_consistent = excludes_zero and mean_delta < 0.0
    else:
        confirm_direction_consistent = excludes_zero and mean_delta > 0.0
    return {
        "corpus": corpus, "per_seed_deltas": list(per_seed_deltas), "n": N_SEEDS,
        "mean_delta": mean_delta, "s": s, "t_mult": T_DIST_975_DF2,
        "pinned_ci_half_width": half_width, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "excludes_zero": excludes_zero,
        "mechanism_direction": mechanism_direction,
        "confirm_direction_consistent": confirm_direction_consistent,
        "prelaunch_reference_threshold": PRELAUNCH_REFERENCE_THRESHOLDS.get(corpus),
    }


def headline_verdict(post_blend_by_corpus: dict, pre_blend_by_corpus: dict) -> dict:
    """sec 1.1/sec 9's headline pass condition: BOTH the post-blend primary AND the pre-blend
    co-primary must clear (CI excludes zero, mechanism-consistent direction) on BOTH corpora
    independently for a CONFIRM; a split on either instrument or either corpus is INCONCLUSIVE
    (sec 7.1's own same-outcome-in-both-corpora-and-both-instruments rule, mirroring
    TRACKB_REDESIGN.md sec 5.1)."""
    corpora = sorted(set(post_blend_by_corpus) | set(pre_blend_by_corpus))
    per_corpus_pass = {}
    for c in corpora:
        post_ok = post_blend_by_corpus.get(c, {}).get("confirm_direction_consistent", False)
        pre_ok = pre_blend_by_corpus.get(c, {}).get("confirm_direction_consistent", False)
        per_corpus_pass[c] = {"post_blend_confirm": post_ok, "pre_blend_confirm": pre_ok,
                               "both_confirm": post_ok and pre_ok}
    all_corpora_confirm = all(v["both_confirm"] for v in per_corpus_pass.values()) and len(corpora) > 0
    any_corpus_confirm = any(v["both_confirm"] for v in per_corpus_pass.values())
    if all_corpora_confirm:
        verdict = "CONFIRM"
    elif any_corpus_confirm:
        verdict = "INCONCLUSIVE (split across corpora)"
    else:
        verdict = "ESTIMATION (no corpus clears both instruments) -- see sec 1.0's plain-language outcome"
    return {"per_corpus": per_corpus_pass, "verdict": verdict}


# ---------------------------------------------------------------------------
# Negative tests (RUN TO COMPLETION, not merely written -- this project's own hard rule):
# synthetic data where the CI includes zero, and synthetic data where it excludes zero.
# ---------------------------------------------------------------------------

def _self_test() -> bool:
    ok = True

    print("[negative test 1] synthetic per-seed deltas straddling zero -> CI must INCLUDE zero")
    # small deltas, mixed sign, high relative variance -- CI should straddle 0.
    deltas_null = [0.002, -0.001, 0.0015]
    r = derive_estimation(deltas_null, "openr1-mix-ext")
    print(f"    mean_delta={r['mean_delta']:.6f} s={r['s']:.6f} "
          f"CI=[{r['ci_lower']:.6f}, {r['ci_upper']:.6f}] excludes_zero={r['excludes_zero']}")
    test1_ok = (r["excludes_zero"] is False) and (not r["confirm_direction_consistent"])
    print(f"    PASS={test1_ok}")
    ok = ok and test1_ok

    print("[negative test 2] synthetic per-seed deltas all substantially negative, low variance "
          "-> CI must EXCLUDE zero, in the mechanism-predicted (negative) direction")
    # A delta magnitude and spread constructed to comfortably clear the pinned openr1 threshold
    # (0.0546) with a tight, consistent negative sign -- exercises the "CI excludes zero, correct
    # direction" positive-control path.
    deltas_confirm = [-0.12, -0.13, -0.115]
    r2 = derive_estimation(deltas_confirm, "openr1-mix-ext")
    print(f"    mean_delta={r2['mean_delta']:.6f} s={r2['s']:.6f} "
          f"CI=[{r2['ci_lower']:.6f}, {r2['ci_upper']:.6f}] excludes_zero={r2['excludes_zero']}")
    test2_ok = (r2["excludes_zero"] is True) and r2["confirm_direction_consistent"]
    print(f"    PASS={test2_ok}")
    ok = ok and test2_ok

    print("[negative test 3] CI excludes zero but in the WRONG (positive) direction -- must NOT "
          "be reported as confirm_direction_consistent under mechanism_direction='negative'")
    deltas_wrong_dir = [0.12, 0.13, 0.115]
    r3 = derive_estimation(deltas_wrong_dir, "wikitext-mix-ext", mechanism_direction="negative")
    print(f"    mean_delta={r3['mean_delta']:.6f} excludes_zero={r3['excludes_zero']} "
          f"confirm_direction_consistent={r3['confirm_direction_consistent']}")
    test3_ok = r3["excludes_zero"] is True and r3["confirm_direction_consistent"] is False
    print(f"    PASS={test3_ok}")
    ok = ok and test3_ok

    print("[negative test 4] wrong seed count must raise AssertionError, not silently proceed")
    raised = False
    try:
        derive_estimation([0.1, 0.2], "openr1-mix-ext")
    except AssertionError:
        raised = True
    print(f"    raised on n=2 (expected n=3): {raised}")
    ok = ok and raised

    print("[negative test 5] headline_verdict: a split (one corpus confirms, other does not) is "
          "INCONCLUSIVE, never silently reported as CONFIRM")
    post_split = {"openr1-mix-ext": r2, "wikitext-mix-ext": derive_estimation(deltas_null, "wikitext-mix-ext")}
    pre_split = {"openr1-mix-ext": r2, "wikitext-mix-ext": derive_estimation(deltas_null, "wikitext-mix-ext")}
    hv = headline_verdict(post_split, pre_split)
    print(f"    verdict: {hv['verdict']}")
    test5_ok = "INCONCLUSIVE" in hv["verdict"] or "ESTIMATION" in hv["verdict"]
    print(f"    PASS (not silently CONFIRM)={test5_ok}")
    ok = ok and test5_ok

    print("[negative test 6] headline_verdict: BOTH corpora confirming -> CONFIRM")
    post_both = {"openr1-mix-ext": r2, "wikitext-mix-ext": r2}
    pre_both = {"openr1-mix-ext": r2, "wikitext-mix-ext": r2}
    hv2 = headline_verdict(post_both, pre_both)
    print(f"    verdict: {hv2['verdict']}")
    test6_ok = hv2["verdict"] == "CONFIRM"
    print(f"    PASS={test6_ok}")
    ok = ok and test6_ok

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arm2-post-blend", type=str, default=None,
                     help="JSON file: {corpus: [span_frac_seed0, seed1, seed2]} -- Arm 2's own "
                          "post-blend span_frac per seed per corpus.")
    ap.add_argument("--arm1prime-post-blend", type=str, default=None,
                     help="JSON file: {corpus: [span_frac_seed0, seed1, seed2]} -- Arm 1''s own "
                          "post-blend span_frac per seed per corpus (same seeds as Arm 1, eval-only).")
    ap.add_argument("--arm2-kraw", type=str, default=None,
                     help="JSON file: {corpus: [span_frac_seed0, seed1, seed2]} -- Arm 2's pre-blend "
                          "k_raw span_frac per seed per corpus (sec 4.a-i co-primary).")
    ap.add_argument("--arm1-kraw", type=str, default=None,
                     help="JSON file: {corpus: [span_frac_seed0, seed1, seed2]} -- Arm 1's pre-blend "
                          "k_raw span_frac per seed per corpus.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.self_test:
        ok = _self_test()
        print("=" * 60)
        print("SELF-TEST: " + ("ALL PASSED" if ok else "FAILURES PRESENT"))
        return 0 if ok else 1

    assert args.arm2_post_blend and args.arm1prime_post_blend and args.arm2_kraw and args.arm1_kraw, (
        "a real (non-self-test) run requires all four input files.")
    with open(args.arm2_post_blend) as f:
        arm2_post = json.load(f)
    with open(args.arm1prime_post_blend) as f:
        arm1prime_post = json.load(f)
    with open(args.arm2_kraw) as f:
        arm2_kraw = json.load(f)
    with open(args.arm1_kraw) as f:
        arm1_kraw = json.load(f)

    corpora = sorted(set(arm2_post) & set(arm1prime_post) & set(arm2_kraw) & set(arm1_kraw))
    post_blend_results, pre_blend_results = {}, {}
    for c in corpora:
        deltas_post = [a2 - a1p for a2, a1p in zip(arm2_post[c], arm1prime_post[c])]
        deltas_pre = [a2 - a1 for a2, a1 in zip(arm2_kraw[c], arm1_kraw[c])]
        post_blend_results[c] = derive_estimation(deltas_post, c)
        pre_blend_results[c] = derive_estimation(deltas_pre, c)

    hv = headline_verdict(post_blend_results, pre_blend_results)
    result = {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 7.1-real.1 (pinned CI, ESTIMATION mode)",
        "post_blend_primary": post_blend_results,
        "pre_blend_co_primary": pre_blend_results,
        "headline": hv,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
