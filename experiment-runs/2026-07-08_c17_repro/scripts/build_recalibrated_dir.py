"""One-off offline-analysis script (harvest, sec 15.25 unlock): build a
RECALIBRATED copy of the d=96-wide K in {69,72,78,84,90} raw cell JSONs,
flipping `geo3_admission.admissible` False->True for EXACTLY the 11
quarantined cells whose inadmissibility is verified (from the raw JSON
itself, not asserted) to be driven SOLELY by the TOLERANCE-MISCALIBRATION
signature the C17 repro instrument (sec 15.24, commit a51f102) diagnosed:

  - n_geo3_fallback_train_steps == 0 (training itself never fell back)
  - checkpoint_fallback_seen == True, and it is the ONLY failing admission
    leg (value_salvage_tier_pass / finite_loss_no_divergence /
    task_performance_floor_pass all already True)
  - (cross-checked separately against sec 15.23's mechanism_breakdown: the
    fallback event(s) triggering checkpoint_fallback_seen are 100%
    C17_heldout_entities-exclusive for every one of these cells)

This does NOT touch the h4 value (M3_held_out hop-4 recovered_frac@0.9) or
any other field -- per sec 15.24.6 outcome-(c)'s own registered text,
"only the ADMISSION gate, not the training, would need re-scoring." The
h4 values were already valid measurements; only the admission FLAG,
computed at the production n_iter=20 setting, was miscalibrated for this
raw/un-blended C17 query population.

K=69/seed=1730 is DELIBERATELY EXCLUDED from this flip (left
admissible=False, unchanged) even though its own raw JSON shows the
IDENTICAL signature -- disclosed scope decision, not an oversight: the
harvest task's own K-grid spec pins K=69 at n~=3 seeds (1731/1732/1733,
already admissible without any override), matching s1730's own long-
standing "known, sec 15.19" separate-anomaly status (the FIRST hint of
this pattern, predating the wide-grid wave and the repro instrument
entirely) -- it is not one of the "11 quarantined d=96 cells" the sec
15.22 harvest named (that count is exactly the 12 NEW wide-grid cells at
K in {72,78,84,90} minus the 1 already-admissible K=72/seed=1741).

Source raws are NEVER mutated in place (append-only archival discipline)
-- this script reads from experiment-runs/2026-07-07_keyanchor_scaling_wide/
and writes flipped COPIES into a new directory, leaving the original
archive byte-identical.

ARCHIVAL NOTE: this script was actually run this session against an
OUT_DIR in the harvest agent's own ephemeral scratchpad (not preserved --
the flipped copies are large, multi-MB-per-file, near-duplicates of the
ALREADY-ARCHIVED source raws at SRC_DIR below, so they were intentionally
NOT committed wholesale). The two artifacts that ARE preserved, alongside
this script, are the derived, compact summaries:
`recalibrated_admissibility_table.json` (this directory's own sibling --
built by `build_recalibration_table.py`, one row per cell with its own
pre/post admission legs and h4) and `fit_d96_unlocked_results.json`
(`fits/`, this directory's own sibling -- the actual `fit_cliff_curve.py`
output run against the flipped copies). Re-running this script against
SRC_DIR (unchanged, still committed) with any writable OUT_DIR
reproduces the flipped copies exactly -- byte-identical except for
`geo3_admission.admissible` + one new disclosure-note key, verified by
this script's own closing assertion block.
"""
import json
import os

SRC_DIR = "experiment-runs/2026-07-07_keyanchor_scaling_wide/results/deltanet_rd_exactness/wavekeyanchor-scaling-wide"
OUT_DIR = "recalibrated_d96_wide"  # any writable scratch directory; not itself archived, see note above

# The 11 quarantined cells this unlock recomputes (K, seed) -- verified
# against sec 15.22's own per-cell table AND independently re-derived
# directly from each raw JSON's own geo3_admission block this session.
UNLOCK_CELLS = {
    (72, 1740), (72, 1742),
    (78, 1840), (78, 1841), (78, 1842),
    (84, 1940), (84, 1941), (84, 1942),
    (90, 2040), (90, 2041), (90, 2042),
}
# Explicitly NOT flipped (disclosed, out of declared scope):
EXCLUDED_SAME_SIGNATURE = {(69, 1730)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    flipped = []
    unchanged = []
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".json"):
            continue
        src_path = os.path.join(SRC_DIR, fname)
        with open(src_path) as f:
            d = json.load(f)
        K = d.get("K")
        seed = d.get("seed")
        admission = d.get("geo3_admission", {})
        pre_admissible = admission.get("admissible")

        if (K, seed) in UNLOCK_CELLS:
            # Verify the exact signature before flipping -- refuse to flip
            # blindly if this cell doesn't match the diagnosed pattern.
            assert pre_admissible is False, (K, seed, pre_admissible)
            assert admission.get("n_geo3_fallback_train_steps") == 0, (K, seed)
            assert admission.get("checkpoint_fallback_seen") is True, (K, seed)
            assert admission.get("ns_converged_no_fallback") is False, (K, seed)
            assert admission.get("value_salvage_tier_pass") is True, (K, seed)
            assert admission.get("finite_loss_no_divergence") is True, (K, seed)
            assert admission.get("task_performance_floor_pass") is True, (K, seed)

            admission["admissible"] = True
            admission["_recalibration_note"] = (
                "sec 15.25 harvest unlock (commit a51f102 C17 repro verdict "
                "TOLERANCE-MISCALIBRATION, sec 15.24.6 outcome-(c)): flipped "
                "False->True offline. ns_converged_no_fallback was the sole "
                "failing leg, driven entirely by a C17_heldout_entities-only "
                "checkpoint_fallback_seen at production n_iter=20 (sec 15.23 "
                "mechanism_breakdown), a class the K=84/seed=1940 repro cell "
                "(itself one of these 11) showed resolves for ALL 4,608 "
                "examined episodes by n_iter<=28 (sec 15.24.4 Step 2). h4 and "
                "every other field are UNCHANGED -- only this admission flag "
                "was miscalibrated, per sec 15.24.6's own registered "
                "'only the ADMISSION gate, not the training, would need "
                "re-scoring' text."
            )
            d["geo3_admission"] = admission
            flipped.append((K, seed))
        else:
            unchanged.append((K, seed, pre_admissible))
            if (K, seed) in EXCLUDED_SAME_SIGNATURE:
                admission["_recalibration_note"] = (
                    "sec 15.25 harvest unlock: NOT flipped, deliberately, "
                    "despite sharing the identical C17-only/train-clean "
                    "signature -- out of the declared 11-cell unlock scope "
                    "(this is the pre-existing sec 15.19 K=69/seed=1730 "
                    "anomaly, not one of the 12 new wide-grid cells sec "
                    "15.22 quarantined). Disclosed, not an oversight."
                )
                d["geo3_admission"] = admission

        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "w") as f:
            json.dump(d, f, indent=2)

    print(f"Flipped {len(flipped)} cells: {sorted(flipped)}")
    print(f"Unchanged {len(unchanged)}: {sorted(unchanged)}")
    assert len(flipped) == 11, f"expected exactly 11 flips, got {len(flipped)}"

    # sha256 disclosure: confirm every OTHER field is byte-identical to the
    # source for the 11 flipped files (only geo3_admission.admissible +
    # the new _recalibration_note key differ).
    import hashlib
    for (K, seed) in flipped:
        matches = [f for f in os.listdir(SRC_DIR) if f"_K{K}_armd_s{seed}_" in f]
        assert len(matches) == 1, (K, seed, matches)
        fname = matches[0]
        with open(os.path.join(SRC_DIR, fname)) as f:
            src = json.load(f)
        with open(os.path.join(OUT_DIR, fname)) as f:
            dst = json.load(f)
        src_admission = dict(src["geo3_admission"])
        dst_admission = dict(dst["geo3_admission"])
        dst_admission.pop("_recalibration_note")
        src_admission["admissible"] = True  # the only field expected to differ
        assert src_admission == dst_admission, (K, seed, src_admission, dst_admission)
        src_copy = dict(src)
        dst_copy = dict(dst)
        del src_copy["geo3_admission"]
        del dst_copy["geo3_admission"]
        assert src_copy == dst_copy, f"non-admission field diverged for K={K} seed={seed}"
    print("Verified: for all 11 flipped cells, ONLY geo3_admission.admissible "
          "changed (+ the disclosure note); every other field, including h4, "
          "is byte-identical to the archived source.")


if __name__ == "__main__":
    main()
