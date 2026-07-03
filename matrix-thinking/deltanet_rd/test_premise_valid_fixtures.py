"""test_premise_valid_fixtures.py -- fixture-based regression tests for
run_deltanet_rd_sweep.py's premise-validity classification
(_premise_valid_entry) and the sweep-level aggregate() roll-up.

Standalone, assert-based (no pytest dependency -- matches this tree's
existing self-contained-script convention, e.g. run_deltanet_rd.py's own
--smoke gate). Run directly:

    python test_premise_valid_fixtures.py

Context (DELTANET_REALDATA_DESIGN.md, section 14.7, dated 2026-07-03): the
gate decision retired tau/tau_v as a real-data validity criterion in favor
of the salvage tier (sigma_K/sigma_1 >= 0.1, both sides) + per-item
alignment (alignment_cos_min >= 0.9), for BOTH the "unconstrained" and
"causal_k_ge_K" arms -- unifying their rules (section 14.7 item (2)). The
`causal_k_ge_K` branch was updated at the time the rule was written; the
`unconstrained` branch was missed and kept gating on the retired
tau=0.03 rule, which silently zeroed `n_premise_valid` in every
Wave A AGGREGATE.json despite 6/11 cells being genuinely alignment-clean
and salvage-tier-valid on both sides (section 16.6's operational note;
per-checkpoint raw diagnostics were always correct -- only the sweep-level
roll-up was stale). These fixtures pin the corrected behavior so the same
drift cannot silently reoccur.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

from run_deltanet_rd_sweep import _premise_valid_entry, aggregate

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


DEFAULT_TH = {"tau_unconstrained": 0.03, "salvage_ratio": 0.1, "alignment_cos": 0.9,
              "tau_value_unconstrained": 0.03, "value_salvage_ratio": 0.1}


# ---------------------------------------------------------------------------
# Part 1: _premise_valid_entry unit fixtures
# ---------------------------------------------------------------------------

def part1_premise_valid_entry():
    print("\n--- Part 1: _premise_valid_entry unit fixtures ---")

    # Task-specified fixture (1): key_gram_dev 2.0 (would fail the retired
    # tau=0.03 rule everywhere) + both salvages 0.4 + align_min 0.95 -> VALID.
    entry = {
        "key_gram_deviation_mean": 2.0, "value_gram_deviation_mean": 2.0,
        "salvage_ratio_mean": 0.4, "value_salvage_ratio_mean": 0.4,
        "alignment_cos_min": 0.95,
    }
    check("unconstrained/salvage+align pass, tau ignored -> VALID",
          _premise_valid_entry(entry, "unconstrained", DEFAULT_TH), True)

    # Task-specified fixture (2): same, value salvage dropped to 0.05 -> INVALID.
    entry2 = dict(entry, value_salvage_ratio_mean=0.05)
    check("unconstrained/low value salvage (0.05 < 0.1) -> INVALID",
          _premise_valid_entry(entry2, "unconstrained", DEFAULT_TH), False)

    # Task-specified fixture (3): same, align_min dropped to 0.6 -> INVALID.
    entry3 = dict(entry, alignment_cos_min=0.6)
    check("unconstrained/low alignment (0.6 < 0.9) -> INVALID",
          _premise_valid_entry(entry3, "unconstrained", DEFAULT_TH), False)

    # Regression pin for the exact bug: measured Wave 0 real-data band
    # (key Gram deviation 1.26-2.77, value 1.32-3.31, salvage 0.26-0.55)
    # must classify VALID when salvage+alignment pass, even though every
    # one of these Gram deviations fails tau=0.03 unconditionally.
    entry4 = {
        "key_gram_deviation_mean": 2.77, "value_gram_deviation_mean": 3.31,
        "salvage_ratio_mean": 0.26, "value_salvage_ratio_mean": 0.26,
        "alignment_cos_min": 1.0,
    }
    check("unconstrained/measured Wave 0 upper band -> VALID (tau no longer gates)",
          _premise_valid_entry(entry4, "unconstrained", DEFAULT_TH), True)

    # causal_k_ge_K branch: unaffected by this fix, still salvage+alignment.
    entry5 = {"salvage_ratio_mean": 0.3, "value_salvage_ratio_mean": 0.3,
              "alignment_cos_min": 0.92}
    check("causal_k_ge_K/salvage+align pass -> VALID (unchanged)",
          _premise_valid_entry(entry5, "causal_k_ge_K", DEFAULT_TH), True)

    entry6 = {"salvage_ratio_mean": 0.05, "value_salvage_ratio_mean": 0.3,
              "alignment_cos_min": 0.92}
    check("causal_k_ge_K/low key salvage -> INVALID (unchanged)",
          _premise_valid_entry(entry6, "causal_k_ge_K", DEFAULT_TH), False)

    # causal_k_lt_K: never premise-gated (R2-1 iii) -- always None regardless
    # of how clean the diagnostics look.
    entry7 = {"salvage_ratio_mean": 0.9, "value_salvage_ratio_mean": 0.9,
              "alignment_cos_min": 0.99, "key_gram_deviation_mean": 0.0,
              "value_gram_deviation_mean": 0.0}
    check("causal_k_lt_K/never premise-gated -> None",
          _premise_valid_entry(entry7, "causal_k_lt_K", DEFAULT_TH), None)

    # Missing fields read conservatively invalid (not a crash, not a pass).
    check("unconstrained/missing fields -> INVALID (conservative default)",
          _premise_valid_entry({}, "unconstrained", DEFAULT_TH), False)
    check("causal_k_ge_K/missing fields -> INVALID (conservative default)",
          _premise_valid_entry({}, "causal_k_ge_K", DEFAULT_TH), False)

    # Section 14.7 item (2): "This also unifies the unconstrained arm's
    # rule with the causal k >= K leg's rule" -- identical diagnostics must
    # now agree across both arms (previously they diverged on the tau leg).
    entry8 = {
        "key_gram_deviation_mean": 5.0, "value_gram_deviation_mean": 5.0,
        "salvage_ratio_mean": 0.15, "value_salvage_ratio_mean": 0.11,
        "alignment_cos_min": 0.91,
    }
    a = _premise_valid_entry(entry8, "unconstrained", DEFAULT_TH)
    b = _premise_valid_entry(entry8, "causal_k_ge_K", DEFAULT_TH)
    check("unconstrained == causal_k_ge_K on identical salvage/align inputs",
          a == b, True)
    check("  (both VALID under the unified rule)", a, True)


# ---------------------------------------------------------------------------
# Part 2: aggregate() end-to-end fixtures (synthetic per-run JSON on disk)
# ---------------------------------------------------------------------------

def _make_checkpoint(step, recovered_frac, key_gd, val_gd, salv, val_salv, align_min,
                      c17_frac=0.8, c19_frac=0.8):
    entry = {
        "h": 1, "effective_hop": 1,
        "recovered_frac@0.9": recovered_frac,
        "entity_subspace_effective_rank_mean": 15.5,
        "key_gram_deviation_mean": key_gd,
        "value_gram_deviation_mean": val_gd,
        "salvage_ratio_mean": salv,
        "value_salvage_ratio_mean": val_salv,
        "alignment_cos_min": align_min,
    }
    return {
        "step": step,
        "M2_in_distribution": {"1": entry},
        "C17_heldout_entities": {"1": {"recovered_frac@0.9": c17_frac}},
        "C19_heldout_template": {"1": {"recovered_frac@0.9": c19_frac}},
    }


def _make_run(K, force_rank_k, seed, checkpoints, steps=10000, complete=True,
              trunc_impl="eigh"):
    return {
        "K": K, "force_rank_k": force_rank_k, "seed": seed, "steps": steps,
        "steps_completed": steps, "complete": complete, "trunc_impl": trunc_impl,
        "H_train": [1], "skip_rate": 0.0,
        "c16_thresholds": {"tau_unconstrained": 0.03, "salvage_ratio": 0.1,
                            "alignment_cos": 0.9, "tau_value_unconstrained": 0.03,
                            "value_salvage_ratio": 0.1},
        "checkpoints": checkpoints,
    }


def part2_aggregate_end_to_end():
    print("\n--- Part 2: aggregate() end-to-end fixtures ---")
    out_dir = tempfile.mkdtemp(prefix="deltanet_rd_agg_fixture_")
    try:
        # Cell K16_frN (unconstrained): run s0 has ONE premise-valid
        # checkpoint (real-data band, tau fails, salvage+align pass) at
        # step 10000 after an earlier invalid checkpoint at step 2000
        # (low alignment) -- "last premise-valid checkpoint carries the
        # run's number" must pick step 10000, not step 2000 or a mean.
        run_s0 = _make_run(16, None, 0, [
            _make_checkpoint(2000, 0.5, key_gd=2.5, val_gd=2.5, salv=0.3, val_salv=0.3,
                              align_min=0.5),   # alignment fails
            _make_checkpoint(10000, 0.99, key_gd=1.5, val_gd=1.5, salv=0.4, val_salv=0.4,
                              align_min=0.97),  # premise-valid
        ])
        # run s1: never crosses the alignment bar -> premise-failed, must
        # be counted separately, never folded into the headline mean.
        run_s1 = _make_run(16, None, 1, [
            _make_checkpoint(2000, 0.4, key_gd=2.6, val_gd=2.6, salv=0.3, val_salv=0.3,
                              align_min=0.6),
            _make_checkpoint(10000, 0.6, key_gd=2.6, val_gd=2.6, salv=0.3, val_salv=0.3,
                              align_min=0.65),
        ])
        with open(os.path.join(out_dir, "w_test_rd_K16_frN_s0.json"), "w") as f:
            json.dump(run_s0, f)
        with open(os.path.join(out_dir, "w_test_rd_K16_frN_s1.json"), "w") as f:
            json.dump(run_s1, f)

        # Cell K16_fr16 (causal_k_ge_K): one premise-valid run, unaffected
        # by this fix -- a regression pin that the causal branch still works.
        run_causal = _make_run(16, 16, 0, [
            _make_checkpoint(10000, 0.95, key_gd=0.01, val_gd=0.01, salv=0.5, val_salv=0.5,
                              align_min=0.93),
        ])
        with open(os.path.join(out_dir, "w_test_rd_K16_fr16_s0.json"), "w") as f:
            json.dump(run_causal, f)

        # Cell K16_fr8 (causal_k_lt_K, fr < K): never premise-gated
        # (R2-1 iii) -- diagnostic-only, always the final checkpoint.
        run_lt_k = _make_run(16, 8, 0, [
            _make_checkpoint(10000, 0.2, key_gd=9.9, val_gd=9.9, salv=0.0, val_salv=0.0,
                              align_min=0.0),
        ])
        with open(os.path.join(out_dir, "w_test_rd_K16_fr8_s0.json"), "w") as f:
            json.dump(run_lt_k, f)

        aggregate(out_dir, failed=[], wave="test")
        with open(os.path.join(out_dir, "AGGREGATE.json")) as f:
            report = json.load(f)

        by_cell = report.get("by_cell", {})
        cell_unconstrained = by_cell.get("K16_frN", {})
        check("K16_frN/n_premise_valid == 1 (was silently 0 pre-fix)",
              cell_unconstrained.get("n_premise_valid"), 1)
        check("K16_frN/n_premise_failed == 1",
              cell_unconstrained.get("n_premise_failed"), 1)
        check("K16_frN/headline_recovered_frac_h1 == 0.99 (from run s0's last-valid ckpt only)",
              cell_unconstrained.get("headline_recovered_frac_h1"), 0.99)
        check("K16_frN/headline_steps == [10000] (step 2000 excluded, run s1 excluded)",
              cell_unconstrained.get("headline_steps"), [10000])

        cell_causal = by_cell.get("K16_fr16", {})
        check("K16_fr16/n_premise_valid == 1 (causal branch unaffected)",
              cell_causal.get("n_premise_valid"), 1)

        cell_lt_k = by_cell.get("K16_fr8", {})
        check("K16_fr8/n_premise_valid == 0 (never gated, diagnostic-only)",
              cell_lt_k.get("n_premise_valid"), 0)
        check("K16_fr8/n_premise_failed == 0 (never gated, diagnostic-only)",
              cell_lt_k.get("n_premise_failed"), 0)
        check("K16_fr8/premise_gating flag set",
              cell_lt_k.get("premise_gating"), "not_applicable_R2-1(iii)_diagnostics_only")
        check("K16_fr8/headline_recovered_frac_h1 == 0.2 (final checkpoint, unconditional)",
              cell_lt_k.get("headline_recovered_frac_h1"), 0.2)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    part1_premise_valid_entry()
    part2_aggregate_end_to_end()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        raise SystemExit(1)
    print("ALL FIXTURES PASS")
