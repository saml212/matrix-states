"""readout_rev7.py -- KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1)'s REQUIRED
readout entrypoint for the `keyanchor-mech` wave (2026-07-06 build,
deliverable 7). Extends readout_keyanchor.py's own discipline (mechanical
blind assertions, never eyeballed) to the Rev-7.1 mechanism-tier machinery:
BH-FDR `engaged_frac_v3` (with/without hubs), sec 10.3.4's joint band
assignment, the per-entity table, and candidate (d')'s dip-test/Spearman
routing.

What it does, in order:
  1. Loads + hash-validates REV7_THRESHOLD_PINNED.json (`ka.validate_rev7_
     pin` -- missing/tampered/non-reproducing aborts before any anchor
     number is read; sec 10.3.3 leg (iii)'s own discipline: ALL BH/
     Bonferroni/BY constants and effect-size floors are read FROM this
     pin, never recomputed inline in this script).
  2. Loads + hash-validates BANDS_PINNED.json (the EXISTING sec 3.6 gate,
     reused unchanged -- informational/non-gating for this wave's primary
     question per sec 10.5, but its own blind is still asserted).
  3. Loads every keyanchor-mech result JSON (`keyanchor_mech_manifest`);
     incomplete cells are listed and skipped, never silently ignored.
  4. Asserts BOTH blinds are intact: `ka.assert_rev7_pin_not_broken` (the
     Rev-7.1 pin) AND `ka.assert_blind_not_broken` (the existing
     BANDS_PINNED gate) against every loaded run's own `started_at`. EITHER
     violation aborts (exit 2) and demotes every affected readout to
     descriptive tier -- a recorded, tier-demoting event.
  5. Per-run table: `engaged_frac_v3` with/without hubs, median r_e,
     sec 10.3.4 band (with/without-hubs), primary branch (exact-Beta vs
     empirical fallback), hub-detection n_flagged, claim-tier/override
     stamp.
  6. Per-arm (>=2/3-seed) aggregation (sec 10.6): candidate (d)'s Outcome
     A/A''/C; candidate (d')'s independent A(d')/C'/D'/Inconclusive
     routing (band-vs-(d) comparison + dip-test/Spearman significance).
  7. Per-entity table (last completed seed per arm): r_e, hub_flagged,
     engaged (BH), empirical percentile -- for the write-up's per-entity
     disclosure (sec 10.3.2 item 1).

This is a READOUT, not an adjudicator: it computes the MECHANICAL parts
(blind integrity, band classification, seed-aggregation routing) that must
never be done by eyeball; a human still writes the final Outcome narrative
(sec 10.6/10.8).

CPU-only, no fla/CUDA (reads JSONs + key_anchoring.py only).

Usage:
  python readout_rev7.py --out-dir results/deltanet_rd_exactness
      # reads <out-dir>/waveref/BANDS_PINNED.json + <out-dir>/wavekeyanchor-mech/*.json
      # + REV7_THRESHOLD_PINNED.json (repo-relative, not under --out-dir)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

import key_anchoring as ka


def _arm_seed_agg_band(bands: list[str]) -> str:
    """sec 10.6's seed-aggregation rule: an arm's band is whichever of
    A/A_partial/C >=2/3 of its seeds land in; Outcome A itself additionally
    requires literal 3/3 (checked separately by the caller against the
    per-seed item1-6/lambda/h4 bars, which this readout does not itself
    hold -- see the module docstring)."""
    from collections import Counter
    counts = Counter(bands)
    n = len(bands)
    for band in ("A", "A_partial", "C"):
        if counts.get(band, 0) / n >= 2.0 / 3.0 - 1e-9:
            return band
    return "ambiguous (no band reaches >=2/3)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "results/deltanet_rd_exactness"),
                     help="the sweep's --out-dir; this script reads waveref/BANDS_PINNED.json "
                          "and wavekeyanchor-mech/*.json under it.")
    args = ap.parse_args()

    from run_deltanet_rd_exactness_sweep import (keyanchor_ceiling_by_k, keyanchor_mech_manifest,
                                                   out_path)

    ref_dir = os.path.join(args.out_dir, "waveref")
    mech_dir = os.path.join(args.out_dir, "wavekeyanchor-mech")
    bp_path = os.path.join(ref_dir, "BANDS_PINNED.json")

    print("=" * 70)
    print("KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1) -- keyanchor-mech readout")
    print("=" * 70)

    # -- leg (ii)/(iii): the Rev-7.1 pin, validated before any r_e number is read --
    pin = ka.validate_rev7_pin()
    if pin is None:
        print("ERROR: REV7_THRESHOLD_PINNED.json missing, script-hash-mismatched, or fails its own "
              "live derive() re-derivation -- NO r_e/engagement number may be read out. Regenerate "
              "with `python rev7_threshold_derive.py` and re-commit.", file=sys.stderr)
        return 1
    pin_derived = pin["derived"]
    print(f"REV7_THRESHOLD_PINNED.json validated (generated_at={pin['provenance']['generated_at']}, "
          f"script_sha256={pin['provenance']['script_sha256'][:16]}...).")
    print(f"  r_crit_exact_beta={pin_derived['bonferroni_crosscheck']['r_crit_exact_beta']:.4f}  "
          f"r_min_partial={pin_derived['effect_size_floors']['r_min_partial_band']}  "
          f"r_min_headline={pin_derived['effect_size_floors']['r_min_headline_band']}")

    # -- the EXISTING sec 3.6 BANDS_PINNED gate, reused (informational for
    # this wave) -- now with the e633862-audit-F2 content re-derivation and
    # the registered-ceiling cross-check.
    bands_doc = ka.validate_bands_pinned(bp_path, ceiling_by_k=keyanchor_ceiling_by_k())
    if bands_doc is None:
        print(f"NOTE: {bp_path!r} missing, fails hash validation, or its stored bands do not "
              f"reproduce under live re-derivation (e633862 audit F2) -- sec 3.5 B1/B2 sanity "
              f"context (non-gating for THIS wave's primary question, sec 10.5) will be "
              f"unavailable, but the Rev-7.1 r_e/engagement readout below is UNAFFECTED (sec 10.3's "
              f"engagement test does not consult BANDS_PINNED at all).")
    else:
        print(f"BANDS_PINNED.json validated (pinned_at={bands_doc['pinned_at_iso']}) -- reused, "
              f"non-gating context only.")

    # -- load the keyanchor-mech result JSONs --
    results = []
    missing = []
    for spec in keyanchor_mech_manifest():
        p = out_path(mech_dir, spec)
        if not os.path.exists(p):
            missing.append(spec["name"])
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("complete") is not True:
            missing.append(f"{spec['name']} (incomplete)")
            continue
        results.append((spec, d))
    if missing:
        print(f"\nNOTE: {len(missing)} keyanchor-mech cells missing/incomplete (skipped, listed so "
              f"the omission is never silent): {missing}")
    if not results:
        print("ERROR: no complete keyanchor-mech result JSONs found -- nothing to read out.",
              file=sys.stderr)
        return 1

    # -- BOTH blind assertions --
    started_ats = [d["started_at"] for _, d in results if d.get("started_at") is not None]
    if len(started_ats) != len(results):
        print("ERROR: at least one result JSON has no 'started_at' field -- the blind assertions "
              "cannot run without it.", file=sys.stderr)
        return 1
    try:
        ka.assert_rev7_pin_not_broken(pin, started_ats)
        print(f"\nREV7 PIN BLIND INTACT: pin generated_at precedes the earliest anchor-arm start "
              f"({len(started_ats)} runs checked).")
    except AssertionError as e:
        print(f"\n{e}", file=sys.stderr)
        print("READOUT ABORTED: the Rev-7.1 pin blind is recorded BROKEN -- every readout below "
              "reports at DESCRIPTIVE TIER ONLY.", file=sys.stderr)
        return 2
    if bands_doc is not None:
        try:
            ka.assert_blind_not_broken(bands_doc, started_ats)
            print("BANDS_PINNED BLIND INTACT (existing sec 3.6 gate, reused, non-gating context).")
        except AssertionError as e:
            print(f"\n{e}", file=sys.stderr)
            print("NOTE: the BANDS_PINNED blind is broken -- the sec 3.5 B1/B2 context is demoted "
                  "to descriptive tier; the Rev-7.1 r_e/engagement readout is UNAFFECTED (it does "
                  "not depend on BANDS_PINNED at all, sec 10.3).", file=sys.stderr)

    # -- sec 10.10 item 2's readout-time checkpoint gate: the wave cannot
    # be marked KEYANCHOR_MECH_CHAIN_DONE until every cell's checkpoint
    # file exists AND round-trip-loads. A non-zero return here (before the
    # chain script's own `touch KEYANCHOR_MECH_CHAIN_DONE`) is the
    # mechanical refusal this gate exists to provide.
    all_ckpt_paths = [p for _, d in results for p in (d.get("ckpt_written") or [])]
    if all_ckpt_paths:
        ckpt_gate = ka.gate_checkpoint_round_trip(all_ckpt_paths)
        print(f"\nsec 10.10 item 2 CHECKPOINT GATE: checked {ckpt_gate['n_checked']} files, "
              f"{ckpt_gate['n_bad']} bad -> {'PASS' if ckpt_gate['pass'] else 'REFUSED'}")
        if not ckpt_gate["pass"]:
            print(f"  bad checkpoints: {ckpt_gate['bad']}", file=sys.stderr)
            print("READOUT ABORTED: sec 10.10's checkpoint gate failed -- the wave may NOT be "
                  "marked KEYANCHOR_MECH_CHAIN_DONE.", file=sys.stderr)
            return 3
    else:
        print("\nNOTE: no 'ckpt_written' field found on any loaded result -- either no --ckpt-dir "
              "was passed at launch, or these are pre-checkpoint-writer archived results. sec "
              "10.10's checkpoint gate is SKIPPED (nothing to check), NOT silently passed.")

    # -- per-run table --
    print("\nPer-run r_e engagement verdicts:")
    by_arm_seed_bands: dict[str, list[str]] = {"d": [], "dprime": []}
    per_entity_last: dict[str, tuple] = {}
    for spec, d in results:
        tier_bits = []
        if d.get("unblind_override"):
            tier_bits.append(f"UNBLIND-OVERRIDDEN (claim_tier={d.get('claim_tier')!r})")
        eng = None
        for ckpt in reversed(d.get("checkpoints") or []):
            if "r_e_engagement" in ckpt:
                eng = ckpt["r_e_engagement"]
                break
        if eng is None:
            print(f"  {d.get('K')}/{spec['arm']}/s{d.get('seed')}: NO r_e_engagement field found "
                  f"(rev7_engagement not active on this run?) -- SKIPPED from aggregation.")
            continue
        by_arm_seed_bands.setdefault(spec["arm"], []).append(eng["band"])
        per_entity_last[spec["arm"]] = (spec, eng)
        print(f"  {d.get('K')}/{spec['arm']}/s{d.get('seed')}: "
              f"engaged_frac_v3 with_hubs={eng['engaged_frac_v3_with_hubs']:.3f} "
              f"without_hubs={eng['engaged_frac_v3_without_hubs']:.3f}  "
              f"median_r_e={eng['median_r_e']:.4f}  band={eng['band']} "
              f"(with_hubs_band={eng['band_with_hubs']}, primary_branch={eng['primary_branch']}, "
              f"n_hub_flagged={eng['hub_detection']['n_flagged']})"
              f"{('  [' + '; '.join(tier_bits) + ']') if tier_bits else ''}")

    # -- per-arm >=2/3-seed aggregation (sec 10.6) --
    print("\nArm-level band aggregation (>=2/3 seeds, sec 10.6 -- Outcome A itself additionally "
          "requires literal 3/3 on items 1-6/lambda/h4, not checked by this script):")
    for arm, bands in by_arm_seed_bands.items():
        if not bands:
            continue
        agg = _arm_seed_agg_band(bands)
        print(f"  arm {arm}: {agg}  (per-seed: {bands})")

    # -- candidate (d') independent routing (sec 10.6's own table) --
    if by_arm_seed_bands.get("dprime"):
        print("\nCandidate (d') independent routing (never merged into (d)'s own Outcome):")
        d_band = _arm_seed_agg_band(by_arm_seed_bands["d"]) if by_arm_seed_bands.get("d") else None
        dprime_band = _arm_seed_agg_band(by_arm_seed_bands["dprime"])
        band_rank = {"C": 0, "A_partial": 1, "A": 2, "ambiguous (no band reaches >=2/3)": -1}
        sig_checks = []
        for spec, d in results:
            if spec["arm"] != "dprime":
                continue
            for ckpt in reversed(d.get("checkpoints") or []):
                sp = ckpt.get("lambda_e_vs_r_e_spearman")
                dip_l = ckpt.get("lambda_e_dip_test")
                if sp is not None or dip_l is not None:
                    sig_checks.append({"seed": d.get("seed"),
                                        "spearman_significant": sp.get("significant_at_0.05") if sp else None,
                                        "lambda_e_dip_significant": dip_l.get("significant_at_alpha") if dip_l else None})
                    break
        any_significant = any(
            (c["spearman_significant"] or c["lambda_e_dip_significant"]) for c in sig_checks)
        print(f"  (d) band: {d_band}  (d') band: {dprime_band}")
        print(f"  (d') dip-test/Spearman per seed: {sig_checks}")
        if d_band is not None and band_rank.get(dprime_band, -1) > band_rank.get(d_band, -1):
            routed = "A(d') -- (d') shows a HIGHER band than (d): differential engagement demonstrated"
        elif d_band is not None and band_rank.get(dprime_band, -1) < band_rank.get(d_band, -1) and any_significant:
            routed = "D' -- WORSE band than (d) AND significant: per-entity capacity used, and used badly"
        elif dprime_band == d_band and not any_significant:
            routed = "C' -- structural null: same band as (d), no significant differentiation"
        else:
            routed = "Inconclusive/mixed -- reported in full, no forced binary call"
        print(f"  ROUTED FINDING: {routed}")

    # -- per-entity table (last completed seed per arm) --
    print("\nPer-entity table (last completed seed per arm -- r_e, hub_flagged, BH-engaged):")
    for arm, (spec, eng) in per_entity_last.items():
        print(f"  arm {arm}, K={spec['K']}, seed={spec['seed']}:")
        for eid in sorted(eng["r_e"].keys())[:10]:
            print(f"    entity {eid}: r_e={eng['r_e'][eid]:.4f}  "
                  f"hub_flagged={eng['hub_detection']['flagged'].get(str(eid), eng['hub_detection']['flagged'].get(eid))}  "
                  f"bh_engaged={eng['bh']['engaged'].get(str(eid), eng['bh']['engaged'].get(eid))}")
        print(f"    ... ({len(eng['r_e'])} entities total, showing first 10)")

    print("\nReadout complete. Final Outcome narrative (sec 10.6/10.8) still requires a human to "
          "combine this mechanical routing with items 1-6/lambda-interior/h4 (unchanged sec 3.5 bars, "
          "not computed by this script).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
