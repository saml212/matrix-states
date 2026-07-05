"""readout_keyanchor.py -- KEY_ANCHORING_DESIGN.md sec 3.6 item (c)'s
REQUIRED readout entrypoint for the keyanchor wave (registered as such per
the 2026-07-04 audit fix, MAJOR: `ka.assert_blind_not_broken` previously
had zero call sites -- this script is its consumer).

What it does, in order:
  1. Loads + hash-validates BANDS_PINNED.json from the reference wave's
     out_dir (`ka.validate_bands_pinned` -- a tampered/missing file aborts
     before any anchor number is read).
  2. Loads every keyanchor-wave anchor-arm result JSON (the 12-cell
     manifest from run_deltanet_rd_exactness_sweep.keyanchor_wave1_manifest;
     incomplete cells are listed and skipped, never silently ignored).
  3. Runs `ka.assert_blind_not_broken(bands_doc, [every run's started_at])`
     -- sec 3.6 item (c)'s mechanical assertion that the pin timestamp
     strictly precedes the earliest anchor-arm start. On violation: the
     readout ABORTS (exit 2), prints the broken-blind verdict, and states
     that every affected anchor readout is demoted to descriptive tier --
     a recorded, tier-demoting event, not a judgment call.
  4. Prints the per-seed lambda band verdicts (sec 3.2's three-part rule,
     from each run's own `anchor_lambda_summary` field), the arm-level
     >=2/3 label (`ka.arm_lambda_label`), each run's claim-tier /
     unblind_override stamp, and the derived engaged bands (or
     UNRESOLVABLE verdicts) per K.

This is a READOUT, not an adjudicator: sec 3.5's Outcome A/A'/A''/B/C
assignment additionally needs the item-5 pre-NS drift bar, items 6a/6b at
final admission, h4 vs the 0.5 bar, and sec 3.7's engaged_frac -- those
reads stay with the human-audited assessment step; this script's job is
the MECHANICAL part (blind integrity + band classification) that must
never be done by eyeball.

CPU-only, no fla/CUDA (reads JSONs + key_anchoring.py only).

Usage:
  python readout_keyanchor.py --out-dir results/deltanet_rd_exactness
      # reads <out-dir>/waveref/BANDS_PINNED.json + <out-dir>/wavekeyanchor/*.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

import key_anchoring as ka


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "results/deltanet_rd_exactness"),
                     help="the sweep's --out-dir; this script reads waveref/BANDS_PINNED.json "
                          "and wavekeyanchor/*.json under it.")
    args = ap.parse_args()

    # deferred import: pulls in key_anchoring only (fla-free) -- the sweep
    # module itself imports no fla either, so this stays CPU-safe.
    from run_deltanet_rd_exactness_sweep import keyanchor_wave1_manifest, out_path

    ref_dir = os.path.join(args.out_dir, "waveref")
    anchor_dir = os.path.join(args.out_dir, "wavekeyanchor")
    bp_path = os.path.join(ref_dir, "BANDS_PINNED.json")

    print("=" * 70)
    print("KEY_ANCHORING readout (sec 3.6 item (c) + sec 3.2 band verdicts)")
    print("=" * 70)

    bands_doc = ka.validate_bands_pinned(bp_path)
    if bands_doc is None:
        print(f"ERROR: {bp_path!r} missing or FAILED hash validation -- no anchor number may be "
              f"read out. Run the reference arms + '--wave keyanchor-bands' first.", file=sys.stderr)
        return 1
    print(f"BANDS_PINNED.json validated (pinned_at={bands_doc['pinned_at_iso']}).")
    for K, entry in bands_doc["bands"].items():
        verdict = "UNRESOLVABLE" if entry["unresolvable"] else f"engaged_K={entry['engaged_k']:.4f}"
        print(f"  K={K}: {verdict} (mean_ref={entry['mean_ref']:.4f} s_ref={entry['s_ref']:.4f} "
              f"ceiling={entry['ceiling']:.4f}; leave-one-out "
              f"{[round(v, 4) for v in entry['leave_one_out_engaged_k']]})")

    # -- load the anchor-arm results --
    results = []
    missing = []
    for spec in keyanchor_wave1_manifest():
        p = out_path(anchor_dir, spec)
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
        print(f"\nNOTE: {len(missing)} keyanchor cells missing/incomplete (skipped, listed so the "
              f"omission is never silent): {missing}")
    if not results:
        print("ERROR: no complete keyanchor result JSONs found -- nothing to read out.",
              file=sys.stderr)
        return 1

    # -- sec 3.6 item (c): the mechanical blind assertion --
    started_ats = []
    for spec, d in results:
        sa = d.get("started_at")
        if sa is None:
            print(f"ERROR: {spec['name']} has no 'started_at' field -- the blind assertion cannot "
                  f"run without it (a pre-audit-fix result JSON? re-run the cell).", file=sys.stderr)
            return 1
        started_ats.append(sa)
    try:
        ka.assert_blind_not_broken(bands_doc, started_ats)
        print(f"\nBLIND INTACT: pin timestamp {bands_doc['pinned_at']:.0f} strictly precedes the "
              f"earliest anchor-arm start {min(started_ats):.0f} ({len(started_ats)} runs checked).")
    except AssertionError as e:
        print(f"\n{e}", file=sys.stderr)
        print("READOUT ABORTED: the blind is recorded BROKEN -- every anchor readout above/below "
              "reports at DESCRIPTIVE TIER ONLY (sec 3.6 item (c)'s tier-demoting event).",
              file=sys.stderr)
        return 2

    # -- per-run verdicts --
    print("\nPer-run verdicts:")
    bands_by_arm_k: dict[tuple, list] = {}
    for spec, d in results:
        tier_bits = []
        if d.get("unblind_override"):
            tier_bits.append(f"UNBLIND-OVERRIDDEN (claim_tier={d.get('claim_tier')!r}, "
                              f"at={d.get('unblind_override_at')})")
        summ = d.get("anchor_lambda_summary")
        if summ is not None:
            band = summ["band"]
            bands_by_arm_k.setdefault((spec["arm"], spec["K"]), []).append(band)
            lam_str = (f"lambda final={summ['final_value']:.4f} "
                       f"mean5={summ['trailing_mean']:.4f} range5={summ['trailing_range']:.4f} "
                       f"-> band={band}")
        else:
            lam_str = "no anchor_lambda_summary (candidate (c) / fixed-lambda cell)"
        print(f"  {d.get('K')}/{spec['arm']}/s{d.get('seed')}: {lam_str}"
              f"{('  [' + '; '.join(tier_bits) + ']') if tier_bits else ''}")

    print("\nArm-level lambda labels (>=2/3 rule, sec 3.2):")
    for (arm, K), band_list in sorted(bands_by_arm_k.items()):
        print(f"  arm {arm} K={K}: {ka.arm_lambda_label(band_list)}  (per-seed: {band_list})")

    print("\nReadout complete. Outcome A/A'/A''/B/C assignment additionally requires the item-5 "
          "pre-NS drift bar, items 6a/6b at final admission, h4 vs 0.5, and sec 3.7's "
          "engaged_frac -- see KEY_ANCHORING_DESIGN.md sec 3.5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
