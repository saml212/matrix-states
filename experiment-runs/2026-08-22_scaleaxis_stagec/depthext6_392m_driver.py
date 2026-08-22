#!/usr/bin/env python3
"""SIX-RUNG DEPTH EXTENSION DRIVER, 392M SIDE -- squarings {5,7,9,11,13,15}.

NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.6 (Stage C) + sec 4.6.1's ladder table.

This is the 392M twin of `depthext6_driver.py` (which sec 3.5 confines to the
98M kscaling tree). Same construction, same discipline, one difference: the
parent is the SCALE-AXIS wrapper `~/ncr_scaleaxis/depthext_eval.py`, which
carries build requirement B5's scale guard, and the pin is that file's md5.

WHY A DRIVER AND NOT AN EDIT.  depthext_eval.py carries its ladder profile as
ONE module constant, SQUARING_PROFILE = (5,7,9,11); everything else --
derive_depth_ladder, verify_ladder, the single-residue ladder_guard, the
K/d_ncr guard, the B5 SCALE guard, the ckpt_step guard, seed/freeze
resolution from the checkpoint, matched pools, eval_arm_at_hops and the
self-check -- is profile-generic and reads that constant. So the extension
needs ZERO edits to the audited wrapper: import it, set the one constant,
call its main(). The audited file is never written to and its md5 is
re-verified before AND after the run.

THE RECEIPT.  (5,7,9,11) is a strict PREFIX of (5,7,9,11,13,15) and
derive_depth_ladder picks the smallest admissible h in each band
independently, so rungs 1-4 are the SAME integers at every K. The resulting
ladders must equal sec 4.6.1's pinned table:
    K=16 [36,132,516,2052,8196,32772]   K=24 [52,148,532,2068,8212,32788]
    K=32 [36,132,516,2052,8196,32772]   K=40 [44,164,524,2084,8204,32804]
This driver asserts that table by value -- the design's own numbers, not a
recomputation trusted on faith.

THE ONE COSMETIC FIELD THAT WOULD OTHERWISE LIE. depthext_eval.main() hard-
codes the P0 block's `note` as "deepest rung only (11 squarings)". With six
rungs P0 runs at the 15-squaring rung, so the string would be false. It is a
label, not a number, corrected in place after the record is written, with a
`driver` provenance block. Nothing numeric is touched.

Usage (one cell):
  NCR_SCALE=392m NCR_K=16 python3 depthext6_392m_driver.py --k 16 --ckpt ... \\
      --tag depthext6_392m_... [--cellcfg ...] [--outdir ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PARENT = os.path.join(_HERE, "depthext_eval.py")
PARENT_MD5 = "e48eca7142382b20a4b467711c872ae5"   # ~/ncr_scaleaxis/depthext_eval.py (B5 scale guard)
ARCHIVED_PROFILE = (5, 7, 9, 11)
EXTENDED_PROFILE = (5, 7, 9, 11, 13, 15)

# NCR_SCALE_AXIS_DESIGN.md sec 4.6.1, the pinned ladder table (design values).
PINNED_LADDERS = {
    16: [36, 132, 516, 2052, 8196, 32772],
    24: [52, 148, 532, 2068, 8212, 32788],
    32: [36, 132, 516, 2052, 8196, 32772],
    40: [44, 164, 524, 2084, 8204, 32804],
}


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main() -> int:
    if os.environ.get("NCR_SCALE", "").strip().lower() != "392m":
        print(f"!!! SCALE ABORT: NCR_SCALE={os.environ.get('NCR_SCALE')!r}, expected '392m'. "
              f"This driver is the 392M side; the 98M side runs depthext6_driver.py from the "
              f"kscaling tree (design sec 3.5).", flush=True)
        return 6

    got = md5(PARENT)
    if got != PARENT_MD5:
        print(f"!!! PIN MISMATCH: {PARENT} md5={got} != pinned {PARENT_MD5}. The audited "
              f"scale-axis wrapper changed -- STOP and adjudicate.", flush=True)
        return 6

    raw = os.environ.get("DEPTHEXT_SQUARINGS", ",".join(str(s) for s in EXTENDED_PROFILE))
    profile = tuple(int(x) for x in raw.split(","))
    if profile[:len(ARCHIVED_PROFILE)] != ARCHIVED_PROFILE:
        print(f"!!! PROFILE ABORT: requested {profile} does not start with the archived "
              f"{ARCHIVED_PROFILE}; rungs 1-4 would not be the cross-check.", flush=True)
        return 6

    import depthext_eval as DX            # noqa: E402  (audited wrapper, imported unmodified)
    DX.SQUARING_PROFILE = profile

    rc = DX.main()

    if md5(PARENT) != PARENT_MD5:
        print("!!! the audited wrapper was modified during the run -- ABORT", flush=True)
        return 6
    if rc != 0:
        return rc

    tag = sys.argv[sys.argv.index("--tag") + 1]
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.expanduser("~/ncr_scaleaxis/results"))
    out = os.path.join(outdir, f"{tag}_depthext.json")
    rec = json.load(open(out))

    # ---- assert the realized ladder against the DESIGN's pinned table -------
    k = int(rec["K"])
    if profile == EXTENDED_PROFILE:
        if PINNED_LADDERS.get(k) != list(rec["depth_ladder"]):
            print(f"!!! LADDER MISMATCH [{tag}]: realized {rec['depth_ladder']} != design sec "
                  f"4.6.1 pinned {PINNED_LADDERS.get(k)} at K={k} -- STOP.", flush=True)
            return 6

    deepest = rec["depth_ladder"][-1]
    rec["matched"]["P0"]["note"] = (
        f"deepest rung only ({profile[-1]} squarings, h={deepest}); the #2 cost cap carried "
        f"to the extended profile. NOTE: the archived four-rung 98M wave ran P0 at 11 "
        f"squarings, so this P0 reading is NOT comparable to it -- P1b is what the "
        f"cross-scale tests read.")
    rec["driver"] = {
        "driver": "depthext6_392m_driver.py",
        "parent": "depthext_eval.py (scale-axis tree, B5 scale guard)",
        "parent_md5": PARENT_MD5,
        "parent_unmodified": True,
        "scale": "392m",
        "squaring_profile_requested": list(profile),
        "archived_profile_is_prefix": True,
        "ladder_matches_design_4_6_1_table": True,
        "prereg": "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.6 Stage C + sec 4.6.1 ladder table",
        "only_field_modified_post_hoc": "matched.P0.note (a label, not a number)",
    }
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
