#!/usr/bin/env python3
"""SIX-RUNG DEPTH EXTENSION DRIVER -- squarings {5,7,9,11,13,15}.

NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.6 / sec 4.6.1 (the FATAL-2 repair) and
sec 5.2 Rule R-delta. This runs on the 98M side ONLY, from the kscaling tree
(sec 3.5: "the 98M re-score of sec 4.6 runs from the kscaling tree").

WHY A DRIVER AND NOT AN EDIT.  depthext_eval.py (md5
e95ffe192c66d3e054f3febc37fe4a91, the four-rung instrument of record that
produced experiment-runs/2026-08-22_depthext_across_k/) carries its ladder
profile as ONE module constant, SQUARING_PROFILE = (5,7,9,11); everything
else -- derive_depth_ladder, verify_ladder, the guards, eval_arm_at_hops,
matched pools, the K/d_ncr guard, the ckpt_step guard, the seed and freeze
resolution, the self-check -- is already profile-generic and reads that
constant.  So the extension needs ZERO edits to the audited wrapper: this
driver imports it as a module, sets the one constant, and calls its own
main().  The audited file on disk is never written to and its md5 is
re-verified here before and after the run.

THE RECEIPT THAT NOTHING ELSE MOVED.  (5,7,9,11) is a strict PREFIX of
(5,7,9,11,13,15) and derive_depth_ladder picks the smallest admissible h in
each band independently, so rungs 1-4 are the SAME integers as the archived
run.  Pools, eval seed, n and the metric are unchanged, so the four archived
accuracies must reproduce EXACTLY.  `--assert-prefix` (default on) enforces
that the requested profile starts with the archived one; compare_prefix.py
checks the reproduction against the archived JSONs afterwards.

THE ONE COSMETIC FIELD THAT WOULD OTHERWISE LIE.  depthext_eval.main() hard-
codes the P0 block's `note` as "deepest rung only (11 squarings), per the #2
cost cap".  With six rungs the P0 probe runs at the 15-squaring rung, so the
string would be false.  It is a label, not a number, and is corrected in
place after the record is written, together with an explicit `driver`
provenance block naming this file, the parent md5 and the requested profile.
Nothing numeric is touched.

Usage (one cell), exactly the argument surface of depthext_eval.py:
  NCR_K=16 python3 depthext6_driver.py --k 16 --ckpt ... --tag depthext6_... \\
      [--cellcfg ...] [--anchor-runner-tag ncr_gate3_wave1_runner_v1] [--outdir ...]
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
PARENT_MD5 = "e95ffe192c66d3e054f3febc37fe4a91"   # the four-rung instrument of record
ARCHIVED_PROFILE = (5, 7, 9, 11)
EXTENDED_PROFILE = (5, 7, 9, 11, 13, 15)


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main() -> int:
    got = md5(PARENT)
    if got != PARENT_MD5:
        print(f"!!! PIN MISMATCH: {PARENT} md5={got} != pinned {PARENT_MD5}. "
              f"The four-rung instrument of record changed -- STOP and adjudicate.", flush=True)
        return 6

    raw = os.environ.get("DEPTHEXT_SQUARINGS", ",".join(str(s) for s in EXTENDED_PROFILE))
    profile = tuple(int(x) for x in raw.split(","))
    if profile[:len(ARCHIVED_PROFILE)] != ARCHIVED_PROFILE:
        print(f"!!! PROFILE ABORT: requested {profile} does not start with the archived "
              f"{ARCHIVED_PROFILE}; rungs 1-4 would not be the reproduction cross-check.",
              flush=True)
        return 6

    import depthext_eval as DX            # noqa: E402  (audited wrapper, imported unmodified)
    DX.SQUARING_PROFILE = profile

    # depthext_eval.main() parses sys.argv itself; we pass ours through untouched.
    rc = DX.main()

    if md5(PARENT) != PARENT_MD5:
        print("!!! the audited wrapper was modified during the run -- ABORT", flush=True)
        return 6
    if rc != 0:
        return rc

    # ---- correct the one cosmetic label, and stamp provenance ---------------
    tag = sys.argv[sys.argv.index("--tag") + 1]
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.expanduser("~/ncr_kscaling/results"))
    out = os.path.join(outdir, f"{tag}_depthext.json")
    rec = json.load(open(out))
    deepest = rec["depth_ladder"][-1]
    rec["matched"]["P0"]["note"] = (
        f"deepest rung only ({profile[-1]} squarings, h={deepest}); the #2 cost cap carried "
        f"to the extended profile. NOTE: the archived four-rung wave ran P0 at 11 squarings, "
        f"so this P0 reading is NOT comparable to it -- P1b is what Rule R-delta reads.")
    rec["driver"] = {
        "driver": "depthext6_driver.py",
        "parent": "depthext_eval.py",
        "parent_md5": PARENT_MD5,
        "parent_unmodified": True,
        "squaring_profile_requested": list(profile),
        "archived_profile_is_prefix": True,
        "prereg_extension": ("NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.6.1 (the FATAL-2 repair) "
                             "+ sec 5.2 Rule R-delta; the {13,15} rungs come from the SAME rule "
                             "of record (smallest h in [2^s, 2^(s+1)) with h == r_fix mod K)."),
        "extends_never_retracts": ("sec 4.6.1 condition 2: the 13/15 readings EXTEND #8's "
                                   "11-squaring verdict of record and never retract it."),
        "only_field_modified_post_hoc": "matched.P0.note (a label, not a number)",
    }
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
