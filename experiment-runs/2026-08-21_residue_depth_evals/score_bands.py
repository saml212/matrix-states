#!/usr/bin/env python3
"""Score the 2026-08-21 #2 pre-registered bands from the raw output JSONs.

Bands are quoted VERBATIM from EXPERIMENT_LOG.md "2026-08-21 #2" and applied
mechanically. This prints verdicts for the coordinator to adjudicate; it does
NOT write to EXPERIMENT_LOG.md or STATE.md.

  A (frozen arms primary_s1 + compA_s1):
    COMPLETE-COVERAGE = retrieval24_acc >= 0.95 at ALL 15 residues on BOTH
    PARTIAL           = any residue in [0.5, 0.95)
    GAP-FOUND         = any residue < 0.5
  B:
    ROBUST   = every deeper reading within 0.02 of the h=13 reading
    DRIFT    = monotone decline > 0.02
    UNSTABLE = non-monotone / NaN
"""
import json
import math
import os
import sys

HOPS_A = (4, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 21, 22, 23)
HOPS_B = (13, 61, 253, 1021, 4093)
M = "retrieval24_acc"
FROZEN_ARMS = ("primary_s1", "compA_s1")


def load(d, tag):
    p = os.path.join(d, f"residue_depth_{tag}.json")
    if not os.path.exists(p):
        sys.exit(f"MISSING OUTPUT: {p}")
    return json.load(open(p))


def band_A(vals):
    if any(v < 0.5 for v in vals.values()):
        return "GAP-FOUND"
    if any(0.5 <= v < 0.95 for v in vals.values()):
        return "PARTIAL"
    return "COMPLETE-COVERAGE (>=0.95 at all 15)"


def band_B(vals):
    seq = [vals[h] for h in HOPS_B]
    if any(not math.isfinite(v) for v in seq):
        return "UNSTABLE (non-finite)"
    ref = seq[0]
    deltas = [v - ref for v in seq[1:]]
    if all(abs(d) <= 0.02 for d in deltas):
        return "ROBUST (all within 0.02 of h=13)"
    monotone_nonincreasing = all(seq[i + 1] <= seq[i] for i in range(len(seq) - 1))
    if monotone_nonincreasing and (ref - seq[-1]) > 0.02:
        return "DRIFT (monotone decline > 0.02)"
    return "UNSTABLE (non-monotone beyond 0.02)"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    tags = sys.argv[2:] or ["primary_s1", "compA_s1", "compB_s1"]
    print(f"metric of record: {M} | regime of record: P1b (teacher_force=True, EXACT-WRITE)")
    print("P0 (teacher_force=False, learned-write) is INFORMATIONAL ONLY.\n")
    for tag in tags:
        rec = load(d, tag)
        frozen = tag in FROZEN_ARMS
        print(f"=== {tag} === ckpt_step={rec['ckpt_step']} freeze={rec['freeze_entity_adapter']} "
              f"pool_seed={rec['pool_seed']} ckpt_seed={rec['ckpt_seed']} "
              f"self_check={rec['self_check']} "
              f"[{'FROZEN/composing arm -- BANDED' if frozen else 'DEGRADED arm -- INFORMATIONAL, not banded'}]")
        for exp, hops, bander in (("experiment_A", HOPS_A, band_A), ("experiment_B", HOPS_B, band_B)):
            for regime in ("P1b", "P0"):
                res = rec[exp][regime]["result"]
                vals = {h: res[f"h={h}"][M] for h in hops}
                line = "  ".join(f"h={h}:{vals[h]:.4f}" for h in hops)
                label = "REGIME OF RECORD" if regime == "P1b" else "informational"
                print(f"  {exp} / {regime} ({label}): {line}")
                if regime == "P1b":
                    if exp == "experiment_A":
                        below = sorted(h for h, v in vals.items() if v < 0.95)
                        print(f"    -> band: {bander(vals)}"
                              + (f" | residues below 0.95: {below}" if below else " | none below 0.95"))
                    else:
                        ref = vals[13]
                        print(f"    -> band: {bander(vals)} | deltas vs h=13: "
                              + ", ".join(f"h={h}:{vals[h]-ref:+.4f}" for h in HOPS_B[1:]))
                        print("    -> NOTE: all five h are == 13 (mod 24); SAME ground truth by "
                              "construction. Numerical robustness only.")
        print()

    print("--- combined A verdict over the two FROZEN arms ---")
    worst = []
    for tag in FROZEN_ARMS:
        rec = load(d, tag)
        res = rec["experiment_A"]["P1b"]["result"]
        worst += [(tag, h, res[f"h={h}"][M]) for h in HOPS_A]
    below = [(t, h, v) for t, h, v in worst if v < 0.95]
    allv = {t: {h: v for tt, h, v in worst if tt == t} for t in FROZEN_ARMS}
    if any(v < 0.5 for _, _, v in worst):
        verdict = "GAP-FOUND"
    elif below:
        verdict = "PARTIAL"
    else:
        verdict = "COMPLETE-COVERAGE"
    print(f"A = {verdict}" + (f" | cells below 0.95: {below}" if below else ""))
    print(f"    min over both frozen arms: " +
          ", ".join(f"{t}={min(v.values()):.4f}" for t, v in allv.items()))


if __name__ == "__main__":
    main()
