"""NCR write-capacity diagnostic harvest -- NOVEL_ARCH_WATERFALL.md §9.7/§9.10.

Reads the 4 diagnostic cell JSONs (spare-probe K=14, spare-probe K=15,
Condition A K=16, Condition B K=16), computes the pre-registered
break-or-rescue readouts, and emits the verdict VERDICT-FIRST. NO number
is invented here that isn't already in the pinned §9.7 classification:

  δ(K,cond)  := deep_probe.phase_resid_max_mean (the write-quality residual)
  conv(cell) := min over train_support points of reads[binexp].recovered_frac@0.9
                (in-distribution convergence; a rescue that doesn't converge
                 isn't a rescue -- §9.7)

Verdict (pinned, §9.7):
  RESCUE-CONFIRMED : δ_A(16)/δ_B(16) >= 1.5  AND  conv(CondB K16) >= 0.9
  BREAK-CONFIRMED  : δ_A(16)/δ_B(16) <  1.5  AND  δ(14),δ(15) continue the
                     K=8->K=12 increasing-δ spare-probe trend (monotone up
                     vs the archived anchors δ(8)=0.0055, δ(12)=0.0072)
  MIXED            : anything else -- reported verdict-first with the
                     specific disagreeing readout named.

Archived anchors (§3.2/§7g, 80K-step cells -- the like-for-like budget this
diagnostic also runs at, §9.9): δ(8)=0.0055, δ(12)=0.0072.
"""
from __future__ import annotations

import json
import os
import sys

DELTA8 = 0.0055     # §3.2 archived K=8 mean converged phase residual
DELTA12 = 0.0072    # §7g/§3.2 archived K=12 pooled
GAP_BAR = 1.5       # §9.2 pre-registered cross-condition gap bar
CONV_BAR = 0.9      # §3.2a HOLD band on recovered_frac@0.9

CELLS = {
    "spareK14": "ncr_ncr_K14_s0.json",
    "spareK15": "ncr_ncr_K15_s0.json",
    "condA_K16": "ncr_ncr_K16_d32_h64_s0.json",
    "condB_K16": "ncr_ncr_K16_d32_h128_s0.json",
}


def _load(outdir: str, fname: str) -> dict:
    p = os.path.join(outdir, fname)
    with open(p) as f:
        return json.load(f)


def _delta(rec: dict) -> float:
    return float(rec["deep_probe"]["phase_resid_max_mean"])


def _conv(rec: dict) -> float:
    pts = [e for e in rec["eval"]["points"] if e["component"] == "train_support"]
    return min(float(e["reads"]["binexp"]["recovered_frac@0.9"]) for e in pts)


def _hstar_recovered(rec: dict) -> float | None:
    pts = [e for e in rec["eval"]["points"] if e["component"] == "h_star"]
    if not pts:
        return None
    return float(pts[0]["reads"]["binexp"]["recovered_frac@0.9"])


def harvest(outdir: str) -> dict:
    recs = {}
    for label, fname in CELLS.items():
        rec = _load(outdir, fname)
        assert rec.get("status") == "COMPLETED", (label, rec.get("status"))
        recs[label] = rec

    d14 = _delta(recs["spareK14"])
    d15 = _delta(recs["spareK15"])
    dA16 = _delta(recs["condA_K16"])
    dB16 = _delta(recs["condB_K16"])
    ratio = dA16 / dB16 if dB16 > 0 else float("inf")

    conv = {label: _conv(recs[label]) for label in CELLS}
    hstar = {label: _hstar_recovered(recs[label]) for label in CELLS}

    # spare-probe trend: is δ(14), δ(15) continuing the increasing K=8->K=12
    # spare-fraction-collapse trend? The archived sequence is
    # δ(8)=0.0055 -> δ(12)=0.0072 (increasing). "Continues" = both new
    # points sit at or above δ(12) AND the (14->15) local step is non-
    # decreasing (spare fraction keeps shrinking 0.25->0.125->0.0625).
    trend_seq = [DELTA8, DELTA12, d14, d15]
    trend_continues = (d14 >= DELTA12) and (d15 >= d14)

    gap_clears = ratio >= GAP_BAR
    condB_converges = conv["condB_K16"] >= CONV_BAR

    if gap_clears and condB_converges:
        verdict = "RESCUE-CONFIRMED"
        rationale = (f"δ_A(16)/δ_B(16)={ratio:.3f} >= {GAP_BAR} AND Condition-B "
                     f"in-dist convergence {conv['condB_K16']:.4f} >= {CONV_BAR}: "
                     f"capacity scaling measurably rescues the write.")
    elif (not gap_clears) and trend_continues:
        verdict = "BREAK-CONFIRMED"
        rationale = (f"δ_A(16)/δ_B(16)={ratio:.3f} < {GAP_BAR} (capacity does NOT "
                     f"rescue) AND spare-probe δ continues the K=8->K=12 increasing "
                     f"trend ({DELTA8}->{DELTA12}->{d14:.4f}->{d15:.4f}): the "
                     f"fixed-capacity write breakdown is confirmed at K~16.")
    else:
        verdict = "MIXED"
        bits = []
        if gap_clears and not condB_converges:
            bits.append(f"gap clears ({ratio:.3f}>={GAP_BAR}) BUT Condition-B "
                        f"fails to converge ({conv['condB_K16']:.4f}<{CONV_BAR})")
        if (not gap_clears) and (not trend_continues):
            bits.append(f"gap does not clear ({ratio:.3f}<{GAP_BAR}) AND spare-probe "
                        f"trend does NOT cleanly continue "
                        f"(seq {DELTA8}->{DELTA12}->{d14:.4f}->{d15:.4f})")
        if not bits:
            bits.append(f"ratio={ratio:.3f}, condB_conv={conv['condB_K16']:.4f}, "
                        f"trend_continues={trend_continues}")
        rationale = "MIXED: " + "; ".join(bits) + " -- reported without smoothing."

    out = dict(
        verdict=verdict,
        rationale=rationale,
        delta=dict(K8_archived=DELTA8, K12_archived=DELTA12,
                   K14_spare=d14, K15_spare=d15,
                   A_K16_d32_h64=dA16, B_K16_d32_h128=dB16),
        cross_condition=dict(ratio_A_over_B=ratio, gap_bar=GAP_BAR,
                             gap_clears=gap_clears),
        condB_convergence=dict(in_dist_min_recovered=conv["condB_K16"],
                               bar=CONV_BAR, converges=condB_converges),
        in_dist_convergence_all=conv,
        h_star_recovered_all=hstar,
        spare_probe_trend=dict(sequence_K8_K12_K14_K15=trend_seq,
                               continues_increasing=trend_continues),
        cell_params={label: recs[label]["params"] for label in CELLS},
        cell_gpu_h={label: recs[label].get("gpu_h") for label in CELLS},
        cell_steps={label: recs[label]["train"].get("step") for label in CELLS},
    )
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_wcap_diag")
    out = harvest(outdir)
    print("=" * 68)
    print(f"  NCR WRITE-CAPACITY DIAGNOSTIC VERDICT: {out['verdict']}")
    print("=" * 68)
    print(f"  {out['rationale']}")
    print("-" * 68)
    print(json.dumps(out, indent=2, default=float))
    vp = os.path.join(outdir, "wcap_verdict.json")
    with open(vp, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nverdict written -> {vp}")


if __name__ == "__main__":
    main()
