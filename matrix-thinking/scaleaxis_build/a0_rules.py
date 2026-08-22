#!/usr/bin/env python3
"""STAGE A0.5 -- Rules P1, P2, P3, P4, EVALUATED.
NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.4 (as corrected by verify-R2 FATAL-1
for P1's instrument and verify-R2 MAJOR-1 for P4's), sec 4.4.1 and sec 3.6.

Every number here is MEASURED at Stage A0, before any training cell exists.
No rule reads a number its elected instrument does not produce -- that is the
defect class (FATAL-3 and its two repeats) this file exists not to commit.

  P1  R := phase0(392M,K).mean_s_per_step_both_arms_combined
         / phase0( 98M,K).mean_s_per_step_both_arms_combined,   max over K
      LIKE-FOR-LIKE: the 1.5500x per-arm-synchronize inflation appears
      identically in numerator and denominator and CANCELS EXACTLY.
        R <= 4.0        NOMINAL
        4.0 < R <= 4.5  PROCEED WITH A RECORDED MISS
        R > 4.5         COST-OUT -> sec 4.4.1's publishable K=24 FROZEN TRIO
                        (n=3, the wave-0 minimum), then STOP.
      The threshold was MOVED 5.0 -> 4.5 at DRAFT-R2 on one consistent basis:
      R=4.5 => 111.6 GPU-h headline (the ~112 tier-(c) line both R0 and R1 used
      as their own boundary); R=5.0 => 123.9, and ~150 with both contingencies.
      Anomaly leg agrees: G = R/3.50 = 1.29, i.e. the graft-specific components
      may add 29% on top of the backbone's own measured 3.48-3.51x scale-up.

  P2  R_8 := 392M 8-way / 392M solo   (phase0/phase0; unaffected by FATAL-1)
        <= 1.25  nominal;  > 1.25  DO NOT QUEUE WAVE 1, re-price and re-enter P1.

  P3  if R(40) > 1.15 x R(24) the graft overhead is T-dependent: re-derive the
      ledger PER K from R(K), with K=32/K=16 interpolated in t_in and FLAGGED
      as interpolations.

  P4  memory, read from scaleaxis_gates.py's B8 block (the ONLY instrument that
      emits one -- run_phase0_timing records NO memory or utilisation field of
      any kind, which is why R1's "peak VRAM at A0.3" was unmeasurable):
        < 40 GB   nominal;  >= 40 GB  not a blocker, but sec 8.3's placement
                  assumption RE-OPENS and must be adjudicated before Stage B.
      SM utilisation likewise comes from B8's external nvidia-smi sampler;
      sustained < 50% is treated as a BUG and diagnosed before any cell queues.

  CROSS-CHECK (free, pinned): the fresh 98M K=24 probe must land within +-10%
  of the ARCHIVED 0.23075456221898397
  (experiment-runs/2026-07-17_ncr_gate3_wave1/phase0_timing.json). A larger gap
  is an instrument-drift signal and is REPORTED before A0.5 is applied -- it
  does not by itself block, because R is a ratio of two FRESH probes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ARCHIVED_98M_K24_PHASE0 = 0.23075456221898397      # sec 4.0's pinned cross-check
CROSSCHECK_TOL = 0.10
GPU_H_98M_MEASURED = {16: 0.8019, 24: 0.8271, 32: 0.9583, 40: 1.1309}   # sec 8.2
S_PER_STEP_98M_REALIZED = {24: 0.14888, 40: 0.20357}                    # sec 8.2
PLAIN_BACKBONE_RATIO = 3.50        # sec 4.4's five-measurement central estimate
P1_NOMINAL, P1_COSTOUT = 4.0, 4.5
P2_BAR = 1.25
P3_BAR = 1.15
P4_LIMIT_GB = 40.0
CELLS = {16: 6, 24: 6, 32: 6, 40: 6}


def load_probes(a0_dir: str) -> dict:
    """-> {(scale, K): mean_s_per_step_both_arms_combined}, plus the 8-way list."""
    solo, eight = {}, []
    for p in sorted(glob.glob(os.path.join(a0_dir, "*.json"))):
        d = json.load(open(p))
        if d.get("mode") != "phase0-timing":
            continue
        ks = d.get("kscaling") or {}
        scale = ks.get("scale") or ("98m" if d["config"]["backbone"]["d_model"] == 768
                                    else "392m")
        k = int(d["config"]["K"])
        v = float(d["measured"]["mean_s_per_step_both_arms_combined"])
        if "8way" in os.path.basename(p):
            eight.append(dict(file=os.path.basename(p), K=k, s_per_step=v))
        else:
            solo[(scale, k)] = dict(file=os.path.basename(p), s_per_step=v,
                                    suggested_ceiling_gpuh=float(
                                        d["projected"]["suggested_ceiling_gpuh"]),
                                    contended_gpuh=float(
                                        d["projected"]["contended_gpuh_for_target_steps"]),
                                    host=d.get("host"), torch=d.get("torch_version"))
    return solo, eight


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-dir", required=True)
    ap.add_argument("--gates", default=None, help="scaleaxis_gates.py output, for P4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    solo, eight = load_probes(args.a0_dir)
    rec: dict = {"stage": "A0.5", "design": "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.4",
                 "a0_dir": os.path.abspath(args.a0_dir),
                 "solo_probes": {f"{s}_K{k}": v for (s, k), v in sorted(solo.items())},
                 "eight_way_probes": eight}

    need = [("392m", 24), ("392m", 40), ("98m", 24), ("98m", 40)]
    absent = [f"{s}_K{k}" for s, k in need if (s, k) not in solo]
    if absent:
        rec["status"] = f"INCOMPLETE -- missing solo probes: {absent}"
        rec["note"] = ("A0.3 emits FOUR probes. R is a like-for-like phase0/phase0 ratio "
                       "(verify-R2 FATAL-1); it CANNOT be formed from three.")
        print(json.dumps(rec, indent=1))
        if args.out:
            open(args.out, "w").write(json.dumps(rec, indent=1))
        return 3

    # ---- cross-check ------------------------------------------------------
    fresh = solo[("98m", 24)]["s_per_step"]
    dev = abs(fresh - ARCHIVED_98M_K24_PHASE0) / ARCHIVED_98M_K24_PHASE0
    rec["crosscheck_98m_K24_vs_archive"] = dict(
        fresh=fresh, archived=ARCHIVED_98M_K24_PHASE0, rel_dev=round(dev, 5),
        tolerance=CROSSCHECK_TOL, within=bool(dev <= CROSSCHECK_TOL),
        verdict=("within +-10%" if dev <= CROSSCHECK_TOL else
                 "INSTRUMENT DRIFT -- reported before A0.5 is applied; does NOT block, because "
                 "R is a ratio of two FRESH probes"))
    # The standing instrument note (sec 4.0): the phase0-vs-realized inflation.
    rec["instrument_note_phase0_inflation"] = dict(
        archived_phase0_98m_K24=ARCHIVED_98M_K24_PHASE0,
        realized_98m_K24_s_per_step=S_PER_STEP_98M_REALIZED[24],
        inflation=round(ARCHIVED_98M_K24_PHASE0 / S_PER_STEP_98M_REALIZED[24], 4),
        mechanism=("run_phase0_timing wraps EACH ARM in torch.cuda.synchronize() and sums; the "
                   "real training loop never synchronizes per arm. At num_heads=1 and 128-286 "
                   "token sequences the workload is launch-bound, so two forced pipeline flushes "
                   "per step cost ~50%. The probe's timer also starts after "
                   "build_task1_document, while the wall-clock denominator includes data "
                   "generation, every eval pass, checkpoint writes and startup."),
        why_it_does_not_bite=("R is phase0/phase0, so this factor appears identically in "
                              "numerator and denominator and cancels EXACTLY."))

    # ---- Rule P1 ----------------------------------------------------------
    R = {k: solo[("392m", k)]["s_per_step"] / solo[("98m", k)]["s_per_step"] for k in (24, 40)}
    Rmax = max(R.values())
    if Rmax <= P1_NOMINAL:
        branch, action = "NOMINAL", "Proceed to Stage A at the re-priced ledger."
    elif Rmax <= P1_COSTOUT:
        branch, action = ("PROCEED WITH A RECORDED MISS",
                          "Above the elected 87-101 envelope, inside tier (c). Re-derive every "
                          "spec's --ceiling-gpuh from the CONTENDED rate and record the "
                          "projection miss in EXPERIMENT_LOG as an instrument note.")
    else:
        branch, action = ("COST-OUT",
                          "Do NOT queue the sweep. Run sec 4.4.1's publishable floor -- the K=24 "
                          "FROZEN TRIO (seeds 0,1,2) at the re-priced rate, as an explicitly "
                          "re-scoped tier-(a) SINGLE-POINT SCALE PROBE at n=3 (the wave-0 rule's "
                          "own minimum) -- then STOP. The three legs of sec 4.2 are read on it "
                          "and reported as a single-point result; NO cross-scale test is run "
                          "(one K is not a curve) and sec 5's per-K verdicts are struck for this "
                          "branch. This branch does NOT end with zero 392M data (verify-R2 "
                          "MAJOR-8's restored floor).")
    # the two ledger bases, LABELLED (verify-R2 m5: R1 mixed them)
    trained_only = sum(CELLS[k] * GPU_H_98M_MEASURED[k] for k in CELLS) * Rmax
    headline = (trained_only + 0.5 + 0.4 + 0.15) * 1.10
    floor_gpuh = 3 * GPU_H_98M_MEASURED[24] * Rmax
    rec["P1_solo_cost_out"] = dict(
        R_per_K={f"K{k}": round(v, 4) for k, v in R.items()}, R_max=round(Rmax, 4),
        definition="phase0(392M,K)/phase0(98M,K) -- LIKE-FOR-LIKE (verify-R2 FATAL-1's fix)",
        thresholds={"nominal": "<= 4.0", "recorded_miss": "4.0 < R <= 4.5", "cost_out": "> 4.5"},
        graft_overhead_G=round(Rmax / PLAIN_BACKBONE_RATIO, 4),
        plain_backbone_ratio=PLAIN_BACKBONE_RATIO,
        ledger_trained_only_gpuh=round(trained_only, 2),
        ledger_headline_gpuh=round(headline, 2),
        headline_basis="trained-only + A0 0.5 + Stage C 0.4 + 98M re-score 0.15, x1.10",
        section_4_4_1_floor_gpuh=round(floor_gpuh, 2),
        branch=branch, action=action)

    # ---- Rule P2 ----------------------------------------------------------
    if eight:
        m8 = sum(e["s_per_step"] for e in eight) / len(eight)
        R8 = m8 / solo[("392m", 24)]["s_per_step"]
        rec["P2_contention_halt"] = dict(
            n_probes=len(eight), mean_8way_s_per_step=round(m8, 6),
            solo_392m_K24_s_per_step=round(solo[("392m", 24)]["s_per_step"], 6),
            R8=round(R8, 4), threshold=P2_BAR,
            branch=("NOMINAL" if R8 <= P2_BAR else "DO NOT QUEUE WAVE 1"),
            action=("The ledger stands and --ceiling-gpuh derives from the 1.5 x R_8-priced "
                    "projection." if R8 <= P2_BAR else
                    "Re-price the WHOLE ledger at the observed contended rate and RE-ENTER Rule "
                    "P1 with that number. sec 10 R2: the unexplained 5.5x co-tenancy regression "
                    "would turn 84 GPU-h into ~460 and 10 h of wall into 2+ days; measuring R_8 "
                    "here means the halt is free, instead of ~140 GPU-h late."))
    else:
        rec["P2_contention_halt"] = {"status": "NOT MEASURED -- run `run_stage_a0.sh contended`"}

    # ---- Rule P3 ----------------------------------------------------------
    ratio = R[40] / R[24]
    rec["P3_per_K_price"] = dict(
        R40_over_R24=round(ratio, 4), threshold=P3_BAR,
        T_dependent=bool(ratio > P3_BAR),
        action=("Single-ratio ledger stands." if ratio <= P3_BAR else
                "The graft overhead is T-DEPENDENT: re-derive the ledger PER K from R(K); "
                "K=32 and K=16 are INTERPOLATED IN t_in and must be FLAGGED as interpolations."),
        re_priced_gpu_h={f"K{k}": round(GPU_H_98M_MEASURED[k] *
                                        (R[24] if k <= 24 else R[40]), 3) for k in CELLS},
        contended_ceiling_gpuh={f"K{k}": round(1.5 * 3.3 * GPU_H_98M_MEASURED[k] *
                                               (R[24] if k <= 24 else R[40]), 3) for k in CELLS})

    # ---- Rule P4 ----------------------------------------------------------
    if args.gates and os.path.exists(args.gates):
        g = json.load(open(args.gates))
        b8 = next((i["detail"] for i in g["items"]
                   if i["item"] == "B8_memory_and_utilisation" and i["status"] == "PASS"), None)
        if b8:
            peak = float(b8["P4_reading_gb"])
            util = b8.get("sm_util_median")
            rec["P4_memory_and_utilisation"] = dict(
                source="scaleaxis_gates.py B8 (ncr_lm_wave1_smoke.py:663/:796/:1056 + the "
                       "production two-arm WITH-EVAL peak + the nvidia-smi sampler)",
                peak_gb_with_eval=peak, limit_gb=P4_LIMIT_GB,
                design_projection_gb=[21, 28],
                branch=("NOMINAL" if peak < P4_LIMIT_GB else "PLACEMENT ASSUMPTION RE-OPENS"),
                sm_util_median=util, sm_util_max=b8.get("sm_util_max"),
                sm_util_verdict=("NOMINAL" if (util or 0) >= 50 else
                                 "SUSTAINED <50% IS A BUG -- diagnose before ANY cell queues"))
        else:
            rec["P4_memory_and_utilisation"] = {"status": "B8 block absent or FAILED in --gates"}
    else:
        rec["P4_memory_and_utilisation"] = {"status": "NOT EVALUATED -- pass --gates"}

    rec["overall"] = dict(
        P1=rec["P1_solo_cost_out"]["branch"],
        P2=rec["P2_contention_halt"].get("branch", "NOT MEASURED"),
        P3=("T-dependent" if rec["P3_per_K_price"]["T_dependent"] else "single-ratio"),
        P4=rec["P4_memory_and_utilisation"].get("branch", "NOT EVALUATED"),
        gate=("A0 CLEARS -- Stage A may be queued"
              if (rec["P1_solo_cost_out"]["branch"] != "COST-OUT"
                  and rec["P2_contention_halt"].get("branch") == "NOMINAL")
              else "A0 DOES NOT CLEAR -- see P1/P2 actions"))

    js = json.dumps(rec, indent=1)
    print(js)
    if args.out:
        open(args.out, "w").write(js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
