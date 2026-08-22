#!/usr/bin/env python3
"""STAGE A0.5 -- Rules P1, P2, P3, P4, EVALUATED.
NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.4 (as corrected by verify-R2 FATAL-1
for P1's instrument and verify-R2 MAJOR-1 for P4's), sec 4.4.1 and sec 3.6.

Every number here is MEASURED at Stage A0, before any training cell exists.
No rule reads a number its elected instrument does not produce -- that is the
defect class (FATAL-3 and its two repeats) this file exists not to commit.

  P1  R := phase0(392M,K).mean_s_per_step_both_arms_combined
         / phase0( 98M,K).mean_s_per_step_both_arms_combined,   max over K
      LIKE-FOR-LIKE, and on a STRONGER receipt than the design argued
      (AUDIT-R1 MAJOR-1 / C1): run_phase0_timing is BYTE-IDENTICAL between the
      98M and 392M runners, so R's probe bias cancels BY CONSTRUCTION whatever
      its absolute level. The design's "1.5500x inflation" narrative is
      FALSIFIED -- measured beta = phase0/realized is 0.8293 (K=24) and 0.8657
      (K=40), i.e. the probe UNDER-reads. See STALE_PIN_NOTE and BETA_98M.
      R is an estimate of rho_realized CONDITIONAL on beta_392 == beta_98, an
      assumption this design does not measure; the post-hoc reconciliation on
      the first calibration cell (>15% => re-enter P1) is what tests it.
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
      ledger PER K from R(K). AUDIT-R1 m5: what is implemented is a STEP
      assignment (R(24) for K<=24, R(40) for K>=32), not a t_in interpolation;
      both assignments OVER-price, so the ledger is safe by ~3.6 GPU-h and the
      LABEL is what was wrong. The contended ceiling uses the MEASURED R_8
      (sec 3.6's primary rule) and falls back to CONTENDED_MULTIPLIER = 3.3
      ONLY when R_8 was not measured -- labelled either way (MAJOR-3 / C3).

  P4  memory, read as the MAX OVER THE FOUR K (AUDIT-R1 m4 / C5) from
      scaleaxis_gates.py's B8 block -- the ONLY instrument that emits one
      (run_phase0_timing records NO memory or utilisation field of any kind,
      which is why R1's "peak VRAM at A0.3" was unmeasurable). Design sec 4.4's
      :663/:796 identification is INVERTED (MAJOR-4 / C7); P4 reads the build's
      own production-shaped two-arm train + eval peak, which is strictly more
      conservative than either named line:
        < 40 GB   nominal;  >= 40 GB  not a blocker, but sec 8.3's placement
                  assumption RE-OPENS and must be adjudicated before Stage B.
      SM utilisation likewise comes from B8's external nvidia-smi sampler;
      sustained < 50% is treated as a BUG and diagnosed before any cell queues.

  CROSS-CHECK: the design pins a +-10% comparison of the fresh 98M K=24 probe
  against the ARCHIVED 0.23075456221898397
  (experiment-runs/2026-07-17_ncr_gate3_wave1/phase0_timing.json).
  THE ARCHIVED PIN IS **STALE** -- AUDIT-R1 MAJOR-1 / condition C1. It is kept
  as a REPORTED-ONLY diagnostic against a stale baseline and never blocks; see
  STALE_PIN_NOTE below for the measurement, the receipt and the reason.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# ---------------------------------------------------------------------------
# AUDIT-R1 MAJOR-1 / condition C1 -- THE ARCHIVED PIN IS STALE, AND THE DESIGN'S
# "1.5500x standing instrument note" IS FALSIFIED BY MEASUREMENT.
# ---------------------------------------------------------------------------
ARCHIVED_98M_K24_PHASE0 = 0.23075456221898397   # sec 4.0's pin -- STALE, see below
AUDIT_FRESH_98M_PHASE0 = {24: 0.123463, 40: 0.176222}   # audit-R1, 2026-08-22, same box
STALE_PIN_STATUS = "STALE -- superseded by the 2026-08-22 audit-R1 probes"
CROSSCHECK_TOL = 0.10
CROSSCHECK_IS_BLOCKING = False       # reported-only; it was never a blocker, and now the
                                     # baseline it compares against is known stale

# beta(K) := phase0(98M,K) / realized(98M,K). THIS TABLE REPLACES the design's
# 1.5500x note (sec 4.0 and the former instrument_note_phase0_inflation block).
BETA_98M = {24: 0.8293, 40: 0.8657}

STALE_PIN_NOTE = {
    "condition": "AUDIT-R1 MAJOR-1, condition C1",
    "what_the_design_said": ("sec 4.0 pinned 0.23075 as a +-10% cross-check AND recorded "
                             "1.5500x as 'a standing instrument note for every future wave "
                             "that prices from phase0-timing', attributing it to the per-arm "
                             "torch.cuda.synchronize() in run_phase0_timing."),
    "what_was_measured": ("Fresh 98M phase0-timing on the SAME box (brev-ukptqsu65), SAME "
                          "torch (2.12.1+cu130), same K/batch/doc_len/warmup/probe: "
                          "K=24 -> 0.123463, K=40 -> 0.176222. Deviation from the pin at "
                          "K=24 is -46.50%, i.e. 4.65x the pinned +-10% tolerance."),
    "the_1_5500_note_is_FALSIFIED": ("The same code today reads 0.8293x realized at K=24 and "
                                     "0.8657x at K=40. The SIGN IS INVERTED: the probe "
                                     "UNDER-reads realized, it does not inflate by 1.55x. The "
                                     "1.5500 figure was a property of the 2026-07-17 box "
                                     "state, not of the instrument, and must NOT be carried "
                                     "to any future wave."),
    "instrument_ruled_out_as_the_cause": ("run_phase0_timing's timed region is functionally "
                                          "identical between the archived gate3 runner and the "
                                          "kscaling runner -- the sole difference is one added "
                                          "output field OUTSIDE the timed block, and the "
                                          "per-arm synchronize is present in BOTH. No code "
                                          "change explains a 1.87x gap."),
    "therefore": ("A 46.5% cross-check failure at A0.3 is a STALE-BASELINE ARTIFACT, NOT A "
                  "LIVE FAULT. It is recorded here so a coordinator reading it immediately "
                  "before committing ~99 GPU-h does not mistake it for one."),
    "receipt_that_R_is_unaffected": ("run_phase0_timing is BYTE-IDENTICAL between the 98M "
                                     "(kscaling) and 392M (scaleaxis) runners -- audit-R1 "
                                     "diffed the extracted function, len 5880 == 5880, a == b "
                                     "True. R = phase0(392M)/phase0(98M) therefore cancels its "
                                     "probe bias BY CONSTRUCTION, whatever that bias's "
                                     "absolute level. This is a STRONGER receipt than the "
                                     "design's 'appears identically in numerator and "
                                     "denominator' argument, and it survives MAJOR-1 intact."),
    "beta_98m_measured": BETA_98M,
}
GPU_H_98M_MEASURED = {16: 0.8019, 24: 0.8271, 32: 0.9583, 40: 1.1309}   # sec 8.2
S_PER_STEP_98M_REALIZED = {24: 0.14888, 40: 0.20357}                    # sec 8.2
PLAIN_BACKBONE_RATIO = 3.50        # sec 4.4's five-measurement central estimate
P1_NOMINAL, P1_COSTOUT = 4.0, 4.5
P2_BAR = 1.25
P3_BAR = 1.15
P4_LIMIT_GB = 40.0
CEILING_MULT = 1.5        # sec 3.6 breaker 1
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


def _gate_verdict(rec: dict) -> str:
    """AUDIT-R1 m3 / condition C5: the SM-utilisation bug check now enters the
    gate. It previously computed from P1 and P2 only, so a sustained <50%
    reading could set sm_util_verdict = "SUSTAINED <50% IS A BUG -- diagnose
    before ANY cell queues" while `gate` still read "A0 CLEARS" -- two
    contradictory statements in one record, on the launch-decision surface.
    P4's MEMORY leg stays correctly excluded: sec 4.4 says >=40 GB is explicitly
    NOT a blocker, only a re-opening of sec 8.3's placement assumption."""
    fails = []
    if rec["P1_solo_cost_out"]["branch"] == "COST-OUT":
        fails.append("P1 COST-OUT")
    if rec["P2_contention_halt"].get("branch") != "NOMINAL":
        fails.append(f"P2 {rec['P2_contention_halt'].get('branch', 'NOT MEASURED')}")
    u = rec["P4_memory_and_utilisation"].get("sm_util_verdict")
    if u is not None and u != "NOMINAL" and not str(u).startswith("NOT EVALUATED"):
        fails.append("SM-UTIL BUG (<50% sustained)")
    if fails:
        return "A0 DOES NOT CLEAR -- " + "; ".join(fails)
    return "A0 CLEARS -- Stage A may be queued"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-dir", required=True)
    ap.add_argument("--gates", default=None, nargs="+",
                    help="scaleaxis_gates.py outputs, for P4. AUDIT-R1 m4 / C5: pass ALL FOUR K "
                         "-- memory grows with t_in (17.09 GB at K=16 up to 23.46 GB at K=40) "
                         "and the gate must read the MAX over K, not one K.")
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
    rec["crosscheck_98m_K24_vs_STALE_archive"] = dict(
        fresh=fresh, archived=ARCHIVED_98M_K24_PHASE0,
        archived_pin_status=STALE_PIN_STATUS,
        audit_fresh_reference=AUDIT_FRESH_98M_PHASE0,
        rel_dev_vs_stale_pin=round(dev, 5), tolerance=CROSSCHECK_TOL,
        within=bool(dev <= CROSSCHECK_TOL), blocking=CROSSCHECK_IS_BLOCKING,
        verdict=("within +-10% of the (stale) pin" if dev <= CROSSCHECK_TOL else
                 "DEVIATION FROM A STALE BASELINE -- NOT A LIVE FAULT. Reported only; it does "
                 "not block, both because R is a ratio of two FRESH probes AND because the "
                 "0.23075 baseline is itself superseded (AUDIT-R1 MAJOR-1 / C1)."),
        dev_vs_audit_fresh=(round(abs(fresh - AUDIT_FRESH_98M_PHASE0[24])
                                  / AUDIT_FRESH_98M_PHASE0[24], 5)))
    # C1: the 1.5500x note is REPLACED by the measured beta table. See MAJOR-2/C2.
    rec["instrument_note_phase0_vs_realized_BETA"] = dict(STALE_PIN_NOTE)
    rec["instrument_note_phase0_vs_realized_BETA"]["beta_measured_this_run"] = {
        f"K{k}": round(solo[("98m", k)]["s_per_step"] / S_PER_STEP_98M_REALIZED[k], 4)
        for k in (24, 40)}

    # ---- MAJOR-2 / C2: R is CONDITIONAL on beta_392 == beta_98 ------------
    beta_here = {k: solo[("98m", k)]["s_per_step"] / S_PER_STEP_98M_REALIZED[k]
                 for k in (24, 40)}
    rec["P1_conditionality_beta"] = dict(
        definition="beta(K) := phase0(98M,K) / realized(98M,K)",
        beta_98_audit_reference=BETA_98M,
        beta_98_this_run={f"K{k}": round(v, 4) for k, v in beta_here.items()},
        beta_rises_with_t_in=round(beta_here[40] / beta_here[24], 4),
        finding=("beta is NOT operating-point-invariant: it rises ~4.4% across a 1.64x t_in "
                 "increase (0.8293 -> 0.8657), consistent with a launch-bound workload where "
                 "more compute per kernel shrinks the relative synchronize penalty. The probe "
                 "also exaggerates K-dependence (fresh K40/K24 = 1.4273 vs realized 1.3673)."),
        what_R_actually_is=("R = rho_realized x (beta_392 / beta_98). This design ASSUMES that "
                            "ratio is 1, WITH NO MEASUREMENT. 98M->392M is a 4x increase in "
                            "per-kernel work -- a far larger move along the SAME axis that "
                            "already moved beta by 4.4%."),
        direction_of_risk=("UNSAFE for the cost-out gate: if beta_392 > beta_98 then R "
                           "UNDER-reads rho_realized and biases Rule P1 toward NOMINAL. A true "
                           "rho ~ 4.8 (COST-OUT) could read R ~ 4.0 (NOMINAL)."),
        post_hoc_reconciliation_RULE=(
            "MANDATORY, and free -- the first calibration cell produces the number anyway. "
            "When the FIRST calibration cell (K=24) completes, compute "
            "realized(392M,24) = gpu_h*3600/steps and form rho_realized(24) = "
            "realized(392M,24)/0.14888. If |rho_realized(24) - R(24)| / R(24) > 0.15, "
            "RE-ENTER RULE P1 ON THE REALIZED RATIO **BEFORE STAGE B QUEUES**. This converts "
            "an untested assumption into a measured one at zero cost."),
        reconciliation_threshold=0.15,
        reconciliation_status="PENDING -- no calibration cell has completed")

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
    R8_measured = None
    if eight:
        m8 = sum(e["s_per_step"] for e in eight) / len(eight)
        R8 = R8_measured = m8 / solo[("392m", 24)]["s_per_step"]
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
    # AUDIT-R1 MAJOR-3 / condition C3. sec 3.6 breaker 1 pins
    #   ceiling = 1.5 x (per-cell projection at the MEASURED CONTENDED rate R_8),
    # with 3.795 x solo as the fallback ONLY "if R_8 cannot be measured". This
    # file previously substituted the runner's PROJECTION CONSTANT
    # CONTENDED_MULTIPLIER = 3.3 for R_8 unconditionally -- IN THE SAME FILE THAT
    # MEASURES R_8 TWENTY LINES EARLIER -- shipping a backstop ~2.9x LOOSER than
    # pinned (and 32% looser than the spec's own PROJECTED placeholder, so the
    # A0.5 "re-price" WEAKENED the breaker instead of tightening it). Direction is
    # loose-not-tight, so MAJOR-1(b)'s "fires on every cell" failure is not
    # reintroduced and B6 remains the fast breaker -- but a pathological cell
    # could burn ~4.95x its projection before the backstop fired.
    ratio = R[40] / R[24]
    if R8_measured is not None:
        ceil_mult, ceil_basis = R8_measured, f"MEASURED R_8 = {R8_measured:.4f} (sec 3.6's PRIMARY rule)"
    else:
        ceil_mult, ceil_basis = 3.3, ("FALLBACK: the runner's CONTENDED_MULTIPLIER = 3.3, used "
                                      "ONLY because R_8 was not measured (run "
                                      "`run_stage_a0.sh contended`). sec 3.6 permits this path "
                                      "only when R_8 cannot be measured.")
    rec["P3_per_K_price"] = dict(
        R40_over_R24=round(ratio, 4), threshold=P3_BAR,
        T_dependent=bool(ratio > P3_BAR),
        action=("Single-ratio ledger stands." if ratio <= P3_BAR else
                "The graft overhead is T-DEPENDENT: re-derive the ledger PER K from R(K)."),
        re_priced_gpu_h={f"K{k}": round(GPU_H_98M_MEASURED[k] *
                                        (R[24] if k <= 24 else R[40]), 3) for k in CELLS},
        re_priced_basis=("AUDIT-R1 m5: this is a STEP assignment (R(24) for K<=24, R(40) for "
                         "K>=32), NOT an interpolation. Both assignments OVER-price relative to "
                         "true t_in interpolation, so the ledger is safe by ~3.6 GPU-h; the "
                         "label is corrected here rather than the arithmetic."),
        contended_ceiling_basis=ceil_basis,
        contended_ceiling_multiplier=round(ceil_mult, 4),
        contended_ceiling_gpuh={f"K{k}": round(CEILING_MULT * ceil_mult * GPU_H_98M_MEASURED[k] *
                                               (R[24] if k <= 24 else R[40]), 3) for k in CELLS})

    # ---- Rule P4 ----------------------------------------------------------
    gate_files = [g for g in (args.gates or []) if os.path.exists(g)]
    if gate_files:
        per_k, missing = {}, []
        for gf in sorted(gate_files):
            g = json.load(open(gf))
            b8 = next((i["detail"] for i in g["items"]
                       if i["item"] == "B8_memory_and_utilisation" and i["status"] == "PASS"),
                      None)
            if b8 is None:
                missing.append(os.path.basename(gf))
                continue
            per_k[f"K{g['K']}"] = dict(peak_gb=float(b8["P4_reading_gb"]),
                                       sm_util_median=b8.get("sm_util_median"),
                                       sm_util_max=b8.get("sm_util_max"),
                                       source=os.path.basename(gf))
        if not per_k:
            rec["P4_memory_and_utilisation"] = {"status": f"B8 block absent/FAILED in {missing}"}
        else:
            peak_k = max(per_k, key=lambda k: per_k[k]["peak_gb"])
            peak = per_k[peak_k]["peak_gb"]
            utils = [v["sm_util_median"] for v in per_k.values() if v["sm_util_median"] is not None]
            util_min = min(utils) if utils else None
            rec["P4_memory_and_utilisation"] = dict(
                source=("scaleaxis_gates.py B8 -- the build's OWN production-shaped two-arm "
                        "train + eval_both_arms peak. AUDIT-R1 MAJOR-4/C7: design sec 4.4's "
                        ":663/:796 identification is INVERTED (:663 is smoke_3's backbone-only "
                        "no_grad EVAL batch; :796 is smoke_7's full-graft TRAINING step), so P4 "
                        "is pointed at the instrument the build actually built, which is "
                        "strictly more conservative than either named line."),
                per_K=per_k, n_K_read=len(per_k), gate_files_missing_B8=missing,
                peak_gb_with_eval=round(peak, 3), peak_at_K=peak_k,
                reads_max_over_K=True, limit_gb=P4_LIMIT_GB,
                design_projection_gb=[21, 28],
                projection_note=("measured below sec 8.1's projected 21-28 GB band at three of "
                                 "four K -- a conservative projection miss, recorded"),
                branch=("NOMINAL" if peak < P4_LIMIT_GB else "PLACEMENT ASSUMPTION RE-OPENS"),
                sm_util_median_min_over_K=util_min,
                sm_util_verdict=("NOMINAL" if (util_min or 0) >= 50 else
                                 "SUSTAINED <50% IS A BUG -- diagnose before ANY cell queues"))
    else:
        rec["P4_memory_and_utilisation"] = {"status": "NOT EVALUATED -- pass --gates"}

    rec["overall"] = dict(
        P1=rec["P1_solo_cost_out"]["branch"],
        P2=rec["P2_contention_halt"].get("branch", "NOT MEASURED"),
        P3=("T-dependent" if rec["P3_per_K_price"]["T_dependent"] else "single-ratio"),
        P4=rec["P4_memory_and_utilisation"].get("branch", "NOT EVALUATED"),
        P4_sm_util=rec["P4_memory_and_utilisation"].get("sm_util_verdict", "NOT EVALUATED"),
        gate=_gate_verdict(rec))

    js = json.dumps(rec, indent=1)
    print(js)
    if args.out:
        open(args.out, "w").write(js)
    return 0


if __name__ == "__main__":
    sys.exit(main())
