"""h2h_calibration_bands_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.7 gate 1:
the calibration-band CONFIG that h2h_calibration_wrappers_rd.py's
check_calibration_band was built (parameterized, deliberately unpopulated)
to consume, populated at the DEPLOY stage from archived, verified numbers.

BAND PROVENANCE, stated per gate 1's own instruction ("checked against the
val-loss/recovered_frac band this task's own small-scale precedent
predicts"):

  TASK 3 (real-data LM) -- PRECEDENT-ANCHORED. The ONLY archived rung-1
  numbers for this exact config family (14M, d_model=256/d_state=64/
  n_layers=2, 20,000 steps, batch 32 x seq 512) are the FROZEN_BIAS_LM
  rung-1 wave's own:
    - Arm 2 (per_token lam=0.58 == THIS design's contender, sec 1.2 pin)
      final val loss, mean over seeds: 2.1184 (openr1-mix-ext) / 4.3426
      (wikitext-mix-ext)  [FROZEN_BIAS_LM_DESIGN.md sec 12.1 verified-
      numbers table; also quoted in HEAD_TO_HEAD sec 1.2 item 3].
    - sec 7.2 val-loss capability-gate ceilings (derived from Arm 1's own
      seed spread, mechanically pinned): 2.1935 (openr1) / 4.3828
      (wikitext)  [same table / FROZEN_BIAS sec 7.2].
  The ablation/transformer ARMS are new architectures -- their own numbers
  are first measurements -- so their task-3 bands are ANCHORED to the
  precedent numbers above with disclosed slack, not derived from any
  arm-specific precedent:
    - ablation FULL cell (20k steps, openr1-mix-ext): [1.90, 2.60].
      hi = 2.1935 ceiling + ~0.40 divergence slack (a param-matched 14M
      arch landing >0.4 nats above the contender family signals a broken/
      diverged cell, not an interesting result); lo = 1.90 (no 14M model
      in this program has ever approached it on this corpus -- below it
      signals an eval bug/leakage, not skill).
    - transformer QUARTER cells (5k steps, LR grid): [1.90, 5.50],
      SANITY-ONLY (no 5k-step archived point exists for ANY arch; init
      loss ~= ln(50257) ~= 10.82 per lm_pretrain_rd's AUDIT FIX-2 note,
      so 5.50 only excludes not-actually-training cells) + the
      must-decrease check below.

  TASKS 1/2 (grammar instruments through the sec 1.3.1 probe head) --
  *** FIRST-MEASUREMENT TERRITORY, SANITY-ONLY BANDS ***. No precedent
  exists for ANY of the three arms on these instruments (M1's own text:
  model_rd.py's cliff numbers "carry ZERO evidentiary weight for these
  three arms"; Task D/E precedents are model_rd-architecture results and
  set only the INSTRUMENT's validity, not these arms' expected values).
  Bands are therefore deliberately WIDE and check only that the cell is a
  functioning measurement:
    - joint train loss DECREASED from its first-step value;
    - recovered_frac@0.9 (train hops) in [0,1] always, and > 0 by the end
      of every FULL-budget cell (a probe that never recovers anything
      after 20k joint-training steps is a broken instrument, sec 1.3.1.3);
    - the K/d=0.75 STRESS cells (quarter-budget, locate-only per M-NEW-3)
      and the transformer's capped-M2 readings carry NO recovered_frac
      floor (an above-cliff or hard-capped reading near 0 is a legitimate
      locate-only result, not an instrument failure).
  FLAG (carried into CALIBRATION_COMPLETE): tasks 1/2 bands are
  sanity-only, NOT precedent-derived; the axis-1 margin freeze must treat
  the calibration cells themselves as the first real numbers.

Run: python h2h_calibration_bands_rd.py --check-dir <results/h2h_rung1/calib>
     python h2h_calibration_bands_rd.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_calibration_wrappers_rd import check_calibration_band

# Each band row: which results it applies to (match on task-prefix/arch/role),
# which result-JSON metric it reads, the [lo, hi] band, its kind, its source.
# "decrease" rows check metric_final < metric_init instead of a fixed band.
BANDS: tuple[dict, ...] = (
    # ---- task 3, ablation full cell: PRECEDENT-ANCHORED ----
    {"task": "task3", "arch": "ablation", "role": "primary",
     "metric": "final_val_loss_own", "lo": 1.90, "hi": 2.60, "kind": "anchored",
     "source": "FROZEN_BIAS_LM_DESIGN.md sec 12.1 (Arm2 2.1184 openr1) + sec 7.2 ceiling 2.1935 "
               "+ disclosed 0.40 divergence slack; lo=1.90 eval-bug floor"},
    # ---- task 3, transformer quarter cells (LR grid): SANITY-ONLY ----
    {"task": "task3", "arch": "transformer", "role": "lr_grid",
     "metric": "final_val_loss_own", "lo": 1.90, "hi": 5.50, "kind": "sanity",
     "source": "FIRST MEASUREMENT (no 5k-step precedent, any arch); hi excludes "
               "not-training cells (init ~= ln(50257) ~= 10.82), lo = eval-bug floor"},
    {"task": "task3", "arch": "*", "role": "*",
     "metric": "__val_decreased__", "lo": None, "hi": None, "kind": "sanity",
     "source": "loss must decrease from init (universal sanity)"},
    # ---- tasks 1/2: SANITY-ONLY (first measurement, flagged) ----
    {"task": "task1", "arch": "*", "role": "*",
     "metric": "__train_loss_decreased__", "lo": None, "hi": None, "kind": "sanity",
     "source": "FIRST MEASUREMENT; joint loss must decrease from step-1 value"},
    {"task": "task2", "arch": "*", "role": "*",
     "metric": "__train_loss_decreased__", "lo": None, "hi": None, "kind": "sanity",
     "source": "FIRST MEASUREMENT; joint loss must decrease from step-1 value"},
    {"task": "task1", "arch": "*", "role": "*",
     "metric": "final_recovered_frac_train_hops", "lo": 0.0, "hi": 1.0, "kind": "sanity",
     "source": "recovered_frac is a fraction (instrument-validity range check)"},
    {"task": "task2", "arch": "*", "role": "*",
     "metric": "final_recovered_frac_train_hops", "lo": 0.0, "hi": 1.0, "kind": "sanity",
     "source": "recovered_frac is a fraction (instrument-validity range check)"},
    # full-budget grammar cells only: the probe must recover SOMETHING by the end
    {"task": "task1", "arch": "*", "role": "primary",
     "metric": "final_recovered_frac_train_hops", "lo": 1e-9, "hi": 1.0, "kind": "sanity",
     "source": "FIRST MEASUREMENT; >0 after 20k joint steps or the instrument is broken "
               "(sec 1.3.1.3); stress/quarter cells deliberately exempt (M-NEW-3 locate-only)"},
    {"task": "task2", "arch": "*", "role": "primary",
     "metric": "final_recovered_frac_train_hops", "lo": 1e-9, "hi": 1.0, "kind": "sanity",
     "source": "same as task1 primary; held-out-hop frac is the SCIENCE, deliberately unbanded"},
)


def _matches(band: dict, result: dict) -> bool:
    if not result.get("task", "").startswith(band["task"]):
        return False
    if band["arch"] not in ("*", result.get("arch")):
        return False
    role = result.get("role") or ""
    if band["role"] == "*":
        return True
    return role.startswith(band["role"])


def check_result(result: dict) -> list[dict]:
    """Evaluates every applicable band; every returned row carries
    within_band. Gate 1: ANY failed band (sanity included) is a hard abort
    of the 27-cell sweep."""
    rows = []
    for band in BANDS:
        if not _matches(band, result):
            continue
        if band["metric"] == "__val_decreased__":
            init, fin = result.get("init_val_loss_own"), result.get("final_val_loss_own")
            ok = init is not None and fin is not None and fin < init
            rows.append({"metric": "val_loss decreased (init->final)", "measured": (init, fin),
                         "band": None, "within_band": ok, "kind": band["kind"],
                         "source": band["source"]})
        elif band["metric"] == "__train_loss_decreased__":
            first, fin = result.get("loss_first"), result.get("loss_final_mean5")
            ok = first is not None and fin is not None and fin < first
            rows.append({"metric": "joint train loss decreased (step1->final)",
                         "measured": (first, fin), "band": None, "within_band": ok,
                         "kind": band["kind"], "source": band["source"]})
        else:
            measured = result.get(band["metric"])
            if measured is None:
                rows.append({"metric": band["metric"], "measured": None, "band": (band["lo"], band["hi"]),
                             "within_band": False, "kind": band["kind"],
                             "source": band["source"] + " [METRIC MISSING FROM RESULT]"})
                continue
            r = check_calibration_band(measured, band["lo"], band["hi"], band["metric"])
            r.update({"kind": band["kind"], "source": band["source"]})
            rows.append(r)
    return rows


def check_dir(path: str) -> int:
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    assert files, f"no calibration result JSONs found under {path}"
    n_fail = 0
    for fp in files:
        with open(fp) as f:
            result = json.load(f)
        rows = check_result(result)
        for r in rows:
            ok = r["within_band"]
            n_fail += 0 if ok else 1
            print(f"[{os.path.basename(fp)}] {r['metric']}: measured={r['measured']} "
                  f"band={r['band']} kind={r['kind']} -> {'PASS' if ok else 'FAIL'}")
    if n_fail:
        print(f"BAND CHECK: {n_fail} FAILURE(S) -- HARD ABORT of the 27-cell sweep "
              f"(sec 1.7 gate 1: never launched silently)", file=sys.stderr)
        return 1
    print("BAND CHECK: every measured band within range")
    return 0


def selftest() -> int:
    failures = []

    def rep(item, ok):
        print(f"[{item}] {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            failures.append(item)

    good_t3 = {"task": "task3_calib", "arch": "ablation", "role": "primary",
               "final_val_loss_own": 2.30, "init_val_loss_own": 10.8}
    bad_t3 = dict(good_t3, final_val_loss_own=3.10)
    rep("teeth 1: in-band task3 ablation passes all its bands",
        all(r["within_band"] for r in check_result(good_t3)))
    rep("teeth 2: diverged task3 ablation (3.10 > hi 2.60) FAILS",
        any(not r["within_band"] for r in check_result(bad_t3)))

    good_t1 = {"task": "task1_calib", "arch": "contender", "role": "primary",
               "loss_first": 9.0, "loss_final_mean5": 2.0,
               "final_recovered_frac_train_hops": 0.4}
    dead_probe = dict(good_t1, final_recovered_frac_train_hops=0.0)
    stress_zero = {"task": "task1_calib", "arch": "contender", "role": "stress_locate_only",
                   "loss_first": 9.0, "loss_final_mean5": 2.0,
                   "final_recovered_frac_train_hops": 0.0}
    no_learn = dict(good_t1, loss_final_mean5=9.5)
    rep("teeth 3: healthy task1 primary passes", all(r["within_band"] for r in check_result(good_t1)))
    rep("teeth 4: dead probe (rf=0.0 after full budget) FAILS the >0 sanity band",
        any(not r["within_band"] for r in check_result(dead_probe)))
    rep("teeth 5: STRESS cell at rf=0.0 is EXEMPT from the >0 floor (locate-only, M-NEW-3)",
        all(r["within_band"] for r in check_result(stress_zero)))
    rep("teeth 6: non-decreasing joint loss FAILS", any(not r["within_band"] for r in check_result(no_learn)))
    rep("teeth 7: missing metric reads as FAIL, never silently passes",
        any(not r["within_band"] for r in check_result({"task": "task3_calib", "arch": "ablation",
                                                        "role": "primary"})))

    print("=" * 70)
    if failures:
        print(f"SELFTEST: {len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print("SELFTEST: ALL ITEMS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-dir", type=str)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="alias for --selftest (house convention)")
    args = ap.parse_args()
    if args.selftest or args.smoke:
        return selftest()
    if args.check_dir:
        return check_dir(args.check_dir)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
