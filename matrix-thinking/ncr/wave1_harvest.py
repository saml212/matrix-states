"""NCR wave-1 harvest / endpoint scorer (NOVEL_ARCH_WATERFALL.md S3.2/S3.2a,
offline: reads cell JSONs + Axis-C lock files, no GPU, no torch).

Implements, for one K (wave 1 = K=8):

  AXIS A (S3.2a): per-arm band at h* from the MEDIAN recovered_frac@0.9
    across seeds of the arm's PRIMARY read (HOLD >= 0.9 / DEGRADED (0.5,
    0.9) / FAIL <= 0.5; boundary assignment: exactly 0.9 -> HOLD, exactly
    0.5 -> FAIL). Best-baseline = max of the LoopedVec/FWM medians (C_MLP
    never enters). 9-cell (NCR x best-baseline) label per the pinned table.
    Residue-strata guard: the label is awarded only if it holds on every
    claim-eligible residue stratum at the h*-adjacent sweep (the novel-
    residue sweep points, labels enforced label-and-exclude) -- an arm
    whose worst novel-stratum band is worse than its h* band drops one
    band before labeling (S3.2a). Identity/train-residue sweep points are
    EXCLUDED from bands and INCLUDED in the reducer signature.

  P1: >= 3/5 NCR seeds at recovered@0.9 >= 0.9 at h*; per-seed failure
    fronts reported against the pre-registered [87, 442] median-seed band;
    post-front revivals reported, never re-admitted (mi5).
  P2: every comparison-of-record arm's median recovered@0.9 < 0.5 by the
    last pinned ladder point before h* (h=29 at K=8).

  AXIS B: per-arm median read time vs h on the timed points; log2(h) and
    h regression fits (which scaling explains the claimed read vs the O(h)
    arms), and the NCR-binexp-vs-best-O(h) ratio at h=1021 (WIN >= 10x).

  AXIS C: per NCR seed, |locked-predicted - measured mean_cos| at every
    ladder h <= 509 (bar 0.05, WIN >= 3/5 seeds; TIE = holds through
    h <= 125 only). Predictions come ONLY from the cell's hash-verified
    Axis-C lock file (never recomputed post-hoc from the Z-dump -- the
    lock IS the pre-registration).

  CROSSCHECK-LENS DISCIPLINE (S2.31a precedent, per the S7d launch terms):
    the deep probe's scale-corrected residual is a DEGAUGED readout (the
    c* lens); its basis-invariant crosscheck -- the raw ||A - Pi||/||Pi||
    plus the phase-only eigenvalue residual (basis-invariant by
    construction) -- is reported alongside in the per-cell table, never
    the degauged number alone.

Also: reducer signatures, NUMERIC-DIVERGENT counts, trust-label tallies,
GPU-h ledger. Every reported h carries (h, h mod K); aggregates never pool
across residues.

Usage: python wave1_harvest.py --dir results_wave1 [--k 8] [--out FILE]

Pooled seed-extension harvests (NOVEL_ARCH_WATERFALL.md §7h/§7i): pass
--expect-seeds-ncr to admit a different ncr-arm seed count than
loopedvec/fwm (baselines are never re-run in an extension, so their
count stays at --expect-seeds while ncr grows). Omitting the flag
reproduces every prior invocation byte-for-byte (--selftest exercises
this as a regression, see below).

Self-test (offline, no fixtures needed): python wave1_harvest.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import ncr_spectral as ns  # verify_axis_c_lock (numpy import only)

H_STAR = {8: 61, 12: 57}
P2_POINT = {8: 29, 12: 45}
PRIMARY = {"ncr": "binexp", "fwm": "fwm_recursive",
           "loopedvec": "loopedvec_iter", "cmlp": "cmlp_onehot"}
COMPARISONS_OF_RECORD = ("loopedvec", "fwm")
AXIS_A_TABLE = {  # (ncr_band, best_baseline_band) -> outcome (S3.2a, 9 cells)
    ("HOLD", "FAIL"): "WIN", ("HOLD", "DEGRADED"): "SEP-PARTIAL", ("HOLD", "HOLD"): "TIE",
    ("DEGRADED", "FAIL"): "SEP-PARTIAL", ("DEGRADED", "DEGRADED"): "TIE",
    ("DEGRADED", "HOLD"): "LOSE",
    ("FAIL", "FAIL"): "TIE", ("FAIL", "DEGRADED"): "LOSE", ("FAIL", "HOLD"): "LOSE",
}
BAND_ORDER = ["FAIL", "DEGRADED", "HOLD"]


def band(x: float) -> str:
    """S3.2a numeric bands; boundaries assigned exactly as pinned."""
    if x >= 0.9:
        return "HOLD"
    if x > 0.5:
        return "DEGRADED"
    return "FAIL"


def drop_one_band(b: str) -> str:
    return BAND_ORDER[max(0, BAND_ORDER.index(b) - 1)]


def load_cells(d: str, K: int) -> dict:
    cells = {}
    for p in sorted(glob.glob(os.path.join(d, f"ncr_*_K{K}_s*.json"))):
        if ".axis_c_lock." in os.path.basename(p):
            continue                      # lock files are read by axis_c(), not cells
        r = json.load(open(p))
        if r.get("status") != "COMPLETED":
            print(f"  SKIPPING non-COMPLETED cell {os.path.basename(p)} "
                  f"(status={r.get('status')}) -- flagged, never silently pooled")
            continue
        arm = r["config"]["arm"]
        cells.setdefault(arm, {})[r["config"]["seed"]] = r
    return cells


def point(rec: dict, h: int, component=None) -> dict | None:
    for e in rec["eval"]["points"]:
        if e["h"] == h and (component is None or e["component"] == component):
            return e
    return None


def rec09(rec: dict, arm: str, h: int, component=None) -> float:
    e = point(rec, h, component)
    return e["reads"][PRIMARY[arm]]["recovered_frac@0.9"]


def median_over_seeds(cells: dict, arm: str, h: int, component=None) -> float:
    return statistics.median(rec09(r, arm, h, component)
                             for r in cells[arm].values())


def axis_a(cells: dict, K: int) -> dict:
    hs = H_STAR[K]
    out = {"h_star": hs, "arms": {}}
    for arm in ("ncr",) + COMPARISONS_OF_RECORD:
        med = median_over_seeds(cells, arm, hs)
        b = band(med)
        # residue-strata guard: novel-residue sweep points near h*
        strata = {}
        worst = b
        for r in cells[arm].values():
            for e in r["eval"]["points"]:
                if e["component"] == "residue_sweep" and e["residue_label"] == "novel":
                    strata.setdefault(e["h"], []).append(
                        e["reads"][PRIMARY[arm]]["recovered_frac@0.9"])
        strata_bands = {h: band(statistics.median(v)) for h, v in sorted(strata.items())}
        for h, sb in strata_bands.items():
            if BAND_ORDER.index(sb) < BAND_ORDER.index(worst):
                worst = sb
        b_final = drop_one_band(b) if worst != b else b
        out["arms"][arm] = dict(median_rec09_at_hstar=med, band_raw=b,
                                strata_bands={str(h): sb for h, sb in strata_bands.items()},
                                band_final=b_final,
                                strata_dropped_band=(b_final != b))
    best_base = max((out["arms"][a]["band_final"] for a in COMPARISONS_OF_RECORD),
                    key=BAND_ORDER.index)
    ncr_b = out["arms"]["ncr"]["band_final"]
    out["best_baseline_band"] = best_base
    out["axis_a_label_K"] = AXIS_A_TABLE[(ncr_b, best_base)]
    return out


def p1_p2(cells: dict, K: int) -> dict:
    hs = H_STAR[K]
    ncr_hold = [s for s, r in cells["ncr"].items() if rec09(r, "ncr", hs) >= 0.9]
    fronts = {s: r["eval"]["failure_front_h"] for s, r in cells["ncr"].items()}
    revivals = {s: r["eval"]["post_front_revivals"] for s, r in cells["ncr"].items()}
    p2 = {}
    for arm in COMPARISONS_OF_RECORD:
        med = median_over_seeds(cells, arm, P2_POINT[K])
        p2[arm] = dict(median_rec09=med, below_half=bool(med < 0.5))
    return dict(
        p1=dict(n_ncr_seeds_hold_at_hstar=len(ncr_hold), holding_seeds=ncr_hold,
                n_seeds=len(cells["ncr"]), pass_=bool(len(ncr_hold) >= 3),
                fronts=fronts, preregistered_front_band=[87, 442],
                revivals_reported_never_readmitted=revivals),
        p2={**p2, "pinned_point": P2_POINT[K],
            "pass_": bool(all(v["below_half"] for v in p2.values()))})


def axis_b(cells: dict) -> dict:
    # per arm: median over seeds of read_time_s at each timed h
    times = {}
    for arm, seeds in cells.items():
        for r in seeds.values():
            for e in r["eval"]["points"]:
                if "read_time_s" in e:
                    for name, t in e["read_time_s"].items():
                        times.setdefault(name, {}).setdefault(e["h"], []).append(t)
    med = {name: {h: statistics.median(v) for h, v in sorted(hs_.items())}
           for name, hs_ in times.items()}

    def fit_r2(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx == 0:
            return 0.0
        b1 = sxy / sxx
        ss_res = sum((y - (my + b1 * (x - mx))) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    fits = {}
    for name, curve in med.items():
        hs_ = sorted(curve)
        ys = [curve[h] for h in hs_]
        fits[name] = dict(r2_vs_log2h=fit_r2([math.log2(h) for h in hs_], ys),
                          r2_vs_h=fit_r2(hs_, ys))
    ratio = None
    if "binexp" in med and 1021 in med["binexp"]:
        oh_times = [med[n][1021] for n in ("loop", "fwm_recursive", "loopedvec_iter")
                    if n in med and 1021 in med[n]]
        if oh_times:
            ratio = min(oh_times) / med["binexp"][1021]
    label = ("WIN" if ratio is not None and ratio >= 10 else
             "TIE" if ratio is not None and ratio > 1 else
             "LOSE-diagnose-first" if ratio is not None else "n/a")
    return dict(median_read_time_s={n: {str(h): t for h, t in c.items()}
                                    for n, c in med.items()},
                scaling_fits=fits,
                ncr_vs_best_oh_ratio_at_1021=ratio, axis_b_label=label)


def axis_c(cells: dict, d: str, K: int) -> dict:
    """Locked-prediction vs measured mean_cos, ladder h <= 509, NCR seeds."""
    per_seed = {}
    for s, r in cells["ncr"].items():
        lock_path = os.path.join(d, f"{r['cell_id']}.axis_c_lock.json")
        lock = ns.verify_axis_c_lock(lock_path)   # hash-verified, the pre-registration
        assert lock["lock_sha256"] == r["axis_c_lock_sha256"], (
            f"lock hash mismatch vs cell record for seed {s}")
        devs = {}
        for e in r["eval"]["points"]:
            if e["component"] in ("ladder", "h_star") and e["h"] <= 509:
                pred = lock["mean_predicted_curve"][str(e["h"])]
                meas = e["reads"]["binexp"]["mean_cos"]
                devs[str(e["h"])] = dict(pred=pred, meas=meas, absdev=abs(pred - meas))
        max_dev = max(v["absdev"] for v in devs.values())
        max_dev_125 = max(v["absdev"] for h, v in devs.items() if int(h) <= 125)
        per_seed[s] = dict(per_h=devs, max_absdev_le509=max_dev,
                           max_absdev_le125=max_dev_125,
                           pass_509=bool(max_dev <= 0.05),
                           pass_125=bool(max_dev_125 <= 0.05))
    n509 = sum(1 for v in per_seed.values() if v["pass_509"])
    n125 = sum(1 for v in per_seed.values() if v["pass_125"])
    label = ("WIN" if n509 >= 3 else "TIE" if n125 >= 3 else "LOSE")
    return dict(per_seed={str(s): {k: v for k, v in d_.items() if k != "per_h"}
                          for s, d_ in per_seed.items()},
                per_seed_per_h={str(s): d_["per_h"] for s, d_ in per_seed.items()},
                n_seeds_pass_509=n509, n_seeds_pass_125=n125, axis_c_label=label)


def crosscheck_lens_table(cells: dict) -> dict:
    """S2.31a discipline: degauged (c*-corrected) residual never reported
    alone -- raw Frobenius residual + basis-invariant phase residual
    alongside, per cell."""
    out = {}
    for arm in ("ncr", "fwm"):
        for s, r in cells.get(arm, {}).items():
            if "deep_probe" not in r:
                continue
            dp = r["deep_probe"]
            out[f"{arm}_s{s}"] = dict(
                c_star=dp["c_star_per_example"],
                degauged_scale_corrected_residual=dp["scale_corrected_residual"],
                basis_invariant_phase_resid_max=dp["phase_resid_max_per_example"],
                eff_rank_A=dp["A_eff_rank"])
    return out


def hygiene(cells: dict) -> dict:
    n_div_shadow = n_div_agree = 0
    trust = {}
    reducers = {}
    gpu_h = 0.0
    for arm, seeds in cells.items():
        for s, r in seeds.items():
            gpu_h += r.get("gpu_h", 0.0)
            reducers[f"{arm}_s{s}"] = r["eval"]["reducer_signature"]["flagged"]
            for e in r["eval"]["points"]:
                for name, st in e["reads"].items():
                    if st.get("numeric_divergent_shadow"):
                        n_div_shadow += 1
                    lab = st.get("trust_label")
                    if lab:
                        trust[lab] = trust.get(lab, 0) + 1
                if e.get("numeric_divergent_agreement"):
                    n_div_agree += 1
    return dict(numeric_divergent_shadow_points=n_div_shadow,
                numeric_divergent_agreement_points=n_div_agree,
                trust_label_tally=trust,
                reducer_flags=reducers,
                any_reducer=any(reducers.values()),
                serial_sum_gpu_h=gpu_h)


def resolve_expected_seeds(expect_seeds: int, expect_seeds_ncr: int | None) -> dict:
    """Per-arm expected COMPLETED-cell count for the harvest's admission
    gate (S7h). ncr may be overridden independently of loopedvec/fwm --
    a seed-extension grows only the contender arm, never the settled
    comparisons-of-record. expect_seeds_ncr=None reproduces the original
    uniform-count behavior exactly (every pre-S7h invocation unaffected)."""
    ncr_n = expect_seeds_ncr if expect_seeds_ncr is not None else expect_seeds
    return {"ncr": ncr_n, "loopedvec": expect_seeds, "fwm": expect_seeds}


def check_seed_counts(cells: dict, expected: dict) -> None:
    """Refuse a partial or miscounted grid -- exact-threshold admission
    gate (CLAUDE.md: structural checks need exact thresholds, and the
    negative test proving it has teeth must actually be executed)."""
    for arm, exp_n in expected.items():
        n = len(cells.get(arm, {}))
        assert n == exp_n, (
            f"{arm}: {n}/{exp_n} COMPLETED cells -- harvest refuses "
            f"a partial grid (rerun the wave or pass "
            f"--expect-seeds/--expect-seeds-ncr explicitly)")


def _selftest() -> None:
    """Offline kill-proof teeth for resolve_expected_seeds/check_seed_counts
    (S7h harvest-code change). No fixtures, no GPU, no torch."""
    # 1. Old-style default (no --expect-seeds-ncr) is unchanged: uniform 5.
    assert resolve_expected_seeds(5, None) == {"ncr": 5, "loopedvec": 5, "fwm": 5}
    cells_5 = {"ncr": {i: {} for i in range(5)},
               "loopedvec": {i: {} for i in range(5)},
               "fwm": {i: {} for i in range(5)}}
    check_seed_counts(cells_5, resolve_expected_seeds(5, None))  # must not raise
    # 2. New pooled override: ncr=10, baselines stay 5.
    assert resolve_expected_seeds(5, 10) == {"ncr": 10, "loopedvec": 5, "fwm": 5}
    cells_10 = {"ncr": {i: {} for i in range(10)},
                "loopedvec": {i: {} for i in range(5)},
                "fwm": {i: {} for i in range(5)}}
    check_seed_counts(cells_10, resolve_expected_seeds(5, 10))  # must not raise
    # 3. KILL PROOFS (executed, not just written) -- each must raise AssertionError.
    killed = 0
    for bad_cells, bad_expected, tag in [
        (cells_10, resolve_expected_seeds(5, 5), "ncr under-count (10 present, 5 expected, no override)"),
        ({**cells_10, "ncr": {i: {} for i in range(9)}}, resolve_expected_seeds(5, 10), "ncr short by 1 (9/10)"),
        ({**cells_10, "loopedvec": {i: {} for i in range(4)}}, resolve_expected_seeds(5, 10), "loopedvec short by 1 (baselines never silently absorb an extension)"),
    ]:
        try:
            check_seed_counts(bad_cells, bad_expected)
        except AssertionError:
            killed += 1
        else:
            raise RuntimeError(f"KILL-PROOF FAILED TO FIRE: {tag}")
    assert killed == 3
    print(f"_selftest: 5/5 checks PASSED (2 positive, 3 kill-proofs executed and fired)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--k", type=int, default=8, choices=(8, 12))
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect-seeds", type=int, default=5,
                    help="trained-arm seed count the grid promises (teeth)")
    ap.add_argument("--expect-seeds-ncr", type=int, default=None,
                    help="override --expect-seeds for the ncr arm only "
                         "(pooled seed-extension harvests, e.g. K=12 5+5=10; "
                         "S7h). Omit to reproduce prior behavior exactly.")
    ap.add_argument("--selftest", action="store_true",
                    help="offline kill-proof teeth for the per-arm count gate, no --dir needed")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.dir:
        ap.error("--dir is required unless --selftest")

    cells = load_cells(args.dir, args.k)
    expected = resolve_expected_seeds(args.expect_seeds, args.expect_seeds_ncr)
    check_seed_counts(cells, expected)

    result = dict(K=args.k, dir=args.dir,
                  n_cells={a: len(s) for a, s in cells.items()})
    result["axis_a"] = axis_a(cells, args.k)
    result["predictions"] = p1_p2(cells, args.k)
    result["axis_b"] = axis_b(cells)
    result["axis_c"] = axis_c(cells, args.dir, args.k)
    result["crosscheck_lens"] = crosscheck_lens_table(cells)
    result["hygiene"] = hygiene(cells)

    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("crosscheck_lens",)}, indent=1, default=float))
    print("\nCROSSCHECK-LENS TABLE (degauged c*-residual NEVER alone; raw + "
          "basis-invariant phase residual alongside):")
    print(json.dumps(result["crosscheck_lens"], indent=1, default=float))
    if args.out:
        with open(args.out + ".tmp", "w") as f:
            json.dump(result, f, indent=1, default=float)
        os.replace(args.out + ".tmp", args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
