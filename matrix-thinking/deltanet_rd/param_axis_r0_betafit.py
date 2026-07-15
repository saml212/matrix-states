#!/usr/bin/env python3
"""R0 BLIND beta-FIT -- the analysis layer. Reads the QUARANTINED per-(rung,corpus)
DiD cells produced by param_axis_r0_v2_driver.py and computes the R0 verdict.

RUN BY A FRESH BLIND AGENT (Phase 4). It is written here (not run) by the build
session.

ORDER IS LOAD-BEARING (record-first, CLAUDE.md / DELTA_D3_BLIND_REPIN.md §6 /
DSTATE_CONFOUND_PREREG.md §5): the two beta-INVARIANT gates are emitted ABOVE the
slope --
  * GATE-A  (attribution)  -- |{d_state(r) : r in A}| == 1 ?   (structural)
  * GATE-1  (power)        -- FLAT available iff  δ >= 2.49·SE(β̂) ?  (β̂-invariant;
                              strikes FLAT from the map at n_seeds=1 BEFORE β̂ is read)
Only after both are printed/recorded does the script compute β̂ / Δ₆₄ / γ̂. The
numeric slopes are written to files (r0_verdict.json + R0_VERDICT.md); they are
NOT splattered to stdout unless `--emit-slope-to-stdout` is passed, so a casual
reader / the coordinator stays blind to the outcome before the record step.

The estimand (frozen instrument): M(r) = DiD(r), the RAW (identity-normalized,
§9.1) paired accuracy difference `hit_placebo_ablated - hit_true_ablated`,
per-corpus in [-1,1]. §11.7-D2: a rung's pooled M(r) is the mean of the pooled
PER-ROW DiD records across BOTH corpora; the clustered bootstrap resamples ROWS
within corpus (row = cluster, corpus = stratum). The cells carry `per_row_did`
(the fit layer's designed interface; probe `_per_row_stat`'s docstring).
"""
import argparse
import glob
import json
import math
import os
import random
import statistics
import sys

# ============================================================================
# HARD-CODED PRE-REGISTERED CONSTANTS -- each with its governing section.
# ============================================================================
DELTA = 0.005                       # DELTA_D3_BLIND_REPIN.md §4: δ = 0.005 DiD-accuracy-
                                    #   fraction per DECADE of log10(params). Absolute,
                                    #   construction+literature-derived, invariant to any measurement.
SIGMA_BETWEEN_BAND = (0.002, 0.008) # DELTA §3/A1 (Madaan 2024, Table 1) & §5: seed-to-seed SD band.
SIGMA_BETWEEN_CENTRAL = 0.005       # DELTA §5: the RISES/DECLINES treatment (central); a FLAT
                                    #   verdict would instead require the band UPPER end (0.008).
POWER_FACTOR = 2.49                 # DELTA §6 GATE-1: (z_.95 + z_.80) = 1.645 + 0.842 ≈ 2.487.
                                    #   FLAT AVAILABLE iff δ >= 2.49·SE(β̂) (80% power to certify
                                    #   equivalence when β is truly 0).
Z_975 = 1.959964                    # DELTA §5: variance treated as KNOWN (z-quantiles, the FLAT-
Z_95 = 1.644854                     #   favourable choice). 95% CI uses z_.975; TOST 90% uses z_.95.
N_SEEDS = 1                         # DSTATE §1/§5, DELTA §7: one seed per rung. FLAT unavailable here.

# d_state map -- β̂-INVARIANT, from the checkpoint configs (DSTATE_CONFOUND_PREREG.md §1;
# filename-confirmed lmC_..._ds<d_state>_...). Used by GATE-A.
D_STATE = {"14M": 64, "98M": 64, "392M": 128, "Y": 64}
NOMINAL_PARAMS = {"14M": 14_048_896, "98M": 97_618_176, "392M": 391_869_440, "Y": 385_577_984}

PRIMARY_LADDER = ["14M", "98M", "392M"]     # §9.6/§22.3 admissible set A (CONFOUNDED: d_state steps).
CLEAN_LADDER_A64 = ["14M", "98M", "Y"]      # DSTATE §5 A₆₄ (d_state=64 homogeneous; GATE-A passes).

# §9.6 item 7 / §11.7 sample-size floors (per (rung, corpus)).
ROW_FLOOR_CONTRIBUTING = 1500
CANDIDATE_FLOOR_RESOLVED = 8000

# DSTATE §3.4 pre-registered blind-spot / VIF figures for the H-3 caveat print.
H3_BLIND_SPOT = (1.11, 1.92)        # DiD-points/decade window where the free attribution can't adjudicate.
H3_VIF = 2.97                       # corr(x, s) = 0.8148 on {14M,98M,392M}.

DEFAULT_CELLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "results", "param_axis_r0_cells")
N_BOOT = 10000


# ============================================================================
# cell loading (quarantined dir)
# ============================================================================
def load_cells(cells_dir):
    """Load all (rung, corpus) cell JSONs. Returns {rung: {corpus: cell}}."""
    by_rung = {}
    for path in sorted(glob.glob(os.path.join(cells_dir, "*_step*.json"))):
        with open(path) as f:
            cell = json.load(f)
        rung, corpus = cell.get("rung"), cell.get("corpus")
        if rung is None or corpus is None:
            continue
        by_rung.setdefault(rung, {})[corpus] = cell
    return by_rung


def cell_admissible(cell):
    """§9.6/§11.7 per-cell admissibility (records reason; does NOT read `did`'s value
    beyond the presence/CI-excludes-0 booleans the instrument already stamped)."""
    reasons = []
    if not cell.get("_verdict_grade"):
        reasons.append("not verdict-grade")
    if cell.get("cell_void_placebo_match"):
        reasons.append("placebo-match VOID (>5% flagged)")
    if cell.get("cell_void_missing_s2_fields"):
        reasons.append("missing S2 log-prob fields (VOID)")
    nres = cell.get("n_candidates_resolved")
    if nres is None or nres < CANDIDATE_FLOOR_RESOLVED:
        reasons.append(f"n_candidates_resolved {nres} < {CANDIDATE_FLOOR_RESOLVED}")
    contributing = len(cell.get("per_row_did") or {})
    if contributing < ROW_FLOOR_CONTRIBUTING:
        reasons.append(f"contributing_rows {contributing} < {ROW_FLOOR_CONTRIBUTING}")
    if not cell.get("t1a_pass_did_ci_excludes_zero"):
        reasons.append("T1a fail (DiD CI includes 0) -> FLOOR rung")
    return (len(reasons) == 0), reasons


# ============================================================================
# M(r) pooled over corpora + clustered row bootstrap (σ_within)  -- §11.7-D2
# ============================================================================
def per_row_values(cell):
    """per_row_did is {row_idx: DiD}; values are the per-row (cluster) DiD records."""
    return [float(v) for v in (cell.get("per_row_did") or {}).values()]


def pooled_M_and_within(rung_cells, n_boot, seed):
    """rung_cells: list of (corpus, cell). Returns (M_pooled, sigma_within, n_rows_total,
    per_corpus_rowcounts). M_pooled = mean of pooled per-row DiD (both corpora). σ_within =
    clustered bootstrap SE, resampling ROWS within corpus stratum."""
    strata = [per_row_values(cell) for _, cell in rung_cells]
    strata = [s for s in strata if s]
    all_vals = [v for s in strata for v in s]
    if not all_vals:
        return None, None, 0, []
    point = statistics.fmean(all_vals)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        resampled = []
        for vals in strata:                    # stratified: resample each corpus's rows to its own n
            n = len(vals)
            resampled.extend(vals[rng.randrange(n)] for _ in range(n))
        boots.append(statistics.fmean(resampled))
    within = statistics.pstdev(boots)
    return point, within, len(all_vals), [len(s) for s in strata]


def rung_x(rung, cells):
    """log10(params). Prefer the MEASURED n_params (probe-stamped); fall back to nominal.
    Assert measured ≈ nominal (within 0.5%) as a config cross-check."""
    measured = [c.get("n_params") for c in cells.values() if c.get("n_params")]
    p = measured[0] if measured else NOMINAL_PARAMS[rung]
    nom = NOMINAL_PARAMS[rung]
    if measured and abs(p - nom) / nom > 0.005:
        print(f"  WARNING: {rung} measured n_params {p} differs >0.5% from nominal {nom}", file=sys.stderr)
    return math.log10(p), p


# ============================================================================
# BM-INFLATE variance + OLS slope  -- DELTA §5/§6
# ============================================================================
def var_M(sigma_within, sigma_between, n_seeds=N_SEEDS):
    """DELTA §5 BM-INFLATE: Var(M̂(r)) = (σ_within² + σ_between²) / n_seeds."""
    return (sigma_within ** 2 + sigma_between ** 2) / n_seeds


def ols_slope(points, sigma_between):
    """points: list of (x, M, sigma_within). Returns dict with β̂, SE, 95% CI, S_xx.
    β̂ = Σ w_r M_r,  w_r = (x_r - x̄)/S_xx;  Var(β̂) = Σ w_r² Var(M̂(r))  (DELTA §5/§6)."""
    xs = [x for x, _, _ in points]
    xbar = statistics.fmean(xs)
    sxx = sum((x - xbar) ** 2 for x in xs)
    weights = [(x - xbar) / sxx for x in xs]
    beta = sum(w * m for w, (_, m, _) in zip(weights, points))
    var_beta = sum(w ** 2 * var_M(sw, sigma_between) for w, (_, _, sw) in zip(weights, points))
    se = math.sqrt(var_beta)
    return {"beta": beta, "se": se, "ci": [beta - Z_975 * se, beta + Z_975 * se],
            "S_xx": sxx, "xbar": xbar, "weights": weights,
            "leverage_w2": [w ** 2 for w in weights], "sigma_between": sigma_between}


def contrast(m_hi, sw_hi, m_lo, sw_lo, sigma_between):
    """A 2-point difference (Δ₆₄ or γ̂). Independent rungs -> variances add.
    SE = sqrt(Var(M_hi) + Var(M_lo)); 95% CI via z_.975 (DELTA §5 known-variance)."""
    d = m_hi - m_lo
    se = math.sqrt(var_M(sw_hi, sigma_between) + var_M(sw_lo, sigma_between))
    return {"delta": d, "se": se, "ci": [d - Z_975 * se, d + Z_975 * se], "sigma_between": sigma_between}


def factor1(ci):
    """§9.5 Factor 1. FLAT is STRUCK (GATE-1 fails at n_seeds=1), so a non-significant
    slope is INDETERMINATE, never FLAT."""
    lo, hi = ci
    if lo > 0:
        return "RISES"
    if hi < 0:
        return "DECLINES"
    return "INDETERMINATE"      # CI includes 0; FLAT unavailable -> INDETERMINATE (DELTA §7)


# ============================================================================
# GATE-A and GATE-1  (β̂-INVARIANT; emitted ABOVE the slope)
# ============================================================================
def gate_a(ladder, label):
    dstates = sorted({D_STATE[r] for r in ladder})
    passes = (len(dstates) == 1)
    return {"ladder": ladder, "d_states": {r: D_STATE[r] for r in ladder},
            "distinct_d_states": dstates, "passes": passes,
            "rule": "|{d_state(r) : r in ladder}| == 1", "label": label}


def gate_1(points, ladder_label):
    """DELTA §6/§7. β̂-INVARIANT power gate. FLAT AVAILABLE iff (a) δ >= 2.49·SE(β̂)
    at the pinned FLAT requirement σ_between=0.008 (band UPPER, DELTA §5) AND (b)
    σ_between is ESTIMABLE (n_seeds>=3 at >=2 rungs). At n_seeds=1, (b) is False
    (Leg A, structural) and (a) also fails across the whole band (Leg B)."""
    ses = {}
    for sb_name, sb in [("floor_0.002", 0.002), ("central_0.005", 0.005), ("upper_0.008", 0.008)]:
        se = ols_slope(points, sb)["se"]
        ses[sb_name] = {"sigma_between": sb, "SE_beta": se,
                        "2.49*SE": POWER_FACTOR * se, "delta_ge_2.49SE": DELTA >= POWER_FACTOR * se}
    leg_b_pass = ses["upper_0.008"]["delta_ge_2.49SE"]     # pinned FLAT requirement
    leg_a_pass = (N_SEEDS >= 3)                            # σ_between estimable? (needs >=3 at >=2 rungs)
    flat_available = leg_a_pass and leg_b_pass
    return {"ladder_label": ladder_label, "delta": DELTA, "power_factor": POWER_FACTOR,
            "n_seeds": N_SEEDS, "SE_by_sigma_between": ses,
            "leg_A_sigma_between_estimable": leg_a_pass,
            "leg_B_delta_ge_2.49SE_at_upper_band": leg_b_pass,
            "FLAT_available": flat_available,
            "note": ("FLAT STRUCK: unavailable at n_seeds=1 for every β̂ incl. exactly 0 "
                     "(DELTA §7). Reachable verdicts: RISES, DECLINES, INDETERMINATE, FLOOR, VOID.")}


# ============================================================================
# the attribution/verdict map  -- §9.5 Factor 1 × DSTATE §5 x coordinator §34
# ============================================================================
# §34 coordinator adjudication (PARAM_AXIS_SCALING_DESIGN.md §34, record-first,
# commit preceding this fix): FINAL_VERDICT = the MORE CONSERVATIVE of {clean
# A₆₄ grade, confounded 3-rung grade}. The confounded fit is verdict-
# WITHHOLDING ONLY (DSTATE_CONFOUND_PREREG.md §5 "when GATE-A PASSES" / this
# file's §33.1: "it may never grant an attribution the clean legs withhold").
# Higher rank == more conservative/withholding. VOID/FLOOR are not reachable
# via factor1()'s current return set (RISES/DECLINES/INDETERMINATE only) but
# are ranked here so the combinator stays correct if that ever changes.
_VERDICT_RANK = {
    "VOID": 3,            # most conservative: data-quality kill
    "FLOOR": 2,            # T1a fail -> forced floor; as withholding as INDETERMINATE
    "INDETERMINATE": 2,    # CI includes 0 / FLAT-struck non-significance
    "DECLINES": 1,         # directional; grantable only if the OTHER fit agrees in sign
    "RISES": 1,            # directional; grantable only if the OTHER fit agrees in sign
}


def _conservative_combine(f1_clean, f1_confounded):
    """§34: combine the clean A₆₄ grade and the confounded 3-rung grade into the
    single basic FINAL grade (RISES/DECLINES/INDETERMINATE/FLOOR/VOID), applying
    the "confounded may withhold, never grant" rule MECHANICALLY (not as a
    special-cased advisory string). Returns (grade, was_downgraded).

    Rule, in order:
      1. No confounded fit available (None/"n/a") -> nothing to veto with;
         the clean grade stands unchanged.
      2. confounded strictly MORE conservative (higher rank) than clean ->
         downgrade to the confounded grade (this only fires when confounded
         is one of VOID/FLOOR/INDETERMINATE, since those are the only
         grades ranked above the directional RISES/DECLINES rank).
      3. Equal rank (both "directional", i.e. one of RISES/DECLINES each) but
         the two DISAGREE in sign (one RISES, one DECLINES) -> a substantive
         contradiction is not an agreement; treat as AT LEAST as conservative
         as INDETERMINATE and withhold.
      4. Otherwise (confounded same-or-less conservative, or the two agree) ->
         the confounded fit does not/cannot grant beyond the clean fit; the
         clean grade stands.
    """
    if f1_confounded in (None, "n/a"):
        return f1_clean, False
    rank_confounded = _VERDICT_RANK.get(f1_confounded, _VERDICT_RANK["INDETERMINATE"])
    rank_clean = _VERDICT_RANK.get(f1_clean, _VERDICT_RANK["INDETERMINATE"])
    if rank_confounded > rank_clean:
        downgraded_to = f1_confounded if f1_confounded in ("VOID", "FLOOR", "INDETERMINATE") else "INDETERMINATE"
        return downgraded_to, True
    if rank_confounded == rank_clean == 1 and f1_clean != f1_confounded:
        return "INDETERMINATE", True
    return f1_clean, False


def map_verdict(f1_clean, delta64, f1_confounded, gate_a_a64):
    """DSTATE §5 map. With rung Y present, GATE-A PASSES on A₆₄, so the CLEAN A₆₄
    β̂ is the ATTRIBUTABLE primary; the confounded {14M,98M,392M} β̂ is a disclosed,
    verdict-WITHHOLDING sensitivity (may never grant what the clean fit withholds;
    §34 coordinator adjudication makes this a mechanical conservative-min over the
    two grades, not just an advisory line -- see `_conservative_combine`).
    Δ₆₄ is the mandatory clean 2-point contrast, used ONLY for the ATTRIBUTION
    sub-text within a granted RISES headline. Factor 2 (span_frac monotonicity)
    is NOT evaluated from these cells, so COUPLED/DECOUPLED is WITHHELD."""
    d_ci = delta64["ci"]
    d_sig_pos = d_ci[0] > 0
    d_sig_neg = d_ci[1] < 0
    lines = []
    if not gate_a_a64["passes"]:
        return "INDETERMINATE (A₆₄ GATE-A fails -- should not happen with rung Y present)", lines

    combined, downgraded = _conservative_combine(f1_clean, f1_confounded)

    if downgraded:
        v = (f"INDETERMINATE -- clean A64 {f1_clean} but confounded sensitivity withholds "
             f"(confounded reads {f1_confounded}); more-conservative reading governs "
             f"(DSTATE §5 / §33.1; coordinator adjudication §34)")
        lines.append(f"ATTRIBUTION (disclosed, NOT headline): clean A₆₄ slope reads {f1_clean}; "
                     f"confounded {{14M,98M,392M}} reads {f1_confounded}, which VETOES the headline "
                     f"per DSTATE §5/§33.1 (confounded fit may withhold, never grant).")
    elif combined == "RISES":
        if d_sig_pos:
            v = "RISES / ATTRIBUTED (clean A₆₄ slope > 0; Δ₆₄ > 0 confirms 14M→98M leg)"
            lines.append("In-context recall capacity increases with parameter count, d_state held at 64.")
        elif d_sig_neg:
            v = "CONTRADICTED (clean slope >0 but Δ₆₄ <0) -- investigate; param headline NOT clean"
        else:
            v = "RISES-but-INDETERMINATE-attribution (Δ₆₄ CI includes 0)"
            lines.append("Δ₆₄ non-significant -> report blind-spot window and n required (never 'no effect').")
    elif combined == "DECLINES":
        v = "DECLINES / ATTRIBUTED a-fortiori (confound is positively signed; DSTATE §5)"
    else:
        v = f"{combined} (clean A₆₄ slope CI includes 0, or FLOOR/VOID; FLAT unavailable at n_seeds=1)"

    if f1_confounded not in (None, "n/a") and not downgraded and f1_clean == f1_confounded:
        lines.append(f"SENSITIVITY: confounded β̂ agrees ({f1_confounded}) -> clean A₆₄ headline stands, "
                     f"not vetoed.")
    lines.append("Factor 2 (span_frac monotonicity over A) NOT evaluated here -> COUPLED/DECOUPLED "
                 "WITHHELD; report as RECALL-TREND-ONLY unless separately licensed by the T3 probe.")
    return v, lines


# ============================================================================
# main
# ============================================================================
def analyze(cells_dir):
    by_rung = load_cells(cells_dir)
    present = [r for r in ["14M", "98M", "392M", "Y"] if r in by_rung and len(by_rung[r]) >= 1]

    # per-rung pooled M + σ_within (clustered bootstrap). Deterministic seed per rung.
    rung_stats, admissibility = {}, {}
    for i, rung in enumerate(present):
        cells = by_rung[rung]
        rung_cells = list(cells.items())
        M, sw, n_rows, rowcounts = pooled_M_and_within(rung_cells, N_BOOT, seed=1000 + i)
        x, p = rung_x(rung, cells)
        rung_stats[rung] = {"M": M, "sigma_within": sw, "x_log10params": x, "params": p,
                            "n_rows_total": n_rows, "rowcounts_per_corpus": rowcounts,
                            "corpora": sorted(cells), "d_state": D_STATE[rung]}
        admissibility[rung] = {c: {"admissible": cell_admissible(cell)[0],
                                    "reasons": cell_admissible(cell)[1]} for c, cell in cells.items()}
    return by_rung, present, rung_stats, admissibility


def points_for(ladder, rung_stats):
    """(x, M, σ_within) for a ladder; None if any rung missing/unmeasured."""
    pts = []
    for r in ladder:
        s = rung_stats.get(r)
        if not s or s["M"] is None:
            return None
        pts.append((s["x_log10params"], s["M"], s["sigma_within"]))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=DEFAULT_CELLS_DIR)
    ap.add_argument("--out-dir", default=None, help="Where to write r0_verdict.json / R0_VERDICT.md "
                                                    "(default: alongside cells-dir).")
    ap.add_argument("--emit-slope-to-stdout", action="store_true",
                    help="Print β̂/Δ₆₄/γ̂ numerics to stdout AFTER the gates. Default off (files only) "
                         "to keep the coordinator blind before the record step.")
    args = ap.parse_args()
    out_dir = args.out_dir or args.cells_dir

    by_rung, present, rung_stats, admissibility = analyze(args.cells_dir)

    # ---- RECORD STEP: emit the two β̂-INVARIANT gates ABOVE the slope. ----
    print("=" * 72)
    print("R0 beta-FIT -- GATES FIRST (β̂-invariant; recorded before the slope)")
    print("=" * 72)
    ga_primary = gate_a(PRIMARY_LADDER, "primary A={14M,98M,392M}")
    ga_a64 = gate_a(CLEAN_LADDER_A64, "clean A₆₄={14M,98M,Y}")
    print(f"GATE-A [{ga_primary['label']}]: d_states={ga_primary['d_states']} -> "
          f"distinct={ga_primary['distinct_d_states']} -> "
          f"{'PASS' if ga_primary['passes'] else 'FAIL'} (H-3; DSTATE §5)")
    print(f"GATE-A [{ga_a64['label']}]: d_states={ga_a64['d_states']} -> "
          f"distinct={ga_a64['distinct_d_states']} -> "
          f"{'PASS' if ga_a64['passes'] else 'FAIL'}")

    gate1_primary = gate1_clean = None
    pts_primary = points_for(PRIMARY_LADDER, rung_stats)
    pts_a64 = points_for(CLEAN_LADDER_A64, rung_stats)
    if pts_primary:
        gate1_primary = gate_1(pts_primary, "confounded {14M,98M,392M}")
    if pts_a64:
        gate1_clean = gate_1(pts_a64, "clean A₆₄ {14M,98M,Y}")
    for g in (gate1_clean, gate1_primary):
        if g:
            print(f"GATE-1 [{g['ladder_label']}]: n_seeds={g['n_seeds']}; "
                  f"legA(σ_between estimable)={g['leg_A_sigma_between_estimable']}, "
                  f"legB(δ>=2.49·SE @0.008)={g['leg_B_delta_ge_2.49SE_at_upper_band']} -> "
                  f"FLAT_available={g['FLAT_available']}")
    print("  -> FLAT STRUCK at n_seeds=1 (DELTA §7). Reachable: RISES/DECLINES/INDETERMINATE/FLOOR/VOID.")
    print(f"  H-3 caveat: confounded β̂'s 392M rung contributes ~0; blind-spot {H3_BLIND_SPOT} "
          f"DiD-pts/decade; VIF={H3_VIF} (DSTATE §3.4).")
    print(f"  SEED-LABEL DISCLOSURE: 98M is seed s0; 14M/392M/Y are s3. n_seeds=1 per rung regardless.")
    print(f"  rungs present: {present}")

    # ---- Only NOW compute the slopes (written to files; not stdout unless asked). ----
    verdict = {"gate_a_primary": ga_primary, "gate_a_A64": ga_a64,
               "gate_1_primary_confounded": gate1_primary, "gate_1_clean_A64": gate1_clean,
               "rung_stats": {r: {k: v for k, v in s.items()} for r, s in rung_stats.items()},
               "admissibility": admissibility, "rungs_present": present,
               "seed_label_disclosure": {"98M": "s0", "14M": "s3", "392M": "s3", "Y": "s3", "n_seeds": 1},
               "h3_caveat": {"blind_spot_pts_per_decade": list(H3_BLIND_SPOT), "VIF": H3_VIF,
                             "note": "392M contributes ~0 to β̂ net of d_state (saturated 3x3; DSTATE §3.2)."},
               "constants": {"delta_per_decade": DELTA, "sigma_between_band": list(SIGMA_BETWEEN_BAND),
                             "sigma_between_central": SIGMA_BETWEEN_CENTRAL, "power_factor": POWER_FACTOR}}

    delta64 = gamma = beta_conf = beta_clean = None
    s = rung_stats
    if "98M" in s and "14M" in s and s["98M"]["M"] is not None and s["14M"]["M"] is not None:
        delta64 = contrast(s["98M"]["M"], s["98M"]["sigma_within"],
                           s["14M"]["M"], s["14M"]["sigma_within"], SIGMA_BETWEEN_CENTRAL)
        verdict["delta_64"] = delta64
    if "392M" in s and "Y" in s and s["392M"]["M"] is not None and s["Y"]["M"] is not None:
        gamma = contrast(s["392M"]["M"], s["392M"]["sigma_within"],
                        s["Y"]["M"], s["Y"]["sigma_within"], SIGMA_BETWEEN_CENTRAL)
        verdict["gamma_state_width"] = gamma
    if pts_primary:
        beta_conf = ols_slope(pts_primary, SIGMA_BETWEEN_CENTRAL)
        beta_conf["factor1"] = factor1(beta_conf["ci"])
        verdict["beta_confounded_primary"] = beta_conf
    if pts_a64:
        beta_clean = ols_slope(pts_a64, SIGMA_BETWEEN_CENTRAL)
        beta_clean["factor1"] = factor1(beta_clean["ci"])
        verdict["beta_clean_A64"] = beta_clean

    final = "INDETERMINATE (insufficient cells)"
    lines = []
    if beta_clean and delta64:
        final, lines = map_verdict(beta_clean["factor1"],
                                   delta64,
                                   beta_conf["factor1"] if beta_conf else "n/a",
                                   ga_a64)
    verdict["FINAL_VERDICT"] = final
    verdict["verdict_lines"] = lines

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "r0_verdict.json")
    with open(json_path, "w") as f:
        json.dump(verdict, f, indent=2)
    md_path = os.path.join(out_dir, "R0_VERDICT.md")
    write_md(md_path, verdict)

    print("\nVerdict written (gates above; slopes in files):")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"FINAL VERDICT: {final}")
    for ln in lines:
        print(f"  - {ln}")
    if args.emit_slope_to_stdout:
        print("\n[slope numerics, post-record]")
        for name, obj in [("Δ₆₄", delta64), ("γ̂", gamma),
                          ("β̂ confounded {14M,98M,392M}", beta_conf), ("β̂ clean A₆₄", beta_clean)]:
            if obj:
                key = "delta" if "delta" in obj else "beta"
                print(f"  {name}: {obj[key]:+.5f}  95% CI [{obj['ci'][0]:+.5f}, {obj['ci'][1]:+.5f}]")
    return 0


def write_md(path, v):
    L = ["# R0 VERDICT (blind β-fit)\n",
         "**Gates first (β̂-invariant, recorded before the slope):**\n",
         f"- GATE-A primary {v['gate_a_primary']['distinct_d_states']} -> "
         f"{'PASS' if v['gate_a_primary']['passes'] else 'FAIL'} (H-3, DSTATE §5)",
         f"- GATE-A A₆₄ {v['gate_a_A64']['distinct_d_states']} -> "
         f"{'PASS' if v['gate_a_A64']['passes'] else 'FAIL'}"]
    if v.get("gate_1_clean_A64"):
        L.append(f"- GATE-1 (clean A₆₄): FLAT_available={v['gate_1_clean_A64']['FLAT_available']} "
                 f"(FLAT struck at n_seeds=1, DELTA §7)")
    L += [f"\n**Seed label:** 98M=s0, others=s3; n_seeds=1 per rung.",
          f"**H-3 caveat:** confounded 392M contributes ~0; blind-spot {v['h3_caveat']['blind_spot_pts_per_decade']} "
          f"pts/decade; VIF={v['h3_caveat']['VIF']}.",
          f"**Factor 2 (span_frac) NOT evaluated here** -> COUPLED/DECOUPLED withheld.\n",
          f"## FINAL VERDICT\n\n**{v['FINAL_VERDICT']}**\n"]
    for ln in v.get("verdict_lines", []):
        L.append(f"- {ln}")
    L.append("\n(Numeric β̂/Δ₆₄/γ̂ in r0_verdict.json.)")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
