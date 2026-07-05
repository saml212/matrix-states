"""rev7_threshold_derive.py — KEY_ANCHORING_DESIGN.md §10.3.3 (Rev 7.1).

Pure derivation of every constant in the Rev-7 engagement test
(REV7_THRESHOLD_PINNED.json). The derivation takes ONLY the registered
constants {n_entities, d_state, alpha} — it reads NO wave data, NO result
JSON, NO checkpoint, and no file of any kind except its own source (to hash
itself into the pin's provenance block). This is the zero-data-dependency
property that §10.9 smoke 5 asserts mechanically: `derive()` is a pure
function of three numbers, and running this script in a fresh sandbox with
no experiment data present must produce a byte-identical `derived` block.

Everything is pure Python + math (no scipy/numpy/torch): the regularized
incomplete beta is a Lentz continued fraction, the normal quantile is a
bisection on math.erf — so the pin can be regenerated and verified on any
machine, including the CPU-only sandboxes attack rounds run in
(KEYANCHOR_REV7_ATTACK.md reproduced these numbers exactly this way).

Registered statistical facts this encodes (all re-derived independently by
the Rev-7 attack round, KEYANCHOR_REV7_ATTACK.md §1, before this script
existed — the constants below must match its published table):
  - For a uniformly random unit direction in R^d, (1+r)/2 ~ Beta(a, a) with
    a = (d-1)/2 EXACTLY (classical marginal f(r) ∝ (1-r²)^((d-3)/2)).
    At d_state=64: a = 31.5, Var(r) = 1/d = 1/64, sigma_chance = 0.125.
  - The PRIMARY per-entity test uses the exact Beta survival function for
    the one-sided p-value; the Gaussian z-approximation is a DISCLOSED
    APPROXIMATION only (its tails are heavier than the true bounded-support
    distribution, by a factor that grows with r — quantified in the
    reference table this script emits).
  - Multiplicity: Benjamini–Hochberg step-up at q across n simultaneous
    tests (primary); Bonferroni and Benjamini–Yekutieli always reported
    alongside as cross-checks, never substituted.

Usage:
    python rev7_threshold_derive.py [--out PATH]
Default output: REV7_THRESHOLD_PINNED.json next to this script.
"""

import argparse
import datetime
import hashlib
import json
import math
import os

# ---------------------------------------------------------------------------
# Registered inputs (the ONLY free quantities in this file, per §10.3.3).
# ---------------------------------------------------------------------------
N_ENTITIES = 107   # train-pool size, fixed since before Wave 1 (grammar_rd.py::build_entity_pools)
D_STATE = 64       # architectural constant, fixed since before Wave 1
ALPHA = 0.05       # generic convention: BH q AND the Bonferroni family alpha

# Registered effect-size floor for the HEADLINE (A-grade) band, §10.3.4.
# This is a REGISTERED LITERAL with disclosed provenance, not a computed
# quantity: the cross-leg median-of-medians of the prior confirm-wave's
# back-solved r_e (0.350 / 0.326 / 0.359 / 0.371, KEY_ANCHORING_DESIGN.md
# §9.7.2's published table -> median-of-medians 0.3545), rounded DOWN to the
# 0.05 grid. Provenance is prior-wave PUBLISHED summaries only — numbers
# fixed and public before any Rev-7 wave data exists; this script still
# reads no file to obtain it. The A″-grade floor is fully derived
# (2·sigma_chance) and computed below, not a literal.
R_MIN_HEADLINE_REGISTERED = 0.35
R_MIN_HEADLINE_PROVENANCE = (
    "prior confirm-wave cross-leg median-of-medians of back-solved r_e "
    "(0.350/0.326/0.359/0.371, KEY_ANCHORING_DESIGN.md sec 9.7.2 table -> "
    "0.3545), rounded down to the 0.05 grid; registered 2026-07-05, before "
    "any Rev-7 wave data exists"
)


# ---------------------------------------------------------------------------
# Exact Beta machinery (pure Python; Numerical-Recipes-style Lentz CF).
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-15) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("betacf failed to converge (a=%r b=%r x=%r)" % (a, b, x))


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_one_sided_exact(r: float, d_state: int) -> float:
    """Exact one-sided null p-value P(R >= r) for the cosine of a random
    unit direction in R^d against a fixed axis: (1+R)/2 ~ Beta(a, a),
    a = (d-1)/2."""
    a = (d_state - 1) / 2.0
    return 1.0 - betainc_reg(a, a, (1.0 + r) / 2.0)


def r_at_tail_exact(p_target: float, d_state: int, tol: float = 1e-12) -> float:
    """Invert p_one_sided_exact by bisection on r in [0, 1)."""
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_one_sided_exact(mid, d_state) > p_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Gaussian machinery (DISCLOSED APPROXIMATION only — never the operative
# number anywhere in §10; kept so the exact-vs-approx gap is quantified in
# the pin itself rather than asserted qualitatively).
# ---------------------------------------------------------------------------
def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def z_quantile(p_upper_tail: float, tol: float = 1e-13) -> float:
    """z with P(Z > z) = p_upper_tail, by bisection on erf."""
    lo, hi = 0.0, 40.0
    target = 1.0 - p_upper_tail
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# The derivation — a pure function of (n_entities, d_state, alpha).
# ---------------------------------------------------------------------------
def derive(n_entities: int = N_ENTITIES, d_state: int = D_STATE,
           alpha: float = ALPHA) -> dict:
    a_shape = (d_state - 1) / 2.0
    sigma_chance = 1.0 / math.sqrt(d_state)

    # Bonferroni cross-check (family-wise), exact Beta = the operative number.
    bonf_tail = alpha / n_entities
    r_crit_exact = r_at_tail_exact(bonf_tail, d_state)
    z_equiv_exact = r_crit_exact * math.sqrt(d_state)
    # Gaussian approximation of the same tail (disclosed approximation only).
    z_crit_gauss = z_quantile(bonf_tail)
    r_crit_gauss = z_crit_gauss / math.sqrt(d_state)

    # BH step-up thresholds (primary procedure): p_(k) <= (k/n)*q.
    bh_thresholds = [(k / n_entities) * alpha for k in range(1, n_entities + 1)]

    # BY (arbitrary-dependence) correction factor, always reported alongside.
    by_factor = sum(1.0 / i for i in range(1, n_entities + 1))
    by_effective_q = alpha / by_factor

    # A″-grade effect-size floor: fully derived, 2·sigma_chance (§3.6's own
    # registered mean+2s multiplier, applied to the null scale). At d=64
    # this is 0.25 exactly.
    r_min_partial = 2.0 * sigma_chance
    # Consistency reference (derived, not a floor): the r at which the exact
    # Beta p equals the BH step-up threshold at the median rank
    # k = ceil(n/2) — i.e. the effect size at which a half-engaged pool's
    # median entity clears its own BH bar.
    k_median = (n_entities + 1) // 2
    r_at_bh_median_rank = r_at_tail_exact((k_median / n_entities) * alpha, d_state)

    # Reference p-value table (exact vs. Gaussian) at load-bearing r values —
    # quantifies the tail gap §10.3.1 discloses (exact Beta has LIGHTER tails;
    # gap grows with r, i.e. the exact primary test is MORE powerful at the
    # r values the prior wave already showed).
    ref_rs = [0.125, 0.25, 0.35, round(r_crit_exact, 4), round(r_crit_gauss, 4), 0.581]
    ref_table = []
    for r in ref_rs:
        pe = p_one_sided_exact(r, d_state)
        pg = 1.0 - phi(r * math.sqrt(d_state))
        ref_table.append({
            "r": r,
            "p_exact_beta": float("%.6g" % pe),
            "p_gaussian_approx": float("%.6g" % pg),
            "ratio_exact_over_gauss": float("%.4g" % (pe / pg)),
        })

    return {
        "inputs": {
            "n_entities": n_entities,
            "d_state": d_state,
            "alpha": alpha,
            "note": ("alpha serves as BOTH the BH q and the Bonferroni "
                     "family alpha; no other free parameter exists"),
        },
        "null": {
            "beta_shape_a": a_shape,
            "distribution": "(1+r)/2 ~ Beta(a, a), a=(d_state-1)/2, EXACT",
            "sigma_chance": sigma_chance,
            "var_r": 1.0 / d_state,
        },
        "primary_bh": {
            "procedure": ("per-entity one-sided p from the EXACT Beta survival "
                          "function; Benjamini-Hochberg step-up at q over "
                          "n_entities simultaneous tests"),
            "q": alpha,
            "n_tests": n_entities,
            "step_up_thresholds": [float("%.8g" % t) for t in bh_thresholds],
        },
        "bonferroni_crosscheck": {
            "tail": float("%.8g" % bonf_tail),
            "r_crit_exact_beta": float("%.6g" % r_crit_exact),
            "z_equiv_exact_beta": float("%.6g" % z_equiv_exact),
            "r_crit_gaussian_approx_DISCLOSED_ONLY": float("%.6g" % r_crit_gauss),
            "z_crit_gaussian_approx_DISCLOSED_ONLY": float("%.6g" % z_crit_gauss),
            "primacy": ("r_crit_exact_beta is the operative cross-check number "
                        "everywhere in sec 10; the Gaussian values are a "
                        "disclosed approximation, never substituted"),
        },
        "by_crosscheck": {
            "by_factor_harmonic": float("%.6g" % by_factor),
            "by_effective_q": float("%.6g" % by_effective_q),
            "role": ("BH assumes independence/PRDS across the 107 per-entity "
                     "tests; BY (arbitrary dependence) discovery counts are "
                     "ALWAYS reported alongside BH, never substituted — same "
                     "pattern as the Bonferroni cross-check"),
        },
        "effect_size_floors": {
            "r_min_partial_band": r_min_partial,
            "r_min_partial_derivation": ("2*sigma_chance = 2/sqrt(d_state) — "
                                          "sec 3.6's own registered mean+2s "
                                          "multiplier applied to the null scale"),
            "r_at_bh_median_rank_consistency_ref": float("%.6g" % r_at_bh_median_rank),
            "r_min_headline_band": R_MIN_HEADLINE_REGISTERED,
            "r_min_headline_provenance": R_MIN_HEADLINE_PROVENANCE,
        },
        "bands": {
            "quantity": ("engaged_frac_v3 = BH discovery rate (a significance-"
                         "rate quantity, NOT the old sec 3.7 magnitude fraction; "
                         "the old 90/50 magnitude bands do NOT carry over)"),
            "A_headline": ("discovery rate >= 0.90 (3/3 seeds) AND median r_e "
                           "(all 107 entities) >= r_min_headline_band"),
            "A_partial": ("discovery rate >= 0.50 AND median r_e >= "
                          "r_min_partial_band"),
            "C_not_engaged": ("discovery rate < 0.50 OR median r_e < "
                              "r_min_partial_band"),
        },
        "reference_p_table": ref_table,
    }


# ---------------------------------------------------------------------------
# Writer (the only I/O in this file: read own source for the hash, write pin).
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "REV7_THRESHOLD_PINNED.json")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    derived = derive()
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()

    pin = {
        "artifact": "REV7_THRESHOLD_PINNED",
        "design_ref": "KEY_ANCHORING_DESIGN.md sec 10.3 (Rev 7.1)",
        # The derived block is DETERMINISTIC: byte-identical across machines,
        # working directories, and data environments (smoke 5's assertion).
        "derived": derived,
        "provenance": {
            "script": "rev7_threshold_derive.py",
            "script_sha256": script_sha,
            "generated_at": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_dependency": ("NONE — derive() is a pure function of "
                                "{n_entities, d_state, alpha}; the only file "
                                "this script ever reads is its own source, "
                                "for this hash"),
        },
    }
    with open(args.out, "w") as f:
        json.dump(pin, f, indent=2)
        f.write("\n")
    print("wrote %s" % args.out)
    print("script_sha256 = %s" % script_sha)
    print("r_crit_exact_beta (Bonferroni tail %.4g) = %.6g"
          % (derived["bonferroni_crosscheck"]["tail"],
             derived["bonferroni_crosscheck"]["r_crit_exact_beta"]))
    print("gaussian approx (disclosed only): z=%.6g r=%.6g"
          % (derived["bonferroni_crosscheck"]["z_crit_gaussian_approx_DISCLOSED_ONLY"],
             derived["bonferroni_crosscheck"]["r_crit_gaussian_approx_DISCLOSED_ONLY"]))
    print("r_min_partial=%.4g  r_at_bh_median_rank=%.6g  r_min_headline=%.4g"
          % (derived["effect_size_floors"]["r_min_partial_band"],
             derived["effect_size_floors"]["r_at_bh_median_rank_consistency_ref"],
             derived["effect_size_floors"]["r_min_headline_band"]))


if __name__ == "__main__":
    main()
