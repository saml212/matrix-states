"""analyze_fixscale_harvest.py -- FIX-AT-SCALE wave (FROZEN_BIAS_LM_DESIGN.md sec 13) harvest
analysis, 2026-07-10. Computes the sec 13.6 pre-registered WIN/PARTIAL/NULL verdict directly
from the raw artifacts (verify-vs-raws, sec 13.10 gate 7):

  - pins/BANDS_PINNED-FrozenBias-{98M,392M}.json  (arm_off's own bands: arm1prime = arm_off'
    per-token eval-retrofit, arm1double = arm_off'' global eval-retrofit, arm1_kraw = pre-blend
    co-primary reference, arm1 val-loss tolerance)
  - measure/comparator_*.json  (harvest-run shared-forward-pass comparator on each
    arm_per_token / arm_global_probe FINAL checkpoint: per_token mode = that arm's own realized
    post-blend geometry, kraw = its pre-blend co-primary, global = probe's own post-blend)
  - train/*.json + calib/*.json  (val losses for the sec 13.4 gate)

CI convention (pinned, sec 7.1-real.1 / the pin's own formula_version): two-sided
mean_delta +/- t(n-1,0.975) * s_ref / sqrt(n), n=3, t=4.303, s_ref = the REFERENCE arm's
(arm_off-retrofit) cross-seed sample std at THIS scale -- i.e. the pin's own
pinned_ci_half_width, never inherited from 14M. The probe is n=1: descriptive only, no CI
(sec 13.6 Rev 1 note: it never gates the WIN/PARTIAL/NULL bar).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCALES = {"98m": "98M", "392m": "392M"}
CORPORA = ["openr1-mix-ext", "wikitext-mix-ext"]
SEEDS = [0, 1, 2]


def span_frac_pooled(comparator_doc, mode):
    """Mean of per-layer span_frac (None-safe) -- byte-matches fixscale_wave._span_frac_for_corpus."""
    vals = [v["span_frac"] for v in comparator_doc["per_mode_per_layer"][mode].values()
            if v.get("span_frac") is not None]
    assert vals, f"no scored layers for mode={mode}"
    return sum(vals) / len(vals)


def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


T_2_975 = 4.303  # t(2, 0.975), pinned


def main():
    report = {"wave": "fix-at-scale (FROZEN_BIAS_LM_DESIGN.md sec 13)", "harvest_date": "2026-07-10",
              "ci_formula": "mean_delta +/- t(2,0.975)*s_ref/sqrt(3), s_ref = reference-arm "
                            "(arm_off eval-retrofit) cross-seed std from this scale's own pin",
              "scales": {}}
    for scale, pin_key in SCALES.items():
        pin = json.load(open(os.path.join(HERE, "pins", f"BANDS_PINNED-FrozenBias-{pin_key}.json")))
        sc = {"corpora": {}}
        for corpus in CORPORA:
            # --- arm_per_token: harvest comparator on its own final checkpoints ---
            pt_post, pt_kraw = [], []
            for seed in SEEDS:
                cd = json.load(open(os.path.join(
                    HERE, "measure",
                    f"comparator_fixscale_train_arm_per_token_{scale}_{corpus}_s{seed}.json")))
                pt_post.append(span_frac_pooled(cd, "per_token"))
                pt_kraw.append(span_frac_pooled(cd, "kraw"))
            # --- probe (n=1) ---
            pd = json.load(open(os.path.join(
                HERE, "measure",
                f"comparator_fixscale_train_arm_global_probe_{scale}_{corpus}_s0.json")))
            probe_post = span_frac_pooled(pd, "global")
            probe_kraw = span_frac_pooled(pd, "kraw")

            ref_prime = pin["arm1prime_span_frac_bands"][corpus]     # arm_off' per-token retrofit
            ref_double = pin["arm1double_span_frac_bands"][corpus]   # arm_off'' global retrofit
            ref_kraw = pin["arm1_kraw_span_frac_bands"][corpus]      # arm_off pre-blend
            vt = pin["arm1_val_loss_tolerance"][corpus]

            # PRIMARY: per_token post-blend vs arm_off' (sec 13.4/13.6)
            d_primary = mean(pt_post) - ref_prime["mean"]
            hw_primary = ref_prime["pinned_ci_half_width"]
            # CO-PRIMARY: per_token pre-blend k_raw vs arm_off k_raw
            d_co = mean(pt_kraw) - ref_kraw["mean"]
            hw_co = ref_kraw["pinned_ci_half_width"]

            # val-loss gate (arm mean vs pass_ceiling, rung-1 convention)
            vl_pt = []
            for seed in SEEDS:
                td = json.load(open(os.path.join(
                    HERE, "train", f"fixscale_train_arm_per_token_{scale}_{corpus}_s{seed}.json")))
                vl_pt.append(td["checkpoints"][-1]["val_loss"][corpus])
            probe_td = json.load(open(os.path.join(
                HERE, "train", f"fixscale_train_arm_global_probe_{scale}_{corpus}_s0.json")))
            vl_probe = probe_td["checkpoints"][-1]["val_loss"][corpus]

            def verdict_ci(d, hw):
                lo, hi = d - hw, d + hw
                if hi < 0:
                    return "CI-excludes-zero NEGATIVE (stabilizing = mechanism-predicted)"
                if lo > 0:
                    return "CI-excludes-zero POSITIVE (destabilizing = the 14M sign)"
                return "CI includes zero (null)"

            sc["corpora"][corpus] = {
                "per_token_post_blend_span_frac_per_seed": pt_post,
                "per_token_post_blend_mean": mean(pt_post),
                "per_token_post_blend_own_std": sample_std(pt_post),
                "arm_off_prime_band": {k: ref_prime[k] for k in ("mean", "s", "pinned_ci_half_width", "per_seed")},
                "PRIMARY_delta": d_primary,
                "PRIMARY_ci": [d_primary - hw_primary, d_primary + hw_primary],
                "PRIMARY_reading": verdict_ci(d_primary, hw_primary),
                "per_token_kraw_span_frac_per_seed": pt_kraw,
                "per_token_kraw_mean": mean(pt_kraw),
                "arm_off_kraw_band": {k: ref_kraw[k] for k in ("mean", "s", "pinned_ci_half_width", "per_seed")},
                "COPRIMARY_delta": d_co,
                "COPRIMARY_ci": [d_co - hw_co, d_co + hw_co],
                "COPRIMARY_reading": verdict_ci(d_co, hw_co),
                "val_loss_gate": {
                    "per_token_per_seed": vl_pt, "per_token_mean": mean(vl_pt),
                    "arm_off_mean": vt["mean"], "pass_ceiling": vt["pass_ceiling"],
                    "PASS_by_arm_mean": mean(vl_pt) <= vt["pass_ceiling"],
                    "per_seed_over_ceiling": [v for v in vl_pt if v > vt["pass_ceiling"]],
                    "probe_val_loss": vl_probe, "probe_PASS": vl_probe <= vt["pass_ceiling"],
                },
                "probe_exploratory_n1_no_CI": {
                    "tier": "exploratory-probe -- NOT a confirmatory bar, n=1",
                    "probe_post_blend_global_span_frac": probe_post,
                    "arm_off_double_band": {k: ref_double[k] for k in ("mean", "s", "pinned_ci_half_width", "per_seed")},
                    "probe_delta_vs_arm_off_double_mean": probe_post - ref_double["mean"],
                    "probe_kraw_span_frac": probe_kraw,
                    "probe_kraw_delta_vs_arm_off_kraw_mean": probe_kraw - ref_kraw["mean"],
                },
            }
        report["scales"][scale] = sc

    # ---- sec 13.6 verdict, mechanical ----
    def scale_outcome(sc):
        prim = [sc["corpora"][c]["PRIMARY_reading"] for c in CORPORA]
        co = [sc["corpora"][c]["COPRIMARY_reading"] for c in CORPORA]
        gate = all(sc["corpora"][c]["val_loss_gate"]["PASS_by_arm_mean"] for c in CORPORA)
        win = all("NEGATIVE" in r for r in prim + co) and gate
        partial = any("POSITIVE" in r for r in prim + co)
        null = all("includes zero" in r for r in prim + co)
        if win:
            return "WIN"
        if partial:
            return "PARTIAL (destabilizing 14M sign persists on >=1 corpus/instrument)"
        if null:
            return "NULL-at-this-scale"
        return "SPLIT (mixed null/negative -- per sec 13.6 PARTIAL's split clause)"

    report["verdict_per_scale"] = {s: scale_outcome(report["scales"][s]) for s in SCALES}
    outcomes = list(report["verdict_per_scale"].values())
    if all(o == "WIN" for o in outcomes):
        overall = "WIN"
    elif all(o.startswith("NULL") for o in outcomes):
        overall = "NULL"
    else:
        overall = "PARTIAL" if any("PARTIAL" in o or "SPLIT" in o for o in outcomes) else "MIXED"
    report["verdict_overall_sec13_6"] = overall

    out = os.path.join(HERE, "fixscale_harvest_verdict.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["verdict_per_scale"], indent=2))
    print("overall:", overall)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
