"""build_fit_inputs_rankstats.py -- FROZEN_BIAS_LM_DESIGN.md sec 12.3.1/12.3.2's
Stage-0 H2 harvest: effective_rank_mean/stable_rank_mean reharvest of the SAME
48 already-archived retrofit_*.json files build_fit_inputs_and_run.py already
parses for span_frac (experiment-runs/2026-07-06_frozen_bias_rung1/results/
frozen_bias_lm/), swapping which field of analyze_probe_wave2.pooled_subset()'s
return dict is read, feeding the resulting per-seed values into the UNMODIFIED
fit_frozenbias_estimation.derive_estimation() -- same pinned t(2,.975)=4.303 CI
formula, zero new statistical machinery.

Sec 12.0 framing applies throughout: this is Tier-2 exploratory/mechanistic,
NEVER confirmatory, regardless of how clean a number comes back. Every output
is persisted through mech_schema.wrap_exploratory (sec 12.3.1's schema
requirement) -- this script does NOT import `headline_verdict` (forbidden
anywhere in sec 12, sec 12.3.1 item 3).

Produces the full 4-way grid (2 comparisons from sec 12.3.1 + the missing
Arm2'-pre-blend comparison from sec 12.3.2, each read for both effective_rank
and stable_rank, on both corpora):
  1. Arm2  - Arm1'  (post-blend, mode="arm1prime")   mechanism_direction="negative"
  2. Arm2  - Arm1   (pre-blend,  mode="kraw")         mechanism_direction="negative"
  3. Arm2' - Arm1'' (post-blend, mode="arm1double")   mechanism_direction="positive"
  4. Arm2' - Arm1   (pre-blend,  mode="kraw", arm2p_) mechanism_direction="positive"  [sec 12.3.2, never-yet-fed]

Run entirely on the Mac -- 0 GPU-h, pure Python over already-repo-committed
JSON, no SSH needed.

Usage:
  DRY_RUN_BYPASS=1 python3 build_fit_inputs_rankstats.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from fit_frozenbias_estimation import derive_estimation  # noqa: E402  (headline_verdict NOT imported -- sec 12.3.1 item 3)
from mech_schema import wrap_exploratory  # noqa: E402

# ---------------------------------------------------------------------------
# Load analyze_probe_wave2.pooled_subset() from its archived location (the
# SAME dependency build_fit_inputs_and_run.py already uses on the box, loaded
# here from the repo-local mirror since this script runs on the Mac).
# ---------------------------------------------------------------------------
_ANALYZE_PATH = os.path.join(
    _REPO_ROOT, "experiment-runs", "2026-07-05_trackc_rung2", "analyze_probe_wave2.py")
_spec = importlib.util.spec_from_file_location("analyze_probe_wave2_rankstats", _ANALYZE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pooled_subset = _mod.pooled_subset

RESULTS_DIR = os.path.join(
    _REPO_ROOT, "experiment-runs", "2026-07-06_frozen_bias_rung1", "results", "frozen_bias_lm")
OUT_DIR = os.path.join(_HERE, "results", "mech_wave")
CORPORA = ["openr1-mix-ext", "wikitext-mix-ext"]
CORPUS_SHORT = {"openr1-mix-ext": "openr1", "wikitext-mix-ext": "wikitext"}
SEEDS = [0, 1, 2]
N_ARCHIVED_RETROFIT_JSONS = 48  # sec 12.3.1's corrected count (MINOR-2 fix), verified below


def rankstats_from_retrofit_json(path: str, corpus: str):
    """Same shape as build_fit_inputs_and_run.py's span_frac_from_retrofit_json,
    but reads effective_rank_mean/stable_rank_mean (already computed by
    pooled_subset, unused as a headline until now) instead of computing
    span_frac from gram_deviation_mean."""
    with open(path) as f:
        d = json.load(f)
    assert len(d["per_checkpoint"]) == 1, f"{path}: expected 1 checkpoint"
    ckpt_entry = list(d["per_checkpoint"].values())[0]
    pooled = pooled_subset(d, [corpus])
    assert pooled is not None and pooled["n_scored"] > 0, f"{path}: no scored episodes for {corpus}"
    return pooled["effective_rank_mean"], pooled["stable_rank_mean"], ckpt_entry


def extract_rankstats_group(mode: str, run_prefix: str, corpus_trained_check: bool = True):
    """mode: 'arm1prime'|'arm1double'|'kraw'. run_prefix: e.g. 'arm2_'/'arm2p_'/'arm1_'.
    Returns (effective_rank_by_corpus, stable_rank_by_corpus), each
    {corpus: [v_seed0, v_seed1, v_seed2]} -- mirrors extract_group's own return shape
    (build_fit_inputs_and_run.py), split into the two stat families."""
    eff_out, stab_out = {}, {}
    for corpus in CORPORA:
        short = CORPUS_SHORT[corpus]
        eff_vals, stab_vals = [], []
        for seed in SEEDS:
            path = os.path.join(RESULTS_DIR, f"retrofit_{mode}_{run_prefix}{short}_s{seed}.json")
            eff, stab, ckpt_entry = rankstats_from_retrofit_json(path, corpus)
            if corpus_trained_check:
                assert ckpt_entry["corpus_trained_on"] == corpus, (
                    f"{path}: corpus_trained_on={ckpt_entry['corpus_trained_on']!r} != {corpus!r}")
            eff_vals.append(eff)
            stab_vals.append(stab)
        eff_out[corpus] = eff_vals
        stab_out[corpus] = stab_vals
    return eff_out, stab_out


def _verify_archived_count():
    n = len([f for f in os.listdir(RESULTS_DIR) if f.startswith("retrofit_") and f.endswith(".json")])
    assert n == N_ARCHIVED_RETROFIT_JSONS, (
        f"expected {N_ARCHIVED_RETROFIT_JSONS} archived retrofit_*.json files "
        f"(sec 12.3.1's corrected count), found {n} at {RESULTS_DIR}")
    return n


def _comparison_grid(eff_a, stab_a, eff_b, stab_b, mechanism_direction: str):
    """Runs derive_estimation for both stats, both corpora, given the two
    groups' per-seed values (group A minus group B), under the pinned
    mechanism_direction for this comparison (sec 12.3.1's MAJOR-3 fix: pinned
    per comparison, never chosen after seeing the result)."""
    out = {"effective_rank_mean": {}, "stable_rank_mean": {}}
    for corpus in CORPORA:
        deltas_eff = [a - b for a, b in zip(eff_a[corpus], eff_b[corpus])]
        deltas_stab = [a - b for a, b in zip(stab_a[corpus], stab_b[corpus])]
        out["effective_rank_mean"][corpus] = derive_estimation(deltas_eff, corpus, mechanism_direction)
        out["stable_rank_mean"][corpus] = derive_estimation(deltas_stab, corpus, mechanism_direction)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_archived = _verify_archived_count()
    print(f"verified {n_archived} archived retrofit_*.json files at {RESULTS_DIR}")

    # ---- Extract the 5 groups needed across the 4-way grid ----
    print("\nextracting rank-stat groups from archived retrofit JSONs...")
    arm2_post_eff, arm2_post_stab = extract_rankstats_group("arm1prime", "arm2_")
    arm1p_post_eff, arm1p_post_stab = extract_rankstats_group("arm1prime", "arm1_")
    arm2_kraw_eff, arm2_kraw_stab = extract_rankstats_group("kraw", "arm2_")
    arm1_kraw_eff, arm1_kraw_stab = extract_rankstats_group("kraw", "arm1_")
    arm2p_post_eff, arm2p_post_stab = extract_rankstats_group("arm1double", "arm2p_")
    arm1pp_post_eff, arm1pp_post_stab = extract_rankstats_group("arm1double", "arm1_")
    arm2p_kraw_eff, arm2p_kraw_stab = extract_rankstats_group("kraw", "arm2p_")  # sec 12.3.2, never-yet-fed

    # ---- 4-way grid, pinned mechanism_direction per comparison (sec 12.3.1 MAJOR-3) ----
    print("\ncomputing 4-way grid (sec 12.3.1 comparisons 1-2, sec 12.3.2 comparison 4)...")
    comparison_1_arm2_vs_arm1prime_post = _comparison_grid(
        arm2_post_eff, arm2_post_stab, arm1p_post_eff, arm1p_post_stab, "negative")
    comparison_2_arm2_vs_arm1_pre = _comparison_grid(
        arm2_kraw_eff, arm2_kraw_stab, arm1_kraw_eff, arm1_kraw_stab, "negative")
    comparison_3_arm2prime_vs_arm1doubleprime_post = _comparison_grid(
        arm2p_post_eff, arm2p_post_stab, arm1pp_post_eff, arm1pp_post_stab, "positive")
    comparison_4_arm2prime_vs_arm1_pre = _comparison_grid(
        arm2p_kraw_eff, arm2p_kraw_stab, arm1_kraw_eff, arm1_kraw_stab, "positive")

    grid = {
        "comparison_1_arm2_vs_arm1prime_post_blend": comparison_1_arm2_vs_arm1prime_post,
        "comparison_2_arm2_vs_arm1_pre_blend": comparison_2_arm2_vs_arm1_pre,
        "comparison_3_arm2prime_vs_arm1doubleprime_post_blend": comparison_3_arm2prime_vs_arm1doubleprime_post,
        "comparison_4_arm2prime_vs_arm1_pre_blend_MISSING_UNTIL_NOW": comparison_4_arm2prime_vs_arm1_pre,
    }

    for name, comp in grid.items():
        print(f"\n--- {name} ---")
        for stat, per_corpus in comp.items():
            for corpus, r in per_corpus.items():
                print(f"  {stat} [{corpus}]: delta={r['mean_delta']:.6f} "
                      f"CI=[{r['ci_lower']:.6f},{r['ci_upper']:.6f}] "
                      f"excludes_zero={r['excludes_zero']} "
                      f"dir_consistent={r['confirm_direction_consistent']}")

    # ---- Sanity-anchor check against the attack-round-executed reference values ----
    # sec 12.3.1's "Expected-sign sanity anchors" (Arm2-vs-Arm1 pre-blend, comparison 2):
    #   effective_rank Δ=-2.09 [-2.73,-1.46] (openr1-mix-ext)
    #   stable_rank    Δ=-1.75 [-2.26,-1.24] (wikitext-mix-ext)
    anchor_eff_openr1 = comparison_2_arm2_vs_arm1_pre["effective_rank_mean"]["openr1-mix-ext"]
    anchor_stab_wikitext = comparison_2_arm2_vs_arm1_pre["stable_rank_mean"]["wikitext-mix-ext"]
    print("\n--- sanity-anchor reproduction check (sec 12.3.1) ---")
    print(f"  openr1-mix-ext effective_rank Δ={anchor_eff_openr1['mean_delta']:.6f} "
          f"[{anchor_eff_openr1['ci_lower']:.6f},{anchor_eff_openr1['ci_upper']:.6f}] "
          f"(attack-round reference: -2.09 [-2.73,-1.46])")
    print(f"  wikitext-mix-ext stable_rank Δ={anchor_stab_wikitext['mean_delta']:.6f} "
          f"[{anchor_stab_wikitext['ci_lower']:.6f},{anchor_stab_wikitext['ci_upper']:.6f}] "
          f"(attack-round reference: -1.75 [-2.26,-1.24])")
    anchor_eff_ok = abs(anchor_eff_openr1["mean_delta"] - (-2.09)) < 0.02
    anchor_stab_ok = abs(anchor_stab_wikitext["mean_delta"] - (-1.75)) < 0.02
    print(f"  anchor reproduction (within 0.02 of attack-round 2-decimal figures): "
          f"effective_rank={anchor_eff_ok} stable_rank={anchor_stab_ok}")

    # ---- Ambiguous-cell disclosure: wikitext-mix-ext effective_rank, comparison 2 ----
    ambiguous_cell = comparison_2_arm2_vs_arm1_pre["effective_rank_mean"]["wikitext-mix-ext"]
    print("\n--- expected-ambiguous cell (sec 12.3.1 honesty requirement) ---")
    print(f"  wikitext-mix-ext effective_rank (comparison 2, Arm2-vs-Arm1 pre-blend): "
          f"Δ={ambiguous_cell['mean_delta']:.6f} "
          f"CI=[{ambiguous_cell['ci_lower']:.6f},{ambiguous_cell['ci_upper']:.6f}] "
          f"excludes_zero={ambiguous_cell['excludes_zero']}")
    if not ambiguous_cell["excludes_zero"]:
        print("  -> CI INCLUDES zero: ambiguous, reported honestly as such (NOT corroboration-consistent, "
              "NOT contradicting -- ESTIMATION tier only, per sec 12.0).")
    else:
        print("  -> CI excludes zero (did NOT reproduce the design's own expected-ambiguous prediction "
              "for this cell -- reported as measured, not forced to match the design's expectation).")

    result = wrap_exploratory({
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 12.3.1/12.3.2 (H2 effective-rank/stable-rank reharvest)",
        "n_archived_retrofit_jsons_verified": n_archived,
        "mechanism_direction_pins": {
            "comparison_1_arm2_vs_arm1prime_post_blend": "negative",
            "comparison_2_arm2_vs_arm1_pre_blend": "negative",
            "comparison_3_arm2prime_vs_arm1doubleprime_post_blend": "positive",
            "comparison_4_arm2prime_vs_arm1_pre_blend_MISSING_UNTIL_NOW": "positive",
        },
        "grid": grid,
        "sanity_anchor_check": {
            "openr1_effective_rank": {
                "measured_mean_delta": anchor_eff_openr1["mean_delta"],
                "measured_ci": [anchor_eff_openr1["ci_lower"], anchor_eff_openr1["ci_upper"]],
                "attack_round_reference_mean_delta": -2.09,
                "attack_round_reference_ci": [-2.73, -1.46],
                "reproduced_within_0_02": anchor_eff_ok,
            },
            "wikitext_stable_rank": {
                "measured_mean_delta": anchor_stab_wikitext["mean_delta"],
                "measured_ci": [anchor_stab_wikitext["ci_lower"], anchor_stab_wikitext["ci_upper"]],
                "attack_round_reference_mean_delta": -1.75,
                "attack_round_reference_ci": [-2.26, -1.24],
                "reproduced_within_0_02": anchor_stab_ok,
            },
        },
        "ambiguous_cell_wikitext_effective_rank_comparison_2": {
            "mean_delta": ambiguous_cell["mean_delta"],
            "ci": [ambiguous_cell["ci_lower"], ambiguous_cell["ci_upper"]],
            "excludes_zero": ambiguous_cell["excludes_zero"],
            "note": "sec 12.3.1 flags this cell as expected-ambiguous; reported honestly regardless of outcome.",
        },
    })
    out_path = os.path.join(OUT_DIR, "build_fit_inputs_rankstats_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
