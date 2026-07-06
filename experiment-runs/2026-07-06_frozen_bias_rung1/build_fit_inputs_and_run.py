"""PHASE D driver: builds the 4 per-corpus-per-seed input JSONs fit_frozenbias_estimation.py
needs (arm2-post-blend, arm1prime-post-blend, arm2-kraw, arm1-kraw), runs the estimation fit for
the primary (Arm2 vs Arm1') + co-primary (kraw), THEN separately computes:
  - Arm2' (global) vs Arm1'' (global control) using the SAME pinned CI formula (sec 7.1a) --
    fit_frozenbias_estimation.py's own CLI does not cover this comparison, so this driver reuses
    its own derive_estimation function directly (not a re-implementation).
  - The mandatory Arm2-training-mediated-delta vs Arm2'-training-mediated-delta comparison
    (sec 7.1a's licensing condition).
  - The val-loss gate reading (Arm2/Arm2' final val loss vs Arm1's pinned tolerance).
  - The lambda mini-sweep cells (0.3, 0.8), descriptive only (n=1, no CI), sec 1.2a.
Run ON THE BOX.
"""
import json
import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bands_pinned_frozenbias import pinned_ci_half_width, _mean_std
from fit_frozenbias_estimation import derive_estimation, headline_verdict

_ANALYZE_PATH = "/home/nvidia/experiment-runs/2026-07-05_trackc_rung2/analyze_probe_wave2.py"
_spec = importlib.util.spec_from_file_location("analyze_probe_wave2_fit", _ANALYZE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pooled_subset = _mod.pooled_subset
span_frac = _mod.span_frac

RESULTS_DIR = "results/frozen_bias_lm"
CORPORA = ["openr1-mix-ext", "wikitext-mix-ext"]
CORPUS_SHORT = {"openr1-mix-ext": "openr1", "wikitext-mix-ext": "wikitext"}
SEEDS = [0, 1, 2]


def span_frac_from_retrofit_json(path, corpus):
    d = json.load(open(path))
    assert len(d["per_checkpoint"]) == 1, f"{path}: expected 1 checkpoint"
    ckpt_entry = list(d["per_checkpoint"].values())[0]
    pooled = pooled_subset(d, [corpus])
    assert pooled is not None and pooled["n_scored"] > 0, f"{path}: no scored episodes for {corpus}"
    cfg = ckpt_entry["config"]
    K = d["chunk_size"]
    head_dim = cfg["d_state"] // cfg.get("num_heads", 1)
    return span_frac(pooled["gram_deviation_mean"], K, head_dim), ckpt_entry


def extract_group(mode, run_prefix, corpus_trained_check=True):
    """mode: 'arm1prime'|'arm1double'|'kraw'. run_prefix: e.g. 'arm2_' or 'arm2p_' or 'arm1_'."""
    out = {}
    for corpus in CORPORA:
        short = CORPUS_SHORT[corpus]
        vals = []
        for seed in SEEDS:
            path = os.path.join(RESULTS_DIR, f"retrofit_{mode}_{run_prefix}{short}_s{seed}.json")
            sf, ckpt_entry = span_frac_from_retrofit_json(path, corpus)
            if corpus_trained_check:
                assert ckpt_entry["corpus_trained_on"] == corpus, (
                    f"{path}: corpus_trained_on={ckpt_entry['corpus_trained_on']!r} != {corpus!r}")
            vals.append(sf)
        out[corpus] = vals
    return out


if __name__ == "__main__":
    # ---- Arm 2 (per_token) post-blend + kraw ----
    arm2_post = extract_group("arm1prime", "arm2_")
    arm2_kraw = extract_group("kraw", "arm2_")

    # ---- Arm 1' / Arm 1 kraw (from Phase A, already extracted once, re-extract here for a
    # self-contained fit-input build rather than depending on build_bands_pin.py's own run) ----
    arm1prime_post = extract_group("arm1prime", "arm1_")
    arm1_kraw = extract_group("kraw", "arm1_")

    # Write the 4 input files fit_frozenbias_estimation.py's CLI expects.
    paths = {}
    for name, data in [("arm2_post_blend", arm2_post), ("arm1prime_post_blend", arm1prime_post),
                        ("arm2_kraw", arm2_kraw), ("arm1_kraw", arm1_kraw)]:
        p = os.path.join(RESULTS_DIR, f"fitinput_{name}.json")
        json.dump(data, open(p, "w"), indent=2)
        paths[name] = p

    print("=== Arm 2 post-blend (arm1prime mode, lam=0.58) span_frac ===")
    print(json.dumps(arm2_post, indent=2))
    print("=== Arm 1' post-blend span_frac (reference) ===")
    print(json.dumps(arm1prime_post, indent=2))
    print("=== Arm 2 kraw (pre-blend) span_frac ===")
    print(json.dumps(arm2_kraw, indent=2))
    print("=== Arm 1 kraw (pre-blend) span_frac (reference) ===")
    print(json.dumps(arm1_kraw, indent=2))

    # ---- Run the fit exactly as fit_frozenbias_estimation.py's own main() would ----
    corpora = sorted(set(arm2_post) & set(arm1prime_post) & set(arm2_kraw) & set(arm1_kraw))
    post_blend_results, pre_blend_results = {}, {}
    for c in corpora:
        deltas_post = [a2 - a1p for a2, a1p in zip(arm2_post[c], arm1prime_post[c])]
        deltas_pre = [a2 - a1 for a2, a1 in zip(arm2_kraw[c], arm1_kraw[c])]
        post_blend_results[c] = derive_estimation(deltas_post, c)
        pre_blend_results[c] = derive_estimation(deltas_pre, c)
    hv = headline_verdict(post_blend_results, pre_blend_results)
    primary_result = {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 7.1-real.1 (pinned CI, ESTIMATION mode)",
        "post_blend_primary": post_blend_results,
        "pre_blend_co_primary": pre_blend_results,
        "headline": hv,
    }
    print("\n=== PRIMARY + CO-PRIMARY ESTIMATION RESULT (Arm2 vs Arm1') ===")
    print(json.dumps(primary_result, indent=2))
    json.dump(primary_result, open(os.path.join(RESULTS_DIR, "estimation_primary_arm2_vs_arm1prime.json"), "w"), indent=2)

    # ---- Arm 2' (global) vs Arm 1'' (global control), sec 7.1a, SAME pinned formula ----
    arm2p_post = extract_group("arm1double", "arm2p_")
    arm1double_post = extract_group("arm1double", "arm1_")
    arm2p_results = {}
    for c in corpora:
        deltas = [a2p - a1pp for a2p, a1pp in zip(arm2p_post[c], arm1double_post[c])]
        arm2p_results[c] = derive_estimation(deltas, c)
    print("\n=== Arm 2' (global) vs Arm 1'' (global control) ESTIMATION ===")
    print(json.dumps(arm2p_results, indent=2))

    # ---- sec 7.1a licensing condition: Arm2's training-mediated delta must be MORE NEGATIVE
    # than Arm2's own delta by t*SE, on both corpora -- i.e. compare the two delta point estimates
    # directly (both already computed above as post_blend_results[c]['mean_delta'] and
    # arm2p_results[c]['mean_delta']). ----
    licensing = {}
    for c in corpora:
        arm2_delta = post_blend_results[c]["mean_delta"]
        arm2p_delta = arm2p_results[c]["mean_delta"]
        arm2_more_negative = arm2_delta < arm2p_delta
        licensing[c] = {
            "arm2_training_mediated_delta": arm2_delta,
            "arm2prime_training_mediated_delta": arm2p_delta,
            "arm2_more_negative_than_arm2prime": arm2_more_negative,
        }
    print("\n=== sec 7.1a licensing condition (Arm2 delta vs Arm2' delta) ===")
    print(json.dumps(licensing, indent=2))

    # ---- val-loss gate reading ----
    bands = json.load(open(os.path.join(RESULTS_DIR, "BANDS_PINNED-FrozenBias.json")))
    val_loss_gate = {}
    for c in corpora:
        short = CORPUS_SHORT[c]
        arm2_losses = []
        arm2p_losses = []
        for seed in SEEDS:
            d2 = json.load(open(os.path.join(RESULTS_DIR, f"frozenbias_lm_per_token_lam0p58_{c}_dm256_ds64_L2_s{seed}.json")))
            arm2_losses.append(d2["checkpoints"][-1]["val_loss"][c])
            d2p = json.load(open(os.path.join(RESULTS_DIR, f"frozenbias_lm_global_lam0p58_{c}_dm256_ds64_L2_s{seed}.json")))
            arm2p_losses.append(d2p["checkpoints"][-1]["val_loss"][c])
        tol = bands["arm1_val_loss_tolerance"][c]
        arm2_mean = sum(arm2_losses) / len(arm2_losses)
        arm2p_mean = sum(arm2p_losses) / len(arm2p_losses)
        val_loss_gate[c] = {
            "arm1_mean": tol["mean"], "pass_ceiling": tol["pass_ceiling"],
            "arm2_mean": arm2_mean, "arm2_per_seed": arm2_losses, "arm2_passes_gate": arm2_mean <= tol["pass_ceiling"],
            "arm2prime_mean": arm2p_mean, "arm2prime_per_seed": arm2p_losses,
            "arm2prime_passes_gate": arm2p_mean <= tol["pass_ceiling"],
        }
    print("\n=== val-loss gate (sec 7.2) ===")
    print(json.dumps(val_loss_gate, indent=2))

    # ---- lambda mini-sweep, descriptive only (n=1, openr1-mix-ext, seed 0) ----
    lam_sweep = {}
    for lam_tag, lam_val in [("lam030", 0.3), ("lam058", 0.58), ("lam080", 0.8)]:
        if lam_tag == "lam058":
            # already have this: arm2_post["openr1-mix-ext"][0] vs arm1prime_post["openr1-mix-ext"][0]
            arm2_sf = arm2_post["openr1-mix-ext"][0]
            arm1p_sf = arm1prime_post["openr1-mix-ext"][0]
        else:
            arm2_path = os.path.join(RESULTS_DIR, f"retrofit_arm1prime_{lam_tag}_openr1_s0.json")
            arm2_sf, _ = span_frac_from_retrofit_json(arm2_path, "openr1-mix-ext")
            # Arm 1' control AT THIS SAME LAMBDA needs its own retrofit pass on the ARM-1 (seed 0,
            # openr1) checkpoint -- was this run? Check for a matching file; if absent, disclose.
            arm1p_path = os.path.join(RESULTS_DIR, f"retrofit_arm1prime_lam_{lam_tag}_arm1_openr1_s0.json")
            arm1p_sf = None
            if os.path.exists(arm1p_path):
                arm1p_sf, _ = span_frac_from_retrofit_json(arm1p_path, "openr1-mix-ext")
        delta = (arm2_sf - arm1p_sf) if arm1p_sf is not None else None
        lam_sweep[lam_tag] = {
            "lam": lam_val, "arm2_span_frac_seed0_openr1": arm2_sf,
            "arm1prime_control_span_frac_seed0_openr1": arm1p_sf,
            "delta_n1_descriptive": delta,
        }
    print("\n=== lambda mini-sweep (descriptive, n=1, openr1-mix-ext seed0) ===")
    print(json.dumps(lam_sweep, indent=2))

    full_report = {
        "primary_and_coprimary": primary_result,
        "arm2prime_vs_arm1double": arm2p_results,
        "sec7_1a_licensing": licensing,
        "val_loss_gate": val_loss_gate,
        "lambda_mini_sweep": lam_sweep,
    }
    json.dump(full_report, open(os.path.join(RESULTS_DIR, "PHASE_D_FULL_REPORT.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(RESULTS_DIR, 'PHASE_D_FULL_REPORT.json')}")
