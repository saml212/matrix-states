"""dose_dial_verify.py -- Rev 14.3 FATAL-fix verification, CPU-only, no fla.

WHY THIS SCRIPT EXISTS (the fix for the failure class that caused the FATAL):
KEY_ANCHORING_DESIGN.md sec 14.1 (Rev 14.2) asserted diffuse-arm
(subspace_rank=d_state=128) calibration numbers (achieved max|cos| =
0.130084 / 0.284077 / 0.400012, stable to <0.0004 across 5 subspace seeds,
Gram spectrum [1.701 .. 1.673]) WITHOUT ever running the code that would
produce them. Round-3 verify re-implemented build_dose_table/calibrate_dose
from the design's own pseudocode and found the diffuse dial is mathematically
degenerate at subspace_rank=d_state: QR of a square (d,d) Gaussian yields a
FULL orthogonal Q (Q @ Q.T == I to float64 precision), so `proj = table @ Q @
Q.T == table` at every row, and the blend `(1-t)*table + t*normalize(proj)`
collapses to `table` itself (already unit-norm) for every t -- the achieved
dose is EXACTLY 0.0 at every t, not the claimed nonzero values. This script
is the committed, re-runnable artifact that (a) proves the degeneracy
directly, (b) scans subspace_rank to find the most-diffuse rank that still
reaches the required doses, and (c) recalibrates+re-measures BOTH structures
honestly, so every number the design doc cites has a script+JSON behind it.

Run: matrix-thinking/deltanet_rd/.verify_venv/bin/python dose_dial_verify.py
Writes: matrix-thinking/deltanet_rd/dose_dial_verify_results.json
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

from key_anchoring import frame_potential_init, raw_table_conditioning, ANCHOR_INIT_SEED

N_ENTITIES = 107
D_STATE = 128
BASE_SEED = ANCHOR_INIT_SEED  # 20260705, per sec 14.0's registered construction seed
DOSE_TARGETS = [0.130, 0.284, 0.400]
RANK_SCAN = [8, 16, 32, 48, 64, 96, 128]  # the mandatory, registered scan grid (task brief's own list)
RANK_SCAN_INFORMATIONAL = [107, 120]  # NOT part of the mandatory grid -- extra points cited in
                                       # the design doc's prose to show the monotone-decreasing
                                       # ceiling continues smoothly toward r=d; included here so
                                       # EVERY number the doc cites is reproducible from this
                                       # committed script + its JSON output, not just the mandatory
                                       # grid (a gap an independent verifier flagged: these two
                                       # points were cited in prose but absent from the first
                                       # committed JSON).
CEILING_TARGET = 0.42  # "as diffuse as achievable while still reaching the 0.40 dose"
N_SUBSPACE_SEEDS = 5
SUBSPACE_SEEDS = list(range(N_SUBSPACE_SEEDS))  # 0..4, distinct from BASE_SEED/ANCHOR_INIT_SEED
TOL = 1e-4


# ---------------------------------------------------------------------------
# The dial itself -- sec 14.1's own spec, transcribed verbatim (this is the
# function the design doc's pseudocode describes; reproduced here rather than
# imported, since it is NOT committed to key_anchoring.py yet per sec 14.1b
# item 1's own build-task registration -- this script is pre-build
# verification, the exact case that caught the degeneracy).
# ---------------------------------------------------------------------------

def build_dose_table(healthy_table: torch.Tensor, t: float, subspace_rank: int,
                      subspace_seed: int) -> torch.Tensor:
    """sec 14.1's spec, verbatim. t=0 -> unchanged. t=1 -> every row replaced
    by its unit-normalized projection onto a random subspace of the given
    rank. subspace_rank=d -> Q is a full (d,d) orthogonal matrix, and
    Q @ Q.T == I EXACTLY (up to float64 rounding) -- this is the degeneracy
    this script exists to demonstrate, not merely assert."""
    n, d = healthy_table.shape
    g = torch.Generator().manual_seed(subspace_seed)
    Q, _ = torch.linalg.qr(torch.randn(d, subspace_rank, generator=g, dtype=torch.float64))
    proj = healthy_table.double() @ Q @ Q.transpose(0, 1)
    proj_unit = F.normalize(proj, dim=-1)
    blended = (1.0 - t) * healthy_table.double() + t * proj_unit
    return F.normalize(blended, dim=-1).float()


def achieved_dose(healthy_table: torch.Tensor, t: float, subspace_rank: int,
                   subspace_seed: int) -> float:
    dosed = build_dose_table(healthy_table, t, subspace_rank, subspace_seed)
    return raw_table_conditioning(dosed)["max_abs_cos"]


def calibrate_dose(healthy_table: torch.Tensor, target_dose: float, subspace_rank: int,
                    subspace_seed: int, tol: float = TOL, max_iter: int = 80):
    """Bisection on t in [0,1] against achieved max|cos|. Returns
    (t, achieved, n_iters, converged). If the achieved dose at t=1 (the
    dial's own maximum reach) is still below target_dose, calibration cannot
    converge -- this is reported explicitly (converged=False, ceiling=dose
    at t=1), never silently clamped or fudged."""
    lo, hi = 0.0, 1.0
    dose_lo = achieved_dose(healthy_table, lo, subspace_rank, subspace_seed)
    dose_hi = achieved_dose(healthy_table, hi, subspace_rank, subspace_seed)
    if dose_hi < target_dose - tol:
        # Cannot reach target even at t=1 -- the dial's ceiling for this rank.
        return {"t": 1.0, "achieved": dose_hi, "n_iters": 0, "converged": False,
                "ceiling_at_t1": dose_hi}
    n_iters = 0
    t = hi
    achieved = dose_hi
    for i in range(max_iter):
        n_iters = i + 1
        t = 0.5 * (lo + hi)
        achieved = achieved_dose(healthy_table, t, subspace_rank, subspace_seed)
        if abs(achieved - target_dose) <= tol:
            break
        # achieved_dose(t) is monotone non-decreasing in t for a fixed
        # subspace (more blend toward a fixed direction set cannot reduce
        # coherence) -- verified empirically below (monotonicity check), so
        # standard bisection applies.
        if achieved < target_dose:
            lo = t
        else:
            hi = t
    converged = abs(achieved - target_dose) <= tol
    return {"t": t, "achieved": achieved, "n_iters": n_iters, "converged": converged,
            "ceiling_at_t1": dose_hi}


def gram_top8_singular_values(table: torch.Tensor) -> list:
    """Top-8 singular values of the table itself (n,d) -- the same
    'Gram-spectrum' quantity sec 14.1 cites (singular values of the raw
    table, descending)."""
    s = torch.linalg.svdvals(table.float())
    return [round(v, 3) for v in s[:8].tolist()]


def monotonicity_check(healthy_table: torch.Tensor, subspace_rank: int, subspace_seed: int):
    """Confirms achieved_dose(t) is monotone non-decreasing in t for a fixed
    (subspace_rank, subspace_seed) -- the assumption bisection in
    calibrate_dose relies on. Sampled at 11 points across [0,1]."""
    ts = [i / 10 for i in range(11)]
    doses = [achieved_dose(healthy_table, t, subspace_rank, subspace_seed) for t in ts]
    is_monotone = all(doses[i] <= doses[i + 1] + 1e-9 for i in range(len(doses) - 1))
    return {"ts": ts, "doses": [round(d, 6) for d in doses], "is_monotone": is_monotone}


def main():
    healthy = frame_potential_init(N_ENTITIES, D_STATE, seed=BASE_SEED)
    base_cond = raw_table_conditioning(healthy)
    print(f"Base table (n={N_ENTITIES}, d={D_STATE}, seed={BASE_SEED}): "
          f"max|cos|={base_cond['max_abs_cos']:.6f}, sigma_ratio={base_cond['sigma_ratio']:.6f}")

    results = {
        "meta": {
            "n_entities": N_ENTITIES, "d_state": D_STATE, "base_seed": BASE_SEED,
            "dose_targets": DOSE_TARGETS, "rank_scan": RANK_SCAN,
            "ceiling_target": CEILING_TARGET, "n_subspace_seeds": N_SUBSPACE_SEEDS,
            "tol": TOL,
            "base_table_max_abs_cos": base_cond["max_abs_cos"],
            "base_table_sigma_ratio": base_cond["sigma_ratio"],
        },
    }

    # -------------------------------------------------------------------
    # STEP 0: Direct proof of the degeneracy at subspace_rank=d_state=128.
    # -------------------------------------------------------------------
    g = torch.Generator().manual_seed(BASE_SEED)
    Q_full, _ = torch.linalg.qr(torch.randn(D_STATE, D_STATE, generator=g, dtype=torch.float64))
    proj_op = Q_full @ Q_full.transpose(0, 1)
    max_dev_from_identity = (proj_op - torch.eye(D_STATE, dtype=torch.float64)).abs().max().item()
    dose_at_rank128_t1 = achieved_dose(healthy, 1.0, D_STATE, BASE_SEED)
    results["degeneracy_proof"] = {
        "claim": "QR(randn(d,d)) gives full orthogonal Q; Q@Q.T == I; "
                 "proj = table @ Q @ Q.T == table; blend is a no-op at every t.",
        "max_abs_deviation_QQt_from_identity_rank128": max_dev_from_identity,
        "achieved_dose_at_t1_rank128": dose_at_rank128_t1,
        "note": "achieved_dose_at_t1_rank128 should equal base_table_max_abs_cos "
                "(i.e. 0 injected dose) if the degeneracy is real.",
    }
    print(f"\n[degeneracy proof] max|Q@Q.T - I| at rank=128: {max_dev_from_identity:.3e}")
    print(f"[degeneracy proof] achieved dose at t=1, rank=128: {dose_at_rank128_t1:.6f} "
          f"(base table dose: {base_cond['max_abs_cos']:.6f})")

    # -------------------------------------------------------------------
    # STEP 1: rank-scan -- for each candidate subspace_rank, find the
    # achieved-dose CEILING (dose at t=1) using the registered subspace seed.
    # -------------------------------------------------------------------
    rank_scan_table = []
    for rank in RANK_SCAN:
        ceiling = achieved_dose(healthy, 1.0, rank, BASE_SEED)
        meets_ceiling_target = ceiling >= CEILING_TARGET
        rank_scan_table.append({
            "subspace_rank": rank,
            "achieved_dose_ceiling_at_t1": ceiling,
            "meets_ceiling_target_0.42": meets_ceiling_target,
        })
        print(f"[rank scan] rank={rank:>3d}  ceiling(dose @ t=1) = {ceiling:.6f}  "
              f"meets >=0.42: {meets_ceiling_target}")
    results["rank_scan_table"] = rank_scan_table

    # Informational-only points (NOT part of the mandatory scan grid, and NOT
    # used to select chosen_rank below) -- cited in the design doc's prose to
    # show the monotone-decreasing ceiling pattern continues smoothly toward
    # r=d; computed and recorded here so they are reproducible from this
    # committed artifact, not merely asserted from an ad hoc re-run.
    rank_scan_informational = []
    for rank in RANK_SCAN_INFORMATIONAL:
        ceiling = achieved_dose(healthy, 1.0, rank, BASE_SEED)
        rank_scan_informational.append({
            "subspace_rank": rank,
            "achieved_dose_ceiling_at_t1": ceiling,
        })
        print(f"[rank scan, informational only] rank={rank:>3d}  "
              f"ceiling(dose @ t=1) = {ceiling:.6f}")
    results["rank_scan_informational"] = rank_scan_informational

    # Pick MAXIMUM rank whose ceiling >= 0.42 (as diffuse as achievable while
    # still reaching the 0.40 dose target).
    eligible = [r for r in rank_scan_table if r["meets_ceiling_target_0.42"]]
    if not eligible:
        raise RuntimeError("No scanned rank reaches the 0.42 ceiling target -- "
                            "the dose grid as specified is unachievable at ANY "
                            "scanned rank; this would itself be a reportable finding.")
    chosen_rank = max(r["subspace_rank"] for r in eligible)
    chosen_ceiling = next(r["achieved_dose_ceiling_at_t1"] for r in eligible
                           if r["subspace_rank"] == chosen_rank)
    results["chosen_diffuse_rank"] = {
        "subspace_rank": chosen_rank,
        "ceiling_at_t1": chosen_ceiling,
        "selection_rule": "MAXIMUM rank among scanned ranks with ceiling >= 0.42",
    }
    print(f"\n[chosen] diffuse subspace_rank = {chosen_rank} "
          f"(ceiling={chosen_ceiling:.6f})")

    # -------------------------------------------------------------------
    # STEP 2: monotonicity sanity check for both structures at the
    # registered subspace seed (justifies bisection).
    # -------------------------------------------------------------------
    results["monotonicity_check"] = {
        "rank4": monotonicity_check(healthy, 4, BASE_SEED),
        f"rank{chosen_rank}_diffuse": monotonicity_check(healthy, chosen_rank, BASE_SEED),
    }

    # -------------------------------------------------------------------
    # STEP 3: calibrate t for BOTH structures at all three dose targets,
    # 5 subspace seeds each.
    # -------------------------------------------------------------------
    structures = {"rank4": 4, f"diffuse_rank{chosen_rank}": chosen_rank}
    calibration = {}
    for struct_name, rank in structures.items():
        calibration[struct_name] = {"subspace_rank": rank, "doses": {}}
        for target in DOSE_TARGETS:
            per_seed = []
            for sseed in SUBSPACE_SEEDS:
                cal = calibrate_dose(healthy, target, rank, sseed)
                per_seed.append({"subspace_seed": sseed, **cal})
                status = "OK" if cal["converged"] else "CEILING-LIMITED"
                print(f"[calibrate] struct={struct_name:<16s} target={target:.3f} "
                      f"seed={sseed} -> t={cal['t']:.5f} achieved={cal['achieved']:.6f} "
                      f"[{status}]")
            achieved_vals = [p["achieved"] for p in per_seed]
            calibration[struct_name]["doses"][str(target)] = {
                "per_seed": per_seed,
                "achieved_mean": sum(achieved_vals) / len(achieved_vals),
                "achieved_min": min(achieved_vals),
                "achieved_max": max(achieved_vals),
                "achieved_spread": max(achieved_vals) - min(achieved_vals),
                "all_converged": all(p["converged"] for p in per_seed),
            }
    results["calibration"] = calibration

    # -------------------------------------------------------------------
    # STEP 4: Gram spectra (top-8 singular values) for rank-4 vs
    # chosen-diffuse-rank at dose 0.284, vs the d=64 organic table's
    # spectrum, at the registered subspace seed (BASE_SEED).
    # -------------------------------------------------------------------
    target_for_spectrum = 0.284
    t_rank4 = calibrate_dose(healthy, target_for_spectrum, 4, BASE_SEED)["t"]
    t_diffuse = calibrate_dose(healthy, target_for_spectrum, chosen_rank, BASE_SEED)["t"]
    table_rank4 = build_dose_table(healthy, t_rank4, 4, BASE_SEED)
    table_diffuse = build_dose_table(healthy, t_diffuse, chosen_rank, BASE_SEED)

    # d=64 organic table (n=107, d=64, same base construction seed) --
    # the "organic tight-frame" comparator per sec 14.1's own d=64 line.
    organic_d64 = frame_potential_init(N_ENTITIES, 64, seed=BASE_SEED)
    organic_d64_cond = raw_table_conditioning(organic_d64)

    results["gram_spectrum_comparison"] = {
        "dose_target_for_comparison": target_for_spectrum,
        "rank4": {
            "t": t_rank4,
            "achieved_max_abs_cos": raw_table_conditioning(table_rank4)["max_abs_cos"],
            "top8_singular_values": gram_top8_singular_values(table_rank4),
        },
        f"diffuse_rank{chosen_rank}": {
            "t": t_diffuse,
            "achieved_max_abs_cos": raw_table_conditioning(table_diffuse)["max_abs_cos"],
            "top8_singular_values": gram_top8_singular_values(table_diffuse),
        },
        "organic_d64": {
            "max_abs_cos": organic_d64_cond["max_abs_cos"],
            "top8_singular_values": gram_top8_singular_values(organic_d64),
        },
    }
    print("\n[gram spectrum @ dose=0.284]")
    print(f"  rank4:                top8 = {results['gram_spectrum_comparison']['rank4']['top8_singular_values']}")
    print(f"  diffuse_rank{chosen_rank}: top8 = {results['gram_spectrum_comparison'][f'diffuse_rank{chosen_rank}']['top8_singular_values']}")
    print(f"  organic_d64:           top8 = {results['gram_spectrum_comparison']['organic_d64']['top8_singular_values']}")

    # -------------------------------------------------------------------
    # STEP 5: verdict -- is the chosen-diffuse-rank structure meaningfully
    # different from rank-4 in spectrum shape, and how close to organic d=64?
    # -------------------------------------------------------------------
    rank4_spec = results["gram_spectrum_comparison"]["rank4"]["top8_singular_values"]
    diffuse_spec = results["gram_spectrum_comparison"][f"diffuse_rank{chosen_rank}"]["top8_singular_values"]
    organic_spec = results["gram_spectrum_comparison"]["organic_d64"]["top8_singular_values"]

    def spread(spec):
        return max(spec) - min(spec)

    rank4_spread = spread(rank4_spec)
    diffuse_spread = spread(diffuse_spec)
    organic_spread = spread(organic_spec)

    results["verdict"] = {
        "rank4_spectrum_spread": rank4_spread,
        "diffuse_spectrum_spread": diffuse_spread,
        "organic_d64_spectrum_spread": organic_spread,
        "diffuse_meaningfully_flatter_than_rank4": diffuse_spread < 0.5 * rank4_spread,
        "diffuse_gap_to_organic_flat": diffuse_spread - organic_spread,
    }
    print(f"\n[verdict] rank4 spread={rank4_spread:.3f}  "
          f"diffuse(rank{chosen_rank}) spread={diffuse_spread:.3f}  "
          f"organic(d=64) spread={organic_spread:.3f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dose_dial_verify_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
