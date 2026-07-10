"""S2.32 RE-METRIC (coordinator tiebreak S2.31a) -- box-side recompute.

Runs on the box, CPU-only (CUDA_VISIBLE_DEVICES="" set by the caller / this
script never touches torch.cuda), against the ALREADY-TRAINED sweep
checkpoints in stage2_results/ -- no training, eval-only forward passes,
mirroring the "~0.0 GPU-h free eval-only forwards" precedent already used
twice in this design (S2.6/S2.7, S2.8 item 2(e)'s D=64 extension).

Does exactly two things neither committed JSON already carries:

  1. CEILING CROSSCHECK AT D=D_TRAIN_MAX=8 -- `m_d0_profile` only persisted
     `recovered_frac_90`/`mean_cos` (the primary lens), never the
     `crosscheck_*` fields `readout.degauge_and_score` ALSO computes on every
     call. This re-invokes the EXACT pinned pipeline
     (`stage2_task.evaluate_composer_at_depth`, unmodified, imported not
     reimplemented) at D=8 with the SAME seed convention
     `m_d0_convergence_profile` already used (`seed=cell_seed*1000+D`) and
     keeps the crosscheck fields the production code silently dropped. The
     primary `recovered_frac_90` this reproduces is asserted BIT-IDENTICAL
     against the committed `m_d0_profile` D=8 row -- proof this harness
     reproduces the production pipeline exactly, not a new metric.

  2. SHUFFLED-TARGET TEETH CONTROL (S2.31a's pinned falsifier) -- on 3
     converged checkpoints (one per group family: A6, S5, and one of
     S3/S4/A5), re-run the SAME D=64 pipeline but with `rho_list` permuted
     across items BEFORE the fit/eval split (breaking state<->target
     correspondence; same shapes/marginals -- a pure relabeling, nothing
     resampled). The REAL (unshuffled) D=64 crosscheck is also recomputed
     alongside and asserted bit-identical against the committed
     `D_test_results` D=64 row, for the same reproduction-proof reason as (1).

No new metric is written -- `readout.degauge_and_score` /
`stage2_task.evaluate_composer_at_depth` /
`stage2_run.build_cell_composer`/`load_cell_composer` are imported and
called exactly as production code calls them; the only new code is the
orchestration loop and the target-shuffle permutation.
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # belt-and-suspenders: CPU-only, never touches GPU 6/7

import numpy as np
import torch

torch.cuda.is_available = lambda: False  # hard local override -- this process must never see a GPU

import readout
import stage2_run as sr
import stage2_task as st
from groups import D_MIN

RESULTS_DIR = "stage2_results"
DEVICE = "cpu"
D_TRAIN_MAX = st.D_TRAIN_MAX  # 8
FAR_DEPTH = 64

CONTROL_CELL_IDS = [
    "A6__arm3_beta02__nh4__seed0",
    "S5__arm3_beta02__nh4__seed0",
    "S4__arm3_beta02__nh2__seed2",
]

SHUFFLE_SEED_OFFSET = 99_999  # a fresh offset, distinct from the pipeline's own 10_000/20_000 conventions


def load_all_cell_jsons(results_dir: str) -> dict:
    out = {}
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json") or fn == "stage2_harvest_report.json":
            continue
        with open(os.path.join(results_dir, fn)) as f:
            d = json.load(f)
        out[d["cell_id"]] = d
    return out


def ceiling_crosscheck_at_d8(cell: dict) -> dict:
    """Item (1): re-run evaluate_composer_at_depth at D=D_TRAIN_MAX, extract
    the crosscheck fields the production m_d0_convergence_profile discarded,
    and assert the primary field reproduces bit-identically."""
    composer = sr.load_cell_composer(cell, RESULTS_DIR, device=DEVICE)
    seed_arg = cell["seed"] * 1000 + D_TRAIN_MAX  # m_d0_convergence_profile's own convention
    s = st.evaluate_composer_at_depth(composer, cell["group"], D_TRAIN_MAX, seed=seed_arg, device=DEVICE)

    committed_row = next(r for r in cell["m_d0_profile"] if r["D"] == D_TRAIN_MAX)
    reproduced_primary = s["recovered_frac_90"]
    committed_primary = committed_row["recovered_frac_90"]
    bit_identical = (committed_primary is not None) and abs(reproduced_primary - committed_primary) < 1e-9

    return dict(
        cell_id=cell["cell_id"],
        reproduced_primary_rf90_d8=reproduced_primary,
        committed_primary_rf90_d8=committed_primary,
        reproduction_bit_identical=bit_identical,
        crosscheck_rf90_d8=s["crosscheck_recovered_frac_90"],
        crosscheck_mean_cos_d8=s["crosscheck_mean_cos"],
    )


def evaluate_composer_at_depth_shuffled(composer, name: str, D: int, seed: int, device="cpu") -> dict:
    """S2.31a's pinned teeth: the EXACT body of
    `stage2_task.evaluate_composer_at_depth`, with ONE inserted step --
    `rho_list` is permuted across items (a pure relabeling, same shapes/
    marginals, no resampling) AFTER the coverage-guarded draw and BEFORE the
    fit/eval split -- breaking state<->target correspondence. Every other
    line is byte-identical to the production function (imported helpers,
    same seeds/offsets) so the REAL (unpermuted) path below can be
    cross-checked bit-identical against the committed JSON before the
    shuffled result is trusted."""
    d_min = D_MIN[name]
    cov_log = st.check_depth_coverage_with_retry(name, seed, D, n_words=st.gt.N_EVAL_WORDS)
    idx_list, rho_list = st.gt.sample_eval_words(name, cov_log["final_seed"], st.gt.N_EVAL_WORDS, L_lo=D, L_hi=D)

    idx_batch = torch.tensor(np.stack(idx_list), dtype=torch.long, device=device)
    composer.eval()
    with torch.no_grad():
        Z = composer(idx_batch)
    Z_np = Z.detach().cpu().numpy().astype(np.float64)

    U = readout.entity_subspace_from_words(Z_np, d_min)
    A_words = np.stack([readout.restrict(Z_np[i], U) for i in range(len(rho_list))])

    fit_idx, eval_idx, split_log = readout._split_with_diversity_retry(
        name, rho_list, seed=cov_log["final_seed"] + 20_000)

    # --- REAL (unpermuted) path, for the bit-identical reproduction check ---
    A_fit_real = [A_words[i] for i in fit_idx]
    A_eval_real = [A_words[i] for i in eval_idx]
    rho_fit_real = [rho_list[i] for i in fit_idx]
    rho_eval_real = [rho_list[i] for i in eval_idx]
    real_scores = readout.degauge_and_score(A_fit_real, A_eval_real, rho_fit_real, rho_eval_real, d_min)

    # --- SHUFFLED path: permute rho_list's ITEM ASSIGNMENT across the whole
    # N=50 sample (fit+eval together) before re-slicing by the SAME fit_idx/
    # eval_idx index sets -- same marginal distribution of targets, same
    # split sizes/diversity, correspondence to A_words broken. ---
    perm_seed = cov_log["final_seed"] + SHUFFLE_SEED_OFFSET
    perm = np.random.default_rng(perm_seed).permutation(len(rho_list))
    n_fixed_points = int(np.sum(perm == np.arange(len(rho_list))))
    rho_list_shuffled = [rho_list[perm[i]] for i in range(len(rho_list))]
    A_fit_sh = [A_words[i] for i in fit_idx]
    A_eval_sh = [A_words[i] for i in eval_idx]
    rho_fit_sh = [rho_list_shuffled[i] for i in fit_idx]
    rho_eval_sh = [rho_list_shuffled[i] for i in eval_idx]
    shuffled_scores = readout.degauge_and_score(A_fit_sh, A_eval_sh, rho_fit_sh, rho_eval_sh, d_min)

    return dict(
        real_recovered_frac_90=real_scores["recovered_frac_90"],
        real_mean_cos=real_scores["mean_cos"],
        real_crosscheck_recovered_frac_90=real_scores["crosscheck_recovered_frac_90"],
        real_crosscheck_mean_cos=real_scores["crosscheck_mean_cos"],
        shuffled_recovered_frac_90=shuffled_scores["recovered_frac_90"],
        shuffled_mean_cos=shuffled_scores["mean_cos"],
        shuffled_crosscheck_recovered_frac_90=shuffled_scores["crosscheck_recovered_frac_90"],
        shuffled_crosscheck_mean_cos=shuffled_scores["mean_cos"],
        perm_seed=perm_seed,
        n_fixed_points_in_permutation=n_fixed_points,
        n_items=len(rho_list),
    )


def shuffled_control_for_cell(cell: dict) -> dict:
    composer = sr.load_cell_composer(cell, RESULTS_DIR, device=DEVICE)
    seed_arg = cell["seed"] * 1000 + FAR_DEPTH  # d_test_grid_eval's own convention
    r = evaluate_composer_at_depth_shuffled(composer, cell["group"], FAR_DEPTH, seed=seed_arg, device=DEVICE)

    committed_row = next(row for row in cell["D_test_results"] if row["D"] == FAR_DEPTH)
    committed_primary = committed_row["recovered_frac_90"]
    committed_xcheck = committed_row["crosscheck_recovered_frac_90"]
    bit_identical_primary = abs(r["real_recovered_frac_90"] - committed_primary) < 1e-9
    bit_identical_xcheck = abs(r["real_crosscheck_recovered_frac_90"] - committed_xcheck) < 1e-9

    out = dict(cell_id=cell["cell_id"], committed_primary_rf90_d64=committed_primary,
               committed_crosscheck_rf90_d64=committed_xcheck,
               reproduction_bit_identical_primary=bit_identical_primary,
               reproduction_bit_identical_crosscheck=bit_identical_xcheck)
    out.update(r)
    return out


def main():
    cells = load_all_cell_jsons(RESULTS_DIR)
    print(f"loaded {len(cells)} cell JSONs from {RESULTS_DIR}")

    # --- Item (1): D=8 ceiling crosscheck, ALL 62 cells ---
    ceiling_results = {}
    for i, (cid, cell) in enumerate(sorted(cells.items())):
        try:
            r = ceiling_crosscheck_at_d8(cell)
        except Exception as e:  # noqa: BLE001 -- report, don't crash the whole sweep on one cell
            r = dict(cell_id=cid, error=f"{type(e).__name__}: {e}")
        ceiling_results[cid] = r
        if (i + 1) % 10 == 0:
            print(f"  ceiling recompute: {i+1}/{len(cells)}")

    n_ok = sum(1 for r in ceiling_results.values() if "error" not in r)
    n_bit_identical = sum(1 for r in ceiling_results.values() if r.get("reproduction_bit_identical"))
    print(f"ceiling D=8 crosscheck: {n_ok}/{len(cells)} computed OK, "
          f"{n_bit_identical}/{n_ok} bit-identical primary reproduction")

    # --- Item (2): shuffled-target teeth control ---
    teeth_results = {}
    for cid in CONTROL_CELL_IDS:
        cell = cells[cid]
        print(f"teeth control: {cid}")
        r = shuffled_control_for_cell(cell)
        teeth_results[cid] = r
        print(f"  real xcheck rf90@64={r['real_crosscheck_recovered_frac_90']:.4f}  "
              f"(committed={r['committed_crosscheck_rf90_d64']:.4f}, "
              f"bit_identical={r['reproduction_bit_identical_crosscheck']})  "
              f"SHUFFLED xcheck rf90@64={r['shuffled_crosscheck_recovered_frac_90']:.4f}")

    out = dict(ceiling_crosscheck_d8=ceiling_results, teeth_shuffled_control=teeth_results)
    with open("remetric_2p32_box_output.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote remetric_2p32_box_output.json")


if __name__ == "__main__":
    main()
