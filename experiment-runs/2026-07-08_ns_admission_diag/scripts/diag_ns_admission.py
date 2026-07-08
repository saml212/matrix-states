"""diag_ns_admission.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.23 diagnostic.

PRE-STATED HYPOTHESIS (the task brief this script answers): Newton-Schulz
(NS) at n_iter=20 under-converges on the LEARNED/drifted anchor tables at
high K/d (sec 15.22's d=96-wide wave: 11/12 new cells fail
geo3_admission.admissible); n_iter in {24,28,32} restores convergence and
flips admission on the SAME tables. FALSIFIER: if rel-change/admission does
not improve monotonically with n_iter on the real learned tables, the
failure is structural (e.g. an ill-conditioned table), not iteration count.

HEADLINE FINDING (discovered by step 1 below, BEFORE any NS sweep was run):
every one of the 12 originally-failing cells' `checkpoint_fallback_seen`
flags traces EXCLUSIVELY to the C17_heldout_entities recovery probe -- never
M2_in_distribution, M3_held_out, or C19_heldout_template. C17's bind items
are drawn from a name pool architecturally DISJOINT from
`pools.train_name_ids` (run_deltanet_rd.py's own module docstring, line 13:
"C17_heldout_entities (disjoint name pool, H_train)"). `anchor_trained_mask`
is built EXCLUSIVELY from `pools.train_name_ids` (model_rd.py line 925:
`trained_mask[anchor_train_ids] = True`, with `anchor_train_ids =
pools.train_name_ids`) and model_rd.py's own held-out-row zero-gradient
invariant (asserted at model_rd.py:2048) confirms held-out rows NEVER
receive an anchor contribution. `anchor_blend_gather_scatter`
(key_anchoring.py:439-469) computes `trained_here = anchor_trained_mask[
key_ids]` and only touches rows where that is True -- for C17 batches this
is False for EVERY item by construction, so `k_blend_raw` is architecturally
IDENTICAL to `k_eff_raw` (the model's own raw, post-conv keys) for every
C17 query, completely independent of `anchor_table.weight`'s own content.

CONSEQUENCE: the anchor table (this script's PRIMARY analysis below, run
because it is the literal pre-registered ask -- sec 15.22's "Pre-registered
next steps" item 1) is PROVABLY NOT an input to the computation that is
actually failing admission. Its own NS-convergence behavior is real,
reproducible, useful background (e.g. for M2/M3/C19 margin), but CANNOT by
itself confirm or refute the observed C17-driven failures. This is reported
as the diagnostic's own headline finding, not smoothed into a footnote --
see step 1's output and the written result JSON's "mechanism_breakdown".

Three analyses, run in order:
  1. mechanism_breakdown() -- per-checkpoint, per-pool fallback attribution,
     computed directly from the archived cell JSONs (no checkpoint/model
     needed). Establishes the C17-exclusivity finding above.
  2. anchor_table_ns_sweep() -- THE PRE-REGISTERED CHECK. For each pulled
     checkpoint's `anchor_table_trained_rows` (107, d_state), draws 512
     random K-row subsets (K = that cell's own registered K), runs the
     production-equivalent NS iteration (geo3_simulator.newton_schulz --
     key_anchoring.py's own docstring, lines 19-33, documents this as
     "mathematically IDENTICAL" to model_rd.py's newton_schulz_orthogonalize
     and cross-verified to convergence-precision agreement across every
     attack round on this design; imported directly here, matching
     key_anchoring.gate2_ns_leg's own "CPU-testable without fla" import,
     never a hand-copied twin) at n_iter in {20,24,28,32,40}, SAME subsets
     across every n_iter (one 40-iter run per subset, residual history
     sliced at each checkpoint -- mathematically identical to independent
     fresh runs at each n_iter, since NS is a deterministic fixed-point
     iteration on the same X_0). Reports per-n_iter admissibility
     (n_fallback==0, resid_tol=0.01, mirroring key_anchoring.GATE2_RESID_TOL)
     AND per-subset pre-NS Gram-matrix condition number, to discriminate
     "more iterations would have fixed it" from "the subset's own geometry
     has no slack regardless of n_iter." Also runs key_anchoring's own
     raw_table_conditioning (6a/6b) on the full 107-row table as a
     cross-check against the value already logged in each cell's own JSON
     (checkpoints[-1]['item6_table_conditioning']).
  3. random_proxy_sweep() -- EXPLORATORY ONLY, not a reproduction. Since the
     TRUE failing object (C17's raw post-conv keys for held-out entities)
     needs the full trained model (embed + k_proj + k_conv1d weights),
     which this wave's checkpoint writer deliberately did NOT save (sec
     10.10 item 1 -- only the anchor's trained-row block, "27KB negligible"
     by design), it cannot be reconstructed offline this session. As an
     architecturally-motivated proxy, this step runs the identical NS
     sweep on key_anchoring.random_unit_rows_init draws (the codebase's
     own "candidate (e), frozen-random-table" construction: seeded random
     unit rows, explicitly NOT frame-potential-optimized -- i.e. a set of
     K vectors that never received the anchor's own engineering, the one
     structural property C17 queries actually share with this proxy) at
     n = 106 (pool_report's own n_heldout_names, every cell) and each
     cell's own (K, d_state=96). Labeled exploratory throughout; does not
     inform the verdict on its own.

Run: matrix-thinking/deltanet_rd/.venv (or any CPU-only torch env; no
fla/CUDA dependency, mirrors key_anchoring.py's own zero-fla discipline).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F

import key_anchoring as ka
from geo3_simulator import newton_schulz

N_ITER_GRID = (20, 24, 28, 32, 40)
N_SUBSETS = 512
RESID_TOL = ka.GATE2_RESID_TOL          # 0.01, sec 14.10 item 2's admission semantics
SUBSET_SEED = 0
N_HELDOUT_PROXY = 106                    # pool_report.n_heldout_names, every cell this wave

POOLS = ("M2_in_distribution", "M3_held_out", "C17_heldout_entities", "C19_heldout_template")


# ---------------------------------------------------------------------------
# Step 1: per-pool fallback attribution from the archived cell JSONs.
# ---------------------------------------------------------------------------

def mechanism_breakdown(wave_json_paths: list[str]) -> dict:
    """For every cell JSON, per checkpoint, which of the 4 recovery-probe
    pools raised geo3_fallback_triggered_this_hop=True at any hop. This is
    the ONLY step that reads the training-run JSONs directly rather than a
    pulled checkpoint -- establishes which pool(s) are architecturally
    capable of explaining the observed admission failures."""
    out = {}
    for p in sorted(wave_json_paths):
        d = json.load(open(p))
        name = os.path.basename(p)
        admissible = d["geo3_admission"]["admissible"]
        per_ckpt = []
        for ck in d["checkpoints"]:
            pools_hit = []
            for pool in POOLS:
                pd = ck.get(pool, {})
                if any(v.get("geo3_fallback_triggered_this_hop", False)
                       for v in pd.values() if isinstance(v, dict)):
                    pools_hit.append(pool)
            if pools_hit:
                per_ckpt.append({"step": ck["step"], "pools_hit": pools_hit})
        out[name] = {
            "K": d["K"], "seed": d["seed"], "admissible": admissible,
            "checkpoint_fallback_seen": d["geo3_admission"]["checkpoint_fallback_seen"],
            "fallback_events": per_ckpt,
        }
    # Cross-cell summary: does any event EVER occur outside C17?
    all_pools_hit = set()
    for v in out.values():
        for ev in v["fallback_events"]:
            all_pools_hit.update(ev["pools_hit"])
    return {"per_cell": out, "pools_ever_hit_across_all_cells": sorted(all_pools_hit),
            "c17_exclusive": all_pools_hit == {"C17_heldout_entities"}}


# ---------------------------------------------------------------------------
# Step 2 (primary, pre-registered): NS n_iter sweep on the REAL learned
# anchor table pulled from each cell's final (step 20000) checkpoint.
# ---------------------------------------------------------------------------

def _subset_condition_numbers(table: torch.Tensor, K: int, n_subsets: int, seed: int
                               ) -> tuple[list[list[int]], torch.Tensor]:
    """Draw n_subsets random K-row index sets (without replacement, seeded
    -- SAME subsets reused across every n_iter for an apples-to-apples
    comparison) and compute each subset's row-normalized Gram matrix
    condition number (largest/smallest eigenvalue -- a K-vector-set
    analog of raw_table_conditioning's own sigma_ratio, but on the
    ROW-NORMALIZED Gram matrix NS actually operates on, not the raw
    table). A subset whose Gram matrix has a near-zero smallest eigenvalue
    cannot be driven to I_K by ANY orthogonalizer regardless of n_iter --
    the K rows do not span a K-dimensional subspace."""
    n = table.shape[0]
    g = torch.Generator().manual_seed(seed)
    idx_list = [torch.randperm(n, generator=g)[:K].tolist() for _ in range(n_subsets)]
    cond = torch.empty(n_subsets)
    for i, idx in enumerate(idx_list):
        sub = F.normalize(table[idx], dim=-1)                  # (K,d)
        gram = sub @ sub.transpose(0, 1)                       # (K,K), PSD
        eigvals = torch.linalg.eigvalsh(gram).clamp(min=1e-12)
        cond[i] = eigvals[-1] / eigvals[0]
    return idx_list, cond


def _ns_sweep_on_table(table: torch.Tensor, K: int, n_iter_grid=N_ITER_GRID,
                        n_subsets=N_SUBSETS, resid_tol=RESID_TOL, seed=SUBSET_SEED) -> dict:
    idx_list, cond = _subset_condition_numbers(table, K, n_subsets, seed)
    max_n_iter = max(n_iter_grid)
    resid_by_niter = {n: torch.empty(n_subsets) for n in n_iter_grid}
    for i, idx in enumerate(idx_list):
        sub = F.normalize(table[idx], dim=-1).unsqueeze(0)     # (1,K,d)
        _, resid_hist = newton_schulz(sub, max_n_iter)         # deterministic fixed-point map;
        for n in n_iter_grid:                                  # resid_hist[n-1] == a fresh n_iter=n
            resid_by_niter[n][i] = resid_hist[n - 1]            # run's own final residual

    per_niter = {}
    prev_fail_idx = None
    for n in n_iter_grid:
        r = resid_by_niter[n]
        fail_mask = r > resid_tol
        n_fallback = int(fail_mask.sum().item())
        entry = {
            "n_iter": n, "n_fallback": n_fallback, "n_subsets": n_subsets,
            "admissible": n_fallback == 0,
            "max_resid": r.max().item(), "mean_resid": r.mean().item(),
            "p90_resid": r.quantile(0.90).item(),
            "cond_number_failed_subsets_mean": (cond[fail_mask].mean().item()
                                                 if n_fallback > 0 else None),
            "cond_number_passed_subsets_mean": (cond[~fail_mask].mean().item()
                                                 if n_fallback < n_subsets else None),
        }
        if prev_fail_idx is not None:
            # Of the subsets that failed at the PREVIOUS (lower) n_iter, how
            # many are STILL failing now -- the direct per-subset falsifier
            # check: are the same hard cases persisting (structural) or
            # clearing out (iteration-count)?
            still_failing = int((fail_mask & prev_fail_idx).sum().item())
            entry["of_prev_failures_still_failing"] = still_failing
            entry["of_prev_failures_total"] = int(prev_fail_idx.sum().item())
        prev_fail_idx = fail_mask
        per_niter[n] = entry

    return {
        "K": K, "n_subsets": n_subsets, "resid_tol": resid_tol,
        "subset_cond_number_mean": cond.mean().item(),
        "subset_cond_number_max": cond.max().item(),
        "subset_cond_number_min": cond.min().item(),
        "per_n_iter": per_niter,
    }


def anchor_table_ns_sweep(ckpt_path: str, K: int, label: str) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    table = payload["anchor_table_trained_rows"].float()        # (107, d_state)
    assert table.shape[0] >= K, f"{label}: K={K} exceeds table rows {table.shape[0]}"
    cond6 = ka.raw_table_conditioning(table)                    # 6a/6b legs, cross-check vs the
                                                                 # cell JSON's own logged value
    sweep = _ns_sweep_on_table(table, K)
    return {"label": label, "ckpt_step": payload["step"], "d_state": table.shape[1],
            "n_train_rows": table.shape[0],
            "anchor_lambda_raw": (payload["anchor_lambda_raw"].item()
                                   if "anchor_lambda_raw" in payload else None),
            "raw_table_conditioning_6a6b": cond6, **sweep}


# ---------------------------------------------------------------------------
# Step 3 (exploratory only): identical sweep on an UN-engineered random
# unit-row proxy, standing in for C17's actual (unavailable) raw keys.
# ---------------------------------------------------------------------------

def random_proxy_sweep(K: int, label: str, seed: int) -> dict:
    table = ka.random_unit_rows_init(n=N_HELDOUT_PROXY, d=96, seed=seed)
    sweep = _ns_sweep_on_table(table, K)
    return {"label": label, "n_pool_rows": N_HELDOUT_PROXY, "d_state": 96,
            "note": "EXPLORATORY proxy only -- NOT the real C17 held-out conv keys "
                    "(full model not checkpointed this wave); see module docstring step 3.",
            **sweep}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CELLS = [
    # (ckpt_dir_name, K, admissible_label)
    ("wkeyanchor-scaling_rdx_K78_armd_s1840_geo3n20_anchor_learned_dprobe_rev7_d96", 78, "FAILING"),
    ("wkeyanchor-scaling_rdx_K84_armd_s1940_geo3n20_anchor_learned_dprobe_rev7_d96", 84, "FAILING"),
    ("wkeyanchor-scaling_rdx_K90_armd_s2040_geo3n20_anchor_learned_dprobe_rev7_d96", 90, "FAILING"),
    ("wkeyanchor-scaling_rdx_K72_armd_s1742_geo3n20_anchor_learned_dprobe_rev7_d96", 72, "FAILING"),
    ("wkeyanchor-scaling_rdx_K69_armd_s1731_geo3n20_anchor_learned_dprobe_rev7_d96", 69, "PASSING_CONTROL"),
    ("wkeyanchor-scaling_rdx_K72_armd_s1741_geo3n20_anchor_learned_dprobe_rev7_d96", 72, "PASSING_CONTROL"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                     help="dir containing <cell_name>/step20000.pt for every CELLS entry")
    ap.add_argument("--wave-json-glob", required=True,
                     help="glob for the archived wide-grid cell JSONs (mechanism_breakdown input)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mech = mechanism_breakdown(glob.glob(args.wave_json_glob))

    anchor_results = []
    for name, K, label in CELLS:
        ckpt_path = os.path.join(args.ckpt_dir, name, "step20000.pt")
        r = anchor_table_ns_sweep(ckpt_path, K, f"{label}:{name}")
        anchor_results.append(r)
        print(f"[anchor] {label:18s} K={K:3d} {name}: "
              f"admissible@20={r['per_n_iter'][20]['admissible']!s:5s} "
              f"admissible@40={r['per_n_iter'][40]['admissible']!s:5s} "
              f"cond_mean={r['subset_cond_number_mean']:.3e}", flush=True)

    proxy_results = []
    for K in sorted({K for _, K, _ in CELLS}):
        r = random_proxy_sweep(K, f"random_proxy_K{K}", seed=1000 + K)
        proxy_results.append(r)
        print(f"[proxy ] K={K:3d}: admissible@20={r['per_n_iter'][20]['admissible']!s:5s} "
              f"admissible@40={r['per_n_iter'][40]['admissible']!s:5s} "
              f"cond_mean={r['subset_cond_number_mean']:.3e}", flush=True)

    result = {
        "hypothesis": "NS at n_iter=20 under-converges on the LEARNED/drifted anchor tables at "
                       "high K/d; n_iter in {24,28,32} restores convergence and flips admission "
                       "on the SAME tables.",
        "falsifier": "if rel-change/admission does not improve monotonically with n_iter on the "
                      "real learned tables, the failure is structural, not iteration count.",
        "headline_finding": "mechanism_breakdown (step 1) shows the observed admission failures "
                              "are C17_heldout_entities-EXCLUSIVE across all 12 originally-failing "
                              "cells; C17 bind items are architecturally anchor-bypassed "
                              "(anchor_trained_mask[key_ids]=False for 100% of C17 items by "
                              "construction), so the anchor table (step 2, the pre-registered "
                              "check) is PROVABLY not an input to the failing computation. See "
                              "module docstring for the full citation chain.",
        "mechanism_breakdown": mech,
        "anchor_table_ns_sweep": anchor_results,
        "random_proxy_sweep_EXPLORATORY_ONLY": proxy_results,
        "n_iter_grid": list(N_ITER_GRID), "n_subsets": N_SUBSETS, "resid_tol": RESID_TOL,
        "subset_seed": SUBSET_SEED,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    print(f"c17_exclusive across all cells: {mech['c17_exclusive']}", flush=True)


if __name__ == "__main__":
    main()
