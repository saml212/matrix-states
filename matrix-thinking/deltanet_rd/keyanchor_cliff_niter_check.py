"""keyanchor_cliff_niter_check.py -- KEY_ANCHORING_DESIGN.md sec 12.2.2
(Rev 12.1, attack finding 7)'s registered zero-GPU Wave -1 item for the
capacity-cliff localization wave (sec 12).

GATE2_N_ITER_BY_K's own {34:20, 38:20, 42:20, 46:20} entries (key_anchoring.py)
were chosen BY ANALOGY to K=32/48 (both already n_iter=20) -- not independently
verified sufficient at these specific K's. This script IS that verification:
sec 12.2.2's own method (identical to sec 11.4.2's already-published lambda=1
post-NS drift ceiling computation) re-run at n_iter in {12,16,20,24} for each
of K=34/38/42/46, checking the ceiling value has CONVERGED (change from
n_iter=20->24 below a small disclosed tolerance, <0.5% relative) before
n_iter=20 is trusted as the production tier for Gate 2 / the wave's own cells.

Method (sec 11.4.2, reproduced verbatim -- CPU-only, pure torch, no GPU, no
training data): on the registered frame-potential anchor table
(frame_potential_init(107, 64, seed=ANCHOR_INIT_SEED)), fix one of 8 sampled
train-pool entities as the anchor row; for each of 32 independent resamples,
draw K-1 CO-DRAWN rows from the other 106 train-pool rows (without
replacement), run the full K-row set through production-tier Newton-Schulz at
the n_iter under test, and keep ONLY the fixed entity's own post-NS output
row from that resample (key_anchoring.py's own measure_entity_rows/
pairwise_drift_stats pattern -- the "drift" statistic is how STABLE one
entity's post-NS direction is across independently-resampled contexts, not
its orthogonality to its co-drawn rows within one resample). The ceiling is
the pooled (mean, p10) of the PAIRWISE COSINE AMONG THOSE 32 output rows
(one entity's own post-NS direction, resample vs. resample), pooled across
all 8 fixed anchors -- the SAME "lambda=1, full-pool, no bypassed
learned-key path" object sec 11.4.2 computed at K=16/32/48 (0.9745/0.9423/
0.8987 at n_iter=20 there) -- this script sweeps n_iter itself as the free
variable, holding K fixed at each of the wave's 4 new points.

Usage:
    python keyanchor_cliff_niter_check.py [--out keyanchor_cliff_niter_check_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import key_anchoring as ka                                   # noqa: E402 (fla-free)
from geo3_simulator import newton_schulz as _newton_schulz    # noqa: E402 (fla-free)

N_ENTITIES = 107          # rev7_threshold_derive.py's own N_ENTITIES -- the fixed train-pool size
D_STATE = 64
N_ITER_GRID = (12, 16, 20, 24)   # sec 12.2.2's own registered sweep
CANDIDATE_KS = (34, 38, 42, 46)  # sec 12.2's re-picked point set
N_FIXED_ANCHORS = 8              # sec 11.4.2's own "8 sampled anchor rows"
N_RESAMPLES = 32                 # sec 11.4.2's own "32 independent resamples"
CONVERGENCE_TOL_REL = 0.005      # <0.5% relative change, 20->24, sec 12.2.2's own disclosed tolerance


def ceiling_at_niter(table: torch.Tensor, K: int, n_iter: int,
                      n_fixed_anchors: int = N_FIXED_ANCHORS,
                      n_resamples: int = N_RESAMPLES, seed: int = 0) -> dict:
    """sec 11.4.2's method, K/n_iter as free parameters. table: (107, 64)
    frame-potential anchor table (train-row block). For each of
    n_fixed_anchors fixed entities, collect its post-NS output row across
    n_resamples independently-resampled co-drawn contexts, then pool the
    pairwise cosine AMONG those n_resamples rows (per entity) across all
    n_fixed_anchors entities -- key_anchoring.py's own
    measure_entity_rows/pairwise_drift_stats statistic, reproduced here
    on the raw anchor table (no model, no training) since this check
    only needs the geometry, not a trained checkpoint."""
    n = table.shape[0]
    assert K <= n, f"cannot draw K={K} rows (incl. the fixed anchor) from a {n}-row table"
    g = torch.Generator().manual_seed(seed)
    fixed_idx = torch.randperm(n, generator=g)[:n_fixed_anchors].tolist()
    all_off_diag = []
    for anchor_i in fixed_idx:
        others = [i for i in range(n) if i != anchor_i]
        anchor_out_rows = []
        for _ in range(n_resamples):
            perm = torch.randperm(len(others), generator=g)[: K - 1]
            co_drawn = [others[j] for j in perm.tolist()]
            row_idx = [anchor_i] + co_drawn                      # anchor row first
            sub = F.normalize(table[row_idx], dim=-1).unsqueeze(0)   # (1,K,d)
            Q, _ = _newton_schulz(sub, n_iter)                    # (1,K,d), post-NS orthogonalized
            anchor_out_rows.append(Q[0, 0])                       # this entity's own output row
        rows = torch.stack(anchor_out_rows, dim=0)                # (n_resamples, d)
        pw = F.cosine_similarity(rows.unsqueeze(0), rows.unsqueeze(1), dim=-1)
        off = pw[~torch.eye(n_resamples, dtype=torch.bool)]
        all_off_diag.append(off)
    t = torch.cat(all_off_diag)
    mean = t.mean().item()
    # p10 = 10th percentile of the pooled pairwise-cosine distribution --
    # key_anchoring.py's own pairwise_drift_stats/measure_drift convention
    # ("mean X / p10 Y", p10 < mean -- a lower-tail-of-consistency
    # statistic; both near 1.0 for a well-conditioned, low-drift entity).
    p10 = torch.quantile(t, 0.10).item()
    return {"K": K, "n_iter": n_iter, "n_fixed_anchors": n_fixed_anchors,
            "n_resamples": n_resamples, "n_pooled": t.numel(),
            "mean": mean, "p10": p10}


def converged(results_by_niter: dict) -> dict:
    """sec 12.2.2's own convergence criterion: relative change in the mean
    ceiling from n_iter=20 -> 24 must be < CONVERGENCE_TOL_REL (0.5%).
    Exact threshold, no slack beyond the disclosed 0.5% figure itself."""
    m20 = results_by_niter[20]["mean"]
    m24 = results_by_niter[24]["mean"]
    rel_change = abs(m24 - m20) / max(abs(m20), 1e-12)
    return {"mean_at_20": m20, "mean_at_24": m24, "rel_change_20_to_24": rel_change,
            "tolerance": CONVERGENCE_TOL_REL, "converged": bool(rel_change < CONVERGENCE_TOL_REL)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "keyanchor_cliff_niter_check_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    table = ka.frame_potential_init(N_ENTITIES, D_STATE, seed=ka.ANCHOR_INIT_SEED)

    per_k = {}
    all_converged = True
    for K in CANDIDATE_KS:
        by_niter = {}
        for n_iter in N_ITER_GRID:
            by_niter[n_iter] = ceiling_at_niter(table, K, n_iter, seed=args.seed + K)
        conv = converged(by_niter)
        all_converged = all_converged and conv["converged"]
        per_k[K] = {"by_n_iter": by_niter, "convergence_20_to_24": conv}
        print(f"K={K}: " + " ".join(f"n_iter={ni} mean={by_niter[ni]['mean']:.6f} "
                                     f"p10={by_niter[ni]['p10']:.6f}" for ni in N_ITER_GRID))
        print(f"  convergence(20->24): rel_change={conv['rel_change_20_to_24']:.6f} "
              f"(tol {conv['tolerance']}) -> {'CONVERGED' if conv['converged'] else 'NOT CONVERGED'}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "keyanchor_cliff_niter_check.py",
        "purpose": "KEY_ANCHORING_DESIGN.md sec 12.2.2 (Rev 12.1, attack finding 7) -- zero-GPU "
                   "n_iter-sufficiency check for GATE2_N_ITER_BY_K's K=34/38/42/46 entries",
        "method": "sec 11.4.2's own method: fix 1 of 8 sampled train-pool anchor rows, resample "
                  "K-1 co-drawn rows from the other 106, full production-tier Newton-Schulz, "
                  "pooled (mean, p10) fixed-row post-NS pairwise cosine across 8*32=256 draws",
        "n_iter_grid": list(N_ITER_GRID),
        "candidate_ks": list(CANDIDATE_KS),
        "n_entities": N_ENTITIES, "d_state": D_STATE,
        "n_fixed_anchors": N_FIXED_ANCHORS, "n_resamples": N_RESAMPLES,
        "convergence_tolerance_rel": CONVERGENCE_TOL_REL,
        "per_k": {str(k): v for k, v in per_k.items()},
        "all_ks_converged": bool(all_converged),
        "wall_s": time.time() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out} ({out['wall_s']:.1f}s)")
    print(f"ALL K's converged (n_iter=20 sufficient, <0.5% rel change to n_iter=24): {all_converged}")
    return 0 if all_converged else 1


if __name__ == "__main__":
    sys.exit(main())
