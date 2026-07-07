"""keyanchor_scaling_wide_niter_check.py -- KEY_ANCHORING_SCALING_DRAFT.md
sec 15.20.1/15.20.6 Gate (b) (Rev 1 build-audit MAJOR-1 fix, 2026-07-07):
the registered, MANDATORY Wave -1 n_iter-sufficiency check for
`KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96]` at K in {72, 78, 84, 90} -- never
actually built before this fix. The design doc's own gate table (sec
15.20.6, Gate (b)) required it; run_deltanet_rd_exactness_sweep.py's own
`KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96].update({72: 20, 78: 20, 84: 20,
90: 20})` set every one of those four "by analogy only" (sec 15.20.1's own
disclosure); and `keyanchor_scaling_wide_stage_gate` never checked a
sufficiency artifact for any of them -- exactly the un-run TODO sec 15.20.1's
own text discloses. This script IS that measurement.

PURPOSE-BUILT DRIVER, same "zero code changes to the reused script, only a
new driver call" convention as sim_cliff_power_wide_grid.py's own treatment
of sim_cliff_power.py (sec 15.20.6 item 7): imports keyanchor_dstate_niter_
check.py's own `ceiling_at_niter`/`converged` functions and
`N_ITER_GRID`/`CONVERGENCE_TOL_REL`/`N_FIXED_ANCHORS`/`N_RESAMPLES`/
`N_ENTITIES` constants UNMODIFIED -- that sibling is already CLI-
parameterized over --d-state/--ks (verified this session: pure torch,
imports only `key_anchoring`/`geo3_simulator.newton_schulz`, zero `fla`/CUDA
references, grepped directly), but this wave gets its OWN purpose-built
driver with its actual registered grid hardcoded as module constants,
matching sim_cliff_power_wide_grid.py's own precedent rather than relying on
argv defaults happening to match. K=69 is deliberately EXCLUDED from
CANDIDATE_KS (sec 15.20.1: "K=69 does not need re-checking, already verified
clean in the original wave").

Method (sec 11.4.2, reused verbatim from the sibling, not reimplemented):
fix 1 of 8 sampled train-pool anchor rows, resample K-1 co-drawn rows from
the other 106 train-pool rows, run the full K-row set through production-
tier Newton-Schulz at the n_iter under test, keep only the fixed anchor's own
post-NS output row. The ceiling is the pooled (mean, p10) of the pairwise
cosine among those rows across 32 resamples x 8 fixed anchors = 256 draws.
Convergence: |mean(n_iter=24) - mean(n_iter=20)| / mean(n_iter=20) < 0.5%
(CONVERGENCE_TOL_REL, exact threshold, no numerical-tolerance slack).

CPU-only, pure torch, no GPU, no training data -- run locally
(../../.venv/bin/python), no box/SSH dependency.

Usage:
    ../../.venv/bin/python keyanchor_scaling_wide_niter_check.py \
        --out results/keyanchor_scaling_wide_niter_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import key_anchoring as ka                                          # noqa: E402 (fla-free)
from keyanchor_dstate_niter_check import (                          # noqa: E402
    ceiling_at_niter, converged, N_ITER_GRID, N_ENTITIES,
    N_FIXED_ANCHORS, N_RESAMPLES, CONVERGENCE_TOL_REL,
)

D_STATE = 96                          # sec 15.20.1: this wave's own d (unchanged from the
                                       # ORIGINAL keyanchor-scaling wave's own d=96 -- only the
                                       # K-grid below is new).
CANDIDATE_KS = (72, 78, 84, 90)       # sec 15.20.1's build task grid, verbatim. K=69 excluded --
                                       # already verified clean in the original wave (sec 15.19),
                                       # no re-check needed (sec 15.20.1's own text).


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                    "keyanchor_scaling_wide_niter_result.json"))
    ap.add_argument("--seed", type=int, default=0,
                     help="same convention as keyanchor_dstate_niter_check.py's own default: "
                          "per-K offset seed=args.seed+K")
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
        "script": "keyanchor_scaling_wide_niter_check.py",
        "purpose": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.20.1/15.20.6 Gate (b) (build-audit "
                   "MAJOR-1 fix): zero-GPU n_iter-sufficiency check at d_state=96 for "
                   "KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96]'s K=72/78/84/90 entries (all "
                   "previously set to n_iter=20 'by analogy only', never mechanically verified "
                   "at d_state=96).",
        "method": "sec 11.4.2's own method, reused verbatim via keyanchor_dstate_niter_check.py's "
                  "ceiling_at_niter/converged (unmodified): fix 1 of 8 sampled train-pool anchor "
                  "rows, resample K-1 co-drawn rows from the other 106, full production-tier "
                  "Newton-Schulz, pooled (mean, p10) fixed-row post-NS pairwise cosine across "
                  "8*32=256 draws per (K, n_iter)",
        "n_iter_grid": list(N_ITER_GRID),
        "candidate_ks": list(CANDIDATE_KS),
        "n_entities": N_ENTITIES, "d_state": D_STATE,
        "n_fixed_anchors": N_FIXED_ANCHORS, "n_resamples": N_RESAMPLES,
        "convergence_tolerance_rel": CONVERGENCE_TOL_REL,
        "per_k": {str(k): v for k, v in per_k.items()},
        "all_ks_converged": bool(all_converged),
        "wall_s": time.time() - t0,
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path} ({out['wall_s']:.1f}s)")
    print(f"ALL K's converged at d_state={D_STATE} (n_iter=20 sufficient, <0.5% rel change to "
          f"n_iter=24): {all_converged}")
    return 0 if all_converged else 1


if __name__ == "__main__":
    sys.exit(main())
