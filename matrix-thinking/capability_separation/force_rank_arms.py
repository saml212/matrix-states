"""CAPABILITY_SEPARATION_DESIGN.md S1.4.2 -- the force-rank arms: Arm 1
(unconstrained) + Arm 2 (train-time force-rank grid straddling d_min(G),
`k in {d_min(G)-1, d_min(G), d_min(G)+1}`, the pre-registered minimum grid).

Wiring: `rank_utils.truncate_to_rank` applied end-of-sequence inside
`GroupWordModel.encode`'s `force_rank_k` branch (group_word_encoder.py,
already built -- identical mechanism to `MatrixMemoryModel.encode`'s C1
path, same `eigh`-based, degenerate-spectrum-safe implementation Task D/E
already smoke-tested for NaN/Inf-free backward). This module defines the
per-group ARM GRID (S1.4.2's exact table) and the subspace-restricted rank
measurement each arm's cell reports -- the SAME `readout.py` pipeline
(S1.4.1's generalization of `analyze_zdump.py`'s pinned-rho ideal-trajectory
method: derive U from the model's OWN output covariance, restrict, degauge,
score on a disjoint eval set), just invoked once per grid point instead of
once for the unconstrained arm alone.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from groups import GROUP_NAMES, D_MIN, D_STATE
from group_word_encoder import GroupWordModel
from group_task import generating_set
import readout


def force_rank_grid(name: str) -> dict:
    """S1.4.2's exact grid for group `name`: {'unconstrained': None,
    'k_dmin_minus_1': d_min-1, 'k_dmin': d_min, 'k_dmin_plus_1': d_min+1}.
    All three force-rank points fit inside [1, d_state] by construction
    (D_STATE = D_MIN + 2, S1.4's uniform-margin rule)."""
    d_min = D_MIN[name]
    d_state = D_STATE[name]
    grid = {
        "unconstrained": None,
        "k_dmin_minus_1": d_min - 1,
        "k_dmin": d_min,
        "k_dmin_plus_1": d_min + 1,
    }
    for label, k in grid.items():
        if k is not None:
            assert 1 <= k <= d_state, f"{name}/{label}: k={k} outside [1,{d_state}]"
    return grid


def cell_types_for_group(name: str, unconditional_n5: bool) -> dict:
    """S1.4.2's per-group seed allocation table. `unconditional_n5` gates
    S4/A5's CA3-M1(a) bump (5 seeds, unconditional, at the unconstrained and
    k=d_min cells only -- the marquee-carrying cell types); every other
    group/cell-type combination uses the S3/S5/A6 default. Returns
    {label: n_seeds}."""
    if unconditional_n5:
        return {"unconstrained": 5, "k_dmin": 5, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2}
    return {"unconstrained": 3, "k_dmin": 3, "k_dmin_minus_1": 2, "k_dmin_plus_1": 2}


def n_cells_per_group(name: str) -> int:
    unconditional_n5 = name in ("S4", "A5")
    return sum(cell_types_for_group(name, unconditional_n5).values())


def run_one_cell(model: GroupWordModel, name: str, arm_label: str, seed: int, device="cpu") -> dict:
    """Evaluate one (group, arm) cell via readout.py's S1.4.1 pipeline,
    force_rank_k threaded through model.encode exactly as C1 requires."""
    grid = force_rank_grid(name)
    k = grid[arm_label]
    result = readout.run_subspace_restriction_pipeline(model, name, base_seed=seed, device=device,
                                                        force_rank_k=k)
    result["arm"] = arm_label
    return result


# ---------------------------------------------------------------------------
# Smoke: all 4 arms, one group, untrained model -- sanity checks the
# force_rank_k wiring end-to-end through the SAME S1.4.1 pipeline gate1
# already exercised for the unconstrained case.
# ---------------------------------------------------------------------------

def smoke(device="cpu"):
    print("=" * 88)
    print("  force_rank_arms.py SMOKE -- all 4 arms (S4), untrained model")
    print("=" * 88)
    torch.manual_seed(0)
    name = "S4"
    d_state, n_gens = D_STATE[name], len(generating_set(name))
    model = GroupWordModel(d_state, n_gens, L_max=16, h=32).to(device)
    grid = force_rank_grid(name)
    print(f"  grid={grid}")
    for label, k in grid.items():
        res = run_one_cell(model, name, label, seed=5000 + hash(label) % 1000, device=device)
        er = res["restricted_effective_rank"]
        print(f"  [{label:<16}] k={str(k):<4} restricted_eff_rank={er:.3f}  "
              f"mean_cos={res['mean_cos']:.4f}  n_fit={res['n_fit']} n_eval={res['n_eval']}")
        if k is not None:
            assert er <= k + 1.0, \
                f"{label}: restricted_eff_rank {er:.3f} implausibly exceeds force_rank_k={k}"
    n_cells = n_cells_per_group("S4")
    print(f"\n  n_cells_per_group('S4') = {n_cells}  (expect 14, S1.4.2 CA3-M1(a) bump)")
    assert n_cells == 14
    n_cells_s3 = n_cells_per_group("S3")
    print(f"  n_cells_per_group('S3') = {n_cells_s3}  (expect 10, unbumped default)")
    assert n_cells_s3 == 10
    total = sum(n_cells_per_group(g) for g in GROUP_NAMES)
    print(f"  total cells across the 5-group family = {total}  (expect 58, S1.4.2/S1.6)")
    assert total == 58
    print("\n" + "=" * 88 + "\n  force_rank_arms.py SMOKE PASSED\n" + "=" * 88)


if __name__ == "__main__":
    smoke("cuda" if torch.cuda.is_available() else "cpu")
