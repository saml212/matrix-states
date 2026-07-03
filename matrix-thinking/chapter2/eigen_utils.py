"""eigen_utils.py -- C8: eigenstructure-fidelity metric for Task E
(NEXT_EXPERIMENT_DESIGN.md, section 5 and section 6 M3_E).

Composability / state-tracking is gated by eigenvalue SIGN and PHASE
structure, not rank magnitude alone (Grazzi et al., "Unlocking State-Tracking
in Linear RNNs via Negative Eigenvalues", arXiv:2411.12537, ICLR 2025;
DeltaProduct, arXiv:2502.10297): a rank-K matrix can satisfy one-hop recall
while having the wrong eigenvalue structure to compose correctly under
repeated self-application. This module measures the distance, in the complex
plane, between a trained Z's top-k eigenvalues and the idealized reference
Z_ideal's eigenvalues (for a pure K-cycle permutation these are exactly the
K-th roots of unity -- see the self-test), under an OPTIMAL (Hungarian /
Kuhn-Munkres) matching, not a greedy nearest-neighbor heuristic.

Self-contained: torch + stdlib only (no numpy/scipy) -- pod-safe, matching
task_d.py / rank_utils.py's convention. scipy.optimize.linear_sum_assignment
would normally do this in one line, but is deliberately avoided here so Task
E's scripts stay dependency-free for pod deployment.
"""
from __future__ import annotations

import itertools

import torch


def _hungarian_min_cost(cost: list) -> list:
    """Kuhn-Munkres (Hungarian) algorithm, O(n^3), minimum-cost perfect
    assignment on a square real cost matrix.

    cost: n x n list of lists of floats. Returns `assign` (length n) such that
    row i is matched to column assign[i], minimizing sum(cost[i][assign[i]]).

    Standard shortest-augmenting-path-with-potentials formulation (1-indexed
    internally). Verified against brute-force enumeration on 200 random
    trials (n=1..7) plus a hand-worked example before being wired in here --
    see the audit notes / build log for the standalone verification script.
    """
    n = len(cost)
    if n == 0:
        return []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)      # p[j] = row currently assigned to column j (1-indexed; 0 = none)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = [0] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assign[p[j] - 1] = j - 1
    return assign


def eigenvalue_fidelity(Z: torch.Tensor, Z_ideal: torch.Tensor, k: int) -> torch.Tensor:
    """Optimally-matched mean distance (complex plane) between the top-k
    (by magnitude) eigenvalues of Z and of Z_ideal, per batch item.

    Z, Z_ideal: (B, d, d) real matrices (may have complex eigenvalues).
    Returns: (B,) mean per-eigenvalue Euclidean distance after Hungarian
    matching -- 0 means the trained Z's dominant spectrum exactly matches the
    classical reference's; large values mean SGD found a rank-sufficient but
    eigenstructurally-wrong operator (the C8 measurement, reported alongside
    every M3_E number, not asserted qualitatively).

    The reference is Z_ideal's ACTUAL per-episode spectrum, not a hardcoded
    roots-of-unity set. As of the 2026-07-01 audit fix
    (gauntlet/AUDIT_task_e_validity.md Finding B), the permutation variant's
    `pi` (task_e.py::_permutation_graph) is always a SINGLE Hamiltonian
    K-cycle -- not a general permutation that could split into several
    disjoint shorter cycles -- so Z_ideal's eigenvalues are always EXACTLY the
    K-th roots of unity, matching the `_self_test` construction below exactly
    rather than as a special case. The chain variant's Z_ideal is nilpotent
    instead (a finite acyclic path graph), with eigenvalues at/near 0. Because
    the metric matches against Z_ideal directly (not a hardcoded reference),
    it is correct for both variants regardless of which spectral story
    applies.

    Not batched inside torch.linalg (Hungarian has no vectorized torch
    primitive available pod-safe / stdlib-only), so this loops over the batch
    in Python -- fine for periodic eval reporting (not called every training
    step), and k is small (<=~32) so the O(k^3) inner assignment is cheap.
    """
    ev_z = torch.linalg.eigvals(Z.float())          # (B, d) complex
    ev_i = torch.linalg.eigvals(Z_ideal.float())     # (B, d) complex
    B = Z.shape[0]
    out = torch.zeros(B, dtype=torch.float32, device=Z.device)   # MINOR fix: match Z's device
    for b in range(B):
        za = ev_z[b]
        zi = ev_i[b]
        za_k = za[torch.argsort(za.abs(), descending=True)[:k]]
        zi_k = zi[torch.argsort(zi.abs(), descending=True)[:k]]
        kk = za_k.shape[0]
        cost = [[(za_k[i] - zi_k[j]).abs().item() for j in range(kk)] for i in range(kk)]
        assign = _hungarian_min_cost(cost)
        dists = [cost[i][assign[i]] for i in range(kk)]
        out[b] = (sum(dists) / kk) if kk > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Self-test (no torch dependency issues expected; also exercises the pure-
# Python Hungarian matcher independently of torch, though torch IS required
# for the eigendecomposition half -- runs where torch is available, part of
# run_task_e.py's --smoke gate).
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # 1) Hungarian matcher: verify optimal assignment on a hand-worked example
    #    against brute-force enumeration (independent ground truth).
    cost = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]
    assign = _hungarian_min_cost(cost)
    total = sum(cost[i][assign[i]] for i in range(3))
    best = min(sum(cost[i][perm[i]] for i in range(3))
              for perm in itertools.permutations(range(3)))
    assert total == best, f"Hungarian gave {total}, optimal is {best}"

    # 2) Z_ideal for a pure K-cycle permutation has eigenvalues = the K-th
    #    roots of unity (magnitude exactly 1) -- the "roots of unity for pure
    #    cycles" claim from NEXT_EXPERIMENT_DESIGN.md section 5 (C8), checked
    #    empirically here, not just asserted in prose.
    K = 6
    Z = torch.zeros(1, K, K)
    perm = torch.roll(torch.arange(K), shifts=-1)   # single K-cycle: 0->1->...->K-1->0
    for i in range(K):
        Z[0, perm[i], i] = 1.0
    ev = torch.linalg.eigvals(Z[0])
    mags = ev.abs().sort(descending=True).values
    assert torch.allclose(mags, torch.ones(K), atol=1e-4), \
        f"K-cycle eigenvalues should all have magnitude 1, got {mags}"

    # 3) Self-distance (Z vs itself) is exactly zero.
    d0 = eigenvalue_fidelity(Z, Z, K)
    assert d0.item() < 1e-4, f"self eigenvalue distance should be ~0, got {d0.item()}"

    print("eigen_utils self-test PASSED")


if __name__ == "__main__":
    _self_test()
