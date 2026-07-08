"""CAPABILITY_SEPARATION_DESIGN.md Rev 1 -- CA1-M1 fix: exact permutation
null distribution for the M1 Spearman bar, numerically executed (not
asserted). Independently reproduces the attack round 1 figures
(S1.13 CA1-M1: P(rho>=0.8|null)=8/120~=6.67%, tie cap 0.9747) via exhaustive
enumeration of all 5!=120 rank permutations, byte-for-byte reasoning, from
scratch in this Rev-1 pass -- not copied from the attack's own numbers.

The d_min(G) sequence (S1.2's table, in family order S3/S4/A5/S5/A6) is
[2,3,3,4,5] -- a TIE at d_min=3 between S4 and A5, the marquee dissociation
pair (intentional: the design's own CONFIRM condition requires these two to
land together). Spearman's rho on tied data uses midranks
(standard convention, scipy.stats.spearmanr's own tie handling) --
midranks([2,3,3,4,5]) = [1, 2.5, 2.5, 4, 5].

Under the null (no true association between d_min(G) and the measured
restricted-rank metric), the metric's ranks are a uniformly random
permutation of {1..5} independent of x. This script computes rho for all
120 such permutations against the FIXED x-midranks and reports the exact
one-sided p-values at rho>=0.8 and rho>=0.9, plus the maximum achievable
rho under the tie and the next-highest achievable value (the size of the
"cliff" a single non-tie misordering falls into).

numpy + stdlib only. Run: python3 spearman_null_calibration.py
Deterministic (exact enumeration, no RNG).
"""
from __future__ import annotations

import itertools
from math import sqrt

DMIN = [2, 3, 3, 4, 5]  # S3, S4, A5, S5, A6 -- S1.2's table order


def midranks(vals: list) -> list:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson(a: list, b: list) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((ai - ma) ** 2 for ai in a)
    vb = sum((bi - mb) ** 2 for bi in b)
    return cov / sqrt(va * vb)


def run():
    x_ranks = midranks(DMIN)
    n = len(DMIN)
    print("=" * 80)
    print("CA1-M1 fix -- exact permutation null for the M1 Spearman bar")
    print("=" * 80)
    print(f"d_min(G) sequence (S3,S4,A5,S5,A6) = {DMIN}")
    print(f"midranks (S4/A5 tie at value 3)     = {x_ranks}")

    rhos = [pearson(x_ranks, list(perm)) for perm in itertools.permutations(range(1, n + 1))]
    total = len(rhos)
    max_rho = max(rhos)
    count_08 = sum(1 for r in rhos if r >= 0.8 - 1e-9)
    count_09 = sum(1 for r in rhos if r >= 0.9 - 1e-9)
    top_distinct = sorted(set(round(r, 4) for r in rhos), reverse=True)[:4]

    print(f"\ntotal permutations enumerated = {total} (5! = 120)")
    print(f"max achievable rho (perfect ordering respecting the S4/A5 tie) = {max_rho:.4f}")
    print(f"next-highest achievable rho (one non-tie misordering)          = {top_distinct[1]:.4f}"
          f"  (cliff size = {max_rho - top_distinct[1]:.4f})")
    print(f"\nP(rho >= 0.8 | null) = {count_08}/{total} = {count_08/total*100:.4f}%")
    print(f"P(rho >= 0.9 | null) = {count_09}/{total} = {count_09/total*100:.4f}%")
    print(f"\ntop 4 distinct achievable rho values: {top_distinct}")

    assert count_08 == 8, f"expected 8/120 at rho>=0.8, got {count_08}"
    assert count_09 == 2, f"expected 2/120 at rho>=0.9, got {count_09}"
    assert abs(max_rho - 0.974679) < 1e-5, f"expected max rho 0.974679, got {max_rho}"
    print("\nASSERTIONS PASSED -- independently reproduces the attack round 1 figures "
          "(S1.13 CA1-M1: 8/120~=6.67%, tie cap 0.9747) via fresh exhaustive enumeration.")


if __name__ == "__main__":
    run()
