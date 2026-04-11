# The Right Matrix-to-Matrix Operation

## Answer: Data-dependent multiplicative composition via Lie group action

```
M_new = g_L · M · g_R + v · kᵀ
```

Where g_L, g_R are data-dependent invertible matrices (parameterized via I+Δ, Cayley, or exp).

## Why this works (DeltaProduct insight):
- Additive: sum of T rank-1 terms → rank ≤ T (linear growth, limited)
- Multiplicative: product of T rank-1 perturbations → ANY rank (exponential expressiveness)

## Implementation tiers:
- Tier 0: g = I + Δ (cheapest, validate concept)
- Tier 1: g = cayley(Δ) (guaranteed invertible, stable)
- Tier 2: g = exp(Δ) (most expressive, uses torch.matrix_exp)

## What we had wrong:
Our bilinear A@M@B is LINEAR in M — equivalent to operating on a flattened vector.
The key is I+Δ structure: MULTIPLICATIVE perturbation of identity.

## For d=16, 8 heads: total state = 2048 dims, per-head cost = 8K FLOPs. Very cheap.
