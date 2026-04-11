# Parameter Matching Notes for H100 Experiments

## The Problem
Matrix V3 with Kronecker projections is extremely parameter-efficient.
Even with mat_dim=32, 12 layers, K=4, we only get ~6M params.
A standard vector transformer starts at 30M+ params for useful sizes.

## Why
- Kronecker projection: 2 × K × d² params (at d=16, K=4: 2048 per projection)
- Linear projection: d_model² params (at d_model=512: 262,144 per projection)
- That's a 128x difference per projection
- The embedding tables (vocab × d) dominate in the matrix model

## Fair Comparison Strategy

### Approach: Matched training compute (FLOPs), not matched params
Since the matrix model does more FLOPs per parameter (matrix multiplications
in Kronecker projections), comparing at matched FLOPs is fairer than matched params.

### Concrete configs for H100

Small scale (~5M params, good for quick validation):
- Matrix V3: mat_dim=32, 12 layers, K=4, heads=8 → ~5.9M params
- Vector: d=96, 6 layers, heads=4 → ~5M params (very small transformer)

Medium scale (~30M params):
- Matrix V3: not achievable without non-Kronecker components
- Alternative: use Kronecker K=16 + more layers + bigger mat_dim
- Vector: d=256, 6 layers, heads=4 → ~30M params

The parameter efficiency IS the story for the matrix model. If a 5.9M matrix
model matches or beats a 30M vector model, that's a 5x parameter reduction.
We should run BOTH comparison types:
1. Matched params (5M each) — does matrix match vector at same size?
2. Cross-scale (5M matrix vs 30M vector) — does matrix match a 5x larger model?
