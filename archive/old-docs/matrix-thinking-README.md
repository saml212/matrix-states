# Matrix Thinking

Each token is a 16×16 matrix. The model "thinks" via multiplicative matrix
composition and (eventually) iterates until the matrix rank converges,
signaling that the thought has crystallized into something expressible as a token.

## Status: Active. V2 under evaluation.

### The architecture (from MATRIX_THINKING_ARCHITECTURE.md):
```
Input tokens → 16×16 matrices → multiplicative layers → iterate until rank ≈ 1 → output token
```

### What's been built and tested:
- **V1 (bilinear):** A@M@B operation. 128K params, 3.42 BPC. Doesn't beat vectors.
- **V2 (multiplicative):** (I+Δ)·M·(I+Γ) + v·kᵀ. 2.18M params, val BPC 1.951
  at step 14000 (still training). Comparison against vector baseline pending.

### What's been measured:
- Rank-difficulty correlation: -0.25 at convergence (hard tokens → lower rank)
- Rank increases through layers: 1.3 → 7.3 across 6 layers
- The v1→v2 improvement validates that multiplicative composition is the right
  operation for matrix states (confirmed by DeltaProduct literature)

### What hasn't been built yet:
- Iterative refinement (PonderNet-style adaptive thinking depth)
- Rank-based convergence halting
- Standard NLP benchmarks (currently byte-level only)

## Directory Structure
```
src/
  matrix_model.py    — V1: bilinear matrix operations
  matrix_model_v2.py — V2: multiplicative composition (audited, verified)
  data.py            — Data pipeline (shared with byte-agnostic)
research/
  matrix-operations-answer.md  — Why multiplicative beats bilinear
  matrix-states-landscape-2025.md — Survey: matrix states in 2025 literature
  matrix-valued-outputs.md — Research on matrix output heads
```

## Key Design Decisions (V2)
- Pre-norm architecture (normalize before sub-layer, not after)
- Bottlenecked projections (d² → d²/4 → d²) for Δ and Γ
- Separate gates for multiplicative and additive paths
- Scale parameter clamped to [0.01, 0.5]
- Lightweight multi-head attention (linear Q/K/V, not full multiplicative layers)
- SwiGLU-style two-layer MLP output head
- Dropout throughout (attention weights, update paths)
- Code audited twice, all checks passing

## Novelty (verified by research agents, March 2026)
No existing work combines:
1. Matrix-valued autoregressive generation units
2. Rank as a convergence/halting signal
3. Multiplicative composition for matrix-to-matrix thinking
See MATRIX_THINKING_ARCHITECTURE.md for full novelty analysis and citations.
