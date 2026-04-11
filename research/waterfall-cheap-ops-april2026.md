# Waterfall Analysis: Cheap Matrix Operations (April 2026)

## Process: Brainstorm → Research → Attack → Validation

## Brainstorm (22 ideas)
Top candidates:
1. Batched reshape for tensor cores: 2-5× wall-clock, zero quality risk
2. Monarch factorization: 8× FLOPs but overhead dominates at d=16
3. DeltaNet rank-1 updates: 3.8× on thinking layer
4. Depthwise-separable rank-1/rank-2 perturbations: 4-16× cheaper
5. Mamba-2 scalar gating: 27× cheaper but loses matrix dynamics

## Research Validation
- Batched reshape: PARTIALLY works (2-5×, not 20×). Memory-bound at d=16.
- Monarch: NO at d=16. Overhead dominates. Need d≥64.
- DeltaNet: YES on paper, strongest candidate. But recurrent, not iterative.
- Depthwise-separable: PROBABLY works. Same math as DeltaNet, different derivation.
- Mamba-2 gating: Works but loses structural advantage entirely.

## Attack (all validated by final research agent)
1. DeltaNet is recurrent, we're iterative — pattern doesn't transfer (CORRECT)
2. Rank-1 replaces SwiGLU — massive expressiveness downgrade (CORRECT per-step)
3. Combined speedup is 3.0-4.2×, not 6× (CORRECT)
4. Speed doesn't fix quality gap (CORRECT — BPB 1.67 vs 0.87 is algorithmic)
5. Rank-1 might kill rank enrichment (UNTESTED — no precedent either way)
6. Run flat-vector ablation first (CORRECT — blocks everything)

## Conclusion
Making matrix operations cheaper does NOT solve the fundamental problem.
The quality gap is algorithmic (per-FLOP, vectors produce better predictions).
The only unambiguous win: batched reshape + torch.compile (2-3× free perf).
Everything else is speculative with no published precedent in our architecture type.

## What This Means
The project must decide: is the structural thesis (cross-domain generalization,
abstract reasoning via rank dynamics) worth pursuing despite the per-FLOP quality gap?
Or is the operations thesis dead and only the embedding finding remains?
