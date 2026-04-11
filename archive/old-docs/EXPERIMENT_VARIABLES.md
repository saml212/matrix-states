# Experiment Variables to Test

Variables that could affect matrix thinking performance. Each needs
a controlled experiment: change ONE variable, hold everything else constant.

## Activation Function
- [x] GELU (current baseline, running now)
- [ ] SwiGLU (code ready, standard in all major LLMs, test next)
- [ ] Spectral activation (apply nonlinearity to singular values, preserves matrix structure)
- [ ] No activation in bottleneck (rely solely on multiplicative nonlinearity)

## Bottleneck Size
- [x] d²/4 = 64 (current default for mat_dim=16)
- [ ] d² = 256 (no compression — does removing the bottleneck help?)
- [ ] d = 16 (extreme compression — how much can we squeeze?)
- [ ] Bottleneck in matrix space (bilinear A·M·B instead of flatten→linear→reshape)

## Matrix Dimension
- [x] mat_dim=16 (256 entries, current choice)
- [ ] mat_dim=8 (64 entries, much cheaper)
- [ ] mat_dim=24 (576 entries, richer but more expensive)
- [ ] mat_dim=32 (1024 entries, equivalent to a large vector model)

## Depth vs Width
- [x] 6 layers (current)
- [ ] 4 layers, larger mat_dim (fewer but richer thinking steps)
- [ ] 8 layers, smaller mat_dim (more but leaner steps)
- [ ] Shared weights across layers (same block applied iteratively — prerequisite for PonderNet)

## Flatten vs Matrix-Native Projections (HIGH PRIORITY)
- [x] Flatten to vector for projections (current — destroys structure temporarily)
- [ ] Bilinear projections: Δ = A·M·B (never flatten, rows and cols stay coupled)
- [ ] Row-then-column: A@M compresses rows, M@B compresses cols, combine
- [ ] Kronecker-structured: equivalent to A@M@Bᵀ but as factored linear map
- [ ] Matrix-native attention: Q = A_q@M@B_q instead of Q = Linear(flatten(M))
- Research agent investigating (March 2026)

## Iterative Refinement (future, after basics are proven)
- [ ] Fixed depth (current — 6 layers, no iteration)
- [ ] Shared weights + PonderNet halting
- [ ] Rank-based convergence criterion
- [ ] Energy-based convergence criterion

## Loss Function (future)
- [x] Cross-entropy on next byte (current)
- [ ] Cross-entropy on next token (NLP benchmark, needed for publication)
- [ ] Rank convergence loss (penalize high rank at output)
- [ ] Multi-token prediction (predict K future tokens jointly)

## Data / Benchmarking
- [x] Byte-level WikiText-103 (current, internal use)
- [ ] Tokenized WikiText-103 with BPE (needed for publishable comparison)
- [ ] Multi-modal bytes (text + images, byte-agnostic direction)
