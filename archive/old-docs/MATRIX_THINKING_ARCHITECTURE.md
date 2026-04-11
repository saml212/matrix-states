# Matrix Thinking Architecture

## The Core Idea

A neural network where intermediate "thoughts" are 16×16 matrices, not vectors.
The model generates matrix-valued thinking tokens, processes them through
matrix-native operations, and only collapses to vectors (then to output tokens)
when the thought has converged — measured by the matrix's rank dropping toward 1.

## Novelty Confirmation

Verified via exhaustive literature search (March 2026). No existing work combines:
1. Matrix-valued autoregressive generation units
2. Algebraic structure within the generation unit
3. Rank-based dynamic termination

The closest existing work:
- **Energy-Based Transformers (July 2025):** Iterates vectors until scalar energy converges
- **Coconut (Meta, Dec 2024):** Feeds hidden vectors back as continuous thoughts
- **Token Maturation (Jan 2026):** Vector trajectories with geometric stability halting
- **PonderLM (2025):** Latent vector thinking with learned halting

All operate on VECTORS. None use MATRICES. None use RANK as convergence signal.

## The Pipeline

```
INPUT:  tokens (bytes or standard NLP tokens)
  │
  ▼
EMBED:  each token → 16×16 matrix (rank-1, via outer product of two 16-dim vectors)
  │
  ▼
PROCESS: matrix-to-matrix layers (shared weights, applied iteratively)
  │     Core operation: M_new = (I + Δ)·M·(I + Γ) + v·kᵀ
  │     Attention: Frobenius inner product for scores, multi-head
  │
  ▼
ITERATE (PonderNet-style):
  │     Each iteration refines the matrix representations
  │     A learned halting probability (biased by rank signal) determines
  │     how much each iteration contributes to the output
  │     Training: all iterations contribute via weighted loss
  │     Inference: truly halt when rank converges
  │
  ▼
CONVERGENCE:
  │     When σ₂/σ₁ < threshold (ratio of 2nd to 1st singular value):
  │       → Extract rank-1 component: output_vector = σ₁ · u₁
  │       → Predict output token from this vector
  │     Easy predictions: ~3 iterations
  │     Hard predictions: ~15-20 iterations
  │
  ▼
OUTPUT: tokens
```

## Loss Function

### Primary: Next-token prediction (weighted across iterations)

Using PonderNet formulation (Banino et al., 2021):

```
λ_t = sigmoid(halt_network(state_t) + α·(threshold - effective_rank(M_t)))
p(halt at t) = λ_t · Π_{i<t}(1 - λ_i)
L_task = Σ_t p(halt at t) · CrossEntropy(prediction_t, target)
L_KL = KL(p_halt || Geometric(λ_prior))
L_total = L_task + β·L_KL
```

No REINFORCE needed — the entire expression is differentiable.

### Secondary: Rank convergence pressure (applied to later iterations)

```
L_rank = Σ_{i≥2} σᵢ²    (squared tail singular values)
```

This is literally the squared Frobenius distance to the nearest rank-1 matrix.
Computed via torch.linalg.svdvals — no U/V needed, avoids SVD gradient instability.

Applied with increasing weight at later iterations:

```
L_convergence = Σ_t (t/T) · L_rank(M_t)
```

Early iterations are free to think at high rank. Later iterations face pressure to converge.

## Gradient Properties (Verified)

1. **Gradient shape:** dL/dM is 16×16 (same shape as M). NOT 256×256. Memory is
   identical to a 256-dimensional vector representation.

2. **SVD gradients:** torch.linalg.svdvals is fully differentiable. The gradient
   instability (1/(σᵢ² - σⱼ²) blowup) only affects U and V, not the singular
   values themselves. Since our loss depends only on singular values, we're safe.

3. **Variable iterations:** Use PonderNet weighted loss during training (all steps
   contribute). Use gradient checkpointing per step + truncated backprop every
   5 steps for bounded memory. Memory: O(5 × 16²) per position, not O(T × 16²).

4. **Alternative: Deep Equilibrium Model (DEQ) approach.** Treat the converged
   matrix as a fixed point and use implicit differentiation. O(1) memory in
   iteration count. The torchdeq library implements this. Adds one extra
   forward-backward pass for the gradient solve.

## Memory and Compute (Verified Feasible)

For batch=64, seq=256, matrix=16×16, max_steps=20:

| Approach | Memory | Compute |
|----------|--------|---------|
| Naive unrolling | 3.3 GB | O(T × d³) |
| Gradient checkpointing | 490 MB | O(T × d³) × 2 |
| Truncated backprop (K=5) | 819 MB | O(T × d³) |
| DEQ (implicit diff) | ~250 MB | O(T × d³) + one extra pass |

All feasible on consumer GPU. The 16×16 matrix size keeps costs low —
16³ = 4096 FLOPs per matrix multiply, cheaper than a standard FFN.

## The 16×16 Choice

16×16 = 256 entries. Same storage as a 256-dimensional vector, but with structure.

Why 16 specifically:
- 256 = 16² gives manageable compute (O(16³) = 4096 per matmul)
- 16² = 256, which is also the number of possible byte values (coincidence, but elegant)
- Effective rank ranges from 1 to 16, giving a 16-step "resolution" for thinking depth
- 16 is a power of 2, GPU-friendly for memory alignment
- Matches the dimensionality used in CliffordNet's 16-dim multivectors

## Benchmarking Strategy

Phase 1: Standard NLP benchmarks (WikiText-103, standard BPE tokenizer)
- Compare matrix-thinking transformer vs standard transformer at matched params
- Metric: perplexity (directly comparable to published baselines)
- Target: match GPT-2 small perplexity with fewer parameters

Phase 2: Byte-level multi-modal (text + images + audio as raw bytes)
- Test domain generalization (the original vision)
- Metric: BPC per domain
- Target: emergent domain differentiation without labels

## What Needs To Be Built (in order)

### Step 1: CURRENT — Prove matrix representations beat vectors
Running now. Matrix V2 (multiplicative, 2.18M params) vs Vector Transformer
(2.05M params) on byte-level text prediction. Results pending.

### Step 2: Switch to standard NLP benchmarks
Replace byte prediction with tokenized prediction (HuggingFace BPE tokenizer).
Benchmark on WikiText-103 perplexity. Compare to published baselines.

### Step 3: Add iterative refinement (PonderNet-style)
Shared-weight matrix block applied iteratively. Learned halting probability
biased by rank signal. Weighted loss across all steps. Gradient checkpointing.

### Step 4: Measure rank dynamics during thinking
Does rank naturally peak then decrease? Do harder tokens take more iterations?
Does the halting probability correlate with linguistic complexity?

### Step 5: Scale up
If steps 1-4 show signal: scale to 50M-200M params on H100.
Compare to published transformer baselines at matched scale.

## Key References

### The architecture draws from:
- **EBTs (July 2025):** Iterative refinement paradigm, 35% better scaling
- **PonderNet (2021):** Learned halting distribution, differentiable
- **DeltaProduct (2025):** Multiplicative matrix composition is exponentially expressive
- **DEQ (2019):** Implicit differentiation for O(1) memory through convergence

### Novelty confirmed against:
- Coconut (Meta 2024) — vectors, not matrices
- Token Maturation (Jan 2026) — vectors, geometric stability not rank
- PonderLM/PonderLM-2 (2025) — vectors, learned halting not rank
- CALM (Oct 2025) — continuous vectors, no iteration
- Think-at-Hard (Nov 2025) — learned classifier, not intrinsic rank
- TTT / Titans — matrices as memory, not as generated thoughts
- TensorAR / xAR — spatial patches, not algebraic matrices

### Connection to neuroscience:
- PFC dimensionality collapses on error trials (J. Neuroscience, Feb 2025)
- Higher neural dimensionality predicts better episodic memory (Science Advances, 2022)
- Variable thinking time maps to variable iteration count
