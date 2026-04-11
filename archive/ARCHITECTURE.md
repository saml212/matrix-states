# Matrix Thinking Architecture

## The Core Idea

The model generates a sequence of **matrix-valued thoughts** before producing output.
Each thought is a NEW 16×16 matrix appended to the context. Thoughts attend to all
input bytes and all previous thoughts via causal attention. The model decides when
it's done thinking and produces an output byte.

**This is autoregressive generation in matrix space.** Not iterative refinement of
existing positions. Not fixed-depth processing. The model GENERATES new abstract
representations — thoughts of increasing complexity — until it's ready to commit
to an output.

```
INPUT:  raw bytes (0-255)
  │
  ▼
EMBED:  each byte → 16×16 matrix (rank-1, outer product of two 16-dim vectors)
  │
  ▼
THINK:  autoregressive thought generation
  │     thought_1 = f(inputs)                          — rank ~2, broad
  │     thought_2 = f(inputs, thought_1)               — rank ~4, building
  │     thought_3 = f(inputs, thought_1, thought_2)    — rank ~6, crystallizing
  │     ...
  │     thought_N = f(inputs, thought_1..N-1)          — unbounded, model decides
  │
  │     Each thought attends to everything before it (standard causal attention).
  │     Rank should INCREASE across thoughts as complexity builds.
  │     The model learns to stop thinking via a halt signal or certainty threshold.
  │
  ▼
OUTPUT: final thought → 16×16 matrix → 256 byte logits (zero-param output)
  │     Or: model generates a special "output" matrix that IS the byte prediction
  │
  ▼
BYTES OUT
```

## What Makes This Different From Iterative Refinement

**Iterative refinement (what we tested):**
- Fixed set of positions, refined T times
- All positions get same treatment
- No new representations generated
- Fixed compute budget (T=8)
- Essentially a deep network with weight sharing

**Autoregressive thought generation (the actual idea):**
- NEW matrix tokens generated and appended to context
- Each thought builds on ALL previous thoughts + inputs
- The thought sequence grows until the model is ready
- Unbounded compute — harder problems get more thinking
- Rank increases across the thought sequence (complexity builds up)
- This IS "matrix thinking" — reasoning in abstract matrix space

## Why This Requires Matrices (Not Vectors)

A sequence of generated vector thoughts would be flat — each thought is just d numbers.
There's no measurable notion of "complexity increasing." Dimension is fixed.

A sequence of matrix thoughts has RANK as a measurable complexity axis:
- thought_1: rank 2 (considering two hypotheses)
- thought_3: rank 5 (five interacting patterns)
- thought_7: rank 8 (detailed, multi-faceted understanding)

Rank increase across thoughts = the model building richer abstractions.
This is structurally impossible with flat vectors.

The multiplicative composition (I+Δ)·M·(I+Γ) naturally builds rank:
each application can increase rank by combining row and column perturbations.
Applied autoregressively across thoughts, this creates a CHAIN of increasing
complexity — each thought is richer than the last.

## Architecture Components

### Byte Embedding
Outer product: byte b → u_b ⊗ v_b (16-dim vectors → 16×16 rank-1 matrix).
256 possible bytes, tiny embedding tables.

### Thought Generation
A shared transformer block that processes the growing sequence:
- Attention: Frobenius (fast, SDPA-compatible) over all inputs + previous thoughts
- Thinking layer: (I+Δ)·M·(I+Γ) multiplicative composition (builds rank)
- Each thought position is initialized from a learned "thought seed" + step embedding

The block is applied ONCE per new thought (not iteratively to all positions).
This is standard autoregressive generation but in matrix space.

### Halt Mechanism
Options (to be tested):
1. Fixed budget: N=4 thoughts per output byte (start here)
2. Certainty threshold: halt when prediction confidence exceeds threshold
3. LoopFormer-style consistency: train at various N, eval at any N
4. Rank-based: halt when rank stops increasing (thought has stabilized)

### Output
At d=16 with byte vocab: flatten the final thought matrix → 256 logits.
257 params total (256 bias + 1 temperature).
The matrix IS the byte prediction.

## What We Proved (22 experiments)

### Findings that support this architecture:
- **Rank enrichment is real.** Model spontaneously builds richer representations during
  thinking (rank 5→6). This should translate to rank increasing across thoughts.
- **Outer-product embedding is better at T=1.** Matrix representations encode more per
  parameter than flat vectors. Good starting point for thought generation.
- **MultiProbeHead unlocks thinking benefit.** The output mechanism determines whether
  the model enriches or solidifies. The zero-param output should enable enrichment.
- **Frobenius attention works.** Fast, SDPA-compatible, no memory issues at d=16.

### Findings that constrain this architecture:
- **3D attention is a dead end.** Use Frobenius, not matrix-product attention.
- **Matrix operations are 130× more parameter-efficient per layer** but ~10× slower
  per FLOP than vector operations. Each thought step is cheap in params, expensive in compute.
- **PonderNet halting collapses.** Don't use learned halting probability. Use fixed budget
  or consistency training.
- **At 218K params, the model barely learned unigram statistics (BPB 3.56).**
  Need ~1M+ params for the thoughts to be meaningful.

## Implementation Status

### Built and tested:
- Outer-product byte embedding (d=16, vocab=256)
- Frobenius attention on matrix tokens
- Multiplicative thinking layer (I+Δ)·M·(I+Γ)
- MultiProbeHead and zero-param byte output (257 params)
- DDP training on 8×H100

### NOT built (the core mechanism):
- CoCoMix-style thought interleaving (insert matrix thoughts at a specific layer)
- Loss masking on thought positions (only loss on output bytes)
- Thought generation from hidden states (not from a seed — seeds were attacked and killed)
- Variable N thoughts per byte

### What was KILLED by pre-experiment analysis:
- Pure autoregressive generation from scratch (rank doesn't increase, gradients vanish)
- "Unbounded" thought count (batching problem, ends up fixed-N anyway)
- Generating thoughts from a learned seed (all thoughts converge to same rank)
- One-at-a-time thought generation (2000× more expensive than all-at-once)

### The RIGHT approach (from code analysis of 5 systems):
CoCoMix-style interleaving: process bytes through layers 1-K, INSERT thought
matrices generated FROM hidden states, then process bytes+thoughts together
through layers K+1-N. All at once in a single forward pass. Loss only on bytes.
This is proven to work from scratch (CoCoMix: 21.5% sample efficiency gain).

### Research completed:
- How to train thinking (COCONUT curriculum, Quiet-STaR, CoCoMix, Thoughtbubbles)
- How to prevent thought collapse (Seq-VCR variance-covariance regularization)
- Long context for bytes (MBLM hierarchy, Mamba, EverMind MSA)
- Cross-domain transfer literature (bGPT, MBLM, Perceiver IO)
- All saved in research/ directory (12 documents)

## Experiment Plan

### Experiment 1: Scale byte model to competence
1M+ params, 32+ layers, single-domain text bytes. Target: <2.5 BPB.
Must work on one domain before attempting multi-domain or thought generation.

### Experiment 2: Autoregressive thought generation (fixed budget)
Same model but with N=4 thought tokens generated before each output byte.
Measure: does BPB improve vs no-thought baseline? Do thought ranks increase?

### Experiment 3: Variable thought budget
Train with random N (1-8 thoughts). LoopFormer-style consistency loss.
Measure: does the model learn to use more thoughts for harder bytes?

### Experiment 4: Multi-domain bytes
Text + images + audio as raw bytes. Measure cross-domain transfer.
Compare matrix thoughts vs flat vector thoughts.

## Key References
- COCONUT (Meta 2024): continuous thought training curriculum
- CoCoMix (Meta 2025): trains continuous thoughts from scratch
- Seq-VCR (ICLR 2025): prevents representation collapse in thought tokens
- Thoughtbubbles (Stanford 2025): unsupervised thought learning from LM loss
- LoopFormer (ICLR 2026): shortcut consistency for variable-depth eval
- MBLM (IBM 2025): hierarchical byte model, 5M byte context
- EverMind MSA (2026): 100M token context via memory sparse attention
- bGPT (2024): multi-domain byte training, cross-domain transfer results
