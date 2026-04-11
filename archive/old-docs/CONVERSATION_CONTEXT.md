# Conversation Context — Everything the Next Agent Needs to Know

## The User's Background
- Never done ML before this project. Learning from first principles.
- Strong intuition about architecture design — most novel ideas in this project
  came from the user, not from the literature.
- Wants rigorous experimental process so results can scale to H100s with confidence.
- Gets frustrated when the agent repeats dead ideas, makes unsupported claims,
  or runs experiments sloppily.
- Prefers honest assessment over optimism. Negative results are data.

## Key Ideas That Came From the User (Not Literature)

1. **"What if thinking was in matrices before output in vectors"**
   The user proposed this before any research confirmed it. The literature has
   vector-valued thinking (COCONUT, PonderLM) but nobody had matrix-valued
   thinking with rank convergence.

2. **"Two modes — matrix thinking and vector speaking"**
   The user's vision: the model thinks in matrices (abstract, unresolved),
   speaks in vectors (crystallized, expressible), and switches between modes
   based on a running average of certainty. Like how humans think abstractly
   then find words, then think again when they hit something hard.

3. **"Running average of certainty drives mode switching"**
   Not per-token decisions (which is what every paper does). The user wanted
   GROUP-LEVEL uncertainty — if the model has been uncertain for several tokens,
   THEN switch to deep thinking. Confirmed novel by research agent. Has a
   neuroscience analog (locus coeruleus adaptive gain, Aston-Jones & Cohen 2005).

4. **"Why flatten? Keep it as a matrix the whole way through"**
   The user pushed hard against flattening. This led to the RowThenCol and
   matrix-native projection research. The current architecture has zero
   flattening in the compute path — confirmed by grep.

5. **"The matrices should be in the context like real tokens"**
   True autoregressive thought generation — thoughts APPENDED to the sequence,
   not refined in place. This is how the current matrix_thinker.py works.

6. **"Don't cap the thoughts — let them go to 200"**
   The user wanted to see natural behavior, not constrained behavior.

## The Evolution of the Project

### Phase 1: PHM + Byte-Agnostic (DEAD)
Started with Parameterized Hypercomplex Multiplication (ICLR 2021) + learned
byte segmentation. PHM converged to nilpotent structure. Learned segmentation
lost to fixed stride. All archived in `archive/` and `byte-agnostic/`.
DO NOT REVISIT.

### Phase 2: Matrix Representations (VALIDATED)
Built matrix-valued token models. V1 (bilinear) didn't beat vectors.
V2 (multiplicative composition) DID beat vectors (1.94 vs 2.03 BPC).
Key finding: the operation matters more than the representation.

### Phase 3: Matrix Thinking (CURRENT)
The autoregressive matrix thinker with:
- True thought appending
- 3D matrix product attention
- PonderNet halting with rank bias
- RowThenCol projections (no flattening)
- Trained on reasoning data
- Thoughts solidify (confirmed on Mac Mini and H100)

### Phase 4: Two-Mode Model (NOT BUILT)
Matrix thinking mode + vector speaking mode. Certainty-driven switching.
Same model, same weights, same context. This is the user's ultimate vision.
See ADAPTIVE_THINKING.md for the design.

## What Was Proven

| Finding | Status |
|---------|--------|
| Matrix V2 beats vectors at matched params | CONFIRMED (1.94 vs 2.03 BPC) |
| Thoughts solidify (rank decreases through thinking) | CONFIRMED (Mac Mini + H100) |
| 3D matrix product attention works | CONFIRMED (trains, gradients flow) |
| Matrix-native ops (no flatten) work | CONFIRMED (all tests pass) |
| Starting rank increases during training | CONFIRMED (model learns richer thoughts) |
| Architecture is computationally feasible | CONFIRMED (research agent + H100 run) |
| The idea is novel | CONFIRMED (exhaustive literature search) |

## What Was NOT Proven

| Question | Status |
|----------|--------|
| Does matrix thinking beat vectors at scale on reasoning? | H100 run in progress |
| Do harder problems trigger more thinking steps? | Not measured yet |
| Does the two-mode model work? | Not built |
| Does certainty-driven switching work? | Not built |
| Does this scale to 100M+ params? | Not tested |
| Can the model generate coherent text? | No inference script built |

## Dead Ends (Don't Repeat)

- **PHM / learned algebra** — converges to nilpotent. Archived.
- **Gumbel-softmax segmentation** — collapses. Use top-K or fixed stride.
- **Learned structure at small scale** — fixed structure wins below ~8B params.
- **Bilinear projections (A@M@B)** — not expressive enough. Use RowThenCol.
- **Householder projections** — worst mini PPL. Use RowThenCol.
- **Kronecker projections** — old PHM idea. RowThenCol beat it.

## Activation Functions
The field has standardized on SiLU (inside SwiGLU pattern). Our model uses
SiLU in the RowThenCol projections (silu(A@M)@B) and in the gate*value
SwiGLU pattern for delta/gamma computation. This is current standard.
GELU is old. Do not use GELU.

## ML Concepts Covered with the User
The user has been taught (with real math):
- Vectors, matrices, tensors, shapes, strides
- Linear layers (y = Wx + b)
- Activation functions (ReLU, sigmoid, SiLU, SwiGLU) and why they matter
- Loss functions (cross-entropy, softmax)
- Backpropagation (chain rule)
- Adam optimizer
- BPC (bits per character)

NOT YET COVERED:
- Embeddings
- Attention mechanism
- Full transformer architecture
- How our architecture differs from standard transformers

The user learns fast and asks sharp questions. Explain with real math,
not analogies. Verify technical claims with research agents.

## Files the Agent Should Also Read

### Additional research docs
- `research/publishability-assessment.md` — honest critical review of findings
- `research/imposed-vs-learned-structure.md` — "Structure Learning Tax" thesis
- `matrix-thinking/H100_EXPERIMENT_QUEUE.md` — full scale-up plan
- `matrix-thinking/EXPERIMENT_VARIABLES.md` — every knob to test

### Scripts that work and have been tested
- `matrix-thinking/h100_scripts/exp1b_vector_baseline.py` — vector transformer
- `matrix-thinking/h100_scripts/exp1c_matrix_v2_flatten.py` — matrix with flatten
- `matrix-thinking/h100_scripts/exp_solidification.py` — solidification experiment
- All `exp2*.py` and `exp3*.py` — operation and activation sweeps

### Key papers to understand (from our research)
- Energy-Based Transformers (arxiv 2507.02092, July 2025) — iterative refinement
- CliffordNet (arxiv 2601.06793, Jan 2026) — fixed geometric algebra, 8x efficiency
- TTT Layers (arxiv 2407.04620, July 2024) — weight matrices as hidden states
- DeltaProduct (arxiv 2502.10297, Feb 2025) — multiplicative matrix composition
- PonderNet (arxiv 2107.05407, 2021) — learned halting for adaptive compute
- COCONUT (arxiv 2412.06769, Dec 2024) — continuous chain of thought
- Token Maturation (arxiv 2601.04854, Jan 2026) — geometric stability halting
- Aston-Jones & Cohen (2005) — LC-NE adaptive gain (neuroscience analog)

## H100 Training Currently Running

```
SSH: ssh root@91.199.227.82 -p 11183 -i ~/.ssh/id_ed25519
Script: /root/run_train.py
Model: 2.46M params, mat_dim=16, 2 thinking layers, 8 thoughts per token
Data: 46M tokens OpenR1-Math reasoning
Status at last check: step 1500, val PPL 220, thoughts solidifying
Code: /root/src/ (matrix_thinker.py) and /root/h100_scripts/
Results: /root/results/
```

Check if still running: `ps aux | grep python`
Check latest output: look at terminal output or results dir
