# Project Analysis: Matrix-Valued Token Representations
**Date:** 2026-03-26
**Scope:** Comprehensive assessment of all experimental results through March 2026

---

## 1. What We Actually Proved

### Findings That Would Survive Peer Review

**1a. The outer-product matrix embedding is a superior token representation at T=1.**

This is the strongest, most reproducible result. At a single iteration (no iterative refinement), the matrix model consistently outperforms matched-parameter vector baselines:

| Comparison | Matrix T=1 | Vector/LoopFormer T=1 | Advantage |
|---|---|---|---|
| Round 1 (5.15M params) | PPL 723 | PPL 8,274 (vector) | 11x |
| Round 1 (5.15M params) | PPL 723 | PPL 24,588 (LoopFormer) | 34x |
| Round 2 (5.16M params, MultiProbeHead) | PPL 141 / BPB 2.12 | PPL 24,588 / BPB 4.29 (LoopFormer) | 175x PPL, 2x BPB |
| Run 18 ablation (d=16) | BPB 2.18 | BPB 3.22 (flat-vector, 10x params) | Better at 10x fewer params |

This held across every configuration tested: different mat_dims, different data mixes, different head architectures. A reviewer would accept this. The outer product u x v creates d^2 structured values from 2d parameters, and the structure (not just the parameter count) appears to matter.

**1b. Iterative matrix refinement provides consistent improvement.**

Thinking benefit grew across all training runs once PonderNet halting was removed:

| Run | T=1 BPB | T=8 BPB | Thinking Benefit |
|---|---|---|---|
| Run 9 (step 1500) | PPL 73.3 | PPL 66.1 | 9.8% |
| Round 1 (d=32) | PPL 723 | PPL 638 | 11.8% |
| Round 2 (MultiProbeHead) | PPL 141 | PPL 72.4 | 48.6% |
| Run 15 (optimized d=32) | BPB 2.12 | BPB 1.72 | 64.4% (BPB improvement 19%) |

The benefit accelerates during training (Run 9: 0.8% at step 500, 5.1% at 1000, 9.8% at 1500). This is a genuine signal, not noise.

**1c. Rank enrichment is an emergent property under MultiProbeHead.**

With the MultiProbeHead (no vector collapse), thinking iterations increase effective rank: [5.02 -> 5.41 -> 5.67 -> 5.83 -> 5.93 -> 6.02 -> 6.09 -> 6.12] in Round 2. This reversed the solidification pattern seen with the old vector-collapse head and coincided with the best BPB results. This is a novel empirical observation -- no prior work reports rank dynamics of matrix-valued representations during iterative refinement.

**1d. PonderNet halting collapses at small scale.**

Run 8 showed expected_steps=1.00 after training. This confirms the known failure mode (gradient signal from later steps vanishes when early halting probability dominates) and validates the decision to switch to fixed iterations.

### Findings That Are Suggestive But Not Conclusive

**1e. The 3D attention solidification vs. Frobenius enrichment result.**

Run 21 showed 3D attention produces solidification (rank drops 2.75 -> 2.66) and worse BPB (2.457 vs 1.906 for Frobenius). But this was a single comparison with potentially confounded variables (throughput differences meant different effective training). The direction is clear, but the mechanism needs more controlled ablation.

**1f. Batch size correlates with richer ranks.**

The incidental finding that bigger batches produce higher effective ranks (b=112: start rank 6.00, b=32: start rank 4.94) is interesting but based on only three data points at the same training step. Could be gradient diversity or could be an artifact of batch normalization dynamics.

### What Is Noise

**1g. The early PHM/segmentation experiments (Runs 1-7) are a dead end for this architecture.**

The PHM layers converged to nilpotent algebra (not quaternions), and learned segmentation was worse than fixed-stride segmentation (Run 7: learned seg BPC 3.20 vs fixed-stride BPC 2.00). The "emergent domain differentiation" in Run 5 (text vs image segment variance diverging) is a real phenomenon but in the wrong architecture -- it does not transfer to the matrix thinking framework.

**1h. The optimized d=32 model's BPB improvement over baseline d=32 is marginal.**

Run 15 (shared gates + low-rank r=8) got BPB 1.72 vs Round 2's BPB 1.67 -- 3% worse while being 12% faster. This is engineering optimization, not a scientific finding.

---

## 2. The Representation vs Operations Question

This is the single most important open question in the project, and the data so far points strongly toward one answer: **the representation is doing most of the work**.

### The Evidence

**Run 18 (the critical ablation):** Same outer-product embedding, flattened to 256-dim vector, standard transformer operations. Results:

- T=1 BPB: 3.219 (flat-vector) vs 2.18 (matrix ops) -- matrix ops win by 32%
- T=8 BPB: 1.011 (flat-vector, 24M params) vs 1.91 (matrix ops, 2.4M params) -- flat-vector wins, but with 10x more parameters

The critical number: **at T=1, the matrix model at 2.4M params beats the flat-vector model at 24M params** (BPB 2.18 vs 3.22). This means the matrix structure itself -- the fact that the model sees a 16x16 grid of structured values rather than a flat 256-dim vector -- provides an advantage that cannot be recovered simply by adding parameters to a vector model.

### What This Means

The outer product embedding creates rank-1 matrices where the row and column structure encodes token identity in a geometrically meaningful way. When the model operates on these as matrices (preserving row-column structure through RowThenCol projections), it can exploit this structure. When the model flattens them to vectors, the structure becomes just another set of features to learn correlations over -- which works, but less efficiently.

However, at T=8 with enough parameters (10x more), the flat-vector model reaches BPB 1.011, substantially better than the matrix model's BPB 1.91. This suggests that **the matrix operations (RowThenCol, multiplicative composition) are not efficient enough to justify their O(d^3) cost**. A sufficiently large flat-vector model with the same embedding can overcome the structural disadvantage through brute-force parameter count.

### The Uncomfortable Conclusion

The strongest publishable finding may be: "Outer-product matrix embeddings provide a structured initialization advantage that persists even when subsequent processing uses standard vector operations, but matrix-native operations (RowThenCol) are too computationally expensive to outperform standard transformers at matched FLOPs."

This is not the result the project was hoping for. But it is honest, and it is itself a contribution -- it tells the field where to look (embeddings) and where not to look (expensive matrix operations) for gains from structured representations.

### The Missing Ablation

What has NOT been tested: outer-product embedding + standard vector ops + **matched parameters** (not 10x). If a 2.4M param flat-vector model with outer-product embed gets close to the matrix model's T=8 BPB 1.91, then the operations thesis is definitively dead. If it cannot, the matrix operations are earning their keep at matched params even though they lose at matched FLOPs. This ablation is essential before any publication.

---

## 3. Why Enrichment Beats Solidification

### The Empirical Pattern

| Attention Type | Rank Trajectory | T=8 BPB | Verdict |
|---|---|---|---|
| 3D Matrix Product | 2.75 -> 2.66 (drops) | 2.457 | Solidification hurts |
| Frobenius (MultiProbeHead) | 5.02 -> 6.12 (rises) | 1.67 | Enrichment helps |
| Frobenius (old collapse head) | drops | PPL 638 | Solidification hurts |

The pattern is consistent: when the model's representations become more complex (higher rank) during iterative refinement, predictions improve. When they simplify (lower rank), predictions suffer.

### Why This Makes Sense for Next-Token Prediction

Next-token prediction is fundamentally about maintaining uncertainty over possibilities until the final moment of commitment. A rank-1 matrix encodes a single hypothesis (one direction of variation). A rank-6 matrix encodes six independent directions of variation -- six simultaneous hypotheses about what the token could mean in context.

Consider predicting the next word after "The bank was...": the model needs to simultaneously consider "closed" (financial), "steep" (river), "robbed" (crime), "foggy" (weather), etc. Each possibility is a rank-1 direction. The model needs rank >= 4 to keep these alive through iterative refinement. Solidification (rank dropping) prematurely commits to one interpretation.

3D attention drives solidification because it computes per-pair coupling matrices, creating strong pairwise constraints that force consistency (and thus rank reduction). Frobenius attention computes scalar scores (like standard attention) and lets the multiplicative thinking layer handle refinement without over-constraining.

### The Architectural Implication

The MultiProbeHead was the key unlock. The old output head collapsed matrices to vectors before prediction, which meant the model was rewarded for producing low-rank representations (easier to collapse cleanly). MultiProbeHead reads both row and column structure from the matrix, which rewards the model for maintaining rank. The output head determines the incentive landscape for the entire model.

### What This Tells Us About Representation Learning

This finding connects to the broader principle that **internal representations should be richer (higher capacity) than outputs**. The model's job during thinking is to accumulate information, not to discard it. Solidification is premature optimization -- it throws away information before the model has decided what to predict. The literature supports this: COCONUT's continuous thoughts maintain superposition of multiple reasoning paths (BFS-like behavior), and Seq-VCR's anti-collapse regularization dramatically improves reasoning.

---

## 4. The Compute Efficiency Problem

### The Numbers

| Model | T/L=8 BPB | FLOPs/step | Time (matched steps) | FLOPs-Matched BPB |
|---|---|---|---|---|
| Matrix d=32 (Round 2) | 1.67 | ~218 TFLOPs | 169 min | 1.67 |
| LoopFormer (3K steps) | 1.27 | ~6.8 TFLOPs | 6 min | -- |
| LoopFormer (FLOPs-matched, 96K steps) | -- | ~6.8 TFLOPs | 162 min | 0.87 |

The matrix model burns ~32x more FLOPs per step. At matched compute budget, LoopFormer achieves BPB 0.87 vs the matrix model's BPB 1.67 -- nearly 2x better. The gap is not close.

### Where the FLOPs Go

The ARCHITECTURE.md FLOPs table tells the story:

| Component | Matrix (d=32) | After All Optimizations (d=16) | LoopFormer |
|---|---|---|---|
| Per-token FLOPs | 1,300M | 36M | 1.2M |
| Ratio to LoopFormer | 1,000x | 30x | 1x |

Even with every planned optimization (shared projections, low-rank, d=16, low-rank attention), the matrix model is still 30x more expensive than LoopFormer. The fundamental cost is O(d^3) for matrix multiplication vs O(d^2) for vector operations, multiplied across all RowThenCol projections and multiplicative composition steps.

### Is There a Path to Competitiveness?

**Option A: Accept the embedding win, use vector operations.** This is the pragmatic path. Keep the outer-product matrix embedding (which provides the T=1 advantage), flatten to a vector, and use standard efficient operations. The Run 18 ablation shows this works at 10x params; the question is whether it works at matched params.

**Option B: Sparse matrix operations.** If the effective rank of most representations is 5-6 out of 16 or 32, then full dense matrix multiply is wasteful. Low-rank matrix operations (project to rank-r, operate, project back) could reduce cost from O(d^3) to O(d^2 * r). This has not been tested.

**Option C: Accept the compute cost and scale data.** The LoopFormer diverged at step 52K from overfitting (40 epochs on 2.2B tokens). With enough data (say 50B tokens), neither model would overfit, and the per-step quality advantage of matrix operations might compound. This is speculative and expensive to test.

**Option D: Exploit the d=16 byte-level structure.** At d=16 with byte vocab, the embedding table becomes tiny (256 tokens x 32 params = 8K params), putting 33-57% of parameters into thinking. This fundamentally changes the parameter allocation problem. The question is whether the thinking layers can do enough at this scale.

### My Assessment

Option A is the most defensible. The representation advantage is proven; the operations advantage is not, and the compute cost is prohibitive. The publishable contribution is the embedding structure, not the matrix operations. However, Option D deserves serious exploration because it changes the economics: byte-level models with tiny vocabularies spend almost all their parameters on processing, which is where matrix structure might earn its keep.

---

## 5. The Byte-Level Direction

### What We Have

Run 19: first byte-level matrix thinker.
- d=16, vocab=256 (raw bytes), 218K params total
- 33% of params in thinking (up from 1-4% at BPE vocab)
- T=8 BPB: 3.56, thinking benefit: 7%
- Data: 539M bytes (WikiText-103 + Python code)
- Throughput: 260K tok/s on 8xH100

### Is BPB 3.56 Promising?

For context:
- Shannon entropy of English text is ~1.0-1.3 BPB
- A character-level LSTM (2016) achieves ~1.25 BPB on enwiki8
- EvaByte (the first byte-level model matching BPE performance) operates at much larger scale
- Random prediction would give log2(256) = 8.0 BPB
- A simple bigram model on English text gets ~5.0-6.0 BPB

BPB 3.56 at 218K params with 7% thinking benefit is not competitive with any published result. But the parameter count is absurdly small -- 218K is less than most ResNets' first layer. The real question is scaling behavior.

### What Scale Would Be Competitive?

EvaByte and MambaByte show that byte-level models need roughly 3.5x more tokens than BPE models to see equivalent text (each character is 1-4 bytes vs 1 BPE token covering ~3.7 characters). This means:

- To match a 14M-param BPE model trained on 300B tokens, a byte-level model needs ~1T bytes
- At d=16, a reasonable byte-level matrix model might have 1-5M params (all in thinking)
- Training budget: 1T bytes at 260K tok/s = ~44 days on 8xH100

This is feasible on cloud hardware but expensive. The question is whether the matrix structure provides any advantage at this scale over a standard byte-level transformer, which has not been demonstrated.

### The d=16 x 256 Alignment

The observation that 16^2 = 256 = number of possible bytes is elegant and enables the zero-parameter output head (read M[i,j] directly as the logit for byte i*16+j). This is genuinely novel. However, it forces d=16, which caps the expressiveness of matrix operations. Whether rank dynamics within a 16x16 matrix (max rank 16, typical effective rank 5-6) are sufficient for complex language modeling is unknown.

### My Assessment

The byte-level direction is the project's best shot at a distinctive contribution. The standard NLP benchmark path (BPE tokens, comparison to Pythia/LoopFormer) is a losing game because the compute efficiency gap is too large. The byte-level path offers:
1. A unique structural alignment (d=16 = sqrt(256))
2. Domain agnosticism (text, code, images, audio all as bytes)
3. A zero-parameter output head (matrix IS prediction)
4. The parameter allocation shift (most params in thinking, not embedding)

But it needs significant scale-up (10x-100x current size) and honest comparison against byte-level baselines (MambaByte, EvaByte) to be publishable.

---

## 6. What Should the Architecture Actually Be?

Given everything learned, here is what the next version should look like, component by component.

### Keep

1. **Outer-product matrix embedding.** u x v creates d^2 structured values from 2d parameters. This is the strongest finding. Keep it.

2. **MultiProbeHead output.** K bilinear probes reading row-and-column structure from the matrix. This unlocked rank enrichment. Never go back to vector collapse.

3. **Frobenius attention.** Scalar scores via flattened dot product. Flash-attention compatible. 3D attention is slower and worse.

4. **Fixed iterations (no PonderNet).** PonderNet collapses at this scale. Use T=8 fixed, with shortcut-consistency training (LoopFormer-style) so the model produces useful output at any depth.

5. **Gradient checkpointing per iteration.** Memory-efficient, enables T=8 without blowing up VRAM.

### Change

1. **Replace RowThenCol multiplicative composition with cheaper alternatives.** RowThenCol is the main FLOPs bottleneck. Test:
   - (a) Standard FFN on flattened matrix (preserving the matrix embedding but using cheap operations)
   - (b) Low-rank multiplicative: (I + U_delta @ V_delta) * M * (I + U_gamma @ V_gamma) with r=4
   - (c) Householder reflections: 2-4 reflections per side instead of full projection

2. **Shrink to d=16 for byte-level.** Accept the quality gap vs d=32 (Run 16-17 showed ~15% BPB loss at d=16). The structural alignment with byte vocab (16^2=256) and 8x compute savings are worth it.

3. **Add shortcut-consistency training.** From LoopFormer: timestep conditioning via Fourier features + AdaLN, consistency loss between shortcut trajectories and full trajectories. This replaces PonderNet and enables variable-depth inference.

4. **Add Seq-VCR regularization on thought matrices.** Variance-covariance regularization prevents rank collapse without requiring an output head that forces it. Hyperparameters from the literature: lambda_var=0.1-1.0, lambda_cov=0.004-0.5.

### Add

1. **Byte vocabulary (256 tokens).** Tiny embedding table (~8K params at d=16), domain agnostic. Almost all parameters go into thinking.

2. **Zero-parameter matrix output head.** Read M[i,j] directly as logit for byte (i*16+j). Novel and parameter-free.

3. **Multi-token lookahead loss.** From Quiet-STaR: predict not just the next byte but the next 2-4 bytes. Forces thought representations to encode longer-range information.

4. **Thought matrix slots (Phase 4).** N=2-4 blank matrix positions before each prediction. Loss only on byte predictions. Train with curriculum: start with 0 slots, add them after base model converges.

### The Concrete Architecture

```
INPUT:  raw bytes (vocab=256)
  |
EMBED:  byte -> u(16) x v(16) = rank-1 matrix M (16x16)
  |     Position: M += 0.1 * (pu x pv)
  |     Total: 256 * 32 + max_len * 32 = ~40K params
  |
THINK:  [optional: N thought matrix slots, no output, filled by model]
  |
PROCESS: shared layers x T iterations (T=8 train, T=2-8 inference)
  |     Attention: Frobenius scores (flash-compatible SDPA)
  |     Thinking: (I + low-rank-delta) * M * (I + low-rank-gamma) + additive
  |     Shortcut-consistency: timestep conditioning + consistency loss
  |     Seq-VCR: variance-covariance regularization on representations
  |
OUTPUT: M[i,j] = logit for byte (i*16 + j)
  |     Zero extra parameters
  |
LOSS:   CE on byte predictions + 0.1*consistency + 0.01*SeqVCR
        Loss on next byte + next-2 + next-3 (multi-token lookahead)
```

Target: ~500K-2M total params, 50-80% in thinking layers.

---

## 7. The Freeform Thinking Vision

### The Vision Restated

Matrix thought tokens generated before byte predictions. The model decides when to think (more thoughts before hard predictions, fewer before easy ones). Thoughts are matrices in the same space as byte representations but produce no output -- they exist purely to help downstream predictions.

### What the Literature Says

| System | Key Lesson for Us |
|---|---|
| COCONUT | Curriculum is essential. Cannot train latent thoughts from scratch. Must scaffold. |
| Pause Tokens | Must train with thought slots from the start. Retrofitting fails. |
| CoCoMix | Only system training continuous thoughts from scratch at scale. Concept prediction loss helps. |
| Thoughtbubbles | Unsupervised adaptive compute from LM loss alone is possible. Score-weighted masking. |
| Quiet-STaR | Mixing head prevents thought destruction. Multi-token lookahead helps. |
| Seq-VCR | Variance-covariance regularization prevents collapse. Dramatic impact on reasoning. |
| Fast Quiet-STaR | Curriculum to compress thoughts: start with many, reduce over training. |
| Soft Thinking | Entropy/rank-based stopping for adaptive compute. |

### Is It Feasible at Small Scale?

The honest answer is: **partially, with significant constraints**.

What works at small scale:
- Fixed thought slots with loss masking (Pause Tokens showed this at 130M params)
- Variance-covariance regularization (Seq-VCR showed this with GPT-2 Small, ~124M params)
- Score-weighted attention masking (Thoughtbubbles works at 150M params)

What does NOT work at small scale:
- Adaptive thought allocation (PonderNet collapsed at our scale; MoR requires 1B+ params for clean routing)
- Training latent thoughts from scratch without any curriculum (COCONUT showed this fails)
- REINFORCE-based credit assignment (high variance, requires large batches)

### The Recommended Training Procedure

**Stage 0 (Steps 0-50K): Standard byte-level matrix LM.** No thought slots. Establish base predictions. This is the foundation -- if base predictions are bad, thoughts have nothing to improve.

**Stage 1 (Steps 50K-100K): Add N=2 fixed thought slots.**
- Slots inserted before each prediction position
- Loss only on byte predictions (masked on thought positions)
- Mixing gate initialized to 0.95 (almost entirely trust base predictions)
- Seq-VCR regularization on thought matrices (lambda_var=0.1, lambda_cov=0.01)
- Residual accumulation: prediction = base_prediction + gate * thought_contribution
- Reset optimizer at transition

**Stage 2 (Steps 100K-200K): Increase to N=4, reduce mixing bias.**
- Mixing gate bias reduced to 0.7
- Seq-VCR strengthened (lambda_var=0.5, lambda_cov=0.05)
- Multi-token lookahead (predict next 3 bytes, not just next 1)

**Stage 3 (Steps 200K+): Full integration.**
- Mixing gate free to learn
- Monitor thought matrix rank (should be 3-8), mixing gate values (should be 0.2-0.8)
- If any thought slot rank < 2 for 1000 consecutive steps, reinitialize its parameters

**Adaptive allocation (much later, 1M+ steps):** Replace fixed N with expert-choice routing (MoR-style). Linear router + sigmoid scores tokens by prediction difficulty. Hard tokens get 4-8 thought slots, easy tokens get 0-1. This requires the base model to already produce useful thought matrices.

### The Key Risk

The model may learn to ignore thought slots entirely (mixing gate stays near 1.0, thoughts become zero matrices). Every system in the literature has encountered this. The defenses are:
1. Residual accumulation (thoughts add, never replace)
2. Seq-VCR regularization (forces non-trivial thought representations)
3. Multi-token lookahead (thoughts that help predict farther ahead get stronger gradients)
4. Curriculum (gradual introduction prevents catastrophic interference)
5. Rank monitoring with reinitialization (mechanical last resort)

If all five defenses fail simultaneously, then freeform matrix thinking does not work at this scale, and that is itself a finding.

---

## 8. What Would Make This a Paper?

### The Publishable Story

**Title:** "Matrix-Valued Token Representations: Structured Embeddings for Efficient Language Modeling"

**Core claim:** Outer-product matrix embeddings encode tokens as rank-1 d x d matrices, creating d^2 structured values from 2d parameters. At a single processing step (no iterative refinement), this representation outperforms flat vectors at 10x the parameter count. The structured embedding enables rank-based iterative refinement where representations grow in complexity during processing, and a zero-parameter output head where the matrix is read directly as byte prediction logits.

### The Evidence Needed

**Already have (strong):**
- T=1 representation advantage across multiple configurations (Runs 10, 12, 18)
- Rank enrichment dynamics under MultiProbeHead (Run 12)
- 3D attention solidification vs. Frobenius enrichment comparison (Runs 20-21)
- PonderNet collapse at small scale (Run 8)

**Need to run (critical):**
1. **Param-matched flat-vector ablation.** Run 18 used 10x params. Must test outer-product embed + flat vector ops at SAME param count as the matrix model. This is the make-or-break experiment for the operations thesis.
2. **Standard benchmarks.** All current results are training BPB on a custom data mix. Need evaluation on enwik8, text8, or WikiText-103 test set with proper train/val/test splits. No paper survives review without standard benchmarks.
3. **Comparison to published models.** Pythia-14M, GPT-2 124M at matched compute. Show where matrix embeddings help and where they don't.
4. **Byte-level evaluation.** If pursuing the byte-level angle: comparison to MambaByte, EvaByte, or standard byte-level baselines on established benchmarks (enwik8 BPC, text8 BPC).

**Nice to have:**
- Scaling curves (does the representation advantage grow or shrink with model size?)
- Visualization of learned matrix structure (what do the rank-1 embeddings look like? do similar bytes get similar row/column vectors?)
- Ablation of rank-enrichment vs. forced-solidification at matched compute

### What Venue?

**If the param-matched ablation confirms the representation advantage:**
- ICML 2027 or NeurIPS 2027 workshop paper (contributions: novel embedding, rank dynamics analysis, byte-level output head)
- Full paper if scaling curves show the advantage persists at 100M+ params

**If the param-matched ablation shows operations don't help:**
- Still publishable as a negative result / analysis paper at a workshop
- "Structured embeddings help, structured operations don't: separating representation from computation in matrix-valued language models"

**If byte-level results become competitive:**
- Could target COLM 2027 or EMNLP 2027 with the byte-level angle
- The zero-parameter output head (matrix IS prediction) and domain agnosticism are strong hooks

### The Honest Assessment

The project is currently between "interesting preliminary results" and "paper." The gap is:
1. Proper evaluation (standard benchmarks, proper splits)
2. One critical ablation (param-matched flat-vector)
3. Scale (5M params is toy-scale; reviewers will want to see 14M+ minimum)

These are all achievable with the available hardware (8xH100). Estimated time: 2-3 weeks of focused experimentation.

### What Would NOT Make a Paper

- Claiming the matrix operations are superior without the param-matched ablation
- Claiming competitive language modeling performance (LoopFormer wins at matched FLOPs)
- Claiming the rank dynamics are "understood" (we observe them but don't have a theory)
- Claiming the byte-level model is competitive (BPB 3.56 is far from SOTA)

The strength of the project is the representation insight (outer-product embeddings) and the empirical phenomena (rank enrichment, enrichment-beats-solidification). The weakness is compute efficiency and scale. The paper should lean into the strengths and be honest about the weaknesses.

---

## Summary of Recommendations

1. **Run the param-matched flat-vector ablation immediately.** This determines whether the paper's story is "matrix embeddings AND operations" or "matrix embeddings only."

2. **Evaluate on standard benchmarks.** enwik8, text8, or WikiText-103 test set. No shortcuts.

3. **Scale to 14M params minimum.** Reviewers will dismiss 5M-param results as toy experiments.

4. **Pursue the byte-level direction.** The structural alignment (d=16, vocab=256, zero-param output head) is the most novel contribution. Standard NLP benchmarks at BPE token scale are a losing game against LoopFormer.

5. **Implement shortcut-consistency training.** This replaces PonderNet, enables flexible-depth inference, and aligns with LoopFormer (the primary baseline).

6. **Be honest about compute efficiency.** Do not claim competitive FLOPs. Claim a representation insight that could be combined with efficient architectures.

7. **For freeform thinking: use the staged curriculum.** Start without thoughts, add fixed slots with high mixing bias, gradually integrate. Monitor rank. Apply Seq-VCR.

8. **Write the paper around the embedding, not the operations.** The outer-product matrix embedding is proven. The matrix operations are not (at matched FLOPs). Lead with what you can defend.
