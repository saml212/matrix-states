# Internal Reasoning Tokens: Deep Research Dive

**Date:** 2026-03-26
**Relevance:** Direct architectural predecessor to our Phase 4 (Freeform Matrix Thinking)

---

## Table of Contents

1. [STaR (2022) — The Bootstrap Loop](#1-star-2022)
2. [Quiet-STaR (2024) — Hidden Rationales at Every Position](#2-quiet-star-2024)
3. [Pause Tokens (ICLR 2024) — Dummy Compute Tokens](#3-pause-tokens-iclr-2024)
4. [COCONUT (2024) — Continuous Latent Reasoning](#4-coconut-2024)
5. [CoCoMix (2025) — Continuous Concepts During Pretraining](#5-cocomix-2025)
6. [Thoughtbubbles (2025) — Unsupervised Parallel Latent Streams](#6-thoughtbubbles-2025)
7. [Inner Thinking Transformer (2025) — Adaptive Depth from Scratch](#7-inner-thinking-transformer-2025)
8. [Seq-VCR (2024) — Preventing Collapse via Regularization](#8-seq-vcr-2024)
9. [Fast Quiet-STaR (2025) — Curriculum to Internalize Thinking](#9-fast-quiet-star-2025)
10. [Soft Thinking (NeurIPS 2025) — Continuous Concept Tokens](#10-soft-thinking-neurips-2025)
11. [Synthesis: The Credit Assignment Problem](#11-synthesis-credit-assignment)
12. [Synthesis: Preventing Lazy Thinking / Collapse](#12-synthesis-preventing-lazy-thinking)
13. [What This Means for Our Architecture](#13-what-this-means-for-our-architecture)
14. [Practical Training Recipes](#14-practical-training-recipes)

---

## 1. STaR (2022)

**Paper:** Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning" (NeurIPS 2022)
**arxiv:** 2203.14465

### The Bootstrap Loop (Exact Algorithm)

```
Initialize: pretrained model M, dataset D with (question, answer) pairs, P few-shot examples

For iteration i = 1 to N:
  1. GENERATE: For each (q, a) in D:
     - Prompt M with P examples + q
     - Generate rationale r_hat and answer y_hat

  2. FILTER: Keep only examples where y_hat == a (correct answer)
     - These go into dataset D_correct

  3. RATIONALIZE: For each (q, a) where y_hat != a:
     - Prompt M with P examples + q + "The answer is a"
     - Generate rationale r_rat given the correct answer as a hint
     - If new answer y_rat == a, add to D_correct

  4. FINE-TUNE: Retrain from ORIGINAL pretrained M on D_correct
     - (Retraining from original M each iteration prevents overfitting)

  5. REPEAT until performance saturates
```

### Key Training Details

- Model: GPT-J 6B (GPT-2 was too small to bootstrap — minimum scale matters)
- Iterations: 16 for arithmetic, 36 for GSM8K
- Per iteration: 40 training steps initially, +20% more each iteration
- 100-step warmup, constant learning rate thereafter
- Low sampling temperature (near-greedy)
- Retrain from original pretrained weights each iteration (critical for stability)

### Key Insight

The rationalization step is crucial. Without it, the model only learns from problems it can already solve, creating a feedback loop. By generating rationales conditioned on correct answers, the model bootstraps reasoning for harder problems.

### Limitations for Us

- Requires discrete text rationales (we want continuous matrices)
- Requires a pretrained model that can already generate coherent text
- Filter-by-correctness requires known-correct answers
- Does NOT work from scratch

---

## 2. Quiet-STaR (2024)

**Paper:** Zelikman, Harik et al., "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"
**arxiv:** 2403.09629

### Architecture Overview

At every token position in a sequence, the model:
1. **Think:** Generate hidden "rationale" tokens (not visible in output)
2. **Talk:** Mix post-rationale predictions with pre-rationale predictions
3. **Learn:** Use REINFORCE to improve rationale quality

### Training Procedure (Exact)

**Phase 1 — Think (Generate Rationales):**
- At each position j in the sequence, generate t thought tokens autoregressively
- Thought tokens: 8, 12, 16, or 24 tokens (fixed, not variable)
- Bounded by learnable `<|startofthought|>` and `<|endofthought|>` tokens
- Multiple rationales per position (r=2-4 candidates)
- Parallel sampling across positions using custom attention masks

**Phase 2 — Talk (Mixing Head):**
Architecture: 3-layer MLP with ReLU activation
- Input: concatenation of h_thought and h_init (2 x hidden_dim)
- Output: scalar weight w_j in [0, 1]

```
w_j = mixing_head(concat(h_thought, h_init))
log p_talk = w_j * log p_init + (1 - w_j) * log p_thought
```

This is critical: early in training, the mixing head can keep w close to 1 (trust the base model), then gradually shift toward thought-based predictions as thoughts improve. Without this, introducing random thoughts destroys the base model's predictions immediately.

**Phase 3 — Learn (REINFORCE + NLL):**

Two loss components:

NLL loss (standard):
```
L_NLL = -log p_talk(X_{j+1:j+n_true+1})
```

REINFORCE loss (policy gradient):
```
reward_j = log p_talk(X_{j+1}) - mean(log p_talk(X_{j+1}))  [across batch]
grad L_REINFORCE = -reward_j * grad log p(thought_j | context)
```

Key tricks:
- Negative rewards excluded from REINFORCE (only reward good thoughts, don't punish bad ones)
- Start/end thought token embeddings get 100x learning rate multiplier
- Sampling at temperature=1, loss computed at temperature=3
- Gradient clipping at max_grad_norm=1.0

**Multi-Token Lookahead (Teacher Forcing):**
- After generating thought at position j, predict not just token j+1 but also j+2, j+3, ..., j+n_true
- Teacher forcing: assumes correct tokens for positions between j+1 and j+n_true
- Each future position gets its own mixing weight
- Ablation: n_true > 1 helps +0.3% GSM8K, +3.1% CommonsenseQA

### Computational Overhead

Substantial. For each token in the sequence, the model generates t thought tokens x r candidates. For a 1024-token sequence with t=16, r=2, that is 1024 x 16 x 2 = 32,768 additional token generations per training step. Custom parallel attention masks help but don't eliminate the cost.

### What They Did NOT Try

- Training from scratch: "it would be valuable to understand whether these techniques work when a model is trained from scratch" — they only fine-tuned Mistral 7B
- Variable-length thoughts: "we do not support dynamically predicting when to generate, or end, a rationale"
- Continuous (non-discrete) thought tokens

### Results

- GSM8K: 5.9% -> 10.9% (Mistral 7B, fine-tuned on OpenWebMath)
- CommonsenseQA: 36.3% -> 47.2%

### Code

Official: https://github.com/ezelikman/quiet-star (patches Huggingface Mistral)
Community: https://github.com/expz/quiet-star (PyTorch Lightning, Qwen2 0.5B / OpenELM 270M)

---

## 3. Pause Tokens (ICLR 2024)

**Paper:** Goyal et al., "Think before you speak: Training Language Models With Pause Tokens"
**arxiv:** 2310.02226

### Implementation

A single learnable `<pause>` token embedding is added to the vocabulary. During training and inference, multiple copies are appended to give the model extra compute steps.

**During pretraining:**
- ~10% of sequence positions replaced with `<pause>` at random locations
- ~200 pause tokens per 2048-length sequence
- Sequence trimmed to original length after insertion

**During finetuning:**
- M_ft copies of `<pause>` appended after the input prefix, before the target
- Tested M_ft in {10, 50}

**During inference:**
- M_inf pause tokens appended after the prefix
- Outputs from pause positions are ignored

### Critical Design Decision

**Loss is NOT computed on pause token positions.** The model never tries to predict what comes after a pause — it only uses pause tokens as extra computation steps for predicting real tokens. This prevents the model from wasting capacity learning to predict meaningless targets.

### The Central Finding

Pause tokens only help when present during BOTH pretraining AND finetuning:

| Condition | Result |
|-----------|--------|
| StdPT + StdFT | Baseline |
| StdPT + PauseFT | Mixed/minimal gains |
| PausePT + StdFT | Minimal gains |
| PausePT + PauseFT | **Significant gains on 8/9 tasks** |

This means: you cannot retrofit pause tokens into a pretrained model. The model must learn to USE the extra compute from the start.

### Results (1B parameter model, C4 dataset, 200B tokens)

- SQuAD: 36.4% -> 55.9% EM (+18 points)
- CommonSenseQA: 26.9% -> 34.8% (+8 points)
- GSM8K: 7.5% -> 8.5% (+1 point)

### Parameter Overhead

One pause token embedding = 1024 parameters = 10^-6 fraction of a 1B model. Essentially free in parameters; cost is in sequence length consumed by pauses.

### Model Sizes Tested

- 1B parameters (primary)
- 130M parameters (secondary)

### Direct Relevance to Us

Our "thought matrix slots" are essentially pause tokens that are matrices instead of vectors. The key lesson: **train with them from the start**. Do not add them later. This validates our Phase 4 plan to include thought slots during pretraining.

---

## 4. COCONUT (2024)

**Paper:** Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space"
**arxiv:** 2412.06769
**Code:** https://github.com/facebookresearch/coconut

### Core Innovation: Continuous Thought Tokens

Instead of generating discrete text tokens for reasoning, COCONUT feeds the last hidden state directly back as the next input embedding:

```
Standard autoregressive: hidden -> LM head -> argmax -> embedding lookup -> next input
COCONUT:                  hidden -> [directly used as] -> next input
```

This is fully differentiable. No sampling, no REINFORCE, no straight-through estimators. Gradients flow directly through the continuous thought chain.

### Training Procedure: Multi-Stage Curriculum

This is the most important detail. You cannot jump straight to latent reasoning — the model needs to be scaffolded:

```
Stage 0: Train on complete Chain-of-Thought data (standard CoT supervision)
Stage 1: Replace first reasoning step with c continuous thoughts, remove that text
Stage 2: Replace first 2 steps with 2*c continuous thoughts
...
Stage k: Replace first k steps with k*c continuous thoughts
Final:   ALL reasoning in latent space, no text CoT remains
```

Parameter c controls how many continuous thoughts replace one language reasoning step (c=1 for logic, c=2 for math).

**Training details:**
- GSM8K: 3 stages plus final, c=2, 6 epochs initial, 3 per stage
- ProntoQA/ProsQA: 6 stages, c=1, 5 epochs each
- Optimizer state reset between stages
- Stage mixing: 30% of data from previous stages (prevents catastrophic forgetting)
- Loss masked on questions AND latent thoughts (only predict answer tokens)
- 4x A100 80GB GPUs

### The Key Insight: Breadth-First Search

Continuous thoughts can encode MULTIPLE potential next steps simultaneously. Analysis reveals the model performs implicit breadth-first search (BFS) — maintaining uncertainty across branches and progressively eliminating wrong paths. This is impossible with discrete token reasoning, which commits to one path per token.

### Results

| Task | CoT Accuracy | COCONUT Accuracy | CoT Tokens | COCONUT Tokens |
|------|-------------|-----------------|------------|---------------|
| GSM8K | 42.9% | 34.1% | 25.0 | 8.2 |
| ProntoQA | 98.8% | 99.8% | 92.5 | 9.0 |
| ProsQA | 77.5% | 97.0% | 49.4 | 14.2 |

COCONUT underperforms on diverse math but dramatically outperforms on planning/search tasks.

### Collapse Prevention

No explicit regularization. They rely on:
1. Multi-stage curriculum (gradual transition)
2. Stage mixing (p=0.3 of data from previous stages)
3. Optimizer resets between stages
4. Loss masking on latent thoughts

### Direct Relevance to Us

COCONUT is the closest existing system to our matrix thinking. Key differences:
- COCONUT: vector-valued continuous thoughts (d-dimensional)
- Ours: matrix-valued continuous thoughts (d x d dimensional)
- COCONUT: requires CoT supervision to bootstrap
- Ours: must work from self-supervised language modeling alone (no CoT data)
- COCONUT: post-training on a pretrained model
- Ours: integrated from pretraining

Their curriculum approach may be adaptable: start with more thought slots that have high mixing weight, gradually shift to fewer but more powerful matrix thoughts.

---

## 5. CoCoMix (2025)

**Paper:** Meta FAIR, "LLM Pretraining with Continuous Concepts"
**arxiv:** 2502.08524

### Why This Matters Most

CoCoMix is the ONLY approach that trains continuous thought-like tokens FROM SCRATCH during pretraining. Everyone else fine-tunes pretrained models.

### Architecture

1. A pretrained Sparse Autoencoder (SAE) extracts "concepts" from hidden states
2. At concept prediction layers, the model predicts which concepts come next
3. Predicted concepts are compressed and INTERLEAVED into the hidden state sequence:
   ```
   (h_1, c^_1, h_2, c^_2, ..., h_t, c^_t)
   ```
4. This interleaved sequence feeds into remaining transformer layers

**Concept prediction:**
```
c^_t = W * TopK(z_t) + b    # Project sparse concept vector to hidden dim d
```

**Training loss:**
```
L = sum[-log f(x_{t+1} | h_<=t, c^_<=t)] + lambda * L_concept
L_concept = (1/K_attr) * sum[CE(z_t, top_attributed_concepts)]
lambda = 0.1, K_attr = 4
```

### Training From Scratch Details

- Models: 69M, 386M, 1.38B parameters
- Data: OpenWebText, 200B tokens, ~382K steps
- Hidden dims: 512, 1024, 2048
- Learning rates: 6e-4, 3e-4, 2e-4
- Batch sizes: 0.5M, 0.5M, 1M tokens
- Optimizer: AdamW, beta1=0.9, beta2=0.95, weight_decay=0.1
- Context length: 1024
- Concept prediction layer: layer 4 (69M) or layer 6 (386M, 1.38B)
- SAE: pretrained on 124M GPT-2, K_concept=32, 32768-dim concept space

### How Concepts Avoid Being Ignored

1. **Attribution-based selection:** Concepts chosen by gradient * input magnitude, not raw activation (17.5% efficiency gain)
2. **The model LEARNS to ignore bad concepts:** 5.8% of concept weights have norm < 1e-2, meaning the model can naturally route around unhelpful concepts
3. **Discrete cross-entropy loss:** Forces explicit concept prediction rather than soft regression
4. **Interleaving > intervention:** Appending concepts as separate tokens works better than adding them into hidden states (the model can attend to or ignore them)

### Performance vs Alternatives

- Consistently outperforms standard next-token prediction at all scales
- Outperforms knowledge distillation
- **Outperforms pause tokens** (demonstrating that meaningful content in the inserted tokens matters)
- 1.38B model achieves equivalent performance with **21.5% fewer training tokens**

### Direct Relevance to Us

This validates our core bet: inserting continuous internal representations during pretraining works and outperforms dummy pause tokens. The concept prediction loss (forcing the model to predict what the thought should be) is directly applicable to our matrix thought slots.

---

## 6. Thoughtbubbles (2025-2026)

**Paper:** Liu, Murty, Manning, Csordas (Stanford NLP), "Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space"
**arxiv:** 2510.00219
**Code:** https://github.com/stanfordnlp/thoughtbubbles

### Core Mechanism

The model learns to FORK residual streams at intermediate layers, creating parallel computation paths that merge at the final layer.

**Forking:**
- Each residual stream gets two scores per layer: fork_score and keep_score
- Scores accumulate multiplicatively across layers: p_cum = p_prev * p_current
- Top-k operation selects which streams survive
- Forked copies get a learned "fork embedding" added to distinguish them from parents

**Merging:**
- At the final layer, multiple streams for the same token are decoded separately
- Merged via score-weighted averaging of probability distributions:
  ```
  output = softmax(Decode(sum_j(p_cum_j * residual_j)))
  ```

### Trained With ONLY Language Modeling Loss

No auxiliary losses, no special supervision. The model learns useful forking behavior purely from next-token prediction. This works because of **residual update attenuation**: attention and residual updates are modulated by cumulative scores. Tokens the model needs to process most become the ones with highest fork scores — a self-reinforcing signal.

### Preventing Collapse (Fork Everything / Fork Nothing)

The multiplicative score mechanism is key:
- Without the attention masking, top-k decisions are essentially random (validation loss jumps from 3.17 to 3.90 in ablation)
- Low-scored streams are actively masked from attention and residual updates — they become dead weight
- The model cannot ignore forking because it needs the score-weighted attention to flow
- Forced keep_score=1 for original streams ensures at least one path always survives

### What The Model Learns

The model autonomously allocates more computation to **moderate-entropy tokens** — not the easiest (already known) nor the hardest (diminishing returns), but tokens where extra thinking actually helps. This is an emergent property, not designed in.

### Training From Scratch

Yes, fully trained from scratch:
- Scales: 150M to 772M (also 1.9B in extended experiments)
- Data: FineWeb + peS2o, 40B tokens at 1.9B scale
- Forking layers: after layers 3, 7, 11 (out of 12 for small models)
- Outperforms standard decoders AND non-adaptive parallel computation on perplexity and zero-shot benchmarks

### Direct Relevance to Us

Thoughtbubbles proves you can learn adaptive internal computation from scratch with only LM loss. Their fork/merge on vector residual streams is analogous to our matrix thought slot mechanism. Key architectural parallel: their "fork embedding" that distinguishes parent/child streams maps to our potential "thought position encoding" for matrix thought slots.

---

## 7. Inner Thinking Transformer (2025)

**Paper:** "Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking"
**arxiv:** 2502.13842

### Architecture

Three components:
1. **Adaptive Token Routing (ATR):** A routing network generates importance scores; tokens above a percentile threshold get extra processing
2. **Residual Thinking Connections (RTC):** Outputs from multiple thinking iterations are summed (not replaced):
   ```
   x^(t) = sum(f(x^(i-1)) * phi^(i))   for i = 1..t
   ```
   where phi^(i) are learnable per-step position encodings
3. **Thinking Step Encoding:** Differentiates which "thinking step" the model is on

### How Thinking Is NOT Ignored

Three mechanisms:
- Residual accumulation: each step's output is ADDED to the final result (can't be zero-gated away)
- Learnable position encodings per step: forces differentiation between steps
- Router gradients: the routing network receives gradients, preventing collapse to uniform selection

### Trained From Scratch

Explicitly: "starting from random initialization and training on the same dataset."
- Scales: 162M, 230M, 466M parameters
- Data: RedPajama, 50B tokens
- LLaMA architecture base

### Direct Relevance

ITT shows that residual accumulation (summing, not replacing) is a robust way to ensure thought iterations contribute. This maps directly to our iterative matrix refinement: each iteration should ADD to the matrix, not replace it.

---

## 8. Seq-VCR (2024)

**Paper:** Shwartz-Ziv et al., "Seq-VCR: Preventing Collapse in Intermediate Transformer Representations for Enhanced Reasoning"
**arxiv:** 2411.02344 (ICLR 2025)

### The Problem

Decoder-only transformers exhibit representation collapse in intermediate layers: hidden states become similar, entropy drops, and the model loses the ability to perform multi-step reasoning (like carrying digits in multiplication).

### The Solution: Variance-Covariance Regularization

Applied per-sequence-position across the batch dimension:

**Variance term** (enforce unit variance per feature):
```
L_var = max(0, 1 - sqrt(C_{i,k,k} + eta))     eta = 0.001
```

**Covariance term** (decorrelate features):
```
L_cov = sum(C_{i,k,k'}^2)   for k != k'
```

**Combined:**
```
L_SeqVCR = (1/(T*d)) * sum[lambda_1 * L_var + lambda_2 * L_cov]
```

### The Dramatic Result

GPT-2 Small + 2 pause tokens + Seq-VCR achieves **99.5% exact match on 5x5 digit multiplication**, compared to:
- 0% for vanilla GPT-2 Small
- 44% for GPT-4 with 5-shot CoT

### Hyperparameters

**For multiplication:** lambda_1=1.0, lambda_2=0.004, batch_size=32
**For other tasks:** lambda_1=0.1, lambda_2=0.5, batch_size=128

**From scratch training:** 100 epochs, lr=1e-4, batch_size=128
- Computing covariance across batch+length dimensions helped dramatically (87-99% vs 3%)
- Best layer: L12 for fine-tuning, L0 for from-scratch
- "More sensitive to hyperparameter choices" when training from scratch

### Direct Relevance to Us

This is the most directly applicable anti-collapse technique for our architecture. We can apply Seq-VCR to our matrix thought representations:
- Variance term: ensure matrix thought slots maintain diverse singular values
- Covariance term: ensure different thought slots don't collapse to the same representation
- This can be applied to the FLATTENED matrix representations across the batch dimension

---

## 9. Fast Quiet-STaR (2025)

**Paper:** "Fast Quiet-STaR: Thinking Without Thought Tokens"
**arxiv:** 2505.17746 (EMNLP 2025 Findings)

### Key Innovation: Curriculum to Internalize Thinking

Uses curriculum learning to progressively reduce the number of explicit thought tokens, forcing the model to internalize reasoning into hidden states:

```
Stage 1: Full thought tokens (like Quiet-STaR)
Stage 2: Fewer thought tokens
...
Final: Zero thought tokens (Fast Quiet-STaR NTP)
```

### Results

- Mistral 7B: +9% average accuracy with ZERO inference overhead
- Qwen2.5 7B: +5.7% average accuracy with ZERO inference overhead
- Outperforms Quiet-STaR under same inference time budget

### Direct Relevance

This validates the curriculum approach: start with more explicit internal computation, gradually compress it. For our architecture, this suggests starting with many thought matrix slots (say 8) and progressively reducing to fewer but richer ones.

---

## 10. Soft Thinking (NeurIPS 2025)

**Paper:** Zhang et al., "Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space"
**arxiv:** 2505.15778
**Code:** https://github.com/eric-ai-lab/Soft-Thinking

### Core Idea

Instead of sampling a single discrete token at each step, keep the full probability distribution as a weighted mixture of embeddings:

```
e_next = sum(p[k] * e(k))    for all k in vocabulary
```

This creates a "soft" continuous token in the convex hull of all embeddings.

### Cold Stop Mechanism (Collapse Prevention)

Monitors entropy at each generation step:
```
If H(p) = -sum(p[k] * log(p[k])) < threshold tau:
    increment counter
    If counter >= k consecutive low-entropy steps:
        Insert </think> token, stop reasoning, produce answer
```

This prevents the model from continuing to reason when it's become overconfident (which leads to OOD drift and collapse).

### Training Free

No weight updates needed. Uses existing embeddings and probability distributions. Temperature=0.6.

### Results

- AIME 2024: 76.88% -> 83.33% (QwQ-32B)
- Token reduction: 11-22%

### Direct Relevance

The cold stop mechanism is relevant for our adaptive thinking. When our matrix thoughts become low-rank (highly certain), we could similarly halt further thinking iterations. The entropy-based stopping criterion could translate to a rank-based criterion for matrices.

---

## 11. Synthesis: The Credit Assignment Problem

How does the model learn that thought X helped prediction Y?

### Approaches in the Literature

| Method | Credit Assignment Strategy |
|--------|---------------------------|
| STaR | Filter by answer correctness (binary, coarse) |
| Quiet-STaR | REINFORCE with relative reward (reward = improvement in prediction) |
| Pause Tokens | None explicit (learned implicitly through backprop, loss only on real tokens) |
| COCONUT | Direct backprop through continuous thoughts (fully differentiable) |
| CoCoMix | Concept prediction loss + interleaved attention (model attends to concepts or ignores them) |
| Thoughtbubbles | Score-weighted attention masking (low-scoring thoughts get masked out) |
| ITT | Residual accumulation (all thinking steps sum into output) |
| Seq-VCR | Regularization ensures representations stay diverse (indirect) |

### The Key Insight

**Continuous thought tokens have a massive advantage: direct backpropagation.** Discrete thoughts (STaR, Quiet-STaR) require REINFORCE or filtering because you can't backprop through argmax/sampling. Continuous thoughts (COCONUT, CoCoMix, our matrices) get gradients for free.

Our matrix thought tokens are continuous. This means:
- Gradients flow directly from the prediction loss through the thought matrices
- No REINFORCE, no variance reduction, no reward baselines needed
- Credit assignment is handled by standard backpropagation
- The model can learn arbitrarily fine-grained "this element of this thought matrix helped this prediction"

### Remaining Challenge

Even with direct gradients, the model might learn to IGNORE thought slots entirely (set mixing weight to zero, or let thought matrices become identity/zero). This is the "lazy thinking" problem, addressed next.

---

## 12. Synthesis: Preventing Lazy Thinking / Collapse

This is the critical engineering challenge for our Phase 4. Every approach in the literature has encountered some form of this problem. Here are all known prevention mechanisms:

### Mechanism 1: Mixing Head with Initialization Bias (Quiet-STaR)

Start the mixing weight close to 1 (trust base model), let it drift toward thought-based predictions. This prevents early destruction of the base model's predictions while allowing gradual thought integration.

**For us:** Initialize a mixing gate between thought-influenced and thought-free predictions. Start biased toward thought-free. The gradient will push toward using thoughts IF they help.

### Mechanism 2: Loss Masking (COCONUT, Pause Tokens)

Never compute loss on thought positions themselves. Thoughts exist purely to help predict real tokens. This prevents the model from wasting capacity predicting "what the thought should look like" instead of using thoughts for downstream predictions.

**For us:** Already planned. Loss only on byte predictions, not on thought matrix slots.

### Mechanism 3: Curriculum / Staged Training (COCONUT, Fast Quiet-STaR, CoCoMix)

Don't start with full thought-based reasoning. Gradually introduce thought tokens:
- Stage 0: no thoughts, standard LM
- Stage 1: thoughts inserted but heavily mixed with direct predictions
- Stage 2: thoughts carry more weight
- ...
- Stage N: full thought-based reasoning

**For us:** Start with 0 thought slots. After N steps of standard training, add 1 thought slot with high mixing bias (mostly ignored). Gradually reduce mixing bias. Add more slots over time.

### Mechanism 4: Variance-Covariance Regularization (Seq-VCR)

Explicitly regularize intermediate representations to maintain diversity:
```
L_var = max(0, 1 - sqrt(variance + eps))     # Push variance toward 1
L_cov = sum(off_diagonal(covariance)^2)       # Push correlations toward 0
```

**For us:** Apply to the vectorized thought matrices across the batch dimension. This ensures:
- Each thought matrix maintains diverse singular values (not rank-0 or rank-1 collapse)
- Different thought slots don't converge to the same representation
- Hyperparameters: lambda_var=0.1-1.0, lambda_cov=0.004-0.5 (task-dependent)

### Mechanism 5: Score-Weighted Attention Masking (Thoughtbubbles)

Make the thought's influence on attention proportional to a learned score. Low-scoring thoughts get masked from attention — they can't contribute OR interfere. This is self-reinforcing: useful thoughts get high scores get more attention get more gradient signal.

**For us:** Each thought matrix slot could have a learned "relevance score" that gates its contribution to attention. The score gets gradient from the prediction loss.

### Mechanism 6: Residual Accumulation (ITT)

Don't REPLACE representations with thought outputs — ADD thought contributions to a residual stream. This means the base prediction always survives; thoughts can only help, not hurt (modulo optimization dynamics).

**For us:** After thought processing, add the thought-derived signal to the base representation rather than replacing it. Even if thoughts collapse to zero, the base prediction survives.

### Mechanism 7: Rank/Entropy Monitoring + Early Stopping (Soft Thinking Cold Stop)

Monitor the "confidence" of thought representations. If thoughts become too certain (low entropy / low rank), stop computing more thoughts.

**For us:** Monitor the rank of thought matrices during training. If rank collapses (approaches 1), apply a rank-encouraging regularization:
```
L_rank = -sum(sigma_i / sigma_1)     # Encourage spread across singular values
```

### Mechanism 8: Concept Prediction Loss (CoCoMix)

Give the thought slots a secondary objective: predict WHAT the thought should represent (using a discrete cross-entropy over concept categories, not just helping downstream predictions).

**For us:** This is harder without a concept vocabulary, but we could use a reconstruction loss: the thought matrix at position t should be predictable from context up to t (self-consistency). Or: thought matrices should be useful for predicting MULTIPLE future tokens (not just the next one), forcing them to encode genuine long-range information.

### Mechanism 9: Information Bottleneck (Research Frontier)

Force thought representations through a low-dimensional bottleneck, then expand. This prevents thoughts from being a simple copy of the input. Combined with a reconstruction loss, this ensures thoughts are both compact AND informative.

**For us:** Our matrices are already a form of bottleneck (d x d with low effective rank). We could enforce this explicitly by projecting through rank-r for r < d, then expanding back.

---

## 13. What This Means for Our Architecture

### Our Unique Position

We have several advantages over every system in the literature:

1. **Matrix thoughts are continuous** -> direct backprop, no REINFORCE
2. **Matrix thoughts are high-dimensional** -> a 16x16 matrix has 256 values per thought (vs ~96 for a standard hidden state), enabling richer internal representations
3. **We plan to train from scratch** -> no distribution shift from pretrained model. Pause token paper shows this is essential.
4. **Multiplicative composition** -> matrix multiplication between thoughts creates nonlinear interactions that vector concatenation cannot

### Our Unique Risks

1. **No CoT data to bootstrap from** -> COCONUT's curriculum relies on having CoT supervision. We need a different curriculum strategy.
2. **Matrix operations are expensive** -> each thought interaction involves matrix multiplication (O(d^3)), not vector operations (O(d))
3. **Higher-dimensional collapse modes** -> matrices can collapse in rank, eigenvalue distribution, spectral structure, not just magnitude. More ways to go wrong.

### The Hierarchy of Evidence

From the literature, ranked by relevance to our architecture:

| Rank | System | Why Relevant |
|------|--------|-------------|
| 1 | **Pause Tokens** | Proves thought slots work from pretraining; simplest version of our approach |
| 2 | **Seq-VCR** | Direct anti-collapse technique applicable to our matrix representations |
| 3 | **COCONUT** | Continuous thoughts with direct backprop; closest to our mechanism |
| 4 | **CoCoMix** | Only system training continuous thoughts from scratch at scale |
| 5 | **Thoughtbubbles** | Unsupervised learning of adaptive internal compute; trains from scratch |
| 6 | **ITT** | Residual accumulation for thought iterations; trains from scratch |
| 7 | **Fast Quiet-STaR** | Curriculum for internalizing explicit thinking |
| 8 | **Quiet-STaR** | REINFORCE for credit assignment (we don't need this, but the mixing head is relevant) |
| 9 | **Soft Thinking** | Cold stop / rank-based early termination for adaptive thinking |
| 10 | **STaR** | Bootstrapping paradigm (conceptual, not directly applicable) |

---

## 14. Practical Training Recipes

### Recipe A: Conservative (Highest Probability of Working)

Based on Pause Tokens + Seq-VCR. Validated components only.

```
Phase 1 (Steps 0-50K): Standard matrix LM training, NO thought slots
  - Establish base model predictions
  - lr=3e-4, warmup=2000 steps

Phase 2 (Steps 50K-100K): Add N=2 thought matrix slots
  - Loss only on real token predictions
  - Mixing gate initialized to 0.95 (mostly ignore thoughts)
  - Seq-VCR regularization on thought matrices:
    lambda_var=0.1, lambda_cov=0.01
  - lr=1e-4 (reduced for stability)

Phase 3 (Steps 100K-200K): Increase to N=4 thought slots
  - Mixing gate bias reduced to 0.7
  - Seq-VCR lambda_var=0.5, lambda_cov=0.05

Phase 4 (Steps 200K+): Full thought integration
  - Mixing gate free to learn
  - Seq-VCR maintained at stable values
  - Monitor: thought matrix rank, mixing gate values, loss curves
```

### Recipe B: Aggressive (Higher Ceiling, More Risk)

Based on Thoughtbubbles + COCONUT approach. More novel.

```
Phase 1 (Steps 0-100K): Train with N=4 thought matrix slots from step 0
  - No curriculum — thoughts are part of the architecture from the start
  - Score-weighted attention masking (Thoughtbubbles-style)
  - Residual accumulation (ITT-style): thoughts ADD to base representation
  - Loss only on real tokens
  - Seq-VCR on both base representations AND thought matrices
  - lr=3e-4, standard warmup

Key architectural choices:
  - Each thought slot has a learned relevance score (scalar, sigmoid activation)
  - Thought contribution: base_hidden += sum(score_i * thought_output_i)
  - Attention mask: thought-to-token attention weighted by relevance scores
  - Multiple thought iterations per slot (our existing iterative refinement)
```

### Recipe C: Matrix-Native (Our Unique Angle)

Based on the observation that matrix rank naturally provides a collapse signal.

```
Phase 1: Standard matrix LM training with thought slots from step 0

Thought slot design:
  - N=4 thought matrix positions interleaved in context
  - Each thought matrix initialized as random rank-4 matrix (not identity, not zero)
  - Thoughts interact via matrix multiplication (our multiplicative composition)

Anti-collapse measures (ALL applied simultaneously):
  1. Spectral regularization:
     L_spectral = -alpha * sum(sigma_i / sigma_max)    # Encourage rank diversity
     alpha = 0.01-0.1

  2. Thought diversity loss:
     L_diverse = beta * sum(||flatten(T_i) . flatten(T_j)|| / (||T_i|| * ||T_j||))
     for all pairs (i,j) of thought matrices
     beta = 0.01

  3. Mixing gate with residual connection:
     output = base_prediction + gate * thought_prediction
     gate initialized near 0, learned

  4. Multi-token lookahead (from Quiet-STaR):
     Loss includes prediction of tokens t+1, t+2, t+3
     Thoughts that help predict FARTHER ahead get more gradient signal

  5. Rank monitoring (not a loss, just diagnostics):
     Track effective rank of each thought matrix each 100 steps
     If any thought slot's average rank < 2 for 1000 consecutive steps:
       -> reinitialize that slot's parameters
```

### Diagnostic Checklist (Run These Every 1K Steps)

1. **Thought matrix rank:** Should be 3-8 for d=16 matrices. Below 2 = collapse.
2. **Mixing gate values:** Should be 0.2-0.8 after warmup. 0.0 = thoughts ignored. 1.0 = base model ignored.
3. **Thought matrix norms:** Should be similar to input embedding norms. Much smaller = thoughts dying. Much larger = thoughts dominating (unstable).
4. **Thought diversity:** Cosine similarity between flattened thought matrices should be < 0.5. Higher = redundant slots.
5. **Prediction improvement:** Compare loss WITH thoughts vs loss with thoughts masked out. If no difference after 10K steps = thoughts not helping.
6. **Gradient magnitude through thoughts:** Should be comparable to gradients through base model. Much smaller = vanishing (thoughts disconnected from loss).

---

## Sources

### Primary Papers
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) (NeurIPS 2022)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) (2024)
- [Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226) (ICLR 2024)
- [Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)](https://arxiv.org/abs/2412.06769) (2024)
- [LLM Pretraining with Continuous Concepts (CoCoMix)](https://arxiv.org/abs/2502.08524) (2025)
- [Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space](https://arxiv.org/abs/2510.00219) (2025)
- [Inner Thinking Transformer](https://arxiv.org/abs/2502.13842) (2025)
- [Seq-VCR: Preventing Collapse in Intermediate Transformer Representations](https://arxiv.org/abs/2411.02344) (ICLR 2025)
- [Fast Quiet-STaR: Thinking Without Thought Tokens](https://arxiv.org/abs/2505.17746) (EMNLP 2025)
- [Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space](https://arxiv.org/abs/2505.15778) (NeurIPS 2025)

### Survey
- [Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.16782) (2025)

### Code Repositories
- [Quiet-STaR official](https://github.com/ezelikman/quiet-star)
- [Quiet-STaR community](https://github.com/expz/quiet-star)
- [COCONUT](https://github.com/facebookresearch/coconut)
- [Thoughtbubbles](https://github.com/stanfordnlp/thoughtbubbles)
- [Soft Thinking](https://github.com/eric-ai-lab/Soft-Thinking)

### Additional References
- [Enhancing Latent Computation in Transformers with Latent Tokens](https://arxiv.org/abs/2505.12629) (2025)
- [A Survey on Latent Reasoning](https://arxiv.org/abs/2507.06203) (2025)
