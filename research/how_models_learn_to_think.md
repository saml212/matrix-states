# How Models Learn to Think: Training Procedures for Internal Reasoning

**Research Date:** 2026-03-26
**Scope:** Training procedures, loss functions, and curricula for 10 families of "thinking" models

---

## 1. COCONUT (Meta, Dec 2024)

**Paper:** arxiv 2412.06769, accepted COLM 2025
**Code:** github.com/facebookresearch/coconut

### How Thinking is Trained

COCONUT uses a **staged curriculum** to teach a pretrained LM to reason in continuous space. The key insight: you cannot train continuous thoughts from scratch. You must gradually wean the model off discrete chain-of-thought.

**Training Stages (verified from code):**

The curriculum is controlled by `scheduled_stage = epoch // epochs_per_stage`. At stage k:
- The first k reasoning steps are **removed** from the text
- k * c continuous thought tokens (`<|latent|>`) are **inserted** after the question
- Remaining reasoning steps + answer stay as text
- Loss is computed ONLY on the remaining text tokens (question and latent tokens are masked with label=-100)

```
Stage 0: Question + [Step1, Step2, Step3] + Answer  (pure CoT)
Stage 1: Question + <latent><latent> + [Step2, Step3] + Answer
Stage 2: Question + <latent><latent><latent><latent> + [Step3] + Answer
Stage 3: Question + <latent><latent><latent><latent><latent><latent> + Answer
```

**The critical trick:** At each latent token position, the model's hidden state from the PREVIOUS position is fed back as the input embedding for the current position. This creates a recurrent computation loop through the continuous space:

```python
# From coconut.py lines 144-149
for idx_pair in filling_indices:
    batch_idx, token_idx = idx_pair
    tensor_list[batch_idx][token_idx] = hidden_states[
        batch_idx, token_idx - 1 - hidden_states_offset, :
    ]
```

**Loss function:** Standard cross-entropy, masked so loss is ZERO on question tokens and latent thought tokens. The model is only supervised on predicting the remaining reasoning steps and the answer.

### How It Learns WHEN to Think

It does not learn this. The number of continuous thoughts is **fixed per stage** during training and **fixed at inference** (padded to constant length). There is no adaptive halting.

### How It Learns WHAT to Think About

Through the curriculum. Each stage forces the model to compress one more reasoning step into continuous space. The gradient signal comes from predicting the remaining text + answer. If the latent thoughts don't encode useful information, the remaining predictions will be wrong, and loss will be high.

### Fixed vs Adaptive

**Fixed throughout.** The paper mentions training a binary classifier for autonomous termination but uses constant padding in all reported experiments.

### What Failed

- **Training without curriculum:** Directly training in the final stage (all latent, no text reasoning) performs "no better than no-CoT." The curriculum is essential.
- **Pause tokens (filler tokens):** 77.7% accuracy on ProntoQA vs COCONUT's 99.8%, with huge variance (±21.0).
- **c=3 (3 thoughts per step):** Training instability and performance drops. c=1 or c=2 is the sweet spot.
- **Optimizer must be reset between stages** (`reset_optimizer: True` in all configs).

### Exact Hyperparameters (from code)

| Setting | GSM8k | ProsQA |
|---------|-------|--------|
| Base model | GPT-2 (pretrained) | GPT-2 (pretrained) |
| c_thought | 2 | 1 |
| epochs_per_stage | 3 | 5 |
| max_latent_stage | 3 | 6 |
| total epochs | 25 | 50 |
| lr | 1e-4 | 1e-4 |
| batch_size | 32 (x4 GPU) | 32 (x4 GPU) |
| weight_decay | 0.01 | 0.01 |
| optimizer | AdamW | AdamW |
| reset_optimizer | Yes (every stage) | Yes (every stage) |
| GSM8k starts from | CoT-finetuned checkpoint | scratch |

### Key Discovery: Breadth-First Search Behavior

The continuous thought representation spontaneously learns to maintain a **superposition of multiple reasoning paths** (BFS-like). The model delays committing to a single path, keeping multiple hypotheses alive in the continuous space, and progressively narrowing until it reaches the answer. This cannot happen with discrete tokens, which force a single path.

---

## 2. DeepSeek-R1 (Jan 2025)

**Paper:** arxiv 2501.12948

### How Thinking is Trained

DeepSeek-R1 proves that **pure reinforcement learning can teach a model to think**, without any supervised chain-of-thought examples. The training has two variants:

**DeepSeek-R1-Zero (pure RL, single stage):**
1. Start with DeepSeek-V3-Base (pretrained, no SFT)
2. Apply GRPO directly with rule-based rewards
3. The model spontaneously develops chain-of-thought reasoning

**DeepSeek-R1 (production, four stages):**

**Stage 1 - Cold Start SFT:** Fine-tune on ~thousands of long CoT examples for readability. These examples were collected from: few-shot prompting with long CoT, prompting for detailed answers with reflection, collecting R1-Zero outputs in readable format, human annotator post-processing.

**Stage 2 - Reasoning RL:** Same GRPO procedure as R1-Zero, focusing on math, code, science, logic.

**Stage 3 - Rejection Sampling + SFT:** Generate solutions from Stage 2 checkpoint, keep correct ones (~600k reasoning samples + ~200k non-reasoning from V3). Fine-tune for 2 epochs.

**Stage 4 - General RL:** Second RL round for helpfulness, harmlessness, and reasoning refinement.

### GRPO Algorithm (exact formulation)

```
J_GRPO(theta) = E[ 1/G * sum_i(
    min(
        pi_theta(o_i|q) / pi_theta_old(o_i|q) * A_i,
        clip(pi_theta(o_i|q) / pi_theta_old(o_i|q), 1-eps, 1+eps) * A_i
    ) - beta * D_KL(pi_theta || pi_ref)
)]
```

- G = group size (multiple outputs sampled per question)
- A_i = advantage, computed by normalizing rewards within the group: `A_i = (r_i - mean(r)) / std(r)`
- No critic model needed (advantages from group statistics)

### Reward Signal

**Purely rule-based, no neural reward model:**
- **Accuracy reward:** Binary/graded. For math: check against ground truth after extracting answer from specified format. For code: compilation + test case execution.
- **Format reward:** Binary. Must use `<think>...</think>` and `<answer>...</answer>` tags.
- **Language consistency reward:** Proportion of target language words in CoT (prevents language mixing).
- **Final reward = accuracy + language_consistency** (summed directly)

They explicitly AVOIDED neural reward models: "the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process."

### The "Aha Moment"

In an intermediate R1-Zero checkpoint, the model spontaneously generated: "Wait, wait. Wait. That's an aha moment I can flag here." It learned to **re-evaluate and self-correct** without being trained to do so. The model's response length grew from hundreds to thousands of tokens during RL training - it learned that longer thinking leads to higher rewards, and developed meta-cognitive strategies (backtracking, verification, re-evaluation) as emergent behaviors.

### How It Learns WHEN to Think

Emergent from RL. Thinking time grows during training because longer, more careful reasoning leads to higher accuracy rewards. The model learns that hard problems require more thinking. There is NO explicit mechanism to control thinking length during RL - it is entirely emergent.

### What Failed

- **Language mixing:** R1-Zero mixed languages in CoT, especially with multilingual prompts. Fixed with language consistency reward.
- **Readability:** R1-Zero output was poorly formatted and hard to read. Fixed with cold-start SFT in Stage 1.
- **Neural reward models:** Suffered from reward hacking at scale. Abandoned in favor of rule-based rewards.
- **Remaining weaknesses:** Function calling, multi-turn, complex role-playing, JSON output still weaker than V3.

---

## 3. OpenAI o1 / o3 (2024-2025)

**Official details are sparse.** What follows is reconstructed from the system card, blog posts, and technical analysis.

### How Thinking is Trained

Best reconstruction of the pipeline:

1. **SFT on CoT data:** The base model is fine-tuned on chain-of-thought demonstrations (possibly including step-level annotations).
2. **RL with reward models:** The model generates multiple reasoning chains, which are evaluated by reward models (likely both process reward models and outcome reward models).
3. **Process Reward Models (PRMs):** Assign rewards to individual reasoning STEPS, not just final answers. OpenAI published "Let's Verify Step by Step" (ICLR 2024) with PRM800K: 800K step-level labels across 75K solutions to 12K MATH problems, labeled by humans as positive/negative/neutral.
4. **Deliberative alignment:** For o3, safety reasoning is baked in - the model is trained to "think about" safety policies during CoT.

**PRM scoring:** The PRM score for a solution = product of correctness probabilities for each step. The PRM predicts correctness after the last token of each step.

### How Thinking Length is Determined

o1/o3 appear to use **test-time compute scaling**: generating multiple candidate reasoning chains and selecting the best one via the reward model. The "thinking budget" can be set by the user (low/medium/high for o3). Internally, the model likely uses a combination of:
- Maximum token budget
- Early stopping when confidence is high
- Best-of-N selection among candidate chains

### What's Known About Failures

- Performance scales with both train-time compute (more RL) and test-time compute (more thinking)
- The "simple query paradox": short questions waste compute on unnecessary thinking
- Safety training reduced performance on some benchmarks

### Key Insight: Process Reward Models

The PRM approach is fundamentally different from DeepSeek's rule-based rewards. PRMs enable:
- Credit assignment to individual steps (not just final answer)
- Detection of correct reasoning that leads to wrong answers (and vice versa)
- Training the model to improve its reasoning process, not just its outputs

---

## 4. PonderNet (DeepMind, 2021)

**Paper:** arxiv 2107.05407

### How Thinking is Trained

PonderNet learns a **halting distribution** end-to-end, jointly with the task. The model decides at each step whether to stop or continue.

**Step function:** At each step n, the model computes:
```
y_hat_n, h_{n+1}, lambda_n = s(x, h_n)
```
where `lambda_n` is the conditional probability of halting at step n (sigmoid output).

**Halting probability distribution:**
```
p(halt at n) = lambda_n * prod_{j=1}^{n-1} (1 - lambda_j)
```
This is exactly a learned geometric-like distribution.

### Loss Function (exact)

```
L = L_Rec + beta * L_Reg

L_Rec = sum_{n=1}^{N} p_n * loss(y, y_hat_n)    [weighted reconstruction]

L_Reg = KL(p || p_G(lambda_p))                   [regularization]
```

- `p_n` = probability of halting at step n
- `p_G(lambda_p)` = truncated geometric prior with parameter lambda_p
- `loss(y, y_hat_n)` = task-specific loss at step n (e.g., cross-entropy)
- `beta` = regularization weight (typically 0.01)
- N = smallest n where `sum_{j=1}^{n} p_j > 1 - epsilon` (epsilon ~ 0.05)

**The geometric prior** encourages the model to halt early (exponentially decaying probability of continuing). The KL term prevents the model from always using maximum steps, while still allowing it to think longer when needed.

### Why PonderNet Collapses at Small Scale

This is the failure mode we experienced. The causes:

1. **Insufficient capacity:** With small models, the per-step improvement is tiny. The KL term toward the geometric prior (which favors halting early) dominates the reconstruction benefit of additional steps. The model learns: "thinking more doesn't help enough to overcome the regularization penalty."

2. **Lambda_p sensitivity:** When lambda_p is high (e.g., 0.9), the prior strongly favors 1 step. PonderNet fails the parity task entirely at lambda_p=0.9. At small scale, even moderate lambda_p values can cause collapse.

3. **Gradient signal weakness:** The reconstruction loss is WEIGHTED by halting probabilities. If the model starts halting early, the gradient signal from later steps vanishes (their weight p_n approaches 0), creating a self-reinforcing collapse cycle.

4. **No curriculum:** PonderNet trains all steps simultaneously from the start. There is no staged curriculum to first teach the model that thinking helps, then introduce halting.

### What Conditions Make PonderNet Work

- Large enough model that per-step improvement is significant
- Low lambda_p (e.g., 0.2-0.5) so the prior doesn't dominate
- Low beta so reconstruction loss matters more than regularization
- Tasks where extra computation has CLEAR, MEASURABLE benefit per step
- The paper's successes were on synthetic tasks (parity, algorithmic reasoning) where depth is strictly necessary

### Comparison to ACT

ACT (Graves, 2017) uses a direct penalty `tau * N(x)` on the number of steps. This is "notably unstable and sensitive to the choice of hyperparameter tau." PonderNet's KL formulation is more principled but still fragile at small scale.

---

## 5. Energy-Based Transformers (July 2025)

**Paper:** arxiv 2507.02092

### How Thinking is Trained

EBTs are a fundamentally different paradigm. Instead of generating tokens, the model defines an **energy function** E_theta(x, y_hat) that assigns a scalar energy to every (input, prediction) pair. Low energy = good prediction.

**Training Algorithm (Algorithm 1):**
```
1. Sample initial prediction: y_hat_0 ~ N(0, I)
2. For i = 0 to N-1:
     y_hat_{i+1} = y_hat_i - alpha * grad_y E_theta(x, y_hat_i) + eta_i
     where eta_i ~ N(0, sigma)                    [Langevin noise]
3. Compute loss: L = J(y_hat_N, y_true)           [cross-entropy or MSE]
4. Backpropagate through entire trajectory         [requires second-order gradients]
5. Update theta
```

**Critical:** The loss is only computed on the FINAL prediction after N optimization steps. But gradients flow through ALL N steps, requiring Hessian-vector products (second-order derivatives). These scale linearly with model size.

### How the Energy Function is Trained

The energy function is trained **implicitly** through the optimization trajectory. There is NO explicit energy loss. Instead:
- The model learns an energy landscape such that gradient descent on that landscape leads to correct predictions
- The supervision is standard (cross-entropy for LM, MSE for images)
- The energy function must be smooth and well-shaped for gradient descent to work

### Key Training Tricks

1. **Langevin Dynamics:** Adding noise during optimization prevents the model from learning energy landscapes with narrow, hard-to-find minima. Without it: -17.2% thinking performance.

2. **Replay Buffer:** Stores previous optimization trajectories and reuses them, simulating longer optimization paths. Without it: -14.8% thinking performance.

3. **Random Step Size:** Alpha is randomized during training, forcing the model to learn energy landscapes that work with different step sizes. Without it: -1.47% thinking performance. "Randomizing the step size is critical - removing it nearly eliminates Thinking gains."

4. **Random Number of Steps:** N is also randomized during training, not fixed.

### Inference (Algorithm 2)

```
For j = 1 to M:                           [M random starting points]
    Sample y_hat_0,j ~ N(0, I)
    For i = 0 to N-1:
        y_hat_{i+1,j} = y_hat_i,j - alpha * grad_y E_theta(x, y_hat_i,j)
Return y_hat* = argmin_j E_theta(x, y_hat_N,j)    [pick lowest energy]
```

**Self-verification:** The energy scalar itself serves as a verifier. No external reward model needed. Generate M candidates, pick the one with lowest energy.

### What Failed

- Without Langevin noise: model learns narrow energy wells that are hard to find from random init
- Without replay buffer: energy landscape is only well-defined very close to the minimum
- Without random step sizes: model overfits to a specific optimization trajectory
- Training requires second-order gradients (expensive but scales linearly)

### Why This Matters

EBTs achieve "System 2 Thinking" from **unsupervised pretraining alone** - no RL, no reward models, no CoT data. The energy function naturally learns to verify predictions. The model gets better by thinking longer (more gradient steps at inference), with no additional training.

---

## 6. LoopFormer (ICLR 2026)

**Paper:** arxiv 2602.11451
**Code:** github.com/armenjeddi/loopformer

### How Thinking is Trained

LoopFormer uses **shortcut-consistency training** to make iterative refinement work at any depth, not just the trained depth.

**Full Training Objective:**
```
L = L_L + lambda_1 * L_S + lambda_2 * L_cons

L_L = CrossEntropy on full L-loop trajectory
L_S = CrossEntropy on shortcut S-loop trajectory (S < L)
L_cons = ||stopgrad(h^(L)) - h^(S)||^2     [consistency loss]
```

With lambda_1 = lambda_2 = 0.1 throughout all experiments.

### How Shortcuts Work

Each training batch:
1. Run full L iterations, compute L_L
2. Sample shortcut length S ~ Uniform{1, ..., L-1}
3. Sample random step sizes Delta_S such that sum(Delta_i) = 1
4. Run S iterations with these step sizes, compute L_S and L_cons
5. Backpropagate all three losses

**Time conditioning:** Each block receives a normalized time t in [0,1] and step size Delta via sine-cosine Fourier features + MLP, injected via AdaLN modulation (scaling/shifting RMSNorm). This tells the model "where in the computation you are" and "how big this step is."

### How It Learns WHEN to Stop

**It does not.** LoopFormer has NO adaptive stopping. The user specifies a compute budget M at inference time. The model produces the best output it can in M steps. This is a design choice, not a limitation - it avoids the PonderNet collapse problem entirely.

### What Failed

- **Naive early exiting (without consistency training):** Representations become "stagnant" - later iterations add nothing. High CKA similarity across loops, indicating no refinement.
- **Without L_cons:** Shorter trajectories produce garbage - only the full L trajectory is useful.
- **The consistency loss is what makes flexible-depth work.** It forces the model to produce useful representations at EVERY depth.

### Exact Hyperparameters

- Training: 50,000 optimizer steps on ~25B tokens
- Peak LR: 6e-4 with cosine decay
- Warmup: 4,000 steps
- Optimizer: AdamW
- Lambda_1 = Lambda_2 = 0.1
- Time/step conditioning: 256-dim Fourier features, 2-layer MLP with SiLU
- Architecture: NanoGPT fork with RMSNorm (not LayerNorm)

---

## 7. Mixture-of-Recursions (NeurIPS 2025)

**Paper:** arxiv 2507.10524
**Code:** github.com/raymin0223/mixture_of_recursions

### How the Router is Trained

The router is trained **end-to-end** during pretraining. No separate pre-training or post-hoc fitting.

**Two routing strategies:**

**Expert-Choice (preferred):**
- At each recursion step r, the router computes a scalar score for each token from its hidden state
- Top-k tokens are selected to continue to step r+1
- Only tokens selected at depth r can be re-evaluated at depth r+1 (hierarchical)
- Router: single linear projection with sigmoid activation

**Token-Choice:**
- Each token gets a single routing decision at the outset
- Softmax over recursion depths, argmax selection
- Requires explicit balancing loss to prevent dead routes

### Loss Function

Primary loss: standard cross-entropy language modeling.

Auxiliary losses:
- **Expert-Choice:** Auxiliary loss to prevent causality violation (information leakage from future tokens). Weight: 0.001.
- **Token-Choice:** Balancing loss for load distribution + z-loss regularization.

### What Failed (Ablations)

- **Token-choice routing:** 40.0% vs expert-choice 42.6% few-shot accuracy
- **Tanh activation with auxiliary router:** 66.7% dead token ratio (tokens never routed to deeper steps)
- **Recursive KV sharing:** Significant degradation for expert-choice
- **MLP router:** Worse than simple linear router

**What worked:**
- Linear router + sigmoid activation + auxiliary loss (optimal combination)
- Middle-Cycle parameter sharing (reusing middle transformer layers for recursion)
- Recursion-wise KV caching (only cache tokens at their active recursion level)

### How Tokens are Assigned Depths

The router learns that **function words and simple tokens** exit early (1-2 recursions) while **content words, reasoning tokens, and ambiguous tokens** continue to deeper recursions. This emerges naturally from end-to-end training - no explicit curriculum or labels for "difficulty."

---

## 8. COCONUT Follow-ups (2025-2026)

### CODI (EMNLP 2025)

**Key innovation:** Self-distillation within a single model.

**Training:**
- Same model serves as both teacher (with text CoT) and student (with continuous thoughts)
- Teacher sees ground-truth CoT tokens (teacher forcing)
- Student generates 6 continuous thought tokens (fixed)
- **Distillation loss:** L1 distance between teacher and student hidden states at a designated token (the colon in "The answer is:")
- Combined loss: `L = alpha*L_teacher + beta*L_student + gamma*L_KD`
- Stop-gradient on teacher: student must match teacher, not vice versa

**What failed:**
- Without L1 loss: 24.5% accuracy (vs 43.7%)
- Including final CoT step in alignment: 31.7% (model takes shortcuts)
- Static pre-trained teacher: 27.1% (teacher must co-evolve)
- Optimal: 6 continuous thoughts (matches average CoT step count)

### MarCos (2025)

**Key innovation:** Models reasoning as a hidden Markov chain with stochastic transitions.

**Two-phase training:**
1. Phase 1: Optimize reconstruction + sparsity losses (randomness predictor frozen)
2. Phase 2: Train randomness predictor to minimize KL divergence (all else frozen)

**Loss:** `L = sum[-L_re + L_KL + lambda*L_sparse]`
- Reconstruction: likelihood of generating ground-truth reasoning steps
- KL: between learned prior and posterior for randomness variables
- Sparsity: encourages each dimension to represent isolated factors

**What improved over COCONUT:**
- 8.66% improvement on GSM8K
- 15.7x faster than token-based CoT
- Critical: removing sparsity loss causes complete model collapse

### CoT2 (2025)

**Key innovation:** Does NOT distill from discrete CoT. Trains continuous representations directly.

**Training stages:**
1. **CSFT (Continuous Supervised Fine-Tuning):** Target distributions reflect empirical frequency of states reachable at each step. Uses teacher forcing. Loss: cross-entropy/KL between predicted and target distributions.
2. **GRPO:** Policy optimization with multi-token sampling (K discrete tokens averaged into one continuous token) or Dirichlet sampling. Geometric mean of policy ratios for stability.

**What makes CoT2 different from COCONUT:**
- No progressive replacement curriculum
- No initialization from discrete CoT
- Directly optimizes continuous representations
- Can do RL on top of continuous thoughts (COCONUT cannot)
- Explicit supervision of parallel trace tracking

### CoLaR (NeurIPS 2025)

**Key innovation:** Compressed embedding prediction with RL.

**Two-stage training:**
1. **SFT:** Next-token prediction + auxiliary next-compressed-embedding prediction. Compression factor c randomly sampled from [1, c_max=5]. Latent head (2-headed MLP) predicts mean and variance of next compressed embedding.
2. **RL (GRPO):** Binary reward (correct/incorrect). Per-token averaged rewards encourage shorter reasoning. Latent head's stochastic nature enables exploration.

**What failed:**
- Deterministic latent head: worse (no exploration for RL)
- Removing compressed token supervision: -1.6% accuracy
- Mean pooling compression: -3.4% (distribution distortion)
- Cannot generalize to non-integer compression factors or beyond c_max

---

## 9. Quiet-STaR (2024)

**Paper:** arxiv 2403.09629

### How Thinking is Trained

Quiet-STaR inserts **thought tokens at every position** in the sequence and trains them to improve next-token prediction.

**Architecture:**
```
..., token_j, <start_thought>, t1, t2, ..., tk, <end_thought>, token_{j+1}, ...
```

Thoughts are generated in parallel at every position (not just special positions).

### Loss Function (exact)

Two components:

**NLL Loss:**
```
L_NLL = -log p_{j:j+n_true}^talk(X_{j+1:j+n_true+1})
```
Predicting n_true tokens ahead (non-myopic), not just the next token.

**REINFORCE Loss:**
```
grad L_REINFORCE = -r_j * grad log p_theta(T_j | [X_{:j}, <start>])
```
where the reward r_j compares the thought-assisted prediction against the base prediction:
```
r_j = log p^talk_{j:j+n_true}(X_{j+1:j+n_true+1}) - log p_bar^talk_{j:j+n_true}(X_{j+1:j+n_true+1})
```
Only positive rewards used (negative rewards excluded for stability).

### The Mixing Head

A shallow MLP that takes concatenated hidden states from end-of-thought position and original token position, outputs a scalar weight for interpolation between thought-assisted and base predictions. This is crucial: without it, "the model quickly learned to simply ignore the thoughts."

### The Non-Myopic Loss

Instead of predicting only the next token, the loss includes n_true future tokens using teacher forcing. Ablations showed predicting 2+ tokens ahead improved CommonsenseQA by ~3.1%.

### Gradient Flow Through Thoughts

- Thought content: trained via REINFORCE (non-differentiable sampling)
- Mixing head: trained via direct backprop (differentiable interpolation)
- Start/end tokens: initialized to em-dash embedding, receive **amplified gradients** (weight 1e2)
- Policy gradient weight: 1e6 (very large to compensate for variance)

### What Failed

- **Without mixing head:** Model ignores thoughts entirely
- **Separate heads for thinking/talking:** Prevents learning
- **Gumbel-Softmax with straight-through estimator:** Vanishing gradients
- **Without non-myopic loss:** Significant degradation

### Training Hyperparameters

- Learning rate: 1e-6
- Optimizer: AdamW, warmup 20 steps, weight decay 0.001
- Batch size: 8
- Sequence spans: random 256 tokens
- Start/end embedding gradient weight: 1e2
- Policy gradient weight: 1e6
- Temperature: T=1 during training
- Thoughts per position: 4-16 tokens (tested 4, 8, 12, 16)
- Rationale samples: 2-4 per position

---

## 10. Universal Transformers (ICLR 2019)

**Paper:** arxiv 1807.03819

### How Thinking is Trained

The Universal Transformer shares weights across depth (recurrent), with optional ACT (Adaptive Computation Time) to learn per-position halting.

**ACT mechanism:**
- At each step, compute halting probability: `p = sigmoid(Linear(h_n))` for each position
- Cumulative probability tracked; position halts when cumulative > threshold (0.9)
- Once halted, state is copied to subsequent steps until all halt or max steps reached
- **Remainder mechanism:** `remainder = 1 - cumulative_before_halt`; final state weighted by remainder

**ACT Loss Penalty:**
```
L = L_task + tau * sum_positions(N(x_i))
```
where N(x_i) = number of steps position i took. The penalty tau directly penalizes computation.

### What Worked and Didn't

**Worked:**
- bAbI question answering: strong results
- Subject-verb agreement: UT matched/exceeded standard transformer
- Algorithmic tasks (copy, reverse, sort): UT with ACT matched/exceeded fixed-depth
- LAMBADA: ACT variant competitive (avg ponder time 8.2 ± 2.1 steps)

**Didn't work:**
- Machine translation: "marginally degraded results" with dynamic halting
- ACT is "notably unstable and sensitive to the choice of hyperparameter tau"
- Fixed-depth UT often comparable or superior to ACT variant on many tasks
- No curriculum learning used: "we do not use any curriculum learning strategy"

### Legacy

The Universal Transformer showed that weight-sharing + adaptive depth CAN work, but ACT's instability (tau sensitivity) motivated PonderNet. PonderNet's KL formulation was more principled but still collapsed at small scale, motivating LoopFormer's fixed-budget approach.

---

## Synthesis: What's the Best Practice for Training Models to Think?

### The Consensus Training Recipes

Looking across all 10 model families, there are **three proven paradigms** for teaching thinking:

#### Paradigm 1: Curriculum Distillation (COCONUT, CODI, CoT2)

**Recipe:**
1. Start with a pretrained model that can do text chain-of-thought
2. Gradually replace text reasoning with continuous representations
3. Supervise on remaining text predictions
4. Reset optimizer between curriculum stages

**Pros:** Works reliably at small scale. Clear training signal. Predictable training dynamics.
**Cons:** Requires CoT training data. Fixed thinking budget. Cannot discover novel reasoning strategies.

#### Paradigm 2: RL on Verifiable Tasks (DeepSeek-R1, OpenAI o1/o3)

**Recipe:**
1. Start with a pretrained + SFT model
2. Use RL (GRPO or PPO) with rule-based rewards on verifiable tasks (math, code)
3. Format rewards to enforce think/answer structure
4. The model discovers thinking strategies through trial and error

**Pros:** Thinking emerges naturally. Adaptive thinking length. Can discover novel strategies (aha moment). Scales to production quality.
**Cons:** Requires verifiable tasks. Reward hacking risk. Expensive. Readability issues without cold-start SFT.

#### Paradigm 3: Energy/Optimization-Based (EBTs)

**Recipe:**
1. Train an energy function E(x, y_hat) on standard objectives
2. Backpropagate through iterative optimization trajectory
3. Use Langevin noise, replay buffers, random step sizes for robustness
4. At inference: run gradient descent on energy, pick lowest-energy prediction

**Pros:** No RL needed. No CoT data needed. Self-verification is free. Adaptive thinking via more iterations. Emerges from unsupervised pretraining.
**Cons:** Requires second-order gradients. Expensive training. Novel and less tested.

### The Lessons for Our Matrix Thinking Model

Based on this research, here is what we should do:

#### 1. Use LoopFormer-Style Consistency Training (Not PonderNet)

PonderNet's halting collapsed for us at small scale. This is a known failure mode when per-step improvement is small relative to the KL regularization. LoopFormer's approach avoids this entirely:
- **No learned halting.** User sets compute budget at inference.
- **Shortcut-consistency loss** ensures representations are useful at EVERY depth.
- **Time/step-size conditioning** via Fourier features + AdaLN makes the model depth-aware.

We already identified this in STATE.md Phase 1 item 4. The research confirms this is the right call.

#### 2. Consider COCONUT-Style Curriculum for Freeform Thought Slots

For Phase 4 (freeform matrix thinking), COCONUT's curriculum is directly relevant:
- Start with text CoT training data
- Progressively replace text reasoning steps with matrix thought slots
- Mask loss on thought positions (already in our plan)
- Reset optimizer between stages
- Use c=1 or c=2 thoughts per reasoning step (not more)

**Critical lesson:** Training without curriculum (directly in latent space) does NOT work. COCONUT showed this conclusively.

#### 3. For Phase 5 (Adaptive Thinking), Use Expert-Choice Routing

Mixture-of-Recursions showed that:
- Simple linear router + sigmoid + auxiliary loss works best
- End-to-end training (no separate router training)
- Expert-choice routing beats token-choice
- The router naturally learns which tokens need more computation

This is directly applicable to our certainty-driven thinking mode switching.

#### 4. Energy-Based Thinking as a Long-Term Direction

EBTs are the most elegant solution: define an energy function over matrix-valued representations, iterate via gradient descent, and self-verify via energy comparison. The matrix representation is naturally suited to energy-based methods because:
- Matrix energy = trace(M^T M) or nuclear norm
- Gradient descent on matrices is well-studied
- The 16x16 matrix has enough capacity for energy landscape learning

However, this requires second-order gradients (expensive with d=16 matrices) and is less tested. Consider as Phase 6+.

#### 5. The RL Path (Phase 7+)

Once we have verifiable task data and a working thinking model, we can apply DeepSeek-style GRPO:
- Rule-based rewards (math correctness, code execution)
- Format rewards for think/output structure
- No neural reward model (reward hacking risk)
- This is how thinking goes from "functional" to "good"

### Priority Order for Our Architecture

1. **Now:** LoopFormer-style consistency training for iterative refinement (replaces PonderNet)
2. **Phase 4:** COCONUT-style curriculum for freeform thought slots
3. **Phase 5:** MoR-style routing for adaptive thinking
4. **Later:** EBT-style energy minimization for self-verification
5. **Scale:** DeepSeek-style RL for emergent thinking strategies

### What NOT to Do

- Do NOT use PonderNet halting (collapses at our scale)
- Do NOT train latent thoughts from scratch without curriculum (COCONUT proved this fails)
- Do NOT use neural reward models for RL (reward hacking)
- Do NOT use separate thinking/output heads without interpolation (Quiet-STaR proved thoughts get ignored)
- Do NOT use fixed step sizes for energy-based training (kills generalization)
- Do NOT skip optimizer reset between curriculum stages (COCONUT requires this)

---

## Sources

- [COCONUT: Training LLMs to Reason in Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [DeepSeek-R1: Incentivizing Reasoning via RL](https://arxiv.org/abs/2501.12948)
- [OpenAI o1 System Card](https://cdn.openai.com/o1-system-card-20241205.pdf)
- [Let's Verify Step by Step (PRM800K)](https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf)
- [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407)
- [Energy-Based Transformers](https://arxiv.org/abs/2507.02092)
- [LoopFormer (ICLR 2026)](https://arxiv.org/abs/2602.11451)
- [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524)
- [CODI: Compressing CoT via Self-Distillation](https://arxiv.org/abs/2502.21074)
- [MarCos: Deep Thinking by Markov Chain of Continuous Thoughts](https://arxiv.org/abs/2509.25020)
- [CoT2: Continuous Chain of Thought](https://arxiv.org/abs/2505.23648)
- [CoLaR: Dynamic Latent Compression of Reasoning Chains](https://arxiv.org/abs/2505.16552)
- [Quiet-STaR: Language Models Can Teach Themselves to Think](https://arxiv.org/abs/2403.09629)
- [Universal Transformers (ICLR 2019)](https://arxiv.org/abs/1807.03819)
- [o1: A Technical Primer (LessWrong)](https://www.lesswrong.com/posts/byNYzsfFmb2TpYFPW/o1-a-technical-primer)
- [Inside Reasoning Models: OpenAI o3 and DeepSeek R1](https://labs.adaline.ai/p/inside-reasoning-models-openai-o3)
- [COCONUT GitHub Repository](https://github.com/facebookresearch/coconut)
- [LoopFormer GitHub Repository](https://github.com/armenjeddi/loopformer)
- [Mixture-of-Recursions GitHub Repository](https://github.com/raymin0223/mixture_of_recursions)
