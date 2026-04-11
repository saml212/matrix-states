# Preventing Boundary Collapse in Learned Byte-Level Segmentation

## The Problem

A Gumbel-softmax segmenter predicting per-byte boundary probabilities collapses to
near-zero boundaries (all-merge) early in training. The model finds it easier to
process long unsegmented sequences than to learn useful boundaries, because the
language modeling loss initially improves faster by ignoring boundaries entirely.

This is a well-documented failure mode across multiple papers (SOMBRERO, H-Net,
MrT5, Zonkey). The core issue: the LM loss gradient through the segmenter is
weak and noisy early in training, while the path of least resistance is to
suppress all boundaries.

---

## Technique 1: Ratio Loss (from H-Net / SOMBRERO)

**The most battle-tested approach.** Used in H-Net (2025) and SOMBRERO (2025).

### Formulation

```
L_ratio = (N / (N-1)) * ((N-1)*F*G + (1-F)*(1-G))

where:
  F = (1/L) * sum(b_t)       # actual fraction of boundaries (hard decisions)
  G = (1/L) * sum(p_t)       # average boundary probability (soft scores)
  N = L / C_tar              # target number of segments
  C_tar                      # target compression ratio (e.g., 4-6 for text)
```

The loss minimizes when F = G = 1/N. Crucially, it uses BOTH soft scores and
hard decisions, which stabilizes learning by providing gradient signal even when
hard boundaries haven't formed yet.

### Hyperparameters (from SOMBRERO)

- **C_tar = 4.0-6.0** for text (4.6 matches BPE compression on typical English)
- **Loss weight lambda_ratio = 1.0** (SOMBRERO found 0.03 insufficient; needed 1.0)
- Combined loss: `L = L_lm + 1.0 * L_ratio + 0.01 * L_cab`

### Implementation

```python
def ratio_loss(boundary_probs, boundary_hard, seq_len, target_compression):
    """
    boundary_probs: (B, L) soft boundary probabilities in [0,1]
    boundary_hard: (B, L) hard boundary decisions in {0,1}
    target_compression: e.g. 4.0
    """
    N = seq_len / target_compression  # target number of segments
    F = boundary_hard.float().mean(dim=-1)  # actual boundary fraction
    G = boundary_probs.mean(dim=-1)          # soft boundary fraction
    loss = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
    return loss.mean()
```

---

## Technique 2: Byte-Level Confidence Smoothing (from SOMBRERO)

**Critical for preventing early-training overcompression with sigmoid boundary predictors.**

### Problem It Solves

With chunk-level smoothing (only applying boundary effects at chunk boundaries),
gradient signal is sparse -- most byte positions get zero gradient for boundary
learning. The sigmoid predictor drifts toward all-merge because low-confidence
boundaries don't generate any learning signal.

### Solution

Apply confidence-weighted smoothing at EVERY byte position, not just at chunk boundaries:

```
x_bar_t = c_t * x_hat_t + (1 - c_t) * x_bar_{t-1}

where c_t = boundary_confidence at position t
```

This gives "dense gradient flow to every boundary score rather than only those
that cross a discretization threshold." Every byte position contributes gradient
to the boundary predictor, preventing the early-training death spiral.

### Implementation

```python
def byte_level_smooth(byte_representations, boundary_probs):
    """
    Instead of hard chunking, apply EMA-like smoothing using boundary probs.
    boundary_probs near 1.0 = "start new segment" (use current repr)
    boundary_probs near 0.0 = "continue segment" (carry forward)
    """
    B, L, D = byte_representations.shape
    smoothed = torch.zeros_like(byte_representations)
    smoothed[:, 0] = byte_representations[:, 0]
    for t in range(1, L):
        c = boundary_probs[:, t:t+1]  # (B, 1)
        smoothed[:, t] = c * byte_representations[:, t] + (1 - c) * smoothed[:, t-1]
    return smoothed
```

---

## Technique 3: Confidence-Alignment Boundary Loss / CAB (from SOMBRERO)

**Provides a direct supervisory signal for WHERE to place boundaries.**

### Formulation

```
L_cab = (1 - sg[P_{t+1}] - p_t)^2

where:
  P_{t+1} = p_theta(x_{t+1} | x_{1:t})  # model's predicted probability for next byte
  p_t = boundary score at position t
  sg = stop-gradient operator
  Both terms clamped to [1e-6, 1 - 1e-6]
```

This aligns boundary placement with prediction difficulty: place boundaries where
the next byte is hard to predict (low P_{t+1} -> high 1-P_{t+1} -> high boundary
score encouraged). The stop-gradient prevents circular optimization.

### Hyperparameters

- **Loss weight: lambda_cab = 0.01** (small but crucial for guiding boundary placement)

---

## Technique 4: PI-Controller Adaptive Regularization (from MrT5)

**Automatically adjusts regularization strength to maintain target deletion/boundary rate.**

### How It Works

Instead of a fixed regularization weight, use a proportional-integral controller
from control theory to dynamically adjust the weight:

```
# Proportional term (with exponential moving average)
p_{t+1} = gamma * p_t + (1 - gamma) * (delta - delta_hat_t)

# Integral term (accumulates error)
i_{t+1} = i_t + (delta - delta_hat_t)

# Regularization weight
alpha_{t+1} = clamp(k_p * p_{t+1} + k_i * i_{t+1}, min=0)

where:
  delta = target boundary/deletion rate (e.g., 0.25 for 4x compression)
  delta_hat_t = current observed boundary rate
  gamma = 0.9 (EMA smoothing)
  k_p = 0.5 (proportional gain)
  k_i = 1e-5 (integral gain, kept small to prevent windup)
```

### Why This Helps

Fixed regularization weights require careful tuning and can't adapt as training
dynamics shift. The PI-controller automatically increases alpha when boundaries
are too few (all-merge) and decreases it when boundaries are too many (all-split).

### Implementation

```python
class BoundaryPIController:
    def __init__(self, target_rate=0.25, k_p=0.5, k_i=1e-5, gamma=0.9):
        self.target = target_rate
        self.k_p = k_p
        self.k_i = k_i
        self.gamma = gamma
        self.p_term = 0.0
        self.i_term = 0.0

    def step(self, current_rate):
        error = self.target - current_rate
        self.p_term = self.gamma * self.p_term + (1 - self.gamma) * error
        self.i_term = self.i_term + error
        alpha = max(0.0, self.k_p * self.p_term + self.k_i * self.i_term)
        return alpha
```

---

## Technique 5: Binomial Prior Regularization (from MAGNET / MANTa)

**Controls compression ratio by penalizing deviations from a target boundary rate.**

### Formulation

For a sequence of length l with k predicted boundaries:

```
L_reg = -log Binomial(k; l, beta)
      = -log C(l,k) - k*log(beta) - (l-k)*log(1-beta)

where:
  beta = target boundary probability per position
  beta is set based on byte-to-word ratio for the script/domain
  For English text: ~4-5 bytes per word -> beta ~ 0.2-0.25
```

The binomial prior nudges the model toward having k close to beta*l boundaries.
MAGNET sets beta per-script based on the average byte-to-word ratio of an anchor
language, ensuring equitable compression across scripts.

### Hard Gumbel-Sigmoid Sampling

MAGNET uses hard Gumbel-sigmoid for boundary decisions:

```python
def gumbel_sigmoid(logits, tau=1.0, hard=True):
    """Gumbel-sigmoid for binary boundary decisions."""
    gumbel_noise1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    gumbel_noise2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + gumbel_noise1 - gumbel_noise2) / tau)
    if hard:
        y_hard = (y_soft > 0.5).float()
        # Straight-through estimator: forward uses hard, backward uses soft
        return y_hard - y_soft.detach() + y_soft
    return y_soft
```

---

## Technique 6: Architectural Bottleneck (from MANTa)

**Force the model to compress by architectural design, not just loss terms.**

### The MANTa Approach

MANTa avoids explicit boundary regularization entirely. Instead:

1. Predict soft boundary probabilities (frontier probabilities) per byte
2. Compute a SOFT assignment of bytes to blocks using truncated Gaussian kernel:
   ```
   mu_i = sum(p_F_k for k <= i)     # cumulative frontier probability
   sigma_i = sqrt(sum(p_F_k * (1-p_F_k) for k <= i))
   P(block=k, byte_i) ~ Gaussian(mu_i, sigma_i)(k) / Z
   ```
3. Truncate the block sequence to L_hat = min(L_B, L/K) where K=4

The K=4 truncation is the key: it forces at most L/4 blocks, which means the
model MUST produce boundaries at a rate of at least 1 per 4 bytes. Without this,
MANTa collapses to L_B = L (one block per byte).

### Key Insight

"Without this bottleneck, truncation incentivizes the frontier predictor to
produce sufficiently long blocks." The architectural constraint removes the need
for explicit regularization and the model naturally converges to sharp (~1e-5
uncertainty) boundary decisions.

### Advantages

- No regularization hyperparameters to tune
- Naturally converges to hard boundaries without temperature annealing
- Compression ratio is guaranteed by architecture

### Implementation Guidance

For your setup with sequence length 1024:
- Set max_segments = 1024 / K where K = 4-8
- Use soft byte-to-segment assignment via the Gaussian kernel
- The decoder only processes max_segments positions

---

## Technique 7: Scaled Sigmoid Gate with Negative Initialization (from MrT5)

**Initialize the gate to KEEP tokens, not delete them, to prevent early collapse.**

### The MrT5 Gate

```python
class ScaledSigmoidGate(nn.Module):
    def __init__(self, hidden_size, scale=-30.0):
        super().__init__()
        self.layer_norm = T5LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.scale = scale  # k = -30

    def forward(self, hidden_states):
        h = self.layer_norm(hidden_states)
        logits = self.linear(h)
        # Output range: [scale, 0] = [-30, 0]
        gate = self.scale * torch.sigmoid(-logits)
        return gate
```

Gate values near 0 = keep token, values near -30 = delete token.
Threshold at k/2 = -15 for hard decisions.

### Key Design Choice: Softmax_1 Attention

When all gate values are at the minimum (k=-30), standard softmax still produces
uniform attention weights, which can create degenerate solutions. MrT5 uses
softmax_1 (aka "quiet attention"):

```python
def softmax1(x, dim=-1):
    """Softmax with +1 in denominator. Allows attention to be zero."""
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return e_x / (1.0 + e_x.sum(dim=dim, keepdim=True))
```

This allows the attention mechanism to express "attend to nothing" when all
tokens in a region have been deleted, preventing the "delete everything" collapse.

---

## Technique 8: Cosine-Similarity Boundary Prediction (from H-Net / SOMBRERO)

**An alternative to sigmoid that naturally handles boundary detection.**

### Formulation

```
q_t = W_q * x_t,    k_t = W_k * x_t
p_t = 0.5 * (1 - cos_sim(q_t, k_{t-1}))
b_t = 1 if p_t >= 0.5 else 0
```

When consecutive representations are similar, boundary probability is low.
When they differ, boundary probability is high. This has a natural inductive
bias: boundaries occur where content changes, which is often the right answer.

### Compared to Sigmoid

SOMBRERO found that sigmoid initially suffers from "overcompression bias"
(drifting to all-merge) but works better once byte-level smoothing is added.
Cosine similarity is more stable out-of-the-box but potentially less expressive.

---

## Technique 9: Entropy-Based Boundary Placement (from BLT)

**Sidestep learned boundaries entirely with an entropy oracle.**

BLT avoids the collapse problem by not learning boundaries during training.
Instead, a pre-trained 100M parameter byte-level LM computes per-byte entropy:

```
H(x_t) = -sum_v p_e(x_t=v | x_{<t}) * log p_e(x_t=v | x_{<t})
```

Boundaries are placed where entropy exceeds a threshold (global constraint) or
where entropy rises sharply (monotonic constraint):

- **Global:** H(x_t) > theta_G
- **Monotonic (preferred):** H(x_t) - H(x_{t-1}) > theta_r

The threshold is calibrated to achieve a desired average patch size.

### Hybrid Approach for Your Project

You could use entropy-based boundaries as a warm-start signal:
1. Pre-train a small byte LM (even 1M params with sliding window)
2. Compute entropy on your training data
3. Initialize boundary predictor bias to match entropy-based boundaries
4. Then fine-tune the boundary predictor end-to-end

---

## Technique 10: Existence-Share Normalization (from Zonkey)

**Prevent the splitter from gaming easy regions while neglecting hard ones.**

### Problem

Without normalization, the segmenter can concentrate boundaries in predictable
text regions (where reconstruction is easy) while sparsely covering complex regions.

### Solution

Zonkey computes "existence shares" -- normalized per-position weights that ensure
uniform contribution regardless of overlap density:

```
raw_share_i = product(1 - p_BOS_k for k in range(start, i))
normalized_share_i = raw_share_i / sum(raw_shares)
```

All losses are weighted by these normalized existence shares, ensuring every
region of the input contributes proportionally to the training signal.

### Additional Regularization from Zonkey

1. **Average BOS penalty:** Discourages overly frequent splits by penalizing
   high average boundary probability across the batch
2. **Segment length penalties:** Large loss for segments below min_length or
   above max_length (e.g., below 4 or above 32)
3. **BOS cross-entropy loss:** Treats boundary prediction as a reconstruction
   task where the model guesses linguistically plausible boundaries

---

## Recommended Recipe for This Project

Based on synthesizing all the above, here is a concrete implementation plan
ordered by priority:

### Step 1: Boundary Predictor Architecture

Use a small 1-layer transformer (or even a 2-layer MLP) on top of byte embeddings.
Output: per-byte logit for boundary probability.

```python
class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_boundary=64):
        super().__init__()
        self.proj = nn.Linear(d_model, d_boundary)
        self.norm = nn.LayerNorm(d_boundary)
        self.out = nn.Linear(d_boundary, 1)
        # CRITICAL: Initialize bias to produce ~25% boundary rate
        # sigmoid(bias) should equal target_rate
        # For target_rate=0.25: bias = log(0.25/0.75) = -1.1
        nn.init.constant_(self.out.bias, -1.1)
        nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, byte_embeddings, tau=1.0, hard=True):
        h = torch.relu(self.proj(byte_embeddings))
        h = self.norm(h)
        logits = self.out(h).squeeze(-1)  # (B, L)
        boundaries = gumbel_sigmoid(logits, tau=tau, hard=hard)
        return boundaries, logits
```

### Step 2: Combined Loss Function

```python
# Total loss
L = L_lm + lambda_ratio * L_ratio + lambda_cab * L_cab

# Recommended weights
lambda_ratio = 1.0    # Must be large enough (SOMBRERO found 0.03 insufficient)
lambda_cab = 0.01     # Small but guides boundary placement
target_compression = 4.0  # For English text (~4 bytes per word)
```

### Step 3: PI-Controller for Adaptive Regularization

Use MrT5's PI-controller to dynamically adjust lambda_ratio if the fixed
weight doesn't produce the desired compression:

```python
controller = BoundaryPIController(
    target_rate=1.0/target_compression,  # 0.25 for C=4
    k_p=0.5,
    k_i=1e-5,
    gamma=0.9
)

# In training loop:
current_rate = boundaries.float().mean().item()
lambda_ratio = controller.step(current_rate)
```

### Step 4: Temperature Annealing (if using Gumbel-sigmoid)

```python
# Start high (soft, good gradients), anneal to low (hard, discrete)
tau_start = 2.0
tau_end = 0.1
tau_decay = 0.997  # per epoch, or use cosine schedule

# Per step:
tau = max(tau_end, tau_start * (tau_decay ** epoch))
```

### Step 5: Byte-Level Smoothing

Replace hard chunking with byte-level confidence smoothing in early training,
then transition to hard chunking:

```python
# Early training: use soft smoothing
smoothed = byte_level_smooth(representations, boundary_probs)

# After warmup (e.g., 20% of training): switch to hard chunking
if step > warmup_steps:
    chunks = hard_chunk(representations, boundary_hard)
```

### Step 6: Monitoring and Diagnostics

Track these metrics every N steps:
1. **Boundary rate:** `mean(boundaries)` -- should be near 1/C_tar
2. **Boundary entropy:** `mean(-p*log(p) - (1-p)*log(1-p))` -- should decrease over training
3. **Boundary sharpness:** `mean(min(p, 1-p))` -- should approach 0
4. **Mean segment length and variance** -- should show structure, not be uniform
5. **Boundary positions on sample text** -- visual inspection is irreplaceable

---

## Summary Table

| Technique | Source | Prevents | Complexity | Recommended |
|-----------|--------|----------|------------|-------------|
| Ratio loss | H-Net/SOMBRERO | All-merge and all-split | Low | YES - primary |
| Byte-level smoothing | SOMBRERO | Early overcompression | Medium | YES - critical for sigmoid |
| CAB loss | SOMBRERO | Misplaced boundaries | Low | YES - easy to add |
| PI-controller | MrT5 | Fixed weight sensitivity | Low | YES - replaces manual tuning |
| Binomial prior | MAGNET/MANTa | Wrong compression rate | Low | Alternative to ratio loss |
| Architectural bottleneck | MANTa | All-merge | Medium | Consider for Phase 2+ |
| Scaled sigmoid + softmax1 | MrT5 | Delete-everything collapse | Low | YES if using delete gates |
| Cosine-sim boundaries | H-Net/SOMBRERO | Overcompression bias | Low | Alternative to sigmoid |
| Entropy warm-start | BLT | Cold-start collapse | Medium | Nice to have |
| Existence-share normalization | Zonkey | Region exploitation | Medium | Consider for Phase 2+ |
| Negative bias initialization | MrT5/literature | Early collapse | Trivial | YES - always do this |
| Temperature annealing | Standard | Premature discretization | Trivial | YES - always do this |

---

## Key Insights from the Literature

1. **The ratio loss weight matters enormously.** SOMBRERO had to increase from
   0.03 to 1.0 to get stable compression. Don't be shy with this weight.

2. **Byte-level smoothing is more important than the boundary predictor architecture.**
   SOMBRERO showed that sigmoid + byte-level smoothing outperforms cosine similarity
   without it.

3. **You don't need Gumbel noise.** MrT5 found "the model performs well even
   without [Gumbel noise]." A straight sigmoid with straight-through estimator
   may be simpler and equally effective.

4. **Architectural bottlenecks can replace regularization entirely.** MANTa's
   approach of soft assignment with a fixed compression factor K=4 avoids all
   boundary regularization and still converges to sharp boundaries.

5. **The cold start is the hardest part.** Most collapses happen in the first
   1-5% of training. Consider warming up the boundary predictor separately, or
   using entropy-based boundaries as initialization.

6. **Monitor boundary entropy, not just boundary rate.** A model can have the
   "right" average boundary rate but with uniform 0.25 probabilities everywhere
   (no actual segmentation). Track that boundaries become sharp (bimodal: near 0
   or near 1).

---

## Sources

- [MAGNET: NeurIPS 2024](https://arxiv.org/abs/2407.08818) - Script-specific boundary predictors with binomial priors
- [MrT5: ICLR 2025](https://arxiv.org/abs/2410.20771) - PI-controller, scaled sigmoid, softmax1
- [MrT5 source code](https://github.com/jkallini/mrt5) - Delete gate and PI-controller implementation
- [BLT: Meta 2024](https://arxiv.org/abs/2412.09871) - Entropy-based patching (no learned boundaries)
- [BLT source code](https://github.com/facebookresearch/blt) - Entropy model and patcher
- [Zonkey: Jan 2026](https://arxiv.org/abs/2601.21768) - Existence shares, segment length penalties
- [MBLM: IBM 2025](https://arxiv.org/abs/2502.14553) - Fixed hierarchical chunking (no learned boundaries)
- [MBLM source code](https://github.com/ai4sd/multiscale-byte-lm) - Hierarchical architecture
- [MANTa: EMNLP 2022](https://arxiv.org/abs/2212.07284) - Soft assignment with truncated Gaussian, architectural bottleneck
- [H-Net / Dynamic Chunking 2025](https://arxiv.org/abs/2507.07955) - Ratio loss, cosine-similarity boundaries, EMA smoothing
- [SOMBRERO 2026](https://arxiv.org/abs/2601.22805) - Byte-level smoothing, CAB loss, comprehensive ablations
- [Softmax1 / Quiet Attention](https://www.evanmiller.org/attention-is-off-by-one.html) - +1 denominator trick
- [Gumbel-Softmax original paper: ICLR 2017](https://arxiv.org/abs/1611.01144) - Temperature annealing foundations
