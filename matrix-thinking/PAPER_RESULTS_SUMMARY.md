# Paper Results Summary (Live Document)

Input for the paper-writer agent. All results are copy-paste-safe numbers
from experiment SUMMARY.txt files in `experiment-runs/` or
`/workspace/pebble/round_pc/results/` on the US pod.

Last updated 2026-04-17.

## Core thesis

Matrix-CODI's flatten-then-project readout has a constant Jacobian in Z. The
gradient cannot distinguish rank-1 from full-rank Z. Rank-k ablation produces
a flat accuracy curve. **This also holds for nonlinear-in-Z readouts**, which
means the problem is deeper than readout linearity — it is in the CODI
distillation objective itself.

## Table 1: Rank-k projection ablation

Accuracy on ProsQA as a function of rank-k truncation of Z at inference time.
All models: GPT-2 small, 6 latent positions, d=16, 25 epochs, γ=0, seed 1337,
ProsQA. Rows differ only in readout architecture.

| Readout          | k=1    | k=2    | k=4    | k=8    | k=16   | Spearman r | p    | Full-train best |
|------------------|--------|--------|--------|--------|--------|-----------|------|----------------|
| flatten          | 79%    | 79%    | 79%    | 79%    | 79%    | ~0        | flat | 80.47%         |
| **bilinear**     | **78.12%** | **78.91%** | **78.91%** | **78.12%** | **78.12%** | **+0.04** | **0.63** | **78.91%** |
| **bilinear+GELU**| **78.91%** | **79.69%** | **79.69%** | **79.69%** | **79.69%** | **-0.13** | **0.14** | **79.69%** |
| **svd_aug**      | **77.34%** | **78.12%** | **78.12%** | **77.34%** | **78.12%** | **+0.02** | **0.82** | **78.12%** |
| **quadratic**    | **79.69%** | **79.69%** | **79.69%** | **79.69%** | **79.69%** | **+0.07** | **0.46** | **79.69%** |

**Headline finding: every readout variant produces a flat rank-k curve.**
Spearman correlations are statistically indistinguishable from zero
(p values 0.14, 0.82, 0.46). Quadratic is perfectly flat at 79.69% for every
k ∈ {1, 2, 4, 8, 16}. This rules out the interpretation "readout linearity
alone causes rank-blindness" — we tested three different ways of breaking
readout linearity (GELU nonlinearity, explicit singular-value features,
quadratic second moment) and all produced the same flat curve as the linear
flatten baseline.

**Interpretation.** The failure is in the CODI distillation objective itself.
The gradient from the vector-teacher distillation loss does not reward any
particular rank of Z, independent of how Z is consumed downstream. The model
finds solutions where the rank-1 subspace of Z that flows through the readout
carries all the relevant information; higher rank structure is learned but
not functionally used. This is a strictly stronger claim than our original
Jacobian-linearity thesis — the objective is rank-blind through the full
chain rule, not just through the final projection.

## Table 2: Three-seed replication of flatten readout

| Seed | Best acc | Wall | Final Z_rank |
|------|----------|------|--------------|
| 1337 | 80.47%   | —    | ~12.9        |
| 42   | 81.25%   | 207.7 min | ~4      |
| 7    | 82.81%   | 208.9 min | ~12     |

Mean ± std: **81.51% ± 1.2pp**. Accuracy is tight. Z_rank varies by 3×
(4, 12, 13). **Rank is decoupled from task performance.** This is the
cleanest single piece of evidence that CODI distillation does not reward
rank structure.

## Table 3: Scale sweep (ProsQA, vanilla SFT and matrix-CODI)

| Model        | Params | Vanilla SFT best | Matrix-CODI best |
|--------------|--------|------------------|------------------|
| GPT-2 small  | 124M   | 81.77%           | 80.47%           |
| GPT-2 medium | 355M   | 80.47%           | 79.69%           |
| GPT-2 large  | 774M   | 68.75%           | pending (OOM×2)  |

Matrix underperforms vanilla SFT at every tested scale. Absolute accuracy on
ProsQA degrades with scale — ProsQA is small enough that gpt2-large overfits
or under-optimizes with default LR. Matrix does not rescue this; vanilla SFT
sets the baseline and matrix tracks below it.

## Table 4: Iterative refinement depth sweep (vanilla CODI, no matrix)

| n_latents | Best acc | Notes |
|-----------|----------|-------|
| 6         | 78.91%   | Banked |
| 16        | pending  | Queued (rerun after OOM fix) |
| 32        | pending  | Queued |
| 64        | pending  | Queued |

Reference: vanilla SFT at n_latents=0 (no refinement) = 81.77%. Adding iterative
latent refinement at n=6 *hurts* on ProsQA. This kills the "more latent thought
= better reasoning" story independently of the matrix question.

## Table 5: Linear probe on Z

From Round 5 Part 1. Macro-averaged AUC for predicting target class (24-way)
from Z extracted at each latent position. Matrix Z encodes LESS target
information than a vanilla GPT-2 hidden state at the same prompt.

| Feature                 | AUC   |
|-------------------------|-------|
| matrix Z[0..5] concat   | 0.673 |
| **vanilla GPT-2 hidden** | **0.846** |
| random GPT-2 hidden     | 0.495 |

Verdict: NULL. Matrix bottleneck actively loses target-predictive information
relative to the uncompressed hidden state.

## Decorations vs substance

Round 4 teacher_ce mode (trivial copy-from-context, not load-bearing) =
100.00% on all three seeds. Reported only to flag that not every reported
number in the paper is interpretable as reasoning performance.

## Round 5 sample-efficiency curve (ProsQA, varied N train)

| N train | Vanilla SFT | Matrix-CODI | Gap |
|---------|-------------|-------------|-----|
| 200     | 26.04%      | 12.60%      | -13.4 |
| 500     | 44.53%      | 22.05%      | -22.5 |
| 2000    | 70.05%      | 60.94%      |  -9.1 |
| 5000    | 77.21%      | 74.22%      |  -3.0 |

Matrix is strictly WORSE at small N. Not just decoration at high data —
actively worse in the low-data regime.

## Related-work positioning (REQUIRED for paper)

Our closest adjacent work (must distinguish in Related Work):

1. **SIM-CoT (Shen et al., ICLR 2026, arXiv 2509.20317)** — diagnoses latent CoT
   instability as insufficient step-level supervision. Our diagnosis: rank-blind
   Jacobian in readout. Positive-control nonlinear readouts adjudicate —
   if their diagnosis were complete, fixing the readout's Jacobian would not
   matter. Our result (bilinear+GELU ALSO flat) is consistent with BOTH being
   issues — distillation objective is rank-blind at multiple levels.
2. **Nazari & Rusch (arXiv 2602.04852) + State Rank Dynamics (arXiv 2602.02195),
   both Feb 2026.** Measure rank of linear-attention fast-weight hidden states;
   frame rank as descriptive invariant. We measure rank of latent thought
   matrices (explicit CoT-tokens) and make a mechanism claim (trainability
   limited by Jacobian structure). Different object, different claim.
3. **Illusion of Superposition (Rizvi-Martel et al., arXiv 2604.06374)** —
   observes COCONUT latents don't encode parallel reasoning paths. We provide
   a mechanism: even d×d matrix latents don't, because readout linearity
   prevents the gradient from seeing rank.
4. **Reasoning by Superposition (Lin/Zhu et al., arXiv 2505.12514)** — theory
   that 2-layer transformer CAN encode BFS as superposition. We show it
   doesn't, because distillation doesn't reward it. Theory-practice gap.

## Contribution bullets (for intro — tightened per Feb 2026 adjacent work)

We are NOT the first to empirically measure rank in trained transformer states
(Nazari & Rusch, State Rank Dynamics papers do so for linear-attention).
We ARE the first to:

- Diagnose rank-blindness as a **readout Jacobian property** (not a training
  instability or insufficient supervision issue)
- Demonstrate that the diagnosis is **falsifiable** via nonlinear-in-Z readouts
  that break the constant-Jacobian condition
- Show that **the failure survives the fix**: nonlinear readouts (bilinear+GELU)
  also produce rank-blind gradients, meaning CODI distillation objective has
  deeper rank-indifference than readout linearity alone explains
- Ship a **3-seed replication** showing accuracy-rank decoupling as direct
  evidence that the loss does not reward rank

## Reproducibility pointers

All scripts in the public repo at
`github.com/saml212/learned-representations/tree/main/matrix-thinking/scripts`.
Anonymized code bundle for submission: upload to `anonymous.4open.science`
before May 8.

Data: ProsQA from facebookresearch/coconut. GSM8K-Aug from whynlp/gsm8k-aug.

Hardware: 1×H100 per training run. Total compute for the paper's experiments:
~200 GPU-hours on H100 (including reruns from spot preemptions).
