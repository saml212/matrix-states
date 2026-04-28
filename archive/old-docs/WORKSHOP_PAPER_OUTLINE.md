# Workshop Paper Outline (target: ICML 2026 Mechanistic Interpretability Workshop)

**Deadline:** May 8, 2026 AoE. Non-archival. 4-8 page format.

**Working title:** The Gradient Does Not See Rank: A Structural Explanation for the Illusion of Superposition in Latent Chain-of-Thought

**Voice constraints** (apply throughout): matter-of-fact, research style, explicit. No "honest", "actually", "obviously", "clearly", "remarkable", "surprising", "unfortunately", hedging language, adjectives or adverbs of judgment. Short sentences over complicated ones. Say what was done, say the number, say what it implies.

## Section structure (6-page version)

### 1. Introduction (0.5 page)
- The dispute: Reasoning by Superposition (arXiv 2505.12514) argues continuous thought tokens carry multi-rank reasoning state; Illusion of Superposition (arXiv 2604.06374) shows behaviorally that fine-tuned latent reasoners collapse to single interpretations.
- Gap: Illusion makes the empirical case via behavioral probing. Nobody has given a mechanistic/structural reason.
- Contribution: a linear-algebra argument that flatten-then-project readouts on matrix latents cannot use rank, plus empirical receipts (four flat rank-k curves, nine killed architectural variants).
- Cite Illusion prominently as the load-bearing empirical precedent.

### 2. Setup: matrix-CODI (0.75 page)
- CODI architecture (Shen et al. arXiv 2502.21074)
- Our matrix-valued variant: `h → w_up: Linear(D, d²) → reshape(d, d) → optional (I+Δ)Z(I+Γ) thinker → flatten(d²) → w_down: Linear(d², D) → LayerNorm`
- d=16, 6 latent positions, GPT-2 124M and GPT-2 medium 355M
- Training: CODI teacher-student, L1-at-colon distillation
- Tasks: ProsQA (facebookresearch/coconut), GSM8K-Aug (whynlp/gsm8k-aug)

### 3. Result 1: Flat rank-k projection ablation across four conditions (1 page)
- For the trained matrix-CODI model, project Z to rank k via truncated SVD at eval time, measure accuracy
- Table: k ∈ {1, 2, 4, 8, 16} × (GSM8K-Aug γ=1 on/GSM8K-Aug γ=1 off/ProsQA γ=1 on/ProsQA γ=0 on/ProsQA γ=0 off)
- All curves flat to within 0.4-0.6pp
- Effective rank at k=16 ranges 5.5 (GSM8K) to 12.8 (ProsQA) — the bottleneck builds rank, but the rank is not read
- Spearman r(rank, correct) near zero in every condition

### 4. Result 2: Vanilla SFT matches matrix-CODI on ProsQA (0.5 page)
- 3 seeds of plain supervised fine-tuning on ProsQA (no CODI, no latents, no matrix bottleneck): 81.25%, 82.03%, 82.03%. Mean 81.77%.
- Matrix-CODI best (Round 2): 82.03%.
- Delta: 0.26pp. Within seed noise.
- Caveat: we are 15pp below Rizvi-Martel et al.'s published 96.6% for the same task. Pipeline/hyperparameter gap. Does not affect the relative comparison.
- [Round 6: same result at GPT-2 medium scale — insert when complete]

### 5. Result 3: Z encodes less target information than vanilla GPT-2 hidden state (0.75 page)
- Linear probe with 5-fold CV on the Round 3 Run 1 checkpoint
- Predict ProsQA target symbol from Z vs from vanilla pretrained GPT-2 hidden state at the same prompt
- Table: Z[0]..Z[5] AUC 0.609-0.693, Z[all] 0.673, vanilla GPT-2 0.846, random GPT-2 0.495
- Pre-registered threshold: Z AUC > max(vanilla, random) + 0.05 = 0.896 to count positive. Matrix Z: 0.673. Verdict: NULL.
- Per-position: peak at Z[1], decay to Z[5]. Opposite of "reasoning deepens over time."
- Binary target-vs-neg_target: 0.50-0.56 across all conditions. Chance. Z cannot distinguish correct from distractor.

### 6. The structural argument (1.25 page)

#### 6.1 Statement
For any readout `h_out = f(Z)` that is linear in Z, the Jacobian `∂h_out/∂Z` is a constant tensor independent of Z. Therefore the gradient of any loss through this readout carries no information about Z's singular structure, and training cannot preferentially reward Z configurations with particular rank properties.

#### 6.2 Proof (short)
`h_out = W · vec(Z) + b` with `W: R^{d²} → R^D`. `∂h_out/∂vec(Z) = W`, which does not depend on Z. For any loss L, `∂L/∂Z = (∂L/∂h_out) · W`, where the first factor may depend on Z (through downstream computation) but is a single vector-valued function, not a rank-aware one. The loss cannot distinguish two Z values with the same `W · vec(Z)` projection, so the training signal cannot reward specific singular structure.

#### 6.3 Which architectures this covers
Nine architectures we audited, all reduce to linear-in-Z under reshape:
1. Flatten-project (the current matrix-CODI readout)
2. K-probe bilinear `Σ_i u_i^T Z v_i` — stack probes → linear map on vec(Z)
3. Quadratic trace `Σ_i tr(A_i Z Z^T)` — linear map on the symmetric matrix space; the model places info in the top eigenvector
4. Spectral polynomial `[tr(Z), tr(Z²), tr(Z³), tr(Z⁴)]` — compresses to ≤d scalars per sample, orthogonally invariant
5. U·V^T factorization at train time — constrains Z's rank but the gradient still flows through the same flatten
6. Per-vocab bilinear LM head `Σ_r u_{w,r}^T Z v_{w,r}` — linear map `R^{d²} → R^{|V|}`, rank ≤ d²
7. Matrix-valued attention (d×d Q/K/V with Frobenius scores) — reshape-equivalent to standard attention with head_dim=d²
8. K-bigram outer-product embedding — destroys pretraining + attribution confound
9. Low-rank w_up factorization — constrains the ensemble subspace but not per-sample rank

Each is killed by the gauntlet (KILL_LIST.md) under the same argument.

#### 6.4 What this argument does NOT say
- Does not say matrix-valued representations are useless in general. HELM-style fully matrix-native architectures with no flatten in the forward pass remain unexplored.
- Does not say rank is never functional. It says the specific mechanism "rank tracks reasoning paths" cannot be tested by rank-k projection ablation on architectures with linear readouts.
- Does not say the matrix bottleneck is fundamentally worthless. It says that for CODI-style distillation on GPT-2 small at this operating point, it contributes nothing above vanilla SFT.

### 7. Related work (0.5 page)
- Reasoning by Superposition (2505.12514): constructs, does not measure
- Illusion of Superposition (2604.06374): measures behaviorally, does not explain mechanistically
- CODI (2502.21074): the base architecture
- COCONUT (2412.06769): the related continuous-thought approach
- Emergence of Superposition (2509.23365): training-dynamics argument, 2-layer toy model
- Latent-SFT / Chain of Superposition (2510.15522): tangentially related
- Bottleneck rank literature (Jacot 2209.15055)

### 8. Limitations (0.5 page)
- GPT-2 small and medium only. Scale effects at billion-parameter models are unexplored.
- ProsQA and GSM8K-Aug only. Not every reasoning task.
- Single matrix dimension d=16. Wider sweeps are not reported.
- Our absolute numbers are below published baselines by ~15pp (ProsQA) and ~36pp (GSM8K-Aug). The internal comparisons are consistent within our pipeline; the structural argument is scale- and baseline-independent.
- Four flat curves are four single-seed runs with varying task/γ/thinker, not four seeds of one condition. The structural argument makes multi-seed variance uninteresting.

### 9. Conclusion (0.25 page)
- Four flat curves + nine architectural variants + the linear-in-Z structural argument support the Illusion of Superposition position at a mechanistic level.
- The paper identifies a specific architectural property (linear-in-Z readouts) whose violation would be required for any future positive result on matrix-valued continuous reasoning.
- HELM-style fully matrix-native architectures are the unexplored direction.

## Figures / tables required

### Figure 1: Four flat rank-k curves (one panel)
- x-axis: k ∈ {1, 2, 4, 8, 16} (log scale)
- y-axis: accuracy % (normalized to percentage-of-best-k per curve, so all four fit on one scale)
- Four lines: Round 1, Round 2, Round 3 Run 1, Round 3 Run 2
- Caption: Spearman r(rank, correct) values per curve

### Figure 2: Per-position probe AUC
- x-axis: latent position index (0 through 5)
- y-axis: multi-class AUC (0.5 chance line drawn)
- Three lines: matrix Z, vanilla GPT-2 hidden (constant), random GPT-2 hidden (constant)
- Shows decay from Z[1] peak to Z[5]

### Table 1: Rank-k ablation results
- Rows: (Round, task, γ, thinker, Z rank)
- Columns: k=1, 2, 4, 8, 16, Spearman r
- All rounds + Round 6 at GPT-2 medium when complete

### Table 2: Vanilla SFT vs matrix-CODI
- Rows: vanilla SFT seeds 1337/42/7, matrix-CODI Round 2, Round 3 Run 1, Round 3 Run 2
- Columns: backbone, accuracy, wall time

### Table 3: Nine killed architectural variants
- Rows: each variant
- Columns: construction, fatal flaw (one line each), reference to KILL_LIST.md

## What needs to land before submission

1. **Round 6 GPT-2 medium results** (~3 hours from now). Adds the scale axis.
2. **Round 5 sample efficiency results** (auto-launches when Round 4 finishes). Adds the data-scale axis.
3. **Round 4 teacher_ce results** (~3 more runs after current). Completes the "CODI machinery contributes nothing beyond params" claim.
4. **Reproduce or diagnose the Illusion baseline gap** (one focused training run with Illusion's hyperparameters). Critical for reviewer acceptance.
5. **Figure generation**: Figure 1 and Figure 2 as SVG.
6. **Writeup**: 4-6 pages, the outline above.

Estimated time after Round 5/6 complete: 2-3 days of focused work to generate the paper. Then 2-3 days of revision passes. Targets May 8 deadline comfortably.

## What would make this a TMLR submission instead

- Scale grid to Pythia-410M or Llama-1B (~50 H100 hours)
- Seed variance on the matrix arm (3 seeds per condition for all rounds, ~18 runs)
- A positive control: nuclear-norm reward with depth-stratified accuracy test (~22 H100 hours)
- Matrix-dim sweep d ∈ {8, 16, 32, 64} at fixed task (~15 hours)
- Full mechanistic analysis: activation patching on Round 3 Run 1 checkpoint

Total TMLR delta: 6-10 weeks of work beyond the workshop submission.
