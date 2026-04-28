# Matrix-CODI Experiment Kill List

This document records every Round 4+ candidate experiment that was killed by the attack-agent waterfall in this session, with the specific fatal flaw(s). **Read this before proposing any new experiment in the matrix-CODI / rank-as-superposition direction.** Do not re-propose anything here without addressing its fatal flaw.

## Empirical results so far (4 flat curves)

| Round/Run | Task | gamma | Thinker | Best Acc | k=1 Acc | k=16 Acc | Spearman r |
|-----------|------|-------|---------|----------|---------|----------|------------|
| 1 | GSM8K-Aug | 1.0 | ON | 7.00% | 6.00% | 6.12% | −0.023 |
| 2 | ProsQA | 1.0 | ON | 82.03% | 78.4% | 78.4% | +0.026 |
| 3 Run 1 | ProsQA | 0.0 | ON | 78.91% | 76.8% | 76.6% | −0.105 |
| 3 Run 2 | ProsQA | 0.0 | OFF | 79.69% | 72.6% | 72.4% | +0.095 |

Four flat curves across four orthogonal conditions: 2 tasks × 2 distillation regimes × 2 thinker settings. Range across k is 0.4–0.6pp. The matrix bottleneck does not learn functional rank structure under any tested combination.

## Killed proposals

### 1. MNNS (Matrix-CoT2 on Minimum Non-Negative Subset Sum)
**Fatal flaws:**
- The cited CoT2 repo `github.com/yaof8/reasoning-with-cot2` **does not exist**. WebFetch returned 404. Research agent fabricated it.
- H2 (capacity bound `|S_t| ≤ d`) is **falsified by construction** at d=16. With 6 latent positions and the paper's 2^t frontier definition, frontiers at depth 5–6 are 32–64, all exceeding d=16. There is no clean regime where |S_t| sweeps across d.
- "Exact ground truth" frontier 2^t is the **unpruned search-tree size**, not the actual distinct partial sums. Any sensible MNNS instance has pruning that collapses 2^t to a much smaller number. The "exact" claim is wrong.
- CODI L1-at-colon distillation is structurally hostile to multi-path rank signals; would fail anyway.

### 2. K-bigram embedding (Priority 2 from QUEUE.md)
**Fatal flaws:**
- **Destroys GPT-2 pretraining.** Replacing GPT-2's `wte` with a freshly initialized k-bigram outer-product matrix throws away ~38M params of pretraining signal. Teacher path still uses pretrained `wte`, so L1 distillation compares "student with broken embeddings" vs "teacher with pretrained embeddings" — the KD objective becomes unachievable.
- **Attribution confound.** Rank-3 embedding mechanically guarantees Z has rank ≥ 3 at step 0. Any measured rank could be embedding bleed-through, not constructed by matrix operations. QUEUE.md flags this exactly at line 423.
- **Reshape equivalence rule (CLAUDE.md).** Building a 16×16 matrix then flattening to 256 then linearly projecting to 768 destroys the matrix structure before GPT-2 sees it. Pays 1.6M params for nothing.
- BPE vocab (50K tokens) breaks the bigram motivation that came from byte vocab (256).

### 3. Low-rank w_up factorization (`w_up = A @ B`)
**Fatal flaw:**
- **Math error.** Factoring w_up as A @ B with k-dim bottleneck constrains the **subspace** that all Zs lie in across the batch, **not** the per-sample rank of any individual Z. At k=1, Z_i = α_i · M + b for some fixed full-rank 16×16 matrix M from reshape of A's single column. The per-sample rank of Z is **up to 16**, not 1. The proposal does not bound what it claims to bound.

### 4. K-probe bilinear readout (`h_out = Σᵢ (uᵢ^T Z vᵢ) · wᵢ`)
**Fatal flaw (F1, mathematically airtight):**
- `u_i^T Z v_i = ⟨u_i v_i^T, Z⟩_F` is a **linear functional on Z**. The full readout `h_out = W_out · scores` is a linear map R^{d×d} → R^K → R^{768}. The Jacobian ∂h_out/∂Z = Σⱼ W_out[:,j] ⊗ (u_j v_j^T) is a **constant tensor independent of Z**. Linear-in-Z = rank-blind by construction.
- It is strictly a **low-rank-factored version** of the current `w_down(Z.flatten())`. Less expressive than what we already have, not more.

### 5. Quadratic readout via `tr(A_i Z Z^T)`
**Fatal flaws:**
- **F1.** Even though tr(A·ZZ^T) is quadratic in Z, the readout is **linear in M = ZZ^T**. M lives in the symmetric matrix space R^{d(d+1)/2}. The model can trivially place all useful information in M's top eigenvalue (= top singular vector of Z), reproducing the rank-1 collapse. The Jacobian-on-Z trick (∂tr(AZZ^T)/∂Z = (A+A^T)Z depending on Z) does not translate to rank-pressure in the loss landscape.
- **F2.** `ZZ^T` **discards V** (right singular structure of Z). For d=16, Z has 256 unique entries; ZZ^T has 136. Half the matrix structure is gone before any readout. SVD(Z) = UΣV^T but SVD(ZZ^T) = UΣ²U^T — V is invisible.

### 6. Per-sample U·V^T train-time rank constraint (Z = U(h)·V(h)^T at rank r)
**Fatal flaws:**
- **F1.** This is **Round 3 in disguise.** The gradient still flows through `w_down(Z.flatten())`, which is rank-blind by linear algebra. Training-time rank constraint changes the architecture but not the gradient signal. At r=1 vs r=16, the model will simply place its available information in whatever direction `w_down` reads. Both succeed, both flat.
- **F2.** Param-matched baseline as written is **mathematically incoherent** — Linear(768, 16*16) gives U,V in R^{16×16} so U·V^T has rank ≤ 16, NOT rank-1. Cannot force rank=1 with full params without an SVD projector that defeats the purpose.
- **F3.** All likely outcomes are **unfalsifiable** — flat r=1 to r=16 admits three interpretations (task doesn't need rank / arch can't use rank / capacity confound) that we cannot disentangle.

### 9. Matrix-valued attention (MVAttn — d×d Q/K/V matrices with Frobenius scores)
**Fatal flaws:**
- **F1: It's mathematically identical to standard multi-head attention with head_dim=d², under reshape.** `tr(Q_i K_j^T) = ⟨vec(Q_i), vec(K_j)⟩` — Frobenius inner product on d×d matrices IS the standard dot product on d²-dim flattened vectors. MVAttn with d=16 and H=4 is GPT-2 attention with head_dim=256, H=4 — a known, uninteresting variant. The "data-dependent Jacobian" claim is also true of vanilla attention's `q·k` (∂score/∂q = k); standard attention has the same property and is a famously rank-1-attractor mechanism. Bilinearity in (Q_i, K_j) is not new.
- **F2: The `w_out: H·d² → D` flatten kills any matrix structure on exit (Lesson 1).** Identical sin to killed MatrixBottleneck w_down. Output is rank-blind by linear algebra. Subsequent layers see vectors. The "matrix-ness" exists for one reshape and is destroyed before the residual stream sees it.
- **F3 (CLAUDE.md reshape equivalence rule, direct hit).** "Any d²-dim vector can be reshaped to d×d matrix and vice versa. Structure only matters if OPERATIONS preserve it." MVAttn applies softmax-weighted convex combination then flatten+linear. Convex combinations don't provide rank-discrimination pressure. Then Lesson 1.

### 8. End-to-end matrix LM head (per-vocab bilinear logits `logit(w) = Σᵣ uᵤᵣ^T Z vᵤᵣ`)
**Fatal flaws:**
- **F1 (Lesson 1 strikes again).** Each `logit(w) = ⟨M_w, Z⟩_F` is a Frobenius inner product, **linear in Z**. The full map `Z → logits ∈ R^|V|` is a linear map `R^{d×d} → R^|V|` representable as a single `W ∈ R^{|V|×d²}` whose row w is `vec(M_w)`. The Jacobian `∂logit(w)/∂Z = M_w` varies across w but is **constant in Z**. The gradient `∂L/∂Z = Σ_w (p_w − 1[w=target]) · M_w` is a data-dependent linear combination of constant matrices, but that's just standard CE through a linear head. The "per-vocab probes vary the output coordinate" intuition confuses output-coordinate variation with Z-dependence — only Z-dependence breaks rank-blindness. **Mathematically equivalent to killed K-probe (#4) under relabeling**: stack all `|V|·R` probes and per-vocab is K-probe with a fixed identity-with-grouping mask. Strict special case, less expressive.
- **F2.** ProsQA's actual answer vocabulary is ~500 tokens (entity names + class names + boilerplate). The "50k simultaneous constraints" intuition is wrong — only ~500 rows of `W` ever appear as labels. Effective constraint count is well within rank-1 reach of Z.
- **F3.** Z at the answer position is not "reasoning state." Per `student_forward` lines 499-510, the answer-position hidden state is the output of the joint transformer pass over prefix+latents+tail. All reasoning already happened in the latent positions. Reshaping a vector to a matrix via `w_up_matrix` doesn't give it meaningful singular structure — you'd be rank-probing the wrong Z.

### 7. Spectral polynomial readout (`[tr(Z), tr(Z²), tr(Z³), tr(Z⁴)]` → MLP)
**Fatal flaws:**
- **F1.** 4-scalar bottleneck is **catastrophically lossy and position-blind.** 6 latent positions × 4 scalars = 24 scalars per sample injected into the residual stream. Vanilla CODI injects 6 × 768 = 4608. Current matrix bottleneck injects 6 × 256. This is a **~200× capacity reduction.** GPT-2 124M cannot encode ProsQA reasoning state in 24 scalars. Expected: accuracy collapses.
- **F2.** Spectral features are **orthogonally invariant** — tr(Z^k) = tr(QZQ^{-1})^k for any similarity transform. They encode only eigenvalues, discarding 240 of 256 DoF (eigenvectors).
- **F3.** **Rank-1 fixed point preserved.** For Z = uv^T, tr(Z^k) = (v^Tu)^k. Feature vector is `(c, c², c³, c⁴)` for c = v^Tu — a 1-parameter family. Rank-1 Z can hit ANY feature vector the model wants by tuning c. There is zero pressure to increase rank.

## Generalized lessons from killed proposals (the deep structural finding)

**Lesson 1.** Any operation that is **linear in Z** (or that factors through a linear map on a fixed-size space derived from Z, like a flatten, a reshape, or the symmetric Gram ZZ^T) is **rank-blind by linear algebra**. Its Jacobian is a constant tensor, so the loss gradient cannot reward Z's singular structure. This kills all "K probes", "low-rank factorizations of the readout", "compressed projections", and "Frobenius readouts" in advance.

**Lesson 2.** The model **always finds the lowest-rank representation** that satisfies the loss. Adding a more rank-aware *readout* doesn't force rank usage — it just makes rank usage *possible*. The model still chooses the easiest path, which is rank-1.

**Lesson 3.** Train-time rank **constraints** at the construction stage (forcing Z to be rank-r by parametrization) don't fix this either, because the loss gradient still cannot distinguish rank-r from rank-1 within the constrained space — the model's gradient signal is the same.

**Lesson 4.** Rank-discrimination requires one of:
  (a) An **explicit loss term** that rewards or penalizes rank (e.g., nuclear norm regularizer, effective-rank reward). But this is unfalsifiable in the standard sense — "of course rank goes up if you reward it" — and has to be carefully designed to test something meaningful.
  (b) A **task whose ground truth provably requires K independent quantities** to solve, and the model literally cannot find a rank-1 shortcut. We have not yet found such a task at the right scale; ProsQA was supposed to be it but the model finds a rank-1 solution.
  (c) Asking a **different question entirely** — e.g., "what is encoded IN Z?" via interpretability/probing rather than "does rank track reasoning?" via ablation.
  (d) A **fundamentally different mechanism** where matrix structure participates in the forward pass in a way that no linear map can flatten — e.g., matrix-valued attention (matrices in Q/K/V), end-to-end matrix LM head (no flatten anywhere).

## Things explicitly RULED OUT empirically (no need to revisit)

- **L_kd (distillation loss) is NOT the structural bottleneck.** Round 3 Run 1 with gamma=0 still flat. Removing distillation actually *increased* Z rank (~10 → ~13) but did not increase rank functionality.
- **Multiplicative thinking iter is NOT critical.** Round 3 Run 2 with thinker disabled still flat. The thinker contributes ~1pp accuracy and zero rank functionality.
- **Task variation between arithmetic (GSM8K) and logic (ProsQA) does not change the curve shape.** Both flat.
- **The best singular value alone is sufficient** for everything the model does. Range across k=1 to k=16 is 0.4pp in every round. Rank is decoration.

## What HASN'T been tried yet (potential survivors)

- **Vanilla GPT-2 SFT control** (Round 4, currently running). Tests whether the entire matrix-CODI stack contributes anything beyond extra params.
- **Linear probe interpretability on Z** (different question, doesn't depend on readout).
- **Larger d sweep** (d ∈ {32, 64}). Capacity ablation — could rule in/out "d=16 was capacity-limiting."
- **COCONUT training (no CODI distillation, curriculum-based latents).** Different optimization regime entirely.
- **Matrix-valued attention** (matrices in Q/K/V projections, not just bottleneck). Major architectural change — matrix structure participates in attention itself, not just feedback.
- **End-to-end matrix LM head** (matrix-valued projection from Z to vocab logits, never flatten). Tests whether a fully matrix-native readout to the OUTPUT (not just the feedback loop) changes anything.
- **Synthetic task with verifiable multi-rank requirement** that GPT-2 can train on at this scale. Hard to design without falling into MNNS's traps.
- **Explicit nuclear-norm reward** in the loss combined with accuracy measurement at varying λ — unfalsifiable on its face but could be designed to test something meaningful.
- **Bigger backbone** (GPT-2 medium/large or another small LM) — does the rank-blindness persist at scale?

Any new proposal must (1) escape Lessons 1–3 above, (2) not duplicate any of the killed proposals or reproduce the same fatal flaw under a new name, and (3) state explicitly which of the "potential survivors" categories it falls under.

## Lesson 5 — Readout nonlinearity ALONE does not break rank-blindness

**Added 2026-04-17 after Round PC bilinear+GELU experiment.**

The simple Jacobian argument (Lesson 1) predicts that breaking linearity of
the readout in Z will make the gradient see rank. We tested this with a
readout that is explicitly nonlinear in Z:

```python
# bilinear+GELU readout
probes = einsum('ki,blik->blk', U, einsum('blij,kj->blik', Z, V))  # K bilinear probes
h_out = Linear(GELU(probes))
```

K = d² = 256 probes, full expressiveness. Unambiguously nonlinear in Z via the
GELU. The Jacobian is NOT constant.

**Result:** flat rank-k curve anyway. 78.91% at k=1, 79.69% at k=2 through k=16.
Spearman r = -0.13, p = 0.14 (not significant).

**Why it doesn't work:** the model learns solutions where the bilinear probes
that are active at the output lie along a rank-1 subspace of Z, so the
effective information flow is rank-1 even when the architecture could support
rank-256. The GELU doesn't force the model to USE higher rank; it only makes
higher-rank paths available. The optimizer still prefers the lowest-rank
solution that satisfies the loss.

**Implication:** the failure is NOT in readout linearity per se. It is in the
**CODI distillation objective** producing gradients that do not reward any
particular rank of Z. Lesson 1 was a sufficient condition for rank-blindness
but not a necessary one.

**Positive controls to run before concluding matrix-thinking is dead:**
- `svd_aug` readout (singular values explicitly fed through MLP) — queued
- `quadratic` readout (ZZ^T and Z^TZ second moments) — banked 79.69%, rank-k
  eval pending
- `bilinear` without GELU (reparametrization control) — queued

If ALL four positive controls produce flat curves, the failure is in the
objective, not any readout variant. If any one produces a bending curve, we
learn WHICH property of the readout breaks the rank-blindness.

## Lesson 6 — Rank is decoupled from accuracy across seeds

**Added 2026-04-17 after 3-seed flatten replication.**

Three training runs of the SAME config (Round 3 gamma=0 flatten, ProsQA,
gpt2-small, 25 epochs, batch 16) with seeds 1337, 42, 7:

| Seed | Best acc | Final Z_rank |
|------|----------|--------------|
| 1337 | 80.47%   | ~13          |
| 42   | 81.25%   | ~4           |
| 7    | 82.81%   | ~12          |

Accuracy: **81.51 ± 1.2pp** (tight).
Rank: **4–13** (3× spread).

The loss does not push toward any particular rank. Whatever rank Z converges
to is essentially arbitrary, determined by initialization. This is direct
evidence that the loss is rank-agnostic — stronger than the flat rank-k
ablation curve, because it shows rank varies at the SAME loss value.

Implication for Chapter 2: if end-to-end matrix-native training on a rank-K-
aware task produces a similar seed-spread in rank with tight accuracy, the
matrix-thinking direction is fully dead. If it shows rank concentrated at K*
across seeds, matrix-thinking is alive.
