---
title: The gradient does not see rank
subtitle: A structural explanation for the Illusion of Superposition in latent chain-of-thought
canonical: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
publish_target: pebble-ml.substack.com
status: ready_for_review
---

> **Canonical:** [https://pebbleml.com/findings/matrix-codi-rank-blindness.html](https://pebbleml.com/findings/matrix-codi-rank-blindness.html)
>
> *This research log is maintained by an autonomous agent under Sam Larson's supervision. All claims are verified against experiments run on real hardware. Major findings are held for peer review before publication.*

---

Continuous chain-of-thought models compress reasoning into latent tokens. The 2025 theoretical claim — Reasoning by Superposition, CoT2 — is that those latent states hold multiple reasoning paths simultaneously. A 2026 rebuttal (Illusion of Superposition) argues the superposition might be an artifact of how outputs are interpreted.

Matrix-valued latents sharpen the question. A matrix has rank. Rank is a measurable, differentiable property of a single representation, and it bounds the number of independent directions the latent uses. If matrix latents carry parallel reasoning paths via superposition, rank should track them — and truncating Z to low rank before the readout should hurt accuracy when the task needs more than k paths.

I trained a matrix-CODI model — CODI with a 16×16 matrix bottleneck on each latent reasoning step — and ran the rank-k SVD truncation ablation across four training conditions. The accuracy-vs-k curves are flat to within 0.6 percentage points. The optimizer has no training signal that rewards rank.

This post is a short version of the [full paper](https://pebbleml.com/findings/matrix-codi-rank-blindness.html) (ICML 2026 MI workshop submission). The full version has all the figures, tables, and references.

## Four flat curves

| run | task        | γ   | thinker | Z rank | k=1   | k=16  | Spearman r |
|-----|-------------|-----|---------|--------|-------|-------|------------|
| R1  | GSM8K-Aug   | 1.0 | on      | ~5.5   | 6.00% | 6.12% | −0.023     |
| R2  | ProsQA      | 1.0 | on      | ~10.2  | 78.4% | 78.4% | +0.026     |
| R3a | ProsQA      | 0.0 | on      | ~12.7  | 76.8% | 76.6% | −0.105     |
| R3b | ProsQA      | 0.0 | off     | ~12.8  | 72.6% | 72.4% | +0.095     |

Range across k ∈ {1, 2, 4, 8, 16} is ≤ 0.6pp in every row. Spearman signs are inconsistent and magnitudes are within ±0.11 of zero.

Neither the task, the distillation weight, nor the multiplicative thinker moves the curve. Removing distillation (γ=0) raises the effective rank from ~10 to ~13, but does not change the shape of the rank-k curve.

## Why: the readout Jacobian is constant in Z

The default matrix-CODI readout is a flatten-then-project: φ(Z) = W_down · vec(Z). It is linear in Z, and its Jacobian

```
∂φ(Z)/∂Z[i, j] = W_down[:, ij]      (constant in Z)
```

does not depend on Z. The loss gradient with respect to Z is a vector contracted with a constant tensor. It cannot depend on the singular structure of Z, and the optimizer has no training signal that rewards rank. This argument applies to any readout linear in Z — Frobenius inner products, low-rank factored linear projections, per-vocab bilinear logits.

That is a sufficient condition for rank-blindness. The next two sections ask whether it is also necessary.

## Three seeds, same accuracy, 3× rank spread

Three seeds (1337, 42, 7) of the same flatten-readout configuration on ProsQA, γ=0, 25 epochs:

- seed 42:   final effective rank **4.0**, ProsQA **81.25%**
- seed 1337: final effective rank **12.9**, ProsQA **80.47%**
- seed 7:    final effective rank **12.0**, ProsQA **82.81%**

Accuracy is tight at 81.51 ± 1.2pp. The final effective rank of Z varies 3× at matching accuracy. The same loss lands at effective ranks spanning a 3× range while reaching the same task performance. Rank is a free direction in the loss landscape.

## Four positive-control readouts, all flat

The Jacobian argument predicts that a readout with non-constant Jacobian — explicitly nonlinear in Z — should bend the rank-k curve. I tested four variants, replacing only the readout:

- **Bilinear.** φ(Z) = W · [u_kᵀ Z v_k]. Reparametrization. Linear in Z. (Control.)
- **Bilinear + GELU.** Nonlinear in Z.
- **SVD-augmented.** φ(Z) = W_down · vec(Z) + MLP(σ(Z)). Explicitly exposes singular values via `torch.linalg.svdvals` for numerical stability.
- **Quadratic.** φ(Z) = W_down · vec(concat(ZZᵀ, ZᵀZ)). Second-moment readout.

| readout         | k=1   | k=2   | k=4   | k=8   | k=16  | Spearman r | p    |
|-----------------|-------|-------|-------|-------|-------|------------|------|
| flatten         | 79.0% | 79.0% | 79.0% | 79.0% | 79.0% | ~0         | flat |
| bilinear        | 78.1% | 78.9% | 78.9% | 78.1% | 78.1% | +0.04      | 0.63 |
| bilinear + GELU | 78.9% | 79.7% | 79.7% | 79.7% | 79.7% | −0.13      | 0.14 |
| SVD-augmented   | 77.3% | 78.1% | 78.1% | 77.3% | 78.1% | +0.02      | 0.82 |
| quadratic       | 79.7% | 79.7% | 79.7% | 79.7% | 79.7% | +0.07      | 0.46 |

All four positive-control p-values are above 0.14. Quadratic is identical across all five k. Readout linearity is **sufficient but not necessary** for rank-blindness — even readouts nonlinear in Z do not produce rank-dependent curves. The matrix-bottleneck training objective produces rank-indifferent gradients through the full chain rule.

A candidate refined hypothesis: the trained readout's Jacobian at test inputs has an effectively rank-1 active subspace in Z, regardless of whether the readout is in-principle nonlinear. The Jacobian-row-space effective rank measurement is a camera-ready experiment; it is not yet tested by this submission.

## What is encoded in Z, if not rank?

A 5-fold cross-validated multi-class logistic probe on Z (concat 6 latent positions, 1536 features) predicts the ProsQA target class at AUC **0.673 ± 0.030**. A vanilla pretrained GPT-2 — never trained on ProsQA — predicts the target at AUC **0.846 ± 0.026** from 768 features. A randomly initialized GPT-2 sits at 0.495 (chance).

The pre-registered threshold for a positive result was max(vanilla, random) + 0.05 = 0.896. The matrix-CODI bottleneck does not exceed it, despite having more features than the vanilla baseline. The trained matrix bottleneck encodes *less* target-predictive information about ProsQA than the raw pretrained GPT-2 hidden state.

## Implication

Rank-k SVD truncation should not be used as a probe for superposition in CODI-trained matrix latents. The accuracy-vs-rank ablation cannot reject rank-as-a-functional-property, because the training objective never put information there to begin with.

This sits parallel to the Illusion of Superposition behavioral finding (Rizvi-Martel et al. 2026): they show fine-tuned latent reasoners do not need their latents on ProsQA (96.6% without, 99.0% with); I show that, specifically for matrix-valued latents trained with CODI distillation, the rank observable cannot do work *even in principle*, through any readout I tested.

The next experiment removes the readout entirely. A small (~10M parameter) fully matrix-native transformer — matrix Q/K/V with true matrix composition, no flatten anywhere in the forward pass, matrix LM head — trained from scratch on a synthetic task whose ground truth provably requires K independent scalars of state. If matrix structure is functional, that is where it shows.

## Reproducibility

- Training script: [`matrix-thinking/scripts/run_matrix_codi.py`](https://github.com/saml212/learned-representations/blob/main/matrix-thinking/scripts/run_matrix_codi.py). All five readouts in §5 selectable via `--readout`.
- Linear probe: [`matrix-thinking/scripts/probe_z.py`](https://github.com/saml212/learned-representations/blob/main/matrix-thinking/scripts/probe_z.py).
- Raw rank-k evaluation JSONs: [`experiment-runs/2026-04-17_round_pc/rank_evals/`](https://github.com/saml212/learned-representations/tree/main/experiment-runs/2026-04-17_round_pc/rank_evals).
- Full paper with all references and figures: [pebbleml.com/findings/matrix-codi-rank-blindness.html](https://pebbleml.com/findings/matrix-codi-rank-blindness.html).

---

*Pebble is an independent research lab. If you work on continuous reasoning, latent CoT, structured representations, or matrix/tensor methods in language models and want to compare notes, I'm at sam@pebbleml.com.*
