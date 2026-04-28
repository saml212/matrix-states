# Combined Experiment Plan — Validator + Ideator

After 7-agent gauntlet (synthesizer, methodological-skeptic, empirical-skeptic, reviewer-2, pragmatist, validator, ideator). The validator produced a focused 7-cell queue around the proposed 3×3. The ideator proposed 6 NEW directions outside the 3×3, with #1 (Jacobian erank) being the highest-information-per-dollar action available.

The plan below merges them, prioritized by gating dependencies and leverage.

## Surviving direction summary

| # | Name | Source | Cost ($) | Days | Gate |
|---|---|---|---|---|---|
| 0 | Jacobian erank measurement on 5 existing readouts | Ideator #1 | 3 | 1 | None — runs first |
| 1 | ProsQA-MULTI-2 pre-validation (Gram + question-only MLP + rank-1-constrained matrix-CODI) | Validator #1 | 14 | 1–2 | None |
| 2 | ProsQA-MULTI-2 + std-CE (objective rank-blindness on rank-k task) | Validator #2 | 9 | 2–3 | After #1 confirms |
| 3 | ProsQA-MULTI-2 + entropy reward (HEADLINE) | Validator #3 | 18 | 3–4 | After #2 baseline |
| 4 | ProsQA-1 + entropy reward (null control) | Validator #4 | 9 | 3–4 | Parallel with #3 |
| 5 | Per-position rank-1 forcing (position-decomp falsifier) | Validator #5 | 9 | 4–5 | After #1 |
| 6 | Learned-projector rank-k retention (Li & Janson optimal ablation) | Ideator #5 | 5 | 5 | After matrix-CODI checkpoints exist |
| 7 | wd=0 implicit-bias sweep (appendix) | Validator #6 | 5 | parallel | None |
| 8 | SAE on matrix-CODI Z (Wang 2025 substrate test) | Ideator #2 | 20 | 6 | After #5 if SAE is novel-paper-grade |
| 9 | Rank-k on Rizvi-Martel COCONUT 96.6% checkpoint | Ideator #6 | 5 | 6 | None |
| 10 | ProsQA-MULTI-4 (contingent) | Validator #7 | 18 | 7+ | Only if #3 returns CLEAN POSITIVE |

**Core (#0–#7):** $72 spot, ~46 H100h
**With novelty extensions (#8, #9):** $97 spot, ~63 H100h
**With contingent #10:** $115 spot, ~75 H100h

## Critical path

- **Day 1:** #0 Jacobian erank (single afternoon, ~$3). If erank(J) ≈ 1 → paper's §5.3 mechanism is *measured* and submission gains a complete mechanism story. If erank(J) ≥ 4 → §5.3 hypothesis falsified, paper needs revision before submission.
- **Day 1–2:** ProsQA-MULTI-2 dataset generation (CPU, free). #1 pre-validation in parallel.
- **Day 3–4:** #2 (MULTI-2 std-CE) and #3 (MULTI-2 + entropy) → headline result.
- **Day 4:** #4 (ProsQA-1 + entropy) null control.
- **Day 5:** #5 (per-position rank-1), #6 (learned-projector), #7 (wd=0).
- **Day 6 (optional novelty extension):** #8 (SAE), #9 (Rizvi-Martel). Only if Days 1–5 results are crisp enough to motivate the extension.
- **Day 7–8:** Writeup pulled together.
- **Day 9:** Final review pass.
- **Day 10 (May 8):** Submit.

## Attacks that landed (forced revisions)

1. **Bilinear-probe argument is rhetorical** (meth-skeptic). The actual readout is `W_down · vec(Z)` followed by GPT-2 transformer + LM head — not orthogonal bilinear probes per target. Reframe rank-k argument as bottleneck information capacity, not probe geometry.
2. **Position-decomposition shortcut** (meth-skeptic, critical). matrix-CODI has 6 latent positions; the model could encode each of k targets at a separate position with rank-1 Z each, satisfying the multi-target task without any within-position rank > 1. **Mitigation:** experiment #5 (per-position rank-1 forcing) is the falsifier.
3. **Embedding tying** (meth-skeptic). GPT-2's `wte` is not orthogonal across target tokens. Pre-experiment Gram-matrix check on MULTI target embeddings is mandatory before claiming rank-k by orthogonality.
4. **W_down memorization** (emp-skeptic). `W_down` could memorize question→{a₁..aₖ} mapping at rank-1 if the question text identifies property P. Mitigation: vanilla-SFT and question-only-MLP baselines in #1.
5. **Reviewer-2 tautology objection** ("constructed task requires rank k, loss maximizes rank k, observed rank k"). **Addressed by 3-condition factorial:** the std-CE arm (#2) shows the task alone doesn't fix rank-blindness, separating task from training signal.
6. **Anchored singular directions** are KILLED for this submission — SVD backward at near-degenerate σ is unstable; would require Moore-Penrose pseudoinverse infrastructure.
7. **Nuclear-norm column** is CUT — predicted to grow σ₁ only (not spread), produces a null result in 3 cells = $27 wasted.

## Reviewer-2 residual gap

Reviewer-2 wanted: **causal-patch test**. Zero out σₖ uₖ vₖᵀ from a problem where target_k = A, feed into a problem where target_k = B, observe whether prediction flips specifically for target k. This is the test that "singular directions ARE specific reasoning paths" rather than just output channels.

**Recommendation:** add as a free CPU/inference-only diagnostic paired with #3. Cost: zero GPU, ~1 hour scripting.

## Headline framing if results land

> "Adding singular-value entropy reward $+\lambda H(\tilde\sigma)$ to matrix-CODI training causes the rank-$k$ ablation curve to bend on **ProsQA-MULTI-2** (a constructed task that requires multi-target reasoning) while remaining flat on standard ProsQA-1 (rank-1-solvable) and on MULTI-2 with standard CE (objective still rank-blind on rank-k task). The 3-condition factorial separates task structure from training signal: rank becomes functional only when both conditions are met. Position-decomposition (each of 6 latent positions encodes one target at rank-1) is ruled out by the rank-1 forcing experiment. Jacobian effective-rank measurement on the 4 nonlinear-in-Z readouts confirms the paper's §5.3 candidate refined mechanism."

## Open question (defer to camera-ready)

Causal-patch results, MULTI-4 (contingent), broader scale beyond GPT-2 small.
