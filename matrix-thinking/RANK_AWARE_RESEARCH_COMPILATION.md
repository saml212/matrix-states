# Rank-Aware Training: Research Compilation (8 agent reports)

Compiled for adversarial team review. Goal: pick the experiments that would make
the most novel, defensible follow-up paper to "The Gradient Does Not See Rank"
(ICML 2026 MI Workshop, deadline May 8 2026).

## Paper status (committed in `b715e36`)

The paper now contains:
- §3-4: 4 flat rank-k curves on matrix-CODI (different γ, thinker, scale)
- §5: 4 nonlinear-in-Z readouts (bilinear, bilinear+GELU, SVD-augmented, quadratic) — all flat
- §5.5 (just added): negative control on vanilla GPT-2 SFT — flat too; random-h floor at same accuracy
- §7: alternative explanation flagged: "ProsQA may be rank-1-solvable; we don't have a task that provably requires rank>1"

The paper's open question (§7) is the key gap: **what experiment would
disambiguate "loss is rank-blind" from "task doesn't need rank"?**

## Agent reports (ordered by paper-impact)

### A. ProsQA-MULTI: an intrinsically rank-k task — STRONG GO

**The killer finding.** Construct ProsQA-MULTI-k by modifying the Coconut DAG
generator: questions with k disjoint positive answers requiring k independent
predictions at the answer position. Argument: bilinear head $u_i^\top Z v_i$
for each of k targets needs k orthogonal directions in Z; rank-1 truncation
provably fails at least one target.

- Cost: ~1 day. ~150 lines for DAG mod + multi-label eval.
- Reuses all existing matrix-CODI / rank_eval infrastructure.
- Directly closes §7's stated gap.
- **No prior work** has constructed a provably rank-k task for matrix-valued CoT.
- Companion task: 1D cellular automaton with state-size N > d, also makes a
  formal lower-bound argument (board-size threshold from arXiv:cCIdxLoLJ5).
- 2D regression: SKIP — non-orthogonal compression escapes the bound.

**Headline if it works:** "Rank-k ablation curve breaks at k=1 on ProsQA-MULTI-2,
stays flat on ProsQA-1. The rank-blindness on standard ProsQA is at least
partly attributable to the task being rank-1-solvable, not solely to the
objective."

### B. Singular-value entropy reward $+\lambda H(\tilde\sigma)$ — GO

Maximization direction: $H(\tilde\sigma) = -\sum_i \tilde\sigma_i \log \tilde\sigma_i$
where $\tilde\sigma_i = \sigma_i / \sum \sigma_j$. Closes the loop on the paper's
own diagnostic (effective rank = $\exp(H)$).

- Bounded $[0, \log d]$, scale-invariant, $\lambda$ stable across runs.
- Backward through `torch.linalg.svdvals` is documented as unconditionally stable.
- **Novelty:** no prior work uses SV entropy as a *differentiable training loss*
  on per-sample latent matrix thoughts. FlexLoRA (arXiv:2601.22905) uses it as
  non-differentiable rank allocation. Matrix-SSL uses entropy on batch
  covariance, not per-sample.
- One-afternoon experiment: same arch, add `+lambda * H(svdvals(Z))` to loss,
  $\lambda \in \{0.01, 0.1, 1.0\}$, rerun rank-k.

**Headline if it works (paired with ProsQA-MULTI):** 
"Training on effective-rank reward causes matrix-CODI to use rank functionally
*on tasks that need it*: rank-k ablation bends on ProsQA-MULTI when trained
with entropy reward, stays flat on ProsQA-1 (no need for rank), and stays
flat on ProsQA-MULTI without entropy reward (objective doesn't see rank)."

This is a **3-condition factorial** that tells the complete story.

### C. Nuclear-norm reward $+\lambda \|Z\|_*$ — GO with caveats

- Adjacent prior art (Scarvelis NeurIPS 2024 — Jacobian nuclear norm; Kobayashi
  2024 — weight decay ↔ nuclear norm equivalence on weight products). No prior
  work applies it to per-sample activation matrices in a CoT bottleneck.
- **Critical caveat:** $\|Z\|_*$ maximization grows the largest singular value,
  not necessarily spreading energy. This is NOT a rank-maximization reward in
  the strict sense. To get rank spread, must use entropy (B) or pair with
  orthogonality penalty.
- Probably *less* paper-worthy than (B); $\|Z\|_*$ is the wrong functional.
- Could be useful as a **comparison baseline** in (B)'s sweep: shows entropy
  works where nuclear norm doesn't, sharpens the mechanism claim.

### D. Anchored singular directions (spectral SIM-CoT) — CONDITIONAL GO

Force singular direction $k$ of Z at latent position $p$ to encode reasoning
step $k$ via an MSE auxiliary loss on $\sigma_k u_k v_k^\top$. SIM-CoT does
this per-position; the spectral variant is orthogonal/novel.

- Numerical: Moore-Penrose pseudoinverse SVD (arXiv:2411.14141, Nov 2024) is
  the principled stable backward; or detach $U,V$ and only train $\sigma$.
- Novelty: no spectral-axis anchoring exists in literature.
- Risk: ProsQA may be rank-1-solvable (need ProsQA-MULTI to test this properly).
- Harder to implement than (B). Saves for camera-ready or follow-up paper.

### E. Implicit bias channel diagnosis — GO with sequencing

Which optimizer ingredient sets the rank-12-13 attractor and the seed-spread {4,12,13}?

- Sequence: wd=0 vs wd=0.01 first (6 runs × 3 seeds = ~$18). If informative,
  add Adam vs SGD and init-zero arms.
- Kobayashi 2024 (already cited) gives the wd ↔ nuclear-norm equivalence.
- Razin 2020 / Arora 2019 give init-sensitivity predictions for matrix factor.
- Adam-vs-SGD: NeurIPS 2025 "Rich and the Simple" shows Adam more resistant
  to simplicity bias — predicts SGD would tighten rank to ~4.
- **Standalone or appendix-grade**, not headline. But could be a clean
  workshop note if wd=0 dramatically changes the rank distribution.

### F. Multi-head spectral readout — NO-GO alone, conditional GO with orthogonality

Symmetry collapse: even with per-head gradient routing, the optimizer can
collapse all heads to the dominant singular direction unless an orthogonality
penalty separates them. Adding such a penalty would confirm the paper's
"fix at objective" thesis (the orthogonality term IS an objective change).
Not standalone-worthy.

### G. GRPO rank-stratified reward — NO-GO this submission

- GRPO policy ratio is undefined for matrix latent steps (no token probability).
- arXiv:2512.11816 (Dec 2025): RL through continuous latents fails — all RL
  variants below SFT baseline on GSM8K.
- 14-day runway too tight to debug latent-step probability workarounds.
- Make this a future-work paragraph in §7 of the paper.

### H. Symmetric extension (no W_down at eval) — NO-GO

Already subsumed by §5.5's random-h sensitivity floor result. Diagnostic
duplicate, not novel.

## Recommended consolidated direction (for team to argue)

**The "Three-by-Three" experiment** combining A + B + C as comparison:

| | std loss | $+\lambda H$ entropy | $+\lambda \|Z\|_*$ nuclear |
|---|---|---|---|
| ProsQA-1 (rank-1-solvable) | flat (paper Round 3) | *predict: still flat* | *predict: still flat* |
| ProsQA-MULTI-2 (rank ≥ 2 by construction) | *predict: still flat or floor-effect* | *predict: bends — paper headline* | *predict: bends only on σ_1 not k=2* |
| ProsQA-MULTI-4 (rank ≥ 4) | *predict: floor effect* | *predict: bends with sharper inflection* | *predict: weak bend* |

Three predictions, each falsifiable, each tells a different part of the story.

**Total compute estimate:** 9 cells × 3 seeds × ~2 H100h = ~54 H100h ≈ $80 spot.
Plus ProsQA-MULTI dataset generation (CPU, free).

**Plus orthogonal experiments (parallel):**
- Implicit bias wd=0 sweep: 3 seeds × 1 H100h ≈ $5
- ProsQA-1 entropy-reward control: 3 seeds × 1 H100h ≈ $5

**Total: ~$90 spot for a paper-grade follow-up.**

## Constraints from user

- ICML MI Workshop deadline: May 8, 2026 (~10 days)
- Spot pricing only
- GPU at 100% or pod down (no idle)
- CPU data work capped at nproc/2
- Subagent cascade for every experiment: idea → research → attack → defend → script → attack → audit → run
- Pivot to roadmap/queue if direction exhausted

## What we want from the team

1. **Argue:** is ProsQA-MULTI's "rank-k by orthogonality" argument actually
   rigorous, or is it a hand-wave? Could the model still find a rank-1
   solution by clever embedding tying?
2. **Refine:** what ablation pairs in the 3x3 matrix are essential vs
   skippable for a rigorous paper? Can we cut compute?
3. **Attack:** what's the strongest reviewer objection that would kill the
   paper even if all experiments work?
4. **Pivot if dead:** if the team finds a fatal flaw in the proposed 3x3,
   recommend the next-best direction from the agent reports.
5. **Output:** a ranked list of experiments to queue, each with hypothesis,
   decision criterion, falsifiable prediction, and compute budget.
