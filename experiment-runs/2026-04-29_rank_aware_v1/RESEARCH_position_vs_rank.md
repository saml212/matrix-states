# Research: Position-Decomposition vs Spectral-Rank-Decomposition in Latent CoT

*Agent report — 2026-04-29*

---

## 1. Position-vs-Rank in Continuous CoT Literature

**COCONUT (Hao et al., arXiv:2412.06769)**
Ablation over k ∈ {0,1,2,3,4,5,6} shows monotonically improving performance as k grows, and a k=2 step solves ProntoQA problems that k=1 cannot. This is pure *position-count* evidence: each additional latent position adds capacity. The paper does not characterize what each position encodes nor does it analyze spectral/rank structure within any individual latent vector. No single-position baseline is isolated as a probe of within-position capacity. In practice, COCONUT uses GPT-2 hidden states (768-dim), so within-position rank is implicitly full; "rank" is not a design variable.

**CODI (Shen et al., arXiv:2502.21074)**
Figure 5 shows accuracy peaks at 6 continuous thought tokens and degrades with either fewer or more. Interpretability analysis (Figure 6) demonstrates that different latent tokens (z₁…z₆) attend to different operands — direct evidence of *position-specific functional specialization*. No spectral or rank analysis of individual z_i. Ablations vary latent count but do not probe within-position structure.

**Dynamics within Latent CoT (arXiv:2602.08783)**
Step-wise do-interventions on COCONUT and CODI show "latent-step budgets behave less like homogeneous extra depth and more like **staged functionality with non-local routing**." Different positions carry causally distinct roles. This is the strongest existing evidence for positional decomposition in the continuous CoT literature — but it comes from a causal/intervention angle, not from forcing a rank constraint. No rank or spectral analysis.

**Reasoning by Superposition (Zhu et al., arXiv:2505.12514)**
Parallelism lives **within each position vector**, not across positions. The model encodes multiple BFS frontiers as `[tc] = 1/√|V_c| Σ_{v∈V_c} u_v` — a normalized superposition within a single vector. This is the theoretical opposite of our finding: the paper proves that continuous thoughts gain power through within-position superposition. Our empirical result (rank-1-per-position suffices) contradicts the within-position-superposition model on ProsQA-MULTI-2. The tension is real and worth calling out.

**Latent Reasoning as Vocabulary-Space Superposition (arXiv:2510.15522)**
Defines latent tokens as linear combinations in embedding space; finds effective parallelism Neff ≈ 1.7–3.0 across competing reasoning paths, interpreted as within-position superposition. Also reports that the token embedding matrix has low effective rank (rapid singular value decay), meaning latent tokens naturally live in a low-dimensional subspace — consistent with our rank-1 finding. Does not test whether within-position rank correlates with task accuracy.

---

## 2. Depth-vs-Width and Related Tradeoffs

**Dong et al., ICML 2021 (arXiv:2103.03404)**
Pure self-attention loses rank doubly-exponentially with depth: token representations converge to rank-1 (identical rows) without skip connections. Skip connections and MLPs counteract collapse. This is the architectural depth-vs-rank story — deeper pure-attention converges to lower rank per position. Relevant inverse analogy: our model uses 6 positions × rank-1, which is structurally similar to what depth-collapse produces.

**Yehudai et al., 2025 (arXiv:2503.01805)**
Depth-width tradeoff on graph tasks: linear width at constant depth solves problems that require logarithmic depth at sub-linear width. The analogy to our setting: n_latents (positions) is the "depth" axis; d² per position (matrix dimension) is the "width" axis. Our finding — rank-1 per position suffices — suggests the model uses more positions (depth) rather than more capacity per position (width), consistent with the graph-task tradeoff result.

---

## 3. Mechanistic Interpretability Angle

No published work directly asks "do SAE features attach to specific latent-positions in a continuous CoT model?" The closest is:

- Anthropic's superposition work: features occupy directions in activation space, not specific sequence positions. This is the standard model of how representational capacity is used in residual streams.
- The 2602.08783 do-intervention work is the closest analogue: it treats each latent step as a distinct causal node, finding staged functionality. This supports position-as-compositional-axis, but via causal intervention rather than spectral analysis.

The MI literature does not have a clean "position vs rank" framing for latent CoT. **This framing appears novel.**

---

## 4. n_latents=1 Prior Art

No published paper explicitly runs a continuous CoT model with n_latents=1 as an isolated probe of *within-position capacity*. COCONUT's k=0 baseline is a no-thinking ablation (zero latent tokens), not a single-position probe. k=1 shows some gain over k=0, but is presented as "one step of planning," not as a test of whether rank > 1 is needed within that step. No paper varies n_latents with the explicit intent of asking "if position-decomp is removed, does the model use rank?"

**This experiment (our planned n_latents=1 control) has no published prior art.**

---

## 5. Novelty Framing

**Claim our finding can support:**

> Prior continuous CoT work characterizes the *number* of latent positions as the key capacity axis (COCONUT, CODI, Dynamics-within-LCoT), and theoretical work posits within-position superposition as the mechanism (Reasoning by Superposition, Vocab-Superposition). We provide the first empirical test that isolates these two axes: forcing d=16 matrix-CODI to rank-1 per position does not degrade accuracy on a multi-target reasoning task. The model uses **position-decomposition** (6 positions × rank-1) as its compositional strategy, not within-position spectral superposition. This is consistent with the depth-vs-width graph-task tradeoff (Yehudai 2025) and the rank-collapse-by-depth result (Dong 2021), but contradicts the within-position superposition prediction of Zhu 2025. The n_latents=1 ablation (no prior art) will determine whether within-position rank capacity can substitute for position capacity.

**Key tension to cite:** Zhu 2025 predicts within-position superposition; our data shows it is not used. Both can be true if the optimizer finds a simpler position-decomposed solution that satisfies the training objective without requiring the expressiveness that within-position superposition provides.
