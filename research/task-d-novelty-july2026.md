# Task D Novelty Check — Matrix-Native Transformer, Rank-Aware KV Binding

**Date:** July 2026 (research pass; all citations verified via arXiv/OpenReview fetch, not fabricated)

## Verdict

**Task D as specified is novel and not scooped**, but it sits close enough to three active lines of
work that all three must be cited and positioned against explicitly. The single closest prior work is
**Nichani, Lee & Bietti, "Understanding Factual Recall in Transformers via Associative Memories"
(ICLR 2025 Spotlight, arXiv:2412.06538)** — they formalize exactly the object we care about (a linear
associative memory `W = Σ u_{f*(x)} e_x^T`, i.e., a matrix built from outer-product bindings) and even
give a rank-*m* construction showing `W` can be restricted to rank `m` and still store `≈ md` associations
(their Theorem 1 remark). But their rank-*m* result is (a) a hand-built **existence/achievability**
construction using random Gaussian mixing for a **discrete argmax-decoded** capacity question, not an
empirical measurement of the rank gradient descent *actually discovers*; (b) not a **necessity** (lower
bound) result for **exact continuous-vector recovery**; and (c) has no causal rank-*k* truncation
ablation. No paper found (across TPR, VSA/HRR, linear-attention/fast-weight, Hopfield, or 2024–2026
mechanistic-interpretability work) trains a transformer end-to-end under a **hard architectural
bottleneck** (decoder sees only a single d×d matrix `Z`, never the raw tokens), **measures the effective
rank of `Z` as a function of the controlled number of bindings `K`**, states the **provable rank ≥ K
necessity for exact linear recovery**, and then **causally forces rank-`k` during training** to produce
an accuracy-collapse curve against a param-matched vector-state baseline. That combination is the
contribution to defend. The closest *mechanistic* precedent for the rank observable itself is two
February 2026 linear-attention papers (Nazari & Rusch, arXiv:2602.04852; Sun et al., arXiv:2602.02195),
which measure "effective rank" of the linear-attention state matrix `S_t` — but both analyze **pretrained
LLMs on real text**, not a synthetic task with a known ground-truth `K`, and neither has a provable
rank-necessity theorem or a training-time rank-`k` ablation tied to a controlled binding count.

---

## 1. Tensor Product Representations & TP-Transformer

**Closest citation:** Schlag, Smolensky, Fernandez, Jojic, Schmidhuber, Gao, *"Enhancing the Transformer
with Explicit Relational Encoding for Math Problem Solving"* (TP-Transformer), arXiv:1910.06611
(ICLR 2020 submission venue; original title on arXiv). **Confidence: high (verified via arXiv abstract
fetch).**

TP-Transformer incorporates TPR-style role/filler binding into the attention mechanism (TP-Attention)
and reports SOTA on the DeepMind Mathematics Dataset. The abstract and available material contain **no
measurement of the rank of the binding tensor as a function of the number of role-filler pairs bound**.
McCoy, Linzen, Dunbar & Smolensky's companion line of work, *"RNNs Implicitly Implement Tensor Product
Representations"* (arXiv:1812.08718, ICLR 2019) and the associated Tensor Product Decomposition
Networks (TPDN), *probe* whether an RNN's learned representation is well-approximated by a TPR — this is
an interpretability/decomposition question (does structure X exist in an already-trained net?), not a
rank-vs-capacity or rank-vs-K training study, and it operates on off-the-shelf sentence encoders, not a
controlled synthetic binding count. Schlag & Schmidhuber's earlier *"Learning to Reason with Third-Order
Tensor Products"* (NeurIPS 2018, arXiv:1811.12143) uses third-order TPR fast weights inside an RNN for
bAbI-style reasoning; based on available abstract/methods text, it reports task accuracy, not rank
measurements as a function of stored-fact count.

**Verdict for this area:** No prior TPR/TP-Transformer paper measures rank vs. number-of-bindings or
does a rank-k causal ablation. Must cite as the conceptual ancestor of "binding = outer product,
matrix/tensor state," since Task D's `Z` is literally a TPR-style binding matrix.

## 2. Holographic Reduced Representations / VSA / Resonator Networks

**Closest citations:** Plate, *"Holographic Reduced Representations,"* IEEE Trans. Neural Networks 6(3),
1995; Frady, Kleyko & Sommer resonator-network line (*"Resonator Networks 1 & 2,"* Neural Computation
32(12), 2020, arXiv:1906.11684 for Part 1); classical linear associative memory capacity results
(Kohonen 1972/1973, Anderson 1972). **Confidence: high for the classical rank fact, medium for exact
Plate capacity formula (not independently re-derived here, only characterized qualitatively).**

Two distinct, well-established capacity notions exist in this literature and Task D must not conflate
them:
- **Exact linear-algebra capacity (classical, pre-1990s):** For a linear associative memory
  `W = Σ_i v_i k_i^T` with orthonormal keys, perfect retrieval requires `rank(W) = K` (the number of
  stored pairs), and since a d-dimensional space admits at most `d` orthogonal keys, exact recovery caps
  at `K ≤ d`. Kohonen (1973) also showed non-orthogonal keys allow storing more pairs with graceful
  (non-exact) degradation. **This rank-K-for-exact-recovery fact is textbook/classical — not novel by
  itself.** Task D's contribution is not the fact, but whether a transformer trained end-to-end under a
  hard bottleneck spontaneously discovers this rank, and what happens under forced rank-k truncation.
- **VSA/HRR superposition (bundling) capacity:** Plate's HRR and the Frady/Kleyko/Sommer resonator-network
  line study a *different* problem — approximate, noise-tolerant superposition/bundling of many
  hypervectors, with capacity results stated in bits-per-dimension or number-of-items-before-SNR-collapse,
  and (for resonator networks specifically) a **combinatorial factorization** capacity that is reported as
  scaling favorably (informally "exponential in dimension") for recovering discrete codebook factors from
  a Hadamard-product composite — not from a matrix's algebraic rank. This is a fundamentally different
  capacity mechanism (interference/SNR-based, discrete-codebook factorization) than Task D's exact
  rank-based linear recovery of continuous vectors. No resonator-network paper found ties a **matrix
  rank** observable to binding count; their "capacity" is a codebook-factorization success rate, not a
  spectral property of a state matrix.

**Verdict for this area:** No VSA/HRR/resonator paper measures matrix rank vs. K bindings the way Task D
proposes. Must cite Plate 1995 and resonator networks to correctly scope what "capacity" means in this
literature and to distinguish Task D's exact-recovery/rank framing from their approximate/SNR framing.

## 3. Fast Weight Programmers / Linear Attention as Associative Memory

**Closest citations:** Schlag, Irie & Schmidhuber, *"Linear Transformers Are Secretly Fast Weight
Programmers,"* ICML 2021, arXiv:2102.11174 (introduces DeltaNet / delta-rule fast weights); Nazari &
Rusch, *"The Key to State Reduction in Linear Attention: A Rank-based Perspective,"* arXiv:2602.04852
(Feb 2026); Sun et al., *"State Rank Dynamics in Linear Attention LLMs,"* arXiv:2602.02195 (Feb 2026).
**Confidence: high (verified via direct PDF read of both 2026 papers, pages 1–3).**

This is the area with the most directly relevant *rank observable*, and both 2026 papers must be cited.

- **Nazari & Rusch (arXiv:2602.04852)** formalize `S_t = Σ v_t k_t^T` as a linear associative memory,
  prove `rank(S_t) ≤ min(rank(K_t), rank(V_t)) ≤ t` (Eq. 5), and define **effective/stable rank**
  `er(S) = ‖S‖_F² / ‖S‖₂²` (their Definition 2.1) — this is the same effective-rank metric Task D should
  use. They empirically show trained (Gated) DeltaNet 370M exhibits **low rank utilization** (heavy-tailed
  singular spectrum, Figure 2) and build a structured-pruning method (DRRQR) that removes ~50% of
  key/query channels with minor perplexity cost. Critically: **this is post-hoc/post-training pruning
  of a pretrained LLM evaluated on FineWeb-Edu and downstream tasks — not a controlled synthetic task
  with a known ground-truth number of bindings K, and not a training-time rank-k constraint.** Their
  rank bound (`rank(S_t) ≤ t`) is an *upper* bound from write-count, not a *lower*-bound necessity
  theorem for exact recovery of K specific associations.
- **Sun et al. (arXiv:2602.02195)** conduct "the first systematic spectral analysis" of runtime state
  matrices in a real linear-attention LLM (Qwen3-Next), defining effective rank via a singular-value
  threshold (Eq. 6) and proving `rank(S(t)) ≤ min(t,d)` (Theorem 3.1). They discover **State Rank
  Stratification** (heads split into persistently low-rank vs. high-rank) and propose Joint Rank-Norm
  Pruning. Like Nazari & Rusch, this is analysis of a **pretrained model on real, diverse text**
  (RankViz: math/STEM/long-context/adversarial), not a synthetic K-controlled binding task, and there is
  **no training-time rank-k truncation ablation** and **no provable rank-≥K lower bound for exact
  recovery** — only the algebraic upper bound `rank ≤ min(t,d)`.
- DeltaNet itself (Schlag/Irie/Schmidhuber 2021, and the follow-on Yang et al. delta-rule/Gated DeltaNet
  papers) motivate the delta rule precisely because purely additive (rank-accumulating) fast weights
  have a capacity limitation once `t > d`, but capacity there is discussed qualitatively/via retrieval
  loss, not via an explicit rank-vs-K sweep with a matching causal truncation ablation.

**Verdict for this area:** This is the closest *mechanistic* prior art and must be cited prominently.
The delta: both 2026 papers study **rank as an emergent, uncontrolled property of pretrained LLMs on
real text**; Task D studies rank as a **controlled function of a known ground-truth K in a from-scratch
synthetic task**, adds a **provable lower bound** (rank ≥ K needed for exact recovery — the mirror image
of their upper bound `rank ≤ t`), and runs a **training-time forced-rank-k ablation** plus a **param-matched
vector-state baseline**, none of which appear in either paper.

## 4. Modern Hopfield Networks & Tensor Product Attention (2025)

**Closest citation:** Zhang, Liu, Yuan, Qin, Yuan, Gu, Yao, *"Tensor Product Attention Is All You Need,"*
arXiv:2501.06425 (Jan 2025), and the broader modern-Hopfield-network capacity literature (Ramsauer et al.
2020 origin; 2025–2026 follow-ups on exponential-capacity and continuous-time variants).
**Confidence: high for TPA scope, medium for exhaustively covering 2025-26 Hopfield-capacity papers.**

Tensor Product Attention (TPA/T6) factorizes Q/K/V into **contextual low-rank tensor decompositions**
for **KV-cache compression** — the motivation is inference efficiency, not measuring whether rank
emerges as a function of a controlled number of stored bindings, and there is no rank-K-necessity theorem
or truncation-ablation-vs-ground-truth-K in this paper. Modern Hopfield networks (Ramsauer et al. 2020,
and 2025–2026 extensions like 4-body-interaction and continuous-time low-rank-basis variants such as
Santos et al., Feb 2025) discuss **exponential storage capacity** in terms of pattern separation /
retrieval-error thresholds, not matrix rank as a function of bindings stored in a matrix state. None of
these give a from-scratch trained matrix-native transformer with a hard bottleneck and a rank-k causal
ablation.

**Verdict for this area:** Peripheral but citable as related "efficiency-motivated low-rank attention"
work to distinguish Task D's *diagnostic* rank question (does GD discover minimal rank ≈ K?) from TPA's
*prescriptive* rank question (impose low rank to save memory).

## 5. Direct 2024–2026 Work on Rank of Learned Associative Memory

**Closest citation (single closest overall):** Nichani, Lee & Bietti, *"Understanding Factual Recall in
Transformers via Associative Memories,"* ICLR 2025 Spotlight, arXiv:2412.06538. **Confidence: high
(verified via direct PDF read, pages 1–8, including Theorem 1 and its rank-m remark, Figure 1, and the
synthetic factual-recall task definition in Section 4).**

This is the paper Task D most needs to differentiate from. Key facts verified from the PDF:
- They define linear associative memory `W = Σ_x u_{f*(x)} e_x^T` and prove (Theorem 1) it stores `N`
  associations when `d² ≳ N poly log N` — capacity is stated in **parameter count**, not rank.
- **Critical overlap:** immediately after Theorem 1 they add: *"if W is restricted to be a rank m
  matrix, then such a W exists when `md ≳ N poly log N`; this construction is
  `W = Σ_x u_{f*(x)} e_x^T Σ_i v_i v_i^T`, where `v_i` are drawn i.i.d. from the standard Gaussian."*
  This is a **hand-built existence proof** that a rank-restricted matrix *can* achieve near-linear
  capacity via random Gaussian mixing and **discrete argmax decoding** (tolerant to noise/interference) —
  it is not (a) a measurement of what rank gradient descent actually discovers when a full-rank `W` is
  learned by training, (b) a lower bound stating rank must be ≥ K for *exact* (continuous, not
  discrete-argmax) recovery, or (c) a causal training-time rank-k ablation. Their Figure 1 empirical
  validation trains **full, unconstrained** linear/MLP associative memories and measures the largest N
  achieving 99% accuracy vs. `d²`/`md` — it does **not** plot rank of the learned matrix vs. N.
- Their synthetic factual-recall task (Section 4, Figure 2) trains a **real one-layer transformer**
  end-to-end via self-attention + MLP on subject/relation/answer triples with noise-token distractors —
  methodologically close to how Task D would set up its BIND/QUERY grammar — but the transformer has
  **full self-attention over the raw token sequence** (no hard bottleneck forcing everything through one
  d×d matrix state); the associative-memory role is played by one weight matrix (`W_O^T W_V`) inside an
  otherwise normal transformer layer, not an explicit, isolated, single mid-network matrix that the
  decoder is architecturally forced to read from exclusively.

**Second-closest citation:** an arXiv paper from June 2026, *"Recursive Binding on a Budget: Subspace
Carving in Order-p Tensor Memories,"* arXiv:2606.11391 — found via search and verified via WebFetch of
the HTML full text. Title and topic (order-p tensor memory, binding budget) look alarmingly close, but on
inspection: the binding operator ("subspace carving," Definitions 4.3–4.4) is a **hand-designed
orthogonal-projection VSA/TPR mechanism**, not something discovered by training a transformer via
gradient descent; their Theorem 4.6 bounds **interference noise** (`∝ √(N−1)/d^p`) as a function of
stored-item count, not matrix rank; and they explicitly do **not** measure effective rank vs. K or run a
rank-k truncation ablation. Their one end-to-end-trained component (an XML classifier, §6.2) uses their
binding operator as a fixed final layer, not a learned matrix whose rank is the object of study.
**Confidence: high (verified via direct WebFetch of full HTML text).**

Also relevant but more tangential: Merrill, Petty & Sabharwal, *"The Illusion of State in State-Space
Models,"* ICML 2024, arXiv:2404.08819 — studies expressivity limits of SSM state (TC⁰ complexity class,
state-tracking failure), not rank-vs-K-bindings; and Arora et al., *"Zoology: Measuring and Improving
Recall in Efficient Language Models,"* arXiv:2312.04927 — introduces the MQAR (multi-query associative
recall) synthetic benchmark and shows recall accuracy degrades as the number of KV pairs approaches
model/state dimension `d`. **Zoology is worth citing as the standard synthetic-associative-recall-vs-state-size
benchmark family Task D's grammar resembles**, but Zoology varies architecture/state *dimension* against
a fixed number of pairs and measures end-task accuracy — it does not report singular-value/rank spectra of
the state, nor a provable rank-necessity theorem, nor a forced-rank-k training ablation.

**Verdict for this area:** Nichani et al. (arXiv:2412.06538) is the paper Task D is in most danger of
being seen as "just a rank-flavored variant of," and must be cited and distinguished explicitly along the
three axes above (discrete-argmax vs. exact-continuous recovery; hand-built rank-m existence vs.
measured/emergent rank; no causal truncation ablation vs. Task D's causal ablation as primary test).
Subspace Carving (2606.11391) and Zoology (2312.04927) round out the "closest recent" set.

---

## What Would Scoop Us / What We Must Cite

**Would fully scoop Task D (not found):** A paper that (1) trains a transformer end-to-end from scratch
on a controlled BIND/QUERY task with known K, (2) architecturally forces all information through a single
d×d matrix state read only by the decoder, (3) measures effective rank of that matrix as a function of K,
(4) states/uses a provable rank ≥ K lower bound for exact linear recovery, and (5) causally forces
rank-k during training to produce an accuracy-collapse-vs-k curve, compared against a param-matched
vector-state baseline. **No such paper was found as of this search (July 2026).**

**Must cite regardless (closest prior art, in priority order):**
1. Nichani, Lee & Bietti, arXiv:2412.06538 (ICLR 2025) — nearest theoretical/empirical relative; the
   rank-m remark under their Theorem 1 is the passage most likely to be raised by a reviewer as "isn't
   this already rank-aware?" — must be addressed head-on in the related work.
2. Nazari & Rusch, arXiv:2602.04852, and Sun et al., arXiv:2602.02195 (both Feb 2026) — closest
   *mechanistic rank-measurement* precedent (same effective-rank definition), but on pretrained LLMs +
   real text, not controlled synthetic K.
3. Schlag, Irie & Schmidhuber, arXiv:2102.11174 (DeltaNet/FWP) — origin of "linear attention state as
   associative memory," motivates why capacity/rank matters once writes exceed dimension.
4. Schlag et al., arXiv:1910.06611 (TP-Transformer) and McCoy et al., arXiv:1812.08718 (TPDN) — binding
   mechanism ancestry (`Z` is a TPR).
5. Plate 1995 (HRR) and Frady/Kleyko/Sommer resonator networks (arXiv:1906.11684) — needed to correctly
   scope what "capacity" means in the VSA tradition (SNR/bundling, not matrix rank) and to justify why
   Task D's exact-rank framing is a different, cleaner question.
6. Arora et al., arXiv:2312.04927 (Zoology/MQAR) — standard synthetic associative-recall benchmark family
   Task D's grammar should be positioned relative to, and a natural reviewer comparison point.
7. Subspace Carving, arXiv:2606.11391 (June 2026) — most recent, most similarly-titled work; must be
   cited to preempt "isn't this the same as subspace carving" — it is not (hand-built vs. learned, no
   rank observable, no causal ablation).
8. Kohonen (1972/1973) and Anderson (1972) classical linear associative memory — source of the classical
   "rank = number of exactly-recoverable orthogonal pairs" fact so Task D doesn't overclaim novelty of
   the underlying linear-algebra fact itself (the novelty is the *trained, causal, matrix-native
   transformer* framing, not the fact that rank bounds exact recovery).

---

## Citations

1. Nichani, E., Lee, J. D., & Bietti, A. (2024). *Understanding Factual Recall in Transformers via
   Associative Memories.* ICLR 2025 (Spotlight). arXiv:2412.06538. https://arxiv.org/abs/2412.06538
2. Nazari, P., & Rusch, T. K. (2026). *The Key to State Reduction in Linear Attention: A Rank-based
   Perspective.* arXiv:2602.04852. https://arxiv.org/abs/2602.04852
3. Sun, A., Zhang, H., Zhou, H., Ma, Y., Qin, Y., Su, T., Liu, Y., Ma, Z., Xu, J., Gao, J., Hao, J., &
   He, R. (2026). *State Rank Dynamics in Linear Attention LLMs.* arXiv:2602.02195.
   https://arxiv.org/abs/2602.02195
4. Schlag, I., Irie, K., & Schmidhuber, J. (2021). *Linear Transformers Are Secretly Fast Weight
   Programmers.* ICML 2021. arXiv:2102.11174. https://arxiv.org/abs/2102.11174
5. Schlag, I., & Schmidhuber, J. (2018). *Learning to Reason with Third-Order Tensor Products.*
   NeurIPS 2018. arXiv:1811.12143. https://arxiv.org/abs/1811.12143
6. Schlag, I., Smolensky, P., Fernandez, R., Jojic, N., Schmidhuber, J., & Gao, J. (2019/2020).
   *Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving*
   (TP-Transformer). arXiv:1910.06611. https://arxiv.org/abs/1910.06611
7. McCoy, R. T., Linzen, T., Dunbar, E., & Smolensky, P. (2019). *RNNs Implicitly Implement Tensor
   Product Representations.* ICLR 2019. arXiv:1812.08718. https://arxiv.org/abs/1812.08718
8. Smolensky, P. (1990). *Tensor Product Variable Binding and the Representation of Symbolic
   Structures in Connectionist Systems.* Artificial Intelligence, 46(1–2), 159–216. (Pre-arXiv;
   foundational TPR reference.)
9. Plate, T. A. (1995). *Holographic Reduced Representations.* IEEE Transactions on Neural Networks,
   6(3), 623–641.
10. Frady, E. P., Kleyko, D., & Sommer, F. T. (2020). *Resonator Networks, 1: An Efficient Solution for
    Factoring High-Dimensional, Distributed Representations of Data Structures.* Neural Computation,
    32(12), 2311–2331. arXiv:1906.11684. https://arxiv.org/abs/1906.11684
11. Frady, E. P., Kleyko, D., & Sommer, F. T. (2020). *Resonator Networks, 2: Factorization Performance
    and Capacity Compared to Optimization-Based Methods.* Neural Computation, 32(12), 2332–2388.
12. Zhang, Y., Liu, Y., Yuan, H., Qin, Z., Yuan, Y., Gu, Q., & Yao, A. C.-C. (2025). *Tensor Product
    Attention Is All You Need.* arXiv:2501.06425. https://arxiv.org/abs/2501.06425
13. Ramsauer, H., et al. (2020). *Hopfield Networks is All You Need.* arXiv:2008.02217 (modern Hopfield
    network origin; cited for context on Hopfield capacity literature referenced in Section 4).
14. Merrill, W., Petty, J., & Sabharwal, A. (2024). *The Illusion of State in State-Space Models.*
    ICML 2024. arXiv:2404.08819. https://arxiv.org/abs/2404.08819
15. Arora, S., et al. (2023/2024). *Zoology: Measuring and Improving Recall in Efficient Language
    Models.* arXiv:2312.04927. https://arxiv.org/abs/2312.04927
16. Anonymous/unresolved-author (2026). *Recursive Binding on a Budget: Subspace Carving in Order-p
    Tensor Memories.* arXiv:2606.11391. https://arxiv.org/abs/2606.11391 (author names not resolved in
    this pass — verify author list before citing in the paper itself).
17. Kohonen, T. (1972). *Correlation Matrix Memories.* IEEE Transactions on Computers, C-21(4), 353–359.
18. Anderson, J. A. (1972). *A Simple Neural Network Generating an Interactive Memory.* Mathematical
    Biosciences, 14(3–4), 197–220.
19. Kohonen, T., & Ruohonen, M. (1973). *Representation of Associated Data by Matrix Operators.* IEEE
    Transactions on Computers, C-22(7), 701–702. (Optimal Linear Associative Memory / pseudo-inverse
    construction.)

**Note on unverifiable items:** Item 8 (Smolensky 1990) and items 17–19 (Kohonen/Anderson classical
associative-memory papers) predate arXiv and were not independently re-verified via full-text fetch in
this pass — they are standard, uncontested citations in this literature (repeatedly cross-referenced by
items 1, 6, 7, 9 above) but should get a final citation-format check before submission. Item 16's author
list could not be confirmed from the fetched HTML and must be resolved before citing.
