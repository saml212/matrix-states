# Research: Prior Art for the K-vs-P Rank Crossover (Chapter 2 Gauntlet)

Date: July 2026. Scope: does published literature test, predict, or refute the
Chapter 2 prediction that a P-slot matrix bottleneck produces a FLAT rank-k
ablation curve for K≤P and a BENT curve for K>P? Searches run via web search +
arXiv/OpenReview fetch, July 2026. Distinguishes VERIFIED (fetched primary
source) from REPORTED (search-snippet only, not independently confirmed) from
UNVERIFIED (could not retrieve full text — flagged, not asserted as fact).

---

## 1. Has the K-vs-P crossover been tested?

**Confidence: Low.** No paper found that runs the specific experiment (sweep
K items vs P slots/rank budget, plot a rank-k or slot-k ablation curve, look
for a bend at K=P). The closest adjacent work is scattered across three
literatures that never intersect on this exact test:

- **Anthropic, Toy Models of Superposition** (2209.10652, VERIFIED via
  fetch): the phase diagram between feature importance/sparsity and
  representation dimension shows a *sharp* transition — below a critical
  sparsity, features get their own orthogonal dimension (capacity 1, no
  interference); above it, superposition appears and per-feature capacity
  degrades smoothly. This is the closest **theoretical** analog of a K-vs-P
  crossover, but it is about sparse *feature* superposition under a
  reconstruction objective, not about K discrete *items* held for a joint
  reasoning readout, and it never uses matrix rank as the axis.
- **The Illusion of Superposition?** (2604.06374, VERIFIED via fetch,
  Rizvi-Martel, Rabusseau, Mosbach): shows depth-as-capacity, not
  rank-as-capacity — "unless model capacity is sufficiently constrained,
  superposition does not emerge as the de facto solution" (their §6,
  quoted below in §4). This is a capacity/shortcut crossover, but the
  capacity axis is *transformer depth* (2/4 vs 8/12 layers), not a
  slot/rank budget. No rank measurement anywhere in the paper.
- **CoT2** (2505.23648, VERIFIED via fetch): reports an *empirical*
  dimension threshold — "above a certain embedding dimension threshold, the
  continuous model is superior to discrete [CoT] in tasks involving
  search" — tested at d ∈ {16,24,32,40} on MNNS/ProsQA/ProntoQA. This is
  the closest **empirical** precedent for a capacity threshold, but (a) no
  formula K≤f(d) is derived, (b) no rank or effective-dimension of the
  learned representation is measured, and (c) the "items" axis (K) is never
  independently swept against d — only d is swept, with task difficulty
  fixed.
- **Slot Attention / Perceiver IO** (object-centric literature): slot-count
  vs object-count mismatches are studied (e.g. Adaptive Slot Attention,
  2406.09196, REPORTED via search snippet, CVPR 2024) but the failure mode
  studied is *segmentation quality and generalization to novel slot/object
  cardinalities*, not a rank-truncation accuracy curve, and slots there are
  literally separate positions/vectors (P-slot, not single-state
  rank-in-superposition) — i.e. it's the "one item per slot" side of the
  Chapter 2 design space, not the rank-superposition side.
- **Classical working-memory literature** (Zhang & Luck 2008; Bays,
  Catalao & Husain 2009, PMC3118422, VERIFIED via fetch): the slot-vs-resource
  debate is structurally the closest human/cognitive analog to K-vs-P, but
  the empirical finding argues *against* a sharp bend — precision degrades
  **gradually** as items increase, even below the classical "3-4 item"
  capacity limit, which a discrete-slot model would not predict. This is a
  headwind for the Chapter 2 prediction of a genuinely FLAT segment for
  K≤P: real resource-limited stores show graded interference before the
  hard pigeonhole limit, not a clean flat-then-bend curve.
- **Internal, unpublished**: this repo already scoped and gauntletted (but
  has not run to completion) essentially this exact experiment —
  `matrix-thinking/scripts/build_prosqa_multi.py` (ProsQA-MULTI-k, K
  disjoint entities merged into one context with a joint single-subject
  query) plus `run_matrix_codi.py --rank-loss {entropy,nuclear}` — see
  `matrix-thinking/team-output-2026-04-28/`. The internal skeptic agent
  flagged the same risk as external Q3 below: "ProsQA-MULTI is rank-k by
  ASSERTION, not CONSTRUCTION" (`EMP_VERDICT.md`, risk #3). This is prior
  work in the same codebase, not published literature, but it is the
  closest artifact to Task A' that exists anywhere, and its own audit
  concluded a rank-1-constrained kill-switch experiment (D4) must run
  before the 3×3 grid is trusted — directly relevant to how Chapter 2
  should be gated.

**Bottom line:** no external paper has run "K items vs P slots, rank-1 per
slot, joint readout, rank-k ablation." The nearest theoretical relative
(toy-model superposition phase diagram) supports the *shape* of the
prediction; the nearest empirical relative (working-memory resource model)
argues the transition will likely be graded, not sharp.

---

## 2. Capacity theory: what do Hopfield / TPR / superposition results predict?

**Confidence: Medium.** Multiple frameworks give bounds, but they disagree
on whether the K=P transition is sharp or graded, and none directly analyze
a *rank-r-constrained* store (they analyze dimension-n stores where rank is
implicitly full).

- **Classical Hopfield (Amit, Gutfreund, Sompolinsky 1985; Phys. Rev. Lett.
  55(14):1530, REPORTED via search, standard citation)**: Hebbian storage
  capacity αc ≈ 0.138, i.e. M = 0.138·N patterns in an N-dimensional
  network. Mechanistically, the weight matrix is W = Σ_μ ξ_μξ_μᵀ, a sum of
  M rank-1 outer products — literally rank ≤ M as long as patterns are
  independent, so the *hard* pigeonhole (rank exceeds ambient dimension N)
  only bites at M=N. But retrieval already fails at M ≈ 0.138N, **well
  before** the rank pigeonhole would force it — crosstalk noise from
  non-orthogonal random patterns degrades recall continuously as M grows,
  long before M=N. **This is the strongest theoretical argument that a
  rank-based K-vs-P crossover in a *linear* readout will be graded, not a
  clean flat/bend step function**, unless the stored patterns are
  constructed to be exactly orthogonal (as Task A'/B's entity vectors could
  be, by construction, unlike random Hebbian patterns).
- **Modern / dense Hopfield networks (Ramsauer et al., "Hopfield Networks
  is All You Need," 2008.02217, REPORTED via search but formula is a
  standard, widely-cited result)**: with a log-sum-exp (softmax-like)
  energy function, storage capacity becomes **exponential** in dimension,
  N ~ 2^(d/2) for well-separated binary patterns, with retrieval in one
  update and exponentially small error. The mechanism is a strongly
  nonlinear (LSE) readout that suppresses crosstalk — directly analogous to
  this project's own finding that a *nonlinear-in-Z* readout
  (bilinear+GELU) is required to make gradients rank-sensitive at all
  (STATE.md "positive control" note). **Prediction: if Chapter 2's readout
  is a genuinely nonlinear multi-probe head (not linear/flatten), theory
  supports a sharper crossover close to the hard pigeonhole; a
  linear/near-linear readout will instead show graded Hopfield-style
  degradation before K=P.**
- **Smolensky's Tensor Product Representations (1990, VERIFIED via
  search/secondary sources, e.g. McCoy/Linzen/Dunbar/Smolensky ICLR 2019
  read of the original)**: TPRs bind K (role, filler) pairs via outer
  products, M = Σ rᵢ⊗fᵢ, with **exact, interference-free retrieval**
  *if and only if* the role vectors are linearly independent — i.e. this is
  the idealized, zero-crosstalk limit, and it predicts a genuinely sharp
  crossover at K = (number of independent roles available) = rank budget.
  This is the closest theoretical vindication of the Chapter-2-style flat
  segment. Caveat: exact orthogonal TPR binding requires roles to be
  pre-specified/orthogonal; a model *learning* roles end-to-end is not
  guaranteed to find an orthogonal role basis (this is exactly the internal
  skeptic's concern in §1 — "disjoint ≠ orthogonal").
- **Polysemanticity and Capacity in Neural Networks (Scherlis et al.,
  2210.01892, REPORTED via search)**: formalizes per-feature "capacity" in
  [0,1] with total capacity bounded by dimension; optimal allocation is
  monosemantic (capacity≈1, i.e. "gets its own slot") for high-importance
  features and polysemantic (shared, degraded capacity) for
  lower-importance ones, with the split determined by where marginal loss
  reduction equalizes across features — again a framework that predicts a
  smooth capacity trade-off rather than an all-or-nothing bend, unless all
  K items are equally important (in which case the trade-off becomes more
  step-like).

**Does theory predict a crossover at K ≈ P·(capacity-per-slot)?** Yes, in
the idealized/orthogonal limit (TPR, and the flat region of the toy-model
superposition phase diagram in §1). In every framework that includes
realistic interference (classical Hopfield, resource-model working memory,
Scherlis capacity allocation), the transition is graded rather than a clean
corner, and becomes sharp only when the readout is strongly nonlinear
(modern/dense Hopfield, log-sum-exp) or the K stored items are exactly
orthogonal by task construction. This is directly actionable for Chapter 2:
the MultiProbeHead's bilinear/nonlinear structure is the right lever to get
a sharp crossover instead of a Hopfield-style graded one; a flatten/linear
readout should be expected to blur it (consistent with this project's own
matrix-CODI H1/H3 failure with the flatten readout).

---

## 3. Ground-truth rank of the candidate tasks (A' and B)

**Confidence: Medium** that no rigorous *minimum*-rank proof exists for
either candidate task in the literature; **Medium-High** that both carry a
real, demonstrated risk of a false-test low-rank collapse, based on directly
analogous empirical precedent.

- **Task B (K-way BFS frontier).** Reasoning by Superposition (2505.12514,
  VERIFIED via fetch) constructs the continuous thought as
  t_c = (1/√|𝒱_c|) Σ_{v∈𝒱_c} u_v — a specific *hand-built* solution whose
  rank equals frontier size **by construction**. This is an **existence
  upper bound** (a rank-K solution exists), not a **lower bound** (no proof
  that rank<K solutions cannot solve the task). Critically, their
  construction sets embedding dimension d = O(|Voc|) (3·d_TE + d_PE, with
  d_TE ≥ log(vocab)) — i.e. **the theory paper deliberately keeps dimension
  large enough that the model never has to compress below one-hot-per-vertex
  fidelity.** It therefore never tests, and gives no evidence about, the
  K>P pigeonhole regime Chapter 2 needs. Nobody has proven BFS-frontier
  tasks require rank K in a genuinely rank-constrained (P<K-capable) model.
- **Task A' (K parallel entity tracking, joint readout)** has no published
  rank-lower-bound proof either — it is not a task studied in this form in
  the literature searched. The closest published relative is ProsQA
  (single-subject reachability query), and there is direct, verified
  evidence that ProsQA-style tasks are prone to **rank-1/single-path
  collapse via shortcut learning**: The Illusion of Superposition
  (2604.06374, VERIFIED via fetch) reports that fine-tuned models on ProsQA
  "learned a shortcut solution: it first obtains the correct answer in a
  single forward pass, then copies it over through the latent tokens" —
  i.e. the model bypasses genuine multi-path/multi-rank reasoning entirely,
  exactly the "false test" failure mode Chapter 2 is worried about for Task
  B, and by close analogy for Task A'.
- **Directly relevant internal finding (same repo, unpublished):** the
  April 2026 rank-aware gauntlet's Empirical-Skeptic verdict
  (`matrix-thinking/team-output-2026-04-28/EMP_VERDICT.md`, risk #3, local
  file, verified by reading) states almost verbatim the same concern as
  Chapter 2's attack #4: *"ProsQA-MULTI is rank-k by ASSERTION, not
  CONSTRUCTION. The bilinear-head argument requires that question text
  alone does not enumerate targets. ProsQA's generator likely lets W_down
  memorize P→{a1..ak} mappings, with rank-1 Z encoding 'which question.'"*
  It specifies a non-negotiable diagnostic (train a rank-1-constrained
  model on the MULTI-k task; if accuracy matches the unconstrained model,
  the task does not actually require rank-K) that has not yet been run.
  Chapter 2 should treat this as a **mandatory pre-registered kill-switch**
  before trusting a bent rank-k curve on either Task A' or Task B — the
  bend must be shown to be functionally necessary (probe-level, per-item
  ablation), not just correlational.
- **Emergence of Superposition** (2509.23365, VERIFIED via fetch) trains
  continuous-CoT models on graph reachability and shows the index-matching
  logit grows then saturates, but explicitly does **not** measure rank or
  give any minimum-rank argument — so it neither supports nor refutes a
  low-rank collapse risk for BFS-style tasks; it's simply silent on the
  rank question.

**Net:** Both candidate tasks are exposed to the same failure mode that
already killed the flatten-readout matrix-CODI experiments in this
project's own history (ProsQA turned out solvable via rank-blind
shortcuts) — and there is direct published evidence (Illusion of
Superposition) that ProsQA-family tasks specifically invite this failure.
Task construction should force disjointness at the level the internal
skeptic recommends (verify empirically via probe cosine similarity / a
rank-1-constrained kill-switch run) rather than assuming disjoint DAG
entities imply orthogonal representations.

---

## 4. Does the theory literature support or undermine the crossover framing?

**Confidence: Medium-High** that none of the four target papers directly
tests or refutes the K-vs-P rank crossover — each sits one step away from
it, consistent with STATE.md's existing "critical gap nobody has filled"
assessment.

- **Reasoning by Superposition** (2505.12514): supports the *qualitative
  shape* of the hypothesis (rank tracks number of held items) as an
  existence construction, but — as in §3 — deliberately avoids the
  capacity-limited regime by letting d scale with vocabulary size. It
  neither confirms nor tests a crossover; it assumes the crossover away.
- **CoT2** (2505.23648): the only paper of the four with an *empirical*
  capacity-threshold result (accuracy jumps once d crosses a
  task-dependent value, tested d∈{16,24,32,40}), but never computes rank
  of the resulting continuous thoughts and never independently sweeps the
  "K items needed" axis against d — d and task difficulty are conflated,
  not orthogonally varied the way Chapter 2 proposes.
- **The Illusion of Superposition?** (2604.06374): actively complicates the
  "superposition emerges naturally" premise. Verified quote (§6 of the
  paper): *"models with too much capacity show signs of learning
  shortcuts... superposition is brittle: unless model capacity is
  sufficiently constrained, superposition does not emerge as the de facto
  solution."* Their capacity axis is **transformer depth** (2/4-layer
  models show genuine step-wise BFS-like probe signatures; 8/12-layer
  models bypass it with single-pass shortcuts, Table 2b: 94.5%→13.8%
  accuracy drop without latents at 2-4L vs 80.7%→62.8% / 61.6%→63.0% at
  8-12L). This is a genuine capacity-vs-shortcut crossover in the
  literature — but on a *different* capacity axis (depth) than the one
  Chapter 2 proposes (matrix rank/slot count P). It is suggestive support
  for the general principle ("capacity constraints matter for whether
  superposition emerges") and for Chapter 2's choice to train
  matrix-native from scratch with a tight architecture (their finding that
  *only* from-scratch, capacity-constrained models show real superposition
  signatures matches Chapter 2's design exactly) — but it is not a test of
  the rank axis specifically, and should not be over-read as validating
  the rank mechanism.
- **Emergence of Superposition** (2509.23365): studies training dynamics
  (index-matching logit growth/saturation) with no rank or dimension
  measurement at all — silent on the crossover question.

**Net:** no paper in this cluster tests or refutes the rank-based K-vs-P
crossover. The literature is unanimous that *some* capacity constraint is
necessary for superposition/parallel-path behavior to emerge under
training (CoT2's dimension threshold; Illusion of Superposition's depth
constraint) but none of them make rank the capacity axis, and none give a
functional form. Chapter 2 would be the first to test rank specifically
as the constrained resource in a from-scratch, K-vs-P swept design.

---

## 5. Novelty check

**Confidence: Medium.** Search coverage: arXiv (cs.LG/cs.CL, 2024-2026),
OpenReview, GitHub, plus general web search; not exhaustive across all
venues/preprint servers, and could miss very recent (last few weeks)
uploads.

"Single-state matrix bottleneck + K/P crossover + rank-k ablation, trained
end-to-end on a reasoning task requiring K items" was **not found published**
as of this search. Closest partial matches, none of which combine all three
elements:

1. **Rank-k truncation as an evaluation methodology** — already used by
   this project's own prior matrix-CODI experiments (Rounds 1-9, bolt-on
   architecture) and is the same protocol Chapter 2 proposes to reuse. Not
   external prior art, but worth noting Chapter 2 is not inventing the
   *measurement* technique, only the *task design* that would give it
   teeth (a task with a provable/near-provable rank requirement, rather
   than a bolt-on vector-pretrained backbone that turned out rank-blind).
2. **Toy Models of Superposition** (2209.10652) — closest theoretical
   crossover (features vs dimensions, sharp phase transition) but for
   sparse autoencoder-style feature packing under a reconstruction
   objective, not K discrete reasoning items with a joint multi-item
   readout, and no matrix/rank framing (their model is vector-valued with
   ordinary Euclidean dimension as the resource, not SVD rank of a
   matrix-valued state).
3. **Slot Attention / Perceiver-style P-slot architectures** — the "P
   slots, rank-1 per item" side of the design space already exists and is
   well-studied for object-centric vision, but never crossed with a
   rank-in-superposition alternative or a K-vs-P sweep with an ablation
   curve; and never applied to a from-scratch reasoning/entity-tracking
   task.
4. **Internal, unpublished (this repo).** The closest existing artifact by
   far: `matrix-thinking/scripts/build_prosqa_multi.py` +
   `run_matrix_codi.py --rank-loss` (April 2026, gauntletted, not yet run
   to completion). This is essentially Task A' already scoped for a
   *vector*-CODI bolt-on backbone with an explicit rank-inducing auxiliary
   loss, rather than Chapter 2's matrix-native from-scratch architecture,
   and its own internal audit trail (SYNTHESIS.md, EMP_VERDICT.md,
   ATTACK_DATASET.md) already raised the identical "task might be
   rank-1-solvable by assertion" concern independently reached in the
   Chapter 2 attack pass. Since it's unpublished, it doesn't affect the
   external novelty verdict, but it means Chapter 2 should explicitly
   reference/reuse or supersede this design rather than re-deriving it
   from scratch — and should not claim novelty over its own prior
   internal work without distinguishing "matrix-native end-to-end" (new)
   from "vector bolt-on + rank loss" (already scoped, same repo).

**Conclusion:** the K/P crossover experiment as specified (matrix-native,
end-to-end, rank-k ablation, K vs P swept independently) is an unoccupied
experiment in the published literature as of July 2026. The largest risk to
the novelty claim is not a competing publication but the project's own
unpublished internal design (ProsQA-MULTI-k), which tests a closely related
question on a different (bolt-on, vector) architecture and has already
surfaced the central methodological risk (task rank by assertion, not
construction) that Chapter 2 must also resolve.

---

## Citations

Verified = fetched full text/abstract directly. Reported = search-snippet
only; content not independently re-derived from primary source, treat
numeric claims with appropriate caution.

**Reasoning by Superposition: A Theoretical Perspective on Chain of
Continuous Thought** — Zhu, Hao, Hu, Jiao, Russell, Tian. NeurIPS 2025.
arXiv:2505.12514. https://arxiv.org/abs/2505.12514 — VERIFIED.

**Continuous Chain of Thought Enables Parallel Exploration and Reasoning
(CoT2)** — Gozeten, Ildiz, Zhang, Harutyunyan, Rawat, Oymak. ICLR 2026.
arXiv:2505.23648. https://arxiv.org/abs/2505.23648 — VERIFIED.

**The Illusion of Superposition? A Principled Analysis of Latent Thinking
in Language Models** — Rizvi-Martel, Rabusseau, Mosbach. 2026.
arXiv:2604.06374. https://arxiv.org/abs/2604.06374 — VERIFIED.

**Emergence of Superposition: Unveiling the Training Dynamics of Chain of
Continuous Thought** — 2025. arXiv:2509.23365.
https://arxiv.org/abs/2509.23365 — VERIFIED.

**Toy Models of Superposition** — Elhage et al., Anthropic. 2022.
arXiv:2209.10652. https://arxiv.org/abs/2209.10652 /
https://transformer-circuits.pub/2022/toy_model/index.html — VERIFIED.

**Polysemanticity and Capacity in Neural Networks** — Scherlis, Sachan,
Jermyn, Elhage, Sharma. 2022. arXiv:2210.01892.
https://arxiv.org/abs/2210.01892 — REPORTED (PDF fetch failed; content from
search snippets and secondary summaries only).

**Hopfield Networks is All You Need** — Ramsauer, Schäfl, et al. 2020.
arXiv:2008.02217. https://arxiv.org/abs/2008.02217 — REPORTED (standard,
widely-cited exponential capacity result 2^(d/2); not independently
re-derived from primary text in this session).

**Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural
Networks** — Amit, Gutfreund, Sompolinsky. Phys. Rev. Lett. 55(14):1530,
1985. — REPORTED (classical 0.138N capacity result; standard citation,
not independently re-derived from primary text in this session).

**Tensor Product Variable Binding and the Representation of Symbolic
Structures in Connectionist Systems** — Smolensky. Artificial Intelligence,
1990. — REPORTED (via secondary sources, incl. McCoy, Linzen, Dunbar,
Smolensky, "RNNs Implicitly Implement Tensor-Product Representations,"
ICLR 2019, https://tallinzen.net/media/papers/mccoy_linzen_dunbar_smolensky_2019_iclr.pdf).

**The precision of visual working memory is set by allocation of a shared
resource** — Bays, Catalao, Husain. Journal of Vision, 2009.
https://www.paulbays.com/pdf/BayCatHus09.pdf / PMC3118422 — VERIFIED
(abstract/summary level via PMC).

**Adaptive Slot Attention: Object Discovery with Dynamic Slot Number** —
CVPR 2024. arXiv:2406.09196. https://arxiv.org/abs/2406.09196 — REPORTED.

**Perceiver IO: A General Architecture for Structured Inputs & Outputs** —
Jaegle et al., DeepMind. 2021. arXiv:2107.14795.
https://arxiv.org/abs/2107.14795 — REPORTED (ablation details on latent
count vs compute/representation trade-off from search summary; already in
project references.md under Multi-Modal/Domain-Agnostic).

**LLM Latent Reasoning as Chain of Superposition** — 2025. arXiv:2510.15522.
https://arxiv.org/abs/2510.15522 — VERIFIED (checked and found NOT closely
relevant on inspection: uses a fixed compression ratio r, not a
K-vs-capacity sweep; does measure effective rank of vocab-space embeddings
via singular-value decay, but not tied to a K-items capacity argument).

**Beyond Dense States: Elevating Sparse Transcoders to Active Operators for
Latent Reasoning** — 2026. arXiv:2602.01695.
https://arxiv.org/abs/2602.01695 — REPORTED (checked and found NOT closely
relevant: sparsity budget governs number of active semantic features per
step, not a K-item/P-slot rank crossover).

**Effective Reasoning Chains Reduce Intrinsic Dimensionality** — Prasad,
Joshi, Lee, Bansal, Shaw. 2026. arXiv:2602.09276.
https://arxiv.org/abs/2602.09276 — REPORTED (PDF fetch did not yield full
text; correlation between intrinsic dimension and generalization reported
via search snippet, Spearman r=0.93 figure not independently verified from
primary source).

**Latent Thinking Optimization** — ICLR 2026. arXiv:2509.26314.
https://arxiv.org/abs/2509.26314 — REPORTED (finding that "effective rank
of correct latent thoughts is consistently lower than incorrect ones" is
from search-summary level only, not independently confirmed against full
text).

**Self-Attention Limits Working Memory Capacity of Transformer-Based
Models** — 2024. arXiv:2409.10715. https://arxiv.org/abs/2409.10715 —
UNVERIFIED (found via search, PDF fetch failed to extract text; flagged,
not used to support any claim above).

**Internal/unpublished, this repository** (not external prior art, cited
for project continuity):
- `matrix-thinking/scripts/build_prosqa_multi.py` — ProsQA-MULTI-k
  generator.
- `matrix-thinking/scripts/run_matrix_codi.py` — `--rank-loss
  {none,entropy,nuclear}` flags added April 2026.
- `matrix-thinking/team-output-2026-04-28/SYNTHESIS.md`,
  `EMP_VERDICT.md`, `ATTACK_DATASET.md`, `ATTACK_RANK_AWARE.md`,
  `AUDIT_DATASET.md`, `AUDIT_RANK_AWARE.md` — gauntlet record for the
  closest existing internal design to Task A'.
