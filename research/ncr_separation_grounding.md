# NCR Capability-Separation Grounding — Literature Verification

**Date:** 2026-07-16
**Scope:** Verify (not recall) the 8 claims underpinning the NCR ("Native Composition Reads")
capability-separation argument, run a searched novelty-threat sweep for 2024–2026 matrix-power /
repeated-squaring / exact-composition fast-weight reads, and draw the precise novelty boundary.
**Method:** live web search + WebFetch against arXiv/OpenReview/ACL Anthology for every item; cross-
checked against this repo's own prior verified research (`research/task-d-novelty-july2026.md`,
`matrix-thinking/NOVEL_ARCH_WATERFALL.md` §1–§5) where those passes already did independent
arXiv-fetch verification, to avoid re-litigating settled citations and to catch any drift.

---

## Part 1 — Claim-by-claim verification

### 1. Log-precision transformers ⊆ uniform TC⁰ (Merrill & Sabharwal)

**VERIFIED**, with one attribution nuance.

- Paper: William Merrill & Ashish Sabharwal, **"The Parallelism Tradeoff: Limitations of
  Log-Precision Transformers,"** *Transactions of the ACL (TACL)*, vol. 11, pp. 531–545, 2023.
  **arXiv:2207.00729.**
- Core result confirmed: any log-precision transformer (precision logarithmic in input length,
  feed-forward nets computable in space linear in their input) can be simulated by a
  constant-depth, logspace-uniform **TC⁰** threshold-circuit family.
- Nuance: the paper's own headline corollaries are about specific problems (e.g., transformers
  cannot solve linear equalities or check CFG-with-ε-productions membership unless **L ≠ P**). The
  precise phrasing "hence transformers cannot solve NC¹-hard problems unless TC⁰=NC¹" is the
  **standard downstream deduction** (TC⁰ ⊆ NC¹ is well known; an NC¹-complete problem outside TC⁰
  unless TC⁰=NC¹ follows immediately) — and it is exactly the deduction Merrill, Petty & Sabharwal
  make explicit themselves in the 2024 follow-up (item 4 below), applying it to the S₅ word
  problem. Safe to cite Merrill & Sabharwal (2023) for the TC⁰ containment and Merrill/Petty/
  Sabharwal (2024) for the explicit NC¹-hardness corollary against S₅.

### 2. Barrington's theorem — S₅ word problem is NC¹-complete

**VERIFIED**, with a date nuance.

- Barrington's result: the word problem of the (non-solvable) permutation group **S₅** is
  **NC¹-complete**; more generally, the word problem of *any* fixed non-solvable finite group is
  NC¹-complete under AC⁰ reductions. Confirmed via UMass CS lecture notes (D. M. Barrington guest
  lecture) and independent complexity-theory summaries.
- Mechanism confirmed: width-5 permutation branching programs can simulate any NC¹ circuit (this is
  the surprising part — a branching program remembering < 3 bits of state per layer still captures
  all of NC¹).
- Date nuance: original result is the **1986 STOC conference paper**; the widely-cited journal
  version is **Barrington, "Bounded-width polynomial-size branching programs recognize exactly
  those languages in NC¹," J. Comput. Syst. Sci. 38(1), 1989**. "1986–89" in the claim is accurate
  as the conference→journal span.
- This is the correct canonical citation for "iterated group composition over a non-solvable group
  is the hardest state-tracking task at NC¹" — used verbatim by item 4's paper for its S₅
  hardness argument.

### 3. "Transformers Learn Shortcuts to Automata" (Liu et al., ICLR 2023)

**VERIFIED.**

- Paper: Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, Cyril Zhang, **"Transformers
  Learn Shortcuts to Automata,"** ICLR 2023 (oral). **arXiv:2210.10749** (v1 Oct 2022, v2 May 2023).
- Headline finding: a **low-depth** transformer (o(T) layers, T = sequence length) can exactly
  replicate *any* finite-state automaton's computation on a length-T input, via a hierarchical
  reparameterization of the recurrent dynamics — the "shortcut."
- Generalization-failure claim confirmed as a **secondary finding**, not the headline: the paper
  explicitly studies and confirms **brittleness** of these learned shortcuts — e.g., on a parity
  task, models trained on sequences with ~50% ones fail when tested on sequences with a
  substantially different bit-proportion, because "the model has leveraged the correlation between
  position and accumulated sum" rather than learning the true recurrent rule. Sparse-supervision
  regimes also break the Transformer while an LSTM baseline stays near-perfect. The authors propose
  a scratchpad/autoregressive mitigation but note it sacrifices the shortcut's parallelism.
- Correct usage in our claim: "learned constant-depth shortcuts fail beyond trained
  lengths/depths" is accurate as the paper's *brittleness* result, not its main construction
  result — cite both halves, don't conflate them.

### 4. "The Illusion of State in State-Space Models" (Merrill, Petty, Sabharwal, ICML 2024)

**VERIFIED.**

- Paper: William Merrill, Jackson Petty, Ashish Sabharwal, **"The Illusion of State in State-Space
  Models,"** ICML 2024. **arXiv:2404.08819.**
- Confirmed claim: diagonal-transition SSMs (the Mamba/S4/S6 class) cannot express computation
  outside **TC⁰**, so — like transformers — they **cannot solve simple state-tracking problems such
  as permutation composition** (the S₅ word problem), assuming TC⁰ ≠ NC¹. This is the direct
  descendant of items 1+2 combined into one paper.

### 5. Negative/extended-eigenvalue linear RNNs unlock state tracking

**VERIFIED**, both papers real and distinct.

- **Grazzi, Siems, Zela, Franke, Hutter, Pontil, "Unlocking State-Tracking in Linear RNNs Through
  Negative Eigenvalues,"** ICLR 2025. **arXiv:2411.12537.** Confirmed: Mamba/DeltaNet-class LRNNs
  restrict diagonal state-transition eigenvalues to [0,1]; this provably prevents solving parity
  (finite-precision LRNNs with only-positive-eigenvalue transitions cannot solve parity; non-
  triangular matrices are needed mod-3 counting); extending the eigenvalue range to include
  negative values empirically fixes parity and improves state-tracking generally.
- **Siems, Carstensen, Zela, Hutter, Pontil, Grazzi, "DeltaProduct: Improving State-Tracking in
  Linear RNNs via Householder Products,"** **arXiv:2502.10297**, NeurIPS 2025 poster. Confirmed as
  a distinct, real follow-up (same core author, Grazzi, is last-author here) — extends the DeltaNet
  "one online-gradient-step-per-token" view to **multiple Householder steps per token** via
  diagonal-plus-rank-`n_h` transition matrices, further improving state-tracking/LM performance.

### 6. Fast Weight Programmer lineage (Schlag, Irie & Schmidhuber, ICML 2021)

**VERIFIED.**

- Paper: Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber, **"Linear Transformers Are Secretly Fast
  Weight Programmers,"** ICML 2021, **PMLR v139, pp. 9355–9366**. Confirmed formal equivalence:
  linearized self-attention ≡ a fast-weight programmer (a slow net gradient-descent-programs the
  fast weights of another net via additive outer-product instructions), tracing to Schmidhuber's
  early-90s fast-weight controllers.
- Recurrent-FWP follow-up confirmed real: **"Going Beyond Linear Transformers with Recurrent Fast
  Weight Programmers,"** **arXiv:2106.06295**, NeurIPS 2021 (IDSIA group; official repo
  `IDSIA/recurrent-fwp`) — adds recurrence to both the slow and fast nets.

### 7. RWKV-7 "Goose" (Peng et al., 2025)

**VERIFIED.**

- Paper: Bo Peng et al. (18 authors), **"RWKV-7 'Goose' with Expressive Dynamic State Evolution,"**
  **arXiv:2503.14456** (March 2025).
- Confirmed expressivity claims (via abstract + follow-up search on the theorem content):
  RWKV-7's transition matrix is **non-diagonal and input-dependent** (generalized delta rule with
  vector-valued gating + in-context learning rates + a relaxed value-replacement rule); this lets a
  **single layer solve an S₅ state-tracking problem (NC¹-hard under AC⁰ reductions)** and a
  **constant number of layers recognize any regular language**. The paper states this **"exceeds
  the capabilities of Transformers under standard complexity conjectures, which are limited to
  TC⁰."** RWKV-7 is a 2.9B-param model, SoTA on multilingual 3B-scale benchmarks despite fewer
  training tokens.
- Relevant distinction for NCR: RWKV-7's state-tracking is achieved via **sequential O(h) recurrent
  state updates** (matrix-valued but rolled out one token at a time), not query-time O(log h)
  reads of an already-written state.

### 8. Nichani, Lee & Bietti (ICLR 2025, arXiv:2412.06538) — rank-1 argmax capacity

**VERIFIED, with a live-verification caveat — flag for spot-check before citing in a paper.**

- Paper confirmed: Eshaan Nichani, Jason D. Lee, Alberto Bietti, **"Understanding Factual Recall in
  Transformers via Associative Memories,"** ICLR 2025 **spotlight**, **arXiv:2412.06538.**
- My own live WebFetch of both the abstract page and the full-text HTML/PDF **could not surface**
  the specific rank-1/argmax-decoding claim in the rendered text (the tool's PDF extraction hit the
  paper's math-heavy remarks/appendix sections and returned mostly object metadata, not prose).
  Read in isolation, this would be a NOT-FOUND.
- However, this repo already has an **independently-verified prior research pass**
  (`research/task-d-novelty-july2026.md`, dated before this session, explicit method note: "all
  citations verified via arXiv/OpenReview fetch, not fabricated") that describes the exact
  mechanism: *"[Nichani, Lee & Bietti] formalize a linear associative memory `W = Σ u_{f*(x)}
  e_x^T`... and even give a rank-m construction showing W can be restricted to rank m and still
  store ≈md associations (their Theorem 1 remark)... for a discrete argmax-decoded capacity
  question."* Setting m=1 gives the ≈d-associations-from-rank-1 claim used in our hard rules.
- **Disposition:** treat as VERIFIED (cross-confirmed by an independent prior fetch that did
  successfully render the theorem text), but the exact theorem/remark number should be re-confirmed
  by direct human read of the PDF (not a tool fetch) before this citation appears in any
  submitted paper — my tooling could not independently re-derive it this session, which is a
  tool-extraction limitation, not evidence of absence, but that distinction is easy to lose. This
  is the one item in this memo I could not close to full personal confidence.

**Verification tally: 8/8 real papers, correctly attributed. 0 fabricated citations found. 2 items
carry attribution/precision nuances worth remembering (items 1, 3); 1 item (8) is VERIFIED only via
cross-reference to prior in-repo verification, not this session's own tool fetch — flagged for a
human spot-check.**

---

## Part 2 — Novelty-threat sweep (2024–2026): matrix-power / repeated-squaring reads, exact
composition at query time, O(log depth) composition reads

**Search strategy:** ran ~10 independent query variants across "matrix power," "repeated squaring,"
"matrix exponentiation," "exact composition," "in-context written operators," "O(log h)
composition," "Cayley graph / group word problem + transformer," "looped transformer + permutation
composition," directly against arXiv, spanning 2024–2026. Cross-checked against this repo's own
prior, more exhaustive sweep in `matrix-thinking/NOVEL_ARCH_WATERFALL.md` §1–§5 (2026-07-09), which
independently arrived at the same closest-neighbor set via a different search pass.

**No paper found that does all three of: (a) writes matrix operators FROM CONTEXT (not learned
purely in weights), (b) reads them via literal matrix-power / repeated-squaring for O(log h)
query-time cost, (c) claims EXACT (not approximate) variable-depth composition.** This is a
searched absence, not an assumed one — the nearest neighbors on each individual axis:

- **Fast Weight Memory (FWM)** — Schlag, Munkhdalai & Schmidhuber, ICLR 2021, **arXiv:2011.07831**
  ("Learning Associative Inference Using Fast Weight Memory," confirmed real, LSTM + associative
  fast-weight matrix, differentiable per-step writes, composable facts for reasoning). This is the
  **single closest prior art** on axis (a) — in-context/online writes into a fast-weight matrix for
  compositional inference — but its reads are **recursive, LN-normalized, and non-linear** (fixed
  small number of recursive hops, N_r=3 in the paper), not a literal matrix power, and composition
  is **approximate**, not exact. (Matches this repo's independent finding in
  `NOVEL_ARCH_WATERFALL.md` §3.5.)
- **Log-Linear Attention** — **arXiv:2506.04761**, ICLR 2026. Replaces the fixed-size linear-
  attention hidden state with a **logarithmically-growing set of hidden states**, giving log-linear
  *compute* complexity. Closest match on the "O(log ·)" axis by name, but: (i) the log-ness is a
  training/inference **compute-complexity** property of a hierarchical summarization structure
  (Fenwick-tree-like), not a query-time repeated-squaring READ of a single written operator matrix;
  (ii) it targets standard LM/retrieval benchmarks, not relational/group composition; (iii) my tool
  could not confirm whether the hierarchy is information-preserving/exact vs. lossy — flagged as
  unresolved, but even in the best case for the threat, it is not framed as exact operator
  composition at variable query-time depth.
- **HOLA — "A Hippocampus for Linear Attention"** — **arXiv:2607.02303** (very recent, July 2026).
  Confirmed different mechanism: augments a decayed linear-attention state with a **bounded,
  non-parametric exact key-value cache** of high-surprise tokens, read via sharpened softmax
  attention. "Exact" here means verbatim KV storage, not exact algebraic composition of operators;
  no matrix powers.
- **Sequential Group Composition** — Marchetti, Kunin, Myers, Acosta, Miolane, **arXiv:2602.03655**
  (Feb 2026, confirmed real). Studies how a network's **weights** come to represent a finite
  group's irreducible representations and compose a sequence via Fourier statistics; proves
  perfect in-weights learning needs hidden width exponential in sequence length. This is the
  in-weights mechanics of group composition — explicitly the complement of NCR's in-context,
  written-operator setting, not a competitor to it.
- **DeltaProduct / negative-eigenvalue LRNNs / RWKV-7** (item 5, 7 above) — all achieve state
  tracking, but via **O(h) sequential recurrent rollout**, with operators baked into
  gradient-learned weight-update rules, not written explicitly in-context and read via O(log h)
  matrix power.
- Also checked and ruled out as direct hits: TensorLog/Neural-LP/DRUM (query-conditioned relation-
  path composition, symbolic/soft-logic, no matrix-power reads); RotatE (arXiv:1902.10197,
  knowledge-graph embeddings via complex rotation composition, static not in-context); Guu et al.
  (arXiv:1506.01094, compositional path queries, documents error-cascading under composition —
  cited by our own waterfall doc as the classic "composition degrades" precedent NCR claims to
  beat); MesaNet (arXiv:2506.05233, query-conditioned matrix-*function* solve, not a power); MAGNA
  (arXiv:2009.14332) and Higher-order/MEA-Linear-Attention (arXiv:2510.27258) — both take powers of
  an attention/mixing operator, but as a **soft geometric-mixture mechanism** over the sequence,
  not exact reads of a single in-context-written state matrix.
- Cayley-graph / group-word-problem ML search turned up only the CayleyPy pathfinding line
  (diffusion models / RL for finding generator sequences in Cayley graphs, e.g. Rubik's-cube-style
  problems) — a different problem (path-finding to an identity, not query-time depth-h read of an
  already-composed operator) and different method family (diffusion, not fast-weight matrices).

**Conclusion of the sweep:** the exact combination NCR claims — in-context-WRITTEN operators +
EXACT composition + O(log h) repeated-squaring READS — remains unclaimed in the literature as of
this search (2026-07-16). This corroborates, via an independently-run search pass, the conclusion
already on record in `matrix-thinking/NOVEL_ARCH_WATERFALL.md` §2–§5.

---

## Part 3 — The hard question: what exactly remains novel in NCR, given items 5 and 7?

Items 5 (Grazzi et al. negative eigenvalues / DeltaProduct) and 7 (RWKV-7) already show, in
published, peer-reviewed work, that matrix-valued linear-RNN state CAN escape TC⁰ and solve
provably-hard state-tracking (the S₅ word problem, NC¹-complete by Barrington). So "matrix-valued
fast-weight state can state-track" is **not** NCR's novelty — that door is already open and NCR
should stop implying otherwise in any framing. Being adversarial about the four candidates:

**(a) EXACTNESS of composition (not approximate expressivity) — thin on its own, but real.**
RWKV-7/DeltaProduct's state-tracking results are existence/expressivity theorems: *there exists* a
setting of weights (learned via gradient descent over many examples) under which the recurrence
computes the correct group product after h sequential steps. They do not claim, and their training
procedure does not enforce, that an arbitrary in-context-written set of operators composes with
zero error at arbitrary query-chosen depth from a **single training regime** generalizing to
depths beyond what's seen. FWM (item's closest neighbor) explicitly documents the opposite —
composition degrades with recursion depth (the brittleness Guu et al. also documented in 2015, and
which our own waterfall doc §2 flagged FWM's own authors as acknowledging). So "exact, depth-
generalizing composition of in-context-written operators" is a real, checkable, currently-unclaimed
empirical property — but by itself it is an *empirical result about one architecture at one scale*,
not a new complexity-class capability. Reviewers will correctly say "prove it's exact, not just
low-error," and will ask for the same depth-generalization stress test RWKV-7/DeltaProduct never
ran. This axis is defensible but not by itself a headline.

**(b) O(log h) query-time depth access via repeated squaring vs. O(h) recurrent rollout — this is
the sharpest, most defensible axis, and it is an algorithmic/complexity claim, not an expressivity
claim.** Items 5 and 7 both still pay O(h) sequential compute to reach depth h — RWKV-7 processes
the input token-by-token; the state at "depth h" only exists after h forward steps have physically
happened. NCR's claim is different in *kind*: given an already-written matrix of operators, reading
the depth-h composition costs O(log h) matrix multiplications via repeated squaring, independent of
how the write was distributed over time. This is a real, well-defined, and — per the sweep above —
unclaimed distinction: no paper found applies repeated-squaring to a fast-weight state for
query-time depth access. It is also the axis best matched to a literal circuit-depth argument (a
CoT/recurrent model doing h sequential steps is depth-Θ(h); a repeated-squaring reader is
depth-Θ(log h)), which is the kind of claim complexity-theory reviewers (the same audience that
cares about items 1–4) will find legible and check rigorously. **This is the axis to lead with.**

**(c) Operators WRITTEN in-context rather than learned in weights — real but the weakest of the
four as an isolated claim, because it's not new in kind, only in combination.** In-context weight
writing is exactly what FWM (2021), fast-weight programmers generally (item 6), and every linear-
attention/DeltaNet-family model already do — the state IS written from context by construction in
all of these. What's not present elsewhere is combining in-context writes with an exact,
log-depth-readable representation. So (c) is necessary scaffolding for (a)+(b), not an independent
novelty claim — do not present it as one; a reviewer who knows the FWP lineage will immediately
say "in-context fast weights are 5 years old."

**(d) A provable rank-necessity harness — orthogonal to the composition-depth story, real, and the
most rigorous single artifact NCR has, but it answers a different question than (a)–(c).** This is
the Task-D/rank-law lineage (already the subject of a separate, more thorough novelty check in
`research/task-d-novelty-july2026.md`, closest neighbor Nichani/Lee/Bietti's rank-m achievability
result, item 8 above), not a composition-depth result. It strengthens the overall NCR paper (proves
the state ISN'T a disguised vector, closing the "any d² vector can be reshaped to d×d" shortcut
this repo's own hard rules warn about) but should be scoped as a **supporting** mechanism-integrity
result, not folded into the "novel composition capability" headline — conflating the two claims in
one abstract invites a reviewer to attack the weaker one and discredit both.

**Verdict on the boundary, stated plainly:** the defensible novelty is narrow and specific — **(b)
is the load-bearing claim** (O(log h) exact query-time composition reads of in-context-written
operators vs. every published alternative's O(h) sequential rollout), with **(a) exactness** as its
necessary empirical companion (a repeated-squaring read is only interesting if it's exact, not
approximately-exact) and **(d) the rank-necessity harness** as a separate, real, but secondary
mechanism-integrity result. **(c) in-context writing is not novel by itself** and should never be
presented as though it were — every fast-weight paper since 2021 already does it. If NCR's paper
draft leads with "matrix-valued state enables state tracking transformers can't do," that claim is
**already published** (items 5, 7) and a reviewer citing Grazzi/Peng will correctly reject it. The
paper must lead with the O(log h)-vs-O(h) **query-time access complexity** distinction on top of
**exact** composition — that combination, per this searched sweep and the repo's independent §1–§5
sweep, is not yet claimed anywhere found.

---

## References (arXiv IDs, consolidated)

| # | Citation | arXiv | Venue |
|---|---|---|---|
| 1 | Merrill & Sabharwal, "The Parallelism Tradeoff" | 2207.00729 | TACL 2023 |
| 2 | Barrington, bounded-width branching programs / NC¹ | — (1986 STOC / 1989 JCSS) | JCSS 38(1), 1989 |
| 3 | Liu, Ash, Goel, Krishnamurthy, Zhang, "Transformers Learn Shortcuts to Automata" | 2210.10749 | ICLR 2023 (oral) |
| 4 | Merrill, Petty, Sabharwal, "The Illusion of State in State-Space Models" | 2404.08819 | ICML 2024 |
| 5a | Grazzi, Siems, Zela, Franke, Hutter, Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues" | 2411.12537 | ICLR 2025 |
| 5b | Siems, Carstensen, Zela, Hutter, Pontil, Grazzi, "DeltaProduct" | 2502.10297 | NeurIPS 2025 |
| 6a | Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" | — (PMLR v139) | ICML 2021 |
| 6b | Irie et al., "Going Beyond Linear Transformers with Recurrent Fast Weight Programmers" | 2106.06295 | NeurIPS 2021 |
| 7 | Peng et al., "RWKV-7 'Goose' with Expressive Dynamic State Evolution" | 2503.14456 | 2025 |
| 8 | Nichani, Lee, Bietti, "Understanding Factual Recall in Transformers via Associative Memories" | 2412.06538 | ICLR 2025 (spotlight) |
| N1 | Schlag, Munkhdalai, Schmidhuber, "Learning Associative Inference Using Fast Weight Memory" (FWM) | 2011.07831 | ICLR 2021 |
| N2 | "Log-Linear Attention" | 2506.04761 | ICLR 2026 |
| N3 | "A Hippocampus for Linear Attention" (HOLA) | 2607.02303 | 2026 (recent) |
| N4 | Marchetti, Kunin, Myers, Acosta, Miolane, "Sequential Group Composition" | 2602.03655 | 2026 |
| N5 | Guu, Miller, Liang, compositional path queries / composition error-cascading | 1506.01094 | 2015 |
| N6 | RotatE (knowledge-graph rotation composition) | 1902.10197 | ICLR 2019 |
| N7 | MAGNA (attention-power geometric mixture) | 2009.14332 | — |
| N8 | Higher-order / MEA Linear Attention | 2510.27258 | 2025 |
| N9 | MesaNet | 2506.05233 | 2025 |

---

## 2026-07-16 novelty-gate re-sweep (PI-directed re-verification, three sweeps + Opus adjudication)

**Context.** A PI-directed novelty re-verification gate ran three fresh
sweeps (by-mechanism, by-task, internal-archive) to discharge the GATE-2
novelty precondition of `matrix-thinking/NCR_REAL_LM_DESIGN.md` before any
Task-2/Axis-A GPU-h. The adjudication is recorded in that design's §N1; the
verified citations, anchor corrections, and the one UNVERIFIED flag are
consolidated here so a future agent does not re-search from zero.

**Headline outcome.** (a) The mechanism-novelty boundary HOLDS — no located
work combines in-context-written full-rank operators + exact composition +
query-time O(log h) repeated-squaring reads + orthogonalized writes; MuonSSM
is closest. (b) The S₅ collapse-vs-hold SEPARATION SHAPE is SCOOPED (an
established genre since 2022; **Yau 2506.10918 is a near-scoop on the
identical task**); the claim was accordingly restructured so the headline is
the exactness-by-construction guarantee + O(log h) access complexity, with
the S₅ separation demoted to disclosed-genre corroboration. (c) Internal
archive clear; the one omission (this design did not cite
`CAPABILITY_SEPARATION_DESIGN.md` §2.35's own S₅ verdict) is reconciled in
§N1 R3.

**New/updated VERIFIED citations (arXiv IDs + one-line characterizations, per
the sweep reports):**

| Citation | arXiv | Status | One-line characterization |
|---|---|---|---|
| Yau et al., "Sequential-Parallel Duality in Prefix Scannable Models" (Andreas lab) | 2506.10918 | VERIFIED | **NEAR-SCOOP** — S₅ "cups and balls", train len 4–18 / test to 180, T-PSM holds while transformer AND Mamba collapse; empirical/learned hold, NO exactness argument. |
| Li, Guo, Andreas, "(How) Do Language Models Track State?" | 2503.02854 | VERIFIED | S₃+S₅ in real LMs; "cutoff length" collapse; two LEARNED mechanisms (associative scan + parity-pruned) — the mechanistic-collapse half of our claim, already published. |
| Lee 2026 | 2606.07254 | VERIFIED-as-real; CHARACTERIZATION AMBIGUOUS | RESOLVED 2026-07-17 (kwall worker, direct arXiv abstract fetch): Jeonghoon Lee, "A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking" — the two sweep descriptions were compatible partial views of ONE paper (an eval-methodology falsifier demonstrated on S₃×S₃ state tracking, train len 8 → eval to 10⁶). Existence-cite is safe; full-body read still recommended before quoting specific numbers. |
| M²RNN | 2603.14360 | VERIFIED | Matrix-valued state (adjacent family), S₃ only, train 128 → 512, ≥99.5% OOD; fixed nonlinear recurrence, not in-context-written operators — active adjacent competition. |
| MuonSSM (Nguyen et al.), ICML 2026 Oral | 2606.30461 | VERIFIED (full-text) | Closest ortho-write prior art — orthogonalizes RANK-1 KV injections with a SINGLE quintic NS step for STABILITY, not full-rank d×d operators for composition-exactness. **Its "O(log L)" = Blelloch associative-scan TRAINING parallelism, NOT query-time O(log h) reads — pre-empt the conflation.** |
| Barrington, bounded-width branching programs | — (1986 STOC / 1989 JCSS) | VERIFIED | Classical ORIGIN of exact composition by repeated squaring/doubling; cite as the math's origin — the learned in-context instantiation is the new part. |

**Anchor corrections (landed this sweep):**
- **FWM (2011.07831):** "fixed hop count" is NOT confirmable from the
  abstract — soften to "recursive, gradient-trained, APPROXIMATE reads" in
  all memos/papers.
- **Log-Linear Attention (2506.04761):** now **PUBLISHED, ICLR 2026** —
  update the venue in every cite (Part 2 above and the References table list
  it as ICLR 2026 already; this confirms it).
- **MuonSSM Blelloch nuance:** its O(log) is associative-scan TRAINING
  parallelism (Mamba2/DeltaNet chunking family), not a query-time
  repeated-squaring read — record wherever the "O(log)" name is compared.
- **Atlas = arXiv:2505.23735** (Titans successor; there is no distinct 2026
  Atlas paper — do not cite a phantom later ID).

**UNVERIFIED flag (carry until closed):**
- **RWKV-7 (2503.14456):** the EXPRESSIVITY claim (single layer solves an S₅
  state-tracking problem; constant layers recognize any regular language) is
  VERIFIED (Part 1 item 7 above). The **empirical S₅ LENGTH-GENERALIZATION
  SPECIFICS are UNVERIFIED — the body fetch failed this session.** Do NOT
  cite RWKV-7 S₅ empirical numbers until a body refetch confirms them.

**Reviewer-bait / cite-and-distinguish (no group-word overlap, VERIFIED-as-real
per the sweeps):** Anil 2207.04901; Zhou 2402.09371 (addition, fragile);
RASP-L 2310.16028 (leverage FOR us — S₅ plausibly lacks a short RASP-L
program); Position Coupling 2405.20671 (a baseline variant, never applied to
S₅); 2402.09268 + 2509.09001 (O(log k) via transformer LAYERS); Echo
2605.06997 (approximate power-iterated filter).

**Adjudication verdict:** GATE-2 novelty condition DISCHARGED (see
`NCR_REAL_LM_DESIGN.md` §N1). The full tiered must-cite list, the exact
amended flagship claim, the data-pinned L_test extension, and the bridge-cell
n=3→n=5 seed amendment are recorded there.

## 2026-07-17 novelty-delta sweep — Cayley/skew-exp parametrized IN-CONTEXT operator writes (post-§10 fallback design support)

Delta on the 2026-07-16 base sweep; live-verified (WebFetch/pdftotext full-text reads for the two load-bearing citations). Raw agent report in session task archive.

**Q1 — the combination (a) in-context-written operator content + (b) Cayley/expm orthogonal parametrization + (c) persisted for O(log h) repeated-squaring reads: UNCLAIMED.** Closest mechanism-level neighbor: **Orthogonal Self-Attention, arXiv:2602.05996 (Feb 2026, VERIFIED abstract)** — in-context query-key content → skew-symmetric generator → matrix exponential → orthogonal attention matrix; but single-use per forward pass (token mixing), never persisted, no matrix-power read. Cite OSA for the parametrization facet AND MuonSSM (2606.30461) for the orthogonalized-fast-weight facet — complementary nearest neighbors. Also: EΔ-MHC-Geo (2605.06729, data-dependent Cayley on the DEPTH axis / hyper-connections, built on Deep Delta Learning 2601.00417) — the depth-wise mirror; RotRNN 2407.07239 (withdrawn; in-context-ness UNVERIFIED — do not cite as in-context); Path Development Network 2204.00740 (O(h) Lie-group composition, covered bucket); 2605.10970 + 2601.15313 = false positives ("orthogonality" = key separation, not parametrization). **Pre-submission flags: body-level re-reads of OSA + EΔ-MHC-Geo (both too recent for abstract-only characterization).**

**Q2 — projection-vs-parametrization trainability: ESTABLISHED THEORY; our §10 = confirming instance in a new application, NOT a discovery.** Frame accordingly in all papers. Evidence: (1) **Lezcano-Casado & Martínez-Rubio, ICML 2019, arXiv:1901.08428 (VERIFIED, FULL TEXT)** — Prop. 3.2 + Remark: the Cayley transform has a hard singularity (skew weights → ∞ when the optimum has eigenvalue −1; Helfrich-2018-style scaling just relocates it); their expm trivialization (EXPRNN) is proven free of it (expm's own singular set — eigenvalue gaps at nonzero multiples of 2πi — is distinct and empirically more benign). (2) **arXiv:2509.11983 (VERIFIED)** — "singular vectors associated with small singular values… instability in orthogonalization via Newton–Schulz" (Muon-optimizer context, forward-only — mechanism known, our differentiated-through application not studied). (3) **projUNN 2203.05483 (VERIFIED, FULL TEXT, App. G.3)** — hard projection differentiated-through accumulates drift; tangent-space variant stable. (4) Classical: polar Fréchet derivative ~1/(σ_i+σ_j) (Higham); Nakatsukasa–Higham NS backward-stability bounds.

**DESIGN RULING (load-bearing for NCR_ORTHO_FALLBACK_DESIGN.md): Cayley (I−W)(I+W)⁻¹ is NOT a safe fallback** — its proven −1-eigenvalue blow-up is plausible-to-certain under composition of permutation-like operators (any 180°-rotation-equivalent / order-2 composed state triggers it). **expm(W−Wᵀ) is the theoretically preferred parametrization** per 1901.08428. Any design retaining a Cayley arm must carry the singularity analysis + a mitigation or demotion-to-comparison-arm rationale.

**Q3 verifications:** 1901.08428 = ICML 2019 PMLR v97 (full-text read); Lezcano Casado "Trivializations…" = NeurIPS 2019, DISTINCT single-author paper (dynamic trivializations + expm gradient formula); geotorch/expRNN = real, maintained, all instances parametrize TRAINED weights (no in-context follow-up found).

**Differentiator sentence (draft, for papers):** "Unlike Orthogonal Self-Attention (2602.05996), which computes an in-context orthogonal matrix fresh each forward pass for immediate token-mixing, our construction writes the operator once from context, parametrizes it via the matrix exponential to guarantee exact orthogonality by construction — not by iterative projection, which we show (confirming Lezcano-Casado & Martínez-Rubio 2019's Cayley singularity and the known NS/polar near-singular instability, 2509.11983) is a training-destabilizing choice for this application — and persists it for O(log h) repeated-squaring reads at arbitrary query-chosen depth."
