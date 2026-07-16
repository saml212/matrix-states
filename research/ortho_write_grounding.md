# Literature Grounding: Orthogonal-Write Fix for Non-Normal Fast-Weight Operators

**Date:** 2026-07-16
**Context:** grounding for the NCR ortho-write fix — learned d×d write operators in a
fast-weight/matrix-state architecture drift non-normal (condition numbers 320–2950
observed), and far-depth matrix-power reads annihilate the weakest eigenmode. Fix:
constrain writes to (near-)orthogonal via a differentiable Newton–Schulz polar
iteration Q = Z(ZᵀZ)^(−1/2), coefficients (1.5, −0.5), spectral pre-scale, 40 iterations.
All claims below were checked live via web search on 2026-07-16, not recalled from
training data. Each item is tagged VERIFIED / CORRECTED / NOT-FOUND.

---

## 1. Unitary/orthogonal RNN line — VERIFIED

Shared motivation across all five papers, confirmed: constrain the recurrent
transition/weight matrix so its eigenvalues sit on (or near) the unit circle,
so repeated application across depth/time neither explodes nor vanishes the
signal — i.e., exactly the pathology class we diagnosed for write operators
(the difference being *which* matrix is constrained; see §4).

- **Arjovsky, Shah & Bengio (2016), "Unitary Evolution Recurrent Neural
  Networks."** arXiv:1511.06464, ICML 2016 (PMLR v48, pp. 1120–1128).
  Parametrizes the recurrence matrix as a product of structured unitary
  building blocks (reflections, Fourier transform, diagonal phase matrices)
  so eigenvalues have modulus exactly 1 by construction, avoiding vanishing/
  exploding gradients without gradient clipping.
- **Wisdom, Powers, Hershey, Le Roux & Atlas (2016), "Full-Capacity Unitary
  Recurrent Neural Networks."** arXiv:1611.00035, NeurIPS 2016. Shows
  Arjovsky et al.'s restricted parametrization cannot reach all unitary
  matrices; optimizes directly over the full unitary group via a
  Stiefel-manifold gradient step, with real performance gains from the
  extra capacity.
- **Helfrich, Willmott & Ye (2018), "Orthogonal Recurrent Neural Networks
  with Scaled Cayley Transform" (scoRNN).** arXiv:1707.09520, ICML 2018,
  pp. 1969–1978. Parametrizes a real orthogonal matrix via the Cayley
  transform of a learned skew-symmetric matrix, with a diagonal ±1 scaling
  to reach matrices with eigenvalue −1 (which the unscaled Cayley map
  cannot represent). Avoids complex arithmetic entirely.
- **Mhammedi, Hellicar, Rahman & Bailey (2017), "Efficient Orthogonal
  Parametrisation of Recurrent Neural Networks Using Householder
  Reflections."** arXiv:1612.00188, ICML 2017, pp. 2401–2409. Builds the
  orthogonal transition matrix as a bounded product of Householder
  reflections — directly the Cartan–Dieudonné decomposition later reused by
  DeltaProduct (§4).
- **Lezcano-Casado & Martínez-Rubio (2019), "Cheap Orthogonal Constraints in
  Neural Networks: A Simple Parametrization of the Orthogonal and Unitary
  Group" (expRNN).** arXiv:1901.08428, ICML 2019. Uses the matrix
  exponential map from Lie group theory (a "trivialization") to
  parametrize O(n)/U(n) cheaply; gradients grow only linearly in sequence
  length; more stable optimization than Cayley-based approaches.

**Takeaway for our fix:** this entire line targets the *recurrent transition
matrix* (the operator applied every timestep to the hidden state), not
in-context-written operators. It establishes the eigenvalue-preservation
motivation we're relying on, but is architecturally category (a) in the
novelty question (§4) — precedent for the *principle*, not the *mechanism*.

---

## 2. Muon optimizer / Newton–Schulz at scale — VERIFIED, one correction

- **Muon ("MomentUm Orthogonalized by Newton–schulz"), Keller Jordan et al.,
  2024.** CORRECTED: there is no formal arXiv paper for the original
  release — it originates as a blog post (kellerjordan.github.io/posts/muon/)
  and code release (github.com/KellerJordan/Muon), not a peer-reviewed arXiv
  preprint. It is nonetheless the de facto citation (`Jordan et al., 2024`)
  used by the arXiv literature that builds on it — e.g. MuonSSM (§4) cites it
  this way. Confirmed mechanism: Muon replaces each 2D parameter's SGD-
  momentum update with its nearest orthogonal matrix, approximated via a
  Newton–Schulz iteration (avoiding an explicit SVD) run for a handful of
  steps (5, in the standard release) using quintic coefficients
  **(3.4445, −4.7750, 2.0315)** — tuned to inflate small singular values fast
  — rather than the classical cubic (1.5, −0.5) iteration our fix uses.
  Framed theoretically as steepest descent in spectral norm / modular duality.
- **Validated at large scale — VERIFIED.** Moonshot AI's Kimi K2 technical
  report: **"Kimi K2: Open Agentic Intelligence,"** Kimi Team, arXiv:2507.20534.
  Confirms Muon (combined with a stabilizing mechanism, QK-Clip, into
  "MuonClip") was used to pretrain a 1.04T-parameter (32B-active) MoE model
  on 15.5T tokens with zero reported training instability — the first
  published trillion-parameter-scale, non-AdamW, end-to-end training recipe.
  This is solid cost/stability-at-scale precedent for Newton–Schulz-based
  orthogonalization machinery in general, though note it is applied to
  optimizer *updates* (category (b) in the novelty question), not to
  in-context fast-weight writes.

---

## 3. Polar decomposition numerics — VERIFIED, with the precise origin of our exact iteration

- **Björck & Bowie (1971), "An Iterative Algorithm for Computing the Best
  Estimate of an Orthogonal Matrix."** SIAM J. Numer. Anal. 8(2), 358–364.
  This is the earliest and most precise match for our exact update rule: the
  canonical Newton–Schulz orthogonalization iteration
  X_{k+1} = X_k(3I − X_kᵀX_k)/2 = 1.5·X_k − 0.5·X_k(X_kᵀX_k) — i.e. the
  (1.5, −0.5) coefficients used in our fix are literally the Björck–Bowie
  form, not an arbitrary choice. (Kovarik, 1970, "Some Iterative Methods for
  Improving Orthonormality," is the other co-earliest reference, per the
  same search trail — cite alongside if a second historical anchor is
  wanted.)
- **Higham (1986), "Computing the Polar Decomposition—with Applications."**
  SIAM J. Sci. Stat. Comput. 7(4), 1160–1174. This is the standard anchor
  for the general theory: a quadratically-convergent scaled Newton method
  for the polar decomposition, with acceleration/scaling parameters derived
  from estimates of the extreme singular values (or Frobenius-norm
  estimates) of the input matrix — this is the literature source for our
  **spectral pre-scaling** step. Confirms scaling trades convergence speed
  for a backward-error that scales with the condition number κ₂(A), and that
  both Higham-scaling and Frobenius-norm scaling remain backward stable in
  practice.
- Follow-on numerical-analysis literature (Byers & Xu 2008, "A New Scaling
  for Newton's Iteration for the Polar Decomposition and its Backward
  Stability," SIAM J. Matrix Anal. Appl.) refines the stability conditions
  further; not required as a primary citation but available as a stability
  deep-cut if reviewers push on numerics.

**Verdict:** the (1.5, −0.5) / 40-iteration design is well-grounded — it is
the original, well-understood cubic Newton–Schulz polar iteration (Björck &
Bowie 1971), not the accelerated quintic Muon variant, run to near-machine-
precision convergence rather than Muon/MuonSSM's 1–5-step "good enough"
regime. That is a legitimate, literature-consistent design choice for a
different goal (near-exact orthogonality for composition, not just
directional conditioning of an update).

---

## 4. NOVELTY CHECK — the important one

**Verdict: NOT clean, unclaimed territory — a closely related paper exists,
posted roughly two weeks before this check, doing the general MOVE we
thought might be novel. Our specific construction and motivation still
appear distinct and unpublished, but this must be cited and the
differentiation made explicit in any writeup. Real scoop risk: it's an
ICML 2026 Oral.**

### The closest prior art found: MuonSSM

**Nguyen, Vo, Vo, Nguyen & Pham, "MuonSSM: Orthogonalizing State Space
Models for Sequence Modeling."** arXiv:2606.30461, posted 2026-06-29,
**ICML 2026 Oral** (PMLR 306). Fetched and read directly (not just
abstract).

What it does: frames Mamba / DeltaNet / Gated DeltaNet / LongHorn as special
cases of one general associative-memory update
S_t = S_{t-1}(α_t(I_m − β_tη k_tk_tᵀ)) + β_tv_tk_tᵀ, then explicitly states
its core move is to condition **"the geometry of memory updates rather than
the recurrent transition matrix"** — i.e., the same category-(a)-vs-writes
distinction the task asked us to search for, and MuonSSM lands on the
writes side. It applies a **single-iteration** Newton–Schulz transform,
using Muon's exact quintic coefficients (3.4445, −4.7750, 2.0315), to each
**rank-1** input injection X_t = τβ_tv_tk_tᵀ before it's accumulated into a
momentum-augmented state. They report an 18× condition-number reduction
(κ≈2.2×10⁶ → 1.2×10⁵) on the *recurrent state* spectrum, gains across
language/vision/time-series benchmarks, and better length generalization on
needle-in-a-haystack retrieval. They cite Arjovsky 2016, Vorontsov 2017, and
Helfrich 2018 — the same orthogonal-RNN lineage as §1.

**Why it's close but not the same claim.** Three material differences,
confirmed by reading the paper's methodology (Definition 2.1, Table 1,
Fig. 2) rather than the abstract alone:

1. **Rank-1 magnitude-conditioning, not full-rank orthogonality.** Their
   write X_t = v_tk_tᵀ is rank-1 by construction (the standard DeltaNet-
   family KV outer-product). A rank-1 matrix in dimension d>1 *cannot*
   satisfy QᵀQ=I (it has d−1 zero singular values) — so "orthogonalizing"
   it is really pushing its single nonzero singular value toward the fixed
   point of the NS polynomial (≈1), a spectral-magnitude normalization, not
   a genuine rotation/reflection operator. Our fix targets full (near-)
   square-rank d×d write operators Z, driving them to genuine QᵀQ≈I via
   40-iteration convergence — a materially different object (an actual
   element of O(d), not a rescaled rank-1 outer product).
2. **Different motivation and diagnostic.** MuonSSM's target is general
   long-horizon gradient propagation / memory stability across LM, vision,
   and time-series tasks; its condition-number analysis (Fig. 2) is on the
   *accumulated recurrent state-transition matrix*, not on individual write
   operators. It does not discuss, test, or cite compositional-depth
   generalization, exact multi-hop composition via repeated matrix powers,
   or state-tracking benchmarks — the specific failure mode (far-depth
   matrix-power reads annihilating the weakest eigenmode of a *learned
   write operator*) that motivates our fix is outside their scope entirely.
3. **Different NS regime.** Single iteration, Muon's quintic coefficients,
   "lightweight" by design (their own ablation, Fig. 4, shows 5 iterations
   *underperforms* 1 iteration on their task — they explicitly do NOT want
   near-exact orthogonality). Ours: classical cubic Björck–Bowie
   coefficients, 40 iterations, deliberately targeting near-machine-
   precision Q — because exact composability (Q₁Q₂ being itself orthogonal,
   eigenvalues staying exactly on the unit circle under repeated powers) is
   the load-bearing property for our use case, not just "less spectral
   collapse than baseline."

**Other adjacent work, checked and distinguished:**

- **DeltaNet (Yang et al.), Gated DeltaNet, LongHorn, DeltaProduct
  (arXiv:2502.10297)** — all constrain the recurrent *transition* matrix
  (I − βkkᵀ, a Householder reflection at β=2; DeltaProduct chains several
  for any orthogonal matrix via Cartan–Dieudonné). This is category (a),
  explicitly excluded by the task framing — the transition operator applied
  *to the old state*, not the new content being written in.
- **RWKV-7 "Goose"** (arXiv:2503.14456) — transition matrix is a scaled
  Householder-type approximation; same category (a).
- **"Deep Delta Learning"** (Zhang, Liu, Wang, Gu; arXiv:2601.00417,
  posted 2026-01-01) — generalizes the DeltaNet Householder operator with a
  learnable data-dependent gate β(X) and direction k(X), applied along the
  *depth* dimension of residual networks. Still a transition/gating
  operator (category a), not a written associative-memory content operator.
- **Schlag, Irie & Schmidhuber (2021), "Linear Transformers Are Secretly
  Fast Weight Programmers."** arXiv:2102.11174, ICML 2021. Establishes
  the FWP framing and notes interference is minimized when *keys* are
  orthogonal — but this constrains the addressing keys, not the written
  value/operator content. Distinct axis from ours.
- **Dangovski et al. (2018), "Rotational Unit of Memory" (RUM).**
  arXiv:1710.09537, ICLR 2018 workshop / TACL 2019. Closest in *type*: RUM's
  associative-memory operation is strictly an orthogonal (rotational)
  matrix, explicitly motivated as unifying unitary learning with
  associative memory, and gets SOTA-for-its-time on the Associative Recall
  task. But it's a hand-composed per-timestep rotation inside a classic RNN
  cell (pre-dates the fast-weight/DeltaNet framing by ~3 years), not a
  Newton–Schulz-projected, gradient-trained fast-weight operator subject to
  matrix-power composition reads, and it is not diagnosed via or fixed for
  a condition-number-explosion failure mode.
- **Plate's Holographic Reduced Representations (HRR/VSA line)** — uses
  unitary (unit-modulus Fourier-domain) vectors as binding operators
  precisely so that unbinding (circular correlation) is an exact/near-exact
  inverse; Generalized HRR extends this to stacks of unitary matrices. Same
  underlying motivation (invertibility/exact composition needs
  orthogonality) but this is a fixed/hand-designed vector-symbolic
  encoding scheme, not a learned fast-weight operator with a differentiable
  Newton–Schulz projection step.
- **"Attention Is Not Retention"** (Zahn, Beton, Chana; arXiv:2601.15313)
  — despite the promising title, this is NOT about constraining write
  operators; it argues semantic embeddings *can't* be made orthogonal
  because training clusters related concepts, and proposes a discrete
  symbolic ("Knowledge Objects") fix instead. Read and ruled out.

### Honest verdict

The general strategic move — "Newton–Schulz-orthogonalize the fast-weight
*write*, as distinct from the recurrent transition matrix or the optimizer
update" — is **no longer unclaimed**: MuonSSM (ICML 2026 Oral, arXiv posted
2026-06-29, ~2 weeks before this check) explicitly frames and executes
exactly that distinction, and cites the same orthogonal-RNN lineage we're
drawing on. Anyone reviewing our fix will very likely know this paper or be
pointed to it.

What appears to remain unclaimed, based on an aggressive search (no hits
found despite many query variants: "orthogonal fast weights," "unitary
associative memory," "norm-preserving fast weight programmers," "orthogonal
key-value updates," Householder/DeltaNet/RWKV-7 variants, "compositional
generalization matrix power orthogonal operator," condition-number/
eigenmode-annihilation framing for write operators specifically): a
Newton–Schulz-orthogonalized fast-weight write that (a) targets full-rank
d×d operators (not rank-1 KV outer products), (b) is driven to near-exact
orthogonality (not single-step magnitude conditioning) specifically so that
repeated composition (matrix powers) exactly preserves eigenstructure, and
(c) is diagnosed and motivated via write-operator condition-number
explosion causing far-depth compositional-read failure, rather than general
long-horizon SSM gradient/memory stability. This is a real but narrower
claim than "we invented orthogonalizing fast-weight writes" — it must be
positioned as a distinct mechanism/target relative to MuonSSM, not as
prior-unexplored territory. Recommend citing MuonSSM explicitly as the
closest related work in any paper/report that comes out of this fix.

---

## 5. Non-normality → transient growth / pseudospectra — VERIFIED

**Trefethen & Embree (2005), "Spectra and Pseudospectra: The Behavior of
Nonnormal Matrices and Operators."** Princeton University Press. Confirmed
as the standard, correct anchor for our diagnosis framing. Key facts
verified: eigenvalues alone only describe asymptotic (large-time) behavior
and give no information about transient growth, which can occur on a much
shorter timescale — precisely the failure mode of interest (far-depth reads
annihilating weak eigenmodes is a transient/finite-power phenomenon, not
necessarily visible from the eigenvalue spectrum alone). For non-normal
matrices/operators, ε-pseudospectral level curves extend O(1) distances
from the eigenvalues; for normal matrices they form tight O(ε) balls around
each eigenvalue — this is the rigorous reason condition number (not
spectral radius) is the right diagnostic for a non-normal write operator,
and directly supports citing pseudospectral theory rather than eigenvalue
analysis alone when reporting the 320–2950 condition-number range.

---

## Summary table

| # | Claim | Status |
|---|---|---|
| 1 | Orthogonal/unitary RNN line (Arjovsky, Wisdom, Helfrich, Mhammedi, Lezcano-Casado) | VERIFIED, all 5 arXiv IDs confirmed |
| 2a | Muon uses Newton–Schulz to orthogonalize optimizer updates | VERIFIED (blog/code, not a formal arXiv paper — CORRECTED) |
| 2b | Muon validated at large scale (Kimi K2) | VERIFIED, arXiv:2507.20534 |
| 3 | Newton–Schulz polar convergence + spectral pre-scaling, Higham as anchor | VERIFIED; exact (1.5,−0.5) coefficients trace to Björck & Bowie 1971, not Higham directly — CORRECTED attribution |
| 4 | Novelty of orthogonality-constrained fast-weight WRITES | **NOT clean — MuonSSM (arXiv:2606.30461, ICML 2026 Oral) does the same general move**; our full-rank/near-exact/compositional-read framing appears distinct but must be positioned against it |
| 5 | Trefethen & Embree pseudospectra as the frame for non-normal condition-number blowup | VERIFIED |
