# DeltaNet Causal-Rank Design — Chapter 2 Extension to a Production Fast-Weight Architecture

**Drafted 2026-07-01, before any code changes.** Status: design only, per
instruction — no model/training code is written here. This is the natural
third leg of the Chapter 2 "does the gradient see rank" lineage:

| Setting | Result |
|---|---|
| Bolt-on matrix-CODI (ICML MI-workshop paper, accepted) | **No** — rank-blind on ProsQA |
| Matrix-native from scratch (Task D confirmed, Task E confirmed with caveats) | **Yes** — SGD recruits provably-necessary rank when a task forces it, and (with the §9 subspace-restriction caveat) the recruited operator composes |
| Mature, hardware-optimized production fast-weight architecture (this doc) | **?** |

Novelty check (2026-07-01) confirmed this exact combination — a provable
rank≥K lower bound, causal train-time rank-forcing, and held-out-depth
compositional transfer, applied to the DeltaNet family — is unoccupied. This
document designs that experiment.

> **CHANGELOG — 2026-07-01, revision 2 (post-attack-round).** An independent
> adversarial review of revision 1 returned BUILD WITH FIXES (2 FATAL-scoped,
> 6 MAJOR, several MINOR). All findings addressed in this revision:
> **F5 (FATAL — sat inside the primary causal gate)** — revision 1's
> chunk-boundary truncation mechanism was confounded with `fla-org`'s kernel
> chunk size (options {16,32,64}, default 64): at the primary cells K=16/32
> at d=64, the entire BIND phase fits inside the first chunk, so the
> truncation would have fired **zero** times before the read for 2 of 3
> primary cells — a spurious K-dependent pattern ("rank matters only at
> K=64") indistinguishable from a real threshold. Replaced with the
> **two-kernel-call split** (§3.5): BIND-only pass with state extraction →
> truncate `S_T` once → decode (external readout / QUERY pass seeded via
> `initial_state`); plus a mandatory post-truncation direct-SVD rank
> assertion with exact-threshold teeth (§3.5, C15).
> **F12 (FATAL-scoped, conv-enabled secondary check)** — `conv_size=4` lets
> QUERY positions within 3 tokens of the phase boundary build their k/v
> from raw BIND-phase embeddings; the one-sided blank-out cannot catch this
> backward-reaching leak. Fixed: ≥conv_size−1 buffer tokens between items
> and at the boundary + a two-sided blank-out (§4.3, C14).
> **F3 (MAJOR)** — the bound's premise restated as linear independence of
> the recovered images `{S_T k_j}` (survives any nonzero learned β_j);
> cosine's blindness to β-driven magnitude decay under composition now
> stated explicitly, with per-hop prediction norms logged (§4.1, §6.3).
> **F6 (MAJOR)** — instability framing rewritten for the F5 fix: ~1
> truncation per forward pass (Task D/E's exact validated surface), not
> T/C per sequence; Wave −1 carries the C15 SVD rank check (§3.5, §6.4).
> **F7 (MAJOR)** — M²RNN positioning rewritten around their flagship S₃
> permutation-composition experiment and their own vector-GRU-matches
> result (§2 item 5).
> **F11 (MAJOR)** — §8's strongest claim tier now carries the H=1 qualifier
> explicitly; an H∈{2,4} per-head-bound probe is pre-registered in the
> Reserve wave (§6.4, §8).
> **F13 (MAJOR)** — pre-committed Wave −1 decision rule (slowdown/skip-rate
> thresholds + fallback actions) replaces pricing ~100 GPU-h downstream of
> an unmeasured slowdown (§6.4).
> **F17 (MAJOR)** — operational-harness build requirements added as a
> blocking section (§6.5): validity-checked resume, sentinels, bounded
> gated waves, tmux+supervisor.
> **MINOR** — F1 (both rank-1-update non-membership conditions stated,
> §3.4), F2 (orthonormality-premise wording clarified, §3.6), F4
> (`‖K_effᵀK_eff − I‖_F` Gram diagnostic added, §6.3), F9 (citation anchors
> corrected: CLAUDE.md vs `TASK_D_PREREGISTRATION.md` §8; §10 list items,
> not nonexistent subsections), F10 (h=7 depth-probe numbers restated
> precisely, §0), F14 (skip counts reported as per-step rates), F15
> (explicit pre-Wave-−1 `fla-org` API verification checkpoint, §6.4).

> **CHANGELOG — 2026-07-01, revision 2.1 (round-2 verify: FIX FIRST on
> three doc-edit-sized findings; verifier pre-cleared these fixes — CLEAN
> TO BUILD once applied, no further review round).**
> **NEW-1 (blocking, F15 spec)** — the F15 checkpoint now includes a
> finite-difference **gradcheck** (small `d`, fp64) through BOTH kernel
> hooks — the backward of `output_final_state` into the recurrence, and
> the backward through `initial_state` seeding — with C15's
> run-to-completion negative-test discipline. §3.5's "gradients flow back
> through the truncation into the recurrence" is an assumption about the
> kernel's autograd wiring until this passes; F15 previously verified the
> forward round-trip only (§6.4).
> **NEW-2 (buffer/split interaction)** — buffer-token embeddings are
> **pinned to exact zero vectors**: under the two-kernel split, the
> QUERY-only call's conv starts from zero padding, so nonzero buffers
> would make single-call and two-call outputs differ at the boundary on
> the conv path — and F15's round-trip check would fail mysteriously.
> Zero-pinning makes zero padding and buffers equivalent by construction;
> C14 and the split become mutually reinforcing (§4.3; F15's round-trip
> equality is specified *given* zero-embedding buffers, §6.4).
> **NEW-3 (β-mask spec)** — §5.2's mask wording ("1 on BIND positions,
> hard 0 from the first QUERY token onward") predated C14's buffer
> tokens: inter-item buffers fell in neither region and would have
> written junk into the state — consuming rank budget under force-rank, a
> spurious collapse mechanism. Rewritten: mask = 1 on BIND **item**
> positions ONLY; hard β = 0 on ALL other positions, including every
> buffer token (§5.2, C9, §4.1).

**Standing house rule this design does NOT violate** (`CLAUDE.md`: *"DeltaNet
rank-1 updates are for recurrent models, not iterative attention. Don't
conflate."*): this design does not bolt DeltaNet's delta rule onto
matrix-thinking's iterative-refinement layer (the thing that rule forbids —
see `research/waterfall-cheap-ops-april2026.md`). It tests DeltaNet **as its
own recurrent architecture**, on its own terms, with Chapter 2's methodology
transplanted onto it. The two lines of work stay separate.

---

## 0. Reading list this design builds on (context, not repeated here)

- `matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md` — the provable
  rank≥K bound, the exact-continuous-recovery readout constraint (Nichani,
  Lee & Bietti, ICLR 2025, arXiv:2412.06538 — argmax/nearest-neighbor
  decoding breaks the bound), the hard single-state bottleneck (P=1), C1
  (train-time `force_rank_k`) as the *primary* causal test vs. post-hoc SVD
  (uninformative in CODI).
- `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md` (Task E) — the
  compositional-transfer methodology: literal iterated matrix-power readout
  `pred(a,h) = Zʰ·key_a`, single-Hamiltonian-K-cycle permutation graph
  (closes the periodicity-collapse trap), the idealized-`Z` classical
  reference, eigenstructure fidelity (C8) as distinct from rank magnitude.
- `matrix-thinking/chapter2/TASK_E_FINDINGS.md` — the **depth-amplification
  finding**: raw iteration count is a sharper rank test than single-hop
  cosine tolerance (the converged rank-(K−1) operator still holds
  `recovered_frac@0.9 ≈ 0.88` at h=7, with mean cosine ≈ 0.92 above the
  bar, then collapses to 0.06 at h=21 — F10 precision); and the **subspace-restriction lesson** (§4,
  §9): whole-matrix effective rank of a converged multi-hop solution is
  polluted by an unconstrained, full-rank orthogonal complement the loss
  never touches — the *entity-subspace-restricted* rank is what actually
  tracks K. This lesson is pulled forward pre-emptively into this design
  (§3.6) rather than re-discovered the hard way.
- `matrix-thinking/chapter2/STAGE0_DESIGN.md` §12 — the budget-artifact
  lesson: **late, seed-stochastic phase transitions are the norm in this
  architecture family, not the exception.** At d=32, all 6/6 baseline seeds
  were flat-or-barely-moving at the exact step (8,000) the original sweep
  stopped at, then transitioned somewhere in a 6,000–16,000-step window. A
  fixed, un-extended step budget systematically misreads "hasn't
  transitioned yet" as "doesn't train." §6.4 below budgets for this from the
  start.

---

## 1. The sharpened question

> **H_DN (primary).** A DeltaNet-family fast-weight layer, trained end-to-end
> under Task D/E's grammar streamed through its **own native recurrence** (no
> attention-based set encoder), with the standard delta rule as its only
> state-update mechanism, (a) develops effective rank of its recurrent state
> `S_T`, *restricted to the task's entity subspace* (§3.6's pre-registered
> metric choice), that tracks the binding count `K`; (b) its per-item
> recovery accuracy collapses under a train-time rank-`k` forcing constraint
> for `k < K` and reaches ceiling for `k ≥ K` (the M3-equivalent causal
> test, weighted as this design's primary gate, exactly as in Task D); and
> (c) the resulting rank-`K` state composes correctly under literal
> repeated self-application (`S_Tʰ`) at held-out hop depths `h > H_train`
> (the M3_E-equivalent test, exactly as in Task E).

- **CONFIRM** (a+b, with c as a secondary, not-blocking extension) ⇒ the
  matrix-native rank-recruitment result is not an artifact of Task D/E's
  bespoke attention-based encoder — it transfers to a real, hardware-
  optimized, already-deployed (Qwen3.5, Qwen3-Next use Gated DeltaNet;
  `research/validated-cheap-matrix-ops.md`) fast-weight family. This closes
  the third cell of the table above and is a materially stronger claim than
  Task D/E alone: "SGD uses rank when forced to" generalizes across
  architecture *families*, not just within one bespoke design.
- **FALSIFY** (rank-forcing shows no step, or the unconstrained model's
  entity-subspace rank does not track K despite high accuracy) ⇒ a decisive,
  publishable negative that sharply qualifies the growing
  rank-of-linear-attention-states literature (Nazari & Rusch,
  arXiv:2602.04852; Sun et al., arXiv:2602.02195): their **descriptive**
  `rank(S_t) ≤ t` upper-bound measurements on pretrained LLMs would then be
  shown to *not* imply SGD converges toward a genuinely rank-disciplined
  solution even when one is forced to exist and is exactly reachable by the
  delta rule's own dynamics (§3.6) — a real architectural boundary
  condition, not a hand-wave.

Both outcomes are decisive and publishable, matching Task D/E's bar.

---

## 2. Positioning against prior art (must-cite, distinguishing)

In priority order, from the 2026-07-01 novelty check:

1. **Nazari & Rusch, arXiv:2602.04852; Sun et al., arXiv:2602.02195 (both
   Feb 2026).** Measure fast-weight state rank **descriptively** on
   pretrained models — same effective-rank instrument, uncontrolled K, no
   necessity theorem, no causal train-time intervention. We are the mirror
   image, exactly as Task D already distinguished itself from these two
   papers in the from-scratch setting (`TASK_D_PREREGISTRATION.md` §10
   item 2):
   controlled K, a provable **lower** bound, a **causal** rank-k ablation —
   now run on the actual architecture family those two papers measure.
2. **Barnfield et al., arXiv:2605.05189.** Proves capacity thresholds for
   **static** rank-constrained associative memories (a fixed matrix, no
   recurrence, no training dynamics). Our delta from this paper must live
   entirely in the **recurrent / causal / compositional** part — the
   *training-time* force-rank ablation on a *time-varying, streamed* state,
   plus the held-out-hop composition test. Distinguished explicitly in §3;
   used as a possible **analytic capacity reference** (a role parallel to
   Task E's classical `Z_ideal`, §3.6) rather than the object under test.
3. **DeltaProduct, arXiv:2502.10297.** Its "rank" is the **transition
   matrix** `A_t = I − β_t k_t k_tᵀ` (a d×d operator applied *to* the state
   at each step, whose own rank-deficiency-from-identity is what governs
   state-tracking expressivity), composed as a product of `n_h` generalized
   Householder reflections per token. This is a **different object** from
   the quantity this design measures — the **state's own rank**,
   `rank(S_t)`, the capacity of what has been *written* into memory so far.
   Conflating the two is the single most likely reviewer confusion; §6.2
   (control C11) pins the primary gate to **vanilla, single-Householder-per-
   token DeltaNet** (`n_h = 1`) specifically to keep these separate.
   DeltaProduct's `n_h > 1` variant is explicitly out of scope for the gate
   (a natural, flagged follow-on: does state-rank capacity trade against
   transition-rank expressivity?).
4. **Based / Arora et al., arXiv:2402.18668, Thm F.1.** Bounds total state
   **size** (bits/dimensions) via communication complexity, architecture-
   and decoding-scheme-agnostic. Justification for **rank, not size**, being
   the binding resource here (this must be stated explicitly, or a reviewer
   will ask "isn't state size just d² regardless of rank, so why does rank
   matter"): our readout is pinned to the **linear unbind**
   `pred = S_T · key_j` (never an arbitrary decode over the full d² entries,
   §4.1). Under a *linear* readout, the classical fact (Kohonen 1972/73;
   Anderson 1972, already the basis of Task D §3) is that exact recovery of
   `K` independent bindings needs `rank ≥ K` — a strictly sharper,
   mechanism-specific claim than "the state has `d²` bits somewhere." Thm
   F.1's communication-complexity bound is a looser, decoding-agnostic
   ceiling that cannot, by construction, isolate rank as the causal
   quantity; our train-time force-rank-k ablation manipulates rank while
   holding `d²` (ambient state size) fixed, which a size-only argument
   cannot do. Same defense Task D already ran against Nichani et al.'s
   argmax-decoding capacity result (`TASK_D_PREREGISTRATION.md` §10
   item 1), applied to a size-based rather than decoding-based attack.
5. **M²RNN, arXiv:2603.14360 (Mishra, Tan, Stoica, Gonzalez, Dao; Mar/May
   2026 — verify all specifics before external citation; abstract-level
   sources only so far).** The closest prior art in this list, and — per
   the attack round (F7) — the one whose flagship experiment must be named
   head-on rather than gestured at: **their headline result IS
   permutation-group (S₃) composition state-tracking on a matrix-valued
   recurrent state, trained at sequence length 128 and evaluated
   near-perfectly at 512** — the same compose-a-permutation-beyond-the-
   training-horizon *shape* as this design's M3_E-equivalent, on the same
   kind of state. The delta, led with what their own results show:
   (i) **their own comparison table has a plain vector-state GRU matching
   the matrix model on that very task** — so their result does not show
   matrix *rank* (or matrix structure at all) is load-bearing, only that
   some sufficiently expressive non-linear recurrence tracks S₃;
   (ii) their Thm 1 is a **representational** (existence/expressivity)
   result, not a rank-**necessity** bound — nothing in it says a solution
   *must* use rank ≥ K, and nothing is measured about the rank SGD's
   solution actually uses; (iii) they run **no causal rank intervention**
   of any kind — no train-time rank forcing, no truncation ablation. This
   design is the mirror image on all three: a task where rank ≥ K is
   provable under the pinned readout, a train-time causal force-rank arm,
   and a controlled-K capacity framing. Cite as both prior art
   (matrix-valued recurrent states length-generalize on group-composition
   tasks; same TC⁰ motivation as Merrill, Petty & Sabharwal,
   arXiv:2404.08819) and as the sharpest illustration of the exact gap
   this experiment fills (no one has shown the rank is *causally used*).
6. **Schlag, Irie & Schmidhuber, arXiv:2102.11174** (Task D §10 item 3,
   unchanged)
   — origin of "linear-attention state as associative memory," now the
   direct architectural object under test rather than a motivating citation.
7. **Nichani, Lee & Bietti, arXiv:2412.06538** (Task D §10 item 1,
   unchanged) —
   still the reason the readout must be exact-continuous, never argmax
   (§4.1).
8. **`research/validated-cheap-matrix-ops.md`** (internal, 2026-04) — already
   derived the delta rule's Householder-reflection structure
   (`A_t = I − β_t k_t k_tᵀ`) and flagged DeltaNet/Gated DeltaNet as the
   best-validated cheap-matrix-op candidate for a *different* purpose
   (compressing matrix-thinking's own thinking-layer update). Reused here
   only for its already-verified algebra (§3.1); the application is
   unrelated and does not reopen the "don't conflate" rule above.

---

## 3. Crux 1 — forcing rank on a time-varying state `S_t`

### 3.1 The delta rule, exactly

For scalar gate `β_t ∈ [0,1]`, key `k_t ∈ ℝ^d` (unit-normalized), value
`v_t ∈ ℝ^d`, state `S_t ∈ ℝ^{d×d}` (square, `d_k = d_v = d`, matching Task
D/E's `Z ∈ ℝ^{d×d}`):

```
S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ
    = S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

`A_t = I − β_t k_t k_tᵀ` is a generalized Householder reflection (rank-1
perturbation of the identity) — already verified in
`research/validated-cheap-matrix-ops.md`.

### 3.2 Option A — truncate the full state every step

`S_t ← truncate_to_rank(S_t, k)` (reusing `rank_utils.py::truncate_to_rank`,
`eigh(S_tS_tᵀ)`, NaN-skip hygiene) after every recurrence step.

- **Cost:** `T` truncations per sequence, each `O(d³)` (full eigendecomposition
  of a `d×d` matrix) — `T×` the cost Task D/E paid *once per forward pass*.
- **Differentiability:** Task E's own M4_E force-rank runs at a **single**
  application per forward pass already showed eigh-backward numerical
  instability (`n_skipped_steps` 3–10 per run, fr=8/9 "all dead",
  `TASK_E_FINDINGS.md` §2). Doing this `T` times per sequence, every
  training step, multiplies that instability surface roughly `T`-fold — a
  serious, not cosmetic, risk.
- **Kernel cost:** inserting a full-`d×d` eigh into every recurrence step
  destroys the chunked-parallel scan that is DeltaNet's entire practical
  reason for being fast (Yang et al.'s WY-representation chunking, the
  basis of `fla-org`'s kernels) — forces a fully sequential Python/CUDA loop.
  This is the real reason **not** to pick this option naively.

### 3.3 Option B — truncate at read time only

Let `S_t` evolve unconstrained through the full sequence (fast kernel,
untouched); truncate only `S_T` (or wherever a query is asked) at eval/decode
time.

- **Cost:** negligible — one truncation per query, and the recurrence itself
  stays on the fast chunked kernel throughout training.
- **Why it cannot be the primary test:** this is exactly the post-hoc-SVD
  pattern Task D's own C1 rules out by name (*"post-hoc SVD gave
  uninformative flat curves in CODI; conflates instrument insensitive with
  mechanism absent,"* `TASK_D_PREREGISTRATION.md` §5). It measures whether a
  freely-trained state *happens* to be well-approximated by a low-rank
  projection, not whether the gradient was ever caused to *use* rank-k
  capacity. Demoted to the same role Task D gave it: an M2-equivalent
  **corroborating** instrument, never the causal claim.

### 3.4 Option C, literal reading — is a fixed-rank factorization closed under the delta rule?

The crux's proposed construction: parameterize `S_t = U_t V_tᵀ`,
`U_t, V_t ∈ ℝ^{d×k}`, and ask whether the delta rule stays inside this form
"by construction, no truncation needed." **Checking the algebra directly —
it does not.**

```
S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ
    = U_{t-1} V_{t-1}ᵀ  +  β_t (v_t − U_{t-1}(V_{t-1}ᵀ k_t)) k_tᵀ
    = U_{t-1} V_{t-1}ᵀ  +  δ_t k_tᵀ            where δ_t = β_t(v_t − S_{t-1}k_t)
```

`U_{t-1}V_{t-1}ᵀ` has rank `≤ k`; `δ_t k_tᵀ` is a fresh rank-1 term. The
precise condition (F1 — revision 1 stated only the column-side half; the
conclusion stands, the derivation was incomplete):
`rank(S + δkᵀ) = rank(S) + 1` **iff both** `δ ∉ col(S)` **and**
`k ∉ row(S)`. Here that means the rank grows unless
`δ_t ∈ span(U_{t-1})` *or* `k_t ∈ span(V_{t-1})` — and the second
condition is routinely met when **re-writing an already-stored key**
(precisely the delta rule's designed overwrite behavior, which correctly
does not add rank), while a genuinely **fresh** binding generically
satisfies both non-membership conditions (measure zero otherwise, and not
guaranteed by the parameterization). **So the naive reading of Option C is
false for exactly the writes that matter — fresh bindings**: an
unconstrained `U_tV_tᵀ` factorization is not closed under the delta rule;
every fresh-binding step can add one new rank direction. This is the
exact algebraic mechanism behind Nazari & Rusch / Sun et al.'s descriptive
upper bound `rank(S_t) ≤ t` (§2, item 1) — it is not an empirical
coincidence, it is what the recurrence's own arithmetic does absent any
constraint.

Two ways to make a factored parameterization genuinely closed, with very
different implications:

- **C-i (fixed subspace, `U`/`V` not updated by the recurrence itself).**
  Project `k_t`, `v_t` into a *fixed* (or slowly, separately learned)
  `k`-dimensional subspace before the delta rule ever runs, and run the
  entire recurrence inside that subspace (`S_t = U C_t Vᵀ`, `C_t ∈ ℝ^{k×k}`
  the only evolving part). This is exactly closed, no truncation, ever — but
  it does not test whether **the delta rule's own gradient-driven dynamics**
  respect a rank constraint; it tests whether a smaller-dimensional DeltaNet
  (behind a fixed input/output compression) can solve the task. That is an
  architectural-ceiling construction, structurally the same failure mode
  Stage 0 diagnosed as `H_cap` (`STAGE0_DESIGN.md` §2.1: a shared low-rank
  projection creates a **provable, training-independent** rank ceiling) —
  useful as a *positive-control upper bound* (§3.6), explicitly **not** the
  causal mechanism under test, and demoted accordingly (§6.2, C12).
- **C-ii (project the rank-(k+1) candidate back to rank k every step,
  exploiting the factored form).** Mathematically **identical** to Option A
  (still "truncate every step"), but computed cheaply: since `S_{t-1}` is
  already rank-`≤k` (given as `U_{t-1}V_{t-1}ᵀ`) and the update adds exactly
  one new rank-1 direction, the best rank-`k` approximation of the resulting
  rank-`(k+1)` matrix is obtainable from a `(k+1)×(k+1)` eigenproblem on the
  low-dimensional coefficient representation (the standard incremental/
  rank-1-update SVD trick — Brand, *"Fast low-rank modifications of the thin
  singular value decomposition,"* Linear Algebra Appl. 2006 — well-established
  numerical linear algebra, not novel here, cite correctly at build time)
  instead of a full `O(d³)` eigh on the reconstituted `d×d` matrix. This is
  a genuine causal constraint on the evolving state at every step, at
  `O(k³)` instead of `O(d³)` per truncation. **Revision 2, however, rejects
  it for the primary gate** (see §3.5): the per-chunk form revision 1 built
  on it is confounded with kernel chunk alignment (F5, FATAL), and the
  per-token form forfeits the fast kernel entirely — while a single
  read-point truncation carries the same causal content Task D's C1
  already validated. Kept here as worked algebra (it is also the honest
  account of *why* the unconstrained state grows rank) and as the basis of
  a possible future whole-trajectory-constraint variant.

### 3.5 Resolution — ranked, with the chosen mechanism (REWRITTEN, revision 2 — F5/F6)

> **F5 (FATAL), why revision 1's choice died.** Revision 1 chose C-ii
> "implemented at chunk-boundary granularity" — truncate once per kernel
> chunk. The attack round killed it on arithmetic: `fla-org`'s DeltaNet
> chunk size is one of **{16, 32, 64}** (default 64), so at the primary
> cells `K=16` and `K=32` at `d=64`, the **entire BIND phase fits inside
> the first chunk** — "truncate at each chunk boundary" fires **zero**
> times before BIND ends for 2 of the 3 primary cells, and the constraint
> silently only exists at `K=64`. The resulting "rank matters only at
> large K" pattern would be a mechanical artifact of chunk alignment,
> indistinguishable from a genuine rank threshold. The mechanism below
> replaces it entirely and sidesteps alignment by construction.

| Rank | Option | Causal? | Cost | Kernel-compatible? | Verdict |
|---|---|---|---|---|---|
| 1 | **Two-kernel-call split: BIND-only pass (`output_final_state=True`-style) → truncate `S_T` ONCE → decode (primary harness: external pinned readout; production check: QUERY-only pass seeded via `initial_state`)** | Yes — Task D's validated C1 pattern: a single train-time truncation per forward pass, gradients flowing through it at every training step | one `O(d³)` eigh per forward pass (`d ≤ 128` → trivial), reusing `rank_utils.truncate_to_rank` verbatim | Yes — the recurrence runs on the unmodified fast kernel in both calls | **Primary mechanism** |
| 2 | B: truncate at eval/read time only, never during training | No (post-hoc) | ~0 | Yes | **Required secondary control** (M2-equivalent), never sufficient alone |
| 3 | A / per-step C-ii: truncate the evolving state every token (or every chunk) | In principle yes, but the per-chunk form is **confounded by chunk alignment (F5)** and the per-token form forfeits the chunked kernel entirely | `O(k³)`–`O(d³)` × `T` (or `T/C`) | No | **Rejected for the gate.** A whole-trajectory-constraint variant remains possible future work, valid only if the alignment confound is closed by construction |
| 4 | C-i: fixed-subspace factored state, closed by construction | No (architectural ceiling, not gradient-caused) | ~0 | Yes | **Positive-control upper bound only** (§3.6/§6.2 C12), not the causal test |

**Chosen primary mechanism — the two-kernel-call split (F5 fix).** Every
forward pass, during training and eval alike:

1. Run the recurrence over the **BIND phase only**, requesting the final
   state via the kernel's state-exposure API (`output_final_state=True`-
   style; the exact kwarg is verified at the F15 checkpoint, §6.4, before
   anything is built against it).
2. Apply `rank_utils.truncate_to_rank(S_T, k)` **once** — the exact,
   battle-tested `eigh(ZZᵀ)` path Task D/E already validated, at the same
   once-per-forward-pass granularity as Task D's C1.
3. Decode from the truncated state: the primary bespoke harness computes
   `pred(a,h) = S_Tʰ · key_a` externally (§4.3, §5.4); the production-block
   secondary check seeds a QUERY-only kernel call with
   `initial_state = truncated S_T`.

Gradients flow back through the truncation into the recurrence at every
training step — a **train-time causal** constraint, not post-hoc eval
surgery, which is exactly the distinction Task D's C1 draws against
option B. Chunk alignment is irrelevant by construction: the truncation
point is the BIND/QUERY phase boundary itself, at every `(d, K)` cell
identically.

**Honest scope of what a single read-point truncation constrains:** it
binds the **interface the readout sees** (and, through training, what the
recurrence learns to pack into `k` directions) — the state trajectory
*during* the BIND phase may transiently exceed rank `k`. This is identical
in scope to Task D's C1 (the encoder's output `Z` was truncated once, not
its intermediate activations), so results remain directly comparable
across the Chapter 2 lineage; a whole-trajectory constraint is explicitly
out of scope (table row 3).

**Mandatory smoke assertion with teeth (F5 fix, part 2):** post-truncation,
`rank(S_T) ≤ k` is asserted by **direct SVD with an exact threshold** —
per the standing house rule that structural/integer checks take no
float-tolerance slack — for **every** `(d, K, k)` cell used anywhere in
the manifest, verified from the tensor itself, never inferred from a
config flag. The negative unit test (feed a known rank-`(k+1)` matrix; the
assertion must fire) is run **to completion** at build time, per the same
rule (C15, §6.2).

**Numerical-instability budget (REVISED — F6, F14):** with the two-kernel
split this is **one** eigh-truncation per forward pass — the same surface
Task D/E ran, *not* the `T/C`-fold multiplication revision 1 budgeted
against. Calibrating expectations from Task E's M4_E probe: its fr≥K dead
runs showed `n_skipped_steps` of 3–10 per 40K-step run — a per-step skip
**rate** of only ~0.008–0.025% (F14: rates, not raw counts) — yet were
still fully dead, so a low skip rate is *not* by itself reassuring. Wave
−1 (§6.4) therefore measures (a) the per-step skip rate on this
architecture and (b) the C15 post-truncation SVD rank assertion across
all manifest cells, and the B-probe gate (§6.4) still stands between the
probe and the full force-rank grid — exactly the discipline Task E's M4_E
applied (the full grid is not launched if the probe predicts a
near-certain dead outcome).

### 3.6 The architecture-native positive control (a genuine bonus, not assumed)

**Proposition.** If `{k_1, ..., k_K}` are exactly orthonormal and `β_t = 1`
for `t = 1..K`, then after streaming the `K` bindings, `S_K k_j = v_j`
**exactly**, for every `j = 1..K`, regardless of order.

*Proof (induction on `t`).* Base case trivial (`S_0 = 0`). Suppose
`S_{t-1} k_j = v_j` for all previously-written `j < t`. For the newly-written
key: `S_t k_t = S_{t-1}k_t − S_{t-1}k_t(k_tᵀk_t) + v_t(k_tᵀk_t) = v_t` (using
`k_tᵀk_t = 1`, `β_t=1`). For any earlier `j`:
`S_t k_j = S_{t-1}k_j − S_{t-1}k_t(k_tᵀk_j) + v_t(k_tᵀk_j) = v_j − 0 + 0 = v_j`
(using `k_tᵀk_j = 0`). ∎ (Wording note, F2: the induction uses only
`k_tᵀk_t = 1` and `k_tᵀk_j = 0` — the orthonormality premise decomposed
into its unit-norm and pairwise-orthogonality parts, not a weaker
assumption than the proposition states.)

This means `S_K = Σ_{j=1}^K v_j k_jᵀ` — **exactly Task E's hand-built
`Z_ideal`** — is reached by DeltaNet's *own native recurrence*, with `S_0=0`,
zero training, zero orthogonal-complement pollution (rank is exactly `K`,
not inflated toward `d` the way Task E's *SGD-trained* solution was,
`TASK_E_FINDINGS.md` §4). This upgrades Task E's `Z_ideal` reference (C7,
already a required control) from a **bolt-on analytic reference** into an
**architecture-native one for this design** — the same classical
Kohonen/Anderson fact (not claimed as novel), now realized by the tested
architecture's own equations rather than a hand-constructed comparison
object.

**Two direct, load-bearing consequences for the design:**

1. **Pre-registered primary rank metric: entity-subspace-restricted, not
   whole-matrix.** Task E discovered — expensively, after a full sweep and a
   dedicated Z-dump instrumented rerun (`TASK_E_FINDINGS.md` §4, §9) — that
   an SGD-trained solution's *whole-matrix* effective rank is polluted by an
   unconstrained, near-full-rank orthogonal complement the loss never
   touches, while the *entity-subspace-restricted* rank is exactly `K`. The
   proposition above shows *why* this is expected: nothing in the delta
   rule or the training objective constrains `S_t`'s behavior outside
   `span({k_j})`. This design pre-registers `analyze_zdump.py`-style
   subspace restriction (project onto the `K`-dimensional entity subspace,
   recovered from the *architecture-native* `Z_ideal`'s own SVD, §3 method)
   as **the** primary M1-equivalent rank metric from the start — not a
   post-hoc rescue.
2. **A real trained model does not directly control `k_t, v_t, β_t`** — they
   are learned projections (`W_k`, `W_v`) and a learned (typically
   sigmoid-gated) scalar of the raw token embedding, not the raw task
   vectors themselves. The open empirical question this design actually
   asks (M1-equivalent, §6.2) is whether SGD discovers `W_k, W_v, β`-gating
   that converges toward this clean orthonormal/`β≈1` regime (entity-
   subspace rank ≈ K, matching the architecture-native ideal) — or a
   messier solution that still achieves exact recovery but only by
   recruiting more than `K` effective rank on the entity subspace itself
   (which the theorem says is impossible for exact recovery in the other
   direction, but "close to exact" under a fixed cosine tolerance leaves
   room for over-provisioned, non-minimal solutions worth measuring, not
   assuming).

---

## 4. Crux 2 — the provable bound, and closing the token-shortcut escape

### 4.1 Proof sketch (adapted from `TASK_D_PREREGISTRATION.md` §3; premise restated per F3)

The bound is a property of the **readout**, not of how `S_T` was produced —
it therefore transfers to a recurrently-computed state without
re-derivation.

**Premise, stated carefully (F3).** A trained model with a *learned* write
gate stores, in the clean regime, `S_T k_j = β_j v_j` for learned per-write
strengths `β_j ∈ (0, 1]` — **not** literally `v_j`. Cosine scoring is
invariant to a positive per-item scalar, so recovery at `cos > τ` is
unaffected by any **nonzero** `β_j`. The bound survives in the same form:
direction-exact recovery of all `K` bindings means `S_T · K_mat = V_mat D`
for some invertible diagonal `D = diag(β_j)`, and
`rank(V_mat D) = rank(V_mat) = K` ⇒ **`rank(S_T) ≥ K`**. The operative
premise is therefore **linear independence of the recovered images
`{S_T k_j}`**, which needs: `{k_j}` linearly independent (orthonormal by
the gate default), `{v_j}` linearly independent (guaranteed by the
injective-graph constraint, §5.1, Task E's C6 verbatim), and every
`β_j ≠ 0` — it does **not** require `β = 1`.

**Explicitly noted (F3), because it is invisible to every headline
metric:** under `h`-fold composition, the prediction's *magnitude* decays
as the product of the traversed β's (`βʰ` for uniform β) while its
*direction* stays exact in the clean solution — and cosine scoring is
blind to this decay **by construction**. This is a designed property (the
same scale-invariance Task E's Z-dump analysis had to fit and remove as
the `c*` scalar, `TASK_E_FINDINGS.md` §9), not an oversight — but it means
a magnitude collapse at large `h` cannot be seen in any cosine-based
number, so **per-hop prediction norms are logged as a standing diagnostic
(§6.3)** rather than discovered post-hoc.

**Scope of the truncation point (replaces revision 1's chunk-boundary
caveat, obsolete under the F5 fix):** every query reads exactly the
once-truncated `S_T` — the truncation is applied at the BIND/QUERY phase
boundary by construction (§3.5), so no mid-sequence alignment ambiguity of
any kind exists.

**Grammar:** `BIND k_1 v_1 ... BIND k_K v_K` streamed as `K` literal sequence
positions (not a set, as in Task D/E — this is the actual difference a
recurrent architecture buys), each position's token embedding carrying
`[k_t ; v_t]`, projected by the layer's own `W_k, W_v` into the recurrence's
key/value. `β_t` is hard-masked to 0 on every position that is not a BIND
**item** position — including every C14 buffer token (NEW-3) — and left
**learned** (token-conditioned) on BIND item positions (§5.2; §7 item 7 — the
mask is architectural, per Task D §4's "enforced by an explicit mask, not
hoped for" standard, while the BIND-phase gate is deliberately NOT forced
to 1, or the M1-equivalent question would be vacuous).

### 4.2 Closing the position-decomposition escape (state-internal — provably closed)

`S_T` is a single `d×d` matrix per (layer, head) — exactly Task D/E's `P=1`
single global state. There is no "K positions, each rank-1" pigeonhole
available *within* the state itself; this escape is closed by construction,
identically to Task D/E, and for the identical reason (no multiple latent
slots to route information into).

**But two new, architecture-specific reincarnations of the same escape must
be closed explicitly, or the gate is vacuous:**

- **Multi-head escape (new to this architecture).** `H` heads, each with its
  own `d_head × d_head` state, could jointly store `K` items via `H`
  independent rank-`(K/H)` states without any single head ever reaching rank
  `K` — the head-level analog of position-decomposition. **Fix, mandatory:**
  the primary gate uses a **single head** (`H=1`), directly mirroring Task
  D/E's own P=1-first sequencing decision. Multi-head is an explicitly
  deferred generality question (§9), not folded into the gate.
- **Multi-layer escape (new to this architecture).** Two DeltaNet layers
  produce two independent final states `S_T^{(1)}, S_T^{(2)}`; if the
  readout is allowed to combine both (e.g. via the residual stream), the
  same joint-storage-across-slots escape reappears at the layer level.
  **Fix, mandatory (control C10, §6.2):** the readout reads **only** the
  final layer's final state. If a second layer is used for capacity reasons
  (§6.1), force-rank is applied to **both** layers' states, or the earlier
  layer's contribution to the readout is blanked out and the blank-out test
  (below) is extended layer-wise.

### 4.3 Closing the raw-token-shortcut escape (the crux this section is really about)

Production `fla-org` DeltaNet blocks are not "just" the delta rule — they
typically wrap it with **short causal convolution** on the k/v/q streams
before mixing, and sit inside a standard transformer block with a **residual
stream** that can carry information from raw token embeddings past the
recurrence entirely. Both are potential leaks around the `S_T`-only
bottleneck Task D/E's whole methodology depends on.

**Two are architecturally different and must be treated differently:**

- **Short conv *before* the recurrence (`k_t, v_t = shortconv(x_t)`, then fed
  into the delta rule as usual).** This is part of *computing* `k_t`/`v_t` —
  it does not bypass the state; the recurrence still only writes what the
  conv produces into `S_t`, and the theorem's premises (linear independence
  of the *effective* keys/values that reach the recurrence) still apply to
  whatever the conv outputs, as long as those effective keys/values are
  still what gets checked for orthonormality/injectivity at generation time.
  **Not a leak by itself** — but it does mean the *raw* task keys are no
  longer guaranteed orthonormal after the conv mixes adjacent positions;
  §5.1's injectivity/orthonormality asserts must be re-checked on the
  **conv-mixed** effective keys at smoke-test time, not just the raw
  generator output, or the bound's premises can silently fail to hold. If
  the conv is enabled, this is a build-time smoke-gate addition (a
  `task_d.py`-style `_self_test` analog, run on the *model's* effective
  keys, not just the generator's).
- **A residual/skip connection *around* the entire DeltaNet mixing block**
  (letting the block's output — and hence anything reading it — see a
  function of raw embeddings independent of `S_t`). **This is a genuine
  leak** and must be disabled (or the readout must be structurally prevented
  from reading it) for the primary gate, identically to how Task D's blank-
  out test exists to catch exactly this class of bug.

**Conv-boundary bypass (F12 — FATAL-scoped for the conv-enabled secondary
check; the primary bespoke harness has no conv and is unaffected).** With
`conv_size = 4` (the `fla-org` default — recorded at the F15 checkpoint,
§6.4), each position's k/q/v are computed from a causal window over the
current and previous `conv_size − 1 = 3` token embeddings. A QUERY position
within 3 tokens of the BIND/QUERY boundary therefore builds its features
**directly from raw BIND-phase embeddings** — bypassing the "readout is a
pure function of `(S_T, query)`" requirement through the feature-
construction path. The one-sided blank-out below (corrupt inputs *after*
state extraction) **structurally cannot catch this**: the leak reaches
*backward* from BIND tokens into QUERY-token feature construction, not
forward. Two mandatory fixes (C14, §6.2):

- **Buffer tokens:** insert ≥ `conv_size − 1` information-free buffer
  tokens between every BIND item **and** at the BIND/QUERY boundary.
  Besides closing the boundary leak, this stops the conv from smearing
  adjacent BIND items into each other's effective k/v — which would
  otherwise impose a **rank-independent accuracy ceiling** and silently
  degrade the C13 orthonormality premise on the effective keys.
  **Buffer-token embeddings are pinned to exact zero vectors, not learned
  (NEW-2):** under the §3.5 two-kernel split, the QUERY-only call's conv
  starts from **zero padding**, so a nonzero buffer embedding would make
  single-call and two-call outputs differ at the boundary on the conv
  path — and F15's round-trip check would fail mysteriously. Zero-pinning
  makes the kernel's zero padding and the buffers equivalent **by
  construction**, so C14 and the two-kernel split reinforce rather than
  conflict; F15's round-trip-equality spec is stated *given* this
  convention (§6.4). Buffer positions also carry a hard β=0 mask (§5.2,
  NEW-3), so they never write into the state either.
- **Two-sided blank-out:** in addition to the forward direction, corrupt
  the last `conv_size − 1` BIND-phase tokens and verify the QUERY phase's
  feature construction (and hence every prediction, given a fixed `S_T`)
  is bit-identical.

**Mandatory pre-training smoke gate item (extends Task D §4 / Task E's
C_composition-purity, C_composition-purity-equivalent here):** the harness
must be built so the readout is a **pure function of `(S_T, query)`**, not
"a transformer block whose output we hope only depends on `S_T`." Concretely
(recommended, matching Task D's `MatrixMemoryModel.unbind` pattern almost
exactly): run the actual `fla-org` DeltaNet recurrence over the BIND-token
sequence, **extract the layer's own exposed final recurrent-state tensor**
(verify the exact API at build time — e.g. an `output_final_state=True`-style
flag; do not assume the exact kwarg without checking the installed package
version), then compute every prediction via an **external, pinned function**
`pred(a,h) = S_Tʰ · key_a` **outside** the transformer block's normal forward
pass entirely — the block's own output projection / residual stream is never
consulted. Under this harness shape, the blank-out test is close to satisfied
*by construction* (the decode function structurally cannot see anything but
`S_T`); its real job becomes verifying that `S_T` **itself**, as extracted
from the black-box kernel, does not already have leaked information baked
into it via an internal residual path the kernel's own forward pass used
*before* exposing the state. Verify by corrupting/zeroing every input token
embedding **after** the point `S_T` is extracted and confirming bit-identical
predictions across the full decode sequence — the exact Task D/E blank-out
procedure, applied at the state-extraction boundary rather than an
attention-mask boundary.

**Recommended build sequencing (not gating this design, but flagged for the
build phase):** (i) primary gate on a **minimal, bespoke single-layer,
single-head DeltaNet-only harness** (delta rule + linear input/output
projections, no conv, no surrounding transformer block) — closest fidelity
to Task D/E's controlled setup, cheapest to audit, and where §4.1's premises
are easiest to verify hold exactly; (ii) a **secondary, explicitly-labeled
robustness check** on the actual `fla-org` production block (conv enabled,
inside a real block) with the extended blank-out test above — this is what
lets the design honestly claim "a mature production architecture," not just
"a DeltaNet-shaped reimplementation." A CONFIRM on (i) alone is not
sufficient for the "production architecture" framing in §8 — (ii) is
required before that specific claim is made, though (i) alone is still a
complete, decisive, publishable result on its own terms (the vanilla delta
rule, not the surrounding production block).

---

## 5. Crux 3 — compositional transfer in a streamed setting

### 5.1 Reuse, don't reimplement

`task_e.py`'s generator is **architecture-agnostic** — it only produces
`(keys, values, query (a,h) pairs)`; nothing in it assumes a set-encoder.
Reuse verbatim: the single-Hamiltonian-K-cycle construction
(`_permutation_graph`), the injective-graph assert-at-generation-time check
(closes the merge/miscounting attack that killed MNNS), the
`TaskEConfig.__post_init__` periodicity guards (`h mod K` avoiding
`{0} ∪ H_train_residues ∪ H_test_residues` — including the addendum fix for
the `H_test`-collision escape the first guard missed), and the
`effective_hop`-stratified reporting convention. None of this needs
re-deriving; all of the hard-won correctness fixes transfer mechanically
because they are properties of the *generator*, not the *encoder*.

### 5.2 Freezing `S_K` for the query phase — a hard mask, not a hope

Unlike Task D/E's non-causal set encoder (which produces `Z` once, from all
`K` bindings simultaneously, then reads it), a streamed recurrence keeps
running unless explicitly stopped. If `β_t` is left to a learned,
soft-gated value on QUERY/SEP-token positions, the state could keep drifting
during the query phase, silently violating the "compose via a pure matrix
power of a **frozen** `S`" pinned-readout requirement — the direct streamed
analog of Task E's C_composition-purity control.

**Fix, mandatory, architecturally enforced (rewritten, NEW-3):** `β_t` is
multiplied by a binary token-type mask — `1` on BIND **item** positions
ONLY (where the learned gate passes through, per §7 item 7), hard `0` on
**ALL** other positions, **including every C14 buffer token** — enforced
**outside** the model's own learned gate, not trained to converge there.
The rewrite matters: revision 2's wording ("1 on BIND positions, 0 from
the first QUERY token onward") predated C14's buffer tokens, which fall
in *neither* of those regions — left unmasked, the inter-item buffers
would write junk into the state, consuming rank budget under force-rank
and manufacturing a spurious collapse mechanism. This is the exact same
"hard bottleneck mask (mandatory)... enforced by an explicit mask, not
hoped for" standard Task D §4 already sets, applied to the temporal axis
instead of the attention axis. `S_T` (the frozen state at the end of the
BIND phase) is what §4.3's external readout function receives.

**Revision-2 note (follows from the F5 fix):** under the §3.5 two-kernel
split, the **primary bespoke harness satisfies this freeze by
construction** — the BIND-only kernel call simply ends at the phase
boundary, and no QUERY token ever enters the recurrence at all. The hard
β-mask above is therefore load-bearing only for the **production-block
secondary check** (§4.3), where QUERY tokens do pass through the full
block; it remains mandatory there.

### 5.3 A streaming-specific bonus: order-invariance under the gate default

Task D/E's "BIND order shuffled" control exists to close a positional-
recency shortcut in a set encoder. In a genuinely sequential architecture,
order could matter for real reasons (later writes can partially overwrite
earlier ones) — **except** that §3.6's proposition shows the opposite is
true in the pre-registered orthonormal-key/β=1 regime: writes are exactly
order-independent there (the inductive proof used only orthogonality, not
write order). This means, in the gate-default regime, the "shuffle BIND
order" control is **provably moot**, not merely empirically checked — a
genuinely stronger guarantee than Task D/E had. Order-*dependence* becomes a
live, informative question only in a secondary, near-orthogonal (Gaussian)
key robustness variant (mirroring Task D's own orthogonal-vs-Gaussian split,
§6.2 note under C5), where the proposition's premises no longer hold exactly
and order effects are a real, measurable, expected phenomenon worth
reporting rather than a bug to control away.

### 5.4 Pinned readout

```
pred(a, h) = S_T^h · key_a        # literal repeated matrix action, no learned per-hop weights
```

Identical in spirit to Task E §2's readout; `S_T` here is DeltaNet's own
extracted final state (§4.3) rather than an attention-encoder's output.
Scored by the same absolute cosine criterion, never a K-way argmax. `H_train`
/ `H_test` / `H_extra` grids reused from Task E's pre-registered values
(`{1,2,3}` / `{4,5,6}` / `{7,21}`) at the operating points this design's
manifest selects (§6), re-validated per-K against the periodicity guard
exactly as Task E's own addendum requires.

---

## 6. Testbed & manifest

### 6.1 Architecture

- `fla-org/flash-linear-attention` DeltaNet layer. The package is **not
  currently installed anywhere in this repo** (confirmed by search; no
  install steps yet in `matrix-thinking/H100_SETUP.md`) — the exact
  class/API, `output_final_state`/`initial_state` hooks, chunk-size
  options, and `conv_size` default are verified at a formal, mandatory
  **pre-Wave-−1 checkpoint (F15, §6.4)**, not an informal build note.
- **Primary gate:** 1 layer, 1 head (`H=1`, closes the multi-head escape,
  §4.2), vanilla delta rule (`n_h=1` Householder per token, closes the
  DeltaProduct conflation, §2 item 3, control C11).
- State dim `d ∈ {64, 128}` per the requested range — **both** tested in
  Wave 0 (§6.4) with a decision rule selecting the primary operating point,
  rather than assumed; d=64 is the a priori favorite (keeps `K ≤ d` scaling
  logic in the already-validated Task D regime, 4× Task D's own gate `d=16`,
  without pre-committing to a new architecture *and* a new scaling frontier
  simultaneously). d=128 is a stretch/generality cell.
- `<5M` params — comfortably satisfied at this `d`/layer count for a
  DeltaNet layer (params scale with `d` roughly linearly for the
  key/value/gate projections, not quadratically the way Task D's shared
  `row_out` encoder did — exact count is a build-time derivation once the
  real `fla-org` module is instantiated; do not hand-derive a formula
  against unverified internals here, per this project's "verify before
  external use" convention).
- A priori reason to expect DeltaNet's *write* mechanism scales more
  gracefully to larger `d` than Task D's own encoder: Task D's `H_collision`
  failure mode (Stage 0 §2.2) is a symptom of `d` *shared* row-reader
  queries needing to differentiate at once from one small identity space;
  DeltaNet's delta rule writes **one token at a time**, each step solving a
  much smaller local update (`v_t − S_{t-1}k_t`), with no analogous
  simultaneous-differentiation requirement. This is a **hypothesis**, not
  assumed — Wave 0 (§6.4) tests it empirically rather than skipping the
  d=32-style trainability check Stage 0 had to run for the encoder.

### 6.2 Controls (Task D/E's C1–C8, C_MLP, C_composition-purity reused verbatim + new)

| # | Control | Closes |
|---|---|---|
| C1–C5, C7, C8 (Task D/E, unchanged) | Train-time force-rank (§3.5's two-kernel split — `rank_utils.truncate_to_rank` applied once per forward pass at the BIND/QUERY boundary), ≥5 seeds, chance-adjusted per-item accuracy, held-out key/value vectors, idealized-Z-equivalent reference (§3.6, now architecture-native), eigenstructure fidelity | Same roles as Task D/E |
| C6 (Task E, unchanged) | Injective functional graph (single K-cycle), asserted at generation time | Merge/miscounting attack |
| C_MLP (Task E, unchanged) | Flatten+one-hot(h) shortcut baseline, expected to fail at held-out h by construction | Establishes the honest floor |
| C_composition-purity → **C9 (streamed form)** | Query-phase state freeze: **structural by construction** in the primary harness (the BIND-only kernel call ends at the phase boundary, §3.5/§5.2); in the production-block check, β-mask = 1 on BIND **item** positions only, hard 0 on **all** other positions including every C14 buffer token (§5.2, NEW-3) | Silent state drift during the query phase; buffer positions writing junk into the state (a spurious rank-budget drain under force-rank) |
| **C10 (new)** | Readout reads **only** the final layer's final state; multi-layer contributions blanked out or separately force-ranked | Multi-layer joint-storage escape (§4.2) |
| **C11 (new)** | Primary gate is vanilla single-Householder-per-token DeltaNet (`n_h=1`), single head (`H=1`) | DeltaProduct transition-rank conflation (§2 item 3); multi-head escape (§4.2) |
| **C12 (new, positive-control only, not the causal arm)** | Fixed-subspace factored variant (§3.4's C-i) run as an **upper-bound reference**, never reported as evidence of gradient-caused rank use | Distinguishes "architecture *can* solve this at rank k" (C12) from "gradient-forced-to-rank-k SGD solution does" (the causal arm) |
| **C13 (new)** | Effective-key re-orthonormality/injectivity check on the *post-shortconv* keys/values if the production-block secondary check (§4.3) is run | Short-conv-before-recurrence premise check |
| **C14 (new, revision 2 — F12)** | ≥`conv_size−1` information-free buffer tokens between BIND items and at the BIND/QUERY boundary, plus a **two-sided** blank-out (conv-enabled secondary check only, §4.3) | Conv-boundary backward leak around `(S_T, query)` purity; conv smearing of adjacent items (a rank-independent accuracy ceiling) |
| **C15 (new, revision 2 — F5)** | Post-truncation `rank(S_T) ≤ k` asserted by **direct SVD, exact threshold**, on the tensor itself, for every `(d, K, k)` manifest cell; negative unit test (rank-`(k+1)` input must trip the assert) run to completion at build time | "Config says rank-forced" ≠ "state is rank-k" — the assertion has teeth, not inference |

### 6.3 Pre-registered primary metric (carried forward from §3.6, stated once here for the manifest)

**Entity-subspace-restricted effective rank of `S_T`** (project onto the
K-dimensional span recovered from the architecture-native `Z_ideal`'s SVD,
`analyze_zdump.py`-style method) is the **primary** M1-equivalent metric.
Whole-matrix effective/stable rank is still logged (comparability with
Task D/E, and to reproduce or contradict the "orthogonal complement fills
with training" pattern independently on this architecture), but is
**secondary**, not the decision instrument.

Two standing diagnostics logged at every eval (revision 2):

- **Effective-key Gram deviation `‖K_effᵀ K_eff − I‖_F` (F4)** — computed
  on the keys that actually reach the recurrence (post-conv on the
  production path). Separates rank-collapse failures from
  keys-drifted-from-orthonormality noise when interpreting the M1/M3-
  equivalent results, and doubles as C13's quantitative instrument.
- **Per-hop prediction norms (F3)** — cosine scoring is blind to β-driven
  magnitude decay under composition (§4.1), so norms are recorded
  explicitly rather than discovered post-hoc.

### 6.4 Manifest (waves, gated, budget for late transitions from the start)

Following Stage 0's own corrected practice (§0 reading list): extended
budgets and mid-training checkpoints are **built into Wave 0 from the
start**, not added after a wasted first round.

| Wave | Purpose | Scope | Est. GPU-h |
|---|---|---|---|
| **F15 checkpoint (pre-−1, blocking)** | **`fla-org` API verification (F15) — the mandatory first action of the build phase; nothing is built against assumed APIs.** Install the pinned package version on the cluster; verify the state-exposure and state-seeding hooks (`output_final_state` / `initial_state` or their actual names) exist and **round-trip** — a state extracted from a BIND-only call, fed back as the initial state of a continuation call, must reproduce a single continuous call's outputs to numerical tolerance, **evaluated given zero-embedding buffer tokens at the boundary (NEW-2)**: the QUERY-only call's conv starts from zero padding, so round-trip equality on the conv path is only *expected* under the C14 zero-pinned-buffer convention — a mismatch under that convention is a real failure, not a mystery. **Gradcheck through both hooks (NEW-1, blocking):** a finite-difference gradcheck (small `d`, fp64) through BOTH the backward of `output_final_state` into the recurrence AND the backward through `initial_state` seeding — §3.5's "gradients flow back through the truncation into the recurrence" is an assumption about the kernel's autograd wiring until this passes; run with C15's run-to-completion negative-test discipline (a deliberately detached/broken gradient path must FAIL the check). Also record the chunk-size options ({16,32,64} per the attack round — re-verify), the `conv_size` default (4 — re-verify), and the exposed state tensor's exact shape/layout/dtype per head. Findings written into the build notes and this doc's §10 before Wave −1 launches. | 1 script, CPU/1 GPU | <0.5 |
| **−1** | **Timing + instability calibration (mandatory, CLAUDE.md).** (a) Unconstrained fast-kernel DeltaNet at `d∈{64,128}`, tier-default steps, 1 seed each; (b) the §3.5 two-kernel-split force-rank arm at the same cells, 1 seed each — measuring the per-step wall-clock **ratio** of (b) vs (a) (the two-kernel overhead + one eigh per forward pass), the per-step eigh skip **rate** (F14), and the C15 post-truncation SVD rank assertion across every `(d,K,k)` cell in the manifest. **Pre-committed decision rule (F13):** ratio ≤ 2× → proceed with the manifest as priced below; ratio 2–5× → apply the pre-registered cut order *before* Wave A launches; ratio > 5× (not expected — one `d≤128` eigh per forward pass) **or** per-step skip rate > 0.1% **or** any C15 assertion failure → **stop**, diagnose, and re-price with a revised mechanism (fp64 eigh path, or the differentiable-surrogate fallback) before any downstream wave launches. | 4 runs | ~6 |
| **0** | **Extended-budget transition calibration (mandatory, STAGE0_DESIGN.md §12 lesson).** Unconstrained arm only, at **2.5× tier-default steps**, dense (≤2K-step) checkpoints, **≥3 seeds** per primary cell: `d=64, K∈{16,32}` and `d=128, K∈{32,64}`. Decision output: which `d` is the healthier primary cell (§6.1), and the empirical transition-onset window (feeds Wave A/B's step budget the way Task E's own 40K-step re-registration was forced to, but pre-emptively here). | 12 runs | ~22 |
| **A** | **Main screening grid** at the Wave-0-selected primary `d`, calibration-selected step budget: unconstrained M1/M2-equivalent (entity-subspace rank vs. K, eval-time truncation curve) across `K` grid, 2 seeds; Task-E-equivalent compositional multi-hop (`H_train={1,2,3}`, `H_test={4,5,6}`, `H_extra={7,21}` re-validated per-K per §5.4) at the primary `K`, 3 seeds. | ~22 runs | ~28 |
| **B-probe** | **Force-rank calibration probe (Task E M4_E discipline — do not launch the full grid blind).** Two-kernel-split force-rank at `k ∈ {K−1, K, K+1}`, primary `d`/`K`, 3 seeds — mirrors Task E's own 9-run M4_E probe that correctly predicted the full grid would be near-certainly dead before spending ~20 GPU-h confirming it. (Re-priced for revision 2: with the F5 fix the force-rank arm costs ≈ the unconstrained arm — one extra eigh per forward pass — not a multiple of it.) | 9 runs | ~12 |
| **B** | **Full force-rank straddle grid (PRIMARY causal test, M3-equivalent) — launched only if B-probe shows life** (≥1 seed reaches non-trivial recovery at `k ≥ K` and a step relative to `k < K`, mirroring Stage 0's "life" bar convention). `k` straddle grid around primary `K`, 5 seeds; held-out-hop composition (M3_E-equivalent) at the same rank grid. | ~15–20 runs if launched | ~30 (contingent) |
| **Reserve** | Robustness + pre-registered probes: production-block secondary check (§4.3, with C14); Gaussian-key order-dependence probe (§5.3); **H∈{2,4} multi-head probe (F11, pre-registered):** 2–4 runs at the primary cell measuring **per-head** entity-subspace-restricted rank against the joint bound `Σ_head rank(S_T^{(head)}) ≥ K` — the first datum behind §8's H=1 qualifier, not a full multi-head study. | ~8–12 runs | ~12 |
| **Total** | | | **≤ ~110 planned against the ≤120 ceiling** (Wave −1 + the F13 rule convert these estimates into a priced budget before Wave A commits anything; Stage 0's experience is that flat-rate guesses run 5–6× conservative once real timing lands — treat this as an upper bound, not a target) |

**Pre-registered cut order if Wave −1/0 reveal the budget is tight** (same
discipline as `STAGE0_DESIGN.md` §6's cut order): (i) drop the d=128 Wave 0
cells to 2 seeds if d=64 is clearly healthier; (ii) drop Wave A's K-grid to
the single primary K plus one straddle point rather than a dense grid;
(iii) B-probe and Wave B's primary-K, 3-seed floor are **never** cut — they
carry the entire causal claim (§1b).

**If B-probe shows death (Task E's actual M4_E outcome for the analogous
production-recipe combination)**: do not launch Wave B blind. Document the
death as a finding (numerical instability under the multi-hop/streamed
objective interacting with force-rank, exactly Task E's own §6 item 2
reasoning), and — before declaring the causal test unmeasurable at this
operating point — spend the Reserve budget on one architecture-level
mitigation probe (e.g., a softer, differentiable rank-forcing surrogate in
place of hard eigh-truncation, or applying the truncation on a subset of
training steps rather than every forward pass) rather than concluding
FALSIFY on an instability artifact. This mirrors Task D/E's own "attack
yourself" discipline of not confusing "the mechanism is broken" with "our
specific implementation of the mechanism is numerically fragile."

> **Advisory (2026-07-02, audit round-2, pre-registered before any Wave B
> data) — F18.** B-probe (and any downstream Wave B cell) may run under
> `trunc_impl="svd_lowrank"` if the F13 decision rule selects it — the
> pre-registered fallback for the eigh path's measured training-time
> instability (audit round-1 FINDING 1, `model_dn.py`'s
> `truncate_to_rank_svd_lowrank` docstring). That fallback is not a neutral
> stand-in for the causal question B-probe exists to answer:
> `torch.svd_lowrank` is a **stochastic** rank-≤k projection (fresh
> Gaussian sketch per call, no fixed generator — same docstring) that
> **loses energy relative to the optimal rank-k truncation specifically on
> near-degenerate spectra** — the exact regime this task's orthonormal-key
> solutions cluster into by construction (`truncate_to_rank_svd_lowrank`'s
> own docstring). Verified this round on synthetic near-degenerate spectra
> at a `sigma_{k-1}/sigma_k` gap ratio of 1.1: captured energy ≈92% of the
> optimal rank-k truncation on average, and materially different low-rank
> factors returned call-to-call on the *identical* input (re-randomized
> subspace estimate — confirmed on this run, not merely asserted from the
> docstring). This bias is **one-directional**: it can only make a
> rank-k-forced state look worse than the model's own converged `S_T`
> actually is, never better.
>
> Three consequences, pre-registered before any B-probe data is read: **(a)
> a marginal collapse observed at `k=K−1` must not, by itself, be read as
> evidence the model needs rank K** — it is equally consistent with
> `svd_lowrank`'s own near-degenerate energy loss, not a causal rank
> effect; **(b) conversely, a non-collapse at `k<K` is stronger falsifying
> evidence than it would be under an unbiased truncation** — the
> implementation's bias runs the other way, so surviving a handicapped
> truncation is the harder result to explain away; **(c) one eval-time
> eigh-truncation control run on the B-probe `k<K` cell is pre-registered
> as mandatory before any Wave B life/death verdict is recorded** — eigh's
> known instability is a *training-time*, backward-pass problem (audit
> round-1 FINDING 1); at eval time, with no backward pass required,
> `rank_utils.truncate_to_rank` is cheap and deterministic, and applying it
> alongside `svd_lowrank` at exactly the B-probe `k<K` cell isolates "the
> trained state can't support rank k" from "`svd_lowrank` under-represents
> rank k at eval time too." Costs one extra eigh per checkpoint on an
> already-trained cell — no retraining required.

### 6.5 Operational-harness build requirements (F17 — blocking, per CLAUDE.md's learned rules)

Carried into the build phase as requirements, not suggestions — every one
is a lesson this project has already paid for once:

- **Run naming carries `(wave, variant, d, K, k, seed)`** — the Stage 0
  MAJOR-2 lesson: two variants at the same `(d, K, k, seed)` must not
  collide on a resume path and be silently skipped.
- **Resume is validity-checked:** `is_done` requires a *parseable* result
  JSON containing the final-metrics fields, not mere file existence —
  a supervisor restart must lose nothing and re-run nothing valid.
- **Waves are bounded and gated:** no wave auto-launches the next; each
  gate's decision (F13 rule outcome, Wave 0 `d`-selection, B-probe
  life/death) is written to the wave's summary file *before* the next
  manifest is generated.
- **Long-running orchestration runs inside `tmux new-session -d -s <name>`
  wrapped in a self-healing supervisor loop** (`while [ ! -f STOP ]; do
  <cmd>; sleep 15; done`), never a backgrounded SSH shell; kills go
  through `tmux kill-session -t <name>` or exact PIDs, **never**
  `pkill -f <pattern>` (the self-kill footgun already learned on this
  cluster).
- **Mid-training eval checkpoints every ≤2K steps** (transition dating +
  any-checkpoint life criteria, per Stage 0's practice), written
  incrementally to the run's JSON.
- **Everything logs to a file; each wave emits a human-readable summary;
  the exact scripts run are archived under `experiment-runs/`** alongside
  results, per the house reproducibility rule.

---

## 7. Attack-yourself

1. **Revision 1's chunk-boundary mechanism was itself the biggest bug in
   this document (F5, FATAL — caught by the independent attack round, not
   by self-audit).** Chunk alignment would have silently zeroed the
   constraint at 2 of 3 primary cells and manufactured a fake K-threshold.
   Resolved by the two-kernel split (§3.5). The surviving residual, stated
   honestly: a single read-point truncation constrains the readout
   interface, not the full BIND-phase trajectory (the state may transiently
   exceed rank `k` mid-BIND) — identical in scope to Task D's own C1, and a
   whole-trajectory constraint is explicitly future work (§3.5).
2. **Numerical instability may make the force-rank arm entirely
   unmeasurable**, as it did for Task E's fr≥K arm — dead runs at a
   per-step skip rate of only ~0.008–0.025% (F14). With the F5 fix the
   surface here is the *same* as Task D/E's (one eigh per forward pass),
   not larger as revision 1 claimed — but Task E's arm died on exactly
   that surface, so the risk stays fully live. Mitigated by Wave −1's
   measured skip rate + C15 assertions + the F13 stop rule, B-probe gating
   (§6.4), and a pre-registered mitigation-probe fallback rather than a
   blind FALSIFY.
3. **Multi-head/multi-layer escapes (§4.2)** are new to this architecture
   and easy to get wrong by defaulting to a "realistic" multi-head config
   for the primary gate. Mitigated by pinning `H=1`, single layer, as
   mandatory controls (C10, C11) — not a suggestion.
4. **Short-conv-before-recurrence silently breaks the orthonormality/
   injectivity premises** the bound depends on, without being a "leak" in
   the blank-out sense (§4.3). A careless build could pass the blank-out
   test cleanly while the theorem's premises are quietly false on the
   conv-mixed effective keys. Mitigated by C13 (re-check premises on
   post-conv effective keys, not just generator output) — but only if the
   production-block secondary check is run; the primary bespoke-harness
   gate (§4.3) has no conv and this attack does not apply there.
5. **DeltaProduct conflation (§2 item 3)** — the single most likely reviewer
   confusion in this entire design. Mitigated by C11 and by explicitly
   naming and separating "state rank" from "transition-matrix rank"
   throughout (§2 item 3, §3.1), not just in a footnote.
6. **"Fixed the wall by picking d=64" is a hypothesis-confirming choice, not
   a neutral one** — §6.1's stated a priori preference for d=64 over d=128
   could bias Wave 0's decision rule if not applied mechanically.
   Mitigated: Wave 0 (§6.4) tests both `d` values with ≥3 seeds each and a
   pre-stated decision rule (healthier cell wins), not a post-hoc
   justification for the a priori guess.
7. **The architecture-native `Z_ideal` (§3.6) could make CONFIRM too easy to
   get for the wrong reason** — if the harness accidentally sets `β_t=1`
   architecturally (rather than letting SGD learn to converge there), the
   "does the gradient recruit rank" question would be vacuous (the
   architecture would recruit exact rank K by construction, regardless of
   training). Mitigated: `β_t` must remain a **learned**, token-conditioned
   gate for BIND positions in the primary gate (only the QUERY-phase
   hard-zero, §5.2, is architecturally forced) — §3.6's proposition is a
   reachability argument (SGD *could* converge there) and an analytic
   reference point, not a claim that the harness enforces it.
8. **This design's "production architecture" framing (§8) is only earned by
   the secondary `fla-org` production-block check (§4.3), not the primary
   bespoke harness alone.** A CONFIRM on the bespoke harness alone is a
   complete, decisive Task-D/E-style result but must not be over-claimed as
   "a production architecture recruits rank" without the secondary check
   also landing. Stated explicitly in §4.3 and repeated here so it cannot
   be silently dropped at write-up time.

---

## 8. What result is publishable where

Completing the 3×1 table from §0/§1:

- **CONFIRM (bespoke harness only):** matches Task D/E's publication bar —
  a decisive, from-scratch-architecture-agnostic extension of "SGD recruits
  provably-necessary rank" to a second, independently-designed recurrent
  mechanism (not just a second task on the same encoder). Strengthens the
  Chapter 2 story's generality claim; publishable alongside Task D/E as a
  companion result, same venue tier (workshop-to-mid-tier, matching the
  existing ICML MI-workshop lineage).
- **CONFIRM (bespoke harness + production-block secondary check, §4.3):**
  the strongest result in this lineage — a real, deployed, hardware-
  optimized architecture family (Gated DeltaNet in Qwen3.5/Qwen3-Next,
  `research/validated-cheap-matrix-ops.md`) causally recruits and depends
  on provably-necessary rank, with a train-time causal ablation not just a
  descriptive measurement. This directly and constructively engages the
  Nazari & Rusch / Sun et al. rank-measurement literature (§2 item 1) —
  worth a main-track submission (ICLR/NeurIPS), not just a workshop,
  precisely because the target architecture has real-world adoption, unlike
  the bolt-on matrix-CODI or the from-scratch matrix-native encoder.
  **Mandatory qualifier at this tier (F11): the gate is single-head
  (`H=1`) by design (§4.2) — multi-head generality is NOT yet validated.**
  The Reserve wave's pre-registered H∈{2,4} per-head-bound probe (§6.4)
  is the first datum; until a proper multi-head study lands, the claim
  must be phrased as "a production fast-weight *mechanism*," not
  "production *configurations*."
- **FALSIFY:** equally decisive and citable, in a different direction —
  shows that even a hard, well-optimized recurrence that can *reach* the
  exact minimal-rank solution by its own native dynamics (§3.6) does not
  get pushed there by ordinary next-token-style training pressure. This
  would be a sharp, mechanistic caveat on how to read Nazari & Rusch / Sun
  et al.'s descriptive `rank(S_t) ≤ t` measurements on real pretrained
  models: **an upper bound on rank does not imply the training process
  causally selects for low rank when a task would benefit from it** — a
  genuinely useful clarification for that literature, not merely a negative
  result for this project. Publishable as a companion negative to Task D/E,
  same house conventions (F9, anchors corrected): negatives are written up
  as clean companions (`TASK_D_PREREGISTRATION.md` §8 item 3), and dead
  directions stay dead (`CLAUDE.md` hard rules).

---

## 9. Sequencing — what this unlocks

1. **This design → build (Stage 1-equivalent: harness + smoke gate,
   including the extended blank-out/composition-purity test, §4.3/§5.2) →
   audit by a fresh-context agent (Build → Audit → Run, per `CLAUDE.md`) →
   Wave −1 → Wave 0 → Wave A → B-probe → (conditional) Wave B.**
2. **If CONFIRM (bespoke harness) →** production-block secondary check
   (§4.3), required before the strongest publication framing (§8); then,
   symmetric to Task D/E's own sequencing, a reasoning-transfer step on
   this architecture (does a DeltaNet-native rank-K state, trained under
   this bottleneck, transfer to real data the way Task D→E's sequencing
   intends for the matrix-native line) — explicitly out of scope for this
   design, a future doc.
3. **Deferred generality questions (explicitly not folded into the
   primary gate, matching Task D/E's own P=1-first, C2-parallel-not-gating
   precedents):** multi-head (`H>1`) sweep — reintroduces the head-
   decomposition pigeonhole (§4.2) as its own controlled question, not a
   confound to avoid (first probe pre-registered in the Reserve wave,
   F11/§6.4); `n_h>1` DeltaProduct-style transition-rank interaction
   with state-rank capacity (§2 item 3); the fixed-subspace C-i variant as
   a deliberate architectural-ceiling study in its own right (distinct
   from, not a substitute for, the §3.5 causal force-rank arm); a
   whole-trajectory rank constraint (§3.5 table row 3), valid only with
   the F5 alignment confound closed by construction.
4. **If FALSIFY →** publish as the companion negative described in §8;
   redirect any DeltaNet-family compute back to the matrix-native Chapter 2
   line, which remains alive independent of this result.

---

## 10. Open questions (explicit, not resolved by fiat)

- Exact `fla-org` DeltaNet API for state extraction/seeding
  (`output_final_state`/`initial_state` or actual names), chunk-size
  options, `conv_size` default, and `n_h`/head-count configuration —
  resolved at the mandatory pre-Wave-−1 F15 checkpoint (§6.4), the first
  action of the build phase (package is not currently installed in this
  repo; no install steps yet in `H100_SETUP.md`).
- The two-kernel split's real overhead (one extra kernel invocation + one
  `d×d` eigh per forward pass) and the extracted-state round-trip's
  numerical fidelity — measured at the F15 checkpoint and Wave −1, with
  the F13 decision rule pre-committing the action at each outcome
  (expected ≈1×, but priced, not assumed).
- Whether d=64 or d=128 is the healthier primary cell for this
  architecture's write mechanism — Wave 0's decision, not assumed here
  (§6.1's a priori lean is a hypothesis, explicitly flagged as such in the
  attack section, §7 item 6).
- Whether multi-head configurations satisfy the joint per-head bound
  `Σ_head rank(S_T^{(head)}) ≥ K` by concentrating rank in few heads or
  spreading it — first data from the Reserve wave's H∈{2,4} probe
  (F11, §6.4); a full multi-head study is deferred (§9).
- Whether SGD's learned `W_k, W_v, β`-gating converges toward the clean
  orthonormal/`β≈1` regime (§3.6) or a messier, higher-rank-on-the-entity-
  subspace solution — the actual empirical content of the M1-equivalent
  test, genuinely open a priori.
- Whether a softer, differentiable rank-forcing surrogate (flagged as the
  Reserve-budget fallback, §6.4) is needed if B-probe shows death, and if
  so, what that surrogate should be — not designed here, deliberately
  deferred until the calibration data says it's needed.

---

## 11. Reproducibility pointers

- This design: `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md`.
- Builds on (read, not modified): `matrix-thinking/chapter2/{
  TASK_D_PREREGISTRATION.md, NEXT_EXPERIMENT_DESIGN.md, TASK_E_FINDINGS.md,
  STAGE0_DESIGN.md, rank_utils.py, eigen_utils.py, task_d.py, model_v4.py,
  task_e.py}` — reuse targets for the generator (`task_e.py`, verbatim),
  the rank-metric machinery (`rank_utils.py` — `truncate_to_rank` applied
  unmodified at the §3.5 two-kernel truncation point — and
  `eigen_utils.py`), and the subspace-restriction analysis pattern
  (`analyze_zdump.py`, generalized to this architecture's `S_T`).
- Internal prior research reused: `research/validated-cheap-matrix-ops.md`
  (delta-rule/Householder algebra, already verified independently of this
  design), `research/task-d-novelty-july2026.md` (DeltaNet as a related-work
  citation, now the object under test).
- Novelty check backing §2: run inline before this document (2026-07-01, per
  the task brief); no separate file written, findings folded in directly,
  matching Task E's own house convention for inline research+attack passes.
- Attack-round report backing the revision-2 changelog: independent
  adversarial review, 2026-07-01 (findings F1–F17 indexed in the header
  changelog; BUILD WITH FIXES verdict, all findings addressed in place).
- Next: F15 API-verification checkpoint (§6.4, first action) → build the
  bespoke single-layer/single-head harness (§4.3) + smoke gate (extended
  blank-out/composition-purity, §4.3/§5.2; C15 negative test to
  completion; C13/C14 if the production-block path is used) + the §6.5
  operational harness → audit by a fresh-context agent → Wave −1 on the
  cluster (F13 decision rule) → Waves 0/A/B-probe/B per §6.4.

---

## 12. Results — Waves −1 and 0 (2026-07-02)

**Status: Wave −1 3/4 cells complete, 1 still running (`d128_K64_fr64`, the
force-rank cell). Wave 0 10/12 cells complete, 1 running
(`d128_K32_frN_s1`), 1 not yet started (`d128_K32_frN_s2`). Wave A already
launched (`--primary-d 64 --primary-k 32`, 13-cell manifest, confirmed live
on GPUs 0–2 via `ps aux`) — per this project's bounded/gated-wave discipline
(§6.5), this is a documented-in-code decision (`run_deltanet_sweep.py`'s own
comments, see §12.2 below), not a formally-written decision memo; this
section is that memo, written retroactively from the artifacts.** Every
number below is read directly from the result JSONs (`checkpoints[-1]`,
top-level run metadata) or from `run_deltanet_sweep.py`/`model_dn.py`'s own
measured-constants comments on the box, not from memory or the task brief
that seeded this write-up. Archived locally:
`experiment-runs/2026-07-02_deltanet_waves/{wave-1,wave0}/` (15 JSONs + logs,
pulled from `youthful-indigo-turkey:/home/nvidia/chapter2/deltanet/results/deltanet/{wave-1,wave0}/`).

### 12.1 Wave −1 — timing + instability calibration

4 cells launched (§6.4's manifest): 2 unconstrained (`frN`) + 2 force-rank
at `k=K` (`fr32` at d=64, `fr64` at d=128), 1 seed each, `trunc_impl=svd_lowrank`
throughout (the F18-advisory fallback was used from the start of Wave −1,
not discovered mid-wave — see §12.2).

| Cell | steps (of manifest) | `wall_s` | s/step | `n_skipped_steps` | skip rate |
|---|---|---|---|---|---|
| d64 K32 frN (unconstrained) | 10,000/10,000 (complete) | 615.5 | 0.0616 | 0 | 0.00% |
| d64 K32 fr32 (force-rank, k=K) | 10,000/10,000 (complete) | 5,145.9 | 0.5146 | 2 | 0.02% |
| d128 K64 frN (unconstrained) | 12,000/12,000 (complete) | 1,484.6 | 0.1237 | 0 | 0.00% |
| d128 K64 fr64 (force-rank, k=K) | **6,000/12,000 — NOT complete, still running** | 7,344.4 (so far) | 1.2241 (so far) | 3 | 0.05% |

The timing constants the task brief cited check out exactly: d64 frN
≈616s (615.5), d64 fr ≈5,146s (5,145.9), d128 frN ≈1,485s (1,484.6) at the
12K-step tier. **The `d128_K64_fr64` cell has NOT finished** — it is at
6,000 of 12,000 steps (50%) and still running on the box (PID confirmed via
`ps aux`, `--internal-timeout 22520`); the `wall_s`/skip-rate figures above
for that cell are a partial read, not a final one.

**Skip rate: the brief's "≤0.02% at both d" is exact at d=64 (0.02%,
2/10,000 steps) but not quite at d=128 — the (partial) measured rate there
is 0.05% (3/6,000 steps), 2.5× the d=64 figure, though still four orders of
magnitude under the F13 stop threshold (0.1%) and under the eigh path's
measured 30–93% (see below).** Both are correctly described as trivial next
to eigh: `model_dn.py`'s own audit-round-1 measurement recorded eigh
force-rank skip rates of 30.00%/71.25% (skip@150/skip@400 steps) at d=64
and 82.00%/93.00% at d=128 — the exact 30–93% range the brief cited,
confirmed in `model_dn.truncate_to_rank_svd_lowrank`'s docstring and
`run_deltanet_sweep.py`'s `_PER_STEP_S` comment block. svd_lowrank measured
**zero** skips over 400 probe steps at both d before Wave −1 launched.

### 12.2 The F13 wall-clock ratio: measured 8.4–9.9×, not ≤2× — a correction to the brief, with the actual repricing traced

**The brief's "F13 VERDICT: proceed with svd_lowrank, d=64 primary" is
correct in outcome but glosses over the actual ratio measured, which lands
in the design's own ">5× → STOP, diagnose, re-price" bucket, not the "≤2× →
proceed as priced" bucket.** Computed directly from the Wave −1 `wall_s`
figures above (force-rank s/step ÷ unconstrained s/step, same cell count):

- **d=64: 0.5146 / 0.0616 = 8.35×**
- **d=128 (partial, in-progress fr cell): 1.2241 / 0.1237 = 9.90×** — likely
  to move as the fr64 cell finishes, but not by enough to plausibly drop
  under 5× from a current 9.9× reading.

Both exceed the pre-committed 5× ceiling stated in §6.4's decision-rule
table, literally read. **What actually happened, traced through
`probe_trunc.py` and `run_deltanet_sweep.py`'s own comments (no separate
decision memo exists on the box — this section is the first place the
reasoning is written down end-to-end):** a dedicated timing+skip-rate probe
(`probe_trunc.py`, run *before* Wave −1's manifest was built) measured
per-step costs of 46ms (d64 unconstrained) / 468ms (d64 svd_lowrank
force-rank, ≈10.2×) and 75ms (d128 unconstrained) / 1,132ms (d128
svd_lowrank force-rank, ≈15.1×) — i.e., the ratio was already known to be
double-digit *before* Wave −1 ran, not discovered as a surprise mid-wave.
Wave −1's own measured ratios (8.35×/9.90×) are close to, and slightly
better than, those probe-stage estimates. `run_deltanet_sweep.py`'s own
comment block ("PROMINENT d-SELECTION FLAG") converts this into a wave-by-
wave GPU-hour budget at each `d`: **d=64-primary path ≈77 GPU-h total
(fits the ≤120 ceiling); d=128-primary path ≈199 GPU-h (blows it)** — and
selects `--primary-d 64` on that basis. This *is* a "diagnose and re-price"
response in substance (a measured overhead, an explicit re-priced total
budget, a d-selection made on cost grounds) — it just happened via
`probe_trunc.py` + code comments ahead of Wave −1, rather than as a
written STOP-then-resume cycle triggered by Wave −1's own numbers landing
in the >5× bucket, which is what §6.4's table literally describes. **Net
assessment: the substance of F13's gate was honored (measured, priced,
decided on cost) but the specific procedural sequencing and the ">5×"
threshold's literal STOP trigger were not — this is a real, if narrow,
process deviation, logged here per this project's documented-deviation
convention (`CLAUDE.md`) rather than left implicit.** It does not, on its
own, cast doubt on any Wave 0 result below (Wave 0 ran the *unconstrained*
arm only, which pays none of this overhead — `frN` skip rate and per-step
cost are identical in Wave 0 to Wave −1's calibration figures above); it
matters for costing Wave B (the force-rank straddle grid), which has not
launched yet.

### 12.3 Wave 0 — extended-budget transition calibration: unconstrained arm saturates at all four `(d,K)` cells

12-cell manifest (§6.4): `d64 K∈{16,32}`, `d128 K∈{32,64}`, 3 seeds each,
2.5× tier-default steps (25,000 at d=64, 30,000 at d=128), unconstrained
arm only. 10/12 complete as of this pull; `d128_K32_frN_s1` is running
(16,000/30,000, 53%) and `d128_K32_frN_s2` has not started.

| Cell (all seeds) | steps | `n_skipped_steps` | `recovered_frac@0.9` (every hop 1–7, 21) | entity-subspace eff. rank | whole-matrix eff. rank | key-gram deviation |
|---|---|---|---|---|---|---|
| d64 K16 (s0,s1,s2) | 25,000/25,000 ×3 | 0 | **1.0000 at every hop, every seed** | 15.99998–16.00000 | 16.00003 | 0.0052–0.0061 |
| d64 K32 (s0,s1,s2) | 25,000/25,000 ×3 | 0 | **1.0000 at every hop, every seed** | 31.99993–31.99994 | 31.99998–32.00000 | 0.0053–0.0056 |
| d128 K32 (s0 complete; s1 running 16K/30K; s2 not started) | 30,000/30,000 (s0); 16,000/30,000 (s1) | 0 | **1.0000 at every hop, both readable seeds (s0 complete, s1's 16K-step partial read is already at ceiling too)** | 31.99999 (s0) / 31.99941 (s1, partial) | 32.00016 (s0) / 31.99957 (s1) | 0.0092 (s0) / 0.0124 (s1, partial) |
| d128 K64 (s0,s1,s2) | 30,000/30,000 ×3 | 0 | **1.0000 at every hop, every seed** | 63.99999–64.00000 | 64.00015 | 0.0092–0.0097 |

**No late-transition regime at all — every completed cell (and even the
one 53%-through partial read, `d128_K32_s1`) is already at
`recovered_frac@0.9 = 1.00` at all 8 tested hops (1,2,3 in-distribution;
4,5,6 held-out; 7,21 held-out-extra).** This is a materially different
outcome from every other line of work in this program: the bespoke
Chapter 2 encoder (`STAGE0_DESIGN.md` §15.1/§15.3) plateaus at
`d=64, K=32` after 150,000 steps at cos 0.45–0.49 with `recovered_frac@0.9`
identically 0.0 in all 3 seeds, with transitions onset scaling
superlinearly and, at that exact `(d,K)`, landing at 24,000–44,000 steps
per the `ext_budget` 60K-step round (`EXPERIMENT_LOG.md`, "Stage 0 — Wave
A + extended-budget arm," `STAGE0_DESIGN.md` §13.3). DeltaNet's own delta
rule reaches ceiling at the same `(d,K)` operating point inside a 2.5× tier
budget (25,000 steps) with zero seeds showing anything but immediate
saturation. Mechanistic reading, consistent with §3.6's architecture-native
positive control: the delta rule's update *is* an associative-memory write
by construction (the proof in §3.6 shows the exact orthonormal-key/`β=1`
solution is reachable by the recurrence's own dynamics with zero training),
so this task is easy for DeltaNet at these budgets in a way it is not for
an attention-based set encoder that has to discover a comparable operator
from a much less directly-aligned parameterization.

### 12.4 The rank result: entity-subspace rank = whole-matrix rank = K, exactly, no inflation

Cross-referencing the two rightmost numeric columns of §12.3's table: at
every converged cell, **entity-subspace-restricted effective rank and
whole-matrix effective rank agree to within measurement noise (4th
decimal place) and both sit at K, not d** — 16.00/16.00 at K=16, 32.00/32.00
at K=32 (both d), 64.00/64.00 at K=64. This is the cleanest version of the
Chapter 2 M1-equivalent result the program has produced: unlike the bespoke
encoder (`TASK_E_FINDINGS.md` §4/§9), where whole-matrix rank inflates to
≈d (an unconstrained, near-full-rank orthogonal complement the loss never
touches) and only the entity-subspace-*restricted* rank tracks K, DeltaNet
shows **no inflation to distinguish** — the two numbers are the same
number. Mechanistic reading: the delta rule's update `S_t = S_{t-1}(I -
β_t k_t k_t^T) + β_t v_t k_t^T` only ever writes within `span({k_j})` by
construction (§3.4's algebra) — there is no orthogonal complement for an
optimizer to leave untouched in the first place, so the whole-matrix-vs-
restricted distinction that Task D/E needed §9's subspace-restriction fix
to resolve is moot here by architecture, not by a fortunate optimization
outcome.

**Effective-key Gram deviation** (`‖K_eff^T K_eff − I‖_F`, the F4 standing
diagnostic, §6.3) is small and consistent across the 10 complete Wave 0
cells: **0.0052–0.0097** (low: d64 K16 s2, 0.0052; high: d128 K64 s0,
0.0097), plus one still-converging partial read (`d128_K32_s1` at 16K/30K
steps, 0.0124, likely still falling). This is the same order of magnitude
as the brief's "0.006–0.010" summary, with the precise range running
slightly wider at both ends once every cell is checked individually rather
than approximated. Learned keys sit close to, but not exactly at,
orthonormality — consistent with §3.6's open question (does SGD converge
toward the clean `β≈1`/orthonormal-key regime) being *answered in the
affirmative to a good approximation*, not exactly.

### 12.5 Primary-cell decision and Wave A

**`--primary-d 64 --primary-k 32`** — selected on the cost/stability
grounds traced in §12.2 (3× cheaper total wave budget than d=128, and the
only d where the force-rank arm's per-step overhead was calibrated before
committing), not because Wave 0 discriminated between the two `d` values
on trainability (§12.3 shows both are perfect at every tested cell — Wave 0
cannot and does not distinguish "healthier" in the trainability sense
§6.1 anticipated; the deciding signal is entirely cost/stability from
Wave −1). **Wave A (13 cells) launched 2026-07-02**, confirmed running on
GPUs 0–2 via `ps aux` (`d64 K8`/`d64 K16` unconstrained cells observed
in-flight at the time of this write-up).

### 12.6 Cross-reference: the unconstrained arm is saturated — the causal claim now rests entirely on Wave B

**Perfect recovery at every Wave 0 cell means the unconstrained arm has no
further discriminating power.** There is no partial-recovery regime, no
K-dependent degradation, and no `d`-dependent degradation left to measure
in this arm — every M1/M2-equivalent question this design can still ask
must come from the force-rank staircase (B-probe → Wave B, §6.4), exactly
as §6.4 anticipated when it wrote "no late-transition regime at all" as one
possible Wave 0 outcome. This raises the stakes on Wave B specifically:
Wave −1's `fr=K` cells (§12.1) are a reassuring but not sufficient preview
— d64's `fr32` cell (k=K=32, the "provably sufficient" boundary) is
climbing steadily (`recovered_frac@0.9` at h=1: 0.378 → 0.408 → 0.422 →
0.450 → 0.451 over its 10,000 steps, mean_cos 0.868 → 0.885) but has not
reached ceiling within the tier-default budget, and shows key-gram
deviation two to three orders of magnitude higher than the unconstrained
arm (3.56 at d64, 4.91 at d128-partial, vs. 0.005–0.010 unconstrained) —
i.e., forcing rank measurably degrades key orthonormality on top of
slowing training, a second mechanism (not just wall-clock) that could
confound a k=K-1 collapse with an under-training artifact if Wave B is not
budgeted generously. **What would falsify H_DN at this point:** (b) fails
if the B-probe/Wave B force-rank grid shows no accuracy step between
`k=K-1` and `k≥K` even at extended budget (mirroring Task E's fr≥K "dead"
outcome would be a different failure mode — here §12.1 already shows life,
not death, at `k=K`, so a Wave B failure would more likely look like "no
threshold, gradual degradation at every k" than "flatline everywhere");
(c) — the compositional-transfer extension — is not yet testable, since it
depends on a converged force-ranked state existing at all. Wave 0's result
is necessary-but-not-sufficient for CONFIRM: it establishes the
architecture-native positive control is realized in practice (§3.6, §12.3,
§12.4) but says nothing yet about whether the recruited rank is *causally
necessary* — that is Wave B's question alone, unweakened and unanswered by
this section.

> **Superseded (2026-07-02, same day, later):** B-probe showed the
> train-time force-rank instrument is unreadable (fr31/fr32/fr33 collapse
> identically — §12.8.1), Wave B is moot, and the causal claim was closed
> instead by the pre-registered F18-(c) eval-time truncation control
> (§12.8.3–12.8.5): CONFIRMED, razor cliff at k=31→32.

### 12.7 Compute

Serial-sum `wall_s`, **as measured so far** — 2 of the 15 archived runs are
partial reads (`wave-1/d128_K64_fr64` at 50%, `wave0/d128_K32_frN_s1` at
53%), so these totals will grow as those cells complete and
`wave0/d128_K32_frN_s2` (not yet started) is added:

- **Wave −1: 4 runs, 14,590.4s = 4.05 GPU-h** (partial; the `fr64` cell's
  eventual full-run total is not yet known).
- **Wave 0: 11 runs (of 12 planned), 21,602.5s = 6.00 GPU-h** (partial;
  excludes the not-yet-started `d128_K32_frN_s2`).
- **Combined so far: 10.05 GPU-h.**

Raw archive: `experiment-runs/2026-07-02_deltanet_waves/{wave-1,wave0}/`
(15 result JSONs + orchestrator logs, pulled from
`youthful-indigo-turkey:/home/nvidia/chapter2/deltanet/results/deltanet/{wave-1,wave0}/`
2026-07-02). Source: `/home/nvidia/chapter2/deltanet/{run_deltanet.py,
run_deltanet_sweep.py, model_dn.py, deltanet_core.py, probe_trunc.py}` on
the box (not yet mirrored into this repo's git tree as of this write-up —
flagged for a follow-up sync, out of scope for this results section).

### 12.8 B-probe + the eval-time truncation staircase — the causal verdict (2026-07-02)

**Status at write-up: B-probe 7 final reads in hand (fr31 s0/s1/s2 and
fr32 s0/s1/s2 complete at 20,000 steps on the box; fr33 s1 complete at
20,000 steps in a preserved archive copy — see the data-integrity note
below), fr33 s0/s2 re-runs in flight at 6,000/20,000. Eval-truncation
staircase COMPLETE (3 seeds).** Every number below is read from the result
JSONs (`experiment-runs/2026-07-02_deltanet_waves/{waveBprobe,eval_trunc}/`,
pulled from `youthful-indigo-turkey:/home/nvidia/chapter2/deltanet/results/
deltanet/`), not from memory.

**Data-integrity note (logged, not hidden):** fr33_s1 completed at 20,000
steps this morning (read and archived at that time), but the box copy was
subsequently overwritten by a relaunch of all three fr33 seeds (currently
at 6,000/20,000, `complete: false`) — a resume-path collision, the exact
Stage 0 MAJOR-2 failure mode §6.5's naming rule was written against
(here the collision is same-name re-launch over a valid result, not
variant collision). The complete read is preserved as
`waveBprobe/wBprobe_bespoke_h1_d64_K32_fr33_s1_tsvd_lowrank.COMPLETE_pre-relaunch-overwrite.json`
in the local archive and is the fr33 datum used below. The two in-flight
fr33 re-runs' step-6,000 checkpoints (h=1 `recovered_frac@0.9` = 0.416,
0.414, 0.454) are already indistinguishable from fr31/fr32's own
step-6,000 values (0.413–0.451), so the relaunch is burning GPU to
re-measure a number that three arms have already agreed on.

#### 12.8.1 B-probe: fr31, fr32, fr33 collapse IDENTICALLY — the train-time force-rank arm is unreadable at this operating point

All cells `d=64, K=32`, 20,000 steps, `trunc_impl=svd_lowrank` (the F13/F18
fallback), final checkpoint:

| cell | h=1 `rec@0.9` | h=1 `mean_cos` | h=1 cos p10/p50/p90 | h≥2 `rec@0.9` (all of 2–7, 21) | h=2 `mean_cos` | entity-subspace rank | key-gram dev |
|---|---|---|---|---|---|---|---|
| fr31 s0/s1/s2 | 0.5204 / 0.5208 / 0.5065 | 0.8959 / 0.8967 / 0.8935 | ≈0.81 / ≈0.90 / ≈0.97 | **0.0000** | 0.0080 / −0.0261 / −0.0311 | 30.62–30.67 | 3.24–3.25 |
| fr32 s0/s1/s2 | 0.5208 / 0.5165 / 0.5565 | 0.8967 / 0.8969 / 0.9038 | ≈0.81 / ≈0.90 / ≈0.98 | **0.0000** | 0.0245 / −0.0216 / −0.0371 | 31.50–31.57 | 3.20–3.33 |
| fr33 s1 (preserved complete read) | 0.4936 | 0.8931 | 0.81 / 0.90 / 0.97 | **0.0000** | −0.0402 | 31.46 | 3.39 |

Skip rates 0.010–0.020% throughout (fine); all seeds still on a slow climb
at 20K (fr31_s0: 0.367 → 0.459 → 0.520 over 2K → 10K → 20K), none near
ceiling. **The decisive pattern: there is NO step at k=K.** k=31 (provably
insufficient), k=32 (exactly sufficient — §12.4 shows the unconstrained
solution has rank exactly 32, so a rank-32 truncation of it would be a
no-op), and k=33 (strictly more than sufficient) are statistically
indistinguishable: h=1 ≈ 0.49–0.56 with mean_cos ≈ 0.89–0.90, and h≥2
exactly 0.0000 with mean_cos at noise (±0.04) in every seed of every arm.
Since the constraint at k≥32 is satisfiable by a solution the recurrence
itself can reach with zero training (§3.6) and Wave 0/A showed
unconstrained training finds it within ~1,000 steps, a collapse that is
*identical at k≥K and k=K−1* cannot be a rank effect — it is the
**training-time interference of the stochastic `svd_lowrank` projection
itself** (fresh Gaussian sketch per call, biased low on the near-degenerate
spectra this task produces by design — exactly F18's pre-registered
mechanism, and its consequence (a) — "a marginal collapse at k=K−1 must
not, by itself, be read as evidence the model needs rank K" — turned out
to be not a caveat but the whole story). Corroborating diagnostics: forced
arms' key-gram deviation is 3.2–3.4 (vs. 0.005–0.010 unconstrained, §12.3)
and the h=1 per-item cosine distribution is *unimodal centered at ≈0.90*
(p10≈0.81, p50≈0.90, p90≈0.97) — a uniformly degraded state whose
`rec@0.9` ≈ 0.5 is literally the mass above a median that sits at the
threshold. **The fr31 "0.51" number is a tolerance-slack artifact of
scoring a uniformly-broken state at τ=0.9, not a rank measurement.**

**Gate decision (per §6.4's B-probe gate): Wave B (the full train-time
force-rank straddle grid) is MOOT and is not launched.** The train-time
instrument cannot distinguish k=K−1 from k=K+1 at the primary cell; a
15–20-run grid would re-measure the same interference at ~0.5 s/step
(§12.2's 8–10× overhead). This mirrors the "B-probe shows death" branch
§6.4 pre-registered — except the death is now *positively attributed* to
the truncation implementation (k≥K arms die identically), not left as an
unexplained instability. The causal question moves to the eval-time
instrument below, which §6.4's F18 advisory consequence (c) pre-registered
as mandatory before any Wave B verdict.

#### 12.8.2 Why eval-time truncation is decisive HERE (the M3 objection, addressed head-on)

Task D's pre-registration (C1) and TASK_D_WRITEUP §M3 rejected post-hoc
truncation as "uninformative" — correctly, in that context: truncating a
state whose learned rank *exceeds* the requirement only removes slack, and
a flat curve conflates "instrument insensitive" with "mechanism absent"
(the CODI lesson). **That objection does not apply here, for a reason
Wave 0/A established empirically: the learned DeltaNet state has
entity-subspace rank = whole-matrix rank = K exactly (31.9993–31.9994 of
K=32 in the three states truncated below; §12.4's no-inflation result).
There is no slack.** Every one of the K retained directions carries
provably-necessary structure (§4.1's bound: exact recovery of K injective
bindings through a matrix-vector readout requires rank ≥ K), so removing
the smallest eigendirection is guaranteed to remove load-bearing
structure *if and only if* the rank-K bound is really binding — which is
exactly the hypothesis under test, and both of its predictions (cliff at
k=K−1→K; no effect at k≥K) are falsifiable in the same table. The
eval-time projection is `rank_utils.truncate_to_rank` (deterministic
eigh, the optimal rank-k approximation — none of svd_lowrank's stochastic
bias), applied read-only to a converged unconstrained state, so none of
the train-time interference of §12.8.1 is in the loop.

**Provenance caveat (honest):** no archived run carries an S_T dump
(`--save-z` exists in `run_deltanet.py` but no sweep cell passed it, and
no state_dict is saved anywhere — verified programmatically across all
four waves' JSONs), so the staircase states come from a fresh exact-recipe
retrain of the primary unconstrained cell (`analyze_eval_truncation.py`,
same config/steps/lr/seeds as Wave A's `d64_K32_frN`; 20,000 steps, seeds
0/1/2). The retrained states reproduce the Wave A cell's signature
exactly — `recovered_frac@0.9` = 1.0 at every hop, entity-subspace rank
31.9993–31.9994 — so they are the same object Wave A measured, not a new
variant.

#### 12.8.3 The eval-truncation staircase: a razor cliff at k=31→32, exactly where the bound says

`recovered_frac@0.9`, mean of 3 seeds (per-seed spread ≤0.003 at every
cell; n = 163,840 query-item scores per (h,k,seed) cell — 20 batches × 256
examples × 32 queries):

| h \ k | 24 | 28 | 30 | 31 | **32** | 33 | 34 | 36 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.7470 | 0.8732 | 0.9362 | 0.9681 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.5262 | 0.7509 | 0.8738 | 0.9366 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 3 | 0.3641 | 0.6430 | 0.8126 | 0.9045 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.2476 | 0.5469 | 0.7561 | 0.8738 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 5 | 0.1660 | 0.4619 | 0.6981 | 0.8408 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 6 | 0.1106 | 0.3883 | 0.6460 | 0.8105 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 7 | 0.0733 | 0.3219 | 0.5931 | 0.7788 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 21 | 0.0001 | 0.0073 | 0.1051 | 0.3412 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

Every prediction of the causal-rank hypothesis lands:

- **Perfect at k ≥ K = 32, at every hop including h=21, all seeds, to four
  decimal places.** These are real eigh truncations (k=32,33,34,36 < d=64,
  so directions above k of the 64-dim state are genuinely removed) — the
  discarded ambient directions carry nothing, confirming §12.4's
  no-inflation result behaviorally, not just spectrally.
- **Degraded at exactly k=31, worse monotonically at lower k, amplified by
  h — with h=21 the sharpest step, as pre-registered:** at h=1 the cliff is
  0.9681 → 1.0000; at h=21 it is 0.3412 → 1.0000, a 0.66 jump from
  removing ONE direction of 32.
- **The degradation is quantitatively the theory's, not merely
  qualitatively "worse."** For this task's construction (orthonormal keys,
  values = same frame permuted by a single K-cycle), dropping exactly one
  entity mode from the ideal operator predicts
  `rec@0.9(h) = (K−h)/K` — a bimodal all-or-nothing distribution (an item
  fails iff the dropped entity lies on its h-step orbit), NOT a uniform
  decay (derivation + synthetic exact-match unit test in
  `analyze_eval_truncation.py`). Measured k=31 column vs `(K−h)/K`:
  0.9681 vs 0.9688 (h=1), 0.9366 vs 0.9375, 0.9045 vs 0.9063, 0.8738 vs
  0.8750, 0.8408 vs 0.8438, 0.8105 vs 0.8125, 0.7788 vs 0.7813, 0.3412 vs
  0.3438 (h=21) — agreement to 0.001–0.003 at every hop. The h=1 row also
  matches the m-modes-dropped generalization `(K−m)/K, m=K−k` across the
  whole grid: k=30 → 0.9362 vs 0.9375, k=28 → 0.8732 vs 0.8750, k=24 →
  0.7470 vs 0.7500. The trained state doesn't just need rank K — its
  eigendirections align with the K entity modes nearly one-for-one.

#### 12.8.4 Per-item distribution at k=31, h=1 — the three-way comparison

Measured (all 3 seeds, n=163,840 each): mean_cos 0.9698–0.9706, **p10 =
p50 = p90 = 1.0000**, `frac(cos>0.9)` = 0.9679–0.9683, `frac(cos<0.1)` =
0.0275–0.0284. Against the two theories and the train-time arm:

- **Entity-aligned theory** (one whole mode of 32 dropped): bimodal,
  31/32 = 0.9688 of items at cos≈1, 1/32 = 0.0313 at cos≈0; mean 0.9688.
  → **Matches: measured is bimodal** (2.8% catastrophic, the rest
  perfect), mean 0.970, `rec@0.9` 0.968.
- **Isotropic theory** (Task E's `sqrt(1−1/K)` convention — the dropped
  direction spread evenly over all items): EVERY item at cos =
  `sqrt(31/32)` = 0.9843 → `rec@0.9` ≈ 1.0 at every h, no catastrophic
  items. → **Refuted**: it predicts p10 = 0.9843 and zero catastrophic
  mass, and cannot produce the k=31, h=21 value (0.3412) from per-item
  decay that clears τ=0.9 at h=1.
- **fr31 train-time arm** (0.51 at the same nominal rank): unimodal
  centered at the threshold (p50 ≈ 0.90) with neither a perfect mass nor a
  catastrophic mass at h=1 — a uniformly-degraded state. Eval-truncating a
  healthy state to the same rank 31 gives 0.9681, i.e. **the train-time
  arm sits ~0.45 below what rank-31 capacity actually supports.** The gap
  between 0.51 and 0.97 at identical k is the direct measurement of how
  much of the fr-arm's collapse is optimizer interference rather than rank.

#### 12.8.5 Verdict and the cross-architecture methodological finding

**H_DN branch (b) — causal necessity — CONFIRMED at the primary cell, via
the eval-time instrument:** the learned `S_T` at `d=64, K=32` cannot lose
its smallest eigendirection without losing recovery on exactly the
predicted 1/K of queries (compounding to 2/3 of queries at h=21), and
loses nothing when any of its non-recruited ambient directions are
removed. Combined with §12.4 (SGD recruits rank exactly K, no inflation)
this closes the causal loop Wave B was designed for, without Wave B:
rank K is recruited, aligned with entity modes, and every direction of it
is load-bearing.

**The train-time finding stands on its own as a cross-architecture
methodological result:** hard rank projection inside the training loop at
k near the operating rank K breaks SGD *even when the constraint is
provably satisfiable* (k≥K arms die identically to k<K), now observed in
two unrelated architectures — Task E's bespoke attention-set encoder
(M4_E: fr arms dead even at fr≥K, `TASK_E_FINDINGS.md`) and DeltaNet's
delta-rule recurrence (this section) — under two different truncation
implementations (eigh with its unstable degenerate-spectrum backward
there; stochastic svd_lowrank with its sketch-noise bias here, per F18).
The common factor is the by-design near-degenerate spectrum of the target
solutions: any spectral cut at k≈K keeps re-deciding which of K
near-equal modes to drop, and the optimization never settles. Eval-time
truncation of the converged state, by contrast, is clean, deterministic,
and — when the learned rank equals the requirement exactly — decisive.
Pre-registration implication for future rank-necessity designs: put the
causal weight on (i) an exact-rank no-inflation result plus (ii) eval-time
optimal truncation, and treat train-time forcing as a secondary arm
expected to be confounded near k=K.

#### 12.8.6 Compute and reproducibility

- B-probe, measured so far: fr31 ×3 (31,552 s) + fr32 ×3 (29,960 s) +
  fr33_s1 preserved (10,005 s) = 71,517 s = **19.87 GPU-h** (the two
  in-flight fr33 re-runs will add ~5.5 GPU-h that this analysis does not
  need).
- Eval-truncation staircase: 3 retrains + staircases = 3,307 s = **0.92
  GPU-h** (GPU 7, no contention with the Bprobe/Stage-G runs on GPUs 0–6).
- Script: `matrix-thinking/deltanet/analyze_eval_truncation.py`
  (rerunnable; `--smoke` runs the synthetic theory-validation gate,
  `--dir` cross-references any directory of force-rank result JSONs;
  mirrored to the box at `/home/nvidia/chapter2/deltanet/`). Archive:
  `experiment-runs/2026-07-02_deltanet_waves/{waveBprobe,eval_trunc}/`
  (JSONs + run logs + the preserved pre-overwrite fr33_s1).

---

### 12.9 Reserve wave — multi-head generality (F11, 2026-07-03): every head recruits full rank, the H=1 qualifier lifts

The §6.4 Reserve wave's pre-registered H∈{2,4} per-head probe is
complete: **6/6 cells** (`d=64, K=32`, unconstrained, H∈{2,4} × 3
seeds), 0 failures, 0 skipped steps. Per-head
`entity_subspace_effective_rank_mean` is **31.9994–31.9998 (≈32.0, of
32) in every head, at both H=2 and H=4, across all 3 seeds** — not a
subset of heads, not a partial recruitment in any one head. The joint
sum (`joint_entity_subspace_effective_rank_sum_mean`) lands at
**63.999/64 (H=2)** and **127.999/128 (H=4)** — i.e. `H × K`, not `K`
split across heads. `rec@0.9 = 1.000` at h=1 in every cell (per-head
key-Gram deviation 0.005–0.02, consistent with the single-head Wave 0
band, §12.3).

**This directly answers §4.2's own open question and lifts §8's F11
qualifier.** The primary gate's H=1 result was never in doubt about
whether *a* head can recruit and depend on provably-necessary rank —
the open question was whether a multi-head architecture would instead
**distribute** the K-item binding load across heads (the
position-decomposition-style pigeonhole §4.2 already closed
state-internally, but not across heads) once concentrating it in one
head was no longer the only option. It does not: SGD **redundantly
replicates** full-rank, full-K structure in every head rather than
dividing K across H. §8's "production fast-weight *mechanism*, not
production *configurations*" qualifier — required because the primary
gate was H=1-only — is lifted for H∈{2,4}; a genuine multi-head study
beyond these two probe points remains future work (§9 item 3), but the
specific pigeonhole this wave was pre-registered to close (spreading,
not just holding, the K-item bound across heads) is closed empirically,
not just architecturally.

**Operational lesson, worth keeping:** the first launch batch ran 2
runs per GPU (to save wall-clock) and hit a timeout before completing —
the relaunch went solo-paced (1 run per GPU) and all 6 cells completed
cleanly. `H=4`'s larger per-step cost (4× the QKV/state work of `H=1`
at fixed `d_state`) makes it more timeout-sensitive to GPU contention
than the single-head waves this design's earlier sections ran; future
multi-head sweeps at `H≥4` should default to solo-paced launches rather
than assuming the single-head 2-per-GPU density still fits.

**Compute:** 6 runs, `d=64, K=32`, H∈{2,4} × 3 seeds, 25,000 steps each.
**Archive:** `experiment-runs/2026-07-02_deltanet_waves/reservemh/` (6
run JSONs + `AGGREGATE.json` + `SUMMARY.txt`, 980 KB, verified against
the box byte-for-byte). Source on box:
`youthful-indigo-turkey:/home/nvidia/chapter2/deltanet/results/deltanet/waveReserveMH/`.
