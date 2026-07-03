# ATTACK: Task Shortcuts for Chapter 2 (Task A' vs Task B vs the P-slot Architecture)

Target: can the model hit high accuracy on Task A' (K parallel entities, joint
readout) or Task B (K-way BFS/reachability) WITHOUT ever holding K items at
rank > 1 in a single matrix slot? Not attacking measurement/interpretability
(see the sibling file `ATTACK_baseline_confound.md` for seed-variance, metric
choice, and crossover-confound issues) — attacking whether a shortcut solution
exists at all.

Sources read: `CHAPTER_2_DESIGN.md`, `chapter2/synthetic_tasks.py`,
`experiment-runs/2026-04-29_rank_aware_v1/SYNTHESIS.md`, `STATE.md`
(Mathematical Foundations + What We've Built), `KILL_LIST.md`,
`src/matrix_output_heads.py`, `QUEUE.md` (Priority A1).

---

## 0. The master attack: rank-1-vector-in-a-matrix-costume

This single attack subsumes most of what follows and applies to **both**
candidate tasks and to any P-slot architecture, so it goes first.

**Claim.** For any joint-readout task where K ≤ d (matrix dimension) — and,
with a P-slot bottleneck, wherever items-per-slot ≤ d — there exists an
*exact*, rank-1 solution that requires no genuine spectral superposition.

**Construction.** Pick a fixed vector `v0` (a learned parameter, not
data-dependent — nothing stops the encoder from choosing this). Let the
encoder pack all K facts into a single vector `u = Σᵢ cᵢ(facts) eᵢ` using K
orthonormal directions `e1..eK` (possible whenever K ≤ d). Set `Z = u ⊗ v0`.
By definition of the outer product, `rank(Z) = 1` for any nonzero `u, v0`.
Recovery: `Z v0 / ‖v0‖² = u` is a **linear** readout (no MLP nonlinearity
even required), then `cᵢ = u · eᵢ` is linear. All K facts recovered exactly
from a matrix whose SVD has exactly one nonzero singular value.

**Why this matters here specifically:** this isn't a hypothetical — it is
what the project has *already observed*, twice:

- `KILL_LIST.md` Lesson 6: 3-seed replication of the flatten-readout
  matrix-CODI arm shows accuracy tight (81.51% ± 1.2pp) while Z_rank varies
  4→13 across seeds — rank is not what the loss is pricing.
- `rank_aware_v1/SYNTHESIS.md`: an unconstrained run reached effective rank
  13.22 (near-full-rank at d=16), yet a **force-rank=1** run on the *same*
  multi-target task scored *higher* (61.72% vs 58.99% mean). The
  "high-rank-looking" representation was functionally rank-1 all along —
  exactly the shortcut this construction predicts.

**Consequence for the K≈P crossover prediction.** The design predicts the
rank-k curve bends at K > P (pigeonhole forces slot rank ≥ ⌈K/P⌉). That
pigeonhole argument is about *storage slots*, not about *spectral rank*. A
single rank-1 slot already carries `2d` real-valued degrees of freedom (`u`
and `v`), and per the construction above, `d` of those can hold K
independent items losslessly with zero interference. So the naive forcing
threshold is not `K > P` but **`K > P·d`** (or `d` per slot for a single
constant-`v` trick; possibly higher still if the decoder is an unconstrained
MLP that can exploit continuous-precision packing beyond even `d`
independent linear directions). At the scales this project actually runs
(d=16, P∈{1..8}), `P·d` is 16–128 — comfortably inside any K range a
synthetic-task generator can produce without becoming an MNNS-style
intractable-to-build monster. **Under the currently specified tasks and an
unconstrained decoder, the crossover at K≈P will almost certainly not
appear; if anything bends, it will bend near K≈P·d, and that bend would be
a capacity-of-P-vector-slots result, not a rank result.**

This is the same shortcut CHAPTER_2_DESIGN.md's own gauntlet item 1
anticipated for the *vector baseline* ("256 dims is enough to track 16
streams by allocating 16 dims each") — the finding here is that the
**matrix arm falls into the identical trap by default**, per Lesson 2
("the model always finds the lowest-rank representation that satisfies the
loss"). No separate vector baseline is even needed to demonstrate the
confound; the matrix arm's own naturally-learned rank statistics already
show it.

---

## 1. Task A' — K parallel entity tracking, joint readout

Read `chapter2/synthetic_tasks.py`. Current grammar: `(SET <entity>
<state>)* QUERY <entity> SEP`, target = single entity's state. The proposed
fix changes the query to ask for all K (or a function of all K).

### A'-1. Position-decomposition survives the joint-readout fix (FATAL at P≥K)

The joint-readout change closes nothing that position-decomposition already
exploits. With P ≥ K slots, the model assigns entity `i` → slot `i`
(static, content-addressed routing — entity IDs are a **fixed small
vocabulary, 1..K, identical across every training sample**, see
`entity_tok()`), each slot stores its entity's last-SET value at rank 1, and
joint decode is just "read slot 1, read slot 2, ..., read slot K" — the
exact `P=6, ≤2 targets` failure mode `rank_aware_v1` already documented
empirically. Joint readout was never the vulnerability that made the
*original* single-query Task A rank-1-solvable; **P ≥ K** was, and P ≥ K
survives the fix untouched. Fixable only by enforcing **P < K** in every
condition where a "no shortcut" claim is being made; treat K ≤ P results as
confirmatory-only.

### A'-2. Concatenated joint report ≠ a jointly-coupled task (FATAL as scoped, FIXABLE)

Look closely at what "report all K entities' states" actually demands. If
the readout is a plain concatenation — decode entity 1's state, then
entity 2's, ..., in a **fixed canonical order** — it decomposes into K
*independent* single-item retrievals with no aggregate coupling between
them. This is functionally identical, per decode step, to asking the
*original* single-query task K times in one forward pass. It adds **zero**
new joint/simultaneous pressure beyond what the original task already
required at the SET-encoding stage (which, note, already forces retention
of all K entities through the encoding phase, since the query entity isn't
revealed until the QUERY token — the "rank-1-solvable-by-retrieval" framing
in the design brief is about the *decode/readout* side leaking, not the
*encode* side).

**Fix:** query a genuinely relational/aggregate function that cannot be
decomposed into K independent per-entity lookups — e.g. "list the entities
that share the same state" (needs pairwise comparison across items held
*simultaneously*), "is any state value repeated," or "the sorted order of
states." Note even this harder version can still be answered from P ≥ K
rank-1 slots via cross-slot attention/comparison (a comparison mechanism
over multiple rank-1 slots doesn't need any single slot at rank > 1) — so
this fix helps close A'-2 but does **not** substitute for A'-1's P < K
requirement.

### A'-3. Static small-vocab IDs → associative-memory shortcut (FIXABLE)

Entity IDs (1..K) and states (0..S-1, S=32 in the self-test config) are
small, fixed, and identical across every sample. The task reduces to "keep
a hash table's most-recent-write per key" — a capability transformers
already have via ordinary induction/copy-head attention, with **no
dependence on matrix rank whatsoever**, whether or not the bottleneck is
leaky (attack in §3 below). One positive note: SET order is correctly
shuffled (`rng.shuffle(entities)`) so there's no positional recency
shortcut — that part of the existing generator is sound. But the fixed ID
vocabulary itself is a shortcut surface. **Fix:** draw entity/state
identifiers from a much larger and/or per-sample-randomized pool so the
model cannot memorize a static identity→slot routing table; ultimately,
Task D below (continuous random keys) removes this class of attack
entirely rather than patching it.

### A'-4. Chance-adjusted accuracy under joint exact-match (FIXABLE)

If the joint report is scored by exact match on all K items, raw accuracy
mechanically drops as K grows purely from combinatorics (chance of getting
all K right shrinks roughly as `(1/S)^K` for random guessing, faster for
partial-credit heuristics too) — a curve that "degrades with K" could be
pure combinatorial chance rather than any capacity signal. **Fix:** report
per-item (Hamming) accuracy and chance-adjusted margins, not raw exact-match
across all K, when interpreting any K-dependence.

---

## 2. Task B — K-way BFS / graph reachability

CHAPTER_2_DESIGN.md gives this one sentence and no generator spec (branching
factor, depth, node-labeling scheme all unspecified) — it is currently much
less concretely defined than Task A', and inherits several of MNNS's fatal
patterns from `KILL_LIST.md` almost verbatim.

### B-1. Same rank-1-vector-in-costume shortcut applies (FATAL, shared with §0)

Frontier-membership is a monadic ("is node v in the frontier") property, not
a relational one — exactly the case §0's construction handles: pack a
frontier indicator vector into `u`, hold `v` constant, `Z = u⊗v0` is rank-1
and exactly recoverable whenever frontier size ≤ d. Nothing about BFS
structure forces the two-sided (row × column) coupling that would make a
matrix representation load-bearing.

### B-2. The task literally re-derives already-published vector superposition (FATAL for novelty, not for feasibility)

`STATE.md`'s own literature notes cite **Reasoning by Superposition**
(arXiv 2505.12514, Lin/Zhu et al.): `t_c = (1/√|Vc|) Σ u_v`, a **plain
vector sum** over frontier vertices. If Task B's frontier-tracking is
solvable this way (and B-1 shows it is, for a matrix architecture that
degenerates to this construction), then building a matrix version of
essentially the same task risks re-deriving Lin/Zhu's result in a costume,
not testing anything matrix-specific. To be a genuine test, the readout or
task structure must require simultaneous ROW-indexed and COLUMN-indexed
independent structure that a flat vector sum cannot replicate (see Task D,
§7, which does this by construction via key-value binding rather than
set-membership).

### B-3. "Optimal solution provably has rank K*" is not actually proven (FATAL as stated)

CHAPTER_2_DESIGN.md's falsifiable prediction says the task's "optimal
solution provably has rank K*." For Task B this rests entirely on the
hand-designed vector construction in Lin/Zhu — which is an **existence /
sufficiency** result (a specific low-effort encoding that happens to work),
not a **necessity / lower-bound** result (no proof that nothing lower-rank
works). Per §0, a rank-1 construction that also works generically exists.
Without an explicit lower-bound argument, "provably requires rank K*" is
currently an assertion, not a fact, and the entire chapter would be testing
an unproven premise. (Task D in §7 supplies an actual provable lower bound
via elementary linear algebra — see that section.)

### B-4. Frontier-size-by-construction is an MNNS-style trap (FATAL as currently unspecified)

`KILL_LIST.md`'s MNNS post-mortem is directly on point: "'Exact ground
truth' frontier `2^t` is the unpruned search-tree size, not the actual
distinct partial sums. Any sensible instance has pruning that collapses
`2^t` to a much smaller number." The same risk applies verbatim to BFS
frontier size on a DAG: without a concretely specified graph generator
(branching factor, depth, merge/collision rate), there is no guarantee that
"frontier size = K" as designed rather than "frontier size = min(K, actual
distinct reachable set after pruning/merging)," which is typically much
smaller for small synthetic graphs. **Fix:** specify the generator
concretely and empirically histogram frontier sizes before claiming K is
controlled by construction — do not assume it.

### B-5. Canonical node numbering / template graphs enable depth-conditioned heuristics (FIXABLE)

If graphs are generated from a small family of templates with predictable
node-ID-to-depth mapping (a common shortcut in synthetic-graph generators),
the model can learn depth-conditioned heuristics instead of tracking the
actual frontier. **Fix:** randomize node labeling per instance; verify held
out combinatorial generalization (see A'-3/§4).

---

## 3. Architecture attacks (P-slot bottleneck + iterative refinement) — shared by both tasks

The mission brief explicitly separates this design from CHAPTER_2_DESIGN.md's
literal spec ("NOT a full-attention sequence transformer"), which is the
correct instinct — the literal design doc as written does not implement a
bottleneck at all.

### Arch-1. Readout re-attending raw input tokens defeats any bottleneck (FATAL as specified in CHAPTER_2_DESIGN.md)

CHAPTER_2_DESIGN.md describes "N matrix-transformer blocks... readout at
final position" over the whole token sequence with standard causal
attention. In a decoder-only transformer, **every prior token remains in
the KV cache and is attendable unless explicitly masked.** If the SET/QUERY
prefix tokens are still attendable when the K-item answer is decoded, the
"P-slot bottleneck" is not an information channel restriction at all — it's
a naming convention for some subset of positions the model is free to
ignore. The model can just do ordinary in-context retrieval/copy from the
raw prefix at each decode step, which transformers are already very good at
and which has nothing to do with matrix rank. This would make the entire
experiment vacuous: high accuracy would reflect standard attention-based
retrieval, not bottleneck rank usage. This is explicitly flagged as a risk
in the mission brief and it is real and currently unaddressed by either
task design.

**Mandatory fix (applies regardless of task choice):** implement a genuine
encoder → P-slot-bottleneck → decoder architecture with a **hard attention
mask**: decode/readout positions may attend to the P frozen matrix slots
(and causally among themselves, for autoregressive multi-item decode) and
**nothing else** — never to raw prefix positions. This must be verified by
construction (an explicit mask, not an emergent property) and sanity-checked
with a "blank-out" unit test: corrupt/zero the raw prefix's cached
keys/values after encoding and confirm decode logits are bit-for-bit
unchanged.

### Arch-2. Iterative refinement is fine on the encoder side; the boundary must be sharp

Multiple "thinking" iterations that read the raw input while populating the
P slots are legitimate (that's the encoder doing its job). The leak is only
at the encode→decode boundary. A subtler version of Arch-1: if refinement
can be re-triggered *per decode step* (i.e., the model gets a fresh
raw-input read for each of the K items it reports), that's Arch-1 again in
disguise — verify refinement runs to completion once, before any decode
step, with the same "blank-out" test extended across the full decode
sequence, not just the first token.

### Arch-3. P ≥ K legitimizes exactly the failure `rank_aware_v1` already found (FATAL at P≥K, restated from A'-1 at the architecture level)

Documented empirically already — not hypothetical. Any P-slot design must
report P < K as the load-bearing test condition; P ≥ K results are
confirmatory noise at best (and per §0, the P ≥ K side is expected to be
flat purely from the rank-1-per-slot shortcut, not because "the hypothesis
survived this regime").

### Arch-4. Unconstrained decoder power always has headroom for §0's shortcut (FATAL as scoped, per KILL_LIST Lessons 1 and 5)

If the readout/decoder is a generic MLP over `flatten(Z)` (or over
`MultiProbeHead`-style probes with enough probes / enough nonlinear
capacity), it can always exploit the §0 construction, and per
`KILL_LIST.md` Lesson 5 (bilinear+GELU readout on ProsQA still produced a
flat rank-k curve — "the GELU doesn't force the model to USE higher rank;
it only makes higher-rank paths available"), adding nonlinearity to the
readout is **not sufficient** to force rank pressure. The K-vs-P sweep
under an unconstrained decoder measures decoder capacity, not the matrix
slot's functional rank. **Fix:** pin the decoder to a small number of
explicit, architecturally-limited bilinear probes (ideally exactly the
mechanism the hypothesis claims — see Task D's linear "unbind" in §7), and
report/justify decoder capacity explicitly rather than leaving it implicit.

---

## 4. Memorization / distributional attacks (shared)

- **No systematic-generalization split.** `synthetic_tasks.py` generates
  infinitely from a live RNG stream (good — avoids exact-sequence
  memorization) but has no held-out split for compositional generalization
  (novel ID combinations, longer sequences than trained on, novel graph
  shapes). A model that overfits shallow statistics of the *typical*
  instance distribution could still show a "clean" rank-k curve that
  reflects memorized regularities rather than the general mechanism.
  **Fix:** explicit held-out splits (train len ≤ L / eval len up to 2L;
  train ID pool A / eval ID pool B; for Task B, train graph family X / eval
  graph family Y).
- **Small state/ID vocabularies (A'-3, B-5)** are memorization surfaces as
  described above.
- **Sequence-length-with-K confound** (cross-referenced from the sibling
  attack file `ATTACK_baseline_confound.md` §2): if total sequence length
  grows with K, a K>P bend could be "sequences got longer/harder to
  optimize," not "P slots ran out of capacity." Decorrelate K from total
  step count.

---

## 5. FATAL / FIXABLE verdict table

| # | Attack | Target | Verdict | Minimal fix |
|---|---|---|---|---|
| 0 | Rank-1-vector-in-costume (`Z=u⊗v0`, v0 constant) exactly solves any K ≤ d (or ≤ P·d) | Both, master attack | **FATAL as scoped** | Constrain decoder to a genuine bilinear "unbind" (no unconstrained MLP) AND move to a task where set-membership isn't enough — see Task D |
| A'-1 / Arch-3 | Position-decomposition at P ≥ K | Both, architecture | **FATAL at P≥K** | Enforce P < K in every condition treated as informative |
| A'-2 | Concatenated joint report decomposes into K independent reads | Task A' | **FATAL as scoped**, fixable | Require a genuinely relational/aggregate query (collision detection, sorted order), not concatenation |
| A'-3 | Static small entity/state vocab → associative-memory shortcut | Task A' | Fixable | Randomize / widen ID pool; or use continuous random keys (Task D) |
| A'-4 | Chance-adjusted combinatorics in joint exact-match scoring | Both | Fixable | Report per-item Hamming accuracy + chance-adjusted margin |
| B-1 | Shares master attack (§0) | Task B | **FATAL as scoped** | Same as #0 |
| B-2 | Reduces to already-published vector superposition (Lin/Zhu) | Task B | **FATAL for novelty** | Require genuine row×column relational structure (Task D) |
| B-3 | "Provably rank K*" is an existence claim, not a proven lower bound | Task B | **FATAL as stated** | Derive an explicit lower bound or drop "provably" |
| B-4 | Frontier-size-by-construction is an MNNS-style miscount risk | Task B | **FATAL as currently unspecified** | Concretely specify generator; empirically histogram frontier sizes |
| B-5 | Canonical/template graph structure enables depth heuristics | Task B | Fixable | Randomize node labeling per instance |
| Arch-1 | Readout re-attends raw input tokens (CHAPTER_2_DESIGN.md's literal full-attention spec) | Both, architecture | **FATAL as specified in CHAPTER_2_DESIGN.md** | Hard-masked encoder→bottleneck→decoder with a verified blank-out test |
| Arch-2 | Per-decode-step re-encoding smuggles Arch-1 back in | Both, architecture | Fixable, needs explicit test | Extend blank-out test across full decode sequence |
| Arch-4 | Unconstrained decoder always has headroom for #0; nonlinearity alone insufficient (Lesson 5) | Both, architecture | **FATAL as scoped** | Restrict decoder to a small number of fixed bilinear probes tied to the task's actual unbind operation |
| §4 | No systematic-generalization split; length/ID/graph memorization | Both | Fixable | Held-out compositional splits |

---

## 6. Robustness ranking: Task A' vs Task B

**Task A' is more robust — closer to fixable with local patches.** It has
working, tested generator code, a controllable K, no unproven mathematical
claims, and its remaining fatal issues (A'-1 through A'-4) are all
addressable without new theory: enforce P<K, use a relational query,
widen/randomize the vocab, fix the scoring metric.

**Task B is currently the weaker candidate.** On top of sharing every
architecture-level and master attack that Task A' shares, it independently
carries B-2 (novelty risk — may just re-derive Lin/Zhu with matrices as
decoration), B-3 (an unproven "provably rank K*" claim resting on an
existence-only construction), and B-4 (an MNNS-shaped frontier-size trap
from an unspecified generator). These require actual new theoretical work
(a lower-bound proof, a concretely specified and empirically verified
generator) before Task B is even a well-formed task, not just an
engineering patch.

**Bottom line: both are fatally flawed as currently scoped**, primarily
because of attacks that are architecture-and-decoder-level rather than
task-specific (§0, Arch-1, Arch-3, Arch-4) — these would sink *any*
joint-readout task built on a small discrete vocabulary, an unconstrained
decoder, and P ≥ K. Fixing the architecture is necessary but not
sufficient; the task itself needs to close the master shortcut in §0, which
neither A' nor B does by construction. Task A' is the better base to patch;
Task B needs foundational rework before it is even attackable on the same
terms.

---

## 7. Recommended hardened design

### 7a. Task D — K simultaneous tensor-product key/value bindings

This operationalizes the math `STATE.md` **already has written down** under
"Superposition encoding" (`M = Σᵢ αᵢ(uᵢ⊗vᵢ)`, `logit(w) = u_w^T M v_w`) as an
actual from-scratch trainable task, rather than an assumed construction.
It is the matrix-native analog of Smolensky's (1990) Tensor Product
Representations and Plate's (1995) Holographic Reduced Representations —
send a research agent to confirm no prior work has trained a matrix
transformer end-to-end on this exact task before building (per CLAUDE.md's
mandatory waterfall / novelty check).

**Grammar:** `(BIND <key> <value>)^K QUERY <key_j> SEP`, repeated for a
(randomized-order) subset or all of the K keys used. Keys and values are
drawn **fresh per sample** from a large pool of near-orthogonal random
d-dimensional embeddings (either literally random Gaussian vectors, or a
large-enough discrete vocabulary — say ≥ 10K distinct key tokens and ≥ 10K
value tokens with randomly initialized, *not* systematically structured,
embeddings) — this closes A'-3/B-5's memorization surface directly, since
keys can't be memorized as a small fixed identity set, and it enables a
genuine held-out generalization test (evaluate on key vectors never seen
during training).

**Readout, architecturally constrained (closes Arch-4):** the decoder is
*not* an arbitrary MLP over `flatten(Z)`. It is the literal unbind
operation the hypothesis claims the matrix performs: given query key
`key_j`, predicted value `= decode(Z · key_j)` (matrix-vector product),
where `decode` is a small fixed classifier over the value vocabulary (or a
nearest-neighbor lookup against the value codebook, standard in the HRR
literature).

**Why this closes the master attack (§0), with an actual proof, not a
construction:** suppose K query keys `key_1..key_K` are linearly
independent (guaranteed with high probability for random vectors, K ≤ d),
and suppose exact recovery is required — i.e., `Z key_j = value_j` for all
j, where the `value_j` are themselves linearly independent (again generic
for random vectors). Stack the keys as columns of `K_mat ∈ ℝ^{d×K}` and the
values as columns of `V_mat ∈ ℝ^{d×K}`; the requirement is `Z K_mat =
V_mat`. Standard linear algebra: `rank(V_mat) = rank(Z K_mat) ≤
rank(Z)`. Since the values are linearly independent, `rank(V_mat) = K`,
therefore **`rank(Z) ≥ K` is forced** — not "typical," not "the
construction we happen to use," but a hard consequence of requiring exact
recovery of K independent key→value mappings through a linear unbind. The
§0 shortcut (`Z = u⊗v0` constant-`v0`) is explicitly ruled out: it would
force `rank(Z)=1`, contradicting `rank(Z)≥K` whenever K>1. **This is the
"no shortcut" property Task A' and Task B both lack.**

**Corollary — corrected crossover prediction.** For K ≤ d, exact recovery
requires rank(Z) ≥ K: a clean, provable, single-slot (P=1) test with no
position-decomposition possible at all (there is only one slot). For K > d,
no exact solution exists (over-complete binding); this is the well-studied
*lossy* HRR-capacity regime (Plate 1995; Frady, Kleyko & Sommer 2018,
"resonator networks") with known closed-form capacity-vs-dimension curves —
a second, literature-grounded prediction to test against, rather than an ad
hoc "K≈P" guess. With P slots, the natural extended prediction is a bend
near **K ≈ c·P·d** (up to the constant from HRR capacity theory), replacing
the unmotivated K≈P line in CHAPTER_2_DESIGN.md.

**Recommended sequencing:** run the **P=1, single-matrix-state** version of
Task D first, sweeping K=1..d. This is the cleanest possible go/no-go test:
no position-decomposition is even architecturally available (P=1), the
lower bound is a proven theorem not an assumption, the readout is pinned to
the actual claimed mechanism, and vocab is randomized so memorization is
closed. Only proceed to a P-slot sweep (reintroducing the P<K pigeonhole
question, §Arch-3) if the P=1 result is positive — that is the point at
which the K≈c·P·d prediction becomes the thing worth testing.

### 7b. Architecture requirements (apply regardless of A'/B/D)

1. Hard-masked encoder → P-slot bottleneck → decoder, verified by a
   blank-out unit test (Arch-1, Arch-2) — mandatory, no task fixes this.
2. P < K enforced in every condition treated as evidence (Arch-3); P ≥ K
   is confirmatory-only and expected to be flat regardless of hypothesis
   truth.
3. Decoder capacity explicitly pinned and justified — ideally the literal
   linear/bilinear unbind the hypothesis claims (Arch-4), not a generic
   MLP with unbounded headroom for §0's shortcut.
4. Report per-item/chance-adjusted accuracy, not raw joint exact-match
   (A'-4).
5. Held-out generalization split (novel keys/IDs, longer sequences) before
   any "the model learned to track K items" claim (§4).

**If Task D is judged too large a redesign to build before the Chapter 2
timeline, the fallback is: Task A' patched with A'-1 (P<K), A'-2 (relational
query), A'-3 (randomized vocab), plus the mandatory architecture fixes in
7b — but go in knowing the crossover will most likely appear near K≈P·d,
not K≈P, and pre-register that corrected prediction so a K≈P·d bend isn't
mistakenly reported as vindicating the original K≈P claim.**
