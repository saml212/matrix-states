# NCR REAL-LM DESIGN — the flagship bet: NCR in a real language model at scale

**DRAFT-STAGE-1 (PRE-ATTACK — NOT BUILD-AUTHORIZED). Opened 2026-07-16.**
This document is a design draft, produced for adversarial attack. No code is
built, no GPU is touched, by this document. Every number below is either
(a) cited from a measured rate elsewhere in this repo, with the exact source
named, or (b) flagged `PRICE-UNKNOWN` / `PROJECTED — Phase 0a confirms`.
Citations to external literature were originally placeholder-tagged
`[TO-VERIFY]`; the grounding memos landed during this drafting session (see
"GROUNDING UPDATE" immediately below) and every citation in this revision
is now either VERIFIED (source named, §3.4's table) or explicitly flagged
as still needing a human spot-check — no `[TO-VERIFY]` tags remain in this
document's claim language.

**Reading list this design builds on** (context, not repeated): `CLAUDE.md`
(hard rules + 7-point pre-experiment checklist); `matrix-thinking/
NOVEL_ARCH_WATERFALL.md` (NCR's narrowed claim: in-context-written operators
composed EXACTLY at read time, O(log h) sequential cost via repeated
squaring — the wave-1 headline is depth-robust EXACTNESS, not reachable-set
generalization, §3.2); `matrix-thinking/NCR_ORTHO_WRITE.md` (the
Newton–Schulz orthogonal-write fix for the K=32 far-depth wall, blind run in
flight, verdict ~2026-07-17 — this document's entire K-axis choice is gated
on that verdict, §9); `STATE.md` ACTIVE CAMPAIGNS (Head-to-Head Demo's
axis-1/axis-2 verdicts, reused directly as baseline methodology below);
`matrix-thinking/FROZEN_BIAS_LM_DESIGN.md` §13 (the ONLY measured real-data
LM training rates in this repo at 98M/392M scale — the pricing anchor for
§6); `matrix-thinking/DELTANET_REALDATA_DESIGN.md` (the naturalistic
probe-task methodology — `grammar_rd.py` bind-clause grammar rendered
through the real GPT-2 tokenizer — reused directly for §3/§5);
`matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` §1.4.2 (inference-memory-
matched axis framing, skimmed for §4.3's KV-cache baseline);
`matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` §1 (the S3/S4/S5/A5/A6
solvable-vs-non-solvable group generator infrastructure, reused directly
for §3's structural-failure task below).

**GROUNDING UPDATE (2026-07-16, coordinator-relayed, both memos
coordinator-spot-checked — supersedes every `[TO-VERIFY]` tag from this
document's first draft):** `research/ncr_separation_grounding.md` and
`research/ortho_write_grounding.md` are VERIFIED (live arXiv/OpenReview
fetch, not recalled). **Load-bearing correction to this document's own
framing, made throughout below:** "matrix-valued state can do state
tracking" is **already published** — Grazzi et al. (negative-eigenvalue
linear RNNs, arXiv:2411.12537, ICLR 2025), DeltaProduct
(arXiv:2502.10297), and RWKV-7 (arXiv:2503.14456) all demonstrate
matrix-valued fast-weight state solving provably NC¹-hard state-tracking
problems (the S₅ word problem, Barrington 1989). **This document's
hypothesis and every WIN band below rest on a narrower, still-unclaimed
axis: EXACT variable-depth composition readable at QUERY TIME in O(log h)
sequential steps via repeated squaring, against every published
alternative's O(h) sequential rollout** — an algorithmic/circuit-depth
claim, not an expressivity claim (`research/ncr_separation_grounding.md`
Part 3). Closest neighbors to design against, all VERIFIED real:
FWM (arXiv:2011.07831), Log-Linear Attention (arXiv:2506.04761), HOLA
(arXiv:2607.02303), Sequential Group Composition (arXiv:2602.03655).
MuonSSM (arXiv:2606.30461, ICML 2026 Oral) is the closest prior art to
the orthogonal write specifically and is cited wherever that mechanism
appears below (§2.2, §9).

---

## §1 HYPOTHESIS

**A DeltaNet-backbone real language model augmented with an NCR (in-context-
written, orthogonally-conditioned d×d operator; O(log h) matrix-power
composition read) fast-weight head, trained at 98M–392M params on a
synthetic-to-real curriculum, will answer depth-h non-solvable-group
relational-composition queries embedded in text EXACTLY at query-time
sequential cost O(log h) via repeated squaring, at depths where (a) a
param-and-data-matched Transformer baseline fails for a citable
STRUCTURAL/complexity-theoretic reason (log-precision transformers ⊆
TC⁰, Merrill & Sabharwal arXiv:2207.00729; TC⁰ cannot solve NC¹-complete
non-solvable-group word problems unless TC⁰=NC¹, Merrill/Petty/Sabharwal
arXiv:2404.08819 applying Barrington 1989's S₅ hardness result), and (b) a
param-matched SEQUENTIAL-ROLLOUT matrix-state baseline (DeltaProduct/
RWKV-7-class, which — per Grazzi et al. arXiv:2411.12537 and Peng et al.
arXiv:2503.14456 — CAN in principle reach the correct answer) requires
Θ(h) sequential state-update steps to reach the same depth, a measurably
different query-time access complexity, not a different reachable
answer set.**

**What this hypothesis explicitly does NOT claim** (the correction this
document's grounding update, above, forced): that matrix-valued
fast-weight state can state-track where vectors/Transformers categorically
cannot — that is already published (Grazzi et al., DeltaProduct, RWKV-7)
and any framing implying otherwise is retired from this design. The claim
is narrower and sharper: EXACT composition, readable at O(log h)
query-time sequential depth, is unclaimed anywhere in the searched
literature (`research/ncr_separation_grounding.md` Part 2) — this is one
falsifiable, one-sentence bet, and everything below either supports
running it or is a pre-registered way it could fail.

---

## §2 ARCHITECTURE OPTIONS

Three options, ranked by measured-cost defensibility (§6 needs a real rate
to price anything; this ranking is not incidental).

### 2.1 Option A — Hybrid Transformer backbone + NCR fast-weight head

**Where writes happen.** A dedicated NCR encoder (`BindingEncoder`-style
trunk, reused verbatim from `matrix-thinking/ncr/`) consumes the
Transformer's own token embeddings at designated bind-clause spans (see §3)
and emits a `d_ncr × d_ncr` operator per relation, passed through the
Newton–Schulz orthogonal-write projection (`NCR_ORTHO_WRITE.md` §2) if the
ortho-write gate (§9) licenses it.

**Where reads happen.** At query positions, a read head computes
`o = binexp_read(Z, q, h)` (scale-managed binary exponentiation, O(log h))
and injects `o` into the residual stream at that position (e.g. added to the
Transformer's own hidden state before the final LM head, or consumed by a
small MLP that produces logits directly for the query token — build-time
decision, not resolved here).

**Param-count delta.** NCR head at the standing K=32/d_ncr=33/h_ncr=64
convention: `P(d,h) = 40h² + 4dh + 46h + d` (verified exact formula,
`NOVEL_ARCH_WATERFALL.md` §9.3) = 40·4096 + 4·33·64 + 46·64 + 33 =
163,840 + 8,448 + 2,944 + 33 = **175,265 params** (matches
`NCR_ORTHO_WRITE.md` §6's independently-stated "~175K"). At a 98M/392M
Transformer backbone this is **+0.18% / +0.045%** — negligible.

**FLOPs delta.** NCR encoder forward FLOPs (leading terms, verified formula):
`F(K,d,h) = 76Kh² + 4dh² + 12K²h + 4Kdh + 4d²h`. At K=32, d=33, h=64:
76·32·4096 + 4·33·4096 + 12·1024·64 + 4·32·33·64 + 4·1089·64 =
9,961,472 + 540,672 + 786,432 + 270,336 + 278,784 = **11,837,696 FLOPs per
relation-write** (≈1.2×10⁷). One binexp read at h=40 costs ≤⌈log₂40⌉=6
matrix-squarings + ≤6 products on a 33×33 matrix ≈ 12×2×33³ ≈ 8.6×10⁵ FLOPs
— negligible next to the write. **PRICE-UNKNOWN: no measured real-corpus
Transformer-LM training rate exists anywhere in this repo at 98M or 392M
scale** (`matrix-thinking/deltanet_rd/transformer_baseline_rd.py` exists
but has only ever been measured on the short (`T_bind≈224`), small-K
Task-E/grammar harness, not a full 512–1024-token real-corpus pretraining
run — see §6.3). This option therefore requires its OWN Phase-0a timing
pilot before any GPU-h can be committed to it, on top of the NCR-head
pilot every option needs.

**Assessment.** Architecturally the cleanest "bolt NCR onto a known-good
backbone" story, and the one most directly comparable to
`HEAD_TO_HEAD_DEMO_DESIGN.md`'s own Transformer arm (reuses its
`transformer_baseline_rd.py` family) — but the backbone itself is an
unpriced unknown at LM scale. Kept as the **mandatory param-matched
baseline architecture** (§4.1) rather than the primary contender, so its
Phase-0a cost is paid regardless of which option is chosen as primary.
**Structural-failure grounding for THIS baseline specifically** (the
Transformer, any depth, any width): log-precision transformers are
contained in uniform TC⁰ (Merrill & Sabharwal, TACL 2023,
arXiv:2207.00729); the word problem of any fixed NON-SOLVABLE finite
group (e.g. S₅, A₅, A₆) is NC¹-complete (Barrington 1989); TC⁰ transformers
therefore cannot solve it unless TC⁰=NC¹ (Merrill, Petty & Sabharwal,
ICML 2024, arXiv:2404.08819, making this deduction explicit against S₅).
This is a genuine complexity-theoretic argument, not an empirical-drift
claim — but it ONLY bites if the task's relation-composition structure is
drawn from a non-solvable group (§3.1 below); the program's existing
single-Hamiltonian-K-CYCLE construction (`task_e.py`, `grammar_rd.py`) is
a CYCLIC (abelian, solvable) group and does NOT license this citation —
this is a load-bearing correction to this document's own §3, addressed
there.

### 2.2 Option B — NCR-augmented DeltaNet-style linear-attention LM (RECOMMENDED)

**Where writes happen.** Identical write mechanism to Option A (same
`BindingEncoder` + Newton–Schulz orthogonal-write pipeline,
`NCR_ORTHO_WRITE.md` §2 — **closest prior art: MuonSSM (Nguyen, Vo, Vo,
Nguyen & Pham, arXiv:2606.30461, ICML 2026 Oral), which orthogonalizes
fast-weight WRITES rather than the recurrent transition matrix, the same
category distinction NCR's fix relies on. Materially different target,
confirmed by direct read
(`research/ortho_write_grounding.md` §4): MuonSSM conditions RANK-1
KV outer-product injections (`X_t=v_tk_tᵀ`) with a single quintic-
coefficient Newton–Schulz step (magnitude conditioning of a rank-1
object, which cannot itself be orthogonal in d>1 dims); NCR's fix targets
FULL-RANK d×d written operators driven to near-exact `QᵀQ≈I` (classical
cubic Björck–Bowie coefficients, many iterations) so that REPEATED
COMPOSITION (matrix powers) exactly preserves eigenstructure — MuonSSM
does not test, cite, or motivate compositional-depth generalization.**),
but the backbone is the DeltaNet
LM family this repo has ALREADY trained and priced at 98M/392M
(`lm_rd_rung_configs.py` `RUNGS`, `d_model=768/n_layers=12/d_state=64` and
`d_model=1536/n_layers=16/d_state=128`, `conv_size=4`, `num_heads=1`,
GPT-2 BPE `vocab_size=50257`, tied embed/unembed — the exact architecture
`FROZEN_BIAS_LM_DESIGN.md` §13.1/§13.7 measured). **Two `d`s must not be
conflated (a live confusion risk this repo has hit before, `FROZEN_BIAS_LM_
DESIGN.md` §13.1's own "scale disambiguation"):** DeltaNet's own
`d_state` (64 or 128, its native per-layer linear-attention recurrent
state, decaying/leaky, updated every token) is a SEPARATE quantity from
`d_ncr` (33, the NCR head's own entity-operator dimension, exact and
orthogonally-conditioned, updated only at bind-clause positions). NCR does
NOT replace or reuse DeltaNet's own state — it is a side-channel module
attached after the backbone's final layer, reading the backbone's own
hidden states at bind-clause/query positions exactly as Option A's NCR
head reads the Transformer's.

**Where reads happen.** Same binexp read as Option A, injected at query
positions before the shared LM head.

**Param-count delta.** Same 175,265-param NCR head. At 98M
(97,618,176 params) and 392M (391,869,440 params): **+0.18% / +0.045%** —
identical to Option A's delta (the NCR head's cost is backbone-agnostic by
construction).

**FLOPs delta — the load-bearing calculation for this option's cost
defensibility.** Backbone FLOPs/token (fwd+bwd), decomposed per
`DELTANET_REALDATA_DESIGN.md` §4.2's own head-dominated method: head term
`≈2·d_model·V` forward, ×3 for fwd+bwd (the measured 2.6×10⁷→7.7×10⁷
ratio at that design's own 14M config); body term `≈6·N_body` where
`N_body = N_total − V·d_model` (the tied-embedding table, counted once in
params but only once more in FLOPs via the head term above). At 98M
(`d_model=768`): head ≈ 2·768·50257·3 ≈ 2.32×10⁸ FLOPs/token; `N_body` ≈
97,618,176 − 768·50257 ≈ 59.0M → body ≈ 6·59.0M ≈ 3.54×10⁸ FLOPs/token.
Total ≈ **5.86×10⁸ FLOPs/token**. At batch=32, seq_len=512 (the measured
config, §6): 16,384 tokens/step → **≈9.6×10¹² FLOPs/step**.

NCR overhead per step: assume ≤2 bind-clause episodes/sequence at
seq_len=512 (T_bind=224 tokens per K=32 episode, §3) × batch=32 →
≤64 write invocations/step × 1.2×10⁷ FLOPs ≈ **7.7×10⁸ FLOPs/step** —
**≈0.008% of backbone FLOPs/step.** This is a raw-FLOP ratio, not a
wall-clock prediction — `NOVEL_ARCH_WATERFALL.md` §9.4 explicitly warns
that "170K-param toy cells are almost certainly kernel-launch/small-batch
overhead-bound... not compute-bound," and the Newton–Schulz iteration
(§2 of `NCR_ORTHO_WRITE.md`, ~12 power-iteration steps + several NS
iterations, all small sequential 33×33 matmuls) is exactly the shape of
op that can cost more in wall-clock than its FLOP share. **Predicted
overhead band: ≤5% wall-clock at 98M/392M, by analogy to the ONE directly
comparable measured precedent in this repo** — the frozen-bias per-token
blend (`FROZEN_BIAS_LM_DESIGN.md` §13.4: "at 14M, all 20 rung-1 cells...
fell in a tight 899–914s band regardless of arm" — a comparable
per-position, small-sequential-op insertion measured at ≤1.2% overhead).
**This is a prediction, not a measured number — Phase 0a (§6.2) exists
specifically to confirm or kill it before any full-scale spend.**

**Assessment — why this is the recommendation.** This is the ONLY option
where the backbone itself has a real, twice-verified, on-this-exact-box
measured training rate at BOTH target scales (`FROZEN_BIAS_LM_DESIGN.md`
§13.7: 98M = 0.236 s/step, 392M = 0.836 s/step, both `batch=32,
seq_len=512`, both VRAM-profiled). It also reuses the naturalistic
probe-task infrastructure (`grammar_rd.py`, the fla chunked-kernel
verification chain, `DELTANET_REALDATA_DESIGN.md` §4.3/§4.4) already
built and box-verified for this exact backbone family. Choosing Option B
means §6's compute pricing can be grounded in measured rates for the
dominant cost term (the backbone) and only the NCR-head delta is
projected — versus Option A, where BOTH the backbone and the delta would
be projected.

**Load-bearing caveat, must be checked at build time, not assumed either
way (this is where §1's O(log h)-vs-O(h) framing meets this specific
backbone).** DeltaNet's native transition is `I − βkkᵀ` (a Householder
reflection); the standard sigmoid gate restricts `β∈(0,1)`, which per
Grazzi et al. (arXiv:2411.12537) restricts transition eigenvalues to
`[0,1]` and provably BARS parity/non-solvable-group state tracking —
i.e., under that gate the backbone is ITSELF still TC⁰-limited (same
class as diagonal SSMs, same Merrill/Petty/Sabharwal argument, §3.1),
which would make it a valid THIRD structural-failure baseline, not an
O(h)-capable one. If the box's fla kernel or a config flag instead allows
`β∈(0,2)` (unlocking negative eigenvalues), the backbone escapes TC⁰ and
must be treated as the O(h)-sequential-rollout baseline (§4.4) — the two
readings lead to different baseline framings and must not be assumed;
**verify `deltanet_core.py`'s actual `β` range at Phase 0 (§6.2)** before
any claim language is finalized. Either reading is fine for this design's
own contender (NCR's read-time complexity claim does not depend on which
regime the backbone sits in), but it changes what the DELTANET BACKBONE
ITSELF is allowed to be called in the results write-up.

### 2.3 Option C — Pure fast-weight LM with NCR reads only (NOT recommended this wave)

**Where writes/reads happen.** No separate attention or DeltaNet
token-mixer at all — every token position writes into and reads from a
single (or small bank of) NCR-composable matrix state(s); the "backbone"
IS the NCR write/read pipeline, generalized from the isolated Task-E
episode format to a full autoregressive per-token stream.

**Param/FLOPs/memory.** Not computed here — this is not a parameter tweak
on an existing, verified backbone; it is a from-scratch LM architecture
(every token needs its own embedding→feature→write-or-read decision path,
which does not exist anywhere in this repo yet) and would need its own
full waterfall (brainstorm → research → attack) before a paper design,
let alone a build. **PRICE-UNKNOWN in every column.**

**Why not recommended.** `NOVEL_ARCH_WATERFALL.md` §2 finding M4 (binding,
not revisitable without a new waterfall pass) already established that
NCR's non-redundant capability delta is narrowly **the READ mechanism**
(O(log h) matrix-power composition beyond training depth), layered onto
an EXISTING fast-weight write substrate — not a claim that NCR should
replace a token-mixer wholesale. DeltaNet's own chunked delta-rule already
IS a fast-weight write mechanism (rank-1 updates every token); reinventing
a second, NCR-native write mechanism as the ONLY token-mixer duplicates
that machinery for no established gain, and risks exactly the redundancy
M4 flagged for the operator-bank sub-experiment. A more interesting
future direction — DeltaNet's own accumulated state AS the object NCR's
binary-exponentiation read composes, i.e. applying the ortho-write fix to
DeltaNet's native rank-1 update instead of to a separate encoder's output
— is named here as the natural Stage-2 follow-on, explicitly OUT OF SCOPE
for this Stage-1 build (it requires re-deriving the ortho-write mechanism
against a decaying rank-1-update state, a different and harder research
question than §9's pending verdict answers).

**Closest neighbors an Option-C design would have to be built against**
(VERIFIED, `research/ncr_separation_grounding.md` Part 2 — recorded here so
a future waterfall pass does not re-search from zero): **FWM** (Schlag,
Munkhdalai & Schmidhuber, ICLR 2021, arXiv:2011.07831) — the single closest
prior art on in-context fast-weight writes for compositional inference, but
recursive/LN-nonlinear/approximate reads, not a matrix power; **Log-Linear
Attention** (arXiv:2506.04761, ICLR 2026) — closest on the "O(log ·)"
NAME, but the log-ness is a hierarchical-summarization COMPUTE-complexity
property, not a query-time repeated-squaring read of one written operator;
**HOLA** (arXiv:2607.02303) — exact VERBATIM key-value recall via a bounded
cache, not exact ALGEBRAIC composition; **Sequential Group Composition**
(Marchetti, Kunin, Myers, Acosta, Miolane, arXiv:2602.03655) — the
in-WEIGHTS complement of NCR's in-context setting (proves perfect
in-weights group-composition learning needs width exponential in sequence
length), not a competing mechanism. None of the four does (write
in-context) + (read via literal matrix power) + (exact) simultaneously —
Option C would be the first architecture to attempt combining all three as
the ONLY token-mixer, which is exactly why it needs its own waterfall
before a build decision, not a paragraph in this document.

### 2.4 Recommendation

**Primary: Option B (NCR-augmented DeltaNet LM).** Reuses this repo's ONLY
measured real-corpus LM training rate at 98M/392M, reuses the verified fla
chunked-kernel path (`DELTANET_REALDATA_DESIGN.md` §4.3/§4.4), reuses the
naturalistic grammar-embedding task infrastructure, and the NCR head's
own param/FLOP delta is independently negligible regardless of backbone
choice. **Secondary (mandatory, not optional): Option A's Transformer
backbone, WITHOUT an NCR head, as the param-matched baseline** (§4.1) —
this is not "Option A" as a contender, it is the comparison arm Option B's
whole claim is measured against, so its own Phase-0a pricing (§6.3) is
paid either way. **Option C: not built, not priced, flagged as a future
waterfall item only.**

---

## §3 TASK SUITE

**Two-pronged design, per the grounding update (above).** Axis (i) —
**structural failure**: the Transformer baseline must fail for a citable
complexity-theoretic reason, which REQUIRES the composed relations to form
a NON-SOLVABLE group (Task 2, §3.2 — the primary, flagship task); the
existing single-cycle construction (Task 1, §3.1, abelian/solvable) is kept
as a secondary, complementary EXACTNESS measurement whose baseline-failure
citation is empirical (composition drift), not structural, and must not be
oversold as the structural claim. Axis (ii) — **query-time access
complexity**: measured against a param-matched SEQUENTIAL-ROLLOUT
matrix-state baseline (§4.4, DeltaProduct/RWKV-7-class) that CAN in
principle reach the correct answer (per Grazzi et al./Peng et al., already
published) but requires Θ(h) sequential steps to do so — NCR's claim here
is O(log h) query-time depth via repeated squaring against that baseline's
O(h), a wall-clock/circuit-depth separation, not an accuracy separation.
Every task below is instrumented with the mod-K-collapse guards `CLAUDE.md`
mandates.

### 3.1 Task 1 — Depth-robust relational composition embedded in text
(SECONDARY — pure exactness measurement, abelian/solvable construction)

**Construction.** Direct real-LM scale-up of `grammar_rd.py`'s bind-clause
grammar (`<buf...> KEY <rel> VALUE .` clauses, real GPT-2-tokenizable
English names/verbs, single-Hamiltonian-K-cycle entity graph,
`DELTANET_REALDATA_DESIGN.md` §5.2) interleaved into real-corpus documents
per the curriculum in §5. A document contains 1–2 bind episodes (K
relations each) plus a query clause; the model must answer the query
entity after `h`-fold relational composition.

**Single-full-K-cycle discipline (`CLAUDE.md` mod-K hard rule, applied,
not re-derived).** One Hamiltonian K-cycle per episode (`grammar_rd.py`
already reuses `task_e.py`'s `_permutation_graph`/periodicity guards
verbatim). Every reported `h` carries its `(h, h mod K)` residue pair;
aggregates never pool across residues; identity (`h mod K = 0`) and
train-residue (`h mod K ∈ {1,2,3}`) points are excluded from claims
(`NOVEL_ARCH_WATERFALL.md` §3.2's eval-grid convention, transplanted
verbatim).

**h-grid.** K∈{15, 32} (both cleared depending on the ortho-write verdict,
§9). Train h∈{1,2,3}. Eval ladder reuses the ortho-write realistic ladder
`{5, 12, 20, 29, 40, 61}` (all novel residues mod both K values, per
`NCR_ORTHO_WRITE.md` §3) plus a full residue sweep at one depth to prove
the mod-K bookkeeping has teeth (`NOVEL_ARCH_WATERFALL.md` §3.2's
"residue_sweep" convention).

**Why the Transformer is predicted to underperform here — empirical, not
structural (a load-bearing demotion vs. this document's first draft).**
The single-Hamiltonian-K-cycle construction generates a CYCLIC group —
abelian, hence SOLVABLE — and Barrington's NC¹-completeness result applies
only to NON-SOLVABLE groups; a solvable group's word problem sits inside
`ACC⁰`/`TC⁰` in principle, so a TC⁰ transformer is NOT structurally barred
from this task the way it is from Task 2 (§3.2, below). The prediction here
rests instead on the EMPIRICAL composition-error-cascading phenomenon
(Guu, Miller & Liang, 2015, arXiv:1506.01094, VERIFIED real —
`research/ncr_separation_grounding.md` reference N5 — composition error
accumulates over path length in trained models) plus this program's own
measured drift precedent (`NOVEL_ARCH_WATERFALL.md` §3.2's P2 prediction:
baseline recovered_frac@0.9 falls below 0.5 by h=29 (K=8) / h=45 (K=12) —
re-derive fresh h-dependent bars for K∈{15,32} at calibration, do not
import those numbers directly). NCR's read is EXACT at every depth here
too (binary exponentiation of an orthogonal operator), so this task still
demonstrates a real KIND difference (bounded numerical drift vs. exact
linear algebra) — but the correct claim register is "empirically more
robust," not "structurally impossible for the baseline." Task 2 carries
the structural claim; this task is its exactness-mechanism companion.

### 3.2 Task 2 — Non-solvable-group word-problem chain (PRIMARY —
structural-failure task, mod-K-trap-proof by construction)

**Construction, merged from two prior drafts of this task (a
non-solvable-group relational task and `NCR_ORTHO_WRITE.md` §4b's
structured-operator discriminator turn out to be the SAME construction
once the operator bank's generators are drawn from a non-solvable finite
group instead of arbitrary random-orthogonal matrices — one task, not
two).** Reuse `CAPABILITY_SEPARATION_DESIGN.md` §1's existing group
generator infrastructure (S₃/S₄/S₅/A₅/A₆, solvable-vs-non-solvable pairs
already built and calibrated in this repo) to draw R DISTINCT generator
matrices from a NON-SOLVABLE group (S₅, order 120, or A₅/A₆) acting on the
K-entity pool, each rendered as its OWN bind-clause relation verb
(`grammar_rd.py`'s existing rel-A/rel-B pools, extended to R pools). A
query specifies a PATH of generator indices `(o_1,...,o_L)`; the target is
the exact group-word product `g_{o_L}∘...∘g_{o_1}` applied to the query
entity, computed by exact integer/permutation composition (no floating
point in the ground truth). Depth = path length L.

**Why this is the structural, not empirical, failure claim.** The word
problem of any fixed non-solvable finite group is NC¹-complete
(Barrington, JCSS 1989, VERIFIED — `research/ncr_separation_grounding.md`
item 2); log-precision transformers are contained in uniform TC⁰ (Merrill
& Sabharwal, TACL 2023, arXiv:2207.00729, VERIFIED); therefore NO
log-precision transformer of ANY depth or width can compute this task's
answer function unless TC⁰=NC¹ — a conjectured-false complexity-class
separation, not a training/data artifact (Merrill, Petty & Sabharwal, ICML
2024, arXiv:2404.08819, VERIFIED, makes exactly this deduction against
S₅). **This is a genuine "cannot, not merely does not" prediction** — the
strongest available structural-failure citation for this document's §1
hypothesis, and the reason this task, not Task 1, is the flagship.

**Mod-K-trap-proof by construction (three guards, `NCR_ORTHO_WRITE.md`
§4b's convention, transplanted verbatim, now doing double duty as BOTH the
held-out-depth hygiene AND part of the non-solvable-word-problem
structure):** (1) distinct generators per hop — a product of DIFFERENT
group elements, not a power of one matrix, so there is no single cycle
length to reduce modulo; (2) no consecutive repeats; (3) fixed-point
exclusion (query/path pairs whose composite fixes the start are excluded).

**h-grid.** Train L∈{1,2,3}; eval L∈{5,8,12,16,20,24,32,40} (the
`NCR_ORTHO_WRITE.md` §4b ladder, R=4 primary, generators drawn from S₅ or
A₅/A₆ per the non-solvable requirement above — this is the one build-time
change from that document's own R=4-random-orthogonal convention).

**Corroborating, secondary citation (not the primary mechanism, per
`research/ncr_separation_grounding.md` item 3's own nuance).** Liu et al.
(ICLR 2023 oral, arXiv:2210.10749, VERIFIED) show low-depth transformers
CAN construct exact shortcuts replicating finite-state-automaton
computation (their headline finding — meaning a sufficiently engineered
transformer is not categorically blocked from SOME automata, only from
the NC¹-hard ones per the TC⁰ argument above) but that these shortcuts are
BRITTLE out-of-distribution (their secondary finding — e.g. parity models
fail when the test bit-proportion differs from training). Cited here only
as corroborating evidence that even a successfully-trained Transformer
shortcut on an in-scope task tends not to generalize robustly — the
primary structural argument for Task 2 remains the TC⁰/NC¹ separation, not
this brittleness result.

### 3.3 Task 3 — Long-horizon entity state tracking

**Construction.** Extends `HEAD_TO_HEAD_DEMO_DESIGN.md`'s axis-2 M*
result (`STATE.md` §1.41: the contender held acc_A≥0.998 per-seed at every
tested horizon out to H8=1798 tokens, 8× T_bind, at a FIXED 32,768-byte
matrix state, on the synthetic Task-E harness) into real text: a document
introduces K entities via bind clauses, is followed by an arbitrary-length
span of REAL narrative/reasoning text (WikiText/OpenR1, unrelated content,
§5), then queries an entity's current relational state. The Transformer's
KV cache grows linearly with the intervening real-text span; NCR's state
does not grow at all. This is the direct real-LM analog of the M*
protocol's "constant-memory minds" property — not yet demonstrated outside
the isolated synthetic harness, which is exactly this task's contribution.

**Horizons.** Intervening real-text span ∈ {0, T_bind, 2×T_bind, 4×T_bind,
8×T_bind} tokens (T_bind = K×clause_len, K=32 primary → 224 tokens,
horizons up to 1,792 tokens of real intervening text — inside a single
seq_len=512–1024 training/eval window at the smaller horizons, requiring
seq_len≥2048 at the largest; see §6.4's saturation plan, which proposes
exactly this seq_len increase for an independent reason).

**Relation to Axis (ii) (query-time complexity, §4.4).** Task 3's relation
composition should use the SAME non-solvable-group construction as Task 2
where feasible (build-time decision — a build agent may descope Task 3 to
the abelian construction if the non-solvable-group encoder does not fit
the long-horizon harness cleanly; disclosed either way, not assumed). The
"constant-memory minds" property is orthogonal to the structural-vs-
sequential-rollout distinction — it is about MEMORY BYTES held constant
under a growing real-text span, not about sequential-step count, so it is
reported as its own axis (§7) regardless of which group construction
underlies it.

### 3.4 Citations — verified (2026-07-16, `research/
ncr_separation_grounding.md` + `research/ortho_write_grounding.md`,
coordinator-spot-checked; supersedes every prior `[TO-VERIFY]` tag)

| # | Citation | arXiv | Role in this design |
|---|---|---|---|
| C1 | Merrill & Sabharwal, "The Parallelism Tradeoff" | 2207.00729 | Log-precision transformers ⊆ TC⁰ — half of Task 2's structural-failure argument (§3.2) |
| C2 | Barrington, bounded-width branching programs / NC¹ | — (JCSS 38(1), 1989) | Non-solvable-group word problem is NC¹-complete — the other half of Task 2's argument |
| C3 | Merrill, Petty & Sabharwal, "The Illusion of State in State-Space Models" | 2404.08819 | Makes the TC⁰-vs-NC¹ deduction explicit against S₅; also applies to diagonal SSMs (relevant to any SSM baseline variant) |
| C4 | Liu, Ash, Goel, Krishnamurthy, Zhang, "Transformers Learn Shortcuts to Automata" | 2210.10749 | SECONDARY/corroborating only (§3.2) — shortcuts can replicate automata (headline) but are brittle OOD (secondary finding); do not conflate the two halves |
| C5a | Grazzi, Siems, Zela, Franke, Hutter, Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues" | 2411.12537 | Grounds §4.4's sequential-rollout baseline (CAN state-track via O(h) rollout) and §2.2's DeltaNet-β-range caveat |
| C5b | Siems, Carstensen, Zela, Hutter, Pontil, Grazzi, "DeltaProduct" | 2502.10297 | Sequential-rollout baseline candidate architecture (§4.4) |
| C6 | Peng et al., "RWKV-7 'Goose'" | 2503.14456 | Sequential-rollout baseline candidate architecture (§4.4); confirms O(h) sequential state-update is the published mechanism, not O(log h) query-time reads |
| C7 | Schlag, Munkhdalai & Schmidhuber, "Learning Associative Inference Using Fast Weight Memory" (FWM) | 2011.07831 | Closest prior art on in-context fast-weight writes for compositional inference (approximate, recursive reads, not matrix powers) |
| C8 | Guu, Miller & Liang, compositional path queries | 1506.01094 | SECONDARY empirical citation for Task 1 only (§3.1) — composition error-cascading, not a structural argument |
| C9 | "Log-Linear Attention" | 2506.04761 | Closest-named neighbor on "O(log ·)" — differentiated in §2.3 |
| C10 | "A Hippocampus for Linear Attention" (HOLA) | 2607.02303 | Exact verbatim KV recall, not exact algebraic composition — differentiated in §2.3 |
| C11 | Marchetti, Kunin, Myers, Acosta, Miolane, "Sequential Group Composition" | 2602.03655 | In-weights complement of NCR's in-context setting — differentiated in §2.3 |
| C12 | Nguyen, Vo, Vo, Nguyen & Pham, "MuonSSM" | 2606.30461 | Closest prior art to the orthogonal write specifically (§2.2, §9) — rank-1 transition-path magnitude-conditioning, differentiated from NCR's full-rank content-operator orthogonalization |
| C13 | Nichani, Lee & Bietti, "Understanding Factual Recall in Transformers via Associative Memories" | 2412.06538 | Grounds the standing `CLAUDE.md` rank-1-argmax-capacity hard rule (§4.2's flat-vector ablation); ICLR 2025 spotlight — VERIFIED via cross-reference to a prior in-repo fetch, flagged in the grounding memo for one human spot-check before any submitted-paper use |

**Retired from this document (do not re-introduce without a fresh
grounding check):** Looped Transformers (arXiv:2409.15647) and
depth-recurrent transformers (arXiv:2603.21676) — these were carried into
the first draft as the param-matched iterated-map baseline precedent
(`NOVEL_ARCH_WATERFALL.md`'s own M3 finding) but were NOT among the 8
items the grounding pass verified this round; MesaNet (arXiv:2506.05233)
— named in the first draft as a nearest-neighbor for matrix-function reads
at scale but likewise not in this round's verified set. Neither claim in
this design depends on them; re-verify before reintroducing.

### 3.5 Effective-distance stratification (mandatory, all tasks)

Every result table reports `(raw depth, depth mod K or path structure)`
pairs; no aggregate is computed across residue/structure strata; identity
and train-support points are excluded from claims and reported only as
the disclosed reducer-detection signature (`NOVEL_ARCH_WATERFALL.md`
§3.2's convention, applied without modification).

---

## §4 BASELINES

### 4.1 Param-matched Transformer (mandatory)

Reuses `matrix-thinking/deltanet_rd/transformer_baseline_rd.py`'s
architecture family, scaled to match the NCR-augmented DeltaNet model's
TOTAL param count (backbone + NCR head) at each rung, within the
program's own standing tolerance (`lm_rd_rung_configs.py`
`PARAM_COUNT_TOLERANCE = 0.15`, reused verbatim — no new tolerance
invented). Depth-matched to the DeltaNet backbone's own `n_layers` per
`HEAD_TO_HEAD_DEMO_DESIGN.md`'s R3-F3 precedent ("a depth-matched
Transformer is the more defensible 'same overall architecture depth
budget' comparison... not an unpinned choice"). No NCR head attached — a
plain Transformer LM trained on the identical curriculum (§5), same
tokenizer, same data budget.

### 4.2 Param-matched flat-vector ablation (MANDATORY per CLAUDE.md)

Direct reuse of `HEAD_TO_HEAD_DEMO_DESIGN.md`'s own ablation construction
(the "contender vs. ablation" arm that already WON at axis-1, n=3,
`STATE.md` §1.40 — contender acc_A 0.999–1.000 vs. ablation acc_A
0.032–0.037, chance-level): the SAME DeltaNet backbone + NCR read/write
pipeline, with the `d_ncr × d_ncr` matrix state replaced by a `d_ncr²`-dim
FLAT VECTOR (the `CLAUDE.md` reshape-equivalence rule — "any d²-dim vector
can be reshaped to d×d matrix and vice versa; structure only matters if
OPERATIONS preserve it") — the write becomes a vector update, the read
becomes a vector-indexed lookup with matched parameter count, and NO
matrix-power composition is available to it. **This ablation is NOT
assumed to transfer from the synthetic-task win** — it is re-run fresh at
LM scale, since the synthetic-task win was measured on the isolated
Task-E harness, and this document's own §1 hypothesis is specifically
about whether the SAME separation holds once the state must also support
ordinary next-token language modeling (a genuinely different regime, not
covered by the existing verdict).

### 4.3 KV-cache-memory-matched variant

Reuses `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.4.2's methodology (inference-only,
no new training — the trained §4.1 Transformer's own checkpoint, KV cache
hard-capped at `M ×` the contender's total inference-memory bytes, fp32
accounting, swept over a geometric `M` grid) with ONE re-derivation this
design's own arithmetic makes necessary: **the pre-registered M∈{1,2,4,8,
16,32} grid from the synthetic-task harness does not transfer to LM
scale and must be re-derived, not inherited.**

`cap_length(M) = M × state_bytes / (2 × n_layers × d_model × bytes_per_elt)`.
NCR's single-relation state (K=32, d_ncr=33, fp32): `33² × 4 = 4,356
bytes`. At the 98M backbone (`n_layers=12, d_model=768, fp32`):
`cap_length(M=1) = 4,356 / (2·12·768·4) = 4,356 / 73,728 ≈ 0.059 tokens`
— **below 1 token**, three orders of magnitude below the
`≈13–20`-token minimum-viable floor (query + one bind clause,
`HEAD_TO_HEAD_DEMO_DESIGN.md`'s own floor convention). Solving for the
`M` that clears a 20-token floor: `M ≈ 20 × 73,728 / 4,356 ≈ 339`. **The
synthetic-task grid's endpoint (M=32) is roughly 10× too small at real
98M-backbone scale** — one full 12-layer, 768-wide Transformer's per-token
KV cache already dwarfs NCR's entire single-relation state.

**Re-derived design.** Two variants, both pre-registered, neither
optional: (i) a wider `M` grid, geometric, `M ∈ {32, 64, 128, 256, 512}`
(clears the floor at every point, `cap_length(512) ≈ 512×4356/73728 ≈
30.2` tokens); (ii) the same grid applied to the STRUCTURED-BANK state
(Task 2, R=4–32 relations, total bytes `R × 4,356`) — at R=32,
`state_bytes ≈ 139,392`, putting the contender's own total inference
memory into a regime where `M=32` already clears `cap_length ≈ 966`
tokens, a genuinely "moderate long-context" comparison point. **This
finding — that NCR's single-relation state is dramatically smaller than
even one Transformer layer's per-token cache at LM scale — is itself
disclosed as a design fact, not spun**: it means axis 2 (§7) is either the
program's most dramatic result (if Task 3's horizon requirement is real)
or a mismatched comparison (if the capped Transformer can trivially win by
holding the entire relevant span in its cache at any affordable `M`) —
the pre-registered bands in §7 score both readings explicitly.

### 4.4 Sequential-rollout matrix-state baseline (MANDATORY, added per the
grounding update — NOT optional, distinguishes NCR on complexity, not
expressivity)

**Why this baseline is mandatory, not a nice-to-have.** Without it, §1's
hypothesis reduces to "a matrix-valued fast-weight model beats a
Transformer/flat-vector baseline at state tracking" — which, per the
grounding update, is **already published** (Grazzi et al., DeltaProduct,
RWKV-7, C5a/C5b/C6, §3.4) and would be a correctly-rejected claim if it
were this design's headline. This baseline exists specifically so the
program measures the axis that IS still open: query-time access
complexity at MATCHED expressivity, not accuracy at matched params.

**Construction.** A param-matched DeltaNet-family backbone configured to
escape TC⁰ by the SAME published mechanism as the baseline literature —
either (a) an extended-β-range DeltaNet (`β∈(0,2)`, unlocking negative
transition eigenvalues per Grazzi et al. arXiv:2411.12537), or (b) a
literal RWKV-7-style generalized-delta-rule layer (arXiv:2503.14456) —
a build-time choice, not resolved here, gated on which is cheaper to
attach to this repo's existing `fla`-based training path (§6.3's own
PRICE-UNKNOWN discipline applies: neither variant has a measured LM-scale
rate in this repo yet; both need their own Phase-0a line item). Trained on
the IDENTICAL curriculum and task supervision as the NCR-augmented model
(same bind-clause/word-problem episodes, §3), with NO separate NCR
read/write head — the backbone's OWN native recurrence is the only
state-tracking mechanism, exactly as in the published architectures.

**What is measured (accuracy is NOT the primary readout here).** At
matched depth `h`/`L`, this baseline is PREDICTED to reach comparable
in-distribution accuracy to NCR (both are expressivity-capable of the
task, per the published results above) — the primary readout is instead
**query-time sequential-dependency-chain length required to answer a
depth-h query**: for the rollout baseline this is Θ(h) (the recurrent
state must be updated h times, in sequence, to reach depth h, regardless
of how cheaply each step runs); for NCR it is Θ(log h) (binary
exponentiation, `NOVEL_ARCH_WATERFALL.md` §3's own Axis B convention,
reused directly). **This is the SAME wall-clock/wall-time methodology
`NOVEL_ARCH_WATERFALL.md` §3.2's Axis B already pre-registered
(≥10×-faster-at-large-h bar) — transplanted here as the LM-scale version
of that measurement, not a new instrument.** If accuracy diverges instead
of converging (the rollout baseline fails to match NCR's accuracy even at
matched depth), that is ALSO reported — it would mean this repo's own
DeltaNet-family implementation does not actually reach the published
expressivity ceiling at this scale, a genuinely different and also
publishable finding (an implementation/scale gap, not a theory gap).

---

## §5 DATA PLAN

### 5.1 What exists (verified, not assumed)

| Location | Content | Status |
|---|---|---|
| `/root/data/reasoning` (box, per `CLAUDE.md`) | OpenR1-Math, GPT-2 tokenized, 43.7M tokens | Cite `CLAUDE.md`'s own listing; `DELTANET_REALDATA_DESIGN.md` §3 separately confirmed the SSD-archived copy (`train_tokens=43,725,587`, `val_tokens=2,301,347`) as of 2026-07-02 — re-verify presence on the CURRENT box before build (boxes have been re-provisioned before, same document's own finding) |
| WikiText-103 (SSD, `wikitext103_tokenized/`) | GPT-2 tokenized, 117.9M train tokens | Same tokenizer as OpenR1-Math — zero-cost reasoning-vs-narrative pair, already the corpus FROZEN_BIAS_LM's own `openr1-mix-ext`/`wikitext-mix-ext` extended mixes are built from |
| `grammar_rd.py` word pools | 213 names / 21+16 relation verbs, GPT-2-single-token-verified | Reused directly for Tasks 1/2/3's bind-clause vocabulary |

### 5.2 Synthetic-to-real curriculum — options

**Option 1 (RECOMMENDED default) — interleaved documents.** Each training
batch mixes plain real-corpus documents (WikiText/OpenR1, ordinary
next-token LM objective, the SAME extended mixes `FROZEN_BIAS_LM_DESIGN.md`
already validated at these two scales) with synthetic grammar-episode
documents (Tasks 1–3, templated bind-clause English embedded via
`grammar_rd.py`'s real-word pools — NOT literal insertion into WikiText
prose, which is a harder unproven construction, §5.2 option 2). Mix ratio
scheduled: **Phase 1 (calibration) 50/50 synthetic/real** (maximize
write/read-plumbing learning signal per step, mirroring the naturalistic-
probe-task ranking `DELTANET_REALDATA_DESIGN.md` §2.5 gave this exact
approach over harder alternatives); **Phase 2 (main wave) 20/80
synthetic/real**, shifting weight toward genuine language-modeling
competence (a model that only ever sees synthetic grammar text is not
credibly evaluated as a "real LM" — the val-loss gate in §7 exists
specifically to catch a model that solves the synthetic task by
overfitting to a narrow synthetic distribution at the cost of general
language modeling).

**Option 2 (disclosed, not adopted this wave) — literal insertion.**
Splice bind clauses into real WikiText/OpenR1 sentences directly (e.g.
appending a clause after a real sentence boundary). Strictly harder to
build correctly (risk of breaking real syntax/tokenization boundaries,
risk of the model learning a spurious "clause always follows sentence-end"
positional shortcut) and NOT needed to test this document's §1 hypothesis
(the hypothesis is about depth/horizon exactness, not about naturalistic
embedding realism) — named here as a disclosed future refinement, not
built.

### 5.3 Held-out-depth / held-out-entity split design

Two independent held-out axes, both required, mirroring `grammar_rd.py`'s
own existing C17 (held-out entities)/C19 (held-out templates) controls:

1. **Held-out depth (h/L).** Train episodes use h∈{1,2,3} (Task 1) /
   L∈{1,2,3} (Task 2) exclusively; every eval ladder point is
   NEVER seen at any training step, at any residue (the single-full-K-cycle
   + distinct-operator-chain constructions of §3.1/§3.2 respectively).
2. **Held-out entities.** A disjoint entity-name pool (drawn from
   `grammar_rd.py`'s 213-name pool, split e.g. 170/43 train/eval,
   build-time-verified disjoint) ensures the model cannot memorize a fixed
   entity→operator mapping — every eval episode uses entity names the
   model's write pathway has never bound at training time, only the
   GENERAL binding/composition mechanism.

Both axes are combined multiplicatively in the eval manifest (held-out
depth × held-out entities), matching this program's own standing
discipline that a capability claim must survive BOTH axes, not just one.

---

## §6 COMPUTE PRICING

**Rule enforced throughout: every number is either cited from a measured
rate (source named) or marked PRICE-UNKNOWN / PROJECTED. No rate is
guessed.**

### 6.1 Measured rate anchors (the only ones that exist)

| Config | Rate | VRAM peak | Source |
|---|---|---|---|
| DeltaNet 98M (`d_model=768, n_layers=12, d_state=64`), batch=32, seq_len=512, ext-mix corpora | **0.236 s/step**, 4.478 GPU-h/67,547-step cell | **23.5 GB / 80 GB (29%)** | `FROZEN_BIAS_LM_DESIGN.md` §13.7, table "Verified realized rates actually used" — realized over 6 completed cells (26.87 GPU-h total), NOT a single-cell extrapolation |
| DeltaNet 392M (`d_model=1536, n_layers=16, d_state=128`), batch=32, seq_len=512, ext-mix corpora | **0.836 s/step**, 21.38 GPU-h/91,552-step cell (or 4.671 GPU-h/20,000-step reduced cell) | **39.0 GB / 80 GB (49%)** | `FROZEN_BIAS_LM_DESIGN.md` §13.7 — realized over 6 completed cells (128.3 GPU-h total), twice-cross-checked against an independent pre-registered estimate |
| Anchor-blend (per-token, per-position insertion) wall-clock overhead at 14M | **≤1.2% measured** (899–914s band across 20 cells, all arms) | n/a | `FROZEN_BIAS_LM_DESIGN.md` §13.4/VERDICT — the nearest measured analog to "small sequential op inserted at select token positions," used ONLY as a directional precedent for the NCR-overhead prediction (§2.2), never as NCR's own rate |
| NCR toy-scale training (K≤32, d_ncr=33, h_ncr=64, ~175–185K params), synthetic Task-E harness | 0.95–2.03 GPU-h per 160K–320K-step cell | "kilobytes-to-low-megabytes per example" (`NOVEL_ARCH_WATERFALL.md` §9.3); **NOT co-resident-optimized in prior runs — the rejected pattern this design's §6.4 corrects** | `NOVEL_ARCH_WATERFALL.md` §11.6 Tables 1–2; `NCR_ORTHO_WRITE.md` §6 |

### 6.2 Phased GPU-h ledger

**Phase 0 — smoke (both backbones, all 5 arms: NCR-augmented DeltaNet,
flat-vector ablation, plain DeltaNet control, plain Transformer control,
§4.4's sequential-rollout matrix-state baseline), 14M scale.**
Forward/backward/grad-flow-through-bf16-kernel-boundary check (per
`DELTANET_REALDATA_DESIGN.md` §4.3/§4.4's three-tier verification chain,
reused; the sequential-rollout arm additionally needs its own β-range or
generalized-delta-rule wiring verified against the same boundary). Cost
anchor: that design's own 14M wall-time band, 0.6–2.2 GPU-h per COMPLETE
400M-token run — smoke needs far less than a complete run. **Estimate: ≤3
GPU-h total** (bounded above by roughly one and a half full 14M runs'
own measured band, generously, to cover the fifth arm).

**Phase 0a — rate probe, ~2,000 steps/cell, ALL candidate configs
(NCR-augmented DeltaNet at 98M/392M; Transformer baseline at 98M/392M;
§4.4's sequential-rollout baseline at 98M/392M — this is the pilot that
RETIRES Option A's AND §4.4's PRICE-UNKNOWN flags, §6.3).**
DeltaNet-family cells priced from §6.1's measured per-step rate, scaled to
2,000 steps: 98M = 2,000×0.236s = 472s = **0.131 GPU-h/arm**; 392M =
2,000×0.836s = 1,672s = **0.464 GPU-h/arm**. Two DeltaNet-family arms
(NCR-augmented + flat-vector ablation) at both scales: `2×(0.131+0.464) =
1.19 GPU-h`. Transformer-family arm: rate UNKNOWN, priced at 2× the
DeltaNet rate as a placeholder UPPER BOUND ONLY (Transformers are not
known to be slower than DeltaNet at this scale — this is a deliberately
conservative ceiling, not a prediction) for BUDGETING purposes only —
**this placeholder must not be read anywhere else in this document as a
measured or predicted number**: `2×(0.262+0.928) = 2.38 GPU-h`. **Phase
0a total, 2× contingency: ≈(1.19+2.38)×2 ≈ 7.14 GPU-h.** No rung proceeds
past 0a without ITS OWN measured rate (`NOVEL_ARCH_WATERFALL.md` §9.4's
own discipline, reused verbatim) — the numbers above are a budgeting
ceiling, not a committed schedule.

**Phase 1 — calibration (MANDATORY per CLAUDE.md's own standing rule: "a
calibration run before a big sweep is mandatory, not optional").** 98M
only, reduced 20,000-step budget (this program's own validated
convention, `FROZEN_BIAS_LM_DESIGN.md` §13.7's citation of the mechanism-
wave's "FULL 20,000-step branch" as already-shown non-degenerate at 14M —
extended here to 98M as a Stage-1 calibration length, not assumed
transferable without re-checking Gate-0 convergence at this scale). 2
arms (NCR-augmented DeltaNet, flat-vector ablation) × n=2 seeds × Task 1
only = 4 cells. Rate: assume ≤5% NCR overhead (§2.2's prediction) on the
NCR arm (0.236×1.05=0.248 s/step), plain rate on the ablation arm.
NCR arm: 2 cells × 20,000×0.248s = 4,960s = 1.378 GPU-h/cell → 2.756
GPU-h. Ablation arm: 2 cells × 20,000×0.236s=4,720s=1.311 GPU-h/cell →
2.622 GPU-h. **Phase 1 total, 1×: 5.38 GPU-h; 2× contingency: 10.76
GPU-h.** Gate-0 precondition (`NCR_ORTHO_WRITE.md` §4's convention,
reused): in-distribution recovery ≥0.9 AND val-loss inside the standing
`k=2·s_ref` gate (`FROZEN_BIAS_LM_DESIGN.md` §13.4) — a cell failing
Gate-0 blocks Phase 2 (§8).

**Phase 2 — main wave, 98M (gated on Phase 1 passing Gate-0).** Full
67,547-step budget. Arms: NCR-augmented DeltaNet (priced at the
NCR-overhead-adjusted rate), flat-vector ablation, plain-Transformer
baseline (rate PENDING Phase 0a, priced here at Phase 0a's own 2×
placeholder ceiling — re-price before launch). n=3 seeds × 2 corpora
(openr1-mix, wikitext-mix) × 2 tasks-worth of eval (training is
task-suite-shared; eval-only passes are near-zero cost per
`FROZEN_BIAS_LM_DESIGN.md` §13.7's own eval-only line-item convention,
~0.02–0.05 GPU-h/pass class). NCR arm: 6 cells × 67,547×0.248s=16,752s=
4.653 GPU-h/cell → 27.92 GPU-h. Ablation arm: 6 cells × 4.428 GPU-h/cell
(measured plain-DeltaNet rate) → 26.57 GPU-h. Transformer arm: 6 cells ×
(placeholder ceiling) 8.86 GPU-h/cell (2× the DeltaNet rate, PENDING
re-price) → 53.16 GPU-h. **Phase 2 (98M) total, 1×: ≈107.7 GPU-h; 2×
contingency: ≈215.3 GPU-h — the Transformer arm's own placeholder pricing
is more than half this total, underscoring why Phase 0a is not
optional.**

**Phase 3 — main wave, 392M (gated on Phase 2's 98M readout AND the
ortho-write verdict, §9).** Same arm/seed/corpus structure, 20,000-step
reduced budget (the same disclosed deviation `FROZEN_BIAS_LM_DESIGN.md`
§13.7 already made for 392M — matching Track C's full 91,552-step budget
at n=3/n=3 would cost `18×21.38=384.8` GPU-h at 1× alone, an order of
magnitude beyond this design's own affordable scope). NCR arm: 6 cells ×
20,000×0.878s (0.836×1.05)=17,560s=4.878 GPU-h/cell → 29.27 GPU-h.
Ablation arm: 6 cells × 4.671 GPU-h/cell (measured) → 28.03 GPU-h.
Transformer arm: 6 cells × (placeholder ceiling) 9.342 GPU-h/cell → 56.05
GPU-h. **Phase 3 (392M) total, 1×: ≈113.3 GPU-h; 2× contingency: ≈226.7
GPU-h.**

**Grand total (Phases 0–3, 2× contingency, Transformer arm at its
UN-re-priced placeholder ceiling): ≈2 + 7.14 + 10.76 + 215.3 + 226.7 ≈
462 GPU-h.** This is a Stage-1 DESIGN ceiling, not a committed ask — it is
priced deliberately conservative (2× contingency stacked on a 2×
Transformer-arm placeholder) specifically so an attack round has a real
number to cut, not a vague "TBD." At the box's own ≈192 GPU-h/day
operative rate, this is ≈2.4 days of full-box supply if run
back-to-back-exclusive; in practice it competes with the ortho-write wave
(~77 GPU-h), fix-at-scale (300 GPU-h cap), and capability-separation
Stage 2 for the shared budget — a coordinator-level sequencing decision,
not resolved by this document.

**This total EXCLUDES §4.4's sequential-rollout matrix-state baseline
(added after the grounding update reframed Axis B as a primary claim, not
a stretch goal) — a load-bearing gap, disclosed, not silently absorbed.**
That arm needs its own Phase-0a rate probe (its architecture, β-range-
extended DeltaNet or RWKV-7-style, is not the plain DeltaNet backbone
§6.1 has measured) and its own training cells at every Phase 1–3 rung,
structurally the same shape as the NCR arm's own line items above. A
same-order-of-magnitude placeholder (mirroring the NCR arm's own 2× the
plain-DeltaNet-rate ceiling used for the Transformer arm, §6.2) would add
roughly another **NCR-arm-sized slice at each phase** — Phase 1: +2.76
GPU-h (1×); Phase 2: +27.9 GPU-h (1×); Phase 3: +29.3 GPU-h (1×) ≈ **+60
GPU-h at 1×, +120 GPU-h at 2× contingency**, revising the grand total to
**≈580 GPU-h at 2× contingency**. This revision is presented as an
order-of-magnitude flag for the attack round, not a firm number — an
attack round should treat §4.4's own pricing as unresolved and require a
real Phase-0a measurement before ANY Phase-1+ commitment, exactly as
items 1–2 below already require for the Transformer and NCR-overhead
arms.

### 6.3 PRICE-UNKNOWN list (explicit, per the task's own instruction — never guessed)

1. **Any real-corpus (512–1024 token) Transformer-LM training rate at
   98M or 392M params, on this box, at any batch size.** Only a
   short-episode (`T_bind≈224`), small-K synthetic-harness Transformer
   rate exists (`HEAD_TO_HEAD_DEMO_DESIGN.md`'s own timing pilots, a
   different regime entirely). Retired only by Phase 0a's own dedicated
   Transformer-arm pilot (§6.2).
2. **NCR-head wall-clock overhead on a real DeltaNet LM at 98M/392M.**
   The FLOP-ratio calculation (§2.2) predicts negligible, but this repo's
   own standing lesson (`NOVEL_ARCH_WATERFALL.md` §9.4) is that small ops
   are frequently kernel-launch-bound, not FLOP-bound — retired only by
   Phase 0a.
3. **Option C (pure fast-weight LM) in every column** — not priced, not
   built this wave (§2.3).
3a. **§4.4's sequential-rollout matrix-state baseline (extended-β-range
   DeltaNet or RWKV-7-style layer) at ANY scale.** No measured rate exists
   for either candidate architecture in this repo. The §6.2 grand-total
   revision above is a same-order-of-magnitude placeholder, not a price —
   retired only by a dedicated Phase-0a pilot for this arm specifically.
4. **Newton–Schulz `n_iter`'s exact step count and its own measured
   wall-clock cost at LM batch sizes** (`NCR_ORTHO_WRITE.md` §2 leaves
   `n_iter` "default set by §6 smoke" — that smoke has not run against an
   LM-scale batched call; the toy-scale ortho-write wave's own measured
   per-cell rates, §6.1's last row, are the nearest analog but at a
   drastically smaller batch/model, not a safe direct transfer).

### 6.4 SATURATION plan (PI directive, 2026-07-16 — sizing so each H100 runs at high SM utilization)

**The rejected pattern, stated precisely.** The toy-scale NCR/ortho-write
cells (K≤32, d_ncr≤33, h_ncr=64, ~175–185K params, §6.1's last row) were
run ONE PROCESS PER GPU, drawing on the order of ~50% SM and low
single-digit GB of VRAM — a tiny model's forward/backward pass simply
does not generate enough parallel work to saturate an H100 regardless of
how it is scheduled solo. This design explicitly does NOT repeat that
pattern for either its main training cells or its calibration cells.

**Main training cells (98M/392M, Phases 1–3).** Already far better
utilized than the toy cells (23.5 GB / 39.0 GB of 80 GB at batch=32,
seq_len=512 — 29% / 49%, not the ~2% the toy cells drew), but not yet
saturated. Two coupled levers, both re-measured before launch, neither
assumed:
1. **Raise batch size.** Propose a dedicated memory/timing pilot sweeping
   `batch ∈ {32, 48, 64, 96}` at 98M and `batch ∈ {32, 48, 64}` at 392M,
   logging `torch.cuda.max_memory_allocated()` + `nvidia-smi` cross-check
   at each point (the exact instrumentation `FROZEN_BIAS_LM_DESIGN.md`
   §13.10 gate 3 already built and used) and SM occupancy via
   `nvidia-smi dmon`. Target ≥80% of 80 GB and ≥80% SM utilization at the
   chosen operating point; do NOT assume linear VRAM-vs-batch scaling (a
   real model has a fixed model+optimizer-state floor that does not grow
   with batch, so the achievable batch increase is likely somewhat more
   than the naive ~3.4× (98M) / ~1.6× (392M) a linear projection from
   current headroom would suggest, but this must be MEASURED, not
   assumed — `CLAUDE.md`'s own "batch=112 fits training but OOMs during
   eval" lesson is the standing reason eval batch is capped and re-tested
   independently of train batch, applied here without exception).
2. **Raise `seq_len`.** Task 3 (§3.3) independently motivates
   `seq_len∈{1024, 2048}` (long-horizon intervening real text needs the
   context window to hold it) — this is not solely a saturation
   convenience; it is a genuine task requirement that HAPPENS to also
   raise tokens/step (hence FLOPs/step and VRAM) at fixed batch, a
   welcome coupling, not a coincidence. Re-measure batch×seq_len jointly,
   not batch alone.

**Calibration / Phase-0/0a cells (14M smoke, NCR-head-only rate probes).**
These remain genuinely too small to saturate a GPU via batch/seq scaling
alone without changing the task itself (the 14M backbone + 175K NCR head
is still a small model). **Packing plan: run 4–8 concurrent processes per
GPU** for Phase 0/0a only (mirrors `H100_SETUP.md`'s and
`DELTANET_REALDATA_DESIGN.md` §3's own standing "pack multiple tiny runs
per GPU" pattern, and is licensed by `NCR_ORTHO_WRITE.md` §6's own
"memory-trivial, co-resident-safe" finding for exactly this param range).
**Main 98M/392M cells (Phases 1–3) are NOT packed** — at a saturating
batch/seq_len (lever 1/2 above), a single 98M or 392M process is
projected to already occupy 70–90% of one GPU's VRAM, making co-residency
infeasible at the saturating operating point; packing and big-batch
saturation are mutually exclusive levers at this scale and this design
picks big-batch for the cells large enough to support it.

---

## §7 SUCCESS CRITERIA — draft pre-registerable WIN/PARTIAL/NULL bands

Reuses the `recovered_frac@0.9` bar and the HOLD(≥0.9)/DEGRADED((0.5,0.9))/
FAIL(≤0.5) band convention (`NOVEL_ARCH_WATERFALL.md` §3.2a) throughout,
extended to per-task, per-axis outcomes. **Gate-0 precondition (both
arms, every cell, per §6.2's Phase-1 convention): in-distribution
recovery ≥0.9 at train-support depth AND val-loss inside the standing
`k=2·s_ref` gate — a cell failing Gate-0 is DEAD and is not scored on any
axis below.**

**Framing correction (binding, per the grounding update, above): the two
PRIMARY axes are Task 2 (structural failure) and Axis B (query-time
complexity) — these two together are the load-bearing flagship claim.
Task 1 and Task 3 are SECONDARY/corroborating axes, reported alongside,
never sufficient alone for the overall program WIN.**

| Axis | WIN | PARTIAL | NULL |
|---|---|---|---|
| **Task 2 (PRIMARY — non-solvable-group structural failure, §3.2)** | NCR HOLDs (≥0.9) at `L*=32` (or the re-derived depth, §9) on BOTH held-out-entity and held-out-depth strata, AND the param-matched Transformer FAILs (≤0.5) at the same depth (the TC⁰/NC¹ structural prediction, C1–C3) — this is the flagship structural-failure result | NCR HOLDs/DEGRADEDs strictly better-banded than the Transformer, but not the full HOLD-vs-FAIL gap | NCR bands at or below the Transformer's band at `L*` — the structural prediction did not manifest empirically, reported as a genuine negative (§8 item 5) |
| **Axis B (PRIMARY — query-time complexity vs. §4.4's sequential-rollout baseline)** | Both NCR and the rollout baseline reach comparable in-distribution accuracy (within the standing ±15% band) at matched depth AND measured query-time wall-clock/dependency-chain length is ≥10× shorter for NCR at the largest tested depth (mirrors `NOVEL_ARCH_WATERFALL.md` §3.2's own Axis-B bar, transplanted) — the O(log h)-vs-O(h) separation, cleanly isolated from any accuracy confound | Scaling separation present but <10×, or accuracy diverges between the two arms in a way that confounds a clean complexity-only reading (reported, not hidden) | NCR is not measurably sub-linear in wall-clock vs. depth — flags a broken implementation, triggers diagnose-first, per the same document's own convention for a non-log-depth NCR result |
| **Task 1 (secondary — abelian/solvable exactness, empirical, §3.1)** | NCR HOLDs (≥0.9) at the re-registered `h*` on both held-out strata, AND the param-matched Transformer AND flat-vector ablation both FAIL (≤0.5) — an empirical-robustness result, NOT structural | Graded, mirrors `NOVEL_ARCH_WATERFALL.md` §3.2a's SEP-PARTIAL cell | NCR bands at or below either baseline |
| **Task 3 (secondary — "constant-memory minds," §3.3)** | NCR holds acc≥0.9 at the largest tested horizon (8×T_bind) AND the KV-cache-capped Transformer (re-derived `M` grid, §4.3) FAILs at that horizon — reproduces the M* "capping never rescues" finding (`STATE.md` §1.41) in a real-text setting | NCR holds where the capped Transformer fails, but the UNCAPPED Transformer also holds — NCR matches, not beats, an unconstrained baseline at fixed memory (itself a genuine, disclosed data point) | NCR itself fails past some horizon shorter than 8×T_bind |
| **Val-loss gate (all tasks, mandatory)** | Every arm's val loss stays inside `arm_off`'s own `mean + 2·s_ref` band (`FROZEN_BIAS_LM_DESIGN.md` §7.2 convention) | n/a — pass/fail gate, not graded | A cell whose NCR head measurably HURTS ordinary language modeling is reported plainly, not excused, even if a primary axis above WINs |

**Overall program verdict (cross-task rule, exhaustive, re-anchored to the
two-pronged flagship claim).** **WIN** requires BOTH primary axes at
WIN — Task 2 (structural failure) AND Axis B (query-time complexity) —
AND the val-loss gate passing everywhere; Task 1/Task 3 are reported
alongside as corroborating/secondary claims, never substitutable for
either primary axis (this is the direct correction of this document's
first-draft framing, which allowed Task 1 OR Task 2 alone to carry the
headline — no longer licensed once Task 1's citation was demoted to
empirical-only, above). **PARTIAL** = exactly one primary axis WINs
(structural-failure without the complexity separation, or vice versa) —
still publishable, reported as "one leg of the two-pronged claim landed,"
not spun as the full flagship result. **NULL** = neither primary axis
clears WIN or PARTIAL, or Gate-0 fails in ≥50% of cells (a trainability
failure at LM scale, distinct from and reported separately from a
capability failure).

---

## §8 RISKS & KILL CRITERIA — what a week-1 result kills before week 4

1. **Ortho-write verdict lands NULL or FAIL (§9) before Phase 1 launches.**
   KILLS the K=32 configuration outright for both Task 1 and Task 2;
   Stage-1 re-scopes to K≤15 (the pre-ortho-write "SCALES" ceiling,
   `NOVEL_ARCH_WATERFALL.md` §11.2) before ANY GPU-h beyond Phase 0 is
   spent. Does not kill the program — changes its headline numbers (§9).
2. **Phase 0 smoke shows the NCR head's gradient does not flow cleanly
   through the DeltaNet backbone's bf16 kernel boundary** (a real risk:
   `DELTANET_REALDATA_DESIGN.md` §4.3 already found `fla`'s production
   kernel categorically rejects fp32 — if the NCR head's own fp32-shadow
   read instrument, `NOVEL_ARCH_WATERFALL.md` §3.1, cannot be reconciled
   with a bf16-only backbone boundary, this blocks Option B specifically).
   KILLS Option B, forces a re-evaluation of Option A (now bearing its
   own full PRICE-UNKNOWN backbone cost) before any further spend.
3. **Phase 0a measures NCR-head wall-clock overhead >20%** (vs. the ≤5%
   predicted, §2.2). Does not kill the hypothesis, but KILLS the current
   compute pricing (§6.2) outright — every Phase 2/3 number above assumes
   ≤5%; a 4×+ miss on this single assumption invalidates the whole
   ledger and requires a full re-price before Phase 1 proceeds.
4. **Phase 1 calibration plateaus below Gate-0's 0.9 in-distribution bar**
   at 98M (the CLAUDE.md-mandated calibration-run purpose: "catches
   convergence ceilings... before you commit a sweep's compute to it").
   KILLS Phase 2 at the current config; does not kill the program —
   triggers a diagnosis round (budget/architecture-attachment-point
   re-check) before any re-attempt, exactly the K32-wall precedent's own
   discipline (`NOVEL_ARCH_WATERFALL.md` §11.6's ANOMALY-check handling).
5. **The param-matched Transformer baseline (§4.1) also HOLDs at Task 2's
   re-registered `L*`** (i.e., the TC⁰/NC¹ structural-failure prediction
   is simply wrong at LM scale — a real possibility even though the
   complexity argument is theoretically airtight, since a trained model
   could still exploit natural-language redundancy the synthetic harness's
   construction didn't allow, or Task 2's actual instance sizes could sit
   inside a regime where the TC⁰/NC¹ gap is not yet empirically visible
   at this model scale). This is the single most informative possible
   negative result for the PRIMARY structural axis. Reported plainly
   (`CLAUDE.md`: "negative results are data. Don't spin.") — does not kill
   the program (Axis B's complexity claim, item 6 below, is independent
   of this one), but caps the paper's structural-failure claim at
   "theoretically predicted, not empirically observed at this scale," a
   materially weaker headline than §1's hypothesis, and would redirect
   Stage-2 effort toward understanding WHY (larger K, larger group,
   different embedding) before any further scale-up.
6. **Axis B (§4.4) shows no wall-clock/dependency-chain separation** — the
   sequential-rollout baseline's measured query-time cost does not scale
   like O(h) relative to NCR's O(log h) (e.g. both are dominated by a
   fixed per-call overhead at the tested depths, or the rollout baseline's
   own implementation is unexpectedly parallel-friendly at this scale).
   Per `NOVEL_ARCH_WATERFALL.md` §3.2's own convention for this exact
   failure shape, this is flagged as an IMPLEMENTATION defect and
   triggers diagnose-first, not an immediate capability conclusion — but
   if diagnosis confirms the measurement is sound, it kills the SECOND
   primary axis and, combined with item 5, would collapse the overall
   verdict to NULL (§7) regardless of how Task 1/Task 3 read.

---

## §9 DEPENDENCIES — gated on the ortho-write verdict (~2026-07-17)

This entire document's K-axis choice, and therefore its Task 1/Task 2
headline depths, is downstream of `NCR_ORTHO_WRITE.md`'s pending verdict
on the Newton–Schulz orthogonal write (**closest prior art: MuonSSM,
arXiv:2606.30461 — see §2.2/§3.4 C12 for the full differentiation; that
paper's own ICML-2026-Oral result is independent evidence that
Newton–Schulz-orthogonalizing a fast-weight write is a live, reviewer-
legible mechanism, not exotic machinery**). **No Phase 1+ GPU-h is
authorized until that verdict lands and this section's branch is
resolved.**

- **If ORTHO-WRITE WIN** (`NCR_ORTHO_WRITE.md` §4 Part A: ortho-write
  median rec@0.9 at h*=40 ≥0.9, free-write baseline <0.5 at h=40; Part B:
  ortho-bank median rec@0.9 at L*=32 ≥0.9): Stage-1 build proceeds with
  **K=32, d_ncr=33** as the PRIMARY NCR configuration for BOTH Task 1
  (h*=40, secondary/empirical axis) and Task 2 (L*=32, PRIMARY/structural
  axis — the non-solvable-group generators, §3.2, must be re-verified to
  still satisfy the K=32 Hamiltonian-structure requirements the ortho-write
  wave calibrated against; S₅ has order 120 > K=32, so the entity pool and
  the group action must be reconciled at build time, not assumed
  compatible), using the Newton–Schulz orthogonal write throughout —
  exactly the configuration priced in §2/§6 above.
- **If ORTHO-WRITE PARTIAL** (cracks at h∈{20,29} but not 40, or L∈{12,20}
  but not 32): Stage-1 re-registers `h*`/`L*` DOWNWARD to match the
  cleared depth (e.g. h*=29, L*=20) — the same K=32/d_ncr=33 config stays,
  only the claimed depth shrinks; §6's GPU-h pricing is essentially
  unchanged (the same training budget, a shallower eval ladder point
  claimed as the headline); §7's WIN bands are re-anchored to the new
  `h*`/`L*` before Phase 1 launches, not after seeing Phase-1 data.
- **If ORTHO-WRITE NULL or FAIL** (no far-depth gain, or Gate-0 breaks):
  Stage-1 does NOT use K=32 at all. Falls back to **K=15, d≈16** (the
  pre-ortho-write free-write "SCALES" regime — 4/4 converged + far-depth
  HOLD, `NOVEL_ARCH_WATERFALL.md` §11.2) as the NCR configuration. This
  changes Task 2's structured-bank design (R×15-cycle products instead of
  R×32) and lowers Task 1's headline `h*` to whatever K=15's own
  free-write frontier supports (re-derive from that config's own archived
  z-dumps before pinning a number — not assumed here). If the failure
  mode is specifically FAIL (trainability breaks under the orthogonality
  constraint), this ALSO flags that Phase 0's smoke (§6.2) must include
  an explicit Gate-0 check at the LM-embedded K=15/32 configuration
  BEFORE any Phase-1 spend — the write-conditioning problem the
  ortho-write wave diagnosed at the isolated Task-E harness is not
  guaranteed to manifest identically once bind clauses are embedded in a
  real-LM training signal (a new confound this document does not resolve
  and must not assume away).

---

**Provenance.** This document supersedes nothing; it is a NEW Stage-1
design, not a revision of any existing registry. `research/
ncr_separation_grounding.md` and `research/ortho_write_grounding.md`
landed 2026-07-16 (coordinator-spot-checked) and are incorporated
throughout (§1, §2.2, §2.3, §3.1–§3.4, §4.4, §7, §9) — every citation in
this revision is VERIFIED or explicitly flagged, no `[TO-VERIFY]` tags
remain. Two items still need attention before any paper-facing claim
language is finalized: (a) `research/ncr_separation_grounding.md` item 8
(Nichani/Lee/Bietti rank-1-argmax capacity) is verified only via
cross-reference to a prior in-repo fetch, not this round's own tool
fetch — flagged for one human spot-check of the PDF directly; (b) the
S₅-order-120-vs-K reconciliation flagged in §9's WIN branch is a genuine
open build-time design question, not yet resolved. This document is
gated for build authorization on: (1) the ortho-write verdict (§9), (2)
an independent attack round on this document itself (which should target
the two-pronged §7 framing, the §9 S₅/K reconciliation, and §4.4's
entirely-unpriced cost, first).
