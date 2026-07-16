# NCR REAL-LM DESIGN — the flagship bet: NCR in a real language model at scale

**DRAFT-STAGE-1-REV-1 (POST-ATTACK-1, PRE-ATTACK-2). Opened 2026-07-16,
revised 2026-07-16 per §A1-ADJUDICATION.** This document is a design draft,
produced for adversarial attack. No code is built, no GPU is touched, by
this document. Every number below is either (a) cited from a measured rate
elsewhere in this repo, with the exact source named, or (b) flagged
`PRICE-UNKNOWN` / `PROJECTED — Phase 0a confirms`. Citations to external
literature were originally placeholder-tagged `[TO-VERIFY]`; the grounding
memos landed during the first drafting session (see "GROUNDING UPDATE"
immediately below) and every citation is either VERIFIED (source named,
§3.4's table) or explicitly flagged as still needing a human spot-check —
no `[TO-VERIFY]` tags remain in this document's claim language. **§1-§9
below are REVISED IN PLACE against ATTACK ROUND 1's BUILD-BLOCKED verdict
(§A1, §A1-ADJUDICATION, left untouched below as historical record); every
finding is mapped to its exact fix in the new §R1 REVISION 1 changelog at
the end of this document.**

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

## §1 HYPOTHESIS (REVISED, §R1(a) — fixes ATTACK ROUND 1's FATAL F1)

**§A1's F1 finding, adopted, not disputed:** non-solvable-group STRUCTURAL
hardness and O(log h) repeated-squaring composition are mutually exclusive
ON THE SAME QUERY FAMILY. A read via `binexp_read` computes `Z^h` for ONE
written operator; `{Z^h : h∈ℤ}` is cyclic, hence abelian, hence SOLVABLE —
Barrington's NC¹-completeness result (the entire structural-failure
citation, C1–C3) does not apply to it. Conversely, a non-solvable-group
word (a product of DISTINCT generators, no two of which commute in
general) is not a power of any single matrix, so repeated squaring cannot
compute it — the read costs Θ(L) matvecs, exactly the sequential-rollout
baseline's own cost. **This design's own binding sources already say so**
(`NOVEL_ARCH_WATERFALL.md` M4: "the relation chain … has NO squaring
shortcut … loses log-depth"; that document's B-CHAIN finding: a
query-controlled, growing chain length "loses log-depth"; `NCR_ORTHO_WRITE.
md` §4b: "`binexp` does not apply (the composite is not a power of one
matrix)", read = `loop_read`, O(L)). The flagship hypothesis below is
rewritten as an EXPLICIT CONJUNCTION of two results, each earned on the
query family it actually separates from, with the incompatibility
disclosed as a feature of the claim's construction, not hidden.

**Axis A — structural (Task 2, §3.2, read at O(L), NO complexity claim).**
On non-solvable-group word chains — a path of `L` DISTINCT written
generator-operators composed by exact matrix multiplication, one matvec
per hop (`o = loop_read(bank, path, q)`, cost Θ(L), stated plainly, not
disguised as sub-linear) — the NCR-augmented LM answers EXACTLY at path
lengths where the param-and-data-matched Transformer baseline fails for a
citable STRUCTURAL/complexity-theoretic reason: log-precision transformers
are contained in uniform TC⁰ (Merrill & Sabharwal, arXiv:2207.00729); the
word problem of any fixed non-solvable finite group is NC¹-complete
(Barrington 1989); therefore no log-precision transformer of any depth or
width computes this task's answer function unless TC⁰=NC¹ (Merrill, Petty
& Sabharwal, arXiv:2404.08819, making the deduction explicit against S₅).
This is a "cannot, not merely does not" prediction. NCR's own advantage
here is EXACTNESS (linear-algebraic composition never drifts), not speed —
its read is the same asymptotic order, Θ(L), as the sequential-rollout
baseline's.

**Axis B — complexity (Task-1-family single-operator power queries, §3.1,
read at O(log h), NO hardness claim).** On queries requiring repeated
application of ONE written operator — `Z^h` for a single Hamiltonian
K-cycle, a CYCLIC hence SOLVABLE group, carrying no TC⁰/NC¹ argument — the
SAME deployed model answers EXACTLY at query-time sequential cost O(log h)
via `binexp_read`'s repeated squaring, where every published matrix-state
alternative (DeltaProduct/RWKV-7-class — which, per Grazzi et al.
arXiv:2411.12537 and Peng et al. arXiv:2503.14456, CAN in principle reach
the correct answer) pays Θ(h) sequential state-update steps for the same
depth. This is an algorithmic/circuit-depth separation at MATCHED
expressivity and MATCHED accuracy, not an accuracy separation, and not a
hardness claim — the Transformer is not predicted to fail this task's
underlying group-word problem (§3.1's own citation is empirical
composition-drift, not structural).

**The flagship claim: ONE deployed model, trained once per scale on a
shared curriculum, delivers BOTH properties — each measured against the
baseline class it actually separates from, on the disclosed query family
that property is earnable on.** No single query family can carry both
properties simultaneously (§A1's F1 group-theory result, adopted above) —
this is a structural feature of the claim's own construction, disclosed
here plainly, not a gap papered over: a non-solvable word has no squaring
shortcut by definition, and a single operator's powers can never be
non-solvable. The two-family design is therefore not a workaround but the
only way to state a claim that is simultaneously true and non-vacuous.

**What this hypothesis explicitly does NOT claim** (the correction this
document's grounding update, above, forced, unchanged by this revision):
that matrix-valued fast-weight state can state-track where
vectors/Transformers categorically cannot — that is already published
(Grazzi et al., DeltaProduct, RWKV-7) and any framing implying otherwise is
retired from this design. Axis A's claim is narrower: EXACT composition
where the Transformer is structurally barred (TC⁰⊊NC¹). Axis B's claim is
narrower still: O(log h) query-time depth, unclaimed anywhere in the
searched literature (`research/ncr_separation_grounding.md` Part 2)
against baselines that are expressivity-capable but not depth-efficient.
Two falsifiable bets, not one, and §7 (revised) scores them independently
before combining them into an overall verdict.

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
ortho-write gate (§9, revised) licenses it. **`d_ncr` is now PER-TASK, not
one global constant (§R1(b) fix, M1)**: Task 1/Task 3's abelian
construction uses `d_ncr = K+1 ∈ {16, 33}` (the standing tight-spare
convention, gated on §9's K-axis branch); Task 2's non-solvable-group
construction uses `d_ncr,2 = d_min(G)+1` (4+1=5 for S₅ primary, §3.2) — a
much smaller matrix, decoupled from the K-axis entirely. A build either
runs two differently-shaped NCR head instances (one per task family,
disjoint parameters) or a single encoder padded to the larger shape with
the smaller task's write zero-padded/masked to its own `d_min` block —
**not resolved here, a build-time decision; both are architecturally
straightforward given the encoder is already per-relation, not shared
weight across d-shapes** — flagged so a build agent does not silently
assume one `d_ncr` for the whole model.

**Where reads happen.** At query positions, a read head computes the
task-appropriate read: Task 1/Task 3 (abelian, single-operator) use
`o = binexp_read(Z, q, h)` (scale-managed binary exponentiation, O(log h),
Axis B's own mechanism); Task 2 (non-solvable, distinct-generator path) use
`o = loop_read(bank, path, q)` (sequential per-hop matvec with per-step
renorm, O(L), Axis A's own mechanism, `NCR_ORTHO_WRITE.md` §4b's construction
transplanted) — **the two reads are NOT the same function and must not be
conflated (§A1 F1's binding correction)**. Either read injects its output
`o` into the residual stream at the query position (e.g. added to the
Transformer's own hidden state before the final LM head, or consumed by a
small MLP that produces logits directly for the query token — build-time
decision, not resolved here).

**Param-count delta.** Task 1/3 NCR head at the standing
K=32/d_ncr=33/h_ncr=64 convention: `P(d,h) = 40h² + 4dh + 46h + d`
(verified exact formula, `NOVEL_ARCH_WATERFALL.md` §9.3) = 40·4096 +
4·33·64 + 46·64 + 33 = 163,840 + 8,448 + 2,944 + 33 = **175,265 params**
(matches `NCR_ORTHO_WRITE.md` §6's independently-stated "~175K"). At a
98M/392M Transformer backbone this is **+0.18% / +0.045%** — negligible.
**Task 2's own NCR head, same formula at `d=d_ncr,2=5, h=64`** (§3.2,
holding `h_ncr=64` fixed as a default, not yet Phase-0-verified for the
much smaller task): `40·4096 + 4·5·64 + 46·64 + 5 = 163,840 + 1,280 +
2,944 + 5 = 168,069 params` — **barely smaller than the K=32 head**,
because the `40h_ncr²` encoder-width term dominates and is unaffected by
`d` shrinking. This is a load-bearing point for §6's pricing (M3/M4
below): shrinking the written-operator dimension does NOT proportionally
shrink cost, so the bridge cell (§6.2) must not be priced as "cheap
because `d` is small" — it is priced from measured toy rates instead,
scaled only by the (genuinely reducible) step/seed budget.

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
way (this is where §1 Axis B's O(log h)-vs-O(h) framing meets this
specific backbone — §R1(f), M5 note: this caveat is scoped to Axis B/
Task 1 only, per §A1 F1; it says nothing about Task 2, whose baseline is
the Transformer, §4.1, on structural grounds).** DeltaNet's native
transition is `I − βkkᵀ` (a Householder reflection); the standard sigmoid
gate restricts `β∈(0,1)`, which per Grazzi et al. (arXiv:2411.12537)
restricts transition eigenvalues to `[0,1]` and provably BARS
parity/non-solvable-group state tracking — i.e., under that gate the
PLAIN-DeltaNet-control arm (§4's implicit baseline, the backbone with no
NCR head) is ITSELF still TC⁰-limited on any hypothetical non-solvable
task (same class as diagonal SSMs, same Merrill/Petty/Sabharwal argument),
though this does not bear on Task 2 (§3.2) since Task 2's structural
baseline of record is the Transformer, not plain DeltaNet. If the box's
`fla` kernel or a config flag instead allows `β∈(0,2)` (unlocking negative
eigenvalues) BY DEFAULT, the plain-DeltaNet-control arm already escapes
TC⁰ and becomes behaviorally close to §4.4's own dedicated Axis-B
sequential-rollout baseline — in which case §4.4's extended-β arm (§R1(f)
pins it to exactly this mechanism) may be the SAME architecture as the
plain-DeltaNet control, and the two arms should be checked for redundancy
at Phase 0 rather than built twice. **Verify `deltanet_core.py`'s actual
`β` range (and, per §4.4's pin below, whether the custom block's `b_proj`
gate is the right patch point to extend it) at Phase 0 (§6.2)** before any
claim language is finalized. Either reading is fine for NCR's own
contender (Axis B's read-time complexity claim does not depend on which
regime the backbone sits in), but it changes what the plain-DeltaNet
CONTROL arm is allowed to be called in the results write-up, and whether
§4.4 needs a genuinely separate build.

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

## §3 TASK SUITE (REVISED, §R1 — F1/M1/M2/M3 fixes; the two axes now sit on
TWO DISCLOSED, DISJOINT query families, per §1's rewritten conjunction)

**Two-family design, per §1's F1-fixed conjunction (§A1 confirmed: the two
properties cannot coexist on one family).** Axis A — **structural**: Task 2
(§3.2, PRIMARY for this axis) composes a PATH of DISTINCT generator
operators drawn from a non-solvable group; the Transformer baseline fails
for a citable complexity-theoretic reason (TC⁰⊊NC¹); NCR's own read here is
Θ(L) — exact, not fast — via `loop_read`, stated plainly, never called
O(log L). Axis B — **complexity**: Task 1 (§3.1, PRIMARY for this axis, no
longer "secondary" — it now carries the ENTIRE O(log h) claim) composes
repeated powers of ONE operator (a cyclic, solvable group — no TC⁰/NC¹
citation applies, and none is claimed); measured against a param-matched
SEQUENTIAL-ROLLOUT matrix-state baseline (§4.4, an extended-eigenvalue-
DeltaNet pin, §R1(f)) that CAN in principle reach the correct answer (per
Grazzi et al./Peng et al., already published) but requires Θ(h) sequential
steps — NCR's claim is O(log h) query-time depth via `binexp_read`'s
repeated squaring against that baseline's Θ(h), a wall-clock/dependency-
chain-length separation, not an accuracy separation. **Task 2 is no longer
labeled "SECONDARY" and Task 1 is no longer labeled the "pure exactness
companion"** — both are now PRIMARY, each for the axis it actually earns
(§7, revised, scores them independently). Task 3 (§3.3) remains secondary/
corroborating on a third, orthogonal axis (memory bytes, not sequential
depth). Every task below is instrumented with the mod-K-collapse guards
`CLAUDE.md` mandates.

### 3.1 Task 1 — Depth-robust relational composition embedded in text
(PRIMARY for Axis B — complexity/O(log h), abelian/solvable construction,
no hardness claim)

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
demonstrates a real KIND difference on accuracy (bounded numerical drift
vs. exact linear algebra) — the correct claim register for THAT
comparison is "empirically more robust," not "structurally impossible for
the baseline." **But accuracy is not this task's primary job in the
revised design (§R1(a)): Task 1 is where Axis B's O(log h)-vs-Θ(h)
query-time-cost separation lives (§4.4), because it is the one family
where a squaring shortcut mathematically exists** (single-operator power,
cyclic group). Task 2 carries the structural (Axis A) claim on a disjoint
family where no such shortcut exists (§3.2, below) — companions on
different axes, neither subordinate to the other.

### 3.2 Task 2 — Non-solvable-group word-problem chain (PRIMARY for
Axis A — structural failure, well-posed construction, §R1(b) fixes M1)

**M1, as adjudicated: construction rewritten to use the GENERATOR SET, not
a permutation acting on the K=32 entity pool.** The first draft conflated
two incompatible objects (`CAPABILITY_SEPARATION_DESIGN.md`'s continuous
`ℝ^{d_min}` rotation-representation infra vs. a discrete permutation on 32
named entities that infra was never built for — §A1 M1's evidence: S₅ has
no transitive action on 32 points, `32 ∤ 120`). The fix, below, pins one
concrete, well-posed object and decouples Task 2 from Task 1/3's K-axis
entirely.

**Group and generators (VERIFIED, reused verbatim from
`CAPABILITY_SEPARATION_DESIGN.md` §1.3's real, calibrated group
infrastructure — no new matrices built).** Primary group: **S₅** (order
120, non-solvable, `d_min=4`, realized as the 4-dimensional standard
representation on the zero-sum hyperplane of ℝ⁵ — `CAPABILITY_SEPARATION_
DESIGN.md` line 229). Its verified symmetric generating set is
`{t, c, c⁻¹}` (that document's own line 897): **`t`** = a transposition
(self-inverse) and **`c`** = a 5-cycle (order 5) — **exactly the 2
generators (`t`, `c`) the adjudication names**, `c⁻¹` included for the
symmetric closure (size 3) that licenses "walk either direction" paths.
Each generator is written as its OWN `4×4` rotation-rep operator (not a
`33×33` or `K×K` matrix) — `d_ncr,2 = d_min(S₅)+1 = 5` (the `+1` tight-
spare margin, mirroring Task 1's `K+1` convention, an ASSUMPTION flagged
for the bridge cell, §9, to confirm or revise).

**The natural action — and where K=5 for Task 2 comes from.** S₅'s
*defining* action is the permutation action on its own **5 letters**; the
4-dim standard rep is exactly that action restricted to the zero-sum
hyperplane (the images of the 5 standard basis vectors, projected, are 5
canonical points in ℝ⁴, and permuting coordinates in ℝ⁵ restricts to
composing the rotation matrices in ℝ⁴). **Task 2's entity pool is
therefore K₂ = 5** — the 5 letters S₅ naturally acts on — filled per
episode by 5 DISTINCT names drawn from `grammar_rd.py`'s 213-name pool
(the held-out-entity split, §5.3, applies exactly as before: a disjoint
train/eval NAME-set split, e.g. 170/43, reassigned onto the same 5
abstract roles per episode — the same mechanism Task 1 already uses,
just filling 5 roles instead of `K`). **This is a genuinely different,
much smaller K than Task 1/3's `K∈{15,32}` — the two are DECOUPLED, not
shared, per the adjudication's explicit instruction.** A query specifies a
PATH of generator indices `(o_1,…,o_L)` from the size-3 symmetric set; the
GROUND TRUTH is the exact permutation composite `g_{o_L}∘…∘g_{o_1}`
applied to the query letter, computed by exact integer index arithmetic —
no floating point anywhere in the target. The MODEL's own answer is
decoded by applying the composed (orthogonally-conditioned) `4×4`
operators to the query point's `ℝ⁴` image and reading off which of the 5
canonical points it lands nearest (cosine similarity over exactly 5
candidates — a legitimate discrete decode here because Task 2 tests
COMPOSITION EXACTNESS, not a rank-K necessity bound; the `CLAUDE.md`
argmax-defeats-rank-proofs rule governs a different, unrelated concern and
does not apply to this readout).

**R (operator/generator count) reconciliation — the other half of M1's
fix.** Task 2's written-operator count is **R = 3** (`{t, c, c⁻¹}`, the
verified generating set size) — **NOT R=4** (the arbitrary
random-orthogonal convention `NCR_ORTHO_WRITE.md` §4b's discriminator cell
used for its own calibration purpose) and **NOT K=32** (the entity-pool
conflation §A1 M1 caught). **What K and R now mean, per task, stated
explicitly (the adjudication's own instruction):**

| Task | K (entity pool) | R (written operators) | Group / structure |
|---|---|---|---|
| Task 1 (§3.1) | `K∈{15,32}`, gated by §9's main ortho-write branch | 1 (single operator, `Z^h`) | Cyclic (solvable) |
| Task 2 (§3.2) | **K₂=5, FIXED, decoupled from Task 1's K** | **R=3** (`t, c, c⁻¹`) | S₅ (non-solvable) |
| Task 3 (§3.3) | Task 1's `K` (abelian default) OR K₂=5 (if it reuses Task 2's construction, build-time choice, unchanged) | Whichever it inherits | Build-time choice, disclosed either way |

**Why this is the structural, not empirical, failure claim (unchanged by
this revision).** The word problem of any fixed non-solvable finite group
is NC¹-complete (Barrington, JCSS 1989, VERIFIED — `research/
ncr_separation_grounding.md` item 2); log-precision transformers are
contained in uniform TC⁰ (Merrill & Sabharwal, TACL 2023, arXiv:2207.00729,
VERIFIED); therefore NO log-precision transformer of ANY depth or width
can compute this task's answer function unless TC⁰=NC¹ — a
conjectured-false complexity-class separation, not a training/data
artifact (Merrill, Petty & Sabharwal, ICML 2024, arXiv:2404.08819,
VERIFIED, makes exactly this deduction against S₅). **This is a genuine
"cannot, not merely does not" prediction** — the strongest available
structural-failure citation in this design, and the reason Task 2 is
Axis A's PRIMARY task. **NCR's own read here costs Θ(L)** (`loop_read`,
§2.1, revised) — the structural claim is about the Transformer's
CATEGORICAL inability, not about NCR being fast; see §1's revised
hypothesis for why these do not conflate.

**Mod-K-trap-proof by construction (three guards, `NCR_ORTHO_WRITE.md`
§4b's convention, transplanted verbatim, now doing double duty as BOTH the
held-out-depth hygiene AND part of the non-solvable-word-problem
structure):** (1) distinct generators per hop — a product of DIFFERENT
group elements, not a power of one matrix, so there is no single cycle
length to reduce modulo; (2) no consecutive repeats; (3) fixed-point
exclusion (query/path pairs whose composite fixes the start are excluded).

**L-grid.** Train L∈{1,2,3}; eval L∈{5,8,12,16,20,24,32,40} (the
`NCR_ORTHO_WRITE.md` §4b ladder, reused for its DEPTH values only — R is
now 3, generators are S₅'s `{t,c,c⁻¹}`, not R=4 random-orthogonal
matrices; this is the corrected build-time change from that document's own
convention, replacing the pre-Rev-1 draft's incorrect "R=4 primary"
claim).

**A5/A6 hard-stop history — disclosed accurately (§R1(b), the other half
of M1's fix).** `CAPABILITY_SEPARATION_DESIGN.md`'s Rev 6 (§1.28) recorded
A5/A6 as HARD-STOPPED (a second consecutive convergence-budget miss on
THAT document's own restricted-rank-RECOVERY task — an SVD-probe readout
over a continuous state, a different loss landscape and readout than Task
2's discrete word-composition-and-next-token task); Rev 7 (§1.30–1.31)
subsequently diagnosed the mechanism (H-ENC: an `L=1` query-independent-
attention degeneracy specific to single-generator words, not a general
training pathology) and LIFTED the hard-stop, re-pinning per-group step
budgets (S₅'s own budget: 8,000 steps) that all 5 groups now clear.
**What transfers to Task 2 and what does not:** the GENERATOR MATRICES
(S₅'s verified 4-dim standard rep, the `{t,c,c⁻¹}` set, its closure/order
checks) are real, exact, and reused verbatim — this much genuinely "is
already built." The CALIBRATION (the step budgets and Gate-1 bars above)
was earned on a DIFFERENT task (continuous rank recovery, not discrete
composition-and-next-token) and does **not** transfer by fiat — the
pre-Rev-1 draft's "already built and calibrated" was an overclaim on
exactly this point (§A1 M1's finding). Task 2 needs its OWN Gate-0
calibration on its OWN construction; the bridge cell (§9, revised, §R1(d)
fixes M3) is the mechanism that earns it, gating Task 2's Phase 1 launch
independently of the main K-axis ortho-write verdict.

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
8×T_bind} tokens (T_bind = K×clause_len; if Task 3 inherits Task 1's
abelian construction, K=32 primary → 224 tokens, horizons up to 1,792
tokens of real intervening text; if it inherits Task 2's construction,
K₂=5 → T_bind is far shorter, and the horizon grid should be re-derived
in T_bind multiples at build time rather than importing Task 1's absolute
token counts — **§R1(b) note, flagged, not resolved here**: the K-value
in this paragraph is now build-time-conditional per §3.2's K/R table,
not a single global K) — inside a single seq_len=512–1024 training/eval
window at the smaller horizons, requiring seq_len≥2048 at the largest;
see §6.4's saturation plan, which proposes exactly this seq_len increase
for an independent reason.

**Relation to Axis A/Axis B (§R1(a) renaming — was "Axis (ii)", now
disambiguated per the two-family conjunction, §1).** Task 3's relation
composition should use the SAME non-solvable-group construction as Task 2
where feasible (build-time decision — a build agent may descope Task 3 to
Task 1's abelian construction if the non-solvable-group encoder does not
fit the long-horizon harness cleanly; disclosed either way, not assumed —
and per §3.2's K/R table, this choice also decides which K/T_bind Task 3
inherits, above). The "constant-memory minds" property is orthogonal to
the Axis-A/Axis-B distinction — it is about MEMORY BYTES held constant
under a growing real-text span, not about sequential-step count or
structural hardness, so it is reported as its own third, secondary axis
(§7, revised) regardless of which group construction underlies it.

### 3.4 Citations — verified (2026-07-16, `research/
ncr_separation_grounding.md` + `research/ortho_write_grounding.md`,
coordinator-spot-checked; supersedes every prior `[TO-VERIFY]` tag)

| # | Citation | arXiv | Role in this design |
|---|---|---|---|
| C1 | Merrill & Sabharwal, "The Parallelism Tradeoff" | 2207.00729 | Log-precision transformers ⊆ TC⁰ — half of Task 2's structural-failure argument (§3.2) |
| C2 | Barrington, bounded-width branching programs / NC¹ | — (JCSS 38(1), 1989) | Non-solvable-group word problem is NC¹-complete — the other half of Task 2's argument |
| C3 | Merrill, Petty & Sabharwal, "The Illusion of State in State-Space Models" | 2404.08819 | Makes the TC⁰-vs-NC¹ deduction explicit against S₅; also applies to diagonal SSMs (relevant to any SSM baseline variant) |
| C4 | Liu, Ash, Goel, Krishnamurthy, Zhang, "Transformers Learn Shortcuts to Automata" | 2210.10749 | SECONDARY/corroborating only (§3.2) — shortcuts can replicate automata (headline) but are brittle OOD (secondary finding); do not conflate the two halves |
| C5a | Grazzi, Siems, Zela, Franke, Hutter, Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues" | 2411.12537 | **PINNED (§R1(f), M5)** as §4.4's sequential-rollout baseline ARCHITECTURE — extended-eigenvalue DeltaNet (`β∈(0,2)`), scoped to Axis B/Task 1 only (§1, §A1 F1) — and grounds §2.2's DeltaNet-β-range caveat |
| C5b | Siems, Carstensen, Zela, Hutter, Pontil, Grazzi, "DeltaProduct" | 2502.10297 | Alternative sequential-rollout candidate, NOT pinned (§4.4 justifies choosing C5a's extended-β DeltaNet instead, cost grounds) |
| C6 | Peng et al., "RWKV-7 'Goose'" | 2503.14456 | Alternative sequential-rollout candidate, NOT pinned (§4.4); confirms O(h) sequential state-update is the published mechanism, not O(log h) query-time reads |
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

### 4.3 KV-cache-memory-matched variant (REVISED, §R1(g) — fixes M6, and
recomputed a second time following M1's Task-2 dimension fix)

Reuses `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.4.2's methodology (inference-only,
no new training — the trained §4.1 Transformer's own checkpoint, KV cache
hard-capped at `M ×` the contender's total inference-memory bytes, fp32
accounting, swept over a geometric `M` grid) with the re-derivation this
design's own arithmetic makes necessary: **the pre-registered M∈{1,2,4,8,
16,32} grid from the synthetic-task harness does not transfer to LM
scale and must be re-derived, not inherited.**

`cap_length(M) = M × state_bytes / (2 × n_layers × d_model × bytes_per_elt)`.
At the 98M backbone (`n_layers=12, d_model=768, fp32`): denominator
`= 2·12·768·4 = 73,728`.

**Case (i) — Task 1's single-relation state (K=32, d_ncr=33, fp32):**
`state_bytes = 33² × 4 = 4,356`. `cap_length(M=1) = 4,356/73,728 ≈ 0.059
tokens` — below 1 token, far under the `≈13–20`-token minimum-viable
floor (query + one bind clause, `HEAD_TO_HEAD_DEMO_DESIGN.md`'s own floor
convention). Floor-clearing minimum: `M ≈ 20×73,728/4,356 ≈ 339`. **§A1
M6, CONFIRMED and extended: the pre-Rev-1 draft's own proposed grid
`M∈{32,64,128,256,512}` does NOT clear the 20-token floor at every point
either** — recomputing each point: `cap_length(32)=32×4356/73728≈1.89`,
`(64)≈3.78`, `(128)≈7.56`, `(256)≈15.12`, `(512)≈30.24` tokens — only the
grid's OWN top end (M=512) clears 20 tokens; M=32 is nowhere near it
(1.89, not the erroneously-quoted 966, which was §A1's finding for a
different R/d entirely, below). **Re-derived grid, anchored above the
actual floor-clearing minimum (339):** `M ∈ {384, 768, 1536, 3072, 6144}`
→ `cap_length ≈ {22.7, 45.3, 90.7, 181.4, 362.8}` tokens — clears the
floor at every point, geometric (2×) spacing preserved. (If §9's K-axis
branch falls back to K=15/d=16, `state_bytes=1,024`, floor-clearing
minimum `M≈1,440` — a build agent must re-derive this sub-grid fresh
under that branch, not reuse the K=32 numbers; flagged, not computed
here since it is conditional on an unresolved §9 branch.)

**Case (ii) — Task 2's structured-bank state, RECOMPUTED under §3.2's own
M1 fix (R=3, d_ncr,2=5 — NOT the pre-Rev-1 draft's R=4–32 at d=33, which
was itself an instance of the K/R conflation §A1 M1 caught).**
`state_bytes = R × d² × 4 = 3 × 25 × 4 = 300` bytes — **even smaller than
Task 1's single-relation state**, not larger (the pre-Rev-1 draft's
`R×4,356` formula silently assumed Task 2 shared Task 1's `d=33`, which
M1's fix retires). `cap_length(M=1) = 300/73,728 ≈ 0.00407` tokens.
Floor-clearing minimum: `M ≈ 20×73,728/300 ≈ 4,916`. Re-derived grid:
`M ∈ {5120, 10240, 20480, 40960, 81920}` → `cap_length ≈ {20.8, 41.7,
83.3, 166.6, 333.3}` tokens — clears the floor throughout, and lands in
the SAME couple-hundred-token ballpark as Case (i)'s top end at its own
top end, a coherence check between the two re-derived grids.

**The finding, restated and sharpened by the M1 fix, not softened.**
NCR's total inference-memory footprint is now shown to be smaller — by a
FURTHER order of magnitude on Task 2 — than the pre-Rev-1 draft's own
already-dramatic finding: an affordable Transformer KV cache (`M` in the
low thousands) already dwarfs Task 2's entire written-operator bank. This
is disclosed as a design fact, not spun: it means Task 3 (§7, revised) is
either the program's most dramatic memory-asymmetry result (if a real
long-horizon requirement is demonstrated) or a mismatched comparison (if
the capped Transformer can trivially win by holding the entire relevant
span in cache at any affordable `M`) — the pre-registered bands in §7
score both readings explicitly, unchanged in structure by this
recomputation, only in the numbers feeding it.

### 4.4 Sequential-rollout matrix-state baseline (MANDATORY; REVISED,
§R1(f) — fixes M5: architecture PINNED, ≥10× bar REPLACED, scope
NARROWED to Axis B/Task 1 only per §A1 F1)

**Why this baseline is mandatory, not a nice-to-have.** Without it, §1's
Axis B hypothesis reduces to "a matrix-valued fast-weight model beats a
Transformer/flat-vector baseline at state tracking" — which, per the
grounding update, is **already published** (Grazzi et al., DeltaProduct,
RWKV-7, C5a/C5b/C6, §3.4) and would be a correctly-rejected claim if it
were this design's headline. This baseline exists specifically so the
program measures the axis that IS still open: query-time access
complexity at MATCHED expressivity, not accuracy at matched params.
**Scope, per §A1 F1's binding correction: this baseline is compared
against NCR on Task 1's family ONLY (§3.1, single-operator power
queries)** — Task 2 has no O(log L) read to compare against (§1, §2.1,
Axis A's read is Θ(L) via `loop_read`, the same order as this baseline),
so this baseline's own complexity comparison does not apply to Task 2.

**Architecture, PINNED (was: "a build-time choice, not resolved here" —
the M5 finding this fixes): extended-eigenvalue DeltaNet (Grazzi et al.,
arXiv:2411.12537), NOT RWKV-7.** Justification: this repo's `fla`-based
DeltaNet training path already calls the delta-rule kernel through a
CUSTOM block, not the stock `fla.layers` module, specifically because the
stock layer computes `β` internally via its own `b_proj` and exposes no
mask hook (`DELTANET_REALDATA_DESIGN.md` §4.3/F15-LM, VERIFIED — this
repo's own prior finding). That custom block is therefore ALREADY the
right patch point: extending `β`'s range from the standard sigmoid's
`(0,1)` to `(0,2)` (per Grazzi et al.'s own construction, unlocking
negative transition eigenvalues) is a SCOPED change to one gate
computation inside code this repo already owns and has kernel-verified at
LM sequence lengths (`T≥512`, bf16, Tier-0/Tier-1 gradcheck PASS,
`DELTANET_REALDATA_DESIGN.md` §4.3/§4.4) — it reuses the SAME measured
chunked-kernel path §6.1 already prices, differing only in the gate's
output range. A literal RWKV-7-style layer would instead require an
entirely new kernel with ZERO measured rate anywhere in this repo and its
own from-scratch Tier-0/Tier-1 verification chain — strictly more
PRICE-UNKNOWN surface for no argued scientific advantage (both C5a and C6
are equally valid per the published literature; this design picks the
cheaper-to-verify one). **This is a cost-grounded engineering choice, not
a scientific claim about which mechanism is superior — flagged for attack
scrutiny like every other pinned choice in this document.** Trained on
Task 1's IDENTICAL curriculum and task supervision (§3.1), with NO
separate NCR read/write head — the backbone's own extended-β recurrence is
the only state-tracking mechanism, exactly as in the published
architecture. **Phase-0a (§6.2) prices this arm's actual per-step rate
before any further commitment** — the extended-`β` gate is a small
compute change (one scalar's range) but its wall-clock effect on the
chunked kernel's numerics/stability is unmeasured and must not be assumed
equal to the standard-gate rate.

**What is measured — REPLACING the transplanted ≥10× bar (M5's core
finding: that bar was earned at `h≈10³–10⁶` in an isolated toy loop,
`NOVEL_ARCH_WATERFALL.md` §3.2, not at this design's own claimable
regime).** Two readouts, one architectural (primary, hardware-independent,
not a measurement) and one wall-clock (secondary, the empirical
confirmation):

1. **Dependency-chain length (PRIMARY, provable by construction, not
   fit).** NCR's read on Task 1 executes exactly `⌈log₂ h⌉` sequential
   matrix operations (`binexp_read`'s own instrumented step counter); the
   rollout baseline's read executes exactly `h` sequential state updates
   (one recurrent step per depth unit, by the architecture's own
   definition — Grazzi et al.'s construction has no shortcut around this).
   This is an exact integer count per query, asserted once via a build-time
   instrumentation check (tag every op on the critical read path with a
   monotonic counter, assert the counter total matches the formula) — a
   PASS/FAIL build gate, not a statistical claim, and the primary evidence
   because it is hardware-independent (does not depend on kernel-launch
   overhead, GPU generation, or batching).
2. **Wall-clock (SECONDARY, empirical confirmation, fit not asserted).**
   Measured at an EXTENDED depth ladder reaching well beyond Task 1's
   accuracy-claimable ladder (`{5,…,61}`, §3.1) — timing is
   accuracy-independent (§A1 M5's own observation, adopted), so this
   ladder may probe far deeper purely to separate the two growth curves,
   mirroring `NOVEL_ARCH_WATERFALL.md` §3.2's own precedent of measuring
   Axis B out to `h=2^20+5` while the accuracy ladder stopped at 125.
   **Ladder: `h ∈ {61, 200, 1000, 5000, 20000}`** (PROJECTED, subject to
   Phase-0a feasibility — if the rollout baseline's per-step cost makes
   `h=20000` impractically slow in wall-clock, the largest FEASIBLE point
   measured is used instead; the fit below does not require this exact
   ladder, only enough spread in `log h` to separate the two functional
   forms). At each `h`, fit two nested OLS models per arm: `Model_log:
   t = a + b·log₂(h)`; `Model_lin: t = c + d·h`. **Pinned statistical
   criterion (a deliberate relaxation of `NOVEL_ARCH_WATERFALL.md` §7f's
   own `R²≥0.998` toy-scale precedent, justified below):** WIN requires,
   for NCR's own series, `Model_log`'s `R² ≥ 0.90` AND `Model_log`'s `R²`
   exceeds `Model_lin`'s `R²` by `≥0.05`; for the rollout baseline's own
   series, the SAME two conditions with `Model_lin`/`Model_log` swapped.
   **Justification for relaxing `0.998` to `0.90`:** the toy-scale
   precedent measured a 175K-param model in a closed-form isolated loop
   with no I/O or kernel-scheduling variance; this design's rollout
   baseline is a full 98M/392M LM's recurrent-state update measured at LM
   batch sizes, a genuinely noisier wall-clock environment — `0.90` is
   still a demanding bar for 5 points and is itself an assumption flagged
   for attack-round scrutiny, not silently inherited. The `≥10×`-at-
   largest-`h` ratio is still REPORTED (disclosed, not hidden) but is no
   longer the WIN gate.

If accuracy diverges instead of converging on Task 1 (the rollout baseline
fails to match NCR's in-distribution accuracy even at matched depth), that
is ALSO reported — it would mean this repo's own DeltaNet-family
implementation does not actually reach the published expressivity ceiling
at this scale, a genuinely different and also publishable finding (an
implementation/scale gap, not a theory gap), independent of the
dependency-chain/wall-clock readouts above.

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
   GENERAL binding/composition mechanism. **Task 2 (§3.2, revised) reuses
   the SAME 170/43 split but samples only K₂=5 DISTINCT names per episode**
   (filling S₅'s 5 abstract roles) — the split discipline is identical, the
   per-episode pool size is not; a build agent must not conflate Task 2's
   small per-episode `K₂=5` with the underlying 170/43 name-pool split,
   which stays shared across all three tasks.

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
| NCR ortho-write toy-scale, Part A single-relation (K=32/d=33), 4× budget (320K steps) | **≈2.8 GPU-h/cell measured** | memory-trivial (co-resident-safe) | `NCR_ORTHO_WRITE.md` § CEILING AMENDMENT (v2 re-launch), measured completion time |
| NCR ortho-write toy-scale, Part B structured-bank (R=4/d=33 random-orthogonal, distinct-operator path composition), 4× budget (320K steps) | **≈4.24 GPU-h/cell measured** | memory-trivial (co-resident-safe) | `NCR_ORTHO_WRITE.md` § CEILING AMENDMENT — the nearest measured analog to Task 2's own distinct-generator-path construction (§3.2); used to price the bridge cell, §6.2 |

**Tokens/sec, derived (§R1(e), M4's fix — the invariant §6.2 now holds
fixed is TOKENS, not steps).** At `batch=32, seq_len=512` (16,384
tokens/step): 98M → `16,384/0.236 ≈ 69,424 tokens/sec`; 392M →
`16,384/0.836 ≈ 19,598 tokens/sec`. With the §2.2 ≤5% NCR-overhead
assumption applied to the NCR arm only: 98M-NCR → `≈66,118 tokens/sec`;
392M-NCR → `≈18,665 tokens/sec`. **These are today's measured rates AT
TODAY's operating point (batch=32/seq=512) — §6.2 below states every
phase's cost as `GPU-h = token_budget / (tokens_per_sec × 3600)` so that
if §6.4's saturation pilot lands on a DIFFERENT (batch, seq) operating
point, the fix is mechanical: hold each phase's own token budget fixed,
recompute `steps = tokens/(batch×seq_len)`, and substitute a FRESH
Phase-0a-unpacked rate measured at that new operating point — never
extrapolate GPU-h from the batch=32/seq=512 numbers below once the
operating point changes.**

### 6.2 Phased GPU-h ledger (REVISED, §R1(d)/(e) — fixes M3/M4: bridge
cell added as its own gated phase; every phase re-expressed as a fixed
TOKEN budget, GPU-h derived from §6.1's tokens/sec)

**Phase 0 — smoke (both backbones, all 5 arms: NCR-augmented DeltaNet,
flat-vector ablation, plain DeltaNet control, plain Transformer control,
§4.4's sequential-rollout matrix-state baseline), 14M scale.**
Forward/backward/grad-flow-through-bf16-kernel-boundary check (per
`DELTANET_REALDATA_DESIGN.md` §4.3/§4.4's three-tier verification chain,
reused; the sequential-rollout arm additionally needs its own extended-β
wiring verified against the same boundary, §4.4's pin). This smoke also
exercises Task 2's distinct read/write path (`loop_read`, S₅ generators,
§3.2) at 14M toy scale, within the same estimate — no separate line item,
the existing ceiling is generous enough to cover it (below). Cost anchor:
that design's own 14M wall-time band, 0.6–2.2 GPU-h per COMPLETE
400M-token run — smoke needs far less than a complete run. **Estimate: ≤3
GPU-h total** (bounded above by roughly one and a half full 14M runs'
own measured band, generously, to cover the fifth arm and Task 2's path).
Packing (§6.4) is licensed for Phase 0 ONLY — see §6.4's revised scope.

**Phase 0a — rate probe, TWO measurements per candidate config, NOT ONE
(§R1(e), M4's other fix: packed ≠ unpacked, and only one of them prices
Phases 1–3).**
1. **Packed** (4–8 concurrent processes/GPU, §6.4): a throughput-oriented
   smoke used ONLY to size Phase 0/0a's own scheduling — NEVER used to
   price a Phase 1–3 training cell (M4's finding: a contended measurement
   is not the rate an uncontended, saturating-batch training cell will
   realize).
2. **Unpacked, AT the operating point Phases 1–3 will actually use** —
   which requires §6.4's batch/seq saturation pilot to have already
   chosen that operating point. **Sequencing, made explicit (was implicit
   and wrong before this fix): §6.4's saturation pilot runs BEFORE this
   unpacked probe, not after or in parallel.** Today's only measured
   unpacked rate is at `batch=32, seq_len=512` (§6.1) — the numbers below
   use it as the CURRENT operating point; if §6.4 lands on a different
   (batch, seq), this entire Phase 0a-unpacked step, and every GPU-h
   number in Phases 1–3 below, must be re-measured and recomputed via
   `steps = tokens/(batch×seq_len)` at the NEW rate (§6.1's closing note)
   — not extrapolated.

~2,000 steps/cell, ALL candidate configs (NCR-augmented DeltaNet at
98M/392M; Transformer baseline at 98M/392M; §4.4's extended-β
sequential-rollout baseline at 98M/392M — this is the pilot that RETIRES
Option A's AND §4.4's PRICE-UNKNOWN flags, §6.3). DeltaNet-family cells
priced from §6.1's measured per-step rate, scaled to 2,000 steps: 98M =
2,000×0.236s = 472s = **0.131 GPU-h/arm**; 392M = 2,000×0.836s = 1,672s =
**0.464 GPU-h/arm**. Two DeltaNet-family arms (NCR-augmented +
flat-vector ablation) at both scales: `2×(0.131+0.464) = 1.19 GPU-h`.
Transformer-family arm: rate UNKNOWN, priced at 2× the DeltaNet rate as a
placeholder UPPER BOUND ONLY (Transformers are not known to be slower
than DeltaNet at this scale — this is a deliberately conservative
ceiling, not a prediction) for BUDGETING purposes only — **this
placeholder must not be read anywhere else in this document as a measured
or predicted number**: `2×(0.262+0.928) = 2.38 GPU-h`. Extended-β
rollout-baseline arm: same 2× placeholder convention (§4.4's own
PRICE-UNKNOWN, §6.3 item 3a) — `2.38 GPU-h`, ALSO a ceiling, not a
prediction. **Phase 0a total, 2× contingency:
≈(1.19+2.38+2.38)×2 ≈ 11.9 GPU-h** (revised up from the pre-Rev-1 draft's
7.14 GPU-h, which omitted the rollout-baseline arm's own probe — an
instance of the same load-bearing gap the pre-Rev-1 draft separately
flagged and is now folded in directly rather than carried as a footnote).
No rung proceeds past 0a without ITS OWN measured rate
(`NOVEL_ARCH_WATERFALL.md` §9.4's own discipline, reused verbatim) — the
numbers above are a budgeting ceiling, not a committed schedule.

**Phase 0b — BRIDGE CELL (NEW, §R1(d), fixes M3): NS-polar orthogonal
write trained on S₅-GENERATOR writes, gating Task 2's Phase 1 arm
specifically.** Why this is needed, restated: the main ortho-write wave
(`NCR_ORTHO_WRITE.md`, blind-run in flight) validates cyclic K-cycle
writes (Part A) and RANDOM-ORTHOGONAL bank writes (Part B, R=4
independent Hamiltonian K-cycles) — **NEITHER cell ever writes a
non-solvable-group generator** (§A1 M3's finding). A WIN on that verdict
says nothing about whether the model can LEARN to write S₅'s `{t,c,c⁻¹}`
generators (§3.2) as orthogonally-conditioned `4×4` (padded to `5×5`)
operators and compose them at depth — this bridge cell is the SEPARATE,
dedicated gate for exactly that question, run BEFORE any Phase-1 GPU-h is
spent on Task 2.

**Pricing (PROJECTED from Part B's own measured rate, §6.1 — NOT
discounted for the smaller `d`/`R`, per §2.1's own finding that `d`
shrinking barely moves cost because the `40h_ncr²` encoder term
dominates; the ONLY genuine cost lever taken here is a REDUCED step
budget and REDUCED seed count, both flagged as this design's own
choice).** Construction: `R=3` (S₅'s `{t,c,c⁻¹}`, not Part B's R=4),
`d=5` (not `d=33`), `n=2` seeds (not Part B's n=4), **1× budget = 80,000
steps** (not Part B's 4× = 320,000 — a bridge/gate cell needs Gate-0 plus
one modest depth checkpoint, not the full realistic-depth frontier sweep
Part B itself runs). Cost: `80,000/320,000 × 4.24 GPU-h/cell = 1.06
GPU-h/cell × 2 seeds = 2.12 GPU-h (1×); 2× contingency ≈ 4.24 GPU-h`.

**Gate (mirrors `NCR_ORTHO_WRITE.md` §4 Part B's own band structure, at
reduced scope — Gate-0 plus ONE depth checkpoint, not the full L-ladder):**
Gate-0 (median rec@0.9 ≥0.9 at L∈{1,2,3}) AND **WIN**: ortho
S₅-generator-write median rec@0.9 at **L=8** (≈2.7× training depth) ≥0.9
AND free-write (unconstrained) baseline <0.5 at L=8. **PARTIAL**: Gate-0
clears, L=8 recovery ∈(0.5,0.9). **NULL**: Gate-0 clears, no gain over
free-write at L=8. **FAIL**: Gate-0 itself fails (the constraint blocks
trainability on this differently-shaped object). Consequence for Task 2 —
see §9, revised (§R1(c), M2's fix): WIN/PARTIAL license Phase 1's Task-2
arm (PARTIAL re-anchors `L*` downward, mirroring the main verdict's own
PARTIAL branch); NULL/FAIL DROP Task 2's structural axis for Stage 1,
disclosed explicitly, collapsing the overall program verdict's ceiling to
PARTIAL (§7, revised) regardless of how Axis B reads.

**Phase 1 — calibration (MANDATORY per CLAUDE.md's own standing rule: "a
calibration run before a big sweep is mandatory, not optional"). NOW TWO
TASK ARMS, not one (§R1(d), the other half of M3's fix — Task 2 is
co-equal PRIMARY, §1, and needs its own calibration, gated by the bridge
cell above, not folded silently into Task 1's).** 98M only, reduced
20,000-step budget (327.68M tokens/cell — this program's own validated
convention, `FROZEN_BIAS_LM_DESIGN.md` §13.7's citation of the mechanism-
wave's "FULL 20,000-step branch" as already-shown non-degenerate at 14M —
extended here to 98M as a Stage-1 calibration length, not assumed
transferable without re-checking Gate-0 convergence at this scale).

*Task 1 arm (unchanged from the pre-Rev-1 draft):* 2 arms (NCR-augmented
DeltaNet, flat-vector ablation) × n=2 seeds = 4 cells. Rate: `327.68M
tokens / (66,118 tok/s × 3600) = 1.377 GPU-h/cell` (NCR, ≤5% overhead
assumption) and `327.68M/(69,424×3600) = 1.311 GPU-h/cell` (ablation) —
both match the pre-Rev-1 draft's own numbers exactly (§6.1's tokens
reframing changes nothing at today's operating point, only the invariant
being held fixed). NCR arm: 2×1.377=2.756 GPU-h. Ablation arm:
2×1.311=2.622 GPU-h. Task-1 subtotal, 1×: 5.38 GPU-h.

*Task 2 arm (NEW, §R1(d)), GATED on the bridge cell (§9, revised):* SAME
2 arms × n=2 seeds = 4 cells, SAME 327.68M-token budget, SAME rate (the
backbone dominates cost, not the NCR head's own `d`, per §2.1's finding —
Task 2's cells cost the SAME per-cell rate as Task 1's). Task-2 subtotal,
1×: 5.38 GPU-h (identical arithmetic).

**Phase 1 total, 1×: 10.76 GPU-h (was 5.38 GPU-h pre-Rev-1 — the doubling
is Task 2's own calibration, not a pricing error); 2× contingency: 21.52
GPU-h.** Gate-0 precondition (`NCR_ORTHO_WRITE.md` §4's convention,
reused, applied per task arm): in-distribution recovery ≥0.9 AND val-loss
inside the standing `k=2·s_ref` gate (`FROZEN_BIAS_LM_DESIGN.md` §13.4) —
a cell failing Gate-0 blocks that TASK's Phase 2 arm specifically (§8),
not necessarily the other task's.

**Phase 2 — main wave, 98M (gated on Phase 1 passing Gate-0, per task
arm).** Full 67,547-step budget (1.108B tokens/cell). Arms:
NCR-augmented DeltaNet (priced at the NCR-overhead-adjusted rate),
flat-vector ablation, plain-Transformer baseline (rate PENDING Phase 0a,
priced here at Phase 0a's own 2× placeholder ceiling — re-price before
launch). Training is task-suite-shared (a single run trains on the full
curriculum, §5.2, covering Task 1 AND Task 2 episodes together — this
does NOT double per-cell cost, only Phase 1's SEPARATE per-task
calibration cells above are doubled); eval-only passes are near-zero cost
per `FROZEN_BIAS_LM_DESIGN.md` §13.7's own eval-only line-item convention
(~0.02–0.05 GPU-h/pass class) and now cover Task 2's own L-ladder eval at
the same near-zero marginal cost. n=3 seeds × 2 corpora (openr1-mix,
wikitext-mix). NCR arm: 6 cells × `1.108B/(66,118×3600)=4.654 GPU-h/cell`
→ 27.92 GPU-h. Ablation arm: 6 cells × 4.428 GPU-h/cell (measured
plain-DeltaNet rate) → 26.57 GPU-h. Transformer arm: 6 cells ×
(placeholder ceiling) 8.86 GPU-h/cell (2× the DeltaNet rate, PENDING
re-price) → 53.16 GPU-h. **Phase 2 (98M) total, 1×: ≈107.7 GPU-h; 2×
contingency: ≈215.3 GPU-h — the Transformer arm's own placeholder pricing
is more than half this total, underscoring why Phase 0a is not
optional.**

**Phase 3 — main wave, 392M (gated on Phase 2's 98M readout AND the
ortho-write verdict, §9).** Same arm/seed/corpus/task-suite structure,
20,000-step reduced budget (327.68M tokens/cell — the same disclosed
non-token-matched-to-Phase-2 deviation `FROZEN_BIAS_LM_DESIGN.md` §13.7
already made for 392M — matching Track C's full 91,552-step budget at
n=3/n=3 would cost `18×21.38=384.8` GPU-h at 1× alone, an order of
magnitude beyond this design's own affordable scope). NCR arm: 6 cells ×
`327.68M/(18,665×3600)=4.877 GPU-h/cell` → 29.27 GPU-h. Ablation arm: 6
cells × 4.671 GPU-h/cell (measured) → 28.03 GPU-h. Transformer arm: 6
cells × (placeholder ceiling) 9.342 GPU-h/cell → 56.05 GPU-h. **Phase 3
(392M) total, 1×: ≈113.3 GPU-h; 2× contingency: ≈226.7 GPU-h.**

**Grand total (Phases 0–3, 2× contingency, Transformer arm at its
UN-re-priced placeholder ceiling, Phase 0a NOW including the rollout-
baseline probe, Phase 1 NOW including Task 2's own calibration, Phase 0b
bridge cell added): ≈2 + 11.9 + 4.24 + 21.52 + 215.3 + 226.7 ≈ 482 GPU-h**
(vs. the pre-Rev-1 draft's own excludes-the-rollout-arm total of 462
GPU-h — the increase is entirely Phase 0a's now-included rollout probe
(+4.76) and the bridge cell + doubled Phase 1 (+15). This is a Stage-1
DESIGN ceiling, not a committed ask — it is priced deliberately
conservative (2× contingency stacked on 2× placeholder ceilings)
specifically so an attack round has a real number to cut, not a vague
"TBD." At the box's own ≈192 GPU-h/day operative rate, this is ≈2.5 days
of full-box supply if run back-to-back-exclusive; in practice it competes
with the ortho-write wave (~77 GPU-h), fix-at-scale (300 GPU-h cap), and
capability-separation Stage 2 for the shared budget — a coordinator-level
sequencing decision, not resolved by this document.

**This total STILL EXCLUDES §4.4's sequential-rollout matrix-state
baseline's own Phase 1–3 TRAINING cells (its Phase-0a RATE PROBE is now
included above, but its full-scale training cells are not) — a
load-bearing gap, disclosed, not silently absorbed.** That arm needs
training cells at every Phase 1–3 rung, structurally the same shape as
the NCR arm's own line items above, once Phase 0a's dedicated probe
retires its PRICE-UNKNOWN flag. A same-order-of-magnitude placeholder
(mirroring the NCR arm's own 2× the plain-DeltaNet-rate ceiling used for
the Transformer arm) would add roughly another **NCR-arm-sized slice at
each phase** — Phase 1: +2.76 GPU-h (1×, Task-1-only, since Axis B is
Task-1-scoped per §A1 F1 — the rollout baseline is never trained on Task
2); Phase 2: +27.9 GPU-h (1×); Phase 3: +29.3 GPU-h (1×) ≈ **+60 GPU-h at
1×, +120 GPU-h at 2× contingency**, revising the grand total to **≈602
GPU-h at 2× contingency**. This revision is presented as an
order-of-magnitude flag for the attack round, not a firm number — an
attack round should treat §4.4's own full-scale pricing as unresolved and
require the real Phase-0a measurement (now budgeted above) before ANY
Phase-1+ commitment for this arm specifically, exactly as items 1–2 below
already require for the Transformer and NCR-overhead arms.

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
5. **The bridge cell's own rate (§6.2 Phase 0b, §R1(d)).** Priced as a
   PROJECTION from `NCR_ORTHO_WRITE.md` Part B's measured rate (§6.1),
   scaled by step/seed budget only — the `R=3/d=5` construction itself has
   never run, so its ACTUAL per-step cost (and whether it trains at all)
   is unmeasured until the cell itself executes. Retired by the bridge
   cell's own completion, which is also its own gate readout (§6.2).

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

**Calibration / Phase-0/0b cells (14M / toy-scale smoke and the bridge
cell) — packing licensed.** These remain genuinely too small to saturate
a GPU via batch/seq scaling alone without changing the task itself (the
14M backbone + 175K NCR head, or the bridge cell's own toy-scale S₅-
generator write, is still a small model). **Packing plan: run 4–8
concurrent processes per GPU** for Phase 0 (smoke) and Phase 0b (the
bridge cell) — mirrors `H100_SETUP.md`'s and `DELTANET_REALDATA_DESIGN.md`
§3's own standing "pack multiple tiny runs per GPU" pattern, and is
licensed by `NCR_ORTHO_WRITE.md` §6's own "memory-trivial, co-resident-
safe" finding for exactly this param range.

**Phase 0a's rate probe — packing explicitly NOT licensed for the
UNPACKED measurement (§R1(e), M4's fix — this is the exact conflict the
attack round caught).** Phase 0a's whole job is to hand Phases 1–3 a
clean per-arm rate at the operating point they will actually run at; a
contended, packed measurement systematically UNDER-states that rate. Per
§6.2's revised Phase 0a: the PACKED measurement (this section's packing
plan, reused) is for Phase-0/0b's OWN scheduling only; the UNPACKED
measurement — one process per GPU, at the operating point THIS section's
saturation pilot (below) selects — is the one that prices Phase 1–3, and
must run after that pilot, not before it and not packed.

**Main 98M/392M cells (Phases 1–3).** Already far better utilized than
the toy cells (23.5 GB / 39.0 GB of 80 GB at batch=32, seq_len=512 — 29%
/ 49%, not the ~2% the toy cells drew), but not yet saturated. Two
coupled levers, both re-measured before launch (specifically, before
Phase 0a's unpacked probe, per the sequencing above), neither assumed:
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

**Main 98M/392M cells are NOT packed** — at a saturating batch/seq_len
(lever 1/2 above), a single 98M or 392M process is projected to already
occupy 70–90% of one GPU's VRAM, making co-residency infeasible at the
saturating operating point; packing and big-batch saturation are mutually
exclusive levers at this scale and this design picks big-batch for the
cells large enough to support it. **If this saturation pilot picks a
(batch, seq) other than (32, 512), §6.1/§6.2's own re-pricing obligation
fires** — every GPU-h number in Phases 1–3 above is provisional on
`(32, 512)` remaining the operating point.

---

## §7 SUCCESS CRITERIA — draft pre-registerable WIN/PARTIAL/NULL bands
(REVISED, §R1(a) — the table and overall-verdict rule are rewritten
around the F1-fixed two-family conjunction, §1; M2's bridge-cell REDUCED
consequence and M5's replaced Axis-B criterion are folded in)

Reuses the `recovered_frac@0.9` bar and the HOLD(≥0.9)/DEGRADED((0.5,0.9))/
FAIL(≤0.5) band convention (`NOVEL_ARCH_WATERFALL.md` §3.2a) throughout,
extended to per-task, per-axis outcomes. **Gate-0 precondition (both
arms, every cell, per §6.2's Phase-1 convention): in-distribution
recovery ≥0.9 at train-support depth AND val-loss inside the standing
`k=2·s_ref` gate — a cell failing Gate-0 is DEAD and is not scored on any
axis below.**

**Framing correction (§R1(a), replaces the pre-Rev-1 draft's unsatisfiable
framing): the two PRIMARY axes are Axis A = Task 2 (structural failure,
read at Θ(L), no speed claim) and Axis B = Task 1 (query-time complexity,
O(log h) vs. Θ(h), no hardness claim) — DISJOINT query families, per §1's
conjunction and §A1 F1's group-theory result. Task 3 is a SECONDARY/
corroborating axis on a third, orthogonal property (memory bytes), never
sufficient alone for the overall program WIN.**

| Axis | WIN | PARTIAL | NULL |
|---|---|---|---|
| **Axis A = Task 2 (PRIMARY — non-solvable-group structural failure, §3.2, read at Θ(L))** | NCR HOLDs (≥0.9) at `L*=32` (or the re-derived depth, §9's REVISED bridge-cell branch) on BOTH held-out-entity and held-out-depth strata, AND the param-matched Transformer FAILs (≤0.5) at the same depth (the TC⁰/NC¹ structural prediction, C1–C3) — the flagship structural-failure result. **Gated precondition (§9, revised, M2/M3): only reachable if the bridge cell (§6.2 Phase 0b) reads WIN or PARTIAL** — a NULL/FAIL bridge cell drops this axis to NULL by construction, disclosed, not silently absorbed (§9) | NCR HOLDs/DEGRADEDs strictly better-banded than the Transformer, but not the full HOLD-vs-FAIL gap; OR the bridge cell read PARTIAL and `L*` was re-anchored downward per §9, with the shallower depth otherwise meeting the WIN bar | NCR bands at or below the Transformer's band at `L*` (the structural prediction did not manifest empirically, a genuine negative, §8 item 5) **OR the bridge cell reads NULL/FAIL (§9) — Task 2 is DROPPED for Stage 1, and this axis is NULL by construction regardless of any LM-scale data** |
| **Axis B = Task 1 (PRIMARY — query-time complexity vs. §4.4's pinned extended-β-DeltaNet rollout baseline, O(log h) vs. Θ(h), no hardness claim)** | (i) The dependency-chain-length fact holds by construction (`⌈log₂h⌉` for NCR vs. `h` for the rollout, §4.4's build-time instrumentation assertion — a PASS/FAIL gate, always checked first) AND (ii) the §4.4 fitted-growth criterion clears: `Model_log` `R²≥0.90` and beats `Model_lin` by `≥0.05` `R²` on NCR's own wall-clock series, AND the mirror-image holds for the rollout baseline's series AND (iii) in-distribution accuracy is within the standing ±15% band between the two arms at matched depth (an accuracy SANITY check, not the primary readout — §4.4) | The dependency-chain fact holds (i) but the wall-clock fit (ii) is inconclusive (e.g. one arm's fit clears its own `R²` bar but the margin over the wrong-family fit is <0.05) — reported as a real but statistically softer separation, not hidden | (i) itself fails (a build defect — NCR's read is not actually `O(log h)` as implemented) OR NCR's own wall-clock is not measurably sub-linear in `h` at all — flags a broken implementation, triggers diagnose-first, per `NOVEL_ARCH_WATERFALL.md`'s own convention for this failure shape |
| **Task 3 (secondary — "constant-memory minds," §3.3)** | NCR holds acc≥0.9 at the largest tested horizon (8×T_bind) AND the KV-cache-capped Transformer (re-derived `M` grid, §4.3, revised) FAILs at that horizon — reproduces the M* "capping never rescues" finding (`STATE.md` §1.41) in a real-text setting | NCR holds where the capped Transformer fails, but the UNCAPPED Transformer also holds — NCR matches, not beats, an unconstrained baseline at fixed memory (itself a genuine, disclosed data point) | NCR itself fails past some horizon shorter than 8×T_bind |
| **Val-loss gate (all tasks, mandatory)** | Every arm's val loss stays inside `arm_off`'s own `mean + 2·s_ref` band (`FROZEN_BIAS_LM_DESIGN.md` §7.2 convention) | n/a — pass/fail gate, not graded | A cell whose NCR head measurably HURTS ordinary language modeling is reported plainly, not excused, even if a primary axis above WINs |

**Overall program verdict (cross-task rule, exhaustive, re-anchored to
the F1-fixed two-family conjunction, §1).** **WIN** requires BOTH primary
axes at WIN — Axis A/Task 2 (structural failure) AND Axis B/Task 1
(query-time complexity) — AND the val-loss gate passing everywhere; Task
3 is reported alongside as a corroborating/secondary claim, never
substitutable for either primary axis. **PARTIAL** = exactly one primary
axis WINs (structural-failure without the complexity separation, or vice
versa) — still publishable, reported as "one leg of the two-family claim
landed, on the family it is earnable on," not spun as the full flagship
result; **this is also the automatic ceiling if the bridge cell (§9)
reads NULL/FAIL, since Axis A is then NULL by construction and only Axis
B remains contestable.** **NULL** = neither primary axis clears WIN or
PARTIAL, or Gate-0 fails in ≥50% of cells (a trainability failure at LM
scale, distinct from and reported separately from a capability failure).
**Removed from this revision (§R1(a)): the pre-Rev-1 draft's own framing
implicitly allowed a version of "WIN" where Task 2 supplied BOTH the
structural failure AND the O(log h) speed claim on the same family — §A1
F1 proved this unsatisfiable; no band above can be cleared that way, by
construction (Task 2's own read cost is Θ(L), stated in its own WIN cell,
never O(log L)).**

---

## §8 RISKS & KILL CRITERIA — what a week-1 result kills before week 4
(REVISED, §R1(h) — fixes m1/m2/m3; items 1/5/6 re-scoped to the F1-fixed
two-family conjunction; item 7 added for the bridge cell, §M2/M3)

1. **Ortho-write verdict lands NULL or FAIL (§9, revised) before Phase 1
   launches.** KILLS the K=32 configuration outright for Task 1 (and Task
   3 if it inherits Task 1's abelian construction); Stage-1 re-scopes to
   K≤15 (the pre-ortho-write "SCALES" ceiling, `NOVEL_ARCH_WATERFALL.md`
   §11.2) before ANY GPU-h beyond Phase 0 is spent on Axis B. **Does NOT
   touch Task 2/Axis A** (§R1(b)/(c), M1/M2's fix decoupled Task 2 from
   this K-axis entirely — Task 2's own viability is governed by the
   SEPARATE bridge-cell gate, item 7 below). Does not kill the program —
   changes Axis B's headline numbers (§9).
2. **Phase 0 smoke shows the NCR head's gradient does not flow through
   the DeltaNet backbone's bf16 kernel boundary within an EXACT tolerance
   (§R1(h), m1/m2 fix — was "cleanly," now pinned).** Concrete smoke item
   (m1's fix): the NCR head casts backbone hidden states to fp32 at its
   own input boundary (`.float()`, a standard differentiable upcast),
   runs its full write/read pipeline in fp32, casts its output back to
   bf16 before injection into the residual stream. Pass criterion (m2's
   fix, reusing `DELTANET_REALDATA_DESIGN.md` §4.4's own Tier-1
   convention verbatim): a gradient cross-check (this cast pipeline vs. a
   small-scale reference run with the ENTIRE model in fp32) at a
   bf16-appropriate relative tolerance **`<1×10⁻²`**, both forward outputs
   and parameter/hidden-state gradients after `.backward()`. Run ONCE at
   Phase 0 (14M scale), before any further spend, logged PASS/FAIL
   explicitly — not a vibes call. **On FAIL: KILLS Option B, forces a
   re-evaluation of Option A** (now bearing its own full PRICE-UNKNOWN
   backbone cost) before any further spend.
3. **Phase 0a measures NCR-head wall-clock overhead outside a TWO-TIER,
   GAP-CLOSED band (§R1(h), m3's fix — was an unguarded 5–20% zone with
   no defined consequence).** **>5% (the §2.2 pricing assumption): MANDATORY
   RE-PRICE** — every Phase 2/3 number in §6.2 assumes ≤5%; ANY measured
   overhead above that invalidates the ledger's point estimates and
   forces a re-price pass through §6.2's token-based formulas with the
   measured rate substituted, before Phase 1 launches (a book-keeping
   trigger, not a stop). **>8% (tightened from the pre-Rev-1 draft's 20%,
   a 2.5× reduction, justified): KILLS the pricing AND the Option-B cost
   argument.** Justification for 8%, not 5% (identical) and not 20%
   (the old, too-loose bound): 8% is still ≈6.7× the ONE directly measured
   analog in this repo (`FROZEN_BIAS_LM_DESIGN.md`'s ≤1.2%-measured
   per-position insertion overhead, §6.1) — a generous but non-vacuous
   margin — while 20% left a 4×-of-the-analog gap with NO defined
   consequence in between, exactly the m3 finding. Every point in
   `(5%, 8%]` now triggers RE-PRICE (defined); every point `>8%` triggers
   KILL (defined) — no unguarded zone remains.
4. **Phase 1 calibration plateaus below Gate-0's 0.9 in-distribution bar**
   at 98M, on EITHER task arm (§6.2, revised — Task 1 and Task 2 now each
   have their own Phase-1 calibration cells) — the CLAUDE.md-mandated
   calibration-run purpose: "catches convergence ceilings... before you
   commit a sweep's compute to it." KILLS Phase 2 for THAT task arm
   specifically; does not kill the program or the other task's arm —
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
   negative result for Axis A. Reported plainly (`CLAUDE.md`: "negative
   results are data. Don't spin.") — does not kill the program (Axis B's
   complexity claim, item 6 below, is on a disjoint query family and
   independent of this one, §A1 F1), but caps the paper's structural-
   failure claim at "theoretically predicted, not empirically observed at
   this scale," a materially weaker headline than §1's Axis A hypothesis,
   and would redirect Stage-2 effort toward understanding WHY (larger K₂,
   larger group, different embedding) before any further scale-up.
6. **Axis B (§4.4) shows no wall-clock/dependency-chain separation on
   Task 1** — the dependency-chain-length build assertion itself fails
   (a genuine implementation defect: NCR's read is not actually
   `O(log h)` as built), or the fitted-growth criterion (§4.4/§7) does
   not clear on either arm's own series (e.g. both are dominated by a
   fixed per-call overhead at the tested depths, or the rollout
   baseline's own implementation is unexpectedly parallel-friendly at
   this scale). Per `NOVEL_ARCH_WATERFALL.md` §3.2's own convention for
   this exact failure shape, this is flagged as an IMPLEMENTATION defect
   and triggers diagnose-first, not an immediate capability conclusion —
   but if diagnosis confirms the measurement is sound, it kills Axis B
   and, combined with item 5, would collapse the overall verdict to NULL
   (§7) regardless of how Task 3 reads.
7. **The bridge cell (§6.2 Phase 0b, §9, revised) reads NULL or FAIL
   (NEW item, §R1(h) — the M2/M3 fix's own risk consequence, made
   explicit here rather than left implicit in §6/§9 alone).** DROPS Task
   2/Axis A for Stage 1 — the structural claim is not earned at any LM
   scale this wave, disclosed, not silently absorbed (§7's overall
   verdict is automatically capped at PARTIAL, contingent only on Axis
   B). Does not kill the program (Axis B's own claim on Task 1 is
   entirely independent of the bridge cell's outcome) but is the single
   cheapest possible way (§6.2: ≈2.1 GPU-h, 1×) to learn this BEFORE any
   Phase-1 Task-2 GPU-h — the entire reason this gate exists ahead of the
   more expensive LM-embedded calibration.

---

## §9 DEPENDENCIES — TWO INDEPENDENT gates, not one (REVISED, §R1(c)/(d) —
fixes M2/M3: Task 2/Axis A is DECOUPLED from the main K-axis verdict and
gated separately by its own bridge cell)

**§A1 M1's fix (§3.2) decoupled Task 2's construction (K₂=5, S₅'s own
generators) from Task 1/3's K-axis (`K∈{15,32}`) entirely — so the single
branch structure below no longer governs both tasks, and pretending it
does was exactly M2's/M3's finding (a WIN on cyclic/random-orthogonal
writes at K=32 does not license a never-tested S₅-generator write, and a
K=15 fallback specifying "15-cycle products" silently drops the
non-solvable structure Task 2 exists for).** Two INDEPENDENT gates now
govern this document's two primary axes:

### 9.1 GATE 1 — main ortho-write verdict (governs Axis B/Task 1, and Task
3 IF it inherits Task 1's abelian construction)

Downstream of `NCR_ORTHO_WRITE.md`'s pending verdict on the Newton–Schulz
orthogonal write (**closest prior art: MuonSSM, arXiv:2606.30461 — see
§2.2/§3.4 C12 for the full differentiation; that paper's own
ICML-2026-Oral result is independent evidence that Newton–Schulz-
orthogonalizing a fast-weight write is a live, reviewer-legible
mechanism, not exotic machinery**). **No Phase 1+ GPU-h is authorized for
Task 1/Axis B until that verdict lands and this branch is resolved — but
this gate does NOT block Task 2/Axis A, which is governed by GATE 2,
below, independently.**

- **If ORTHO-WRITE WIN** (`NCR_ORTHO_WRITE.md` §4 Part A: ortho-write
  median rec@0.9 at h*=40 ≥0.9, free-write baseline <0.5 at h=40): Stage-1
  build proceeds with **K=32, d_ncr=33** as Task 1's PRIMARY NCR
  configuration (h*=40, Axis B's own carrier task), using the
  Newton–Schulz orthogonal write throughout — exactly the configuration
  priced in §2/§6 above.
- **If ORTHO-WRITE PARTIAL** (cracks at h∈{20,29} but not 40): Stage-1
  re-registers `h*` DOWNWARD to match the cleared depth (e.g. h*=29) — the
  same K=32/d_ncr=33 config stays, only the claimed depth shrinks; §6's
  GPU-h pricing is essentially unchanged; §7's Axis-B WIN band is
  re-anchored to the new `h*` before Phase 1 launches, not after seeing
  Phase-1 data.
- **If ORTHO-WRITE NULL or FAIL** (no far-depth gain, or Gate-0 breaks):
  Stage-1 does NOT use K=32 for Task 1. Falls back to **K=15, d≈16** (the
  pre-ortho-write free-write "SCALES" regime — 4/4 converged + far-depth
  HOLD, `NOVEL_ARCH_WATERFALL.md` §11.2). This lowers Task 1's headline
  `h*` to whatever K=15's own free-write frontier supports (re-derive from
  that config's own archived z-dumps before pinning a number — not
  assumed here). If the failure mode is specifically FAIL (trainability
  breaks under the orthogonality constraint), this ALSO flags that Phase
  0's smoke (§6.2) must include an explicit Gate-0 check at the
  LM-embedded K=15 configuration BEFORE any Phase-1 spend — the
  write-conditioning problem the ortho-write wave diagnosed at the
  isolated Task-E harness is not guaranteed to manifest identically once
  bind clauses are embedded in a real-LM training signal (a new confound
  this document does not resolve and must not assume away). **This
  branch does NOT change Task 2's own construction** (§R1(c)'s direct fix
  of M2: the pre-Rev-1 draft's "R×15-cycle products instead of R×32"
  silently swapped Task 2 onto a CYCLIC — solvable — group, which cannot
  carry Axis A's structural claim; that swap is retired, not repaired,
  because it is no longer needed — Task 2 never depended on this K-axis
  to begin with).

### 9.2 GATE 2 — the bridge cell verdict (NEW, §R1(d) — governs Axis A/
Task 2 exclusively; independent of GATE 1)

`NCR_ORTHO_WRITE.md`'s existing blind run (GATE 1, above) trains cyclic
K-cycle writes (Part A) and RANDOM-ORTHOGONAL bank writes (Part B, R=4
independent K-cycles) — **never an S₅-generator write** (§A1 M3's
finding). The bridge cell (§6.2 Phase 0b) is the dedicated, independent
gate for exactly that untested object, and its own verdict — NOT GATE
1's — determines whether Task 2/Axis A proceeds, at what depth, or at
all:

- **If BRIDGE CELL WIN** (median rec@0.9 at L=8 ≥0.9, free-write baseline
  <0.5 at L=8, §6.2): Task 2's Phase 1 arm (§6.2) proceeds with
  **K₂=5, d_ncr,2=5, R=3** (S₅'s `{t,c,c⁻¹}`) at the full eval ladder
  `L∈{5,…,40}`, `L*=32` — exactly the configuration priced in §3.2/§6
  above. Axis A (§7) is fully contestable at WIN.
- **If BRIDGE CELL PARTIAL** (Gate-0 clears, L=8 recovery ∈(0.5,0.9)):
  Task 2 proceeds but `L*` is re-anchored DOWNWARD to whatever depth the
  bridge cell's own (limited) ladder supports — a build agent must
  extend the bridge cell's own eval readout to at least one lower
  checkpoint (e.g. L=5) before pinning a number, not assumed here; §7's
  Axis-A WIN band is re-anchored accordingly, and Axis A is EXPLICITLY
  marked REDUCED in any results write-up (§R1(c)'s direct fix of M2's
  "preserve both axes or mark the structural axis REDUCED" instruction)
  — still contestable, at a weaker headline depth.
- **If BRIDGE CELL NULL or FAIL** (no depth gain over free-write, or
  Gate-0 itself fails on the S₅-generator object): Task 2/Axis A is
  **DROPPED for Stage 1** — disclosed explicitly, not silently absorbed.
  No Phase 1+ GPU-h is spent on Task 2's own arm (§6.2's Phase 1 Task-2
  subtotal, 5.38 GPU-h at 1×, is NOT spent); §7's overall program verdict
  is capped at PARTIAL regardless of how GATE 1/Axis B reads (§7's
  overall-verdict rule, revised). If NULL (trains, no depth gain): flags
  a genuine "does the write-conditioning fix generalize beyond cyclic/
  random-orthogonal objects" open question for a future waterfall pass,
  not assumed resolved either way. If FAIL (does not train at all):
  flags that S₅'s specific generator structure (a transposition +
  5-cycle, order-5 rotation) may need its own parametrization variant
  (§2 of `NCR_ORTHO_WRITE.md`'s own Cayley/matrix-exponential fallbacks,
  named there for exactly this kind of failure) before any retry — not
  more budget on the same parametrization.

**Both gates are independent and may resolve differently** (e.g. GATE 1
WIN + GATE 2 FAIL is a fully coherent outcome under this design: Task 1/
Axis B proceeds at K=32 while Task 2/Axis A is dropped) — this is the
direct structural consequence of M1's decoupling fix, and a build agent
must not assume the two verdicts move together.

---

**Provenance.** This document was a NEW Stage-1 design (opened
2026-07-16), attacked once (§A1, BUILD-BLOCKED, §A1-ADJUDICATION), and is
now REV-1 (§R1, this revision) — §1–§9 revised in place per the
adjudication's binding items (a)–(h); §A1/§A1-ADJUDICATION left untouched
as historical record, below. `research/ncr_separation_grounding.md` and
`research/ortho_write_grounding.md` landed 2026-07-16 (coordinator-spot-
checked) and remain incorporated throughout (§1, §2.2, §2.3, §3.1–§3.4,
§4.4, §7, §9) — every citation is VERIFIED or explicitly flagged, no
`[TO-VERIFY]` tags remain. **One item from the pre-Rev-1 draft is now
RESOLVED, not merely flagged:** the S₅-order-120-vs-K reconciliation
(item (b) of the pre-Rev-1 provenance note) is fixed by §3.2's M1 rewrite
— Task 2 now uses S₅'s own 5-point defining action (`K₂=5`), decoupled
from Task 1/3's K-axis, so the "order 120 > K=32" conflict this note
flagged no longer arises. **One item is still open, unchanged by this
revision:** `research/ncr_separation_grounding.md` item 8 (Nichani/Lee/
Bietti rank-1-argmax capacity) is verified only via cross-reference to a
prior in-repo fetch, not a fresh tool fetch this round — flagged for one
human spot-check of the PDF directly before any paper-facing use. This
document is gated for build authorization on: (1) GATE 1 (§9.1, the main
ortho-write verdict) AND (2) GATE 2 (§9.2, the bridge-cell verdict, NEW
this revision) — independent, both must resolve before their respective
task's Phase 1+ GPU-h is spent — AND (3) ATTACK ROUND 2 on this revision
itself, which should target (per the pattern of round 1) whichever pinned
assumption looks most load-bearing: candidates include the §4.4 Axis-B
statistical criterion's `R²` thresholds, the bridge cell's un-measured
cost projection, and whether M1's S₅-on-5-points construction is itself
well-posed for the FULL two-axis conjunction once built.

---

## §A1 ATTACK ROUND 1 (2026-07-16, pre-build, independent)

Adversarial attack against DRAFT-STAGE-1. Every finding recomputed or
quoted from a source file (line-cited). Verdict at the end.

### F1 [FATAL] — §1's hypothesis is mathematically unsatisfiable: it demands non-solvable-group STRUCTURAL hardness and O(log h)-via-repeated-squaring on the SAME query family, and those two properties are provably mutually exclusive.

**Defective quote (§1, lines 66–68):** "will answer depth-h
**non-solvable-group** relational-composition queries embedded in text
**EXACTLY at query-time sequential cost O(log h) via repeated squaring**".

**Worked group theory (the adjudication the coordinator asked for).**
"Repeated squaring" / `binexp_read` computes `Z^h` for a SINGLE written
operator `Z` (design §2.1 line 111: `o = binexp_read(Z, q, h)`;
`NOVEL_ARCH_WATERFALL.md` line 434: "read = scale-managed bin-exp `Z^h q`
| O(log h)"). The set `{Z^h : h∈ℤ} = ⟨Z⟩` is the **cyclic** subgroup
generated by one element — **abelian, hence solvable**. Barrington's
NC¹-completeness (and therefore the entire TC⁰⊊NC¹ structural-failure
citation C1–C3) applies ONLY to the word problem of a **non-solvable**
group, which requires an **arbitrary interleaved word of DISTINCT
generators** `g_{o_L}∘…∘g_{o_1}`. A product of distinct generators is
**not a power of any single matrix**, so repeated squaring does not compute
it: the read costs Θ(L) matvecs, exactly the sequential-rollout baseline's
cost. The two properties cannot coexist on one query family:
- **Structural hardness ⇒ distinct-generator word ⇒ NO squaring shortcut ⇒ O(L), not O(log L).**
- **O(log h) repeated squaring ⇒ single-operator power ⇒ cyclic/solvable ⇒ NOT NC¹-hard ⇒ the Transformer is NOT structurally barred.**

**This is confirmed by the design's OWN binding sources, not just first
principles:**
1. `NOVEL_ARCH_WATERFALL.md` §2 finding **M4 (lines 110–113), marked
   "binding, not revisitable without a new waterfall pass":** "The relation
   chain Z_rn···Z_r1 has **NO squaring shortcut (heterogeneous products
   sequential), loses log-depth**, sits in Neural-LP/DRUM/FWM territory".
2. `NOVEL_ARCH_WATERFALL.md` line 1834–1839 (Axis B-CHAIN): O(log h) holds
   "since **B is FIXED at 2, never query-controlled** … a heterogeneous
   chain whose LENGTH grows with the query **loses log-depth** … B=2 is
   disclosed as a scope limit, NOT claimed to generalize to variable B".
   Task 2's path length L is exactly a query-controlled, growing chain
   length (eval `L∈{5,8,12,16,20,24,32,40}`, §3.2 line 439).
3. `NCR_ORTHO_WRITE.md` §4b line 239 — the very construction §3.2 claims to
   transplant — states outright: "**`binexp` does not apply (the composite
   is not a power of one matrix)**"; its read is `loop_read`, **O(L)**.
4. The design's **own anti-cheat guard defeats its own claim**: §3.2 guard
   (2) (line 434, "no consecutive repeats… `o_{t+1}≠o_t`") is precisely what
   forbids any single-operator-power sub-run — i.e. it *guarantees by
   construction* that no repeated-squaring shortcut exists on Task 2.

The design's own §3.1/§3.2 split already lives on the correct side of this:
Task 1 (single K-cycle, `Z^h`) is where O(log h) is real but the group is
cyclic→solvable (so §3.1 line 379–397 correctly demotes its baseline
failure to EMPIRICAL); Task 2 (distinct generators) is where the structural
citation bites but there is no O(log h) read. **The bug is that §1's
one-sentence hypothesis — and §7's overall verdict, which requires Task 2
(structural) AND Axis B (O(log h)) at WIN simultaneously — re-bundle the two
onto the single non-solvable family, which is impossible.** §4.4 line 639
("for NCR it is Θ(log h) (binary exponentiation…)") measured "at matched
depth h/L" inherits the same error: binary exponentiation is Θ(log h) only
on Task 1's family; on Task 2 NCR is Θ(L), tying the rollout baseline.

**Minimal fix (build-blocking).** Rewrite §1 as an EXPLICIT CONJUNCTION of
two results on two DIFFERENT query families, with disclosure that the
flagship is a conjunction, not one bet: (R1) Task 1 / single-operator /
cyclic — EXACT `Z^h` read at O(log h) via repeated squaring, beating O(h)
rollout, baseline-failure EMPIRICAL; (R2) Task 2 / distinct-generator /
non-solvable — EXACT word product where the TC⁰ Transformer structurally
cannot (C1–C3), read cost **O(L)** (or, if an O(log L) *circuit-DEPTH*
advantage is wanted, BUILD and PRICE a balanced parallel-prefix product-tree
reader — an NC¹ reader, O(L) work / O(log L) depth — and STOP calling it
"repeated squaring"; `binexp_read` cannot do it). Scope every "O(log h) via
repeated squaring" / Axis-B ≥10× claim to Task 1 ONLY, and strike it from
Task 2 and from §7's Task-2 WIN row. Until §1 and §7 are rewritten, the
flagship claim as stated cannot be earned by any experiment.

### M1 [MAJOR] — Task 2 (the flagship task) is not well-posed at the calibrated K, and the infrastructure it claims to "reuse directly" does not produce its answer type.

**Defective quote (§3.2, lines 411–416):** "draw R DISTINCT generator
matrices from a NON-SOLVABLE group (S₅, order 120, or A₅/A₆) **acting on the
K-entity pool** … the target is the exact group-word product … **applied to
the query entity**, computed by exact integer/permutation composition".
And (§3.2 line 407–409): the `CAPABILITY_SEPARATION_DESIGN.md` group
infrastructure is "**already built and calibrated in this repo**".

**Evidence.** `CAPABILITY_SEPARATION_DESIGN.md` lines 224–230 build
S₅/A₅/A₆ as **d_min-dimensional ROTATION matrices** in their minimal
faithful *representation* (S₅ → 4-dim standard rep on the zero-sum
hyperplane of ℝ⁵; A₅ → 3-dim icosahedral; A₆ → 5-dim), acting on a
**continuous ℝ^{d_min} vector space** — NOT as permutations of K discrete
entity names. Its task is "rank tracks d_min," and its "query" is a
continuous vector. Task 2 instead needs a **permutation action on K discrete
entities** producing a **discrete answer entity** (the `grammar_rd.py`
bind-clause → next-token format). These do not compose:
- S₅ acts naturally on **5** points. There is **no transitive action on 32
  points** (32 ∤ 120); a faithful action on K=32 requires a hand-built
  non-transitive orbit decomposition (e.g. 20+12) that
  `CAPABILITY_SEPARATION` never built and never calibrated.
- If instead the reused ROTATION reps are used verbatim, the "answer" is a
  continuous vector in ℝ^{d_min}, which **cannot be rendered as a
  next-token prediction over the 213 discrete GPT-2-tokenizable names** —
  breaking the `grammar_rd.py`/bind-clause reuse the whole task rests on.
- "Already built and calibrated" is an overclaim for exactly the
  non-solvable cells: `CAPABILITY_SEPARATION_DESIGN.md` lines 13–15/79–95
  record **A5/A6 HARD-STOPPED** and **S4/S5 needing a 2–2.5× escalation
  retest**, with a known "encoder degeneracy specific to single-generator"
  ceiling for order-5 groups.

The design honestly flags the tension (§9 line 1039–1041; provenance
(b) line 1080–1081) but leaves it "not yet resolved" while §1 states the
flagship hypothesis as if Task 2 at K=32 is constructed and ready. A
flagship task whose construction is undefined cannot anchor a build.

**Minimal fix.** Before build, PIN Task 2's group action explicitly: either
(a) use the natural permutation action ⇒ K = |acted-on set| (5 for S₅, and
re-open the entire K-axis/ortho-write gating, which was never run at K=5),
or (b) construct + verify a faithful permutation representation S₅→S_K on
the chosen K (with the orbit arithmetic written out and disjointness
asserted), or (c) explicitly define a discrete readout over the rotation
rep's orbit and prove the entity-pool arithmetic closes. State which, with
the entity-count arithmetic, in §3.2 — do not defer to "build time."

### M2 [MAJOR] — the §9 K≤15 fallback silently guts the flagship task: it specifies CYCLIC "15-cycle products," which cannot carry Task 2's non-solvable structural claim.

**Defective quote (§9, line 1054):** "This changes Task 2's structured-bank
design (**R×15-cycle products** instead of R×32)".

**Evidence.** A "15-cycle" is a single cyclic permutation; a product of
K-cycles from the Task-1/`NCR_ORTHO_WRITE.md`-§4b random-orthogonal-cycle
construction is drawn from cyclic generators, not from a non-solvable group.
Under the NULL/FAIL branch, Task 2 would therefore lose the non-solvable
structure that is its ENTIRE reason to exist (the C1–C3 TC⁰⊊NC¹ citation
requires non-solvable), collapsing Task 2 into a K=15 copy of Task 1 —
while §7 still lists Task 2 as the PRIMARY structural axis. The fallback
also does not carry the M1 well-posedness fix down to K=15 (A₅, order 60,
*does* admit a transitive action on 15 points — a genuinely better fit than
S₅-on-32 — but the design does not use or note this; it reverts to cyclic
15-cycles).

**Minimal fix.** The K≤15 fallback must keep the non-solvable generators
(e.g. A₅ acting on 15 via cosets of V₄, order 60) for Task 2, or §7/§9 must
disclose that under NULL/FAIL the structural axis is DROPPED (not merely
"shrunk"), demoting the program to Task-1-class empirical claims only.

### M3 [MAJOR] — the §9 ortho-write gate is calibrated on the wrong object: the blind run tests cyclic K-cycles / random-orthogonal banks read at O(L), never S₅/A₅ generators; a WIN verdict does not license Task 2.

**Defective quote (§9, lines 1032–1041):** "**If ORTHO-WRITE WIN** … Stage-1
build proceeds with **K=32, d_ncr=33** … for BOTH Task 1 … and Task 2".

**Evidence.** `NCR_ORTHO_WRITE.md` §4b lines 210–216 pins Part B to "R
distinct **random-orthogonal** operators … **each `R_r` an independent
Hamiltonian K-cycle**, R=4," read via `loop_read` (O(L), line 236–240).
Part A (line 121–145) is a **single** K-cycle `Z^h`. Neither cell writes a
non-solvable-group generator. So even an unambiguous ortho-write WIN
validates orthogonal-write *conditioning* for cyclic/random-orthogonal
writes only — it says nothing about whether a LEARNED S₅/A₅ generator write
trains, orthogonalizes, or composes at depth. The §9 branch structure treats
the verdict as transitive to Task 2's object; it is not. (This compounds F1:
`NCR_ORTHO_WRITE.md` §4b's own O(L) `loop_read` is independent confirmation
that the distinct-operator chain has no O(log L) read.)

**Minimal fix.** Add a Phase-0 gate that re-verifies write-conditioning +
Gate-0 for the ACTUAL Task-2 generator object (the M1-pinned S₅/A₅
permutation writes) before any Task-2 spend; do not let the cyclic-cell
ortho-write WIN stand in for it.

### M4 [MAJOR] — the GPU-h ledger and the saturation plan price two different machines; the ledger's operating point contradicts §6.4.

**Defective quotes.** §6.1 line 724 anchors all pricing at "**0.236 s/step**
… batch=32, seq_len=512" (confirmed the batch=32 realized rate,
`FROZEN_BIAS_LM_DESIGN.md` §13.7 table, verified). §6.2 prices every phase by
**fixed step count** (e.g. Phase 2 "67,547×0.248s"). But §6.4 lines 882–902
plans to run at "**batch ∈ {32,48,64,96}**" and "**seq_len∈{1024,2048}**".

**Evidence / why it's wrong.** 67,547 steps is the batch=32/seq=512 *token
budget*. Running 67,547 steps at batch=96/seq=2048 processes ~6× the tokens
at ~6× the s/step ⇒ ~6× the GPU-h the ledger states (or, if you hold GPU-h,
you are running 6× fewer tokens than FROZEN_BIAS's converged recipe). The
ledger fixes STEPS while the saturation plan changes the per-step token
count — the two cannot both hold unless the invariant is stated as
TOKENS, and the step count re-derived at the chosen (batch, seq). Separately,
**§6.4 plans to PACK 4–8 processes/GPU for "Phase 0/0a"** — but Phase 0a
(§6.2) is the *rate probe* whose entire job is to hand Phases 1–3 a clean
per-arm rate. A contended packed measurement is not the uncontended
big-batch rate Phases 1–3 will run at, so the probe does not retire the
PRICE-UNKNOWNs it is supposed to.

**Minimal fix.** Re-express the ledger in tokens (fixed token budget →
steps = tokens/(batch·seq) at the pinned operating point) and re-price at
the saturating (batch, seq); measure the Phase-0a rate UNPACKED at the
Phase-1–3 operating point (pack only throughput-oriented smoke, never the
rate probe).

### M5 [MAJOR] — Axis B is elevated to a PRIMARY (flagship-gating) axis while its baseline is architecturally unchosen, entirely unpriced, and its ≥10× bar is transplanted from a depth (h≈10³–10⁶) the design never reaches on the accuracy-claimable ladder.

**Defective quotes.** §7 makes Axis B a PRIMARY axis and §7's overall WIN
"requires BOTH primary axes at WIN". Yet §4.4 line 620–624 leaves the
baseline "a build-time choice, not resolved here" between extended-β
DeltaNet and RWKV-7, and §6.2 line 821–838 concedes it is "**entirely
UNPRICED**… a same-order-of-magnitude placeholder, not a price." The ≥10×
bar (§7 Axis-B row) is "transplanted" from `NOVEL_ARCH_WATERFALL.md` §3.2,
where it was measured at **h=1021** (line 1478) and h=2²⁰+5 (line 1569).

**Evidence.** A flagship-gating primary axis cannot rest on an unbuilt,
unspecified, unpriced baseline. And the ≥10× wall-clock separation was
earned at h≈10³–10⁶; the design's Task-1 accuracy frontier is capped by
ortho-write at h*≈40–61 (`NCR_ORTHO_WRITE.md` §3), and Task-2 has no
O(log·) read at all (F1). Read-cost timing is accuracy-independent so ≥10×
*can* be demonstrated at huge h on Task 1's single operator — but then the
two "primary" axes are measured on different tasks at wildly different
depths, which is exactly the conjunction F1 says must be disclosed, not a
single coherent flagship result. At the depths where NCR's Task-1 answer is
still exact (≤61), the design's own §2.2/§8-item-6 warning (fixed
kernel-launch overhead dominates small ops) makes ≥10× unproven.

**Minimal fix.** Either demote Axis B to a disclosed secondary/efficiency
result (matching `research/ncr_separation_grounding.md` Part 3's own "(b) is
the load-bearing claim … efficiency/exactness separation" framing), or
resolve the §4.4 architecture + Phase-0a price BEFORE it can gate the
flagship verdict, and re-derive the ≥10× bar at the actual claimable depth
on Task 1's family only.

### M6 [MAJOR] — §4.3's pre-registered "moderate long-context" comparison point is wrong by ~16×.

**Defective quote (§4.3, lines 592–595):** "at R=32 … putting the
contender's own total inference memory into a regime where **`M=32` already
clears `cap_length ≈ 966` tokens**, a genuinely 'moderate long-context'
comparison point".

**Recomputed (design's own formula, line 576).** `cap_length =
M·state_bytes/(2·n_layers·d_model·bytes)`; at R=32, state_bytes=139,392,
denom=73,728: `cap_length(M=32) = 32·139,392/73,728 = **60.5 tokens**`, not
966. The value 966 corresponds to **M=512** (`512·139,392/73,728 = 968`).
So the pre-registered claim that the R=32 variant reaches "moderate
long-context" at the grid's *low* end (M=32) is false by ~16×; it reaches
~60 tokens there and needs the grid's *top* (M=512) for the ~966-token
regime.

**Minimal fix.** Correct the sentence to "M=512 clears ≈966 tokens (M=32
clears ≈60)," and re-anchor whatever §7 Task-3 band depends on the "moderate
long-context" framing to the corrected M.

### m1 [MINOR] — bf16/fp32 read-boundary reconciliation is deferred with a kill gate but no concrete casting design.

§8 item 2 correctly flags that `fla`'s production kernel rejects fp32
(`DELTANET_REALDATA_DESIGN.md` §4.3, verified lines 650–696) while NCR's
far-depth read wants an fp32 shadow instrument, and it KILLS Option B if
irreconcilable — but it does not specify the casting boundary (the NCR head
is a post-backbone side-channel per §2.2 line 188–191, so running it in fp32
on detached bf16 hidden states is almost certainly feasible). Write the
concrete fp32-read / bf16-backbone interface into the Phase-0 smoke so the
gate has something to test, not just a kill condition.

### m2 [MINOR] — kill trigger §8-item-2 is adjudication-by-vibes ("cleanly").

"gradient does not flow **cleanly** through the … bf16 kernel boundary"
(§8 item 2) has no threshold, unlike items 3/4/5's exact bars — set one
(e.g. finite-difference grad-check relative error ≤ a bf16-appropriate tol,
matching `DELTANET_REALDATA_DESIGN.md` §4.3's Tier-1 convention).

### m3 [MINOR] — the ≤5% overhead assumption and the >20% kill trigger leave an unguarded 5–20% band.

§6.2 prices every NCR training cell at the ≤5% overhead assumption (§2.2),
but §8 item 3 only KILLS the pricing at >20%. An overhead of 5–20% passes
the kill gate yet silently invalidates the specific ledger numbers. Tighten
item 3 to "re-price if Phase-0a measures >5% (the pricing assumption), kill
if >20%."

### Verdict

**BUILD-BLOCKED** (F1 is FATAL; M1–M6 are MAJOR).

- **F1 [FATAL]** — §1 (and §7's overall verdict) demand non-solvable
  STRUCTURAL hardness and O(log h) repeated-squaring on ONE query family;
  these are mutually exclusive (single-operator power ⇒ cyclic/solvable ⇒
  not NC¹-hard; distinct-generator word ⇒ no squaring shortcut ⇒ O(L)) —
  confirmed by the design's own M4/§4b/guard-2. Fix: rewrite §1/§7 as an
  explicit conjunction of two results on two families; scope O(log h)/Axis-B
  to Task 1 only.
- **M1 [MAJOR]** — Task 2 not well-posed at K=32: S₅ (order 120) has no
  action on 32 entities, and the reused `CAPABILITY_SEPARATION` infra is
  ℝ^{d_min} rotation reps, not discrete-entity permutations; "already
  calibrated" overclaims A5/A6 (hard-stopped).
- **M2 [MAJOR]** — §9 K≤15 fallback specifies cyclic "15-cycle products,"
  which cannot carry Task 2's non-solvable structural claim; the fallback
  silently drops the flagship axis.
- **M3 [MAJOR]** — the §9 ortho-write gate is calibrated on cyclic/random-
  orthogonal K-cycles read at O(L), never on S₅/A₅ writes; a WIN verdict
  does not transfer to Task 2's object.
- **M4 [MAJOR]** — the GPU-h ledger prices at batch=32/seq=512 (0.236 s/step,
  fixed step count) while §6.4 plans batch=96/seq=2048; and Phase-0a's rate
  probe is planned PACKED, so it cannot yield the unpacked Phase-1–3 rate.
- **M5 [MAJOR]** — Axis B gates the flagship verdict but its baseline is
  unchosen + unpriced and its ≥10× bar is transplanted from h≈10³–10⁶ onto a
  ≤61 accuracy-claimable ladder (and, per F1, cannot apply to Task 2 at all).
- **M6 [MAJOR]** — §4.3's pre-registered `cap_length(M=32,R=32)≈966` is wrong;
  recompute gives 60.5 (966 needs M=512).

---

## §A1-ADJUDICATION (coordinator, 2026-07-16 — recorded before dispatching Rev 1)

**FATAL (two-properties-one-family): CONFIRMED — by convergent independent
derivation.** The coordinator derived the same defect independently BEFORE
dispatching the attack (recorded verbatim as the attack's surface #1); the
attacker then confirmed it with three line-level citations from this
design's own binding sources (NOVEL_ARCH_WATERFALL.md M4:110-113 +
B-CHAIN:1834; NCR_ORTHO_WRITE.md §4b:239). Powers of one operator = cyclic
= solvable ⇒ no Barrington hardness; non-solvable words = no squaring
shortcut ⇒ O(L). §1/§7 as written are unsatisfiable.
**M6: CONFIRMED by coordinator recomputation** — `32×139,392/73,728 = 60.5`
tokens; the quoted `≈966` is the M=512 row (`512×1.8906 ≈ 968`).
Remaining MAJORs accepted on the attacker's line-level evidence (M1–M5),
consistent with the coordinator's §9-tension read at dispatch time.

**Disposition:** BUILD-BLOCKED ACCEPTED. Rev 1 dispatched with binding
requirements:
(a) FATAL fix: rewrite §1 hypothesis + §7 overall verdict as an EXPLICIT
CONJUNCTION of two results on two DISCLOSED query families — Axis A
(structural: Task 2 non-solvable word chains, read at O(L), NO complexity
claim) AND Axis B (complexity: Task-1-family single-operator powers,
O(log h) vs O(h), NO hardness claim) — the flagship claim being that ONE
deployed model delivers both, which the grounding memos still certify as
unclaimed;
(b) M1: make Task 2 well-posed — write the GENERATOR SET (S₅ needs 2
generators) as d_min-dim rotation-rep operators (the actual
CAPABILITY_SEPARATION infra), decoupling Task 2 from the K=32 entity-pool
arithmetic; disclose A5/A6 hard-stop history honestly ("reused" ≠
"calibrated");
(c) M2: the K≤15 fallback must preserve both axes or mark the structural
axis REDUCED — no silent drop;
(d) M3: add a cheap pre-Phase-1 BRIDGE CELL — ortho-write trained on
S₅-generator writes — because the cyclic-calibrated ortho verdict does not
transfer by fiat;
(e) M4: re-express the ledger in TOKENS; Phase-0a measures packed AND
unpacked rates;
(f) M5: pin the Axis-B baseline architecture, price it via Phase-0a, and
replace the transplanted ≥10× bar with a depth-range-appropriate
discriminator (e.g. fitted wall-clock-vs-h scaling exponent, log vs
linear) or justify 10× on the actual ladder;
(g) M6: fix the arithmetic and re-derive the M grid; (h) minors folded in.
Rev 1 output → fresh independent ATTACK ROUND 2 before build authorization.
Everything remains CONDITIONAL on the ortho-write verdict.

---

## §R1 REVISION 1 (2026-07-16) — changelog, every §A1 finding mapped to
its exact fix, with section references

**Scope discipline.** §1–§9 above are revised IN PLACE; §A1 and
§A1-ADJUDICATION are left byte-intact as historical record, per the
gauntlet-bookkeeping convention (`CLAUDE.md`). This changelog is the
single place a reader can verify every one of the adjudication's eight
binding items (a)–(h) was actually discharged, and where.

| Finding | §A1-ADJUDICATION item | Exact fix | Where |
|---|---|---|---|
| **F1 [FATAL]** — §1's hypothesis demands non-solvable structural hardness AND O(log h) repeated-squaring on the SAME query family; provably mutually exclusive (single-operator power ⇒ cyclic/solvable; distinct-generator word ⇒ no squaring shortcut ⇒ Θ(L)) | (a) | §1 rewritten as an explicit CONJUNCTION of Axis A (Task 2, structural, read Θ(L), no speed claim) and Axis B (Task 1, complexity, O(log h), no hardness claim), with an explicit "no single family carries both" disclosure sentence. §2.1's read-mechanism paragraph splits `binexp_read` (Task 1/3) from `loop_read` (Task 2) and states they are not the same function. §3's intro paragraph and §3.1/§3.2 headers re-labeled (both now PRIMARY, one per axis, neither "secondary"). §4.4 re-scoped to Task 1 only. §7's table and overall-verdict rule rewritten around the two-family conjunction, with an explicit "removed" note stating the old single-family WIN path is no longer reachable. | §1; §2.1; §3 (intro), §3.1, §3.2; §4.4; §7 |
| **M1 [MAJOR]** — Task 2 not well-posed at K=32: S₅ (order 120) has no action on 32 points; `CAPABILITY_SEPARATION`'s infra is ℝ^{d_min} rotation reps, not discrete-entity permutations; "already calibrated" overclaims A5/A6 (hard-stopped on a DIFFERENT task) | (b) | §3.2 rewritten: S₅'s verified 2-generator set `{t,c}` (symmetric closure `{t,c,c⁻¹}`, size 3) rendered as `4×4` (padded `5×5`) rotation-rep operators (`CAPABILITY_SEPARATION_DESIGN.md` §1.3, reused verbatim, verified real); the natural action is on S₅'s own 5 letters, giving Task 2 its own `K₂=5`, decoupled from Task 1/3's `K∈{15,32}`; a K/R table states what K and R mean per task; the A5/A6 hard-stop history is disclosed accurately (Rev 6 hard-stop was on `CAPABILITY_SEPARATION`'s OWN restricted-rank-recovery task, later diagnosed/lifted in Rev 7 — none of that calibration transfers to Task 2's discrete composition task, which needs its own Gate-0, provided by the bridge cell, M3). §5.3 updated to state the held-out-entity split applies per-episode at `K₂=5`. §2.1 adds the per-task `d_ncr` note and recomputes Task 2's own param count. | §3.2 (full rewrite); §5.3; §2.1 |
| **M2 [MAJOR]** — §9's K≤15 fallback specifies cyclic "15-cycle products," silently dropping Task 2's non-solvable structure | (c) | §9 split into two INDEPENDENT gates (§9.1 main K-axis, §9.2 the new bridge-cell gate). GATE 1's NULL/FAIL branch no longer touches Task 2 at all (M1's decoupling makes the old "R×15-cycle" swap unnecessary, not just wrong — retired, not repaired). GATE 2 (bridge cell) explicitly marks Axis A REDUCED on its own PARTIAL branch (re-anchored `L*`, disclosed) and DROPPED on NULL/FAIL (disclosed, §7's overall verdict capped at PARTIAL) — no silent-drop path remains. §7's Axis-A WIN/PARTIAL/NULL cells cross-reference the bridge-cell gate explicitly. §8 item 7 (new) states the risk plainly. | §9.1, §9.2; §7 (Axis A row, overall verdict); §8 item 7 |
| **M3 [MAJOR]** — the ortho-write gate is calibrated on cyclic/random-orthogonal K-cycles read at O(L), never S₅/A₅ writes; a WIN does not transfer to Task 2 | (d) | New Phase 0b BRIDGE CELL (§6.2): NS-polar write trained on S₅'s actual `{t,c,c⁻¹}` generators (`R=3, d=5`), priced as a step/seed-reduced PROJECTION from `NCR_ORTHO_WRITE.md` Part B's measured rate (≈4.24 GPU-h/320K-step cell) — ≈2.12 GPU-h (1×) / ≈4.24 GPU-h (2×) — with its own WIN/PARTIAL/NULL/FAIL bands (Gate-0 + one L=8 checkpoint) as a HARD GATE for Phase 1's new Task-2 calibration arm (§6.2 Phase 1). §9.2 states the gate's consequences for Axis A explicitly, independent of GATE 1's own verdict. | §6.2 (Phase 0b, Phase 1 Task-2 arm); §9.2; §6.3 item 5 |
| **M4 [MAJOR]** — the GPU-h ledger fixes STEPS while §6.4 changes batch/seq (token-count mismatch); Phase-0a's rate probe is planned PACKED, which cannot yield the Phase-1–3 unpacked rate | (e) | §6.1 adds a tokens/sec derivation and states the held-fixed invariant is TOKENS, not steps, with an explicit re-pricing rule if the operating point changes. §6.2 restates every phase's budget in tokens (327.68M/cell calibration & 392M-main, 1.108B/cell 98M-main) and re-derives each phase's GPU-h as `tokens/(tokens_per_sec×3600)` — verified to reproduce the pre-Rev-1 numbers exactly at today's (32,512) operating point, so no number silently moved. §6.2 Phase 0a is split into a PACKED throughput-only measurement (Phase-0/0b scheduling only) and an UNPACKED measurement taken AT the operating point §6.4's saturation pilot selects, with explicit sequencing (pilot before probe). §6.4's own packing paragraph is rewritten to scope packing to Phase 0/0b ONLY and explicitly forbid packing the Phase-0a unpacked rate probe. | §6.1; §6.2 (Phase 0a, Phase 1/2/3 token restatement); §6.4 |
| **M5 [MAJOR]** — Axis B gates the flagship verdict but its baseline is unchosen + unpriced; the ≥10× bar is transplanted from h≈10³–10⁶ onto a ≤61 accuracy-claimable ladder; and (per F1) cannot apply to Task 2 at all | (f) | §4.4 rewritten: architecture PINNED to extended-eigenvalue DeltaNet (Grazzi et al., `β∈(0,2)`), justified on cost grounds (reuses this repo's already-kernel-verified custom `fla` block, vs. RWKV-7's zero-measured-rate new kernel); explicitly re-scoped to Task 1 only (Task 2 has no O(log L) claim to compare, per F1). The ≥10× bar is REPLACED with (i) a provable, hardware-independent dependency-chain-length assertion (`⌈log₂h⌉` vs `h`, a build-time PASS/FAIL check) as the PRIMARY criterion, and (ii) a secondary wall-clock fit (`Model_log` vs `Model_lin`, OLS, `R²≥0.90` with a `≥0.05` margin over the wrong-family fit — a stated, justified relaxation of the toy-scale `R²≥0.998` precedent) measured on an extended timing-only ladder (`h∈{61,…,20000}`) decoupled from the accuracy-claimable ladder. §7's Axis-B row and §8 item 6 rewritten to match. Phase 0a (§6.2) now prices this arm's own rate probe explicitly (previously excluded). | §4.4 (full rewrite); §7 (Axis B row); §8 item 6; §6.2 (Phase 0a) |
| **M6 [MAJOR]** — §4.3's `cap_length(M=32,R=32)≈966` is arithmetically wrong (correct value 60.5; 966 is the M=512 row) | (g) | §4.3 rewritten: the arithmetic error is fixed and explained (966 belongs to M=512). Recomputing under M1's own fix (Task 2 is now `R=3, d=5`, not `R=4–32, d=33`) finds the structured-bank state is even smaller (300 bytes) than previously assumed — a NEW, more extreme finding, disclosed. TWO coherent M-grids are re-derived, each verified to clear the 20-token floor at every grid point (the pre-Rev-1 draft's own grid did not, for either case, on direct recomputation) — Case (i) single-relation `M∈{384,…,6144}`, Case (ii) structured-bank `M∈{5120,…,81920}`. | §4.3 (full rewrite) |
| **m1 [MINOR]** — bf16/fp32 boundary reconciliation has a kill gate but no concrete casting design | (h) | §8 item 2 now specifies the concrete cast pipeline (fp32 upcast at the NCR head's own input boundary, full fp32 internal pipeline, bf16 downcast at output) as a named Phase-0 smoke item. | §8 item 2 |
| **m2 [MINOR]** — the "cleanly" kill-trigger has no threshold | (h) | §8 item 2 pins an exact threshold: gradient cross-check at `<1×10⁻²` relative tolerance, reusing `DELTANET_REALDATA_DESIGN.md` §4.4's own Tier-1 convention verbatim (both forward outputs and gradients checked). | §8 item 2 |
| **m3 [MINOR]** — the ≤5% pricing assumption and the >20% kill trigger leave an unguarded 5–20% band | (h) | §8 item 3 rewritten as a two-tier rule with the gap closed, not left open: `>5%` triggers a defined MANDATORY RE-PRICE (was previously undefined in that band); `>8%` (tightened from 20%, a 2.5× reduction) triggers KILL, justified against the ≤1.2%-measured `FROZEN_BIAS_LM_DESIGN.md` precedent (8% ≈ 6.7× that analog — generous but non-vacuous). Every point in `(5%,∞)` now has a defined consequence. | §8 item 3 |

**Net effect on the compute ledger (§6.2).** Grand total moves from the
pre-Rev-1 draft's ≈462 GPU-h (2×, excluding the rollout baseline) / ≈580
GPU-h (2×, including a same-order-of-magnitude rollout placeholder) to
**≈482 GPU-h (2×, now including the rollout baseline's OWN Phase-0a rate
probe, plus the bridge cell and Task 2's doubled Phase-1 calibration) /
≈602 GPU-h (2×, same rollout-arm full-training-cell placeholder logic
applied on top)**. The increase is fully attributable to closing gaps the
attack round found (M3's bridge cell, M4/M5's now-included rollout-arm
Phase-0a probe, M1/M3's Task-2 calibration arm) — no number moved without
a named cause.

**What is NOT yet resolved, carried forward explicitly for ATTACK ROUND
2 (not silently deferred):** (1) the bridge cell's own cost is a
PROJECTION, not a measurement (§6.3 item 5) — its true rate, and whether
the `R=3/d=5` object trains at all, are open until it runs; (2) the §4.4
Axis-B statistical criterion's `R²` thresholds (`0.90`, margin `0.05`)
are this design's own pinned choice, justified but not measured against
any precedent at LM scale; (3) whether S₅-on-5-points, once actually
built, supports BOTH held-out axes (depth and entity) cleanly at such a
small `K₂` is asserted by analogy to Task 1's existing mechanism, not
independently verified; (4) the extended-β DeltaNet patch's own
wall-clock/stability behavior at LM batch sizes is entirely unmeasured
(§4.4, §6.3 item 3a) — Phase 0a is now BUDGETED for it (§6.2) but has not
run. Everything in this document remains CONDITIONAL on both §9 gates
(GATE 1: main ortho-write verdict; GATE 2: the bridge cell) and on a
fresh, independent ATTACK ROUND 2 before build authorization.
