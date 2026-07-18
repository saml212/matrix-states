# NCR REAL-LM DESIGN — the flagship bet: NCR in a real language model at scale

**REV-2.1 — CLEAR-FOR-CONDITIONAL-BUILD (coordinator freeze 2026-07-16
after direct verification of the §R2.1 edits, per §A3's no-fourth-round
ruling; gauntlet: draft → attack 1 [BUILD-BLOCKED, FATAL F1] → Rev 1 →
attack 2 [BUILD-BLOCKED, FATAL F2] → Rev 2 → round 3 [REVISE, narrow] →
Rev 2.1 → coordinator verify [purge complete, margin pinned, ledger
re-summed 483.78≈484]). Execution TRIPLE-GATED per §9: GATE 1 = ortho-write
verdict WIN/PARTIAL; GATE 2 = bridge cell (n=3, 3.18 GPU-h, pinned ≥0.2
margin); GATE 3 = Phase-0/1 calibration Gate-0. Opened 2026-07-16, revised
2026-07-16 (Rev 1) per §A1-ADJUDICATION, revised again 2026-07-16 (Rev 2)
per §A2-ADJUDICATION, revised a third time 2026-07-16 (Rev 2.1, surgical
freeze-scope only) per §A3-ADJUDICATION following §A3 ROUND 3's REVISE
verdict — changelog §R2.1, appended at the end of this document.** This
document is a design draft,
produced for adversarial attack. No code is built, no GPU is touched, by
this document. Every number below is either (a) cited from a measured rate
elsewhere in this repo, with the exact source named, or (b) flagged
`PRICE-UNKNOWN` / `PROJECTED — Phase 0a confirms`. Citations to external
literature were originally placeholder-tagged `[TO-VERIFY]`; the grounding
memos landed during the first drafting session (see "GROUNDING UPDATE"
immediately below) and every citation is either VERIFIED (source named,
§3.4's table) or explicitly flagged as still needing a human spot-check —
no `[TO-VERIFY]` tags remain in this document's claim language. **§1-§9
below were first REVISED IN PLACE against ATTACK ROUND 1's BUILD-BLOCKED
verdict (§A1/§A1-ADJUDICATION, historical record; the fix map is §R1) and
are now REVISED IN PLACE A SECOND TIME against ATTACK ROUND 2's
BUILD-BLOCKED verdict (§A2/§A2-ADJUDICATION, left untouched below as
historical record); every round-2 finding is mapped to its exact fix in
the new §R2 REVISION 2 changelog at the end of this document.**

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
for §3's mechanistic-length-generalization task below).

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

## §1 HYPOTHESIS (REVISED, §R2(a) — fixes ATTACK ROUND 2's FATAL F2; §R1(a)'s
two-family conjunction, which fixed ROUND 1's FATAL F1, is KEPT — F2 did
not reopen F1, it found a second, deeper defect INSIDE Axis A specifically)

**§A2's F2 finding, adopted, not disputed.** Axis A's "cannot, not merely
does not" framing is UNEARNABLE at any depth this design can test. The
barrier it invokes (the word problem of any non-solvable finite group is
NC¹-complete, Barrington 1989; log-precision transformers ⊆ uniform TC⁰,
Merrill & Sabharwal, arXiv:2207.00729) is an ASYMPTOTIC, ALL-`L` statement
— it makes no prediction of failure at any FIXED, bounded path length. For
fixed `L` over a fixed finite group, the answer function is a
constant-size computation, trivially in AC⁰⊆TC⁰. This design's own
corroborating citation, Liu et al. ("Transformers Learn Shortcuts to
Automata," ICLR 2023 oral, arXiv:2210.10749, C4), proves the opposite at
the tested scale: a transformer of depth O(log T) EXACTLY simulates any
semiautomaton on length-T inputs, including the S₅ word-problem automaton
— for Task 2's own eval ladder, `⌈log₂40⌉=6` layers suffice, and the
param-matched Transformer baseline (§4.1) is depth-matched to 12 (98M) /
16 (392M) layers, i.e. it carries 2–2.7× the layers a C4-style exact
shortcut needs. "The Transformer FAILs because it is structurally barred"
is therefore not merely untested at `L≤40` — it is PREDICTED AGAINST by
this design's own cited literature. Pushing `L` past the barrier's real
regime (`L≳2^{n_layers}≈4096` at 98M, `≈65536` at 392M) does not rescue
the claim either: NCR's own measured far-depth exactness ceiling
(`NCR_ORTHO_WRITE.md` §3 — even a perfectly polar-orthogonalized operator
recovers only ~0.14–0.35 by physical depth 253) sits well below either
barrier regime. **There is no depth at which the Transformer is
structurally barred AND NCR is still exact.** This is round-1's F1 one
level deeper — the two-family split (below) relocated the impossibility,
it did not remove it.

**The reframe (§A2-ADJUDICATION's binding disposition, adopted verbatim,
consistent with the PI's capability-first directive, which explicitly
covers separations "functionally or AS OBSERVED/TESTED," not only
provably-in-principle).** Axis A is no longer a structural-impossibility
claim. It is a MECHANISTIC LENGTH-GENERALIZATION separation: both arms
train on non-solvable-group word chains of length `L≤L_train` and are
evaluated at `L_test≫L_train`, still safely inside NCR's own established
exactness range (§3.2 pins the exact split and justifies the numbers).
The pre-registered prediction is that the Transformer LEARNS a shortcut —
C4's own constructive result guarantees one EXISTS and is learnable at
every tested `L` — that succeeds in-distribution but FAILS out-of-
distribution in length: the exact brittleness C4's own secondary finding
documents (its parity models "leverage the correlation between position
and accumulated sum" rather than the true recurrent rule, and fail when
that correlation is disturbed), independently corroborated by this
program's own Guu, Miller & Liang precedent (composition error cascades
with path length, C8/N5, already cited in this design for Task 1's
empirical-drift claim — the SAME mechanism now grounds Task 2's). NCR's
read executes the IDENTICAL exact linear-algebraic composition at every
depth by construction, so it has no length-dependent shortcut to overfit
and no mechanism by which OOD length degrades it (short of the same
fp-exactness ceiling that governs Axis B). **This is a claim about what
the two architectures' LEARNED SOLUTIONS generalize like, not about what
either architecture can in-principle compute.** C1–C3 (TC⁰⊊NC¹,
Barrington) are retained only as MOTIVATION for why a bounded-depth
shortcut — not a length-general algorithm — is the object SGD is expected
to find (a TC⁰-computable object is by definition a constant-depth,
hence length-insensitive-capacity, circuit family, the standard reason
such circuits tend toward position/length-keyed pattern-matching rather
than unbounded recursion) — **never again as a finite-`L` impossibility
bar.** Every "cannot, not merely does not" / "structurally barred at
tested depths" sentence is purged from this design (§3.2, §7, §8, §9,
below; §R2.1(a) COMPLETES this purge — Rev 2's own claim that the scope
was already §3.2/§7/§8/§9 was FALSE against the actual text, §A3 F3-1 —
the residual stale framing in §0, §2.1, §2.2, §3.1, and the §3.2 HEADER
itself is fixed there too, so the true purge scope is §0, §2.1, §2.2,
§3.1, §3.2, §7, §8, §9). The flagship headline downgrades honestly from
"cannot" to "does-not-and-we-show-why-ours-must," disclosed here plainly,
not softened.

**Mechanism instrument (§A2(a)(ii)'s requirement, priced).** Beyond the
accuracy gap, this design instruments the SHAPE of each arm's own
accuracy-vs-`L` curve, reusing data every eval pass already collects
(§3.5's mandatory per-stratum reporting) — no new measurement, only a
specific, pre-registered READING of it: a SHORTCUT signature is high
in-distribution accuracy (`L≤L_train`) that COLLAPSES at OOD lengths; an
ALGORITHM signature is flat accuracy across the whole tested range, in-
and out-of-distribution alike. This is the operational content of
"shortcut vs. algorithm" and is what makes the WIN band below a claim
about MECHANISM, not merely a threshold crossing (§7, revised). An
OPTIONAL, cheap, non-gating secondary diagnostic — attention-map
extraction on the trained Transformer checkpoint at OOD `L`, checking
whether attention mass stays spread over all `L` generator positions (an
algorithm signature) or collapses to a fixed/truncated window (the
position-correlation shortcut signature C4 itself diagnoses) — is named
here as a REPORTED-NOT-GATING enrichment, priced at near-zero (post-hoc
analysis of an already-trained checkpoint, no new GPU-h), run only if
Phase-2 checkpoints are available.

**Axis A — mechanistic length-generalization (Task 2, §3.2, read at O(L),
NO complexity claim, NO finite-`L` impossibility claim).** On non-solvable-
group word chains — a path of `L` DISTINCT written generator-operators
composed by exact matrix multiplication, one matvec per hop
(`o = loop_read(bank, path, q)`, cost Θ(L), stated plainly, not disguised
as sub-linear) — trained on `L≤L_train` and evaluated at `L_test≫L_train`
(§3.2 pins the split), the NCR-augmented LM answers EXACTLY at every
tested length, in- and out-of-distribution alike, while the param-matched
Transformer baseline learns a shortcut that succeeds in-distribution (C4's
own constructive guarantee that one exists and is learnable) but degrades
at OOD lengths (C4's own brittleness finding + Guu/Miller/Liang's
composition-error-cascading precedent, C8/N5). NCR's own advantage here is
EXACTNESS THAT SURVIVES LENGTH SHIFT (linear-algebraic composition never
drifts with depth), not speed — its read is the same asymptotic order,
Θ(L), as the baseline's.

**Axis B — complexity (Task-1-family single-operator power queries, §3.1,
read at O(log h), NO hardness claim). UNCHANGED by this revision — F2 is
an Axis-A-only defect; the §4.4/§7 statistical-criterion fix (M8, below)
is a measurement-protocol repair, not a hypothesis change.** On queries
requiring repeated application of ONE written operator — `Z^h` for a
single Hamiltonian K-cycle, a CYCLIC hence SOLVABLE group, carrying no
TC⁰/NC¹ argument — the SAME deployed model answers EXACTLY at query-time
sequential cost O(log h) via `binexp_read`'s repeated squaring, where
every published matrix-state alternative (DeltaProduct/RWKV-7-class —
which, per Grazzi et al. arXiv:2411.12537 and Peng et al. arXiv:2503.14456,
CAN in principle reach the correct answer) pays Θ(h) sequential
state-update steps for the same depth. This is an algorithmic/circuit-
depth separation at MATCHED expressivity and MATCHED accuracy, not an
accuracy separation, and not a hardness claim — the Transformer is not
predicted to fail this task's underlying group-word problem (§3.1's own
citation is empirical composition-drift, not structural).

**The flagship claim: ONE deployed model — literally one NCR head, one
write mechanism, one shared parameter set (§2.1, revised, fixes M7+M9
jointly) — trained once per scale on a shared curriculum, delivers BOTH
properties: NCR composition length-generalizes on non-solvable-group
chains where a comparably-sized Transformer's learned shortcut does not
(Axis A), AND the same read mechanism accesses single-operator powers at
O(log h) query-time depth where a sequential-rollout baseline of matched
expressivity pays O(h) (Axis B).** No single query family can carry both
properties simultaneously (§A1's F1 group-theory result, unaffected by
this revision — a non-solvable word has no squaring shortcut by
definition, and a single operator's powers can never be non-solvable) —
disclosed plainly, not papered over; the two-family design remains the
only way to state a claim that is simultaneously true and non-vacuous.
**"One model" is now an architectural FACT, not an aspiration:** §2.1's
fix (M7+M9) makes both axes' writes full-rank `33×33` orthogonal objects
produced by the SAME encoder at the SAME shape — Task 1 writes a K-cycle
permutation, Task 2 writes `ρ_{S₅}(g) ⊕ I_{29}` (§3.2's cited
`CAPABILITY_SEPARATION_DESIGN.md` §1.4 realization) — so there is no
separate Task-2 head, no separate param line, and no rank-deficiency
escape hatch left in the design.

**What this hypothesis explicitly does NOT claim** (the correction this
document's grounding update, above, forced, unchanged by this revision):
that matrix-valued fast-weight state can state-track where
vectors/Transformers categorically cannot — that is already published
(Grazzi et al., DeltaProduct, RWKV-7) and any framing implying otherwise is
retired from this design. **Axis A's claim, narrowed again by this
revision: EXACT composition that GENERALIZES ACROSS LENGTH where the
Transformer's learned, C4-guaranteed-to-exist shortcut empirically does
not — not that the Transformer is categorically incapable of the task at
any fixed length** (it is not; C4 constructs a shortcut for every tested
`L`). Axis B's claim is narrower still: O(log h) query-time depth,
unclaimed anywhere in the searched literature
(`research/ncr_separation_grounding.md` Part 2) against baselines that are
expressivity-capable but not depth-efficient. Two falsifiable bets, not
one, and §7 (revised) scores them independently before combining them
into an overall verdict.

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
ortho-write gate (§9, revised) licenses it. **`d_ncr` is now ONE SHARED
CONSTANT for BOTH task families, not per-task (§R2(b) fix, M7+M9 —
supersedes Rev 1's per-task `d_ncr,2` split, which M7 proved
rank-deficient and M9 proved undercut the "one model" claim either way it
was resolved).** Task 1/Task 3's abelian construction still uses
`d_ncr = K+1 ∈ {16, 33}` (the standing tight-spare convention, gated on
§9's K-axis branch); **Task 2's non-solvable-group construction now uses
the SAME `d_ncr`** — its `4×4` S₅ generator matrices are embedded as
`ρ_{S₅}(g) ⊕ I_{d_ncr−4}` (block-diagonal, identity on the ambient
complement — the FULL-RANK realization `CAPABILITY_SEPARATION_DESIGN.md`
§1.4 actually built and verified, `d_state(G)=d_min(G)+2` there, generalized
here to whatever `d_ncr` Task 1/3's own K-axis lands on, since the identity
block absorbs any ambient dimension ≥4; §3.2 states the construction and
its full-rank floor precisely). At the primary K=32 operating point,
`d_ncr=33` and Task 2 writes `ρ_{S₅}(g) ⊕ I_{29}`; if GATE 1 (§9.1) falls
back to K=15/`d_ncr=16`, Task 2 writes `ρ_{S₅}(g) ⊕ I_{12}` instead — the
SAME construction at whatever shared `d_ncr` is in force, never a
separate Task-2 dimension. **One NCR head, one encoder, one set of
weights, serving both query families — the "one model" claim (§1) is now
architecturally literal**, not a build-time choice between two lossy
options (Rev 1's own unresolved dilemma, M9): a single padded head is no
longer rank-deficient (M7 is closed by construction, not by assumption)
and two disjoint heads are no longer needed.

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

**Param-count delta.** ONE NCR head, standing
K=32/d_ncr=33/h_ncr=64 convention: `P(d,h) = 40h² + 4dh + 46h + d`
(verified exact formula, `NOVEL_ARCH_WATERFALL.md` §9.3) = 40·4096 +
4·33·64 + 46·64 + 33 = 163,840 + 8,448 + 2,944 + 33 = **175,265 params**
(matches `NCR_ORTHO_WRITE.md` §6's independently-stated "~175K"). At a
98M/392M Transformer backbone this is **+0.18% / +0.045%** — negligible.
**Task 2 (§R2(b) fix, M7+M9) adds ZERO incremental params**: it is the
SAME head, the SAME `d=33, h=64` shape, writing a differently-CONTENTed
but identically-SHAPED full-rank orthogonal operator (`ρ_{S₅}(g) ⊕ I_{29}`
in place of a K-cycle permutation) — there is no separate Task-2 param
line to report anymore (Rev 1's own `d=5` head, 168,069 params, and its
"barely smaller than the K=32 head" observation, are both retired by this
fix, not merely revised). Rev 1's underlying finding — that the `40h_ncr²`
encoder-width term dominates and is barely moved by `d` shrinking —
generalizes directly to why unifying the two tasks' write shape costs
nothing extra: the same term dominates regardless of what CONTENT the
head is trained to write. This remains the load-bearing point for the
bridge cell's own pricing (§6.2): cost is governed by the encoder's own
width, not by the written object's dimension or count, and the transfer
from Part B's measured rate is now even TIGHTER than Rev 1's own
conservative non-discount, since Task 2's `d` is no longer smaller than
Part B's own `d=33` at all — it is identical.

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
**Complexity-theoretic MOTIVATION for THIS baseline specifically — asymptotic,
NOT a finite-`L` bar (§R2.1(a) completes the F2 purge here, §A3 F3-1: this
paragraph previously re-asserted the retired "structurally barred" framing
that §1's reframe, above, already retires)** (the Transformer, any depth,
any width): log-precision transformers are contained in uniform TC⁰
(Merrill & Sabharwal, TACL 2023, arXiv:2207.00729); the word problem of any
fixed NON-SOLVABLE finite group (e.g. S₅, A₅, A₆) is NC¹-complete
(Barrington 1989); this is an ALL-`L` ASYMPTOTIC deduction (Merrill, Petty
& Sabharwal, ICML 2024, arXiv:2404.08819, making it explicit against S₅)
that does NOT bite at any tested finite `L` — §1's own C4 citation
constructs an exact bounded-depth Transformer shortcut at every tested
length, so the Transformer is NOT predicted to fail this task outright.
This is a genuine complexity-theoretic MOTIVATION, not a finite-`L`
impossibility claim — it grounds WHY a bounded-depth, length-brittle
shortcut, not a length-general algorithm, is the object SGD is expected to
find (§1's mechanism instrument), and it is only RELEVANT (never "biting"
in the impossibility sense) if the task's relation-composition structure is
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
the Transformer, §4.1, per §2.1's complexity-theoretic MOTIVATION, not a
structural bar, §R2.1(a)).** DeltaNet's native
transition is `I − βkkᵀ` (a Householder reflection); the standard sigmoid
gate restricts `β∈(0,1)`, which per Grazzi et al. (arXiv:2411.12537)
restricts transition eigenvalues to `[0,1]` and provably BARS
parity/non-solvable-group state tracking — i.e., under that gate the
PLAIN-DeltaNet-control arm (§4's implicit baseline, the backbone with no
NCR head) is ITSELF still TC⁰-limited on any hypothetical non-solvable
task (same class as diagonal SSMs, same Merrill/Petty/Sabharwal argument),
though this does not bear on Task 2 (§3.2) since Task 2's baseline of
record is the Transformer, not plain DeltaNet. If the box's
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

## §3 TASK SUITE (REVISED, §R1/§R2 — F1/M1/M2/M3/F2/M7/M9/M10 fixes; the two
axes now sit on TWO DISCLOSED, DISJOINT query families, per §1's rewritten
conjunction, with Axis A reframed as mechanistic length-generalization)

**Two-family design, per §1's F1-fixed conjunction (§A1 confirmed: the two
properties cannot coexist on one family).** Axis A — **mechanistic
length-generalization (§R2(a) reframe, fixes F2)**: Task 2 (§3.2, PRIMARY
for this axis) composes a PATH of DISTINCT generator operators drawn from
a non-solvable group; trained on `L≤L_train`, the param-matched Transformer
baseline learns a shortcut (C4 guarantees one exists and is learnable at
every tested `L`) that succeeds in-distribution but is predicted to fail
at `L_test≫L_train` (C4's own brittleness finding + Guu/Miller/Liang's
composition-error-cascading precedent, C8/N5) — an EMPIRICAL, OOD-length
prediction, never again framed as a finite-`L` structural impossibility;
NCR's own read here is Θ(L) — exact, not fast — via `loop_read`, stated
plainly, never called O(log L). Axis B — **complexity**: Task 1 (§3.1, PRIMARY for this axis, no
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
`ACC⁰`/`TC⁰` in principle, so a TC⁰ transformer has NO complexity-theoretic
motivation for failing this task at all — unlike Task 2 (§3.2, below),
whose motivation is asymptotic and non-binding at any tested `L` in any
case (§1's/§2.1's reframe, §R2.1(a)). The prediction here
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
cyclic group). Task 2 carries the mechanistic length-generalization
(Axis A) claim on a disjoint family where no such shortcut exists (§3.2,
below) — companions on different axes, neither subordinate to the other.

### 3.2 Task 2 — Non-solvable-group word-problem chain (PRIMARY for
Axis A — mechanistic length-generalization, well-posed construction,
§R1(b) fixes M1)

**M1, as adjudicated: construction rewritten to use the GENERATOR SET, not
a permutation acting on the K=32 entity pool.** The first draft conflated
two incompatible objects (`CAPABILITY_SEPARATION_DESIGN.md`'s continuous
`ℝ^{d_min}` rotation-representation infra vs. a discrete permutation on 32
named entities that infra was never built for — §A1 M1's evidence: S₅ has
no transitive action on 32 points, `32 ∤ 120`). The fix, below, pins one
concrete, well-posed object and decouples Task 2 from Task 1/3's K-axis
entirely.

**Group and generators (VERIFIED — reused verbatim from
`CAPABILITY_SEPARATION_DESIGN.md` §1.3's real, calibrated group
infrastructure for the generator matrices, AND §1.4's real, calibrated
FULL-RANK embedding construction for how they attach to an ambient
dimension — §R2(b) fix, M7+M9: Rev 1 cited §1.3 for the generators but
invented its own uncited `d_min+1` zero-pad for the embedding; this
revision corrects the citation to the construction that actually governs
orthogonalizability).** Primary group: **S₅** (order 120, non-solvable,
`d_min=4`, realized as the 4-dimensional standard representation on the
zero-sum hyperplane of ℝ⁵ — `CAPABILITY_SEPARATION_DESIGN.md` line 229,
`ρ_{S₅}`). Its verified symmetric generating set is `{t, c, c⁻¹}` (that
document's own line 897): **`t`** = a transposition (self-inverse) and
**`c`** = a 5-cycle (order 5) — **exactly the 2 generators (`t`, `c`) the
adjudication names**, `c⁻¹` included for the symmetric closure (size 3)
that licenses "walk either direction" paths. `ρ_{S₅}` is PINNED
REAL-ORTHOGONAL by construction (`CAPABILITY_SEPARATION_DESIGN.md` line
1088, "`ρ_G` is pinned real-orthogonal, §1.3.1"), so each generator is
itself an exactly orthogonal `4×4` matrix, not merely full-rank.

**The embedding — full-rank, not zero-padded (M7's fix).** Each generator
is embedded into the SHARED `d_ncr × d_ncr` ambient space (§2.1's fix,
M7+M9: `d_ncr=33` at the primary K=32 operating point, or 16 under GATE
1's K=15 fallback, §9.1) as **`ρ_{S₅}(g) ⊕ I_{d_ncr−4}`** — the EXACT
block-diagonal, identity-on-the-ambient-complement construction
`CAPABILITY_SEPARATION_DESIGN.md` §1.4 actually built (line 1012:
`d_state(G)=d_min(G)+2`; lines 1037–1038: `rho_G_embedded(g) = rho_G(g) ⊕
I_{d_state−d_min(G)}`) — Rev 1's "reused verbatim" claim was true of the
generator MATRICES but not of the embedding, which invented an un-cited
`d_min+1` zero-pad; this revision fixes the citation to the construction
that is actually load-bearing here. **Both blocks are orthogonal**
(`ρ_{S₅}` pinned real-orthogonal, above; `I` trivially orthogonal), so the
embedded generator is EXACTLY orthogonal in `ℝ^{d_ncr}`. **Singular-value
floor, stated explicitly (the M7 fix's core content): every singular
value of the embedded target is exactly 1, none is 0** — unlike Rev 1's
zero-padded `5×5` object, whose one structurally-zero singular value made
`‖QᵀQ−I‖_F=1` unfixable by any amount of Newton–Schulz iteration (§A2
M7's arithmetic, verified). NS-polar (`NCR_ORTHO_WRITE.md` §2) is
therefore WELL-POSED on this object in exactly the sense it is already
well-posed on Task 1's own K-cycle writes and Part B's own `R=4`
random-orthogonal bank (§4b) — a raw, generically full-rank (but not yet
orthogonal) encoder output pre-scaled and iterated toward its nearest
orthogonal matrix, with no dimension structurally excluded from
convergence. The `d_min+1` "tight-spare" convention (Task 1's own `K+1`
margin) does NOT transplant here and is retired, not reused: Task 2's
construction needs no spare dimension at all, because the identity block
already fills the ambient space exactly and orthogonally.

**The natural action — and where K₂=5 for Task 2 comes from (unaffected
by the M7/M9 embedding fix — this paragraph is about the ENTITY POOL,
K₂, a different axis from `d_ncr`, the WRITE dimension, §2.1).** S₅'s
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
much smaller K than Task 1/3's `K∈{15,32}` — the two remain DECOUPLED,
even though `d_ncr` (the write dimension) is now SHARED across tasks
(M7+M9's fix, above); K₂ (entity-pool size) and `d_ncr` (write-operator
dimension) are independent axes and must not be conflated.** A query
specifies a PATH of generator indices `(o_1,…,o_L)` from the size-3
symmetric set; the GROUND TRUTH is the exact permutation composite
`g_{o_L}∘…∘g_{o_1}` applied to the query letter, computed by exact integer
index arithmetic — no floating point anywhere in the target. The MODEL's
own answer is decoded by applying the composed (orthogonally-conditioned)
`d_ncr×d_ncr` operators to the query point's `ℝ⁴` image, ZERO-PADDED into
the shared `ℝ^{d_ncr}` embedding (the same padding-of-VECTORS convention
Task 1's own entity embeddings already use for their `K+1`-dim space — an
unproblematic operation, unrelated to M7's operator-padding defect: only
the WRITTEN OPERATOR's rank was ever at issue, never an input vector's),
and reading off which of the 5 canonical points (identically zero-padded)
it lands nearest (cosine similarity over exactly 5 candidates — a
legitimate discrete decode here because Task 2 tests COMPOSITION
EXACTNESS, not a rank-K necessity bound; the `CLAUDE.md`
argmax-defeats-rank-proofs rule governs a different, unrelated concern and
does not apply to this readout).

**R (operator/generator count) reconciliation — the other half of M1's
fix; `d` (write dimension) reconciliation ADDED this revision (M7+M9).**
Task 2's written-operator count is **R = 3** (`{t, c, c⁻¹}`, the verified
generating set size) — **NOT R=4** (the arbitrary random-orthogonal
convention `NCR_ORTHO_WRITE.md` §4b's discriminator cell used for its own
calibration purpose) and **NOT K=32** (the entity-pool conflation §A1 M1
caught). **What K, R, and `d` now mean, per task, stated explicitly (the
adjudication's own instruction, extended to cover `d` after M7/M9):**

| Task | K (entity pool) | R (written operators) | `d` (write-operator dimension) | Group / structure |
|---|---|---|---|---|
| Task 1 (§3.1) | `K∈{15,32}`, gated by §9's main ortho-write branch | 1 (single operator, `Z^h`) | `d_ncr=K+1∈{16,33}` | Cyclic (solvable) |
| Task 2 (§3.2) | **K₂=5, FIXED, decoupled from Task 1's K** | **R=3** (`t, c, c⁻¹`) | **`d_ncr` — SAME shared value as Task 1 (§R2(b) fix; no separate `d_ncr,2` anymore)** | S₅ (non-solvable) |
| Task 3 (§3.3) | Task 1's `K` (abelian default) OR K₂=5 (if it reuses Task 2's construction, build-time choice, unchanged) | Whichever it inherits | Whichever it inherits (trivially consistent either way now that `d_ncr` is shared) | Build-time choice, disclosed either way |

**K₂ (entity pool) stays decoupled from Task 1's K; `d_ncr` (write
dimension) is now SHARED — two different axes, kept apart on purpose so
they are not conflated again.**

**Why this is now a mechanistic length-generalization claim, not a
structural one (§R2(a), fixes F2 — supersedes Rev 1's "unchanged by this
revision" framing, which is exactly what F2 found unearnable).** §1's
revised hypothesis carries the full argument; restated at Task-2-
construction level: the word problem of any fixed non-solvable finite
group is NC¹-complete (Barrington, JCSS 1989, VERIFIED — `research/
ncr_separation_grounding.md` item 2) and log-precision transformers are
contained in uniform TC⁰ (Merrill & Sabharwal, TACL 2023, arXiv:2207.00729,
VERIFIED) — **but this is an ALL-`L` asymptotic statement, not a
finite-`L` one, and this design's own cited construction shows the
Transformer is NOT barred at any tested `L`.** Liu et al. (ICLR 2023 oral,
arXiv:2210.10749, C4, VERIFIED) is PROMOTED this revision from
"secondary/corroborating" to the PRIMARY predicted mechanism
(§A2-ADJUDICATION's instruction): their headline theorem constructs, for
ANY finite semiautomaton (S₅'s word-problem automaton included), an exact
O(log T)-depth transformer shortcut on length-T inputs — `⌈log₂40⌉=6`
layers suffice for Task 2's own `L≤40` ladder, well inside the 12/16-layer
depth-matched Transformer baseline (§4.1). So the baseline is PREDICTED TO
SUCCEED in-distribution (a shortcut provably exists and is learnable), not
predicted to fail. What C4 ALSO shows, as its SECONDARY finding, is that
such learned shortcuts are BRITTLE out-of-distribution (their parity
models "leverage the correlation between position and accumulated sum"
rather than the true recurrent rule, and fail when that correlation is
disturbed) — independently corroborated by this program's own Guu, Miller
& Liang citation (compositional path queries, arXiv:1506.01094, C8/N5,
VERIFIED: composition error cascades with path length). **This is the
actual predicted mechanism for Task 2:** train on `L≤L_train`, and the
Transformer's C4-guaranteed shortcut is predicted to succeed
in-distribution and COLLAPSE at `L_test≫L_train` — an empirical,
falsifiable, OOD-length-generalization prediction, not a finite-`L`
impossibility. **C1–C3 (Barrington, Merrill & Sabharwal, Merrill/Petty/
Sabharwal) are retained ONLY as MOTIVATION** for why SGD is expected to
converge on a bounded-depth, length-brittle shortcut rather than a
length-general algorithm on a task whose asymptotic word problem is
NC¹-hard (a TC⁰-computable object is, by definition, a constant-depth
circuit family — the standard reason such objects tend toward
position/length-keyed pattern matching) — **never again as a "cannot, not
merely does not" bar.** `NCR_ORTHO_WRITE.md`'s own §4b Part B construction
(the discriminator this bridge cell and this task both descend from)
already predicts the mechanism NCR's own advantage rests on for entirely
independent reasons: "a product of L orthogonal matrices is EXACTLY
orthogonal for every L" — composition exactness that is, by construction,
INDIFFERENT to whether `L` is in- or out-of-distribution, which is exactly
why NCR has no shortcut to overfit and nothing for a length shift to
break (short of the shared fp-exactness ceiling, §9). **NCR's own read
here costs Θ(L)** (`loop_read`, §2.1, revised) — the length-generalization
claim is about the Transformer's LEARNED-SOLUTION brittleness, not about
NCR being fast; see §1's revised hypothesis for why these do not conflate.

**Depth-ladder informativeness under fast Cayley-graph mixing — addressed
honestly, not assumed away (§R2(d), the second half of M10's fix).** The
S₅ `{t,c,c⁻¹}`-generator random walk MIXES in O(1) steps (~4–5 hops) — for
`L≳5` the TARGET distribution `w` is ≈uniform over S₅ at every rung, so
the ladder presents no graded TARGET-DIFFICULTY gradient (M10's finding,
accepted, not disputed: "L=5 and L=40 are statistically identical
tasks"). **This does not undermine the reframed design, because the
`L`-axis's job under length-generalization is different from a difficulty
gradient: it measures OOD DISTANCE from the trained length range, not
target-distribution hardness.** Even though the TARGET distribution is
fixed for `L≥5`, the INPUT REPRESENTATION differs sharply across `L` (a
physically longer generator-index sequence, more write/read invocations,
more sequential hops) — exactly the axis a bounded-depth, POSITION/
LENGTH-KEYED shortcut (C4's own diagnosed mechanism) is predicted to fail
to generalize across, even while the target distribution itself never
shifts. So "`L` is the OOD distance, not a difficulty gradient" is the
correct and sufficient reading post-reframe — mixing speed is a property
of the TARGET, irrelevant to whether a LEARNED SOLUTION keyed to input
length/position transfers across it. Any earlier framing in this design
implying `L=5` vs. `L=40` differ in intrinsic word-problem difficulty is
retired; the eval ladder's only job now is to place OOD probes at
increasing distance from `L_train`, which it still does.

**Mod-K-trap-proof by construction (three guards, `NCR_ORTHO_WRITE.md`
§4b's convention, transplanted verbatim, now doing double duty as BOTH the
held-out-depth hygiene AND part of the non-solvable-word-problem
structure):** (1) distinct generators per hop — a product of DIFFERENT
group elements, not a power of one matrix, so there is no single cycle
length to reduce modulo; (2) no consecutive repeats; (3) fixed-point
exclusion (query/path pairs whose composite fixes the start are excluded).

**L-train / L-test split (§R2(a), replaces Rev 1's `L∈{1,2,3}` train /
`L*=32` single-checkpoint design with an explicit ID-vs-OOD length
split).** **Train `L∈{1,…,8}`** (a dense range, not a sparse `{1,2,3}`
set — a length-generalization design needs the baseline to have genuinely
seen, and be fairly judged in-distribution across, the WHOLE trained
range, not just its bottom 3 rungs; by `L=8` the underlying Cayley walk
has already mixed several times over, which is expected and harmless per
the paragraph above). **Eval reuses the SAME `NCR_ORTHO_WRITE.md` §4b
ladder unchanged** (`L∈{5,8,12,16,20,24,32,40}`, no new infrastructure),
now explicitly re-partitioned around the new `L_train=8` boundary:
**`{5,8}` are IN-DISTRIBUTION anchor points** (both `≤L_train`, used to
measure the baseline's in-distribution-success conjunct, §7) and
**`{12,16,20,24,32,40}` are the OOD ladder** (`L_test`, all `≫L_train`,
used to measure the predicted shortcut collapse). `L_test`'s top rung, 40,
is `≈5×L_train` and sits `≪` NCR's own established far-depth exactness
range (`NCR_ORTHO_WRITE.md` §3: physical depth 253 is where a perfectly
polar-orthogonalized operator starts losing recovery to fp accumulation;
40 sits comfortably inside the already-audited Part B ladder, §4b, which
predicts exact recovery at every `L` for a product of orthogonal
matrices) — justifying "safely inside NCR's fp-exactness bound of ~253"
without extending past already-audited infrastructure. R is 3, generators
are S₅'s `{t,c,c⁻¹}`, embedded full-rank at the shared `d_ncr` (§2.1,
above) — not R=4 random-orthogonal matrices at a separate `d`; this
remains the corrected build-time change from `NCR_ORTHO_WRITE.md`'s own
Part B convention.

**Chance floor and re-derived accuracy bands (§R2(d), the first half of
M10's fix).** The answer is one of 5 letters; fixed-point exclusion
(guard 3, above) removes composites that fix the query letter, leaving
chance ≈0.25 over the 4 remaining candidates (M10's arithmetic, verified).
Rev 1's inherited `FAIL≤0.5` bar sat only 2× above this chance floor — a
model reading 0.5 is already double chance, not "no better than chance,"
so HOLD-vs-FAIL was not the clean categorical split the structural framing
implied. **Fix, chosen over the alternative (multi-point queries pushing
chance to 1/120 over the full permutation): re-derive the bands against
the measured chance floor, not extend the answer space.** Multi-point
(full-permutation) queries would need a new answer-encoding/readout
construction and its own re-design and re-audit — undischargeable within
this revision's scope; named here as a possible Stage-2 enhancement, not
built. **Re-derived bands (§7 states these formally): HOLD/WIN stays
≥0.9** (still a meaningful, ~3.6×-chance target, unaffected by the small
answer space); **FAIL moves to ≤0.35** (chance 0.25 + a 0.10 margin —
comfortably separates genuine at-chance performance from any real signal,
without requiring the old, misleadingly tight 0.5 cutoff); **DEGRADED
widens to (0.35, 0.9)**, capturing "learned something real, short of
exact" honestly instead of splitting it awkwardly across the old bands.

**A5/A6 hard-stop history — disclosed accurately, extended to cover the
NEW §1.4 embedding citation (§R2(e) closes M1's round-2 partial:
`CAPABILITY_SEPARATION` infra is now cited twice in this section — §1.3
for the generator matrices, §1.4 for the embedding construction — and the
disclosure below covers both).** `CAPABILITY_SEPARATION_DESIGN.md`'s Rev 6
(§1.28) recorded A5/A6 as HARD-STOPPED (a second consecutive
convergence-budget miss on THAT document's own restricted-rank-RECOVERY
task — an SVD-probe readout over a continuous state, a different loss
landscape and readout than Task 2's discrete word-composition-and-
next-token task); Rev 7 (§1.30–1.31) subsequently diagnosed the mechanism
(H-ENC: an `L=1` query-independent-attention degeneracy specific to
single-generator words, not a general training pathology) and LIFTED the
hard-stop, re-pinning per-group step budgets (S₅'s own budget: 8,000
steps) that all 5 groups now clear. **What transfers to Task 2 and what
does not, stated for BOTH citations:** the GENERATOR MATRICES (§1.3, S₅'s
verified 4-dim standard rep, the `{t,c,c⁻¹}` set, its closure/order
checks) and the EMBEDDING CONSTRUCTION (§1.4, `ρ_G(g) ⊕
I_{d_state−d_min(G)}`, the block-diagonal full-rank realization, §R2(b)
above) are both real, exact, and now genuinely reused verbatim — this
much genuinely "is already built," on BOTH counts. The CALIBRATION (the
step budgets and Gate-0/Gate-1 bars above) was earned on a DIFFERENT task
(continuous rank recovery, not discrete composition-and-next-token) and
does **not** transfer by fiat, for either citation — Rev 1's own overclaim
on this exact point (§A1 M1's finding) is not repeated by adding a second
citation. Task 2 needs its OWN Gate-0 calibration on its OWN construction;
the bridge cell (§9, revised, §R1(d)/§R2(b) fixes M3/M7) is the mechanism
that earns it, gating Task 2's Phase 1 launch independently of the main
K-axis ortho-write verdict.

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
length-generalization brittleness (§R2(a)'s reframe of Axis A does not
touch this task's own claim), so it is reported as its own third,
secondary axis (§7, revised) regardless of which group construction
underlies it.

### 3.4 Citations — verified (2026-07-16, `research/
ncr_separation_grounding.md` + `research/ortho_write_grounding.md`,
coordinator-spot-checked; supersedes every prior `[TO-VERIFY]` tag)

| # | Citation | arXiv | Role in this design |
|---|---|---|---|
| C1 | Merrill & Sabharwal, "The Parallelism Tradeoff" | 2207.00729 | **DEMOTED to MOTIVATION only (§R2(a), fixes F2)** — log-precision transformers ⊆ TC⁰ is an ALL-`L` asymptotic fact, not a finite-`L` failure prediction; retained to explain why SGD is expected to find a bounded-depth (length-brittle) shortcut, never again as a "cannot" bar (§3.2) |
| C2 | Barrington, bounded-width branching programs / NC¹ | — (JCSS 38(1), 1989) | **DEMOTED to MOTIVATION only (§R2(a))** — same status as C1: the non-solvable-group word problem's NC¹-completeness motivates why a length-general algorithm is NOT the object SGD converges on, never a finite-`L` impossibility bar |
| C3 | Merrill, Petty & Sabharwal, "The Illusion of State in State-Space Models" | 2404.08819 | **DEMOTED to MOTIVATION only (§R2(a))** — makes the TC⁰-vs-NC¹ deduction explicit against S₅ (motivation for C1/C2's use here); also applies to diagonal SSMs (relevant to any SSM baseline variant) |
| C4 | Liu, Ash, Goel, Krishnamurthy, Zhang, "Transformers Learn Shortcuts to Automata" | 2210.10749 | **PROMOTED to the PRIMARY predicted mechanism for Task 2 (§R2(a), fixes F2 — supersedes Rev 1's "secondary/corroborating only")** — BOTH halves now load-bearing: the headline construction (an O(log T)-depth shortcut EXISTS and is learnable at every tested `L`, predicting in-distribution success) AND the brittleness finding (that shortcut fails OOD, predicting the length-generalization collapse Axis A now measures) — §3.2 |
| C5a | Grazzi, Siems, Zela, Franke, Hutter, Pontil, "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues" | 2411.12537 | **PINNED (§R1(f), M5)** as §4.4's sequential-rollout baseline ARCHITECTURE — extended-eigenvalue DeltaNet (`β∈(0,2)`), scoped to Axis B/Task 1 only (§1, §A1 F1) — and grounds §2.2's DeltaNet-β-range caveat |
| C5b | Siems, Carstensen, Zela, Hutter, Pontil, Grazzi, "DeltaProduct" | 2502.10297 | Alternative sequential-rollout candidate, NOT pinned (§4.4 justifies choosing C5a's extended-β DeltaNet instead, cost grounds) |
| C6 | Peng et al., "RWKV-7 'Goose'" | 2503.14456 | Alternative sequential-rollout candidate, NOT pinned (§4.4); confirms O(h) sequential state-update is the published mechanism, not O(log h) query-time reads |
| C7 | Schlag, Munkhdalai & Schmidhuber, "Learning Associative Inference Using Fast Weight Memory" (FWM) | 2011.07831 | Closest prior art on in-context fast-weight writes for compositional inference (approximate, recursive reads, not matrix powers) |
| C8 | Guu, Miller & Liang, compositional path queries | 1506.01094 | SECONDARY empirical citation for BOTH Task 1 (§3.1) AND Task 2 (§3.2, §R2(a) — extended this revision) — composition error cascades with path length, corroborating the length-generalization-collapse prediction on both tasks, not a structural argument |
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
M7+M9 fix (R=3, `d_ncr`=33 — the SAME shared `d_ncr` as Task 1, NOT Rev
1's own `d_ncr,2=5`, which M7 proved unorthogonalizable and this design no
longer uses).** `state_bytes = R × d² × 4 = 3 × 33² × 4 = 3 × 1,089 × 4 =
13,068` bytes — **LARGER than Task 1's single-relation state (4,356
bytes) by exactly 3× (=R), not smaller** — a full reversal of Rev 1's own
M6-fix finding (which found 300 bytes, ~14.5× smaller, under the now-
retired `d_ncr,2=5` construction). `cap_length(M=1) = 13,068/73,728 ≈
0.177` tokens. Floor-clearing minimum: `M ≈ 20×73,728/13,068 ≈ 113`.
Re-derived grid: `M ∈ {128, 256, 512, 1024, 2048}` → `cap_length ≈ {22.7,
45.4, 90.7, 181.5, 363.0}` tokens — clears the floor throughout, and
(because `13,068 = 3×4,356` exactly) reproduces Case (i)'s own
`cap_length` values at exactly `1/3` the `M`, a clean coherence check
between the two re-derived grids that falls directly out of the M7/M9
fix's own arithmetic, not a coincidence.

**The finding, restated and REVERSED by the M7/M9 fix, disclosed plainly,
not softened.** Rev 1's own M6-fix headline — "NCR's total
inference-memory footprint is smaller by a FURTHER order of magnitude on
Task 2" — no longer holds: making Task 2's write full-rank at the SHARED
`d_ncr=33` (required to fix M7's rank-deficiency defect) makes its
3-operator bank exactly 3× LARGER than Task 1's own single-relation
state, not ~14.5× smaller. The qualitative finding survives at a smaller
margin: an affordable Transformer KV cache (`M` in the low hundreds) still
dwarfs Task 2's entire written-operator bank, just not as dramatically as
Rev 1 reported. This is disclosed as a direct, honest consequence of
closing M7 (a correctness fix, not a cosmetic one) — it means Task 3
(§7, revised) is either the program's most dramatic memory-asymmetry
result (if a real long-horizon requirement is demonstrated) or a
mismatched comparison (if the capped Transformer can trivially win by
holding the entire relevant span in cache at any affordable `M`) — the
pre-registered bands in §7 score both readings explicitly, unchanged in
structure by this recomputation, only in the numbers feeding it.

### 4.4 Sequential-rollout matrix-state baseline (MANDATORY; REVISED,
§R1(f) — fixes M5: architecture PINNED, scope NARROWED to Axis B/Task 1
only per §A1 F1; §R2(c) — fixes M8: the ≥10× bar Rev 1 replaced with an
unsatisfiable `R²` log-fit is RESTORED, on the correct object)

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
   fit, §R2(c) restates in the coordinator's own exact form).** NCR's read
   on Task 1 executes AT MOST `2·⌈log₂h⌉` sequential matrix operations
   (binary exponentiation by squaring: up to `⌈log₂h⌉` squaring rounds,
   each contributing up to 2 matmuls — the square and the conditional
   multiply-into-the-accumulator; `binexp_read`'s own instrumented step
   counter asserts the exact realized count, which can be below the upper
   bound for an `h` with few set bits); the rollout baseline's read
   executes exactly `h` sequential state updates (one recurrent step per
   depth unit, by the architecture's own definition — Grazzi et al.'s
   construction has no shortcut around this). This is an exact,
   hardware-independent, UNFAKEABLE integer count per query, asserted
   once via a build-time instrumentation check (tag every op on the
   critical read path with a monotonic counter, assert the counter total
   matches the formula) — a PASS/FAIL build gate, not a statistical claim,
   and the primary evidence because it does not depend on kernel-launch
   overhead, GPU generation, or batching.
2. **Wall-clock (SECONDARY, empirical confirmation) — RESTORED to a
   flat-vs-linear / ratio discriminator (§R2(c), fixes M8: Rev 1's own
   `R²` log-fit replacement is UNSATISFIABLE on this program's OWN prior
   measurement of the IDENTICAL `binexp_read` mechanism —
   `NOVEL_ARCH_WATERFALL.md` §7f: "bin-exp flat at ~1-3 ms from h=61 to
   h=2^20+5" — a FLAT series regressed on `log₂h` yields slope `b≈0`,
   `R²≈0`, failing `Model_log R²≥0.90` BY CONSTRUCTION; M8's diagnosis,
   verified against §7f's raw numbers, adopted without dispute).** The
   `Model_log`-vs-`Model_lin` OLS fit is DEMOTED to a reported-not-gating
   diagnostic (still computed and disclosed, per this design's own
   never-hide-a-measurement discipline, but no longer a WIN/FAIL gate).

   **The restored gating criterion** (mirroring round-1 M5's ORIGINAL,
   already-cleared shape at toy scale — `NOVEL_ARCH_WATERFALL.md`
   §7e/§7f: "AXIS B: WIN — bin-exp 20.9× faster... at h=1021 (bar ≥10×)",
   VERIFIED — Rev 1's `R²`-fit was a replacement M8 shows should never
   have happened): **(i)** NCR's own wall-clock series is FLAT — a fitted
   `Model_lin` slope `d` whose 95% CI includes 0 — the PREDICTED NCR
   signature (kernel-launch-bound, not compute-bound, exactly as §7f's
   prior measurement of this identical mechanism shows) — **AND (ii)** the
   rollout baseline's own series is LINEAR — `Model_lin R²≥0.99`, slope
   CI excludes 0 — **AND (iii)** the measured wall-clock RATIO
   (rollout/NCR) at the LARGEST feasible tested `h` is **≥10×** — the
   exact bar this program already cleared at toy scale (20.9× at h=1021,
   ≈13,155–25,064× at h=2^20+5, `NOVEL_ARCH_WATERFALL.md` §7e/§7f-erratum,
   VERIFIED).

   **Ladder, and where the ≥10× bar becomes readable (M8's
   under-specification fix — the depth at which overhead no longer masks
   the gap, derived from §7f's own numbers, not assumed).** `h ∈ {61, 200,
   1000, 5000, 20000}` (PROJECTED, subject to Phase-0a feasibility — if
   the rollout baseline's per-step cost makes `h=20000` impractically
   slow, the largest FEASIBLE point measured is used instead). Using the
   nearest measured analog in this repo for an `O(h)` sequential-rollout
   arm's OWN per-step cost (`NOVEL_ARCH_WATERFALL.md` §7f's toy "naive
   loop" baseline: 64.4 ms at h=1021 ⇒ ≈0.063 ms/step; 61.3 s at h=2^20+5
   ⇒ ≈0.058 ms/step — a consistent ≈0.06 ms/sequential-step rate,
   disclosed as an ANALOG, not this design's own extended-β-DeltaNet
   rate, which remains PRICE-UNKNOWN, §6.3 item 3a), the rollout
   baseline's cumulative wall-clock first exceeds `10×` NCR's own
   measured flat floor (`1–3` ms, same §7f citation) at `h ≈
   10ms/0.06ms ≈ 167` to `30ms/0.06ms ≈ 500` sequential steps. So `h=61`
   is BELOW this mask-clearance zone (reported only, not where the gate
   is read) and `h=200` sits at its lower, borderline edge; `h=1000`
   clears it unambiguously (predicted rollout cost `≈1000×0.06=60`
   ms `≫30` ms). **The gating `≥10×` ratio (iii, above) is therefore read
   at `h≥1000`, preferring the largest feasible tested point** —
   consistent with how this repo's own toy measurement reported a GROWING
   ratio with `h` (20.9× at 1021 → ~13,000–25,000× at 2^20+5) rather than
   a single fixed-`h` snapshot.

   **Measurement protocol, pinned (M8's under-specification fix — none of
   this was pinned in Rev 1).** `B=32` standardized probes per `h`
   (matching this program's own existing Axis-B timing convention,
   `NOVEL_ARCH_WATERFALL.md` line 1292, "measured, standardized B=32
   probes"); **7 repeats per `(arm, h)` cell, first repeat discarded as a
   CUDA/kernel warmup** (standard GPU-benchmarking practice, previously
   unstated); **statistic = MEDIAN of the remaining 6 repeats** (robust to
   scheduler/contention outliers, not the mean); **noise model:** report
   the median absolute deviation alongside the median, and treat any
   `(arm,h)` cell whose MAD exceeds 20% of its own median as
   under-powered, requiring additional repeats before that point enters
   the ratio/fit — a concrete, checkable rule where Rev 1 left the
   statistic, repeat count, and noise handling entirely unspecified.

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

**Phase 0b — BRIDGE CELL (NEW, §R1(d), fixes M3; §R2(b), fixes M7's
follow-on): NS-polar orthogonal write trained on S₅-GENERATOR writes, NOW
FULL-RANK at the shared `d_ncr` (§3.2's M7 fix), gating Task 2's Phase 1
arm specifically.** Why this is needed, restated: the main ortho-write
wave (`NCR_ORTHO_WRITE.md`, blind-run in flight) validates cyclic K-cycle
writes (Part A) and RANDOM-ORTHOGONAL bank writes (Part B, R=4
independent Hamiltonian K-cycles) — **NEITHER cell ever writes a
non-solvable-group generator** (§A1 M3's finding). A WIN on that verdict
says nothing about whether the model can LEARN to write S₅'s `{t,c,c⁻¹}`
generators (§3.2) as orthogonally-conditioned, FULL-RANK `d_ncr×d_ncr`
operators (`ρ_{S₅}(g)⊕I_{d_ncr−4}`, §3.2's M7 fix) and compose them at
depth — this bridge cell is the SEPARATE, dedicated gate for exactly that
question, run BEFORE any Phase-1 GPU-h is spent on Task 2.

**Pricing (PROJECTED from Part B's own measured rate, §6.1 — an EVEN
TIGHTER transfer than Rev 1's, since Task 2's `d` is no longer smaller
than Part B's own `d=33` at all — it is IDENTICAL).** Construction: `R=3`
(S₅'s `{t,c,c⁻¹}`, not Part B's R=4), `d=d_ncr=33` (SAME as Part B's own
`d`, NOT Rev 1's retired `d=5`), **`n=3` seeds** (RAISED from Rev 1/Rev
2's `n=2`, §R2.1(c) — closes F3-3/m4, UPGRADED from disclosed-minor to
REQUIRED by §A3-ADJUDICATION: a gate that can DROP an entire primary axis
on a median statistic does not run at n=2, matching this program's own
n≥3 seed-variance norm; still one seed short of Part B's own n=4, an
additional un-costed margin of conservatism), **1× budget = 80,000
steps** (not Part B's 4× = 320,000 — a bridge/gate cell needs Gate-0 plus
one modest depth checkpoint, not the full realistic-depth frontier sweep
Part B itself runs). Cost: `80,000/320,000 × 4.24 GPU-h/cell = 1.06
GPU-h/cell (per-seed rate) × 3 seeds = 3.18 GPU-h (1×); 2× contingency ≈
6.36 GPU-h` — **`+1.06` GPU-h (1×) / `+2.12` GPU-h (2×) over Rev 1/Rev 2's
own `n=2` projection (`2.12`/`4.24` GPU-h), propagated through the §6.2
grand total below (§R2.1(c))** (R=3 vs. Part B's R=4 remains, if anything,
an additional un-costed margin of conservatism, not priced as savings, per
this design's own discipline against inventing unmeasured discounts).

**Orthogonality tolerance, stated explicitly (M7's fix — this check was
STRUCTURALLY UNPASSABLE under Rev 1's zero-padded object, §A2 M7's
arithmetic; it is well-posed now).** Mechanistic corroboration bar reused
verbatim from Part A/B's own WIN convention (`NCR_ORTHO_WRITE.md` §4:
"departure-from-normality ≤ 0.02, cond# ≤ ~2, min|λ|/c* ≥ 0.9"): the
trained bridge-cell write's departure-from-normality metric must clear
the SAME ≤0.02 bar Part A already requires for its own WIN — a check that
is now MEANINGFUL (the target has every singular value at exactly 1,
§3.2) rather than structurally forced to its worst possible value
regardless of training (Rev 1's `‖QᵀQ−I‖_F=1` floor, retired).

**Gate (mirrors `NCR_ORTHO_WRITE.md` §4 Part B's own band structure, at
reduced scope — Gate-0 plus ONE depth checkpoint, not the full L-ladder;
checkpoint RE-ANCHORED from Rev 1's `L=8` to `L=20`, §R2(a): under the
length-generalization reframe, `L_train=8` is now the boundary of the
MAIN Phase-1 design, §3.2, so the bridge cell's own single checkpoint must
sit clearly OOD relative to it to preview the property Phase 1 actually
needs — `L=20` is an existing, already-audited Part-B ladder rung, §4b,
so no new eval infrastructure is needed):** Gate-0 (median rec@0.9 ≥0.9 at
L∈{1,2,3}) AND **WIN**: ortho S₅-generator-write median rec@0.9 at
**L=20** (2.5× the new `L_train=8`) ≥0.9 AND free-write (unconstrained)
baseline ≤0.35 at L=20 (the chance-adjacent FAIL bar, §3.2's M10 fix —
replaces Rev 1's `<0.5`) AND the orthogonality corroboration bar above
clears. **PARTIAL**: Gate-0 clears AND L=20 recovery ∈(0.35,0.9) AND
(ortho rec@L=20 − free-write rec@L=20) ≥0.2 (the PINNED MARGIN, §R2.1(b),
mirroring the WIN row's own free-write-gap convention above — closes
m7/F3-2's band-overlap: the old bands had no delta threshold, so a
borderline result, e.g. ortho=0.60/free=0.58, satisfied BOTH PARTIAL's
numeric range and NULL's "no gain" description at once). **NULL**:
Gate-0 clears AND the WIN and PARTIAL conditions above are NOT both met —
i.e. L=20 recovery ≤0.35, OR L=20 recovery ∈(0.35,0.9) but the margin
(ortho − free) <0.2 — the exact band-overlap case is now resolved
unambiguously (ortho=0.60/free=0.58 ⇒ recovery nominally in-band but
margin=0.02≪0.2 ⇒ NULL). **FAIL**: Gate-0 itself fails (the
constraint blocks trainability on this differently-shaped object — though
M7's fix removes the STRUCTURAL reason this would happen, a genuine
optimization-difficulty FAIL remains possible and is still scored).
Consequence for Task 2 — see §9, revised (§R1(c)/§R2(a), M2/F2's fixes):
WIN/PARTIAL license Phase 1's Task-2 arm (PARTIAL re-anchors the OOD
ladder's claimed floor downward, mirroring the main verdict's own PARTIAL
branch); NULL/FAIL DROP Task 2's Axis-A arm for Stage 1, disclosed
explicitly, collapsing the overall program verdict's ceiling to PARTIAL
(§7, revised) regardless of how Axis B reads.

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

**Cross-task interference criterion (NEW, §R2.1(e), wires §A3
CHECK 3(i)'s carried-forward gap — an EXACT threshold, not a vibes call).**
Because training is task-suite-shared (above), the shared `d_ncr` head
sees BOTH Task 1's K-cycle writes and Task 2's `ρ_{S₅}(g)⊕I` writes in one
curriculum — M7+M9 resolved that this is architecturally ONE valid
construction, but not whether joint training DEGRADES either task
relative to training it alone (§R2's own open item (5)). **For EACH task
family independently: Phase-2's shared-curriculum per-task accuracy
(the SAME recovery metric Gate-0 already reads, evaluated per-task on the
co-trained checkpoint) must be ≥ (that task's OWN Phase-1 single-task
calibration accuracy, §6.2 Phase 1 above − 0.05 absolute).** A breach on
EITHER task triggers **DIAGNOSE-FIRST**: HOLD Phase 3 for that scale, and
adjudicate using the single-family ablation arms Phase 1 already ran
(Task-1-only, Task-2-only calibration cells, §6.2 Phase 1 above — already
run, NO new GPU-h) — compare the co-trained Phase-2 checkpoint's per-task
accuracy against the corresponding ISOLATED Phase-1 arm's own accuracy to
separate genuine cross-task interference (the shared head degrades under
joint training) from an unrelated per-task regression (e.g. a
scale-specific optimization issue), before spending any further budget at
that scale. **No silent pass**: a breach without a filed DIAGNOSE-FIRST
adjudication blocks Phase 3 authorization for that scale by construction
(mirrors §8 item 4's existing per-task Gate-0 discipline, extended here to
a cross-task comparison Gate-0 alone does not make). Full statement and
consequence restated as its own numbered risk, §8 item 8.

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
bridge cell added, NOW at `n=3` seeds §R2.1(c)): ≈2 + 11.9 + 6.36 + 21.52
+ 215.3 + 226.7 ≈ 484 GPU-h** (vs. the pre-Rev-1 draft's own
excludes-the-rollout-arm total of 462 GPU-h — the increase is Phase 0a's
now-included rollout probe (+4.76), the bridge cell + doubled Phase 1
(+15), and Rev 2.1's own bridge-cell `n=2→n=3` seed raise (+2.12 at 2×,
§R2.1(c), ledger delta: 482→484 GPU-h). This is a Stage-1
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
1×, +120 GPU-h at 2× contingency**, revising the grand total to **≈604
GPU-h at 2× contingency** (484+120, §R2.1(c)'s +2.12 propagated). This
revision is presented as an
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
5. **The bridge cell's own rate (§6.2 Phase 0b, §R1(d)/§R2(b)).** Priced
   as a PROJECTION from `NCR_ORTHO_WRITE.md` Part B's measured rate
   (§6.1), scaled by step/seed budget only — the `R=3/d=33` construction
   itself has never run (only Part B's own `R=4/d=33` has), so its ACTUAL
   per-step cost (and whether it trains at all) is unmeasured until the
   cell itself executes, EVEN THOUGH `d` now matches Part B's own `d`
   exactly (a tighter transfer than Rev 1's `R=3/d=5` projection, not a
   measurement). Retired by the bridge cell's own completion, which is
   also its own gate readout (§6.2).

### 6.4 SATURATION plan (PI directive, 2026-07-16 — sizing so each H100 runs at high SM utilization)

**The rejected pattern, stated precisely.** The toy-scale NCR/ortho-write
cells (K≤32, d_ncr≤33, h_ncr=64, ~175–185K params, §6.1's last row) were
run ONE PROCESS PER GPU, drawing on the order of ~50% SM and low
single-digit GB of VRAM — a tiny model's forward/backward pass simply
does not generate enough parallel work to saturate an H100 regardless of
how it is scheduled solo. This design explicitly does NOT repeat that
pattern for either its main training cells or its calibration cells.
(§R2.1(d) closes m6/F3-4: the "Main 98M/392M cells… Two coupled levers"
block that used to appear here a second time, near-verbatim, is de-duped
— see that single surviving copy below, just before "Main 98M/392M cells
are NOT packed.")

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
(REVISED, §R1(a) — the table and overall-verdict rule were first rewritten
around the F1-fixed two-family conjunction, §1; §R2(a)/(c)/(d) — Axis A's
row is REWRITTEN AGAIN around the length-generalization reframe (fixes
F2), Axis B's row restores the M8-fixed dependency-count + ratio
criterion, and Task 2's bands are re-derived against chance (M10))

Reuses the `recovered_frac@0.9` bar and the HOLD(≥0.9)/DEGRADED((0.5,0.9))/
FAIL(≤0.5) band convention (`NOVEL_ARCH_WATERFALL.md` §3.2a) throughout,
extended to per-task, per-axis outcomes; **Task 2 specifically uses the
chance-re-derived HOLD(≥0.9)/DEGRADED((0.35,0.9))/FAIL(≤0.35) bands
instead (§3.2's M10 fix — the 5-way answer's chance floor sits at ≈0.25,
so the old FAIL≤0.5 bar was only 2× chance).** **Gate-0 precondition (both
arms, every cell, per §6.2's Phase-1 convention): in-distribution
recovery ≥0.9 at train-support depth AND val-loss inside the standing
`k=2·s_ref` gate — a cell failing Gate-0 is DEAD and is not scored on any
axis below.**

**Framing correction (§R1(a), then §R2(a) — replaces the pre-Rev-1 draft's
unsatisfiable framing, then replaces Rev 1's own unearnable structural
framing): the two PRIMARY axes are Axis A = Task 2 (MECHANISTIC
LENGTH-GENERALIZATION, read at Θ(L), no speed claim, no finite-`L`
impossibility claim) and Axis B = Task 1 (query-time complexity, O(log h)
vs. Θ(h), no hardness claim) — DISJOINT query families, per §1's
conjunction and §A1 F1's group-theory result (unaffected by the Axis-A
reframe). Task 3 is a SECONDARY/corroborating axis on a third, orthogonal
property (memory bytes), never sufficient alone for the overall program
WIN.**

| Axis | WIN | PARTIAL | NULL |
|---|---|---|---|
| **Axis A = Task 2 (PRIMARY — mechanistic length-generalization, §3.2, read at Θ(L))** | NCR HOLDs (≥0.9) at EVERY `L_test∈{12,16,20,24,32,40}` rung (both held-out-entity and held-out-depth strata) AND the param-matched Transformer baseline shows the SHORTCUT-BRITTLENESS SIGNATURE (§1's mechanism instrument): in-distribution success — median over `L∈{5,8}` — ≥0.9 AND OOD accuracy at the largest tested `L=40` ≤0.35 (the chance-adjacent FAIL bar, §3.2's M10 fix). **The in-distribution-success conjunct is what makes this shortcut-BRITTLENESS, not general task failure** — a baseline that never succeeds even in-distribution would say nothing about shortcuts vs. algorithms (§1(a)(iv), the task's own binding requirement). **Gated precondition (§9, revised): only reachable if the bridge cell (§6.2 Phase 0b) reads WIN or PARTIAL** — a NULL/FAIL bridge cell drops this axis to NULL by construction, disclosed, not silently absorbed (§9) | NCR HOLDs at SOME but not all OOD `L_test` rungs (e.g. holds through L=20 but degrades by L=40 — a genuine, partial length-generalization result); OR the baseline's in-distribution-success + OOD-collapse signature is present but the OOD band lands in DEGRADED (0.35,0.9) rather than clean FAIL (a real but softer brittleness signature); OR the bridge cell read PARTIAL and the OOD floor was re-anchored downward per §9, with the shallower ladder otherwise meeting the WIN bar | The baseline does NOT show in-distribution success (median over `L∈{5,8}` <0.9 — Task 2 is too hard to learn at all, uninformative about shortcuts vs. algorithms, §8 item 5) OR the baseline's OOD accuracy does NOT collapse relative to its own in-distribution accuracy (the baseline ALSO length-generalizes — the single most informative possible negative result for Axis A, §8 item 5) OR NCR itself fails to HOLD at some OOD rung (both arms behave alike, no separation demonstrated) **OR the bridge cell reads NULL/FAIL (§9) — Task 2 is DROPPED for Stage 1, and this axis is NULL by construction regardless of any LM-scale data** |
| **Axis B = Task 1 (PRIMARY — query-time complexity vs. §4.4's pinned extended-β-DeltaNet rollout baseline, O(log h) vs. Θ(h), no hardness claim)** | (i) The dependency-chain-length fact holds by construction (`2·⌈log₂h⌉` for NCR vs. `h` for the rollout, §4.4's build-time instrumentation assertion — a PASS/FAIL gate, always checked first) AND (ii) §4.4's RESTORED flat-vs-linear/ratio criterion clears (§R2(c), fixes M8): NCR's own wall-clock series is FLAT (`Model_lin` slope 95% CI includes 0) AND the rollout baseline's own series is LINEAR (`Model_lin R²≥0.99`, slope CI excludes 0) AND the measured ratio (rollout/NCR) at the largest feasible tested `h` (`h≥1000`, §4.4) is **≥10×** — the bar this program already cleared at toy scale (20.9× at h=1021) — AND (iii) in-distribution accuracy is within the standing ±15% band between the two arms at matched depth (an accuracy SANITY check, not the primary readout — §4.4). The `Model_log`-vs-`Model_lin` `R²` fit (Rev 1's own replaced criterion, M8's fix) is REPORTED alongside, never gating | The dependency-chain fact holds (i) but the ratio criterion (ii) is inconclusive (e.g. NCR's own series is not cleanly flat, or the rollout's own series is not cleanly linear, or the measured ratio at the largest feasible `h` is `<10×`) — reported as a real but statistically softer separation, not hidden | (i) itself fails (a build defect — NCR's read is not actually `O(log h)` as implemented) OR NCR's own wall-clock is not measurably flat/sub-linear in `h` at all — flags a broken implementation, triggers diagnose-first, per `NOVEL_ARCH_WATERFALL.md`'s own convention for this failure shape |
| **Task 3 (secondary — "constant-memory minds," §3.3)** | NCR holds acc≥0.9 at the largest tested horizon (8×T_bind) AND the KV-cache-capped Transformer (re-derived `M` grid, §4.3, revised) FAILs at that horizon — reproduces the M* "capping never rescues" finding (`STATE.md` §1.41) in a real-text setting | NCR holds where the capped Transformer fails, but the UNCAPPED Transformer also holds — NCR matches, not beats, an unconstrained baseline at fixed memory (itself a genuine, disclosed data point) | NCR itself fails past some horizon shorter than 8×T_bind |
| **Val-loss gate (all tasks, mandatory)** | Every arm's val loss stays inside `arm_off`'s own `mean + 2·s_ref` band (`FROZEN_BIAS_LM_DESIGN.md` §7.2 convention) | n/a — pass/fail gate, not graded | A cell whose NCR head measurably HURTS ordinary language modeling is reported plainly, not excused, even if a primary axis above WINs |

**Mechanism-instrument reporting (§1(a)(ii)'s requirement, not itself
gating — required alongside every Axis-A verdict above).** Every Axis-A
readout is accompanied by the FULL per-`L` accuracy curve for both arms
(§3.5's existing mandatory stratification — no new measurement), read for
SHAPE (a shortcut signature is high-then-collapsing; an algorithm
signature is flat throughout) — this is what distinguishes a WIN band
crossed for the RIGHT mechanistic reason from one crossed by coincidence,
and is reported even when the accuracy bands alone already resolve the
verdict. The optional attention-map diagnostic (§1) is included when
available, at near-zero marginal cost.

**Overall program verdict (cross-task rule, exhaustive, re-anchored first
to the F1-fixed two-family conjunction, §1, then to the F2-fixed
length-generalization reframe of Axis A specifically).** **WIN** requires
BOTH primary axes at WIN — Axis A/Task 2 (mechanistic length-
generalization) AND Axis B/Task 1 (query-time complexity) — AND the
val-loss gate passing everywhere; Task 3 is reported alongside as a
corroborating/secondary claim, never substitutable for either primary
axis. **PARTIAL** = exactly one primary axis WINs (length-generalization
without the complexity separation, or vice versa) — still publishable,
reported as "one leg of the two-family claim landed, on the family it is
earnable on," not spun as the full flagship result; **this is also the
automatic ceiling if the bridge cell (§9) reads NULL/FAIL, since Axis A
is then NULL by construction and only Axis B remains contestable.**
**NULL** = neither primary axis clears WIN or PARTIAL, or Gate-0 fails in
≥50% of cells (a trainability failure at LM scale, distinct from and
reported separately from a capability failure). **Removed by §R1(a): the
pre-Rev-1 draft's own framing implicitly allowed a version of "WIN" where
Task 2 supplied BOTH the structural failure AND the O(log h) speed claim
on the same family — §A1 F1 proved this unsatisfiable; no band above can
be cleared that way, by construction (Task 2's own read cost is Θ(L),
stated in its own WIN cell, never O(log L)).** **Removed by §R2(a): Rev
1's own "cannot, not merely does not" reading of Axis A's WIN band — F2
proved this unearnable at any tested depth; Axis A's WIN band above is
now an empirical, falsifiable, in-distribution-success-plus-OOD-collapse
claim, never a finite-`L` impossibility, and the mechanism-instrument
reporting requirement above exists specifically so a WIN cannot be
misread as the retired structural claim.**

---

## §8 RISKS & KILL CRITERIA — what a week-1 result kills before week 4
(REVISED, §R1(h) — fixes m1/m2/m3; items 1/5/6 re-scoped to the F1-fixed
two-family conjunction; item 7 added for the bridge cell, §M2/M3; §R2 —
items 5/6/7 updated again for the F2 length-generalization reframe and
the M7/M8/M10 fixes; §R2.1 — item 8 added, wiring the cross-task
interference criterion, §A3-ADJUDICATION item (e))

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
   (§R1(h), m1/m2 fix — was "cleanly," now pinned; §R2(e) closes round-2's
   m2 partial: the ENTIRE-model-in-fp32 reference was infeasible, `fla`'s
   production kernel rejects fp32 inputs outright,
   `DELTANET_REALDATA_DESIGN.md` §4.3 VERIFIED).** Concrete smoke item
   (m1's fix): the NCR head casts backbone hidden states to fp32 at its
   own input boundary (`.float()`, a standard differentiable upcast),
   runs its full write/read pipeline in fp32, casts its output back to
   bf16 before injection into the residual stream. **Pass criterion
   (m2's original fix, RE-SCOPED this revision to fix its own infeasible
   reference — chosen over the alternative of accepting the naive-vs-
   chunked ~10–20% Jacobian gap `DELTANET_REALDATA_DESIGN.md` §4.3 also
   documents, which would swamp a `<1×10⁻²` tolerance if the whole model
   were included):** a gradient cross-check SCOPED TO THE NCR HEAD ONLY —
   this cast pipeline (fed a batch of fixed, detached backbone hidden
   states, exactly as it consumes them at build time) vs. a standalone
   fp32 instantiation of the SAME head fed the IDENTICAL fixed inputs
   (bypassing the backbone entirely, so the backbone's own bf16-kernel
   restriction never enters this check) — at a bf16-appropriate relative
   tolerance **`<1×10⁻²`**, both forward outputs and parameter/hidden-
   state gradients after `.backward()`. This tests exactly the boundary
   at risk (the cast pipeline) without conflating it with the backbone's
   own separate, already-documented naive-vs-chunked discrepancy. Run
   ONCE at Phase 0 (14M scale), before any further spend, logged PASS/FAIL
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
5. **The param-matched Transformer baseline (§4.1) does NOT show the
   predicted shortcut-brittleness signature at Task 2's OOD ladder (§R2(a)
   fixes F2 — supersedes Rev 1's "also HOLDs at `L*`" framing, which was
   itself downstream of the retired structural-impossibility claim).**
   Two distinct sub-cases, both disclosed separately: **(a)** the baseline
   succeeds in-distribution (as C4 predicts) AND ALSO HOLDs at
   `L_test≫L_train` — i.e. it length-generalizes as well as NCR does, a
   real possibility even though C1–C3's motivating argument is
   theoretically airtight, since a trained model could exploit
   natural-language redundancy the synthetic harness's construction
   didn't allow, or a bounded-depth shortcut could simply generalize
   better across THIS particular length range than C4's own brittleness
   precedent predicts; **(b)** the baseline fails to show in-distribution
   success at all (§7's NULL band's first disjunct) — Task 2 is simply
   too hard to LEARN at this scale, a trainability finding (diagnosed via
   item 4 above), not a shortcut-vs-algorithm finding. **(a) is the single
   most informative possible negative result for Axis A** — reported
   plainly (`CLAUDE.md`: "negative results are data. Don't spin.") — does
   not kill the program (Axis B's complexity claim, item 6 below, is on a
   disjoint query family and independent of this one, §A1 F1), but caps
   the paper's Axis-A claim at "the predicted shortcut-brittleness
   mechanism did not manifest at this scale/length range," a materially
   weaker headline than §1's Axis A hypothesis, and would redirect Stage-2
   effort toward understanding WHY (larger `L_train`/`L_test` gap, larger
   K₂, larger group) before any further scale-up.
6. **Axis B (§4.4) shows no wall-clock/dependency-chain separation on
   Task 1** — the dependency-chain-length build assertion itself fails
   (a genuine implementation defect: NCR's read is not actually
   `O(log h)` as built), or the restored flat-vs-linear/ratio criterion
   (§4.4/§7, §R2(c)'s M8 fix) does not clear on either arm's own series
   (e.g. NCR's own series is not cleanly flat, or the rollout's own series
   is not cleanly linear, or the measured ratio at the largest feasible
   `h` is `<10×` — or both are dominated by a fixed per-call overhead at
   the tested depths, or the rollout baseline's own implementation is
   unexpectedly parallel-friendly at this scale). Per
   `NOVEL_ARCH_WATERFALL.md` §3.2's own convention for this exact failure
   shape, this is flagged as an IMPLEMENTATION defect and triggers
   diagnose-first, not an immediate capability conclusion — but if
   diagnosis confirms the measurement is sound, it kills Axis B and,
   combined with item 5(a), would collapse the overall verdict to NULL
   (§7) regardless of how Task 3 reads.
7. **The bridge cell (§6.2 Phase 0b, §9, revised) reads NULL or FAIL
   (§R1(h) added this item; §R2(b) updates the object it gates — S₅
   generators embedded FULL-RANK at the shared `d_ncr`, checkpoint
   re-anchored to `L=20`, M7's fix).** DROPS Task 2/Axis A for Stage 1 —
   the length-generalization claim is not earned at any LM scale this
   wave, disclosed, not silently absorbed (§7's overall verdict is
   automatically capped at PARTIAL, contingent only on Axis B). Does not
   kill the program (Axis B's own claim on Task 1 is entirely independent
   of the bridge cell's outcome) but is the single cheapest possible way
   (§6.2: ≈3.18 GPU-h, 1×, at the `n=3` seeds pinned §R2.1(c)) to learn
   this BEFORE any Phase-1 Task-2 GPU-h — the entire reason this gate
   exists ahead of the more expensive
   LM-embedded calibration. **M7's fix removes the STRUCTURAL reason this
   cell was guaranteed to be at least partially uninformative (Rev 1's
   zero-padded object could never pass the orthogonality corroboration
   check regardless of training) — a FAIL verdict now genuinely means
   "this construction doesn't train," not "this construction can't be
   orthogonal by arithmetic."**
8. **Cross-task interference: Phase-2's shared-curriculum per-task
   accuracy falls below (that task's OWN Phase-1 single-task calibration
   accuracy − 0.05 absolute), for EITHER task family (NEW, §R2.1(e),
   §A3-ADJUDICATION item (e) — closes §A3 CHECK 3(i)'s carried-forward
   gap: a per-arm Gate-0 floor already existed, but no comparison AGAINST
   the isolated single-task calibration did, so interference that
   degraded one task WITHOUT dropping it below the absolute 0.9 Gate-0 bar
   would previously have passed every gate silently).** Full criterion
   restated at its point of use, §6.2 Phase 2. Triggers **DIAGNOSE-FIRST**:
   HOLD Phase 3 for that scale; adjudicate using the single-family
   ablation arms Phase 1 already provides (§6.2 Phase 1's separate
   Task-1-only / Task-2-only calibration cells, already run, NO new
   GPU-h) to determine whether the degradation is genuine cross-task
   interference (the shared `d_ncr` head, trained on two write CONTENTS,
   degrades either task relative to training it alone) or an unrelated
   regression, before any further spend on that task/scale. Does not kill
   the program — a confirmed-interference finding motivates a future
   two-disjoint-head ablation (out of this Stage-1 build's scope, named
   as a Stage-2 follow-on), not an immediate program kill. **No silent
   pass**: a breach without a filed DIAGNOSE-FIRST adjudication blocks
   Phase 3 authorization for that scale.

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
  carry Axis A's (mechanistic length-generalization) claim; that swap is
  retired, not repaired, because it is no longer needed — Task 2 never
  depended on this K-axis
  to begin with).

### 9.2 GATE 2 — the bridge cell verdict (NEW, §R1(d); §R2(b) updates the
object it gates — governs Axis A/Task 2 exclusively; independent of GATE
1)

`NCR_ORTHO_WRITE.md`'s existing blind run (GATE 1, above) trains cyclic
K-cycle writes (Part A) and RANDOM-ORTHOGONAL bank writes (Part B, R=4
independent K-cycles) — **never an S₅-generator write** (§A1 M3's
finding). The bridge cell (§6.2 Phase 0b) is the dedicated, independent
gate for exactly that untested object — now FULL-RANK at the shared
`d_ncr` (§3.2's M7 fix, §R2(b)) — and its own verdict — NOT GATE 1's —
determines whether Task 2/Axis A proceeds, at what OOD floor, or at all:

- **If BRIDGE CELL WIN** (median rec@0.9 at L=20 ≥0.9, free-write baseline
  ≤0.35 at L=20, orthogonality corroboration clears, §6.2): Task 2's
  Phase 1 arm (§6.2) proceeds with **K₂=5, `d_ncr`=33 (shared with Task
  1), R=3** (S₅'s `{t,c,c⁻¹}`, embedded `ρ_{S₅}(g)⊕I_{29}`) at the full
  L-train/L-test split `L_train∈{1,…,8}` / `L_test∈{12,16,20,24,32,40}`
  (§3.2) — exactly the configuration priced in §3.2/§6 above. Axis A (§7)
  is fully contestable at WIN.
- **If BRIDGE CELL PARTIAL** (Gate-0 clears AND L=20 recovery ∈(0.35,0.9)
  AND (ortho − free-write) ≥0.2, the PINNED MARGIN §6.2 Phase 0b/§R2.1(b)):
  Task 2 proceeds but the OOD claimed floor is re-anchored DOWNWARD to
  whatever depth the bridge cell's own (limited) ladder supports — a
  build agent must extend the bridge cell's own eval readout to at least
  one lower OOD checkpoint (e.g. L=12) before pinning a number, not
  assumed here; §7's Axis-A WIN band is re-anchored accordingly, and
  Axis A is EXPLICITLY marked REDUCED in any results write-up (§R1(c)'s
  direct fix of M2's "preserve both axes or mark the axis REDUCED"
  instruction) — still contestable, at a weaker headline OOD floor.
- **If BRIDGE CELL NULL or FAIL** (Gate-0 clears but the WIN/PARTIAL
  margin conditions above are NOT both met — L=20 recovery ≤0.35, or
  recovery ∈(0.35,0.9) but (ortho − free) <0.2 (the pinned margin now
  makes this boundary unambiguous, §6.2 Phase 0b/§R2.1(b) — closes m7/
  F3-2's band-overlap), or Gate-0 itself fails on the S₅-generator
  object): Task 2/Axis A is **DROPPED for Stage 1** — disclosed
  explicitly, not silently absorbed.
  No Phase 1+ GPU-h is spent on Task 2's own arm (§6.2's Phase 1 Task-2
  subtotal, 5.38 GPU-h at 1×, is NOT spent); §7's overall program verdict
  is capped at PARTIAL regardless of how GATE 1/Axis B reads (§7's
  overall-verdict rule, revised). **If NULL (trains, no depth gain):**
  flags a genuine "does the write-conditioning fix generalize beyond
  cyclic/random-orthogonal objects" open question for a future waterfall
  pass, not assumed resolved either way — M7's fix already rules out one
  candidate EXPLANATION for a NULL (rank-deficiency), so a NULL under the
  now-fixed construction is more informative than it would have been
  under Rev 1's own broken object. **If FAIL (does not train at all):**
  flags that S₅'s specific generator structure (a transposition +
  5-cycle, order-5 rotation) may need its own parametrization variant
  (§2 of `NCR_ORTHO_WRITE.md`'s own Cayley/matrix-exponential fallbacks,
  named there for exactly this kind of failure) before any retry — not
  more budget on the same parametrization; a FAIL here is now a genuine
  optimization-difficulty finding, not an arithmetic inevitability (§8
  item 7's own note).

**Both gates are independent and may resolve differently** (e.g. GATE 1
WIN + GATE 2 FAIL is a fully coherent outcome under this design: Task 1/
Axis B proceeds at K=32 while Task 2/Axis A is dropped) — this is the
direct structural consequence of M1's decoupling fix, and a build agent
must not assume the two verdicts move together.

---

**Provenance.** This document was a NEW Stage-1 design (opened
2026-07-16), attacked twice (§A1, BUILD-BLOCKED, §A1-ADJUDICATION → Rev 1,
§R1; §A2, BUILD-BLOCKED, §A2-ADJUDICATION → Rev 2, §R2, this revision) —
§1–§9 revised in place first per §A1-ADJUDICATION's binding items (a)–(h),
then again per §A2-ADJUDICATION's binding items (a)–(e); §A1/
§A1-ADJUDICATION/§A2/§A2-ADJUDICATION all left untouched as historical
record, below. `research/ncr_separation_grounding.md` and
`research/ortho_write_grounding.md` landed 2026-07-16 (coordinator-spot-
checked) and remain incorporated throughout (§1, §2.1–§2.3, §3.1–§3.4,
§4.4, §7, §9) — every citation is VERIFIED or explicitly flagged, no
`[TO-VERIFY]` tags remain. **Items now RESOLVED, not merely flagged, as of
Rev 2:** F2 (the "cannot, not merely does not" claim is unearnable at any
tested depth) is fixed by reframing Axis A as mechanistic
length-generalization (§1, §3.2, §7); M7 (Task 2's write was rank-
deficient by construction) is fixed by adopting `CAPABILITY_SEPARATION_
DESIGN.md` §1.4's actual full-rank `ρ⊕I` embedding at the shared `d_ncr`
(§2.1, §3.2); M9 (the "one model" claim was architecturally unresolved)
is fixed by the SAME shared-head construction, which makes "one model"
literal rather than a build-time choice between two lossy options (§1,
§2.1); M10 (Task 2's bands sat too close to chance, and the depth ladder's
informativeness under fast mixing was unaddressed) is fixed by re-deriving
the bands against the measured chance floor and reframing `L` as OOD
distance rather than a difficulty gradient (§3.2, §7); M8 (the Axis-B
statistical criterion was unsatisfiable on this program's own prior
timing data) is fixed by restoring the flat-vs-linear/ratio discriminator
and demoting the `R²` fit to reported-only (§4.4, §7). **One item from the
pre-Rev-1 draft remains resolved, unchanged by this revision:** the
S₅-order-120-vs-K reconciliation (§3.2's M1 rewrite — Task 2 uses S₅'s own
5-point defining action, `K₂=5`, decoupled from Task 1/3's K-axis). **One
item is still open, unchanged since the first draft:** `research/
ncr_separation_grounding.md` item 8 (Nichani/Lee/Bietti rank-1-argmax
capacity) is verified only via cross-reference to a prior in-repo fetch,
not a fresh tool fetch — flagged for one human spot-check of the PDF
directly before any paper-facing use. **Items explicitly carried forward,
NOT fixed this revision (outside the §A2-ADJUDICATION binding scope
(a)–(e), disclosed rather than silently dropped):** m4 (the bridge cell's
`n=2` seeds may be underpowered for a hard axis-dropping gate, §6.2/§9.2);
m5 (§8 item 2's fp32 gradient-cross-check reference is infeasible against
`fla`'s production kernel, unresolved); m6 (a duplicated paragraph in
§6.4, harmless doc-slop); m7 (GATE-2's PARTIAL/NULL bands still overlap
with no pinned margin, §9.2, §6.2 Phase 0b). This document is gated for
build authorization on: (1) GATE 1 (§9.1, the main ortho-write verdict)
AND (2) GATE 2 (§9.2, the bridge-cell verdict, re-specified this
revision) — independent, both must resolve before their respective task's
Phase 1+ GPU-h is spent — AND (3) ATTACK ROUND 3 on this revision itself,
which should target whichever pinned assumption looks most load-bearing:
candidates include whether the length-generalization reframe (§1, §3.2)
is itself falsifiable/well-specified enough to gate a flagship claim,
whether the shared-`d_ncr` "one model" construction (§2.1, M7+M9's fix)
survives contact with actually training two different write CONTENTS
through one encoder without interference, the re-derived Task-2 chance
bands (§3.2/§7, M10's fix), and the carried-forward minors above (m4–m7).

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

---

## §A2 ATTACK ROUND 2 (2026-07-16, post-Rev-1, independent)

Second adversarial pass against DRAFT-STAGE-1-REV-1. Mandate: (1) verify
each §R1 discharge against the ACTUAL revised text, (2) find NEW defects
Rev 1 missed or introduced. Every finding is recomputed or line-cited
against a source. Rev-1 changed the failure surface substantially, and the
M1 decoupling fix (K₂=5, S₅-on-5-points) introduced a cluster of new
well-posedness defects that did not exist in the pre-Rev-1 draft. **Verdict
at the end. Lead finding (F2) is the adjudication of the single-point-vs-
full-word-problem question the coordinator flagged.**

### F2 [FATAL] — Axis A has NO depth regime that is simultaneously (structurally hard for the Transformer) AND (exact for NCR): at the tested L≤40 the Transformer is NOT structurally barred (the design's OWN citation C4 constructs a ~6-layer exact solver), and the regime where the TC⁰⊊NC¹ barrier bites (L≫2^{n_layers}≈4096) is far above both the tested ladder and NCR's own fp-exactness ceiling (~253). This is the direct structural analog of round-1 F1, one level deeper, and it is the answer to check 2(c).

**On 2(c) first (single-point vs full word).** The design's answer function
is `w(x)` — the image of ONE query letter under the composite `w =
g_{o_L}∘…∘g_{o_1}` (§3.2 line 546: "applying the composed … operators to
the query point's ℝ⁴ image and reading off which of the 5 canonical points
it lands nearest"). Is single-point tracking still NC¹-hard, or a TC⁰
shortcut? **It is the hard version, in the worst case:** Barrington's own
read-out distinguishes the two output permutations (identity vs a fixed
5-cycle ρ) by a SINGLE point's image (does `w(1)=1` or `w(1)=2`), so
`{(w,x,y): w(x)=y}` over adversarial S₅ words is NC¹-hard. So the design is
on the RIGHT side of the "track one point in TC⁰" trap the coordinator
worried about — the query form does NOT admit a single-point TC⁰ shortcut
**in the asymptotic worst case**. **But that is exactly what makes F2
fatal, not what saves it:** the hardness is a WORST-CASE, ASYMPTOTIC-in-L
statement (it bites only for adversarially-constructed words whose length
grows without bound), and the design tests neither adversarial words nor
growing L.

**Worked regime arithmetic (the gap Rev 1 never closed).**
- The barrier the design invokes (§3.2 lines 570–576): "the word problem of
  any fixed non-solvable finite group is NC¹-complete … therefore **NO
  log-precision transformer of ANY depth or width** can compute this task's
  answer function unless TC⁰=NC¹." This is true for the **family** (all word
  lengths L→∞). It gives **no** prediction of failure at any FIXED L. For
  fixed bounded L over a fixed finite group, `w(x)` is a constant-size
  computation — trivially in AC⁰⊆TC⁰.
- The design's OWN corroborating citation **C4 = Liu et al. (2210.10749)**,
  which §3.2 (lines 623–634) demotes to "secondary/corroborating," has as
  its **headline theorem** (grounding memo item 3, VERIFIED): a transformer
  of depth **O(log T)** exactly simulates ANY semiautomaton on length-T
  inputs — including the S₅ word-problem automaton. For non-solvable groups
  the depth needed is Θ(log L); ~⌈log₂ L⌉ layers SUFFICE.
- Task 2 eval ladder: `L∈{5,8,12,16,20,24,32,40}`, `L*=32` (§3.2 line 593).
  `⌈log₂ 40⌉ = 6`. The baseline Transformer (§4.1) is depth-matched to the
  DeltaNet backbone: **n_layers = 12 (98M) / 16 (392M)**. `2^12 = 4096`.
  **Every tested L (≤40) is ≪ 4096**, i.e. every tested L sits deep inside
  the regime where Liu et al. CONSTRUCTIVELY exhibit an exact transformer
  solver with layers to spare (6 needed, 12–16 available).
- Therefore the pre-registered Axis-A WIN condition — "the param-matched
  Transformer **FAILs (≤0.5)** at the same depth" (§7 Axis A row) — is not
  merely uncertain; it is **predicted-against by the design's own C4**. The
  design frames Transformer-success as a surprising negative "even though
  the complexity argument is theoretically airtight" (§8 item 5), when it is
  in fact the LITERATURE-PREDICTED outcome at L≤40.

**Why "just go deeper on L" cannot rescue it (the F1-analog impossibility).**
To make the Transformer structurally fail you need L past its log-depth-
shortcut capacity, i.e. `L ≳ 2^{n_layers} ≈ 4096`. But NCR's own exactness
does not survive there: `NCR_ORTHO_WRITE.md` §3 (lines 114–119) records that
even a *perfectly* polar-orthogonalized operator recovers only ~0.14–0.35
by h≈253 ("fp accumulation + residual write-imperfection"), and the design's
registered NCR far-depth target is `h*/L*≈40–61`, not thousands. So:
- **L≤40 (tested):** Transformer not structurally barred (C4) → Axis-A WIN
  unreachable.
- **L≫4096 (barrier bites):** NCR no longer exact (ORTHO_WRITE §3) → NCR
  loses too.
There is **no L at which the Transformer is structurally barred AND NCR is
exact** — structurally identical to round-1 F1 (which showed no query family
carries both structural-hardness and O(log h)). F1's two-family split moved
the contradiction; it did not remove it.

**The internal contradiction this exposes.** §3.2 asserts the failure is "a
genuine complexity-theoretic argument, not an empirical-drift claim" (lines
230/579), yet the ONLY basis on which the Transformer could fail at L≤40 is
the C4 *brittleness* result the design itself leans on ("even a
successfully-trained Transformer shortcut … tends not to generalize
robustly," line 631–634) — which is an **empirical OOD-generalization**
failure, the SAME mechanism §3.1 attributes to Task 1 and the SAME register
(Guu/composition-drift) F1's fix demoted Task 1 to. So at the tested scale
Axis A collapses into "another empirical composition-generalization result,"
NOT the structural separation §1/§7 sell — and the two-family distinction
F1's fix was built to preserve partially dissolves.

**Minimal fix (build-blocking).** One of: (a) STRIKE the structural
"cannot, not merely does not" framing for Task 2 at the tested L and re-cast
Axis A honestly as an **empirical held-out-depth generalization** claim
(distinct from Task 1 only in using a non-solvable group's fast-mixing walk,
not in complexity class), demoting C1–C3 to motivation and promoting C4 to
the actual predicted mechanism; OR (b) redesign Axis A to make L (or the
group size / adversarial word structure) GROW into the barrier regime AND
prove NCR stays exact there (which ORTHO_WRITE §3 currently says it does
not) — i.e. re-open the exactness-ceiling question before claiming a
structural win; OR (c) drop the structural-failure headline entirely and
run Task 2 only as a corroborating exact-composition demo. Until Axis A is
re-cast, §1's flagship "one model delivers a STRUCTURAL separation" and §7's
Axis-A WIN row cannot be earned by any run at L≤40.

### M7 [MAJOR] — Task 2's write realization (`d_ncr,2 = d_min+1 = 5`, "zero-padded/masked to its d_min block") is RANK-DEFICIENT and therefore cannot be orthogonalized by the NS-polar machinery the whole fix depends on; it also diverges from the CAPABILITY_SEPARATION infra it claims to reuse "verbatim."

**Defective quotes.** §2.1 line 164: "the smaller task's write **zero-
padded/masked to its own d_min block**." §3.2 line 526: "`d_ncr,2 =
d_min(S₅)+1 = 5` (the **+1 tight-spare margin, mirroring Task 1's K+1
convention**)." §6.2 Phase 0b line 1094: "orthogonally-conditioned `4×4`
(**padded to 5×5**) operators."

**Worked mechanism.** NS-polar (`NCR_ORTHO_WRITE.md` §2, cubic Björck–Bowie
`(1.5,−0.5)`, confirmed `research/ortho_write_grounding.md` §4) acts on each
singular value as `σ ← 1.5σ − 0.5σ³`. Fixed points: **σ=0 → 0** (`1.5·0 −
0.5·0 = 0`) and σ→1. The pre-scale `X₀ = Z/σ̂` divides by the LARGEST
singular value, so a zero singular value STAYS zero. A `4×4` operator
zero-padded into `5×5` has rank 4 → one singular value is exactly 0 → after
NS it is still 0 → `Q` has singular values `(1,1,1,1,0)` → **`‖QᵀQ − I‖_F =
1`, i.e. NOT orthogonal.** The Gate-0 target check "`‖QᵀQ − I‖_F` small"
(ORTHO_WRITE §2 step 3) **structurally cannot pass** for a zero-padded
write. This is not a training risk — it is arithmetic, independent of
optimization.

**The "reused verbatim" overclaim (a fresh instance of the exact M1
pattern).** `CAPABILITY_SEPARATION_DESIGN.md` does NOT use `d_min+1` with
zero-pad. It uses **`d_state(G) = d_min(G) + 2`** (line 1012) realized as
**`ρ_G(g) ⊕ I_{d_state − d_min(G)}`** (line 1038, "block-diagonal, identity
on the ambient dims"). A rotation ⊕ identity is **full-rank orthogonal**
(both blocks orthogonal), so NS-polar works on it. The design silently
changed `d_min+2` (full-rank ρ⊕I) to `d_min+1` (rank-deficient zero-pad),
so "reused verbatim from `CAPABILITY_SEPARATION_DESIGN.md` §1.3's real,
calibrated group infrastructure — no new matrices built" (§3.2 lines
513–516) is inaccurate on the one property (full-rank orthogonalizability)
that matters for this document's mechanism. **This directly breaks GATE 2
(the bridge cell), which trains exactly this object** — the bridge cell may
FAIL for this mechanical reason and be mis-read as a scientific NULL/FAIL on
S₅-write trainability.

**Minimal fix.** Adopt CAPABILITY_SEPARATION's actual realization: write
`ρ_G(g) ⊕ I` at `d = d_min+2 = 6` (full-rank orthogonal), or the natural
`5×5` permutation rep (full-rank orthogonal, reducible = trivial⊕standard),
and DROP the "+1 tight-spare / zero-pad" language — the K-cycle "+1 spare"
convention does not transplant to a `d_min`-dim group rep. State the exact
`d` and realization in §3.2/§6.2; do not defer to build time.

### M8 [MAJOR] — §4.4's replacement Axis-B WIN criterion (the M5 "fix") is UNSATISFIABLE on the design's OWN measured binexp timing signature, and is under-specified as a pre-registered gate.

**Defective quote (§4.4 lines 894–909, and §7 Axis B row (ii)).** "WIN
requires, for NCR's own series, `Model_log`'s `R² ≥ 0.90` AND `Model_log`'s
`R²` exceeds `Model_lin`'s `R²` by `≥0.05`."

**Evidence it cannot be met.** `NOVEL_ARCH_WATERFALL.md` §7f (line 1402–1404,
this program's OWN prior measurement of the SAME `binexp_read`): "bin-exp
**flat at ~1-3 ms from h=61 to h=2^20+5** (kernel-launch-bound … at 2^20+5
the measured gap is ≈13,000-25,000×: 2.6 ms vs 34-64 s)." The design's own
FLOP estimate agrees (§2.1: one h=40 read ≈ `8.6×10⁵` FLOPs ≈ 86 ns compute
— utterly kernel-launch-bound). **A FLAT series (constant + timing jitter)
regressed on `log₂ h` yields `b≈0` and `R²≈0`** — the model explains ~none
of the variance because the variance IS noise. So `Model_log R² ≥ 0.90`
FAILS, and it does not beat `Model_lin` (both ≈0) by ≥0.05. **By the
design's own prior data, the NCR side of the WIN gate is unreachable**, so
Axis B auto-caps at PARTIAL by construction — the exact failure M5's fix was
supposed to remove, re-introduced in a subtler form.

**Why M5's premise was half-wrong.** M5 argued the `≥10×` bar was
"transplanted from a depth (h≈10³–10⁶) the design never reaches." But timing
is accuracy-independent (§4.4 item 2 concedes this), so measuring the FLAT-
vs-LINEAR ratio at large h is legitimate REGARDLESS of the accuracy ladder —
which is exactly what §7f did (bin-exp flat vs O(h) arms linear at R²≥0.998,
ratio 20.9× at h=1021, ≈13,000–25,000× at h=2²⁰). The right discriminator
for a FLAT vs LINEAR pair is the ratio (or a flat-vs-linear form test), NOT
a log-fit. M5's fix demoted the correct, already-validated ratio criterion
BELOW a log-fit gate the data cannot pass.

**Under-specification (independent of the above).** The criterion never
pins: number of timing repeats per `h`, single-query vs batched timing, what
statistic is regressed (mean? median?), or the noise model. Without these,
`R²` is undefined, so the gate is not actually pre-registerable — matching
this document's own carried-forward open item (2).

**Minimal fix.** Restore the flat-vs-linear discriminator: NCR series
"flat" (slope not distinguishable from 0, or `Model_lin` slope `d` with a CI
including 0) vs rollout series linear (`Model_lin R²≥0.99`, slope CI excludes
0), plus the `≥10×`/reported-ratio at the largest feasible `h`. Keep the
hardware-independent `⌈log₂h⌉`-vs-`h` dependency-chain assertion as PRIMARY
(it is sound). Pin repeats/batching/statistic/noise model explicitly.

### M9 [MAJOR] — the flagship "ONE deployed model delivers BOTH properties" rests on the §2.1-unresolved single-head-two-shapes question, and BOTH mechanically-viable options undercut the headline.

**Defective quote.** §1 lines 118–120: "**ONE deployed model, trained once
per scale on a shared curriculum, delivers BOTH properties**." §2.1 lines
163–168 then concedes the head must EITHER "run two differently-shaped NCR
head instances (one per task family, disjoint parameters)" OR be "a single
encoder padded to the larger shape with the smaller task's write zero-
padded/masked to its own d_min block — **not resolved here, a build-time
decision**." §6.2 Phase 2 (line 1160): "Training is task-suite-shared (a
single run trains … Task 1 AND Task 2 episodes together)."

**Why it is load-bearing, not a detail.** The novelty memo
(`research/ncr_separation_grounding.md` Part 3) makes the "one model, both
axes" the flagship's whole distinction. But:
- **Single padded head** (the only literal "one model" reading): to serve
  Task 1's `d=33` and Task 2's `d≤6` writes in one head, the head is sized
  `d=33` and Task 2's operator occupies a ≤6×6 block → rank ≤6 in 33 dims →
  NS-polar leaves ≥27 singular values at 0 → catastrophically non-orthogonal
  (M7 at extreme). Mechanically the WORSE option.
- **Two disjoint heads:** mechanically fine, but then Axis A and Axis B are
  produced by two DIFFERENT modules bolted to a shared backbone — "one model
  delivers both" becomes "one backbone hosts two task-specific heads," a
  materially weaker headline, and the val-loss/interference story between
  the two heads is unmeasured.
Either way the §1 headline as written is not established, and the choice —
deferred to "build time" — determines whether the flagship claim is even
true. This is a NEW defect: it is a direct consequence of M1's decoupling
(K=32/d=33 vs K₂=5/d=5), which did not exist pre-Rev-1.

**Minimal fix.** Resolve the head architecture in the design, not at build
time; if two disjoint heads, rewrite §1's headline to "one backbone, two
task-specific NCR heads" and add a head-interference control to §7; if one
padded head, confront M7/M9's rank collapse head-on with the ρ⊕I full-rank
realization at a COMMON `d`.

### M10 [MAJOR] — at K₂=5 Task 2's held-out-DEPTH ladder is distributionally degenerate (fast mixing) and its accuracy scale is compressed near chance (5-way answers), so the pre-registered ladder does not probe graded depth and the HOLD/FAIL bands sit close to chance.

**Evidence.** The composite is a random walk on the Cayley graph of S₅
(order 120) over `{t,c,c⁻¹}`. This walk MIXES in O(1) steps — after ~4–5
hops the composite `w` is ≈uniform over S₅. Consequences for the eval
ladder `L∈{5,8,12,…,40}` (§3.2 line 593):
- Every ladder rung `L≥~5` presents the SAME distribution (≈uniform `w`).
  There is **no graded depth-difficulty** — L=5 and L=40 are statistically
  identical tasks. Reporting a "depth ladder" implies a difficulty gradient
  that does not exist; held-out-depth generalization from `L∈{1,2,3}` is a
  single unmixed→mixed jump at L≈4, not a ladder.
- The answer is one of **5 letters**; for ≈uniform `w`, `w(x)` is ≈uniform
  over the orbit → **chance ≈ 0.20**. Fixed-point exclusion (guard 3)
  removes the ≈20% with `w(x)=x`, leaving chance ≈ 0.25 over 4 letters. The
  pre-registered `HOLD(≥0.9)/DEGRADED(0.5,0.9)/FAIL(≤0.5)` bands therefore
  sit only ~2× above chance; a Transformer at 0.5 is well above chance, so a
  "0.9 vs 0.5" HOLD-vs-FAIL split is not the clean categorical separation
  the structural framing implies.

This is the concrete form of this document's own open item (3) ("whether
S₅-on-5-points supports BOTH held-out axes cleanly at such a small K₂ is
asserted by analogy … not independently verified") — and the answer, on
inspection, is that the DEPTH axis is degenerate at K₂=5, not merely
unverified.

**Minimal fix.** Either move to a larger group whose walk mixes slowly
enough to make a real depth ladder (but that re-opens the never-run
`d_min`/ortho-write calibration at that group), or drop the "depth ladder"
framing for Task 2 and report a single mixed-regime accuracy with an
explicit chance baseline and a many-way (not 5-way) readout. Reconcile with
F2's re-cast.

### m4 [MINOR] — GATE-2 (bridge cell) and Phase-1 calibration run at n=2, a "median over 2 seeds," to gate/drop an entire primary axis — contradicting this repo's own documented trainability-variance.

§6.2 Phase 0b pins `n=2` seeds and §9.2 gates all of Axis A on "median
rec@0.9" of 2 seeds. Median-of-2 = mean-of-2 (no tie-break, no outvote), and
STATE.md's own Task-2 lesson (`CLAUDE.md` Research Direction: "one fresh seed
cleared the bar — §1.40's surprise") is precisely that trainability is
seed-variable here. The main ortho-write wave uses n=4; a HARD gate that can
DROP the flagship structural axis on n=2 is under-powered. Raise the
axis-dropping gate to n≥3 (ideally 4).

### m5 [MINOR] — §8 item 2 / m2's pinned gradient cross-check names an INFEASIBLE reference ("the ENTIRE model in fp32"); fla's production kernel rejects fp32.

§8 item 2 (m2's fix) pins "a gradient cross-check (this cast pipeline vs. a
small-scale reference run with the **ENTIRE model in fp32**) at `<1×10⁻²`."
But `DELTANET_REALDATA_DESIGN.md` §4.3 (line 650, VERIFIED) records that
`fla.ops.delta_rule.chunk_delta_rule` "**rejects float32 inputs outright**"
— the production backbone CANNOT run in fp32. The reference must instead be
the naive fp32 recurrence (accepting the 10–20% naive-vs-chunked Jacobian
gap that same section documents) or be scoped to the NCR head only. m2's
threshold is pinned against an unrunnable reference — so m2 is only
partially discharged.

### m6 [MINOR] — §6.4 contains a duplicated paragraph (doc-slop from the M4 revision).

The entire "Main 98M/392M cells … Two coupled levers, both re-measured …
1. Raise batch size … 2. Raise seq_len …" block appears TWICE, near-verbatim
(once ≈lines 1272–1295, again ≈lines 1319–1344). Harmless but should be
de-duplicated so a build agent does not read two subtly-divergent copies.

### m7 [MINOR] — GATE-2's NULL and PARTIAL bands overlap.

§9.2 defines PARTIAL as "L=8 recovery ∈(0.5,0.9)" and NULL as "no gain over
free-write at L=8" with NO delta threshold. If ortho=0.60 and free=0.55,
both descriptions apply. Pin a minimum (ortho − free) margin (e.g. ≥0.2) to
separate PARTIAL from NULL, mirroring the WIN row's explicit `<0.5` free-
write clause.

### Positive verifications (fair-witness record — these Rev-1 fixes hold)

- **F1 fix (two-family split):** genuinely present — §1 is a real
  conjunction, §2.1 splits `binexp_read`/`loop_read` as distinct functions,
  §4.4 is scoped to Task 1, §7 removes the old single-family WIN path. No
  residual Task-2 speed claim found anywhere (§1/§3.2/§4.3/§7 all state
  Θ(L)). F1's *specific* defect is discharged (F2 is a deeper, distinct
  defect the split did not reach).
- **M6 arithmetic:** independently recomputed — `cap_length` denom `2·12·768·4
  = 73,728` ✓; Case (i) grid `{384,…,6144}` → `{22.7,45.3,90.7,181.4,362.8}`
  ✓ all clear the 20-tok floor; Case (ii) `state_bytes = 3·25·4 = 300`,
  grid `{5120,…,81920}` → `{20.8,…,333.3}` ✓; `966 = M=512` ✓. Clean.
- **M4 token re-pricing:** re-derived — 98M `16,384/0.236 = 69,424 tok/s` ✓;
  NCR `69,424/1.05 = 66,118` ✓; Phase-1 `327.68M/(66,118·3600) = 1.377` ✓;
  grand total `2+11.9+4.24+21.52+215.3+226.7 = 482` ✓. Reproduces pre-Rev-1
  numbers as claimed; no number moved silently.
- **M5 architecture pin:** the extended-β DeltaNet choice is well-grounded —
  `DELTANET_REALDATA_DESIGN.md` §4.3 (lines 685–687) confirms the custom
  block calling `chunk_delta_rule` with externally-computed masked β (stock
  `DeltaNet` computes β via `b_proj`, no mask hook) IS the right patch point.
- **CAP_SEP citations:** S₅ d_min=4 / 4-dim standard rep (line 229) ✓;
  generators `{t,c,c⁻¹}`, size 3 (line 900) ✓; A5/A6 Rev-6 hard-stop →
  Rev-7 H-ENC diagnosis/lift, S₅ budget 8K steps (lines 88–104) — disclosed
  accurately ✓. `{t=(12), c=(12345)}` generate S₅ (5 prime; p-cycle + any
  transposition generate S_p) — group theory sound.
- **No PRICE-UNKNOWN commit:** Phase 2/3 Transformer arm placeholder is
  explicitly re-priced by Phase 0a before launch; no committed spend depends
  on an un-retired PRICE-UNKNOWN.

### DISCHARGE TABLE (round-1 findings vs Rev-1's actual text)

| Round-1 finding | §R1 claim | Verified disposition |
|---|---|---|
| **F1 [FATAL]** two-properties-one-family | rewrite as two-family conjunction, scope Axis B to Task 1 | **DISCHARGED** — split is real; but exposed **F2** (structural axis has no valid depth regime at all) which the split does not reach |
| **M1 [MAJOR]** Task 2 not well-posed | S₅ generator set as rotation-rep ops, K₂=5 decoupled, A5/A6 disclosed | **PARTIALLY** — entity-pool conflation fixed & history honest, but introduced **M7** (rank-deficient d_min+1 zero-pad, `d_state` diverges from CAP_SEP's d_min+2 ρ⊕I; "reused verbatim" still overclaims) and **M10** (degenerate depth ladder at K₂=5) |
| **M2 [MAJOR]** K≤15 fallback drops non-solvable | two independent gates; GATE-1 NULL/FAIL no longer touches Task 2; R×15-cycle swap retired | **DISCHARGED** — silent-drop path removed; §9.1 NULL/FAIL branch explicitly leaves Task 2 untouched |
| **M3 [MAJOR]** ortho gate on wrong object | Phase-0b bridge cell (S₅ writes) gates Task 2 | **DISCHARGED (structure)** — bridge cell + GATE 2 added; but the gate's own object inherits **M7** (may fail mechanically, not scientifically) |
| **M4 [MAJOR]** ledger prices two machines / packed probe | tokens invariant; Phase-0a packed vs unpacked; pilot-before-probe | **DISCHARGED** — arithmetic reproduces; only doc-dup **m6** remains |
| **M5 [MAJOR]** Axis-B baseline unchosen/unpriced, ≥10× bar transplanted | pin extended-β DeltaNet, price via Phase-0a, replace bar with log-fit | **PARTIALLY** — arch/scope/price fixed & well-grounded, but the replacement criterion is **M8** (unsatisfiable on the design's own flat timing + under-specified) |
| **M6 [MAJOR]** cap_length arithmetic wrong | fix arithmetic, re-derive two floor-clearing M-grids | **DISCHARGED** — recomputed clean |
| **m1 [MINOR]** no concrete casting design | fp32-upcast/internal/bf16-downcast pipeline named | **DISCHARGED** |
| **m2 [MINOR]** "cleanly" has no threshold | pin `<1×10⁻²` cross-check | **PARTIALLY** — threshold pinned, but the "entire model in fp32" reference is infeasible (**m5**; fla rejects fp32) |
| **m3 [MINOR]** unguarded 5–20% band | two-tier `>5%` re-price / `>8%` kill | **DISCHARGED** |

**Discharge tally:** DISCHARGED 6/10 (F1, M2, M4, M6, m1, m3); PARTIALLY
4/10 (M1, M5, m2 — plus M3 structurally-discharged-but-object-broken);
NOT-DISCHARGED 0/10. New defects: **1 FATAL (F2)**, **4 MAJOR (M7, M8, M9,
M10)**, **4 MINOR (m4, m5, m6, m7)**.

### Verdict

**BUILD-BLOCKED.**

F2 is FATAL: the flagship structural axis (Axis A) has no depth regime in
which the Transformer is structurally barred AND NCR is exact — at the
tested L≤40 the design's own citation C4 predicts the Transformer SUCCEEDS,
and the barrier's regime (L≫4096) is above NCR's fp-exactness ceiling
(~253). This is round-1 F1 one level deeper; the two-family split relocated
the impossibility rather than removing it. Compounding: M7 (the S₅ write as
specified is rank-deficient and cannot be orthogonalized — mechanically
breaking GATE 2), M8 (the Axis-B WIN criterion is unsatisfiable on the
design's own flat timing data — Axis B auto-caps at PARTIAL), M9 (the "one
model delivers both" headline is unresolved and undercut by both viable head
architectures), and M10 (K₂=5 makes the Task-2 depth ladder degenerate and
the accuracy bands sit near chance). Rev 1 correctly discharged the round-1
FATAL's stated form and 5 of 6 MAJORs on arithmetic/structure, but the M1
decoupling that fixed round-1's well-posedness introduced a new cluster of
construction defects that must be resolved before any GPU-h — including the
≈2.1 GPU-h bridge cell, which as specified (M7) tests an object NS-polar
cannot orthogonalize.

**Required before ATTACK ROUND 3 / conditional build:** (a) re-cast Axis A
per F2 (honest empirical claim, or a barrier-regime redesign that also
proves NCR exactness there); (b) fix the S₅ write to a full-rank realization
(ρ⊕I at a common d), M7; (c) restore a flat-vs-linear/ratio Axis-B
discriminator and fully specify the timing protocol, M8; (d) resolve the
head architecture and re-word or control the "one model" headline, M9; (e)
address the K₂=5 ladder degeneracy, M10; (f) fold m4–m7. Everything remains
CONDITIONAL on both §9 gates AND this round's re-revision.

---

## §A2-ADJUDICATION (coordinator, 2026-07-16 — recorded before dispatching Rev 2)

**F2 ACCEPTED as FATAL — coordinator-verified against the cited sources'
own numbers** (log₂40≈5.3 ⇒ a ~6-layer exact S₅ shortcut exists at every
tested L per C4's constructive result; the barrier regime L≫2^12 exceeds
the ~253 fp-exactness ceiling in NCR_ORTHO_WRITE.md §3): "cannot, not
merely does not" is UNEARNABLE at any testable depth. The structural axis
as a finite-L impossibility claim is DEAD — permanently, not fixably.
**M7 verified by arithmetic** (a zero-padded dimension has singular value
0; NS-polar fixes 0; ‖QᵀQ−I‖=1 ≥ any tolerance). **M8 verified against
§7f's own measured flat 1–3ms binexp series** (a flat series fits neither
log nor linear; the R² criterion is unpassable). **M9/M10 accepted** on
the attacker's worked arguments (M10 chance floor: 5-point answer ⇒
0.20–0.25; HOLD/FAIL bands must be re-derived vs chance).

**Strategic disposition (the honest landing spot, consistent with the
PI's capability-first directive, which explicitly covers separations
"functionally or AS OBSERVED/TESTED"):** Axis A is REFRAMED from
structural-impossibility to MECHANISTIC LENGTH-GENERALIZATION separation —
train both arms on words ≤L_train, evaluate at L_test ≫ L_train (inside
NCR's exactness ceiling); the pre-registered prediction: baselines learn
shortcut solutions (which provably EXIST at all tested L — C4) that
empirically FAIL out-of-distribution length (C4's own brittleness finding
+ the published S₅ length-generalization empirics), while NCR's exact
composition length-generalizes BY CONSTRUCTION, with the mechanism
(shortcut vs algorithm) instrumented, not just the accuracy gap. The
TC⁰/NC¹ chain (C1–C3) is cited as MOTIVATION for why shortcut solutions
are the expected learned object — never as a finite-L impossibility bar.
The flagship headline downgrades honestly from "cannot" to
"does-not-and-we-show-why-ours-must" — surfaced to the PI at next
check-in, not buried.

**Rev 2 binding requirements:**
(a) F2: rewrite §1 Axis A + §3.2 + §7 Task-2 bands per the reframe above;
WIN = NCR HOLDs at L_test where the baseline FAILs, PLUS the
length-generalization curve characterizes the baseline failure as
shortcut-brittleness (in-distribution success + OOD-length collapse);
purge every "structurally barred at tested depths" implication;
(b) M7+M9 JOINTLY: ONE head at d=33 serving both families; Task-2
generator writes = ρ⊕I_{d−4} FULL-RANK orthogonal embeddings (the ρ⊕I
realization CAPABILITY_SEPARATION actually built — cite it correctly);
NS-polar is then well-posed; bridge cell re-specified on this object;
"one model, both properties" preserved architecturally and stated;
(c) M8: Axis-B criteria = sequential-dependency-call count (exact:
2·log₂h vs h) as PRIMARY hardware-independent metric + wall-clock ratio
≥10× at the largest tested depth as the corroborating measured metric
(restore round-1 M5's correct shape); the R² log-fit demoted to
reported-not-gating diagnostic; specify repeats/batching/noise;
(d) M10: re-derive Task-2 bands against the 0.20–0.25 chance floor
(FAIL band must sit at-or-below chance+margin, or the answer function
extended to multi-point queries to push chance down — Rev 2 chooses with
justification) and address depth-ladder informativeness (error-compounding
curve expectation, or a larger-n Sₙ option priced);
(e) partials from round 2's discharge table (M1, M5, m2) closed in the
same pass. Rev 2 → ROUND 3 before any build authorization. Everything
remains CONDITIONAL on the ortho-write verdict.

---

## §R2 REVISION 2 (2026-07-16) — changelog, every §A2 finding mapped to
its exact fix, with section references

**Scope discipline.** §1–§9 above are revised IN PLACE a second time;
§A1/§A1-ADJUDICATION/§A2/§A2-ADJUDICATION are left byte-intact as
historical record, per the gauntlet-bookkeeping convention (`CLAUDE.md`).
This changelog is the single place a reader can verify every one of
§A2-ADJUDICATION's five binding items (a)–(e) was actually discharged,
and where — mirroring §R1's own table for round 1.

| Finding | §A2-ADJUDICATION item | Exact fix | Where |
|---|---|---|---|
| **F2 [FATAL]** — Axis A has NO depth regime that is simultaneously structurally-hard-for-the-Transformer AND exact-for-NCR: at tested `L≤40` the design's own C4 citation constructs a 6-layer exact Transformer shortcut (12–16 layers available); the barrier regime (`L≳2^{n_layers}≈4096–65536`) sits far above NCR's own fp-exactness ceiling (~253, `NCR_ORTHO_WRITE.md` §3) | (a) | Axis A REFRAMED end-to-end as MECHANISTIC LENGTH-GENERALIZATION: §1 rewrites the hypothesis around train-`L≤L_train`/eval-`L_test≫L_train`; §3.2 pins the split (`L_train∈{1,…,8}`, `L_test∈{12,16,20,24,32,40}`, reusing the existing `NCR_ORTHO_WRITE.md` §4b ladder, no new infra) and promotes C4 from secondary to the PRIMARY predicted mechanism (shortcut exists+learnable ⇒ in-distribution success; shortcut is brittle ⇒ OOD collapse), demoting C1–C3 to MOTIVATION-only with an explicit "never again a finite-`L` impossibility bar" disclosure; §7's Axis-A WIN/PARTIAL/NULL bands rewritten around in-distribution-success-plus-OOD-collapse (baseline) vs. HOLD-at-every-OOD-rung (NCR), with the in-distribution-success conjunct explicitly required so a WIN reads as shortcut-brittleness, not general baseline failure; a MECHANISM INSTRUMENT (accuracy-vs-`L` curve SHAPE, reusing existing per-stratum data, plus an optional near-zero-cost attention-map diagnostic) is added as a required-report, non-gating enrichment; every "cannot, not merely does not" / "structurally barred at tested depths" sentence purged (§1, §3.2, §7, §8, §9); §3.2 separately addresses depth-ladder informativeness under fast Cayley-graph mixing (M10's second half, folded in here since it is the SAME reframe's own honesty requirement) | §1 (full rewrite); §3.2 (full rewrite of the "why structural" section + new L-train/L-test paragraph + mixing paragraph); §3.4 (C1–C4 role table); §7 (Axis-A row + overall verdict); §8 items 5/7; §9.2 |
| **M7 [MAJOR]** — Task 2's write (`d_ncr,2=d_min+1=5`, zero-padded) is RANK-DEFICIENT (one singular value structurally 0, `‖QᵀQ−I‖_F=1` unfixable by NS-polar) and diverges from the `CAPABILITY_SEPARATION_DESIGN.md` realization it claimed to reuse "verbatim" | (b) | Task 2's write is now `ρ_{S₅}(g) ⊕ I_{d_ncr−4}` — the EXACT full-rank, block-diagonal construction `CAPABILITY_SEPARATION_DESIGN.md` §1.4 actually built (`d_state(G)=d_min(G)+2`, lines 1012/1037–1038), embedded at the SHARED `d_ncr` (33 at the primary K=32 operating point, or 16 under GATE 1's K=15 fallback) rather than a separate, smaller `d_ncr,2`; both blocks pinned orthogonal (`ρ_{S₅}` real-orthogonal per `CAPABILITY_SEPARATION_DESIGN.md` line 1088; `I` trivially orthogonal), so every singular value of the target is exactly 1 — the singular-value floor is stated explicitly and NS-polar is well-posed on it in exactly the sense it already is on Task 1/Part B's own writes. The bridge cell (§6.2 Phase 0b) is re-specified on this object with an explicit orthogonality-tolerance corroboration bar (≤0.02 departure-from-normality, reused verbatim from Part A's own WIN convention) that is now MEANINGFUL rather than structurally unpassable | §2.1 (param-count + `d_ncr` unification); §3.2 ("Group and generators" + "The embedding" full rewrite); §6.2 (Phase 0b pricing + orthogonality tolerance); §4.3 Case (ii) (state-bytes recomputed at `d=33`); §8 item 7; §9.2 |
| **M9 [MAJOR]** — the flagship "ONE model delivers BOTH properties" headline rested on an unresolved single-head-two-shapes question; BOTH mechanically-viable options (single padded head, or two disjoint heads) undercut the headline | (b), jointly with M7 | Resolved by the SAME fix: ONE NCR head at ONE shared `d_ncr`, writing two different CONTENTS (a K-cycle permutation for Task 1, `ρ_{S₅}(g)⊕I_{d_ncr−4}` for Task 2) through the identical encoder/NS-polar pipeline — no separate Task-2 head, no separate param line (Task 2 now costs ZERO incremental params over Task 1's own 175,265), no rank-deficiency escape hatch. "One model" is stated as an architectural FACT in §1's flagship-claim paragraph, not an aspiration or a build-time choice | §1 (flagship-claim paragraph); §2.1 (full rewrite of the writes-happen and param-count paragraphs); §3.2 (K/R/`d` table, new `d` column) |
| **M8 [MAJOR]** — the M5-fix's `Model_log R²≥0.90` Axis-B WIN criterion is UNSATISFIABLE on this program's OWN prior measurement of the identical `binexp_read` mechanism (`NOVEL_ARCH_WATERFALL.md` §7f: flat at ~1–3ms from h=61 to h=2^20+5 ⇒ `R²≈0` on a log-fit), and the protocol left repeats/batching/statistic/noise-model entirely unspecified | (c) | §4.4 restores a flat-vs-linear/ratio discriminator as the gating criterion: NCR's own series FLAT (slope CI includes 0) AND rollout's own series LINEAR (`R²≥0.99`, slope CI excludes 0) AND the measured ratio at the largest feasible `h` `≥10×` (the exact bar this program already cleared at toy scale, 20.9× at h=1021, VERIFIED against `NOVEL_ARCH_WATERFALL.md` §7e/§7f-erratum) — mirroring round-1 M5's ORIGINAL, already-cleared shape; the `Model_log`-vs-`Model_lin` `R²` fit is DEMOTED to reported-not-gating. The depth at which overhead stops masking the gap is derived, not assumed, from the nearest measured `O(h)`-arm analog's own per-step rate (≈0.06 ms/step) against NCR's own flat floor (1–3ms): the gating ratio is read at `h≥1000`, not at `h=61`. The dependency-chain-length PRIMARY criterion is restated in the coordinator's own exact form, `2·⌈log₂h⌉` vs. `h`. Measurement protocol pinned: `B=32` standardized probes/point, 7 repeats (1 warmup discarded), median statistic, MAD-based noise-adequacy rule | §4.4 (full rewrite of point 1 and point 2); §7 (Axis-B row); §8 item 6 |
| **M10 [MAJOR]** — at K₂=5 the depth ladder is distributionally degenerate under fast Cayley-graph mixing (no graded target-difficulty gradient past `L≈5`) and the accuracy scale is compressed near chance (5-way answer ⇒ chance≈0.20–0.25), so the old `HOLD(≥0.9)/FAIL(≤0.5)` bands sat only ~2× above chance | (d) | **Bands re-derived against the measured chance floor** (chosen over the priced-but-not-built multi-point-query alternative, justified in §3.2): `HOLD/WIN` stays `≥0.9`; `FAIL` moves to `≤0.35` (chance 0.25 + a stated 0.10 margin); `DEGRADED` widens to `(0.35,0.9)`. **Ladder informativeness addressed honestly, not assumed away:** `L` is reframed as OOD DISTANCE from `L_train`, not a target-difficulty gradient — a property that SURVIVES fast mixing because it is about the INPUT REPRESENTATION's distance from the trained range, not the (admittedly ≈uniform-for-`L≥5`) target distribution; explicitly stated, with the earlier "L=5 vs L=40 differ in difficulty" implication retired | §3.2 (new "Chance floor and re-derived accuracy bands" + "Depth-ladder informativeness" paragraphs); §7 (Axis-A row uses ≤0.35/∈(0.35,0.9)); §6.2 Phase 0b (bridge-cell WIN/PARTIAL bands) |
| **M1 partial** [round-2 discharge: "entity-pool conflation fixed & history honest, but introduced M7/M10"] — A5/A6 hard-stop disclosure needed to stay accurate once a SECOND `CAPABILITY_SEPARATION` citation (the §1.4 embedding, not just §1.3's generators) entered this section | (e) | §3.2's A5/A6 paragraph extended to cover BOTH citations explicitly: the GENERATOR MATRICES (§1.3) and the EMBEDDING CONSTRUCTION (§1.4) are both stated as genuinely "already built" (real, exact, reused verbatim), while the CALIBRATION (step budgets, Gate-0/Gate-1 bars) is stated as NOT transferring by fiat for either citation — the same non-overclaim discipline Rev 1 established, now applied to the new citation too | §3.2 ("A5/A6 hard-stop history" paragraph, extended) |
| **M5 partial** [round-2 discharge: "arch/scope/price fixed & well-grounded, but the replacement criterion is M8"] — the pinned rollout baseline's remaining under-specification was entirely M8's own defect (the replacement WIN criterion), not a separate gap | (e) | Discharged BY discharging M8 above — no additional fix needed beyond M8's own §4.4 rewrite; the architecture pin (extended-β DeltaNet), scope (Task 1 only), and Phase-0a pricing hook from Rev 1 are UNCHANGED and were already sound per round 2's own "Positive verifications" | §4.4 (already covered by the M8 entry above) |
| **m2 partial** [round-2 discharge: "threshold pinned, but the 'entire model in fp32' reference is infeasible — fla rejects fp32"] — §8 item 2's gradient cross-check named an unrunnable reference | (e) | §8 item 2's pass criterion RE-SCOPED to the NCR head only: cast pipeline vs. a standalone fp32 instantiation of the SAME head fed identical fixed (detached) backbone hidden states, bypassing the backbone (and its `fla`-imposed fp32 rejection) entirely — chosen over the alternative of accepting the backbone's own documented ~10–20% naive-vs-chunked Jacobian gap, which would swamp the pinned `<1×10⁻²` tolerance. This also discharges round-2's own new minor m5 (the identical defect, different numbering), not formally required by this revision's binding scope but closed for free by the same fix | §8 item 2 |

**Net effect on the compute ledger (§6.2).** The GPU-h grand total is
UNCHANGED by this revision (still ≈482 GPU-h at 2× contingency,
excluding the rollout baseline's own untrained Phase 1–3 cells; ≈602
GPU-h including that same-order-of-magnitude placeholder, §6.2's closing
notes, unaffected by anything in this pass). Every number that DID move
moved because a `d` or bar changed, not because a schedule changed:
Task 2's Case (ii) KV-cache-matched `M`-grid (§4.3) is COMPLETELY
recomputed (`state_bytes` 300→13,068 bytes, a full reversal of which arm
has the smaller footprint) and the bridge cell's own headline number is
UNCHANGED (`≈2.12`/`≈4.24` GPU-h) but now rests on a strictly tighter
transfer (Task 2's `d` no longer differs from Part B's own `d` at all).
No PRICE-UNKNOWN item was retired or newly introduced by this revision;
§6.3's list is otherwise untouched.

**What is NOT yet resolved, carried forward explicitly for ATTACK ROUND 3
(not silently deferred — restated from the Provenance paragraph above for
visibility at the changelog's own close):** (1) m4 (round 2) — the bridge
cell's `n=2` seeds may be underpowered for a gate that can DROP an entire
primary axis (§6.2 Phase 0b, §9.2); not fixed this revision, outside the
binding scope (a)–(e). (2) m6 (round 2) — §6.4 contains a duplicated
paragraph, harmless doc-slop, unfixed. (3) m7 (round 2) — GATE-2's
PARTIAL and NULL bands still overlap with no pinned minimum margin (§9.2,
§6.2 Phase 0b); explicitly disclosed as unfixed at each place it recurs
in this revision's own new text, rather than silently inherited. (4)
Whether the length-generalization reframe's own OOD prediction
(in-distribution success + OOD collapse) is itself well-calibrated at
`L_train=8`/`L_test≥12` — chosen by analogy to standard
length-generalization practice and to reuse existing `NCR_ORTHO_WRITE.md`
§4b infrastructure, not independently validated at THIS task's own
mixing rate. (5) Whether training ONE shared-`d_ncr` head on two
DIFFERENT write contents (Task 1's K-cycle vs. Task 2's `ρ⊕I` embedding)
in a single shared curriculum (§5.2) introduces cross-task interference
that a two-disjoint-head design would have avoided — the M7+M9 fix
resolves the ARCHITECTURAL question (one shape, one head) but not this
TRAINABILITY question, which Phase 1's own per-task calibration cells
(§6.2) are the first empirical check of, not a resolved certainty.
Everything in this document remains CONDITIONAL on both §9 gates (GATE 1:
main ortho-write verdict; GATE 2: the re-specified bridge cell) and on a
fresh, independent ATTACK ROUND 3 before build authorization.

---

## §A3 ROUND 3 — FINAL VERIFICATION (2026-07-16, independent)

Narrow final-gate pass on DRAFT-STAGE-1-REV-2: (1) discharge-table fidelity,
(2) spot-arithmetic recomputed, (3) the self-flagged cross-task-interference
spot, (4) carried-forward minors m4/m6/m7, (5) light coherence sweep,
(6) CLAUDE.md hard-rule pass. Every number below was recomputed, not skimmed;
every external citation was opened at the cited file/line.

### Verdict: **REVISE** — two narrow, in-place fixes block CLEAR; nothing structural, no arithmetic error, no citation defect.

The design's substance — the two-family conjunction, the length-generalization
reframe, the M7/M9 shared-head fix, the M8 flat-vs-linear restore, the M10
chance-floor bands, the two-gate dependency structure, the entire compute
ledger — is **sound and gate-ready**. All five §R2 discharges and all three
round-2 partials are substantively present in the revised text. All
spot-arithmetic reproduces. All load-bearing citations are faithful. The two
blocking items are localized editorial/gate-wording defects, fixable in one
Rev-3 pass without touching any experiment, gate threshold, or number.

### FINDINGS

**F3-1 [REVISE] — The F2 stale-language purge is INCOMPLETE; the retired
structural register survives in §0/§2.1/§2.2/§3.1 and, most prominently, the
§3.2 SECTION HEADER — contradicting this document's own purge-completeness
claim.** §1 line 129 asserts: *"Every 'cannot, not merely does not' /
'structurally barred at tested depths' sentence is purged from this design as
of this revision (§3.2, §7, §8, §9, below)."* §R2's changelog (line 2898)
scopes the purge to "§1, §3.2, §7, §8, §9." Both claims are FALSE against the
actual text. Surviving live-stale sentences (each re-asserts the twice-retired
F1/F2 structural claim as a current framing, not as a retirement reference):
- **§3.2 header, line 607** (inside the claimed-purged §3.2): *"Task 2 —
  Non-solvable-group word-problem chain (PRIMARY for Axis A — **structural
  failure**, well-posed construction…)"* — the primary-task header still
  labels Axis A "structural failure."
- **§2.1 Assessment, lines 325–330:** *"TC⁰ transformers therefore **cannot
  solve it** unless TC⁰=NC¹… This is a **genuine complexity-theoretic
  argument, not an empirical-drift claim** — but it ONLY **bites** if… a
  non-solvable group."* This directly contradicts §1 line 120–122 ("a claim
  about what the two architectures' LEARNED SOLUTIONS generalize like, not
  about what either architecture can in-principle compute") and §3.4's demotion
  of C1–C3 to MOTIVATION-only. The "it ONLY bites" clause re-asserts a finite-L
  bite that F2 proved does not exist at any tested L.
- **§2.2, lines 425/433:** *"the Transformer, §4.1, on **structural
  grounds**"* and *"Task 2's **structural baseline of record** is the
  Transformer"* — stale rationale for the baseline choice.
- **§3.1, line 584:** *"a TC⁰ transformer is NOT structurally barred from this
  task **the way it is from Task 2**"* — asserts Task 2 has a live
  structural-barred property.
- **§3.1, line 602:** *"Task 2 carries the **structural (Axis A) claim**"* —
  calls Axis A structural.
- **§0 reading list, line 41:** *"for §3's **structural-failure task** below."*
- Evidence: F2/§A2-ADJUDICATION retired "cannot, not merely does not" as
  UNEARNABLE at any testable depth; §1/§3.2-body/§7/§3.4 correctly reframe to
  MECHANISTIC LENGTH-GENERALIZATION. The WIN bands and hypothesis are clean —
  the defect is that the purge missed §0/§2.1/§2.2/§3.1 and the §3.2 header, so
  a build agent or reviewer reading those sections in isolation would take Axis
  A as a live structural-impossibility claim. **Minimal fix:** complete the
  purge — retitle the §3.2 header ("mechanistic length-generalization" per §1);
  in §2.1 add the F2 caveat (the TC⁰⊊NC¹ argument is ASYMPTOTIC / MOTIVATION,
  does NOT bite at any tested finite L) or cross-ref §1/§3.2's reframe; strike
  "structural grounds"/"structural baseline of record" (§2.2), "the way it is
  from Task 2" (§3.1 584), "structural (Axis A) claim" (§3.1 602), and
  "structural-failure task" (§0 line 41); then correct line 129's/§R2's purge
  scope to reflect the actual sections touched.

**F3-2 [REVISE] — GATE-2's PARTIAL and NULL bands still overlap with no pinned
margin, and this gate decides whether a PRIMARY axis is DROPPED — an
ambiguous-at-read-time gate (round-2 m7, carried forward unfixed).** §9.2 /
§6.2 Phase 0b define **PARTIAL** = "Gate-0 clears, L=20 recovery ∈(0.35,0.9)"
and **NULL** = "Gate-0 clears, no gain over free-write at L=20" with NO delta
threshold. A borderline result — e.g. ortho rec@L=20 = 0.60 (∈(0.35,0.9) ⇒
PARTIAL) with free-write = 0.58 (no meaningful gain ⇒ NULL) — satisfies BOTH
descriptions. The PARTIAL/NULL boundary is precisely the boundary between
"proceed with Task 2/Axis A (at a re-anchored floor)" and "DROP Task 2/Axis A
for Stage 1" (§9.2 branches; §8 item 7's trigger inherits this ambiguity).
Per the round-3 mandate, a read-time-ambiguous GATING criterion is
REVISE-level, not a minor — the WIN row already carries an explicit free-write
clause (≤0.35), so the fix is symmetric and trivial. **Minimal fix:** pin a
minimum `(ortho − free)` margin (e.g. ≥0.2, mirroring the WIN row) separating
PARTIAL from NULL in BOTH §9.2 and §6.2 Phase 0b; the design already flags this
exact gap (lines 1436/1980/2048/2927) but leaves it unfixed.

**F3-3 [MINOR, disclosed — recommend fold] — m4: the axis-dropping GATE-2
bridge cell and Phase-1 calibration run at n=2 seeds.** §6.2 Phase 0b pins
`n=2`; §9.2 can DROP Axis A on "median rec@0.9" of 2 seeds (median-of-2 =
mean-of-2, no tie-break). This contradicts this program's own documented
trainability-variance (CLAUDE.md Research Direction: "one fresh seed cleared
the bar — §1.40's surprise") and the n≥3/n=4 norm the main ortho-write wave
and head-to-head axis-1 used. Disclosed (lines 1396/2044/2923) but unfixed.
Cheap fix (n=3 ≈ +1.06 GPU-h at 1×). Does not hard-block conditional build but
SHOULD be raised to n≥3 before the bridge cell runs, since it gates an entire
primary axis.

**F3-4 [MINOR, disclosed] — m6: §6.4 duplicates the "Main 98M/392M cells… Two
coupled levers… 1. Raise batch size… 2. Raise seq_len" block near-verbatim**
(≈lines 1597–1622 and ≈1646–1671). Confirmed present, disclosed (lines
2047/2926), harmless doc-slop. De-dup so a build agent does not read two
subtly-divergent copies. Non-blocking.

### CHECK 3 — cross-task interference (the self-flagged spot)

**(i) Exact kill/branch criterion?** PARTIAL. There is an exact per-task-arm
trigger — Phase-1 Gate-0 (in-distribution recovery ≥0.9 AND val-loss inside
`k=2·s_ref`), and §8 item 4 KILLS Phase 2 for a task arm that plateaus below
0.9 — so a task whose learnability collapses under co-training would be caught
per-arm. But there is **NO dedicated interference criterion** that compares the
shared-curriculum Phase-2 per-task accuracy against the single-task Phase-1
calibration with a pinned degradation threshold. Interference that degrades one
task in the shared Phase-2 run WITHOUT dropping it below the absolute 0.9 bar
would pass every gate silently. The design discloses this honestly as
carried-forward open item (5) (lines 2936–2942) and names Phase-1's per-task
cells as "the first empirical check… not a resolved certainty" — so it is
gate-adjacent, not pure vibes, but the interference-specific instrument is
under-specified. Recommend (non-blocking) wiring an explicit
Phase-1-single-task vs Phase-2-shared per-task degradation threshold.

**(ii) One-axis-rule violation in §5's curriculum?** NO. The M7/M9 fix makes
the two families share ONE head at ONE shape (architecture fixed), so
co-training two task families is standard multi-task, not "two unproven
architectural axes bundled" (CLAUDE.md's rule targets the latter). A clean
isolating ablation EXISTS structurally: Phase 1 runs SEPARATE per-task
calibration cells (Task-1 arm, Task-2 arm), single-family, before Phase 2
co-trains — comparing the two isolates interference. The pieces are present;
only the explicit comparison-with-threshold (i, above) is unwired.

### CHECK 4 — carried-forward minors

- **m4:** disclosed; recommend fold (F3-3). Non-blocking but touches an
  axis-dropping decision.
- **m6:** disclosed; harmless doc-slop (F3-4). Non-blocking.
- **m7:** disclosed BUT **DOES block (F3-2)** — a band overlap in a gating
  criterion that decides a primary-axis drop is ambiguous at read time; the
  round-3 mandate classifies this REVISE-level, not minor. This is the one
  carried-forward minor that must be pinned before CLEAR.

### SPOT-ARITHMETIC (recomputed — all PASS)

- **(a) Task-2 bands vs chance:** 5-letter answer ⇒ chance 1/5=0.20; fixed-point
  exclusion (guard 3) ⇒ 1/4=0.25. FAIL ≤0.35 = 0.25 + 0.10 margin. HOLD ≥0.9
  (≈3.6× chance), DEGRADED (0.35,0.9). Coherent. ✓
- **(b) ρ⊕I_29:** 4+29 = 33 = d_ncr ✓; ρ_{S₅} orthogonal (4 unit σ) ⊕ I_29 (29
  unit σ) ⇒ all 33 singular values exactly 1, none 0 ⇒ NS-polar well-posed
  (orthogonal is a σ→1 fixed point, no structural 0 to strand). ✓
  CAPABILITY_SEPARATION_DESIGN.md §1.4 VERIFIED to contain the cited
  realization: line 1012 `d_state(G)=d_min(G)+2`; lines 1037–1038
  `rho_G_embedded = rho_G ⊕ I_{d_state−d_min(G)}`; line 1088 "ρ_G pinned
  real-orthogonal, §1.3.1"; line 229 S₅ d_min=4 (4-dim standard rep); line ~899
  generating set `{t,c,c⁻¹}` size 3. The design's generalization of d_state=6
  to the shared d_ncr=33 (identity block absorbs any ambient ≥4) is valid and
  disclosed. ✓
- **(c) Axis-B dependency counts:** 2·⌈log₂h⌉ vs h — {61→12, 200→16, 1000→20,
  5000→26, 20000→30} vs h; ratios 5.1/12.5/50/192/667. ✓ Ratio-mask
  derivation: rollout ≈ h·0.06 ms vs NCR flat 1–3 ms; 10× the NCR floor
  (10–30 ms) is reached at h ≈ 167–500 — matches the design's stated
  "≈167 to ≈500." h=61 below (3.7 ms), h=200 borderline (12 ms), h=1000 clears
  (60 ms ≫ 30 ms). Pinned gating h≥1000 sits above the meaningfulness
  threshold. ✓ (0.06 ms/step verified from source: 64.4 ms/1021=0.0631,
  61.3 s/1,048,581=0.0585, NOVEL_ARCH_WATERFALL.md §7f lines 1293–1294.)
- **(d) KV-cache reversal:** R=3 (design's current value, VERIFIED as S₅'s
  `{t,c,c⁻¹}`, not R=4 or K=32). state_bytes = 3·33²·4 = 13,068 ✓ = 3× Task-1's
  4,356 (a genuine reversal of Rev-1's 300 bytes / ~14.5×-smaller). denom
  2·12·768·4 = 73,728 ✓. Case (i) grid {384…6144}→{22.7,45.3,90.7,181.4,362.8},
  floor-min 339 ✓; Case (ii) grid {128…2048}→{22.7,45.4,90.7,181.5,363.0},
  floor-min 113 ✓; Case (ii) = Case (i) at 1/3 the M (13,068=3·4,356) ✓. The
  round-1 M6 number (cap_length(M=32,R=32)=60.5, from state_bytes=32·33²·4=
  139,392) is a HISTORICAL R=32 correction; the current table uses R=3 and is
  internally consistent — the new R supersedes the R=32 row, no contradiction. ✓
- **(e) Bridge cell re-price:** 80,000/320,000 × 4.24 = 1.06 GPU-h/cell × 2
  seeds = 2.12 (1×); 2× = 4.24. ✓ (Part B measured 4.24 GPU-h/320K-step cell,
  NCR_ORTHO_WRITE.md § CEILING AMENDMENT.)
- **(f) L_test top rung 40 vs ceiling 253:** 40 ≪ 253 ✓. NCR_ORTHO_WRITE.md §3
  (lines 114–117) VERIFIED: h*=253 (=8K−3) recovers only ~0.14–0.35 (fp
  accumulation). §4b ladder {5,8,12,16,20,24,32,40} VERIFIED (line 233); Part B
  R=4 / loop_read / "binexp does not apply" VERIFIED (lines 213/238–239); WIN
  convention departure-from-normality ≤0.02 VERIFIED (lines 166–167). Task-1's
  separate ladder {5,12,20,29,40,61} VERIFIED (line 131). The 253 (a Part-A
  Z^h ceiling) applied to Part-B distinct-generator products is CONSERVATIVE
  (a product of exactly-orthogonal factors has no eigenvalue amplification),
  and the design frames it as an upper bound — coherent, not overclaimed. ✓
- **Param/FLOP cross-checks:** P(33,64)=40·4096+4·33·64+46·64+33=175,265 ✓
  (NOVEL_ARCH_WATERFALL.md §9.3 formula, line 3066). F(32,33,64)=11,837,696 ✓.
  Deltas 0.18%/0.045% ✓. Ledger: Phase0a 11.9, Phase1 21.52, Phase2 215.3,
  Phase3 226.7, bridge 4.24, smoke 2 ⇒ 482 ✓; +120 rollout ⇒ 602 ✓. Rates
  0.236/0.836 s/step VERIFIED (FROZEN_BIAS_LM_DESIGN.md §13.7). K=15 SCALES
  "4/4 converged + far-depth HOLD" VERIFIED (NOVEL_ARCH_WATERFALL.md §11.2 line
  4188, early-LN recipe). All PASS.

### DISCHARGE TABLE (round-2 findings vs Rev-2's actual text)

| Round-2 finding | §R2 claimed disposition | Verified against revised text |
|---|---|---|
| **F2 [FATAL]** structural axis has no valid depth regime | reframe Axis A as mechanistic length-generalization; purge every "cannot"/"structurally barred" sentence (§1,§3.2,§7,§8,§9) | **DISCHARGED IN SUBSTANCE** — §1/§3.2-body/§7/§3.4 reframe is real and correct — **but PURGE INCOMPLETE (F3-1):** stale structural framing survives in §0/§2.1/§2.2/§3.1 and the §3.2 header; line-129/§R2 purge-completeness claim is false |
| **M7 [MAJOR]** rank-deficient zero-pad write | ρ_{S₅}(g)⊕I_{d_ncr−4} full-rank at shared d_ncr; σ-floor=1 stated; NS-polar well-posed; bridge-cell orthogonality bar meaningful | **DISCHARGED** — §2.1/§3.2 rewritten; all σ=1 verified; CAP_SEP §1.4 citation accurate |
| **M8 [MAJOR]** unsatisfiable R²-log-fit criterion | restore flat-vs-linear/ratio ≥10×; demote R² fit to reported-only; 2·⌈log₂h⌉ vs h primary; pin B=32/7-repeats/median/MAD | **DISCHARGED** — §4.4/§7 rewritten; ratio-mask h≥1000 derivation checks out; protocol pinned |
| **M9 [MAJOR]** "one model" unresolved head architecture | ONE head, one d_ncr, two write CONTENTS; Task 2 = ZERO incremental params; "one model" stated as architectural fact | **DISCHARGED** — §1/§2.1 rewritten; the interference *trainability* question is correctly left open as item (5), not silently claimed resolved |
| **M10 [MAJOR]** bands near chance + degenerate ladder | re-derive bands vs 0.25 chance (FAIL≤0.35); reframe L as OOD distance not difficulty gradient | **DISCHARGED** — §3.2/§7 rewritten; chance arithmetic verified; mixing/OOD-distance reframe coherent |
| **M1 partial** A5/A6 disclosure vs 2nd (§1.4) citation | extend disclosure to cover both §1.3 (generators) and §1.4 (embedding); calibration does not transfer | **DISCHARGED** — §3.2 A5/A6 paragraph covers both citations honestly |
| **M5 partial** rollout-baseline under-specification | discharged by discharging M8; arch pin/scope/price unchanged & sound | **DISCHARGED** — no residual beyond M8's own fix |
| **m2 partial** infeasible fp32 reference | re-scope §8 item 2 cross-check to NCR head only, bypass backbone | **DISCHARGED** — §8 item 2 rewritten; also closes round-2's m5 |

**Discharge tally:** 8/8 round-2 dispositions substantively present. F2's
reframe is real but its *purge-completeness* sub-claim is FALSE (F3-1). Two
carried-forward items block: m7 (F3-2, gate ambiguity). New/other:
0 FATAL, 0 arithmetic errors, 0 citation defects; 2 REVISE (F3-1 purge, F3-2
gate margin), 2 disclosed minors (m4/F3-3, m6/F3-4), 1 non-blocking gap
(interference instrument).

### PATH TO CLEAR

Fold F3-1 (complete the F2 purge + correct the purge-scope claim) and F3-2
(pin the GATE-2 PARTIAL/NULL margin); folding F3-3 (n≥3) and F3-4 (de-dup) is
recommended in the same pass. No re-attack is required for these — they are
verifiable in-place edits. Once folded, this design is
**CLEAR-FOR-CONDITIONAL-BUILD**, gated on: (1) GATE 1 = the ortho-write verdict
(§9.1, ~2026-07-17); (2) GATE 2 = the bridge-cell verdict (§9.2, §6.2 Phase 0b);
(3) Phase 0/0a smoke + Phase 1 per-task calibration passing Gate-0. Both §9
gates remain independent and either may resolve NULL/FAIL without killing the
program.

---

## §A3-ADJUDICATION (coordinator, 2026-07-16 — recorded before dispatching Rev 2.1)

REVISE ACCEPTED. Zero arithmetic/citation defects; 8/8 round-2
dispositions substantively present; the two blockers are narrow:
**F3-1** (stale "structural/cannot" framing surviving in §0/§2.1/§2.2/
§3.1/§3.2 — the purge-completeness claim was false) and **F3-2** (GATE-2
PARTIAL/NULL band overlap: a borderline bridge result satisfies both
bands at the boundary that drops a PRIMARY axis).

**Rev 2.1 dispatched (surgical, freeze-scope):**
(a) F3-1: complete the purge grep-driven (every quoted location + any
residual match for structural-failure/complexity-barred framing outside
the motivation paragraph and record sections);
(b) F3-2: pin the GATE-2 margin — PARTIAL requires (ortho − free) ≥ 0.2
(mirroring the WIN row per §A3's recommendation) in BOTH §9.2 and §6.2
Phase 0b; NULL = everything below;
(c) F3-3 UPGRADED from disclosed-minor to REQUIRED: a gate that can drop
a primary axis does not run at n=2 — bridge cell n=3 seeds, re-priced
(~6.4 GPU-h at 1×), ledger updated;
(d) F3-4: de-dup §6.4;
(e) the interference comparison WIRED with an exact threshold: Phase-2
shared-curriculum per-task accuracy ≥ (Phase-1 single-task calibration −
0.05 absolute) per task, else DIAGNOSE-FIRST branch (no silent pass) —
closing §A3's carried-forward item (5).
Per §A3's own ruling both blockers are verifiable in-place edits: after
Rev 2.1 the COORDINATOR verifies the edits directly (no fourth
independent round) and freezes the header CLEAR-FOR-CONDITIONAL-BUILD,
triple-gated (GATE 1 ortho verdict, GATE 2 bridge cell n=3, Phase-0/1
calibration Gate-0).

---

## §R2.1 REVISION 2.1 (2026-07-16, freeze-scope) — changelog, every
§A3-ADJUDICATION binding item (a)–(e) mapped to its exact edit

**Scope discipline (surgical revision, Rev 2.1).** §0–§9 above are
revised IN PLACE a third time, strictly to the five items
§A3-ADJUDICATION bound; §A1/§A1-ADJUDICATION/§A2/§A2-ADJUDICATION/§A3/
§A3-ADJUDICATION are left byte-intact as historical record. No number
outside the ones this table lists moved.

| §A3-ADJUDICATION item | Finding | Exact fix | Where |
|---|---|---|---|
| (a) | **F3-1** — the F2 stale-"structural/cannot" purge was INCOMPLETE: §1 line 129's own purge-completeness claim ("§3.2, §7, §8, §9") was FALSE against the actual text — live-stale framing survived in §0, §2.1 (header + body), §2.2 (×2), §3.1 (×2), and the §3.2 SECTION HEADER itself | Fixed every location §A3 quoted, verbatim: §0's reading-list line ("structural-failure task" → "mechanistic-length-generalization task"); §1 line 129's purge-scope sentence corrected to name the true scope (§0, §2.1, §2.2, §3.1, §3.2, §7, §8, §9); §2.1's Assessment paragraph header ("Structural-failure grounding" → "Complexity-theoretic MOTIVATION… asymptotic, NOT a finite-`L` bar") and body ("TC⁰ transformers therefore cannot solve it… ONLY bites" → explicit ASYMPTOTIC/MOTIVATION framing, cross-referencing §1's reframe, never again "bites" in the impossibility sense); §2.2's two "structural grounds"/"structural baseline of record" phrases struck (→ "per §2.1's complexity-theoretic MOTIVATION, not a structural bar" / "baseline of record"); §3.1 line 584 ("NOT structurally barred… the way it is from Task 2" → "has NO complexity-theoretic motivation… unlike Task 2… asymptotic and non-binding at any tested `L`"); §3.1 line 602 + the §3.2 header ("structural (Axis A) claim" / "PRIMARY for Axis A — structural failure" → "mechanistic length-generalization" throughout). **THEN grep-driven**: a full re-scan of §0–§9 for residual `structural`/`barred`/`cannot`/`complexity-theoretic` matches found 2 additional live-stale instances beyond the 6 quoted locations (neither flagged by §A3's own quote list) — §2.1's paragraph header (part of the same defect as the quoted body text, fixed together) and §9.1's fallback-branch note ("cannot carry Axis A's structural claim" → "cannot carry Axis A's (mechanistic length-generalization) claim"). Every other grep hit was verified to be an ALLOWED survivor: the §1 motivation-only paragraph (explicitly disclaims finite-`L` force), historical/retired-claim references that already say "retired"/"removed"/quote the old framing as false, or architecture-sense "structural" usage (e.g. "structurally-zero singular value," "no dimension structurally excluded," M7's "STRUCTURAL reason" for a rank-deficiency defect) unrelated to Axis A's complexity register. Axis A's label is now uniformly "mechanistic length-generalization" everywhere in §0–§9. | §0 (reading list); §1 (purge-scope sentence, line ~129); §2.1 (Assessment header + body); §2.2 (load-bearing-caveat paragraph, ×2); §3.1 (line ~584, line ~602); §3.2 (section header); §9.1 (K=15-fallback paragraph) |
| (b) | **F3-2** — GATE-2's PARTIAL and NULL bands overlapped with NO pinned margin (carried-forward m7), and this gate can DROP a PRIMARY axis — a read-time-ambiguous gating criterion, REVISE-level per the round-3 mandate. Worked example: ortho rec@L=20=0.60 (∈(0.35,0.9) ⇒ nominally PARTIAL) with free-write=0.58 (no meaningful gain ⇒ nominally NULL) satisfied BOTH band descriptions at once. | **Pinned margin**: PARTIAL now requires Gate-0 clears AND L=20 recovery ∈(0.35,0.9) AND **(ortho rec@L=20 − free-write rec@L=20) ≥0.2** (mirroring the WIN row's own free-write-gap convention). NULL = Gate-0 clears and neither the WIN nor the (now margin-gated) PARTIAL condition is met — i.e. recovery ≤0.35, OR recovery ∈(0.35,0.9) but the margin is <0.2. The worked example now resolves unambiguously: ortho=0.60/free=0.58 ⇒ margin=0.02≪0.2 ⇒ **NULL**. Applied identically in both places the bands are stated; verified no other section restates the old ambiguous bands (only the Provenance paragraph's historical "m7… still overlap" note remains, left untouched as dated record per the historical-record convention, not a live re-statement of the bands themselves). | §6.2 Phase 0b (Gate paragraph, PARTIAL/NULL sentences); §9.2 (GATE 2's PARTIAL and NULL/FAIL bullets) |
| (c) | **F3-3** — UPGRADED by §A3-ADJUDICATION from disclosed-minor (m4) to REQUIRED: the bridge cell (GATE 2) runs at `n=2` seeds on a "median rec@0.9" statistic to decide whether to DROP an entire primary axis (Axis A) — median-of-2 has no tie-break, and this program's own documented trainability-variance precedent (CLAUDE.md: "one fresh seed cleared the bar") sets an n≥3 norm this gate violated. | **Bridge cell raised to `n=3` seeds.** Recomputed price: per-seed rate `80,000/320,000 × 4.24 GPU-h/cell = 1.06 GPU-h/cell` (unchanged) `× 3 seeds = 3.18 GPU-h (1×)`; `2× contingency ≈ 6.36 GPU-h` — a delta of **`+1.06` GPU-h (1×) / `+2.12` GPU-h (2×)** over Rev 1/Rev 2's own `n=2` projection (`2.12`/`4.24` GPU-h). **Ledger propagated**: §6.2's Phase-0b line item (4.24→6.36 GPU-h, 2×); §6.2's grand total (≈482→**≈484 GPU-h**, 2×, excluding the rollout baseline: `2 + 11.9 + 6.36 + 21.52 + 215.3 + 226.7 = 483.78 ≈ 484`); the rollout-inclusive total (≈602→**≈604 GPU-h**, 2×: `484 + 120 = 604`); §8 item 7's own cost citation (`≈2.1 GPU-h, 1×` → `≈3.18 GPU-h, 1×`). Phase 1's SEPARATE per-task calibration cells (also `n=2`, a different line item m4 did NOT single out and §A3-ADJUDICATION did NOT include in this item's binding scope) are UNCHANGED — only the bridge cell's own seed count moved. | §6.2 (Phase 0b pricing paragraph; grand-total sentence; rollout-inclusive total sentence); §8 (item 7's cost parenthetical) |
| (d) | **F3-4** — m6: §6.4 contained a near-verbatim duplicate of the "Main 98M/392M cells… Two coupled levers… 1. Raise batch size… 2. Raise seq_len" block, appearing twice (once before the "Calibration/Phase-0/0b cells" paragraph, once directly before "Main 98M/392M cells are NOT packed"). | De-duped: the FIRST (earlier, less complete) copy is deleted and replaced with a one-line forward-pointer; the SECOND copy is KEPT because it is the one that already carries the more complete cross-reference ("re-measured before launch, specifically, before Phase 0a's unpacked probe, per the sequencing above") and sits immediately before its own natural continuation ("Main 98M/392M cells are NOT packed"). No content lost — the surviving copy is a strict superset of the deleted one. | §6.4 (first "Main training cells" occurrence deleted; second "Main 98M/392M cells" occurrence retained, immediately followed by "…are NOT packed") |
| (e) | **CHECK 3(i)** — no dedicated cross-task interference criterion existed: Phase-1's per-arm Gate-0 (≥0.9 absolute) could pass while Phase-2's shared-curriculum co-training silently degraded one task's accuracy relative to its own isolated Phase-1 calibration, with no comparison ever made against that isolated baseline — passing every existing gate silently. | **Wired with an exact threshold, in BOTH places**: for EACH task family independently, Phase-2's shared-curriculum per-task accuracy must be **≥ (that task's OWN Phase-1 single-task calibration accuracy − 0.05 absolute)**. A breach on either task triggers **DIAGNOSE-FIRST**: HOLD Phase 3 for that scale, adjudicate using the single-family ablation arms Phase 1 already ran (Task-1-only / Task-2-only calibration cells — already run, NO new GPU-h) to separate genuine cross-task interference from an unrelated regression, before any further spend at that scale. No silent pass: an unadjudicated breach blocks Phase 3 authorization for that scale by construction. | §6.2 (new "Cross-task interference criterion" paragraph, end of the Phase 2 subsection); §8 (new item 8, cross-referencing §6.2) |

**Net effect on the compute ledger (§6.2).** The ONLY ledger-moving fix
in this pass is (c) — the bridge cell's seed raise. Delta: **+1.06 GPU-h
(1×) / +2.12 GPU-h (2×)** at the bridge-cell line item, propagating to
**+2.12 GPU-h** on both grand-total figures: **≈482→≈484 GPU-h** (2×,
excluding the untrained rollout-baseline arm) and **≈602→≈604 GPU-h**
(2×, including that same-order-of-magnitude placeholder). Every other
number in §6 is UNCHANGED — (a), (b), (d), and (e) are wording/threshold/
de-dup fixes with zero GPU-h impact. §6.3's PRICE-UNKNOWN list is
untouched (no item retired or introduced).

**Note on §A3-ADJUDICATION's own parenthetical price estimate.** The
adjudication's dispatch note (c) above estimated the re-price as "~6.4
GPU-h at 1×" — recomputing exactly (per this agent's mandate to recompute
every touched number) gives **3.18 GPU-h at 1×** (`1.06 × 3`) and **6.36
GPU-h at 2×** (`3.18 × 2`); the adjudication's "~6.4" figure matches the
**2× contingency** number, not 1×. The ledger above uses the recomputed,
internally-consistent 1×/2× figures (3.18/6.36), not the adjudication's
loosely-labeled shorthand.

**What remains open, unchanged by this pass (outside §A3-ADJUDICATION's
binding scope (a)–(e), disclosed rather than silently dropped):** m5
(§8 item 2's fp32 gradient-cross-check reference, already re-scoped in
Rev 2 and not reopened by §A3); the length-generalization reframe's own
OOD-calibration-at-`L_train=8` open question (§R2's item (4)); whether
the shared-`d_ncr` "one model" construction's TRAINABILITY (as opposed to
its now-wired interference GATE, item (e) above) holds in practice —
Phase 1's calibration cells remain the first empirical check of that,
not a resolved certainty. Everything in this document remains
CONDITIONAL on both §9 gates (GATE 1: main ortho-write verdict; GATE 2:
the bridge cell, now at `n=3`) and, per §A3-ADJUDICATION's own closing
instruction, on the COORDINATOR's direct verification of this table
against the edits above (no fourth independent attack round required).

---

## §R2.1-ADJUDICATION — COORDINATOR FREEZE (2026-07-16)

Rev 2.1's five fixes verified DIRECTLY by the coordinator (per §A3's
ruling that the two blockers were verifiable in-place edits needing no
fourth independent round): (1) purge complete — every residual
structural/cannot hit in §0–§9 classified as an allowed survivor
(motivation-only paragraph, retirement-provenance notes, architecture-
sense usage); (2) GATE-2 margin (ortho − free ≥ 0.2) pinned verbatim in
§6.2 Phase 0b AND §9.2, worked borderline example lands NULL; (3) bridge
n=3 re-price verified (1.06 × 3 = 3.18 GPU-h at 1×) and the ledger
re-summed by the coordinator (2 + 11.9 + 6.36 + 21.52 + 215.3 + 226.7 =
483.78 ≈ 484); (4) §6.4 de-dup confirmed; (5) interference threshold
(Phase-2 per-task ≥ Phase-1 calibration − 0.05 absolute, DIAGNOSE-FIRST
on breach) wired in §6.2 AND §8. Rev 2.1 also corrected the
§A3-ADJUDICATION's own "~6.4 GPU-h at 1×" shorthand (that was the 2×
figure) — steer-verification working as required. STATUS:
CLEAR-FOR-CONDITIONAL-BUILD, triple-gated (header). The gauntlet on this
design is CLOSED.

---

## §N1 NOVELTY-GATE ADJUDICATION (2026-07-16)

**Standing.** This section is APPEND-ONLY and does not alter any frozen §0–§9
or §A/§R record above; where it amends a frozen band, grid, or seed count, it
does so as an EXPLICITLY LABELLED AMENDMENT with reasoning, per the
freeze convention. It records the coordinator/Opus adjudication of the
PI-directed novelty re-verification gate (three sweeps, dispatched
2026-07-16, transcripts under the session `tasks/` dir; sweep summaries in
the session scratchpad `novelty-gate/`). The gate exists to DISCHARGE (or
refuse) the GATE-2 novelty precondition before any Task-2/Axis-A GPU-h is
committed. Rulings are adversarial toward this design's own claims by
mandate.

### The three sweeps — verdicts of record

**Sweep 1 (BY-MECHANISM) — the mechanism's novelty boundary HOLDS.** No
located prior work combines (a) in-context-WRITTEN full-rank d×d operators,
(b) exact algebraic composition, (c) query-time O(log h) repeated-squaring
reads, and (d) orthogonalized writes for decay-free composition. Closest
prior art is **MuonSSM (arXiv:2606.30461, ICML 2026 Oral, VERIFIED
full-text)** — but it orthogonalizes RANK-1 KV outer-product injections
with a SINGLE quintic Newton–Schulz step for stability, not FULL-RANK d×d
operators driven to near-exact QᵀQ≈I over many cubic iterations for
composition-exactness. **Critical conflation pre-empted:** MuonSSM's
"O(log L)" is Blelloch associative-scan TRAINING parallelism (the
Mamba2/DeltaNet chunking family), NOT a query-time O(log h) read of one
written operator — the flagship and kwall papers must state this distinction
explicitly wherever the "O(log)" name appears (also applies to Log-Linear
Attention, §N1 R4). Anchor corrections landed: FWM's "fixed hop count"
UNCONFIRMABLE from the abstract (soften); Log-Linear Attention 2506.04761
now PUBLISHED ICLR 2026; Atlas = 2505.23735 (Titans successor, no distinct
2026 paper). Sweep-1 conclusion: Axis-B / the mechanism combination is a
SEARCHED ABSENCE, not an assumed one — boundary intact.

**Sweep 2 (BY-TASK) — the SEPARATION SHAPE is SCOOPED; the mechanism claim
is not.** The collapse-vs-hold separation on non-solvable-group word
problems (S₅ specifically) is an ESTABLISHED GENRE since 2022, demonstrated
on S₅ itself repeatedly. **Near-scoop: Yau et al. arXiv:2506.10918
("Sequential-Parallel Duality in Prefix Scannable Models," Andreas lab) —
S₅ "cups and balls" directly, train len 4–18, test to 180, T-PSM holds far
beyond transformer AND Mamba (both collapse); an EMPIRICAL/LEARNED hold,
no exactness argument.** Genre-defining priors: Liu 2210.10749 (S₅ shortcut
collapse, 2022); Merrill/Petty/Sabharwal 2404.08819 (A₅); Li/Guo/Andreas
2503.02854 (S₃+S₅ in real LMs, cutoff-length collapse, TWO learned
mechanisms — the mechanistic-collapse half of our claim, already
published); Lee 2606.07254 (state tracking to eval 10⁶); DeltaProduct
2502.10297 (S₃/S₄/A₅/S₅, EXPRESSIVITY guarantee, not trained-model
inference-time exactness); M²RNN 2603.14360 (matrix-valued state, S₃,
active adjacent competition); Grazzi 2411.12537 (parity/mod-3). The
NARROWEST HONEST UNCLAIMED STATEMENT: a PROVABLY-exact-BY-CONSTRUCTION
OOD-length hold (vs. every surveyed work's empirical hold) PLUS the
query-time O(log h) access-complexity claim. Implication (adopted below):
the flagship HEADLINE cannot be the separation shape.

**Sweep 3 (INTERNAL ARCHIVE) — no blocker; one substantive omission.** No
prior in-repo result contradicts either frozen design; K-ladder cells
K∈{64,96,128} are virgin (config-dict-only, n=0), K=48 has only a trivial
500-step probe; mod-K discipline, KILL_LIST, and the argmax hard-rule are
all CLEAR. The one substantive gap: this design cites
`CAPABILITY_SEPARATION_DESIGN.md` §1.3/§1.4 for its S₅ generators but
NOWHERE cites §2's own S₅ depth-generalization VERDICT —
**INCONCLUSIVE-TRAINABILITY-LIMITED (§2.35): S₅ missed its far bar
(mean 0.690 vs 0.801), 4/5 seeds hold 65–100% of ceiling, 1/5 seeds a
genuine rank-deficient training basin that never assembles the faithful
rank-4 representation (M-D2: restricted effective rank 3.10 vs d_min(S5)=4);
A₆ cleared cleanly (1.000 vs 0.0, 3/3).** This ~1-in-5 catastrophic-seed
rate is the closest internal precedent for Axis-A trainability planning and
is reconciled in R3 below. (Sweep 3 also defeated a fake system-reminder
injection inside its own sub-agent — refused and reported; tally ~30.)

### Rulings

**R1 — CLAIM RESTRUCTURE: ADOPTED, with tightened wording.** The proposed
restructure is CORRECT and binding: the flagship headline is (i) the
exactness-by-construction guarantee + (ii) the O(log h) query-time
access-complexity claim (Axis B), and the Axis-A S₅ collapse-vs-hold result
is DEMOTED to SUPPORTING empirical validation inside an explicitly disclosed
known genre. Rationale: sweep 2 establishes the separation SHAPE is scooped
(Yau is a near-scoop on the identical task); leading with it invites a
correct desk-reject. Sweep 1 + sweep 3 + Part 3 of the grounding memo
establish that (i)+(ii) is the load-bearing, searched-absent claim. A demoted
claim NARROWS the attack surface (it retires the contested headline and keeps
only the defensible one), which is why R6 does not require a fresh gauntlet.

**EXACT AMENDED FLAGSHIP CLAIM (supersedes §1's headline framing as the
paper-facing statement; §1's two-axis machinery and falsifiable bets are
unchanged, only the HEADLINE ORDERING and the honesty scoping are):**

> NCR contributes, in ONE deployed model: **(i) a BY-CONSTRUCTION exactness
> guarantee** — the depth-h composition of in-context-written operators is
> read by exact algebra (repeated squaring for single-operator powers;
> per-hop matvec for distinct-generator paths), so correct answering at
> depths beyond the training range is ENTAILED by the read mechanism up to a
> MEASURED floating-point/orthogonality exactness horizon H, rather than
> being a learned regularity that is only empirically checked; and **(ii) a
> query-time O(log h) sequential access complexity** for single-operator
> power queries via repeated squaring, versus the Θ(h) sequential rollout
> every published matrix-state alternative pays at matched expressivity. The
> S₅ non-solvable-group collapse-vs-hold length-generalization result is
> reported as SUPPORTING empirical validation of (i), inside an EXPLICITLY
> DISCLOSED and established genre (Yau 2506.10918; Liu 2210.10749;
> Merrill/Petty/Sabharwal 2404.08819; Li/Guo/Andreas 2503.02854; Lee
> 2606.07254; DeltaProduct 2502.10297; M²RNN 2603.14360) — NOT as a novel
> separation shape.

**What we show that Yau (2506.10918) does not — the hostile-reviewer
sentence, stated precisely (Yau held in hand):** Yau's T-PSM exhibits an
EMPIRICALLY-OBSERVED S₅ length-generalization to test-length 180 with NO
exactness argument. NCR contributes two things Yau does not: **(a)** the
OOD-length hold is ENTAILED BY CONSTRUCTION (the read is exact algebra up to
horizon H) rather than learned-and-observed — the NATURE of the guarantee
differs, not merely its numeric reach; and **(b)** the query-time O(log h)
access complexity for single-operator powers, a SEQUENTIAL-READ-DEPTH
property distinct in kind from Yau's parallel-prefix-scan O(log) THROUGHPUT
complexity (a Blelloch-scan-over-the-sequence property, the same conflation
sweep 1 pre-empts for MuonSSM). **Honesty scope, binding:** the contribution
is the NATURE of the guarantee plus the access-complexity separation — it is
NOT a claim to reach greater empirical depth than Yau. If GATE-1's measured
exactness horizon H is below 180, that MUST be disclosed and Yau's greater
empirical reach conceded; the paper's edge is guarantee-nature + O(log h),
never "we go deeper." This scoping is required because this design's own §1
(lines ~100–104) records that even a perfectly polar-orthogonalized operator
recovers only ~0.14–0.35 by physical depth 253 — i.e. H is a MEASURED,
GATE-1-pinned quantity, not the naive fp-precision ~253.

**Consequential WIN-band amendment (AMENDS §7's overall-program-verdict
rule).** §7 currently requires BOTH primary axes at WIN for an overall WIN.
Under R1, the FLAGSHIP-HEADLINE WIN is carried by **(i) exactness-by-
construction (validated on Task 1 exactness + the Task-2 hold) AND (ii)
Axis-B O(log h) (Task 1, §4.4)**; the Axis-A S₅ separation (Task 2, §7 row
1) is retained as REQUIRED-corroborating evidence for (i) but is NO LONGER
the sole novel headline — a build agent implementing the paper must
re-express §7's overall rule so that a bridge-cell NULL/FAIL (Axis A dropped,
§9.2) caps the paper at "the exactness + access-complexity claim, S₅
corroboration deferred," which is still a publishable flagship result rather
than the automatic PARTIAL §7 currently records. This is a reporting/framing
amendment (no training-protocol change); it is flagged here for build-time
implementation and is NOT a re-opening of the gauntlet.

**R2 — L_test PROTOCOL AMENDMENT: EXTEND, conditionally and DATA-PINNED;
both arms evaluated.** The frozen top rung L_test=40 is defensibly weak vs
Yau's 180 and Lee's 10⁶ — a reviewer WILL object, and the demoted-but-still-
reported Axis-A validation must be credible. Eval-only extension is cheap
(no retraining; forward passes on the trained checkpoints for BOTH arms).
BUT a fixed extension to {80,160,240} is REFUSED as unsound: this design's
own §1 records NCR far-depth recovery of only ~0.14–0.35 by depth 253, so
the "~253 fp-exactness ceiling" is NOT a clean exact-to-253 guarantee — the
BINDING constraint is the ortho-write mechanism's MEASURED far-depth
recovery, exactly what GATE 1 (ortho-write verdict, ~2026-07-17) resolves.
Extending blindly to 240 risks manufacturing an NCR self-collapse and a
self-inflicted Axis-A NULL.

> **AMENDED Axis-A eval grid.** Keep the frozen in-distribution anchors
> {5,8} and OOD rungs {12,16,20,24,32,40}. ADD eval-only OOD strata at
> `{60, 100, 160}`, EVALUATED FOR BOTH ARMS (NCR and the param-matched
> Transformer), with the TOP rung capped at the GATE-1-measured exactness
> horizon H: include an added rung L only if NCR's own measured recovery at
> L clears the HOLD bar (≥0.9) in the GATE-1/bridge-cell data; any rung
> where NCR's measured recovery falls below HOLD is reported as NCR's
> DISCLOSED exactness horizon (an honest boundary, NOT a silent drop and NOT
> scored as an Axis-A NULL provided H already exceeds the frozen L_test=40).
> If GATE-1 pins H ≥ 160, report the full {60,100,160}; if H lands between
> rungs, report up to the last rung ≤ H and disclose H. The baseline's
> collapse (or, informatively, its non-collapse) at every added rung is
> reported alongside. No training-context change: L_train stays {1,…,8};
> all added strata are eval-only.

Cost: eval-only forward passes at three added depths for two arms across the
existing seeds — negligible next to the training ledger (no line-item change
to §6's grand total; folded into the existing eval budget). This directly
answers the Yau objection where the mechanism supports it and stays honest
where it does not.

**R3 — SEED/TRAINABILITY RECONCILIATION: RAISE the bridge cell to n=5, and
AMEND the WIN band with a catastrophic-seed disposition clause.** §2.35's S₅
precedent is a ~1-in-5 catastrophic (rank-deficient-basin) seed rate,
measured AT n=5 — and the internal sweep's finding 1.5 (free-write
far-depth 2/9 seeds ≥0.9 at h*=61) independently shows heavy S₅-adjacent
trainability variance. The frozen bridge cell is n=3 (raised from n=2 at
§A3-ADJUDICATION). n=3 is INADEQUATE for THIS object, on two grounds: **(1)
characterization** — a 1-in-5 phenomenon cannot be reliably observed at
n=3 (E[catastrophic in 3]=0.6, so >40% of n=3 draws see ZERO catastrophic
seeds and would wrongly read the object as clean); to reconcile against
§2.35 at all you need the sample size at which the basin was measured, n≥5.
**(2) false-drop risk** — with median-of-3 and p=0.2, P(≥2 of 3
catastrophic ⇒ median flips to FAIL)≈10.4%, a non-trivial false-NULL rate
for a gate that DROPS a primary axis.

> **AMENDMENT to the frozen GATE-2 bridge-cell spec (§6.2 Phase 0b / §9.2).**
> (a) Bridge-cell seed count RAISED n=3 → **n=5** (matching §2.35's own S₅
> evidence base). Re-price: per-seed rate 1.06 GPU-h (unchanged) × 5 =
> **5.30 GPU-h (1×)**, **10.60 GPU-h (2×)** — a delta of +2.12 GPU-h (1×) /
> +4.24 GPU-h (2×) over the frozen n=3 figure (3.18/6.36). This propagates
> the §6.2 grand total by +4.24 GPU-h at 2× (frozen ≈484 → **≈488 GPU-h**,
> 2×, ex-rollout; rollout-inclusive ≈604 → **≈608**). Small, and mandatory
> for a gate that can drop a primary axis.
> (b) **Catastrophic-seed disposition clause** (NEW, added to the WIN band).
> Report median-of-5 rec@0.9 at L=20 AND the full per-seed distribution AND
> each seed's M-D2 restricted-effective-rank vs d_min(S₅)=4 (the §2.34
> diagnostic that dissociates a rank-deficient basin from a capability
> failure). A seed diagnosed as a rank-deficient training basin (restricted
> effective rank measurably < d_min, train loss satisfied) is DISCLOSED and
> EXCLUDED from the median — but ONLY under that explicit diagnosis, mirroring
> §2.34/§2.35's own disposition; it is never dropped as a bare outlier. WIN
> requires median-of-the-faithful-seeds ≥0.9 AND ≤1 catastrophic seed of 5.
> **≥2 catastrophic seeds of 5 = the gate does NOT clear** (a genuine
> trainability wall for the S₅-generator write, disclosed, Axis-A dropped
> per §9.2's NULL branch — a scientifically informative result under the now
> rank-deficiency-free construction, §R2(b)'s M7 fix). The pinned PARTIAL
> margin (ortho − free ≥0.2, §R2.1(b)) is unchanged and applies to the
> faithful-seed median.

This is a RECORDED AMENDMENT to the frozen WIN band, with reasoning, per the
instruction — the frozen n=3 spec is NOT silently changed.

**R4 — REQUIRED CITATIONS: finalized (see the consolidated list below and
the mirror in `research/ncr_separation_grounding.md`'s 2026-07-16 section).**
Tiered must-cite-and-differentiate list, VERIFIED/UNVERIFIED tags carried
from the sweeps.

**R5 — K-LADDER CLEARANCE: CONFIRMED clean.** The novelty sweeps concern the
S₅ mechanistic-length-generalization (Task 2 / Design B) axis and the
mechanism-boundary; they do NOT touch the K-ladder's cyclic single-K-cycle
Part-A / random-orthogonal-bank Part-B cells. Per sweep 3: K∈{64,96,128}
virgin (n=0), K=48 only a trivial 500-step probe — zero re-run overlap with
any internal inventory. `NCR_KLADDER_DESIGN.md`'s frozen gates (double-gated
on the ortho-write verdict + the Stage-0 K=128 calibration cell) are
UNDISTURBED by anything in the three sweeps. The K-ladder's own pre-existing
structural flag (the 0.9·K ≤ 65 achievable-gate ceiling capping K=96/128) is
untouched and remains that design's own item, not a novelty-gate finding.
Recorded as a note in `NCR_KLADDER_DESIGN.md` §N1.

**R6 — CEREMONY: this adjudication is SUFFICIENT; no fresh independent
gauntlet round required.** Per CLAUDE.md tiered ceremony, this design is
publication-bound and already survived a full 3-round adversarial gauntlet +
coordinator freeze. The R1–R3 amendments change CLAIM LANGUAGE (a NARROWING
— retires the scooped headline, keeps the defensible one), the EVAL GRID (a
conservative, data-pinned extension — more testing), and a SEED COUNT (n=3→5
— more statistical power). None touches the training protocol, architecture,
or the two-axis structure; all three STRENGTHEN or NARROW rather than expand
the attack surface, and the by-task sweep (explicitly "THE risk sweep")
already served as the adversarial novelty round motivating them. Sufficiency
carries ONE build-time coherence condition, NOT a gauntlet round: the §7
overall-verdict re-expression flagged in R1 (a single build-agent audit
item). GATE-1 (ortho-write verdict) and GATE-2 (bridge cell, now n=5) remain
in force; the horizon H (R2) and the catastrophic-seed count (R3) are
resolved by their data, not by more design ceremony.

### Consolidated must-cite-and-differentiate list (R4) — flagship + kwall

**Tier 1 — MANDATORY cite + explicit differentiation (survival-critical):**
- **Yau et al. 2506.10918** [VERIFIED] — near-scoop, identical S₅ task,
  empirical hold to 180; differentiate on guarantee-nature + O(log h)
  (R1's hostile-reviewer sentence).
- **Liu 2210.10749** [VERIFIED] + **Merrill/Petty/Sabharwal 2404.08819**
  [VERIFIED] — the genre known since 2022 (S₅ shortcut collapse / A₅).
- **Li/Guo/Andreas 2503.02854** [VERIFIED] — S₃+S₅ in real LMs, learned
  collapse mechanism; align/contrast, don't re-claim.
- **DeltaProduct 2502.10297** [VERIFIED] + **Grazzi 2411.12537** [VERIFIED]
  — EXPRESSIVITY guarantee ≠ trained-model inference-time exactness (the
  distinction that protects our exactness claim).
- **MuonSSM 2606.30461** [VERIFIED] — closest to the ortho-write; pre-empt
  the Blelloch-scan-vs-query-read conflation (rank-1 stability NS vs
  full-rank composition NS).
- **Barrington 1989** [VERIFIED] — classical ORIGIN of exact composition by
  repeated squaring/doubling; cite as the math's origin (its LEARNED
  in-context instantiation is what is new).

**Tier 2 — cite as concurrent/adjacent context:**
- **M²RNN 2603.14360** [VERIFIED] — matrix-valued state, S₃, active adjacent
  competition (fixed nonlinear recurrence, not in-context-written operators).
- **Lee 2606.07254** [VERIFIED-as-real; CHARACTERIZATION AMBIGUOUS — the two
  sweeps describe this SAME arXiv ID differently: sweep 1 as a "held-out
  transition-pair falsifier" eval-methodology paper, sweep 2 as an S₃×S₃
  state-tracking-to-10⁶ result. Reconcile by body refetch BEFORE any
  paper-facing use; do not state either characterization as settled fact.]
- **FWM 2011.07831** [VERIFIED, with correction] — closest in-context-write
  prior art; "fixed hop count" is UNCONFIRMABLE from the abstract — state as
  "recursive, gradient-trained, APPROXIMATE reads," not a fixed hop count.
- **Log-Linear Attention 2506.04761** [VERIFIED — now PUBLISHED ICLR 2026,
  update the cite] — closest on the "O(log)" NAME; differentiate
  (hierarchical-summarization compute complexity, not a query-time
  repeated-squaring read).
- **RWKV-7 2503.14456** — expressivity claim VERIFIED; **empirical S₅
  length-generalization SPECIFICS UNVERIFIED (body fetch failed) — do NOT
  cite S₅ empirical numbers until a body refetch confirms them.**
- **Sequential Group Composition 2602.03655** [VERIFIED] — in-WEIGHTS
  complement (not a competitor). **HOLA 2607.02303** [VERIFIED] — exact KV
  cache, not exact algebraic composition. **Guu/Miller/Liang 1506.01094**
  [VERIFIED] — composition-error-cascading precedent (grounds the predicted
  baseline drift).

**Tier 3 — reviewer-bait, cite-and-distinguish (no group-word overlap):**
- Anil 2207.04901; Zhou 2402.09371 (addition, fragile); RASP-L 2310.16028
  (leverage FOR us — S₅ plausibly lacks a short RASP-L program ⇒ predicts
  collapse); Position Coupling 2405.20671 (a baseline VARIANT a reviewer may
  demand, never applied to S₅ — not a scoop); 2402.09268 + 2509.09001
  (O(log k) via transformer LAYERS, tangential); Echo 2605.06997
  (approximate power-iterated filter). All [VERIFIED-as-real per the sweeps].
- **Rank-necessity harness:** Nichani/Lee/Bietti 2412.06538 [VERIFIED via
  cross-reference only — human PDF spot-check flagged before paper-facing
  use]; scope as SUPPORTING mechanism-integrity, never folded into the
  composition-capability headline.

### DISCHARGE

**The GATE-2 NOVELTY CONDITION is DISCHARGED upon this §N1 section being
committed.** The mechanism-novelty boundary HOLDS (sweep 1); the separation
shape is scooped but the claim is restructured to the defensible
exactness-by-construction + O(log h) headline with the S₅ result demoted to
disclosed-genre corroboration (R1); the eval grid is extended data-pinned
(R2); the bridge cell rises to n=5 with a catastrophic-seed disposition
clause (R3); the citation obligations are finalized (R4); the K-ladder is
clear (R5); and no further gauntlet round is required (R6). GATE-2's
REMAINING (empirical) preconditions — the bridge-cell verdict itself (now at
n=5) and GATE-1's ortho-write verdict pinning the exactness horizon H — are
UNCHANGED and still gate any Task-2/Axis-A GPU-h. Build authorization
continues to require both §9 gates plus the R6 build-time §7 coherence
re-expression.

---

## §N2 GATE-1 AMENDMENT + SCALE-TARGET (PI-ratified 2026-07-17)

**Standing.** APPEND-ONLY, per §N1's own convention (line 3342-3344 above):
this section does not alter any frozen §0-§9, §A/§R, or §N1 text; every
amendment to a frozen gate, dimension, or ledger line is EXPLICITLY
LABELLED with reasoning below, never silently substituted.

**Trigger — the verdict §9.1 was downstream of has now landed.**
`NCR_ORTHO_WRITE.md` §9 VERDICT OF RECORD (2026-07-17): Part A (K=32, the
§9.1 primary) reads **FAIL** — Gate-0 dead 4/4 seeds, in-dist
`recovered_frac@0.9`=0.000 at h∈{1,2,3}, loss dips then collapses to ≈1.0;
Part B (structured-operator discriminator) also reads **FAIL**, compounded
by a dead free-bank baseline. §10 POST-FAIL CODE RE-AUDIT (independent,
read-only) rules **(C) MECHANISM-CONFIRMED, no bug**: in the tight-spare
`d=K+1` geometry the cosine-similarity read exerts zero gradient pressure
on the write's overall scale or its `(d-K)`-dim spare direction; that
direction random-walks toward zero, and once it crosses `~1e-7` the
Newton-Schulz polar projection's backward Jacobian explodes (`~1/σ_min`),
gradient-clipping converts the explosion into a task-destroying step, and
the resulting near-singular state is absorbing at K≥24 (§10.7). **This FAIL
is not K=32-specific** — `ortho_K24_s{0-3}` fails identically (Gate-0 dead
4/4, same dip-then-collapse signature, §9.1's own per-cell table). §9.1's
own branch structure ("If ORTHO-WRITE WIN/PARTIAL/NULL/FAIL") is therefore
resolved on its FAIL leg — but that leg's own K=15 fallback ("Falls back to
K=15, d≈16... re-derive from that config's own archived z-dumps before
pinning a number," line 1990-1994) is superseded, per the PI-ratified pivot
below, by a config §9.1 never named: **K=24, d_ncr=25, free-write (no
NS-polar projection at all)**.

### N2.1 GATE-1 discharge by amendment, not satisfaction

§9.1's own text conditions the flagship's K-axis on two premises, both now
independently falsified:

1. **"Ortho-write is the route to larger K."** FALSIFIED by §9/§10 above:
   the NS-polar constraint is mechanism-confirmed untrainable at K∈{24,32}
   under the tight-spare geometry this document's own §2.1/§3.2 use — not a
   budget or calibration miss, an absorbing optimization pathology (§10.0).
2. **"Larger K is required."** FALSIFIED independently of (1): nothing in
   this document's own hypothesis (§1), Axis-B claim (§4.4/§7), or Task-2
   construction (§3.2) is a function of K's numeric value. The O(log h)
   read is `binexp_read`'s repeated squaring over a `d_ncr×d_ncr`
   matrix — exact by construction (matmul associativity) at ANY d,
   K-independent. Axis-A's `K₂=5` (§3.2, entity-pool size) is ALREADY
   decoupled from Task 1/3's K-axis and was never K=32-contingent. The
   `d_ncr=K+1` head-width choice is a per-relation write-conditioning
   parameter, not a capability requirement — §3.2 itself already states a
   `d=16` (K=15) contingency for exactly this reason (line 665, "or 16
   under GATE 1's K=15 fallback"). Head params/FLOPs are shown negligible
   at either d (N2.2(c) below) — "bigger K" bought nothing this design's
   own hypothesis needed and cost real trainability.

**The amended enabling condition (supersedes §9.1's WIN/PARTIAL/NULL/FAIL
branch dispatch for GATE-1 ONLY; §9.2/GATE-2 is untouched — see N2.7 item
1):** the free-write K=24/d_ncr=25 healthy verdict of record —
`NCR_ORTHO_WRITE.md` §9.1's own "Secondary observation (K24, not scored by
§4)" and §9.3's spectral summary: `free_K24_s{0-3}` — in-dist
`recovered_frac@0.9`=1.000 at h∈{1,2,3} in all 4 seeds, far-depth
`recovered_frac@0.9`∈{0.999,1.000} at every audited ladder rung
h∈{5,12,20,29,40,61} (§9.1's per-cell table), `A_eff_rank`=23.99-24.00 (=
full K), departure-from-normality 0.004-0.009, condition number 1.0-1.1,
`min|λ|/c*`=0.99-1.00, all 4/4 seeds, at the 320,000-step (4×) budget —
**SATISFIES the flagship's write-viability precondition**: a shared-`d_ncr`
NCR head trains a converged, well-conditioned, far-depth-exact write at a
real (K,d) operating point this document already uses (§3.1's abelian
construction, §2.1's shared-head architecture). **This DISCHARGES GATE-1.**
No ortho-write mechanism is used anywhere downstream; every place
§2.1/§2.2/§3.1/§3.2 above says "passed through the Newton-Schulz orthogonal
-write projection... if the ortho-write gate licenses it" now reads "the
free (unconstrained) write, licensed by this amendment" — the NS-polar
pipeline (`NCR_ORTHO_WRITE.md` §2) is RETIRED from the Stage-1 build, not
merely unused this wave (§9.1's own pre-registered next move on FAIL — "a
fallback parametrization... not more budget" — is deferred to the
expm/Cayley track, `STATE.md` 2026-07-17 briefing, DEMOTED off the critical
path, idle-filler only).

### N2.2 The three mechanical re-derivations

**(a) `d_ncr`: 33 → 25.** K=24, tight-spare `d=K+1=25` — verified against
§3.2's OWN convention, not asserted fresh: line 665, "`d_ncr = K+1 ∈ {16,
33}`," and §9.1's own K=15 fallback already instantiates the identical rule
at a different K (`d≈16`, i.e. 15+1). 24+1=25 is the SAME formula at the
amended K, not a new convention. Task 1/3's abelian construction (§3.1) and
the shared-head architecture (§2.1) both parametrize directly on `d_ncr`;
no other frozen text pins `d_ncr` to 33 specifically except by way of K=32
(§9.1's now-discharged primary branch).

**(b) Task-2 writes: `ρ_{S₅}(g) ⊕ I_{29}` → `ρ_{S₅}(g) ⊕ I_{21}`.** Verified
against §3.2's own general formula (line 666, "embedded... as `ρ_{S₅}(g) ⊕
I_{d_ncr−4}`") and its own dimension claim (line 651-653: "`d_min=4`,
realized as the 4-dimensional standard representation... on the zero-sum
hyperplane of ℝ⁵"). §9.2's `I_{29}` is that general formula evaluated at
`d_ncr=33` (33−4=29); at the amended `d_ncr=25`, the identical formula
gives `I_{25−4}=I_{21}`. Both blocks remain exactly orthogonal by the same
argument §3.2 already makes (`ρ_{S₅}` pinned real-orthogonal, `I` trivially
orthogonal, singular-value floor at exactly 1 — none of this is
d-dependent). No other §3.2 quantity moves: `K₂=5` (entity pool) and `R=3`
(generator count) are independent axes, already decoupled from `d_ncr` by
§3.2's own K/R/d table (line 735-739).

**(c) Param counts + §6.2 GPU-h ledger, re-priced at K=24/d=25.**

*Params* (formula `P(d,h)=40h²+4dh+46h+d`, verified exact, §2.1 line 291,
`NOVEL_ARCH_WATERFALL.md` §9.3): `P(25,64) = 40·4096 + 4·25·64 + 46·64 + 25
= 163,840+6,400+2,944+25 =` **173,209 params** (was 175,265 at d=33) —
**2,056 fewer params, −1.17% of the head's own size.** As a fraction of the
backbone: 98M → 173,209/97,618,176 = 0.1774% (was 0.1795%); 392M →
173,209/391,869,440 = 0.0442% (was 0.0447%). **Direction: cheaper, as
expected** (a smaller `d` shrinks the `4dh+d` terms); **magnitude:
invisible at the design's own reporting precision** — both figures still
round to "+0.18% / +0.045%," confirming §2.1's own finding that the `40h²`
encoder-width term (163,840 of 175,265 → 94.6% of the head, UNCHANGED since
`h_ncr=64` is untouched by this amendment) dominates, not `d`.

*FLOPs* (formula `F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h`, verified exact,
§2.1 line 314): `F(24,25,64) = 76·24·4096 + 4·25·4096 + 12·576·64 +
4·24·25·64 + 4·625·64 = 7,471,104+409,600+442,368+153,600+160,000 =`
**8,636,672 FLOPs/relation-write** (was 11,837,696 at K=32/d=33) —
**3,201,024 fewer FLOPs, a 27.0% reduction per write.** §2.1's own
overhead-share calculation (line 412-415: "≤64 write invocations/step ×
1.2×10⁷ FLOPs ≈ 7.7×10⁸ FLOPs/step — ≈0.008% of backbone FLOPs/step")
shrinks proportionally to ≈5.5×10⁸ FLOPs/step ≈ **0.006%** of the same
≈9.6×10¹² FLOPs/step backbone figure (invocation count carried over
unchanged — §2.1's own `T_bind=224` was derived FOR K=32 specifically and
is not re-derived here for K=24; a build agent should re-check it, but
even a 2× miss on invocation count leaves this in the same negligible
band).

*§6.2 GPU-h ledger — UNCHANGED, not merely small.* Phase 0 (smoke, ≈2
GPU-h), Phase 1 (calibration, 10.76 GPU-h 1×/21.52 GPU-h 2×), Phase 2 (98M
main, ≈107.7/215.3 GPU-h), and Phase 3 (392M main, ≈113.3/226.7 GPU-h) are
all priced from §6.1's measured tokens/sec anchors (69,424 tok/s @ 98M,
19,598 tok/s @ 392M, both plain-backbone measurements) times a FLAT ≤5%
NCR-overhead ceiling (§2.2 line 325, "by analogy to the ONE directly
comparable measured precedent... ≤1.2% overhead" — a projected assumption,
never a `d`-derived one) — **neither term is a function of `d_ncr`.** §2.1's
own line 306-307 states this explicitly as the load-bearing point: "cost
is governed by the encoder's own width, not by the written object's
dimension." **GATE-3's ~22 GPU-h calibration price (Phase 0 + Phase 1,
`STATE.md` 2026-07-17 briefing) therefore does not move under this
amendment** — the only line items that move are the on-paper param/FLOP
figures above, both already negligible and now marginally more so. (One
qualification, flagged not resolved here: Phase 1's own 21.52 GPU-h (2×)
splits evenly, 10.76/10.76, between a Task-1 arm and a Task-2 arm gated
separately by GATE-2 — see N2.7 item 1, since GATE-2 is untouched by this
amendment.)

### N2.3 Headline reframe

Per §N1 R1 (line 3412-3443, already-ratified, this amendment does not
alter it) and `research/ncr_separation_grounding.md` Part 3(a)-(b) (lines
239-265): the flagship capability headline is **the composition READ is
exact BY CONSTRUCTION** — `binexp_read`'s repeated squaring is ordinary
matrix multiplication, associative and error-free up to floating-point
accumulation, true at ANY K or scale because it is an algebraic identity,
not a learned regularity (§N1's own "ENTAILED by the read mechanism...
rather than being a learned regularity that is only empirically checked,"
line 3432) — **plus O(log h) query-time sequential access complexity**,
versus every published matrix-state alternative's Θ(h) rollout (§4.4, Axis
B). **The learned WRITE is a separate claim, empirically demonstrated, not
by-construction:** the K=24/d=25 free-write verdict (§9.1/§9.3) shows the
encoder trains a well-conditioned (cond≈1.0), full-rank (`A_eff_rank`=24.0)
operator WITHOUT an explicit orthogonality projection — a genuine, measured
result about one architecture at one (K,d) operating point (grounding memo
Part 3(a): "an empirical result about one architecture at one scale, not a
new complexity-class capability... exactness [of the write] as its
necessary empirical companion"). **This reframe is consistent everywhere it
touches:** §N1's own EXACT AMENDED FLAGSHIP CLAIM (line 3428-3443) already
states the read is "ENTAILED... up to a MEASURED... exactness horizon H" —
this amendment supplies the mechanism by which H is now measured (free-
write's own far-depth recovery, §9.1, not ortho-write's — since ortho-write
is dead, H is re-anchored to the free-write ladder: h≈61 clears at ≈1.0
recovery in 4/4 K=24 seeds, the new empirical floor for H pending any
deeper eval, N2.7 item 2). No frozen claim language is altered by this
reframe — §1's hypothesis and §7's WIN bands are untouched; this section
states which existing sentences the amendment's mechanism now satisfies.

### N2.4 The load-bearing caveat — RECIPE-CONDITIONAL, GATE-3 stays fully load-bearing

**The K=24-healthy result does not hold at every `(K,d)` recipe — it is
specific to the tight-spare `d=K+1` geometry with EXTENDED (4×/320,000-step)
budget, and the recipe-dependence is documented, not assumed:**

- **Dead at the older `d=2K` ("2K") recipe.** `NOVEL_ARCH_WATERFALL.md`
  §11.4's own joint summary (line 4569-4596): K=24 at `d=48` (the `d=2K`
  convention) reads **0/4 Gate-0 CONVERGED at every budget/anneal
  combination tried** (1×, 2×, 4×, anneal 0.75 — 16 cells) — the SAME K, a
  categorically dead recipe at the wider `d`. `NCR_ORTHO_WRITE.md` §5's own
  in-silico table corroborates from the opposite direction: `K24@d48 (2K,
  dead)` has `cond#=2951.5`, `front asis@.9=0.0` — "even post-hoc polar
  barely helps," vs. `K24@d25 (K+1)` "the far healthier baseline this wave
  actually uses." **This is the regate precedent**: the SAME K produces a
  dead cell at one `d` convention and a healthy one at another — nothing
  about "K=24" alone was ever the free variable; the `(K,d)` PAIR is.
- **Weak at 1× (80,000-step) budget, even at the healthy `d=K+1` geometry.**
  `NOVEL_ARCH_WATERFALL.md` §11.4 Table 2 (line 4477-4480): K24@d25 at 1×
  budget reaches 4/4 Gate-0 CONVERGED (in-dist recovery=1.000 uniformly)
  but far-depth recovery is **weak-and-seed-variable**
  (`sweep_min_rec`={0.0511, 0.0000, 0.0448, 0.0000}, fronts={21,93,189,189})
  — NOT the clean 1.0-at-all-depths result cited in N2.1. That clean result
  is specifically the `NCR_ORTHO_WRITE.md` §9 wave's **4× (320,000-step)
  EXTENDED-budget** cell. **"Extended budget" is therefore a load-bearing
  qualifier of the cited verdict, not a detail** — a build agent porting
  this to the real-LM Phase-1/2/3 token budgets (327.68M-1.108B tokens/cell,
  §6.2) must not assume the reduced-length calibration run automatically
  inherits the 4×-toy-harness far-depth quality; §11.4's own open item (4),
  "K=24's far-depth reliability... an unexplained variance this wave does
  not resolve," stays unresolved.

**Consequently: GATE-3 (Phase-0/1 calibration, ~22 GPU-h) STAYS fully
load-bearing, unweakened by GATE-1's discharge.** The NCR head has NEVER
trained inside a real LM — every cited free-write K=24 number above is
from the isolated synthetic Task-E harness (`NCR_ORTHO_WRITE.md`), not from
a bind-clause-in-real-text training signal. §9.1's own text already names
this exact risk for a different branch and it applies unchanged here: "the
write-conditioning problem... is not guaranteed to manifest identically
once bind clauses are embedded in a real-LM training signal (a new confound
this document does not resolve and must not assume away)" (line
1999-2003). GATE-3 is the make-or-break de-risk for THIS confound,
independent of and downstream from GATE-1's now-discharged ortho-vs-free
question — it must run, unconditionally, before any scale spend. **The
demonstrated capability envelope is bounded honestly: K≤24 at `d=K+1`, with
extended (4×) budget, on the synthetic harness — documented as a boundary,
not a floor to build past without re-verification.** No claim in this
document extends the free-write finding to K>24, to the 1×-budget-
equivalent calibration length, or to the real-LM training signal ahead of
GATE-3's own readout.

### N2.5 SCALE-TARGET extension (PI-ratified)

The flagship ladder's TOP is amended from 392M to **1B+ parameters on the
capability** — 392M becomes a checkpoint, not the finish line. Rationale
(PI, `STATE.md` 2026-07-17 briefing): the field's credibility bar for "real
architecture, not a toy" sits near 1B; a clean NCR separation at 392M is a
strong result, but the honest ambition given the grant window (≈192 GPU-h/
day operative rate, CLAUDE.md Hardware) is to reach it — a ~1000-2000 GPU-h
1B run is an order of magnitude smaller than the remaining multi-thousand-
GPU-h window at that daily rate, and does not displace the committed
484-608 GPU-h §6.2 ledger or the ~300 GPU-h fix-at-scale/other campaigns
already in flight.

**Recording discipline, binding:** the 1B rung is recorded here as **INTENT
ONLY**. It is NOT a committed spec — no `(K,d,n_layers,d_model)`
configuration, token budget, or GPU-h ceiling is pinned by this amendment.
Before any 1B GPU-h is spent, the 1B rung requires its OWN: (i) pre-launch
attack round, (ii) resource/placement red-team, and (iii) its own priced
ledger, sequenced exactly like every other scale step in this document's
own §6.2 discipline ("every number is either cited from a measured rate...
or marked PRICE-UNKNOWN/PROJECTED. No rate is guessed," line 1307-1309) —
a 1B config has ZERO measured rate anywhere in this repo today, and per
CLAUDE.md's own ceremony tiers (">50 GPU-h or publication-bound → full
multi-round adversarial gauntlet") a ~1000-2000 GPU-h run requires the
FULL gauntlet, not the lighter 10-50 GPU-h tier this document's own Phase
1-3 cells mostly clear. **This escalation is gated on the 392M rung
holding** (§7's own WIN/PARTIAL bands, unaltered by this amendment).

**The sequence, stated explicitly (supersedes no frozen text — §6.2's
Phase 0-3 structure already ends at 392M; this appends the next rung, not
committed):**

`GATE-3 calibration (K=24, 98M, ~22 GPU-h, Phase 0+1) → 98M main (Phase 2,
≈107.7-215.3 GPU-h) → 392M main (Phase 3, ≈113.3-226.7 GPU-h) → [1B's own
ceremony gate — attack + red-team + ledger, not yet run] → 1B+.`

The immediate committed action, and the only one this amendment authorizes
GPU-h for, is **GATE-3 at 98M.**

### N2.6 Novelty check

**This amendment does NOT trigger a fresh novelty re-verification sweep.**
Per the novelty-reverification-gate doctrine (a re-check is required at a
CLAIM PIVOT), this is not one: §N1's three sweeps (2026-07-16) already
adjudicated the claim this document makes — "(i) a BY-CONSTRUCTION
exactness guarantee... and (ii) a query-time O(log h) sequential access
complexity" (line 3428-3436) — and that claim is UNCHANGED by moving from
K=32/d=33 to K=24/d=25. Sweep 1 (BY-MECHANISM, line 3355) searched the
mechanism combination, not a specific K; sweep 2 (BY-TASK, line 3374)
searched the S₅ separation SHAPE, which N2.2(b) shows is K/d-invariant by
construction (`ρ_{S₅}(g)⊕I_{d−4}` at ANY `d≥4`); sweep 3 (INTERNAL ARCHIVE,
line 3394) is unaffected (K=24 was already an in-repo cell, not a novel
internal configuration). **K and parameter-scale are not axes that move the
competitive field** — no located prior-art distinction in §N1's Tier-1/2/3
citation list (line 3590-3630) turns on the NCR head's specific `d` or the
backbone's specific parameter count; MuonSSM, FWM, DeltaProduct, Log-Linear
Attention, Yau et al., and every other must-cite work is differentiated on
MECHANISM (rank-1 vs full-rank orthogonalization, expressivity-existence
vs trained-exactness, parallel-scan-throughput vs query-time-sequential-
depth), never on K or scale. The §N1 verdict stands: **GATE-2's novelty
precondition remains DISCHARGED** (§N1's own DISCHARGE, line 3644-3658);
this amendment changes GATE-1's mechanism (ortho→free) and the flagship's
scale target, neither of which §N1 adjudicated or needs to re-adjudicate.

### N2.7 Resistance points flagged for the coordinator (not resolved by this amendment)

Surfaced per this amendment's own instruction, not silently absorbed:

1. **GATE-2 (bridge cell, §9.2/§6.2 Phase 0b) is UNTOUCHED by this
   amendment and is now internally in tension with N2.3's headline
   reframe.** GATE-2 as frozen trains an NS-polar orthogonal write on S₅
   generators (`ρ_{S₅}(g)⊕I_{d_ncr−4}` — the SAME target object N2.2(b)
   re-derives for free-write) — but §9/§10's mechanism finding applies to
   the NS-polar projection itself, not to K=32 specifically, and both
   `ortho_K24` cells in §9.1's own table ALSO failed Gate-0 4/4 (identical
   dip-then-collapse signature). There is no principled reason to expect
   GATE-2's bridge cell — an NS-polar write on a DIFFERENT full-rank
   orthogonal target at the identical tight-spare `d_ncr` — to escape the
   same absorbing ill-conditioning trap §10 diagnoses. **Axis A/Task 2
   therefore remains gated behind a bridge cell built on a mechanism this
   amendment's own evidence predicts will FAIL**, unless GATE-2 receives
   its own amendment (a free-write analog of the bridge cell — untested,
   not built, not priced here). Until that happens, this amendment
   discharges GATE-1 (unblocking Axis B/Task 1 and Task 3, which inherit
   Task 1's abelian construction) but does NOT unblock Axis A/Task 2's
   Phase-1 calibration arm. **Concretely: of GATE-3's ~22 GPU-h (Phase 0
   ≈2 GPU-h + Phase 1 10.76/21.52 GPU-h at 1×/2×), only Phase 0 + the
   Task-1 half of Phase 1 (≈7.4-12.8 GPU-h) is actually unblocked by THIS
   amendment; the Task-2 half (≈5.38/10.76 GPU-h) stays behind GATE-2
   pending its own reconciliation.** The coordinator should decide whether
   GATE-3 launches Task-1-only now or waits for a GATE-2 amendment to
   launch both arms together.
2. **§N1 R2's eval-grid extension (line 3479-3511) pins its added OOD
   rungs `{60,100,160}` to "the GATE-1-measured exactness horizon H"** — a
   quantity §N1 expected ortho-write to supply. Under this amendment, H
   must instead come from the free-write ladder (N2.3 re-anchors it to
   h≈61, §9.1's own audited rung) — a build agent implementing §N1 R2
   needs this substitution stated explicitly, which this amendment does
   (N2.3) but §N1's own text (untouched, per append-only) still reads as
   if ortho-write will supply H.
3. **§4.3's KV-cache-memory-matched baseline (Case i/ii, line 1034-1068)
   computes `state_bytes` from `d_ncr=33` explicitly** (`33²×4=4,356` bytes
   Task 1; `3×33²×4=13,068` bytes Task 2) — these numbers move under
   `d_ncr=25` (`25²×4=2,500`; `3×25²×4=7,500`) and the re-derived `M`-grids
   (line 1046, 1063) move with them. This is NOT one of the three
   re-derivations this amendment was scoped to (N2.2), and is NOT
   recomputed here — flagged so a build agent does not silently carry
   forward the d=33 grid.

### N2.8 COORDINATOR ADJUDICATION of N2.7 (Fable, verified vs raw text 2026-07-17)

Accept §N2 (clean append, 3 re-derivations verified vs §3.2's own
formulas). Ruling on the N2.7 flags:

**N2.7-item-1 (the load-bearing one) — PARTIALLY OVERTURNED, split the
launch.** The agent's claim "no principled reason to expect GATE-2's
bridge cell to escape the §10 trap" is too strong. §10's trap requires an
UNCONSTRAINED direction whose σ_min random-walks to 0 (the scale-invariant
loss puts zero pressure on it). Task-1's tight-spare write HAS exactly one
such direction (the d−K=1 spare, never queried) → §9 FAIL confirmed.
Task-2's target `ρ_{S₅}(g)⊕I_{21}` is FULL-RANK (all σ=1, the M7 fix) —
whether the trap forms depends on whether Task-2's training loss
CONSTRAINS the I_{21} padding block or the read ignores it. That is
UNTESTED and mechanistically distinct from §9; the §9 FAIL does NOT
cleanly transfer, and neither does a guarantee of success. **So Task-2 is
neither auto-unblocked (agent's over-optimism the free-write reframe would
imply) nor doomed (agent's over-pessimism).**

**DECISION — GATE-3 launches in TWO WAVES:**
- **WAVE 1 (NOW): Phase-0 + Task-1 only** (≈7.4-12.8 GPU-h). Cleanly
  unblocked by §N2 (free-write K=24 = §9 healthy verdict). De-risks the
  make-or-break question — does the NCR head train inside a real 98M LM
  AT ALL — on the axis that's actually ready. Task-1/Axis-B (O(log h)
  exact composition, the access-complexity separation) is ALSO the
  cleaner headline per `research/ncr_separation_grounding.md` Part 3
  (the O(log h) read is THE load-bearing novelty; length-gen was demoted
  to disclosed-genre corroboration). Leading with it at scale is
  on-strategy, not a consolation.
- **WAVE 2 (after a cheap Task-2 write resolution): Task-2 calibration.**
  Resolve Task-2's write BEFORE committing its ~5.4-10.8 GPU-h: first a
  ~0-GPU paper check of whether Task-2's loss constrains the I_{21} block
  (reading the read/loss code — if constrained, §10 trap cannot form and
  the NS-polar bridge cell is expected-safe; if not, the trap risk is
  live); then the ≈3.18 GPU-h bridge cell as the empirical test. Candidate
  write mechanisms if NS-polar traps: free-write (§9 shows it's naturally
  cond-1.0 — but inexact), or expm/Cayley (note the det-parity subtlety —
  ρ_{S₅}(transposition) has det=−1 in the 4-dim rep, so plain expm∈SO(d)
  can't hit odd generators without a reflection; same issue the fallback
  gauntlet §A1 surfaced). This is a real open design fork, recorded as
  such — not hand-waved.

**N2.7-item-2 (H re-anchor) — ACCEPTED**, §N2.3's re-anchor of H to the
free-write ladder (h≈61) stands; a build agent follows §N2.3 not §N1's
stale ortho-write H.
**N2.7-item-3 (§4.3 KV-cache d=25 bytes) — ACCEPTED as a build-time
TODO**: state_bytes 4356/13068 → 2500/7500, M-grids re-derive; not a
Wave-1 blocker (Wave-1 is calibration, not the KV-matched baseline);
must be fixed before the Phase-2 memory-matched comparison.

GATE-1 DISCHARGED for Task-1/Axis-B + Task-3. GATE-2 (Task-2) stays live,
re-scoped to the Wave-2 write-resolution above.

---

## §G3-B1 WAVE-1 BUILD RECORD (2026-07-17, build agent — Phase-0 + Task-1 only)

**Scope discharge check (done first, per this section's own dependency).**
`git log --oneline | grep f9414ff` confirms commit f9414ff ("NCR flagship
§N2 GATE-1 AMENDMENT... GATE-3 SPLIT: Wave-1 NOW = Phase-0 + Task-1") is
in history; §N2.1/§N2.8 above are the ratified text. No duplicate build
found (`find . -iname "*ncr_lm*" -o -iname "*ncr_real*" -o -iname
"*gate3*"` returned only this design doc before this section was appended;
no prior `experiment-runs/2026-07-17_ncr_gate3_wave1/` directory existed).

### Integration surface (identified BEFORE building, per the build brief)

**Confirmed from-scratch graft — no existing code combines the NCR module
with the DeltaNet LM backbone anywhere in this repo.**
`grep -rl "DeltaNetLM\|lm_pretrain_rd" matrix-thinking/ncr/` and
`grep -rl "ncr_earlyln_scale\|ncr_models\|BindingEncoder\|ncr_ortho_write"
matrix-thinking/deltanet_rd/` both return empty. The two halves are
mechanically well-specified in isolation:

- **Backbone**: `DeltaNetLM(vocab_size=50257, d_model=768, d_state=64,
  n_layers=12, conv_size=4, num_heads=1, ffn_mult=4)` —
  `matrix-thinking/deltanet_rd/lm_pretrain_rd.py`, rung-1 config
  (`lm_rd_rung_configs.py RUNGS[1]`). `forward(token_ids,
  return_hidden=True)` returns the POST-`norm_f` hidden state
  `(B,T,768)` — this IS §2.2's own tap point ("a side-channel module
  attached after the backbone's final layer, reading the backbone's own
  hidden states at bind-clause/query positions," line 388-389).
- **NCR head**: `ncr_earlyln_scale.NCREarlyLNModel(d=25, h=64)` — the
  FREE-WRITE arm (`ncr_ortho_write.py build_primary_model("free", ...)`
  returns this class verbatim), K=24/d_ncr=25 per §N2.2(a). No NS-polar
  projection anywhere (§N2.1: "No ortho-write mechanism is used anywhere
  downstream" — the NS-polar pipeline is retired from this build).
  `.encode(keys, values) -> Z (B,25,25)`; read via the module-level pure
  function `ncr_models.binexp_read(Z, q, h)`.

**Three integration decisions this design leaves genuinely unresolved
(none invented by this build — see the module docstring of the harness
below for the full citation trail):**

1. **Write-side adapter.** §2.2 (line 388-389) pins WHERE the taps come
   from (post-backbone-final-layer hidden states, at bind-clause/query
   positions) but not the projection architecture that turns a 768-dim
   hidden state into BindingEncoder's expected 25-dim key/value vector —
   dimension, shared-vs-separate key/value adapters, and whether these
   adapter params are counted in the "175K/173K-param NCR head" budget
   (§2.1's `P(d,h)` formula counts BindingEncoder's own internal params
   only) are all absent from the text.
2. **Read injection.** §2.1 (line 284-288) states VERBATIM: *"Either read
   injects its output o into the residual stream at the query position
   (e.g. added to the Transformer's own hidden state before the final LM
   head, or consumed by a small MLP that produces logits directly for the
   query token — build-time decision, not resolved here)."* Explicitly,
   textually unresolved by the frozen design's own authors, not a gap
   this build introduced.
3. **Training-loss / `recovered_frac@0.9` scoring protocol on real text.**
   §7's bands reuse the `recovered_frac@0.9` cosine-recovery metric
   convention (`NOVEL_ARCH_WATERFALL.md` §3.2a), which presupposes a
   *target vector* to score cosine similarity against — but no target
   construction (frozen per-entity table, à la
   `matrix-thinking/deltanet_rd/probe_head_rd.py`'s H2H-established
   `build_probe_target_table`/`cosine_recovery_frac`/`probe_aux_loss`
   pattern, vs. some other scheme), no CE-vs-aux loss weighting, and no
   schedule is pinned anywhere in this document for Task 1's real-LM
   setting.

Per the build brief's own instruction ("If the integration is
underspecified in the design, STOP and report the specific gap; do not
invent it"), this build does **NOT** wire a spec-authorized end-to-end
graft. It builds and real-CUDA-smokes the two well-specified components
independently, PLUS one clearly-labeled, non-authoritative placeholder
instantiation of the three open decisions — built solely to prove the
mechanical plumbing (shapes/dtypes/gradients/memory co-residency) is
viable, never to be read as the coordinator's pick.

### What was built

One new file, additive only — **zero edits to any existing file**
(`ncr_earlyln_scale.py`, `ncr_models.py`, `lm_pretrain_rd.py` all reused
verbatim, unmodified):

`experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py` (≈420
lines) — builds the 98M backbone and the K=24/d=25 free-write NCR head,
verifies both param counts against §N2.2(c)'s exact formulas, and runs a
9-item real-CUDA smoke suite. Contains `class PlaceholderIntegrationGlue`
implementing ONE precedented instantiation of gaps (1)-(3) above (linear
768↔25 adapters mirroring `probe_head_rd.py`'s `build_adapter_arm`;
additive-residual injection, the first of §2.1's two disclosed options;
joint CE + cosine-aux-probe loss mirroring `probe_head_rd.py`'s
`joint_loss`/`AUX_WEIGHT_DEFAULT=0.1`) — every non-spec choice is
commented in-place with its precedent and its "NOT ratified" status.

**Base file md5s (verbatim reuse, box vs. local repo — confirmed
byte-identical):**

| File | md5 |
|---|---|
| `matrix-thinking/ncr/ncr_earlyln_scale.py` | `3a87fcc92bb8341203c5e8c1f039a0af` |
| `matrix-thinking/ncr/ncr_models.py` | `6d7b30a592bee11f6c2135165801742d` |
| `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` | `34addd9d8cc6a3df5a367d0f18a2ee0e` |
| `experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py` (new, box copy) | `1e8811d0ced020d154250b98c5401242` |

### Real-CUDA smoke results (box `youthful-indigo-turkey`, GPU 7, 2026-07-17)

Ran twice: a CPU param-count-only pre-check, then the full CUDA suite.
**9/9 items PASS, 0 FAILURES, wall-clock 10.9s.** Full JSON:
`experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_results.json`.

| # | Item | Result |
|---|---|---|
| 0a | Backbone param count vs. 98M target (15% tol) | PASS — 97,618,176 (rel_err 0.39%) |
| 0b | NCR head param count vs. §N2.2(c) formula (exact) | PASS — 173,209 == 173,209 |
| 1 | Backbone forward+backward, finite grads | PASS — loss=11.0006, grad_norm=18.08 |
| 2 | Backbone checkpoint save→load→forward | PASS — bit-identical, max_abs_diff=0.00e+00 |
| 3 | Backbone eval batch, B=32/T=512 (Phase-1 op point), no_grad | PASS — peak 9.52 GB |
| 4 | NCR head (K=24,d=25) forward+backward, finite grads | PASS — loss=0.9675 |
| 5 | NCR head checkpoint round-trip + `binexp_read` determinism | PASS |
| 6 | `binexp_read` finite at ladder h∈{5,12,20,29,40,61} | PASS — all finite |
| 7 | **Co-residency**: backbone+NCR+placeholder-glue, ONE joint fwd/bwd/opt-step | PASS — loss_ce=10.98, loss_aux=1.00, 208 params w/ grad, peak 2.14 GB |

Smoke item 7 originally crashed on a shape bug in the harness's OWN
placeholder aux-loss call (`entity_ids` shaped `(B,3)` against an
`ncr_pred` shaped `(B,24,25)`) — fixed (entity_ids now `(B,K)`), re-run,
passed. This was a harness bug, not a backbone/NCR-head defect.

### Pre-launch red-team (answers recorded)

1. **Memory fit.** 98M backbone alone, full training op point (B=32/
   T=512): design's own §6.1 anchor is 23.5GB/80GB (measured, uncontended,
   full training incl. optimizer state). This build's own eval-only
   no_grad measurement at the same shape: 9.52GB peak (no optimizer
   states, explaining the gap). NCR head adds 173,209 params (0.18% of
   backbone) plus a K=24×d=25 activation footprint — negligible next to a
   (32,512,768) backbone activation tensor; expected total training
   footprint ≈23.5-24GB, consistent with §2.1's own "cost is governed by
   the encoder's own width [64], not the written object's dimension [25]"
   finding. Every one of the 8 GPUs currently carries ~44-45GB of existing
   production jobs, leaving ~36-37GB free — a 24GB Wave-1 cell fits
   comfortably in that headroom (44+24=68GB < 80GB), matching this box's
   own standing 1-job-plus-headroom packing pattern (CLAUDE.md
   saturation-packing).
2. **Measured-rate timeout/ceiling.** §6.1's 0.236 s/step (98M) is an
   UNCONTENDED rate. All 8 GPUs are currently production-busy (86-100%
   util, confirmed below) — per the operating doctrine's own measured
   cautionary precedent ("Stage-0 measured 3.3× slowdown contended"), a
   Wave-1 cell packed onto a currently-busy GPU should be PRICED at
   ≈3.3× wall-clock, not the bare uncontended rate: Phase-0 (~2 GPU-h
   uncontended) → ≈6.6h wall-clock contended; Phase-1 Task-1 arm (5.38
   GPU-h at 1×, 4 cells) → ≈17.75 GPU-h-equivalent wall-clock contended
   if serialized on one GPU, or ≈4.5-4.6h per cell if the 4 cells run in
   parallel across 4 separate (each individually contended) GPUs. Any
   real launch's `--internal-timeout` must be set well above the
   contended estimate, not the uncontended one (production jobs on this
   box already set 36,000-86,400s timeouts for their own, larger 392M
   cells — the same discipline applies here).
3. **Placement.** `nvidia-smi` (both before and after this smoke run):
   **all 8 GPUs are production-busy**, 86-100% util, 43.5-45.1GB/80GB
   used each (fixscale 392M seed-ext waves + queue workers, confirmed via
   `ps aux` — job names `fixscale_seedext_arm_*`, `fixscale_98m_*`,
   `queue_worker_g{0-7}` tmux sessions). **No drained GPU exists.** This
   build's own smoke ran on GPU 7 (37,533 MiB free, the most headroom of
   the 8) in a brief, ~11-second slot; `nvidia-smi` before/after confirms
   memory returned exactly to the pre-smoke baseline (43,549 MiB used)
   and the compute-process count on the box is unchanged (9 rows,
   including the CSV header, i.e. 8 processes) — no production job was
   disturbed. A real Phase-0/Phase-1 launch must price the §item-2
   contention factor above rather than assume an idle GPU.
4. **Duplicate check.** No pre-existing Wave-1 harness, no pre-existing
   `experiment-runs/2026-07-17_ncr_gate3_wave1/` directory, no prior
   `§G3-B1`/"BUILD RECORD" section in this file — confirmed by grep
   before this section was written (see the dependency check above).
5. **Gate-discharge check.** `f9414ff` is in `git log`; its subject and
   this document's own §N2.1/§N2.8 text agree — GATE-1 is discharged for
   Task-1/Axis-B (Wave-1's scope), GATE-2/Task-2 stays live and is
   correctly OUT of this build's scope (Wave-1 = Phase-0 + Task-1 only,
   per the build brief).

### Launch command — NOT RUNNABLE YET (blocked on the 3 gaps above)

No honest "exact command" exists for the NCR-augmented Phase-0/Phase-1
training cells, because the training SCRIPT implementing gaps (1)-(3) is
itself the blocked artifact — writing it would mean inventing the
architecture this section exists to flag. The shape it would take, once
the coordinator resolves the 3 gaps, mirrors `lm_pretrain_rd.py`'s own
existing CLI convention plus new NCR-specific flags this build would add:

```
python3 lm_pretrain_rd.py \
  --corpus openr1-mix-ext --data-dir /data/deltanet_rd_data \
  --d-model 768 --d-state 64 --n-layers 12 --seq-len 512 --batch-size 32 \
  --steps 20000 --ckpt-every 1000 --seed <n> \
  --internal-timeout <>= contended estimate above, item 2, NOT the uncontended rate> \
  --ncr-active --ncr-d 25 --ncr-h 64 --ncr-k 24 --ncr-write-arm free \
  --ncr-adapter <UNRESOLVED, gap 1> --ncr-inject <UNRESOLVED, gap 2> \
  --ncr-aux-weight <UNRESOLVED, gap 3> \
  --ckpt-dir /data/ncr_wave1_ckpts/... --out results/ncr_wave1/...
```

Flags after `--ncr-write-arm free` do not exist in any script today and
are placeholders for what the resolved design would need.

### Archive

Harness + results archived at
`experiment-runs/2026-07-17_ncr_gate3_wave1/` (`ncr_lm_wave1_smoke.py`,
`ncr_lm_wave1_results.json`), per `CLAUDE.md`'s reproducibility rule.

### Readiness verdict

**BLOCKED-with-gap on the end-to-end integration** (the three items
above — write-side adapter, read injection, training-loss protocol —
none pinned by the frozen+amended spec, one explicitly self-flagged as
"not resolved here"). **NOT BLOCKED at the component level**: the 98M
rung-1 backbone and the K=24/d_ncr=25 free-write NCR head are each
independently real-CUDA-verified (forward, backward, finite grads,
checkpoint save/resume, eval batch) and mechanically co-resident on one
GPU with headroom to spare. The coordinator's two options: (a) resolve
gaps (1)-(3) explicitly (a short adjudication, not a rebuild — the
components are ready) and re-dispatch a build-continuation agent to wire
the ratified choices, replacing `PlaceholderIntegrationGlue`; or (b)
ratify this build's placeholder as a provisional Phase-0 architecture
outright, disclosed as such in any resulting write-up. Either way, GPU
spend on the full Phase-0/Phase-1 training run stays gated on that
decision plus the coordinator's own independent audit, per the build
brief.

## §G3-B2 COORDINATOR ADJUDICATION of the Wave-1 integration gaps (Fable, verified vs §2.1/§3.2/§6.2 text, 2026-07-17)

The §G3-B1 build correctly STOPPED rather than invent architecture. On
reading the design directly, the "three unspecified gaps" reduce to TWO
bounded adapter decisions — the design already pins the rest:
- **Task/data:** the `grammar_rd.py` bind-clause grammar (`<buf...> KEY
  <rel> VALUE .`) rendered through the real GPT-2 tokenizer, interleaved
  into real-corpus (WikiText/OpenR1) documents, autoregressive per-token
  stream (§3.2, lines 578-583, 479). SPECIFIED.
- **Tap points:** operators written at bind-clause positions, reads at
  query positions, off the post-`norm_f` hidden (§2.1 lines 250, 386-389,
  §2.2). SPECIFIED.
- **Objective:** next-token CE on the query-answer tokens within the
  autoregressive stream; `recovered_frac@0.9` is the EVAL metric, not the
  train loss. SPECIFIED.

**RULING — the two remaining adapters, minimal calibration-scoped defaults:**
1. **Write adapter (768→d_ncr):** a learned `Linear(d_model=768 →
   encoder-input-dim)` applied to the post-norm hidden at each bind-clause's
   KEY and VALUE token positions, feeding the BindingEncoder's key/value
   inputs at their ACTUAL signature (the build agent matches the real
   `NCREarlyLNModel`/`BindingEncoder` input shapes — do not guess shapes).
   One Linear per role (key, value). Rationale: fewest degrees of freedom
   → fewest ways for a bad adapter to confound the signal.
2. **Read injection:** the read output `o ∈ R^{d_ncr}` (from `binexp_read`
   at the query position — Wave-1 is Task-1/abelian ONLY) → a learned
   `Linear(d_ncr → 768)`, ADDED to the query-position post-norm hidden
   before the SHARED LM head (design §2.1's option (a), not the MLP→logits
   option (b)). Rationale: directly tests whether the composed read is
   usable by the backbone's own head; no separate output pathway to
   confound.

**DISCLOSURE + FAIL-INFORMATIVENESS (the Stage-0 lesson, binding):** these
are CALIBRATION defaults, NOT the frozen flagship architecture. GATE-3
Wave-1's job is BINARY — does the NCR head train through a real 98M
DeltaNet backbone at all. A **PASS** validates the graft is trainable
(final architecture refined afterward). A **FAIL is NOT conclusive** until
a wiring ablation (MLP write-adapter; the alternative read-injection
option (b); teacher-forced-operator control that isolates read-vs-write)
rules out a wiring artifact — precedent: §B4/§B6, where a guard/wiring
defect masqueraded as a mechanism FAIL and only an independent audit
caught it. The build-continuation records these ablation arms as
pre-authorized so a FAIL routes to them, not to "NCR can't train in a real
LM."

**Wave-1 scope:** Task-1/Axis-B only (abelian, `binexp_read`); Phase-0
timing (the PRICE-UNKNOWN real-corpus rate, §6.2) + a small Phase-1
calibration cell at short horizons — NOT the full length-generalization
sweep. NEXT: build-continuation replaces §G3-B1's `PlaceholderIntegrationGlue`
with this ratified wiring → mandatory independent audit (correctness AND
wiring-bias/FAIL-informativeness) BEFORE any GPU spend → coordinator
red-team → blind launch.

## §G3-B3 WAVE-1 BUILD-CONTINUATION (2026-07-17, build agent — replaces
`PlaceholderIntegrationGlue` with the §G3-B2 ratified wiring)

**Scope discharge check.** §G3-B2 (above) is the ratified adjudication this
section builds against; no duplicate build found (`find . -iname
"*ncr_lm_wave1*"` returns only §G3-B1's existing
`experiment-runs/2026-07-17_ncr_gate3_wave1/` — this build edits that
directory's script IN PLACE, additive to the design doc, no new directory).

### What was implemented

**One file, edited in place** (the §G3-B1 build's own path, per the build
brief's "REPLACE" instruction):
`experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py` (≈700
lines, up from ≈420). `ncr_earlyln_scale.py`/`ncr_models.py`/
`lm_pretrain_rd.py` remain reused verbatim, unmodified (md5s below,
byte-identical to §G3-B1's own table); this build adds ONE new verbatim
dependency, `grammar_rd.py` (the real bind-clause grammar generator §G3-B1
never invoked — its own smoke used random tensors, not real task data).

**`class NCRIntegration`** replaces `PlaceholderIntegrationGlue` — a real,
gradient-trainable module, not a smoke-only stand-in:

1. **Write adapter (RULING 1, exact shapes).** `key_adapter =
   Linear(768, 25, bias=False)`, `value_adapter = Linear(768, 25,
   bias=False)` — one per role, applied to the post-norm hidden at each
   bind-clause's KEY/VALUE token positions (`grammar_rd.sample_batch_rd`'s
   own `item_pos - 2` / `item_pos`). Shape matched by DIRECT INSPECTION of
   `matrix-thinking/chapter2/model_v4.py BindingEncoder.__init__`
   (`self.in_proj = nn.Linear(2*d, h)`, `forward(self, keys, values)` with
   keys/values: `(B,K,d)`) — encoder-input-dim IS `d_ncr=25` exactly, NOT
   ambiguous. These are the IDENTICAL shapes §G3-B1's own placeholder
   guessed; what changed is everything downstream (real data, real
   objective, real metric, ablation flags — see below).
2. **Read injection (RULING 2, exact shape).** `read_injector =
   Linear(25, 768, bias=False)`, output ADDED to the query's own `<Q>`
   marker-position post-norm hidden, before the shared (tied) LM head —
   design §2.1 option (a) verbatim. The disclosed option (b) (MLP→vocab
   logits) is ALSO built, flag-gated (`--read-inject mlp_logits`), ADDING
   to (not replacing) the base LM-head logits — a build-time
   interpretation, disclosed in the script's own module docstring item
   (iii), never the default.
3. **Param delta, disclosed SEPARATELY from the NCR head's own 173,209
   (§G3-B1 gap 1's open question, now answered by disclosure, not
   silently folded in):** `NCRIntegration` (linear/add, the ratified
   default) = 2×768×25 (adapters) + 25×768 (read_injector) = **57,600
   params exactly** (smoke item 0c asserts this exactly, not a tolerance
   band). Total NCR-related addition to the 97,618,176-param 98M backbone:
   173,209 + 57,600 = 230,809 params (**+0.236%**) — still negligible,
   moved from §2.1's own "+0.18%" (NCR head alone) by the adapters'
   inclusion, a number that document never separately priced.

**Ablation flags pre-wired (§G3-B2's FAIL-informativeness list), all
CODE-VERIFIED this wave (not merely declared):**
- `--adapter {linear,mlp}`: `mlp` = `Linear(768,192)→GELU→Linear(192,25)`
  per role (width `d_model//4`, a build-time convention, not pinned by the
  design — item (iii)). Verified by CONSTRUCTION + shape/finite forward
  (smoke item 11, CPU-fast) — never run through the full backbone/CE/
  backward pipeline this wave, per the build brief.
- `--read-inject {add,mlp_logits}`: `mlp_logits` =
  `Linear(25,100)→GELU→Linear(100,vocab_size)` (width `4*d_ncr`, same
  convention discipline). Same verification level as `--adapter`.
- `--teacher-force-operator`: a closed-form least-squares operator fit,
  `Z_teacher^T = pinv(keys_v.detach()) @ values_v.detach()`, bypassing
  `ncr_head`'s own `BindingEncoder` ENTIRELY (its parameters never enter
  the autograd graph) — isolates "can read+inject+head consume a
  perfect-for-the-current-adapters operator" from "can the encoder learn a
  good operator" (module docstring item (ii), a build-time mechanism
  choice, not literal design text). UNLIKE the other two flags, this one
  IS real-CUDA-verified end-to-end this wave (smoke item 10): closed-form
  fit residual `7.30e-06` (near-exact, K=24<d=25 generically consistent
  system), `ncr_head` parameters receive **ZERO** gradient
  (`ncr_untouched=True`), while `backbone`/`key_adapter`/`read_injector`
  DO train (`backbone_trained=key_adapter_trained=read_injector_trained=
  True`) — the isolation property holds exactly as designed.

**DISCOVERED WIRING GAP (mechanical, singularly-determined, fixed and
disclosed — not a STOP-worthy ambiguity, not silently invented):**
`grammar_rd.py` mints TWO reserved token ids beyond GPT-2's real 50,257
vocabulary (`BUFFER=50257`, `<Q>=50258` — its own module docstring). A
`DeltaNetLM` built at the design's own pinned `vocab_size=50257` (§2.2) has
NO embedding/head rows for these ids — feeding it a real grammar_rd
document crashed with a CUDA out-of-bounds embedding-gather assert (hit
and confirmed during this build's first on-box run, full traceback
archived in the build session). Fix: every grammar_rd-dependent smoke item
(7-10) builds `DeltaNetLM(vocab_size=pools.vocab_size_total=50259, ...)`
instead — +2 embedding/tied-head rows, **+1,536 params (0.0016% of the
backbone)**, inside every existing tolerance; smoke items 0-6 (component-
only, no grammar_rd) are UNCHANGED at `vocab_size=50257`. ONE DISCLOSED
DEVIATION from `grammar_rd.py`'s own convention: that module's docstring
also states the BUFFER row is "zero-pinned AND frozen... (R2-3 — see
`model_rd.py`)" — a property of a DIFFERENT model class
(`DELTANET_REALDATA_DESIGN.md`'s own `model_rd.py`), never invoked by THIS
design (which names `lm_pretrain_rd.DeltaNetLM`, a plain vanilla-embedding
LM, as its backbone, §2.2). This build does NOT replicate the zero-pin/
freeze (`DeltaNetLM` has no such mechanism; adding one is a genuine
architecture change, out of scope) — flagged for audit/coordinator: does
this deviation matter, given NCR's write path never uses the causal-rank
design's own beta-mask mechanism that motivated R2-3 in the first place.

**Other build-time interpretations disclosed (not literal design text,
full rationale in the script's own module docstring):**
- **`recovered_frac@0.9`'s eval-only target (item (i)):** SELF-consistent
  — `target = key_adapter(hidden at the answer entity's OWN bind-clause
  KEY position)`, mirroring `ncr_models.py`'s own `query_keys`-same-space-
  as-`keys` convention (no external frozen probe table invented; one was
  considered — mirroring `probe_head_rd.py`'s `build_probe_target_table`
  — and rejected, since the CE-only training objective gives `o` no
  reason to align with an EXTERNAL target, which would make the metric
  read near-zero regardless of graft quality).
- **Document construction:** ONE bind episode (K=24) + ONE query clause
  per synthetic document (§3.1's own "1-2 bind episodes... plus a query
  clause," the minimal instance); `n_query=1` (avoids drawing/discarding
  23 unused queries per row). Real+synthetic BATCH-level mixing (§5.2
  Option 1) is NOT built this wave — out of the "BUILD+SMOKE, do not
  launch" scope; the production trainer that adds it is the documented
  next step (launch command below).
- **Read mechanism uniformity:** `binexp_read` used for BOTH the h=2
  train-range smoke (item 7) AND the h=5 held-out eval smoke (item 9) —
  NOT the isolated synthetic harness's train(masked-compose)/eval
  (binexp_read) split, since §2.1 line 279 pins `binexp_read` as Task 1's
  read mechanism unconditionally and it is exact+differentiable at every
  h≥1. `ncr_head.encode()` is called directly (bypassing `forward()`'s
  masked-compose path), the same precedent §G3-B1's own smoke_6
  established.

### Real-CUDA smoke results (box `youthful-indigo-turkey`, GPU 7, 2026-07-17)

Ran twice: a CPU pre-check (param counts + ablation-flag construction,
1.8s), then the full CUDA suite. **11/11 items PASS, 0 FAILURES,
wall-clock 14.6s.** Full JSON:
`experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_results.json`.

| # | Item | Result |
|---|---|---|
| 0a | Backbone param count vs. 98M target (15% tol) | PASS — 97,618,176 (rel_err 0.39%) |
| 0b | NCR head param count vs. §N2.2(c) formula (exact) | PASS — 173,209 == 173,209 |
| 0c | NCRIntegration (linear/add) param count (exact) | PASS — 57,600 == 57,600 |
| 1 | Backbone forward+backward, finite grads | PASS — loss=11.0006, grad_norm=18.08 |
| 2 | Backbone checkpoint save→load→forward | PASS — bit-identical, max_abs_diff=0.00e+00 |
| 3 | Backbone eval batch, B=32/T=512 (Phase-1 op point), no_grad | PASS — peak 9.52 GB |
| 4 | NCR head (K=24,d=25) forward+backward, finite grads | PASS — loss=0.9675 |
| 5 | NCR head checkpoint round-trip + `binexp_read` determinism | PASS |
| 6 | `binexp_read` finite at ladder h∈{5,12,20,29,40,61} | PASS — all finite |
| 7 | **THE make-or-break: full graft** (backbone→write-adapter→encoder→ `binexp_read`→read-inject→LM-head→CE) on a REAL `grammar_rd` Task-1 document (K=24 clauses + 1 query, h=2 in-dist), ONE joint fwd/bwd/opt-step | PASS — loss_ce=11.0706, 208 params w/ grad (backbone=158, ncr=47, integ=3), peak 2.03 GB, doc_len=175 tokens |
| 8 | Full-graft checkpoint save→fresh backbone+ncr+integ→load→forward | PASS — bit-identical (logits, o, Z), max_diff=0.00e+00 |
| 9 | Eval batch (no_grad) at held-out depth h=5, `recovered_frac@0.9` computed | PASS — rf=0.0000 (untrained-at-init, EXPECTED — proves the metric COMPUTES, not convergence), mean_answer_logprob=-10.84 |
| 10 | `--teacher-force-operator` isolation | PASS — fit_residual=7.30e-06, `ncr_head` receives ZERO grad, backbone/key_adapter/read_injector DO train |
| 11 | Ablation-flag construction (`--adapter mlp`, `--read-inject mlp_logits`, both) | PASS — all 3 combinations shape-correct + finite, CPU-fast |

**Base file md5s (verbatim reuse — confirmed byte-identical, box vs. local
repo, and identical to §G3-B1's own table for the three shared files):**

| File | md5 |
|---|---|
| `matrix-thinking/ncr/ncr_earlyln_scale.py` | `3a87fcc92bb8341203c5e8c1f039a0af` |
| `matrix-thinking/ncr/ncr_models.py` | `6d7b30a592bee11f6c2135165801742d` |
| `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` | `34addd9d8cc6a3df5a367d0f18a2ee0e` |
| `matrix-thinking/deltanet_rd/grammar_rd.py` (NEW dependency this build) | `b7eeca0f6fc56210ef9c633fe719b540` |
| `matrix-thinking/chapter2/model_v4.py` (NEW — BindingEncoder signature source) | `633775b55a802df12a0dab45e0d223d7` |
| `experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py` (edited in place) | `c54ef692061c02294c38d5dc154166ae` |

### Pre-launch red-team (answers recorded)

1. **Memory fit.** Phase-0-scale smoke footprint: 2.03 GB peak (B=4,
   doc_len=175, full graft co-resident: backbone+ncr_head+integ+Adam
   optimizer states). §G3-B1's own 98M-alone eval-only anchor (9.52 GB,
   no optimizer states) and §6.1's full-training anchor (23.5 GB,
   B=32/T=512, optimizer states included) both stand unchanged — this
   wave's ADDITIONAL footprint (NCR head + integration adapters +
   grammar_rd batch) is negligible next to either anchor, confirming
   §G3-B1's own prediction ("expected total training footprint ≈23.5-24GB
   ... negligible next to a (32,512,768) backbone activation tensor").
2. **Placement / contention.** `nvidia-smi` before AND after this smoke
   run: all 8 GPUs remained production-busy (86-100% util both times,
   43,833-45,081 MiB/80GB each) — SAME finding as §G3-B1, re-verified
   fresh, not assumed stale. This build's own smoke ran on GPU 7 (the most
   headroom, matching §G3-B1's own choice) for an 14.6s wall-clock CUDA
   window; `nvidia-smi` before/after confirms memory returned EXACTLY to
   the pre-smoke baseline (43,833 MiB) and the compute-process count is
   unchanged (9 rows incl. header = 8 processes) — no production job
   disturbed, same discipline as §G3-B1.
3. **Measured-rate timeout/ceiling.** UNCHANGED from §G3-B1's own pricing
   (this build did not re-measure a full training cell, only the smoke
   suite above) — §6.1's 0.236 s/step (98M) uncontended rate, ≈3.3×
   contended multiplier per §G3-B1 item 2's own precedent, still applies
   to any real Phase-0/Phase-1 launch.
4. **Duplicate check.** `grep -c "§G3-B3" NCR_REAL_LM_DESIGN.md` returns 1
   occurrence (this section, being written) before this edit; no prior
   `ncr_lm_wave1_smoke.py` REWRITE session found in git history at this
   path beyond §G3-B1's own original commit.
5. **Gate-discharge check.** §G3-B2 (immediately above, same file) is the
   ratified adjudication this section implements; its two RULINGs are
   both implemented EXACTLY as stated (shapes verified against the real
   `BindingEncoder` signature, not guessed) — no design decision was
   re-opened.

### Launch command — STILL NOT RUNNABLE (by design, per the build brief)

Unlike §G3-B1's version (which had two UNRESOLVED adapter flags), this
build's wiring is COMPLETE and real-CUDA-verified — but a full Phase-0/
Phase-1 TRAINING SCRIPT (real+synthetic batch mixing per §5.2 Option 1,
step loop, LR schedule, checkpoint cadence, logging) does not exist yet;
only the INTEGRATION MODULE (`NCRIntegration`) + its own standalone
smoke driver do. Wiring `NCRIntegration` into `lm_pretrain_rd.py`'s
existing training loop is the NEXT build step, gated on the independent
audit this section's own STOP now hands off to. The shape the eventual
command takes, mirroring `lm_pretrain_rd.py`'s own CLI convention, with
the adapter/inject choices now RESOLVED (not "UNRESOLVED, gap 1/2" as in
§G3-B1):

```
python3 lm_pretrain_rd.py \
  --corpus openr1-mix-ext --data-dir /data/deltanet_rd_data \
  --d-model 768 --d-state 64 --n-layers 12 --seq-len 512 --batch-size 32 \
  --steps 20000 --ckpt-every 1000 --seed <n> \
  --internal-timeout <>= contended estimate, §G3-B1 item 2: ≈3.3x the
      uncontended rate (Phase-1: 1.377 GPU-h/cell uncontended -> ≈4.5-4.6h
      wall-clock contended per cell if parallelized across 4 GPUs)> \
  --ncr-active --ncr-d 25 --ncr-h 64 --ncr-k 24 --ncr-write-arm free \
  --ncr-adapter linear --ncr-read-inject add --ncr-vocab-size-total 50259 \
  --ckpt-dir /data/ncr_wave1_ckpts/... --out results/ncr_wave1/...
```

Flags after `--ncr-write-arm free` do not exist in `lm_pretrain_rd.py`
today (this build's `NCRIntegration` module would be imported and wired
into that script's training loop, not re-derived) — `--ncr-adapter
linear --ncr-read-inject add` are now the RATIFIED, real-CUDA-verified
defaults (not placeholders); `--ncr-vocab-size-total 50259` is this
section's own discovered-and-fixed wiring requirement.

**Expected wall-clock.** Measured (this smoke): 14.6s wall-clock for the
full 11-item CUDA suite (Phase-0-scale shapes only, B≤16, doc_len=175 —
NOT a training-cell timing measurement). §G3-B1's own Phase-0a/Phase-1
GPU-h estimates (§6.2 text, unchanged by this build) remain the governing
projections for any real training cell; this wave produced no NEW timing
data at training scale (that is Phase-0a's own job, §6.2, still gated on
the independent audit below).

### Archive

Harness + results archived at
`experiment-runs/2026-07-17_ncr_gate3_wave1/` (`ncr_lm_wave1_smoke.py`,
`ncr_lm_wave1_results.json`), per `CLAUDE.md`'s reproducibility rule —
same directory as §G3-B1, script edited in place per the build brief's
"REPLACE" instruction, results JSON overwritten with this wave's own
11-item run.

### Readiness verdict

**READY-FOR-AUDIT.** The §G3-B2 ratified wiring is fully implemented,
real-CUDA-verified end-to-end on REAL `grammar_rd` Task-1 data (not
random tensors) — forward, backward, finite grads through BOTH adapters
AND the backbone, a real optimizer step, full-graft checkpoint save/
resume (bit-identical), an eval batch computing `recovered_frac@0.9`, and
the `--teacher-force-operator` isolation property holding exactly as
designed. One discovered wiring gap (grammar_rd's reserved token ids
needing a +2 vocab extension) was fixed and disclosed, not silently
invented or left as a STOP — it is mechanically singular (grammar_rd's
own docstring names the exact convention) and negligible in cost
(+1,536 params). Two ablation flags (`--adapter mlp`, `--read-inject
mlp_logits`) are construction-verified but NOT run through the full
pipeline this wave, per the build brief's "pre-wire... do not run them."
GPU spend on any real Phase-0/Phase-1 training cell stays gated on the
mandatory independent audit §G3-B2 itself requires (correctness AND
wiring-bias/FAIL-informativeness) plus the coordinator's own red-team,
per the build brief. STOPPING here.

## §G3-B4 INDEPENDENT PRE-SPEND AUDIT (2026-07-17)

Independent auditor, did NOT write the graft. Target: `experiment-runs/
2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py`, md5
`c54ef692061c02294c38d5dc154166ae` (VERIFIED, matches). Read-only except this
section. Signatures cross-checked against the REAL sources: `model_v4.py`
`BindingEncoder`, `ncr_models.py` `binexp_read`, `ncr_earlyln_scale.py`
`NCREarlyLNModel`, `lm_pretrain_rd.py` `DeltaNetLM`, `grammar_rd.py`
`sample_batch_rd`/`DeltaNetRDTaskConfig`.

### Dimension 1 — Graft correctness vs §G3-B2 ratified spec: PASS (no defects)

Every wiring claim checks out against the real code, not the build's prose:
- **Write adapters at the right positions.** `extract_kv` gathers hidden at
  `key_pos = item_pos-2` (KEY) and `val_pos = item_pos` (VALUE) — byte-for-byte
  grammar_rd's own convention (`grammar_rd.py:452-453`: `item_pos =
  arange(K)*clause_len + buf_len + 2`, `key_pos = item_pos-2`). fp32-cast at the
  boundary. Fed to `ncr_head.encode(keys_v, values_v)` → `BindingEncoder.forward(
  keys, values)` whose real signature is `(B,K,d)`/`in_proj=Linear(2*d,h)`, d=25
  — EXACT match; `d_ncr=25` = the encoder's own `d`. RULING 1 satisfied.
- **Read injection = option (a).** `o` (25-d, from `binexp_read`) → `read_injector
  = Linear(25,768,bias=False)` → ADDED to `hidden[:, query_mark_col]` (the <Q>
  position, col 173) → `F.linear(·, backbone.embed.weight)` = the SHARED tied
  head (`DeltaNetLM.forward` tied-head path confirmed). Not degenerate. RULING 2
  satisfied.
- **CE on the query-answer token only.** `loss = F.cross_entropy(logits,
  answer_token)` at the single <Q> position; `answer_token = entity at tgt_slot =
  π^h(a)`; `input_ids = doc[:,:-1]` drops the answer so <Q>@173 is the last input
  col and its next-token target is the answer@174. Standard AR next-token at ONE
  position — not all-tokens, not the wrong span. Correct.
- **`recovered_frac@0.9` is eval-only.** Computed solely inside `smoke_9` under
  `no_grad`; never enters `loss`. Correct.
- Column arithmetic (`query_key_col=T_bind+buf_len=171` = query KEY;
  `query_mark_col=T_bind+query_len-1=173` = <Q>), the single-int hop into
  `binexp_read` (`assert isinstance(h,int)` honored), K=24→T_bind=168→input=174
  (≥ the 128 `_MIN_KERNEL_T` floor), and the tied-head/`config()` round-trip are
  all correct. Param counts exact (backbone 97,618,176; NCR head 173,209 =
  40·64²+4·25·64+46·64+25; integ 57,600). The graft IS the ratified wiring.

### Dimension 2 — FAIL-informativeness / wiring-bias: **FATAL (verdict-blocking)**

**DEFECT G3-B4-1 [FATAL] — the calibration cannot localize a FAIL to the NCR
head, because the ratified FAIL-informativeness ablation set omits the one
control that closes the dominant confound: a plain-DeltaNet backbone-only
(read-ablated, o≡0) arm.**

Trace of the concern:
1. The read is ADDITIVE and NON-bottlenecked: `logits = tied_head(h_q +
   read_injector(o))`. `h_q` is the backbone's own post-`norm_f` hidden at <Q>,
   which has causally attended over every bind clause and the query. The 98M
   DeltaNet is a fast-weight associative memory; **Task 1 is the abelian
   single-K-cycle (§2.2 confirms it is CYCLIC/solvable), so the backbone is NOT
   structurally barred** from learning the in-distribution h∈{1,2,3} lookup
   directly through `h_q`. This is exactly the P=1-bottleneck failure the repo's
   own standing learnings warn about ("decoder reads ONLY the state, never raw
   inputs… verify with a blank-out test") and which `model_v4.py`'s synthetic
   task enforces by construction (`decode = pure fn of Z`) but this graft does
   NOT.
2. Consequence for a **FAIL**: if the backbone solves in-distribution via `h_q`,
   the CE loss saturates and the NCR pathway (adapters→encoder→read_injector) is
   gradient-STARVED — so `recovered_frac@0.9` stays low. Phase-1's Gate-0 keys on
   "**in-distribution** recovery ≥0.9" (§6.2), which is precisely the
   starvation-sensitive quantity. A Gate-0 FAIL then reads as "the NCR head can't
   train through a real LM" when the true cause is "the backbone made the NCR
   read non-load-bearing." §G3-B2's ratified ablations (MLP write-adapter,
   read-inject option (b), teacher-force) vary the adapter/read FORM and isolate
   encoder-vs-rest — **none of them tests whether the backbone alone already
   solves the task**, so the FAIL routes to no ablation and stays uninterpretable.
   This is the §B4/§B6 pattern: the design proof + spec + build all missed it.
3. Consequence for a **PASS/tie**: both Phase-1 arms (NCR + flat-vector) can ride
   the same backbone shortcut and tie at the backbone's level — a hollow,
   uninterpretable comparison.

Gradient path itself is CLEAN (no wrongful detach): loss→read_injector→o→Z→
{key_adapter,value_adapter, encoder}, and →q_key→key_adapter, and →h_q→backbone;
smoke 7 confirms all three modules receive finite grad (backbone=158, ncr=47,
integ=3). No vanishing at train depths (h∈{1,2,3}; renorm is positive-scalar,
cosine-invariant). The problem is not a broken pathway — it is a COMPETING,
un-bottlenecked pathway that can starve the one under test.

**DEFECT G3-B4-2 [MAJOR] — value↔next-key space-alignment burden absent from the
synthetic task that validated NCR.** `binexp_read` composes ONE `Z` repeatedly, so
multi-hop coherence requires `value_adapter(h_val[i]) ≈ key_adapter(h_key[π(i)])`
(the VALUE of clause i and the KEY of clause π(i) are the same entity but reached
through two INDEPENDENT no-bias Linears at two DIFFERENT token positions). The
synthetic NCR task had keys/values in ONE space by construction (`query_keys =
keys.clone()`, `value_j = key_{π(j)}` literally). Here the model must LEARN that
alignment, driven only by CE at h∈{1,2,3}. It is learnable in-distribution (h=2,3
exert some pressure) but is an extra graft-specific burden that compounds
DEFECT-1's starvation: a FAIL could reflect "the two adapters couldn't align
through a real backbone," not "NCR composition doesn't work." Adapter dimensional
CAPACITY is adequate (768→25 no-bias places 24 near-orthogonal entity directions
in R^25, 24≤25; read_injector 25→768 is full-width, not a masking bottleneck) —
so a FAIL is NOT an adapter-capacity artifact; it is this alignment + the shortcut.

### Dimension 3 — Unfrozen BUFFER row (§G3-B3 disclosed deviation): HARMLESS (MINOR/note)

Ruled decisively harmless FOR THIS CALIBRATION. R2-3's zero-pin+freeze rationale
is specific to `model_rd.py`'s beta-mask WRITE machinery, which this backbone
(`lm_pretrain_rd.DeltaNetLM`, plain learned beta) does not use. The BUFFER token
(id 50257) is a single GLOBAL learnable embedding (no positional embedding), fills
only non-content positions, is never tapped by `extract_kv` (which reads KEY/VALUE
positions only), and — being a constant vector — cannot carry per-example answer
info into the sequence; the answer is dropped from `input_ids` entirely. Worst
case it injects a data-independent constant into the DeltaNet recurrence at buffer
positions: architectural noise, not a leak, shortcut, or PASS/FAIL bias. Note-only.
(grammar_rd's own docstring EXPECTS zero-pin; a real flagship freeze is cheap
cleanliness but not load-bearing for the calibration's validity.)

### Dimension 4 — Teacher-force control validity: VALID as scoped (MINOR caveats)

The closed-form fit is mathematically correct: `Z = (pinv(k)@v)^T` gives `Z k_i =
v_i` exactly when K=24 ≤ d=25 with k full-row-rank (generic for distinct
entities); `k@pinv(k)=I_24` verifies it, and smoke 10 measures fit_residual
7.3e-6. Isolation is real: `k,v` are `.detach()`ed, so the encoder
(`BindingEncoder`) receives ZERO grad (smoke 10 confirms `ncr_untouched=True`)
while backbone/key_adapter/read_injector still train. So (teacher-force PASS +
free-write FAIL) DOES localize to encoder write-learning — a VALID encoder-vs-rest
isolation. Caveats: (a) it does NOT isolate NCR-vs-backbone — DEFECT-1's shortcut
applies to teacher-force too, so a teacher-force PASS can also ride `h_q`; (b) at
HELD-OUT depth the multi-hop teacher-force read inherits DEFECT-2's alignment
burden, so it isolates one-hop write-quality cleanly only in-distribution.
value_adapter legitimately gets no grad in this mode (operator built from detached
values) — not a bug. Control is sound for its stated purpose.

### Dimension 5 — Vocab 50257→50259 fix: SANE (no inconsistency)

grammar_rd mints `buffer_id=50257`, `query_id=50258`; a 50257-row embedding
CUDA-asserts OOB on those ids (build hit this). Building the grammar-dependent
backbone at `vocab_size_total=50259` adds exactly +2 tied embedding rows (+1,536
params, inside tolerance); the tied head emits 50259 logits; the answer is always
a real entity id < 50257 (valid class index); the +2 reserved ids are valid but
low-prob logit slots. The `add` read-injector is vocab-independent (only the
unused `mlp_logits` arm touches vocab). No embedding/logit inconsistency. Clean.

### Dimension 6 — Uninterpretable-signal / vacuous-smoke sweep

- **DEFECT G3-B4-3 [MINOR] — smoke 5's checkpoint-round-trip sub-claim is
  VACUOUS.** It saves+loads `ncr2` but the assertion compares `o1` vs `o2`, both
  `binexp_read(Z_probe,q_probe,5)` on the SAME random tensors — `ncr2` is never
  called. It verifies only that a pure function is deterministic; the ncr-head
  round-trip is untested here. (Real full-graft round-trip coverage DOES exist in
  smoke 8, so this is a labeling/coverage weakness, not a validity hole.)
- **Framing note (not a code defect):** all 11 smokes prove TRAINABILITY
  (gradients flow, shapes/finite/determinism) — none tests, or can test, whether
  the read is LOAD-BEARING. smoke 9's `rf=0.0000` at init is expected and proves
  nothing about the eventual verdict. So the all-PASS smoke suite is silent on the
  DEFECT-1 confound by construction; closing it is a TRAINING-time control, not a
  smoke.

### DISPOSITION

| Defect | Sev | Disposition |
|---|---|---|
| G3-B4-1 backbone-only control missing; in-dist-recovery FAIL/PASS not localizable | **FATAL** | BLOCK GPU spend until Phase-1 adds a plain-DeltaNet backbone-only (o≡0, read-ablated) arm AND registers the attribution rule (below) |
| G3-B4-2 value↔next-key alignment burden | MAJOR | Register as a known FAIL-mode; add an alignment diagnostic (or a shared/tied write adapter) as a pre-authorized ablation; do NOT read teacher-force at held-out depth as a clean write-isolation |
| G3-B4-3 vacuous smoke-5 round-trip | MINOR | Fix the assertion to compare `ncr` vs `ncr2` outputs (or delete the vacuous half); non-blocking |
| BUFFER unfrozen (dim 3) | MINOR | Note-only; freeze at flagship time for cleanliness |

**Required attribution rule to make GATE-3 interpretable:** a Phase-1 Gate-0
in-distribution-recovery FAIL may be attributed to "the NCR head can't train
through a real LM" ONLY IF the backbone-only arm does NOT itself solve the task
in-distribution (i.e., its answer accuracy is materially below the full graft's —
the NCR read is demonstrably load-bearing). If backbone-only already solves it,
the calibration is uninterpretable and the graft must be re-bottlenecked (e.g.
read-only decode / harder P=1 bottleneck) BEFORE main-wave GPU. This is the repo's
own blank-out-test discipline, applied at the graft.

**The graft code is CORRECT and may be committed.** What is not yet safe is
spending GPU on a make-or-break verdict, because a FAIL (and a hollow PASS) cannot
currently be localized to the NCR head.

**VERDICT: BLOCKED** (add the backbone-only control arm + attribution rule to
Phase-1; the two MINORs and the MAJOR-2 diagnostic are recommended alongside but
G3-B4-1 is the launch-blocker). Re-audit not required for the code — only
confirmation that Phase-1's runner build wires the backbone-only arm.

## §G3-B5 COORDINATOR ADJUDICATION of the §G3-B4 audit (Fable, 2026-07-17)

Audit ACCEPTED. The graft code is correct (§G3-B4 dim-1 PASS); the block is
experimental-design, and the FATAL is in MY OWN §G3-B2 wiring ruling — owned
here.

**G3-B4-1 [FATAL] — ACCEPTED.** My ratified read-injection (`tied_head(h_q +
read_injector(o))`, option a) is additive/non-bottlenecked. Task-1 is the
abelian K-cycle (solvable), so a 98M DeltaNet can solve in-distribution
h∈{1,2,3} through `h_q` alone, gradient-starving the NCR read → a FAIL would
misread as "NCR can't train" (hollow PASS symmetric). This is the repo's own
P=1-bottleneck / blank-out hard rule (CLAUDE.md), the same class as the
§B4/§B6 Stage-0 miss. The fix does NOT require a graft code change — it is a
CONTROL ARM + a registered interpretation rule, both baked into the runner:

**GATE-3 Wave-1 calibration = TWO arms + registered attribution (frozen
before spend):**
1. **Full-graft** (NCR read active).
2. **Backbone-only control** (`o ≡ 0`, read-ablated — the wiring already
   supports it trivially; this IS the blank-out discipline at the graft).
- **PRIMARY interpretable signal = the recovery GAP (full-graft −
  backbone-only), at DEEP composition depth** (h beyond in-distribution),
  where a solvable-task backbone cannot shortcut via length-generalization —
  the program's own thesis. 
- **PASS** (NCR head trains AND is load-bearing) = full-graft recovers deep
  composition (recovered_frac@0.9 ≥ the §6.2 Gate-0 bar at deep h) AND
  materially exceeds backbone-only there.
- **FAIL** = full-graft does NOT recover deep composition. Attributable to
  the NCR head as "can't train" ONLY IF the read was demonstrably
  load-bearing — i.e. NOT the case that backbone-only already solves
  in-distribution (which would mean the read got no gradient → re-bottleneck,
  not "NCR can't train"). The pre-wired `--teacher-force-operator` +
  `--adapter mlp` + `--read-inject b` arms disambiguate on a FAIL.
- **UNINTERPRETABLE** = backbone-only also solves the deep task → the task
  isn't testing composition → redesign before main-wave spend.

**G3-B4-2 [MAJOR] — ACCEPTED, controlled not blocking.** The value↔next-key
adapter-alignment burden means a FAIL could be adapter-misalignment not NCR
mechanism; the `--adapter mlp` + `--teacher-force` arms are the registered
disambiguators (adapter CAPACITY is adequate per the audit, so not a capacity
artifact). Route a FAIL through them before concluding.
**G3-B4-3 [MINOR] — fix in runner** (vacuous smoke-5 checkpoint assertion).
**BUFFER deviation — HARMLESS** (audit dim-3: R2-3's zero-pin is
beta-mask-specific; this backbone doesn't use it; BUFFER is a global constant,
never tapped for the write). Note-only.

**DISPOSITION: BLOCKED → runner-build with the two-arm design + attribution
rule + smoke-5 fix baked in; NO graft code change needed. Then re-verify the
runner wires the control correctly (I check) → red-team → launch.**

## §G3-B6 WAVE-1 RUNNER BUILD + COORDINATOR VALIDATION (Fable, 2026-07-17)

The runner-build agent thrashed (3× monitor-defer + re-launch of its own
smoke, never closing); coordinator took over, cleaned up the runaway smoke
PIDs (exact-PID kills), and VALIDATED the runner directly.

**Runner VALIDATED (`ncr_lm_wave1_runner.py`, + `orchestration/run_wave1_calibration.sh`):**
- **Control zeroes the read EXACTLY, pre- AND post-train** (`read_ablation_check`:
  pre/post `max_abs_diff = 0.0`, both `verified_exact_zero = true`) — the §G3-B5
  fix confirmed: `backbone_only`'s read contributes exactly zero to its logits,
  a genuine frozen-at-init null. Audited graft code (`ncr_lm_forward`) untouched;
  ablation via a separate `ncr_lm_forward_ablatable` wrapper.
- **Two arms** `['full_graft','backbone_only']`, bit-identical init, same batch.
- **Attribution schema in results JSON** (blind-assessor-ready): metric (a)
  `recovered_frac@0.9` on `o_raw` → PRIMARY = full−backbone GAP at deep ladder
  {5,12,20,29,40,61}; metric (b) `answer_accuracy` on actual logits → the
  PRECONDITION (backbone-shortcut check); `attribution.frozen_rule_text` verbatim.
- **Phase-0 measured rate** (box, GPU5): full 0.1174 / backbone 0.1133 / both-arms
  0.2308 s/step; 5568 tok/step/arm; suggested ceiling **4.865 GPU-h @ 20K steps
  batch=32** (contended ×3.3 applied).
- Launch cmd: `bash orchestration/run_wave1_calibration.sh <GPU> 4.865 20000 0`
  (batch=32 train / 64 eval, self-healing supervisor, terminal-status-gated,
  stop-file, blind).

**COORDINATOR RED-TEAM FINDING — memory-fit gate (not yet cleared):** the two
arms are CO-RESIDENT (`restore_arms_and_opts` + `build_two_arms` place both full
98M+head+opt on device). At batch=32 train / 64 eval, the ~2× footprint (single
arm ≈23.5GB per FROZEN_BIAS §13.7) risks OOM in the ~36GB free alongside the
production queue (all 8 GPUs ~44/81GB used); eval-batch=64 is the named eval-OOM
hazard. **LAUNCH IS GATED on a real-config memory probe** (batch=32/eval=64/both
arms, running now). If it OOMs/approaches the limit: reduce eval-batch, or free
one arm's activations between arms (sequential), or drop to a fitting batch —
the calibration's job is binary trainability, robust to a smaller batch. Do NOT
blind-launch the ~5 GPU-h cell until fit is confirmed.

## §G3-B7 WAVE-1 CALIBRATION LAUNCHED (make-or-break, 2026-07-17 22:10 UTC)

Memory-fit red-team gate CLEARED empirically: a bounded 70-step real-config
probe (batch=32 train / 64 eval, both arms) COMPLETED on GPU 2 (~37GB free
alongside production) through TWO evals with ZERO OOM — the co-resident arms
train SEQUENTIALLY per step (additive Phase-0 timing confirmed), so only one
arm's activations peak at a time (~23GB, not the ~47GB worst case). Red-team
did its job (probed, not blind-launched).

**LIVE:** tmux `ncr_g3w1_g2`, GPU 2, cell `wave1_calib_K24_s0`, runner md5
`5c0442c9` (box == repo, validated), orchestration supervisor (terminal-status-
gated, stop-file `/home/nvidia/results_gate3_wave1/STOP`), 20000 steps, batch=32/
eval=64, seed 0, ceiling 4.865 GPU-h (timeout 20141s), BLIND. Both arms
(full_graft + backbone_only o≡0 control). Launched 22:10 UTC, clean start (step
1, 0 errors); expected ~4.2-4.9h contended.

**ON COMPLETION (the frozen §G3-B5 attribution rule — blind assessor applies
mechanically to `wave1_calib_K24_s0.json`):** PRECONDITION (metric-b
answer_accuracy on actual logits): is backbone_only materially BELOW full_graft
(read demonstrably load-bearing)? If NO (backbone shortcuts) → UNINTERPRETABLE,
re-bottleneck. If YES → PRIMARY (metric-a recovered_frac@0.9 GAP on o_raw at deep
ladder): does full_graft materially exceed the backbone_only frozen-null at deep
depth? PASS = make-or-break passed (NCR head trains + load-bearing in a real LM)
→ up ladder 98M→392M→[own gate]→1B+. FAIL → route pre-wired ablations
(--teacher-force / --adapter mlp / --read-inject b) FIRST; only if genuine,
honest "NCR can't train in a real LM" → PI immediately (bet in trouble; scaling
paper + kwall = shippable core). n=1 first signal; add seeds if borderline
(§2.35 precedent).

## §G3-B8 STAGE/GATE-3 WAVE-1 VERDICT (blind assess, 2026-07-18)

Blind assessor applying the FROZEN §G3-B5 / §G3-B7 attribution rule
mechanically to the raw `wave1_calib_K24_s0.json`. No expectations carried in.

**FINAL VERDICT: UNINTERPRETABLE.** The make-or-break precondition was NOT
met: the integrated model failed to learn the task in **BOTH** arms
(answer_accuracy at chance ≈1/24, recovered_frac@0.9 floored at 0.0 at every
depth including in-distribution h=1). The NCR read is therefore **NOT
demonstrably load-bearing**, so the metric-a deep-recovery null CANNOT be
attributed to "the NCR head can't train." Per the frozen rule this routes to
UNINTERPRETABLE → re-bottleneck / fix the integration so the full graft at
least learns in-distribution, THEN re-run. It explicitly does **NOT** trigger
the FAIL branch ("NCR can't train in a real LM → PI immediately") — that branch
is gated on the read being demonstrably load-bearing, which is false here.

### Integrity (all checks pass; run is a valid, substantial terminal state)
- **status = ABORTED-BUDGET at step 19026/20000** (95.1% of target), elapsed
  17521s, gpu_h 4.867 ≈ ceiling 4.865. Valid terminal (budget ceiling hit), rc=0,
  clean driver exit. Both eval bands (in_dist + deep) fully populated.
- **Both arms present** with complete eval data: `full_graft`, `backbone_only`.
- **Runner md5 = `5c0442c952922b8e20af1981b954267f`** (box == local archive copy;
  prefix `5c0442c9` as expected). runner_tag `ncr_gate3_wave1_runner_v1`.
- **read_ablation_check:** pre_train max_abs_diff 0.0 (verified_exact_zero True)
  AND post_train max_abs_diff 0.0 (verified_exact_zero True) — control's read is
  EXACTLY zero pre and post train (genuine frozen-at-init null). Post-train
  exact-zero also re-confirmed in the run log.
- **Log clean:** 0 Traceback, 0 OOM/CUDA-error. n_skipped_steps 0/0 both arms.
- Minor: `git_commit: UNKNOWN` (runner did not stamp commit) — not fatal; the
  runner is md5-pinned and byte-identical to the archived copy.
- **Truncation (19026 vs 20000) does NOT matter:** loss had plateaued by ~step
  5k (last-100-step mean 3.92 full / 3.97 backbone), LR already decayed to
  3.16e-5 on the cosine tail; the final 974 steps cannot turn a chance-level,
  loss-3.9 model into a K=24-composition solver. Full-budget for all
  interpretive purposes.

### Metric-b PRECONDITION — answer_accuracy on actual logits (n=64/hop)
Chance (uniform over K=24 answer tokens) = 1/24 = **0.041667**. Per-hop
SE(p≈0.042, n=64) ≈ 0.025; in-dist 3-hop mean SE (n=192) ≈ 0.0144.

| band | hop | full_graft | backbone_only |
|------|-----|-----------|---------------|
| in-dist | h=1 | 0.06250 | 0.01563 |
| in-dist | h=2 | 0.01563 | 0.01563 |
| in-dist | h=3 | 0.07813 | 0.04688 |
| **in-dist MEAN** | | **0.05208** | **0.02604** |
| deep | h=5  | 0.03125 | 0.04688 |
| deep | h=12 | 0.07813 | 0.04688 |
| deep | h=20 | 0.03125 | 0.06250 |
| deep | h=29 | 0.06250 | 0.01563 |
| deep | h=40 | 0.06250 | 0.03125 |
| deep | h=61 | 0.04688 | 0.04688 |
| **deep MEAN** | | **0.05208** | **0.04167** |

Every one of the 18 arm×depth cells lies in [0.0156, 0.0781], i.e. within ~1.5 SE
of chance 0.0417. **full_graft itself does not solve in-distribution:** mean
0.05208 is only +0.72 SE above chance (not significant). backbone_only in-dist
mean 0.02604 is −1.09 SE (noise). The full−backbone in-dist gap = 0.05208 −
0.02604 = **+0.02604**, z = 0.02604/0.0204 = **1.27** (n.s., p≈0.20).

**Precondition verdict: NOT satisfied.** The rule (and the JSON's own embedded
`frozen_rule_text`) defines "read demonstrably load-bearing" as backbone_only's
answer_accuracy being *materially below the full graft's*. That operationalization
tacitly assumes the full graft SOLVES in-distribution; here it does not (chance).
When the full graft is itself at chance, "backbone below full graft" (a 0.026,
1.27-SE difference between two chance-level values) does not establish that the
read carries any competence. The read is not demonstrably load-bearing — not via
the anticipated backbone-shortcut, but because **neither arm learned the task at
all.** Note: the naive one-sided literal reading ("backbone does not solve it →
load-bearing → FAIL") is DEFEATED by full_graft also sitting at chance; a FAIL
requires demonstrated load-bearing, which is absent. → UNINTERPRETABLE.

### Metric-a PRIMARY — recovered_frac@0.9 on o_raw (reported for completeness)
Reached only conditionally; recorded because it independently corroborates
"didn't learn." recovered_frac@0.9 = **0.0 for BOTH arms at EVERY depth**, in-dist
AND deep. Per-depth GAP (full − backbone), deep ladder:

| h | full_graft | backbone_only | GAP |
|---|-----------|---------------|-----|
| 5 | 0.0 | 0.0 | 0.0 |
| 12 | 0.0 | 0.0 | 0.0 |
| 20 | 0.0 | 0.0 | 0.0 |
| 29 | 0.0 | 0.0 | 0.0 |
| 40 | 0.0 | 0.0 | 0.0 |
| 61 | 0.0 | 0.0 | 0.0 |

Deepest-rung (h=61) gap = **0.0**. In-distribution recovered_frac@0.9 is ALSO 0.0
for full_graft (even at h=1). The primary capability signal is a hard floor of
zero — no PASS is possible, and the zero-in-distribution recovery reinforces that
this is a "model did not learn / read not producing recoverable o_raw" state, not
a clean deep-composition FAIL.

### Loss corroboration
Both arms start ~10.95 (≈ ln(50259)=10.82, uniform over full vocab) and plateau
at ~3.9 (full 3.916 / backbone 3.970, last-100-step mean). ln(24)=3.178 — the
model did not even collapse to uniform-over-the-24-answer-alphabet; it learned
sequence/marginal token structure, not the K-cycle mapping. Consistent across
both metrics and both arms.

### Caveats (foregrounded)
- **n=1 seed.** What it CAN support: at this exact config (K=24, 98M DeltaNet
  backbone, 20k steps, batch 32, lr 3e-4, this task formulation) the calibration
  cell did NOT produce an interpretable make-or-break test — the integrated model
  failed to learn in-distribution in both arms. What it CANNOT support: (a) the
  general claim "NCR can't train in a real LM" (requires demonstrable
  load-bearing, absent, and would need seeds); (b) ruling out that a different
  lr/schedule/longer horizon/easier curriculum lets the full graft learn
  in-distribution; (c) distinguishing "model didn't train" from "o_raw recovery
  instrument mis-wired" — the recovered_frac@0.9 hard-zero across all 18 cells is
  consistent with either, though the independent answer_accuracy-at-chance signal
  makes "didn't learn the task" the more parsimonious read.
- **Per-cell eval n=64 is small** (per-hop SE≈0.025); no single hop is trusted,
  but the "everything at chance / recovery hard-zero" pattern is consistent across
  all 18 arm×depth cells, so the conclusion is robust to per-cell noise.
- **Anomaly / rule-gap:** the frozen rule enumerated only the *backbone-shortcut*
  cause of UNINTERPRETABLE (backbone solves it, gradient-starving the read). The
  observed cause is the opposite corner it did not enumerate — BOTH arms at
  chance. Same disposition (cannot attribute → re-bottleneck/fix before main-wave
  GPU), but flagged so the re-design targets whole-model in-distribution
  convergence, not only the P=1 bottleneck.

### Recommended next action (not a spend decision — assessor note)
Re-run the calibration only after the FULL graft can learn in-distribution
h∈{1,2,3} (i.e. full_graft in-dist answer_accuracy materially and significantly
above 1/24). Candidate levers before any main-wave GPU: verify the o_raw recovery
instrument taps the correct tensor (negative-control it), check task/curriculum
difficulty at K=24, lr/warmup/horizon, and the read-injection bottleneck. Do NOT
escalate "NCR can't train" to PI on this run — the data does not license that
attribution.

## §G3-B9 TEACHER-FORCE DIAGNOSTIC BUILD + coordinator validation (Fable, 2026-07-18)

Post-§G3-B8 (UNINTERPRETABLE, whole model didn't learn a PURE-synthetic K=24
task). Build agent added two runner edits then deferred (monitor-thrash, same
as prior); coordinator took over + VALIDATED the code directly:
- **mean_cos instrument (resolves §G3-B8's flagged ambiguity):** new
  `cosine_and_recovered_frac()` returns (recovered_frac@0.9, mean_cos) from ONE
  shared cosine tensor — re-derives graft.recovered_frac_at_09's IDENTICAL
  target (target = key_adapter(hidden at the answer entity's own bind-clause KEY
  position)) byte-for-byte, so the two metrics CANNOT silently disagree. Now a
  high mean_cos with rec@0.9=0 ⇒ threshold discarded real signal; near-zero
  mean_cos ⇒ read carries nothing. VALIDATED: consistent-by-construction.
- **--teacher-force-operator training+eval mode:** `if teacher_force: Z =
  integ.teacher_force_operator(keys_v,values_v)` — VALIDATED the runner imports
  NCRIntegration from the audited smoke module (`import ncr_lm_wave1_smoke as
  graft; NCRIntegration = graft.NCRIntegration`, lines 144/155), so
  teacher_force_operator (smoke:282, the audited closed-form op fit that
  bypasses the encoder, §G3-B3 smoke item 10: encoder zero-grad, residual
  7.3e-6) resolves correctly — no crash on that path. Isolates: teacher-force
  LEARNS in-dist ⇒ WRITE-learning (encoder tokens→operator) is the blocker;
  STAYS at chance ⇒ read-injection/task-setup broken.
- Audited two-arm path (ncr_lm_forward) untouched; smoke file md5 unchanged.

**PLAN:** launch a GENEROUS-config teacher-force diagnostic (higher LR + more
steps than the make-or-break's LR 3e-4 / 20K — "most favorable conditions" so a
FAIL can't be dismissed as under-training), pure synthetic, small ~2 GPU-h
ceiling, BLIND. On complete: does teacher-force learn in-dist (answer_acc ≫
chance)? + what does mean_cos show? → localizes WRITE vs READ/setup → §G3-B10.

### §G3-B9-LAUNCH: teacher-force diagnostic LIVE (2026-07-18 04:30 UTC)
tmux `ncr_g3b9_tf_g2` GPU2, cell `g3b9_tf_diag`, runner md5 29531d0c (box==local),
`--teacher-force-operator --lr 1e-3 --steps 8000` (generous "most favorable" config
vs the make-or-break's 3e-4/20K), batch32/eval64, ceiling 2.0 GPU-h (+timeout 8000s),
BLIND, out /home/nvidia/results_gate3_wave1/g3b9_tf_diag.json. Clean start (step 1,
read-ablation exact-zero PASSED pre-train, 0 errors). teacher-force applies to
full_graft ONLY (backbone_only stays the o≡0 null). Expected done ~06:30 UTC. ON
COMPLETE → blind assess §G3-B10: does full_graft+teacher-force learn in-dist
(answer_acc ≫ 0.042)? + mean_cos → localizes WRITE-blocker vs READ/setup-broken.

## §G3-B10 TEACHER-FORCE DIAGNOSTIC VERDICT (blind assess, 2026-07-18)

Blind assessor reading the raw `g3b9_tf_diag.json` (archived to
`experiment-runs/2026-07-17_ncr_gate3_wave1/g3b9_tf_diag.json` + `g3b9_tf_run.log`
in this pull). No expectations carried in.

**FINAL READING: READ/setup-broken, NOT a WRITE-blocker.** Handed the TRUE
operator directly (encoder fully bypassed), the model still does **not** learn
the in-distribution task above chance, does not beat the read-ablated (o≡0)
null, and its read vector carries no measurable cosine signal toward the
correct target. The final loss is *worse* than the prior (non-teacher-forced)
run's plateau despite strictly more favorable training conditions. If the
write-encoder were the sole blocker, a perfect operator should have let the
read+decode path solve in-distribution composition — it did not.

### Integrity — CLEAN
- `status="COMPLETED"`, `step=8000`, `steps_target=8000` (100%, not truncated).
- Both arms present with complete `in_dist`/`deep` eval blocks, n=64/cell.
- `read_ablation_check`: pre_train and post_train `max_abs_diff=0.0`,
  `verified_exact_zero=true` both times — backbone_only is a genuine frozen
  o≡0 null, confirmed at both ends of training.
- `teacher_force_check`: `active=true`, `ncr_zero_grad_checks_passed=8000` —
  matches the full 8000-step run; the encoder-zero-grad assertion (teacher-force
  actually bypassing the encoder, per §G3-B9's smoke-audited
  `teacher_force_operator`) passed every single step, none failed.
- `run.log`: 0 Traceback, 0 OOM/CUDA-error, 0 generic "error" hits. Log shows
  read-ablation exact-zero PASSED both pre-train and post-train, and a clean
  `COMPLETED at step 8000/8000 in 6714s` terminal line. `gpu_h=1.865` (under the
  2.0 ceiling). `n_skipped_steps={0,0}`. Minor: `git_commit="UNKNOWN"`
  (runner didn't stamp commit) — not fatal, matches the same non-fatal gap
  noted in §G3-B8.

### Q1 — Does teacher-force LEARN in-distribution? NO.
Chance = 1/24 = 0.041667. Per-hop SE (n=64) = 0.0250; in-dist pooled SE
(n=192) = 0.0144.

| band | hop | full_graft (teacher-forced) | z vs chance | backbone_only (o≡0 null) | z vs chance |
|------|-----|------|------|------|------|
| in-dist | h=1 | 0.01563 | −1.04 | 0.06250 | +0.83 |
| in-dist | h=2 | 0.04688 | +0.21 | 0.00000 | −1.67 |
| in-dist | h=3 | 0.01563 | −1.04 | 0.04688 | +0.21 |
| **in-dist MEAN** | | **0.02604** | **−1.08** | **0.03646** | **−0.36** |

full_graft's in-dist mean (0.02604) is **not materially above chance** — it
sits 1.08 SE *below* chance, i.e. statistically indistinguishable from (or
under) pure guessing. It is also **not** materially above backbone_only's own
mean (0.03646): gap = full − backbone = **−0.01042** (full is actually the
*lower* of the two), SE of the difference = 0.0204, z = **−0.51** (n.s., wrong
sign from what a "read works" story would predict). Handing the model the
correct operator produced no detectable in-distribution learning and did not
distinguish it from the read-ablated null.

Deep ladder, for completeness (pooled SE, n=384, both arms): full_graft mean
0.01302 (z=−2.81), backbone_only mean 0.01042 (z=−3.06) — both **significantly
below** the naive 1/24 chance level (an anomaly not seen in-dist; flagged, not
resolved — plausibly a non-uniform deep-eval answer/mode-collapse effect
independent of teacher-forcing, since both arms show it equally).

### Q2 — mean_cos: does the read carry any signal at all? NO (noise-level).
d_ncr=25 ⇒ expected cosine SD for two independent random unit vectors ≈
1/√25 = **0.20**. All observed |mean_cos| values are ≤ ~1.1× that null SD:

| band | hop | full_graft mean_cos | backbone_only mean_cos |
|------|-----|------|------|
| in-dist | h=1 | 0.2198 | 0.0868 |
| in-dist | h=2 | 0.0420 | −0.0793 |
| in-dist | h=3 | −0.0838 | 0.0847 |
| **in-dist MEAN** | | **0.0593** | **0.0307** |
| deep | h=5 | −0.0163 | 0.0903 |
| deep | h=12 | −0.0624 | −0.0856 |
| deep | h=20 | −0.0046 | −0.0952 |
| deep | h=29 | −0.0326 | 0.0632 |
| deep | h=40 | 0.0116 | −0.0596 |
| deep | h=61 | −0.0093 | 0.0749 |
| **deep MEAN** | | **−0.0189** | **−0.0020** |

The largest single value (full_graft h=1, 0.2198) is only ~1.1 null-SD above
zero — unremarkable against 9 cells tested, and not corroborated by any
neighboring hop. Every other cell in both arms sits well within one null SD of
zero. Per §G3-B9's shared-tensor instrument (mean_cos and recovered_frac@0.9
derived from the identical target, so they cannot silently disagree): a high
mean_cos with recovered_frac@0.9=0 would mean "real sub-threshold signal
hidden by the 0.9 bar." That is **not** what's observed — mean_cos is
noise-level everywhere, so recovered_frac@0.9=0 reflects a read that carries
**no signal**, not a good-but-under-threshold one.

### Q3 — recovered_frac@0.9 (completeness)
**0.0 for both arms, at every hop, in-dist and deep** — hard floor, consistent
with the mean_cos noise-level reading above (Q2).

### Q4 — Loss
Final (step 8000): full_graft = **4.6164**, backbone_only = **4.6737**.
Last-20-step mean: full_graft 4.6554, backbone_only 4.6269. Trajectory: both
arms fall from ~10.95 (step 1, ≈ln(50259)=10.82, uniform-over-full-vocab) to
~4.6–4.9 by step 50–100, then are **flat/oscillating in the 4.5–4.9 band for
the remaining ~7900 steps** — no further improvement despite LR still at 1e-3
through warmup and a long cosine decay tail.

- vs **ln(24) = 3.178**: final loss (4.616) is **~1.44 nats WORSE** (higher)
  than uniform guessing over the 24-way answer alphabet — the model has not
  even collapsed onto the correct answer set, let alone the correct mapping.
- vs the **prior run's ~3.9 plateau** (§G3-B8, non-teacher-forced,
  lr=3e-4/20K): this run's ~4.62–4.66 plateau is **~0.7–0.75 nats WORSE**,
  despite strictly more favorable conditions (perfect operator via
  teacher-force, 3.3× higher LR, and a run that reaches its plateau within the
  first ~100 of 8000 steps). More favorable training conditions produced a
  *worse* converged loss, not a better one — a red flag against "just the
  write-encoder was undertrained."

### Localization: READ/setup-broken (not WRITE-blocker)
Per the frozen isolation logic (§G3-B9): teacher-force LEARNS in-dist ⇒
write-encoder was the blocker; teacher-force STAYS at chance ⇒
read-injection/task-setup is broken even with a perfect operator. The data
lands unambiguously in the second bucket:
- In-dist answer_accuracy is at/below chance (z=−1.08) and statistically
  indistinguishable from the o≡0 ablated-read null (z=−0.51, wrong sign).
- mean_cos at every hop, both bands, is noise-level (≤~1.1 null-SD) —
  independent confirmation the read vector carries no recoverable signal
  toward the correct target, not merely "real but sub-0.9" signal.
- Loss is *worse* than a prior, less-favorable run, not better — inconsistent
  with "the only missing piece was a correctly-trained operator."

This rules out "the head is fine, only the encoder (tokens→operator) needs to
learn" as the explanation, because the encoder is entirely bypassed here and
performance still doesn't clear chance. The likelier fault sits in the
injection/read/decode path itself or in the loss/task wiring around it (e.g.
the injected operator not reaching the tensor the decode head actually reads,
a target/label misalignment, or a basis/scale mismatch between
`teacher_force_operator`'s closed-form fit and what the read consumes) — the
blind numbers alone cannot distinguish between these specific code-level
causes; that requires a follow-up code audit, not another training run.

### Caveats
- n=1 seed, n=64/cell — per-cell z-scores are noisy individually; the
  cross-cutting pattern (in-dist at/below chance in both arms, mean_cos
  noise-level in both arms at all 9 hops, loss worse than a prior weaker
  config) is what carries the conclusion, not any single cell.
- The deep-ladder both-arms-below-chance anomaly (z=−2.81 / −3.06) is flagged
  but unexplained by this run — orthogonal to the WRITE-vs-READ question since
  it appears equally in both arms.
- This diagnostic isolates WRITE-vs-READ; it does not itself identify which
  specific line/tensor in the read/injection/loss path is broken. That is the
  natural next step, gated on this verdict per §G3-B9's plan.

## §G3-B11 INTEGRATION HARNESS DEBUG (2026-07-18)

Static code audit + faithful CPU repros of the §G3-B10 teacher-force null
(the "natural next step" §G3-B10 gated). No GPU, no training. Faithful repro
uses the REAL `ncr_models.binexp_read` and a byte-exact copy of the
`NCRIntegration` adapter/teacher-force/inject ops (the DeltaNet backbone has
no local CPU path — `fla` is box-only — but the bug lives entirely
DOWNSTREAM of the backbone, driven here with random stand-in `hidden`).
Repro: `scratchpad/ncr_repro.py` (this session).

**BOTTOM LINE — the §G3-B10 diagnostic is MIS-SPECIFIED; its own premise
("handed the TRUE operator directly") is FALSE.** The loss/label/position/
head wiring is all CORRECT (suspects 1,2,4,5 CLEARED). The failure is a
structural READ-path defect (suspect 3, BUG-FOUND): the teacher-forced
operator `Z` is fit to the BIND-clause key reps but the read applies it to
the QUERY key rep — a *different* contextualized hidden of the same token —
so `Z` is not "perfect" for the read's actual input. Even a perfectly-fit
`Z` yields a garbage read. Three compounding defects (frozen value-adapter,
key/value two-space composition break, and a mis-based recovery instrument)
independently sabotage the diagnostic. The "loss WORSE than ln(24)" premise
is a MISREADING: 4.62 ≈ ln(107) (the answer-token marginal over the 107-name
train pool), the correct plateau for a task-blind model — NOT a malformed
loss.

### Per-suspect verdict

**Suspect 1 — loss on wrong token/span/label. CLEARED.** Traced end to end:
- `build_task1_document` (smoke `ncr_lm_wave1_smoke.py:369-395`): `doc =
  [bind tokens (T_bind) | query_window (query_len) | answer_token (1)]`
  (line 384). `answer_token = entity_ids[tgt_slot]` = the entity at
  `π^h(a_slot)` (line 381) — the CORRECT h-hop answer, a real GPT-2 name
  token id in [0, 50256].
- Forward drops the answer token: `input_ids = doc[:, :-1]`
  (`runner.py:260`), length `T_bind+query_len`, last col = `query_mark_col
  = T_bind+query_len-1` = the `<Q>` position (`smoke.py:389`).
- Logits are taken at `query_mark_col` ONLY (`inject_and_logits_last`,
  `smoke.py:292-306`); loss = `F.cross_entropy(logits, answer_token)`
  (`runner.py:624`) — a single answer position, NOT averaged over the
  sequence. Label/logit alignment is standard next-token (predict answer at
  `<Q>` from hidden at `<Q>`). No off-by-one. The loss target is well-formed.

**Suspect 2 — read injection never reaches decode. CLEARED.** `o →
read_injector(25→768) → ADDED to hidden[:, query_mark_col] → tied head`
(`smoke.py:303-306`). Verified against the backbone: `return_hidden=True`
returns the POST-`norm_f` hidden (`lm_pretrain_rd.py:1306-1308`) and the LM
head is `F.linear(x, self.embed.weight)` — TIED, no bias (`:1310`). So
`inject_and_logits_last` using `backbone.embed.weight` reproduces the
model's own head EXACTLY, and the injection perturbs the exact pre-logit
vector whose logits feed CE, at the exact position CE scores. Gradient flows
to `read_injector` (repro TEST A: `|grad|=7.6`). Injection reaches the
decode.

**Suspect 3 — operator/read basis or scale mismatch. BUG-FOUND (ROOT
CAUSE).** Four distinct defects, all demonstrated:

- **(3a) THE root cause — the teacher operator is fit to the wrong keys.**
  `teacher_force_operator` (`smoke.py:282-290`) builds `Z` s.t. `Z @
  keys_v[i] = values_v[i]`, where `keys_v = key_adapter(hidden @ BIND KEY
  positions)` (`extract_kv`, `smoke.py:259-270`). But the read computes `o =
  binexp_read(Z, q_key, h)` with `q_key = key_adapter(hidden @ query_key_col)`
  (`query_key`, `smoke.py:272-280`; called at `runner.py:267-268`). `q_key`
  and `keys_v[a_slot]` are `key_adapter` of the SAME token (`entity_{a_slot}`)
  at DIFFERENT positions/contexts of a causal backbone → different vectors.
  `Z` is "perfect" only for the bind keys it was fit on, never for the query
  key it is actually applied to.
  Repro TEST B (real `binexp_read`, h=1): `cos(o, ideal answer value rep)` =
  **1.000** when `q_key == keys_v[a_slot]` exactly, but **0.59 / 0.23 / 0.10**
  as the query rep drifts from the bind rep by 0.3/1.0/2.0×noise. The
  isolated-harness assumption `query_keys = keys.clone()` (`smoke_4`,
  `smoke.py:536`) is SILENTLY VIOLATED in the LM graft — nothing in the
  CE-at-`<Q>`-only objective ties `q_key` to `keys_v[a_slot]`. This directly
  falsifies §G3-B10's "handed the TRUE operator" premise: the read was never
  handed a usable operator.

- **(3b) value_adapter is FROZEN under teacher-force (gradient starvation).**
  `teacher_force_operator` detaches BOTH inputs (`k, v = keys_v.detach(),
  values_v.detach()`, `smoke.py:288`). `values_v`'s only consumer is that
  detached fit → `value_adapter` receives EXACTLY zero gradient; `key_adapter`
  gets gradient ONLY via the query path (its bind-side use is also detached).
  Repro TEST A: `value_adapter.grad = None`, `key_adapter.grad = 108`
  (query-path only). So "teacher-force isolates and lets the READ side train"
  is false — the read's output basis (value-adapter space) is frozen at random
  init, and `Z` is a NON-STATIONARY target rebuilt every step from drifting
  detached reps. read_injector chases a moving code.

- **(3c) key/value are separate spaces → deep composition (h≥2) is
  impossible.** `key_adapter` and `value_adapter` are DISTINCT maps
  (`smoke.py:251-252`). `Z` maps key-space→value-space, so `Z^h q` for h≥2
  chains a value-space vector back through a key→value operator — undefined
  unless `value_adapter ≈ key_adapter`, which nothing enforces. Repro TEST D
  (single K-cycle, real `binexp_read`): `cos(o, true h-hop answer)` =
  **+1.000 (h=1), −0.58 (h=2), −0.26 (h=3)**. The O(log h) deep-composition
  read — the ENTIRE point of NCR — cannot function with two adapters, even
  with perfectly matched keys.

- **(3d) the recovery INSTRUMENT is mis-based → reads ~0 even for a perfect
  read.** `recovered_frac_at_09` / `cosine_and_recovered_frac`
  (`smoke.py:417-427`, `runner.py:315-324`) compare `o` (which lives in
  VALUE-adapter space) to `target = key_adapter(hidden @ answer KEY position)`
  (`smoke.py:425`, `runner.py:322`) — a DIFFERENT linear image. Repro TEST C:
  a PERFECT read (`o` == the answer's value rep) scores `mean_cos = 0.05`,
  `recovered_frac@0.9 = 0.0`. So §G3-B10 Q2's "mean_cos noise-level → read
  carries no signal" is PARTLY an instrument artifact: this instrument CANNOT
  register a working read. It does not independently confirm the read is dead.

**Suspect 4 — answer alphabet / vocab / invalid label. CLEARED (and it
explains the numbers).** The label is a genuine single-token GPT-2 name id,
in-range for `vocab_size_total = 50259`; logits are over the full 50259
vocab. The "below chance" (0.026 vs 1/24=0.042) and "loss worse than ln(24)"
observations are BOTH explained by task-blind marginal collapse, not a
malformed label:
- With 213 verified names and `heldout_frac=0.5`, `n_train_names = 213 −
  round(213·0.5) = 213 − 106 = 107` (the training answer pool,
  `use_heldout_entities=False`). **ln(107) = 4.673.** Observed plateau ≈
  **4.62**. The model converged to the ANSWER-TOKEN MARGINAL over the whole
  107-name pool.
- `ln(24) = 3.178` is the WRONG floor: the model has no mechanism to restrict
  its output to the 24 in-document entities (it isn't told the answer is one
  of the present entities), so spreading mass over all 107 pool names is the
  correct task-blind optimum and is NECESSARILY "worse than ln(24)."
- Same reason accuracy is BELOW the 1/24 in-doc chance: argmax lands on a
  globally-frequent name (~1/107 hit rate on the specific answer), not on one
  of the 24 in-doc entities. A correctly-wired loss on a signal-free read
  produces exactly this. "Worse than ln(24)" is therefore NOT evidence of a
  broken loss.

**Suspect 5 — position/masking in the DeltaNet backbone. CLEARED as a
cause, with a load-bearing caveat.** The `<Q>` position carries the full
prefix via the recurrent state; taps (`key_pos`, `val_pos`, `query_key_col`,
`query_mark_col`) are all in-range and correct (verified against
`grammar_rd.sample_batch_rd`'s `item_pos = arange(K)·clause_len+buf_len+2`,
`grammar_rd.py:452`). CAVEAT (not a bug — intended asymmetry): the hop depth
`h` NEVER appears in the surface form — the query window is `[buf, query_key,
rel, <Q>]` with the SAME relation verb for all h (R2-7 congruence pin,
`grammar_rd.py:20-23`, `:474-478`); `h` enters ONLY out-of-band via
`batch["hop"]` into `binexp_read`. So `backbone_only` sees IDENTICAL input
for h∈{1,2,3} but must predict three different answers → it CANNOT beat the
marginal by construction. This is by design (it makes the read the only
h-aware pathway, hence "load-bearing"), and it is WHY both arms sit at
chance: backbone_only can't (no h), and full_graft's read doesn't work
(3a–3c).

**Suspect 6 — shared upstream bug failing both arms. RESOLVED (not a single
loss/label bug).** Both arms plateau at the ln(107) marginal for DIFFERENT
reasons that share one consequence: `backbone_only` lacks `h` (suspect 5
caveat); `full_graft`'s teacher-forced read carries no answer signal (3a–3c).
Loss/label/head are correct (suspects 1,2,4). There is no shared malformed
tensor — there is a shared OUTCOME (marginal collapse) with two causes.

### Root-cause ranking (what to fix, in order)

1. **(3a) query-key ↔ bind-key correspondence** — the decisive falsifier of
   the "perfect operator" premise. `Z` must be applied to a query vector that
   lives in the space it was fit on. Minimal diagnostic fix (to make the
   teacher-force test VALID): derive both `keys_v` and `q_key` from a
   CONTEXT-FREE source so the query key and its matching bind key are
   identical by construction — e.g. adapt the raw token EMBEDDING
   (`backbone.embed(entity_id)`) instead of the post-backbone contextualized
   `hidden`, OR (stronger, closes the whole graft) add an auxiliary alignment
   loss pulling `q_key → keys_v[a_slot]`. Until `q_key ≈ keys_v[a_slot]`, NO
   operator — learned or teacher-forced — can produce a correct read.
2. **(3c) single key/value space for composition** — tie/share ONE adapter
   for keys and values (or map values back to key space before re-application)
   so `Z: space→space` and `Z^h` chains. Without this, only h=1 could ever
   work; the deep ladder (h∈{5..61}) is structurally impossible.
3. **(3b) value_adapter gradient under teacher-force** — secondary: with (3a)
   fixed, a frozen-but-consistent value space is decodable by `read_injector`;
   but for a non-teacher-forced graft the detach means the write basis never
   co-trains. If keys/values are unified per (3c), this largely dissolves.
4. **(3d) recovery instrument** — compare `o` to a target in `o`'s OWN space
   (the value/composition space), not `key_adapter(answer KEY)`. Required
   before any future recovered_frac/mean_cos reading can be trusted; the
   §G3-B10 "read carries no signal" conclusion should be treated as
   INSTRUMENT-UNCONFIRMED.

### Disposition

The integration is **cheaply fixable at the DIAGNOSTIC level** — the
teacher-force run does not license "READ/setup broken" (§G3-B10) because it
never handed the read a usable operator (3a) and its signal instrument is
blind (3d). But the graft has a **genuine structural gap** (query↔bind key
correspondence + single-space composition) that must be fixed before ANY
"NCR can/can't train inside an LM" claim. A re-run is required to test
sufficiency (static analysis cannot prove convergence); the necessary fixes
are (3a)+(3c). Recommend: re-run the teacher-force diagnostic ONLY after
(3a) makes `q_key ≈ keys_v[a_slot]` by construction and (3c) unifies the
key/value space — and re-base the recovery instrument (3d) first so its read
is interpretable.

**Confidence.** HIGH that (3a)–(3d) are real and are the operative defects
(all four reproduced with the real `binexp_read` + exact `NCRIntegration`
ops; loss reframe is exact arithmetic: ln(107)=4.673 vs observed 4.62).
HIGH that the §G3-B10 diagnostic is mis-specified and does NOT establish
"read path broken." MODERATE-HIGH that (3a)+(3c) are NECESSARY fixes;
SUFFICIENCY (does the full graft then learn in-dist?) is unprovable
statically and needs the re-run.

## §G3-B12 SINGLE-SPACE READ FIX BUILD (Fable build agent, 2026-07-18)

Implements the coordinator-designed fix for §G3-B11's (3a)+(3c)+(3d)
structural read-path defects. BUILD + full real-CUDA SMOKE only — STOP
before the sanity launch (coordinator audits + runs the sanity cell).
Independent pre-run static audit dispatched (fresh agent, adversarial) —
PASS on all 10 checked dimensions, one non-diff-logic pre-launch hazard
flagged (old-schema checkpoint collision, mitigated below).

### The exact diff (both files, edited in place)

`experiment-runs/2026-07-17_ncr_gate3_wave1/{ncr_lm_wave1_smoke.py,
ncr_lm_wave1_runner.py}`. The mechanism change, verbatim:

1. **ONE shared `entity_adapter` on RAW (context-free) token embeddings.**
   `NCRIntegration.__init__` (smoke.py) — the separate `key_adapter` +
   `value_adapter` (each `Linear(768,25,bias=False)`) are REPLACED by a
   single `entity_adapter = Linear(768,25,bias=False)`. It is applied to
   `backbone.embed(entity_token_id)` — the backbone's OWN raw token
   embedding table, a plain lookup upstream of every DeltaNet layer,
   context-free by construction — for the KEY role, the VALUE role, AND the
   QUERY role. `extract_kv(token_ids, key_pos, val_pos, embed)` and
   `query_key(token_ids, query_key_col, embed)` now take the RAW
   `input_ids` (int64, `= batch["doc"][:, :-1]`, the exact tensor fed to
   the backbone) + `backbone.embed`, NOT the post-backbone contextualized
   `hidden`. Consequences, both closed: (a) Z: entity-space→entity-space is
   an ENDOMORPHISM so `Z^h` composes for any h (closes 3c); (b) the query's
   key vector for entity e is BIT-IDENTICAL to the bind key vector for e —
   same embedding row → same one Linear instance → same output, no context
   dependence — closing 3a exactly (`q_key == keys_v[a_slot]`, not merely
   ≈). `hidden` is still computed (unavoidable backbone forward) and STILL
   used for the read-INJECTION tap (RULING 2, unchanged: `o` is added into
   the contextualized hidden at the `<Q>` mark before the tied head — a
   separate design axis from where the KEYS come from, out of this fix's
   scope).
2. **Multi-subword entity reduction — NOT APPLICABLE BY CONSTRUCTION**
   (build brief asked to pick+document a reduction). `grammar_rd.py`'s
   `build_entity_pools`/`_verify_words` VERIFIES every entity candidate is
   single-token under GPT-2 BPE at build time and REJECTS any multi-token
   candidate outright (`EntityPools.train_name_ids`: "single-token-
   verified"). So `backbone.embed(entity_token_id)` is always exactly ONE
   embedding row per entity — there is no multi-subword case to reduce.
   Documented in `NCRIntegration`'s docstring as "not applicable by
   construction," not a silent assumption. (No STOP-and-report was needed:
   the flagged ambiguity does not arise for this task/tokenizer.)
3. **teacher_force_operator — mechanism UNCHANGED** (3 code lines byte-
   identical: `detach` both inputs, `pinv`, `transpose`). It still bypasses
   `ncr_head`'s BindingEncoder entirely (zero-grad, verified live). The
   only change is the space of its inputs: with the single shared adapter,
   the keys/values it is fit against and the query it is applied to now
   live in ONE consistent entity space.
4. **Recovery instrument re-based (closes 3d).** `recovered_frac_at_09`
   (smoke.py) and `cosine_and_recovered_frac` (runner.py) now compare `o`
   to `entity_adapter(embed(true answer_token))` — o's OWN (single, shared)
   space — instead of the old `key_adapter(hidden at answer KEY position)`
   (a different linear image). `answer_token` is the true h-hop answer
   entity's own token id (`build_task1_document`'s `gather(entity_ids,
   tgt_slot)`), so no position-gather through `hidden` is needed.
5. **Two-arm structure + audited safety preserved.** `full_graft` vs
   `backbone_only` (o≡0 via `torch.zeros_like`), the read-ablation
   EXACT-zero assertion, and the teacher-force per-step encoder-zero-grad
   assertion are all UNTOUCHED (the change only affects keys/values/query
   extraction, never `o_injected` or the ablation edge). Param count drops
   `3·768·25 → 2·768·25` (one adapter removed): NCRIntegration linear/add
   now **38,400** params (entity_adapter 19,200 + read_injector 19,200),
   was 57,600. Total NCR-related addition to the 97,618,176-param backbone:
   173,209 + 38,400 = **211,609 (+0.217%)**.

### Smoke results (real CUDA, GPU 2, box `youthful-indigo-turkey`,
`/home/nvidia/ncr_g3b12_fix/`, fresh dir to avoid the old-schema ckpt
collision the audit flagged; production not disturbed, all 8 GPUs stayed
100% util)

**`ncr_lm_wave1_smoke.py` — ALL 15 ITEMS PASSED (wall 15.3s):**
- smoke 0c: NCRIntegration param count `38,400` == expected (the new
  single-adapter formula), confirming one adapter removed.
- smoke 7 (full graft, real grammar_rd Task-1 doc, h=2, CE-only, joint
  fwd/bwd/opt): finite grads through backbone(158)+ncr(47)+integ(2),
  loss_ce=11.05, peak_mem 2.03 GB.
- smoke 10 (teacher-force isolation): `Z` fit residual **1.64e-07**,
  `ncr_untouched=True` (encoder ZERO grad), backbone/**entity_adapter**/
  read_injector all train — isolation property holds under the single
  adapter.
- **smoke 9b — the (3d)-fix's OWN self-check (NEW):** a synthetic perfect
  read `o := entity_adapter(embed(answer_token))` scored through the REAL
  `recovered_frac_at_09` → **recovered_frac@0.9 = 1.000000**. The OLD
  instrument (§G3-B11 repro TEST C) scored a perfect read at
  mean_cos=0.05 / rec@0.9=0.0. The re-base is verified live, not just
  reasoned about. (smoke 9 also confirms the metric COMPUTES on a real
  untrained eval batch: rf=0.0 at init, as expected.)

**`ncr_lm_wave1_runner.py --mode smoke` — ALL 6 SUB-TESTS PASSED:**
- A (full 300-step two-arm run), B (checkpoint/resume 100→300, resumed not
  restarted), C (ceiling ABORT-BUDGET at step 3), D (whole-cell skip-if-
  COMPLETED), E1/E2 (teacher-force run + resume). **Read-ablation exact-zero
  check PASSED pre+post at EVERY subprocess (`max_abs_diff=0.00e+00`).**
  **Teacher-force encoder-zero-grad assertion passed every step
  (checks_passed=100 in E1, again post-resume in E2).**

**Diagnostic signal (teacher-force smoke, 200 steps, NOT a verdict — the
coordinator's sanity cell is the verdict):** under a teacher-forced
(perfect) operator the `full_graft` read now recovers the answer entity at
**mean_cos = 1.000 in-dist (h=1,2,3)** and the mean_cos GAP over the frozen
`backbone_only` baseline is **≈1.0 at EVERY deep depth through h=61**
(h=5:1.06, h=12:0.95, h=20:0.99, h=29:0.98, h=40:0.98, h=61:0.97);
recovered_frac gap ≈1.0 at nearly every depth. This is exactly the deep-
composition read §G3-B11 said was structurally impossible with two adapters
(repro TEST D: cos +1.0/−0.58/−0.26 at h=1/2/3) — it now composes because Z
is an endomorphism, and the re-based instrument registers it. **CAVEAT
(honest):** `answer_accuracy` is still 0.0 in-dist at 200 smoke steps — the
read is correct in ENTITY-ADAPTER space (cos=1.0 with the re-based target),
but the `read_injector → tied-LM-head` DECODE pathway is untrained at 200
steps, so the correct read does not yet land the right vocab logit. The
smoke proves the READ is fixed (composes + instrument registers); it does
NOT prove the full graft solves the task end-to-end — that is precisely
what the coordinator's sanity launch (below) tests.

### New md5s (box == local, byte-identical, VERIFIED)

| File | md5 (was, §G3-B11 HEAD) | md5 (now, §G3-B12) |
|---|---|---|
| `ncr_lm_wave1_smoke.py` | `6521ea5adf9208f7724561efc1b76296` | `bc105af69661e488ff95f5046e2bcd8a` |
| `ncr_lm_wave1_runner.py` | `29531d0c36e76e305514af631c130d8f` | `a411a87de08e4bc46bc80854fcb5b37f` |

Base dependencies UNCHANGED, box==local re-verified this build:
`ncr_models.py` `6d7b30a592bee11f6c2135165801742d`, `ncr_earlyln_scale.py`
`3a87fcc92bb8341203c5e8c1f039a0af`, `lm_pretrain_rd.py`
`34addd9d8cc6a3df5a367d0f18a2ee0e`, `grammar_rd.py`
`b7eeca0f6fc56210ef9c633fe719b540`.

### Independent pre-run audit (fresh adversarial agent) — one non-blocking
pre-launch hazard

Static audit PASSED all 10 dimensions (shapes/dtype, gather-index
correctness vs `sample_batch_rd`, the `q_key==keys_v[a_slot]` bit-identity,
re-based-target correctness, teacher-force unchanged, read-ablation
unaffected, param formula, 9b non-circularity, dead-refs, full-file
consistency). ONE real finding, NOT in the diff's logic: the state_dict key
rename (`key_adapter`+`value_adapter` → `entity_adapter`) breaks resume-
compat with the pre-fix on-box artifacts `wave1_calib_K24_s0.*` (ABORTED-
BUDGET, step 19026) and `g3b9_tf_diag.*` (COMPLETED) that sit at exactly the
default `--cell-id`/`--out`/`--ckpt-dir` paths. `restore_arms_and_opts`
would raise an uncaught RuntimeError on `integ.load_state_dict` against an
old checkpoint. **MITIGATION (applied):** the sanity cell below uses a FRESH
`--cell-id`/`--out`/`--ckpt-dir` (`_g3b12` suffix / fresh dir); the smoke
itself already ran in the fresh `/home/nvidia/ncr_g3b12_fix/` dir. No old
artifact is touched.

### PROPOSED SANITY LAUNCH (STOP HERE — coordinator runs it, ~1–2 GPU-h)

A SHORT teacher-force calibration to confirm the graft now LEARNS in-dist
end-to-end (the sufficiency test §G3-B11 said needs a re-run — does the
read_injector→head decode actually train given the now-correct,
now-composing read). Fresh paths, one free GPU (pick least-loaded via
`nvidia-smi`), do NOT disturb production:

```
# on box, from /home/nvidia/ncr_g3b12_fix/ , inside a tmux session:
CUDA_VISIBLE_DEVICES=<free-gpu> /home/nvidia/tdenv/bin/python3 \
  ncr_lm_wave1_runner.py --mode calibration --device cuda \
  --teacher-force-operator \
  --cell-id sanity_g3b12_tf_s0 --steps 3000 \
  --batch-size 32 --eval-batch-size 64 --warmup-steps 200 \
  --lr 3e-4 --ckpt-every 500 --eval-every 250 --ceiling-gpuh 2.0 \
  --out results/sanity_g3b12_tf_s0.json \
  --ckpt-dir results/sanity_g3b12_ckpts
```

Interpretation (coordinator, blind): with the perfect operator handed in,
watch `full_graft` **answer_accuracy** in-dist (h∈{1,2,3}) climb above the
`backbone_only` baseline (the read is load-bearing) AND `recovered_frac@0.9`
hold ≈1.0 in-dist and deep (the read composes). If accuracy trains up → the
graft LEARNS in-dist and the encoder (free-write, non-teacher-forced) arm is
the next test. If accuracy stays at chance despite mean_cos=1.0 reads → the
read_injector→head decode is the remaining blocker (a read-inject / P=1-
bottleneck question, not a composition question). Either way is now
INTERPRETABLE — the read is no longer the confound.

### Readiness verdict

**READY for the coordinator's audit + sanity launch.** Both files build,
all 15 smoke items + all 6 runner sub-tests PASS on real CUDA, the (3a)/
(3c)/(3d) defects are closed and each fix has its own live PASS (endomorphic
composition to h=61 under teacher-force; `q_key==keys_v[a_slot]` by
construction; the re-based instrument scores a perfect read at cos=1.0 via
smoke 9b), audited-safety invariants (read-ablation exact-zero, teacher-force
zero-grad) hold post-fix, md5s pinned box==local, and the one audit-flagged
pre-launch hazard (old-schema ckpt collision) is mitigated by fresh paths.
STOPPING before launch per the build brief.

## §G3-B13 SANITY CELL VERDICT — DECODE PATH HEALTHY, HARNESS PROVEN (coordinator, 2026-07-18)

**Cell:** `sanity_g3b12_tf_s0` — teacher-force calibration, 3000 steps,
lr 3e-4, batch 32, seed 0, GPU2 co-resident with production training (all 8
GPUs stayed hot). Ran to term COMPLETED (~2480s wall / ~0.7 GPU-h, under the
2.0 ceiling), tmux `ncr_sanity_g3b12`. Raw: `results/sanity_g3b12_tf_s0.json`
(archived). Coordinator read the raw JSON directly (not an agent summary).

**VERDICT: DECODE PATH FULLY HEALTHY. The GATE-3 harness is proven end-to-end
(write-given → read composes → decode trains).** With the true operator handed
in (teacher-force), the trained `full_graft` arm decodes the composed read to
the EXACT answer token:

| depth h | full_graft answer_acc | full_graft mean_cos | backbone_only (o≡0 null) answer_acc |
|---|---|---|---|
| 1, 2, 3 (in-dist) | **1.0** | 1.0 | 0.094 / 0.0 / 0.078 |
| 5, 12, 20, 29, 40 (deep) | **1.0** | 1.0 | 0.016–0.078 |
| 61 (deep) | **1.0** | 0.9971 | 0.047 |

`recovered_frac@0.9` = 1.0 full_graft at every depth (h=61: 0.984); 0.0 at
every depth for the null. Loss: full_graft 11 → **0.24**, backbone_only pinned
at **~4.5** (≈ ln 107, the train-answer marginal) — the read is load-bearing,
the null cannot decode without it. Clean two-arm separation (1.0 vs ~0.05).

**What this proves / does NOT prove (honest):**
- PROVES: the read-injection → tied-head DECODE path trains; the two-adapter
  bug (§G3-B11/§G3-B12) was the ONLY thing breaking §G3-B8's UNINTERPRETABLE
  run — with it fixed, the full pipeline works. The make-or-break is now
  INTERPRETABLE: the only remaining unknown is the ENCODER's WRITE-learning.
- Also demonstrates (part of NCR's value prop, exactness-by-construction):
  GIVEN in-context operators, a real 98M LM performs exact O(log h)
  composition reads recovering the answer at **h=61** from a fixed
  constant-size state — deep-h generalization here is BY CONSTRUCTION under
  teacher-force (exact operator + exact Z^h), NOT yet a learned-write claim.
- Does NOT prove: that the model LEARNS to WRITE operators from context that
  compose to deep h. That is exactly the make-or-break (non-teacher-force)
  capability test — the real claim, and the sole remaining question.

**NEXT:** design the make-or-break re-run (non-teacher-force; encoder writes
the operator; two-arm; n-seeds; ceiling). Re-verify the novelty gate FIRST —
the make-or-break is the real capability claim and re-enters the gate.
Integration odds materially improved: NCR's core mechanism now demonstrably
works inside a real LM; the flagship's make-or-break is one write-learning
question away, no longer confounded by a broken read.

## §G3-B14 MAKE-OR-BREAK RE-RUN — PLAN (coordinator, 2026-07-18; gated on novelty + audit)

The make-or-break is ALREADY fully pre-registered and gauntlet-frozen — this
re-run reuses it verbatim, NO redesign. What changed since §G3-B8's
UNINTERPRETABLE verdict is ONLY the harness: the single-space read fix
(§G3-B12, verified) + the §G3-B13 sanity that DISCHARGED the exact confound
(the two-adapter read bug) that made §G3-B8 uninterpretable. An in-dist lift
is now achievable, so the frozen PRECONDITION can actually be met.

**Config (frozen §G3-B7, minus `--teacher-force-operator`, with the FIXED
runner md5 `a411a87d` single `entity_adapter`):**
`--mode calibration --steps 20000 --lr 3e-4 --batch-size 32 --eval-batch-size
64 --warmup-steps 200 --ckpt-every 10000 (HIGH, cap disk) --eval-every 1000
--seed 0 --ceiling-gpuh 5.0`, both arms (full_graft + backbone_only o≡0),
BLIND (this IS the capability verdict — coordinator does NOT read live metric
values pre-assess). NON-teacher-force: the encoder (BindingEncoder) must WRITE
the composing operator from context ITSELF — the sole remaining unknown.

**Attribution rule: UNCHANGED, frozen §G3-B5/§G3-B7.** PRECONDITION (metric-b
answer_accuracy): backbone_only materially BELOW full_graft (read load-bearing)?
NO → UNINTERPRETABLE (but the §G3-B13-discharged confound means both-at-chance
is now far less likely). YES → PRIMARY (metric-a recovered_frac@0.9 GAP at deep
ladder {5,12,20,29,40,61}): full_graft materially exceeds the frozen null at
deep depth? PASS = make-or-break passed (encoder LEARNS to write composing
operators in a real LM → the capability) → up the 98M→392M→1B+ ladder. FAIL →
pre-wired ablations (--adapter mlp / --read-inject b / curriculum / higher LR)
BEFORE any "NCR can't write" conclusion.

**Discipline:** n=1 FIRST-SIGNAL (frozen §G3-B7 "n=1 first, add seeds if
borderline" + CLAUDE.md "calibration before big sweep mandatory"); ~4.9 GPU-h,
<10 GPU-h → 1 audit round. If PASS/borderline → escalate to n=3 verdict-of-
record (~15 GPU-h, >10 → full red-team). The CRUX the audit must probe: under
teacher-force deep-h=61 worked BY CONSTRUCTION (exact operator); under non-TF
the LEARNED operator must be precise enough that Z^61 still recovers — a high
bar (the free-write K=24 toy reached rec 1.0 all depths, but the graft is
unknown).

**Two GATES before launch:** (1) NOVELTY re-gate — 2 external sweeps (by-task +
by-mechanism) dispatched + internal-archive sweep done by coordinator (CLEAR:
no prior clean make-or-break verdict; §G3-B8 was the bug; KILL_LIST entries are
the retired bolt-on matrix-CODI rank work). Adjudicate before launch. (2) ONE
Opus adversarial audit of the non-TF re-run config + any single-adapter
write-path interaction + resource/placement. STATUS: gated, NOT yet launched.

### §G3-B14 NOVELTY RE-GATE — DISCHARGED (coordinator adjudication, 2026-07-18)

Triple sweep complete (2 external Sonnet, web-verified live; 1 internal by
coordinator). **VERDICT: CLEAR / differentiable — NOT scooped. Gate discharged
for the make-or-break launch.**
- **By-mechanism: CLEAR.** No verified 2023–2026 work combines (a) an
  in-context-WRITTEN matrix operator from a learned encoder, (b) EXACT
  repeated-squaring composition, (c) O(log h) reads from a fixed constant-size
  state, trained end-to-end. Nearest split the mechanism: log-depth over the
  TOKEN SEQUENCE (Log-Linear Attention 2506.04761 ICLR'26; PSM/Yau 2506.10918)
  or sequentially-composed structured transitions (MuonSSM 2606.30461;
  DeltaProduct 2502.10297; PD-SSM 2509.22284).
- **By-task: PARTIAL-OVERLAP (differentiable), not scooped.** Every neighbor
  overlaps ≤2 of the 5 defining components. New near-neighbors to CITE +
  differentiate: Wang/Nichani/Bietti 2505.23683 (COLT'25, O(log k) via
  transformer LAYERS — add to standing prior-art, same lineage as the cited
  Nichani-Lee-Bietti 2412.06538); recurrent-depth LOOPING depth-generalization
  (Kohli 2604.07822, Chen 2603.21676); KLA 2602.10743 (A5 via Bayesian scan).
- **Internal-archive: CLEAR.** No prior CLEAN make-or-break verdict (the only
  one, §G3-B8, was the two-adapter bug); KILL_LIST entries are the retired
  bolt-on matrix-CODI rank work, not this.
- **DIFFERENTIATOR (state at launch/in paper):** in a real 98M+ LM the model
  LEARNS to WRITE the composing operators itself from in-context bindings into
  a fixed constant-size matrix state, and reads exact composed answers at
  unseen hop-depths via O(log h) repeated-squaring — vs O(log k)-via-LAYERS,
  depth-gen-via-LOOPING, or composition-via-Bayesian-scan.
- **Non-blocking flags (full-text recheck IF the paper cites specifics):** Yau
  2506.10918's "S5 train 4-18/test 180" numbers (abstract didn't confirm);
  MLP-LDRU 2605.26035 (second structural analog).

## §G3-B15 MAKE-OR-BREAK LAUNCHED (non-TF, n=1 first-signal, 2026-07-18 09:17 UTC)

**BOTH pre-launch gates cleared.** (1) Novelty DISCHARGED (§G3-B14). (2) Opus
adversarial audit = FIX-FIRST → CLEAR: the #1 target (does the §G3-B12
single-adapter fix, verified only under teacher-force, leave the ENCODER
write-path coherent under non-TF?) came back **CLEAN** — the encoder consumes
`entity_adapter(raw embed)` (no dangling removed-adapter ref), writes Z as a
(25,25) endomorphism in the SAME entity space as q_key/target by construction,
empirically corroborated by §G3-B13's teacher-force composition to h=61; the
encoder is trainable under non-TF (runner:274 `ncr_head.encode` on the non-TF
branch, runner:370 ncr params in optimizer — coordinator cross-checked vs
code). Two REQUIRED launch-hygiene fixes APPLIED: (i) fresh non-colliding paths
in the `ncr_g3b12_fix` tree + the stale old-schema `wave1_calib_K24_s0.ckpt.pt`
(2.2G) deleted (avoids the silent-no-op / load_state_dict-crash-loop the audit
found); (ii) runner md5 `a411a87d` verified at the launch site + correct flags
(ckpt-every 10000 / eval-every 1000).

**LIVE:** tmux `ncr_mob_g3b14_s0`, GPU 2 (~37GB free co-resident, memprobe-
cleared), proc 2530483, self-healing supervisor `run_mob_g3b14.sh` (terminal-
status-gated break on `"status":"COMPLETED|ABORTED"`, ckpt-resume-safe on crash,
BLIND — greps status only, never metrics). Config: `--mode calibration --steps
20000 --lr 3e-4 --batch-size 32 --eval-batch-size 64 --warmup-steps 200
--ckpt-every 10000 --eval-every 1000 --ceiling-gpuh 5.0 --seed 0`, NON-teacher-
force (encoder writes Z), both arms (full_graft + backbone_only o≡0), out
`/home/nvidia/ncr_g3b12_fix/results/mob_g3b14_s0.json`. ~4.7h wall / ~4.9 GPU-h.
n=1 FIRST-SIGNAL. BLIND — coordinator does NOT read live metric values (the
out-JSON has live eval values; poll structural status only).

**ON COMPLETION — apply the FROZEN §G3-B5/§G3-B7 attribution rule (blind
assess):** PRECONDITION (metric-b answer_accuracy): backbone_only materially
BELOW full_graft (read load-bearing)? YES → PRIMARY (metric-a rec@0.9 GAP at
deep ladder): full_graft materially exceeds the frozen null at deep depth? PASS
= make-or-break passed → escalate n=3 verdict-of-record + up the 98M→392M→1B+
ladder. NULL/FAIL → route the PRE-REGISTERED ablations (--teacher-force /
--adapter mlp / --read-inject b / curriculum / higher-LR / more-steps) FIRST —
**audit (g) caution: 20K×batch32 = 640K docs is ~32× under the free-write toy's
convergence budget (20.5M) with CE-only indirect signal, so a deep-h NULL may
be write-learning-UNDERPOWERED, NOT "NCR can't write" — do NOT over-read a
null.** Stratify deep-h by h%K (h=5≡h=29 mod 24; audit (e)).

## §G3-B16 MAKE-OR-BREAK VERDICT (blind Opus judge + coordinator cross-check, 2026-07-18)

**VERDICT: UNINTERPRETABLE (precondition fail — both arms at the answer-marginal
floor).** Clean COMPLETED run (20000/20000 steps, 1.10 GPU-h, 0 errors). Blind
Opus judge assessed the raw JSON against the frozen §G3-B5/§G3-B7 rule; coordinator
cross-checked the load-bearing numbers directly from the raw JSON — CONFIRMED:
- PRECONDITION (metric-b answer_accuracy, in-dist h∈{1,2,3}): full_graft mean
  **0.036458** vs backbone_only mean **0.036458** — **IDENTICAL**, both ≈ the
  answer-marginal floor (chance 1/107=0.0094; all cells 0.016–0.094 = binomial
  noise). Read NOT demonstrably load-bearing → **precondition FAILS**.
- PRIMARY (metric-a recovered_frac@0.9, deep ladder): **0.0 in all 18 (arm×depth)
  cells**, mean_cos ≈ 0 everywhere (−0.049..+0.034). Zero deep-composition signal
  (moot given precondition, but a clean FAIL-branch had it mattered).
- read_ablation exact-zero verified pre+post (max_abs_diff 0.0) — the null is a
  true frozen null; the result is NOT a harness artifact.

**HONEST READ — what this IS and ISN'T:**
- NOT a harness bug. §G3-B13 PROVED the harness end-to-end (teacher-force
  answer_acc 1.0 to h=61). The read + decode work when handed the operator.
- NOT "NCR can't write." The isolated failure is WRITE-LEARNING: the encoder
  did not learn to write a useful composing operator from context in 20K steps
  of CE-only indirect signal. The Opus pre-launch audit (§G3-B15 audit (g))
  PRE-FLAGGED this exact risk: 20K×batch32 = 640K docs is **~32× UNDER** the
  free-write toy's 20.5M convergence budget, AND the toy used a DIRECT cosine
  read-loss while the graft has only indirect CE signal to the encoder.
- FAINT signal that the encoder is trying: full_graft final LM loss **3.91** vs
  backbone_only **4.31** (~0.39 nats lower) — the read reduces sequence loss but
  does NOT convert to answer_accuracy (still at floor) ⇒ the encoder found a
  weakly-useful-but-not-composing use for o. Under-trained, not inert.

**DIAGNOSIS: write-learning is signal-starved (indirect CE) and budget-starved
(32× under toy). Teacher-force isolates the gap to the WRITE specifically.**

**NEXT (pre-registered FAIL-branch route — the direct-read-supervision arm):**
NOT n=3 of the same config (a clean identical-floor is not a seed fluke; re-running
underpowered only re-confirms the null). Instead the highest-leverage pre-registered
ablation: **add a DIRECT read-supervision auxiliary loss** (cosine/recovery on the
read output o vs the true answer-entity target — the dense signal the toy used to
converge), optionally + more steps/curriculum. This converts the encoder's
gradient from indirect-CE to direct, matching the regime that made the standalone
free-write toy reach rec 1.0 all depths. Still a valid capability demo: the
encoder does its own writing; the aux loss is a disclosed training aid; at
inference NO teacher-forcing. If the encoder learns to write under direct
supervision → the capability holds; if not → a deeper (real) negative. This is a
code change (new loss term + flag) → build → audit → launch, mirroring the
§G3-B12 fix arc.
