# NOVEL-ARCHITECTURE WATERFALL — design registry

Opened 2026-07-09 under the PI's not-just-a-fix directive ("I don't think we
can publish just a fix to DeltaNet... get some novel architectures"). This
file is the waterfall's canonical record (created at attack-stage recording
time; stages 1-2 predate it and are summarized from `STATE.md` §6 — their
full transcripts lived in coordinator context and were compacted, a gap this
registry exists to prevent recurring).

## §1 Stages 1-2 (brainstorm + research, RETURNED 2026-07-09)

**TOP CANDIDATE — Native Composition Reads (NCR):** read via query-selected
matrix powers/products of the fast-weight state (`o = read(Z^h, q)`,
generalizing to relation chains `Z_rn···Z_r1`). Capability claim as
originally recorded: single-pass variable-depth exact relational
composition, no CoT. Novelty recorded as OPEN with closest prior art on
different axes: fast-weight PKM (arXiv:2601.00671), MAGNA
(arXiv:2009.14332), DeltaProduct (arXiv:2502.10297); operator-bank
sub-experiment to distinguish from RotatE (arXiv:1902.10197,
offline/static vs ours online/in-context). First wave ≈35-50 GPU-h on the
Task E harness. PonderNet halting-collapse objection claimed pre-answered
by a closed-form `‖C‖·h` leakage stopping rule.

**SECOND TRACK (parallel-able, not yet attacked):** rank-budgeted writes
(per-context rank allocation at the write step; novelty gap verified against
arXiv:2602.04852 / 2602.02195, both descriptive-only, and Elastic Spectral
SSM's global-only budget); ≈25-35 GPU-h.

**Cheap piggyback:** orthogonal-complement novelty detector on archived
Z-dumps (near-zero GPU).

**DO-NOT-BUILD list (reconstructed from `STATE.md` §6; full transcript
compacted):** Grazzi / DeltaProduct / RWKV-7 / TPR / RotatE territory.

## §2 ATTACK ROUND 1 (fresh-eyes adversarial, 2026-07-09 overnight): NEEDS-MAJOR-REVISION

Read-only round; primary sources web-verified; Task E artifacts
(`TASK_E_FINDINGS.md` §3/§9/§10) used as measured evidence. No kill-shot
lands on the literal mechanism (no paper found reading query-selected powers
of an in-context-written fast-weight matrix), but two configurations are
fatally broken as specified and the novelty record omitted its closest
prior art.

### Findings (ranked)

- **F1 [FATAL to the soft/learned-h variant].** A differentiable mixture
  over powers is a MATRIX POLYNOMIAL, not "reading Z^h": for cycle
  operators (eigenvalues = roots of unity, TASK_E §9) mixing powers mixes
  phases — destroys the spectral exactness the claim rests on — and the op
  is occupied (MAGNA arXiv:2009.14332 = geometric mixture of attention
  powers; MEA/HLA arXiv:2510.27258 = powers of input-dependent operator to
  fixed order). Hard selection (ST/REINFORCE over h) is the documented
  PonderNet-collapse cousin; the stopping rule does NOT pre-answer it (it
  bounds numerical trust of Z^h, it does not select h). Only INPUT-SUPPLIED
  h survives wave 1 — and then "query-selected" overclaims; frame as
  input-controlled-depth program execution, all baselines receiving the
  same h signal.
- **F2 [FATAL to the "held-out depth" headline on the single-cycle task].**
  On a single K-cycle, h≫train collapses via `h mod K` to shallow
  in-distribution hops (the documented hard-rule trap; Task E's own
  effective-hop stratification). What's genuinely new at large h is
  spectral/numerical amplification — a depth-ROBUSTNESS/exactness claim,
  not depth-generalization. Re-scope the claim or change the task.
- **M1 [MAJOR, novelty].** Closest prior art omitted: **FWM (Schlag,
  Munkhdalai & Schmidhuber, ICLR 2021, arXiv:2011.07831)** does single-pass
  in-context read-time multi-hop relational composition through fast
  weights (recursive reads, retrieved value re-used as next key), for
  transitive inference, 2020. Surviving deltas: FWM hop count fixed
  (N_r=3), reads nonlinear (LN between hops), composition approximate, no
  exactness/provable-rank verification. For the operator bank, the right
  nearest neighbors are TensorLog (arXiv:1605.06523) / Neural-LP
  (arXiv:1702.08367) / DRUM (arXiv:1911.00055) — query-conditioned
  variable-length products of relation matrices over static KGs — not
  RotatE; Guu et al. (arXiv:1506.01094) documented composition
  error-cascading in 2015. MesaNet (arXiv:2506.05233) ships
  query-conditioned matrix-FUNCTION reads ((G+λI)⁻¹q via CG) at scale.
  **Surviving novelty = the CONJUNCTION ONLY: in-context-written operators
  + exact linear composition + variable/held-out depth + provable-rank/
  causal verification.** FWM (or an FWM-style recursive-read arm) belongs
  in the baseline set.
- **M2 [MAJOR].** The `‖C‖·h` stopping rule is UNSOUND against this
  project's own data: TASK_E §9(d) measures spectral_radius(D) at 1.0-2.9
  (≥1 in every converged seed), so leakage compounds geometrically
  (≈ ‖C‖·ρ(D)^h/(ρ−1)), not linearly — the linear rule underestimates
  error exactly at decisive far depth. Also scale-gameable (free isotropic
  scale c* ∈ [1.0, 2.8] invisible to cosine loss; ‖C‖ not
  scale-invariant — normalize, e.g. ‖C‖/σ_min(A)). Binding: re-derive with
  the geometric term + scale normalization AND run the negative test
  (construct a Z with small ‖C‖, ρ(D)>1 that the linear rule wrongly
  admits; corrected rule must reject).
- **M3 [MAJOR, launch-blocking].** Strongest vector counterfactual absent;
  the inherited C_MLP one-hot(h) baseline is a strawman by TASK_E §5's own
  admission ("architecturally unable to extrapolate"). Reshape-equivalence:
  Z^h q IS h steps of a linear RNN at read time — "matrix vs vector"
  cannot be the claim; the claim is "exact linear-operator composition vs
  iterated NONLINEAR maps (which drift)". Required baselines: (i)
  param-matched iterated vector map, looped/UT-style (Looped Transformers,
  arXiv:2409.15647; depth-recurrent arXiv:2603.21676), same h signal;
  (ii) FWM-style recursive read. Depth-amplification predicts they lose at
  large h — that prediction IS the experiment. Note: looped transformers
  existing makes "current architectures cannot do variable-depth
  single-pass composition" false as stated (see M4 for what survives).
- **M4 [MAJOR, redundancy + charter].** Stage 2's recurrent composer
  already adjudicates single-pass exact composition at held-out depths
  (O(D) write-time). NCR's non-redundant delta is exactly ONE thing:
  **composition depth decoupled from context length at O(log h) sequential
  cost via repeated squaring** (associativity; in-weights analysis
  arXiv:2602.03655 — cite; in-context version open). Naive Z^h loop is
  O(h) — same order as looping — so binary exponentiation must be BUILT
  and CLAIMED or "single-pass" is an accounting convention. The relation
  chain Z_rn···Z_r1 has NO squaring shortcut (heterogeneous products
  sequential), loses log-depth, sits in Neural-LP/DRUM/FWM territory, and
  skirts the DO-NOT-BUILD edges. Binding: wave 1 = single-relation,
  input-supplied h, log-depth+exactness headline; operator bank gated
  behind wave 1 readout AND Stage 2's calibration readout (which settles
  fla-vs-torch and readout diagnostics first — no GPU before those
  lessons land).
- **M5 [MAJOR, cost].** 35-50 GPU-h is 2-4× light for the stated unified
  wave (measured anchor: Task-E 80K-step run ≈2.4 GPU-h; the figure buys
  ~15-20 runs — one arm × one K × 5 seeds — with no calibration cell, no
  iterated-map arm, no operator bank, no 2-2.5× budget-artifact retest
  headroom). Bespoke fp32 torch at d=16 is the right call (no fla kernel
  computes powers; the 3-8× envelope band doesn't apply). Binding:
  Stage-2-style ledger — calibration first, capped single-sub-experiment
  wave 1, operator bank separately ledgered/gated.
- **M6 [MAJOR if attention-reader readout; MINOR if Task-E direct read].**
  Task E's native read (o = Z^h q, continuous cosine) does NOT use the
  row_queries reader, so §1.30's degeneracy doesn't auto-apply — but any
  integration into the Stage-1/2 model family (reshaped (B,32,32) state
  through the reused reader) inherits it at full strength (Z^h rank-K,
  near-collinear rows). Binding: pin the readout; if attention-reader, the
  read-vector-std diagnostic goes in the calibration gate
  (CAPABILITY_STAGE2_ATTACK_R1.md finding 1).
- **m1-m4 [MINOR]:** gradient norms through Z^h grow like h·ρ^{h-1}
  (Task E only ever backpropped h≤3 — keep or re-adjudicate; eigh-backward
  instability precedent); fp32 mandatory at far depth; binary
  exponentiation doesn't fix backward norm growth and is incompatible with
  soft selection. Blank-out/P=1 re-verified for the NEW forward pass, not
  inherited. Depth-signal hygiene: identical h signal to all arms; stratify
  by effective hop; exact thresholds with executed negative tests.
  Cosine-bar slack known (fr=7 passes @0.9 through h=7): deep probe
  mandatory per cell; report §9-style spectral phase residual alongside.

### Verdict and binding disposition

**NEEDS-MAJOR-REVISION.** Surviving claims: (1) in-context-written relation
operators composed EXACTLY at read time beyond training depth; (2) O(log h)
sequential cost via repeated squaring — genuine separation from CoT (O(h)
tokens), looped transformers (O(h) loops), recurrent composers (O(D)
steps), FWM (fixed N_r, approximate); (3) depth-amplification as the
pre-owned instrument making "exact" measurable. NOT surviving: "novelty
OPEN" as recorded; learned h-selection in any form; "held-out depth
generalization" on a single K-cycle; the linear stopping rule; the 35-50
GPU-h unified budget.

**Single highest-value change (binding on Rev 1):** pin wave 1 to
input-supplied h, single relation, binary-exponentiation read, with a
param-matched looped/iterated-vector-map baseline given the same h signal,
and FWM cited and distinguished — simultaneously defuses F1, M3, M4, and
most of M1, leaving F2/M2 as bounded revisions.

**Strongest post-revision risk (pre-register the answer):** the narrowed
claim may read as an efficiency/exactness separation rather than the PI-bar
capability separation, since looped baselines reach the same answers at
O(h). Rev 1 must pre-register an operating regime (h large enough,
precision tight enough) where O(h) baselines measurably FAIL and NCR does
not — that regime is the capability claim.

Security: the round reported one system-channel date-change+concealment
sighting (same vector already tallied in `STATE.md` SECURITY NOTE ≥69; not
double-counted); no stdout-embedded fakes this round.

## §3 Rev 1 (2026-07-09)

Design-only revision addressing every §2 finding (F1/F2, M1-M6, m1-m4)
per the binding disposition. One-sentence summary of what changed:
wave 1 is pinned to input-supplied h / single relation / scale-managed
binary-exponentiation reads on the Task-E d=16 harness; the headline is
RE-SCOPED from "held-out depth generalization" to depth-robust EXACTNESS
at O(log h) sequential cost, with a pre-registered capability regime
where O(h) baselines are predicted to fail; the baseline set, trust
rule, novelty record, and ledger are rebuilt from scratch.

### 3.1 Wave-1 pinned configuration (F1, M6, m1-m2)

| Axis | Pin |
|---|---|
| Harness | Task E, `d=16`, fp32 throughout, orthonormal keys, permutation variant with the single-full-K-cycle + residue guard (`_permutation_graph`, `TaskEConfig.__post_init__`) — the `CLAUDE.md` mod-K hard rule's own fix, inherited verbatim |
| Relation | ONE in-context-written operator per episode (single relation; the operator bank is M4-gated, §3.6) |
| Depth signal | h is INPUT-SUPPLIED, identical to every arm, as a raw integer consumed as exponent (NCR) or loop count (baselines); no arm receives one-hot(h) or any `h mod K` feature |
| Read | Direct matvec `o = Z^h q`, continuous cosine scoring — Task E's native read, NOT the Stage-1 `row_queries` attention reader; per M6, any arm whose read deviates from this gets the read-vector-std diagnostic (§1.30 P3 method; pass ≥ 0.04 across queries, degenerate reference 0.000e+00 vs healthy 0.41 — bar HARMONIZED (Rev 2, mi4) to Stage 2's derived value with the same derivation, `CAPABILITY_SEPARATION_DESIGN.md` §2.8 item 2(e): one decade below the 0.41 healthy anchor, bias-toward-FAIL by asymmetric costs; Rev 1's 0.05 had no independent basis and is retired) as a pass/fail item in its calibration gate |
| Power computation | Binary exponentiation (⌈log₂h⌉ squarings + ≤⌈log₂h⌉ products) — this IS the headline; the naive O(h) loop is DISALLOWED as the claimed configuration and kept only as a disclosed cost-control arm (§3.3) and fp32 cross-check |
| Scale management | Frobenius renormalization at BOTH pin points (Rev 2, MA5 — Rev 1 said only "running product"): the squared BASE, `Z_{k+1} := (Z_k·Z_k)/‖Z_k·Z_k‖_F` after EVERY squaring, AND the running partial product after every multiply, log-scales tracked separately for each — exactly cosine-invariant (a positive scalar per step cannot move cosine), and REQUIRED: measured ρ(D) reaches 2.9 and c* reaches 2.8 (`TASK_E_FINDINGS.md` §9), so unmanaged fp32 squaring overflows (3.4e38) at h ≈ 83 — inside the capability window |
| Train-time supervision | Cosine loss at h ∈ {1,2,3} only, backprop through ≤3 naive matmuls — Task E's own regime; all deeper h are eval-only under `no_grad` (m1 adopted, no re-adjudication: gradient norms through `Z^h` grow like h·ρ^(h-1), and the fr=8/9 eigh-backward instability precedent, n_skipped_steps 3-10, is exactly the class of risk this avoids; no eigendecomposition anywhere in the training path) |
| Precision instrument | fp64 shadow reads on the full h-grid per cell (d=16 makes this CPU-trivial); a (cell, h) point with \|cos_fp32 − cos_fp64\| > 0.005 is flagged NUMERIC-DIVERGENT |

**Why learned h-selection is dead BOTH ways (F1, documented as
binding, not revisitable without a new waterfall pass):** (i) SOFT — a
differentiable mixture over powers Σ_i w_i Z^i is a matrix POLYNOMIAL,
not Z^h: for cycle operators the eigenvalues are roots of unity scaled
by c* (§9), so mixing powers mixes eigenphases and destroys the
spectral exactness the entire claim rests on; the op is also occupied
(MAGNA arXiv:2009.14332 = geometric mixture of attention powers;
MEA/HLA arXiv:2510.27258 = powers of an input-dependent operator to
fixed order). Binary exponentiation is additionally incompatible with
soft selection (it needs one concrete integer exponent) — m2's
incompatibility note is therefore moot, as is its forward-only caveat
(bin-exp does not fix backward norm growth; we never backprop through
deep powers, §3.1 train pin). (ii) HARD — straight-through/REINFORCE
over discrete h is the documented PonderNet-collapse cousin
(`CLAUDE.md` hard rule), and the trust rule cannot rescue it: it
bounds numerical trust of Z^h, it does not select h (§3.4). "Query-
selected" is retired from all claim language; the honest frame is
input-controlled-depth program execution.

### 3.2 The capability regime (F2 + the §2 "strongest post-revision risk")

**Headline re-scope (F2, Option A adopted with justification).** On a
single K-cycle the reachable set of any start entity has size K and
the correct answer depends only on `h mod K` — at d=16 with
orthonormal keys, K ≤ d = 16 caps the reachable set BY CONSTRUCTION,
so no h-grid choice inside this harness can make the reachable set
grow with h; the task variant that genuinely grows it (multiple
relations / heterogeneous chains) IS the operator-bank sub-experiment
M4 gates behind wave-1 + Stage-2 readouts, and reaching for it now
would also forfeit the O(log h) squaring shortcut (heterogeneous
products have none, §2 M4). The wave-1 headline is therefore
re-scoped: **depth-robust EXACTNESS — an in-context-written operator
composed h-fold at read time stays exact (recovered_frac@0.9, the
standing bar) to depths where every O(h) sequential baseline's
compounding per-step error has destroyed recovery, at O(log h)
sequential cost — stratified by effective hop, never claimed as
reachable-set/compositional-generalization growth.**

**h-grid (pinned; every reported h carries its `(h, h mod K)` pair;
aggregates never pool across residues).** Two components per K:

- **Fixed-residue geometric ladder** (isolates pure composition count
  at constant effective hop — the §3/§9 h=21≡5 (mod 8) precedent,
  generalized): K=8: h ∈ {5, 13, 21, 29, 61, 125, 253, 509, 1021},
  all ≡ 5 (mod 8). K=12: h ∈ {9, 21, 45, 93, 189, 381, 765, 1533},
  all ≡ 9 (mod 12) (21 mod 12 = 9 is Task E's own "genuinely novel
  ground" residue). Train-support {1,2,3} and legacy {4,5,6,7}
  retained for table continuity.
- **Full residue sweep at one depth** (verifies residue-independence
  and proves the mod-K bookkeeping has teeth): K=8: h ∈ {57..64}
  (residues 1..7 and 0; h=64 ≡ 0 is the identity residue, included
  DELIBERATELY, labeled, and excluded from all aggregates). K=12:
  h ∈ {49..60}, same convention. **Eval-grid pathway (Rev 2, MA2):**
  as pinned, the sweep ASSERT-CRASHES against the inherited
  periodicity guard (`TaskEConfig.__post_init__`, `task_e.py:121-132`
  — h=64/60 hit the identity assert; h=57-59 (K=8) / 49-51 (K=12)
  hit the train-residue assert). Resolution is a pinned TWO-MODE
  constructor, design-level spec the build implements: config flag
  `TaskEConfig.eval_grid_mode ∈ {"claim", "residue_sweep"}`.
  `"claim"` (the default and the ONLY mode any claim-feeding path may
  use — ladder, h\*, train/legacy points) keeps the inherited assert
  verbatim. `"residue_sweep"` (the sweep component only) bypasses the
  assert and instead REQUIRES a per-point `residue_label ∈ {identity,
  train-residue, novel}` in the results schema; identity and
  train-residue points are EXCLUDED from all generalization claims
  and aggregates but INCLUDED in the reducer-detection signature (the
  disclosed-confound probe below — a mod-reducer is residue-exact at
  those labeled points too, uniformly, with no decay front). Both h\*
  values (61 ≡ 5 mod 8; 57 ≡ 9 mod 12) pass the claim-mode assert and
  are computed on the claim path, never via the sweep.
- **Cost-scaling probe** (wall-clock only, behavioral values recorded
  but out-of-window): h ∈ {2^10+5, 2^14+5, 2^17+5, 2^20+5} (≡ 5 mod
  8 preserved), all arms timed.

**Precision bar.** recovered_frac@0.9 is the bar of record
(continuity with every Task D/E number); recovered_frac@0.99 and
mean_cos are mandatory secondaries (the known @0.9 slack: fr=7 passes
@0.9 through h=7, §3 of `TASK_E_FINDINGS.md`); the per-cell deep probe
(§3.7 m4) reports the restricted-operator eigenvalue phase residual
alongside every behavioral number. No argmax/codebook decoding
anywhere (`CLAUDE.md` exact-continuous-recovery hard rule).

**The pre-registered predictions, justified from measured TASK_E
numbers.** Measured per-mode max phase residuals of converged K=8
operators: s1 0.0020, s2 0.0052, s3 0.0031, s4 0.0117 (§9 table).
Under h-fold LINEAR composition a phase residual δ drifts to h·δ;
per-item cosine at the @0.9 bar tolerates worst-case aggregate drift
arccos(0.9) = 0.451 rad (all-modes-drift regime) up to 1.37 rad
(single-mode regime at K=8) — giving seed-level hold horizons
h ∈ [0.45/δ, 1.37/δ]: s1 [225, 685], s2 [87, 263], s3 [145, 442],
s4 [39, 117]. The heuristic band is planning-only: the EXACT per-seed
decay curve is computable from the dumped restricted operator A via
literal matrix powers with no fitting (§9(e) machinery, predicted-vs-
measured within 0.001-0.02 through h=21) and is LOCKED IN at
calibration, archived BEFORE any far-h behavioral eval runs (Axis C
below).

**K=12 arithmetic (Rev 2, MA3 — published; Rev 1 pinned h\* = 57
with no supporting numbers).** Archived K=12 seeds' max phase
residuals (`TASK_E_FINDINGS.md` §10 table): s1 0.0044, s2 0.0099,
s0 0.0125. The all-modes (conservative) tolerance is K-independent:
arccos(0.9) = 0.451 rad. The single-mode tolerance GROWS with K (one
mode of K carries less cosine weight, cos θ = 1 − 0.1K): K=12 gives
θ = arccos(−0.2) = 1.772 rad (vs 1.369 at K=8). Hold-horizon bands
[0.451/δ, 1.772/δ], same rounding convention as the K=8 bands above:
s1 [103, 403], s2 [46, 179], s0 [36, 142]. **At h\* = 57 exactly 1/3
archived seeds (s1) holds under the conservative bound; 3/3 hold
under the single-mode bound.** Decision (of MA3's two options):
**h\*(K=12) = 57 is KEPT, with pre-registered ASYMMETRIC
confidence** — moving h\* to ≤ 36 (the all-seeds-conservative point)
would land at-or-below ladder point 21, at-or-before the depth where
P2 predicts baselines have even failed (h = 45 at K=12, below),
destroying the separation window. The asymmetric pre-registration,
scored via the LOCKED Axis-C exact curves: a fresh K=12 seed with
locked residual δ ≤ 0.0079 (= 0.451/57) is PREDICTED-HOLD; δ ∈
(0.0079, 0.0311] (= 1.772/57) is STRADDLE; δ > 0.0311 is
PREDICTED-FAIL. **P1-K12 (prediction of record):** every
PREDICTED-HOLD seed holds at 57, no PREDICTED-FAIL seed holds, and
≥ half the STRADDLE seeds hold (the directional claim: truth sits
nearer the single-mode bound, as K=8's measured-exact h=21 behavior
sat far inside its own conservative bound). Archived analogs predict
a seed mix of roughly 1/3 PREDICTED-HOLD, 2/3 STRADDLE — so the
K=12 NCR band at h\* is predicted HOLD-or-DEGRADED, and per §3.2a a
K=12 SEP-PARTIAL alongside a K=8 WIN scores WIN-PARTIAL
(publishable-with-caveat), pre-registered here, not negotiated after
the readout.

Predictions of record, at the separation depth **h\* = 61
(K=8) / 57 (K=12)**:

- **P1 (NCR holds):** ≥3/5 NCR seeds at recovered_frac@0.9 ≥ 0.9 at
  h\* (s1/s2/s3 hold under even the conservative bound; s4 is the
  straddle case), with NCR's OWN failure front located on the ladder
  between h ≈ 87 and h ≈ 442 (median-seed band) — pre-registered as
  depth-robustness fine-structure, not hidden. **Phase-wrap revival
  caveat (Rev 2, mi5), pinned to the front-location report:**
  single-mode drift h·δ crossing π WRAPS, so apparent cosine recovery
  ("revival") at deeper ladder points is expected for higher-residual
  seeds (first-wrap h ≈ π/δ: K=8 s4 ≈ 269, K=12 s0 ≈ 251, K=12 s2 ≈
  317 — i.e. at-or-before ladder points 509/381). The failure front
  is defined as the FIRST ladder crossing below bar; post-front
  revivals are reported (they are Axis-C-predicted fine-structure,
  the locked exact curves capture them) but never re-admitted as
  holds.
- **P2 (O(h) baselines fail):** each comparison-of-record O(h) arm
  falls below median recovered_frac@0.9 = 0.5 by the LAST ladder
  point before h\*: h = 29 (K=8) / h = 45 (K=12) (Rev 2, MA4 — Rev
  1's "by h ≤ 32" named no grid point; 32 is not on any grid). Basis: fr=7
  — a LINEAR map with one dead mode, the most charitable possible
  drift model — fell 0.88 → 0.06 between h=7 and h=21 (§3); K16 s2
  (diffuse δ=0.0334) fell 0.9997 → 0.2617 over the same span (§10);
  composition error-cascading in trained nonlinear chains is
  documented since Guu et al. arXiv:1506.01094; FWM's own authors
  fixed N_r=3 and never pushed recursive reads deeper
  (arXiv:2011.07831); an LN-wrapped nonlinear step map supervised
  only at h ≤ 3 has NO mechanism pinning per-step phase error below
  the ~0.01-0.03 range (horizons 15-45).
- **Disclosed residual confound (self-attack, pinned):** a baseline
  that learned arithmetic mod-K reduction from h ∈ {1,2,3}
  supervision would legitimately solve the task without deep
  composition. We predict it cannot; if it does, it is detectable
  (a mod-reducer is residue-exact at ALL depths uniformly — no decay
  front anywhere — distinguishable from genuine composition, which
  drifts) and the outcome is scored per the §3.2a partition below
  (a reducer win is a baseline win, disclosed via its
  no-decay-front signature), not excused.

**§3.2a Axis-A bands, exhaustive partition, and cross-K rule (Rev 2,
MA4 — Rev 1's WIN/TIE/LOSE left gaps, e.g. NCR ≥ 0.9 with a baseline
in (0.5, 0.9) was unlabeled).** Per-arm numeric band at h\*, on
median recovered_frac@0.9 across seeds: **HOLD ≥ 0.9; DEGRADED ∈
(0.5, 0.9); FAIL ≤ 0.5** (boundaries assigned — exactly 0.9 → HOLD,
exactly 0.5 → FAIL; the three bands tile [0, 1] exhaustively). The
baseline axis is the BEST-performing comparison-of-record arm (max
of LoopedVec/FWM-read medians — most favorable to the baselines;
C_MLP never enters). Per-K outcome over (NCR band, best-baseline
band), all 9 cells:

| NCR \ best baseline | FAIL | DEGRADED | HOLD |
|---|---|---|---|
| **HOLD** | WIN | SEP-PARTIAL | TIE |
| **DEGRADED** | SEP-PARTIAL | TIE | LOSE |
| **FAIL** | TIE | LOSE | LOSE |

Reading: WIN = full separation (NCR intact, baselines destroyed);
SEP-PARTIAL = NCR strictly one band better but not the full
HOLD-vs-FAIL gap; TIE = same band (including both-FAIL "both fail
before h\*", Rev 1's own TIE case); LOSE = baseline strictly better
banded than NCR. A per-K label is awarded only if it holds on every
claim-eligible (non-identity, non-train-residue) residue stratum at
h\*; otherwise the offending arm drops one band before labeling.
Cross-K aggregation (K=8 and K=12 each yield one label; all 10
unordered pairs covered):

| K-pair (unordered) | Overall Axis-A outcome |
|---|---|
| WIN + WIN | **WIN** (the capability headline) |
| WIN + SEP-PARTIAL | **WIN-PARTIAL** — pre-registered publishable-with-caveat |
| WIN + TIE | **WIN-PARTIAL** |
| SEP-PARTIAL + SEP-PARTIAL | **SEP-PARTIAL** (robustness-gap result, no capability headline) |
| SEP-PARTIAL + TIE | **TIE** |
| TIE + TIE | **TIE** |
| any pair containing LOSE | **LOSE** (a WIN + LOSE split additionally triggers diagnose-first: K-dependence of the separation is itself a reportable finding) |

Expected under P1/P1-K12 + P2: K=8 WIN, K=12 WIN-or-SEP-PARTIAL →
overall WIN or WIN-PARTIAL. The mod-reducer confound above scores
through this same table (a reducer-driven baseline HOLD lands in the
rightmost column — a reducer win is still a baseline win, disclosed
via its no-decay-front signature, never excused).

**Pre-registered outcomes — per axis, all publishable:**

| Axis | WIN | TIE | LOSE |
|---|---|---|---|
| **A — depth-robust exactness** | §3.2a cross-K overall = WIN (NCR HOLD + every comparison-of-record arm FAIL, at both K, all claim-eligible residue strata) | §3.2a overall ∈ {WIN-PARTIAL, SEP-PARTIAL, TIE} — graded per §3.2a; WIN-PARTIAL keeps a caveated capability claim, TIE collapses the separation to cost-only | §3.2a overall = LOSE (at either K the best baseline out-bands NCR) |
| **B — sequential cost** (claimable as capability ONLY if Axis A ≥ TIE, else it is the M4 "accounting convention") | Measured wall-clock fits log (NCR) vs linear (O(h) arms) and NCR is ≥10× faster at h=1021 | Scaling separation present but <10× at h=1021 | NCR not measurably sub-linear (would indicate a broken implementation, triggers diagnose-first) |
| **C — spectral predictability (instrument claim)** | \|predicted − measured mean_cos\| ≤ 0.05 at every ladder h ≤ 509 for ≥3/5 NCR seeds, predictions locked at calibration (precedent: ≤ 0.02 at h ≤ 21, worst case 0.067) | Prediction holds through h ≤ 125 only | Predictions fail inside the trusted window |

WIN = the capability separation (PI bar); WIN-PARTIAL = the same
claim with a disclosed single-K caveat (publishable per §3.2a's
pre-registration); SEP-PARTIAL/TIE on A = an honest robustness-gap /
efficiency-exactness paper (workshop-grade, per §2's pre-registered
risk); LOSE on A = a genuinely informative negative (in-context-
written operators do not survive their own spectral drift, or
iterated maps are more robust than their prior art suggests). Axis C
is publishable as instrument methodology under every A outcome.

### 3.3 Baseline set (M3)

All arms receive the identical raw-integer h signal (§3.1); all are
scored by the same continuous cosine readout; trainable-parameter
match tolerance pinned at **±15%** (the Stage-2 attack round's own
suggested house number, `CAPABILITY_STAGE2_ATTACK_R1.md` "also
noted"), computed exactly at build time.

| Arm | What it is | Sequential cost | Status |
|---|---|---|---|
| **NCR (contender)** | Task-E harness: encoder writes Z in context; read = scale-managed bin-exp `Z^h q` | O(log h) | Claimed configuration |
| **NCR naive-loop** | SAME trained checkpoints as NCR, read = literal h-fold matvec loop WITH per-step Frobenius renormalization of the running iterate, log-scale tracked (Rev 2, MA5 — the same positive-scalar argument as §3.1 applies verbatim: a positive scalar per step is invisible to cosine; REQUIRED because at the measured worst c\* = 2.843 an unrenormalized fp32 loop overflows at h ≈ 85, 2.843¹²⁵ ≈ 5.3×10⁵⁶ ≫ 3.4×10³⁸ — INSIDE the old h ≤ 125 window; the not-adopted alternative was a window cap ≤ 61) | O(h) | Disclosed cost-control arm + fp32 cross-check, agreement bar PINNED (Rev 2, MA5): \|cos_binexp − cos_loop\| ≤ 5×10⁻⁴ per (cell, h ≤ 125), justified from fp32 unit roundoff u = 2⁻²⁴ ≈ 6.0×10⁻⁸ — ≤125 renormalized d=16 matvecs accumulate ≲ 125·16u ≈ 1.2×10⁻⁴ relative error, the bin-exp side ≤14 matmuls ≈ 1.3×10⁻⁵, so the bar carries ≈3.5× headroom; a breach flags NUMERIC-DIVERGENT and is arbitrated by the fp64 shadow; eval-only, zero training cost |
| **LoopedVec** | Param-matched iterated VECTOR map: trained step function applied h times to a vector state, linear decode, same episode input, same h signal (Looped Transformers arXiv:2409.15647; depth-recurrent arXiv:2603.21676 — their existence is exactly why "current architectures cannot do variable-depth single-pass composition" was retired as a claim, §2 M3). Step-map family PINNED (Rev 2, mi6): weight-tied pre-LN residual two-layer GELU MLP, `x_{t+1} = x_t + W₂·GELU(W₁·LN(x_t) + b₁) + b₂` on the d=16 state, hidden width fixed at build time to land total trainable params inside the ±15% match to the NCR arm; no attention, no gating, no per-step weights; this ONE family is the comparison of record — no post-hoc family swaps | O(h) | Comparison of record #1 |
| **FWM-read** | FWM-style recursive fast-weight read (Schlag, Munkhdalai & Schmidhuber, ICLR 2021, arXiv:2011.07831, cited AND distinguished: FWM fixes N_r=3, reads are LN-nonlinear, composition approximate): the SAME written Z, read via h-fold recursive one-hop LN reads — isolates "exact linear power vs recursive nonlinear reads" on an identical state | O(h) | Comparison of record #2; deviating read ⇒ read-vector-std diagnostic in its calibration gate (§3.1) |
| **C_MLP one-hot(h)** | Task E's inherited control | O(1) | DISCLOSED WEAK CONTROL ONLY — architecturally unable to extrapolate by its own §5 admission (one-hot(h) is out-of-vocabulary at held-out h); never the comparison of record; evaluated and labeled as such |

### 3.4 Corrected stopping/trust rule (M2; rebuilt Rev 2 per §4 MA1)

**The rule (Rev 2: worst-case σ-form — supersedes Rev 1's ρ-form,
which §4 MA1 showed is NOT worst-case: ρ(D) does not bound ‖D^m‖₂
for non-normal D, and our own dead seeds measure cond(D) up to
1.09×10¹⁰, so the non-normal regime is MEASURED, not hypothetical;
Rev 1's injection term also grew the E-trajectory at σ_min(A)
instead of σ_max(A), understating injected leakage).** Norm
convention, pinned: every norm in this rule is the OPERATOR 2-norm
(largest singular value, `numpy.linalg.svd`); Frobenius appears
nowhere in the rule itself — where §9-table values (which are
Frobenius) are substituted below, they are valid conservative
stand-ins (‖·‖₂ ≤ ‖·‖_F) and each such use is disclosed. With the §9
block decomposition Z → (A, B, C, D) in the [E, E⊥] basis: leakage
injected at application j is at most ‖C‖₂·σ_max(A)^(j−1) (the
E-trajectory grows at worst like σ_max(A)); the remaining h−j
applications amplify it by at most ‖D^(h−j)‖₂ ≤ σ_max(D)^(h−j).
Relative to worst-case signal scale σ_min(A)^h, with

  a := σ_max(A)/σ_min(A) ≥ 1,   r := σ_max(D)/σ_min(A),

  T(h) = (‖C‖₂/σ_min(A)) · Σ_{j=1..h} a^(j−1)·r^(h−j)
       = (‖C‖₂/σ_min(A)) · (a^h − r^h)/(a − r)   [a ≠ r]
       = (‖C‖₂/σ_min(A)) · h·a^(h−1)             [a = r]

with the a→1, r→1 limit **T(h) = h·‖C‖₂/σ_min(A)** — the Rev-1
linear rule is exactly this doubly-degenerate limit, stated
explicitly because the measured healthy regime sits ADJACENT to the
r = 1 singularity (s1's ρ-based r = 1.004). Evaluation pin: if
\|a − r\| < 10⁻⁶ use the limit branch h·max(a,r)^(h−1); the simple
envelope T(h) ≤ h·m^(h−1)·‖C‖₂/σ_min(A) with m := max(a,r) bounds
both branches and is the implementation's sanity cross-check.
**Disclosed neglect, explicit (MA1):** the B-feedback path
(E⊥ → E re-entry, then re-injection) is second-order in ‖B‖₂‖C‖₂ and
is NEGLECTED by T(h); for the healthy §9 seeds ‖B‖‖C‖/σ_min(A)² ≤
0.006 per round trip, but the omission is one-sided and is disclosed
wherever T is reported. **Trust threshold τ = 0.2 unchanged**
(worst-case cosine floor 1/√(1+τ²) ≈ 0.98, above the 0.9 bar).
**Stated plainly, as before: this rule bounds the numerical/leakage
trust of computing Z^h; it does not and cannot select h** (§3.1).

**Restated example values (Rev 2 — recomputed honestly; Rev 1's
numbers superseded).** From §9's table for the healthiest seed s1
(c\* = 1.375, scale-corrected residual 0.0074, ‖C‖_F = 0.024,
‖D‖_F = 3.89, ρ(D) = 1.38, cond(D) = 1.010), with disclosed
Frobenius→operator conversions (σ(A) = c\* ± ‖A − c\*Π‖₂, bounded by
the Frobenius residual: 1.375 ± 0.029; flat-spectrum bound
σ_max(D) ≤ ‖D‖_F/√(1 + 7/cond²) = 1.387): a ≤ 1.043, r ≤ 1.031,
‖C‖₂/σ_min(A) ≤ 0.0178, giving **T(61) ≈ 9.6 at the K=8 separation
depth and a trust horizon of h ≤ 8** (T(8) = 0.18, T(9) = 0.21).
Under the tightest defensible reading (zero σ(A) spread and
σ_max(D) = ρ(D), i.e. a = 1, r = 1.004): T(61) ≈ 1.20, horizon
h ≤ 11; the r→1 limit form gives T(61) = 61·0.0175 ≈ 1.07. **Every
reading refuses every decisive far-h point — said plainly: under
the corrected rule the a-priori screen REFUSES h\* = 61 and the
entire deep ladder for every seed, including the healthiest.** That
is the honest price of a true worst-case bound, and the rule's role
of record narrows accordingly: it exists to REJECT
adversarial/degenerate constructions (the negative-test cases below;
dead seeds with cond(D) ~ 10¹⁰) and to rank seeds a-priori — never
to admit far-h behavioral claims. **Disclosure of record (mi3):**
every decisive far-h behavioral point (h\* and beyond, both K) is
therefore RULE-UNTRUSTED by construction; leakage attribution at
those depths rides ENTIRELY on the Axis-C calibration-locked exact
per-seed decay curves (§3.2), while the fp64 shadow (§3.1) certifies
ROUNDING only, not leakage. Reporting labels per (cell, h)
unchanged: RULE-TRUSTED (T ≤ τ) / SHADOW-VERIFIED (fp64 agreement) /
UNTRUSTED — flagged, never silently dropped.

**Conservatism arithmetic (mi1 — Rev 1's "≥30×" RETRACTED; shown,
not asserted).** At h = 21, the deepest measured-exact depth, the
angle-domain ratio of the rule's worst-case bound to measured
aggregate drift, arctan(T_lin(21)) / (21·δ_seed), is: s1 8.5×
(arctan(0.374) = 0.358 rad vs 21·0.0020 = 0.042 rad), s2 6.9×,
s3 10.0×, s4 3.7× — the honest claim is **≈4-10× conservative** on
healthy seeds (the §4 attacker's independent 6-9× recomputation sits
inside this range), not ≥30×.

**Mandatory EXECUTED negative test — TWO cases (pre-launch gate item
§3.8(c); `CLAUDE.md` negative-tests-run-to-completion hard rule).
Construction pinned (MA1/mi2): numpy `default_rng(20260709)` for
every random draw; d = 16, K = 8; E = span{e₀..e₇}; A = 1.0·Π (exact
8-cycle, σ_min = σ_max = 1); B = 0; query q = e₀; probe depth
h = 21; all rule norms operator 2-norm (Frobenius reported alongside
for the §9-table cross-walk); junk/signal := ‖E⊥-component of
Z²¹q‖₂ / ‖E-component of Z²¹q‖₂.**

- **N1 (amplifying near-normal D — carried from Rev 1, re-pinned):**
  C = Gaussian draw rescaled to ‖C‖₂ = 0.01 exactly; D = 1.5·Q with
  Q = QR-orthogonalization of a Gaussian draw (ρ(D) = σ_max(D) =
  1.5). The OLD linear rule scores T_lin(21) = 0.21 and MUST admit
  (any calibration loose enough to admit healthy s1's own
  T_lin(21) = 0.37, measured-exact behavior, admits this garbage
  too); the corrected rule scores T(21) = 0.01·(1.5²¹ − 1)/0.5 ≈
  99.7 → REJECT at τ = 0.2. Pass criteria: T_lin(21) ∈ [0.20, 0.22];
  corrected T(21) > 10; measured junk/signal > 1 (Rev-1's computed
  instance ≈ 66×, cosine ≈ 0.015, recovery 0.0).
- **N2 (non-normal D, admit-direction — NEW per §4 MA1's executed
  counterexample):** D = 100·E₀₁ (the matrix unit e₀e₁ᵀ on the E⊥
  block — nilpotent: D² = 0, ρ(D) = 0, σ_max(D) = 100); C =
  0.01·e₁^⊥e₃ᵀ, fully DETERMINISTIC (injects only from E-basis
  vector e₃, which the cycle trajectory occupies exactly at step 19,
  so the single surviving junk term is D¹·C·Π¹⁹e₀ and junk/signal =
  100·0.01 = 1.0 exactly — no RNG in this case at all, so the
  archived output cannot spuriously mismatch). The Rev-1 ρ-BASED
  rule scores r = ρ(D)/σ_min(A) = 0 → T(21) = 0.01 ≤ τ → **falsely
  ADMITS**; the corrected σ-based rule scores r = 100 → T(21) ≈
  1.0×10³⁸ → REJECT. Pass criteria: ρ-based T(21) ≤ 0.011; corrected
  T(21) > 10³⁰; measured junk/signal ∈ [0.9, 1.1].

Both cases, both halves (rule evaluations AND empirical arms), must
execute to completion (numpy, CPU, no early exit; pass criteria are
the pinned inequalities above, not exact-float matches), output
archived as `matrix-thinking/chapter2/test_trust_rule_negative`
results, before launch — not merely be written.

### 3.5 Novelty (M1, rebuilt)

**The claim is the CONJUNCTION ONLY — no component is claimed novel
in isolation:** (1) operators WRITTEN IN CONTEXT (episode-specific
fast weights), (2) EXACT linear composition at read time, (3)
variable, input-controlled depth including depths far beyond train
support, (4) provable-rank/causal verification instrumentation
(restricted-operator spectral analysis, blank-out, the Task D/E
lineage), (5) O(log h) sequential read cost via repeated squaring.

Related-work skeleton (each line = the axis it occupies + the delta
that keeps the conjunction open):

- **FWM, arXiv:2011.07831** — closest prior art: in-context
  fast-weight multi-hop relational composition at read time (2020).
  Delta: hop count fixed (N_r=3), reads LN-nonlinear, composition
  approximate, no exactness/rank verification, O(h) sequential.
- **TensorLog arXiv:1605.06523 / Neural-LP arXiv:1702.08367 / DRUM
  arXiv:1911.00055** — query-conditioned variable-length products of
  relation matrices, but over STATIC KG operators learned in weights,
  not written in context; the operator-bank nearest neighbors, which
  is precisely why that sub-experiment stays M4-gated.
- **Guu et al., arXiv:1506.01094** — compositional path queries in
  embedding space; documented composition error-cascading in 2015 —
  the motivating failure our exactness axis addresses.
- **MesaNet, arXiv:2506.05233** — query-conditioned matrix-FUNCTION
  reads ((G+λI)⁻¹q via CG) at scale: matrix-function reads exist;
  not powers, not variable composition depth.
- **MAGNA, arXiv:2009.14332; MEA/HLA, arXiv:2510.27258** — the soft
  power-mixture op is occupied (fixed geometric mixture / fixed-order
  powers); also the F1 kill for soft selection.
- **arXiv:2602.03655** — log-depth analysis of matrix powers
  IN-WEIGHTS; the in-context-written version is our open ground (the
  M4-surviving delta, cited as such).
- **DO-NOT-BUILD perimeter (unchanged from §1):** Grazzi
  arXiv:2411.12537 / DeltaProduct arXiv:2502.10297 / RWKV-7 / TPR /
  RotatE arXiv:1902.10197; fast-weight PKM arXiv:2601.00671 remains
  a different axis (retrieval, not composition).

**Non-redundancy with Stage 2 (M4), stated as one sentence per
side:** Stage 2's recurrent composer (`CAPABILITY_SEPARATION_DESIGN.md`
§2.2.2) adjudicates WRITE-time recurrence — state built token-by-token,
depth = context length, O(D) sequential writes, eigenvalue-range
expressivity per Grazzi; NCR adjudicates READ-time composition of an
already-written operator — depth DECOUPLED from context length at
O(log h) sequential cost — sharing instruments and harness lessons
but zero claims.

### 3.6 Ledger (M5) — phased, calibration-first, capped

Rate anchor: **≈2.4 GPU-h per 80K-step Task-E run, MEASURED**
(`TASK_E_FINDINGS.md` §10: 6 runs ≈ 14.5 GPU-h). Every training cell
is priced at the 80K anchor (K=8 typically converges by 40K ≈ 1.2
GPU-h; priced conservatively at 2.4). Read-time arms (NCR bin-exp,
naive-loop) share the NCR arm's checkpoints — eval-only, ≈0 GPU-h;
Task E's archived K=8/K=12 frN checkpoints/Z-dumps are a disclosed
REUSE upside for the NCR arm (Stage 2's §2.7 "reused, not
double-charged" convention) but the ledger is priced at the
conservative fresh-retrain case so launch does not depend on cluster
artifact retrieval.

| Phase | Cells | GPU-h (priced) | Gate |
|---|---|---|---|
| **0 — calibration (mandatory first, `CLAUDE.md`)** | 3 cells: one per trained arm (NCR, LoopedVec, FWM-read) at K=8, run to completion | 7.2 | Duties: real per-cell rate (supersedes the anchor); per-arm convergence profile; blank-out/P=1 execution (§3.7 m3); read-vector-std pass/fail for deviating arms; Z-dump + LOCKED per-seed predicted decay curves (Axis C); fp64-shadow wiring check |
| **1 — wave-1 core (K=8)** | 3 trained arms × 5 seeds = 15 (Phase-0 cells counted inside) + C_MLP × 3 seeds | 43.2 | Sub-cap **50**; fires only after Phase 0 passes and §3.8 gates are open |
| **2 — K=12 replication** | Same 18-cell structure at K=12 | 43.2 | Sub-cap **50**; gated on Phase-1 readout (not just completion) |
| **Reserve** | Budget-artifact retests (any dead cell re-run at 2-2.5× steps per the three-budget-artifacts program finding, §10) + contingency | 20 | Draws logged individually |
| **TOTAL CAP** | | **120 GPU-h** | Per-cell abort at 1.5× the Phase-0 calibrated rate (budget_guard-style circuit breaker) |

Eval-only far-h grids (bin-exp, naive-loop, fp64 shadows, cost probe
to h = 2^20+5) are <1 GPU-h combined, disclosed, inside the cap.
**Said plainly, per the disposition: the honest full-program figure
(raw ≈ 86.4 + reserve → 120 cap) lands well above the §1 sketch's
35-50 GPU-h — that sketch bought one arm × one K × 5 seeds with no
calibration cell, no iterated-map arm, and no retest headroom (§2
M5). The phased structure keeps wave 1 alone lean: Phases 0+1 ≤ 50
GPU-h, and Phase 2 does not fire on autopilot.** The
**operator-bank sub-experiment is SEPARATELY ledgered** (sketch
30-60 GPU-h, NOT authorized by this cap, own design round required)
and double-gated behind (i) wave-1 readout and (ii) Stage 2's
calibration readout (M4) — no GPU before both land.

**Disclosed gaps (Rev 2).** (mi7) Phase 0's three calibration cells
are all K=8 — there is NO K=12 calibration cell. Mitigation of
record: Phase 2 fires only on Phase-1 READOUT (already pinned
above), which carries the calibrated rate/convergence/diagnostic
lessons; additionally the first K=12 cell per trained arm is
designated a calibration-duty cell — its rate is checked against the
1.5× breaker and its Axis-C per-seed exact decay curves are computed
and LOCKED from its Z-dump BEFORE any far-h behavioral eval at K=12
runs (this is also where MA3's asymmetric-confidence classification,
§3.2, gets its locked residuals). (mi8) The phase rows sum to 106.4
GPU-h under the Phase-0-inside-Phase-1 convention; the 13.6 GPU-h
gap to the 120 cap is deliberate, undesignated headroom for
rate overruns caught by the per-cell breaker — any draw on it is
logged individually, like the reserve.

### 3.7 Minor findings folded (m1-m4) — finding → resolution map

- **m1 (gradient norms h·ρ^(h-1); eigh precedent)** → ADOPTED, no
  re-adjudication: backprop only through h ≤ 3 at train time; deep h
  eval-only; no eigendecomposition in the training path (§3.1).
- **m2 (bin-exp forward-only; incompatible with soft selection)** →
  both moot by construction: no deep-power backprop exists, and soft
  selection is dead (§3.1); noted for the record that bin-exp fixes
  neither backward norm growth nor selection.
- **m3 (blank-out/P=1 not inherited)** → gradient-based blank-out
  specified for THIS forward pass, executed at Phase-0 calibration
  per trained arm: corrupt raw episode inputs post-write; decode from
  `Z^h q` must be bit-identical AND ∂o/∂(raw inputs post-write) must
  be exactly zero — not a shape check (`CLAUDE.md` hard rule).
- **m4 (h-signal hygiene; stratification; deep probe)** → identical
  raw-integer h to all arms (§3.1); effective-hop `(h, h mod K)`
  stratification in ALL reporting with identity and train-residue
  sweep points labeled and excluded from claim aggregates (§3.2,
  Rev-2 MA2 pathway); deep probe MANDATORY per cell:
  `--save-z` + restricted-operator analysis (`analyze_zdump.py`
  lineage) with the spectral phase residual reported alongside every
  behavioral recovery number, plus the @0.99 secondary bar for the
  known @0.9 slack (§3.2).

### 3.8 Launch gate and carried backlog

Wave 1 (Phase 1) fires only when ALL of: **(a)** Stage 2's
calibration readout has landed (`CAPABILITY_SEPARATION_DESIGN.md`
§2.8 item 2 — its fla-vs-torch adjudication and readout-diagnostic
lessons transfer here first; per M4, no NCR GPU before those lessons
exist); **(b)** this Rev 1 survives its own fresh-eyes micro-attack
round (recorded in this file as §4 before any build dispatch, per the
gauntlet-bookkeeping hard rule); **(c)** the §3.4 corrected-rule
negative test — BOTH cases, N1 (amplifying near-normal) + N2
(non-normal nilpotent, Rev 2) — has EXECUTED to completion with
archived output — plus
the standing chain: independent build audit, real-path smoke test,
and the Phase-0 calibration gate itself. Design-only note: no code
exists yet for any §3 item; nothing in this revision has touched the
box.

**Carried forward unchanged as unattacked backlog (one line each):**
SECOND TRACK — rank-budgeted writes (per-context rank allocation at
the write step; novelty gap verified vs arXiv:2602.04852 /
2602.02195 / Elastic Spectral SSM; ≈25-35 GPU-h sketch; not yet
attacked). PIGGYBACK — orthogonal-complement novelty detector on
archived Z-dumps (near-zero GPU; not yet attacked).

**STATUS (2026-07-09): Rev 1 COMPLETE** *(historical snapshot —
superseded by §4's NEEDS-REVISION verdict and the §3.9 Rev-2
changelog below)*. All §2 findings addressed:
F1 (input-supplied h pinned, both selection modes documented dead),
F2 (headline re-scoped to depth-robust exactness with the reachable-
set cap argument, effective-hop stratification pinned), M1 (novelty =
conjunction only, related-work skeleton rebuilt around FWM), M2
(geometric scale-normalized trust rule + executed negative test
specified as a launch gate), M3 (LoopedVec + FWM-read comparisons of
record, ±15% param tolerance, C_MLP demoted to disclosed weak
control), M4 (single relation, O(log h) headline, operator bank
separately ledgered and double-gated), M5 (120 GPU-h phased cap from
the measured 2.4 GPU-h anchor, wave 1 ≤ 50), M6 (direct matvec read
pinned, read-vector-std diagnostic wired into deviating arms'
calibration gates), m1-m4 (folded, §3.7). Micro-attack round on this
Rev 1 is PENDING (§3.8(b)); launch remains gated behind Stage 2's
calibration readout (§3.8(a)) and the executed negative test
(§3.8(c)). No GPU spent; no code written.

### 3.9 Rev 2 (2026-07-09) — changelog (finding → fix)

Design-only revision addressing the §4 micro-attack verdict (5
MAJOR, 8 minor); every fix edited in place in the subsection where
the defect lived. Map:

- **MA1 → §3.4 rebuilt as a true worst-case bound:** r :=
  σ_max(D)/σ_min(A) (ρ(D) is not a bound for non-normal D — a
  MEASURED regime, dead-seed cond(D) up to 1.09×10¹⁰), injected
  growth σ_max(A)^(j−1), B-feedback neglect (second-order ‖B‖‖C‖)
  disclosed explicitly, the a→1/r→1 limit form h·‖C‖₂/σ_min(A)
  stated explicitly (the measured healthy regime sits ADJACENT to
  the r = 1 singularity, ρ-based r = 1.004) with a pinned
  \|a − r\| < 10⁻⁶ evaluation branch. Negative test extended to TWO
  pinned cases: N1 carried (amplifying near-normal, seed-pinned
  RNG), N2 NEW (D = 100·E₀₁ nilpotent admit-direction — the old
  ρ-based rule falsely admits at T = 0.01, the σ-based rule rejects
  at ≈10³⁸; deterministic construction, junk/signal = 1.0 exactly;
  inequality pass criteria; `default_rng(20260709)`; operator-2-norm
  convention pinned). Example values recomputed honestly: the
  corrected rule REFUSES every decisive far-h point (s1 at h\* = 61:
  T ≈ 9.6 worst-case table read, ≈ 1.20 tightest, ≈ 1.07 r→1 limit;
  trust horizon h ≤ 8-11) — said plainly in §3.4, role narrowed to
  a-priori screen, decisive-depth attribution leaning on the
  disclosed shadow+Axis-C split per mi3.
- **MA2 → §3.2 h-grid:** eval-grid pathway pinned —
  `TaskEConfig.eval_grid_mode = "claim"` (default) keeps the
  inherited `__post_init__` periodicity assert (`task_e.py:121-132`)
  on every claim-feeding path; `"residue_sweep"` is label-and-exclude
  for the sweep component only (identity + train-residue points
  labeled, excluded from generalization claims/aggregates, INCLUDED
  in the reducer-detection signature). Design-level spec; the build
  implements.
- **MA3 → §3.2 K=12 arithmetic published:** bands s1 [103, 403] /
  s2 [46, 179] / s0 [36, 142] from residuals 0.0044/0.0099/0.0125
  (tolerances arccos(0.9) = 0.451, arccos(−0.2) = 1.772);
  **h\*(K=12) = 57 KEPT with pre-registered asymmetric confidence**
  (PREDICTED-HOLD δ ≤ 0.0079 / STRADDLE ≤ 0.0311 / PREDICTED-FAIL,
  scored via locked Axis-C curves; archived analogs: 1/3 hold
  conservatively, 3/3 single-mode; P1-K12 added); cross-K statements
  recomputed via §3.2a.
- **MA4 → §3.2a added:** exhaustive HOLD (≥0.9) / DEGRADED
  ((0.5,0.9)) / FAIL (≤0.5) bands, all 9 (NCR × best-baseline) cells
  labeled WIN/SEP-PARTIAL/TIE/LOSE, 10-pair cross-K table (WIN at
  both K = WIN; WIN + SEP-PARTIAL-or-TIE = WIN-PARTIAL,
  pre-registered publishable-with-caveat; any LOSE = LOSE, with
  diagnose-first on a WIN+LOSE split); P2's "by h ≤ 32" fixed to the
  actual grid points 29 (K=8) / 45 (K=12); Axis-A row rewritten over
  the §3.2a overall label.
- **MA5 → §3.1 + §3.3:** naive-loop arm gets per-step Frobenius
  renormalization (same positive-scalar cosine-invariance argument
  as §3.1, restated there; the not-adopted alternative was a ≤61
  window cap — overflow arithmetic shown: 2.843¹²⁵ ≈ 5.3×10⁵⁶,
  unrenormalized onset h ≈ 85); "fp32 tolerance" pinned at
  \|Δcos\| ≤ 5×10⁻⁴ justified from u = 2⁻²⁴ (≤125·16u ≈ 1.2×10⁻⁴
  accumulation, ≈3.5× headroom); §3.1's scale row now pins
  renormalization of the squared BASE after every squaring AND the
  running product after every multiply.
- **mi1 → §3.4:** "≥30×" conservatism RETRACTED; shown arithmetic
  gives ≈4-10× (s1 8.5×, s2 6.9×, s3 10.0×, s4 3.7×; the attacker's
  6-9× sits inside). **mi2 → §3.4:** negative-test construction
  pinned (seed, operator-2-norm convention with disclosed
  Frobenius stand-ins, inequality pass criteria; N2 deterministic).
  **mi3 → §3.4:** disclosure of record added — all decisive far-h
  points are RULE-UNTRUSTED; leakage attribution rides on Axis-C
  calibration-locked curves; the fp64 shadow certifies rounding
  only. **mi4 → §3.1:** read-vector-std bar harmonized to Stage 2's
  derived 0.04 with the same derivation citation
  (`CAPABILITY_SEPARATION_DESIGN.md` §2.8 item 2(e)); Rev 1's 0.05
  retired as unjustified. **mi5 → §3.2:** phase-wrap revival caveat
  pinned to the front-location report (first-wrap h ≈ π/δ: K=8 s4 ≈
  269, K=12 s0 ≈ 251, s2 ≈ 317 — at-or-before ladder points
  509/381); front = FIRST crossing below bar; revivals reported,
  never re-admitted. **mi6 → §3.3:** LoopedVec step-map family
  pinned (weight-tied pre-LN residual 2-layer GELU MLP, width set
  for the ±15% param match; no post-hoc swaps). **mi7 → §3.6:**
  no-K=12-calibration-cell gap disclosed with the Phase-1-readout +
  first-K=12-cell-calibration-duty mitigation. **mi8 → §3.6:** the
  13.6 GPU-h cap headroom (106.4 raw vs the 120 cap) disclosed and
  purposed.

**STATUS (2026-07-09): Rev 2 COMPLETE** — all §4 findings (MA1-MA5,
mi1-mi8) addressed in place. Next: ONE more micro-attack pass, scoped
specifically to the §3.4 MA1 rule math (cheap, scoped), before
DESIGN-CLEARED-FOR-BUILD-QUEUE; the launch double-gate is unchanged
(§3.8). No GPU spent; no code written.

## §4 MICRO-ATTACK ON REV 1 (2026-07-09 overnight): NEEDS-REVISION — 5 MAJOR, 0 FATAL; Rev 1 confirmed to resolve every §2 finding

Byte-level citation audit: ALL §3 citations verify against TASK_E_FINDINGS
(phase residuals, fr7 collapse, K16 s2, ρ(D), c*, the 2.4 GPU-h anchor);
all arithmetic re-derived exact (mod-K residues, arccos bounds, seed bands,
T-rule values, the fp32-overflow-at-h≈83 claim, τ floor); the suspected
h*=61-vs-hold-front-87 inconsistency does NOT land (87-442 is a FAILURE
front; 61 < every healthy K=8 seed's conservative onset). The
renormalization-invariance claim CONFIRMED numerically (cos=1.0 at
h=21/61/125 incl. non-power-of-2 partial products). Ledger closes at 120
(Phase-0-inside-Phase-1 convention).

**MAJOR findings (binding on Rev 2, all design-text-level, no GPU
implications):**
- **MA1 (highest-value fix):** §3.4's rule is NOT worst-case: ρ(D) doesn't
  bound ‖D^m‖ for non-normal D, and injected growth needs σ_max(A) not
  σ_min(A). Executed counterexample: D=100·E₀₁ (nilpotent, ρ=0, ‖D‖₂=100),
  ‖C‖=0.01 → RULE-TRUSTED at every h while actual junk/signal at h=21 is
  0.29 > τ. Our own dead seeds measure cond(D) up to 1.09e10 — a MEASURED
  regime. Fix: r := σ_max(D)/σ_min(A), σ_max(A)^{j−1} growth, disclose
  B-feedback neglect, add a non-normal-D admit-direction case to the
  §3.8(c) executed negative test, special-case r≈1.
- **MA2 (new-in-Rev-1):** the pinned residue sweep (h=64/60 identity;
  57-59/49-51 ≡ train residues) ASSERT-CRASHES against task_e.py:121-132's
  inherited eval-hop guard. Specify a label-and-exclude eval-grid pathway
  for the sweep component; keep the assert for claim-feeding paths.
- **MA3:** h*=57 (K=12) has no supporting arithmetic; §10's K=12 residuals
  give conservative holds 103/46/36 — only 1/3 archived seeds holds at 57;
  K=12 tolerance is arccos(−0.2)=1.77 never computed. Publish the K=12
  arithmetic; move h* or pre-register asymmetric confidence.
- **MA4:** Axis-A outcome partition non-exhaustive (NCR ≥0.9 with baseline
  in (0.5,0.9) is neither WIN/TIE/LOSE; cross-K aggregation unstated; P2's
  "by h≤32" has no grid point at 32). Partition exhaustively with numeric
  hold/fail bands + a cross-K rule.
- **MA5:** the naive-loop fp32 cross-check overflows INSIDE its own h≤125
  window at measured c*=2.843 (2.843^125≈5e56); loop arm needs per-step
  renorm or window cap ≤61; "fp32 tolerance" unpinned; §3.1 must pin
  renormalization of the squared BASE, not just the running product.
- 8 MINOR (mi1-mi8): incl. harmonize the read-vector-std bar (0.05 here vs
  Stage 2's derived 0.04, same instrument); pin the negative test's
  construction (seed/norm convention); disclose that decisive far-h points
  are all RULE-UNTRUSTED and ride on shadow+Axis-C; phase-wrap revivals
  near ladder point 509; pin the LoopedVec step-map family; K=12
  calibration gap disclosure; 13.6 GPU-h headroom disclosure.

**DISPOSITION: Rev 2 dispatched (design-only). Gates §3.8 verified as
hard ALL-of conditions and unchanged. Security: one system-channel
date-change+concealment sighting (the recurring ambiguous vector),
reported.**

## §5 SCOPED MICRO-ATTACK ON THE §3.4 RULE MATH (2026-07-09 overnight, on 8972a07): CLEARED → DESIGN-CLEARED-FOR-BUILD-QUEUE

Zero math-level defects. Verified by independent re-derivation +
numerical simulation: (1) with B=0 the mixed sum is EXACT (block
lower-triangular Z^h), the telescoping identity checks to 1.2e-16, both
worst-case path endpoints carried with the right prefactor, and the
B-feedback neglect holds with margin (recomputed ≤0.0037 per healthy
seed vs the disclosed 0.006 ceiling). (2) The bound is UNCONDITIONALLY
VALID for arbitrary non-normal A and D — stronger than the doc claims:
the signal side needs a LOWER bound and σ_min(A)^h ≤ σ_min(A^h) holds
for every A; non-normal A only adds reject-safe looseness. Scale
invariance under Z→cZ verified symbolically. (3) N2 verified by hand AND
by exact simulation: old rule T=0.0100 (false admit against a genuine
behavioral failure — junk/signal exactly 1.0, cosine 0.707), corrected
rule 1.010×10³⁸; the 38-order looseness is entirely in the conceded
false-refuse direction; the refuse-everything-deep concession is
arithmetically FORCED even at the absolute-floor constants reading
(T(61) ≥ 0.231 > τ). (4) Every restated value reproduces (worst 9.6195,
tightest 1.1896, r→1 1.0647, horizons h≤8/h≤11, conservatism
8.5/6.9/10.0/3.7×, N1 0.2100/99.74). (5) The 1e-6 evaluation-branch pin
is sound (cancellation onset at |a−r|~u, not 1/h; limit branch
overestimates reject-safe by ~6e-5 at the pin; overflow → inf → reject,
fail-safe). (6) Both negative tests executable, criteria numeric, no
degenerate divisions.

**Three non-binding nits, BINDING ON THE BUILD AGENT as text-level
guards (do not code from the flawed prose):** (n1) N2's "occupies e₃
exactly at step 19" — the trajectory also occupies e₃ after steps 3 and
11; uniqueness of the surviving term comes from D²=0 annihilating the
j=4/j=12 injections, NOT unique occupancy — do not assert "C fires only
once." (n2) N1's old-rule ADMIT must be scored against the s1-calibrated
threshold (>0.37) as an explicit inequality, since 0.21 > τ=0.2
literally rejects. (n3) mi1's "4-10× conservative" quantifies T_lin, not
the σ-form (whose own s1@h=21 conservatism is ~15.6×) — do not quote it
against the Rev-2 rule.

**STATUS: NCR wave 1 = DESIGN-CLEARED-FOR-BUILD-QUEUE.** Launch remains
double-gated per §3.8 (Stage-2 calibration readout + build audit + smoke
+ Phase-0 calibration + the executed negative tests N1/N2). Second track
(rank-budgeted writes) and the Z-dump orthogonal-complement piggyback
remain unattacked backlog.
