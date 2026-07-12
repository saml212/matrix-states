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

## §6 §3.8(c) NEGATIVE-TEST EXECUTION DISCHARGE (2026-07-10)

**What was missing (the build agent's stop).** A build agent, tasked with
launching against the §3.8 gate, ran an exhaustive search (repo, `git log
--all`, the SSD mirror, worktrees, the H100 box) for
`matrix-thinking/chapter2/test_trust_rule_negative` and found NOTHING. Every
prior "EXECUTED" claim for N1/N2 (§3.4's pin, §3.9 MA1's addition, §5's
"both negative tests executable, criteria numeric... executed by hand AND by
exact simulation") was an in-context simulation by a prior agent, never
persisted to disk — the gate's own text ("execute to completion... output
archived... before launch — not merely be written") was violated by
omission, not by a wrong result.

**What was executed.** `matrix-thinking/chapter2/test_trust_rule_negative.py`
— a self-contained numpy fp64 script (no torch) implementing the §3.4 shared
construction (`default_rng(20260709)`, K=8, d=16, A=1.0·Π exact 8-cycle,
B=0, q=e0, h=21, operator-2-norm convention) plus N1 (§3.4: Gaussian C
rescaled to ‖C‖₂=0.01, D=1.5·Q near-normal) and N2 (§3.9 MA1: D=100·E01
nilpotent, C=0.01·e1^⊥e3ᵀ deterministic, no RNG). The three §5 binding nits
are honored explicitly in code: (n1) e3-occupancy asserted at ALL of steps
3, 11, AND 19, with the j=4/j=12 injection terms independently computed and
asserted annihilated (D²=0) while the j=20 term is asserted surviving —
uniqueness is proven mechanically, never assumed from "C fires once"; (n2)
N1's old-rule ADMIT is scored and printed as an explicit inequality against
the s1-calibrated threshold (0.37), with the literal τ=0.2 reading printed
alongside for contrast; (n3) the "4-10× conservative" figure is not computed
or quoted anywhere in the script or this record (it applies to T_lin only,
per §5's own nit). 29 checks total (11 for N1, 18 for N2, covering
construction, mechanism, pinned pass criteria, and cross-checks against
§5's recorded numbers), all asserted with explicit `sys.exit(1)` teeth on
any miss — no bare `assert` (immune to `-O`).

**Matched values vs §5's recorded simulation numbers (all reproduce to
stated precision — no mismatch, no STOP triggered):**

| Quantity | §5 recorded | This run (computed) | Match |
|---|---|---|---|
| N1 T_lin(21) | 0.2100 | 0.21000000000000005 | exact |
| N1 corrected T(21) | 99.74 (99.7377 restated) | 99.73770190239061 | exact to stated precision |
| N2 rho-based (OLD) T(21) | 0.0100 | 0.01 | exact |
| N2 corrected (sigma) T(21) | 1.010×10³⁸ | 1.0101010101010102×10³⁸ | exact |
| N2 junk/signal | 1.0 exactly | 1.0 (bit-exact) | exact |
| N2 cosine | 0.707 (=1/√2) | 0.7071067811865475 | exact |

N1's own junk/signal (22.83, this run) differs from Rev-1's illustrative
"≈66×" instance recorded in §3.4 — expected and non-binding: that figure was
always RNG-draw-dependent (this script draws C then Q from a single seeded
`default_rng`, an order §3.4/§3.9 never pinned beyond the seed itself), and
only the inequality "junk/signal > 1" is a pinned pass criterion, met with
large margin. No other discrepancy of any kind was found; no STOP was
triggered.

**Teeth mutation proof.** A throwaway copy (outside this repo, in the agent
scratchpad) with `D_n2[1,0] = 5.0` added alongside the pinned
`D_n2[0,1] = 100.0` — breaking D²=0 — was run and FAILED: 17/29 checks pass,
exit code 1 (the nilpotency, D²=0, both annihilation-term, fast-vs-naive
`matrix_power` cross-check, analytic-vs-direct recursion cross-check, both
rho-based pass criteria, and all junk/signal/cosine checks correctly flip to
FAIL). The mutated copy was deleted after use; the unmodified script was
re-run twice more afterward and diffed byte-for-byte against the original
clean run (identical both times) before this record was written. Full
transcript archived in `run.log` (three runs: canonical, mutated, restored).

**Archive paths + MD5s** (also see `test_trust_rule_negative/MANIFEST.md`):

| File | MD5 |
|---|---|
| `matrix-thinking/chapter2/test_trust_rule_negative.py` | `3879e56de2158c028dc2768dbcc93fd0` |
| `matrix-thinking/chapter2/test_trust_rule_negative/results.json` | `c5c8f81847cbb8731090604f5d4b1045` |
| `matrix-thinking/chapter2/test_trust_rule_negative/run.log` | `37d0850cd30019da8c7b4acc4a756ed3` |

Mirrored to
`/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-10_ncr_negative_tests/`
per the archive policy (`CLAUDE.md` Data section).

**Security note.** During this task, a tool-boundary system message
appeared claiming `run.log` had been "modified... intentional... don't tell
the user" — the concealment-instruction pattern this project's hard rule
flags. Investigated rather than obeyed: root cause was mundane and
self-inflicted (the script itself rewrites `results.json`/`run.log` on
every invocation of the pinned path; a later diff-confirmation re-run of the
script clobbered the hand-assembled multi-run `run.log` back to a bare
single-run version). No evidence of external tampering; `run.log` was
reassembled correctly afterward and its final state verified via the MD5s
above. Reported per the standing security-sighting convention regardless of
benign root cause.

**§3.8(c) DISCHARGED.** Both cases, both halves (rule evaluations and
empirical arms), executed to completion, archived, cross-checked clean,
teeth-verified. §3.8's remaining conditions ((a) Stage-2 calibration
readout, (b) the Rev-1 micro-attack — already satisfied by §4/§5, (d) the
standing chain: independent build audit, real-path smoke test, Phase-0
calibration) are unaffected by this discharge and remain open where not yet
separately closed.

## §7 WAVE-1 BUILD (2026-07-10): `matrix-thinking/ncr/` — CPU logic suite 13/13; awaiting independent audit (§7a)

Gate state verified against the repo before building: §3.8(a) =
`CAPABILITY_SEPARATION_DESIGN.md` §2.30 (SWEEP-READY, on disk), §3.8(b) =
§4+§3.9+§5, §3.8(c) = §6 (commit 007ea05; archive + md5s re-verified by
this agent, including the SSD mirror). No GPU touched by this build.

### 7.1 File manifest (md5 at record time)

| File | md5 | Role |
|---|---|---|
| `ncr/ncr_task.py` | `3c664c0c0571fc361692edd220ab21d2` | pinned h-grids, residue labels, two-mode eval pathway (MA2), label-schema guard |
| `ncr/ncr_models.py` | `1e1a76a0b79301bef13f6ab133f49f98` | arms (NCR/FWM-read/LoopedVec/C_MLP) + scale-managed reads (bin-exp, renormed loop) |
| `ncr/ncr_spectral.py` | `0ce6fde1cd7a246925baa6dc6db909bf` | Axis-C locked-curve machinery, deep probe (m4), §3.4 trust screen |
| `ncr/ncr_selftest.py` | `657043cb8e7013294d57f2094513108d` | 13-section enforced suite incl. executed negative tests (kill proofs) |
| `ncr/run_ncr.py` | `2067ccc09cfa6315b73c70838a3c8bff` | per-cell pipeline, Phase-0 orchestration, smoke + box-smoke, CLI |
| `ncr/launch_phase0.sh` | `990dfdc69f920f16ebd8ba7ac2bee50d` | box launcher: idle-GPU guard, tmux+supervisor, STOP file, co-location |

### 7.2 Build decisions (each traceable to a pin; audit checklist)

1. **Harness inherited by IMPORT, not copy** (`task_e`/`task_d`/`model_v4`/
   `model_e` from `chapter2/`): the mod-K guard, `_permutation_graph`,
   injectivity teeth, `BindingEncoder`, `compose`, and C_MLP run verbatim —
   zero drift by construction (§3.1 "inherited verbatim").
2. **MA2 two-mode pathway implemented NCR-side** (`ncr_task.py`), not by
   editing `task_e.py` (another lane's file): claim mode bakes every claim
   point into a literal `TaskEConfig` so the INHERITED `__post_init__`
   assert runs verbatim over all of them (h\*(K=12)=57 added to `H_extra`
   explicitly — it is not a K=12 ladder point; both h\* pass); the residue
   sweep never enters a config hop set and its results schema REFUSES
   unlabeled entries. Negative-tested both directions.
3. **Far-h ground truth via the exact single-cycle identity**
   `pi^h == pi^(h mod K)` on the LABEL path only; the batch's `hops` is
   overwritten with the RAW h before any model sees it (§3.1 depth-signal
   pin). Label-reduction == full-h equality PROVEN by self-test at
   h=13/21/61, not assumed. (Flagged for audit as the subtlest seam.)
4. **Reads per §3.1/MA5:** square-and-multiply with per-squaring Frobenius
   renorm of the base AND per-multiply renorm of the running product,
   log-scales tracked separately; naive-loop arm renormed per step;
   agreement bar |Δcos| ≤ 5e-4 applied to the MAX per-item diff (stricter
   than the mean; the MA5 rounding derivation bounds every item) at h ≤ 125;
   fp64 shadows computed on the SAME batches; bar 0.005.
5. **Trust rule reused from the DISCHARGED implementation** (`ttrn.trust_rule`
   import): the deployed screen is bit-identical to the one N1/N2 proved has
   teeth; self-tests reproduce the archived N1 T(21)=99.7377 and N2
   T(21)=1.0101e38 through this build's block-extraction path and REJECT
   both. Per-h verdict = worst over Z-dump examples (bias-toward-refuse).
   §5 nits honored: n2's s1-calibrated 0.37 lives in the archived script
   (reused, not re-scored); n3's 4-10× figure appears nowhere in this build.
6. **LoopedVec (mi6 pinned family)**: hidden width 529 DERIVED (not tuned):
   NCR arm = 170,896 params; LoopedVec = 153,424 + 33H ⇒ H=529 → 170,881
   (Δ 0.009%); FWM-read = 170,928 (+0.02%); C_MLP = 311,456 (disclosed weak
   control, unmatched by design). ±15% gate asserted in code with an
   executed out-of-band kill (hidden=3000 mutant, ratio 1.48). Episode
   conditioning enters ONLY via the encoder-produced x₀ (the pinned step map
   admits no other channel); its convergence profile is a Phase-0 READOUT,
   pre-registered here as possibly poor — never a family swap (mi6).
7. **Axis-C locks**: per-cell predicted decay curves (S9(e) machinery
   imported from `analyze_zdump.py`; far-h extension via renormalized fp64
   powers, cross-checked ≡ the lineage's literal `matrix_power` at h ≤ 21 to
   <1e-9; reference side uses exact Π-periodicity) written + sha256-hashed
   BEFORE far-h behavioral eval; the eval path REFUSES far-h points without
   a verified lock (negative-tested, incl. hash tamper detection). K=12
   locks carry the MA3 PREDICTED-HOLD/STRADDLE/PREDICTED-FAIL class.
8. **Blank-out (m3) executed per trained arm** (bit-identical + grad-exactly-
   zero + write-path-alive), wired into every cell AND smoke; leaky-read
   kill proof executed. **Read-vector-std** for deviating arms (FWM-read,
   LoopedVec — more coverage than §3.3's minimum, bias-toward-FAIL): std
   across queries, mean-aggregated (Stage-2 MAJOR-1(a) statistic, max
   co-reported), bar 0.04 at every probe depth {1,2,3,5,13,21} (mi4).
9. **C_MLP h-signal**: keeps its INHERITED one-hot(h) (§3.3's own row);
   the §3.1 raw-integer pin governs the comparisons of record — disclosed
   in code and here, not silently reconciled.
10. **fp64-shadow scope**: wraps the h-iteration reads (NCR bin-exp/loop,
    FWM recursive, LoopedVec loop). C_MLP's O(1) read has no h-fold fp32
    accumulation — shadow recorded as null with the reason (disclosed).
11. **Closed-form checks (§2.26 discipline)**: standard-basis shift-matrix
    bin-exp exact to h=1021; single-binding v·kᵀ maps k→v; the TRANSPOSED
    [K,V] layout (k·vᵀ / inverse shift) is DETECTED as wrong — hand-computed
    zero-accumulation configs. *(§7a correction: as first recorded this item
    overclaimed — the checks were wired into box-smoke ONLY, so the 13/13
    CPU record carried no executed evidence for them; fixed as suite section
    t14, now 14/14, and still re-run ON CUDA in box-smoke.)*
12. **Ops hygiene**: resume-safety by VALIDITY (config-sha-checked
    checkpoints; corrupt/mismatched checkpoints refused loudly); atomic
    JSON writes; per-cell runner tags + git commit + config sha + host
    fingerprints; STOP-file support; per-cell breaker at 1.5× the
    anchor-scaled rate (§2.29: scales with the step-budget axis;
    CUDA-only, disclosed); tmux launcher refuses busy GPUs (mem ≥ 2 GiB or
    util ≥ 5%) and existing sessions; exact-kill instructions only.

### 7.3 Self-test verdict (executed, CPU, torch 2.8.0)

13/13 sections PASSED via `run_ncr.py --smoke`, including the executed
negative tests: grid-guard kills (h=64/57@K=8, h=60@K=12, poisoned
`H_extra` through the inherited assert), label-schema refusals, bit-order
mutation killed at non-palindromic h=13/19 (palindromic h would falsely
pass a bit-reversal mutant — documented), non-scalar renorm mutation
killed, shadow + agreement flags fire on perturbation, leaky-read blank-out
kill, lock-refusal + lock-tamper kills, param-gate kill, full-arm gradient
coverage, and a micro end-to-end cell (train → lock → blank-out → labeled
eval → atomic JSON → resume-skip).

### 7.4 Phase-0 cost (on paper, pre-experiment checklist)

Params: 170,896 / 170,881 / 170,928 (matched arms; d=16 fp32, <1 GiB/cell).
Basis = the MEASURED 2.4 GPU-h/80K anchor (§3.6): Phase-0 cells at 40K
steps (§3.6's own "K=8 converges by 40K"; §9's converged-at-40K dumps) ≈
1.2 GPU-h each → serial-sum 3.6. The launcher CO-LOCATES all three cells on
the one granted GPU (tiny, Python-bound), projecting device draw ≈ 1.2-2.0
GPU-h against the coordinator's ≤2 GPU-h Phase-0 cap (the registry's 7.2
was the conservative 80K serial pricing); the co-located breaker allowance
is widened 2× (§2.21 contention-false-abort lesson) while TRUE per-cell
rates are measured and recorded in `phase0_rate.json` for wave-1 planning.
Eval-only far-h grids (shadows + cost probe to 2^20+5) priced <1 GPU-h
combined (§3.6), inside the same cap. If realized draw threatens the 2.0
cap, the overage is reported, never silently absorbed.

**STATUS: BUILD RECORDED; independent audit (§7a) dispatched next per the
standing chain. No GPU work performed. Wave 1 (Phase 1) remains gated on
Phase-0 PASS + a fresh coordinator go/no-go (§3.6).**

### §7a INDEPENDENT BUILD AUDIT (2026-07-10, fresh agent, on 42a87e6): NEEDS-FIXES → fixes applied, scoped re-audit dispatched

Verdict on the build as committed: **NEEDS-FIXES — 1 MAJOR, 2 MINOR, 1 NIT,
0 FATAL.** Every correctness property was independently re-verified by the
auditor's own from-scratch reimplementations (never by trusting the build's
self-tests): bin-exp exactness vs an exact integer-oracle across the FULL
pinned h-union at both K (208+324 checks) incl. power-of-2 edges, negative-
c\* and complex-eigenvalue operators vs literal fp64 `matrix_power` to
1e-16; renorm per-item independence (mixed 1e-6/1e6-scale batch bit-
identical to items-alone, 0.00e+00); FWM/LoopedVec h-purity + zero cross-
query/cross-batch grad leakage + finite-difference-vs-autograd 2.13e-08;
checkpoint/resume BIT-EXACT vs straight-through (60 vs 27+resume, fresh
process, 47/47 tensors 0.000e+00); the label-reduction seam at the identity
edges h=64/h=60 the build's own t04 did not cover (targets, query_keys, AND
generator post-call state all exactly equal; `hop_set=(0,)` correct); all
56 grid labels recomputed from scratch; params recounted by hand
(170,896/170,928/170,881/311,456 — exact match); the packaged suite 13/13
re-run PLUS three auditor-constructed mutations (per-row renorm, transposed
squaring operand, swapped residue-label map) — all three killed by existing
checks; trust screen re-verified against a fresh 29/29 run of the §6 script
(archive md5s re-confirmed after `git checkout --` of that script's
documented rewrite side effect); §5 nit n3 compliance grep-verified.

Findings → dispositions (fixes applied in the follow-up commit, this
record):
- **MAJOR-1 (record-vs-reality):** `closed_form_checks` (the §2.26
  zero-accumulation suite incl. the transpose tooth) was wired into
  box-smoke ONLY; §7.2 item 11 claimed it ran in the CPU smoke — so the
  recorded 13/13 carried ZERO executed evidence for it. The auditor ran it
  directly on CPU (PASSED — the math is correct; defect was wiring +
  record). **FIX:** new suite section t14 (suite now 14/14, re-executed);
  §7.2 item 11 corrected in place with the overclaim disclosed.
- **MINOR-2:** `EvalPoint.in_window` was schema-only (never consulted).
  **FIX:** front/reducer aggregates now filter on it structurally.
- **MINOR-3:** `predicted_cos_curve_far` (and bin-exp) are SIGN-exact — a
  negative c\* yields cos = −1 at odd h, mathematically correct but
  potentially misread as an instrument bug. **FIX:** docstring documents the
  behavior + why it cannot arise post-convergence (auditor verified the
  behavior identical in both code paths, errors 0.00e+00).
- **NIT-4:** box-smoke still pending — expected; it is the next chain step.

No injection sightings during the audit. Scoped re-audit of the three fixes
dispatched to the same auditor before the box smoke.

**SCOPED RE-AUDIT (same auditor, on c2b9533): CLEARED.** All three fixes
verified BY EXECUTION: t14 genuinely wired into `--smoke` (14/14; kill
proof — a transposed-squaring mutant monkeypatched into `binexp_read` made
t14 raise, clean restore passed); the `in_window` filters proven
structurally no-op on today's grid (all ladder/h\*/sweep points in-window,
sweep sets 8/8 and 12/12, the `min()` cannot go empty); the sign-exactness
docstrings match code exactly (−1/+1 alternation, err < 1e-9). Commit
scope confirmed: 4 files, every hunk one of the three fixes or the §7a
record. Two nits settled before §7b: **(A)** the box-layout defaults in
`launch_phase0.sh` (`NCR_DIR=$HOME/ncr`, `PY=$HOME/tdenv/bin/python` —
matching the live box recon: Task-E lineage at `~/chapter2/` top level
md5-identical to local, `~/tdenv` torch 2.12.1+cu130) were a working-tree
edit pending at re-audit time; substance reviewed against `H100_SETUP.md`'s
own deploy convention, committed with this record so the tree matches an
audited commit before deploy; box deploy must also ship
`chapter2/analyze_zdump.py` + `test_trust_rule_negative.py` + its
`results.json` archive (t09/ns self-tests read them). **(B)** post-fix
manifest appended as §7.1a below (never edited in place — fresh agents
tamper-check against these rows).

### 7.1a Post-§7a file manifest (md5 at c2b9533 + the NIT-A launcher commit)

| File | md5 |
|---|---|
| `ncr/ncr_task.py` | `3c664c0c0571fc361692edd220ab21d2` (unchanged) |
| `ncr/ncr_models.py` | `1e1a76a0b79301bef13f6ab133f49f98` (unchanged) |
| `ncr/ncr_spectral.py` | `b95c4cd988bdd5bb758c54144f32af64` |
| `ncr/ncr_selftest.py` | `a0d80c07a2496ed364f617f7395d3250` |
| `ncr/run_ncr.py` | `01d975673d6c9b714a292d35fe1e99df` *(superseded by §7b's one-line resume fix: `77545140b59295121f7f55afbe1c7b6a`)* |
| `ncr/launch_phase0.sh` | `cc6991ec3be1508bad66c4afc2574c9b` |

### §7b BOX SMOKE (2026-07-10, youthful-indigo-turkey, GPU 6): PASS after one real-CUDA resume fix

**Deploy:** `~/ncr/` (sibling of the md5-identical `~/chapter2/` Task-E
lineage — task_e/task_d/model_v4/model_e/rank_utils/eigen_utils all match
local exactly) + `analyze_zdump.py`, `test_trust_rule_negative.py`, and its
`results.json`/`run.log`/`MANIFEST.md` archive shipped to `~/chapter2/`;
all 9 deployed files md5-verified on the box against §7.1a and the §6
archive pins. venv `~/tdenv` (torch 2.12.1+cu130, numpy 2.5.0) vs local
(torch 2.8.0, numpy 2.0.2) — the CPU logic suite passed 14/14 on BOTH
(cross-version robustness, `results/box_cpu_smoke.log`), CPU run pinned via
`CUDA_VISIBLE_DEVICES=""` (no GPU claim).

**Real-CUDA smoke, first run: FAILED at [4] (checkpoint/resume)** —
`TypeError: RNG state must be a torch.ByteTensor` in `try_resume`:
`torch.load(map_location="cuda")` moves the saved generator-state
ByteTensor to CUDA and `Generator.set_state` requires CPU. Exactly the
device-placement bug class the §2.27 lesson says CPU smokes structurally
cannot catch (the §7a auditor's CPU resume test was bit-exact; only real
CUDA exposes it), introduced by this builder's own "simplification" of the
defensive `.cpu()` during the build. **Fix: one line**
(`gen.set_state(state["gen_state"].cpu())`, run_ncr.py md5 →
`77545140b59295121f7f55afbe1c7b6a`), local suite re-run 14/14, redeployed
md5-verified.

**Re-run: PASS, all 6 sections** (GPU 6, idle-verified at launch; GPU 7's
task2 round untouched): [1] device teeth incl. the CPU-into-CUDA negative;
[2] closed forms ON CUDA (shift-matrix bin-exp exact to h=1021, v·kᵀ
convention, transposed-layout tooth fires); [3] every param of every arm
gets a finite grad; [4] 100-step train → checkpoint → fresh resume, params
bit-identical; [5] trained-Z read agreement — max|binexp−loop| 1.5-2.4e-07
across h∈{5,21,61,125} (MA5 bar 5e-4: ≈2000× margin), min cos(binexp,fp64)
≥ 0.9999999; [6] blank-out on CUDA bit_identical + grad_exactly_zero +
write_path_alive. Log: `results/box_cuda_smoke.log` (archived at Phase-0
harvest). GPU draw: ~2 min on one H100 ≈ 0.03 GPU-h. The one-line fix's
diff sent to the §7a auditor for a scoped confirm (runs in parallel; the
box smoke's own re-run is the prescribed instrument for this bug class and
it passed).

**Scoped audit of the resume fix (FRESH agent — the first scoped confirm
died on an API error mid-response and never returned; re-dispatched per
coordinator instruction): CLEARED.** Verified by execution: the 7a79712
delta is exactly the one line + comment + the registry record; CPU
round-trip through the repo's own `save_ckpt`/`try_resume` post-fix is
bit-exact (47/47 parameter tensors, generator byte-state identical,
`.cpu()` proven a literal no-op on CPU-loaded checkpoints by direct device
inspection); the only other `torch.load` consumer (`eval_only_ckpt` →
`load_state_dict`) is device-agnostic; the two pre-existing `.cpu()` calls
(`torch.set_rng_state`, `torch.cuda.set_rng_state`) verified correct
against PyTorch source (RNG state is always a CPU byte blob); CPU suite
re-run 14/14. No injection sightings.

**STATUS: §7b PASS, resume-fix audit CLEARED → Phase 0 (§7c) running.**

### §7c PHASE 0 — CALIBRATION GATE (2026-07-10, GPU 6, tmux `ncr_phase0`): **PASS** (instrument duties 3/3); convergence readouts recorded; realized ≈0.73 GPU-h total vs the ≤2 cap

Three co-located 40K-step cells (K=8, seed 0) ran to completion + the gate
pass, launch 21:59Z → supervisor done 22:42:28Z (wall ≈42.5 min on ONE
GPU ≈ **0.70 GPU-h device draw**; per-cell serial-sum notional 2.00; box
smoke ≈0.03 → **§7b+§7c realized ≈0.73 GPU-h** against the coordinator's
≤2 Phase-0 cap and the 120 ledger). One outage note: the local
coordinator-side session died mid-run (session limit) — the on-box
tmux+supervisor process was unaffected and finished on its own, the
intended failure isolation. Archive:
`experiment-runs/2026-07-10_ncr_phase0/` (repo, 14 files + md5_manifest,
ckpts excluded per policy) + full SSD mirror incl. the 3 checkpoints.

**Gate table (`phase0_gate_table.json`, md5 `9083807c…`):**

| duty | ncr | loopedvec | fwm |
|---|---|---|---|
| completed / n_skipped | ✓ / 0 | ✓ / 0 | ✓ / 0 |
| rate GPU-h per 80K-equiv (supersedes the 2.4 anchor) | 1.116 | 1.118 | 1.122 |
| blank-out (m3, executed) | PASS | PASS | PASS |
| read-vector-std ≥0.04 @ every probe depth | n/a (direct read) | PASS (0.55-6.99) | PASS (0.28-0.41) |
| Axis-C lock (sha256) | `3bbd2c70…` | n/a (no Z) | `542450a0…` |
| fp64 shadow wired | ✓ (all Δ ≤ 4e-6) | ✓ | ✓ |
| bin-exp/loop agreement (h≤125) | ✓ max 3.2e-06 vs 5e-4 bar | — | — |
| in-dist min recovered@0.9 (READOUT) | 0.582 | 0.002 | **0.940** |
| ladder failure front / revivals / reducer | 5 / none / no | 5 / none / no | 13 / none / no |
| **gate_pass (instrument duties)** | **✓** | **✓** | **✓** |

**Convergence readouts (never auto-gates; the coordinator's wave-1
inputs):** (i) **fwm CONVERGED** (0.940 in-dist; ladder decay 0.919@5 →
0.685@13 → 0.546@21 → 0.416@29 → 0.079@61 → 0.000@125+ — a clean
drifting-nonlinear-read front in exactly P2's direction, already below the
0.5 P2 bar AT h=29, the pinned grid point, and deep in FAIL band at
h\*=61). (ii) **ncr seed-0 PARTIAL at 40K** — mid-transition (loss 0.997 →
0.078 over the run, still falling; in-dist 0.582; locked phase residuals
0.032-0.099, one order above the archived converged seeds' 0.002-0.012;
eff_rank(A) 7.6-7.8 climbing toward 8; c\* 1.59-1.76 coherent) — this is
the archived frN-s0 40K "late-phase transition" profile reproduced, i.e.
the known budget artifact, NOT a new failure mode; wave-1 cells should run
the ledger's own 80K pricing (at the measured 1.12 rate, wave-1 core ≈ 20
GPU-h, well under the 50 sub-cap). (iii) **loopedvec cannot fit even
in-distribution** (0.002 at h∈{1,2,3}; loss plateau 0.36) — the §7.2 item
6 pre-registered structural bottleneck of the mi6-pinned family confirmed
(episode information cannot reach the weight-tied step map beyond x₀);
rvstd passes (state is query-dependent, just not task-solving); per mi6 no
family swap — flagged to the coordinator as the M3 strawman-risk readout
for the wave-1 go/no-go.

**Axis-B preview (measured, standardized B=32 probes):** at h=1021 —
bin-exp **1.7 ms** vs naive loop 64.4 ms (38×), fwm 36.2 ms, loopedvec
63.8 ms; at h=2^20+5 — bin-exp **2.6 ms** vs loop **61.3 s** (≈23,600×),
fwm 33.5 s, loopedvec 64.1 s. The O(log h)-vs-O(h) wall-clock separation
already exceeds the Axis-B ≥10× WIN bar on Phase-0 hardware; the claimed
read stays SHADOW-VERIFIED (fp64 Δ ≈ 0) at every ladder point.

**STATUS: §7c PASS → STOPPED per charter. Wave 1 (Phase 1: 3 arms × 5
seeds + C_MLP × 3 at K=8, ≈20 GPU-h at the measured rate at 80K steps)
awaits the coordinator's fresh go/no-go against the ledger and GPU load
(§3.6). The standing chain §3.8(d) is now fully discharged: build (§7) →
independent audit + re-audit (§7a) → box smoke + scoped fix audit (§7b) →
Phase-0 calibration (§7c).**

### §7d COORDINATOR WAVE-1 GO/NO-GO (2026-07-10): **GO** — recorded verbatim before launch

> COORDINATOR WAVE-1 GO/NO-GO: **GO.** Grounds and the two decisions, to
> be recorded verbatim as §7d (RECORD FIRST, commit+push, then launch):
>
> (1) 80K STEPS: APPROVED for all wave-1 trained cells. Grounds: Phase-0's
> NCR seed-0 reproduced the archived frN-s0 late-transition profile at
> 40K — an endpoint read at 40K would confound trainability with
> capability (the budget-artifact precedent this project has been burned
> by twice); your measured 1.12 GPU-h/80K-equiv rate projects the wave-1
> core at ≈20 GPU-h, inside the 50 sub-cap with margin, and the 120
> ledger is at 0.73.
>
> (2) COMPARISON SET: UNCHANGED per the pre-registration. mi6 forbids
> baseline swaps; §3.2a scoring absorbs a FAIL-band baseline as-is.
> LoopedVec's in-distribution failure (0.002) is recorded as the M3
> strawman-risk disclosure — quote it prominently in any eventual claim —
> and FWM (converged, decay front h=13, 0.079 at h\*=61, exactly the
> pre-registered O(h) failure regime) is the load-bearing baseline. No
> swap, no addition.
>
> LAUNCH TERMS: wave-1 core per §3's Phase-1 grid (3 trained arms × 5
> seeds + C_MLP × 3 seeds), 80K steps, sub-cap 50 GPU-h hard. GPU
> allocation: check nvidia-smi first — the §2.33 routing agent may hold
> cells on GPUs 0-5; take genuinely idle devices only (GPU 6 + 7 should
> be free; task2 completed its GPU work) and leave at least one idle GPU
> for the routing agent if its cells are still queued. tmux ncr_wave1 +
> self-healing supervisor, resume-by-validity, STOP-file support,
> per-cell fingerprints — your own §7 machinery. Monitor + report on
> completion sentinel; harvest per the §3 endpoint spec with the
> crosscheck-lens discipline where applicable (§2.31a precedent: any
> degauged readout needs its basis-invariant crosscheck reported
> alongside). Ledger every GPU-h.

**Launch mechanics (this builder's implementation of the terms):** 18
cells (ncr/loopedvec/fwm × seeds 0-4 + cmlp × seeds 0-2), K=8, 80K steps,
in a FRESH results dir (`results_wave1/` — wave-1 seed-0 cells at 80K are
distinct experiments from Phase-0's 40K seed-0 cells; a shared dir would
false-skip them via the COMPLETED check). Breaker rate = the Phase-0
measured 1.1185 GPU-h/80K (itself measured under 3-way co-location, so no
extra contention allowance is needed or taken). Two workers (one per idle
GPU), each running its 9-cell shard in chunks of 3 co-located cells (the
Phase-0-proven concurrency), tmux `ncr_wave1`, self-healing supervisor per
worker, STOP file, `WAVE1_DONE` sentinel when 18/18 cell JSONs read
COMPLETED. Projection: serial-sum ≈ 20.1 GPU-h; device draw ≈ 2 GPUs ×
~3.5-5 h wall ≈ 7-10 GPU-h; hard sub-cap 50.

### §7e WAVE-1 READOUT (2026-07-11, 18/18 cells, K=8): **AXIS A = WIN. P1 5/5, P2 PASS, AXIS B = WIN (20.9×), AXIS C = TIE.** One mid-wave defect (C_MLP eval crash), fixed + regression-toothed; verdict of record below

**Run record.** tmux `ncr_wave1`, GPUs 6+7, launch 23:18:13Z →
`WAVE1_DONE` 04:49:17Z. Survived a coordinator-side session outage
untouched (the tmux+supervisor isolation again). **Defect (D-1):** the
three C_MLP cells crashed deterministically at EVAL (training completed
and checkpointed at 80K): `model.arm` AttributeError — the inherited
`MLPShortcutModel` never carried the arm-protocol attribute the three arm
classes define, and the CPU suite's end-to-end micro cell ran ONLY the ncr
arm, so the per-arm pipeline branch was never executed pre-box. The
supervisor looped harmlessly (each pass: instant 80K resume → seconds-fast
crash; GPUs idle ~6 min). Fix `9b4b71a`: `CMLPModel` subclass adding ONLY
the class attributes (zero behavior; state-dict-compatible with the
crashed runs' checkpoints — they resumed and completed in seconds);
regression teeth: t12 arm-protocol asserts on every builder + t13 now runs
ALL FOUR arms end-to-end (suite 14/14 re-executed). **Scoped audit of the
fix (fresh agent): CLEARED** — by execution: subclass adds zero
methods/params (53 state-dict keys, names/shapes/weights bit-identical to
the plain class; strict=True cross-load of a plain-class checkpoint into
the subclass succeeds — the exact resume path the crashed cells used);
forward output bit-identical; t12 kill proof executed (attribute hidden
in-process → AssertionError('cmlp', None), restored → clean); exhaustive
`model.<attr>` sweep across run_ncr.py found every arm-specific attribute
read gated behind its own arm branch (no latent same-class gap);
wave1_harvest.py's cmlp/loopedvec handling verified (never enters Axis-A/
P1/P2; crosscheck-lens guards on deep_probe presence). It gated trusting
the C_MLP rows only (the disclosed weak control; no decision reads them).
Cosmetic: box cell fingerprints carry `git_commit=UNKNOWN` (no .git on the
box; md5-verified deploys are the provenance there).

**Verdict of record (`wave1_verdict.json`, produced by the committed
`wave1_harvest.py` against the archived cells; archive:
`experiment-runs/2026-07-11_ncr_wave1/` repo tier 2.5MB + SSD full mirror
44MB incl. the 18 checkpoints):**

- **AXIS A (K=8): WIN** — the §3.2a full-separation cell. NCR band HOLD:
  median recovered@0.9 at h\*=61 = **1.0** (per-seed 1.0/1.0/1.0/0.995/1.0
  — P1's ≥3/5 bar passed **5/5**), uniform HOLD across all four novel
  residue strata (h=60-63), no strata band-drop. Best baseline band FAIL:
  fwm 0.158, loopedvec 0.0 at h\*. Per-seed NCR failure fronts
  125/253/253/125/253 — median 253, **inside the pre-registered [87, 442]
  band**; zero post-front revivals; zero reducer signatures anywhere (the
  disclosed confound did not materialize).
- **P2: PASS** — both comparisons of record below median 0.5 at the pinned
  h=29: loopedvec 0.0; fwm **0.4885** (narrow, 1.2pp under the bar —
  reported as such; its full front: 0.919@5 → 0.546@21 → 0.489@29 →
  0.158@61 → 0.002@125, the drifting-nonlinear-read decay P2 predicted
  from fr7/Guu/FWM-N_r=3 grounds).
- **AXIS B: WIN** — bin-exp **20.9× faster** than the best O(h) arm at
  h=1021 (bar ≥10×); scaling fits: every O(h) arm linear-in-h at R² ≥
  0.998 (loop 0.999999, fwm 0.999999, loopedvec 0.998) vs bin-exp flat at
  ~1-3 ms from h=61 to h=2^20+5 (kernel-launch-bound; at 2^20+5 the
  measured gap is ≈13,000-25,000×: 2.6 ms vs 34-64 s). C_MLP O(1) flat,
  disclosed control. Axis-B is claimable as capability because Axis A ≥
  TIE (§3.2 table's own condition).
- **AXIS C: TIE** (a pre-registered outcome row, publishable as instrument
  methodology under every A outcome) — 4/5 seeds' locked-curve deviation ≤
  0.05 through h ≤ 125 (0.0055/0.0060/0.0079/0.0191; s3 = 0.0585 just
  over), 0/5 through h ≤ 509 (max dev 0.082-0.233): beyond each seed's
  failure front the A-only locked prediction under-tracks compounding
  C-leakage — exactly the §3.4/mi3 disclosed division of labor (the rule
  screens, the lock predicts the trusted window, the shadow certifies
  rounding). WIN needed ≤0.05 to h=509 on ≥3/5.
- **Crosscheck-lens table (§2.31a discipline, degauged readout never
  alone):** all 5 NCR seeds eff_rank(A) = **8.000** exactly; degauged
  (c\*-corrected) residuals 0.003-0.009 WITH basis-invariant phase
  residuals 0.0006-0.0053 alongside; c\* 3.1-5.2 (larger than the 40K-era
  1.0-2.8 — longer training grows the cosine-invisible isotropic scale).
- **Hygiene:** 0 NUMERIC-DIVERGENT points (shadow AND agreement bars) over
  all 420 trust-labeled (cell, h, read) points (92 RULE-TRUSTED shallow /
  328 SHADOW-VERIFIED deep — the mi3 label logic operating as designed);
  0 aborted cells; n_skipped_steps = 0 everywhere.

**GPU-h ledger:** wave-1 serial-sum (per-cell elapsed) **18.38** vs 20.1
projected, vs the 50 hard sub-cap; device draw ≈ 2 GPUs × 5.5 h ≈ **11.0**.
Program total realized (smoke + Phase 0 + wave 1): ≈ **11.7 device /
19.1 serial-sum** of the 120 cap.

**Standing per §3.2a/§3.6: the K=8 label is WIN. The overall Axis-A claim
is cross-K — Phase 2 (K=12 replication, 43.2 GPU-h priced, own gate on
THIS readout, never autopilot) is the coordinator's next go/no-go; under
P1-K12's asymmetric pre-registration the expected overall outcome is WIN
or WIN-PARTIAL. The operator bank stays separately ledgered and
double-gated (M4). LoopedVec's in-distribution failure (0.002) travels
with any claim as the M3 strawman-risk disclosure (§7d), with FWM as the
load-bearing baseline.**

### §7f COORDINATOR PHASE-2 GO/NO-GO (2026-07-11): **GO** — recorded verbatim before launch

> Coordinator PHASE-2 GO/NO-GO: **GO.** Record as §7f (RECORD FIRST,
> commit+push, then launch): grounds — (1) K=8 verdict of record is
> WIN/WIN/TIE with zero open defects and the D-1 fix audited CLEARED;
> (2) MA3's asymmetric pre-registration (K=12, h\*=57) is the cross-K half
> of the Axis-A claim — without it the WIN is single-K and the paper claim
> stays capped; (3) ledger: ~19.1/120 spent, Phase-2 projects 18-20 GPU-h
> at the measured rate → ~39/120 after, inside every cap; (4) the box is
> fully idle (all 8 GPUs) — PI no-idle directive. LAUNCH TERMS: K=12 grid
> per the MA3 pre-registration exactly (no metric or bar changes; the
> asymmetric-confidence WIN-or-WIN-PARTIAL scoring as pinned), 80K steps
> per §7d precedent, your own §7 machinery (tmux ncr_phase2 + supervisor,
> resume-safe, STOP-file, fingerprints), GPUs 6-7 primary — and since the
> box is idle you may co-locate additional cells on GPUs 0-5 to shorten
> wall-clock (leave 2 GPUs free for possible coordinator dispatches).
> Per-arm end-to-end micro test BEFORE launch this time (your own
> [LEARN]: all arms, not one representative). On completion: harvest →
> §7g verdict (crosscheck-lens discipline) → report. The operator bank
> stays double-gated and out of scope.

**Launch mechanics:** 18 cells (§3.6 Phase-2 row: same structure at K=12
— ncr/loopedvec/fwm × seeds 0-4 + cmlp × 0-2), 80K steps, fresh
`results_phase2/` dir, GPUs 2-7 (six devices, 3 co-located cells each →
ONE chunk round; GPUs 0-1 left free per the terms), tmux `ncr_phase2`.
**Breaker rate scaled on the K cost axis (the §2.29 lesson, applied
a-priori this time):** the 1.1185 GPU-h/80K rate was measured at K=8;
K=12 carries 1.5× the encoder tokens (K bindings per episode) and 1.5×
the eval queries, so the per-cell breaker rate is set to 1.1185 × 1.5 =
**1.678** (ceiling ≈ 9,060 s vs ~6,000 s expected train wall — healthy
margin instead of the near-certain false abort the unscaled ceiling would
have produced). The realized K=12 rate gets measured and recorded in §7g.
**mi7 calibration-duty discharge:** every cell's pipeline already locks
Axis-C curves from its own Z-dump BEFORE any far-h behavioral eval and
prices itself against the breaker — the first-K=12-cell-per-arm duty is
satisfied by construction, and the MA3 asymmetric-confidence class
(PREDICTED-HOLD δ≤0.0079 / STRADDLE ≤0.0311 / PREDICTED-FAIL) is stamped
into every K=12 lock at write time (`ncr_spectral.k12_confidence_class`).
**Axis-B note:** the ≥10× WIN bar is pinned at h=1021, a K=8 grid point;
K=12 has no cost-probe row (pinned K=8-only) and its ladder carries no
1021 — K=12 timing at 765/1533 is recorded as supplementary, and the
Axis-B label of record remains K=8's WIN (no bar is moved). **Pre-launch
gate (the coordinator's term + this builder's own [LEARN]): the all-four-
arms end-to-end micro test is executed at K=12 BEFORE launch** — recorded
below with its output.

**Pre-launch gate EXECUTED (2026-07-11, local CPU):** all four arms ran the
end-to-end micro cell AT K=12 (30 steps, reduced grid keeping the full
12-residue sweep + h\*=57): 4/4 COMPLETED, sweep 12/12 labeled with all
three label classes present, h\*=57 claim-eligible on the claim path,
blank-outs PASS, K=12 locks written with the MA3 class stamped (untrained
→ PREDICTED-FAIL, as expected). Launch followed: tmux `ncr_phase2`, GPUs
2-7 (all idle-verified), 18 cells in one 3-per-GPU chunk round, launched
20:33:28Z.

### §7g PHASE-2 READOUT + CROSS-K VERDICT (2026-07-11, 18/18 K=12 cells, `PHASE2_DONE` 21:45:39Z): **K=12 AXIS A = SEP-PARTIAL → CROSS-K OVERALL = WIN-PARTIAL (the pre-registered publishable-with-caveat outcome)**

Archive: `experiment-runs/2026-07-11_ncr_phase2/` (repo tier + SSD full
mirror incl. 18 checkpoints); verdict `phase2_verdict.json` from the same
committed scorer (`--k 12`).

**AXIS A (K=12): SEP-PARTIAL.** NCR band DEGRADED — median recovered@0.9
at h\*=57 = **0.753** (per-seed 0.149 / 1.000 / 0.753 / 0.000 / 1.000),
uniform DEGRADED across all eight novel residue strata (52-59, no band
drop); best baseline FAIL (fwm 0.270, loopedvec 0.0). Cross-K per the
pinned §3.2a 10-pair table: **WIN (K=8, §7e) + SEP-PARTIAL (K=12) =
WIN-PARTIAL** — MA3's own pre-registered sentence realized verbatim ("a
K=12 SEP-PARTIAL alongside a K=8 WIN scores WIN-PARTIAL
(publishable-with-caveat), pre-registered here, not negotiated after the
readout").

**P1-K12 (the MA3 asymmetric-confidence prediction of record): 2/3 legs
PASS, 1 leg FAIL — scored via the LOCKED classes, never post-hoc.**
(i) Every PREDICTED-HOLD seed held: 2/2 (s1 δ=0.0044 → 1.000; s4
δ=0.0028 → 1.000). (ii) No PREDICTED-FAIL seed held: 0/1 (s3, a DEAD
seed — below). (iii) ≥half the STRADDLE seeds hold: **0/2 — FAILED**
(s0 δ=0.0125 → 0.149; s2 δ=0.0099 → 0.753). The directional claim ("truth
sits nearer the single-mode bound") is REFUTED at K=12 — and the sharper
instrument finding is that the CONSERVATIVE all-modes bound alone called
every seed exactly: conservative horizons 0.451/δ = 36 (s0), 103 (s1), 46
(s2), 161 (s4) → hold-at-57 iff horizon ≥ 57 predicts 5/5 outcomes
(including both STRADDLE misses, whose fronts landed at 45 and 57 — at
their conservative horizons, not their single-mode ones). K=8's
h=21-era evidence had suggested the single-mode reading; K=12 says the
conservative bound is the operative predictor. Pre-registered fronts:
s1/s4 at 189 (inside [87,442]); s0/s2 at 45/57 (before h\*, consistent
with their conservative horizons); zero post-front revivals (mi5's
predicted wrap points for these residuals sit at ≈714+/π-wrap, beyond the
trusted window — consistent).

**Dead seed s3 (1/5), disclosed:** never transitioned at 80K (in-dist
0.000, eff_rank(A) ≈ 1.3, locked δ = 1.41 → PREDICTED-FAIL, c\*
incoherent) — the archived dead/stuck-seed trainability-variance class
(K=16 precedent: 2/5 stuck), NOT a new failure mode; the locked
instrument classified it correctly before any far-h eval ran. **The 16
agreement-divergent points (first nonzero tally) sit EXCLUSIVELY on this
dead seed:** per-item |cos_binexp − cos_loop| 0.0007-0.008 over the 5e-4
bar while BOTH reads match their own fp64 shadows to ~1e-7 and mean_cos ≈
0 — the renormalized direction of a near-zero vector under a
rank-collapsed operator is ill-conditioned, so the two operation orders
legitimately diverge per item; MA5's pinned arbitration (fp64 shadow)
resolves it as an operator-degeneracy property, not an instrument defect.
Flagged, never dropped; ZERO flags on any converged seed at either K.

**P2 (K=12): PASS** — at the pinned h=45: loopedvec 0.0, fwm 0.4125
(margin 8.8pp — wider than K=8's 1.1pp). **AXIS C (K=12): TIE again** —
4/5 seeds ≤0.05 through h≤125 (0.0064/0.0085/0.0095/0.0101; the dead s3
0.0955), 0/5 to 509. **AXIS B (supplementary at K=12; the label of
record stays K=8's WIN per §7f):** binexp flat 1.6 ms at h=765/1533 vs
loop 49/101 ms, fwm 26/53 ms — the log-vs-linear separation replicates.
**Crosscheck-lens:** converged seeds eff_rank(A) = 12.00 with degauged
residuals 0.008-0.039 beside phase residuals 0.002-0.018; s3's degauged
column is meaningless (c\* sign-flipping) and is reported only with its
basis-invariant crosscheck per the §2.31a discipline. **Hygiene:** 0
shadow-divergent points; 0 reducer signatures; n_skipped 0.

**GPU-h ledger:** Phase-2 serial-sum **21.12** (rate ≈ realized within the
K-scaled allowance; wall 20:33→21:45 ≈ 1.20 h × 6 GPUs ≈ **7.2 device**).
Program totals: ≈ **40.2 serial-sum / ≈ 18.9 device** of the 120 cap;
wave-1+2 combined 39.5 serial vs the two 50 sub-caps.

**§7e ERRATUM (recorded per the finding-14 publisher's re-verification;
each value re-verified against the archived raws by this agent before
recording; §7e itself left untouched):** (1) §7e's quoted fwm decay curve
blended Phase-0 single-seed calibration values with wave-1 5-seed
medians; the wave-1 raw-derived median curve is 0.923@5 → 0.748@13 →
0.611@21 → 0.489@29 → 0.158@61 → 0.014@125 → 0.001@253 → 0.000@509.
(2) The P2 margin is 1.147pp → "1.1pp under the bar", not 1.2pp. (3) The
extreme-depth timing quoted Phase-0 preview values; wave-1's own archive
reads binexp 2.64 ms vs O(h) arms 34.69-66.10 s at h=2^20+5 → ≈
13,155-25,064×. **Claim-language note (binding on any write-up):** learned
depth selection was killed ANALYTICALLY at design review (§2 F1/§3.1 —
matrix-polynomial phase-mixing + the PonderNet-cousin argument), never
empirically tried; phrase as a design exclusion, not an experimental
result.

**STATUS: the NCR wave-1+2 program verdict of record is CROSS-K
WIN-PARTIAL (Axis A), WIN (Axis B, K=8 bar), TIE (Axis C), with P2
confirmed at both K and P1 5/5 at K=8 / P1-K12's straddle leg refuted in
favor of the conservative bound. The wave-2/operator-bank go/no-go
returns to the coordinator with this record; the operator bank remains
separately ledgered and double-gated (M4), untouched.**

### §7h K=12 SEED-EXTENSION — PRE-REGISTRATION (2026-07-11, before any
GPU touched)

§7g's own text names the path: "a future K=12 seed-extension moves the
K=12 band." This section pre-registers that extension exactly, per the
`h2h` §1.8 seed-extension template (extension seeds only, cost disclosed
up front, outcome map fixed BEFORE the readout — never re-argued
post-hoc) and MA3's asymmetric-confidence machinery (§3.2, classify each
seed from its own locked calibration residual BEFORE any far-h eval,
score leg-by-leg).

**(1) Scope.** N=5 fresh NCR-arm seeds at K=12 — seeds **5, 6, 7, 8, 9**
(never used at K=12; seeds 0-4 are the archived §7g pool) — 80,000
steps, the IDENTICAL frozen config §7f/§7g ran (`ncr_task.py` md5
`3c664c0c0571fc361692edd220ab21d2`, `run_ncr.py` md5
`77545140b59295121f7f55afbe1c7b6a`, `ncr_models.py`/`ncr_spectral.py`
unchanged — re-verified byte-identical against the box, this session,
before writing this section). **NCR arm only.** LoopedVec and FWM are
settled per §7g (`fwm` median 0.270, `loopedvec` median 0.0 at h\*=57,
both FAIL band) and are explicitly NOT re-run — mi6/§7d's own
no-baseline-swap-or-addition rule extends here by the same logic: an
already-scored comparison-of-record arm does not get a second sample
just because the contender is being extended.

**(2) Per-seed locked prediction, scored leg-by-leg (MA3 machinery,
reused verbatim).** Every trained ncr cell's pipeline (`run_ncr.py`
`run_cell`, lines 578-618) writes the Axis-C lock —
`mean_predicted_curve`, `phase_resid_max_mean`, and (K=12 only)
`k12_confidence_class` via the pinned `ncr_spectral.k12_confidence_class`
thresholds (`K12_PREDICTED_HOLD_DELTA=0.0079`, `K12_STRADDLE_DELTA=
0.0311`, unchanged) — BEFORE `eval_cell` runs the far-h ladder/h\*
behavioral eval, and the lock is hash-verified (`verify_axis_c_lock`) on
read. This satisfies MA3's "locked before far-h eval" duty **by
construction**, exactly as §7f recorded for the original 5. The scoring
procedure, fixed now: for each of the 5 new seeds, read
`k12_confidence_class` + `phase_resid_max_mean` from its `.axis_c_lock.
json` and RECORD the class (PREDICTED-HOLD / STRADDLE / PREDICTED-FAIL)
as the seed's locked prediction; only then read `recovered_frac@0.9` at
h\*=57 from `eval.points` and classify HOLD (≥0.9) / not-hold (<0.9).
Score three legs, both for the 5 new seeds alone and pooled with the 5
archived seeds into a 10-seed tally: **(i)** every PREDICTED-HOLD seed
holds; **(ii)** no PREDICTED-FAIL seed holds; **(iii)** at least
`ceil(n_STRADDLE / 2)` STRADDLE seeds hold. The archived pool's own
locked classes (re-stated, never re-scored): s0 STRADDLE (δ=0.0125,
FAILED to hold, 0.149), s1 PREDICTED-HOLD (δ=0.0044, held, 1.000), s2
STRADDLE (δ=0.0099, held, 0.753 ≥0.9? **no** — 0.753 < 0.9, so s2 is
scored not-hold for THIS leg-tally even though its DEGRADED-band value
feeds Axis-A's median differently; leg-scoring uses the 0.9 threshold
directly, band assignment is a separate downstream step), s3
PREDICTED-FAIL (δ=1.41, dead/not-hold, 0.000), s4 PREDICTED-HOLD
(δ=0.0028, held, 1.000).

**(3) Pooled Axis-A outcome map, on the frozen §3.2a text, quoted
verbatim.** "Per-arm numeric band at h\*, on median recovered_frac@0.9
across seeds: **HOLD ≥ 0.9; DEGRADED ∈ (0.5, 0.9); FAIL ≤ 0.5**" and the
9-cell (NCR band, best-baseline band) table: `(HOLD,FAIL)=WIN`,
`(DEGRADED,FAIL)=SEP-PARTIAL`, `(FAIL,FAIL)=TIE` (the three cells
reachable here, since best-baseline stays FAIL — see below). Cross-K
rule, also quoted verbatim: `WIN+WIN=WIN`, `WIN+SEP-PARTIAL=WIN-PARTIAL`,
`WIN+TIE=WIN-PARTIAL`. K=8 is frozen at WIN (§7e). The pooled ncr median
is computed over all 10 seeds' recovered_frac@0.9 at h\*=57 (order
statistics: for n=10, median = mean of the 5th and 6th ascending
values); best_baseline_band is recomputed by the SAME extended harvest
against the SAME 5 archived loopedvec/fwm cells (unmodified, unrerun) —
it is mathematically fixed at **FAIL** (0.0 and 0.270, both ≤0.5;
nothing about extending the ncr arm can move it) unless the harvest
code has a defect, which the audit below checks.

Derivation (order-statistics, not invented — the archived 5 raw values
are 0.000, 0.149, 0.753, 1.000, 1.000; 2 already HOLD, 3 already
sub-0.9):

- **Pooled HOLD, guaranteed** (→ K=12 label **WIN** → cross-K
  **WIN+WIN=WIN**, the full unqualified capability headline, upgrading
  from WIN-PARTIAL): if **≥4 of the 5 new seeds** hold (recovered_frac@
  0.9 ≥ 0.9 at h\*=57) — this leaves at most 4 of the pooled 10 below
  0.9, which forces both the 5th and 6th ascending order statistics
  into the ≥0.9 group, so the median is guaranteed ≥0.9 regardless of
  the exact sub-0.9 values. Sufficient, not necessary — the literal
  computed median is the scoring rule of record, this is a planning
  aid.
- **Pooled FAIL, guaranteed** (→ K=12 label **TIE** → cross-K
  **WIN+TIE=WIN-PARTIAL**, SAME overall label as today, even though the
  per-K band itself moved down): if **≥4 of the 5 new seeds** fail
  (recovered_frac@0.9 ≤ 0.5 — i.e. more dead/stuck seeds, the archived
  trainability-variance class, K=16 precedent 2/5 stuck) — this pushes
  ≥6 of 10 pooled values to ≤0.5, forcing both middle order statistics
  ≤0.5.
- **Otherwise (the modal expectation** — MA3's own text: "archived
  analogs predict a seed mix of roughly 1/3 PREDICTED-HOLD, 2/3
  STRADDLE," i.e. most new seeds land DEGRADED-ish, not clean holds or
  clean fails): pooled median stays in (0.5, 0.9) → K=12 label stays
  **SEP-PARTIAL** → cross-K stays **WIN-PARTIAL**, UNCHANGED from
  §7g's recorded verdict, but now resting on n=10 instead of n=5 —
  reported as a firmed-up, not moved, verdict.

**Explicit statement per the charter: cross-K moves to the full WIN
headline ONLY in the guaranteed-pooled-HOLD case above; every other
reachable outcome (stays SEP-PARTIAL, or drops to TIE) leaves the
recorded cross-K label at WIN-PARTIAL — §7g's WIN-PARTIAL is NOT at
risk of downgrading to TIE or LOSE from this extension, because the
best-baseline band cannot move (it is not re-run) and NCR's band can
only move within {FAIL, DEGRADED, HOLD}, none of which pairs with a
frozen FAIL baseline to anything worse than SEP-PARTIAL.** **NO K=8
number changes:** K=8's 18/18 cells (§7e, `experiment-runs/
2026-07-11_ncr_wave1/wave1_verdict.json`) are untouched, unread, and
unrerun by this extension; the K=8 label stays WIN, closed, not part of
this wave's scope.

**(4) Cost.** Conservative (breaker-ceiling) projection, matching the
§7f-pinned rate exactly (identical config ⇒ identical breaker):
5 × 1.678 ≈ **8.4 GPU-h** ceiling. Informed projection from the
MEASURED §7g ncr-arm-specific realized rate (this session, read from
the 5 archived `ncr_ncr_K12_s*.json` `gpu_h` fields: 1.1422, 1.1422,
1.1422, 1.1642, 1.1643 — mean ≈ **1.151 GPU-h/cell**): serial-sum ≈
**5.76 GPU-h**; device draw ≈ the same (one cell per GPU, no
co-location contention planned — see §7i launch mechanics) ⇒ wall
≈ 1.15 h. Ledger: 120 GPU-h cap, §7g realized ≈40.2 serial-sum / ≈18.9
device; after this extension ≈46.0-48.6 serial-sum / ≈24.7-27.3 device
— inside the cap with more than 70 GPU-h of headroom untouched, and
this wave is NOT charged against either closed wave-1/wave-2 50-GPU-h
sub-cap (both already closed at their own totals; this is a new,
separately-ledgered Phase-2-extension line item against the overall
120 only).

**Harvest-code note, pre-registered (audited before use, per CLAUDE.md
"the implementer does not review their own work"):** the committed
`wave1_harvest.py` requires a UNIFORM `--expect-seeds` across ncr/
loopedvec/fwm; the pooled read needs ncr=10 against loopedvec=fwm=5. The
planned change is a minimal, additive `--expect-seeds-ncr` override
(defaults to `--expect-seeds`, so every EXISTING invocation — the K=8
and K=12 wave harvests already on record — is byte-for-byte unchanged);
teeth = (a) replay the modified script against the two ALREADY-ARCHIVED
result directories (`experiment-runs/2026-07-11_ncr_wave1/`,
`experiment-runs/2026-07-11_ncr_phase2/`) with the OLD-style invocation
and diff the output against the archived `wave1_verdict.json`/
`phase2_verdict.json` byte-for-byte (regression: proves the change
altered nothing for existing call sites); (b) an executed positive +
negative (kill-proof) unit test of the new per-arm count-resolution
function in isolation. An independent (opus-tier) agent audits the diff
+ executed test output before the harvest is trusted for the real
pooled read.

**EXECUTION (this section commits BEFORE any of it runs):** per-arm
(ncr-only, since baselines don't re-run) micro end-to-end test at K=12,
reduced steps, one fresh seed, mirroring §7f's own pre-launch gate;
launch via a new `launch_k12ext.sh` (the §7f/launch_phase2.sh worker +
supervisor + tmux + STOP-file + DONE-sentinel pattern, unmodified
architecture), tmux session `ncr_k12ext`, GPUs 2-7 (this wave uses 5
cells on 5 of the 6 idle GPUs — 2,3,4,5,6 — one cell per GPU, no
co-location needed or planned since cell count < GPU count; GPU 7 held
idle as spare/reserve capacity, satisfying "GPUs 2-7" as the available
pool without forcing artificial contention); resume-safe by construction
(`run_cell`'s existing COMPLETED-status skip, unchanged). GPUs 0-1 (the
FIX-5 grid + reasoning-link validation) are never touched. On
completion: harvest (per the audited extension above) → §7i (RECORD
FIRST) → archive → EXPERIMENT_LOG → commits.

## §8 OPERATOR BANK (opened 2026-07-11/12, GPU refill campaign)

**Gate discharge, both legs of M4.** (1) Stage 2's calibration readout
is discharged (`CAPABILITY_SEPARATION_DESIGN.md` §2.30, SWEEP-READY on
disk) — the fla-vs-torch and readout-diagnostic lessons it exists to
transfer are already absorbed into the single-relation NCR build
(§7 throughout: torch-bespoke fp32, no fla consumer, the closed-form
checks at §7's build item 11). (2) The single-relation wave-1+2 program
is COMPLETE and WON: K=8 WIN (§7e), cross-K WIN-PARTIAL (§7g), a K=12
seed-extension pre-registered (§7h) and executed on the box
(`K12EXT_DONE` sentinel present, `/home/nvidia/ncr/results_k12ext/`,
5 fresh seeds) — that extension's harvest/record is a DIFFERENT thread's
work-in-progress and is explicitly NOT touched by this section. Both M4
legs are open; the operator bank proceeds.

**Scope discipline (this round, under an explicit no-idle-GPU push):**
this pre-registration is deliberately MINIMAL, not the full wave design
— Phase-0 informs wave-1 sizing, exactly as the single-relation
program's own §3.6 did. Cut relative to a maximal design: no K=12 cell
this round (backlog, §8.1.6); the bounded-chain axis gets a small
disclosed probe, not a full grid (§8.1.3).

### §8.1 Design + pre-registration

**8.1.1 Task.** One shared entity pool, `d=16`, `K=8`, `N=K=8`
(orthonormal, permutation variant — identical convention to
`task_e.TaskEConfig`). **R=3 independent relations** π₀,π₁,π₂, each its
OWN single Hamiltonian K-cycle over the SAME 8 entities (task_e's
`_permutation_graph` generator called R times per episode, independent
draws). Distinctness is checked, not assumed: at generation time assert
the R cycle-permutation matrices are pairwise distinct (`assert not
torch.equal(...)` for every pair) — with 5040 directed Hamiltonian
K=8-cycles the collision probability of 3 iid draws is astronomically
small, but the assert makes it a checked invariant, not a hoped-for one,
per the injectivity-check precedent in `task_e.py`'s own `_assert_injective`.
One episode presents all `R*K = 24` bindings together (key_i → π_r(key_i)
for every r, every i), each binding token tagged with a learned
relation embedding (added post-projection, not concatenated — keeps
`in_proj` unchanged from `BindingEncoder`, only `rel_embed: (R,h)` is
new). This is the load-bearing design choice: R operators are written
FROM ONE SHARED CONTEXT, by a SHARED transformer trunk, competing for
the same attention/parameter budget — the thing that makes "bank"
(interference risk) a genuine empirical question rather than R
independent copies of the already-won single-relation result.

**8.1.2 Architecture — `BankBindingEncoder`.** Extends
`model_v4.BindingEncoder` (chapter2, verbatim trunk) with: (a)
`rel_embed: nn.Parameter(R, h)` added to each binding token post-`in_proj`
(query tokens for extraction, see below, get no relation tag — they are
relation-agnostic reads); (b) `row_queries: (R, d, h)` replacing the
single-relation `(d, h)` — R independent sets of d row-readers, one set
per relation; (c) the `reader`/`row_norm`/`row_out` weights are SHARED
across all R extractions (this is deliberate: forcing R different
learned query vectors to pull DIFFERENT information out of the SAME
shared `mem` via the SAME attention/projection weights is a strictly
harder interference test than giving each relation its own private
reader would be). `forward(keys, values, rel_ids) -> Z_bank: (B,R,d,d)`,
batched over `B*R` for the reader call. Params: shared trunk unchanged
(≈ the single-relation encoder's trunk cost); the only new parameters
are `rel_embed` (R·h = 192 for R=3,h=64) and `row_queries` growing R×
(3·16·64=3072 vs 1024) — a small, cheap extension, computed exactly at
build time via the same `param_report`/`assert_param_match` pattern
(§3.3 of the single-relation design), not eyeballed.

**8.1.3 Axes.**
- **Axis R-BANK (headline, primary claim).** Single-block queries
  `(r, h)`: `pred = π_r^h(query_entity)`, read via `Z_bank[:, r]` through
  the EXISTING `binexp_read`/`loop_read` (relation-agnostic once `Z_r`
  is extracted — zero new read code). Held-out depth per relation reuses
  the PINNED `ncr_task.GRIDS[8]` verbatim (train {1,2,3}, ladder, h*=61,
  sweep, cost_probe) — `r` is an orthogonal selection axis, not a
  depth-periodicity axis, so no new mod-K reasoning is needed; the
  EXISTING `task_e.TaskEConfig.__post_init__` guard is reused per
  relation, unmodified. Capability claim: exact composition + O(log h)
  read SURVIVES when R operators share one write context (vs. the
  single-relation claim: exact composition + O(log h) read for ONE
  operator). Novelty axis (M1, unchanged): the delta vs. TensorLog/
  Neural-LP/DRUM (arXiv:1605.06523/1702.08367/1911.00055) stays
  "in-context-written, not static-weight" — now literally demonstrated
  with R>1 operators coexisting in one context, closer to their
  multi-relation KG setting than the single-relation wave ever was.
- **Axis LOG-DEPTH (headline, cost, inherited).** Identical bar to the
  single-relation Axis B (§3.2/§7e): bin-exp ≥10× faster than the best
  O(h) arm at large h. Measured on relation r=0 only — binexp/loop_read
  cost is a pure function of `h` and the extracted `Z_r`'s shape, not of
  which relation was selected, so measuring on 3 relations would be
  redundant compute, not a stronger test; disclosed as such.
- **Axis B-CHAIN (exploratory, secondary, non-blocking, disclosed
  scope-limited).** Two-block programs `(r1,h1,r2,h2)`, r1≠r2 required
  (r1=r2 degenerates to Axis R-BANK and is run ONLY as an internal
  consistency check, never scored as a claim): `pred =
  π_{r2}^{h2}(π_{r1}^{h1}(query_entity))`, read as `binexp_read(Z_{r2},
  binexp_read(Z_{r1}, q, h1)['o'], h2)`. Cost = O(log h1) + O(log h2) =
  O(log h) since **B is FIXED at 2, never query-controlled** — this is
  exactly the boundary M4 drew: a heterogeneous chain whose LENGTH grows
  with the query loses log-depth and sits in Neural-LP/DRUM/FWM
  territory (M4, quoted: "the relation chain Z_rn···Z_r1 has NO squaring
  shortcut... loses log-depth"); a chain whose length is a fixed O(1)
  constant does not — B=2 is disclosed as a scope limit, NOT claimed to
  generalize to variable B. Eval grid deliberately small (Phase-0
  informs sizing; ballpark: h1=h2=h*=61 crossed over the 6 ordered
  (r1,r2) pairs, ≈0 GPU-h, eval-only). Reported as HOLD/DEGRADED/FAIL,
  no WIN/TIE/LOSE label required for this wave — a LOSE here does not
  sink the Axis R-BANK or LOG-DEPTH verdicts.

**8.1.4 Baselines (M3, mi6 pinned families, ±15% param match).**
- `loopedvec-bank`: SAME pinned step-map family (weight-tied 2-layer
  GELU MLP residual step) as the single-relation `LoopedVecModel`; the
  only change is the encoder producing `x0` per `(r, query)` pair from
  the R*K-token shared context (same relation-embedding tagging as
  `BankBindingEncoder`). `h` and `r` both consumed only via masking /
  initial-state selection — no new episode conditioning beyond the
  state, per mi6.
- `fwm-bank`: SAME write (`BankBindingEncoder`, own weights), read =
  h-fold recursive LN read on the SELECTED `Z_{r}` (or `Z_{r2}` after a
  prior `Z_{r1}` block for Axis B), O(h) sequential, LN affine weight-
  tied across hops — verbatim extension of the single-relation
  `FWMReadModel.read`.
- `cmlp-bank`: disclosed weak control, one-hot(h)⊗one-hot(r) extension
  of the inherited `MLPShortcutModel`; never the comparison of record,
  never param-matched by design (same convention as the single-relation
  `CMLPModel`).
- Param match computed exactly at build time (`assert_param_match`
  extended, ±15% vs `ncr-bank`), not eyeballed.

**8.1.5 Hard-rule bakein (all apply fresh to this forward pass, per
`CLAUDE.md` — none inherited from the single-relation build without
re-verification).**
- **Single full K-cycle, not general permutation:** each of the R
  relations independently a full Hamiltonian K-cycle (task_e generator,
  reused verbatim); mod-K periodicity guard reused per relation via the
  existing `TaskEConfig.__post_init__` assert, unmodified.
- **Exact continuous readout:** cosine recovery against continuous
  targets throughout, no argmax/codebook anywhere (unchanged from the
  single-relation design).
- **P=1 hard bottleneck / blank-out:** extended and re-verified for THIS
  forward pass (not inherited): corrupt the raw `R*K` binding tokens
  post-write, confirm `Z_bank` (hence every read) is bit-identical and
  `∂o/∂(raw inputs post-write)` is exactly zero — gradient-based, not a
  shape check.
- **NEW gate specific to the bank (this section's own P=1 analog):
  relation-ID-swap ablation.** At eval, feed the WRONG relation id
  `r'≠r` for a query whose true relation is `r`, same entity/depth.
  Since π_r are independent random cycles, `π_r(a) ≠ π_{r'}(a)` almost
  surely for K=8 — a model that ignores relation-id and just memorizes
  "the" operator would show NO degradation under this swap. MANDATORY,
  executed to completion, numeric threshold (not a shape check): median
  cos(pred_wrong_r, true_target_r) at h* must be indistinguishable from
  a random-direction control (bar: < 0.3, pinned before any cell runs).
  This is a validity GATE (must-pass precondition), not a scored claim
  axis — same status as blank-out.
- **Geometric trust rule (M2/§3.4), re-verified per relation:** the
  corrected scale-normalized `‖C‖/σ_min(A)` rule with its N1/N2 negative
  tests (amplifying near-normal + non-normal nilpotent, both already
  executed once for the single-relation build) is re-run at Phase-0
  against EACH of the R extracted `Z_r`, not assumed to transfer as a
  single global check — each `Z_r` is an independently-trained
  sub-operator and could in principle have different leakage geometry.
- **fla-transpose lesson (§17/§2.26): explicitly N/A, disclosed.**
  `BankBindingEncoder` is bespoke fp32 torch (verbatim `BindingEncoder`
  trunk extension) — no `chunk_delta_rule` or any fla consumer anywhere
  in this build, identical to the single-relation arm's own M5 finding
  ("no fla kernel computes powers"). No closed-form fla-state-layout
  check applies; the closed-form checks that DO apply are the same
  `binexp_read`/`loop_read`-vs-literal-fp64-power agreement checks
  already in `ncr_models.py`'s self-test, extended to run against a
  `Z_r` slice of `Z_bank` instead of a bare `Z` (shape-only delta).
- **Negative tests run to completion, not just written (hard rule):**
  relation-distinctness assert (8.1.1), blank-out, relation-ID-swap
  ablation, N1/N2 geometric-rule negative tests ×R, param-match assert —
  ALL must execute with archived output before the build is trusted, at
  Phase-0, mirroring the single-relation §3.8(c) gate.

**8.1.6 WIN/TIE/LOSE and scope cuts.**
Bank-level score for Axis R-BANK = **min over r∈{0,1,2}** of each
relation's median recovered_frac@0.9 at h* (the conservative,
worst-relation aggregation — a bank claim requires ALL three relations
to work, not the easiest one). HOLD/DEGRADED/FAIL bands unchanged from
§3.2a (≥0.9 / (0.5,0.9] / ≤0.5). WIN = ncr-bank band HOLD AND
best-baseline bank band FAIL (baseline's own bank score computed the
same min-over-r way). Axis LOG-DEPTH: unchanged ≥10× bar. Axis B-CHAIN:
reported, non-blocking (8.1.3). **Explicit cuts, disclosed, backlog for
a later wave, not silently dropped:** no K=12 cell this round (K=8
only); Axis B-CHAIN eval grid is a small disclosed probe, not the full
crossed ladder; R is fixed at 3 for this wave (not swept).

**8.1.7 GPU-h ledger — OWN cap, separate from the single-relation
program's 120 (per M4/§3.6: "NOT authorized by this cap, own design
round required").** Coordinator-set: **80 GPU-h total cap, wave-1
(Phase-0+Phase-1) sub-cap ≤50.** Rate: UNKNOWN a priori (R*K=24 tokens
vs the single-relation arm's K=8 — token count 3×, but transformer
self-attention cost is not necessarily linear in token count at this
tiny scale, d_model=64 dominates) — Phase-0's job is to MEASURE the real
rate, exactly as §3.6/§7c did for the single-relation build; no wave-1
cell count is committed until that number exists.

| Phase | Cells | Plan | Gate |
|---|---|---|---|
| **0 — calibration (mandatory first)** | 3 (one per trained arm: ncr-bank, loopedvec-bank, fwm-bank) | reduced calibration steps (mirrors §7c precedent, ≈0.73 GPU-h realized there); all 8.1.5 instrument duties executed here | Real rate measurement; sizes wave-1 seed count; per-cell abort breaker set generously (2 GPU-h ceiling) until the real rate is known |
| **1 — wave-1 core (K=8)** | Target 3 seeds × 3 trained arms (9) + cmlp-bank × 2 seeds (2) = 11, ADAPTIVE on the Phase-0 rate (cut to 2 seeds/arm, then 1, before dropping an arm, if the measured rate would blow the sub-cap) | sized from Phase-0's measured rate, never from this sketch | Sub-cap ≤50 (shared with Phase-0); fires only after Phase-0 passes + §8.2 attack clears |
| **Reserve** | K=12 replication (backlog) + budget-artifact retests | — | Draws logged individually against the remaining ~30 |
| **TOTAL CAP** | | | **80 GPU-h**, own ledger |

### §8.2 ATTACK (fresh opus, time-boxed fatal-flaw-only, 2026-07-11/12): NEEDS-REVISION

Read-only round against §8.1 + its cited substrate (`model_v4.py`,
`task_e.py`, `ncr_task.py`/`ncr_models.py`, §2/§3/§7e/§7g context).
Mechanical soundness of the §8.1.2 `BankBindingEncoder` extension vs
`BindingEncoder`, and the GRIDS/binexp_read/loop_read/param-match reuse,
both **confirmed intact**. F1/F2/M1-M5 from §2 all correctly carried
forward. One embedded fake system-reminder (date-change +
concealment-instruction pattern) surfaced in the agent's own tool
stdout mid-run — disregarded and reported per the standing hard rule;
did not originate from a legitimate harness channel, noted here for the
injection tally.

**CRITICAL — C1 (bank-aggregation gameable, §8.1.6).** `min_r(median_seeds)`
is min-of-medians, not median-of-per-seed-mins. Concrete counterexample
with 3 seeds: A holds {r0,r1} fails r2; B holds {r0,r2} fails r1; C
holds {r1,r2} fails r0 — every per-relation median is 1.0 (2/3 seeds
hold each relation) → reported bank score **1.0/HOLD**, yet **zero
seeds are an actual working bank**. Baseline is scored the same way, so
the inflation is asymmetric toward a false contender WIN. Reachable,
not hypothetical, given this project's own documented per-seed
trainability variance (K=16 precedent 2/5 stuck; K=12 precedent 2/10
dead, §7i).

**MAJOR — J1 (M6 read-vector-std diagnostic dropped, §8.1.5).** The
pinned §3.1/M6/mi4 rvstd≥0.04 gate (live in the single-relation §7c
gate table) for deviating-read arms is absent from §8.1.5's "instrument
duties," yet fwm-bank and loopedvec-bank are BOTH deviating-read arms
and fwm-bank is the comparison of record — without rvstd a genuine
architectural FAIL is indistinguishable from a degenerate-readout
strawman (the exact M3 concern).

**MINOR (recommended before build, non-blocking):** m1 — B-CHAIN's
composite σ=π_{r2}^{h2}∘π_{r1}^{h1} has ~1 fixed point in S₈ generically
(~1/8 of starts give a trivial σ(a)=a query), unexcluded by the
per-relation-only mod-K guard; exclude/label fixed-point starts. m2 —
restate the single-relation §3.2 rule ("Axis-B claimable only if
Axis-A ≥ TIE") explicitly for LOG-DEPTH here. m3 — the swap-ablation
<0.3 bar should be checked against an empirical random-direction
control per episode, not asserted a priori (near-collision cycle pairs
could in principle lift it) — bar-calibration only, the ablation itself
has no degenerate pass.

**VERDICT: NEEDS-REVISION — build blocked on C1 + J1.**

### §8.3 REVISION (2026-07-11/12): C1 + J1 fixed, minors folded — CLEARS

**Fix C1.** §8.1.6 bank score redefined: per-seed bank recovery =
`min_r(recovered_frac@0.9(r, seed))` computed FIRST, THEN
`median_seeds` of that per-seed number (median-of-mins, not
min-of-medians). Additionally report `n_seeds_all3_hold` (count of
seeds where all 3 relations individually HOLD) alongside the median, so
a HOLD verdict is legible as "most seeds are genuine working banks,"
not inferable only from the aggregate. Re-run against the attack's own
counterexample: per-seed mins are A=0 (fails r2), B=0 (fails r1), C=0
(fails r0) → median-of-mins = **0.0 → correctly FAIL**, not the false
1.0/HOLD the old formula gave. Baseline scored identically (median-of-
mins), so the fix is symmetric, not a thumb on the scale either
direction.

**Fix J1.** §8.1.5 instrument duties gain a fourth bullet: **read-
vector-std diagnostic (§3.1/M6/mi4, rvstd ≥ 0.04), executed at Phase-0
for both deviating-read bank arms (fwm-bank, loopedvec-bank)** —
verbatim reuse of the single-relation `read_vector_std` function
(`run_ncr.py`), applied to the bank arms' own read vectors. A FAIL here
routes the same way it did in the single-relation build: flagged, not
silently absorbed into the arm's headline FAIL band.

**Minors folded.** m1: B-CHAIN eval excludes/labels any (r1,h1,r2,h2)
whose composite fixed point coincides with the query start (`σ(a)=a`
checked at eval-batch construction, mirroring the identity-residue
exclusion already in `ncr_task.residue_label`). m2: §8.1.3 LOG-DEPTH
bullet gains one sentence: "claimable as capability only if Axis
R-BANK ≥ TIE, per the single-relation §3.2 rule, restated here." m3:
swap-ablation gate computes its own per-episode random-direction control
(cos of `pred_wrong_r` against a batch of freshly-sampled random unit
vectors) alongside the pinned <0.3 bar, both reported.

**C1 was a CRITICAL — a scoped re-attack is dispatched (§8.3a), not
waived.** The in-line counterexample re-check above is the implementer's
own verification and does not substitute for independent review (the
"implementer does not review their own work" hard rule applies to a
self-certified fix exactly as it does to the original build). Build
does not start until §8.3a returns clean.

### §8.3a SCOPED RE-ATTACK (fresh opus, independent, 2026-07-11/12): **C1 CLOSED — CLEARS FOR BUILD**

Independently re-derived the C1 fix's correctness (median-of-per-seed-
mins at odd n makes a false HOLD impossible — the middle order statistic
≥0.9 forces a majority-genuine-bank population; the original C1
counterexample's per-seed mins {0,0,0} now correctly median to 0.0/FAIL)
and confirmed baseline scoring is symmetric (identical formula, max-
over-baselines-must-FAIL is the anti-inflation direction) and J1's rvstd
restore faithful (exact 0.04 threshold, both deviating arms named,
non-absorbing routing). m1/m2/m3 all independently confirmed CLOSED
(m1's exact-integer fixed-point check satisfies the exact-threshold hard
rule; m3's ambiguity is conservative-direction, non-blocking). No new
CRITICAL/MAJOR. Two MINOR non-blocking recommendations, folded now
rather than deferred: **(a)** `n_seeds_all3_hold` is reported but was
not gated at even seed counts (2/2 moderate-but-sub-0.9 per-seed-mins
can't average to HOLD, but the misrepresentation floor is weaker than
odd-n's) — build gates `n_seeds_all3_hold ≥ ceil(n/2)` as an EXPLICIT
co-condition of any bank-level HOLD verdict, not just a reported number.
**(b)** rvstd≥0.04 is re-checked at the wave-1 fwm-bank/loopedvec-bank
result that actually decides the WIN/LOSE verdict, not only at Phase-0
(a Phase-0-only check would miss a wave-1-only readout collapse).

**§8.1/§8.3 pre-registration is CLEAR. Proceeding to §8.4 BUILD.**

### §8.4 BUILD (2026-07-11/12): CPU self-test 10/10, real-CUDA smoke 5/5 after one fix — awaiting independent audit (§8.4a)

**New files (`matrix-thinking/ncr/`, extend the substrate, do not touch the
WON single-relation code):** `ncr_opbank_task.py` (BankConfig, per-relation
Hamiltonian K-cycle generation + distinctness check, Axis R-BANK/Axis
B-CHAIN batch samplers, `bank_score` — the C1-fixed median-of-per-seed-mins
aggregation with the `n_seeds_all3_hold`/quorum gate), `ncr_opbank_models.py`
(`BankBindingEncoder` — R independent row-query sets, shared trunk/reader/
row_out, per-relation embedding; `NCRBankModel`/`FWMBankModel`/
`LoopedVecBankModel`/`CMLPBankModel`, `assert_param_match`),
`ncr_opbank_selftest.py` (CPU suite, 10 tests, all negative tests executed
to completion with kill-proofs, not just written), `run_ncr_opbank.py`
(Phase-0-only driver: `train_cell_bank`, `z_dump_bank`/`deep_probe_bank`
[`ncr_spectral`'s discharged S3.4 machinery reused verbatim, called once
PER RELATION], `blank_out_check_bank`, `relation_id_swap_ablation`,
`read_vector_std_bank`, `eval_cell_bank_small`, `run_cell_bank`,
`phase0_bank`). Wave-1's full eval grid (ladder/sweep/cost_probe/B-CHAIN
crossed grid) is OUT OF SCOPE for this build — sized after Phase-0's real
rate, per §8.1.7's own pin.

**CPU self-test suite, `ncr_opbank_selftest.py`, 10/10 PASSED, every
negative test a genuine kill-proof (constructed to fail first, confirmed to
catch it, not a vacuous pass):** t1 relation-distinctness (duplicate-
relation kill-proof + positive); t2 mod-K guard reused per relation
(train-residue held-out point correctly rejected); t3 the C1 counterexample
literally re-run — OLD min-of-medians formula reproduces the false HOLD
(1.0), FIXED median-of-mins correctly gives FAIL (0.0); t4 the S8.3a
even-n quorum-gate fold (2/2 HOLD+gated vs 1/2 correctly DEGRADED+not-
gated); t5 B-CHAIN fixed-point exclusion (WITHOUT exclusion: 20/240 fixed
points occurred on a fixed generator stream — the confound is real, not
hypothetical; WITH exclusion, same stream: 0/240); t6 param match (all
trained arms ±15%, `fwm-bank` ratio 1.0002, `loopedvec-bank` 0.9881); t7
closed-form binexp/loop-vs-literal-fp64-power agreement on EVERY bank
relation (not just r=0); t8 blank-out (P=1 bottleneck) PASSING for all 3
gradient-capable arms, untrained (mechanism-level); t9 relation-ID-swap
ablation kill-proof — an oracle model (`Z_bank` = the true per-relation
`z_ideal`, C7's classical operator) shows right-r recovery 1.000 vs
wrong-r 0.000 (gap 1.000, clears the 0.3 bar by a wide margin), proving the
diagnostic detects a genuine difference when one exists, complementing the
untrained near-null result (no false positive) with a true positive; t10
end-to-end micro cell COMPLETES for ALL 4 arms (the §7e "all arms, not one
representative" lesson).

**Real-device smoke caught a genuine bug the CPU suite did not (exactly the
CLAUDE.md "CPU-stub suites test logic only" hard rule doing its job):**
first real-CUDA `--smoke` run on the box crashed with a relation-
distinctness assertion — the original §8.1.1 text's "collision probability
... astronomically small" claim was WRONG. Corrected math: for K=8 there
are (K−1)!=5040 distinct Hamiltonian-cycle functions, so P(≥1 collision
among R=3 iid draws) ≈ 5.95e-4 **per episode row** — small per draw, but a
single Phase-0 cell's full pipeline (train steps + z_dump + blank_out +
swap ablation + rvstd + eval grid) draws thousands of independent rows
across dozens of batches, so the expected collision count per full cell
run is ≈0.5–1 — likely, not astronomical. **Fix:** `_relation_graphs`
retries the whole-batch draw (cheap; P(any collision in a B=64 batch)
≈3.7%, so ≥2 retries needed only ≈0.1% of the time) up to 20 times before
raising via the unchanged `assert_relation_distinct` — the checked
invariant survives (a PERSISTENT duplicate, e.g. a genuine generator bug,
still raises), only the treatment of an expected, non-adversarial
collision changes from a hard crash to a transparent resample. Re-deployed
(md5-verified) and re-run: **5/5 clean real-CUDA smoke passes** (GPU 0,
`~/tdenv/bin/python`, all 4 arms each run). `ncr_opbank_task.py`'s S8.1.1
collision-probability claim is corrected in the build record here rather
than reopening §8.1 (an implementation-robustness fix, not a design-surface
change — no re-attack triggered).

**fla-transpose lesson (§17/§2.26), executed as a closed-form check per
S8.1.5's disclosed N/A:** no fla consumer exists in this build (bespoke
torch fp32 throughout); the applicable closed-form check —
`binexp_read`/`loop_read` vs the literal fp64 matrix power — is executed
per relation (t7), not just once, satisfying the discipline's intent
(every new consumer of a shared read mechanism gets its own executed
closed-form check) even though the specific fla-state-layout check does
not apply.

**Deploy record:** `ncr_opbank_task.py` md5 `cee5eae9ca725e31f4896f35f58ae4c1`,
`ncr_opbank_models.py` md5 `2ee5ea2d31af3a8f5525ab8d30680971`,
`ncr_opbank_selftest.py` md5 `294bffb978167b6b9c9507a9c8ec0f2a`,
`run_ncr_opbank.py` md5 `b973c8d4323b43f45da58682315ecbc6` — local and
`/home/nvidia/ncr/` on the box confirmed byte-identical via `md5sum`
before every real-CUDA smoke run.

**Independent opus build audit dispatched next (§8.4a), per "the
implementer does not review their own work" — mutation kill-proofs must
run to completion before Phase-0 launches on all 8 GPUs.**

### §7i K=12 SEED-EXTENSION READOUT (2026-07-11, 5/5 cells,
`K12EXT_DONE` 23:09:14Z): **pooled 10-seed K=12 AXIS A = SEP-PARTIAL
(median 0.8704, DEGRADED — moved UP within band from §7g's 0.753) →
CROSS-K OVERALL STAYS WIN-PARTIAL, now on n=10 — the §7h pre-registered
modal branch realized; NO K=8 number changed**

**Run record.** Harvest-code change + launcher audited CLEARED by an
independent opus agent BEFORE deploy (0 FATAL/MAJOR/MINOR, 1 NIT
no-fix; the auditor independently reproduced the selftest 5/5 and the
byte-identical regression replay against both archived verdicts, and
adversarially stress-tested the GPU-0/1 hard-refuse — commit 1f2f11f).
Box deploy md5-verified; per-arm micro test (ncr, K=12, seed 5, 30
steps) PASSED all §7f-pattern gates (a 4-arm test is not applicable —
ncr is the only arm run, per §7h). Launch tmux `ncr_k12ext` 22:42:11Z →
sentinel 23:09:14Z (wall ≈27 min); 5 cells, one per GPU on 2-6, GPU 7
idle reserve, GPUs 0-1 verified untouched before and after (nvidia-smi
recorded). All 24 result files scp'd + md5-verified against the box,
zero mismatches.

**Per-seed locked predictions vs outcomes (MA3 discipline: every lock
written by the pipeline BEFORE its cell's far-h eval; classes read from
the locks before any outcome was examined):**

| seed | locked δ | class (locked) | conservative horizon 0.451/δ | rec@0.9 @h\*=57 | held? | front |
|---|---|---|---|---|---|---|
| s5 | 0.0034 | PREDICTED-HOLD | 133.0 | 1.0000 | YES | 189 |
| s6 | 0.0058 | PREDICTED-HOLD | 77.3 | 0.9779 | YES | 93 |
| s7 | 0.0072 | PREDICTED-HOLD | 62.8 | 0.8253 | **no** | 57 |
| s8 | 0.0072 | PREDICTED-HOLD | 62.5 | 0.9155 | YES | 93 |
| s9 | 1.5283 | PREDICTED-FAIL | 0.3 | 0.0000 | no (dead) | 9 |

**Leg scoring (§7h's fixed procedure).** Extension-5: leg (i) every
PREDICTED-HOLD seed holds — **3/4, FAIL** (s7 the miss); leg (ii) no
PREDICTED-FAIL seed holds — **1/1, PASS** (s9 did not hold); leg (iii)
STRADDLE — **vacuous** (0 STRADDLE seeds drawn; MA3's own "≈2/3
STRADDLE" seed-mix expectation did NOT materialize — all four converged
fresh seeds locked δ ≤ 0.0079, i.e. PREDICTED-HOLD). Pooled-10: leg (i)
**5/6 FAIL** (s7 the only PREDICTED-HOLD miss across both waves); leg
(ii) **2/2 PASS** (s3+s9, both dead, neither held); leg (iii) **0/2
FAIL** (unchanged from §7g's straddle refutation).

**Instrument refinement (the honest sharpening of §7g's finding, not
re-argued into any leg).** §7g crowned the conservative all-modes bound
after it called 5/5 archived seeds exactly. The pooled record is now
**9/10**: s7 (conservative horizon 62.8 — hold predicted with only
~10% margin over h\*=57) measured 0.8253, front exactly AT 57. s8's
horizon (62.5) is statistically identical yet it held at 0.9155 — the
two straddle the bar from indistinguishable locked residuals. Refined
statement, replacing §7g's unconditional one: **the conservative
horizon reliably predicts hold-at-h\* when it clears h\* with margin**
(≥ ~35%: s1 103, s4 161, s5 133, s6 77 all called correctly, and the
0.3-36 dead/early seeds too) **and is a coin-flip within ~10% of the
boundary** (s7/s8, horizons 62.5-62.8, split). Recorded as
fine-structure.

**THE POOLED VERDICT (verdict of record — produced by the audited
committed `wave1_harvest.py --k 12 --expect-seeds 5 --expect-seeds-ncr
10` against the pooled directory = the 18 archived §7g cells + the 5
extension cells).** Pooled NCR values ascending: 0.000 (s3), 0.000
(s9), 0.149 (s0), 0.753 (s2), 0.825 (s7), 0.916 (s8), 0.978 (s6), 1.000
(s1), 1.000 (s4), 1.000 (s5); median = mean(5th, 6th) = **0.8704 →
DEGRADED** per the frozen §3.2a bands ("HOLD ≥ 0.9; DEGRADED ∈ (0.5,
0.9); FAIL ≤ 0.5"). Best baseline FAIL (fwm 0.2705, loopedvec 0.0 —
unrerun, as pre-registered). Novel-residue strata guard: no band drop
(pooled strata medians 52-55 HOLD 0.9025-0.9408, 56-59 DEGRADED
0.8324-0.8887; worst novel stratum = DEGRADED = the h\* band). **K=12
label: SEP-PARTIAL. Cross-K per the pinned §3.2a table: WIN (K=8, §7e,
untouched) + SEP-PARTIAL = WIN-PARTIAL — UNCHANGED, now resting on
n=10.** This is §7h's pre-registered modal branch verbatim ("pooled
median stays in (0.5, 0.9) → ... reported as a firmed-up, not moved,
verdict"); the guaranteed-pooled-HOLD branch needed ≥4/5 fresh holds
and got 3/5. Dead-seed rate now **2/10 at K=12** (s3, s9 — both
locked-classified PREDICTED-FAIL before far-h eval, both the archived
trainability-variance profile: in-dist 0.000, eff_rank(A) collapsed to
1.1-2.8 vs the converged seeds' exact 12.00, c\* sign-flipping/
incoherent), consistent with the K=16 2/5-stuck precedent class.

**Secondary readouts (labels unchanged, disclosed):** Axis C pooled:
8/10 seeds ≤0.05 through h≤125 (only s3 0.0955 and s9 0.0541 over —
both dead), 1/10 through 509 (s5, max dev 0.0128 — the best locked-curve
seed on record at K=12) → **TIE unchanged** under both the literal
≥3-seeds and ≥3/5-fraction readings. P2 re-confirms mechanically on the
pooled set (fwm 0.4125 < 0.5 at the pinned h=45; loopedvec 0.0). Pooled
fronts median 93, inside the pre-registered [87, 442]; zero post-front
revivals; zero reducer signatures anywhere. **The harvest's mechanical
`p1.pass_=True` field (5/10 holding) is DISCLOSED as NOT the scoring of
record here** — it applies the n=5-era ≥3-seeds bar to n=10; the §7h
leg-scoring above is the pre-registered procedure for this extension.
**Hygiene:** 0 shadow-divergent points; 17 agreement-divergent points
sit EXCLUSIVELY on the two dead seeds (s3 16 — §7g's finding, unchanged;
s9 1, at ladder h=93, with BOTH reads matching their own fp64 shadows —
MA5's pinned arbitration resolves it identically: an operator-degeneracy
property of a rank-collapsed state, not an instrument defect). Zero
flags on any converged seed at either K, now over 15 converged cells.

**GPU-h ledger.** Extension serial-sum **2.07** (per-cell 0.3955-0.4494)
/ device ≈ **2.25** (≈0.45 h wall × 5 GPUs) — under BOTH the §7h 8.4
ceiling and the 5.76 informed projection, because the projection
carried the §7g rate measured under 3-way co-location; solo cells run
≈2.8× faster. Program totals: ≈ **42.3 serial-sum / ≈21.2 device** of
the 120 cap (this extension is a new Phase-2-extension line item, not
charged to either closed 50-GPU-h wave sub-cap).

**STATUS: the NCR program verdict of record after the §7h/§7i
seed-extension is CROSS-K WIN-PARTIAL (Axis A: K=8 WIN n=5 + K=12
SEP-PARTIAL n=10), WIN (Axis B, K=8 bar), TIE (Axis C) — the §7g
verdict FIRMED, not moved. Two new standing facts: the K=12
trainability-variance rate (2/10 dead, both locked-classified) and the
boundary-regime conservative-horizon caveat (reliable with margin,
coin-flip inside ~10% of h\*). The wave-2/operator-bank go/no-go
remains with the coordinator; the operator bank stays separately
ledgered and double-gated (M4).**
