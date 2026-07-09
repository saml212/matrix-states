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
| Read | Direct matvec `o = Z^h q`, continuous cosine scoring — Task E's native read, NOT the Stage-1 `row_queries` attention reader; per M6, any arm whose read deviates from this gets the read-vector-std diagnostic (§1.30 P3 method; pass ≥ 0.05 across queries, degenerate reference 0.000e+00 vs healthy 0.41) as a pass/fail item in its calibration gate |
| Power computation | Binary exponentiation (⌈log₂h⌉ squarings + ≤⌈log₂h⌉ products) — this IS the headline; the naive O(h) loop is DISALLOWED as the claimed configuration and kept only as a disclosed cost-control arm (§3.3) and fp32 cross-check |
| Scale management | Per-squaring Frobenius renormalization of the running product, log-scale tracked separately — exactly cosine-invariant (cosine scoring cannot see a scalar), and REQUIRED: measured ρ(D) reaches 2.9 and c* reaches 2.8 (`TASK_E_FINDINGS.md` §9), so unmanaged fp32 squaring overflows (3.4e38) at h ≈ 83 — inside the capability window |
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
  h ∈ {49..60}, same convention.
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
below). Predictions of record, at the separation depth **h\* = 61
(K=8) / 57 (K=12)**:

- **P1 (NCR holds):** ≥3/5 NCR seeds at recovered_frac@0.9 ≥ 0.9 at
  h\* (s1/s2/s3 hold under even the conservative bound; s4 is the
  straddle case), with NCR's OWN failure front located on the ladder
  between h ≈ 87 and h ≈ 442 (median-seed band) — pre-registered as
  depth-robustness fine-structure, not hidden.
- **P2 (O(h) baselines fail):** each comparison-of-record O(h) arm
  falls below median recovered_frac@0.9 = 0.5 by h ≤ 32. Basis: fr=7
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
  drifts) and the outcome is scored TIE, not excused.

**Pre-registered outcomes — per axis, all publishable:**

| Axis | WIN | TIE | LOSE |
|---|---|---|---|
| **A — depth-robust exactness** | At h\*, NCR median ≥ 0.9 recovered_frac@0.9 AND every comparison-of-record O(h) arm median ≤ 0.5, consistent across both K and all non-identity residue strata | Both hold at h\* (separation collapses to cost-only) or both fail before h\* | NCR fails at h\* while any O(h) arm holds, or NCR's front lands earlier than a baseline's |
| **B — sequential cost** (claimable as capability ONLY if Axis A ≥ TIE, else it is the M4 "accounting convention") | Measured wall-clock fits log (NCR) vs linear (O(h) arms) and NCR is ≥10× faster at h=1021 | Scaling separation present but <10× at h=1021 | NCR not measurably sub-linear (would indicate a broken implementation, triggers diagnose-first) |
| **C — spectral predictability (instrument claim)** | \|predicted − measured mean_cos\| ≤ 0.05 at every ladder h ≤ 509 for ≥3/5 NCR seeds, predictions locked at calibration (precedent: ≤ 0.02 at h ≤ 21, worst case 0.067) | Prediction holds through h ≤ 125 only | Predictions fail inside the trusted window |

WIN = the capability separation (PI bar); TIE on A = an honest
efficiency/exactness paper (workshop-grade, per §2's pre-registered
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
| **NCR naive-loop** | SAME trained checkpoints as NCR, read = literal h-fold matvec loop | O(h) | Disclosed cost-control arm + fp32 cross-check (bin-exp vs loop must agree within fp32 tolerance at h ≤ 125); eval-only, zero training cost |
| **LoopedVec** | Param-matched iterated VECTOR map: trained step function applied h times to a vector state, linear decode, same episode input, same h signal (Looped Transformers arXiv:2409.15647; depth-recurrent arXiv:2603.21676 — their existence is exactly why "current architectures cannot do variable-depth single-pass composition" was retired as a claim, §2 M3) | O(h) | Comparison of record #1 |
| **FWM-read** | FWM-style recursive fast-weight read (Schlag, Munkhdalai & Schmidhuber, ICLR 2021, arXiv:2011.07831, cited AND distinguished: FWM fixes N_r=3, reads are LN-nonlinear, composition approximate): the SAME written Z, read via h-fold recursive one-hop LN reads — isolates "exact linear power vs recursive nonlinear reads" on an identical state | O(h) | Comparison of record #2; deviating read ⇒ read-vector-std diagnostic in its calibration gate (§3.1) |
| **C_MLP one-hot(h)** | Task E's inherited control | O(1) | DISCLOSED WEAK CONTROL ONLY — architecturally unable to extrapolate by its own §5 admission (one-hot(h) is out-of-vocabulary at held-out h); never the comparison of record; evaluated and labeled as such |

### 3.4 Corrected stopping/trust rule (M2)

**The rule (geometric form, scale-normalized).** With the §9 block
decomposition Z → (A, B, C, D) in the [E, E⊥] basis: leakage injected
at step j scales like ‖C‖·σ(A)^(j-1) and is then amplified by
D^(h-j); relative to signal scale σ_min(A)^h, the worst-case relative
contamination after h applications is

  T(h) = (‖C‖ / σ_min(A)) · (r^h − 1)/(r − 1),  r := ρ(D)/σ_min(A)

(linear limit T(h) = h·‖C‖/σ_min(A) as r → 1 — the old rule is the
degenerate case, which is exactly why it under-bounded at decisive
far depth). **Trust threshold τ = 0.2** (worst-case cosine floor
1/√(1+τ²) ≈ 0.98, above the 0.9 bar). **Stated plainly: this rule
bounds the numerical/leakage trust of computing Z^h; it does not and
cannot select h** (h is input-supplied, §3.1). **Disclosed
conservatism:** for the healthy §9 seeds, measured ρ(D) ≈ c* ≈
σ(A) (r ≈ 1.004), so the rule refuses trust beyond h ≈ 12 (s1:
T(12) ≈ 0.21) while measured behavior is exact through h=21 —
conservative by ≥30× (operator-norm slack on ‖C‖ plus phase
cancellation). The rule is therefore the A-PRIORI SCREEN (its role of
record is rejecting constructions like the negative test below and
dead seeds); the fp64 shadow (§3.1) is the empirical trust
instrument. Reporting labels per (cell, h): RULE-TRUSTED (T ≤ τ) /
SHADOW-VERIFIED (fp64 agreement) / UNTRUSTED — flagged, never
silently dropped.

**Mandatory EXECUTED negative test (pre-launch gate item, §3.8(c);
`CLAUDE.md` negative-tests-run-to-completion hard rule).** Construct
Z at d=16, K=8: A = 1.0·Π (exact cycle, σ_min(A)=1), ‖C‖ = 0.01,
B = 0, D = 1.5·(random 8×8 orthogonal) so ρ(D) = 1.5 > 1. The OLD
linear rule scores T_lin(21) = 0.21 — it MUST admit this Z under any
calibration that admits the healthy §9 seeds (s1's own T_lin(21) =
0.37 with measured-exact behavior). The corrected rule scores
T(21) = 0.01·(1.5²¹−1)/0.5 ≈ 99.7 → REJECT at τ = 0.2. The test's
empirical arm then measures actual behavior: leaked E⊥ junk at h=21
is ≈ 66× signal, cosine ≈ 0.015, recovery 0.0 — the old rule admits
garbage, the corrected rule rejects it, and BOTH halves must execute
to completion (numpy, CPU, output archived as
`matrix-thinking/chapter2/test_trust_rule_negative` results) before
launch, not merely be written.

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
  stratification in ALL reporting with the identity residue labeled
  and excluded from aggregates (§3.2); deep probe MANDATORY per cell:
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
negative test has EXECUTED to completion with archived output — plus
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

**STATUS (2026-07-09): Rev 1 COMPLETE.** All §2 findings addressed:
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
