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

## §3 Rev 1 (PENDING)

Dispatched 2026-07-09 overnight. Must address F1/F2/M1-M6/m1-m4 per the
binding disposition above; design-only (no GPU); wave 1 launch remains
gated behind Stage 2's calibration readout per M4.
