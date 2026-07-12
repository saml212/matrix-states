# Paper outline — Three Bounds on a Null

Page budgets from `brief.md`; venue limit 4.0 pages main content
(`venue-requirements.md`). Claims by id from the brief's evidence map.

## Section plan

| # | Section | Pages | Claims (ids) | Figures | Notes |
|---|---|---|---|---|---|
| 1 | Introduction | 0.75 | C1-phase1, C10-instrumentfix, C11-remetric, C12-nullvalidation, C5-power, C7-replication (headline versions) | — | keystone question; why bounded nulls; contribution bullets; instrument-fix falsifier |
| 2 | Setting and instruments | 0.65 | C0-ladder | — | ladder, grammar, three readouts, pre-registration/gates, correspondence-null pointer (C9-compute carried in Appendix B) |
| 3 | Bound 1: the geometric readout never fires | 1.05 | C10-instrumentfix, C11-remetric, C12-nullvalidation, C1-phase1, C2-rung3, C3-phase1b, C4-dissoc, C8-teeth | Fig 1, Fig 2, Table 1 | the instrument-fix case study (centerpiece); scale series; waves 3–4 disclosure; dissociation; enforcement arc |
| 4 | Bound 2: the behavioral contrast is power-bounded | 0.60 | C5-power, C6-transient | Fig 3 (left) | 3/4 UNRESOLVED at the registered floor; the transient, sign-disciplined (unchanged, round-1 verdict) |
| 5 | Bound 3: the replication gate | 0.55 | C7-replication | Fig 3 (right) | batch-effect flag; cohort CIs; what stands at what weight (unchanged, round-1 verdict) |
| 6 | Related work | 0.20 | — | — | Based, Zoology, Okpekpe–Orvieto, Nichani et al., by name |
| 7 | What is and is not bounded | 0.20 | C12-nullvalidation, C3-phase1b/C4-dissoc (scope), C5-power (restated scope) | — | scope, alternatives, small-scale lessons, one-paragraph close |
|   | **Total** | **4.00** | | | equals venue limit; Appendix A (unlimited) carries the full C10–C12 re-verification detail |

## Outline sanity checks

- [x] Page budgets sum to 4.0 (references/supplementary excluded per CFP).
- [x] Every claim id appears in exactly one primary section (headline
      restatements in §1/§7 cite the same rows; the primary carrier is
      marked above).
- [x] Every figure in the brief is placed (Fig 1, 2 → §3; Fig 3 → §4/§5
      split by panel; Table 1 → §3).
- [x] Related work distinguishes nearest neighbors by name (four named).
- [x] No section carries a claim with a missing or pending evidence row.

## Per-section beat sheet

### 1. Introduction
- Fixed-state models bind in-context associations in a matrix state; a
  prior program measured a write-geometry observable (span_frac ladder,
  frozen-key interventions) with no established behavioral correlate.
- The keystone question: does that geometry predict or causally improve
  in-context multi-hop composition?
- The deliverable is a null bounded three ways, each bound pre-registered
  and each naming its own falsifier — the form a trustworthy negative
  result should take.
- Contribution bullets (three, mirroring the brief).

### 2. Setting and instruments
- Checkpoint family: DeltaNet-family LMs, four rungs (14M→1.31B), two
  corpora, three intervention arms (control, per-token, global) matched on
  val loss. <C0-ladder>
- Task: K-cycle bind/query grammar; hop depth h ∈ {1..4}; load axis
  K/d_state near the located capacity operating point.
- Instrument 1 (geometric): pred(a,h) = S_T^h · q_eff, exact-recovery
  cosine ≥ 0.9, with pre-registered validity gates (h=1 sanity floor;
  premises iii/iv vs shuffled nulls) and a probe-invalid routing rule.
- Instrument 2 (behavioral): vocabulary-space L_query contrast (off − arm)
  with a pre-registered six-bucket trajectory classification.
- Discipline: every wave gate-first, verdicts routed mechanically
  (compute disclosure carried in Appendix B). <C9-compute>

### 3. Bound 1: the geometric readout never fires
- **The instrument-fix case study (centerpiece).** A pre-submission
  positive control on the production readout FAILS (0/256 at every h);
  root-caused to a [K,V]-vs-[V,K] state-layout transpose; fixed and
  independently audited; the fixed control PASSES 1.0000.
  <C10-instrumentfix>
- Re-running the 80-cell wave-1/wave-2 grid (320 readings) under the fix
  does NOT reproduce the pre-fix zero: 78/320 nonzero, h=1 up to 0.87.
  <C11-remetric>
- Two pre-registered correspondence nulls (label-shuffle, then a
  positional derangement null after the shuffle null is found vacuous at
  h=1 by construction) both reproduce the apparent signal at every one of
  the 320 readings (0/320 clears either); mechanism = collapsed
  prediction directions vs a near-collinear value population.
  <C12-nullvalidation>
- Claim-shape conclusion: recovery is null-indistinguishable, not zero;
  the alignment-premise gates fail at every reading regardless, before
  and after the fix.
- Phase 1 (zero-shot, marker template) + rung-3 scale completion: the
  exact grid re-verified above; premise gates 0/80 cells each, unaffected
  by the fix. <C1-phase1, C2-rung3> [Fig 1]
- Phase-1b (natural language, two template families): 16/16 zero under
  the pre-fix instrument; not independently re-run after the fix;
  probe-invalid routing rests on the premise gates alone. <C3-phase1b>
- Phase-2 (task-familiarized): 30/30 gate readings zero (pre-fix
  instrument, same disclosed limitation as Phase-1b) while L_query falls
  21.8–46.4% in 6/6 cells — vocabulary/geometry dissociation; the
  checkpoint family most likely to succeed still never clears a
  correspondence null. <C4-dissoc> [Fig 2, Table 1]
- The enforcement arc: Phase 1 computed but did not enforce its abort
  gate; every later chain refused launches mechanically. <C8-teeth>
- What this bound is: a doubly instrument-validated null (a defective
  readout AND two correspondence nulls both checked), never an unlicensed
  refutation of the underlying hypothesis.

### 4. Bound 2: the behavioral contrast is power-bounded
- The registered contrast and its n=3 power floor, derived before harvest:
  detectable |Δ| ≈ 1.5–1.7 loss units ≈ the whole familiarization effect.
  <C5-power>
- Result: 3/4 contrasts UNRESOLVED (the pre-registered most-likely
  outcome); reported as a measurement bound, not evidence of no effect.
- The single determinate signal: TRANSIENT, harm-direction,
  Δ(K=32,c=2500) = −0.500 CI [−0.624,−0.376], gone by c=5000; every
  determinate reading in the primary table is negative; the held-out-hop
  readout fires at the same cell, same direction. <C6-transient>
  [Fig 3 left]
- Sign discipline: why this cannot be read as "the intervention helps",
  and why TRANSIENT cannot be read as durable.

### 5. Bound 3: the replication gate
- The n=12 targeted extension: 9 fresh pretraining seeds, archived-value
  loader (the original 3 never re-scored), batch-effect pre-pooling gate
  with a pinned variance-ratio cutoff — all registered before launch.
- Result: the gate flags the anchor (var ratio 4.47 > 4.0); old cohort CI
  [−0.624,−0.376] vs new cohort CI [−0.506,+0.357]; the naive pool
  (diagnostic only) spans zero. Outcome: BATCH-EFFECT-FLAGGED.
  <C7-replication> [Fig 3 right]
- The honest reading: the transient stands only at its original n=3
  weight; the extension neither confirms nor refutes it; the gate did the
  job it was registered to do.

### 6. Related work
- Based (capacity/ladder methodology precedent — different claim type).
- Zoology (recall gaps — trained-for-probe vs post-hoc general checkpoints).
- Okpekpe & Orvieto (positive causal account — we bound one mechanism).
- Nichani, Lee & Bietti (readout-design provenance: exact recovery, not
  argmax).

### 7. What is and is not bounded
- Bounded: the specific geometric readout construct (three ways, now
  including a correspondence-null-cleared re-verification of the largest
  sub-grid), the causal contrast above the n=3 floor, the transient's
  durability and its n=12 poolability. <C12-nullvalidation>
- Not bounded: the keystone hypothesis itself at effect sizes below the
  floor; mechanisms the S_T^h readout cannot see (cross-layer hand-off);
  other checkpoint families; waves 3–4's raw zero readings, which were
  never independently re-verified under the fixed instrument (disclosed,
  not assumed corrected by analogy). <C3-phase1b, C4-dissoc>
- The real-kernel positive control that the pre-revision draft flagged as
  "the next wave's priority" has now run; update this sentence to report
  what it found rather than that it is pending.
- Lessons for small-scale practice: gates need enforcing code paths;
  registered routing rules prevent verdict inflation; replication gates
  can refuse — and that refusal is a result; a positive control that
  fails pre-submission is itself a result worth reporting, not just a
  blocker to clear.
- One-paragraph close.
