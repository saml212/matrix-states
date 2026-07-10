# Paper outline — measurement-ws ("the instrument is the first suspect")

Stage 2 of the `paper` skill (repo mode). Page budgets from the brief;
claims by id from the brief's evidence map.

## Section plan

| # | Section (file) | Pages | Claims (ids) | Figures | Notes |
|---|---|---|---|---|---|
| 1 | Introduction (`01_intro.tex`) | 0.60 | (previews I1, W1, X2) | — | the question, the program, the six incidents, contributions |
| 2 | The adjudication discipline (`02_discipline.tex`) | 0.50 | N1 | — | five rules, each anchored to the incident that taught it |
| 3 | Case I: the tolerance that manufactured a cliff (`03_case_tolerance.tex`) | 0.55 | I1, I2, I3, I4 | — | phenomenon → wrong first diagnosis → repro instrument → unlock |
| 4 | Case II: the instrument at the wrong layer (`04_case_wronglayer.tex`) | 0.65 | W1, W2, W3, W4, W5 | fig2 | dissociation → refits refuted → causal localization → nonlinear storage |
| 5 | Case III: the gauge that did not transfer (`05_case_gauge.tex`) | 0.80 | X1, X2, X3, X4, X5 | fig1 | contradiction → four-ground tiebreak → teeth → re-metric |
| 6 | Three more lenses, briefly (`06_three_lenses.tex`) | 0.35 | B1, B2, B3 | — | covariance / rank tax / transposed convention, one paragraph each |
| 7 | What the discipline did not flip (`07_not_flipped.tex`) | 0.20 | X5, I5 | — | FALSIFY stands; frontier/ceiling findings withheld or non-replicating |
| 8 | Related work and limitations (`08_related_limits.tex`) | 0.35 | I6 | — | by-name distinctions; single-program scope; no counterfactual rate |
|   | Abstract (`00_abstract.tex`) | — | — | — | 200–230 words |
|   | Appendix (`09_appendix.tex`) | n/a | X2, W4, I3 detail | Table A1–A3 | full 62-cell lens table (or its per-group condensation), tap-variant table, decision-rule walk |
|   | **Total (body)** | **4.00** | | | = assumed venue limit (flagged in venue-requirements.md) |

## Outline sanity checks

- [x] Page budgets sum to 4.00 (excl. references and appendix).
- [x] Every claim id in the brief's map appears in at least one section,
      and each id has exactly one PRIMARY carrier section (I1–I4 → §3;
      I5 → §7 with a §3 mention; I6 → §8; W1–W5 → §4; X1–X4 → §5; X5 →
      §5+§7 split by half: mechanics in §5, verdict-boundary in §7;
      B1–B3 → §6; N1 → §2).
- [x] Both figures placed (fig1 → §5, fig2 → §4).
- [x] Related work distinguishes Silberzahn, D'Amour, Henderson/
      Bouthillier, Kapoor (REFORMS), Nichani by name (brief § nearest).
- [x] No section carries a claim whose evidence row is missing — all 20
      rows verified against raws this session (md5s in the brief).

## Per-section beat sheet

### 1. Introduction
- Empirical claims are only as good as the instrument; a program that
  measures aggressively will break lenses more often than models.
- One program, five days, six incidents; every apparent model failure
  that was adjudicated traced to the instrument; two genuine model
  failures survived.
- The discipline in one sentence: adjudicate instrument health BEFORE
  reading any model verdict, with pre-registered crosschecks and
  falsifier teeth.
- Contribution bullets (catalogue, discipline, honest boundary).

### 2. The adjudication discipline
- Rule 1: instrument-health adjudication precedes model verdicts (gate
  routes, convergence profile, contradiction scan) — anchored to Case III
  where it was coordinator-mandated before any reading.
- Rule 2: pre-register a crosscheck that removes the primary's strongest
  assumption; grade the crosscheck wherever that assumption provably
  fails (the oracle-injection precedent).
- Rule 3: teeth — negative controls run to completion, mutation-tested
  guards, planted-signal positive controls on every probe rung (W5's
  lesson), and a pre-registered falsifier for the adjudication itself.
- Rule 4: contradiction between two reads is resolved by reading the raw
  artifacts, never by recency or averaging.
- Rule 5: argmax closure (N1) — exact continuous recovery wherever the
  claim needs it; discrete/continuous gaps pre-registered.

### 3. Case I: the tolerance that manufactured a cliff
- The phenomenon: 11 of 12 new cells quarantined at d=96, training clean
  (I1) — a headline negative result if reported as measured.
- The wrong first diagnosis, refuted with margins (I2): residuals 7,800×
  below tolerance; the failing pool is architecturally anchor-bypassed.
- The repro instrument (I3): four audit rounds, pre-registered decision
  rule, the walk, 4,608/4,608 resolve by n_iter=28 — near-miss, not wall.
- The unlock (I4): flag-only recalibration, zero GPU, no cliff through
  K/d=0.9375.

### 4. Case II: the instrument at the wrong layer
- The dissociation (W1): the task metric says near-perfect recall; the
  probe says zero. THIS is the finding, not the 0.0.
- Kill the easy outs (W2): closed-form ridge = the linear optimum, still
  0.0; probe output is the membership oracle.
- The answer is in the network (W3): full forward decodes at 0.9957,
  arm-ordered.
- Causal localization (W4, fig2): the tapped layer is causally inert;
  the load-bearing state's own linear tap ALSO fails; only the
  post-nonlinearity hidden exposes it — storage is nonlinear, the model's
  own forward is the only decoder found.
- The second defect found in passing (W5) → positive-control rule.

### 5. Case III: the gauge that did not transfer
- Setup: two pre-registered lenses on one sweep; primary (scale-only
  degauge under Q̂≈I) vs crosscheck (fit/eval-split full-Q Procrustes).
- The signature (X1, X2, fig1): mechanical FALSIFY; lenses agree on all
  non-converged cells, contradict 0-vs-1.0 exactly on converged ones.
- The tiebreak (X3): four grounds, none "the newer claim wins"; the
  gauge was measured on another architecture — no entitlement transfers.
- Teeth (X4): the tiebreak's own pre-registered falsifier — shuffled
  targets read 0.00/0.00/0.05, real reads reproduce committed values
  bit-identically.
- The re-metric (X5): harness reproduces the committed verdict first;
  crosscheck-lens endpoint STILL FALSIFY, different reason; what each
  lens's reason would have meant.

### 6. Three more lenses, briefly
- B1: the covariance that could not see (uncentered SVD isotropic on
  orthogonal targets; perfect-model injection fails production bars).
- B2: the target that taxed every budget (ρ⊕I; cells at their exact
  rank-k ceiling √(k/d_state); causal test undelivered, not failed).
- B3: the reference on the wrong side (closed-form hand computation;
  √2 signature; mutant kill; composer exonerated, cross-check fixed).

### 7. What the discipline did not flip
- The endpoint verdict stands under the corrected lens (X5): A6's
  decisive config never converges (lens-independent); S5's outlier seed
  is a genuine trainability failure.
- Case I's own follow-ups stayed withheld or non-replicating (I5): the
  12×-noise pool-margin mechanism refused promotion; the exact ceiling
  did not replicate.
- The point: the discipline surfaces signal AND refuses unearned claims.

### 8. Related work and limitations
- By-name distinctions (brief § nearest prior work).
- Limitations: one program, one team; no counterfactual rate of what a
  lighter process would miss; incident count too small for prevalence
  claims; economics stated (I6) but not controlled.
