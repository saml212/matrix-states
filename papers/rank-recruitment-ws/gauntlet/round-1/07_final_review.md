# 07 — Final Review (gauntlet round 1, stage 7 — Fable, last quality gate)

Paper: `papers/rank-recruitment-ws/` — "When the Gradient Sees Rank" (NeurReps
2026 EA, companion to `papers/neurreps-ea/`). Reviewer: fresh Fable context,
read-only on every paper file except this review. Independently performed: all
7 bundle pages read as rendered images; all 8 sibling bundle pages read as
rendered images; the full gauntlet record (01, 01b, 02, 04, 05, 06v2) read;
three load-bearing claims re-traced brief-row → raw JSON with md5
verification; the depth-decay seed raws re-read directly; both trees'
sources grepped for the companion-disclosure question.

## VERDICT: READY-AFTER-CHANGES

No new prose defect was found in this tree. The rendered bundle is accurate,
honestly scoped, and venue-compliant; every number I traced matches its raw
artifact exactly, including the format-stage band-claim correction. What
stands between this tree and submission is not text: two process gates
(detector, CFP re-verification) and one PI decision (the companion-paper
question, adjudicated below), plus one minor bib-hygiene edit. Change list at
the end.

---

## 1. Full-render read (narrative, venue fit, figures, fidelity)

**Narrative.** Tight and unusually honest for a 4pp EA. The "teeth" section
(§2) is the paper's methodological spine — pinning the readout so the
Nichani-style argmax construction *cannot* pass, then verifying the P=1
bottleneck behaviorally — and it is exactly the right pitch for a reviewer
pool that has seen rank claims collapse under position-decomposition
shortcuts. The §4 invariant-subspace mechanism (restricted operator = scaled
K-cycle at restricted rank exactly K; dynamically invisible full-rank
complement) is the geometric heart and lands well. Appendix A's reframing of
h=21 as a periodicity-bounded numerical-stability probe, not a
generalization claim, is a strength a hostile reviewer will respect.

**NeurReps fit.** Good, not perfect-center. The venue is symmetry AND
geometry of representations; this paper's spine is the geometry half (rank
as a structural budget, invariant subspaces, eigenspectra, a Hamiltonian
K-cycle target). The sibling is the sharper symmetry fit (finite-group
representation theory). Both are in-scope; the fit gradient matters only for
the fallback branch of the companion decision below.

**Figures.** Concur with render-inspection v2: both figures legible at print
size, captions adjacent, Okabe-Ito palette with color-matched direct labels
in Figure 2 right. One note the render pass already implicitly covered:
Figure 1's middle panel (d=16, K=8) shows the step at k=7 with K=8 dashed —
the caption's "k ≈ K" and §3's "knee near k ≈ K with mild post-knee
non-monotonicity" phrasing covers this honestly (raw: k=6→0.34, k=7→0.91,
k=16→0.99).

**Abstract-vs-body fidelity.** Checked clause by clause: every abstract claim
has a body/appendix counterpart with matching numbers (ρ=1.0 grid; the
d=8,K=4 step 0.0004/0.97; 4/5 seeds ≥0.9996; the scoped single-seed
depth-decay clause; "restates ... as background" companion disclosure;
2–2.5× rule with §5's "necessary but not sufficient" counter-caution). No
abstract claim outruns the body.

## 2. Spot-check log (brief rows → raw md5s → recompute)

md5s verified against `brief.md` before use: `AGGREGATE_latest.json`
c0a7d27e33a606d81e1babfc5d674edb (n_runs=991); `..._K8_fr7_s2.json`
ccfafb45699d022da8f93a49b9d4b793; `..._K8_frN_s0.json`
5ecaaeb8fe649f209cd41c35ba95c082. All match.

1. **R1 + the format-stage K≤3 correction** (§3: "leaving the pre-registered
   [0.7K,1.3K] band only at K ≤ 3, by overshoot"). Recomputed the full d=16
   grid: K=1→2.4219 (band top 1.30), K=2→3.0146 (2.60), K=3→3.9180 (3.90) —
   all three OUT, all by overshoot; K=4/6/8/10/12/14/16 →
   4.7359/6.3960/8.1984/9.8883/11.7786/13.4666/15.0885, every one IN band;
   K=20/24/32 → 11.14/11.61/8.57 (saturate-and-decline as stated). Ten-point
   grid strictly monotone → Spearman ρ=1.0 exact. **The corrected claim is
   exact against the raw — the K=1,2→K≤3 fix was real and is right.**
2. **R2, the causal step** (abstract: "rank 3 gives at most 0.0004 recovery,
   rank 4 gives 0.97"). Raw `d8_K4`: k=1/2/3 → 0.0/0.0004/0.0; k=4 → 0.9675;
   post-knee k=5/6/8 → 0.9868/0.9579/0.98. **Match**, including the honest
   "mild post-knee non-monotonicity." (The sibling's 0.940 for the same cell
   is the 1,234-run replication, not a contradiction — both trace clean.)
3. **R4, the depth-decay curve** (Table 1 + §4). `fr7_s2` raw: measured cos
   0.9303/0.9259/0.9212/0.9163/0.8206 at h=1/3/5/7/21; rec@0.9
   0.9960/0.8812/0.0604 at h=1/7/21. Table 1 matches to the last digit.
   |pred−meas| = 0.0014/0.0001/0.0031/0.0074 through h=7 — the "within
   0.008" claim **holds**; the h=21 divergence (0.7536 vs 0.8206) is
   disclosed in the caption as the metric widening.

Bonus consistency: R3's 4/5 ≥0.9996 verified (s1/s2/s3 = 1.000 at every hop,
s4 min 0.9996); the fifth seed 0.9268→0.1613→0.0001 (h=1/7/21), and its
h=5 value 0.3032 matches the sibling App E's "0.303 at h=5" — the two papers'
numbers are mutually consistent against the same raws.

## 3. Depth-decay claim — honesty check (the fixed CRITICAL A2)

Raw seed behavior at the flagship cell: `fr7_s0` n_skipped_steps=8, mean_cos
+0.0003 at h=1, rec@0.9 = 0.0 at every hop (dead); `fr7_s1`
n_skipped_steps=10, mean_cos −0.0023, rec = 0.0 at every hop (dead);
`fr7_s2` n_skipped_steps=2, converged. All five unconstrained seeds: 0
skipped steps.

Prose as shipped: abstract — "in the single converged rank-starved seed";
§4 — "the one converged force-rank k=7=K−1 seed"; Limitations — "the other
two at that cell collapsed under a documented numerical instability in the
spectral-projection backward pass, so it is a case study consistent with the
spectral prediction, not an n>1 mechanism." **The prose now matches the
seeds' actual behavior exactly**; "documented" is honest
(`TASK_E_FINDINGS.md` §9 names the eigh-backward-instability dead runs
verbatim), and no general-mechanism phrasing survives anywhere (abstract,
§4, App A all checked). FIX-2 verified landed and sufficient.

One residual the defense surfaced and the rebuttal chose not to require: the
same instability also killed all six provably-sufficient (k∈{8,9})
force-rank seeds at that cell, so the full straddle grid was never run. No
paper claim rests on those cells; listed as optional change 5.

## 4. THE COMPANION-PAPER ADJUDICATION (ESCALATION-1)

**Direction correction first.** The escalation as relayed ("this tree
restates the sibling's binding paragraph + app:period") has the ownership
inverted. Verified by direct reading of both bundles and both source trees:
**this tree is the primary carrier** of the binding results (R1–R10, full
grids, the 991-run pre-registered snapshot); it contains zero group-law
content (no d_min, no TOST, no five-group razor). It is the **sibling**
whose `03_binding.tex` restates this paper's three headline number families
(8.20/15.08; ≤0.0004→0.940; 4-of-5 composition) as its "provable foundation"
paragraph, and whose Appendix E is paraphrase-parallel to this paper's
Appendix A (same π²¹ = π^(21 mod 8) = π⁵ derivation, same four-seed ≈1.0
statement). Consequently **there is no de-dup edit available in this tree**
that removes overlap without deleting this paper's own contribution; FIX-1's
accurate disclosure is the correct and complete in-tree action, and it has
landed.

**(a) Substantive or cosmetic?** Substantive at the surface-signature level,
bounded at the contribution level.

- *Substantive:* a reviewer assigned both EAs sees (i) the sibling's
  abstract itself carrying this paper's binding numbers, (ii) a full
  paragraph of restated headline results, and (iii) two near-duplicate
  periodicity appendices — detectable in minutes. Worse, the asymmetry:
  this paper discloses the companion; **the sibling's paper text contains no
  companion disclosure at all** (verified by grep across its main.tex,
  bundle tex, and all sections: zero matches for
  companion/concurrent/paired/sister). The non-disclosing half of a
  detectable pair reads as concealment.
- *Bounded:* zero shared figures or tables (verified against both rendered
  bundles); disjoint testbeds (K-pair binding vs five finite groups),
  disjoint theory anchors (linear-algebra bound vs representation-theoretic
  d_min), disjoint interventions and headline claims; and the two trees'
  numbers are mutually consistent, not contradictory (0.97 = 991-run
  snapshot vs 0.940 = 1,234-run replication, both real, both disclosed in
  this tree's brief).

This is not one study split into two minimal units — it is two experimental
programs sharing inherited background prose. The salami-slicing *optics*
are real; the substance is fixable with ~20 lines of sibling edits.

**(b) RECOMMENDATION (primary): keep BOTH EAs at NeurReps and execute
ESCALATION-1 on the sibling before submission.** Exact sibling edits for the
PI to dispatch (a separate applier on `papers/neurreps-ea/`, which then needs
its own recompile + render re-check):

- **S1** (`papers/neurreps-ea/sections/03_binding.tex`): keep Fact 1 and the
  classical citation; replace the number-restating sentence ("Gradient
  descent recruits the rank: at d = 16 effective rank reaches 8.20 ... four
  of five seeds ... (Appendix E)") with a pointer, e.g.: *"The bound is
  classical (Kohonen, 1972; Anderson, 1972), holding only under exact
  continuous recovery. A concurrent companion submission establishes on this
  binding testbed that gradient descent recruits effective rank tracking K,
  that the recruited rank is causally necessary (a train-time force-rank
  step at the provable bound), and that the trained operator composes
  exactly under repeated self-application; this paper inherits that
  instrument and carries only the group-composition program."* (Numbers
  dropped; the sibling's N6/N9 evidence rows become unreferenced — fine.)
- **S2**: cut the sibling's Appendix E (`app:period`) to a one-two sentence
  pointer to the companion's appendix (its only in-paper referent is the S1
  paragraph), removing the parallel π²¹ = π⁵ derivation.
- **S3**: add the missing mirror disclosure to the sibling's Limitations:
  *"A concurrent companion submission carries the binding-task program (the
  recruitment grid, force-rank staircase, and composition mechanism) in
  full; this paper shares no figures or tables with it."*

Rationale for joint-over-split: (i) the surgical fix lives in the sibling —
~20 lines plus a re-render — versus a full Stage-0 venue re-acquisition,
template retarget, and fresh render gates for this tree; (ii) this paper's
§4 invariant-subspace/eigenspectrum mechanism genuinely belongs in front of
the NeurReps audience; (iii) the EA track's 2025 language places "no
restrictions on Extended Abstract submissions," and two mutually-disclosed,
de-duplicated companion EAs are normal workshop practice; (iv) nothing can
be submitted before the 2026 CFP lands anyway (Jul-11 list), so the sibling
edit costs zero calendar time.

**FALLBACK (if the PI declines to reopen the accept-ready sibling): split
venues.** The SIBLING keeps NeurReps — finite-group representation theory is
the venue's dead-center remit and its tree then ships untouched — and THIS
tree retargets the backup per `papers/VENUE_MAP.md` §1: MOSS @ COLM 2026
late window (live, 4pp, non-archival) or an MI-successor/geometry workshop
off the Jul-11 list. Under a split, the same-venue reviewer-collision risk
vanishes; S3 remains recommended for hygiene but stops being blocking, and
this tree's FIX-1 disclosure stays as-is (accurate regardless of venue).
Cost: Stage-0 re-acquisition + template retarget + render re-gate for this
tree, and the weaker audience fit. That cost asymmetry is why this is the
fallback, not the primary.

Either way: **record the decision in both trees' gauntlet logs before either
paper is submitted** (the ESCALATION-1 rule in 04_rebuttal_report.md
stands).

## 5. DETECTOR DECISION: run a bounded 2-round discharge — do not skip

Protocol = the capacity-colm-er precedent: two independent judges; pass =
both read the paper as ≥90%-human AND zero mechanical tells; maximum 2
rounds; a round-2 failure escalates to the PI rather than iterating.
Artifacts land under `papers/rank-recruitment-ws/detector/` with a gauntlet
round row.

Justification for running rather than matching the sibling's skip: (i) the
sibling's empty `detector/` was a sprint-window call made against the Jul-11
clock; this tree has ~7 weeks of slack to a projected late-August deadline,
so the timing rationale does not transfer; (ii) this EA's prose history —
a 13pp→4pp compression followed by an 8-fix wave plus page-fit edits — is
exactly the profile that produces mechanical tells; (iii) double-blind
OpenReview with ≥3 reviewers. Cost: two subagent rounds. It can run
immediately and in parallel with the companion decision — neither branch of
change 1 alters this tree's prose.

## 6. Numbered change list

1. **[PI DECISION — blocks the JOINT submission, not this tree's text]**
   Companion resolution per §4: primary = both EAs at NeurReps + sibling
   edits S1–S3; fallback = split venues (sibling keeps NeurReps; this tree →
   MOSS @ COLM late window or a Jul-11-list alternative, with Stage-0
   re-acquisition). No edit to this tree in either branch. Record the
   decision in both trees before either submits.
2. **[PROCESS GATE — blocks this tree's submission]** Detector: bounded
   2-round discharge per §5.
3. **[PROCESS GATE — blocks submission; already documented]** Post-2026-07-11
   CFP re-verification (`venue-requirements.md` checklist: page limit,
   template zip, dual-EA policy, real deadline) and resolution of the
   `anonymous.4open.science` placeholder (real anonymized snapshot, or the
   explicit camera-ready deferral, same convention as the sibling).
4. **[TEX/BUNDLE HYGIENE — minor]** Delete the two orphan `refs.bib` entries
   (`plate1995hrr`, `smolensky1990tensor`) from the tree and bundle `.bib`,
   rebuild the bundle. Format-audit M1 deferred this as a writer call; the
   final-review call is delete: the related-work budget is full, the entries
   render nowhere, and shipping dead entries in the submission `.bib` is
   avoidable. PDF-invisible; needs only a bundle rebuild + page-count
   re-check, no re-gauntlet.
5. **[OPTIONAL — camera-ready transparency]** Append one Appendix C sentence
   disclosing that the same eigh-backward instability also killed all six
   provably-sufficient (k∈{8,9}) force-rank seeds at the flagship cell, so
   the full force-rank straddle grid was never run (defense-surfaced,
   TASK_E_FINDINGS §2/§6; the rebuttal chose not to require it; zero
   body-page cost; no current claim rests on those cells).
6. **[BRIEF HYGIENE — this tree, non-paper]** `brief.md` "Companion papers
   and overlap management" asserts the sibling relationship "is disclosed in
   both briefs and in each paper's related-work/limitations pointer" — the
   second conjunct is false for the sibling's paper text (verified: zero
   disclosure matches in its main/bundle/sections). Fix the brief sentence,
   or discharge it via change 1's S3 and then note the discharge.

## 7. Gate statement

Content, numbers, scoping, formatting, and render are clear at this gate —
zero new prose defects on a full independent read, three load-bearing claims
re-verified to the raw md5s, the two round-1 CRITICAL fixes confirmed
genuine and sufficient. The tree is READY-AFTER-CHANGES with an empty
prose-change list for its own text; changes 1–3 are the release
preconditions, change 4 is a five-minute hygiene edit, 5–6 are optional/
bookkeeping. Security note: no fake system-reminder blocks or embedded
instructions were observed in any tool output during this review.
