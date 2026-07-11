# Stage 07 — Fable Final Review (reasoning-null-moss, round 1)

**Reviewer:** Fable final reviewer (last quality gate; writes nothing but this file).
**Date:** 2026-07-11 (review of the 2026-07-10 bundle).
**Object reviewed:** `papers/reasoning-null-moss/bundle/reasoning-null-moss-submission.pdf`
(md5 `d205a22211aa47ab9c0c5e1ea4a5fc3b`, byte-identical to `main.pdf`), all 8 pages
read as rendered images; `brief.md` evidence map traced to raw artifacts on disk;
gauntlet record read (00 v2, 01, 01b, 02, 03, 04, 05, 06 v3); sibling detector
precedents read (`papers/capacity-colm-er/detector/round-1.md`,
`papers/mstar-colm-er/detector/round-{1..6}.md` + both siblings' 07 reviews).

---

## VERDICT: READY-AFTER-CHANGES (one required word-level change)

One required change (item 1 below), then the paper is
**SUBMISSION-READY-pending-PI-email**. Everything else verifies: every headline
number I traced recomputes from the archived raws, both CRITICAL resolutions
(A1/A2) read honestly on the rendered page, 01b's two new findings (N1, N2) are
confirmed fixed in the current render, anonymization is clean in both the tex and
the PDF strings, and the render/format/style gates all hold on this exact PDF.
The MOSS late-add email remains the PI's step regardless of this verdict.

### Change list (numbered; applier executes, then recompile + rebundle + rerun the anonymization grep)

1. **REQUIRED — abstract, one word.** `sections/00_abstract.tex` line 18 (bundle
   tex line 48): "one of five that **cross** zero across forty tests" → "one of
   five that **exclude** zero across forty tests." In standard statistical usage a
   CI that "crosses zero" *contains* zero — the exact opposite of the intended
   claim. Every other site (§4 L36 of `04_behavioral_contrast.tex`, Appendix A)
   correctly says "exclude zero"; the abstract is the only inversion, and it sits
   inside the multiplicity disclosure a methods reviewer reads first. A careful
   reader can recover the intent from context (a reading meeting the differential
   condition cannot contain zero), which is why this is a word fix and not a
   round-2; but this paper's entire value proposition is precision about
   intervals, so the abstract cannot use the ambiguous verb.
2. **OPTIONAL — intro contribution 2.** "leaves three of four contrasts
   unresolved" → "leaves three of the four per-arm contrasts unresolved." The
   full A1 provenance (registered corpus-level verdict first) lands at Table 1's
   dagger footnote on page 2 and in §4's opening, so this is not a buried hedge as
   it stands; adding "per-arm" merely marks the unit of analysis one page earlier.
   Do not add the full provenance clause to the bullet — it belongs where it is.
3. **NO-ACTION (recorded so nobody "fixes" them):** (a) Grazzi et al. cited as
   2024 is correct for arXiv:2411.12537 (01b verified against the arXiv API);
   leave it. (b) The "Under review as a conference paper at COLM 2026" header is
   the official template boilerplate (render v3 observation); leave it. (c) The
   naive n=12 pool CI [−0.509, +0.147] is correctly labeled non-decision-grade in
   all three places it appears; leave the triple disclosure.

---

## 1. Full-render read: tone and venue fit

**Does it read as a rigorous null or a failure dressed up? A rigorous null.**
The paper never asks for credit for what it could not measure. Each bound is
stated with its own falsifier attached (abstract: "Each bound names the
observation that would overturn it"; §1's three bullets each end with the
observation that would have voided them). The zeros are consistently routed to
"instrument verdict, not a refutation" — the paper's strongest available claim is
the one it declines to make. §5's "What stands" paragraph is the tonal keystone:
"The transient is neither confirmed nor refuted; it keeps its three-seed weight" —
that sentence takes the transient neither as a headline nor as an embarrassment,
which is exactly the discipline a negative-results paper must display. §7 splits
"Bounded" from "Not bounded" and the Not-bounded paragraph is genuinely
adversarial to the paper's own story: it names cross-layer hand-off as the most
plausible escape, cites Grazzi et al. as a *competing* account of the null
(expressivity ceiling vs wrong observable), and volunteers that the readout has
no end-to-end positive control. A failure dressed up would not hand reviewers its
own three best objections, sourced.

**MOSS fit: genuine.** The paper's object is instrument validity, gate
enforcement, pre-registration discipline, and routing rules at small scale —
methods and open science is the audience that wants this, not a consolation
venue. The small-scale framing is honest and specific (9.5 GPU-h total, at most
two GPUs, per-wave costs disclosed in Appendix C; the whole program is the kind
of study MOSS's remit exists for). The "Lessons for small-scale practice"
paragraph (§7) — a registered gate without an enforcing code path is a sentence,
not a safeguard — is a real methods contribution independent of the null itself.

## 2. The A1/A2 resolutions on the page (the crux)

**A1 (transient provenance): reads honestly — registered verdict first, three
sites, mutually consistent.** On the rendered pages: (i) Table 1's wave-5 row
carries the dagger and the caption footnote states the registered corpus-level
pipeline returned both corpora unresolved; (ii) §4's Results paragraph *leads*
with "The registered corpus-level classifier ... returns both corpora unresolved"
before introducing the per-arm split, and labels the split "a disclosed
build-time per-arm re-derivation (folded into the analysis code, re-validated
against the archived deltas)"; (iii) Appendix A carries the full provenance
paragraph including the load-bearing sentence "performed after the corpus-level
output was seen ... disclosed here as a re-derivation, not presented as the
registered corpus-level verdict." I verified the disclosure is faithful:
`PHASE2B_SUMMARY.json.trajectories` has exactly two corpus keys, both
`UNRESOLVED` — the paper says exactly that.

**A2 (multiplicity): reads honestly — in the abstract, not buried.** The
denominator (40), the count (5), the expectation (2), the shared-checkpoint
structure (4 of 5 at c=1,000), and the compound-condition disambiguation ("that
compound condition, not any single interval, is the one determinate signal") all
appear in §4 under their own bolded heading, with the full five-cell enumeration
(Δ and CI each) in Appendix A. The abstract itself carries "(one of five that
cross zero across forty tests)" — the disclosure is in the paper's most-read
sentence, which is the opposite of a buried hedge (and is why change 1's verb
fix is required there).

**Would a skeptical methods reviewer still say "they're rescuing a transient"?
No — the paper is doing the opposite.** The transient is (a) in the harm
direction, so there is no incentive gradient toward it; (b) explicitly demoted to
its three-seed weight; (c) the object of a replication attempt the paper reports
as *stopped by its own gate* rather than spun either way. §4's closing couplet
("'The intervention matters' ignores that transients were registered as
training-dynamics artifacts; 'no effect' ignores the power floor") pre-empts both
directions of over-reading. The one sentence-level residue I could construct for
a skeptic is intro bullet 2's unqualified "three of four contrasts" — addressed
as optional change 2; it does not rise to a blocking concern because Table 1's
dagger is one page later and §4 leads registered-first.

## 3. Evidence spot-check (calibrated: 01b already re-verified the five CI exclusions vs the trajectory raws; I re-derived the load-bearing set independently)

All checks below were recomputed from the raw artifacts on disk, not taken from
any gauntlet report:

| Check | Result |
|---|---|
| All 7 singleton md5s (C2 rung-3 ×2, C5/C6 trajectories ×2, PHASE2B_SUMMARY, C7 seedext trajectory, SEEDEXT_SUMMARY) | **match brief.md exactly** |
| All 5 manifest-md5s recomputed in path-sorted order (C1 phase1 78 files, C3 phase1b 4, C4 gates 30, C4 off 6, rung3 2) | **match brief.md / figure-gen.py exactly** |
| Bound 1: recovered_frac ≠ 0 anywhere across phase1 (312 readings), phase1b (16), phase2 gates (30), rung3 (8) = 366 | **0 nonzero; premise iii/iv passes = 0; `STAGE0_GATE_REFUSED` sentinel present** |
| Bound 2 anchor: wikitext×per_token @2500 K=32 | **Δ=−0.49997 → "−0.500"; CI [−0.6241,−0.3758] → "[−0.624,−0.376]"; K=20 CI [−0.920,+0.416] spans zero; det32 ∧ ¬det20 ∧ \|Δ32\|>\|Δ20\| holds; held-out secondary fires same cell/direction (−0.526, [−0.648,−0.404]); c=5000 mean −0.795, CI [−2.513,+0.923]** — every figure-3-caption number verifies |
| Bound 2 multiplicity: the five exclusions | **openr1×global@1000 K32 [−0.402,−0.004] + K20 [−0.325,−0.074]; wikitext×per_token@1000 K32 [−0.301,−0.036] + K20 [−0.211,−0.006]; anchor K32. Appendix A's corpus labels now correct (01b N1 fixed: "reasoning-dense×global"); 5-of-40 framing consistent (N2 fixed)** |
| Bound 3 gate: seedext anchor cell | **sd_old 1.1543 ("1.154"), sd_new 0.5459 ("0.546"), var_ratio 4.4715 ("4.47"), mean shift 0.3637 ("0.364"), threshold 2×pooled_se = 1.3817 ("1.382"), new-cohort CI [−0.5060,+0.3569] ("[−0.506,+0.357]"), pooled CI = null in the raw (no decision-grade pool); naive n=12 pool recomputed from the 12 deltas with t(11)=2.201 → [−0.509,+0.147] exactly as printed; F(2,8) 5% ≈ 4.46 correct** |
| Registered verdict (A1 ground truth) | **`PHASE2B_SUMMARY.json.trajectories`: openr1-mix-ext UNRESOLVED, wikitext-mix-ext UNRESOLVED** |
| Internal tallies | **312+8+16+30=366; 78+2+4+6=90 (Fig 1 caption); per-wave GPU-h sum 9.51 → "≈9.5"** |
| Bundle integrity / anonymization | **bundle PDF ≡ main.pdf (same md5); 0 hits for the full brief.md token list in bundle tex; 0 hits in PDF strings** |

Not re-verified (calibration, per the gauntlet record): the 21.8–46.4% L_query
fall values (01b window-label check only; figure-gen.py asserts the source md5s)
and the wave-6 cohort SDs beyond internal consistency — both were in scope of
earlier stages and nothing in my read contradicts them.

## 4. Decision (a): detector gate — SKIP, discharged by precedent (recorded here)

**Decision: skip; do not run detector rounds for this paper.** Grounds:

1. **The class ruling already exists, twice.** mstar-colm-er ran the gate to its
   6-round cap; its Fable final review adjudicated SUBMIT HEAD AS-IS, finding the
   residual signal (antithesis cadence, parallel structure, takeaway captions) is
   the house honesty-pin discipline, judged content-motivated by the detector's
   own judges. capacity-colm-er then pre-bounded at 2 rounds and its coordinator
   discharged at bounded-terminal after round 1 on exactly that class ruling:
   only *mechanical* tells license iteration, and both judges cited zero.
2. **I read all 8 rendered pages specifically for tells.** Zero mechanical tells
   (no filler scaffolding, no generic transitions, no hedge-stacking, no banned
   words — consistent with the style gate's PASS at zero violations). The
   residual style class present (antithesis: "an instrument verdict, not a
   refutation"; epigrammatic headings: "The Replication Gate Refuses the Pool";
   three-part parallelism) is the identical class twice adjudicated
   non-actionable — and here the three-part structure *is the content*: the paper
   is three bounds.
3. **The expected information value of a run is ~zero and the downside is real.**
   The only possible outcomes are a pass or a reproduction of the
   already-adjudicated terminal state; meanwhile the MOSS late window is
   capacity-gated and iterating prose against an instrument the venue never sees
   risks the precision the gauntlet just verified (mstar's review: "varying them
   risks real precision for a detector the venue never sees").

If the PI wants a for-the-record entry anyway, a single bounded round in
parallel with the email costs nothing and blocks nothing — but it is not needed
for submission and this review formally discharges the gate as
**skipped-by-precedent, bounded-terminal class ruling applied**.

## 5. Decision (b): real-kernel positive control — PRE-SUBMISSION, run in parallel, non-blocking on the PI email, blocking on the upload

**Recommendation: dispatch the control now (pre-submission), in parallel with the
PI's late-add email; it must complete before the OpenReview upload, not before
the email.** The email is the PI's step and carries no paper content; the upload
is where the bundle becomes a submission. Sequencing:

- **Dispatch now.** Rebuttal FIX-6 tier 2 specifies it: one synthetic sequence
  with an analytically-known h-hop target through a real trained checkpoint's
  production `fla` forward path, confirm recovery at cosine ≥ 0.9. Minutes of
  GPU on a box that is already burning uptime (the grant is uptime-metered; the
  saturation directive applies).
- **If it passes:** submit the frozen bundle *unchanged* — do not reopen the
  prose pre-upload; fold the validated statement into camera-ready, where §7's
  hedge ("a real-kernel positive control is the next wave's priority") softens
  into a demonstrated result. MOSS is non-archival; camera-ready is the natural
  slot for the upgrade.
- **If it fails:** HARD STOP on the upload. A failure means a real
  extraction/convention bug exists and Bound 1's diagnostic framing
  ("construct-validity gap") must be revised before this paper is submitted
  anywhere — this is precisely the falsifier the paper itself names in Appendix
  A. Submitting first and discovering this at camera-ready would burn exactly the
  methods-venue credibility the paper is built on.
- **Timeout clause:** if the organizers say yes and the upload window would
  close before the control can land (it should not — the run is minutes), submit
  as-is: the residual is disclosed on-page in §7 *and* Appendix A, the paper's
  epistemic claims are bug-robust (it refutes nothing), and the control then runs
  before camera-ready. Capacity-gating argues for never letting this control
  delay the email by a minute — and it doesn't have to.

The asymmetry decides it: cost is minutes on hot hardware; the downside it
insures against is the headline bound of the paper being an instrument artifact
discovered post-submission at a methods venue.

## 6. Status after this review

- Gauntlet round 1: attack → defense → style → rebuttal → 01b re-attack (A1/A2
  CLOSED, N1/N2 fixed on-page) → format (0C) → render v3 (PASS 8/8) → **final
  review: READY-AFTER-CHANGES (1 required word fix)**.
- On the applier landing change 1 (+ recompile, rebundle, anonymization grep):
  **SUBMISSION-READY-pending-PI-email**, with the pre-upload positive-control
  gate of §5 above.
- Detector gate: **discharged (skipped-by-precedent)** per §4 above.
- The MOSS late-add email to the organizers remains the PI's step.
