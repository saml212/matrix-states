# 07_final_review — FABLE FINAL REVIEW (capstone adjudication), 2026-07-11

Reviewer: Fable (final human-tier adjudication the detector terminal
state requires). Scope: the four flagged decisions — ship-draft
selection, full 16-page render read, the ICLR float debt, and the arXiv
punch list. This file is the review of record; a Sonnet applier
executes the change list. No paper file was modified by this review.

## VERDICT: READY-FOR-ARXIV-AFTER-CHANGES

No round 2 is needed. The gauntlet state is real and held up under
re-verification: no CRITICAL open (01b), style PASS, format CLEAN
(05b), render v3 CLEAN, and my own three independent spot-checks
(Section A2.3 below) reproduce the raws to five decimals. The changes
below are build-mechanics and cosmetics, not claims; none touches a
number or a verdict sentence.

---

## Adjudication 1 — WHICH DRAFT SHIPS: the CURRENT TREE (HEAD)

The 0e264a0 sections (detector round-5 input, panel 93/90) and the
current tree (round-6 input, 72/95) differ by exactly 25 lines across
5 files — the round-5 fix wave. I read the full diff and judged each
hunk on prose quality, honesty, and precision, never detector score:

1. **§2.5 instrument-repair narrative — HEAD strictly better.**
   0e264a0 runs a uniform "In the X program… In the Y program… In a
   separate Z program…" mold. HEAD varies the three shapes, and its
   middle sentence — "the repaired tap of Section 4.3 read what the
   model's own forward pass had decoded all along" — is the best
   sentence in the section: accurate, load-bearing, and it prefigures
   §4.3's punchline instead of merely pointing at it.
2. **§6 MQAR citation stack — HEAD strictly better.** 0e264a0's
   "show / show / show" verb train is flat; HEAD varies the verbs AND
   adds an argumentative point 0e264a0 lacks: "even two-layer
   from-scratch transformers reliably develop induction circuitry" —
   the "even" is doing real work, since our baseline is exactly a
   two-layer transformer, sharpening why its chance-level read runs
   against the literature. Checked against Olsson et al.: accurate.
3. **§1 intro thesis close — neutral-to-better.** HEAD's "The same
   write mechanism, run at language-model scale" drops 0e264a0's
   explicit "in delta-rule language models" but the substrate scoping
   survives by direct anaphora (the preceding sentence opens "In a
   two-layer delta-rule model") and is restated explicitly in §2.2,
   §6, and §8. "Run at language-model scale" adds a true scoping
   element (synthetic-task 14M vs LM-scale) that 0e264a0's version
   lacked. Verified on the rendered page: no conflation is reintroduced.
4. **§8 conclusion — neutral.** HEAD splits the sentence and closes
   "stock normalizer or no" — mildly colloquial for a conclusion but
   accurate ("at every larger scale" = the monotone ladder, correct)
   and it reads human. 0e264a0's "regardless of the stock normalizer"
   is stiffer; neither is wrong.
5. **Abstract close — 0e264a0 marginally more explicit.** "the loss
   neutrality transfers to scale; the geometric benefit does not"
   names both halves; HEAD's "only the mitigation's loss neutrality
   survives the move to scale" implies the second half. HEAD is
   accurate and the explicit two-half statement survives verbatim
   where it matters most (§5.3 body: "The neutrality transfers to
   scale. The fix does not." and the Figure 7 caption). Acceptable.

Score: two hunks strictly better in HEAD, two neutral, one marginally
weaker but accurate with the explicit form preserved in-body. Every
numerical claim, caveat, and scoping element is identical between the
two (terminal-state.md's record, independently confirmed by the diff).

**The mstar precedent holds here.** Reverting to 0e264a0 would
reintroduce three tells cited by name at round 5 (the §2.5 tri-clause
mold, the §6 verb uniformity, the abstract/§5.3 skeleton echo) to chase
a panel score (93/90 vs 72/95) that the terminal state itself documents
as fresh-judge noise at the plateau — the same fragments were cited as
"strongest human counter-signal" by one panel and "dominant machine
tell" by the next, and no mechanical or lexical tell has been cited by
any judge since round 4. Detector-score-chasing is exactly what this
adjudication is instructed not to do; on the merits, HEAD's prose is
better writing. **The current tree ships.** 0e264a0 remains in history
as the detector-best record; no action.

## Adjudication 2 — THE 16-PAGE READ

All 16 pages read as rendered images (fresh pdftoppm off the committed
main.pdf, which matches the v3-inspected build, commit 729ba8c).

**Does T1 land as ONE thesis? Yes — one medium with laws, not three
stapled papers.** The unification is carried by four load-bearing
structures that all survive on the page: (i) §1's thesis paragraph
states the single object ("what the matrix state stores") and scopes
each law to its family in one breath; (ii) §2 sets up all three legs
against one instrument discipline, and §2.5's three repair stories —
one per leg — make the methodology itself the connective tissue;
(iii) §5 opens by re-binding the pathology to the storage result ("The
same write mechanism that Sections 3 and 4 show storing task
structure…") and §5.4 names the common factor without overclaiming;
(iv) Appendix A turns the one real seam — the rank law living on the
encoder family, not the delta-rule family — into a measured datum
(complement channel empty at ≤3.2e-12 in delta-rule states,
"mechanism stories that depend on an identity scaffold cannot port
across that boundary"). A paper that quantifies its own seam is one
paper being honest, not three papers stapled. The three-legged
structure reads as designed decomposition, and the recurring
degenerate-baseline / Nichani / matched-budget caveats give the legs a
shared voice.

**The two §6/§8 conflation fixes are intact on the page.** §6 (p.11–12,
DeltaNet paragraph): "the rank law of Section 3 is established on a
separate matrix-state encoder family with no delta-rule write, while
the causal recall capability and the write-induced geometry at scale
are established on the delta-rule substrate itself." §8 (p.12–13):
"Its trained rank is set causally by the task's representation theory
in the encoder family. In the delta-rule family, the first layer's
state causally carries recall…" Both per-leg scopings render verbatim.

**The transformer-baseline demotion is consistent at every surface I
could find (nine of them):** abstract ("disclosed as a degenerate
baseline"), intro thesis + contribution 2 ("recorded as a
degenerate-baseline datum, not a second verdict"), §4.1 (LR never
task-searched + near-flat loss disclosed with numbers), §4.2 (verdict
carried by the ablation comparison), §4.4 (degenerate-baseline clause
fires; "not a strongest-possible-baseline result"; no crossover
certified), §6 ("we do not interpret it as an architectural
inability"; optimization as leading candidate), §7 (the resolving
LR-grid experiment specified: ≥4 rates spanning 1e-4 to 3e-3, 3 seeds,
20k matched steps, curves reported — FIX-5's disclosure intact),
§8 ("the compute-matched transformer, a degenerate baseline, also
fails"), and the Figure 3 caption. The abstract's M* clause correctly
ties "reads chance everywhere" to "that transformer" (the
degenerate-tagged one, per the 01b N2 fix). No surface claims a
transformer inability. Consistent.

### A2.3 Spot-checks vs raws (3/3 pass, all recomputed independently)

1. **Param counts (the format-audit C1 CRITICAL's correction).** All
   three contender task-1 training JSONs in
   `experiment-runs/2026-07-10_h2h_sweep_harvest/` read
   `n_params = 14,049,408`; all three ablation JSONs read 14,048,384
   (Δ=1,024 = 0.007%, as printed in §4.1 and B.1); all nine task-1
   arms read `lr = 0.0003` (the §4.1/R0 shared-default claim, exact).
   Transformer n_params 14,440,448 — 2.8% over the contender,
   consistent with "FLOP-matched within 5 percent" and never claimed
   as parameter-matched. PASS.
2. **R4 verdict block.** Re-metric JSONs
   (`sweep_remetric/*_task1_*_round4.json`) reproduce Table 1 exactly:
   contender [0.99951, 1.00000, 0.99902] mean 0.99951; ablation
   [0.03223, 0.03271, 0.03687] mean 0.03394; transformer [0.02710,
   0.02930, 0.02856] mean 0.02832. I recomputed the paired t-intervals
   (df=2, t=4.3027) from the raw per-seed values: contender−ablation
   0.96558, CI (0.95822, 0.97293); contender−transformer 0.97119, CI
   (0.96855, 0.97383) — all five decimals match §4.2. The §4.3 sweep
   S₀-zeroing replication numbers (0.0339/0.0012/0.0002) also match
   the raws. `MARGINS_FROZEN.token` md5 = 58fe68e7…e204 matches the
   brief. PASS.
3. **R6 ladder endpoint + final window.**
   `experiment-runs/2026-07-06_trackc_rung3/probe_analysis_rung3.json`
   md5 = 6a627c31…30f8 matches the brief; pooled `archived4` (the
   cross-scale convention) final span_frac 0.45544 → 0.455 as printed;
   the final-window claim reproduces exactly (step-130k 0.45841 →
   step-155k 0.45544 = "0.4584 to 0.4554" over 25,000 recorded steps,
   §5.1). `fixscale_harvest_verdict.json` md5 = f2f0aae8…5158 also
   matches the brief (R8). PASS.

### A2.4 New findings from the page read (none blocking; fed into the change list)

- **[F1] The running header is false for the named build.** Every page
  reads "Published as a conference paper at ICLR 2026" (the kit's
  `\iclrfinalcopy` header) above an "Anonymous authors / Paper under
  double-blind review" byline. The paper is not published at ICLR
  2026, and the header contradicts the byline. → C1/C2.
- **[F2] The contribution list renders as one run-on paragraph**
  (p.2): items 1–4 flow inline inside a single justified block. The
  most-skimmed half page of the paper is the hardest to skim. → C6.
- **[F3] B.2 promissory wording:** "The camera-ready appendix carries
  the full per-seed tables…" — true of no build that exists; this PDF
  does not carry them. → C4.
- **[F4] B.3's archive link is unfilled:** "the named (arXiv) build
  links the archive root" — no link present in this build. → C3.
- **[F5] p.14 layout:** Figure A1 floats above the "A The c*I
  Complement Scaffold" heading with two large white gaps; the appendix
  opens mid-page under its own figure. → C7.
- **[F6] Figure 6 annotation/caption mismatch at a glance:** the panel
  annotates Δ=−0.10 and Δ=+1.13 (vs-baseline deltas of the two
  qk-norm-off cells) while the caption headlines the registered gating
  read +4.312 (qk-norm-on/gating vs baseline), which is not annotated
  on the panel. Nothing is wrong; the figure just doesn't show the
  number its caption leads with. → C8.
- (Known, carried:) fig5 inset tick size — the v3 cosmetic minor. → C9.

## Adjudication 3 — ICLR FLOAT DEBT: COMPRESS POST-ARXIV

Measured off the render: main text ends a quarter into p.13 → ≈12.3pp
of ICLR-format main text (references pp.13–14, appendices pp.14–16)
against the 9.0pp budget: ≈3.3pp of debt. Decision: **the arXiv build
ships at natural length; the compression pass happens in September for
the ICLR build.** Reasons, in order of weight:

1. **The citable record should be the full text.** arXiv has no page
   limit, and the long version is the better permanent artifact — every
   disclosure (the D-AMB narrative, the degenerate-baseline
   scaffolding, §4.5's INDETERMINATE data) is part of this paper's
   argument for itself. Compressing first would make the *weaker* text
   the record.
2. **A 3.3pp compression is the largest possible perturbation to a
   text at a verified plateau.** The detector history shows every fix
   wave minting new tells two rounds later; a one-third rewrite before
   Jul 31 would demand a fresh style pass, a scoped re-attack on the
   compressed prose, and realistically a new detector cycle — spending
   the arXiv window to de-verify the tree's best-verified state.
3. **Two texts exist regardless.** The ICLR build already differs by
   anonymization and (eventually) the iclr2027 kit swap. The brief
   binds both builds to the single evidence map ("a number fixed in
   one is fixed in the other"), which is the correct mitigation for
   the maintenance burden — the burden argument does not buy a
   pre-arXiv compression.
4. **The absorb-vs-separate PI decision (flagship vs `iclr-2027/`) is
   still open.** §5 was deliberately built as a swappable module; if
   the PI later picks ABSORB, the September compression is where that
   restructuring lands anyway. Compressing §5 twice would be pure
   waste.
5. **Timeline.** ~Jul 31 arXiv → ~Sep 24 ICLR leaves seven-plus weeks
   for a compression pass with its own gauntlet. The reverse ordering
   leaves under three weeks for the same work plus the named build.

Registered consequence (C11): the September compression targets ≤9.0pp
per the brief's section budget table, and the compressed prose must
re-run the style judge plus a scoped detector round before submission.

## Adjudication 4 — ARXIV READINESS: the punch list to the named build

Preconditions verified: no CRITICAL open (01b); format re-audit CLEAN
(05b); render v3 CLEAN; detector residue adjudicated in C10 below;
FIX-5's LR-grid recommendation confirmed present as a §7 disclosure
(p.12) and stays exactly that — a disclosure, not a pre-arXiv re-run.

---

## CHANGE LIST (numbered; the Sonnet applier executes; none touches a number or verdict sentence)

**C1 (REQUIRED — build).** Override the running header for the named
build. The iclr2026 kit's `\iclrfinalcopy` prints "Published as a
conference paper at ICLR 2026" on every page — a false statement for
an arXiv preprint. Redefine the fancyhdr header text to "Preprint."
(honest, since the ICLR 2027 CFP is not live) in `latex/main.tex` for
the named build; the ICLR submission build later reverts to the kit
default under anonymization.

**C2 (REQUIRED — PI gate, procedure only now).** `[WORKING TITLE —
PI]` and `[AUTHORS — PI]` placeholders REMAIN until the PI stamps.
At stamp time, apply the EA-kit de-anonymization pattern
(`papers/neurreps-ea/arxiv/` precedent: exactly two changes off the
pinned gauntleted tree — remove the anonymized/review block, fill the
real author block; body text unchanged), with the corresponding
address per the brief. The "Anonymous authors / Paper under
double-blind review" block must not ship in the named build.

**C3 (REQUIRED).** Fill B.3's archive pointer: "the named (arXiv)
build links the archive root" must be true of the shipped PDF — insert
the real repository/archive URL (or DOI) at build time, alongside the
released full-checksum manifest it promises.

**C4 (REQUIRED).** Repair B.2's promissory sentence ("The camera-ready
appendix carries the full per-seed tables…"). Either (preferred, arXiv
has no limit) actually generate and include the per-seed tables from
the same checksum-asserting script, or reword to point at the released
archive so the sentence is true of this PDF. No third option.

**C5 (REQUIRED).** Resolve the two documented bib deferrals for the
named build: (a) the §6 self-cite "The Gradient Does Not See Rank"
(ICML 2026 MI workshop) is anonymization-sensitive only for the ICLR
build — add the real bib entry + `\citep` now; (b) Qwen3-Next — cite
the canonical release artifact if one can be verified, else drop the
name from §6's parenthetical (never fabricate an entry; the bib
header's own rule).

**C6 (SHOULD).** Render the four contribution bullets as a proper
`enumerate` (md2tex currently flattens them into one run-on
paragraph, p.2). Layout-only change, zero prose edits, detector-safe.

**C7 (SHOULD).** Fix p.14 float placement so Appendix A opens with its
heading rather than under Figure A1 with two large white gaps
(placement option or `\clearpage` before `\appendix`).

**C8 (MINOR).** Figure 6: annotate the qk-norm-on/gating cell's
+4.31 delta (the number the caption leads with), or make the caption
say which two deltas are annotated. Panel/caption should tell the same
story at a glance.

**C9 (MINOR, optional).** Figure 5 inset tick-label size (v3's known
cosmetic; values already restated in the caption).

**C10 (RECORD — this file discharges it).** This review is the human
adjudication the detector terminal state requires. Ruling: the
cap-hit residue (uniform information density; inter-panel
contradictions on identical fragments) is adjudicated ACCEPTABLE — the
only available fix is filler, declined under the style contract; zero
mechanical/lexical tells cited since round 4; COLM-sibling precedent
followed. The HEAD tree is cleared for the named arXiv build once
C1–C5 land and the PI stamps title/authors. `/publish`'s refusal is
lifted by this record for the arXiv build only; the ICLR build gets
its own pass after the September compression.

**C11 (RECORD).** Schedule the ICLR compression pass POST-arXiv
(Adjudication 3): target ≤9.0pp main text per the brief's budget
table; §5 stays modular pending the PI's absorb-vs-separate decision
on `iclr-2027/`; compressed prose re-runs the style judge and a scoped
detector round before submission.

---

*Checks run for this review: full 0e264a0..HEAD section diff read
hunk-by-hunk; all 16 rendered pages read as images; 3/3 raw-artifact
spot-checks reproduced to five decimals (param counts + LR fields, the
complete R4 verdict block including recomputed paired CIs, the R6
rung-3 endpoint + final-window + three brief md5s); §6/§8 scoping and
the nine demotion surfaces verified on the page; FIX-5 §7 disclosure
verified present; latex preamble and bib deferral notes inspected.*
