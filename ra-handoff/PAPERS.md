# PAPERS — the complete portfolio inventory

Fifteen paper trees. One published, one live, seven complete-but-unsubmitted,
six superseded or consumed. This file says what each one claims, how it builds,
how finished it is, and what the venue situation actually is.

> **⚠ Read this first: every deadline in this repo has passed.**
> The whole portfolio was written 2026-07-09 → 2026-07-17 and then frozen while
> the GPU program ran. Every venue note in every tree was live-verified in
> mid-July and then went stale. **A live CFP re-fetch on 2026-08-24 —
> `EXPERIMENT_LOG.md` 2026-08-24 #3, the report of record — changed the picture
> materially:**
>
> - **NeurReps 2026 EA closed Aug 24 2026 AoE**, not Aug 29. The workshop set its
>   own date; "Aug 29" was the NeurIPS-wide *suggested* default that this repo had
>   been planning against. **B1 and B3 both targeted that track.** Confirm with
>   Sam whether he submitted before it closed.
> - **UniReps 2026 does not currently exist as a venue** — the site says TBA and
>   it is absent from all 102 accepted NeurIPS 2026 workshops. **B2 needs a
>   re-home** independent of any clock.
> - **ICBINB 2026 pivoted to biology**, invalidating the on-record backup for B7.
> - Efficient Reasoning @ COLM (Jul 19) and MOSS @ COLM (Jul 3 + late window,
>   notification Jul 24) had already passed.
>
> **Net: no live clock is currently known, and six trees need re-homing.** The
> NeurIPS accepted-workshop list *is* now published (102 workshops) but has never
> been scanned end-to-end for candidates — that scan is the highest-value venue
> action available. Re-verify every CFP live before acting on anything below.
> Details in `SUBMISSION_PLAYBOOK.md` §4–§5.

---

## Two generations of trees

| Generation | Location | Written | What it is |
|---|---|---|---|
| **Gen 2** — `paper`-skill trees | `papers/*` | Jul 10 – **Aug 23 2026** | 9 trees: brief → outline → sections → gauntlet → detector → bundle. `flagship` is the only live one |
| **Gen 1** — hand-rolled | `matrix-thinking/submissions/*` | Jul 3 – Jul 17 2026 | 6 trees: 1 published, 1 unresolved, 4 superseded by Gen-2 successors |

Every Gen-1 tree except `icml-mi-workshop-2026/` and `iclr-2027/` has a named
Gen-2 successor that treats it as a read-only source. Don't edit Gen-1 trees.

**Build systems, summarized:** only `papers/flagship/` has a markdown→LaTeX
pipeline (`md2tex.py`). **Every other tree is hand-written LaTeX.** Most use
`Makefile` + three `tectonic` passes. Every tree with figures has an
md5-asserting generator in-tree that reads archived JSONs and fails loudly if an
archive changed — preserve that property.

---

# A. THE LIVE ONE

## A1. `papers/flagship/` — the ICLR 2027 full paper ⭐ START HERE

**Title:** *Fast-Weight Matrix States Are a Representational Medium, Not an
Efficiency Trick* (chosen by the PI 2026-07-11)
**Status:** live; committed 2026-08-23, ~36 commits, most recent work in the repo
**Venue of record:** arXiv preprint (named build) → **ICLR 2027** (anonymized,
double-blind, archival). Projected abstract Sep 19 / full paper Sep 24 2026 —
**third-party aggregator dates; the official ICLR 2027 CFP was 404-verified not
live as of 2026-07-10 and has not been rechecked.**
**Build:**
```bash
cd papers/flagship/latex
python3 md2tex.py            # sections/*.md -> latex/sections/*.tex  (markdown is the source)
tectonic -X compile main.tex # -> main.pdf
```
**Verified 2026-08-24:** compiles clean from a cold cache, exit 0, **26 pages**.
(The committed `main.pdf` was stale — built 2026-07-11, before the two new
chapters. It has been rebuilt as part of this handoff.)

**What it claims.** Three independent legs plus a mechanism appendix, arguing
the d×d matrix state is a representational medium:

- **§3 Rank law** (rows R1–R3, R1b). Five permutation groups, `d_min` 2→5.
  Recruited effective rank tracks `d_min` at ρ = 0.9747 (the tie-capped
  maximum), 19/19 cells in the pre-registered band. S₄-vs-A₅ TOST DECLARES
  equivalence — dimension, not solvability. Causal razor: 0.000 at `d_min − 1`
  in all 5 groups and all 4 S₃ seeds; recovery returns at `d_min` 5/5.
  Substrate is the matrix-state **encoder** family, not delta-rule — flagged in
  the brief's "T1 WORDING FLAG."
- **§4 Capability separation** (R4, R5, R10, R4a, R12). Delta-rule contender
  0.99951 episodic recall vs param-matched vector ablation 0.03394 and a
  compute-matched transformer at chance after a 4-point LR search. S₀-zeroing
  collapses it; S₁-zeroing doesn't. Legible only post-nonlinearity. ≥0.998 to
  1798 tokens on a fixed 32,768-byte state vs a cache-capped transformer at
  chance everywhere. **R4, R5, R9 appear in no sibling paper — this is the
  flagship's exclusive territory.**
- **§5 Pathology at scale** (R6–R8). Span fraction 0.248 → 0.455 across
  14M→1.31B. qk-norm removal is a within-noise null. The frozen-key-bias fix is
  loss-neutral at every scale but its geometric benefit does not transfer past
  14M.
- **§6 Breadth scaling** and **§7 Parameter scaling** — **NEW, added
  2026-08-23.** The NCR chapters. Full technical summary in
  `AGENT_INSTRUCTIONS.md` §1.
- **The third scale point (1.31B) — landing; tables update pending final
  harvest.** ⚠ **Do not write it into the paper yet, but plan for it.**
  Calibration **cleared all license legs** (root `EXPERIMENT_LOG.md`
  **2026-08-24 #4**): deep P1b κ at h_top = 36 reads **1.0000** primary / 0.9796
  compB against a 0.90 bar, P0 in band both. The 22-cell sweep is draining; the
  interim readout —
  `matrix-thinking/scaleaxis_build/job_specs_1b/EXPERIMENT_LOG.md`
  **2026-08-25 #1**, explicitly labelled **informal, 11/16 scored** — reads
  **frozen at ceiling at every measured K** (K=24 0.9918/0.9878; K=32
  0.9798/1.0000/0.9960; K=40 0.9920/0.9960), wall in band 11/11, trainable
  degradation persisting (K=32 0.8065, K=40 0.7877). Last-cell projection was
  ~02:00–05:00Z 2026-08-26. **Verdicts come from the pinned tests at full drain,
  not from that interim readout**, and that entry lives in the wave-local log,
  **not** the root `EXPERIMENT_LOG.md` — cite the full path.
  **The instrument question is now settled** (root `EXPERIMENT_LOG.md`
  **2026-08-24 #2**, Ruling 2, pinned before any 1B harvest): 1.31B verdicts use
  the **same pairwise instruments** as the 98M-vs-392M chapter applied to the
  **(392M, 1.31B)** pair at **identical thresholds**; the three-point figure is
  **DESCRIPTIVE**; a "scaling law to 13.4×" sentence requires **both** pairwise
  capability verdicts to read no-detectable-shift/STABLE; **no new statistic is
  invented post-design.** A three-point *figure* is fine. A three-point *trend
  statistic* is not, and must not be invented to get one.
- **Appendix A** (R9): the c\*·I complement scaffold, architecture-conditional.

**Completeness.** All 13 section files written. Evidence rows R0–R12 in
`brief.md`, three-field format, md5-pinned. Eight figures with an md5-asserting
generator. Gauntlet round 1 complete (attack, re-attack, defense, style,
rebuttal, format, format re-audit, three render inspections, final review).
Detector gate ran six rounds.

### The open editorial questions

**Two were named by the consolidation pass itself**, in the comment block at
`papers/flagship/latex/main.tex` lines 62–81:

1. **Cross-reference resync.** The existing chapters address sections by
   **literal number**, not `\ref`. The consolidation charter forbade editing
   existing chapters, so eight literals now point one or two sections short and
   must be bumped by **+2**. They are enumerated exactly in that comment block
   (`01_introduction.md` ×1, `04_capability_separation.md` ×4 including a figure
   caption, `06_related_work.md` ×2, `07_discussion_limitations.md` ×2,
   `08_conclusion.md` ×1). Note the trap the comment flags: "Section 4 of
   arXiv:2510.26692" in `06_related_work.md` is a section of a *cited paper* and
   must NOT be bumped. **Consider converting these to `\ref` while you're in
   there** — it's the actual fix.
2. **Placement.** The new chapters currently compile as Sections 6 and 7, after
   the pathology chapter. But they extend §4's capability separation, not §5's
   pathology, so **positions 5 and 6 are editorially stronger**. The resync cost
   is identical either way. This is a genuine editorial call and it is yours to
   make.

**One is carried in `brief.md`** and is the biggest strategic question in the
portfolio:

3. **Does the flagship ABSORB `matrix-thinking/submissions/iclr-2027/`, or do
   they stay separate submissions?** They cannot both go to ICLR 2027 with the
   same §5 evidence. The drafting assumption of record since 2026-07-10 is
   **SEPARATE** — flagship §5 was written self-contained from the raw archives
   without copying a paragraph from that tree, precisely so it can be swapped
   for a cross-reference if the answer flips. That decision has not been
   revisited, and all activity since has gone to the flagship. See §C4 below for
   what the other tree actually contains.

**Three more I verified during this handoff, not previously written down:**

4. **The page budget has blown out.** The brief budgets 9.0 pages of main text
   to exactly hit ICLR's limit. The pre-consolidation build was 15 pages total.
   The current build is **26 pages: main text runs to ~p23, references p23–24,
   appendix p25–26.** Main text is roughly 23 pages against a 9-page limit. The
   two new chapters are ~48KB of markdown between them and carry six tables.
   Something has to give: compress the new chapters hard, move most of their
   tables to the appendix, or reconsider the venue. This is the largest single
   piece of work in the tree.
5. **The detector gate hit its cap without passing.** Six rounds, twelve fresh
   judges, never two consecutive all-"100% human" rounds (worst-judge by round:
   55 / 75 / 90 / 80 / 90 / 72). Per the procedure the draft is **not
   accept-ready** and requires human adjudication.
   `papers/flagship/detector/terminal-state.md` surfaces commit `0e264a0` as the
   best-scoring version and flags "which version ships" as a PI decision. A
   genuine rewrite in your own voice moots the question — which is the point of
   the handoff.
6. **Authors are unset.** `\author{Anonymous authors}` with `[AUTHORS — PI
   decision pending]` in `main.tex`, `brief.md`, and `00_abstract.md`. Settle
   with Sam before any arXiv posting.

**Next steps, in order:** rewrite in your own voice → fix the page budget → fix
the cross-references (convert to `\ref`) → settle placement → settle
absorb-vs-separate with Sam → settle authors → re-verify the ICLR 2027 CFP →
post the named build to arXiv → prepare the anonymized ICLR build (comment out
`\iclrfinalcopy`, delete the `\lhead{Preprint.}` override, run the anonymization
grep in `brief.md` §"Anonymization surface", swap in the real `iclr2027` kit if
it has shipped).

---

# B. THE WORKSHOP STACK (Gen 2, `papers/`)

All eight are content-complete. All were frozen mid-July. The blockers are
venue decisions and PI stamps, not writing.

## B1. `papers/neurreps-ea/` — ACCEPT-READY, **deadline passed 2026-08-24**

**Title:** *The Rank the Task Demands: A Causal Rank Law for Matrix Memories
Trained on Group Composition* (Candidate A, adjudicated final)
**Claim:** Recruited rank equals the group's minimal faithful real
representation dimension `d_min`; the dimension-matched solvable/non-solvable
pair S₄/A₅ is TOST-equivalent; the force-rank razor is exact in both directions.
Closing line: *"Representation theory, not computational complexity class, sets
what gradient descent buys."*
**Venue:** **NeurReps 2026, Extended Abstract track** — 4pp excl. refs/appendix,
non-archival, "no restrictions" on dual submission (so ICLR-flagship-safe),
double-blind on OpenReview. Required template is the NeurReps JMLR/PMLR style
zip; the EA track uses `\documentclass[mlabstract,onecolumn]{jmlr}` and the
paper must be a **single .tex file**.
**⚠ Deadline: AUG 24 2026 AoE — PASSED.** Live-verified 2026-08-24
(`EXPERIMENT_LOG.md` 2026-08-24 #3): the workshop set its own date; the "Aug 29"
this repo planned against was the NeurIPS-wide *suggested* default, and the 2025
Aug 29 → Sep 4 pattern did not repeat. Everything else verified good at the same
time: anonymized on the official template, EA dual-submission **unrestricted**
(so flagship-safe), portal
`openreview.net/group?id=NeurIPS.cc/2026/Workshop/NeurReps_Extended_Abstracts`.
**First action: confirm with Sam whether he submitted before it closed.** If he
did, this is in review and the next step is camera-ready prep. If not, the paper
is finished and homeless — re-home it.
**Build:** `make` → 3× `tectonic --keep-intermediates main.tex`. 8 pages
(4pp body). `make bundle` flattens to a single .tex. `make figures` regenerates
2 md5-asserted figures.
**⚠ Build trap:** a clean-tree build transits a mis-paginated 12pp layout and
only converges to the correct dense 8pp layout on the **third** pass (aux
fixed-point transient). **Upload the pinned `bundle/neurreps-ea-submission.pdf`;
never a single-pass recompile.**
**Completeness:** ACCEPT-READY. 14/14 bibliography entries, all 10 arXiv
entries verified against the live API — **including two that were fabricated and
were caught and fixed** (`nazari2026rank`, `sun2026staterank`). Anonymization
grep zero hits. Render inspection PASS 0/0/0 across all 8 pages. Abstract 223
words.
**arXiv:** `arxiv/neurreps-ea-arxiv-v1.zip` is **built and ready**, never
uploaded. cs.LG primary / cs.NE secondary, CC BY 4.0, Sam Larson (Pebble AI).
**Next steps:** **confirm submitted-or-not with Sam** → if not submitted, re-home
(it is fully finished, so any representation-learning or geometry workshop is a
candidate) → post arXiv v1 either way, which is unblocked by the venue question
except for the cs.LG endorsement. At camera-ready: uncomment the author block,
restore the self-citation that was cut for double-blind, remove the
draftwatermark.

## B2. `papers/unireps-ea/` — ACCEPT-READY, **its venue does not exist in 2026**

**Title:** *Dimension, Not Solvability: Trained Matrix States Converge to the
Minimal Faithful Representation Dimension* (Candidate A, adjudicated final)
**Claim:** Same evidence base as B1, different headline — angled at convergent
representations. Effective ranks 1.88/2.85/2.83/3.59/4.74 vs `d_min` 2/3/3/4/5,
all 19 seeds in band, ρ = 0.9747; the S₄/A₅ TOST is the designed head-to-head.
**Venue of record (now dead):** UniReps 2026, Extended Abstract track — 4pp main
text (camera-ready 5pp), non-archival, NeurIPS LaTeX template, anonymized,
OpenReview. Deadlines were "TBD AoE" on a 2026 CFP *skeleton* that never became a
real venue. **The format spec is still a useful target shape; the venue is not.**
**⚠ THE VENUE DOES NOT EXIST THIS YEAR — RE-HOME REQUIRED.** Live-verified
2026-08-24 (`EXPERIMENT_LOG.md` 2026-08-24 #3): **UniReps 2026 does not currently
exist as a venue.** `unireps.org` says TBA, and UniReps is absent from all **102**
accepted NeurIPS 2026 workshops. This supersedes the earlier URL correction (the
tree's `VENUE_REQUIREMENTS.md` points at `unireps-2025.netlify.app` and guesses a
`unireps-2026.netlify.app` successor that 404s; `papers/VENUE_MAP.md` was
corrected to `unireps.org/2026/call-for-papers`) — the URL was the smaller
problem. The paper is finished and has nowhere to go until you pick somewhere.
Note it shares its evidence base with B1, so if B1 also ends up homeless, the two
should be re-homed as a coordinated pair rather than independently.
**Build:** `make` → 3× tectonic. 7 pages (body ends p.4). No pagination
bistability in this tree, but the triple pass is kept for determinism.
**Completeness:** ACCEPT-READY. 10/10 bib entries, 8 arXiv-verified. 30 evidence
tokens in the tex. Bundle byte-identical to a fresh flatten. Render inspection
v4 PASS 0/0/0.
**arXiv:** `arxiv/unireps-ea-arxiv-v1.zip` **built and ready**, never uploaded.
**Note:** this tree has **no `detector/` directory** and no `07_final_review.md`
in either gauntlet round — the only Gen-2 tree missing both. Its readiness call
comes from `papers/SUBMISSION_PACKAGE.md` §2, not from an in-tree final review.
**Next steps:** **pick a new venue** (scan the 102-workshop NeurIPS list; ICLR
2027 workshops are the other natural pool) → retarget the template if the new
venue needs it → post arXiv v1, which does not wait on the venue.

## B3. `papers/rank-recruitment-ws/` — READY, **deadline passed 2026-08-24**

**Title:** *When the Gradient Sees Rank: Provable Necessity, Causal Recruitment,
and Exact Composition in Trained Matrix Memories*
**Claim:** Where the exact solution provably requires rank(Z) ≥ K and every
rank-evading shortcut is closed by construction (exact continuous readout, never
argmax; a single-matrix-state P=1 bottleneck; a 2–2.5× budget-extension rule
before declaring a cell dead), SGD recruits the rank the task demands. Learned
effective rank tracks K (ρ = 1.0 at d=16); the causal step at k ≈ K is sharp
(rank 3 → ≤0.0004, rank 4 → 0.97 at d=8, K=4); the trained operator composes
exactly through 21-fold self-application in 4/5 seeds.
**Venue:** **NeurReps 2026 EA** — the *same venue* as B1. This is deliberate and
disclosed in both briefs; an "ESCALATION-1" de-dup pass edited B1 to point at
this paper as a companion rather than repeat its numbers. Confirm the venue
permits two submissions from one group before submitting both.
**⚠ Deadline: AUG 24 2026 AoE — PASSED**, same as B1 and verified in the same
2026-08-24 fetch. The dual-submission question that was open here is **answered**:
NeurReps EA dual submission is **unrestricted**, so two submissions from one group
was never the blocker — the clock was. **Confirm with Sam whether he submitted;
otherwise re-home**, ideally alongside B1 since the de-dup pass made them a
deliberate companion pair.
**Build:** `make` → 3× tectonic. 7 pages. `make figures` needs
`DRY_RUN_BYPASS=1` (the repo's pre-train hook pattern-matches `python3 …py`).
**Completeness:** READY-AFTER-CHANGES with the change list dispositioned;
detector DISCHARGED at round 1. 12 evidence rows (R1–R10 with R1b, R2b),
including two documented in-row corrections against the base draft.
**Next steps:** confirm submitted-or-not with Sam → PI title stamp → re-home if
not submitted → swap the `anonymous.4open.science` placeholder for a real
anonymized code snapshot (flagged non-blocking) → OpenReview (a **PI action** —
it needs Sam's account).

## B4. `papers/capacity-colm-er/` — SUBMISSION-READY, **deadline passed**

**Title:** *The Capacity of a Fixed-Size Fast-Weight State: A Located Frontier
That Grows Super-Linearly with Dimension*
**Claim:** The safe associative load of a trained fast-weight state is **not**
set by a universal K/d_state ratio. The d=64 frontier is sharply located at
K/d = 0.5455 (CI [0.5385, 0.5513]); the identical K/d window is flat at ceiling
at d=128; the d=80 frontier is 0.6779, excluding the pre-registered
ratio-invariance band; d=96 shows no transition through K/d = 0.9375. Capacity
grows as **d^1.97** (CI [1.86, 2.09]) — consistent with state *bytes*, not
width. Anchor-table coherence, the leading single-variable account, fails a
direct frozen-dose test (19/19 at ceiling under both concentrated and diffuse
injection).
**Venue:** 2nd Workshop on Efficient Reasoning @ COLM 2026. **Deadline was
Jul 19 2026 AoE — PASSED.** Non-archival, 4–10pp, double-blind, OpenReview.
**Build:** the odd one out — **no Makefile, and the build root is `bundle/`**:
```bash
cd papers/capacity-colm-er/bundle
tectonic --keep-intermediates main.tex   # pulls ../sections/, ../figures/
```
Figures: `python3 papers/capacity-colm-er/figures/figure-gen.py` from the repo
root. 10 pages.
**Completeness:** the cleanest tree in the repo — **zero** TODO/PENDING/TBD hits
of any kind. 19 evidence rows (C1–C18, C20). Uniquely carries
`bundle/citation-verification.json`, a machine-readable record of every bib
entry checked against the arXiv API (or CrossRef for the pre-arXiv classics).
Detector round 1 was split (68% / 90%) and was signed off as
"bounded-terminal, no-actionable-defect."
**Honesty note the brief flags:** the super-linear claim is per *dimension*; per
*byte*, capacity is approximately conserved over the measured range. Both sides
of that arithmetic are stated in the paper. Keep it that way.
**Next steps:** **needs re-homing.** Options: a NeurIPS 2026 memory/efficiency
workshop off the 102-item accepted list (**verify each workshop's own deadline —
several have already closed**), ICLR 2027 workshops, or arXiv-only.

## B5. `papers/mstar-colm-er/` — READY, **deadline passed**

**Title:** *Constant-Memory Recall: A Fixed 32 kB Fast-Weight State Retains
In-Context Bindings That a Matched Transformer Does Not Learn*
**Claim:** At matched params, data and schedule on 32-way episode-restricted
associative recall, a two-block fast-weight model with a fixed 32,768-byte state
reads 0.9995 while a param-matched flat-vector recurrence and a param-matched
transformer both sit at chance. Accuracy holds ≥0.998 from 454 to 1798 tokens
with the state fixed; the transformer reads chance uncapped and at every KV-cache
budget from 1× to 32×. State-zeroing localizes recall causally to block 1.
**The registered posture is negative-space honesty:** the degenerate-baseline
clause fired, so **no memory-multiplier is quotable** — the sanctioned claim is
"baseline non-competitive at matched params/tokens," not "N× better."
**Venue:** Efficient Reasoning @ COLM 2026. **Jul 19 2026 AoE — PASSED.**
**Build:** `make` → 3× tectonic. 10 pages (main text ends p.8).
**Completeness:** **zero** TODO-class markers. 11 evidence rows (C1–C11 incl.
C6b). `figures/figure-gen.py` generates all 3 figures **and every table cell**
(`figures/tables_generated.tex`) with md5 assertions — no hand-entered numbers,
and `tables_generated.tex` is never hand-edited. Final review:
READY-AFTER-CHANGES, and the one required change (two false "at or below
uncapped" claims in §4) **has been applied and verified**. Detector hit its cap
(worst-judge 95/96/93/92/90/92, 11 of 12 verdicts "human-written", zero
mechanical tells found in any round) and was adjudicated **SUBMIT AS-IS**.
**Next steps:** **needs re-homing**, same options as B4. This one has the
best-looking money figure in the portfolio (`fig1_horizon.pdf`) and is a strong
candidate to lead the workshop stack.

## B6. `papers/measurement-ws/` — READY, **venue never chosen**

**Title:** *The Instrument Is the First Suspect: Six Broken Lenses in One
Empirical Program*
**Claim:** Six pre-publication incidents in one measurement-heavy program where
the model appeared to fail and the failure traced to the instrument: a tolerance
calibrated on the wrong key population; a probe reading a layer that causal
state-zeroing later proved inert; an identity-gauge assumption carried across
architectures; an uncentered covariance estimator degenerate on near-orthogonal
targets; a target whose ambient identity block taxed every rank budget; a
cross-check invoked on the transposed side of a state convention. One
five-rule adjudication discipline resolved all six. **The discipline is itself
falsifiable** — its decisive crosscheck survived a pre-registered shuffled-target
control (0.00/0.00/0.05) — **and it does not launder results**: the corrected
lens left the endpoint verdict standing and two genuine model failures intact.
**Venue: UNCHOSEN.** Intended for "a NeurIPS 2026 measurement / evaluation /
science-of-DL workshop, to be selected from the accepted-workshop list" — a list
that was 404 at both fetch attempts and, per the repo, was never checked
afterward. Backup on record: MOSS @ COLM (would need a COLM template retarget
and a cut to 4pp). Format assumptions are flagged **UNVERIFIED — cache
fallback**: 4pp main text (the strictest plausible bar), NeurIPS 2025 kit as
stand-in, double-blind and non-archival assumed.
**Build:** `make` → 3× tectonic. 8 pages. Uniquely has a standalone
`flatten_bundle.py` that also **strips unescaped `%` comments** so source-only
annotations never reach a source upload.
**Completeness:** all 10 sections written; zero TODO/FIXME in prose. **20
evidence rows** (I1–I6, W1–W5, X1–X5, B1–B3, N1). Detector round 2 read 97%/97%
unqualified human and the gate was signed off as DISCHARGED. Recorded status:
"SUBMISSION-READY pending venue confirmation + PI stamps."
**A conditional worth knowing:** a FIX-4 reversal is recorded — `tab:catalogue`
currently sits in the appendix under the 4pp working limit; **if the confirmed
CFP allows ≥5pp main text, move it back into the body.**
**Next steps:** **this is the one that most needs your judgment.** The venue was
never picked. Pick one, confirm its format, apply the FIX-4 reversal if the page
limit allows, and ship. It is a genuinely good methods paper and it is sitting
finished with nowhere to go.

## B7. `papers/reasoning-null-moss/` — READY, blocked on a PI email

**Title:** *Three Bounds on a Null: Testing the Link Between Fast-Weight Write
Geometry and In-Context Composition*
**Claim:** In DeltaNet-family LMs from 14M to 1.31B, a pre-registered instrument
battery finds no evidence that write geometry predicts or causally improves
in-context multi-hop composition — and the null is bounded three ways.
(1) A composition readout deployed three structurally different ways returned
exactly zero at 366/366 readings; a pre-submission positive control then caught
a **state-layout transpose defect**, and after the fix the 320-reading sub-grid
no longer reads zero — **but two pre-registered correspondence nulls reproduce
the signal at every one of those readings**, so the valid claim is
"null-indistinguishable," not "reads zero." (2) A behavioral contrast bounds any
causal effect below the 3-seed detection floor of 1.5–1.7 loss units. (3) A
12-seed replication of the single transient was stopped by its pre-registered
batch-effect gate (variance ratio 4.47 vs cutoff 4.0) and the new cohort's CI
spans zero. Each bound names what would overturn it.
**Venue:** MOSS @ COLM 2026, Small-Scale Frontier Track — 4pp, non-archival,
models ≤3B (this paper: 14M–1.31B, in scope). **Submission was Jul 3 2026 AoE;
notification Jul 24 2026. Both passed.**
**⚠ Recorded blocker, verbatim from `venue-requirements.md`:** entry is via the
capacity-gated late window, and "the required first step is a **late-add email
to the organizers** — that email is a PI decision and is NOT sent by this paper
run." **There is no record that it was ever sent.** Given the dates, it is
almost certainly moot now.
**Build:** `make` → 3× tectonic. 10 pages. ⚠ The 8 `sections/*.md` files are
**stale pre-LaTeX drafts, not sources** — the build consumes `sections/*.tex`,
which have diverged substantially. Ignore the `.md` files or delete them.
**Completeness:** two full gauntlet rounds (the second triggered by the positive
control *failing after* round-1 sign-off — an unusually good process receipt).
Round-2 final review: SUBMISSION-READY. The brief carries an explicit
**"Claim-shape correction (2026-07-11)"** declaring the old "reads exactly zero"
claim dead for the re-verified readings, and discloses that waves 3–4 (46 of 366
readings) were **not** independently re-verified — "an open item, not silently
extended by analogy." Keep that disclosure.
**⚠ Its backup venue is also gone.** Live-verified 2026-08-24
(`EXPERIMENT_LOG.md` 2026-08-24 #3): **ICBINB 2026 pivoted to biology**, so the
on-record backup for this paper is **invalid**. A carefully-bounded null now has
no identified home at all.
**Next steps:** **needs re-homing, from scratch.** Candidates to verify live: a
NeurIPS 2026 workshop off the 102-item accepted list with a negative-results or
science-of-DL angle; a future ICBINB instance if it returns to ML; or arXiv-only,
which for a bounded null is a perfectly respectable outcome and takes the
priority date now.

## B8. `papers/kwall/` — DRAFT, the least finished tree

**Title:** *The K-Axis Closes at 32 — And Why: Diagnosing a Spectrum-Blind Write
Objective in a Fast-Weight Composition Reader*
**Claim:** NCR writes relation operators into a d×d state from context and reads
depth-h composition via O(log h) repeated squaring. The load-bearing claim is
**query-time access complexity**, explicitly *not* a new state-tracking
expressivity result. At K=8 it holds exact recovery to a pre-registered
separation depth h\*=61 (median 1.0 vs the best O(h) baseline's 0.158). At K=32
the recipe closes: 24 cells, far-depth recovery exactly 0.0000 in every one.
Diagnosis: a shallow-depth cosine write loss constrains direction but not
conditioning, so the trained operator becomes non-normal (0.055–0.063) and
ill-conditioned (cond# 321–2952 vs ≈1 healthy). A no-retraining polar projection
extends surviving depth from h≈6 to h≈27–51 — the wall is a property of the
**objective**, not architectural capacity. The pre-registered fix experiment
landed **FAIL** on both parts, with an independent post-FAIL re-audit closing
MECHANISM-CONFIRMED.
**Venue:** MOSS @ COLM 2026, Small-Scale Frontier Track (all models <200K
params). The brief itself says the CFP status was **carried forward, not
re-fetched**, and asks for re-verification before submission. It is listed in
neither `VENUE_MAP.md` nor `SUBMISSION_PACKAGE.md` (both predate the tree).
**Build:** `make` → 3× tectonic. **13 pages against a 4pp cap.**
**⚠ This tree is materially less finished than the others:**
- **13pp vs a 4pp target** — a real ~70% editorial cut, not a formatting pass.
- **No figures at all** (two LaTeX tables instead). No `figure-gen.py`.
- **The anonymization grep was never run** — spot-checked by eye only.
- **`bundle/` has never been built** (the make target exists; the directory
  doesn't).
- **`main.pdf` is the only main PDF in `papers/` not committed to git.**
- Two live `\textbf{[NEEDS-FINAL-SPOT-CHECK: …]}` markers remain in the body
  (`02_background.tex`, `07_related.tex`) on the Nichani/Lee/Bietti argmax
  remark, mirrored in `refs.bib`.
- `refs.bib` has **surname-only entries** (given names were never invented —
  correct behavior, but they need filling), and two cited works (Log-Linear
  Attention, HOLA) have no bib entry at all and are cited by inline arXiv URL.
- **An unmet external obligation:** `research/novelty-gate-2026-07-27.md` §5
  item 3 requires kwall's ortho sections to cite and engage
  arXiv:2607.19390 (a read-time NS-scaffold skeptic result). That gate was filed
  **11 days after** the tree's last commit and has not been applied.
**Why it still matters:** it is the direct ancestor of the flagship's NCR
chapters, and `STATE.md` names it as a ship target. It is the honest record of
the K-wall that the breadth chapter later crossed by a different route.
**Next steps:** decide whether it survives as a standalone paper at all, or gets
absorbed as a flagship appendix / a related-work paragraph. If standalone: cut
to 4pp, build figures, fill the bib, run the anonymization grep, discharge the
novelty-gate obligation, re-verify a venue.

---

# C. GEN 1 — `matrix-thinking/submissions/`

## C1. `icml-mi-workshop-2026/` — ✅ PUBLISHED (reference only)

**Title:** *The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on
ProsQA*
**Venue:** ICML 2026 Mechanistic Interpretability Workshop, Seoul, 2026-07-10.
**Accept (Virtual Poster)**, ~44% acceptance, both reviewers accept.
**OpenReview:** `https://openreview.net/forum?id=Spof4PusVI`, submission #572.
**Code:** `https://github.com/saml212/matrix-codi-rank-blindness`
**Archival:** **non-archival** — virtual posters don't go to PMLR, so arXiv may
freely differ or extend.
**Claim:** A matrix latent bolted onto a vector-pretrained CODI model is
rank-blind: rank-k ablation curves are flat to within 0.6pp across four training
regimes; 3-seed replication gives 81.5 ± 1.2pp accuracy while final effective
rank of Z spans {4, 12, 13}. Four alternative readouts all stay flat. **And the
honest self-limitation:** a GPT-2-SFT negative control with no matrix bottleneck
reproduces the same flat curve, so the rank-k ablation alone conflates
rank-blindness with position-irrelevance.
**Build:** `make` → pdflatex → bibtex → pdflatex ×2. 9 pages. (The only tree
using pdflatex rather than tectonic.) Note `icml2026.sty` was deliberately
patched at `\ICML@appearing` to the workshop footer string.

**Two open items — both actionable, both yours if you want them:**

1. **arXiv was never posted.** The package is built and clean-room compiled at
   `~/Desktop/arxiv-572-submission.tar.gz` (~120 KB, 9pp). Metadata staged:
   cs.LG primary, cs.CL/cs.AI cross-list, CC BY 4.0. **The blocker is
   endorsement** — Sam has no prior arXiv paper, so cs.LG requires an endorser.
   See `SUBMISSION_PLAYBOOK.md` §1. **If you have arXiv standing, this unblocks
   in one step and is the single cheapest win in the portfolio.**
2. **Two inconsistencies the tree's own README flags.** (a) §6 quotes
   matrix-CODI at 82.03% while §4's γ=0 scale sweep puts it ~1.3pp below the SFT
   baseline — treated as seed variance (§3 spread 80.47–82.81) but "confirm this
   is seed variance, not two different configs, before any arXiv revision."
   (b) **Author-name split:** OpenReview #572 is registered "Samuel Larson /
   Pebble Machine Learning"; the arXiv package says "Sam Larson / Pebble AI /
   sam@pebbleml.com". Reconcile before upload — this matters for citation
   indexing.

## C2. `matrix-thinking/submissions/iclr-2027/` — the attractor draft. **FROZEN, and the portfolio's biggest open question**

**Title:** *Rank Is Recruited, Exactness Is Not: Diagnosing and Fixing a
Write-Geometry Attractor in a Real-Data Fast-Weight Language Model*
**Status:** frozen 2026-07-17. Content draft-complete, ~17,084 words, 13
figures, 11 body sections + 2 appendices. `NARRATIVE.md` is 1,841 lines through
revision round 10.
**Not superseded by anything.** No Gen-2 tree claims it as a base.

**Claim:** Rank *recruitment* and rank *exactness* are different achievements. A
DeltaNet trained end-to-end on real GPT-2-tokenized language reliably recruits
the provably-necessary rank at every K, but per-binding exactness decays sharply
with K and compounds geometrically (~ε^h) under composition. Traced to one
measurable cause: a **non-orthonormal write-time key attractor that every input
geometry converges to** (learned, frozen-orthonormal, frozen-real-embedding, and
Gram-matched all converge to the same attractor — ruling out value geometry,
NCE-crowding, and kernel precision). A surgical existence proof (hand-built
per-identity orthonormal key pin) hits rec@0.9 = 1.00/1.00/1.00 at h=1/2/3, K=32,
vs learned 0.78/0.26/0.05. The fix — per-episode Newton–Schulz key
orthogonalization at the write site — lifts K=16 four-hop recovery from 0.42–0.47
to 0.95–1.00 and improves K=32 ~45×, **but narrowly misses its pre-registered
≥0.5 bar (mean 0.44, one seed at the line)** — foregrounded, never spun, with the
cause pre-measured.

**How it differs from the flagship** (this is the crux of open question 3 above):

| | `iclr-2027/` | `papers/flagship/` |
|---|---|---|
| Shape | **Deep** — one architecture, one phenomenon, traced to one cause and fixed | **Wide** — three independent legs across two architecture families |
| Rank law on permutation groups | absent | §3, primary |
| Capability separation vs matched baselines | absent | §4, primary (flagship-exclusive) |
| Attractor scaling ladder 0.248→0.455 | Fig 9 / §9–§10 | §5.1 — **THE OVERLAP** |
| qk-norm / gating 2×2 | folded in §06/§09 | §5.2 — also overlap |
| Frozen-bias fix-at-scale non-transfer | queued fold into §07/§09 | §5.3 — also overlap |
| Exactness decay with K, ε^h compounding | **core, §4** | absent |
| The Newton–Schulz fix and the 0.44-vs-0.5 miss | **core, §7–§8** | absent |
| Key-anchoring 4-wave arc | **core** | absent |
| Capacity cliff located/dissolved/exonerated | §4 | absent |

**Blocked on toolchain, not science.** `main.tex` calls
`\usepackage[preprint]{iclr2027_conference}` — **that .sty does not exist and is
not in the tree**, so there is no `main.pdf` and no verified page count. It
compiles clean at ~24pp against a stub .sty. Remaining work per
`SHIP_PUNCHLIST.md`, all deferred to a TeX-capable machine — **which this
machine now is** (tectonic is installed and working; the "dylib-broken"
note is stale):
1. Compile and get a real page count. 12,079 body words very likely exceeds
   ICLR's limit → expect a compression pass.
2. Trim to the venue limit.
3. **Strip ~49 internal design-doc filenames from LaTeX source comments** before
   any arXiv *source* upload (or upload PDF-only).
4. A final honest-disclosure read on the 0.44-vs-0.5 framing.
Cosmetic: `sections/00_abstract.tex` still opens with a "placeholder" comment
though the content is real (~180 words); `main.tex` still carries the
placeholder-.sty comment block.
**Estimate on record:** 4–6 agent-days, **zero GPU-h**.
**Next steps:** settle absorb-vs-separate with Sam. If SEPARATE, this is a real
second full paper that is ~85% done and needs a venue that isn't ICLR 2027 (or
is, if the flagship goes elsewhere). If ABSORB, harvest its §4/§7–§8 into the
flagship and retire the tree.

## C3. `matrix-thinking/submissions/neurips-ws-2026/` — SUPERSEDED (but read its venue file)

13-page long-form draft: *When the Gradient Sees Rank*. Superseded by
`papers/rank-recruitment-ws/` (page-cut Strategy B) and `papers/neurreps-ea/`
(Strategy A). Build: `make` / `make anon` → tectonic; dual anonymized/named
builds from one source via `\providecommand{\anon}{0}`.
**Worth opening for one file:** `VENUE_DECISION.md` (189 lines) is the richest
venue-intelligence document in the repo. It ranks candidate venues with
reasoning, records which are archival (**NeSy 2026 is PMLR-archival — AVOID**),
documents the arXiv endorsement requirement, and contains **five drafted
late-add emails** with per-venue one-liners, all PI-gated on author/title. Some
of those emails may still be useful for re-homing B4/B5/B7.

## C4. `matrix-thinking/submissions/workshop-2026/` — SUPERSEDED

6-page capacity-trilogy draft. Superseded by `papers/capacity-colm-er/`, which
**materially advances the science**: the Gen-1 draft left "raw state capacity"
as an unadjudicated candidate; the Gen-2 tree resolves it with the super-linear
d^1.97 law. Do not submit this version.
**One useful artifact:** it documents a real LaTeX trap — a literal
`$[0.5385, 0.5513]$` inside `\twocolumn[...]` breaks the bracket scanner (the
`]` closes the optional argument early regardless of `\left`/`\right` or brace
grouping). Fix: `\lbrack`/`\rbrack`, **inside the `\twocolumn[...]` block only**.

## C5. `matrix-thinking/submissions/measurement-2026/` — SUPERSEDED

5-page single-file draft: *The Cliff That Wasn't*. Superseded by
`papers/measurement-ws/`, which subsumes it as **Case I** and extends to six
incidents. No figures, no Makefile.

## C6. `matrix-thinking/submissions/capability-ws-2026/` — CONSUMED

A single 128-line file, `RANK_LAW_SKELETON.md`. No .tex, no figures, no PDF.
Its content became `papers/neurreps-ea/`, `papers/unireps-ea/`, and the
flagship's §3. Nothing to do. Read it only as a compact summary of the rank-law
trilogy — it is the clearest single statement of that arc in the repo.

---

# D. Portfolio at a glance

| # | Tree | Status | Venue of record | Next step |
|---|---|---|---|---|
| A1 | `papers/flagship/` | **LIVE**, 26pp, detector cap-hit; 1.31B third scale point landing | arXiv → ICLR 2027 (CFP unverified) | Rewrite; fix a 23pp-vs-9pp page budget; 6 open questions; hold the 1.31B tables for final harvest |
| B1 | `papers/neurreps-ea/` | ACCEPT-READY | NeurReps 2026 EA — **closed Aug 24 AoE** | **Confirm with Sam whether submitted; else re-home.** arXiv zip is built |
| B2 | `papers/unireps-ea/` | ACCEPT-READY | **UniReps 2026 does not exist** | **Re-home** (pair with B1 if B1 is also homeless). arXiv zip is built |
| B3 | `papers/rank-recruitment-ws/` | READY | NeurReps 2026 EA — **closed Aug 24 AoE** | **Confirm with Sam whether submitted; else re-home.** PI title stamp |
| B4 | `papers/capacity-colm-er/` | SUBMISSION-READY, zero TODOs | Efficient Reasoning @ COLM — **passed** | **Re-home** |
| B5 | `papers/mstar-colm-er/` | READY, detector adjudicated | Efficient Reasoning @ COLM — **passed** | **Re-home** |
| B6 | `papers/measurement-ws/` | SUBMISSION-READY | **NONE — never chosen** | **Pick a venue.** Best orphan in the set |
| B7 | `papers/reasoning-null-moss/` | SUBMISSION-READY | MOSS @ COLM late window — **passed**, email never sent | **Re-home from scratch** — the ICBINB backup died too (pivoted to biology) |
| B8 | `papers/kwall/` | DRAFT, 13pp vs 4pp cap | MOSS @ COLM (stale) | Decide: cut to 4pp, absorb, or retire |
| C1 | `matrix-thinking/submissions/icml-mi-workshop-2026/` | ✅ **PUBLISHED** | ICML 2026 MI Workshop | **arXiv (blocked on endorsement)**; reconcile author name |
| C2 | `matrix-thinking/submissions/iclr-2027/` | FROZEN, ~85% done | ICLR 2027 — conflicts with A1 | **Settle absorb-vs-separate**; compile it (tectonic works now) |
| C3 | `.../neurips-ws-2026/` | superseded | — | Read `VENUE_DECISION.md` |
| C4 | `.../workshop-2026/` | superseded | — | none |
| C5 | `.../measurement-2026/` | superseded | — | none |
| C6 | `.../capability-ws-2026/` | consumed | — | none |

**Universal across every unsubmitted tree:** the author block is
`Author Name(s) TBD`, the title carries a `PENDING-USER` comment even where a
title was adjudicated in `papers/SUBMISSION_PACKAGE.md`, and the official venue
`.sty` is a stand-in. None of these is a research blocker. All are decisions.

---

# E. The existing venue and submission notes, consolidated

These four files are the venue record. Everything above is drawn from them plus
each tree's own `venue-requirements.md`.

**`papers/VENUE_MAP.md`** (2026-07-10, 284 lines) — the venue scout. Six
per-paper assignments, each with backup and risk. A 12-row live-evidence ledger
where every claim carries a source URL and a `[LIVE 2026-07-10]` or
`[PROJECTED]` tag. A conflicts-and-interplay section (papers 2+4 both to
Efficient Reasoning — allowed; papers 3+5 both to MOSS — allowed if capacity
permits; all sibling carriers non-archival so none compromises ICLR
eligibility). Two explicitly flagged open questions: the flagship/iclr-2027
absorb decision, and paper 3's entire venue assignment. **And a 4-item refresh
checklist for when the NeurIPS accepted-workshop list lands.** Status as of
2026-08-25: **the list has landed (102 accepted workshops) and was consulted
once**, on 2026-08-24, but only to answer two targeted questions — it confirmed
NeurReps' real Aug 24 deadline and established that UniReps is absent. **The full
4-item refresh, and any end-to-end scan for re-homing candidates, remains
unrun.**

**`papers/SUBMISSION_PACKAGE.md`** (2026-07-10, 195 lines) — the submission
dossier for **only** `neurreps-ea` and `unireps-ea`. Final titles (both
Candidate A), decisions of record (**Authors: Sam Larson, Pebble AI — solo;
corresponding samlarson16@gmail.com; arXiv CC BY 4.0**), per-paper compliance
checklists all ticked, the three-pass build note, and the human-only remaining
steps. The other seven trees have no equivalent — their status lives in
`bundle/README.md` and in git commit messages.

**`matrix-thinking/submissions/PAPER_SPRINT_PLAN.md`** (2026-07-07, 578 lines) —
a campaign-closing planning memo, self-described as "a plan to execute, not an
executed plan." Its §1 page-cut strategies (A/B/C) are still the best guidance
in the repo for cutting a long draft, and its mechanical rule is worth adopting
verbatim: *"every sentence surviving the cut should be a subset (possibly
re-ordered) of what already exists"* — because new prose written under page
pressure reintroduces unverified claims. **Stale in two ways:** `NARRATIVE.md`
has gone from round 5 to round 10, and the entire Gen-2 generation postdates it.

**Each tree's own `venue-requirements.md`** — Stage-0 artifacts of the `paper`
skill. Every requirement carries its source URL, fetch date, and a verification
tag. Where a live fetch failed, the file says **"UNVERIFIED — cache fallback"**
and names the sanctioned stand-in rather than guessing. That discipline is
worth keeping when you refresh them.

---

# F. The `paper` skill conventions this repo follows

The portable `paper` skill lives at
`/Users/samuellarson/Pebble/github/rockie/platform-skills/skills/paper`. This
repo uses its **repo mode** (`references/repo-mode.md`). You don't have to run
the skill, but knowing the conventions explains why the trees look the way they
do.

- **Sources in, artifacts out.** Design docs, `EXPERIMENT_LOG.md`, and results
  JSONs are read-only sources. The skill never modifies them; it only writes
  under `papers/<slug>/`.
- **The tree layout** you see everywhere: `venue-requirements.md` (Stage 0) →
  `brief.md` (Stage 1) → `outline.md` (Stage 2) → `sections/` (Stage 3) →
  `figures/figure-gen.py` → `gauntlet/round-N/` (01 attack, 02 defense, 04
  rebuttal — numbered 04 but run third, 03 style, 05 format, 06 render
  inspection) → `detector/round-N.md` → `bundle/`.
- **Three-field evidence rows.** Every numerical claim maps to a row carrying
  the pre-registered verdict record, the raw artifact path + md5, and the figure
  or table. "A number with no raw artifact is a CRITICAL finding, not a claim."
- **Commits are the persistence events.** One logical commit per stage or round;
  `git log -- papers/<slug>/` reconstructs the run. Don't squash.
- **Fresh-context review agents.** Every gauntlet stage and every detector judge
  runs with no memory of prior rounds — a judge that remembers its own prior
  verdict grades softly.
- **The render inspector** is a distinct final gate after the detector passes:
  page-by-page visual inspection of the compiled PDF, not a source read. It is
  what caught the figure-float and legibility defects across several trees.
- **Never fabricate.** No invented citations, no invented numbers, no fabricated
  canonical URL. This one has teeth here — **two fabricated bibliography entries
  were caught by API verification** in `neurreps-ea` and again in
  `capacity-colm-er`. Verify every citation against the live arXiv API, never
  from memory.
