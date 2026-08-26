# SUBMISSION PLAYBOOK

The generic motion for getting these papers out the door, plus the specific
decisions nobody has made yet.

**Standing caveat.** Every venue policy below was true when this repo last
checked it — **mid-July 2026 for most of it, with a targeted live re-fetch on
2026-08-24** (`EXPERIMENT_LOG.md` 2026-08-24 #3) that corrected the NeurReps
deadline, established that UniReps 2026 does not exist, and killed the ICBINB
backup — or is generic mechanics. **Policies change and deadlines move. Verify
against the live CFP before you act.** The repo's own convention is good: when a
live fetch fails, write `UNVERIFIED — cache fallback` and name the stand-in
rather than guessing. Keep doing that.

⚠ **The lesson from the Aug-24 correction, stated once because it cost this
portfolio a deadline:** a conference-wide *suggested* workshop deadline is not a
deadline. Individual workshops set their own and they are usually earlier. Always
fetch the workshop's own CFP.

---

## 1. arXiv

### 1.1 Account and endorsement — the actual blocker

Register at arxiv.org with an institutional or personal email. The account is
free and immediate.

**Endorsement is the real gate.** arXiv requires an endorsement before your
first submission to a given archive/category if you have no submission history
there. cs.LG is an endorsement-gated category. An endorser is someone who has
themselves published enough in that category recently; they receive a request
with a code and approve it.

This is a **live, recorded blocker for this program**:
`matrix-thinking/submissions/icml-mi-workshop-2026/ARXIV_PLAN.md` notes that Sam
has no prior arXiv paper, so a personal cs.LG endorser is required, and that is
why the ICML workshop paper — **accepted, presented, and non-archival, so free
to post** — has never gone up.

**If you already have arXiv standing in cs.LG, say so early.** It unblocks the
whole portfolio in one step, and it is a concrete reason your co-authorship is
load-bearing rather than ceremonial. If you don't, the fastest paths are: ask
the ICML workshop reviewers or organizers, ask a collaborator at Berkeley or
Stanford (the PI has been pursuing that collaboration), or use arXiv's own
endorsement-request flow and wait.

**Note:** an endorsement is per-archive. Getting endorsed for cs.LG does not
automatically cover cs.NE or cs.CL, though cross-lists from an endorsed primary
are generally fine.

### 1.2 Categories

The repo's staged metadata, which is a sensible default for all of these:

| Paper | Primary | Cross-list |
|---|---|---|
| `neurreps-ea` | cs.LG | cs.NE |
| `unireps-ea` | cs.LG | cs.NE |
| ICML MI workshop paper | cs.LG | cs.CL, cs.AI |
| flagship | cs.LG | cs.CL and/or cs.NE |

cs.LG primary is right for all of this work. Consider `stat.ML` as an additional
cross-list for the rank-law papers (they are genuinely a statistics-of-learned-
representations result). Don't over-cross-list — two or three is normal, five
reads as spam to moderators.

### 1.3 Licensing

The decision of record for this program is **CC BY 4.0**, recorded in
`papers/SUBMISSION_PACKAGE.md` §3 and in each `arxiv/metadata.md`. Keep it
consistent across the portfolio — mixed licensing across papers from one group
is a small but real annoyance for anyone reusing figures.

The alternative is arXiv's own non-exclusive license, which is more restrictive
for reusers. CC BY 4.0 is the right call for work that wants to be built on.
**The license is irrevocable once a version is announced** — you can change it
for future versions, not past ones.

### 1.4 What to upload

arXiv strongly prefers **TeX source** over a bare PDF, and compiles it itself.
Practical consequences that bite:

- **Include the `.bbl` file.** arXiv does not run BibTeX. Every `arxiv/` package
  in this repo already ships its `.bbl` for exactly this reason, and one commit
  in the history exists solely to force-add `.bbl` files past a `*.bbl` gitignore.
- **Include every `.sty` and `.cls`** the document needs that isn't in arXiv's
  standard TeX distribution (`jmlr.cls`, `jmlrutils.sty`, `neurips.sty`,
  `colm2026_conference.sty`, `iclr2026_conference.sty`, `fancyhdr.sty`,
  `natbib.sty`). The staged packages do this.
- **Figures as PDF**, in the same relative paths the source expects.
- **Strip internal comments before a source upload.** LaTeX source comments ship
  with the source and are readable by anyone. Two known cases here:
  `matrix-thinking/submissions/iclr-2027/` carries **~49 internal design-doc
  filenames** in source comments, and `papers/flagship/latex/main.tex` carries
  the full PI to-do comment block. `papers/measurement-ws/flatten_bundle.py`
  already implements comment-stripping and is worth reusing.
  If stripping is too fiddly for a given paper, **PDF-only upload is allowed**
  and is the safe fallback.

**Verify the compiled preview before you announce.** arXiv shows you its own
build. This matters especially for `neurreps-ea`, which has a documented
**pagination bistability**: a cold build renders 12 mis-paginated pages and only
converges to the correct dense 8-page layout on the third pass. If arXiv's
preview shows 12 pages, that is the transient, not your paper.

### 1.5 The comments field

Free text under the abstract. Use it for: page count and figure count, the venue
if accepted ("Accepted at the ICML 2026 Mechanistic Interpretability Workshop"),
and companion-paper pointers. The staged `metadata.md` files have these written
already. After both EA arXiv IDs exist, a v2 of each can cross-link the other —
noted as optional in the metadata.

### 1.6 Announcement timing and versions

Submissions go through a moderation queue; most clear quickly, but a moderator
can reclassify your category or hold a submission, which adds days. Announcement
runs on a daily cycle with a submission cutoff — a submission just after the
cutoff announces the following cycle, and weekends push to Monday. **Don't
schedule a coordinated announcement for the same day you submit.**

Replacements (v2, v3) are easy and preserve the arXiv ID and the v1 timestamp.
**Withdrawal does not delete anything** — a withdrawn paper still shows its
prior versions. So the cost of posting a v1 that you later improve is low, but
the cost of posting something wrong is permanent. Given that every number here
traces to an archive, the risk is low; take the priority date.

---

## 2. OpenReview

Every workshop venue in play uses OpenReview.

**Profile setup takes longer than you expect.** Create a profile well before any
deadline. OpenReview requires a reasonably complete profile (names including
name variants, emails, affiliation history, and often an ORCID or a DBLP/Google
Scholar link) before it will let you be listed as an author. New profiles are
sometimes held for moderation. **Do this the week you start, not the week you
submit.**

**Author-name consistency matters.** There is already a split in this program:
OpenReview submission #572 is registered "Samuel Larson / Pebble Machine
Learning" while the arXiv package says "Sam Larson / Pebble AI". Pick one form
for each person and use it everywhere — OpenReview merges profiles by name and
email, and a split identity fragments citation and reviewer-assignment history.
Settle your own form (Will Larson / W. Larson) once, up front.

**Double-blind submissions.** Every workshop in play is double-blind. That means:

- No author block, no affiliations, no acknowledgments in the PDF.
- No identifying URLs — code links go through `anonymous.4open.science` or
  equivalent. (Several trees still carry an *unfilled placeholder* anonymous
  link; flagged as non-blocking, but fill them if you can.)
- **Run the anonymization grep.** Each brief has an "Anonymization surface"
  section listing the tokens. The union across the portfolio:
  `Sam Larson`, `Samuel Larson`, `samlarson16`, `samuellarson`, `pebble`,
  `pebbleml`, `idastone`, `Anthropic`, plus repo strings
  `learned-representations`, `matrix-states`, `matrix-thinking`,
  `KEY_ANCHORING`, `CAPABILITY_SEPARATION`, `HEAD_TO_HEAD`, and
  `youthful-indigo-turkey`. **Add your own name and handles.**
  `papers/kwall/` is the one tree where this grep was never actually run.
- **Watch self-citations.** `neurreps-ea` had a real leak: a self-citation to the
  ICML workshop paper rendered the real author name into the double-blind
  bibliography. It was **cut** for the review build and must be **restored at
  camera-ready**. Check every tree for the same pattern.
- Keep the draftwatermark where the template supplies one (NeurReps requires it
  during review); remove at camera-ready.

**Submission mechanics.** Find the venue group
(`openreview.net/group?id=<venue path>`), hit "Submit", fill title / abstract /
authors / keywords, upload the PDF, answer the venue's questions. Deadlines are
**AoE (Anywhere on Earth, UTC−12)** — that buys you most of an extra day over
US time, but don't rely on it. Most venues allow PDF replacement until the
deadline; some allow abstract edits after. Check.

---

## 3. Camera-ready vs preprint ordering

The plan of record for the flagship is **arXiv preprint first, then the venue
submission.** That ordering is deliberate and generally right — it stakes the
priority date and lets people cite the work while review runs.

Three things to keep straight:

**Preprints and double-blind review.** ICLR (and most ML venues) permit
concurrent preprints and do not treat a public arXiv version as a blinding
violation, but they typically ask that you not *actively advertise* the
submission during the review period, and some venues have a cutoff date before
which a preprint must have been posted. **Read the venue's current policy on
concurrent submissions and preprints before posting.** This is the single most
venue-variable rule in the whole process.

**Archival vs non-archival.** Every workshop in this portfolio is **non-archival**
— no proceedings — which is why none of them compromises the flagship's ICLR
eligibility, and why the ICML MI workshop paper is free to go to arXiv. This was
checked deliberately: `VENUE_MAP.md` records **NeSy 2026 as PMLR-archival —
AVOID**, and that reasoning still applies to any new venue you consider. If you
re-home a paper, **check archival status first**; an archival workshop
acceptance can block a later full-paper submission of the same content.

**Camera-ready is a real work item, not a formality.** Across these trees it
means: uncomment the author block, restore any citation cut for blinding, remove
the draftwatermark, and honor the camera-ready page limit (often one page more
than the review limit — UniReps EA goes 4pp → 5pp). Each `bundle/README.md`
carries the specific checklist.

---

## 4. The per-venue flows in play

| Venue | Platform | Format | Archival | Status as last checked |
|---|---|---|---|---|
| **ICLR 2027** | OpenReview | 9pp main text, ICLR kit, double-blind | **Archival** | CFP **not live** (404 on iclr.cc/Conferences/2027 as of 2026-07-10). Projected abstract Sep 19 / full Sep 24 2026 — **third-party aggregators, never official** |
| **NeurReps 2026 (EA)** | OpenReview, group `NeurIPS.cc/2026/Workshop/NeurReps_Extended_Abstracts` | 4pp excl. refs/appendix, **single .tex**, JMLR kit from the NeurReps style zip, draftwatermark kept | Non-archival, **dual submission unrestricted** (flagship-safe) | **Deadline was AUG 24 2026 AoE — PASSED.** Live-verified 2026-08-24 |
| **UniReps 2026 (EA)** | — | 4pp main text (CR 5pp), NeurIPS kit | — | **DOES NOT EXIST as a 2026 venue.** `unireps.org` says TBA; absent from all 102 accepted NeurIPS workshops. Verified 2026-08-24 |
| **NeurIPS 2026 workshops** (generic) | OpenReview | per workshop | usually non-archival | Accepted list **is published: 102 workshops**. ⚠ The Aug 29 2026 AoE date is only the NeurIPS-wide **suggested** default — **individual workshops set their own, earlier dates** (NeurReps set Aug 24). Never plan against the suggested date. Mandatory notification Sep 29; workshops Dec 11–13 |
| **Efficient Reasoning @ COLM 2026** | OpenReview | 4–10pp main text, COLM 2026 kit (GitHub tag `2026`) | Non-archival | Deadline was **Jul 19 2026 AoE — passed** |
| **MOSS @ COLM 2026** | OpenReview | 4pp main content, models ≤3B params, soft 10²⁰ FLOP cap | Non-archival | Deadline was Jul 3 2026 AoE with a capacity-gated late window; notification Jul 24 — **both passed** |
| **ICBINB** | OpenReview | per instance | non-archival | ⚠ **ICBINB 2026 pivoted to biology** (verified 2026-08-24) — **no longer a valid backup** for the null paper. Last ML instance was an ICLR 2026 workshop |
| **AXIOM / PALM (Paris)** | — | — | — | Both **exist**; CFPs **UNVERIFIED** (the verification agent ran out of budget). Candidate re-homes — fetch live before proposing |
| **TAI-Eval** | — | — | — | **Unchecked.** Named as a candidate, never fetched |

**ICLR-specific mechanics** for the flagship: abstract deadline precedes the
full-paper deadline by several days and is **binding** — you cannot submit a
paper whose abstract wasn't registered. Reviews are public on OpenReview; there
is a rebuttal period; the LaTeX kit changes year to year and **you must not hand-
edit year strings** in a prior year's `.sty` — get the real
`iclr2027` kit from `github.com/ICLR/Master-Template` when it ships. The repo's
current build uses the `iclr2026` kit as the sanctioned stand-in.

---

## 5. ⚠ The calendar reality

The portfolio was frozen 2026-07-17. It is now **2026-08-25**, and a live CFP
re-fetch on 2026-08-24 (`EXPERIMENT_LOG.md` 2026-08-24 #3, the report of record)
closed the last open clock. The honest state:

- **Passed — COLM family.** Efficient Reasoning @ COLM (Jul 19) kills B4 and B5
  at that venue. MOSS @ COLM (Jul 3 + late window; notification Jul 24) kills B7
  and B8 at that venue, and the required late-add email was never sent.
- **Passed — NeurReps 2026 EA, Aug 24 2026 AoE.** ⚠ This is the correction that
  matters most. The repo had been planning against **Aug 29**, which is only the
  **NeurIPS-wide *suggested* default**; NeurReps set its own, earlier date. Two
  ACCEPT-READY papers (**B1 `neurreps-ea`, B3 `rank-recruitment-ws`**) were built
  for exactly that track, both anonymized on the official template, dual
  submission unrestricted. **Whether they went in is a question only Sam can
  answer — ask him in your first conversation.** If they did not, they are
  finished papers that need re-homing.
  **Generalize the lesson:** never plan against a conference-wide *suggested*
  deadline. Individual workshops set their own, and they are usually earlier.
- **Never existed — UniReps 2026.** Not a live venue this year (site says TBA;
  absent from all 102 accepted NeurIPS workshops). **B2 needs a re-home**
  regardless of clocks.
- **Died as a backup — ICBINB 2026**, which pivoted to biology. **B7 now has no
  identified home at all.**
- **Partially run:** the Jul-11 accepted-workshop-list refresh that `VENUE_MAP.md`
  and three `venue-requirements.md` files all schedule. The list **has landed —
  102 accepted workshops** — and was consulted on 2026-08-24, but only to answer
  the two targeted NeurReps/UniReps questions above. **The end-to-end scan has
  still never been done**, and it is now the single highest-value venue action:
  it supplies the venue B6 has been waiting on since July, and re-homing
  candidates for B1, B2, B3, B4, B5 and B7 at once. Candidates already named but
  **unverified**: AXIOM and PALM (Paris) exist with unfetched CFPs; TAI-Eval is
  unchecked.
- **Ahead:** ICLR 2027, projected late September. The flagship's real deadline,
  and now the only forward-looking one in the portfolio.

**Practical reading.** With no live clock left, the short-run exception that used
to outrank the flagship is gone, and **the PI's stated priority — the flagship —
is now simply correct.** Two things still belong in week one because they are
cheap and they unblock everything else: (1) ask Sam whether B1 and B3 were
submitted before Aug 24 closed, and (2) spend an afternoon on the 102-workshop
list producing a re-homing shortlist for the homeless papers. Then the flagship,
which has weeks and needs all of them.

**Submission itself is a PI action** — it requires Sam's OpenReview account.
Prepare the bundles and run the §7 checklist; he presses the button.

---

## 6. The venue-homing decisions nobody has made

Each of these is genuinely open. The repo does not decide them; you and Sam do.

1. **Does the flagship absorb `matrix-thinking/submissions/iclr-2027/`?**
   They cannot both go to ICLR 2027 with the same §5 evidence. Drafting
   assumption of record is SEPARATE, and flagship §5 was built swappable so the
   answer can flip cheaply. **Weigh:** the flagship is already 23 pages of main
   text against a 9-page limit, so absorbing makes that worse, not better;
   whereas the attractor draft is a coherent standalone paper that is ~85% done
   and blocked only on a compile. Separate looks right, but then the attractor
   paper needs its own venue.
2. **Where does `measurement-ws` go?** It has never had a venue. It is finished,
   it passed its detector gate at 97%/97%, and it is a good methods paper. It
   needs a measurement / evaluation / science-of-DL workshop off the NeurIPS
   list, or a retarget to a COLM-style venue. **The most tractable open decision
   in the portfolio.**
3. **Where do `capacity-colm-er` and `mstar-colm-er` go now?** Both were built
   for a COLM workshop whose deadline passed. Both are submission-ready. `mstar`
   has the strongest single figure in the portfolio. Candidates: a NeurIPS
   memory / long-context / efficiency workshop, ICLR 2027 workshops, or
   arXiv-only.
4. **Where does `reasoning-null-moss` go now that ICBINB is gone?** ⚠ Updated
   2026-08-24: **ICBINB 2026 pivoted to biology**, so the on-record backup — the
   natural home for a carefully-bounded null — is **invalid**. This paper now has
   no identified venue at all. Candidates: a negative-results or science-of-DL
   workshop off the 102-item NeurIPS list, a future ICBINB instance if it returns
   to ML, or **arXiv-only**, which for a well-bounded null is a respectable
   outcome and takes the priority date immediately.
5. **Does `kwall` survive as a standalone paper?** It is 13 pages against a 4-page
   cap, has no figures, and has an unmet novelty-gate obligation. Its content is
   the ancestor of the flagship's NCR chapters. Options: cut it hard and ship it
   as a workshop paper, fold it into the flagship as an appendix, or retire it
   with the K-wall story told inside the breadth chapter.
6. ~~**Can NeurReps take two submissions from one group?**~~ **ANSWERED
   2026-08-24: yes — NeurReps EA dual submission is unrestricted.** `neurreps-ea`
   and `rank-recruitment-ws` could both have gone, and the de-dup pass that made
   them non-overlapping companions was the right call. The blocker was never
   permission; it was the **Aug 24 AoE deadline**, which has passed. The live
   question is now **whether Sam submitted them**, and if not, where they both go.
7. **Does the ICML workshop paper go to arXiv, and who posts it?** It is
   accepted, presented, non-archival, and the package is built. The only blocker
   is cs.LG endorsement. **Also reconcile the Samuel-vs-Sam author name before
   upload.**
8. **Author list and order, on every paper.** Currently `[AUTHORS — PI decision
   pending]` everywhere. Your co-authorship is the PI's instruction; order and
   the full list are his call. **Settle before the first arXiv posting** —
   arXiv author lists are painful to change afterward, and inconsistent author
   lists across a portfolio hurt everyone's citation record.
9. **Where do the findings pages live now?** pebbleml.com was repurposed on
   2026-08-19 and the sync workflow has been dead since 2026-07-14, so the
   findings pages are internal documents, not public prior art. If the papers
   want to point at a public artifact, that needs a home. See `DATA_MAP.md` §3.

---

## 7. A pre-submission checklist

Adapted from the `bundle/README.md` checklists that already exist per tree. Run
this for each paper.

- [ ] CFP **re-verified live**, this week, with the URL and date recorded
- [ ] Archival status confirmed (and confirmed not to block a later full paper)
- [ ] Page limit confirmed and **actually measured on the compiled PDF**, not
      assumed
- [ ] Official template in use — never a hand-edited year string
- [ ] Title final
- [ ] Author list and order final; name forms consistent with OpenReview and arXiv
- [ ] Anonymization grep run and clean (including your own name and handles)
- [ ] Self-citations checked for blinding leaks
- [ ] Anonymous code link real, or the placeholder consciously accepted
- [ ] Every bibliography entry verified against the live arXiv API or CrossRef —
      **never from memory** (two fabricated entries have already been caught in
      this portfolio)
- [ ] Every number traced to its archive, one more time, from the raw file
- [ ] Figures regenerated from the md5-asserting generator; the build did not
      warn about a changed archive
- [ ] PDF compiled with the tree's documented number of passes (three, for most)
      and page count verified on the final artifact
- [ ] Rendered PDF read page by page — not just the source
- [ ] Venue's LLM-assistance disclosure policy read and followed
- [ ] Committed, with a commit message naming the artifact and stage
