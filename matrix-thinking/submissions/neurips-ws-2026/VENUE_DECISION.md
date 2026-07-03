# Venue Decision — "When the Gradient Sees Rank..."

**Status: PI decision needed.** Research below is current as of 2026-07-03.
None of this has been acted on (no submissions, no organizer emails sent).

## Page-budget reality check (measured on the current draft, do not skip this)

The current draft compiles to 13 pages: **main body (Sec. 1-9, "Reproducibility")
≈ 10 pages, references ≈ 1.3 pages, appendix ≈ 1.5 pages** (measured directly
from `main.pdf` via `pdftotext`, 2026-07-03 build). Against the candidates
below, most of which count body length excluding refs/appendix:

- **NeurReps/UniReps Extended Abstract (4pp)** — the leading candidate — would
  require cutting the ~10-page body to 4 pages, roughly a 60% cut. That is a
  different, much shorter paper (likely: keep the headline Task D causal
  result and one figure, cut Task E's mechanism section and the frontier
  section to a paragraph each, move everything else to an appendix/arXiv
  version). Not a formatting pass — a real editorial decision the PI should
  make, not something to attempt silently in a later pass.
- **COLM Actionable Interpretability, Long track (9pp excl. refs/appendix)** —
  closed for this cycle, but noted because the fit is otherwise close: only
  ~1 page over. If a late add happens, this is a light trim, not a rewrite.
- **COLM Short track (5pp)** — same drastic-cut problem as the Extended
  Abstract track above.

**Bottom line:** whichever venue is picked, the 4pp/5pp tracks are the
realistic ones (9pp Proceedings-style tracks are either archival or closed),
and none of them fit this draft as-is. Budget real editorial time for a length
cut, not just a template swap, before the actual deadline.

## Constraints driving the search

- We already have a companion negative-result paper accepted at an ICML 2026
  MI workshop (archival status of that acceptance not re-litigated here).
- We intend to write an extended full paper for **ICLR 2027** on this same
  arc (the geo3/exactness/DeltaNet-causal extensions currently out of scope
  for this draft). The venue for *this* workshop paper must not burn that
  novelty claim.
- Two hard requirements: (1) **non-archival** (no formal proceedings /
  copyright transfer — an arXiv-only or extended-abstract track), and (2)
  **dual-submission-friendly** (explicitly permits later submission of an
  extended version elsewhere).
- Draft is ready now (13 pages total, ~10-page body; compiles clean; see the
  page-budget note above) — deadline should be reachable without rushing the
  science, but expect an editorial trim regardless of venue.

## Ranked candidates

| Rank | Venue | Deadline | Page limit | Archival policy | Dual-submission | Fit |
|---|---|---|---|---|---|---|
| **1** | **NeurIPS 2026 — NeurReps or UniReps** (Extended Abstract track) | CFP not yet published. Accepted-workshop list drops ~2026-07-11; individual CFPs historically open shortly after with deadlines late Aug (2025: NeurReps Aug 29→Sep 4 ext.; UniReps Aug 22→Aug 29 ext.). | Extended Abstract: 4pp excl. refs/appendix. (A separate 9pp *Proceedings* track exists but is archival — avoid it.) | Extended Abstract track: **non-archival**, "no restrictions on submissions appearing elsewhere" (2025 CFP language). | Explicitly unrestricted — can arXiv now, submit extended version to ICLR 2027 later. | Strong topical match (rank/geometry of learned representations). Only candidate that is genuinely still *open* for this cycle. Risk: which 2026 instance(s) get accepted, and exact CFP text, aren't confirmed yet. |
| 2 | **Workshop on Actionable Interpretability @ COLM 2026** | **Closed** — extended deadline was 2026-06-24 AOE (9 days before this research). Workshop itself is 2026-10-09, SF; notifications not yet out. | Long 9pp / Short 5pp excl. refs/appendix. | "**All accepted papers are considered non-archival** (so they can be submitted to other venues)." Best-worded policy found. | Explicit: work under review elsewhere is fine; must withdraw only if already *presented* elsewhere before Oct 9. | Best topical fit ("actionable" MI results) and best-worded dual-submission policy of anything surveyed — but administratively closed for this cycle unless organizers grant a late add. Worth a direct email regardless of low odds. |
| 3 | **Mechanistic Interpretability Workshop** (mechinterpworkshop.com) | **Closed for 2026** (hosted at ICML 2026, deadline was 2026-05-08). Alternates host conference (ICML 2024 → NeurIPS 2025 → ICML 2026) — next instance likely **NeurIPS 2027**, too far out. | Short 4–5pp / Long 8–9pp. | "**The workshop is non-archival.**" | Explicitly welcomes work under review elsewhere (e.g. COLM, NeurIPS); only bars work already accepted at an archival venue. | Best topical/brand match overall — bookmark for the 2027 cycle. Not usable for this draft's timeline. |

**Ruled out:** SciForDL moved off NeurIPS to ICLR co-location; its "2nd
edition" already happened (deadline 2026-02-04, held 2026-04-26, Rio) — not
usable. ATTRIB (data/model attribution) has no confirmed 2025/2026 NeurIPS
instance — likely dormant.

## Recommendation

1. **Primary path:** hold the draft ready, watch for NeurIPS 2026 workshop
   CFPs starting ~2026-07-11, and submit to NeurReps or UniReps's Extended
   Abstract track (or an MI-specific workshop if a new one appears in the
   accepted list) the moment its CFP opens. This is the only venue in this
   search that is both open and unambiguously non-archival + dual-submission
   safe.
2. **Parallel, low-cost:** email the Actionable Interpretability @ COLM 2026
   organizers asking about a late add. Best-worded policy and best topical
   fit found in this search; downside is just the time to send one email.
3. Do **not** wait on the Mechanistic Interpretability Workshop — it will
   not have a 2026 second instance on the historical alternation pattern.

**Final venue pick is the PI's call** — in particular whether to gate
submission entirely on NeurIPS 2026 workshop CFPs opening (likely mid-to-late
August deadline) versus accepting the COLM long shot as a hedge for an
earlier date.

## arXiv cs.LG endorsement (if needed before any submission)

Confirmed via arxiv.org's endorsement help page and its Jan 2026 policy
update: automatic endorsement now requires **both** an institutional
academic email **and** prior claimed co-authorship on an existing arXiv
paper in-domain. Lacking either, the fallback is the standard personal
endorsement flow: register → get an endorsement-request link → find an
eligible in-domain endorser (≥1 arXiv cs.LG paper submitted 3 months–5 years
ago, findable via "Which authors of this paper are endorsers?" on any
relevant abstract page) → email them the link individually. No official SLA;
typically resolves in a day or two with a responsive endorser. Note: the
companion ICML 2026 MI-workshop paper (`refs.bib`'s `larson2026gradient`) is
currently cited as a single-author workshop paper with no arXiv ID recorded
— check whether it has already been posted to arXiv (that submission would
itself establish an endorsed account) before starting the endorsement-request
flow from scratch; if it hasn't been posted yet, doing so first is probably
the fastest path to an endorsed account for this paper too.
