# NeurReps — what the workshop actually requires (acquired 2026-07-09)

Source: https://www.neurreps.org/call-for-papers (the live page currently
serves the NeurIPS 2025 edition CFP; the 2026 CFP is expected after the
NeurIPS 2026 accepted-workshop list drops on 2026-07-11 — RE-VERIFY every
item below against the 2026 CFP before submitting).

- **Tracks:** Proceedings (9pp, archival, PMLR) and **Extended Abstract
  (4pp excluding references and appendices, NON-archival, may be posted
  to arXiv)** — this submission targets the EA track.
- **Dual submission:** "Extended abstracts have no restrictions" —
  ICLR-2027-flagship-safe.
- **Template (REQUIRED, official):** the NeurReps LaTeX style zip
  (Google Drive id `12VSgMuQ-QH0V6sGl4KKnsXoJkkdfsCQR`, linked from the
  CFP), a JMLR/PMLR-based kit. `jmlr.cls` + `jmlrutils.sty` are copied
  into this tree from that zip verbatim. Per its INSTRUCTIONS.txt:
  - EA track: `\documentclass[mlabstract,onecolumn]{jmlr}`
  - use only the auto-loaded packages (amsmath, amssymb, natbib,
    graphicx, url, algorithm2e) plus the sample's own booktabs
  - references in a `.bib` file
  - **the paper must be a single `.tex` file** (this tree keeps modular
    `sections/` for drafting; `bundle/` holds the flattened single-file
    submission tex)
  - manuscript, data and code anonymized during review (no author
    block at all in the review build; camera-ready adds it)
  - keep the draftwatermark block (stamps the track); remove only at
    camera-ready
- **Review:** double-blind, OpenReview, >= 3 reviews.
- **Deadline (2025 reference):** Aug 29 -> Sep 4 extended, AoE; 2026
  expected similar (late August).

## Venue scout after the missed NeurReps deadline (2026-09-02, web-verified on each venue's own page)

NeurReps 2026 EA/Proceedings closed Aug 24 (extended Aug 28); only its Findings track is open (Sep 8 11:59 UTC) and its scope (experimentalist+theorist neuroscience collaborations) does not fit. Every on-topic NeurIPS 2026 workshop (Interpretability as a Science, ATTRIB, MATH-AI, Interp4Discovery, GDDL, AXIOM, Representations for the Physical Sciences) closed Aug 28–Sep 6; all NeurIPS workshops are non-archival with mandatory Sep 29 notification.

Ranked options for this paper (arXiv v1 posts ~Sep 8 regardless):
1. **UniReps 2026 Extended Abstract track** (NeurIPS Paris; https://unireps.org/2026/call-for-papers): non-archival, 4 pp main text, NeurIPS 2026 style, anonymized; topics list names symmetry/equivariance, identifiability, learning dynamics, mechanistic interpretability. Deadline officially "TBD"; the OpenReview form carries a placeholder duedate 2026-09-25 22:00 UTC but was closed Sep 1. WATCH unireps.org/2026 daily; if it reopens this is the home. Preprint policy unpublished.
2. **ICLR 2027 workshops** (CFPs ~Dec 2026, paper deadlines ~late Jan/early Feb 2027 by the 2026 template): best expected non-archival fit (Sci4DL-type, interpretability, geometry). Nothing to submit to yet.
3. **TMLR** (rolling, ~9-week decisions, short papers welcome, double-blind, arXiv allowed): archival; a later conference version would need substantially new content.
4. ICLR 2027 main (abstract Sep 18, paper Sep 25 AoE, 9 pp main text, reciprocal reviewing): not realistic for a 4-page body without a full-paper expansion; archival.
5. NLDL 2027 EA track (Sep 15 CET, 2 pp, non-archival, in-person Tromsø required): feasible but generic-DL, weak fit.
6. CPAL 2027 Recent Spotlight (non-archival, accepts arXiv work; 2027 not announced, ~Jan 2027 if it runs), AAAI-27 workshops (calls Oct 2, papers ~Nov 20), ESANN 2027 (Nov 25, proceedings).
Unverified: UniReps live deadline/preprint policy; AISTATS 2027 (Dates page Oct 6 AoE, trackers Oct 8, CFP "not announced").
