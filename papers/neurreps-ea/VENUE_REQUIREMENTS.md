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
