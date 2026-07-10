# NeurReps 2026 Extended Abstract track — venue requirements (Stage 0)

Live-fetched 2026-07-10 from https://www.neurreps.org/call-for-papers.
The page currently serves the **NeurIPS 2025 edition CFP**; the 2026 CFP
is expected after the NeurIPS 2026 accepted-workshop list drops on
2026-07-11 (per https://neurips.cc/Conferences/2026/Dates, live-verified
in `papers/VENUE_MAP.md`, commit 52eca3a, same day). **Every item below
is therefore a 2025-pattern PROJECTION and must be re-verified against
the live 2026 CFP before submission.**

## Requirements (2025 CFP language, fetched live 2026-07-10)

- **Track:** Extended Abstract. Verbatim: "Extended abstracts may be up
  to 4 pages long, excluding references and appendices."
- **Archival:** EA track is NON-archival (not included in PMLR; may be
  posted to arXiv). Proceedings track (9pp) is archival PMLR — not this
  submission's track.
- **Dual submission:** verbatim, fetched live 2026-07-10: "There are no
  restrictions on Extended Abstract submissions." (The 30%-new-material
  rule applies to the Proceedings track only.) ICLR-2027-flagship-safe.
- **Template (REQUIRED, official):** "All submissions must use the
  NeurReps 2025 LaTeX style files" — the NeurReps style zip (Google
  Drive id `12VSgMuQ-QH0V6sGl4KKnsXoJkkdfsCQR`, linked from the CFP), a
  JMLR/PMLR-based kit. `jmlr.cls` + `jmlrutils.sty` in this tree are
  copied verbatim from the sibling `papers/neurreps-ea/` tree, which
  vendored them from that zip (see its `VENUE_REQUIREMENTS.md`).
  Per the kit's INSTRUCTIONS.txt:
  - EA track: `\documentclass[mlabstract,onecolumn]{jmlr}`
  - use only the auto-loaded packages (amsmath, amssymb, natbib,
    graphicx, url, algorithm2e) plus the sample's own booktabs
  - references in a `.bib` file
  - the submission must be a **single `.tex` file** (this tree keeps
    modular `sections/` for drafting; `bundle/` holds the flattened
    single-file submission tex)
  - manuscript, data, and code anonymized during review — **no author
    block at all** in the review build; camera-ready adds it
  - keep the draftwatermark block (stamps the track); remove only at
    camera-ready
- **Review:** double-blind, OpenReview, >= 3 reviews per submission.
- **Deadline (2025 reference):** Aug 29 2025 -> Sep 4 2025 extended,
  AoE; 2026 PROJECTED similar (late August). NeurIPS 2026 suggested
  workshop-paper deadline: Aug 29 2026 AoE (live-verified in
  `papers/VENUE_MAP.md`).
- **Submission platform:** OpenReview portal.
- **Contact:** organizers@neurreps.org.

## Re-verify checklist (run when the 2026 CFP lands, expected after 2026-07-11)

1. Confirm NeurReps appears on the NeurIPS 2026 accepted-workshop list
   (https://neurips.cc/Conferences/2026/Workshops).
2. Diff the 2026 CFP against every item above (page limit, template zip,
   track structure, anonymization, deadlines, dual-submission language).
3. Swap the style files if the 2026 zip differs from the vendored 2025 kit.
4. Fill the real deadline into this file and `brief.md`.

## Sibling-submission disclosure

`papers/neurreps-ea/` is a DIFFERENT extended abstract targeting the same
venue (the group-composition causal rank law). This paper is the Task D/E
binding-and-composition program. Headline claims, figures, and tables are
disjoint by design; the overlap-management record is in `brief.md`
(§ "Companion papers and overlap management"). The 2025 CFP places no
restriction on multiple EA submissions; re-check on the 2026 CFP.

## Anonymization surface note

Double-blind review triggers the identity-leak grep (tokens recorded in
`brief.md`). One documented exception: the published companion negative
result (ICML 2026 MI workshop) is cited in third person with its real
bibliography entry, per standard double-blind self-citation practice;
the token `larson` is permitted ONLY inside that one `refs.bib` entry
and its rendered citation, nowhere else in the source or PDF.
