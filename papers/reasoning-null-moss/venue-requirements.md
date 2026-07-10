# Venue requirements — MOSS @ COLM 2026 (late window)

Stage-0 venue style acquisition per the `paper` skill (`references/method.md`
§ "Stage 0"). Every field below was live-fetched from the venue's own pages on
**2026-07-10** by this paper run; nothing is carried from memory or from a
prior paper's brief. Cross-checked against `papers/VENUE_MAP.md` (commit
`52eca3a`, venue scout 2026-07-10) — no deltas found.

## Venue identity

- **Workshop:** MOSS — Methods and Opportunities at Small Scale, a workshop at
  COLM 2026.
  Evidence: https://sites.google.com/view/moss-colm-2026/home [LIVE 2026-07-10]
- **Workshop date:** October 9, 2026 (COLM 2026 workshop day; location TBD on
  the workshop page). Evidence: home page above; COLM workshop-day date
  corroborated by https://colmweb.org/dates.html [LIVE 2026-07-10, VENUE_MAP
  ledger].
- **Contact:** colm2026-moss-workshop [at] googlegroups [dot] com
  (home page, live-fetched 2026-07-10).
- **Scope (quoted from the home page):** the workshop studies the small-scale
  regime across compute, data, and model size — "to what extent is scale
  necessary, and how far can we push toward smaller settings while maintaining
  competitive performance and enabling scientifically meaningful discoveries?"
  It values "controlled, systematic experimentation with rapid iterations".

## Submission requirements (all from the CFP page)

CFP: https://sites.google.com/view/moss-colm-2026/call-for-papers
[LIVE 2026-07-10]

- **Track:** Small-Scale Frontier Track (the fit for this paper; the other
  track is Free-Tier Colab).
- **Page limit:** **4 pages max main content**, unlimited supplementary
  material. References do not count against the 4 pages (COLM convention;
  the CFP caps "main content").
- **Model/compute caps:** models ≤ 3B parameters; soft training budget
  10^20 FLOPs per single run. (This paper's models are 14M–1.31B; every run
  is far under the FLOP cap.)
- **Template:** COLM 2026 official template, from
  https://github.com/COLM-org/Template/releases/tag/2026 — fetched
  2026-07-10 as `Template-2026/` (tag `2026`, files:
  `colm2026_conference.sty/.tex/.bst/.bib`, `math_commands.tex`,
  `fancyhdr.sty`, `natbib.sty`). Single column.
- **Review:** double-blind, on OpenReview
  (https://openreview.net/group?id=colmweb.org/COLM/2026/Workshop/MOSS).
  Anonymized submission: no author names, affiliations, or identifying
  information. → triggers the anonymization grep
  (`references/styleguide.md`).
- **Archival status:** **non-archival** — CFP quote: "This workshop is
  non-archival and will not have official proceedings."
- **Dual submission:** CFP quote: "Workshop submissions can be submitted to
  other venues... We do not accept submissions that have been accepted for
  publication in other venues with archival proceedings, with the only
  exception being COLM 2026 main conference papers." → flagship-safe (the
  reasoning-link lane is CLOSED and appears in no flagship evidence row, per
  `papers/VENUE_MAP.md` assignment 5).
- **Deadlines:** submission was **July 3, 2026 AoE**; reviews July 6–21;
  notification July 24, 2026. **Late window:** CFP quote: "We will allow
  later submissions depending on capacity."

## SUBMISSION BLOCKER (recorded, not actioned by this run)

The deadline has passed; entry is via the capacity-gated late window. Per
`papers/VENUE_MAP.md` (assignment 5) the required first step is a **late-add
email to the organizers** (colm2026-moss-workshop [at] googlegroups [dot]
com) — **that email is a PI decision and is NOT sent by this paper run.**
This package is prepared so the submission can go out the moment the PI's
email receives a yes. Backup venue recorded in the map: **ICBINB, next
instance** (~Jan–Feb 2027, PROJECTED from the ICLR cadence; ICBINB 2026
already ran at ICLR 2026, deadline Jan 31, 2026 — evidence:
https://sites.google.com/view/icbinb-2026/home [LIVE 2026-07-10, VENUE_MAP
ledger]).

## Provenance summary

| Requirement | Value | Source URL | Fetched |
|---|---|---|---|
| Page limit | 4 pp main content + unlimited supplementary | CFP page | 2026-07-10 |
| Template | COLM 2026 kit, GitHub tag `2026` | github.com/COLM-org/Template/releases/tag/2026 | 2026-07-10 |
| Anonymization | double-blind, fully anonymized | CFP page | 2026-07-10 |
| Archival | non-archival, no proceedings | CFP page | 2026-07-10 |
| Dual submission | allowed (bars archival-accepted work; COLM 2026 main excepted) | CFP page | 2026-07-10 |
| Late window | "later submissions depending on capacity" | CFP page | 2026-07-10 |
| Model cap | ≤3B params, soft 10^20 FLOP/run | CFP page | 2026-07-10 |
| Notification | July 24, 2026 | CFP page | 2026-07-10 |
