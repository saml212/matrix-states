# Venue requirements — 2nd Workshop on Efficient Reasoning @ COLM 2026

Stage 0 artifact (paper skill, repo mode). Live-fetched 2026-07-10 by the
paper-writer agent. Every field below carries its evidence URL.

## Venue identity

- **Name:** The 2nd Workshop on Efficient Reasoning (COLM 2026 workshop)
- **Workshop date:** October 9, 2026 (hybrid; COLM workshops day)
- **CFP URL (live-fetched 2026-07-10):**
  https://wdlctc.github.io/efficient-reasoning-2026/
- **Submission site:**
  https://openreview.net/group?id=colmweb.org/COLM/2026/Workshop/Efficient_Reasoning

## Deadlines (from the CFP page, fetched 2026-07-10)

- **Submission deadline: July 19, 2026 (AoE)** — extended from Jul 12 per the
  repo venue scout (`papers/VENUE_MAP.md`, live-verified 2026-07-10; the CFP
  page fetched today shows Jul 19 AoE directly).
- Notification: July 31, 2026 (AoE).

## Format

- **Page limit:** "4-10 pages of main text (references excluded)" — CFP page,
  fetched 2026-07-10.
- **Template:** "All submissions must be a single PDF file in the COLM
  submission format" — CFP page, fetched 2026-07-10. This sentence on the
  workshop's own CFP sanctions the COLM template (Stage 0 rule a satisfied).
- **Official COLM 2026 LaTeX kit:** https://github.com/COLM-org/Template/releases/tag/2026
  — linked from the COLM 2026 CFP (https://colmweb.org/cfp.html, fetched
  2026-07-10). Kit fetched 2026-07-10 as the `2026` release tag archive
  (zip md5 `5cd328a20f52ff350e668d83cc6a5039`); files:
  `colm2026_conference.sty/.tex/.bst/.bib`, `math_commands.tex`,
  `fancyhdr.sty`, `natbib.sty`. Single column.
- COLM main-conference format notes (context only; the workshop's own 4-10pp
  limit governs): 9-page main-text cap at the main conference, unlimited
  references — https://colmweb.org/cfp.html, fetched 2026-07-10.

## Review policy

- **Double-blind.** "All submissions must be anonymized and may not contain
  any identifying information that may violate the double-blind reviewing
  policy" — CFP page, fetched 2026-07-10. COLM CFP adds: "the submission must
  not contain acknowledgments or any link (e.g., github) that would reveal
  authors' identity" (colmweb.org/cfp.html, fetched 2026-07-10).
- Anonymization grep is therefore mandatory (styleguide § anonymization).

## Archival status and dual submission

- **Non-archival.** "The workshop is a non-archival venue and will not have
  official proceedings" — CFP page, fetched 2026-07-10.
- Accepts ongoing/unpublished work and papers under review at the time of
  submission or recently accepted, subject to the other venue's policy — CFP
  page, fetched 2026-07-10.

## Topical fit

The CFP topic list (fetched 2026-07-10) includes "KV-cache compression and
memory management for long-context reasoning" — the exact frame of this
paper (fixed-size fast-weight state vs a growing/capped KV cache on
long-horizon recall).

## Cache cross-check

`skills/ml-paper-writing/templates/` carries a `colm2025/` kit (cache,
reviewed 2026-07-09). The live-fetched official 2026 release supersedes it;
the cache was consulted only to confirm vintage (2025 vs 2026), per Stage 0
rule c. No `UNVERIFIED` flag needed: the live fetch succeeded.
