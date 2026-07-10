# Venue requirements — measurement-ws (instrument-methodology paper)

Stage 0 of the `paper` skill (repo mode). Fetched live 2026-07-10 by this
drafting session; every load-bearing claim below carries its evidence URL
and a verification tag. **The exact venue CANNOT be finalized today** —
this is the one assignment `papers/VENUE_MAP.md` (commit 52eca3a,
live-verified 2026-07-10) flags as pending the NeurIPS 2026
accepted-workshop list, due tomorrow.

## Target venue (primary, PENDING)

- **Name:** a NeurIPS 2026 measurement / evaluation / science-of-deep-
  learning-class workshop, to be selected from the accepted-workshop list.
- **Status [LIVE 2026-07-10, this session's own fetch]:**
  `https://neurips.cc/Conferences/2026/Workshops` returns **404** — the
  accepted list is not posted. `https://neurips.cc/Conferences/2026/Dates`
  reads: workshop acceptance notifications **Jul 11 '26 (AoE)**; suggested
  submission date for workshop contributions **Aug 29 '26 (AoE)**;
  mandatory accept/reject notification **Sep 29 '26 (AoE)**; workshops
  Dec 11–12. (`papers/VENUE_MAP.md`'s same-day fetch of
  `https://neurips.cc/Conferences/2026/CallForWorkshops` adds the
  multi-site detail: Sydney Dec 11–12, Paris/Atlanta Dec 12–13.)
- **Selection precedent:** the 2025 roster carried recurring
  evaluation/benchmarking and science-of-DL workshops
  (`papers/VENUE_MAP.md` assignment 3), making a matching 2026 slot
  likely but not confirmable today.
- **Refresh obligation:** when the Jul-11 list lands, run
  `papers/VENUE_MAP.md` § "Refresh checklist" item 2 (enumerate the list
  for a measurement/evaluation/science-of-DL slot), pull the chosen
  workshop's own CFP, and re-verify every assumption below against it
  before submission. Until then, every format field in this file is a
  working assumption, not a venue fact.

## Backup venue (live now)

- **MOSS @ COLM 2026, late window.** Deadline was Jul 3 AoE; the CFP
  states "later submissions depending on capacity"
  [LIVE 2026-07-10 per `papers/VENUE_MAP.md`:
  `https://sites.google.com/view/moss-colm-2026/call-for-papers`].
  Small-Scale Frontier Track: **4pp max main content**, models ≤3B params
  (ours: ≤15M-param testbeds, far under), non-archival, no official
  proceedings, dual submission OK. COLM format would require a template
  retarget from the NeurIPS stand-in below.

## Format assumptions (UNVERIFIED — stand-in, flagged per Stage 0 rule d)

- **Page limit: 4 pages main text, excluding references and appendix.**
  Basis: the strictest plausible bar across the candidate set — MOSS
  backup is 4pp [LIVE]; the sibling EA venues (NeurReps, UniReps) are
  4pp excl. refs/appendix; NeurIPS-workshop short-paper tracks
  conventionally sit at 4–5pp. Drafting to 4pp keeps every candidate
  reachable without a cut.
- **Template: NeurIPS 2025 kit as the sanctioned stand-in.** The live
  fetch cannot resolve the actual venue (it does not exist yet), so per
  Stage 0 rule d this is an explicit **`UNVERIFIED — cache fallback`**:
  the kit is `neurips.sty` (April 2025 revision), taken from the
  in-repo precedent copy `papers/unireps-ea/neurips.sty`, byte-identical
  (md5 `aa8b84b0e1cfdd9d2dff0cc280140d5d`) to the `ml-paper-writing`
  skill's template-library copy
  (`skills/ml-paper-writing/templates/neurips2025/neurips.sty`, logged
  there as "current release" for NeurIPS 2025, "confirm against the
  official style files when NeurIPS 2026 opens"). The stand-in is
  sanctioned by the coordinating task brief and by the target being a
  NeurIPS workshop; it is NOT sanctioned by the (nonexistent) venue page
  — swap to the chosen workshop's own kit on the refresh pass.
- **Review style: double-blind assumed.** Both the 2025-pattern
  measurement workshops and the MOSS backup review double-blind; the
  submission build is anonymized and the anonymization grep gates it.
  Re-verify on the chosen CFP.
- **Archival: non-archival assumed** (2025-pattern for workshop tracks;
  MOSS backup confirmed non-archival [LIVE]). Nothing in this paper may
  burn the ICLR 2027 flagship regardless; the flagship carries none of
  this paper's evidence rows as claims.
- **Deadline: Aug 29 '26 AoE (suggested)** [LIVE, this session's fetch of
  the NeurIPS Dates page]; the chosen workshop may set an earlier one.

## Anonymization surface (for the grep; styleguide § anonymization)

Author/handle/org tokens: `Larson`, `Sam Larson`, `samlarson16`,
`pebble`, `pebbleml`, `idastone`, `Anthropic`. URL/acknowledgment
patterns (`github.com/`, `huggingface.co/`, `acknowledg`, `funded by`,
`self-funded`) apply as always. Repo-identifying strings
(`learned-representations`, `matrix-thinking`, `KEY_ANCHORING`,
`CAPABILITY_SEPARATION`, `HEAD_TO_HEAD`) must not appear in the
anonymized build's prose; internal section anchors (e.g. "S15.22") are
cited in source comments only, never in rendered text.

## Provenance of this file

- Fetched 2026-07-10 by the measurement-ws drafting session (WebFetch,
  neurips.cc, two pages, results quoted above).
- Cross-checked against `papers/VENUE_MAP.md` @ 52eca3a (same-day venue
  scout; no deltas found on the shared claims).
- Template provenance chain: `papers/unireps-ea/neurips.sty` ==
  `ml-paper-writing/templates/neurips2025/neurips.sty`, md5 above.
