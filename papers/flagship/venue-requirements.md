# Venue requirements — flagship full paper (arXiv preprint → ICLR 2027)

Stage-0 artifact of the `paper` skill (repo mode). Every requirement below
carries its source and verification status. Primary acquisition:
`papers/VENUE_MAP.md` (venue scout, live-verified 2026-07-10, commit
52eca3a) — assignment 6 CONFIRMS the PI-pinned venue plan; this file
records the requirements the flagship drafts against.

## Target venue chain

1. **arXiv preprint** (named build), target ~2026-07-31.
2. **ICLR 2027** (anonymized build), submission projected ~Sept 2026.

## ICLR 2027 — status of the live fetch

- **The official ICLR 2027 CFP is NOT live.** `iclr.cc`'s year menu stops
  at 2026 and `/Conferences/2027` returns 404 — [LIVE 2026-07-10, per
  `papers/VENUE_MAP.md` live-evidence ledger].
- **Projected dates (third-party aggregators, NOT official):** abstract
  ~Sep 19, 2026; full paper ~Sep 24, 2026; conference Apr 24–28, 2027
  [PROJECTED — mlciv.com/ai-deadlines et al.; location claims conflict
  between sources and are treated as unknown]. Never trust these for the
  final submission clock; re-verify the moment iclr.cc posts the 2027 CFP.

## Format requirements (drafting basis)

- **Page limit:** 9 pages main text, excluding references and appendix
  (the ICLR 2026 limit; carried as the 2027 drafting basis until the 2027
  CFP lands — flagged, not assumed final).
- **Template:** `UNVERIFIED — cache fallback (sanctioned stand-in).` ICLR
  has not released a 2027 style file. Per
  `skills/ml-paper-writing/templates/VENUE_STATUS.md` (reviewed
  2026-07-09), the `iclr2026/` kit is the ONLY sanctioned stand-in for
  ICLR 2027: draft in `iclr2026/`, add a real `iclr2027/` kit from
  `https://github.com/ICLR/Master-Template` when it ships, never hand-edit
  year strings. Kit provenance: the `iclr2026/` directory in the
  ml-paper-writing template cache, vintage "current release" per
  VENUE_STATUS.md.
- **Review style:** double-blind (ICLR standing policy) — the
  anonymization grep in the brief's "Anonymization surface" section
  applies to the ICLR build only; the arXiv build is named.
- **Archival:** ICLR proceedings are archival. Eligibility guard: every
  sibling workshop assignment in `papers/VENUE_MAP.md` is non-archival
  (live-verified where the CFP is live), so no sibling submission
  compromises this paper's ICLR eligibility.

## arXiv build

- No page limit; same source, named authors, full figures,
  reproducibility pointers into `experiment-runs/`.
- Corresponding contact: samlarson@pebbleml.com (named build only).

## Refresh obligations

- Re-run the live fetch when iclr.cc posts the 2027 CFP (dates, page
  limit, style kit, anonymization policy) — see `papers/VENUE_MAP.md`
  refresh checklist.
- Swap `iclr2026/` → official `iclr2027/` kit before the actual ICLR
  submission; the arXiv preprint may ship on the 2026 kit.

Date recorded: 2026-07-10.
