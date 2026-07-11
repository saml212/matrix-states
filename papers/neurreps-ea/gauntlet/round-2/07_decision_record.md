# 07 — Companion decision record + applied de-dup edits

Source authority: `papers/rank-recruitment-ws/gauntlet/round-1/07_final_review.md`
(Fable, READY-AFTER-CHANGES), §4 THE COMPANION-PAPER ADJUDICATION
(ESCALATION-1). Coordinator ratified the review's PRIMARY recommendation
under the PI's impact-first delegation: **both EAs stay at NeurReps**;
this tree (the sibling, restating carrier) receives the de-dup edits.
Applied by a fresh change-applier agent, 2026-07-11.

## Edits applied (exactly S1-S3 from the review, plus one consistency
## extension)

- **S1** (`sections/03_binding.tex`): replaced the number-restating
  sentence (8.20/15.08 at K=8/K=16; rank-3 ≤0.0004 / rank-4 0.940) with
  a pointer to the concurrent companion submission. Numbers dropped;
  brief.md's N6/N9 evidence rows are now unreferenced by design (per
  the review's own note, "fine").
- **Extension (not in the review's S1-S3 list, applied for internal
  consistency):** `main.tex`'s abstract carried the identical restated
  binding numbers in its own sentence — the review's §4(a) "Substantive"
  bullet explicitly names "the sibling's abstract itself carrying this
  paper's binding numbers" as one of three detectability signals, but
  the enumerated S1-S3 edit list only names `03_binding.tex`. Leaving
  the abstract untouched would have (a) left exactly the flagged
  detectability signal live and (b) created a fresh abstract/body
  mismatch of the kind the review's own methodology treats as a defect.
  Replaced the abstract's binding-numbers sentence with the same
  pointer framing. Flagged here explicitly as a judgment call, not
  silently bundled into "S1."
- **S2** (`sections/07_limitations.tex`, `app:period`): cut to a
  one-two sentence pointer to the companion's Appendix A, removing the
  parallel $\pi^{21} = \pi^5$ derivation and the depth-decay table.
- **S3** (`sections/07_limitations.tex`, Limitations paragraph): added
  the missing mirror disclosure — "A concurrent companion submission
  carries the binding-task program (the recruitment grid, force-rank
  staircase, and composition mechanism) in full; this paper shares no
  figures or tables with it."

## Verification

- Rebuilt 3x tectonic passes (per Makefile's documented aux
  fixed-point-transient note); converges, 0 errors, page count
  unchanged at 8 (body pages 1-4 exactly, same as pre-edit — verified
  via `pdftotext -f/-l` section-boundary check).
- Render-inspected the four changed pages (1, 2, 5, 7) as rendered
  PNGs at 200dpi: abstract (p1) and §2 binding paragraph (p2) read
  cleanly with no restated companion numbers; Limitations (p5) carries
  the new mirror disclosure; Appendix E (p7) is now a short pointer
  immediately followed by References, no orphaned heading. One
  self-caught defect during this pass: an initial abstract edit left a
  stray mid-word space ("causal- necessity" from a source line-wrap
  landing right after a hyphen) — fixed and re-rendered clean.
- Anonymization: grepped source and rendered pages 1-4 text for the
  full token list (`larson`, `pebble`, `rockie`, etc.) — zero matches.
- Residual-numbers check: grepped the full tree for the four dropped
  companion numbers (8.20, 15.08, 0.940, 0.0004) — zero matches
  outside this decision record.
- `make bundle` rebuilt; bundle PDF matches main.pdf (8pp).
- arXiv named-build kit rebuilt end to end: `arxiv/neurreps-ea-arxiv.tex`
  regenerated from the new bundle with the same two-change
  de-anonymization pattern (watermark removed, real author block
  added); 3x tectonic passes, 8pp, render-inspected page 1 clean;
  `neurreps-ea-arxiv-v1.zip` and `~/Desktop/arxiv-neurreps/` both
  updated (metadata.md's abstract field refreshed to match, rebuild
  noted inline, and a pre-existing stale companion-title in the
  Comments field corrected to the companion's actual title, "When the
  Gradient Sees Rank..."). PI has not uploaded to arXiv yet, so
  overwriting the v1 kit in place is safe.

## Outcome

Tree remains ACCEPT-READY for NeurReps 2026 EA under the joint-venue
path. No new gauntlet round required — this was a scoped edit against
an already-adjudicated escalation, not a new prose defect. Companion
resolution recorded in both trees per the review's §4 closing
instruction ("record the decision in both trees' gauntlet logs before
either paper is submitted"); see the mirror entry in
`papers/rank-recruitment-ws/gauntlet/round-1/08_decision_record.md`.
