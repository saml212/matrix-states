# Paper outline -- kwall

Stage 2 of the `paper` skill (repo mode). Written after the draft
(unusual order for this house convention -- the write-up task specified
exact section content directly; this file records the resulting structure
for the next stage's page-fit pass, rather than having pre-planned it).

## Section plan

| # | Section (file) | Content | Source |
|---|---|---|---|
| -- | Abstract (`00_abstract.tex`) | toy-scale statement up front; the win; the wall; the diagnosis; the counterfactual numbers; pending-fix disclosure | S7e, S11.5/S11.6, NCR_ORTHO_WRITE S1/S5 |
| 1 | Introduction (`01_introduction.tex`) | corrected novelty framing (O(log h) vs O(h), not "state can state-track"); the sweep's closest-neighbor finding; contribution list | research/ncr_separation_grounding.md Part 2/3 |
| 2 | Setup (`02_background.tex`) | NCR mechanism; exact-continuous-recovery vs argmax (Nichani, flagged); mod-K collapse trap + single-cycle construction; naming-collision disclaimer vs Task E's older K-wall; gates | S2 F2, S3.2/S3.2a, S11 |
| 3 | The K=8 result (`03_win.tex`) | Axis A/B/C verdicts, P1/P2, K=12 replication -> WIN-PARTIAL | S7e, S7g |
| 4 | The K=32 wall (`04_wall.tex`) | origin cliff; earlyln K-ladder; d=K+1 convention + leakage mechanism (scope-corrected to K<=24); K=32 full grid CLOSED; budget-rescue ANOMALY, still CLOSED | S9.10, S11.2, S11.4, S11.4a, S11.5, S11.6 |
| 5 | Diagnosis (`05_diagnosis.tex`) | spectrum-blind loss hypothesis; measured non-normality/cond#; diagnosis table (5 groups); the fix (NS-polar); MuonSSM differentiation | NCR_ORTHO_WRITE S1/S2/S5, research/ortho_write_grounding.md S1-S4 |
| 6 | Pending (`06_pending.tex`) | live-run disclosure; WIN/PARTIAL/NULL/FAIL bands Part A + B; pre-registered odds | NCR_ORTHO_WRITE S3/S4/S4b/S7 |
| 7 | Related work (`07_related.tex`) | composition mechanism neighbors; state-tracking-expressivity disclaimer; rank/readout; orthogonal parametrizations incl. MuonSSM | both grounding memos |
| 8 | Discussion (`08_discussion.tex`) | scale/scope; five explicit non-claims; falsification recap | throughout |
| A | Appendix: gates (main.tex `app:gates`) | Axis-A bands; S11 Gate 1/2 + K=32 stopping rule; ortho-write WIN/PARTIAL/NULL/FAIL | S3.2a, S11, NCR_ORTHO_WRITE S4 |
| A | Appendix: reproducibility (main.tex `app:repro`) | evidence-comment discipline; diagnosis script paths; pending-run disclosure | -- |

## Outline sanity checks

- [x] Every one of the write-up task's five content items lands in
      exactly one section (1->S3, 2->S4, 3->S5, 4->S6, 5->S7).
- [x] Every numeric claim carries a `% evidence: <doc> S<n>` LaTeX
      comment in the source immediately after the sentence/table that
      states it.
- [x] The pending section is clearly marked `[PENDING --- ortho-write
      verdict, lands ~2026-07-17]` and states both a WIN and a NULL
      reading path (no result is pre-spun).
- [x] Related work uses only entries the two grounding memos mark
      VERIFIED, cross-checked by this write-up reading the memos
      directly; one flagged entry carries its caveat into the draft
      text (`[NEEDS-FINAL-SPOT-CHECK]`) rather than being silently
      upgraded to a clean citation.
- [x] No invented given/first names in `refs.bib` -- verified against
      each grounding memo's own author-name granularity; two overlapping
      entries reused verbatim from `papers/reasoning-null-moss/refs.bib`
      (arXiv-API-verified 2026-07-10) instead of re-deriving.
- [x] `results_ortho_write*` paths under
      `experiment-runs/2026-07-16_ncr_ortho_write/` were never read by
      this write-up (grep-checked: zero references in any section file
      or in the tool-call history that produced them).
- [ ] NOT done: page-fit to MOSS's 4pp main-content cap. Current build
      is 10pp total (main text + 2 appendices + references); no
      section-by-section budget was set before drafting, unlike the
      sibling papers' `outline.md` convention -- a follow-on structural
      pass (fold tables to prose, move detail to appendix, compress
      contribution lists) is needed before this is submission-shaped.
- [ ] NOT done: anonymization grep against the program's standing token
      list (`Sam Larson`, `samlarson16`, `learned-representations`,
      `youthful-indigo-turkey`, etc.) -- spot-checked by eye during
      drafting (design-doc filenames are cited only in comments/appendix
      prose, never in body prose) but not run as an automated pass.
- [ ] NOT done: figures. This draft uses two LaTeX tables
      (`sections/05_diagnosis.tex`) instead of the house's usual
      versioned `figure-gen.py` + md5-manifest pipeline, given the
      write-up task's scope was drafting, not a full figure build. A
      figure pass (mirroring `papers/reasoning-null-moss/figures/
      figure-gen.py`'s pattern) is a reasonable next step once the
      pending verdict lands and the final table set is stable.
