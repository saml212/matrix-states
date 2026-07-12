# Stage 05 — Format & Acceptance Audit (Round 2, DELTA)

**Paper:** *Three Bounds on a Null: Testing the Link Between Fast-Weight Write
Geometry and In-Context Composition* — MOSS @ COLM 2026 (double-blind, 4pp main).
**Audited:** 2026-07-11, fresh context, verify-not-trust. Delta focus: the
2026-07-11 Bound-1 rewrite (abstract, §1, §2, §3, §7, Appendix A, Fig 2 caption,
Appendix C compute sentence, brief rows C10/C11/C12). Whole-document mechanical
checks run across the full compiled PDF.

## Summary

**0 critical / 0 serious / 3 minor. Nothing blocks submission.** The revision is
**critical-clean**: no broken cross-reference (0 `??`), no unmatched/fabricated
number, no anonymization leak, no banned word, no placeholder. Every new number in
the changed material (§3, Appendix A, abstract, intro, §7) carries an adjacent
`% evidence:` comment; all three new rows (C10/C11/C12) exist in `brief.md`, point
at raw artifacts, and their **7 named artifacts all recompute to the recorded
md5s**. I independently recomputed the headline numbers of the instrument-fix saga
from the raws and **all match exactly**: 78/320 nonzero, the Leg-A 30/60 and Leg-B
8/20 splits, premise 0/320 both, max h=1 0.8691 (per-token wikitext s1 K32), h≥2
0.6375; the positive control (0/256 pre-fix at all four hops, cos≈1.2e-5, transposed
arm 1.0/cos≈0.99999; post-fix production 1.0 at all hops, transposed arm 0.0); the
√2 closed-form transpose signature and its pre/post verdict swap; the 0.17%
reference-recurrence error; both correspondence nulls at 0/320 clear (shuffle mean
real 0.3023/null 0.3010, derange strongest 0.8691→0.8125=94%); and the 20/21 stable
+ per-token 20/72 vs off 1/36 vs global 0/72 concentration. Round-1's sole SERIOUS
(S1: Appendix A + captions lacking `% evidence:` comments) is **fully resolved** —
those loci now carry the comments. Citations unchanged and clean (11/11, 0 orphans).
Abstract 227 words (band 200–230). Main content ends on **page 4**; References open
page 5.

---

## Critical

None.

---

## Serious

None. (Round-1 S1 resolved: `main.tex` Appendix A and all three figure captions now
carry `% evidence:` comments — C10 ×9, C11, C12, C5, C6, C7, C4, C1–C4 as
appropriate.)

---

## Minor

### M1 — Fig 1 (`fig:gates`) is present and captioned but no longer cross-referenced from the running text
The §3 rewrite deleted the wave-1 paragraph whose parenthetical
`(Figure~\ref{fig:gates}, Appendix~\ref{app:figs})` was Fig 1's only in-text
pointer. `fig:gates` is now `\label`'d (`main.tex:279`) but never `\ref`'d; the only
`Figure~\ref` targets remaining are `fig:dissoc` (§3) and `fig:transient` (§4, §5).
This is **not** a broken reference (no `??`, figure still renders in Appendix B with
its caption), but a reviewer opening the PDF meets Fig 1 with no textual referent.
**Fix:** add a `Figure~\ref{fig:gates}` in §3 where the premise-gate failure at all
cells is stated (or in the Fig-1 sentence of Appendix A). Not blocking.

### M2 — Mechanism covariates 0.9996 / 0.9648 and the 4.1% healthy-draw figure are prose-only (Appendix A) — disclosure is adequate; residual only
These three numbers have no separately archived raw artifact (the original harvest
saved no state/Z-dumps to recompute from). The draft flags this **explicitly** in a
bold "Provenance flag" sentence naming the exact numbers, states they are "a
design-registry record of an on-box diagnostic on the single strongest cell, not
backed by a separately archived raw artifact," and states the empirical verdict
(0/320 clear either null) **does not depend on them** — they color the *why*
(geometric, not compositional), not the verdict. `brief.md` C12 discloses the same
gap identically. **Assessment:** the round-1 method treats a prose-only number as
CRITICAL *unless honestly disclosed as such*; here it is disclosed, non-load-bearing,
and confined to interpretive Appendix-A text (not the abstract/body as a decision
claim). The disclosure **discharges the CRITICAL concern**; I record it as a MINOR
residual — the honest ceiling on this paper's traceability, correctly labeled.

### M3 — New "≈0.52 GPU-hours" re-verification compute figure is an author-summed approximation, not a single raw field
Appendix C's new sentence discloses ≈0.52 GPU-h for the instrument re-verification
(tagged C10/C11/C12). The re-verification raws hold only per-forward wall times
(e.g. `forward_wall_s=0.448/0.454`), not an aggregate GPU-hour field, so this is an
author-computed sum in the same class as C9's accepted "≈9.5 GPU-hours" disclosure —
approximate, plausible for 80 re-metric cells + two 320-reading null sweeps +
control/adjudication on one GPU, and non-load-bearing. Acceptable compute disclosure;
noted for completeness. Not blocking. (Also pre-existing, not a delta: the abstract's
"one of five that exclude zero across forty tests" carries no adjacent comment but is
unchanged round-1 context, traceable to C6-transient, and was verified 5/40 in
round 1.)

---

## Recomputation ledger (all MATCH the draft)

**md5 — all 7 new artifacts verify:** C10 poscontrol-prefix `71e53be…`, C10 PREFIX
adjudication log `3e23fd60…`, C10 POSTFIX adjudication `1daa4c80…`, C10 poscontrol-
postfix `e5abae88…`, C11 `AGGREGATE_SUMMARY.txt` `34186cbc…`, C12 shuffle
`ANALYSIS_SUMMARY.txt` `922bfc8b…`, C12 derange `DERANGE_SUMMARY.txt` `2edc16da…`. ✓

**C11 re-metric** (`04_remetric/results/AGGREGATE_SUMMARY.txt`): GRAND TOTAL 78/320
nonzero; Leg A 30/60 cells; Leg B 8/20 cells; premise iii 0/320, iv 0/320; h=1 max
0.8691 (`leg_a_per_token_wikitext-mix-ext_s1_k32_native`); h≥2 max 0.6375
(`leg_a_per_token_openr1…s2_k20_off` h4). ✓

**C10 positive control** (`…poscontrol_result.json` / `…_POSTFIX.json`): pre-fix
`production_recovered_frac=0.0` at all four h, `n_scored=256`, cos_mean≈1.2e-5;
`deliberately_transposed_recovered_frac=1.0`, cos≈0.99999; `key_orthonormality_max_abs_dev=3.576e-7`
(→3.6e-7); `n_beta_near_zero=0/256`; `kernel_vs_independent_reference_S_rel_fro_err=0.001710`
(→0.17%). Post-fix: production 1.0 all h (cos≈0.99999), transposed 0.0. ✓

**C10 closed-form adjudication** (PREFIX log / POSTFIX json): pre-fix
`S_squeeze_vs_closed_form_fla_rel_fro=0.0`, `…_design_rel_fro=1.4142` (√2), verdict
"returns RAW fla [K,V] (the defect): True"; post-fix numbers swap
(`…_design_rel_fro=0.0`, `…_fla_rel_fro=1.4142`), verdict "returns DESIGN [V,K]: True".
Matches the draft's "exactly √2 pure-transpose signature" and role-swap. ✓

**C12 correspondence nulls** (shuffle / derange summaries): shuffle 320 readings,
NULL-CLEARS 0, 0/38 signal-bearing cells clear, real@floor 0.3023 vs null 0.3010,
OUTCOME TRIVIAL-ARTIFACT; episode resample 21 ON → 20 STABLE/1 FLIPPED (→20/21);
per_token 20/72, off 1/36, global 0/72. Derange 320 readings, NULL-CLEARS 0,
real@floor 0.3023 vs deranged 0.2960, strongest 0.8691 real / 0.8125 deranged (94%). ✓

**Citations:** 11 in-text = 11 bib, 0 orphans either direction; same set as round 1
(no add/drop). Entry types unchanged (9 `@article`, `bertinetto2021` `@inproceedings`,
`forde2020` `@proceedings`). ✓

---

## Counts

- **Body word count:** ≈2,340 words (§§1–7 + abstract, LaTeX-stripped estimate;
  excludes Table 1 / Appendix captions). Rendered §§1–7 end on **page 4**; References
  page 5; Appendices A–C span pages 5–10 (10 pages total incl. appendix). Within the
  4pp main-content limit. Render-inspector is the authoritative page count.
- **Abstract length:** 227 words (band 200–230). ✓
- **Figures referenced vs present:** 3 present (`fig1_validity_gates`,
  `fig2_dissociation`, `fig3_transient_replication`), all `\includegraphics`'d; 2 of
  3 `\ref`'d — `fig:dissoc`, `fig:transient` cited; **`fig:gates` labeled but
  uncited** (M1). No orphaned bundle files.
- **Citations:** 11 in-text vs 11 in-bib; **0 orphans** either direction.
- **Cross-refs:** all `\ref` resolve; **0 broken / 0 `??`** in `main.pdf`.
- **Anonymization matches:** **0** across `main.tex`, `sections/*.tex`, `refs.bib`,
  compiled `pdftotext`, `strings main.pdf`, and figure-PDF `strings`/metadata (only
  "Matplotlib v3.9.4" as Creator/Producer). Full token list checked (Larson,
  samuellarson, pebble(ml), Brev, anthropic, Claude, github.com/, huggingface.co/,
  acknowledg, self-funded, /Users//Volumes//root/, youthful-indigo-turkey).
- **Banned-word hits:** **0** (17-word list, whole-word, case-insensitive, over
  rendered prose and source). DO-NOT word "audit": **0** (paper uses "independent
  review" instead).
- **Literal placeholders:** **0** (TODO/pending/forthcoming/will be added/Table
  X/Figure Y/[CITE]). The Appendix C "released with the camera-ready" line is the
  round-1-accepted standard disclosure, not a placeholder.
- **Evidence tokens:** C0×1, C1×6, C2×3, C3×4, C4×7, C5×10, C6×6, C7×11, C8×3, C9×1,
  C10×9, C11×8, C12×10 — all 13 ids resolve to real `brief.md` rows.

**Termination contribution:** format audit is **critical-clean**. No CRITICAL format
finding; no SERIOUS finding (round-1 S1 resolved). The 3 minor items are
recommended-fix, none blocking. Nothing blocks submission.
