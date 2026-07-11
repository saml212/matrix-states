# 05 — Format + Acceptance Audit (gauntlet stage 5)

Paper: `papers/rank-recruitment-ws/` — NeurReps 2026 Extended Abstract,
"When the Gradient Sees Rank". Fresh-context audit of the post-rebuttal-fix
draft. Read-only; no paper files edited.

## Summary

**0 critical / 0 serious / 4 minor. Nothing blocks submission.**

The draft is acceptance-clean. Every numerical claim in the body, abstract,
and both appendix tables carries an immediately-following
`% <!-- evidence: Rx -->` comment pointing to a real row in `brief.md`, and
every row names a raw artifact by path. I independently re-verified the md5s
of all 11 raw files I relied on (all match `brief.md`) and **recomputed the
headline numbers for rows R1, R2, R3, R4, R5, R7, and R9 directly from the
raw JSONs — every one matches the draft.** The two deliberate post-brief
fixes both check out against the raws: whole-matrix stable rank min is exactly
`14.64867` (rounds to the printed "14.6"), and the d=16 anchor is correctly
absent from the §5 frontier list. Cross-references all resolve (no `??` in the
PDF), anonymization is clean except the documented `larson2026gradient`
third-person exception, the author block is fully commented out, the abstract
is ~205 words (in the 200–230 band), and the body ends on page 4 (compliant
with the 4-page limit excluding references and appendices). No banned words,
no contractions, no placeholders.

The four minor items are hygiene/precision nits (two orphan bib entries, one
missing evidence comment on a derived rule, one boundary-case band claim, two
uncross-referenced floats). None affect a number, a reference, or
anonymization.

## Critical

None.

## Serious

None.

## Minor

### M1. Two orphan bib entries (present in `refs.bib`, never cited)
`refs.bib` contains 20 entries but only 18 are cited. `plate1995hrr` (Plate,
Holographic Reduced Representations) and `smolensky1990tensor` (Smolensky,
Tensor Product Variable Binding) are never `\cite`d anywhere in `main.tex` or
`sections/`. They do **not** render in the compiled PDF (bibtex emits only the
18 cited entries — the References list is correct), so this is invisible to a
reviewer reading the PDF, but it violates the no-orphans bib-hygiene rule and
leaves dead entries in the bundled `.bib`. These are the two classic
variable-binding references; they were almost certainly inherited from the
13pp base draft's related work and lost their `\citep` in the EA compression.
**Fix:** either delete both entries, or (if the related-work paragraph on
binding should cite them) add the `\citep` — a writer/content call, not a
format call.

### M2. Derived "2–2.5×" rule lacks a co-located evidence comment
The abstract ("no cell is called dead without re-testing at 2--2.5$\times$
budget", `main.tex`:67) and the §5 rule statement ("a cell is not dead until
re-tested at 2--2.5$\times$ budget", `sections/05_frontier.tex`:14) state the
`2–2.5×` multiplier with no `% <!-- evidence: -->` comment immediately after
it. The number is traceable — it is a derived characterization of the budget
reversals evidenced by R8 (20K→40K = 2×, 8K→20K = 2.5×) and R6 (40K→80K = 2×),
whose comments sit on the adjacent measured numbers in the same paragraph — so
this is not an untraceable number, only a gap against the paper's own DO-NOT
discipline ("every number carries a comment"). **Fix:** append
`% <!-- evidence: R8 -->` / `% <!-- evidence: R6 -->` after the rule
statements, or leave as-is (defensible, since it is a rule name not a
measurement).

### M3. K=3 band-membership claim is a boundary case on the extended grid
§3 says the d=16 effective rank leaves the pre-registered `[0.7K, 1.3K]` band
"only at K=1, 2, by overshoot" (`sections/03_recruitment.tex`:8). Recomputed
from the raw (`AGGREGATE_latest.json`, n_runs=991): the K=3 effective rank is
**3.9180**, which marginally exceeds `1.3·3 = 3.900` (0.46% over) — so on the
full **ten-point** grid the sentence describes, K=3 also (barely) leaves the
band. The claim is *exactly* correct on the pre-registered grid points
`{1,2,4,8,12,16}` (`TASK_D_PREREGISTRATION.md`:196 — K=3/6/10/14 are added
points, not pre-registered), on which only K=1,2 overshoot. **Fix:** either
scope the sentence to the pre-registered points ("of the pre-registered
K-values, only K=1,2 leave the band") or add "(and marginally at K=3)". Pure
boundary rounding; does not touch the Spearman ρ=1.0 headline.

### M4. Figure 2 and Table 2 are never cross-referenced by `\ref`
`fig:depth` (Figure 2, the depth-amplification figure) and `tab:subspace`
(Table 2, the per-seed subspace decomposition) are included and captioned but
never referenced via `\ref` in the prose — the text discusses their content
and points the reader to the appendix instead
(`Appendix~\ref{app:subspace}`). Not a broken reference (nothing renders as
`??`), just the best-practice miss of not calling out every float by number.
Cosmetic. **Fix:** add `Figure~\ref{fig:depth}` in §4's depth-amplification
paragraph and `Table~\ref{tab:subspace}` where Appendix B is pointed to.

## Recompute log (acceptance check — the one that matters)

All raw files verified by md5 against `brief.md` before use. All recomputes
via read-only reads of the raw JSONs (`analyze_zdump.py` for the R5 subspace
decomposition and the recorded `stable_rank_mean` fields).

| Row | Draft claim (sample) | Raw recompute | Verdict |
|---|---|---|---|
| R1 | d=16 grid 2.42/3.01/…/8.20/…/15.09; ρ=1.0; K>d declines 11.1/11.6/8.6 | 2.4219/3.0146/…/8.1984/…/15.0885; strictly monotone → ρ=1.0; 20→11.14, 24→11.61, 32→8.57 | **MATCH** (K=3 band nit → M3) |
| R2 | d8,K4: k≤3 ≤0.0004, k=4 = 0.97; d16 knees at k≈K | k1/2/3 = 0/0.0004/0, k4 = 0.9675; d16_K8 k7=0.9122, d16_K12 k12=0.9354 | **MATCH** |
| R3 | 4/5 seeds ≥0.9996 all hops; 3/5 = 1.000; 5th 0.93→0.16→0.0001 | s1/s2/s3 = 1.0 all hops, s4 min 0.9996; s0 h1=0.9268, h7=0.1613, h21=0.0001 | **MATCH** |
| R4 | fr7_s2 rec 0.996→0.881→0.060 at h=1/7/21; Table 1 cos 0.9303/…/0.8206 | rec 0.9960/0.8812/0.0604; cos 0.9303/0.9259/0.9212/0.9163/0.8206 | **MATCH** |
| R5 | whole-matrix effrank 16.0, stable rank 14.6–15.6; restricted A 7.999–8.000, resid 0.7–2.4%, leak <1.5%, ρ(D)≥1 | effrank 15.9958–15.9996; stable 14.64867–15.59185; A 7.9987–7.9998, resid 0.74–2.38%, leak 0.44–1.46%, ρ(D) 1.018–2.856 | **MATCH** (incl. deliberate stable-rank fix; raw min = 14.64867) |
| R7 | d32 cos 0.877/0.909/0.915, rec 0.413/0.632/0.653, rank 91–97%; d48 0.7196/0.002; d64 0.3882/0.0 | d32 0.8768/0.9093/0.9145, 0.4131/0.6324/0.6533, effrank 7.296/7.762/7.680 (91.2–97.0%); d48 0.7196/0.0020; d64 0.3882/0.0000 | **MATCH** |
| R9 | d=16 model = 170,896 params | `n_params` = 170896 | **MATCH** |

md5s verified matching `brief.md`: `AGGREGATE_latest.json`
(c0a7d27e…, n_runs=991); the five `..._K8_frN_s0..4.json`; `..._K8_fr7_s2.json`;
and the five R7 stage-0 files (d32 s0/s1/s2, d48 s0, d64 s0) — all OK.

## Counts

- **Body word count:** ~1230 (sections 1–6) + abstract ~205. Abstract in the
  200–230 band under a natural reading (each inline `$…$` = 1 word); a naive
  whitespace count that splits math tokens reads ~242.
- **Page budget:** body (through §6 Limitations) ends on **page 4**; appendices
  A–C and references occupy pages 5–7. Compliant with "4 pages excluding
  references and appendices." 7 pages total. (Body is at the limit — the
  render-inspector stage is the authoritative page count.)
- **Figures:** 2 present (`fig_forcerank.pdf`, `fig_depth.pdf`), 2 included via
  `\includegraphics`, **0 orphaned**; 1 of 2 cross-referenced by `\ref`
  (`fig_depth` is not — see M4). `figure_gen.py` present; both figures load only
  md5-asserted archives.
- **Citations:** 20 in `refs.bib`, 18 cited in text, **2 orphans**
  (`plate1995hrr`, `smolensky1990tensor` — M1), **0 undefined** (nothing cited
  is missing from the bib). Entry types sane (`@inproceedings`+`booktitle` for
  conference papers, `@article` for arXiv/journal); every entry carries an
  arXiv id / DOI-equivalent venue; bib header documents programmatic
  verification.
- **Cross-refs:** 10 `\ref` occurrences (9 unique targets), all resolve;
  **0 broken** (no `??` in the PDF).
- **Anonymization:** 3 matches in the compiled PDF, **all** the documented
  `larson2026gradient` third-person exception (2 in-text "Larson (2026)" + 1
  bib "Sam Larson"); **0 disallowed** matches (no pebble/rockie/github/hf/path
  leaks in source, PDF, or figure PDFs; PDF metadata carries no author). Author
  block fully commented out (`main.tex`:48–50).
- **Banned words / style:** "audit" 0, contractions 0, GPU-hour/dollar/funding
  language 0.
- **Placeholders:** 0 in the body. The only `PENDING-USER` markers are the two
  documented, fully-commented author-block lines in `main.tex` (do not render).
  The `https://anonymous.4open.science/` code link is the documented
  anonymization convention.

## Gate

Format audit is **CRITICAL-CLEAN**. It contributes no blocking finding to the
gauntlet's termination gate.
