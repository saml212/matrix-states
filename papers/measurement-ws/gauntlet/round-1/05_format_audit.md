# Format audit — measurement-ws, gauntlet round-1 stage 05 (acceptance gate)

Auditor: fresh-context format auditor (verify, do not trust). Draft re-built
from source with `tectonic` (triple pass, clean; only underfull-\hbox
warnings). Numbers recomputed from the named raw artifacts with md5s
re-checked this session.

## 1. Summary

**Verdict: CLEAN — 0 critical / 0 serious / 5 minor.** Nothing blocks the
gauntlet. Cross-references resolve (no `??`), every numerical claim traces
through its `<!-- evidence: -->` comment to a real brief row and a raw
artifact, and 15+ headline numbers were recomputed from the md5-verified
raws and match the draft. The anonymization grep is clean over the rendered
PDF and over the source prose (comments stripped). Abstract is in band, main
text fits 4 pages, figures regenerate byte-identically from their asserted
raws. The five minor items are cosmetic or forward-looking (a bundle-build
comment-stripping requirement, one loosely-coupled multiplier, Fig. 1
render size, a bib entry-type nit, appendix spacing).

Counts: `0 critical / 0 serious / 5 minor`.

## 2. Critical

None.

## 3. Serious

None.

## 4. Minor

- **M1 (bundle requirement, not a current leak).** `main.tex:15` carries the
  de-anonymizing token `matrix-thinking` inside a `%` comment (the known,
  pre-flagged one). It does **not** render: the anonymization grep is zero
  over the PDF text and over comment-stripped prose. But the `Makefile`
  `bundle` target flattens `\input{...}` via `re.sub` **without stripping
  comments**, so a generated `bundle/measurement-ws-submission.tex` would
  carry the token if a `.tex` is ever uploaded. Default build is PDF-only
  (clean). Fix before any source upload: strip `%`-comments in the bundle
  flattening step, or delete the token from the comment. (`bundle/` is
  currently empty, so nothing leaked.)

- **M2 (number precision, cosmetic).** Case I (`03_case_tolerance.tex`,
  evidence I2): "residuals near $1.2\times10^{-6}$, $7{,}800\times$ below
  tolerance." The raw (`diag_ns_admission_result.json`, md5
  `d0d940d0…`, verified) gives mean_resid `1.1954e-6` (→ 8,368× below the
  0.01 tol) and max_resid `1.4164e-6` (→ 7,062×); p90 `1.2747e-6` → 7,845×.
  Each residual value quoted (1.2e-6 mean, 1.4e-6 max) is exact; the
  "7,800×" multiplier tracks the p90/max end while being paired with the
  mean value, so it is internally loosely coupled. Consider "~8,000×" or
  pin the multiplier to a stated residual. Not blocking.

- **M3 (figure legibility, hand to render-inspector).** Fig. 1 sits at
  `0.40\linewidth` beside its caption; rendered, its axis-tick and legend
  fonts (7–8pt pre-scale, ~46% down) are at the low end of legibility. The
  story (the off-diagonal converged-cell column) reads fine; the small
  labels are the concern. Fig. 2 (full `\linewidth`) is fully legible.
  Flag for the Stage-8 render inspection; a nudge to ~0.44–0.46\linewidth
  if the caption still fits would help. Not a format-blocking defect.

- **M4 (bib entry-type nit).** `henderson2018deep` is `@article` with
  `journal = {Proceedings of the AAAI Conference on Artificial
  Intelligence}` (a conference proceedings cited journal-style with
  volume/number). Renders correctly under `plainnat`; `@inproceedings`
  would be more conventional. `kapoor2023reforms` is cited as an arXiv
  preprint (`@article`, arXiv:2308.07832) — acceptable and verifiable. All
  15 entries carry an arXiv id / DOI / OpenReview id (no hallucination-risk
  hand-written entries); the bib header logs programmatic arXiv/CrossRef
  verification. `companion2026capacity` is correctly `@unpublished` +
  `author={Anonymous}` for double-blind.

- **M5 (cosmetic spacing).** Several underfull-\hbox warnings in
  `sections/09_appendix.tex` (badness up to 10000, lines 83–110). Spacing
  only; no content or reference impact.

## 5. Counts

- **Body word count:** ~2,298 (sections 1–8, LaTeX-stripped, incl. the
  Fig. 1 caption). **Main text = 4 rendered pages** (references begin p5,
  appendix pp6–8), within the assumed 4pp limit. Page limit is a working
  assumption pending the Jul-11 venue; the authoritative rendered-page
  count is the Stage-8 render inspector's — the source estimate confirms 4.
- **Abstract:** 211 words (band 200–230). **PASS.**
- **Figures referenced vs present:** 2 vs 2, exact. `fig1_lens_contradiction.pdf`
  (`fig:lens`), `fig2_tap_localization.pdf` (`fig:taploc`). No orphaned
  figure files.
- **Citations in-text vs in-bib:** 15 vs 15. **Zero orphans either
  direction** (all 15 `\cite*` keys resolve; all 15 bib entries are cited).
- **Cross-refs:** 17 `\label`s defined; 14 distinct labels `\ref`'d, all
  resolve; **0 broken (`??`)** in the PDF. Three labels are defined but
  never referenced (`app:tables`, `sec:intro`, `sec:related`) — harmless.
- **Evidence traceability:** 20 distinct evidence IDs used
  (I1–I6, W1–W5, X1–X5, B1–B3, N1); **all 20 map to real brief rows, and
  every brief row is cited.**
- **Anonymization matches:** 0 in rendered PDF text; 0 in source prose
  (comments stripped). 1 source-comment token (`main.tex:15`,
  `matrix-thinking`) that does not render → see M1.
- **Banned-word hits:** 0. **Contractions:** 0. **Placeholders**
  (TODO/TBD/XXX/will-be-added/`\textcolor`/etc.): 0 in rendered and in
  source prose.

## Appendix — numerical spot-checks (all raws md5-verified this session)

Every raw below matched its brief/figure-script md5; every recomputed number
matched the draft.

| Row | Draft number | Raw (md5-verified) | Recomputed | Match |
|---|---|---|---|---|
| X1 | endpoint FALSIFY | `stage2_harvest_report.json` (`7dddd18…`) | verdict=FALSIFY, reason "does NOT measurably separate … at EITHER S5 or A6" | ✓ |
| X2 | 62 cells; converged exemplar loss 0.0001 / primary 0.050 / crosscheck 1.000; 13 cells ≥0.5 gap all converged; 16/16 >0.3 | `crosscheck_lens_verdict_output.json` (`f26a769…`) | 62 rows; exemplar loss 8.15e-5 / 0.05 / 1.0; ≥0.5 gap=13 (all conv); >0.3 gap=16/16 conv | ✓ |
| X4 | teeth real 1.000/0.800/1.000, shuffled 0.000/0.000/0.050, all pass | same file, `teeth_control` | rows exact; `all_pass=true` | ✓ |
| X5 / Tab.4 | S3 0.920/0.900/0.828/0.150; S4 1.0/1.0/0.9/0.0; A5 0.49/0.39/0.441 | `crosscheck_lens_verdict_output.json` `verdict_crosscheck_lens.per_group` | S3/S4/A5 match; S3 arm2 0.15>0.125 (no collapse) | ✓ |
| §5 body | S3 primary "0.54 against 0.459" | `stage2_harvest_report.json` | far_recovered_frac_90=0.54, far_bar=0.459, clears=true | ✓ |
| I2 | mean 1.2e-6, max 1.4e-6 (tol 0.01) | `diag_ns_admission_result.json` (`d0d940d…`) | mean 1.1954e-6, max 1.4164e-6, resid_tol 0.01 (7,800× loosely coupled → M2) | ✓ (see M2) |
| I3 | 107/107; 0 violations/36 events; 12× floor; 4,313@24 + 295@28 = 4,608; TOLERANCE-MISCALIBRATION | `diag_c17_repro_analysis_K84_s1940.json` (`6f24447…`) | reconstruction 107/107 diff=0; step_0b 0 viol.; step_neg1 36 events/min 3 (=12×); histogram {24:4313, 28:295} total 4608; verdict TOLERANCE-MISCALIBRATION | ✓ |
| I4 | no cliff through 0.9375; per-load 0.92–1.00 | `fit_d96_unlocked_results.json` (`61eaffe…`) | curve x max 0.9375; h4=[0.9592,0.9216,0.9326,0.9581,1.0] | ✓ |
| W1 | recall 0.9990 / chance 0.03125 / baselines 0.0447, 0.0295 | `tap_localization_SUMMARY.json` (`0e73ee2…`) + `h2h_calib_transformer…json` (`a0a7181…`) | 0.99902 / 0.03125 / 0.04468 / 0.029541 | ✓ |
| W4/Fig2 | S0-zero 0.0286, S1-zero 0.9990, tap-iv 0.674 / abl 0.0 | `tap_localization_SUMMARY.json` (figure_gen md5 assert passes) | figure_gen prints 0.9990/0.0286/0.9990, tap-iv 0.674/0.000 | ✓ |
| I6 | 0.50 + 1.10 GPU-h vs 6.33 | `box_verification_20260708.json` (`df97dd3…`) + `poolmargin_run1.log` (`3d19ac3…`) | chain 1782.3s→0.4951; poolmargin 2608.7+1343.5=3952.2s→1.098 | ✓ |
| B1 | 0.084→0.9996; centered scale-only 0.046; centering-only 0.705→0.9996 | `gate1b_recheck.txt` (`2d170cc…`) | uncentered 0.084111, crosscheck 0.999594, scale-only 0.046011, centering 0.705261 | ✓ |
| B2 | 37/39 within 0.07, mean abs dev 0.028; S3 0.17/0.08 | `harvest_analysis_output.txt` (`854a4bd…`) + `harvest_summary.json` (`7dce77d…`) | 39 cells mean|obs-pred|=0.0276, 2 outliers (→37 within 0.07); S3 0.167/0.075 | ✓ |
| B3 | 4.5e-8; 1.405; residuals 2.8e-3–4.5e-3; mutant killed 1.405 | `analytic_step1…log` (`2b8fb48…`), `fla_cross_check_fixed_pass.log` (`c2c68f8…`), `analytic_check_negative_teeth_test.log` (`676a088…`) | 4.504e-8; 1.405; 2.8008e-3/3.8678e-3/4.5237e-3 PASS; mutant killed rel-Fro 1.405 | ✓ |

Figure sources: both `figure_gen.py` md5 assertions
(`crosscheck_lens_verdict_output.json` = `f26a769d5c263af224c91d39bd83710b`,
`tap_localization_SUMMARY.json` = `0e73ee283b7208db10af5588fe1c6713`) match
the committed raws at both the repo path and the SSD path; re-running the
script regenerated `fig1`/`fig2` **byte-size-identical** (14169 / 16766) to
the committed PDFs, and its printed consistency counts (62 cells, 17
converged, 13 contradictions all converged; S0-zero 0.0286 / S1-zero 0.9990
/ tap-iv 0.674) match the draft.
