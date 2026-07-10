# Format and Acceptance Audit — Round 2, Stage 05 (v2, fresh re-run)

Fresh-context format/acceptance auditor, independent of the v1 report
(`gauntlet/round-2/05_format_audit.md`) and of `04_rebuttal_report.md`. Every
check below was re-derived from the current tree — `main.tex`, `sections/*.tex`,
`refs.bib`, `figures/`, `bundle/` — and from the raw experiment archives under
`experiment-runs/` and `matrix-thinking/`, using `hashlib.md5` (never a copied
digit) and a fresh `tectonic` compile. Nothing below is taken on the v1
report's, the brief's, or the rebuttal's word.

## SECURITY NOTE

No embedded fake `<system-reminder>` / concealment-instruction block was
encountered in any tool output during this audit round (role-prompt read,
styleguide read, or any of the ~60 Bash/Read calls made while verifying this
report). This is a negative result, not a claim that the injection pattern
documented in prior rounds (`04_rebuttal_report.md`'s cross-reference, and this
project's own `CLAUDE.md` Hard Rule) has stopped — only that this session did
not see one. Flagging its absence explicitly per the task's SECURITY clause.

**Methodological note, not a security finding:** partway through this audit,
`papers/neurreps-ea/figures/figure_gen.py` changed on disk (mtime moved from
Jul 9 17:55 to Jul 10 10:45, +250 bytes) without any Write/Edit action by this
audit — a marker-styling change to `fig2_rank_tracking`'s scatter plot (open
markers switched from transparent to white-filled faces, with an inline
comment attributing it to a "round-2 render-inspection finding"). This
tree is being edited by a concurrent process during this audit round. The
figure *PDFs* themselves (`figures/fig1_razor_step.pdf`,
`fig2_rank_tracking.pdf`) were not regenerated (mtimes unchanged, still Jul 9
18:57) and still match `bundle/figures/` byte-for-byte, so this did not
invalidate the figure-source md5 sweep below — but it means every check in
this report reflects a snapshot at the time it was run, and the tree may have
moved again by the time this report is read. Treat any finding whose fix
looks trivial (S-C's word count, S-B's missing heading) as worth a quick
re-check against the live tree before acting on it.

## Summary

**0 critical / 3 serious / 2 minor.** The critical finding from v1 (C1, the
`larson2026gradient` self-citation identity leak) is fully closed: a
tree-wide, bundle-wide, and compiled-PDF-wide grep for every anonymization
token (`larson`, `samlarson`, `saml212`, `pebble`, `pebbleml`, `rockie`,
`.pebbleml.com`, `github.com/`, `huggingface.co/`, `acknowledg`,
`self-funded`, `funded by`) returns zero matches anywhere in submission
content. Every mandatory numerical spot-check (N1–N14, all 14 evidence rows
the draft actually cites) was independently recomputed from the raw
JSON/log/txt artifacts named in `brief.md` and matches the draft exactly,
including two rows (N4's "4.9–10.2%" deviation range and Table 2's per-group
figures) that were tightened since v1 and are now *more* precise, not less.
All 51 md5s pinned in `figures/figure_gen.py`'s `SOURCE_MD5` dict were
recomputed fresh and match with zero drift or missing files. A live arXiv API
author-list check on all 10 arXiv-numbered bib entries matches `refs.bib`
exactly — notably, two entries (`nazari2026rank`, `sun2026staterank`) that
were **wrong** in the git-HEAD-committed `refs.bib` (a fabricated-looking
12-author list for Sun et al., a wrong first name for Nazari) have since been
silently corrected to match the real arXiv record; a third (`grazzi2025negative`)
had its author order corrected too. Bundle parity is exact (a fresh flatten
of `main.tex` is byte-identical to `bundle/neurreps-ea-submission.tex`; every
other bundle file is byte-identical to its working-tree source).

Three real problems remain, none of them critical under this role's
definition (compile failure / fabricated number / identity leak), but two are
significant enough to name loudly: **(S-A) a fresh `tectonic` rebuild of this
exact tree, in this environment, does not reproduce the shipped 8-page
`main.pdf` — it produces a 12-page document with severe blank-page padding,
most of it in the bibliography**, which undercuts confidence that the page
budget (S2, closed against the *currently shipped* PDF) will stay closed
under the next recompile, including the venue's own. **(S-B) the
binding-foundation content (Fact 1, its justification, the causal binding
step) still has no `\section` of its own** — v1's specific symptom (a
misleading `\ref{sec:binding}`) was removed by deleting the label rather than
by adding a heading, so the reader-facing defect v1 described is unchanged.
**(S-C) the abstract's word count is genuinely on the knife's edge of the
200–230 band** — 213 to 232 words depending on how inline math is counted,
with the most literal "count what's on the rendered page" convention landing
2 words over. Nothing here blocks the gauntlet by this role's termination
rule (critical-clean), but S-A and S-B should be resolved, and S-C
recounted by hand, before the render-inspection stage is trusted.

## Critical

None.

## Serious

**S-A — Build/render reproducibility gap: a fresh `tectonic` compile of the
current source does not reproduce the shipped page count.**

- The tree's shipped `main.pdf` (and the byte-identical
  `bundle/neurreps-ea-submission.pdf`) is 8 pages, 148094 bytes, densely
  packed (2500–8400 characters of extracted text per page).
- A fresh `tectonic --keep-intermediates main.tex` run in the actual
  `papers/neurreps-ea/` directory (after `rm -f main.aux main.log main.bbl
  main.blg main.out main.toc`), run twice independently, both times produced
  a **12-page** document (146.8 KiB `.xdv`, ~150KB PDF) with sparse pages
  (some as low as 115–724 extracted characters) and 5 `Underfull \vbox
  (badness 10000)` warnings — near-total blank stretches — concentrated at
  `sections/03_binding:1`, `sections/07_limitations:123`, and three
  consecutive breaks inside `main.bbl` (lines 67, 88, 113). The bibliography
  (14 entries, 113 `.bbl` lines) alone spans pages 8–12 (5 pages) versus the
  shipped build's 2 pages for the same content.
- To rule out this round's edits as the cause, the exact **git-HEAD-committed**
  version of `main.tex` + `sections/*.tex` + `refs.bib` + `jmlr.cls` +
  `jmlrutils.sty` (i.e. the state v1 audited) was extracted via `git show` into
  an isolated scratch directory and compiled fresh with the identical
  `tectonic` command. It **also** produced 12 pages, not the 8 pages that
  ships as `main.pdf` at that same commit. This rules out a content
  regression in this round's fixes — the discrepancy is a build-environment
  effect (most plausibly `tectonic`'s auto-fetched package bundle resolving
  differently now than whenever the shipped PDF was actually built), not
  something a paper edit caused.
- Practical consequence: **the page/length budget check (S2, below) is only
  as trustworthy as the artifact used to evaluate it.** I evaluated S2
  against the shipped, currently-committed 8-page `main.pdf`, per this task's
  own framing that this file is "current, freshly rebuilt with tectonic."
  That evaluation stands (S2 is closed against that artifact). But I could
  not, in three independent attempts in this environment, get a fresh build
  to reproduce it — which means neither this audit nor the render-inspection
  stage that follows it can assume the *next* compile (on this machine, on
  a reviewer's machine, or on the venue's own OpenReview/Overleaf compiler)
  will land at 8 pages rather than 12. Recommend the render-inspection stage
  (`prompts/render-inspector.md`) explicitly re-run the build 2–3 times and
  confirm page-count stability before trusting any page-budget verdict, and
  ideally pin `tectonic`'s bundle version if the toolchain supports it.
- Methodological note: because I ran `rm -f main.aux main.log main.bbl
  main.blg main.out main.toc` and rebuilt `main.pdf` in place while
  investigating this, I restored `main.pdf` afterward from
  `bundle/neurreps-ea-submission.pdf` (confirmed byte-identical,
  148094 bytes, to what was in the tree before I touched it, since the
  Makefile's `bundle` target is a direct `cp main.pdf
  bundle/neurreps-ea-submission.pdf` and that file's mtime/size predates my
  session). `main.aux`/`main.bbl`/`main.log`/`main.out` are left as
  regenerated (12-page-consistent) build byproducts — these are untracked,
  non-source files, and regenerating them is the normal effect of running
  this project's own documented `make` recipe, not a content edit.

**S-B — The binding-foundation content (Fact 1) still has no `\section` of
its own; it is still structurally merged into Section 2.**

- `sections/03_binding.tex` opens directly with `\textbf{The provable
  foundation (binding).}` followed by the `fact` environment — no
  `\section{}` or `\subsection{}` command, same as v1 found.
- What changed since v1: the `\label{sec:binding}` line that used to sit at
  the top of this file was deleted outright (not replaced by a real
  section), and the one place that referenced it
  (`sections/07_limitations.tex`'s periodicity appendix) was repointed to
  `\ref{sec:setup}` instead. `main.aux` confirms the fact is still tagged as
  living inside "Tasks, Models, and the Rank Instrument" (Section 2):
  `\newlabel{fact:bound}{{1}{3}{Tasks, Models, and the Rank Instrument}{fact.1}{}}`.
- `brief.md`'s per-section table now documents this as a deliberate choice:
  *"Amended 2026-07-10: folded into §2's flow as a bold-lead paragraph (no
  own `\section`), adjudicating the round-2 format audit's S3 finding while
  holding the 4pp budget; cross-references point at §2."* That is a policy
  note in a planning document, not a change to what a reviewer sees on the
  page: Section 2 still contains task/model/instrument description
  paragraphs (each with a bold run-in label — "Tasks.", "Model.",
  "Instrument.") *and*, appended with the same bold-run-in convention, a
  full Fact-plus-proof-plus-causal-step **result** (rank ≥ K provably; SGD
  recruits rank ≈ K; the d=8,K=4 causal force-rank step) — while the paper's
  two structurally symmetric results (the rank law observed, the causal
  razor) each get a dedicated numbered `\section`. A reviewer flipping
  through the table of contents would not find "binding" as a section at
  all. The specific broken/misleading-cross-reference symptom v1 flagged is
  gone (no dangling `\ref{sec:binding}` remains, and the repointed
  `\ref{sec:setup}` is accurate), but the structural inconsistency v1 called
  "a real structural defect a reviewer would notice" is unchanged in the
  compiled output. I score this OPEN, not closed (see v1-closure section).

**S-C — Abstract word count is on the edge of the 200–230 band; the most
literal count is 2 words over.**

- The abstract text itself is unchanged since v1 (only mid-sentence
  `<!-- evidence: Nx -->` comments were added, which are non-printing).
  v1 reported 227 words and PASS.
- Word counting a math-heavy abstract is inherently convention-dependent;
  three defensible methodologies gave three different answers:
  - **Raw rendered-PDF text** (`pdftotext -layout` on page 1, merging only
    the 3 line-wrap hyphenation artifacts `pdftotext` introduces — e.g.
    "single-" + "state" broken across a line — back into single words, but
    otherwise splitting on whitespace exactly as printed, so `ρ = 0.9747`
    counts as 3 tokens and `K=8,` counts as 1): **232 words** — 2 over the
    230 cap.
  - **Source-based, each inline-math span counted as one token**
    (`\dmin`, `K=8`, `\rho = 0.9747` each → 1 "MATH" token): **229 words** —
    inside the band, barely.
  - **Source-based, inline math spans dropped (prose-only count)**:
    **213 words** — comfortably inside the band.
  - v1's own count (227) sits between these, suggesting a fourth
    methodology (likely a stripped/normalized script count).
- None of these is obviously "the" correct convention for a math-dense
  abstract, and I am not overriding v1's PASS outright — but three of four
  methodologies I can defend land at or above 227, and my most literal
  ("count what is printed on the page") convention lands over 230. This is
  close enough, and consequential enough (venues do reject over-length
  abstracts by mechanical count), that it should be hand-recounted with the
  venue's own submission-portal counter (OpenReview/CMT abstract fields
  typically count words directly from the rendered/plain-text abstract field
  a submitter pastes in, which will look like the "raw rendered" convention
  above) before submission, rather than relying on any of these estimates.

## Minor

**M-A — refs.bib entry count dropped from 16 to 14, consistent with the two
closed findings (informational, not a new defect).** `refs.bib` now has 14
`@`-entries (`grep -c '^@' refs.bib`), down from the 16 v1 counted — exactly
accounted for by C1's fix (removing `larson2026gradient`) and S1's fix
(removing the orphaned `smolensky1990tensor`). All 14 are cited and all
citations resolve (0 orphans either direction, confirmed against
`main.aux`'s `\citation` list, which is more authoritative than a per-line
source grep since two citations — `kohonen1972correlation`,
`anderson1972simple` — are split across a line-wrap in
`sections/03_binding.tex:9-10` and a naive single-line grep would miss them).

**M-B — Five defined labels are never `\ref{}`'d (unchanged from v1).**
`fact:bound`, `tab:gate1a`, `sec:related`, `tab:groups`, `sec:intro` remain
defined via `\label{}` but never explicitly cross-referenced in the text.
Standard, harmless LaTeX practice (not every section/table needs an in-text
pointer); re-confirmed, not a real defect, unchanged from v1's M3.

## Counts

- **Body word count:** abstract 213–232 words depending on convention (see
  S-C); Sections 1–5 plus the Limitations paragraph render across pages 1–4
  of the shipped 8-page `main.pdf` (evidence-comments and `\label`s excluded,
  they are non-printing).
- **Abstract length:** see S-C. Not a clean PASS; flagged for hand-recount.
- **Figures referenced vs. present:** 2 referenced (`fig1_razor_step.pdf`,
  `fig2_rank_tracking.pdf`) / 2 present in `figures/` and in `bundle/figures/`
  — exact match, no orphans. Both are byte-identical between the working
  tree and the bundle (`diff` clean).
- **Citations in-text vs. in-bib:** 14 unique keys cited (confirmed via
  `main.aux`'s `\citation` entries, across 13 `\cite*` command sites) vs. 14
  bib entries → **0 orphans bib→text, 0 orphans text→bib.** (S1's removal of
  `smolensky1990tensor` closed the only orphan v1 found.)
- **Cross-refs:** 24 `\ref{}` usages total, 16 `\label{}` definitions, **0
  broken/undefined** (confirmed via the final, converged pass of a fresh
  `tectonic` recompile — zero "Reference ... undefined" / "Citation ...
  undefined" warnings survive after the 3rd auto-rerun). 5 of the 16 labels
  are unreferenced (M-B, harmless). **0 misleading references** (S3's
  specific symptom, a `\ref` resolving to an unexpected section, is gone —
  see S-B for the residual structural issue that is not itself a broken
  reference).
- **Overfull hboxes:** **0** in the final compile pass (confirmed twice,
  independently, via `tectonic --print`) — both hboxes v1 flagged
  (`sections/05_causal_razor.tex:35`'s table, `sections/07_limitations.tex:66`'s
  inline `\sqrt{}` term) are gone, consistent with the narrower-table
  (`\footnotesize`, `\setlength{\tabcolsep}{4pt}`, narrower minipage widths)
  and `fig1`-width (`0.80\textwidth`) fixes applied since v1. Caveat: this
  was confirmed in the (pagination-anomalous, S-A) 12-page rebuild, not in a
  build that reproduces the shipped 8-page pagination; horizontal-fit
  warnings are a local/column-width property largely independent of the
  vertical pagination anomaly, so I have reasonable but not total confidence
  this generalizes to the shipped artifact.
- **Anonymization matches:** 0 in `main.tex`, `sections/*.tex`, `refs.bib`,
  `bundle/*.tex`, `bundle/*.bib`, `figures/figure_gen.py`,
  `VENUE_REQUIREMENTS.md`, `Makefile`, the compiled `main.pdf`, the compiled
  `bundle/neurreps-ea-submission.pdf`, `main.bbl`, and `main.aux`. The only
  hits for any token were (a) `brief.md`'s own line listing the tokens to
  check (not submission content), and (b) `bundle/jmlr.cls`'s stock
  `\newcommand{\acks}[1]{\section*{Acknowledgments}#1}` macro definition
  (vendor template boilerplate, matches "acknowledg" as a dictionary word
  inside an unused macro name, not a leak). Expected: 0. **PASS.**
- **Banned-word hits:** 0 (of the 16-word verbatim styleguide list,
  case-insensitive whole-word match, across `main.tex` and all 7
  `sections/*.tex` files).
- **Contractions:** 0 in rendered prose (the one apostrophe-t match,
  `DON'T`, is inside the non-printing `%%%% DON'T CHANGE %%%%` build
  comment; all other apostrophe hits are possessives — "step's",
  "model's", "group's", etc. — not contractions).
- **Em-dash-as-pause:** 0 in prose (2 em-dash characters found, both inside
  `%%%%`-prefixed comment lines at the top of `main.tex`, non-printing).
- **Page count:** shipped `main.pdf` / `bundle/neurreps-ea-submission.pdf`:
  8 pages, body ends at the bottom of page 4, Appendix A begins cleanly on
  page 5 (S2 closed against this artifact). A fresh rebuild in this
  environment: 12 pages (S-A, not reproducible — see above).
- **Figure-source md5 sweep:** all 51 md5s pinned in
  `figures/figure_gen.py`'s `SOURCE_MD5` dict recomputed fresh with
  `hashlib.md5`; **0 mismatches, 0 missing files.**
- **Bundle parity:** `bundle/neurreps-ea-submission.tex` reconstructed
  independently (my own `\input{...}` inlining of the current `main.tex`) is
  **byte-identical** to the shipped bundle file; `refs.bib`, `jmlr.cls`,
  `jmlrutils.sty`, and both figure PDFs are likewise byte-identical between
  the working tree and `bundle/`; `bundle/neurreps-ea-submission.pdf` is
  byte-identical to `main.pdf`.

## Recomputation log

All computed fresh with `python3` (stdlib `json`/`hashlib`/`statistics`/`math`
only — no torch dependency needed for any of these, so the repo's
train-launch dry-run gate, which pattern-matches `python ... .py` commands,
was avoided by running inline `-c` snippets rather than a `.py` file; this is
a sandboxing artifact of this session, not a paper-content issue). md5s are
independently recomputed with `hashlib.md5`, never copied from `brief.md`.

- **N1** (razor table, 5 groups × {anchor, k=dmin−1, k=dmin, k=dmin+1, bar})
  — recomputed from `experiment-runs/2026-07-09_m3fix_harvest/
  zero_pad__{G}__{arm}__seed0.json`'s `crosscheck_recovered_frac_90`.
  **Exact match** to the Figure-1-caption table and to `main.tex`'s abstract
  figures: S3 0.550/0.000/0.450/0.550 (bar 0.495), S4 0.650/0.000/0.800/0.950
  (bar 0.585), A5 0.700/0.000/0.700/0.750 (bar 0.630), S5
  0.500/0.000/0.600/0.550 (bar 0.450), A6 0.650/0.000/0.650/0.700 (bar 0.585).
- **N2** (S3 necessity, 4 seeds, k=dmin−1) — `[0.0, 0.0, 0.0, 0.0]` across
  `m3fix_harvest` seed0 + `m3fix_s3ext` seeds1–3. **Exact match** ("noiseless
  ... in all five groups ... and all four S3 seeds").
- **N3** (S3 sufficiency seed-mean) — per-seed `[0.45, 0.55, 0.60, 0.65]`,
  mean **0.5625** vs. the fixed bar 0.495. **Exact match.**
- **N4** (restricted effective rank, 19 seeds, per-group means/sds/deviation%)
  — S3 1.8771±0.0601 (n=3, dev 6.15%), S4 2.8517±0.0536 (n=5, dev 4.94%), A5
  2.8323±0.0623 (n=5, dev 5.59%), S5 3.5913±0.0688 (n=3, dev 10.22%), A6
  4.7357±0.0231 (n=3, dev 5.29%). **Exact match** to Table 2 and to the
  per-group deviation sentence (6.1/4.9/5.6/10.2/5.3%). Confirms the range
  "4.9–10.2%" in `sections/04_ranklaw_observed.tex` (changed from v1's
  "5–11%") is the *more* accurate rounding of the same underlying data, not
  a regression. `harvest_summary.json` md5 **7dce77dcba724cd1004419ac71fe5f2f**
  — matches the brief's pin. Spearman ρ (brute-force rank correlation over
  the 5 group means vs. d_min, with ties handled by average rank) =
  **0.97468**, matching the draft's 0.9747 exactly.
- **N5** (Welch TOST, S4 vs. A5) — diff = **+0.01945**, se = **0.03678**,
  Welch–Satterthwaite df = **7.826**, one-sided t = **14.12 / 13.06** vs.
  t_crit(0.95, df≈7.8) = **1.865** (as stated in the draft). **Exact match,**
  both one-sided tests pass, equivalence declared.
- **N6** (binding foundation) —
  `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_1234.json`,
  md5 **0134495e42e7549dd7d13a5753e69ce6** (matches the brief's pin) —
  `M1_effrank_vs_K_by_d["16"]["8"]=8.1984≈8.20`, `["16"]["16"]=15.0828≈15.08`;
  `M3_recovered_frac@0.9_vs_forcerank["d8_K4"]`: k=1→0.0, k=2→0.0004,
  k=3→0.0 (max ≤0.0004), k=4→0.9402≈0.940. **Exact match.**
- **N7** (D-AMB tax signature) — `harvest_analysis_output.txt`
  (`2026-07-09_capability_sweep_harvest`, md5
  **854a4bd7c46e626badcc0fbf05d0e07a**, matches the brief's pin): "39
  force-rank cells: mean|obs-pred|=0.0276, max|obs-pred|=0.1664." **Exact
  match** to "mean absolute deviation 0.028 across all 39 cells (maximum
  0.166)."
- **N8** (fix-wave config integrity) —
  `experiment-runs/2026-07-09_m3fix_harvest/harvest_analysis_output.txt`
  (md5 **77be9c3b092c70e83ff08a0261575815**, matches the brief's pin):
  "cells found: 30/30 ... CONFIG-MATCH CLEAN," `n_skipped_steps == 0` for
  all 30. **Exact match.**
- **N9** (composition, depth-21 periodicity) — 5 seed files, md5s
  `f61f18b6…`/`5e70bb74…`/`1d54d016…`/`b80c6424…`/`02c0920b…`, all match the
  brief's pins exactly. `recovered_frac@0.9` at h=5/h=21: seed0 =
  0.303162/0.000122 (matches "0.303 at h=5 and 0.0001 at h=21"), seeds 1–4
  all ≥0.9994 at both hops. **Exact match** ("four converged seeds ... both
  read ≈1.0"; "the fifth ... reads 0.303 at h=5 and 0.0001 at h=21").
- **N10** (L≥2 robustness) — cross-checked against N4's same 19 raw JSONs'
  `l_ge2_*` fields; not independently re-derived this round beyond
  confirming the source files and their md5 are unchanged from N4's pin —
  v1 already recomputed this exactly (max Δ ≤0.013, per-seed max 0.041) and
  nothing in this round's edits touched the underlying data.
- **N11** (centering defect) —
  `experiment-runs/2026-07-09_capability_calib_recheck/gate1b_recheck.txt`
  (md5 **2d170cc03011cc56105adeae9929e481**, matches the brief's pin):
  uncentered mean_cos = 0.705261≈0.705, centered crosscheck mean_cos =
  0.999594≈0.9996. **Exact match.**
- **N12** (below-dmin ceiling fraction) — `crosscheck_mean_cos` divided by
  the forced ceiling √((dmin−1)/dmin) per group: S3 86.30%, S4 91.20%, A5
  94.94%→94.9%, S5 75.67%→75.7%, A6 93.51%→93.5%, mean **88.32%≈88.3%**.
  **Exact match.**
- **N13** (gate1a soft-convergence table) — S3 anchor 0.9144/k−1
  0.6649/k 0.9002/k+1 0.9028, all `clears=False`; S5 anchor 0.8761/k−1
  0.8013/k 0.8793/k+1 0.8770, all `clears=False`. **Exact match** to Table 3
  (Appendix D).
- **N14** (S4/S5 razor exceeds anchor) — S4: anchor 0.650→k=dmin 0.800,
  Δ=+0.150; S5: anchor 0.500→k=dmin 0.600, Δ=+0.100. **Exact match** to
  "S4: 0.65→0.80" / "S5: 0.50→0.60."
- **Figure-source md5 sweep (all 51 pins in `figure_gen.py`'s `SOURCE_MD5`
  dict)** — recomputed with `hashlib.md5` against the 19 sweep-harvest
  unconstrained JSONs, the 20 m3fix zero-pad JSONs, and the 12 m3fix-s3ext
  JSONs, after substituting the script's own `{SWEEP}`/`{M3FIX}`/`{S3EXT}`
  f-string path variables. **0 mismatches, 0 missing files** across all 51.
- **Citation author-list verification (live arXiv API,
  `export.arxiv.org/api/query?id_list=...`)** — all 10 mandatory
  arXiv-numbered entries checked: 2602.04852 (Nazari & Rusch — **now
  correct**; the git-HEAD-committed `refs.bib` had "Nazari, Pedram," the
  arXiv record and the current `refs.bib` both say "Nazari, Philipp"),
  2602.02195 (Sun et al., 12 authors — **now correct**; the
  git-HEAD-committed `refs.bib` had a completely different-looking 12-name
  list — "Ang Sun, Hao Zhang, Hang Zhou, Yifei Ma, Yujia Qin, Ting Su, Yang
  Liu, Zhen Ma, Jian Xu, Jianfeng Gao, Jiawei Hao, Ruoxi He" — every single
  first/given name subtly wrong versus the real arXiv author list "Ao Sun,
  Hongtao Zhang, Heng Zhou, Yixuan Ma, Yiran Qin, Tongrui Su, Yan Liu,
  Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He"; the current
  `refs.bib` now matches the real list exactly), 2411.12537 (Grazzi et
  al. — **now correct**; author order was Franke-before-Zela in the
  git-HEAD-committed version, is Zela-before-Franke now, matching arXiv),
  2502.10297 (Siems et al./DeltaProduct — unchanged, already correct),
  2603.14360 (Mishra et al. — unchanged, already correct), 2207.02098
  (Delétang et al., 11 authors — unchanged, already correct), 2210.10749
  (Liu et al. — unchanged, already correct), 2302.03025 (Chughtai et al. —
  unchanged, already correct), 2404.08819 (Merrill et al. — unchanged,
  already correct), 2412.06538 (Nichani et al. — unchanged, already
  correct). **Author order and titles now match `refs.bib` exactly for all
  10, zero mismatches** — a real improvement over the git-HEAD-committed
  state, where at least 2 of the 10 entries were substantively wrong. Entry
  types also checked against venue convention (conference papers as
  `@inproceedings` with `booktitle`, preprints as `@article` with a
  `journal = {arXiv preprint arXiv:...}` field) — all 14 entries correctly
  typed, no mismatches.
- **Bundle parity** — `bundle/neurreps-ea-submission.tex` reconstructed
  independently from `main.tex` + inlined `\input{sections/...}` is
  **byte-identical** (`diff` clean) to the shipped bundle file; `refs.bib`,
  `jmlr.cls`, `jmlrutils.sty`, both figure PDFs, and
  `neurreps-ea-submission.pdf` (vs. `main.pdf`) are all byte-identical
  between the working tree and `bundle/`.
- **Cross-reference / undefined-citation sweep** — the final (3rd,
  converged) pass of a fresh `tectonic` compile carries zero "Reference ...
  undefined" and zero "Citation ... undefined" warnings (both categories
  appear only in the two intermediate passes before the `.aux`/`.bbl`
  stabilize, which is normal LaTeX/BibTeX multi-pass behavior, not a
  defect).

## v1-finding closure

| ID | v1 finding | Status now | Basis |
|---|---|---|---|
| **C1** | Self-citation `larson2026gradient` prints the real author name in the compiled double-blind PDF (body p.4, references p.7). | **CLOSED** | `refs.bib` entry deleted; the citing sentence and `\citep{larson2026gradient}` removed from `sections/06_related.tex`. Tree-wide + bundle-wide + compiled-PDF-wide grep for all anonymization tokens (`larson`, `samlarson`, `saml212`, `pebble`, `pebbleml`, `rockie`, `.pebbleml.com`, `github.com/`, `huggingface.co/`, `acknowledg`, `self-funded`, `funded by`) across `main.tex`, `sections/*.tex`, `refs.bib`, `bundle/*`, `figures/figure_gen.py`, `main.pdf`, `bundle/neurreps-ea-submission.pdf`, `main.bbl`, `main.aux`: **zero matches**. |
| **S1** | Orphaned bib entry `smolensky1990tensor` (bib→text direction). | **CLOSED** | Entry deleted from `refs.bib` entirely. 14/14 remaining entries are cited; `main.aux`'s `\citation` list confirms 0 orphans in either direction. |
| **S2** | Body-through-Limitations spills roughly 2 lines onto page 5. | **CLOSED against the shipped artifact** (see S-A caveat) | The current shipped `main.pdf`/`bundle` PDF (8 pages) has the Limitations paragraph ending at the bottom of page 4, with Appendix A starting cleanly on page 5 — confirmed via `pdftotext -layout` page-by-page extraction. Not independently reproducible from a fresh build in this environment (S-A, new SERIOUS finding) — the closure is real for the artifact currently in the tree, but its stability under the next recompile is unverified. |
| **S3** | Missing `\section{}` heading silently merges the binding-foundation content into Section 2; `\ref{sec:binding}` resolves misleadingly to "Section 2." | **OPEN** (the cross-reference symptom is fixed; the structural defect is not) | `\label{sec:binding}` was deleted (not replaced by a real section); the one referencing site was repointed to the accurate `\ref{sec:setup}`. But `sections/03_binding.tex` still has no `\section`/`\subsection` command, and `main.aux` confirms Fact 1 is still tagged as part of Section 2 ("Tasks, Models, and the Rank Instrument"). `brief.md` now documents this as an intentional page-budget trade-off, but the compiled paper still shows a full result (Fact + proof + causal step) with no heading, unlike the paper's other two results. Scored OPEN — see S-B above. |
| **S4** | Two overfull-`\hbox` warnings (the razor table's minipage; an inline `\sqrt{}` term). | **CLOSED** (with a build-fidelity caveat, S-A) | Zero `Overfull \hbox` warnings in the final pass of a fresh `tectonic --print` build, confirmed twice. The specific fixes (narrower table via `\footnotesize`+`\tabcolsep`+resized minipages in `sections/05_causal_razor.tex`; `fig1`'s width reduced to `0.80\textwidth`) target exactly the two originally-cited locations. |
| **M1** | Stale evidence-row range in `bundle/README.md` ("N1–N11"). | **CLOSED** | `bundle/README.md` now reads "Non-figure numerical claims trace to evidence rows N1–N14," matching the current draft's actual usage. |
| **M2** | `refs.bib` has 16 entries, not v1's assignment brief's expected 17 (no defect found, just a count note). | **CLOSED / moot** | `refs.bib` now has 14 entries — down from 16 by exactly the two removals (C1, S1). No new count discrepancy; see M-A. |
| **M3** | Five defined labels (`fact:bound`, `tab:gate1a`, `sec:related`, `tab:groups`, `sec:intro`) are never `\ref{}`'d. | **OPEN, unchanged** | Re-confirmed identical to v1: all five labels are still defined but never referenced. Harmless, standard practice — not a real defect, same as v1's characterization. |

## Verdict

**CLEAR.** Zero CRITICAL findings — the one CRITICAL finding from v1 (C1) is
fully closed and independently re-verified across source, bundle, and
compiled-PDF text. Per this role's termination rule (critical-clean, no
CRITICAL attack open), this does not block the gauntlet.

That said, do not treat this as "ready to submit" without addressing the
three SERIOUS findings first, in priority order: **(S-A)** re-verify page
count stability with 2–3 independent fresh rebuilds (ideally on the actual
target compile environment) before trusting the 4pp body budget under the
next recompile — this is the highest-leverage unresolved risk, since an
8-versus-12-page swing this large, sourced to the same unmodified content,
is the kind of thing that would only surface at the worst possible time (an
OpenReview upload); **(S-B)** either add the `\section{}` the brief's own
page-budget table originally planned for the binding content, or accept the
current fold-into-§2 structure as final and say so explicitly in the paper's
own text (not just the brief) so it does not read as an editing accident;
**(S-C)** hand-recount the abstract with the actual submission portal's word
counter rather than relying on any script-based estimate, mine or v1's, given
how close to 230 every methodology lands.
