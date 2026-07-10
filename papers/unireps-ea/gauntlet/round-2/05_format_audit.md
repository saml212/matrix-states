# Format and Acceptance Audit — UniReps EA "Dimension, Not Solvability" (round 2, stage 05)

Auditor: fresh-context format/acceptance auditor. Every recompute below was
run independently against the raw JSON/txt artifacts named in `brief.md`'s
claims-to-evidence map (not taken from the draft's or brief's own prose),
using `.venv/bin/python` (`DRY_RUN_BYPASS=1` prefix where the dry-run hook
gated a call). The draft was compiled with `tectonic main.tex` (clean exit,
no undefined-reference/citation warnings in the log). All ten citation keys
were cross-checked against `refs.bib`; seven of ten author lists were
independently re-verified against live arXiv listings (the other three —
`chughtai2023toy`, `huh2024platonic`, `barrington1989` — were already
live-verified in round 1's gauntlet or, for `barrington1989`, re-verified
here).

## 1. Summary

**3 critical / 4 serious / 3 minor. The draft as it stands on disk is NOT
submission-ready — but none of the three criticals require new evidence or
a content retraction; all three are mechanical fixes** (regenerate a stale
build artifact, add missing evidence-comment tokens whose underlying numbers
already check out, correct two bibliography author fields against the live
record). Every headline number I recomputed from the raw artifacts (the
five M1 group means/sds, Spearman ρ and its exact permutation null, the
Welch TOST diff/se/df/t1/t2, the five-group razor table and its 0.9×-anchor
bars, the S3 four-seed per-seed table and its seed-mean/fixed-bar/recomputed-
bar arithmetic, the D-AMB 39-cell tax stat, the centering-defect gap, the
geometric ceilings √((d_min−1)/d_min), and the length-robustness deltas)
**matches the draft exactly**, including two numbers I had to hunt for
outside their named artifact (see S-1). Cross-references, figure/table
labels, and bib-key resolution are all clean: zero broken `\ref`s, zero
citation orphans in either direction, zero anonymization leaks in the
content itself, zero banned-word hits, zero literal placeholders that would
print in the compiled body, abstract at 228 words (in the 200–230 band),
body ending on page 4 with References, appendix pages 5–7 — matching the
task brief's stated page layout exactly.

The one finding that would most embarrass the authors if missed is **C-1**:
the `bundle/` directory — the actual flattened, submission-ready package,
per its own README — was built before either round 1's or round 2's
gauntlet fixes landed. It still contains the retracted "restores recovery"
/ "step function" language round 1 explicitly fixed, and its `refs.bib` is
missing the two MUST-cite additions from round 1 (`chughtai2023toy`,
`huh2024platonic`). Submitting `bundle/unireps-ea-submission.pdf` today
would silently resubmit every CRITICAL finding both gauntlet rounds already
closed.

## 2. Critical

### C-1: `bundle/` is stale — predates both gauntlet rounds' fixes

**File(s):** `bundle/unireps-ea-submission.tex`, `bundle/unireps-ea-submission.pdf`, `bundle/refs.bib`
**Problem:** `bundle/` was generated (per file mtimes, 19:07–19:09) before
`main.tex` and every `sections/*.tex` file's most recent edits (19:48–20:19),
which carry round 1's CRITICAL fixes (FIX-1/2/3/4, the "restores recovery" →
"pins recovery under 0.9 by the metric's own geometry..." rewrite; two new
MUST-cite references) and round 2's CRITICAL fix (FIX-1, the abstract's
"in most groups" → "in every group" correction). Direct greps confirm:
`bundle/unireps-ea-submission.tex` still contains `restoring $\dmin$
restores recovery` (line 77) and `Figure~\ref{fig:razor} shows a step
function at $\dmin$` (line 257) — both phrases round 1's rebuttal
identified as CRITICAL overclaims and rewrote. `bundle/refs.bib` has 8
entries and the flattened tex has 9 citation uses, vs. the current tree's
10/10 — missing exactly the two round-1 MUST-cite additions. The bundle's
two figure PDFs are current (byte-identical to `figures/*.pdf`), so only
the text/bib are stale.
**Why it matters:** `bundle/README.md` describes this directory as the
thing that gets submitted ("`unireps-ea-submission.tex` is the flattened
single-file source... `unireps-ea-submission.pdf` is its compiled
anonymized submission build"). As-is, it is not the audited draft.
**Fix:** run `make bundle` (target exists, depends on `main.pdf`, already
wired to flatten `main.tex` + copy `refs.bib`/figures) after this report's
other fixes land, then re-diff `bundle/unireps-ea-submission.tex` against
the current section files to confirm parity before submission.

### C-2: Evidence-comment token entirely absent from the Abstract and the Introduction

**File(s):** `main.tex` (Abstract), `sections/01_intro.tex`
**Problem:** The acceptance mechanism this stage enforces requires a
`% <!-- evidence: Ux --> ` comment immediately after every numerical claim.
`grep -rn "evidence:" sections/ main.tex` returns **zero hits** in
`sections/01_intro.tex` and the Abstract block of `main.tex` carries no
tag anywhere, despite both restating headline result numbers: the abstract
states the five M1 means (`1.88/2.85/2.83/3.59/4.74`), "all 19 seeds," `ρ =
0.9747`, the TOST "difference 0.019" and "seven times the critical value";
the introduction restates "within 11% of d_min," "all 19 seeds," and `ρ =
0.9747`. None of these clauses is followed by a token. By the letter of the
check ("a number with no evidence comment is an untraceable number —
CRITICAL"), every one of these is a finding.
**Not a fabrication** — I independently recomputed all of these values from
the raw JSONs (§4 below) and they match exactly the correctly-tagged
canonical statements in `sections/03_convergence.tex` and
`sections/04_equivalence.tex` (both of which properly bracket their numbers
with `U1`/`U2` tags). The gap is purely mechanical: the token is missing
where the numbers are restated.
**Fix:** add `% <!-- evidence: U1 -->` / `% <!-- evidence: U2 -->` comments
after the relevant abstract clauses and after the introduction's
contribution-list sentences (LaTeX comments do not render — zero page-budget
cost, zero risk to the verified-clean tectonic compile).

### C-3: Two bibliography entries misattribute authorship (verified against live arXiv records)

**File(s):** `refs.bib`
**Problem:** Independently fetched the live arXiv abstract pages (not the
draft's or brief's transcription) for both 2026 preprints cited under
`nazari2026rank` and `sun2026staterank`:
- `nazari2026rank` (arXiv:2602.04852): bib lists `author = {Nazari, Pedram
  and Rusch, T. Konstantin}`. The actual first author is **Philipp
  Nazari**, not Pedram (confirmed via WebFetch of the abstract page and an
  independent WebSearch cross-check).
- `sun2026staterank` (arXiv:2602.02195): bib lists 12 authors — `Sun, Ang;
  Zhang, Hao; Zhou, Hang; Ma, Yifei; Qin, Yujia; Su, Ting; Liu, Yang; Ma,
  Zhen; Xu, Jian; Gao, Jianfeng; Hao, Jiawei; He, Ruoxi`. The actual
  author list (same 12 surnames, same order — confirming the paper
  identification is correct) is **Ao Sun, Hongtao Zhang, Heng Zhou,
  Yixuan Ma, Yiran Qin, Tongrui Su, Yan Liu, Zhanyu Ma, Jun Xu, Jiuchong
  Gao, Jinghua Hao, Renqing He**. Every one of the 12 first names in the
  bib entry is wrong while every surname and the ordering is right — a
  fabrication pattern (correct surnames, invented given names), not a
  transcription slip.
- Third 2026 preprint spot-checked, `mishra2026m2rnn` (arXiv:2603.14360):
  author list matches exactly, no issue.
- Four more-established citations re-verified live (`merrill2024illusion`,
  `nichani2025factual`, `siems2025deltaproduct`, `barrington1989`): all
  match exactly. `grazzi2025negative` has a minor author-order swap
  (Franke/Zela) — see S-4.
**Why it matters:** this is precisely the "entry that looks hand-written
rather than programmatically fetched... a hallucination risk" the format-
auditor brief calls out, now confirmed rather than merely suspected.
Misattributing a real paper's authorship (correct surname, wrong given
name) is an embarrassing, independently-checkable error a reviewer or the
paper's actual authors could catch immediately.
**Fix:** correct both author fields verbatim against the live arXiv
listings quoted above.

## 3. Serious

### S-1: The "100% power" claim and the "8k–40k" step-budget claim trace only to prose, not to any evidence-mapped raw artifact

**File(s):** `sections/04_equivalence.tex` (paragraph 1), `sections/02_setup.tex`
**Problem:** `04_equivalence.tex` states "a pre-run power simulation showing
100\% power to \emph{reject} equivalence at a true gap of 1.0
rank-units" — this number appears in `brief.md`'s Thesis prose (citing
`§1.4.2.1`) but has no row in the `brief.md` claims-to-evidence table
(U1–U7) and no `<!-- evidence -->` tag in the draft. I located the
generating script (`matrix-thinking/capability_separation/
marquee_power_simulation.py`) but no md5-pinned output artifact for it is
cited anywhere in the paper's evidence chain, so the number is currently
unverifiable by the mechanism this stage checks (it traces to a design-doc
paraphrase, not a raw artifact + token). Similarly, `02_setup.tex`'s
"Per-group step budgets (8k--40k)" is a real, verifiable number (confirmed
against `CAPABILITY_SEPARATION_DESIGN.md:101`, `S3=8K, S4=20K, A5=20K,
S5=8K, A6=40K`, consistent with round-1's attack/defense exchange) but
likewise has no Ux row and no tag.
**Fix:** either add a `U8` row to `brief.md` naming the power-simulation
script's output file (with md5) and tag both claims, or, if no archived
output file exists, soften the "100%" figure to a qualitative statement the
design doc supports without a specific percentage, and cite the step-budget
range to the design doc explicitly in-text.

### S-2: Two numbers sit mid-paragraph, not immediately before their governing evidence tag

**File(s):** `sections/05_causal.tex`, `sections/07_appendix.tex`
**Problem:**
- `05_causal.tex`: "the same cells' recovery cosines (0.61--0.84) sit under
  that ceiling..." — independently recomputed and confirmed correct
  (per-group k=d_min−1 cosines: S3 0.610, S4 0.745, A5 0.775, S5 0.655, A6
  0.836, range 0.610–0.836 → "0.61–0.84") — but the nearest `U3` tag comes
  a full sentence later, after the S4/A5/S5/A6 sufficiency-bar numbers,
  not immediately after this clause.
- "An earlier **58-cell sweep**..." appears in both `05_causal.tex` and
  `07_appendix.tex`; independently confirmed accurate (`experiment-runs/
  2026-07-09_capability_sweep_harvest/results/` contains exactly 58
  non-calibration result JSONs, and `harvest_analysis_output.txt` line 1
  says "INVENTORY: 58/58 cells present"), but in both locations the nearest
  tag (`U6`) arrives 2–3 sentences later, after the "39 force-rank cells"
  sub-claim.
**Fix:** add a second `<!-- evidence -->` comment immediately after each of
these two clauses (same row id, `U3`/`U6` respectively) rather than relying
on the paragraph-terminal tag to cover the whole paragraph.

### S-3: U5's parenthetical "per-seed at most 0.041" is not derivable from the raw artifact named in its evidence row

**File(s):** `sections/03_convergence.tex` (footnote), `sections/07_appendix.tex`
(length-robustness paragraph); brief.md row U5
**Problem:** U5's named raw artifact is `harvest_analysis_output.txt`
(md5-verified: `854a4bd7c46e626badcc0fbf05d0e07a`, matches). That file
reports only **per-group mean** deltas (max 0.0126, S3–A6 rounding to the
draft's correctly-cited "0.013"). It contains no per-seed figures. I traced
the "(per-seed at most 0.041)" parenthetical to the individual per-seed
result JSONs (`experiment-runs/2026-07-09_capability_sweep_harvest/
results/*__unconstrained__seed*.json`, fields `mean_cos` /
`l_ge2_mean_cos`) — those files are cited under **U1**, not U5 — and
independently recomputed the max per-seed `|mean_cos − l_ge2_mean_cos|` as
0.0412 (S4, seed 3), which rounds to the draft's "0.041" and is correct.
**Fix:** either add the per-seed JSONs to U5's artifact list in `brief.md`
(they are already md5-pinned under U1, so this is a documentation-only
change) or split the sentence so the per-group figure keeps its U5 tag and
the per-seed figure carries a `U1` tag.

### S-4: `grazzi2025negative` lists two authors in the wrong order

**File(s):** `refs.bib`
**Problem:** Bib lists `Grazzi, Riccardo and Siems, Julien and Franke,
Jörg K. H. and Zela, Arber and Hutter, Frank and Pontil, Massimiliano`.
The live arXiv listing (independently fetched) orders them Grazzi, Siems,
**Zela**, **Franke**, Hutter, Pontil — Franke and Zela are swapped. Both
names are correct and present; only their position differs.
**Fix:** swap the two entries in the `author` field to match the published
order.

## 4. Minor

### M-1: `Makefile`'s `anon` target references a file that does not exist
`bundle`/`main` builds work; `make anon` (line 13, depends on
`main-anon.tex`) would fail — no such file exists in the directory. The
default `main.tex` already compiles to the fully anonymized build (verified
via `pdftotext`: page 1 renders "Anonymous Author(s) / Affiliation /
Address / email", the NeurIPS style's own anonymous-mode override, not the
literal `email@example.com` placeholder text written in `\author{}`), so
this does not block the actual submission path (`make` / `make bundle`).
Dead Makefile target; remove or wire it to the current single-file
anonymized-by-default design.

### M-2: `main.tex` header comments flag two PI-pending decisions (informational, not a rendering defect)
Lines 7–10 and 45–46: `% TITLE: PENDING-USER` and `% AUTHOR BLOCK:
PENDING-USER`. These are LaTeX comments (invisible in the compiled PDF,
confirmed by `pdftotext` inspection) and match `brief.md`'s own disclosed
"PI placeholders (deliberately left open)" section — title candidate (A)
vs (B), and the camera-ready author block. Not a placeholder that prints;
listed here only so it is not lost before the actual PI sign-off.

### M-3: `nichani2025factual`'s `booktitle` field carries a non-standard `, Spotlight` annotation
`booktitle = {International Conference on Learning Representations (ICLR),
Spotlight}` — the paper's acceptance tier ("Spotlight") is appended inside
the venue field rather than as a separate `note`. Not incorrect, but
inconsistent with the other `@inproceedings` entries' formatting.

## 5. Counts

- **Body word count (Introduction–Limitations, markup-stripped estimate):**
  ≈1,626 words across 6 sections, fitting the compiled 4-page (two-column)
  budget alongside 2 figures.
- **Abstract word count:** 228 (target band 200–230 — in band).
- **Figures referenced vs. present:** 2 referenced (`fig:convergence`,
  `fig:razor`) / 2 present in `figures/` — no orphans. Both PDFs verified
  byte-reproducible from `figure_gen.py` (identical to a fresh regeneration
  after stripping the embedded matplotlib `CreationDate` timestamp; PNG
  renders pixel-identical).
- **Citations in-text vs. in-bib:** 10 unique `\citep`/`\citet` keys used,
  10 entries in `refs.bib`, exact 1:1 match — **zero orphans either
  direction**.
- **Cross-references:** 23 `\ref` uses, all resolve to an existing
  `\label` (8 section/appendix labels, 2 figure labels, 2 table labels);
  0 broken. `tectonic main.tex` compiles clean (exit 0), zero
  undefined-reference or undefined-citation warnings in `main.log`.
- **Page layout:** 7 pages total; body ends and References begins on page
  4 (confirmed via `pdftotext` page-by-page dump); Appendix A–D spans pages
  5–7 — matches the task brief's stated layout exactly.
- **Anonymization matches:** 0 in actual content (`main.tex`, `sections/`,
  `refs.bib`, `figures/figure_gen.py`, `bundle/*.tex`, `bundle/*.bib`). The
  only grep hits are inside `bundle/neurips.sty` / the local `neurips.sty`,
  which are the unmodified NeurIPS template's own `\acksection` macro
  definitions (never invoked by this paper) — not a content leak.
- **Banned-word hits:** 0. **Contractions:** 0. **First-person "I":** 0
  (the two `I` hits found are the math symbol `\hat{Q} = I` and `\hat{Q}
  \approx I`, not the pronoun). **Literal placeholders that would print**
  (`TODO`, `pending`, `forthcoming`, `will be added`, `[CITE]`,
  `\textcolor{red}`, `Table X`, `Figure X/Y`): 0.

## 6. Recomputation log (mandatory spot-checks, all confirmed exact)

- **U1** (five group means/sds, ρ, exact null, band, deviations): means
  1.877±0.060 / 2.852±0.054 / 2.832±0.062 / 3.591±0.069 / 4.736±0.023
  (sample sd, n=3/5/5/3/3=19) — exact match. Spearman ρ = 0.974679
  (rounds to 0.9747) against the tie-respecting d_min sequence [2,3,3,4,5]
  — confirmed to be the maximum achievable over all 120 permutations
  (achieved by exactly 2/120). Exact permutation null P(ρ≥0.8) = 8/120 =
  6.67% — confirmed (an initial naive re-check that ignored the S4/A5 tie
  gave 5/120; correcting for the tie reproduces the draft's 8/120 exactly).
  All 19 seeds confirmed inside `[0.7,1.3]·d_min`. Deviations: S5 10.217%
  (draft: 10.2%), all others ≤6.15% (draft's "within 11%" bound holds).
- **U2** (Welch TOST): diff=0.01945 (0.019), se=0.03678 (0.037), df=7.826
  (7.8/7.83), t1=14.12, t2=13.06, tcrit(df,0.95)=1.865 — exact match.
- **U3** (razor table): k=d_min−1 recovery = 0.000 in all 5 groups; k=d_min
  clears 0.9×anchor bar in S4/A5/S5/A6 (0.800/0.700/0.600/0.650 vs.
  0.585/0.630/0.450/0.585) — exact match; S3 seed0 alone does not clear
  (0.450 < 0.495), consistent with the draft routing S3 to the 4-seed
  extension.
- **U4** (S3 4-seed table): seed0–3 anchor/k-1/k/ownbar/clears all match
  the draft's Table (`tab:s3seeds`) exactly; seed-mean 0.5625 clears the
  fixed 0.495 bar; recomputed bar from the 4-seed anchor mean (0.6375) is
  0.57375 (0.574), 0.0112 (0.011) above the seed-mean — exact match.
- **U6** (D-AMB tax): 39 force-rank cells, mean|obs−pred|=0.0276 (0.028),
  max=0.1664 (0.166) — exact match.
- **U7** (centering): uncentered 0.705261, centered 0.999594 — exact match
  against `gate1b_recheck.txt` (md5 confirmed).
- **Geometric ceilings** √((d_min−1)/d_min): 0.7071/0.8165/0.8165/0.8660/
  0.8944 for S3/S4/A5/S5/A6 — exact match, all below the 0.9 threshold.
- **Md5s:** `harvest_summary.json` (7dce77dcba724cd1004419ac71fe5f2f),
  `harvest_analysis_output.txt` (854a4bd7c46e626badcc0fbf05d0e07a),
  `gate1b_recheck.txt` (2d170cc03011cc56105adeae9929e481) — all three match
  `brief.md` exactly. Running `figures/figure_gen.py` end-to-end against
  the live repo asserts every one of its 40+ pinned source md5s and
  completes without error.

## 7. Verdict

**Blocks submission until C-1, C-2, and C-3 are fixed.** All three fixes
are mechanical (regenerate a build artifact; add comment tokens over
already-verified numbers; correct two bib author fields against the live
record) — none requires new experiments, a content retraction, or
re-opening either gauntlet round's adjudicated science. S-1 through S-4
should be fixed in the same pass but do not, on their own, block
submission. Re-run this stage after the fixes land to confirm the bundle
parity and the new tags compile clean.
