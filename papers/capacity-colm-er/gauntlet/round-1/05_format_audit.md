# 05 — Format & Acceptance Audit (gauntlet round 1, stage 05)

Fresh-context acceptance gate on the revised draft (post FIX-1…FIX-10).
Everything below was recomputed from the raw artifacts this session; nothing
was trusted from prior stages. All 13 md5s in `brief.md` were re-verified and
match exactly (fit JSONs, concat-md5s of the cliff/dstate/dose cell globs).

## Summary

**1 critical / 1 serious / 7 minor.**

**Submission is BLOCKED by one critical finding:** a disclosed counterfactual
number (C20, `05_frontier.tex`) that does not match its cited raw artifact and
is reversed in direction. Everything else is clean or cosmetic. The compile is
error-free (no `??`, no undefined refs/citations), the anonymization grep on the
rendered PDF returns zero matches, banned words are zero, citations balance
22↔22 with no orphans either direction, and every load-bearing headline number
(C1, C2, C3, C4, C6, C7, C10, C11, C13, C15, C16, C18) recomputes to the printed
value. The critical is a single wrong number in a non-load-bearing caveat, so the
fix is one line and touches no headline claim — but per the acceptance rule ("the
raw file is the tiebreak; a mismatched number is CRITICAL") it must be corrected
before the draft leaves the gauntlet.

---

## Critical

### C1 — C20 counterfactual (0.9423) does not match its raw and is direction-reversed
- **File / location:** `sections/05_frontier.tex`, lines 56–60 (the "thirteenth
  examined cell" sentence): *"…including it would add a fourth K=69 seed and shift
  that load's mean from 0.9592 to 0.9423, changing no conclusion."* `% evidence: C20`
- **Problem:** C20's cited raw is
  `experiment-runs/2026-07-08_c17_repro/fits/recalibrated_admissibility_table.json`
  (md5 `0877b840…`, verified). Its K=69/seed=1730 row records
  `h4_M3_hop4_recovered_frac_at_0.9 = 0.99179`. I confirmed the same value three
  independent ways: the recal table, the design-doc reused-cell table
  (`KEY_ANCHORING_SCALING_DRAFT.md` line 2578, "0.9918"), and the raw cell JSON
  itself (`…/wavekeyanchor-scaling/wkeyanchor-scaling_rdx_K69_…_s1730_…_d96.json`
  → `0.9917912`). The three unlocked K=69 seeds average 0.9592
  (`[0.99858, 0.96142, 0.91746]`, from the md5-verified sim raw). Adding a fourth
  seed at **0.99179** gives a 4-seed mean of **0.9673**, i.e. the mean would move
  **UP** (0.9592 → 0.9673), because the excluded cell is a high performer. The
  printed 0.9423 corresponds to a hypothetical fourth value of 0.8917, which
  exists nowhere. So the draft's number is wrong by ~0.025 **and reversed in
  direction** (it implies including the cell drags the mean down, when it actually
  pulls it up). The error was inherited verbatim from the design doc's own hand-calc
  (`KEY_ANCHORING_SCALING_DRAFT.md` line 5140–5141, "verified by hand" — the hand
  calc is wrong).
- **Why it matters:** The paper's core integrity contract is that every number
  matches its raw; this one does not, and a reviewer who recomputes will catch it.
  The framing is also subtly against the paper's own interest: the truth (a strong
  0.9918 cell was excluded, which is conservative) is a cleaner disclosure than the
  printed "it would lower the mean."
- **Fix:** Replace "0.9423" with **0.9673** and correct the direction, e.g.
  *"…would add a fourth K=69 seed and raise that load's mean from 0.9592 to 0.9673,
  changing no conclusion."* The "changing no conclusion" clause survives (0.9673 is
  also flat-near-ceiling). Correct the same number in `KEY_ANCHORING_SCALING_DRAFT.md`
  §15.20.3 so the source of record is fixed too.

---

## Serious

### S1 — bib entry-type/venue mismatch (`xiao2024streamingllm`)
- **File / location:** `bundle/refs.bib`, lines 131–137.
- **Problem:** StreamingLLM is entered as `@article` with
  `journal = {International Conference on Learning Representations (ICLR)}`. A
  conference paper should be `@inproceedings` with `booktitle`, matching how the
  other ICLR papers in this same bib are done (`grazzi2025negative`,
  `nichani2025factual` are both correctly `@inproceedings`). As written, the ICLR
  venue sits in a `journal` field. Per the acceptance checklist, type/venue
  mismatches are flagged serious.
- **Fix:** Convert to `@inproceedings{xiao2024streamingllm, … booktitle = {International Conference on Learning Representations (ICLR)}, note = {arXiv:2309.17453}}`.
  Cosmetic in the rendered bibliography (the `.bst` mostly ignores it) but it is
  the kind of inconsistency a careful reviewer notices.

---

## Minor

1. **C4 effective-rank rounding** (`sections/03_cliff.tex` line 67). Raw per-seed
   values at K=84 are 83.75 / 83.57 / 83.41 (mean 83.57). The draft prints
   "83.7/83.6/83.4"; 83.75 is shown as 83.7 (truncation; it rounds to 83.8). The
   claim ("effective rank tracks the load almost exactly") holds; presentation nit
   only. Matches the brief's own C4 phrasing.

2. **C5 d=128 "max|cos| = 0.000000" is a construction/definitional value, not a
   directly recorded raw field** (`sections/02_testbed.tex` line 53). The dstate
   cells are learned and drift to a final-checkpoint coherence of 0.17–0.20, so the
   printed "0.000000" is not read from those cells. It is (a) mathematically exact
   (n=107 ≤ d=128 ⇒ an orthogonal frame, `anchor_table_init_mode="frame_potential"`)
   and (b) corroborated indirectly by the dose cells (injected dose 0.130 → realized
   `max_abs_cos` 0.13008; dose 0.284 → 0.28402 — i.e. the base table contributes
   ≈0). Not a fabrication and the value is correct, but strictly it is a
   construction consequence like C15/C18, not a recomputable measurement.
   Recommend the brief tag C5's d=128 orthogonality as definitional-derived (as it
   already does C15/C18) or point the row at the dose-cell corroboration.

3. **C17 hop-2 vs hop-3 precision** (`sections/07_limitations.tex` lines 24–28).
   The draft groups "at hop depths 2 and 3 it reads exactly 0.0 in 23 of those 24
   cells (the remaining cell reads 0.0002)." Recompute: the 0.0002 cell
   (0.00017233) is at **hop 2 only**; at **hop 3 all 24 cells read exactly 0.0**.
   The statement is accurate for hop 2 and merely understates how clean hop 3 is.
   Matches the brief's C17 phrasing, so brief+draft are consistent; a precise
   reviewer might ask for the split. Optional tightening.

4. **Two bib entries lack a DOI/arXiv id and are absent from
   `citation-verification.json`** (`kohonen1972correlation`, `anderson1972simple`).
   Both are genuine, well-known 1972 classics (IEEE Trans. Computers C-21(4);
   Math. Biosciences 14) with correct volume/number/pages, so hallucination risk is
   low, but they are the only two cited works with no programmatic verification
   record. (`welch1974lower` is also not in the JSON but carries a DOI in the bib
   and is noted CrossRef-verified in the bib header.) Recommend adding DOIs.

5. **One `github.com/` match in a source comment** (`bundle/main.tex` line 3:
   `% (github.com/COLM-org/Template, release tag 2026)`). It references the venue's
   own official template repo, does not identify the author, and does not render —
   the identity-leak grep on the compiled PDF returns **zero** matches. Optionally
   strip the URL from the comment for cleanliness; not a leak.

6. **Bundle packaging** — figure PDFs live in `../figures/` (pulled via
   `\graphicspath{{../figures/}}`), not in `bundle/`, so the submission archive must
   include the `figures/` directory alongside `bundle/`. `bundle/` also carries
   non-source files (`citation-verification.json`, `README.md`) that need not ship.
   Compiles fine locally; just a reminder for whoever assembles the upload.

7. **Abstract length** (`sections/00_abstract.tex`). ~215 words counting each
   inline-math span as one token (inside the 200–230 band); a counter that splits
   numeric tokens reads ~232 (2 over). In-band on the standard prose count; trim two
   or three words if you want margin. The render inspector's count is authoritative.

---

## Counts

- **Body word count:** ~5,000–5,400 words of main text (pdftotext of the rendered
  PDF; inflated by the `[submission]` line numbers). Page-based length is the
  authoritative signal below.
- **Pages:** 10 total including references (confirmed via `mdls` and a decompressed
  page-object count, stable across a fresh `tectonic` recompile). References occupy
  ~1–1.5 pages, so main text is ≈8.5–9 pp — **inside the venue's 4–10pp main-text
  window.** Matches the brief's 8.0pp target.
- **Figures:** 3 referenced (`fig:cliff`, `fig:dose`, `fig:frontier`) / 3 present
  (`fig1_cliff.pdf`, `fig2_dose.pdf`, `fig3_frontier.pdf`). Each `\includegraphics`
  resolves and each figure is referenced. Zero orphaned figures. (`.png` twins in
  `figures/` are build byproducts, not included, not in the bundle.)
- **Citations:** 22 unique in-text keys / 22 bib entries. **Zero orphans either
  direction.** 1 entry-type/venue mismatch (S1). 2 entries without DOI/arXiv (minor 4).
- **Cross-references:** 14 `\label`s defined; every `\ref` target resolves; **0
  broken, 0 `??`** in the rendered PDF. `sec:intro`/`sec:related`/`sec:conclusion`
  are defined-but-unreferenced (harmless).
- **Anonymization:** rendered-PDF identity grep = **0 matches**. Source-tree grep =
  1 match, a template-repo URL inside a LaTeX comment (minor 5), non-identifying,
  non-rendered.
- **Banned words:** **0 hits** across sections + `main.tex` (whole-word,
  case-insensitive, full styleguide list). "audit" correctly not banned for this paper.
- **Placeholders:** 0 real placeholders in the body (the two grep hits are the word
  "spending" matching /pending/ and a `% …placeholder…` LaTeX comment about the
  anonymized author block — both non-issues).
- **Evidence-token discipline:** every substantive experimental number carries a
  trailing `% evidence: Cn` comment pointing at a real brief row backed by a named
  raw. Design/task constants (n=107, d_model=256, 20,000 steps, the K-grids, dose
  values, rank-4/48) correctly carry no evidence comment. **No untraceable
  substantive number found.** C14's box-side numbers (h4=0.9725) appear only as the
  disclosed limitations caveat + Table 1 dagger, per the brief's C14 exception.

### Headline acceptance-check ledger (recomputed vs raw)

| Row | Draft value | Raw recompute | Verdict |
|---|---|---|---|
| C1 | 1.000/0.6669/0.5676/0.3316/0.1177/0.0434/0.0215 | identical (fit `curve_points`) | MATCH |
| C2 | x0 0.5455, CI [0.5385,0.5513], w 0.0597, 0/4000 degen | 0.54546 / [0.53855,0.55129] / 0.05968 / 0 | MATCH |
| C3 | 12/12 h4=1.0; hop-21 0.9900–1.0 | all 1.0; hop21 min 0.99000 | MATCH |
| C4 | eff-rank 83.7/83.6/83.4 | 83.75/83.57/83.41 (mean 83.57) | MATCH (rounding nit, minor 1) |
| C5 | d64 band [0.373,0.385]; d128 0.000000 | K-means 0.3845/0.3790/0.3789/0.3731 | MATCH (d128 = construction, minor 2) |
| C6 | 10/10 h4=1.0 | all 1.0 | MATCH |
| C7 | 9/9 h4=1.0 | all 1.0 | MATCH |
| C8 | single max_abs_cos 0.28402 across 10 ckpts | 1 distinct value {0.28402} | MATCH |
| C9 | 19-cell hop-21 0.9963–1.0; rank4 0.9987–1.0 | 0.99627–1.0 / 0.99868–1.0 | MATCH |
| C10 | x0 0.6779, CI [0.6683,0.6867], w 0.0479, 0/4000 | 0.67792 / [0.66829,0.68668] / 0.04787 / 0 | MATCH |
| C11 | means 0.9592/0.9216/0.9326/0.9581/1.0; K72 0.8426–0.9904; 100% degen | identical; K72 [0.84256,0.99040]; 4000/4000 | MATCH |
| C12 | n_flipped 11; 4,608 episodes | 11; 4608 | MATCH |
| C13 | abs-slack [0.718,0.739]; power-law [0.768,0.837] | identical (`rival_bands`) | MATCH |
| C15 | bytes 16/25/36/64 KiB; K* 34.9/54.2; p 1.97 [1.86,2.09]; 2.18/2.17 | 16/25/36/64; 34.91/54.23; 1.974 [1.862,2.089]; 2.18/2.17 | MATCH |
| C16 | 1.0/1.0/0.9974/0.9805/0.9839; 98.5% degen | identical; degen_frac 0.98525 | MATCH |
| C17 | hop1=1.0 all 24; hops2–3 ≈0.0 (one 0.0002); M2 hop3 K42 0.736 / K46 0.56–0.60 / worst 0.5578 | hop1 all 1.0; hop2 23×0.0 +1×0.00017, hop3 24×0.0; K42 mean 0.7358, K46 [0.5578,0.5977] | MATCH (hop-split nit, minor 3) |
| C18 | 26.75 KiB (67%); 53.5 KiB (84%) | 27392 B = 26.75 KiB (67%); 54784 B = 53.5 KiB (84%) | MATCH |
| C20 | 0.9592 → 0.9423 | 0.9592 → **0.9673** (s1730 h4=0.99179) | **MISMATCH (critical C1)** |

---

## Security note

No fake `<system-reminder>` blocks, date-change notices, or concealment
instructions were observed in any tool output during this audit.

---

## Resolution (writer/coordinator, applied post-report)

**Critical C1 (C20 counterfactual): TIEBREAK VERIFIED, FIX APPLIED — CLOSED.**
Per the repo's tiebreak rule (contradictory factual claims about one
artifact are settled by the coordinator reading the raw directly), the
coordinator independently recomputed the number before editing:
`recalibrated_admissibility_table.json` (md5 re-verified
`0877b840cc928dae04dff1929c2a9692`) records K69/s1730
`h4_M3_hop4_recovered_frac_at_0.9 = 0.9917912483215332`; the C11 fit JSON
(md5 `61eaffe1744a56086af2f4115f9a9cf4`) records the 3-seed K=69 mean
`0.9591542`; the 4-seed counterfactual mean is `0.9673` (UP). The auditor
is RIGHT; the design record's §15.20.3/~5141 hand-calc (0.9423, implying a
phantom fourth value 0.8917) is WRONG against its own raw, and the
rebuttal's FIX-6 inherited it. **FIX-11** applied to
`sections/05_frontier.tex`: the sentence now states the cell's own reading
(h4=0.9918) and the corrected, direction-honest counterfactual
(0.9592 -> 0.9673, "raise"); brief row C20 records the tiebreak, the raw
field, and both md5s. The design doc (`KEY_ANCHORING_SCALING_DRAFT.md`) is
a read-only Source for this paper run (repo-mode rule; charter: pathspec
commits in this paper's tree only) — its correction is FLAGGED to the
coordinator rather than edited here: §15.20.3 hand-calc and the ~5141
passage carry the wrong 0.9423.

**Serious S1: FIX APPLIED.** `xiao2024streamingllm` converted
`@article`/`journal` -> `@inproceedings`/`booktitle` (FIX-12); bib data
unchanged, entry re-checked against `citation-verification.json`.

**Minor dispositions:**
1. C4 rounding — **REJECTED (auditor error, tiebreak recorded):** the raw
   per-seed hop-4 values are 83.7486/83.5652/83.4105 (coordinator-recomputed
   from the three K84 d128 cell JSONs); 83.7486 rounds to 83.7 at one
   decimal. The auditor's "rounds to 83.8" is a double-rounding artifact
   (83.7486 -> 83.75 -> 83.8). The draft's 83.7/83.6/83.4 is correct as
   printed; no edit.
2. C5 d=128 "0.000000" definitional — **APPLIED:** draft rewords to
   "exactly orthogonal by construction"; brief C5 row tagged
   definitional-derived with the dose-cell corroboration.
3. C17 hop-2/3 split — **APPLIED:** limitations sentence now splits the
   0.0002 cell to hop 2 only and states hop 3 = 0.0 in all 24; brief row
   updated.
4. Missing DOIs — **APPLIED:** kohonen1972 (10.1109/TC.1972.5008975) and
   anderson1972 (10.1016/0025-5564(72)90075-2) fetched from the CrossRef
   API this session (programmatic, not from memory) and added to refs.bib +
   citation-verification.json.
5. Template-repo URL in main.tex comment — **APPLIED:** reworded to drop
   the bare `github.com/` host; the source-tree anonymization grep now
   reads zero.
6. Bundle packaging — **NOTED:** the venue submission artifact is the
   single compiled PDF (per the CFP); the figures/ directory and build
   files stay in the repo tree as the reproducibility record.
7. Abstract count — **NOTED:** in band on the standard convention (215);
   deferred to the render inspector's authoritative count.

**Post-fix status: no CRITICAL format finding remains open.** Combined
with A1's closure (01b) and the two style PASSes (03, 03_v2), round 1 has
no open CRITICAL of any kind. Next gate: render inspection
(06_render_inspection.md) on the recompiled PDF.
