# 07 — Final Review (Stage 9, Fable, fresh context)

**Paper:** `papers/measurement-ws/` — "The Instrument Is the First Suspect:
Six Broken Lenses in One Empirical Program"
**Artifact reviewed:** `bundle/measurement-ws-submission.pdf`
(md5 `102a44ddb10d021e7145b52aa0425e9e`, byte-identical to `main.pdf`),
all 8 pages read as rendered page images; plus `brief.md`, all six prior
round reports (01, 01b, 02, 03, 04, 05, 06), the section sources, and the
raw artifacts named below, re-read directly.
**Date:** 2026-07-11

## Verdict: READY-AFTER-CHANGES

One SERIOUS numeric defect found (change 1 — a margin claim in §3 that the
raw contradicts at file scope; no conclusion flips). Everything else
standing between this tree and submission is process, not text: the
detector discharge (change 2) and the Jul-11 venue confirmation with the
recorded FIX-4 reversal decision (change 3). After change 1 lands and the
two gates discharge, the verdict converts to
SUBMISSION-READY-pending-venue-confirmation.

---

## 1. Full-render read (all 8 pages)

**Does each case study land as evidence for the discipline, or does the
paper read as a list of mistakes?** It lands as evidence. The five rules
(§2) are each anchored to the incident that exposed them, and every case
names the rules as they fire: §3 walks a pre-registered precedence order
mechanically (Table 2) and closes with the discipline *withholding* two
findings it could not support; §4's punchline is explicitly "the
dissociation, not the zero, is the finding (Rule 5)" and its second defect
motivates Rule 3's positive-control clause on the page; §5 is
structured as signature → tiebreak (four graded grounds) → teeth →
re-metric, i.e., the discipline itself is the narrative spine. §7 then
directly rebuts the laundering objection with the two lens-independent
genuine failures and Case I's refusals. The catalogue (Table 1) gives the
six incidents one comparable schema (defect / signature / resolution), so
the paper reads as one instrument-adjudication record, not six anecdotes.

**Is the freshest chapter (§2.31a/§2.32 — the broken-lens signature +
shuffled-target falsifier) integrated or bolted on?** Integrated — it is
the paper's centerpiece, not an appendage. It is §5 (the longest case, per
the brief's page budget), it owns the paper's only body figure (Fig. 1),
the abstract's falsifiability sentence rests on its teeth table
(Table 5), Rule 1's "(the rule that fired in §5)" plants it in §2, and §7's
boundary argument depends on its FALSIFY-stands outcome. The arc
signature → tiebreak → teeth → re-metric is the thesis in miniature.

**Render quality.** Concur with 06: 0 critical / 0 serious. Body ends on
page 4 (references p5, appendix pp6–8); no unresolved refs; anonymized
author block intact; math renders cleanly throughout. 06's two minors
(Fig. 1 in-plot font at the low end of legible; Fig. 2(b) tick-label
crowding at (iii)/(iv)) are confirmed and remain optional polish
(change 4). One placement note of my own: Table 1 is first referenced in
§1 (page 1) but floats to page 7, after References and Appendix A — a
body-only reviewer flips six pages to find the catalogue. Optional
(change 5); unavoidable in section order under the appendix fallback,
but the float could anchor at the top of the appendix (page 6).

**Anonymization re-check (this session):** grep of the bundle `.tex`,
`refs.bib`, and `pdftotext` output against the full
`venue-requirements.md` token list — zero hits in rendered content; the
bundle TeX carries zero comment lines (format-audit M1 discharged via
`flatten_bundle.py`). Style stage (03) PASS stands; abstract 211 words.

---

## 2. Spot-checks: brief rows → raw artifacts (md5s recomputed this session)

All four raw files named below md5-matched their brief pointers exactly
before reading. Every number was recomputed from the raw JSON/log, not
read from any prior round's prose.

### 2.1 Row X2 — the 25-vs-30 correction (gauntlet r1 A1) — VERIFIED EXACT

`crosscheck_lens_verdict_output.json` (md5 `f26a769d5c263af224c91d39bd83710b`):
`flat_per_cell_table` holds exactly **62 rows = 37 `arm3_beta02` + 25
`arm2_beta01`**; the 25 are precisely {A5,A6,S3,S4,S5} × seeds {0–4}.
Lens agreement on arm-2 is exact: max |primary − crosscheck| over all 25
cells × 2 depths (50 checkpoint values) = **0.0**. Converged exemplar A6
n_h=4 seed0: final_loss 8.15e-5, primary far64 0.050, crosscheck 1.000 —
the §5 sentence verbatim. Disagreement ≥0.5: 13 cells, all converged;
>0.3: 16 cells, 16/16 converged; converged total 17 = Fig. 1's legend
(n=17 / n=45). The corrected "25" is right against the raw; the false
"30" appears nowhere.

### 2.2 Row X4 — the teeth (the thesis's falsifier) — VERIFIED EXACT

Same file, `teeth_control` block: three pre-registered converged
checkpoints (A6 nh4 s0 / S5 nh4 s0 / S4 nh2 s2), real crosscheck reads
1.000/0.800/1.000 all flagged `real_bit_identical: true` against the
committed values, shuffled reads 0.000/0.000/0.050, rule "falsifier fires
if any shuffled ≥ 0.5", `all_pass: true`. Matches §5 "Teeth." and
Table 5 to the digit.

### 2.3 Row I2 — the "over 7,000×" re-pairing (format-audit M2) — **DEFECT FOUND**

`diag_ns_admission_result.json` (md5 `d0d940d0afcdf30af8f45497168fb297`):
the file's `anchor_table_ns_sweep` holds **six** audited anchor tables —
4 FAILING (K78 s1840, K84 s1940, K90 s2040, K72 s1742) + 2
PASSING_CONTROL (K69 s1731, K72 s1741) — not one. At the operative
n_iter=20:

| entry | max_resid | mean_resid |
|---|---|---|
| FAILING K78 s1840 | 1.4164e-06 | 1.1954e-06 |
| FAILING K84 s1940 | 1.4679e-06 | 1.2813e-06 |
| FAILING K90 s2040 | **1.5555e-06** | 1.3981e-06 |
| FAILING K72 s1742 | 1.3109e-06 | 1.0829e-06 |
| PASSING K69 s1731 | 1.3124e-06 | 1.0358e-06 |
| PASSING K72 s1741 | 1.2822e-06 | 1.0991e-06 |

File-global max = **1.5555e-06 → 6,429× below the 0.01 tolerance** (also
the worst multiplier across all n_iter ∈ {20,24,28,32,40}). §3's shipped
sentence — "residuals at most 1.4×10⁻⁶, over 7,000× below tolerance,
identical across failing and passing cells" — quotes sweep[0]'s statistics
as if file-global, while its own "across failing and passing cells" clause
fixes the population at all six tables; two of the six exceed the quoted
max. The brief's I2 row carries the same error ("max 1.416e-06 → 7,060×
... re-paired to the conservative max per format-audit M2"): M2's
re-pairing was executed against sweep[0] of 6, and every prior round
(attack, defense, format audit, re-attack) read only that entry. The
qualitative claim survives untouched — all six tables sit at 1.0–1.6e-06,
failing and passing indistinguishable, ~4 orders below tolerance, and
6,429× is exactly as decisive a refutation as 7,060× — but a wrong margin
quote inside the incident whose lesson is "recompute from the raws" is
SERIOUS in this paper specifically. Fix is one line + one brief row
(change 1).

### 2.4 Rows W1/W2/W4 (pulled in by decision (b) below) — VERIFIED EXACT

`diagnosis_contender.json` (md5 `cdd3ea2e…`) and
`tap_localization_SUMMARY.json` (md5 `0e73ee28…`): ridge/SGD-cold/
SGD-warm/MLP all rf@0.9 = 0.0 exactly; trained-probe cos 0.17596 vs
1/√32 = 0.17678 ("0.176 against the 0.177 ceiling" ✓); membership cos
0.8956 ✓; ridge-vs-shuffled gap 0.16778 − 0.10928 = +0.0585 → "at most
+0.059" ✓; LM-head route episode top-1 0.99573 → "0.9957 (31.9× chance)"
✓; zeroing 0.9990 / 0.0286 / 0.9990 ✓; tap gaps +0.060/+0.006/+0.063/
+0.800 and tap-iv rf@0.9 = 0.67432 → Table 3 exact; ablation iv 0.0 ✓.

---

## 3. Decision (a): FIX-4 hard fallback (catalogue → appendix) — SOUND

The rebuttal's FIX-4 sanctioned exactly this fallback ("if … the style
pass still overflows after those cuts … A9's table reverts to the
appendix; A7 is never sacrificed"). The record shows the in-body move was
*attempted and measured* (~0.5pp over the 4pp working limit even after
the three named funding cuts — brief, "Per-section page budget"), the
three funding cuts themselves landed anyway (verified in the render: §1's
enumeration is gone, §6 is compressed to the two-sentence-per-lens form,
§8 absorbed the A7 citation block), and the reversal condition is
recorded twice (`sections/01_intro.tex` comment; `venue-requirements.md`
refresh note). A7's five probing-methodology citations are all present
and engaged in §8 — never sacrificed. The 4pp working limit is the
correct posture while the venue is a 404: it is the strictest bar across
the candidate set (MOSS backup is a LIVE-verified 4pp), and
appendix→body is a cheap recorded move while body→appendix under
deadline pressure is not.

**Do I think the catalogue belongs in the body badly enough to fight for
5pp venues?** It belongs in the body — contribution 1 *is* the catalogue,
and the body-only read currently has one figure and zero tables — but
not badly enough to drive venue selection. The body-only read is
coherent (§3–§6 carry all six incidents structurally; the forward
reference is honest), and workshop reviewers of 4-page papers read
appendices. Venue choice should be driven by audience fit
(measurement/eval/science-of-DL), with page limit as a tiebreak between
otherwise-equal slots. If the chosen CFP allows ≥5pp, execute the
recorded reversal exactly as drafted (change 3); do not pick a worse-fit
workshop to get the table into the body.

## 4. Decision (b): FIX-13 deviation ("λ=100 fixed", not "λ swept") — WRITER CORRECT

Checked against the raw, per the writer's obligation. The rebuttal's
sanctioned After-text asserted "(λ swept)"; the raw
`diagnosis_contender.json` shows `ridge_Tval.lambda = 100.0` and
`ridge_tied_embed_target.lambda = 100.0` — single fixed values, **no λ
sweep exists anywhere in the file**. Shipping the sanctioned wording
would have planted a new false claim while fixing A12's old one. The
shipped alternative — dropping the optimality claim and ruling out
over-regularization "by the rig's post-nonlinearity recovery below" — is
supported by the raw and is in fact stronger than the sanctioned
version: in `tap_localization_SUMMARY.json`, tap (iii) at **λ=1.0** reads
rf@0.9 = 0.0 while tap (iv) at the same λ=1.0 reads 0.674, so the
over-regularization story dies at both λ=100 (deployed-matched, taps
i/ii) and λ=1.0 (taps iii/iv) — a penalty 100× weaker still reads zero at
the state and 0.674 post-nonlinearity. The brief's W2 parenthetical
records the deviation accurately. This is the discipline the paper
preaches, applied to its own fix wave. No change required.

## 5. Detector decision: run a bounded 2-round discharge — do not skip

`papers/measurement-ws/detector/` is empty. The two available precedents
split: the EA-sprint skips were Jul-11-clock calls; rank-recruitment-ws
(07_final_review §5) ran the bounded discharge *because* it had ~7 weeks
of slack and a heavy-compression prose history. This tree matches
rank-recruitment on every justification: (i) ~7 weeks to the Aug 29 AoE
suggested deadline — the sprint-window timing rationale does not
transfer; (ii) the prose history (a six-source synthesis + 16-fix
rebuttal wave + FIX-4 funding-cut compression) is exactly the profile
that produces mechanical tells; (iii) double-blind review assumed; and
(iv) this specific paper — a measurement-methodology paper addressed to
reviewers who hunt instrument artifacts — is the worst possible place to
ship a detectable one. Consistency therefore selects the
rank-recruitment protocol, not the skip: two independent judges, pass =
both read the paper as ≥90%-human AND zero mechanical tells, maximum 2
rounds, a round-2 failure escalates to the PI rather than iterating;
artifacts land under `papers/measurement-ws/detector/` with a gauntlet
round row. It can run immediately, in parallel with the venue
confirmation; change 1 is a one-line numeric edit and does not reset it.

---

## 6. Numbered change list

1. **[SERIOUS — blocks submission; text + brief + rebuild]** §3
   (`sections/03_case_tolerance.tex`, the "refuted it with margins"
   parenthetical): change "residuals at most $1.4\times10^{-6}$, over
   $7{,}000\times$ below tolerance, identical across failing and passing
   cells" → "residuals at most $1.6\times10^{-6}$, over $6{,}000\times$
   below tolerance, indistinguishable across failing and passing cells"
   (over $6{,}400\times$ also acceptable: 0.01/1.5555e-06 = 6,429).
   Basis: §2.3 above — the raw's six audited tables, file-global max
   1.5555e-06 at the operative n_iter=20; the quoted 1.4164e-06/7,060×
   is sweep[0] (K78 s1840) alone, exceeded by K84 s1940 (1.4679e-06)
   and K90 s2040 (1.5555e-06). Update `brief.md` row I2 to the
   file-global statistics (max 1.5555e-06 → 6,429×; per-table
   mean-of-means 1.182e-06 → 8,460×) and record that M2's re-pairing had
   been executed against one of six sweep entries. Rebuild `main.pdf` +
   bundle; re-verify the §3 page-2 boundary (edit is width-neutral to
   within a few characters; no reflow expected). Scoped re-verify of the
   one edited sentence only — no full re-gauntlet warranted.
2. **[PROCESS GATE — blocks submission]** Detector: bounded 2-round
   discharge per §5. Record the round row + artifacts under
   `detector/` before the submission build is declared final.
3. **[PROCESS GATE — blocks submission; already documented]** Post-Jul-11
   venue confirmation per `venue-requirements.md` "Refresh obligation":
   enumerate the accepted-workshop list, pull the chosen CFP, re-verify
   every format assumption (page limit, kit, blind, archival, deadline),
   and apply the FIX-4 reversal decision against the confirmed limit
   (≥5pp main text → move `tab:catalogue` into the body per the recorded
   plan and re-run the affected render checks; 4pp → no change).
4. **[OPTIONAL — cosmetic]** Render-inspection M1/M2: Fig. 1 to
   ~0.45\linewidth or +1 font step; shorten Fig. 2(b) tick labels
   ("(iii) $S_1$ query" / "(iv) pre-LM-head"). Only if figures are
   touched for other reasons; neither blocks accept-ready.
5. **[OPTIONAL — placement]** Anchor the `tab:catalogue` float at the
   start of the appendix so Table 1 lands on page 6 rather than 7,
   closer to its §1 forward reference. Non-blocking; moot if change 3
   moves it into the body.

---

## 7. Scorecard

- Pages read: 8/8 (bundle PDF, rendered). Figures/tables inspected: all.
- Raw artifacts re-verified this session (md5 + values recomputed):
  `diag_ns_admission_result.json`, `crosscheck_lens_verdict_output.json`,
  `diagnosis_contender.json`, `oracles.json`,
  `tap_localization_SUMMARY.json` — 4/4 pointer md5s matched; every
  recomputed number matched the shipped prose **except** the §3 margin
  pair (change 1), which no prior round caught because all of them read
  sweep[0] of the six-table raw.
- Writer decisions adjudicated: FIX-4 fallback SOUND (reversal recorded);
  FIX-13 deviation CORRECT against the raw (sanctioned wording would
  have been false).
- Detector: bounded 2-round discharge selected over skip (precedent:
  rank-recruitment-ws; all four of its justifications transfer).

**Verdict: READY-AFTER-CHANGES** — change 1 is the only text edit;
changes 2–3 are gates; 4–5 optional. On their discharge this paper is
submission-ready pending only the venue's own CFP facts.
