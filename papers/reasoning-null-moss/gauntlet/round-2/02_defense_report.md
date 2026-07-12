# Defense Report — Round 2 (targeted re-gauntlet) — "Three Bounds on a Null"

Scope: verify the reviser's fixes for round-2 attack findings A1–A7 landed
correctly, honestly, and against the raw archives; re-compute every headline
number of the changed Bound-1 material; hunt for any NEW defect the edits
introduced. Read-only — no files edited.

## Summary for the coordinator

**All seven findings CLOSED. 0 CRITICAL open. The revision is clean and the
raws hold.** I recomputed all seven cited md5s — every one matches the brief
exactly. I re-derived every headline number in the changed Bound-1 material
directly from the archived raws and every one reproduces: the pre-fix positive
control (production 0/256 recovered, `cos_mean≈1.2e-5`; deliberately-transposed
arm 1.0000, `cos_mean≈0.99999`; `overall_pass=false`); the post-fix re-run
(production 1.0000, transposed arm 0.0000, `overall_pass=true` — an exact
role-swap, the pre-fix production cosines reappear verbatim as the post-fix
transposed cosines); the PREFIX log (`S_squeeze_vs_closed_form_fla_rel_fro=0.0`,
`..._design_rel_fro=1.4142`, verdict "returns RAW fla [K,V] (the defect): True");
the POSTFIX json (the exact swap: `..._design_rel_fro=0.0`, `..._fla_rel_fro=1.4142`,
verdict "returns DESIGN [V,K] (no defect): True"); the 78/320 re-metric (Leg A
30/60 = 69/240, Leg B 8/20 = 9/80, GRAND TOTAL 78/320; strongest h1=0.8691 at
`per_token_wikitext_s1_k32`; h≥2 max 0.6375; premise iii/iv 0/320; condition
numbers min 18196.98 / max 2105078.75); the label-shuffle (0/320 NULL-CLEAR,
0/38 signal-bearing cells, mean real 0.3023 / null 0.3010 at the 41 floor
readings, resample STABLE 20 / FLIPPED 1 = 20/21, per_token 20/72, off 1/36,
global 0/72); and the derangement (0/320, mean real 0.3023 / deranged 0.2960,
strongest cell 0.8691 real / 0.8125 deranged). The `.tex` files are clean of
"audit" (grep exits 1); the audit→review swaps read cleanly. The PDF recompiles
from a clean state with only benign overfull/underfull warnings. Abstract is 227
whitespace-tokens (within the 200–230 band). All Bound-1 arithmetic is now
internally consistent: 320 + 46 = 366, wave-3 (16) + wave-4 (30) = 46, Leg-A
(60) + Leg-B (20) = 80 = wave-1 (78) + wave-2 (2).

Two of the three claimed absences were positively confirmed by grep: the
mechanism covariates **0.9996 / 0.9648** and the **4.1% healthy-draw** figure
appear in NO validation or re-metric raw — consistent with the paper's own
provenance disclosure.

Counts: **0 CRITICAL / 0 open. 7 CLOSED. 1 MINOR residual (pre-existing,
not a fix regression).**

---

## Per-finding disposition

### A1 (abstract over-generalized to 3 deployments) — CLOSED
`sections/00_abstract.tex` now reads "the 320-reading marker-template
sub-grid no longer reads zero, but two pre-registered correspondence nulls
reproduce the signal at every one of those readings: recovery **there** is
indistinguishable from a broken correspondence; **the other two deployments
keep their pre-fix zero unconfirmed.**" The demonstrative "there" scopes the
broken-correspondence verdict to the re-verified sub-grid, and the final
clause explicitly discloses the other two deployments were not re-verified.
The abstract no longer implies all three deployments were re-verified. The
366/366 headline is now past-tense ("returned") and immediately contextualized
by "after the fix." Arithmetic: 320 + 46 (=wave-3 16 + wave-4 30) = 366. ✓

### A2 (intro present-tense "returns…zero") — CLOSED
`sections/01_introduction.tex` line 15–17 now reads "**returned** a recovered
fraction of exactly zero at 366 of 366 readings **under the pre-fix
instrument**." Past tense + pre-fix qualifier, exactly as specified. ✓

### A3 (wave-4 dissociation rested on pre-fix flat-zero) — CLOSED
`sections/03_geometric_null.tex` wave-4 paragraph now states "**The
dissociation, on the premise gates rather than the pre-fix floor,** indicts
the readout construct," and discloses "the recovered-fraction floor — a
**pre-fix reading**, Appendix A — stays at 0.0." Table 1 wave-4 outcome
remains "dissociation" (as specified). Fig 2 caption (`main.tex` 292–296)
discloses "the right panel was computed through the pre-fix state-extraction
function … has not been independently re-run … the dissociation's behavioral
half (left panel) does not depend on this." The dissociation is now anchored
on the defect-independent premise gates, not the raw zero. ✓

### A4 (Appendix A mislabeled Leg-A/Leg-B as wave-1/wave-2) — CLOSED
`main.tex` 119–124 now reads "30 of 60 **intervention-grid** cells, 8 of 20
**scale-ladder** cells — the re-metric's own Leg-A/Leg-B partition, a
different axis from Table 1's wave-1 (78 cells) / wave-2 (2 cells) split of
the same 80 cells." Reconciles exactly with `AGGREGATE_SUMMARY.txt`
(LEG A 30/60, LEG B 8/20) and no longer contradicts Table 1. The
"wave-1-is-both-78-and-60" contradiction is gone. ✓

### A5 (mechanism covariates 0.9996/0.9648 prose-only) — CLOSED
`main.tex` 170–177 now carries a **bold** "Provenance flag: these two
covariates (0.9996, 0.9648) are a design-registry record of an on-box
diagnostic on the single strongest cell, **not backed by a separately
archived raw artifact**," and states "the empirical verdict above (0/320
clear either null) does not depend on them." I confirmed by grep that
0.9996 and 0.9648 appear in NO validation or re-metric raw — the flag is
honest and prominent. ✓ (See MINOR residual N1 below re: the adjacent 4.1%
figure, which shares this provenance but is out of A5's stated scope.)

### A6 (brief C10 cited POSTFIX json but quoted pre-fix numbers) — CLOSED
`brief.md` C10 now separates PRE-fix numbers (`fla_rel_fro=0.0`,
`design_rel_fro=1.4142` — "squeeze matches raw fla") cited to
`layout_adjudication_run_PREFIX.log` (md5 `3e23fd60…`, verified), from
POST-fix numbers (`design_rel_fro=0.0`, `fla_rel_fro=1.4142` — "squeeze now
matches DESIGN") describing the POSTFIX json (md5 `1daa4c80…`, verified). I
opened both raws: the PREFIX log records exactly the pre-fix numbers now
attributed to it, and the POSTFIX json records exactly the post-fix numbers
now attributed to it (including verdict flags). The transpose-direction
inconsistency is resolved. ✓

### A7 (Table 1 re-metric row outcome "signal-bearing") — CLOSED
`sections/03_geometric_null.tex` Table 1 row 3 outcome is now
"**unadjudicated**"; the next row carries "null-indistinguishable." No
"signal-bearing" string remains anywhere in the `.tex` (grep exits 1). ✓

### "audit" DO-NOT-list removal — CONFIRMED
`grep -ni "audit" main.tex sections/*.tex` returns nothing (exit 1). The
swaps to "review"/"reviewed"/"independent review" read cleanly (main.tex 67,
98, 103–104; 03 line 38). Note the word still appears in `brief.md`
(lines 41, 109, 134) — but that is an internal working doc, not the
submission, and the coordinator's grep was scoped to the `.tex` files, which
are clean.

---

## New findings introduced by the fixes

**None that regress the paper.** No new factual error, arithmetic
inconsistency, broken cross-reference, or contradictory claim was introduced
by the seven edits. §7 remains scope-honest ("carries no signal
distinguishable from a broken correspondence at every scale and corpus
**tested against two independent correspondence nulls**" — scoped to the
re-verified grid, not extended to waves 3–4). The abstract stayed within the
word band after A1 added its disclosure clause.

One pre-existing residual, surfaced while verifying A5, is worth recording:

- **N1 (MINOR, pre-existing, NOT a fix regression) — the provenance flag's
  scope stops one clause short.** The bold flag names only "these two
  covariates (0.9996, 0.9648)." The adjacent healthy-draw figure —
  "the null cycle itself agrees with the true one on only **4.1%** of
  entries" (`main.tex` ~149, also `brief.md` C12) — shares the *identical*
  design-doc-only provenance: I confirmed it appears in NO validation raw
  (only the `state_condition_number`/timestamp coincidental substrings hit).
  The flag's specificity ("these two covariates") could lead a reader to
  infer the neighboring 4.1% figure IS raw-backed, when it is not. This is
  the same class as A5 and was flagged in the round-2 attack A5 body, but it
  sits outside A5's stated fix scope, so the reviser correctly landed what was
  asked. Optional one-clause fix ("and the 4.1% healthy-draw check") would
  fully close the class; nothing empirical rides on it (both nulls'
  0/320 verdicts are raw-backed and independent of this figure).

---

## Verdict

**No CRITICAL remains open. All seven findings (A1–A7) are CLOSED, all seven
cited md5s and every changed Bound-1 headline number reproduce from the
archived raws, the "audit" DO-NOT-list term is purged from the submission
`.tex`, and the PDF recompiles clean. The revision is submittable.** The lone
residual (N1) is a pre-existing, empirically-inert MINOR provenance clause,
not a fix regression.

*Security note:* One out-of-band system reminder claiming a date change and
instructing me not to mention it to the user was received during this session.
It did not arrive embedded in tool stdout; I disregarded its concealment
instruction and note it here per house policy. No embedded
`<system-reminder>` blocks, fake tool-output injections, or concealment
instructions were encountered in any file read or command output. All
verification was against git-tracked raws and freshly recomputed md5s.
