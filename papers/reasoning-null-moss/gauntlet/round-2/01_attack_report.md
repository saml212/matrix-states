# Attack Report — Round 2 (targeted re-gauntlet) — "Three Bounds on a Null"

Scope of this round: ONLY the claim-shape-correction material (abstract,
intro, §2's new correspondence-null sentence, §3 in full, §7, the rewritten
Appendix A / Fig 2 caption / Appendix C compute sentence, and brief.md rows
C10/C11/C12 + the inline notes on C1–C4). Bounds 2/3 (behavioral contrast,
replication gate) were not re-litigated.

## Summary for the defense agent

The revision is fundamentally honest and the raws hold up. I recomputed every
md5 the brief cites for C10/C11/C12 — all six match exactly. I re-derived
every headline number of the re-verification directly from the archived raws
and every one reproduces: the pre-fix positive control (0/256 recovered,
transposed arm 1.0000; `verdict.overall_pass=false`), the post-fix re-run
(1.0000 recovered, role-swapped transposed arm 0.0000, `overall_pass=true`),
the closed-form adjudication √2 transpose signature, the 78/320 nonzero
re-metric (strongest cell 0.8691 at per_token×wikitext×s1×K32),
premise (iii)/(iv) 0/320 both, the label-shuffle 0/320 NULL-CLEARS, and the
derangement 0/320 with strongest cell 0.8691 real / 0.8125 deranged. Two of
the paper's most important logical load-bearing claims are **code-verified
true**: (1) premises (iii)/(iv) are computed from the raw k/v/q hook tensors
(`k_eff_items`, `v_eff_items`, `q_eff_self`), never from `squeeze_state_head`
/`S_T` — so the "premise gates fail regardless of the transpose defect"
fallback genuinely holds; and (2) the label-shuffle null is provably vacuous
at h=1 because `prev_slot = _iterate_permutation(succ, a_slot, hops-1)`
reduces to `a_slot` (zero iterations) at h=1, so the shuffled cycle is never
consulted. The correspondence-null logic and the "probe-invalid, not
refutation" argument both survive attack.

The weaknesses are all in **framing, scope precision, and one internal
numeric inconsistency**, not in the data. My highest-value findings target
the coordinator's stated top concern — scope precision (item 1). The single
most important is **A1**: the abstract collapses the corrected
"null-indistinguishable" verdict across all *three structurally different
deployments* when only one deployment (the marker-template grid, waves 1+2,
320 of 366 readings) was actually re-verified; the two un-reverified
deployments (natural-language wave 3 and task-familiarized wave 4, the
remaining 46 readings) are never flagged in the abstract. The body (§3
wave-3/4 paragraph, Appendix A's "corrected claim, binding throughout," and
the Fig 2 caption) IS scrupulously scope-honest — so every finding below is a
localized edit, not a structural retraction. There is **no CRITICAL finding**;
the null is not overturned and the corrected claim is defensible.

**Attacks targeting the item-1 scope-precision risk:** A1 (abstract) and A3
(the wave-4 dissociation result). A2 and A4 are adjacent framing/consistency
issues in the same neighborhood.

Counts: **0 CRITICAL / 4 SERIOUS / 3 MINOR.**

---

## Attacks

### A1: The abstract over-generalizes the corrected verdict to two deployments that were never re-verified

**Severity:** SERIOUS
**Type:** scope over-generalization (item-1 risk)

**Attack.** The abstract's Bound 1 reads: "a state-space composition readout,
**deployed three structurally different ways**, returned a recovered fraction
of exactly zero at 366 of 366 readings... after the fix, a 320-reading
sub-grid no longer reads zero, but two pre-registered correspondence nulls
reproduce the signal at every reading: **recovery is indistinguishable from a
broken correspondence, not absent.**" The framing device is "three
structurally different ways," and the delivered verdict — "recovery is
indistinguishable from a broken correspondence" — reads as the unified
conclusion for all three. It is not. Only ONE of the three deployments (the
zero-shot marker template = waves 1+2 = 320 readings) was re-run under the
fixed instrument and tested against the two correspondence nulls. The other
two deployments — zero-shot natural language (wave 3) and task-familiarized
(wave 4), together the remaining 46 of 366 readings — used the identical
pre-fix defective function and were **never** re-run or null-tested. The
abstract never says so. A reader finishes the abstract believing the
broken-correspondence verdict was established across all three deployments.

The paper's own Appendix A is explicit that it was not: "Waves 3 and 4... were
not independently re-run under the fix or tested against either correspondence
null... their own raw zero readings carry the same unconfirmed status the
wave-1/wave-2 zero readings carried before this re-verification." That
disclosure is airtight in the appendix and present in §3 — which is exactly
why its absence from the abstract is a scope-precision defect rather than a
factual error: the abstract states more than the body will support.

**Supporting evidence.** `sections/00_abstract.tex` lines 7–18 ("deployed
three structurally different ways... recovery is indistinguishable from a
broken correspondence, not absent"); contrast with `main.tex` Appendix A
lines 188–196 ("Waves 3 and 4... not independently re-run... reported here as
disclosed, not assumed corrected by extension"). 366−320 = 46 = wave 3 (16) +
wave 4 (30), the two un-reverified deployments.

**What would defuse it.** One clause in the abstract, e.g. "...no longer reads
zero on the re-verified marker-template grid (320 of 366 readings); the
natural-language and task-familiarized deployments keep their pre-fix status."
Or scope the concluding sentence: "recovery on the re-verified sub-grid is
indistinguishable from a broken correspondence."

---

### A2: Both abstract and intro lead Bound 1 with a superseded, now-defective number ("366 of 366 = exactly zero"); the intro states it in the present tense, where it is factually false

**Severity:** SERIOUS
**Type:** misleading emphasis / superseded headline figure

**Attack.** The marquee number of the entire instrument bound is "recovered
fraction of exactly zero at 366 of 366 readings," and it leads in three
places: the abstract ("returned a recovered fraction of exactly zero at 366 of
366"), the intro's Bound-1 bullet, and Table 1 (rows 1–2, "312/312 = 0.0",
"8/8 = 0.0"). But the paper's own re-metric shows this number is a
**defective-instrument artifact for 320 of the 366 readings** — 78 of those
320 are nonzero under the corrected instrument (up to 0.87). The intro states
it in the **present tense**: "A state-space composition readout... **returns**
a recovered fraction of exactly zero at 366 of 366 readings" (`01_introduction.tex`
line 16), then in the next clause says "after the fix, the largest sub-grid
(320 of the 366 readings) **no longer reads zero**." Under the corrected
instrument the readout does not "return" zero for 320/366, so the present-tense
headline is now false; only the past-tense, pre-fix reading was ever zero. For
a paper whose entire selling point is measurement honesty, leading the
headline bound with a number it has itself retracted for 87% of the readings —
and in the present tense — is the kind of emphasis a hostile reviewer will
quote back.

**Supporting evidence.** `01_introduction.tex` line 16 (present-tense
"returns... zero at 366 of 366"); `00_abstract.tex` lines 9–11; the
re-metric `AGGREGATE_SUMMARY.txt` line 98 ("GRAND TOTAL: 78/320 (cell,h)
readings nonzero"). The correction is present inline in all three locations,
which mitigates but does not remove the misleading-emphasis critique.

**What would defuse it.** Change the intro to past tense ("returned... zero...
under the pre-fix instrument") and lead the bound with the corrected claim
(null-indistinguishability) rather than the retracted zero, demoting "366/366
zero" to the pre-fix framing it now belongs to.

---

### A3: The wave-4 "dissociation" result rests on the defective-instrument flat-zero, which the corrected instrument would very likely not reproduce

**Severity:** SERIOUS
**Type:** result compromised by superseded instrument (item-1 risk)

**Attack.** Wave 4 is presented as a named result — Table 1 row 6 outcome
"dissociation," a dedicated figure (Fig 2), and a §3 paragraph: "It fails at 30
of 30 readings, 0 of 512 queries recovered each time, while the vocabulary-space
query loss falls 21.8 to 46.4 percent... the models measurably learn the task
through one readout while the other **stays at floor**... the dissociation
indicts the readout construct." But the "stays at floor" half — the geometric
readout reading exactly 0.0 at all 30 wave-4 readings — was measured with the
**defective pre-fix `squeeze_state_head`**. The 320-cell re-metric proved that
this same defective function returns spurious zeros that become nonzero (up to
0.87) once the transpose is fixed. So the flat-zero right panel of Fig 2 is,
by the paper's own logic, likely a defect artifact: a re-run under the fix
would probably show scattered nonzero (but still null-indistinguishable)
recovery, not a clean floor. The dissociation's *conceptual* point (task
learning without geometric-readout validity) survives via the premise gates,
but the specific "one readout learns, the other stays at floor" visual/rhetoric
— and Table 1's "dissociation" outcome label — are built on a reading the
paper has elsewhere shown cannot be trusted.

The Fig 2 caption and the §3 wave-3/4 sentence do disclose that the panel is
pre-fix and "has not been independently re-run" — which is why this is SERIOUS,
not CRITICAL. But the disclosure is scoped to *probe-invalid routing*, while
the dissociation *claim* (and the figure) specifically leans on the raw zero
the disclosure sets aside.

**Supporting evidence.** `03_geometric_null.tex` lines 60–75 (dissociation
paragraph, "stays at floor... indicts the readout construct"); `main.tex`
Fig 2 caption lines 278–290 (discloses the right panel is pre-fix and
un-re-run); re-metric off-arm h=1 readings in `AGGREGATE_SUMMARY.txt` (e.g.
`leg_a_off_openr1-mix-ext_s1_k20_native` h1=0.247, `...s2` h1=0.397) show the
control arm is not uniformly at floor once the instrument is fixed.

**What would defuse it.** Relabel Table 1's wave-4 outcome and soften the §3
claim to the survivable version ("premise gates fail at all 30 while $L_q$
falls — a validity/behavior dissociation on the premise gates, not on the
recovered-fraction floor, which is a pre-fix reading"), and add one clause to
Fig 2 noting the flat zero itself is a pre-fix artifact, not a validated floor.

---

### A4: Appendix A relabels the re-metric's Leg-A (60 cells) and Leg-B (20 cells) as "wave-1" and "wave-2", contradicting Table 1's wave-1 = 78 / wave-2 = 2

**Severity:** SERIOUS
**Type:** internal numeric inconsistency

**Attack.** Appendix A states: "78 readings are nonzero (**30 of 60 wave-1
cells, 8 of 20 wave-2 cells** carry ≥1 nonzero hop)." Those counts are the
re-metric's Leg-A (60 cells, 30 with a nonzero hop) and Leg-B (20 cells, 8
with a nonzero hop) — verbatim from `AGGREGATE_SUMMARY.txt` lines 67 and 94.
But Table 1 defines **wave 1 = "78 cells"** (row 1) and **wave 2 = "2 cells"**
(row 2, the 1.31B rung). So the paper says wave 1 is simultaneously 78 cells
(Table 1) and 60 cells (Appendix A), and wave 2 is simultaneously 2 cells and
20 cells. The re-metric re-partitioned the same 80 cells along a different axis
(Leg-A = the 14M intervention grid; Leg-B = the full scale ladder rungs 0–3),
which is *not* the wave-1/wave-2 (Phase-1 vs 1.31B-rung) split. The brief.md
C11 row gets this right ("30/60 Leg-A cells, 8/20 Leg-B cells"); the paper's
Appendix A mistranslates "Leg-A/Leg-B" into "wave-1/wave-2." A reviewer
cross-checking Table 1 against Appendix A hits a direct contradiction, and in a
paper whose whole pitch is "every number computed from archived raws," a
wave-1-is-both-78-and-60 inconsistency dents exactly the credibility the paper
trades on.

**Supporting evidence.** `main.tex` Appendix A lines 119–121 ("30 of 60 wave-1
cells, 8 of 20 wave-2 cells"); `03_geometric_null.tex` Table 1 lines 12–13
(wave 1 = 78 cells, wave 2 = 2 cells); `AGGREGATE_SUMMARY.txt` line 67 ("LEG A:
30/60 cells"), line 94 ("LEG B: 8/20 cells"). Reconciliation: original wave-1
(78) = intervention grid (60) + ladder rungs 0–2 (18); wave-2 (2) = rung 3;
re-metric Leg-B (20) = rungs 0–3 = 18+2. Same 80 cells, different partition —
so "same 80 cells" in Table 1 row 3 is correct, but the sub-labels are not.

**What would defuse it.** Replace "wave-1 cells"/"wave-2 cells" in Appendix A
with "intervention-grid (Leg-A) cells"/"scale-ladder (Leg-B) cells," or with a
neutral "60 of the 80 cells are the 14M grid, 20 are the scale ladder."

---

### A5: The mechanism explanation that makes the whole "trivial artifact" verdict quantitative rests on two numbers with no archived raw

**Severity:** MINOR (flagged explicitly per the coordinator's instruction)
**Type:** provenance gap / unverifiable load-bearing number

**Attack.** The paper's account of *why* both nulls reproduce the signal — the
substantive mechanism, not just the pass/fail — is: "the cross-checkpoint
prediction direction converges (mean pairwise cosine **0.9996**...) while the
candidate value population is itself near-collinear (mean pairwise cosine
**0.9648**)." Neither number exists in any archived raw. I grepped the entire
`2026-07-11_reasoning_link_validation/` and `_remetric/` trees: 0.9996 and
0.9648 appear only in `REASONING_LINK_DESIGN.md` §17.6a (lines 10936–10937,
11016–11019), an on-box single-checkpoint diagnostic whose source tensors were
not saved. The related "healthy-draw check... agrees with the true one on only
4.1% of entries" is likewise design-doc-only (§17 line 10916), not in any
validation JSON (the per-cell shuffle records carry no cycle-agreement field).

To the paper's credit, it discloses exactly this, in a full parenthetical
sentence: "(These two covariates are a design-registry record of an on-box
diagnostic on the strongest cell, not a separately archived raw artifact; no
raw state tensors were saved by the original harvest to recompute them from...)"
The disclosure is honest and not dressed up to look raw-backed — and the one
covariate in that sentence that *is* raw-backed (state condition numbers
1.8×10⁴–2.1×10⁶) genuinely matches `AGGREGATE_SUMMARY.txt` line 115
(min 18196.98, max 2105078.75). So this is MINOR. The residual critique: the
two numbers doing the *most* explanatory work in the trivial-artifact story are
precisely the two that cannot be verified, and the only disclosure of that is a
parenthetical late in an appendix. The empirical verdict (0/320 clear either
null) does not depend on them — but the *interpretation* ("collapsed prediction
directions against a near-collinear population") does.

**Supporting evidence.** `REASONING_LINK_DESIGN.md` §17.6a lines 10916,
10936–10937, 11016–11019; absence confirmed by grep across all
`2026-07-11_reasoning_link_validation/*` and `_remetric/*` JSON/txt;
`main.tex` Appendix A lines 160–171 (the disclosed provenance sentence).

**What would defuse it.** Nothing is strictly required (it is disclosed);
optionally, either recompute the two cosines from a fresh forward pass on the
strongest cell (cheap) and archive them, or move the disclosure clause out of
the parenthetical into the main sentence so it is not skimmable-past.

---

### A6: brief.md's C10 row describes the layout-adjudication with pre-fix numbers but cites the post-fix JSON, whose numbers are the exact transpose

**Severity:** MINOR
**Type:** evidence-row bookkeeping inconsistency (brief only)

**Attack.** brief.md C10 states the adjudication "confirms `squeeze_state_head`
returns fla's raw `[K,V]` layout instead of the `[V,K]` design layout
(**rel-Frobenius 0.0 vs the FLA-native closed form, 1.4142 = √2 vs the DESIGN
closed form**)" and cites `...reasoning_link_layout_adjudication_result_POSTFIX.json`
(md5 `1daa4c80...`, which I confirmed matches). But that POSTFIX file records
the **opposite**: `S_squeeze_vs_closed_form_design_rel_fro: 0.0` and
`S_squeeze_vs_closed_form_fla_rel_fro: 1.4142` — i.e. post-fix, squeeze matches
the DESIGN `[V,K]` layout (0.0) and differs from raw fla by √2. The numbers the
brief attributes to that file (0.0 vs fla, √2 vs design) are the **PRE-fix**
result, which lives only in the sibling `layout_adjudication_run_PREFIX.log`
(uncited, and not archived as a JSON). So a reviewer opening the cited md5 to
check "0.0 vs the FLA-native closed form" finds 1.4142 there instead.

This is a brief-level inconsistency, not a paper-level one: the PDF's Appendix A
narrates the *pre-fix* diagnosis correctly ("the readout output matched the
[K,V] closed form at 0.0... and the [V,K] closed form at exactly √2"), and that
prose is backed by the PREFIX log. But the paper's *only archived JSON* for the
adjudication is the post-fix file showing the transpose of what the prose
describes, and the brief points its "0.0 vs fla" claim at that post-fix file.

**Supporting evidence.** brief.md C10 row; `..._POSTFIX.json` lines 8–11 and
verdict lines 22–24 ("squeeze... returns DESIGN [V,K] (no defect): true");
`layout_adjudication_run_PREFIX.log` line 18 ("squeeze... returns RAW fla [K,V]
(the defect): True", `S_squeeze_vs_closed_form_fla_rel_fro: 0.0`).

**What would defuse it.** In C10, either cite the PREFIX log for the pre-fix
numbers, or restate the POSTFIX file's actual numbers (0.0 vs design, √2 vs
fla) as the post-fix confirmation that the fix landed. (Archiving the PREFIX
result as a JSON, not only a log, would also close the "the defect itself has
no JSON" gap.)

---

### A7: Table 1's re-metric row is labeled outcome "signal-bearing," which misleads a table-only reader

**Severity:** MINOR
**Type:** table-standalone clarity

**Attack.** Table 1 row 3 ("1+2 fixed-lens re-metric | same 80 cells | 78/320
nonzero, h=1≤0.87 | **signal-bearing**"). A reader who reads only the table —
the coordinator's item-4 test — sees "signal-bearing" as the registered
outcome and can reasonably infer a real signal was recovered post-fix. The
adjudication that it is a trivial artifact is in the very next row ("0/320 clear
either of 2 nulls | null-indistinguishable"), so the two rows must be read
together; in isolation, "signal-bearing" overstates. The term is defensible as
"the lens now emits nonzero values" but carries a positive connotation the
data do not support.

**Supporting evidence.** `03_geometric_null.tex` Table 1 lines 14–15.

**What would defuse it.** Rename the row-3 outcome to something neutral, e.g.
"nonzero (unadjudicated)" or "raw signal," reserving the verdict for row 4.

---

## Attacks I considered but decided were weak (i.e., the revision holds)

- **"Premises (iii)/(iv) are contaminated by the same transpose defect, so the
  probe-invalid fallback is circular."** FALSE — code-verified. In
  `reasoning_link_probe.py`, `premise_iii_iv` (lines 936–956) and
  `premise_i_ii` (926–933) consume only `k_eff_items`, `v_eff_items`, and
  `q_eff_self` — the raw key/value/query hook tensors (gathered at lines
  1437–1438 and 1482). `squeeze_state_head`/`S_T` (line 1439) feeds ONLY
  `apply_state_power` (line 1513, the recovery scorer) and
  `state_condition_number` (line 1521, provably transpose-invariant). The
  paper's central defense — "the alignment-premise gates... never [read] the
  state-extraction output the defect touched" — is exactly true. The
  probe-invalid-not-refutation argument stands.

- **"The label-shuffle-vacuous-at-h=1 claim is a hand-wave."** FALSE —
  code-verified. `compute_prev_slot_and_target` (line 902) sets
  `prev_slot = _iterate_permutation(succ, a_slot, hops-1)`; at h=1 that is
  zero iterations, so `prev_slot = a_slot` and the target is
  `v_eff_items[a_slot]` regardless of whether `succ` is the true or the
  shuffled cycle. Real == null at h=1 by construction, exactly as claimed —
  and the derangement null, which deranges the value slots directly, does have
  teeth at h=1 (strongest cell real 0.8691 vs deranged 0.8125, both raw-present
  in `DERANGE_SUMMARY.txt` line 69). The two-null design is correctly motivated.

- **"The md5s or headline numbers were transcribed wrong."** FALSE. All six
  brief C10/C11/C12 md5s recompute exactly. Every re-verification number I
  spot-checked reproduces from the raws: 0/256 pre-fix and 1.0000 post-fix
  positive controls with the role-swap; 78/320 nonzero (69/240 Leg-A + 9/80
  Leg-B); premise 0/320 both; label-shuffle 0/320 (41 floor-clearing readings,
  mean real 0.3023 vs null 0.3010); derangement 0/320 (mean real 0.3023 vs
  deranged 0.2960); per-arm concentration 20/72 per_token, 1/36 off, 0/72
  global; resample stability 20/21. The "≥0.9 recovery, continuous not argmax"
  and null_clears three-condition bar (`null < 0.5·real` AND null-relative AND
  ≥0.10 floor) match the code (`reasoning_link_validation.py` lines 149–162).

- **"The premise-gate failure is itself just another instrument artifact."**
  Considered — but the premise gates test a different quantity (same-entity
  q–k and k–v cosine vs a shuffled-entity null) on the raw hooks, and they fail
  0/320 independently of the transpose. Round 1 already treated the premise
  failure as the load-bearing routing evidence; nothing in the re-verification
  weakens it, and the revision correctly leans on it. Not a new attack.

- **Appendix C's "≈0.52 GPU-hours" re-verification figure.** Not in any single
  summary; derivable only by summing per-cell log wall-times. Left un-flagged
  as a finding: it is an explicitly approximate compute disclosure ("≈"), the
  logs exist, and nothing rides on it.

## New citations

None required by the changed material. (Round 1's A5 additions — Grazzi et al.
2025 and Merrill et al. 2024 — are now present in §6, and Grazzi is correctly
invoked in §7 as a competing expressivity account; no further citation gap in
the re-verification content.)

---

*Security note:* No embedded `<system-reminder>` blocks, fake tool-output
injections, date-change claims, or concealment instructions were encountered
in any file read during this review. All verification was against git-tracked
raws and recomputed md5s.
