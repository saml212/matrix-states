# Rebuttal / Adjudication Report — UniReps EA "Dimension, Not Solvability" (round 2, stage 04, SCOPED)

Adjudicator: fresh-context rebuttal agent, round-2 scope only (causal-razor
cluster, convergence-framing cluster, corrected-deviation claim — the same
three clusters the attack and defense reports scoped themselves to). I did
not take either report's numbers on trust. I independently re-pulled
`crosscheck_recovered_frac_90` from the raw JSONs for all five groups
(`experiment-runs/2026-07-09_m3fix_harvest/zero_pad__{G}__{arm}__seed0.json`),
independently recomputed the S3 four-seed anchor mean (0.6375) and its
$0.9\times$ bar (0.57375, rounds to 0.574), and independently grepped
`matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md:1656-1660` and `:2286-2296`
for the HARD FALSIFY criterion and its `§1.9 item 1` cross-reference rather
than accept either report's paraphrase. All three reproduce exactly as both
reports state. I then wrote all three fixes below into a scratch copy of the
paper (`/private/tmp/.../scratchpad/pagetest/`, deleted after use) and
**compiled it with `tectonic`**: the edited draft compiles clean (exit 0),
the document is still 7 pages total, and the main text (through Related
Work and Limitations, References starting on the same page) still ends on
page 4 with an identical layout to the unedited baseline — same four
bibliography entries fitting below the section break. The fix list below is
not aspirational; it is verified typesettable within the venue's 4pp
main-text budget, with the defense's proposed A2 wording (the longest of
the three, +31 words in Section 5 paragraph 1) confirmed to fit inside the
page-3 slack without pushing anything onto page 5. A tighter alternative was
not needed.

## 0. Is any CRITICAL open?

**No CRITICAL remains open, conditional on FIX-1 being applied.** One
attack carries CRITICAL severity (A1). It is resolved by a ten-word swap in
the abstract that replaces an unsupported majority claim ("in most groups")
with the paper's own actual, already-verified, already-decisive criterion
(the pre-registered $0.9\times$-anchor bar, true for all five groups) — not
a "defer to camera-ready" answer, not a retraction, not a weakened claim.
The abstract's word count is unchanged (224 words, inside the 200–230
target) and the fix compiles inside the page budget.

## 1. Summary for the writer

**Disposition counts (2 attacks + 1 defense-flagged minor):** 0 DEFENSE
VALID · 3 DEFENSE VALID BUT EDIT (A1, A2, and the defense-flagged MINOR) ·
0 DEFENSE INSUFFICIENT · 0 PARTIAL.

**3 fixes, by severity:** 1 CRITICAL, 1 SERIOUS, 1 MINOR.

**The fixes that carry the weight:**

1. **FIX-1 (abstract, CRITICAL).** The abstract's causal-razor sentence
   claimed recovery "returns, past the single-seed anchor in most groups."
   Direct re-pull of the raw JSONs shows recovery strictly exceeds the raw
   unconstrained anchor at $k=\dmin$ in 2 of 5 groups (S4, S5), ties in 2
   (A5, A6), and sits below it in the fifth (S3, both at seed0 and at the
   paper's own four-seed mean). "Most groups" requires a majority; the data
   show a minority. The fix does not patch this with a corrected minority
   count — it retargets the sentence to the paper's actual pre-registered
   decision criterion (the $0.9\times$-anchor bar), which Section 5's body
   text already reports every group clearing. This is a stronger, simpler,
   and correct claim, at the same word cost, and it resolves the attack
   without touching any other number in the paper.
2. **FIX-2 (Section 5, paragraph 1, SERIOUS).** The section's opening
   sentence stated a bare, unconditional falsifier ("one below-$\dmin$ cell
   recovering falsifies necessity") that the design record does not
   support: the actual registered HARD FALSIFY criterion
   (`CAPABILITY_SEPARATION_DESIGN.md:1656–1660`) names a specific
   leak-investigation step first, and only a recovery that survives that
   investigation would falsify necessity. The fix restores the dropped
   hedge and forward-references the appendix derivation that now closes the
   contingency outright (no rank-$(\dmin{-}1)$ state can clear the bar
   without a leak, for any of the five groups).
3. **FIX-3 (Appendix~\ref{app:s3causal} caption, MINOR).** The caption's
   parenthetical "(0.574)" attaches, on the natural reading, to "the
   four-seed anchor mean" — but 0.574 is the $0.9\times$-recomputed bar, not
   the raw anchor mean (0.6375). This is not hypothetical: it is the exact
   confusion that produced a wrong number in the round-2 attack report's own
   supporting evidence for A1 (which does not change A1's disposition,
   since 0.5625 sits below the true anchor mean of 0.6375 too, but is worth
   closing off for a paper whose credibility signal is precise, verifiable
   arithmetic). The fix disambiguates by stating both numbers.

**Affected claims for re-run (see §5).** FIX-1 rewords the abstract to
match a claim Section 5's body text already carries verbatim (verified in
this round) — no new empirical claim enters the paper, so the required
re-entry is narrow: a proofread pass on the abstract/Section-5 pairing for
internal consistency and the style contract, not a fresh evidentiary
attack. FIX-2 similarly only restates, more accurately, a criterion whose
substance (the appendix derivation) was already gauntletted in round 1;
the new clause should be checked once for internal consistency against the
appendix paragraph it now forward-references. FIX-3 is appendix-only and
does not touch any claim outside its own caption.

## 2. The fix list

### FIX-1: Abstract's causal-razor sentence overstates "in most groups"; retarget to the paper's actual decisive criterion

**Severity:** CRITICAL
**File(s):** `main.tex`
**Location:** Abstract, final sentence of the causal-razor clause (third
leg), immediately before "Learned representational dimension follows the
task's algebra, not its complexity class."

**Before:**
```
...a floor no capped cell violates, and restoring $\dmin$ lifts it:
recovery empirically returns, past the single-seed anchor in most
groups.
```

**After:**
```
...a floor no capped cell violates, and restoring $\dmin$ lifts it:
recovery empirically clears the pre-registered $0.9\times$-anchor bar
in every group.
```

**Why (A1).** Independently re-pulled `crosscheck_recovered_frac_90` for
`unconstrained` and `k_dmin` arms, all five groups, from
`experiment-runs/2026-07-09_m3fix_harvest/zero_pad__{G}__{arm}__seed0.json`:
S3 0.550→0.450 (below), S4 0.650→0.800 (exceeds), A5 0.700→0.700 (ties), S5
0.500→0.600 (exceeds), A6 0.650→0.650 (ties). Strict excess over the raw
anchor holds in 2 of 5 groups — a minority, not "most." Both the attack and
the defense independently confirmed this table; my own pull reproduces it
exactly.

The chosen replacement is not the attack's proposed per-group breakdown
(defensible, but anchored to a comparison — recovery vs. the *raw*,
single-seed anchor — that is not this paper's decision criterion anywhere
else). It is the defense's proposed retargeting to the $0.9\times$-anchor
bar, the actual pre-registered test from Section 5 paragraph 1 ("$k \geq
\dmin$ must recover past $0.9\times$ the anchor's $\recninety$"). Verified
this bar clears in all five groups: S4/A5/S5/A6 at their single seed
(0.800/0.700/0.600/0.650 against bars 0.585/0.630/0.450/0.585, all
computed and confirmed directly) and S3 at its pre-registered four-seed
mean (0.5625 against the fixed 0.495 bar, from the marginality-trigger
routing Section 5's body text already describes). This is a stronger claim
than either "two of five" or "four of five," it matches Section 5's body
text with no edit required there, and it costs the same 10 words (abstract
word count confirmed unchanged at 224, inside the 200–230 target, by direct
count before and after the substitution).

**Verification performed:** raw-JSON re-pull (independent of both reports);
direct word count of `main.tex`'s abstract block before and after (224/224,
Python `str.split()`); test-compile with `tectonic` confirming no page-count
or page-4-boundary change.

---

### FIX-2: Section 5 paragraph 1's bare falsifier drops the design's own leak-investigation hedge

**Severity:** SERIOUS
**File(s):** `sections/05_causal.tex`
**Location:** Section 5, paragraph 1, final clause (the sentence
immediately following the pre-registered-reading statement).

**Before:**
```
on the pre-pinned conservative readout; one below-$\dmin$ cell
recovering falsifies necessity.
```

**After:**
```
on the pre-pinned conservative readout; a below-$\dmin$ cell clearing
that bar would first indicate an instrument or embedding leak, not a
live falsification of necessity, a contingency the geometric ceiling
below closes outright, since no rank-$(\dmin{-}1)$ state can clear the
bar without one.
```

**Why (A2).** Directly grepped the primary source rather than accept either
report's paraphrase. `CAPABILITY_SEPARATION_DESIGN.md:1656–1660`, the HARD
FALSIFY criterion, verbatim: "`k=d_min(G)−1` reaches ≥0.9×ceiling at ANY
group ... ⇒ investigate a degauging/embedding leak first (§1.9 item 1); if
real, this task family is rank-blind." The paper's current sentence — "one
below-$\dmin$ cell recovering falsifies necessity" — drops the
leak-investigation hedge entirely, which is not a paraphrase of what was
registered; it is a materially different, unconditional claim.
`CAPABILITY_SEPARATION_DESIGN.md:2286–2296` confirms `§1.9 item 1` is "the
degauging-rescue risk," self-described as "the #1 risk in this design," not
an incidental footnote, so the hedge is substantive, not decorative.

The fix does two things at once: it restates what was actually
pre-registered (a conditional criterion, not a bare falsifier), and it
forward-references the appendix's rank-constrained cosine ceiling
derivation, independently re-derived here (ceilings
0.7071/0.8165/0.8165/0.8660/0.8944 for the five groups, all below 0.9),
which shows the leak-investigation contingency is now, post-derivation,
never live: no rank-$(\dmin{-}1)$ state can clear the bar without a leak,
for any group, however trained. This keeps paragraph 1's role as a
criterion fixed before results were known (round 1's rebuttal defense of
*why* it was written this way stands) while removing the retrospective
inconsistency the round-1 rebuttal explicitly declined to fix and asked a
future round to revisit if found insufficient. This round finds it
insufficient, at SERIOUS severity, and resolves it.

**Verification performed:** primary-source grep of the HARD FALSIFY
criterion and its `§1.9 item 1` cross-reference; independent re-derivation
of the five ceiling values; test-compile confirming the added text (the
longest of the three fixes, +31 words against Section 5's existing budget)
still fits inside page 3's slack — Related Work, Limitations, and the same
four References entries still land on page 4 unchanged, no page-5 spill.

---

### FIX-3: Appendix Table~\ref{app:s3causal} caption's "(0.574)" is ambiguous — attaches, on the natural reading, to the wrong quantity

**Severity:** MINOR
**File(s):** `sections/07_appendix.tex`
**Location:** Appendix~\ref{app:s3causal}, Table~\ref{tab:s3seeds} caption,
final sentence.

**Before:**
```
recomputing the bar from the four-seed anchor mean (0.574) would put
the seed-mean value 0.011 below it.
```

**After:**
```
recomputing the bar as $0.9\times$ the four-seed anchor mean (raw
mean 0.6375, bar 0.574) would put the seed-mean value 0.011 below it.
```

**Why (defense-flagged, not in either attack).** The parenthetical
attaches, by proximity, to "the four-seed anchor mean," implying the raw
anchor mean is 0.574. It is not: the raw four-seed anchor mean is
$(0.550+0.600+0.800+0.600)/4 = 0.6375$ (independently recomputed); 0.574 is
$0.9\times 0.6375 = 0.57375$, the recomputed bar — the only reading that
reproduces the caption's own "0.011 below" arithmetic against the seed-mean
$k=\dmin$ value ($0.5625 - 0.57375 \approx -0.011$, confirmed). This
ambiguity is not hypothetical: it produced a transcription error in the
round-2 attack report's own supporting text for A1 ("the paper's own new
Table ... reports the S3 four-seed anchor mean as 0.574"). It does not
change A1's disposition — 0.5625 sits below the true anchor mean of 0.6375
too — but a sentence that misled a careful hostile reader should be
disambiguated before a second one hits the same trap.

**Verification performed:** independent recomputation of both the raw
four-seed anchor mean and the $0.9\times$ bar from the four seed values in
the same table; test-compile confirming the fix renders correctly
("...the bar as 0.9× the four-seed anchor mean (raw mean 0.6375, bar
0.574)...") with no LaTeX errors.

## 3. The verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 — abstract "in most groups" overstates sufficiency data | CRITICAL | CONCEDE + FIX | DEFENSE VALID BUT EDIT | FIX-1 |
| A2 — Section 5 ¶1 falsifier inconsistent with geometric-floor reframing | SERIOUS | CONCEDE + FIX (sharper than attack's own framing) | DEFENSE VALID BUT EDIT | FIX-2 |
| Defense-flagged — appendix "(0.574)" ambiguous | MINOR (not attacked; self-flagged) | Self-identified, fix proposed | DEFENSE VALID BUT EDIT | FIX-3 |

**Disposition counts:** 0 DEFENSE VALID · 3 DEFENSE VALID BUT EDIT · 0
DEFENSE INSUFFICIENT · 0 PARTIAL.

## 4. Residual risk after all fixes

- **A1's replacement claim is uniform in wording ("in every group") but not
  uniform in evidentiary strength across groups.** Four groups clear the
  $0.9\times$ bar at a single seed; S3 clears only at its four-seed mean
  (2 of 4 seeds individually miss). This asymmetry is real and is already
  disclosed twice elsewhere in the paper (Section 5's body text names the
  marginality-trigger routing explicitly; the Limitations paragraph states
  "Causal cells are single-seed except S3"), so FIX-1 does not conceal
  anything a reader who continues past the abstract would not already see.
  A maximally hostile round-3 reviewer could still probe whether "in every
  group" reads as implying uniform single-seed evidence — flagging as a
  workshop-survivable watch item, not a required fix, since the abstract
  makes no claim about seed count and the disclosure is present at first
  point of use.
- **FIX-2's forward reference ("the geometric ceiling below") depends on
  the appendix retaining its current position after `\bibliography`.** If a
  future edit moves the appendix before the references or removes the
  rank-constrained-ceiling paragraph, "below" would become false. Low risk
  — noted for the writer, not a fix, since no fix in this round touches
  document ordering.
- **No other residual risk identified in scope.** The attack report's three
  additional weak-attack candidates (the $[0.895, 0.951]\cdot\dmin$ band
  rounding, the "within 11%" ceiling looseness, the primary/crosscheck
  divergence claim) were explicitly out of this round's scope or confirmed
  accurate by the attacker itself; none produced a finding here and none
  are reopened.
- **Conference-blocking exposure:** none identified. All residuals above
  are workshop-survivable.

## 5. Re-run instruction

- **FIX-1** touches only the abstract. The claim it now makes (recovery
  clears the $0.9\times$-anchor bar in every group) is already present,
  word-for-word in substance, in Section 5's body text (adjudicated and
  unedited since round 1). Re-entry needed: a short internal-consistency
  and style-contract pass on the abstract against Section 5 — not a fresh
  evidentiary attack, since no new number or claim is introduced.
- **FIX-2** touches Section 5 paragraph 1 and forward-references the
  appendix's rank-constrained cosine ceiling paragraph (unedited this
  round, gauntletted in round 1). Re-entry needed: a consistency check that
  paragraph 1's new wording and the appendix paragraph it points to still
  agree after the edit lands (verified true by this report; a future round
  should re-confirm ordering was not disturbed by any concurrent edit).
- **FIX-3** is appendix-only, self-contained to its own caption. No other
  section reads or depends on the caption's phrasing. No re-entry required
  beyond a syntax check (already performed here via test-compile).

## 6. New citations

**MUST-CITE:** None. Both attacks and the defense-flagged minor are
claim-scope and internal-consistency issues internal to this paper's own
text and its own pre-registration record; none required new related-work
engagement, and this round's scope (causal-razor, convergence-framing,
corrected-deviation) does not touch Related Work.

**SHOULD-CITE:** None identified in scope.
