# Defense Report — UniReps EA "Dimension, Not Solvability" (round 2, stage 02, SCOPED)

Author's-side review, fresh context, round-2 scope only (causal-razor cluster,
convergence-framing cluster, corrected-deviation claim — the same three
clusters the attack report scoped itself to). I independently re-pulled every
raw JSON field the attack cites (not trusting either the attack report's or
the draft's transcription) and independently re-read the pre-registered
falsifiability criterion in `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md`
(the HARD FALSIFY line, its `§1.9 item 1` cross-reference, and the M3 CONFIRM
definition it belongs to) rather than take the attack's characterization of
it on trust.

## 1. Summary for the rebuttal agent

**0 DEFEND, 0 PARTIAL, 2 CONCEDE + FIX.** Both attacks' factual premises hold
up exactly as stated, and in A2's case the design record supplies *stronger*
grounds for the fix than the attack itself cited. Neither fix requires new
data or a new experiment; both are single-clause rewrites. The paper is
submittable after fixes.

- **A1 (CRITICAL, CONCEDE+FIX).** Confirmed by direct re-pull of
  `crosscheck_recovered_frac_90` from the raw JSONs: only 2/5 groups (S4, S5)
  strictly exceed the raw unconstrained anchor at $k=\dmin$; 2/5 (A5, A6) tie;
  1/5 (S3) sits below. "Most groups" is false under the natural reading. But
  I do **not** recommend the attack's own proposed replacement (the
  group-by-group breakdown) for the abstract — it anchors the claim to a
  comparison (`k=\dmin` vs. the *raw*, single-seed anchor) that is not
  actually this paper's pre-registered decision criterion anywhere else in
  the text. The paper's real, already-verified, already-decisive bar is the
  **pre-registered $0.9\times$-anchor bar**, which Section 5's own body text
  already reports every group clearing (S4/A5/S5/A6 at their single seed,
  S3 at its four-seed mean under the pre-registered marginality-trigger
  routing). I recommend anchoring the abstract to that bar instead: it is
  true for *all five* groups (stronger and simpler than "two of five" or
  "four of five"), it is the paper's own actual test, and — checked — it
  costs the abstract exactly the same word count as the current, defective
  sentence.
- **A2 (SERIOUS, CONCEDE+FIX).** Confirmed against the primary source: the
  registered HARD FALSIFY criterion in `CAPABILITY_SEPARATION_DESIGN.md`
  (lines 1656–1660) does **not** say "one below-$\dmin$ cell recovering
  falsifies necessity" — it says such an event triggers investigating a
  degauging/embedding leak *first* (`§1.9 item 1`, the design's own
  self-attacked #1 risk), and *only if that recovery survives the leak
  investigation* does it mean the task family is rank-blind. Section 5
  paragraph 1's current sentence drops this hedge entirely, which is not
  merely "in tension" with paragraph 2 and the appendix, as the attack
  frames it — it actually misstates what was pre-registered. The fix I
  recommend goes one step further than the attack's suggested text: state
  the real registered contingency, and note that the appendix's own
  trace-inequality derivation (added in round 1) now closes that
  contingency outright, since no rank-$(\dmin{-}1)$ state can clear the bar
  without a leak.

Both fixes are textual only. No experiment is required for either.

## 2. Defenses

### A1: Abstract's "past the single-seed anchor in most groups" overstates the sufficiency data — a minority, not a majority, of groups strictly exceed the anchor at $k=\dmin$

**Disposition:** CONCEDE + FIX (framing fix; no new evidence needed)

**Response.** I re-pulled `crosscheck_recovered_frac_90` directly from
`experiment-runs/2026-07-09_m3fix_harvest/zero_pad__{G}__{arm}__seed0.json`
for all five groups, independently of the attack report's transcription:

| G | anchor | $k=\dmin$ | vs. anchor |
|---|---|---|---|
| S3 | 0.550 | 0.450 | below |
| S4 | 0.650 | 0.800 | exceeds |
| A5 | 0.700 | 0.700 | ties |
| S5 | 0.500 | 0.600 | exceeds |
| A6 | 0.650 | 0.650 | ties |

This reproduces the attack's table exactly. "Past the anchor" strictly holds
in 2/5 groups; "most groups" (a majority reading, ≥3/5) is not supported.
I also re-pulled the S3 four-seed extension
(`experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__{unconstrained,k_dmin}__seed{1,2,3}.json`)
and confirm the appendix table's numbers reproduce exactly (anchors
0.550/0.600/0.800/0.600, $k=\dmin$ values 0.450/0.550/0.600/0.650). The
attack's central claim is correct and the fix is not optional — this is the
same defect class FIX-1 (round 1) was written to close, reappearing in the
replacement text.

**Where I diverge from the attack's proposed fix.** The attack recommends
replacing "in most groups" with an explicit per-group breakdown anchored to
the *raw, single-seed unconstrained anchor* ("two of five groups… matched
exactly in two more… S3 the one group where the $k=\dmin$ mean sits below
its own anchor mean"). Two problems with keeping that comparison as the
abstract's headline number, even corrected:

1. **It is not this paper's actual decision criterion.** Nowhere else in
   the paper — not in Section 5's CONFIRM criterion, not in the
   pre-registered design — is "recovery vs. the *raw* anchor" the test that
   gates a verdict. The registered test (Section 5 paragraph 1, unedited:
   "$k \geq \dmin$ must recover past $0.9\times$ the anchor's $\recninety$")
   is the **$0.9\times$-anchor bar**, and Section 5's body text already
   reports every group clearing it: S4/A5/S5/A6 at their single seed
   (0.800/0.700/0.600/0.650 against bars 0.585/0.630/0.450/0.585, all
   cleared) and S3 at its four-seed mean (0.5625 against the fixed
   0.495 bar, cleared) under the pre-registered marginality-trigger
   routing (Table~\ref{tab:s3seeds}). The raw-anchor comparison the attack's
   fix leans on is a *different*, non-decisional number that happens to
   also appear in the same cells — using it as the abstract's headline
   figure re-introduces exactly the kind of ad hoc, non-pre-registered
   metric-shopping appearance A2/A8 (round 1) already had to clean up
   elsewhere in this paper.
2. **I independently caught an error in the attack's own supporting
   arithmetic that this alternative sidesteps entirely.** The attack states
   the paper's appendix table "reports the S3 four-seed anchor mean as
   0.574." I recomputed: $(0.550+0.600+0.800+0.600)/4 = 0.6375$, not
   0.574. `0.574` is actually $0.9 \times 0.6375 = 0.57375$ — the
   *recomputed bar*, not the raw anchor mean — which is exactly consistent
   with the appendix caption's own arithmetic ($0.5625 - 0.57375 \approx
   -0.011$, matching "0.011 below it" verbatim). The attack's proposed
   abstract sentence ("$S_3$… sits below its own anchor mean") is still
   *true* under the correct number (0.5625 < 0.6375), so this does not
   change A1's disposition — but it shows the raw-anchor framing is
   error-prone even for a careful hostile reviewer, which is itself a mark
   against building the abstract's headline sentence on it. See §4 below;
   I recommend a companion fix to the appendix wording that caused this.

**My recommended fix instead:** anchor the abstract to the $0.9\times$-bar
comparison, which is true for all five groups (not "two of five" or even
"four of five" — a strictly cleaner, stronger, and *correct* claim), matches
the paper's own decisive criterion everywhere else, and — checked by direct
word count on the current `main.tex` abstract (224 words, naive
whitespace-split) — costs exactly the same 10 words as the sentence it
replaces, so the abstract stays at 224 words, comfortably inside the
200–230 target the round-1 rebuttal verified compiles cleanly.

**Supporting evidence.** Direct JSON pulls above (independent of attack
report and design doc); `sections/05_causal.tex` body text ("clear their
pre-registered bars… S3… the seed-mean (0.5625) clears the fixed 0.495
bar"); `sections/07_appendix.tex` Table~\ref{tab:s3seeds} (own-bar column
matches $0.9\times$anchor exactly for all four S3 seeds); word count via
`main.tex`'s `\begin{abstract}...\end{abstract}` block, naive split = 224.

**What goes in the paper if this defense is accepted.** In `main.tex`'s
abstract, replace:
```
...a floor no capped cell violates, and restoring $\dmin$ lifts it:
recovery empirically returns, past the single-seed anchor in most groups.
```
with:
```
...a floor no capped cell violates, and restoring $\dmin$ lifts it:
recovery empirically clears the pre-registered $0.9\times$-anchor bar
in every group.
```
(10-for-10 word swap; no other clause in the sentence changes.) Section 5's
body text needs **no edit** for A1 — it already states the bar-clearance
result correctly (majority-of-groups framing there is already accurate,
since it correctly scopes "majority" to the four single-seed groups and
separately discloses S3's routing). If the authors prefer to keep the
raw-anchor comparison instead (e.g., because the "exceeds by up to +0.30"
pattern is judged independently interesting), the attack's own breakdown
sentence is an acceptable **alternative**, but belongs in Section 5's body
(where there is no tight word budget and the per-group table is already
adjacent), not the abstract, and should use the corrected anchor-mean
number (0.6375, not 0.574) if it cites S3's four-seed figure.

---

### A2: Section 5 paragraph 1's "one below-$\dmin$ cell recovering falsifies necessity" is left inconsistent with the section's own geometric-floor reframing two sentences later

**Disposition:** CONCEDE + FIX (framing fix; no new evidence needed)

**Response.** I read the actual pre-registered criterion in
`matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md:1656-1660` directly, rather
than accept either the attack's or the paper's paraphrase of it:

> "HARD FALSIFY (premise dead for this task family): `k=d_min(G)−1` reaches
> ≥0.9×ceiling at ANY group — a below-minimal-dimension solution exists
> despite the pinned block-embedded readout ⇒ **investigate a
> degauging/embedding leak first** (§1.9 item 1); **if real**, this task
> family is rank-blind in the way Task D's premise would have been."

This is not what Section 5 paragraph 1 currently says. The paper's sentence
— "one below-$\dmin$ cell recovering falsifies necessity" — presents a
bare, unconditional falsifier. The actual registered criterion was never
unconditional: it names a specific first hypothesis (instrument/embedding
leak) to rule out before any below-$\dmin$ recovery could be read as
falsifying anything, and I confirmed `§1.9 item 1` is literally the design's
own self-attacked #1 risk ("The degauging-rescue risk is the #1 risk in this
design"), not an incidental footnote. So the attack's framing — that
paragraph 1 is merely "in tension with" paragraph 2 and the appendix — is if
anything an *understatement*: paragraph 1 as written is a simplification
that drops content the pre-registration always carried, and the paper's
current text gives no reader access to the fact that a hedge was ever there.

Where the attack's diagnosis is exactly right and independently confirmed:
$\sqrt{2/3} = 0.8165 < 0.9$ (and all five groups' ceilings, recomputed
myself: 0.7071/0.8165/0.8165/0.8660/0.8944, all below 0.9) means no
rank-$(\dmin{-}1)$ state can clear the bar "however trained" — so the
"if real" branch of the registered criterion is now, post-derivation,
**never live**: any observed below-$\dmin$ recovery above the bar would
have to be a leak, full stop, not a coin flip between "leak" and
"real". Paragraph 1 should say this, not the bare-falsifier version.

**Supporting evidence.**
`matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md:1656-1660` (HARD FALSIFY
criterion, quoted above); `:2286-2296` (`§1.9` item 1, "The degauging-rescue
risk is the #1 risk in this design," confirming the cross-reference is to a
substantive, load-bearing risk item, not filler); `sections/05_causal.tex`
paragraph 1 (current text, confirmed unedited since round 1 per
`git show 788f452 -- sections/05_causal.tex`, matching the attack's own
citation); `sections/07_appendix.tex` "The rank-constrained cosine ceiling"
paragraph (the derivation that closes the leak-investigation contingency);
independently re-derived ceiling values
(0.7071/0.8165/0.8165/0.8660/0.8944 for $\dmin=2,3,3,4,5$).

**What goes in the paper if this defense is accepted.** In
`sections/05_causal.tex`, replace the paragraph-1 clause:
```
...on the pre-pinned conservative readout; one below-$\dmin$ cell
recovering falsifies necessity.
```
with:
```
...on the pre-pinned conservative readout; a below-$\dmin$ cell clearing
that bar would first indicate an instrument or embedding leak, not a
live falsification of necessity, a contingency the geometric ceiling
below closes outright, since no rank-$(\dmin{-}1)$ state can clear the
bar without one.
```
This keeps pre-registration's actual role (a criterion whose contingencies
were fixed before results were known — round 1's rebuttal defense of *why*
it was written this way stands) while stating what was actually
pre-registered, and it forward-references the appendix derivation instead
of contradicting it two sentences later.

---

## 3. New citations found during defense

None. Both attacks are claim-scope/internal-consistency issues internal to
this paper's own text and its own pre-registration record; neither required
new related-work engagement, and this round's scope (causal-razor,
convergence-framing, corrected-deviation) does not touch Related Work.

## 4. Attack the attacker missed

**The appendix's "(0.574)" is genuinely ambiguous and misled the round-2
attacker itself — worth an independent, low-severity fix.**
`sections/07_appendix.tex`, Table~\ref{tab:s3seeds}'s caption reads:
"recomputing the bar from the four-seed anchor mean (0.574) would put the
seed-mean value 0.011 below it." Read the natural way (the parenthetical
attaches to the nearest preceding noun phrase, "the four-seed anchor
mean"), a reader concludes the anchor mean is 0.574. It is not — the raw
four-seed anchor mean is 0.6375; 0.574 is $0.9\times$ that (the recomputed
bar), which is the only reading that reproduces "0.011 below" arithmetically
against the seed-mean $k=\dmin$ value (0.5625). This is not a hypothetical
confusion: it is exactly what led the round-2 attack report to mis-cite
"the paper's own new Table… reports the S3 four-seed anchor mean as 0.574"
in its A1 supporting evidence. It does not change A1's disposition (0.5625
is below the true anchor mean of 0.6375 too, so the attack's conclusion
survives), but a sentence that fooled a hostile close reader should be
disambiguated. **Rating: MINOR. Fix:** in `sections/07_appendix.tex`,
Table~\ref{tab:s3seeds}'s caption, replace "recomputing the bar from the
four-seed anchor mean (0.574)" with "recomputing the bar as $0.9\times$ the
four-seed anchor mean (raw mean 0.6375, bar 0.574)."

## 5. Attack ordering note

- **A1 (CRITICAL):** Rating agreed. It is the abstract's headline
  third-leg sentence and is factually unsupported as worded — the same
  category and location (the abstract's causal-razor clause) as round 1's
  A1/A10, both CRITICAL. Consistent precedent, correct severity.
- **A2 (SERIOUS):** Rating agreed, but I'd flag it as sitting at the high
  end of SERIOUS rather than the low end, now that the design-doc check is
  in: this is not merely an internal-consistency wrinkle between adjacent
  sentences (as the attack's own framing suggests) but a place where the
  paper's stated pre-registered criterion does not match what was actually
  registered. That is a sharper defect for a paper whose entire causal
  section leans on "pre-registered" as a credibility signal. It does not
  reach CRITICAL because no reported number, verdict, or headline claim
  changes — only the description of the criterion itself — and the fix
  remains a single clause with no experiment required.
- No attack in this round's scope struck me as over-rated; both hold at
  the severity assigned once checked against primary sources (raw JSONs
  for A1, the design record's literal HARD FALSIFY text for A2).
