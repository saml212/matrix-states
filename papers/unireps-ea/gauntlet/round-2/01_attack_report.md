# Attack Report — UniReps EA "Dimension, Not Solvability" (round 2, stage 01, SCOPED re-attack)

Reviewer: hostile attack agent, fresh context, round-2 scope only (per the
coordinator's brief: causal-razor cluster, convergence-framing cluster,
corrected-deviation claim). I did not re-attack anything outside these three
clusters — round 1's other dispositions (A2/A4/A6/A7/A8 residuals, the
citation additions) are treated as adjudicated and out of bounds here.

All numbers below were independently recomputed from the raw JSON artifacts
using `/Users/samuellarson/Experiments/learned-representations/.venv/bin/python`,
not taken from the draft, the brief, or the round-1 reports' prose.

## 1. Summary for the defense agent

**Verdict: one CRITICAL survives in the rewritten text, scoped narrowly to
five words in the abstract; everything else in scope holds.** The von
Neumann/Cauchy–Schwarz derivation in the new appendix is mathematically
correct end-to-end (I re-derived it independently and it reproduces
0.7071/0.8165/0.8165/0.8660/0.8944 exactly); the new S3 per-seed table
(`app:s3causal`) reproduces exactly from the raw JSONs, including the
self-critical "recomputed bar" arithmetic (0.57375 → 0.011 below); the
corrected "within 11%" deviation claim is numerically true (true max is
10.225%, correctly disclosed for S5); the primary-vs-crosscheck appendix
paragraph is consistent with a large, verified divergence between the two
metrics on this grid; no banned words or contractions appear in any of the
rewritten text.

The one finding that does not survive: the abstract's rewritten causal-razor
sentence ends "...restoring $\dmin$ lifts it: recovery empirically returns,
past the single-seed anchor **in most groups**." I recomputed
`crosscheck_recovered_frac_90` at $k=\dmin$ vs. the unconstrained anchor for
all five groups directly from
`experiment-runs/2026-07-09_m3fix_harvest/zero_pad__*__seed0.json`: recovery
strictly exceeds the anchor in **2 of 5** groups (S4, S5), exactly ties it in
2 (A5, A6), and is strictly **below** the anchor in the fifth (S3, both at
seed0 and at the four-seed mean the paper itself reports in
Table~\ref{tab:s3seeds}: 0.5625 < the four-seed anchor mean 0.574). "Most
groups" claims a majority (≥3 of 5); the data show a minority (2 of 5)
strictly exceeding. This is the same category of defect FIX-1 was written to
resolve (a quantified claim in the abstract that overstates what the
sufficiency data show), reappearing in the replacement text.

I also confirm a second, lower-severity finding the coordinator flagged for
scrutiny: Section 5 paragraph 1's pre-registration sentence ("one
below-$\dmin$ cell recovering falsifies necessity") is left, by design,
inconsistent with the same section's paragraph 2 and the new appendix, which
now state the below-$\dmin$ zero is a mathematical certainty ("not
discovered by training," "however trained") rather than a live empirical
outcome the design was ever at risk of observing otherwise. The round-1
rebuttal explicitly declined to fix this and asked a future round to
revisit if found insufficient — I find it insufficient, at SERIOUS severity.

**Salvageable:** the geometric-floor reframing project as a whole is sound
and well-executed — the appendix derivation, the S3 disclosure table, and
the "necessity is geometry, sufficiency is the discovery" framing are all
correct and internally consistent with each other. **Not yet salvageable as
written:** the specific five-word quantifier "in most groups" in the
abstract, and the residual falsifiability framing in Section 5's opening
paragraph.

## 2. Attacks

### A1: Abstract's "past the single-seed anchor in most groups" overstates the sufficiency data — a minority, not a majority, of groups strictly exceed the anchor at $k=\dmin$

**Severity:** CRITICAL
**Type:** number provenance / claim-scope

**Attack.** The abstract's rewritten third-leg sentence (the direct
replacement for round 1's "restores recovery," itself the paper's most
contested claim) reads:

> "...a floor no capped cell violates, and restoring $\dmin$ lifts it:
> recovery empirically returns, past the single-seed anchor in most groups."

I pulled `crosscheck_recovered_frac_90` for the `unconstrained` and `k_dmin`
arms of every group directly from
`experiment-runs/2026-07-09_m3fix_harvest/zero_pad__{G}__{arm}__seed0.json`:

| G | anchor (unconstrained) | $k=\dmin$ | vs. anchor |
|---|---|---|---|
| S3 | 0.550 | 0.450 | **below** ($-0.100$) |
| S4 | 0.650 | 0.800 | **exceeds** ($+0.150$) |
| A5 | 0.700 | 0.700 | **ties** ($0.000$) |
| S5 | 0.500 | 0.600 | **exceeds** ($+0.100$) |
| A6 | 0.650 | 0.650 | **ties** ($0.000$) |

Restoring rank to exactly $\dmin$ (the operation the sentence names —
"restoring $\dmin$") sends recovery *strictly past* the anchor in 2 of 5
groups (S4, S5), leaves it *exactly tied* in 2 (A5, A6), and leaves it
*below* the anchor in the fifth (S3). "Most groups" requires a majority
(≥3 of 5); the data show 2 of 5 strictly exceeding — a minority, and the
same fraction (2/5) whether or not one is generous about what "in most
groups" is meant to quantify over.

This is not an artifact of the single S3 seed: the paper's own new Table
(`tab:s3seeds`, Appendix~\ref{app:s3causal}) reports the S3 four-seed
anchor mean as 0.574 against the four-seed $k=\dmin$ mean of 0.5625 — S3 is
below the anchor under the more robust multi-seed comparison too, not just
at seed0.

The sentence is bolted onto the section's own parallel prose in a way that
undercuts the "most groups" reading further: Section 5's body text lists the
identical set of $k=\dmin$ values (0.800/0.700/0.600/0.650) when describing
sufficiency, i.e. the abstract's "restoring $\dmin$" and Section 5's
"clear their bars" describe the same $k=\dmin$ comparison — so the natural,
internally-consistent reading of "past the single-seed anchor in most
groups" is exactly the $k=\dmin$-vs-anchor comparison computed above, which
fails the "most" quantifier.

A more generous reading — crediting a group as "past the anchor" if
*either* $k=\dmin$ or $k=\dmin{+}1$ exceeds it — would get to 4 of 5 (S4,
A5, S5, A6 each exceed at $k=\dmin{+}1$ even where $k=\dmin$ ties or, for
none of them, falls below; S3 still fails, tying at $k=\dmin{+}1$, 0.550 =
0.550). But that reading requires conflating "restoring $\dmin$" with "using
one spare dimension beyond $\dmin$," which is not what the sentence says,
and it would contradict Section 5's own choice to cite $k=\dmin$ values
specifically as the sufficiency evidence.

**Supporting evidence.** Raw values: `zero_pad__S3__unconstrained__seed0.json`
(`crosscheck_recovered_frac_90=0.55`) vs. `zero_pad__S3__k_dmin__seed0.json`
(`=0.45`); `zero_pad__S4__unconstrained__seed0.json` (`=0.65`) vs.
`zero_pad__S4__k_dmin__seed0.json` (`=0.80`); `zero_pad__A5__unconstrained__seed0.json`
(`=0.70`) vs. `zero_pad__A5__k_dmin__seed0.json` (`=0.70`);
`zero_pad__S5__unconstrained__seed0.json` (`=0.50`) vs.
`zero_pad__S5__k_dmin__seed0.json` (`=0.60`);
`zero_pad__A6__unconstrained__seed0.json` (`=0.65`) vs.
`zero_pad__A6__k_dmin__seed0.json` (`=0.65`). S3 four-seed cross-check:
`experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__{unconstrained,k_dmin}__seed{1,2,3}.json`,
reproducing the paper's own Table~\ref{tab:s3seeds} anchors
(0.550/0.600/0.800/0.600, mean 0.574) and $k=\dmin$ values
(0.450/0.550/0.600/0.650, mean 0.5625).

**What the paper would need to do to defuse this.** Replace "in most
groups" with the accurate count, e.g. "recovery empirically returns, past
the single-seed anchor in two of five groups (S4, S5), matched exactly in
two more (A5, A6), with $S_3$ the one group where the $k=\dmin$ mean sits
below its own anchor mean." This is the same length of fix as FIX-1 itself
(reword only, no new data), and it keeps every clause the fix already
earned — the geometric-floor reframing of necessity, and "recovery
empirically returns" (true in all 5 groups, since $k{=}\dmin$ is nonzero
everywhere) — while dropping the one quantifier the raw data do not support.

---

### A2: Section 5 paragraph 1's "one below-$\dmin$ cell recovering falsifies necessity" is left inconsistent with the section's own geometric-floor reframing two sentences later

**Severity:** SERIOUS
**Type:** claim-scope / internal consistency

**Attack.** Section 5 opens with the pre-registered reading (unedited by
the round-1 fix, confirmed via `git show 788f452 -- sections/05_causal.tex`,
where the diff starts after this sentence):

> "Pre-registered reading: if $\dmin$ is load-bearing, $k = \dmin{-}1$ must
> fail and $k \geq \dmin$ must recover past $0.9\times$ the anchor's
> $\recninety$ on the pre-pinned conservative readout; **one below-$\dmin$
> cell recovering falsifies necessity.**"

Two sentences later, the (rewritten) body text states:

> "The floor is set by the readout's own geometry, **not discovered by
> training**..."

And the new appendix derivation is explicit that this outcome could never
have occurred, independent of what SGD does:

> "no rank-$(\dmin{-}1)$ state, **however trained**, can register a single
> word above cosine 0.9 against this target."

If a rank-$(\dmin{-}1)$ state cannot register above cosine 0.9 "however
trained," then "one below-$\dmin$ cell recovering" was never a live
possibility the pre-registered design was at risk of observing — the
"falsifier" in paragraph 1 had probability zero of firing regardless of
what SGD found, given the metric's own geometry (verified: $\sqrt{2/3} =
0.8165 < 0.9$ even for the most favorable case, $S_4$/$A_5$). Paragraph 1's
framing — a criterion whose failure "falsifies necessity" — presents this as
a genuine, at-risk pre-registered empirical test, precisely the framing
paragraph 2 and the appendix now go out of their way to retract two
sentences later, in the same section, for the same result. A reader who
stops at paragraph 1 (or who reads the section once, linearly, as written)
is told the below-$\dmin$ leg is a live falsifier; a reader who continues to
the appendix is told the same leg's outcome was fixed before training
started. Both statements are in the paper; they contradict each other about
what kind of claim "necessity" is.

The round-1 rebuttal report explicitly flagged this and chose not to fix it
("Not fixed, by design... Revisit only if a future round finds this
insufficient on its own," §5 of `gauntlet/round-1/04_rebuttal_report.md`).
This round finds it insufficient: the geometric-floor rewrite elsewhere in
the same section makes the tension more visible, not less, because the
reader now has the correct framing in hand by the time they finish the
paragraph that still uses the old one.

**Supporting evidence.** `sections/05_causal.tex` paragraph 1 (unedited,
confirmed via `git show 788f452`); `sections/07_appendix.tex`,
"The rank-constrained cosine ceiling" paragraph (rewritten in the same
commit); `gauntlet/round-1/04_rebuttal_report.md` §5, "Not fixed, by
design."

**What the paper would need to do to defuse this.** One clause in
paragraph 1: replace "one below-$\dmin$ cell recovering falsifies
necessity" with language that matches what such an outcome would actually
mean given the now-derived ceiling — e.g. "one below-$\dmin$ cell recovering
above the metric's own geometric ceiling would indicate an instrument
error, not a live empirical outcome (Appendix~\ref{app:instrument})." This
keeps the pre-registration's role (a criterion fixed before results were
known — the rebuttal's point stands as a defense of *why* it was written
this way originally) while removing the retrospective inconsistency with
what the paper now knows and states two sentences later.

---

## 3. Attacks I considered but decided were weak

- **The appendix's "$[0.895, 0.951]\cdot\dmin$" band range (M1 non-binding
  upper-half paragraph).** Recomputing the five group ratios
  (mean/$\dmin$) gives $[0.8978, 0.9507]$, which more tightly rounds to
  $[0.898, 0.951]$ rather than the stated $[0.895, 0.951]$ — a 0.003 gap on
  the lower bound. This paragraph is **not** one of the three items named in
  this round's scope (rank-constrained cosine ceiling; primary-vs-crosscheck
  paragraph; S3 per-seed table) — it is FIX-9, adjudicated under A2 in round
  1 and not part of the causal-razor or convergence-framing clusters this
  round re-attacks. Noting it here for the record rather than as a formal
  finding, since pursuing it would violate the round's scope discipline.
- **"Within 11%" is a loose ceiling given the true max deviation is
  10.225%.** Recomputed all five deviations exactly (S3 6.15%, S4 4.93%, A5
  5.60%, S5 10.225%, A6 5.28%); "within 11%" is true (10.225 < 11) and the
  same sentence immediately discloses the exact S5 figure (10.2%) in
  parentheses, so there is no concealment — a reader gets the precise number
  one clause later. A tighter ceiling (e.g. "within 10.3%") would be
  marginally more informative, but "within 11%" is not inaccurate, and the
  round-1 rebuttal already fixed the actual defect (the prior "4-8%" range
  that excluded S5 entirely). Not pursuing as an independent finding.
- **The primary-vs-crosscheck paragraph's claim that "the two diverge
  sharply on this grid."** Recomputed `mean_cos`/`recovered_frac_90` (the
  scale-only primary) against `crosscheck_mean_cos`/`crosscheck_recovered_frac_90`
  for all 20 zero-pad seed0 cells: the divergence is large and consistent
  everywhere (e.g. $S_4$ unconstrained: primary `recovered_frac_90=0.05` vs.
  crosscheck `=0.65`; $S_5$ unconstrained: primary `recovered_frac_90=0.00`
  vs. crosscheck `=0.50`). The claim is accurate as written; no attack
  surface here.
- **The von Neumann/Cauchy–Schwarz derivation's tightness.** The appendix
  does not claim the bound $\cos(M,T) \leq \sqrt{k/\dmin}$ is achieved by
  the model's actual trained states (only that it is an upper bound), so no
  claim about tightness needs to be defended; I confirmed the inequality
  chain independently (trace inequality reduces to the first $k$ terms
  because $M$ has at most $k$ nonzero singular values regardless of $T$'s
  spectrum; $\sigma_i(T)=1$ for $i \leq \dmin$ holds because $\rho_G$ is
  orthogonal by construction and orthogonal matrices have unit singular
  values, so every word's target image is exactly rank $\dmin$ with a flat
  spectrum) and it reproduces the stated ceilings to four decimal places.
  No attack surface.
- **Whether the derivation's bound applies to the *degauged* ($\recninety$)
  score, not just a raw cosine.** Checked: the fitted gauge $(c,Q)$ is a
  scale-and-orthogonal-rotation transform, which cannot increase matrix
  rank (Section 2 states this explicitly: "Rank is invariant under the
  fitted gauge"), so a force-rank-$k$ state's degauged image is still rank
  $\leq k$, and the bound applies to it unchanged. This closes what would
  otherwise be a real gap (a fitted transform that increased effective
  rank would break the appendix's claim that "however trained" no
  rank-$(\dmin{-}1)$ state can exceed the ceiling). Confirmed sound, not
  an attack.

## 4. New citations found during this round

None. The related-work engagement (Chughtai et al. 2023, Huh et al. 2024)
is outside this round's three scoped clusters and was independently
verified as applied in round 1; no new citation gaps surfaced within the
causal-razor, convergence-framing, or deviation-claim text reviewed here.
