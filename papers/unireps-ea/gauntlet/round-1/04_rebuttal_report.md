# Rebuttal / Adjudication Report — UniReps EA "Dimension, Not Solvability" (round 1, stage 04)

Adjudicator: fresh-context rebuttal agent. I did not take the attack or defense
report's numbers on trust. I re-verified the load-bearing new claim (A10) from
first principles (a von Neumann trace-inequality / Cauchy-Schwarz derivation,
independently coded and checked against `sqrt((d_min-1)/d_min)` for
`d_min = 2,3,3,4,5`), cross-checked it against the design record's own
"oracle upper bound" language (`CAPABILITY_SEPARATION_DESIGN.md:6001`, which
independently states "xcos 0.610-0.836 all under the 0.894 oracle upper
bound"), grepped the current draft directly for the defense's quoted "PRIMARY"
language (not present in `sections/02_setup.tex` — see A8 below, a real
correction to the defense report), and re-pulled the design doc's S3
per-seed table (`CAPABILITY_SEPARATION_DESIGN.md:6144-6150`) to confirm the
2/4 disclosure verbatim. I then wrote every fix below into a scratch copy of
the paper and **compiled it with `tectonic`**: the edited draft compiles
clean, and the main text (through Related Work and Limitations) still ends
on page 4 with References — an identical page boundary to the unedited
baseline — while the (page-unlimited) appendix absorbs the new derivations
and grows from page 5 onward. The fix list below is not aspirational; it is
verified typesettable within the venue's 4pp main-text budget.

## 0. Is any CRITICAL open?

**No CRITICAL remains open, conditional on FIX-1, FIX-2, and FIX-3 being
applied and the affected claims re-entering attack/defense/rebuttal (see
§6).** Two attacks carry CRITICAL severity (A1 and the coordinator-supplied
A10). Both are resolved by a real rescoping of the abstract's and Section
5's causal-razor language — not a "defer to camera-ready" answer, not a
retraction of any result — and that rescoped language has been test-compiled
inside the page budget. Nothing in this paper's results is false; what
changes is which half of the causal razor is claimed as an SGD discovery
(sufficiency) versus a property of the readout's own geometry verified by
construction (necessity).

## 1. Summary for the writer

**Disposition counts (10 attacks, A1-A10):** 0 DEFENSE VALID · 4 DEFENSE
VALID BUT EDIT (A1, A5, A7, A9) · 3 PARTIAL — ATTACK SURVIVES IN REDUCED FORM
(A4, A6, A8) · 1 DEFENSE INSUFFICIENT, resolved via fix (A3) · 1 no formal
defense, adjudicated directly and resolved via fix (A10, CRITICAL by import,
not by either round's own label — see §2).

**13 fixes, by severity:** 3 CRITICAL, 9 SERIOUS, 1 MINOR.

**The four fixes that carry the weight:**

1. **FIX-1 (abstract) + FIX-3 (Section 5 body).** Recast the causal razor's
   necessity leg ("capping rank one below $d_{\min}$ zeroes exact recovery")
   from an SGD discovery into a metric-geometry certainty verified by
   construction, and elevate the true discovery — recovery empirically
   *returning* at $d_{\min}$, and often exceeding the single-seed anchor —
   as the paper's actual causal finding. This resolves A1, A5, A9, and A10
   together, because all four attacks land on the same two sentences.
2. **FIX-2 (introduction, contribution bullet 3).** The same rescoping at
   the contribution-list level, so the abstract and the intro's own summary
   of themselves do not disagree after FIX-1/FIX-3 land.
3. **FIX-8 (new appendix derivation).** The math backing FIX-1/FIX-3: a
   four-line trace-inequality proof that a rank-$(d_{\min}-1)$ state cannot
   exceed cosine $\sqrt{(d_{\min}-1)/d_{\min}}$ against this paper's
   all-unit-singular-value target, and that this ceiling is below 0.9 for
   every group tested (0.707/0.816/0.816/0.866/0.894). Verified against the
   design record's own independent "oracle upper bound" language.
4. **FIX-12 (related work, two new citations).** Chughtai, Chan & Nanda
   (ICML 2023) and Huh, Cheung, Wang & Isola (ICML 2024) are the two
   MUST-cite gaps confirmed live by both the attack and defense rounds and
   by my own independent fetch of both papers' arXiv/PMLR records.

**A correction to the defense report, surfaced during adjudication (affects
A8's severity, not its disposition):** the defense report quotes "§2 states
plainly: 'scale-only ($\hat c$) is PRIMARY... retained and reported as a
cross-check'" as if this were text in the current draft. It is not — I
grepped `sections/*.tex` directly and that language does not appear anywhere
in the paper; it is design-record-only language
(`CAPABILITY_SEPARATION_DESIGN.md:2381,3282`). The paper's own §2 never
introduces a "primary" metric at all, so there is no internal
self-contradiction between a stated primary and the reported crosscheck —
the defense's "silent switch" framing overstates what is actually on the
page. The real, narrower gap survives: the term "crosscheck" appears in the
Figure 2 caption and Appendix B undefined, and the paper never discloses
that a simpler alternative metric exists or why the causal wave specifically
requires the fuller fit. Fixed via FIX-10 (appendix-only; no main-text
budget spent), not the main-text rewrite the defense's framing might have
implied.

## 2. Adjudication and final verdict per attack

### A1 — "restoring $d_{\min}$ restores recovery" (unconstrained anchor unreliable)
**Attack severity:** CRITICAL. **Final verdict: DEFENSE VALID BUT EDIT.**
The pre-registered CONFIRM criterion itself (recover past $0.9\times$ a
single anchor draw) is a legitimate, if underpowered, one-sided floor test —
the defense is right that nothing in the *test design* is broken. What
cannot survive as written is the abstract's and §5's prose: "restores"
implies parity, and the data show capped cells beating the anchor by up to
+0.30 (46% relative) in a majority of groups. Resolved by FIX-1, FIX-3.
Interacts with A9 and A10 (below) — all three land on the same two
sentences and are fixed together.

### A2 — M1's band is one-sided dressed as two-sided
**Attack severity:** SERIOUS. **Final verdict: DEFENSE VALID BUT EDIT.**
Verified independently: `entity_subspace_from_words` returns exactly
$d_{\min}$ columns by construction, so `restricted_effective_rank` cannot
exceed $d_{\min}$, hence cannot exceed $1.3\cdot d_{\min}$, for any seed of
any group. The draft's existing "partially favors rank $\approx d_{\min}$"
language already gestures at this but stops short of the sharp,
purely-mathematical statement. Resolved by FIX-6 (combined with A7) and
FIX-9 (appendix proof).

### A3 — Introduction's "4-8%" contradicts the paper's own Table 1
**Attack severity:** SERIOUS (attacker); defense downgrades to
MINOR-priority/free-fix. **Final verdict: DEFENSE INSUFFICIENT** (no
defense was mounted or possible — the arithmetic is a plain, uncontested
error: $(4-3.591)/4 = 0.1023$, 2.2 points outside the stated range) —
**resolved via FIX-13**, a one-character-class number correction touching
no other claim. Final severity for the fix list: MINOR.

### A4 — Unequal, non-monotonic training budgets confound the comparison
**Attack severity:** SERIOUS. **Final verdict: PARTIAL — ATTACK SURVIVES IN
REDUCED FORM.** The design record's §1.30 shows budgets were the output of a
symmetric, pre-registered per-group convergence-bar clearance process, not
picked to favor a narrative — this meaningfully narrows "arbitrary
confound." It does not eliminate the residual pattern: $S_3$ and $S_5$, the
two groups that stayed at the un-escalated base budget, are also the two
with the largest observational deviation (A3) and the one marginal causal
cell. Resolved (as a disclosed caveat, not a redesign) by FIX-7.

### A5 — "Step function" overclaims a plateau
**Attack severity:** SERIOUS. **Final verdict: DEFENSE VALID BUT EDIT.** The
necessity half of the metaphor (a hard floor below $d_{\min}$) is exact and
earned. The plateau half is not: recovery keeps climbing past $d_{\min}$ in
3 of 5 groups (S4 +0.150, A5 +0.050, A6 +0.050). Resolved by FIX-3, FIX-4
(figure caption).

### A6 — The target is directly supervised at rank $d_{\min}$; "convergence" is partly definitional
**Attack severity:** SERIOUS. **Final verdict: PARTIAL — ATTACK SURVIVES IN
REDUCED FORM**, narrower after A10 than the defense report allowed. The
attack's premise is exactly right: `rho_G_embedded` builds the target as
`eye` with `rho` overwriting only the top-left $d_{\min}\times d_{\min}$
block, so *some* minimal notion of "$d_{\min}$ suffices" is baked into the
loss before training starts. The defense's counter rested on two things
remaining non-definitional: (1) M1 shows the model does not recruit rank
into its 2 spare ambient dimensions, and (2) the causal necessity result's
"cleanness" is an empirical fact about training dynamics. A10 removes most
of (2) — the necessity zero's cleanness is now understood to be a metric
ceiling, not a training-dynamics fact — so what survives is narrower than
the defense claimed: (1) alone, plus sufficiency (§5's now-correctly-scoped
empirical finding). Resolved by FIX-2, FIX-5 (intro reframe to "does
recruited capacity settle at $d_{\min}$," not "does the network discover
its own algebra").

### A7 — Title's "Not Solvability" rests on one pair, not five groups
**Attack severity:** MINOR. **Final verdict: DEFENSE VALID BUT EDIT.**
Confirmed: dimension and solvability are confounded in this 5-group set
outside the engineered $S_4$/$A_5$ tie. Already partly self-aware in the
body ("corroborating rather than independently decisive"). One clarifying
clause suffices; no title change needed. Resolved by FIX-6 (combined with A2).

### A8 — Silent primary/crosscheck metric switch [defense-surfaced]
**Attack severity (defense's rating):** SERIOUS. **Final verdict: PARTIAL —
ATTACK SURVIVES IN REDUCED FORM, on a corrected factual basis** (see §1
above). The paper's own §2 never states a "primary" metric to contradict —
I grepped `sections/*.tex` and confirmed zero hits for "primary." What
survives: "crosscheck" is used undefined in the Figure 2 caption and
Appendix B, and the paper never discloses that (a) a simpler scale-only
degauging exists and is the default elsewhere in the underlying pipeline,
or (b) why the causal wave specifically overrides that default (the
zero-padded target's degenerate spectrum breaks the $\hat Q \approx I$
regularity scale-only relies on). This is a real disclosure gap, not
evidence of post-hoc metric-shopping. Resolved via FIX-10, appendix-only.

### A9 — S3's 2/4 per-seed bar clearance undisclosed [defense-surfaced]
**Attack severity (defense's rating):** folds into A1. **Final verdict:
DEFENSE VALID BUT EDIT.** Verified against
`CAPABILITY_SEPARATION_DESIGN.md:6144-6150`: seeds 1 and 3 individually
clear their own $0.9\times$anchor bar (+0.010, +0.110); seeds 0 and 2 miss
(−0.045, −0.120). The pre-registered criterion is the seed-mean against a
bar *fixed before the extension ran* (0.495, from seed 0's own anchor) —
deliberately not recomputed from the extension's own noisier anchors, to
avoid laundering anchor noise into the threshold the new data is judged
against. **This is good methodology, and S3 remains legitimately CONFIRMED
under its own pre-registered criterion** — the fix is disclosure of the
per-seed detail, not a retraction. Resolved by FIX-3 (pointer) + FIX-11
(new appendix table with the full breakdown).

### A10 — Necessity zero is forced by metric geometry, not an SGD discovery [coordinator-supplied]
**No formal defense round exists for this attack** (introduced directly to
the rebuttal stage). **Final verdict: adjudicated directly — attack
confirmed true by independent derivation, CRITICAL, resolved via fix.**

I re-derived the bound from scratch rather than trust the coordinator's
numbers. The causal wave's target $T = \rho_G(\cdot)\oplus 0$ has rank
exactly $d_{\min}$ with all $d_{\min}$ informative singular values equal to
1 (the reference representation is orthogonal by construction). For any
matrix $M$ of rank $\leq k \leq d_{\min}$, von Neumann's trace inequality
gives $\langle M,T\rangle_F \leq \sum_{i=1}^k \sigma_i(M)\sigma_i(T) =
\sum_{i=1}^k \sigma_i(M)$, and Cauchy-Schwarz bounds
$\sum_{i=1}^k\sigma_i(M) \leq \sqrt{k}\lVert M\rVert_F$, so
$\cos(M,T) \leq \sqrt{k/d_{\min}}$, tight when $M$'s $k$ equal singular
values align with any $k$ of $T$'s directions. Computed for
$k=d_{\min}-1$, $d_{\min}\in\{2,3,3,4,5\}$:

```
d_min=2  ->  sqrt(1/2) = 0.7071
d_min=3  ->  sqrt(2/3) = 0.8165  (S4 and A5)
d_min=4  ->  sqrt(3/4) = 0.8660
d_min=5  ->  sqrt(4/5) = 0.8944
```

All five values are below the $\mathrm{rec}@0.9$ threshold. So
$\mathrm{rec}@0.9 = 0.000$ at $k = d_{\min}-1$ is **mathematically
guaranteed for any rank-$(d_{\min}-1)$ state, trained well or trained
badly** — not a fact SGD discovered. This is independently corroborated by
the design record itself, which already calls this quantity the "oracle
upper bound" (`CAPABILITY_SEPARATION_DESIGN.md:6001-6002`: "xcos 0.610-0.836
all under the 0.894 oracle upper bound") — the project's own instrumentation
already knew this was a ceiling, but the paper's prose ("Necessity is exact
and noiseless") never surfaces that it is a *forced* ceiling rather than an
emergent one.

This does not evacuate the causal section's content. At $k \geq d_{\min}$
the ceiling is 1.0 (exact recovery is representable), so recovery
*returning* above 0 at $d_{\min}$ is not guaranteed by the metric and
remains the paper's genuine, non-tautological causal finding — this is
exactly the reframe the coordinator's brief requested: necessity recast as
verified-by-construction, sufficiency kept as the discovery. The below-bar
cosines (0.61-0.84) sitting under, and the at/above-bar cosines
approaching, their respective rank-constrained optima is itself a real
empirical fact (nothing forces SGD to reach the achievable ceiling rather
than get stuck below it), and is now stated explicitly rather than left
implicit. Resolved by FIX-1, FIX-2, FIX-3, FIX-4, FIX-8.

## 3. The fix list

### CRITICAL

#### FIX-1: Recast the abstract's causal-razor sentence
**Severity:** CRITICAL
**File(s):** `main.tex`
**Location:** Abstract, final third-headline sentence (before "Learned representational dimension follows...")
**Before:**
```
A pre-registered causal force-rank razor then confirms the
converged dimension is load-bearing rather than a correlate of training:
capping rank one below $\dmin$ zeroes exact recovery in every group, and
restoring $\dmin$ restores recovery.
```
**After:**
```
A pre-registered causal force-rank razor adds a third leg: capping rank
one below $\dmin$ pins recovery under 0.9 by the metric's own geometry
($\sqrt{(\dmin{-}1)/\dmin} < 0.9$ for every group tested), a floor no
capped cell violates, and restoring $\dmin$ lifts it: recovery
empirically returns, past the single-seed anchor in most groups.
```
**Why:** Resolves A1 (drops "restores," which implies parity when capped
cells beat the anchor by up to +0.30 in a majority of groups) and A10
(necessity reframed as a geometric floor, not a training-discovered fact).
Test-compiled: new abstract word count is 224 (within the 200-230
requirement).

#### FIX-2: Recast introduction's contribution-3 bullet
**Severity:** CRITICAL
**File(s):** `sections/01_intro.tex`
**Location:** Paragraph 2 (contribution list), third sentence
**Before:**
```
Third, the converged dimension is causally load-bearing, not a correlate:
capping rank at $\dmin{-}1$ during training zeroes exact recovery in every
group while $\dmin$ restores it (Section~\ref{sec:causal}).
```
**After:**
```
Third, the converged dimension is causally load-bearing: capping rank at
$\dmin{-}1$ pins recovery below the readout's own geometric ceiling in
every group, and at $\dmin$ that ceiling lifts and recovery empirically
returns (Section~\ref{sec:causal}).
```
**Why:** A1, A6, A10 — keeps the contribution list consistent with FIX-1
and FIX-3 after they land; otherwise the abstract and the intro's own
one-paragraph summary of the paper would state the finding two different,
now-inconsistent ways.

#### FIX-3: Rewrite Section 5's necessity/sufficiency paragraph
**Severity:** CRITICAL
**File(s):** `sections/05_causal.tex`
**Location:** Second body paragraph (immediately after Figure 2)
**Before:**
```
Figure~\ref{fig:razor} shows a step function at $\dmin$. Necessity is
exact and noiseless: $k = \dmin{-}1$ yields $\recninety = 0.000$ in all
five groups
% <!-- evidence: U3 -->
and in all four independent seeds of the $S_3$ extension,
% <!-- evidence: U4 -->
while the same cells' recovery cosines (0.61--0.84) show the zero is a
threshold, not a training collapse. Sufficiency holds at $\dmin$: $S_4$,
$A_5$, $S_5$, $A_6$ recover past their bars outright (0.800/0.700/0.600/
0.650 against 0.585/0.630/0.450/0.585),
% <!-- evidence: U3 -->
and $S_3$, whose decisive cell landed inside a pre-stated $\pm 0.05$
marginality trigger, was extended to four seeds under the trigger's
pre-registered routing: seed-mean 0.5625 against the fixed 0.495 bar
confirms the fifth group.
% <!-- evidence: U4 -->
```
**After:**
```
Figure~\ref{fig:razor} shows a necessity floor at $\dmin$, not a
plateau. The floor is set by the readout's own geometry, not discovered
by training: $k = \dmin{-}1$ yields $\recninety = 0.000$ in all five
groups
% <!-- evidence: U3 -->
and in all four independent seeds of the $S_3$ extension,
% <!-- evidence: U4 -->
exactly as the bound $\sqrt{(\dmin{-}1)/\dmin} < 0.9$ requires
(Appendix~\ref{app:instrument}); the same cells' recovery cosines
(0.61--0.84) sit under that ceiling, evidence the capped models reached
their rank-constrained optimum rather than collapsing. Sufficiency is
the empirical result: $S_4$, $A_5$, $S_5$, $A_6$ clear their
pre-registered bars (0.800/0.700/0.600/0.650 against
0.585/0.630/0.450/0.585),
% <!-- evidence: U3 -->
in a majority of groups against a single noisy anchor seed rather than a
stable ceiling,
and $S_3$, whose decisive cell landed inside a pre-stated $\pm 0.05$
marginality trigger, was extended to four seeds under the trigger's
pre-registered routing: the seed-mean (0.5625) clears the fixed 0.495
bar, though only 2 of 4 seeds individually (Table~\ref{tab:s3seeds}).
% <!-- evidence: U4 -->
```
**Why:** Resolves A1 (anchor-noise disclosure, "in a majority of groups
against a single noisy anchor seed"), A5 ("necessity floor... not a
plateau" replaces "step function"), A9 (per-seed 2/4 disclosure, pointed at
the new Appendix table via FIX-11), and A10 (necessity reframed as
"set by the readout's own geometry, not discovered by training," with the
exact bound cited and pointed at the new Appendix proof, FIX-8). All
evidence tags (U3, U4) preserved at their original anchor points — no
number loses its citation. Note S3 remains CONFIRMED under the
pre-registered fixed-bar-on-seed-mean criterion; this fix discloses, not
retracts.

### SERIOUS

#### FIX-4: Figure 2 caption — geometric-floor pointer
**Severity:** SERIOUS
**File(s):** `sections/05_causal.tex`
**Location:** `\caption{...}` of `fig:razor`
**Before:**
```
bar (dotted). Recovery is exactly 0.000 at $\dmin{-}1$ in every group
and returns at $\dmin$. $S_3$ overlays its four seeds (thin lines; bold
mean); its below-$\dmin$ zero is unanimous.}
```
**After:**
```
bar (dotted). Recovery is exactly 0.000 at $\dmin{-}1$ in every group,
the readout's geometric floor (Appendix~\ref{app:instrument}), and
returns at $\dmin$. $S_3$ overlays its four seeds (thin lines; bold
mean); its below-$\dmin$ zero is unanimous.}
```
**Why:** A5, A10 — a reader who only reads the figure caption (common
reviewer behavior) should not come away with "step function, SGD-discovered
necessity" once the body text says otherwise.

#### FIX-5: Introduction paragraph 1 — scope the "converges to the algebra" framing
**Severity:** SERIOUS
**File(s):** `sections/01_intro.tex`
**Location:** Paragraph 1, second-to-last sentence
**Before:**
```
faithful real linear representation. If trained models converge to the
task's algebra, states trained on different groups should recruit
representational dimension tracking $\dmin$; if instead the difficulty
axis governs, dimension should sort with the group's computational class,
solvable versus non-solvable, the divide at which word problems become
$\mathrm{NC}^1$-complete \citep{barrington1989} and at which
state-tracking expressivity results separate recurrent architectures
\citep{merrill2024illusion, grazzi2025negative}.
```
**After:**
```
faithful real linear representation. The loss target already fixes
$\dmin$ as the informative rank; the open question is whether the
model's own recruited capacity, two ambient dimensions wider than
$\dmin$, settles there too, tracking the group's algebra, or instead
sorts with the group's computational class, solvable versus
non-solvable, the divide at which word problems become
$\mathrm{NC}^1$-complete \citep{barrington1989} and at which
state-tracking expressivity results separate recurrent architectures
\citep{merrill2024illusion, grazzi2025negative}.
```
**Why:** A6 — the original phrasing ("if trained models converge to the
task's algebra") reads as though $d_{\min}$ is an unknown quantity the
network discovers, when the loss target's own rank already fixes it before
training starts. The rewrite states the true open question: does the
model's *own* rank (which has 2 spare ambient dimensions of freedom the
loss does not force) settle at $d_{\min}$ anyway.

#### FIX-6: Section 3 — M1 band non-binding upper half + solvability-dissociation scope
**Severity:** SERIOUS
**File(s):** `sections/03_convergence.tex`
**Location:** Final sentence of the section
**Before:**
```
By pre-registration this observational leg is corroborating rather than
independently decisive, because a $\dmin$-restricted readout partially
favors rank $\approx \dmin$ for near-orthogonal blocks; the causal weight
is carried by Section~\ref{sec:causal}.
```
**After:**
```
By pre-registration this observational leg is corroborating: the
readout's $[1,\dmin]$ range makes the band's upper half non-binding by
construction (Appendix~\ref{app:instrument}), and the dissociation is
carried specifically by $S_4$/$A_5$ (Section~\ref{sec:equivalence}), not
the five-group trend; the causal weight is carried by
Section~\ref{sec:causal}.
```
**Why:** A2 (states plainly, not just implies, that the band's upper half
$\leq 1.3\cdot\dmin$ cannot bind — `restricted_effective_rank` has range
$[1,\dmin]$ by construction of `entity_subspace_from_words`, so it can
never approach let alone exceed $\dmin$) and A7 (the dimension/solvability
dissociation is carried by the $S_4$/$A_5$ pair specifically, not by the
five-group correlational trend — one clause, no title change needed).

#### FIX-7: Limitations — link the budget schedule to the two flagged groups
**Severity:** SERIOUS
**File(s):** `sections/06_related_limits.tex`
**Location:** Limitations paragraph, second sentence
**Before:**
```
Causal cells are single-seed except $S_3$ (four seeds), with the necessity
zeros exact wherever seeds exist; two groups at the shortest pinned budget
show soft convergence, disclosed in the run records, with the causal
reading anchor-relative and unaffected.
```
**After:**
```
Causal cells are single-seed except $S_3$ (four seeds), with the
necessity zeros exact wherever seeds exist; two groups at the shortest
pinned budget ($S_3$, $S_5$) show soft convergence and are also the two
with the largest observational deviation and the one marginal causal
cell, a pre-registered, not post hoc, budget schedule but a caveat on
cross-group comparison strength.
```
**Why:** A4 — the design record's §1.30 shows the per-group budgets were
set by a symmetric, pre-registered convergence-bar clearance process (not
picked to favor a narrative), which narrows the confound concern; but $S_3$
and $S_5$ (the two groups that stayed at the base 8K budget) are also the
two with the largest deviations elsewhere in the paper, a pattern the
original text disclosed piecemeal but never connected. This sentence
connects it without retracting anything.

#### FIX-8: New appendix paragraph — the rank-constrained cosine ceiling (derivation)
**Severity:** SERIOUS
**File(s):** `sections/07_appendix.tex`
**Location:** End of `app:instrument` (Instrument Details and the Two
Defects), immediately before `\section{Per-Seed Observational Rank}`
**Before:** (insert point — no existing text is removed)
```
The sweep verdict was registered as inconclusive with the tax as
mechanism; the zero-padded causal wave of Section~\ref{sec:causal} is
the registered fix, its 30 cells configuration-verified one-by-one
against a manifest re-derived independently from the design record. In
that wave an eye-padded corroboration arm reproduces the failure on
demand at raw $k = \dmin{+}1$ (effective budget $\dmin{-}1$) while the
tax-paid point recovers.
% <!-- evidence: U3 -->

\section{Per-Seed Observational Rank}
\label{app:m1}
```
**After:** (insert the following new paragraph between the two, keep both
existing pieces unchanged)
```
\textbf{The rank-constrained cosine ceiling.} The causal wave's target
is $\rho_G(\cdot)\oplus 0$, of rank exactly $\dmin$ with all $\dmin$
informative singular values equal to 1 (the reference representation is
orthogonal by construction, Appendix~\ref{app:groups}). For any matrix
$M$ of rank $\leq k \leq \dmin$, von Neumann's trace inequality gives
$\langle M, T\rangle_F \leq \sum_{i=1}^{k}\sigma_i(M)\,\sigma_i(T) =
\sum_{i=1}^k \sigma_i(M)$, and Cauchy--Schwarz bounds
$\sum_{i=1}^k \sigma_i(M) \leq \sqrt{k}\,\lVert M\rVert_F$, so
$\cos(M,T) \leq \sqrt{k/\dmin}$, achieved when $M$'s $k$ equal singular
values align with any $k$ of $T$'s $\dmin$ directions. At
$k = \dmin{-}1$ this ceiling is $0.707/0.816/0.816/0.866/0.894$ for
$S_3/S_4/A_5/S_5/A_6$ respectively, below the $\recninety$ threshold of
$0.9$ in every case: no rank-$(\dmin{-}1)$ state, however trained, can
register a single word above cosine 0.9 against this target. The
necessity result ($\recninety = 0.000$ at $\dmin{-}1$, all groups, all
seeds) is this bound realized, not an independent discovery about what
SGD finds; the corresponding empirical content is that the observed
cosines at $\dmin{-}1$ (0.61--0.84) sit under, and the observed cosines
at $\dmin$ and above regularly approach, their respective
rank-constrained optima, evidence optimization reaches the geometry's
own ceiling rather than an artifact of the ceiling itself. At
$k \geq \dmin$ the ceiling is 1.0 (exact recovery is representable), so
recovery above 0 at $\dmin$ is not guaranteed by the metric and is the
paper's genuine causal finding (sufficiency).
```
**Why:** A10 — this is the derivation FIX-1, FIX-2, FIX-3, and FIX-4 all
point to. Independently re-derived and verified (see §2 above); consistent
with the design record's own "oracle upper bound" language. Placed in the
appendix because it is not page-limited; main text carries only the
one-clause pointer + numeric range.

#### FIX-9: New appendix paragraph — M1 band's non-binding upper half (proof)
**Severity:** SERIOUS
**File(s):** `sections/07_appendix.tex`
**Location:** Immediately before FIX-8's paragraph (same insertion point,
i.e. before FIX-8's text in final document order — insert this paragraph
first)
**After:** (new paragraph)
```
\textbf{The observational band's non-binding upper half.}
\texttt{entity\_subspace\_from\_words} returns the top-$\dmin$ left
singular vectors of the centered covariance by construction
(\texttt{U\_full[:, :d\_min]}), so
$\mathrm{restrict}(Z,U) = U^{\top} Z U$ is
always $\dmin \times \dmin$, and \texttt{effective\_rank} on a
$\dmin \times \dmin$ input has documented range $[1,\dmin]$.
\texttt{restricted\_effective\_rank} therefore cannot exceed $\dmin$,
and so cannot exceed $1.3\cdot\dmin$, for any seed of any group; only
the band's lower half, $\geq 0.7\cdot\dmin$, is a test the data could
fail. Every group mean lands in $[0.895, 0.951]\cdot\dmin$.
```
**Why:** A2 — backs FIX-6's main-text pointer with the code-level proof
(exact source paths: `matrix-thinking/capability_separation/readout.py`
lines ~93-99, `matrix-thinking/chapter2/rank_utils.py` line ~47).

#### FIX-10: New appendix paragraph — primary vs. crosscheck degauging
**Severity:** SERIOUS
**File(s):** `sections/07_appendix.tex`
**Location:** Immediately after FIX-8's paragraph, still before
`\section{Per-Seed Observational Rank}`
**After:** (new paragraph)
```
\textbf{Primary and crosscheck degauging.} The design record's default
pipeline treats scale-only degauging ($\hat Q = I$) as the primary
metric, with the fitted-$(c,Q)$ Procrustes score retained as a
robustness cross-check, after gate-1 calibration on unconstrained
checkpoints found $\hat Q \approx I$ empirically (fitting a full
rotation adds no recovery beyond fitting scale alone off this grid).
Under the force-rank grid's zero-padded target this equivalence breaks:
the $\dmin$-dimensional informative block's degenerate singular
spectrum (all singular values equal to 1) makes the scale-only fit's
basis arbitrary, so the pre-registration for the causal wave
specifically pins the fitted-$(c,Q)$ score, denoted $\recninety$
throughout Section~\ref{sec:causal} and Figure~\ref{fig:razor}, as
decisional; no conclusion in this paper reads the scale-only score. The
two diverge sharply on this grid (e.g.\ one unconstrained cell:
scale-only recovered-fraction 0.000 vs.\ fitted-$(c,Q)$ 0.500), which is
why the distinction is disclosed here rather than left for a reader to
notice.
```
**Why:** A8, on the corrected factual basis from §1/§2 above: the paper
never states a "primary" metric to contradict, so this is a disclosure gap
(what "crosscheck" means, why it and not a simpler score is decisional
here), not evidence of metric-shopping. Appendix-only; the main text's use
of the word "crosscheck" (Figure 2 caption, already present) is now backed
by a definition somewhere in the paper.

#### FIX-11: New appendix section — $S_3$ per-seed causal detail (table)
**Severity:** SERIOUS
**File(s):** `sections/07_appendix.tex`
**Location:** New `\section{$S_3$ Per-Seed Causal Detail}` /
`\label{app:s3causal}`, inserted after FIX-8/FIX-9/FIX-10's paragraphs and
before `\section{Per-Seed Observational Rank}`
**After:** (new section + table)
```
\section{$S_3$ Per-Seed Causal Detail}
\label{app:s3causal}

\begin{table}[H]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
seed & anchor & $k=\dmin{-}1$ & $k=\dmin$ & $k=\dmin{+}1$ & own bar ($0.9\times$anchor) & clears own bar \\
\midrule
0 & 0.550 & 0.000 & 0.450 & 0.550 & 0.495 & no ($-0.045$) \\
1 & 0.600 & 0.000 & 0.550 & 0.750 & 0.540 & yes ($+0.010$) \\
2 & 0.800 & 0.000 & 0.600 & 0.750 & 0.720 & no ($-0.120$) \\
3 & 0.600 & 0.000 & 0.650 & 0.600 & 0.540 & yes ($+0.110$) \\
\bottomrule
\end{tabular}
\caption{$S_3$ four-seed causal detail (crosscheck $\recninety$) behind
the seed-mean confirmation in Section~\ref{sec:causal}. $k=\dmin{-}1$
reads exactly 0.000 in all four seeds; $k=\dmin$ clears each seed's own
$0.9\times$anchor bar in 2 of 4 (seeds 1, 3), missing in the other 2
(seeds 0, 2) because the anchor itself ranges 0.550--0.800 across
seeds. The pre-registered criterion compares the seed-mean $k=\dmin$
value (0.5625) against the fixed bar set before the extension ran
(0.495, from seed 0's anchor), specifically to avoid computing the bar
from the same noisy seeds it is judged against; recomputing the bar
from the four-seed anchor mean (0.574) would put the seed-mean
$k=\dmin$ value 0.011 below it.}
% <!-- evidence: U4 -->
\label{tab:s3seeds}
\end{table}
```
**Why:** A9 — the concrete disclosure FIX-3's main-text pointer
(`Table~\ref{tab:s3seeds}`) references. Verified against
`CAPABILITY_SEPARATION_DESIGN.md:6144-6150,6169-6172` exactly, including
the self-referential-bar sensitivity note (recomputing from the
extension's own anchor mean would put the seed-mean $-0.011$ below its own
bar) — this is disclosed *for* the paper's rigor, not despite it: it is why
the design record used a fixed literal instead of a self-referential
recompute, and the paper should get credit for that discipline, not just
exposure for the raw numbers.

#### FIX-12: Related Work — two MUST-cite additions
**Severity:** SERIOUS
**File(s):** `sections/06_related_limits.tex`, `refs.bib`
**Location:** End of the Related Work paragraph (before "\textbf{Limitations.}")
**Before:**
```
under argmax decoding, the loophole our exact-continuous-recovery
scoring closes; \citet{mishra2026m2rnn} train matrix-state RNNs on $S_3$
composition but measure no state rank and run no rank intervention.
```
**After:**
```
under argmax decoding, the loophole our exact-continuous-recovery
scoring closes; \citet{mishra2026m2rnn} train matrix-state RNNs on $S_3$
composition but measure no state rank and run no rank intervention.
\citet{chughtai2023toy} reverse-engineer which representation-theoretic
structure networks recruit on finite-group composition tasks; this work
asks a narrower question on the same substrate, at what dimension, with
a pre-registered causal intervention absent there.
\citet{huh2024platonic} propose that models trained on different tasks
converge to a shared representation; the present law is a narrower,
algebra-driven instance of that convergence.
```
**`refs.bib` addition (insert before the `barrington1989` entry):**
```
@inproceedings{chughtai2023toy,
  title     = {A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations},
  author    = {Chughtai, Bilal and Chan, Lawrence and Nanda, Neel},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  note      = {arXiv:2302.03025}
}

@inproceedings{huh2024platonic,
  title     = {Position: The Platonic Representation Hypothesis},
  author    = {Huh, Minyoung and Cheung, Brian and Wang, Tongzhou and Isola, Phillip},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  note      = {arXiv:2405.07987}
}
```
**Why:** Both papers independently re-verified (WebFetch on
`arxiv.org/abs/2302.03025`, `arxiv.org/abs/2405.07987`, and
`proceedings.mlr.press/v202/chughtai23a.html`): Chughtai, Chan & Nanda,
ICML 2023, PMLR v202 pp. 6243-6267, arXiv:2302.03025; Huh, Cheung, Wang &
Isola, "Position: The Platonic Representation Hypothesis," ICML 2024
(PMLR v235 pp. 20617-20642), arXiv:2405.07987. Chughtai et al. studies
the identical question (which representation-theoretic structure a
network converges to when trained on finite-group composition, including
symmetric/alternating families) on an overlapping substrate; its absence
is the single gap a UniReps reviewer familiar with the mechanistic-
interpretability literature would flag first. Huh et al.'s title claim is
close enough to this paper's own opening framing ("when systems trained
on different problems arrive at similar internal structure") that not
citing it by name reads as an omission at this venue.

### MINOR

#### FIX-13: Correct "4-8%" to "4-10%" (two locations)
**Severity:** MINOR
**File(s):** `sections/01_intro.tex`, `sections/03_convergence.tex`
**Location 1 (intro, contribution bullet 1):**
**Before:** `mean rank lands within 4--8\% of $\dmin$, all 19 seeds inside a`
**After:** `mean rank lands within 4--10\% of $\dmin$, all 19 seeds inside a`
**Location 2 (§3, headline sentence):**
**Before:**
```
within 4--8\% of the algebraic minimum at every group, with all 19 seeds
inside the pre-registered $[0.7, 1.3] \cdot \dmin$ band.
```
**After:**
```
within 4--10\% of the algebraic minimum at every group (S5 the outlier,
10.2\%), with all 19 seeds
inside the pre-registered $[0.7, 1.3] \cdot \dmin$ band.
```
**Why:** A3 — $(4-3.591)/4 = 0.1023$ for S5, 2.2 points outside the stated
4-8% range; both numbers already carry the existing `U1` evidence tag span
(§3's is inside the pre-existing `U1`-to-`U1` bracket, so no new tag is
needed). Zero interpretive cost, apply first.

## 4. Verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 — anchor unreliable / "restores recovery" | CRITICAL | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-1, FIX-3 |
| A2 — M1 band one-sided | SERIOUS | PARTIAL | DEFENSE VALID BUT EDIT | FIX-6, FIX-9 |
| A3 — "4-8%" contradicts Table 1 | SERIOUS (downgraded to MINOR) | CONCEDE+FIX | DEFENSE INSUFFICIENT → resolved | FIX-13 |
| A4 — budget confound | SERIOUS | PARTIAL | PARTIAL — SURVIVES REDUCED | FIX-7 |
| A5 — "step function" overclaim | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-3, FIX-4 |
| A6 — target definitional | SERIOUS | PARTIAL | PARTIAL — SURVIVES REDUCED (narrower after A10) | FIX-2, FIX-5 |
| A7 — title/solvability single pair | MINOR | PARTIAL | DEFENSE VALID BUT EDIT | FIX-6 |
| A8 — primary/crosscheck switch [defense-surfaced] | SERIOUS (defense) | (self-surfaced) | PARTIAL — SURVIVES REDUCED, corrected basis | FIX-10 |
| A9 — S3 2/4-per-seed disclosure [defense-surfaced] | folds into A1 | (self-surfaced) | DEFENSE VALID BUT EDIT | FIX-3, FIX-11 |
| A10 — necessity forced by geometry [coordinator] | CRITICAL (assigned here) | no formal defense | resolved via fix | FIX-1, FIX-2, FIX-3, FIX-4, FIX-8 |

**Disposition counts:** 4 DEFENSE VALID BUT EDIT (A1, A5, A7, A9) · 3
PARTIAL — ATTACK SURVIVES IN REDUCED FORM (A4, A6, A8) · 1 DEFENSE
INSUFFICIENT resolved via trivial fix (A3) · 1 CRITICAL resolved via
substantive fix with no formal defense round (A10) · 0 DEFENSE VALID
(no attack was a clean, no-change-needed factual disproof — every attack's
factual premise held up, consistent with the defense report's own summary).

## 5. Residual risk after all fixes

- **A4 (budget confound), residual.** The pre-registered clearance process
  narrows but does not eliminate the pattern that $S_3$/$S_5$ are both the
  least-converged and most-anomalous groups. A camera-ready-only
  strengthening (not a submission blocker): rerun $S_3$/$S_5$ at doubled
  step budget and confirm the rank estimates do not move outside the
  reported bands. Workshop-survivable as disclosed; would be
  conference-blocking if a full paper leaned on the five-group trend as
  independently decisive (it does not — the marquee TOST carries that
  weight).
- **A6 (definitional target), residual.** After FIX-2/FIX-5, the paper's
  surviving non-definitional claims are narrower and more precise: (1) the
  model does not recruit rank into its 2 spare ambient dimensions (M1),
  and (2) capacity that cannot represent the target below $d_{\min}$
  (by construction) empirically does represent it at $d_{\min}$
  (sufficiency, genuinely non-tautological). Both are real and
  publishable; neither is "the network discovers its own algebra from
  nothing," which the original framing invited. Workshop-survivable.
- **A1/A9 anchor noise, residual.** Even after the wording and disclosure
  fixes, four of five groups' causal cells are still single-seed against a
  single-seed anchor. This is disclosed (Limitations, and now sharpened by
  FIX-3/FIX-11), not hidden, and the defense's own non-blocking
  recommendation stands: 3-5 anchor seeds per group for camera-ready would
  convert a disclosed weakness into a measured one. Not required for the
  EA.
- **A8, residual.** The appendix now defines "crosscheck" and explains the
  primary/crosscheck pin. A reader who does not open the appendix still
  encounters an undefined-sounding term in the Figure 2 caption; this is
  an acceptable trade at a 4pp EA (the alternative costs main-text budget
  this paper does not have) but is worth one clause of main-text
  visibility if a page ever opens up (e.g. if FIX-13's savings or further
  trimming elsewhere creates room).
- **Page budget, verified not residual.** All 13 fixes were applied to a
  scratch copy and compiled with `tectonic`: the main text (through
  Related Work/Limitations) still ends on page 4 with References, an
  identical boundary to the unedited baseline; the appendix (unlimited)
  absorbed the growth, extending from 1 page to roughly 3. No further
  trimming is required for this to fit, though FIX-12's two citation
  sentences are the most compressible if a future edit needs the room.
- **Not fixed, by design.** The paragraph-1 "Pre-registered reading"
  sentence in §5 ("one below-$d_{\min}$ cell recovering falsifies
  necessity") was left unedited: it accurately describes a criterion fixed
  *before* results were known, and FIX-3 immediately below it now supplies
  the geometric context. Revisit only if a future round finds this
  insufficient on its own.

## 6. Re-run instruction

The gauntlet cannot proceed to the detector gate until FIX-1, FIX-2, and
FIX-3 (the two CRITICAL-affected claim clusters) re-enter attack/defense/
rebuttal. Specifically:

1. **The causal-razor claim cluster** — abstract's third headline sentence
   (FIX-1), Section 5's necessity/sufficiency paragraph and Figure 2
   caption (FIX-3, FIX-4), and the new Appendix derivations (FIX-8, FIX-9,
   FIX-10, FIX-11) — re-attack this cluster as a unit. It is the paper's
   most decisive result and now carries the most rewritten text.
2. **The convergence-framing cluster** — intro's contribution bullet 3 and
   paragraph-1 framing sentence (FIX-2, FIX-5) — check these against the
   re-attacked abstract for consistency; a lighter pass than (1) since the
   underlying claims are unchanged, only the framing.
3. **Light consistency check only** (no full re-attack needed): FIX-6
   (§3 M1 band + solvability clause), FIX-7 (Limitations budget caveat),
   FIX-12 (new citations), FIX-13 (4-8%→4-10%) are self-contained,
   low-risk textual disclosures that do not change any claim's substance.

## 7. New citations

**MUST-cite (see FIX-12 for exact bib entries and sentences):**
- Chughtai, Chan & Nanda, "A Toy Model of Universality: Reverse Engineering
  How Networks Learn Group Operations," ICML 2023 (PMLR v202, pp.
  6243-6267), arXiv:2302.03025.
- Huh, Cheung, Wang & Isola, "Position: The Platonic Representation
  Hypothesis," ICML 2024 (PMLR v235, pp. 20617-20642), arXiv:2405.07987.

**SHOULD-cite (camera-ready deferral, not a submission blocker — not
included in the fix list above given the page budget; add if room opens):**
- Nanda, Chan, Lieberum, Smith & Steinhardt, "Progress Measures for
  Grokking via Mechanistic Interpretability," ICLR 2023, arXiv:2301.05217
  — same phenomenon (training recovers representation-theoretic
  structure), different group family (abelian, cyclic).
- Moschella, Maiorca, Fumero, Norelli, Locatello & Rodolà, "Relative
  Representations Enable Zero-Shot Latent Space Communication," ICLR 2023,
  arXiv:2209.15430 — UniReps-community touchstone, venue-fit citation.
- Li, Yosinski, Clune, Lipson & Hopcroft, "Convergent Learning: Do
  Different Neural Networks Learn the Same Representations?," ICLR 2016
  workshop, arXiv:1511.07543 — historical precedent, one-line acknowledgment.
