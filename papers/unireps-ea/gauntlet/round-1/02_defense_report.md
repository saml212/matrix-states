# Defense Report — UniReps EA "Dimension, Not Solvability" (round 1, stage 02)

Author's-side review, fresh context. I re-derived or re-verified every load-bearing
claim independently against primary sources (not against the attack report's own
numbers): I re-grepped raw JSON fields directly (`crosscheck_recovered_frac_90`,
`crosscheck_mean_cos`, `restricted_effective_rank`, `mean_cos` / `recovered_frac_90`
for the primary metric) for S3, S4, S5's causal cells and for S3's unconstrained
seeds; read `readout.py`'s `entity_subspace_from_words`/`restrict` and
`rank_utils.py`'s `effective_rank` docstring directly; read `groups.py`'s
`rho_G_embedded`; read the design doc's §1.30, §1.33, §1.35, §1.36, §1.36a in full;
and independently confirmed the two disputed citations (Chughtai, Chan & Nanda,
ICML 2023, PMLR v202; Nanda et al., ICLR 2023) resolve to real, on-topic papers via
live lookup. Every factual premise in every attack held up. This is not a report
where I found the attacker overstating things — I found the opposite in one place
(§4, "missed attacks") and no clean disproof anywhere.

## 1. Summary for the rebuttal agent

**0 DEFEND, 3 PARTIAL, 4 CONCEDE+FIX**, out of 7 attacks. Every attack's factual
premise independently verified as true against primary sources (raw JSON, source
code, the design doc, live citation lookups) — there is no attack here that rests
on a misread of the code or a wrong number. The paper is **submittable after fixes,
and every required fix is textual** (rewording, an added caveat sentence, or a
short related-work paragraph) — **no new experiment is required to reach a sound
submission.** The most important fixes, in priority order: (1) A1 — reword
"restoring $d_{\min}$ **restores** recovery" (abstract + §5), since capped cells
beat the single-seed anchor in a majority of groups; add an explicit anchor-noise
caveat inline in §5, not just in Limitations. (2) A3 — the Introduction's "4–8%"
range is arithmetically wrong against the paper's own Table 1 (S5 is 10.2%); fix
the number. (3) A5 — "step function" overclaims a plateau; the necessity half is a
real step, the sufficiency half keeps climbing past $d_{\min}$ in 3/5 groups; rename
to "necessity threshold." (4) A7 — add Chughtai, Chan & Nanda (ICML 2023) by name
to Related Work; it is the closest prior work and its absence is a real gap for a
UniReps submission. (5) A2, A4, A6 — each needs one added caveat sentence (band
upper-bound non-binding by construction; budget-convergence correlation for S3/S5;
scope "converges"/"discovers" language to what was actually measured), none needs
new data. I additionally surface two gaps the attacker did not raise (§4 below):
the paper never explains, anywhere in the main text or appendix, why §5's decisive
metric is the *crosscheck* (full-$Q$) score rather than the *primary* (scale-only)
metric §2 designates as primary — a silent metric switch with a large empirical
effect (§4 item 1); and the S3 seed extension that "confirms the fifth group" on
its seed-mean also reveals that only 2 of 4 individual seeds clear their own
per-seed bar, a fact the paper never discloses (§4 item 2).

## 2. Defenses

### A1: The "unconstrained anchor" is not a reliable ceiling — capped runs beat it outright, undermining "restores recovery"

**Disposition:** CONCEDE + FIX (framing fix, no new evidence required for submission; more anchor seeds is a valuable but non-blocking camera-ready strengthening)

**Response.** I re-pulled the five raw JSON pairs myself (not trusting the attack
report's transcription) and they match exactly: S4 anchor 0.65 vs $k=d_{\min}$ 0.80
vs $k=d_{\min}{+}1$ 0.95; S5 anchor 0.50 vs $k=d_{\min}$ 0.60. The attack's
arithmetic is correct and the pattern — capped cells matching or beating the
"unconstrained" anchor in every group — is real, not a misread.

Where I push back on the *implication*, not the *fact*: the pre-registered decision
criterion (design doc §1.35, "Oracle ceilings for the harvest: $k\geq d_{\min}$
exact (1.0), $k=d_{\min}{-}1$ bounded $\leq 0.894$") was never a claim that the
anchor is a ceiling. The theoretical oracle ceiling for any $k\geq d_{\min}$ cell,
including the anchor, is exact recovery (1.0) — a *rank-$d_{\min}$-capped* run has
exactly enough capacity to hit that ceiling with a tighter, better-conditioned
optimization landscape than the "unconstrained" run's 2 extra ambient dimensions,
so a capped cell beating an n=1 unconstrained draw is consistent with the
pre-registered theory, not a violation of it. The CONFIRM criterion itself is a
one-sided floor test ("$k\geq d_{\min}$ must recover past $0.9\times$ the anchor's
$\recninety$"), not a claim that the anchor is unbeatable — so the *test that was
actually run* is defensible as written.

What is not defensible is the abstract's specific prose: "restoring $d_{\min}$
**restores** recovery" reads as "recovery comes back up to where it was," when the
data show recovery frequently exceeds where it was, by up to +0.30 (S4, a 46%
relative gain). That is a materially different, more interesting, and currently
unexplained pattern than "restoration," and the current wording invites exactly the
misreading the attack describes. This is compounded by the fact that the paper's
own Limitations paragraph already discloses "anchor-relative" causal readings and
n=1 cells in general terms, but never states the specific, systematic version of
this pattern (S4/S5 capped cells beating anchor by wide margins) anywhere a reader
would see it before reaching Limitations.

**Supporting evidence.** Direct re-pull (independent of both the attack report and
the design doc): `experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S4__unconstrained__seed0.json`
`crosscheck_recovered_frac_90=0.65`; `..._S4__k_dmin__seed0.json` `=0.8`;
`..._S4__k_dmin_plus_1__seed0.json` `=0.95`; `..._S5__unconstrained__seed0.json`
`=0.5`; `..._S5__k_dmin__seed0.json` `=0.6`. Design doc §1.35: "Oracle ceilings for
the harvest: $k\geq d_{\min}$ exact (1.0)... k=d_min−1 bounded ≤0.894." Design doc
§1.36a's own secondary disclosure (not in the paper): S3's per-seed own-bar
clearance is only 2/4 (seeds 1, 3 clear; seeds 0, 2 miss by −0.045/−0.120) — the
systematic version of the anchor-noise problem, present in the project's own
records but never surfaced in the draft (see §4 item 2 below).

**What goes in the paper if this defense is accepted.** In the abstract and §5,
replace "restoring $d_{\min}$ restores recovery" with language the data actually
support, e.g. "restoring $d_{\min}$ clears the pre-registered $0.9\times$-anchor
bar in every group — in a majority of groups by a wide margin, since the anchor
itself is a single noisy draw (Limitations)." Add one sentence to §5's body (not
just Limitations) stating explicitly that capped cells exceed the anchor in 3/5
groups and that this is consistent with the anchor being an unbiased-but-noisy
single-seed estimate rather than a hard ceiling. Optionally (camera-ready, not a
submission blocker): run the unconstrained anchor at 3–5 seeds per group to report
a variance next to the bar.

---

### A2: M1's "$[0.7,1.3]\cdot d_{\min}$ band" is a one-sided test dressed as two-sided — the metric cannot exceed $d_{\min}$ by construction

**Disposition:** PARTIAL

**Response.** I read `readout.py` directly. `entity_subspace_from_words` returns
`U_full[:, :d_min]` (line 94) — exactly `d_min` columns, by construction, not by
any data-dependent process. `restrict(Z, U) = U.T @ Z @ U` (line 99) is therefore
always $d_{\min}\times d_{\min}$. `rank_utils.effective_rank`'s docstring (line 47)
states "Range `[1, d]`" for a `d×d` input — for a $d_{\min}\times d_{\min}$ matrix
that is `[1, d_min]`. I additionally pulled the actual numbers: S3's three
`restricted_effective_rank` values are 1.808, 1.905, 1.919 (all $< d_{\min}=2$,
never above). The attack's technical claim is exactly right: the upper half of the
band, $\leq 1.3\cdot d_{\min}$, cannot be violated by any seed of any group, ever —
it is not a live falsifier.

Where this is not a fresh gap: the design doc already carries a permanent
"M1-PREVIEW CAVEAT" (§1.5) stating almost this exact point — "The restriction
machinery... partially FAVORS restricted_eff_rank ≈ $d_{\min}(G)$ by construction
for near-orthogonal blocks, because restricting TO a $d_{\min}$-dimensional
subspace before measuring rank caps what the metric can even see" — and this is
reflected nearly verbatim in the draft's §3: "a $\dmin$-restricted readout partially
favors rank $\approx \dmin$ for near-orthogonal blocks" is the reason given for why
M1 is demoted to corroborating. So the paper does not hide this; it states the
mechanism and demotes M1's evidentiary weight because of it, exactly as the design
record does. What it does not say is the sharper, purely mathematical fact the
attack states — that the upper bound is not merely "partially favored" but
*structurally impossible to violate* — and the abstract's flat statement ("all 19
seeds inside the... band") reads, on its own, like a two-sided test the data
could have failed.

**Supporting evidence.** `matrix-thinking/capability_separation/readout.py:93-99`
(quoted above); `matrix-thinking/chapter2/rank_utils.py:47` ("Range `[1, d]`");
raw values `experiment-runs/2026-07-09_capability_sweep_harvest/results/S3__unconstrained__seed{0,1,2}.json`
→ `restricted_effective_rank` = 1.8082/1.9045/1.9185 (all below $d_{\min}=2$).
Design doc §1.5 M1-PREVIEW CAVEAT (already partially reflected in draft §3 prose).

**What goes in the paper if this defense is accepted.** One added sentence,
either as a footnote to the band definition in §2 or immediately after "all 19
seeds inside... band" in §3: state plainly that `restricted_effective_rank` is
upper-bounded at $d_{\min}$ by the readout's own construction (cite the SVD
truncation), so the band's lower half ($\geq 0.7\cdot d_{\min}$) is the only side
that could have failed, and report that fact next to the existing corroborating-only
framing rather than leaving it implicit.

---

### A3: Introduction's "4–8%" deviation claim is contradicted by the paper's own Table 1

**Disposition:** CONCEDE + FIX (framing fix, trivial)

**Response.** I recomputed $(d_{\min}-\text{mean})/d_{\min}$ from Table 1's own
numbers myself: S3 $(2-1.877)/2=6.15\%$, S4 $(3-2.852)/3=4.93\%$, A5
$(3-2.832)/3=5.60\%$, S5 $(4-3.591)/4=10.22\%$, A6 $(5-4.736)/5=5.28\%$. The
attack's numbers are exactly right. S5 at 10.2% is a full 2.2 points outside the
Introduction's claimed 4–8% range — a plain, indefensible mismatch between the
Introduction's summary sentence and the paper's own Appendix table. There is
nothing to defend here.

**Supporting evidence.** Table 1 (`07_appendix.tex`, `tab:m1`): $S_5$ mean
$3.591\pm0.069$ vs. $d_{\min}=4$. Independently recomputed above.

**What goes in the paper if this defense is accepted.** Change "within 4–8% of
$\dmin$" (§1) to "within 4–10% of $\dmin$" (or "4–8% for four of five groups, up to
10% for S5"). One-word/one-number fix, no other section is affected.

---

### A4: Unequal, non-monotonic per-group training budgets confound the cross-group comparison

**Disposition:** PARTIAL

**Response.** The attack's factual claims are correct: per-group budgets are
S3=8K, S4=20K, A5=20K, S5=8K, A6=40K, and this does not track $d_{\min}$
monotonically. But I checked the design doc's §1.30 (Rev 7 budget pins) — which
the attack did not cite — and the budgets were not picked freely or to favor a
narrative; they were the output of a symmetric, pre-registered per-group
convergence-bar clearance process applied identically to all five groups *before*
the decisive sweep ran: "Budget pins: S3=8K (0.9649), S4=20K (0.9796), A5=20K
(0.9755), S5=8K (0.9213), A6=40K (0.9633...)" — each group escalates only as far
as needed to clear the same bar, and S5's 0.9213 is in fact the thinnest margin of
the five, i.e. S5 sits *closest to its own convergence floor*, consistent with (not
evidence against) the pre-registered process working as designed rather than an
arbitrary confound.

That said, this does not fully dispose of the concern. The attack's own
"alternative explanation" framing over-reaches when it implies the budget
asymmetry could explain the *ordering* — reproducing rank tracking $d_{\min}$
across five different groups via a training-budget artifact would require an
oddly specific coincidence, since budget does not vary monotonically with
$d_{\min}$ the way the observed ranks do. But the narrower version of the concern —
that S3 and S5, the two groups sitting at the un-escalated base 8K budget, are also
the two showing the largest observational deviation (A3) and the marginal causal
cell — is a real, disclosed-but-unconnected pattern in the paper's own appendix
that deserves a sentence linking it explicitly, rather than leaving a reader to
notice the correlation unaided.

**Supporting evidence.** Design doc §1.30 ("MECHANISM FOUND — encoder degeneracy
at L=1... per-group budget pins"), the exact per-group clearance margins quoted
above; §1.36's gate1a disclosure ("S3's four variant-A cells (anchor min_val
0.9143) and S5's anchor/k_dmin/k_dmin+1 (0.876-0.879) sit just under the bar").

**What goes in the paper if this defense is accepted.** One sentence in §2 or the
Limitations paragraph: budgets were set by a per-group, pre-registered
convergence-bar clearance process (not picked post hoc), but the two groups that
stayed at the un-escalated base budget (S3, S5) are also the two showing the
largest observational deviation and the one marginal causal cell — disclosed as a
caveat on cross-group comparison strength, not retracted.

---

### A5: "Step function" language overclaims a plateau at $d_{\min}$ that the raw recovery curve does not show

**Disposition:** CONCEDE + FIX (framing fix)

**Response.** Checked against the same C1 decisional table used for A1: recovery
keeps climbing past $d_{\min}$ in S4 (0.800→0.950), A5 (0.700→0.750), and A6
(0.650→0.700); only S5 dips slightly (0.600→0.550); S3 is flat
(0.450→0.550→ties the anchor). A "step function" implies a hard floor *and* a
plateau; the necessity half (floor) is exact and true in all five groups with zero
exceptions — that part of the metaphor is earned. The sufficiency half is not a
plateau in 3 of 5 groups; it is a monotone climb with a floor below $d_{\min}$.
"Step function" overclaims the sufficiency side specifically, and the figure
caption states it flatly ("Figure~\ref{fig:razor} shows a step function at
$\dmin$"), which a reader would reasonably expect to mean recovery flattens out at
$d_{\min}$.

**Supporting evidence.** The same C1 table verified for A1 (design doc §1.36 and
independently re-pulled JSONs).

**What goes in the paper if this defense is accepted.** Replace "step function" in
§5's body text and the Figure 2 caption with "necessity threshold" or "onset of
recovery," and add one clause noting recovery continues to rise through
$k=d_{\min}+1$ in most groups — this is consistent with, and does not weaken, the
necessity claim, since necessity only requires the floor to be exact (which it is).

---

### A6: The training target is directly supervised at rank $d_{\min}$ — "convergence" is partly definitional, not emergent

**Disposition:** PARTIAL

**Response.** Read `groups.py`'s `rho_G_embedded` directly: the target is
`out = eye or zeros; out[:d_min, :d_min] = rho` — literally block-diagonal with
the group's own $d_{\min}$-dimensional faithful representation in the top-left
block and a constant pad elsewhere. The attack's premise is exactly right: the
loss's informative content is $d_{\min}$-dimensional by construction, so *some*
minimal notion of "$d_{\min}$ suffices" is baked into the setup before training
starts, not discovered by it.

But the attack overstates how much this empties the paper's actual claims. Two
things remain genuinely non-trivial and are not definitional: (1) the model has
$d_{\mathrm{state}} = d_{\min}+2$ available — 2 more dimensions than the target
needs — and nothing in the architecture or loss explicitly forces the *learned
state's own effective rank* down to $d_{\min}$; a network could in principle
recruit rank into those 2 spare ambient dimensions (via optimization noise,
imperfect convergence, or degenerate solutions) and M1 shows it largely does not,
across five structurally different groups. (2) The causal necessity result
($k=d_{\min}-1$ yields exactly 0.000 $\recninety$, not just low $\recninety$, in
every group and all four independent S3 seeds) is an intervention on the *model's*
representable capacity, and its cleanness (a hard, noiseless zero, not a curve)
is an empirical fact about training dynamics under a rank cap, not a tautology of
the loss definition. Where the attack is right is that the *intro's rhetorical
question* — "what, exactly, converges... If trained models converge to the task's
algebra" — reads as though the network discovers an unknown quantity, when the
quantity ($d_{\min}$) was already encoded in the supervision target's own shape.
That framing oversells what was actually tested.

**Supporting evidence.** `matrix-thinking/capability_separation/groups.py:109-131`
(`rho_G_embedded`, quoted above); design doc §1.33 P1 ("the as-built target =
`rho ⊕ I_2`").

**What goes in the paper if this defense is accepted.** Scope the intro/abstract
language: replace "if trained models converge to the task's algebra" with something
closer to "does the model's own recruited capacity settle at $d_{\min}$ despite 2
spare ambient dimensions, and is that capacity load-bearing" — the actually-tested,
still-publishable claim. No experiment required for the EA; a genuinely stronger
(camera-ready-only) control would use a non-minimal but still-faithful target
embedding to show the network still contracts to $d_{\min}$ on its own, but this is
a strengthening, not a submission requirement.

---

### A7: The title's "Not Solvability" half rests on a single group-pair contrast, not the "five groups" the abstract implies

**Disposition:** PARTIAL

**Response.** Checked Table 1 / Appendix A: solvable = {S3 ($d_{\min}=2$), S4 (3)},
non-solvable = {A5 (3), S5 (4), A6 (5)} — confirmed, $d_{\min}$ and solvability are
confounded in this 5-group family outside the engineered S4/A5 tie. The attacker
itself rates this MINOR because §3 already discloses M1 is "corroborating rather
than independently decisive" — I agree the paper is already self-aware that the
5-group correlational trend does not itself dissociate the two axes, and that the
S4/A5 TOST is what actually carries the dissociation claim. Given that awareness
already exists in the body, this is a title/abstract precision issue, not a fresh
methodological hole.

**Supporting evidence.** Table 1 / `tab:groups` (`07_appendix.tex`).

**What goes in the paper if this defense is accepted.** One clarifying clause in
the abstract or intro: "...the five-group trend is correlational; the dissociation
between dimension and solvability is carried specifically by the $S_4$/$A_5$
pair." No structural change to the title needed if this clause is added.

---

## 3. New citations found during defense

Independently verified all five, two by live fetch (not taken on the attack
report's word):

- **Chughtai, Chan & Nanda, "A Toy Model of Universality: Reverse Engineering How
  Networks Learn Group Operations," ICML 2023** (PMLR v202, confirmed via
  `proceedings.mlr.press/v202/chughtai23a.html`; arXiv:2302.03025). **MUST cite.**
  This is not a tangential related-work item — it studies the identical question
  ("which representation-theoretic structure does a network converge to when
  trained on a finite group's composition") on a substrate (finite groups,
  including symmetric/alternating families) close enough to this paper's own that
  a UniReps reviewer familiar with the mechanistic-interpretability literature will
  flag its absence immediately.
- **Nanda, Chan, Lieberum, Smith & Steinhardt, "Progress Measures for Grokking via
  Mechanistic Interpretability," ICLR 2023** (confirmed via live fetch;
  arXiv:2301.05217). **SHOULD cite**, not MUST — same broad phenomenon (training
  recovers representation-theoretic structure, Fourier features for
  $\mathbb{Z}/p$), but a different group family (abelian, single group) and a
  different research question (progress measures across training time, not
  cross-group convergence to $d_{\min}$). One sentence suffices.
- **Huh, Cheung, Wang & Isola, "Position: The Platonic Representation Hypothesis,"
  ICML 2024** (arXiv:2405.07987). **MUST cite** — the draft's own opening line
  ("when systems trained on different problems arrive at similar internal
  structure") is close enough to this paper's title claim that failing to name it
  reads as an omission, not an independent framing, especially at a UniReps venue
  where this paper is a foundational reference point.
- **Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023** (arXiv:2209.15430). **SHOULD cite** for venue fit
  (UniReps community touchstone) even though the mechanism (relative geometry
  across independently-trained models) differs from this paper's algebraic
  convergence story.
- **Li, Yosinski, Clune, Lipson & Hopcroft, "Convergent Learning," ICLR 2016
  workshop** (arXiv:1511.07543). **SHOULD cite**, one-line historical
  acknowledgment only.

## 4. Attacks the attacker missed

1. **A silent, unexplained metric switch between §2's stated primary and §5's
   actually-reported numbers.** §2 states plainly: "scale-only ($\hat c$) is
   PRIMARY... full-$Q$ Procrustes... is retained and reported as a cross-check, not
   dropped." But every number in §5/Table 2/Figure 2 is the *crosscheck*
   ($\recninety$ in the figure caption is explicitly labeled "crosscheck"), and the
   main text body never says so or explains why. I pulled the raw numbers myself:
   for `S4__unconstrained__seed0`, the primary metric reads `mean_cos=0.467,
   recovered_frac_90=0.05` while the crosscheck reads `mean_cos=0.921,
   recovered_frac_90=0.65` — a huge divergence. The design doc explains this fully
   (§1.35's "C1 metric pin": under zero-padding, the rho-block's degenerate
   eigenvalues make the scale-only primary's basis-fitting arbitrary, so the
   causal wave pre-registers the crosscheck as decisional *before* launch) — but
   none of that appears anywhere in the paper, not even the appendix's instrument
   section. As written, a careful reader who read §2's "primary" claim and then
   §5's numbers with no explanation has grounds to suspect post-hoc
   metric-shopping, even though the actual justification is sound and
   pre-registered. **Rating: SERIOUS** — this is a transparency gap on the causal
   leg, the paper's most decisive result. **Fix:** one paragraph in Appendix
   §app:instrument (or a footnote at the top of §5) stating the C1 pin and why the
   zero-padded causal wave's decisive metric is the crosscheck, not primary. No
   new experiment required — the design doc's own justification suffices, ported
   into the paper.
2. **The S3 seed extension's per-seed bar-clearance rate is not disclosed.** §5
   reports S3's seed-mean (0.5625) clears the fixed bar (0.495) and calls this
   "confirms the fifth group." What it does not say: individually, only 2 of the 4
   seeds clear their *own* per-seed $0.9\times$anchor bar (seeds 1, 3: yes; seeds
   0, 2: no, by −0.045 and −0.120 respectively) — disclosed in the design doc
   (§1.36a) but never in the paper. This is the concrete, quantified version of
   A1's anchor-noise concern, for the one group where enough seeds exist to see it
   directly. **Rating: folds into A1** — I'd fix it in the same edit pass as A1's
   fix (the added anchor-noise sentence in §5 should cite this concrete 2/4 number
   for S3 as the evidence, not just gesture at "anchor noise" abstractly).

## 5. Attack ordering note

- **A1 (CRITICAL):** Rating agreed. It is the abstract's third headline claim, and
  the fix — while narrow (wording, not data) — is not optional; as currently
  worded the claim is not supported by the data shown in the paper's own figure.
- **A2 (SERIOUS):** Slightly over-rated relative to A1. The draft already carries
  substantial disclosure (the "partially favors rank ≈ $d_{\min}$... corroborating"
  language in §3 is a near-verbatim echo of the design doc's own permanent caveat)
  — the fix is a sharper, more explicit statement of a fact the paper already
  gestures at, not a retraction. I'd put this at the SERIOUS/MODERATE boundary.
- **A3 (SERIOUS):** Over-rated for what it costs to fix — a one-clause number
  correction that touches no other claim in the paper. The *embarrassment* of a
  reviewer catching a plain arithmetic contradiction in the Introduction against
  the paper's own table is real, which justifies flagging it prominently, but the
  fix itself is trivial; I'd call this MINOR-to-MODERATE severity with a
  HIGH-priority (fix-first, it's free) label.
- **A4 (SERIOUS):** Rating agreed, but I found a mitigating fact the attacker did
  not have (the principled §1.30 convergence-bar-based budget-pinning process) that
  meaningfully narrows the concern from "arbitrary confound" to "a disclosed,
  bar-cleared budget schedule that happens to correlate with which groups converged
  least cleanly" — still worth a caveat sentence, not a redesign.
- **A5 (SERIOUS):** Rating agreed, clean catch, easy fix.
- **A6 (SERIOUS):** Rating agreed; I'd flag this as the one attack that comes
  closest to touching the paper's *title* and opening framing directly (more than
  A7, which the attacker rated MINOR for a similar title-precision reason), so if
  anything this is the SERIOUS attack I'd promote first for editorial attention,
  even though the fix remains a scoping edit rather than new evidence.
- **A7 (MINOR):** Rating agreed.
- **New (missed) items:** the primary/crosscheck metric switch (§4 item 1) I'd
  rate SERIOUS — it's a transparency issue on the paper's most decisive leg,
  comparable in kind to A2/A4 even though the attacker did not surface it; the S3
  per-seed disclosure gap (§4 item 2) I'd fold into A1 rather than treat as
  independently severity-rated.
