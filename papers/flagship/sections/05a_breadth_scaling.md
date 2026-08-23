# 6 Breadth Scaling

<!-- DRAFT, flagship consolidation 2026-08-23. Sources of record: the raw
archives under experiment-runs/2026-08-21_* and 2026-08-22_*; the
recomputed findings notes pebble-ai-site/findings/ncr-breadth-scaling.html
and ncr-depth-robustness.html (their raw-recomputed values govern wherever
they differ from EXPERIMENT_LOG prose); EXPERIMENT_LOG.md 2026-08-21 #1
through 2026-08-22 #9. Novelty gate: research/kscaling-novelty-2026-08-21.md
(3/3 legs clear). Intended position: after Section 5. -->

Section 4 measures a capability separation at one binding load. Two
questions follow, and this paper now answers both. Does the separation
survive more bindings, and does it survive more parameters? This chapter
answers the first; Section 7 answers the second. The substrate changes:
instead of the 14M two-block contender, the measurements below graft a
small composition head onto a 98M-parameter delta-rule language model,
so that what is being asked is whether a real language model can carry
an in-context-written relation operator and compose it exactly.

## 6.1 The Grafted Composition Head and Its Two Write Regimes

The backbone is a DeltaNet-family causal language model
(DeltaNet and Gated DeltaNet (Yang et al.)) with $d_{model}=768$, 12
layers, $d_{state}=64$, and 97,619,712 backbone parameters. Grafted onto
it is a composition head that holds a single $d \times d$ relation
operator $O$ written from the context. $K$ entities are arranged in one
Hamiltonian $K$-cycle; the answer to a query at hop depth $h$ is the
image of the queried entity under $O^{h}$, which the read path computes
by repeated squaring in $O(\log h)$ matrix multiplies rather than $h$
sequential applications. The operator width is $d = K{+}1$ at every
point, so the axis of this chapter is the *pair* $(K, d)$ and never $K$
alone.

Two write regimes are measured at every point, through the identical
read path and the identical trained weights.

- **P1b, exact teacher-forced operator substitution.** The operator is
  computed in closed form from the true binding and substituted into the
  trained checkpoint's own read path. This is a read-path capability
  measurement and never a learned write; every capability number in this
  chapter and the next is P1b.
- **P0, the model's own SGD-learned write,** read through the identical
  path. This is the wall.

The metric is argmax retrieval over exactly the $K$ episode slots at the
answer position, reported chance-corrected as
$\kappa = (\mathrm{acc} - 1/K)/(1 - 1/K)$ so that readings at different
$K$ are comparable; chance is $1/K$. All curves of record use $n=256$
evaluation episodes at eval seed 90210, on checkpoints at 20,000 steps.

Read depths follow a six-rung ladder derived per $K$ by one mechanical
rule. Because the answer depends only on $h \bmod K$, a ladder chosen
without reference to residue structure silently re-asks in-distribution
questions at nominally deep $h$; every rung's residue is therefore
checked against the training hops $\{1,2,3\}$ before the ladder is
accepted. The top rung is the smallest $h$ in $[32,63]$ with
$h \equiv K/2 \pmod K$: the antipodal residue, the position furthest
from any trained hop, evaluated at a squaring count (five) that the rule
holds fixed across the entire curve. A second read at a fixed effective
distance is carried alongside as a control, so that a change in $\kappa$
across $K$ can be separated from a change in the depth of the read.

Two training recipes are run at every $K$: the entity adapter frozen at
initialization, or trained alongside the backbone. Both use the same
contrastive-plus-cosine auxiliary objective; the adapter is the only
axis varied. Three seeds per $(K, \mathrm{recipe})$.

## 6.2 Protocol: Pre-Registration, Adversarial Audit, and Independent Recomputation

The two chapters share one protocol, described once here.

**Bands before data.** Every wave's decision bands, thresholds, and test
statistic are written into a dated record before the corresponding cells
run, and verdicts are read strictly against those bands. The capability
band is CAPABILITY-HOLDS $= \kappa \geq 0.90$ at the antipodal top rung
on at least 2 of 3 frozen seeds. The ordering threshold below,
$T \geq 53$ of 72, was enumerated together with its exact two-sided $p$
at threshold (0.009868) before any eleven-squaring cell was scored.

**Adversarial review before compute.** Designs pass independent
adversarial rounds whose depth scales with cost: one audit round for
eval-only work, an audit plus a separate resource red-team for the
breadth wave, and a five-round design gauntlet (draft, attack, revision,
verify, revision) for the parameter wave of Section 7, formally closed
only when every FATAL-class finding had been discharged. What those
rounds caught before launch, stated as a record rather than a claim:
two launch-losing defects self-caught by the builder before any audit
round (a sequence-length assertion that fails at the two smallest $K$,
and an evaluation index hard-coded to one $K$'s hop label that would
have raised at the first evaluation of every cell); one further FATAL
found independently by both the audit and the red-team (an allow-list
bug in the battery scorer); four FATAL-class design defects across the
parameter wave's two adversarial rounds, one of them created by the fix
to an earlier one, and one of them a pre-registered target outcome that
was arithmetically unreachable at 15 of 16 cells; and, at wave launch, a
flat per-step cost ceiling that would have aborted every $K \geq 32$
cell mid-run. Both independent build audits read zero FATAL.

**Re-measure clauses.** A single-seed reading that exceeds a band is not
a finding. The clause, written before the runs, sends it back for
re-measurement at an independent evaluation draw (seed 31337);
reproduction promotes it to a real reading, non-reproduction records it
as an outlier and the verdict stands. Section 6.4's toehold survived
this clause and four other excursions did not.

**Mandatory attribution.** No scale-degradation claim was permitted to
publish before a token-budget attribution arm had run, because the token
budget is held fixed across scales and can manufacture a degradation.
That clause is what withdrew one of our own results (Section 7.5).

**Independent recomputation before publication.** Every headline number
is recomputed directly from the raw per-cell records by a role separate
from the one that wrote the prose. Four rounds of corrections resulted;
none changed a verdict, and each is carried on the published note rather
than silently absorbed. A control's floor was overstated (the
fixed-distance control reads $\kappa \geq 0.97$ in 35 of 36 cells of the
first wave, not all 36; the floor is 0.9470). Two re-measured values had
been transcribed from the wrong hop of the right file
($K{=}20$ at $h{=}1$ re-measures 0.0664, $K{=}24$ at $h{=}52$
re-measures 0.0430; both inside band on either reading). A cell count
had double-counted: 42 trained cells plus 6 eval-only anchor re-scores,
not 48 trained. A drift-slope extrapolation predicting a wall breach
near $K \approx 48$ was retracted outright, because that slope came from
a single endpoint segment while global fits over the whole curve read
$-0.0146$ to $-0.0162$ and support no breach-$K$ prediction at all. In
Section 7's chapter a direction-of-effect tally was corrected from 1/5/6
to 1 helped, 4 flat, 7 hurt under the aggregator's own $\pm 0.01$ rule,
and a wall statement was corrected from a cell-level "all 12" to a
reading-level 118 of 120. Compute ledgers were corrected on the same
basis: a projected 69 GPU-h became 81.415 measured, and an extension
total of 88.424 became a marginal 44.493 once parent spend was no longer
re-billed.

**The instrument catch.** Before either wave, the evaluation harness
drew its entity pool from a hard-coded seed while reseeded checkpoints
had trained on their own pools. Under that defective instrument the
frozen-versus-trainable comparison read as a *complete* separation: all
754 of 754 pairs ordered, $p = 1.34 \times 10^{-10}$. Under matched
pools the same checkpoints read 596 of 754 with 216 violating pairs and
a median gap of $+0.0098$, a thirty-two-fold shrink. The two trainable
quadrants move from 0.7266 to 0.9922 and from 0.2793 to 0.9785 in median
recovery, while the two frozen quadrants move by at most 0.0039: frozen
adapters are pool-agnostic and trained adapters had specialized to their
training entities, which the defective instrument scored as a capability
difference. The separation was retracted internally and never published.
Every ordering number in Section 6.5 is a matched-pool re-score.

None of this is offered as evidence that the readings below are correct.
It is offered as the reason they are reported at the precision they are:
each has passed a band fixed before the data existed, an adversarial
round before the compute was spent, and a recomputation by someone other
than its author.

## 6.3 Read-Path Capability Is Flat at Ceiling from (12,13) to (40,41)

Forty-eight cells of record carry this curve: 42 freshly trained (six
calibration cells at $K{=}32$, 24 in the main sweep, 12 at the frontier)
plus six eval-only re-scores of the $K{=}24$ anchor onto this ladder.

**Table 2 caption.** Read-path capability across breadth at 98M
parameters, under P1b (exact teacher-forced operator substitution), at
each $K$'s antipodal top rung. $\kappa$ is chance-corrected retrieval
over $K$ slots; $n{=}256$, eval seed 90210, checkpoints at 20,000 steps;
$d = K{+}1$ throughout. The last two columns give the fixed-distance
control's hop and its minimum $\kappa$ over the six cells at that $K$.
$K{=}24$ is the eval-only anchor. The curve ends at $K{=}40$ by
construction, not by measurement: the antipodal probe requires
$3K/2 \leq 63$ to exist inside the matched five-squaring band, so no
reading here licenses any claim about $K \geq 44$ in either direction
(Section 6.7). Archives: `experiment-runs/2026-08-22_kscaling_wave0/`
(the $K{=}32$ calibration sextet),
`experiment-runs/2026-08-22_kscaling_sweep/` ($K{=}12$ to $K{=}32$ and
the anchor re-scores), `experiment-runs/2026-08-22_kscaling_frontier/`
($K{=}36$, $K{=}40$).

| pair $(K, d)$ | top rung $h$ (residue) | $\kappa$ frozen, 3 seeds | $\kappa$ trainable, 3 seeds | control $h_{\mathrm{fix}}$ | control $\min \kappa$ |
|---|---|---|---|---|---|
| (12, 13) | 42 (6) | 1.0000 / 0.9957 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 40 | 0.9830 |
| (16, 17) | 40 (8) | 1.0000 / 1.0000 / 0.9958 | 0.9958 / 0.9708 / 1.0000 | 36 | 0.9958 |
| (20, 21) | 50 (10) | 1.0000 / 0.9959 / 0.9959 | 1.0000 / 0.9918 / 0.9959 | 44 | 0.9794 |
| (24, 25) | 36 (12) | 1.0000 / 0.9959 / 1.0000 | 0.9878 / 0.9878 / 1.0000 | 52 | 0.9470 |
| (28, 29) | 42 (14) | 0.9959 / 1.0000 / 0.9959 | 0.9959 / 1.0000 / 0.9959 | 32 | 0.9878 |
| (32, 33) | 48 (16) | 0.9960 / 0.9798 / 1.0000 | 0.9919 / 0.9960 / 0.9919 | 36 | 0.9879 |
| (36, 37) | 54 (18) | 1.0000 / 0.9879 / 0.9960 | 1.0000 / 0.9839 / 0.9960 | 40 | 0.9839 |
| (40, 41) | 60 (20) | 0.9920 / 0.9960 / 0.9920 | 0.9880 / 0.9800 / 1.0000 | 44 | 0.9720 |

Every $K$ clears the pre-registered CAPABILITY-HOLDS band on 3 of 3
frozen seeds, not the 2 of 3 the band requires. The floor across all 48
cells is $\kappa = 0.9708$, at $K{=}16$ in a trainable seed; frozen
medians run 0.9920 to 1.0000. The curve does not decline: the two
frontier points, added specifically to look for a frontier, read frozen
medians 0.9960 at $K{=}36$ and 0.9920 at $K{=}40$, with 12 of 12 cells
at ceiling. The fixed-distance control reads $\kappa \geq 0.97$ in 47 of
48 cells with a floor of 0.9470 at one anchor cell, so the flatness is
not an artifact of the top rung's particular depth.

The calibration gate ran alone before the rest of the curve was
licensed. At $K{=}32$ it moved cross-entropy from 11.037 to 4.528 and
read in-distribution $\kappa = 1.000$ at $h \in \{1,2,3\}$ with deep
$\kappa = 0.9960$ at $h{=}48$; the license to spend the remaining cells
was granted on that measurement rather than assumed from the anchor.

Over a 3.3-fold range of binding load there is no measurable capability
slope. Given an exactly written operator, the read path composes it
through the antipodal depth at every breadth tested.

<!-- FIGURE SLOT (not generated in this pass): capability-vs-breadth
curve, kappa at the antipodal top rung against K, frozen and trainable
per-seed points, the 0.90 band drawn, and the K=44 construction limit
marked as a vertical boundary rather than a data point. Source:
experiment-runs/2026-08-22_kscaling_{wave0,sweep,frontier}/. -->

## 6.4 The Learned Write Is at Chance at Every Breadth, With One Toehold at K=12

The same checkpoints, read through the same path with the model's own
learned operator, give the complementary picture. The band is chance
$\pm 3$ binomial standard deviations at $n{=}256$, computed per $K$; a
breach requires an above-band reading replicated across at least two
seeds, and a single-seed excursion goes to the re-measure clause.

**Table 3 caption.** The learned-write wall across breadth at 98M
parameters, under P0 (the model's own SGD-learned write, read through
the identical path). Band top is chance plus three binomial standard
deviations at $n{=}256$. The fourth column pools the six cells at each
$K$ (two recipes, three seeds) at $h{=}1$, the shallowest and easiest
hop; the last column counts readings above band across the six deep-hop
rungs in all six cells. Archives as Table 2.

| pair $(K, d)$ | chance $1/K$ | band top | P0 at $h{=}1$, min to max (median) | cells above band | deep-hop readings above band |
|---|---|---|---|---|---|
| (12, 13) | 0.0833 | 0.1352 | 0.1211 to 0.1484 (0.1387) | 5 of 6 | 0 of 42 |
| (16, 17) | 0.0625 | 0.1079 | 0.0703 to 0.1250 (0.0957) | 1 of 6 | 0 of 42 |
| (20, 21) | 0.0500 | 0.0909 | 0.0430 to 0.1094 (0.0605) | 1 of 6 | 0 of 42 |
| (24, 25) | 0.0417 | 0.0791 | 0.0273 to 0.0703 (0.0488) | 0 of 6 | 1 of 42 |
| (28, 29) | 0.0357 | 0.0705 | 0.0273 to 0.0508 (0.0371) | 0 of 6 | 0 of 42 |
| (32, 33) | 0.0312 | 0.0639 | 0.0156 to 0.0312 (0.0254) | 0 of 6 | 0 of 42 |
| (36, 37) | 0.0278 | 0.0586 | 0.0156 to 0.0469 (0.0293) | 0 of 6 | 0 of 42 |
| (40, 41) | 0.0250 | 0.0543 | 0.0117 to 0.0273 (0.0215) | 0 of 6 | 1 of 42 |

The median chance-corrected reading at $h{=}1$ decays across the curve:
0.0604, 0.0354, 0.0111, 0.0075, 0.0014, $-0.0060$, $+0.0016$, $-0.0036$
at $K{=}12$ through $K{=}40$. Across the whole curve 334 of 336
deep-hop readings sit inside band, and both exceptions are single cells
that re-measure inside. The verdict is WALL-BREACHED-AT-$K{=}12$ and
WALL-HOLDS at $K \geq 16$.

The $K{=}12$ breach is reported in full because it is the only one, and
its structure is what makes it uninteresting as a capability. The six
cells read 0.1211 to 0.1484 at $h{=}1$, median 0.1387, against chance
0.0833: $\kappa$ between 0.041 and 0.071, roughly 3.2 binomial standard
deviations, about fourteen extra correct answers in 256. It reproduces
at an independent evaluation draw, where both re-measured cells read
0.1406, so it is real. It is also confined to $h{=}1$, the shallowest
hop and one of the three depths the model trains on: at $K{=}12$, zero
of 42 deep-hop readings clear the band. A one-hop advantage at a trained
depth that vanishes at every untrained depth is a toehold, not partial
competence, and it does not compose.

Four further single-cell excursions occurred and all four failed the
re-measure clause: $K{=}16$ at $h{=}1$, 0.1250 falling to 0.0977;
$K{=}20$ at $h{=}1$, 0.1094 falling to 0.0664; $K{=}24$ at $h{=}52$,
0.0859 falling to 0.0430; and $K{=}40$ at $h{=}4$, 0.0586 against a band
top of 0.0543, whose re-measured cell reads 0.0156 at $h{=}4$ and 0.0391
as its maximum across all ten hops.

Read together with Table 2, the dissociation is between reading and
writing, not between shallow and deep. The same weights that compose a
supplied operator through the antipodal depth at every breadth cannot
produce that operator at any breadth.

## 6.5 Freezing Buys Depth Robustness, and the Advantage Grows with Breadth

Whether the entity adapter is frozen or trained turns out not to matter
for the capability of Section 6.3 and to matter a great deal for how
that capability survives read depth.

The statistic is a stratified within-$K$ exact permutation count. At
each $K$, $U_K$ is the number of the nine frozen-versus-trainable seed
pairs in which the frozen cell reads higher, ties counted as one half;
$T = \sum_K U_K$ out of nine times the number of strata. Stratifying
within $K$ rather than pooling is deliberate: the pooled form mixes
strata whose $\kappa$ ceilings differ, and it was replaced during the
design gauntlet for that reason.

At the antipodal top rung, five squarings, the ordering is negligible:
$T = 43.0$ of 72 across the eight strata against a threshold of 53
($T = 32.0$ of 54 across the six strata of the first wave). The verdict
is ORDERING-NEGLIGIBLE, and it is what a reader looking only at Table 2
would expect, since both recipes sit at ceiling there.

Extending the read to eleven squarings makes the effect appear.

**Table 4 caption.** The freeze ordering and the depth cost across
breadth, at 98M parameters, under P1b. Columns 2 to 4 give median
$\kappa$ per recipe at eleven squarings and their gap; $U_K$ counts the
nine within-$K$ frozen-versus-trainable seed pairs the frozen cell wins
(ties one half). The last two columns give the depth cost, the change in
retrieval accuracy between five and eleven squarings, per recipe. All
readings are matched-pool re-scores (Section 6.2). Archive:
`experiment-runs/2026-08-22_depthext_across_k/`.

| $K$ | frozen $\kappa$ | trainable $\kappa$ | gap | $U_K$ (of 9) | depth cost, frozen | depth cost, trainable |
|---|---|---|---|---|---|---|
| 12 | 0.9744 | 0.9744 | $+0.0000$ | 6.0 | $-0.0195$ | $-0.0195$ |
| 16 | 0.9667 | 0.9417 | $+0.0250$ | 6.5 | $-0.0312$ | $-0.0547$ |
| 20 | 0.9465 | 0.9342 | $+0.0123$ | 7.0 | $-0.0469$ | $-0.0625$ |
| 24 | 0.9755 | 0.9307 | $+0.0448$ | 9.0 | $-0.0234$ | $-0.0625$ |
| 28 | 0.9797 | 0.9473 | $+0.0324$ | 9.0 | $-0.0195$ | $-0.0469$ |
| 32 | 0.9637 | 0.9234 | $+0.0403$ | 6.0 | $-0.0312$ | $-0.0742$ |
| 36 | 0.9719 | 0.9036 | $+0.0683$ | 9.0 | $-0.0273$ | $-0.0820$ |
| 40 | 0.9599 | 0.8758 | $+0.0841$ | 9.0 | $-0.0391$ | $-0.0938$ |

The eight-stratum statistic reads $T = 61.5$ of 72 against the pre-data
threshold of 53, two-sided exact $p = 3.071 \times 10^{-5}$, and the
median gap is $+0.0364$. The verdict is ORDERING-ROBUST-CONFIRMED.
Dropping any single $K$ leaves the remaining seven strata between 52.5
and 55.5 of 63, which rescales to 60.0 to 63.4 of 72: the threshold is
cleared on all eight leave-one-stratum-out subsets. Restricted to the
six strata of the wave that preceded the frontier extension, the same
eleven-squaring statistic reads 43.5 of 54, reproducing the number that
pass had reported.

The result is graded, not a separation, and the paper states it that
way. The two frontier strata are perfectly ordered, 18 of 18 pairs at
$K{=}36$ and $K{=}40$, as are $K{=}24$ and $K{=}28$; but four of the
eight strata, $K \in \{12, 16, 20, 32\}$, still contain violating pairs.
Nor is the statistic monotone in depth: across five, seven, nine, and
eleven squarings it reads 48.0, 57.0, 65.0, 61.5, and the dip at eleven
traces to a single frozen $K{=}32$ cell reading $\kappa = 0.8669$, below
all three trainable cells in its stratum. That non-monotonicity is
reported rather than smoothed.

The depth cost carries the breadth interaction. The pre-registered band
DRIFT-K-INDEPENDENT compares each $K$'s drift against the $K{=}24$
reference; the largest deviation anywhere on the curve is 0.0430, inside
the $\pm 0.05$ band, so the band passes at all eight $K$ exactly as
pinned. The raw pattern shows something the band was not constructed to
detect. The frozen arm is flat with no trend in $K$, ranging $-0.0195$
to $-0.0469$; the trainable arm grows roughly monotonically, from
$-0.0195$ at $K{=}12$ to $-0.0938$ at $K{=}40$. The pinned verdict is
therefore true as pinned, and true of the frozen arm outright; for the
trainable arm it is a statement about this range of $K$ and nothing
more. An earlier attempt to convert that growth into a numeric breach
prediction was retracted on independent recomputation (Section 6.2), and
no extrapolation past the measured range is claimed here.

The combined reading is one sentence: freezing buys depth robustness
that is breadth-stable, and trainable adapters pay a depth cost that
grows with breadth. Meanwhile the wall is unmoved by depth: at eleven
squarings, 48 of 48 cells sit inside the chance band, the closest
approach being a $K{=}36$ maximum of 0.0547 against a band top of
0.0586.

## 6.6 Coverage and Depth Drift

Two supporting measurements at the anchor breadth $K{=}24$ bound how
much of the claim rests on a favorable choice of read.

**Coverage.** Fifteen previously unmeasured residues were evaluated on
both frozen arms at $n{=}256$, against a pre-registered
COMPLETE-COVERAGE band of 0.95 at all fifteen. The floor across those 30
readings is 0.9922; the contrastive-plus-cosine arm reads a minimum of
0.9961 and a mean of 0.9990, the cosine-only arm a minimum of 0.9922 and
a mean of 0.9982, and both re-read at a minimum of 0.9961 under matched
pools. Together with the three training residues and five previously
measured ones, all 23 nontrivial positions of the $K{=}24$ cycle are now
measured at 0.99 or above under exact write, while the model's own
learned write reads 0.019 to 0.059 against chance 0.0417 at every one of
them. The exact-write ceiling is not an average over a favorable subset
of the cycle.

**Depth drift.** Holding the residue fixed at 13 and varying physical
depth over $h \in \{13, 61, 253, 1021, 4093\}$, all congruent to 13
modulo 24, leaves the correct answer identical by construction and
changes only the number of squarings, from three to eleven. The
pre-registered bands were ROBUST if every deeper reading sits within
0.02 of the $h{=}13$ reading and DRIFT if the decline is monotone and
larger. The verdict is DRIFT: the two frozen arms fall from 1.0000 to
0.9219 and from 1.0000 to 0.9062, strictly monotone across the ladder.
Extended across the checkpoint population, 42 of 55 cells are strictly
monotone non-increasing and the trainable cosine-only arm is monotone in
8 of 8. Three of the 58 checkpoints predate the configuration flags that
define the recipe grid and are reported separately.

The interpretation is narrow and worth stating: repeated squaring is
exact in arithmetic and is not exact in floating point, so the
composition read degrades slowly with the number of squarings even when
the answer it should return does not change. That cost belongs to the
read, and the training recipe modulates it. At eleven squarings the
freeze effect is a median gap of $+0.1133$ while the auxiliary-loss main
effect is 0.0137; but within the trainable arm, switching from
contrastive-plus-cosine to cosine-only costs $-0.2832$, whereas the same
switch within the frozen arms moves $+0.0312$ in the other direction.
Freezing and the contrastive auxiliary are substitutes for the same
robustness, not independent additive gains.

## 6.7 Scope and Limitations

**The curve ends at $K{=}40$ for an arithmetic reason, not a measured
one.** The ladder rule takes the top rung to be the smallest $h$ in
$[32,63]$ congruent to $K/2$ modulo $K$, and $[32,63]$ is exactly the
band costing five squarings, held fixed across the curve so that breadth
is the only moving axis. Such an $h$ exists only when $3K/2 \leq 63$,
that is $K \leq 42$. At $K{=}44$ the derivation raises rather than
returning a ladder. An independent audit re-derived both frontier
ladders and all six of record from the rule alone, without importing the
design's configuration, and confirmed that the $K{=}44$ derivation
raises, so the limit is a verified property of the construction rather
than an assertion. The two ways past it, a different squaring count or a
non-antipodal top rung, each break one of the two things the design
holds fixed and make the breadth axis uninterpretable; $K{=}44$ was
therefore dropped at design time. No reading on this curve licenses any
claim about $K \geq 44$ in either direction, and every table caption
carries that limit.

**P1b is exact teacher-forced operator substitution, not a learned
write.** Every capability number in this chapter is a read-path
measurement on a supplied operator. The model's own writes are Section
6.4's wall, and the two must not be quoted as one result.

**The task is a synthetic graft.** The composition head and its
entity-binding task exist to exercise the read path inside a real
language model. Nothing here claims the backbone uses this circuit in
ordinary language modeling, and the graft is not a language-modeling
benchmark.

**Three seeds per $(K, \mathrm{recipe})$.** A per-$K$ rank test on three
versus three has a minimum attainable two-sided $p$ of 0.10, so no
per-$K$ significance claim is made or is possible; every test above is
stratified across $K$ for that reason. This was disclosed in the design,
not discovered at harvest.

**$d = K{+}1$ co-varies with $K$ throughout.** The manipulated variable
is the pair, never $K$ alone, and a curve at fixed $d$ would be a
different experiment.

**One evaluation seed for the curves of record** (90210, $n{=}256$); the
re-measures use 31337. Seed variation here is across training
checkpoints, not across evaluation draws.

**The ordering result is depth-conditional.** It is negligible at the
antipodal top rung and confirmed at eleven squarings; it is a claim
about a deep read, not about the capability of Section 6.3, and it is
graded rather than separating at this scale.
