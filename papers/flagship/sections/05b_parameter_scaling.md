# 7 Parameter Scaling

<!-- DRAFT, flagship consolidation 2026-08-23. Sources of record: the raw
archives under experiment-runs/2026-08-22_scaleaxis_* and
experiment-runs/2026-08-23_{attribution_arm,v2prime}/; the recomputed
findings note pebble-ai-site/findings/ncr-scale-axis.html (its
raw-recomputed values govern wherever they differ from EXPERIMENT_LOG
prose; the correction entries are 2026-08-22 #23 and 2026-08-23 #5);
EXPERIMENT_LOG.md 2026-08-22 #10 through 2026-08-23 #5. Novelty gate:
research/scale-axis-novelty-2026-08-22.md (3/3 legs clear). Intended
position: immediately after the breadth chapter. -->

Section 6 varies binding load at one model size. This chapter holds the
task, the ladder, the metric, and the training budget fixed and varies
the model instead, from 98M to 392M parameters. The protocol is the one
described in Section 6.2.

## 7.1 The Second Scale Point

The reference point is Section 6's backbone: $d_{model}=768$, 12 layers,
$d_{state}=64$, 97,619,712 backbone parameters, and per-arm totals from
97,816,977 at $K{=}16$ to 97,860,009 at $K{=}40$. The second point is
$d_{model}=1536$, 16 layers, $d_{state}=128$, 391,872,512 backbone
parameters, per-arm totals from 392,095,889 to 392,175,785: a factor of
4.008 per arm, 4.014 on the backbone alone. Steps (20,000), batch size
(32), learning rate ($3 \times 10^{-4}$), warmup (200 steps), the ladder
rule, the metric, the recipes, and the evaluation seed are identical
across the two points. Four of Section 6's eight breadths are ported,
$K \in \{16, 24, 32, 40\}$, always with $d = K{+}1$.

One asymmetry is load-bearing throughout and is stated before any
result. The token budget is held fixed rather than scaled with
parameters, so tokens per parameter falls from 1.87 to 0.47 at $K{=}40$.
The resulting confound runs in one direction only: an under-trained
larger model can manufacture an apparent degradation, but it cannot
manufacture a capability. A flat capability curve and a stable wall are
therefore safe against it, while every degradation is a candidate
artifact until an attribution control rules the budget out. Section 7.5
is that control, and one of the two degradations did not survive it.

The flat shape was pre-registered as the expected null rather than
reported as a surprise. The recall-capacity literature already reports
structural ceilings that move little with model size: Arora et al. (2023)
find associative-recall capacity governed by state dimension
rather than parameter count, and Jelassi et al. (2024) find
state-limited copying behavior that scale does not lift. The design
recorded both as the precedent class for a flat reading before any 392M
cell was queued.

## 7.2 Capability: No Detectable Directional Shift at Four Times the Parameters

Table 5 gives the per-seed readings at both scale points side by side.

**Table 5 caption.** Read-path capability at 392M parameters against the
98M reference, under P1b (exact teacher-forced operator substitution),
at each $K$'s antipodal top rung; $\kappa$ chance-corrected, $n{=}256$,
eval seed 90210, checkpoints at 20,000 steps, $d = K{+}1$, three seeds
per cell. Both scales train on the identical step, batch, and
learning-rate schedule; the token budget is not scaled with parameters
(Section 7.1). $K{=}24$'s 98M cells are the eval-only anchor, which is
why the pre-registered sensitivity form drops that stratum. Archives:
`experiment-runs/2026-08-22_scaleaxis_sweep/` (392M training and
batteries), `experiment-runs/2026-08-22_scaleaxis_stagec/` (depth
extension, re-measures, and the 98M reference re-scores), with the 98M
originals in `experiment-runs/2026-08-22_kscaling_sweep/` and
`experiment-runs/2026-08-22_kscaling_frontier/`.

| pair $(K, d)$ | top rung $h$ | $\kappa$ frozen, 3 seeds (392M) | frozen median (98M) | $\kappa$ trainable, 3 seeds (392M) | trainable median (98M) |
|---|---|---|---|---|---|
| (16, 17) | 40 | 1.0000 / 1.0000 / 1.0000 | 1.0000 | 0.9792 / 0.9708 / 0.9917 | 0.9958 |
| (24, 25) | 36 | 0.9959 / 0.9959 / 0.9959 | 1.0000 | 0.9878 / 0.9674 / 0.9837 | 0.9878 |
| (32, 33) | 48 | 0.9960 / 0.9879 / 0.9839 | 0.9960 | 0.9516 / 0.8750 / 0.8065 | 0.9919 |
| (40, 41) | 60 | 0.9880 / 0.9960 / 0.9880 | 0.9920 | 0.7596 / 0.8438 / 0.8798 | 0.9880 |

All twelve frozen cells clear the $\kappa \geq 0.90$ gate on 3 of 3
seeds at every ported $K$. Frozen medians read 1.0000, 0.9959, 0.9879,
0.9880 with a floor of 0.9839, and move between 0.0000 and 0.0081
relative to their 98M counterparts.

The cross-scale test is a stratified exact permutation statistic over
eight strata (four $K$ by two recipes) and nine cross-scale seed pairs
each, counting how often the 392M cell exceeds its 98M counterpart. It
reads $T = 17.0$ of 72, below the degradation bar of 19: nominally a
degradation. The pre-registered sensitivity form drops the $K{=}24$
stratum pair, whose 98M cells are an eval-only anchor trained under an
older runner, leaving six strata and $T = 14.5$ of 54 against a
degradation bar of 12: inside band. The precedence rule was fixed before
data, and says that where the two forms disagree the six-stratum verdict
governs. They disagree. The verdict of record is NO DETECTABLE
DIRECTIONAL SHIFT at four times the parameters.

The eight-stratum reading is not argued away but reported with what is
wrong with it. It is tie-heavy: 14 of its 72 pairs are exact ties, a
ceiling effect rather than a signal. It survives only 4 of 8
leave-one-stratum-out subsets, where the six-stratum form is the one the
design nominated in advance. And its improvement branch was
arithmetically unreachable from the 98M numbers before the wave ran,
since the maximum attainable improvement per cell ranged 0.0000 to
0.0122 against a 0.05 threshold, at 0 of 8 cells; a statistic that
cannot register the outcome it is nominally testing for is not the one
to carry the verdict.

Per $K$, six of the eight cells are stable within $\pm 0.05$. The two
that are not are both trainable, at $K{=}32$ ($-0.1169$) and $K{=}40$
($-0.1442$), and both also fall below the $\kappa \geq 0.90$ gate at
392M. Section 7.5 is about exactly those two cells and about what
happened when we tested them.

"No detectable shift" is not "no shift". It is the pre-registered band's
label for a reading inside the noise floor at this seed and stratum
count, and it should be read as the resolution of the instrument rather
than as a proof of invariance.

<!-- FIGURE SLOT (not generated in this pass): two-panel scale figure.
Left, kappa at the antipodal top rung against K, one line per (scale,
recipe), with the 0.90 gate drawn: the frozen pair overlays, the
trainable pair separates at K=32 and K=40. Right, the attribution
sequence of Section 7.5 as a slope chart, 20k to 40k steps, V1 rising
over the gate and V2-prime flat below it. Sources:
experiment-runs/2026-08-22_scaleaxis_sweep/,
experiment-runs/2026-08-23_attribution_arm/,
experiment-runs/2026-08-23_v2prime/. -->

## 7.3 The Wall Is Scale-Stable

The learned write was measured at 392M under the same band construction
as Section 6.4, chance plus three binomial standard deviations at
$n{=}256$, per $K$, with the same re-measure clause.

Of the 60 P0 readings at each ported breadth, $K{=}24$ and $K{=}32$ read
60 of 60 inside band. $K{=}16$ reads 59 of 60, the exception a trainable
seed at $h{=}1$ reading 0.1133 against a band top of 0.1079, which
re-measures to 0.0859 at the independent draw. $K{=}40$ reads 59 of 60,
the exception a trainable seed at $h{=}8$ reading 0.0547 against a band
top of 0.0543, an excess of a single correct answer in 256, which
re-measures to 0.0156 with a cell maximum of 0.0352 across all ten hops.
No breach reproduces at any ported $K$, so the verdict is WALL-HOLDS at
all four and, across the two scales, WALL-SCALE-STABLE. Four times the
parameters, on this budget, does not teach the model to write the
operator.

The extended cells of Section 7.5 give the same reading at 40,000
steps: 118 of 120 P0 readings inside band, with two single-seed $h{=}1$
excursions ($K{=}16$ at 0.1133, $K{=}24$ at 0.0820) unreplicated at
their own $K$ and therefore verdict-compatible under the excursion
clause. Within 392M alone, the deepest read available puts P0 in band
in 24 of 24 cells at fifteen squarings ($h$ from 32,772 to 32,804); that
reading is labeled in every record as not comparable to the archived 98M
P0 at eleven squarings, and it is not used as a cross-scale claim.

## 7.4 The Freeze Ordering Sharpens to Perfect Separation

Section 6.5's ordering is graded at 98M: real under a stratified test at
depth, with violating pairs in half the strata. At 392M it stops being
graded.

The within-scale statistic $T_W$ counts, out of 36, the four ported $K$
times nine frozen-versus-trainable seed pairs the frozen cell wins. At
98M it reads 30.5 at eleven squarings and 25.0 at the antipodal top
rung. At 392M it reads 36.0 of 36 at both readouts: the statistic's
maximum, with zero realized ties, two-sided exact
$p = 1.25 \times 10^{-5}$ under the conditional permutation null. Every
frozen seed beats every trainable seed, at every ported breadth, at both
readouts. Leave-one-stratum-out reads 27.0 of 27 in all four subsets at
both readouts, where the 98M reference clears its own bar of 24 of 27 in
only 2 of 4 subsets at eleven squarings (24.0, 21.5, 24.5, 21.5).

The threshold deserves its own sentence, because a weaker one was
available. A raw exact $p < 0.01$ at four strata is $T \geq 30$, which
the 98M reference itself clears by half a pair; confirming the 392M wave
against that bar would have proved nothing about scale. The design
instead required $T_W > 31.5$ together with leave-one-stratum-out
clearing in at least 3 of 4 subsets, which is to say that confirmation
required the 392M wave to be strictly more robust than its own 98M
reference. The sub-case of record is ORDERING-SCALE-STRENGTHENS, with
descriptive deltas of $+5.5$ at eleven squarings and $+11.0$ at the top
rung.

This comparison carries no token-budget caveat. Both recipes train on
the identical budget at the identical scale, so the confound cancels
inside the comparison rather than being argued around it.

Two bounds. Thirty-six of 36 is the ceiling of the statistic, so it
cannot read as more than perfect at this resolution, and a larger seed
count would be needed to say how much more than perfect it is. And this
is a four-stratum test where Section 6.5's was an eight-stratum test;
the design compensated by raising the confirmation threshold rather than
by treating the shorter instrument as equally strong.

## 7.5 Scale by Breadth: One Degradation Withdrawn by Our Own Control, One Architectural

The two axes interact, and the interaction is confined to the trainable
recipe. At 392M the frozen arm's top-rung $\kappa$ ranges 0.0121 across
the four breadths while the trainable arm's ranges 0.1399; the
fixed-distance control shows the same shape, 0.0080 against 0.1040,
which is what licenses reading the effect as breadth-driven rather than
as a depth cost in disguise.

Read naively, that is two scale degradations, at $K{=}32$ and $K{=}40$.
The design forbade publishing either one before an attribution arm had
run, for the reason given in Section 7.1: the token budget is fixed
across scales and can manufacture exactly this shape. Table 6 gives that
arm in the order its three controls resolved it.

**Table 6 caption.** The attribution sequence at 392M, top rung, P1b,
trainable recipe. Each control resumes the 20,000-step checkpoints of
seeds 0 and 1 for a further 20,000 steps, doubling tokens. V1 tests the
token budget at $K{=}32$; V2 tests it at $K{=}40$ through the same
resume path, which re-opens the schedule's warmup; V2$'$ repeats V2 with
the learning rate held constant at the schedule's own 20,000-step floor
and nothing else changed. Medians are over the seeds available at that
step count (three at 20,000, two at 40,000), and the two-versus-three
seed convention gap is disclosed in Section 7.7. Archives:
`experiment-runs/2026-08-23_attribution_arm/` (V1, V2) and
`experiment-runs/2026-08-23_v2prime/` (V2$'$), against the 20,000-step
parents in `experiment-runs/2026-08-22_scaleaxis_sweep/`.

| cell and control | resume schedule | $\kappa$ at 20,000 steps | $\kappa$ at 40,000 steps | verdict |
|---|---|---|---|---|
| $K{=}32$, V1 | warm restart | 0.9516 / 0.8750 / 0.8065 (0.8750) | 0.9153 / 0.9073 (0.9113) | token-budget-limited; claim withdrawn |
| $K{=}40$, V2 | warm restart | 0.7596 / 0.8438 / 0.8798 (0.8438) | 0.4391 / 0.6635 (0.5513) | control damaged; uninformative |
| $K{=}40$, V2$'$ | constant $3.00 \times 10^{-5}$ | 0.7596 / 0.8438 (parents of the two cells) | 0.7276 / 0.8598 (0.7937) | unrecovered and undamaged; architectural |

**$K{=}32$ is withdrawn.** V1 resumed seeds 0 and 1 to 40,000 steps at
twice the tokens, and both cleared the $\kappa \geq 0.90$ gate: 0.9153
and 0.9073, median 0.9113, against a 20,000-step median of 0.8750. The
below-bar seed rose from 0.8750 to 0.9073 and the above-bar seed settled
from 0.9516 to 0.9153. The verdict is TOKEN-BUDGET-LIMITED and the
$K{=}32$ scale-degradation claim is withdrawn. We report this as the
protocol working rather than as a result: the control that removed the
claim was mandated before the claim existed, and the claim never
published.

**$K{=}40$ needed two controls.** The first, V2, used the same resume
mechanism, which re-opens the linear-warmup-plus-cosine schedule and
lifts the learning rate from $3.00 \times 10^{-5}$ to approximately
$1.66 \times 10^{-4}$, a 5.5-fold warm restart disclosed before the arm
ran. Those cells did not merely fail to recover; they were damaged,
reading 0.4391 and 0.6635 against parents of 0.7596 and 0.8438, for
seed-matched changes of $-0.3205$ and $-0.1803$, the largest movement
anywhere in the arm. A control that damages its own cells cannot rule
the budget in or out, so the verdict was recorded as SCALE-DEGRADES,
unstrengthened, rather than claiming the pre-registered word on a broken
instrument, and the arm was re-run.

The second control, V2$'$, held the learning rate constant at
$3.00 \times 10^{-5}$, the schedule's own 20,000-step floor, for the
same two cells over the same marginal 20,000 steps, with the restart
removed and nothing else changed. The cells read 0.7276 and 0.8598
against parents of 0.7596 and 0.8438: unrecovered, 0 of 2 above the
$\kappa \geq 0.90$ bar, and undamaged, with seed-matched changes of
$-0.0321$ and $+0.0160$ inside the parent band. Doubling the tokens does
not recover the degradation and removing the restart does not either.
The verdict is SCALE-DEGRADES, cleanly strengthened: at $K{=}40$ the
trainable arm's degradation at 392M is architectural rather than a
budget or schedule artifact. The wall check on those cells holds, with
P0 maxima of 0.0391 and 0.0469 against a band top of 0.0543.

The recipe separation therefore widens with scale exactly where breadth
is largest. At $K{=}40$ and 20,000 steps the frozen-minus-trainable gap
is 0.144 in $\kappa$; measured against the frozen arm's own 40,000-step
cells (0.9840 and 0.9880) it is 0.192.

**The depth tail, in the strong form.** One cross-scale degradation
survives the attribution arm in full, and it is depth robustness rather
than capability. At eleven squarings the cross-scale statistic reads
$T = 10.5$ of 72, exact $p = 3.6 \times 10^{-5}$; at the elected depth
$s^{*}{=}13$ it reads $T = 6.5$ of 72, exact
$p = 9.5 \times 10^{-7}$, against a degradation bar of 19. Unlike the
capability curve of Section 7.2, the six-stratum sensitivity form agrees
with the eight-stratum form here, and leave-one-stratum-out survives 8
of 8. The depth $s^{*}{=}13$ was elected by a mechanical rule applied to
the 98M re-score before any 392M cell was queued.

The strong form is what the attribution arm established. At twice the
tokens all six extended cells move further from recovery rather than
toward it; the closest miss is still 0.062 outside the $\pm 0.095$
recovery band; and marginal cross-entropy fell in 10 of 12 extended
cells, median $-0.125$ and largest $-2.686$, so the models demonstrably
kept training while the tail did not recover. The effect reaches both
recipes, not only the trainable one: frozen $K{=}16$ and $K{=}24$ read
stable ($-0.0125$ and $-0.0285$ against 98M), while frozen $K{=}32$
moves 0.9194 to 0.7863 and then 0.7621 at 40,000 steps, and frozen
$K{=}40$ moves 0.9038 to 0.7837 and then 0.7236. The trainable arm's
tail is close to gone at the widest breadth, 0.6715 at 98M against
0.0184 at 392M and 0.0004 after extension.

Two of the twelve extended cells failed a pre-registered clause
requiring the marginal segment to reduce cross-entropy, reading $+0.026$
and $+0.030$, both at $K{=}40$. The clause was ruled mis-scoped for a
resumed segment already at a plateau, and the flat cross-entropy was
recorded as attribution evidence in its own right, since it says the
budget was delivered and bought nothing. The scoping decision is on the
record rather than applied silently.

## 7.6 Two Instrument Findings

Two results here are about the measurement apparatus rather than about
the model, and both are reported because the second one changed a
verdict.

**Warm restarts destabilize trained trainable adapters at the breadth
frontier, and nothing else measured here.** Under the identical restart,
every other extended cell stays within $\pm 0.04$ at the top rung:
$K{=}16$ trainable at $-0.0083$ and $+0.0208$, $K{=}24$ trainable at
$-0.0041$ and $-0.0041$, $K{=}32$ trainable at $-0.0363$ and $+0.0323$,
$K{=}32$ frozen at $+0.0040$ and $+0.0040$, $K{=}40$ frozen at
$-0.0040$ and $-0.0080$. Only the $K{=}40$ trainable cells move by
$-0.3205$ and $-0.1803$. A trained trainable adapter at the widest
breadth is the fragile object, and frozen adapters shrug the restart
off. The practical consequence for anyone extending a fast-weight
adapter from a checkpoint is that the resume should either hold the
learning rate constant or disclose the schedule it re-opens, because the
schedule alone can move a frontier cell by a third of the metric's
range.

**Direction of effect.** Extension arms are ordinarily licensed on an
unstated assumption that more training cannot hurt. Scoring all twelve
(cell by readout) comparisons of 20,000-step against 40,000-step medians
against a $\pm 0.01$ flat band gives 1 helped, 4 flat, and 7 hurt. The
single case that helped is the $K{=}32$ top rung at $+0.0363$, the cell
that withdrew our own claim. One of the seven that hurt is a boundary
call, $K{=}24$ trainable at the top rung at $-0.0102$, a thousandth past
the band, and it is named as such rather than counted silently; the
honest restatement is that the extension helped once and hurt or was
flat everywhere else. Checking that assumption rather than asserting it
is what forced V2$'$ and converted the $K{=}40$ verdict from standing on
a broken control to standing on a clean one.

## 7.7 Scope and Limitations

**Two model sizes are not a scaling law.** 98M and 392M are two points.
No slope is fitted, "survives four times the parameters" is exactly and
only what was measured, and nothing here extrapolates to a billion
parameters or to a different architecture family.

**The token budget is fixed and its confound is one-directional.**
Tokens per parameter falls from 1.87 to 0.47 at $K{=}40$. Under-training
can manufacture a degradation and cannot manufacture a capability, so
the flat capability curve of Section 7.2 and the stable wall of Section
7.3 are safe against it, while every degradation required the
attribution arm. One of the two degradations did not survive that arm.
Nothing in Section 7.5 should be cited as evidence that the capability
is scale-fragile.

**The extended arm is two seeds, not three.** The 40,000-step cells are
seeds 0 and 1, compared against 20,000-step medians over three seeds. A
seed-matched sensitivity that restricts the 98M twin to the same two
seeds leaves 5 of 6 depth-tail verdicts unchanged and flips only frozen
$K{=}32$, the cell nearest the bar. The convention gap is recorded as a
design gap of this chapter, not resolved by it.

**The $K{=}40$ trainable depth-tail row at 40,000 steps comes from the
schedule-damaged cells.** V2$'$ was scored on the breadth battery and
not on the depth ladder, so that one row is a warm-restart reading. The
verdict does not turn on it: the 20,000-step reading already missed the
$\pm 0.095$ recovery band by 0.558.

**"Double the tokens" means tokens seen, not loss moved.** At $K{=}40$
in particular the marginal cross-entropy barely moved, $+0.026$ and
$-0.055$ under the restart and $-0.048$ and $+0.005$ at constant
learning rate. Those cells are at a plateau, so the $K{=}40$
token-budget explanation is ruled out on budget delivered rather than on
further fitting achieved.

**V2$'$ is a two-cell schedule control at one breadth and one constant
learning rate.** It rules out this restart as the cause of this damage.
It is not a learning-rate sweep, and it does not establish that no
schedule could recover the $K{=}40$ trainable cells.

**Four breadths, not eight.** $K \in \{12, 20, 28, 36\}$ have no 392M
counterpart, and $K{=}44$ is construction-impossible for the arithmetic
reason given in Section 6.7. Three seeds per $(K, \mathrm{recipe},
\mathrm{scale})$, with every test stratified across $K$ for the reason
given there. The task remains the synthetic graft of Section 6.1, and
nothing here claims the backbone uses this circuit in ordinary language
modeling.

**Two recorded process disclosures.** The chapter's compute ledger
overran its pre-registered 130 GPU-h gate at approximately 135.7 GPU-h
all-in; the overrun was licensed by a recorded adjudication before the
cells were queued, and is reported as authorized rather than fitted
after the fact. And the fixed-distance control's three pre-registered
labels were defined on $\kappa$ without naming an arm, so the two arms
return different labels; the control is reported per arm rather than
elected, and the ambiguity is recorded as a design gap.
