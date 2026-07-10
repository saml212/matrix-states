# Attack Report — UniReps EA "Dimension, Not Solvability" (round 1, stage 01)

Reviewer: hostile attack agent, fresh context. All headline numbers were
independently recomputed from the raw JSON artifacts (not taken from the
draft, brief, or design doc prose) using
`/Users/samuellarson/Experiments/learned-representations/.venv/bin/python`.
Recomputation script and full console output are reproducible from the
paths cited inline.

## 1. Summary for the defense agent

**Verdict: strong revision required, not a reject-on-arithmetic paper.**
Every headline number I recomputed from the raw artifacts (U1's five group
means, sd's, band membership, ρ=0.9747, the 8/120 exact permutation null;
U2's TOST diff/se/df/t1/t2; U3's five-group razor table; U4's four-seed S3
necessity zeros and seed-mean; U7's centering gap) **matches the draft
exactly**. The necessity leg of the causal razor (`k=d_min-1` → exactly
0.000 recovery, zero exceptions across 24 independent cells) is the single
most bulletproof result in the paper and survives every attack below.

The central weakness is not fabrication, it is **overclaiming past what an
n=1-per-cell, self-referentially-bounded instrument can support**: (1) the
"unconstrained anchor" the causal section benchmarks against is not
behaving as a capacity ceiling — rank-*capped* cells beat it outright in a
majority of groups, which the paper's own "restores recovery" language
glosses over; (2) the M1 band's upper bound (`≤1.3·d_min`) is
mathematically impossible to violate given how `restricted_effective_rank`
is constructed, so "all 19 seeds in band" is a partially vacuous claim
presented as a live two-sided pre-registered test; (3) the introduction's
"4–8%" deviation claim is contradicted by the paper's own Table 1 (S5 is
10.2% off); and (4) the paper never engages the closest prior work in the
literature — networks learning group-representation-theoretic structure
from training on finite-group tasks is a studied phenomenon (Chughtai,
Chan & Nanda 2023; Nanda et al. 2023) that this draft's related-work
section omits entirely, despite writing in almost the same terms
("what, exactly, converges").

**Salvageable:** the causal necessity result, the TOST design and its
executed power simulation, the D-AMB diagnosis-and-fix narrative (a
genuinely good methods story), and the basic five-group correlational
trend. **Not yet salvageable as written:** the "restores recovery" /
"step function" characterization of sufficiency, the two-sided-band
framing of M1, and the missing engagement with the group-representation-
learning literature that shares this paper's exact substrate.

## 2. Attacks

### A1: The "unconstrained anchor" is not a reliable ceiling — capped runs beat it outright, undermining "restores recovery"

**Severity:** CRITICAL
**Type:** positive-control adequacy / reproducibility

**Attack.** The abstract's third headline claim is: "capping rank one
below $d_{\min}$ zeroes exact recovery in every group, and restoring
$d_{\min}$ **restores** recovery." Section 5 echoes this: "Sufficiency
holds at $d_{\min}$: $S_4$, $A_5$, $S_5$, $A_6$ recover past their bars
outright (0.800/0.700/0.600/0.650 against 0.585/0.630/0.450/0.585)." I
recomputed `crosscheck_recovered_frac_90` directly from
`experiment-runs/2026-07-09_m3fix_harvest/zero_pad__*__seed0.json` for the
`unconstrained`, `k_dmin`, and `k_dmin_plus_1` arms of every group:

| G | anchor (unconstrained) | k=d_min | k=d_min+1 |
|---|---|---|---|
| S3 | 0.550 | 0.450 | 0.550 (=anchor) |
| S4 | 0.650 | **0.800** (+0.150) | **0.950** (+0.300) |
| A5 | 0.700 | 0.700 (=anchor) | **0.750** (+0.050) |
| S5 | 0.500 | **0.600** (+0.100) | 0.550 (+0.050) |
| A6 | 0.650 | 0.650 (=anchor) | **0.700** (+0.050) |

In **every single group**, `k=d_min+1` — a rank-capped cell with *less*
expressive freedom than the "unconstrained" anchor
($d_{\mathrm{state}}=d_{\min}+2$, i.e. two more free dimensions than
`k=d_min+1`'s cap) — matches or beats the anchor it is supposedly
"restoring recovery" toward. In S4 and S5, even the exact `k=d_min` cell
beats the anchor. A properly-converged unconstrained run should be a
*ceiling*: nothing with less rank budget should systematically
outperform it. That capped cells repeatedly beat it, by margins up to
+0.300 (S4, a 46% relative gain), is strong evidence the "anchor" value
is not a stable ceiling but a noisy **n=1** point estimate (the paper's
own Limitations concede: "Causal cells are single-seed except $S_3$").
This directly threatens the causal section's rhetorical structure: the
"0.9× anchor" bar that every CONFIRM verdict is graded against is
computed from a baseline the data show is not reliably the best-achievable
value. Had the single anchor run landed 0.15–0.30 higher by chance (fully
plausible given what `k=d_min+1` already achieves), several of the
CONFIRM cells would not have cleared the resulting bar. The paper's own
design record acknowledges "anchor noise" once, narrowly, for $S_3$'s
marginal cell (`CAPABILITY_SEPARATION_DESIGN.md` §1.36a: "driven by
anchor noise (0.550-0.800, seed 2's anchor is a high outlier)") but never
extends that diagnosis to the larger, more systematic version of the same
problem in S4 and S5.

**Supporting evidence.** Raw files: `zero_pad__S4__unconstrained__seed0.json`
(`crosscheck_recovered_frac_90=0.65`), `zero_pad__S4__k_dmin__seed0.json`
(`=0.80`), `zero_pad__S4__k_dmin_plus_1__seed0.json` (`=0.95`);
`zero_pad__S5__unconstrained__seed0.json` (`=0.50`) vs.
`zero_pad__S5__k_dmin__seed0.json` (`=0.60`). All in
`experiment-runs/2026-07-09_m3fix_harvest/`.

**What the paper would need to do to defuse this.** Run the unconstrained
anchor at ≥3–5 seeds per group (matching M1's own seed budget, which
already exists at 3–5 seeds for the *other*, eye-padded target) so the
bar is a stable estimate with a reportable variance, not a single draw.
Report that variance next to the 0.9× bar. Reword "restoring $d_{\min}$
restores recovery" to something the data actually support — e.g.
"clears a pre-registered bar" — since in a majority of groups the capped
cell does not merely restore but *exceeds* the reference condition, which
is a materially different (and currently unexplained) empirical pattern.

---

### A2: M1's "$[0.7,1.3]\cdot d_{\min}$ band" is a one-sided test dressed as two-sided — the metric cannot exceed $d_{\min}$ by construction

**Severity:** SERIOUS (near-critical; would be CRITICAL if M1 alone carried the paper's decisive weight, but the paper itself demotes M1 to corroborating)
**Type:** statistical / claim-scope

**Attack.** `matrix-thinking/capability_separation/readout.py`,
`entity_subspace_from_words`:

```python
U_full, _, _ = np.linalg.svd(cov)
return U_full[:, :d_min]
```

`U` has exactly `d_min` columns by construction. `restrict(Z, U) =
U^T Z U` is therefore always a `d_min × d_min` matrix, and
`rank_utils.effective_rank` is documented "Range `[1, d]`" for a `d×d`
input. So `restricted_effective_rank ∈ [1, d_min]` for **every** seed of
**every** group, unconditionally — it is mathematically impossible for it
to exceed `d_min`, and therefore impossible to exceed `1.3·d_min` (since
`d_min ≥ 2`). Yet the abstract, §3, and Table 1 all present "all 19 seeds
inside the pre-registered $[0.7, 1.3]\cdot d_{\min}$ band" as though both
sides of the band were live falsification tests the data could have
failed. Empirically every group mean lands in `[0.895, 0.951]·d_min` —
consistent with a metric that structurally can never approach, let alone
exceed, its own ceiling. Only the lower-bound half (`≥0.7·d_min`) is a
genuine test the model could fail; the upper half is decorative. This
compounds A6 below (the target itself is pinned at rank `d_min`): the
instrument that reads out "convergence" is built so it cannot report
anything above `d_min`, on a target that cannot supervise anything above
`d_min`.

**Supporting evidence.** `matrix-thinking/capability_separation/readout.py`
lines ~51–96 (`entity_subspace_from_words`, `restrict`);
`matrix-thinking/capability_separation/rank_utils.py` / `chapter2/rank_utils.py`
`effective_rank` docstring: "Range `[1, d]`". Consistent with the observed
data: no group mean or per-seed value in any of the 19 unconstrained files
approaches `d_min`, let alone `1.3·d_min`.

**What the paper would need to do to defuse this.** Either drop the
upper-bound framing from the headline claim and report only the genuine
lower-bound test, or explicitly state in the main text (not just imply
via "a $d_{\min}$-restricted readout partially favors rank $\approx
d_{\min}$") that the band's ceiling side cannot bind by construction and
is reported for completeness, not as independent evidence.

---

### A3: Introduction's "4–8%" deviation claim is contradicted by the paper's own Table 1

**Severity:** SERIOUS
**Type:** number provenance

**Attack.** §1 (Introduction) states: "per-group mean rank lands within
4–8% of $d_{\min}$." Recomputing `(d_min − mean)/d_min` from Table 1's own
numbers:

| G | mean | d_min | % below d_min |
|---|---|---|---|
| S3 | 1.877 | 2 | 6.15% |
| S4 | 2.852 | 3 | 4.93% |
| A5 | 2.832 | 3 | 5.60% |
| S5 | 3.591 | 4 | **10.22%** |
| A6 | 4.736 | 5 | 5.28% |

S5's 10.22% is well outside the claimed 4–8% window — not a rounding
artifact (it is ~2.2 points over the stated ceiling). This is a plain
arithmetic mismatch between the Introduction's prose summary and the
paper's own Appendix table.

**Supporting evidence.** Table 1 (`07_appendix.tex`, `app:m1`): S5 mean
$3.591 \pm 0.069$ vs. $d_{\min}=4$. `(4-3.591)/4 = 0.1023`.

**What the paper would need to do to defuse this.** Correct the range
(e.g. "4–10%," or "4–8% for four of five groups, 10% for S5") — ideally
tied to the S5 soft-convergence disclosure (see A4), rather than silently
rounded into a tighter-looking range.

---

### A4: Unequal, non-monotonic per-group training budgets confound the cross-group comparison, and correlate with the two flagged "soft convergence" groups

**Severity:** SERIOUS
**Type:** alternative explanation

**Attack.** Per-group step budgets are pinned at S3=8K, S4=20K, A5=20K,
S5=8K, A6=40K (`CAPABILITY_SEPARATION_DESIGN.md` §1.33; the draft's own
§2 states "8k–40k"). Budget does **not** track $d_{\min}$ monotonically:
S5 ($d_{\min}=4$) gets the *same* 8K budget as S3 ($d_{\min}=2$), while
S4/A5 ($d_{\min}=3$, smaller) get 2.5× more steps and A6 ($d_{\min}=5$,
largest) gets 5× more. The paper's own appendix discloses that exactly
the two 8K-step groups — S3 and S5 — show soft convergence on gate1a
("$S_3$'s four variant-A cells (anchor min\_val 0.9143) and $S_5$'s
anchor/$k_{\min}$/$k_{\min+1}$ (0.876–0.879) sit just under the bar").
S5 is also the group with by far the largest deviation from $d_{\min}$
(A3, 10.2%), and S3 is the group whose causal cell triggered the
marginality re-run (§5, the $\pm0.05$ trigger). The paper does not rule
out a simple confound: the appearance of clean convergence partly tracks
which groups received enough steps to actually converge, not a clean
asymptotic representational law — since the two under-budgeted groups are
also the two showing the largest anomalies in both the observational
(A3) and causal (marginality trigger) legs.

**Supporting evidence.** `CAPABILITY_SEPARATION_DESIGN.md` §1.33 gate1a
disclosures ("S3's four variant-A cells... S5's anchor/k_dmin/k_dmin+1
(0.876-0.879) sit just under the bar"); §1.36a's own routing of the S3
extension on exactly this basis.

**What the paper would need to do to defuse this.** Either equalize step
budgets across groups (compute-matched), or run a budget-doubling
robustness check specifically on S3 and S5 and report whether the rank
estimate / recovery numbers move outside the reported bands.

---

### A5: "Step function" language overclaims a plateau at $d_{\min}$ that the raw recovery curve does not show

**Severity:** SERIOUS
**Type:** claim-scope

**Attack.** §5 states plainly: "Figure~\ref{fig:razor} shows a step
function at $\dmin$." A step function implies a hard floor below the
threshold *and* a plateau at/above it. The necessity half is real and
robust (`k=d_min-1` reads exactly 0.000 in all 5 groups and all 4
independent S3 seeds — no exceptions). But recovery does **not** plateau
at $d_{\min}$: it keeps rising through $k=d_{\min}+1$ in most groups —
S4: $0.800\to0.950$ (+0.150); A5: $0.700\to0.750$ (+0.050); A6:
$0.650\to0.700$ (+0.050); S3: ties ($0.450\to0.550$, back to the anchor
level). Only S5 shows a small dip ($0.600\to0.550$). A curve that
continues climbing past the claimed threshold in 3 of 5 groups is a
monotone ramp with a hard floor below $d_{\min}$, not a step/plateau —
this is a materially weaker (though still real) causal story: the raw
data establish that $d_{\min}$ is where recovery *first appears*
above 0, not that $d_{\min}$ is where sufficiency *saturates*.

**Supporting evidence.** The C1 decisional table (design doc §1.36),
independently reproduced from the raw JSONs (see A1's table, which
includes the same $k=d_{\min}+1$ column).

**What the paper would need to do to defuse this.** Replace "step
function" with "necessity threshold" or "onset of recovery" in §5 and
the Figure 2 caption; either show the curve reaching a genuine plateau
further out (e.g. $k=d_{\min}+2$) or drop the plateau implication
entirely.

---

### A6: The training target is directly supervised at rank $d_{\min}$ — "convergence" is partly definitional, not emergent

**Severity:** SERIOUS
**Type:** claim-scope / alternative-explanation

**Attack.** §2 states the loss is "cosine distance to the block-diagonal
embedding of the reference ($\rho_G(\cdot) \oplus I_2$ in the
observational sweep, $\rho_G(\cdot) \oplus 0$ in the causal wave)." The
model is trained end-to-end against a target whose informative content
**is** exactly $d_{\min}$-dimensional (the $\rho_G$ block); the remaining
2 ambient dimensions carry either a constant identity or zero — zero
group-relevant information either way. Under this loss, the optimum
*requires* rank exactly $d_{\min}$ on the informative subspace by
definition; there is no "discovery" of $d_{\min}$ from the group's
algebra independent of the supervision signal — the supervision signal
already encodes $d_{\min}$ as a design choice made by the experimenter
(`groups.py`'s `rho_G_embedded`, confirmed in design-doc §1.33 P1: "the
as-built target = `rho ⊕ I_2`"). The paper's framing — "A central
question for unifying accounts of neural representations is what,
exactly, converges... If trained models converge to the task's algebra"
— reads as though the network is discovering an algebraic fact through
training dynamics, when in fact the algebraic fact was hard-coded into
the loss target's own shape before training started. The genuinely
non-trivial content is narrower than the framing suggests: does the
network avoid *wasting* capacity on its 2 spare ambient dims (yes, per
M1), and does removing capacity below $d_{\min}$ break the fit (yes, per
the causal necessity leg, robustly). Both are real findings, but "trained
matrix states converge to the minimal faithful representation dimension"
(title) oversells a setup in which the minimal dimension was the
experimenter's supervision target, not an emergent discovery from a more
dimension-agnostic task (e.g. classification over group elements, or a
downstream task that does not literally require an $\mathbb{R}^{d_{\min}}$
output).

**Supporting evidence.** §2 (loss definition); `CAPABILITY_SEPARATION_
DESIGN.md` §1.33 P1 ("`groups.py:157-158` builds the target as
`torch.eye(d_state)` with `rho` overwriting only the top-left
`d_min×d_min` block").

**What the paper would need to do to defuse this.** Either scope the
claim explicitly to "the network does not waste or lose capacity beyond
what its supervision target requires" (a real, more modest, still
publishable finding), or add a control where the target's rank is *not*
pinned to $d_{\min}$ by construction (e.g. a higher-dimensional, still
faithful, non-minimal embedding of $\rho_G$, or a classification-style
loss) to show the network still contracts to $d_{\min}$ on its own. This
would substantially strengthen the "emergent convergence" framing the
title and abstract currently claim without earning it.

---

### A7: The title's "Not Solvability" half rests on a single group-pair contrast, not the "five groups" the abstract implies

**Severity:** MINOR
**Type:** claim-scope

**Attack.** The title, "Dimension, Not Solvability," and abstract frame
the result as spanning "five finite groups... spanning the
solvable/non-solvable divide." But only one pair in the family
($S_4$/$A_5$) actually dissociates dimension from solvability — every
other group's solvability class is confounded with its $d_{\min}$ rank
in this specific 5-group set (the two solvable groups, S3/S4, happen to
have the two smallest $d_{\min}$ values tested; the three non-solvable
groups, A5/S5/A6, have the three largest). Outside the engineered
S4/A5 tie, the 5-group correlational result (M1, $\rho=0.9747$) is
equally consistent with "rank tracks $d_{\min}$" and "rank tracks a
solvability-correlated severity ordering" — which is presumably *why*
the paper correctly built the marquee TOST as the decisive test rather
than resting on M1 alone (§3 already discloses M1 is "corroborating
rather than independently decisive"). Given that self-awareness already
exists in the body text, this is a framing nit on the title/abstract, not
a fresh methodological gap — hence MINOR rather than SERIOUS.

**Supporting evidence.** Table 1 / Appendix A group table: solvable =
{S3 ($d_{\min}=2$), S4 (3)}; non-solvable = {A5 (3), S5 (4), A6 (5)}.

**What the paper would need to do to defuse this.** A one-sentence
clarification in the abstract or intro that the solvability dissociation
is carried specifically by the $S_4$/$A_5$ pair, with the broader
5-group trend serving a separate (correlational, not dissociative) role.

---

## 3. Attacks I considered but decided were weak

- **"n=3–5 seeds per group is too small."** Considered, but the specific
  comparison that most needs statistical power — S4 vs A5 — is the one
  pre-registered with an explicit executed power simulation
  (`marquee_power_simulation.py`, §1.4.2.1) showing 100% power to reject
  equivalence at a true 1.0-rank-unit gap across a plausible noise grid.
  The broader 5-group correlational claim is explicitly and correctly
  demoted to "corroborating" by the paper itself. Not pressing this
  further as an independent attack.
- **"Single architecture, sub-1M parameters, no claim about pretrained
  models."** Already disclosed prominently in the Limitations paragraph
  ("nothing here is a claim about pretrained networks"). No fresh
  ground to attack here.
- **"No comparison to alternative architectures' learned rank on the same
  task."** Would strengthen generality, but the paper does not claim
  architecture-generality — it studies what *this* architecture converges
  to. Weak as a rejection-grade attack; noted only as a natural follow-up.
- **"$d_{\mathrm{state}} = d_{\min}+2$ is an arbitrary choice."**
  Justified in §2 ("so over-recruitment is expressible") and the causal
  grid ($d_{\min}-1, d_{\min}, d_{\min}+1$) fits cleanly inside it with
  one dimension of headroom. Not worth pressing.
- **"Welch's unpaired TOST should have been paired since S4/A5 share
  seed-index initialization."** The design doc (§1.4.2.1) explicitly
  reasons through this and picks unpaired as the conservative default
  precisely because shared-init correlation surviving 8,000 steps of
  divergent-data training is an unverified assumption. Well-argued;
  dropping this attack.
- **"The exact permutation null (8/120) is a strange choice — why not
  report $P(\rho \geq 0.9747)$ directly?"** This would make the result
  look *more* significant, not less (fewer permutations would reach the
  actual observed value than the 0.8 threshold used). Since the paper
  under-claims here rather than over-claims, this is not an attack
  surface.

## 4. New citations that should be in Related Work

- **Chughtai, Chan & Nanda, "A Toy Model of Universality: Reverse
  Engineering How Networks Learn Group Operations," ICML 2023,
  arXiv:2302.03025.** The single closest prior work not cited: trains
  networks on general finite-group composition tasks (including
  symmetric/alternating-group families) and studies which
  representations — specifically, which irreducible representations —
  the network converges to, and whether that convergence is universal
  across seeds/architectures. This paper's central question ("what
  representation does training on a group's word problem converge to")
  is directly anticipated by Chughtai et al.'s substrate and mechanism;
  the related-work section (§6) must engage it by name.
- **Nanda, Chan, Lieberum, Smith & Steinhardt, "Progress Measures for
  Grokking via Mechanistic Interpretability," ICLR 2023,
  arXiv:2301.05217.** Shows networks trained on modular arithmetic (the
  cyclic group $\mathbb{Z}/p$) learn representations built from that
  group's irreducible representations (Fourier features) — the same
  "training recovers representation-theoretic structure" phenomenon this
  paper studies, on a different (but closely related, abelian) group
  family.
- **Huh, Cheung, Wang & Isola, "Position: The Platonic Representation
  Hypothesis," ICML 2024, arXiv:2405.07987.** The paper's own opening
  framing — "when systems trained on different problems arrive at
  similar internal structure" — is close to a restatement of this
  hypothesis's title claim; a UniReps submission should position itself
  explicitly relative to it (as supporting evidence for a *structural*,
  algebra-driven variant of the hypothesis, distinct from the
  statistical/task-driven convergence Huh et al. study).
- **Moschella, Maiorca, Fumero, Norelli, Locatello & Rodolà, "Relative
  Representations Enable Zero-Shot Latent Space Communication," ICLR
  2023, arXiv:2209.15430.** A touchstone paper for the UniReps community
  specifically (shared organizer lineage); studies when and why
  differently-trained models land on comparable representational
  geometry. Should be engaged for venue fit even if the mechanism differs.
- **Li, Yosinski, Clune, Lipson & Hopcroft, "Convergent Learning: Do
  Different Neural Networks Learn the Same Representations?," ICLR 2016
  workshop, arXiv:1511.07543.** The earliest empirical precedent for the
  paper's opening question; worth a one-line acknowledgment as the
  historical starting point of this line of inquiry.
