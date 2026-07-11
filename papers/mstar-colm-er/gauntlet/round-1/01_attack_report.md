# Attack report — round 1 — mstar-colm-er

## Summary for the defense agent

This is a well-built paper: every quoted number I recomputed from the cited raw
artifacts (all nine C1 sweep files, `MSTAR_VERDICT.json`, both tap-localization
files, `TASK2DIAG_VERDICT.json`, the K48 stress file, and both sweep-harvest
n_params files) matched the draft to the last printed digit, the paired-CI
t-statistic (df=2, t=4.3027) reproduces exactly, the md5s in the brief all
verify against the files on disk, the reproducibility pipeline
(`figure-gen.py`) is real and does compute every table cell from md5-asserted
sources, the page budget renders at 7 main-text pages / 9 total (comfortably
inside the 4–10pp venue limit), and the anonymization grep is clean. The
"honest scope" sections (task2 seed-variance, K48 chance row) are not
window-dressing — I traced every number in them too and they hold up,
including one number (`0.010` for the horizon-collapse of "the first
bar-clearing seed") that only resolves correctly once you find the *second*
horizon-refs file it's drawn from; it is correct, just non-obviously sourced.

That said, two things are genuinely central and not yet defused. First, the
paper never shows or discusses its own training-loss curves, and those curves
(sitting unexamined in the same JSON files behind C1/C9) show both
non-contender arms flatlining within the first ~500 of 20,000 steps while the
contender's loss keeps falling for the whole run — a pattern that is at least
as consistent with an under-tuned baseline as with "the task is beyond this
architecture at this budget," and the schema's own gradient-health field
(`grad_ratio_at_tap_step500`) is null in every single archived checkpoint, so
there is no artifact ruling the confound out. Second, the "structure control"
(the flat-vector ablation) differs from the contender in raw state *capacity*
(512 vs 32,768 bytes — 64×) and in *whether the write is key-conditioned at
all*, not only in "matrix vs. vector." Contribution #1's licensed reading
("the outer-product update is doing task-critical work its parameter-matched
additive counterpart does not do") is not supported by an ablation that
differs on three axes at once when the paper's own framing implies it differs
on one. Both are fixable — one by disclosure/diagnostics, one by rescoping or
a new control — without touching the core empirical separation, which is
otherwise the best-verified part of the draft. The figure and one statistical
framing choice (both SERIOUS, below) also need attention before this is
submission-ready.

---

## Attacks

### A1: Both non-contender arms' training loss flatlines almost immediately; no gradient-health check backs the "it never learns" story

**Severity:** CRITICAL
**Type:** positive-control adequacy / alternative-explanation

**Attack.** The paper's central comparative claim — "a parameter-matched
flat-vector recurrence and a parameter-matched transformer... both sit at
chance" (Section 3) — rests on the assumption that all three arms received a
genuinely fair, competently-optimized training run. The `curve` field
recorded in every training JSON (e.g.
`experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_{arch}_task1_sweep_s{seed}.json`)
tells a different story than the paper reports. I extracted `train_loss` at
every 2,500-step checkpoint for all three arms, all three seeds:

| step | contender s0 | ablation s0 | transformer s0 |
|---|---|---|---|
| 500 | 7.771 | 7.833 | 7.836 |
| 3000 | 7.003 | 7.840 | 7.641 |
| 5500 | 3.853 | 7.769 | 7.459 |
| 8000 | 2.827 | 7.794 | 7.574 |
| 10500 | 1.929 | 7.797 | 7.503 |
| 13000 | 1.496 | 7.776 | 7.495 |
| 15500 | 1.420 | 7.720 | 7.406 |
| 18000 | 1.398 | 7.707 | 7.451 |
| 20000 (final) | 1.377 | 7.694 | 7.514 |

The contender's loss falls monotonically and substantially (7.77 → 1.38, a
factor of ~5.6× in the last 19,500 steps). Both baseline arms drop from
~23.8 (init) to ~7.8 in the *first 500 steps* and then are flat — noisy but
directionless — for the remaining 39 of 40 logged checkpoints (19,500 of
20,000 steps). This pattern is identical across all three seeds for both
architectures (I pulled seeds 1 and 2 too: ablation ends at 7.791/7.723,
transformer at 7.453/7.459 — same shape). A model that is "trying and failing"
at a hard task under a competent optimizer typically shows continued, if slow,
improvement or at least noisy exploration; a near-perfectly flat loss for
97.5% of a training run, identically across two architecturally unrelated
models (an elementwise-gated vector recurrence and a standard 2-layer
transformer), is the textbook signature of a stuck optimization trajectory
(dead/vanishing gradients, an LR unsuited to that architecture, missing
warmup, or an auxiliary-loss interaction) — not of "the task ceiling."

This matters doubly for the transformer arm specifically: 2-layer attention
models are the textbook example of architectures that spontaneously form
induction heads and solve exactly this class of "copy the value bound to a
just-seen key" task, often within a few thousand steps at far smaller scale
(Olsson et al., "In-context Learning and Induction Heads," Anthropic 2022,
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
A 2-layer, 14.4M-param transformer trained for 20,000 steps reading exactly
chance (0.027–0.029) on a task this class of model is famous for solving is
the kind of result that should trigger, not skip, a baseline-adequacy check.

The project's own house rule (`CLAUDE.md`, Hard Rules) states: "Smoke test
every model (forward pass, backward pass, gradient check) before training."
The training-JSON schema *has* a field for exactly this —
`grad_ratio_at_tap_step500` — and it is `null` in every one of the 9 sweep
files I checked (3 arms × 3 seeds). No gradient-health artifact exists behind
this comparison. LR (3e-4) and the aux/CE loss weighting (`aux_weight=2.0` vs.
`ce_answer_weight=1.0`) are applied identically across three architecturally
distinct mixers with no disclosed per-architecture tuning, despite it being
well established that attention and gated-recurrent mixers often need
different LR/warmup schedules.

I do not claim this *is* a bug — an aleatoric floor from the aux loss (which
appears to include prediction of freshly-randomized entity tokens, inherently
unpredictable before they're revealed) is a plausible partial explanation for
some of the flatness. But that explanation does not fully cover the
observation: the contender faces the identical objective, weighting, and
episode randomness and still drives the loss to 1.38, roughly 6 nats below
where the baselines get stuck — which shows the loss can go much lower on
this exact objective, undercutting a pure-aleatoric-floor account of why the
baselines don't move.

**Supporting evidence.**
`experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_ablation_task1_sweep_s{0,1,2}.json`
and `h2h_transformer_task1_sweep_s{0,1,2}.json`, field `curve` (40 checkpoints
each) and `grad_ratio_at_tap_step500` (null in all 9); Olsson et al. 2022
(induction heads, cited above); `CLAUDE.md` Hard Rules ("Smoke test every
model... gradient check").

**What the paper would need to do to defuse this.** Either (a) report the
training-loss curves for all three arms (a standard figure any reviewer would
expect and currently absent from the draft) alongside a gradient-norm or
update-ratio diagnostic at a few checkpoints for the two baseline arms, or
(b) run a small per-architecture LR/warmup sensitivity check for the
transformer and flat-vector arms and show the flatline persists across at
least 2–3 reasonable LR choices, or (c) explicitly scope the "non-competitive"
finding to the disclosed hyperparameter choice ("baseline non-competitive at
this LR/warmup/aux-weighting choice," not "at matched params/tokens") if no
such check is run before the deadline.

---

### A2: The "structure control" ablation confounds state capacity (64×) and content-addressability with matrix structure

**Severity:** CRITICAL
**Type:** positive-control adequacy / alternative-explanation / missing-citation

**Attack.** Section 2.2 introduces the flat-vector arm as "the structure
control," describes its update as
$s_t = s_{t-1}\odot g_t + v_t$ with read $o_t = s_t \odot q_t$, and Section 3
draws this "licensed" conclusion: "the outer-product state update is doing
task-critical work its parameter-matched additive counterpart does not do,
since the flat-vector arm differs only in the mixer recurrence" (emphasis
mine — "only"). This is not true on the evidence in the paper's own setup
section, on two independent counts:

1. **State size is unmatched by 64×.** The contender's recurrent state is
   $2\times64\times64$ fp32 = 32,768 bytes; the ablation's is $2\times64$ fp32
   = 512 bytes (both numbers confirmed against C10/C9 raws). The paper
   anticipates the obvious fix — reshape a $d^2$-vector to match — and
   explicitly rejects it: "rather than reshaping the same operator, since any
   $d^2$-dimensional vector can encode a $d\times d$ matrix and structure
   matters only if the operations use it." But rejecting the reshape solution
   does not make the size confound disappear; it just means the paper chose
   to confound on *capacity* instead of leaving open the (real) risk that a
   reshaped ablation could trivially recover matrix behavior. Either choice
   is a confound; the paper only argues why the alternative confound would
   have been worse, not why 64× less raw memory is safe to fold into a claim
   about "outer-product update... doing task-critical work." A model with
   64× fewer state bytes losing at a recall task that scales with $K/d$
   (as the paper's own Table 3/K48 result shows recall load-boundedness by
   *state size*) is the expected, unsurprising result regardless of any
   matrix-specific mechanism.
2. **The write rule has no key-conditioning at all.** $s_t = s_{t-1}\odot g_t
   + v_t$ writes $v_t$ into the state with no dependence on $k_t$ shown
   anywhere in the formula. This means the ablation is not just "a vector
   instead of a matrix" — it structurally cannot perform content-addressed
   binding under *any* training outcome, because there is no mechanism by
   which the write could be selective on which of the 32 keys is present.
   This is a fundamentally different and much less interesting ablation than
   "does matrix structure help," and the paper never states this explicitly
   nor cites the literature on vector-native binding that this ablation
   fails to instantiate (see missing citations below). A vector control that
   *can* do key-conditioned binding (e.g., circular-convolution / holographic
   binding, Plate 1995) failing would be informative about matrix structure
   specifically; a vector control that structurally cannot bind at all,
   failing, tells us little beyond "you need a content-addressable write,"
   which is not new.

Together, these mean Section 3's "licensed" sentence is not licensed by the
comparison as constructed — the separation is at least as well explained by
"has 64× more state + any content-addressed write" as by "has an
outer-product/matrix update." The empirical separation itself (accuracy
numbers) is not in question; the *interpretation offered for it* is.

**Supporting evidence.** `sections/02_setup.tex` lines describing the
flat-vector arm; C9/C10 byte accounting; Plate, T.A. (1995), "Holographic
Reduced Representations," IEEE Trans. Neural Networks, 6(3):623–641 (vector
binding via circular convolution — exactly the kind of vector-native,
content-addressable binding this ablation does not implement); Smolensky, P.
(1990), "Tensor Product Variable Binding and the Representation of Symbolic
Structures in Connectionist Systems," Artificial Intelligence 46(1-2):159-216.

**What the paper would need to do to defuse this.** Either (a) rescope the
Section 3 "licensed reading" to something the comparison actually supports —
e.g., "a matrix state with more raw capacity and a multiplicative,
key-conditioned write outperforms a smaller, non-content-addressable vector
state" — and move the size/write-rule gap into an explicit disclosed
confound in Limitations, dropping the "structure control" name, or (b) add a
second ablation that holds state *bytes* fixed at 32,768 (e.g., a
$d=4096$ diagonal/elementwise-gated vector, rebalancing other widths to stay
param-matched) and/or gives the vector write access to $k_t$ (e.g., an HRR
circular-convolution bind), so "matrix structure" is isolated from "more
memory" and "has any content-addressed write."

---

### A3: The money figure (Fig. 1) does not actually show the capped-KV data it is captioned to show

**Severity:** SERIOUS
**Type:** presentation / positive-control adequacy

**Attack.** The Figure 1 caption promises: "the transformer reads chance at
every horizon, both uncapped (dashed) and under sink-plus-FIFO KV caps at
$M \times 32{,}768$ bytes for $M \in \{2, \ldots, 32\}$ (**crosses**)." I
rendered `figures/fig1_horizon.pdf` directly: there is a solid black line
(contender, ~1.0) and a dashed red line (uncapped transformer, ~0.03) — no
gray "×" markers are visible anywhere in the plot, despite the legend listing
them. Reading `figure-gen.py` (`fig1_horizon`, lines 141–151) confirms the
crosses *are* plotted (`ax.plot(t, by_h[h][s], ..., marker="x", ..., zorder=3,
...)`), but at `zorder=3`, strictly *below* the uncapped-transformer dashed
line at `zorder=4` — and all of the capped-M values (0.020–0.033, per Table 4)
sit in the same ~0.01-wide band as the uncapped readings (0.027–0.036) on a
y-axis spanning $-0.04$ to $1.06$. The crosses are real data, occluded by the
opaque dashed line drawn on top of them and compressed to a few pixels by the
axis scale. As rendered, a reader cannot verify "no cap rescues the baseline"
from the money figure itself — only from the appendix table. For a figure the
paper's own brief calls out as the "MONEY FIGURE," this is a real defect: a
skeptical reviewer's first move is often "show me the capped points in the
figure," and right now they aren't visible.

**Supporting evidence.**
`papers/mstar-colm-er/figures/fig1_horizon.pdf` (rendered and visually
inspected); `papers/mstar-colm-er/figures/figure-gen.py` lines 141–151
(zorder=3 crosses vs. zorder=4 dashed line, zorder=5 solid line).

**What the paper would need to do to defuse this.** Raise the crosses'
zorder above the dashed line (or draw the dashed line with partial alpha),
and/or add a small inset/zoomed sub-panel over the 0–0.1 accuracy band so the
cap sweep is visibly distinguishable from the uncapped reading, then note in
the caption that all capped and uncapped readings visually coincide near
chance.

---

### A4: The "$S_1$-zeroing leaves it unchanged" statistical framing for seed 1 may have the causality backwards

**Severity:** SERIOUS
**Type:** statistical

**Attack.** Section 5 discloses: "at a seed reading exactly 1.0 intact, the
binomial band used for the 'unchanged' check degenerates to zero width, so
the 0.0051 dip at seed 1 under $S_1$-zeroing is disclosed as an edge case of
the check rather than adjudicated by it." I pulled the raw field
(`h2h_contender_task1_sweep_s1_round4.json`, `s0_necessity_check`):
`acc_intact=1.0`, `acc_s1_zeroed=0.994873`, `delta_s1=0.005127`, `sigma=0.0`,
`two_sigma=0.0`, `unchanged=False`, `passed=False`. The paper's framing treats
the zero-width band as a broken instrument to be waved off. But note what
`sigma=0.0` actually means here: with $p=1.0$ intact accuracy on $n=4096$
fixed, deterministic (argmax) queries, the *expected* delta under a
true-null ("$S_1$ is inert") is not "small but nonzero within some tolerance"
— it is *exactly* zero, because there is no stochastic sampling process
generating the intact-vs-zeroed comparison; both are deterministic forward
passes on the same fixed checkpoint and fixed query set. A reproducible,
deterministic flip on 21 of 4,096 queries (0.0051 × 4096 ≈ 21) when $S_1$ is
zeroed is not measurement noise inflating a CI — it is a real, repeatable
change in the model's output caused by the intervention. The paper's
explanation ("edge case of the check") is precisely backwards: the band isn't
failing to protect against a real effect, it's correctly reporting that *any*
nonzero delta is meaningful when the reference distribution truly has zero
variance. This is a small effect (21/4096 queries) that does not threaten
the headline S0-necessity claim, but the "$S_1$ is causally inert" sub-claim
for this specific seed is not actually established the way the prose implies.

**Supporting evidence.**
`experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s1_round4.json`,
field `s0_necessity_check` (`sigma: 0.0`, `unchanged: false`, `passed:
false`); `sections/05_mechanism.tex` lines 30–33.

**What the paper would need to do to defuse this.** Replace the
binomial-band heuristic (which is ill-posed at $p=1$) with a test suited to
the deterministic, paired nature of the comparison — e.g., report the raw
count of discordant queries (21/4096) and whether they are reproducible
across a repeated forward pass or a bootstrap over query subsets, rather than
asserting the check itself is the artifact. If the 21 queries are indeed
noise-adjacent (e.g., near-tied logits), say so with evidence; if they are a
consistent small subset, disclose that $S_1$ has a small nonzero causal
contribution for that seed rather than calling it "unchanged."

---

### A5: The paper's motivating framing (deployment-scale KV-cache growth) is tested at a horizon three to four orders of magnitude below where that framing applies

**Severity:** SERIOUS
**Type:** claim-scope

**Attack.** The introduction opens: "Serving a transformer over a long
context is a memory problem before it is a compute problem... at deployment
scale that growth, not arithmetic throughput, sets the limit on batch size,
context window, and cost." This is the paper's motivating stakes, and it is
true — but it is true at the context lengths (tens of thousands to
millions of tokens) where production KV-cache compression work
(StreamingLLM, H2O, both cited) actually operates. The paper's own longest
tested horizon is 1,798 tokens (C3) — "eight times the binding span." Nothing
in the Limitations section flags the specific gap between the motivating
scale (where KV-cache cost is a genuine deployment bottleneck) and the tested
scale (a small multiple of a 224-token synthetic binding phase). The
Limitations section says "nothing here speaks to pretrained models, natural
text, or other scales," which is a scale disclaimer for model *size*, not
explicitly for *context length* — the two are conflated. A reviewer working
on long-context serving would reasonably ask whether the "constant-memory"
demonstration would still hold if the same episode were embedded at, say,
32K or 128K tokens (i.e., at scales where StreamingLLM/H2O-style compression
is actually deployed), and the paper as written gives no evidence either way.

**Supporting evidence.** `sections/01_intro.tex` opening paragraph;
`sections/08_limitations.tex` (no explicit context-length-vs-motivation
disclaimer); `sections/04_horizon.tex` (max horizon 1,798 tokens).

**What the paper would need to do to defuse this.** Add one sentence to
Limitations explicitly scoping the "constant-memory" demonstration to the
tested horizon band (≤1,798 tokens, ≤8× the binding span) and noting this is
far short of the context lengths where KV-cache cost is a practical
deployment constraint — i.e., the paper demonstrates the *mechanism*
(byte-constant state, no decay to 8×) as an existence result, not a
claim that it holds at deployment-relevant context lengths.

---

### A6: "A blank-out test... enforces this by construction" is asserted without a reported result

**Severity:** SERIOUS
**Type:** number provenance / reproducibility

**Attack.** Section 5 states: "the decoder reads only the state, never the
raw bind tokens, and a blank-out test on the continuation path enforces this
by construction." This is a load-bearing methodological claim — it is what
licenses treating the S0/S1-zeroing intervention as a clean causal read
rather than one confounded by the decoder secretly attending back to raw
bind-phase tokens. Every *other* claim in this paper — including much less
central ones (task2's 9-seed table, the K48 stress row) — carries a specific
`% <!-- evidence: Cx -->` tag pointing to a raw artifact with a reported
number. This sentence has no such tag and no adjacent quoted metric (e.g., a
pass/fail result, an accuracy delta under a forced-corruption test). As
written, it is asserted architecturally ("by construction") rather than
empirically verified and reported, which is inconsistent with the paper's
own Appendix C promise that "every inline numerical claim in the source
carries a machine-checkable comment naming its row in a claims-to-evidence
map." This one isn't a numerical claim, technically, which is presumably why
it's exempt — but for a paper whose central rhetorical move is "everything
here traces to evidence," a structural claim this load-bearing deserves the
same treatment.

**Supporting evidence.** `sections/05_mechanism.tex` lines 16–20; `brief.md`
claims-to-evidence table (no row corresponds to "blank-out test" specifically
— it is presumably folded into the HTH §1.30 localization protocol
narratively, but not quotable from an artifact in the same way C5's numbers
are).

**What the paper would need to do to defuse this.** Either report the
blank-out test's actual result (e.g., "corrupting raw bind tokens
post-encoding while holding the state fixed changes accuracy by
$\leq X$", with an evidence tag) or soften the sentence to describe the
architectural guarantee without implying an empirical test was run and
passed, if in fact only the architecture (not an explicit corruption test)
is doing the work.

---

### A7: Three seeds is thin for the primary comparison by typical ML replication norms

**Severity:** MINOR
**Type:** statistical

**Attack.** The primary axis-1 comparison (Table 1, C1/C2) uses $n=3$ seeds
per arm. This is generically underpowered by common replication standards
(5+ seeds is a more typical bar for a headline comparison). In this specific
case the effect size is large enough (accuracy gap ≈0.97 against
seed-to-seed variance on the order of $10^{-3}$) that the resulting CIs are
tight regardless, and the registered protocol includes a seed-extension
trigger for marginal cases, so this doesn't threaten the conclusion — but a
reviewer counting seeds will still note it, especially since the paper's own
task2 analysis (correctly) uses 9 seeds once trainability variance is a live
concern, implicitly conceding that 3 seeds can be too few for this task
family.

**What the paper would need to do to defuse this.** A one-line acknowledgment
in Limitations that $n=3$ is used for the primary comparison because the
pre-registered protocol's seed-extension trigger did not fire (no seed came
close to straddling the margin), with a pointer to the trigger's exact
threshold, would preempt this.

---

### A8: No multiple-comparison discussion across the paper's many pre-registered tests

**Severity:** MINOR
**Type:** statistical

**Attack.** The paper runs a substantial number of statistical tests across
sections: 2 paired CIs (C2), 5 per-$M$ paired-gap CIs (C4), a pooled paired CI
plus a batch-effect gate (C7), and implicit accuracy-vs-bar comparisons per
seed per arm across two tasks. None of these are adjusted for multiple
comparisons. Given the effect sizes involved (differences of 0.9+ accuracy
against seed variance of $10^{-3}$–$10^{-2}$), no plausible correction would
flip a conclusion, so this is not a live threat to any specific claim — but a
statistically careful reviewer may ask for an explicit acknowledgment.

**What the paper would need to do to defuse this.** A one-sentence note (e.g.
in the Analysis protocol subsection) that the primary decision rule is a
single pre-registered comparison and the remaining reported intervals are
descriptive/diagnostic, not additional hypothesis tests requiring
correction, would close this off.

---

## Attacks I considered but decided were weak

- **The `0.010` horizon-collapse number in Section 6 doesn't trace to a raw
  artifact.** I initially could not find this number in
  `2026-07-10_h2h_task2diag/results/contender_task2_s5s7_horizon_refs.json`
  (which only covers seeds 5 and 7, reading 0.0005 and 0.0071–0.0076 — neither
  rounds to 0.010). On a second, wider search I found it in
  `2026-07-10_h2h_mstar/fanout/contender_horizon_refs.json`
  (`task2_sweep|s2|H2/H4/H8` = 0.010009765625 / 0.010009765625 /
  0.01025390625), which is seed 2 — the actual first (lowest-index)
  bar-clearing seed among the nine. The number is correct; I'm noting this
  here because it took real digging (the natural-seeming source file for a
  "horizon" claim in the task2diag results directory is the wrong one), and a
  less careful check could have flagged this as a fabricated number. It
  isn't.
- **Anonymization / venue-fit / page-budget non-compliance.** Grepped the
  anonymization token list from the brief across all `.tex` sources: zero
  hits. Rendered page count: 9 total (7 main text pages through the end of
  Conclusion, 1 page of references, 1 page holding all three appendix
  sections) — comfortably inside the workshop's 4–10 main-text-page limit
  and the brief's own 10-page hard cap. Not an attack.
- **Bibliography accuracy.** `refs.bib` carries a comment claiming every entry
  was checked against the arXiv API and Schmidhuber 1992 against CrossRef. I
  did not independently re-verify each arXiv ID against the live API, but
  spot-checked that the arXiv IDs correspond to titles/authors I recognize as
  correct from training knowledge (e.g. 2412.06538 = Nichani/Lee/Bietti,
  2406.06484 = Yang et al. parallelizing DeltaNet, 2312.04927 = Zoology).
  No mismatches found. Weak as a standalone attack given the paper's own
  documented verification pass.
- **Episode-restricted (K-way) argmax metric inflates apparent accuracy
  relative to open-vocabulary decoding.** True, but fully disclosed (chance
  = 1/K stated everywhere, Nichani caveat travels with the metric
  definition) and standard practice in this literature (matches Zoology's
  own evaluation convention). Downgraded to a MINOR framing note rather than
  a real attack; a supplementary open-vocab top-1 robustness number would
  strengthen but its absence isn't disqualifying.
- **fp32 KV-cache byte accounting vs. deployment-standard fp16/bf16.** The
  paper computes both the fast-weight state and the transformer's KV cache in
  fp32 bytes, which is internally consistent, and it explicitly declines to
  quote any memory-multiplier claim (per the pre-registered degenerate-
  baseline clause), so the precision choice has no live rhetorical
  consequence. Would matter if the paper later claimed a byte-for-byte
  deployment comparison; it currently doesn't.

---

## New citations that should be in Related Work

- **Plate, T. A. (1995), "Holographic Reduced Representations," IEEE
  Transactions on Neural Networks, 6(3):623–641.** Directly competes with the
  paper's flat-vector "structure control": HRR shows a single vector *can*
  store and retrieve multiple key-value bindings via circular convolution, a
  content-addressable write the paper's ablation does not implement. Needed
  to support (or to honestly scope around) the A2 attack above.
- **Smolensky, P. (1990), "Tensor Product Variable Binding and the
  Representation of Symbolic Structures in Connectionist Systems," Artificial
  Intelligence, 46(1-2):159-216.** Same relevance as Plate 1995 — a second,
  earlier vector-native binding mechanism the ablation could have used
  instead of a non-key-conditioned gated sum.
- **Yang, S. et al., "Gated Delta Networks: Improving Mamba2 with Delta
  Rule," arXiv:2412.06464 (2024/2025, ICLR 2025).** A direct extension of the
  exact delta-rule update the contender uses, adding a gating term; a close
  competitor on the mechanism that the current Related Work (which stops at
  Yang et al. 2024's parallelization paper) does not engage.
- **Behrouz, A., Zhong, P., Mirrokni, V., "Titans: Learning to Memorize at
  Test Time," arXiv:2501.00663 (2025).** A prominent, very recent paper on
  exactly this paper's framing — a fixed-size neural memory module meant to
  retain long-context information without a growing KV cache. A workshop
  reviewer in this space would expect this to be engaged with or
  distinguished from.
- **Arora, S. et al., "Simple linear attention language models balance the
  recall-throughput tradeoff" (Based), arXiv:2402.18668 (2024).** Direct
  follow-up to the already-cited Zoology paper, from the same group, on
  recall-vs-efficiency tradeoffs in linear-attention/recurrent hybrids —
  natural companion citation next to `arora2023zoology`.
- **Olsson, C. et al., "In-context Learning and Induction Heads," Anthropic
  Transformer Circuits Thread, 2022.** Not primarily a "competitor" citation,
  but directly relevant context for why a 2-layer transformer's complete
  failure on a copy/recall task (A1 above) is a surprising result worth
  addressing head-on rather than only citing Zoology's opposite-direction
  headline.
