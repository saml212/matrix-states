# Attack Report — Round 1 — "Three Bounds on a Null"

## Summary for the defense agent

This is a well-engineered null-result paper: I recomputed every headline number
(C0 through C9) directly from the raw JSON archives, and every one of them
matches the draft exactly, including several derived quantities (the naive
n=12 pool CI, the mean familiarization effect 1.69, the GPU-hour ledger).
That discipline is real and should be preserved. But the paper's actual
weak point is not "did the numbers get transcribed correctly" — it's
**where the paper's one positive, non-null finding (the wikitext × per_token
"transient") comes from**. I traced it into the design registry
(`REASONING_LINK_DESIGN.md` §16.18.3) and found that the pre-registered
analysis *pipeline* — the code that was actually fixed before unblinding —
computed only two corpus-level verdicts (both UNRESOLVED, using the global
arm as a representative for the corpus). The "4 (corpus, arm) contrasts"
framing and the specific promotion of one cell to TRANSIENT is a **build-time
reanalysis performed after the registered pipeline's output was seen and
judged to be masking something** — the design doc says so explicitly, in
its own words. Two of the paper's three headline bounds (the power bound and
the replication bound) are organized entirely around this one cell. Compounding
this, the underlying statistics show the "transient" is not distinguishable
from noise among the ~40 uncorrected confidence intervals the wave computes,
and two *other* cells hit the same "CI excludes zero" bar at the same
checkpoint under a different corpus/arm combination — a pattern the paper
never discusses. These two findings (A1, A2) are CRITICAL: they attack the
credibility of the one thing in this paper that isn't a flat zero, and that
thing is what two of the three bounds are about. The instrument-null bound
(Bound 1, §3) is comparatively solid — I could not break the 366/366 zero
claim or the gate-failure counts, though I found a real gap in that story too
(A6: no real-kernel positive control for the readout, only a CPU-stub unit
test). The related-work section is thin on the one literature that bears
most directly on the paper's own task (DeltaNet's proven ability to learn
permutation composition, and SSMs' proven state-tracking ceiling) — both are
real, on-point, and currently absent (A5). Salvageable: yes, with (1) an
explicit, prominent disclosure of the corpus-level-vs-per-arm scoping change
and a demotion of "the single determinate signal" language to something
that survives a multiple-comparisons reading, (2) the two missing citations,
and (3) a sentence acknowledging the readout's positive-control gap. None of
this requires new experiments — it requires the paper to say what the design
doc's own §16.18.3 already admits.

---

## Attacks

### A1: The paper's one non-null finding is a post-hoc reanalysis, not the pre-registered pipeline's output

**Severity:** CRITICAL
**Type:** unfalsifiable/post-hoc reasoning dressed as pre-registration

**Attack.** Section 4 states: "Three of four (corpus, arm) contrasts classify
as unresolved... The fourth (encyclopedic corpus, per-token arm) classifies
as transient." Table 1's Wave-5 row states "18 cells, 4 contrasts | 3
unresolved, 1 transient." This "4 (corpus, arm) contrasts, independently
classified" framing is presented as the pre-registered analysis. It is not.

I read `PHASE2B_SUMMARY.json` (`experiment-runs/2026-07-08_phase2b/results/`)
directly: its own `"trajectories"` field — the actual output of the
registered analysis pipeline — contains exactly **two** entries,
`openr1-mix-ext` and `wikitext-mix-ext`, and **both are classified
UNRESOLVED**. There is no per-arm classification field anywhere in that
summary. I then read the design registry's own account of why, at
`REASONING_LINK_DESIGN.md` §16.18.3 ("THE FOUR PRIMARY (corpus × arm)
CONTRASTS — a build-time scoping finding, disclosed"):

> `phase2_trajectory_analysis.analyze_corpus` (L312-321) computes only ONE
> hexachotomy classification per corpus... using the **global** arm's own
> `holds_by_c` pattern as the corpus's representative signal... `per_token`'s
> own `holds_by_c` pattern... is **never consulted for the registered
> corpus-level verdict** except at the single terminal checkpoint... this
> section supplies the 4-way breakdown by applying the SAME
> `classify_trajectory` primitives... to EACH arm's own `holds_by_c`...
> [wikitext × per_token] independently classifies TRANSIENT... before the
> classifier ever reaches the #4/#5 branch **the registered pipeline's
> global-representative scoping used to fold this cell into UNRESOLVED**.

And immediately after: "**Registered as a follow-up fix, not self-authorized
here**... a future revision of `analyze_corpus` should classify BOTH arms'
own `holds_by_c` independently per corpus (4 verdicts, not 2)... this
harvest's own §16.18.3 table is the concrete evidence that the current
scoping **can mask a real, non-noise signal**."

In plain terms: the code that was pre-registered and actually run produced
two UNRESOLVED verdicts. Someone then noticed, after looking at the
per-arm data the pipeline had computed but not surfaced, that a different
level of aggregation would produce a more interesting result, wrote a new
per-arm analysis at harvest time, and that new analysis's output — not the
registered pipeline's — is what the paper reports as "the fourth contrast."
This is the exact pattern the attack brief singles out: "unfalsifiable or
post-hoc reasoning dressed as pre-registration." The underlying `holds_by_c`
arithmetic is independently verified correct (§16.18.2) — my objection is
not that the numbers are wrong, it's that **the decision to report at
per-arm granularity, rather than the registered corpus-level granularity,
was made after seeing that the registered output was UNRESOLVED**. That is
precisely the degree of freedom pre-registration exists to close.

This matters because the "transient" is not a footnote: it is the sole
non-zero, non-unresolved finding in the entire paper, and Bound 2 (§4,
"power-bounded... one bounded transient") and the *entire* Bound 3 / §5
(the twelve-seed replication wave, Figure 3 right panel) exist only to
characterize and re-test this one cell. Nowhere in `main.tex` or the
appendix is the corpus-level-vs-per-arm scoping change disclosed — I grepped
every section file and the appendix for "mask," "scoping," "representative,"
"global-arm," "build-time," "corpus-level" and found zero hits.

**Supporting evidence.** `experiment-runs/2026-07-08_phase2b/results/PHASE2B_SUMMARY.json`
(`"trajectories"` field, both corpora UNRESOLVED); `REASONING_LINK_DESIGN.md`
§16.18.3 (quoted above, the design doc's own words); §16.18.6 ("THE KEYSTONE
VERDICT") which lists the per-arm 4-way breakdown as the harvest's finding,
not the registered pipeline's.

**What the paper would need to do to defuse this.** Add one disclosure
sentence to §4 (and ideally Appendix A, which is supposed to reproduce "the
pre-registered gates and routing rules the main text relies on" but currently
omits this entirely): state plainly that the registered corpus-level pipeline
classified both corpora UNRESOLVED using the global arm as representative,
and that the per-arm TRANSIENT reading is a disclosed build-time
re-derivation, not the registered verdict. This does not require retracting
the finding — the underlying CI arithmetic is independently verified — but
it does require the paper to stop presenting "4 (corpus, arm) contrasts,
independently classified" as if that were the pre-registered unit of
analysis.

---

### A2: The "single determinate signal" is not distinguishable from multiple-comparisons noise, and the raw data show an unaddressed shared-checkpoint alternative explanation

**Severity:** CRITICAL
**Type:** statistical / alternative-explanation

**Attack.** Wave 5's primary table computes 95%-CI "determinate" tests on
4 (corpus × arm) contrasts × 5 checkpoints × 2 K-loads = **40 independent
CI tests**, with no multiple-comparisons correction anywhere in the design
or the draft. Under a global null (no true effect anywhere), 40 tests at
nominal α=0.05 predict **≈2 false "determinate" (CI-excludes-zero) readings
by chance alone**.

I pulled every `det32`/`det20` value from both corpora's raw trajectory
JSONs (`trajectory_openr1-mix-ext_phase2b.json`,
`trajectory_wikitext-mix-ext_phase2b.json`) and found **three**, not one,
K=32 cells where the CI excludes zero — almost exactly the chance
expectation, not a dramatic excess:

| Corpus × Arm | Checkpoint | Δ mean (K=32) | CI | det32 |
|---|---|---|---|---|
| wikitext × per_token | c=1000 | −0.168 | [−0.301, −0.036] | **True** |
| wikitext × per_token | c=2500 | −0.500 | [−0.624, −0.376] | **True** (the reported "transient") |
| openr1 × global | c=1000 | −0.203 | [−0.402, −0.004] | **True** |

The paper's own sign-discipline sentence in §4 — "every determinate
high-load reading in the primary table is negative" — silently
acknowledges the plural (there are three, not one), but the paper never
reports the count, never applies a multiple-comparisons correction, and
never discusses the second cell it elevates to nothing: `openr1-mix-ext ×
global` at c=1000 is *equally* a CI-excluding-zero, harm-direction K=32
reading, at the *same checkpoint* as the wikitext × per_token hit that
almost triggers the differential condition there too (its K=20 CI, [−0.325,
−0.074], is itself determinate, which is the *only* reason the registered
differential condition doesn't also fire on it).

This is a genuine alternative explanation the paper does not rule out: two
of the three "hits" cluster at the **same checkpoint** (c=1000) in **two
different (corpus, arm) pairs**. A shared checkpoint-level artifact —
e.g., a curriculum/data-order effect that happens to make the `off`
(control) arm's own loss transiently lower at that point in training,
independent of which arm is being contrasted against it — is at least as
parsimonious an explanation as "the per-token causal package transiently
hurts acquisition," and it would explain the *checkpoint*-clustering the
"one real transient" framing cannot: if this were a genuine per_token-arm
effect, there's no reason it should also show up (nearly) at c=1000 in the
unrelated openr1 × global cell. The paper's own §5 discussion comes close to
this reasoning for the *variance* mismatch ("most consistent with small-n
variance imprecision") but never extends it to the *mean* pattern across
cells, and never reports the 40-test denominator that would let a reader
judge whether 3 hits is remarkable (it is not).

**Supporting evidence.** Computed directly from
`experiment-runs/2026-07-08_phase2b/results/trajectory_openr1-mix-ext_phase2b.json`
and `trajectory_wikitext-mix-ext_phase2b.json`, `per_arm.{global,per_token}.raw`
blocks, all 4×5=20 K=32 cells and 20 K=20 cells inspected directly (script
run this session, not read from a summary).

**What the paper would need to do to defuse this.** State the denominator
(40 uncorrected tests) explicitly, report all three CI-excluding-zero
cells (not just the one satisfying the differential condition), and address
the checkpoint-clustering pattern — either with a multiple-comparisons-aware
statement ("consistent with chance at this test count") or with a positive
argument for why c=1000's openr1×global hit is unrelated. As written, "the
single determinate signal... a transient deviation in the harm direction"
overstates how isolated this reading is.

---

### A3: The batch-effect gate's variance-ratio cutoff (4.0) is an undefended round number, and a proper F-test shows the observed ratio is only marginally more extreme than chance at these degrees of freedom

**Severity:** SERIOUS
**Type:** statistical

**Attack.** §5 and Appendix A present the replication gate as a rigorous
integrity check: "cohort variance ratio 4.47 against a pinned cutoff of 4.0
(twelve percent past the cutoff)." I searched `REASONING_LINK_DESIGN.md`
for any derivation of "4" as a variance-ratio threshold and found none — it
first appears, unexplained, as `variance_ratio > 4` in the MINOR-1 fix entry
(line 9801: "flag if `|mean(new_off) − mean(old_off)| > 2 × pooled_SE` or
`variance_ratio > 4`"). The mean-shift half of the gate (`2 × pooled_SE`) is
a recognizable ≈95% heuristic; the variance-ratio half has no analogous
justification anywhere in the document — no F-distribution critical value,
no citation, no simulation calibration. This is the same "picked by feel"
pattern the design doc itself flags elsewhere as a MINOR issue (e.g. its own
data-mix-fraction MINOR-1 at line 5549), just not flagged here.

I computed the appropriate reference distribution myself: the old cohort has
n=3 (df=2), the new cohort n=9 (df=8). Under H0 (equal population
variances), `var_old/var_new ~ F(2,8)`. The observed ratio is 4.4715
(recomputed from `sd_old=1.1543176522782124`, `sd_new=0.5458814268155713` in
`trajectory_seedext_wikitext_n12.json`'s `batch_gate` block — matches the
paper's 4.47 exactly). Against `F(2,8)`:

- One-sided p-value (P(F ≥ 4.4715)) ≈ **0.0497**
- Two-sided p-value ≈ **0.099**
- The one-sided 5% critical value for F(2,8) is **4.459** — almost exactly
  equal to the observed 4.4715, and *higher* than the design's pinned
  cutoff of 4.0.

So the "pinned cutoff of 4.0" is actually *more permissive* than a properly
calibrated one-sided 5% F-test would be (a real 5% test would need F≥4.459,
essentially the observed value), and a genuinely conservative two-sided test
at the same df pair would need F≥6.06 — nowhere close to 4.47. In other
words, at df=(2,8), a variance ratio this large is not a rare event under a
true null; it is within the range one should expect from ordinary sampling
noise in a 3-sample variance estimate roughly 1 time in 10 (two-sided) or 1
time in 20 (one-sided). The paper's own closing sentence in §5 ("most
consistent with small-n variance imprecision in the archived cohort")
already half-concedes this, which makes the earlier framing — "the gate
fired," "twelve percent past the cutoff," "stopped by its own... integrity
gate" — read as more decisive than the underlying statistics support. This
does not change the substantive conclusion (the naive pool CI spans zero
regardless of whether the gate fires), but it does mean the gate's
"pinned cutoff of 4.0" language is dressed with more precision than it has.

**Supporting evidence.** `trajectory_seedext_wikitext_n12.json`
(`primary.2500.delta_k32.batch_gate`); `REASONING_LINK_DESIGN.md` line 9801
(MINOR-1, `variance_ratio > 4`, no derivation given); F(2,8) computation
performed this session via a pure-Python regularized-incomplete-beta
implementation, cross-checked against the closed-form F↔Beta relation.

**What the paper would need to do to defuse this.** Either (a) derive the
4.0 cutoff from an actual F-test at a stated α (and note it would need to be
≈4.46 for a genuine one-sided 5% test at these df), or (b) soften the
framing to "a heuristic pre-registered threshold, not a calibrated
significance test" and let the already-present "small-n variance
imprecision" sentence carry the epistemic weight it deserves.

---

### A4: Okpekpe & Orvieto (2025) is mischaracterized in Related Work

**Severity:** SERIOUS
**Type:** missing/mischaracterized citation

**Attack.** §6 states: "`\citet{okpekpe2025revisiting}` give a positive
causal account of recall differences in modern recurrent models; this paper
makes no positive mechanism claim and instead bounds one named mechanism
three ways." I checked the actual paper (arXiv:2508.19029, "Revisiting
associative recall in modern recurrent models," Okpekpe & Orvieto). Its
abstract states its actual contribution: the choice of **learning rate**
plays a critical role in recurrent-model performance on associative recall,
"an issue that can severely affect reported performance in previous works" —
i.e., the paper's finding is that **optimization instability is a
methodological confound** in prior SSM-vs-Transformer recall comparisons,
not a mechanistic account of *why* recurrent models recall differently.
Describing this as "a positive causal account of recall differences" recasts
a paper about hyperparameter-sensitivity-as-confound into a paper about
architecture-level mechanism — a materially different claim, and ironically
the actual paper's finding (naive comparisons can be artifacts of tuning) is
close in spirit to this draft's own methodological carefulness, which makes
the mischaracterization more visible to a reviewer who knows the paper.

**Supporting evidence.** arXiv:2508.19029 abstract (verified live,
2026-07-10): "we first demonstrate that, unlike standard transformers, the
choice of learning rate plays a critical role in the performance of modern
recurrent models... an issue that can severely affect reported performance
in previous works."

**What the paper would need to do to defuse this.** Rewrite the one-clause
characterization: e.g., "Okpekpe & Orvieto (2025) show that optimization
instability, not architecture alone, explains much of the reported
Transformer-vs-SSM recall gap; this paper's bounds concern a different
axis (write-geometry validity), not recall throughput or training
stability."

---

### A5: Two directly on-point papers about DeltaNet/SSM state-tracking and permutation composition are missing from Related Work

**Severity:** SERIOUS
**Type:** missing-citation

**Attack.** The bind/query task in §2 is, structurally, exactly a
**permutation-composition tracking task**: entities are bound into a
directed K-cycle and queried at h hops around it. There is a specific,
recent theoretical literature on whether DeltaNet-family models can
represent and compose permutations in their state, and it is absent from
§6:

1. **Grazzi, Siems, et al., "Unlocking State-Tracking in Linear RNNs Through
   Negative Eigenvalues," ICLR 2025, arXiv:2411.12537.** This paper studies
   DeltaNet specifically (not just SSMs generally) and proves results about
   which linear RNNs can and cannot track state-tracking problems including
   **permutation composition** — showing standard DeltaNet's positive-
   eigenvalue state-transition structure is provably insufficient for
   certain group-tracking problems (e.g. parity) while richer
   (negative-eigenvalue / Householder-product) variants can compose
   permutations. This is arguably the single most relevant piece of prior
   work to this paper's own task and model family, and it offers a
   *theoretical* candidate explanation for the null (an expressivity
   limitation of the specific DeltaNet variant under test) that competes
   with the paper's own "wrong observable, not wrong capability" framing in
   §7's "Not bounded" paragraph.
2. **Merrill, Petty & Sabharwal, "The Illusion of State in State-Space
   Models," ICML 2024, arXiv:2404.08819.** Proves S4/Mamba-class SSMs are
   confined to TC0 and cannot solve state-tracking problems like permutation
   composition or entity tracking in narratives — a general theoretical
   ceiling on exactly the capability class this paper probes empirically.

Both were confirmed live (2026-07-10) to exist, be correctly attributed, and
be on the general topic claimed above. Their omission is notable because
the paper's Related Work already engages the recall-throughput literature
(Based, Zoology) but skips the state-tracking-theory literature that speaks
most directly to "can a fixed-state model compose a queried relation," which
is this paper's own framing question in the introduction.

**Supporting evidence.** arXiv:2411.12537 (Grazzi et al., ICLR 2025,
verified live); arXiv:2404.08819 (Merrill et al., ICML 2024, verified live).

**What the paper would need to do to defuse this.** Add one or two sentences
to §6 positioning against Grazzi et al. in particular — since it is
DeltaNet-specific and directly bears on whether a null on this exact task
family should be theoretically expected for the specific DeltaNet
configuration trained here (frozen-key-bias arms, no Householder-product or
negative-eigenvalue modification), this citation could materially change how
a reviewer reads the paper's own "not bounded... cross-layer hand-off"
speculation in §7.

---

### A6: No real-kernel positive control for the geometric readout — only a CPU-stub unit test on a hand-built matrix

**Severity:** SERIOUS
**Type:** positive-control adequacy

**Attack.** The entire Bound-1 argument (366/366 zero readings, §3) rests on
interpreting the zeros as an "instrument verdict" — the readout construct is
wrong, not the models. That interpretive move is only as strong as the
evidence that the readout's *implementation*, applied to the *actual*
checkpoints via the *actual* multi-layer, `fla`-kernel extraction path (the
"two-forward protocol" described at `REASONING_LINK_DESIGN.md`'s FATAL-2
fix), is bug-free. I found exactly one positive control in the whole
program: `test_item_2_hand_built_composition` in
`reasoning_link_stage_minus1.py` (lines 76-96), which validates
`apply_state_power(S, q, h)` and the cosine-recovery scorer against a
**hand-built 4×4 diagonal matrix on CPU** — pure math-function correctness,
with no model, no forward pass, no `fla` kernels, no multi-layer extraction.

This project's own CLAUDE.md states the exact lesson that applies here:
"CPU-stub self-test suites test logic only; real-kernel coverage needs a
separate narrow smoke of the PRODUCTION path (forward/backward/grad/
checkpoint/resume on real fla/CUDA)." I found no such real-kernel positive
control anywhere in the six archived waves — not even in Wave 4, which is
explicitly framed in §3 as "the strongest-case test" (a model directly
trained on the bind/query task). Wave 4 is the closest thing to a positive
control the program has, and it *also* returns 0/30 with gates failing
(§3, Fig. 2 right panel) — meaning **every attempt in this program to get
the readout to fire, including on a model that demonstrably learns the task
in vocabulary space, has failed**. That is consistent with the paper's own
"wrong construct" story, but it is *equally* consistent with an undiscovered
extraction/convention bug specific to this exact multi-layer `fla`/DeltaNet
setup (this same broader project's own git history records at least one
prior case of a state-transpose-convention scare requiring an "analytical
exoneration" for a different architecture in a sibling campaign) — and the
paper never rules this out with a state built directly from a *real* forward
pass where the h-hop answer is known analytically (e.g., feed a synthetic
sequence through the actual trained model with keys/values constructed so
the correct v_target is knowable in closed form, and confirm the extraction
pipeline recovers it before trusting a genuine-checkpoint zero as
diagnostic).

**Supporting evidence.**
`experiment-runs/2026-07-07_reasoning_link_phase1/reasoning_link_stage_minus1.py`
lines 76-96 (`test_item_2_hand_built_composition`, CPU-only, 4×4 diagonal
matrix); `CLAUDE.md` "CPU-stub self-test suites test logic only..." rule;
Wave 4's own 0/30 result under the "strongest-case" framing (§3, C4-dissoc).

**What the paper would need to do to defuse this.** Either report a
real-kernel positive control (a constructed sequence through an actual
forward pass with an analytically-known target, scored end-to-end through
the same extraction path used for the real evaluations) or add an explicit
limitation sentence in §7 acknowledging that the readout's implementation
has not been validated end-to-end on the production kernel path, only at
the unit-math level.

---

### A7: The task-familiarization "21.8 to 46.4 percent" fall is measured from checkpoint 250, not from training start, and the paper never says so

**Severity:** MINOR
**Type:** methodological transparency

**Attack.** §3 (Wave 4) and Figure 2's caption report the vocabulary-space
query loss "falls 21.8 to 46.4 percent (mean 35.9 percent)" over the
familiarization run. I recomputed this directly from
`experiment-runs/2026-07-08_phase2_familiarization/results/off_*.json` and
it matches exactly — **but only when the baseline is checkpoint 250**, the
first of the five reported checkpoints. Measured from the true start of
familiarization (step 1, which is also recorded in every trajectory file),
the same six cells fall **70.7% to 81.2%** (mean ≈77.5%) — more than double
the reported figure. Neither the text nor the Figure 2 caption states that
the reported percentages are computed checkpoint-250-to-checkpoint-5000
rather than start-to-finish; a reader would reasonably assume "falls X
percent over 5,000 familiarization steps" (the caption's own phrase) means
from step 0. This doesn't change the paper's qualitative conclusion (task
learning without geometric-readout validity) and if anything the omitted
number is *more* supportive of "the model learns the task," but the
un-stated baseline choice should be disclosed for reproducibility, since
someone recomputing "the fall" from these files without knowing the intended
checkpoint window would not reproduce the quoted band.

**Supporting evidence.** Computed directly from
`experiment-runs/2026-07-08_phase2_familiarization/results/off_*.json`
`trajectory` arrays (steps 1 through 5000 are all present in each file).

**What the paper would need to do to defuse this.** Add "(measured from
checkpoint 250, the first post-warmup reading, to checkpoint 5000)" to the
Figure 2 caption or the §3 sentence.

---

### A8: Bibliography year inconsistency for Merity et al.

**Severity:** MINOR
**Type:** citation hygiene

**Attack.** `refs.bib`'s `merity2017pointer` entry lists `year = {2016}`
while the citation key itself encodes 2017 (the paper's actual venue year —
Pointer Sentinel Mixture Models appeared at ICLR 2017; the 2016 date is only
the arXiv v1 submission date). The in-text citation `\citealp{merity2017pointer}`
will therefore render "(Merity et al., 2016)" in the compiled PDF while every
other reference to this well-known paper in the literature dates it 2017.

**Supporting evidence.** `refs.bib` lines 50-55; compiled bibliography on
PDF page 5 line 193-194 ("Stephen Merity... Pointer sentinel mixture models.
arXiv preprint arXiv:1609.07843, 2016.").

**What the paper would need to do to defuse this.** Change `year = {2016}`
to `year = {2017}` (or cite the ICLR 2017 proceedings entry directly) for
consistency with standard convention.

---

## Attacks I considered but decided were weak

- **"n=12 replication" is misleading since pooling never happens.** The
  abstract/§5 call this "a targeted twelve-seed replication," but the gate
  blocks any n=12 pooled analysis. I considered this a claim-scope attack,
  but the paper is actually careful throughout — "twelve-seed" describes the
  wave's *target sample size*, and every sentence that follows immediately
  clarifies that pooling didn't happen and reports the cohorts separately.
  Not misleading once read past the label.
- **Premises (i)/(ii) (`premise_i_key_gram_deviation_mean`,
  `premise_ii_value_gram_deviation_mean`) appear in the raw JSON but are
  never mentioned in the paper.** I checked the design doc (§8.2/§8.3):
  these are explicitly registered as a "standing covariate" / diagnostic,
  never a formal gate, so their absence from the paper's gate description is
  not an omission of a decision-relevant instrument.
- **Venue page-limit compliance.** I checked page-by-page: `main.tex`'s
  sections 1-7 end exactly at the bottom of PDF page 4; References begin at
  the top of page 5. This is a legitimate 4.0-page main-content submission
  under MOSS @ COLM 2026's stated limit — no violation, despite how close it
  cuts it.
- **Anonymization leaks.** I grepped `main.tex`, all `sections/*.tex`, and
  `refs.bib` against the full anonymization-surface list in `brief.md`
  (author/handle/org tokens, absolute paths, `acknowledg`, `funded by`,
  `github.com/`, `huggingface.co/`) — zero hits.
- **The "1.31B" rung's readout layer (21, 0-indexed of 22 layers) being a
  genuinely terminal-state readout.** Verified directly against
  `leg_b_rung3_*.json`'s `readout_layer`/`n_layers` fields for all four
  rungs — consistently the last layer at every scale. Not an attack.

## New citations you found that should be in Related Work

- **Grazzi, Siems, Franke, Zela, Hutter, Pontil, "Unlocking State-Tracking
  in Linear RNNs Through Negative Eigenvalues," ICLR 2025, arXiv:2411.12537.**
  DeltaNet-specific state-tracking/permutation-composition theory — the
  single most on-point missing citation (see A5).
- **Merrill, Petty, Sabharwal, "The Illusion of State in State-Space
  Models," ICML 2024, arXiv:2404.08819.** General SSM state-tracking
  ceiling (TC0), a theoretical competitor account for the empirical null
  (see A5).
- **Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers
  are Better than State Space Models at Copying," arXiv:2402.01032 (ICML
  2024).** Establishes a fixed-state-capacity argument for why GSSMs
  (DeltaNet's architectural family) underperform at context-copying/recall
  tasks — complements Based/Zoology as a capacity-side account and is a
  natural citation alongside them in §6's first sentence.

---

*Security note:* No embedded `<system-reminder>` blocks, fake tool-output
injections, or concealment instructions were encountered in any file read
during this review. (The design doc I consulted, `REASONING_LINK_DESIGN.md`,
documents — in its own historical record — that earlier sessions on this
same project repeatedly encountered and correctly disregarded fake injected
system-reminders in tool output; none appeared in anything I read this
session.)
