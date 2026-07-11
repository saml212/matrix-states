# Defense report — round 1 — mstar-colm-er

## Summary for the rebuttal agent

**0 DEFEND, 4 PARTIAL (A5, A6, A7, A8), 4 CONCEDE+FIX (A1, A2, A3, A4).** I
independently re-verified every attack's factual premise against the raw
JSONs, the rendered PDFs, and the read-only design registry
(`HEAD_TO_HEAD_DEMO_DESIGN.md`) rather than taking either the paper's or the
attack's word for it, and every attack held up on its facts to some degree —
this is a genuinely well-built paper being reviewed by a genuinely careful
attacker, and I did not find a clean factual disproof for any of the eight.
That said, the two CRITICALs are both **cheap to fix without new GPU
compute**: A1's fix is a training-curve figure from data that already exists
in the archived JSONs plus a softened claim; A2's fix is a disclosure plus
citing an already-executed, already-verified diagnostic
(`HEAD_TO_HEAD_DEMO_DESIGN.md` §1.3.1.5/R3-F1) that the design's own
self-attack (item 10, M-NEW-2) flagged as the identical confound eleven
months before this gauntlet round and explicitly said was "mitigated, not
eliminated." A3 (occluded figure markers) and A4 (backwards statistical
framing) are both confirmed defects with mechanical, same-day fixes. The
four MINOR/SERIOUS PARTIALs are all one-to-two-sentence disclosure edits.
**The paper is submittable after these fixes; none requires new training
runs or new GPU-hours** — the required work is entirely in the LaTeX source,
`figure-gen.py`, and (for A2's strongest version) a short addition to
`refs.bib`. I flag below that A6 looks over-rated (SERIOUS) relative to what
I found in the registry, and that A2, while correctly rated CRITICAL, has a
faster and stronger fix available than the one the attacker proposed.

---

## Defenses

### A1: Both non-contender arms' training loss flatlines almost immediately; no gradient-health check backs the "it never learns" story

**Disposition:** CONCEDE + FIX

**Response.** I re-extracted `curve` from all nine task1-sweep training
JSONs (not just the ones the attack quoted) and the attack's table is
accurate: ablation and transformer losses drop from ~23.8 to ~7.7-7.8 in the
first 500 of 20,000 steps and are flat (noisy, non-monotone) for the
remaining 19,500; the contender falls monotonically to 1.38. I confirmed
`grad_ratio_at_tap_step500` is `None` in all nine files, not just the three
the attack sampled. The paper's Section 3 "licensed" sentence — "the
outer-product state update is doing task-critical work its parameter-matched
additive counterpart does not do" — reads as a claim about architectural
capability, and the flatline pattern is genuinely consistent with either
that or an LR/warmup mismatch for two structurally different mixers trained
under one shared hyperparameter setting. I looked for a mitigant and found a
partial one: the design registry's Wave −1(A) item ("probe smoke —
forward/backward/grad-check with the §1.3.1 probe+adapter attached, all 3
arms") shows a basic gradient-flow smoke test *did* run for all three arms
before launch, and a live `GRAD_RATIO_STEP=500` dial guards every training
run for pathological aux/CE loss-weight imbalance (confirmed elsewhere in
the registry to actually fire and abort a cell when triggered, e.g. the K48
dial-exhaustion incident at §1.42) — none of these nine runs tripped that
guard. So this is not "zero gradient-health signal exists anywhere in the
project," but it is a narrow loss-weighting-balance check, not a
per-architecture LR-suitability or dead-gradient diagnostic, so it does not
fully answer the attack's question. The attack's own point that the
contender reaches 1.38 nats under the identical objective/weighting also
rules out a *pure* aleatoric floor account, which the attack itself
concedes. Net: the attack's factual premise is fully correct and its
alternative-explanation concern is not defused by anything already on
record.

**Supporting evidence.**
`experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_{ablation,transformer,contender}_task1_sweep_s{0,1,2}.json`,
field `curve` (verified all 9, not just the attack's 3) and
`grad_ratio_at_tap_step500` (confirmed `null` in all 9);
`HEAD_TO_HEAD_DEMO_DESIGN.md` line ~1527 (Wave −1(A) smoke, all 3 arms) and
the K48 dial-exhaustion record at §1.42 (evidence the live grad-ratio dial
is a real, firing guard, not dead code).

**What goes in the paper if this defense is accepted.** (1) Add a
training-loss-curve panel (or a compact appendix table) for all three arms
using the already-archived `curve` field — zero new compute, just a
`figure-gen.py` addition. (2) Add one sentence to Section 2.2 or Limitations
disclosing that LR/warmup were shared across architecturally distinct
mixers and were not independently tuned per arm. (3) Soften Section 3's
"licensed" sentence from "is doing task-critical work its parameter-matched
additive counterpart does not do" to something scoped to the shared budget,
e.g. "reaches ceiling on this task under the shared training budget where
the additive update's loss does not move past its first-500-step value" —
removing the implication that the additive update is architecturally
incapable, leaving only the (fully supported) matched-budget empirical
claim. (4) Recommended but not a blocker: a same-day, cheap per-arch LR
sensitivity spot-check (2-3 LR values × 2 baseline arms, short runs) would
let the paper make the stronger claim outright instead of scoping around
it; flagged as a camera-ready strengthening, not required for submission
given (1)-(3).

---

### A2: The "structure control" ablation confounds state capacity (64×) and content-addressability with matrix structure

**Disposition:** CONCEDE + FIX

**Response.** The byte-capacity numbers are exactly as the attack states
(32,768 vs 512 bytes, 64×) and the write rule as described in
`02_setup.tex` (`s_t = s_{t-1} \odot g_t + v_t`) indeed has no visible
dependence on `k_t`. More importantly, I searched the design registry for
whether this exact concern had been raised and resolved during the
project's own gauntlet, and found that it had — independently, months
before this attack — as **the project's own self-attack item 10 (M-NEW-2)**:
"an irreversible expressivity gap the linear probe/adapter cannot repair
… This structurally favors the CONTENDER on axis 1 (data-efficiency) for
reasons that have nothing to do with matrix-vs-vector state CAPACITY, only
matrix-vs-vector READ mechanism." The registry's own verdict on its
mitigation: "**Mitigation, not elimination** … It does not repair the
confound; it bounds it." That is a stronger and more precise statement of
the same problem the attack raises, on file in the project's own source of
truth, and the current paper text ("differs only in the mixer recurrence")
contradicts it. This is not defensible as written.

However, I also found the actual numerical content of that mitigation, and
it is real ammunition for the paper, not just a caveat to bolt on. The
registry's §1.3.1.5 K-simultaneous-bindings diagnostic (Rev 3, R3-F1,
numerically executed and independently confirmed, 200 trials/K, `d=64`)
fits the *best possible* state under each tap family — full matrix
least-squares (`S = T @ pinv(Q)`) versus the exact closed-form per-coordinate
fit for the paper's own read rule `o = s \odot q`
(`s_j = \sum_i q_i[j] t_i[j] / \sum_i q_i[j]^2`) — to `K` simultaneous
(query, target) bindings, `K \in \{1,2,4,8,16,32,48,64\}`. Result: the
matvec tap recovers `rf@0.9 = 1.000000` at every `K \le 64 = d`; the
elementwise/Hadamard tap recovers `1.000000` only at `K=1` and
`0.000000` for every `K \ge 2` (mean cosine decays `~1/\sqrt K`, independent
of `d`, because each coordinate only has one scalar of freedom to satisfy
`K` constraints). This is the attacker's own proposed fix (b) — "give the
vector state more bytes and see if it still fails" — *already run*, by
proxy: because the derivation is `d`-independent, resizing the ablation's
`d_state` to match bytes (the attacker's 4096-dim suggestion) would not
close the gap for `K \ge 2` bindings. That means the "maybe it's just 64×
less capacity" alternative explanation the attack raises is not merely
undefended, it is actively contradicted by evidence the project already
has on hand — this is a much cheaper and stronger fix than running the new
ablation the attack recommends.

**Supporting evidence.** `sections/02_setup.tex` (ablation write rule,
byte counts); `HEAD_TO_HEAD_DEMO_DESIGN.md` self-attack item 10 (M-NEW-2,
"structurally favors the contender … not eliminated") and §1.3.1.5/R3-F1
(the K-simultaneous-bindings numeric table, matvec 1.0 at all `K\le d`
vs. Hadamard 1.0 only at `K=1`, 0.0 at `K\ge2`); Plate (1995) and Smolensky
(1990), below, as the missing citations for vector-native binding schemes
this ablation does not instantiate.

**What goes in the paper if this defense is accepted.** (a) Rescope Section
3's "licensed" sentence per the attacker's own suggested language — "a
matrix state with more raw capacity and a multiplicative, key-conditioned
write outperforms a smaller, non-content-addressable vector state" — and
move the byte-capacity gap into an explicit disclosed confound near the
ablation's description in Section 2.2, dropping or qualifying the word
"only." (b) Add 2-3 sentences citing the K-simultaneous-bindings result
(no new experiment — it is already computed and verified) to explain why
resizing the vector state would not close the gap: this both satisfies the
disclosure obligation and pre-empts the reviewer question the attack raises
about a bigger vector control. (c) Add Plate (1995) and Smolensky (1990) to
Related Work as the literature on vector-native, key-conditioned binding
this particular ablation does not implement — one sentence, cheap in the
0.5pp Related Work budget. None of this requires new GPU compute; it is a
same-day text/citation fix drawing on evidence the project already has but
never surfaced in the paper.

---

### A3: The money figure (Fig. 1) does not actually show the capped-KV data it is captioned to show

**Disposition:** CONCEDE + FIX

**Response.** I rendered `fig1_horizon.pdf` at 300 dpi myself
(`pdftoppm`) and confirm the attack's read exactly: the plot shows only the
solid black contender line and the dashed dark-red uncapped-transformer
line. No gray × markers are visible anywhere in the plotted region, despite
the legend listing them. This is a real, confirmable rendering defect —
`figure-gen.py` lines 141-151 do plot the crosses at `zorder=3`, strictly
beneath the dashed line at `zorder=4`, and the capped values (0.020-0.033)
and uncapped values (0.027-0.036) both sit compressed into roughly 1% of
the y-axis height given the axis spans −0.04 to 1.06. For a figure the
brief itself calls the "MONEY FIGURE," this is not acceptable as
submitted.

**Supporting evidence.** Rendered
`papers/mstar-colm-er/figures/fig1_horizon.pdf` (visually inspected, no
crosses visible); `figures/figure-gen.py` lines 141-151 (zorder ordering
confirmed by direct code read).

**What goes in the paper if this defense is accepted.** Fix
`figure-gen.py`'s `fig1_horizon`: raise the crosses' `zorder` above the
dashed line (e.g. `zorder=6`), and/or draw the uncapped dashed line at
reduced alpha, and/or add a small zoomed inset over the 0-0.1 accuracy band
so the cap sweep is visibly distinguishable. Re-run `figure-gen.py` and
re-render `fig1_horizon.pdf`. This is a same-day fix — no data changes,
purely a plotting-order bug.

---

### A4: The "$S_1$-zeroing leaves it unchanged" statistical framing for seed 1 may have the causality backwards

**Disposition:** CONCEDE + FIX

**Response.** I traced the `sigma` computation to its source
(`h2h_round4_driver_rd.py::s0_hard_stop_check`, line ~275):
`sigma = sqrt(p_hat * (1 - p_hat) / n)` with `p_hat = acc_intact`. At
`acc_intact = 1.0` (seed 1), this is mechanically zero by construction. The
attack's statistical point is correct: this formula is the standard error
of a *sampling* process (repeated i.i.d. Bernoulli(p) draws), and it is
being applied to a *paired, deterministic* comparison — two forward passes
of the same frozen checkpoint on the same fixed 4,096-query set, one intact
and one with $S_1$ zeroed, with no randomness in either evaluation. There is
no sampling process here for the binomial SE to characterize; treating
"$p_{hat}=1.0$ therefore $\sigma=0$ therefore any nonzero delta is an edge
case of a degenerate check" inverts what actually happened, which is a
reproducible, deterministic behavior change on 21 of 4,096 queries under a
real architectural intervention. I confirmed the raw numbers directly:
`acc_intact=1.0`, `acc_s1_zeroed=0.994873`, `delta_s1=0.005127`,
`sigma=0.0`, `unchanged=False`, `passed=False` — the code's own `unchanged`
field is already `False`; the paper's prose describes this as "disclosed as
an edge case of the check rather than adjudicated by it," which
undersells what the JSON itself already correctly reports as a failed
check. This is a real but small and clearly scoped issue: it is specific to
seed 1's $S_1$ (the block the paper's own causal story says is inert), not
$S_0$ (the load-bearing block), and 21/4096 ≈ 0.5% does not threaten the
paper's headline S0-necessity claim, which the attack itself concedes.

**Supporting evidence.**
`experiment-runs/2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s1_round4.json`,
field `s0_necessity_check` (`sigma: 0.0`, `unchanged: false`, `passed:
false` — verified directly, matches the attack exactly);
`matrix-thinking/deltanet_rd/h2h_round4_driver_rd.py` line ~275
(`sigma = (p_hat * (1.0 - p_hat) / max(1, n)) ** 0.5`, confirms the
binomial-SE formula and its degeneracy at `p_hat=1`).

**What goes in the paper if this defense is accepted.** Replace
`sections/05_mechanism.tex`'s framing of the seed-1 $S_1$ result. Instead
of "the 0.0051 dip … is disclosed as an edge case of the check rather than
adjudicated by it," report the raw finding plainly: "at seed 1, $S_1$-
zeroing produces a small, reproducible (deterministic, non-sampled) change
on 21 of 4,096 queries (0.51%); the binomial band used elsewhere for this
check is ill-posed at $p=1$ and is not used to adjudicate this cell. The
effect is confined to $S_1$ (the block the causal read identifies as
inert) and does not affect the $S_0$-necessity result." No new experiment
is required — the comparison is already deterministic, so "reproducibility
across a repeated forward pass" is guaranteed by construction and can be
stated as such rather than run again.

---

### A5: The paper's motivating framing (deployment-scale KV-cache growth) is tested at a horizon three to four orders of magnitude below where that framing applies

**Disposition:** PARTIAL

**Response.** The gap is real: the intro's motivating stakes are set at
deployment context lengths (StreamingLLM/H2O's actual operating regime,
tens of thousands to millions of tokens) and the longest tested horizon is
1,798 tokens. I checked `08_limitations.tex` directly and confirm the
attack's read: the disclosed scale caveat ("14M-parameter class, trained
20,000 steps … nothing here speaks to pretrained models, natural text, or
other scales") is phrased around model/training scale, not explicitly
context length, so a careful reader could reasonably ask whether it covers
this gap. That said, the Conclusion already partially pre-empts this
("this is a small-scale existence result on the recall substrate such
reasoning requires"), which softens but does not eliminate the need for an
explicit context-length disclaimer.

**Supporting evidence.** `sections/01_intro.tex` opening paragraph
(deployment-scale framing); `sections/08_limitations.tex` (scale caveat
present but scoped to model/training size, not context length);
`sections/04_horizon.tex` (max horizon 1,798 tokens, confirmed).

**What goes in the paper if this defense is accepted.** Add one sentence to
Limitations explicitly scoping the constant-memory demonstration to the
tested horizon band (≤1,798 tokens, ≤8× the binding span) and noting this
falls short of the context lengths where KV-cache cost is a practical
deployment constraint — framing the result as an existence proof of the
mechanism, not a claim it holds at deployment-relevant lengths. No new
experiment required.

---

### A6: "A blank-out test... enforces this by construction" is asserted without a reported result

**Disposition:** PARTIAL

**Response.** I looked for whether this claim is actually backed by
anything, since the attack's suspicion — an unsupported architectural
assertion masquerading as a tested claim — would be a serious problem if
true. It is not true: the design registry documents a real, executed,
independently-verified check. §1.3.1.3 states the guarantee is "verified by
the §1.3.1.3 blank-out test … not asserted," and separately documents an
**R5-F4 companion check** — "fresh-model-instance continuation (only $S_T$
passed) must produce bit-identical logits — closes the hidden-module-cache
channel the in-place test can't" — which is registered in the box-smoke
checklist as part of the binding BUILD-FIX item list. So this is not an
unverified architectural hand-wave; it is a real pass/fail structural
guarantee that was checked. Where the attack is right: I could not find a
JSON artifact in `experiment-runs/` with a quotable blank-out result (I
grepped for "blank" across the three relevant experiment-runs directories
and found nothing), so the paper cannot currently point to a numbered row
the way it does for C1-C10, and the sentence as written gives the reader no
way to independently verify it the way every other claim in the paper is
designed to be verified. The attack itself concedes this isn't technically
a numerical claim (so Appendix C's literal promise about numerical claims
isn't broken), which is why I'm not rating this CONCEDE+FIX — the
underlying claim is true and independently checked, just uncited.

**Supporting evidence.** `HEAD_TO_HEAD_DEMO_DESIGN.md` line ~2140
("verified by the §1.3.1.3 blank-out test, not asserted"); lines
~3178-3188 (R5-F4, "fresh-model-instance continuation … must produce
bit-identical logits," registered in the build-fix item list); line ~3889
("verified by the §1.3.1.3 blank-out test plus the R5-F4 fresh-instance
companion, not asserted"). No JSON artifact with a quotable pass/fail
result found in `experiment-runs/2026-07-10_h2h_mstar`,
`2026-07-10_h2h_sweep_harvest`, or `2026-07-09_h2h_tap_localization`
(grepped for "blank", zero hits).

**What goes in the paper if this defense is accepted.** Add a citation or
footnote to `sections/05_mechanism.tex`'s blank-out sentence pointing at
the verified check and stating what it actually established: "verified via
a fresh-model-instance continuation producing bit-identical logits when
only the cached state (not the raw bind tokens) is passed forward
(registry §1.3.1.3, R5-F4)." This is a citation-hygiene fix, not a new
experiment, since the check already ran.

---

### A7: Three seeds is thin for the primary comparison by typical ML replication norms

**Disposition:** PARTIAL

**Response.** Factually correct as stated — $n=3$ is the primary
comparison's seed count, and the paper's own task2 analysis uses 9 seeds
once trainability variance became a live concern, which is a fair internal
inconsistency to point out. The attack itself concedes the effect size
(≈0.97 accuracy gap against seed variance ~$10^{-3}$) makes this a
non-issue for the conclusion, and I agree — recomputing the paired CIs from
the raw per-seed numbers (already verified via `figure-gen.py`'s
`paired_ci`) shows no seed anywhere close to the frozen 0.30 margin. This
doesn't threaten anything, but it is not incorrect, so it isn't a clean
DEFEND either.

**Supporting evidence.** `figures/tables_generated.tex` (per-seed accuracy
values, CI bounds); `sections/06_scope.tex` (9-seed task2 table, confirming
the internal inconsistency the attack notes).

**What goes in the paper if this defense is accepted.** One line in
Limitations: "$n=3$ is used for the primary comparison because the
pre-registered seed-extension trigger did not fire (no seed straddled the
0.30 margin); task2's later 9-seed extension reflects a different,
seed-variance-driven trigger, not a change in standard." No new experiment
required.

---

### A8: No multiple-comparison discussion across the paper's many pre-registered tests

**Disposition:** PARTIAL

**Response.** Correct as a count (2 + 5 + pooled + per-seed comparisons
across two tasks, no correction applied), and the attack correctly notes
that no plausible correction flips any conclusion given the effect sizes
involved (0.9+ accuracy gaps against $10^{-3}$-$10^{-2}$ seed variance).
This is a real gap in statistical-hygiene disclosure, not a threat to any
specific number.

**Supporting evidence.** Confirmed by inspection of `03_recall.tex`,
`04_horizon.tex`, and `06_scope.tex` — the CIs and gates listed by the
attack are all present as described.

**What goes in the paper if this defense is accepted.** One sentence in
the Analysis Protocol subsection (`02_setup.tex`) noting the primary
decision rule is the single pre-registered contender-vs-ablation
comparison, and the remaining reported intervals (per-$M$ gaps, pooled
task2 CI, etc.) are descriptive/diagnostic rather than additional
hypothesis tests requiring correction. No new experiment required.

---

## New citations found during defense

- **Plate, T. A. (1995), "Holographic Reduced Representations," IEEE
  Trans. Neural Networks, 6(3):623-641.** Confirmed absent from
  `refs.bib` (grepped, zero hits). Needed for A2's fix — the established
  vector-native, content-addressable binding scheme the flat-vector
  ablation does not implement, and the natural citation for the
  "removes content-addressability, not just structure" disclosure.
- **Smolensky, P. (1990), "Tensor Product Variable Binding...," Artificial
  Intelligence 46(1-2):159-216.** Also confirmed absent. Same relevance as
  Plate 1995, earlier work in the same vein.
- **Olsson, C. et al. (2022), "In-context Learning and Induction Heads,"
  Anthropic Transformer Circuits Thread.** Confirmed absent from
  `refs.bib`. Optional strengthening for A1's fix — relevant context for
  why a 2-layer transformer's complete failure at a copy/recall task is
  surprising enough to warrant the training-curve disclosure, but not
  load-bearing the way Plate/Smolensky are for A2; add only if the 0.5pp
  Related Work budget allows.
- **Behrouz, Zhong & Mirrokni (2025), "Titans," arXiv:2501.00663**, and
  **Yang et al., "Gated Delta Networks," arXiv:2412.06464.** Confirmed
  absent. Both are legitimate nearest-prior-work gaps the attack correctly
  identifies, but neither is load-bearing for any of the eight attacks
  above — recommend adding only if page budget allows; not required by any
  CONCEDE+FIX or PARTIAL in this report.

---

## Attack ordering note

The attacker's CRITICAL/SERIOUS/MINOR split is largely right, with two
adjustments:

- **A1 and A2 are correctly rated CRITICAL** — both threaten a headline
  interpretive claim (not the raw accuracy numbers, which are solid), and
  neither survives as a clean DEFEND. But both are cheaper to fix than
  their severity label might suggest to a reader triaging effort: neither
  needs new GPU compute, and A2 in particular has its rebuttal evidence
  already sitting in the design registry, unused. I'd flag this
  explicitly for the rebuttal agent so the fix isn't over-scoped into a
  new experiment when a citation and two sentences would do.
- **A6 looks over-rated at SERIOUS.** Once traced to the registry, the
  claim it attacks is true and independently verified (not merely
  architecturally asserted), and the attacker's own text concedes the
  Appendix C promise technically doesn't apply (non-numerical claim). I'd
  call this MINOR — a citation-hygiene gap, not a validity threat — and
  would not want it competing for attention with A1/A2/A3/A4 in a
  rebuttal's priority ordering.
- **A5 is a defensible SERIOUS but is already partially pre-empted** by
  the Conclusion's "small-scale existence result" framing; I'd call it a
  borderline SERIOUS/MINOR rather than push back on the rating outright.
- **A3 and A4 are correctly rated SERIOUS** — both are confirmed, concrete
  defects (one visual, one statistical) with same-day mechanical fixes;
  I have no adjustment to offer here.
- **A7 and A8 are correctly rated MINOR** — both true, both
  non-conclusion-threatening, both one-line fixes.

One thing the attacker did not raise that I checked and found benign,
worth recording so a future round doesn't re-open it: the sweep-harvest
directory also contains `task3_sweep` JSONs (e.g.
`h2h_contender_task3_sweep_s0.json`) that the paper never mentions. I
traced this — it is a natural-language-corpus (`openr1-mix-ext`)
perplexity control lane with a completely different metric
(`final_val_loss_own`), unrelated to the recall-accuracy claims this paper
makes, and its exclusion is not selective reporting of a recall result.
Not an attack; recorded so the next round doesn't have to re-derive this.
