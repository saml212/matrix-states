# Defense report — round 1 — `papers/rank-recruitment-ws/` ("When the Gradient Sees Rank")

**Defense stance:** fresh context, no stake in the attacks. Every attack's
factual premise was re-verified against the raw artifacts and design docs it
cites — not trusted from the attack report's prose. All seven premises checked
out. I could not honestly DEFEND either CRITICAL: the sibling-overlap sentence
is factually false as written (A1), and the depth-decay headline does rest on
one of three seeds with an undisclosed named instability (A2). Both are
CONCEDE+FIX (framing), not new-experiment blockers, but the rebuttal should
treat the two reframes as mandatory, not cosmetic.

## Summary for the rebuttal agent

**Counts: 0 DEFEND, 3 PARTIAL (A4, A5, A7), 4 CONCEDE+FIX (A1, A2, A3, A6).**
Neither CRITICAL was DEFENDed. The paper is **submittable after fixes** — none
requires a new experiment; all are text/number/code edits — with two load-
bearing exceptions the rebuttal must escalate:

1. **A1 (highest priority) is a cross-paper coordination problem, not just a
   sentence.** Rewriting this paper's Limitations sentence *discloses* the
   overlap but does not *remove* it: the sibling (`papers/neurreps-ea/`) still
   restates this paper's three headline numbers as its "provable foundation"
   paragraph, and the two `app:period` appendices are near-duplicate prose.
   Only a coordinator action on the **sibling** (convert its restatement to a
   pointer-cite; de-duplicate the depth-21 appendix) actually defuses the
   salami-slicing risk to a reviewer assigned both EAs at the same venue.
2. **A2's honest reframe demotes the flagship depth-amplification result** from
   "an established spectral *mechanism*" to "a single converged rank-starved
   seed, consistent with (not proven as) a spectral prediction, with a disclosed
   2-of-3 same-cell training failure whose cause is a named numerical
   instability." The within-seed mechanism (prediction matches the measured
   curve within 0.008 through h=7) is real and survives; the generality and the
   silence about the failure mode do not. Keeping it as a *mechanism* headline
   would need more converged force-rank-(K−1) seeds (new evidence, camera-ready
   deferral); it is submittable *now* only as an honestly-scoped case study.

The cheap, unambiguous fixes: A3 (drop or relabel the spliced d=16 frontier
anchor), A4 (add "in this architecture family" + one Limitations clause), A5
(one caveat sentence), A6 (change 14.7→14.6), A7 (make the existing warnings
filter effective and/or one Appendix C footnote).

---

## Defenses

### A1: The sibling EA's "binding foundation" paragraph restates this paper's three headline results with the same numbers — the Limitations claim of "no shared... headline claims" is false

**Disposition:** CONCEDE + FIX (framing fix for this paper; escalate the
cross-paper half)

**Response.** The attack is correct and I cannot DEFEND it. I read the sibling
directly. `papers/neurreps-ea/sections/03_binding.tex`, its "The provable
foundation (binding)" paragraph, states verbatim: *"Gradient descent recruits
the rank: at d = 16 effective rank reaches 8.20 at K = 8 and 15.08 at K = 16
... the recruited rank is causally necessary: a spectral rank cap at d = 8,
K = 4 leaves exact recovery ≤ 0.0004 for every cap k ≤ 3 and restores it to
0.940 at k = 4, a step exactly at the provable bound; the same operator
composes exactly (Zʰ, 21 self-applications) in four of five seeds."* Those are,
in substance, this paper's three headline results: recruitment (R1/R1b:
8.20/15.08–15.09), causal necessity (R2/R2b: ≤0.0004 below K, step at K), and
composition (R3: 4/5 seeds). This paper's `07_limitations.tex` says the
companion shares *"no figures or headline claims."* The "no figures" half is
true (I confirmed zero shared figures/tables — the brief's overlap section is
accurate on that). The "no headline claims" half is **false as written**: the
sibling restates all three of this paper's headline claims, and the two
`app:period` appendices are near-duplicate prose (compare this paper's *"Under
a single 8-cycle, π²¹ = π^{21 mod 8} = π⁵, so the nominal depth-21 probe shares
its target with the already-tested depth 5"* against the sibling's *"Under a
single 8-cycle, π²¹ = π^{21 mod 8} = π⁵ exactly, so nominal depth 21 shares its
target with the already-tested depth 5"* — same claim, same arithmetic,
paraphrased).

The one place the attack slightly overstates: the causal-step figure is not
literally the *same number* in both papers — this paper's abstract/R2 uses the
991-run value **0.97 (0.9675)** for the d=8,K=4 k=4 step, while the sibling uses
the 1,234-run replication value **0.940**. The recruitment endpoints
(8.20/15.08) and the "four of five seeds" composition claim *do* match. So the
substance of A1 — the sibling restates this paper's headline results as its own
load-bearing premise — is fully correct; only "identical to the last digit
everywhere" is loose.

Because the sibling's *own* contribution (the group-dimension d_min law, TOST,
the five-group razor) genuinely does not overlap this paper, the intended
meaning of the Limitations sentence is defensible — but the sentence as printed
denies a shared *headline claim* that demonstrably exists as the sibling's
background. That is a disclosure inaccuracy in a double-blind venue, which the
honesty rubric says cannot be a DEFEND.

**Supporting evidence.** `papers/neurreps-ea/sections/03_binding.tex` lines
9–22 (the Fact-1 paragraph, quoted above); `papers/neurreps-ea/sections/07_limitations.tex`
lines 149–165 (`app:period`, near-duplicate of this paper's
`sections/07_limitations.tex` lines 3–19); this paper's own brief
("Companion papers and overlap management") already concedes the sibling "cites
four numbers from the same Task D/E archives this paper is the primary carrier
of" — i.e., the brief knows about the overlap the Limitations text denies. Both
EAs target NeurReps 2026, same submission window (brief "Venue").

**What goes in the paper if this defense is accepted.** Replace the sentence
*"A concurrent companion submission covers group-composition state tracking,
sharing no figures or headline claims."* with an accurate disclosure, e.g.:
*"A concurrent companion submission (group-composition state tracking) shares no
figures or tables; its introductory paragraph restates this paper's
recruitment, causal-necessity, and composition numbers as background grounding
for its own distinct contribution — a group-dimension rank law — which this
paper does not make."* **Escalation (out of this paper's tree, for the
coordinator):** the sentence discloses but does not remove the salami-slicing
risk; the durable fix is to convert the sibling's `03_binding.tex` restatement
to a pointer-cite and de-duplicate the `app:period` appendix so the two are not
parallel prose. Flag to the rebuttal that A1 is only half-closable by editing
this paper alone.

---

### A2: The abstract's depth-amplification headline rests on one seed out of three, and the other two are a named, undisclosed instability in the paper's own causal-intervention mechanism

**Disposition:** CONCEDE + FIX (framing fix; the mechanism claim survives, its
generality and the silence do not)

**Response.** The attack is correct on every factual point; I re-derived them
from the raw JSON. At the flagship cell (`t1_matrix_permutation_K8_fr7_s*`, the
force-rank k=7=K−1 depth-amplification cell):

- `fr7_s2` (the seed the paper uses): `n_skipped_steps = 2`, mean_cos 0.930→0.821
  (h=1→21), effective rank ≈ 6.99, rec@0.9 0.996→0.060. Converged.
- `fr7_s0`: `n_skipped_steps = 8`, mean_cos **+0.00029** at h=1, effective rank
  **1.036**, rec@0.9 **0.0 at every hop**. Dead.
- `fr7_s1`: `n_skipped_steps = 10`, mean_cos **−0.00230** at h=1, effective rank
  **1.001**, rec@0.9 **0.0 at every hop**. Dead.
- All five unconstrained `frN` seeds: `n_skipped_steps = 0`.

So the depth-decay headline rests on 1 of 3 same-cell seeds, and the missing two
are statistically indistinguishable from a random rank-1 matrix from the first
evaluated hop — not "still transitioning." The design record names the cause
exactly: `TASK_E_FINDINGS.md` §9, *"`fr7` s0/s1 (the eigh-backward-instability
dead runs, `n_skipped_steps` 8 and 10) collapse `A` to `effective_rank(A) ≈
1.00`."* And `rank_utils.py`'s `truncate_to_rank` docstring is the very rationale
this failure contradicts: it chose `eigh(ZZᵀ)` over full SVD *"avoiding the
1/(σ_i²−σ_j²) → inf that full-SVD backward produces at degenerate spectra"* —
yet the observed failure is precisely a backward-instability at that cell.

One point that makes A2 *stronger* than the report states, and which the
rebuttal should know: `TASK_E_FINDINGS.md` §2/§6 shows the same eigh-backward
instability (`n_skipped_steps` 3–10) killed **all six** provably-*sufficient*
force-rank seeds (fr=8, fr=9) too, and the program therefore *deliberately did
not run* the full force-rank straddle grid ("not measurable at this operating
point with the current training recipe"). So the one converged rank-starved
seed is the sole survivor of a force-rank regime that is broadly unstable — the
paper's "one converged rank-starved seed" wording undersells how isolated it is.

What A2 does **not** overturn: the mechanism claim *for that one seed* is a
genuine, checkable, within-seed result — the eigenspectrum-only prediction
matches the measured cosine within 0.008 through h=7 (Table 2 / R4, which I
confirmed against `fr7_s2`'s per-hop `mean_cos`). The abstract's clause *"a
rank-starved operator's depth-decay curve is predicted from its eigenspectrum
alone"* is therefore *true of the seed shown* but reads as an established,
general mechanism. The defect is over-scoping + non-disclosure, not a wrong
number.

**Supporting evidence.** Recomputed from
`experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_fr7_s{0,1,2}.json`
(`n_skipped_steps` = 8/10/2; fr7_s0/s1 rec@0.9 = 0.0 at all hops, effrank ≈
1.0). `matrix-thinking/chapter2/TASK_E_FINDINGS.md` §9 (line 518, verbatim
"eigh-backward-instability dead runs") and §2/§6 (fr=8/fr=9 all dead; grid not
launched). `matrix-thinking/chapter2/rank_utils.py` lines 24–33
(`truncate_to_rank` docstring).

**What goes in the paper if this defense is accepted.** (1) Scope the abstract's
closing clause: *"…and in the one converged rank-starved seed, its depth-decay
curve is predicted from its eigenspectrum alone."* (2) Expand the Limitations
sentence from *"The force-rank contrast rests on one converged rank-starved
seed"* to disclose the failure mode: *"The depth-amplification contrast rests on
the single converged force-rank-(K−1) seed; the other two at that cell — and all
provably-sufficient force-rank seeds — collapsed under a numerical instability
in the spectral-projection backward pass (documented in the source design
record), so the surviving seed is read as a case study consistent with the
spectral prediction, not an n>1 established mechanism."* Keeping the depth-
amplification result as a *mechanism* headline (rather than a case study) is a
new-evidence item — more converged force-rank-(K−1) seeds under an
instability-guarded projection — and is a **camera-ready deferral, not a
submission blocker**, provided the reframe above ships now.

---

### A3: The §5 "exactness frontier" splices in a d=16 point from a different task/K-ratio/architecture, contradicting the paper's own source design document's explicit warning

**Disposition:** CONCEDE + FIX (framing fix — drop or relabel the anchor)

**Response.** Correct, and verified against the design doc. The paper's §5
sentence reads: *"Fixing K = d/4 at encoder width h = 64 (every cell confirmed
flat before being read as converged), mean recovery cosine falls monotonically
from ≈1.00 at d=16 to 0.877/0.909/0.915 at d=32 … 0.7196 at d=48 … 0.3882 at
d=64."* The three high-d points are genuine K=d/4, h=64 Stage-0 cells (R7 cites
`p100k_baseline_d32_K8`, `p100k_baseline_d48_K12`, `p150k_baseline_d64_K16` —
all K=d/4). But there is **no K=d/4 d=16 Stage-0 file** in R7's evidence row,
because none exists at that configuration:

- `STAGE0_DESIGN.md` §15.7.1's *"K=d/4 slice"* table lists only d=32 (K=8) →
  0.877–0.915 and d=48 (K=12) → 0.7196. **It has no d=16 row.** The design doc
  explicitly calls this the *"cleaner"* slice and contrasts it with the
  *"K=d/2 slice [that] is not monotone-clean … two confounds make that reading
  premature."*
- The "≈1.00 at d=16" number is the d=16, K=8 Task E *composition* model —
  `STAGE0_DESIGN.md` §15.3 sources the d=16 frontier endpoint from *"Task E
  40K/80K rounds."* That model is K/d = 0.5 (**K=d/2, not d/4**), a different
  task (multi-hop composition, not the single-hop Stage-0 trainability probe),
  and a different architecture (n_params = 170,896, no `h` field, per the raw
  JSON) versus the Stage-0 probes' n_params = 175,008 with explicit h=64.

So the paper's "Fixing K=d/4 … from ≈1.00 at d=16" makes a claim (K held at
d/4 across d=16) that is false, and its blanket "every cell confirmed flat
before being read as converged" cannot cover d=16 because d=16 is not a cell of
this sweep. A separate confirmation the anchor is a splice: STAGE0 §7 records
that a *genuine* d=16, K=4 (=d/4) Stage-0 cell has a documented low-ceiling
anomaly — it would *not* read ≈1.00 — so the ≈1.00 can only be the K=8
composition model.

What survives: the three real K=d/4 points already show a clean, well-separated
monotone decline (0.877–0.915 → 0.7196 → 0.3882). The qualitative claim does not
need the borrowed anchor.

**Supporting evidence.** `papers/rank-recruitment-ws/sections/05_frontier.tex`
lines 24–31; `matrix-thinking/chapter2/STAGE0_DESIGN.md` §15.3 (line 1505,
d=16 sourced from "Task E 40K/80K rounds") and §15.7.1 (lines 1664–1701, K=d/4
slice has no d=16 entry; K=d/2 slice flagged "not monotone-clean … premature");
raw fields: `t1_matrix_permutation_K8_frN_s1.json` (n_params=170896, no `h`) vs
`p100k_baseline_d32_K8_s0.json` (n_params=175008, h=64), both confirmed here.

**What goes in the paper if this defense is accepted.** Either (preferred) drop
the d=16 anchor and state the K=d/4 frontier over its three real cells: *"Fixing
K = d/4 at encoder width h = 64, mean recovery cosine falls monotonically from
0.877/0.909/0.915 at d=32 (three seeds, 100K) to 0.7196 at d=48 (100K) and
0.3882 at d=64 (150K)…"*; or keep d=16 only as an explicitly-labeled reference:
*"…with the d=16 composition model (K=d/2) exact at ≈1.00 for orientation."*
Remove d=16 from the scope of "every cell confirmed flat."

---

### A4: "Gradient descent recruits the rank the task demands" is stated as a general finding from a single sub-1M-parameter architecture and two synthetic task families

**Disposition:** PARTIAL (add an architecture-family caveat; one edit, no
experiment)

**Response.** Partly right, partly already handled. The abstract's central
sentence is *"Under these conditions gradient descent recruits the rank the task
demands"* — it *does* carry a scope qualifier ("Under these conditions," i.e.,
the teeth), so the attack's framing that the claim "has no scope qualifier at
the point of assertion" is not quite accurate. But A4's real point stands: the
qualifier scopes the *task/readout* conditions, not the *architecture*. The
finding is from one encoder family (row-reader latents producing rows of Z,
§2), and the Limitations paragraph names only scale ("synthetic and under 1M
parameters") — it is silent on architecture generality. An attention-readout,
a DeltaNet-style recurrent fast-weight, or a modern-Hopfield associative memory
could recruit rank differently under the same bound, and none of that is tested
or flagged as untested. The title's rhetorical mirror of the negative result
("When the Gradient Sees Rank" ↔ "The Gradient Does Not See Rank") does invite a
more-general reading than one architecture supports.

**Supporting evidence.** `main.tex` abstract line 68 ("Under these conditions
…"); `sections/02_setup.tex` "Model and controls" (single architecture);
`sections/06_related.tex` Limitations (scale-only scope).

**What goes in the paper if this defense is accepted.** Insert an
architecture qualifier in the abstract's central sentence — *"…gradient descent
recruits the rank the task demands in this architecture family"* — and add one
Limitations clause: *"generality across state-writing architectures (recurrent
fast-weight, attention-readout, associative-memory) is likewise untested."*

---

### A5: The M3 causal-necessity staircase is an aggregate over unseen-per-seed instability; A2's eigh-instability collapse is not ruled out as a contaminant in the "0.0 below K" cells

**Disposition:** PARTIAL (one caveat sentence; the load-bearing direction is
robust)

**Response.** The premise is factually correct and honestly hedged by the
attacker. I confirmed `AGGREGATE_latest.json`'s `M3_recovered_frac@0.9_vs_forcerank`
is a dict of (d,K) cells each mapping force-rank k → a **single float** — no
per-seed breakdown, no `n_skipped_steps`, no instability flag retained at the
aggregate level. Given A2's demonstrated instability in the same
`truncate_to_rank` train-time mechanism, one cannot tell from the aggregate
whether a given sub-K "≈0" cell is a geometrically-capped solution or an
instability collapse. As the attacker concedes, this does **not** threaten the
necessity claim (recovery ≈0 below K reads the same either way — the
lower-bound direction survives regardless of cause), and the paper's Figure 1
caption claim is about *where the step lands* (k≈K), not about smooth dynamics.
So this is a real but modest transparency gap, not a defect in any load-bearing
result.

**Supporting evidence.** `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_latest.json`
— `M3_recovered_frac@0.9_vs_forcerank[cell][k]` is one float per (cell,k),
confirmed here; same `rank_utils.py` / `TASK_E_FINDINGS.md` §9 instability
evidence as A2.

**What goes in the paper if this defense is accepted.** Add one sentence to §3
or Appendix C: *"The M3 aggregate retains one recovery figure per (d,K,k) cell,
not per-seed instability diagnostics; the necessity direction (recovery ≈0 below
K) is robust to the cause of any individual sub-K collapse, while the smoothness
of the post-knee approach should be read at the aggregate level only."*

---

### A6: Minor rounding slip in the reported whole-matrix stable-rank range

**Disposition:** CONCEDE + FIX (trivial number correction)

**Response.** Correct. Recomputing `stable_rank_mean` across every hop of the
four converged frN seeds gives a true range of **14.6487** (seed 4, h=21) to
**15.5919** (seed 1, h=6). 14.6487 rounds to 14.6, not 14.7. The paper's §4
prints "14.7--15.6"; the upper bound is fine, the lower bound is off by ~0.05.
The brief already carries the more precise "14.66--15.59."

**Supporting evidence.** Recomputed here from
`experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_frN_s{1,2,3,4}.json`,
field `stable_rank_mean` over all `M2_in_distribution`/`M3_held_out` hop
entries: global min 14.64867 (frN_s4, h=21), global max 15.59185 (frN_s1, h=6).

**What goes in the paper if this defense is accepted.** In
`sections/04_composition.tex`, change *"stable rank 14.7--15.6"* to *"stable
rank 14.6--15.6"* (or the brief's precise "14.66--15.59").

---

### A7: Undisclosed RuntimeWarnings in the analysis pipeline when computing the entity-subspace decomposition

**Disposition:** PARTIAL (attack overstates "undisclosed"; a code fix and/or a
one-line Appendix C note, not a paper-correctness issue)

**Response.** Partly right. I could not re-run `analyze_zdump.py` in this
sandbox (the `.venv` the attack invokes is absent and system numpy is not
installed), but the attack's factual basis is confirmed by reading the code:
`block_decompose` (line 208) and `synthetic_keys_from_pi` (line 223) exist and
do near-singular matmuls, and the pipeline has exactly the finiteness guard the
attack cites — line 214–215: *"if not np.isfinite(M).all(): raise
FloatingPointError(...real bug, not BLAS noise)"* — so any warning that
signalled real corruption would halt the build, and the attacker independently
confirmed the numbers match. Where the attack **overstates**: the warnings are
not "undisclosed" — `analyze_zdump.py` lines 86–93 already carry an explicit
comment (*"Apple Accelerate … emits spurious divide-by-zero/overflow/invalid
RuntimeWarnings … a known Accelerate quirk, not a real numerical issue here"*)
**and** a `warnings.filterwarnings("ignore", category=RuntimeWarning,
module="numpy")` call meant to silence them. The reason the attacker still saw
~60 warnings is almost certainly that the `module="numpy"` predicate does not
match: numpy's matmul RuntimeWarnings are attributed to the *calling* module
(`analyze_zdump`), not `"numpy"`, so the filter is ineffective as written. So
the true state is "documented-benign but not actually suppressed," which is
better than the attack's "undisclosed," but a reviewer rerunning from the
anonymized code link would indeed see a wall of warnings that Appendix C's
"changed artifact fails the build" language does not prepare them for.

**Supporting evidence.** `matrix-thinking/chapter2/analyze_zdump.py` lines
86–93 (comment + `warnings.filterwarnings(..., module="numpy")`), 208
(`block_decompose`), 214–215 (finiteness guard), 223 (`synthetic_keys_from_pi`).
`sections/07_limitations.tex` `app:repro` (the "fails the build rather than
silently plotting stale data" clause).

**What goes in the paper if this defense is accepted.** No paper-text
correctness change. Either (code) make the existing filter effective — drop the
`module="numpy"` predicate or wrap the near-singular matmuls in
`np.errstate(all="ignore")` — so the pipeline is actually quiet; and/or add one
sentence to Appendix C: *"The analysis pipeline emits benign BLAS Runtime
warnings from near-singular intermediate products on some platforms; every
decomposition output is asserted finite before use, so these do not affect any
reported number."*

---

## New citations found during defense

None was required to *defend* an attack — all three the attacker proposed are
related-work-completeness suggestions, not literature I had to cite to rebut a
factual premise. Ranked by value for a 0.30-page related-work budget:

- **DeltaNet — Yang, Wang, Shen, Panda & Kim, NeurIPS 2024
  (arXiv:2406.06484).** Highest value: it is the direct antecedent of
  `siems2025deltaproduct` (DeltaProduct), which the paper already cites as the
  Householder-product extension of exactly this method. Citing the extension
  without the base is a gap reviewers in the linear-attention/fast-weight
  subfield (the paper's own Nazari/Sun/Grazzi/Siems neighborhood) will notice.
  Recommend adding.
- **Gated Linear Attention — Yang et al., ICML 2024 (arXiv:2312.06635)** and
  **Titans — Behrouz, Zhong & Mirrokni (arXiv:2501.00663).** Reasonable but
  optional given the page budget; GLA sits with the Nazari/Sun observational-
  rank line, Titans with the introduction's "matrix state as a capacity-budgeted
  associative store" framing. Defer to the rebuttal on whether the 0.30-page
  related-work section can absorb them.

## Attack ordering note

- **A1 (CRITICAL): agree.** A false disclosure sentence plus near-duplicate
  appendix prose plus restated headline numbers across two same-venue,
  same-window double-blind submissions is correctly the top severity — and,
  importantly, is the *only* attack this paper cannot fully close on its own
  (the sibling must change too). Rebuttal should treat it as blocking on
  coordinator action, not on this paper's text alone.
- **A2 (CRITICAL): agree.** It is the flagship depth-amplification claim resting
  on 1-of-3 seeds with an undisclosed, *named* instability that also killed
  every provably-sufficient force-rank seed. The reframe is mandatory; whether
  the result stays a "mechanism" or becomes a "case study" is a real scientific
  demotion, not a wording nicety.
- **A3 (SERIOUS): agree it is SERIOUS, but borderline SERIOUS/MINOR** — the
  splice is real and the "Fixing K=d/4" sentence is false as printed, yet the
  remedy is a one-line drop/relabel and the qualitative monotone-decline claim
  survives on the three genuine cells. Not a data problem.
- **A4 (SERIOUS): I would downgrade toward MINOR.** The "Under these conditions"
  qualifier already exists; the gap is one missing architecture-generality
  caveat, a standard scope edit, not a structural overclaim.
- **A5 (SERIOUS, hedged): the hedge is right; effectively MINOR.** The attacker
  concedes the necessity direction is robust; it threatens no load-bearing
  claim and closes with a single caveat sentence.
- **A6, A7 (MINOR): agree.** A6 is a one-digit correction; A7 is code hygiene
  the paper text never actually got wrong.

**Attack the attacker missed (surfaced for a better paper, not to win the
argument).** Figure 1's caption — *"The step lands at k≈K in every cell"* — is
scoped to the **three** cells the figure plots (d8_K4, d16_K8, d16_K12). The
*same* `AGGREGATE_latest.json` contains four more M3 staircase cells where no
step at k≈K appears: `d16_K4` caps at rec@0.9 ≈ 0.13 even at k=16 despite an
in-band recruited rank of 4.74 (the documented d=16,K=4 "ceiling anomaly"), and
`d32_K16`, `d64_K32`, `d128_K64` read **identically 0.0 at every k, including
k ≥ K** (Stage-0 trainability failures — the very cells §5's frontier is about).
This is methodologically *defensible* — a step to ceiling can only appear in a
cell whose unconstrained model reaches the ceiling — but "in every cell" reads
as "every cell in the grid" when it means "every converged cell." Recommend
scoping the caption/§3 to *"the step lands at k≈K in every cell whose
unconstrained model reaches exact recovery"* so a reviewer who opens the same
aggregate and finds four all-fail cells is not left with a cherry-pick
impression. Cheap, and it pre-empts a follow-on attack.
