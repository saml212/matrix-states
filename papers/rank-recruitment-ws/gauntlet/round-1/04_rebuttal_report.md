# Rebuttal report — round 1 — `papers/rank-recruitment-ws/` ("When the Gradient Sees Rank")

**Adjudicator stance:** fresh context, trusts neither attacker nor defender.
Every load-bearing factual dispute was re-verified directly against the raw
artifacts (read-only), not averaged from the two reports. Confirmations below.

## CRITICAL-open statement (read this first)

**No CRITICAL remains open *inside this paper's tree* after the fix list is
applied.** Both CRITICALs are closable here:

- **A2** (depth-amplification single-seed) is fully resolved in-tree by FIX-2
  (scope the abstract clause + disclose the 2-of-3 named instability in
  Limitations). The within-seed spectral prediction is a genuine, verified
  result; only its scope and the silence about the failure mode were the
  defect, and both are text fixes. **Ship-now decision: YES** — the
  depth-amplification result ships as an honestly-scoped single-seed **case
  study**, not as an established general mechanism. Getting `n>1` (more
  converged force-rank-`(K-1)` seeds under an instability-guarded projection)
  is a **camera-ready / future-evidence** item, not a submission blocker,
  provided the scoping and disclosure of FIX-2 land now.

- **A1** (sibling-overlap disclosure) has two halves. The **in-tree half — a
  factually false sentence in this paper's Limitations** ("sharing no figures
  or headline claims") — is fully closed by FIX-1 (an accurate disclosure
  rewrite). The **cross-paper half** — the sibling `papers/neurreps-ea/` still
  restates this paper's three headline numbers as its "provable foundation"
  paragraph, and the two `app:period` appendices are near-duplicate prose —
  **cannot be fixed by this paper's writer** and is not a defect in this
  paper's text once FIX-1 lands. It is escalated to the PI/coordinator as a
  **blocking item on the joint two-paper submission decision** (see
  ESCALATION-1). It does **not** hold this paper's gauntlet open, because it is
  out-of-tree by construction and the CRITICAL termination rule governs defects
  the writer can and must fix in this tree.

So: the gauntlet on this tree may proceed to the detector gate after FIX-1..8
are applied and the affected claims re-enter a scoped re-attack (list below).
The human must still resolve ESCALATION-1 before **both** EAs go to the same
venue.

## Summary for the edit agent

**8 fixes: 2 CRITICAL, 2 SERIOUS, 4 MINOR. Plus 1 cross-paper ESCALATION (not a
writer fix). 0 CRITICAL open in-tree.**

Disposition counts: **0 DEFENSE VALID · 6 DEFENSE VALID BUT EDIT · 0 DEFENSE
INSUFFICIENT · 1 PARTIAL (A1)** (plus the self-surfaced Figure 1 caption edit,
FIX-7).

The three structural fixes that carry the weight:

1. **FIX-1 (A1, CRITICAL):** replace the false "sharing no figures or headline
   claims" Limitations sentence with an accurate disclosure that the companion
   *restates* this paper's recruitment/causal/composition numbers as
   background. In-tree only — the durable cross-paper de-dup is ESCALATION-1.
2. **FIX-2 (A2, CRITICAL):** demote the depth-amplification headline from
   "established mechanism" to a scoped single-seed case study — (a) scope the
   abstract's closing clause to "the single converged rank-starved seed," and
   (b) expand the Limitations sentence to disclose that the other two seeds at
   that cell (and every provably-sufficient force-rank seed) collapsed under a
   *named* numerical instability, so the surviving seed is a case study, not an
   `n>1` mechanism.
3. **FIX-3 (A3, SERIOUS):** drop the spliced `d=16` anchor (a `K=d/2`
   composition model, `n_params=170896`, no `h` field) from the "Fixing
   `K=d/4`" frontier sentence; the three genuine `K=d/4` Stage-0 cells
   (d=32/48/64) carry the monotone-decline claim on their own, and "every cell
   confirmed flat" then covers only real cells of the sweep.

The rest are cheap and local: FIX-4 (architecture-family scope qualifier),
FIX-5 (one Appendix-C caveat), FIX-6 (14.7→14.6), FIX-7 (scope the Figure 1
caption to converged cells), FIX-8 (Appendix-C footnote on benign BLAS
warnings).

**Page budget:** the body is at exactly 4pp; the fixes net a small addition
(mostly in the tiny Limitations paragraph). Compression plan is spelled out in
"Page-budget accounting" at the end of the fix list — FIX-5 and FIX-8 go to
appendices (excluded from 4pp), FIX-3 frees ~half a line, and a designated
reserve cut (the §4 `sqrt(1-1/8)` intuition sentence, which restates the
figure+table and carries no independent claim) is applied **only if** the
recompile exceeds 4pp.

**Facts I re-verified directly (read-only) before adjudicating:**
- A1: `papers/neurreps-ea/sections/03_binding.tex` "provable foundation
  (binding)" paragraph restates recruitment (8.20@K=8, 15.08@K=16), causal
  necessity (≤0.0004 below `k=4`, 0.940 at `k=4`), and composition (4/5 seeds);
  `papers/neurreps-ea/sections/07_limitations.tex` `app:period` is near-duplicate
  prose of this paper's `app:period`. **Confirmed.** (Nuance: the sibling's
  causal-step figure is the 1234-run replication value **0.940**, whereas this
  paper's abstract uses the 991-run **0.97**; the recruitment endpoints and the
  "four of five seeds" match. The overlap is real; "identical to the last digit
  everywhere" was the attacker's only overstatement, and it does not change the
  disposition.)
- A2: raw JSON `t1_matrix_permutation_K8_fr7_s{0,1,2}.json`:
  `n_skipped_steps = 8/10/2`; `fr7_s0` mean_cos@h1 = 0.000287, effrank ≈ 1.036,
  rec@0.9 = 0.0 at **every** hop; `fr7_s1` mean_cos@h1 = -0.0023, effrank ≈
  1.001, rec@0.9 = 0.0 at every hop; `fr7_s2` (the seed the paper uses) mean_cos
  0.930, rec@0.9 0.996→0.881→0.060 (h=1/7/21). **2-of-3 dead. Confirmed.**
- A3: `t1_matrix_permutation_K8_frN_s1.json` → `d=16, K=8, n_params=170896`, no
  `h` field, K/d=0.5; Stage-0 `p100k_baseline_d32_K8_s0.json` → `n_params=175008,
  h=64`, K/d=0.25. The `d=16` anchor is a different task/architecture/K-ratio.
  **Confirmed.**
- A6: `stable_rank_mean` over the four converged frN seeds: min **14.64867**
  (frN_s4, h=21), max **15.59185** (frN_s1, h=6). 14.64867 → **14.6**, not 14.7.
  **Confirmed.**
- Figure 1 caption (self-surfaced): `AGGREGATE_latest.json` M3 has 7 cells; only
  d8_K4 (max 0.987), d16_K8 (0.992), d16_K12 (0.999) reach exact recovery
  unconstrained and are plotted; d16_K4 caps at 0.134 and d32_K16/d64_K32/
  d128_K64 read **0.0 at every `k` including `k>=K`**. "in every cell" is true
  only of converged cells. **Confirmed; caption scope fix is accurate.**

---

## Ordered fix list

### FIX-1: Replace the false sibling-overlap disclosure with an accurate one (in-tree half of A1)

**Severity:** CRITICAL
**File(s):** `sections/07_limitations.tex`
**Location:** final sentence of the `\textbf{Limitations.}` paragraph.
**Before:**
> A concurrent companion submission covers group-composition state tracking, sharing no figures or headline claims.

**After:**
> A concurrent companion submission (group-composition state tracking) shares no figures or tables; its introductory paragraph restates this paper's recruitment, causal-necessity, and composition numbers as background for its own distinct contribution, a group-dimension rank law this paper does not make.

**Why:** A1. The as-written sentence denies a shared *headline claim* that
demonstrably exists — the sibling's `03_binding.tex` "provable foundation
(binding)" paragraph restates all three of this paper's headline results
(recruitment 8.20/15.08, causal step ≤0.0004→0.940, composition 4/5 seeds),
verified directly. A false disclosure in a double-blind venue cannot stand;
the rewrite is honest ("no figures or tables" is true; "no headline claims" is
false and is removed). This closes the in-tree defect. **It does not remove the
salami-slicing risk** — that requires editing the sibling — see ESCALATION-1.

---

### FIX-2: Demote the depth-amplification result to a scoped single-seed case study and disclose the 2-of-3 named instability (A2)

**Severity:** CRITICAL
**File(s):** `main.tex` (abstract), `sections/07_limitations.tex`
**Location (a) — abstract:** the closing clause of the `\begin{abstract}` block.
**Before:**
> and a rank-starved operator's depth-decay curve is predicted from its
> eigenspectrum alone.

**After:**
> and in the single converged rank-starved seed, the depth-decay curve is
> predicted from its eigenspectrum alone.

**Location (b) — Limitations:** first sentence of the `\textbf{Limitations.}`
paragraph.
**Before:**
> The force-rank contrast rests on one converged rank-starved seed, the
> $d \geq 48$ frontier cells on single seeds.

**After:**
> The depth-amplification contrast rests on the single force-rank-$(K{-}1)$
> seed that converged; the other two at that cell collapsed under a numerical
> instability in the spectral-projection backward pass (documented in the
> source design record), so it is read as a case study consistent with the
> spectral prediction, not an $n{>}1$ mechanism. The $d \geq 48$ frontier cells
> likewise rest on single seeds.

**Why:** A2. Verified directly: at the flagship cell, `fr7_s0`/`fr7_s1` are dead
(rec@0.9 = 0.0 at every hop, effective rank ≈ 1.0, `n_skipped_steps` 8/10),
`fr7_s2` is the only converged seed. The abstract's clause reads as an
established general mechanism; scoping it to "the single converged
rank-starved seed" makes it true-as-shown. The Limitations expansion discloses
(i) it is 1-of-3 at that cell, (ii) the missing two are a *named* numerical
instability in the very `truncate_to_rank` mechanism the method relies on (not
slow convergence), and (iii) the demotion from mechanism to case study. The
within-seed prediction (|pred-meas| ≤ 0.008 through h=7, Table 1 /
`app:period`) survives untouched. **Submission path, not camera-ready
deferral** — the reframe ships now; an `n>1` mechanism claim is a future
evidence item.

*Note for the writer:* the §4 "Depth amplification" paragraph already says "the
one converged seed," so no change is needed there — but do **not** re-introduce
general-mechanism phrasing anywhere; the paragraph's "The mechanism is spectral
and predictable" clause is scoped to that seed's operator and is fine as-is.

---

### FIX-3: Drop the spliced `d=16` anchor from the `K=d/4` exactness-frontier sentence (A3)

**Severity:** SERIOUS
**File(s):** `sections/05_frontier.tex`
**Location:** the `\textbf{The frontier that survives}` sentence.
**Before:**
> Fixing $K = d/4$ at encoder width $h = 64$ (every cell confirmed flat before
> being read as converged), mean recovery cosine falls monotonically
> from $\approx 1.00$ at $d{=}16$ to 0.877/0.909/0.915 at $d{=}32$ (three
> seeds, 100K steps), 0.7196 at $d{=}48$ (100K), and 0.3882 at $d{=}64$
> (150K); the exact-match metric is harsher, with $\recninety$
> 0.413/0.632/0.653 at $d{=}32$ (all below the pre-registered 0.7 pass
> bar), 0.002 at $d{=}48$, and 0.0 at $d{=}64$.

**After:**
> Fixing $K = d/4$ at encoder width $h = 64$ (every cell confirmed flat before
> being read as converged), mean recovery cosine falls monotonically across
> $d$: 0.877/0.909/0.915 at $d{=}32$ (three
> seeds, 100K steps), 0.7196 at $d{=}48$ (100K), and 0.3882 at $d{=}64$
> (150K); the exact-match metric is harsher, with $\recninety$
> 0.413/0.632/0.653 at $d{=}32$ (all below the pre-registered 0.7 pass
> bar), 0.002 at $d{=}48$, and 0.0 at $d{=}64$.

**Why:** A3. Verified directly: the "≈1.00 at d=16" point is the `d=16, K=8`
Task-E *composition* model (`n_params=170896`, no `h` field, **K/d=0.5**), not
a `K=d/4`, `h=64` Stage-0 cell — none exists at that config, and
`STAGE0_DESIGN.md` §15.7.1 explicitly puts `d=16, K=8` in the noisier "K=d/2
slice … not monotone-clean … premature" and gives the `K=d/4` slice **no d=16
entry**. Dropping the anchor makes "every cell confirmed flat" cover only real
cells of this sweep; the three genuine points still show a clean monotone
decline. (Acceptable alternative if the writer wants to keep a visual anchor:
retain d=16 only as an explicitly-labeled reference — *"…with the d=16
composition model (K=d/2) exact at ≈1.00 for orientation only, outside this
sweep"* — but the drop is cleaner and is the instruction.)

---

### FIX-4: Add an architecture-family scope qualifier to the central claim (A4)

**Severity:** SERIOUS
**File(s):** `main.tex` (abstract), `sections/07_limitations.tex`
**Location (a) — abstract:** the central sentence introducing the results list.
**Before:**
> Under these conditions gradient descent recruits the rank the task demands:

**After:**
> Under these conditions gradient descent recruits the rank the task demands in
> this architecture family:

**Location (b) — Limitations:** append one clause to the scale sentence.
**Before:**
> Every experiment is synthetic and under 1M parameters; generality at scale is
> untested.

**After:**
> Every experiment is synthetic and under 1M parameters; generality at scale,
> and across state-writing architectures (recurrent fast-weight,
> attention-readout, associative-memory), is untested.

**Why:** A4. The existing "Under these conditions" qualifier scopes the
task/readout, not the architecture; the finding is from one encoder family
(row-reader latents, §2), and the title rhetorically mirrors "The Gradient Does
Not See Rank," inviting a more-general reading than one architecture supports.
Two minimal scope edits close it; no experiment needed.

---

### FIX-5: Disclose that the M3 aggregate does not retain per-seed instability diagnostics (A5)

**Severity:** MINOR
**File(s):** `sections/07_limitations.tex` (Appendix C, `app:repro`)
**Location:** append one sentence to the `\section{Reproducibility}`
(`app:repro`) paragraph. **Route to the appendix, not the body, to protect the
4pp budget.**
**Before:** *(end of the `app:repro` paragraph — append after "…recorded inline
in those documents.")*
**After (append):**
> The force-rank staircase aggregate retains one recovery figure per $(d,K,k)$
> cell, not per-seed instability diagnostics; the necessity direction (recovery
> $\approx 0$ below $K$) is robust to the cause of any individual sub-$K$
> collapse, while the smoothness of the post-knee approach should be read at the
> aggregate level only.

**Why:** A5. The aggregate stores a single float per cell (confirmed), so an
instability-collapsed sub-`K` cell and a genuinely rank-capped one both read
≈0. The load-bearing *necessity* direction survives either way — the attacker
concedes this — so the caveat is a transparency note, not a correction. Placing
it in Appendix C keeps the body at 4pp.

---

### FIX-6: Correct the stable-rank lower bound 14.7 → 14.6 (A6)

**Severity:** MINOR
**File(s):** `sections/04_composition.tex`
**Location:** `\textbf{The invariant-subspace mechanism.}` sentence.
**Before:**
> reads the full ambient dimension (16.0 of $d{=}16$; stable rank 14.7--15.6);

**After:**
> reads the full ambient dimension (16.0 of $d{=}16$; stable rank 14.6--15.6);

**Why:** A6. Verified directly: min `stable_rank_mean` = 14.64867 (frN_s4,
h=21), which rounds to 14.6, not 14.7; max = 15.59185 → 15.6. (If the writer
prefers, the brief's more precise "14.66--15.59" is also acceptable, but "14.6--15.6"
matches the surrounding rounding convention.)

---

### FIX-7: Scope the Figure 1 caption to converged cells (defense self-surfaced)

**Severity:** MINOR
**File(s):** `sections/03_recruitment.tex`
**Location:** last sentence of the `fig:forcerank` caption.
**Before:**
> The step lands at $k \approx K$ in every cell.

**After:**
> The step lands at $k \approx K$ in every cell whose unconstrained model
> reaches exact recovery.

**Why:** Self-surfaced by the defense and independently verified: the figure
plots the three cells (d8_K4, d16_K8, d16_K12) whose unconstrained models reach
exact recovery, but the same `AGGREGATE_latest.json` contains four more M3
cells where no `k≈K` step appears — `d16_K4` caps at 0.134 (the documented
d=16,K=4 ceiling anomaly) and `d32_K16`/`d64_K32`/`d128_K64` read 0.0 at every
`k` including `k>=K` (Stage-0 trainability failures, the very cells §5's
frontier is about). "in every cell" reads as "every cell in the grid"; the
scope qualifier pre-empts a reviewer who opens the aggregate and finds four
all-fail cells. **Decision: enter the fix list** — cheap (≈7 words to a
caption), accurate, and it forecloses a follow-on attack.

---

### FIX-8: Note the benign BLAS RuntimeWarnings in Appendix C (in-tree half of A7)

**Severity:** MINOR
**File(s):** `sections/07_limitations.tex` (Appendix C, `app:repro`)
**Location:** append one sentence to the `\section{Reproducibility}`
(`app:repro`) paragraph (after FIX-5's sentence, or as a footnote). **Appendix,
not body.**
**After (append):**
> The analysis pipeline emits benign BLAS \texttt{RuntimeWarning}s from
> near-singular intermediate products on some platforms; every decomposition
> output is asserted finite before use, so these do not affect any reported
> number.

**Why:** A7. The warnings are documented-benign but not actually suppressed
(the existing `warnings.filterwarnings(..., module="numpy")` predicate does not
match numpy's matmul warnings, which are attributed to the calling module), so
a reviewer rerunning from the code link sees ~60 warnings that Appendix C's
"changed artifact fails the build" language does not prepare them for. The
in-tree fix is this one-sentence note. **Out-of-tree, optional escalation (not
a writer instruction, not a blocker):** making the filter effective in
`matrix-thinking/chapter2/analyze_zdump.py` (drop the `module="numpy"`
predicate or wrap the near-singular matmuls in `np.errstate(all="ignore")`) is
a code hygiene item for whoever owns that file — flag to the coordinator, do
not edit from the paper tree.

---

### Page-budget accounting (writer: apply, recompile, then check)

Body is at exactly 4pp; appendices are excluded. Net effect of the fixes:

- **Free / no body cost:** FIX-5 and FIX-8 land in Appendix C (excluded).
  FIX-3 removes "from ≈1.00 at d=16 to" (frees ~½ line in §5). FIX-2a, FIX-4a,
  FIX-6 are roughly length-neutral in the abstract/§4.
- **Small body add (all in the short Limitations paragraph):** FIX-1 (~+1
  line), FIX-2b (~+2 lines), FIX-4b (~+1 line), FIX-7 (~+1 caption line).
- **Designated reserve cut — apply ONLY IF the recompile exceeds 4pp:** in §4
  (`04_composition.tex`), delete the sentence *"dropping one of $K{=}8$ modes
  leaves per-item cosine near $\sqrt{1 - 1/8} \approx 0.94$ at shallow depth,
  and repeated self-application amplifies the deficit past the bar
  (Figure~\ref{fig:depth})."* It restates what Table 1 (`tab:depthcurve`, in
  Appendix A) and Figure 2 already show and carries no independent claim, so
  its removal loses nothing at the claim level. This clears ~2 lines, more than
  covering the Limitations additions.

Recompile `main.pdf` (and rebuild `bundle/` via `make bundle`) after the fixes;
confirm the body still ends at ≤4pp before the detector gate.

---

## Verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 | CRITICAL | CONCEDE+FIX (escalate cross-paper) | **PARTIAL — in-tree false-disclosure closed; cross-paper overlap survives, escalated** | FIX-1 (+ ESCALATION-1) |
| A2 | CRITICAL | CONCEDE+FIX | DEFENSE VALID BUT EDIT (resolves CRITICAL) | FIX-2 |
| A3 | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-3 |
| A4 | SERIOUS | PARTIAL | DEFENSE VALID BUT EDIT | FIX-4 |
| A5 | SERIOUS (hedged) | PARTIAL | DEFENSE VALID BUT EDIT (effectively MINOR) | FIX-5 |
| A6 | MINOR | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-6 |
| A7 | MINOR | PARTIAL | DEFENSE VALID BUT EDIT (in-tree = Appendix-C note) | FIX-8 |
| Fig 1 caption | — (self-surfaced, MINOR) | defense self-surfaced | EDIT | FIX-7 |

**Disposition counts:** 0 DEFENSE VALID · 6 DEFENSE VALID BUT EDIT · 0 DEFENSE
INSUFFICIENT · 1 PARTIAL (A1). (The Figure 1 caption is a self-surfaced EDIT,
not one of the eight attacks.)

**CRITICAL status:** 0 open in-tree. A2 resolved by FIX-2. A1's in-tree defect
resolved by FIX-1; A1's cross-paper residual is escalated (ESCALATION-1), not
open against this paper's writer.

---

## ESCALATION-1 (PI / coordinator — outside this run, blocks the JOINT submission)

The writer **cannot** touch `papers/neurreps-ea/` (READ-ONLY, separately-run
accept-ready track). FIX-1 *discloses* the overlap accurately but does not
*remove* it. The durable de-duplication — which only the coordinator can do —
is:

1. Convert the sibling's `03_binding.tex` "provable foundation (binding)"
   paragraph from a **restatement of this paper's numbers** (8.20/15.08;
   ≤0.0004→0.940; 4/5 seeds) to a **pointer-cite** ("see the companion
   submission for the recruitment grid, force-rank staircase, and composition
   mechanism"), keeping only what the sibling's own contribution needs.
2. De-duplicate the near-identical `app:period` appendix prose across the two
   papers (both derive `π^21 = π^{21 mod 8} = π^5` in paraphrased-parallel
   sentences).

Until this is done, a reviewer assigned both same-venue, same-window EAs can
read the pair as salami-slicing regardless of FIX-1's accurate disclosure.
**This is a blocking precondition on submitting both EAs to NeurReps 2026, and
it lives with the human who owns both trees — record the decision before either
paper is submitted.**

---

## Residual risk after all fixes

- **A1 cross-paper overlap (workshop-relevant, JOINT-submission-blocking):** the
  single genuinely unresolved item. In-tree the paper is now honest; the risk
  is entirely a function of whether ESCALATION-1 is executed on the sibling.
  Workshop-survivable *only if* the coordinator de-dups the sibling; if both
  EAs ship with the sibling's restatement intact, both are exposed. **Owner:
  PI/coordinator.**
- **A2 residual (workshop-survivable):** even after the honest reframe, the
  flagship depth-amplification result is a **single converged seed** whose two
  same-cell siblings died to a numerical instability, and *all*
  provably-sufficient force-rank seeds also died (per the design record). A
  hostile conference reviewer could still discount the amplification claim as
  `n=1`. For a NeurReps EA (non-archival, 4pp, mechanism-as-case-study is
  acceptable when disclosed) this is survivable; for the ICLR-2027 flagship it
  is **conference-blocking** and needs `n>1` under an instability-guarded
  projection. Camera-ready / future-evidence item.
- **A3 residual (none material):** after dropping the anchor, the frontier claim
  rests on three genuine cells; monotone decline holds. Closed.
- **A4 residual (workshop-survivable):** architecture-generality is now flagged
  as untested; the title still rhetorically generalizes, but the scoped
  abstract sentence + Limitations clause make the actual claim honest.
- **A5 / A7 residual (none material):** transparency notes; no load-bearing
  claim depends on them. A7's code-level filter (out-of-tree) remains a
  cosmetic reproducibility wart until the coordinator fixes
  `analyze_zdump.py`.
- **A6 residual (none):** one-digit correction, closed.

No new attack surface is created by the fixes, provided FIX-2 does not
re-introduce general-mechanism phrasing anywhere and FIX-1's new disclosure is
not itself over/under-stated (both are checked in the re-attack below).

---

## Affected claims — scoped re-attack list

After the fixes, these must re-enter a **targeted** attack/defense/rebuttal pass
(not a full restart). The two structural reframes (A1, A2) are the priority;
the rest are confirmation-only.

- **Abstract** — FIX-2a (depth clause scoping) and FIX-4a (architecture
  qualifier). Re-check: no residual "established/general mechanism" reading of
  the depth clause; the architecture qualifier reads cleanly with the results
  list. *(Affects claims R4, and the central recruitment claim's scope.)*
- **§4 `04_composition.tex`** — FIX-6 (14.6) and the FIX-2 knock-on. Re-check:
  the depth-amplification paragraph reads as a scoped case study end-to-end; no
  general-mechanism phrasing survives. *(Claims R4, R5.)*
- **§5 `05_frontier.tex`** — FIX-3. Re-check: the `K=d/4` sentence no longer
  references d=16; "every cell confirmed flat" now covers only d=32/48/64.
  *(Claim R7.)*
- **§7 `07_limitations.tex`** — FIX-1, FIX-2b, FIX-4b, FIX-5, FIX-8. Re-check:
  the new companion disclosure is accurate (matches the sibling's actual
  content) and neither over- nor under-claims; the depth-amplification
  disclosure names the instability correctly. *(Claim scope, R9; A1/A2
  disclosure accuracy.)*
- **§3 `03_recruitment.tex`** — FIX-7 caption scope. Re-check: the qualifier
  matches the plotted cells and the aggregate's non-converged cells.
  *(Figure 1 / claim R2 framing.)*
- **Appendix A/C** — FIX-5, FIX-8 land here; confirm they compile and the
  depth-curve table (`tab:depthcurve`) is unaffected.

Re-attack priority: **A1-disclosure accuracy** and **A2 case-study
completeness** are the two that genuinely need an adversarial second look; A3
and the MINORs are near-mechanical.

---

## New citations

### MUST-CITE

- **DeltaNet — Yang, Wang, Shen, Panda & Kim, "Parallelizing Linear
  Transformers with the Delta Rule over Sequence Length," NeurIPS 2024,
  arXiv:2406.06484.** Both the attacker and the defender flag this as the top
  gap, and it is nearly free to add: the paper already cites
  `\citet{siems2025deltaproduct}` (DeltaProduct) in the same related-work
  sentence as the Householder-product extension of exactly this method. Citing
  the extension without its antecedent is the kind of omission
  linear-attention/fast-weight reviewers (the paper's own
  Nazari/Sun/Grazzi/Siems neighborhood) notice immediately.
  **Insertion (compact, ~5 words):** in `sections/06_related.tex`, change
  *"\citet{grazzi2025negative} and \citet{siems2025deltaproduct} characterize
  the per-token \emph{transition} operator"* to *"\citet{grazzi2025negative}
  and \citet{siems2025deltaproduct} (the Householder extension of
  \citealp{yang2024deltanet}) characterize the per-token \emph{transition}
  operator"*. Add the `yang2024deltanet` entry to `refs.bib`.
  **Writer must verify the bibtex** (authors, venue, arXiv id) before adding —
  do not trust this line from memory; confirm against the arXiv abstract page.

### SHOULD-CITE (optional; the 0.30pp related-work budget likely fits at most one)

- **Gated Linear Attention (GLA) — Yang et al., ICML 2024, arXiv:2312.06635.**
  Sits with the Nazari/Sun observational effective-rank line (gating's effect
  on effective state rank). If only one SHOULD-CITE fits, prefer this one — it
  is closest to the *measured object* (trained state rank) the paper is about.
- **Titans — Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at Test
  Time," arXiv:2501.00663 (2025).** Matches the introduction's "matrix state as
  a capacity-budgeted associative store" framing; recent and prominent for a
  NeurReps audience, but further from the paper's rank-measurement core than
  GLA. Defer unless the budget clearly allows both.

Neither SHOULD-CITE is a blocker; add only if the 0.30pp related-work section
absorbs them without pushing the body past 4pp. Verify all bibtex entries
before adding.
