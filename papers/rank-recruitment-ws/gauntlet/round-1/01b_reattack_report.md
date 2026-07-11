# Scoped re-attack report — round 1, stage 01b — `papers/rank-recruitment-ws/` ("When the Gradient Sees Rank")

**Reviewer stance:** fresh-context hostile reviewer, no memory of prior rounds.
This is a SCOPED re-attack: I verify only whether the 8 applied fixes actually
resolve the attacks on the affected claims, and whether the fixes broke or newly
exposed anything. Every load-bearing factual claim was re-verified read-only
against the raw JSON artifacts and the source records myself (paths and recomputed
values below). I did not edit any paper file.

## Verdict summary

| Item | Verdict | Severity now |
|---|---|---|
| FIX-1 (A1, was CRITICAL) — sibling-disclosure accuracy | **CLOSED** in-tree | — (cross-paper residual = ESCALATION-1, PI-owned, unchanged) |
| FIX-2 (A2, was CRITICAL) — depth-decay scoping + instability disclosure | **CLOSED** | — |
| FIX-3 (A3, SERIOUS) — drop d=16 from K=d/4 sentence | **CLOSED** | — |
| FIX-4 (A4, SERIOUS) — architecture-family qualifier | **CLOSED** | — |
| FIX-5 (A5, MINOR) — Appendix C per-seed-diagnostics caveat | **CLOSED** | — |
| FIX-6 (A6, MINOR) — stable rank 14.7→14.6 | **CLOSED** | — |
| FIX-7 (Fig 1 caption scope) | **CLOSED** | — |
| FIX-8 (A7, MINOR) — Appendix C BLAS-warning note | **CLOSED** | — |
| New citation `yang2024deltanet` | **CLOSED** — bibtex accurate, correctly used | — |
| Page-fit integrity (compressions + evidence comments) | **CLOSED** — no claim changed/dropped | — |

**No CRITICAL remains open in-tree. No new findings.** The one residual — the
sibling's near-duplicate `app:period` prose and its restatement of these numbers
— is the pre-existing ESCALATION-1 (cross-paper, PI/coordinator-owned); it is not
a writer-fixable in-tree defect and FIX-1 correctly does not claim it away.

---

## Per-item detail

### FIX-1 (A1) — sibling-disclosure accuracy — CLOSED (in-tree)

Revised sentence (`sections/06_related.tex`, Limitations paragraph):

> A concurrent companion submission (group-composition state tracking) shares no
> figures or tables but restates this paper's recruitment, causal-necessity, and
> composition numbers as background for its distinct contribution, a
> group-dimension rank law this paper does not make.

I read the sibling directly (`papers/neurreps-ea/sections/03_binding.tex`, the
"provable foundation (binding)" paragraph). It restates **all three** number
families:
- **recruitment:** "at d = 16 effective rank reaches 8.20 at K = 8 and 15.08 at
  K = 16";
- **causal necessity:** "a spectral rank cap at d = 8, K = 4 leaves exact
  recovery ≤ 0.0004 for every cap k ≤ 3 and restores it to 0.940 at k = 4, a step
  exactly at the provable bound";
- **composition:** "the same operator composes exactly (Z^h, 21 self-applications)
  in four of five seeds."

The revised sentence is therefore **accurate and not over/under-stated**:
- "shares no figures or tables" is TRUE — the sibling's tables (`tab:groups`,
  `tab:m1`, `tab:gate1a`) and figure (`fig:tracking`) are all group-specific
  (S₃/S₄/A₅/S₅/A₆) and share nothing with this paper's `fig_forcerank`,
  `fig_depth`, `tab:depthcurve`, `tab:subspace`;
- "restates ... recruitment, causal-necessity, and composition numbers" is TRUE
  and errs on the honest side (it discloses the restatement rather than denying a
  headline overlap, which was the original CRITICAL defect: the false "sharing no
  ... headline claims");
- "a group-dimension rank law this paper does not make" is TRUE — the sibling's
  own contribution (d_min tracking, S₄/A₅ TOST, the five-group razor) is absent
  from this paper.

**Meaning-survival of the page-fit shortening (checked as instructed):** the
rebuttal proposed "…shares no figures or tables; its introductory paragraph
restates … for its own distinct contribution …"; the applied text uses
"…shares no figures or tables **but restates** … for its **distinct**
contribution …". The "but restates" construction preserves (arguably sharpens)
the disclosure; dropping "its introductory paragraph" loses only the location of
the restatement, immaterial to accuracy; "distinct" vs "own distinct" is
semantically identical. **Meaning survived.**

Minor nuance re-confirmed (does not change the verdict): the sibling's causal
figure is the 1234-run replication value **0.940**, whereas this paper's abstract
uses the 991-run **0.97**; recruitment 8.20 matches, 15.08 vs this paper's 15.09
is a replication-snapshot rounding, composition "four of five seeds" matches. The
disclosure's phrase "restates … numbers" is a fair (if anything conservative)
characterization of a same-result-family overlap with mostly-matching figures —
it does not assert digit-identity, so it is not over-stated.

**Cross-paper residual (unchanged, NOT a new finding, NOT in-tree):** the two
papers' `app:period` appendices remain near-duplicate prose (both derive
π²¹ = π^(21 mod 8) = π⁵ in paraphrased-parallel sentences — confirmed against
`papers/neurreps-ea/sections/07_limitations.tex` `app:period`). A disclosure
sentence in this paper cannot remove that; it is exactly ESCALATION-1 (PI/
coordinator-owned, blocks the JOINT two-paper submission). The writer's FIX-1
correctly discloses the overlap without claiming to have removed it.

**Verdict: CLOSED in-tree.** The false-disclosure CRITICAL is resolved; the
residual is the escalated cross-paper item, not open against this writer.

### FIX-2 (A2) — depth-decay scoping + instability disclosure — CLOSED

**No general-mechanism reading survives anywhere.** Checked every locus:
- **Abstract** (`main.tex`): "and **in the single converged rank-starved seed**,
  the depth-decay curve is predicted from its eigenspectrum alone." Scoped.
- **§4** (`04_composition.tex`, "Depth amplification"): "…**the one converged
  seed** passes shallow hops…"; "The mechanism is spectral and predictable: from
  **the operator's** eigenspectrum alone…" — scoped to that seed's operator.
- **Appendix A** (`07_limitations.tex`, `app:period`): "**in the rank-starved
  operator** the same 16 additional applications compound … which is exactly the
  amplification claim of §4, **not a generalization claim**." Scoped.
- **Limitations** (`06_related.tex`): demotes to case study explicitly (below).

**Instability disclosure accuracy — verified against the raws.** Applied
Limitations text:

> The depth-amplification contrast rests on the single converged
> force-rank-(K−1) seed; the other two at that cell collapsed under a documented
> numerical instability in the spectral-projection backward pass, so it is a case
> study consistent with the spectral prediction, not an n>1 mechanism; the d ≥ 48
> frontier cells rest on single seeds.

Recomputed from
`experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_fr7_s{0,1,2}.json`:
- `n_skipped_steps` = **8 / 10 / 2** (s0/s1/s2) — matches.
- `fr7_s0`: rec@0.9 = **0.0 at every hop** (h=1..7,21), mean_cos ≈ 0.0003/−0.001/…
  (statistically random). `fr7_s1`: rec@0.9 = **0.0 at every hop**, mean_cos ≈
  −0.002/… **2-of-3 dead confirmed.**
- `fr7_s2` (the seed the paper uses): rec@0.9 0.996→0.881→0.060 (h=1/7/21),
  mean_cos 0.9303→0.9163→0.8206. **The one converged seed confirmed.**

The word "documented" is honest: `matrix-thinking/chapter2/TASK_E_FINDINGS.md`
§9 (line 518) reads verbatim *"`fr7` s0/s1 (the eigh-backward-instability dead
runs, `n_skipped_steps` 8 and 10) collapse `A` to `effective_rank(A) ≈ 1.00`"*,
and §9 (lines 250–251) names it *"eigh-backward numerical instability."* The
spectral projection's backward pass **is** the eigh backward pass, so
"numerical instability in the spectral-projection backward pass" is exact. The
writer's shortening of the rebuttal's "(documented in the source design record)"
to "a documented numerical instability" is **still honest** — it asserts a
record exists (true) without citing an internal, non-anonymizable doc, which is
the right call for a review build.

The within-seed spectral prediction survives untouched: `app:period` Table 1
measured column (0.9303/0.9259/0.9212/0.9163/0.8206) matches `fr7_s2` raw
exactly; |pred−meas| ≤ 0.008 through h=7 holds.

**Verdict: CLOSED.** CRITICAL resolved; the result ships as an honestly-scoped
single-seed case study.

### FIX-3 (A3) — drop d=16 from the K=d/4 sentence — CLOSED

`sections/05_frontier.tex` now reads: "…mean recovery cosine falls monotonically
**across d**: 0.877/0.909/0.915 at d=32 …, 0.7196 at d=48 …, and 0.3882 at
d=64…". The spliced "≈1.00 at d=16" anchor (which was the d=16,K=8 Task-E
*composition* model, K/d=0.5, n_params=170896, no `h` field — a different
task/architecture/K-ratio) is gone. "Every cell confirmed flat before being read
as converged" now covers only the three genuine K=d/4, h=64 Stage-0 cells
(d=32/48/64). The monotone-decline claim holds on the real cells alone.
**CLOSED.**

### FIX-4 (A4) — architecture-family qualifier — CLOSED

- Abstract central sentence: "gradient descent recruits the rank the task demands
  **in this architecture family**:" — reads cleanly into the results list.
- Limitations: "generality at scale, **and across state-writing architectures
  (recurrent fast-weight, attention-readout, associative-memory), is untested.**"
  The named families match A4's examples (DeltaNet-style recurrent fast-weight,
  attention-readout, Hopfield-style associative memory). **CLOSED.**

### FIX-5 (A5) — Appendix C per-seed-diagnostics caveat — CLOSED

Present in `07_limitations.tex` (`app:repro`): "The force-rank staircase
aggregate retains one recovery figure per (d, K, k) cell, not per-seed
instability diagnostics; the necessity direction (recovery ≈ 0 below K) is robust
to the cause of any individual sub-K collapse, while the smoothness of the
post-knee approach should be read at the aggregate level only." Matches the
rebuttal's required content; in the appendix (off the 4pp body). **CLOSED.**

### FIX-6 (A6) — stable rank 14.7 → 14.6 — CLOSED

`sections/04_composition.tex` now reads "stable rank 14.6--15.6." Recomputed
`stable_rank_mean` across all hops of the four converged frN seeds
(`..._K8_frN_s{1,2,3,4}.json`): overall **min 14.64867** (s4, h=21) → 14.6;
overall **max 15.59185** (s1, h=6) → 15.6. The paper's "14.6--15.6" is the
correct 1-dp rounding (and more accurate than the brief's own "14.66"). **CLOSED.**

### FIX-7 (Fig 1 caption) — CLOSED

`sections/03_recruitment.tex` `fig:forcerank` caption: "The step lands at k ≈ K
in every cell **whose unconstrained model reaches exact recovery**." This scopes
away the four non-converged M3 cells (d16_K4 caps at 0.134; d32_K16/d64_K32/
d128_K64 read 0.0 at every k) that the earlier unqualified "in every cell" would
have exposed. **CLOSED.**

### FIX-8 (A7) — Appendix C BLAS-warning note — CLOSED

Present in `07_limitations.tex` (`app:repro`): "The analysis pipeline emits benign
BLAS RuntimeWarnings from near-singular intermediate products on some platforms;
every decomposition output is asserted finite before use, so these do not affect
any reported number." In the appendix, accurate to the observed behavior.
**CLOSED.**

### New citation `yang2024deltanet` — CLOSED (bibtex accurate + correctly used)

`refs.bib` entry:
- title "Parallelizing Linear Transformers with the Delta Rule over Sequence
  Length" ✓
- authors "Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim,
  Yoon" ✓ (exactly the required Yang, Wang, Zhang, Shen, Kim)
- `note = {arXiv:2406.06484}` ✓
- `booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}`,
  `year = 2024` ✓

The bib header comment documents that the writer verified this against the arXiv
Atom feed and DBLP and **corrected** the rebuttal's from-memory "Panda" author
list — the writer caught the rebuttal's own error. Used in `sections/06_related.tex`:
"…\citet{siems2025deltaproduct} (the Householder extension of
\citealp{yang2024deltanet}) characterize the per-token transition operator…" —
the antecedent-before-extension insertion the rebuttal required. **CLOSED.**

### Page-fit integrity — CLOSED (no claim changed or dropped)

- **Every evidence comment survived.** Full inventory across `main.tex` +
  `sections/`: R1, R1b, R2, R2b, R3, R4, R5, R6, R7, R8, R9, R10 all present
  (R1×2, R1b×1, R2×3, R2b×1, R3×4, R4×6, R5×4, R6×2, R7×2, R8×3, R9×2, R10×1).
  None dropped.
- **Intro contribution tail:** all four contribution bullets present
  (teeth/§2; recruitment+causal/§3; composition+mechanism+depth-amplification/§4;
  frontier+budget-rule/§5). No claim lost.
- **§2 registration parenthetical → Appendix C:** the "one metric re-registration,
  logged before the affected sweep" disclosure is preserved verbatim in
  `app:repro`. Relocated, not dropped.
- **§3 openers:** recruitment numbers intact (2.42/8.20/15.09; ρ=1.0; band left
  only at K=1,2), matching R1.
- **Figure heights:** both figures at `width=0.92\textwidth` with their R2/R3/R4
  caption evidence comments intact; no captioned number altered.
- **Falsifiers sentence → Appendix C with body pointer:** body pointer "The
  registered falsifiers stand (Appendix~\ref{app:repro})" present in Limitations;
  the full pre-registered falsifier ("force-rank-(K−1) at ≥ 0.9× ceiling at
  multiple (d, K) points, or a d ≥ 32 cell clearing the 0.7 bar, would overturn
  the causal claim") present in `app:repro`. Matches the brief's thesis falsifier.
  Not dropped.
- **Designated reserve cut applied:** the §4 "√(1−1/8) ≈ 0.94" intuition sentence
  is absent — the rebuttal's reserve cut was applied. It carried no `evidence: Rx`
  number and restated the figure/table, so its removal is claim-neutral (the
  measured shallow cosine 0.92–0.93 remains stated in §4). Not a finding.
- `main.pdf` compiled (7 pp total = 4pp body + refs + Appendices A/B/C, the
  appendices excluded from the 4pp count); `tab:depthcurve` values match the raw
  `fr7_s2` and are unaffected by the FIX-5/FIX-8 additions to the separate
  `app:repro` section. Body page-fit vs the 4pp limit is a render-gate concern,
  not a claim defect, and is outside this content re-attack's mandate.

**CLOSED.**

---

## Bottom line

All 8 fixes plus the new citation land as required; no attack survives at its
original severity, and **no CRITICAL remains open in-tree.** No fix introduced a
new defect. The sole residual is the pre-existing, PI-owned **ESCALATION-1**
(sibling `app:period` prose duplication + the sibling's restatement of these
numbers), which blocks the JOINT two-paper submission and which FIX-1 correctly
discloses but cannot itself remove. This tree is clear to proceed to the detector
gate.
