# Rebuttal Report — Round 1 — "Three Bounds on a Null"

## 0. CRITICAL-open status (read first)

**No CRITICAL remains open *after* the fix list is applied — conditional on FIX-1
and FIX-2 being applied and the affected claims being re-run.**

- **A1 (CRITICAL, provenance):** the defense conceded; the attack's factual claim
  is true and I could not weaken it. It is resolved *below CRITICAL* by **FIX-1**,
  a disclosure-and-reframe edit that needs **no new experiment** and whose bulk can
  live in Appendix A (which does not count against the 4-page limit). Until FIX-1
  is applied, A1 is **open**.
- **A2 (rated CRITICAL, adjudicated down to SERIOUS):** the defense's `holds()`
  vs. `det32` distinction is correct and load-bearing — the phrase "the *one*
  determinate signal" denotes the single cell satisfying the full pre-registered
  differential condition (verified: exactly 1 of 20 primary cells), not "one CI
  excludes zero." That, plus the independent secondary-readout corroboration at the
  same cell, defeats the attack's "indistinguishable from noise" framing. What
  survives is a real **disclosure gap** (the 40-test denominator, the 3-hit count,
  the term ambiguity, the c=1000 clustering), which is **SERIOUS, not CRITICAL**,
  and is closed by **FIX-2**. A2 does **not** hold a CRITICAL open.

Both CRITICAL fixes are framing/disclosure edits, not new measurements, so the
gauntlet can proceed once FIX-1 and FIX-2 land and the transient-provenance thread
is re-run (see §5, Re-run instruction).

The one item I want on the coordinator's radar even though it does not gate the
gauntlet: **A6** (no real-kernel positive control for the headline instrument) is
the most substantive of the SERIOUS attacks and the biggest residual on Bound 1.
The gauntlet rule is satisfied by an honest limitation sentence (FIX-6, tier 1),
but the *real* fix — a single synthetic forward pass, minutes of GPU — is cheap
relative to how much of the paper leans on Bound 1, and I recommend building it
before submission rather than deferring to camera-ready (see FIX-6 and Residual
Risk #1).

---

## 1. Summary for the edit agent

**Disposition counts:** 0 DEFENSE VALID · 4 DEFENSE VALID BUT EDIT (A3, A4, A7, A8)
· 2 DEFENSE INSUFFICIENT (A1, A5) · 2 PARTIAL (A2, A6). Every attack requires an
edit; none of the eight is dismissed.

**The three structural fixes that carry the weight:**

1. **FIX-1 (A1) — provenance of the transient.** The paper presents "4 (corpus,
   arm) contrasts, independently classified" (§4, Table 1, and by implication the
   abstract/§1) as the pre-registered unit of analysis. It is not: the registered
   pipeline classified *two corpora* (using the global arm as each corpus's
   representative) and returned *both UNRESOLVED*; the 4-way per-arm split is a
   disclosed build-time re-derivation performed *after* that null output was seen,
   later folded into code and re-validated against the archived deltas. The paper
   must say so. This touches the abstract, §1, Table 1, §4, §5, and §7. The honest
   reframe (register-level UNRESOLVED is the headline; the transient is a disclosed
   per-arm re-derivation) actually *strengthens* the null.

2. **FIX-2 (A2) — multiplicity and terminology.** State the denominator (40
   uncorrected 95% CI tests), report that **three** K=32 intervals exclude zero
   (≈ the 2 expected by chance), disambiguate "determinate signal" (= the compound
   `holds()` condition, fires once) from "determinate reading" (= any CI-excludes-
   zero, three of them), and note honestly that two of the three share checkpoint
   1,000 across different cells — consistent with checkpoint-level sampling noise.
   **Do not** repeat the defense's arm-asymmetry argument (it is unsound: all four
   c=1000 cells are negative, exactly what a shared control-dip predicts; only CI
   width differs). The transient stays distinguished by the differential condition
   *and* the independent secondary readout — say only that.

3. **FIX-5 + FIX-6 (A5, A6) — the two substantive discussion additions.** Add
   Grazzi et al. (2025) as a *competing theoretical account* for the null (a
   positive-eigenvalue DeltaNet expressivity ceiling), not just a citation, and add
   an explicit limitation that the readout has no real-kernel positive control so
   the "instrument verdict" reading cannot fully exclude a systematic extraction/
   convention bug. These two make the paper's central interpretive claims honest.

**Page constraint.** The draft is exactly 4.0 pages (the hard cap). The additions
concentrate in §4 (FIX-1, FIX-2), §6 (FIX-5), and §7 (FIX-5, FIX-6). The release
valve is the **appendix**: Appendices A–C and the figures do **not** count against
the limit, so push the full provenance table (A1), the full 3-cell multiplicity
detail (A2), and the F-test note (A3) into Appendix A, keep the main-text versions
compact (the wordings below already are), and recover main-text space with the
named cuts in each fix. Net main-text growth after the offloads and cuts should be
near zero.

---

## 2. The ordered fix list

### FIX-1: Disclose that the transient is a post-hoc per-arm re-derivation, not the registered pipeline's verdict
**Severity:** CRITICAL (resolves A1 below CRITICAL)
**File(s):** `sections/04_behavioral_contrast.tex` (main text); `sections/03_geometric_null.tex` (Table 1); `main.tex` (Appendix A); and knock-on precision edits to `sections/00_abstract.tex` and `sections/01_introduction.tex` (folded into FIX-2's precision wording).
**Location:** §4 "Results" paragraph; Table 1 wave-5 row; Appendix A "Behavioral-contrast classification (wave 5)" paragraph.

**Why:** A1. The registered `analyze_corpus` computed one classification per corpus
using the global arm as representative and returned both corpora UNRESOLVED
(verified in `PHASE2B_SUMMARY.json`: `trajectories` has exactly two keys, both
UNRESOLVED, no per-arm field). The 4-way per-arm breakdown the paper reports is a
build-time re-derivation the design registry itself (§16.18.3) calls a "build-time
scoping finding... registered as a follow-up fix, not self-authorized here" —
produced *after* the corpus-level null was seen. Presenting it as the
pre-registered unit of analysis is the exact researcher-degree-of-freedom
pre-registration exists to close. The underlying Δ statistic *was* pre-registered
per-arm and the CI arithmetic is independently verified, so the finding is not
retracted — only its provenance must be disclosed and its framing demoted.

**Before (§4 Results):**
```
\textbf{Results.} Three of four (corpus, arm) contrasts classify as
unresolved. % evidence: C5-power
The fourth (encyclopedic corpus, per-token arm) classifies as transient:
```
**After (§4 Results):**
```
\textbf{Results.} The registered corpus-level classifier used the global
arm as each corpus's representative signal and returned both corpora
unresolved. Reclassifying each arm's own trajectory at the same registered
primitives---a disclosed build-time re-derivation, since folded into the
analysis code and re-validated against the archived per-cell deltas
(identical verdicts; Appendix~\ref{app:gates})---splits this into four
(corpus, arm) contrasts, three unresolved. % evidence: C5-power
The fourth (encyclopedic corpus, per-token arm) classifies as transient:
```

**Before (Table 1 row, `sections/03_geometric_null.tex`):**
```
5 behavioral contrast & 18 cells, 4 contrasts & 3 unresolved, 1 transient & power-bounded \\ % evidence: C5-power
```
**After (Table 1 row):**
```
5 behavioral contrast & 18 cells, 4 contrasts$^{\dagger}$ & 3 unresolved, 1 transient & power-bounded \\ % evidence: C5-power
```
and append to the Table 1 caption (`\caption{The program at a glance. ...}`) the
sentence:
```
$^{\dagger}$Per-arm re-derivation; the registered corpus-level pipeline returned both corpora unresolved (Appendix~\ref{app:gates}).
```

**Appendix A insertion** (in `main.tex`, end of the "Behavioral-contrast
classification (wave 5)" paragraph — this is supplementary, so it costs no main-text
page budget; put the full detail here):
```
The registered pipeline computed one classification per corpus, using the
global arm as that corpus's representative signal, and classified both
corpora unresolved. The four-way (corpus, arm) breakdown reported in the
main text and Table~\ref{tab:program} applies the same registered
classification primitives to each arm's own trajectory. This finer split
was a build-time re-derivation performed after the corpus-level output was
seen; it was later folded into the analysis code and re-validated against
the archived per-cell deltas, reproducing the same four verdicts. It is
disclosed here as a re-derivation, not presented as the registered
corpus-level verdict.
```

**Space:** the §4 Results replacement adds ~2 lines; recover them via the inline-
number trim in FIX-2 (the K=20 and c=5000 interval numbers move to the Fig 3
caption). The Appendix and Table-caption additions are free.

---

### FIX-2: State the 40-test denominator, report all three CI-excursions, disambiguate "determinate signal," and contextualize the c=1000 clustering
**Severity:** CRITICAL-rated attack, adjudicated SERIOUS (resolves the surviving reduced form of A2)
**File(s):** `sections/04_behavioral_contrast.tex`; precision edits to `sections/00_abstract.tex` and `sections/01_introduction.tex`; full detail to Appendix A (`main.tex`).
**Location:** §4 "Sign discipline" paragraph; abstract sentence; §1 power-bound bullet.

**Why:** A2. The primary table computes 40 uncorrected 95% CIs (4 contrasts × 5
checkpoints × 2 loads); under a global null ≈2 exclude zero by chance. Three do
(verified from the raw trajectory JSONs): `openr1×global@1000` (Δ=−0.203,
CI [−0.402,−0.004]), `wikitext×per_token@1000` (Δ=−0.168, CI [−0.301,−0.036]),
`wikitext×per_token@2500` (Δ=−0.500, CI [−0.624,−0.376], the reported transient).
The paper never states the denominator, never reports that the count is three, and
never distinguishes "the *one* determinate signal" (the compound `holds()`
condition: det32 ∧ ¬det20 ∧ |Δ32|>|Δ20|, which fires at exactly 1 of 20 cells)
from "every determinate high-load reading" (any det32, three cells). The
adjudicated position: the paper's narrowest claim is factually correct and the
transient is genuinely distinguished (it is the only cell meeting the full
condition, and the independent held-out-hop readout corroborates it at the same
cell/checkpoint/direction), so this is **not** "indistinguishable from noise" — but
the disclosure gap is real and must close. **Note for the writer:** do *not* adopt
the defense's claim that arm-asymmetry rules out a shared-checkpoint artifact — it
does not (all four c=1000 cells are negative, precisely what a transiently-low
control loss at c=1000 predicts; the fire/no-fire difference is CI width, not
sign). Disclose the clustering as consistent-with-chance instead.

**Before (§4 Sign discipline):**
```
\textbf{Sign discipline.} The intervention arm's loss sits \emph{above}
control: the one determinate signal points in the harm direction, and
every determinate high-load reading in the primary table is negative.
% evidence: C6-transient
```
**After (§4 Sign discipline):**
```
\textbf{Sign discipline and multiplicity.} The primary table computes 40
uncorrected 95-percent intervals (4 contrasts $\times$ 5 checkpoints
$\times$ 2 loads); three $K{=}32$ intervals exclude zero---near the two
expected by chance at this count---all in the harm direction (arm loss
above control). Only the checkpoint-2{,}500 encyclopedic per-token cell
also satisfies the full pre-registered differential condition ($K{=}32$
determinate, $K{=}20$ not, $|\Delta_{32}|>|\Delta_{20}|$), and only it is
corroborated by the independent held-out-hop readout; that is the sense in
which the signal is singular. The other two exclusions share checkpoint
1{,}000 across different cells, consistent with checkpoint-level sampling
noise rather than an arm-specific effect. % evidence: C6-transient
```

**Abstract precision edit** (`sections/00_abstract.tex`) — makes "single" precise
without bloating the abstract:
Before:
```
the single determinate signal is a transient deviation in the harm
direction.
```
After:
```
the sole reading meeting the pre-registered differential condition (one of
three that cross zero across forty tests) is a transient, harm-direction
deviation.
```

**§1 power-bound bullet precision edit** (`sections/01_introduction.tex`):
Before:
```
leaves three of four contrasts unresolved; its single determinate signal is
a transient deviation in the harm direction. % evidence: C6-transient
```
After:
```
leaves three of four contrasts unresolved; the sole reading meeting the
pre-registered differential condition is a transient, harm-direction
deviation. % evidence: C6-transient
```

**Space recovery in §4** (to fund FIX-1 + FIX-2 additions): in the §4 Results
paragraph, move the parenthetical K=20 and c=5000 interval numbers into the Fig 3
caption, where they are already plotted. Specifically trim from the Results
paragraph the two parentheticals `while $K{=}20$ is not (mean $-0.252$, interval
$[-0.920, +0.416]$)` → `while $K{=}20$ is not` and `by checkpoint 5{,}000 it
dissolves (mean $-0.795$, interval $[-2.513, +0.923]$)` → `by checkpoint 5{,}000 it
dissolves`. Add those four numbers to the Fig 3 caption (supplementary, free).

**Appendix A insertion** (optional but recommended — full multiplicity table, free):
list all three CI-excluding-zero K=32 cells with their Δ and CI, and state the
`holds()` condition explicitly and which single cell satisfies it.

---

### FIX-3: Reframe the variance-ratio cutoff as a pre-registered heuristic, not a calibrated F-test
**Severity:** SERIOUS (resolves A3)
**File(s):** `sections/05_replication.tex`; optional one-clause note in Appendix A (`main.tex`).
**Location:** §5 "The gate fired" paragraph.

**Why:** A3. The 4.0 cutoff has no derivation anywhere in the design registry
(first appears as a bare `variance_ratio > 4`). An F(2,8) test — the correct
reference for old n=3 (df 2) vs new n=9 (df 8) — gives one-sided p≈0.0497 and a
one-sided 5% critical value ≈4.459, essentially equal to the observed 4.47 (a
two-sided 5% test would need ≈6.06). So 4.0 is slightly *more permissive* than a
real one-sided 5% test, and the "twelve percent past the cutoff" framing reads as
more calibrated than it is. This does not change the substantive result (the naive
pool spans zero regardless), and the fix strengthens the section's existing
"small-n variance imprecision" reading.

**Before (§5):**
```
\textbf{The gate fired.} The archived cohort's between-seed standard
deviation (1.154) is 2.1 times the new cohort's (0.546): variance ratio
4.47, twelve percent past the cutoff, while the means agree comfortably
(shift 0.364 against a 1.382 threshold). % evidence: C7-replication
```
**After (§5):**
```
\textbf{The gate fired.} The archived cohort's between-seed standard
deviation (1.154) is 2.1 times the new cohort's (0.546): variance ratio
4.47, past the pre-registered heuristic cutoff of 4.0---a routing
threshold, not a calibrated significance test (a one-sided 5-percent
$F(2,8)$ test would itself require $\approx 4.46$, essentially the observed
value). The means agree comfortably (shift 0.364 against a 1.382
threshold). % evidence: C7-replication
```

**Optional Appendix A note** (free): in the "Replication integrity gates (wave 6)"
paragraph, change "a variance ratio below a pinned 4.0" to "a variance ratio below
a pre-registered heuristic 4.0 (not a calibrated $F$-test; the one-sided 5-percent
$F(2,8)$ value is $\approx 4.46$)."

**Space:** roughly net-neutral.

---

### FIX-4: Correct the Okpekpe & Orvieto (2025) characterization
**Severity:** SERIOUS (resolves A4)
**File(s):** `sections/06_related.tex`.
**Location:** §6, the Okpekpe–Orvieto sentence.

**Why:** A4. arXiv:2508.19029's lead contribution is that *learning-rate choice*
is a methodological confound in recurrent-model recall comparisons (optimization
instability, not an architecture-level mechanism). Calling it "a positive causal
account of recall differences" recasts a tuning-confound finding as a mechanism
claim — a materially different characterization that a reviewer who knows the paper
will flag.

**Before (§6):**
```
\citet{okpekpe2025revisiting} give a positive causal account of recall
differences in modern recurrent models; this paper makes no positive
mechanism claim and instead bounds one named mechanism three ways.
```
**After (§6):**
```
\citet{okpekpe2025revisiting} show that optimizer choice, not architecture
alone, drives much of the reported Transformer--recurrent recall gap; this
paper bounds a different axis (write-geometry validity) and makes no
positive mechanism claim.
```

**Space:** net-neutral (slightly shorter).

---

### FIX-5: Add the DeltaNet/SSM state-tracking literature and engage its competing account of the null
**Severity:** SERIOUS (resolves A5)
**File(s):** `sections/06_related.tex`; `sections/07_discussion.tex`; `refs.bib`.
**Location:** §6 (add citations); §7 "Not bounded" (engage the competing account).

**Why:** A5. The bind/query task is a permutation-composition tracking problem, and
the most on-point prior work — Grazzi et al. (ICLR 2025, arXiv:2411.12537), which
proves DeltaNet with positive-eigenvalue state transitions cannot compose certain
groups (parity), a ceiling lifted only by negative-eigenvalue/Householder-product
transitions — is absent. It offers a *competing theoretical explanation* for the
null (an expressivity ceiling of the specific frozen-bias DeltaNet variant under
test) that directly rivals §7's "wrong observable" reading, so it needs engagement,
not just a citation. Merrill et al. (ICML 2024, arXiv:2404.08819) gives the
analogous TC0 ceiling for diagonal SSMs.

**§6 — add (place after the Okpekpe–Orvieto sentence):**
```
\citet{grazzi2025unlocking} prove that DeltaNet-family linear RNNs with
positive-eigenvalue state transitions cannot compose certain groups (e.g.\
parity), a ceiling lifted only by negative-eigenvalue or Householder-product
transitions; \citet{merrill2024illusion} give the analogous $\mathrm{TC}^0$
ceiling for diagonal state-space models on permutation composition.
```

**§7 "Not bounded" — add one clause engaging the competing account:**
Before:
```
\textbf{Not bounded.} The hypothesis survives at effect sizes below the
power floor and through mechanisms the readout cannot see, most plausibly
cross-layer hand-off, where one layer resolves the first hop and a later
layer the next; the readout tests single-layer state self-iteration, a
strategy a multi-layer model never trained on the grammar has no
particular reason to adopt.
```
After:
```
\textbf{Not bounded.} The hypothesis survives at effect sizes below the
power floor and through mechanisms the readout cannot see, most plausibly
cross-layer hand-off, where one layer resolves the first hop and a later
layer the next; the readout tests single-layer state self-iteration, a
strategy a multi-layer model never trained on the grammar has no
particular reason to adopt. A competing account is expressivity: the
frozen-bias arms use positive-eigenvalue state transitions, which
\citet{grazzi2025unlocking} prove cannot compose certain permutation
groups, so the null may partly reflect a representational ceiling of this
specific variant rather than only a wrong observable.
```

**refs.bib — add** (verified live 2026-07-10 by both the attack and defense
agents; entries in the file's arXiv-`@article` convention — upgrade to
`@inproceedings` if proceedings metadata is on hand):
```
@article{grazzi2025unlocking,
  title   = {Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues},
  author  = {Grazzi, Riccardo and Siems, Julien and Franke, J{\"o}rg K. H. and Zela, Arber and Hutter, Frank and Pontil, Massimiliano},
  journal = {arXiv preprint arXiv:2411.12537},
  year    = {2025}
}

@article{merrill2024illusion,
  title   = {The Illusion of State in State-Space Models},
  author  = {Merrill, William and Petty, Jackson and Sabharwal, Ashish},
  journal = {arXiv preprint arXiv:2404.08819},
  year    = {2024}
}
```

**Space:** §6 grows ~2 lines and §7 grows ~2 lines. Recover space by tightening
§6's final sentence (the pre-registration/negative-results tradition sentence can
drop "outcome partitions fixed before unblinding, refusals reported as results" —
that phrasing is already made in §7's Lessons paragraph) and by tightening §7's
Lessons paragraph. Verify author spellings/initials against the arXiv abstract page
before compiling.

---

### FIX-6: Add the real-kernel positive-control limitation (mandatory) and, if any GPU is available, build the control (strongly recommended)
**Severity:** SERIOUS (resolves the surviving reduced form of A6)
**File(s):** `sections/07_discussion.tex` (tier 1, mandatory); new experiment (tier 2, strongly recommended pre-submission).
**Location:** §7 (a limitation sentence in "Not bounded" or as a short standalone).

**Why:** A6. The entire Bound-1 "instrument verdict, not refutation" reading rests
on the readout *implementation* being correct on the production multi-layer
`fla`/DeltaNet extraction path. The only positive control in the whole program is a
CPU-stub unit test on a hand-built 4×4 diagonal matrix (`reasoning_link_stage_minus1.py`
lines 76–96) — no model, no forward pass, no kernel, no multi-layer extraction.
Every attempt to make the readout fire, including Wave 4's model trained on the task
(0/30), failed. This project's own CLAUDE.md flags exactly this failure mode
("CPU-stub self-test suites test logic only; real-kernel coverage needs a separate
narrow smoke of the PRODUCTION path"), and the broader program has a prior
state-transpose-convention scare in a sibling campaign. The 366 zeros are therefore
equally consistent with genuine construct-invalidity (the paper's story) and with a
systematic extraction/convention bug specific to this setup — a distinction the
paper draws but does not license. The paper's *epistemic output* is bug-robust (it
declines to refute anything), which is why this is SERIOUS not CRITICAL, but its
*diagnostic* claim ("the construct is wrong") is under-supported versus "the code is
wrong," and that must be scoped.

**Tier 1 (mandatory, submission-viable) — add to §7:**
```
One caveat scopes the instrument verdict: the readout's recovery math is
validated only at unit level (hand-built matrices), not end-to-end against
a real forward pass with a known target, so a systematic extraction or
state-convention error in the multi-layer kernel path cannot be fully
excluded. The failure's uniformity across four structurally different
instruments and four scales is consistent with a genuine construct-validity
gap but does not prove one over such an error; a real-kernel positive
control is the priority for the next wave.
```

**Tier 2 (strongly recommended before submission — not a gauntlet blocker, but
cheap and high-value):** construct one synthetic sequence whose $h$-hop answer is
knowable in closed form (keys/values built so the correct $v_{\text{target}}$ is
analytic), run it through an *actual* trained checkpoint's real forward pass, and
confirm the production extraction path recovers it at cosine ≥ 0.9 before any
genuine-checkpoint zero is treated as diagnostic. **Confirming result:** the
control recovers the known target (readout implementation validated → the 366 zeros
are a construct verdict, and Tier-1's hedge can be softened to a validated
statement). **Falsifying result:** the control fails to recover a known-present
target → a real extraction/convention bug exists, and Bound 1's interpretation must
be revised before submission. Given the program cost only 9.5 GPU-h total and this
is a single forward pass (minutes on the available hardware), the cost/benefit
strongly favors building it now: it converts the paper's single largest residual on
its headline bound into a demonstrated result.

**Space:** ~3 lines in §7; recover via the §7 Lessons-paragraph tightening noted in
FIX-5.

---

### FIX-7: Disclose that the familiarization fall is measured from checkpoint 250
**Severity:** MINOR (resolves A7)
**File(s):** `sections/03_geometric_null.tex` (main text); `main.tex` (Fig 2 caption, supplementary).
**Location:** §3 Wave-4 sentence; Fig 2 caption.

**Why:** A7. The "21.8 to 46.4 percent" fall matches the archives only when the
baseline is checkpoint 250 (the first post-warmup, gate-matched checkpoint); from
step 1 the same six cells fall 70.7–81.2%. The caption's "over 5,000
familiarization steps" reads as start-to-finish. The baseline choice is defensible
(gate-matched) and the omitted number is *larger* (no incentive to hide), so this is
a transparency fix only.

**Before (§3):**
```
while the vocabulary-space query loss falls 21.8 to 46.4 percent (mean
35.9 percent) in 6 of 6 cells % evidence: C4-dissoc
```
**After (§3):**
```
while the vocabulary-space query loss falls 21.8 to 46.4 percent (mean
35.9 percent) in 6 of 6 cells, measured from checkpoint 250, the first
post-warmup reading, to checkpoint 5{,}000 % evidence: C4-dissoc
```

**Fig 2 caption** (`main.tex`, supplementary — also update for consistency):
change "falls by 21.8 to 46.4 percent over 5{,}000 familiarization steps" to
"falls by 21.8 to 46.4 percent between checkpoint 250 (the first post-warmup
reading) and checkpoint 5{,}000."

**Space:** ~half a line in §3; the caption is free.

---

### FIX-8: Fix the Merity et al. bibliography year
**Severity:** MINOR (resolves A8)
**File(s):** `refs.bib`.
**Location:** `merity2017pointer` entry, line 54.

**Why:** A8. Pointer Sentinel Mixture Models is ICLR 2017; 2016 is only the arXiv
v1 date, and the citation key itself encodes 2017, so the compiled "(Merity et al.,
2016)" is inconsistent with the key and with standard convention.

**Before:**
```
  year    = {2016}
```
**After:**
```
  year    = {2017}
```

**Space:** none (bib).

---

## 3. Verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 | CRITICAL | CONCEDE+FIX | DEFENSE INSUFFICIENT (as written) — resolved below CRITICAL by FIX-1 | FIX-1 |
| A2 | CRITICAL | PARTIAL | PARTIAL — attack survives in reduced form as a SERIOUS disclosure gap | FIX-2 |
| A3 | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-3 |
| A4 | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-4 |
| A5 | SERIOUS | CONCEDE+FIX | DEFENSE INSUFFICIENT — missing cite + un-engaged competing account | FIX-5 |
| A6 | SERIOUS | CONCEDE+FIX | PARTIAL — attack survives (positive-control gap); mandatory limitation, control recommended | FIX-6 |
| A7 | MINOR | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-7 |
| A8 | MINOR | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-8 |

**Disposition counts:** DEFENSE VALID 0 · DEFENSE VALID BUT EDIT 4 (A3, A4, A7, A8)
· DEFENSE INSUFFICIENT 2 (A1, A5) · PARTIAL 2 (A2, A6).

**Severity adjudications where I departed from the reports:**
- **A2 down-rated CRITICAL → SERIOUS.** The `holds()` vs. `det32` distinction is
  correct and the transient is genuinely distinguished (unique differential-
  condition firing + independent secondary-readout corroboration). The attack's
  "indistinguishable from noise" framing does not survive; a disclosure gap does.
  It does **not** hold a CRITICAL open.
- **A6 held at SERIOUS (declined the defense's escalation to CRITICAL).** The
  attack does not falsify any claim the paper actually makes — Bound 1 is hedged to
  "instrument verdict / probe-invalid / not a refutation," and a bug only reinforces
  "not a refutation." What is under-supported is the *diagnostic* attribution
  (construct vs. implementation), which a limitation sentence scopes. I require the
  fix be substantive (Tier 1 mandatory, Tier 2 strongly recommended), not a
  throwaway sentence — but it is not an open CRITICAL.
- **A4 noted near the SERIOUS/MINOR boundary** (single mischaracterized clause, no
  effect on the paper's own results), but the fix is trivial so the classification
  is moot; kept at SERIOUS.

---

## 4. Residual risk after all fixes

Ordered by how much a skeptical MOSS/COLM reviewer would weight it.

1. **Positive-control gap on the headline bound (from A6) — the top residual.**
   Even with FIX-6 Tier 1, the "instrument verdict" reading of Bound 1 rests on the
   *internal consistency* of the failure pattern, not a positive control. A hostile
   reviewer of the paper's most load-bearing result can still say "your 366 zeros
   might be an extraction bug." **Workshop-survivable** with the prominent
   limitation; **conference-blocking** if the paper's central claim were the
   construct verdict. Strong recommendation: build the FIX-6 Tier-2 control before
   submission — it is minutes of GPU and it retires this residual outright.

2. **Post-hoc granularity underlying Bounds 2–3 (from A1).** After FIX-1 the
   provenance is disclosed and the registered corpus-level UNRESOLVED is the
   headline, but Bounds 2 and 3 are still *organized around* a cell selected at a
   granularity chosen after the registered null was seen. Disclosure defuses the
   integrity objection (undisclosed post-hoc → disclosed post-hoc in a null paper
   that explicitly forbids reading the transient as a real effect), but a reviewer
   may still find the transient thread thin. **Workshop-survivable**, especially
   because the honest framing strengthens the null; **conference-riskier.**
   Mitigation already in the fix: demote the transient to a disclosed sub-finding
   and let the registered UNRESOLVED carry Bound 2.

3. **Multiplicity of the transient (from A2).** After FIX-2 the paper reports 3/40
   ≈ chance and distinguishes the transient by the differential condition + the
   secondary readout. A reviewer may still note that one cell clearing a compound
   condition among 40 uncorrected tests is weak standalone evidence — but the paper
   no longer overclaims it, and it is a null paper. **Workshop-survivable.**

4. **Expressivity-ceiling competing account (from A5).** After FIX-5 the paper
   acknowledges Grazzi et al.'s ceiling as a rival to its "wrong observable"
   reading. A reviewer may push "so is your null just the known ceiling of
   positive-eigenvalue DeltaNet?" — but the paper now raises this itself and does
   not claim the capability is present, which is the honest position.
   **Workshop-survivable.**

5. **Page pressure (cross-cutting).** The paper is at the 4.0-page hard cap and the
   fixes add net main-text content in §4, §6, and §7. The offloads to Appendix A
   (free) plus the named per-fix cuts should keep it at 4.0, but the writer must
   verify the compiled page count after editing and re-check the anonymization grep
   and MOSS/COLM template compliance. **Submission-blocking if the cap is exceeded;**
   fully avoidable.

No residual is conference-blocking *for this venue* (a non-archival small-scale
workshop late window) once the fixes land, with the caveat that residual #1 is worth
retiring by building the cheap control.

---

## 5. Re-run instruction (which claims re-enter attack/defense/rebuttal)

The fixes rescope the paper's one non-null finding, so the **transient-provenance
thread** must re-run through attack/defense/rebuttal after editing. Targeted, not a
full restart — the raw numbers C0–C9 were all independently recomputed and matched
in both prior rounds, so this re-run is about **framing/provenance/disclosure**, not
re-measurement. Re-attack these specific surfaces:

- **Abstract** (`00_abstract.tex`) — the "sole reading meeting the pre-registered
  differential condition" wording (FIX-2) and the implicit provenance (FIX-1).
- **§1 contributions** (`01_introduction.tex`) — the power-bound and replication-
  bound bullets that name the transient (FIX-2 precision edit).
- **Table 1** (`03_geometric_null.tex`) — the wave-5 row and its new footnote
  (FIX-1).
- **§4** (`04_behavioral_contrast.tex`) — the Results provenance sentence (FIX-1)
  and the Sign-discipline/multiplicity paragraph (FIX-2).
- **§5** (`05_replication.tex`) — the gate-firing paragraph (FIX-3), since it
  characterizes the same transient's replication.
- **§6 / §7** (`06_related.tex`, `07_discussion.tex`) — the new competing-account
  engagement (FIX-5) and the positive-control limitation (FIX-6).
- **Appendix A** (`main.tex`) — the new provenance and multiplicity paragraphs
  (FIX-1, FIX-2) and the F-test note (FIX-3).

Bound 1's raw 366/366 zeros and the gate-failure counts do **not** need re-measuring
(verified clean by both prior rounds); only A6's positive-control caveat attaches to
them, and FIX-6 scopes it. FIX-4, FIX-7, FIX-8 are self-contained corrections that
do not require a claims re-run.

---

## 6. New citations

**MUST-CITE (submission-blocking omission for this paper's topic):**
- **Grazzi, Siems, Franke, Zela, Hutter, Pontil, "Unlocking State-Tracking in
  Linear RNNs Through Negative Eigenvalues," ICLR 2025, arXiv:2411.12537.** The
  single most on-point prior work: DeltaNet-specific, about exactly the
  permutation-composition capability the bind/query task instantiates, and it
  supplies a *competing theoretical explanation* for the null (a positive-
  eigenvalue expressivity ceiling of the specific frozen-bias variant). Omitting the
  one paper that theoretically predicts your null on your exact model family and
  task is the kind of gap a knowledgeable reviewer will catch. Added and engaged via
  FIX-5.

**SHOULD-CITE (strengthens the paper; not blocking):**
- **Merrill, Petty, Sabharwal, "The Illusion of State in State-Space Models," ICML
  2024, arXiv:2404.08819.** The general TC0 state-tracking ceiling for diagonal
  SSMs; pairs naturally with Grazzi as the general-class complement. Added via
  FIX-5. (Near-MUST as Grazzi's companion, but Grazzi is the essential one.)
- **Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers are
  Better than State Space Models at Copying," ICML 2024, arXiv:2402.01032.** A
  fixed-latent-state capacity account for why GSSMs underperform at copying/recall;
  a reasonable third addition alongside Based/Zoology in §6's first sentence if
  space permits. Lower priority than the two state-tracking papers; **omit if the
  page cap bites.**

All three were independently verified live (2026-07-10) by both the attack and
defense agents. Verify author initials/spellings against the arXiv abstract pages
before compiling; bibtex for the two added entries is in FIX-5 (Jelassi is not
provided — add only if used).

---

*Adjudication note.* I trusted neither prior report on severity: I re-derived the
key numeric facts from the reports' own agreed evidence, adopted the defense's
`holds()`/`det32` distinction where it was load-bearing (A2), and independently
rejected the defense's arm-asymmetry sub-argument (A2's checkpoint-clustering) as
unsound because all four c=1000 cells are negative — exactly what a shared
control-dip predicts — so the fix wording must not lean on it. No CRITICAL remains
open after FIX-1 and FIX-2 are applied and the transient-provenance thread is
re-run.
