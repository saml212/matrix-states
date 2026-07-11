# Rebuttal report — measurement-ws, gauntlet round 1 (adjudicator / fix list)

Adjudicator: fresh-context, trusts neither report, 2026-07-10. Every
load-bearing attack re-verified **directly against the raw artifacts this
session** (md5s re-checked, numbers recomputed) — not taken on either
report's word. The seven catches that carry the paper's credibility were
each reproduced from the named JSON/log: A1 (25 not 30), A2 (S3 non-zero),
A3 (the two-axis centering confound), A4 (S5 fails its bar without the
outlier), A7 (zero probing citations in refs.bib), A13 (the threshold
sweep), A14 (S3 cleared its primary bar; byte-identical reason strings).
The defense reproduces every major catch and — importantly — **correctly
overturns the attacker on A13**; my recompute confirms the defense, not the
attacker, on the threshold sweep.

---

## 1. Summary for the edit agent

**No CRITICAL remains open after this fix list is applied.** The one
attacker-labelled CRITICAL (A1, the 30-vs-25 count) is a factually-correct
catch that the paper's own provenance discipline should have fired on; it is
resolved by a one-number edit **plus a mandatory verify-vs-raws sweep over
every count-type claim** (FIX-1). I additionally **re-rate A7 from SERIOUS
to CRITICAL (acceptance-blocking)** — a probing-pitfalls case study that
cites none of the probing-methodology canon is a rejection-grade omission
for the target measurement/eval venue, independent of the numbers being
right (which they are). A7 is fully resolved by adding four MUST-CITE
references and one distinguishing paragraph (FIX-2). So there are **two
CRITICAL-severity items, both closed by fixes; none stays open.**

**Disposition counts:** 13 ATTACK-UPHELD-and-FIX, 1 PARTIAL (A8, survives
in reduced form), 1 split (A16: (a) upheld+fix, (b) DEFENSE VALID / no
change), plus A13 upheld **only in the defense's corrected form** (the
attacker's own proposed remedy sentence is false and must not be pasted).

**The four structural fixes that carry the weight:**
1. **FIX-1 (A1, CRITICAL):** `30 → 25` in §5 and brief row X2, plus a
   count-claim verification sweep. This is the paper's ironic core wound —
   a number the "we re-verified everything" paper did not re-verify.
2. **FIX-2 (A7, re-rated CRITICAL):** add the probing-methodology
   literature (Hewitt & Liang, Elazar et al., Adebayo et al., Schaeffer et
   al.) with the serial/pre-registered/teeth distinction stated.
3. **FIX-3 (A2, SERIOUS):** scope "zeros at every rank" in **four**
   locations (§1, §6, appendix B2, catalogue row 5) — S3 reads 0.17/0.08.
4. **FIX-4 (A9, SERIOUS):** pull `tab:catalogue` into the body so the
   catalogue paper has a catalogue; this is where the A7-vs-A9 page-budget
   collision is resolved (see §4 and FIX-4 below).

**Page-budget collision (A7 vs A9), resolved:** A7 lands **unconditionally**
(it is CRITICAL); A9's catalogue table lands **in-body in compact form**,
funded by three named cuts the fixes themselves enable — (i) replacing §1's
now-redundant inline six-failure enumeration with a one-clause pointer to
Table 1, (ii) compressing §6's three-lens prose to two sentences that defer
to the appendix (the three lenses now appear as catalogue rows), and (iii)
tightening §8's variance/checklist sentences to absorb A7's citation block.
**Hard fallback:** if the Jul-11 venue confirms a strict 4pp and the style
pass still overflows after those cuts, A9's table reverts to the appendix
(its current safe location) and brief row X2 is corrected to "Fig. 1" —
**A7 is never sacrificed.** (NeurIPS workshops commonly allow 5pp; MOSS is a
hard 4pp — the refresh decides whether the fallback is needed.)

**One place the writer must NOT follow the attacker:** A13. The attacker's
proposed clause "the 13-cell count is identical for any cut in [0.3, 0.7]"
is **false** — I recomputed the raw: the counts are **16 / 14 / 13 / 12 /
11** at cuts 0.3 / 0.4 / 0.5 / 0.6 / 0.7. Use the defense's corrected,
verified framing (disagreement > 0.3 ⟺ converged, 16/16) in FIX-12.

---

## 2. Ordered fix list (CRITICAL → SERIOUS → MINOR)

### FIX-1: Correct the baseline-cell count 30 → 25 and run the count-claim verification sweep
**Severity:** CRITICAL
**File(s):** `sections/05_case_gauge.tex`; `brief.md` (row X2); the count sweep touches §3 and appendix
**Location:** §5, the "signature" paragraph (the sentence ending "...the lenses agree exactly")
**Before:** "All 13 cells with a lens disagreement of at least $0.5$ are converged cells; on all 30 baseline cells the lenses agree exactly."
**After:** "All 13 cells with a lens disagreement of at least $0.5$ are converged cells; on all 25 Arm-2 baseline cells (5 groups $\times$ 5 seeds, 50 checkpoint values) the lenses agree exactly, with zero divergence."
Also fix `brief.md` row X2: "all 30 Arm-2 baseline cells: crosscheck == primary exactly" → "all 25 Arm-2 baseline cells (5 groups × 5 seeds, 50 checkpoint values): crosscheck == primary exactly." Optionally note the design-doc §2.32 prose is internally inconsistent (says "5 groups × 5 seeds … 30"; 5×5=25) so future copies do not re-inherit the error.
**Verification sweep (mandatory tail, submission blocker):** re-derive every bare *count* in the paper from its raw before submission. I re-confirmed this session: arm2 = **25** cells (Counter over `flat_per_cell_table`), agreement **25/25** at far64 and **25/25** at ceiling; the 62 rows = 37 arm3 + 25 arm2; **13** cells with disagreement ≥ 0.5. Those reproduce. The writer must additionally re-confirm the other counts the draft asserts (the "11 quarantined/verified cells" in §3/§7, "4,608 = 4,313 + 295", "107/107", "36 events", "37 of 39") against their named raws and correct any that do not reproduce.
**Why:** A1. The number is contradicted by the artifact the paper itself cites (`experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/crosscheck_lens_verdict_output.json`, md5 f26a769d…, verified matching). The qualitative claim (lenses agree on every baseline cell) is true; only the printed count is wrong. Because this paper's entire credibility is "every number re-verified against raws," an unverified count in the flagship case study is rejection-grade for *this* paper — hence CRITICAL despite the one-character fix. The verification sweep is the load-bearing part: A2, A11, and A14 show counts are where the draft's own verification was weakest.

---

### FIX-2: Add the probing-methodology literature and the distinguishing paragraph
**Severity:** CRITICAL (re-rated from the attack's SERIOUS — see §5 adjudication)
**File(s):** `sections/08_related_limits.tex` (new citation block); `sections/04_case_wronglayer.tex` (two inline cites); `refs.bib` (five new entries)
**Location:** §8 Related work, appended to the existing related-work paragraph; §4 at the shuffled-tap control and at the state-zeroing
**Before:** §8 related work engages many-analysts / underspecification / variance accounting / reporting standards but no probing-methodology work; refs.bib has zero probing citations (verified: the nine non-companion entries are schlag, nichani, silberzahn, damour, henderson, bouthillier, kapoor, sculley, lipton).
**After:** add ~3–4 sentences to §8, e.g.: "The probing-methodology literature supplies the instrument-hygiene checks these incidents rediscover: control tasks and selectivity~\citep{hewitt2019control}, the probe-versus-behavior dissociation surveyed by~\citet{belinkov2022probing}, amnesic (causal-removal) probing~\citep{elazar2021amnesic}, and randomization-based sanity checks for interpretability instruments~\citep{adebayo2018sanity}. \citet{schaeffer2023emergent} run the inverse of our thesis — an apparent model \emph{property} (emergence) shown to be a metric artifact — from the same instrument-is-the-suspect stance. Each of these diagnoses a single instrument class post hoc; this paper contributes a serial, pre-registered, cross-instrument adjudication record with falsifier teeth." Add inline `\citep{hewitt2019control}` at the §4 shuffled-tap control ("cleared its shuffled-tap control") and `\citep{elazar2021amnesic}` at the §4 state-zeroing ("A state-zeroing experiment located the storage"). Add the five entries to refs.bib (details in §5 MUST-CITE list); **re-verify each arXiv id against the arXiv export API before insertion** (the refs.bib header asserts this bar for every arXiv-bearing entry — the new ones must meet it).
**Why:** A7. Case II's machinery is item-for-item the published probe-hygiene playbook; omitting it reads as not-engaging-the-field to exactly the measurement/eval reviewer this paper targets — a rejection independent of the numbers. Costs citations + prose, not a claim; the distinction (serial, pre-registered, cross-instrument, teeth) is real and defensible.

---

### FIX-3: Scope "zeros at every rank" — S3's capped arms are non-zero
**Severity:** SERIOUS
**File(s):** `sections/01_intro.tex`, `sections/06_three_lenses.tex`, `sections/09_appendix.tex` (B2 prose and catalogue row 5)
**Location:** four sites — the intro's six-failure enumeration; §6 B2 clause; appendix B2 paragraph; the `tab:catalogue` row 5.
**Before / After (per site):**
- **§1 intro** — Before: "a causal rank test reading zero at every rank"; After: "a causal rank test reading near-chance at every rank".
- **§6** — Before: "a causal rank experiment returned zeros at every rank while the capped arms sat at their exact theoretical ceiling"; After: "a causal rank experiment returned near-chance recovery at every rank (zero in four of five groups; $0.17$ and $0.08$ in the fifth) while the capped arms sat at their exact theoretical ceiling".
- **Appendix B2** — Before: "Every capped arm returned zero, at every group, on both lenses."; After: "Every capped arm returned near-chance recovery on both lenses — zero in four of the five groups, and $0.17$/$0.08$ in the fifth (S3), still far below the recruitment ceiling."
- **`tab:catalogue` row 5** — Before: "Causal rank test returns zeros at every rank"; After: "Causal rank test returns near-chance recovery at every rank".
**Why:** A2. `harvest_summary.json` (md5 7dce77dc…, verified) `m3.S3.arms` reads `k_dmin: rec/xrec = 0.1667` and `k_dmin_plus_1: rec/xrec = 0.075` — non-zero on **both** lenses; S4/A5/S5/A6 are all-zero. The universal is falsifiable by anyone who opens the file, in a paper about numeric hygiene. The incident's substance (all five groups `confirm=False`; 37/39 cells at the √(k/d) ceiling) is untouched — only the false universal moves.

---

### FIX-4: Pull the catalogue table into the body (and resolve the page-budget collision)
**Severity:** SERIOUS
**File(s):** `sections/09_appendix.tex` (remove `tab:catalogue`), body placement (end of `sections/01_intro.tex` or top of `sections/02_discipline.tex`); `sections/01_intro.tex`, `sections/06_three_lenses.tex`, `sections/08_related_limits.tex` (funding cuts); `brief.md` (row X2)
**Location:** move the `tab:catalogue` float (currently appendix, `sections/09_appendix.tex` lines ~64–116) into the body as **Table 1**, kept `\scriptsize`. The §1 `(Table~\ref{tab:catalogue})` reference becomes a body reference. Keep `tab:walk`, `tab:taps`, `tab:remetric`, `tab:teeth` and `fig:taploc` in the appendix.
**Funding (apply in order until the body fits the confirmed limit):**
1. `sections/01_intro.tex`: replace the inline six-failure enumeration ("a capacity collapse (§3); three consecutive failed evaluation rounds (§4); …; a recurrence contradicted by a reference kernel (§6)") with a one-clause pointer to Table 1's first column (it is now redundant with the table). ~0.10pp.
2. `sections/06_three_lenses.tex`: compress the three-lens prose to two sentences that defer detail to Appendix~\ref{app:threelenses} and cite the catalogue rows (the three lenses now appear as Table 1 rows 4–6). ~0.15–0.20pp. Keep the corrected A2/FIX-3 scoping in whatever §6 text remains.
3. `sections/08_related_limits.tex`: tighten the variance-accounting and checklist sentences to absorb FIX-2's citation block within §8's existing allocation. ~0.10pp.
**Also:** fix `brief.md` row X2's "**Fig. 1** + Appendix table" → "**Fig. 1**" (no 62-cell appendix table exists in the draft — the map's figure/table column is stale).
**Hard fallback:** if the Jul-11 venue confirms exactly 4pp and the style pass overflows after cuts 1–3, leave `tab:catalogue` in the appendix (revert this fix's move) and keep only the brief-row X2 correction; **do not cut FIX-2.**
**Why:** A9. Contribution 1 is "the catalogue," yet a body-only read (which NeurIPS-pattern reviewers may do) currently contains one figure and zero tables — the catalogue paper has no catalogue in the body.

---

### FIX-5: Kill the false causal "because" in the S5 boundary claim
**Severity:** SERIOUS
**File(s):** `sections/07_not_flipped.tex`; `sections/09_appendix.tex` (`tab:remetric` caption)
**Location:** §7, the second decisive-group sentence; and the `tab:remetric` caption
**Before (§7):** "and the other decisive group sits below its pre-registered bar ($0.483$ against $0.735$) because one seed trains to low loss yet generalizes at $0.00$ under both lenses while its siblings individually clear $80\%$ and $65\%$ of their own ceilings against a baseline pinned at exactly $0.0$."
**After (§7):** "and the other decisive group sits below its pre-registered bar ($0.483$ against $0.735$) lens-independently: one seed is a total generalization failure (recovery $0.00$ under both lenses at low training loss), and even its two siblings reach only $80\%$ and $65\%$ of their own ceilings — both under the $90\%$ criterion — so the group misses its bar with or without the outlier, against a baseline pinned at exactly $0.0$."
**Before (`tab:remetric` caption):** "S5 is held under its bar by one non-generalizing seed."
**After (`tab:remetric` caption):** "S5 misses its bar lens-independently: one seed generalizes at $0.00$, and its two siblings reach only $80\%$/$65\%$ of their own ceilings (both below the $90\%$ bar)."
**Why:** A4. Verified this session from `crosscheck_lens_verdict_output.json`: S5 n_h=4 triad far64 = 0.80 / 0.00 / 0.65, own ceilings 1.0 / 0.45 / 1.0, group far 0.4833, bar 0.735. Drop seed 1 → mean of the two siblings = **0.725 < 0.735**. There is no counterfactual under which S5 clears its bar, so "because one seed" is arithmetically indefensible — and it sits in the exact section whose job is to prove no spin. The boundary claim *survives* (S5 is a robust, lens-independent model failure); only the causal attribution goes.

---

### FIX-6: Scope the opening prevalence claim, state the incident-selection rule, name the flattering-defect blind spot
**Severity:** SERIOUS
**File(s):** `sections/01_intro.tex`; `sections/08_related_limits.tex` (Limitations)
**Location:** §1 opening sentence + one added sentence; §8 Limitations paragraph
**Before (§1 opening):** "A measurement-heavy empirical program breaks its instruments more often than its models. Within five days, one research program on fast-weight sequence models produced six apparent headline model failures: …"
**After (§1 opening):** "Within one research program's five-day window on fast-weight sequence models, six apparent headline model failures traced to broken instruments and two to broken models; this paper is that catalogue and the fixed procedure that made each call. The six instrument cases: …" (keep the §-anchored enumeration, or replace it per FIX-4 cut 1).
**Add (§1, one sentence, after the contributions or the enumeration):** "We count an incident as a pre-registered endpoint or eval round read as a model verdict; instrument defects caught outside such a reading (e.g. the auxiliary-classifier null of \S\ref{sec:wronglayer}) are noted but not counted."
**Add (§8 Limitations, after the survivor-bias sentence):** "It is also biased by trigger: the discipline fires on apparent model failure, so instrument defects that \emph{flatter} the model are under-represented, and five of the six fixes moved results in the program-favorable direction."
**Why:** A5. The opening is the exact prevalence claim §8 disclaims ("six incidents support no prevalence claim"); scoping it to the program's window makes it a factual report, not a generalization. The unstated counting rule lets W5 and §2.29-class defects be excluded silently; printing the rule makes the exclusion by-rule, not by-omission. The current Limitations names only the missed-by-everything bias, not the selection asymmetry. (I verified the "five of six favorable" count from the catalogue: Cases I, II, B1, B2, B3 moved program-favorable; Case III stayed FALSIFY — the exception. The writer should confirm this count holds before printing it.)

---

### FIX-7: Rescope "one and the same discipline" — two rule-clauses were codified mid-catalogue
**Severity:** SERIOUS
**File(s):** `sections/00_abstract.tex`; `sections/02_discipline.tex`
**Location:** abstract, the "Each incident was caught by one and the same discipline" sentence; §2 intro paragraph
**Before (abstract):** "Each incident was caught by one and the same discipline: instrument-health adjudication before any model verdict, …"
**After (abstract):** "The same discipline adjudicated every incident — assembled across the catalogue and in full force before the final, decisive one: instrument-health adjudication before any model verdict, …"
**Add (§2 intro, after "…a lens-failure detector with a measured record."):** "Two clauses were codified mid-catalogue by the incidents that exposed them — Rule 3's positive-control clause after Case II, Rule 2's graded crosscheck after the covariance incident — and the full five-rule discipline was in force before the decisive Case III."
**Why:** A6. The draft's own text falsifies "fixed": §4 says Rule 3's positive-control clause exists "hence" Case II, and Rule 2's graded crosscheck was built as the B1 fix. The honest framing ("the discipline is the distillate of the catalogue") is a better story and leaves the teeth and the pre-registered switching rule untouched — Case III *was* adjudicated by all five rules.

---

### FIX-8: De-confound the centering number in the uncentered-covariance incident
**Severity:** SERIOUS (borderline MINOR — see §5)
**File(s):** `sections/06_three_lenses.tex`; `sections/09_appendix.tex` (B1 paragraph)
**Location:** §6 B1 parenthetical; appendix B1 (the "0.084 … versus 0.9996 … once the covariance was centered" sentence)
**Before (§6):** "(mean cosine $0.084$ versus $0.9996$ centered)"
**After (§6):** "(mean cosine $0.084$ under the old lens versus $0.9996$ under the fixed lens; App.~\ref{app:threelenses} isolates centering alone, $0.705\!\to\!0.9996$ under a matched full-intertwiner metric)"
**Before (appendix B1):** "an oracle injection scored mean cosine $0.084$ and recovery $0.05$ under the uncentered lens versus $0.9996$ and $1.00$ once the covariance was centered; the same artifact records the scale-only primary reading $0.046$ on data whose true gauge is non-identity while the full-intertwiner crosscheck reads $0.9996$, the pre-echo of \S\ref{sec:gauge}."
**After (appendix B1):** "the headline $0.084$-versus-$0.9996$ rescue folds in two registered changes — centering \emph{and} a switch from the old scale-only lens to the full-intertwiner crosscheck. Isolating centering under a matched full-intertwiner metric moves recovery $0.705\!\to\!0.9996$; under the scale-only lens the centered pipeline reads $0.046$ (no better than the uncentered $0.084$, because this injection's true gauge is non-identity — the pre-echo of \S\ref{sec:gauge}). Centering is a genuine fix, but the full rescue is centering plus a full-intertwiner degauge, not centering alone."
**Why:** A3. Verified against `gate1b_recheck.txt` (md5 2d170cc0…): 0.084 = uncentered/scale-only (line 32), 0.9996 = centered/full-Q (line 19), 0.705 = uncentered/full-Q (line 43), 0.046 = centered/scale-only (line 19); the matched-metric centering isolation is 0.705→0.9996. Attributing the whole rescue to "centering" is a misattribution in the incident whose lesson is *don't misattribute*. Most of this fix lands in the **uncounted appendix**, so its body cost is ~half a line in §6 (offset by the FIX-4 §6 compression).

---

### FIX-9: Soften the two companion-dependent sentences and add the in-paper anchor
**Severity:** SERIOUS → **PARTIAL** (attack survives in reduced form)
**File(s):** `sections/03_case_tolerance.tex`; `sections/09_appendix.tex` (B2 punchline)
**Location:** §3, the "capacity cliff … super-linear growth" sentence; appendix B2, the "zero-padded re-run subsequently delivered it" sentence
**Before (§3):** "the shape of a capacity cliff, contradicting the same architecture's established super-linear growth at lower state sizes~\citep{companion2026capacity}."
**After (§3):** "the shape of a capacity cliff — surprising because the same wide grid unlocked to no cliff through $K/\dstate=0.9375$ once the tolerance was fixed (below), consistent with the architecture's established super-linear growth~\citep{companion2026capacity}."
**Before (appendix B2):** "a zero-padded re-run subsequently delivered it~\citep{companion2026capacity}."
**After (appendix B2):** "a corrected-target (zero-padded) re-run was subsequently able to deliver the test (companion work, anonymized)~\citep{companion2026capacity}."
**Why:** A8. The anonymized companion is standard for double-blind, so no CONCEDE, but two of six incidents leaning on it is a real, actionable weakness. The §3 anomaly can be anchored in this paper's own I4 unlock fit (no cliff through K/d=0.9375). The B2 punchline still leans on the companion after softening — **residual exposure recorded**; workshop-survivable.

---

### FIX-10: Scope the economics to one incident, admit the missing false-alarm denominator, fix row I6
**Severity:** SERIOUS
**File(s):** `sections/08_related_limits.tex`; `brief.md` (row I6)
**Location:** §8 Limitations, the economics sentence
**Before:** "The economics still favor adoption: Case I's two decisive diagnostic instruments together cost under 2 GPU-hours against the 6.33 GPU-hours of training their uncorrected readings would have discredited; auditing the lens is cheap relative to the compute, and the claims, it protects."
**After:** "The economics favored adoption in the one incident we costed: Case I's two diagnostic instruments together cost under 2 GPU-hours against the 6.33 GPU-hours of training their uncorrected readings would have discredited. We do not report a false-alarm rate — the records do not tally how many instrument-health adjudications in the window escalated versus confirmed the instrument healthy — so the policy's precision is unmeasured."
**Also:** fix `brief.md` row I6 to name the three raw sources — the per-cell `wall_s` JSONs (6.33 GPU-h), `poolmargin_run1.log` (1.09 GPU-h, verified: wrapper_wall_s 2608.7 + 1343.5 = 3952.2 s), and the c17 repro log — and **either locate the 0.82 GPU-h repro-instrument figure in a named raw or drop it** (unsourced this session, per both prior reports). If dropped, §8's "under 2 GPU-hours" should rest on the sourced numbers only.
**Why:** A10. The paper recommends a policy ("suspect the instrument first") with no measured precision, generalizes a one-incident GPU-hour result to "the economics," and row I6 is the single evidence row that violates the paper's own Rule 4 (its raw column reads "no single raw file carries all three").

---

### FIX-11: Faithfully describe the primary verdict — S3 cleared its bar; the "different reason" is authorial gloss
**Severity:** MINOR
**File(s):** `sections/05_case_gauge.tex`; `sections/09_appendix.tex` (catalogue row 3)
**Location:** §5, the "re-metric" closing sentence; the `tab:catalogue` row-3 first column
**Before (§5):** "The corrected endpoint is \emph{still} \textsc{falsify}, for a more informative reason (\S\ref{sec:notflipped}, Table~\ref{tab:remetric}): not ``no signal anywhere'' but ``two specific, lens-independent failures at the two decisive groups.'' Fixing the instrument changed the reason, not the verdict."
**After (§5):** "The corrected endpoint keeps the same \textsc{falsify} verdict \emph{and the same machine-emitted reason} (\S\ref{sec:notflipped}, Table~\ref{tab:remetric}); what the fix changes is the readable geometry. The primary lens was not a blanket null — under it S3 in fact cleared its bar ($0.54$ vs $0.459$) — but it masked which groups fail; under the crosscheck the two decisive failures (S5 near-miss, A6 non-convergent) stand out lens-independently. Fixing the instrument changed what the verdict \emph{means}, not the verdict."
**Before (catalogue row 3):** "Generalization endpoint falsified with no signal anywhere"
**After (catalogue row 3):** "Generalization endpoint falsified (\S\ref{sec:gauge})"
**Why:** A14. Verified: under the primary lens S3 far 0.54 > bar 0.459, `far_clears=True` (`stage2_harvest_report.json`, md5 7dddd19b…) — so "no signal anywhere" mis-states the primary as a blanket null. And the primary and crosscheck `reason` strings are **byte-identical** (confirmed string-equal this session) — the "more informative reason" is the authors' reading of the per-cell geometry, not something the machinery emitted.

---

### FIX-12: State the teeth-checkpoint selection rule and the CORRECT robustness clause
**Severity:** MINOR
**File(s):** `sections/05_case_gauge.tex`
**Location:** §5, the "Teeth" paragraph and the "All 13 cells … at least $0.5$" sentence
**Add (Teeth paragraph):** "(The three teeth checkpoints, pre-registered, span three groups and include the two exemplars of this section.)"
**Append to the "13 cells … at least $0.5$" sentence:** "— a threshold-robust signature: every cell whose lens disagreement exceeds $0.3$ is a converged cell ($16$ of $16$), the same qualitative split the $0.5$ cut reports."
**DO NOT ADD** the attacker's proposed clause "the 13-cell count is identical for any cut in $[0.3, 0.7]$" — it is **false**.
**Why:** A13. Verified this session: the count is **16 / 14 / 13 / 12 / 11** at cuts 0.3 / 0.4 / 0.5 / 0.6 / 0.7 (NOT invariant); min gap among the ≥0.5 group is 0.55 (attacker said 0.65), max nonzero gap below is 0.4 (attacker said 0.1). What **is** robust and true: at cut 0.3, 16/16 cells are converged (loss<0.02); at cut 0.5, 13/13; the one non-converged cell with a nonzero gap has gap 0.15 (loss 0.073). So "disagreement > 0.3 ⟺ converged" holds; the exact count does not. This is the one place the rebuttal corrects, not relays, the attacker.

---

### FIX-13: Fix the "global optimum for the linear probe class" imprecision
**Severity:** MINOR
**File(s):** `sections/04_case_wronglayer.tex`
**Location:** §4, the offline-refit-battery sentence
**Before:** "closed-form ridge regression, the global optimum for the linear probe class, reproduced $\rf=0.0$ exactly, as did pinned SGD and MLP probes;"
**After:** "closed-form ridge regression ($\lambda$ swept), pinned SGD, and MLP probes all reproduced $\rf=0.0$ exactly — the same rig recovers $\rf=0.674$ at the post-nonlinearity tap, ruling out over-regularization;"
**Why:** A12. Ridge at λ≈100 is the optimum of the *penalized* objective, not the unregularized linear-probe class; the phrase as written invites a "the λ shrank the cosines" rebuttal. The three-probe convergence and the variant-(iv) 0.674 recovery already kill that rebuttal — say so instead of asserting a false optimality.

---

### FIX-14: Add the separation rule to the `tab:remetric` caption
**Severity:** MINOR
**File(s):** `sections/09_appendix.tex`
**Location:** `tab:remetric` caption
**Before:** (caption ends) "…A6 never converges at its decisive configuration under either lens, and S5 is held under its bar by one non-generalizing seed. Evidence row X5." (note: the S5 clause here is also replaced by FIX-5)
**After:** append "Separation requires the contender to clear its bar \emph{and} the Arm-2 baseline to collapse below $50\%$ of its own ceiling; S3 clears its bar but its baseline does not collapse ($0.15>0.125$), hence no separation."
**Why:** A15. Verified faithful to the pre-registration (`verdict_crosscheck_lens.per_group.S3`: `arm2_collapses=False, separates=False`). Without the rule printed, the row "S3 far 0.900 vs baseline 0.150 → separates: no" reads as a contradiction against "S4 1.000 vs 0.000 → yes."

---

### FIX-15: Stop framing all six as trained-model failures — one was a synthetic oracle
**Severity:** MINOR
**File(s):** `sections/00_abstract.tex`
**Location:** abstract, the "in which a trained model appeared to fail" clause
**Before:** "…in which a trained model appeared to fail and the failure traced instead to the instrument:"
**After:** "…in which a model — in one case a synthetic oracle — appeared to fail and the failure traced instead to the instrument:"
**Why:** A16(a). B1's failing "model" is a synthetic *perfect* injection in an acceptance test; the abstract's uniform "a trained model appeared to fail" over-covers it by one. The §1 enumeration already lists "a perfect synthetic model rejected by its acceptance test," so only the abstract needs the qualifier. (A16(b), the venue-is-an-assumption item, is **not a paper defect** — no text change; the writer runs the Jul-11 venue refresh and applies the FIX-4 page-budget decision against the confirmed limit.)

---

### FIX-16: Brief-map hygiene — cite 0.0295 to the file that contains it; correct the I1 directory count
**Severity:** MINOR
**File(s):** `brief.md` (rows W1, I1) — **no draft-text change**
**Location:** brief rows W1 and I1
**After:** row W1 — add the transformer baseline recall source: `experiment-runs/2026-07-09_h2h_calib3/results/h2h_calib_transformer_task1_calib_primary_K32_auxrev2.json` (`final_rung1_accuracy = 0.029541…`, with md5), since the currently-cited `tap_localization_SUMMARY.json` has no transformer field and the diagnosis JSONs carry 0.0304 (a different quantity, W3's). Row I1 — reword "(12 new-cell JSONs)" to "16 JSONs on disk: 12 new per §15.22's cell list (K∈{72,78,84,90}×3) plus 3 reused K69 and one contingency seed."
**Why:** A11. A verifier following the map cannot find 0.0295 in the cited artifacts, and the I1 directory holds 16 JSONs, not 12. Pure map-hygiene; changes no claim.

---

## 3. Verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 | CRITICAL | CONCEDE + FIX | ATTACK UPHELD — resolved by fix (stays CRITICAL, closed) | FIX-1 |
| A2 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-3 |
| A3 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix (borderline MINOR) | FIX-8 |
| A4 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-5 |
| A5 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-6 |
| A6 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-7 |
| A7 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — **RE-RATED CRITICAL**; resolved by fix | FIX-2 |
| A8 | SERIOUS | PARTIAL | PARTIAL — survives in reduced form | FIX-9 |
| A9 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-4 |
| A10 | SERIOUS | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-10 |
| A11 | MINOR | CONCEDE + FIX | ATTACK UPHELD — brief-map fix | FIX-16 |
| A12 | MINOR | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-13 |
| A13 | MINOR | PARTIAL (attacker's remedy false) | ATTACK UPHELD **in corrected form only** | FIX-12 |
| A14 | MINOR | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-11 |
| A15 | MINOR | CONCEDE + FIX | ATTACK UPHELD — fix | FIX-14 |
| A16 | MINOR | PARTIAL (concede a, defend b) | (a) ATTACK UPHELD — fix; (b) DEFENSE VALID — no change | FIX-15 |

**Disposition counts (adjudicated):** ATTACK UPHELD + fix: 13 (A1, A2, A3, A4, A5, A6, A7, A9, A10, A11, A12, A14, A15); ATTACK UPHELD in corrected form: 1 (A13); PARTIAL / survives reduced: 1 (A8); split (a upheld+fix / b DEFENSE VALID): 1 (A16). **DEFENSE VALID (no change):** the single half A16(b). **CRITICAL open after fixes: none** (A1 closed by FIX-1; A7 re-rated CRITICAL, closed by FIX-2).

---

## 4. Residual risk after all fixes

Workshop-survivable unless noted; none is conference-blocking after the fix wave.

1. **A8 residual (companion still load-bearing once, PARTIAL).** After FIX-9, Case I's premise is anchored in-paper (the I4 unlock fit), but B2's punchline ("the corrected-target re-run delivered the test") still rests on the anonymized companion. Standard for double-blind; a reviewer can note it but not act on it. **Low.**
2. **The two genuine model failures both live inside Case III** (defense "weakness the attacker missed" #1). Contribution 3 presents A6 and S5 as a program-level property, but both come from the single Case III re-metric (Case I's two withheld findings corroborate). FIX-5/FIX-11 tighten the language; if space allows, add a half-sentence scoping the anti-laundering evidence to "one incident's re-read, corroborated by Case I." **Low–medium; not blocking.**
3. **The falsifiability headline generalizes from one falsifier instance** (defense #3). "The discipline is itself falsifiable" is demonstrated only for Rule 3's teeth on Case III (three checkpoints). FIX-12 adds the selection rule and the correct robustness clause; the abstract's "the discipline is itself falsifiable" is defensible as scoped ("its decisive crosscheck survived a pre-registered control"), but a sharp reviewer may ask about the other four rules. **Low.**
4. **A10 residual (0.82 GPU-h provenance).** If the 0.82 figure cannot be located in a named raw, §8's "under 2 GPU-hours" must rest on the sourced 1.09 + a stated estimate. **Low.**
5. **A16(b) — the venue itself is unconfirmed.** The 4pp limit, template, review style, and archival status are working assumptions until the Jul-11 NeurIPS accepted-workshop list. The FIX-4 fallback keeps the paper valid at a hard 4pp; a 5pp confirmation removes all tension. **Process item, not a paper defect** — but it gates the FIX-4 page-budget decision, so the refresh is a hard pre-submission step.

---

## 5. New citations

**MUST-CITE (all required by A7 / FIX-2; re-verify each arXiv id against the arXiv export API before insertion, per the refs.bib header discipline):**
- **Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?," NeurIPS 2023, arXiv:2304.15004** — the highest-profile "apparent model property is a metric artifact" result; this paper's thesis inverted, the comparison a measurement-workshop reviewer expects named first. **Highest priority.**
- **Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks," EMNLP 2019, arXiv:1909.03368** — control tasks / selectivity = Case II's shuffled-tap controls; cite inline at the §4 control.
- **Elazar, Ravfogel, Jacovi & Goldberg, "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals," TACL 2021, arXiv:2006.00995** — causal removal vs decodability = Case II's state-zeroing localization; cite inline at the §4 zeroing.
- **Adebayo et al., "Sanity Checks for Saliency Maps," NeurIPS 2018, arXiv:1810.03292** — instrument falsification via randomization controls = the direct ancestor of Rule 3's teeth.
- **Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances," Computational Linguistics 48(1), 2022, arXiv:2102.12452** — the survey naming the probe-vs-behavior dissociation Case II reports.

**SHOULD-CITE (add if space after the MUST-CITE block; not submission blockers):**
- **Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science," Patterns 2023, arXiv:2207.07048** — leakage taxonomy supporting the crosscheck's fit/eval-split leakage-guard (tiebreak ground 1); shares a first author with the already-cited REFORMS, so nearly free to add.
- **Northcutt, Athalye & Mueller, "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks," NeurIPS 2021 D&B, arXiv:2103.14749** — broken measurement substrate flipping verdicts; benchmark-side sibling of the broken-lens catalogue.
- *(Optional)* **Voita & Titov, "Information-Theoretic Probing with Minimum Description Length," EMNLP 2020, arXiv:2003.12298** — preempts a "why rf@0.9 at all" review with one §4 clause.

---

## 6. Re-run instruction (scoped, not a full restart)

After the fixes, these claims/sections must re-enter a scoped attack → defense → rebuttal pass; everything else in the draft is untouched and need not re-run:

- **FIX-1 (A1) → claim X2 / §5 + the count-verification sweep.** Re-attack must confirm "25 Arm-2 baseline cells, 50 checkpoint values, zero divergence" against the raw, and confirm the sweep re-derived every other bare count (§3 "11 cells", 4,608, 107/107, 36 events, 37/39). This is the CRITICAL closure check — do not skip.
- **FIX-2 (A7) → §8 Related work + §4 inline cites.** Re-attack verifies the five arXiv ids resolve, the distinction paragraph is present, and the two inline cites land at the control and the zeroing.
- **FIX-3 (A2) → claims B2 + catalogue row 5 + §1 + §6.** Confirm the "zeros at every rank" universal is gone in all four sites and the S3 0.17/0.08 exception reads correctly.
- **FIX-4 (A9) → structural: §1, §6, §8 layout + Table 1 placement + brief X2.** Re-attack confirms the catalogue is in the body (or the fallback triggered), the page limit is met against the Jul-11-confirmed venue, and no float reference broke.
- **FIX-5 (A4) → claim X5 / §7 + `tab:remetric` caption.** Confirm the causal "because" is gone and the lens-independent-failure framing is arithmetically clean.
- **FIX-7 (A6) → abstract + §2.** Confirm "one and the same discipline" is rescoped and the "in force since" note is present.
- **FIX-9 (A8) → §3 premise + appendix B2 punchline.** Confirm the in-paper anchor and the softened companion dependence; record the residual.
- **FIX-11 (A14) → claim X5 / §5 + catalogue row 3.** Confirm the primary-verdict description (S3 cleared its bar; byte-identical reason string) is faithful.
- **FIX-12 (A13) → §5 teeth + robustness clause.** Confirm the attacker's false "identical count" clause was NOT added and the "disagreement>0.3 ⟺ converged, 16/16" clause is present.

The **abstract** and the **contribution list (§1)** are touched by FIX-6, FIX-7, and FIX-15, so both re-enter the pass. FIX-8, FIX-10, FIX-13, FIX-14, FIX-16 are local edits/brief-map fixes and can be verified inline without a full re-attack.

Emit after the fixes are applied; the gauntlet re-runs the scoped pass above, then proceeds to the detector gate.
