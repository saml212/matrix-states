# 01b_reattack_report — gauntlet round 1, SCOPED RE-ATTACK (fresh-context attack agent, 2026-07-10)

Fresh-context hostile re-review. Two mandates: (A) independently verify
closure of the eight original attacks (A1–A8 of `01_attack_report.md`)
against the CURRENT draft and the raw artifacts; (B) fresh attack on the
revised/new prose. All numeric checks recomputed from the
`experiment-runs/` JSONs with pure-python (numpy absent).

---

## 1. Summary

### 1.1 Per-original-attack closure table

| Id | Original severity | Status now | Residual (severity) |
|---|---|---|---|
| A1 (transformer LR / MQAR literature) | CRITICAL | **REDUCED** | The undertuned transformer is still displayed prominently (abstract, Table 1, both figures, and the constant-memory clause), and the paper's own calibration selected a *higher* LR (1e-3) for the transformer on the LM control while recall ran at 3e-4 with near-flat loss; the LR-grid re-run is named but un-run. Fully disclosed, verdict now carried by the ablation alone. (residual: SERIOUS as a reviewer-ask, not a claim-invalidator) |
| A2 (substrate conflation) | CRITICAL | **CLOSED** | none found — grep for any rank/dimension/recruitment attribution to "delta-rule"/"DeltaNet" returns zero hits; §6 and §8 rewritten with per-leg scoping |
| A3 (K12 rank-tolerance false) | SERIOUS | **CLOSED** | Appendix A now discloses both loads honestly (K8 0.03–0.26; K12 0.59–0.78), verified against the raw. Tiny transparency asymmetry: the K8 line silently uses the converged subset (one of five K8 runs, s0, excluded as non-converged) without naming the exclusion, though "converged" is stated at the appendix head. (MINOR) |
| A4 (37-vs-36 of 39 count) | SERIOUS | **CLOSED** | §3.3 now says "36 of 39"; my independent recompute = 36/39 exactly; borderline (0.072) and outliers (0.15/0.17) correctly named |
| A5 (stale "in flight") | SERIOUS | **CLOSED** | §4.5/§7 now report the landed task2 diagnosis + K48 table; no "in flight"/"incomplete" language survives anywhere; numbers verified against `TASK2DIAG_VERDICT.json` |
| A6 (df=2 CIs undisclosed) | SERIOUS | **REDUCED** | §2.4 now names n=3 / df=2 / t=4.303 explicitly and points to Appendix B per-seed tables; no bootstrap/permutation cross-check was added (residual: MINOR) |
| A7 (shuffled-control metric ambiguity) | MINOR | **CLOSED** | §4.3 parenthetical now names it "mean-cosine gaps, a separate continuous sanity check on a different scale from rf@0.9" |
| A8 (per-seed val-loss exceedance undisclosed) | MINOR | **CLOSED** | §5.3 now discloses the 3.2038 seed vs the 3.2020 ceiling with the passing arm mean (3.1961); verified exactly against `fixscale_harvest_verdict.json` |

### 1.2 New findings by severity

- CRITICAL: 0
- SERIOUS: 0 (the strongest surviving concern is the A1 residual above, which is a disclosed limitation, not a fresh defect)
- MINOR: 5 (N1–N5 below)

### 1.3 Bottom line

**No CRITICAL survives in the current draft.** Both original CRITICALs
(A1, A2) are resolved as central claims: A2 completely, A1 by scoping the
capability-separation verdict onto the parameter-matched ablation and
labelling the transformer a disclosed degenerate baseline. Every numeric
item I re-verified (A3, A4, A8, the task2/K48 diagnosis, the transformer
training curves + LR, the fix-at-scale deltas/CIs, the M\* horizon reads,
and R1's rank means/ρ) matched the raw artifacts. The remaining issues
are wording/precision nits (N1–N5) and one disclosed-but-un-run reviewer
ask (the transformer LR grid). The draft is in submittable shape pending
those minor edits.

---

## 2. Closure verifications (independent evidence)

### A1 — REDUCED (was CRITICAL). Transformer LR / MQAR literature.

**What changed and holds.** Three of the original A1 defuse actions are
now in the draft and check out:

1. *Literature engaged.* §6 carries a new paragraph "Associative recall
   in transformers": "Arora et al. (2023) introduce the MQAR benchmark
   and show attention solves it where sub-quadratic models struggle;
   Jelassi et al. (2024) show transformers outperform state-space models
   at copying; Olsson et al. (2022) show that two-layer transformers
   reliably develop induction circuitry from scratch. Our transformer
   baseline's chance-level reading therefore runs against the expectation
   this literature sets, and we do not interpret it as an architectural
   inability." All four bib entries (`arora2023zoology`,
   `jelassi2024repeat`, `olsson2022induction`) are present and the prose
   years match the bib years.
2. *Training curves + LR disclosed and correct.* §4.1: "its learning
   rate was the shared default 3×10⁻⁴... no architecture-specific
   learning-rate search was performed for the transformer on the recall
   task... Its recall-task training loss is near flat across the run
   (7.81 to 7.84 at step 500, 7.45 to 7.51 at step 20,000...)."
   Verified against `h2h_transformer_task1_sweep_s{0,1,2}.json`: `lr`
   field = 0.0003 in all three seeds; step-500 train_loss = 7.836 / 7.811
   / 7.822 (range 7.81–7.84 ✓), step-20000 = 7.514 / 7.453 / 7.459 (range
   7.45–7.51 ✓). The contender and ablation task-1 cells are also lr=3e-4
   (confirmed), so the "shared default" claim is exact.
3. *The 1e-3 grid was task-3-only, and task-3 is genuinely the LM
   control.* §4.1: "the calibration record's three-point learning-rate
   grid was run on the language-modeling control task." Verified:
   `h2h_transformer_task3_sweep_s0.json` has `lr`=0.001, task=task3_sweep,
   corpus=openr1-mix-ext, init_val_loss 10.81 → final 1.78 (it learned
   fine); `HEAD_TO_HEAD_DEMO_DESIGN.md` line 1043 defines Task 3 as
   "real-data LM quality" and line 4660 calls it the "LM control." The
   contender/ablation task-3 cells stayed at 3e-4; only the transformer
   got the 1e-3 override — exactly the FIX-1 disclosure.
4. *Billing downgraded.* Abstract ("a compute-matched transformer also
   reads chance, disclosed as a degenerate baseline"), Intro contribution
   2 ("recorded as a degenerate-baseline datum, not a second verdict"),
   §4.2 ("the separation verdict is carried by the ablation comparison"),
   and §6 ("carried by the vector-state ablation comparison alone") all
   now scope the verdict off the transformer.

**Residual (why REDUCED, not CLOSED).** The transformer is still shown
prominently at chance in the abstract headline, Table 1, Figures 3 and 4,
and the constant-memory clause — and the paper's *own* calibration data
show the transformer prefers a higher LR (its grid selected 1e-3 for the
LM control) while recall was run at 3e-4 with a near-flat loss curve. A
hostile reviewer can still argue the resolving experiment (the §7 LR
grid, 10⁻⁴ to 3×10⁻³) should be run before submission rather than
deferred, because displaying a baseline the authors have positive reason
to believe is mis-tuned inflates the apparent separation. This does not
invalidate any claim — every verdict now rests on the ablation — so it is
a disclosed limitation and reviewer-ask, not a live CRITICAL. See also N1
and N2, which are wording residues of this same seam.

### A2 — CLOSED (was CRITICAL). Substrate conflation.

The original A2 pointed at two loci. Both are rewritten:

- §8 Conclusion now reads "In a matrix-state encoder family, recruited
  dimensionality equals the task's minimal faithful representation
  dimension... In a two-layer delta-rule model, the first layer's state
  carries an episodic-recall capability..." — the rank law is scoped to
  the encoder family, the capability to the delta-rule model, per-leg.
- §6 "DeltaNet and gated variants" now reads "DeltaNet and Gated DeltaNet
  (Yang et al.) supply the architecture substrate for the capability and
  pathology legs (Sections 4 and 5)... the rank law of Section 3 is
  established on a separate matrix-state encoder family with no delta-rule
  write." — the false "DeltaNet supplies the substrate for recruited
  dimensionality" attribution is gone.

**Independent sweep for any surviving instance.** `grep -rni` across all
11 section files for a delta-rule/DeltaNet token co-occurring with
rank/dimension/recruit returns **zero hits**. Every "fast-weight" mention
that survives is correctly scoped: §2.1 (delta-rule states, definitional),
§4.1 ("the fast-weight mixer" = the contender, which is delta-rule),
§4.3 heading ("Fast-Weight-Resident" — the recall capability, on the
delta-rule substrate), §6 (Nichani, general), §1/abstract (general
model-class framing). The conflation the brief's T1 WORDING FLAG warned
about is not realized anywhere in the current draft.

### A3 — CLOSED (was SERIOUS). K12 rank tolerance.

Appendix A (§9) now reads: "The effective rank of Z − c\*I sits within 0.3
of the K−1 target at the K=8 load (deviations 0.03 to 0.26); at the K=12
load the three converged runs read 10.22 to 10.41 against the target of
11, deviations of 0.59 to 0.78." Verified against
`2026-07-09_zdump_complement/complement_results.json` (per-example mean of
`effrank_Z_minus_cI`):

- K12_frN_s0/s1/s2 = 10.4149 / 10.2248 / 10.3046 → deviations from 11 =
  **0.585 / 0.775 / 0.695** → "0.59 to 0.78" ✓; readings "10.22 to 10.41" ✓.
- K8_frN converged runs s1–s4 = 6.7444 / 6.9265 / 6.8216 / 6.9693 →
  deviations 0.256 / 0.074 / 0.178 / 0.031 → "0.03 to 0.26" ✓.

The false "within 0.3 at both loads" claim is gone; both loads are now
disclosed exactly. **MINOR residual:** the K8 line silently uses the
converged subset — K8_frN_s0 (mean effrank 9.331, deviation 2.33) is
excluded, legitimately (its h21_recov = 0.0001, task cos 0.49 vs ~0.98–1.0
for s1–s4, i.e. genuinely non-converged). The appendix opens "In the
converged matrix-state encoder family," which covers the scope, but it
names "three converged runs" at K12 while giving no run count / exclusion
note at K8. A one-clause parallelism fix ("four converged K=8 runs,
s1–s4") would remove the asymmetry.

### A4 — CLOSED (was SERIOUS). The 36-vs-37 of 39 count.

§3.3 now reads "36 of 39 force-rank cells sat within 0.07 of the predicted
ceiling, one further cell sat marginally outside at 0.072, and the two
remaining outliers (0.15 and 0.17, both in one group's below-d_min arm)."
I recomputed |Δ| = |obs − √(k/d_state)| for all 39 force-rank cells using
the harvest script's own definition (obs = mean of
`convergence_profile[1..8]`, d_state = d_min+2 per group), reading each
cell's raw JSON in `2026-07-09_capability_sweep_harvest/results/`:

- **within 0.07: 36 of 39** ✓ (matches the corrected draft exactly).
- The three outside: `S5__k_dmin_minus_1__seed1` = 0.166425,
  `S5__k_dmin_minus_1__seed0` = 0.153678, `S5__k_dmin_plus_1__seed0` =
  0.071571.
- Borderline "0.072" = 0.071571 rounded to 3 d.p. ✓; the two outliers
  "0.15 and 0.17" = 0.1537 / 0.1664 ✓, both in S5's k_dmin−1
  (below-d_min) arm ✓.

The original A4 (37/39 off-by-one) is fully corrected and the borderline
cell is now named in-prose.

### A5 — CLOSED (was SERIOUS). Stale "in flight" currency.

`grep -rni "in flight|in-flight|incomplete|pending"` across all sections
returns only the "[AUTHORS — PI decision pending]" HTML comment; the
substantive "in flight" language is gone. §4.5 now reports the landed
diagnosis and §7 reports the completed K48 table. Verified against
`2026-07-10_h2h_task2diag/results/TASK2DIAG_VERDICT.json`:

- §4.5 "three contender seeds clear the demonstration bar (0.334, 0.391,
  0.479)" = pooled seeds s2/s7/s5 = 0.33447 / 0.39087 / 0.47949 ✓; "zero
  of nine ablation seeds" (all ≤0.037) ✓.
- §4.5 "pooled paired interval (−0.020, +0.268)" = ci_low −0.01999 /
  ci_high 0.26782 ✓; "strict-tier reading is a TIE, flagged
  non-decision-grade by a pre-registered batch-effect gate" =
  `strict_tier_reading_disclosed_only: "TIE [NON-DECISION-GRADE...]"`,
  `pooled_reading_decision_grade: false`, ablation `var_ratio 6.14 > 4.0`
  flagged ✓; "hard-capability-boundary hypothesis is rejected" ✓.
- §7 "contender 0.0189, ablation 0.0195, transformer 0.0218, against
  chance 0.0208" = 0.018880 / 0.019531 / 0.021810 / 0.020833 ✓; locate-only
  bar 0.0625, none clears ✓.

### A6 — REDUCED (was SERIOUS). df=2 CIs.

§2.4 now has a "Confidence intervals" paragraph: "Paired confidence
intervals in this paper are Student-t intervals on n=3 per-seed paired
differences (two degrees of freedom, t.975 = 4.303), except the nine-seed
diagnosis pool of Section 4.5." t(2, .975) = 4.30265 ✓. It adds "Per-seed
values are tabulated in Appendix B so any alternative construction can be
applied." The explicit-df disclosure prong of the original A6 is CLOSED.
**Residual (MINOR):** no nonparametric/bootstrap cross-check was added
for R4's CI; the draft leans on the effect clearing the margin by
multiples of the width instead. Acceptable given the effect size, but a
reviewer could still ask for the one-line robustness footnote.

### A7 — CLOSED (was MINOR). Shuffled-control metric.

§4.3 now: "the accompanying shuffled-control gaps of at most 0.063 are
mean-cosine gaps, a separate continuous sanity check on a different scale
from rf@0.9." The metric is named; the original ambiguity (a reader
mistaking 0.063 for an rf@0.9-scale gap) is removed. Consistent with
`tap_localization_contender.json`'s `gap_vs_shuffled_cos_mean`.

### A8 — CLOSED (was MINOR). Per-seed val-loss exceedance.

§5.3 now: "one individual seed in the 98M narrative-mix cell reads 3.2038
against the 3.2020 ceiling while its arm mean (3.1961) passes." Verified
against `fixscale_harvest_verdict.json`,
`scales/98m/corpora/wikitext-mix-ext/val_loss_gate`:
`per_seed_over_ceiling` = [3.2037888…] (→ 3.2038 ✓), `pass_ceiling` =
3.2020253… (→ 3.2020 ✓), `per_token_mean` = 3.1960760… (→ 3.1961 ✓),
`PASS_by_arm_mean` = true. Exact.

---

## 3. New attacks on the revised/new prose (sorted by severity)

All MINOR — none blocks submission. Listed for completeness so the
defense can dispatch them in one pass.

### N1: Intro contribution 2 says "three times its width" where §4.2 (correctly) says "three times the margin"

**Severity:** MINOR. **Type:** internal-inconsistency / clarity.

**Attack.** Intro contribution 2 (revised): "the paired confidence
interval excludes the pre-registered 0.30 margin at **more than three
times its width**." §4.2 states the same fact as "both intervals exclude
the frozen 0.30 margin, the first **by more than three times the margin
at its floor**." The §4.2 version is correct: the contender−ablation CI
floor is 0.95822, which is 3.19× the 0.30 margin. The intro's "three
times its width" uses the wrong referent — the interval's actual width is
0.97293 − 0.95822 = 0.0147, and the floor sits 0.658 above the margin,
i.e. ~45× the width, not 3×. The clause is either wildly understated (if
"width" means the CI width) or a mislabeling of "the margin" as "its
width." Either way it disagrees with §4.2's cleaner statement of the same
number.

**Supporting evidence.** `01_introduction.md` contribution 2;
`04_capability_separation.md` §4.2; CI (0.95822, 0.97293) from
`h2h_sweep_harvest` / `compute_verdict.py`.

**Defuse.** Change the intro to match §4.2: "…excludes the 0.30 margin,
its floor sitting at more than three times the margin."

### N2: The abstract's constant-memory clause drops the degenerate-baseline caveat that §4.4 attaches

**Severity:** MINOR. **Type:** claim-scope / abstract-body consistency.

**Attack.** The abstract's capability sentence ends "…and holds at 0.998
or higher to 1798 tokens on a fixed 32,768-byte state **while cache-capped
transformers read chance at every budget**." This is the same
compute-matched transformer the abstract *earlier* labels "a degenerate
baseline," but here in the constant-memory clause the caveat is not
re-attached. §4.4 is careful: it fires "the pre-registered
degenerate-baseline clause," states the verdict is "the baseline is
non-competitive at matched parameters and tokens, not a
strongest-possible-baseline result, and no crossover-point value is
certified," and names the two informative reads (the contender's flat
horizon; "capping never helps"). The abstract compresses this into a
clean-looking win. The informative content is genuinely tuning-independent
(capping cannot rescue a model that already fails uncapped — verified:
`MSTAR_VERDICT.json` uncapped H4 reads 0.031/0.033/0.030, capped reads
0.020–0.033), so the claim is defensible, but the headline invites
over-reading the untuned baseline.

**Supporting evidence.** `00_abstract.md` final capability clause;
`04_capability_separation.md` §4.4; `2026-07-10_h2h_mstar/MSTAR_VERDICT.json`
(`degenerate_baseline_clause_fires: true`,
`verdict_of_record: BASELINE_NON_COMPETITIVE_AT_MATCHED_BUDGET`).

**Defuse.** Add "(a degenerate baseline at this matched budget)" or fold
the M\* clause under the same "disclosed as a degenerate baseline" tag the
abstract already uses two clauses earlier.

### N3: Xiao et al. cited as "2023" in prose but the bib entry is year 2024

**Severity:** MINOR. **Type:** citation-correctness.

**Attack.** §4.4 and the Figure 3 caption both cite "(Xiao et al., 2023)"
for streaming/sink attention. `references.bib` has
`@misc{xiao2024efficientstreaminglanguagemodels, … year={2024}, …
eprint={2309.17453}}`. The prose author-year (2023) does not match the
bib year (2024). arXiv 2309.17453 was first posted Sept 2023 but the
programmatic export uses 2024 (its ICLR-2024 year); the prose must match
whichever the bib carries or the `\citep` will render "Xiao et al., 2024"
against the prose's "2023."

**Supporting evidence.** `04_capability_separation.md` §4.4 and Fig 3
caption; `references.bib` lines 44–52.

**Defuse.** Make the prose read "(Xiao et al., 2024)" to match the bib, or
edit the bib `year` to 2023 (pick one; the `\citeyear` must agree).

### N4: §5.1 "two orders of magnitude smaller" overstates a ~22× ratio

**Severity:** MINOR. **Type:** numeric-framing (pre-existing / untouched prose).

**Attack.** §5.1: "completing the budget would move the endpoint by
roughly 0.003, **two orders of magnitude smaller** than the 0.066
increment from the previous rung." 0.066 / 0.003 = 22, i.e. ~1.34 orders
of magnitude, not two (two orders would require the shift to be ≈0.0007).
The point (the 1.31B shortfall doesn't rescue the trend) stands regardless
— it is only the "two orders of magnitude" quantifier that is loose. This
sits in untouched prose (outside the fix wave), but a hostile reviewer
scanning the pathology section will catch it.

**Supporting evidence.** `05_pathology_at_scale.md` §5.1; span 0.4584 →
0.4554 = 0.0030; prior-rung increment 0.455 − 0.389 = 0.066.

**Defuse.** "…more than twenty times smaller than the 0.066 increment"
(or "roughly an order of magnitude smaller").

### N5: §2.4 "the one threshold-adjacent reading" undercounts

**Severity:** MINOR (borderline-weak). **Type:** claim-scope / statistical framing.

**Attack.** §2.4: "the verdicts that rest on these intervals clear their
margins by multiples of the interval width, and **the one
threshold-adjacent reading (Section 5.2)** is reported as an unconfirmed
trend." But §5.3's R8 transfer wave has a second threshold-adjacent
reading: the 392M reasoning-mix span-fraction delta (+0.0065, CI straddles
zero — verified `PRIMARY_delta` 0.006505, half-width 0.0421 →
(−0.0356, +0.0486)). §2.4 calls the gating result "the one"
threshold-adjacent reading; the 392M reasoning-mix null is a second. The
defense is that the 392M reading is disclosed as a *null* (not claimed as
a trend or a verdict), so §2.4's scope — "the one reading reported as a
trend" — is arguably intact. Weak, but a precise reviewer would note the
count.

**Supporting evidence.** `02_background_setup.md` §2.4;
`05_pathology_at_scale.md` §5.3; `fixscale_harvest_verdict.json` 392m
openr1 `PRIMARY_delta` and band half-width.

**Defuse.** Soften to "the one threshold-adjacent reading claimed as a
trend (Section 5.2)," which distinguishes it from the disclosed 392M null.

---

## 4. Attacks considered and dropped

- **Abstract framing "written by an outer-product update" bleeding onto
  the rank law.** The abstract's opening frames matrix states generally as
  "written by an outer-product update," then scopes the rank law to "a
  matrix-state encoder family" — which does *not* use an outer-product
  write. I considered flagging this as a residual A2. Dropped: §2.2 fully
  discloses the encoder family as "a transformer encoder that emits a
  single matrix Z," and the abstract's scoping clause ("in a matrix-state
  encoder family") is explicit, so no reader is told the rank law was
  measured on a delta-rule write. Weak.
- **454 ≠ 2×224 in §4.4.** The horizons "454, 902, 1798 tokens" are called
  "2, 4, and 8 times the bind phase" (224), but 2×224 = 448. The +6 offset
  is the query length; the raw token counts (454/902/1798) match
  `MSTAR_VERDICT`/brief R10 exactly. Too pedantic to raise.
- **R4/R5 intact-recall 0.9990 vs 0.99951.** §4.3 uses 0.9990 (the frozen
  round checkpoint, R5/§1.30) while §4.2 uses 0.99951 (the n=3 sweep mean,
  R4). Looks like an inconsistency but is correctly attributed: §4.3 says
  "On the frozen round checkpoint" and separately reports the sweep
  replication (0.0339/0.0012/0.0002). Not an error.
- **Multiple-comparisons across ~10 evidence rows.** Same disposition as
  the round-1 report: each row is a distinct pre-registered question, not
  repeated looks; the one borderline reading (R7 gating) is already
  reported as unconfirmed. Weak.

## 5. New citations for Related Work

None beyond the round-1 set. The four MUST-CITE entries the original A1
demanded (Zoology/Arora 2023, Repeat-After-Me/Jelassi 2024, Induction
Heads/Olsson 2022, Attention Sinks/Xiao) are now present in
`references.bib` and, except the Xiao year (N3), correctly reflected in
prose. The SHOULD-CITE strengtheners (Based/Arora 2025, Mamba/Gu-Dao
2024, Hopfield/Ramsauer 2021) are in the bib but not yet in prose — an
optional strengthener, not a defect.

## 6. Security note

No embedded instructions, fake system-reminder blocks, date-change
claims, or concealment requests were present in any file I read (the 11
section `.md` files, `brief.md`, `references.bib`, the raw result JSONs,
the `analyze_*.py` / `compute_verdict.py` scripts, or the design-doc
lines I grepped in `HEAD_TO_HEAD_DEMO_DESIGN.md`). The
`01_attack_report.md` I read for closure context ended with its own clean
security note. Nothing to disregard or report.
