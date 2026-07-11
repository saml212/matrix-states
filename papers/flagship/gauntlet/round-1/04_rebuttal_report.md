# 04_rebuttal_report — gauntlet round 1 (fresh-context rebuttal agent, 2026-07-10)

> Coordinator relay note: the rebuttal agent completed after its parent writer
> was stopped (model-tiering change); it returned this report as message text
> with no file written. Saved verbatim by the coordinator; the successor writer
> applies the fix list and commits this artifact.

Adjudicated against the attack report, the defense report, all eleven section files, `brief.md`, and independent spot-checks of the raws (transformer task-1 training JSONs `lr: 0.0003` all three seeds; `TASK2DIAG_VERDICT.json` and the K48 round-4 JSON, both landed; `complement_results.json` K12 effranks 10.415/10.225/10.305; `analyze_sweep_harvest.py` D-AMBIENT logic and its hand-written outlier string; `compute_verdict.py` `T975_DF2 = 4.303`; `fixscale_harvest_verdict.json` per-seed 3.2038 vs ceiling 3.2020; `model_v4.py` `BindingEncoder` docstring; `HEAD_TO_HEAD_DEMO_DESIGN.md` lines 264-265, 1529-1531, 4574, 4649). Every factual claim I checked in both reports reproduced exactly. Neither report overstated its evidence.

---

## 1. Summary for the edit agent

**Counts:** 8 attacks adjudicated. Final verdicts: 0 DEFENSE VALID, 3 DEFENSE VALID BUT EDIT (A6, A7, A8), 4 DEFENSE INSUFFICIENT — attack confirmed, resolved by the listed fix (A2, A3, A4, A5), 1 PARTIAL — ATTACK SURVIVES IN REDUCED FORM (A1). Fix list: 15 fixes — 7 CRITICAL-tier (FIX-1 through FIX-7), 5 SERIOUS (FIX-8 through FIX-12), 3 MINOR (FIX-13 through FIX-15).

**CRITICAL status after the fix list is applied: NO CRITICAL REMAINS OPEN.** Both CRITICALs close by the surgical-rewrite path, without new GPU evidence:

- **A1** closes to a SERIOUS residual via FIX-1 through FIX-4: the factually false learning-rate sentence in §4.1 is corrected (the three-point grid was run on the LM control task; the recall-task transformer trained at the untuned shared default `3e-4`, confirmed in all three raw training JSONs), the transformer is demoted from co-equal headline billing to the degenerate-baseline datum that the frozen pre-registration itself specifies (`HEAD_TO_HEAD_DEMO_DESIGN.md` lines 264-265 demote FLOP-matched to "a disclosed control... not a primary"), the near-flat training curve is disclosed, and the MQAR/induction-heads literature is engaged by name with the untuned learning rate stated as the leading unexcluded explanation. After these fixes the paper makes no claim the transformer datum must carry: the WIN verdict is carried by the ablation comparison per the frozen registration, which the raws support at more than three times the margin. The transformer task-1 learning-rate grid (FIX-5) is specified precisely as a measurement recommendation; it is NOT a submission blocker for the arXiv build once FIX-1 through FIX-4 are applied, and is strongly recommended before the ICLR submission.
- **A2** closes fully via FIX-6 and FIX-7: the Conclusion, the Related Work DeltaNet paragraph, and one additional location the attacker missed (§5.4's opening sentence) are rewritten with the per-leg substrate scoping the Introduction already uses correctly. After the rewrite the conflation does not exist anywhere in the draft.

**The four structural fixes that carry most of the weight:** FIX-1 (the false LR sentence), FIX-2/FIX-3 (transformer billing in abstract and introduction), FIX-6 (Conclusion re-scoping), FIX-10/FIX-11 (the landed task-2 diagnosis and K48 verdicts replacing "in flight").

**Bookkeeping the fixes force on the brief:** row R0 must be amended (its "lr frozen at calibration (1e-3)" claim is true only of the Task-3 override); row R1b must be corrected (37→36); row R9 must be amended with the per-load effective ranks; and a **new row R11** must be added for the landed task-2 diagnosis + K48 artifacts (`experiment-runs/2026-07-10_h2h_task2diag/results/TASK2DIAG_VERDICT.json`, md5 `66d2291d8e65932d368d8978bfd16bdc`; `transformer_task1_stress_K48_round4.json`, md5 `14e0c93f56c2a55983f929b7313eb5ac`), plus an optional **row R4a** for the transformer task-1 training curves (already inside the R4 archive) if FIX-1's curve numbers are kept. The brief's own "K48 stress table — IN FLIGHT" and "`results/` is empty" passages are stale and must be updated to match.

**Re-run instruction (targeted, not a full restart).** After the fixes, re-enter attack/defense/rebuttal ONLY on: the abstract; the Introduction thesis paragraph and contribution 2; §4.1-§4.2 framing and §4.5; §5.3 (one clause) and §5.4 (one sentence); §6 (the rewritten DeltaNet paragraph plus the new MQAR paragraph); §7 (the two updated passages); §8 in full; Appendix A's tolerance passage; §3.3's count sentence; §2.4's new CI-disclosure sentence. The rank-law numbers (R1-R3), the ladder (R6), the 2×2 (R7), and the M* fan-out (R10) verified clean against the raws and need no re-attack.

---

## 2. The ordered fix list

### FIX-1: Correct the factually false learning-rate sentence in §4.1 and disclose the transformer's flat training loss
**Severity:** CRITICAL
**File(s):** `papers/flagship/sections/04_capability_separation.md`; `papers/flagship/brief.md` (row R0)
**Location:** §4.1, second paragraph, transformer-baseline sentences
**Before:**
> The transformer baseline is a two-layer pre-norm decoder with rotary position embeddings and the contender's own MLP class, FLOP-matched to the contender within 5 percent <!-- evidence: R0 -->; its learning rate was selected on a calibration cell from a three-point grid and frozen at $10^{-3}$ before the sweep <!-- evidence: R0 -->.

**After:**
> The transformer baseline is a two-layer pre-norm decoder with rotary position embeddings and the contender's own MLP class, FLOP-matched to the contender within 5 percent <!-- evidence: R0 -->. On this task its learning rate was the shared default $3 \times 10^{-4}$, identical to the contender's and the ablation's <!-- evidence: R0 (brief row R0 must be amended: the frozen $10^{-3}$ grid selection applies to the language-modeling control task only) -->; the calibration record's three-point learning-rate grid was run on the language-modeling control task, and no architecture-specific learning-rate search was performed for the transformer on the recall task. Its recall-task training loss is near flat across the run (7.84 at step 500 to 7.51 at step 20,000) <!-- evidence: new brief row R4a required (transformer task-1 training curves; the curves are in the archived R4 training JSONs) -->, a signature consistent with an under-optimized arm; Sections 6 and 7 state the resulting interpretive limit and the measurement that would resolve it.

**Why:** Resolves the factual core of A1. The draft attaches a true statement about Task 3 to Task 1, where it is false: all three raw task-1 training JSONs read `lr: 0.0003`, and the design record (lines 1531, 4574, 4649) confirms the grid and the $10^{-3}$ override were Task-3-only. A paper whose methodology is checksum-traceable claims cannot carry a hedge that describes the wrong task.

### FIX-2: Demote the transformer from co-equal headline billing in the Abstract
**Severity:** CRITICAL
**File(s):** `papers/flagship/sections/00_abstract.md`
**Location:** the "Second, a capability separation" sentence
**Before:**
> Second, a capability separation: a two-layer delta-rule model performs episodic recall at accuracy 0.9995 <!-- evidence: R4 --> while a parameter-matched vector-state ablation and a compute-matched transformer read chance;

**After:**
> Second, a capability separation: a two-layer delta-rule model performs episodic recall at accuracy 0.9995 <!-- evidence: R4 --> while a parameter-matched vector-state ablation reads chance, the comparison that carries the pre-registered verdict; a compute-matched transformer also reads chance and is disclosed as a degenerate baseline whose learning rate was never searched on this task <!-- evidence: R4 -->;

**Why:** Resolves A1's billing half and the defense's independently found internal-consistency defect: the program's own PI-ratified pre-registration demotes the FLOP-matched baseline to "a disclosed control... not a primary" (design doc lines 264-265), and §4.2 already carries the verdict on the ablation. The abstract must match both the registration and §4.2.

### FIX-3: Demote the transformer billing in the Introduction (thesis paragraph and contribution 2)
**Severity:** CRITICAL
**File(s):** `papers/flagship/sections/01_introduction.md`
**Location:** (a) the thesis paragraph; (b) contribution bullet 2
**Before (a):**
> In a two-layer delta-rule model, the first layer's matrix state causally carries an episodic-recall capability that parameter-matched vector-state and compute-matched transformer baselines lack at equal training budget, and the stored content is linearly legible only after downstream nonlinear processing.

**After (a):**
> In a two-layer delta-rule model, the first layer's matrix state causally carries an episodic-recall capability that a parameter-matched vector-state ablation lacks at equal training budget (a compute-matched transformer also fails, disclosed as a degenerate baseline), and the stored content is linearly legible only after downstream nonlinear processing.

**Before (b):**
> On single-pass episodic recall, the delta-rule contender reads accuracy 0.99951 <!-- evidence: R4 --> against 0.03394 for its vector-state ablation and 0.02832 for a compute-matched transformer <!-- evidence: R4 -->; paired confidence intervals exclude the pre-registered 0.30 margin at more than three times its width <!-- evidence: R4 -->.

**After (b):**
> On single-pass episodic recall, the delta-rule contender reads accuracy 0.99951 <!-- evidence: R4 --> against 0.03394 for its vector-state ablation <!-- evidence: R4 -->; the paired confidence interval excludes the pre-registered 0.30 margin at more than three times its width <!-- evidence: R4 -->, and this ablation comparison carries the verdict. A compute-matched transformer also reads chance (0.02832) <!-- evidence: R4 -->; because its learning rate was never searched on this task and its training loss is near flat, it is recorded as a degenerate-baseline datum, not a second verdict (Sections 4.1 and 6).

**Why:** Same grounds as FIX-2; the introduction is the other section a skimming reviewer reads.

### FIX-4: Engage the MQAR/induction-heads literature by name in Related Work
**Severity:** CRITICAL
**File(s):** `papers/flagship/sections/06_related_work.md`
**Location:** new paragraph, inserted after the "Associative capacity of fast-weight states" paragraph
**Before:** (no such paragraph exists)
**After:**
> **Associative recall in transformers.** Multi-query associative recall is a regime where softmax attention is documented as strong: Arora et al. (2023) introduce the MQAR benchmark and show attention solves it where sub-quadratic models struggle; Jelassi et al. (2024) show transformers outperform state-space models at copying; Olsson et al. (2022) show that two-layer transformers reliably develop induction circuitry from scratch. Our transformer baseline's chance-level reading therefore runs against the expectation this literature sets, and we do not interpret it as an architectural inability. The leading candidate explanation is optimization: the arm trained at the shared default learning rate with no architecture-specific search on the recall task, and its training loss is near flat across the run (Section 4.1). The separation verdict is accordingly carried by the vector-state ablation comparison alone, with the transformer reading disclosed as a degenerate-baseline datum; Section 7 states the measurement that would resolve the question.

**Why:** Resolves A1's missing-literature half. Without this paragraph, the first MQAR-literate reviewer writes A1's attack verbatim; with it, the paper has pre-empted the objection and scoped the claim to what the raws support.

### FIX-5: The transformer task-1 learning-rate grid, specified as a measurement recommendation
**Severity:** CRITICAL (the classification decision; the prose lands in §7)
**File(s):** `papers/flagship/sections/07_discussion_limitations.md`
**Location:** "Scale scope of the capability legs," after "...requires its own matched re-run before any extrapolation."
**Before:**
> a differently scaled or differently trained transformer requires its own matched re-run before any extrapolation.

**After:**
> a differently scaled or differently trained transformer requires its own matched re-run before any extrapolation. The specific unresolved measurement is a learning-rate search for the transformer arm on the recall task itself: the frozen protocol's three-point grid was run on the language-modeling control task only, so the chance-level recall reading has not been separated from an optimization failure. The resolving experiment is a re-run of the transformer arm under the identical frozen protocol at a grid of at least four learning rates spanning $10^{-4}$ to $3 \times 10^{-3}$, three seeds, 20,000 matched steps, with training curves reported. A tuned transformer that still reads below the demonstration bar would strengthen the separation to a two-baseline result; a tuned transformer that clears it would confine the separation to the vector-state comparison, which is the comparison the frozen registration designates as the verdict carrier in either case.

**Why:** The defense's recommended experiment, made precise. **Classification: not a submission blocker for the ~Jul 31 arXiv build** once FIX-1 through FIX-4 are applied, because the rescoped paper claims nothing that depends on the transformer datum (the frozen registration carries the WIN on the ablation; either grid outcome leaves the registered verdict standing). **Strongly recommended before the ICLR 2027 submission** (~late Sept; ample time; the defense estimates 1-2 GPU-h): an ICLR reviewer will request it, and either outcome is publishable. The writer has no authority to launch it; this fix is the recorded recommendation for the experimental program, and the §7 prose above is the disclosure path that stands until it runs.

### FIX-6: Rewrite the Conclusion with per-leg substrate scoping
**Severity:** CRITICAL
**File(s):** `papers/flagship/sections/08_conclusion.md`
**Location:** first two sentences (through "...the one construction that worked small.")
**Before:**
> The matrix state of fast-weight sequence models is a representational medium with measurable laws, not an implementation detail of efficient attention. Its recruited dimensionality equals the task's minimal faithful representation dimension, causally in both directions; its first-layer contents carry an episodic-recall capability that parameter-matched vector and compute-matched transformer baselines lack at equal budget, stored nonlinearly and read out only downstream; and the same writes that store also drag the key population toward collapse, more at every scale tested, unmoved by the stock normalization and unrescued at scale by the one construction that worked small.

**After:**
> Matrix-valued state representations are a representational medium with measurable laws, not an implementation detail of efficient attention. In a matrix-state encoder family, recruited dimensionality equals the task's minimal faithful representation dimension, causally in both directions. In a two-layer delta-rule model, the first layer's state carries an episodic-recall capability that a parameter-matched vector-state ablation lacks at equal budget, stored nonlinearly and read out only downstream, with a compute-matched transformer disclosed as a degenerate baseline. And in delta-rule language models, the writes that store also drag the key population toward collapse, more at every scale tested, unmoved by the stock normalization and unrescued at scale by the one construction that worked small.

**Why:** Resolves A2 in the Conclusion (and folds in A1's billing correction). The rank-law substrate has no fast-weight write: `model_v4.py`'s `BindingEncoder` is an `nn.TransformerEncoder` whose docstring states it does NOT hardcode the outer-product solution. "Fast-weight sequence models" cannot own the rank-law clause. This adopts the Introduction's own correct scoping.

### FIX-7: Rewrite the Related Work DeltaNet paragraph and the §5.4 opening sentence with the same scoping
**Severity:** CRITICAL
**File(s):** `papers/flagship/sections/06_related_work.md`; `papers/flagship/sections/05_pathology_at_scale.md`
**Location:** (a) §6 "DeltaNet and gated variants" paragraph, first two sentences; (b) §5.4, first sentence
**Before (a):**
> DeltaNet and Gated DeltaNet (Yang et al.) supply the architecture substrate and benchmark it for quality and throughput. This paper holds the substrate fixed and asks what the states represent: their recruited dimensionality, their causal role in a capability, and the geometry their writes induce at scale.

**After (a):**
> DeltaNet and Gated DeltaNet (Yang et al.) supply the architecture substrate for the capability and pathology legs (Sections 4 and 5) and benchmark it for quality and throughput. This paper asks what the states represent: the rank law of Section 3 is established on a separate matrix-state encoder family with no delta-rule write, while the causal recall capability and the write-induced geometry at scale are established on the delta-rule substrate itself.

**Before (b):**
> Sections 3 and 4 show the delta-rule write is what stores: recruited rank sits at the task minimum in the encoder family, and zeroing the written state deletes the capability in the delta-rule contender.

**After (b):**
> Sections 3 and 4 show the matrix state is what stores: recruited rank sits at the task minimum in the encoder family, and in the delta-rule contender, zeroing the written state deletes the capability.

**Why:** Resolves A2 in Related Work, where the draft names DeltaNet as the substrate for "recruited dimensionality," which is false. Location (b) is an additional instance of the same conflation that neither the attacker nor the defender flagged: §5.4 attributes the encoder-family rank result to "the delta-rule write." All instances of the conflation are now enumerated; after this fix and FIX-6, none remain in the draft.

### FIX-8: Correct Appendix A's rank-tolerance claim to be true per load
**Severity:** SERIOUS
**File(s):** `papers/flagship/sections/09_appendix_a_complement.md`; `papers/flagship/brief.md` (row R9)
**Location:** first paragraph
**Before:**
> the whole-state law holds at 0.5 to 2.9 percent Frobenius residual, with per-example identity alignment $\tau \ge 0.9997$ and effective rank of $Z - c^{*} I$ within 0.3 of the $K{-}1$ target at both tested loads <!-- evidence: R9 -->.

**After:**
> the whole-state law holds at 0.5 to 2.9 percent Frobenius residual, with per-example identity alignment $\tau \ge 0.9997$ <!-- evidence: R9 -->. The effective rank of $Z - c^{*} I$ sits within 0.3 of the $K{-}1$ target at the $K{=}8$ load (deviations 0.03 to 0.26); at the $K{=}12$ load the three converged runs read 10.22 to 10.41 against the target of 11, deviations of 0.59 to 0.78 <!-- evidence: R9 (brief row must be amended with the per-load effective ranks) -->. The rank-$(K{-}1)$ characterization is therefore exact-order at the lighter load and approximate at the heavier one.

**Why:** Resolves A3. Verified directly: K12 effranks 10.415/10.225/10.305, deviations 2 to 2.6 times the claimed tolerance. Per the defense, restricting rather than loosening the band is correct; loosening would read as post-hoc. Appendix A is framed as a mechanism candidate, so the corrected statement costs nothing.

### FIX-9: Correct "37 of 39" to "36 of 39" and name the borderline cell; add the per-cell delta table
**Severity:** SERIOUS
**File(s):** `papers/flagship/sections/03_rank_law.md`; `papers/flagship/sections/10_appendix_b_reproducibility.md` (B.2); `papers/flagship/brief.md` (row R1b)
**Location:** §3.3, the harvest-prediction sentence; B.2's table list
**Before:**
> and the harvest matched this prediction: 37 of 39 force-rank cells sat within 0.07 of the predicted ceiling <!-- evidence: R1b -->.

**After:**
> and the harvest matched this prediction: 36 of 39 force-rank cells sat within 0.07 of the predicted ceiling, one further cell sat marginally outside at 0.072, and the two remaining outliers (0.15 and 0.17, both in one group's below-$d_{\min}$ arm) reflect an additional optimization shortfall <!-- evidence: R1b (brief row must be corrected from 37 to 36; borderline cell S5, $k = d_{\min}{+}1$, seed 0) -->.

**Why:** Resolves A4. Both the attacker and the defender independently recomputed 36/39 with the harvest's own script logic; the third violator is `S5__k_dmin_plus_1__seed0` at |Δ|=0.071571. Additionally instruct the writer to add the 39-cell per-cell |Δ| column to the Appendix B.2 table list (B.2 currently promises the repaired 30-cell grid only; the D-AMBIENT diagnostic table is from the first sweep and must appear so the count is checkable). Process note for the repo, outside the writer's scope: the harvest script's "only outliers" line is a hand-written string literal and the pipeline never computes a within-tolerance count; this belongs in the project's learnings, not the paper.

### FIX-10: Replace §4.5's stale "in flight" with the landed task-2 diagnosis verdict
**Severity:** SERIOUS
**File(s):** `papers/flagship/sections/04_capability_separation.md`; `papers/flagship/brief.md` (new row R11)
**Location:** §4.5, final sentence
**Before:**
> The single bar-clearing seed shows the failure mode is at least partly trainability variance rather than a hard capability boundary; a pre-registered diagnosis round is in flight and nothing beyond single-hop recall is claimed here.

**After:**
> The single bar-clearing seed suggested the failure mode is at least partly trainability variance rather than a hard capability boundary, and the pre-registered diagnosis round confirmed this: across nine pooled seeds, three contender seeds clear the demonstration bar (0.334, 0.391, 0.479) while zero of nine ablation seeds do; the hard-capability-boundary hypothesis is rejected for this task at this scale and budget <!-- evidence: new brief row R11 required (task-2 diagnosis verdict of record, experiment-runs/2026-07-10_h2h_task2diag/results/TASK2DIAG_VERDICT.json, md5 66d2291d8e65932d368d8978bfd16bdc) -->. The pooled paired interval $(-0.020, +0.268)$ sits below the 0.30 margin and the strict-tier reading is a TIE, flagged non-decision-grade by a pre-registered batch-effect gate <!-- evidence: R11 (new row) -->; every bar-clearing seed remains fragile at held-out hop depths and extended horizons. Nothing beyond single-hop recall is claimed here.

**Why:** Resolves A5's first half. The verdict landed after the section files' timestamps but before this review; a paper whose method is traceability cannot describe a landed, archived verdict as in flight. The substance is unchanged (still no task-2 claim), so this is a currency and citation fix. The brief's own "K48 stress table — IN FLIGHT" and "`results/` is empty" passages must be updated in the same pass.

### FIX-11: Replace §7's "in flight" stress-point passage with the landed three-arm K48 table
**Severity:** SERIOUS
**File(s):** `papers/flagship/sections/07_discussion_limitations.md`; `papers/flagship/brief.md` (row R11)
**Location:** §7, "One stress point is incomplete" paragraph
**Before:**
> **One stress point is incomplete.** At the disclosed above-capacity stress load ($K/d = 0.75$), the two recurrent arms read near chance and the transformer arm's fresh training cell is in flight at the time of writing; the three-arm stress table is therefore not claimed, and no number from it appears in this paper.

**After:**
> **One stress point is a locate-only null.** At the disclosed above-capacity stress load ($K/d = 0.75$), the completed three-arm table reads chance in every arm: contender 0.0189, ablation 0.0195, transformer 0.0218, against chance 0.0208 <!-- evidence: R11 (new row; the transformer cell from experiment-runs/2026-07-10_h2h_task2diag/results/transformer_task1_stress_K48_round4.json, md5 14e0c93f56c2a55983f929b7313eb5ac; the recurrent-arm reads per the round-4 re-metric records) -->. The table is locate-only by pre-registration, not a verdict-grade cell; no capability claim is made at this load, and it bounds the single-hop recall separation to loads below this stress point at this training budget.

**Why:** Resolves A5's second half. The fresh transformer K48 cell the draft says it is waiting on has landed (round 4, `fresh: true` in the artifact's provenance). Reporting the completed null is stronger and more accurate than claiming incompleteness.

### FIX-12: Disclose the n=3 / df=2 basis of every paired confidence interval
**Severity:** SERIOUS
**File(s):** `papers/flagship/sections/02_background_setup.md`
**Location:** §2.4, end of the Instruments subsection (a matching sentence may be echoed in Appendix B.4)
**Before:** (no such sentence exists)
**After:**
> All paired confidence intervals in this paper are Student-$t$ intervals on $n{=}3$ per-seed paired differences (two degrees of freedom, $t_{.975} = 4.303$) <!-- evidence: R4 -->. Per-seed values are tabulated in Appendix B so any alternative construction can be applied; the verdicts that rest on these intervals clear their margins by multiples of the interval width, and the one threshold-adjacent reading (Section 5.2) is reported as an unconfirmed trend.

**Why:** Resolves A6 as a disclosure fix. Confirmed against `compute_verdict.py`. On the attacker's bootstrap/permutation request, I adjudicate against adding it: with three paired differences an exact sign-permutation test has eight arrangements and a minimum attainable p of 0.125, and a bootstrap over three points is not informative; a weak crosscheck would invite a sharper statistical attack than the plain disclosure plus per-seed tables. The R7 fragile instance is already reported below its confirmation bar and must not be walked back.

### FIX-13: Name the shuffled-control metric in §4.3
**Severity:** MINOR
**File(s):** `papers/flagship/sections/04_capability_separation.md`
**Location:** §4.3, second paragraph, parenthetical
**Before:**
> ($\mathrm{rf@0.9} = 0.0$ at every state-level tap, with shuffled-control gaps of at most 0.063) <!-- evidence: R5 -->.

**After:**
> ($\mathrm{rf@0.9} = 0.0$ at every state-level tap; the accompanying shuffled-control gaps of at most 0.063 are mean-cosine gaps, a separate continuous sanity check on a different scale from $\mathrm{rf@0.9}$) <!-- evidence: R5 -->.

**Why:** Resolves A7. The number is correct (`gap_vs_shuffled_cos_mean`, max 0.0626) but as written a reader can misattach it to the headline threshold metric.

### FIX-14: Disclose the single per-seed validation-loss exceedance in §5.3
**Severity:** MINOR
**File(s):** `papers/flagship/sections/05_pathology_at_scale.md`
**Location:** §5.3, the validation-loss-neutrality sentence
**Before:**
> Meanwhile the constructions are free at the loss level: validation-loss neutrality passes its blind-pinned ceiling in all eight gate cells <!-- evidence: R8 -->.

**After:**
> Meanwhile the constructions are free at the loss level: validation-loss neutrality passes its blind-pinned ceiling in all eight gate cells under the pre-registered arm-mean criterion; one individual seed in the 98M narrative-mix cell reads 3.2038 against the 3.2020 ceiling while its arm mean (3.1961) passes <!-- evidence: R8 -->.

**Why:** Resolves A8. Verified in `fixscale_harvest_verdict.json`. The mean-based gate is the pre-registered criterion, so the claim stands; the exceedance costs one clause to disclose and would cost credibility if a reviewer found it first.

### FIX-15: Cite the source of the sink-plus-FIFO eviction scheme
**Severity:** MINOR
**File(s):** `papers/flagship/sections/04_capability_separation.md`
**Location:** §4.4, the KV-cap sentence
**Before:**
> caps the transformer's KV cache at $M$ times the contender's state bytes with sink-plus-FIFO eviction, $M \in \{1, 2, 4, 8, 16, 32\}$;

**After:**
> caps the transformer's KV cache at $M$ times the contender's state bytes with sink-plus-FIFO eviction in the style of streaming attention caches (Xiao et al., 2023), $M \in \{1, 2, 4, 8, 16, 32\}$;

**Why:** The defense's finding: the terminology is used without its source. One citation closes it.

---

## 3. The verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 | CRITICAL | CONCEDE + FIX (defense strengthened it: Task-3-only grid, `lr=3e-4` raw, flat curve) | PARTIAL — ATTACK SURVIVES IN REDUCED FORM (rescope + disclosure drops it below CRITICAL; the open scientific question survives as a specified measurement) | FIX-1, FIX-2, FIX-3, FIX-4, FIX-5 |
| A2 | CRITICAL | CONCEDE + FIX | DEFENSE INSUFFICIENT — attack confirmed; fully resolved by rewrite (third instance found in §5.4) | FIX-6, FIX-7 |
| A3 | SERIOUS | CONCEDE + FIX | DEFENSE INSUFFICIENT — attack confirmed; resolved by per-load correction | FIX-8 |
| A4 | SERIOUS | CONCEDE + FIX (defense strengthened it: hand-written outlier string, no computed count) | DEFENSE INSUFFICIENT — attack confirmed; resolved by count correction + per-cell table | FIX-9 |
| A5 | SERIOUS | CONCEDE + FIX | DEFENSE INSUFFICIENT — attack confirmed; resolved by currency update + new brief row R11 | FIX-10, FIX-11 |
| A6 | SERIOUS | PARTIAL | DEFENSE VALID BUT EDIT (numbers sound, R7 already hedged; the df=2 basis must be surfaced; crosscheck request declined with reasons) | FIX-12 |
| A7 | MINOR | PARTIAL | DEFENSE VALID BUT EDIT | FIX-13 |
| A8 | MINOR | PARTIAL | DEFENSE VALID BUT EDIT | FIX-14 |

**Disposition counts:** DEFENSE VALID 0; DEFENSE VALID BUT EDIT 3; DEFENSE INSUFFICIENT (confirmed, resolved by fix) 4; PARTIAL — ATTACK SURVIVES IN REDUCED FORM 1. Per the termination rule: both CRITICALs have fixes whose application drops them below CRITICAL on re-run; **no CRITICAL remains open after the fix list is applied**, and the gauntlet may proceed to the targeted re-run and then the detector gate.

---

## 4. Residual risk after all fixes

1. **A1 residual (SERIOUS; conference-relevant, workshop-survivable).** After the rescope, the paper is accurate and internally consistent, but an MQAR-literate ICLR reviewer can still press: "your only non-degenerate baseline is the ablation; the field's standard recall baseline was never given a fair optimization." The frozen registration answers this formally (the ablation IS the registered verdict carrier), and the disclosure answers it ethically, but the strongest answer is FIX-5's grid. For the arXiv preprint this residual is acceptable; for ICLR 2027 it is the single most likely rebuttal-phase demand, and the ~two-month window before the projected deadline is ample to run it. Either outcome preserves the registered WIN.
2. **Statistical thinness (MINOR-to-SERIOUS at ICLR).** All intervals rest on df=2 even after disclosure. The R4/R10 effects are large enough (CI floors above 0.95 against a 0.30 margin) that no construction choice changes the verdicts, but a reviewer may still ask for more seeds on the capability leg. The disclosed per-seed tables are the mitigation.
3. **R7 gating trend remains unconfirmed.** Already reported below its bar; no fix needed or possible without new seeds. A reviewer may ask for the n that would settle it.
4. **Task-2 batch-effect flag.** FIX-10 discloses a non-decision-grade pooled reading; a careful reviewer may probe the 3-of-9 variance story and the flagged gate. The paper claims nothing on task 2, so exposure is bounded to a robustness question.
5. **Brief-to-prose drift.** Four brief amendments (R0, R1b, R9, R11 new, optionally R4a) are forced by this fix list; if the writer applies prose fixes without amending the brief, the next verify-vs-raws round will re-flag the same rows. The re-run must check the amended brief, not the current one.
6. **Repo process gap (outside the paper).** The harvest analysis script never computes the within-tolerance count it narrates, and the "in flight" staleness arose because sections were drafted hours before an autonomous run landed. Both are worth repo-side learnings entries; neither blocks submission.

---

## 5. Citations

**MUST-CITE (load-bearing for the fixes):**
- Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," arXiv:2312.04927 (2023) — required by FIX-4; the paper's recall task is MQAR-shaped and the divergence must be engaged by name.
- Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers are Better than State Space Models at Copying," arXiv:2402.01032 (2024) — required by FIX-4.
- Olsson et al., "In-context Learning and Induction Heads," Transformer Circuits Thread (2022) — required by FIX-4; two-layer from-scratch transformers developing copy circuitry is the exact expectation the baseline reading contradicts.
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453 (2023) — required by FIX-15; the sink-plus-FIFO scheme is used verbatim in §4.4.

**SHOULD-CITE (strengthening, not load-bearing):**
- Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff" (Based), arXiv:2402.18668 (2024) — the direct competitor line engineering recall into linear attention; strengthens the FIX-4 paragraph.
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752 (2023) — broadens the fixed-size-state family in Related Work beyond the delta-rule line.
- Ramsauer et al., "Hopfield Networks is All You Need," arXiv:2008.02217 (2020) — associative-memory capacity framing for the "representational medium" thesis sentence.

---

**Note to the caller:** this report is returned as message text per the pipeline's instruction; no files were written and no fixes were applied. The writer applies the fix list next, amends the brief rows named above, and the gauntlet re-runs on the affected claims listed in the summary before the detector gate.
