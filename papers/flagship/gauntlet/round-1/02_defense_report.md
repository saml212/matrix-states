# 02_defense_report — gauntlet round 1 (fresh-context defense agent, 2026-07-10)

Verbatim report of the stage-02 defense subagent (dispositions verified
against the raws independently of the attacker).

## Summary for the rebuttal agent

**0 DEFEND, 3 PARTIAL, 5 CONCEDE+FIX.** This attack report was unusually well-verified going in (the attacker independently recomputed most headline numbers and only advanced attacks it could ground in raw artifacts), and my own independent re-verification — recomputing A3's and A4's numbers from scratch against the raw JSONs and the harvest's own analysis script, and reading the raw sweep JSONs and design-doc records for A1 and A5 — confirmed every one of the eight attacks has real substance; none is factually wrong. Two results are new and were **not** in the attack report: (1) for A1, the raw per-cell JSON (`h2h_transformer_task1_sweep_s0.json`) shows the transformer's task-1 (recall) cells trained at the **untuned default `lr=0.0003`**, not the `1e-3` the draft's §4.1 implies — the three-point LR grid was run on **Task 3** (the unrelated LM-quality control task) only, and its winning LR was frozen *only* for Task 3 cells (`HEAD_TO_HEAD_DEMO_DESIGN.md` line 4649, line 4574-4575, line 1529 vs 1531). The transformer's task-1 training-loss curve is also nearly flat from step 500 (7.836) to step 20000 (7.514) — a stalled-optimization signature the draft never reports. This makes A1 *stronger* than as attacked, not weaker. (2) for A4, I recomputed the D-AMBIENT delta using the harvest's own `analyze_sweep_harvest.py` logic (which uses the mean of `convergence_profile[1..8]`, not `mean_cos`) and independently got **36/39**, with the third violator exactly `S5__k_dmin_plus_1__seed0` at `|Δ|=0.071571` — matching the attacker's number exactly, and additionally revealing that the pipeline's own script never actually *computes* a within-tolerance count (the "only outliers" line is a hand-written string literal, not a derived value), so this was never caught by the verification tooling itself. **The most important fixes**: (1) A1 needs an honest correction of the LR-grid claim plus, ideally, a cheap (<1-2 GPU-h) task-1-specific LR grid for the transformer before the headline claim can be fully closed against a hostile MQAR-literature reviewer; (2) A2's substrate-conflation in the Conclusion and Related Work needs the same per-leg scoping the Introduction already uses; (3) A3 and A4 need one-line number corrections. **Submittable after fixes**: yes, with the caveat that A1's framing/disclosure fix is necessary and sufficient to make the paper *honest*, but does not fully *resolve* the underlying scientific question a hostile reviewer will keep pressing — that requires the cheap follow-up experiment, which I recommend running before submission given how inexpensive it is relative to the campaign's GPU posture.

---

## Defenses

### A1: The transformer baseline's chance-level recall failure is unexplained and contradicts the established associative-recall literature

**Disposition:** CONCEDE + FIX (needs new evidence; the framing/disclosure half is a no-cost fix that must happen regardless)

**Response.** The attack is correct, and I found the underlying problem is worse than what the attacker had visibility into. The draft's §4.1 states: "its learning rate was selected on a calibration cell from a three-point grid and frozen at 10⁻³ before the sweep" — written in the paragraph that describes the transformer's role in the recall (Task 1) comparison, which is what carries the headline claim. But the design record and the raw artifacts show this grid was run on **Task 3** (the real-data LM-quality control task), not Task 1. `HEAD_TO_HEAD_DEMO_DESIGN.md` line 1529 ("Wave −1 (C) Task-1 K/d calibration... 3 arms × 1 full cell") shows no LR grid for Task 1 — a single cell per arm, at the default LR. Line 1531 ("Wave −1 (E) Task-3 calibration... Transformer's 3-point LR grid (M3)") shows the grid was Task-3-specific. Line 4574-4575 states the freeze-time override is `transformer_task3_lr = 1e-3` — named specifically for task3. Line 4649 states directly: "all 27 cells at step_count=20000, budget_frac=1.0, lr=3e-4 (transformer task3 at the one freeze-time override 1e-3)". I independently opened the raw training JSON and confirmed this: `experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s0.json` has `"lr": 0.0003` — the untuned default, identical to the contender's and ablation's own LR, never gridded for the transformer on the task that produced the headline chance-level reading. I also pulled the training curve from that same file: `train_loss` is 7.836 at step 500 and 7.514 at step 20000 (min 7.277, essentially flat/noisy across 19,500 steps) — a stalled-optimization signature the draft never reports or discusses. (The ablation shows a similar flat aggregate loss, but its LR-reuse rationale is far more defensible: it shares the contender's embedding/FFN/output head and differs only in the mixer, so inheriting the contender's own "already-validated" LR — per `HEAD_TO_HEAD_DEMO_DESIGN.md` line 1531, "ablation reuses the contender's pinned LR" — is a reasoned choice, not an untested cross-architecture transfer the way it is for a from-scratch softmax-attention model.) Separately, I note the paper's own abstract/introduction give the transformer co-equal headline billing with the ablation ("a parameter-matched vector-state ablation and a compute-matched transformer read chance"), even though the paper's own PI-ratified pre-registration explicitly demotes this baseline: `HEAD_TO_HEAD_DEMO_DESIGN.md` lines 264-265, "FLOP-matched is demoted to a disclosed control ('today's scarcity caveat'), not a primary." The draft's headline framing is inconsistent with its own design record. This is a genuine CRITICAL: the paper currently reports a hedge ("a three-point grid... frozen at 10⁻³") that is not actually true of the task it is attached to, and the raw evidence I found is at least as consistent with a stalled/undertrained baseline as with a genuine architectural inability, exactly as A1 argues from the literature side.

**Supporting evidence.** `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` lines 264-265, 1529, 1531, 4574-4575, 4649; `experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s{0,1,2}.json` (`"lr": 0.0003`, `curve` field showing flat train_loss 7.28-7.84 across the whole run). Draft quote: `04_capability_separation.md` §4.1.

**What goes in the paper if this defense is accepted.** No-cost, must-happen-regardless fixes: (1) correct §4.1 to state plainly that the 3-point LR grid was run on the LM-control task (Task 3), and that Task 1's transformer trained at the untuned default `3e-4`, identical to the contender's own already-validated setting, never independently searched for this arm on this task; (2) consistently downgrade the transformer's billing in the Abstract, Introduction, and Conclusion to match §4.2's already-hedged "degenerate-baseline datum... the separation verdict is carried by the ablation comparison" framing; (3) cite and directly engage Zoology/Based/Repeat-After-Me/induction-heads in §6 or §7, stating the divergence from the literature's expectation plainly and naming the untuned-LR gap as the leading candidate explanation, not yet ruled out. **Recommended new evidence** (cheap): run an LR sweep for the transformer specifically on Task 1 with reported training curves, and either report the corrected number or report that a genuine LR search still reads at chance (which would be a *stronger*, more citable result). This is the only way to fully close the CRITICAL rather than responsibly disclose it — a framing fix alone makes the paper honest but does not resolve the open scientific question the literature comparison raises.

---

### A2: The Conclusion and Related Work sections reintroduce the exact substrate-conflation the paper's own pre-registration flags as unresolved

**Disposition:** CONCEDE + FIX (framing fix only, no new evidence)

**Response.** Confirmed by direct code inspection. `matrix-thinking/chapter2/model_v4.py` lines 25-32 show `BindingEncoder` is an `nn.TransformerEncoder` over a set of bindings with cross-attention row-readers; its own docstring states it "does NOT hardcode the Σ v_j k_j^T outer-product solution" — there is no recurrence, no erase term, no β-gate, nothing resembling a delta-rule write. This is the substrate for R1-R3 (the rank law). Yet `08_conclusion.md` opens "The matrix state of fast-weight sequence models is a representational medium with measurable laws... Its recruited dimensionality equals the task's minimal faithful representation dimension" — one unbroken sentence chain attributing the rank law to "fast-weight sequence models," a term the paper itself reserves for the delta-rule write. `06_related_work.md`'s DeltaNet paragraph repeats the error: "DeltaNet and Gated DeltaNet (Yang et al.) supply the architecture substrate... This paper holds the substrate fixed and asks what the states represent: their recruited dimensionality..." — naming DeltaNet as the substrate for the rank-law leg, which is false. The Introduction and Background sections get this right with explicit per-leg scoping, which is exactly the pattern the Conclusion and Related Work need to adopt.

**Supporting evidence.** `matrix-thinking/chapter2/model_v4.py:25-32`; `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` line 1854; `papers/flagship/sections/08_conclusion.md`; `papers/flagship/sections/06_related_work.md`; `papers/flagship/brief.md` "T1 WORDING FLAG".

**What goes in the paper if this defense is accepted.** Rewrite the Conclusion's first two sentences and the Related Work "DeltaNet and gated variants" paragraph using the same explicit per-leg scoping the Introduction already uses correctly. A bounded, two-paragraph fix.

---

### A3: Appendix A's rank-tolerance claim is false at one of its two stated loads

**Disposition:** CONCEDE + FIX (framing/data-correction fix only, no new evidence)

**Response.** Confirmed by independent recomputation from the raw JSON. `experiment-runs/2026-07-09_zdump_complement/complement_results.json`, `summaries[8..10]` (`run` = `K12_frN_s0_80k`, `s1_80k`, `s2_80k`) give `effrank_Z_minus_cI` = 10.415, 10.225, 10.305 against the K=12 target of K−1=11: deviations 0.585, 0.775, 0.695 — all roughly 2-2.6× the claimed 0.3 tolerance. At K=8, `summaries[4..7]` give deviations 0.256, 0.073, 0.178, 0.031 — inside 0.3, as claimed. The appendix's "within 0.3 of the K−1 target at both tested loads" is false for the K=12 load exactly as attacked.

**Supporting evidence.** `experiment-runs/2026-07-09_zdump_complement/complement_results.json` (`summaries`, `effrank_Z_minus_cI`, runs `K12_frN_s{0,1,2}_80k`). Draft quote: `09_appendix_a_complement.md`.

**What goes in the paper if this defense is accepted.** Restrict the claim to K=8, and separately and honestly report the K=12 deviation (0.585-0.775) rather than loosening the stated tolerance band to retroactively cover the miss (which would read as post-hoc). Since Appendix A is explicitly framed as a mechanism candidate, this is low-stakes to correct honestly.

---

### A4: §3.3's "37 of 39" instrument-defect count does not reproduce; independent recomputation gives 36 of 39

**Disposition:** CONCEDE + FIX (framing/data-correction fix only, no new evidence)

**Response.** Confirmed exactly, including the borderline cell, by re-running the harvest's own analysis logic. `experiment-runs/2026-07-09_capability_sweep_harvest/analyze_sweep_harvest.py` lines 136-149 define the D-AMBIENT delta as `obs = mean(convergence_profile[str(L)] for L in 1..8)` compared to `pred = sqrt(k/D_STATE[group])` — **not** the `mean_cos` field. I reimplemented this exactly against all 39 raw per-cell JSONs and got: `mean|Δ|=0.0276`, `max|Δ|=0.1664` (both match `harvest_analysis_output.txt` exactly), and **36 of 39 within 0.07**, not 37. The three cells outside tolerance are `S5__k_dmin_minus_1__seed0` (|Δ|=0.1537), `S5__k_dmin_minus_1__seed1` (|Δ|=0.1664), and `S5__k_dmin_plus_1__seed0` at `|Δ|=0.071571`, exceeding 0.07 by 0.0016 at full float precision. I additionally traced this back one more level: the analysis script's own printed diagnostic line ("only outliers: S5 k_dmin_minus_1 pair at -0.15/-0.17...") is a **hand-written string literal**, not a computed count — the script never actually computes or stores a "number of cells within 0.07" value anywhere (`harvest_summary.json` only stores `d_ambient_mean_abs_delta`/`d_ambient_max_abs_delta`, both aggregates). So "37 of 39" was never mechanically verified by the pipeline's own tooling; it was eyeballed against a comment that itself named only two of the three violators. This is a genuine finding beyond what the attacker reported, and worth recording as a process gap, not just a prose typo.

**Supporting evidence.** `experiment-runs/2026-07-09_capability_sweep_harvest/analyze_sweep_harvest.py:136-149`; `experiment-runs/2026-07-09_capability_sweep_harvest/results/*.json`; `MANIFEST.md` ("37/39... mean |Δ| 0.028"). Draft quote: `03_rank_law.md` §3.3.

**What goes in the paper if this defense is accepted.** Correct "37 of 39" to "36 of 39" in §3.3 and note the borderline cell explicitly if the per-cell table is published in Appendix B (already promised). Recommend (non-blocking, tooling hygiene) adding an explicit computed `cells_within_tol` field to `analyze_sweep_harvest.py` so this class of miscount can not recur silently.

---

### A5: A landed diagnosis-round verdict exists for the two-hop task and K48 stress cell that the draft still describes as "in flight"/"incomplete"

**Disposition:** CONCEDE + FIX (framing/citation update only, no new evidence; substantive conclusions unchanged)

**Response.** Confirmed directly. `experiment-runs/2026-07-10_h2h_task2diag/results/` is fully populated (`TASK2DIAG_VERDICT.json`, 12 training JSONs, 12 re-metric JSONs, the K48 cell, md5 manifest — 47/47 files present, not empty), and `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.43 (lines 5001-5103) records the landed verdict. I read `TASK2DIAG_VERDICT.json` directly: pooled n=9 task2 analysis gives `"adjudication_sec_1_42_A3": "TRAINABILITY/SEED-VARIANCE CONFIRMED at pooled rate 3/9... the hard-capability-boundary hypothesis is REJECTED for task2 at this scale/budget"`, explicitly flagged `"pooled_reading_decision_grade": false` and `"strict_tier_reading_disclosed_only": "TIE [NON-DECISION-GRADE: batch-effect gate flagged]"` (the ablation-arm variance-ratio gate fired, 6.14 > 4.0 threshold). The K48 three-arm table is complete: contender 0.0189, ablation 0.0195, transformer 0.0218 (fresh, this round), all at/near chance 0.0208 — exactly as the attacker described. The draft's §4.5 and §7 "in flight" language is stale. Substantively nothing changes: the paper's own cautious framing already anticipated a non-decisive outcome.

**Supporting evidence.** `experiment-runs/2026-07-10_h2h_task2diag/results/TASK2DIAG_VERDICT.json`; `experiment-runs/2026-07-10_h2h_task2diag/results/transformer_task1_stress_K48_round4.json` (`acc_A: 0.0218`); `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` §1.43 lines 5001-5103.

**What goes in the paper if this defense is accepted.** Cite the landed §1.42/§1.43 record and `TASK2DIAG_VERDICT.json`, and replace "in flight" language in §4.5 and §7 with the actual outcome: task2 is a high-variance trainability problem for the contender (3/9 seeds clear the bar, 0/9 ablation seeds ever do), non-decision-grade on a flagged batch-effect gate, with all bar-clearing seeds horizon-fragile and held-out-hop-fragile, and the K48 stress table complete with all three arms at chance.

---

### A6: Every confidence interval in the paper is a df=2 (n=3-seed) paired t-interval, and at least one pre-registered decision sits close enough to its threshold for that to matter

**Disposition:** PARTIAL

**Response.** Confirmed mechanically: `experiment-runs/2026-07-10_h2h_sweep_harvest/compute_verdict.py` hard-codes `T975_DF2 = 4.303` and implements `paired_ci` as `t(n-1,.975) * s/sqrt(n)` at `n=3` — exactly as attacked. The paper indeed never states its CIs rest on 3 degrees of freedom as a general methodological point, and never reports a bootstrap/permutation cross-check. That said, the practical stakes are narrow: R4's effect (Δ≈0.966-0.971) clears the 0.30 margin by more than 3× at the CI floor, so no plausible alternative CI construction would change that verdict; the one genuinely fragile instance the attacker flags, R7's gating trend, is *already* reported by the paper with appropriate hedging — the paper is not overclaiming there. I agree this is a real disclosure gap, not a wrong number.

**Supporting evidence.** `experiment-runs/2026-07-10_h2h_sweep_harvest/compute_verdict.py:1-45`. Draft quotes: §4.2, §5.2.

**What goes in the paper if this defense is accepted.** Add one sentence in §2.4 or Appendix B.4 naming n=3/df=2 as the CI basis explicitly. As a cheap addition (pure recomputation from already-archived per-seed numbers, no new GPU compute), add a bootstrap or exact-permutation cross-check for R4's CI in an appendix footnote to demonstrate robustness given the large effect size.

---

### A7: §4.3's "shuffled-control gaps of at most 0.063" is a different metric than the sentence's headline number, which could mislead a skimming reader

**Disposition:** PARTIAL

**Response.** Confirmed factually accurate as literally written, but the clarity concern is real. `experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_contender.json` gives `gap_vs_shuffled_cos_mean` = 0.0600 (`i_s1_qshallow`), 0.0063 (`ii_s0_q0`), 0.0626 (`iii_s1_qtrue`) for the three state-level taps — max 0.0626 ≈ the "0.063" quoted, and `rf@0.9` is indeed exactly 0.0 at all three. The `iv_prelmhead` tap (not state-level) reads a much larger gap (0.80), consistent with R5's own legibility finding. The number is correct; a skimming reader could still conflate it with the headline `rf@0.9` metric as the attacker notes.

**Supporting evidence.** `experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_contender.json` (`tap_variants` fields).

**What goes in the paper if this defense is accepted.** One clause naming the metric explicitly: "(shuffled-control cosine-mean gaps of at most 0.063, a separate continuous sanity check on a different scale from rf@0.9)."

---

### A8: One individual seed in the fix-at-scale wave exceeds its per-seed validation-loss ceiling; the paper's "8/8 gate cells pass" claim is true only under the pre-registered mean-based reading

**Disposition:** PARTIAL

**Response.** Confirmed exactly. `experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json`, `scales/98m/corpora/wikitext-mix-ext/val_loss_gate`: `per_seed_over_ceiling[0] = 3.2037888...` against `pass_ceiling = 3.2020253...`, while `per_token_mean = 3.1960760...` and `PASS_by_arm_mean = true`. The mean-based gate is the pre-registered criterion (per `pins/BANDS_PINNED-FrozenBias-98M.json`) and the paper's claim is technically correct under it; the individual-seed exceedance is real and undisclosed.

**Supporting evidence.** `experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json` (`scales.98m.corpora.wikitext-mix-ext.val_loss_gate`).

**What goes in the paper if this defense is accepted.** A one-clause footnote noting the pre-registered gate is mean-based and one seed's individual reading (3.2038 vs ceiling 3.2020) sits marginally above it.

---

## New citations found during defense

- **Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM), arXiv:2309.17453 (2023).** The draft's §4.4 uses "sink-plus-FIFO eviction" terminology for the KV-capped transformer baseline without citing its source; a one-line citation closes this.
- All six citations proposed by the attack report (Zoology, Based, Repeat After Me, Olsson et al., Mamba, Hopfield Networks) are appropriate and independently confirmed on point for A1's literature gap.

## Attack ordering note

A1 is rated CRITICAL by the attacker and I agree — but the attacker under-weighted how strong the evidence for it actually is: once you pull `h2h_transformer_task1_sweep_s0.json` and see `lr: 0.0003` plus a flat training-loss curve, and cross-reference the design doc's own line 4574 confirming the LR grid was Task-3-only, the "unexplained anomaly" framing understates it: the paper's own hedge sentence is describing the wrong task. This is the single highest-priority fix in the whole gauntlet. A2 is correctly CRITICAL — confirmed by direct code read, cheap to fix. A4's SERIOUS rating is right, with the added note that the miscount traces to a verification-tooling gap. A6 is borderline MINOR/SERIOUS since its one materially fragile instance (R7) is already honestly hedged — no pushback if the rebuttal downgrades it. A5 is correctly rated but the lowest-stakes SERIOUS here. A3/A7/A8 severities look right.

**Weaknesses the attacker missed:** (1) the abstract/introduction give the transformer co-equal headline billing with the ablation despite the program's own pre-registration explicitly demoting it to "a disclosed control... not a primary" — an internal-consistency defect independent of the literature argument; (2) the D-AMBIENT verification script computing aggregate deltas but never a pass/fail count is a reusable-tooling gap worth a [LEARN]-style note in the project's records, separate from the paper fix.

**Security note.** No embedded or adversarial instructions were found in any file, JSON artifact, or script read during this defense.
