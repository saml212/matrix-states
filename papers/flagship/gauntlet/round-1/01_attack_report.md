# 01_attack_report — gauntlet round 1 (fresh-context attack agent, 2026-07-10)

Verbatim report of the stage-01 attack subagent; verify-vs-raws performed
against the experiment-runs/ artifacts named in brief.md.

## Summary for the defense agent

**Verdict: moderate-to-strong revision required, but largely salvageable.** The empirical pipeline is unusually clean for a paper this dense: I independently recomputed essentially every headline number across all ten evidence rows (R1–R10) from the raw JSON artifacts named in the brief, and the overwhelming majority matched exactly — including the hardest numbers to get right (the paired t(2,.975) CIs in R4, the exact-permutation Spearman null in R1, the sigma-normalized mitigation deltas in R7, all eight md5 checksums I sampled across every row). That discipline is real and should be defended vigorously. The paper has two genuine CRITICAL problems, however, neither of which is a numbers-don't-match problem. First, Finding 2's marquee "capability separation" rests on a compute-matched transformer reading at-or-below chance on a small, short-context, from-scratch associative-recall task — a result that directly contradicts a substantial, well-known body of prior work (MQAR/Zoology, Based, Repeat-After-Me, induction heads) showing transformers are *unusually good*, not bad, at exactly this task family. The paper never engages this literature or rules out a mistrained/undertuned baseline as the explanation, even though it structurally could (training-accuracy diagnostics, broader LR search). Second, the paper's own working brief pre-flagged an unresolved wording risk — attributing the rank law (established on a non-delta-rule transformer-encoder substrate) to "delta-rule"/"fast-weight" language — and while the Introduction scopes this correctly, both the Conclusion and the Related Work section reintroduce the exact conflation the brief warned against, in language that is provably wrong (I confirmed via the actual model code that the rank-law substrate has no outer-product write at all). Both are bounded, addressable fixes — not evidence of a broken experiment — but as written they are real defects a hostile reviewer would not let pass. Below that tier are two concrete, verified small-numeric errors (an appendix tolerance claim that's false at one of its two stated loads, and an instrument-defect count that's off by one), a currency issue (a diagnosis round the paper calls "in flight" has actually landed), and a fair-but-worth-flagging statistical thinness note. Nothing here threatens R1–R8's causal/observational core as verified against the raws.

---

## Attacks

### A1: The transformer baseline's chance-level recall failure is unexplained and contradicts the established associative-recall literature

**Severity:** CRITICAL

**Type:** alternative-explanation / baseline-adequacy / missing-citation

**Attack.** Finding 2's headline claim — "a two-layer delta-rule model performs episodic recall at accuracy 0.9995 while a parameter-matched vector-state ablation and a compute-matched transformer read chance" (abstract) — leans on a transformer that reads **0.02710, 0.02930, 0.02856** across its three seeds (§4.2 Table 1), which is at or *below* the task's own naive chance rate of 0.03125 in two of three seeds. This transformer is "a two-layer pre-norm decoder with rotary position embeddings," FLOP-matched within 5%, trained 20,000 steps with full (non-causally-bottlenecked) attention over a mere 224-token bind phase, on a task that is structurally a textbook multi-query associative-recall (MQAR) problem: bind K=32 key→value token pairs, then retrieve the value for one queried key. This is precisely the regime where the field's best-known empirical result runs the *opposite* direction: Arora et al.'s "Zoology" paper (arXiv:2312.04927) introduces MQAR and shows attention essentially solves it while sub-quadratic/linear-attention models are the ones that struggle, with the gap growing with the number of KV pairs; the follow-up "Based" paper (arXiv:2402.18668) is built around closing exactly that gap *in favor of* attention; Jelassi et al., "Repeat After Me" (arXiv:2402.01032) shows transformers beat state-space models at copying/recall specifically; and Olsson et al.'s induction-heads work establishes that 2-layer transformers reliably develop copy/recall circuitry from scratch, early in training, on far less favorable setups than this one. A 2-layer, FLOP-matched, fully-attending transformer reading *at or below chance* on 32-way recall over 224 tokens is the surprising result in this literature, not the expected one — yet the paper treats it as an unremarkable confirmation of the thesis rather than a result demanding its own explanation. The paper's only hedge is a thin one: "its learning rate was selected on a calibration cell from a three-point grid and frozen at 10⁻³ before the sweep" (§4.1) — a three-point LR grid is a weak search for a from-scratch architecture being asked to carry a comparison this load-bearing, and no training-set/in-distribution accuracy, loss curve, or architecture sanity check is reported to rule out a simple undertraining or miscalibration explanation. This same transformer failure is also the entire evidentiary basis for R10's "constant-memory" claim (§4.4: "the pre-registered degenerate-baseline clause fires... no crossover-point value is certified") — so the same unexplained anomaly propagates into a second headline finding.

**Supporting evidence.** Arora, Sabri, Roberts, et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," arXiv:2312.04927. Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff" (Based), arXiv:2402.18668. Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers are Better than State Space Models at Copying," arXiv:2402.01032. Olsson et al., "In-context Learning and Induction Heads," Transformer Circuits Thread (2022). Draft quotes: §4.1 ("its learning rate was selected on a calibration cell from a three-point grid"); §4.2 Table 1 (transformer per-seed 0.02710/0.02930/0.02856 vs chance 0.03125); §7 ("a differently scaled or differently trained transformer requires its own matched re-run before any extrapolation" — the paper's own admission the comparison is fragile, immediately followed by using it as a headline number anyway).

**What the paper would need to do to defuse this.** Engage the MQAR/induction-head literature by name in §6 and explain the divergence; report the transformer's in-distribution/training accuracy and loss curve to positively rule out a training-time bug or LR miscalibration; broaden the LR search (or justify why three points suffice) and disclose the full grid; and downgrade the transformer's role in the abstract/intro to match the hedged role it already has in §4.2 ("the separation verdict is carried by the ablation comparison... the transformer's non-competitiveness disclosed alongside") rather than presenting it with equal billing to the ablation in the headline sentences.

---

### A2: The Conclusion and Related Work sections reintroduce the exact substrate-conflation the paper's own pre-registration flags as unresolved

**Severity:** CRITICAL

**Type:** claim-scope

**Attack.** The paper's brief (`papers/flagship/brief.md`, "T1 WORDING FLAG") explicitly documents an unresolved risk: the rank-law finding (R1–R3) is established on the **matrix-state encoder family** (`model_v4.py`'s `BindingEncoder`, confirmed by direct code inspection to be an `nn.TransformerEncoder` over a *set* of K bindings followed by cross-attention read-out queries — no recurrence, no erase term, no outer-product write, no β-gate of any kind), while capability (R4/R5) and pathology (R6–R8) are established on the **delta-rule family**, which does use the outer-product update given in §2.1. The design doc itself states this in so many words: "Stage 1 uses the bespoke `BindingEncoder` architecture throughout, not `fla`/DeltaNet." The Introduction (§1) and Background (§2.2) get this right, scoping each finding to its own architecture explicitly. But two other sections do not: **the Conclusion (§8)** opens "The matrix state of fast-weight sequence models is a representational medium with measurable laws" and then, in one unbroken sentence chain with no re-scoping, attributes to that single subject the rank law ("its recruited dimensionality equals the task's minimal faithful representation dimension"), the capability ("its first-layer contents carry an episodic-recall capability"), and the pathology ("the same writes that store also drag the key population toward collapse") — using "fast-weight," a term the paper itself reserves for the delta-rule write mechanism, as the grammatical owner of a finding (the rank law) whose substrate has no fast-weight write at all. **Related Work §6** independently repeats the error: "DeltaNet and Gated DeltaNet (Yang et al.) supply the architecture substrate and benchmark it for quality and throughput. This paper holds the substrate fixed and asks what the states represent: their recruited dimensionality, their causal role in a capability, and the geometry their writes induce at scale" — explicitly naming DeltaNet as the substrate for "recruited dimensionality," which is false. This is not a hypothetical risk the brief anticipated and the draft avoided; it is the brief's flagged failure mode, realized twice, in the two sections most likely to be read in isolation (a skimming reviewer reads abstract + conclusion; a citation-checking reviewer reads related work first).

**Supporting evidence.** `papers/flagship/brief.md`, "T1 WORDING FLAG" section. `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` ("Stage 1 uses the bespoke BindingEncoder architecture throughout, not fla/DeltaNet"). `matrix-thinking/chapter2/model_v4.py`, class `BindingEncoder` — docstring explicitly states it does "NOT hardcode the Σ v_j k_j^T outer-product solution." Draft quotes: `08_conclusion.md`; `06_related_work.md` ("This paper holds the substrate fixed and asks... their recruited dimensionality").

**What the paper would need to do to defuse this.** Rewrite the Conclusion's first two sentences and the "DeltaNet and gated variants" paragraph of Related Work using the same explicit per-leg scoping the Introduction already uses correctly ("matrix-valued state representations... In a matrix-state encoder family... In a two-layer delta-rule model..."). This is a bounded, two-paragraph fix, not a re-analysis.

---

### A3: Appendix A's rank-tolerance claim is false at one of its two stated loads

**Severity:** SERIOUS

**Type:** number-provenance / correctness

**Attack.** §9 (Appendix A) claims: "effective rank of $Z - c^*I$ within 0.3 of the $K{-}1$ target at both tested loads." Recomputing directly from `experiment-runs/2026-07-09_zdump_complement/complement_results.json`: at K=8 (target K−1=7), deviations are 0.031–0.256 — inside 0.3, as claimed. At K=12 (target K−1=11), the three converged runs read effective ranks 10.415, 10.225, 10.305, i.e. deviations of **0.585, 0.695, 0.775** — roughly 2–2.6× the claimed 0.3 tolerance. "At both tested loads" is factually incorrect for the K=12 load as written.

**Supporting evidence.** `experiment-runs/2026-07-09_zdump_complement/complement_results.json` (`effrank_Z_minus_cI` field, K12_frN_s0/s1/s2_80k runs). Draft quote: `09_appendix_a_complement.md`.

**What the paper would need to do to defuse this.** Either restrict the claim to K=8 and separately report the K=12 deviation honestly (0.59–0.78), or revise the stated tolerance band upward to cover both loads (~0.8) and update the accompanying prose about what "within tolerance" means at each load.

---

### A4: §3.3's "37 of 39" instrument-defect count does not reproduce; independent recomputation gives 36 of 39

**Severity:** SERIOUS

**Type:** number-provenance

**Attack.** §3.3 states: "the harvest matched this prediction: 37 of 39 force-rank cells sat within 0.07 of the predicted ceiling." Independently recomputing per-cell |Δ| = |observed cosine − √(k/d_state)| from the 39 raw per-cell JSONs in `2026-07-09_capability_sweep_harvest/` using the paper's own stated formula finds only **36 of 39** within 0.07: a third cell (`S5__k_dmin_plus_1__seed0`) reads |Δ| = 0.071571, exceeding the 0.07 threshold by 0.0016 — a razor-thin miss, but a miss at full float precision, not a rounding artifact. This is exactly the class of tolerance-boundary sensitivity this codebase's own accumulated lessons flag as a recurring failure mode (see CLAUDE.md: "Integer/structural correctness checks... need EXACT thresholds. A numerical-tolerance slack... silently defeats single-instance violations").

**Supporting evidence.** `experiment-runs/2026-07-09_capability_sweep_harvest/` per-cell force-rank JSONs (`d_ambient_mean_abs_delta`/`d_ambient_max_abs_delta` fields, `MANIFEST.md`'s own "37/39... mean |Δ| 0.028" prose). Draft quote: `03_rank_law.md`.

**What the paper would need to do to defuse this.** Recompute the count directly with the paper's own pipeline (not a reviewer's reconstruction), publish the per-cell |Δ| table in Appendix B (already promised as the "30-cell force-rank grid" — add the delta column), and correct the prose count and/or explicitly name the borderline cell.

---

### A5: A landed diagnosis-round verdict exists for the two-hop task and K48 stress cell that the draft still describes as "in flight"/"incomplete"

**Severity:** SERIOUS

**Type:** reproducibility / currency

**Attack.** §4.5 states the two-hop diagnosis round "is in flight," and §7 states "the transformer arm's fresh training cell is in flight at the time of writing; the three-arm stress table is therefore not claimed." As of this review, `experiment-runs/2026-07-10_h2h_task2diag/` is fully populated (not empty, contra the brief's own note that it "is empty" at brief-authoring time) and contains a completed verdict record: `TASK2DIAG_VERDICT.json` reports a full n=9 pooled task2 analysis ("TRAINABILITY/SEED-VARIANCE CONFIRMED at pooled rate 3/9... hard-capability-boundary hypothesis is REJECTED for task2 at this scale/budget," flagged non-decision-grade on a batch-effect gate) and a completed K48 three-arm locate-only table (contender 0.0189 / ablation 0.0195 / transformer 0.0218, all ≈chance 0.0208 — the fresh transformer read the §7 text is explicitly waiting on). The design doc's own `MANIFEST.md` records this as §1.42 (pre-run) / §1.43 (verdict of record), launched 2026-07-10 17:55:54Z and completed autonomously — after the draft section files' own modification timestamps (~16:25–16:48 the same day), which explains but does not excuse the staleness at a review checkpoint. This does not change the paper's substantive conclusions (the landed reads are null/non-decisive, consistent with "nothing beyond single-hop recall is claimed"), but a paper whose entire methodology is "every numerical claim traces to... an archived, checksummed raw artifact" (§1) should not describe as "in flight" something that has, by the time a reviewer checks, already landed and sits uncited in the archive it names.

**Supporting evidence.** `experiment-runs/2026-07-10_h2h_task2diag/MANIFEST.md`, `results/TASK2DIAG_VERDICT.json`, `results/transformer_task1_stress_K48_round4.json`. Draft quotes: `04_capability_separation.md` §4.5 ("a pre-registered diagnosis round is in flight"); `07_discussion_limitations.md` ("the transformer arm's fresh training cell is in flight at the time of writing").

**What the paper would need to do to defuse this.** Cite the landed §1.42/§1.43 record and its artifact, and update the two passages to report the actual (still non-decisive) outcome rather than "in flight."

---

### A6: Every confidence interval in the paper is a df=2 (n=3-seed) paired t-interval, and at least one pre-registered decision sits close enough to its threshold for that to matter

**Severity:** SERIOUS

**Type:** statistical

**Attack.** R4, R7, and R8's intervals all use t(2, 0.975)=4.303 paired CIs from n=3 seeds. For R4 this is inconsequential — the effect (Δ≈0.97) is so far from the 0.30 margin (CI floor 0.958) that no plausible alternative CI-construction method would overturn it. But R7's gating read (+4.312, "1.92σ," against a pre-registered 2σ=4.489 bar, Welch p=0.062) sits close enough to its own threshold that small-sample fragility is material: with n=3, a single additional seed could plausibly move this over or under the bar in either direction. The paper already handles this specific case with appropriate honesty ("a direction-consistent trend toward amplification: not a confirmed effect, and not a null" — this is good practice and should be defended, not walked back), but the paper nowhere states, as a general methodological point, that its CIs are constructed with only 3 degrees of freedom, nor does it report any nonparametric/bootstrap cross-check anywhere as a robustness sanity check on the construction itself.

**Supporting evidence.** `compute_verdict.py` (confirmed via independent agent recomputation to implement exactly paired-difference / t(2,.975) / sd·t/√3); draft quotes: §4.2 ("95 percent confidence interval"), §5.2 ("$1.92\sigma$, below the pre-registered $2\sigma$ confirmation bar of 4.489").

**What the paper would need to do to defuse this.** Add one sentence in §2.4 or Appendix B.4 naming n=3/df=2 as the CI basis explicitly, and add a bootstrap or exact-permutation cross-check for R4's CI (cheap, given the enormous effect size, and would simply confirm robustness) as an appendix footnote.

---

### A7: §4.3's "shuffled-control gaps of at most 0.063" is a different metric than the sentence's headline number, which could mislead a skimming reader

**Severity:** MINOR

**Type:** claim-scope / clarity

**Attack.** §4.3 states: "Ridge probes at three state-level taps... recover nothing at the strict threshold ($\mathrm{rf@0.9} = 0.0$ at every state-level tap, with shuffled-control gaps of at most 0.063)." Verified against `tap_localization_contender.json`: the 0.063 figure is the `gap_vs_shuffled_cos_mean` field (a raw cosine-mean gap, max 0.0626 across the three state-level taps), not an rf@0.9-based gap — which would trivially be 0 since rf@0.9 is exactly 0.0 in every case. As written, a reader could mistake the 0.063 as bounding how close the *headline* metric (rf@0.9) came to the 0.9 threshold, when it is in fact an unrelated, always-present sanity-check metric on a different (continuous cosine) scale.

**Supporting evidence.** `experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_contender.json` (`gap_vs_shuffled_cos_mean` field).

**What the paper would need to do to defuse this.** One clause naming the metric explicitly, e.g. "(shuffled-control cosine-mean gaps of at most 0.063, a separate continuous sanity check)."

---

### A8: One individual seed in the fix-at-scale wave exceeds its per-seed validation-loss ceiling; the paper's "8/8 gate cells pass" claim is true only under the pre-registered mean-based reading

**Severity:** MINOR

**Type:** reproducibility / disclosure

**Attack.** §5.3 states validation-loss neutrality "passes its blind-pinned ceiling in all eight gate cells." Verified against `fixscale_harvest_verdict.json`: this is true under the pre-registered arm-mean gate (`PASS_by_arm_mean`), but the 98M wikitext per_token arm has one individual seed (`per_seed_over_ceiling: [3.2038]`) exceeding the per-seed ceiling (3.2020) while the arm mean still clears it. This is defensible — the pre-registration (`pins/BANDS_PINNED-FrozenBias-98M.json`) specifies a mean-based, not per-seed, criterion — but the individual-seed exceedance is not disclosed anywhere in the draft.

**Supporting evidence.** `experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json` (`per_seed_over_ceiling` field, 98M wikitext per_token arm); `pins/BANDS_PINNED-FrozenBias-98M.json`.

**What the paper would need to do to defuse this.** A one-clause footnote noting the pre-registered gate is mean-based and one seed's individual reading sits marginally above ceiling.

---

## Attacks I considered but decided were weak

1. **Generic implicit low-rank bias as an alternative to the rank law.** SGD is known to have a generic bias toward low-rank solutions independent of task content (implicit regularization literature). But this is well-preempted by the paper's own causal test: a generic bias predicts smooth degradation as forced rank drops, not the observed step function (exactly 0.000 at $d_{\min}-1$, clean recovery at $d_{\min}$, in all 5 groups and all 4 extension seeds). Rejected.

2. **Span-fraction cross-width confound across the parameter ladder.** The ladder's d_state changes (64 for 14M/98M, 128 for 392M/1.31B), and the random-vector anchor used to normalize span fraction is itself d-dependent, raising a question of whether part of the "monotone climb" is an instrument artifact of widening state rather than real geometric collapse. Rejected as a live attack because the paper's own instrument (§2.4) is explicitly designed and per-width-calibrated for cross-width comparability (two analytic anchors computed at each state's own width), and the observed trend (0.248→0.455) is far larger than any plausible anchor-shift magnitude from the width change alone.

3. **Nichani capacity bound "explaining away" the recall win.** Since d_state=64 > K=32, Nichani et al.'s result (rank-1 matrix supports ~d associations under argmax) could in principle make the contender's win unsurprising. Rejected: the paper already explicitly disclaims any rank/capacity interpretation of acc_A (the caveat travels with every number), frames the win as a *capability* not a *capacity* result, and Nichani's bound is specific to matrix states — it doesn't bound why the (non-matrix, vector-recurrence) ablation fails at chance, which is the actual comparison the WIN verdict is carried by.

4. **Multiple-comparisons correction across ~10 pre-registered evidence rows.** With this many pre-registered tests, a family-wise error-rate concern is a standard reviewer reflex. Rejected as weak: each row tests a genuinely distinct scientific question (rank law, capability, pathology) with its own independently pre-registered margin, not repeated looks at one hypothesis — standard practice does not require a Bonferroni-style correction across unrelated pre-registered claims, and the one borderline result (R7 gating) is already honestly reported as unconfirmed rather than claimed.

5. **Unequal per-group seed counts in the rank-law sweep (S3/S5/A6=3 seeds vs S4/A5=5 seeds).** Initially looked like an undisclosed asymmetry. Rejected: S4/A5 is the marquee designed dissociation pair carrying the TOST equivalence test, so allocating more seeds to it for statistical power is a defensible, if implicit, design choice, and the totals (19) and per-group means both check out exactly against the raw artifacts.

---

## New citations that should be in Related Work

- **Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," arXiv:2312.04927 (2023).** Introduces the multi-query associative-recall (MQAR) benchmark this paper's Section 4 task closely resembles; documents that attention closes recall gaps sub-quadratic models leave open — the paper's transformer result runs opposite to this literature's central finding and must engage it directly.
- **Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff" (Based), arXiv:2402.18668 (2024).** Direct competitor work engineering linear-attention/hybrid architectures specifically to close the recall gap Zoology documents; highly relevant to how this paper frames its own recall separation.
- **Jelassi, Brandfonbrener, Kakade, Malach, "Repeat After Me: Transformers are Better than State Space Models at Copying," arXiv:2402.01032 (2024).** Directly on point for explaining (or contesting) why this paper's transformer baseline fails a copy/recall-shaped task.
- **Olsson et al., "In-context Learning and Induction Heads," Transformer Circuits Thread (2022).** Establishes that even small, from-scratch, 2-layer transformers reliably develop induction/copy circuitry — raises the bar for why this paper's transformer reads at chance.
- **Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752 (2023).** The related-work section currently discusses only delta-rule/DeltaNet-family linear attention; the broader fixed-size-state model family (SSMs) is otherwise entirely absent and would strengthen the "matrix state as representational medium" framing's generality claim.
- **Ramsauer et al., "Hopfield Networks is All You Need," arXiv:2008.02217 (2020).** Associative-memory capacity framing directly relevant to treating a fixed-size state as "a representational medium with its own capacity structure" (the paper's own Introduction language).

---

**Security note.** The attack agent reported no embedded or adversarial instructions inside the paper's sections, figure-gen.py, or any raw artifact read during verification.
