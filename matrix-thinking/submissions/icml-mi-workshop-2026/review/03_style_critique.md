# Style critique

## Summary for edit agent

The paper is already quite close to venue style. It uses impersonal "we" (the plural-investigator "we"), numbered contribution bullets, quantified claims, and MI-workshop-typical framing (mechanism claim + falsification via positive controls). The math, notation, and proposition structure are appropriately formal. The biggest structural asset is that contributions are already quantified (not just "we propose").

The main issues are (1) a handful of first-person-about-author / narrative-process phrasings that leak into the prose ("We report a negative result with a mechanism", "our original Jacobian hypothesis", "Taken together these results suggest", "A reader who has accepted the readout Jacobian argument might reasonably ask", "What would move us"); (2) a meta commentary note at the top of main.tex that must be removed before submission; (3) the abstract is ~288 words and should be trimmed to 200-230; (4) a few spots where the paper slips into workshop-blog voice ("the cleanest single piece of evidence", "the paper's sharpest claim", "Our original Jacobian hypothesis predicted"); (5) some banned words (`actually`, `really`, `just`, `clearly`, `obviously`, `interestingly`, `nicely`) are absent, but `essentially`, `wildly`, `parsimonious`, and em-dash-as-conversational-pause appear frequently enough to flag.

No section requires a full rewrite. Everything is line-edit-level. The priorities in order are: (a) remove the meta-TODO comment in main.tex; (b) strip the handful of narrative-process phrasings (listed below) — this is the user's top priority; (c) trim the abstract; (d) tighten em-dash and hedge-word usage; (e) make figure captions fully self-contained and remove "will be added in the appendix when the re-runs complete" from §4. **Overall readiness: needs polish, not overhaul.**

## Reference papers surveyed

1. **Rizvi-Martel et al., "The Illusion of Superposition?" (arXiv 2604.06374)** — the direct-adjacency paper. Uses declarative "we investigate / we find / we argue" with minimal narrative. Abstract is a tight 220 words, problem → setup (3 regimes) → findings → mechanism → "Together, our results offer a unified explanation." No meta-comments about author process. Hedged conclusions ("suggests that", "consistent with"). No "we were surprised" or "our original hypothesis." The submitted paper should aim to match this paper in voice exactly since it is the closest stylistic neighbor.

2. **Nanda et al., "Progress measures for grokking" (arXiv 2301.05217)** — canonical MI workshop-to-ICLR style. Contributions phrased as discoveries ("we reverse engineer … and find that …"), not proposals. Passive voice for mechanisms ("the algorithm implemented by …"), active "we" for findings. No self-disclosure of research process.

3. **Shaham et al., "Optimal Ablation for Interpretability" (arXiv 2409.09951)** — 150-word abstract in the problem-method-advantage-applications mold. Figure captions are fully self-contained 1–2 sentences that name the subtask, the ablation method, and what each axis shows. No captions assume the reader has read the prose. This is the figure-caption bar.

4. **Dong, Cordonnier & Loukas 2021 "Attention is not all you need"** — cited in the paper; useful as a style reference because it is a mechanism + falsification paper. Abstract gives the claim, the mechanism, and the empirical demonstration, in that order, in ~180 words. No "we were surprised." No "our original hypothesis."

5. **ICML 2024 MI Workshop accepted-paper titles** (e.g., "Hypothesis Testing the Circuit Hypothesis in LLMs", "InversionView", "Measuring Progress in Dictionary Learning …") — the venue's section-title convention is descriptive-of-claim ("Hypothesis testing X", "Measuring Y") rather than descriptive-of-result-shape. Section titles in the submitted paper that are descriptive-of-result-shape ("The Flatten-Then-Project Readout is Rank-Blind", "Depth and Scale Do Not Rescue Matrix-CODI") are slightly punchier than the venue norm but are not out-of-range; they just tilt toward a workshop-paper voice rather than a main-conference voice. Acceptable for the MI workshop.

## Section-by-section critique

### main.tex preamble and title

**Critical issue:** Lines 2–17 are a `% NOTE TO THE AUTHOR` block with build instructions. This must be deleted before submission — it is both a reviewer-visible tell of author identity (references to swapping in the ICML sty file, a build command) if left in a comment-reveal scenario, and is unprofessional in a final submission. Remove in full.

**Title (line 46–48):** "The Gradient Does Not See Rank: A Structural Explanation for the Illusion of Superposition in Latent Chain-of-Thought." Good — declarative, claim-forward, references the adjacent paper. Slight concern: "Does Not See" is slightly anthropomorphic for a gradient. A minor alternative is "The Gradient Is Blind to Rank," which matches the paper's own terminology ("rank-blind") and is slightly more formal. Not a must-fix.

**Author block (line 50–52):** `\anonymous{} Authors \\ \anonymous{} Institution`. OK for submission; ICML sty file will handle this.

### Abstract

**Voice:** OK (uses plural investigator "we"), with two phrasings worth flagging.

**Structure:** Roughly follows context → observable (rank) → hypothesis → finding → falsification → positioning. Missing a crisp implication sentence at the end; instead ends on "our positive controls specifically test," which is a method statement rather than an implication.

**Length:** ~288 words. Target is 200–230. Should trim ~60–80 words.

**Specific edits needed:**

- Line 68: "We report a negative result with a mechanism." → "This paper reports a negative result accompanied by a mechanism." *(removes the self-disclosing "we report"; this is the sort of phrase that reads as personal narrative.)* Alternative: just delete this sentence and let the claims speak — it functions as editorial voice rather than content.
- Line 69–70: "Across four training regimes of a matrix-CODI model on ProsQA, the rank-$k$ projection ablation curve is flat to within 0.6 percentage points." → fine. Keep.
- Line 71: "A three-seed replication shows accuracy tight at $81.5\pm 1.2$pp while the final effective rank of $\Z$ ranges over $4, 12, 13$ --- the loss does not reward any particular rank." The em-dash glosses a causal claim. Change to a period: "… ranges over $4, 12, 13$. The loss does not reward any particular rank."
- Line 73–74: "We then construct four readout variants designed to falsify the hypothesis that \emph{rank-blindness comes from the flatten-then-project readout}:" → keep, but change "We then construct" to "To test whether rank-blindness comes from the flatten-then-project readout alone, we construct four readout variants:" (removes the sequential-narrative "then").
- Line 80: "The readout Jacobian is not the sole culprit: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule." → fine. Keep.
- Line 84: "A linear probe on the matrix thought also underperforms the raw pretrained hidden state at predicting the target (AUC $0.673$ vs $0.846$)." → fine.
- Line 85–88: "We position the finding against two February 2026 papers that measure rank in linear-attention hidden states: those papers describe rank as a structural invariant of trained states, while we make a falsifiable mechanism claim about the objective that our positive controls specifically test." → The last clause is method, not result. Replace with an implication sentence: "Rank-$k$ ablation should not be used as a probe for superposition in CODI-trained matrix latents." This also gives the abstract a proper landing.

### §1 Introduction

**Voice:** Mostly OK; one narrative-process phrasing.

**Specific edits:**

- Line 30–32: "We report four pieces of evidence that this probe is uninformative in practice and give a mechanism that explains why." → acceptable. Minor: "report" implies the act of reporting; consider "This paper presents four pieces of evidence …" or just "Four pieces of evidence are reported below, together with a mechanism that explains why." Either is fine.
- Line 61–67: "We explicitly position against two February 2026 papers …" The word "explicitly" is a hedge-word flag but is OK here because it is doing disambiguating work. Keep.
- Line 69–72: "Taken together these results suggest that rank-$k$ ablation is not a valid measure …" The phrase "Taken together" is fine in intros but appears again in §3.4 closing; consider varying one of the two instances.
- Contribution bullet 3, line 44–52: "The simple Jacobian-linearity story is not the full picture" — "the simple … story" is a slightly colloquial framing. Replace with: "Readout linearity is a sufficient but not necessary condition: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule." This is more formal and matches the wording used later in §5.3.
- Contribution bullet 4, line 55–58: "The matrix bottleneck actively loses target-predictive information relative to the uncompressed hidden state." — "actively" is a mild overclaim flag; the bottleneck does not "actively" do anything, it just does. Change to "The matrix bottleneck loses target-predictive information relative to the uncompressed hidden state."

### §2 Background

**Voice:** Clean. Paragraph-style `\paragraph{}` headings are venue-appropriate.

**Specific edits:**

- Line 17–18: "We fork CODI and insert a \emph{matrix bottleneck} on the latent feedback path." — "We fork CODI" is informal. Change to: "We extend CODI with a \emph{matrix bottleneck} on the latent feedback path." "Fork" is a repo-operations word, not a methods word.
- Line 40: "a smooth proxy used in the spectral analysis literature." → missing citation. Effective rank (entropy of normalized spectrum) is from Roy & Vetterli 2007 or Del Giudice 2021; add one reference. Otherwise reviewers will ding this.
- Line 54–59: "Throughout the paper ``rank'' means \emph{effective rank} … We are not claiming that the dense-matrix numerical rank of $\Z$ changes across training; it does not." — this is a clarifying paragraph and is fine, but the "We are not claiming X; it does not" phrasing is slightly blog-style. Tighten to: "Throughout, ``rank'' refers to \emph{effective rank} for training-curve reporting and to \emph{hard SVD truncation rank} for the rank-$k$ ablation. The dense-matrix numerical rank of $\Z$ is always $d$ and does not change during training; the question is whether the structural capacity offered by rank $>1$ is functionally used."

### §3 The Flatten-Then-Project Readout is Rank-Blind

**Voice:** Has the most narrative-voice issues of any section.

**Specific edits:**

- Line 4–7: "We first establish the empirical phenomenon and its simplest mechanism. The subsequent section will then show that this mechanism is not the whole story: even readouts that escape it produce flat curves." — this is a narrative-process sentence. It tells the reader what the author is doing structurally rather than advancing the claim. Replace with: "This section establishes the empirical phenomenon and its simplest mechanism. §\ref{sec:pc} shows that the mechanism is not the whole story: readouts that escape it also produce flat curves." (Just one less "we first … then show.")
- Line 11–14: "We ran the matrix bottleneck under four training conditions …" Past tense "we ran" is fine and is standard in experimental sections. Keep.
- Line 105–109: "This is the cleanest single piece of evidence in the paper that the distillation objective does not reward rank. It is not an ablation artefact --- the three models achieve statistically indistinguishable accuracy at wildly different ranks." — **critical voice flag.** "the cleanest single piece of evidence in the paper" is author-voice meta-commentary. Rewrite: "Figure~\ref{fig:seed-decoupling} directly demonstrates that the distillation objective does not reward rank: three models at the same configuration achieve statistically indistinguishable accuracy at effective ranks spanning $\{4, 12, 13\}$." Also "wildly different" — replace with "over a $3\times$ range" or "spanning a factor of three."
- Line 141–144: "Taken together, the flat rank-$k$ curves, the three-seed decoupling, and the linear probe gap point to the same conclusion: the matrix thought in a flatten-readout matrix-CODI does not construct reasoning state. The structural capacity of $\Z$ is available but not used." — fine. Keep.
- Line 52–58 (Jacobian argument): "The contraction preserves the row-space of $W_{\text{down}}$ but carries no information about the singular structure of $\Z$." — the "carries no information" is strong but defensible given the proposition below. Keep.
- Proposition 1 (line 60–68): Proposition is informally stated. Consider tightening: "The contraction preserves the row-space" is imprecise — what is claimed is that the Jacobian of $\phi$ with respect to $\Z$ is constant in $\Z$. Recommend rewriting the body as: "Let $\phi(\Z) = W\vecop(\Z)$ with $W \in \R^{D \times d^2}$ constant in $\Z$. Then $\partial\phi/\partial\Z$ is constant in $\Z$, so for any differentiable $\mathcal{L}$, $\partial\mathcal{L}/\partial\Z = W^\top(\partial\mathcal{L}/\partial\phi)$ reshaped to $\R^{d\times d}$, with coefficients independent of the singular structure of $\Z$."
- Line 70–78 (proof sketch): fine. "\qedhere" inside `proof` is OK.
- Line 82–84: "--- the symmetric Gram $\Z\Z^\top$, a linear low-rank factorization of $W_{\text{down}}$, or a per-vocab bilinear logit head. All of these have constant Jacobians and therefore produce rank-blind gradients \emph{through the readout}." — "All of these have constant Jacobians" is wrong for $\Z\Z^\top$; the Gram itself depends on $\Z$ nonlinearly. This is a **substance concern**, not style; flagging for attack/defense branches. For style purposes: the list should be tightened.
- Line 87 subsection heading "Accuracy is decoupled from rank across seeds" — fine.
- Line 111 subsection heading "What is encoded in $\Z$?" — this is a rhetorical question. Banned pattern. Change to: "Linear probe on $\Z$" or "Predictive content of $\Z$."

### §4 Depth and Scale Do Not Rescue Matrix-CODI

**Voice:** Has one narrative-reader phrase and one incomplete-experiment note that should be handled.

**Specific edits:**

- Line 4–7: "A reader who has accepted the readout Jacobian argument might reasonably ask whether the failure is specific to $d=16$ GPT-2 small with six latent positions. We tested two axes: the number of iterative latent refinement steps (depth) and the backbone scale." — **banned pattern.** "A reader who has accepted … might reasonably ask" is the conversational-reader narrative device. Replace with: "The results in §\ref{sec:flat} and §\ref{sec:pc} may be specific to $d=16$, GPT-2 small, and six latent positions. We test two axes: depth (number of iterative latent refinement steps) and backbone scale."
- Line 15: "The $n\in\{16,32,64\}$ configurations OOM'd at the default batch sizes and are pending re-runs at smaller batches." — "OOM'd" is informal. Change to: "The $n\in\{16,32,64\}$ configurations exceeded memory at the default batch sizes; re-runs at smaller batches are pending."
- Line 22–24 (figure caption): "The $n=16,32,64$ points are partial data and will be added in the appendix when the re-runs complete." — this reads as a TODO in a reviewer-facing caption. Either complete the re-runs before submission, or rewrite to: "Results for $n \in \{16, 32, 64\}$ are omitted due to memory constraints at the default configuration." Do not leave "will be added" in a submission.
- Line 37: "Matrix-CODI at large OOM'd at batch $4$ and $2$; we report it as pending." — same fix. "Matrix-CODI at GPT-2 large exceeded memory at batches of 2 and 4; it is omitted from Fig.~\ref{fig:scale}." Remove "pending" from the submission version.
- Line 44–47 (Fig.~\ref{fig:scale} caption): "ProsQA is small enough that the larger backbone overfits or underoptimizes at default learning rate." — this is a hedged interpretation inside a caption, which is fine, but "overfits or underoptimizes" is vague. Consider: "ProsQA is small enough (17,886 examples) that default AdamW at $\text{lr}=10^{-4}$ likely under-optimizes larger backbones."
- Line 52–65 (bulleted observations): fine — quantified, sharp.
- Line 67–71: "The scale sweep should therefore be read narrowly: at the tested operating points, matrix-CODI underperforms its own matched vanilla SFT baseline. It is not a statement about whether matrix bottlenecks can ever help at larger scale under different training regimes; it is a statement that they do not help in this one." — **slightly self-defensive voice.** Reviewers read this as anticipating a criticism. Tighten to: "This sweep establishes that matrix-CODI underperforms a matched vanilla SFT baseline at the three tested scales and training configurations. It does not rule out settings in which matrix bottlenecks help at larger scale under different training regimes."
- Line 73–79: "A complementary sample-efficiency sweep …" — fine and quantified.

### §5 Positive Control

**Voice:** Contains the paper's single biggest narrative-process slip.

**Specific edits:**

- Line 8–9: "This section runs that positive control and reports the falsification of our own hypothesis." — **critical voice flag.** "falsification of our own hypothesis" is narrative-about-author. Rewrite: "This section runs that positive control. The hypothesis — that readout linearity is the full mechanism — is falsified."
- Line 26–30: "This variant is a reparametrization control --- it verifies that the low-rank factoring of $W_{\text{down}}$ does not change the curve." — fine.
- Line 36–38: "$\sigma$ is a non-differentiable-at-collision function of $\Z$ in the worst case, but PyTorch's SVD provides subgradients that route accuracy gradients through the singular spectrum." — "in the worst case" is filler; rephrase: "Although $\sigma$ is non-differentiable at singular-value collisions, PyTorch's SVD provides subgradients that route gradients through the singular spectrum."
- Line 48–51: "Each of Bilinear+GELU, SVD-augmented, and Quadratic violates the sufficient condition of Proposition~\ref{prop:jac}. The hypothesis under test is: if readout linearity is the mechanism behind the flat rank-$k$ curves, then breaking readout linearity should bend the curve." — clean and properly falsification-framed. Keep.
- Line 95–99: "Our original Jacobian hypothesis predicted that at least SVD-augmented (which literally exposes singular values) should show a rank-dependent accuracy. It does not." — **critical voice flag.** "Our original Jacobian hypothesis" is self-narrative. Rewrite: "Under the Jacobian hypothesis, the SVD-augmented readout — which exposes singular values directly to the optimizer — should produce a rank-dependent curve. It does not." Also drop "literally" (banned-word-adjacent).
- Line 104–107: "The positive control falsifies the narrow reading of Proposition~\ref{prop:jac} as the \emph{full} explanation of the flat curves. Readouts with non-constant Jacobians still produce flat rank-$k$ curves. The mechanism is therefore deeper than readout linearity alone." — fine.
- Line 109–118: "The simplest consistent explanation is that the CODI distillation objective produces gradients that do not reward any particular rank of $\Z$, \emph{through the full chain rule}. … Proposition~\ref{prop:jac} remains correct as a sufficient condition; it is not necessary." — good.
- Line 120–126: "Combined with the three-seed decoupling result in Fig.~\ref{fig:seed-decoupling} --- where the same loss lands at wildly different ranks across seeds --- the parsimonious conclusion is that the rank of $\Z$ under CODI distillation is a free direction in the loss landscape." — "wildly different" again; "parsimonious conclusion" is a touch rhetorical. Change to: "Combined with the three-seed decoupling (Fig.~\ref{fig:seed-decoupling}), in which the same loss lands at effective ranks spanning a $3\times$ range across seeds, the natural conclusion is that the rank of $\Z$ under CODI distillation is a free direction in the loss landscape."
- Line 128–151 (numbered implications): fine, quantified, sharp.
- Line 153–156: "The strongest reading of the result is: inside a matrix-CODI stack, the rank of the matrix thought is \emph{not} a functional observable, independently of how the thought is read. The CODI distillation objective does not shape it." — "The strongest reading of the result is" is author-voice; prefer a direct claim. Rewrite: "In a matrix-CODI stack, the rank of the matrix thought is not a functional observable, regardless of how the thought is read out; the CODI distillation objective does not shape it."

### §6 Related Work

**Voice:** Mostly impersonal, but one paragraph slips into debate-voice.

**Specific edits:**

- Line 17–25: "We do not reproduce that empirical finding in our setting: our vanilla SFT baseline at gpt2-small reaches only $81.77\%$, roughly $15$pp below their number. What we add to their picture is a mechanism …" — "What we add to their picture" is slightly colloquial. Replace with: "Our contribution relative to that work is a mechanism at the architecture level that does not depend on matching their operating point: …"
- Line 27–39 (SIM-CoT paragraph): "Our diagnosis is different" is fine. "The two diagnoses are compatible and non-redundant." — fine.
- Line 70–84: the §6 two-axis positioning paragraph ("Our paper differs from both on two axes that reviewers will check") — **banned-pattern flag.** "that reviewers will check" is meta-review-voice. Drop: "Our paper differs from both on two axes: (i) \emph{object of study} …"
- Line 79–84: "We are the first (to our knowledge) to diagnose rank-blindness as a property of the \emph{readout-plus-objective} and to falsifiably test that diagnosis with explicit nonlinear-in-$\Z$ readouts." — acceptable hedging ("to our knowledge"). Keep.
- Line 93–99 (rank-decay paragraph): "Our rank-blindness is not a consequence of theirs --- it persists with a single bottleneck at each latent position, with or without deeper stacks --- but readers should not confuse the two." — "readers should not confuse the two" is again talking to the reader. Replace: "The two phenomena are distinct: rank-blindness in our setting persists with a single bottleneck at each latent position, independently of stack depth."

### §7 Discussion and Limitations

**Voice:** Limitations section has the highest density of narrative-process voice.

**Specific edits:**

- Line 10–11: "we do not interpret its flat rank-$k$ curve as strong evidence on its own." → fine.
- Line 11–12: "Cross-dataset replication on GSM8K at a higher-accuracy operating point is queued and will be reported in the camera-ready if accepted." — **TODO-in-submission flag.** Camera-ready promises are OK in venue culture but phrase cleanly: "Cross-dataset replication on GSM8K at a higher-accuracy operating point is deferred to the camera-ready version."
- Line 22–32 (Seed-dependent $\Z$ rank paragraph): "The three-seed decoupling (Fig.~\ref{fig:seed-decoupling}) is itself worth reporting." — "is itself worth reporting" is author-editorializing. Drop: "The three-seed decoupling (Fig.~\ref{fig:seed-decoupling}) is a distinct finding: the same configuration, varying only the seed, produces models at effective ranks $\{4, 12, 13\}$ with accuracies $\{81.25, 82.81, 80.47\}$."
- Line 29: "The final rank of $\Z$ in a trained matrix-CODI is essentially arbitrary, governed by initialization noise rather than by any signal in the loss." — "essentially arbitrary" is acceptable; "governed by initialization noise" is a testable claim that the paper has not directly tested (the paper observes seed-dependence, which is not yet "initialization noise"). Hedge: "The final rank of $\Z$ in a trained matrix-CODI appears to be determined by initialization rather than by any signal in the loss; a direct test via controlled $W_{\text{up}}$ initialization is a natural next experiment."
- Line 34–42 (One seed per positive control paragraph): "We do not expect this to change the qualitative picture --- the observed flat curves are flat by a margin that far exceeds seed variation in the three-seed flatten replication --- but it would narrow confidence intervals." — fine, appropriately hedged.
- Line 54–62: "What would move us" subsection title — **critical voice flag.** "What would move us" is explicitly personal. Rename the paragraph heading to "What would change the conclusion" or "Falsifying evidence we do not have." Body text: "A result that would force us to re-interpret would be:" → "A result that would revise this conclusion would be: …" (removes "us").
- Line 59–62: "We designed the four readouts in \S\ref{sec:pc} to maximize the chance of seeing such a result and did not. A sufficiently different training objective, explicitly rewarding rank, is a distinct direction we flag but do not address." — "we flag but do not address" is fine. Keep.
- Line 64–71 (Broader implication): "The structural argument in \S\ref{sec:flat} and the positive control in \S\ref{sec:pc} together close off readout-architecture fixes for matrix-CODI's rank-blindness." — "close off" is slightly strong; recommend softening to "together argue against readout-architecture fixes for matrix-CODI's rank-blindness." But acceptable as-is given the workshop's tolerance for sharper claims.

### §8 Conclusion

**Voice:** Clean, but contains a meta-commentary phrase.

**Specific edits:**

- Line 10–11: "The last finding is the paper's sharpest claim:" — **banned pattern** (paper commenting on itself in author-voice). Drop the meta clause: "Even readouts with non-constant Jacobians (bilinear-plus-GELU, SVD-augmented, quadratic) produce flat rank-$k$ curves: the CODI distillation objective is rank-blind through the full chain rule, not only through the final projection."
- Line 17–18: "don't use their latents" — the contraction "don't" is informal for a paper. Change to "do not use their latents."
- Line 22–25: "Rank-$k$ ablation should not be used as a probe for superposition in CODI-trained matrix latents; the probe does not measure what the name suggests it measures." — good closing line. Keep.

### §9 Reproducibility

**Voice:** Clean.

**Specific edits:**

- Line 11–14: the file-name typography is inconsistent — `w_{\text{up}}`, `w_{\text{down}}` with lowercase subscript conflicts with the rest of the paper's `W_{\text{up}}`, `W_{\text{down}}`. Make consistent (uppercase matches the math convention used elsewhere).
- Line 20–21: "Raw rank-$k$ evaluation JSONs for the four positive-control readouts" — JSONs is colloquial; use "Raw rank-$k$ evaluation output files (JSON) for the four positive-control readouts."

### Figure and table captions

- **Table~\ref{tab:four-flat}** (line 30–35): "Rank-$k$ projection ablation across four training conditions." — the caption does a good job of defining $r_s$ and stating the key observation. Self-contained. Keep. Minor: end of caption "Signs of $r_s$ are inconsistent and magnitudes are within $\pm 0.11$ of zero." → replace with "Signs of $r_s$ are inconsistent across rows; $|r_s| < 0.11$ in all rows."
- **Fig.~\ref{fig:seed-decoupling}** (line 98–103): self-contained and good. "the loss does not push $\Z$ toward any particular rank" — appropriate hedging.
- **Fig.~\ref{fig:probe}** (line 123–129): "Vanilla GPT-2 --- never trained on ProsQA --- predicts the target class better than the trained matrix-CODI bottleneck." — this is an editorial claim inside a caption and is appropriate for the MI workshop. Keep.
- **Fig.~\ref{fig:depth}** (line 19–24): has the "will be added in the appendix when the re-runs complete" phrasing. Must fix (see §4 notes).
- **Fig.~\ref{fig:scale}** (line 42–47): self-contained. Keep.
- **Fig.~\ref{fig:pc}** (line 62–70): "The Quadratic readout is perfectly flat at $79.69\%$ across all five $k$." — "perfectly flat" is precise here (since the numbers are identical). Keep.
- **Table~\ref{tab:pc}** (line 88–92): self-contained. Keep. Minor: "two-sided" should be "two-sided Spearman test."

## Global issues

### Narrative voice (CRITICAL)

Instances of first-person-about-author / research-process narrative:

1. **main.tex line 2–17** — entire `% NOTE TO THE AUTHOR` comment block. DELETE.
2. **Abstract line 68**: "We report a negative result with a mechanism." → "This paper reports a negative result accompanied by a mechanism." (Or delete.)
3. **Abstract line 73**: "We then construct four readout variants …" → "To test …, four readout variants are constructed:" (removes sequential "then").
4. **§3 line 4**: "We first establish the empirical phenomenon and its simplest mechanism. The subsequent section will then show that this mechanism is not the whole story" → "This section establishes the empirical phenomenon and its simplest mechanism. §\ref{sec:pc} shows that the mechanism is not the whole story."
5. **§3 line 105**: "This is the cleanest single piece of evidence in the paper that the distillation objective does not reward rank." → "Figure~\ref{fig:seed-decoupling} directly demonstrates that the distillation objective does not reward rank."
6. **§3 line 108**: "the three models achieve statistically indistinguishable accuracy at wildly different ranks." → "… at effective ranks spanning a $3\times$ range."
7. **§4 line 4**: "A reader who has accepted the readout Jacobian argument might reasonably ask …" → "The results in §\ref{sec:flat} and §\ref{sec:pc} may be specific to …"
8. **§5 line 8**: "This section runs that positive control and reports the falsification of our own hypothesis." → "This section runs that positive control. The hypothesis — that readout linearity is the full mechanism — is falsified."
9. **§5 line 95**: "Our original Jacobian hypothesis predicted …" → "Under the Jacobian hypothesis, the SVD-augmented readout … should produce a rank-dependent curve. It does not."
10. **§5 line 153**: "The strongest reading of the result is:" → direct claim.
11. **§6 line 22**: "What we add to their picture is a mechanism …" → "Our contribution relative to that work is a mechanism …"
12. **§6 line 72**: "Our paper differs from both on two axes that reviewers will check:" → drop "that reviewers will check".
13. **§6 line 99**: "readers should not confuse the two." → "The two phenomena are distinct."
14. **§7 line 22**: "The three-seed decoupling (Fig.~\ref{fig:seed-decoupling}) is itself worth reporting." → "The three-seed decoupling (Fig.~\ref{fig:seed-decoupling}) is a distinct finding:"
15. **§7 line 54**: paragraph heading "What would move us" → "What would change the conclusion" (and strip "us" from the body).
16. **§8 line 10**: "The last finding is the paper's sharpest claim:" → delete the meta clause.

**Count: 16 first-person-about-author / narrative-process instances flagged.**

### Banned words/phrases found

- "actually" — not found.
- "really" — not found.
- "just" — not found.
- "clearly" — not found.
- "obviously" — not found.
- "interestingly" — not found.
- "nicely" — not found.
- "honestly" — not found.
- "basically" — not found.
- "pretty" — not found.

Good — the paper is free of the classic draft-voice filler adverbs. What IS present and worth flagging:

- **"essentially"** — §7 line 29 "is essentially arbitrary". Replace with the concrete claim: "appears to be determined by initialization rather than by any signal in the loss."
- **"wildly"** — §3 line 108 and §5 line 122 ("wildly different ranks"). Replace both with "spanning a $3\times$ range."
- **"literally"** — §5 line 97 "(which literally exposes singular values)". Delete "literally": "(which exposes singular values)".
- **"parsimonious"** — §5 line 124 "the parsimonious conclusion is that". Replace with "the natural conclusion is that" or simply "the conclusion is that".
- **"cleanest"** — §3 line 105 "the cleanest single piece of evidence in the paper". Drop.
- **"sharpest"** — §8 line 10 "the paper's sharpest claim". Drop.
- **"strongest reading"** — §5 line 153. Drop.

**Rhetorical questions found:**

- §3 line 111 subsection heading "What is encoded in $\Z$?" — rename to "Linear probe on $\Z$" or "Predictive content of the matrix bottleneck."

**Em-dash-as-conversational-pause:**

The paper uses `---` (em-dash) liberally (~30 instances). Some are fine (parenthetical asides). Some double as colons and should be replaced for tighter prose:

- Abstract line 80: "the loss does not reward any particular rank." (replace `---` with period).
- §3 line 81: "The same argument applies to any readout that factors through a fixed-size linearly-derived space --- the symmetric Gram $\Z\Z^\top$, a linear low-rank factorization of $W_{\text{down}}$, or a per-vocab bilinear logit head." — OK as list-intro.
- §3 line 107–108: "It is not an ablation artefact --- the three models achieve statistically indistinguishable accuracy at wildly different ranks." → colon: "It is not an ablation artefact: the three models achieve statistically indistinguishable accuracy at ranks spanning a $3\times$ range."
- §3 line 126–127: "Vanilla GPT-2 --- never trained on ProsQA --- predicts the target class better" — parenthetical; keep.
- §5 line 120–122: "Combined with the three-seed decoupling result in Fig.~\ref{fig:seed-decoupling} --- where the same loss lands at wildly different ranks across seeds ---" → use parentheses: "(Fig.~\ref{fig:seed-decoupling}, in which the same loss lands at ranks spanning a $3\times$ range across seeds)."

Count: roughly 6-8 em-dashes that should become colons, periods, or parentheses.

**"Taken together":**

Appears twice (§1 line 69, §3 line 141). Vary the second instance: "In combination, …" or "Collectively, …".

**Contractions:**

- §8 line 17 "don't" → "do not."

### Notation consistency

- `\Z` is defined in main.tex line 44 as `\newcommand{\Z}{Z}` — just italic $Z$. Everywhere in the paper $\Z$ is used (good). **However**, bold $\mathbf{Z}$ is not used, and matrices $W_{\text{up}}$, $W_{\text{down}}$ are italic-capital. Convention within the paper is consistent (italic capital for matrices). No drift found. Good.
- §9 line 11–14: `w_{\text{up}}`, `w_{\text{down}}` (lowercase) — **inconsistent** with the rest of the paper's uppercase $W_{\text{up}}$, $W_{\text{down}}$. Fix.
- $\sigma$ used for both (a) vector of singular values (§5 line 35) and (b) normalized singular values $\tilde\sigma_i$ (§2 line 42). OK since tilde distinguishes; keep.
- $k$ used for both rank truncation ($k\in\{1,2,4,8,16\}$) and number of probes ($K=d^2$, §5 line 22) — uppercase $K$ is used for the count, fine. Keep.
- Consider defining $\phi$ once in §2 and not redefining in §5. Already defined in §2 line 27. OK.

### Figure/table captions

All five figures have captions that state what the figure shows plus a one-sentence interpretation — appropriate for venue. The one problem is **Fig.~\ref{fig:depth}**: "will be added in the appendix when the re-runs complete" — this is a TODO, not a caption. Must fix before submission.

Plot titles inside the figure PDFs are not readable from the tex source; the edit agent should check that the PDFs themselves do not have informal titles ("flat curves!", "negative result", etc.). If the plots were generated during experimentation they may have draft titles that need to be regenerated for submission.

### Hedge calibration

**Places the paper overclaims:**

- §3 line 82–84: "All of these have constant Jacobians and therefore produce rank-blind gradients \emph{through the readout}." — $\Z\Z^\top$ does not have a constant Jacobian in $\Z$. This overclaims and is also a substance issue. **Flag for attack/defense branches.**
- §7 line 29: "governed by initialization noise" — overclaims; the paper has evidence for seed-dependence, not yet for initialization-noise specifically. Hedge to "appears to be determined by initialization."
- §1 line 54–58: "The matrix bottleneck actively loses target-predictive information" — "actively" overclaims; change to "loses target-predictive information."

**Places the paper underhedges / is appropriately assertive:**

- §5 line 153: "the rank of the matrix thought is \emph{not} a functional observable" — this is the paper's central claim and is appropriately strong given the positive-control evidence. Keep (but reword to remove "The strongest reading of the result is:" prefix).
- §8 line 22: "Rank-$k$ ablation should not be used as a probe for superposition in CODI-trained matrix latents" — the claim is scoped correctly ("CODI-trained"). Appropriately assertive. Keep.

**Places the paper is appropriately hedged:**

- §7 line 22–32 on seed-dependence.
- §6 line 79–84 "to our knowledge."
- §7 line 55–61 "What would move us" — the falsification criterion is clearly stated. Good scientific practice. Keep the content; fix the heading.

## Suggested rewrites for key passages

### Rewrite 1: Abstract (trim to ~220 words and de-narrate)

**Before (current, ~288 words):**
> Continuous chain-of-thought models compress reasoning into latent tokens, and matrix-valued variants of these latents introduce a new structural observable: rank. If matrix latents carry parallel reasoning paths as superposition, rank should track them, and truncating the thought matrix $\Z$ to low rank should hurt accuracy on tasks whose solutions plausibly need multiple components. We report a negative result with a mechanism. Across four training regimes of a matrix-CODI model on ProsQA, the rank-$k$ projection ablation curve is flat to within 0.6 percentage points. A three-seed replication shows accuracy tight at $81.5\pm 1.2$pp while the final effective rank of $\Z$ ranges over $4, 12, 13$ --- the loss does not reward any particular rank. We then construct four readout variants designed to falsify the hypothesis that \emph{rank-blindness comes from the flatten-then-project readout}: a bilinear reparametrization, a bilinear-plus-GELU readout explicitly nonlinear in $\Z$, an SVD-augmented readout that feeds the singular values of $\Z$ through an MLP, and a quadratic readout in the second moment $\Z\Z^\top$. All four rank-$k$ curves are flat (Spearman $p$ values $0.63, 0.14, 0.82, 0.46$). The readout Jacobian is not the sole culprit: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule. A linear probe on the matrix thought also underperforms the raw pretrained hidden state at predicting the target (AUC $0.673$ vs $0.846$). We position the finding against two February 2026 papers that measure rank in linear-attention hidden states: those papers describe rank as a structural invariant of trained states, while we make a falsifiable mechanism claim about the objective that our positive controls specifically test.

**After (~220 words, no narrative-about-author, ends on implication):**
> Continuous chain-of-thought models compress reasoning into latent tokens. Matrix-valued variants introduce a new structural observable — the rank of the latent matrix $\Z$. If matrix latents carry parallel reasoning paths via superposition, rank should track them, and truncating $\Z$ to low rank should hurt accuracy on tasks whose solutions plausibly require multiple components. Across four training regimes of a matrix-CODI model on ProsQA, the rank-$k$ projection ablation curve is flat to within 0.6 percentage points. A three-seed replication yields accuracy at $81.5 \pm 1.2$pp while the final effective rank of $\Z$ spans $\{4, 12, 13\}$; the loss does not reward any particular rank. To test whether rank-blindness arises from the flatten-then-project readout alone, four readout variants are constructed: a bilinear reparametrization, a bilinear-plus-GELU readout explicitly nonlinear in $\Z$, an SVD-augmented readout feeding singular values through an MLP, and a quadratic readout in $\Z\Z^\top$. All four rank-$k$ curves remain flat (Spearman $p$-values $0.63, 0.14, 0.82, 0.46$). Readout linearity is therefore a sufficient but not necessary condition: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule. A linear probe on $\Z$ underperforms a raw pretrained hidden state at target prediction (AUC $0.673$ vs $0.846$). Rank-$k$ ablation should not be used as a probe for superposition in CODI-trained matrix latents.

### Rewrite 2: §3.3 closing paragraph (remove meta-commentary)

**Before (lines 105–109):**
> This is the cleanest single piece of evidence in the paper that the distillation objective does not reward rank. It is not an ablation artefact --- the three models achieve statistically indistinguishable accuracy at wildly different ranks.

**After:**
> Figure~\ref{fig:seed-decoupling} directly demonstrates that the distillation objective does not reward rank: three models at identical hyperparameters achieve statistically indistinguishable accuracy at effective ranks spanning a $3\times$ range. The result is not an ablation artifact, since the accuracies are computed on the same test split under standard decoding.

### Rewrite 3: §5 opening + §5.2 closing paragraph (remove "our own hypothesis" narrative)

**Before (§5 lines 8–9 and §5 lines 95–99):**
> This section runs that positive control and reports the falsification of our own hypothesis.

> All four $p$-values are above $0.14$. The Quadratic readout is identical across all five $k$. Our original Jacobian hypothesis predicted that at least SVD-augmented (which literally exposes singular values) should show a rank-dependent accuracy. It does not.

**After:**
> This section runs that positive control. The hypothesis that readout linearity is the full mechanism is falsified.

> All four $p$-values are above $0.14$; the Quadratic readout is identical across all five $k$. Under the Jacobian hypothesis, the SVD-augmented readout — which exposes singular values directly to the optimizer — should produce a rank-dependent curve. It does not.

## New style-level nits that don't fit above

- **"percentage points" vs "pp":** The paper uses both "pp" (shorthand) and spells out "percentage points" in the abstract. Pick one and use consistently. Preferred: "pp" throughout after first-use expansion.
- **"GPT-2 small", "GPT-2 medium", "GPT-2 large", "gpt2-small":** Mixed capitalization. Standardize to GPT-2 {small, medium, large} in prose and `gpt2-small` only when referring to the HuggingFace identifier (in typewriter font). Currently drifts in §3 line 92 ("gpt2-small") and §4 line 42 ("gpt2-large").
- **$17{,}886$ vs $17886$:** §4 line 76 uses "17,886" but §4 line 77 uses "$N=17886$". Standardize to "17{,}886" throughout (LaTeX: `$17{,}886$`).
- **"matrix-CODI" vs "Matrix-CODI":** §6 line 7 capitalizes "The Matrix-CODI variant", elsewhere lowercase. Pick lowercase hyphenated ("matrix-CODI") as the default, capitalizing only at sentence start.
- **"thinker on/off" terminology:** The model has a component called the "multiplicative thinker" (§3) or "thinker" (abstract, §3). Define once in §2 and use consistently.
- **Abstract $\Z$ ranges:** "ranges over $4, 12, 13$" (abstract) and "varies over $4, 12, 13$" (§1 contribution 2) — phrasings differ. Use "spans $\{4, 12, 13\}$" in both.
- **Table 1 (`tab:four-flat`):** Header `k{=}1` / `k{=}16` is LaTeX macro to suppress kerning; fine, but consider `$k{=}1$` in math mode for consistency. Minor.
- **Proposition 1 title:** "Constant-Jacobian rank-blindness" — descriptive. OK. Could be tightened to "Rank-blindness of linear readouts."
- **"facebookresearch/coconut":** §2 line 51 refers to the repo. Use `\texttt{facebookresearch/coconut}` consistently (already in typewriter).
- **GSM8K-Aug at 6% accuracy** — the paper notes this is below where the result is interpretable. Consider mentioning in the abstract that the core claim is on ProsQA, to pre-empt reviewers asking why GSM8K-Aug is in Table 1. (Currently the discussion in §7 handles this but the abstract implies four regimes are equal evidence.)
- **Appendix promises:** §4 line 23 and §4 line 37 mention pending re-runs / appendix additions. §7 line 12 mentions camera-ready additions. At submission, the paper should either complete those or phrase them as "deferred" / "omitted due to compute constraints." Do not use future-tense-active ("will be added") in a submission.
- **Acronyms:** "CoT" is expanded on first use (§1 line 4). "CODI" is cited-first without expansion — acceptable since it is a named method. "SVD" is never expanded; for the MI workshop this is fine (audience is expected to know). "AUC" is never expanded; expand once: "area under the ROC curve (AUC)."

End of critique.
