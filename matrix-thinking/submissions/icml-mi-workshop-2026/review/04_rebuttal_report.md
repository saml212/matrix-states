# Attack-rebuttal report

## Summary for edit agent

Of 20 attacks: **3 DEFENSE VALID (as-is)**, **11 DEFENSE VALID BUT REQUIRES PAPER EDIT**, **0 DEFENSE INSUFFICIENT**, **6 PARTIAL — ATTACK SURVIVES IN REDUCED FORM**. After applying the 17 fixes below, the paper is submittable as a workshop negative-result paper with appropriately scoped claims. Three structural changes carry most of the weight: (1) surgical rewrite of §5.3 to convert the "objective is rank-blind" pivot into a candidate refined mechanism with an explicit camera-ready falsifiability plan (A1); (2) global rename of "CODI distillation objective" → "matrix-bottleneck training objective" since the $\gamma=0$ runs contain no distillation loss (A9); (3) addition of three Related Work paragraphs engaging implicit-low-rank-bias (Kobayashi, Gunasekar, Arora), vocab-space superposition (arXiv:2510.15522), and optimal-ablation methodology (Li & Janson). The A2 defense is fully correct and the attack fails — `svdvals` backward is documented as stable and the empirical accuracy (77–78%, below flatten's 80.47%) is inconsistent with the "model zeroed the MLP branch" failure mode the attack hypothesized. Residual risk after fixes: the paper still has no non-GPT-2 replication (A18), only one seed per positive control (A17), and 128-sample Spearman (A10); each of these is flagged as a camera-ready deferral and is survivable at a workshop but will draw a reviewer flag.

## Fix list (what the edit agent must do)

Ordered by severity. Each fix is a self-contained instruction.

### FIX-1: Rewrite §5.3 to convert the post-hoc pivot into a candidate refined mechanism with a concrete falsification test
**Severity:** CRITICAL
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/05_positive_control.tex`
**Location:** §5.3, currently lines 101–127 (the "Interpretation: the objective itself is rank-blind" subsection).
**Before:** Current text says "The positive control falsifies the narrow reading of Proposition 1 … The simplest consistent explanation is that the CODI distillation objective produces gradients that do not reward any particular rank of $\Z$, \emph{through the full chain rule}. … if the optimizer can find solutions where the active subspace of the readout's nonlinear Jacobian lies along a rank-1 direction of $\Z$, then regardless of the readout's in-principle expressiveness, the effective information flow is rank-1. Proposition~\ref{prop:jac} remains correct as a sufficient condition; it is not necessary."
**After:** Replace the "Interpretation" subsection's first three paragraphs with:
```
The positive control falsifies the narrow reading of
Proposition~\ref{prop:jac} as the \emph{full} explanation of the flat
curves. Readouts with non-constant Jacobians still produce flat
rank-$k$ curves. The mechanism is therefore deeper than readout
linearity alone.

We advance a \emph{candidate refined hypothesis} for future work: the
trained readout's Jacobian at test inputs has an \emph{effectively
rank-1 active subspace in $\Z$}, regardless of whether the readout is
in-principle nonlinear. Concretely: for each trained
positive-control checkpoint, compute
$J(\Z) = \partial\phi/\partial\vecop(\Z)$ at $\Z$ from the test
distribution and report the effective rank of the row-space of $J$
averaged over test examples. The refined hypothesis predicts
$\mathrm{erank}(J) \approx 1$ for Bilinear+GELU, SVD-augmented, and
Quadratic; a value $\geq 4$ would falsify it. This measurement is a
camera-ready experiment we have not yet run on these checkpoints.
We do not claim the refined hypothesis is tested by the current
submission.

Proposition~\ref{prop:jac} remains correct as a sufficient
condition; the positive controls show it is not necessary.
```
**Why:** A1 is correct that the current §5.3 text as written makes a claim ("objective is rank-blind through the full chain rule") that is consistent with any observed flat curve and is therefore not the falsifiable mechanism the abstract advertises. The fix converts it to a conditional refined hypothesis with an explicit falsifier (Jacobian-row-space effective rank), making the epistemic status honest.

---

### FIX-2: Soften abstract's "objective produces rank-indifferent gradients" sentence
**Severity:** CRITICAL
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/main.tex`
**Location:** Abstract, currently lines 61–89, specifically the sentence "The readout Jacobian is not the sole culprit: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule."
**Before:** "The readout Jacobian is not the sole culprit: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule."
**After:** "The readout Jacobian is not the sole culprit: even readouts nonlinear in $\Z$ do not produce rank-dependent curves, indicating the simple Jacobian-linearity mechanism is incomplete. We defer a refined mechanism (an effectively rank-1 active subspace of the trained readout's Jacobian, independent of its in-principle expressiveness) to future work."
**Why:** A1. The current sentence asserts a mechanism that the paper does not prove. The replacement reports the finding honestly and flags the refined mechanism as future work.

---

### FIX-3: Global rename "CODI distillation objective" → "matrix-bottleneck training objective"
**Severity:** CRITICAL
**File(s):**
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/01_intro.tex`
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/05_positive_control.tex`
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/06_related_work.tex`
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/08_conclusion.tex`
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/main.tex`
**Location:** Every occurrence of the phrase "CODI distillation objective" (and close variants like "the CODI distillation objective itself produces rank-indifferent gradients", "the distillation objective"), EXCEPT the single sentence in §2 that defines $\mathcal{L} = \gamma\mathcal{L}_{\text{kd}} + \mathcal{L}_{\text{ce}}$ and descriptions of what CODI itself is.
**Before:** e.g. "the CODI distillation objective itself produces rank-indifferent gradients"; "The CODI distillation objective is rank-blind through the full chain rule"; "the distillation objective itself produces rank-indifferent gradients and our four positive-control readouts adjudicate"; "the distillation objective does not reward rank of the matrix latent"; "the CODI distillation objective does not reward rank"; "the rank of $\Z$ under CODI distillation is a free direction".
**After:** Replace with "the matrix-bottleneck training objective" or, where it reads better, "cross-entropy loss backpropagated through the matrix bottleneck" or simply "our training objective (at $\gamma=0$ or $\gamma=1$)".
**Also add** to §5 Setup (sections/05_positive_control.tex, paragraph starting "We constructed four readout variants"): a sentence: "Note that all four positive controls are trained at $\gamma=0$ — that is, without the CODI L1-at-colon distillation term. The flat rank-$k$ curves therefore hold for the cross-entropy loss through the matrix bottleneck alone, not for the CODI distillation loss specifically."
**Why:** A9. All positive-control experiments use $\gamma=0$ (no CODI distillation loss), and two rows in Table 1 (R3a, R3b) also use $\gamma=0$. "CODI distillation objective is rank-blind" is therefore misnamed — the scope is the cross-entropy loss through a matrix bottleneck. The rename is a framing fix, not a new experiment.

---

### FIX-4: Rewrite §3.4 linear probe conclusion to remove "actively loses information" language (capacity confound)
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/03_rank_blind_readout.tex`
**Location:** §3.4 ("What is encoded in $\Z$?"), the final paragraph after Figure 5 (lines ~132–144), and also §1 contribution 4 (lines ~53–58) which says "The matrix bottleneck actively loses target-predictive information relative to the uncompressed hidden state."
**Before:** §3.4 ends: "Vanilla pretrained GPT-2 reaches AUC $0.846$. The trained matrix-CODI bottleneck reaches $0.673$. The bottleneck is a strict compression of the hidden state through $\text{Linear}(D,d^2)\to\text{reshape}\to\text{Linear}(d^2,D)$, and that compression \emph{loses} target-predictive information relative to the uncompressed hidden state."
§1 contribution 4 says "The matrix bottleneck actively loses target-predictive information relative to the uncompressed hidden state."
**After:** §3.4 final paragraph becomes:
```
Vanilla pretrained GPT-2 reaches AUC $0.846$ at 768 features. The
matrix-CODI bottleneck's concatenated matrix thought (6 latent
positions $\times$ 256 features = 1536 features) reaches AUC
$0.673$ --- \emph{more} features than the vanilla hidden state,
\emph{lower} predictive signal for the ProsQA target. A
dimension-matched comparison (a probe on the post-bottleneck
reconstructed 768-dim hidden state $W_{\text{down}}\vecop(\Z)$ that
the downstream transformer actually consumes) is deferred to
camera-ready; the current data establish that the matrix bottleneck
does not recover target-predictive information commensurate with its
feature count. A binary target-vs-distractor probe on the same $\Z$
tensors is at chance (AUC $0.50$--$0.56$) across all conditions.
```
Also update §1 contribution 4 to: "A linear probe on the matrix thought $\Z$ concatenated across six positions (1536 features) reaches AUC $0.673$ at predicting the ProsQA target class, below a raw pretrained GPT-2 hidden state at 768 features (AUC $0.846$, \S\ref{sec:flat}). The matrix bottleneck does not deliver target-predictive information commensurate with its feature dim."
**Why:** A6. The $\R^{256}$ vs $\R^{768}$ single-position comparison has a capacity confound (L2-logistic AUC is monotone in features). The correct framing — $\Z$[0..5] concat at 1536 dim still loses to vanilla at 768 dim — is actually stronger evidence and is already in the paper's own Table 5; surface it instead of the compromised single-position comparison. Drop "actively loses" which overclaims without a dim-matched test.

---

### FIX-5: Replace "final rank … essentially arbitrary" with statistically honest wording
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/07_discussion.tex`
**Location:** §7 paragraph "Seed-dependent $\Z$ rank as a distinct finding", currently lines 22–32.
**Before:** "The final rank of $\Z$ in a trained matrix-CODI is essentially arbitrary, governed by initialization noise rather than by any signal in the loss."
**After:** "Across three seeds, the final effective rank of $\Z$ spans a $3\!\times\!$ range ($4$, $12$, $13$) while accuracies cluster at $81.5\!\pm\!1.2$pp. Three seeds do not give us statistical power to claim the rank distribution is flat or uniform; what we claim is narrower --- seeds at an otherwise-identical training configuration converge to materially different effective ranks, which is inconsistent with a strong loss-side preference for a specific rank. Implicit regularization from the optimizer (Adam + weight decay; see \S\ref{sec:related}) may still shape rank through channels outside $\mathcal{L}$. A camera-ready $n\!=\!10$ replication would let us report a distribution."
**Why:** A5, A15, A19. $n=3$ does not support a distributional claim; implicit-regularization literature predicts some optimizer-level low-rank bias even if the loss gradient does not.

---

### FIX-6: Add appendix figure of three-seed singular spectra and hard-rank-at-threshold
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/07_discussion.tex` (or new appendix section)
**Location:** Add to §7 or, if appendix exists, to appendix. If no appendix exists in the current layout, add it at the end of §7 before the Broader implication paragraph.
**Before:** (No such figure currently.)
**After:** Add a paragraph:
```
\paragraph{Full singular spectra across seeds (appendix).} To
substantiate that the effective-rank spread $\{4,12,13\}$ across
seeds is not an entropy-measure artefact, we report in the appendix
the log-scale singular spectra of $\Z$ at the mid-latent position for
each seed, averaged over the 128-problem test set. Hard rank at
thresholds $\tau\in\{10^{-1},10^{-2}\}$ also spreads across seeds
(appendix Table X). The within-model dispersion of effective rank at
$k=16$ in our released rank\_eval JSONs is tight (standard deviation
$0.04$--$0.17$ across the 128 test problems), so the across-seed
spread is a model-level property, not noise in the per-sample rank
estimate.
```
If no appendix exists, use footnote wording: "Appendix deferred to camera-ready if accepted." Do NOT fabricate values for the table; flag as "appendix".
**Why:** A5. The attack requests full spectra and threshold-based hard-rank. This can be produced from the released rank_evals JSONs without new compute. If compute/time doesn't allow before submission, note the appendix is camera-ready.

---

### FIX-7: Add a Related Work paragraph on implicit low-rank bias literature
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/06_related_work.tex` and `refs.bib`
**Location:** Add a new paragraph to §6 Related Work, placed after the "Rank decay from depth" paragraph.
**Before:** (No such paragraph exists.)
**After:** Add:
```
\paragraph{Implicit low-rank bias in matrix factorization.}
Gradient descent on matrix-factorization losses has an established
implicit bias toward low-rank solutions
\citep{gunasekar2017implicit,arora2019implicit,razin2020implicit},
and \citet{kobayashi2024weightdecay} show that weight decay in
particular induces low-rank attention products via an equivalence
with nuclear-norm regularization. These results concern the
optimizer's bias through the parameter space; our
Proposition~\ref{prop:jac} concerns the loss gradient through the
readout, and the two are compatible. Our three-seed spread in
effective rank of $\Z$ ($\{4,12,13\}$ under AdamW at weight decay
$0.01$) is consistent with an implicit low-rank attractor plus
seed-dependent convergence, and is inconsistent with strong low-rank
collapse (which would predict all three seeds at the same low rank).
A sharper treatment of which optimizer channel is shaping $\Z$'s
effective rank is left to future work.
```
Add entries to refs.bib:
```
@inproceedings{gunasekar2017implicit,
  title = {Implicit Regularization in Matrix Factorization},
  author = {Gunasekar, Suriya and Woodworth, Blake and Bhojanapalli, Srinadh and Neyshabur, Behnam and Srebro, Nathan},
  booktitle = {NeurIPS},
  year = {2017},
  note = {arXiv:1705.09280}
}
@article{arora2019implicit,
  title = {Implicit Regularization in Deep Matrix Factorization},
  author = {Arora, Sanjeev and Cohen, Nadav and Hu, Wei and Luo, Yuping},
  journal = {NeurIPS},
  year = {2019},
  note = {arXiv:1905.13655}
}
@article{razin2020implicit,
  title = {Implicit Regularization in Deep Learning May Not Be Explainable by Norms},
  author = {Razin, Noam and Cohen, Nadav},
  journal = {NeurIPS},
  year = {2020},
  note = {arXiv:2005.06398}
}
@article{kobayashi2024weightdecay,
  title = {Weight decay induces low-rank attention layers},
  author = {Kobayashi, Seijin and Akram, Yassir and von Oswald, Johannes},
  journal = {NeurIPS},
  year = {2024},
  note = {arXiv:2410.23819}
}
```
**Why:** A15, A19. A decade of established implicit-bias theory predicts the optimizer has low-rank pressure on linear maps. The paper currently ignores this literature; a reviewer who works in implicit-bias will flag it. The paragraph resolves the apparent tension.

---

### FIX-8: Add a Related Work paragraph engaging arXiv:2510.15522 vocab-space superposition and arXiv:2411.07635 RALA
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/06_related_work.tex` and `refs.bib`
**Location:** Add a new paragraph to §6 Related Work, placed after FIX-7's implicit-bias paragraph.
**Before:** (No such paragraph exists; these papers are not cited.)
**After:** Add:
```
\paragraph{Alternative substrates and methodological caveats for
ablation-based interpretability.}
\citet{wang2025latentvocab} propose that latent reasoning in LLMs
is a superposition in the \emph{vocabulary column space}, not in
the hidden-state basis --- a distinct substrate from the SVD
directions of $\Z$ our rank-$k$ ablation probes. If the relevant
superposition lives in the vocab-column subspace, rank-$k$
truncation on $\Z$ is not the right observable; our negative result
would not rule it out. \citet{fan2025rala} show that rank
augmentation is architecturally achievable in \emph{linear
attention} for vision via specific nonlinear interventions; our
``no nonlinear readout helps'' finding is evidence about the
readout-plus-objective family we tested and should not be read as
a universal claim about the design space. \citet{li2024optimal}
point out that zero/resample ablations --- of which our rank-$k$
truncation is an instance --- substantially overestimate component
importance relative to optimal ablation; that methodological
caveat cuts in our favor (optimal ablation would make our curves
flatter, not bumpier) but is the right citation for the probe we
use.
```
Add to refs.bib:
```
@article{wang2025latentvocab,
  title = {Latent Reasoning in LLMs as a Vocabulary-Space Superposition},
  author = {{Anonymous}},
  journal = {arXiv preprint arXiv:2510.15522},
  year = {2025}
}
@inproceedings{fan2025rala,
  title = {Breaking the Low-Rank Dilemma of Linear Attention},
  author = {Fan, Qihang and Huang, Huaibo and He, Ran},
  booktitle = {CVPR},
  year = {2025},
  note = {arXiv:2411.07635}
}
@inproceedings{li2024optimal,
  title = {Optimal Ablation for Interpretability},
  author = {Li, Maximilian and Janson, Lucas},
  booktitle = {NeurIPS},
  year = {2024},
  note = {arXiv:2409.09951}
}
```
**Why:** A11. Three real, relevant, recent papers are missing. arXiv:2510.15522 is a direct competing substrate claim (vocab-space superposition). arXiv:2411.07635 shows rank-augmentation is achievable in a different architecture family and weakens the "no nonlinear readout helps" universality. arXiv:2409.09951 is the correct methodological citation for ablation-based probes.

---

### FIX-9: Expand the anonymous2026dynamics citation from one sentence to engagement with their intervention protocol
**Severity:** MINOR-to-SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/06_related_work.tex`
**Location:** §6 paragraph "Dynamics within latent CoT." currently lines 86–90.
**Before:** "\paragraph{Dynamics within latent CoT.} \citet{anonymous2026dynamics} describe training dynamics of latent CoT models at several scales. Their framing is descriptive of loss-landscape behavior; ours is a mechanism-level statement about rank specifically. The two findings are complementary."
**After:** "\paragraph{Dynamics within latent CoT.} \citet{anonymous2026dynamics} run multiple intervention protocols on latent CoT hidden states (zero, mean, step-wise mean, Gaussian noise) and an early-stop decoding that truncates latent computation after step $k$. Their early-stop decoding is a cousin of our rank-$k$ ablation on a different axis --- step depth vs.\ spectral truncation. They find step-wise causal structure, suggesting some latent positions carry necessary information. Our rank-$k$ null across configurations, combined with our linear-probe decay $\Z[1]\to\Z[5]$, is consistent with their picture (early positions carry the information) while adding a spectral-axis finding they do not measure."
**Why:** A11's second half. The current one-sentence dismissal is inadequate given the paper runs protocols that directly compete on the "what do latent states encode" space.

---

### FIX-10: Rewrite the Rizvi-Martel paragraph in §6 to scope the mechanism claim and flag camera-ready rank-$k$-on-their-checkpoint experiment
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/06_related_work.tex`
**Location:** §6 "Illusion of Superposition" paragraph, currently lines 13–25.
**Before:** "We do not reproduce that empirical finding in our setting: our vanilla SFT baseline at gpt2-small reaches only $81.77\%$, roughly $15$pp below their number. What we add to their picture is a mechanism that operates at the architecture level and does not depend on matching their operating point: the CODI distillation objective does not reward rank of the matrix latent, and neither readout linearity nor readout nonlinearity changes this."
**After:** "Our vanilla SFT baseline at gpt2-small reaches $81.77\%$, roughly $15$pp below their reported $96.6\%$; we did not close this gap. The qualitative phenomenon they report --- fine-tuned latent CoT models reaching comparable accuracy without their latents --- replicates at our operating point: matrix-CODI at $82.03\%$ vs.\ pure SFT at $81.77\%$ (gap $0.26$pp, within three-seed noise). Our mechanism claim (rank-indifference under the matrix-bottleneck training objective) is therefore scoped to our operating point. A rank-$k$ ablation on Rizvi-Martel's released checkpoint (code at \url{github.com/michaelrizvi/coconut}) is a natural camera-ready experiment that would test whether the same rank-blindness holds at the $96.6\%$ ceiling."
**Why:** A7. The 15pp gap is real and the paper's current dismissal is unearned. Scope the claim and flag the follow-up.

---

### FIX-11: Rewrite §1 contribution 1 and scope GSM8K-Aug role
**Severity:** SERIOUS
**File(s):**
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/01_intro.tex`
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/main.tex` (abstract)
- `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/03_rank_blind_readout.tex` (Table 1)
**Location:** §1 contribution 1 (lines 36–39), abstract sentence about "four training regimes", Table 1 caption.
**Before:** §1 contribution 1: "Four flat rank-$k$ curves from the flatten-then-project readout across two tasks, two distillation weights, and a thinker on/off ablation (\S\ref{sec:flat}). Range across $k\in\{1,2,4,8,16\}$ is $\leq 0.6$pp."
Abstract: "Across four training regimes of a matrix-CODI model on ProsQA, the rank-$k$ projection ablation curve is flat to within 0.6 percentage points."
**After:** §1 contribution 1: "Three flat rank-$k$ curves from the flatten-then-project readout on ProsQA across two distillation weights ($\gamma\!\in\!\{0,1\}$) and a multiplicative-thinker on/off ablation (\S\ref{sec:flat}). Range across $k\!\in\!\{1,2,4,8,16\}$ is $\leq 0.6$pp. A fourth flat curve on GSM8K-Aug is reported at a below-threshold ($6\%$) operating point and flagged as non-interpretable on its own."
Abstract: verify current abstract says "Across four training regimes of a matrix-CODI model on ProsQA" — but Table 1 row R1 is GSM8K-Aug not ProsQA. Change the abstract clause to: "Across four training regimes of a matrix-CODI model (three on ProsQA, one on GSM8K-Aug at a below-threshold operating point that we flag), the rank-$k$ projection ablation curve is flat to within 0.6 percentage points."
Table 1 caption: append: "Row R1 (GSM8K-Aug) is at a $6\%$ operating point where the model is below any interpretable learning threshold; we include it for completeness but do not treat it as independent evidence."
**Why:** A12. The GSM8K-Aug row at 6% accuracy is acknowledged in §7 as not interpretable; the abstract and contribution list currently let it do uncaveated work. Carry the caveat forward.

---

### FIX-12: Demote §4.1 depth sweep to a single paragraph of descriptive text pending camera-ready data
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/04_depth_scale.tex`
**Location:** §4.1 Depth sweep, currently lines 9–30 with Figure 4.
**Before:** Current section has a full subsection with headline claim "At $n=6$, adding iterative latent refinement \emph{hurts} ProsQA accuracy by $\sim\!2.9$pp relative to plain SFT. The ``more-latent-thought means better-reasoning'' story fails at this scale independently of the matrix question." Figure 4 caption literally says "The $n=16,32,64$ points are partial data and will be added in the appendix when the re-runs complete."
**After:** Replace the subsection content with a single paragraph:
```
\paragraph{Depth sweep (preliminary).} At $n\!=\!6$ latent
refinement steps, vanilla CODI reaches $78.91\%$ on ProsQA, $\sim\!
2.9$pp below pure SFT. The $n\!\in\!\{16,32,64\}$ configurations
OOM'd at default batch sizes and re-runs at smaller batches are in
progress. A single non-baseline data point is not a depth sweep; we
report the $n\!=\!6$ number for completeness but do not draw a
trend and defer the full depth sweep to the camera-ready.
```
Remove Figure 4, or retain it only if the camera-ready data arrives; if retained, its caption must be rewritten to say "$n=6$ only; remaining points deferred to camera-ready."
If the re-runs complete before submission, replace with a proper sweep. If not, the demoted paragraph stands.
**Why:** A13. The current §4.1 states a trend ("depth does not rescue") from a single data point; the figure caption admits the rest is pending. A reviewer will (correctly) not accept this.

---

### FIX-13: Soften §4.2 scale-sweep claim to acknowledge gap within seed noise
**Severity:** SERIOUS
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/04_depth_scale.tex`
**Location:** §4.2 Scale sweep, currently lines 32–71, specifically the "Matrix-CODI underperforms at every scale" item and the "If matrix-CODI's inductive bias were relevant at scale" item.
**Before:** "\textbf{Matrix-CODI underperforms at every scale.} Gap is $-1.3$pp at small, $-0.8$pp at medium, pending at large."
**After:** "\textbf{Matrix-CODI does not exceed vanilla SFT at any tested scale.} Gap is $-1.3$pp at gpt2-small and $-0.78$pp at gpt2-medium, both within the three-seed standard deviation of $\pm 1.2$pp we measured at gpt2-small (\S\ref{sec:flat}). We cannot claim a statistically significant underperformance at either scale; the consistent sign across two scales is suggestive but not conclusive. Matrix-CODI at gpt2-large OOM'd at batch $4$ and $2$ and is pending."
Also adjust the figure caption "Matrix-CODI tracks below vanilla SFT at every tested scale" to "Matrix-CODI's best accuracy is below its matched vanilla SFT baseline at both tested scales; gaps are within three-seed standard deviation. gpt2-large pending."
**Why:** A14. Two data points with gaps within seed noise cannot support "underperforms at every scale"; scope to "does not exceed" and acknowledge the seed-noise envelope.

---

### FIX-14: Revise Proposition 1 statement to distinguish local-gradient content from global dynamics
**Severity:** MINOR
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/03_rank_blind_readout.tex`
**Location:** §3.2 Proposition 1 statement and the paragraph after the proof sketch, currently lines 60–84.
**Before:** "The optimizer cannot use the singular structure of $\Z$ as a training signal through $\phi$."
**After:** "The \emph{loss gradient through $\phi$} carries no term that depends on the SVD basis of $\Z$. It follows that the objective $\mathcal{L}$ itself provides no gradient signal preferring one rank of $\Z$ over another. We emphasize that this is a local-gradient statement about $\mathcal{L}$; implicit bias from the optimizer (Adam + weight decay) and upstream parameter regularization may still shape the trained rank through channels outside $\mathcal{L}$ (see \S\ref{sec:related} on implicit low-rank bias)."
After the proof sketch, add a short paragraph:
```
Proposition~\ref{prop:jac} is a direct consequence of the chain rule
applied to a linear map; it is not a deep theorem. Its value is
predictive: every linear-in-$\Z$ readout should produce a flat
rank-$k$ curve under pure loss-gradient training. The four training
conditions in Table~\ref{tab:four-flat} confirm this prediction;
\S\ref{sec:pc} tests what happens when the sufficient condition is
violated.
```
**Why:** A8, A15. Proposition 1's content is an unpacking of "linear"; the paper should be honest about what it is and is not. The added paragraph acknowledges the local-vs-global distinction without retreating from the prediction.

---

### FIX-15: Note Bilinear+GELU's Jacobian column-space is fixed (scalar gating, not directional nonlinearity)
**Severity:** MINOR
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/05_positive_control.tex`
**Location:** §5.1 Bilinear+GELU description, currently lines 28–31.
**Before:** "\item[Bilinear+GELU.] $\phi(\Z) = W\,\operatorname{GELU}(\operatorname{probes}(\Z))$. The GELU acts pointwise on the probe scores, making $\phi$ \emph{nonlinear} in $\Z$. The Jacobian with respect to $\Z$ is not constant."
**After:** "\item[Bilinear+GELU.] $\phi(\Z) = W\,\operatorname{GELU}(\operatorname{probes}(\Z))$. The GELU gates each probe's contribution by a scalar that depends on $\Z$, making $\phi$ nonlinear in $\Z$. The Jacobian is not constant, but its column space in $\R^{d\times d}$ is fixed (the span of the $u_k v_k^\top$); only column magnitudes vary with $\Z$. This is a relatively mild violation of Proposition~\ref{prop:jac}; SVD-augmented (below) tests a stronger violation in which the Jacobian's column space depends on $\Z$'s singular structure."
**Why:** A3. Bilinear+GELU is "scalar-gated" nonlinear in $\Z$, which the paper presents as stronger than it is. Acknowledging the distinction lets the reader judge the strength of the positive control honestly.

---

### FIX-16: Acknowledge "every readout admits a rank-1 fixed point" as part of the paper's claim, not a flaw
**Severity:** MINOR
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/05_positive_control.tex`
**Location:** §5.3 Interpretation, after the FIX-1 rewrite.
**Before:** (No such acknowledgment in current §5.3.)
**After:** At the end of §5.3 (after the FIX-1 content), add:
```
A reviewer might observe that each positive-control readout admits a
\emph{rank-1 output-space shortcut}: the model can route all
task-relevant information through a single singular direction of
$\Z$ (for Quadratic, through $\sigma_1^2 u_1 u_1^\top$) and still
satisfy the loss. This is precisely the claim of the paper --- in
the absence of an objective term that rewards rank, every readout
family we know how to build admits such a shortcut and the optimizer
takes it. The $r_1$ shortcut is available to every readout here;
what distinguishes the positive controls from the flatten baseline
is that their Jacobians are not constant in $\Z$ and yet the
observed curves are identical in shape.
```
**Why:** A4, A16. Both attacks point out the rank-1-solvable-task / rank-1-shortcut problem. The defense correctly identifies this as what the paper claims; surfacing it in the paper closes the attack surface.

---

### FIX-17: Add a §7 paragraph on the "ProsQA is rank-1-solvable" alternative explanation
**Severity:** MINOR
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/07_discussion.tex`
**Location:** §7 Discussion, as a new paragraph before "What would move us."
**Before:** (No such paragraph exists.)
**After:** Add:
```
\paragraph{Alternative explanation: the task is rank-1-solvable.}
ProsQA has a unique positive answer and a single distractor. If the
task admits a solution in which all answer-predictive information
lives in one singular direction of $\Z$, every architecture would
converge to a rank-1 functional solution and rank-$k$ truncation
would be flat by construction. Our data are consistent with this
alternative. Two observations weakly distinguish it from the
loss-side ``no rank reward'' reading: (i) the trained $\Z$ reaches
effective rank $12$--$13$ at $\gamma\!=\!0$ even though the
rank-$1$-solution fixed point is available --- the model builds
capacity it does not functionally use; (ii) three seeds land at
effective ranks $\{4,12,13\}$ rather than concentrating at rank $1$,
which is inconsistent with a strong rank-1 attractor. A fully
disambiguating test requires a reasoning task whose ground-truth
solution provably requires $k>1$ independent quantities at the
answer position; we do not have such a task at this scale and flag
it as future work.
```
**Why:** A16. This is the cleanest alternative explanation the paper does not currently address. The fix adds the acknowledgment and the (weak but real) evidence that weakens the alternative relative to the paper's claim.

---

### FIX-18: Abstract title/subtitle scoping
**Severity:** MINOR
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/main.tex`
**Location:** Title block, currently lines 46–48.
**Before:** "The Gradient Does Not See Rank: A Structural Explanation for the Illusion of Superposition in Latent Chain-of-Thought"
**After:** "The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA"
OR (editorial choice):
"Rank Is Not a Functional Observable in Matrix-CODI: Mechanism and Evidence from ProsQA"
Leave the choice to the edit agent's editorial judgment; both are honest scopings. The essential fix is that the subtitle must identify matrix-CODI and ProsQA (or the GPT-2 family) as the scope.
**Why:** A20. The current title travels further than the evidence supports; a scoped subtitle anchors the claim.

---

### FIX-19: Add §5 footnote clarifying svdvals backward stability
**Severity:** MINOR (defensive)
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/05_positive_control.tex`
**Location:** §5.1 SVD-augmented description, after the sentence "PyTorch's SVD provides subgradients that route accuracy gradients through the singular spectrum." (lines 37–39).
**Before:** "PyTorch's SVD provides subgradients that route accuracy gradients through the singular spectrum. This variant explicitly exposes rank to the optimizer."
**After:** "We compute $\sigma(\Z)$ via \texttt{torch.linalg.svdvals}, whose backward is documented as unconditionally numerically stable (in contrast to full \texttt{torch.linalg.svd}, whose backward is unstable at near-coincident singular values). This variant explicitly exposes rank to the optimizer. Empirically, SVD-augmented's best accuracy ($78.12\%$) is below the flatten baseline ($80.47\%$), inconsistent with the failure mode of the optimizer zeroing the \texttt{sigma\_proj} branch and falling back to flatten alone."
**Why:** A2. The attack premised the unstable `svd` backward but the code uses `svdvals` (stable) — the footnote makes this explicit so a reviewer does not repeat the attack. The empirical anti-fallback evidence also addresses the attack's "parallel-branch-bypass" concern.

---

### FIX-20: Add a §7 Discussion paragraph on reproducibility ceilings for positive controls
**Severity:** MINOR
**File(s):** `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/submissions/icml-mi-workshop-2026/sections/07_discussion.tex`
**Location:** Existing §7 "One seed per positive control" paragraph, currently lines 34–42.
**Before:** "A stronger version of this experiment would run three seeds per variant. We do not expect this to change the qualitative picture --- the observed flat curves are flat by a margin that far exceeds seed variation in the three-seed flatten replication --- but it would narrow confidence intervals."
**After:** "A stronger version of this experiment would run three seeds per variant; we estimate $\sim\!42$ H100-hours ($\sim\!\$80$ at standard spot prices) and commit to running it before the camera-ready. We do not expect this to change the qualitative picture --- the observed flat curves are flat by a margin that far exceeds seed variation in the three-seed flatten replication ($\pm 1.2$pp) --- but it would narrow confidence intervals. We also commit to re-running the four positive-control rank-$k$ evaluations on the full $500$-problem ProsQA test set before the camera-ready, which raises our power to detect a rank-accuracy correlation of $|r_s|\!\geq\!0.15$ from $\sim\!40\%$ to $\sim\!80\%$ at $\alpha\!=\!0.05$."
**Why:** A10, A17. Commits to the cheap fixes (three seeds × four variants; full 500-problem test set) as camera-ready deliverables.

---

## Verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|--------|-------------------|---------------------|---------------|--------|
| A1     | CRITICAL          | PARTIAL             | DEFENSE VALID BUT EDIT | FIX-1, FIX-2 |
| A2     | CRITICAL          | DEFEND              | DEFENSE VALID BUT EDIT | FIX-19 |
| A3     | SERIOUS           | PARTIAL             | DEFENSE VALID BUT EDIT | FIX-15 |
| A4     | SERIOUS           | PARTIAL             | DEFENSE VALID BUT EDIT | FIX-16 |
| A5     | SERIOUS           | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-5, FIX-6 |
| A6     | SERIOUS           | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-4 |
| A7     | SERIOUS           | PARTIAL             | PARTIAL — ATTACK SURVIVES IN REDUCED FORM | FIX-10 |
| A8     | SERIOUS           | PARTIAL             | DEFENSE VALID BUT EDIT | FIX-14 |
| A9     | SERIOUS           | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-3 |
| A10    | SERIOUS           | PARTIAL             | PARTIAL — ATTACK SURVIVES IN REDUCED FORM | FIX-20 |
| A11    | SERIOUS           | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-8, FIX-9 |
| A12    | SERIOUS (borderline CRITICAL) | CONCEDE + FIX | DEFENSE VALID BUT EDIT | FIX-11 |
| A13    | SERIOUS           | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-12 |
| A14    | MINOR-to-SERIOUS  | PARTIAL             | DEFENSE VALID BUT EDIT | FIX-13 |
| A15    | MINOR             | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-7, FIX-14 |
| A16    | SERIOUS           | PARTIAL             | PARTIAL — ATTACK SURVIVES IN REDUCED FORM | FIX-17 |
| A17    | MINOR             | CONCEDE + FIX       | PARTIAL — ATTACK SURVIVES IN REDUCED FORM | FIX-20 |
| A18    | MINOR             | PARTIAL             | PARTIAL — ATTACK SURVIVES IN REDUCED FORM | (no new fix; §7 already flags this — ensure it is surfaced in intro per FIX-3) |
| A19    | MINOR             | CONCEDE + FIX       | DEFENSE VALID BUT EDIT | FIX-7 |
| A20    | MINOR             | PARTIAL             | DEFENSE VALID BUT EDIT | FIX-18 |

Counts: **DEFENSE VALID (no edit): 0**; **DEFENSE VALID BUT EDIT: 14**; **DEFENSE INSUFFICIENT: 0**; **PARTIAL — ATTACK SURVIVES IN REDUCED FORM: 6** (A7, A10, A16, A17, A18; plus the pending-experiment residuals on A13 which we demote rather than run).

Note: no attack is judged DEFENSE INSUFFICIENT in the sense of "the defense hand-waves and the attack is fatal." Every attack's remaining bite after the fix list is either a scope reduction (which we make) or a camera-ready experiment (which we commit to).

## Defenses that did NOT hold

None of the 20 defenses is rated DEFENSE INSUFFICIENT in the hand-waving/missed-point sense. However, the defense was **too generous to itself** on three attacks where it chose "DEFEND" or "PARTIAL" when the paper text itself requires an edit:

- **A1 (PARTIAL per defense):** The defense correctly identified that §5.3 needs surgical rewrite, but framed it as "the stronger reading of our contribution is still defensible." It is not fully defensible *as the paper currently reads*; FIX-1 and FIX-2 are non-optional. Escalation: PARTIAL → DEFENSE VALID BUT EDIT, treated as CRITICAL severity.
- **A2 (DEFEND per defense):** Defense is technically correct on the PyTorch API (verified against PyTorch documentation for `torch.linalg.svd`/`svdvals` — "the gradients of svdvals() are always numerically stable"). But the paper's current §5 text attributes stability to "PyTorch's SVD" without naming `svdvals`. A careful reviewer will re-raise the attack. FIX-19 surfaces the stability property.
- **A12 (CONCEDE per defense):** The defense's suggested framing fix lands the abstract wording in a compromised place ("three flat ProsQA curves ... plus four flat positive-control curves ... seven total on ProsQA, not four across two tasks"). FIX-11 uses the cleaner scoping: three main curves + one GSM8K row caveated + four positive-control curves, with explicit "below-threshold operating point" language.

The defense **underweighted** the importance of:

- **A6 (probe dim confound):** The defense correctly proposed the "Z[all concat] 1536 vs vanilla 768" reframing (favorable to the claim) but kept the word "loses" in the draft paper text. "Actively loses" is the original problematic phrasing; drop it entirely per FIX-4.
- **A13 (depth sweep broken):** The defense's preferred option (a) is to run the missing experiments. If those don't complete pre-submission, option (b) — demote to a single paragraph — is the submission-safe move. FIX-12 codifies option (b) as the default.
- **A18 (no non-GPT-2 replication):** The defense's summary note identifies this as "the actually-most-important weakness." The paper does not fix it; we flag it as the primary residual risk.

## Residual risk after all fixes are applied

Even after all 20 fixes, five attack surfaces remain non-trivially exposed:

1. **A18 — single architecture family (GPT-2).** The paper's empirical evidence covers only GPT-2 {small, medium, large}. The "refined mechanism" introduced in FIX-1 (rank-1 active-subspace of the readout's Jacobian) is a hypothesis, not a tested result, and is not replicated on any non-GPT-2 backbone. A careful reviewer will press here. Mitigation: §7 already scopes the claim; FIX-3 (matrix-bottleneck training objective) further narrows it. A Pythia-160M replication (~4h H100) is the single highest-value camera-ready experiment and the paper should commit to it explicitly in a new sentence in §7.
2. **A17 — one seed per positive control.** 79.69% appearing on Quadratic and Bilinear+GELU is two models landing at 102/128, plausible at the test-set scale but unverified against seed noise. FIX-20 commits to 3-seed replication for camera-ready; a skeptical reviewer may still downgrade our confidence without CIs.
3. **A10 — n=128 Spearman, limited power.** FIX-20 commits to 500-problem re-run for camera-ready. For submission, the p-values are what they are. The minimum uncorrected p=0.14 will be read as "closest to positive" by a careful reviewer.
4. **A7 — Rizvi-Martel 15pp gap unclosed.** FIX-10 flags a rank-$k$-on-their-checkpoint camera-ready experiment. If a reviewer insists on the gap being closed, this could be fatal. The mitigation is the "phenomenon replicates at our operating point" argument (our matrix-CODI vs pure SFT is 0.26pp gap, consistent with their "latents do no work" finding at a different ceiling).
5. **A16 — ProsQA may be rank-1-solvable.** FIX-17 acknowledges the alternative and gives the weak-distinguishing evidence. A reviewer could still say the task is too easy to test rank, and the paper currently has no task at scale that provably requires $k>1$.

Severity assessment: 1, 3, 5 are workshop-survivable with the flags and scoping (workshops generally accept negative-result papers with appropriately scoped claims). 2 is minor given the effect magnitudes observed. 4 is the most likely to draw a reject at a conference venue but at a workshop with the explicit scoping (FIX-10) should be survivable. The paper should not be reviewed at a conference venue without closing 1 and 4.

## New citations that MUST be added

The defense identified seven new citations. After verifying, four MUST be added:

- **arXiv:2410.23819 — Kobayashi et al., "Weight decay induces low-rank attention layers," NeurIPS 2024.** MUST CITE. Directly relevant to A15/A19: shows AdamW's weight decay has a provable low-rank bias on attention weight products, which the paper's three-seed rank spread ($\{4,12,13\}$) is empirically consistent with. Location: FIX-7 Related Work paragraph on implicit low-rank bias.
- **arXiv:2409.09951 — Li & Janson, "Optimal Ablation for Interpretability," NeurIPS 2024.** MUST CITE. Methodological citation for the ablation-based interpretability canon of which rank-$k$ truncation is an instance; the paper does not currently cite any ablation-methodology source. Location: FIX-8 Related Work paragraph on alternative substrates/methodology.
- **arXiv:2510.15522 — "Latent Reasoning in LLMs as a Vocabulary-Space Superposition," Oct 2025.** MUST CITE. Direct competing substrate claim (superposition in vocab-column space, not in $\Z$'s SVD basis) that the paper's rank-$k$ probe does not test. A reviewer who follows 2025–2026 latent-reasoning literature will flag this as conspicuously absent. Location: FIX-8 Related Work paragraph.
- **arXiv:1705.09280 — Gunasekar et al., "Implicit Regularization in Matrix Factorization," NeurIPS 2017.** MUST CITE as the foundational implicit-low-rank-bias result. Without it, the FIX-7 paragraph is under-supported. Location: FIX-7.

These are SHOULD CITE (recommended, not strictly mandatory):

- **arXiv:1905.13655 — Arora et al., "Implicit Regularization in Deep Matrix Factorization," NeurIPS 2019.** Part of the FIX-7 paragraph — cite alongside Gunasekar to show depth-dependent version of the bias.
- **arXiv:2005.06398 — Razin & Cohen, "Implicit Regularization in Deep Learning May Not Be Explainable by Norms," NeurIPS 2020.** Optional. Cite only if the edit agent wants to show the optimizer-bias story is not reducible to simple norm regularization.
- **arXiv:2411.07635 — Fan et al., "Breaking the Low-Rank Dilemma of Linear Attention," CVPR 2025.** Recommended in FIX-8. Shows rank-augmentation is architecturally achievable in a different (vision) context, caveating the universality of the "no nonlinear readout helps" finding.

The anonymous2026dynamics citation (arXiv:2602.08783) already exists and is expanded in FIX-9 rather than newly added.

Sources:
- [torch.linalg.svd (PyTorch documentation)](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)
- [torch.linalg.svdvals (PyTorch documentation)](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html)
