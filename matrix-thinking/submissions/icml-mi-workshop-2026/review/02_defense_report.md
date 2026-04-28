# Defense report

## Summary for attack-rebuttal agent

Of 20 attacks: **7 DEFEND, 8 PARTIAL, 5 CONCEDE+FIX**. The paper is submittable *after* three classes of fixes: (1) reframe the §5.3 pivot as a constrained falsifiable follow-up (A1), (2) soften the abstract/intro headline claim from "four flat curves across two tasks" and "objective-level rank-blindness" to tightly scoped variants (A9, A12, A20), and (3) add three honest caveat paragraphs to Related Work / Discussion engaging the implicit-regularization and vocabulary-space-superposition literature (A11, A19) and the probe-dim confound (A6). The two CRITICAL attacks (A1, A2) are NOT fatal: A2 is wrong on a technical detail (we use `torch.linalg.svdvals`, whose backward is documented as unconditionally stable, not `torch.linalg.svd`, which is not), and A1 reduces to a framing issue for the §5.3 paragraph — fixable in ~3 sentences. The empirical base (seven flat curves, three-seed decoupling, probe gap, sample-efficiency curve) is not weakened by any attack. The workshop's own CfP explicitly welcomes "works that clearly document the strengths and weaknesses of their evidence... even if it weakens the narrative" (mechinterpworkshop.com/cfp), which is the shape of our paper after fixes.

## Defenses (one per attack, in the same order)

### A1: The post-hoc pivot from "readout Jacobian" to "objective is rank-blind" is an unfalsifiable move
**Disposition:** PARTIAL
**Response:**
The attack's framing is correct on one point and wrong on a second. Correct: §5.3 as currently written *does* shift into a conditional ("if the optimizer can find solutions where the active subspace... lies along a rank-1 direction") that a reader could parse as any-result-is-consistent. We should fix that. Wrong: the attack misses that the §5.3 conditional makes a concrete, measurable prediction — *the active subspace of the trained readout's Jacobian at test inputs is effectively rank-1* — which is a 20-line PyTorch computation on our released checkpoints (compute $\partial\phi/\partial\vecop(\Z)$ at $\Z$ from the test set for each of Bilinear+GELU, SVD-augmented, Quadratic; report the effective rank of the Jacobian's row-space). If that number is ~1, the §5.3 claim survives; if it is high (say ≥4), the claim is falsified and we have a new puzzle. This measurement was not in the original submission and we should add it as a "defer to camera-ready" appendix experiment.

The stronger reading of our contribution — "we state a falsifiable mechanism claim and run the control" — is still defensible: Prop 1 was cleanly falsified (we ran it, we reported it). What we should NOT do in the revision is claim that the §5.3 re-description is itself a falsified-and-tested mechanism; it is a *candidate refined mechanism* that we will test in the camera-ready appendix.

**Supporting evidence:**
Our own raw eval JSONs (round_pc rank_evals/*.json) contain per-sample `effective_ranks` at $k=16$ with std across 128 problems of only ~0.04–0.17 — i.e., every trained checkpoint is at a very concentrated effective-rank value, which is the empirical pattern consistent with "active Jacobian subspace is narrow across the test distribution." This is suggestive but not proof.

**What goes in the paper if this defense is accepted:**
(a) Rewrite §5.3 to state: "The narrow reading of Prop 1 is falsified. We advance a refined hypothesis — that the trained readout's Jacobian at $\Z$ has a rank-1 active subspace regardless of architecture — and flag an empirical test of this refined hypothesis as the natural next step. We do not claim the refined hypothesis is tested by our current experiments." (b) Soften abstract sentence "the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule" to "even readouts nonlinear in $\Z$ do not produce rank-dependent curves; the simple Jacobian-linearity mechanism is not the full explanation."

---

### A2: The "SVD-augmented" positive control is broken (PyTorch SVD backward unstable)
**Disposition:** DEFEND
**Response:**
The attack's technical premise is **wrong for our code**. We use `torch.linalg.svdvals`, not `torch.linalg.svd` (run_matrix_codi.py line 318: `sigma = torch.linalg.svdvals(Z_out.float())`). Per the PyTorch 2.9 documentation for `torch.linalg.svd`: *"the gradients of svdvals() are always numerically stable,"* in contrast to full SVD, whose gradients are unstable for repeated/near-zero singular values. `svdvals` backward does **not** involve the $1/(\sigma_i^2-\sigma_j^2)$ term the attack cites — that term arises only in $U$/$V$ backward, which `svdvals` does not compute. The attack conflates the two functions.

The attack's second point — that the SVD-aug readout has a parallel linear flatten branch that the model could use to bypass the MLP — has a kernel of truth worth acknowledging (Prop 1 guarantees *that branch* is rank-blind). But the attack's implication ("the model will zero the MLP and use only flatten") is not what we observe: the SVD-aug best accuracy (78.12%) is *lower* than the flatten baseline (80.47%), which is inconsistent with the model simply falling back to flatten. If it were using only flatten and zeroing sigma_proj, we would expect ≥ flatten accuracy. The fact that sigma_proj is active and costs ~2pp of accuracy is weak but real evidence that gradient flowed through it.

**Supporting evidence:**
- PyTorch docs (2.9): "the gradients of svdvals() are always numerically stable." (https://docs.pytorch.org/docs/2.9/generated/torch.linalg.svd.html)
- run_matrix_codi.py line 318 uses `svdvals`, not `svd`.
- SVD-aug achieves 77.34–78.12% accuracy (below flatten 80.47%), inconsistent with "model ignored sigma_proj."

**What goes in the paper if this defense is accepted:**
Add one sentence to §5 footnote: "We compute $\sigma(\Z)$ via `torch.linalg.svdvals`, whose backward is unconditionally numerically stable, as opposed to full `torch.linalg.svd` which has a documented instability at coincident singular values." Optionally, defer-to-camera-ready a gradient-magnitude log of $\partial\mathcal{L}/\partial\sigma$ during training.

---

### A3: Bilinear+GELU does not break the Jacobian structure as claimed
**Disposition:** PARTIAL
**Response:**
The attack's math is correct: with $\phi(\Z)=W\operatorname{GELU}(\operatorname{probes}(\Z))$ and probes linear in $\Z$, the Jacobian $\partial\phi/\partial\Z = W \operatorname{diag}(\operatorname{GELU}') \cdot P$ is a constant tensor $P$ times a data-dependent *scalar* diagonal. The Jacobian's *column space* in $\R^{d\times d}$ is fixed (the span of the $u_k v_k^\top$); only the *magnitudes* of those columns vary with $\Z$. We agree this is a weaker violation of Prop 1 than "nonlinear in $\Z$" suggests colloquially.

However, the attack understates one thing: even a scalar-gated Jacobian is strictly not constant (the gating depends on $\Z$ through the probe scores), and this is what Prop 1 as stated requires for rank-blindness to hold by the narrow reading. The fact that a *scalar-gated* nonlinear Jacobian is still flat-curve is informative — it tells us that gating-only nonlinearities are not enough. That is what we should say. The attack's suggestion of a "directional" nonlinearity (Jacobian column space varies with $\Z$) is a worthwhile camera-ready addition but not one that invalidates what we have.

**Supporting evidence:**
The SVD-aug readout is a stronger test along the attack's axis: through the $\operatorname{MLP}(\sigma(\Z))$ branch, the Jacobian's column space changes with $\Z$'s singular structure (not just magnitudes). SVD-aug is also flat. So "directional" nonlinearity was tested, at least in one form, and did not help.

**What goes in the paper if this defense is accepted:**
Revise §5.1 Bilinear+GELU description: "The GELU gates each probe's contribution by a scalar that depends on $\Z$, making $\phi$ nonlinear in $\Z$ but leaving the Jacobian's column space fixed. This is a relatively mild violation of Proposition 1; SVD-augmented below tests a stronger violation where the Jacobian's column space depends on $\Z$'s singular structure."

---

### A4: Quadratic discards structure and has a rank-1 fixed point
**Disposition:** PARTIAL
**Response:**
The attack has two claims. First: $\Z\Z^\top$ alone loses $V$ (right singular vectors). We anticipated this and use $\operatorname{concat}(\Z\Z^\top, \Z^\top\Z)$ — which preserves $U$ (from $\Z\Z^\top$), $V$ (from $\Z^\top\Z$), and $\Sigma^2$. Only the relative signs of singular directions are lost (a $d$-bit ambiguity), not "half the structure." The attack is correct that $\Sigma^2$ rather than $\Sigma$ is what's exposed, but this is not a fatal information loss for rank structure.

Second: the quadratic readout admits a rank-1 fixed point where the model places everything in $\sigma_1^2 u_1 u_1^\top$. This is true, but it is true of **every** readout we tested — including flatten, where the model can place everything in a rank-1 direction of $W_{\text{down}}$'s row space. The rank-1-shortcut availability is the *claim* of the paper, not a flaw in the control. Our framing already says this: KILL_LIST Lesson 2 — "The model always finds the lowest-rank representation that satisfies the loss." We should make this explicit in §5.

**Supporting evidence:**
KILL_LIST.md Lesson 2 states: "Adding a more rank-aware readout doesn't force rank usage — it just makes rank usage possible. The model still chooses the easiest path, which is rank-1." The attack's critique is a restatement of this lesson; we should surface it.

**What goes in the paper if this defense is accepted:**
Add to §5.3: "A reviewer might note that each positive-control readout admits a rank-1 fixed point: the model can route all task-relevant information through the top singular direction of $\Z$ and hit an output-space fixed point. This is precisely the claim we make — in the absence of an objective that rewards rank, every readout family we know how to build admits such a shortcut and the optimizer takes it."

---

### A5: Three-seed decoupling: n=3 with entropy rank is not adequate
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct on the statistical claim. $n=3$ with a smooth entropy measure is too few to make a *distributional* claim ("the final rank is arbitrary"). Our current §7 wording — "the final rank of $\Z$ in a trained matrix-CODI is essentially arbitrary, governed by initialization noise" — overclaims.

What we actually have is: three seeds with identical hyperparameters produce accuracies tightly clustered at 81.5±1.2pp while effective ranks span 4 to 13. That is informative, but as a distributional claim it requires either more seeds or weaker phrasing. We have compute to run ~5 more seeds (~$50 H100-hours) before camera-ready. For submission, we should downgrade the claim language.

The attack's specific suggestion — report full spectra and hard-rank-at-threshold — is a cheap fix we should just do. The existing rank_evals JSONs have per-sample effective_ranks and the checkpoints are available.

**Supporting evidence:**
Our rank_evals JSONs show per-sample effective_rank std within each trained model of 0.04–0.17 — the within-model dispersion is small, so the 3× spread *across seeds* (4, 12, 13) is not a within-distribution artifact of noisy estimation. But n=3 is n=3.

**What goes in the paper if this defense is accepted:**
(1) Replace "essentially arbitrary, governed by initialization noise" with "span a 3× range (4, 12, 13) despite accuracies tightly clustered at 81.5±1.2pp; we do not have statistical power from three seeds to claim the distribution is flat, only that seeds at the same config converge to materially different effective ranks." (2) Add an appendix figure with the three full singular spectra (log-scale). (3) Report hard rank at $\tau=\{10^{-1},10^{-2}\}$ in the caption. (4) Defer n=10 seed replication to camera-ready.

---

### A6: Linear probe comparison has a capacity confound
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct. $\Z\in\R^{256}$ vs. vanilla hidden at $\R^{768}$ is a capacity-asymmetric comparison, and L2-regularized logistic regression AUC is (weakly) monotone in feature dim. Our §3.4 framing "the matrix bottleneck actively loses target-predictive information relative to the uncompressed hidden state" is not justified by the comparison as stated.

The cheap fix: the post-bottleneck reconstructed hidden state $W_{\text{down}}\vecop(\Z)\in\R^{768}$ *is* what the downstream transformer sees, and a probe on that at matched 768-dim is the right comparison. That is a 30-minute probe run on our existing checkpoint. We should do it before submission or defer the claim.

The attack also notes that Z[all concat] is 1536-dim, which is *more* features than vanilla's 768, and still underperforms. That direction of comparison is actually favorable to the claim — we should surface it instead of the single-position numbers.

**Supporting evidence:**
Table 5 of PAPER_RESULTS_SUMMARY: "matrix Z[0..5] concat" = 1536 features at AUC 0.673 vs. vanilla 768 features at AUC 0.846 — the matrix concat is dim-advantaged and still loses by 0.17.

**What goes in the paper if this defense is accepted:**
Replace the §3.4 conclusion sentence with: "At matched or greater feature dimensionality (Z-concat: 1536 features; vanilla: 768 features), the matrix-CODI bottleneck reaches lower target-predictive AUC than the raw pretrained GPT-2 hidden state. A dimension-matched comparison via the post-bottleneck reconstructed 768-dim hidden state is deferred to camera-ready." Drop "actively loses" — use "underperforms" which is literal.

---

### A7: Vanilla SFT baseline 15pp below Rizvi-Martel's
**Disposition:** PARTIAL
**Response:**
The attack correctly identifies that the gap exists and is not closed. But the attack's bigger claim — that mechanism at 81.77% does not automatically transfer to 96.6% — is a scope concern, not a validity concern. We already state the limitation in §7 (implicitly) and in §6 (explicitly: "does not depend on matching their operating point"). The latter is under-argued.

What we should NOT do is pretend the gap doesn't matter. What we SHOULD do is (a) scope the claim to our operating point explicitly, and (b) note that the Illusion-of-Superposition *phenomenon* (latents not doing work) appears at 81.77% too — our matrix-CODI is within 0.26pp of pure SFT at matched seeds, consistent with Rizvi-Martel's "latents not doing work" finding at a different accuracy ceiling. The phenomenon replicates at our operating point; the mechanism we propose is at our operating point.

Running rank-$k$ ablation on Rizvi-Martel's released checkpoint is an attractive camera-ready experiment but not a submission-blocker for a workshop paper. Their code is public (github.com/michaelrizvi/coconut), so this is tractable.

**Supporting evidence:**
Rizvi-Martel paper (arXiv 2604.06374): "Coconut, a fine-tuned model achieves 96.6% accuracy without any latent tokens" — supports the view that the phenomenon is about latents-not-doing-work, which replicates for us at 81.77% (matrix-CODI 82.03% vs pure SFT 81.77%, gap 0.26pp, within seed noise).

**What goes in the paper if this defense is accepted:**
Rewrite §6 para 2 last sentence as: "The qualitative phenomenon Rizvi-Martel report — fine-tuned latent CoT models reaching comparable accuracy without their latents — replicates in our setting: matrix-CODI at 82.03% vs pure SFT at 81.77% (gap 0.26pp, within three-seed noise). Our mechanism claim is scoped to our operating point; whether the same mechanism operates at their 96.6% ceiling is an open question we flag for camera-ready." Add "rank-$k$ ablation on Rizvi-Martel's released checkpoint" to the Discussion as a planned camera-ready experiment.

---

### A8: "Linear in Z → rank-blind" argument is tautological
**Disposition:** PARTIAL
**Response:**
The attack's technical content is correct: Prop 1 *is* an unpacking of what "linear" means. But the attack's conclusion — "this is not a mechanism" — is too strong. A proposition that identifies a *necessary and sufficient structural condition* for an observed phenomenon is a mechanism, even if the proof is one line of chain rule. What's new in Prop 1 is the connection to the observed flat curves, not the linear algebra.

That said, the attack's real point is that Prop 1 is a *local* statement about gradients and doesn't constrain training dynamics. This is correct. We should soften the proposition statement slightly to match what the proof actually shows.

**Supporting evidence:**
The §3.2 proof sketch says exactly what the attack claims: "the gradient of a linear map does not depend on the input's SVD." This is the derivation, not a hidden mechanism. The paper should be honest that Prop 1 is a structural observation with empirical teeth (it predicts flat curves for every linear-in-$\Z$ readout, and we confirm) rather than a deep theorem.

**What goes in the paper if this defense is accepted:**
Revise §3.2 wording: "Proposition 1 (constant-Jacobian rank-blindness) is a structural observation: if $\phi$ is linear in $\Z$, then the loss gradient through $\phi$ cannot carry basis-dependent information about $\Z$. It is not a deep result — it is a direct consequence of the chain rule applied to a linear map. Its value is predictive: every linear-in-$\Z$ readout should produce a flat rank-$k$ curve. The four training conditions in Table 1 confirm this prediction; §5 tests what happens when the prediction's precondition is violated."

---

### A9: Objective-level claim conflates CODI distillation with any flatten-bottleneck loss
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct and identifies a real overclaim. Our $\gamma=0$ runs (R3a, R3b) have no CODI distillation loss by construction, and the positive controls in §5 also use $\gamma=0$. So "CODI distillation objective is rank-blind" is misnamed — the actual scope is "next-token CE loss backpropagated through a matrix bottleneck in our pipeline." This is a framing fix, not a new experiment.

**Supporting evidence:**
§5 setup line 18: "$\gamma=0$" — all positive controls run without CODI distillation loss. R3a, R3b in Table 1 are $\gamma=0$. KILL_LIST.md: "L_kd (distillation loss) is NOT the structural bottleneck. Round 3 Run 1 with gamma=0 still flat."

**What goes in the paper if this defense is accepted:**
Global find-replace: "CODI distillation objective" → "matrix-bottleneck training objective" (or "cross-entropy loss through the matrix bottleneck"). In the abstract, §1 contribution 3, and §5.3. Add one sentence to §5 setup: "Note that these positive controls use $\gamma=0$; the flat rank-$k$ curves therefore hold for the cross-entropy loss alone, without the L1-at-colon distillation term. We retain the 'CODI' branding for the stack (L1-at-colon, teacher/student, latent feedback) but the mechanism claim applies to the CE-through-bottleneck objective specifically."

---

### A10: Statistical power insufficient at n=128
**Disposition:** PARTIAL
**Response:**
The attack is correct that $n=128$ has limited power (~80% at $|r_s|=0.25$, <20% at $|r_s|=0.10$). "Absence of evidence is not evidence of absence" is fair. But the attack's own framing concedes the point: "non-corrected minimum is 0.14, already well above corrected thresholds, so the multiple-comparisons problem doesn't bite here." And crucially, the paper's evidence is NOT just the Spearman $p$-values — it is the $\leq 0.8$pp range across $k$, across seven training conditions. At $n=128$ with accuracy ≈ 79%, the one-sample binomial 95% CI on accuracy is ±7pp — but the *within-curve* variation across $k$ shares the same test problems and is therefore much tighter. The flat-curve pattern is what it looks like.

The fix is easy: run rank_eval on the full ProsQA test set (500 problems) instead of 128. This is inference-only, ~10 minutes per checkpoint. We should do this before submission.

**Supporting evidence:**
ProsQA test set has 500 problems (paper §2 notes "128 test problems" as the standard CODI eval split — a convention choice, not a statistical necessity). KILL_LIST entries and summary tables suggest full 500 evaluation was done for earlier rounds (Round 2: "Rank-k Projection Ablation (Run C)" reports on 500).

**What goes in the paper if this defense is accepted:**
(a) Re-run the four positive-control rank_evals on the full 500 ProsQA test problems; report accuracy and Spearman CI at $n=500$. (b) Add one sentence to §5: "At $n=500$, our power to detect a rank-accuracy correlation of $|r_s|\geq 0.15$ at $\alpha=0.05$ is $\approx 80\%$." (c) Report 95% bootstrap CIs on the accuracy at each $k$ rather than point estimates.

---

### A11: Missing citations to 2026 competing work
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct on three of four missing citations. (1) arXiv:2510.15522 "LLM Latent Reasoning as Chain of Superposition" (Oct 2025): confirmed, abstract proposes latent reasoning IS superposition in vocab column space (Latent-Vocab / Latent-SFT framework). This is a competing substrate — if superposition lives in vocab space, rank-$k$ on $\Z$ wouldn't probe it. We must cite and engage. (2) arXiv:2411.07635 RALA (CVPR 2025): confirmed, shows rank augmentation is practically achievable in linear attention via specific architectural interventions — weakens our "no nonlinear readout helps" universality claim. Should cite as a caveat on the family-tested. (3) arXiv:2409.09951 Li & Janson "Optimal ablation for interpretability" NeurIPS 2024: confirmed — zero-ablation substantially overestimates vs optimal ablation (11.1% ratio). Cuts both ways (optimal ablation would make our curves *flatter*), but is the right methodological citation for rank-$k$ truncation as an instance of zero-ablation.

For arXiv:2602.08783 (anonymous2026dynamics), we already cite it but dismissively; the attack is right that it deserves fuller engagement.

**Supporting evidence:**
All four arXiv IDs verified via WebSearch. arXiv:2510.15522 abstract on HuggingFace/OpenReview explicitly frames latent reasoning as "superposition over vocabulary probabilities" via Latent-Vocab constraint.

**What goes in the paper if this defense is accepted:**
Add one Related Work paragraph (~6 lines) engaging arXiv:2510.15522 as a competing substrate claim (vocab-space superposition that our rank-$k$ probe does not test), cite arXiv:2411.07635 as architectural evidence that rank-augmentation is achievable *for a different task family*, and cite arXiv:2409.09951 as methodological caveat on rank-$k$ truncation as zero-ablation. Expand the anonymous2026dynamics citation from one sentence to a paragraph engaging their step-wise intervention protocol as a distinct probe axis.

---

### A12: GSM8K-Aug at 6% accuracy shouldn't count toward "four flat curves across two tasks"
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct. At 6% accuracy the GSM8K-Aug model is essentially at chance and its flat rank-$k$ curve is not interpretable. §7 admits this; the abstract does not carry the caveat. Either we re-run GSM8K-Aug at a higher operating point before submission (the paper says this is "queued") or we fix the abstract's count.

The cheap fix is the framing fix: the headline is "three flat ProsQA curves across two distillation weights and a thinker on/off ablation, plus four flat positive-control curves" — seven total on ProsQA, not four across two tasks. That is a stronger, honest claim.

**Supporting evidence:**
§7 explicit caveat: "a low-accuracy operating point (6%) where the model is barely learning the task; we do not interpret its flat rank-$k$ curve as strong evidence on its own."

**What goes in the paper if this defense is accepted:**
(1) Abstract: replace "Across four training regimes of a matrix-CODI model on ProsQA" (current wording actually says ProsQA — good) — verify. Check §1 contribution 1 which says "across two tasks, two distillation weights, and a thinker on/off ablation" — change to "across two distillation weights and a thinker on/off ablation on ProsQA, plus a replication on GSM8K-Aug at a below-threshold operating point which we flag." (2) Drop R1 from Table 1 main body, move to appendix with explicit caveat. Or retitle the table "four flat curves on ProsQA and one at-chance curve on GSM8K-Aug."

---

### A13: Depth sweep is broken — only one data point reported
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct. §4.1 has $n=6$ reported and $n\in\{16,32,64\}$ pending. A single data point is not a sweep and cannot support "depth doesn't help." We have two options: (a) complete the $n\in\{16,32,64\}$ re-runs before submission (each is ~3h on 1×H100 at reduced batch, so ~10h total, tractable), or (b) remove §4.1 from the main body and move to "future work" in §7.

Preferred: (a). If compute doesn't allow, (b) is acceptable and the paper's core claim (rank-blindness) does not depend on the depth sweep.

**Supporting evidence:**
Figure 4 caption literally says: "The $n=16,32,64$ points are partial data and will be added in the appendix when the re-runs complete." This is a submission-grade flag, not defensible in a hostile review.

**What goes in the paper if this defense is accepted:**
Preferred: launch $n\in\{16,32,64\}$ re-runs today at batch=8/4/2 (queue ~10h H100 compute). If results arrive before submission, update figure. If not, demote §4.1 to a paragraph: "We ran vanilla CODI at $n=6$ and found it underperformed pure SFT by ~3pp; larger $n$ runs are in progress and will be reported in the camera-ready. This paragraph is descriptive only and is not part of the paper's evidence base."

---

### A14: Scale-sweep with n=2 scales (large OOM'd)
**Disposition:** PARTIAL
**Response:**
The attack is correct that two scales within seed noise is thin. But the claim "matrix-CODI underperforms at every scale" is also honest: the gap is $-1.3$pp at small and $-0.78$pp at medium, both same-sign, both within seed noise. This is not a strong claim — we should present it as "consistent with underperformance, not distinguishable from tie at seed noise." That is what we currently write ("tracks below"), but we should make the "within seed noise" caveat explicit.

For the gpt2-large OOM, the honest fix is to remove the pending row from the figure and say "gpt2-large pending, will be in camera-ready." We should NOT pretend the sweep is complete.

**Supporting evidence:**
Three-seed std at gpt2-small is ±1.2pp. Observed gap at small is $-1.3$pp. So the gap is barely outside one-sigma seed noise.

**What goes in the paper if this defense is accepted:**
§4.2 revised: "Across two backbone scales (small and medium), matrix-CODI's best accuracy is below its matched vanilla SFT baseline by 1.3pp and 0.78pp respectively. Both gaps are within the three-seed standard deviation of ±1.2pp at gpt2-small, so we cannot claim a significant underperformance at any individual scale; the consistent sign across two scales is suggestive but not conclusive. Matrix-CODI at gpt2-large OOM'd at batch 4 and 2 and is pending."

---

### A15: Proposition 1 not tight — doesn't address implicit-regularization dynamics
**Disposition:** CONCEDE + FIX (minor)
**Response:**
Agreed; Prop 1 is a local gradient statement, not a global-dynamics statement. The implicit-regularization literature (Gunasekar 2017, Arora 2019, Razin-Cohen 2020, Kobayashi 2024) shows gradient descent on matrix factorizations has implicit low-rank bias. Our Prop 1 as written does not account for this, and our three-seed data (two seeds at rank ~12, one at rank ~4) is mildly consistent with "weak low-rank attractor plus seed-dependent escape." Acknowledging this in §3.2 and connecting to the implicit-bias literature is the right fix.

**Supporting evidence:**
Gunasekar et al. 2017 (arXiv:1705.09280), Arora et al. 2019 (arXiv:1905.13655), Kobayashi et al. 2024 (arXiv:2410.23819) — latter is directly relevant: weight decay induces low-rank attention products. Under AdamW at default wd=0.01, we would expect *some* low-rank bias on $W_{\text{up}}$ and $W_{\text{down}}$.

**What goes in the paper if this defense is accepted:**
Revise §3.2 Proposition statement to: "gradient descent on $\mathcal{L}(\phi(\Z))$ with $\phi$ linear in $\Z$ carries no gradient term that prefers one rank of $\Z$ over another through the loss. Implicit regularization from the optimizer (Adam + weight decay) or the upstream parameter space may still shape rank through channels outside $\mathcal{L}$." Add this to the related work paragraph per A19.

---

### A16: Simpler explanation — ProsQA is rank-1-solvable
**Disposition:** PARTIAL
**Response:**
The attack raises a legitimate alternative explanation: if ProsQA is rank-1-solvable, every architecture would land on a rank-1 solution, and we would observe exactly our data. This is a version of "the task is too easy to test rank." We cannot rule it out from our current data. We should acknowledge it as the primary alternative explanation in §7.

However, note one piece of evidence the attack overlooks: across three seeds, the TRAINED $\Z$ has effective rank 4, 12, and 13 — **not** rank 1. If the task required rank-1, we would expect training to converge to rank-1 matrices, not rank-13 matrices that are then truncatable to rank-1 without loss. The observation that $\Z$ is trained to high rank but only the top-1 direction is functional is consistent with "task is rank-1-solvable AND the model learns extra capacity as wasted decoration" — which is the paper's claim, not a competing hypothesis.

A cleaner falsification would require a task where ground-truth reasoning provably requires $k>1$ independent quantities. KILL_LIST flags this as an open direction; we should defer to camera-ready.

**Supporting evidence:**
rank_evals JSONs: three-seed trained $\Z$ effective rank 4, 12, 13. Not rank 1. The model builds rank it doesn't functionally use.

**What goes in the paper if this defense is accepted:**
Add a §7 paragraph: "An alternative explanation for the flat curves is that ProsQA is rank-1-solvable — the task admits a solution in which all answer-predictive information lives in a single singular direction of $\Z$. Our data are consistent with this alternative. Two observations distinguish it weakly from the 'objective does not reward rank' reading: (i) the trained $\Z$ reaches effective rank 12–13 at $\gamma=0$, far above rank 1 — so even if the task is rank-1-solvable, the model is building capacity it does not use; (ii) three seeds land at rank 4, 12, 13 rather than concentrating at rank 1, which is inconsistent with an implicit rank-1 attractor. A fully disambiguating experiment requires a task whose ground-truth solution provably needs $k>1$ independent quantities; we do not have such a task at this scale and flag it as future work."

---

### A17: Reproducibility — one seed per positive control
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct. Four positive-control variants at one seed each is thin. Each variant is ~3.5h on 1×H100 so three seeds per variant is 12×3.5 = 42 GPU-hours, ~$80. This is small and should be run before camera-ready. For submission, we acknowledge the limitation (we already do in §7) and the per-variant numbers (79.69% appearing twice is real, not a rounding artifact — both Quadratic and Bilinear+GELU land at 102/128 on the eval set).

**Supporting evidence:**
Quadratic reports $79.69\% \approx 102/128$; Bilinear+GELU $79.69\% \approx 102/128$. The match is two models landing on identical correct/incorrect counts on the same 128-problem eval set. This is plausible at the scale of the test set, not a rounding artifact.

**What goes in the paper if this defense is accepted:**
Keep the §7 limitation. Add to Discussion: "We have launched a three-seed replication for each positive-control readout and will include CIs in the camera-ready."

---

### A18: "One architecture family" limitation inconsistent with architecture-agnostic mechanism claim
**Disposition:** PARTIAL
**Response:**
The attack correctly notes tension between "Prop 1 is architecture-agnostic in principle" and "we only ran GPT-2." We should resolve this by scoping the mechanism claim to what we tested. Prop 1 is a linear-algebra observation that applies to any linear map on any backbone; the *empirical* mechanism claim (that $\gamma=0$ matrix-bottleneck training produces rank-indifferent dynamics) is scoped to GPT-2. We should state both cleanly.

**Supporting evidence:**
§7: "The structural mechanism (Proposition~\ref{prop:jac}) is stated in terms of the readout $\phi$ and is therefore architecture-agnostic, but our empirical evidence covers only GPT-2." This is already correct; it just needs to be surfaced into the abstract/intro.

**What goes in the paper if this defense is accepted:**
Add to §1 final paragraph: "Our structural argument (Prop 1) is a property of linear maps and applies to any backbone. Our empirical evidence is scoped to GPT-2 {small, medium, large}. Whether the 'refined mechanism' — that the optimizer finds rank-1-active-subspace solutions in any CODI-trained matrix bottleneck — holds on non-GPT-2 backbones (Pythia, LLaMA-family) is an open question." A Pythia-160M run is ~4h on 1×H100 and would materially strengthen the paper for camera-ready.

---

### A19: Implicit-regularization literature contradicts "rank is a free direction"
**Disposition:** CONCEDE + FIX
**Response:**
The attack is correct and this is the most important missing-citation issue. Gunasekar, Arora, Razin-Cohen, Galanti, Kobayashi are all well-established and directly relevant. Our "rank is a free direction" framing is in mild tension with a decade of established theory predicting low-rank bias under GD+wd on matrix factorizations. The resolution is straightforward: we're not claiming rank is uniformly distributed; we're claiming the loss gradient carries no rank-shaping signal. Implicit bias from the *optimizer* (Adam + weight decay) operates through separate channels and can still shape rank. Our three-seed data (two at rank ~12, one at rank ~4) is *consistent* with implicit bias plus seed-dependent convergence.

**Supporting evidence:**
Kobayashi et al. 2024 NeurIPS (arXiv:2410.23819): weight decay induces low-rank attention via nuclear-norm regularization equivalence. We use AdamW at wd=0.01 throughout, so this applies to $W_{\text{up}}$ and $W_{\text{down}}$. Our effective rank of trained $\Z$ ~12 out of max 16 is broadly consistent with partial low-rank collapse — not rank 1, but not full rank either.

**What goes in the paper if this defense is accepted:**
Add one Related Work paragraph: "Implicit low-rank bias of gradient descent on matrix factorizations is a well-established line (Gunasekar et al. 2017, Arora et al. 2019, Razin \& Cohen 2020, Kobayashi et al. 2024). These results concern the optimizer's bias through the parameter space; our claim concerns the *loss gradient* through the readout. The two are compatible: our three-seed data (rank 4, 12, 13 at matched accuracy) is consistent with implicit bias plus seed-dependent convergence, but is inconsistent with strong low-rank collapse (which would predict all three seeds at the same low rank). A richer treatment of which optimizer channel is shaping $\Z$'s effective rank is deferred to future work."

---

### A20: Title overclaims
**Disposition:** PARTIAL
**Response:**
"The Gradient Does Not See Rank" does travel further than our results support. The title is not dishonest — it is the most compact expression of Prop 1 combined with the §5 controls — but a reviewer can legitimately push for more scoping. We should either narrow the title or add a clear subtitle scope.

**Supporting evidence:**
Our actual result: rank-$k$ ablation and Spearman tests are flat/null across seven training regimes on one task (ProsQA) with one backbone family (GPT-2). Not a universal claim about gradients and rank.

**What goes in the paper if this defense is accepted:**
Keep current title ("The Gradient Does Not See Rank") but revise the subtitle from "A Structural Explanation for the Illusion of Superposition in Latent Chain-of-Thought" to "Rank-Indifference in Matrix-CODI Gradients on ProsQA." Alternatively: "Rank Is Not a Functional Observable in Matrix-CODI: Mechanism and Evidence." Let final choice be an editorial call after the rest of the revision lands.

---

## New citations found during defense

- **Kobayashi, Akram, von Oswald 2024** "Weight decay induces low-rank attention layers." NeurIPS 2024. arXiv:2410.23819. Direct evidence that Adam + weight decay drives low-rank bias in attention products; relevant to A15, A19, and our Prop 1 framing.
- **Gunasekar, Woodworth, Bhojanapalli, Neyshabur, Srebro 2017** "Implicit Regularization in Matrix Factorization." arXiv:1705.09280. Foundational implicit-bias-toward-low-rank result; A19.
- **Arora, Cohen, Hu, Luo 2019** "Implicit Regularization in Deep Matrix Factorization." arXiv:1905.13655. Greedy low-rank learning dynamics; A15, A19.
- **Razin, Cohen 2020** "Implicit Regularization in Deep Learning May Not Be Explainable by Norms." NeurIPS 2020. arXiv:2005.06398. Implicit bias is not captured by simple norm regularization; A15, A19.
- **Li, Janson 2024** "Optimal Ablation for Interpretability." NeurIPS 2024. arXiv:2409.09951. Zero-ablation significantly overestimates component importance (11.1% ratio for optimal); methodological caveat for our rank-$k$ truncation in A11.
- **Fan, Huang, He 2024** "Breaking the Low-Rank Dilemma of Linear Attention." CVPR 2025. arXiv:2411.07635. Rank-augmented linear attention (RALA) shows rank augmentation is achievable architecturally in a different context; caveat on our "no nonlinear readout helps" family-tested claim (A11).
- **(Anonymous) 2025** "LLM Latent Reasoning as Chain of Superposition" / "Latent Reasoning in LLMs as a Vocabulary-Space Superposition." arXiv:2510.15522. Competing substrate claim (vocab-column-space superposition) that our rank-$k$ probe does not test (A11).

## Attack ordering note

The attack agent weighted A1 and A2 as CRITICAL. A2 is technically incorrect on its central premise (we use `svdvals`, which has stable backward per PyTorch docs) and is therefore the *least* severe critical. A1 is real but reduces to framing surgery on §5.3 and the abstract — also not critical, more "serious-plus."

The attack **missed** what I think is the actually-most-important weakness: we do not have a non-GPT-2 replication (A18 touches this but rates it MINOR). Cross-backbone replication on a small non-GPT-2 model (Pythia-160M, ~4h H100) would be the single highest-value camera-ready experiment. The attack also underweights A6 (probe dim confound) — this is a real, easy-to-fix issue that a careful reviewer WILL catch and that undermines one of our four headline contributions. A6 should have been SERIOUS-to-CRITICAL, not SERIOUS.

The attack's overweighting is on A3, A4, A8, A15, A20 — these are editorial/framing concerns, not evidence concerns, and reviewers will generally accept paragraph-level softenings without a second round.
