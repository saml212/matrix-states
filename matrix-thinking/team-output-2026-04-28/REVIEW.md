# Reviewer 2 — Hostile MI Workshop Review

## Killer objection: methodological circularity, plus the experiment is already in §5 in disguise.

**Recommendation: Reject.** The proposed 3×3 follow-up does not extend the contribution — it constructs a synthetic task whose answer is fixed in advance and a training term that explicitly optimises for the property the authors then claim to "discover." Both halves of the headline are tautological, and the only non-tautological cell of the 3×3 — entropy reward on a task that doesn't need rank — is already covered by the SVD-augmented readout in §5 of the existing paper.

### 1. The headline claim is a tautology dressed up as a mechanism finding.

The proposed headline (RANK_AWARE_RESEARCH_COMPILATION.md, lines 56–60) is: "Training on effective-rank reward causes matrix-CODI to use rank functionally on tasks that need it: rank-k ablation bends on ProsQA-MULTI when trained with entropy reward, stays flat on ProsQA-1 (no need for rank), and stays flat on ProsQA-MULTI without entropy reward." Restated honestly: *we constructed a task that we engineered to require rank k by orthogonality (§A), added a loss term H(σ̃) that explicitly maximises spectral entropy (§B), and observed that the model now uses spectral entropy.* This is the gradient-descent analogue of "we trained the model on label X and found that the model predicts X." A reviewer in the MI community will not credit this as a mechanism claim.

### 2. The "rank-k by orthogonality" lower bound is hand-wavy and almost certainly false in practice.

The §A argument — "k disjoint positive answers needs k orthogonal directions in Z" — assumes the bilinear probes (u_i, v_i) are fixed and orthogonal. They aren't; they're *learned*. A learned readout can share a single dominant singular direction across k targets via aliasing in the unembedding column space, exactly the failure mode Wang 2025 (already cited in §6) describes — vocab-space superposition over a single hidden direction. The compilation explicitly asks (line 156) "could the model still find a rank-1 solution by clever embedding tying?" — yes, almost certainly, and there is no proof in the document that says otherwise. Without a formal lower bound, "ProsQA-MULTI requires rank k" is a hope, not a property of the task.

### 3. The proposed entropy-reward experiment is already subsumed by §5's SVD-augmented readout.

The current paper's §5 constructs a readout `MLP(σ(Z))` that *exposes the singular values directly to the gradient* via differentiable `torch.linalg.svdvals`. That is the cleanest possible way to give the loss a gradient signal through the spectrum, and the rank-k curve was still flat (Spearman p = 0.82). Adding `+λ H(σ̃)` to the loss is a stronger version of the same intervention — it replaces "spectrum can flow through MLP" with "spectrum is directly penalised." If the §5 result generalises (which the authors insist it does, abstract line 17), the entropy reward will spread the spectrum *of Z* without making any individual singular direction *functional*; the model will simply find a flat-spectrum rank-1 solution. The experiment has been pre-empted by the authors' own positive control.

### 4. Missing baselines that would discriminate the claim.

There is no proposed control where ProsQA-MULTI is run *without* entropy reward at *higher capacity* (more latent positions, larger d). If the bend on the headline 3×3 cell comes from "task got harder, ablation hurts more" rather than "rank now functional," the claim collapses. There is also no probe-based check that the bent rank-k curve corresponds to k *reasoning paths* rather than k *output channels* — without that, the bend is consistent with the model storing k independent answer logits in k singular directions, which is bookkeeping, not superposition.

---

## Rebuttal direction (what would convert reject → weak accept).

To recover, the authors must (a) prove or empirically demonstrate that ProsQA-MULTI-k cannot be solved by any rank-1 Z under the realised readout — e.g., by exhibiting a Frobenius-norm gap between the optimal rank-1 and rank-k solutions of the trained probe pairs, on at least one held-out instance — and report the gap *before* training under entropy reward. Without this, the lower-bound claim is unsupported. (b) Add a control cell where ProsQA-MULTI is trained with the standard loss but at a higher latent budget (d=24 or n_latent=12) so that "harder task hurts ablation" is ruled out. (c) Decode the singular directions of Z post-training under entropy reward and show, via causal patching, that singular direction i carries *the i-th answer's information* and not "answer i's logit-space embedding." Patching σ_i u_i v_i^⊤ from a problem where target i = A into a problem where target i = B should flip prediction i specifically, not all k predictions. (d) Reframe the contribution as "we identify an objective term that is necessary, not merely sufficient, for spectral functionality" — i.e., show the mirror-image experiment that breaks the entropy-reward result, e.g. by removing the bilinear separation between targets while keeping the entropy term. Without a falsification arm, this is still confirmation-bias engineering. With (a)–(d), the contribution becomes a genuine objective-vs-task-vs-readout factorial that the field can build on, and the paper crosses the bar.
