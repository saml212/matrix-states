# Attack report on matrix-CODI rank-blindness paper

## Summary for defense agent

This is a careful, honest negative-result paper whose *empirical work* is real but whose *structural claim* is significantly overreaching relative to the evidence. Its most-cited strength — a "falsifiable mechanism claim" about the CODI distillation objective adjudicated by four positive controls — is weaker than advertised: three of the four positive controls have technical problems that keep the readout-Jacobian argument from being properly falsified, the statistical power for the probe (n=128, p-values 0.14–0.82) is too low to turn a null result into a structural claim, and the post-hoc shift from "readout Jacobian is the culprit" (§3) to "objective is rank-blind through the full chain rule" (§5) is an unfalsifiable dodge rather than a new testable claim. Additional serious issues: n=3 seeds with effective-rank computed from a smooth entropy proxy is not strong evidence of "seed decoupling"; the probe AUC comparison has a capacity confound (256-dim matrix bottleneck vs. 768-dim vanilla hidden state); the paper's vanilla SFT baseline is 15pp below Rizvi-Martel on the same task and same backbone, undercutting the whole "we add to the Illusion of Superposition" framing; and the contribution relative to Feb 2026 adjacent work is narrower than claimed. Verdict: **strong revision required**. The negative-result empirical summary is publishable with scope narrowed; the "objective-level rank-blindness" framing is not.

## Attacks (numbered, sorted by severity)

### A1: The post-hoc pivot from "readout Jacobian" to "objective is rank-blind" is an unfalsifiable move that the paper advertises as the opposite.
**Severity:** CRITICAL
**Type:** methodological / claim-scope

**Attack.**
The paper's central selling point is that it makes a *falsifiable* mechanism claim and tests it with four positive controls (§5). But look at what actually happens. §3 introduces Proposition 1: "readouts linear in $\Z$ have $\Z$-independent Jacobian, therefore rank-blind." The prediction is cleanly falsifiable: nonlinear-in-$\Z$ readouts should bend the rank-$k$ curve. §5 runs that test and the curves are flat. §5.3 then concludes: "Proposition 1 remains correct as a sufficient condition; it is not necessary. … the optimizer can find solutions where the active subspace of the readout's nonlinear Jacobian lies along a rank-1 direction of $\Z$." This is not a falsified hypothesis replaced by a new falsifiable one — it is a **shift to a claim about what the optimizer "can" do** that is consistent with any observed curve, flat or bent. A rank-bent curve could also be explained as "the optimizer happened not to find the rank-1 shortcut this time." The §5.3 reasoning is post-hoc rescue, not a new prediction.

The abstract and contribution list (intro §1) sell this as "the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule." A reviewer will correctly note: that is not a mechanism claim; it is a restatement of "we observed flat curves in every experiment we ran" plus a speculative attribution. The only *mechanism* the paper actually proves is Proposition 1, and Proposition 1 has been (by the authors' own admission) falsified as a complete explanation. The paper needs to either (a) state a new falsifiable mechanism that explains why nonlinear readouts still produce flat curves and test it, or (b) honestly downgrade to "we observe rank-blindness across these seven training regimes; we do not yet have a mechanism that explains it."

**Supporting evidence.**
- Direct quote §5.3 lines 112–117: "rank-blindness can hold under weaker conditions: if the optimizer can find solutions where the active subspace of the readout's nonlinear Jacobian lies along a rank-1 direction of $\Z$, then regardless of the readout's in-principle expressiveness, the effective information flow is rank-1."
- Direct quote §5.3 line 117–118: "Proposition~\ref{prop:jac} remains correct as a sufficient condition; it is not necessary."
- Compare with §1 contribution 3: "The simple Jacobian-linearity story is not the full picture: the CODI distillation objective itself produces rank-indifferent gradients through the full chain rule." This is stated as a *contribution* but is not proven anywhere in the paper — it is inferred from four flat curves without a mechanistic argument that distinguishes it from other explanations (e.g., ProsQA is a single-answer task that genuinely needs only rank-1).

**What the paper would need to do to defuse this.**
Rewrite §5.3 and the abstract to either (a) commit to a specific falsifiable follow-up — e.g., "the active subspace of the readout at convergence is rank-1 regardless of readout architecture" — and run the measurement (take trained checkpoints, compute the effective rank of the Jacobian of $\phi$ at the test inputs, report it; this is a 10-line PyTorch script), or (b) retreat to the descriptive claim: "we observe that rank-$k$ ablation curves are flat across the seven configurations we tried. We do not yet have a mechanism that predicts this." The current framing promises falsification-grade epistemics and delivers "the curves were flat again."

---

### A2: The "SVD-augmented" positive control is broken as a test of the Jacobian hypothesis, and the authors know PyTorch's SVD backward is unstable for the relevant regime.
**Severity:** CRITICAL
**Type:** positive-control adequacy

**Attack.**
The SVD-augmented readout feeds $\sigma(\Z)$ through an MLP. §5 footnote: "PyTorch's SVD provides subgradients that route accuracy gradients through the singular spectrum." This is not accurate in the regime of interest. PyTorch's SVD backward has documented instability:

1. The gradient formula involves $1/(\sigma_i - \sigma_j)$. If any two singular values of $\Z$ are close (which is typical for a trained $\Z$ that has not been explicitly rank-constrained), this term explodes or produces NaN.
2. PyTorch's SVD backward is mathematically undefined when multiple singular values are repeated or near zero. The framework's subgradient in this case is not guaranteed to route useful signal.
3. If during training the optimizer drives $\Z$ toward any degenerate spectrum (as one might expect given the claimed rank-indifference), the gradient path through $\sigma(\Z)$ effectively vanishes or is noisy.

The consequence is that the SVD-aug readout does NOT reliably expose singular values to the optimizer. A flat rank-$k$ curve on this variant is consistent with two very different hypotheses: (a) the CODI objective is rank-blind (the paper's claim), or (b) the SVD gradient path is broken and the model learned to ignore it and route everything through the parallel $W_{\text{down}}\vecop(\Z)$ branch (which IS constant-Jacobian). The paper does not distinguish these. Critically, because the SVD-aug readout has BOTH a linear flatten branch AND the MLP(σ) branch ADDED together, the model can simply zero out or attenuate the MLP branch's contribution during training and solve the task through the linear branch alone. That would produce an exactly flat curve and tell us nothing about whether "exposing singular values to the optimizer" would break rank-blindness.

**Supporting evidence.**
- PyTorch issue #49886 (SVD backward with zero eigenvalues): "SVD backward isn't correctly defined when multiple 0 eigenvalues"
- PyTorch forum on numerical stability: "SVD backward operations are only stable when the input is full rank with all distinct singular values"
- TensorFlow issue #17476: "SVD gradient is unstable for non-unique singular values" — same issue, same reason.
- The paper's own §5 description: `$\phi(\Z) = W_{\text{down}}\vecop(\Z) + \operatorname{MLP}(\sigma(\Z))$`. The `+` makes the two branches parallel and separately zeroable.

**What the paper would need to do to defuse this.**
(i) Verify the gradient actually flows through $\sigma(\Z)$ during training. A sanity check: compute $\|\partial \mathcal{L}/\partial \sigma(\Z)\|$ at several training steps and confirm it is not vanishing. Report. (ii) Ablate: train a variant where the linear flatten branch is REMOVED, so the only path from $\Z$ to the loss is through $\operatorname{MLP}(\sigma(\Z))$. If the curve is still flat, the claim holds in this variant. If training fails, that is itself interesting and should be reported. (iii) Acknowledge the PyTorch SVD backward stability issue in the limitations section as a caveat on the SVD-aug evidence.

---

### A3: The "Bilinear+GELU" positive control does not break the Jacobian structure the authors claim.
**Severity:** SERIOUS
**Type:** positive-control adequacy / math

**Attack.**
The Bilinear+GELU readout is $\phi(\Z) = W\operatorname{GELU}(\operatorname{probes}(\Z))$ where $\operatorname{probes}(\Z)_k = u_k^\top \Z v_k$. The probe scores are linear in $\Z$: each is a Frobenius inner product $\langle u_k v_k^\top, \Z\rangle_F$. The Jacobian of $\phi$ with respect to $\Z$ is
$$\frac{\partial\phi}{\partial\Z} = W \cdot \operatorname{diag}(\operatorname{GELU}'(\operatorname{probes}(\Z))) \cdot P,$$
where $P$ is the constant tensor with $P_{k,ij} = u_{k,i}v_{k,j}$. This Jacobian is the constant-rank-structured tensor $P$ multiplied on the left by a data-dependent diagonal. That diagonal only rescales which probes' contributions are "alive" at a given input; it cannot introduce any rank information about $\Z$ that is not already present as a per-probe linear functional.

Concretely: the set of directions in $\R^{d\times d}$ that $\phi$ is sensitive to at any particular $\Z$ is still spanned by $\{u_k v_k^\top\}$. These are rank-1 matrices. Their span is at most $K = d^2$ rank-1 matrices; no GELU nonlinearity on their scalar scores can increase the ambient rank of this span. Put differently, $\phi$ is "locally linear in $\Z$" — its Jacobian depends on $\Z$ only through scalar gates, not through directional information about $\Z$'s SVD.

This matters because the paper's Proposition 1 argues rank-blindness comes from "Jacobian constant in $\Z$." Bilinear+GELU has a Jacobian that is *rescaled* in $\Z$ but not *reshaped* in $\Z$. That is a much weaker violation of Proposition 1 than the reader is led to believe by "making $\phi$ \emph{nonlinear} in $\Z$." The flat curve on Bilinear+GELU is therefore NOT strong evidence that "nonlinearity in $\Z$ is not enough"; it is evidence that *this specific nonlinearity*, which only gates probe magnitudes, is not enough. A more honest positive control would use a readout whose Jacobian *direction* in $\Z$ depends on $\Z$ — e.g., $\phi(\Z) = W\operatorname{GELU}(\Z^2 v)$ where $\Z^2$ changes the bilinear form.

The paper's own KILL_LIST.md (Lesson 1) actually makes this point internally but the paper does not: "Any operation that is linear in Z (or that factors through a linear map on a fixed-size space derived from Z) is rank-blind by linear algebra." Bilinear+GELU's probe stage is a linear map on $\Z$, and the GELU acts on its output. The paper is relying on readers not to re-derive this.

**Supporting evidence.**
- The paper's own KILL_LIST.md identifies this exact issue as "Lesson 1" (in the authors' internal reasoning, not in the paper).
- Standard result: if $f(x) = g(Ax)$ for linear $A$ and pointwise $g$, then $\partial f/\partial x = A^\top \operatorname{diag}(g'(Ax))$, whose column span is exactly the span of rows of $A$ regardless of $g$.

**What the paper would need to do to defuse this.**
Either (a) run a genuinely rank-sensitive readout as a positive control — $\phi$ whose Jacobian's column space in $\R^{d\times d}$ changes direction with $\Z$, e.g., a gated attention over singular vectors of $\Z$ with the gates depending on $\Z$'s singular vectors (hard to implement but is what the Proposition-1-violating regime requires), or (b) narrow the claim to "readouts whose Jacobian is $\Z$-constant up to scalar gating" are rank-blind, and acknowledge that the category of positive controls the paper tested is smaller than "nonlinear in $\Z$" suggests.

---

### A4: The "Quadratic" positive control discards half the structure of $\Z$ and is therefore rank-blind by construction, not by objective.
**Severity:** SERIOUS
**Type:** positive-control adequacy / math

**Attack.**
The Quadratic readout is $\phi(\Z) = W_{\text{down}}\vecop(\operatorname{concat}(\Z\Z^\top, \Z^\top\Z))$. This is linear in the pair $(\Z\Z^\top, \Z^\top\Z)$ but those two second-moment matrices have the SVD $\Z = U\Sigma V^\top \Rightarrow \Z\Z^\top = U\Sigma^2 U^\top$ and $\Z^\top\Z = V\Sigma^2 V^\top$. The concat therefore encodes $U$, $V$, and $\Sigma^2$ — it preserves all of $\Z$'s singular structure up to sign ambiguity in $(U, V)$.

So far so good. BUT: the readout is then $W_{\text{down}}\vecop(\cdot)$ applied to this concat. That is a linear map from $\R^{2d^2}$ to $\R^D$. The Jacobian of this map with respect to the second-moment tensor is a constant matrix $W_{\text{down}}$. The Jacobian with respect to $\Z$ is $W_{\text{down}} \cdot \partial(\text{concat}(\Z\Z^\top, \Z^\top\Z))/\partial \Z$, where the second factor IS $\Z$-dependent but has a specific structure: it is linear in $\Z$ (differentiating a quadratic gives a linear form). So $\partial\phi/\partial\Z_{ij} = \text{linear}(\Z)$, not constant — formally this violates Proposition 1.

But here is the problem: the resulting gradient signal $\partial\mathcal{L}/\partial\Z$ depends on $\Z$, but the *structural* information this gradient contains about $\Z$ is only whatever structure the Gram matrices preserve. Crucially, because the second-moment tensor pair $(\Z\Z^\top, \Z^\top\Z)$ is rank-invariant under any sign flip of a singular direction (SVD of $\Z$ is identified only up to signs), the readout is insensitive to at least $d$ sign bits of $\Z$. More importantly, the Quadratic readout has the same "rank-1 shortcut" property as every other one: the model can place all task-relevant information in the top singular direction of $\Z$, concentrating it in $\sigma_1^2 u_1 u_1^\top$, which dominates both Gram matrices, and the readout will see essentially the rank-1 truncation. A rank-1 $\Z$ is a fixed point of the readout's information pathway.

This is the same "rank-1 collapse is available, optimizer will take it" argument that Lesson 2 in KILL_LIST.md already identifies — and that appears nowhere in the paper. The paper presents Quadratic as "nonlinear-in-$\Z$, therefore Jacobian not constant, therefore positive control for Prop 1." In fact, Quadratic is nonlinear-in-$\Z$ but still admits a rank-1 fixed point for the loss; a flat curve on Quadratic is not Prop-1-falsifying evidence but expected if the task is rank-1-solvable.

**Supporting evidence.**
- The KILL_LIST.md entry for killed proposal #5 (which is essentially the Quadratic readout) contains the exact argument: "even though $\operatorname{tr}(A\cdot\Z\Z^\top)$ is quadratic in $\Z$, the readout is linear in $M = \Z\Z^\top$. M lives in the symmetric matrix space. The model can trivially place all useful information in $M$'s top eigenvalue (= top singular vector of $\Z$), reproducing the rank-1 collapse."
- Fact: the map $\Z \mapsto (\Z\Z^\top, \Z^\top\Z)$ loses the relative signs of singular directions. $(-\sigma_1, -\sigma_2, ...)$ maps to the same Gram pair as $(\sigma_1, \sigma_2, ...)$.

**What the paper would need to do to defuse this.**
Acknowledge in §5 that Quadratic has a rank-1 fixed point in its output pathway, and adjust the interpretation: "Quadratic's flat curve is consistent with either (i) the objective is rank-blind or (ii) the task has a rank-1 solution that the readout can express." The paper cannot claim Quadratic's flat curve as evidence for (i) without ruling out (ii) — and the 81.77% vanilla SFT ceiling suggests ProsQA is very close to a rank-1-solvable task.

---

### A5: The three-seed decoupling is the paper's strongest claim but $n=3$ with a smooth entropy-based rank measure is not adequate.
**Severity:** SERIOUS
**Type:** statistical / measurement

**Attack.**
§3.3 presents the three-seed replication (seeds 1337, 42, 7) as "the cleanest single piece of evidence in the paper." Accuracies are 80.47%, 81.25%, 82.81% (range 2.34pp); reported effective ranks are ~13, ~4, ~12. This is a striking result *if* it is a real spread in the underlying rank structure and not a measurement artifact.

Two problems.

(a) **The effective-rank measure is smooth and threshold-free.** The paper uses $\operatorname{erank}(\Z) = \exp(-\sum_i \tilde\sigma_i \log \tilde\sigma_i)$ with $\tilde\sigma_i = \sigma_i/\sum_j\sigma_j$. A single slightly-larger top singular value can push erank from 12 to 4 without any actual change in the number of *functionally independent* directions. Seed 42's $\Z$ could have singular spectrum $(10, 1, 1, ..., 1)$ giving erank ≈ 4 by the entropy formula, while seed 1337 could have $(2, 2, ..., 2, 1)$ giving erank ≈ 13 — with the two matrices carrying essentially the same information content because the seed-42 top-1 direction is the dominant-information direction and seeds 1337/7 spread information across many directions. The paper does not test this: it does not report the raw singular spectrum for each seed, it does not compute a hard-rank-at-threshold (e.g., number of $\sigma_i > 0.01\sigma_1$), and it does not report accuracy under rank-1 truncation per seed to see if seed 42 (claimed erank 4) behaves identically under truncation to seed 1337 (claimed erank 13). If seed 42 truncation to $k=1$ drops more than seed 1337 does, the "decoupling" breaks.

(b) **$n=3$ is low for a claim about "arbitrary rank."** Three seeds with a 3× spread in erank could be noise around a mean — the paper does not report standard error or the full distribution. The paper explicitly claims (§7) "the final rank of $\Z$ in a trained matrix-CODI is essentially arbitrary, governed by initialization noise." That is a distributional claim (the rank has no preferred value) that cannot be supported by three draws. It is consistent with three draws from a distribution peaked at, say, 12, with one outlier at 4 that would regress to the mean with $n=30$.

**Supporting evidence.**
- Standard references on effective rank (Roy & Vetterli 2007, "The effective rank: A measure of effective dimensionality") note that the entropy-based measure is a smooth summary sensitive to the full spectrum; sharp rank claims need threshold-based measures.
- NumPy's default rank tolerance is $\max(M,N)\cdot\epsilon\cdot\sigma_{\max}$; the paper does not report what threshold would recover the claim of "rank 4 vs rank 13."

**What the paper would need to do to defuse this.**
(i) Report the full singular spectra of $\Z$ for all three seeds in an appendix figure (log-scale plot). (ii) Report hard rank at a set of thresholds ($\sigma_i > \tau\sigma_1$ for $\tau \in \{10^{-1}, 10^{-2}, 10^{-3}\}$). (iii) If the claim is distributional ("rank is arbitrary"), run at least $n=10$ seeds. (iv) Weaken the claim to "three seeds at the same config produce very different effective ranks" and acknowledge that three seeds is too few to claim the distribution is flat or arbitrary.

---

### A6: The linear probe comparison has a capacity confound that the paper does not address.
**Severity:** SERIOUS
**Type:** methodological

**Attack.**
§3.4 compares a linear probe on the matrix thought $\Z$ (AUC 0.673) to a linear probe on the vanilla pretrained GPT-2 hidden state at the same prompt (AUC 0.846). The paper concludes: "the matrix bottleneck actively loses target-predictive information relative to the uncompressed hidden state." This is presented as a strong negative-evidence point in §1 (contribution 4) and §8 (conclusion).

Problem: $\Z \in \R^{16\times 16} = \R^{256}$ while the vanilla GPT-2 hidden state is $\R^{768}$. The two linear probes have different input dimensionalities. Linear probes' AUC is strictly monotonic in feature dimension for a fixed downstream task (more features → at least as much expressive power, up to the regularization term), and the 3× capacity gap is a trivially expected source of AUC difference.

To isolate the "compression loses information" claim, the paper needs one of:
- A probe on the 768-dim *reconstructed* hidden state $W_{\text{down}}\vecop(\Z) \in \R^{768}$ that the matrix bottleneck actually outputs to the residual stream. This is the fair comparison because it is what the downstream transformer sees.
- A probe on a 256-dim *truncation* or *PCA projection* of the vanilla GPT-2 hidden state, matching dimensionality.
- A probe on the 768-dim hidden state *of the trained matrix-CODI model* at the same post-bottleneck position, to see if the bottleneck-reconstructed state has reduced information content relative to both the pretrained vanilla hidden state AND the matrix-CODI's un-bottlenecked positions.

The paper does the concat of Z[0..5], which is 6 × 256 = 1536 dim — MORE dim than vanilla hidden state at 768 — and still only reaches 0.673. That is the comparison that would actually support "bottleneck loses info." But the headline 0.673 is Z[all concat] at 1536 vs vanilla at 768, which is now a capacity comparison in the opposite direction and the paper's framing is "vanilla beats matrix despite having one-third the features." That weakens the claim further: the matrix bottleneck, even with 2× the feature count concatenated across positions, does not match vanilla.

**Supporting evidence.**
- Standard result: for L2-regularized logistic regression, AUC is monotonic in the feature space (no information is ever removed by adding features).
- The paper's own Table 5 (EXPERIMENT_LOG line 1283): "matrix Z[all concat] 0.673" — this is concat across 6 latent positions × 256 dim = 1536 feature dim, vs. vanilla hidden state at 768.

**What the paper would need to do to defuse this.**
Add a single-position comparison with matched dim (Z[1] at 256 dim vs. a 256-dim projection of vanilla hidden state at the same prompt). Run a probe on the post-bottleneck reconstructed 768-dim hidden state. Update the claim to reflect the actual comparison: either "at matched feature dim, $\Z$ carries less target info than a vanilla hidden state" (if true) or "$\Z$ concatenated across 6 positions with 1536 features carries less target info than a single 768-dim vanilla hidden state." Both are informative but neither is "the matrix bottleneck actively loses information" as currently phrased — that requires an apples-to-apples dim-matched test.

---

### A7: Our vanilla SFT baseline is 15pp below Rizvi-Martel's. The paper dismisses this but cannot claim mechanism generality without addressing it.
**Severity:** SERIOUS
**Type:** claim-scope / baseline adequacy

**Attack.**
§6 Related Work paragraph 2: "our vanilla SFT baseline at gpt2-small reaches only $81.77\%$, roughly $15$pp below [Rizvi-Martel's 96.6%]. What we add to their picture is a mechanism that operates at the architecture level and does not depend on matching their operating point." This is the only place the gap is discussed. Its dismissal is unearned.

A reviewer will push back hard. Rizvi-Martel use the same backbone (GPT-2 small), the same task (ProsQA), and report 96.6% on fine-tuned COCONUT *without latents*. The authors' vanilla SFT is 81.77%. The 15pp gap means one of (a) hyperparameter/prompt-format/data differences that materially change the regime, (b) the fine-tuning recipe they replicate is not the relevant one, or (c) their training pipeline has a bug or an optimization issue. In any of these cases, the mechanism claim about CODI rank-blindness at 81.77% does not automatically transfer to the regime where the task is being solved at 96.6%.

Specifically, at 81.77% the model is not reaching the task's ceiling, meaning there is headroom that could be consumed by additional representational capacity. It is plausible that in the underperforming regime, the model *happens* to find a rank-1 shortcut because it hasn't been pushed into the harder part of the task; in the 96.6% regime, higher-rank structure might be necessary for the remaining 15pp of performance. The paper's "operates at the architecture level and does not depend on matching their operating point" is an assertion, not an argument.

This directly threatens the paper's positioning. The authors explicitly cite "The Illusion of Superposition" as the empirical phenomenon they are providing a mechanism for. If they are not in the same regime as the phenomenon, they are providing a mechanism for a different phenomenon — "matrix-CODI at 81.77% operating point is rank-blind" — which is a much narrower and less interesting claim.

**Supporting evidence.**
- Rizvi-Martel (arXiv:2604.06374): 96.6% on ProsQA with fine-tuned COCONUT sans latents.
- The paper: 81.77% vanilla SFT (three-seed mean), 82.03% matrix-CODI best.
- EXPERIMENT_LOG.md line 1165-1167: "Our 81.77% vanilla SFT on ProsQA is ~15 percentage points below the 96.6% reported by Rizvi-Martel et al." and "We did not close this gap."

**What the paper would need to do to defuse this.**
Either (a) close the gap by finding the hyperparameter/prompt/training difference (Rizvi-Martel's repo is public; there is a known delta), or (b) report the rank-$k$ ablation ON Rizvi-Martel's released checkpoint (or a re-trained copy in their regime) to show the same rank-blindness holds at 96.6%. Option (b) would materially strengthen the paper. Currently neither is done; the paper just asserts the gap doesn't matter.

---

### A8: The "linear in Z → rank-blind" argument is tautological, not a mechanism.
**Severity:** SERIOUS
**Type:** theoretical / claim-scope

**Attack.**
Proposition 1 (§3.2) says: if $\phi$ is linear in $\Z$, then $\partial\phi/\partial\Z$ is constant, therefore "the optimizer cannot use the singular structure of $\Z$ as a training signal through $\phi$." But this is a restatement, not a derivation. The rank of $\Z$ is a property defined by $\Z$'s SVD. A linear function of $\Z$ is exactly a function that does not depend on the SVD basis except via the (basis-free) Frobenius inner product. Saying "a linear map is basis-independent" and "a linear map cannot carry rank information" is the same statement phrased two ways. The proposition "proves" the following: for any linear $\phi$, gradient descent on $\phi(\Z)$ does not uniquely pin down $\Z$'s SVD among $\Z$s that produce the same image. This is just the rank-nullity theorem.

This matters because the paper's contribution is sold as "we provide a mechanism." The "mechanism" is (i) linear map has rank-null kernel, (ii) therefore gradient descent through a linear map cannot prefer any point in the kernel over any other. Step (ii) is true by definition of gradient descent. Step (i) is the definition of linear. There is no content to Proposition 1 beyond "linear maps are linear." A reviewer will correctly say: this is not a mechanism, this is a reframing.

Contrast with what a real mechanism would look like: (a) identify a specific inductive bias in the CODI objective that drives the optimizer *away from* high-rank solutions (weight decay? implicit regularization of Adam? the L1-at-colon teacher target being low-rank?); (b) prove the bias drives rank↓; (c) predict what would happen if the bias were removed. None of this is done.

**Supporting evidence.**
- The "proof sketch" of Proposition 1 literally says: "$\partial\mathcal{L}/\partial\Z_{ij}$ is the inner product of $W[:,ij]$ with $\partial\mathcal{L}/\partial\phi$, which is independent of $\Z$'s SVD." This is a tautology: the gradient of a linear map does not depend on the input's SVD because a linear map does not depend on the input's SVD.
- The paper's own §5.3 walks this back: "Proposition~\ref{prop:jac} remains correct as a sufficient condition; it is not necessary."

**What the paper would need to do to defuse this.**
Either strengthen Proposition 1 to a claim that actually constrains training dynamics (e.g., "under Adam with weight decay $\lambda$, $\operatorname{erank}(\Z)$ decays monotonically with $\lambda$"), or downgrade it to a "structural observation" and acknowledge that the observation is a restatement of a well-known property of linear maps.

---

### A9: The objective-level rank-blindness claim conflates "CODI distillation objective" with "any loss that goes through a flatten bottleneck."
**Severity:** SERIOUS
**Type:** claim-scope

**Attack.**
§1 contribution 3 and §5.3 repeatedly attribute the flat curves to "the CODI distillation objective." But the experiments vary the distillation weight $\gamma \in \{0, 1\}$ (Run R3a, R3b in Table 1 have $\gamma=0$) and still report flat curves. If the CODI distillation loss were the culprit, $\gamma=0$ should remove the effect. The paper notes this in passing: "removing the L1-at-colon loss \emph{raises} effective rank from $\sim 10$ to $\sim 13$, but does not change the curve shape."

So the culprit is NOT the distillation loss specifically. It is whatever is happening when you put a reshape/flatten bottleneck on the feedback path and train with cross-entropy. Yet the paper continues to call this "CODI distillation objective rank-blindness" and names the readout and objective as the locus. The actual scope is narrower: *any standard next-token cross-entropy loss through a matrix bottleneck with flatten-then-project readout is rank-indifferent at this scale*. The "CODI" label is doing work the evidence does not support; it implies the issue is specific to CODI's teacher-student setup, when in fact removing the teacher-student loss entirely changes nothing about the rank-$k$ curve.

This is a meaningful overclaim. Readers will cite the paper as "CODI objective is rank-blind" when what the evidence supports is "our training pipeline is rank-blind."

**Supporting evidence.**
- §3.1 Table 1: R3a and R3b have $\gamma=0$ (no distillation loss). Their curves are still flat.
- §3.1 line 40–42: "removing the L1-at-colon loss \emph{raises} effective rank from $\sim 10$ to $\sim 13$, but does not change the curve shape."
- §5 positive-control runs all use $\gamma=0$ (§5 setup: "$\gamma=0$"). So the "CODI distillation objective is rank-blind" conclusion is drawn from experiments *without* CODI distillation.

**What the paper would need to do to defuse this.**
Rename the mechanism claim from "the CODI distillation objective" to "the matrix-bottleneck training loss in our pipeline" or "next-token CE loss backpropagated through a matrix bottleneck." Explicitly note that $\gamma=0$ experiments have no CODI distillation loss, so the rank-blindness is not about CODI. This is a framing fix, not a new experiment.

---

### A10: Statistical power is insufficient to turn the null results into a structural claim.
**Severity:** SERIOUS
**Type:** statistical

**Attack.**
§5 Table 2 reports Spearman $p$-values of 0.63, 0.14, 0.82, 0.46 on $n=128$ test problems. The paper interprets "$p > 0.14$" as "no effect." But "null result at $n=128$ with $p=0.14$" means the correlation is estimated at $r_s \approx -0.13$ with a standard error that still admits $|r_s| \sim 0.20$ as plausible. A $|r_s| = 0.20$ rank-correctness correlation would be a small but nonzero signal. The paper's $p=0.14$ in Bilinear+GELU is the *closest-to-significant* result and is exactly the one the reader should worry about as a weak-but-real effect being masked by small $n$.

Moreover, the paper is running five Spearman tests (one per readout) and reporting the minimum $p$ without multiple-comparison correction. Under Bonferroni the threshold for significance at $\alpha=0.05$ across 5 tests is 0.01. Under Holm–Bonferroni the thresholds are more generous but still much stricter than 0.05. The paper does not apply any correction — and its *non-corrected* minimum is 0.14, already well above corrected thresholds, so the multiple-comparisons problem doesn't bite here. But the flip side is that the paper also has no positive-signal Spearman to discipline the interpretation.

More fundamental: absence of evidence is not evidence of absence. With $n=128$, the paper has ~80% power to detect $|r_s| \geq 0.25$ at $\alpha=0.05$. It has <20% power to detect $|r_s| \geq 0.10$. The claim "rank does not track accuracy" needs either a power analysis or a larger $n$. ProsQA's full test set is 500; using 500 would triple the power. The paper uses 128 to match the "standard CODI eval split" but this is a convention choice, not a statistical necessity.

**Supporting evidence.**
- Table 2 §5: Spearman $p$-values 0.63, 0.14, 0.82, 0.46 on $n=128$.
- ProsQA test set size is 500 (paper §2: "The test set has 128 problems, matching the evaluation protocol in the original CODI release").

**What the paper would need to do to defuse this.**
Run the rank-$k$ evals and Spearman tests on the full 500-problem test set (cheap — inference only). Report effect-size confidence intervals, not just $p$-values. Add a short power-analysis sentence: "we have $X\%$ power to detect a rank-accuracy correlation of $|r_s| \geq Y$ at our sample size."

---

### A11: The paper's related-work positioning misses a highly-relevant 2026 ICLR paper that directly competes on the mechanism claim.
**Severity:** SERIOUS
**Type:** missing-citation

**Attack.**
The paper cites SIM-CoT (Shen et al., ICLR 2026, arXiv:2509.20317) as a parallel diagnosis ("insufficient step-level supervision"). But it does NOT cite "Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure" (arXiv:2602.08783) except as a vague reference — actually it DOES cite it via `anonymous2026dynamics` and dismisses it as "descriptive of loss-landscape behavior." This is wrong in detail.

That paper actually runs *multiple* intervention/ablation protocols on latent CoT states including: (a) zero intervention, (b) mean intervention, (c) gaussian noise intervention, (d) early-stop decoding that "truncates latent computation after step $k$ and decodes directly from $h_k$." Their early-stop decoding is a cousin of the paper's rank-$k$ ablation at a different axis (step depth vs spectral truncation). They find specific step-wise causal structure — some latent steps are causally necessary, some are not, and this depends on the task. This is a direct competitor for the paper's "latents don't encode reasoning state" claim, and the paper should engage with it more than a one-sentence dismissal.

Also missing:
- **Latent Reasoning in LLMs as a Vocabulary-Space Superposition** (arXiv:2510.15522, Oct 2025). Claims latent reasoning IS a superposition *in the vocab column space*, with a specific mechanism (Latent-Vocab constraint, Induction-Supervision Masking). Directly contradicts the paper's "latents don't superpose" framing at the mechanism level. The paper does not cite it.
- **Breaking the Low-Rank Dilemma of Linear Attention** (CVPR 2025, arXiv:2411.07635). Proposes rank-augmented linear attention for specifically the rank-bottleneck problem the paper is studying, though in a different context (vision). Shows rank augmentation is achievable in practice with specific architectural choices. Relevant to §5's claim that "nonlinear-in-$\Z$ readouts are not a neutral design choice" because this paper shows some rank-augmentation designs DO work; the paper's design space may have missed the working variants.
- **Optimal ablation for interpretability** (Li & Janson, NeurIPS 2024, arXiv:2409.09951). Shows that zero/mean/resample ablation (the entire ablation-based interpretability canon the paper's rank-$k$ is an instance of) substantially overestimates component importance vs. "optimal ablation" which replaces a component with the constant that minimizes expected loss. Methodological caveat that rank-$k$ truncation is a zero-like ablation on singular directions and may be biased in ways the paper does not discuss.

**Supporting evidence.**
- arXiv:2602.08783, §3 (intervention protocols): zero, mean, mean_step, gaussian_h, gaussian_mu, gaussian_mu_step.
- arXiv:2510.15522, abstract: "latent reasoning is not merely a compression of a single reasoning chain but also a superposition of multiple chains" — a counter-claim.
- arXiv:2411.07635, RALA: rank augmentation works in linear attention via explicit token-level and channel-level nonlinear augmentation — suggests the paper's "nonlinear $\phi$ doesn't help" may be specific to the family tried.
- arXiv:2409.09951: "Among attention heads and MLP blocks, optimal ablation accounts for only 11.1% of Δzero, 33.0% of Δmean, 17.7% of Δresample for the median component" — zero/truncation-style ablations substantially overestimate effect magnitudes (but this is an *upper bound* — they wouldn't explain why the paper's rank-$k$ curves are flat; if anything, optimal ablation should make them even flatter, so this one cuts both ways).

**What the paper would need to do to defuse this.**
Add one paragraph to Related Work addressing arXiv:2602.08783 and arXiv:2510.15522 specifically. Note that arXiv:2510.15522 makes a competing "superposition in vocab space" claim that the paper's rank-$k$ ablation is not designed to test (rank is not the right observable if the superposition lives in vocab column space). This weakens the paper's universality claim about rank-$k$ ablation as a "natural probe" (§1).

---

### A12: The GSM8K-Aug flat curve (Table 1, Row R1) should not be in the headline claim — the model is at 6% accuracy and the curve is meaningless.
**Severity:** SERIOUS (bordering CRITICAL for reader trust)
**Type:** methodological

**Attack.**
Table 1 row R1 reports GSM8K-Aug at 6.00%→6.12% across $k$. The abstract and §1 cite "Four flat rank-$k$ curves from the flatten-then-project readout across two tasks, two distillation weights, and a thinker on/off ablation." The "two tasks" framing leans on this GSM8K row. But at 6% accuracy the model is essentially at chance — it is not solving the task, so the rank-$k$ curve is measuring nothing. The paper admits this in §7 ("a low-accuracy operating point ($6\%$) where the model is barely learning the task; we do not interpret its flat rank-$k$ curve as strong evidence on its own") but still counts it in the abstract's headline. The abstract does not carry this caveat.

A reviewer will correctly note: three flat curves, one of which is at 6% accuracy, is not "four flat curves across two tasks." It is "three flat curves on ProsQA and one meaningless curve on GSM8K-Aug." The paper is padding the count.

This matters because the "single task family" limitation in §7 is the most natural reviewer attack on the paper, and the abstract's "two tasks" language is specifically designed to blunt that attack. But the GSM8K-Aug row does not deliver what the abstract implies.

**Supporting evidence.**
- Table 1 R1: 6.00%/6.12% at $k$=1/16.
- §7 line 6-11: "a low-accuracy operating point ($6\%$) where the model is barely learning the task; we do not interpret its flat rank-$k$ curve as strong evidence on its own."
- Abstract does not carry this caveat.

**What the paper would need to do to defuse this.**
Either (a) carry the caveat forward: "three flat ProsQA curves plus one GSM8K-Aug curve at a below-threshold operating point that we flag as uninterpretable." Or (b) re-run GSM8K-Aug at a higher operating point (the paper admits this is "queued") and update the claim to reflect the actual evidence. The current abstract language is misleading.

---

### A13: The depth sweep is completely broken — only one data point reported, the rest pending.
**Severity:** SERIOUS
**Type:** missing-experiment

**Attack.**
§4.1 Depth sweep: only $n=6$ is reported. "$n\in\{16,32,64\}$ configurations OOM'd at the default batch sizes and are pending re-runs at smaller batches." The figure 4 caption even says: "The $n=16,32,64$ points are partial data and will be added in the appendix when the re-runs complete." A single data point is not a sweep. The section's headline claim — "The 'more-latent-thought means better-reasoning' story fails at this scale" — is drawn from $n=6$ alone: 78.91% vs. vanilla SFT's 81.77%.

But going from 0 latent refinement steps (vanilla SFT) to 6 refinement steps could produce any non-monotone behavior with $n_\text{latents}$; with only two data points (0 and 6), you cannot draw a trend. The headline is overclaimed.

This is also a framing issue: the paper presents depth-and-scale (§4) as evidence that "depth and scale do not rescue matrix-CODI." But §4's matrix-CODI-at-gpt2-large row is literally "pending" (OOM'd at batch 4 and 2), and §4.1's depth sweep has three out of four points pending. The section presents as a completed piece of negative evidence; it is 2/6 done. A reviewer will flag this as low-quality evidence gatekeeping a structural claim.

**Supporting evidence.**
- §4.1 lines 14–15: "$n\in\{16,32,64\}$ configurations OOM'd at the default batch sizes and are pending re-runs at smaller batches."
- §4.2 lines 37–38: "Matrix-CODI at large OOM'd at batch $4$ and $2$; we report it as pending."
- Figure 4 caption: "The $n=16,32,64$ points are partial data and will be added in the appendix when the re-runs complete."
- Figure 3 (scale sweep) omits the matrix-CODI-at-large point entirely.

**What the paper would need to do to defuse this.**
Either complete the experiments before submission or remove §4 from the main body and pitch it as future work in §7. The current state is half-baked evidence supporting a "depth and scale don't help" claim; a reviewer will (correctly) not accept it.

---

### A14: The scale-sweep finding "matrix-CODI tracks below vanilla SFT at every tested scale" has a sample size of two scales (small and medium), with large OOM'd.
**Severity:** MINOR-to-SERIOUS
**Type:** statistical / claim-scope

**Attack.**
§4.2 concludes "Matrix-CODI underperforms at every scale." The evidence: gpt2-small matrix-CODI 80.47% vs. vanilla 81.77% (−1.3pp); gpt2-medium matrix-CODI 79.69% vs. vanilla 80.47% (−0.78pp); gpt2-large pending. Two data points with gaps of 1.3pp and 0.78pp, both within the reported seed variance of ±1.2pp at gpt2-small. The "matrix underperforms at every scale" is two coin flips coming up in the same direction with overlapping confidence intervals. Not strong evidence.

**Supporting evidence.**
- PAPER_RESULTS_SUMMARY.md Table 3 and §4.2.
- Three-seed std at gpt2-small is ±1.2pp (§3.3). The observed gap at small is 1.3pp, barely outside seed noise.

**What the paper would need to do to defuse this.**
Either run matrix-CODI at multiple seeds at each scale to get actual confidence intervals, or downgrade the claim to "matrix-CODI does not exceed vanilla SFT at any tested scale, with gaps within seed noise."

---

### A15: Proposition 1's assumptions are not tight — the proposition holds but the paper does not leverage this correctly.
**Severity:** MINOR
**Type:** theoretical

**Attack.**
Proposition 1 says "Let $\phi$ be linear in $\Z$. … The optimizer cannot use the singular structure of $\Z$ as a training signal through $\phi$." The proof handles the local gradient but does not address global optimization dynamics. In fact, even a linear readout $\phi(\Z) = W\vecop(\Z)$ coupled with a regularizer (e.g., weight decay, $\ell_2$ on $W_{\text{up}}$, implicit bias of Adam) can drive $\Z$ to a specific rank. The proposition's claim is about what gradient descent *cannot* prefer given a pure reconstruction loss; it does not rule out that trained $\Z$ has structure from other sources.

This is a minor nit because the paper's actual experiments don't depend on the proposition's tightness — they are empirical. But the proposition as stated could be read as stronger than it is, and the three-seed rank spread ($\{4, 12, 13\}$) is explicitly *not* compatible with "rank is purely a free direction": if it were, the distribution would be uniform; instead, two of three seeds land at rank ~12–13 (suggesting a weak attractor), with one outlier at rank ~4. The proposition as stated doesn't account for the weak attractor.

**Supporting evidence.**
- Literature on implicit bias of SGD/Adam (Gunasekar et al. 2017 et seq.) shows linear maps trained with gradient-based methods have implicit rank-regularization even in the absence of explicit regularizers.
- Nuclear norm regularization literature (arXiv:2405.14544): explicit low-rank bias from gradient descent on linear networks.

**What the paper would need to do to defuse this.**
Tighten §3.2 to "gradient descent on a pure reconstruction loss $\mathcal{L}(\phi(\Z))$ with $\phi$ linear in $\Z$ has no term that prefers one rank of $\Z$ over another *through the loss gradient*." Acknowledge that implicit bias from the optimizer and regularizers may still shape rank. This matches the paper's own three-seed data (weak attractor around rank 12) better than the stronger phrasing.

---

### A16: "Effective rank" changes 3× between seeds but accuracy is tight — what if rank just isn't what matters and the task is solvable at rank 4?
**Severity:** SERIOUS
**Type:** alternative-explanation

**Attack.**
§3.3 presents the three-seed spread in rank (4, 12, 13) with tight accuracy (81.51 ± 1.2pp) as "the loss does not push $\Z$ toward any particular rank." But there is a different, simpler explanation the paper does not consider: **ProsQA is a rank-4-solvable task**. Seed 42 happens to find the minimal-rank solution; seeds 1337 and 7 find over-parameterized solutions that also work. The tight accuracy is explained by all three being above the task's rank-4 threshold. The rank-4 seed is the "efficient" solution and the rank-12–13 seeds are the "lazy" solutions that use more rank than needed. Under this view, the rank-$k$ ablation curve would be flat because truncating a rank-13 $\Z$ down to rank 4 still preserves the functional rank-4 subspace.

This matches observations:
- The rank-$k$ curve is flat from $k=1$ to $k=16$: if the task is rank-4-solvable and higher-rank components are redundant, truncating from 16 to 4 doesn't hurt (matches). Why does truncating from 4 to 1 also not hurt? Because seed 42's solution has most of its "active" singular direction concentrated in the top-1 component even when erank is 4 (plausible).
- The linear probe peaks at Z[1] (AUC 0.693) and decays to Z[5] (0.633): the effective reasoning state is in the first 1–2 latent positions; the rest are redundant. Compatible with "task is simple."

The paper's framing is "rank is a free direction in the loss landscape." An equally consistent framing is "task fits in rank-1, model uses rank-1, everything else is noise the model has to tolerate but doesn't prefer." These have identical predictions for every experiment the paper runs. The paper's positioning (mechanism-level claim about the objective) is the stronger of the two; simpler explanation is "the task needs rank-1."

**Supporting evidence.**
- Linear probe peaks at Z[1] and monotonically decays (EXPERIMENT_LOG.md lines 1280-1286). This is consistent with early latent positions carrying most of the functional information and later positions being redundant — a pattern compatible with "the task is solvable early."
- ProsQA is a single-answer task with one positive and one distractor. There is no a priori reason it needs more than one degree of freedom at the answer position.
- Vanilla SFT at 81.77% accuracy with no matrix bottleneck, no latents, no special machinery, is close to matrix-CODI's 82.03%. This is consistent with "the task is just easy; any reasonable architecture does ≈82%."

**What the paper would need to do to defuse this.**
Run the positive-control experiments on a task that *provably* requires multi-rank reasoning (the paper's KILL_LIST.md even flags this as an unfilled gap). Or provide an argument that ProsQA specifically requires rank > 1 — e.g., a probe that succeeds on target-vs-distractor only when given higher singular components of $\Z$. Currently the paper provides no such evidence; the "rank is free" claim is consistent with "task is rank-1."

---

### A17: The reproducibility section is thin on the positive-control experiments — only one seed per variant, so results may not be reproducible.
**Severity:** MINOR
**Type:** reproducibility

**Attack.**
§7: "The four positive-control variants in §5 were each trained once (compute-bounded)." §9: "Headline numbers by source. … Table 2 and Fig. 1 from run_matrix_codi.py with --readout in {...} evaluated via rank_eval.py." One seed per variant. The paper's central claim rests on four data points each of which is a single seed. With the three-seed flatten baseline showing accuracy variation of ±1.2pp, the per-variant numbers in Table 2 (differences of ~0.8pp between variants) are within seed noise.

Specifically: Quadratic reports 79.69% perfectly flat across all $k$. Bilinear+GELU also hits 79.69% at $k\geq 2$. These are the same number. Is this a real match or a rounding artifact? The paper reports two decimal places (0.79 vs 0.7969) — need more precision or more seeds to interpret.

**Supporting evidence.**
- §7 limitations: "one seed per positive control"
- Table 2 has three variants (bilinear+GELU, quadratic) ending at 79.69% and one at 78.12% — suspicious coincidence of 79.69% appearing at exactly $128 \times 0.7969 \approx 102$ correct out of 128 test problems for two different architectures.

**What the paper would need to do to defuse this.**
Commit to running at least three seeds for each positive-control readout before the camera-ready. Note that "compute-bounded" is not a great excuse here — the paper uses 1×H100 runs of ~3.5 hours each, totaling ~14 hours of H100 per variant for 3 seeds. At RunPod spot prices that is ~$30 per variant. For a paper whose central claim turns on the positive controls, this is a small ask.

---

### A18: The paper's "one architecture family" limitation in §7 is honest but the mechanism claim is stated as architecture-agnostic — these are inconsistent.
**Severity:** MINOR
**Type:** claim-scope

**Attack.**
§7: "The structural mechanism (Proposition~\ref{prop:jac}) is stated in terms of the readout $\phi$ and is therefore architecture-agnostic, but our empirical evidence covers only GPT-2 $\{$small, medium, large$\}$."

Proposition 1 is architecture-agnostic in the sense that it is about any linear map. But the §5.3 "pivot" — which is the paper's actual central claim — is about the interaction between the readout, the optimizer, the specific training objective, and the *specific backbone*. "The optimizer can find solutions where the active subspace of the readout's nonlinear Jacobian lies along a rank-1 direction of $\Z$" depends on which solutions exist in the loss landscape, which depends on the model. The GPT-2 residual stream's structure (768-dim, specific layer norm, specific attention pattern) is doing work in what solutions are accessible. Whether this generalizes to, say, Llama-style models with RMSNorm and SwiGLU is an empirical question not tested here.

**Supporting evidence.**
- §7 "one architecture family" paragraph.
- No experiment with a non-GPT-2 backbone.

**What the paper would need to do to defuse this.**
State the scope as "within the GPT-2 family, at the tested sizes, the objective is rank-indifferent." Acknowledge that the mechanism claim is at most as general as the empirical evidence. A single non-GPT-2 control (e.g., Pythia-160M) would go a long way.

---

### A19: The paper asserts novelty of "objective-level rank-blindness" but does not engage with the implicit-regularization literature that says the opposite.
**Severity:** MINOR
**Type:** missing-citation

**Attack.**
§6 does not cite any of: Gunasekar et al. 2017 "Implicit Regularization in Matrix Factorization"; Arora et al. 2019 "Implicit regularization in deep matrix factorization"; Razin & Cohen 2020 "Implicit regularization in deep learning may not be explainable by norms"; Kamalakara et al. 2022 "Exploring Low-rank Structure of Transformer"; and more recent work like "Weight decay induces low-rank attention layers" (arXiv:2410.06001, NeurIPS 2024). These lines argue that gradient descent on matrix factorizations has IMPLICIT bias toward low-rank solutions. If these results apply to the paper's setting, then the "rank-indifference" claim is at odds with a decade of established theory — there *should* be implicit pressure toward low rank, and this would predict the three-seed result that seeds 1337 and 7 converge near rank 12 while seed 42 collapses to rank 4 (= implicit low-rank attractor with seed-dependent escape).

The paper's "objective is rank-blind" framing is in tension with this literature. A reviewer will ask: given that gradient descent on matrix factorizations is known to be implicitly biased toward low rank, why does the paper claim rank is a "free direction"? One of these is wrong, or the setting of the paper is special in a way that breaks the implicit-bias result.

**Supporting evidence.**
- Gunasekar, Lee, Soudry, Srebro 2017 "Implicit Regularization in Matrix Factorization"
- Arora, Cohen, Hu, Luo 2019 "Implicit regularization in deep matrix factorization"
- Galanti, Siegel, Gupte, Poggio 2022 "SGD and weight decay provably induce a low-rank bias in neural networks"
- Kobayashi, Akyürek, Andreas, et al. 2024 "Weight decay induces low-rank attention layers"

**What the paper would need to do to defuse this.**
Add a paragraph to Related Work discussing implicit low-rank bias and acknowledge that the three-seed result (two seeds ≈ rank 12, one seed at rank 4) is consistent with implicit low-rank bias plus seed-dependent convergence. The "objective is rank-blind" claim should be tempered: it's really "objective has rank-indifference in the reconstruction term but may have implicit low-rank bias from the optimizer+regularizer."

---

### A20: The paper's title ("The Gradient Does Not See Rank") overclaims vs. the actual result.
**Severity:** MINOR
**Type:** claim-scope

**Attack.**
The title promises a universal result about gradients and rank. The actual result is: "In our four positive-control readouts on ProsQA with GPT-2 small, the rank-$k$ ablation curves are flat and Spearman correlations are non-significant." The gradient DOES see rank in many settings (every paper in the implicit-regularization literature, see A19). What the paper has is: in a specific matrix-CODI pipeline, rank does not appear to matter for one task.

A reviewer will find this click-baity and push for a more honest title. The current title is the kind of sentence that invites "citation-by-headline" misreadings downstream.

**Supporting evidence.**
- Compare title ("The Gradient Does Not See Rank") to actual scope in §7.

**What the paper would need to do to defuse this.**
Change title to something like "The CODI Gradient Does Not Shape Rank: Evidence from Matrix-Valued Latent CoT" or even more narrowly "Rank-$k$ Ablation Is Uninformative in Matrix-CODI: Mechanism and Evidence." The current title is the abstract's most-shared sentence and will travel independently of the qualifiers.

---

## Attacks I considered but decided were weak

- **"Your matrix thought has only 256 dims vs the residual stream 768 dims, so of course it can't encode enough."** — Valid concern but the paper's claim is comparative (matrix-CODI vs vanilla SFT at matched everything else); the capacity comparison is informative, not a flaw.
- **"Proposition 1 is literally trivial so you shouldn't dignify it with a theorem."** — Nitpick. The authors are honest that it's a sufficient condition and the body of the argument is empirical.
- **"Why only 25 epochs? Longer training might change rank."** — The three-seed result at 25 epochs shows rank spread; longer training might converge more but probably would not fix the central finding. Don't push this.
- **"GSM8K-Aug at 6% means your whole pipeline is broken."** — Covered adequately by the paper in §7; reviewer-valid but already acknowledged.
- **"The paper's SUBMISSION_CHECKLIST.md and author reasoning are visible to us but not to reviewers."** — Correct but reviewers can't cite internal documents. Not an attack axis.

## New citations you found that should be in Related Work

- **Latent Reasoning in LLMs as a Vocabulary-Space Superposition** (arXiv:2510.15522, Oct 2025). Competing claim: latent reasoning IS superposition, but in the vocab column space. The paper's rank-based observable doesn't test this mechanism. Should be cited in §6 and §5 as an alternative superposition substrate that rank-$k$ truncation doesn't probe.
- **Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure** (arXiv:2602.08783, Feb 2026). The paper cites this as `anonymous2026dynamics` and dismisses it in one sentence; it deserves more than that. They run step-wise causal interventions — a direct competitor in the space of "what do latent CoT states actually encode."
- **Optimal ablation for interpretability** (Li & Janson, NeurIPS 2024, arXiv:2409.09951). Methodological caveat on the entire ablation-based interpretability canon; rank-$k$ truncation is a special case of a zero-ablation and may overestimate effect sizes (though in this paper's case the null finding is robust to this).
- **Breaking the Low-Rank Dilemma of Linear Attention** (CVPR 2025, arXiv:2411.07635). Shows architectural rank-augmentation is achievable in linear attention settings; suggests the paper's "no nonlinear readout helps" finding may be specific to the family tried.
- **Weight decay induces low-rank attention layers** (Kobayashi et al., NeurIPS 2024, OpenReview oDeqjIM9Sk). Directly competes with the "rank is a free direction" claim — shows that weight decay has a *known* low-rank bias in attention, which predicts the paper's rank-12 attractor and might explain seed 42's rank-4 outlier.
- **Nuclear Norm Regularization for Deep Learning** (arXiv:2405.14544, 2024). Theoretical basis for implicit low-rank bias from gradient descent; connects to A15 and A19.
