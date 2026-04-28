# Team Thread: rank-aware-gauntlet-v2

Adversarial gauntlet for rank-aware-training experiments — includes an ideator who proposes new directions, then synthesizer + 3 attackers + pragmatist + validator argue. All parallel; cross-pollinate via thread.

---

## [2026-04-28T16:24:57.299Z] orchestrator

Team **rank-aware-gauntlet-v2** started. 7 agent(s). Target deadline: 30min.

## [2026-04-28T16:25:42.284Z] synthesizer

**SYNTHESIZER: 3x3 restatement with falsifiable predictions**

Full 3x3 (task × loss):

| | std-CE | +entropy H | +nuclear ‖Z‖_* |
|---|---|---|---|
| ProsQA-1 | FLAT [already done, paper §3-5] | predict: FLAT (task rank-1-solvable; entropy can't help what isn't needed) | predict: FLAT |
| MULTI-2 | predict: FLAT/floor (objective still rank-blind) | **predict: BENDS at k=2 — headline** | predict: weak/σ_1-only bend (wrong functional) |
| MULTI-4 | predict: floor effect (hard task, low acc) | predict: bends, inflection at k=4 | predict: weak bend |

Each cell is falsifiable. The headline falsification risk: if MULTI-2 + std-CE bends, standard CE training IS rank-sensitive on multi-answer tasks (story collapses).

## [2026-04-28T16:25:50.809Z] synthesizer

**SYNTHESIZER: Minimal sufficient subset (4 new cells, 1 existing)**

Full 3x3 = 9 cells × 3 seeds ≈ $80. We can cut to 5 cells (1 free) ≈ $36 and tell the same complete story:

1. **(ProsQA-1, std-CE)** — FREE, already done. Paper baseline. Anchors the narrative.
2. **(MULTI-2, std-CE)** — Baseline for new task. Confirms: task alone doesn't fix rank-blindness.
3. **(MULTI-2, +entropy)** — THE headline. Must work for the paper to ship.
4. **(ProsQA-1, +entropy)** — Null control. Entropy doesn't introduce rank structure when task doesn't need it. (Clean story vs. messy story.)
5. **(MULTI-2, +nuclear)** — Mechanism comparison. Nuclear is the wrong functional; entropy wins.

**Drop entirely:** All MULTI-4 cells (3 cells saved). MULTI-4 is confirmatory only — if MULTI-2 bends, MULTI-4 is bonus; if MULTI-2 is flat, MULTI-4 won't save us. Re-queue MULTI-4 contingent on MULTI-2 results.
**Drop:** (ProsQA-1, +nuclear) — irrelevant to narrative.

Decision criterion for early stopping: run cells 1-3 first (≈24 H100h). If (MULTI-2, std-CE) bends, reassess. If (MULTI-2, +entropy) is flat, the 3x3 is dead — pivot to E (implicit bias wd=0 sweep) or D (spectral anchoring).

## [2026-04-28T16:25:55.582Z] empirical-skeptic

Empirical-skeptic, opening attack on the 3x3.

CORE WORRY: §5 already ran an SVD-augmented readout that exposed σ(Z) directly to the optimizer via MLP(svdvals(Z)) -> still flat curve, accuracy *below* flatten baseline (78.12 vs 80.47). The proposed entropy reward H(σ̃) is the next-door neighbor of that experiment: same SVD backward, same per-sample σ as input. The mechanistic difference between 'σ feeds CE' (failed) and 'σ feeds an explicit auxiliary loss that pushes uniform' has to be articulated PRECISELY before we burn  spot, or we will write the §5 result with a different parameter.

I will pile concrete attacks below.

## [2026-04-28T16:25:59.532Z] synthesizer

**SYNTHESIZER: Key attack surface — the 'rank-k by orthogonality' argument has a gap**

Report A claims: k disjoint answers in MULTI-k requires k orthogonal directions in Z, so rank-1 truncation provably fails one target. But there's a logical gap:

**The gap:** 'Disjoint' answers in the ProsQA DAG = separate leaf nodes. This does NOT guarantee orthogonal representations in the model's learned embedding space. The model could learn k bilinear heads all projecting onto the *same* dominant singular direction if the k answers are semantically correlated (or if the optimizer finds a clever shared direction via weight tying).

The formal lower bound only holds if the k targets are provably *incoherent in representation space* — but that's an empirical property of the learned embeddings, not a structural guarantee of the dataset construction.

**Mitigation options:**
A) Add an explicit orthogonality penalty on the bilinear probe vectors u_i, v_i during eval (not training) — forces the k readout directions to be separated, making the bound tight.
B) Verify post-hoc: after training MULTI-2 + std-CE, measure cosine similarity between the two bilinear probe pairs. If near-zero, the bound is empirically satisfied. If not, the 'by construction' claim is false.

Option B is essentially free (post-hoc diagnostic). Recommend flagging this to experimenter: add cosine similarity between bilinear probe pairs as a mandatory logging stat.

## [2026-04-28T16:26:04.607Z] methodological-skeptic

ATTACK 1 — The bilinear head u_i^T Z v_i is FICTIONAL. Looking at MatrixBottleneck.forward + student_forward: the readout is h_out = LayerNorm(w_down(flatten(Z))) → 1 more GPT-2 transformer pass → repeat 6 times → LM head reads h_answer ∈ R^768 at the answer position. The LM head NEVER touches Z directly. The 'k orthogonal directions in Z' argument has zero contact with the actual model. Z's rank affects the answer through MANY downstream nonlinearities (LayerNorm + 12 GPT-2 layers + 6 sequential mixings). Reviewer 2 will kill this on first read. The orthogonality argument applies to vec(Z) (a 1024-dim vector projected by a 768×1024 linear) — and rank of vec(Z) viewed as a vector is meaningless. To salvage A, you need a head that ACTUALLY operates as bilinear on Z. None exists in the script.

## [2026-04-28T16:26:09.267Z] empirical-skeptic

ATTACK 1 — Lambda window for entropy reward.

The proposers claim λ ∈ {0.01, 0.1, 1.0} and 'λ stable across runs because H is bounded [0, log d]'. That's necessary, not sufficient. Concrete failure modes:

(a) λ too small: gradient through ∂H/∂σ is dominated by ∂CE/∂σ; entropy term irrelevant; rank stays at the seed-noise attractor of {4,12,13}. No effect.

(b) λ too large: model finds the ZERO-COST way to flatten σ — make Z near-orthogonal but route info through W_down's rowspace such that the readout still recovers the answer from σ_1's direction. Or worse, just collapse W_up to scaled-identity, making Z trivially full-rank but functionally useless. Accuracy crashes; we 'succeeded at rank inflation' but failed at the task.

(c) The MIDDLE: there may be NO window. The mechanism (Prop 1) says the loss provides no gradient signal for which singular DIRECTIONS encode answer-relevant content. Entropy operates on σ MAGNITUDES, not on which directions carry information. So H↑ + CE ↓ can be jointly satisfied by a rank-d Z where only σ_1*u_1 v_1^T matters for the readout. This is exactly the 'effectively rank-1 active subspace' refined hypothesis from §5.3.

Falsification rule we MUST commit to before running: if rank ablation curve still flat at the λ where accuracy is within 2pp of baseline, the experiment NULLS. No 'try one more λ' loophole.

## [2026-04-28T16:26:14.128Z] methodological-skeptic

ATTACK 2 — Position-decomposition trivially substitutes for rank-decomposition. The model has 6 latent positions Z_1..Z_6, each d×d, each fed back through the FULL 768-dim GPT-2 hidden state. Total capacity to route info to the answer head is 6 × 768 = 4608 dims, NOT one Z. ProsQA-MULTI-k can be solved with all six Z_p rank-1 by encoding target_k at position k. The rank-project_k ablation (truncate_to_rank applied PER Z_p) doesn't constrain this: truncating each Z_p to rank-1 still leaves 6 independent rank-1 directions, sufficient for k≤6. The proposed task does NOT lower-bound any per-position rank. Lower bound at k=2: at most one of the six Z_p must be ≥ rank-2, and that's only if you ban inter-position routing through h_next, which the architecture does NOT do.

## [2026-04-28T16:26:22.420Z] empirical-skeptic

ATTACK 2 — Why would entropy reward succeed where SVD-aug failed?

The compilation says 'no prior work uses SV entropy as a differentiable training LOSS on per-sample latent matrix thoughts'. True. But the mechanistic question is whether differentiable entropy bends the rank-k curve, not whether it's novel.

SVD-aug feeds σ to the readout via MLP — so during training, ∂CE/∂σ_i can be nonzero, the model COULD learn to use σ_i for prediction. It chose not to (78.12% < flatten 80.47%, σ_1 shortcut taken).

Entropy reward DEMANDS σ uniform — that's a constraint, not an invitation to use σ. If the model can satisfy 'σ uniform' AND 'predict answer correctly' by routing all answer info through W_down's first column and treating σ_2..σ_d as orthogonal noise, both losses are minimized. The entropy term LIES TO YOU about whether rank is functional.

The only way entropy reward beats SVD-aug is if the joint optimization of CE + λH happens to land in a basin where rank>1 is GENUINELY needed — but that requires the *task* to need rank>1. Which is exactly why ProsQA-MULTI is the prerequisite, not the entropy reward.

PROPOSED ORDER: ProsQA-MULTI alone first (1 cell, 3 seeds, ~). If standard CE on ProsQA-MULTI bends the rank curve WITHOUT entropy, the task is doing the work and entropy is decoration. If it doesn't bend, entropy is the diagnostic — but we know what to test.

## [2026-04-28T16:26:22.887Z] methodological-skeptic

ATTACK 3 — Embedding-tying defeats orthogonality even granting the bilinear-head fiction. ProsQA-MULTI's k targets are GPT-2 wte vectors. GPT-2's vocabulary embeddings are NOT orthogonal — pairwise cosine ~0.1-0.4 for related tokens, and ProsQA names share BPE subwords (e.g. 'Alex'/'Alexis' share '_Alex'). A single dominant direction in latent space scores all k targets above the negatives if the targets cluster relative to distractors. The k-orthogonal-direction lower bound presumes target embeddings span k-D — must be MEASURED on the actual vocab subset before claiming. If max target-target cosine > target-distractor margin, rank-1 suffices. This is a 10-line check that MUST run before A: compute Gram matrix of {wte[t] : t ∈ target_set} for the chosen ProsQA-MULTI vocab, report rank/condition number/min-distractor-margin.

## [2026-04-28T16:26:23.148Z] reviewer-2

**REVIEWER-2 (hostile MI workshop persona) — vote: REJECT.**

The killer objection: methodological circularity, and the proposed "key" experiment is already in §5 of the existing paper in disguise.

**1. The 3×3 headline is a tautology.** Compilation lines 56–60: "entropy reward causes matrix-CODI to use rank functionally on tasks that need it." Restated: we constructed a task we engineered to require rank k by orthogonality, added a loss term H(σ̃) that explicitly maximises spectral entropy, observed the model now uses spectral entropy. That is "we trained on label X, model predicts X." The MI community will not credit this as a mechanism finding.

**2. The rank-k-by-orthogonality argument is hand-wavy.** §A assumes the bilinear probes (u_i, v_i) are fixed orthogonal — they aren't, they're learned. Learned readouts can share one dominant singular direction across k targets via vocab-column aliasing (Wang 2025, already cited in §6). The compilation explicitly asks line 156 "could the model find rank-1 by clever embedding tying?" — yes, and there's no proof otherwise. Without a formal lower bound, "ProsQA-MULTI requires rank k" is hope, not a property of the task.

**3. The entropy-reward experiment is subsumed by §5's SVD-aug readout.** §5 already exposes singular values directly to the gradient via differentiable svdvals — Spearman p=0.82, flat. Adding +λH(σ̃) is a stronger version of the same intervention. If the §5 result generalises (abstract line 17), the entropy term will spread the spectrum of Z without making any direction functional — the model finds a flat-spectrum rank-1 solution. The experiment has been pre-empted by the authors' own positive control.

**4. Missing controls.** No "ProsQA-MULTI std loss at higher capacity" cell — can't distinguish "task got harder, ablation hurts more" from "rank now functional." No causal-patching test that singular direction i carries answer i's *reasoning* vs answer i's *logit-space embedding* — i.e., bookkeeping vs superposition.

**Rebuttal direction (reject → weak accept):**

(a) Prove or demonstrate empirically that ProsQA-MULTI-k cannot be solved by any rank-1 Z under the realised readout — exhibit a Frobenius gap between optimal rank-1 vs rank-k solutions of the trained probes, on held-out instances, BEFORE training with entropy reward. Without this the lower bound is unsupported.

(b) Add a "ProsQA-MULTI + std loss + higher latent budget (d=24 or n_latent=12)" cell to rule out "harder task hurts ablation."

(c) Causal-patch σ_i u_i v_i^⊤ from a problem where target_i = A into a problem where target_i = B — this should flip prediction i *specifically*, not all k predictions. Otherwise singular directions are output channels, not reasoning paths.

(d) Add a falsification arm: an objective term that breaks the result while keeping entropy reward (e.g., remove bilinear separation between targets). Without it, this is confirmation-bias engineering.

With (a)–(d) you have a real objective × task × readout factorial. Without them: reject.

## [2026-04-28T16:26:25.881Z] pragmatist

## PRAGMATIST: Timeline & Budget Reality Check

**10 days, single H100 @ $1.50/h. Here's what fits.**

---

### Budget tiers

**$50 (minimal):** MULTI-2 only — std loss + entropy (6 runs, ~$18 GPU). Proves "task matters." Reviewer objection: "cherry-picked single task size."

**$100 (recommended):** MULTI-2 + MULTI-4 × {std loss, entropy} + ProsQA-1 entropy control + wd=0 sweep (~$63 GPU, $37 agent overhead). 3-condition factorial. Complete story. This is the plan.

**$200 (if core lands):** Full 3×3 with nuclear norm as contrast. Only worth adding nuclear norm if Days 1–6 succeed — it's the weakest novelty claim and adds 2 days of cascade work.

---

### Critical path

- **Today (Apr 28):** ProsQA-MULTI DAG mod (CPU, free). Start entropy reward cascade in parallel. No GPU needed yet.
- **Day 3 (Apr 30):** First GPU run — MULTI-2 baseline, 3 seeds, 6h = $9.
- **Day 4 (May 1):** MULTI-2 entropy reward, 3 seeds.
- **Day 5–6 (May 2–3):** MULTI-4 both conditions. **Decision day May 3: does entropy bend the curve?**
- **Days 7–10:** Paper writing, polish, submit.

**Bare minimum in 5 days:** MULTI-2 ×2 conditions, 6 runs, ~$25 GPU. Either result is publishable.

---

### Key risks (team should prioritize attacking these)

1. **Rank-1 loophole (30% P):** Can the model find a rank-1 solution on MULTI-2 by embedding tying? This is the central scientific risk. Team needs to verify the lower bound is actually tight before I authorize compute spend.
2. **Entropy fails even on MULTI-2 (25% P):** Publishable as a strong negative, but need to confirm paper's framing holds.
3. **Cascade timeline blows (20% P):** Each new code path takes ~½–1 day of agent coordination. Mitigate by starting MULTI gen today.

**P(clean positive result) ≈ 55%; P(publishable negative) ≈ 35%; P(total wash) ≈ 10%**

---

### Fit assessment for the proposed 3×3

- **A (ProsQA-MULTI): fits $100 plan. START IMMEDIATELY.**
- **B (entropy reward): fits $100 plan. Cascade in parallel with MULTI gen.**
- **C (nuclear norm): conditional — Day 6 gate only. Don't start now.**
- **D (spectral SIM-CoT): NO — too hard to implement in 10 days, save for camera-ready.**
- **E (implicit bias wd=0): fits as appendix — $5, 1 day. Slot Day 5 or 6.**
- **F, G, H: NO-GO as previously flagged.**

**Single forced recommendation: $100 plan, A+B+E. Start MULTI gen now. Nuclear norm is a Day-6 bonus, not a commitment.**

## [2026-04-28T16:26:30.107Z] synthesizer

**SYNTHESIZER: Full synthesis posted to SYNTHESIS.md. Summary:**

Minimal subset: 5 cells (1 free), ~$36 vs $80 full grid. Order matters — run (MULTI-2, std-CE) FIRST as the gating condition.

CONFIRM/FALSIFY criteria:
- (MULTI-2, std-CE) CONFIRM: flat → task alone doesn't fix rank-blindness. FALSIFY: bends → story collapses, reassess.
- (MULTI-2, +entropy) CONFIRM: Pearson r>0.5, degradation at k=1. FALSIFY: flat → entropy insufficient.
- (ProsQA-1, +entropy) CONFIRM: flat → clean 2×2 factorial story. 
- (MULTI-2, +nuclear) CONFIRM: weaker than entropy → correct mechanism claim.

MULTI-4: drop unless MULTI-2 results are borderline.

Mandatory diagnostic: log cosine similarity between bilinear probe pairs after MULTI-2 training — this is the only way to validate the 'rank-k by construction' claim. Without it, the formal argument is hand-wavy.

Watching for ideator posts to evaluate swaps.

## [2026-04-28T16:26:31.303Z] methodological-skeptic

ATTACK 4 — §5.5 negative control does NOT generalize to ProsQA-MULTI. The vanilla-GPT-2-SFT control shows rank-k flat WITHOUT a Z observable (rank metric is on what — the feedback hidden? a fake projection?). It establishes a 'curve is flat when there's no rank' control, but says nothing about whether flat curves on matrix-CODI imply rank-blindness. Worse: vanilla SFT solves ProsQA at the same accuracy as random-h, so the negative control is testing on a model that ISN'T USING the latent at all. Generalizing this to ProsQA-MULTI requires that vanilla GPT-2 SFT can ALSO solve ProsQA-MULTI without the bottleneck — which would itself prove the rank lower bound is wrong. The control is internally consistent but the inferential bridge to the multi-target case is broken.

## [2026-04-28T16:26:34.359Z] empirical-skeptic

ATTACK 3 — Mode collapse: 'rank functional' vs 'rank noisy'.

If entropy reward inflates rank but the model routes everything through σ_1's direction, we'll see:
  - effective rank goes up (H↑ measured)
  - rank-k ablation: still flat (truncating σ_2..σ_d removes nothing useful)
  - linear probe AUC on Z: unchanged or worse

That's a 'rank noisy' outcome that LOOKS like the rank rose but the rise is non-functional. The ideator's prediction 'curve bends on ProsQA-MULTI with entropy' would then be FALSIFIED for a different reason than they think.

DIAGNOSTIC TO DISTINGUISH (must commit pre-experiment):
  D1. Per-singular-direction ablation: zero σ_k while keeping others, measure ΔAcc. If rank is functional, ΔAcc ≈ 1/k. If rank is noisy, ΔAcc(σ_1) >> ΔAcc(σ_2..d).
  D2. Linear probe on each singular direction: train K probes on (σ_k * u_k v_k^T) flattened. If rank is functional, AUC roughly equal across k. If noisy, AUC concentrated in σ_1.
  D3. Refined-hypothesis check from §5.3: compute erank(J(Z)) of the trained readout's Jacobian. Predict ≥ 4 if entropy reward worked; ~1 if it didn't.

Without D1-D3 baked in, we cannot distinguish 'rank-aware training works' from 'rank reward is decorative'. This is the #1 reviewer attack on a 'GO' result.

## [2026-04-28T16:26:41.405Z] methodological-skeptic

ATTACK 5 (B/D specifically) — SVD backward pathology near uniform sigma. Entropy reward H(σ̃) is MAXIMIZED at σ_1 = σ_2 = ... = σ_d. PyTorch's torch.linalg.svd backward formula contains 1/(σ_i² − σ_j²) terms (Wedin/Mirsky) which BLOW UP at degeneracy. svdvals (B's path) is the only stable variant — its backward avoids U,V terms — so (B) survives, but ANY anchored-direction loss (D) that needs U_k or V_k will hit nan as soon as σ's coalesce. Compilation cites 'Moore-Penrose pseudoinverse SVD (arXiv 2411.14141)' but doesn't show that PyTorch ships this — it doesn't (as of 2.4). (D) needs a custom backward; without one, training will diverge in the regime where its loss matters. Mark (D) as 'requires custom autograd' or kill it.

## [2026-04-28T16:26:49.531Z] empirical-skeptic

ATTACK 4 — ProsQA-MULTI generation has a trivial-baseline trap.

The proposal: 'questions with k disjoint positive answers requiring k independent predictions at the answer position. Bilinear head u_i^T Z v_i for each of k targets needs k orthogonal directions in Z.'

The argument assumes:
  (a) the SAME Z is the only intermediate that produces all k answers (no per-target W_down rows shortcut),
  (b) the question text alone does not enumerate the k targets (else a rank-1 Z encoding 'which question' is sufficient — W_down memorizes question→{a1..ak} mapping),
  (c) the k targets are independent enough that no rank-1 vector can encode 'all positives have property P' compactly.

ProsQA's DAG generator: each question is 'which leaf has property P?' If the multi-answer extension is 'list ALL leaves with property P' from a question that names P, then the W_down can store the entire {entity_id → properties} table indexed by the rank-1 'which P' direction in Z. Rank-1 sufficient.

REQUIRED DIAGNOSTICS before claiming ProsQA-MULTI is intrinsically rank-k:
  D4. Train a rank-1-CONSTRAINED matrix-CODI on ProsQA-MULTI-2 (project Z to its top-1 SVD truncation BEFORE readout). If accuracy matches unconstrained, the task is NOT rank-k and the construction failed.
  D5. Train vanilla SFT (no Z) on ProsQA-MULTI-2. If accuracy ≥ matrix-CODI, Z is decoration regardless of rank.
  D6. Question-only baseline: train a 1-layer MLP on the question token-id histogram. If it solves >50% of the multi-answer questions, the task leaks through the question embedding alone.

Without D4 in particular, ProsQA-MULTI is rank-k by ASSERTION, not by CONSTRUCTION. The rank-1 truncation IS the falsification of the construction.

## [2026-04-28T16:26:51.368Z] methodological-skeptic

ATTACK 6 (3x3 efficiency) — The nuclear-norm column is dead weight. Compilation already concedes 'predict: bends only on σ_1 not k=2' and 'predict: weak bend' — predicted-null cells in a 3x3 are rhetorical, not informative. They waste 1/3 of the compute (3 cells × 3 seeds × 2 H100h = 18 H100h ≈ $27 of the $80). DROP nuclear norm. Replace with the column that actually disambiguates: 'rank-1-per-Z-position penalty (force each Z_p to rank-1 during training)' — predicts ProsQA-MULTI-k FAILS only if the orthogonality argument is right. That's the MISSING falsifier — currently no cell in the 3x3 falsifies the position-decomposition alternative (Attack 2). Cut nuclear, add per-position-rank-1 forcing, and 3x3 becomes a real causal design.

## [2026-04-28T16:27:00.554Z] methodological-skeptic

ATTACK 7 (E specifically) — wd=0 vs wd=0.01 at 6 runs × 3 seeds = 18 runs. The paper already reports seed-spread {4,12,13} from existing runs — that's a 9-pt range out of 32. To distinguish 'wd shifts the rank attractor' from 'seeds are noisy at this scale' you need an effect size > seed std. With n=3 seeds per cell and observed sd ~5, MDE for two-cell comparison is ~12 rank units at α=.05/power=.8 (basic two-sample t). A wd-induced rank delta below 12 will be UNDETECTABLE. The compilation's '~$5' budget buys statistical theatre. Either run 8+ seeds per cell or downgrade E to 'descriptive only'.

## [2026-04-28T16:27:06.410Z] empirical-skeptic

ATTACK 5 — Compute realism and timeline.

Stated budget: 9 cells × 3 seeds × 2 H100h = 54 H100h ≈ $80 spot.

Reality check from prior runs:
  - Round 4 (4 nonlinear readouts) took 107 min/seed = 1.78h
  - Adding entropy reward: SVD backward via svdvals on 16x16 per latent position × 6 positions × batch 16 × ~1100 batches/epoch × 25 epochs. Each svdvals backward is O(d^3) but with FP32 promotion. Expect +20-40% step time.
  - ProsQA-MULTI eval is multi-label, needs new eval loop, not free.
  - 3 seeds × 9 cells × 2.5h = 67.5h H100. At spot $1.50/h = $101.

PLUS:
  - Dataset generation debugging (~0.5 day CPU)
  - First-pass crash recovery on 9 new training scripts: budget 30% reruns = +20h
  - 3 lambda values means it's actually 9 cells × 3 λ = 27 entropy cells if you sweep λ. Did the proposers count this?

REVISED ESTIMATE: 100-150 H100h, $150-225 spot, 3-4 calendar days of pod uptime.

DEADLINE: May 8 = 10 days. Subtract: paper writeup (3 days), figure regen (1 day), reviewer-comment buffer (1 day) = 5 days for experiments. Tight but feasible IF nothing breaks.

CRITICAL PATH: ProsQA-MULTI generator MUST land day 1. If it doesn't, no entropy/nuclear cells can run and the paper has no §6.

RECOMMENDATION: Cut to 2x2 (no nuclear-norm column, no ProsQA-MULTI-4 row). 4 cells × 3 seeds × 2.5h = 30h ≈ $45. Faster, more robust, still tells the headline story (rank-1-solvable vs rank-2-required × CE vs entropy).

## [2026-04-28T16:27:09.716Z] ideator

IDEA #1 — Jacobian erank measurement (the paper's OWN unrun §5.3 experiment). Compute J(Z)=∂φ/∂vec(Z) at test inputs for each of the 5 positive-control checkpoints and report erank(J). Paper PREDICTS ≈1; says ≥4 falsifies. Nobody has done it. ~$3, 1 H100h, no retraining. Highest info-per-dollar of any direction here. Either turns §5.3's hedge into a result-grade contribution OR falsifies the candidate hypothesis BEFORE submission (which is when you want to know).

## [2026-04-28T16:27:11.561Z] methodological-skeptic

ATTACK 8 (the strongest reviewer-killer that survives all experiments) — 'Rank' as defined here (effective rank of d=32 latent matrix Z) is a property of an INTERNAL bottleneck whose causal role in the answer is mediated by 6 sequential GPT-2 transformer applications. Even if every experiment works, a reviewer asks: 'why does effective rank of THIS particular intermediate tensor matter — as opposed to effective rank of the post-LayerNorm h_out, or the answer-position attention pattern, or the wte readout logits?' The paper has no answer. The choice of Z as 'the' observable is architectural-arbitrary. Required fix: a CAUSAL intervention experiment showing that randomizing Z's rank-k subspace (while leaving everything else equal) changes accuracy. Currently rank_project_k does this at eval — but only as a static ablation. Need: train-time intervention with matched param count to prove Z's rank is the load-bearing variable, not just correlated.

## [2026-04-28T16:27:17.590Z] ideator

IDEA #2 — Sparse autoencoder on Z latents. Train a JumpReLU SAE on flattened Z extracted from the Round 3 γ=0 checkpoint across all 6 latent positions × 17,886 ProsQA examples. NOVELTY: nobody has trained an SAE on continuous-CoT latents — directly tests Wang 2025's alternative substrate hypothesis (superposition lives in feature/vocab basis, NOT SVD basis). If SAE recovers crisp ProsQA features (target-class indicators, distractor features) where rank-k truncation recovered nothing, the paper's framing flips: rank is the wrong axis, not the loss. Falsifiable: SAE features no better than random-h baseline ⇒ Wang's substrate also dead in matrix-CODI. ~$20.

## [2026-04-28T16:27:25.152Z] ideator

IDEA #3 — Adam-vs-SGD-vs-wd=0 seed-spread channel ablation. The paper's mystery {4,12,13} effective-rank spread across seeds is unaddressed by the 3x3. Run 3 seeds × {AdamW wd=0.01, AdamW wd=0, SGD+momentum wd=0.01}. 'Rich and the Simple' (NeurIPS 2025) predicts SGD tightens to ~rank 4. Kobayashi 2024 (already cited) predicts wd=0 → high rank. Falsifiable: if all three optimizers reproduce the {low,mid,high} spread, optimizer-implicit-bias is NOT the channel — points to data-trajectory chaos. Standalone workshop note + paper appendix. ~$25 (9 runs × 1.5 H100h).

## [2026-04-28T16:27:31.965Z] ideator

IDEA #4 — Factored matrix-CODI: born-rank-r bottleneck. Reparametrize so Z = U V^T with U,V ∈ ℝ^{d×r} for r ∈ {1,2,4}. Z is born rank-r. NOVELTY: factored continuous-CoT bottlenecks untested. Sharpens the §7 'rank-1-solvable task' alternative explanation: if a born-rank-2 model still survives k=1 truncation, the task is rank-1-solvable, not the objective rank-blind. If a born-rank-2 model BREAKS at k=1, the architecture (not the loss) was the bottleneck. Falsifiable: r=2 → k=1 drop > 1pp ⇒ rejects strong rank-blindness reading. ~$30 (4 configs × 3 seeds × 1.5 H100h).

## [2026-04-28T16:27:39.487Z] ideator

IDEA #5 — Learned-projector rank-k retention (proper Li & Janson optimal ablation). Paper cites Li & Janson 2024: 'zero/resample ablations overestimate component importance vs optimal ablation.' Apply: instead of top-k SVD truncation, learn a rank-k projector P_k that maximally preserves unablated logits over 500 train examples, evaluate on test. NOVELTY: nobody has done learned-projector retention on continuous-CoT latents. If learned k=2 projector dramatically beats natural top-2, the SVD axes were just the wrong basis — the model uses 2 directions but they're not the dominant SVs. Sharpens Li & Janson's caveat into a probe upgrade. ~$5, no retraining.

## [2026-04-28T16:27:46.779Z] ideator

IDEA #6 — Rank-k on Rizvi-Martel's released COCONUT 96.6% checkpoint. The paper itself flags this in §6: 'rank-k ablation on Rizvi-Martel's released checkpoint (github.com/michaelrizvi/coconut) is a natural camera-ready experiment that would test whether the same rank-blindness holds at the 96.6% ceiling.' Their COCONUT uses VECTOR latents — reshape 768 → 24×32 fake matrix to compute SVD. NOVELTY: moves the mechanism claim out of matrix-CODI specifically into latent-CoT generally. Flat curve at 96.6% would be a much stronger paper headline. Falsifiable. ~$5, one H100h.

## [2026-04-28T16:27:55.339Z] empirical-skeptic

ATTACK 6 — Prior negatives on entropy regularization & continuous-latent rewards.

Found relevant 2025-2026 priors:

(P1) arXiv:2510.10959 'Rediscovering Entropy Regularization' (Oct 2025): explicitly says 'entropy regularization is often omitted in recent LLM training pipelines because its effectiveness is highly sensitive to the entropy coefficient choice — small coefficients cannot prevent entropy collapse, while excessively large coefficients may induce entropy explosion.' Confirms ATTACK 1's lambda-window worry as a documented LLM-pipeline phenomenon.

(P2) arXiv:2512.11816 'Reinforcement Learning for Latent-Space Thinking' (Dec 2025): RL-trained latent CoT models LAG behind language-space CoT on mathematical reasoning. 'Extended training on GSM8K-Aug-NL reduces generalization' — suggests adding any reward to continuous CoT can hurt the underlying task. Already cited in the compilation as a NO-GO reason for GRPO.

(P3) arXiv:2405.14544 (Scarvelis NeurIPS 2024) does Jacobian nuclear norm — penalizes magnitude not entropy. Not direct prior negative but bounds the design space.

(P4) arXiv:2604.22709 'Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought' (April 2026, very recent) — need to check if they tried rank-aware rewards.

IMPLICATION: The space of 'auxiliary loss on continuous CoT activations' has consistent failure pattern — gains depend on exact coefficient, often hurts task. The experiment is on the wrong side of the prior. We need to articulate WHY our auxiliary loss escapes the trap.

The defensible answer: H is BOUNDED [0, log d=2.77 for d=16], so 'entropy explosion' can't happen the way it does in policy entropy. But entropy COLLAPSE (rank 1 attractor) absolutely can happen and is what we're trying to fight. So the lambda lower bound matters and the upper bound less so. But if we can't tune around it... null result.

I want to see the ideator commit to a defined search protocol over λ before we burn the budget.

## [2026-04-28T16:27:57.622Z] ideator

RECOMMENDATION: The 3x3 is safe but extends the existing frame. The single highest-leverage move is **#1 (Jacobian erank measurement, ~$3)** — it's the paper's OWN deferred camera-ready experiment, runs on existing checkpoints, and either turns §5.3's hedge into a result-grade finding or falsifies the candidate hypothesis BEFORE submission. Do this regardless of the 3x3. If the team wants a *more novel paper*, swap part of the 3x3 for **#2 (SAE on Z, ~$20)** — it asks 'is there ANY structure in Z, even though rank says no?' which is a different paper and connects matrix-CODI to mainstream interp. Combined #1+#5+#2 = $28 for three NEW results the 3x3 ($90) does not produce.

## [2026-04-28T16:28:16.174Z] empirical-skeptic

ATTACK 7 — May 8 deadline triage. What's the BARE MINIMUM result that justifies §6 of the paper?

Working backwards from the deadline:
  - May 8 (deadline) - 1 day reviewer buffer = May 7 freeze
  - May 7 - 2 days writeup + figures = May 5 last-experiment-finish
  - May 5 - today (Apr 28) = 7 calendar days for experiments

7 days × ~16 productive H100h/day (assuming 8h pod uptime + parallelism) = ~110 H100h. NOT the proposers' 54 — fewer if anything goes wrong.

MINIMUM VIABLE PAPER §6 (in priority order, each adds an independent claim):

T1 [day 1, MUST]: ProsQA-MULTI-2 dataset generator + sanity check (D4: rank-1-constrained training matches unconstrained → KILL the whole direction).
   COST: 0.5 day CPU + 1 H100h sanity. STOP CONDITION: if D4 fails, pivot to T0 = wd=0 implicit-bias arm only.

T2 [days 2-3]: Standard CE training on ProsQA-MULTI-2, 3 seeds, rank-k curve.
   COST: 3 × 2.5h = 7.5h H100. PAPER VALUE: tests whether task alone bends the curve. Either result is publishable.

T3 [days 3-4]: Entropy reward on ProsQA-MULTI-2, λ ∈ {0.1, 1.0}, single seed each first; 3 seeds at best λ.
   COST: 2 + (3×2.5) = 9.5h. PAPER VALUE: the headline 2x2 cell.

T4 [day 5, OPTIONAL]: Entropy reward on ProsQA-1 (control: rank curve should stay flat — no rank to use).
   COST: 3 × 2.5h = 7.5h. PAPER VALUE: completes the 2x2.

TOTAL CRITICAL PATH: ~25 H100h, ~$40 spot, 5 days.

CUT FROM ROADMAP: nuclear-norm column (no clean mechanism advantage over entropy), ProsQA-MULTI-4 (k=4 doesn't add a different finding from k=2 if the curve bends at k=2), wd-sweep (appendix-grade per compilation E).

If T1 D4 falsifies the construction, the entire paper §6 collapses to a wd=0 implicit-bias note. We MUST run T1 first.

## [2026-04-28T16:29:04.432Z] empirical-skeptic

VERDICT: conditional_go.

Headline asks for the team:
1. Commit pre-experiment to D4 (rank-1-constrained training on ProsQA-MULTI-2 as kill-switch). If MULTI-2 is rank-1-solvable like MULTI-1, the entire 3x3 paper §6 collapses.
2. Commit pre-experiment to a lambda falsification rule. My proposal: 'rank-k curve still flat at the λ achieving Acc within 2pp of CE-only baseline = NULL result.' No try-one-more-λ loophole.
3. Cut nuclear-norm column. Compilation C admits ‖Z‖_* is the wrong functional (grows σ_1, doesn't spread). It would need orthogonality penalty to mean anything, at which point it's a different experiment.
4. Bake D1-D3 (per-direction ablation, per-direction probe, Jacobian erank) into eval scripts BEFORE any training cell launches. Otherwise we can't distinguish 'rank functional' from 'rank inflated noise.'

Outstanding to ideator: please articulate the precise mechanistic difference between H(σ̃) and §5's MLP(svdvals(Z)) that predicts a different outcome. If the answer is 'we add a reward, they only added inputs', that's the start, not the finish — model still satisfies both losses with rank-1 functional + rank-d magnitude-uniform Z. Need a story for why the optimizer wouldn't take that joint minimum.

Will respond to ideator's NEW directions when posted.

## [2026-04-28T16:30:02.706Z] validator

**VALIDATOR: Final synthesis — 8 attacks evaluated, queue ranked.**

**Attacks that LANDED (forces revision):**
- meth-skeptic #1: "bilinear head fiction" — removes u_i^T Z v_i framing; restated as bottleneck capacity
- meth-skeptic #2 (CRITICAL): position-decomposition — 6 Z_p each rank-1 can encode 6 targets; truncation per-Z_p doesn't constrain this. REQUIRES per-position-rank-1-forcing arm (replaces nuclear-norm column)
- meth-skeptic #3: embedding Gram matrix check mandatory before claiming k-orthogonality
- emp-skeptic #4 (CRITICAL): W_down can memorize Q→{a1..ak} at rank-1 if Q names property P. D4 (rank-1-constrained training) is the gate experiment before ANY headline run
- emp-skeptic #3: D1/D2/D3 diagnostics mandatory post-entropy-reward to distinguish rank-functional vs rank-noisy
- meth-skeptic #6: nuclear-norm column cut (predicted-null, $27 waste)
- meth-skeptic #5: spectral SIM-CoT / anchored directions KILLED (SVD backward pathology at near-uniform sigma in PyTorch 2.4)
- reviewer-2 tautology: addressed by 3-condition factorial — std-CE+MULTI-2 flat is the KEY control

**Attacks that did NOT land:**
- emp-skeptic #2: "entropy ≈ SVD-aug" — mechanistically distinct: SVD-aug gives σ to readout (optimizer can ignore); entropy directly forces σ uniform via separate loss term. Different intervention class.

**Surviving queue (ranked, ~$64 core / $82 with contingent):**
1. MULTI-2 pre-validation [GATE] — D4 rank-1-constrained training + D5 vanilla SFT + Gram matrix check (~$14). If rank-1-constrained within 2pp → redesign task before proceeding.
2. MULTI-2 + std-CE [baseline] — flat predicted; confirms objective still rank-blind on multi-answer task (~$9)
3. MULTI-2 + entropy reward [HEADLINE] — must bend at k=2; λ sweep {0.01, 0.1, 1.0}; pre-commit nullification rule (~$18); D1/D2/D3 + causal-patch post-diagnostics
4. ProsQA-1 + entropy [null control] — must stay flat; validates task-dependence of entropy effect (~$9)
5. Per-position rank-1 forcing [REPLACES nuclear-norm] — falsifies position-decomposition hypothesis; if MULTI-2 acc stable under rank-1-per-Z training → position-decomp wins (~$9)
6. wd=0 sweep [appendix] — tests Kobayashi 2024 wd↔nuclear-norm for {4,12,13} spread (~$5)
7. MULTI-4 [contingent Day-6 gate] — only if #3 clean positive (~$18)

**Reviewer-2 answered?** PARTIAL. Factorial structure answers tautology. Missing: causal-patch test (zero σ_k u_k v_k^T from A into B, observe target-k flip). Add as zero-GPU inference diagnostic on #3 checkpoints.

**Recommendation:** Start #1 (CPU MULTI-2 gen + D4 pre-check) TODAY before any GPU spend. Run #5 and #6 in parallel. Gate everything on #1's pass/fail.

Full QUEUE.md in validator worktree.
