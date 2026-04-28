# Final Experiment Queue (ranked)
*Validator synthesis — inputs from: synthesizer (2 posts), empirical-skeptic (4 attacks), methodological-skeptic (6 attacks), reviewer-2 (1 full review), pragmatist (1 reality check). Ideator not yet posted; queue is complete without it.*

---

## Attacks audit — what LANDED vs. bounced

### LANDED (forces revision or mandatory pre-check)

| Attack | Source | Severity | Disposition |
|--------|--------|----------|-------------|
| "Bilinear head u_i^T Z v_i is fictional — actual readout is W_down·vec(Z)→GPT-2→LM head" | meth-skeptic #1 | High | Removes 'bilinear heads on Z' language; rank-k-by-orthogonality framing must be restated as bottleneck information capacity, not probe geometry |
| "Position-decomposition: 6 Z_p each rank-1 encodes 6 independent targets; truncation per-Z_p leaves them all intact" | meth-skeptic #2 | Critical | Requires: (a) mandatory D4 pre-check, (b) add per-position-rank-1-forcing arm to REPLACE nuclear-norm column |
| "Embedding tying: GPT-2 wte not orthogonal; must measure Gram matrix of MULTI target embeddings" | meth-skeptic #3 | High | Mandatory pre-experiment 10-line check; gates A proceeding |
| "W_down can memorize question→{a1..ak} at rank-1 if question names property P" | emp-skeptic #4 | Critical | Mandatory D4 (rank-1-constrained matrix-CODI), D5 (vanilla SFT), D6 (question-only MLP) before headline experiments |
| "Must distinguish rank-functional vs rank-noisy post entropy-reward; D1/D2/D3 diagnostics required" | emp-skeptic #3 | High | Mandatory post-training diagnostics baked into experiment design |
| "Nuclear-norm column is rhetorical — predicted-null 3 cells waste $27 compute" | meth-skeptic #6 | Medium | Nuclear norm CUT from primary plan; replaced by per-position-rank-1 forcing arm |
| "SVD backward at near-degenerate sigma → NaN for U/V-dependent losses; (D) needs custom autograd" | meth-skeptic #5 | High | Anchored singular directions (D) → KILLED for this submission |
| "Reviewer-2 tautology: constructed task requires rank k, entropy maximizes rank k, observed rank k" | reviewer-2 | Critical | Addressed by 3-condition factorial structure: std-CE + MULTI-2 is the KEY control showing task alone doesn't fix rank-blindness |

### DID NOT LAND (vague or addressed by design)

| Attack | Source | Why bounced |
|--------|--------|-------------|
| "Entropy ≈ SVD-aug therefore pre-empted by §5" | emp-skeptic #2 | SVD-aug exposes σ to readout (optimizer can ignore it); entropy reward directly forces σ uniform via separate loss term — mechanistically distinct. Different intervention class. |
| "Lambda window may not exist" | emp-skeptic #1 | Real risk but handled by pre-commit nullification rule (flat at acc within 2pp = experiment nulls). Sweep λ ∈ {0.01, 0.1, 1.0} is standard. |
| "§5.5 negative control doesn't generalize to MULTI" | meth-skeptic #4 | Correctly noted but doesn't kill MULTI — it means we can't use §5.5 as standalone justification for MULTI being rank-k. Handled by D4/D5 pre-checks. |

---

## Final Experiment Queue (ranked)

### #1 ProsQA-MULTI-2 Pre-Validation (mandatory gate)

- **Hypothesis:** ProsQA-MULTI-2 is NOT rank-1-solvable under the actual model/readout — i.e., a rank-1-constrained matrix-CODI achieves materially lower accuracy than unconstrained
- **Decision criterion:** If rank-1-constrained acc (Z projected to rank-1 before readout, at EVERY step, during TRAINING) is within 2pp of unconstrained → task IS rank-1-solvable → STOP, redesign MULTI construction before proceeding to #3. If gap ≥ 3pp → proceed.
- **Compute (spot $):** ~$14 ($9 rank-1-constrained matrix-CODI × 3 seeds + $5 vanilla SFT × 3 seeds). GPU-bound runs ~1–2 H100h/seed.
- **Dependency:** None — this is the gating experiment. Also run: (a) embedding Gram matrix of MULTI-2 target tokens [free, CPU], (b) question-only MLP baseline [free, CPU]. Both must confirm rank-k is not trivially satisfied before proceeding.
- **Smoke-test plan:** Forward pass with rank-1-forced Z (truncate_to_rank(Z, 1) at each bottleneck step, not just eval). Check gradients flow through Z without NaN. Verify effective_rank of Z ≈ 1 throughout training curve.
- **Predicted outcome:** Rank-1-constrained acc drops ≥ 5pp on MULTI-2 (task requires >1 independent direction). If it doesn't drop, the position-decomposition attack (meth-skeptic #2) has won and we need MULTI-k with k > 6 or per-answer readout isolation.

---

### #2 ProsQA-MULTI-2 + std-CE (baseline control)

- **Hypothesis:** Standard CE training on MULTI-2 produces a flat rank-k ablation curve despite the task requiring rank ≥ 2 by construction — objective rank-blindness holds even on multi-answer tasks
- **Decision criterion:** CONFIRM = all k ∈ {1,2,4,8,16} within 2pp → task alone doesn't fix rank-blindness. FALSIFY = k=1 accuracy drops ≥ 3pp below k=16 → task is itself forcing rank usage (story reshapes: the task fix works without any reward)
- **Compute (spot $):** ~$9 (3 seeds × ~2 H100h × $1.50/h)
- **Dependency:** #1 must CONFIRM (rank-1-constrained fails)
- **Smoke-test plan:** Train 1 epoch, check accuracy climbing above baseline (MULTI-2 harder than MULTI-1 → expect lower ceiling ~60–70pp). Check rank-k eval returns valid accuracy distribution.
- **Predicted outcome:** FLAT (consistent with Prop. 1 — objective still doesn't see rank even when task needs it). If FLAT, CONFIRMS that entropy reward is the needed intervention and #3 is properly motivated. Post-training MANDATORY: log cosine(u_i, u_j) between bilinear probe pairs.

---

### #3 ProsQA-MULTI-2 + Entropy Reward (headline experiment)

- **Hypothesis:** Adding +λH(σ̃) to CE loss causes the rank-k ablation curve to BEND on MULTI-2 — i.e., k=1 accuracy is materially lower than k=2 — because the reward forces σ toward uniformity, which is only compatible with functional multi-direction usage when the task requires it
- **Decision criterion:** CONFIRM = acc(k=1) < acc(k=2) by ≥ 3pp at λ where overall acc within 2pp of std-CE baseline. FALSIFY = curve flat at every λ in {0.01, 0.1, 1.0} where acc is within 2pp of baseline → entropy reward insufficient; experiment NULLS (do not seek more λ values).
- **Compute (spot $):** ~$18 (3 λ values × 3 seeds × ~2 H100h = ~$27; can narrow to 2 λ after first seed = ~$18)
- **Dependency:** #1 PASS, #2 runs in parallel or before; must be FLAT to properly interpret #3
- **Smoke-test plan:** Single-seed forward+backward at each λ. Verify ∂H/∂Z flows nonzero. Check σ distribution becomes more uniform over training (log effective rank rising). Check no NaN from svdvals backward (should be stable).
- **Predicted outcome:** Bends at λ=0.1 with acc within 2pp of baseline. Post-training MANDATORY diagnostics: D1 = per-singular-direction ablation (zero σ_k, measure ΔAcc — expect ΔAcc(σ_1) ≈ ΔAcc(σ_2) if rank functional); D2 = linear probe on each σ_k u_k v_k^T slice; D3 = erank of trained readout Jacobian J(Z) at test inputs (predict ≥ 2 if rank functional, ≈ 1 if noise).

---

### #4 ProsQA-1 + Entropy Reward (null control — essential for 3-condition story)

- **Hypothesis:** +λH(σ̃) on ProsQA-1 (rank-1-solvable) does NOT bend the rank-k curve — entropy reward only creates functional rank when the task demands it
- **Decision criterion:** CONFIRM = curve flat (within 2pp all k) → clean factorial story: task determines whether rank is functional, reward reveals it. FALSIFY = curve bends → entropy reward inflates non-functional rank (noise rank), undermines #3 interpretation.
- **Compute (spot $):** ~$9 (3 seeds × 2 H100h × $1.50/h, same λ as #3 optimal)
- **Dependency:** #3 optimal λ identified; can run in parallel with #3
- **Smoke-test plan:** Same as #3. Check effective rank increases from baseline ~12 (should — entropy pushes it to log(16) ≈ 2.77). Verify accuracy stays near 80pp.
- **Predicted outcome:** FLAT rank-k curve (confirming 3-condition story: ProsQA-1 is rank-1-solvable → entropy irrelevant to readout path → no functional rank change).

---

### #5 Per-Position Rank-1 Forcing (replaces nuclear-norm column — position-decomp falsifier)

- **Hypothesis:** If the model uses position-decomposition (encode answer_i at latent position i, all Z_p rank-1), then forcing Z_p to rank-1 at every position DURING TRAINING will not hurt MULTI-2 accuracy — position-decomp is already rank-1 everywhere
- **Decision criterion:** CONFIRM = rank-1-forced training acc ≈ unconstrained (within 3pp) → position-decomp IS the model's strategy (each position carries one answer, rank-1 per position). FALSIFY = acc drops ≥ 3pp → model NEEDS rank > 1 at some position (genuine within-position superposition).
- **Compute (spot $):** ~$9 (3 seeds × 2 H100h × $1.50/h; requires adding truncate_to_rank(Z, 1) in the training forward pass, not just eval)
- **Dependency:** #1 PASS (otherwise, if #1 fails, we already know rank-1 is insufficient)
- **Smoke-test plan:** Modify MatrixBottleneck.forward to call truncate_to_rank(Z, 1) unconditionally during training (not just eval). Verify gradient flows through the rank-1 truncation (SVD backward active). Check training loss decreases at normal rate.
- **Predicted outcome:** This experiment is the MISSING FALSIFIER of position-decomposition. If it CONFIRMS (rank-1 per position works fine), we know the model distributes multi-answer information across positions, not across singular directions within a position. Important mechanistic finding either way.

---

### #6 Implicit Bias wd=0 Sweep (appendix)

- **Hypothesis:** The observed {4, 12, 13} seed-spread in effective rank under AdamW + wd=0.01 is partly caused by weight decay's nuclear-norm-equivalence bias; removing wd=0 changes the distribution
- **Decision criterion:** If wd=0 produces a DIFFERENT rank distribution (e.g., concentrates at rank ~12 or spreads further) at same accuracy → confirms Kobayashi 2024 wd↔nuclear-norm path shapes rank. If distribution unchanged → optimizer implicit bias is NOT driving seed-spread; seed-sensitive initialization is the cause.
- **Compute (spot $):** ~$5 (3 seeds × ~1 H100h × $1.50/h, same arch as baseline R3a)
- **Dependency:** None — run in parallel with #1/#2
- **Smoke-test plan:** Standard train + eval. Check that training loss converges normally (wd=0 can cause slight instability; monitor loss curve first 1k steps).
- **Predicted outcome:** wd=0 reduces rank spread and pushes toward higher effective rank (without nuclear-norm implicit pull toward low-rank). If {12, 13, 14} under wd=0 vs {4, 12, 13} under wd=0.01 → supports the "wd tightens toward rank-1 attractor" reading from Kobayashi 2024.

---

### #7 ProsQA-MULTI-4 (contingent — Day 6 gate)

- **Hypothesis:** MULTI-4 (4 disjoint answers) bends at k=4 under entropy reward and stays flat at k=1 under std-CE
- **Decision criterion:** Same as MULTI-2 with k=4 inflection. ONLY RUN if #3 on MULTI-2 returns a CLEAN POSITIVE (acc(k=1) < acc(k=2) by ≥ 3pp).
- **Compute (spot $):** ~$18 (3 seeds × 2 conditions × 2 H100h; not started until Day 6 go/no-go)
- **Dependency:** #3 CLEAN POSITIVE. Do not run if #3 is ambiguous.
- **Smoke-test plan:** Same as MULTI-2 variants. Additional check: MULTI-4 may have lower accuracy ceiling (harder task) — verify model exceeds 50% before treating as informative.
- **Predicted outcome:** Steeper inflection than MULTI-2 under entropy reward (more rank is needed). If it works, this is the second panel of the headline figure.

---

## Addresses Reviewer-2's objection?

**Reviewer-2's killer objection:** "Tautology — you built a task requiring rank k, added a loss maximizing rank k, observed rank k."

**Does the surviving plan answer it?**

PARTIAL YES with specific gaps:

**Addressed by:**
1. #2 (MULTI-2 + std-CE) proves the task alone doesn't force rank — objective remains rank-blind on a task that requires rank. If #2 is flat, the story is: (task needs rank) + (std-CE can't see rank) → flat. Then (task needs rank) + (entropy rewards rank) → bends. This is NOT tautological — it separates the task structure from the training signal.
2. #4 (ProsQA-1 + entropy) proves entropy reward doesn't create functional rank when the task doesn't need it — entropy is rank-inflationary, not rank-forcing.
3. Pre-validation (#1) with the mandatory D4/D5 checks removes the "circular construction" attack — the task is shown to be rank-k by a HELD-OUT evaluation, not by construction assumption.

**Still unaddressed (reviewer-2's remaining requests):**
- **(c) Causal-patch test:** Zero out σ_k u_k v_k^T from a problem where target_k = A, feed into a problem where target_k = B, observe if prediction flips specifically for target k. This tests whether singular directions ARE specific reasoning paths (superposition) vs output channels (bookkeeping). **Recommend adding as a ≤1-day scripting task paired with #3's diagnostics.** Cost: 0 GPU (inference only on trained checkpoints).
- **(b) Higher latent budget cell:** MULTI-4 (experiment #7) partially addresses "harder task hurts ablation" but the concern is about whether matrix size d=16 is limiting. Not addressed; deferred to camera-ready.

**Assessment:** With the surviving plan (especially #1 pre-validation + #2 + #3 + #4 factorial), reviewer-2's objection is deflected but not fully answered. The causal-patch diagnostic should be added to #3's post-training diagnostics. Without it, the claim "singular directions encode separate reasoning paths" remains unsupported even if the curve bends.

---

## Compute summary

| Experiment | Spot $ | H100h | Required |
|------------|--------|-------|----------|
| #1 Pre-validation | $14 | ~9 | Gate — mandatory |
| #2 MULTI-2 std-CE | $9 | 6 | Mandatory |
| #3 MULTI-2 + entropy | $18 | 12 | Headline |
| #4 ProsQA-1 + entropy | $9 | 6 | Null control |
| #5 Per-pos rank-1 | $9 | 6 | Falsifier |
| #6 wd=0 sweep | $5 | 3 | Appendix |
| #7 MULTI-4 (contingent) | $18 | 12 | Day-6 gate |
| **Core (#1–#6)** | **$64** | ~42 | |
| **With contingent** | **$82** | ~54 | |

Critical path: #1 gates all others → #2 and #5 in parallel → #3 and #4 in parallel → #6 anytime → #7 if Day-6 go.

**Timeline fits $100 plan with 4 days to spare before May 8 deadline.**
