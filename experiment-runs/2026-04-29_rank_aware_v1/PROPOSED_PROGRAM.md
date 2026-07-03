# Proposed Experimental Program — Position-Decomposition as a Standalone Paper

**Goal:** convert the rank-aware-v1 finding ("matrix-CODI composes via *position*, not *within-position spectral rank*") into a defensible standalone contribution to mechanistic interpretability and continuous-CoT literature, on a budget of ≤ 30 H100h (~$30 spot at $1/h).

**Builds on (does not replace):** the ICML-MI workshop paper "The Gradient Does Not See Rank" (Sections 3, 5, 5.5, 7). That paper establishes flat rank-k curves and falsifies the readout-linearity-only mechanism. This program advances a *positive* mechanism — position-as-decomposition-axis — and tests it causally.

---

## 1. Hypothesis statements

We commit to the following numbered hypotheses; each experiment is mapped to the H it tests and the outcome that would falsify it.

- **H1 (rank-1 sufficiency).** Forcing Z to rank-1 throughout training does not reduce accuracy on any matrix-CODI task we have tested. Equivalently, rank-k truncation at eval time on the unconstrained baseline is flat down to k=1, and the force-rank=1 checkpoint matches the baseline at every k. *Falsifier:* baseline beats force-rank=1 by ≥ 2pp at matched seeds and N, on any task where vanilla matrix-CODI exceeds chance.
- **H2 (position scaling).** Accuracy on matrix-CODI scales with the number of latent positions n_latents, but the per-position effective rank stays ~1 regardless of n_latents. *Falsifier:* n_latents=1 reaches the same accuracy as n_latents=6 (positions are not a useful axis), OR per-position erank grows materially (e.g. ≥ 3) as n_latents shrinks (model is recruiting within-position rank when starved of positions).
- **H3 (position non-exchangeability).** At inference, randomly shuffling the order of the trained latent positions causes a severe accuracy drop. Positions encode position-specific information. *Falsifier:* shuffle-eval accuracy is within 2pp of unshuffled (positions are a "bag of rank-1 directions" — order does not matter, so the structure is not position-indexed).
- **H4 (within-position content is position-specific).** A linear probe on Z_p (the p-th latent position) best predicts task content indexed by p (e.g. reasoning step p, target k=p in MULTI-k tasks). *Falsifier:* probes on every Z_p achieve similar AUC for every step k (positions are interchangeable carriers), OR no Z_p achieves above-chance step prediction (positions don't carry structured content).
- **H5 (cross-task generality).** H1–H4 hold across ProsQA-1, ProsQA-MULTI-2, ProsQA-MULTI-4, and (stretch) GSM8K-Aug. Position-decomposition is not an artefact of a single task family. *Falsifier:* on at least one task, force-rank=1 hurts ≥ 2pp OR shuffle-eval is within 2pp of unshuffled.

---

## 2. Per-experiment specs

Each cell below: name → hypothesis → decision criterion → compute estimate → dependency.

Compute base rate: a single matrix-CODI run on ProsQA-1 (5000 train / 500 test, 25 epochs, batch 16, gpt2-small, d=16, n_latents=6) takes ~2 H100h on H100-80GB (extrapolated from the rank-aware-v1 A100-PCIe runs). MULTI-2 / MULTI-4 use the same dataset size — same wall-clock. n_latents scaling is approximately linear in the bottleneck pass; we use 1.0× for n_latents=6 default, 0.6× for n_latents=1, 1.5× for n_latents=12. GSM8K-Aug at standard CODI scale is ~5 H100h.

### E1 — Headline rank-k confirmation on rank-aware-v1 checkpoints

- **Hypothesis tested:** H1.
- **Method:** load the existing baseline (seed 7, 13.22 erank) and force-rank=1 (seed 1337 and seed 42) checkpoints. Evaluate rank-k truncation at k ∈ {1, 2, 4, 8, 16} on a 500-problem ProsQA-MULTI-2 test split. Report the curve and Spearman r_s.
- **Decision criterion:** baseline curve flat to ≤ 1pp across k AND force-rank=1 curve flat to ≤ 1pp across k AND k=1 accuracy of baseline within 2pp of k=1 accuracy of force-rank=1.
- **Compute:** 0.3 H100h (eval only on existing checkpoints).
- **Dependency:** none — runs first, gates the rest.
- **Why first:** if E1 fails, the position-decomposition story is wrong on its own training data and the rest is moot.

### E2 — n_latents sweep (position scaling)

- **Hypothesis tested:** H2.
- **Method:** train matrix-CODI on ProsQA-MULTI-2 with n_latents ∈ {1, 2, 3, 6, 12}, two seeds each (seeds 1337, 42). γ=0, d=16, batch=16, 25 epochs. Measure (a) best test accuracy, (b) per-position effective rank averaged over test set, (c) total effective rank summed across positions.
- **Decision criterion:**
  - Accuracy at n_latents=6 ≥ accuracy at n_latents=1 by ≥ 5pp (positions help).
  - Per-position erank ≤ 2.0 across all n_latents settings (positions stay rank-1).
  - Total erank (sum across positions) tracks n_latents linearly (r ≥ 0.85).
- **Compute:** 5 settings × 2 seeds × {0.6, 0.7, 0.8, 1.0, 1.5}× = ~9.2 H100h. Round to **10 H100h**.
- **Dependency:** none (parallel with E1).

### E3 — Position-shuffle eval (causal evidence)

- **Hypothesis tested:** H3.
- **Method:** on the rank-aware-v1 baseline checkpoint and the n_latents=6 baseline from E2, run inference under three conditions: (a) unshuffled (control), (b) random permutation of the 6 latent positions per problem, (c) reverse-order positions. Three random-permutation seeds. Report accuracy mean ± SD.
- **Decision criterion:** shuffled accuracy ≥ 5pp below unshuffled (H3 supported). If shuffled within 2pp of unshuffled, H3 falsified.
- **Compute:** 0.3 H100h (eval only).
- **Dependency:** uses rank-aware-v1 + E2 checkpoints. Can run end of E2.

### E4 — Within-position content probing

- **Hypothesis tested:** H4.
- **Method:** on the n_latents=6 baseline (rank-aware-v1 baseline checkpoint), extract Z_p for each p ∈ {0..5} on 500 test problems. Train 6 linear probes (one per position) for each of: (a) target k for MULTI-2 (k=1 vs k=2), (b) reasoning step index in the underlying ProsQA chain (computed from the DAG path). Report a 6×K AUC matrix (positions × content-axes). Also: a *cross-position* probe — train on Z_p, test on Z_q (p≠q) — to test whether positions are interchangeable.
- **Decision criterion:**
  - The diagonal of the position×step AUC matrix exceeds the off-diagonal mean by ≥ 0.10 AUC (position p best predicts step p / target p).
  - Cross-position probes degrade by ≥ 0.10 AUC vs within-position (positions are not interchangeable carriers).
- **Compute:** 0.2 H100h (CPU-bound + small GPU for hidden-state extraction).
- **Dependency:** rank-aware-v1 baseline checkpoint + ProsQA-MULTI-2 step labels (need to generate from the DAG; CPU job, ~1h).

### E5 — Cross-task generality battery

- **Hypothesis tested:** H5.
- **Method:** train baseline (unconstrained) and force-rank=1 matrix-CODI on three tasks: ProsQA-1, ProsQA-MULTI-2 (already done in rank-aware-v1), ProsQA-MULTI-4. Two seeds each. For each, run E1 (rank-k curve) + E3 (shuffle) on resulting checkpoints. Stretch goal: GSM8K-Aug at the paper's R1 operating point if budget allows.
- **Decision criterion:**
  - On all 3 ProsQA-flavored tasks: |baseline_acc − forceRank1_acc| ≤ 2pp.
  - On all 3 tasks: shuffled accuracy ≥ 5pp below unshuffled.
- **Compute:** ProsQA-1 + ProsQA-MULTI-4 each need 2 seeds × 2 conditions = 4 runs × 2 H100h = 8 H100h per task = 16 H100h. ProsQA-MULTI-2 already done (rank-aware-v1). GSM8K-Aug stretch: 4 runs × 5 H100h = 20 H100h (over budget — flagged as camera-ready).
- **Dependency:** E1 must pass; E2 informs whether n_latents needs adjustment for MULTI-4.

---

## 3. Total compute table

| Cell | Description | H100h | $ at $1/h | Priority |
|------|-------------|------:|----------:|:--------:|
| E1 | Rank-k confirm on existing checkpoints | 0.3 | 0.30 | P0 (first) |
| E2 | n_latents ∈ {1,2,3,6,12} × 2 seeds on MULTI-2 | 10.0 | 10.00 | P0 |
| E3 | Position-shuffle eval | 0.3 | 0.30 | P0 |
| E4 | Within-position probing | 0.2 | 0.20 | P0 |
| E5a | ProsQA-1: 2 seeds × 2 conds (baseline, rank=1) | 8.0 | 8.00 | P1 |
| E5b | ProsQA-MULTI-4: 2 seeds × 2 conds | 8.0 | 8.00 | P1 |
| **Subtotal (core, P0+P1)** | | **26.8** | **26.80** | within budget |
| E5c (stretch) | GSM8K-Aug × 2 seeds × 2 conds | 20.0 | 20.00 | P2 (camera-ready) |
| Buffer (preempts, debugging) | | 3.0 | 3.00 | reserve |
| **Total submitted-paper compute** | | **~30** | **~30** | on budget |

If E1 fails → stop, revise hypothesis. If E2 shows per-position erank > 2 at n_latents=1, the model is "spilling" within-position rank when starved — this is itself a paper-grade finding (position-and-rank trade-off) and changes the framing rather than killing it.

---

## 4. Falsification criteria (paper-killing thresholds)

The position-decomposition story dies if **any one** of:

1. **E1 fails:** force-rank=1 checkpoint loses ≥ 2pp at k=1 vs baseline at k=1. Means rank IS used and we've been wrong about ProsQA-MULTI-2.
2. **E2 fails (positions):** n_latents=1 matches n_latents=6 within 2pp. Means positions are not actually a compositional axis — the model is doing something else (e.g. only the final position matters).
3. **E2 fails (rank):** per-position erank at n_latents=1 is ≥ 4. Means the model recruits within-position rank when positions are scarce — the rank-1 result is conditional on having ≥ 6 positions, not a property of the architecture.
4. **E3 fails:** position-shuffled accuracy is within 2pp of unshuffled. Means positions are a bag, not an indexed axis. Position-decomposition becomes "rank-1 vector ensemble," weaker claim.
5. **E4 fails:** position p does not predict step p above off-diagonal. Means positions are not carrying position-specific content — the per-position rank-1 directions are redundant copies, not a decomposition.
6. **E5 fails on ProsQA-1:** force-rank=1 hurts on ProsQA-1 even though it doesn't on MULTI-2. Would be a strange inversion suggesting the simpler task uses rank but the harder one doesn't — would force a re-think.

A *partial* falsification (e.g. E5 on ProsQA-MULTI-4 fails but ProsQA-1 and MULTI-2 pass) becomes a scoped paper: "position-decomposition holds for 1- and 2-target ProsQA tasks; multi-target tasks recruit rank."

---

## 5. Paper plan

### If all results land as predicted

**Title (working):** *Position is the Compositional Axis: Matrix-CODI Decomposes Reasoning Across Latent Positions, Not Spectral Rank*.

**Abstract framing:**

> Matrix-valued continuous chain-of-thought introduces two structural axes for compositional reasoning: spectral rank within each matrix latent, and position across multiple latents. Prior work (companion paper, ICML-MI 2026) shows that the rank-k truncation curve is flat across training regimes, falsifying rank as the carrier. We provide positive evidence that *position* is. Forcing every latent matrix to rank 1 throughout training preserves accuracy across ProsQA, ProsQA-MULTI-2, and ProsQA-MULTI-4 (n=2 seeds each, all gaps ≤ 2pp). Accuracy scales with the number of latent positions while per-position effective rank stays ≈ 1.0 regardless of n_latents (n=10 cells). At inference, randomly shuffling the order of latent positions drops accuracy by ≥ X pp on every task, providing causal evidence that positions are not exchangeable. Linear probes recover step-specific content from each Z_p in alignment with its position index, with a ≥ 0.10 AUC drop under cross-position probing. Together, these establish position-as-decomposition-axis as the operative compositional structure of CODI-trained matrix latents and confirm that the d² spectral parameters of the matrix bottleneck are vestigial.

**Three claims, in priority order:**

1. **Primary (E1+E2+E5):** Continuous-CoT matrix-bottleneck models compose via position, not spectral rank, regardless of task structure or matrix capacity.
2. **Secondary (E1+E5):** Forcing rank-1 throughout training is a clean, lossless way to remove the matrix bottleneck's spectral capacity, confirming the d² parameters are vestigial. This is also a practical compression result: rank-1 matrix-CODI is a 16× parameter reduction in Z with no accuracy cost.
3. **Tertiary (E3+E4):** Position-shuffle invariance failure and within-position step-specific probing provide causal and correlational evidence respectively that position is the carrier of compositional structure.

**Sections:** intro / background (reuses workshop paper §2) / E1+E5 rank-1 sufficiency / E2 position scaling / E3+E4 position-as-axis evidence / E5 generality / discussion (relation to Rizvi-Martel illusion, SIM-CoT, Wang vocab-superposition) / limitations.

### If some results fail

- **E1 only fails:** withdraw the standalone paper; merge findings into workshop-paper revision flagging that ProsQA-MULTI-2 changes the picture.
- **E2 partially fails (n_latents=1 already strong):** reframe as "matrix-CODI uses neither rank nor positions — it's just a slightly different vanilla SFT." Heavy revision; possibly merge with the Illusion of Superposition framing.
- **E3 fails:** retitle to "Rank-1 Position-Distributed Latents" and downgrade the position-as-axis claim to "positions are a useful capacity dimension but not an indexed compositional structure." Still publishable; weaker contribution.
- **E5 fails on one task only:** scope the paper to the tasks where it holds; treat the failing task as the discussion's open question.
- **E4 fails (probes don't discriminate):** drop the within-position-content section; rely on E1+E2+E3 alone. Still a coherent paper.

### Causal-patch follow-up (not in budget)

The reviewer-2 ask from COMBINED_PLAN.md (zero out σ_k u_k v_k^T from a problem with target_k=A, feed into a problem with target_k=B, observe whether prediction flips for target k) is the natural next causal test. We flag it for camera-ready / follow-up.

---

## 6. Risks and mitigations

- **Pod preemption (rank-aware-v1 lost ~$8 to this).** Use a launcher that writes a per-experiment SUMMARY-of-runs incrementally. Pre-experiment dry-run: single forward step CPU-only on each CLI to catch flag mismatches.
- **Force-rank=1 implementation correctness.** Two routes: (a) train W_up to output a rank-1 factorization (Z = u v^T directly), (b) post-hoc SVD-truncate every batch. (a) is cleaner; (b) is a sanity baseline. Run both for E1 to confirm equivalence on at least one cell.
- **n_latents=12 OOM risk on 80GB.** Pre-tested at d=16: total bottleneck memory is ~12 × (768 × 256) ≈ 2.4M params per layer, well under VRAM. Should be fine at batch=16.
- **MULTI-4 dataset doesn't exist yet.** CPU job to generate (extend the MULTI-2 generator to k=4 targets); allocate 2h before E5b. No GPU cost.
- **Probe code reuse.** `probe_z.py` from the workshop paper extracts Z_p already; need only add the cross-position and step-index variants. ~half-day of scripting.

---

## 7. Sequencing

- **Day 0 (today):** generate MULTI-4 dataset; verify force-rank=1 implementation parity (a) vs (b). 0 GPU.
- **Day 1:** E1 (rank-k confirm, ~20 min) → E3 (shuffle, ~20 min) → kick off E2 sweep in parallel. Decision gate: E1 must pass before launching E5a/E5b.
- **Day 2:** E2 finishes; E4 probes on rank-aware-v1 baseline (CPU + small GPU).
- **Day 3:** E5a (ProsQA-1) + E5b (ProsQA-MULTI-4) launch in parallel, 8 H100h each.
- **Day 4:** Results consolidated; figures generated; draft outline.
- **Day 5–6:** Writing.

Total wall-clock: 6 days from launch to draft. ICML-MI deadline is May 8 — workshop paper submits there; this standalone targets a longer venue (NeurIPS 2026 main / ICLR 2027 / a follow-up workshop).
