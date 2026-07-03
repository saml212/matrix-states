# Experiment Queue

**Last updated:** 2026-04-20 (superseded — see banner below)

This document is the engineering queue for future experiments. It contains the specs that an experimenter agent (or you) needs to actually run the experiments. Strategic narrative and project state live in [STATE.md](../STATE.md).

---

> **STATUS AS OF 2026-07-01 — everything below this banner is historical
> (matrix-CODI / workshop-paper era, all items closed or superseded).**
> The active queue moved to `matrix-thinking/chapter2/`:
> - Control A (Priority 0 below): **done**, folded into the accepted workshop paper.
> - Priority 1 (matrix-CODI rank dynamics): **done** — negative result (rank-blind),
>   this is the paper's headline finding.
> - Chapter 2 (Task D, tensor-product key/value binding): **done, CONFIRMED**
>   at d=8,16. Spec: `matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md`.
>   Write-up: `matrix-thinking/chapter2/TASK_D_WRITEUP.md`.
> - Chapter 2.5 (Task E, compositional reasoning transfer): **running now** on
>   the Brev 8×H100 cluster. Spec: `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md`.
> - Follow-on, not yet started: Stage 0 (d≥32 trainability precursor, blocks
>   the large-matrix tranche of Task E); the real-data reasoning-transfer step
>   (gated on Task E's result); byte-level input as an isolated follow-on
>   ablation once matrix-native scales to real data (see STATE.md's Path
>   Forward — tokenization must be held fixed, not bundled with the matrix
>   change).
>
> See STATE.md's "Chapter 2 — STATUS" section for the full current narrative.

---

## PRIORITY 0 (PRE-SUBMISSION): Control A — fake-Z rank-k ablation on vanilla GPT-2 SFT

**Status:** Specified. Must run before ICML MI Workshop 2026 paper submission (deadline 2026-05-08). Small — reuses existing vanilla SFT checkpoints (Round 4), ~1 GPU-hour.

### One-sentence hypothesis

On vanilla GPT-2 SFT for ProsQA (no matrix bottleneck, no CODI distillation), rank-k truncation of a fake-Z derived from the hidden state will also produce a flat accuracy curve — if and only if ProsQA itself is rank-1-solvable, which would mean the five matrix-CODI flat curves are non-evidence, not proof that CODI is rank-indifferent.

### Why this experiment matters for the paper

The paper's central mechanism claim is that the CODI distillation objective produces rank-indifferent gradients. §7 Discussion acknowledges "ProsQA might be rank-1-solvable" as the primary alternative explanation, and the rebuttal report (review/04_rebuttal_report.md, attacks A4 + A16) handles it by surfacing it in the paper. Control A converts that acknowledged caveat into either strong supporting evidence or a concrete result bound.

- **If the vanilla-SFT fake-Z rank-k curve is also flat**: the rank-k ablation method has no discriminating power on ProsQA. All five matrix-CODI flat curves are task-determined, not objective-determined. Paper must pull back on the "CODI is rank-blind" framing and reposition as "on ProsQA, no tested architecture/objective produces rank-dependent behavior — which we interpret as the task being rank-1-solvable." Paper still submits; framing shifts.
- **If the vanilla-SFT fake-Z rank-k curve bends**: the ablation method does discriminate when rank matters, and the matrix-CODI flat curves ARE informative about the objective. Paper's current framing is vindicated. Strong win.

Either outcome ships a better paper. Not running it leaves the primary A-list attack open.

### Protocol

Reuse one of the existing vanilla SFT seed checkpoints (already produced in Round 4):
- `experiment-runs/2026-04-13_round4_vanilla_sft/pure_sft_seed1337/` (or seed42/seed7)

Construct a "fake-Z" from vanilla GPT-2's hidden state at the same latent-position conceit used in matrix-CODI:
1. At answer position, take the 768-dim hidden state h.
2. Reshape first 256 entries into a 16×16 matrix `fake_Z` (matches matrix-CODI's d=16).
3. Apply the same rank-k truncation ablation used in matrix-CODI rank_projection_ablation: SVD, keep top k singular values, reconstruct, replace the corresponding dimensions in h.
4. Forward through the rest of the model to get logits, compute accuracy.
5. Sweep k ∈ {1, 2, 4, 8, 16}.

### Success / falsification criteria (pre-registered)

- Flat curve (Spearman |r| < 0.3 at k vs accuracy across the sweep, or range < 2pp): ablation method does not discriminate on ProsQA. Rewrite paper §3/§5 framing.
- Bending curve (monotone accuracy decrease as k decreases, range > 5pp): ablation has discriminating power. Paper's objective-level claim holds.
- Ambiguous (range 2-5pp, or non-monotone): run seed42 and seed7 replications. If all three are ambiguous, treat as flat (task-determined) by the stricter Round-2 lesson-6 pattern.

### Compute estimate

Single GPU-hour on H100. No training, just eval:
- Load vanilla SFT checkpoint.
- For each ProsQA eval example (≤ 200 for fast turnaround, full set if time allows): forward with the rank-k-ablated hidden, record accuracy.
- 5 k values × 200 examples × ~1s/example = ~15 min. With seed replication, < 1 GPU-hour total.

### Risks / failure modes to watch

- **Fake-Z construction is artificial.** Ablating the first 256 dims of a 768-dim hidden state is not the same operation as ablating matrix-CODI's Z. Mitigation: also try "ablate all 768 dims reshaped to 16×48" or "ablate each rank-k of the full 768 via low-rank SVD" to test the claim at multiple fake-Z constructions; if all flat, the finding is robust.
- **Seed variance.** Lesson 6 showed 3× rank spread across seeds at same accuracy. Run all three seeds (1337, 42, 7) before concluding.
- **Position effect.** Answer-position hidden state may not encode reasoning the way Z does. This is actually part of what's being tested — the ablation either finds signal or doesn't, and either outcome is informative.

### Deliverables for the paper

- `rank_projection_ablation_vanilla_sft.json` per seed.
- New paper subsection §5.X ("Control: does the ablation discriminate at all?") — 1-2 paragraphs + one figure (rank-k curve for vanilla SFT overlaid with matrix-CODI flat curve).
- Updated §7 Discussion — strengthen or recontextualize depending on outcome.

### Source of this experiment

Surfaced by the `/deploy-team` post-run-analysis run
`.team-runs/20260420-193321-post-run-analysis-2688/` on 2026-04-20. The
contextualizer agent ("Control A — fake-Z GPT-2 SFT rank-k eval") and
adversarial agent ("no null baseline for the rank-k ablation has ever
been run on ProsQA") independently converged on this gap. Already
acknowledged as caveat in paper §7, converting to evidence.

---

## PRIORITY 1: Matrix-CODI Rank Dynamics

**Status:** Fully specified, ready for an experimenter agent to implement and run.

### One-sentence hypothesis

In a continuous-reasoning language model where thoughts are represented as `d × d` matrices, the effective rank of a thought matrix at reasoning step `t` correlates with the number of distinct reasoning paths the model holds in superposition at that step.

### Why this experiment

This bridges three lines of research that have not been connected:

1. **Theoretical superposition story** (Reasoning by Superposition arXiv 2505.12514, CoT2 arXiv 2505.23648). Argues continuous thoughts hold parallel reasoning paths but only proves it via existence constructions and accuracy curves.
2. **Empirical rank dynamics** (this project, Round 2, May 2025). MultiProbeHead drives rank enrichment during iterative refinement (5.0 → 6.1).
3. **The 2026 rebuttal** (The Illusion of Superposition arXiv 2604.06374). Argues fine-tuned COCONUT reaches 96.6% without latent tokens; the superposition claim is contested.

The experiment can adjudicate this dispute. Nobody has measured rank as a structural correlate of reasoning capacity.

### The hypotheses (precisely)

**H1 (correlation):** During iterative reasoning on a task with known frontier size `|S_t|` at step `t`, the stable rank of the thought matrix `Z_t` correlates monotonically with `|S_t|`. Spearman correlation `ρ > 0.3` at `p < 0.01` over a held-out evaluation set.

**H2 (capacity bound):** Accuracy degrades sharply when `|S_t|` exceeds the matrix dimension `d`. A `d × d` matrix can correctly hold at most `d` linearly independent reasoning paths.

**H3 (causation):** Forced low-rank projection of `Z_t` to rank `k < |S_t|` causes accuracy degradation in proportion to `|S_t| - k`.

**Falsification conditions:**
- Rank is roughly constant across reasoning steps with no correlation to task difficulty → falsifies H1
- Rank correlates but a vector model at matched parameters achieves the same accuracy → falsifies the structural-advantage claim
- Forced low-rank projection has no effect on accuracy → falsifies H3
- Same backbone with different output heads produces different rank correlations → suggests rank is an artifact of the output head

### Experimental design

#### Base system: CODI

Fork CODI from https://github.com/zhenyi4/codi. CODI is the strongest published continuous-reasoning baseline:

- 43.7% on GSM8K (vs COCONUT 34.1%, vs explicit CoT 44.1%)
- Single-stage training (no curriculum)
- Joint teacher-student via shared weights
- Complete public codebase
- GPT-2 scale (124M params) — fits the budget

CODI's mechanism:

```
Teacher pass: [question][CoT tokens]"The answer is:"[answer]   → standard CE loss
Student pass: [question]<bot>z₁z₂z₃z₄z₅z₆<eot>"The answer is:"[answer] → CE on answer

Distillation: L1 distance between teacher and student hidden states at the ":" token,
across all layers, normalized by per-layer std.

Combined: L = α·L_teacher + β·L_student + γ·L_KD
         (α=β=1, γ=1 for GPT-2, γ=20 for Llama)
```

The 6 latent thoughts use COCONUT-style hidden-state feedback: the last hidden state from the previous step is fed back as the next input embedding.

#### Matrix-CODI modification

Insert a matrix bottleneck at each latent step:

```
Standard CODI:                          Matrix-CODI:
hidden state h ∈ ℝ^D                    hidden state h ∈ ℝ^D
        ↓                                       ↓
   [feed back as embedding]              W_up: ℝ^D → ℝ^{d × d}
                                                 ↓
                                         Z ∈ ℝ^{d × d}  (the matrix thought)
                                                 ↓
                                         [optional: 1 matrix-thinking iteration
                                          via (I+Δ)·Z·(I+Γ)]
                                                 ↓
                                         W_down: ℝ^{d × d} → ℝ^D
                                                 ↓
                                         [feed back as embedding]
```

**Critical design points:**

1. **Keep the CODI backbone (GPT-2) unchanged.** The matrix bottleneck inserts only at the latent step feedback, not throughout the network. This preserves CODI's training procedure and the L1 distillation loss as-is.

2. **The matrix `Z` is the observable.** SVD `Z` at every reasoning step on every held-out problem. This is what the experiment measures.

3. **Choose `d` carefully.** Recommend `d = 16`. Large enough that rank is meaningful (avoid `d = 4` where ranks 1-4 are all you have) but small enough that ranks below `d` are achievable.

4. **Use rank-1 outer-product token initialization for the FIRST experiment.** This gives the cleanest "rank growth" signal. The k-bigram embedding (rank-3 by construction) is a follow-up experiment, not a prerequisite. See "Embedding Designs" section below.

5. **L1 distillation stays as-is.** The teacher path doesn't have a matrix; the L1 loss is computed on the flat hidden states at the `":"` position. The matrix bottleneck shapes the student's hidden state internally.

#### Tasks (in priority order)

**Priority 1a — GSM8K with annotated reasoning step counts.** Standard CODI training task. Reasoning step count is available per problem. Hypothesis test: bin problems by reasoning step count (2, 3, 4, 5, 6+), measure stable rank of `Z` at each latent step, correlate.

**Priority 1b — MNNS (Minimum Non-Negative Subset Sum), CoT2-style.** Synthetic backup. Frontier at depth `k` has exact size `2^k`. Provides perfect ground-truth `|S_t|` to correlate against.

**Priority 1c — ProsQA / ProntoQA.** Graph reachability tasks. Frontier sizes can be computed from graph structure.

Start with GSM8K. Add MNNS as a secondary task if GSM8K signal is too noisy.

#### Three runs

**Run A — Vanilla vector CODI baseline.** Reproduces CODI's published GSM8K result (43.7%) within 1 point. Sanity check.

**Run B — Matrix-CODI (the experiment).** Same training procedure with the matrix bottleneck. `d = 16`, no extra losses, no extra hyperparameters.

**Run C — Rank-projection ablation.** At eval time on Run B's checkpoint, project `Z` to rank `k` for `k ∈ {1, 2, 4, 8, 16}` via truncated SVD before feeding through `W_down`. Measure accuracy at each `k`. Tests H3.

### Logging requirements

Per training step (every 50 steps):
- Standard: train loss, val loss, learning rate, grad norm
- Matrix-specific: stable rank of the average `Z` over the batch, at each latent step (6 values per step)

Per eval epoch:
- GSM8K accuracy (per the CODI evaluation protocol)
- Per-problem: stable rank of `Z` at each latent step, reasoning step count of the problem
- Save raw `Z` matrices for a sample of held-out problems for post-hoc analysis

Final analysis:
- Spearman correlation between `(reasoning_step_count, mean_stable_rank_across_latents)` over the held-out set
- Plot: stable rank vs latent step, grouped by reasoning depth bin
- Plot: accuracy vs forced rank projection (Run C)

### Smoke test plan (MANDATORY before training)

Per CLAUDE.md hard rules:

1. **Forward pass:** single batch through Run A and Run B. Check shapes at every layer. Verify the matrix bottleneck produces `(B, d, d)` tensors at each latent step.
2. **Backward pass:** compute loss, call `.backward()`. Verify all parameters get gradients. No NaN, no Inf.
3. **Eval batch size smoke test:** verify the eval batch fits in VRAM. Eval can OOM even when training fits.
4. **CODI baseline reproduction:** run Run A for 100 training steps. Verify the loss curve matches the public implementation.
5. **Rank measurement:** SVD a sample of `Z` matrices. Verify singular values are non-negative, stable rank is in `[1, d]`, the computation is numerically stable.

If any smoke test fails, fix before proceeding to training.

### Compute estimate

- Smoke tests: ~10 minutes
- Run A (vanilla CODI baseline reproduction): ~30 minutes
- Run B (matrix-CODI training): ~30 minutes
- Run C (rank projection ablation, eval-time only): ~15 minutes
- Post-hoc SVD analysis: ~10 minutes

**Total compute on 8×H100: ~2 hours.** Implementation time: 1-2 days.

### Pre-experiment checklist (per CLAUDE.md)

1. **Hypothesis stated in one sentence.** ✓
2. **FLOPs / memory / params on paper.** Matrix bottleneck adds two linear layers per latent step: `W_up: D → d²` and `W_down: d² → D`. At GPT-2 D=768 and d=16: `768 × 256 + 256 × 768 = 393,216` extra params per bottleneck × 1 bottleneck = 0.4M extra params. Negligible.
3. **Try to disprove in 5 minutes.** Strongest objection: rank correlates with reasoning depth because of how the output head regularizes, not because of structural superposition. Mitigation: Run C addresses causation. A future Run F could ablate the output head.
4. **Check literature first.** Done — see STATE.md "What the Field Has Shown Us" section. Confirmed: nobody has measured rank as a structural correlate of reasoning capacity.
5. **Design comparison before the experiment.** Done. Three runs (A, B, C) with explicit baselines.
6. **Define success criteria.** Done — see "Success / failure / falsification" below.
7. **Verify novelty.** Done — confirmed gap.

### Success / failure / falsification criteria

**Strong positive (publishable as-is):**
- Run A reproduces vanilla CODI accuracy within 1 point
- Run B accuracy is within 2 points of Run A
- H1 holds: Spearman ρ > 0.3 at p < 0.01
- H3 holds: forced low-rank projection degrades accuracy roughly in proportion to gap

→ Write up: "Matrix-valued continuous thoughts in CODI enable direct measurement of reasoning capacity via stable rank, providing the first structural correlate of the superposition hypothesis."

**Weak positive (still interesting):** H1 holds but not H3, or H3 holds but H1 is weaker than expected.

**Negative (also publishable):** H1 fails. Write up: "We tested the rank-superposition hypothesis directly using matrix-valued continuous thoughts and found no structural correlate. This supports the conclusions of Illusion of Superposition (arXiv 2604.06374) over Reasoning by Superposition (arXiv 2505.12514)."

**Inconclusive:** Run B fails to train, rank measurements are dominated by noise, or GSM8K reasoning depth annotations are too coarse. Diagnose and retry on MNNS.

### Output to save

```
matrix-thinking/scripts/run_matrix_codi.py             ← the script (commit to repo)
/toy_story_slam/results/matrix_codi_session/
  script.py                                             ← copy of the script
  train.log                                             ← full training log with timestamps
  results.json                                          ← all metrics, all eval points
  SUMMARY.txt                                           ← human-readable summary
  best_run_a_vanilla.pt                                 ← Run A checkpoint
  best_run_b_matrix.pt                                  ← Run B checkpoint
  rank_dynamics.json                                    ← per-problem stable rank trajectories
  rank_projection_ablation.json                         ← Run C results
  analysis_plots/
    rank_vs_reasoning_depth.png
    rank_vs_latent_step.png
    accuracy_vs_forced_rank.png
```

After completion, copy relevant small files to `experiment-runs/matrix-codi-session1/` in the repo. Large logs and checkpoints stay on `/toy_story_slam/`.

### Risks and mitigations

**Matrix bottleneck breaks CODI training.** The bottleneck might disrupt the L1 distillation by changing the geometry of the student's hidden state at the `:` token. CODI's static-teacher ablation showed accuracy collapses from 43.7% to 27.1% if teacher and student diverge.
*Mitigation:* The bottleneck only modifies latent positions, not the answer position. Verify in smoke testing that Run A and Run B losses look similar in early training.

**SVD is too noisy at small `d`.** With `d = 16` and finite batch size, individual singular values may be noisy.
*Mitigation:* Use stable rank (a ratio, more robust). Average over multiple problems and seeds. Use larger eval batches.

**GSM8K reasoning step counts are too coarse.** Step counts conflate "reasoning depth" with "arithmetic complexity."
*Mitigation:* Use MNNS as a secondary task where frontier size is exact.

**8×H100 DDP eval timeout.** NCCL default timeout is 10 minutes; eval can exceed this.
*Mitigation:* Set `dist.init_process_group("nccl", timeout=timedelta(minutes=30))`. Cap eval to 50 batches max.

---

## EXPERIMENTER AGENT PROMPT (for Priority 1)

The following is a self-contained prompt for an experimenter agent that will implement and run the matrix-CODI experiment on an 8×H100 pod.

### Brief

You are an ML experimenter agent. Your job is to implement and run the matrix-CODI rank dynamics experiment on an 8×H100 pod, following the specification in this document. You will fork the existing CODI codebase, add a matrix bottleneck at each latent reasoning step, train three runs (vanilla CODI baseline, matrix-CODI, rank-projection ablation), measure stable rank dynamics, and produce a final report.

### Read these files first, in this order, before writing any code

1. `CLAUDE.md` — workflow rules, hard rules from prior experiments, pre-experiment checklist (MANDATORY)
2. `STATE.md` — current project state and context for why this experiment matters, including the mathematical foundations section
3. `matrix-thinking/QUEUE.md` (this document) — the full experiment spec
4. `matrix-thinking/H100_SETUP.md` — pod environment and hard-won lessons
5. `references.md` — paper bibliography

### What to build

Fork CODI from https://github.com/zhenyi4/codi onto the 8×H100 pod at `/toy_story_slam/`. Modify the model to insert a matrix bottleneck at each of the 6 latent reasoning steps:

- At each latent step where CODI feeds the last hidden state back as the next input embedding, intercept the hidden state.
- Add a matrix bottleneck module: `W_up: ℝ^D → ℝ^{d × d}` (linear, where D=768 for GPT-2 and d=16) → optional 1 iteration of matrix-thinking via `(I+Δ)·Z·(I+Γ)` (use the existing `MultiplicativeThinkingLayer` from `matrix-thinking/src/matrix_thinker.py` if helpful) → `W_down: ℝ^{d × d} → ℝ^D` (linear).
- Feed the result back as the next input embedding, exactly as CODI does with the unmodified hidden state.
- The matrix `Z` at each step is the observable. Save it for SVD analysis.

**Do not modify CODI's L1 distillation loss, the teacher pass, the optimizer, or any of CODI's existing hyperparameters.** The goal is to test whether the matrix bottleneck preserves CODI's behavior while making rank dynamics observable. Minimal change.

### What to NOT do

- **Do not modify CODI's training procedure** beyond inserting the matrix bottleneck.
- **Do not add features not in the spec.** No fancy initializations, no new regularizers, no architecture search.
- **Do not use contextualized embeddings** (k-bigram, conv, etc.) for this experiment. The spec calls for the rank-1 outer-product byte/token embedding to keep the rank-growth attribution clean. Contextualized embeddings are Priority 2.
- **Do not skip the smoke tests.** Every prior experiment that skipped smoke testing wasted compute.
- **Do not silently fail on numerical issues.** If you see NaN or Inf gradients, stop training, diagnose, and report.
- **Do not invent new evaluation metrics.** Use the same GSM8K evaluation protocol as vanilla CODI, plus the rank dynamics measurements specified above.

### Reporting

When the experiment is complete, write a final report and save it to `/toy_story_slam/results/matrix_codi_session/SUMMARY.txt`. Contents:

1. Headline result: did H1 hold? Spearman correlation, p-value, sample size.
2. Sanity check: Run A vs published CODI on GSM8K. Did it reproduce?
3. No regression: Run B vs Run A on GSM8K. Within 2 points?
4. Causation test: Run C results. Plot accuracy vs forced rank.
5. Stable rank trajectories: per-step rank, grouped by reasoning depth bin.
6. Honest assessment: if any of H1/H2/H3 failed, say so.
7. Compute used: total H100-hours.
8. Next steps: based on the result, what should be the next experiment? Reference QUEUE.md.

After saving the report, also update `STATE.md` and `EXPERIMENT_LOG.md` in the local repo with a summary of the run, following the format of prior experiment entries.

### If you get stuck

- **Numerical issues:** check the multiplicative thinking layer's `(I+Δ)·M·(I+Γ)` parameterization. Use the `scale.clamp(0.01, 0.5)` safeguard in `matrix-thinking/src/matrix_thinker.py`.
- **CODI baseline doesn't reproduce:** check hyperparameters (γ=1 for GPT-2, α=β=1, 6 latent thoughts, distill_loss_type=l1, distill_loss_div_std=True).
- **Run B trains but accuracy collapses:** the matrix bottleneck might be too aggressive. Try setting the matrix-thinking iteration to identity (just up-project and down-project) as a sanity check.
- **Eval times out under DDP:** cap eval to 50 batches max. Run eval on rank 0 only.
- **You think the spec is wrong:** stop and ask the user. Do not silently deviate.

### Time budget

- Implementation + smoke tests: ~1 day
- Three training runs: ~2 hours of compute total
- Analysis and report: ~half a day
- **Total wall-clock from start to results: 2-3 days, of which compute is ~2 hours**

---

## POST-NEGATIVE QUEUE (added 2026-04-13 after Priority 1 returned Stack-decorative)

Priority 1 (matrix-CODI rank dynamics) returned a negative result across 5 axes (rank-k ablation, vanilla SFT control, distillation ablation, thinker ablation, probe on Z). See EXPERIMENT_LOG.md Rounds 1-5 and KILL_LIST.md for details. The following experiments are the surviving research directions toward the larger goals of the program (generalizability, abstract reasoning at inference, matrix-produced inference, matrix-embedded contextualization). They do not try to rescue the matrix-CODI bottleneck framing — that's dead. They aim at different mechanisms and different scales.

### Priority 1b: Round 6 — GPT-2 medium scale grid (in flight as of 2026-04-13)

**Status:** Running on 1×H100. Vanilla SFT + matrix CODI on GPT-2 medium (354M), single seed, ProsQA. Addresses the "did you try at scale?" reviewer question for the workshop submission.

**Expected completion:** ~3 hours from launch.

### Priority 1c: Round 5 — Sample efficiency curve (auto-launches after Round 4)

**Status:** Queued. Vanilla SFT vs matrix CODI at N ∈ {200, 500, 2000, 5000} on ProsQA. Addresses "did you try in low-data regimes?" reviewer question.

**Expected completion:** ~2-3 hours total after launch.

### Priority 1d: Reproduce Illusion of Superposition baseline (required before workshop)

**Hypothesis:** Our vanilla SFT achieves 81.77% on ProsQA; Rizvi-Martel et al. (arXiv 2604.06374) report 96.6% for fine-tuned COCONUT-no-latents on the same task. Close the gap or explain it convincingly.

**Design:** Read their hyperparameters from the paper, match them, run on 1×H100.

**Status:** Required for workshop submission. Blocking.

### Priority A1: Matrix-native from-scratch on a provably rank-K task (moonshot)

**Hypothesis:** A small (~10M param) fully matrix-native transformer (matrix Q/K/V with true matrix composition, no flatten in the forward pass, matrix LM head that does not reduce to a linear-in-Z map) trained from scratch on a synthetic task whose minimum solving rank is k will develop functional rank and show a non-flat rank-k ablation curve at k ≤ d.

**Task design (critical, hard):** Must construct a task where the ground truth requires tracking K independent quantities and the model cannot find a rank-1 shortcut. ProsQA failed here because the diamond DAG admits a rank-1 best-path solution. Candidate: multi-source path lookup — given k unrelated graphs and k start nodes embedded in one sequence, output the union of reachable terminals. Requires k orthogonal "current frontier" registers by construction.

**Escape mechanism:** Lesson 4(b) + 4(d) from KILL_LIST. No flatten anywhere, and the task has no rank-1 shortcut.

**Compute:** ~24 hours on 1×H100 (smoke test + k sweep × 6 + param-matched vector baseline at d²=256).

**Risk:** Synthetic task design is where MNNS died. Needs full attack-agent waterfall on the task spec before any code is written.

**Status:** Moonshot. Run only after workshop submission is out the door.

### Priority A2: Iterative refinement depth sweep (Round 2 of "abstract reasoning at inference")

**Hypothesis:** COCONUT-style continuous iteration on vanilla GPT-2 124M with iteration depth ∈ {6, 16, 64, 256} will show monotone accuracy gains on ProsQA up to a saturation point, providing evidence that inference-time compute via latent loops is the active mechanism — independent of representation structure.

**Design:** Fork the CODI repo's iteration loop, disable matrix bottleneck, sweep iteration count. Report accuracy-per-FLOP.

**Why this matters:** CODI (2502.21074) and COCONUT (2412.06769) both fix iteration count at 4-6. No published depth sweep exists beyond 8 iterations. Real gap in the literature.

**Compute:** 4 depths × 3 seeds × 0.5-3h (depth-dependent) = ~30h on 1×H100.

**Status:** Next candidate after workshop paper is out. Directly tests Goal 2 (abstract reasoning at inference).

### Priority A3: Scale study with a bigger backbone on the negative result

**Hypothesis:** Same matrix-CODI experiments on Pythia-410M or Llama-1B. If the decoration finding holds at larger scale, the TMLR submission becomes possible.

**Compute:** 6-10h per run × 6 conditions = ~50h on 1×H100.

**Status:** Required for TMLR submission, not workshop. Run during the workshop acceptance wait.

### Priority A4: Nuclear-norm reward with depth-stratified test (revived P5 with amendments)

**Hypothesis:** Adding `λ · ‖Z‖_* / ‖Z‖_F` (normalized to avoid scale explosion) as a reward in the matrix-CODI loss, sweeping λ, and measuring ProsQA accuracy stratified by reasoning depth, will reveal whether forced rank ever becomes functional.

**Amendments from killed P5:**
- Normalize nuclear norm by Frobenius norm (bounded, scale-invariant)
- Fix save_Z=True in training path
- fp32 SVD with ε-smoothing
- Multi-seed with Bonferroni correction over depth bins
- Pre-committed interpretation: flat accuracy across λ kills the rank-capacity hypothesis for good

**Compute:** 5 λ values × 3 seeds × ~1.5h = ~22h.

**Status:** Last-chance test before abandoning rank-as-capacity entirely.

---

## DEFERRED QUEUE

### Priority 2: K-bigram Embedding Ablation

**Hypothesis:** Starting matrix tokens at rank 3 (instead of rank 1) via a contextualized k-bigram embedding makes the rank-tracks-reasoning signal stronger.

**Design:** Same matrix-CODI architecture, but replace the rank-1 outer-product embedding with the k-bigram design (see "Embedding Designs" below). Compare to Run B from Priority 1.

**Status:** Run only after Priority 1 produces clean results. Rank-1 first gives the cleanest "rank growth from nothing" signal.

### Priority 3: Byte-Level JEPA with LeJEPA SIGReg

**Hypothesis:** A 50M-param byte-level JEPA model with SIGReg regularization can learn non-collapsed isotropic byte representations on raw English text.

**Design:** Standard byte-level transformer (12 layers, d=512, conv byte patcher), JEPA two-view objective on consecutive byte spans, LeJEPA SIGReg loss instead of EMA + stop-grad.

**Compute:** 4-8 H100 hours per run.

**Go/no-go:** Effective rank ≥ 115/128 of full embedding dimension; SIGReg loss within 2× the N(0,I) reference; on par with EMA+stop-grad baseline on a downstream linear probe.

**Status:** Deferred until Thread A (Priority 1) produces signal.

### Priority 4: Matrix-CoT2 on MNNS (synthetic backup)

**Hypothesis:** Same as Priority 1 (rank tracks reasoning) on a synthetic task with exact ground-truth frontier sizes (`2^k` partial sums at depth `k`), using CoT2's CSFT + GRPO training procedure instead of CODI's distillation.

**Design:** Replace CoT2's vector continuous tokens with d×d matrices, train on Minimum Non-Negative Subset Sum, measure stable rank vs ground-truth `|frontier_t|`.

**Status:** Backup for Priority 1 if GSM8K signal is too noisy.

### Priority 5: Param-Matched Three-Way Embedding Ablation

Three models, all ~2.5M total params, same data, same training:
- Model A: Standard embedding (256-dim vector lookup)
- Model B: ALBERT-style bottleneck (byte → 16-dim → Linear(16, 256))
- Model C: Outer-product embedding (byte → u(16) ⊗ v(16) → flatten)

**CRITICAL: Fix initialization.** u and v std = sqrt(0.02) ≈ 0.1414 so outer-product entries have std 0.02.

**Status:** Adjudicates "is the embedding advantage from structure or compression?" Was previous top priority before Priority 1 displaced it.

### Priority 6: Cross-Domain Generalization at Scale

**Hypothesis:** Mixed-byte-data training (text + code + images + audio) shows different transfer patterns for matrix vs vector representations.

**Status:** Heavily deferred. Was attacked by 6 fatal arguments in the original cross-domain hypothesis. Worth revisiting only with a sharper formulation and only at >10M params.

### Priority 7: HELM-Style Fully Matrix-Native Architecture

**Hypothesis:** A fully matrix-native architecture (matrix attention, matrix FFN, matrix normalization, no flatten anywhere) beats vector transformers at matched compute, replicating HELM's hyperbolic-everywhere result.

**Status:** High cost. Only justified if Priority 1 produces strong signal that matrix structure matters in a measurable way.

---

## Embedding Designs (for Priority 2 and beyond)

The current outer-product byte embedding `byte → u_b ⊗ v_b` produces a rank-1 matrix with only 32 degrees of freedom in 256 entries. A research investigation (April 9, 2026) confirmed that no published work builds contextualized matrix-valued token embeddings. The four designs below are ordered by complexity.

### Design 1 (Recommended for Priority 2): K-bigram Outer-Product

**Cheapest, simplest, drops in.** Replace `byte_b → u_b ⊗ v_b` with:

```
M_i = u_{xi} ⊗ v_{xi}                    (rank-1 identity term)
    + α_L · u_{x_{i-1}} ⊗ w^L_{xi}        (left-bigram outer product)
    + α_R · w^R_{xi} ⊗ v_{x_{i+1}}        (right-bigram outer product)
```

Where `u, v ∈ ℝ^{256 × 16}` (existing tables), `w^L, w^R ∈ ℝ^{256 × 16}` (new tables), and `α_L, α_R` are learned scalars (init 0.5).

**Properties:**
- Rank exactly 3 by construction
- Adds 2 × 256 × 16 = 8,192 params + 2 scalars
- No pretraining required
- Drops in to existing matrix-token architecture
- Causal-safe variant: drop the right term, use only `{-1, -2}` offsets

**When to use:** Priority 2 follow-up after Priority 1 produces a baseline.

### Design 2: Conv Encoder Embedding

Take a window of 16 bytes around position `t`. Run a small 1D convolutional encoder over the window. Reshape the output to a 16×16 matrix.

```python
window_bytes = bytes[t-7 : t+8]    # 16 bytes
features = Conv1D(window_bytes)    # learned features over the window
M_t = reshape(features, 16, 16)
```

**Properties:**
- Rank emerges from data
- Conv learns context features automatically
- More expressive than k-bigram, more parameters
- Closest analog to BLT's local encoder

### Design 3: Local Attention Contextualization

Run a small attention layer over a local window. Output two contextualized vectors per byte (one for u, one for v), take outer product:

```python
u_t = Attention_u(bytes[t-K : t+K])
v_t = Attention_v(bytes[t-K : t+K])
M_t = u_t ⊗ v_t
```

For higher rank, K parallel heads → sum K outer products:

```python
M_t = Σᵢ₌₁ᴷ uᵢ_t ⊗ vᵢ_t
```

**Properties:**
- Rank-1 single-head version: same shape as current embedding but contextualized
- Multi-head: rank up to K
- BERT-style contextualization

### Design 4: Pairwise Interaction Matrix (the most matrix-native)

Take a window of 16 bytes. Build a 16×16 matrix where entry `[i, j]` encodes a learned interaction between byte `i` and byte `j` in the window:

```python
W ∈ ℝ^{256 × 16}                              # learned per-byte feature map
M[i, j] = ⟨W[byte_i], W[byte_j]⟩               # inner product of byte features
```

**Properties:**
- Matrix entries are intrinsically about pairwise relationships within the window
- Rank reflects the number of distinct interaction patterns
- Most "matrix-native" — entries genuinely depend on relationships, not identities
- Closest to attention's similarity matrix, but used as the embedding

**When to use:** Most ambitious. Use to test whether matrix structure can encode pairwise relationships from the start.

### Important note for Priority 1

Do NOT use contextualized embeddings for the matrix-CODI rank dynamics experiment. Higher input rank creates an attribution confound: rank measured during reasoning could be inherited from the embedding's contextual structure rather than constructed by the matrix-token operations. Rank-1 gives the cleanest "rank growth" signal. Priority 2 (k-bigram) is the controlled A/B that tests whether contextualized starting embeddings change the rank dynamics picture.

---

## Pre-Experiment Checklist (MANDATORY for any experiment)

Per CLAUDE.md hard rules:

1. **State the hypothesis in one sentence.** If you can't, don't run it.
2. **Compute FLOPs, memory, and param count on paper.** 10 minutes. No exceptions.
3. **Try to disprove it in 5 minutes.** "Could a vector do this?" "Is this known to fail?"
4. **Check the literature first.** Send a research agent BEFORE building.
5. **Design the comparison before the experiment.** What's the baseline? Are params matched?
6. **Define success criteria.** What metric improvement justifies the compute cost?
7. **Verify the claim is novel.** Don't claim uniqueness without checking if vectors can do the same thing.
