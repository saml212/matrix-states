# Chapter 2: Matrix-Native From Scratch on a Synthetic Rank-K Task

## Purpose

Decide whether the matrix-thinking direction is alive. The ICML MI workshop
paper kills one variant (matrix-CODI with flatten readout) via a specific
failure mode (rank-blind gradient through the bolt-on architecture). Chapter 2
tests the broader hypothesis — that a matrix-native end-to-end architecture
on a task that provably rewards rank-K structure will learn that structure.

Two outcomes, both decisive:
- Rank-k curve bends: matrix-thinking survives, proceed to Chapter 3 (real
  data, byte-level input).
- Rank-k curve stays flat: matrix-thinking is dead, pivot direction.

## Falsifiable prediction

A small transformer with d×d matrix tokens throughout, trained end-to-end on
a task whose optimal solution provably has rank K*, will produce trained
matrix representations whose effective rank ≈ K* and whose accuracy degrades
monotonically under rank-k truncation when k < K*.

## Task candidates

Pick whichever survives a short attack pass. All three are synthetic,
controllable, cheap.

### Task A — K parallel entity tracking (leading candidate)

Generate sequences describing K independent entity-state streams that
interleave. At each step, one of K entities takes an action that updates
its local state. Test: query the current state of entity i.

- K is tunable (2, 4, 8, 16). Optimal solution is a d×d matrix whose
  top-K singular directions encode each entity's state independently.
- Rank-1 solutions should fail at K > 1 (they can only track one stream).
- Rank-K solutions should succeed.
- Vanilla SFT with matched params should also have a clear rank upper
  bound (d-dim vector can track at most d parallel streams without
  collapse — so K=16 at d=16 is the breaking point for vectors).

### Task B — K-way BFS (Reasoning by Superposition inspired)

DAG reachability query where the optimal latent is a superposition of K
frontier nodes. Already studied by Lin/Zhu et al. for vectors (arXiv
2505.12514). Running it on matrix tokens gives a direct comparison with
their theoretical result. Harder to set up than A.

### Task C — Composition of K linear transformations

Given sequence "x, M_1, M_2, ..., M_K", predict M_K(...M_2(M_1(x))...).
Optimal latent at step t: M_t ∘ M_{t-1} ∘ ... ∘ M_1. A d×d matrix is
the natural representation. Rank of the composition is the natural
measure of information bottleneck.

## Architecture

Keep it minimal. No CODI, no distillation, no pretraining.

- Input: sequence of symbols. Each symbol has a learned d×d matrix embedding.
- Layers: N matrix-transformer blocks. Each block:
  - Matrix attention: Q, K, V are d×d matrices. Attention score between
    tokens i and j: `Frobenius(Q_i, K_j) / sqrt(d²)`. Output: weighted sum
    of V matrices.
  - Matrix MLP: `Z → ReLU(Z W_1) W_2` where W_1, W_2 are d×d.
  - Residual + LayerNorm (applied across d²).
- Readout at final position: MultiProbeHead (bilinear probes,
  `matrix-thinking/src/matrix_output_heads.py`) to vocab logits.
- NO FLATTENING anywhere. Enforced by construction.

Start with N=4 layers, d=16. Total params ~10M at vocab=256 (byte-level).

## Training

- Pure next-token prediction. No distillation. No auxiliary losses.
- Adam, lr=3e-4, cosine schedule, batch=128.
- 1M–10M training sequences. Cheap.
- Evaluate:
  - Task accuracy (standard)
  - Effective rank of Z at the last reasoning position as a function of K
  - Rank-k truncation accuracy curve (same protocol as the main paper)

## Gauntlet before building

Run the waterfall (brainstorm → research → attack → validation) on this
design before writing any code. Specific attack vectors to address:

1. **"A vector model of equivalent param count will solve this too."**
   Attack: For K=16 at d=16 vectors, there are 256 vector dims available,
   which is enough to track 16 parallel streams by allocating 16 dims each.
   Defense: Go to K=32 at d=16 — vectors have 256 dims but can't track 32
   independent streams without collapse, while matrices can still represent
   rank-32 ≈ full (d²=256 = 16²). Or go to K=4 at d=8 where matrix has
   d²=64 dims vs vector 64 — same param count but rank-4 matrix
   representation is topologically distinct.

2. **"The task is memorizable without rank structure."**
   Attack: Generate infinite task instances on the fly. Each batch is fresh.
   Defense: Run eval on held-out seeds, verify accuracy doesn't collapse.

3. **"Matrix attention is just block-diagonal vector attention."**
   Attack: Formally, a d×d matrix attending to another via Frobenius
   inner product is equivalent to flatten-then-dot-product between two
   d²-dim vectors. The "matrix structure" only matters if later operations
   preserve it.
   Defense: The matrix MLP (`ReLU(Z W_1) W_2`) does preserve structure —
   it's multiplicative mixing along both row and column axes of Z. A vector
   MLP of the same param budget can only do one such axis. This is where
   the architecture earns its expressive power.

4. **"Rank-K solution exists but is not the only solution."**
   Attack: The model may learn a rank-full solution that routes the task
   through brute-force memorization of the d² feature space.
   Defense: Include a rank-regularization ablation — penalize nuclear norm
   of Z during training. If rank-regularized version works and
   rank-full version also works, we've shown rank-K is SUFFICIENT. If
   rank-full works but rank-regularized fails, we've shown rank-K is
   NECESSARY. Either is publishable.

## Timeline

Post-workshop paper. Not before May 8.

Rough plan:
- Week 1 (May 9-15): Task generator + unit tests. Architecture stub. CPU
  smoke test on tiny config.
- Week 2 (May 16-22): GPU training on full config. Rank-k ablation.
- Week 3 (May 23-29): Attack-agent pass on results. If it survives,
  write it up as a second finding / technical report.

Compute budget: Chapter 2 should be cheap. ~5M-param models, synthetic
data, fast iteration. Single GPU ~24h per configuration. Total budget
50-100 GPU-hours including ablations.

## Decision criteria

After Chapter 2:

- **Survive (rank-k curve bends for task A/B/C, spearman r significant):**
  Proceed to Chapter 3 (byte-level, real data, modality switching per
  the architecture brainstorm prompt). Matrix-thinking is alive.
- **Fail (rank-k curve flat across all tasks):** Pivot. Matrix-thinking
  was a wrong bet. Write a short negative-result note for the repo
  (not publication) and pick a different direction.
- **Mixed (some tasks yes, others no):** Investigate task-structure
  dependence. This is the most interesting outcome — identifies what
  class of problems matrix structure helps with.

## Files that will be created when we start

- `matrix-thinking/chapter2/synthetic_tasks.py` — task generators
- `matrix-thinking/chapter2/matrix_transformer.py` — end-to-end matrix model
- `matrix-thinking/chapter2/train.py` — training loop
- `matrix-thinking/chapter2/rank_ablation.py` — evaluation
- `matrix-thinking/chapter2/CHAPTER_2_RESULTS.md` — results log
