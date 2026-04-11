# Why PHM Layers Converge to Near-Nilpotent Structure

Research from agent on 2026-03-24.

## Bottom Line

**No one has ever demonstrated PHM layers learning meaningful algebraic structure on real tasks.** Our finding is the FIRST empirical report of what PHM actually learns. The original paper only showed Hamilton product recovery on synthetic data.

## Root Causes

1. **No gradient signal for algebra** — next-byte prediction rewards accuracy, not algebraic structure
2. **Optimization favors low-rank** — PHM is mathematically equivalent to structured low-rank factorization (proven by Santiago's blog). Optimizer finds lowest-rank solution.
3. **Our initialization starts in nilpotent basin** — scaled-diagonal (A_i[i,i] = 0.25) is already near-nilpotent. No gradient pushes away from this.
4. **S matrices absorb all work** — 160×160 = 25,600 params per S vs 4×4 = 16 params per A. S does everything, A drifts.
5. **Adam favors flat directions** — small gradients on A get large LR adjustments, pushing toward nearest minimum (nilpotent).

## What CAN Make Algebra Work

### Fixed Algebras (proven to work)
- **CliffordNet (2026)**: Fixed geometric product, 8x parameter efficiency on CIFAR-100, FFN layers become optional
- **GATr (2023)**: Fixed Cl(3,0,1), excellent for physics simulation
- **Quaternion Networks (2019)**: Fixed Hamilton product, 75% param reduction

These prove genuine algebraic structure CAN help — but it must be IMPOSED, not learned.

### Untried Approaches (novel research directions)
1. **Quaternion init + soft regularizer**: A starts as quaternion, 0.01 * ||A - A_quat|| prevents drift
2. **Closure loss**: ||A_i @ A_j - proj(A_i@A_j, span(A))|| forces products to stay in basis
3. **Division algebra loss**: penalize det(A_i) → 0 to prevent nilpotent collapse
4. **Low-rank control**: Compare PHM to plain low-rank W = U@V at same param count

### Key Insight
The benefit of quaternion structure (when fixed) comes from **inter-component coupling**: the Hamilton product forces each weight component to interact with all 4 input components. This creates richer inter-channel dependencies than diagonal structure. But the optimizer finds it easier to learn these dependencies through the S matrices than through the A matrices.

## For Our Experiment

The PHM "algebra discovery" hypothesis is NOT supported. But we have two paths forward:

**Path A: Fix the algebra, vary the rest.** Use fixed quaternion A matrices. The S matrices still learn freely. This gives us genuine inter-dimension coupling without asking the optimizer to discover it. Our ablation (condition 1) is testing this.

**Path B: Add algebraic regularization.** Keep A learnable but add a closure loss + determinant regularization to push it toward a genuine algebra. This is unexplored territory — could be a paper contribution itself.
