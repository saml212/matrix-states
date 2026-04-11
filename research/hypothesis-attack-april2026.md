# Hypothesis Attack: Cross-Domain Transfer via Matrix Structure

## The Hypothesis
"Matrix representations trained on mixed-domain bytes will show less negative
cross-domain transfer than flat vector representations, because the matrix
row/column structure can encode domain-specific geometry."

## Verdict: Hypothesis as stated is WRONG. Redesign needed.

## Six Fatal Attacks

### 1. Reshape Equivalence
256-dim vector = 16×16 matrix in degrees of freedom. The model can reshape internally.
Our own data: flat model beats matrix at T=8 with 2.2× params.

### 2. Distribution Mismatch, Not Geometry
bGPT's negative transfer is from different byte STATISTICS (ASCII vs uniform vs Gaussian),
not different geometries. Matrices don't change byte statistics.

### 3. Fatal Parameter Confound
212K matrix vs 3.5M flat. "Better transfer" = "too small to memorize, forced to share."
Not evidence of structural advantage.

### 4. Model Too Small
At BPB 3.56, barely past unigram statistics. Can't measure transfer when the model
hasn't learned single-domain patterns.

### 5. Reviewer Kills It
"Less negative transfer from a worse model is evidence of insufficient capacity,
not better representation."

### 6. 16×16 Doesn't Match Any Domain
Text: hierarchical. Images: width-specific adjacency (96 bytes for CIFAR). Audio: 1D.
The matrix factorization is arbitrary, not aligned to any domain.

## What Survives
Weaker hypothesis: "Does outer-product embedding lead to different multi-domain
LEARNING DYNAMICS, measurable through rank dynamics?"

## Prerequisites Before ANY Multi-Domain Experiment
1. Param-matched single-domain ablation (never run)
2. Scale byte model to <2.0 BPB on one domain (current: 3.56)
3. Flat baseline at matched params on same data
