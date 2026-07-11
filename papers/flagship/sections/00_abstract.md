# Abstract

<!-- Title: [WORKING TITLE — PI] "The Matrix State Is Real: Minimal-Rank
Recruitment, a Recall Capability Baselines Lack, and a Scale-Worsening
Write-Geometry Pathology in Matrix-State Sequence Models" -->
<!-- Authors: [AUTHORS — PI decision pending]; corresponding
samlarson@pebbleml.com (named build only) -->

Fast-weight and linear-attention models maintain a matrix-valued state,
written by an outer-product update and evaluated mainly as an
efficiency substitute for attention. We ask instead what the matrix
state stores, at what dimensionality, and what the storage costs at
scale: three pre-registered findings. First, a rank law: in a matrix-state encoder
family trained on five permutation-group word problems (minimal
faithful representation dimension 2 to 5), recruited effective rank
tracks $d_{\min}$ (Spearman $\rho=0.9747$ <!-- evidence: R1 -->, the
tie-capped maximum), a designed $S_4$-versus-$A_5$ test shows rank
follows dimension, not solvability <!-- evidence: R2 -->, and
forcing rank one below $d_{\min}$ zeroes recovery while $d_{\min}$
restores it <!-- evidence: R3 -->. Second, a
capability separation: a two-layer delta-rule model performs episodic
recall at accuracy 0.9995 <!-- evidence: R4 --> against a
parameter-matched vector-state ablation at chance, the pre-registered
decisive comparison; a compute-matched transformer also reads chance
and is disclosed as a degenerate baseline <!-- evidence: R4 -->. The
capability resides causally in the first layer's state, is linearly
legible only after downstream nonlinear processing
<!-- evidence: R5 -->, and holds at 0.998 or higher to 1798 tokens on
a fixed 32,768-byte state while that transformer, cache-capped, reads
chance everywhere <!-- evidence: R10 -->. Third, the same write
mechanism drives a population-geometry pathology that worsens
monotonically from 14M to 1.31B parameters <!-- evidence: R6 -->.
Removing qk-normalization leaves it unchanged
<!-- evidence: R7 -->, and a frozen-key-bias mitigation fails to
remove it: the loss neutrality transfers to scale; the geometric
benefit does not <!-- evidence: R8 -->.
