# Abstract

<!-- Title: [WORKING TITLE — PI] "The Matrix State Is Real: Minimal-Rank
Recruitment, a Recall Capability Baselines Lack, and a Scale-Worsening
Write-Geometry Pathology in Matrix-State Sequence Models" -->
<!-- Authors: [AUTHORS — PI decision pending]; corresponding
samlarson@pebbleml.com (named build only) -->

Fast-weight and linear-attention models maintain a matrix-valued state,
written by an outer-product update and evaluated mainly as an
efficiency substitute for attention. We ask instead what the matrix
state stores, at what dimensionality, and at what cost, reporting three
pre-registered findings. First, a rank law: in a matrix-state encoder
family trained on five permutation-group word problems (minimal
faithful representation dimension 2 to 5), recruited effective rank
tracks $d_{\min}$ (Spearman $\rho=0.9747$ <!-- evidence: R1 -->, the
tie-capped maximum), a designed $S_4$-versus-$A_5$ test shows rank
follows dimension, not solvability <!-- evidence: R2 -->, and
forcing rank one below $d_{\min}$ zeroes recovery while $d_{\min}$
restores it <!-- evidence: R3 -->. Second, a
capability separation: a two-layer delta-rule model performs episodic
recall at accuracy 0.9995 <!-- evidence: R4 --> while a
parameter-matched vector-state ablation reads chance, the comparison
carrying the pre-registered verdict (a compute-matched transformer
also reads chance, disclosed as a degenerate baseline)
<!-- evidence: R4 -->; the capability resides causally in the first
layer's state, is linearly legible only after downstream nonlinear
processing <!-- evidence: R5 -->, and holds at 0.998 or higher to 1798
tokens on a fixed 32,768-byte state while that transformer,
cache-capped, reads chance at every cap <!-- evidence: R10 -->. Third, the
same write mechanism drives a population-geometry pathology worsening
monotonically from 14M to 1.31B parameters <!-- evidence: R6 -->, not
explained by qk-normalization <!-- evidence: R7 --> and not removed by
a frozen-key-bias mitigation whose loss neutrality transfers to scale
but whose geometric benefit does not <!-- evidence: R8 -->. Capability
and pathology are two faces of one storage mechanism.
