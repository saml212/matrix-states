# Why byte-agnostic/ was archived

**Archived:** 2026-04-09

This was the project's original Phase 1-2 work: a single architecture that processes any domain (text, images, audio) from raw bytes, using **Parameterized Hypercomplex Multiplication (PHM)** layers for richer-than-vector representations and **learned byte segmentation** instead of fixed tokenizers.

## Status when archived

- **Best result:** 2.34 BPC on multi-modal data with a 6.2M param fixed-stride model
- **Emergent finding:** Multi-modal training showed domain differentiation in segment lengths without labels (text σ=3.3, image σ=4.5)
- **Key negative result:** Learned segmentation lost to fixed-stride at every scale tested
- **Key negative result:** PHM "learned algebra" converged to nilpotent rather than learning quaternion-like or Clifford-like structure

## Why it was archived (not just paused)

Two reasons:

1. **The PHM "learned algebra" hypothesis is empirically dead.** The deep analysis at Run 6 showed all 36 PHM layers converge to dual-number-like (near-nilpotent) algebra. Squared traces ≈ 0, near-zero determinants, low-rank Kronecker factors. The optimizer treats PHM as a structured low-rank factorization, not as a learned algebra in the mathematical sense. CliffordNet (2026) confirmed this from the other side: structured algebra works when **fixed**, not when learned.

2. **The matrix-thinking thread (the project's pivot) is a stronger version of the same intuition.** The matrix-thinking work asks the same question — "do richer-than-vector representations help?" — with a clearer observable (rank) and a sharper hypothesis. The byte-agnostic work conflated two questions (segmentation + algebraic structure); matrix-thinking isolates one of them.

## What was preserved from this work

- **Byte-level processing** is now Thread 1 of the narrowed thesis (see STATE.md)
- **The cross-domain generalization question** survives, but the framing was attacked and rewritten (see `research/hypothesis-attack-april2026.md`)
- **The fixed-stride segmentation result** stands as a valid empirical finding for byte-level models at small scale
- **The PHM-converges-to-nilpotent finding** is a publishable negative result (see `research/phm-nilpotent-convergence.md`)

## Contents

```
src/
  model.py        — BytePHMTransformer (segmenter + PHM transformer)
  segmenter.py    — Learned segmenter with top-K boundary selection
  data.py         — Byte-level data pipeline (text, images, audio)
  train.py        — Training loop
  ablations.py    — Ablation models (fixed-stride, standard linear, no-seg)
  analyze.py      — Boundary visualization, PHM algebra analysis
configs/
  phase1.yaml, phase1_scaled.yaml, phase2.yaml
research/
  boundary-collapse-prevention.md
  cross-domain-transfer.md
  imposed-vs-learned-structure.md
results/
  (empty — actual results live in /Volumes/1TB_SSD/learned-representations/results/)
```

## Read this folder for

- The complete PHM + learned segmentation experimental record (Runs 1-7)
- The boundary-collapse-prevention research
- The cross-domain transfer measurement protocols
- The imposed-vs-learned-structure analysis

Anything else: read STATE.md.
