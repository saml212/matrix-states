# Byte-Agnostic Processing

A single architecture that processes any domain (text, images, audio) from raw
bytes without domain-specific tokenizers or encoders.

## Status: Partially Validated. On Hold.

### What works:
- Fixed-stride segmentation (4-byte chunks) improves BPC by 0.85 over no segmentation
- Multi-modal training (text + images) shows emergent domain differentiation
  without labels: text segments σ=3.3, image segments σ=4.5
- Best result: 2.34 BPC on multi-modal data with 6.2M param fixed-stride model

### What doesn't work (at this scale):
- Learned segmentation loses to fixed stride
- PHM learned algebra collapses to nilpotent (archived)

### Key findings:
- Segmentation helps, but fixed stride is enough at 2-13M params
- Domain differentiation is real but doesn't improve BPC
- These may change at larger scale (BLT works at 8B params)

## Directory Structure
```
src/
  model.py        — BytePHMTransformer (segmenter + PHM transformer)
  segmenter.py    — Learned segmenter with top-K boundary selection
  data.py         — Byte-level data pipeline (text, images, audio)
  train.py        — Training loop
  ablations.py    — Ablation models (fixed-stride, standard linear, no-seg)
  analyze.py      — Boundary visualization, PHM algebra analysis
configs/
  phase1.yaml     — 10M params, text only
  phase1_scaled.yaml — 10M params, 100MB data
  phase2.yaml     — Multi-modal (text + images)
research/
  boundary-collapse-prevention.md
  cross-domain-transfer.md
  imposed-vs-learned-structure.md
```

## Data (on SSD)
- `/Volumes/1TB_SSD/learned-representations/data/text.bin` — 100MB WikiText-103 UTF-8
- `/Volumes/1TB_SSD/learned-representations/data/images.bin` — 154MB CIFAR-10 raw pixels
