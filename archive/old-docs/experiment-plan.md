# Experiment Plan

## Overview

Three phases, increasing in scale. Each phase has a clear go/no-go criterion before proceeding to the next.

## Phase 1: Proof of Life (M4 Mac Mini, 32GB, 1-2 days)

### Goal
Get the combined architecture training. Verify loss decreases. See if boundaries emerge. Text only.

### Setup
- **Model:** ~10M parameters
  - 4 layers, d=256, 4 heads
  - PHM layers (n=4) replacing all nn.Linear
  - Learned segmenter: 1-layer transformer producing boundary probabilities via Gumbel-softmax
  - Tied PHM embeddings
- **Data:** 10MB of FineWeb text, raw UTF-8 bytes
- **Objective:** Next-byte prediction (cross-entropy)
- **Hardware:** M4 Mac Mini, MPS backend
- **Training:** ~12 hours, batch size 16, sequence length 1024 bytes

### Implementation Steps
1. Install PyTorch with MPS support
2. Implement PHM layer (reference: ICLR 2021 code, ~100 lines)
3. Implement byte segmenter with Gumbel-softmax boundaries (reference: MAGNET, MrT5)
4. Build the transformer with PHM layers and segmenter
5. Data pipeline: load text as raw bytes, no tokenizer
6. Training loop with next-byte prediction loss + boundary regularization loss
7. Log: loss curve, boundary positions over training, learned Kronecker factors

### Go/No-Go
- **Go:** Loss decreases consistently. Boundary positions show structure (not uniform random). Training is stable.
- **No-go:** Loss doesn't decrease or is unstable. PHM layers cause numerical issues. Segmenter collapses to trivial boundaries (all-merge or all-split).

### Deliverable
Training curves. Visualization of learned boundaries on sample text. Comparison of PHM vs standard linear at matched parameter count.

---

## Phase 2: Multi-Modal Signal (Mac Studio, 256GB, 3-5 days)

### Goal
Train on text + images + audio simultaneously. See if the segmenter differentiates by domain. See if cross-domain training helps.

### Setup
- **Model:** ~50M parameters
  - 8 layers, d=384, 6 heads
  - PHM layers (n=4)
  - Learned segmenter (2-layer, larger)
- **Data:** (300MB total)
  - Text: 100MB FineWeb (UTF-8 bytes)
  - Images: 100MB CIFAR-10 (raw pixel bytes: each image = 3072 bytes for 32x32x3)
  - Audio: 100MB LibriSpeech (raw 16-bit PCM, 2 bytes per sample)
- **Objective:** Next-byte prediction on all three streams, interleaved
- **Training:** 2-3 days, batch size 32, sequence length 2048 bytes

### Implementation Steps
1. Data pipeline: three data loaders, randomly interleaved in each batch
   - Each sequence is pure bytes, no header or metadata indicating domain
   - Prepend nothing. The model sees [byte, byte, byte, ...] and has to figure it out
2. Train multi-modal model (3 days)
3. Train three single-domain control models (1 day each, can run in parallel if Mac Studio supports it)
4. Analysis (1 day)

### Measurements
1. **Boundary analysis by domain:**
   - For each domain, pass 1000 sequences through the trained model
   - Record boundary probabilities at each byte position
   - Compute mean segment length and segment length variance per domain
   - Visualize: do text sequences get segmented at word/morpheme boundaries? Do image sequences get segmented at row/edge boundaries? Do audio sequences get segmented at zero-crossings or amplitude changes?

2. **PHM algebra analysis:**
   - Extract the learned Kronecker factors from each layer
   - Compute the "effective multiplication table" (reconstruct the full hypercomplex multiplication from the learned factors)
   - Compare across layers: do early and late layers learn different algebras?
   - Compare to known algebras: does the learned structure resemble quaternions, split-quaternions, dual numbers, or something novel?

3. **Cross-domain transfer:**
   - Compare multi-modal model's per-domain loss to single-domain control models
   - If multi-modal text loss < single-domain text loss at matched compute: positive transfer
   - If multi-modal text loss > single-domain text loss: interference
   - Break down by domain pair: does image pretraining help text? Does audio help images?

### Go/No-Go
- **Go (strong):** Segmenter produces visibly different boundaries for different domains. Cross-domain transfer is positive for at least one pair.
- **Go (weak):** Segmenter differentiates domains even if transfer is negative. The differentiation alone is a finding.
- **No-go:** Segmenter produces identical boundaries for all domains. No evidence of domain-aware processing.

### Deliverable
Boundary visualizations across domains. Transfer matrix (which domains help which). PHM algebra analysis. Comparison table: multi-modal vs single-domain per-domain loss.

---

## Phase 3: Scale and Publish (1x H100 cloud, 1-2 days)

### Goal
Reproduce Phase 2 findings at larger scale. Get numbers suitable for publication.

### Setup
- **Model:** ~200M parameters
  - 12 layers, d=512, 8 heads
  - PHM layers (n=4)
  - Learned segmenter (4-layer)
- **Data:** (3GB total)
  - Text: 1GB FineWeb
  - Images: 1GB ImageNet (raw bytes)
  - Audio: 1GB LibriSpeech
- **Hardware:** 1x H100 80GB ($2-3/hour, ~$50-100 total)
- **Training:** 1-2 days

### Additional Measurements
1. **Scaling behavior:** Does the multi-modal advantage grow or shrink with scale?
2. **Comparison to baselines:**
   - PHM transformer vs standard transformer (matched params, single domain)
   - Learned segmentation vs BPE vs fixed byte chunks (matched architecture)
   - Full system vs ablations (no learned segmentation, or no PHM, or both removed)
3. **Downstream evaluation:** Can the multi-modal model do zero-shot classification? (Present an image as bytes, see if the model's byte predictions reflect the image class)

### Deliverable
Full paper draft with tables, figures, ablations.

---

## Code Structure

```
learned-representations/
  README.md                    # project overview
  references.md                # all papers and links
  conceptual-framework.md      # the argument
  experiment-plan.md           # this file
  src/
    model.py                   # transformer + PHM layers + segmenter
    phm.py                     # PHM layer implementation
    segmenter.py               # learned boundary predictor
    data.py                    # multi-domain byte data pipeline
    train.py                   # training loop
    analyze.py                 # boundary visualization, algebra analysis
  configs/
    phase1.yaml                # 10M model, text only
    phase2.yaml                # 50M model, multi-modal
    phase3.yaml                # 200M model, scaled
  results/
    phase1/                    # training curves, boundary plots
    phase2/                    # cross-domain analysis
    phase3/                    # publication figures
```

## Timeline

| Week | Activity |
|------|----------|
| 1 | Phase 1: build architecture, train on text, verify |
| 2 | Phase 2: multi-modal training on Mac Studio |
| 3 | Phase 2 analysis + Phase 3 if results are positive |
| 4 | Write up results |
