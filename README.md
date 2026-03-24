# Learned Representations

Can a single architecture learn to process any domain — text, images, audio, scientific data — from raw bytes, without domain-specific tokenizers or encoders, using richer-than-vector representations?

## The Idea

Current models use two domain-specific bottlenecks:

1. **Tokenizers** — hand-designed per domain (BPE for text, patch embedding for images, mel spectrogram for audio). A human decides how to chop up the input before the model ever sees it.

2. **Flat vector representations** — every token is a 1D vector of d numbers. This encodes d features but not the relationships between features. A matrix or tensor representation would encode both features and their structure.

This project removes both bottlenecks:

- Replace fixed tokenizers with **learned segmentation** — the model discovers its own boundaries from raw bytes, adapting to whatever domain it's processing.
- Replace flat vectors with **learned hypercomplex multiplication (PHM)** — each layer's computation uses algebraic structure (sums of Kronecker products) that the model learns from data, giving richer per-token representations at the same parameter count.

The hypothesis: a model with learned segmentation and learned algebraic structure will discover domain-appropriate processing strategies without being told what domain it's looking at.

## Why This Might Work

### From neuroscience
- The brain uses the same cortical column architecture everywhere — visual cortex, auditory cortex, prefrontal cortex all share the same six-layer structure. Different areas learn different representations from different inputs using the same algorithm.
- The brain operates in high-dimensional spaces. PFC dimensionality predicts cognitive performance and collapses on error trials. Grid cells use the same multi-scale encoding for both spatial navigation and abstract reasoning.
- Higher neural representational dimensionality predicts better episodic memory (Science Advances, 2022).

### From ML
- bGPT (2024) processes raw bytes from text, audio, images, and executables with one architecture and zero domain customization. Cross-domain pretraining helps.
- BLT (Meta, 2024) discovers linguistically meaningful segmentation boundaries from raw bytes using entropy, with no linguistic supervision.
- PHM layers (ICLR 2021 Outstanding Paper) learn algebraic multiplication rules via Kronecker products, achieving 4x parameter reduction at matched quality. The learned rules subsume quaternions but can discover better algebras.
- Perceiver IO (DeepMind, 2021) processes raw bytes for text and raw pixels for vision through one architecture. Matched BERT on GLUE from raw UTF-8 bytes.
- MBLM (IBM, Feb 2025) combines raw bytes, learned hierarchical segmentation, and multi-modal training with no domain encoders. Hits 4/5 of our target properties — missing only higher-dimensional representations.

### The gap
Nobody has combined learned segmentation with higher-dimensional representations across modalities. These two research threads have never intersected, even at toy scale.

## Architecture

### Input
Raw bytes. No tokenizer. Text comes in as UTF-8 bytes. Images as raw pixel bytes. Audio as raw PCM samples. Everything is a byte stream.

### Learned Segmentation (from MBLM / BLT)
A small initial transformer processes bytes and outputs boundary probabilities via Gumbel-softmax (differentiable). The model learns where to place segment boundaries — grouping bytes into variable-length tokens. During training, boundaries emerge based on information density: more tokens for surprising content, fewer for predictable content.

### PHM Layers (from ICLR 2021)
All linear layers in the transformer are replaced with Parameterized Hypercomplex Multiplication layers. Instead of y = Wx, each layer computes y = (sum of Kronecker products) * x. The Kronecker factors are learned, so the model discovers its own algebraic structure. This gives:
- 4x parameter reduction per layer (or 4x more capacity at matched params)
- Richer inter-component interactions (the algebraic structure couples dimensions)
- The possibility that different layers or different domains converge to different algebras

### Transformer
Standard transformer stack (attention + PHM-MLP), operating on the variable-length segments produced by the learned segmenter. Tied embeddings via PHM-structured projection.

### Output
Next-byte prediction. The model predicts the next byte in the stream, regardless of domain.

## Experiment Phases

### Phase 1: Get It Running (1-2 days, M4 Mac Mini)
- Build the architecture: MBLM's byte segmentation + PHM layers
- 10M parameters, tiny model
- Train on 10MB of text only (FineWeb subset)
- Verify: does it train? Does loss decrease? Do boundaries emerge?
- No multi-modal yet. Just prove the components work together.

### Phase 2: Multi-Modal Signal (3-5 days, Mac Studio)
- 50M parameters
- Train on three interleaved byte streams:
  - Text: 100MB of FineWeb (UTF-8 bytes)
  - Images: 100MB of CIFAR-10 (raw pixel bytes, 3072 bytes per image)
  - Audio: 100MB of LibriSpeech (raw 16-bit PCM samples as bytes)
- Interleave randomly: each batch has a mix of text, image, and audio sequences
- The model does NOT know which domain each sequence is from
- **Measure:**
  1. Do boundary probabilities differ by domain? (Word boundaries for text? Edge boundaries for images? Phoneme boundaries for audio?)
  2. Do the learned Kronecker factors (PHM algebra) differ by layer or converge to one structure?
  3. Does multi-modal training help or hurt per-domain loss compared to single-domain?

### Phase 3: Scale and Analyze (1-2 days, rented H100)
- If Phase 2 shows the signal, scale to 200M parameters with 1GB per domain
- Quantitative comparison:
  - Learned segmentation vs BPE/fixed patches: which gives lower loss per domain?
  - PHM layers vs standard linear: does the algebraic structure help at matched params?
  - Cross-domain transfer: does text pretraining help image compression? Does image pretraining help text?
- Visualization of learned boundaries across domains
- Analysis of learned algebraic structure

### Phase 4: Write It Up
- If Phase 2/3 show emergent domain-specific segmentation from a single architecture, that's the paper
- If the PHM algebra converges to something interpretable, that's a second finding
- If cross-domain transfer is positive, that's a third

## What Success Looks Like

**Minimum viable result:** The boundary predictor produces visibly different segmentation patterns for text vs images vs audio, without being told which is which. This alone is publishable.

**Strong result:** Multi-modal pretraining improves per-domain loss compared to single-domain training at matched compute. This would demonstrate that the architecture learns transferable structure.

**Home run:** The learned PHM algebraic structure reveals something interpretable — e.g., text processing converges to a rotation-like algebra while image processing converges to a reflection-like algebra. This would be a genuine discovery about the relationship between data structure and computational structure.

## What Failure Looks Like

- The boundary predictor produces uniform segmentation regardless of domain (no emergent structure)
- PHM layers underperform standard linear layers at matched params (the algebraic bias hurts more than it helps)
- Multi-modal training hurts all domains (interference outweighs transfer)

Any of these would be a useful negative result worth documenting.

## Hardware Requirements

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Phase 1 | M4 Mac Mini 32GB | 1-2 days | $0 (owned) |
| Phase 2 | M4 Ultra Mac Studio 256GB | 3-5 days | $0 (friend's) |
| Phase 3 | 1x H100 cloud | 1-2 days | ~$50-100 |

Total estimated cost: under $100 for the full experiment.

## Key Dependencies

### Code
- PyTorch (runs on Apple Silicon via MPS)
- MBLM: github.com/ai4sd/multiscale-byte-lm (byte segmentation reference)
- PHM layers: from the ICLR 2021 paper, open source
- BLT: github.com/facebookresearch/blt (entropy-based patching reference)

### Data
- Text: HuggingFace datasets (FineWeb, or any web text)
- Images: CIFAR-10 (torchvision, trivial to load as raw bytes)
- Audio: LibriSpeech (torchaudio, load as raw PCM)

### No special hardware needed for Phase 1-2.
