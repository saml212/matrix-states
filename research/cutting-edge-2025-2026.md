# Cutting Edge: 2025-2026 Research Relevant to Our Project

Live web search results, March 2026.

## The Most Important Recent Ideas

### 1. Energy-Based Transformers (July 2025)
- Paper: arxiv.org/abs/2507.02092
- **Key idea:** Instead of next-token prediction, assign an ENERGY to each (input, prediction) pair.
  Inference = gradient descent to minimize energy until convergence.
- **"Thinking" = iterative refinement from random noise toward a converged prediction.**
- 35% higher scaling rate than standard transformers.
- Easier tokens (like "the") converge faster. Harder tokens stay high-energy longer.
- **THIS IS CLOSEST TO THE MATRIX-THINKING IDEA.** Replace "energy of a vector"
  with "rank of a matrix" and you have the user's proposal.

### 2. CliffordNet (January 2026)
- Paper: arxiv.org/abs/2601.06793
- Fixed geometric algebra (Clifford product). NOT learned — imposed.
- 8x parameter efficiency. FFN layers become OPTIONAL.
- 77.82% CIFAR-100 with 1.4M params (matches ResNet-18 at 11.2M).
- **Proves algebraic structure helps when FIXED, not learned.** Confirms our finding.

### 3. EvaByte (2025)
- hkunlp.github.io/blog/2025/evabyte/
- First open-source byte-level model matching tokenized LMs at scale.
- Can treat JPEG images as byte streams. Interleaves image + text bytes.
- No architecture modifications needed for multi-modal — just bytes.
- **Validates our entire premise.**

### 4. TensorLens (January 2026)
- Paper: arxiv.org/abs/2601.17958
- Represents the ENTIRE transformer as a single high-order tensor.
- Captures attention, FFNs, activations, normalization as one unified tensor operator.
- Shows transformers already implicitly work with tensor structure — we just don't
  expose it or use it intentionally.

### 5. Test-Time Compute / Inference Scaling (2025-2026)
- Paper: arxiv.org/abs/2512.02008
- A 7B model with 100x inference compute can match a 70B model.
- "Thinking longer" matters more than "being bigger" for reasoning.
- DeepSeek-R1 proved this at scale.
- **Connects to matrix-thinking:** if the model thinks in matrices and converges,
  harder problems naturally get more compute (higher rank = more iterations).

### 6. Genie 3 (August 2025, Google DeepMind)
- First real-time interactive general-purpose world model.
- Navigable 3D worlds at 24fps from raw visual data.
- Learns physics implicitly from sensory input.
- **Validates that transformers can learn world dynamics from raw data.**

### 7. TokenFlow (CVPR 2025)
- Unified image tokenizer for understanding AND generation.
- Dual-codebook: separates semantic vs pixel-level features.
- Beats LLaVA-1.5 on understanding, matches SDXL on generation.
- **Shows you can unify perception and generation in one tokenizer.**

## What This Means for Our Project

### What we should KEEP:
- Byte-level processing (validated by EvaByte, BLT)
- Fixed algebraic structure like CliffordNet (not learned PHM)
- Segmentation/compression before transformer (validated by all byte models)

### What we should ADD:
- **Energy-based or iterative refinement** (EBTs show 35% better scaling)
- **Higher-dimensional internal representations** (CliffordNet uses 16D multivectors)
- **Test-time compute scaling** (think longer on harder inputs)

### What we should CHANGE:
- Drop "learned algebra" — use fixed Clifford/geometric algebra instead
- Consider CliffordNet's geometric product as replacement for PHM
- Consider EBT's energy minimization as the "thinking in matrices" mechanism

### The Novel Gap (what nobody has done):
Nobody has combined:
1. Byte-level input (any domain)
2. Fixed geometric algebra (CliffordNet-style) for structured representations
3. Energy-based iterative refinement (EBT-style) for adaptive thinking
4. Multi-modal processing without domain labels

This combination is our paper.
