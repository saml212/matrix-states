# References

## Core Building Blocks

### Learned Segmentation / Byte-Level Models

**Byte Latent Transformer (BLT)** — Meta, Dec 2024
Entropy-based dynamic patching. First byte-level model to match tokenized LLMs at 8B scale.
- Paper: https://arxiv.org/abs/2412.09871
- Code: https://github.com/facebookresearch/blt

**Multiscale Byte Language Models (MBLM)** — IBM, Feb 2025
Hierarchical dynamic chunking from raw bytes. Multi-modal (text, images, audio, music, CPU traces). No domain-specific components. Hits 4/5 of our target properties.
- Paper: https://arxiv.org/abs/2502.14553
- Code: https://github.com/ai4sd/multiscale-byte-lm

**bGPT** — 2024
Raw bytes from text, audio, images, executables. Same architecture, same hyperparameters, same objective for everything. Cross-modality pretraining helps.
- Paper: https://arxiv.org/abs/2402.19155
- Project: https://byte-gpt.github.io/

**MrT5** — Stanford, ICLR 2025
Learned delete gates for byte-level models. 30-75% sequence reduction. Discovers task-specific structure (word boundaries, vowel patterns) without supervision.
- Paper: https://arxiv.org/abs/2410.20771

**ByteFlow** — March 2026
Compression-driven segmentation via coding rate of latent representations. Outperforms BPE-based transformers and prior byte-level architectures.
- Paper: https://arxiv.org/abs/2603.03583

**MEGABYTE** — Meta, 2023
Multi-scale transformer for million-byte sequences. Local model within patches, global model between patches.
- Paper: https://arxiv.org/abs/2305.07185

**EvaByte** — 2025
6.5B parameter byte-level LLM. Demonstrates language-specific compression patterns emerge at scale.
- Project: https://hkunlp.github.io/blog/2025/evabyte/

**Bolmo** — AI2, Dec 2025
Byteifies existing subword models. 7B model matches Olmo 3 on broad benchmarks, +20 points on character tasks.
- Paper: https://arxiv.org/abs/2512.15586
- Blog: https://allenai.org/blog/bolmo

### Higher-Dimensional Representations

**PHM Layers (Parameterized Hypercomplex Multiplication)** — ICLR 2021 Outstanding Paper
Learned algebraic multiplication rules via sums of Kronecker products. 1/n parameters vs dense layers. The multiplication rules are learned, not fixed to quaternion/octonion.
- Paper: https://openreview.net/forum?id=rcQdycl0zyk
- arXiv: https://arxiv.org/abs/2102.08597

**Quaternion Transformer** — ACL 2019
75% parameter reduction with competitive performance across 8 NLP tasks. Hamilton product couples 4 sub-dimensions.
- Paper: https://arxiv.org/abs/1906.04393

**Compacter** — NeurIPS 2021
PHM-based adapter layers. Training only 0.047% of parameters matches full fine-tuning on GLUE.
- Paper: https://arxiv.org/abs/2106.04647

**Geometric Algebra Transformer (GATr)** — 2023
16-dimensional multivector representations. Outperforms baselines on physics simulation. FFN layers become optional.
- Paper: https://arxiv.org/abs/2305.18415
- Code: https://github.com/Qualcomm-AI-research/geometric-algebra-transformer

**CliffordNet** — Jan 2026
Geometric algebra for vision. 8x parameter efficiency on CIFAR-100. The geometric product is so representationally dense that FFN layers are redundant.
- Paper: https://arxiv.org/abs/2601.06793

**Higher-Order Transformers (HOT)** — Dec 2024
Kronecker-structured attention for tensor-valued inputs. Reduces attention complexity for multi-way data.
- Paper: https://arxiv.org/abs/2412.02919

**Deep Tensor Network** — 2023, updated 2025
Degree-2 polynomial tensor attention. Captures higher-order statistics dot-product attention misses.
- Paper: https://arxiv.org/abs/2311.11091

**Training Tensor Attention Efficiently** — 2024
Proves tensor attention backward pass is near-linear time. Makes higher-order attention practical.
- Paper: https://arxiv.org/abs/2405.16411

**Tensor Fusion Network** — EMNLP 2017
Outer product of modality embeddings for multi-modal fusion. Parameter-free fusion layer.
- Paper: https://aclanthology.org/D17-1115/

**Matrix Neural Networks (MatNet)** — 2016
Neurons take matrices directly as inputs via bilinear mapping. Proof of concept.
- Paper: https://arxiv.org/abs/1601.03805

**Smolensky's Tensor Product Representations** — 1990
Variable bindings as tensor products of filler and role vectors. Foundational theory.
- Paper: http://www.lscp.net/persons/dupoux/teaching/AT1_2014/papers/Smolensky_1990_TensorProductVariableBinding.AI.pdf

**Soft TPR** — NeurIPS 2024
Continuous, fully distributed tensor product representations. Learns compositional structure without symbolic decomposition.
- Paper: https://arxiv.org/abs/2412.04671
- Code: https://github.com/gomb0c/soft_tpr

### Multi-Modal / Domain-Agnostic Architectures

**Perceiver IO** — DeepMind, 2021
Raw bytes for text, raw pixels for vision, raw audio. One architecture. Matched BERT on GLUE from UTF-8 bytes.
- Paper: https://arxiv.org/abs/2107.14795

**Zonkey** — Jan 2026
Differentiable tokenization via probabilistic segmentation. Boundaries emerge as linguistically meaningful. Fully end-to-end.
- Paper: https://arxiv.org/abs/2601.21768

**MAGNET** — NeurIPS 2024
Gradient-based boundary prediction via Gumbel-softmax, end-to-end with the LM.
- Paper: https://arxiv.org/abs/2407.08818

**HighMMT** — 2023
10 modalities, 15 tasks. Quantifies interaction heterogeneity across modalities.
- Paper: https://openreview.net/forum?id=ttzypy3kT7

### Hypercomplex Multi-Modal Fusion

**Hierarchical Hypercomplex Network** — 2024
PHM-based multi-modal emotion recognition. Hypercomplex multiplication captures inter-modal correlations.
- Paper: https://arxiv.org/abs/2409.09194

**RP-KrossFuse** — NeurIPS 2025
Kronecker product fusion. Theoretical justification: fused kernel encodes union of discriminative structures from each modality.
- Paper: https://arxiv.org/abs/2506.08645

### Tokenizer Research

**VOLT** — ACL 2021 Best Paper
Vocabulary via optimal transport. 70% vocabulary reduction with 0.5 BLEU gain.
- Paper: https://arxiv.org/abs/2012.15671

**SuperBPE** — COLM 2025
Two-pass BPE with cross-word merges. 33% fewer tokens, +4% across 30 benchmarks at 8B scale.
- Paper: https://arxiv.org/abs/2503.13423

**ADAT** — NeurIPS 2024
Adaptive tokenizer refined during LM training based on perplexity signals.
- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdf00c97c0cb2cc35179f03363da6c4f-Abstract-Conference.html

**T-FREE** — EMNLP 2024
Tokenizer-free via sparse character trigram hashing. 85% embedding parameter reduction.
- Paper: https://arxiv.org/abs/2406.19223

**Scaling Laws with Vocabulary** — NeurIPS 2024
Larger models need larger vocabularies. Llama2-70B optimal vocab predicted at 216K+ (7x its actual 32K).
- Paper: https://arxiv.org/abs/2407.13623

**Information-Theoretic Perspective on Tokenizers** — Jan 2026
Fundamental tension between compression efficiency, linguistic granularity, and domain robustness.
- Paper: https://arxiv.org/abs/2601.09039

### Neuroscience

**Dimensionality of Cognition** — Trends in Cognitive Sciences, 2024
Brain uses gradient of dimensionality across PFC. High dimensionality is causally important for correct cognition.
- Paper: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00189-X

**PFC Dimensionality and Cognitive Control** — Journal of Neuroscience, Feb 2025
Dimensionality collapses on error trials. Maximal during correct cognitive performance.
- Paper: https://www.jneurosci.org/content/45/6/e0233242024

**Higher-Dimensional Representations and Memory** — Science Advances, 2022
Greater neural representational dimensionality predicts better subsequent episodic memory.
- Paper: https://www.science.org/doi/10.1126/sciadv.abm3829

**Grid Cells and Abstract Reasoning** — bioRxiv, 2024
Grid-cell-like codes operate in non-spatial conceptual spaces. Their maturation predicts intelligence.
- Paper: https://www.biorxiv.org/content/10.1101/2024.11.20.624569v1.full

**High-Dimensional Brain in High-Dimensional World** — 2020
The brain exploits the blessing of dimensionality: in high-dimensional spaces, random vectors are nearly orthogonal.
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7516518/

**Thousand Brains Theory** — Numenta, 2024
One cortical algorithm repeated everywhere. Same structure learns different representations from different inputs.
- Paper: https://arxiv.org/abs/2412.18354
- Code: https://github.com/thousandbrainsproject/tbp.monty

### Bio-Inspired Computation

**Dendritic Computation** — Nature Communications, 2025
Dendritic ANNs match or outperform standard ANNs with fewer parameters. Multi-class mixed selectivity.
- Paper: https://www.nature.com/articles/s41467-025-56297-9

**ReSU (Rectified Spectral Units)** — AAAI 2026
Self-supervised units from temporal statistics. No backpropagation. Matches biological fly neurons.
- Paper: https://arxiv.org/abs/2512.23146

**KAN (Kolmogorov-Arnold Networks)** — ICLR 2025
Learnable activation functions per edge. Faster scaling laws than MLPs on function fitting.
- Paper: https://arxiv.org/abs/2404.19756

**SpikingLLM** — submitted ICLR 2026
Spiking LLM achieves ANN-level performance at 4-6% compute cost.
- Paper: https://openreview.net/forum?id=r6fNn987rr

**Deep Oscillatory Neural Network** — Scientific Reports, 2025
Complex-valued Hopf oscillators. Phase relationships enable binding that standard ANNs lack.
- Paper: https://www.nature.com/articles/s41598-025-24837-4

**Predictive Coding** — Nature Communications, 2025
Brain-inspired prediction error propagation. No global backpropagation needed.
- Paper: https://www.nature.com/articles/s41467-025-64234-z
