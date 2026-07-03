# References

This file lists all papers and code referenced across the project. Organized by topic. The April 2026 multi-agent research session significantly expanded this list — see `research/april-2026-synthesis.md` for context on which papers matter for which research questions.

---

## Continuous Reasoning (the experiment's intellectual home)

### Foundational

**COCONUT — Training Large Language Models to Reason in a Continuous Latent Space** — Hao et al., Meta, ICLR 2025
The seminal paper. Last hidden state fed back as next input embedding, curriculum to replace text reasoning with continuous thoughts. GSM8K 34.1%, ProsQA 97.0%.
- Paper: https://arxiv.org/abs/2412.06769
- Code: https://github.com/facebookresearch/coconut

**CODI — Compressing Chain-of-Thought into Continuous Space via Self-Distillation** — Shen et al., EMNLP 2025
Strongest COCONUT successor. Joint teacher-student via shared weights, L1 distillation at the ":" token across all layers, simpler training (no curriculum). **GSM8K 43.7%** vs CoT 44.1%. **Strongest baseline for the matrix-CODI experiment.**
- Paper: https://arxiv.org/abs/2502.21074
- Code: https://github.com/zhenyi4/codi

**Quiet-STaR — Language Models Can Teach Themselves to Think Before Speaking** — Zelikman et al., 2024
Inserts thought tokens at every position, REINFORCE-trained. Mixing head prevents thought destruction.
- Paper: https://arxiv.org/abs/2403.09629

**Pause Tokens — Think before you speak** — Goyal et al., ICLR 2024
Single learnable pause token used as extra computation slots. Works only when present in both pretraining and finetuning.
- Paper: https://arxiv.org/abs/2310.02226

### Theoretical foundation for the rank-superposition hypothesis

**Reasoning by Superposition** — Zhu et al., May 2025
Theoretical paper proving 2-layer transformer with continuous CoT can do parallel BFS. Hand-designed continuous thoughts as `t_c = (1/√|V_c|) Σ u_v` over BFS frontier vertices. Rank equals frontier size by construction. **Defines the hypothesis the matrix-CODI experiment will test empirically.**
- Paper: https://arxiv.org/abs/2505.12514

**Continuous Chain of Thought Enables Parallel Exploration and Reasoning (CoT2)** — Gozeten et al., ICLR 2026
Most relevant prior art for the rank-superposition argument. Proves parallelism scales with embedding dimension via existence construction (1-layer transformer + MoE MLP solving MNNS). Empirically tests dimension thresholds via accuracy curves. **Never measures rank or any structural property.**
- Paper: https://arxiv.org/abs/2505.23648
- Code: https://github.com/alperengozeten/CoT2

**Emergence of Superposition** — Sep 2025
Studies training dynamics of continuous CoT on graph reachability. Proves index-matching logit grows then saturates. Does NOT measure rank.
- Paper: https://arxiv.org/abs/2509.23365

### The 2026 rebuttal

**The Illusion of Superposition** — 2026
Argues fine-tuned COCONUT reaches 96.6% **without any latent tokens** and entity-probes show no stepwise computation. The superposition claim is contested. **The matrix-CODI experiment can adjudicate this dispute.**
- Paper: https://arxiv.org/abs/2604.06374

### COCONUT successors

**CoLaR — Dynamic Latent Compression of Reasoning Chains** — Xiaomi, NeurIPS 2025
Two-stage SFT + RL with learnable compression factor. 14.1% improvement over latent baselines.
- Paper: https://arxiv.org/abs/2505.16552

**MarCos — Markov Chain of Continuous Thoughts** — 2509
Latent HMM formulation, separates thinking (Markov transitions) from speaking (emissions). Claims parity with token CoT and 15.7× speedup.
- Paper: https://arxiv.org/abs/2509.25020

**SIM-CoT — Supervised Implicit Chain of Thought** — Sep 2025
Addresses COCONUT's unstable latent training.
- Paper: https://arxiv.org/abs/2509.20317

**PCCoT — Parallel Continuous Chain-of-Thought with Jacobi Iteration** — Wu et al., Jun 2025
Jacobi-iteration parallel updates of all latent slots simultaneously.
- Paper: https://arxiv.org/abs/2506.18582

**Towards Inference-time Scaling for Continuous Space Reasoning** — Oct 2025
- Paper: https://arxiv.org/abs/2510.12167

**A Survey on Latent Reasoning** — Zhu et al., 2025
- Paper: https://arxiv.org/abs/2505.16782

---

## JEPA Family (LeCun's non-generative direction)

**LeJEPA — Provable and Scalable Self-Supervised Learning Without the Heuristics** — Balestriero & LeCun, Nov 2025
**SIGReg collapse fix.** Sketched Isotropic Gaussian Regularization. Single hyperparameter, no stop-gradient, no EMA. Proven optimal target distribution under linear/k-NN/kernel probes. 20 lines of code, drops in.
- Paper: https://arxiv.org/abs/2511.08544
- Code: https://github.com/galilai-group/lejepa (formerly rbalestr-lab/lejepa)

**LLM-JEPA — LLMs Meet Joint Embedding Predictive Architectures** — Huang, LeCun et al., Sep 2025
Adds JEPA aux loss to standard LLM training. Uses paired-modality views (NL ↔ code/regex/SQL). Statistically significant gains on GSM8K and other benchmarks. **Template for matrix-aux-loss approaches.**
- Paper: https://arxiv.org/abs/2509.14252
- Code: https://github.com/rbalestr-lab/llm-jepa

**LeWorldModel (LeWM)** — Maes, LeCun et al., Mar 2026
First JEPA trained end-to-end from raw pixels with no stop-grad / EMA / frozen encoders. Only 15M params, single GPU. Uses SIGReg from LeJEPA, unchanged.
- Paper: https://arxiv.org/abs/2603.19312
- Project: https://le-wm.github.io/

**V-JEPA 2** — Meta FAIR, Jun 2025
1.2B params, video JEPA. SSv2 77.3 top-1, EK100 39.7 R@5. Robot pick-and-place 65-80% on novel objects.
- Paper: https://arxiv.org/abs/2506.09985

**V-JEPA 2.1** — Meta FAIR, Mar 2026
Dense features, hierarchical self-supervision. +23-27 pts segmentation vs V-JEPA 2.
- Paper: https://arxiv.org/abs/2603.14482

**VL-JEPA** — Meta, Dec 2025
Predicts continuous text embeddings instead of autoregressive tokens. 1.6B params. Matches CLIP/SigLIP with 50% fewer trainable params.
- Paper: https://arxiv.org/abs/2512.10942

**ThinkJEPA** — Mar 2026
VLM acts as "thinker" on sparse frames while JEPA latent predictor handles dense temporal dynamics.
- Paper: https://arxiv.org/abs/2603.22281

**WavJEPA** — 2025
JEPA on raw audio waveforms. **Closest existing analog to byte-level JEPA.** Predicts patch-level semantic embeddings, not individual samples.
- Paper: https://arxiv.org/abs/2509.23238

**Audio-JEPA** — 2025
JEPA on mel-spectrogram patches. Vanilla I-JEPA recipe transferred to audio.
- Paper: https://arxiv.org/abs/2507.02915

**JEPA as a Neural Tokenizer** — Dec 2025
Closest to "JEPA as alternative to learned vocabularies" framing. Stage 1 JEPA + Density Adaptive Attention learns latents at 2.5 Hz, Stage 2 FSQ discretizes to 47.5 tokens/sec with 16,384 vocabulary. Speech-only.
- Paper: https://arxiv.org/abs/2512.07168

**ACT-JEPA — Joint-Embedding Predictive Architecture for Efficient Policy Representation** — Jan 2025
- Paper: https://arxiv.org/abs/2501.14622

---

## Pure-Sensor / Language-Free Representation Learning

**DINOv3** — Siméoni, Vo, Seitzer et al., Meta AI, Aug 2025
ViT-7B, 1.7B images, no text, no labels. **First SSL model to beat weakly-supervised peers across the board** including classification. Gram anchoring fixes dense-feature degradation at scale.
- Paper: https://arxiv.org/abs/2508.10104

**Web-SSL — Scaling Language-Free Visual Representation Learning** — Fan et al., Meta, Apr 2025
Trained SSL and CLIP on identical MetaCLIP data, scaled to 7B. Visual SSL matches CLIP on VQA and classic vision benchmarks. **CLIP's prior advantage was data, not language.**
- Paper: https://arxiv.org/abs/2504.01017

**Data or Language Supervision: What Makes CLIP Better than DINO?** — Liu et al., Oct 2025, EMNLP Findings
Same architecture, data, config, matched ImageNet accuracy. CLIP wins text-intensive and fine-grained tasks; DINO wins vision-centric and low-level. Language is a narrow bias toward nameable categories.
- Paper: https://arxiv.org/abs/2510.11835

**Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?** — Oct 2025
Probes patch embeddings with IsSameObject classifier. >90% accuracy in DINOv2/DINO/MAE but **NOT** in supervised ViTs. Object binding emerges from SSL specifically.
- Paper: https://arxiv.org/abs/2510.24709

**Massively Multimodal Foundation Models with Specialized MoE** — Sep 2025
- Paper: https://arxiv.org/abs/2509.25678

**SONAR — Self-Distilled Continual Pre-training for Domain-adaptive Audio SSL** — Sep 2025
- Paper: https://arxiv.org/abs/2509.15703

**Revisiting the Platonic Representation Hypothesis: An Aristotelian View** — 2026
- Paper: https://arxiv.org/abs/2602.14486

**The Platonic Representation Hypothesis (original)** — 2024
Argues representations converge across modalities at scale.
- Paper: https://arxiv.org/abs/2405.07987

---

## Structured Representations Beyond Flat Vectors

### Hyperbolic / Lie / Manifold

**HELM — Hyperbolic Large Language Models via Mixture-of-Curvature Experts** — He et al., NeurIPS 2025
**First billion-parameter fully hyperbolic LLM.** Reports up to +4% on MMLU/ARC vs Euclidean baselines (actual deltas 0.5-2.3 points on near-chance benchmarks). All operations live on the Lorentz manifold via space-like-only ops + constraint reconstruction. **Existence proof that pervasive structured architecture works at scale.**
- Paper: https://arxiv.org/abs/2505.24722
- Code: https://github.com/Graph-and-Geometric-Learning/helm

**HyperET — Efficient Training in Hyperbolic Space for Multi-modal LLMs** — NeurIPS 2025 Oral
- Paper: https://arxiv.org/abs/2510.20322

**HypLoRA — Hyperbolic Fine-tuning for LLMs** — Yang et al., NeurIPS 2025
- Paper: https://arxiv.org/abs/2410.04010

**Hyperbolic Large Language Models (survey)** — Patil et al., Dec 2025
- Paper: https://arxiv.org/abs/2509.05757

**HyperHELM — Hyperbolic Hierarchy Encoding for mRNA Language Modeling** — Sep 2025
- Paper: https://arxiv.org/abs/2509.24655

**Clebsch-Gordan Transformer** — Sep 2025
- Paper: https://arxiv.org/abs/2509.24093

### Geometric Algebra / Clifford

**CliffordNet** — Jan 2026
Vision backbone with full geometric product. Nano variant: 77.82% on CIFAR-100 with 1.4M params, matching ResNet-18 with 8× fewer params. Claims FFN becomes redundant.
- Paper: https://arxiv.org/abs/2601.06793

**Geometric Algebra Transformer (GATr)** — 2023
16-dimensional multivector representations. Strong on physics simulation.
- Paper: https://arxiv.org/abs/2305.18415
- Code: https://github.com/Qualcomm-AI-research/geometric-algebra-transformer

**ViNE-GATr** — ICLR 2025 workshop
GATr scaled via Perceiver-style latent tokens.

**Equivariant Spherical Transformer** — May 2025
- Paper: https://arxiv.org/abs/2505.23086

**Platonic Transformers: A Solid Choice For Equivariance** — Oct 2025
- Paper: https://arxiv.org/abs/2510.03511

### TPR / Tensor Product Representations

**Smolensky's Tensor Product Representations** — 1990
Foundational. Variable bindings as tensor products of filler and role vectors.
- Paper: http://www.lscp.net/persons/dupoux/teaching/AT1_2014/papers/Smolensky_1990_TensorProductVariableBinding.AI.pdf

**Soft TPR** — Sun et al., NeurIPS 2024
Continuous, fully distributed tensor product representations. Visual disentanglement, not language.
- Paper: https://arxiv.org/abs/2412.04671
- Code: https://github.com/gomb0c/soft_tpr

**TP-Transformer** — 2021
Each token gets a filler and role bound by tensor product inside layers. Not at input embedding.
- Paper: https://arxiv.org/abs/2106.01317

**Attention-based Iterative Decomposition for TPR** — Park et al., Jun 2024
- Paper: https://arxiv.org/abs/2406.01012

### PHM / Quaternion / Hypercomplex

**PHM Layers** — ICLR 2021 Outstanding Paper
Learned algebraic multiplication via Kronecker products. 1/n parameters vs dense.
- Paper: https://arxiv.org/abs/2102.08597

**Quaternion Transformer** — ACL 2019
75% parameter reduction with competitive performance.
- Paper: https://arxiv.org/abs/1906.04393

**Compacter** — NeurIPS 2021
PHM-based adapter layers.
- Paper: https://arxiv.org/abs/2106.04647

### Higher-Order Tensor

**Higher-Order Transformers (HOT)** — Dec 2024
- Paper: https://arxiv.org/abs/2412.02919

**Deep Tensor Network** — 2023, updated 2025
- Paper: https://arxiv.org/abs/2311.11091

**Training Tensor Attention Efficiently** — 2024
- Paper: https://arxiv.org/abs/2405.16411

**Matrix Neural Networks (MatNet)** — 2016
Neurons take matrices directly via bilinear mapping.
- Paper: https://arxiv.org/abs/1601.03805

---

## Scaling vs Structure Debate

**Scaling can lead to compositional generalization** — Redhardt, Akram, Schug, NeurIPS 2025 Spotlight
Synthetic MLPs achieve compositional generalization given full task-space coverage. **The strongest scaling-side result** but the synthetic setup doesn't transfer to language.
- Paper: https://arxiv.org/abs/2507.07207

**Does equivariance matter at scale?** — Brehmer et al., TMLR Jul 2025
Often misread as pro-scaling. Actually shows equivariant models maintain ~2x compute-efficiency advantage at every budget tested across 10^16-10^19 FLOPs. **Pro-structure result.**
- Paper: https://arxiv.org/abs/2410.23179

**Deep Learning is Not So Mysterious or Different** — Wilson, ICML 2025
Soft simplicity prior from overparameterization explains generalization without architectural restriction. PAC-Bayes framework.
- Paper: https://arxiv.org/abs/2503.02113

**Compositional Generalization Requires More Than Disentangled Representations** — Jan 2025
- Paper: https://arxiv.org/abs/2501.18797

**Does Data Scaling Lead to Visual Compositional Generalization?** — Jul 2025
- Paper: https://arxiv.org/abs/2507.07102

**Revisiting Compositional Generalization Capability of LLMs** — Jun 2025
- Paper: https://arxiv.org/abs/2506.15629

---

## Discrete Vocabularies / Tokenization (Modern VQ)

**Emu3.5 — Native Multimodal Models are World Learners** — BAAI, Oct 2025
**34.1B params, largest native-discrete-token multimodal model.** Uses separate text BPE + visual VQ codebooks.
- Paper: https://arxiv.org/abs/2510.26583

**Emu3 — Next-Token Prediction is All You Need** — BAAI, Sep 2024
- Paper: https://arxiv.org/abs/2409.18869

**Chameleon — Mixed-Modal Early-Fusion Foundation Models** — Meta, May 2024
**Abandoned by Meta** in favor of BLT (byte-level, tokenizer-free).
- Paper: https://arxiv.org/abs/2405.09818

**Janus-Pro** — DeepSeek, Jan 2025
- Paper: https://arxiv.org/abs/2501.17811

**TokenFlow** — Qu et al., CVPR 2025
- Paper: https://arxiv.org/abs/2412.03069

**UniTok** — Ma, Jiang et al., NeurIPS 2025 Spotlight
- Paper: https://arxiv.org/abs/2502.20321

**QLIP — Text-Aligned BSQ Visual Tokenization** — Feb 2025
- Paper: https://arxiv.org/abs/2502.05178

**BSQ — Image and Video Tokenization with Binary Spherical Quantization** — Zhao et al., ICLR 2025
- Paper: https://arxiv.org/abs/2406.07548

**FSQ — VQ-VAE Made Simple** — Mentzer et al., ICLR 2024
- Paper: https://arxiv.org/abs/2309.15505

**MAGVIT-v2 / LFQ — Language Model Beats Diffusion** — Yu et al., ICLR 2024
- Paper: https://arxiv.org/abs/2310.05737

**Open-MAGVIT2** — Sep 2024
- Paper: https://arxiv.org/abs/2409.04410

**SimVQ — Addressing Representation Collapse with One Linear Layer** — ICCV 2025
100% codebook utilization at 262K codes.

**Robust Residual Finite Scalar Quantization (RFSQ)** — Aug 2025
- Paper: https://arxiv.org/abs/2508.15860

**Towards Improved Text-Aligned Codebook Learning (TA-VQ)** — Mar 2025, CVPR 2025
- Paper: https://arxiv.org/abs/2503.01261

**SemHiTok — Semantic-Guided Hierarchical Codebook** — Mar 2025
- Paper: https://arxiv.org/abs/2503.06764

**Discrete Tokenization for Multimodal LLMs (survey)** — Jul 2025
- Paper: https://arxiv.org/abs/2507.22920

**UniWeTok — Unified Binary Tokenizer with Codebook 2^128** — Feb 2026
- Paper: https://arxiv.org/abs/2602.14178

**VAEVQ — Variational Modeling for Discrete Tokenization** — Nov 2025
- Paper: https://arxiv.org/abs/2511.06863

**VQRAE — Representation Quantization Autoencoders** — Nov 2025
- Paper: https://arxiv.org/abs/2511.23386

---

## Byte-Level Models

**BLT — Byte Latent Transformer** — Meta, ACL 2025
Entropy-based dynamic patching. Matches Llama 3 at 8B scale with 50% fewer inference FLOPs. **Meta's bet against learned tokenizers.**
- Paper: https://arxiv.org/abs/2412.09871
- Code: https://github.com/facebookresearch/blt

**MBLM — Multiscale Byte Language Models** — IBM, ICML 2025
Hierarchical decoder stack. Mamba+Transformer hybrid. 5M byte context on single A100. CLEVR VQA 52.1% with no image encoder.
- Paper: https://arxiv.org/abs/2502.14553
- Code: https://github.com/ai4sd/multiscale-byte-lm

**MambaByte** — Cornell, COLM 2024
Pure Mamba SSM byte model. O(1) memory, O(n) compute.
- Paper: https://arxiv.org/abs/2401.13660

**EvaByte** — HKU + SambaNova, 2025
6.5B byte model with EVA attention + multibyte prediction. 5-10× faster decoding than vanilla byte models.
- Project: https://hkunlp.github.io/blog/2025/evabyte/
- Code: https://github.com/OpenEvaByte/evabyte

**Bolmo — Byteification of OLMo 3** — AI2, Dec 2025
Retrofits existing tokenized models to byte-level. +16.5% absolute STEM improvement over BLT 7B.
- Paper: https://arxiv.org/abs/2512.15586
- Code: https://github.com/allenai/bolmo-core

**bGPT** — Microsoft Research Asia, Feb 2024
Hierarchical Transformer for binary data. Cross-domain byte transfer (text harms non-text more than non-text harms text).
- Paper: https://arxiv.org/abs/2402.19155

**MEGABYTE** — Meta, NeurIPS 2023
The original hierarchical byte model.
- Paper: https://arxiv.org/abs/2305.07185

**MrT5** — Stanford, ICLR 2025
Learned delete gates for byte models. PI-controller, scaled sigmoid.
- Paper: https://arxiv.org/abs/2410.20771

**ByteFlow** — Mar 2026
Compression-driven segmentation via coding rate.
- Paper: https://arxiv.org/abs/2603.03583

---

## Long-Context

**EverMind MSA — Memory Sparse Attention** — EverMind-AI, Mar 2026
**100M token context** on 2× A800 GPUs. Document-wise RoPE, KV cache compression.
- Paper: https://arxiv.org/abs/2603.23516

**Native Sparse Attention (NSA)** — DeepSeek, ACL 2025 Best Paper
Three parallel branches: compressed + selected + sliding.
- Paper: https://arxiv.org/abs/2502.11089

**DeepSeek Sparse Attention (DSA)** — DeepSeek-V3.2, 2025
Lightning indexer at every layer. O(L*k) complexity.
- Paper: https://arxiv.org/abs/2512.02556

**Infini-Attention** — Google, 2024
Compressive memory + local attention.
- Paper: https://arxiv.org/abs/2404.07143

**Ring Attention** — UC Berkeley, ICLR 2024
Distributed context across GPU ring.
- Paper: https://arxiv.org/abs/2310.01889

**MInference** — Microsoft, NeurIPS 2024 Spotlight
Three discovered sparse patterns: A-shape, vertical-slash, block-sparse.

**StreamingLLM** — MIT, ICLR 2024
Attention sinks for infinite generation.
- Paper: https://arxiv.org/abs/2309.17453

**Titans** — Google, NeurIPS 2025
Surprise-gated neural memory.
- Paper: https://arxiv.org/abs/2501.00663

**TTT-Linear / TTT-MLP** — Stanford, 2024
Weight matrices as hidden state.
- Paper: https://arxiv.org/abs/2407.04620

**TTT-E2E** — Dec 2025
Meta-learned test-time training. Constant inference latency.
- Paper: https://arxiv.org/abs/2512.23675

**Gated DeltaNet** — NVIDIA, ICLR 2025
Gating + delta rule. Production in Qwen3.5/Qwen3-Next.
- Paper: https://arxiv.org/abs/2412.06464
- Code: https://github.com/NVlabs/GatedDeltaNet

**DeltaProduct — Products of Householders** — NeurIPS 2025
Tunable expressivity via iterated rank-1 perturbations.
- Paper: https://arxiv.org/abs/2502.10297

**Kimi Linear** — Moonshot AI, Oct 2025
First linear attention hybrid to outperform full attention. 3:1 KDA-to-MLA ratio.
- Paper: https://arxiv.org/abs/2510.26692

---

## Auxiliary Loss / Multi-Token / Context Prediction

**ContextLM — Context-level Language Modeling by Learning Predictive Context Embeddings** — Oct 2025, ICLR 2026
**Single-modality alternative to LLM-JEPA, no view construction needed.** No auxiliary loss — context predictor trained purely via NTP gradient flow. Mean-pool over chunks → 2-layer predictor → broadcast-add to decoder.
- Paper: https://arxiv.org/abs/2510.20280
- Code: https://github.com/LUMIA-Group/ContextLM

**Future Summary Prediction (FSP)** — Oct 2025
Auxiliary head predicts compact summary of long-term future. +5% on math/coding.
- Paper: https://arxiv.org/abs/2510.14751

**Multi-Token Prediction (Gloeckle et al.)** — 2024
Predict next n tokens with parallel heads.
- Paper: https://arxiv.org/abs/2404.19737

**OLA-VLM** — 2024
Multimodal auxiliary visual embedding distillation.
- Paper: https://arxiv.org/abs/2412.09585

---

## Rank / Dimensionality Measurement

**Attention is not all you need: Pure attention loses rank doubly exponentially with depth** — Dong, Cordonnier & Loukas, ICML 2021
The canonical rank-collapse result. Pure self-attention stacks converge to a rank-1 matrix doubly exponentially in depth; residual connections, MLP blocks, and layer norm are the mitigators. The framing that "depth drives collapse" in transformers traces to this paper. Cited as the baseline assumption that the output-head-dynamics note pushes back on.
- Paper: https://arxiv.org/abs/2103.03404

**Dimensional Collapse in Transformer Attention Outputs** — Aug 2025
Effective rank of attention outputs vs residual streams across GPT-2, Llama 3, Gemma 2, Qwen 3.
- Paper: https://arxiv.org/abs/2508.16929

**Less is More: Local Intrinsic Dimensions of Contextual Language Models** — Jun 2025
Localized TwoNN estimator on contextual embeddings. About fine-tuning and grokking, not reasoning.
- Paper: https://arxiv.org/abs/2506.01034

**Higher Embedding Dimension Creates a Stronger World Model for a Simple Sorting Task** — Oct 2025
Embedding dimensionality directly affects downstream world-model quality at small scale.
- Paper: https://arxiv.org/abs/2510.18315

**Mind the Gap: Spectral Analysis of Rank Collapse** — 2024
- Paper: https://arxiv.org/abs/2410.07799

**Statistical Physics of Language Model Reasoning**
Uses rank-40 PCA projection capturing ~50% of variance to fit SDE on sentence-level hidden-state trajectories.
- Paper: OpenReview MbJXVbwSir

**Token Embeddings Violate the Manifold Hypothesis** — Apr 2025
- Paper: https://arxiv.org/abs/2504.01002

**Measuring Intrinsic Dimension of Token Embeddings** — Mar 2025
GPT-2 family token embeddings have intrinsic dimension ~24-31 across ext dims 768-1600.
- Paper: https://arxiv.org/abs/2503.02142

**Origin of Self-Attention** — Jul 2025
Frames self-attention as a learned pairwise affinity matrix.
- Paper: https://arxiv.org/abs/2507.14560

**Minimax Rates for Learning Pairwise Interactions in Attention-Style Models** — Oct 2025
Treats tokens as particles with pairwise interactions.
- Paper: https://arxiv.org/abs/2510.11789

**Understanding Transformers for Time Series: Rank Structure** — Oct 2025
- Paper: https://arxiv.org/abs/2510.03358

### Added 2026-07-01 (Chapter 2 — Task D/E research + novelty passes)

**Understanding Factual Recall in Transformers via Associative Memories** — Nichani, Lee & Bietti, ICLR 2025 Spotlight
The single closest prior work to Task D. Defines the same linear associative memory `W = Σ u_f(x) e_x^T` and gives a rank-m construction storing ≈md associations — but as a hand-built EXISTENCE proof under DISCRETE ARGMAX decoding, not a measured/trained rank or a necessity bound for exact continuous recovery. This is the exact reason Task D pins its readout to exact continuous recovery (never argmax): under argmax, a rank-1 matrix can recover ≈d associations and the rank≥K necessity collapses.
- Paper: https://arxiv.org/abs/2412.06538

**The Key to State Reduction in Linear Attention: A Rank-based Perspective** — Nazari & Rusch, Feb 2026
**State Rank Dynamics in Linear Attention LLMs** — Sun et al., Feb 2026
Both measure effective rank of the linear-attention state matrix `S_t = Σ v_t k_t^T` (same metric Task D uses) and prove the UPPER bound `rank(S_t) ≤ t`, on pretrained LLMs over real text with uncontrolled t. Task D is the mirror image: controlled K, a LOWER bound (`rank(Z)≥K` for exact recovery), and a causal force-rank-k training ablation — neither paper has any of the three.
- Papers: https://arxiv.org/abs/2602.04852, https://arxiv.org/abs/2602.02195

**Linear Transformers Are Secretly Fast Weight Programmers** — Schlag, Irie & Schmidhuber, ICML 2021
Origin of "linear-attention state as associative memory"; motivates why rank/capacity matters once writes exceed dimension. FWP lineage — Schlag is a Tier-1 outreach target for this project's papers.
- Paper: https://arxiv.org/abs/2102.11174

**Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving (TP-Transformer)** — Schlag et al., 2019/2020
TPR-style binding for math reasoning — the direct ancestor of "matrix structure for reasoning" and the principled design for the eventual byte-window TPR ablation (see STATE.md's byte-vs-token decision).
- Paper: https://arxiv.org/abs/1910.06611

**RNNs Implicitly Implement Tensor Product Representations** — McCoy, Linzen, Dunbar & Smolensky, ICLR 2019
- Paper: https://arxiv.org/abs/1812.08718

**Unlocking State-Tracking in Linear RNNs via Negative Eigenvalues** — Grazzi et al., ICLR 2025
**DeltaProduct**
Composability/state-tracking is gated by eigenvalue SIGN AND PHASE, not rank magnitude alone — directly motivates Task E's C8 eigenstructure-fidelity metric (a rank-K matrix with the wrong eigenvalue structure can satisfy one-hop recall while failing composition).
- Papers: https://arxiv.org/abs/2411.12537, https://arxiv.org/abs/2502.10297

**Zoology: Measuring and Improving Recall in Efficient Language Models** — Arora et al., 2023/2024
Standard synthetic associative-recall (MQAR) benchmark family Task D/E's grammar resembles; varies state dimension against fixed pair count, no rank spectra.
- Paper: https://arxiv.org/abs/2312.04927

**Grokked Transformers are Implicit Reasoners** — Wang et al., NeurIPS 2024
**Faith and Fate: Limits of Transformers on Compositionality** — Dziri et al., 2023
Standing counter-evidence/caution for any compositional-generalization claim (Task E): transformers can grok in-distribution composition yet fail to generalize to novel combinations, or reduce to linearized subgraph matching. Task E's claim is deliberately narrower — about the specific rank-K matrix state Task D showed is causally used, not about transformer compositionality in general.
- Papers: https://arxiv.org/abs/2405.15071, https://arxiv.org/abs/2305.18654

**Holographic Reduced Representations** — Plate, IEEE TNN 1995
**Resonator Networks** — Frady, Kleyko & Sommer, Neural Computation 2020
VSA/HRR capacity is SNR/bundling/codebook-factorization, a DIFFERENT notion from matrix rank — needed to correctly scope what "capacity" means in that tradition vs. Task D's exact-rank framing.
- Papers: (Plate, pre-arXiv), https://arxiv.org/abs/1906.11684

**Tokenization Counts** — Singh & Strouse, 2024
**Efficient Numeracy via BitTokens** — Kreitner et al., 2026
Number/token granularity causally changes arithmetic performance — the evidence that keeps byte-level input a high-priority (not "someday") follow-on ablation once matrix-native scales to real data, per STATE.md's Path Forward.
- Papers: https://arxiv.org/abs/2402.14903, https://arxiv.org/abs/2510.06824

Full annotated novelty/positioning passes: `research/task-d-novelty-july2026.md`, `research/bytes-vs-tokens-matrix-native-june2026.md`. Full related-work section with all Task D/E citations assembled: `matrix-thinking/chapter2/TASK_D_WRITEUP.md` §2.

---

## Multi-Modal / Domain-Agnostic

**Perceiver IO** — DeepMind, 2021
Raw bytes for text, raw pixels for vision, raw audio. One architecture.
- Paper: https://arxiv.org/abs/2107.14795

**Zonkey** — Jan 2026
Differentiable tokenization via probabilistic segmentation.
- Paper: https://arxiv.org/abs/2601.21768

**MAGNET** — NeurIPS 2024
Gradient-based boundary prediction.
- Paper: https://arxiv.org/abs/2407.08818

**HighMMT** — 2023
10 modalities, 15 tasks.
- Paper: https://openreview.net/forum?id=ttzypy3kT7

**CoCoMix** — Meta 2025
Continuous concept mixing during pretraining. 21.5% sample efficiency gain.

**ThoughtBubbles** — Stanford 2025
Score-weighted attention masking for unsupervised thought learning.

---

## Tokenizer Research

**VOLT** — ACL 2021 Best Paper
Vocabulary via optimal transport.
- Paper: https://arxiv.org/abs/2012.15671

**SuperBPE** — COLM 2025
Two-pass BPE with cross-word merges.
- Paper: https://arxiv.org/abs/2503.13423

**ADAT** — NeurIPS 2024
Adaptive tokenizer refined during training.

**T-FREE** — EMNLP 2024
Tokenizer-free via sparse character trigram hashing.
- Paper: https://arxiv.org/abs/2406.19223

**Scaling Laws with Vocabulary** — NeurIPS 2024
- Paper: https://arxiv.org/abs/2407.13623

**Information-Theoretic Perspective on Tokenizers** — Jan 2026
- Paper: https://arxiv.org/abs/2601.09039

---

## Energy-Based / Test-Time Compute

**Energy-Based Transformers** — July 2025
Iterative refinement from random noise. 35% better scaling rate.
- Paper: https://arxiv.org/abs/2507.02092

**EBM-CoT** — Chen et al., Nov 2025
Per-thought Langevin dynamics against learned energy.

---

## Hypercomplex Multi-Modal Fusion

**Hierarchical Hypercomplex Network** — 2024
PHM-based multi-modal emotion recognition.
- Paper: https://arxiv.org/abs/2409.09194

**RP-KrossFuse** — NeurIPS 2025
Kronecker product fusion.
- Paper: https://arxiv.org/abs/2506.08645

**Tensor Fusion Network** — EMNLP 2017
Outer product of modality embeddings. Parameter-free fusion.
- Paper: https://aclanthology.org/D17-1115/

---

## Neuroscience / Information Theory of Language

### The mainstream case for non-linguistic cognition

**Fedorenko, Piantadosi, Gibson — "Language is primarily a tool for communication rather than thought"** — *Nature*, 2024
**The strongest published case** that language isn't the cognitive medium. fMRI dissociates language network from reasoning, math, theory of mind, music.

### Neural primitives below language

**Moser et al. — Grid Cells in Cognition: Mechanisms and Function** — *Annual Review of Neuroscience*, 2024

**Chaudhuri, Gerçek, Pandey, Peyrache, Fiete — Intrinsic attractor manifold and population dynamics** — *Nature Neuroscience*, 2019

**Khona & Fiete — Attractor and integrator networks in the brain** — *Nature Reviews Neuroscience*, 2022

**Olshausen & Field — Emergence of simple-cell receptive field properties by sparse coding** — *Nature*, 1996

**Lian & Burkitt — Learning an Efficient Hippocampal Place Map from Entorhinal Inputs Using Non-Negative Sparse Coding** — *eNeuro*, 2021

**Veit & Nieder — Abstract rule neurons in corvid songbirds** — *Nature Communications*, 2013

### Predictive coding / efficient coding

**Barlow — Possible principles underlying the transformations of sensory messages** — 1961
Foundational efficient coding hypothesis.

**Rao & Ballard — Predictive coding in the visual cortex** — *Nature Neuroscience*, 1999

**Chalk, Marre, Tkačik — Toward a unified theory of efficient, predictive, and sparse coding** — *PNAS*, 2018

**Benjamin et al. — Efficient neural codes naturally emerge through gradient descent learning** — *Nature Communications*, 2022

**Friston — The free-energy principle: a unified brain theory?** — *Nature Reviews Neuroscience*, 2010

**Parr, Pezzulo, Friston — Active Inference: The Free Energy Principle in Mind, Brain, and Behavior** — MIT Press, 2022

**Millidge et al. — A survey on neuro-mimetic deep learning via predictive coding** — *Neural Networks*, 2025

### Information theory of language

**Shannon — Prediction and entropy of printed English** — *Bell System Technical Journal*, 1951
~1.0-1.1 bits/character.

**Bentz et al. — Entropy Rate Estimates for Natural Language** — *Entropy*, 2017

**Zaslavsky, Kemp, Regier, Tishby — Efficient compression in color naming and its evolution** — *PNAS*, 2018
Languages near information bottleneck optimum **for communication**, not for thought.

**Levy & Jaeger — Speakers optimize information density through syntactic reduction** — *NeurIPS*, 2007

**Ferrer-i-Cancho & Solé — Least effort and the origins of scaling in human language** — *PNAS*, 2003

**Kanwal et al. — Zipf's Law of Abbreviation and the Principle of Least Effort** — *Cognition*, 2017

**Gustison et al. — Gelada vocal sequences follow Menzerath's linguistic law** — *PNAS*, 2016

### Pre-linguistic / non-verbal cognition

**Spelke & Kinzler — Core knowledge** — *Developmental Science*, 2007

**Dehaene — The Number Sense** — Oxford, 2011

### Cognitive science / language of thought

**Fodor — The Language of Thought** — 1975

**Piantadosi — The computational origin of representation** — *Minds and Machines*, 2021

### Dimensionality of cognition

**Dimensionality of Cognition** — *Trends in Cognitive Sciences*, 2024
Brain uses gradient of dimensionality across PFC.
- Paper: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00189-X

**PFC Dimensionality and Cognitive Control** — *Journal of Neuroscience*, Feb 2025
Dimensionality collapses on error trials.
- Paper: https://www.jneurosci.org/content/45/6/e0233242024

**Higher-Dimensional Representations and Memory** — *Science Advances*, 2022
Greater dimensionality predicts better episodic memory.
- Paper: https://www.science.org/doi/10.1126/sciadv.abm3829

**Grid Cells and Abstract Reasoning** — bioRxiv, 2024
Grid-cell-like codes operate in non-spatial conceptual spaces.
- Paper: https://www.biorxiv.org/content/10.1101/2024.11.20.624569v1.full

**High-Dimensional Brain in High-Dimensional World** — 2020
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7516518/

**Thousand Brains Theory** — Numenta, 2024
- Paper: https://arxiv.org/abs/2412.18354
- Code: https://github.com/thousandbrainsproject/tbp.monty

---

## Bio-Inspired Computation

**Dendritic Computation** — *Nature Communications*, 2025
- Paper: https://www.nature.com/articles/s41467-025-56297-9

**ReSU (Rectified Spectral Units)** — AAAI 2026
- Paper: https://arxiv.org/abs/2512.23146

**KAN (Kolmogorov-Arnold Networks)** — ICLR 2025
- Paper: https://arxiv.org/abs/2404.19756

**SpikingLLM** — submitted ICLR 2026
- Paper: https://openreview.net/forum?id=r6fNn987rr

**Deep Oscillatory Neural Network** — *Scientific Reports*, 2025
- Paper: https://www.nature.com/articles/s41598-025-24837-4

**Predictive Coding** — *Nature Communications*, 2025
- Paper: https://www.nature.com/articles/s41467-025-64234-z

---

## Consensus Training Methods (for thinking models)

**DeepSeek-R1** — Jan 2025
Pure RL via GRPO with rule-based rewards. The "aha moment" emergence.
- Paper: https://arxiv.org/abs/2501.12948

**Let's Verify Step by Step (PRM800K)** — OpenAI, ICLR 2024
Process reward models for step-level reasoning.

**PonderNet** — DeepMind, 2021
Learned halting distribution.
- Paper: https://arxiv.org/abs/2107.05407

**Universal Transformers** — ICLR 2019
Weight-sharing recurrent transformer with ACT.
- Paper: https://arxiv.org/abs/1807.03819

**LoopFormer** — ICLR 2026
Shortcut consistency for variable-depth eval. **Strongest baseline this project has compared against.**
- Paper: https://arxiv.org/abs/2602.11451
- Code: https://github.com/armenjeddi/loopformer

**Mixture-of-Recursions** — NeurIPS 2025
Expert-choice routing for adaptive depth.
- Paper: https://arxiv.org/abs/2507.10524
- Code: https://github.com/raymin0223/mixture_of_recursions

---

## Quantization (Not Currently Relevant)

**TurboQuant** — Google DeepMind, ICLR 2026
KV cache quantization. 6× cache compression. Not applicable to small models.
- Paper: https://arxiv.org/abs/2504.19874

---

## Code Repositories Referenced

- **CODI:** https://github.com/zhenyi4/codi (the experiment will fork from here)
- **COCONUT:** https://github.com/facebookresearch/coconut
- **CoT2:** https://github.com/alperengozeten/CoT2
- **LLM-JEPA:** https://github.com/rbalestr-lab/llm-jepa
- **LeJEPA:** https://github.com/galilai-group/lejepa
- **HELM:** https://github.com/Graph-and-Geometric-Learning/helm
- **LoopFormer:** https://github.com/armenjeddi/loopformer
- **GATr:** https://github.com/Qualcomm-AI-research/geometric-algebra-transformer
- **Soft TPR:** https://github.com/gomb0c/soft_tpr
- **BLT:** https://github.com/facebookresearch/blt
- **MBLM:** https://github.com/ai4sd/multiscale-byte-lm
- **EvaByte:** https://github.com/OpenEvaByte/evabyte
- **Bolmo:** https://github.com/allenai/bolmo-core
- **Mamba:** https://github.com/state-spaces/mamba
- **Gated DeltaNet:** https://github.com/NVlabs/GatedDeltaNet
- **ContextLM:** https://github.com/LUMIA-Group/ContextLM
- **OpenCoconut (community reimplementation):** https://github.com/casper-hansen/OpenCoconut
