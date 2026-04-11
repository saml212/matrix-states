# Long Context for Byte-Level Models: State of the Art (April 2026)

Research sweep conducted 2026-04-01. Focus: achieving 100K-1M byte context efficiently,
with special attention to architectures relevant to matrix-valued token representations.

---

## Table of Contents

1. [The Chinese Lab: EverMind-AI and Memory Sparse Attention](#1-evermind-ai-memory-sparse-attention)
2. [DeepSeek Sparse Attention Family (NSA + DSA)](#2-deepseek-sparse-attention)
3. [Infini-Attention (Google)](#3-infini-attention)
4. [Ring Attention (UC Berkeley)](#4-ring-attention)
5. [MInference (Microsoft)](#5-minference)
6. [StreamingLLM (MIT)](#6-streamingllm)
7. [Titans + MIRAS (Google)](#7-titans-and-miras)
8. [TTT Layers (Test-Time Training)](#8-ttt-layers)
9. [Gated DeltaNet + Kimi Linear](#9-gated-deltanet-and-kimi-linear)
10. [Byte-Level Architectures](#10-byte-level-architectures)
11. [Sparse Attention Patterns Across Modalities](#11-sparse-attention-patterns)
12. [Practical Recommendations for Our Project](#12-recommendations)

---

## 1. EverMind-AI: Memory Sparse Attention (MSA)

**This is the Chinese lab you were remembering.**

- **Paper:** "Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens"
  arXiv: 2603.23516 (March 2026)
- **Lab:** EverMind-AI (San Mateo, CA / China-linked)
- **Code:** https://github.com/EverMind-AI/MSA (MIT license, code "coming soon")
- **Claim:** 100 million token context, end-to-end trainable

### Architecture

Three innovations stacked together:

1. **Memory-Sparse Attention Layer:** Documents are chunk-mean-pooled for compression.
   A router projector selects top-k relevant documents via cosine similarity, then
   concatenates their compressed KV pairs with query context. Uses document-wise
   Rotary Position Embeddings (RoPE) -- each document resets position indices, preventing
   drift between training context (64K tokens) and inference context (100M tokens).
   Near-linear complexity.

2. **KV Cache Compression + Memory Parallel:** Tiered storage -- GPU-resident routing
   keys with CPU-stored content. Asynchronous fetching enables 100M-token inference
   on two A800 GPUs.

3. **Memory Interleave:** Adaptive multi-round "retrieval -> expansion -> generation"
   cycles for multi-hop reasoning across documents.

### Key Results

- RULER NIAH test: 94.84% accuracy at 1M tokens, <9% degradation from 16K to 100M
- Unmodified baselines collapse to 24.69% at 1M tokens
- +16% over standard RAG, +11.5% over RAG+reranker on 9 QA datasets
- Training: 158.95B tokens continuous pretraining with auxiliary routing loss,
  two-stage SFT using 8K -> 64K curriculum

### Relevance to Our Project

MSA treats memory as a pluggable, content-addressable layer. This is conceptually
close to matrix-valued tokens serving as memory: both use structured representations
that can be queried and updated. The document-wise RoPE trick for context extrapolation
is directly applicable to byte-level models where position indices grow 3.5x faster.

### Related: MemOS (EverMind's broader project)

EverMind also released MemOS -- a "memory operating system" for LLMs with three-layer
architecture (Interface / Operation / Infrastructure). Introduces MemCube abstraction
unifying plaintext, activation, and parameter memories. Separate from MSA but shows
their broader vision of memory as a first-class system resource.

---

## 2. DeepSeek Sparse Attention Family

### Native Sparse Attention (NSA) -- February 2025

- **Paper:** arXiv:2502.11089 (won Best Paper at ACL 2025)
- **Lab:** DeepSeek

Three parallel attention branches per head:
1. **Compressed attention** -- coarse-grained token compression for global patterns
2. **Selected attention** -- fine-grained important token block selection
3. **Sliding attention** -- local context window

All three are hardware-aligned with optimized CUDA kernels. End-to-end trainable.
Substantial speedups on 64K sequences across decoding, forward, and backward passes.

### DeepSeek Sparse Attention (DSA) -- DeepSeek-V3.2 (2025)

- **Paper:** arXiv:2512.02556
- **Architecture:** Lightweight "lightning indexer" at every layer scores all preceding
  tokens, selects small batch for core attention
- **Complexity:** O(L*k) where k << L (linear in sequence length)
- **Result:** 2x per-token GPU cost reduction on long sequences
- **Built on MLA** (Multi-Head Latent Attention) -- compresses KV projections to
  lower-dimensional space before attention, cutting memory bandwidth 40-50%

### IndexCache (March 2026, Tsinghua + Z.ai)

Optimization for DSA-style models: cuts 75% of redundant computation, 1.82x faster
time-to-first-token, 1.48x faster generation at 200K context.

### Relevance

The hierarchical sparse pattern (compress + select + local) is the most practical
approach for our byte-level work. At 3.5x more tokens, even DSA's linear O(L*k)
matters -- k needs to stay small. The lightning indexer concept could work with
matrix-valued tokens: route based on matrix similarity (Frobenius inner product
or trace-based metric) rather than vector dot product.

---

## 3. Infini-Attention (Google, April 2024)

- **Paper:** arXiv:2404.07143 ("Leave No Context Behind")
- **Lab:** Google

### Architecture

Combines in a single Transformer block:
- **Masked local attention** -- standard attention within current segment
- **Compressive memory** -- earlier segments compressed into fixed-size buffer
- **Linear attention** -- long-term retrieval from compressive memory

Sequence is divided into segments. Each segment compresses its KV into the memory
buffer, and the next segment retrieves from it while attending locally.

### Results

- 1M passkey retrieval with 1B model (trained on only 5K token examples)
- SOTA on 500K book summarization with 8B model
- Bounded memory regardless of context length

### Reality Check

A detailed HuggingFace blog post ("A failed experiment: Infini-Attention") found
that community reproductions struggled. The compressive memory appears lossy in
practice -- fine for needle-in-haystack but degrades on tasks requiring precise
recall of specific content from distant context.

### Relevance

The segment-and-compress paradigm is directly applicable to bytes. Process bytes
in 4K-8K chunks, compress each chunk's representation into a fixed matrix memory,
then let subsequent chunks query the accumulated memory. Matrix-valued tokens could
serve as the compressed representation naturally -- a 16x16 matrix holds 256 values,
far richer than a single vector for summarizing a segment.

---

## 4. Ring Attention (UC Berkeley, ICLR 2024)

- **Paper:** arXiv:2310.01889
- **Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel

### How It Works

Devices form a conceptual ring. Each device:
1. Holds its block of the sequence (Q, K, V)
2. Computes blockwise attention on local KV
3. Sends its KV block to the next device in the ring
4. Receives KV from the previous device
5. Accumulates attention scores across all received blocks

As long as block computation time > block transfer time, communication is fully
overlapped. Context length scales linearly with device count.

### Extensions

- **Striped Attention** (2023) -- fixes workload imbalance in causal attention,
  1.45x throughput improvement at 256K
- **RingX** (2025) -- HPC-optimized variant
- **Mesh-Attention** (2025) -- improved communication patterns for data locality

### Relevance

Ring Attention is a distributed strategy, not an architecture. It works with any
attention mechanism. For our eventual H100 scale-up, Ring Attention would let us
distribute 1M byte context across 8 GPUs (125K bytes per GPU). Combined with sparse
attention within each block, this is the most straightforward path to million-byte context.

---

## 5. MInference (Microsoft, NeurIPS 2024 Spotlight)

- **Paper:** NeurIPS 2024
- **Code:** https://github.com/microsoft/MInference

### Three Discovered Sparse Patterns

By analyzing attention matrices of long-context LLMs, Microsoft found three patterns:
1. **A-shape** -- attention to initial tokens + diagonal (combining sink + local)
2. **Vertical-Slash** -- attention to specific key positions across all queries
3. **Block-Sparse** -- clustered attention in blocks

### How It Works

1. Offline: determine which sparse pattern each head belongs to
2. Online: approximate the sparse index dynamically
3. Compute attention with optimal custom kernels per pattern

Drop-in replacement -- no retraining needed.

### Results

- 10x latency reduction on A100 for pre-filling
- Works with LLaMA-3-1M, GLM4-1M, Yi-200K, Phi-3-128K, Qwen2-128K
- Accuracy maintained on InfinitiBench, RULER, PG-19, NIAH

### Extensions

- **MMInference** (ICLR 2025, ICML 2025) -- multimodal extension for VLMs,
  8.3x speedup at 1M tokens with modality-aware permutation sparse attention

### Relevance

MInference's pattern discovery is empirical -- derived from BPE-tokenized text models.
Byte-level models may exhibit different patterns due to finer granularity. However,
the A-shape pattern (attention sinks + sliding window) is likely universal across
tokenization schemes. The key question: do byte models develop more block-sparse
patterns because local byte groups form implicit "tokens"?

---

## 6. StreamingLLM (MIT Han Lab, ICLR 2024)

- **Paper:** arXiv:2309.17453
- **Code:** https://github.com/mit-han-lab/streaming-llm

### Key Discovery: Attention Sinks

Initial tokens in the sequence receive disproportionately high attention scores
even when semantically irrelevant. This is because softmax attention must sum to 1,
so initial tokens become "sinks" for residual attention mass.

### Architecture

Keep KV cache of:
1. First few tokens (attention sinks) -- typically 4 tokens
2. Sliding window of recent tokens -- typically 1024-4096

Everything in between is evicted.

### Results

- Stable language modeling with 4M+ tokens
- 22.2x speedup over sliding window recomputation
- Works with Llama-2, MPT, Falcon, Pythia without fine-tuning
- Adding a dedicated sink token during pre-training further improves performance

### Relevance

For byte-level streaming inference, StreamingLLM is directly applicable but the
window size needs to be larger (3.5x) to capture the same semantic content. A 4K
token window in BPE ~ 14K byte window. The attention sink phenomenon likely holds
for byte models too, but needs verification.

---

## 7. Titans + MIRAS (Google, NeurIPS 2025)

- **Paper:** arXiv:2501.00663 (Titans)
- **Blog:** https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/

### Titans: Neural Long-Term Memory

**Core idea:** A deep neural network (MLP) serves as long-term memory, updated via
a "surprise" metric during forward pass.

**Surprise metric:** The gradient of the loss w.r.t. memory parameters. Large gradients
= surprising (unexpected) tokens that should be memorized. Small gradients = expected
tokens that can be ignored. This is inspired by human memory: unexpected events are
more memorable.

**Two refinements:**
- **Momentum** -- balances "momentary surprise" (current token) with "past surprise"
  (recent context flow), ensuring related follow-up tokens are also captured
- **Adaptive weight decay** -- forgetting gate that discards stale information

### Three Architecture Variants

1. **MAC (Memory as Context):** Memory output is concatenated with attention context
2. **MAG (Memory as Gate):** Memory gates the attention output
3. **MAL (Memory as Layer):** Memory replaces some attention layers entirely

### MIRAS: The Unifying Framework

MIRAS (Matrix Inner Retrieval Associative memory) views most modern sequence models
as instances of online optimization over associative memory. Four design axes:
1. **Memory architecture** -- vector, matrix, or deep MLP
2. **Attentional bias** -- the learning objective (MSE, Lp norms, Huber loss)
3. **Retention gate** -- regularization/forgetting (weight decay, KL divergence, elastic net)
4. **Memory algorithm** -- the optimization process

From this framework, three new attention-free models:
- **Moneta** -- 2-layer MLP memory + Lp attentional bias
- **Yaad** -- MLP memory + Huber loss (robust to outliers)
- **Memora** -- probability simplex memory + KL divergence retention

### Results

- Outperforms Mamba-2 and Gated DeltaNet on language modeling (C4, WikiText)
- Outperforms GPT-4 on extreme long-context tasks (BABILong) despite far fewer parameters
- Scales to 2M+ token context windows

### **Critical Relevance to Our Project**

This is the most directly relevant work to our matrix-valued token architecture:

- **Matrix tokens as memory:** Our 16x16 matrix tokens are essentially small memory
  matrices. Titans shows that matrix/MLP memories updated by surprise work. Our
  "iterative thinking" steps are analogous to memory updates.

- **Surprise-gated updates:** Our rank enrichment during thinking (experiment 15-18)
  is conceptually similar -- the model builds richer representations when input is
  "surprising" (high-entropy regions). We could formalize this with an explicit
  surprise metric.

- **MIRAS framework:** Our matrix-valued tokens fit naturally into the MIRAS design
  space. Memory architecture = matrix. Attentional bias = Frobenius attention (our
  current approach). Retention gate = our thinking step dynamics. This gives us a
  theoretical framework to justify our design choices.

- **MAL variant:** Using memory as a layer (replacing attention entirely) is closest
  to what we're doing with matrix thinking -- the matrices ARE the computation, not
  just the state.

---

## 8. TTT Layers (Test-Time Training)

### TTT-Linear / TTT-MLP (July 2024)

- **Paper:** arXiv:2407.04620

Hidden state = a model (linear layer or MLP) that is updated by gradient descent
on each input token at test time. The "hidden state" learns to predict the current
sequence, effectively treating context as training data.

- TTT-Linear: hidden state is a linear model. Fast but limited.
- TTT-MLP: hidden state is a 2-layer MLP. More expressive but memory I/O challenges.

### TTT Done Right (May 2025)

Addresses gradient magnitude explosion in vanilla TTT via large-chunk updates
(accumulating gradients over chunks before applying). More stable and efficient.

### TTT-E2E (End-to-End, December 2025)

- **Paper:** arXiv:2512.23675
- Meta-learns the initialization for test-time learning during training
- Constant inference latency regardless of context length
- 2.7x speedup over full attention at 128K context

### Relevance

TTT is conceptually close to our iterative thinking: both update internal
representations based on input. The difference: TTT uses gradient descent to
update weights, while our thinking steps use forward-pass matrix operations.
TTT-E2E's constant-latency property is exactly what byte models need -- context
length independence is critical when bytes are 3.5x more numerous.

---

## 9. Gated DeltaNet + Kimi Linear

### Gated DeltaNet (ICLR 2025)

- **Paper:** arXiv:2412.06464
- **Lab:** NVIDIA
- **Code:** https://github.com/NVlabs/GatedDeltaNet

Combines:
- **Gating** (from Mamba-2) -- adaptive memory control, rapid erasure
- **Delta rule** (from DeltaNet) -- precise, targeted memory updates

The two are complementary: gating handles "what to forget," delta rule handles
"what to write precisely." Uses L2-normalized queries and keys for stability.

### Kimi Linear (Moonshot AI, October 2025)

- **Paper:** arXiv:2510.26692
- **Code:** https://github.com/MoonshotAI/Kimi-Linear

**Kimi Delta Attention (KDA):** Extends Gated DeltaNet with finer-grained gating
via DPLR (Diagonal-Plus-Low-Rank) transition matrices. The hybrid architecture
uses 3:1 ratio of KDA-to-global-MLA (Multi-Head Latent Attention).

**Results:**
- First linear attention hybrid to outperform full attention under fair comparison
- 75% KV cache reduction
- 6x decoding throughput at 1M context
- 48B total params, 3B activated (MoE)

### Relevance

The KDA approach -- linear attention with delta-rule updates -- is essentially
maintaining a matrix memory that gets surgically updated. This is strikingly
similar to matrix-valued tokens being iteratively refined. The 3:1 ratio of
linear-to-full attention is a practical hybrid that byte models should adopt:
most layers use cheap linear attention, with occasional full attention layers
for precise retrieval.

---

## 10. Byte-Level Architectures

### 10a. MambaByte (COLM 2024)

- **Paper:** arXiv:2401.13660
- **Lab:** Cornell

Straightforward adaptation of Mamba SSM to byte-level. O(1) memory, O(n) compute.
Competitive with subword Transformers. 2.6x inference speedup via speculative
decoding with tokenized drafter.

**Key insight:** Mamba's fixed-size state handles the 3.5x token increase naturally.
No quadratic penalty.

### 10b. Byte Latent Transformer / BLT (Meta, ACL 2025)

- **Paper:** arXiv:2412.09871
- **Code:** https://github.com/facebookresearch/blt

**Entropy-based patching:** Groups bytes into variable-length patches based on
next-byte entropy. High entropy = hard to predict = patch boundary.

Three modules:
1. **Local Encoder** -- lightweight, encodes bytes into patch representations
   via cross-attention + n-gram hash embeddings
2. **Latent Transformer** -- heavy, processes patches (much shorter sequence)
3. **Local Decoder** -- lightweight, decodes patches back to bytes

**Results:**
- Matches Llama 3 at training FLOP parity
- 50% fewer inference FLOPs
- New scaling dimension: simultaneously grow patch and model size
- 1B and 7B models released on HuggingFace

**Key insight for us:** Entropy patching reduces effective sequence length by ~4-6x
for text, nearly canceling the 3.5x byte overhead. For images/audio, entropy patterns
differ -- smooth regions compress more, edges/transients less.

### 10c. MBLM (IBM, ICML 2025)

- **Paper:** arXiv:2502.14553
- **Code:** https://github.com/ai4sd/multiscale-byte-lm (pip install mblm)

**Hierarchical decoder stack:** N stages, first N-1 are "global" (inter-patch),
last is "local" (intra-patch byte prediction).

**Best configuration:** Global Mamba + Local Transformer
- 3-stage hierarchy with patch sizes (1000, 200, 25)
- 350M params, 5M byte context on single A100 80GB
- Training: 100GB UTF-8 in ~15 hours
- Gradient checkpointing enables memory/compute tradeoff

**Benchmark results (PG19):**
| Config | Context | BPB |
|--------|---------|-----|
| MegaByte (2D Transformer) | 98K | 1.370 |
| MBLM (Mamba+Transformer) | 98K | 1.240 |
| MBLM (Mamba+Mamba) | 98K | 1.164 |
| MBLM (3D Hybrid) | 1M+ | 2.089 |

**Multimodal:** Processes raw RGB pixels + text as single bytestream.
On CLEVR VQA: 52.1% accuracy matching CNN+LSTM baseline (52.3%) with
NO image encoder, NO classification head -- pure next-byte prediction.
Cross-domain transfer: text pre-training IMPROVED VQA performance.

### 10d. EvaByte (HKU + SambaNova, January 2025)

- **Code:** https://github.com/OpenEvaByte/evabyte
- **Weights:** EvaByte/EvaByte on HuggingFace

6.5B byte-level LM with two innovations:
1. **EVA attention** -- linearized attention distributed across multiple local memory
   slots (chunks of KV pairs get separate linearized attention, not one global state).
   Linear complexity, but richer than pure linear attention.
2. **Multibyte prediction** -- 8 output heads predict 8 consecutive bytes. At inference,
   enables self-speculative decoding via tree attention. 5-10x faster decoding than
   vanilla byte architectures, up to 2x faster than tokenized models.

Trained on 1.5T bytes (5x less data than comparable tokenized models).
Matches SOTA tokenized models despite the data disadvantage.

### 10e. Bolmo (AI2, December 2025)

- **Paper:** arXiv:2512.15586
- **Code:** https://github.com/allenai/bolmo-core

**Byteification of existing models:** Takes OLMo 3 (tokenized) and retrofits it
to byte-level with:
1. mLSTM local encoder for byte-to-patch encoding
2. Non-causal boundary predictor (uses future context, like tokenizers do)
3. Original OLMo 3 transformer processes patches
4. Local decoder back to bytes

**Two-stage training:**
- Stage 1: Freeze transformer, train encoder/decoder/predictor (9.8B tokens / 43B bytes)
- Stage 2: Unfreeze everything, train 39.3B tokens (173B bytes)
- Total: ~1% of original training cost

**Results:** +16.5% absolute improvement in STEM over BLT 7B.

### 10f. bGPT (Microsoft Research Asia, February 2024)

- **Paper:** arXiv:2402.19155
- **Code:** https://github.com/sanderwood/bgpt

Hierarchical Transformer for binary data. Can process ANY file type (text, audio,
images, executables, encrypted files). 0.0011 BPB on ABC-to-MIDI conversion,
>99.99% accuracy simulating CPU operations.

### 10g. MEGABYTE (Meta, NeurIPS 2023)

- **Paper:** arXiv:2305.07185

The original hierarchical byte model. Global Transformer over patches + Local
Transformer within patches. Inspired BLT and MBLM.

---

## 11. Sparse Attention Patterns Across Modalities

### The Question: Can ONE sparse pattern work for text, images, and audio bytes?

**Short answer: No, but a small adaptive set can.**

### Text Patterns (well-studied)

- **Sliding window** -- local context (5-15 byte positions)
- **Attention sinks** -- first few positions get high attention
- **Vertical slashes** -- certain key positions attended by all queries
- These compose into MInference's A-shape / Vertical-Slash / Block-Sparse trichotomy

### Image Patterns (emerging)

- **2D locality** -- pixels attend to spatial neighbors (not sequential neighbors)
- **Block-sparse** -- aligned with spatial regions
- For byte-serialized images: need to map 1D byte position back to 2D spatial position
  (stride-aware attention mask)
- MMInference (Microsoft, 2025) shows modality-aware permutation helps -- reorder
  image bytes to group spatially related bytes, then use standard block-sparse

### Audio Patterns (less studied for bytes)

- **Dilated/strided attention** -- captures periodic temporal patterns (pitch, rhythm)
- **Multi-resolution** -- fine attention for transients, coarse for sustained tones
- Similar to text sliding window but with larger strides

### Unified Approaches

- **SpargeAttn** (ICLR 2025) -- training-free sparse attention that works across
  language, image generation, and video. Uses runtime sparsity detection.
- **Content-adaptive routing** -- let the model learn which pattern to use per-head
  (cf. Mixture of Sparse Attention, 2025)
- **Hierarchical patching** -- BLT/MBLM approach naturally handles modality differences
  because patch boundaries adapt to data entropy

### Recommendation for Matrix Thinking

For our multi-domain byte model, the practical approach:
1. **Sliding window** for all modalities (cheap, universal baseline)
2. **Global tokens / sinks** -- first N positions always attended
3. **Learned sparse routing** per head -- let heads specialize (some do local text
   attention, some do 2D image attention, some do dilated audio attention)
4. **Hierarchical compression** -- matrix-valued patch representations reduce sequence
   length at global levels, where full attention is affordable

---

## 12. Practical Recommendations for Our Project

### The Byte-Level Long Context Problem

Our situation: 100K bytes of text = ~27K BPE tokens. 100K bytes of image = one small
image. We need to handle at least 100K bytes efficiently, ideally 1M.

Standard O(L^2) attention at L=100K bytes = 10 billion operations per layer per head.
At L=1M bytes = 1 trillion. Impossible without sparse or linear methods.

### Recommended Architecture Stack (Ranked by Practicality)

#### Tier 1: Use Now (M4 Mac Mini scale)

1. **Hierarchical patching (BLT/MBLM style)**
   - Group bytes into patches of 8-32 bytes
   - Local model processes within patches (Transformer, cheap at L=8-32)
   - Global model processes between patches (sequence length / patch_size)
   - Reduces effective sequence by 8-32x
   - 100K bytes -> 3K-12K patches at global level -> tractable
   - **Our matrix tokens ARE patches.** A 16x16 matrix = 256 values = natural
     representation of a 16-byte patch (16 bytes x 16 embedding dims)

2. **Sliding window + attention sinks**
   - Simple, proven, works at byte level
   - Window of 2048-4096 bytes + 4-8 sink positions
   - Combine with hierarchical patching: local model uses sliding window,
     global model can use full attention on reduced sequence

3. **Mamba/SSM for global stages**
   - MBLM proved Mamba for global + Transformer for local is optimal
   - O(n) compute, O(1) memory for global context
   - Our M4 Mac can handle this at 15M param scale

#### Tier 2: Use for Scale-Up (Mac Studio / H100)

4. **DeepSeek-style hierarchical sparse attention**
   - Compress + Select + Local (three parallel branches)
   - Most mature production-grade sparse attention
   - Needs custom CUDA kernels (not MPS-friendly)

5. **Kimi Linear / Gated DeltaNet hybrid**
   - 3:1 ratio of linear attention to full attention layers
   - 75% KV cache reduction, 6x decoding throughput
   - Good library support (Flash Linear Attention)

6. **Ring Attention for multi-GPU distribution**
   - 8 H100s: 1M bytes / 8 = 125K bytes per GPU
   - Combine with sparse attention within each GPU's block

#### Tier 3: Research Frontier (Experimental)

7. **Matrix-valued Titans memory**
   - Our matrix tokens + surprise-gated updates
   - MIRAS framework gives theoretical grounding
   - Use matrix Frobenius norm of gradient as surprise metric
   - Novel: no one has combined matrix token representations with Titans-style memory

8. **TTT-style test-time learning**
   - Matrix tokens as fast weights updated per-input
   - Constant latency regardless of context length
   - Needs careful stability engineering

9. **EverMind MSA-style pluggable memory**
   - Separate memory layer that can be mixed with any backbone
   - Document-wise RoPE for extreme extrapolation

### Concrete Next Steps

For the immediate next experiment on M4 Mac Mini:

1. **Implement 2-stage MBLM-style hierarchy with matrix tokens:**
   - Stage 1 (global): Mamba-2 or linear attention over matrix-valued patch representations
   - Stage 2 (local): Small Transformer predicting bytes within each patch
   - Patch size = 16 (matches our 16x16 matrix dimension)
   - Context: 16K bytes = 1K patches at global level

2. **Add Titans-style surprise gating to thinking steps:**
   - Compute gradient magnitude of matrix memory w.r.t. prediction loss
   - High surprise -> more thinking iterations
   - Low surprise -> skip/reduce thinking
   - This formalizes our observed rank enrichment phenomenon

3. **Sliding window + sinks at local level:**
   - Window = 128 bytes, 4 sink positions
   - Cheap and proven

4. **Benchmark on both text and image bytes:**
   - PG19 text (standard benchmark for byte models)
   - CIFAR-10 raw pixels (we already have data)
   - Same model, same weights, different data

---

## Key Papers Reference List

| Paper | Lab | Year | Key Innovation | arXiv/Venue |
|-------|-----|------|----------------|-------------|
| Memory Sparse Attention | EverMind-AI | 2026 | 100M token context, doc-wise RoPE | 2603.23516 |
| Native Sparse Attention | DeepSeek | 2025 | Hardware-aligned hierarchical sparse | 2502.11089 (ACL Best Paper) |
| DeepSeek Sparse Attention | DeepSeek | 2025 | Lightning indexer, O(Lk) complexity | 2512.02556 |
| Infini-Attention | Google | 2024 | Compressive memory + local attention | 2404.07143 |
| Ring Attention | UC Berkeley | 2024 | Distributed context across GPU ring | 2310.01889 (ICLR) |
| MInference | Microsoft | 2024 | Three empirical sparse patterns | NeurIPS Spotlight |
| StreamingLLM | MIT | 2024 | Attention sinks for infinite gen | 2309.17453 (ICLR) |
| Titans | Google | 2025 | Surprise-gated neural memory | 2501.00663 (NeurIPS) |
| MIRAS | Google | 2025 | Unifying framework for sequence models | NeurIPS 2025 |
| TTT-Linear/MLP | UC San Diego | 2024 | Weight matrices as hidden state | 2407.04620 |
| TTT-E2E | UC San Diego | 2025 | Meta-learned test-time training | 2512.23675 |
| Gated DeltaNet | NVIDIA | 2025 | Gating + delta rule for memory | 2412.06464 (ICLR) |
| Kimi Linear | Moonshot AI | 2025 | Linear attention beats full attention | 2510.26692 |
| BLT | Meta | 2024 | Entropy-based byte patching | 2412.09871 (ACL) |
| MBLM | IBM | 2025 | Hierarchical Mamba+Transformer bytes | 2502.14553 (ICML) |
| MambaByte | Cornell | 2024 | SSM for byte-level modeling | 2401.13660 (COLM) |
| EvaByte | HKU + SambaNova | 2025 | EVA attention + multibyte prediction | OpenEvaByte/evabyte |
| Bolmo | AI2 | 2025 | Byteification of existing models | 2512.15586 |
| bGPT | Microsoft Asia | 2024 | Binary data simulation | 2402.19155 |
| MEGABYTE | Meta | 2023 | Original hierarchical byte model | 2305.07185 (NeurIPS) |

---

## Code Repositories

- **MSA:** https://github.com/EverMind-AI/MSA (MIT, code coming soon)
- **MInference:** https://github.com/microsoft/MInference
- **StreamingLLM:** https://github.com/mit-han-lab/streaming-llm
- **Gated DeltaNet:** https://github.com/NVlabs/GatedDeltaNet
- **Kimi Linear:** https://github.com/MoonshotAI/Kimi-Linear
- **Flash Linear Attention:** https://github.com/fla-org/flash-linear-attention
- **BLT:** https://github.com/facebookresearch/blt
- **MBLM:** https://github.com/ai4sd/multiscale-byte-lm (pip install mblm)
- **EvaByte:** https://github.com/OpenEvaByte/evabyte
- **Bolmo:** https://github.com/allenai/bolmo-core
- **bGPT:** https://github.com/sanderwood/bgpt
- **Mamba:** https://github.com/state-spaces/mamba
- **Awesome Byte LLM:** https://github.com/zjysteven/Awesome-Byte-LLM
