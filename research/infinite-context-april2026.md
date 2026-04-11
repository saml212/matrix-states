# Infinite Context & Long-Context SOTA (April 2026)

## The Paper: EverMind MSA (Memory Sparse Attention)
- **arxiv:** 2603.23516 (March 18, 2026)
- **Lab:** EverMind-AI (Chinese-founded, Shanda Group)
- **Code:** github.com/EverMind-AI/MSA (open source)
- **Result:** 100M token context on 2× A800 GPUs, <9% degradation from 16K→100M
- **Mechanism:**
  1. Content-based sparse routing (select relevant memory subsets, near-linear complexity)
  2. Document-wise RoPE (positions reset per document, 4K training → 100M inference)
  3. KV cache compression + tiered GPU/CPU storage
  4. Memory interleave (iterative retrieval for multi-hop reasoning)

## Long-Context Landscape (2025-2026)

### Production-Ready
| System | Lab | Context | Key Mechanism |
|--------|-----|---------|---------------|
| **NSA** | DeepSeek | 163K (V3.2), 1M target (V4) | Compress+select+local branches. Best Paper ACL 2025 |
| **MoBA** | Moonshot/Kimi | 1M+ | MoE-style block selection. In production. |
| **Lightning Attn** | MiniMax | 1M native, 4M extrapolated | Linear attention. 456B MoE model. |
| **MSA** | EverMind | 100M | Memory sparse routing. Open source. |

### Research Frontier
| System | Lab | Context | Key Mechanism |
|--------|-----|---------|---------------|
| **Titans/MIRAS** | Google | 2M+ | Surprise-gated neural memory |
| **TTT-E2E** | Stanford | Unlimited | Constant latency regardless of context |
| **InfSA** | — | Infinite | Spectral attention as diffusion on token graph |
| **Infini-attention** | Google | Infinite | Compressive memory + local attention |

## What Works for Byte-Level Long Context

| Approach | Mechanism | Byte Suitability |
|----------|-----------|-----------------|
| **MBLM hierarchy** | Mamba global + Transformer local, patches of 16-25 bytes | Best proven approach. 5M bytes on single A100. |
| **BLT patching** | Entropy-based dynamic patches, 4-6× compression | Cancels byte overhead. Text-focused. |
| **MambaByte** | Pure Mamba SSM, O(1) memory | 3.5× more tokens is painless with O(1) memory. |
| **EvaByte** | Chunked linear attention + multi-byte prediction | 2× faster decoding than tokenized. |

## The Architecture for Our Project

**2-stage hierarchy matching our matrix dimension:**
1. **Global:** Mamba/linear attention over 16-byte patch representations
   - Each patch = 16 bytes → encoded as 16×16 matrix (our outer-product embedding)
   - Mamba processes patch sequence in O(L) not O(L²)
   - 196K byte image = 12K patches — manageable
2. **Local:** Small transformer for byte prediction within each 16-byte patch
   - Standard attention at L=16 is trivial
   - Matrix thinking happens here (iterative refinement within patch)

**Why this matches our architecture:**
- 16-byte patch = 16 entries, one per row of the 16×16 matrix
- The matrix column dimension encodes cross-byte relationships WITHIN the patch
- Global Mamba handles cross-patch (long-range) dependencies
- Local matrix thinking handles within-patch (structural) dependencies
- Patch size matches matrix dimension — not coincidence, design choice

## Key Connections to Our Work

**Titans (Google NeurIPS 2025):** Surprise-gated MLP memory is conceptually similar to
our iterative matrix thinking. MIRAS framework: Memory=matrix, Attention=Frobenius,
Retention=thinking dynamics. Our matrix tokens could formalize as Titans-style memory.

**Gated DeltaNet / Kimi Linear:** Matrix memories with surgical delta-rule updates.
Similar to our (I+Δ)·M·(I+Γ) multiplicative composition.

## Sources
- EverMind MSA: arxiv.org/abs/2603.23516, github.com/EverMind-AI/MSA
- DeepSeek NSA: arxiv.org/abs/2502.11089 (ACL 2025 Best Paper)
- MoBA: arxiv.org/abs/2502.13189, github.com/MoonshotAI/MoBA
- MiniMax-01: arxiv.org/abs/2501.08313
- Titans: arxiv.org/abs/2501.00663
- MBLM: arxiv.org/abs/2502.14553
- BLT: arxiv.org/abs/2412.09871
- MambaByte: arxiv.org/abs/2401.13660
- InfSA: arxiv.org/abs/2603.00175
