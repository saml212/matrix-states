# Byte-Level Multi-Domain Models — SOTA Research (April 2026)

## The Critical Finding: bGPT Cross-Domain Transfer

bGPT (2024) trained on text + images + audio from raw bytes. Results:
- **Negative transfer between text and non-text.** "Distinct byte-level organizational patterns."
- **Positive transfer between audio and images.** Visual/signal domains share byte patterns.
- **Mixed training dilutes domain-specific understanding.** bGPT_mix worse than single-domain.

This is the gap our architecture could fill: if matrix structure can bridge text and non-text
byte patterns, that would be a genuine contribution.

## The Spatial Locality Problem

Flattening a 2D image to 1D bytes loses spatial structure:
- Pixel (0,0) and (0,1) are 3 bytes apart
- Pixel (0,0) and (1,0) are 96 bytes apart (for 32×32 RGB)
- The model must learn that bytes 96 positions apart are spatially adjacent

Current approaches:
1. Do nothing (bGPT, MBLM) — works poorly for images
2. Fourier position concatenation (Perceiver IO) — works well but needs to know modality
3. Compressed file formats (ByteFormer) — JPEG groups spatial info
4. Space-filling curves (Hilbert) — preserves 2D locality in 1D
5. Higher-Order Transformers (HOT) — Kronecker attention, don't flatten at all

**Matrix opportunity:** Row = one spatial axis, column = another. The matrix structure
could naturally encode 2D layout without special position encodings.

## What Exists vs Our Gap

| Model | Bytes | Multi-Domain | Structured Repr | Cross-Domain Transfer |
|-------|-------|-------------|-----------------|----------------------|
| bGPT | Yes | Yes (4 domains) | No (flat vectors) | Negative text↔other |
| MBLM | Yes | Yes (text+images) | No | 52% CLEVR VQA |
| Perceiver IO | Yes | Yes (native) | Latent array | Good with Fourier pos |
| EvaByte | Yes | Preliminary | No | Not measured |
| CliffordNet | No | No (vision only) | Yes (multivectors) | N/A |
| HOT | No | Yes (tensor data) | Yes (Kronecker) | N/A |
| **Ours** | **Yes** | **Planned** | **Yes (matrices)** | **Untested** |

Nobody combines: byte-level input + matrix representations + multi-domain + cross-domain transfer.

## Key References
- bGPT: arxiv.org/abs/2402.19155 (cross-domain byte model)
- MBLM: arxiv.org/abs/2502.14553 (byte-level VQA)
- Perceiver IO: arxiv.org/abs/2107.14795 (multi-modal from raw input)
- HOT: arxiv.org/abs/2412.02919 (Kronecker attention for tensor data)
- CliffordNet: arxiv.org/abs/2601.06793 (structured algebra, vision)
- ByteFormer: arxiv.org/abs/2306.00238 (file bytes classification)
- Awesome-Byte-LLM: github.com/zjysteven/Awesome-Byte-LLM

## Cross-Domain Transfer Measurement
- Transfer coefficient: Delta_D = BPB_single(D) - BPB_multi(D)
- Domain classification probe on frozen representations
- Gradient cosine similarity between domain batches
- CKA for representation geometry comparison
- Held-out domain generalization
