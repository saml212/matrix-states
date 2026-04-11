# Literature Check: Cross-Domain Transfer with Structured Representations

## Supports Our Hypothesis
- **MBLM (2025):** Text-to-vision byte transfer WORKS with hierarchical architecture (contradicts bGPT)
- **Tiny Time Mixers (NeurIPS 2024):** 1M param cross-domain transfer works in time series
- **CMSMs (ACL 2010):** Matrix composition more expressive than vector addition for language
- **TPRs (Smolensky++):** Tensor product representations improve compositional generalization
- **GATr (NeurIPS 2023):** Geometric algebra generalizes under domain shifts in 3D tasks
- **No one has tested matrix tokens for cross-domain byte transfer** — genuine gap

## Challenges Our Hypothesis
- No DIRECT evidence structured representations improve cross-domain transfer
- GATr's gains are domain-specific (3D geometry), not general
- ORCA (2024): fine-tuning drives cross-modal transfer, not representation alignment
- Tree Cross Attention: small bottlenecks lose too much information
- 61% of capabilities show irregular scaling — we might be below threshold

## Key bGPT Transfer Numbers
- Image→Image: +12% (strong positive)
- Audio→Audio: +3.6% (positive)
- Image→Audio: +3.9% (positive, cross-domain)
- Text→Image: +6.4% (positive but weaker)
- Image→Text: -1.2% (NEGATIVE)
- Text harms non-text more than non-text harms text

## MBLM Contradicts bGPT
MBLM (360M params, hierarchical Mamba+Transformer) shows text pre-training
POSITIVELY impacts vision VQA performance. The hierarchical multi-scale approach
overcomes the bottleneck that caused bGPT's negative transfer.

## Recommended Metrics for Small Scale
1. BPB improvement on domain B after pre-training on domain A vs from-scratch
2. Learning speed (steps to reach X% of from-scratch performance)
3. Rank dynamics across domains (our unique measurement capability)
4. Forward/backward transfer coefficients from continual learning

## Sources
- MBLM: arxiv.org/abs/2502.14553
- TTM: arxiv.org/abs/2401.03955
- CMSMs: aclanthology.org/P10-1093
- GATr: arxiv.org/abs/2305.18415
- bGPT: arxiv.org/abs/2402.19155
- ORCA: arxiv.org/abs/2403.13537
- Tree Cross Attention: ICLR 2024
