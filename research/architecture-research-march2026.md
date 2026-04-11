# Architecture Research — March 2026

## Key Findings From Research Agent

### 1. Cheaper Matrix Operations
- **Monarch matrices** (Dao, ICML 2022): 16× FLOP reduction on 256-dim. Naturally decomposes into row/col operations on our 16×16 matrices. Code: github.com/HazyResearch/m2
- **BLAST** (NeurIPS 2024): Learns optimal mixture of block-diagonal + low-rank per layer. 40% complexity reduction on GPT-2. Code: github.com/changwoolee/BLAST
- **Spectral operations**: Modify top-k singular values via power iteration. O(d²k) vs O(d³). 4× cheaper with k=4.

### 2. Why LoopFormer Works (from the actual paper)
Three specific design choices:
1. Shortcut consistency: `L = L_full + 0.1*L_short + 0.1*||stopgrad(h_full) - h_short||²`
2. Time + step-size conditioning (AdaLN): sinusoidal embeddings of progress (0-1) and step size
3. Zero-initialized modulation so early training matches unmodulated behavior
Without shortcut consistency, representations stagnate (flat metrics across loops).

### 3. The Enrichment Finding is Novel
No paper explicitly studies rank INCREASE during iterative refinement. The literature focuses on preventing collapse, not encouraging enrichment. Seq-VCR (ICLR 2025) is closest — their variance-covariance regularization maintains representation diversity. Combined with pause tokens: 99.5% on 5×5 multiplication (GPT-4 CoT = 44%).

### 4. "Matrix IS the Prediction" is Novel
No prior work uses internal representation directly as logits with zero additional parameters. Weight tying still requires a matmul. Our idea (flatten 16×16 → 256 byte logits) appears genuinely novel.

### 5. Byte-Level Baselines at Small Scale
No major byte-level model reports results below 100M params. Our 1.67 BPB at ~5M params is actually reasonable for the scale. Extrapolating BLT scaling curves: ~2.0-2.5 BPB at 1M params is the realistic target.

### 6. Recommended Hybrid Architecture
```
byte → outer-product embed → 16×16 matrix
  → flatten → 2 standard attention layers (cheap, SDPA)
  → reshape → matrix thinking (Monarch-factored ops)
  → flatten → 2 standard attention layers
  → reshape → repeat T times
  → output: flatten matrix → 256 byte logits (zero params)
```
This mixes cheap vector attention with structured matrix thinking.

## Sources
- LoopFormer: arxiv 2602.11451, github.com/armenjeddi/loopformer
- Monarch: ICML 2022, github.com/HazyResearch/m2
- BLAST: NeurIPS 2024, github.com/changwoolee/BLAST
- CliffordNet: arxiv 2601.06793
- Seq-VCR: ICLR 2025, github.com/rarefin/seq_vcr
- Inner Thinking Transformer: ACL 2025, arxiv 2502.13842
- MonarchAttention: NeurIPS 2025, arxiv 2505.18698
- Awesome Byte-LLM: github.com/zjysteven/Awesome-Byte-LLM
