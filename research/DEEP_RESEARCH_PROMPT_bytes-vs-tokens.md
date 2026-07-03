# Deep-research prompt — Bytes vs. tokens for a matrix-native reasoning model

**How to use this:** paste everything under the horizontal rule into an external deep-research agent (OpenAI/Gemini/Perplexity deep research). Ask it to return a cited report. When it's done, save its output to `research/bytes-vs-tokens-EXTERNAL-gpt-june2026.md` in this repo so it sits next to the internal Sonnet-5 agent's write-up (`research/bytes-vs-tokens-matrix-native-june2026.md`) for side-by-side comparison.

---

## Research task: Is byte-level (tokenizer-free) input the right abstraction for scaling a matrix-native reasoning transformer, or should the tokenizer be held standard?

### Background you need

I am building a **matrix-native transformer**: each token is represented as a **d×d matrix** (not a vector). Attention scores are Frobenius inner products between matrix tokens; the MLP is `Z → ReLU(Z·W1)·W2` (multiplicative mixing along both matrix axes); the output head is a set of bilinear probes (`u_kᵀ Z v_k`) — crucially, **the matrix is never flattened to a vector in the forward pass**. The scientific hypothesis I am testing is that **matrix rank is a measurable and functionally-used observable of reasoning capacity** — i.e., the effective rank of a token/latent matrix tracks how many independent reasoning paths the model holds in superposition, and truncating rank below what a task needs should degrade accuracy.

Prior result: in a *bolted-on* setting (a matrix bottleneck inserted into a CODI-style continuous-chain-of-thought model distilled from a vector teacher), the model was **rank-blind** — rank-k truncation of the latent matrix did not change accuracy across many readout variants. I now want to test the hypothesis in a **from-scratch, fully matrix-native** model, scaled up to real reasoning data (math word problems; I currently have an OpenR1-Math corpus in GPT-2/BPE tokens).

The decision this research informs: **for the scaled-up model, should the input be byte-level / tokenizer-free (BLT/ByT5 style), or a standard BPE tokenizer?** My current lean is to hold the tokenizer *standard* (BPE) and treat byte-level as a separate, isolated follow-on ablation — because the rank hypothesis seems orthogonal to tokenization, because changing two architectural variables at once (matrix tokens AND byte input) would make any failure uninterpretable, and because byte sequences are much longer and slow iteration. I want this pressure-tested with evidence, including evidence that I'm wrong.

### Questions to answer (cite sources for each; prefer 2023–2026 work)

1. **State of tokenizer-free / byte-level LLMs (2023–2026).** Since Meta's BLT (Byte Latent Transformer with entropy patching) and the Chameleon→BLT pivot: do byte-level / tokenizer-free models match or beat standard BPE tokenization at *matched compute*, especially on reasoning and math benchmarks? Quantify the sequence-length and FLOP/compute overhead of byte-level. Cover at least: BLT, ByT5, CANINE, MegaByte, MambaByte, SpaceByte, EvaByte, H-Net / dynamic-chunking hierarchical models, and any 2025–2026 successors. What is the current consensus on whether tokenization is a bottleneck worth removing?

2. **Prior art: byte/character input combined with structured (non-vector) token representations.** Has anyone paired tokenizer-free/byte/char-level input with matrix-valued, bilinear, outer-product, tensor, hypercomplex / parameterized-hypercomplex (PHM) / quaternion, or otherwise *structured* (non-plain-vector) token embeddings? Is "matrix-native on bytes" a genuinely unoccupied combination, or has something close been tried? Give the closest existing works.

3. **Interaction between tokenization granularity and reasoning / latent structure / rank / superposition.** Is there any evidence that byte- vs. token-level granularity changes how models represent multi-step reasoning, latent chain-of-thought, effective rank of intermediate representations, or feature superposition? Or is tokenization essentially orthogonal to these phenomena? Include any interpretability work touching on tokenization effects.

4. **Methodology: one-variable-at-a-time in architecture research.** What is the established best practice and precedent for controlled architecture comparisons — changing a single variable while holding confounds fixed (data, tokenizer, compute) — in LLM / representation-learning research? Cite examples where conflating two changes produced uninterpretable results, and examples of clean controlled comparisons (e.g., Meta Web-SSL data-vs-objective controls, DINOv3, controlled scaling-law studies). Does this literature support or undercut the "hold the tokenizer standard" recommendation?

5. **Steelman the byte side.** Make the strongest evidence-backed case that byte-level input specifically *helps* a matrix/rank model. For example: contextualized byte-window encoders (CANINE/BLT local encoder) or outer-product byte embeddings could produce naturally higher-rank input matrices; richer sub-word structure could make rank dynamics more expressive or more measurable. Is there a real synergy between tokenizer-free input and matrix/rank structure, or not?

### Output format

- Start with a **one-paragraph bottom-line answer** to the core decision (bytes vs. standard tokenizer for this specific experiment).
- One section per question (1–5), each ending with a **confidence rating** (high / medium / low) and the single most important citation.
- A section: **"Strongest argument FOR standard tokenizer"** and **"Strongest argument FOR byte-level"**, each 3–5 bullets with citations.
- A final **Citations** list: title, authors, venue, year, arXiv ID / DOI / URL. Do not fabricate — if a claim is unverified, label it clearly.
- Flag anything where the literature is genuinely split or absent.
