# Conceptual Framework

## The Core Argument

Intelligence might scale in dimensionality of representation, not just speed or memory.

Current neural networks represent every token as a flat vector — a list of d numbers. This is a design choice inherited from the 1980s perceptron, not a principled decision. The brain doesn't work this way. Cortical representations are high-dimensional, structured, and multi-scale. The dimensionality of neural representations in prefrontal cortex is causally linked to cognitive performance.

If we want machines that reason at a higher level, we might need representations that are structurally richer — not just longer vectors, but matrices or tensors that encode relationships between features, not just the features themselves.

## Two Bottlenecks in Current Architectures

### Bottleneck 1: The Tokenizer

Every current model uses a hand-designed tokenizer that locks it to a domain:
- BPE for text (splits at character/subword boundaries)
- Patch embedding for images (chops into fixed 16x16 pixel grids)
- Mel spectrogram for audio (frequency decomposition at fixed time windows)

These are human decisions about what the "atoms" of each domain are. The model never gets to question these decisions. A text model that receives "un" + "break" + "able" never sees the raw characters. A vision model that receives 16x16 patches never sees individual pixels in context.

What if the model could learn its own atoms? Given raw bytes from any domain, discover the right segmentation for that domain? BLT and MBLM have shown this is possible. But nobody has combined it with richer representations.

### Bottleneck 2: The Flat Vector

A token represented as a d-dimensional vector encodes d independent features. But features aren't independent — "noun-ness" interacts with "plural-ness" interacts with "position-in-sentence." A vector can learn these interactions across layers (via attention and MLP), but each layer starts from a flat representation that has lost the structure.

A matrix-valued representation encodes d1 x d2 values. A 32x32 matrix has 1024 entries — same as a 1024-dimensional vector in terms of storage. But the matrix has structure: rows, columns, rank, eigenvalues, singular values. Operations on matrices (multiplication, decomposition, projection) are structurally richer than operations on vectors (dot product, element-wise operations).

The question: does this structural richness buy anything for learning? PHM layers suggest yes — they achieve 4x parameter efficiency by exploiting algebraic structure (Kronecker products) in the weight matrices. CliffordNet suggests yes — the geometric product is so representationally dense that entire FFN layers become unnecessary.

## The Hypothesis

**A model that learns its own tokenization from raw bytes AND processes tokens as structured (matrix-valued) representations will discover domain-appropriate processing strategies without domain-specific engineering, because the combination of adaptive granularity and structural representations is rich enough to capture the relevant patterns in any domain.**

Simpler: let the model decide what the atoms are AND give it richer math to process them with. See if something interesting emerges.

## Why Richer Math Might Help

### The vector case

Standard attention: Q and K are vectors. Their dot product is a scalar measuring similarity.

```
similarity = sum(Q_i * K_i)  →  one number
```

This is a first-order interaction. It asks: "overall, how similar are these two tokens?"

### The matrix case

If Q and K are matrices, their interaction can be richer:

```
similarity_matrix = Q @ K.T  →  a matrix of numbers
```

This captures how each "feature dimension" of Q relates to each "feature dimension" of K. Not just "are these tokens similar?" but "in what ways are they similar?" The result is a matrix of similarities, not a scalar. The attention mechanism can then select different aspects of the value for different reasons.

AlphaFold does exactly this. Its outer product mean module computes matrix-valued pair representations that capture how each amino acid relates to every other. This structural richness is why it works for protein folding — the problem has inherently relational structure that flat vectors struggle to represent.

Language, vision, and audio also have relational structure. A word's meaning depends on its syntactic role, its semantic content, its morphological form, and its discourse position — all simultaneously. A matrix representation could encode these as separate dimensions of a structured object rather than cramming them into a single vector.

### The PHM shortcut

You don't need to go full d x d matrices to get some of this benefit. PHM layers use Kronecker products to create structured interactions between groups of dimensions. A PHM layer with n=4 treats the d-dimensional vector as 4 groups of d/4 dimensions, and applies learned inter-group interactions via Kronecker structure. This is like having a 4x(d/4) matrix representation, but stored as a vector. It's a practical middle ground between flat vectors and full matrices.

## Why Learned Tokenization Complements This

If the model processes rich representations but receives dumb tokens (fixed BPE chunks), the representations have to spend capacity undoing the tokenizer's mistakes. "un" + "break" + "able" forces the model to reconstruct "unbreakable" before it can reason about it.

If the model learns its own segmentation, it can create tokens that are already meaningful for its internal representations. The segmenter and the representations co-evolve: the segmenter learns to create chunks that the PHM layers can process well, and the PHM layers learn algebra that works well with the segmenter's chunks.

This co-evolution is the key insight. Neither learned tokenization alone (MBLM, BLT) nor richer representations alone (PHM, geometric algebra) are the full story. The combination creates a feedback loop where each part makes the other better.

## The Multi-Modal Test

Training on multiple domains simultaneously is the test of whether this is real. If the architecture is truly domain-general:

1. The segmenter should produce different boundary patterns for text vs images vs audio, without being told which is which
2. The PHM algebra should either converge to one universal structure (suggesting domain-general computation) or specialize by layer (suggesting the model routes different domains through different computational strategies)
3. Cross-domain pretraining should help — structure learned from images should improve text processing and vice versa

If (1) happens, the model has learned to see structure in raw data.
If (2) happens, the model has learned domain-appropriate computation.
If (3) happens, the model has learned transferable structure across domains.

Any one of these would be a finding. All three together would suggest that adaptive segmentation + structured representations is a viable path toward domain-general intelligence.

## Connection to Neuroscience

The brain arguably implements something like this:

- **Learned segmentation:** Sensory cortex segments raw input into meaningful units. Visual cortex discovers edges, shapes, objects. Auditory cortex discovers phonemes, words, prosodic boundaries. This segmentation is learned, not hard-coded, and develops differently based on experience (visual cortex in blind people processes language).

- **Structured representations:** Neural representations aren't flat vectors. They have dimensionality structure — the prefrontal cortex uses a gradient from low-dimensional (generalizable) to high-dimensional (discriminative) depending on cognitive demands. Grid cells encode information as multi-scale periodic patterns (essentially a structured mathematical object, not a flat list).

- **Domain-general architecture:** The cortical column repeats across the entire cortex. The same six-layer structure, the same local computation, different inputs. Mountcastle's uniformity principle suggests one algorithm underlies all cortical processing.

The learned-representations experiment is a computational test of whether these properties are sufficient for domain-general learning, or whether the brain's other features (spiking, dendritic computation, neuromodulation, embodiment) are also necessary.

## What This Is NOT

- This is not AGI. It's a test of whether two specific architectural choices (learned segmentation + structured representations) enable emergent domain-general behavior.
- This is not trying to replace transformers. The transformer backbone stays. We're changing the input pipeline and the linear layers, not the attention mechanism.
- This is not about scale. The experiment is designed to show emergent behavior at small scale (10-50M parameters). If the behavior doesn't emerge small, scaling won't create it.
