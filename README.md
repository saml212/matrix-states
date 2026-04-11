# Learned Representations

Better-fitting representations for machine cognition: byte-level inputs, contextualized matrix tokens, measurable reasoning capacity.

## The Question

Language is a powerful cognitive tool. It encodes most of accumulated human knowledge, and any useful model has to interpret it and communicate in it. It developed under specific constraints — embodied, serial, social communication at ~50 bit/s — and machines operate under different constraints. We investigate whether using language as the cognitive medium for machines is an artificial constraint that limits abstract reasoning.

## Three Research Threads

**Byte-level inputs.** The model ingests data the way it lives in computers: raw bytes. Text as UTF-8, images as pixels, audio as samples, code as source bytes. Removing the tokenization layer lets the model develop its own vocabulary from the data.

**Matrix-valued token representations.** Each token is a d×d matrix. The matrix structure provides a measurable, differentiable observable — rank — that quantifies how many independent reasoning paths a representation holds in superposition. The matrix structure also affords contextualized token embeddings, where each token's matrix encodes learned pairwise interactions across a local window of inputs.

**Inference-time reasoning with structured representations.** We test whether matrix tokens make abstract reasoning measurable during inference-time compute (the GPT-o1 "think before answering" regime). The setting is continuous-reasoning models like CODI and COCONUT, where the model "thinks" by generating latent representations before producing an output. The empirical question: does matrix rank track the number of distinct reasoning paths a model holds during this process?

## What We've Shown So Far

26 experiments on 8×H100 pods. Findings:

- **Outer-product matrix embedding gives better per-parameter representations at T=1** — reproduced across every configuration tested
- **Rank enrichment** is an emergent property of matrix tokens with bilinear output heads (effective rank 5.0 → 6.1 across iterations) — novel empirical finding
- **The output head determines whether representations enrich or solidify** during iterative refinement — novel finding
- **Matrix operations lose at matched FLOPs** to standard vector transformers — honest negative result that narrowed the project from "matrices everywhere" to "matrices where they provide a unique observable"

Full project state: [STATE.md](STATE.md). Engineering queue: [matrix-thinking/QUEUE.md](matrix-thinking/QUEUE.md). Bibliography: [references.md](references.md). Experiment history: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Next Experiment

Matrix-CODI rank dynamics on GSM8K. Tests whether matrix rank correlates with reasoning depth in continuous-reasoning models. ~2 hours of compute on 8×H100, falsifiable in a single training run, publishable either way. Spec at [matrix-thinking/QUEUE.md](matrix-thinking/QUEUE.md).
