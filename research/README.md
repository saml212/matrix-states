# Research Notes

This folder holds individual research notes from literature investigations and analysis. Each note is a focused output from a research agent or a deep dive on a specific question. The strategic synthesis of these findings lives in [STATE.md](../STATE.md). The bibliography of cited papers lives in [references.md](../references.md).

## Index

### April 2026 multi-agent research session

The most recent work. A multi-agent session investigated 11 topics in parallel and produced a synthesis that informed the project's narrowed direction. The synthesis itself was folded into STATE.md (sections "What the Field Has Shown Us" and "Path Forward"). The individual agent outputs would have been here but were consolidated into STATE.md to reduce doc count. The themes covered were:

- JEPA family (LeCun direction)
- Learned discrete vocabularies (VQ-VAE)
- Pure-sensor / non-linguistic training
- Structured representations beyond flat vectors
- Neuroscience and information theory of language
- HELM deep dive
- LLM-JEPA deep dive
- COCONUT deep dive
- CoT2 deep dive (parallelism vs dimension)
- LeJEPA / SIGReg deep dive
- ContextLM deep dive
- Contextualized matrix embeddings (the question that surfaced the rank-1 limitation of the current embedding)

### March 2026 research backing

- **project-analysis-march2026.md** — Honest assessment of the project after 22 experiments. Identifies what's proven, what's not, and what should change.
- **architecture-research-march2026.md** — Architecture recommendations from literature: Monarch, BLAST, LoopFormer, CliffordNet.
- **how_models_learn_to_think.md** — Training recipes for thinking models: COCONUT, DeepSeek-R1, EBT, LoopFormer, MoR, Quiet-STaR, Universal Transformers.
- **internal-reasoning-tokens.md** — Prior work on internal reasoning: STaR, Quiet-STaR, Pause Tokens, COCONUT, CoCoMix, Thoughtbubbles, Inner Thinking Transformer, Seq-VCR, Fast Quiet-STaR, Soft Thinking.
- **byte-multimodal-sota-april2026.md** — State of the art in byte-level multi-domain models: bGPT, MBLM, Perceiver IO.
- **long-context-byte-models-april2026.md** — Long context for byte models: Mamba, MBLM, EverMind MSA, NSA, Ring Attention, Titans.
- **infinite-context-april2026.md** — Infinite-context SOTA: EverMind MSA, DeepSeek NSA, MoBA.
- **hypothesis-attack-april2026.md** — Honest attack on the cross-domain generalization hypothesis. Six fatal arguments. Killed the original framing.
- **literature-cross-domain-april2026.md** — What supports and challenges the generalization thesis: MBLM contradicts bGPT.
- **zero-param-output-audit.md** — Design audit of the zero-parameter byte output head (16²=256 = byte vocab).
- **quantization-analysis-april2026.md** — Why quantization (TurboQuant etc.) doesn't apply at our current scale.
- **thought-generation-feasibility-april2026.md** — Feasibility of autoregressive matrix thought generation. Killed the pure-autoregressive version.
- **thought-interleaving-research-april2026.md** — Why CoCoMix-style interleaving works (or doesn't) at small scale.
- **brainstorm-cheap-matrix-ops.md** — 22 ideas brainstormed for making matrix operations cheaper.
- **validated-cheap-matrix-ops.md** — Top 5 cheap-ops ideas validated against literature. DeltaNet was strongest, but recurrent ≠ iterative.
- **waterfall-cheap-ops-april2026.md** — Full waterfall (brainstorm → research → attack → validation): speed doesn't fix the quality gap.
- **cutting-edge-2025-2026.md** — Snapshot of recent ML research relevant to the project.

### Older research (March 2026 and before)

- **publishability-assessment.md** — Honest review of which findings are publishable and what's missing.
- **phm-algebra-analysis.md** — Methods for analyzing learned PHM Kronecker factors (squared trace, Killing form, etc.).
- **phm-nilpotent-convergence.md** — Why PHM layers converge to near-nilpotent algebra under standard training. First empirical analysis of what PHM actually learns.

### Matrix operation specifics (moved from `matrix-thinking/research/`)

- **matrix-3d-operations.md** — 3D matrix attention designs (the version that turned out to drive solidification and lose).
- **matrix-native-projections.md** — Sum-of-Kroneckers projection design (K=4 sweet spot at d=16).
- **matrix-native-operations-code.md** — Working code patterns for matrix-native ops (no flatten anywhere).
- **matrix-operations-answer.md** — Multiplicative Lie group action (I+Δ)·M·(I+Γ) as the right matrix-to-matrix update.
- **matrix-states-landscape-2025.md** — The 2025-2026 landscape of matrix-valued hidden states (Mamba-2, RWKV-5/6, TTT, Titans, GLA/DeltaNet).
- **matrix-valued-outputs.md** — Design space for matrix-valued output heads.

## Convention for new research notes

- One topic per file
- Filename = `topic-month-year.md` for time-sensitive surveys, or `topic.md` for stable references
- Each new note gets an entry in this README
- Findings that change project direction get folded into STATE.md
- Don't create a new note for every minor question — fold short investigations into existing notes
