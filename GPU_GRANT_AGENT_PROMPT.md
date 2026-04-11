# Prompt for GPU Credit Application Agent

You are writing a GPU compute grant application for a novel ML research project. Before writing anything, read ALL of the following files in this repository to build complete context:

## Required Reading (in order):

### Core thesis and current state:
- `STATE.md` — The thesis (narrowed in April 2026), what's proven, what's next
- `ARCHITECTURE.md` — Architecture spec for matrix-valued thoughts
- `matrix-thinking/MATRIX_CODI_EXPERIMENT.md` — The next experiment to run, fully specified
- `matrix-thinking/EXPERIMENTS.md` — Experiment queue and embedding designs

### Experiment results (26 experiments, all documented):
- `EXPERIMENT_LOG.md` — Every experiment with exact numbers

### Research backing (17 documents):
- `research/april-2026-synthesis.md` — **Most recent synthesis. Read this first.** Consolidates findings from a multi-agent research session covering JEPA, VQ-VAE, pure-sensor learning, scaling-vs-structure, neuroscience, COCONUT, CODI, LeJEPA, HELM, LLM-JEPA, ContextLM, and contextualized matrix embeddings.
- `research/project-analysis-march2026.md` — Honest assessment of findings
- `research/architecture-research-march2026.md` — Architecture recommendations from literature
- `research/how_models_learn_to_think.md` — Training recipes for thinking models
- `research/internal-reasoning-tokens.md` — Prior work on internal reasoning
- `research/byte-multimodal-sota-april2026.md` — State of the art in byte-level multi-domain models
- `research/long-context-byte-models-april2026.md` — Long/infinite context SOTA
- `research/infinite-context-april2026.md` — EverMind MSA and recent breakthroughs
- `research/hypothesis-attack-april2026.md` — Honest attack on the original hypothesis (shows rigor)
- `research/literature-cross-domain-april2026.md` — Cross-domain transfer literature
- `research/zero-param-output-audit.md` — Novel output head design
- `research/quantization-analysis-april2026.md` — Why quantization doesn't apply yet
- `research/thought-generation-feasibility-april2026.md` — Feasibility analysis of thought interleaving
- `research/thought-interleaving-research-april2026.md` — Why scale matters
- `research/brainstorm-cheap-matrix-ops.md` — 22 ideas for cheaper operations
- `research/validated-cheap-matrix-ops.md` — Top 5 validated, why they don't fix the quality gap
- `research/waterfall-cheap-ops-april2026.md` — Full waterfall: speed doesn't fix quality

### Experiment scripts and results archive:
- `experiment-runs/8xh100-session1/` — All logs, scripts, results from 26 experiments
- `matrix-thinking/h100_scripts/` — All runnable experiment scripts
- `references.md` — Comprehensive paper bibliography

---

## What to emphasize in the application

### The Thesis (Narrowed — April 2026)

After 26 experiments and a multi-agent research synthesis, the project has narrowed from a broad "matrices are better" thesis to a sharp, falsifiable hypothesis with a single high-priority experiment.

**The narrowed claim:** *In a continuous-reasoning language model with matrix-valued thoughts, the effective rank of the thought matrix at reasoning step `t` correlates with the number of distinct reasoning paths held in superposition. The matrix structure provides a measurable, differentiable observable (rank) that vector representations cannot cleanly access.*

This narrower hypothesis bridges three existing research threads that nobody has connected:
1. **Theoretical superposition story** (Reasoning by Superposition arXiv 2505.12514, CoT2 arXiv 2505.23648). Argues continuous thoughts hold parallel reasoning paths but only proves it via existence constructions and accuracy curves.
2. **Empirical rank dynamics** (this project, Round 2, May 2025). Discovered MultiProbeHead drives rank enrichment during iterative refinement (5.0 → 6.1) — unprecedented finding.
3. **The 2026 rebuttal** (The Illusion of Superposition arXiv 2604.06374). Argues fine-tuned COCONUT reaches 96.6% without latent tokens; superposition is contested.

**Nobody has measured rank as a structural correlate of reasoning capacity.** This is the unfilled gap. The matrix-CODI experiment can adjudicate this dispute.

### What We've Already Proven (with personal compute)

- **26 experiments** on 8×H100 GPUs across multiple sessions
- **Outer-product matrix embedding gives better per-parameter representations at T=1** — proven, reproducible across all configurations (T=1 BPB 2.12 vs vector baseline 4.29 at matched params)
- **Rank enrichment is an emergent novel phenomenon** — never reported in literature, occurs only with MultiProbeHead output
- **Output head determines representation dynamics** — MultiProbeHead enriches (rank rises 5.0 → 6.1), vector-collapse solidifies (rank falls). Same model, different output head. Novel finding.
- **130× parameter efficiency per layer** for matrix operations vs standard vector layers
- **Comprehensive comparison against LoopFormer** (ICLR 2026 SOTA)
- **Rigorous negative results documented:** PonderNet collapse (Run 8), 3D attention dead end (Run 20), thought interleaving doesn't beat depth at small scale (Run 25), matrix operations lose at matched FLOPs (Runs 13-14)
- **Verified novelty against exhaustive literature search** (March 2026 + April 2026 multi-agent synthesis)

### Why this matters now (April 2026 field trajectory)

The April 2026 research synthesis confirmed the field is moving in directions that strengthen the narrowed thesis:

- **JEPA is gaining momentum.** LeCun left Meta in February 2026 and raised ~$1B for AMI Labs to pursue JEPA exclusively. LeJEPA (Nov 2025) finally solved collapse via SIGReg.
- **Pure-sensor models match language-supervised models at scale.** DINOv3 (7B params, no text) is the first SSL model to beat weakly-supervised peers across the board.
- **Discrete vocabularies have lost the text race.** Meta abandoned Chameleon (VQ multimodal) for BLT (byte-level, tokenizer-free).
- **Neuroscience case for non-linguistic cognition is mainstream.** Fedorenko et al. 2024 *Nature*: "Language is primarily a tool for communication rather than thought."
- **HELM (May 2025, NeurIPS 2025)** showed billion-parameter fully-hyperbolic LLMs can match Euclidean baselines, but only with full architectural commitment. Existence proof that pervasive structured architectures work at scale.

### What the compute is for

**Phase 1 (~5 H100 hours): The matrix-CODI rank dynamics experiment.** Fully specified at `matrix-thinking/MATRIX_CODI_EXPERIMENT.md`. Three runs: vanilla CODI baseline, matrix-CODI with d=16 bottleneck, rank-projection ablation. Falsifiable in a single training run. Publishable either way.

**Phase 2 (~10 H100 hours): K-bigram embedding ablation.** Conditional on Phase 1 producing signal. Tests whether contextualized starting embeddings (rank 3 by construction) change the rank dynamics picture. See `matrix-thinking/EXPERIMENTS.md` for the design catalog.

**Phase 3 (~50 H100 hours): Scale to 10M+ params on standard benchmarks.** Conditional on Phases 1-2 showing structural signal. Required to make the rank-tracks-reasoning result publishable beyond a workshop.

**Phase 4 (~100 H100 hours): Byte-level JEPA with LeJEPA SIGReg.** Thread B. Tests whether the byte-level direction is viable for the project's broader goals. Deferred until Thread A produces a result.

**Total compute requested: ~150-200 H100-hours (~$400-600 at cloud rates)**

### Why this matters

- **The rank-tracks-reasoning hypothesis is genuinely unfilled in the literature.** Three research lines have come within one step of measuring it (Reasoning by Superposition, mechanistic interpretability, dimensional collapse work) and none have connected the dots.
- **The matrix architecture provides a unique observability advantage.** Rank is measurable on a single matrix; vectors only allow estimation across an ensemble.
- **The experiment adjudicates a real published dispute** (Reasoning by Superposition vs Illusion of Superposition rebuttal). Either result is a contribution.
- **Byte-level domain-agnostic models are an active research frontier** (BLT, EvaByte, MBLM, Bolmo all published 2024-2025). The byte-level direction has Meta's bet behind it (BLT replaced Chameleon).

### The team's approach

- **Rigorous experimental methodology** with mandatory pre-experiment checklists (hypothesis statement, FLOPs/memory analysis, literature check via research agents, attack-attempt by separate agents, success criteria)
- **Every experiment is logged** with exact scripts, training curves, and honest assessment
- **Negative results are documented** (3D attention dead end, PonderNet collapse, matrix ops losing to vectors at matched FLOPs, thought interleaving losing to depth)
- **All code and data are reproducible** — exact scripts saved alongside results
- **Multi-agent research workflow** with brainstorm + research + attack + validation stages before any experiment is built (see CLAUDE.md hard rules)

### Tone guidance

Be honest, not hype. Lead with what's proven, acknowledge what's not. Emphasize:

1. **The narrowness of the new claim** — this is a sharper hypothesis than the original "matrices are better" framing, and a sharper hypothesis is more credible.
2. **The convergence of recent literature** that supports the narrower direction (JEPA momentum, pure-sensor results, continuous reasoning research).
3. **The rigor of the experimental process** — 26 experiments documented, honest negative results, exhaustive literature review, narrowed direction based on evidence.
4. **The small ask** — the next experiment is ~5 H100 hours. The full research program is under 200 H100 hours. This is an unusually cheap research program for a potentially high-impact result.

The strength of this application is that the researcher has already invested significant personal compute to validate the direction, identified the unfilled gap precisely, and designed a falsifiable experiment that can settle a real published dispute in the field.
