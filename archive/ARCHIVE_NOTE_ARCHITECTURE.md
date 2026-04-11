# Why ARCHITECTURE.md was archived

**Archived:** 2026-04-09

This document described the "autoregressive matrix thought generation" architecture — a model that would generate new matrix-valued thoughts one at a time, append them to context, and decide when to stop via a halt mechanism. The idea: each thought builds on prior thoughts and inputs, rank should grow across the sequence, the model halts when its representation crystallizes.

## Why it was killed

A pre-experiment attack analysis (April 2, 2026) found multiple fatal flaws in the "pure autoregressive matrix thought generation" formulation:

1. **Rank does not grow across independently generated thoughts.** The architecture assumes successive thoughts encode richer hypotheses, but with no constraint on the generation process, thoughts converge to similar representations.
2. **Vanishing gradients through the attention chain.** Long thought sequences lose gradient signal to early input tokens.
3. **Cannot train from scratch without a curriculum.** COCONUT proved this conclusively.
4. **"Just add layers" beats it at matched FLOPs.** Run 25 confirmed this empirically: at 288K params, a no-thought baseline with more layers outperformed every thought interleaving config.
5. **One-at-a-time thought generation is ~2000× more expensive than all-at-once processing** — the architecture's core training mechanism is impractical.
6. **A learned "thought seed" makes all thoughts converge to the same starting point** — losing the diversity the architecture needed.

Full attack analysis: `research/hypothesis-attack-april2026.md`.

## What replaced it

The narrowed direction (April 2026) is the matrix-CODI rank dynamics experiment, specified in `matrix-thinking/QUEUE.md`. Instead of generating new matrix thoughts autoregressively, the experiment uses CODI's existing continuous-reasoning mechanism (hidden-state feedback at fixed latent slots) and inserts a matrix bottleneck. The hypothesis is no longer "matrices everywhere" but the narrower, testable claim that **matrix rank correlates with the number of reasoning paths a model holds in superposition during continuous reasoning**.

## What was kept

Some architectural primitives from this document survived the narrowing:

- **Outer-product token embedding** (`byte → u_b ⊗ v_b`) — proven empirically, still in use
- **Multiplicative thinking layer** `(I+Δ)·M·(I+Γ)` — used in matrix-CODI as the optional matrix-thinking iteration inside the bottleneck
- **Frobenius attention** (Frobenius inner product as attention scores, flash-compatible) — used in all matrix-token models
- **MultiProbeHead output** (bilinear probes reading the matrix) — drives rank enrichment, kept

These primitives live in `matrix-thinking/src/` and are documented in `matrix-thinking/QUEUE.md`.

## Read this document for

- Historical context on the original "matrices generate thoughts" framing
- The killed mechanisms (halt signal, learned thought seed, autoregressive thought appending)
- Why CoCoMix-style interleaving (which we later also abandoned at small scale) was once seen as the "right" approach

Anything else: read STATE.md.
