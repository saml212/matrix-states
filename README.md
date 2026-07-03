# Learned Representations

Better-fitting representations for machine cognition: byte-level inputs, contextualized matrix tokens, measurable reasoning capacity.

## The Question

Language is a powerful cognitive tool. It encodes most of accumulated human knowledge, and any useful model has to interpret it and communicate in it. It developed under specific constraints — embodied, serial, social communication at ~50 bit/s — and machines operate under different constraints. We investigate whether using language as the cognitive medium for machines is an artificial constraint that limits abstract reasoning.

## Three Research Threads

**Byte-level inputs.** The model ingests data the way it lives in computers: raw bytes. Text as UTF-8, images as pixels, audio as samples, code as source bytes. Removing the tokenization layer lets the model develop its own vocabulary from the data.

**Matrix-valued token representations.** Each token is a d×d matrix. The matrix structure provides a measurable, differentiable observable — rank — that quantifies how many independent reasoning paths a representation holds in superposition. The matrix structure also affords contextualized token embeddings, where each token's matrix encodes learned pairwise interactions across a local window of inputs.

**Inference-time reasoning with structured representations.** We test whether matrix tokens make abstract reasoning measurable during inference-time compute (the GPT-o1 "think before answering" regime). The setting is continuous-reasoning models like CODI and COCONUT, where the model "thinks" by generating latent representations before producing an output. The empirical question: does matrix rank track the number of distinct reasoning paths a model holds during this process?

## What We've Shown So Far

Two eras of findings. First, on a **bolt-on** matrix bottleneck grafted onto
a vector-pretrained continuous-reasoning model (CODI-style): matrix
representations gave a large per-parameter efficiency win at equal
iterations, but were **rank-blind** — the training gradient never used
matrix rank to encode reasoning structure. That negative result, with its
mechanism diagnosed and a positive-control falsification test, was accepted
at ICML MI Workshop 2026 ("The Gradient Does Not See Rank").

Second, and more recent: a **matrix-native, trained-from-scratch** model
(no bolt-on, no vector teacher) resolves the open question the workshop
paper left behind. Five programs (2026-07-01 → 07-04) each designed,
adversarially attacked, built, independently audited, run, and closed:

- **Gradient descent recruits provably-necessary matrix rank** when a task
  is constructed so that no rank-1 shortcut exists — effective rank tracks
  the required K almost exactly, and forcing rank below K causally breaks
  recovery (a sharp step at k≈K, not a gradual slope).
- That rank **composes**: the recruited K-rank matrix survives repeated
  self-application to hold-out reasoning depths never seen in training.
- The same rank-K binding, and its composition, was then confirmed on a
  **production fast-weight architecture** (DeltaNet) — first on hand-built
  synthetic keys (a razor-sharp exactness cliff), then on **real
  GPT-2-tokenized text** (a graded exactness frontier, since the model's
  own learned keys aren't perfectly orthonormal).
- The project's oldest negative result — "matrix ops lose at matched
  FLOPs" — now has a **named mechanism**: a specific projection-family
  restriction, not matrix-valued tokens per se, accounts for most of the
  per-FLOP gap.

An active follow-on studies *why* real-text composition falls short of the
synthetic exactness cliff, with a first structural fix (differentiable
per-episode key orthogonalization) already showing a large, partial
recovery on the box.

Full project state and the exact numbers behind every claim above:
[STATE.md](STATE.md). Chronological experiment history:
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). Engineering queue:
[matrix-thinking/QUEUE.md](matrix-thinking/QUEUE.md). Bibliography:
[references.md](references.md).

## Papers

- **"The Gradient Does Not See Rank"** — ICML MI Workshop 2026, **accepted**.
  `matrix-thinking/submissions/icml-mi-workshop-2026/`.
- A second paper covering the matrix-native rank-recruitment results is in
  draft. `matrix-thinking/submissions/neurips-ws-2026/`.

## Now Running

The exactness-mechanism follow-on above, on the Brev 8×H100 cluster, plus a
gated check of whether the matrix-vs-vector per-FLOP gap's named mechanism
generalizes to a composition-heavy language-modeling task. Current status
and in-flight work: [STATE.md](STATE.md) "Path Forward."
