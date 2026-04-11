# Handoff Prompt for New Experimentor Agent

## READ THESE FILES FIRST (in this order)

### 1. Project Context
- `CLAUDE.md` — Workflow rules and project structure
- `STATE.md` — Current project state and what's running
- `MATRIX_THINKING_ARCHITECTURE.md` — The full architecture spec with verified citations
- `matrix-thinking/PROPOSAL.md` — What to build and run on H100s
- `matrix-thinking/ADAPTIVE_THINKING.md` — Certainty-driven adaptive compute design (novel, verified)
- `CONVERSATION_CONTEXT.md` - What the user wants and the direction of the experiment - the authority
### 2. Architecture Code
- `matrix-thinking/src/matrix_thinker.py` — THE MODEL. Autoregressive matrix thinker with:
  - True thought appending (not refinement-in-place)
  - 3D matrix product attention (einsum 'bhlik,bhmjk->bhlmij')
  - SiLU/SwiGLU activations throughout
  - PonderNet halting with rank-biased convergence
  - RowThenCol projections (silu(A@M)@B) — no flattening anywhere

### 3. Research (read all of these)
- `matrix-thinking/research/3d-operations.md` — How 3D coupling works
- `matrix-thinking/research/matrix-native-projections.md` — Why we don't flatten
- `matrix-thinking/research/matrix-native-operations-code.md` — Working code for every operation type
- `matrix-thinking/research/matrix-operations-answer.md` — Why multiplicative beats bilinear
- `research/cutting-edge-2025-2026.md` — State of the art survey
- `research/matrix-states-landscape-2025.md` — Matrix-valued states in recent literature

### 4. H100 Scripts
- `matrix-thinking/h100_scripts/exp_thinker_main.py` — Multi-config launcher
- `matrix-thinking/h100_scripts/exp_solidification.py` — Solidification experiment
- `matrix-thinking/h100_scripts/common.py` — Shared training loop
- `matrix-thinking/h100_scripts/launch_h100.sh` — 8-GPU parallel launcher
- `matrix-thinking/H100_ENVIRONMENT.md` — Exact H100 specs and setup
- `matrix-thinking/H100_EXPERIMENT_QUEUE.md` — Experiment plan

### 5. Experiment Results
- `EXPERIMENT_LOG.md` — All experiments and results from this conversation
- `matrix-thinking/BUILD_PLAN.md` — Build status and mini test results
- `matrix-thinking/EXPERIMENT_VARIABLES.md` — All knobs to test

### 6. Request Audit
- `REQUEST_AUDIT.md` — What has been fulfilled and what hasn't

## THE BIG IDEA

A neural network where intermediate "thoughts" are 16×16 matrices, not vectors.
The model generates matrix-valued thinking tokens autoregressively — each thought
is APPENDED to the context sequence. The model attends over input tokens + all
previous thoughts. When a thought's effective rank drops close to 1 (crystallizes),
it collapses to a vector and produces a token prediction.

This is novel. Verified against literature (March 2026). Nobody has combined:
1. Matrix-valued autoregressive generation
2. Rank as convergence/halting signal
3. 3D matrix product attention (rows at pos s couple with cols at pos t)

## KEY RESULTS SO FAR

1. **Thoughts solidify.** Rank drops monotonically across thinking steps on reasoning
   data. Confirmed on both Mac Mini and H100.

2. **H100 training is LIVE.** A 2.46M param model is training on the H100.
   SSH: `ssh root@91.199.227.82 -p 11183 -i ~/.ssh/id_ed25519`
   Script: `/root/run_train.py` (5000 steps, 8 thinking steps per token)

   Results at step 1500:
   - Loss: 10.8 → 5.3 (PPL 50274 → 203)
   - Starting rank INCREASES: 2.7 → 4.7 (richer initial thoughts)
   - Thought 7 rank consistently ~2.1 (crystallization point)
   - Thoughts solidify: YES at every measurement
   - Val PPL: 220 (best so far, still improving)
   - Speed: ~15s per 50 steps with bfloat16 autocast

   The training script is at /root/run_train.py. Check progress:
   `ssh ... "tail -20 /proc/$(pgrep python3)/fd/1"` or check results dir.

3. **Matrix V2 beat vectors.** On byte-level prediction: 1.94 vs 2.03 BPC at matched
   params. Matrix operations have an advantage.

## WHAT THE USER WANTS

The user's vision (understand this deeply before experimenting):

**Two modes in one model:**
- MATRIX THINKING MODE: generates 16×16 matrix thoughts (abstract, unresolved)
- VECTOR SPEAKING MODE: generates standard tokens (crystallized, expressible)

The model switches between modes:
- Matrix → Vector: when a thought's rank approaches 1 (crystallized)
- Vector → Matrix: when a running average of prediction certainty drops
  below a threshold (the model is struggling, needs to think)

Both matrices and vectors live in the same context. The model attends over all of
them. This models human cognition: think abstractly → speak → encounter difficulty →
think again → speak again.

The same 256 numbers in memory can be interpreted as a 16×16 matrix (structured,
with row-column coupling via matrix multiplication) or as a 256-dim vector (flat).
Same weights, same model. The difference is which OPERATIONS are applied.

## WHAT HASN'T BEEN BUILT

1. **Vector speaking mode** — the model only does matrix thinking right now, no
   fast vector mode for confident predictions
2. **Certainty-driven mode switching** — designed in ADAPTIVE_THINKING.md but not coded
3. **The two-mode unified model** — see above
4. **ML curriculum for the user** — stopped at backpropagation, need embeddings,
   attention, transformers
5. **Chunked matrix attention** for sequences > 512 tokens

## H100 ACCESS

```
ssh root@91.199.227.82 -p 11183 -i ~/.ssh/id_ed25519
```

- 1x H100 80GB HBM3 currently running
- PyTorch 2.9.1 + CUDA 12.8
- Code at /root/src/ and /root/h100_scripts/
- Data at /root/data/reasoning/ (46M tokens OpenR1-Math)
- Results at /root/results/
- A training run is IN PROGRESS — check with: `ps aux | grep python`

## USER PREFERENCES

- Don't repeat dead ideas (PHM is archived, don't mention it)
- Verify claims with research agents before stating them
- Audit code with separate agents before running
- The user is learning ML fundamentals — explain with real math when asked
- Use the hardware fully
- Be honest about negative results
- Follow the plan→research→build→audit→run→assess→codify workflow
- Never run tasks in background — do everything sequentially unless parallel is needed
- The user values continuous iteration over perfection
