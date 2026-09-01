# Parameter-Matched Matrix-vs-Flat Embedding Ablation — Pre-Registration

**Status:** DESIGNED, NOT LAUNCHED. Queue specs staged only
(`matrix-thinking/embed_ablation_specs/0640-0661`). Nothing in this
document authorizes a GPU launch; it is a pre-registration for the
research-cascade novelty/audit gate to review first.

**Author:** builder agent, 2026-09-01. **Concurrent-edit note:** this
design does not touch `matrix-thinking/deltanet_rd/h2h_strengthen_rd.py`
or `HEAD_TO_HEAD_DEMO_DESIGN.md` (both under edit by another agent as of
this writing) and shares no code with either.

## 0. What this fixes

`pebble-ai-site/findings/outer-product-embedding.html` (finding 01) and
`pebble-ai-site/findings/parameter-efficiency.html` (finding 04) are the
project's two earliest published claims. Both rest on comparisons that,
on inspection of the raw evidence, are weaker than their own text admits:

- **Run 18** (`EXPERIMENT_LOG.md` line 673): matrix 2.4M params vs flat
  24.0M params (**10x asymmetric**, flat favored). The finding page itself
  flags this as "unfair on params... supporting data, not headline."
- **Run 22** (`EXPERIMENT_LOG.md` line 745, titled "Param-Matched
  Ablation" despite not being one): matrix 2,552,788 params (mat_dim=16,
  12 layers) vs flat 5,658,428 params (d_model=256, 12 layers) — **2.2x
  asymmetric, flat favored**. `outer-product-embedding.html` S02
  confirms this in its own prose: *"Param-matching at d=16 is
  structurally hard... Run 22 takes the version of the test where the
  flat model is over-parameterized relative to the matrix model."* The
  2.2x came from fixing `d_model = mat_dim² = 256` (the "reshape parity"
  choice), not from solving for total-parameter equality.
- The flat arm in Run 22 **died at step ~2800** before a matched final
  T=1/T=8 readout existed for both arms at the SAME checkpoint (per
  `CLAUDE.md`'s Run-22 lesson: "the param-matched flat-vector ablation
  blocks ALL downstream decisions. Run it first" — it was run, but never
  finished cleanly).
- All of Run 18/22/Round-2's BPB numbers are **n=1**, no seed replication,
  and (per the finding page's own Limitations section) mostly **byte-level
  BPB**, which `CLAUDE.md` names internal-use-only, not a headline metric.
- The one genuinely params-matched historical comparison — **Round 1**
  (Run 10 matrix vs Run 11 vector, ~5.15M params each, `round1_vector_
  script.py` / `round1_matrix_script.py`) — is n=1, PPL-only (no BPB was
  recorded), and on a small 118M-token WikiText-103 corpus. The finding
  page itself calls it "a pre-registration check... not a BPB comparison."

This design replaces all of that with one properly gated ablation: 3
seeds, 2 sizes, a total-parameter match verified by instantiating the
real `nn.Module` (not eyeballing config numbers), a hard negative test
that a mismatch fails loudly, both arms guaranteed to finish (or record an
explicit `CEILING_STOP`, never silently truncate), and a metric that is
not byte BPB.

## 1. Hypothesis and falsifier

**Hypothesis (one sentence):** At equal total parameter count (±1%) and
equal depth (same `n_layers`, same iterative-refinement `T`), a
matrix-native model — outer-product embedding (`u⊗v`) plus RowThenCol
matrix operations — reaches a lower T=1 token-BPB on a GPT-2-tokenized
corpus than a flat-vector model with a direct embedding table and
standard transformer operations.

**Falsifier (one sentence):** The matrix arm does **not** beat the
params-matched flat arm's T=1 token-BPB, by a margin exceeding the
observed seed spread, on at least 2 of 3 seeds, at **both** registered
model sizes.

No third outcome is defined (see §7, Decision rule). This deliberately
narrows finding 01's three-comparison, mechanism-agnostic claim to a
single clean test; it does **not** attempt to adjudicate finding 01's own
open question (rank-1 structure vs factored parameterization — the
ALBERT-style-bottleneck three-way ablation the finding page names as its
own Priority 1) or finding 04's compute-side claim (that is a per-layer
parameter-count argument, not a trainability claim, and is not testable
by a training run at all — see §2.3).

## 2. Closing the reshape-equivalence loophole

`CLAUDE.md`'s hard rule: *"any d²-dim vector can be reshaped to a d×d
matrix and vice versa. Structure only matters if OPERATIONS preserve
it. Flatten = structure gone."* Two places this bites, both closed below.

### 2.1 Embedding

- **Matrix arm:** `embed_u, embed_v: Embedding(V, d)`; token embedding is
  the explicit outer product `M = u ⊗ v`, a **rank-1** `d×d` matrix built
  from `2d` free parameters. This is `matrix_thinker.py`'s
  `MatrixEmbedding`/`round2_matrix_script.py`'s embed pattern, unchanged.
- **Flat arm:** `embed: Embedding(V, d_model)`, a **direct lookup table**
  with `d_model` free parameters per token, no bilinear construction, no
  rank restriction. The flat embedding is never computed as `u⊗v` and is
  never a reshape of the matrix arm's `M` — it is an independently
  parameterized embedding of a chosen width. `d_model` is solved for
  total-parameter equality (§4), not fixed at `d²` (the historical choice
  that produced Run 22's asymmetry) or forced to a particular relation to
  `2d`.

### 2.2 Backbone operations

- **Matrix arm:** every projection (`RowThenColProjection`) computes
  `silu(A @ M) @ B` with `A, B ∈ ℝ^{d×d}`. The induced linear map on
  `vec(M)` is `(Bᵀ ⊗ A)` — a **Kronecker-restricted** subspace of the
  full `d²×d²` linear group. Not every `d×d → d×d` linear map is
  expressible this way (this is exactly finding 04's own framing).
- **Flat arm:** every layer uses `nn.MultiheadAttention` and `nn.Linear`
  on the flat `d_model`-vector — **full, unrestricted** `d_model×d_model`
  weight matrices (in_proj, out_proj, both FFN layers). The flat arm's
  linear maps are strictly more general than the Kronecker family the
  matrix arm is confined to, at any `d_model`, and in particular strictly
  more general than "the matrix arm reshaped" would be even at
  `d_model = d²` (a full `d²×d²` Linear can express `(Bᵀ⊗A)` as a special
  case, but the flat arm here trains its own independent full weight
  matrix — it is not initialized from, tied to, or reachable by any
  operation on the matrix arm's parameters).

**Net effect:** if the matrix arm still wins at equal total parameters
despite the flat arm having access to a strictly larger per-parameter
hypothesis class at every layer, that is evidence the win is about
*structure*, not about which family happens to be more constrained. If
the flat arm wins or ties, that is evidence for the reverse — either
outcome is informative, which is what a clean ablation requires.

### 2.3 What this design does NOT re-test

Finding 04's `d⁴` vs `2d²` per-layer parameter-count claim is a **counting
fact** about two specific operator families at a **fixed `d`** — it is
true by construction (verified by instantiating both `nn.Module`s, see
`matrix-thinking/src/matrix_output_heads.py`'s own benchmark harness) and
is not something a training run confirms or falsifies. What this design
*does* inherit from finding 04 is the practical consequence: because we
solve `d_model` for parameter equality rather than fixing it at `d²`
(reshape parity), the flat arm's backbone FLOP cost lands close to the
matrix arm's own (§4.2), not at the historical 8×-128× blowup — which is
precisely why every registered cell fits comfortably under the 2 GPU-h
ceiling (§5).

## 3. Metric: token-BPB, not byte BPB

`CLAUDE.md`: *"Use standard benchmarks for publishable claims. Byte BPC
is for internal use."* Per the task brief, the headline metric is **not**
literal bytes-per-byte BPB (which would require a bytes-per-token
conversion factor that the box's corpus `meta.json` files do not record —
confirmed by reading `rebuild_lm_corpora_rd.py`'s `meta` dict: it carries
`vocab_size`, `tokenizer`, `eot_separated`, `source`, `split`, `recipe`,
but no `bytes_per_token` field). Instead:

- **Data:** GPT-2-tokenized corpus already materialized on the box at
  `/data/deltanet_rd_data/wikitext103_mix_eot_extended/` (corpus name
  `wikitext-mix-ext` in `lm_pretrain_rd.py`'s `CORPUS_DIRS`), confirmed
  live in the completed queue spec
  `experiment-runs/2026-08-29_box_final_archive/queue/completed/
  645_laneB_392m_seedext_off_wikitext-mix-ext_s29.json`. `meta.json` for
  this corpus asserts `vocab_size == 50257`, `tokenizer == "gpt2"`,
  `eot_separated == True` — both arms' legacy architecture (from
  `round1_matrix_script.py` / `round1_vector_script.py`) already consumes
  a GPT-2-vocab `Embedding(vocab_size, d)` table unmodified, so **no
  architecture change is needed** to switch off byte-level vocab.
  `wikitext-mix-ext` (not `openr1-mix-ext`) is used for **all** cells per
  `CLAUDE.md`'s "use the same dataset for all experiments in a
  comparison" rule.
- **Headline metric — token-BPB:** `cross_entropy_nats / ln(2)`, i.e.
  bits per GPT-2 token (NOT bits per raw byte — no byte-length claim is
  made anywhere in this design). Also reported: val loss in nats, and
  perplexity, for continuity with the historical Round-1/Round-2 numbers.
- **Both T=1 and T=8** are reported for both arms at every checkpoint and
  at the final step, closing Run 22's actual failure mode (its flat arm
  never reached a finished T=1/T=8 pair — it died mid-run). The runner
  (`embed_ablation_rd.py`) writes `final_evals: {T1: {...}, T8: {...}}`
  unconditionally at every `--eval-interval` and forces one last eval at
  the terminal step (`COMPLETED` or `CEILING_STOP`), so a JSON with
  `complete: false` still carries whatever T1/T8 pair its last completed
  eval produced — the harvest script only accepts `complete: true` cells
  into the decision rule (§7), but a `CEILING_STOP` cell's partial curve
  is still inspectable for diagnosis.

## 4. Architecture and parameter matching

Both arms reuse the **exact** module definitions from
`experiment-runs/8xh100-session1/round1_matrix_script.py` /
`round1_vector_script.py` / `round2_matrix_script.py` (copied
module-for-module into `matrix-thinking/src/embed_ablation_rd.py`, single-
GPU, DDP stripped since every cell fits on one H100) — this ablation
changes the comparison's rigor, not the architecture the claims were made
about. `T` (iterative-refinement count) never changes parameter count in
either arm — the `n_layers`-deep stack is weight-shared and applied `T`
times (`_one_iteration`); only `n_layers` and `d`/`d_model` set the
parameter budget. This is why the "depth vs params" tension the task
brief anticipated (Run 22's "130× per layer" problem) turns out to be
solvable at equal `n_layers` in this regime (§4.3) — the knob that trades
against total params is width (`d_model`), not iteration count.

### 4.1 Registered sizes

| Size | `mat_dim` (matrix) | `n_layers` | `n_heads` | max_len |
|---|---|---|---|---|
| S | 16 | 6 | 4 | 512 |
| M | 24 | 8 | 4 | 512 |

`n_iterations` (T) = 8 for training and for the "matrix model's iterative
T" eval leg, at both sizes (matches Round 2's T=8 convention). Eval always
reports T∈{1, 8}.

### 4.2 Parameter counts (verified by real `nn.Module` instantiation, not
hand algebra — see `embed_ablation_rd.py --selftest` output in §8)

Vocab `V = 50,257` (GPT-2), `max_len = 512`.

| Size | Arm | Config | embed | backbone | head | **total** |
|---|---|---|---|---|---|---|
| S | matrix | mat_dim=16, L=6 | 1,608,224 | 37,074 | 804,880¹ | **2,466,562** |
| S | flat-P (primary, params-matched) | d_model=**24**, L=6 | 1,206,168 | 76,224² | 1,206,168 | **2,468,016** (ratio 1.0006, diff **0.059%**) |
| S | flat-D (disclosed control, unmatched) | d_model=**32** (=2·mat_dim), L=6 | 1,608,224 | 76,224² | 1,608,224 | **3,309,120** (ratio **1.342**) |
| M | matrix | mat_dim=24, L=8 | 2,412,336 | 111,000 | 1,207,320¹ | **3,755,808** |
| M | flat-P (primary, params-matched) | d_model=**36**, L=8 | 1,809,252 | 226,176² | 1,809,252 | **3,765,168** (ratio 1.0025, diff **0.249%**) |
| M | flat-D (disclosed control, unmatched) | d_model=**48** (=2·mat_dim), L=8 | 2,412,336 | 226,176² | 2,412,336 | **5,075,520** (ratio **1.351**) |

¹ matrix head = `MultiProbeHead(d, V, K=d)`: `2·K·d + K·V`.
² flat backbone per layer = `4d_model²+4d_model` (MHA) `+ 4d_model` (2×LayerNorm)
`+ 8d_model²+5d_model` (FFN) = `12·d_model² + 13·d_model`, × `n_layers`.

**Key upshot (closes the "structurally impossible" framing from Run 22 /
`CLAUDE.md`'s "130× per layer" note):** because `V·d_model` (embed+head)
dominates both arms' budgets at GPT-2 vocab scale, and because `d_model`
is *solved* rather than fixed at `d²`, the search space is large enough
that params-matched (flat-P) converges to **<0.3% mismatch at equal
`n_layers`** for both registered sizes — the historical "can't match both
per-layer FLOPs and total params" tension only bites when `d_model` is
constrained to equal `d²` (reshape parity), which this design never does.
flat-D is kept as the disclosed, PRE-REGISTERED params-**unmatched**
control (§4.3) precisely so the old Run-22-style comparison (flat gets
more params, ratio ~1.34-1.35×, structurally similar direction to Run 22's
2.2× and Run 18's 10×) stays visible alongside the new one.

### 4.3 The two pre-registered arms, and which is the verdict carrier

Per the task brief's requirement: *"pre-register BOTH a depth-matched/
params-unmatched arm and a params-matched/depth-unmatched arm."* Since
depth-matching-at-equal-params turns out to be achievable here (§4.2),
the second arm is realized as depth-matched-but-params-**unmatched**
(flat-D) rather than params-matched-but-depth-unmatched — the task brief
explicitly permits substituting whichever pairing is the one that
actually binds in a given regime, and here that is a `d_model` choice, not
an `n_layers` choice:

- **Arm P — params-matched, `n_layers` equal (PRIMARY / VERDICT CARRIER).**
  `d_model` solved via `solve_matched_d_model()` to land within ±1% of the
  matrix arm's total params, gated by `check_param_match()` — a >1%
  mismatch **raises `RuntimeError`** before training starts (verified in
  §8's negative test). This is the arm §1's falsifier and §7's decision
  rule are evaluated against.
- **Arm D — depth-matched (`n_layers` equal), params UNMATCHED
  (secondary/disclosed control).** `d_model = 2·mat_dim` fixed (the
  embedding-only "natural" width — same free-parameter count per token as
  the matrix embedding's `(u,v)` pair, but the backbone and head end up
  larger). Ratio disclosed (~1.34-1.35×, flat favored), never gated. Not
  used for §7's decision — reported so a reader can see how much the
  ±1%-gated matching (vs. the old un-gated reshape-parity choice) actually
  moves the T=1 gap.

If any future re-registration of these sizes ever pushes Arm P's search
outside ±1% (e.g. a much smaller `mat_dim` where the vocab term stops
dominating), the fallback is explicit in `build_arm()`: the search widens
first (`lo`/`hi` bounds), and only if that still fails does the design
need a THIRD arm (params-matched via `n_layers_flat` search instead of
`d_model`) — not built here because it is not needed for either registered
size (§8 confirms both pass at <0.3%).

## 5. FLOPs and memory (CLAUDE.md checklist item 2 — computed on paper,
cross-checked against a real prior run's measured wall time)

**FLOPs, standard forward+backward estimate:** `FLOPs/step ≈ 6 · total_params
· batch · seq_len · T` (Kaplan-style, doubled for backward, × the T
weight-shared iterations). At `T=8`, `batch=64`, `seq_len=512`, and
`total_params≈2.5-3.8M`, this is on the order of `10¹²-10¹³` FLOPs/step —
trivial for an H100's ~streaming capability. **This estimate is NOT used
for wall-clock GPU-h projection**, per finding 04's own documented
caution: small `d×d` matmuls (the matrix arm's RowThenCol at `d=16` or
`24`) are kernel-launch/memory-bandwidth bound on an H100, not
FLOP-bound, so a naive FLOPs/throughput estimate would badly underestimate
real wall time (finding 04 measured only an 8× realized speedup at `d=16`
against a 128× parameter-count gap, for exactly this reason).

**Empirical anchor instead:** `exp_d16_v2_SUMMARY.txt` (a real completed
8×H100 run, same architecture family) — `mat_dim=16, n_layers=8, T=8,
batch=96/GPU, seq_len=512, 3000 steps, GPT-2 vocab, 2,552,788 params` took
**82.0 min wall time**. Since DDP replicates the model per GPU with no
per-step cross-GPU dependency beyond the gradient all-reduce, this 82 min
is also the single-GPU wall time for that exact per-GPU workload — i.e.
**1.367 GPU-h** for that config. Every cell's `gpu_h_estimate` in
§6/`embed_ablation_specs/*.json` is this anchor scaled linearly by
`(batch/96) · (steps/3000) · (params/2,552,788)` — a linear scaling in
tokens-processed and total-params, which is the right first-order model
given both arms are embed/head-GEMM-dominated (§4.2) rather than
backbone-FLOP-dominated.

**Memory:** activations are `(B, L, d, d)` tensors (matrix arm) or
`(B, L, d_model)` (flat arm) × `n_layers` × the gradient-checkpointing
recompute factor (`torch.utils.checkpoint`, already wired into both
`_one_iteration` methods, ported unchanged from `round1_*_script.py`).
At `B=64, L=512, d≤36`, the largest activation tensor is on the order of
`64·512·36·36·4 bytes ≈ 212 MB` per matrix-valued activation buffer, and
checkpointing means only per-layer boundaries are retained — total
training memory footprint is in the **low single-digit GB**, nowhere near
the 80GB H100 ceiling that `CLAUDE.md`'s other hard rules (batch=96 at
mat_dim=32 being the max, the 50K-vocab logits tensor being the VRAM
bottleneck) were written about — those rules were calibrated for models
an order of magnitude larger in vocab-projection batch×seq product than
what a single-GPU, batch=64 cell here produces. **Memory is not a binding
constraint for this design**; the binding constraint is wall-clock GPU-h,
addressed by the `--ceiling-gpuh` hard stop (§6).

## 6. GPU-h ledger (≤2 GPU-h/cell, ≤30 GPU-h total — both satisfied)

22 queue specs total: 4 rate probes (500 steps each, mirrors
`005_laneA_probe_K128_s0.json`'s own Phase-0a discipline: measure real
per-step cost before committing the seeded-cell budget) + 18 seeded cells
(2 sizes × 3 arm-configs {matrix, flat-P, flat-D} × 3 seeds).

| Cell group | n | GPU-h/cell (formula) | Subtotal |
|---|---|---|---|
| probe_matrix_S, probe_flat_S, probe_matrix_M, probe_flat_M | 4 | 0.147–0.224 | 0.741 |
| matrix_S_s{0,1,2} | 3 | 0.587 | 1.761 |
| flatp_S_s{0,1,2} | 3 | 0.587 | 1.762 |
| flatd_S_s{0,1,2} | 3 | 0.788 | 2.363 |
| matrix_M_s{0,1,2} | 3 | 0.894 | 2.682 |
| flatp_M_s{0,1,2} | 3 | 0.896 | 2.689 |
| flatd_M_s{0,1,2} | 3 | 1.208 | 3.624 |
| **Total** | **22** | max single cell **1.208** | **15.622 GPU-h** |

Max single-cell estimate (1.208, flatd_M) is well under the 2 GPU-h cap
(1.66× headroom); the 15.622 GPU-h total leaves ~1.9× headroom under the
30 GPU-h budget for the case where real wall time runs above the formula
estimate (the matrix arm's small-matmul kernel-launch overhead is the
main risk direction per finding 04's own caution, §5). Every `cmd` also
carries `--ceiling-gpuh 2.0` (0.5 for probes) as a hard stop, matching
`005`'s own `--ceiling-gpuh` convention — a cell that runs long writes
`status: "CEILING_STOP"` and exits cleanly rather than overrunning.

**Sequencing:** run the 4 probes first (0640-0643). Their measured
`gpu_h_actual_approx` should be read back before launching 0644+ — if any
probe's actual GPU-h/step differs from the formula estimate by more than
~2×, STOP and re-derive the seeded cells' `--steps` before proceeding
(same discipline `005`'s own hypothesis field documents for its campaign).
This document does not authorize skipping that check.

## 7. Decision rule (pre-registered, no third outcome)

For each registered size independently: sort each arm's 3 seeds' T=1
token-BPB. Compute `seed_spread = max(spread_matrix, spread_flat-P)`
(max − min within each arm, whichever is larger). Pair the seeds (sorted
order) and count a "win" per pair where `flat_BPB − matrix_BPB >
seed_spread`. `size_pass = (wins ≥ 2) AND (n_pairs ≥ 3)`.

- **STRENGTHEN** iff `size_pass` is true at **both** S and M.
- **DROP** otherwise (any size failing, including ties or reversals).

No partial/ambiguous verdict is defined — `matrix-thinking/src/
embed_ablation_rd.py`'s `harvest()` function implements exactly this rule
and only reports `STRENGTHEN`/`DROP`/`PENDING` (pending = not all cells
landed yet), never a third label. This rule only consumes Arm P (flat-P)
cells; Arm D (flat-D) is reported in the harvest for context but never
gates the decision (§4.3).

## 8. Smoke test (CPU, run before any box deployment)

`embed_ablation_rd.py --selftest` does, on CPU with a scratch CPU-only
torch venv (this sandbox has no GPU and no pre-installed torch; a fresh
`venv` + `pip install torch --index-url https://download.pytorch.org/
whl/cpu` was used to actually execute this, not merely inspected by
reading):

1. Builds both arms at two TOY configs (`tiny-S`: mat_dim=8, `tiny-M`:
   mat_dim=12, vocab=97) and runs one real forward + backward pass each,
   asserting: output shape is `(B, L, vocab)`; loss is finite; **every**
   trainable parameter receives a gradient (`n_with_grad == n_params`,
   not just "most"); gradients are finite; and >50% of parameters get a
   *nonzero* gradient (a coarse dead-parameter check).
2. **Negative test:** constructs a deliberately mismatched flat model
   (`d_model=4`, ~89% off) against a real matrix model's param count, and
   confirms `check_param_match()` reports `FAIL` (not silently `PASS`) and
   that the `build_arm(match="P")`-style code path **raises**
   `RuntimeError` rather than proceeding.
3. Confirms the two **registered** sizes (S, M) at the **real GPT-2 vocab
   scale (50,257)** — the actual configuration the box will run — solve
   to within the ±1% gate (§4.2's numbers), and confirms Arm D at both
   sizes is correctly flagged as outside the gate (informational,
   pre-registered as unmatched, not a failure of the script).

**Verbatim output** (`DRY_RUN_BYPASS=1 /tmp/embed_ablation_venv/bin/
python3 matrix-thinking/src/embed_ablation_rd.py --selftest`, CPU,
`torch==2.14.0`):

```
==========================================================================
EMBED ABLATION SELFTEST (CPU, tiny configs)
==========================================================================

--- size tiny-S: {'mat_dim': 8, 'n_layers': 2, 'n_heads': 2} ---
  [matrix/tiny-S] forward OK shape=(3, 11, 97) loss=4.6934 grads: 66/66 present, 66/66 nonzero (100%)
  flat-P/tiny-S param match: matrix=5,806 flat=6,252 ratio=1.0768 diff=7.682% tol=1.0% -> FAIL  [informational at toy vocab=97; real gate is the 'registered sizes at real GPT-2 vocab' section below]
  [flat-P/tiny-S (d_model=12)] forward OK shape=(3, 11, 97) loss=4.8606 grads: 29/29 present, 29/29 nonzero (100%)

--- size tiny-M: {'mat_dim': 12, 'n_layers': 2, 'n_heads': 4} ---
  [matrix/tiny-M] forward OK shape=(3, 11, 97) loss=4.6326 grads: 66/66 present, 66/66 nonzero (100%)
  flat-P/tiny-M param match: matrix=11,154 flat=9,872 ratio=0.8851 diff=11.494% tol=1.0% -> FAIL  [informational at toy vocab=97; real gate is the 'registered sizes at real GPT-2 vocab' section below]
  [flat-P/tiny-M (d_model=16)] forward OK shape=(3, 11, 97) loss=4.7868 grads: 29/29 present, 29/29 nonzero (100%)

--- negative test: forced param mismatch must raise ---
  NEGATIVE param match: matrix=87,890 flat=9,728 ratio=0.1107 diff=88.932% tol=1.0% -> FAIL
  negative test correctly reports FAIL for the mismatched pair (as expected)
  build_arm-style gate correctly RAISED: simulated build_arm(match='P') gate: NEGATIVE-build_arm-path param match: matrix=87,890 flat=9,728 ratio=0.1107 diff=88.932% tol=1.0% -> FAIL

--- registered sizes at real GPT-2 vocab (50257), match=P ---
  size S param match: matrix=2,466,562 flat=2,468,016 ratio=1.0006 diff=0.059% tol=1.0% -> PASS  (d_model=24)
  size S (match=D, expected UNMATCHED) param match: matrix=2,466,562 flat=3,309,120 ratio=1.3416 diff=34.159% tol=1.0% -> FAIL  (d_model=32) -- D is pre-registered as unmatched, this is informational
  size M param match: matrix=3,755,808 flat=3,765,168 ratio=1.0025 diff=0.249% tol=1.0% -> PASS  (d_model=36)
  size M (match=D, expected UNMATCHED) param match: matrix=3,755,808 flat=5,075,520 ratio=1.3514 diff=35.138% tol=1.0% -> FAIL  (d_model=48) -- D is pre-registered as unmatched, this is informational

==========================================================================
SELFTEST: ALL CHECKS PASSED
==========================================================================
```

**Flag for ruling, not assumption:** the `tiny-S`/`tiny-M` toy-vocab
sub-checks report `FAIL` on the param-match ratio (7.7% and 11.5% off) —
this is **expected and disclosed in the script's own print output**, not
a bug: at `vocab=97` the integer `d_model` search space is too coarse
(only a handful of multiples of `n_heads` exist below the point where
`flat_total` overshoots) for ±1% to be achievable at all. This sub-check
is explicitly excluded from the pass/fail gate (see the code comment
added after the first run surfaced it) precisely because it is a toy-scale
artifact; the real precision claim is the "registered sizes at real
GPT-2 vocab" block, which passes at 0.059% and 0.249%. Anyone re-running
this selftest should not be alarmed by the tiny-config `FAIL` lines.

## 6b. Deployment — exactly what must move to the box, and where

The box (`youthful-indigo-turkey` per `matrix-thinking/H100_SETUP.md`)
has **neither** `matrix-thinking/src/` nor the historical
`experiment-runs/8xh100-session1/*.py` scripts deployed by default — only
`matrix-thinking/chapter2/` and `matrix-thinking/deltanet_rd/`-derived
code is scp'd there per that doc's own "Environment on the box" section.

| Local path | Box path | Why |
|---|---|---|
| `matrix-thinking/src/embed_ablation_rd.py` | `/home/nvidia/embed_ablation/embed_ablation_rd.py` | the entire runner; self-contained, zero import dependency on any other repo file (deliberately — `lm_pretrain_rd.py` is under concurrent edit and pulls in frozen-bias/DeltaNet machinery this ablation does not need) |
| *(nothing else)* | — | no other file is required: the corpus loader, both model arms, the training loop, eval, and harvest are all in the one script |

**Data already on the box, not transferred by this design:** the GPT-2
tokenized `wikitext-mix-ext` corpus must already exist at
`/data/deltanet_rd_data/wikitext103_mix_eot_extended/` (confirmed live —
see §3's citation of the completed `645_laneB_...` queue spec that trains
on this exact directory). If a fresh box or fresh data volume does NOT
have this directory, that is a hard blocker for every cell in this
design and must be flagged before Wave 0 (the rate probes) launches — do
not assume it is there without checking `ls /data/deltanet_rd_data/
wikitext103_mix_eot_extended/meta.json` first.

**Checkpoints:** every cell's `--ckpt-dir` is under
`/data/embed_ablation_ckpts/<cell_name>/` — **never** the box's root
filesystem, per the task brief and `CLAUDE.md`'s general data-placement
discipline (mirrors `/data/fixscale_ckpts/...` in the `645_laneB_...`
spec).

**Outputs:** every cell's `--out` is under
`/home/nvidia/embed_ablation/results/<cell_name>.json`. The harvest step
(`--harvest --results-dir /home/nvidia/embed_ablation/results --out
.../embed_ablation_summary.json`) runs on the box against that directory
directly, or against a copy pulled back to the repo's `experiment-runs/`
archive per that directory's own size-capped hybrid-archive policy
(`experiment-runs/README.md`) — result JSONs here are tiny (no large
Z-dumps or checkpoints), so they belong in the committed
`experiment-runs/` tree, not SSD-only.

**Python:** `/home/nvidia/tdenv/bin/python3` (the box's existing venv,
per `H100_SETUP.md` and every queue spec's `cmd` field) — already has
`torch` per that doc's `pip install torch numpy`. No new dependency is
introduced by this script (it uses only `torch`, stdlib `json`/`math`/
`argparse`/`time`/`pathlib`).

## 9. Queue specs

22 files at `matrix-thinking/embed_ablation_specs/0640-0661_embed_
ablation_*.json`, schema copied field-for-field from
`experiment-runs/2026-08-29_box_final_archive/queue/completed/
005_laneA_probe_K128_s0.json` (`id`, `lane`, `hypothesis`, `cmd`,
`gpu_h_estimate`, `output_dir`, `validity_check`, `notes`). IDs 0640+ sort
behind the running 1.31B K=16 grace wave (0478-0485) and the 0600-0629
recall-strengthening sweep, per the task brief. `lane: "embed-ablation"`
(a new lane tag — this campaign is independent of lanes A/B used
elsewhere). Each spec's `notes` field states the exact scp prerequisite
(§6b) and the formula basis for its `gpu_h_estimate` (§5/§6); each `cmd`
includes `--ceiling-gpuh` as the hard-stop safety valve; each
`validity_check` asserts `complete is True`, `steps_completed` reached
target, and both `T1`/`T8` are present in `final_evals` — a cell that
silently truncates (Run 22's actual failure mode) fails its own
validity check rather than being harvested as if it had finished.

Sequencing: 0640-0643 (probes) → confirm real wall time (§6) → 0644-0652
(size S: matrix, flat-P, flat-D, 3 seeds each) → 0653-0661 (size M, same
structure). Nothing in `embed_ablation_specs/` is queued for execution by
this design — staging only, per the task brief's "queue specs only"
instruction.

## 10. Flags for the audit round (things that do not transfer cleanly —
reported, not assumed away)

- **Simplified window sampling, not `lm_pretrain_rd.py`'s document-aware
  sampler.** `embed_ablation_rd.py`'s `get_batch()` is the plain
  random-contiguous-window sampler from `round1_matrix_script.py`/
  `round2_matrix_script.py` (uniform start index, no document-boundary
  awareness), NOT `lm_pretrain_rd.py`'s AUDIT-FIX-3 machinery
  (`boundary_stats`, `train_doc_offsets`/`val_doc_offsets`-aware
  sampling). Both arms see the identical sampling procedure and the
  corpus IS EOT-separated (so a window that crosses a document boundary
  still carries an in-band `<|endoftext|>` signal, per that file's own
  comment), so this does not introduce an asymmetry between arms — but it
  means this design does not inherit `lm_pretrain_rd.py`'s own
  boundary-crossing-rate quantification. If a future reviewer wants that
  diagnostic, `boundary_stats()` would need to be ported in; it was left
  out here to keep the runner import-independent (see §6b).
- **`--ceiling-gpuh` is checked only at the top of the training loop, not
  inside an eval pass.** A cell that hits its ceiling mid-eval will run
  that eval to completion before stopping (bounded overshoot of at most
  one `--eval-interval`'s worth of eval batches, small relative to the 2.0
  GPU-h ceiling).
- **The GPU-h estimates in §5/§6/every spec's `gpu_h_estimate` are
  formula-extrapolated from one empirical anchor** (`exp_d16_v2`, a
  different-but-architecturally-related run), scaled linearly in
  batch×steps×params — they are NOT measured for this exact script,
  this exact flat-arm `d_model`, or this exact box. This is exactly why
  §6 stages 4 rate-probe cells (0640-0643) ahead of the 18 seeded cells,
  and why every `cmd` still carries a hard `--ceiling-gpuh` stop
  regardless of the formula estimate. Treat the ledger as a planning
  budget, not a guarantee, until the probes report back.
- **This design was authored and selftested entirely off-box, on a CPU-
  only scratch venv (`torch==2.14.0`, no CUDA) in a sandbox with no
  access to the H100 box, `/data/deltanet_rd_data`, or the box's own
  `tdenv`.** The corpus-loading code (`load_corpus_tokens`) and the CUDA/
  bf16-autocast training path have been read against `lm_pretrain_rd.py`'s
  contract and written to match it, but have not been exercised against a
  real `wikitext-mix-ext` corpus file or a real GPU — that first real
  exercise IS what the Wave-0 probes (0640-0643) are for. Do not treat
  the CPU selftest's "ALL CHECKS PASSED" as evidence the box-side data
  loading or CUDA path is bug-free; it only certifies the model
  forward/backward/grad-flow and the parameter-matching arithmetic.
