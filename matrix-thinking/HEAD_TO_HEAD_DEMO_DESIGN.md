# HEAD-TO-HEAD DEMO — the program's capstone question

## §1 DESIGN — HEAD-TO-HEAD DEMO (Rev 2, post-attack-round-2 revision,
2026-07-08) — does the matrix-native fast-weight model beat a matched
conventional baseline on real tasks at meaningful scale, or is its value
honestly bounded?

Status: **Rev 2, pre-attack (round 3 pending), zero GPU spent.** Rev 0
(§1.1-1.12 below) was independently attacked and returned NEEDS-REVISION
(§1.13, retained verbatim as the record: 2 FATAL-caliber + 3 MAJOR + 2
MINOR findings), resolved by Rev 1 (§1.14). Rev 1 was independently
attacked a second time and ALSO returned NEEDS-REVISION (§1.15, retained
verbatim as the record: 2 FATAL findings on axis 2 — a degenerate
cap_length at rung-1 and an undisclosed train/eval attention mismatch,
both with mandated INFERENCE-ONLY fixes — + 4 MAJOR + 2 minor). This Rev 2
resolves every §1.15 finding **structurally, staying inference-only per
the binding M-NEW-4 budget constraint** (a memory-multiplier sweep
replaces the single degenerate equal-byte point; an attention-sink
eviction patch partially mitigates the train/eval mismatch, with the full
cap-trained fix registered, costed, and deferred as a follow-on, never
silently absorbed) — §1.16 maps each finding to its exact resolution and
flags the one item that could not be closed with full margin (the Wave −1
GPU-h fit, now thinner than any prior revision — §1.6). Ratified by the PI
as the program's capstone question. **This design absorbs one mid-drafting
PI directive** (received during Rev-0 authorship, before first commit —
see the note at the top of §1.3) that re-pointed the comparison framing
toward the future's binding constraints (data/quality scarcity,
inference-memory/HBM scarcity) rather than today's compute-FLOPs framing;
nothing below §1.3 predates that directive.

---

### 1.0 Reading list this design builds on (context, not repeated here)

- **`CLAUDE.md`** Hard Rules (full file read this session) — the ones with
  direct teeth on this design, cited by name throughout: the param-matched
  flat-vector ablation "blocks ALL downstream decisions... run it first";
  "use the SAME dataset for ALL experiments in a comparison"; the
  single-full-K-cycle permutation lesson (a general permutation's `π^h` is
  periodic with cycle length, silently collapsing nominally-held-out hops);
  "readout must force EXACT CONTINUOUS recovery... never argmax/nearest-
  neighbor... when a rank≥K bound depends on it"; the position-decomposition
  shortcut (a hard single-state bottleneck is required for within-
  representation rank to be load-bearing); "a calibration run... before a
  big sweep is mandatory, not optional"; "Making matrix ops cheaper does
  NOT fix the quality gap. Speed ≠ quality"; "Multiple independent
  adversarial audit rounds catch different bugs each round"; "hold
  tokenization... FIXED when testing a primary hypothesis... treat the
  second axis as a separate, explicitly-sequenced follow-on" (this is the
  rule that keeps byte-level input explicitly parked/out of scope, §1.4).
- **`STATE.md`** — PI check-in package (2026-07-08 refresh), Hardware
  section (Brev grant correction, ~192 GPU-h/day operative budget), Chapter
  2 — STATUS block, Track C rung-3 harvest block. GPUs 0-7 confirmed **all
  free** as of this design.
- **`FROZEN_BIAS_LM_DESIGN.md`** — the fix's exact mechanism, its only
  real training evidence (14M, DESCRIPTIVE TIER ONLY), and the shared 135
  GPU-h program ceiling this wave draws against.
- **`KEY_ANCHORING_SCALING_DRAFT.md`** §15.25-15.27 — the standing capacity
  finding (no cliff to K/d=0.9375) and the instrument limits (tolerance
  miscalibration, pool-margin degeneracy) that must NOT be silently reused
  past where they were calibrated.
- **`REASONING_LINK_DESIGN.md`** §16.15-16.20 — the dead `d_state`-space
  zero-shot geometric-transfer readout (triple-null, PROBE-INVALID) and the
  live lesson on seed-pooling batch-effects.
- **`matrix-thinking/chapter2/TASK_D_WRITEUP.md`**, **`TASK_E_FINDINGS.md`**
  — the small-scale (≤1M param, d≤16 confirmed) proof that SGD recruits
  provably-necessary rank, and that the recruited operator composes
  correctly (exact multiplicative composition) under repeated
  self-application.
- **`EXPERIMENT_LOG.md`** — Track C rung entries (14M/98M/392M/1.31B
  scale ladder) and the frozen-bias rung-1 verdict entry.

---

### 1.1 The hypothesis, the honest framing, and the pre-registered outcomes

**One-sentence hypothesis:** A matrix-native fast-weight LM (`DeltaNetLM`,
frozen-bias-fixed, `matrix-thinking/deltanet_rd/lm_pretrain_rd.py`) beats a
parameter- and FLOP/data-matched conventional baseline on at least one of
two future-facing primary comparison axes (data-efficiency;
inference-memory-matched long-horizon recall), at real tasks and a scale
(14M, escalating to 392M) large enough that the result is not a
small-scale artifact.

**This is a demonstration-or-bound wave, not a confirm-only wave.** Every
prior program in this project (Chapter 2, key-anchoring, reasoning-link)
was explicitly built to survive a null result and still ship a paper. This
wave is the same: every outcome below is pre-registered as reportable
BEFORE any cell launches, each with its own paper implication decided now,
not at write-up time (mirrors `DELTANET_REALDATA_DESIGN.md` §6.3's "what
claim tier this earns — restated, not allowed to drift at write-up time").

**Two primary axes, co-registered with an OR-win rule (full statement,
including the FLOP-matched control's demoted role, in §1.3-1.4):**

| Outcome | Trigger | Paper implication |
|---|---|---|
| **WIN** | Contender beats its axis-appropriate baseline beyond the pre-registered margin on **at least one** of the two primary axes (data-efficiency OR inference-memory-matched), CI excludes the margin | Headline capstone result, framed by whichever axis won (the "future's constraints" framing the PI directive supplies: quality-data scarcity for data-efficiency, HBM/memory scarcity for inference-memory-matched — "constant-memory minds" if axis 2 wins). The OTHER axis's actual result (win, tie, or loss) is disclosed alongside, never hidden. The FLOP-matched control result is reported too, as the "today's scarcity caveat" — if the contender loses that control while winning a primary axis, the paper says so explicitly (a real, common, honestly-reportable pattern: worse today, better where the future is heading). |
| **TIE** | Neither primary axis wins NOR loses beyond its margin (both CIs land inside their TIE bands) | Itself informative: matrix structure costs nothing on the axis(es) that matter for the future, while carrying the program's other proven properties (rank-recruitment, composition) a vector/capped-cache baseline cannot match by construction. |
| **LOSE** | Neither primary axis wins, AND at least one primary axis's baseline beats the contender beyond its own margin (CI excludes the margin) | Reported plainly — this is `CLAUDE.md`'s "making matrix ops cheaper does NOT fix the quality gap" rule biting for real, on axes chosen specifically to favor the contender. Paper framing narrows to the smaller, still-true claims already banked (rank recruitment, exact composition, no-cliff capacity, in isolation) with the end-to-end claim explicitly bounded. No result is re-run or re-scoped to avoid this outcome (§1.5's escalation rule: never escalate a loss to make it win). |

Multiple-comparison discipline for the 2-primary-axis OR-win structure
(full statement in §1.8): each axis keeps its own nominal 95% CI, no
downward alpha-correction applied per axis — this is a standard
co-primary-endpoints design where success-on-either is the intended
semantics — but the paper honestly discloses the resulting family-wise
inflation (≈9.75% vs. a single axis's 5%, under a global null) rather than
silently presenting either win as a clean single-test 95% result.

---

### 1.2 The contender — exact architecture pin

**Pinned: `DeltaNetLM` (`DeltaNetLMMixer` + `DeltaNetLMBlock`,
`matrix-thinking/deltanet_rd/lm_pretrain_rd.py:563/994/1038`) with the
frozen-bias fix active, arm `"per_token"` (Arm 2, the FROZEN_BIAS_LM_DESIGN.md
primary arm), λ=0.58.**

**Mechanism, verified against the running code (`lm_pretrain_rd.py:178-246`,
`:849-867`):** inside `DeltaNetLMMixer.forward`, the raw key projection is
blended with a frozen, never-trained per-token bias table before entering
the delta-rule recurrence:

```
k_biased = normalize((1 - λ) * k_raw + λ * B[token_id])   # arm="per_token"
```

`B` is a `(vocab_size, d_state)` buffer built once per layer from
`build_frozen_bias_table` (`random_unit_rows_init`, `requires_grad_(False)`
— a real `nn.Module.register_buffer`, confirmed never touched by the
optimizer) and applied uniformly to every token position, not gathered or
selected. This deliberately supersedes the earlier Track-B key-anchoring
mechanism (`model_rd.py`'s `anchor_active REQUIRES geo3_active` hard
assert), which required a closed ~107-entity synthetic vocabulary and hit
two fatal barriers at LM scale: β-uniformity (Gini≈0.099, no
write-worthy subset to select) and Newton-Schulz duplicate-key instability
(`skip_rate=0.6319` vs. the ≤0.01 bar). The frozen-bias fix strips out
geo3/Newton-Schulz/selection entirely and keeps only the "constancy in the
blend" component identified as load-bearing in `KEY_ANCHORING_DESIGN.md`
§10.13.4.

**Constant inference-memory state — the property axis 2 (§1.4) leverages.**
`DeltaNetLM`'s fast-weight state is a `(d_state × d_state)` matrix per
layer, held constant in size regardless of context length T: at
d_state=64, that's `64×64×4 bytes = 16,384 bytes/layer` (fp32; 8,192 in
bf16). **This per-layer figure is NOT the MATCH-GATE quantity (M2, Rev
1)** — the byte budget a memory-matched Transformer's KV-cache cap must
be pinned against is the TOTAL across all `n_layers` layers (`32,768`
bytes at rung-1's `n_layers=2`), derived in §1.3(b) and checked
end-to-end by §1.7's MATCH-GATE.

**Why this is the right contender, stated with the honest caveat this
program's own rules demand:**

1. **Trainability at scale is proven for the base architecture.** Track C
   trained `DeltaNetLM` cleanly at four points — 14M (14,048,896 params,
   `d_model=256/n_layers=2/d_state=64`), 98M (97,618,176,
   `d_model=768/n_layers=12/d_state=64`), 392M (391,869,440,
   `d_model=1536/n_layers=16/d_state=128`), 1.31B (1,311,135,488,
   `d_model=2560/n_layers=22/d_state=128`). **m1 fix (Rev 1):** the 14M
   point's config is `frozen_bias_lm_sweep.py`'s `RUNG1_CFG` (identically
   `run_lm_rd_trackc_sweep.py`'s `CONTROL_CFG`) — `{d_model=256,
   d_state=64, n_layers=2, conv_size=4, num_heads=1}` — NOT
   `matrix-thinking/deltanet_rd/lm_rd_rung_configs.py`'s `RUNGS` table,
   which holds only Track C's own 98M/392M/1.31B rungs under its
   colliding "rung 1/2/3" numbering (see the collision note above); the
   14M point's param count is verified by `lm_pretrain_rd.py`'s own
   smoke item [1b] (`m = DeltaNetLM(50257, d_model=256, d_state=64, ...)`,
   a real `numel()` sum), not by `lm_rd_rung_configs.verify_param_count`
   (which cannot target a rung absent from its own `RUNGS` dict). The
   98M/392M/1.31B points ARE verified via `lm_rd_rung_configs.py`'s
   `RUNGS`/`verify_param_count`, exactly as Rev 0 stated.
   The 1.31B cell self-terminated at 84.7% of its token budget on a
   **timeout-miscalibration bug**, not an architecture failure (loss
   curves clean, `skip_rate=0.0`, checkpoints healthy through step
   155,000; `EXPERIMENT_LOG.md` rung-3 harvest entry).
2. **The fix itself removes the ONLY known instability path** at LM scale
   (the Newton-Schulz/selection failure above), which is why it — not the
   raw un-fixed key-anchoring mechanism — is the correct variant to scale
   up for a comparison this expensive.
3. **The honest caveat, stated plainly, not buried:** the frozen-bias fix's
   OWN training evidence is thin. It has only ever been trained at **14M**
   (`FROZEN_BIAS_LM_DESIGN.md` §6.1, the design's own "rung-1" — note this
   collides in name, not in config, with Track C's own "rung-1"=98M; this
   doc uses raw param counts everywhere specifically to avoid that
   collision). Its rung-2 (98M, in that design's numbering) was **PARKED,
   never launched**. And its one real result is **FOURTH OUTCOME,
   "sim-training divergence"** — DESCRIPTIVE TIER ONLY (the blind-pin was
   written post-hoc, forfeiting confirmatory license) — the primary
   mechanistic probe (`span_frac`, post-blend) moved **opposite** every
   sim-derived prediction (openr1 +0.1955 [0.0937,0.2973], wikitext
   +0.2273 [0.0926,0.3621], CI excludes zero but in the destabilizing
   direction). The one thing that DID hold cleanly: the **val-loss
   capability gate passed in both corpora** (Arm2 2.1184/4.3426 vs.
   ceilings 2.1935/4.3828) — no downstream quality regression from
   applying the fix, even though the fix's own internal mechanistic story
   is unresolved.
4. **What follows from (3):** this design doc is explicitly the FIRST wave
   to ask whether the fixed architecture wins on downstream TASK
   performance — a question `FROZEN_BIAS_LM_DESIGN.md` never posed (it
   only ever measured a mechanistic write-geometry probe, not task-level
   quality or capability). We are not re-litigating the frozen-bias
   mechanism story; we are using the one config it proved *doesn't break
   anything* as the contender, and testing whether the resulting model,
   at scale, does anything a matched conventional model can't.

**Out of scope for the primary matrix (§1.3):** an ON/OFF frozen-bias
ablation (contender-with-fix vs. contender-without-fix) is a natural,
cheap follow-on given #3 above, but adding a 4th arch breaks the clean
3-way matching structure. Registered as an explicit candidate for a
follow-on wave (§1.9 item 6), not built here.

---

### 1.3 The baselines and comparison axes — the make-or-break decision

**Mid-drafting PI directive absorbed here (2026-07-08, before first
commit).** The original brief specified two co-equal baselines — a
param-matched flat-vector ablation and a FLOP-matched standard Transformer
— treated as two independent primary comparisons. The PI's directive
supersedes that framing: **FLOP-matched is demoted to a disclosed control
("today's scarcity caveat"), not a primary.** The two axes now promoted to
primary are chosen for the constraints expected to bind in the future
(compute grows fastest; quality data and HBM are the coming walls), not
the constraint (raw FLOPs) that dominates today. The param-matched
flat-vector ablation is explicitly **kept, unchanged, as the mandated
`CLAUDE.md` control** — and it turns out to already BE axis 1's baseline
by construction (same params, and — once run on the same data budget —
already data-matched too), so nothing is lost, only reframed and
re-purposed for a sharper, more future-relevant win-margin.

Neither the flat-vector ablation nor the standard-Transformer baseline
exists in the `deltanet_rd` lineage today (checked:
`grep -rl "flat.vector\|vector_baseline"` and `grep -rl "class.*Transformer\|GPT2"`
across `matrix-thinking/deltanet_rd/` and `matrix-thinking/src/` returns zero
hits for a matched implementation of either — only unrelated legacy
`matrix-thinking/scripts/` files and the GPT-2 tokenizer import). Both must
be built fresh, which is exactly why the matching arithmetic gets its own
independent gate (§1.7, MATCH-GATE) before any GPU cell — now covering
THREE matched quantities (params, FLOPs, and inference-memory bytes), not
two.

**Three architectures total, each used in the specific role(s) below —
this does NOT grow the arch count from the original 3-arch plan, it
reuses the Transformer arch in two evaluation configs instead of adding a
4th architecture:**

**(a) Param-matched flat-vector ablation — the mandated `CLAUDE.md`
control, and axis 1's baseline.**

Construction unchanged from the original brief: identical embedding
table, output head, and FFN blocks to the contender; the ONLY change is
the fast-weight mixer. The contender's `DeltaNetLMMixer` recurrence is a
rank-1 (Householder-style) delta-rule update on a `(d_state × d_state)`
matrix state, `S_t = S_{t-1}(I - β k_t k_t^T) + β v_t k_t^T`. The ablation
replaces this with a same-parameter-count **vector-state** gated-linear
recurrence (diagonal/elementwise gate, `s_t = s_{t-1} ⊙ g_t + v_t`,
`s_t ∈ R^{d_state}`) — per the `CLAUDE.md` reshape-equivalence rule ("any
d²-dim vector can be reshaped to a d×d matrix and vice versa; structure
only matters if OPERATIONS preserve it"), the ablation does NOT reshape
the same matrix operator into a vector shape (that would silently
preserve the structure it's supposed to remove) — it replaces the
multiplicative outer-product state UPDATE itself with an additive/gated
one, at matched total parameter count via mixer width. Total param-count
match target: within ≤1% (§1.7 MATCH-GATE).

**Native-read parity (F1, added Rev 1 — see §1.3.1 for the full probe
spec this feeds).** The ablation gets its own `q_proj`/`q_conv1d`,
constructed identically to the contender's (`DeltaNetLMMixer.__init__`'s
`self.q_proj = nn.Linear(d_model, d_state, bias=False)` +
`self.q_conv1d = ShortConvolution(d_state, conv_size, ...)`, same shapes,
same init) — the contender's mixer already has a real, learned `q_proj`
(§1.2's architecture pin: LM mode needs per-token `o_t` at every
position), so giving the ablation the identical projection is a
zero-asymmetry addition, not a new capability invented for the ablation
alone. The ablation's own native per-token read, analogous to the
contender's kernel-internal `o_t = read(S_t, q_t)`, is the vector-state
elementwise analogue: `o_t = s_t ⊙ q_t` (Hadamard product, the vector
recurrence's own "matrix-vector product" is a Hadamard product by
construction, since `s_t` has no off-diagonal structure to contract
against). **`d_state` is PINNED EQUAL to the contender's own `d_state`
(64 at rung-1, 128 at the escalation rung) — a deliberate Rev-1 design
choice, not left to fall out of the width-matching arithmetic** — so the
ablation's native tap is exactly `value_dim`-dimensional with no adapter
capacity mismatch against the contender (§1.3.1's per-arm adapter
symmetry argument). Any further parameter-matching slack needed to hit
the ≤1% total-param target (§1.7 MATCH-GATE) is absorbed by the mixer's
OTHER weights (the gate projection producing `g_t`), never by `d_state`
itself.

**(b) Standard Transformer — TWO evaluation roles from ONE trained arch
per seed (the cost-saving reuse the PI directive's re-framing enables).**

A vanilla GPT-2-style causal decoder-only Transformer (standard softmax
attention + FFN), same GPT-2 tokenizer/vocab (50257, shared with every
other arch and corpus in this program — `grammar_rd.py`'s
`load_gpt2_tokenizer`), trained ONCE per seed with full/uncapped
attention, matched to the contender's **training FLOPs** (the standard
`6ND` estimate, N=non-embedding params, D=tokens, adjusted for
`DeltaNetLM`'s own measured per-token FLOP profile — reuses
`DELTANET_REALDATA_DESIGN.md` §4.2's already-verified FLOP-derivation
method **at the 14M/rung-1 scale it was derived at; §1.9 m2 flags that
this basis needs re-derivation, not reuse, at 392M**) and matched
**training data** (same corpora, same token count, same step count —
`CLAUDE.md`'s "same dataset for ALL experiments" rule applied literally:
`wikitext-mix-ext` + `openr1-mix-ext`, the exact two corpora
`FROZEN_BIAS_LM_DESIGN.md` used). This one trained model is then evaluated
under two DISTINCT inference-time attention configurations, which is
where its two roles diverge (no extra training cost either way):

**Baseline recipe and tuning-budget parity (M3, added Rev 1 — closes the
attack's "#1 hostile-reviewer rejection vector").** Pinned "modern-strong"
recipe, not a strawman vanilla GPT-2 clone:
  - **Positional encoding: RoPE** (rotary, applied to Q/K inside
    attention), not learned-absolute. Two independent reasons, both
    load-bearing: (i) "modern-strong" per this finding's own text: RoPE
    is the current standard for causal decoder-only Transformers, unlike
    GPT-2's original learned-absolute table; (ii) axis 2 (§1.4.2) evals
    at horizons out to **8× `cap_length`**, which can exceed the longest
    sequence length ever seen in training — a learned-absolute table has
    a hard, non-extrapolating position ceiling at its trained max length,
    which would silently confound a LOSE on axis 2 with "ran out of
    position embeddings" rather than "the capped cache genuinely can't
    retain the information." RoPE's relative-position construction does
    not have this failure mode by construction (still evaluated
    empirically at the calibration gate, §1.7, not assumed).
  - **Pre-norm** (already stated in Rev 0), standard residual placement,
    matching the contender's own pre-norm `DeltaNetLMBlock` convention.
  - **FFN: reuses the contender's own `FFN` class verbatim**
    (`lm_pretrain_rd.py:549`, plain 2-matrix GELU MLP, no gate, `mult=4`)
    rather than a SwiGLU 3-matrix FFN. Justified two ways: (i) this
    removes one entire axis of arch-vs-arch asymmetry — the Transformer
    and the contender share the literal same FFN sublayer class, so any
    measured difference is attributable to the mixer (attention vs.
    delta-rule), not to a stronger/weaker FFN; (ii) the contender's own
    docstring already rejected SwiGLU at this param budget ("a SwiGLU-
    style 3-matrix FFN would overshoot [the ~590K/layer target] by
    ~50%") — giving the Transformer a SwiGLU FFN the contender itself
    was deliberately built without would be an unpinned asymmetry, not a
    fairness improvement.
  - **Standard init:** GPT-2-convention embedding init `std=0.02`
    (matching the contender's own AUDIT FIX-2, `lm_pretrain_rd.py`'s
    documented correction from PyTorch's `N(0,1)` default), tied
    input/output embedding (matching the contender's own tied head,
    §1.2's cost-model assumption).
  - **Equal-tuning-budget rule, split into an LR grid (gridded) and a
    warmup POLICY (pinned, not gridded — M3's own text distinguishes
    "same LR grid size" from "same warmup POLICY").** Warmup: 100 steps,
    IDENTICAL for every arm (contender, ablation, Transformer) — the
    contender's own already-validated value
    (`lm_pretrain_rd.py --warmup-steps 100` default), reused verbatim, no
    per-arm search. LR: the contender's LR is ALREADY known-good and
    fixed by `FROZEN_BIAS_LM_DESIGN.md`/`lm_pretrain_rd.py`'s own default
    (`--lr 3e-4`, AdamW) — disclosed as the contender's grid center, an
    ungridded, already-validated setting (its search cost was already
    paid in a separate, already-completed prior wave, not re-spent here).
    The Transformer, as a genuinely new architecture in this codebase,
    gets an EQUALLY-SIZED **3-point LR grid** `{1e-4, 3e-4, 1e-3}`
    centered on that same published-standard AdamW LR — evaluated ONLY at
    the calibration cell (§1.7 item 1), never re-swept inside the
    27-cell main sweep. The grid winner (lowest calibration-cell
    val-loss) is FROZEN before the main sweep launches, exactly like
    axis-1's `X` and axis-2's `cap_length` (§1.7 item 1's "both BEFORE
    the full sweep launches, not after" discipline). Costed explicitly
    in §1.6's Wave −1 table (item E).

  - **Role (b-control): full/uncapped attention** — the disclosed,
    non-gating FLOP-matched control (§1.1). Reports today's-compute-regime
    comparison honestly alongside whatever the primary axes show.
  - **Role (b-primary, axis 2's baseline): hard-capped KV cache**, capped
    at an **M×-multiplier** of the contender's **TOTAL, across-all-layers**
    fixed matrix state (**M2 fix, Rev 1, byte-basis unchanged; M-sweep
    parametrization F-NEW-1 fix, Rev 2** — Rev 0 left per-layer vs.
    total ambiguous; total-across-layers is the defensible quantity,
    because it is what a deployed instance of the contender actually
    holds in memory end-to-end, and per-layer parity alone would silently
    unbalance total memory whenever the two archs' `n_layers` differ,
    which they generally will once param-matching picks each arch's own
    depth/width split). At rung-1, contender total state bytes =
    `n_layers_contender (2) × d_state² (64×64) × 4 bytes (fp32) = 32,768
    bytes`, not the 16,384-bytes/layer figure Rev 0 quoted (that number
    is the correct PER-LAYER figure, §1.2, but §1.7 MATCH-GATE now checks
    the TOTAL, not the per-layer, quantity). **Rev 2 (F-NEW-1): the single
    `M=1` (exact-byte-parity) cap is replaced by a geometric
    memory-multiplier sweep, `M ∈ {1,2,4,8,16,32}`, capping at `M ×
    32,768` bytes** — full derivation, the pinned accounting dtype, and
    the crossover-multiplier decision rule are in §1.4.2 (the single-point
    version was independently found DEGENERATE at rung-1: §1.15 F-NEW-1).
    Eviction policy, for every `M`: `2 (K&V) × n_layers_transformer ×
    n_heads × d_head × cap_length(M) × bytes_per_elt = M × 32,768`, solved
    for `cap_length(M)` (in tokens) once `n_layers_transformer`/
    `n_heads`/`d_head` are fixed by the Transformer's own param-matching
    build, over a **sink+FIFO** window (**F-NEW-2 fix, Rev 2**): the first
    `k_sink=4` tokens of the sequence are ALWAYS retained, and the
    remaining `cap_length(M) − k_sink` slots hold the most-recent tokens
    on a FIFO basis (one-line rationale: StreamingLLM, Xiao et al. 2023,
    showed a handful of early "attention sink" tokens absorb
    disproportionate attention mass in models trained with full/uncapped
    attention, and dropping them under a naive sliding window causes
    catastrophic, distribution-shifted degradation independent of the
    cap's actual information content — retaining them is the standard,
    minimal, still-simple mitigation). Deliberately still the SIMPLEST
    policy that addresses the disclosed train/eval mismatch (§1.9 item
    11), not a sophisticated KV-compression or attention-score-based
    scheme (§1.9 item 5) — this is the fair, conservative comparison, not
    a strawman tilted further toward the contender than the byte-budget
    match alone already tilts it. **Disclosed, not fully closed (F-NEW-2,
    Rev 2):** the sink patch is a PARTIAL mitigation for the checkpoint
    being trained fully uncapped and only capped at inference; the full
    fix — a genuinely cap-trained `(b)` arm — is registered as a costed,
    explicitly unaffordable-this-wave follow-on (§1.9 item 7).

**Matching arithmetic table, rung-1 (14M tier) — to be verified for real
by MATCH-GATE (§1.7) before any cell launches, not assumed from this
table. Inference-memory column corrected Rev 1 (M2): TOTAL across all
layers, not per-layer.**

| Arch | Params (target) | Training FLOPs | Training data | Inference-memory (fixed-state bytes, TOTAL across all layers) |
|---|---|---|---|---|
| Contender (`DeltaNetLM`, frozen-bias `per_token`, λ=0.58) | 14,048,896 (measured, `d_model=256/n_layers=2/d_state=64`) | measured per-token profile × steps × batch × seq_len (pin once) | `wikitext-mix-ext` + `openr1-mix-ext`, same step count as (a)/(b) | 32,768 = 2 layers × 64×64 × 4 bytes (fp32; CONSTANT in context length T) |
| (a) Flat-vector ablation | matched to contender within ≤1% | falls out (reported) | same corpora, same step count (data-matched by construction — axis 1's premise) | 32,768 = 2 layers × 64 (`d_state`, pinned equal to the contender's, §1.3(a)) × 4 bytes — reported, not the axis-2 comparison arm |
| (b) Standard Transformer | falls out (reported) | matched to contender within ≤5% | same corpora, same step count | role (b-control): grows `O(T)` uncapped, reported at eval horizon; role (b-primary): hard-capped so TOTAL K+V bytes across all layers/heads = `M × 32,768`, swept over `M ∈ {1,2,4,8,16,32}` (`cap_length(M)` solved from `n_layers_transformer/n_heads/d_head`, pinned by MATCH-GATE, sink+FIFO eviction — F-NEW-1/F-NEW-2, Rev 2, full derivation §1.4.2) |

The ≤1% / ≤5% / exact-byte-match tolerances are deliberately tighter than
the 15% rung-config planning tolerance in `lm_rd_rung_configs.py` — that
tolerance governs whether a scale-ladder point is "close enough to its
target" for an unrelated study; here the match arithmetic between arms IS
the finding, so it needs its own, much tighter bar, with the
inference-memory byte match held EXACT at every `M` (an integer token cap
solved from the contender's real measured TOTAL byte count, not a rounded
approximation, and not the per-layer figure — M2).

---

### 1.3.1 THE WAVE −1 SHARED PROBE HEAD (F1, FATAL — new in Rev 1)

**The gap this closes.** Rev 0's `recovered_frac@0.9` instrument had no
architecture-neutral implementation. The only LIVE code computing it
(`key_anchoring.py:953` `measure_h1_behavioral_companion`, calling
`model_rd.py`'s `model(b, force_rank_k=None)` 5-tuple forward) is welded
to `model_rd.py`'s Wave-1 `DeltaNetRDBlock` — a FOURTH architecture, not
any of this design's three arms, and one whose "readout" (`readout()`,
`model_rd.py:1194`) is a **zero-parameter analytic operation**
(`apply_state_power(S_T, q_eff, hops)`, reusing the SAME `k_proj` weights
`bind()` used) that only makes sense because `model_rd.py` has no
per-token `q_proj` at all and never computes a trained per-token output.
None of that transfers: the contender DOES have a real, learned `q_proj`
(§1.2), the flat-vector ablation has no matrix to "power," and the
Transformer has neither a state matrix nor a `k_proj`-reuse trick. This
section designs ONE new, trained, architecture-neutral probe from
scratch — to the level a build agent can implement without further
design decisions — reusing this program's own validated PRIMITIVES
(`random_unit_rows_init`, the cosine-recovery metric, the frozen-buffer
discipline) rather than the dead forward-signature pattern itself.

**1.3.1.1 The frozen target table (replaces `model_rd.py`'s `v_eff_items`
target).** `model_rd.py`'s `targets` (line 1257) are gathered from
`v_eff_items` — the answer entity's value representation under THAT
model's own, currently-training `W_v`. Reusing that pattern here would
weld the ground truth to one arch's own learned weights, which is exactly
the kind of arch-specific coupling F1 forbids. Instead:

```
value_dim = d_state of the contender AT THIS RUNG (64 at rung-1, 128 at
            the escalation rung — re-pinned per rung, never assumed fixed)
PROBE_TARGET_SEED = 20260709   # NEW, deliberately DIFFERENT from
                                # ANCHOR_INIT_SEED (20260705, key_anchoring.py) —
                                # never share a seed between the frozen KEY-bias
                                # table (§1.2) and this frozen VALUE-target table;
                                # a shared seed would silently correlate the two
                                # frozen structures and confound the frozen-bias
                                # mechanism's own probe.
T_val = random_unit_rows_init(vocab_size_total, value_dim, seed=PROBE_TARGET_SEED)
        # (key_anchoring.py:200 — reused UNMODIFIED, the exact same frozen-random-
        # unit-rows construction already audited for the frozen-bias table, applied
        # to a NEW, independent seed/purpose. requires_grad_(False), registered as
        # a buffer. ONE table, shared byte-identically across ALL THREE arms and
        # ALL THREE seeds — the target space is a property of the TASK, not the model.)
target(entity_token_id) = T_val[entity_token_id]     # (value_dim,), fixed ground truth
```

This is the same frozen-random-unit-row primitive `CLAUDE.md`'s own Hard
Rules already bless for exact-continuous recovery (never argmax/codebook)
— reused for a new purpose (probe target, not key bias), not reinvented.

**1.3.1.2 Per-arm native tap → adapter → ONE shared probe head.**

```
shared_probe = nn.Linear(value_dim, value_dim, bias=True)   # IDENTICAL
    # module, IDENTICAL init (PyTorch default nn.Linear init — no special
    # scaling), IDENTICAL LR (folded into each arm's own optimizer at the
    # SAME pinned aux weight, below), across all three arms. A linear
    # probe is the right complexity match, not an arbitrary simplification:
    # this program's own zero-param analytic precedent (model_rd.py's
    # `apply_state_power(S_T, q_eff, hops)` for h=1 IS `S_T @ q_eff`, a
    # LINEAR map) already established that a correctly-functioning h=1
    # read is linear in the query; a probe with strictly more expressive
    # power than the ground-truth relationship it is meant to reveal would
    # let the PROBE, not the backbone's state, do the compositional work —
    # exactly the confound this section's Wave −1(B) null check is
    # designed to catch. An MLP probe is NOT built this wave (registered
    # as an §1.9 follow-on if the linear probe's own capacity sanity check,
    # 1.3.1.4, itself fails to separate signal from the null).

adapter_arm = nn.Linear(native_tap_dim_arm, value_dim, bias=False)  # ONE
    # per arm, SAME class/init/LR as every other arm's adapter — the ONE
    # place this design cannot force literal parameter-count identity
    # across arms (native_tap_dim differs by construction: it's exactly
    # what "param-matched via width, not via state dim" MEANS). Every arm
    # gets an adapter, even when native_tap_dim_arm == value_dim already
    # (contender, ablation — see §1.3(a)'s d_state-pinning fix), so no
    # arm's code path is special-cased to skip it. The adapter is
    # DELIBERATELY LINEAR-ONLY (no nonlinearity, no hidden layer) so it
    # cannot itself absorb representational work beyond dimensionality
    # normalization — closing exactly the "does a fatter adapter secretly
    # do the work for one arm" version of the probe-parity confound
    # (verified by 1.3.1.4's null, which exercises every arm's REAL
    # adapter shape, not a placeholder).
```

Per-arm native tap (`state_summary_raw`, fed into that arm's adapter):

- **Contender (`DeltaNetLM`).** `native_tap_dim = d_state` (64 at
  rung-1). Two pieces, computed WITHOUT ever letting the query touch the
  recurrence (preserving the P=1 hard-bottleneck discipline, `CLAUDE.md`):
  1. `S_T_last = final_states[-1]` from `DeltaNetLM.forward(...,
     return_states=True)` (`lm_pretrain_rd.py:1178-1205`) — the LAST
     layer's `(B,H,head_dim,head_dim)` state after streaming the BIND
     phase (+ filler, for axis 2's long-horizon setting) ONLY. The query
     window is NEVER appended to this forward call.
  2. `q_query = ` the query window `[buf...,KEY,REL,<Q>]` (`grammar_rd.py`'s
     `query_tokens`, shape `(B,Q,query_len)`) run through the LAST
     block's OWN `embed → norm1 → q_proj → q_conv1d` pathway ONLY — a
     new function mirroring `model_rd.py`'s `effective_key_window`
     exactly (`grammar_rd.py`'s own module docstring states the
     precedent this generalizes: "the query span passes through the
     feature path only, never the recurrence"), generalized to use the
     contender's real `q_proj`/`q_conv1d` (which `model_rd.py` does not
     have) instead of `model_rd.py`'s `k_proj`-reuse hack. Output: the
     LAST position's (`<Q>`'s) post-conv `q` vector, `(B,Q,head_dim)` per
     head.
  `state_summary_raw = S_T_last @ q_query` (per-head matrix-vector
  product, concatenated across heads — num_heads=1 at every registered
  rung config, so this is exactly `(B,Q,d_state)` with no concat step
  needed in practice, but the formula generalizes). This is architecturally
  a genuine function of `(S_T, q_query)` alone — never of raw bind-phase
  tokens directly — satisfying the P=1 bottleneck by construction, the
  same way the kernel's own `o_t = read(S_t, q_t)` already does for
  next-token prediction (§1.2's architecture description).
- **Flat-vector ablation.** `native_tap_dim = d_state` (64, pinned equal
  to the contender's, §1.3(a)). `state_summary_raw = s_T ⊙ q_query`
  (Hadamard), `q_query` derived via the ablation's OWN `q_proj`/`q_conv1d`
  on the query window, same non-recurrent discipline as the contender.
- **Standard Transformer.** `native_tap_dim = d_model_transformer`
  (whatever value the param/FLOP-match arithmetic yields, generally
  `!= value_dim` — this is the one arm whose adapter does real
  dimensionality work, disclosed, not hidden). `state_summary_raw =` the
  final-block hidden state at the `<Q>` position from a SINGLE forward
  pass over `[BIND-phase tokens (+ filler for axis 2) ++ query_window]`,
  under whichever attention config the eval role calls for (uncapped for
  b-control, hard-capped-KV for b-primary). **Disclosed, structural
  asymmetry (flagged in the new §1.9 item 8, not fixable):** this is the
  ONE arm where the query DOES enter the same forward pass as the
  context, because a Transformer has no non-attention "read channel" —
  attending IS how it reads. The contender/ablation's query is
  deliberately kept OUT of their recurrence (above) to protect the P=1
  bottleneck; the Transformer has no equivalent bottleneck to protect
  (§1.4.2's own text already says this: "it has no fixed state; its tap
  is the hidden at query time, and axis 2's cap is what constrains its
  memory").

`pred = shared_probe(adapter_arm(state_summary_raw))`, scored against
`target(answer_entity_token_id)` via the SAME `F.cosine_similarity(pred,
target, dim=-1)` thresholded at 0.9 (`recovered_frac@0.9`) Rev 0 already
pinned — the metric formula is unchanged; only the machinery producing
`pred` and `target` is new.

**1.3.1.3 Training regime — pinned per F1: joint auxiliary loss, from
step 0, never zero-shot post-hoc.** Every arm's `shared_probe` +
`adapter_arm` parameters are optimized JOINTLY with the backbone by the
SAME optimizer, added to the main next-token cross-entropy loss:

```
loss_total = loss_CE + aux_weight * mean_over_queries(1 - cos(pred, target))
```

`aux_weight`: pinned default **0.1**, REVISED by the calibration cell
(§1.7 item 1, extended) if the two losses' gradient norms at the tap
point differ by more than 10× at step 500 (mirrors this program's own
calibration-revises-a-default convention, e.g. axis-1's `X`) — frozen
before the 27-cell sweep launches, not after. The probe trains
THROUGHOUT training (from step 0), consistent with §1.4's own
already-stated premise ("every task below trains the model end-to-end
WITH the readout as (part of) the actual training objective") — Rev 0
stated this premise but never built the mechanism; this subsection is
that mechanism.

**1.3.1.4 Probe-capacity sanity null (the MANDATORY control F1
requires).** Before the shared probe is trusted for ANY arm's real cells:
train `adapter_arm + shared_probe` (each arm's REAL adapter shape) on
FROZEN, RANDOM `state_summary_raw` vectors (i.i.d. Gaussian, no backbone
forward pass at all — the null does not need a real model, only the
probe+adapter's own shapes) against the SAME frozen `T_val` targets. **Null
protocol, pinned (m-new-2, Rev 2 — Rev 1 left fresh-vs-fixed-pool
ambiguous, and a fixed pool risks a memorization false-pass):** draw a
FRESH i.i.d. `state_summary_raw` batch every training step (never reuse a
fixed pool of pre-drawn vectors across steps), and evaluate
`recovered_frac@0.9` on a SEPARATE held-out set of fresh draws never seen
during the null's own training — mirrors this program's standing
train/eval-split discipline, applied to a synthetic-input null exactly as
it would be to a real task. Pass condition: `recovered_frac@0.9` on the
held-out fresh draws stays statistically indistinguishable from chance
(chance ≈ 0 for cosine recovery of a random unit vector in `R^64`; pass
bar set at `< 0.05`, i.e. the probe alone, with no real state to read and
no fixed pool to memorize, must not "solve" the task). FAIL on this null
is a hard launch-block for the responsible arm's probe design — this is
what closes the F1-mandated "probe-parity confound... closed by
construction" requirement with an actual negative test, not an assertion.
Cost: negligible (no backbone), see §1.6 Wave −1 item B (item B's scope
extended, Rev 2, to also run §1.3.1.5's cross-dimension tap diagnostic —
same CPU-feasible, no-backbone harness, no separate cost line).

**1.3.1.5 Synthetic cross-dimension tap diagnostic (M-NEW-2, Rev 2 — new,
CPU-only/negligible-GPU, reuses §1.3.1.4's frozen-synthetic-state
harness).** Closes the gap self-attack item 10 names: the contender's
native tap is a full matrix-vector product `S_T @ q` (every output
dimension is a function of every input dimension of `q` — full
cross-dimension mixing), while the flat-vector ablation's native tap is a
Hadamard product `s_T ⊙ q` (diagonal-only — output dimension `i` is a
function of input dimension `i` alone, structurally). This is an
irreversible expressivity gap the shared linear `adapter_arm`/
`shared_probe` (§1.3.1.2) CANNOT repair (a linear map fed a
diagonal-only-derived vector cannot recover information that was never
routed through cross-dimension terms in the first place), and it is NOT
exercised by §1.3.1.4's null (which uses i.i.d. Gaussian inputs with no
adversarial diagonal/off-diagonal structure at all). This diagnostic
bounds — not eliminates — how much of any REAL axis-1 gap tap
expressivity alone could explain:

```
for alpha in {0.0, 0.25, 0.5, 0.75, 1.0}:        # diagonal-energy fraction
    q      = unit vector, R^{d_state}            # fresh draw, independent RNG
    t      = unit vector, R^{d_state}             # fresh target, seed != PROBE_TARGET_SEED,
                                                    # seed != ANCHOR_INIT_SEED (never reuse a
                                                    # frozen-table seed for a diagnostic input)
    D      = diag(alpha * t / q)                  # elementwise solve: D @ q == alpha * t exactly
                                                    # (guard |q_i| >= eps via the same
                                                    # random_unit_rows_init-style draw that
                                                    # already avoids axis-degenerate vectors)
    O      = ((1 - alpha) * t) @ q^T / (q . q)     # rank-1 solve: O @ q == (1-alpha)*t exactly
    O_offdiag = O - diag(diag(O))                  # zero O's own diagonal; keep only cross-dim terms
    S_star = D + O_offdiag                         # synthetic state: exact diag/off-diag energy split
    target_effective = S_star @ q                  # the REAL, exactly-achievable target for THIS S_star
                                                     # (recomputed after zeroing O's diagonal, so the
                                                     # construction is exact, never approximate)
    hadamard_pred = diag(S_star) . q                # = D . q = alpha * t, by construction
    matvec_pred   = S_star @ q                      # = target_effective, by construction (exact)
    score both preds against target_effective via the SAME F.cosine_similarity @ 0.9 metric
```

At `alpha=0`, `D=0` by construction: the Hadamard tap is mathematically
guaranteed `recovered_frac@0.9 = 0` (it reads zero diagonal, always) while
the matvec tap is guaranteed `recovered_frac@0.9 = 1` (it reads the exact
target by construction) — the worst-case, maximal-adversarial bound (a
100-percentage-point gap attributable to tap expressivity ALONE, on a
construction designed to defeat a diagonal-only reader). Sweeping
`alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}` traces a **calibration curve**
(`alpha → Hadamard-tap recovered_frac`, monotonically increasing from 0 to
1) that a real ablation's measured axis-1 Hadamard-tap `recovered_frac`
can be read against, at analysis time, to report an implied "as-if diagonal
fraction" for the ablation's own trained states — a genuine, quantitative
bound on the tap-expressivity confound's contribution to any observed
axis-1 gap, not merely a qualitative disclosure. **What this diagnostic
does NOT do:** it does not measure the REAL ablation's actual diagonal
fraction (that would require a separate post-hoc measurement on real
trained states, out of scope this wave, registered as a cheap potential
follow-on); it only establishes the theoretical mapping a real number can
be read against. Cost: CPU-only (linear algebra on `d_state`-dimensional
vectors, no backbone forward pass, no training loop), folds into §1.6
Wave −1 item B at no additional GPU-h.

**F1b — Tasks 1/2 data-matching (compounding finding, resolved here).**
Rev 0's §1.3(b) same-data prescription covered only Task 3's static
corpora; Tasks 1/2 use `grammar_rd.py`'s on-the-fly `sample_batch_rd`
generation, which was left unpinned. Fix, mirroring
`reasoning_link_probe.py`'s own `episode_seed` mixed-radix precedent
(`PURPOSE_BASE`/`LEG_BASE`/`STRIDE_*`, collision-free by construction,
each digit's stride exceeding the previous digit's max reach):

```
# TASK_BASE widened to 5 keys (M-NEW-1, Rev 2) -- Rev 1's 3-key table had
# no task1_stress / task2_calib slot, yet Wave -1 items C/D (sec 1.6) need
# them; a build agent facing a missing key would either invent one or
# silently reuse task2_sweep's base, colliding calibration with sweep
# seeds -- exactly the bug class F1b exists to close.
TASK_BASE = {
    "task1_calib":  0,
    "task1_stress": 500_000,
    "task1_sweep":  1_000_000,
    "task2_calib":  1_500_000,
    "task2_sweep":  2_000_000,
}
STRIDE_SEED_RD = 10_000   # per-seed-index stride (n=3 seeds, extendable to 12 -- 11*10_000=110_000 < 500_000 margin between adjacent TASK_BASE keys)
STRIDE_STEP_RD = 1        # per-checkpoint-index stride (raw checkpoint index, not raw step count)

def rd_episode_seed(task: str, seed_idx: int, ckpt_idx: int) -> int:
    assert task in TASK_BASE, f"unknown task key {task!r}"
    assert 0 <= ckpt_idx < STRIDE_SEED_RD, \
        f"ckpt_idx={ckpt_idx} >= STRIDE_SEED_RD={STRIDE_SEED_RD} -- would overflow into the " \
        f"next seed_idx's own seed range (m-new-1, Rev 2 runtime assert)"
    return TASK_BASE[task] + seed_idx * STRIDE_SEED_RD + ckpt_idx * STRIDE_STEP_RD
```

**Deliberately excludes architecture/arm from the formula** — this is the
load-bearing property: for a given `(task, seed_idx, ckpt_idx)`, all
THREE arms draw from `torch.Generator().manual_seed(rd_episode_seed(...))`
and therefore see the byte-identical stream of `sample_batch_rd` episodes
(same K-cycle draws, same entity draws, same hop draws) at that
checkpoint — the on-the-fly-generation analogue of `CLAUDE.md`'s "use the
SAME dataset for ALL experiments in a comparison," operationalized for a
generator that has no file to share. Batch size and step count are
pinned identical across arms by the existing 27-cell sweep design (§1.6);
this formula is the piece Rev 0 omitted. **Collision-freedom (Rev 2:
extended to enumerate ALL FIVE `TASK_BASE` keys, M-NEW-1 — Rev 1's smoke
covered only the 3 keys that existed then)** — no two distinct
`(task,seed_idx,ckpt_idx)` triples across `{task1_calib, task1_stress,
task1_sweep, task2_calib, task2_sweep} × seed_idx∈[0,11] ×
ckpt_idx∈[0,STRIDE_SEED_RD)` mapping to the same seed — is mechanically
verified by a dedicated negative-test smoke item, mirroring
`phase2b_seedext_stage_minus1.py`'s own SE4/SE5 exhaustive-enumeration
check (`matrix-thinking/deltanet_rd/phase2b_seedext_stage_minus1.py:165-235`)
— not assumed from the arithmetic alone (`CLAUDE.md`'s own "run the
negative unit test... to completion" rule on structural checks). The
`seed_idx∈[0,11]` bound in the smoke's own enumeration matches the
seed-extension contingency's own ceiling (§1.8, n=12); the `ckpt_idx`
bound is the m-new-1 runtime assert above, exercised by its own
negative-test cell (`ckpt_idx = STRIDE_SEED_RD` must raise).

---

### 1.4 Tasks, axes, instruments, and pre-registered win margins

Both primary axes and the disclosed control reuse **already-validated,
exact-continuous-recovery instruments** from this program's own archive —
none of them touch the dead `d_state`-space zero-shot geometric-transfer
readout (`REASONING_LINK_DESIGN.md` §16.15, PROBE-INVALID/triple-null,
0.0000 recovered_frac across 30/30 readings and 3 structurally different
instrument variants — that readout targeted zero-shot geometric recovery
on a model never supervised to expose it; every task below trains the
model end-to-end WITH the readout as (part of) the actual training
objective, which is the regime where `key_anchoring.py`'s cosine-based
readout is proven, not the regime where it's dead).

**Explicitly out of scope: byte-level input.** `CLAUDE.md`'s
Byte-Agnostic lane is on hold, and the "hold tokenization... FIXED when
testing a primary hypothesis... treat the second axis as a separate,
explicitly-sequenced follow-on" rule applies directly — the
"constant-memory" framing (§1.4 axis 2) might tempt someone to also
stress-test long raw-byte sequences, but that bundles two unproven axes
(matrix fast-weight structure AND byte-level tokenization) into one
uninterpretable result. Tokenization stays GPT-2/BPE, fixed, throughout
this entire wave, both rungs.

**Task 1 — high-load / long-horizon associative recall (MQAR-style).**
Instrument: `grammar_rd.py`'s `DeltaNetRDTaskConfig` / `sample_batch_rd`,
single-hop (`hop_set=(1,)`) bind→query recall over a K-entity pool per
episode, drawn from `EntityPools` (real GPT-2 token IDs,
`_assert_injective_entities` guard — exact equality, no numerical-tolerance
slack per the `CLAUDE.md` hard rule on structural checks). Readout:
`F.cosine_similarity(pred, targets, dim=-1)` thresholded at 0.9
(`recovered_frac@0.9`) — the metric FORMULA is the same direct-continuous-
comparison primitive `key_anchoring.py`'s `measure_h1_behavioral_companion`
and `readout_rev7.py` use, but `pred`/`targets` are now produced by
**§1.3.1's shared Wave −1 probe head**, not by that function's own
`model_rd.py`-welded 5-tuple forward (F1's fix — the formula transfers,
the surrounding machinery does not). Never argmax/nearest-neighbor.

**M1 (Rev 1 re-pin) — load point.** Rev 0's HIGH-LOAD setting was
K/d=0.75, d_state=64 — the attack found this is ABOVE
`KEY_ANCHORING_SCALING_DRAFT.md`'s measured cliff (x0=0.5455), and that
measurement is from `model_rd.py`'s anchor-table/Newton-Schulz
architecture, NOT any of this design's three arms. **`model_rd.py`'s
cliff numbers carry ZERO evidentiary weight for the contender, the
flat-vector ablation, or the Transformer** — different architecture, not
merely different `d` (repeated in §1.9's self-attack, per M1). Fix: the
PRIMARY load point is re-pinned to **K/d=0.5** (K=32 at d_state=64),
comfortably below the only number this program has ever measured for
ANY architecture, with the explicit understanding that number does not
transfer here — K/d=0.5 is a conservative, load-bearing-but-not-yet-
adversarial starting point, not a "safe" point in any measured sense for
these three arms. K/d=0.75 is RETAINED as a disclosed STRESS point (not
primary) — both points are now covered by §1.3.1's Wave −1 K/d
calibration sweep (§1.6 item C), extended to **ALL THREE arms**
(including the Transformer's own capped-cache behavior at each K/d,
b-primary role), with axis-1's real win margin frozen only AFTER that
calibration returns, never assumed from this table. LONG-HORIZON setting
(axis 2): same K (32, the newly-primary load), bind clauses separated
from their query by filler/distractor tokens out past `(b-primary)`'s
`cap_length`, using `DeltaNetRDTaskConfig`'s own existing
`buf_len`/`T_bind`/`query_len` sequence-structure parameters — confirming
this parametrization already supports variable bind-query distance is a
build-time verification item, not assumed here.

**Task 2 — multi-hop compositional generalization (held-out hop depths;
"stays a primary task" per the PI directive — the binding-problem/
reasoning story).** Instrument: same `grammar_rd.py` machinery, multi-hop
(`H_train=(1,2)`, `H_test=(3,4)`), single Hamiltonian K-cycle per episode
(`_permutation_graph`, "a random SINGLE Hamiltonian K-cycle per row, NOT a
general permutation" — verbatim comment at `grammar_rd.py:262-264`), with
`DeltaNetRDTaskConfig.__post_init__`'s `h % K` guard (`:349-360`)
mechanically refusing any held-out hop whose residue coincides with a
training hop or with 0 (identity) — the exact single-full-K-cycle fix this
program's own hard rule demands, reused verbatim rather than re-derived.
Readout: same `recovered_frac@0.9`, scored at held-out hops. Feeds BOTH
primary axes (data-efficiency learning curves at matched hop depths; long
-horizon composition at hop-chains that push the query past
`(b-primary)`'s cap).

**Task 3 — real-data LM quality (secondary, unaffected by the PI
directive — "stays" per its explicit text).** Instrument:
`FROZEN_BIAS_LM_DESIGN.md`'s own pretraining + eval machinery, unchanged —
`wikitext-mix-ext` and `openr1-mix-ext` corpora, validation cross-entropy
loss (nats/token), reported as byte-BPC for internal comparison
(`CLAUDE.md`: "Use standard benchmarks for publishable claims. Byte BPC is
for internal use" — this wave's Task 3 number is internal; a future
paper-grade claim would additionally need a recognized external
benchmark, out of scope here). Evaluated for the contender vs. (a) and vs.
(b-control) — the FLOP-matched, uncapped role. **Deliberately the task
most likely to produce a LOSE, pre-registered as reportable, not hidden**
(`CLAUDE.md`: "Making matrix ops cheaper does NOT fix the quality gap.
Speed ≠ quality"). Win margin unchanged from the original brief: contender
beats both by ≥0.02 nats/token (≈0.029 BPC), CI excluding the margin; TIE
band |Δ|<0.02 nats/token vs. (b-control) specifically (the fairer
comparison for a quality claim); LOSE reported as a headline number if it
triggers, not a footnote.

---

#### 1.4.1 Axis 1 (PRIMARY) — DATA-EFFICIENCY, param+data-matched

**Comparison:** contender vs. (a) flat-vector ablation, same params
(≤1%-matched), same token budget by construction. **What's new vs. the
original brief:** instead of a single endpoint Δ, this axis measures the
full **learning curve** — `recovered_frac@0.9` (Task 2, primary; Task 1
high-load, secondary bonus at no extra cost) logged at fixed checkpoint
intervals across training, not just at the final step.

**Win margin (two operationalizations, pre-registered together — either
suffices):**
1. **Tokens-to-threshold ratio:** contender first reaches
   `recovered_frac@0.9 ≥ 0.9` at held-out hops using ≤**X%** of the token
   budget the flat-vector ablation needs to reach the same threshold. `X`
   is NOT hand-picked here — it is pinned from a power sketch run during
   the calibration cell (§1.7 item 1): using the calibration run's own
   observed curve shape and this program's n=3 seed noise floor
   (`delta_ci_n`, §1.8), compute the smallest token-ratio gap that would
   be CI-detectable at n=3, and set `X` there (Rev-0 provisional default,
   to be confirmed or revised by that power sketch before the sweep
   launches: **X=50%**).
2. **Sustained CI-separated curve:** at every matched checkpoint, the
   contender's `recovered_frac@0.9` CI (`delta_ci_n`, paired by seed) sits
   above the ablation's, for ≥3 consecutive logged checkpoints (guards
   against a single-checkpoint fluke driving the verdict).

TIE: neither operationalization fires, and the ablation doesn't beat the
contender by either measure either. LOSE: the ablation reaches threshold
first, using ≤X% of the CONTENDER's tokens, CI-detectable — i.e. structure
actively costs data efficiency, the reportable-not-hidden case.

---

#### 1.4.2 Axis 2 (PRIMARY) — INFERENCE-MEMORY-MATCHED, "the program's
strongest card" (F-NEW-1/F-NEW-2 memory-multiplier redesign, Rev 2 — the
Rev-1 single equal-byte point was independently found DEGENERATE at
rung-1, §1.15's own re-derivation, restated below)

**Why Rev 1's single point failed (kept, not hidden, per this program's
own convention of showing the wreck before the fix).** Re-deriving from
`DELTANET_REALDATA_DESIGN.md` §4.2's head-dominated FLOP method: at
rung-1 the fixed `V=50,257` head is `~92%` of FLOPs/token, which pins the
FLOP-matched (§1.3(b), ≤5%) Transformer's admissible family to
`d_model≈256`, `n_layers∈{1,2}` (deeper/narrower configs either overshoot
the FLOP band or lose the match outright). Solving `2 (K&V) ×
n_layers_transformer × d_model × bytes_per_elt × cap_length =
32,768` for `cap_length` at `M=1` (fp32, `bytes_per_elt=4`) gives
**`cap_length = 8` tokens at `n_layers=2`, `16` at `n_layers=1`** — both
below `query_len (6) + one full bind clause (clause_len=7) = 13` tokens,
against the task's own `T_bind = K×clause_len = 224` tokens at the
primary load `K=32` (`grammar_rd.py:307-332`, §1.4's M1 re-pin). The
capped baseline retains 1-2 of 32 bind clauses regardless of eviction
policy, so the contender clears the old 0.20 margin trivially — a rigged,
not a hard, test. **This does not mean the byte-parity comparison is
wrong in principle** — it means a SINGLE point at exact parity is the
wrong instrument; the fix below tests a RANGE of memory budgets and
reports where (if anywhere) the crossover sits, which is strictly more
informative for the "constant-memory minds" framing than a single
pass/fail point ever was.

**Comparison, redesigned (F-NEW-1):** contender (fixed-size matrix state,
constant in T) vs. (b-primary) the SAME trained Transformer checkpoint
(no new training — inference-only, per M-NEW-4's binding budget
constraint), with its KV cache hard-capped at **`M ×` the contender's own
total state bytes**, `M` swept over a fixed, pre-registered geometric grid
**`M ∈ {1, 2, 4, 8, 16, 32}`** (6 points — deterministic, cost-bounded;
derivation of the grid's own endpoint below). **Accounting dtype, pinned
(part of F-NEW-1's resolution):** fp32. The contender's fast-weight state
persists in fp32 OUTSIDE the delta-rule kernel's own internal boundary —
`final_state = final_state.float()` immediately after the
`chunk_delta_rule` call, `lm_pretrain_rd.py:986` — the kernel's bf16 usage
(`:978-984`) is a compute-boundary detail (`chunk_delta_rule`
"categorically rejects float32," per that file's own comment), not the
dtype a deployed instance actually HOLDS in memory end-to-end (§1.3(b)'s
own byte-parity language). The byte-parity comparison must therefore use
the dtype each side actually PERSISTS, and the contender persists fp32 —
using bf16 for the Transformer's cache while the contender's own
accounted bytes are fp32 would silently unbalance the match (M2's own
"total, not per-layer" logic, applied to dtype instead of layer count).
**bf16 reported as a disclosed secondary variant** (a bf16-cache
deployment is a realistic, if not primary-accounted, alternative): every
`cap_length(M)` value in the table below doubles under bf16
(`bytes_per_elt=2`), never the decision quantity.

**cap_length(M), in tokens, for the admissible transformer family
(`d_model≈256`, per the head-dominated FLOP match above), fp32 pinned,
bf16 disclosed:**

| M | n_layers=1, fp32 | n_layers=1, bf16 | n_layers=2, fp32 | n_layers=2, bf16 |
|---|---|---|---|---|
| 1 | 16 | 32 | 8 | 16 |
| 2 | 32 | 64 | 16 | 32 |
| 4 | 64 | 128 | 32 | 64 |
| 8 | 128 | 256 | 64 | 128 |
| 16 | 256 | 512 | 128 | 256 |
| 32 | 512 | 1024 | 256 | 512 |

(formula: `cap_length(M) = M × 32,768 / (2 × n_layers_transformer ×
d_model × bytes_per_elt)`; MATCH-GATE, §1.7 gate 6, resolves the REAL
`n_layers_transformer` from the actual param/FLOP-matching build and
computes the real table, not this illustrative `d_model≈256` sketch.)

**Grid-endpoint rule, pinned exactly (F-NEW-1).** Run the full fixed grid
`M ∈ {1,2,4,8,16,32}` for the MATCH-GATE-resolved config, UNLESS the
empirical crossover `M*` (defined below) is found at a smaller `M`, in
which case remaining larger-`M` cells for that (task, horizon) MAY be
skipped (pre-registered cost-saving, not a post-hoc stopping rule). Two
conditions justify `M=32` as the grid's fixed, cost-bounded practical
ceiling if no crossover is found earlier: (i) **floor clearance** — every
grid config clears the `≥13`-token minimum-viable floor (holds at least
one bind clause + the query window) by `M=2` at the latest (worst case
`n_layers=2`, fp32: `cap_length(2)=16≥13`; `n_layers=1` clears it already
at `M=1`); grid points BELOW the floor for the resolved config (only
possible at `M=1`, `n_layers=2`, fp32: `cap_length=8<13`) are reported as
descriptive-only and are NEVER eligible to set `M*` (avoids an ill-posed
"crossover" whose cause is "can't hold the query window at all," not
memory-insufficiency); (ii) **cost bound** — `M=32` is exactly the
`≈6 M-values` this design's own Wave −1 budget (§1.6 item F) prices; it is
NOT claimed to be the mathematically exhaustive point past which the
Transformer's capped role provably equals its uncapped (b-control) role at
every tested horizon (that would require `M` in the low hundreds at the
primary horizon, well beyond this wave's affordable, inference-only
scope) — if `M*` is not found by `M=32`, Rev 2 reports **"`M*` not
reached within the tested grid"** honestly (the `M*=∞` edge case, below),
not a false claim of having found the true asymptote.

**Horizons, re-pinned as ABSOLUTE token counts derived from task geometry
(F-NEW-1's own mandate — the old `2×/4×/8× cap_length` multiples are now
incoherent, since `cap_length` varies by two orders of magnitude across
the `M` grid).** Anchored to `T_bind` (224 tokens at the primary load
`K=32`, §1.4's M1 — a TASK-fixed constant, independent of `M` and of
`n_layers_transformer`), mirroring the old `2×/4×/8×` graded-frontier
shape (`DELTANET_RD_EXACTNESS_DESIGN.md` §17.5 precedent) at the new,
coherent anchor:

| Horizon | Formula | Tokens | Role |
|---|---|---|---|
| H2 | `2×T_bind + query_len` | 454 | secondary (clearly past low-`M` eviction) |
| H4 | `4×T_bind + query_len` | **902** | **PRIMARY — the `M*` decision horizon** |
| H8 | `8×T_bind + query_len` | 1798 | secondary (comfortably past) |

Task 1 (long-horizon recall) is the PRIMARY task for `M*` (its
`T_bind`/`query_len` geometry is what F-NEW-1's own re-derivation used,
and it mirrors axis 1's own primary/secondary task split, §1.4.1). Task 2
shares the identical `DeltaNetRDTaskConfig` bind-phase geometry (same
`K`/`clause_len`/`T_bind`), so the SAME horizon table applies; its `M*`
(same formula, same grid) is reported as a secondary, non-gating
bonus read, at no extra design cost.

**Rationale this is the strongest card (unchanged):** the byte-budget
match makes this an intentionally hard test FOR the contender in one
sense (its state must do real work in a small fixed footprint) and an
intentionally hard test FOR the baseline in another (a capped cache with a
naive-plus-sink eviction policy provably cannot retain information past
`cap_length(M)` tokens old, while a correctly-functioning fixed-size
recurrent state is not architecturally prevented from doing so — whether
it ACTUALLY does, and at what memory premium, is exactly what the `M`
sweep measures, not assumed).

**Win margin, re-registered in terms of the crossover multiplier `M*`
(F-NEW-1).** `M*` = the SMALLEST `M` in the grid at which
`recovered_frac@0.9`(contender) − `recovered_frac@0.9`(b-primary, `M`) at
the PRIMARY horizon (H4, 902 tokens) comes within the old 0.20 margin —
i.e. the paired `delta_ci_n` (n=3) CI on that gap EXCLUDES 0.20 on the
high side (the gap is CI-detectably `<0.20`), restricted to grid points
that clear the `≥13`-token floor (above). **Thresholds, pinned with their
one-sentence justification each:**

- **WIN if `M* ≥ 4`** — "the transformer needs at least 4× the
  contender's own bytes to match its recall" is a defensible, inspiring
  headline without overclaiming an unbounded advantage; the ACTUAL `M*`
  value (4, 8, 16, 32, or `∞`) is reported alongside the WIN tier, since
  "needs 8×" and "needs 32×" are both WINs but very different in
  narrative strength.
- **TIE if `M* = 2`** — a modest, real memory premium; matches this
  design's own TIE semantics (structure costs nothing extra, doesn't
  obviously save much either) at a byte-doubling a careful reviewer would
  call "close."
- **LOSE if `M* ≤ 1`** — the transformer already matches at exact byte
  parity (the old Rev-1 point); the constant-memory story has no
  empirical teeth at this scale, reported plainly, not softened.
- **Edge case `M* = ∞`** (pinned explicitly, per F-NEW-1's own mandate):
  the gap never comes within 0.20 anywhere in the tested grid, through
  `M=32` — the STRONGEST form of WIN (falls inside the `M*≥4` tier by
  construction). **Caveat, disclosed, not resolved:** at `M=32` the
  capped Transformer may already be close to or at its own UNCAPPED
  (b-control) ceiling for SOME resolved configs (§1.4.2's cap_length
  table: `n_layers=1`, fp32, `M=32` → `cap_length=512`, still below H4's
  902 tokens, so this specific edge is not actually reached in the
  registered grid — but a build-time re-derivation at a different
  `n_layers`/`d_model` resolution could reach it) — if `cap_length(32)`
  ever exceeds the primary horizon for the resolved config, an `M*=∞`
  reading there would partially reflect a residual FLOP-matched QUALITY
  gap (a Task-3-style effect), not pure memory-insufficiency; report
  alongside the Task 3 / b-control read in that event, never presented as
  pure memory-story evidence in isolation.
- **Edge case `M* ≤ 1`** (pinned explicitly, per F-NEW-1's own mandate):
  identical to the LOSE bullet above — "matches at parity" is the
  strongest possible LOSE for the constant-memory framing, since it means
  the contender's structural byte advantage buys nothing even before any
  multiplier is applied.

The old exact-parity `M=1` point is **RETAINED and REPORTED as a
disclosed descriptive datum** (the first row of the sweep table) — it is
no longer, by itself, the decision quantity; `M*` is. TIE and LOSE tiers
above subsume it exactly where it would have driven the old Rev-1 verdict
anyway, so no information is lost, only re-contextualized inside the full
dose-response curve.

---

### 1.5 Scale ladder

**Rung 1 = 14M** (`d_model=256/n_layers=2/d_state=64`, 14,048,896 params —
the exact config `FROZEN_BIAS_LM_DESIGN.md`'s own rung-1 already proved
trainable with the frozen-bias fix active, and the cheapest point inside
the broader 14M-98M tier Track C separately proved trainable for the base
architecture). Runs the calibration cell AND the full comparison across
both primary axes plus the control (§1.6). Opens at 14M, not 98M, because
the frozen-bias fix itself has zero training evidence at 98M (§1.2 caveat
#3); running it there for the first time inside an already-expensive
sweep would confound "does the fix train stably at 98M" with "does the
architecture win," exactly the bundled-axes failure `CLAUDE.md` warns
against.

**Escalation rung = 392M** (`d_model=1536/n_layers=16/d_state=128`,
391,869,440 params — Track C's own proven-stable rung-2 config, base
architecture only; frozen-bias fix untested here either, same caveat
applies and must be re-checked, not assumed, before escalating).

**Escalation rule, pinned now, updated for the two-axis structure:**
escalate to 392M **only** on a WIN or TIE verdict (§1.1 — i.e., at least
one primary axis did not LOSE) at rung-1. **Never escalate a LOSE** — if
BOTH primary axes lose at rung-1, the program reports the bounded result
at 14M and does not spend further compute trying to make a bigger model
win. A rung-1 TIE, or a WIN on one axis with the other axis ambiguous, is
explicitly escalation-eligible — the whole point of an escalation rung is
resolving exactly that kind of underpowered or borderline n=3 read.

**Escalation scope is NOT decided by this design (Rev 0 or Rev 1) —
flagged as an open, load-bearing item, not silently assumed (§1.6 shows
why, and Rev 1 shows the cushion for it has gotten tighter, not
looser).**

---

### 1.6 Cost arithmetic

**Rate.** The only directly-measured rate at the 14M config is
`FROZEN_BIAS_LM_DESIGN.md`'s own realized rung-1 rate: 18,175.744s / 20
cells = 908.79s/cell = **0.2524 GPU-h per 20,000-step cell** (matches the
brief's own cited figure to 4 significant figures). Used uniformly across
tasks in this table as the Rev-0 planning rate — each task's own step
budget gets a dedicated timing pilot before its cells launch (§1.7), which
may revise this table before launch, not after.

**Key cost consequence of the PI directive: the two-axis restructure adds
NO new training cells.** Axis 1 (data-efficiency) only requires logging
`recovered_frac@0.9` at more checkpoints during training runs that were
already planned — additional eval compute, not additional training.
Axis 2 (inference-memory-matched) only requires SECOND, THIRD, ... capped-
cache INFERENCE passes over (b)'s already-trained checkpoint, one per
`M`-grid point (**Rev 2: the F-NEW-1 memory-multiplier sweep multiplies
axis 2's own inference-pass count, but stays, by construction,
INFERENCE-ONLY** — no new training run for role (b-primary), which is
exactly why (b) was described in §1.3 as "two roles from one trained
arch," and exactly the constraint M-NEW-4 makes binding this revision).
The FLOP-matched control (b-control) is the same trained checkpoint's
normal, uncapped eval. Architecture and training-cell count are therefore
**unchanged from the original 3-arch plan, unchanged again in Rev 2.**

**Rung-1 cell count:** 3 archs × 3 tasks × 3 seeds = **27 training cells**
(Task 2's held-out hops are read at eval time from the same trained
cells, not separate training runs — sub-condition variation there is an
eval-time axis, not a training-cell multiplier. **M-NEW-3 (Rev 2)
narrows this claim: the K/d LOAD sub-condition is explicitly EXCLUDED**
— `K` is baked into the training episode structure itself
(`T_bind=K×clause_len`, `grammar_rd.py`), so a different `K` is a
different TRAINING cell, not an eval-time read of an existing one; Wave
−1 item C's own separate full+quarter cells already concede this in their
costing, this line only fixes the PROSE to match). Seed count (n=3)
matches this program's own standing convention for a first-look wave —
justified as the cheapest cell width this program's own CI machinery
(`delta_ci_n`) already has a pinned t-quantile for (df=2, t=4.303), with a
pre-registered, separately-costed seed-extension option (§1.8) available
if a primary axis's CI is ambiguous, mirroring the
`REASONING_LINK_DESIGN.md` §16.19/§16.20 n=3→n=12 precedent.

| Item | Cells / basis | GPU-h |
|---|---|---|
| Training (27 cells × 0.2524 GPU-h) | 3 archs × 3 tasks × 3 seeds | 6.8148 |
| Eval overhead — adds axis-1's multi-checkpoint logging + axis-2's original (single, `M=1`) capped inference pass on top of the FROZEN_BIAS_LM_DESIGN.md-observed ≈32% ratio (1.6/5.05); budgeted at ≈50% of training to cover both additions | — | 3.4074 |
| **Wave −1 (A) probe smoke** — forward/backward/grad-check with the §1.3.1 probe+adapter attached, all 3 arms | nominal | 0.05 |
| **Wave −1 (B) probe-capacity sanity null + cross-dimension tap diagnostic** (§1.3.1.4, §1.3.1.5 — M-NEW-2, Rev 2 folds the new diagnostic into this line, same CPU-feasible no-backbone harness) — frozen-random-state control, NO backbone forward pass | negligible | 0.02 |
| **Wave −1 (C) Task-1 K/d calibration** (M1) — PRIMARY point K/d=0.5 run to completion (3 arms × 1 full cell, ALSO serves as Task-1's own §1.7 gate-item-1 calibration cell) + ONE disclosed above-cliff stress point K/d=0.75 at quarter-budget, locate-only (3 arms × 1 quarter cell) — extended to all 3 arms incl. the Transformer's own capped-cache behavior | 3×(1 full + 1 quarter) | 0.95 |
| **Wave −1 (D) Task-2 calibration** — target hop-depth config, held-out-hop guard re-verified, full cells, all 3 arms | 3 cells | 0.76 |
| **Wave −1 (E) Task-3 calibration** — contender's own rung-1 config already calibrated by `FROZEN_BIAS_LM_DESIGN.md`'s own rung-1 run (not re-run); ablation reuses the contender's pinned LR, 1 full cell; Transformer's 3-point LR grid (M3) at quarter-budget, 3 cells | 1 full + 3 quarter | 0.44 |
| **Wave −1 (F) Axis-2 memory-multiplier sweep, INFERENCE overhead** (F-NEW-1, Rev 2, new — explicitly bumped, NOT silently folded into the eval-overhead line above, since that line's ≈50%-of-training ratio was calibrated to ONE capped inference pass per checkpoint, not this differently-shaped addition) — `6 M-values × 3 horizons × 2 tasks × 3 seeds = 108` short (≤1,798-token) forward-only inference passes over the already-trained (b) checkpoints, no backward/optimizer step; priced at a deliberately padded ≈5s/pass wall-clock (dominated by short-pass fixed overhead — eval-harness setup, batch construction, checkpoint access — not raw FLOPs, at this ≤1,798-token/14M-param scale) → `108 × 5s = 540s ≈ 0.15 GPU-h` | 108 inference passes | 0.15 |
| MATCH-GATE verification (§1.7) — now 3 matched quantities (incl. the M2 total-across-layers byte check and the F-NEW-1 per-`M` `cap_length` table) plus the M-NEW-1 5-key collision smoke, all still CPU-only | — | 0.0000 |
| **Raw total** | | **≈12.59 GPU-h** |

**Still meets the ≤15 GPU-h raw target**, tighter than both the original
brief's ≈9.75 GPU-h and Rev 0's own ≈11.23 GPU-h, and only ≈0.15 GPU-h
above Rev 1's own independently-reproduced ≈12.4376 GPU-h (§1.15) — the
ENTIRE Rev 2 cost delta is the axis-2 M-sweep's inference overhead
(item F); F-NEW-2's sink-eviction patch and M-NEW-1's widened
`TASK_BASE` are genuinely zero-added-GPU-h fixes (a different eviction
RULE inside the same inference passes; a wider dict + a CPU-only smoke,
respectively), disclosed as such rather than assumed. Enforced ceiling
(circuit-breaker, not expected spend — this program's realized/estimate
ratios have run 97-122% historically): bracket at **10× raw ≈ 125.88
GPU-h**, mechanically enforced by a pre-launch timing pilot exactly like
`phase2b_off_cache.py --time-pilot`'s existing pattern.

**Ledger against the shared 135 GPU-h `FROZEN_BIAS_LM_DESIGN.md` program
ceiling.** The brief cites "~123 headroom" — that is the *pre-execution
planning estimate* from that design's own §8.1. The **current, realized**
figure (post rung-1 + its mechanism follow-on wave, `FROZEN_BIAS_LM_DESIGN.md`
§12 closing ledger) is **≈7.672/135 GPU-h spent**, i.e. **≈127.33 GPU-h
headroom** — this design uses that current, verified figure, UNCHANGED by
Rev 2 (Rev 2 adds inference-only cost against the SAME headroom; it does
not touch the program's own spend ledger). Rung-1's own enforced ceiling
(≈125.88 GPU-h) would consume **≈98.86%** of that headroom in the
worst-case abort scenario — tighter again than Rev 1's own ≈97.7% draw,
flagged honestly, not smoothed over: **the margin is real (≈1.45 GPU-h,
≈1.14% of headroom) but the THINNEST this design has carried through any
revision — roughly half of Rev 1's already-razor-thin ≈2.93 GPU-h/2.3%
margin.** This tightening is the direct, disclosed cost of keeping
F-NEW-1/F-NEW-2's fixes strictly inference-only per M-NEW-4 (the
alternative — a genuinely cap-trained `(b)` arm, +0.76 raw/+7.6 at
bracket — would have consumed FOUR TIMES the entire remaining margin and
is registered as an explicit, costed, unaffordable-this-wave follow-on
instead, §1.9 item 7, never silently absorbed). **The bracket still fits
(125.88 < 127.33), so no training cell, Wave −1 gate, or seed count was
touched to make it fit** — but this is now flagged, explicitly, as the
single most fragile arithmetic in the design (below Rev 1's own already-
flagged item), and an attack-round-3 agent should re-derive the ≈0.15
GPU-h item-F estimate's own assumptions (pass count, per-pass wall-clock)
before accepting the bracket's fit, exactly as §1.15 did for Rev 1's
Wave −1 costing as a whole. This leaves near-zero slack for the
escalation rung under the SAME shared ceiling (below) — tighter than Rev
1 already flagged it to be. Expected realized spend (≈100-130% of raw,
this program's typical range) is ≈12.6-16.4 GPU-h, ≈9.9-12.9% of
headroom — essentially unchanged from Rev 1's own ≈9.8-12.7% EXPECTED-
case read (item F is small in absolute terms; it is specifically the
worst-case 10× circuit breaker that tightened materially, exactly as Rev
1's own text already flagged for its own additions). **Flagged in §1.16
as the one resolution this revision could not close with full margin** —
an attack-round-3 agent should treat this bracket's fit as live, not
settled.

**Escalation-rung cost — flagged now, not resolved now.** Track C's own
realized 392M rate: 128.3 GPU-h / 91,552 steps = 0.0014017 GPU-h/step →
**≈28.03 GPU-h per 20,000-step-equivalent cell** — ≈111× the 14M rate. A
reduced-scope escalation matrix of even just 2 archs × 1 task × 3 seeds =
6 cells at the SAME 20,000-step budget would cost **≈168 GPU-h** — more
than the entire current headroom, and now competing with rung-1's own
larger (≈12.6-16.4 GPU-h realized, ≈125.88 GPU-h worst-case, Rev 2) draw
on the same 135 GPU-h ceiling — a materially smaller cushion than Rev 0's own
already-tight ≈112.3 GPU-h worst-case draw, per the Wave −1 additions
above. **The escalation rung, as currently scoped, does not fit the
existing ceiling at rung-1's own step budget, full stop.**
Resolving this is explicitly NOT decided here (§1.9 item 1): the likely
resolution is a shorter step budget at 392M (this is a comparison wave,
not a full pretrain) and/or a reduced arch/task/seed count restricted to
whichever axis won at rung-1, but the exact numbers require the
escalation rung's own dedicated design addendum and GPU-h re-derivation,
gated on rung-1's own verdict. **m2 (Rev 1):** that addendum must also
RE-DERIVE, not reuse, §1.3(b)'s FLOP-matching basis
(`DELTANET_REALDATA_DESIGN.md` §4.2's "head-dominated" cost model) at
392M — §4.2's own text ties the head-dominated regime to the 14M/rung-1
probe-tier `d_model=256` scale specifically; at `d_model=1536` (392M),
the `6ND` non-embedding term grows much faster than the 50,257-vocab
softmax head's own cost, so the "head-dominated" assumption may not hold
at all at the escalation rung, and the FLOP-match tolerance (§1.3, ≤5%)
cannot be verified against a formula derived at a different regime.
Escalation is not pre-authorized by this document.

---

### 1.7 Gates

Standing machinery, reused verbatim from this program's own precedent, not
re-derived:

1. **Calibration-first, now explicitly 3-ARM (M1, Rev 1) — Rev 0 left
   whether this covered all three arms ambiguous; it did not, and the
   attack caught it.** One real training cell at each task's target
   config, run to completion, **for EACH of the three arms, including the
   Transformer's own capped-cache (b-primary) behavior** — checked against
   the val-loss/`recovered_frac` band this task's own small-scale
   precedent (Task D/E for tasks 1-2, `FROZEN_BIAS_LM_DESIGN.md`'s own
   val-loss ceilings for task 3) predicts, BEFORE the 27-cell sweep
   launches. Costed explicitly as §1.6's Wave −1 items C/D/E. **This
   calibration cell also does double duty for the two-axis restructure:**
   it is where axis 1's power sketch pins `X` (§1.4.1) and where axis 2's
   `cap_length` is derived and verified from the contender's real measured
   TOTAL-across-layers state-byte count (§1.4.2, M2) — both BEFORE the
   full sweep launches, not after, per the `CLAUDE.md` calibration-first
   rule applied literally to the new quantities the PI directive
   introduced. **M1's K/d margin is likewise frozen only AFTER the
   §1.3.1/§1.6-item-C K/d calibration sweep returns for all three arms —
   never assumed from Rev 0's table, and never from `model_rd.py`'s own
   K/d=0.5455 cliff measurement, which carries ZERO evidentiary weight
   for these three arms (different architecture, not just different `d`
   — see §1.9's self-attack, restated there per M1).**
2. **Timing pilots, mechanical enforced abort.** One real cell per
   arch×task pair measured for wall-clock before its own seed-fan-out
   launches; if the projected cost for that pair exceeds its share of the
   §1.6 ceiling, the chain hard-aborts before spending it.
3. **Enforced aborts with negative tests.** Every gate ships with its own
   negative test proven to have teeth — e.g. the injectivity guard reused
   from `grammar_rd.py` (`_assert_injective_entities`) gets its own
   `_test_injectivity_guard_detects_merge` re-run as part of Stage −1 for
   this wave, not assumed still-correct from its original file.
4. **Sha closure.** Every shipped script/config gets a sha256 manifest,
   verified against the box copy before launch and re-verified after
   completion.
5. **PI-signoff tokens.** Two independent env-var tokens required before
   any GPU cell: `HEADTOHEAD_PI_SIGNOFF=1` (launch authorization) and
   `HEADTOHEAD_MATCH_GATE_SIGNOFF=1` (distinct — specifically attests the
   MATCH-GATE below has passed). Both required independently, with a
   negative-test triple.
6. **MATCH-GATE — the make-or-break gate, now covering THREE matched
   quantities per the PI directive (params, FLOPs, inference-memory
   bytes), not two, and (Rev 2) a `cap_length(M)` TABLE rather than a
   single point.** Structured as two independent passes that must agree:
   - **Pass 1 (implementer):** `verify_match_gate.py` (to be built)
     instantiates all three real models/roles, sums `numel()` for param
     counts (mirrors `lm_rd_rung_configs.verify_param_count`), computes
     the measured FLOP profile for each (mirrors
     `DELTANET_REALDATA_DESIGN.md` §4.2's method), AND derives/verifies
     `cap_length(M)` **for every `M ∈ {1,2,4,8,16,32}`** (F-NEW-1, Rev 2 —
     Rev 1 derived a single point) from the contender's real measured
     **TOTAL-across-all-layers** state-byte count (M2, Rev 1 — NOT
     per-layer), **at the pinned fp32 accounting dtype** (§1.4.2 — the
     contender's real persisted-state dtype, `lm_pretrain_rd.py:986`,
     NOT the kernel-internal bf16 cast), against (b-primary)'s real
     `n_layers/n_heads/d_head` — asserting the ≤1% / ≤5% / exact-byte
     tolerances from §1.3's table at EVERY `M`, not just `M=1`.
   - **Pass 2 (independent audit agent, NOT the implementer — `CLAUDE.md`'s
     "the implementer does not review their own work" rule, plus
     "multiple independent adversarial audit rounds catch different bugs"):**
     re-derives all three matched quantities (incl. the full `cap_length(M)`
     table, all 6 grid points) from scratch, from the model configs alone,
     without reading Pass 1's code, and must land within the same
     tolerances — INCLUDING an independent check that (i) the capped
     eviction policy (§1.3(b), **sink+FIFO, k_sink=4** — F-NEW-2, Rev 2,
     was plain sliding/FIFO in Rev 1) is implemented exactly as disclosed
     (first `k_sink=4` tokens always retained, remainder FIFO), not
     something quietly more or less sophisticated that would over- or
     under-state the memory constraint's real bite, (ii) the byte
     accounting is genuinely TOTAL-across-layers on BOTH sides at the
     PINNED fp32 dtype (M2 + the Rev-2 dtype pin) — a per-layer-vs-total
     OR an fp32-vs-bf16 mismatch on either side is exactly the kind of
     silent-imbalance bug this Pass-2 re-derivation exists to catch, and
     (iii) the `M`-grid itself matches the pinned `{1,2,4,8,16,32}` set
     and the grid-endpoint rule (§1.4.2) — no silently truncated or
     extended grid.
   - Disagreement between passes is a hard launch-block, full stop.
7. **Probe-capacity null gate (F1, new in Rev 1).** §1.3.1.4's
   probe-capacity sanity null must PASS (`recovered_frac@0.9 < 0.05` on
   frozen-random states, for EVERY arm's own adapter shape) before
   `HEADTOHEAD_MATCH_GATE_SIGNOFF=1` may be set — folded into the same
   signoff token as MATCH-GATE (both are "is the instrument itself sound
   before we trust what it reads" checks), not a separate token, to avoid
   a signoff-token proliferation Rev 0's §1.7 gate 5 did not anticipate.

---

### 1.8 Pre-registered analysis

**CI machinery.** Every axis's Δ (contender − axis-appropriate baseline)
is computed via `reasoning_link_probe.py`'s already-verified `delta_ci_n`
(paired-seed t-CI, `half_width = t(n-1,.975) * s/sqrt(n)`, t-quantile read
from the PINNED `CI_T_975_BY_DF` table, never silently interpolated). At
n=3 seeds, `df=2, t=4.303` — already pinned. A seed-extension to n=9
(`df=8, t=2.306`) or n=12 (`df=11, t=2.201`) is also already pinned —
zero new CI derivation work either way.

**Multiple-comparison discipline — co-primary axes with an OR-win rule,
stated with its honest statistical cost.** The PI directive's "a win on
EITHER primary axis is a publishable WIN" is a standard co-primary-
endpoints design: each axis keeps its own nominal 95% CI (no per-axis
alpha-correction), because the pre-registered semantics genuinely is
"succeed if either fires," not "succeed only if both fire" (which WOULD
warrant loosening, not tightening, in the opposite direction). The
honest cost, disclosed here and required to be disclosed again at write-up
time: under a global null (contender truly no different from either
baseline on either axis), the chance that at least one of the two
independent 95%-CI axis tests nominally "wins" by chance is
`1 - 0.95² ≈ 9.75%`, not 5%. This is reported in the paper's own
methods/limitations text, not silently absorbed into a single-test 95%
claim. Task 3 (LM quality) and the FLOP-matched control (b-control) are
both non-gating, reported reads, unaffected by this correction question.

**Seed-extension contingency (pre-registered, not open-ended).** If a
primary axis's n=3 CI is ambiguous (straddles its margin boundary) AND the
point estimate's direction is consistent with a win, a seed-extension to
n=9 is available for THAT axis's specific cells only — costed explicitly
(6 more seed-cells × 0.2524 GPU-h ≈ 3.03 GPU-h raw per axis extended,
inside remaining headroom) — and gated through the SAME batch-effect
variance-ratio check `REASONING_LINK_DESIGN.md` §16.19.5 registered
(`var_ratio > 4.0` → flag, do not silently pool old and new seed cohorts)
before any pooled reading is treated as decision-grade. This directly
avoids the exact trap that produced §16.20's BATCH-EFFECT-FLAGGED outcome.

---

### 1.9 Attack-yourself — self-attack round 0 (minimum 5, per house
convention; 7 in Rev 0, 9 in Rev 1, now **11 in Rev 2** — F-NEW-1's memory
sweep and F-NEW-2's sink patch are themselves new attack surface and get
their own self-attack, items 10-11)

1. **The escalation-rung budget does not fit the current ceiling at the
   step budget this doc assumes (§1.6).** Tighter at every revision
   (rung-1's own worst-case ceiling draw grew from ≈76.5% pre-directive →
   ≈88% in Rev 0 → ≈97.7% in Rev 1 → **≈98.86% in Rev 2**, because
   F-NEW-1's mandatory memory-multiplier sweep, itself a genuine,
   non-optional inference cost, adds a further ≈0.15 GPU-h raw on top of
   Rev 1's own already-thin margin) — this is the single biggest open
   item, deliberately not papered over, and NOW the closest to its own
   ceiling this design has EVER been (§1.6's own honest disclosure: "the
   THINNEST this design has carried through any revision").
2. **model_rd.py's K/d cliff numbers carry ZERO evidentiary weight for
   any of this design's three arms (M1, resolved structurally in Rev 1,
   restated here per the attack's own instruction).** Rev 0's K/d=0.75
   high-load point was ABOVE `KEY_ANCHORING_SCALING_DRAFT.md`'s measured
   cliff (x0=0.5455) — and, more fundamentally than "measured at a
   different `d`," that measurement is from `model_rd.py`'s
   anchor-table/Newton-Schulz architecture, which is NOT the contender,
   NOT the flat-vector ablation, and NOT the Transformer. Rev 1's fix:
   the primary load point is re-pinned to K/d=0.5 (§1.4, below the only
   number this program has measured for ANY architecture, without
   claiming that number transfers), K/d=0.75 is retained as a disclosed
   stress point, and §1.3.1/§1.6 Wave −1 item C runs a real K/d
   calibration sweep for THESE THREE ARMS, extended to the Transformer's
   own capped-cache behavior, with the margin frozen only after that
   sweep returns (§1.7 gate 1). **This does not make the cliff question
   solved — it makes it MEASURED, for the first time, on architectures
   this design actually uses**, which is the honest bar M1 set.
3. **Does the flat-vector ablation actually remove multiplicative
   structure, or does a sufficiently expressive gated-linear recurrence
   secretly recover comparable capacity?** The ablation's own capacity
   ceiling should be independently verified analytically before axis 1's
   result is trusted as diagnosing "matrix vs. vector" rather than "this
   particular vector recurrence vs. this particular matrix recurrence."
4. **Is the byte-for-byte KV-cache cap a genuinely fair inference-memory
   comparison, or does implementation overhead (KV cache metadata,
   attention-score buffers, etc.) make "same bytes" ambiguous?** MATCH-
   GATE's Pass 2 must pin exactly what counts toward each side's byte
   budget (raw K/V tensor storage only? or all inference-time buffers?)
   BEFORE `cap_length` is derived — an under-specified byte accounting
   would silently favor whichever side counts less.
5. **Is the sink+FIFO eviction policy a fair baseline, or a strawman?**
   §1.3(b) deliberately used the simplest policy that closes the
   disclosed train/eval mismatch (F-NEW-2, Rev 2: sink+FIFO, `k_sink=4`,
   replacing Rev 1's plain sliding/FIFO) and disclosed it as such — but a
   reviewer will reasonably ask whether a modestly smarter, still-standard
   policy (e.g. attention-score-based eviction, or a larger/tuned
   `k_sink`) would close axis 2's gap further. Registered as an explicit
   follow-on robustness check (item 7's list, below), not required to win
   the Rev-2 verdict, but flagged for the paper's limitations section
   either way.
6. **Is n=3 really justified for two co-primary axes**, given this exact
   program's own repeated, costly experience that n=3 is chronically
   underpowered (`REASONING_LINK_DESIGN.md`'s §16.18-20 arc)? Now doubly
   relevant with two axes each needing their own clean read. The
   seed-extension contingency (§1.8) is the answer given here, but an
   attack round should sanity-check whether opening at n=6 directly for
   both primary axes is cheaper overall than a likely n=3→n=9 extension
   on one or both.
7. **Scope creep risk in the other direction.** Cut from this scope: the
   fix-ON/OFF ablation (§1.2), length generalization (no reusable
   instrument found), byte-level input (explicitly parked per the PI
   directive), a smarter-eviction-policy robustness check (item 5 above),
   **and (Rev 2, F-NEW-2) a genuinely cap-trained `(b)` arm** — training
   the Transformer WITH its KV cache capped from step 0, rather than
   capping only at inference on a fully-uncapped checkpoint, is the
   COMPLETE fix for F-NEW-2's train/eval mismatch (the sink patch is
   PARTIAL); explicitly costed at **+0.76 GPU-h raw / +7.6 GPU-h at the
   10× bracket** (a new 3-arm training-cell block, M-NEW-4) — more than
   FOUR TIMES Rev 2's entire remaining ≈1.45 GPU-h bracket margin, hence
   categorically unaffordable this wave and never silently absorbed into
   the sink-patch's own (zero-added-cost) scope. If the PI's intent for
   "the program's capstone question" implies more than this Rev 2 scopes,
   that should surface now, before rung-1 data creates sunk-cost pressure
   to declare a narrower victory than intended.
8. **(NEW, Rev 1) The Transformer's probe adapter is not capacity-matched
   to the other two arms' adapters, by construction, and cannot be.**
   §1.3.1.2 pins every arm's adapter to the SAME `nn.Linear` class/init/LR,
   but the Transformer's `native_tap_dim = d_model_transformer` generally
   differs from `value_dim`, so its adapter has strictly more (or fewer)
   parameters than the contender's/ablation's `value_dim → value_dim`
   adapters. This is disclosed as unavoidable (native tap dims are a
   direct consequence of param-matching via width, not state dim), and
   the probe-capacity null (§1.3.1.4, run separately per arm's REAL
   adapter shape) is the mitigation — but a hostile reviewer could still
   argue a bigger adapter gives the Transformer's probe strictly more
   linear-readout capacity than the matrix-state arms get. Registered as
   a residual, not-fully-closed asymmetry; the null check bounds it, does
   not eliminate it.
9. **(NEW, Rev 1) The contender/ablation's query never touches their
   recurrence; the Transformer's query IS part of its context — an
   inherent, not incidental, asymmetry in HOW each arch is probed.**
   §1.3.1.2 deliberately keeps the contender's and ablation's query
   OUT of `bind()`'s recurrence (protecting the P=1 bottleneck,
   `CLAUDE.md`), while the Transformer's query is necessarily APPENDED to
   its context (attention has no other read channel) — meaning the
   Transformer's own capped cache can be partially evicted BY the query
   window's own tokens immediately before the read happens, a failure
   mode the contender/ablation structurally cannot suffer. This could
   make axis 2 (§1.4.2) MORE punishing to the Transformer than the byte
   budget alone already is — favoring a contender WIN, not disfavoring
   it, which is the opposite direction of most fairness concerns raised
   elsewhere in this design (contrast item 5's "is the eviction policy a
   strawman" concern, which cuts the same way). Flagged, not fixed: fixing
   it would require inventing a non-attention "side-channel read" for the
   Transformer that has no real-world deployment analogue, which this
   design declines to do (a real deployed capped-cache Transformer has
   this exact same limitation).
10. **(NEW, Rev 2, M-NEW-2) The contender's and the ablation's native taps
    are not expressivity-matched, by construction, and cannot be fully
    repaired by the shared linear probe.** The contender reads via a full
    matrix-vector product `S_T @ q` (every output dimension mixes every
    input dimension of the query); the flat-vector ablation reads via a
    Hadamard product `s_T ⊙ q` (diagonal-only — output dimension `i`
    depends on input dimension `i` alone). This is an irreversible
    expressivity gap: information the ablation's state encodes ONLY in
    cross-dimension (off-diagonal-equivalent) structure is invisible to
    its own native tap, and no LINEAR adapter/probe downstream can
    recover it post hoc (§1.3.1.2's adapters are deliberately
    linear-only, for the F1-mandated reason of not letting the probe do
    compositional work the backbone should be doing — which means they
    also cannot silently paper over this asymmetry). §1.3.1.4's null does
    NOT exercise this confound (it scores i.i.d. Gaussian inputs with no
    adversarial diagonal/off-diagonal structure). This structurally
    favors the CONTENDER on axis 1 (data-efficiency) for reasons that
    have nothing to do with matrix-vs-vector state CAPACITY, only
    matrix-vs-vector READ mechanism — a confound the original F1/M2
    probe-parity work never closed because it targeted probe capacity,
    not tap expressivity. **Mitigation, not elimination (§1.3.1.5, Rev
    2):** a synthetic cross-dimension diagnostic bounds — via a
    diagonal/off-diagonal energy-split calibration curve — how much of
    any REAL axis-1 gap tap expressivity alone COULD explain, letting a
    reader cross-reference the ablation's real measured Hadamard-tap
    `recovered_frac` against the curve for an implied "as-if diagonal
    fraction." It does not measure the ablation's actual diagonal
    fraction on real trained states (a cheap potential follow-on, out of
    scope this wave) and does not repair the confound; it bounds it.
11. **(NEW, Rev 2, F-NEW-2 residual) The sink+FIFO patch mitigates, but
    does not eliminate, the train/eval attention mismatch.** Retaining
    `k_sink=4` tokens addresses the specific, well-documented failure mode
    StreamingLLM identifies (catastrophic collapse from evicting early
    high-attention-mass tokens) but does not address every way a
    fully-uncapped-trained Transformer's attention distribution could
    differ from what it would have learned under a genuinely cap-trained
    regime — e.g. the model may have learned to spread retrieval-relevant
    signal across MID-sequence positions in a way no fixed sink size
    protects, and `k_sink=4` itself is not empirically tuned this wave
    (borrowed as a standard small default, not searched). The gap between
    "sink+FIFO-patched, inference-only" and "genuinely cap-trained" is
    exactly what item 7's costed, deferred follow-on (+0.76 raw/+7.6
    bracket) would close; until that follow-on runs, any axis-2 LOSE or
    weak-WIN result carries this residual, disclosed uncertainty — it
    could reflect the Transformer's genuine memory-limited capability, OR
    an artifact of the imperfect train/eval match, and this wave's design
    cannot fully distinguish the two. Flagged for the paper's limitations
    section regardless of which way axis 2 resolves.

---

### 1.10 What this design does and does NOT do

**Does:** pins one exact contender architecture and config; designs the
`CLAUDE.md`-mandated flat-vector control and reuses one Transformer
architecture in two inference-time roles (uncapped disclosed control;
byte-capped primary axis) rather than building a 4th architecture; frames
two future-facing co-primary axes (data-efficiency; inference-memory-
matched) per the PI's mid-drafting directive, with an OR-win rule and its
honest family-wise statistical cost disclosed; keeps the LM-quality
honesty check as a non-gating secondary; pre-registers margins, CI
machinery, and every outcome framing (including LOSE) before any data
exists; derives a rung-1 cost that still fits its ≤15 GPU-h target, with
real, sourced arithmetic reflecting the restructure's added eval cost;
states plainly where the escalation rung's own budget does not yet close.
**Rev 1 adds:** designs ONE architecture-neutral, trained shared probe
head (§1.3.1) with a mandatory capacity-null control, closing the F1
FATAL that Rev 0 had no working readout for any of its three arms; pins
an exact on-the-fly episode-seed schedule for Tasks 1/2 (F1b); re-pins
Task 1's load point below the only measured cliff and states the old
cliff's zero evidentiary weight for these arms (M1); fixes the
byte-match to total-across-layers (M2); pins a modern-strong,
equal-tuning-budget Transformer baseline recipe (M3); corrects the 14M
config citation (m1) and flags the 392M FLOP-basis re-derivation (m2).
**Rev 2 adds:** replaces axis 2's single, independently-found-DEGENERATE
equal-byte point with a pre-registered geometric memory-multiplier sweep
(`M ∈ {1,2,4,8,16,32}`) and re-registers the axis-2 verdict around the
crossover multiplier `M*`, with pinned WIN/TIE/LOSE thresholds and both
`M*` edge cases defined (F-NEW-1); patches the capped-cache eviction rule
to sink+FIFO (`k_sink=4`) as a disclosed PARTIAL fix for the train/eval
attention mismatch, registering the full cap-trained fix as an explicit,
costed, unaffordable-this-wave follow-on rather than absorbing it
silently (F-NEW-2); widens `TASK_BASE` to 5 keys with a runtime
collision-guard assert and an extended collision smoke (M-NEW-1);
discloses the contender-vs-ablation native-tap expressivity asymmetry as
its own self-attack item and adds a synthetic cross-dimension diagnostic
bounding its contribution to any axis-1 gap (M-NEW-2); narrows the
§1.6 "eval-time axis" claim to exclude the K/load sub-condition, which is
baked into training structure (M-NEW-3); pins the probe-capacity null's
protocol to fresh-per-step draws scored on held-out fresh draws
(m-new-2); re-derives §1.6's cost table with the M-sweep's own inference
overhead as an explicit, disclosed line item, keeping every fix
inference-only per the binding M-NEW-4 budget constraint.

**Does NOT:** authorize any GPU launch (Rev 2, pre-attack-round-3);
decide the 392M escalation rung's exact scope or budget (§1.6, §1.9 item
1 — explicitly deferred, now with the thinnest headroom cushion this
design has carried); reopen the frozen-bias mechanism story, the
key-anchoring capacity/pool-margin instrument questions, or the
reasoning-link geometric-transfer readout (all cited, none re-litigated);
claim a paper-grade benchmark result from Task 3's internal byte-BPC
number without a follow-on standard-benchmark step; introduce byte-level
input (explicitly out of scope per the PI directive); build the fix-ON/OFF
ablation, a length-generalization task, a smarter-eviction robustness
arm, an MLP-probe upgrade, or **a genuinely cap-trained `(b)` arm** (all
explicitly cut, registered as costed follow-on candidates, §1.9 item 7);
fully close the Transformer-adapter capacity asymmetry, the
query-continuation asymmetry, the contender/ablation tap-expressivity
asymmetry, or the sink-patch's own residual train/eval mismatch (§1.9
items 8-11 — each bounded or partially mitigated by its own control, not
eliminated).

---

### 1.11 Sequencing

design (Rev 0) → attack round 1 → NEEDS-REVISION (§1.13) → design (Rev 1)
→ attack round 2 → NEEDS-REVISION (§1.15) → design (this doc, Rev 2) →
attack round 3 → ... (iterate until DESIGN-CLEARED-FOR-BUILD, per this
program's own standing gauntlet discipline) → build (contender wiring for
the frozen-bias arm already exists; new code needed: flat-vector ablation
mixer WITH its own `q_proj`/`q_conv1d` (§1.3(a)), standard Transformer
with RoPE + the contender's own `FFN` class + switchable uncapped/
capped-KV inference mode with sink+FIFO eviction (§1.3(b), F-NEW-2), the
§1.3.1 shared probe head + per-arm adapters + frozen `T_val` target table
+ probe-capacity null harness + the cross-dimension tap diagnostic
(§1.3.1.5, M-NEW-2), the `rd_episode_seed` schedule with its 5-key
`TASK_BASE` and runtime collision assert (F1b, M-NEW-1),
`verify_match_gate.py` (now total-across-layers at the pinned fp32
accounting dtype, M2 + Rev-2 dtype pin), per-task/per-axis/per-arm
calibration wrappers incl. the K/d sweep (M1) and the Transformer's LR
grid (M3), the axis-2 `cap_length(M)` sweep across the pinned
`{1,2,4,8,16,32}` grid and the `M*` crossover-detection logic (F-NEW-1))
→ independent build audit (separate agent, per `CLAUDE.md`) → launch
rung-1 on GPUs 0-6 (GPU 7 held as pool/overflow) → harvest →
escalation-rung decision (§1.5 rule, mechanically applied) → IF
win-or-tie: escalation-rung design addendum (resolves §1.9 item 1, incl.
m2's 392M FLOP-basis re-derivation, and decides whether the cap-trained
`(b)` follow-on, §1.9 item 7, is affordable at that point) → attack →
build → audit → launch 392M → harvest → paper fold-in (iclr-2027,
workshop-2026) either way.

---

### 1.12 Reproducibility pointers

- Contender: `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` (`DeltaNetLM`,
  `DeltaNetLMMixer`, `DeltaNetLMBlock`), frozen-bias mechanism in the same
  file (`build_frozen_bias_table`, `apply_frozen_bias_blend`, lines
  178-246) plus `matrix-thinking/FROZEN_BIAS_LM_DESIGN.md` for the
  mechanism's own design history.
- 14M config: `frozen_bias_lm_sweep.py`'s `RUNG1_CFG` /
  `run_lm_rd_trackc_sweep.py`'s `CONTROL_CFG` (m1 fix, Rev 1 — NOT
  `lm_rd_rung_configs.py`'s `RUNGS`). Escalation-rung (98M/392M/1.31B)
  configs: `matrix-thinking/deltanet_rd/lm_rd_rung_configs.py` (`RUNGS`,
  `verify_param_count`) — unchanged from Rev 0, correct for those three
  rungs.
- Task 1/2 instrument: `matrix-thinking/deltanet_rd/grammar_rd.py`
  (`DeltaNetRDTaskConfig`, `sample_batch_rd`, `_permutation_graph`,
  `_assert_injective_entities`). Readout: §1.3.1's shared probe head (F1)
  — the metric FORMULA (`F.cosine_similarity` @ 0.9) is the same
  primitive `matrix-thinking/deltanet_rd/key_anchoring.py`'s
  `measure_h1_behavioral_companion`/`readout_rev7.py` use, but the
  `pred`/`target` machinery is new (§1.3.1), NOT that function's own
  `model_rd.py`-welded 5-tuple forward.
- Task 3 instrument: `FROZEN_BIAS_LM_DESIGN.md`'s eval/corpora pipeline
  (`wikitext-mix-ext`, `openr1-mix-ext` via
  `rebuild_lm_corpora_rd.py`/`build_mix_corpora_rd.py`).
- CI machinery: `matrix-thinking/deltanet_rd/reasoning_link_probe.py`
  (`delta_ci_n`, `CI_T_975_BY_DF`); its `episode_seed`
  mixed-radix-stride pattern is also the precedent for §1.3.1's new
  `rd_episode_seed` (F1b).
- Frozen-table precedent: `matrix-thinking/deltanet_rd/key_anchoring.py`
  (`random_unit_rows_init`, `ANCHOR_INIT_SEED=20260705`) — reused
  unmodified for §1.3.1's `T_val` probe-target table, at a NEW,
  deliberately distinct `PROBE_TARGET_SEED=20260709`, and again (Rev 2,
  §1.3.1.5) for the cross-dimension diagnostic's own fresh, independent
  `q`/`t` draws (never sharing a seed with `T_val` or the frozen bias
  table).
- Gate precedent: `phase2b_off_cache.py` (`--time-pilot`),
  `run_poolmargin_k84s1943_k90s2043.py` / `run_k69_s1733_contingency.py`
  (dual PI-signoff tokens), `lm_rd_rung_configs.py` (`verify_param_count`
  as the MATCH-GATE's param-counting precedent),
  `phase2b_seedext_stage_minus1.py` (`:165-235`, SE4/SE5
  exhaustive-enumeration collision-freedom check, the precedent for
  `rd_episode_seed`'s own negative test, F1b, extended Rev 2 to 5 keys
  per M-NEW-1).
- **FLOP-basis precedent (Rev 2, F-NEW-1):**
  `matrix-thinking/DELTANET_REALDATA_DESIGN.md` §4.2's head-dominated
  FLOP re-derivation (`:568-641`) — the same method §1.3(b) already cited
  for the ≤5% training-FLOP match is reused, Rev 2, to derive the
  admissible `d_model≈256`/`n_layers∈{1,2}` Transformer family the
  axis-2 `cap_length(M)` table is built from.
- **StreamingLLM precedent (Rev 2, F-NEW-2):** Xiao et al. 2023's
  attention-sink observation — the one-line rationale for §1.3(b)'s
  sink+FIFO eviction patch (`k_sink=4`); not otherwise reused code in
  this repo, cited for the mechanism only.
- **New, not yet built:** the standard-Transformer arch with RoPE, the
  contender's own `FFN` class, standard init, and a switchable
  uncapped/hard-capped-KV inference mode with **sink+FIFO eviction**
  (axis 2's baseline, §1.3(b), M3, F-NEW-2); `verify_match_gate.py`, now
  total-across-layers at the pinned fp32 accounting dtype, over the
  **`cap_length(M)` table across `M ∈ {1,2,4,8,16,32}`** (§1.7 item 6,
  M2 + Rev-2 dtype pin + F-NEW-1); the axis-1 power-sketch script that
  pins `X` (§1.4.1); the flat-vector-ablation mixer WITH its own
  `q_proj`/`q_conv1d` (§1.3(a)); **§1.3.1's shared probe head**
  (`shared_probe`, per-arm `adapter_arm`, `T_val` target table, the
  probe-capacity null harness, **and (Rev 2) the §1.3.1.5
  cross-dimension tap diagnostic**) — the F1 FATAL's entire resolution
  plus its Rev-2 M-NEW-2 extension; the `rd_episode_seed` schedule
  module, now 5-keyed with a runtime collision-guard assert (F1b,
  M-NEW-1); the Transformer's 3-point LR-grid calibration harness (M3);
  the axis-2 `M*` crossover-detection/reporting logic (F-NEW-1).

---

### 1.13 ATTACK ROUND 1 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 1.
Full report retained in the session transcript; findings binding on Rev 1:

**FATAL-caliber (must be resolved structurally, not disclaimed):**

- **F1 — The architecture-neutral readout does not exist for ANY of the
  three comparison arms.** The `recovered_frac@0.9` cosine FORMULA is
  neutral in principle, but the only live implementation
  (`key_anchoring.py:953` `measure_h1_behavioral_companion`) is welded to
  `model_rd.py`'s Wave-1 5-tuple forward signature — a FOURTH architecture,
  not any of the three being compared. `DeltaNetLM` has zero
  `grammar_rd`/`recovered_frac` references (its own docstring says so,
  lines 9-16); the only DeltaNetLM↔grammar_rd bridge in the codebase is
  the dead §16.15-16.20 zero-shot readout this design correctly refuses.
  The flat-vector ablation and the Transformer don't exist at all.
  → Rev 1 must add a pre-build design item: ONE shared probe-head
  architecture (tap point, dimensionality, training regime — trained as
  part of the objective, not zero-shot), implemented with matched
  capacity across all three arms, costed and gated as its own Wave −1
  item in §1.6/§1.7, listed in §1.12. Probe-parity is itself a
  confound to be closed by construction.
- **F1b (compounding) — Tasks 1/2 data-matching is unoperationalized.**
  §1.3(b)'s same-data prescription covers only Task 3's static corpora;
  Tasks 1/2 use `grammar_rd.py` on-the-fly generation. Rev 1 must pin
  the same-RNG/seed-schedule + matched-steps rule explicitly.

**MAJOR:**

- **M1 — K/d=0.75 at d=64 is ABOVE the measured cliff (x0=0.5455) and the
  measurement is from a different architecture.** K=48/d=64 measured
  h4=0.0215 (near-total collapse) — but on `model_rd.py`'s
  anchor-table/Newton-Schulz architecture, NOT on DeltaNetLM/ablation/
  Transformer. Neither the "above-cliff" hazard nor the
  different-ARCHITECTURE (not just different-d) caveat is named in the
  self-attack. Rev 1: re-pin the Task-1 load point (or justify it) and
  extend the calibration gate to cover ALL THREE arms, incl. the
  Transformer's capped-cache behavior, before margins freeze.
- **M2 — Byte-match ambiguity: per-layer vs total.** §1.3(b)'s formula
  and §1.7 MATCH-GATE Pass 1 disagree on whether cap_length matches the
  contender's per-layer 16,384 bytes or its total-across-layers state.
  At different n_layers this silently unbalances total memory. Rev 1:
  pin ONE interpretation (total-across-layers is the defensible one) and
  make the MATCH-GATE check it end-to-end.
- **M3 — Baseline tuning parity entirely absent.** No LR/warmup/tuning
  budget/positional-scheme (RoPE vs learned-absolute) parity rule.
  The #1 hostile-reviewer rejection vector. Rev 1: pin a modern-strong
  baseline recipe + an explicit equal-tuning-budget rule.

**MINOR:** m1 — §1.2 miscites the 14M config to `lm_rd_rung_configs.py`
`RUNGS` (it holds only 98M/392M/1.3B; the 14M is `RUNG1_CFG`/
`CONTROL_CFG`). m2 — §4.2's head-dominated FLOP basis is 14M-specific;
re-derive at 392M in the escalation addendum (already implicitly
disclosed).

**Independently re-verified clean:** rate 0.25244 GPU-h/cell; rung-1
≈11.23 GPU-h and ×10=112.3 vs 127.33 headroom; ledger 7.672/135
(→127.33); escalation-rung 168 GPU-h infeasibility; frozen-bias
mechanism vs `lm_pretrain_rd.py:178-247`; 14,048,896 params; K-cycle
h%K guard; position-decomposition closed for axis 1 (P=1 both arms) and
deliberately exercised-then-capped for axis 2.

---

### 1.14 REV 1 CHANGES — finding → resolution map

Every §1.13 finding, mapped to its exact Rev 1 resolution. Nothing below
is a disclaimer-only close — each row points at new or rewritten design
text, not a footnote.

| Finding | Resolution (Rev 1) | Where |
|---|---|---|
| **F1** (FATAL) — no architecture-neutral readout exists for any arm; `measure_h1_behavioral_companion` welded to `model_rd.py`'s 5-tuple forward | New §1.3.1: ONE shared, trained probe head (`shared_probe` linear map + per-arm `adapter_arm`), a frozen `T_val` target table (reusing `random_unit_rows_init` at a new seed), per-arm native-tap specs (contender: `S_T_last @ q_query` via a non-recurrent query pathway; ablation: `s_T ⊙ q_query`, `d_state` pinned equal to the contender's, §1.3(a); Transformer: final-block hidden at `<Q>` from a real context-extending pass), a pinned joint-auxiliary training regime (`aux_weight=0.1`, revised at calibration), and a MANDATORY probe-capacity sanity null (§1.3.1.4) that must pass before either signoff token is set (§1.7 gate 7). Costed as Wave −1 items A/B in §1.6. | §1.3.1, §1.4 (Task 1/2 readout references updated), §1.6, §1.7 gate 7, §1.12 |
| **F1b** (compounding) — Tasks 1/2 data-matching unoperationalized | `rd_episode_seed(task, seed_idx, ckpt_idx)`, a mixed-radix stride formula (precedent: `reasoning_link_probe.py`'s `episode_seed`/`STRIDE_SEED`) that DELIBERATELY EXCLUDES architecture from its inputs, so all three arms draw byte-identical `sample_batch_rd` episode streams at matched (task, seed, checkpoint); collision-freedom gets its own negative-test smoke (precedent: `phase2b_seedext_stage_minus1.py`), not assumed from the arithmetic. | §1.3.1 ("F1b" subsection), §1.12 |
| **M1** — K/d=0.75 above the measured cliff; cliff measured on a different architecture, un-flagged in self-attack | Primary load re-pinned to K/d=0.5 (below the only measured number, transfer explicitly NOT claimed); K/d=0.75 kept as a disclosed stress point; §1.3.1/§1.6 Wave −1 item C runs a real K/d calibration sweep for all three arms (incl. Transformer capped-cache); margin frozen only after that sweep (§1.7 gate 1, now explicitly 3-arm); "zero evidentiary weight" of `model_rd.py`'s cliff stated in BOTH §1.4 and §1.9 item 2, per the finding's own instruction. | §1.4 (Task 1), §1.6 item C, §1.7 gate 1, §1.9 item 2 |
| **M2** — byte-match ambiguous (per-layer vs. total) | Pinned: TOTAL-across-all-layers state bytes is the match quantity (defensible: what a deployed instance actually holds). §1.3(b)'s formula rewritten (`32,768` total at rung-1, not `16,384`/layer); §1.7 MATCH-GATE Pass 1/Pass 2 both rewritten to check the total, with Pass 2 explicitly re-deriving it independently. | §1.3(b), matching table, §1.7 gate 6 |
| **M3** — no baseline tuning-parity rule; "#1 hostile-reviewer rejection vector" | Modern-strong recipe pinned: RoPE (justified twice — modern standard, AND avoids axis-2's 8×`cap_length` horizon exceeding a learned-absolute position ceiling), pre-norm, the contender's OWN `FFN` class reused verbatim (removes an asymmetry axis, matches the contender's own SwiGLU-rejection reasoning), GPT-2-standard init. Equal-tuning-budget rule: warmup PINNED identical (100 steps, no grid) across all arms; LR gridded only for the Transformer (3-point grid around the contender's own already-validated LR, evaluated at the calibration cell, frozen before the sweep) — the contender's own LR search cost was already paid in a separate, already-completed prior wave. Costed as Wave −1 item E. | §1.3(b), §1.6 item E |
| **m1** — §1.2 miscites 14M to `lm_rd_rung_configs.py` `RUNGS` | Corrected: 14M is `frozen_bias_lm_sweep.py`'s `RUNG1_CFG` / `run_lm_rd_trackc_sweep.py`'s `CONTROL_CFG`; `RUNGS` correctly covers only 98M/392M/1.31B. Verification path corrected too (smoke item [1b]'s real `numel()`, not `verify_param_count`, for the 14M point). | §1.2, §1.12 |
| **m2** — §4.2 head-dominated FLOP basis is 14M-specific, needs re-derivation at 392M | Explicit re-derivation requirement added to the escalation-rung addendum's scope, with the reason stated (the `6ND` term outgrows the softmax-head cost as `d_model` grows, so "head-dominated" may not hold at 392M at all). | §1.3(b), §1.6 (escalation-rung paragraph) |

**What this revision could NOT close with full margin (flagged, not
papered over — the task's own instruction):** the Wave −1 machinery F1/
M1/M3 require is real, necessary cost, and fitting it under the existing
10× ceiling discipline (§1.6) left only **≈2.9 GPU-h (≈2.3%) of headroom
margin** — reached through genuine scope discipline (reduced-budget
locate-only calibration cells, waiving the contender's own
already-proven Task-3 calibration, pinning rather than gridding warmup),
not by cutting anything a finding actually required, but still the
thinnest margin this design has carried through any revision. An
attack-round-2 agent should treat the Wave −1 costing (§1.6) as live,
adversarial territory, not settled arithmetic — and should independently
re-derive the ≈12.44 GPU-h raw total and its ≈124.4 GPU-h ×10 bracket
against the ≈127.33 GPU-h headroom before accepting it.

---

### 1.15 ATTACK ROUND 2 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 2.
Scope was: verify §1.13 resolutions real, attack everything Rev 1 added,
adjudicate 6 coordinator-flagged items. Not FATAL-REDESIGN — every
finding has a targeted, budget-compatible fix preserving the
3-arch/2-axis/OR-win structure. Findings binding on Rev 2:

**FATAL (axis 2 — must be fixed structurally, both have inference-only
resolutions):**

- **F-NEW-1 — cap_length is DEGENERATE at rung-1; the equal-byte axis-2
  test as written is vacuous.** Re-derived from the design's own
  head-dominated FLOP method (`DELTANET_REALDATA_DESIGN.md` §4.2): the
  fixed `V=50257` head (~92% of FLOPs/token) pins the FLOP-matched
  Transformer to `d_model≈256`, `n_layers∈{1,2}` (n_layers=3 overshoots
  the ≤5% band; deep-narrow escapes fail the match entirely and are
  WORSE). `cap_length = 32768/(2×n_layers×d_model×4)` = **8-16 tokens
  fp32 (16-32 bf16)** vs task geometry `query_len=6`,
  `T_bind=K×clause_len=224` tokens at K=32 (`grammar_rd.py:307-332`).
  The capped baseline retains 1-2 of 32 bind clauses regardless of
  eviction policy → contender clears the 0.20 margin trivially →
  reviewers correctly call the headline rigged. Degeneracy is robust
  across the entire admissible config family.
  → **Rev 2 resolution (proposed by the attack, inference-only, no new
  training cells): replace the single byte-parity point with a
  MEMORY-MULTIPLIER SWEEP** — cap at `M×` contender total bytes,
  `M∈{1,2,4,8,16,32,...}` geometric until cap_length clears query+one
  clause; report the crossover multiplier `M*`; re-register the axis-2
  WIN in terms of `M*` ("Transformer needs ≥X× the contender's bytes to
  match"). Strictly more informative for the constant-memory framing.
- **F-NEW-2 — train/eval mismatch: (b-primary) caps at INFERENCE ONLY a
  checkpoint trained fully uncapped.** Zero train-with-cap precedent in
  the repo (grepped). Full-attention-trained models concentrate mass on
  early tokens (attention sinks); FIFO eviction removes exactly those →
  unpredictable distribution-shift degradation, not graceful
  memory-limited degradation → strawman objection independent of
  cap_length. Compounds F-NEW-1 at the informative low-M end.
  → **Rev 2 resolution: attention-sink patch** (always retain first
  k_sink≈4 tokens in the eviction rule), disclosed as a PARTIAL
  mitigation; the full fix (a genuinely cap-trained arm) is explicitly
  unaffordable this wave (+0.76 raw → +7.6 bracket > the 2.93 margin)
  and must be registered as a costed follow-on, not silently absorbed.

**MAJOR:**

- **M-NEW-1 — F1b's `TASK_BASE` is incomplete:** no `task2_calib` (or
  `task1_stress`) key, yet Wave −1 items C/D require those cells; a
  build agent must invent a key or silently reuse `task2_sweep`'s base →
  seed collision between calibration and sweep, the exact bug class F1b
  closes. Fix: 5 keys with spaced bases (task1_calib=0,
  task1_stress=500k, task1_sweep=1M, task2_calib=1.5M, task2_sweep=2M) +
  extend the collision smoke to all five.
- **M-NEW-2 — probe native-tap asymmetry, undisclosed:** contender reads
  via full matvec `S_T @ q` (any-to-any mixing); ablation reads via
  Hadamard `s_T ⊙ q` (diagonal-only) — an irreversible expressivity gap
  the linear adapter/probe cannot repair, NOT covered by the §1.3.1.4
  null (which bypasses the taps). Structurally favors the contender on
  axis 1. Fix: disclose alongside self-attack items 8-9 + add the
  synthetic cross-dimension diagnostic bounding how much of any axis-1
  gap tap expressivity alone could explain.
- **M-NEW-3 — "eval-time axis" prose inconsistency:** K is baked into
  training episode structure (`T_bind=K×clause_len`), so K/d=0.75 is NOT
  an eval-time sub-condition; Wave −1(C)'s own separate quarter-cells
  concede this. Costing already correct; prose isn't. Fix: narrow the
  §1.6 claim to exclude the load axis; state K/d=0.75 is permanently
  locate-only/non-decision-grade.
- **M-NEW-4 — budget arithmetic verified clean but LOAD-BEARING:**
  12.4376≈12.44 raw, 124.4 bracket, 127.33 headroom, 2.93 margin all
  independently reproduced. Any fix requiring one new 3-arm cell block
  (+7.6 at bracket) blows the margin — hence F-NEW-1/2's resolutions
  MUST stay inference-only.

**MINOR:** m-new-1 — `ckpt_idx < STRIDE_SEED_RD` never asserted (add
runtime assert); seed independence of PROBE_TARGET_SEED vs
ANCHOR_INIT_SEED empirically CONFIRMED (matched-row cos mean≈-0.0005,
std≈0.1255 vs theoretical 1/√64=0.125). m-new-2 — §1.3.1.4 null protocol
ambiguous between fresh-per-step vs fixed-pool draws (fixed pool risks a
memorization false-pass); pin fresh draws + held-out eval draws.

**Verified clean this round:** all §1.13 resolutions real; rd_episode_seed
mixed-radix arithmetic; seed independence (empirical); 0.05 null bar
(given m-new-2 pinned); the 10× circuit-breaker multiple vs realized
97-122% ratios; param figures 14,048,896/1,183,104 cross-validated.

---

### 1.16 REV 2 CHANGES — finding → resolution map

Every §1.15 finding, mapped to its exact Rev 2 resolution. Every fix stays
**inference-only** per the binding M-NEW-4 constraint (one new 3-arm
training-cell block would cost +7.6 GPU-h at the 10× bracket — more than
FOUR TIMES Rev 2's entire remaining ≈1.45 GPU-h margin).

| Finding | Resolution (Rev 2) | Where |
|---|---|---|
| **F-NEW-1** (FATAL) — `cap_length` degenerate at rung-1 (8-16 tokens fp32 vs. `T_bind=224`/`query_len=6`); the equal-byte axis-2 test as written is vacuous | The single `M=1` equal-byte point is replaced by a pre-registered geometric **memory-multiplier sweep**, `M ∈ {1,2,4,8,16,32}`, capping `(b-primary)` at `M × 32,768` bytes; a `cap_length(M)` table is derived for the admissible `d_model≈256`/`n_layers∈{1,2}` Transformer family at the PINNED fp32 accounting dtype (justified by the contender's own persisted-state dtype, `lm_pretrain_rd.py:986`), with the bf16 variant disclosed as a non-decision secondary column; the grid's endpoint rule is pinned exactly (floor clearance by `M≤2`; cost-bounded practical ceiling at `M=32`, disclosed as not mathematically exhaustive); axis-2's horizons are re-pinned as ABSOLUTE token counts anchored to the task-fixed `T_bind` (`H2=454`, `H4=902` PRIMARY, `H8=1798`), replacing the now-incoherent `cap_length`-multiple horizons; the axis-2 verdict is re-registered around the crossover multiplier `M*` (smallest grid `M` with the contender-vs-`(b-primary)` gap at H4 CI-detectably `<0.20`), with pinned thresholds (WIN `M*≥4`, TIE `M*=2`, LOSE `M*≤1`) and both `M*=∞`/`M*≤1` edge cases defined explicitly, including the disclosed `M*=∞` interpretation caveat (residual quality-gap confound if `cap_length(32)` ever exceeds the primary horizon at a different resolved config). | §1.3(b), §1.4.2 (full rewrite), §1.6 item F, §1.7 gate 6, §1.9 items 1/7 |
| **F-NEW-2** (FATAL) — train/eval mismatch: `(b-primary)` caps at INFERENCE ONLY a checkpoint trained fully uncapped; naive FIFO evicts exactly the early "attention sink" tokens full-attention training concentrates mass on | Eviction rule upgraded to **sink+FIFO**: always retain the first `k_sink=4` tokens, FIFO over the remainder (one-line StreamingLLM rationale, Xiao et al. 2023), disclosed as a PARTIAL mitigation, not a full fix; the full fix — a genuinely cap-trained `(b)` arm — is explicitly costed (+0.76 raw/+7.6 bracket) and registered as an unaffordable-this-wave follow-on (never silently absorbed); the residual gap between "sink-patched" and "genuinely cap-trained" is named as its own self-attack item (11), disclosed for the paper's limitations section regardless of axis-2's outcome. | §1.3(b), §1.7 gate 6, §1.9 items 5/7/11 |
| **M-NEW-1** — `TASK_BASE` incomplete (3 keys; Wave −1 items C/D need `task1_stress`/`task2_calib`, inviting a silent seed collision) | Widened to 5 keys (`task1_calib=0, task1_stress=500_000, task1_sweep=1_000_000, task2_calib=1_500_000, task2_sweep=2_000_000`); `rd_episode_seed` gets a runtime assert (`ckpt_idx < STRIDE_SEED_RD`); the collision-freedom smoke (SE4/SE5 precedent) is extended to enumerate all 5 keys × `seed_idx∈[0,11]` × the full `ckpt_idx` range, plus a dedicated negative test for the new assert. | §1.3.1 (F1b subsection), §1.12 |
| **M-NEW-2** — undisclosed native-tap asymmetry (contender: full matvec, any-to-any mixing; ablation: Hadamard, diagonal-only) — an irreversible expressivity gap the linear probe/adapter cannot repair, not covered by §1.3.1.4's null, structurally favoring the contender on axis 1 | Disclosed as self-attack item 10; new §1.3.1.5 **synthetic cross-dimension diagnostic** — exact diagonal/off-diagonal energy-split constructions swept over `alpha∈{0,0.25,0.5,0.75,1.0}`, scoring both taps against the same cosine-recovery metric, tracing a calibration curve a real measured Hadamard-tap `recovered_frac` can be read against for an implied "as-if diagonal fraction"; CPU-only, folds into Wave −1 item B at zero added GPU-h. | §1.3.1.5, §1.6 item B, §1.9 item 10 |
| **M-NEW-3** — "eval-time axis" prose claims K/d is an eval-time sub-condition; contradicted by `T_bind=K×clause_len` being baked into training structure and by Wave −1(C)'s own separate cells | §1.6's cell-count paragraph narrowed to explicitly EXCLUDE the K/load sub-condition from the "eval-time axis, not a training-cell multiplier" claim; states K/d=0.75 is permanently locate-only/non-decision-grade (unchanged from Rev 1's own M1 text, now cross-referenced correctly). | §1.6 |
| **M-NEW-4** — budget arithmetic verified clean but load-bearing; any fix requiring a new 3-arm training-cell block blows the ≈2.93 GPU-h margin | Binding constraint honored by construction: F-NEW-1 (inference-only M-sweep on already-trained checkpoints) and F-NEW-2 (an eviction-rule patch inside the same inference passes) add **zero new training cells**. The M-sweep's own inference overhead is priced explicitly (not silently absorbed into the unchanged 50%-of-training eval line) at **+0.15 GPU-h raw**, re-derived in full in §1.6. | §1.6 (full re-derivation) |
| **m-new-1** — `ckpt_idx < STRIDE_SEED_RD` never asserted; seed independence unconfirmed | Runtime assert added to `rd_episode_seed`; seed independence was already empirically confirmed per §1.15's own verification (cos mean≈-0.0005, std≈0.1255 vs. theoretical 1/√64=0.125) — no further action needed, restated here for completeness. | §1.3.1 (F1b subsection) |
| **m-new-2** — §1.3.1.4 null protocol ambiguous (fresh-per-step vs. fixed-pool draws; fixed pool risks a memorization false-pass) | Pinned: fresh i.i.d. `state_summary_raw` draws every training step, `recovered_frac@0.9` evaluated on a SEPARATE held-out set of fresh draws never seen during the null's own training. | §1.3.1.4 |

**Budget re-derivation (M-NEW-4's own instruction).** Rev 1's
independently-reproduced raw total (§1.15: 12.4376≈12.44 GPU-h) plus the
ONLY new cost this revision adds — Wave −1 item F, the M-sweep's
inference overhead, **+0.15 GPU-h** (108 short forward-only passes: 6
`M`-values × 3 horizons × 2 tasks × 3 seeds, priced at a padded ≈5s/pass
— F-NEW-2's sink patch and M-NEW-1's `TASK_BASE` widening are genuine
zero-added-GPU-h fixes, verified, not assumed) — gives:

- **New raw total: ≈12.59 GPU-h** (still ≤ the 15 GPU-h raw target).
- **New 10× bracket: ≈125.88 GPU-h.**
- **Headroom (unchanged, verified figure): 127.33 GPU-h.**
- **New margin: ≈1.45 GPU-h (≈1.14% of headroom)** — the bracket still
  fits (125.88 < 127.33), so no training cell, Wave −1 gate, or seed
  count was trimmed to make it fit — but this margin is roughly HALF of
  Rev 1's own already-razor-thin ≈2.93 GPU-h (≈2.3%), and is the
  thinnest this design has carried through any revision.
- **Pinned `M*` WIN/TIE/LOSE thresholds (F-NEW-1):** WIN if `M*≥4`; TIE
  if `M*=2`; LOSE if `M*≤1`; `M*=∞` (gap never closes within the tested
  grid, through `M=32`) is the strongest form of WIN, with a disclosed
  interpretive caveat about residual quality-gap confounding at the
  grid's own practical ceiling.

**What this revision could NOT close with full margin (flagged, not
papered over — the task's own instruction):** the M-sweep's inference
overhead is real, necessary, and inference-only, exactly as M-NEW-4
requires — but pricing it explicitly (rather than silently stretching the
unchanged 50%-of-training eval-overhead ratio to cover a structurally
different addition) pushed the worst-case 10× bracket to ≈98.86% of the
current 127.33 GPU-h headroom, the tightest this design has been at any
revision. The ≈0.15 GPU-h item-F estimate itself rests on assumptions
(pass count, ≈5s/pass wall-clock) that have not been measured on real
hardware — an attack-round-3 agent should treat this specific number as
the most likely place a re-derivation would move the bracket's fit,
exactly as §1.15 treated Rev 1's Wave −1 costing as a whole. The `M*=∞`
edge case's own interpretive caveat (§1.4.2 — a residual quality-gap
confound if `cap_length(32)` ever exceeds the primary horizon for a
different resolved `n_layers`/`d_model`) is disclosed but not resolved;
it would require either a tighter admissible-config pin at MATCH-GATE
time or an explicit cross-reference rule against the Task-3/b-control
read, neither of which this revision builds. Both are flagged as live
items for attack round 3, not silently assumed away.

---

### 1.17 ATTACK ROUND 3 VERDICT (independent fresh-eyes agent, 2026-07-08): NEEDS-REVISION

Recorded per the gauntlet-bookkeeping hard rule before dispatching Rev 3.
The round numerically EXECUTED Rev 2's new formulas (a first for this
gauntlet — 500-trial pure-Python simulation of §1.3.1.5) and
cross-checked the M* rewrite against sections Rev 2 did not touch.
Findings binding on Rev 3:

**FATAL:**

- **R3-F1 — §1.3.1.5's cross-dimension diagnostic is mathematically
  broken.** Numerically verified: cosine similarity is scale-invariant
  and `target_effective := S_star @ q` is near-parallel to `t` for any
  α>0 (off-diagonal zeroing perturbs only O(1/√d)); the "calibration
  curve" is a degenerate step function (cos≈0.9999 at α=0.25..1.0, 0
  only at exactly α=0); the matvec side is tautologically 1.0 by
  construction. ROOT CAUSE: single-(q,t)-pair reachability — a diagonal
  read can always hit ONE target exactly; the diagonal-vs-matvec gap
  only bites under MULTIPLE simultaneous bindings in one state. So
  M-NEW-2 is NOT resolved (disclosure honest, mechanism nonfunctional).
  → Rev 3: replace with a K-simultaneous-bindings diagnostic (fit each
  tap family's optimal state to K (q_i,t_i) pairs by least squares;
  report recovered_frac vs K for both taps — matvec holds to K≤d w/
  orthogonal keys, Hadamard collapses generically at K≥2, giving a real
  graduated bound) AND numerically verify the replacement's formulas
  pre-commit exactly the way this round did.
- **R3-F2 — M* is underpowered at n=3 and biased toward the
  strongest-WIN default.** Wide n=3 CIs make excluding 0.20 HARDER →
  pushes M* larger or to "not found," and M*=∞ is pre-registered as the
  STRONGEST WIN — noise systematically favors the contender. Plus: (a)
  no within-axis multiplicity discipline for the up-to-6-point ordered
  search (§1.8 discloses only the between-axis 9.75%); (b) §1.8's
  seed-extension contingency was never updated for the M* rewrite (no
  rule for an intermediate grid point's CI straddling 0.20); (c)
  "underpowered" and "confirmed no-crossover" are conflated inside
  M*=∞. → Rev 3: pin a fixed-sequence testing procedure over the grid
  w/ an honest FWER statement; a straddle at any would-be-M* grid point
  triggers the EXISTING §1.8 seed-extension contingency (already
  costed, PI-gated) before M* finalizes; split M*=∞ into
  CONFIRMED-no-crossover (every grid CI excludes crossover) vs
  INDETERMINATE (any straddle) — only the former reportable as
  strongest WIN.

**MAJOR:**

- **R3-F3 — LOSE is structurally unreachable if the transformer
  resolves to n_layers=2** (M=1 is floor-excluded at n_layers=2/fp32:
  cap 8<13; LOSE requires M*≤1) — an unpinned implementation choice
  (n_layers∈{1,2} both FLOP-admissible) can gate the §1.5 escalation
  decision. → Rev 3: pin n_layers_transformer explicitly AND remap the
  verdict tiers onto ELIGIBLE grid points so every tier is reachable
  under the pinned config (justify the pin).
- **R3-F4 — the ≈5s/pass M-sweep assumption has no scoped timing-pilot
  gate and no de-scope rule** against a 1.45 GPU-h margin (a
  reload-per-pass implementation could run 3-6×). → Rev 3: extend §1.7
  gate 2 to the M-sweep inference fan-out (pilot 2 M-values on 1
  checkpoint before fan-out); pin checkpoints-resident-across-passes as
  a build requirement; pre-register the de-scope order (drop M=32,
  then H8) if the pilot overruns.

**MINOR:** R3-F5 — `seed_idx` has no runtime bound (seed_idx=50 on
task1_calib collides with task1_stress seed 0); add the assert + its
negative test (ckpt_idx has one; seed_idx doesn't). R3-F6 — item F's
108-pass count double-counts the M=1 pass already claimed by the
eval-overhead line (trivial, cost-shrinking; fix the stale
cross-reference).

**Also fold in (cross-campaign, from the capability research wave,
recorded in STATE 58de0fa):** add to the §1.9 caveats register that the
contender's β gate is plain sigmoid (β∈[0,1]) — per Grazzi et al.
(arXiv:2411.12537) it has NO TC0-escape property as configured, so
Task-2 held-out-depth results are empirical-only, carrying no formal
state-tracking-separation implication; the β∈[0,2] variant is
deliberately RESERVED for the separate capability campaign to protect
the frozen-bias evidence provenance (λ=0.58 was tuned under sigmoid β).

**Verified clean this round (attack's own list):** cap_length(M) table
(4 entries re-derived incl. sink-inside-cap 8−4=4 at M=1/n_layers=2);
fp32 pin vs lm_pretrain_rd.py:986; sink+FIFO spec buildable as a
windowed-attention mask; raw 12.5922 line-sum; 108-pass factorization
(6M×3H×2tasks×3seeds); TASK_BASE margins (380,001 headroom); ckpt_idx
assert; CI table; §1.7 gate 6 coherence.

---

*(End §1. Rev 0 → §1.13 → Rev 1 → §1.15 → Rev 2 → §1.17 NEEDS-REVISION
(R3-F1 broken diagnostic, R3-F2 M* statistics; 2 MAJOR, 2 minor).
Rev 3 in progress.)*
