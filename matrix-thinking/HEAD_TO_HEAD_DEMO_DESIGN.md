# HEAD-TO-HEAD DEMO — the program's capstone question

## §1 DESIGN — HEAD-TO-HEAD DEMO (Rev 1, post-attack-round-1 revision,
2026-07-08) — does the matrix-native fast-weight model beat a matched
conventional baseline on real tasks at meaningful scale, or is its value
honestly bounded?

Status: **Rev 1, pre-attack (round 2 pending), zero GPU spent.** Rev 0
(§1.1-1.12 below) was independently attacked and returned NEEDS-REVISION
(§1.13, retained verbatim as the record: 2 FATAL-caliber + 3 MAJOR + 2
MINOR findings). This Rev 1 resolves every binding finding **structurally**
(new design machinery, not a disclaimer bolted on) — §1.14 maps each
finding to its exact resolution and flags the one item that could not be
closed with full margin (the Wave −1 GPU-h fit, §1.6). Ratified by the PI
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
    at the SAME byte budget as the contender's **TOTAL, across-all-layers**
    fixed matrix state (**M2 fix, Rev 1** — Rev 0 left per-layer vs.
    total ambiguous; total-across-layers is the defensible quantity,
    because it is what a deployed instance of the contender actually
    holds in memory end-to-end, and per-layer parity alone would silently
    unbalance total memory whenever the two archs' `n_layers` differ,
    which they generally will once param-matching picks each arch's own
    depth/width split). At rung-1, contender total state bytes =
    `n_layers_contender (2) × d_state² (64×64) × 4 bytes (fp32) = 32,768
    bytes`, not the 16,384-bytes/layer figure Rev 0 quoted (that number
    is the correct PER-LAYER figure, §1.2, but §1.7 MATCH-GATE now checks
    the TOTAL, not the per-layer, quantity) — via the simplest, most
    conservative eviction policy available: a sliding/FIFO window sized
    so `2 (K&V) × n_layers_transformer × n_heads × d_head × cap_length ×
    bytes_per_elt = contender's TOTAL state bytes (32,768 at rung-1)`,
    solved for `cap_length` (in tokens) once `n_layers_transformer`/
    `n_heads`/`d_head` are fixed by the Transformer's own param-matching
    build. Deliberately the SIMPLEST policy, not a sophisticated
    KV-compression scheme, disclosed as such (§1.9 item 7) — this is the
    fair, conservative comparison, not a strawman tilted further toward
    the contender than the byte-budget match alone already tilts it.

**Matching arithmetic table, rung-1 (14M tier) — to be verified for real
by MATCH-GATE (§1.7) before any cell launches, not assumed from this
table. Inference-memory column corrected Rev 1 (M2): TOTAL across all
layers, not per-layer.**

| Arch | Params (target) | Training FLOPs | Training data | Inference-memory (fixed-state bytes, TOTAL across all layers) |
|---|---|---|---|---|
| Contender (`DeltaNetLM`, frozen-bias `per_token`, λ=0.58) | 14,048,896 (measured, `d_model=256/n_layers=2/d_state=64`) | measured per-token profile × steps × batch × seq_len (pin once) | `wikitext-mix-ext` + `openr1-mix-ext`, same step count as (a)/(b) | 32,768 = 2 layers × 64×64 × 4 bytes (fp32; CONSTANT in context length T) |
| (a) Flat-vector ablation | matched to contender within ≤1% | falls out (reported) | same corpora, same step count (data-matched by construction — axis 1's premise) | 32,768 = 2 layers × 64 (`d_state`, pinned equal to the contender's, §1.3(a)) × 4 bytes — reported, not the axis-2 comparison arm |
| (b) Standard Transformer | falls out (reported) | matched to contender within ≤5% | same corpora, same step count | role (b-control): grows `O(T)` uncapped, reported at eval horizon; role (b-primary): hard-capped so TOTAL K+V bytes across all layers/heads = 32,768 (`cap_length` solved from `n_layers_transformer/n_heads/d_head`, pinned by MATCH-GATE) |

The ≤1% / ≤5% / exact-byte-match tolerances are deliberately tighter than
the 15% rung-config planning tolerance in `lm_rd_rung_configs.py` — that
tolerance governs whether a scale-ladder point is "close enough to its
target" for an unrelated study; here the match arithmetic between arms IS
the finding, so it needs its own, much tighter bar, with the
inference-memory byte match held EXACT (an integer token cap solved from
the contender's real measured TOTAL byte count, not a rounded
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
probe+adapter's own shapes) against the SAME frozen `T_val` targets. Pass
condition: `recovered_frac@0.9` after training stays statistically
indistinguishable from chance (chance ≈ 0 for cosine recovery of a
random unit vector in `R^64`; pass bar set at `< 0.05`, i.e. the probe
alone, with no real state to read, must not "solve" the task by
memorizing the frozen target table's own structure). FAIL on this null
is a hard launch-block for the responsible arm's probe design — this is
what closes the F1-mandated "probe-parity confound... closed by
construction" requirement with an actual negative test, not an assertion.
Cost: negligible (no backbone), see §1.6 Wave −1 item B.

**F1b — Tasks 1/2 data-matching (compounding finding, resolved here).**
Rev 0's §1.3(b) same-data prescription covered only Task 3's static
corpora; Tasks 1/2 use `grammar_rd.py`'s on-the-fly `sample_batch_rd`
generation, which was left unpinned. Fix, mirroring
`reasoning_link_probe.py`'s own `episode_seed` mixed-radix precedent
(`PURPOSE_BASE`/`LEG_BASE`/`STRIDE_*`, collision-free by construction,
each digit's stride exceeding the previous digit's max reach):

```
TASK_BASE = {"task1_calib": 0, "task1_sweep": 1_000_000, "task2_sweep": 2_000_000}
STRIDE_SEED_RD = 10_000   # per-seed-index stride (n=3 seeds, extendable to 12 -- 11*10_000=110_000 < 1_000_000 margin)
STRIDE_STEP_RD = 1        # per-checkpoint-index stride (raw checkpoint index, not raw step count)

def rd_episode_seed(task: str, seed_idx: int, ckpt_idx: int) -> int:
    return TASK_BASE[task] + seed_idx * STRIDE_SEED_RD + ckpt_idx
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
this formula is the piece Rev 0 omitted. Collision-freedom (no two
distinct `(task,seed_idx,ckpt_idx)` triples mapping to the same seed) is
mechanically verified by a dedicated negative-test smoke item, mirroring
`phase2b_seedext_stage_minus1.py`'s own exhaustive-enumeration check —
not assumed from the arithmetic alone (`CLAUDE.md`'s own "run the
negative unit test... to completion" rule on structural checks).

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
strongest card"

**Comparison:** contender (fixed-size matrix state, constant in T) vs.
(b-primary) the same trained Transformer with its KV cache hard-capped at
the contender's own byte budget (§1.3(b), exact-byte MATCH-GATE item).
Tested at pre-registered horizons **T = 2×, 4×, and 8× `cap_length`** on
Task 1 (long-horizon recall) and Task 2 (multi-hop composition where the
hop chain's span exceeds `cap_length`) — horizons chosen to be clearly
past where the cap forces eviction (2×) through comfortably past it (8×),
giving a dose-response reading rather than a single point, and following
this program's own repeated preference for a graded frontier over a single
razor-edge test (`DELTANET_RD_EXACTNESS_DESIGN.md` §17.5's "graded
frontier, not a razor cliff" precedent).

**Rationale this is the strongest card:** the byte-budget match makes this
an intentionally hard test FOR the contender in one sense (its state must
do real work in a small fixed footprint) and an intentionally hard test
FOR the baseline in another (a capped cache with a naive eviction policy
provably cannot retain information past `cap_length` tokens old, while a
correctly-functioning fixed-size recurrent state is not architecturally
prevented from doing so — whether it ACTUALLY does is exactly what this
axis measures, not assumed).

**Win margin:** at the primary pre-registered horizon **T = 4× cap_length**,
`recovered_frac@0.9` (contender) − `recovered_frac@0.9` (b-primary) ≥
**0.20** absolute (paired `delta_ci_n`, n=3), CI excluding 0.20 — the
largest margin in this design, appropriate to a claim billed as the
program's strongest card (a small effect here would already be an
ambiguous, secondary-tier result, not the headline). The 2× and 8×
horizons are pre-registered SECONDARY reads at the same 0.20 margin,
reported as a dose-response curve alongside the 4× primary point, not
separately gating the axis verdict. TIE: |Δ|<0.20 at 4×, or CI includes
0.20. LOSE: the capped baseline wins by ≥0.20 at 4×, CI excluding −0.20 —
striking and reportable precisely because the setup is designed to favor
the contender.

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
Axis 2 (inference-memory-matched) only requires a SECOND, capped-cache
INFERENCE pass over (b)'s already-trained checkpoint — no new training
run for role (b-primary), which is exactly why (b) was described in §1.3
as "two roles from one trained arch." The FLOP-matched control (b-control)
is the same trained checkpoint's normal, uncapped eval. Architecture and
training-cell count are therefore **unchanged from the original 3-arch
plan.**

**Rung-1 cell count:** 3 archs × 3 tasks × 3 seeds = **27 training cells**
(Task 1 now split load/horizon sub-conditions and Task 2's held-out hops
are read at eval time from the same trained cells, not separate training
runs — sub-condition variation is an eval-time axis, not a training-cell
multiplier). Seed count (n=3) matches this program's own standing
convention for a first-look wave — justified as the cheapest cell width
this program's own CI machinery (`delta_ci_n`) already has a pinned
t-quantile for (df=2, t=4.303), with a pre-registered, separately-costed
seed-extension option (§1.8) available if a primary axis's CI is
ambiguous, mirroring the `REASONING_LINK_DESIGN.md` §16.19/§16.20
n=3→n=12 precedent.

| Item | Cells / basis | GPU-h |
|---|---|---|
| Training (27 cells × 0.2524 GPU-h) | 3 archs × 3 tasks × 3 seeds | 6.8148 |
| Eval overhead — adds axis-1's multi-checkpoint logging + axis-2's second (capped) inference pass on top of the FROZEN_BIAS_LM_DESIGN.md-observed ≈32% ratio (1.6/5.05); budgeted at ≈50% of training to cover both additions | — | 3.4074 |
| **Wave −1 (A) probe smoke** — forward/backward/grad-check with the §1.3.1 probe+adapter attached, all 3 arms | nominal | 0.05 |
| **Wave −1 (B) probe-capacity sanity null** (§1.3.1.4) — frozen-random-state control, NO backbone forward pass (probe+adapter shapes only) | negligible | 0.02 |
| **Wave −1 (C) Task-1 K/d calibration** (M1) — PRIMARY point K/d=0.5 run to completion (3 arms × 1 full cell, ALSO serves as Task-1's own §1.7 gate-item-1 calibration cell) + ONE disclosed above-cliff stress point K/d=0.75 at quarter-budget, locate-only (3 arms × 1 quarter cell) — extended to all 3 arms incl. the Transformer's own capped-cache behavior | 3×(1 full + 1 quarter) | 0.95 |
| **Wave −1 (D) Task-2 calibration** — target hop-depth config, held-out-hop guard re-verified, full cells, all 3 arms | 3 cells | 0.76 |
| **Wave −1 (E) Task-3 calibration** — contender's own rung-1 config already calibrated by `FROZEN_BIAS_LM_DESIGN.md`'s own rung-1 run (not re-run); ablation reuses the contender's pinned LR, 1 full cell; Transformer's 3-point LR grid (M3) at quarter-budget, 3 cells | 1 full + 3 quarter | 0.44 |
| MATCH-GATE verification (§1.7) — now 3 matched quantities (incl. the M2 total-across-layers byte check), still CPU-only | — | 0.0000 |
| **Raw total** | | **≈12.44 GPU-h** |

**Still meets the ≤15 GPU-h raw target**, tighter than both the original
brief's ≈9.75 GPU-h and Rev 0's own ≈11.23 GPU-h — F1's Wave −1 machinery
(probe integration, its mandatory null, and the M1/M3-driven 3-arm
calibration extensions) is real, disclosed added cost, not hand-waved
away. Enforced ceiling (circuit-breaker, not expected spend — this
program's realized/estimate ratios have run 97-122% historically):
bracket at **10× raw ≈ 124.4 GPU-h**, mechanically enforced by a
pre-launch timing pilot exactly like `phase2b_off_cache.py --time-pilot`'s
existing pattern.

**Ledger against the shared 135 GPU-h `FROZEN_BIAS_LM_DESIGN.md` program
ceiling.** The brief cites "~123 headroom" — that is the *pre-execution
planning estimate* from that design's own §8.1. The **current, realized**
figure (post rung-1 + its mechanism follow-on wave, `FROZEN_BIAS_LM_DESIGN.md`
§12 closing ledger) is **≈7.672/135 GPU-h spent**, i.e. **≈127.33 GPU-h
headroom** — this design uses that current, verified figure. Rung-1's own
enforced ceiling (≈124.4 GPU-h) would consume **≈97.7%** of that headroom
in the worst-case abort scenario — markedly tighter than Rev 0's own
≈88% draw, and flagged honestly, not smoothed over: **the margin is real
(≈2.9 GPU-h, ≈2.3% of headroom) but razor-thin, and was reached by
genuine scope trimming (reduced-budget locate-only calibration cells,
waiving the contender's own already-proven Task-3 calibration, pinning
warmup instead of gridding it), not by cutting anything F1/M1/M3 actually
require.** This leaves near-zero slack for the escalation rung under the
SAME shared ceiling (below) — tighter than Rev 0 already flagged it to
be. Expected realized spend (≈100-130% of raw, this program's typical
range) is ≈12.4-16.2 GPU-h, ≈9.8-12.7% of headroom — comparable to Rev
0's ≈9-12% in the EXPECTED case; it is only the worst-case 10× circuit
breaker that tightened materially. **Flagged in §1.14 as the one
resolution this revision could not close with full margin** — an
attack-round-2 agent should treat this bracket's fit as live, not settled.

**Escalation-rung cost — flagged now, not resolved now.** Track C's own
realized 392M rate: 128.3 GPU-h / 91,552 steps = 0.0014017 GPU-h/step →
**≈28.03 GPU-h per 20,000-step-equivalent cell** — ≈111× the 14M rate. A
reduced-scope escalation matrix of even just 2 archs × 1 task × 3 seeds =
6 cells at the SAME 20,000-step budget would cost **≈168 GPU-h** — more
than the entire current headroom, and now competing with rung-1's own
larger (≈12.4-16.2 GPU-h realized, ≈124.4 GPU-h worst-case) draw on the
same 135 GPU-h ceiling — a materially smaller cushion than Rev 0's own
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
   bytes), not two.** Structured as two independent passes that must
   agree:
   - **Pass 1 (implementer):** `verify_match_gate.py` (to be built)
     instantiates all three real models/roles, sums `numel()` for param
     counts (mirrors `lm_rd_rung_configs.verify_param_count`), computes
     the measured FLOP profile for each (mirrors
     `DELTANET_REALDATA_DESIGN.md` §4.2's method), AND derives/verifies
     `cap_length` from the contender's real measured **TOTAL-across-all-
     layers** state-byte count (M2, Rev 1 — NOT per-layer) against
     (b-primary)'s real `n_layers/n_heads/d_head` — asserting the ≤1% /
     ≤5% / exact-byte tolerances from §1.3's table.
   - **Pass 2 (independent audit agent, NOT the implementer — `CLAUDE.md`'s
     "the implementer does not review their own work" rule, plus
     "multiple independent adversarial audit rounds catch different bugs"):**
     re-derives all three matched quantities from scratch, from the model
     configs alone, without reading Pass 1's code, and must land within
     the same tolerances — INCLUDING an independent check that (i) the
     capped eviction policy (§1.3(b), sliding/FIFO window) is implemented
     as the simple policy disclosed, not something quietly more
     sophisticated that would understate the memory constraint's real
     bite, and (ii) the byte accounting is genuinely TOTAL-across-layers
     on BOTH sides (M2) — a per-layer vs. total mismatch is exactly the
     kind of silent-imbalance bug this Pass-2 re-derivation exists to
     catch.
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
convention; 7 in Rev 0, now **9 in Rev 1** — F1's new probe machinery is
itself new attack surface and gets its own self-attack, items 8-9)

1. **The escalation-rung budget does not fit the current ceiling at the
   step budget this doc assumes (§1.6).** Tighter again in Rev 1 than in
   Rev 0 (rung-1's own worst-case ceiling draw grew from ≈76.5%
   pre-directive → ≈88% in Rev 0 → **≈97.7% in Rev 1**, because F1's
   mandatory Wave −1 probe/calibration machinery, itself a genuine,
   non-optional cost, adds ≈1.2 GPU-h raw on top of Rev 0's own added
   eval overhead) — this is the single biggest open item, deliberately
   not papered over, and NOW the closest to its own ceiling this design
   has ever been (§1.6's own honest disclosure: "the margin is real...
   but razor-thin").
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
5. **Is the sliding/FIFO eviction policy a fair baseline, or a strawman?**
   §1.3(b) deliberately used the simplest policy and disclosed it as such
   — but a reviewer will reasonably ask whether a modestly smarter,
   still-standard policy (e.g. attention-score-based eviction) would close
   axis 2's gap. Registered as an explicit follow-on robustness check
   (§1.9 item 6 list), not required to win the Rev-0 verdict, but flagged
   for the paper's limitations section either way.
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
   directive), and a smarter-eviction-policy robustness check (item 5
   above). If the PI's intent for "the program's capstone question"
   implies more than this Rev 0 scopes, that should surface now, before
   rung-1 data creates sunk-cost pressure to declare a narrower victory
   than intended.
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

**Does NOT:** authorize any GPU launch (Rev 1, pre-attack-round-2);
decide the 392M escalation rung's exact scope or budget (§1.6, §1.9 item
1 — explicitly deferred, now with an even thinner headroom cushion);
reopen the frozen-bias mechanism story, the key-anchoring capacity/
pool-margin instrument questions, or the reasoning-link geometric-transfer
readout (all cited, none re-litigated); claim a paper-grade benchmark
result from Task 3's internal byte-BPC number without a follow-on
standard-benchmark step; introduce byte-level input (explicitly out of
scope per the PI directive); build the fix-ON/OFF ablation, a
length-generalization task, a smarter-eviction robustness arm, or an
MLP-probe upgrade (all explicitly cut, registered as follow-on
candidates); fully close the Transformer-adapter capacity asymmetry or
the query-continuation asymmetry (§1.9 items 8-9 — bounded by the
probe-capacity null, not eliminated).

---

### 1.11 Sequencing

design (Rev 0) → attack round 1 → NEEDS-REVISION (§1.13) → design (this
doc, Rev 1) → attack round 2 → ... (iterate until DESIGN-CLEARED-FOR-BUILD,
per this program's own standing gauntlet discipline) → build (contender
wiring for the frozen-bias arm already exists; new code needed:
flat-vector ablation mixer WITH its own `q_proj`/`q_conv1d` (§1.3(a)),
standard Transformer with RoPE + the contender's own `FFN` class +
switchable uncapped/capped-KV inference mode (§1.3(b)), the §1.3.1 shared
probe head + per-arm adapters + frozen `T_val` target table + probe-
capacity null harness, the `rd_episode_seed` schedule (F1b),
`verify_match_gate.py` (now total-across-layers, M2), per-task/per-axis/
per-arm calibration wrappers incl. the K/d sweep (M1) and the
Transformer's LR grid (M3), the axis-2 `cap_length` derivation) →
independent build audit (separate agent, per `CLAUDE.md`) → launch rung-1
on GPUs 0-6 (GPU 7 held as pool/overflow) → harvest → escalation-rung
decision (§1.5 rule, mechanically applied) → IF win-or-tie: escalation-rung
design addendum (resolves §1.9 item 1, incl. m2's 392M FLOP-basis
re-derivation) → attack → build → audit → launch 392M → harvest → paper
fold-in (iclr-2027, workshop-2026) either way.

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
  deliberately distinct `PROBE_TARGET_SEED=20260709`.
- Gate precedent: `phase2b_off_cache.py` (`--time-pilot`),
  `run_poolmargin_k84s1943_k90s2043.py` / `run_k69_s1733_contingency.py`
  (dual PI-signoff tokens), `lm_rd_rung_configs.py` (`verify_param_count`
  as the MATCH-GATE's param-counting precedent),
  `phase2b_seedext_stage_minus1.py` (exhaustive-enumeration
  collision-freedom check, the precedent for `rd_episode_seed`'s own
  negative test, F1b).
- **New, not yet built:** the standard-Transformer arch with RoPE, the
  contender's own `FFN` class, standard init, and a switchable
  uncapped/hard-capped-KV inference mode (axis 2's baseline, §1.3(b), M3);
  `verify_match_gate.py`, now total-across-layers (§1.7 item 6, M2); the
  axis-1 power-sketch script that pins `X` (§1.4.1); the flat-vector-
  ablation mixer WITH its own `q_proj`/`q_conv1d` (§1.3(a)); **§1.3.1's
  shared probe head** (`shared_probe`, per-arm `adapter_arm`, `T_val`
  target table, the probe-capacity null harness) — the F1 FATAL's entire
  resolution; the `rd_episode_seed` schedule module (F1b); the
  Transformer's 3-point LR-grid calibration harness (M3).

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

*(End §1. Rev 0 attacked → NEEDS-REVISION (§1.13). Rev 1 (this revision)
resolves every binding finding — §1.14. Awaiting attack round 2.)*
