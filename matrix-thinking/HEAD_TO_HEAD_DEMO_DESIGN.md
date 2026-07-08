# HEAD-TO-HEAD DEMO — the program's capstone question

## §1 DESIGN — HEAD-TO-HEAD DEMO (Rev 0, pre-attack, 2026-07-08) — does the
matrix-native fast-weight model beat a matched conventional baseline on
real tasks at meaningful scale, or is its value honestly bounded?

Status: **Rev 0, pre-attack, zero GPU spent.** This section faces
independent attack rounds before any build/launch. Ratified by the PI as
the program's capstone question. **This Rev 0 absorbs one mid-drafting PI
directive** (received during authorship, before first commit — see the
note at the top of §1.3) that re-pointed the comparison framing toward the
future's binding constraints (data/quality scarcity, inference-memory/HBM
scarcity) rather than today's compute-FLOPs framing; nothing below §1.3
predates that directive — this is the single coherent Rev 0, not a patch
over a stale draft.

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
bf16). This is the exact number a memory-matched Transformer's KV-cache
cap must be pinned against (§1.3/§1.7 MATCH-GATE).

**Why this is the right contender, stated with the honest caveat this
program's own rules demand:**

1. **Trainability at scale is proven for the base architecture.** Track C
   trained `DeltaNetLM` cleanly at four points — 14M (14,048,896 params,
   `d_model=256/n_layers=2/d_state=64`), 98M (97,618,176,
   `d_model=768/n_layers=12/d_state=64`), 392M (391,869,440,
   `d_model=1536/n_layers=16/d_state=128`), 1.31B (1,311,135,488,
   `d_model=2560/n_layers=22/d_state=128`) — `matrix-thinking/deltanet_rd/lm_rd_rung_configs.py`
   `RUNGS`, param-verified within tolerance at every rung
   (`verify_param_count`, real `numel()` sum, not the planning formula).
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
method) and matched **training data** (same corpora, same token count,
same step count — `CLAUDE.md`'s "same dataset for ALL experiments" rule
applied literally: `wikitext-mix-ext` + `openr1-mix-ext`, the exact two
corpora `FROZEN_BIAS_LM_DESIGN.md` used). This one trained model is then
evaluated under two DISTINCT inference-time attention configurations,
which is where its two roles diverge (no extra training cost either way):

  - **Role (b-control): full/uncapped attention** — the disclosed,
    non-gating FLOP-matched control (§1.1). Reports today's-compute-regime
    comparison honestly alongside whatever the primary axes show.
  - **Role (b-primary, axis 2's baseline): hard-capped KV cache**, capped
    at the SAME byte budget as the contender's fixed matrix state (§1.2's
    16,384 bytes/layer at d_state=64, fp32) via the simplest, most
    conservative eviction policy available — a sliding/FIFO window sized
    so `2 (K&V) × n_layers × n_heads × d_head × cap_length × bytes_per_elt
    = contender's total state bytes`, solved for `cap_length` (in
    tokens). Deliberately the SIMPLEST policy, not a sophisticated
    KV-compression scheme, disclosed as such (§1.9 item 7) — this is the
    fair, conservative comparison, not a strawman tilted further toward
    the contender than the byte-budget match alone already tilts it.

**Matching arithmetic table, rung-1 (14M tier) — to be verified for real
by MATCH-GATE (§1.7) before any cell launches, not assumed from this
table:**

| Arch | Params (target) | Training FLOPs | Training data | Inference-memory (fixed-state bytes/layer) |
|---|---|---|---|---|
| Contender (`DeltaNetLM`, frozen-bias `per_token`, λ=0.58) | 14,048,896 (measured, `d_model=256/n_layers=2/d_state=64`) | measured per-token profile × steps × batch × seq_len (pin once) | `wikitext-mix-ext` + `openr1-mix-ext`, same step count as (a)/(b) | 16,384 (fp32; CONSTANT in context length T) |
| (a) Flat-vector ablation | matched to contender within ≤1% | falls out (reported) | same corpora, same step count (data-matched by construction — axis 1's premise) | 16,384 (same construction, vector state) — reported, not the axis-2 comparison arm |
| (b) Standard Transformer | falls out (reported) | matched to contender within ≤5% | same corpora, same step count | role (b-control): grows `O(T)` uncapped, reported at eval horizon; role (b-primary): hard-capped to 16,384 bytes/layer (`cap_length` solved from `n_layers/n_heads/d_head`, pinned by MATCH-GATE) |

The ≤1% / ≤5% / exact-byte-match tolerances are deliberately tighter than
the 15% rung-config planning tolerance in `lm_rd_rung_configs.py` — that
tolerance governs whether a scale-ladder point is "close enough to its
target" for an unrelated study; here the match arithmetic between arms IS
the finding, so it needs its own, much tighter bar, with the
inference-memory byte match held EXACT (an integer token cap solved from
the contender's real measured byte count, not a rounded approximation).

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
(`recovered_frac@0.9`), the same direct-continuous-comparison primitive
used throughout `key_anchoring.py`'s `measure_h1_behavioral_companion` and
`readout_rev7.py` — never argmax/nearest-neighbor. Used at two distinct
settings for the two axes below: a fixed HIGH-LOAD setting (K/d=0.75,
d_state=64 — see §1.9 item 2 for why this specific point, not the more
recent K/d=0.9375 no-cliff point) for axis-1's data-efficiency read, and a
LONG-HORIZON setting (same K, but bind clauses separated from their query
by filler/distractor tokens out past `(b-primary)`'s `cap_length`, using
`DeltaNetRDTaskConfig`'s own existing `buf_len`/`T_bind`/`query_len`
sequence-structure parameters — confirming this parametrization already
supports variable bind-query distance is a build-time verification item,
not assumed here) for axis-2's memory-matched read.

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

**Escalation scope is NOT decided by this Rev 0 doc — flagged as an open,
load-bearing item, not silently assumed (§1.6 shows why).**

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
| Eval overhead — now larger than the original brief's estimate: adds axis-1's multi-checkpoint logging + axis-2's second (capped) inference pass on top of the FROZEN_BIAS_LM_DESIGN.md-observed ≈32% ratio (1.6/5.05); budgeted at ≈50% of training to cover both additions | — | 3.4074 |
| Calibration + per-task/per-axis timing pilots (incl. the axis-1 power sketch that pins X, and the axis-2 `cap_length` derivation) | ~4 cells × 0.2524 | 1.0096 |
| MATCH-GATE verification (§1.7) — now 3 matched quantities, still CPU-only | — | 0.0000 |
| **Raw total** | | **≈11.23 GPU-h** |

**Still meets the ≤15 GPU-h raw target**, with less margin than the
original brief's ≈9.75 GPU-h (the added eval-overhead line is the honest
cost of the richer two-axis read, not hand-waved away). Enforced ceiling
(circuit-breaker, not expected spend — this program's realized/estimate
ratios have run 97-122% historically): bracket at **10× raw ≈ 112.3
GPU-h**, mechanically enforced by a pre-launch timing pilot exactly like
`phase2b_off_cache.py --time-pilot`'s existing pattern.

**Ledger against the shared 135 GPU-h `FROZEN_BIAS_LM_DESIGN.md` program
ceiling.** The brief cites "~123 headroom" — that is the *pre-execution
planning estimate* from that design's own §8.1. The **current, realized**
figure (post rung-1 + its mechanism follow-on wave, `FROZEN_BIAS_LM_DESIGN.md`
§12 closing ledger) is **≈7.672/135 GPU-h spent**, i.e. **≈127.33 GPU-h
headroom** — this design uses that current, verified figure. Rung-1's own
enforced ceiling (≈112.3 GPU-h) would consume ≈88% of that headroom in the
worst-case abort scenario — tighter than before, and flagged honestly:
this leaves comparatively little slack for the escalation rung under the
SAME shared ceiling (below). Expected realized spend (≈100-130% of raw,
this program's typical range) is ≈11-15 GPU-h, ≈9-12% of headroom.

**Escalation-rung cost — flagged now, not resolved now.** Track C's own
realized 392M rate: 128.3 GPU-h / 91,552 steps = 0.0014017 GPU-h/step →
**≈28.03 GPU-h per 20,000-step-equivalent cell** — ≈111× the 14M rate. A
reduced-scope escalation matrix of even just 2 archs × 1 task × 3 seeds =
6 cells at the SAME 20,000-step budget would cost **≈168 GPU-h** — more
than the entire current headroom, and now competing with rung-1's own
larger (≈11-15 GPU-h realized, ≈112.3 GPU-h worst-case) draw on the same
135 GPU-h ceiling. **The escalation rung, as currently scoped, does not
fit the existing ceiling at rung-1's own step budget, full stop.**
Resolving this is explicitly NOT decided here (§1.9 item 1): the likely
resolution is a shorter step budget at 392M (this is a comparison wave,
not a full pretrain) and/or a reduced arch/task/seed count restricted to
whichever axis won at rung-1, but the exact numbers require the
escalation rung's own dedicated design addendum and GPU-h re-derivation,
gated on rung-1's own verdict. Escalation is not pre-authorized by this
document.

---

### 1.7 Gates

Standing machinery, reused verbatim from this program's own precedent, not
re-derived:

1. **Calibration-first.** One real training cell at each task's target
   config, run to completion, checked against the val-loss/`recovered_frac`
   band this task's own small-scale precedent (Task D/E for tasks 1-2,
   `FROZEN_BIAS_LM_DESIGN.md`'s own val-loss ceilings for task 3) predicts,
   BEFORE the 27-cell sweep launches. **This calibration cell also does
   double duty for the two-axis restructure:** it is where axis 1's power
   sketch pins `X` (§1.4.1) and where axis 2's `cap_length` is derived and
   verified from the contender's real measured state-byte count (§1.4.2) —
   both BEFORE the full sweep launches, not after, per the `CLAUDE.md`
   calibration-first rule applied literally to the new quantities the PI
   directive introduced.
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
     `cap_length` from the contender's real measured per-layer state-byte
     count against (b-primary)'s real `n_layers/n_heads/d_head` — asserting
     the ≤1% / ≤5% / exact-byte tolerances from §1.3's table.
   - **Pass 2 (independent audit agent, NOT the implementer — `CLAUDE.md`'s
     "the implementer does not review their own work" rule, plus
     "multiple independent adversarial audit rounds catch different bugs"):**
     re-derives all three matched quantities from scratch, from the model
     configs alone, without reading Pass 1's code, and must land within
     the same tolerances — INCLUDING an independent check that the capped
     eviction policy (§1.3(b), sliding/FIFO window) is implemented as the
     simple policy disclosed, not something quietly more sophisticated
     that would understate the memory constraint's real bite.
   - Disagreement between passes is a hard launch-block, full stop.

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
convention; now 7, given the mid-drafting restructure adds new attack
surface)

1. **The escalation-rung budget does not fit the current ceiling at the
   step budget this doc assumes (§1.6).** Now tighter than the
   pre-directive draft (rung-1's own worst-case ceiling draw grew from
   ≈76.5% to ≈88% of headroom because of the added eval overhead) — this
   is the single biggest open item and is deliberately not papered over.
2. **Is K/d=0.75 the right high-load point for Task 1's axis-1 read?** We
   deliberately did NOT use the more recent, more favorable K/d=0.9375
   no-cliff point — but K/d=0.75 was itself only ever measured at a
   different `d` than this wave's own d_state=64 config
   (`KEY_ANCHORING_DESIGN.md`'s K=48 curve and `KEY_ANCHORING_SCALING_DRAFT.md`'s
   d=96 no-cliff finding neither transfer cleanly to d_state=64 without
   their own calibration). The calibration gate (§1.7 item 1) must include
   a dedicated K/d sweep at d_state=64 for the contender AND (a) before
   any margin is treated as final.
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

**Does NOT:** authorize any GPU launch (Rev 0, pre-attack); decide the
392M escalation rung's exact scope or budget (§1.6, §1.9 item 1 —
explicitly deferred); reopen the frozen-bias mechanism story, the
key-anchoring capacity/pool-margin instrument questions, or the
reasoning-link geometric-transfer readout (all cited, none re-litigated);
claim a paper-grade benchmark result from Task 3's internal byte-BPC
number without a follow-on standard-benchmark step; introduce byte-level
input (explicitly out of scope per the PI directive); build the
fix-ON/OFF ablation, a length-generalization task, or a smarter-eviction
robustness arm (all explicitly cut, registered as follow-on candidates).

---

### 1.11 Sequencing

design (this doc, Rev 0) → attack round 1 → ... (iterate until
DESIGN-CLEARED-FOR-BUILD, per this program's own standing gauntlet
discipline) → build (contender wiring for the frozen-bias arm already
exists; new code needed: flat-vector ablation mixer, standard Transformer
with a switchable uncapped/capped-KV inference mode, `verify_match_gate.py`,
per-task/per-axis calibration/timing-pilot wrappers, the axis-1 power
sketch, the axis-2 `cap_length` derivation) → independent build audit
(separate agent, per `CLAUDE.md`) → launch rung-1 on GPUs 0-6 (GPU 7 held
as pool/overflow) → harvest → escalation-rung decision (§1.5 rule,
mechanically applied) → IF win-or-tie: escalation-rung design addendum
(resolves §1.9 item 1) → attack → build → audit → launch 392M → harvest →
paper fold-in (iclr-2027, workshop-2026) either way.

---

### 1.12 Reproducibility pointers

- Contender: `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` (`DeltaNetLM`,
  `DeltaNetLMMixer`, `DeltaNetLMBlock`), frozen-bias mechanism in the same
  file (`build_frozen_bias_table`, `apply_frozen_bias_blend`, lines
  178-246) plus `matrix-thinking/FROZEN_BIAS_LM_DESIGN.md` for the
  mechanism's own design history.
- Scale configs: `matrix-thinking/deltanet_rd/lm_rd_rung_configs.py`
  (`RUNGS`, `verify_param_count`).
- Task 1/2 instrument: `matrix-thinking/deltanet_rd/grammar_rd.py`
  (`DeltaNetRDTaskConfig`, `sample_batch_rd`, `_permutation_graph`,
  `_assert_injective_entities`), readout pattern from
  `matrix-thinking/deltanet_rd/key_anchoring.py`
  (`measure_h1_behavioral_companion`) and `readout_rev7.py`.
- Task 3 instrument: `FROZEN_BIAS_LM_DESIGN.md`'s eval/corpora pipeline
  (`wikitext-mix-ext`, `openr1-mix-ext` via
  `rebuild_lm_corpora_rd.py`/`build_mix_corpora_rd.py`).
- CI machinery: `matrix-thinking/deltanet_rd/reasoning_link_probe.py`
  (`delta_ci_n`, `CI_T_975_BY_DF`).
- Gate precedent: `phase2b_off_cache.py` (`--time-pilot`),
  `run_poolmargin_k84s1943_k90s2043.py` / `run_k69_s1733_contingency.py`
  (dual PI-signoff tokens), `lm_rd_rung_configs.py` (`verify_param_count`
  as the MATCH-GATE's param-counting precedent).
- **New, not yet built:** the standard-Transformer arch with a switchable
  uncapped/hard-capped-KV inference mode (axis 2's baseline, §1.3(b));
  `verify_match_gate.py` (§1.7 item 6); the axis-1 power-sketch script
  that pins `X` (§1.4.1); the flat-vector-ablation mixer (§1.3(a)).

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

*(End §1. Rev 0 attacked → NEEDS-REVISION (§1.13). Rev 1 in progress.)*
