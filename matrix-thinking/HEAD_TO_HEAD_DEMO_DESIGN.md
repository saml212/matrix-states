# HEAD-TO-HEAD DEMO — the program's capstone question

## §1 DESIGN — HEAD-TO-HEAD DEMO (Rev 4, post-diagnosis revision,
2026-07-09) — does the matrix-native fast-weight model beat a matched
conventional baseline on real tasks at meaningful scale, or is its value
honestly bounded?

Status: **Rev 4, pre-attack (round 5 pending), ≈11.43 GPU-h realized
to date (build/deploy smoke + calibration rounds 1-2 + §1.21 diagnosis;
~2.3 GPU-h round-3 re-calibration about to launch, §1.6).** Rev 0
(§1.1-1.12 below) was independently attacked and returned NEEDS-REVISION
(§1.13, retained verbatim as the record: 2 FATAL-caliber + 3 MAJOR + 2
MINOR findings), resolved by Rev 1 (§1.14). Rev 1 was independently
attacked a second time and ALSO returned NEEDS-REVISION (§1.15, retained
verbatim as the record: 2 FATAL findings on axis 2 — a degenerate
cap_length at rung-1 and an undisclosed train/eval attention mismatch,
both with mandated INFERENCE-ONLY fixes — + 4 MAJOR + 2 minor). Rev 2
resolved every §1.15 finding structurally, staying inference-only per the
binding M-NEW-4 budget constraint (§1.16). Rev 2 was independently attacked
a third time and ALSO returned NEEDS-REVISION (§1.17, retained verbatim as
the record: 2 FATAL findings — a mathematically broken tap-expressivity
diagnostic, numerically proven broken by the attack itself, and an
underpowered/biased `M*` statistical procedure — + 2 MAJOR + 2 minor).
Rev 3 resolved every §1.17 finding, replacing the broken diagnostic with a
K-simultaneous-bindings construction that revision itself numerically
executed pre-commit (§1.3.1.5), replacing the `M*` search with a
DESCENDING fixed-sequence gatekeeping test that actually controls axis-2's
within-grid FWER (§1.4.2), pinning `n_layers_transformer=2` and remapping
the verdict tiers onto the resulting eligible grid with the WIN bar made
HARDER, not easier (§1.4.2), and closing the remaining MAJOR/minor
findings (§1.7 gate 2's M-sweep timing pilot, the `seed_idx` runtime
bound, and the `M=1` cost double-count) — §1.18 maps each §1.17 finding to
its exact Rev 3 resolution. Rev 3 was independently attacked a fourth time
and returned **DESIGN-CLEARED-FOR-BUILD** (§1.19). The design was BUILT
(§1.20, independently audited, fixed, re-audit FIXES-VERIFIED-CLEARED),
DEPLOYED and ran on the box: gate-1 calibration round 1 (13 cells) FAILED
its band on Tasks 1/2 (`recovered_frac@0.9=0` in all 9 task-1/2 cells, all
arms) with a step-500 gradient-ratio diagnostic pointing at an
under-weighted aux loss; the pre-registered `aux_weight` dial fired
(2.0, parity) and round 2 (`_auxrev2`, 9-cell re-run) achieved gradient
parity but ALL arms plateaued at `probe_cos_mean≈0.12-0.22` —
`recovered_frac@0.9` still 0 everywhere, an INSTRUMENT-DESIGN-caliber
HARD-STOP with the pre-registered revision path exhausted. A dedicated
box diagnosis (§1.21, executed evidence, 0.08 GPU-h) traced the plateau to
its root cause: the objective (`loss_CE_lm + aux_weight·probe_cosine_loss`)
contains NO gradient pressure toward the queried answer — `loss_CE_lm` is
structurally retrieval-blind (each key appears once at bind time; query
windows never enter it) and the aux loss alone converges to a cheaper
EPISODE-MEMBERSHIP local optimum whose analytic ceiling, `1/√K`, matches
every observed plateau exactly. **This Rev 4 adopts §1.21's option (f)**
— the sole surviving option after five others (longer training, tied-
embedding targets, an MLP probe, a bar re-pin, K de-load) were each
REFUTED or found COUNTERPRODUCTIVE with executed evidence: add an
answer-token cross-entropy term at the query position, symmetric across
all three arms, making recall NECESSARY rather than merely available.
§1.3.1's instrument (frozen `T_val`, linear probe, `rf@0.9` decision
metric) is UNCHANGED; only the training objective and its pre-registered
diagnostic ladder change — see §1.22 for the full changes map. Ratified by
the PI as the program's capstone question. **This design absorbs one
mid-drafting PI directive** (received during Rev-0 authorship, before
first commit — see the note at the top of §1.3) that re-pointed the
comparison framing toward the future's binding constraints (data/quality
scarcity, inference-memory/HBM scarcity) rather than today's compute-FLOPs
framing; nothing below §1.3 predates that directive.

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

**Axis 2's WIN/TIE/LOSE cells above are themselves determined by a
crossover-multiplier `M*` read off a pinned, descending fixed-sequence
statistical test over a 6-point memory-multiplier grid, not a single CI
(R3-F2/R3-F3, Rev 3, full statement in §1.4.2) — an `INDETERMINATE` fourth
read is possible (an unresolved CI straddle after the §1.8 contingency
path) and is explicitly NOT a WIN, reported as underpowered and treated as
TIE-adjacent for escalation eligibility (§1.5).**

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

**1.3.1.3 Training regime — REWRITTEN, Rev 4 (§1.21's option (f)): joint
THREE-term objective, from step 0, never zero-shot post-hoc.** Rev 0-3's
two-term objective (`loss_CE + aux_weight·probe_cosine_loss`, `aux_weight`
originally pinned **0.1**, REVISED to **2.0** by the round-2 calibration
dial firing — both figures retained below as history, §1.21) is
**superseded**: the box diagnosis (§1.21, executed evidence) proved that
objective contains no gradient pressure toward the queried answer at all
— `loss_CE` is structurally retrieval-blind (each key appears exactly
once at bind time; query windows never enter it) and the aux loss alone
converges to a cheaper EPISODE-MEMBERSHIP local optimum (`1/√K` ceiling)
that satisfies `probe_cosine_loss` without ever learning to answer the
query. Every arm's `shared_probe` + `adapter_arm` parameters, PLUS now
the backbone's own LM head at the query answer position, are optimized
JOINTLY with the backbone by the SAME optimizer:

```
loss_total = loss_CE_lm
           + ce_answer_weight * CE_answer(logits_at_query_answer_pos, answer_token)
           + aux_weight       * mean_over_queries(1 - cos(pred, target))
```

`CE_answer` is standard next-token cross-entropy, scored ONLY at the
query's designated answer position (`grammar_rd.py`'s existing
query/answer positional convention, unchanged) against the true answer
entity's token id — it reads the model's OWN native LM-head logits
(full-vocab softmax), **not** the §1.3.1.1-1.3.1.2 `shared_probe`/
`adapter_arm`/`T_val` machinery, which is untouched by this addition and
continues to produce `probe_cosine_loss` exactly as before. This is
deliberate: `CE_answer` makes recall NECESSARY for the loss to go down at
all (closing §1.21's root cause), while `rf@0.9` — read off the SAME,
unchanged §1.3.1 probe — stays the sole axis-1/axis-2 DECISION metric,
never promoted or demoted by this change (the Nichani argmax-vs-exact-
continuous-recovery distinction, `CLAUDE.md`, restated in the diagnostic
ladder below).

**Per-arm mechanics, pinned exactly — ALL THREE ARMS SYMMETRIC in loss
term, asymmetric only in how the query-position logits are produced (the
SAME §1.3.1.2 asymmetry this design already discloses, item 9 below, now
load-bearing for training rather than only for the probe read):**

- **Recurrent arms (contender, ablation).** The query window runs as a
  **continuation** from the cached bind-phase state, a SEPARATE forward
  call from the bind-phase pass:
  ```
  S_T = model.forward(bind_tokens, return_states=True).final_states   # bind phase, unchanged
  logits_query = model.forward(query_tokens, initial_states=S_T)      # NEW, Rev 4: the continuation
  ```
  `forward(query_tokens, initial_states=S_T)`'s signature takes ONLY
  `query_tokens` and `initial_states` — it has no argument, and therefore
  no computational path, back to `bind_tokens` or any other raw
  bind-phase tensor. The continuation's output is thus a PURE function of
  `(S_T, query_tokens)` alone: **P=1 is preserved BY CAUSALITY, not by a
  vacuous shape check** — the same class of argument §1.3.1.2 already
  makes for the probe tap's `q_query`, extended one level up (there the
  query is kept OUT of the recurrence entirely; here it deliberately
  enters the recurrence, seeded from `S_T`, because next-token prediction
  at the answer position requires exactly the kernel's own standing
  `o_t = read(S_t, q_t)` mechanism — what stays protected is only that it
  can never reach BEHIND `S_T` into raw bind tokens it has not already
  been causally summarized through).
  **Blank-out verification, extended (spec'd here per §1.21's own
  instruction — a negative test, not an assertion, per `CLAUDE.md`'s
  "run the negative unit test... to completion" rule):** after computing
  and caching `S_T` from the real `bind_tokens` once, overwrite
  `bind_tokens` in place with a fresh `torch.randint_like` draw of valid
  token ids (a hard corruption, not a subtle perturbation), then call
  `model.forward(query_tokens, initial_states=S_T)` again with the SAME
  cached `S_T`. `logits_query` (and therefore `CE_answer`) must be
  numerically IDENTICAL (same `atol` this program's own existing
  negative tests use) whether or not the corruption happened — because
  the corrupted tensor is never passed into the continuation call. Wired
  as its own Wave −1 (A) probe-smoke negative-test row (§1.6, §1.7 gate
  3), one per recurrent arm, run once at smoke time, not per training
  step.
- **Standard Transformer.** §1.3.1.2's existing single forward pass over
  `[bind-phase tokens (+ filler) ++ query_window]` ALREADY materializes
  LM-head logits at every position, including the query answer position
  — `CE_answer` for this arm costs reading out one more position's
  logits and computing one more cross-entropy term, **near-zero
  incremental cost**, no second forward pass. The arm's disclosed
  attend-to-raw asymmetry (item 9) is unchanged by this addition (see
  the item-9 addendum below): the Transformer's query was already part
  of its own context for every other purpose; `CE_answer` does not make
  that asymmetry any better or worse, it only reads out logits the
  forward pass already computed.

`ce_answer_weight`: **pinned default 1.0**, calibrated by the SAME
step-500 gradient-ratio dial the original `aux_weight` used, **extended to
cover all three loss terms.** At step 500 of the calibration cell (§1.7
gate 1, extended), log each term's OWN isolated backbone-gradient norm —
`ce_grad_norm_backbone` (from `loss_CE_lm` alone), `aux_grad_norm_backbone`
(from `aux_weight·probe_cosine_loss` alone, unchanged mechanism), and
`ce_answer_grad_norm_backbone` (from `ce_answer_weight·CE_answer` alone,
NEW) — via the same per-loss isolated-backward technique the original
overshoot guard already used. **Pass condition:** both
`aux_grad_norm_backbone` and `ce_answer_grad_norm_backbone` sit within
10× of `ce_grad_norm_backbone` (mirrors the original `exceeds_10x_trigger`
check, now checked for two auxiliaries instead of one). **Revision rule,
pinned exactly (closes the "which weight moves" ambiguity a two-
auxiliary dial introduces that the original one-auxiliary dial never
had to resolve):** if exactly one auxiliary's ratio exceeds 10×, revise
that ONE weight toward parity (`new_weight = old_weight × observed_ratio`,
the same rounding convention the original `aux_weight` 0.1→2.0 revision
used). **If BOTH auxiliaries exceed 10× simultaneously, revise only the
LARGER-ratio deviation this round** — the more urgent parity violation —
**and defer the other to the NEXT calibration round's own one revision**,
never both in the same round: this is what "one revision per calibration
round max" means operationally, and it exists so a plateau's cause is
never confounded by two simultaneous weight changes landing in the same
re-run. `aux_weight` itself carries forward round 2's already-calibrated
value (**2.0**, parity achieved at ratios 1.3-3.6, §1.21) into Rev 4's
round-3 re-calibration as its own starting point — round 3's dial check
therefore evaluates whether 2.0 STILL holds parity now that a third loss
term shares the same backward pass, not whether to re-derive it from
0.1 again. Both weights are frozen before the 27-cell sweep launches, not
after. The probe and the answer head train THROUGHOUT training (from step
0), consistent with §1.4's own already-stated premise ("every task below
trains the model end-to-end WITH the readout as (part of) the actual
training objective").

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
extended, Rev 2, to also run §1.3.1.5's tap-expressivity diagnostic —
now the Rev-3 K-simultaneous-bindings construction, R3-F1 — same
CPU-feasible, no-backbone harness, no separate cost line).

**1.3.1.5 K-simultaneous-bindings tap-expressivity diagnostic (M-NEW-2,
Rev 2 — REPLACED, Rev 3, R3-F1: the Rev-2 single-pair diagonal/off-diagonal
energy-split diagnostic was independently NUMERICALLY EXECUTED by attack
round 3 and found mathematically broken.** Cosine similarity is
scale-invariant, and `target_effective := S_star @ q` stays near-parallel
to `t` for ANY `alpha > 0` (the off-diagonal-zeroing perturbation is only
`O(1/√d)`) — the "calibration curve" the Rev-2 construction traced was a
degenerate step function (cos≈0.9999 at `alpha=0.25..1.0`, dropping to 0
only at the single point `alpha=0`), not the graduated bound it was meant
to be; the matvec side was tautologically 1.0 by construction throughout.
**ROOT CAUSE (§1.17):** single-`(q,t)`-pair reachability — a diagonal
(Hadamard) read can always hit exactly ONE target exactly (`d` unknowns,
`d` constraints), so the diagonal-vs-matvec expressivity gap only bites
under MULTIPLE SIMULTANEOUS bindings held in one state, which the
single-pair construction never tested. M-NEW-2 was therefore disclosed
honestly but its mechanism was nonfunctional — this section is the
working replacement, itself numerically verified below BEFORE being
committed, per the round-3 process mandate ("a second broken diagnostic
would be a gauntlet failure").

**Construction, pinned exactly.** For `K` simultaneous `(q_i, t_i)`
bindings, `K ∈ {1, 2, 4, 8, 16, 32, 48, 64}` (`d_state=64` at rung-1 — the
diagnostic is re-run at `d_state=128` for the escalation rung, same code,
new `d`), draw `q_i`, `t_i` as independent random unit vectors in
`R^{d_state}` (fresh RNG each trial; seed independent of `T_val`'s
`PROBE_TARGET_SEED` and `key_anchoring.py`'s `ANCHOR_INIT_SEED`, per this
design's own never-share-a-frozen-table-seed rule), and fit each tap
family's BEST POSSIBLE state to ALL `K` pairs SIMULTANEOUSLY (not one pair
at a time, which is exactly what let the retired diagnostic slip past the
real bound):

```
# matvec tap family: S in R^{d x d} minimizing sum_i ||S q_i - t_i||^2
#   Q = [q_1 ... q_K]   (d x K, columns are the queries)
#   T = [t_1 ... t_K]   (d x K, columns are the targets)
#   S = T @ pinv(Q)                       # least-squares / minimum-norm solution
#   matvec_pred_i = S @ q_i  for each i   # = (S @ Q).T, one matmul

# Hadamard tap family: s in R^d minimizing sum_i ||s (.) q_i - t_i||^2
# NOTE (per R3-F1's own instruction): this least-squares problem is
# SEPARABLE PER-COORDINATE -- coordinate j appears only in the j'th term
# of every ||...||^2, so each s_j is an INDEPENDENT 1-D least-squares fit,
# solvable in closed form (no numerical solver needed):
#   s_j = ( sum_i q_i[j] * t_i[j] ) / ( sum_i q_i[j]^2 )      for j = 1..d
#   had_pred_i = s (.) q_i  for each i
```

Score both families' predictions against the SAME frozen-formula
cosine-recovery metric this design uses everywhere
(`F.cosine_similarity(pred_i, t_i, dim=-1)` @ 0.9), report
`recovered_frac@0.9` = fraction of `(trial, i)` pairs clearing the
threshold, over `>=100` trials per `K` (200 used below).

**Expected analytic behavior, stated before the numbers (R3-F1's own
requirement).** The matvec tap has `d^2` free parameters against `K*d`
scalar constraints — for `K <= d` and generic (random) `q_i`, `Q` has full
column rank `K`, the system `S @ Q = T` is under-determined-but-consistent,
and the least-squares/minimum-norm solution recovers every target
NEAR-EXACTLY (`recovered_frac@0.9 -> 1` for all `K <= d`). The Hadamard tap
has only `d` free parameters (one scalar per coordinate) against the same
`K*d` constraints — for `K=1` it is ALSO exactly solvable (`d` equations,
`d` unknowns, one per coordinate); for `K >= 2`, each coordinate's single
scalar `s_j` must simultaneously satisfy `K` generically-inconsistent
equations, so the fitted read collapses toward the noise floor of an
uninformative 1-parameter regression on independent random data
(`recovered_frac@0.9 -> 0` for `K >= 2`, with the mean cosine of the
best-fit read decaying `~ 1/sqrt(K)`).

**Numerically executed, pre-commit — `d=64`, `n_trials=200` per `K`,
`threshold=0.9`, seed reuses `PROBE_TARGET_SEED=20260709` for
continuity/reproducibility, fresh independent `q_i`/`t_i` draws every
trial (script: pure-numpy least-squares/closed-form fit, no backbone, no
GPU; run in the design-review scratchpad before this commit, exactly the
process R3-F1 mandated):**

| K | matvec `recovered_frac@0.9` | Hadamard `recovered_frac@0.9` | Hadamard mean cosine |
|---|---|---|---|
| 1 | 1.000000 | 1.000000 | 1.0000 |
| 2 | 1.000000 | 0.000000 | 0.7056 |
| 4 | 1.000000 | 0.000000 | 0.4964 |
| 8 | 1.000000 | 0.000000 | 0.3506 |
| 16 | 1.000000 | 0.000000 | 0.2459 |
| 32 | 1.000000 | 0.000000 | 0.1747 |
| 48 | 1.000000 | 0.000000 | 0.1433 |
| 64 | 1.000000 | 0.000000 | 0.1226 |

**The numbers confirm the expected analytic behavior exactly — no spec
change needed** (per the round-3 process instruction's own contingency:
"if your numbers contradict the expected behavior, fix the spec, not the
numbers" — here they agree). matvec recovers every `K <= d=64` exactly
(the MINIMUM cosine across all `200*64=12,800` `(trial,i)` pairs at `K=64`
was `1.00000000` — checked directly, not inferred from the mean, since
`K=64` makes `Q` square and occasionally near-singular by chance, which
produced large-but-non-corrupting intermediate values in the least-squares
solve, verified clean). Hadamard recovers `K=1` exactly (trivially, one
scalar per coordinate against one constraint) and then COLLAPSES
COMPLETELY at `K>=2` (zero items cleared 0.9 out of `200*K` trials at
every `K` from 2 to 64), with the mean cosine decaying as a clean
`~1/sqrt(K)` power law (0.7056 -> 0.4964 -> 0.3506 -> 0.2459 -> 0.1747 ->
0.1433 -> 0.1226, consistent with successive `sqrt(2)`-ratio steps in
`K`), matching the "correlation floor of a 1-parameter fit to independent
random data" prediction exactly.

**This IS the tap-expressivity bound for axis-1 interpretation (restated,
now on a working instrument):** any observed contender-vs-ablation
`recovered_frac@0.9` gap at a real load `K` (the K-entity pool, §1.4's M1
K/d re-pin) SMALLER than this diagnostic's gap at that SAME `K` cannot be
attributed to architecture (matrix state vs. vector state) over tap
(matvec-read vs. Hadamard-read) — a Hadamard tap facing `K>=2` simultaneous
bindings is expected to collapse toward the `~1/sqrt(K)` floor REGARDLESS
of what the underlying vector state actually stores, on this diagnostic's
own exact, adversarial-fit construction. Concretely, at the primary load
`K=32` (§1.4, M1), this diagnostic's own floor is `recovered_frac@0.9 = 0`
for the Hadamard tap family — so if the real trained ablation's axis-1
`recovered_frac@0.9` at `K=32` is ALSO near zero, that alone does not
distinguish "the vector state failed to hold the bindings" from "the
vector state held the bindings fine, but the Hadamard tap structurally
cannot read out more than one of them" — the two readings are
observationally identical at this diagnostic's own worst-case bound, and
only a real, trained-state measurement (a genuine follow-on, out of scope
this wave, same limit already disclosed for the retired Rev-2 diagnostic)
can separate them. **What this diagnostic does NOT do (unchanged from Rev
2's own honest scope limit):** it does not measure the ablation's REAL
trained states' diagonal/off-diagonal energy split; it establishes the
theoretical K-vs-recovery mapping a real measured number is read against.
Cost: CPU-only (a `pinv`/least-squares solve per trial on
`d_state`-dimensional vectors, no backbone forward pass, no training
loop), folds into §1.6 Wave −1 item B at no additional GPU-h, unchanged
from Rev 2.

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
    assert 0 <= seed_idx < 50, \
        f"seed_idx={seed_idx} out of bounds [0,50) -- seed_idx*STRIDE_SEED_RD would overflow " \
        f"into the next TASK_BASE key's own 500,000-wide range (R3-F5, Rev 3 runtime assert -- " \
        f"the ckpt_idx bound below had one from M-NEW-1/Rev 2; seed_idx did not, an asymmetry " \
        f"attack round 3 caught: seed_idx=50 on task1_calib collides with task1_stress's own seed_idx=0)"
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
negative-test cell (`ckpt_idx = STRIDE_SEED_RD` must raise). **The
`seed_idx` bound is likewise now a runtime assert (R3-F5, Rev 3 — the
`ckpt_idx` bound had one from M-NEW-1, `seed_idx` did not, an asymmetry
attack round 3 caught), exercised by its own dedicated negative-test cell
(`seed_idx = 50` must raise).**

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
d_model × bytes_per_elt)`; MATCH-GATE, §1.7 gate 6, **verifies the PINNED
`n_layers_transformer=2` (R3-F3, below) actually achieves the ≤1%/≤5%
param/FLOP match within the admissible `d_model≈256` family, and computes
the real table from that pinned config**, not this illustrative sketch,
and NOT an arbitrary value the matching build happens to fall out to.)

**`n_layers_transformer`, PINNED (R3-F3, Rev 3 — previously left to fall
out of the param/FLOP-matching build; attack round 3 found this unpinned
choice could silently gate which verdict tiers are even REACHABLE, since
the two FLOP-admissible options, `n_layers∈{1,2}`, land at different
floor-eligibility boundaries).** Pinned: **`n_layers_transformer = 2`**,
matching the contender's own `n_layers=2` at rung-1 (§1.2's architecture
pin) — the same "remove an asymmetry axis wherever the design has a free
choice" logic already applied to the shared `FFN` class and the pre-norm
convention (§1.3(b)): a depth-matched Transformer is the more defensible
"same overall architecture depth budget" comparison, not an unpinned
choice that happens to fall out of whichever `d_model`/`n_layers`
combination the FLOP-matching search finds first. **Consequence, verified
(the exact arithmetic R3-F3 asked to be checked):** at `n_layers=2`, fp32,
`cap_length(M=1) = 1×32,768/(2×2×256×4) = 8` tokens — BELOW the
`≥13`-token floor (query_len 6 + one bind clause 7) — so `M=1` is a
floor-excluded, descriptive-only grid point at this pinned config
(unchanged fact, already stated in the grid-endpoint rule below; what's
NEW is that `M=1` can therefore never be assigned as `M*`, closing off
Rev 2's own `LOSE if M*≤1` tier by construction). The smallest ELIGIBLE
grid point is `M=2` (`cap_length = 2×32,768/(2×2×256×4) = 16 ≥ 13`, clears
the floor). This reshapes which `M*` values are reachable at all:
`M* ∈ {2, 4, 8, 16, 32, ∞}` only — the verdict tiers are remapped onto
this eligible set below, after the statistical procedure that determines
`M*` itself.

**Grid-endpoint rule, pinned exactly (F-NEW-1; skip DIRECTION corrected,
Rev 3, R3-F2 — the descending fixed-sequence procedure below walks
`M=32→...→2`, so the cells that MAY be skipped are the SMALLER-`M` cells
below a clean stop, not "remaining larger-`M`" as an ascending framing
would have it).** The DEFAULT is to run the full fixed grid
`M ∈ {1,2,4,8,16,32}` (6 points; `M=1` REPORTED but floor-excluded from
`M*`-eligibility at the pinned `n_layers=2` config, R3-F3, above) for
every `(task, horizon, seed)` combination — this is what §1.6 item F's own
90-pass budget prices, as the conservative default. **Cost-saving
exception (pre-registered, not a post-hoc stopping rule):** if the
fixed-sequence walk (§1.4.2's win-margin procedure, below) hits a CLEAN
(non-straddling) failure-to-reject at some eligible `M_0` while descending
from `M=32`, the cells for the SMALLER eligible `M`'s below `M_0` for that
`(task, horizon)` MAY be skipped — under the monotonicity assumption the
walk already relies on, they would also fail to reject. **This shortcut is
NEVER used to certify the `CONFIRMED no-crossover` / `M*=∞` STRONGEST-WIN
claim** (below) — that claim specifically requires the FULL eligible grid
to have been run and every point individually confirmed clean, the more
defensible, assumption-light bar for a headline claim, even though the
bare fixed-sequence theory would permit stopping after a single clean
non-rejection at `M=32` itself. Two conditions justify `M=32` as the
grid's fixed,
cost-bounded practical ceiling if no crossover is found earlier: (i)
**floor clearance** — the eligible grid clears the `≥13`-token
minimum-viable floor (holds at least one bind clause + the query window)
starting at `M=2` (`cap_length(2)=16≥13` at the pinned `n_layers=2`,
fp32); `M=1` is the only grid point BELOW the floor at this pin
(`cap_length=8<13`) and is reported as descriptive-only, NEVER eligible to
set `M*` (avoids an ill-posed "crossover" whose cause is "can't hold the
query window at all," not memory-insufficiency); (ii) **cost bound** —
`M=32` is exactly the practical ceiling this design's own Wave −1 budget
prices (6-point grid total: 5 eligible `M`-values priced in §1.6 item F,
plus the descriptive `M=1` point priced inside the eval-overhead line,
R3-F6); it is NOT claimed to be the mathematically exhaustive point past
which the
Transformer's capped role provably equals its uncapped (b-control) role at
every tested horizon (that would require `M` in the low hundreds at the
primary horizon, well beyond this wave's affordable, inference-only
scope) — if the fixed-sequence walk (below) exhausts the eligible grid
without ever rejecting, this design reports **"`M*` not reached within the
tested grid"** honestly (the `CONFIRMED no-crossover` / `INDETERMINATE`
split, below, R3-F2c), not a false claim of having found the true
asymptote.

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

**Win margin, statistical procedure (R3-F2, Rev 3 — REPLACES Rev 2's
single-CI-per-grid-point framing, which attack round 3 found (a)
systematically biased toward the strongest-WIN default under n=3 noise,
(b) silent on within-axis-2 multiplicity across the 6-point grid, and (c)
conflated "underpowered" with "confirmed no-crossover" inside the
`M*=∞` edge case).**

**(a) Fixed-sequence (gatekeeping) test, DESCENDING order `M = 32 → 16 →
8 → 4 → 2`, stopping at the first FAILURE TO REJECT — the FWER-controlling
direction, replacing the informal "search upward for the first success"
reading Rev 2's prose implied.** At each `M`, test `H_M`: "the contender's
gap over `(b-primary)` at the primary horizon (H4, 902 tokens) is
`≥ 0.20`" against the one-sided alternative "`< 0.20`," via the SAME
paired `delta_ci_n` CI (§1.8) — REJECT `H_M` (crossover established at
this `M`) iff the CI's lower bound excludes 0.20 (the whole CI sits below
0.20). Test `M=32` first; continue to the NEXT SMALLER eligible `M` only
if the CURRENT `M` was REJECTED; STOP at the first `M` that is NOT
rejected. `M*` = the smallest `M` in the resulting unbroken chain of
rejections (the last `M` tested and rejected before the stop).

**Why descending, not ascending (the load-bearing correction).**
Fixed-sequence testing controls FWER at the nominal per-test level `α`
PRECISELY WHEN testing continues only while REJECTING and stops at the
FIRST NON-REJECTION (Maurer, Hothorn & Lehmacher, 1995, "Multiple
comparisons in drug clinical trials and preclinical assays: a-priori
ordered hypotheses," *Biometrie in der Chemisch-Pharmazeutischen
Industrie* 6:3-18 — the originating fixed-sequence/gatekeeping result,
restated without an independence assumption by the sequential rejection
principle, Goeman & Solari 2010). The argument: under ANY true
configuration of the 5 eligible `H_M`'s, let `m` be the FIRST true null
encountered in test order. Every `H_M` tested before `m` is a false null
(rejecting it is a correct decision, not a Type-I error). The FIRST
opportunity for a genuine Type-I error is therefore exactly at `m` — and
because the procedure stops the moment it fails to reject, `H_m` is
tested AT MOST once, at its own nominal `α`. So
`P(FWER event) = P(reach m) × P(falsely reject H_m | reached m) ≤ α`,
regardless of how many hypotheses lie beyond `m`, and WITHOUT assuming
independence between the `M`-grid's tests (a genuine, disclosed benefit —
the grid points share the same trained checkpoint and are plausibly
correlated). **This requires testing in the direction where nulls become
MORE likely false as the sequence progresses** (a standard
minimum-effective-dose construction: descending `M`, so "the current
point rejects" is the expected direction of travel, and running out of
budget to keep rejecting is the natural stopping condition). Testing in
the OPPOSITE, ascending direction ("start at `M=1`/`2`, stop at the first
`M` where the gap comes within margin") does NOT enjoy this guarantee —
worked example: under the global null that the contender's advantage
never actually closes anywhere on the grid, an ascending "test until you
find a rejection" walk over the 5 eligible points at nominal `α=0.05` each
has a `≈1-(1-0.05)^5 ≈ 22.6%` chance of a spurious early stop somewhere in
the sequence, not 5%. This is the bug R3-F2 caught and this is the fix.

**What IS controlled, stated exactly:** the probability of the
fixed-sequence procedure EVER declaring a finite `M*` when the truth is
`M*=∞` (no true crossover anywhere on the eligible grid) is `≤ α` (the
nominal 95% CI's own `α=0.05`), for BOTH axis-2's primary read (Task 1,
H4) and, independently, Task 2's own non-gating secondary read (same
grid, same procedure, its own separate `α` budget, never pooled with
Task 1's). **What is NOT controlled, disclosed honestly:** (i) this `α` is
INTERNAL to axis 2 — it does not change or subsume the existing
between-axis OR-win inflation already disclosed in §1.8
(`1-0.95²≈9.75%` across axis 1 and axis 2's own headline tests); the two
disclosures compose, they do not replace each other. (ii) [corrected per
attack round 4, §1.19] **FWER control does NOT depend on monotonicity** —
the Maurer-Hothorn-Lehmacher first-true-null argument holds for ANY true
configuration of the five hypotheses; monotonicity of the TRUE gap in `M`
(more memory helps or does not hurt the baseline — scientifically
reasonable but UNVERIFIED at design time) affects only whether the
reported `M*` is a UNIQUE, cleanly interpretable crossover. If the
empirical, noisy `n=3` gap is non-monotonic across the grid, the
pre-registered stopping rule still applies MECHANICALLY (stop at the
first non-rejection walking down from `M=32`, never peek past it even if
a smaller `M` would have looked like a rejection by chance) — FWER
remains controlled; the cost is potentially reporting a coarser `M*`
than a hypothetical noise-free re-ordering would find. This
interpretability trade-off is disclosed, not resolved, and named again
in §1.18's own "not closed with full margin" note.

**(b) Straddle rule (R3-F2 — extends the EXISTING §1.8 seed-extension
contingency to the grid; the contingency itself is unchanged, costed, and
PI-gated, only its TRIGGER condition is generalized here from a
single-CI axis test to a fixed-sequence grid walk; canonical registration
in §1.8).** If the CI at the `M` where the fixed-sequence walk would
otherwise STOP (the deciding, boundary `M` for that task's `M*`)
STRADDLES 0.20 (the CI contains 0.20, excluding it on neither side) — as
opposed to a CLEAN non-rejection (CI lower bound `> 0.20`, confidently no
crossover at this `M`) — the walk does NOT finalize `M*` at that boundary.
Instead, the EXISTING §1.8 seed-extension contingency (n=3→n=9,
already-pinned CI machinery, already costed at ≈3.03 GPU-h raw per axis
extended, already gated through the `var_ratio>4.0` batch-effect check)
triggers for THAT SPECIFIC `(task, M)` cell-set only — never the whole
grid, since the fixed-sequence structure guarantees at most ONE cell per
task can ever be the deciding boundary (testing stops there by
construction, so no cell downstream is ever reached to also need
extension). Worst case across both Task 1 (primary) and Task 2
(secondary): 2 extended cells, ≈6.06 GPU-h raw, PI-gated, inside remaining
headroom — bounded, not open-ended. Only after the extended-seed CI
resolves the straddle (clean reject or clean non-reject) does `M*`
finalize for that task.

**(c) `M*=∞`, split (R3-F2 — replaces Rev 2's single, conflated edge
case):**

- **`CONFIRMED no-crossover`** — this claim OVERRIDES the bare
  fixed-sequence stopping rule's own cost-saving shortcut (the
  Grid-endpoint rule's own disclosure, above: a single clean non-rejection
  at `M=32` would already suffice to stop the walk and, under the
  monotonicity assumption, imply no crossover anywhere smaller). For THIS
  specific, strongest-possible claim, the full eligible grid is run
  regardless, and the walk traverses the ENTIRE eligible grid (`M=32` down
  to `M=2`) WITHOUT ever rejecting, AND every one of those 5 eligible
  points' CIs cleanly excludes crossover (lower bound `> 0.20`, no
  unresolved straddles anywhere — any straddle encountered was resolved
  clean by its own §1.8 extension) — reportable
  as the STRONGEST WIN (falls inside the `M*≥8` tier below by
  construction), with the same `cap_length(32)`-vs-primary-horizon
  interpretive caveat Rev 2 already disclosed (residual quality-gap
  confound if `cap_length(32)` ever exceeded H4 for a DIFFERENT resolved
  config than this pin's own — not reached at the pinned `n_layers=2`
  config: `cap_length(32)=256 < 902`).
- **`INDETERMINATE`** — the walk stops on a straddle that remains
  UNRESOLVED even after its own §1.8 seed-extension (the n=9 CI still
  straddles 0.20). **NOT reportable as a WIN** (R3-F2's own explicit
  instruction) — reported as underpowered, naming the specific straddle
  point(s) by `(task, M)`, alongside whatever tier the LARGER,
  already-rejected `M`'s in the chain independently support as a lower
  bound (e.g., if `M=32,16,8` all cleanly rejected and `M=4` is the
  unresolved straddle, the honest report is "`M*≥8` confirmed (WIN tier
  already reached), upper bound INDETERMINATE" — a real, disclosable
  partial result, not a null report). For escalation purposes (§1.5),
  `INDETERMINATE` is treated the same as `TIE` — escalation-eligible, not
  a `LOSE`, exactly the "underpowered or borderline n=3 read" case §1.5
  already names as the reason an escalation rung exists.

**Verdict tiers, remapped onto the eligible grid (R3-F3) — every tier now
REACHABLE under the pinned config, which was NOT true of Rev 2's generic
`≥4/=2/≤1` tiers once `M=1` is excluded (Rev 2's own `LOSE if M*≤1` tier
was UNREACHABLE at this pin, since `M*` could never legally equal 1):**

- **LOSE if `M* ≤ 2`** — i.e. `M*=2`, the smallest eligible point: the
  transformer already matches the contender within the old 0.20 margin at
  the least memory this design can even test at this pin. Plain-language
  reading unchanged from Rev 2's spirit ("the constant-memory story has no
  empirical teeth at this scale"), re-anchored to the smallest testable
  point rather than an unreachable `M=1`.
- **TIE if `M* = 4`.**
- **WIN if `M* ≥ 8`** — the actual value (8, 16, 32, or the `M*=∞`
  `CONFIRMED no-crossover` case above) is reported alongside the tier,
  exactly as Rev 2 already did for its own `≥4` tier.
- **INDETERMINATE** — see (c) above; NOT a WIN, treated as TIE-adjacent
  for escalation eligibility (§1.5).

**Disclosed explicitly, per R3-F3's own instruction: this is a THRESHOLD
SHIFT, not a re-scoping to make winning easier.** Every tier boundary
moved UP by exactly one grid step relative to Rev 2's generic tiers (LOSE
`≤1`→`≤2`; TIE `=2`→`=4`; WIN `≥4`→`≥8`) — driven mechanically by
floor-eligibility (`M=1` dropping out of the eligible set), not by any
discretionary loosening or tightening. The net effect is that the WIN bar
is HARDER to clear under this pin (needs `M*≥8`, an 8× memory multiplier,
where Rev 2's unpinned table would have called `M*≥4` a WIN) — the honest
direction, disclosed rather than obscured, consistent with `CLAUDE.md`'s
standing instruction to never re-scope a design step to make a win
easier.

The old exact-parity `M=1` point is **RETAINED and REPORTED as a
disclosed descriptive datum** (the first row of the sweep table,
floor-excluded, never `M*`-eligible per R3-F3) — it is no longer, by
itself, the decision quantity; `M*` is. The remapped LOSE/TIE tiers above
subsume the same qualitative reads Rev 1's original exact-parity point
would have driven, so no information is lost, only re-contextualized
inside the full dose-response curve and the now-eligible grid.

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
resolving exactly that kind of underpowered or borderline n=3 read. **An
axis-2 `INDETERMINATE` read (R3-F2c, §1.4.2 — an unresolved `M*` CI
straddle even after the §1.8 seed-extension) is treated identically to a
TIE for this rule: escalation-eligible, never treated as a LOSE.**

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
| **Wave −1 (B) probe-capacity sanity null + K-simultaneous-bindings tap-expressivity diagnostic** (§1.3.1.4, §1.3.1.5 — M-NEW-2/R3-F1, same CPU-feasible no-backbone harness) — frozen-random-state control, NO backbone forward pass | negligible | 0.02 |
| **Wave −1 (C) Task-1 K/d calibration** (M1) — PRIMARY point K/d=0.5 run to completion (3 arms × 1 full cell, ALSO serves as Task-1's own §1.7 gate-item-1 calibration cell) + ONE disclosed above-cliff stress point K/d=0.75 at quarter-budget, locate-only (3 arms × 1 quarter cell) — extended to all 3 arms incl. the Transformer's own capped-cache behavior | 3×(1 full + 1 quarter) | 0.95 |
| **Wave −1 (D) Task-2 calibration** — target hop-depth config, held-out-hop guard re-verified, full cells, all 3 arms | 3 cells | 0.76 |
| **Wave −1 (E) Task-3 calibration** — contender's own rung-1 config already calibrated by `FROZEN_BIAS_LM_DESIGN.md`'s own rung-1 run (not re-run); ablation reuses the contender's pinned LR, 1 full cell; Transformer's 3-point LR grid (M3) at quarter-budget, 3 cells | 1 full + 3 quarter | 0.44 |
| **Wave −1 (F) Axis-2 memory-multiplier sweep, INFERENCE overhead** (F-NEW-1, Rev 2, new — explicitly bumped, NOT silently folded into the eval-overhead line above, since that line's ≈50%-of-training ratio was calibrated to ONE capped inference pass per checkpoint, not this differently-shaped addition) — **`5` (R3-F6, Rev 3: `M∈{2,4,8,16,32}` — EXCLUDES `M=1`, which the eval-overhead line above ALREADY prices as "axis-2's original (single, M=1) capped inference pass"; Rev 2's own `6`-value count silently double-counted that pass, caught by attack round 3) M-values × 3 horizons × 2 tasks × 3 seeds = 90** short (≤1,798-token) forward-only inference passes over the already-trained (b) checkpoints, no backward/optimizer step; priced at a deliberately padded ≈5s/pass wall-clock (dominated by short-pass fixed overhead — eval-harness setup, batch construction, checkpoint access — not raw FLOPs, at this ≤1,798-token/14M-param scale; **R3-F4, Rev 3: this ≈5s/pass figure is a design-time ASSUMPTION, to be REPLACED by a real measured value from the §1.7 gate-2 M-sweep timing pilot before the fan-out launches**) → `90 × 5s = 450s ≈ 0.125 GPU-h` | 90 inference passes | 0.125 |
| MATCH-GATE verification (§1.7) — now 3 matched quantities (incl. the M2 total-across-layers byte check and the F-NEW-1 per-`M` `cap_length` table, verified against the R3-F3-pinned `n_layers_transformer=2`) plus the M-NEW-1 5-key collision smoke and the R3-F5 `seed_idx` bound, all still CPU-only | — | 0.0000 |
| **Raw total** | | **≈12.5672 GPU-h** |

**Still meets the ≤15 GPU-h raw target**, tighter than both the original
brief's ≈9.75 GPU-h and Rev 0's own ≈11.23 GPU-h, and only ≈0.1296 GPU-h
above Rev 1's own independently-reproduced ≈12.4376 GPU-h (§1.15) — **the
ENTIRE Rev 2→Rev 3 cost delta is the axis-2 M-sweep's inference overhead,
now correctly counted at 90 passes not 108 (R3-F6, Rev 3 — the fix SHRANK
item F from Rev 2's own now-superseded ≈0.15 to ≈0.125 GPU-h, §1.16's own
record)**; F-NEW-2's sink-eviction patch and M-NEW-1's widened
`TASK_BASE` remain genuinely zero-added-GPU-h fixes, unchanged from Rev 2.
Enforced ceiling (circuit-breaker, not expected spend — this program's
realized/estimate ratios have run 97-122% historically): bracket at **10×
raw ≈ 125.672 GPU-h**, mechanically enforced by a pre-launch timing pilot
exactly like `phase2b_off_cache.py --time-pilot`'s existing pattern, **now
EXTENDED to the axis-2 M-sweep fan-out specifically (§1.7 gate 2, R3-F4)**
— the ≈5s/pass assumption inside this very bracket is exactly the number
that pilot exists to replace before the 90-pass fan-out is authorized.

**Ledger against the shared 135 GPU-h `FROZEN_BIAS_LM_DESIGN.md` program
ceiling.** The brief cites "~123 headroom" — that is the *pre-execution
planning estimate* from that design's own §8.1. The **current, realized**
figure (post rung-1 + its mechanism follow-on wave, `FROZEN_BIAS_LM_DESIGN.md`
§12 closing ledger) is **≈7.672/135 GPU-h spent**, i.e. **≈127.33 GPU-h
headroom** — this design uses that current, verified figure, UNCHANGED by
Rev 3 (Rev 3 adds no new training cost against the ledger; R3-F6's fix
SHRANK the inference-only addition instead). Rung-1's own enforced ceiling
(**≈125.672 GPU-h**, R3-F6-corrected) would consume **≈98.70%** of that
headroom in the worst-case abort scenario — still tighter than Rev 1's own
≈97.7% draw, but a genuine, if small, IMPROVEMENT over Rev 2's own
now-superseded ≈98.86% (the exact double-count R3-F6 caught and fixed):
**the margin is real (≈1.658 GPU-h, ≈1.30% of headroom)** — thinner than
Rev 1's ≈2.93 GPU-h/2.3% margin, but no longer the single thinnest figure
this design has carried (that distinction now belongs to Rev 2's own,
corrected, ≈1.45 GPU-h/1.14% read, §1.16's record). This is the direct,
disclosed accounting benefit of R3-F6's fix, layered on top of the SAME
F-NEW-1/F-NEW-2 inference-only discipline M-NEW-4 already required
(unchanged — Rev 3 adds no new training cells either; the alternative, a
genuinely cap-trained `(b)` arm, +0.76 raw/+7.6 at bracket, remains
registered as an explicit, costed, unaffordable-this-wave follow-on, §1.9
item 7, never silently absorbed). **The bracket still fits (125.672 <
127.33), so no training cell, Wave −1 gate, or seed count was touched to
make it fit.** **[Accounting-regime reconciliation, added per attack
round 4 §1.19 finding B:] the 125.672 circuit-breaker bracket and the
§1.8 straddle-extension bound (≤6.06 GPU-h real spend) are MUTUALLY
EXCLUSIVE scenarios, never additive — the bracket fires only on an
anomalous ~10×-over run that ABORTS before completion (no completed CI
exists to straddle in that world); the extension triggers only on a
COMPLETED run, whose realized cost falls in the disclosed
12.57-16.34 GPU-h expected range (16.34 + 6.06 = 22.4 ≪ 127.33). A
naive sum (125.672 + 6.06 = 131.73 > 127.33) misreads the regimes.**
This arithmetic is now flagged as re-derived by THREE
independent passes (§1.15's Rev-1 re-derivation, §1.17's Rev-2
re-derivation AND its own R3-F6 catch, this Rev-3 correction), which
tightens confidence but does not retire the standing instruction: the
≈5s/pass assumption inside item F remains a DESIGN-TIME assumption, not a
measured one, until the R3-F4 timing pilot (§1.7 gate 2) runs for real —
this is now the single most likely place a future re-derivation would
still move the bracket's fit, restated as a live item in §1.18. This
leaves near-zero slack for the escalation rung under the SAME shared
ceiling (below) — tighter than Rev 1 already flagged it to be, marginally
less tight than Rev 2. Expected realized spend (≈100-130% of raw, this
program's typical range) is **≈12.57-16.34 GPU-h, ≈9.87-12.83% of
headroom** — essentially unchanged from Rev 1/Rev 2's own expected-case
reads (the R3-F6 fix moves the WORST-CASE circuit-breaker bracket, not the
expected-case range materially). **Flagged in §1.18 as the one item this
revision could improve but not fully retire (the timing-pilot
measurement)** — an attack-round-4 agent should treat the bracket's fit as
better-verified, not settled.

**Escalation-rung cost — flagged now, not resolved now.** Track C's own
realized 392M rate: 128.3 GPU-h / 91,552 steps = 0.0014017 GPU-h/step →
**≈28.03 GPU-h per 20,000-step-equivalent cell** — ≈111× the 14M rate. A
reduced-scope escalation matrix of even just 2 archs × 1 task × 3 seeds =
6 cells at the SAME 20,000-step budget would cost **≈168 GPU-h** — more
than the entire current headroom, and now competing with rung-1's own
larger (≈12.57-16.34 GPU-h realized, ≈125.672 GPU-h worst-case, Rev 3)
draw on the same 135 GPU-h ceiling — a materially smaller cushion than Rev 0's own
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

**REV 4 COST UPDATE — round-3 re-calibration + realized-spend
reconciliation (does NOT touch the main 27-cell sweep's own raw/bracket
figures, unchanged below).**

**Round-3 re-calibration cost, derived from MEASURED wall-clock (not the
flat 0.2524 GPU-h/cell planning rate; this is the same 9-cell shape as
Wave −1 items C+D, re-run under §1.3.1.3's new three-term objective):**
task1_calib (K/d=0.5, full, 3 arms) + task1_stress (K/d=0.75, quarter, 3
arms, locate-only) + task2_calib (full, 3 arms) = 9 cells. Per-arm
FULL-cell wall-clock, box-measured across rounds 1-2: contender ≈10 min
(0.167 GPU-h), ablation ≈19-20 min (0.325 GPU-h, midpoint), transformer
≈27 min (0.45 GPU-h); quarter-budget cells scale ≈1/4:

| Cell group | Arms × shape | GPU-h |
|---|---|---|
| task1_calib + task2_calib (full) | 2 tasks × (0.167+0.325+0.45) | 1.883 |
| task1_stress (quarter) | 1 task × (0.167+0.325+0.45)/4 | 0.236 |
| Base re-run subtotal | | **2.119** |
| Diagnostic-ladder overhead (rung 1 reuses `CE_answer`'s own logits, near-zero; rung 2 is one small extra linear-classifier fit/cell) | ≈8.5% of base | 0.181 |
| **Round-3 total** | | **≈2.300 GPU-h** |

Wall-clock to completion: 9 cells over 7 parallel GPU slots (GPU 7 held
as pool/overflow, §1.11) — worst case 2 sequential batches bounded by the
slowest cell (~27 min transformer + ladder overhead) → **≈55-60 min**,
comfortably under an hour.

**Realized spend so far (pulled from `STATE.md`'s own live ledger, not
re-derived from scratch):** `STATE.md`'s LEDGERS section reads
**"frozen-bias: 11.43/135 (~123 headroom)"** — **11.43 GPU-h spent,
123.57 GPU-h headroom**, exact. This figure is the CURRENT authoritative
realized total: the pre-h2h program baseline (7.672 GPU-h, unchanged
since Rev 3, `FROZEN_BIAS_LM_DESIGN.md` §12) PLUS the h2h wave's own
build/deploy smoke (Wave −1 A+B, ≈0.07 GPU-h), calibration round 1
(13 cells, `aux_weight=0.1`, ≈2.15 GPU-h at the original Wave −1 C/D/E
raw estimate), calibration round 2 (`_auxrev2`, 9-cell C+D rerun at
`aux_weight=2.0`), and the §1.21 diagnosis (0.08 GPU-h, explicit).

**Reconciliation flag (disclosed, not resolved here — genuinely
unresolved, not papered over):** a bottom-up reconstruction using THIS
revision's own measured per-arch full-cell rates (10/19.5/27 min),
applied retroactively to rounds 1 (13 cells) and 2 (9 cells) on the
theory that real per-arch wall-clock is a property of the model/hardware
and should apply equally regardless of which round or `aux_weight` ran,
sums to **≈4.98 GPU-h** for the h2h wave's own calibration-phase spend
(this figure INCLUDES the §1.21 diagnosis pass's own 0.08 GPU-h, stated
here explicitly per attack round 5's own instruction, §1.23 — it is not
an additional, separately-hidden cost) — against the **≈3.76 GPU-h**
(`11.43−7.672`) the ledger line implies. The
**≈1.2 GPU-h gap** is most plausibly either (a) `STATE.md`'s ledger line
not yet reflecting round 2/diagnosis's full cost (a bookkeeping-lag
pattern this doc's own STATE.md record already flags elsewhere, "Git
blemish noted"), or (b) round 1 genuinely running faster before some
instrumentation/checkpointing overhead was added. **Flagged for the next
attack round or a dedicated ledger-reconciliation pass — it changes the
MARGIN's size below, not the fits/doesn't-fit verdict (both readings
clear the bar, see below).**

**Updated margin, primary (ledger-anchored) reading:**
- Realized-to-date (11.43) + round-3 (2.30) = **13.73/135 GPU-h**,
  headroom remaining for the still-pending 27-cell sweep + M-sweep:
  **135 − 13.73 = 121.27 GPU-h.**
- The still-pending scope's own raw estimate is UPDATED with `STATE.md`'s
  own already-disclosed timing-pilot revision ("9 pilots PASSED,
  projected training 11.675 GPU-h... msweep 1.1-1.5 s/pass real vs 5s
  assumed") — this SUPERSEDES Rev 3's stale planning-rate sub-estimate
  (10.3472 GPU-h) and **closes Rev 3's own R3-F4-flagged open item**
  ("the single most likely place a future re-derivation would still move
  the bracket's fit," §1.6/§1.18): training+eval-overhead ≈11.675 GPU-h
  (pilot-measured, ~14% over the flat-rate plan — ablation/transformer
  run slower than the uniform 0.2524 GPU-h/cell rate, contender runs
  faster, net overrun) + M-sweep ≈0.03-0.04 GPU-h (`90 passes ×
  1.1-1.5s ≈ 99-135s`, down from the 0.125 GPU-h/5s-per-pass design-time
  assumption) + MATCH-GATE 0 = **≈11.71 GPU-h revised raw** for the
  still-pending portion.
- Enforced 10× circuit-breaker bracket for the still-pending portion
  (unchanged convention, applied to the portion that has genuinely not
  launched yet — realized-to-date spend is actual, not re-hedged at 10×,
  since the 10× multiplier exists specifically to cover cost SURPRISE on
  cells that have not run, not to re-inflate a cost already measured):
  **≈117.1 GPU-h.**
- **Bracket fits: 117.1 < 121.27 GPU-h remaining headroom. Margin ≈4.17
  GPU-h (≈3.1% of the 135 GPU-h shared ceiling).** Looser in absolute
  GPU-h than Rev 3's own 1.658 GPU-h margin (against the narrower 127.33
  pre-h2h-headroom denominator) specifically because the M-sweep's real
  measured rate (1.1-1.5s/pass) came in far under the 5s/pass design-time
  assumption — an improvement that more than offsets three rounds of real
  calibration-phase spend.

**Conservative (bottom-up-anchored) cross-check, using the ≈4.98 GPU-h
reconstruction instead of the ledger-implied ≈3.76 GPU-h:**
- Realized-to-date-after-round-3 = `7.672 + 4.98 + 2.30` = **14.95/135
  GPU-h**, headroom **120.05 GPU-h**.
- Same still-pending bracket (117.1 GPU-h) → **margin ≈2.95 GPU-h
  (≈2.2%)** — thinner, comparable in order of magnitude to Rev 1's own
  2.93 GPU-h margin, but still clearly positive.
- **Either way, the bracket fits — the reconciliation gap changes the
  margin's size (4.17 vs 2.95 GPU-h), not the verdict.** No training
  cell, gate, or seed count was touched to make either reading fit.

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

   **1a. THE DIAGNOSTIC LADDER (NEW, Rev 4, pre-registered per §1.21's
   own instruction — a gate-1 extension).** **Scope, corrected (R5-F3,
   attack round 5, §1.23 — the original "EVERY calibration cell" phrasing
   overclaimed):** the ladder runs for the `grammar_rd` calibration cells
   only — `task1_calib` (both the primary K/d=0.5 load and the
   stress/locate-only K/d=0.75 load) and `task2_calib` — never for
   `task3_calib`, which has no query/answer-position structure for the
   ladder's rungs to read and keeps its OWN anchored `[1.90,2.60]`
   val-loss band (§1.7 gate 1's opening paragraph, unchanged) instead.
   §1.21's diagnosis took a full box-diagnosis
   pass (0.08 GPU-h, offline probe study) to attribute a `rf@0.9=0`
   plateau to its root cause, because no cheaper, standing instrument
   existed to separate "task not learned" from "info not in the tap"
   from "the continuous-recovery instrument itself is the bottleneck."
   The ladder makes that attribution immediate and automatic, logged in
   every calibration cell's own JSON alongside the existing step-500
   gradient-ratio check:
   - **Rung 1 — LM-head/native answer top-1 at the query position,
     EPISODE-RESTRICTED.** "Chance" is `1/K` over the episode's own `K`
     candidate entities (`1/32 = 3.125%` at the primary load, §1.4's M1
     re-pin) — **never** a global-vocab or global-entity-pool chance
     figure (the ad hoc §1.21 diagnosis run used a looser, ~0.93%
     global-pool baseline; the pre-registered ladder deliberately
     tightens this, a harder bar to clear, going forward). **PASS bar:
     `> 3×` episode-restricted chance** (`> 9.375%` at K=32). **This
     rung is a disclosed GATE — a sanity check that recall is happening
     ANYWHERE in the model — and is NEVER promoted to the WIN metric.**
     Argmax/top-1 readout is exactly the decoding mode `CLAUDE.md`'s own
     hard rule warns inflates apparent capacity (Nichani, Lee & Bietti,
     ICLR 2025, arXiv:2412.06538 — a rank-1 state can support `≈d`
     argmax-recoverable associations while supporting far fewer under
     exact continuous recovery); rung 1 passing does NOT imply `rf@0.9`
     (rung 3) will pass, and only rung 3 is evidentially load-bearing for
     the axis-1/axis-2 WIN claims.
   - **Rung 2 — trained linear identity-classifier on the tap.** A
     freshly-trained linear classifier (separate from `shared_probe`,
     which does continuous regression, not classification) predicts
     WHICH of the episode's `K` entities is being queried, from
     `state_summary_raw`/`adapter_arm`'s own tap. **PASS bar: `> 3×`
     episode-restricted chance**, same `1/K` convention as rung 1. This
     rung answers "is the identifying information in the tap AT ALL,"
     independent of whether the CONTINUOUS probe machinery can recover
     it — the same `>` vs `<` chance test as rung 1, applied one layer
     downstream.
   - **Rung 3 — `rf@0.9` (UNCHANGED decision metric).** §1.3.1's own
     cosine-recovery instrument, exactly as pinned since Rev 0-1. This
     rung, and only this rung, feeds axis-1/axis-2 WIN/TIE/LOSE.
   - **Attribution table (any rung failing = immediately attributable,
     no fresh diagnosis pass required):** rung 1 FAILS → the task itself
     was never learned by ANY channel (a task-learning problem, not an
     instrument problem). Rung 1 PASSES, rung 2 FAILS → the task is
     learned somewhere, but this specific tap point does not carry the
     identifying signal (a tap-placement problem). Rungs 1-2 PASS, rung 3
     FAILS → the tap carries the signal (rung 2 proves a classifier can
     extract it beyond chance) but the exact-continuous-recovery probe
     cannot (a probe-capacity/probe-training problem — the Nichani gap,
     concretely observed rather than only theoretically flagged). All
     three PASS → healthy cell, proceed.
   - **The membership-oracle signature (§1.21's own root cause,
     recorded as a standing tell).** Log, per cell, BOTH `probe_cos_mean`
     (rung 3's own raw pre-threshold score) AND `cos(pred,
     episode_mean_of_T_val_rows)` (NEW this revision — the predicted
     probe vector's cosine similarity to the mean of the episode's `K`
     `T_val` target rows, not to the true individual target). A future
     plateau showing `probe_cos_mean` near the analytic ceiling `1/√K`
     for that cell's `K` (`0.1768` at K=32, matching every §1.21
     plateau exactly) **together with** an elevated
     `cos(pred, episode_mean)` (`≥0.85`, calibrated off §1.21's own
     measured 0.93-0.94) is diagnosed as the SAME membership-oracle
     local optimum without a fresh box-diagnosis pass — the model/probe
     learned "which K-set this episode is," not "which entity was
     queried," a geometric artifact of `K` i.i.d. random unit targets
     (`‖episode_mean‖ ≈ 1/√K`, and `cos(episode_mean, t_i) ≈ 1/√K` for
     any constituent `t_i`), not evidence about the backbone's own
     recall capability.

   **1b. BANDS UPDATE (NEW, Rev 4) — Tasks 1/2 FULL-cell gate-1 pass bar,
   made explicit and quantitative for the first time** (STATE.md's own
   round-1/round-2 record flagged Tasks 1/2 as "SANITY-ONLY," i.e. no
   hard numeric band existed before this revision; Task 3's own anchored
   `[1.90,2.60]` val-loss band, `FROZEN_BIAS_LM_DESIGN.md`-derived, is
   unchanged and not addressed here): **a task-1/2 FULL cell (task1_calib
   at K/d=0.5, task2_calib) PASSES gate 1 iff BOTH (i) rung-1 answer
   accuracy `> 3×` episode-restricted chance AND (ii) `rf@0.9 > 0`** —
   the ladder's rung 1 and rung 3 jointly, rung 2 remaining diagnostic-
   only (used for attribution on a FAIL, not itself a pass/fail
   criterion). **Stress/locate-only cells (task1_stress, K/d=0.75) stay
   EXEMPT from this band, as before** — they are disclosed
   above-primary-load stress points, not gating cells, and never were
   (§1.6 Wave −1 item C's own "locate-only" framing, unchanged).
2. **Timing pilots, mechanical enforced abort.** One real cell per
   arch×task pair measured for wall-clock before its own seed-fan-out
   launches; if the projected cost for that pair exceeds its share of the
   §1.6 ceiling, the chain hard-aborts before spending it. **Extended, Rev
   3 (R3-F4) — the axis-2 `M`-sweep inference fan-out gets its OWN scoped
   timing pilot**, separate from the per-arch×task pilots above: 2
   `M`-values (one small, one large — `M=2` and `M=32`) × 1 already-trained
   checkpoint × 1 horizon (H4, the primary), measured for real wall-clock
   BEFORE the full 90-pass (§1.6 item F, post-R3-F6) fan-out launches. The
   measured `s/pass` REPLACES the ≈5s/pass design-time assumption in §1.6
   item F's own bracket arithmetic before that fan-out is authorized.
   **Build requirement, pinned:** the trained checkpoints for role
   `(b-primary)` MUST be held RESIDENT in memory across all inference
   passes for a given `(task, seed)` (loaded once, evaluated at every
   `M`/horizon combination), not reloaded from disk per pass — reload-per-
   pass is the specific failure mode a real pilot exists to catch (a naive
   implementation could run 3-6× the padded estimate, per R3-F4's own
   finding). **De-scope order, pre-registered, not decided post-hoc:** if
   the measured bracket would exceed the §1.6 127.33 GPU-h headroom
   figure, drop grid cells in this order until it fits: (1) `M=32` first
   (the single most expensive point, least load-bearing once a WIN is
   already established at a smaller `M` via the fixed-sequence walk,
   §1.4.2); (2) `H8` next (the least-informative secondary horizon). `H4`
   (primary) and the floor-eligible `M∈{2,4,8,16}` core are NEVER
   de-scoped by this rule — a bracket overrun that would require cutting
   those instead is a hard abort, not a silent re-scope.
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
     tolerances from §1.3's table at EVERY `M`, not just `M=1`. **New,
     Rev 3 (R3-F3): Pass 1 must VERIFY the PINNED `n_layers_transformer=2`
     actually achieves the ≤1%/≤5% param/FLOP match within the admissible
     `d_model≈256` family** — a pin that fails the match tolerance is
     itself a launch-block, requiring either a `d_model` adjustment within
     the `n_layers=2` family or an escalation to design revision, never a
     silent fallback to `n_layers=1` (which would silently re-open the
     LOSE-tier-unreachable bug R3-F3 closed).
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
     extended grid, and (iv) **(R3-F3, Rev 3)** the ELIGIBLE subset for
     `M*` purposes correctly EXCLUDES `M=1` at the pinned `n_layers=2`/fp32
     config (`cap_length=8<13`, floor-excluded) — an independent audit
     that silently treats `M=1` as `M*`-eligible would wrongly re-open the
     LOSE tier Rev 3 closed off by construction (§1.4.2).
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
(6 more seeds × BOTH compared arms = 12 cells × 0.2524 GPU-h ≈ 3.03
GPU-h raw per axis extended [formula corrected per attack round 4,
§1.19 — the 3.03 bottom line was always the 12-cell figure; the stated
multiplier previously omitted the ×2-arms factor], inside remaining
headroom) — and gated through the SAME batch-effect
variance-ratio check `REASONING_LINK_DESIGN.md` §16.19.5 registered
(`var_ratio > 4.0` → flag, do not silently pool old and new seed cohorts)
before any pooled reading is treated as decision-grade. This directly
avoids the exact trap that produced §16.20's BATCH-EFFECT-FLAGGED outcome.

**Extended trigger for axis 2's grid (R3-F2b, Rev 3 — this contingency
previously referenced only a single-CI axis test; attack round 3 found
that framing did not cover axis 2's Rev-3 fixed-sequence `M`-grid
procedure, §1.4.2, at all).** Axis 1 (and any single-CI axis-2 read)
triggers this contingency on a straddle of THAT AXIS'S OWN CI, as above,
unchanged. Axis 2's fixed-sequence `M`-grid procedure (§1.4.2) triggers it
more narrowly: ONLY the specific `(task, M)` cell-set at which the
descending fixed-sequence walk would otherwise STOP on a STRADDLING (not a
clean) non-rejection — never the whole grid, and never a cell the walk
never reaches, since the fixed-sequence structure guarantees at most ONE
cell per task can ever be the deciding boundary (testing stops there
mechanically). This is the SAME contingency (n=3→n=9, same CI machinery,
same per-cell cost, same `var_ratio>4.0` gate), re-triggered per boundary
cell rather than per axis; worst case 2 extended cells (Task 1 primary +
Task 2 secondary, if both independently straddle at their own boundary
`M`) ≈6.06 GPU-h raw, PI-gated, bounded by the fixed-sequence structure
itself, not open-ended.

---

### 1.9 Attack-yourself — self-attack round 0 (minimum 5, per house
convention; 7 in Rev 0, 9 in Rev 1, 11 in Rev 2, 12 in Rev 3, now **14 in
Rev 4** — F-NEW-1's memory sweep and F-NEW-2's sink patch got their own
self-attack in Rev 2 (items 10-11); Rev 3 folded in a cross-campaign
caveat (item 12); Rev 4 adds items 8/9's own addenda (§1.21's answer-CE
mechanics touch both) plus two new items on the DEPLOY-PIN-1 scope
expansion and the M-NEW-4 sanctioning question (items 13-14))

1. **The escalation-rung budget does not fit the current ceiling at the
   step budget this doc assumes (§1.6).** Tighter at every revision through
   Rev 2 (rung-1's own worst-case ceiling draw grew from ≈76.5%
   pre-directive → ≈88% in Rev 0 → ≈97.7% in Rev 1 → ≈98.86% in Rev 2,
   because F-NEW-1's mandatory memory-multiplier sweep, itself a genuine,
   non-optional inference cost, added ≈0.15 GPU-h raw on top of Rev 1's
   own already-thin margin) — **Rev 3's R3-F6 double-count fix nudges this
   BACK to ≈98.70%** (item F corrected from 108 to 90 passes, §1.6),
   still the single biggest open item, deliberately not papered over,
   and still among the closest to its own ceiling this design has carried
   (§1.6's own honest disclosure: the margin improved but remains thin).
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
   FOUR TIMES this design's own remaining bracket margin at every revision
   so far (≈2.93 GPU-h, Rev 1; ≈1.45 GPU-h, Rev 2; ≈1.658 GPU-h, Rev 3 —
   R3-F6's fix narrowed but did not close this gap), hence categorically
   unaffordable this wave and never silently absorbed into the
   sink-patch's own (zero-added-cost) scope. If the PI's intent for "the
   program's capstone question" implies more than this design scopes,
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
   not eliminate it. **[Rev 4 addendum]** `CE_answer` (§1.3.1.3) reads the
   backbone's own native LM-head logits directly — it does **not** route
   through `shared_probe`/`adapter_arm` at all. This item's adapter-
   capacity asymmetry is therefore ORTHOGONAL to Rev 4's change: neither
   worsened nor mitigated by it. Flagged so a future reader does not
   conflate the two distinct "Transformer gets something extra" concerns
   in this design (this item's adapter width vs. `CE_answer`'s near-free
   logit reuse, item 9's addendum below).
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
   this exact same limitation). **[Rev 4 addendum]** §1.3.1.3's training
   continuation makes a related but DISTINCT asymmetry load-bearing for
   TRAINING, not only for the probe read this item originally described.
   Precision, to avoid conflating the two: the contender/ablation's query
   DOES touch their recurrence during the continuation
   (`forward(query_tokens, initial_states=S_T)` updates state as it
   processes `query_tokens`, exactly the kernel's own standing
   `o_t = read(S_t, q_t)` mechanism every normal LM step already uses) —
   what stays protected, and is what this item and §1.3.1.2 actually
   guard, is only that the query can never reach BEHIND `S_T` into raw
   bind-phase tokens it has not already been causally summarized through
   (verified by the §1.3.1.3 blank-out test, not asserted). The
   Transformer's own attend-to-raw asymmetry this item describes is
   otherwise unchanged by Rev 4 — `CE_answer` for that arm reads out
   logits its existing forward pass already computes (§1.3.1.3), it does
   not change what the Transformer attends to or when. Axis 2's cap
   remains the disclosed counterweight for that pre-existing asymmetry,
   as before. **[Attack round 5 clarification, §1.23 — fairness
   adjudicated, NOT a new asymmetry]** Axis 1 (§1.4.1) is IMMUNE to
   `CE_answer`'s own continuation mechanics regardless: axis 1 compares
   ONLY the contender against the flat-vector ablation, both recurrent,
   both under the IDENTICAL continuation construction described above —
   the Transformer is not a party to axis 1 at all. For axis 2 (§1.4.2),
   `CE_answer` IMPROVES rather than worsens the comparison's
   interpretability: the capped-vs-uncapped Transformer contrast now
   reflects genuinely TRAINED retrieval behavior, closing the pre-Rev-4
   confound where `rf@0.9` plateaued at the untrained episode-membership
   optimum (§1.21) regardless of which side of axis 2 was being read.
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
12. **(NEW, Rev 3 — fold-in from the capability research wave,
    `STATE.md` commit 58de0fa) The contender's β gate is plain sigmoid
    (β∈[0,1]), which forfeits the known state-tracking expressivity
    gain as configured.** [Citation precision, per attack round 4
    §1.19: Grazzi et al. (arXiv:2411.12537) prove finite-precision
    linear RNNs with positive-eigenvalue-only transitions cannot solve
    parity, and that extending eigenvalues to [-1,1] (β∈[0,2],
    specifically `β>1`) enables learning any regular language — the
    "TC0" framing used elsewhere in this repo is a gloss imported from
    the adjacent Merrill-Sabharwal circuit-complexity line
    (arXiv:2207.00729, 2404.08819), NOT Grazzi's own vocabulary;
    re-source the exact class statement before any paper draft.]
    This design's contender uses `arm="per_token"` with a
    plain sigmoid β restricted to `[0,1]` (§1.2), so **Task 2's
    held-out-hop-depth results carry NO formal state-tracking-separation
    implication** — they are empirical-only evidence about what THIS
    specific gate configuration learns to do on THIS task, not evidence
    bearing on the broader state-tracking/TC0 question. The β∈[0,2]
    variant is deliberately RESERVED for the separate capability campaign
    (not built or tested here), to protect this design's own frozen-bias
    evidence provenance — λ=0.58 (§1.2) was tuned under the sigmoid-β
    configuration, and swapping the gate here would silently invalidate
    that tuning without re-validating it, exactly the kind of unpinned
    axis-bundling `CLAUDE.md` warns against. Disclosed for the paper's
    limitations section regardless of Task 2's outcome — a WIN or LOSE on
    Task 2 is a claim about THIS architecture's actual trained behavior,
    not a claim about matrix-state models' formal expressivity ceiling.
13. **(NEW, Rev 4) DEPLOY-PIN-1 (`n_query_train=8`) now feeds BOTH the
    aux loss and `CE_answer`, not just the former — a scope expansion of
    an already-deployed pin, not a new pin.** The deploy-time wiring
    (`h2h_cell_train_rd.py` + 3, §1.20's closure record) fixed
    `n_query_train=8` queries sampled per bind-phase episode, originally
    to feed `mean_over_queries` in the aux probe loss alone. §1.3.1.3's
    `CE_answer` is computed at the SAME per-query granularity over the
    SAME 8 sampled queries — no new data-sampling cost (one shared
    query-batch draw serves both loss terms), but it means the two
    losses' gradients are correlated by construction (computed from the
    identical 8 queries at every step), which is intentional (both terms
    exist to shape the SAME state for the SAME retrieval purpose) and not
    a confound this design is trying to avoid — flagged so a future
    reader does not mistake the correlation for an oversight. No other
    deploy pin (task3 corpus, vocab size, filler policy, task2
    K/hop-split) is touched by this revision.
14. **(NEW, Rev 4) Why a TRAINING-objective change is sanctioned despite
    M-NEW-4's standing inference-only constraint (item 7, above).**
    M-NEW-4 bound Rev 1-3's fixes to inference-only specifically because,
    at the time, a cheaper inference-only alternative existed for the
    F-NEW-1/F-NEW-2 findings (the sink+FIFO patch) against a genuinely
    cap-trained Transformer's +7.6 GPU-h bracket cost — more than 4× the
    remaining margin at every revision so far. Rev 4's situation is
    categorically different on three counts, stated explicitly so this
    is not read as a silent erosion of M-NEW-4's discipline: **(a) the
    wave never launched its 27-cell sweep** — gate 1 (calibration-first)
    caught the objective defect BEFORE any of the expensive downstream
    compute ran, which is exactly what `CLAUDE.md`'s "a calibration
    run... before a big sweep is mandatory, not optional... catches
    convergence ceilings... before you commit a sweep's compute to it"
    rule exists for; this is the rule working as designed, not a failure
    of it. **(b) There is no inference-only fix available** — §1.21's
    root cause (the objective contains no gradient pressure toward the
    answer at all) is irreducibly a training-objective defect; an
    inference-time-only change cannot retroactively teach a model
    something its training loss never rewarded it for learning, unlike
    F-NEW-1/F-NEW-2, where the underlying capability (retrieval under a
    byte budget) was already present and only the EVALUATION protocol
    needed patching. **(c) The cost is small and disclosed** — ≈2.3
    GPU-h re-calibration (§1.6's Rev 4 update), nowhere near the +7.6
    GPU-h bracket cost M-NEW-4 was created to block, and the bracket
    still fits with a real (if reconciliation-flagged) margin either way.
    The fix went through the full required gauntlet — diagnosis with
    executed evidence (§1.21) → this design revision → attack round 5 →
    build → audit — exactly like every other structural decision in this
    document, not a shortcut around it.

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

**Rev 3 adds:** replaces §1.3.1.5's mathematically-broken single-pair
diagonal/off-diagonal diagnostic (independently found degenerate by
numerical execution, §1.17) with a K-simultaneous-bindings diagnostic,
itself numerically executed pre-commit (200 trials/K, `d=64`,
`K∈{1,...,64}`) confirming the expected matvec-exact/Hadamard-collapses-
at-K≥2 behavior (R3-F1); replaces the informal ascending "search for first
success" `M*` procedure with a DESCENDING fixed-sequence gatekeeping test
(Maurer, Hothorn & Lehmacher 1995) that actually controls axis-2's
within-grid FWER at the nominal α, extends the EXISTING §1.8
seed-extension contingency to trigger on a grid-boundary CI straddle, and
splits the `M*=∞` edge case into `CONFIRMED no-crossover` (reportable WIN)
vs `INDETERMINATE` (not a WIN, escalation-eligible like TIE) (R3-F2); pins
`n_layers_transformer=2` (matching the contender's own depth) and remaps
the WIN/TIE/LOSE tiers onto the resulting eligible grid (`M=1`
floor-excluded), disclosing the resulting WIN bar as HARDER, not easier
(R3-F3); adds an M-sweep-specific timing pilot to §1.7 gate 2 with a
pre-registered de-scope order, replacing the ≈5s/pass design-time
assumption with a measured value before the 90-pass fan-out launches
(R3-F4); adds the missing `seed_idx` runtime bound (mirroring the existing
`ckpt_idx` one) plus its negative test (R3-F5); fixes an `M=1` double-count
between item F and the eval-overhead line, SHRINKING the raw total from
≈12.5922 to ≈12.5672 GPU-h and widening the worst-case bracket margin from
≈1.45 to ≈1.658 GPU-h (R3-F6); folds in the cross-campaign β-gate
TC0-escape caveat (self-attack item 12).

**Does NOT:** authorize any GPU launch (Rev 3, pre-attack-round-4);
decide the 392M escalation rung's exact scope or budget (§1.6, §1.9 item
1 — explicitly deferred, still a thin headroom cushion); reopen the
frozen-bias mechanism story, the
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
→ attack round 2 → NEEDS-REVISION (§1.15) → design (Rev 2) → attack round
3 → NEEDS-REVISION (§1.17) → design (this doc, Rev 3) → attack round 4 →
... (iterate until DESIGN-CLEARED-FOR-BUILD, per this program's own
standing gauntlet discipline) → build (contender wiring for the
frozen-bias arm already exists; new code needed: flat-vector ablation
mixer WITH its own `q_proj`/`q_conv1d` (§1.3(a)), standard Transformer
with RoPE + the contender's own `FFN` class + switchable uncapped/
capped-KV inference mode with sink+FIFO eviction (§1.3(b), F-NEW-2), the
§1.3.1 shared probe head + per-arm adapters + frozen `T_val` target table
+ probe-capacity null harness + the K-simultaneous-bindings
tap-expressivity diagnostic (§1.3.1.5, M-NEW-2 + R3-F1 replacement,
including its pre-commit numerical verification), the `rd_episode_seed`
schedule with its 5-key `TASK_BASE`, runtime collision assert, and the
`seed_idx` bound (F1b, M-NEW-1, R3-F5), `verify_match_gate.py` (now
total-across-layers at the pinned fp32 accounting dtype, M2 + Rev-2 dtype
pin, PLUS verification that the R3-F3-pinned `n_layers_transformer=2`
achieves the match tolerance), per-task/per-axis/per-arm calibration
wrappers incl. the K/d sweep (M1) and the Transformer's LR grid (M3), the
axis-2 `cap_length(M)` sweep across the pinned `{1,2,4,8,16,32}` grid, the
`M*` DESCENDING fixed-sequence gatekeeping test and its straddle/§1.8
extension logic (F-NEW-1, R3-F2), and its OWN scoped timing pilot (R3-F4))
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
  deliberately distinct `PROBE_TARGET_SEED=20260709`, and again (Rev 3,
  §1.3.1.5) for the K-simultaneous-bindings diagnostic's own fresh,
  independent `q_i`/`t_i` draws each trial (never sharing a seed with
  `T_val` or the frozen bias table).
- **Fixed-sequence/gatekeeping testing precedent (Rev 3, R3-F2):** Maurer,
  Hothorn & Lehmacher (1995), "Multiple comparisons in drug clinical
  trials and preclinical assays: a-priori ordered hypotheses," *Biometrie
  in der Chemisch-Pharmazeutischen Industrie* 6:3-18 — the originating
  fixed-sequence procedure §1.4.2's descending-`M` `M*` test is built on;
  the FWER argument's independence-free generalization is the sequential
  rejection principle (Goeman & Solari, 2010). Not otherwise reused code
  in this repo, cited for the statistical method only.
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
  contender's own `FFN` class, standard init, PINNED `n_layers=2`
  (R3-F3), and a switchable uncapped/hard-capped-KV inference mode with
  **sink+FIFO eviction** (axis 2's baseline, §1.3(b), M3, F-NEW-2);
  `verify_match_gate.py`, now total-across-layers at the pinned fp32
  accounting dtype, over the **`cap_length(M)` table across
  `M ∈ {1,2,4,8,16,32}`** (§1.7 item 6, M2 + Rev-2 dtype pin + F-NEW-1),
  PLUS verification that the pinned `n_layers=2` clears the match
  tolerance and that `M=1` is correctly excluded from the eligible set
  (R3-F3); the axis-1 power-sketch script that pins `X` (§1.4.1); the
  flat-vector-ablation mixer WITH its own `q_proj`/`q_conv1d` (§1.3(a));
  **§1.3.1's shared probe head** (`shared_probe`, per-arm `adapter_arm`,
  `T_val` target table, the probe-capacity null harness, **and (Rev 3)
  the §1.3.1.5 K-simultaneous-bindings diagnostic**) — the F1 FATAL's
  entire resolution plus its Rev-3 R3-F1 replacement; the
  `rd_episode_seed` schedule module, now 5-keyed with runtime
  collision-guard asserts on BOTH `ckpt_idx` and `seed_idx` (F1b,
  M-NEW-1, R3-F5); the Transformer's 3-point LR-grid calibration harness
  (M3); the axis-2 `M*` DESCENDING fixed-sequence gatekeeping test, its
  straddle-triggered §1.8 extension hook, and the `CONFIRMED`/
  `INDETERMINATE` split reporting logic (F-NEW-1, R3-F2); the M-sweep's
  own scoped timing pilot (R3-F4, §1.7 gate 2).

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

### 1.18 REV 3 CHANGES — finding → resolution map

Every §1.17 finding, mapped to its exact Rev 3 resolution. §1.13, §1.15,
§1.16, and §1.17 are retained verbatim above as the historical record —
this section maps forward from §1.17 only; it does not edit those
sections' own numbers, which reflect what was known/claimed AT THAT TIME
(including the R3-F6 double-count §1.16 itself did not yet know about).

| Finding | Resolution (Rev 3) | Where |
|---|---|---|
| **R3-F1** (FATAL) — §1.3.1.5's single-pair diagonal/off-diagonal diagnostic is mathematically broken: cosine similarity is scale-invariant, so `target_effective` stays near-parallel to `t` for any `alpha>0`, producing a degenerate step function, not a graduated bound; single-pair reachability means a diagonal read can always hit ONE target exactly, so the real gap only bites under MULTIPLE simultaneous bindings | §1.3.1.5 REPLACED with a K-simultaneous-bindings diagnostic: for `K∈{1,2,4,8,16,32,48,64}` bindings, fit the matvec tap (`S=T@pinv(Q)`, least-squares/minimum-norm) and the Hadamard tap (per-coordinate closed-form, `s_j = Σ q_i[j]t_i[j] / Σ q_i[j]²` — separability stated explicitly, per the round's own instruction) to ALL `K` pairs simultaneously, score `recovered_frac@0.9` for both. **Numerically EXECUTED pre-commit** (200 trials/K, `d=64`): matvec = 1.000000 at every `K≤64=d`; Hadamard = 1.000000 only at `K=1`, then 0.000000 for every `K≥2` (mean cosine decaying `~1/√K`: 0.7056→0.4964→0.3506→0.2459→0.1747→0.1433→0.1226) — matches the expected analytic behavior exactly, no spec change needed. | §1.3.1.5 (full rewrite) |
| **R3-F2** (FATAL) — `M*` is underpowered at n=3 and biased toward the strongest-WIN default (an ascending "search until rejection" walk over 5 grid points has a `≈22.6%`, not 5%, chance of a spurious early stop under the global null); no within-axis-2 multiplicity discipline; §1.8's seed-extension contingency never updated for a grid; `M*=∞` conflated "underpowered" with "confirmed no-crossover" | `M*` now determined by a DESCENDING fixed-sequence gatekeeping test (`M=32→...→2`, stop at first non-rejection; Maurer, Hothorn & Lehmacher 1995) that controls axis-2's within-grid FWER at nominal `α`, WITHOUT an independence assumption, CONDITIONAL on a disclosed monotonicity assumption (gap non-increasing in `M`); §1.8's EXISTING seed-extension contingency is extended to trigger on a straddling (not clean) non-rejection at the deciding boundary `(task,M)` cell, bounded at ≤2 extended cells (~6.06 GPU-h) across both tasks; `M*=∞` split into `CONFIRMED no-crossover` (every eligible CI clean, reportable strongest WIN) vs `INDETERMINATE` (unresolved straddle even after extension, NOT a WIN, treated as TIE for escalation). §1.1's table and §1.5's escalation rule both updated to reference `INDETERMINATE` coherently. | §1.4.2 (full rewrite), §1.1, §1.5, §1.8 |
| **R3-F3** (MAJOR) — LOSE structurally unreachable if the Transformer resolves to `n_layers=2` (`M=1` floor-excluded there, but LOSE required `M*≤1`) — an unpinned implementation choice could silently gate the escalation decision | `n_layers_transformer` PINNED to 2 (matches the contender's own depth, verified arithmetic: `cap_length(1)=8<13` excluded, `cap_length(2)=16≥13` smallest eligible); verdict tiers remapped onto the eligible grid: LOSE `M*≤2`, TIE `M*=4`, WIN `M*≥8` — every tier reachable; disclosed explicitly as a THRESHOLD SHIFT (every boundary moved up one grid step) that makes the WIN bar HARDER, not easier. MATCH-GATE (§1.7 gate 6) extended to verify the pin achieves the match tolerance and that `M=1` is correctly excluded from `M*`-eligibility. | §1.4.2, §1.7 gate 6 |
| **R3-F4** (MAJOR) — the ≈5s/pass M-sweep assumption has no scoped timing-pilot gate and no de-scope rule against a 1.45 GPU-h margin (a reload-per-pass implementation could run 3-6×) | §1.7 gate 2 extended with an M-sweep-specific timing pilot (2 `M`-values × 1 checkpoint × 1 horizon before the 90-pass fan-out); "checkpoints resident across all passes" pinned as a build requirement; pre-registered de-scope order (drop `M=32` first, then `H8`) if the measured bracket would exceed the 127.33 GPU-h headroom. | §1.7 gate 2, §1.6 (item F cross-reference) |
| **R3-F5** (minor) — `seed_idx` has no runtime bound (`seed_idx=50` on `task1_calib` collides with `task1_stress` seed 0); `ckpt_idx` has an assert, `seed_idx` doesn't | `rd_episode_seed` gets `assert 0 <= seed_idx < 50`, mirroring the existing `ckpt_idx` assert; the collision smoke gets a dedicated negative test (`seed_idx=50` must raise). | §1.3.1 (F1b subsection) |
| **R3-F6** (minor) — item F's 108-pass count double-counts the `M=1` pass already priced in the eval-overhead line | Item F's count corrected to 5 `M`-values (`{2,4,8,16,32}`, excluding `M=1`) × 3 horizons × 2 tasks × 3 seeds = 90 passes, ≈0.125 GPU-h (was 108 passes/≈0.15 GPU-h). Raw total: ≈12.5922 → **≈12.5672 GPU-h**. 10× bracket: ≈125.88 → **≈125.672 GPU-h**. Margin vs. the unchanged 127.33 GPU-h headroom: ≈1.45 GPU-h/1.14% → **≈1.658 GPU-h/1.30%** — the fix SHRANK cost and widened the margin, as required. | §1.6 (full re-derivation) |
| **FOLD-IN** — cross-campaign: contender's β gate is plain sigmoid, no TC0-escape property, per Grazzi et al. (arXiv:2411.12537) | Added as self-attack item 12: Task-2 held-out-depth results are empirical-only, no formal state-tracking-separation implication; β∈[0,2] reserved for the separate capability campaign to protect frozen-bias provenance (λ=0.58 tuned under sigmoid β). | §1.9 item 12 |

**What this revision could NOT close with full margin (flagged, not
papered over — the task's own instruction):**

1. **The ≈5s/pass M-sweep assumption is STILL a design-time assumption,
   not a measured one.** R3-F4 added the GATE (a real timing pilot before
   fan-out) and a pre-registered de-scope order, but — per this design's
   own "zero GPU spent" status — no real hardware measurement exists yet
   to replace the ≈5s/pass figure itself. This remains the single most
   likely place a future re-derivation moves the bracket's fit, exactly as
   flagged in §1.6's own closing paragraph.
2. **The fixed-sequence procedure's monotonicity assumption (gap
   non-increasing in `M`) is disclosed but unverified.** If the real,
   noisy `n=3` gap is non-monotonic across the grid, the pre-registered
   descending-walk stopping rule still applies mechanically (never peek
   past the first non-rejection) — this preserves FWER validity at the
   cost of potentially reporting a coarser `M*` than a hypothetical
   noise-free re-ordering would find. This trade-off is a genuine,
   disclosed cost of buying valid statistics, not a bug, but it is a real
   one: an attack-round-4 agent should confirm the trade-off is
   acceptable, not merely correctly described.
3. **R3-F2's straddle-compounding cost bound (≤2 cells, ≈6.06 GPU-h) is
   itself an upper bound reasoned from the fixed-sequence structure, not
   independently re-derived by a fresh pass this round.** It follows
   directly from "testing stops at the first non-rejection, so at most one
   cell per task is ever the deciding boundary" — an attack-round-4 agent
   should re-verify this claim holds for the ACTUAL build (i.e. that no
   implementation shortcut re-tests a cell after the walk has already
   stopped there).
4. **The margin (≈1.658 GPU-h, ≈1.30% of headroom) improved from Rev 2's
   ≈1.45 GPU-h/1.14% but remains thin** — R3-F6's fix is a genuine, real
   improvement (not a re-scope), but the escalation-rung budget problem
   (§1.9 item 1) is unchanged and unresolved: the escalation rung, as
   currently scoped, still does not fit the existing ceiling at rung-1's
   own step budget.
5. **This design does not re-verify §1.17's own "verified clean this
   round" list** (cap_length(M) table, fp32 pin, sink+FIFO buildability,
   TASK_BASE margins, ckpt_idx assert, CI table, §1.7 gate 6 coherence) —
   those are trusted as still valid since nothing in Rev 3 touched their
   underlying arithmetic, but an attack-round-4 agent should confirm this
   assumption explicitly rather than accept it silently.

---

### 1.19 ATTACK ROUND 4 VERDICT (independent fresh-eyes agent, 2026-07-08): **DESIGN-CLEARED-FOR-BUILD**

Recorded per the gauntlet-bookkeeping hard rule; the build stage
dispatches against THIS recorded verdict. Fourth independent round; the
attacker RE-IMPLEMENTED the load-bearing numerics from the prose spec
(own code, own seed 13371, lstsq + independently-derived closed form):

- **§1.3.1.5 diagnostic independently CONFIRMED** — full table match
  within Monte Carlo noise at every K (incl. the K=64 near-singularity
  edge); R3-F1's fix is real.
- **Descending M\* gatekeeping** — all four edge cases hand-walked
  (grid-exhaustion → LOSE; stop-at-4 → WIN M*=8; immediate non-reject →
  M*=∞ pathway w/ full-grid requirement verified unambiguous at lines
  1242-1249; boundary straddle → §1.8). Tier mapping coherent;
  termination well-defined. BONUS CORRECTION: FWER control does NOT
  require the monotonicity assumption (MHL first-true-null holds for any
  true configuration) — the doc's own "CONDITIONAL" sentence was wrong
  in the safe direction; patched this revision per the round's exact
  prescription.
- **Tier reachability** — cap_length(M)=M×8 at n_layers=2/fp32
  re-derived; floor 13 re-verified against grammar_rd.py:552's LIVE
  smoke assert (not the doc's claim); all tiers reachable.
- **Budget** — 12.5672/125.672/1.658 re-summed exactly; every Wave −1
  sub-item re-derived; Grazzi citation fact-checked against the paper.

**Four MINOR documentation-precision findings, ALL PATCHED this
revision per the round's prescriptions (no design change, no threshold
change, no budget change):** (1) the monotonicity-sentence
self-contradiction → corrected (§1.4.2); (2) bracket-vs-straddle
accounting regimes never reconciled → mutual-exclusivity note added
(§1.6); (3) §1.8's seed-extension formula stated "6 cells × 0.2524 ≈
3.03" — the multiplier omitted the ×2-arms factor (12 cells; bottom
line was always right) → corrected; (4) "TC0" attributed to Grazzi,
who never uses the term (parity/regular-language is their vocabulary;
TC0 is the Merrill-Sabharwal gloss) → re-attributed in self-attack
item 12, with a re-source-before-paper-draft note.

**Interpretation-rule disclosure (round 4, non-blocking):** at the real
load points (K=32/48) the diagnostic's own tap gap is already maximal,
so the §1.3.1.5 curve can EXPLAIN AWAY a near-null axis-1 result but
can never POSITIVELY confirm a WIN reflects state capacity over tap
mechanism — the K=32 disclosure at lines 750-760 generalizes; carried
into the analysis stage as a standing interpretive bound.

**Build-stage dependency order (round 4's list, binding on the build
agent):** (1) CPU-parallel: verify_match_gate.py (2-pass),
rd_episode_seed 5-key + seed_idx/ckpt_idx asserts + collision smoke,
flat-vector mixer, Transformer (RoPE/shared FFN/n_layers=2/sink+FIFO
capped mode), §1.3.1 probe head + adapters + frozen T_val +
probe-capacity-null harness, K-bindings diagnostic script; (2) gates
6+7 PASS before PI-signoff tokens; (3) gate 1 three-arm calibration;
(4) gate 2 timing pilots incl. the R3-F4 M-sweep pilot
(checkpoints-resident); (5) 27-cell training sweep; (6) 90-pass
M-sweep fan-out; (7) analysis (axis-1 margins; descending M* walk w/
§1.8 as needed, PI-gated); (8) escalation decision (§1.5, mechanical,
NOT pre-authorized).

---

### 1.20 INDEPENDENT BUILD AUDIT VERDICT (2026-07-08): NEEDS-FIXES (non-blocking for CPU stage; 1 substantive)

Recorded per the gauntlet-bookkeeping hard rule before dispatching the
fix stage. Build = commits 8d55f17..9480ced (12 files). The audit
re-ran every suite fresh, confirmed EVERY builder number exactly
(param/FLOP/bytes/cap-table/floor/eligible-M; gate-7 nulls 0.0000 all
arms; K-bindings anchors; all four M* synthetic cases; delta_ci_n
verified paired-per-seed), and ran the 6-mutation protocol on
scratchpad copies (repo untouched, verified):

| Mutation | Result |
|---|---|
| (a) match-gate Pass-2 constant 4→2 | CAUGHT (passes disagree, exit 1) |
| (b) ablation width drift >1% | CAUGHT (both checks fail) |
| (c) TASK_BASE spacing collision | **PARTIAL** — caught for seed_idx≤11; smoke blind for seed_idx∈[12,49] (smoke_5 compares a hardcoded 500_000, not the live dict; currently INERT — no call site exceeds seed_idx=11) |
| (d) sink-retention drop | CAUGHT TWICE (independent checks — real defense-in-depth) |
| (e) M* walk inverted to ascending | CAUGHT DECISIVELY (4/7 smoke items incl. the LOSE→WIN flip) |
| (f) corrupted sweep JSON | CAUGHT end-to-end (resume re-executes only corrupted cells) |

**FINDINGS (binding on the fix stage):**

- **AUD-F1 (substantive) — smoke_7's joint-training gradient claim is
  VACUOUS for the contender arm.** The CPU stub's `final_state =
  zeros(requires_grad=False)` kills the aux-loss gradient path through
  the tap (`S @ q_last`), so the aux loss contributes ZERO gradient to
  every contender backbone param (verified by aux-only isolation:
  q_proj/k_proj/v_proj/b_proj/embed all grad=0.0); smoke_7's PASS was
  confounded by loss_ce touching q_proj through the stub's own gate.
  §1.3.1.3's core premise (probe loss backprops into the BACKBONE) has
  NO valid test for the contender, CPU or box. Ablation + Transformer
  arms verified genuinely nonzero. → FIX: aux-loss-only (CE-excluded)
  gradient-isolation test checking k_proj/v_proj/b_proj (q_proj stays
  confounded even on real hardware), honest CPU-side split (ablation/
  transformer provable now; contender registered as a BOX-ONLY deploy
  gate mirroring smoke_3's discipline).
- **AUD-F2** — TASK_BASE collision smoke: derive the offset check from
  the LIVE dict (not the hardcoded 500_000) and extend the exhaustive
  range to seed_idx<50. Currently inert; fix before any seed-extension
  beyond n=12 is ever authorized.
- **AUD-F3** — `gate_extra_width>0` path is 100% untested; add one CPU
  smoke at a nonzero value (param count exact, forward OK, match-gate
  fails when it drifts >1%).
- **AUD-F4 (cosmetic)** — task3 calibration manifest reuses
  task2_calib's TASK_BASE key without the explanatory comment its
  sibling has.

**Builder's 7 flagged limitations:** all adjudicated honest/complete;
none launch-blocking; the disclosed box-only items are correctly
registered.

**REGISTERED BOX-SMOKE CHECKLIST (deploy stage, binding):** (1) real
fla/Triton forward+backward+grad smoke for DeltaNetLM; (2) tap-changes-
with-q on a real nonzero S_T_last; (3) [NEW, AUD-F1] aux-only gradient
isolation for the contender on real kernels (k/v/b_proj nonzero); (4)
state-bytes/dtype round-trip = 32,768 fp32 on real kernel; (5) R3-F4
M-sweep timing pilot before the 90-pass fan-out; (6) gate-2 per-
arch×task timing pilots; (7) gate-1 full 14-cell 3-arm calibration
run to completion w/ bands checked; (8) deferred: d=128 diagnostic
only if escalation authorized.

---

**§1.20 ADDENDUM — SCOPED RE-AUDIT (2026-07-08): FIXES-VERIFIED-CLEARED.**
Fix commit ed6996c (5 files, 572+/16−, zero scope creep verified via
diff-stat). Independent re-verification: AUD-F1's smoke_8 read +
re-run (genuinely CE-excluded; ablation/transformer aux grads nonzero;
detach negatives collapse to None; contender box-only branch traced —
fires with zero new wiring when real fla imports); AUD-F2's mutation
RE-RUN by the re-auditor (400k spacing shrink caught by BOTH smoke_1b
— the exact (task1_stress,0,0)-vs-(task1_calib,40,0) collision — and
smoke_5's live-dict spacing proof; the smoke_1/1b split adjudicated
NO-COVERAGE-LOSS: exhaustive-at-low-seed + sampled-at-full-range +
algebraic spacing proof is a superset of the mandate); AUD-F3's
drifted-count catch confirmed against the UNMODIFIED gate function;
AUD-F4 comment verified; checklist manifest 1:1 with §1.20's 8 items,
consumable. Full regression: 7 suites exit 0 with real PASS sentinels.
**BUILD STAGE CLOSED. DEPLOY STAGE AUTHORIZED** (closure checksum →
box smoke per h2h_box_smoke_checklist → gates 6+7 on box → pilots →
gate-1 calibration → margins freeze recorded → sweep release).

---

*(End §1. Rev 0 → ... → §1.19 DESIGN-CLEARED → BUILD (9480ced) →
§1.20 audit NEEDS-FIXES → fixes (ed6996c) → **re-audit
FIXES-VERIFIED-CLEARED. DEPLOY STAGE ACTIVE.**)*

---

### 1.21 CALIBRATION RECORD + PROBE DIAGNOSIS (2026-07-09): ROOT CAUSE = NO RECALL PRESSURE IN THE OBJECTIVE — Rev 4 revision round opened

Recorded per the gauntlet-bookkeeping hard rule. Chronology: gate-1
bands FAILED (rf@0.9=0 on all 9 task-1/2 cells, all arms) → the
pre-registered §1.3.1.3 aux_weight dial fired (ratio 20.9×) and was
executed (0.1→2.0 parity pin, commit f3b8343; _auxrev2 re-run) →
parity ACHIEVED (ratios 1.3-3.6) but all arms plateaued at
probe_cos_mean 0.12-0.22 → HARD-STOP → box diagnosis on the saved
checkpoints (0.08 GPU-h; artifacts at results/h2h_rung1/
probe_diagnosis/ on box).

**DIAGNOSIS (conclusive, chain-of-elimination executed):** task-routing
failure. The models NEVER LEARNED RECALL: LM-head answer accuracy at or
below chance in ALL arms (incl. the transformer); a trained
identity-classifier on the tap reads 1.2-3.2% vs 0.93% chance (the
answer is NOT in the representation); offline probe refits to
convergence equal the online plateaus (no under-fit); MLP probes are
WORSE held-out (no nonlinear treasure). The "CE learns fine" premise
was a MISREAD — CE≈1.1 is grammar/format statistics; structurally CE
contains no retrieval signal (each key appears exactly once at bind
time; query windows never enter CE). The only recall pressure was the
aux loss, which converged to the EPISODE-MEMBERSHIP local optimum:
predicted vectors align with the episode-mean of T_val rows at cos
0.94/0.93 (contender/transformer), whose analytic ceiling 1/√K =
0.1768 at K=32 matches every plateau. The M-NEW-2 Hadamard asymmetry
is real and visible one level down (ablation can only reach 0.65
membership alignment). Codebook exonerated (tied-embedding jump is
entirely the non-orthogonal-embedding floor: 0.419/0.730/0.424). No
bar separates anything (best-probe p99=0.45; codebook-correct ≈
chance).

**Options adjudicated w/ executed evidence:** (a) longer training
REFUTED (offline convergence = online plateau); (b) tied-embedding
targets REFUTED (floor artifact + arch-non-neutral, floors
0.42/0.73/0.42); (c) MLP probe REFUTED (worse held-out; info absent);
(d) bar re-pin REFUTED (no separation to threshold); (e) K de-load
COUNTERPRODUCTIVE (membership ceiling rises as 1/√K — amplifies the
confound). **(f) ADOPTED FOR REV 4: add answer-token CE at the query
position to the training objective, ALL THREE ARMS SYMMETRIC** —
recurrent arms continue from cached S_T (forward(query, initial_states
=S_T): a function of (S_T, query) only — P=1 preserved by causality,
blank-out-verifiable); transformer already materializes these logits.
Makes recall NECESSARY; §1.3.1's instrument (frozen T_val, linear
probe, rf@0.9 decision metric) stays UNCHANGED; M-NEW-2's disclosed
asymmetry becomes the pre-registered PREDICTION (matvec can reach
rf@0.9 per §1.3.1.5's own table, Hadamard stays bounded). New pinned
ce_answer_weight calibrated by the existing step-500 dial. LM-head
accuracy = disclosed GATE, never the WIN metric (Nichani rule).
PRE-REGISTERED DIAGNOSTIC LADDER rides along: LM-head accuracy (task
learned?) → identity-classifier (info in tap?) → rf@0.9 (instrument) —
any future plateau immediately attributable. Cost: Rev 4 + attack
round + ~2.3 GPU-h re-calibration (fits margin). Disclosures touched:
§1.3.1.3 loss formula, §1.9 items 8/9, DEPLOY-PIN-1, M-NEW-4 table.

---

### 1.22 REV 4 CHANGES — diagnosis item → resolution map

Unlike §1.14/§1.16/§1.18, this map's trigger is not an attack-round
verdict but §1.21's own executed-evidence diagnosis (option (f) adopted
after five alternatives were REFUTED or found COUNTERPRODUCTIVE). Each
row maps a §1.21 item to its exact Rev 4 resolution and section.

| §1.21 item | Rev 4 resolution | Section |
|---|---|---|
| Root cause: `loss_CE_lm + aux_weight·probe_cosine_loss` has no gradient pressure toward the queried answer (CE structurally retrieval-blind; aux converges to the cheaper episode-membership local optimum, `1/√K` ceiling) | Joint THREE-term objective: `loss_CE_lm + ce_answer_weight·CE_answer + aux_weight·probe_cosine_loss`, `CE_answer` scored at the query answer position off the backbone's own native LM-head logits, symmetric across all three arms | §1.3.1.3 (rewrite) |
| Recurrent arms must gain answer-position logits WITHOUT breaking the P=1 hard bottleneck (`CLAUDE.md`) | Query runs as a continuation, `forward(query_tokens, initial_states=S_T)` — a function of `(S_T, query_tokens)` only by the call's own signature ("P=1 preserved by causality"); blank-out test spec'd (corrupt `bind_tokens` post-`S_T`-caching, decode must be unchanged) and wired as a Wave −1 (A) negative-test row | §1.3.1.3, §1.7 gate 3 |
| Transformer's `CE_answer` cost | Near-zero — its existing single forward pass already materializes query-position logits; no second forward | §1.3.1.3 |
| `ce_answer_weight` needs a pin and a calibration path | Pinned default 1.0; step-500 gradient-ratio dial extended to three losses; "one revision per calibration round max" rule pinned exactly (larger-ratio deviation revised first if both auxiliaries fire; `aux_weight` carries forward round 2's 2.0, re-checked not re-derived) | §1.3.1.3 |
| §1.3.1's instrument (frozen `T_val`, linear probe, `rf@0.9`) — does it change? | No. Explicitly UNCHANGED; only the training objective and its diagnostics change | §1.3.1.1-1.3.1.2 (untouched) |
| M-NEW-2's Hadamard-vs-matvec asymmetry becomes a testable prediction under real recall pressure | Restated as the pre-registered prediction: matvec can reach `rf@0.9` per §1.3.1.5's own K-bindings table, Hadamard stays bounded — no new construction needed, §1.3.1.5 stands as-is | §1.3.1.5 (untouched, re-purposed) |
| "LM-head accuracy = disclosed GATE, never the WIN metric (Nichani rule)" — needs a formal home | THE DIAGNOSTIC LADDER: rung 1 (episode-restricted LM-head/native top-1, `>3×` chance, gate-only) → rung 2 (linear identity-classifier on the tap, `>3×` chance, diagnostic-only) → rung 3 (`rf@0.9`, unchanged decision metric); attribution table + membership-oracle tell (`probe_cos_mean≈1/√K` AND `cos(pred,episode_mean)≥0.85`) logged per cell | §1.7 gate 1, item 1a |
| Tasks 1/2 had no explicit numeric gate-1 band (STATE.md: "SANITY-ONLY") | FULL-cell band pinned: `rung-1 answer accuracy >3× chance AND rf@0.9>0`; stress/locate-only cells stay exempt | §1.7 gate 1, item 1b |
| Cost: "~2.3 GPU-h re-calibration (fits margin)" | Derived from measured per-arch full-cell wall-clock (contender/ablation/transformer); realized-to-date pulled from `STATE.md` (11.43/135); updated margin ≈4.17 GPU-h ledger-anchored / ≈2.95 GPU-h bottom-up-anchored, both fit; reconciliation gap between the two readings flagged, not resolved; closes Rev 3's own R3-F4-flagged M-sweep timing-pilot open item using `STATE.md`'s already-measured 1.1-1.5s/pass figure | §1.6 (Rev 4 cost update) |
| Disclosures touched: §1.3.1.3, §1.9 items 8/9, DEPLOY-PIN-1, M-NEW-4 table | Item 8 addendum (adapter asymmetry orthogonal to `CE_answer`, which bypasses the probe entirely); item 9 addendum (query-touches-recurrence during the continuation is expected and distinct from the probe-tap's own query-isolation, which still holds); new item 13 (DEPLOY-PIN-1 scope expansion, intentional correlation disclosed); new item 14 (why a training-objective change is sanctioned despite M-NEW-4 — calibration-first caught it pre-sweep, no inference-only fix exists, cost is small and disclosed) | §1.9 items 8, 9, 13, 14 |

**What Rev 4 explicitly does NOT touch (kept verbatim, per this
revision's own mandate):** §1.13-§1.21's gauntlet records stand as
history, unedited; §1.3.1.1, §1.3.1.2, §1.3.1.4, §1.3.1.5, F1b's
`rd_episode_seed` schedule; §1.4's task/axis/margin definitions and the
M* gatekeeping machinery (§1.4.2); §1.5's scale ladder; §1.8's
pre-registered analysis; §1.10-1.12; the 27-cell sweep's own raw/bracket
figures (unchanged except the R3-F4 timing-pilot closure folded into the
Rev 4 cost update, §1.6).

**Unresolved, flagged for attack round 5 (not decided here, per this
program's own "flag, don't paper over" convention):** (1) the ≈1.2 GPU-h
ledger-reconciliation gap between `STATE.md`'s live figure and a
bottom-up per-arch-rate reconstruction of rounds 1-2 (§1.6); (2) whether
the "one revision per calibration round max" rule's tie-break (larger-
ratio deviation first) is the right priority if round 3's dial finds
BOTH `aux_weight` and `ce_answer_weight` out of parity simultaneously —
untested until round 3 actually runs; (3) the identity-classifier's
(rung 2) own capacity sanity — unlike `shared_probe` (§1.3.1.4's null),
no equivalent frozen-random-tap null is specified for the NEW
classifier, so a rung-2 PASS on real data is not yet distinguishable
from the classifier itself being oddly well-suited to noise, a gap
attack round 5 should close before round-3 results are trusted at face
value.

---

*(End §1. ... → §1.20 build CLEARED → deploy → calibration rounds 1-2
→ §1.21 DIAGNOSIS: objective lacked recall pressure; option (f)
adopted → **Rev 4 (this doc): three-term objective + diagnostic ladder
+ bands + cost update, §1.22 changes map** → attack round 5 (pending)
→ build fix → calibration round 3 (~2.3 GPU-h) → margins freeze →
27-cell sweep + 90-pass M-sweep → harvest.)*

---

### 1.23 ATTACK ROUND 5 VERDICT (independent fresh-eyes agent, 2026-07-09): **CLEARED-FOR-BUILD-FIX**

Recorded per the gauntlet-bookkeeping hard rule; the scoped build-fix
dispatches against THIS verdict. The round verified prose against CODE
and against fla-org's real ShortConvolution source (fetched live):

- **P=1-by-causality SOUND** — every named leak channel walked and
  closed: no positional/RoPE machinery exists in the recurrent arms
  (grep-verified); ShortConvolution is stateless with cache=None
  (zero-padded starts, verified against the real source); the kernel's
  only carried state is the explicit initial_state/final_states = S_T.
- **FAIRNESS ADJUDICATED (the make-or-break): NOT a new asymmetry.**
  Axis 1 compares ONLY contender vs ablation — both recurrent, both
  under the identical continuation construction; the transformer isn't
  in axis 1. For axis 2, CE_answer IMPROVES interpretability (the
  capped-vs-uncapped comparison now reflects trained retrieval, not the
  pre-Rev-4 untrained confound). §1.9 item 9 was already the honest
  disclosure; one clarifying sentence added per the round.
- **LEDGER GAP RECONCILED** (root cause found): round 1 was ledgered at
  the flat planning rate (0.2524/cell), the bottom-up uses real
  arch-differentiated wall-clock — mundane bookkeeping provenance, both
  margins stay positive; downgraded to explained. The 4.98 figure
  includes the 0.08 diagnosis (now stated).

**FINDINGS (folded into the scoped build-fix, binding):**
- **R5-F1 (substantive) — three-loss dial had no termination guarantee:**
  PINNED per the round: at most ONE contingency dial round (round 4,
  ≈2.3 GPU-h, absorbed by both margin readings: 4.17→1.87 /
  2.95→0.65); if round 4's dial still fails on either weight →
  HARD-STOP into a fresh §1.21-style diagnosis, never a further dial.
- **R5-F2 (substantive) — rung-2 identity-classifier gets its OWN
  frozen-random-tap capacity null** (fresh i.i.d. Gaussian per real
  adapter shape, fresh-per-step, held-out eval, labels independent;
  pass bar ≤1.5× chance — strictly separated from the real 3× gate);
  folds into the Wave −1(B) CPU harness.
- **R5-F3 (textual)** — gate 1a's "EVERY calibration cell" overclaim
  rescoped to task1_calib/task1_stress/task2_calib (the ladder is
  grammar_rd-specific; Task 3 keeps its own anchored band).
- **R5-F4 (non-blocking)** — blank-out companion: fresh-model-instance
  continuation (only S_T passed) must produce bit-identical logits —
  closes the hidden-module-cache channel the in-place test can't;
  registered in the box-smoke checklist.
- Build precision note: rung-1 = K-restricted gather+argmax (NOT
  global-vocab top-1 vs a 1/K bar).

**BUILD-FIX ITEM LIST (6 items, §1.23's binding order):** CE_answer
continuation path per-arm; 3-loss dial + R5-F1 cap; ladder rungs 1-2
(+R5-F2 null) + attribution logging; bands wired
(task1_calib/task2_calib only); blank-out in-place + R5-F4 companion
(box-registered); doc rewordings (R5-F3, 4.98 note, axis-1-immunity
sentence).

---

*(End §1. ... → §1.21 diagnosis → Rev 4 → **§1.23
CLEARED-FOR-BUILD-FIX** (2 substantive findings folded into the fix).
SCOPED BUILD-FIX ACTIVE → scoped audit → calibration round 3 → ladder
+ bands review → margin freeze → sweep.)*

---

### 1.24 SCOPED BUILD-FIX AUDIT (2026-07-09): RESIDUAL-FINDINGS — fix logic sound; cost estimate 4× low via an avoidable LM-head waste; one untested precision note

Recorded per the gauntlet-bookkeeping hard rule. Import-chain repair
(1db9594) confirmed; all baseline suites pass; padding causal-inertness
INDEPENDENTLY VERIFIED (bit-identical answer logits at min_t 128/256/
512); mutations (b) dial-cap and (c) band-weakening CAUGHT.

**FINDINGS (binding on the pre-launch fix):**
- **AUD2-F1 (MAJOR, economics): the CE_answer continuation applies the
  50,259-way LM head to ALL 128 padded positions then discards 127/128**
  (lm_pretrain_rd.py:1298 path) — measured added cost 6.57×/6.81×
  per-step on contender/ablation → corrected round-3 total ≈9.18 GPU-h
  vs the §1.6 2.300 figure (≈4.0×, past the 1.5× flag bar); the
  27-cell sweep's 11.675 GPU-h projection is ALSO pre-CE_answer stale.
  → FIX: slice the hidden state to the answer position BEFORE the
  LM-head matmul (architecturally clean, cuts the waste ~99%); then a
  FRESH on-box timing pilot under the real three-term objective
  re-prices round 3 AND the sweep before any launch; §1.6 cost-note.
- **AUD2-F2 (substantive): mutation (a) NOT CAUGHT — no test defends
  the K-restricted gather+argmax** (§1.23's own load-bearing precision
  note). → FIX: a planted-answer synthetic test where global-vocab and
  K-restricted argmax provably differ (a non-candidate token gets the
  max logit; K-restricted must ignore it), run to completion.
- **AUD2-F3 (defense-in-depth): mutation (d) not caught** (selftest-13
  weakening invisible) — current code verified correct; add the cheap
  guard if convenient, else record as accepted residual.
- **AUD2-F4 (deploy mechanics, binding on the chain patch):** the
  H2H_DIAL_ROUND export line (exact text in the audit transcript) +
  TWO caveats: (i) unset/reset the var before Stage D (else a noisy
  sweep-cell step-500 ratio could spuriously DialExhausted-abort);
  (ii) a round-4 re-run must invalidate/move round-3's calibration
  outputs or resume-safety will SKIP the cells entirely.

**Cross-stream:** both commits leave the 2×2 runner + capability_
separation untouched; CPU-side 2×2 flag grid passes 4/4 post-1db9594.

---

*(End §1. ... → §1.23 CLEARED-FOR-BUILD-FIX → build-fix cc89a4f →
**§1.24 audit: RESIDUAL-FINDINGS (LM-head slice fix + rung-1 test
required pre-launch)**. PRE-LAUNCH FIX ACTIVE → fresh timing pilot →
calibration round 3. NOTE: margin freeze ALSO blocked on the
fix-at-scale attack's per_token-vs-global adjudication (§13/660cffc) —
the contender pin verifies before any freeze.)*

### 1.25 BUILD-FIX VERIFICATION (2026-07-09 overnight): AUD2-F1..F4 ALL LANDED — commit 68e2768; ~1.2-1.5× cost-target closure REGISTERED to the box-side timing pilot

Independent fix agent implemented and tested all four §1.24 findings;
coordinator verified the commit (4 files, specific-path staging, clean-audit
sentinel) and records here per gauntlet bookkeeping.

- **AUD2-F1 (LM-head-over-padding, the 4× waste):**
  `_recurrent_continuation_answer_logits` now forwards with
  `return_hidden=True` (flag added to `DeltaNetLM.forward` AND
  `AblationLM.forward`, mirrored) and slices to the answer position BEFORE
  the vocab matmul. New selftest 16 re-derives the pre-fix computation and
  asserts bit-identical answer-position logits — PASS both arches.
  **Op-level result: LM-head matmul 60×/126× faster (1204→20ms contender,
  1183→9ms ablation; ≈99% FLOP cut, exactly as §1.24 designed).**
  End-to-end CPU-stub ratio landed at 2.49×/3.55× (vs old 5.47×/7.57×
  reproduced) — NOT the ~1.2-1.5× target, but the residual is 97.7-98.6%
  CPU-stub padded-recurrence overhead (the stub is python tensor ops, not
  the fused Triton kernel), i.e. an instrument confound, not fix failure.
  **The target's pass/fail is therefore REGISTERED to the box-side fresh
  timing pilot (real kernel), which was already the §1.24 mandate.**
  [LEARN] recorded: don't let stub-dominated end-to-end wall-clock stand in
  for a GPU-kernel-cost claim — decompose and register the box number.
- **AUD2-F2 (rung-1 gather untested):** logic extracted to
  `_rung1_k_restricted_pred_slot`; new selftest 17 (planted-answer
  synthetic: non-candidate holds global max, candidate holds max-among-K).
  §1.24 mutation (a) run as a REAL negative control against the production
  function: exactly 1 failure (selftest 17), isolated, exit 1 — the test
  has teeth.
- **AUD2-F3:** selftest 13 hardened with `id()`/`data_ptr()` distinctness
  asserts (aliased-instance weakening can't pass).
- **AUD2-F4 (chain dial-round mechanics):** `export H2H_DIAL_ROUND=3`
  before Stage B + documented round-4 invalidation procedure; Stage-D guard
  fixed STRUCTURALLY — step-500 dial now also requires
  `cell.get("role") != "sweep"` (sweep cells carry role=="sweep" by
  construction; can't leak the way an unset env var could).

Full CPU selftest suite 17/17 PASS under the stub. Security: one more
fake-system-reminder injection sighted in `git pull` stdout (date-change +
concealment), disregarded/reported — tallied in `STATE.md`.

**NEXT (unchanged §1.24 chain): deploy patch to box (md5) → FRESH timing
pilot (re-price calibration round 3 + sweep; closes the 1.2-1.5× target on
the real kernel) → calibration round 3 (9 cells) → ladder+bands → margin
freeze → token → sweep. GPU note: fix-at-scale gate tier holds GPUs 1-4;
h2h pilot/calibration go to GPUs 5-7 (and 0 after the 2×2 finishes).**

### 1.26 DEPLOY + FRESH TIMING PILOT (2026-07-09, GPU 5, real fla/Triton kernel): AUD2-F1 target CLOSED on contender/ablation; round-3 decision gate FAILED — round 3 NOT launched

Recorded per the gauntlet-bookkeeping hard rule. Full record:
`experiment-runs/2026-07-09_h2h_timing2_launch/MANIFEST.md`.

**Deploy:** the four §1.24 fix files (`lm_pretrain_rd.py`, `h2h_cell_train_rd.py`,
`ablation_mixer_rd.py`, `h2h_rung1_chain.sh`, commit `68e2768`) deployed to box, md5-verified
local=box on all five (incl. the untouched `h2h_box_smoke_checklist.py`).

**Re-smoke (GPU 5, fresh, real kernel — the prior deploy's tokens do not cover this build):**
items 1-4 + gates 6/7 (`h2h_box_smoke_driver.py`, unmodified) ALL PASS, fresh tokens under
`results/h2h_rung1/gates_v2_20260709/`. Items 9-10 (sec 1.23's blank-out box-only residual, not
yet wired into the driver) verified via a throwaway harness replicating `mode_selftest`'s own
selftest-12/13 construction under a forced CUDA default device (zero logic changes to any
audited file) — PASS both items, both arches, real nonzero S_T confirmed
(contender |S_T|=620.6/583.7, ablation |S_T|=6.3/3.9).

**Fresh timing pilot (GPU 5, real fused kernel, task1_calib K=32 production dims):** measured
FIXED-vs-pre-Rev4-baseline per-step ratio: **contender 1.244×, ablation 1.487×** — both cleanly
inside the ~1.2-1.5× AUD2-F1 target, both ≤1.6× — the fix is CONFIRMED effective on real
hardware (the CPU stub's own 2.49×/3.55× was an instrument confound, as diagnosed at §1.25).
VRAM cross-check: pre-fix variant peaks at ~2.4× the fixed variant's VRAM (16.4GB vs 6.9GB),
consistent with the eliminated (B·Q,128,vocab) intermediate. **New finding, not an AUD2-F1
regression:** the transformer's own ratio measured at 1.736× — its fused "no second forward"
path reprocesses the full context once per query and was never priced under the real Rev4
three-term objective before now.

**Re-priced round 3 (9 cells, measured rates): 3.593 GPU-h vs the §1.6 registry's 2.300 GPU-h
prior — ratio 1.562×, past the pre-registered 1.5× gate.** Per the pre-registered decision rule
(measured ratio ≤1.6× AND round-3 re-price ≤1.5× registry → chain; else stop and report),
**condition 2 fails → calibration round 3 was NOT launched.** Pilot-only spend (≈0.02 GPU-h).
Coordinator decision pending: re-scope round 3's transformer budget, widen the pre-registered
margin, or accept the 1.562× spend given the fix's own confirmed contender/ablation success.

---

*(End §1. ... → §1.25 build-fix VERIFIED (68e2768) → **§1.26 deploy + fresh real-kernel timing
pilot: AUD2-F1 target CLOSED (contender 1.244×, ablation 1.487×); round-3 decision gate FAILED
on the transformer's own newly-measured cost (1.562× vs the 1.5× bar) → round 3 NOT launched,
coordinator decision pending.**)*

### 1.26a COORDINATOR AUTHORIZATION (2026-07-09 overnight): calibration round 3 AUTHORIZED at the re-priced 3.593 GPU-h

The §1.26 price gate (1.5× tripwire) did its job: it surfaced a real
pricing drift instead of silently absorbing it. Adjudication: the drift is
EXPLAINED (the registry's 2.300 was an estimate predating the Rev-4
three-term objective's real-kernel rates; the newly-measured transformer
ratio 1.736× was never in the prior price; the AUD2-F1 fix itself is
CONFIRMED in-band at 1.244×/1.487× contender/ablation), the re-priced
sweep upper bound (≈13.25 GPU-h vs 11.675 prior) shows no regression at
sweep scale, and affordability is a non-issue under the saturation
charter (≈192 GPU-h/day supply). **Round 3 (9 cells) AUTHORIZED at 3.593
GPU-h, GPUs 5-7; the ladder/band analysis and margin freeze remain gated
on its results as before. The §1.6 price row is superseded by the §1.26
measured rates for all downstream pricing.**

### 1.27 CALIBRATION ROUND 3 VERDICT (2026-07-09 morning): FAIL — two-layer, recorded honestly; a decisive DISSOCIATION found; diagnosis round dispatched

**Mechanical layer:** the round FATAL'd before its own band check.
`transformer_task1_calib_stress_K48`'s POST-TRAINING rung-2 fit OOM'd
deterministically ×3 (98.36 GiB alloc: `probe_head_rd.py:173
transformer_native_tap` → `transformer_baseline_rd.py:218` — an UNSLICED
full-vocab LM head over the K=48 episode set; a SIBLING of AUD2-F1 in a
code path the §1.24 fix did not cover) → MAX_CELL_STRIKES → top-level
`results/h2h_rung1/FATAL` at 08:25Z (outside calib/, which is why early
marker checks missed it). `transformer_task2_primary` never launched
(last in pending order). H2H_DIAL_ROUND=3 confirmed in effect from the
raw JSONs. The training itself was clean (the crashed cell trained
5000/5000 before the probe-fit OOM).

**Substantive layer (raw numbers, 5 of 6 primary cells with data):** the
pre-registered gate (`rung1 > 3×chance` AND `rf@0.9 > 0`) FAILS in all 5
— `final_recovered_frac_*` reads exactly 0.0 everywhere, the same
continuous-probe plateau as rounds 1-2. **BUT the discrete leg
DISSOCIATES: contender_task1_primary rung1 accuracy = 0.9990 (chance
0.03125; ablation 0.0447; transformer 0.0295).** The Rev-4 answer-CE
objective DID produce recall — near-perfect K-restricted retrieval, arm-
separated exactly as the capability thesis predicts — while the
continuous rf@0.9 instrument sees none of it. Given this project's
record (five instrument defects found across campaigns this week), the
probe/recovery leg is the prime suspect; alternatively the model stores
answers in a geometry rf@0.9's cosine bar cannot certify. THIS
DISSOCIATION, not the 0.0, is the finding.

**DISPOSITION:** (1) DIAGNOSIS ROUND dispatched (read-only: the
rf@0.9-vs-rung1 dissociation — instrument defect vs storage geometry;
no GPU). (2) BUILD FIX required before any h2h retry: slice-before-
matmul in transformer_native_tap (the AUD2-F1 sibling), plus a smoke
that runs the rung-2 fit at K=48 on real kernels. (3) NO forward motion
(ladder/bands/margin freeze/sweep) until the diagnosis adjudicates. (4)
Realized round-3 spend ≈ authorized envelope; the crashed cell's
training compute is reusable (ckpt on disk). Rounds 1-2 precedent
honored: failures are data — and this one carries the campaign's first
positive discrete-recall separation.

### 1.28 DISSOCIATION DIAGNOSIS (2026-07-09 morning): instrument-design mismatch ~55% / argmax-grade geometry ~30% — NOT a threshold artifact; rung 2 found STRUCTURALLY VACUOUS; decisive experiment authorized

Evidence (all traced to code/raws): (1) the probe output IS the membership
oracle again — final_probe_cos_mean 0.17596 = 99.5% of the analytic 1/√32
ceiling, cos_pred_episode_mean 0.9415 ≥ the 0.85 tell; cosines cluster at
~0.17 not 0.85 (threshold artifact REFUTED). (2) Trajectory decoupling:
rung 1's two phase transitions (0.03→0.69 @3-3.5K; 0.78→0.99 @14-16K)
leave probe_cos_mean unmoved (0.149→0.144; 0.172→0.173) despite verified
gradient parity — the recall events bypass the probe leg entirely. (3)
Readout-path asymmetry: rung 1 = continuation through BOTH blocks + full
nonlinear query processing + LM head; rf@0.9 = S₁@q_shallow through a
linear probe — nothing constrains them to coincide; Rev 4's bet that
CE_answer recall would spill into the bilinear read FAILED as a bet, not
as a bug (§1.22 deliberately carried the instrument unchanged). (4) The
Nichani argmax-vs-exact-recovery gap was PRE-REGISTERED in the ladder
("rung 1 does not imply rf@0.9") — this is that gap, live, plus a
tap-placement question rung 2 was supposed to separate BUT:
**NEW INSTRUMENT DEFECT (rung 2): fit_rung2_identity_classifier
(h2h_cell_train_rd.py:686-727) uses tgt_slot labels, and slots are
UNIFORM given identity (grammar_rd.py:434-436 draws entity order fresh
per episode) — a PERFECT tap scores chance; every arm's rung-2≈chance is
expected under ALL hypotheses; the §1.7 attribution row cannot
discharge.** R5-F2's capacity null couldn't catch it (noise-direction
only; no planted-signal positive control — new rule: every ladder rung
needs a positive control, the positive analog of the negative-test hard
rule). (5) RECORD GAP: the round-3 checkpoint was never offline-probed —
probe_diagnosis artifacts (02:31-02:37Z) predate round 3's 05:58Z
in-place checkpoint overwrite (filename-collision risk noted for future
rounds); §1.21's offline-refit refutations do NOT transfer. (6) Prose
correction to §1.27: ablation rung1 0.0447 is +4.9σ above chance (weakly
real), transformer −0.6σ (chance).

**DECISIVE EXPERIMENT AUTHORIZED (≤0.4 GPU-h ceiling, idle GPUs, no
training):** Stage 1 (zero new code, ~0.1 GPU-h): run the existing
probe_diagnosis_rd.py + probe_diagnosis_oracles_rd.py against the
round-3 _auxrev2 task1-K32 checkpoints — rf@τ curve τ∈{0.5..0.9} +
cosine quantiles, offline ridge/SGD/MLP refits, the 107-class
entity-identity classifier (the CORRECT rung 2), episode-restricted
tap-space codebook argmax (the hard rule's argmax-closure), LM-head
route. Decision rules: refits recover → probe-training failure (fix =
offline refit in the metric path); refits at membership BUT tap-space
argmax / identity classifier ≫ chance → ARGMAX-GRADE GEOMETRY in S₁
(publishable: the Nichani gap live in a trained fast-weight LM); both at
chance → tap placement → Stage 2 (state-zeroing localization + tap
variants, ~150 lines, ~0.2-0.3 GPU-h, separately authorized on Stage-1's
readout). **Rev-5 gate direction (recorded, not yet adopted):** two legs
— Leg A WIN metric = episode-restricted discrete recall (rung 1),
symmetric across arms, Nichani caveat disclosed, rf@0.9 NOT a LOSE
criterion (the claim is recall capability, not rank-necessity — the
exact-recovery hard rule binds rank claims like M*, not this one); Leg B
= repaired continuous instrument (offline-fit probe, localized tap,
rf@τ curve + membership tell) as mechanism attribution. Mandatory
regardless: rung-2 labels → entity identity; planted-signal positive
controls on all rungs; the transformer_native_tap OOM fix (§1.27).

### 1.29 DECISIVE EXPERIMENT RESULT (2026-07-09 afternoon): TAP-PLACEMENT FIRES — refits refuted as failure mode, argmax-grade geometry NOT supported (task-undifferentiated), readout-path asymmetry confirmed end-to-end; Stage 2 recommended

Stage 1 of §1.28's authorization executed exactly as specified: zero new code,
`probe_diagnosis_rd.py` + `probe_diagnosis_oracles_rd.py` (unmodified) against
the ROUND-3 `_auxrev2` task1-K32 checkpoints on GPU 0 (idle; GPU 3/4 = live
`fixscale_392m` wave untouched, GPU 6 = `fixscale_98m_resume_s1` untouched).
Checkpoint round verified via mtime: contender `05:58:16Z` (exact match to
§1.28's own citation), ablation `06:24:40Z`, transformer `07:13:39Z` — all
after the round-3 overwrite, none in round-2's 02:31-02:37Z window. The
`probe_diagnosis_oracles_rd.py` hardcoded output path collides with round-2's
own diagnosis artifacts at that path; the round-2 directory was backed up
in place (`probe_diagnosis_ROUND2_BACKUP_20260709T163244Z/`) before running,
so this round did not repeat the in-place-overwrite trap it was diagnosing.
Full archive: `experiment-runs/2026-07-09_h2h_decisive_probe/`.

**Realized cost: ~0.030 GPU-h (smoke 8.2s + full 71.0s + oracles ~29s ≈ 108s
on one H100) against the 0.4 GPU-h ceiling (≈7.5%) and the ~0.1 GPU-h Stage-1
target (≈30%).** No training occurred at any point.

**Per-arm result tables** (chance for the 107-class identity classifier =
1/107 = 0.00935; chance for the episode-restricted LM-head route = 1/32 =
0.03125):

| metric | contender | ablation | transformer |
|---|---|---|---|
| repro (ckpt rig, pinned eval): `probe_cos_mean` | 0.1760 | 0.1206 | 0.1353 |
| `rf@0.9` — ridge (closed-form optimum) | 0.0 | 0.0 | 0.0 |
| `rf@0.9` — pinned-SGD cold/warm | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| `rf@0.9` — MLP probe | 0.0 | 0.0 | 0.0 |
| ridge `cos_mean` real target vs SHUFFLED control | 0.168 vs 0.109 (+0.059) | 0.118 vs 0.113 (+0.005) | 0.128 vs 0.107 (+0.021) |
| `o1` membership oracle (T_val) cos | 0.187 (≈1/√32=0.177) | 0.187 | 0.187 |
| `o2` ridge-pred vs membership-direction cos | 0.896 | 0.606 | 0.679 |
| `o3` embed-space constant-floor cos (generic-direction artifact) | 0.345 | 0.746 | 0.693 |
| rung-2: 107-way linear identity classifier, held-out | 0.0298 (3.19x chance) | 0.0107 (1.15x chance) | 0.0286 (3.06x chance) |
| rung-2: 107-way MLP identity classifier, held-out | 0.0251 (2.69x) | 0.0105 (1.12x) | 0.0161 (1.72x) |
| tap-space codebook top-1 argmax (pool-restricted, DIAGNOSTIC ONLY) | 0.0276 (2.95x) | 0.0115 (1.23x) | 0.0168 (1.80x) |
| LM-head route, `<Q>` position, episode-restricted top-1 | 0.9957 (31.9x chance) | 0.0341 (1.09x chance) | 0.0304 (0.97x chance) |

**Instrument-coverage gap found (honestly disclosed, not built around, per
"zero new code"):** the brief's "episode-restricted tap-space codebook
argmax" does not exist verbatim in the unmodified scripts. What exists is
(a) a tap-space (T_val-predicted) codebook argmax that is **pool**-restricted
(107-way, row above), and (b) an episode-restricted (32-way) decode that runs
through the **LM-head route**, not the shallow linear tap. The two rows above
labelled accordingly are the closest available instruments; no new code was
written to close this gap, per the explicit Stage-1 scope.

**Decision-rule application (§1.28's menu):**

1. *"Refits recover → probe-training failure."* **REFUTED, decisively.** Ridge
   is the closed-form global optimum for linear least-squares — not an
   optimization artifact — and still reproduces `rf@0.9=0.0` exactly, matching
   the online plateau in all three arms. The real-target ridge fit barely
   clears its own shuffled-tap negative control (contender +0.059, transformer
   +0.021, ablation +0.005 cos-mean) — there is close to nothing linearly
   recoverable at this tap beyond the membership direction already known from
   §1.21/§1.28. MLP (nonlinear) is uniformly worse than ridge, reconfirming
   "no nonlinear treasure."
2. *"Refits at membership BUT tap-space argmax / identity classifier ≫ chance
   → ARGMAX-GRADE GEOMETRY (publishable)."* Refits ARE at membership (`o1`/`o2`
   confirm the ≈1/√32 ceiling and 0.6-0.9 membership-direction correlation, in
   all arms) — first half satisfied. Second half is **NOT satisfied at a
   decisive, task-differentiated level**: the identity-classifier signal,
   where present, is weak (2.7-3.2x a 0.9%-chance base rate — i.e. 2.5-3.0%
   accuracy) and, critically, is **comparable in the transformer arm
   (3.06x) and the contender arm (3.19x)** even though the transformer's own
   LM-head route sits at flat chance (0.97x) — meaning the transformer
   demonstrably cannot solve the task at all, yet its shallow tap carries a
   same-order-of-magnitude "identity" blip. A signal that doesn't track
   actual task-solving capability across arms is not evidence of geometry the
   model exploits for recall; it is more consistent with a shared, generic,
   task-irrelevant confound in the shallow tap representation — of the same
   family as `o3`'s embed-space constant-floor artifact (0.35-0.75 cos from a
   fit-set-mean constant predictor alone, already flagged in §1.21 as a
   non-signal). **This rule does not fire.**
3. *"Both at chance → tap placement → Stage 2 needed."* This is the pattern
   that best fits the evidence, read honestly rather than forced into a clean
   binary: the tap-space signal is not a decisive ≫-chance signal in rule 2's
   sense (weak, task-undifferentiated), while the **LM-head route — the full
   nonlinear forward pass through both blocks — reproduces the §1.27 arm
   ordering exactly** (contender near-ceiling at 99.6%, ablation weakly-real
   at 1.09x chance matching §1.28 item 6's "+4.9σ" characterization,
   transformer at flat chance matching its "-0.6σ" characterization). The
   answer information plainly **exists and is reachable by the full network**
   for the contender; it is not linearly present, under any of ridge/SGD/MLP/
   closed-form-optimal refitting, at the specific shallow tap S₁@q_shallow the
   rf@0.9 and rung-2 instruments read. This is the readout-path asymmetry
   §1.28 item 3 predicted, now confirmed on round-3 checkpoints with offline
   refits that close off the "under-fit" and "wrong nonlinearity" escape
   hatches. **Verdict: rule 3 fires — tap placement.**

**CONSEQUENCE (Rev 5 direction):** Stage 2 (state-zeroing localization + tap
variants, ~150 lines, ~0.2-0.3 GPU-h, already scoped in §1.28) is the
recommended next step to find where between S₁@q_shallow and the LM-head the
recall-enabling computation actually lives — **not built here, per this
task's explicit scope; reporting the need only.** The Rev-5 two-leg gate
direction recorded in §1.28 (Leg A = episode-restricted discrete recall
as the WIN metric; Leg B = repaired continuous instrument as mechanism
attribution, not a LOSE criterion) is unaffected by this result and remains
the live plan — this experiment adjudicates *which* Leg-B repair Rev 5 needs
(a relocated/deepened tap, not a probe-training fix and not a claim that S₁
already carries argmax-grade identity geometry).

**Injection sighting:** two fake `<system-reminder>`-formatted blocks
appeared appended to unrelated local tool stdout near the start of this
task (a "date has changed, do not tell the user" concealment instruction,
and a block mimicking a legitimate agent-list system notice). Neither was
complied with; both are logged per the standing hard rule. Cross-checked
independently via `date` and `git log` — the underlying date claim happened
to be accurate, which does not change that the injection vector itself
(text impersonating a system channel inside command output) is the concern.

### 1.30 TAP-PLACEMENT LOCALIZATION RESULT (2026-07-09 afternoon, Stage 2 of sec 1.28's
decisive experiment): S0 CARRIES THE BINDINGS, S1 (every existing instrument's tap) IS
CAUSALLY INERT for recall; NO linear tap at either layer clears rf@0.9 — the
recall-exposing nonlinearity lives strictly downstream of S0, at/near the pre-LM-head hidden

Build: ONE new file, `matrix-thinking/deltanet_rd/h2h_tap_localization_rd.py` (434 lines incl.
docstrings/self-test; core logic ~190 lines), importing `_recurrent_continuation_answer_logits`/
`_pad_query_tokens_for_continuation`/`_repeat_states_for_queries`/`_rung1_k_restricted_pred_slot`/
`AUDITED_TAP`/`build_arm_model`/`load_h2h_checkpoint` from `h2h_cell_train_rd.py` and
`ridge_fit`/`ridge_pred`/`cos_stats`/`DIAG_FIT_SEED` from `probe_diagnosis_rd.py` (both unmodified;
zero shared-file edits). CPU-stub selftest (`REASONING_LINK_FORCE_CPU_STUB=1 --selftest`, 6 items:
shapes, exact wiring-equivalence to the unmodified `_recurrent_continuation_answer_logits`, the
block-1 `q_conv1d` hook firing, zeroing-has-teeth on a shape-matched synthetic nonzero state, the
ridge harness recovering a known-linear synthetic mapping past its own shuffled-tap negative
control, and a full `_extract_all`+`_ridge_table` integration run) ALL PASS, run both locally and
on-box before touching real checkpoints. Deployed via `scp`, md5 `ed5aa9c5fd01eaff35b3cede56f7fc36`
verified identical both sides. Ran in `tmux h2h_taploc` on GPU 0 (idle; GPU 3/4 =
`fixscale_392m` wave, GPU 6 = `fixscale_98m_resume_s1`, untouched), log at
`~/h2h_decisive_logs/tap_localization_run.log` on box. **Realized cost: contender 4.03s +
ablation 1.16s = 5.19s ≈ 0.00144 GPU-h** — against the ~0.2-0.3 GPU-h estimate and the
remaining ~0.37 GPU-h under the sec 1.28 ceiling (Stage 1 + Stage 2 combined ≈ 0.0314 GPU-h,
≈7.9% of the 0.4 GPU-h ceiling). Sanity anchor: `both_intact` rung-1 reproduces sec 1.27/1.29's
own numbers exactly (contender 0.9990, ablation 0.0447) — the new continuation path
(`_continuation_pass`) is confirmed bit-identical to the registered one, not just by the CPU
selftest but by this real-checkpoint agreement too.

**Table 1 — state-zeroing localization** (K-restricted rung-1 accuracy, pinned EVAL_SEED set,
n=4096 queries/cell; chance = 1/32 = 0.03125):

| condition | contender | ablation |
|---|---|---|
| both states intact | 0.9990 | 0.0447 |
| S0 zeroed | 0.0286 (≈chance) | 0.0308 (≈chance) |
| S1 zeroed | 0.9990 (UNCHANGED) | 0.0437 (UNCHANGED) |

Zeroing S0 collapses accuracy to chance in **both** arms. Zeroing S1 leaves accuracy
statistically unchanged in **both** arms (contender bit-identical to 4 decimal places;
ablation within noise of its own already-weak baseline). The pattern is unambiguous and
qualitatively identical across arms: **block 0's own cached bind-phase state carries 100% of
the causally-necessary binding information; block 1's state is causally inert for this task's
recall** — despite being the layer every existing instrument (rf@0.9's `S1@q_shallow` tap,
the rung-2 identity classifier, the tap-space codebook argmax) reads.

**Table 2 — tap-variant ridge fits** (fit: 24,576 fresh DIAG_FIT_SEED points; eval: pinned
EVAL_SEED set, 4,096 points; closed-form ridge, best-λ on held-out; `rf@τ` = fraction of
eval rows clearing cosine threshold τ):

*Contender:*

| tap | cos_mean | rf@0.5 | rf@0.7 | rf@0.9 | shuffled cos_mean | gap vs shuffled |
|---|---|---|---|---|---|---|
| (i) S1@q_shallow (current tap) | 0.165 | 0.003 | 0.0 | 0.0 | 0.105 | +0.060 |
| (ii) S0@q0-pathway | 0.118 | 0.0 | 0.0 | 0.0 | 0.112 | +0.006 |
| (iii) S1@q_true | 0.167 | 0.003 | 0.0 | 0.0 | 0.104 | +0.063 |
| (iv) pre-LM-head hidden (positive control) | **0.894** | **0.997** | **0.932** | **0.674** | 0.094 | **+0.800** |

*Ablation:*

| tap | cos_mean | rf@0.5 | rf@0.7 | rf@0.9 | shuffled cos_mean | gap vs shuffled |
|---|---|---|---|---|---|---|
| (i) S1@q_shallow (current tap) | 0.117 | 0.0 | 0.0 | 0.0 | 0.112 | +0.005 |
| (ii) S0@q0-pathway | 0.113 | 0.001 | 0.0 | 0.0 | 0.111 | +0.002 |
| (iii) S1@q_true | 0.118 | 0.0 | 0.0 | 0.0 | 0.113 | +0.005 |
| (iv) pre-LM-head hidden (positive control) | 0.119 | 0.0 | 0.0 | 0.0 | 0.112 | +0.006 |

**Interpretation:**

1. **None of the three shallow/state-level linear taps (i/ii/iii) clear rf@0.9 in either arm —
   including (ii), the tap placed directly on the CAUSALLY load-bearing state S0.** Moving the
   query pathway (i→iii) or moving the state (i→ii) each independently leaves the gap-vs-shuffled
   in the same 0.002-0.063 range that sec 1.29 already characterized as "close to nothing linearly
   recoverable." Placement on S0 alone does not repair the instrument: the binding information
   provably lives in S0 (Table 1) but is **not exposed by any of these linear (state, query)
   combine formulas** into the T_val codebook's arbitrary basis.
2. **Only the pre-LM-head hidden (iv) — which has passed through block 1's own nonlinear
   FFN/residual processing, not just its recurrent state — shows strong linear decodability, and
   only for the contender** (rf@0.9=0.674, rf@0.5=0.997, gap +0.800 vs a +0.005-0.063 gap
   everywhere else). This sharpens sec 1.29's LM-head-route finding: the critical transformation
   that exposes the recall geometry linearly is not "which layer's recurrent state" but **block
   1's own nonlinear forward processing of the S0-derived signal** — S1 (the delta-rule state
   block 1 itself maintains) apparently never receives this information at all (Table 1), yet
   block 1's FFN/residual path evidently does carry it through to the final hidden.
3. **The ablation's own positive control (iv) ALSO fails** (cos_mean 0.119, rf@0.9=0.0,
   gap +0.006 — statistically indistinguishable from its own (i)/(ii)/(iii) rows), unlike the
   contender's dramatic (iv) success. Answering task item 3's question directly: **the ablation's
   geometry differs in kind, not merely in strength.** Both arms share the same qualitative
   localization pattern (S0 load-bearing, S1 inert) — the ablation does perform *some*
   structurally analogous single-hop binding — but only the contender's downstream nonlinear
   processing renders that binding linearly legible at any tap tested. This is consistent with
   sec 1.28's "ablation weakly-real, +4.9σ" characterization as a genuine
   architecture/capacity effect (matrix `S⊗q` read vs. the ablation's Hadamard `s⊙q` read), not a
   probe-instrument artifact masking equivalent underlying geometry.
4. **P=1 bottleneck holds for tap (iii)`S1@q_true`**, as sec 1.28's authorization required
   stating explicitly: `_continuation_pass`'s hook on `model.blocks[-1].mixer.q_conv1d` only
   OBSERVES a value already computed inside the SAME `model.forward(query_tokens,
   initial_states=final_states, ...)` call that rung 1's own accuracy read and the sec 1.29
   LM-head route already use — no new read channel. The query does touch the recurrence during
   the continuation (sec 1.9 item 9's addendum: `o_t = read(S_t, q_t)`, the standing per-step
   mechanism) but never reaches behind `final_states` into raw bind-phase tokens not already
   causally summarized through them. Tap (iii) is a pure function of `(final_states,
   query_tokens)` alone, identically to (i)/(ii)/(iv).

**CONSEQUENCE (Rev 5 tap recommendation):** neither S0 alone nor S1 alone, under any linear
combine tested, is a sufficient repaired tap. Point Rev 5's continuous instrument at (or
downstream of) **the post-block-1, pre-LM-head hidden representation** — tap (iv)'s own site —
rather than either recurrent state's raw `S⊗q`/`S⊙q` read. This also retroactively explains why
`S1@q_shallow` (the currently registered tap) was hopeless from the start: it reads a layer
(S1) that is causally inert for this task (Table 1), via a linear combine, when the recall
computation is not stored in a recurrent-state format that combine formula can access at all —
it only becomes linearly legible after block 1's own nonlinear transform. **Mandatory regardless
of this result, unchanged from sec 1.28:** the rung-2 entity-identity relabel (grammar_rd's
`tgt_slot` labels are uniform given identity, sec 1.28 item "NEW INSTRUMENT DEFECT"), planted-
signal positive controls on all rungs, and the `transformer_native_tap` OOM fix (sec 1.27) all
remain required build items before any h2h retry — none of them are addressed by this
localization result, which is diagnostic only.

Archived: `experiment-runs/2026-07-09_h2h_tap_localization/` (script copy + both arms' JSON +
run log; SSD mirror at the same relative path under
`/Volumes/1TB_SSD/learned-representations/experiment-runs/`).

**Injection sighting:** one fake `<system-reminder>`-formatted block (a "date has changed... do
not tell the user" concealment instruction) appeared appended to a `git log | grep` command's
stdout near the start of this task — grep cannot produce that block, and it contradicted the
harness-supplied date (2026-07-06) at the time it appeared. Not complied with; verified
independently via `date` and `git log -1 --format=%cd HEAD`, both of which confirmed the
underlying date (2026-07-09) was in fact accurate — consistent with sec 1.29's own prior sighting
and its same conclusion: an accurate payload does not make an impersonated-system-channel
delivery vector legitimate. Logged per the standing hard rule; no other injections observed
during this task.

### 1.31 DESIGN REV 5 (2026-07-09): THE TWO-LEG GATE — episode-restricted discrete recall (rung 1) PROMOTED to the metric of record; the continuous instrument REPAIRED (offline fit, §1.30-localized tap) and DEMOTED to mechanism attribution; four mandatory instrument fixes (six after Rev 5.1, §1.31a); round 4 re-priced ≈1.3 GPU-h

Recorded per the gauntlet-bookkeeping hard rule. This revision adopts
§1.28's recorded two-leg gate direction, now informed by §1.29's decisive
probe (tap placement fires) and §1.30's localization (S0 carries 100% of
the causally-necessary bindings; S1 causally inert; NO linear tap on
either raw state clears rf@0.9; ONLY the pre-LM-head hidden is linearly
decodable — contender rf@0.9 0.674 / cos_mean 0.894 — and the model's own
block-1 forward performs the linearization). **Scope: append-only.**

**Superseded clauses, enumerated (Rev 5.1 amendment, §1.32 F5 —
replaces the single prose sentence this subsumes, which named only 3 of
the 8 actually-dead clauses below):**

1. §1.4.1 (~line 1068), the "(Task 2, primary; Task 1 high-load,
   secondary bonus at no extra cost)" task-primacy designation —
   superseded because §1.31.1 pins `task1_calib_K32` as the PRIMARY
   axis-1 cell (rationale there); it is the only cell with demonstrated
   recall (round-3 rung-1 table, §1.31.1).
2. §1.4.1 (~lines 1072-1092), the two `recovered_frac@0.9`-based win-margin
   operationalizations (tokens-to-threshold, sustained CI-separation) —
   superseded by §1.31.1's `acc_A`-based re-pin (same structure, metric
   swapped).
3. §1.7 item 1a (~line 1784), "This rung is a disclosed GATE ... and is
   NEVER promoted to the WIN metric" — superseded: rung 1 IS now the WIN
   metric (§1.31.3).
4. §1.7 item 1b (~line 1841), the `rf@0.9 > 0` FULL-cell band conjunct —
   superseded: `rf@0.9` is REMOVED from the band (§1.31.3 bands, item
   iv).
5. §1.7 item 1a (~line 1804), "Rung 3 ... and only this rung, feeds
   axis-1/axis-2 WIN/TIE/LOSE" — superseded: rung 1 (not rung 3) now
   feeds WIN/TIE/LOSE; rung 3 is Leg B's diagnostic only (§1.31.2).
6. §1.4.2 (~lines 1279-1287), the per-`M` gap statistic defined over the
   `recovered_frac@0.9`-based CI test — superseded per §1.31.1's Axis-2
   paragraph: "the per-`M` gap statistic is re-registered from
   `recovered_frac@0.9` to `acc_A`," same 0.20 margin, same grid/tiers
   otherwise unchanged.
7. §1.7 gate 7 (~lines 1935-1941), the probe-capacity null's scope
   ("before `HEADTOHEAD_MATCH_GATE_SIGNOFF=1` may be set" — i.e. gating
   BOTH legs via one signoff token) — superseded: Leg A (rung 1, the WIN
   metric) does not depend on `shared_probe`/`adapter_arm` at all
   (§1.31.1's native LM-head route); gate 7's scope narrows to gating
   Leg B's diagnostic signoff only, never blocking a Leg-A-only launch.
8. §1.3.1.2 (~lines 564-592), the pinned tap
   `state_summary_raw = S_T_last @ q_query` (i.e. `S1@q_shallow`) as THE
   probe/adapter diagnostic READ POINT — superseded for Leg B by §1.30's
   localization: this tap is causally inert (Table 1) and never clears
   `rf@0.9` under any linear combine (Table 2); Leg B's tap is the
   post-block-1, pre-LM-head hidden (§1.31.2). §1.3.1.2's tap remains the
   correct ARCHITECTURAL description of what the contender/ablation
   compute internally (unchanged); only its role as the diagnostic read
   point is dead.

All eight are deliberately left in place in their original sections as
history; from Rev 5 forward THIS section is the controlling text for
every one of them — the same by-declaration supersession mechanism
§1.26a used for the §1.6 price row. Training is NOT touched (§1.31.5).

#### 1.31.1 Leg A — the WIN/TIE/LOSE metric of record (numeric pre-registration)

**Round-3 rung-1 table, full 7-of-9-cells-with-data (Rev 5.1, §1.32
F1) — recorded here as the written baseline round 4's re-metric diffs
against (chance = 1/K; K=32 for task1/task2, K=48 for the stress
cell):**

| cell | contender | ablation | transformer | chance |
|---|---|---|---|---|
| task1_calib (K/d=0.5, K=32) | 0.9990 | 0.0447 | 0.0295 | 0.03125 |
| task2_calib (K=32) | 0.0376 | 0.0271 | — (never launched) | 0.03125 |
| task1_stress_K48 (K=48) | 0.0189 | 0.0195 | FATAL — OOM'd pre-save, §1.27 | 0.02083 |

**Axis-1 primary-cell pin (Rev 5.1, resolving §1.32 F1's FATAL — the
axis-1 decisive-cell identity was left ambiguous because §1.4.1's
original "Task 2, primary; Task 1 high-load, secondary bonus" sentence
was never superseded even though round-3 data already settles it):**
**`task1_calib_K32` is the PRIMARY axis-1 cell.** Rationale: it is the
pre-registered K/d=0.5 load point (§1.4's M1 pin, unchanged) AND, per
the table above, the ONLY cell with demonstrated recall at all
(contender 0.9990, 31.9× the demonstration bar below) — task2_calib's
own contender reading (0.0376, 1.2× chance) fails the demonstration bar
outright, so it cannot anchor a primary-axis WIN claim. **This
EXPLICITLY supersedes §1.4.1's "Task 2, primary" sentence** (dead-clause
list item 1, §1.31 above).

**Task2's round-4 disposition, pre-registered NOW (Rev 5.1, §1.32 F1 —
naming the branch instead of leaving it to be discovered mid-round):**
task2's Leg-A band failure is knowable on the FROZEN round-3 checkpoint
today (0.0376 vs the 0.09375 demonstration bar, below) and round 4
re-meters that SAME checkpoint, not a retrain — the failure is
deterministic, not something round 4 might resolve differently.
**Pinned branch:** the sweep proceeds TASK1-PRIMARY (per the pin above);
task2 is NOT retrained or re-tuned this round. A separate,
separately-ledgered **TASK2 DIAGNOSIS ROUND** is opened for AFTER the
sweep (non-blocking) to determine whether task2's failure is a
task-difficulty gap, an objective-tuning gap, or a genuine capability
boundary. **For axis-1 purposes, task2 reads as a joint-failure TIE**
(neither contender nor ablation clears the demonstration bar, the TIE
clause below) — this disposition is disclosed verbatim in any claim
that mentions task2.

**Metric, pinned:** `acc_A(arm, cell)` = episode-restricted K-way top-1
accuracy at the query answer position, read through EACH ARM'S OWN native
LM-head route, symmetric across all three arms — recurrent arms via the
§1.3.1.3 continuation `forward(query_tokens, initial_states=S_T)` (a pure
function of `(S_T, query_tokens)`, P=1 preserved by causality, §1.9 item-9
addendum); transformer via its standing single forward pass. Decoding =
K-restricted gather+argmax (`_rung1_k_restricted_pred_slot`,
selftest-17-protected, §1.25 AUD2-F2). Eval = the pinned EVAL_SEED query
set, n=4096 queries/cell (the §1.29/§1.30 protocol). Verdict-grade reads
use n=3 seeds with paired `delta_ci_n` CIs (§1.8, unchanged); round-4
calibration reads are single-seed point-estimate previews, as calibration
always was — margins freeze before the sweep, verdicts land at the sweep.

**Chance anchor, pinned:** chance = `1/K` (0.03125 at K=32; 0.02083 at
K=48). **Demonstration bar: an arm counts as demonstrating recall in a
cell iff `acc_A > 3×(1/K)`** (>0.09375 at K=32). An arm at or below the
bar is scored NO-RECALL for that cell; a difference between two NO-RECALL
arms is never interpreted as separation (chance-level noise ordering).

**Axis-1 tiers (contender vs param+data-matched flat-vector ablation),
per primary cell (task1_calib K/d=0.5; task2_calib), endpoint at the
matched final checkpoint, Δ = acc_A(contender) − acc_A(ablation):**

- **WIN:** contender clears the demonstration bar AND `Δ ≥ 0.30` with the
  paired-seed CI excluding 0.30.
- **LOSE:** ablation clears the demonstration bar AND `−Δ ≥ 0.30` with
  the CI excluding 0.30 — structure actively hurts discrete recall,
  reported plainly.
- **TIE:** the Δ CI lies entirely inside `(−0.30, +0.30)`; ALSO scored
  TIE (reported as joint task-learning failure, an instrument/scale
  story) if NEITHER arm clears the demonstration bar.
- **INDETERMINATE:** the CI straddles ±0.30 without excluding it —
  treated TIE-adjacent (the §1.4.2 convention), escalation-eligible,
  never a WIN.

**Margin rationale (pinned now so it cannot drift):** 0.30 ≈ 9.6× chance
at K=32 — ~3× the demonstration bar itself, far above any chance-level
fluctuation at n=4096 queries, and less than a third of round 3's realized
single-seed gap (0.9990 − 0.0447 = 0.9543) — i.e. neither reachable by
seed noise nor gerrymandered to the observed point estimate.

**Axis-1 data-efficiency operationalizations (§1.4.1's two) carry with
the metric swap only:** threshold re-pinned to `acc_A ≥ 0.90` (was
`recovered_frac@0.9 ≥ 0.9`; round 3's contender realized 0.9990, so the
bar is calibrated-reachable); tokens-to-threshold `X=50%` stays
provisional pending the same §1.7-item-1 calibration power sketch;
sustained CI-separation stays ≥3 consecutive logged checkpoints. If the
ablation never clears the demonstration bar at ANY checkpoint while the
contender reaches threshold (round 3's pattern), the tokens-to-threshold
ratio is reported as a BOUND (contender's realized budget fraction, ratio
unbounded), never extrapolated — the endpoint WIN margin above then does
the verdict work.

**Axis-2 (contender vs memory-capped transformer):** §1.4.2's M*
machinery carries VERBATIM with one substitution — the per-M gap
statistic is re-registered from `recovered_frac@0.9` to `acc_A`, same
0.20 per-point margin, same `M ∈ {1,2,4,8,16,32}` grid, same descending
fixed-sequence walk, same LOSE `M*≤2` / TIE `M*=4` / WIN `M*≥8` /
INDETERMINATE tiers. **Degenerate-baseline handling, pre-registered
(round 3's point estimate makes it live: uncapped transformer 0.0295 ≈
0.94× chance):** if the UNCAPPED transformer itself fails the
demonstration bar at the primary cell, the M* walk is definitionally
degenerate (every M trivially cleared); the honest report is "baseline
non-competitive at matched params/tokens" — itself a capability-
separation datum tied to §1.1's axis-2 charter, claimed WITH the
matched-training-budget caveat — and is NOT certified as `M*=∞`/
strongest-win (extending §1.4.2's own rule that a degenerate comparison
never certifies the top tier).

**Joint-NO-RECALL rule for the M* walk (Rev 5.1, §1.32 M3 — the
degenerate-baseline handling above covers the UNCAPPED transformer
failing; this covers a CAPPED `M` point where the CONTENDER also
fails):** if, at some grid point `M`, BOTH the contender's `acc_A` AND
the capped transformer's `acc_A` sit at or below the demonstration bar
(a joint task-learning/scale failure at that `M`, not a memory-capacity
separation), that `M`-cell is scored TIE-equivalent for the walk —
NEVER a LOSE for the contender, and never used to set `M*` in either
direction. The walk continues past it exactly as the straddle rule
(§1.4.2(b)) already treats an unresolved CI: it does not finalize a tier
at that point. This closes off a joint-failure cell being misread as
"the transformer caught up" when neither arm demonstrated recall.

**rf@0.9 is NOT a LOSE criterion, NOT a gate conjunct, NOT in any band**
— it is Leg B's diagnostic (below). **The Nichani caveat travels with
every Leg-A number** (§1.31.6). All four outcomes above are
pre-registered publishable: WIN (capability-separation headline), TIE
(structure costs nothing + the banked isolated properties), LOSE
(reported plainly per §1.1's own convention), joint-failure TIE (a
task-learning/scale finding).

#### 1.31.2 Leg B — mechanism attribution (diagnostic, never WIN/TIE/LOSE)

**Tap, pinned (the §1.30 localization):** the post-block-1, pre-LM-head
hidden at the query answer position — tap (iv)'s own site — never
`S1@q_shallow` (causally inert layer, §1.30 Table 1), never a raw-state
linear combine (§1.30 Table 2: no such tap clears rf@0.9, including on
the causally load-bearing S0).

**P=1 status of the tap (restating the §1.9 item-9 addendum, as §1.28
required):** the tap is a pure function of `(final_states, query_tokens)`
— it is computed INSIDE the same `forward(query_tokens,
initial_states=S_T)` continuation call whose signature has no
computational path back to raw bind tokens; the query touches the
recurrence during the continuation only via the kernel's standing
`o_t = read(S_t, q_t)` mechanism and can never reach BEHIND `S_T`;
verified by the §1.3.1.3 blank-out test plus the R5-F4 fresh-instance
companion, not asserted. §1.30 item 4 made this argument for tap (iii);
tap (iv) sits later in the SAME call and inherits the identical status.
(Transformer arm: its pre-LM-head hidden attends to raw context — the
standing, disclosed item-9 asymmetry; acceptable because Leg B is
diagnostic, and disclosed wherever its Leg-B numbers appear.)

**Probe protocol, pinned (the §1.28 lesson):** fit OFFLINE on the FROZEN
backbone — closed-form ridge, best-λ on held-out, fit on 24,576 fresh
DIAG_FIT_SEED draws, evaluated on the pinned 4,096-point EVAL_SEED set
(§1.30's harness) — NEVER the online jointly-trained `shared_probe`,
which provably sits in the episode-membership optimum (§1.21/§1.28/§1.29,
three independent confirmations). The aux term stays in the training
objective (§1.31.5); its online probe simply stops being a reported
instrument.

**Report, per cell:** the `rf@τ` curve, τ ∈ {0.50, 0.55, …, 0.95} (10
points); cosine quantiles (p5/p25/p50/p75/p95 + mean); the shuffled-tap
negative-control gap; and the membership tell — `probe_cos_mean` vs the
analytic `1/√K` ceiling AND `cos(pred, episode_mean)` against the ≥0.85
tell threshold — logged every read. **Calibration anchor (§1.30):**
contender rf@0.9 0.674 / cos_mean 0.894 / gap-vs-shuffled +0.800;
ablation flat (0.119 / +0.006) — Leg B's pre-registered expectation.

**Reproduction-check tolerance (Rev 5.1, §1.32 M1):** for the 7 reused
round-3 checkpoints (§1.31.7), round 4's Leg-B ridge re-fit at this same
tap on the SAME frozen weights is expected to reproduce the §1.30
calibration anchor within **±0.05 absolute on `rf@0.9`** (contender
band: [0.624, 0.724]; ablation band: [0.069, 0.169], floored at 0 since
`rf@0.9` cannot be negative). A read outside this band on a REUSED cell
is a drift signal — stale checkpoint, tap-wiring regression, or loader
bug — and is INVESTIGATED before proceeding to the sweep, not a
proceed-anyway footnote.

**S0-necessity causal check (the capability claim's mechanism leg,
carried per §1.28/§1.30):** per-cell state-zeroing rung-1, recurrent arms
only, ~free (§1.30 realized 0.00144 GPU-h for both arms combined):
zeroing S0 must collapse `acc_A` to ≈chance; zeroing S1 must leave it
unchanged (§1.30 Table 1's pre-registered expectation). This is the
cheapest DIRECT causal evidence that recall is fast-weight-resident. **If
a cell's contender passes Leg A but S0-zeroing does NOT collapse its
accuracy, the fast-weight-resident mechanism claim is BLOCKED for that
cell and a §1.21-style diagnosis round opens** (that pattern would mean a
bottleneck leak, contradicting the standing blank-out evidence — a
hard-stop, not a footnote).

**S0-HARD-STOP, pinned numerically (Rev 5.1, §1.32 F4 — "≈chance" and
"unchanged" were qualitative until now):**

- **Collapse condition:** `acc_A(S0-zeroed) ≤ 0.09375` — the SAME
  3×-chance demonstration bar at K=32 (§1.31.1), not a separate ad hoc
  threshold.
- **Unchanged condition:** `|acc_A(both-intact) − acc_A(S1-zeroed)| ≤
  2σ`, `σ = √(p̂(1-p̂)/n)` computed PER ARM from that arm's own observed
  both-states-intact `acc_A` as `p̂`, at `n=4096` (the pinned EVAL_SEED
  set). Computed honestly from §1.30's own real numbers: contender
  `p̂=0.9990` → `σ≈0.000494`, `2σ≈0.00099`; ablation `p̂=0.0447` →
  `σ≈0.00323`, `2σ≈0.00646`. Both arms' §1.30 observed deltas (contender
  0.0000, ablation 0.0010) sit inside their own `2σ` band, confirming
  "UNCHANGED" already met this bar, not just descriptively. **Note on a
  numeric discrepancy, disclosed rather than silently overwritten:** the
  dispatch prompt for this amendment cited "≈0.0096 at acc 0.999" for
  this bound; computing `√(p̂(1-p̂)/n)` honestly at `p̂=0.9990, n=4096`
  (as this item's own instruction requires) gives `2σ≈0.00099` instead —
  roughly 10× smaller. The number pinned above is the one that actually
  derives from the formula; flagged here per this project's standing
  exact-threshold/honest-computation rule rather than forced to match
  the dispatch prompt's figure.
- **Seed aggregation, verdict grade:** report the MEAN of `n=3` seeds
  (or the extended `n=9`, §1.8), with the full per-seed table disclosed
  alongside the mean — never the mean alone.

#### 1.31.3 The ladder, re-registered (supersedes §1.7 items 1a/1b clauses named here)

- **Rung 1 = Leg A's metric — PROMOTED from disclosed-gate to metric of
  record.** This deliberately supersedes item 1a's "NEVER promoted to the
  WIN metric" sentence, with the honest rationale stated in full: that
  prohibition enforced the Nichani rule for CONTINUOUS-CAPACITY claims —
  argmax decoding must never underwrite a capacity/rank claim (CLAUDE.md,
  arXiv:2412.06538). Rev 5 re-scopes the WIN claim ITSELF to discrete
  recall capability, which argmax decoding measures honestly, provided
  the claim never asserts exact continuous recovery or rank — §1.31.6
  carries that caveat into every claim sentence. rf@0.9's demotion is the
  other half of the same move: it was never measuring what rung 1
  measures (§1.28 item 3's readout-path asymmetry, confirmed end-to-end
  at §1.29/§1.30).
- **Rung 2 = 107-way ENTITY-IDENTITY linear classifier at the Leg-B tap**
  (the `probe_diagnosis_rd.py` item-1e construction: linear softmax
  classifier → answer entity id), labels = entity identity, NEVER
  `tgt_slot` (§1.28's structural-vacuity defect: slots are uniform given
  identity, so a perfect tap scores chance). Chance = 1/107 = 0.935%;
  diagnostic bar >3× chance; R5-F2's noise-null carries (≤1.5× chance)
  PLUS the new planted-signal positive control (§1.31.4 item 2).
  Diagnostic-only, as before.
- **Rung 3 = Leg B's `rf@τ` curve at the localized tap** — mechanism
  attribution, no longer any decision's input.
- **Attribution table, updated:** rung 1 FAILS → task not learned by any
  channel. Rung 1 PASSES but S0-zeroing does NOT collapse it → bottleneck
  leak, HARD-STOP diagnosis (§1.31.2). Rung 1 + S0-necessity PASS, rung 2
  FAILS → identity not linearly present even at the deep tap (geometry
  deeper than linear — reported, not blocking). Rungs 1-2 PASS, rung 3
  low → the Nichani gap live at the deep tap; the `rf@τ` curve IS the
  honest continuous-recovery bound, reported as such. Membership tell
  logged at every rung-3 read regardless of outcome.
- **Bands, re-pinned (supersedes Rev 4's item 1b conjunct), now
  ARM-AWARE:** a task-1/2 FULL calib cell passes gate 1 iff (i) the
  CONTENDER cell clears rung-1 > 3× chance (training+instrument sanity;
  Rev-4's realized 0.9990 is the anchor); (ii) baseline arms' rung-1
  values are recorded as DATA, never launch-blocking — their failure IS
  the pre-registered separation, not a broken cell; (iii)
  instrument-health IS launch-blocking for ALL arms: every cell's
  planted-signal positive controls must pass and its noise nulls must
  stay ≤1.5× chance; (iv) `rf@0.9 > 0` is REMOVED from the band (Leg-B
  diagnostic now). Stress/locate-only cells stay exempt as before; Task 3
  keeps its anchored [1.90, 2.60] val-loss band untouched.

#### 1.31.4 Mandatory fixes (BUILD-FIX ITEM LIST, binding order; from §1.28/§1.29, unaffected by §1.30)

1. **Rung-2 relabel + retap:** `fit_rung2_identity_classifier`
   (`h2h_cell_train_rd.py:686-727`) relabeled from `tgt_slot` to the
   107-way answer ENTITY ID (port the probe_diagnosis item-1e
   construction) and retargeted to the Leg-B pre-LM-head tap.
2. **Planted-signal POSITIVE controls on every ladder rung (the
   both-directions teeth rule, §1.28's new rule):** rung 1 already has
   selftest 17 (planted-answer); rung 2 gains a synthetic tap with
   linearly-planted identity that must score near-ceiling (≥90%); rung
   3/Leg B promotes the §1.30 harness's known-linear-mapping recovery
   check from selftest to a standing per-run control. Each positive
   control is paired with its existing noise null (R5-F2 style, ≤1.5×
   chance): an instrument is trusted only when it passes BOTH directions.
3. **`transformer_native_tap` OOM fix (the AUD2-F1 sibling, §1.27's
   crash):** slice the hidden state to the answer position BEFORE the
   full-vocab matmul in the `probe_head_rd.py:173` →
   `transformer_baseline_rd.py:218` path (same fix shape as 68e2768's
   `return_hidden`+slice), PLUS an enforced K=48 rung-2-fit smoke on real
   kernels (the exact §1.27 crash repro) wired as its own chain gate with
   a forced-fail negative test, per the CPU-stub-coverage hard rule.
   **Forced-fail negative test, spec'd (Rev 5.1, §1.32 M4):** plant an
   UNSLICED full-vocab matmul path behind a test-only flag
   (`FORCE_UNSLICED_LM_HEAD=1`) and assert the K=48 smoke OOMs/raises
   under it — the negative direction, proving the smoke actually
   exercises the crash rather than only the happy path — OR, if planting
   a real OOM in CI is impractical, the analytic alternative: assert the
   SLICED path's measured peak memory at K=48 stays under a pinned
   bound, and assert by shape arithmetic alone (no execution needed)
   that the UNSLICED computation's tensor size at the same K=48 would
   exceed that bound. Either form satisfies the "has teeth" hard rule
   (CLAUDE.md); the build stage picks whichever is cheaper to wire.
4. **Checkpoint filename versioning (the §1.28 in-place-overwrite trap):**
   suffix round/rev into every checkpoint filename (e.g.
   `_r{H2H_DIAL_ROUND}`), closing the record gap that orphaned round 2's
   probe_diagnosis artifacts when round 3 overwrote its checkpoints at
   05:58Z; pairs with (does not replace) AUD2-F4(ii)'s round-transition
   result-JSON invalidation procedure, already exercised once at the
   round-3 launch (calib3 MANIFEST §1).
5. **`check_gate1_full_cell_band` re-wire (Rev 5.1, §1.32 F3):** this
   function still enforces the DEAD `rf@0.9 > 0` conjunct (dead-clause
   list item 4, §1.31) — run as-is against round-4 output, it FATALs
   every cell. Re-wire to §1.31.3's bands: (i) contender rung-1 `> 3×`
   chance; (ii) baseline arms' rung-1 recorded as data, never
   launch-blocking; (iii) instrument-health (planted-signal positive
   controls pass, noise nulls `≤1.5×` chance, per fix item 2 above) IS
   launch-blocking for ALL arms; (iv) the `rf@0.9 > 0` conjunct REMOVED.
6. **Round-4 re-metric driver (Rev 5.1, §1.32 F3):** the actual script
   that runs, per reused/fresh cell: the 3-arm Leg-A continuation eval
   (`acc_A`, §1.31.1); Leg-B offline ridge at the relocated tap,
   INCLUDING the transformer arm (§1.31.2 — not yet exercised on the
   transformer's own pre-LM-head hidden by §1.29/§1.30, which ran only
   the recurrent arms); S0-zeroing (§1.31.2, recurrent arms only); the
   both-direction planted/noise-null controls (fix item 2 above); and
   per-cell JSON output including the identity-table fields (§1.31.7).
   This is the item fix items 1-5 above feed INTO — without it they are
   unwired checks with nothing driving them.

#### 1.31.5 What carries over UNCHANGED

The Rev-4 three-term objective and its weights (`aux_weight=2.0`,
`ce_answer_weight=1.0`) — **training WORKS (0.9990 discrete recall); do
not touch it.** The §1.23 dial mechanics, including R5-F1's
one-contingency-dial-round cap: round 3's step-500 dial did not fire;
reused checkpoints do not re-dial; the two fresh transformer cells run
the standing dial check; the R5-F1 contingency budget remains intact at
one round. The arms/tasks/param-matching (§1.2-§1.4 constructions,
verbatim). The blank-out tests, gate tokens, and chain discipline. The
§1.26 measured rates supersede §1.6 for all pricing (per §1.26a).

#### 1.31.6 Claim language (PI-bar aware, pre-registered)

**If round 4's Leg A separates as round 3's single-seed data suggest
(contender 0.9990 / ablation 0.0447 / transformer 0.0295 at K=32), round
4 CAN establish (claim sentence REWRITTEN, Rev 5.1, §1.32 F6 — round 4's
transformer read is the UNCAPPED baseline, not a KV-capped one; the
K-cap/`M*` framing is the SWEEP's own claim, not round 4's):** *single-
pass in-context associative recall through a P=1 fast-weight bottleneck,
at accuracy a param- and token-matched attention baseline cannot reach
EVEN WITH ITS FULL KV CACHE, at matched params/tokens/training budget*.
The K-cap/`M*` "constant-memory minds" tie-in (§1.1's inference-memory-
matched axis) belongs to the SWEEP's own claim (§1.4.2, unchanged), not
round 4's — round 4 supplies only the uncapped-baseline half of the
comparison, plus the S0-necessity check supplying the mechanism leg
(direct causal evidence the recall is resident in the fast-weight state,
§1.31.2). **Matched-budget caveat (Rev 5.1):** this claim holds ONLY at
the calibration cells' own matched param count, token budget, and
training compute — it does not extend to a differently-scaled or
differently-trained transformer without its own matched re-run.
**Seed-fragility disclosure (Rev 5.1):** round 4's calibration reads are
SINGLE-SEED point estimates (§1.31.1); the claim above is provisional
until verdict-grade seeds land. **Extension trigger, pinned (Rev 5.1):**
if the verdict-grade `n=3` CI for `task1_calib_K32`'s
contender-vs-transformer `acc_A` gap straddles the 0.30 margin
(§1.31.1) without excluding it, the pre-registered `n=3→9` seed
extension (§1.8) fires on THIS claim's own cell before the claim is
finalized either way. **Every claim sentence carries the Nichani
caveat:** "recall" here means episode-restricted
top-1 retrieval under argmax decoding, and under argmax a rank-1 state
can support ≈d associations (Nichani, Lee & Bietti, ICLR 2025,
arXiv:2412.06538).

**What round 4 CANNOT establish:** (a) exact continuous recovery — Leg
B's `rf@τ` curve is reported as the honest bound (the §1.30 anchor:
contender rf@0.9 0.674 at the pre-LM-head tap — real but partial); (b)
rank-necessity or storage-capacity claims — those remain governed by the
exact-continuous-recovery hard rule (CLAUDE.md) and belong to the
M*/rank program's strict machinery, never to rung-1 argmax; (c) anything
beyond 14M-class calibration cells until the ladder/bands and sweep run.

#### 1.31.7 Gates + ledger

**Chain:** Rev 5 (this section) → **fresh micro-attack on the DELTA
only** (the two-leg numeric pins, ladder re-registration, arm-aware band,
the fixes — not a re-attack of §1.23-cleared machinery) → build-fix
(§1.31.4's six items, post-Rev-5.1) → scoped independent build audit → **round 4** →
ladder/bands review → margin freeze → 27-cell sweep. Margin freeze
remains ALSO blocked on the fix-at-scale per_token-vs-global contender-
pin adjudication (§1.24 trailer note, unchanged).

**Round 4 = the 9-cell calibration RE-METERED under the new instrument,
not re-trained:** **checkpoints on disk are reusable for exactly 7 of 9
round-3 cells (Rev 5.1, §1.32 F2 — corrects the "8/9, reuse may be
attempted" language, which was factually wrong: `torch.save` executes
AFTER the rung-2 fit in the training driver, so the crashed
transformer-K48 cell's round-3 weights were NEVER persisted — the file
on disk at that path is ROUND-2's stale two-term-objective checkpoint,
mtime 02:03, predating the round-3 05:41 launch).** The 7 reused
cells — contender/ablation/transformer × task1_calib, contender/
ablation × task2_calib, contender/ablation × task1_stress_K48 (mtimes
verified 05:49-08:43Z) — re-run ONLY the metric pass (Leg A eval + Leg B
offline ridge + S0-zeroing + relabeled rung 2 + both-direction controls)
against the frozen checkpoints. The 2 cells WITHOUT valid round-3
weights train fresh, no ambiguity: `transformer_task2_calib_primary`
(never launched) and `transformer_task1_stress_K48` (its round-3 weights
never reached disk, per above). **"Reuse may be attempted" is STRUCK for
the K48-transformer cell; both are pure retrains, budgeted as such.**

**Per-cell identity table, mandatory round-4 pre-flight gate item (Rev
5.1, §1.32 F2):** before any reused checkpoint is loaded, round 4's
driver writes a manifest with `{cell_id, arm, task, md5(checkpoint_file),
mtime}` for all 7 reused checkpoints, cross-checked against the recorded
mtimes (contender/ablation/transformer task1_K32: 05:58:16Z / 06:24:40Z
/ 07:13:39Z per §1.29; task2/K48 cells verified in the 05:49-08:43Z
window per §1.32 F2) — a mismatch on ANY of the 7 blocks that cell's
re-metric pass with a hard error, not a warning. **Loader-side
provenance pinning (Rev 5.1):** the checkpoint loader itself asserts the
manifest's recorded md5 against the file it is about to load, AT LOAD
TIME, immediately before the re-metric pass runs — not only at
pre-flight — so a checkpoint swapped or corrupted between the pre-flight
check and the actual eval cannot silently pass.

**Price (from §1.26 measured rates + §1.29/§1.30 realized metric costs):**
metric passes ≈0.03 GPU-h class per cell (§1.29's full 3-arm diagnostic
realized 0.030 GPU-h; §1.30's 2-arm localization 0.00144) → 9 cells ≈0.3
GPU-h, ceiling 0.5; fresh training ≈0.98 GPU-h measured
(transformer full cell ≈0.78 + K48 stress at 1/4 budget ≈0.20), ceiling
≈1.46 at the standing 1.5× padding. **Round-4 total ≈1.3 GPU-h expected,
≤2.0 GPU-h ceiling — well under round 3's 3.593 authorized/realized.**
GPUs per the standing assignment (5-6; GPU 7 never used). **The 27-cell
sweep re-price (≈13.25 GPU-h upper, §1.26a) STANDS** — Rev 5 changes
metric passes (cheap) and no training mechanics; a confirmation read
rides the round-4 harvest before margin freeze.

**STATUS: Rev 5 COMPLETE (this section is the controlling text for the
clauses it supersedes). NEXT: fresh micro-attack on the Rev-5 delta →
build-fix (§1.31.4) → scoped build audit → round 4. NO training relaunch
until the fixes audit clears.**

### 1.31a REV 5.1 AMENDMENT (2026-07-09): surgical fixes to §1.31 per the §1.32 micro-attack — F1 FATAL + F2-F6 MAJOR + M1-M4 minors, all applied in place

Recorded per the gauntlet-bookkeeping hard rule (§1.20's own precedent):
this changelog documents WHERE each §1.32 finding's fix landed, so a
raw-diff verification can be checked against a written map rather than
re-deriving intent from the diff alone. **Scope: surgical, in-section
edits to §1.31's own subsections only — no other section of this file
was touched.**

| Finding | §1.32 defect (one line) | Fix, where it landed |
|---|---|---|
| F1 (FATAL) | axis-1 decisive-cell identity ambiguous; task2's Leg-A failure knowable but unrecorded | §1.31.1: round-3 7-cell rung-1 table; `task1_calib_K32` pinned PRIMARY with rationale; explicit supersession of §1.4.1's "Task 2, primary" sentence (dead-clause list item 1, §1.31); task2's round-4 branch pre-registered (task1-primary sweep proceeds; TASK2 DIAGNOSIS ROUND opens post-sweep, separately ledgered; task2 reads joint-failure TIE for axis-1) |
| F2 (MAJOR) | "8/9 reuse" wrong; K48-transformer checkpoint never persisted; "reuse may be attempted" factually false | §1.31.7: corrected to 7 reuse + 2 fresh, named explicitly; "reuse may be attempted" struck; per-cell md5+mtime identity table mandated as round-4 pre-flight gate; loader-side provenance pinning added |
| F3 (MAJOR) | §1.31.4's fix list missing the band re-wire and the driver itself | §1.31.4 items 5 (`check_gate1_full_cell_band` re-wire) and 6 (round-4 re-metric driver) added |
| F4 (MAJOR) | S0-HARD-STOP thresholds qualitative ("≈chance", "unchanged"), no seed-aggregation rule | §1.31.2: collapse pinned at `acc_A ≤ 0.09375`; unchanged pinned at `|Δacc| ≤ 2σ`, `σ=√(p̂(1-p̂)/n)` computed per arm (contender `2σ≈0.00099`, ablation `2σ≈0.00646` at n=4096 — see the disclosed discrepancy note there re: the dispatch prompt's `≈0.0096` figure); verdict-grade seed aggregation pinned (mean + disclosed per-seed table) |
| F5 (MAJOR) | prose supersession named only 3 of 8 actually-dead clauses | §1.31's opening: prose replaced with an 8-item enumerated dead-clause list (line-area ref + one-line reason each) |
| F6 (MAJOR) | claim sentence undersells (ties to a KV-CAPPED baseline round 4 doesn't test) and conflates round-4's claim with the sweep's K-cap/M* framing | §1.31.6: claim rewritten to "...cannot reach even with its full KV cache..."; K-cap/M* tie-in moved to the sweep's own claim; matched-budget caveat and seed-fragility disclosure (with the pinned n=3→9 extension trigger) added to the claim path |
| M1 | Leg-B anchor has no reproduction tolerance | §1.31.2: ±0.05 absolute on `rf@0.9` for the 7 reused cells; drift outside band → investigate before proceeding |
| M2 | "8/9" arithmetic wrong | fixed as part of F2's §1.31.7 rewrite (7+2=9) |
| M3 | M* walk has no rule for a joint contender+transformer failure at one `M` | §1.31.1 Axis-2 paragraph: joint-NO-RECALL rule added — TIE-equivalent, walk continues, never a LOSE from joint failure |
| M4 | K48 forced-fail negative test unspecified | §1.31.4 item 3: forced-fail spec added (unsliced-path-behind-flag OOM assertion, or the analytic peak-memory-bound alternative) |

**Note on the F4 numeric discrepancy, disclosed rather than silently
corrected:** the dispatch prompt for this amendment cited "≈0.0096 at
acc 0.999" for the S1-unchanged `2σ` bound. Computing `√(p̂(1-p̂)/n)`
honestly at `p̂=0.9990, n=4096` (as the item's own instruction requires)
gives `σ≈0.000494`, `2σ≈0.00099` — roughly 10× smaller than the cited
figure. §1.31.2 now pins the mathematically verified number, per this
project's standing exact-threshold/honest-computation rule; the
discrepancy is flagged for the coordinator here rather than silently
overwritten, since a raw-diff check might otherwise read the deviation
as an error rather than a correction.

**STATUS: Rev 5.1 complete; coordinator raw-diff verification per the
§2.16/§2.18 precedent is the final design gate; then build-fix (§1.31.4
items 1-6) → scoped audit → round 4.**

### 1.32 MICRO-ATTACK ON REV 5 (2026-07-09): NEEDS-REVISION — 1 FATAL-class pre-registration hole, 5 MAJOR; the two-leg direction itself is SOUND

Verified clean: Leg-A tier logic is a complete partition; INDETERMINATE
can't be gamed (pinned EVAL_SEED, one pre-registered n=3→9 extension);
the MHL walk survives the metric swap; the degenerate-baseline trigger is
pinned (0.94× was descriptive); price re-derives exactly (≈1.25-1.28
expected / 1.96 ≤ 2.0 ceiling); fix order coherent; Leg-B split hygiene
disjoint-seeded; §1.26a chain consistent.

**F1 (FATAL-class, binding):** contender_task2 rung1 = 0.0376 (1.2×
chance, band FAIL) and both K48-stress cells ≈ chance ON THE FROZEN
CHECKPOINTS — the dissociation exists ONLY in task1_K32; round 4
re-meters the same checkpoints, so task2's Leg-A band fails
DETERMINISTICALLY. Rev 5 records none of this AND leaves §1.4.1's
"Task 2, primary" designation un-superseded → the axis-1 decisive-cell
identity is ambiguous while one resolution is already settled by on-disk
data (the §1.29-precedent trap re-fired). FIX (Rev 5.1): record the full
round-3 rung-1 table (7 cells); pin the axis-1 aggregation rule with an
explicit supersession of the "Task 2, primary" sentence; pre-register
task2's round-4 disposition NOW (its band failure is knowable — name the
branch: task1-only sweep vs task2 diagnosis vs objective iteration).
**F2 (MAJOR):** the checkpoint-reuse record is factually wrong —
torch.save executes AFTER the OOM'd rung-2 fit, so round-3 K48-transformer
weights were NEVER persisted; the on-disk file is ROUND-2's two-term-
objective model (mtime 02:03 < launch 05:41). Correct plan: 7 reuse
(mtimes verified 05:49-08:43) + 2 fresh; STRIKE "reuse may be attempted";
mandate a per-cell identity table (md5+mtime, all 7) as round-4
pre-flight; the loader needs pinned provenance, not just versioned saves.
**F3 (MAJOR):** two build items missing from §1.31.4's binding list:
(5) re-wire check_gate1_full_cell_band (still enforces rf@0.9>0 — run
as-is it FATALs every cell) per §1.31.3's bands; (6) the round-4
re-metric driver itself (3-arm Leg-A continuation eval + Leg-B ridge
incl. the transformer arm + S₀-zeroing + both-direction controls).
**F4 (MAJOR):** pin the S₀-HARD-STOP numerically: collapse = S₀-zeroed
acc ≤ the 3×-chance bar; S₁-unchanged = within ±2 binomial σ at n=4096;
pin seed-aggregation at verdict grade. **F5 (MAJOR):** replace the prose
supersession with an ENUMERATED dead-clause list (adds: §1.7's
"only this rung feeds WIN/TIE/LOSE", §1.4.2's per-M gap statistic,
§1.4.1 task-primacy, §1.7 gate-7 scope, §1.3.1.2 tap pin). **F6
(MAJOR):** fix the claim sentence — round-4's transformer is UNCAPPED,
so the honest (and STRONGER) claim is "cannot reach even with its full
KV cache at matched params/tokens/budget"; the K-cap/M* tie-in belongs
to the sweep; add the matched-budget caveat and the seed-fragility
disclosure w/ extension trigger. Minors M1-M4 (Leg-B anchor status +
tolerance; 7+2 arithmetic; joint-NO-RECALL rule for the M* walk; K48
forced-fail spec). **Bookkeeping: the injection tally is RACING under
concurrent agents (79 vs ≈89 in parallel logs) — treat as ≥80;
single-pass reconciliation at the next STATE consolidation.**

**DISPOSITION: Rev 5.1 amendment dispatched (surgical, prescription
above); coordinator raw-diff verification per the §2.16/§2.18 precedent;
build-fix waits on 5.1.**

### 1.32a COORDINATOR RAW-DIFF VERIFICATION (2026-07-09): Rev 5.1 delta MATCHES the §1.32 prescription → REV 5 PRE-REGISTRATION FINAL; build-fix dispatched

Verified against git show 505fbe0 directly: F1 seven-cell table +
task1-primary pin + enumerated §1.4.1 supersession + task2 disposition
(joint-failure TIE, post-sweep diagnosis round); F2 7+2 correction +
identity-table pre-flight + loader provenance pinning; F3 items 5-6
added; F4 numeric HARD-STOP (collapse ≤0.09375; unchanged ≤2σ at the
HONESTLY-computed per-arm values 0.00099/0.00646 — the revision agent
correctly REFUSED the coordinator's own wrong ≈0.0096 figure and
disclosed the discrepancy, which is the discipline working); F5 8-item
dead-clause list; F6 the corrected-and-stronger claim sentence; M1-M4.
Single file, 258+/24−, §1.32 untouched. **The Rev-5 pre-registration is
FINAL. BUILD-FIX DISPATCHED (§1.31.4 items 1-6) → scoped audit →
round 4 (7 re-metric + 2 fresh, ≈1.3 GPU-h) → ladder → margin freeze →
task1-primary sweep.**

### 1.33 REV-5 BUILD-FIX RECORD (2026-07-09): §1.31.4 items 1-6 ALL LANDED — commit 7c7acd5; scoped audit DISPATCHED

All six items built + CPU-proven: (1) rung-2 = 107-way entity-identity
at the Leg-B tap via the new _leg_b_tap dispatcher reading the
pre-LM-head hidden from the SAME continuation call rung 1 uses
(bit-equivalence confirmed vs the taploc script); capacity null kept.
(2) Positive controls both directions: planted-identity tap reads
1.0000 under the relabel vs 0.0305 ≈ chance under the OLD slot labels
on the SAME plant — the relabel-matters proof; rung-1 plant
re-verified; Leg-B known-linear-recovery promoted to a standing per-run
control. (3) OOM fix: return_hidden skips the vocab matmul;
slice-before-matmul bit-close; M4 analytic gate at the real K=48 shape
(sliced 0.288 GiB < 2 GiB bound; unsliced 98.35 GiB > bound, 49× —
teeth without allocation). (4) Save BEFORE the rung-2 fit, filenames
_r{round}-suffixed, loader asserts manifest md5 (forced-crash proof:
new order survives, old order loses the checkpoint). (5) Band re-wire:
rf@0.9 conjunct REMOVED (not even a parameter); arm-aware blocking;
instrument-health blocking all arms; the round-3 contender case now
PASSES its band and the OLD formula FATALs it — both directions proven.
(6) h2h_round4_driver_rd.py (650 ln): 7+2 cell spec, provenance-checked
loads, Leg A + relabeled rung 2 + Leg B ridge incl. the transformer arm
+ S₀ hard-stop at the honestly-pinned thresholds; chain Stage B3 gated
behind ROUND4_AUTHORIZED.token (cannot auto-fire pre-audit). Selftests
64 green across six modules; one test-fixture bug found+fixed en route.
Box-only disclosed: real-kernel K=48 smoke; the actual re-metric +
2 fresh cells; FRESH_CELL_CONFIGS.json is coordinator-authored on box.
**NEXT: scoped worktree audit (re-run the four regression kills +
new-defect sweep on the delta) → deploy (md5) → identity-table
pre-flight → ROUND4_AUTHORIZED.token → round 4 (≈1.3 GPU-h).**

### 1.34 BUILD-FIX AUDIT VERDICT (2026-07-09, on 7c7acd5): NEEDS-FIXES — 1 BLOCKING + 2 MAJOR; all six suites green ×2, all four regression kills verified both directions

CLEAN: bit-EXACT tap equivalence (torch.equal, incl. vs the taploc
script); 107-way relabel round-trips to the right entity, fails closed
out-of-pool; ROUND4_CELL_SPEC == 7+2 cell-by-cell; provenance fails
closed at all three layers; S₀ pins computed-not-hardcoded
(0.09375 / 0.000988 / 0.006458); band re-wire teeth through the REAL
wiring (unhealthy-ablation blocks, chance-transformer data-only,
chance-contender blocks); token gate proven on the verbatim Stage-B3
block, 4 cases.
**F1 (BLOCKING): round 4 cannot run** — _round3_ckpt_filename matches
cell.get("K")==K but task2 calibration cells carry NO K key → KeyError
at pre-flight on both task2 cells (demonstrated live); driver selftest
3 masked it (FileNotFoundError on cell 1 in an empty dir). Fix: match K
only when present (or key on arch/task/role) + a selftest resolving ALL
7 reused names. **F2 (MAJOR):** task2's round-4 Leg-A read uses H_test
(3,4) but the §1.31.1 written baseline was measured on H_train (1,2) —
comparability broken (outcome-neutral, both ≈chance, but the
pre-registration diff is hop-set-incomparable). Fix: read H_train to
match the baseline (and ideally both, disclosed). **F3 (MAJOR):** the
§1.31.4-item-3 "wired as its own chain gate" real-kernel K=48 smoke is
NOT wired — no chain stage, no box-smoke item — and the coverage hole
is DEMONSTRATED: restoring the exact §1.27 OOM shape at the call site
leaves all suites green (selftest 21 is pure arithmetic decoupled from
the code under test). Fix: wire it as a box-smoke checklist item + a
chain check before Stage B3. Minor notes i-v carried (manifest
cross-check stays an explicit coordinator pre-flight; mtime recorded
md5 compared; the chain-level-only token gate; shared diagnostic seeds
disclosed; **coordinator decision on note v: round-4 fresh cells export
H2H_DIAL_ROUND=4** — no clobber either way, but the label must match
the round). Security: 3 more false file-modified fakes, md5-refuted.
**DISPOSITION: narrow fixes dispatched (F1/F2 one-line-class + F3
wiring + the note-v label) → scoped re-check → deploy checklist
(K=48 real-kernel smoke FIRST, identity-table pre-flight,
FRESH_CELL_CONFIGS.json, THEN the token).**

### 1.35 F1-F3 FIXES + COORDINATOR VERIFICATION (2026-07-09): delta matches the §1.34 prescription — ROUND-4 SOFTWARE COMPLETE; deploy chain dispatched

Commit 5107638, three files, 368+/23−, raw-diff verified: F1 K-match
guarded on key presence + selftest 9 resolving ALL 7 reused names
(revert reproduces the audit's KeyError); F2 primary hop-set = H_train
pinned by literal assert, H_test as a disclosed non-primary secondary in
cell JSONs + summary (selftest 10); F3 box-smoke item [11] = the one
executable checklist entry (real fit_rung2_identity_classifier at the
K=48 shape on CUDA, measured peak < 2 GiB, device-guarded never-PASS on
CPU, token BOX_SMOKE_ITEM_11_K48_REAL_KERNEL_PASSED) + Stage-B3
pre-flight refuses without it (extracted-block proof: FATAL without,
proceeds with); note v H2H_DIAL_ROUND=4 scoped to the driver invocation;
minor iv explicit seed pins. All suites green (driver 10/10, checklist
10/10, cell_train 22/22, wrappers 9/9, sweep/fanout/bands 5/5/5/7).
Kill proofs run by the fixes agent and verified in the delta — per the
§2.16/§2.18 precedent no further audit round is needed for
one-line-class F1/F2; F3's wiring is proven by the extracted-block
token-gate test. **DEPLOY CHAIN DISPATCHED: md5 deploy → box item [11]
K=48 real-kernel smoke FIRST → identity-table pre-flight (7 ckpts,
md5+mtime vs the recorded manifest) → coordinator-authored
FRESH_CELL_CONFIGS.json → ROUND4_AUTHORIZED.token → ROUND 4
(≈1.3 GPU-h, task1-primary per Rev 5.1).**

### 1.36[h2h] ITEM-11 DIAGNOSIS+FIX (2026-07-09): 6.14 GiB traced to transformer_native_tap's own forward activations, NOT the LM head — row-chunking FIXED it to 1.05 GiB; deploy chain RESUMED, round 4 launched

**Diagnosis.** The deploy halt (commit `d8f764b`, `experiment-runs/2026-07-09_h2h_round4_launch/`)
measured `fit_rung2_identity_classifier`'s real-kernel peak at 6.14 GiB (3.07x the 2 GiB bound),
reproducible x2. On-box `torch.cuda` memory-stat instrumentation (checkpointed sub-step by
sub-step through `transformer_native_tap`'s internals, K=48 real shape: B=32 episodes, Q=48
full-K eval queries, B*Q=1536 "mega-batch" rows, T_total=342) traced the cost: the LM-head
matmul is confirmed NOT the driver (`return_hidden=True` already skips it, analytically
0.29 GiB — §1.33/§1.35 correctly measured THIS component, then incorrectly cited it as if it
covered the whole call, off by ~21x). `F.scaled_dot_product_attention` is also NOT the driver
(dispatches to a memory-efficient/flash backend on this box, no O(T²) score matrix
materialized). The real driver is the Transformer's OWN forward activations over all 1536 rows
processed in one Python-level batch: the FFN's (B*Q,T,4·d_model) GELU intermediate (~2 GiB
tensor, 2 non-fused copies alive simultaneously = ~4-4.5 GiB spike per layer,
`lm_pretrain_rd.FFN.forward` = `fc2(gelu(fc1(x)))`, no dropout) and RoPE's per-head elementwise
intermediates (~2 GiB spike), x2 layers — a genuine, shape-driven cost of the B*Q mega-batch
design, not a bug in what gets computed.

**Resolution taken: FIX (path a), not re-pin.** More than half the allocation is avoidable:
`fit_rung2_identity_classifier` runs `model.eval()` before its loop (no dropout anywhere in this
model), RMSNorm normalizes over `d_model` only (no batch statistics), and SDPA never mixes
across the batch dimension — every one of the B*Q rows is numerically INDEPENDENT of every other
row. `probe_head_rd.transformer_native_tap` now row-chunks the B*Q dimension at
`TRANSFORMER_TAP_MAX_ROWS_PER_CHUNK=128` (sized from the measured ~4.09 MiB/row scaling,
6.14 GiB / 1536 rows), concatenating chunk outputs — bit-identical to the unchunked computation
by construction, proven (not merely asserted) by new `smoke_11` (`torch.equal`, both
uncapped/capped mask modes, chunk boundary genuinely exercised, both single-shot and >=6-chunk
branches forced). `h2h_cell_train_rd._fused_transformer_tap_and_answer_logits` (the
TRAINING-time twin, same B*Q-mega-batch structure) is DISCLOSED but deliberately left unfixed —
it stays safe today only because `N_QUERY_TRAIN=8` is fixed regardless of K (B*Q=256, predicted
peak ~1.0-1.1 GiB), never because it was patched; a comment there points to the same fix pattern
if that assumption ever changes (training needs gradient accumulation across chunks, not the
no_grad-only chunking used here). Selftest 21 (`h2h_cell_train_rd.py`, the M4 analytic
forced-fail) had its own docstring corrected — it is honestly rescoped to bound ONLY the
LM-head-slice component (still true, still useful), no longer implying it bounds the whole
real-kernel call; item 11 remains the sole real-kernel authority on total peak.

**Files changed (commit pending):** `probe_head_rd.py` (chunking fix + `smoke_11`, now 11/11),
`h2h_cell_train_rd.py` (selftest 21 docstring correction + disclosure comment on the training
twin, still 22/22). `h2h_box_smoke_checklist.py` UNCHANGED (`ITEM_11_BOUND_BYTES=2 GiB` stays —
the fix now genuinely clears it, no re-pin needed). CPU suites green locally (Mac, stub) and
on-box (`REASONING_LINK_FORCE_CPU_STUB=1`, since real fla's RMSNorm has no CPU fallback).
Deployed via scp + md5-verified EXACT local==box on both changed files.

**Item-11 re-run (real CUDA, GPU 0, K=48 production shape), x2 for reproducibility:**
`{'status': 'ran_on_cuda', 'peak_bytes': 1126852096, 'bound_bytes': 2147483648, 'passed': True}`
— **1.0496 GiB, deterministic both runs, 1.9x headroom under the 2 GiB bound** (down from
6.14 GiB, a 5.83x reduction). Token `BOX_SMOKE_ITEM_11_K48_REAL_KERNEL_PASSED.token` written to
`/home/nvidia/chapter2/gates/h2h_round4/`.

**Amendment, pre-registered before resuming, per the dispatch brief's own instruction:** the
2 GiB bound exists to block the §1.27 ~98 GiB OOM class, not a well-understood ~1-6 GiB range —
no re-pin was needed or performed; the bound stands at 2 GiB, now genuinely met (not
mis-measured) by the fixed code.

**Identity-table pre-flight (resumed deploy chain step 3), verified before authorizing round 4:**
the 7 reused `_auxrev2`-suffixed checkpoints' on-disk mtimes (UTC) were cross-checked against
this design doc's own §1.31.7-recorded values — all match essentially to the second:

| cell | recorded mtime (§1.31.7) | on-disk mtime | md5 |
|---|---|---|---|
| contender_task1_calib (K32) | 05:58:16Z | 05:58:16.394Z | `2bd6dbce2f4187f54851f97c0ea5e57e` |
| ablation_task1_calib (K32) | 06:24:40Z | 06:24:40.530Z | `6fa7b6b074252a5b3bb225437cce1689` |
| transformer_task1_calib (K32) | 07:13:39Z | 07:13:39.236Z | `647de170551253031756c0c4b4040f06`|
| contender_task1_stress_K48 | in 05:49-08:43Z | 05:49:50.830Z | `6714c5141f8c2baa70a6b77b83deab42` |
| ablation_task1_stress_K48 | in 05:49-08:43Z | 06:07:00.946Z | `a5f8add0c2e134265ad5ebcfe77aa86e` |
| contender_task2_calib | in 05:49-08:43Z | 08:02:20.993Z | `e144f91b834cbe218354aa45e3eb094b` |
| ablation_task2_calib | in 05:49-08:43Z | 08:43:22.308Z | `b980097587570799b794f7e8effb3702` |

(An initial pass checked the PLAIN, non-`_auxrev2` filenames and found mtimes ~5-6h earlier than
recorded — a false alarm, resolved by reading `h2h_cell_train_rd.calibration_cells()` directly:
`AUX_REV_SUFFIX="_auxrev2"` is appended for every non-task3 cell name, so `_round3_ckpt_filename`
never resolves to the plain names; those are leftover pre-`_auxrev2` artifacts, not what round 4
loads. Recorded here per this project's own "read the raw artifact, don't average two
contradictory claims" precedent, since the false alarm could otherwise resurface.) The 2 fresh
cells' on-disk files (`transformer_task2_calib_primary_auxrev2.pt` mtime 02:48Z,
`transformer_task1_calib_stress_locate_only_K48_auxrev2.pt` mtime 02:03Z — the LATTER an EXACT
match to §1.31.7's own cited stale-K48-transformer mtime "02:03") are correctly excluded from
reuse by `ROUND4_CELL_SPEC`'s `fresh=True` flag (`build_provenance_manifest` skips them by
construction) — they will train fresh under `H2H_DIAL_ROUND=4`, saving to newly-versioned
`_r4.pt`-suffixed files, no clobber of the stale files or of each other.

**FRESH_CELL_CONFIGS.json** generated programmatically from `h2h_cell_train_rd.calibration_cells()`
(never hand-transcribed — matched by (arch,task,role,K) against the real 13-cell manifest), at
`results/h2h_rung1/round4/FRESH_CELL_CONFIGS.json`: `transformer_task2_calib`
(role=primary, budget_frac=1.0, lr=3e-4) and `transformer_task1_stress_K48`
(role=stress_locate_only, K=48, budget_frac=0.25, lr=3e-4) — matching the dispatch brief's exact
spec.

**ROUND4_AUTHORIZED.token** written to `/home/nvidia/chapter2/gates/h2h_round4/`, citing this
section and the item-11 pass result verbatim.

**LAUNCHED:** `h2h_round4_driver_rd.py --run-all` (H2H_DIAL_ROUND=4), tmux session `h2h_round4`,
GPU 0 (idle at dispatch; GPUs 2-7 all busy with named fixscale_392m/98m post-processing waves +
`h2h_decisive`, none touched), `--ckpt-dir /data/h2h_rung1_ckpts --out-dir
results/h2h_rung1/round4` (matches `h2h_rung1_chain.sh`'s own Stage-B3 path convention exactly).
First cells confirmed healthy post-launch (see MANIFEST for the live tail). Harvest watch-path:
`results/h2h_rung1/round4/ROUND4_SUMMARY.json` (9 cell entries expected;
`ROUND4_PROVENANCE_MANIFEST.json` and per-cell `{cell_id}_round4.json` alongside it).
