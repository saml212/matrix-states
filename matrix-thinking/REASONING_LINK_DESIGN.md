# REASONING-LINK: does key-geometry stabilization causally improve in-context multi-hop composition? (Rev 6 (2026-07-07) — post-attack-6, CLEARED-FOR-BUILD)

**Status: DESIGN CLEARED-FOR-BUILD. Attack-round-6 (fresh eyes,
2026-07-07) returned CLEARED-WITH-MINORS — the FIRST round with zero
FATALs after five consecutive FATAL-bearing rounds (seven fatal axes
found and fixed across rounds 1-5; every prior round's fixes re-verified
HOLD by later rounds). Rev 6 (orchestrator-applied) resolves round 6's
three findings: the MAJOR scope gap (premise/floor validity gates were
certified at one 2-layer/d_state=64 point and silently applied to
12/16/22-layer, d_state=128 rungs — now measured PER RUNG from the same
already-captured tensors, each rung's confirmatory eligibility gated on
its own first-pass premise read, zero new GPU passes, §4.4/§8.4), the
Stage-0 checkpoint-identity conflation (pinned to the Leg-A control-arm
checkpoint, §9), and an orphaned editing fragment (deleted). Attack
records: §13.1-§13.5. Still not built, no GPU spent — build + independent
code audit are the remaining gates before launch.** **Rev 5 headline:** attack-round-5 found that every
`q_eff` extraction in this design, through Rev 4, hooked the WRONG
projection. §4.4 said `q_eff` is captured "via the same `k_conv1d` hook at
the query's own final position" — but this checkpoint family (unlike the
from-scratch harness that convention was inherited from, `model_rd.py`,
which has NO `q_proj` at all) has a real, separately-trained
`q_proj`/`q_conv1d` path, and `chunk_delta_rule` is called with `q=q_bf` —
the Q-path's own output — not `k_bf` (`lm_pretrain_rd.py` L45-49 header
comment, L787-793 the three untied projections/convs, L837-839 the
per-token conv calls, L983 the kernel call). Substituting `W_k` for `W_q`
at the query position was an unexamined carry-over of the from-scratch
harness's own forced convention (that harness has no `q_conv1d` to hook —
this one does), never an intentional design choice. Forward-B's
query-position extraction is retargeted to a new `q_conv1d` hook (§4.4,
§13.5) — a ~5-line, same-forward-pass, zero-new-GPU-cost fix, mirroring
Rev 4's own `v_conv1d` hook exactly; every `k_conv1d` mention describing
`q_eff` extraction (§4.4, §7.9) is corrected, while every mention
describing bind-phase `k_eff` extraction (unaffected) is left unchanged.
This retargeting **strengthens, rather than merely corrects, premise
(iii)** (bind↔query alignment, `DELTANET_REALDATA_DESIGN.md` §5.2):
`cos(q_eff_a, k_eff_a)` is now a genuine comparison of two
independently-trained projections of the same entity, not a
near-tautological reuse of one weight matrix (`W_k`) applied twice — Rev 5
gives premise (iii) the same measured, Stage-0-calibrated treatment
premise (iv) already had, and replaces both premises' reused, unjustified
"0.9 alignment bar" with one shared, mechanical, null-relative action-rule
table (§4.4) that pre-registers h=1's and h≥2's confirmatory status
*before* any grid launches. Rev 5 also adds an absolute chance-plus-margin
backstop to the h=1 sanity floor (§8.4), a registered OOM fallback for
Stage 0.5's rung-3 calibration cell (§9), and two reporting-precedence
rules to §12 (a Leg-A READOUT-DIVERGENCE verdict and a Leg-B outcome may
coexist, reported side by side, neither overriding the other;
READOUT-FORM-INVALID takes precedence over AMBIGUOUS when both gates would
otherwise fire, being the more specific diagnosis). Every fix is
hook/doc-level — §10's budget is confirmed unchanged (≈24.20/25 GPU-h,
≈0.80 GPU-h margin). Rev 4's own fixes (the `v_eff` revert, the
FATAL-ADJACENT Leg-B agreement gate, READOUT-FORM-INVALID) all HOLD,
independently re-verified this round, including the gather indexing
(re-derived by hand, correct) — see §13.5.

**One-paragraph summary.** This project has three closed or near-closed
results sitting unconnected: (1) a located, real capacity cliff at
`K/d_state≈0.55` for `d_state=64`, now attributed to absolute state capacity
rather than key-population coherence (`KEY_ANCHORING_DESIGN.md` §12.9,
§14.12/§14.13); (2) a frozen key-bias intervention that provably moves
write-geometry in a from-scratch 14M-param LM — global-vector bias
*stabilizes* (`span_frac` falls, effective/stable rank rises), per-token bias
*destabilizes* (`span_frac` rises, rank falls) — at zero val-loss cost, with a
mechanism-wave rank-collapse corroboration already run
(`FROZEN_BIAS_LM_DESIGN.md` VERDICT + §12.10); and (3) a clean, mix-controlled,
pure-scale attractor ladder (`span_frac` 0.248→0.344→0.389 at 14M/98M/392M,
`d_state` stepping 64→64→128, rung-3 at 1.31B finishing ≈2026-07-08,
`SCALE_TRANSFER_DESIGN.md` §5.9/§5.10/Addendum). None of these three results
has ever been connected to whether the model can actually *do* anything
harder than store one-hop bindings. REASONING-LINK is the keystone experiment
that connects them: it takes the **already-trained, already-archived**
checkpoints from (2) and (3) — zero new training — and asks whether the
measured geometry differences causally track **in-context multi-hop
relational composition**, using an eval-only probe built by adapting this
project's own validated BIND/QUERY-as-token-sequences harness
(`matrix-thinking/deltanet_rd/grammar_rd.py`, already used to build the
n=107-entity table underlying the capacity-cliff work) to zero-shot,
no-fine-tuning presentation. Two legs run in parallel on existing
checkpoints (Leg A: does the frozen-bias intervention improve composition;
Leg B: does composition rise with scale and track the span_frac ladder), a
shared eval-only instrument, an estimated Phase-1 budget of **≈24.20 GPU-h**
(Rev 3 — re-derived from Rev 2's ≈24.95 GPU-h after attack-round-3 found two
new FATALs in the shared instrument itself, §4.4: a dimension mismatch that
forced dropping the trained probe entirely in favor of scoring directly in
`d_state` space against the target entity's own effective key [freeing the
probe-training and heldout-pool-control budget, ≈3.89 GPU-h, §4.5], and a
query-phase extraction gap that forced a two-forward protocol for
multi-layer checkpoints [doubling the per-pass cost, 4×→8× anchor
multiplier]. The freed probe budget covers only a fraction of the
two-forward tax — Rev 2's grid at the doubled rate would have landed
≈42.1 GPU-h, ≈17.1 over the unchanged 25 GPU-h ceiling — so the committed
grid is re-scoped per-leg: Leg A keeps its structurally-required two-point
K-contrast `{20,32}`, Leg B narrows to its own single primary near-cliff K
per `d_state` (`K=32` at d=64, `K=64` at d=128 — the only K's its headline
ever used), demoting `K=48`/`K=40`/Leg-B `K=20` to named extensions —
landing under the 25 GPU-h ceiling with margin ≈0.80 GPU-h, reconciled
old→new row-by-row in §10), and a gated, sketched-only Phase 2
(new training, standard benchmark) if Leg A shows a real effect.
**Rev 4 addendum (attack-round-4, §13.4): the ≈24.20 GPU-h Phase-1 total
above is UNCHANGED** — the FATAL fix reverts the scoring target from
`k_eff` back to `v_eff` (a same-space, `d_state`-dimensional swap inside an
already-priced forward pass) and adds one new forward hook capturing an
already-existing submodule's output alongside the existing `k_conv1d`
hook, neither of which changes the number or shape of GPU passes §10
prices; §10 now carries an explicit Rev 4 confirmation line rather than
leaving this to be re-derived by inspection.

**Rev 5 addendum (attack-round-5, §13.5): the ≈24.20 GPU-h Phase-1 total
above is, again, UNCHANGED.** The FATAL fix retargets forward-B's
query-position hook from `k_conv1d` to a new `q_conv1d` hook — a same-pass,
zero-new-GPU-cost addition exactly like Rev 4's own `v_conv1d` hook,
firing on the already-priced forward-B call, adding no new mixer forward
and no measurable wall-clock delta. Every other Rev 5 fix (the premise
action-rule table, the h1 absolute backstop, the Stage 0.5 OOM fallback,
the two new §12 precedence rules) is either a Stage −1/Stage 0
measurement already priced inside the existing calibration cells, or a
pure interpretation/reporting rule with no GPU cost at all. §10 carries an
explicit Rev 5 confirmation line rather than leaving this to be re-derived
by inspection.

---

## 0. Reading list this design builds on (context, not repeated here)

- `STATE.md` — "Chapter 2 — STATUS" (rank-necessity and composition results;
  Task D/E; DeltaNet real-data causal-rank chain) and "Path Forward"
  (frozen-bias LM outcome, capacity trilogy, scale-transfer ladder,
  everything currently running/closed as of 2026-07-07).
- `matrix-thinking/FROZEN_BIAS_LM_DESIGN.md` — VERDICT (rung-1, the fourth
  outcome / sim-training-divergence classification, the control contrast)
  and §12 (mechanism wave: H2 rank-collapse corroborated §12.10.2, H4
  compensatory-drift consistent §12.10.3). Checkpoints:
  `/data/deltanet_rd_frozenbias_ckpts/` on the Brev box, 3 arms
  (`off`/`per_token`/`global`, λ≈0.58) × 2 corpora (`openr1-mix-ext`,
  `wikitext-mix-ext`) × 3 seeds × 400 archived `.pt` checkpoints (every
  1,000 of 20,000 steps, verified present 2026-07-07). Architecture:
  `d_model=256, d_state=64, n_layers=2, conv_size=4, num_heads=1`,
  14,048,896 params (identical to Wave C / the exactness program's own
  cell).
- `matrix-thinking/SCALE_TRANSFER_DESIGN.md` — Track C ladder design
  (§5.3 rung configs), §5.9/§5.10/Addendum (results). Checkpoints:
  `/data/lm_rd_trackc_ckpts/{mixcontrol,wave1ext,wave2,wave3}/` (naming
  per-wave, not uniform), staying on-box, never archived to the repo.
  Rungs: 14M mixcontrol (`d_model=256,n_layers=2,d_state=64`, 2 corpora × 3
  seeds), 98M wave-1ext (`d_model=768,n_layers=12,d_state=64`, 2×3), 392M
  rung-2 (`d_model=1536,n_layers=16,d_state=128`, 2×3), 1.31B rung-3
  (`d_model=2560,n_layers=22,d_state=128`, **2 runs only — PINNED at Rev 3
  as 2 corpora × 1 seed from the launch manifest itself, §6.1** —
  token-matched to rung-2 at 1.5B tok/run, ETA ≈2026-07-08, not yet
  harvested at draft time). **d_state steps 64→64→128→128 across the ladder
  — not constant** — load-bearing for §6's K-sweep stratification below.
- `CLAUDE.md` "Hard Rules" — every anti-shortcut rule is binding on this
  design: exact continuous recovery via cosine, never argmax/codebook
  nearest-neighbor, when a rank-necessity-flavored claim rides on it; single
  full K-cycle permutations, stratified by effective distance; the
  position-decomposition escape must be closed by construction (a hard
  single-state bottleneck) or named as an open risk; never compress
  matrices to vectors; same dataset/instrument across arms in one
  comparison; calibration before sweep; multiple independent audit rounds.
- `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md` — Task E (the
  bespoke matrix-native composition test): single-Hamiltonian-K-cycle
  generator, pinned `pred(a,h) = Z^h · key` readout, `Z_ideal` ceiling,
  cosine>τ scoring, the C6 injectivity control, the eigenstructure-fidelity
  (C8) motivation from Grazzi et al. (ICLR 2025) / DeltaProduct. This
  design's probe reuses the *readout formula* and *generator discipline*
  verbatim, adapted to a production DeltaNet kernel and a general-purpose
  pretrained checkpoint (§4 below).
- `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md` §16 — the F-geo-3
  fix wave (K=16 bar HIT 3/3, K=32 narrowly missed, attributed to
  cross-episode drift) and `matrix-thinking/DELTANET_REALDATA_DESIGN.md`
  §5.2/§16 — the production-kernel realization of BIND/QUERY as ordinary
  GPT-2-BPE token sequences (`grammar_rd.py`), the graded K-exactness
  frontier at fixed `d=64` (K=8/16/24/32), and `recovered_frac@0.9` /
  `M3_held_out` / `effective_hop` conventions this design inherits.
  **Critical difference this design must carry throughout:** that harness
  trains a model FROM SCRATCH on the BIND/QUERY grammar with a hard,
  externally-computed β-mask (`β=1` only at K value-token positions, 0
  elsewhere) and reserved, zero-pinned buffer/query token IDs beyond GPT-2's
  50,257-token vocabulary. The frozen-bias-LM and scale-transfer checkpoints
  this design evaluates were pretrained on ordinary running text
  (OpenR1-Math-mix / WikiText-mix) via `lm_pretrain_rd.py`, with **no**
  BIND/QUERY grammar ever in their training data, **no** hard β-mask (β is
  the model's own learned, per-token `b_proj` output), and a standard
  50,257-row embedding table with **no reserved token IDs**. §4.1 below
  registers the exact build-time adaptation this gap requires.
- `matrix-thinking/KEY_ANCHORING_DESIGN.md` §12 (capacity-cliff
  localization: sigmoid fit `x0=0.5455`, 95% CI `[0.5385,0.5513]`,
  `w=0.0597`, at `d_state=64`), §13 (cliff-universality at `d=128`: **no
  cliff in the same K/d window**, `h4=1.0` at all 12 cells, CIs disjoint
  from d=64's), §14.12/§14.13 (coherence-dose-response: `h4=1.0` at every
  dose 0.130/0.284/0.40 at BOTH rank-4 and diffuse injection structures —
  coherence fully exonerated as the driver). **Read together, these three
  results say the K/d≈0.55 cliff at d=64 is an absolute-capacity fact about
  d=64 specifically — the same K/d ratio does not reproduce it at d=128,
  and directly injecting anchor-table coherence at or above d=64's own band
  does not reproduce it either.** This is load-bearing for §5.3/§6.1 below:
  the K sweep must be calibrated per-`d_state`, not held at a fixed K/d
  fraction and assumed to transfer.
- `grammar_rd.py`, `key_anchoring.py`, `lm_pretrain_rd.py`,
  `lm_attractor_probe_rd.py`, `frozen_bias_retrofit_eval_rd.py`
  (`matrix-thinking/deltanet_rd/`) — read directly this session, not
  assumed. `build_entity_pools()` (in `grammar_rd.py`) already produces a
  GPT-2-tokenizer-verified, single-token entity/relation vocabulary
  (`train_name_ids`/`heldout_name_ids`, `heldout_frac=0.5`) — the **same**
  n=107 pool `KEY_ANCHORING_DESIGN.md` uses for its anchor table.
  `_permutation_graph()` is the verbatim single-Hamiltonian-K-cycle
  generator from `task_dn.py`/Task E. `lm_attractor_probe_rd.py`'s
  `capture_raw_keys`-style hooks are the existing, non-invasive mechanism
  for reading `k_eff` (post-conv, pre-kernel) off these exact checkpoints —
  the same hook the frozen-bias program's co-primary (`k_raw`) observable
  already uses; **verified this revision (Rev 3, load-bearing for §4.4's
  FATAL-1 fix): `capture_raw_keys` hooks `k_conv1d`'s output ONLY — no
  `v_conv1d` hook exists anywhere in this codebase**, and both that tool
  and `frozen_bias_retrofit_eval_rd.py` run exactly ONE hooked mixer
  forward per batch with a shared `--batch-size` default of **16**
  (load-bearing for §10's Rev 3 two-forward multiplier and §4.2's pinned
  batch size). **Verified this revision (Rev 5, load-bearing for §4.4's
  FATAL fix): no `capture_raw_queries` function, and no `q_conv1d` hook of
  any kind, exists anywhere in this codebase** — `capture_raw_keys` is the
  ONLY hook-style extractor `lm_attractor_probe_rd.py` defines, mirroring
  the already-verified `v_conv1d` absence (Rev 4, above) for the query
  path as well. This matters because `lm_pretrain_rd.py`'s own header
  comment states this LM block, unlike `model_rd.py`'s Wave-1 harness
  (external pinned readout only, **NO `q_proj` at all** — "the kernel's
  own per-token output is architecturally irrelevant there"), **"DOES use
  a real, learned q_proj/q_conv1d: an LM needs `chunk_delta_rule`'s own
  per-token output `o_t` at EVERY position for next-token prediction, not
  just the final recurrent state"** (L45-49, quoted verbatim) — realized
  as three UNTIED projections/convs, `self.q_proj`/`self.k_proj`/`self.v_proj`
  and `self.q_conv1d`/`self.k_conv1d`/`self.v_conv1d` (L787-793), each
  called on every forward pass (`q, _ = self.q_conv1d(self.q_proj(x))` etc.,
  L837-839), with `chunk_delta_rule` invoked as `chunk_delta_rule(q=q_bf,
  k=k_bf, v=v_bf, ...)` (L983) — the model's own read-through at any
  position runs on `q_bf` (the Q-PATH's output), never `k_bf`. Building a
  `q_conv1d` hook (mirroring the already-existing `k_conv1d`/`v_conv1d`
  hook pattern) is this revision's own FATAL fix, not a pre-existing tool
  this design merely reuses (§4.4, §13.5). `DeltaNetLM.forward(token_ids, initial_states=...)`
  (`lm_pretrain_rd.py`) accepts/returns per-layer recurrent state, so a
  BIND-phase forward pass can be truncated and its final state captured
  without modifying the architecture — but its mixer hard-asserts
  `T ≥ _MIN_KERNEL_T=128` BEFORE the k/q/v convs (L831-836), and
  `model_rd.py::effective_key_window`'s short-window direct-submodule
  extraction is a SINGLE-mixer precedent that does NOT generalize to these
  multi-layer checkpoints (Rev 3, load-bearing for §4.4's FATAL-2 fix).

---

## 1. Core hypothesis (falsifiable, restated precisely)

> **H_LINK.** The key-geometry stabilization the frozen-bias intervention
> produces (global-vector arm) — and, separately, the geometry shift scale
> itself produces along the span_frac attractor ladder — causally improves
> in-context relational binding and multi-hop composition in real
> DeltaNet-family language models, specifically in the load regime the
> capacity law identifies (`K/d_state` at and below the located d=64 cliff,
> `x0≈0.5455`), and the improvement (where present) grows with model scale.

Two independent sub-claims, each separately falsifiable, sharing one probe
instrument:

- **H_LINK-A (intervention → reasoning).** Among checkpoints matched on
  architecture, corpus, and val-loss (already established equal,
  `FROZEN_BIAS_LM_DESIGN.md` §7.2), the global-vector (stabilizing) arm
  recovers held-out-hop composition at least as well as the off (control)
  arm, and the per-token (destabilizing) arm recovers it no better than
  control — with the largest arm separation concentrated near
  `K/d_state≈0.55`, not uniformly across all K.
- **H_LINK-B (scale → reasoning).** Composition recovery, measured with the
  identical probe and readout, rises monotonically across the Track C
  ladder (14M→98M→392M→1.31B) and its rise correlates with the already-measured
  span_frac ladder (0.248→0.344→0.389→[pending]).

**What would make this uninterpretable rather than a clean confirm/refute
(pre-registered now, before any data exists, per the project's own
`FROZEN_BIAS_LM_DESIGN.md` §1.3 "fourth outcome" precedent):** if the h=1
sanity floor (§8.4) fails on the "off"/control arm itself, the entire probe
is broken for this checkpoint family and no arm/scale comparison built on
top of it is meaningful — reported as **probe-invalid**, not folded into
either CONFIRM or REFUTE.

**What H_LINK does NOT require Option 1 to prove in absolute terms (Rev 4,
attack-round-4 MAJOR framing adjudication, full derivation and disclosure
at §4.4; restated here because it bears on how H_LINK-A/B themselves should
be read, not just on the instrument).** Option 1 (`pred(a,h)=S_T^h@q_eff`,
one layer's own recurrent state self-applied `h` times) tests ONE SPECIFIC
mechanism — single-layer state self-iteration — that is an a priori
UNLIKELY implementation strategy for a multi-layer, general-purpose
pretrained LM never trained on this grammar: the standard prior for how a
multi-layer Transformer/hybrid model chains multi-hop information is
CROSS-LAYER hand-off (layer `i` resolves hop 1, a later layer resolves hop
2), not one layer repeatedly re-applying its own fixed state. H_LINK-A/B
are therefore read, throughout this document, as claims about **arm and
scale CONTRASTS** measured through Option 1 (and cross-checked against
Option 2, §4.4), never as claims that Option 1's absolute recovery level at
h≥2 is itself a direct measure of "how much composition this checkpoint
can do." A contrast can be genuinely informative (the intervention or
scale changes how much signal routes through single-layer self-iteration
specifically) even when the absolute level is compressed toward floor by
testing the wrong mechanistic hypothesis — which is also why a uniform
floor at h≥2 across every arm/rung is registered as its own outcome
(READOUT-FORM-INVALID, §12) rather than silently read as REFUTE.

---

## 2. Why this design, why now (the keystone framing)

Three closed or near-closed programs in this project have each, independently,
measured a *geometric* quantity (write-geometry `span_frac`, effective/stable
rank of the key population, an absolute-capacity cliff in `K/d_state`) on
DeltaNet-family checkpoints, without ever asking whether that geometry
predicts anything a downstream reasoning task would care about. This is
explicitly named as an open gap in two design docs already:
`FROZEN_BIAS_LM_DESIGN.md`'s "What this does NOT show" (VERDICT section:
"`span_frac` has no established behavioral correlate in an LM ... A
positive (or negative) shift in this observable has never been shown to
track next-token prediction quality, downstream task performance, or any
other user-legible capability") and `SCALE_TRANSFER_DESIGN.md` §5.7/§5.9
("Geometry-leg-only; no compositional-recovery cross-check at any rung").
REASONING-LINK is that cross-check, run against **both** open threads at
once, at **zero new training cost**, because the exact checkpoints needed
already exist on the box.

This is also the first time this project asks a reasoning question about
checkpoints that were **never trained on the reasoning task at all** — every
prior composition result (Task D/E, DeltaNet-causal-rank, DeltaNet real-data
Wave A/B, F-geo-3) trained a model FROM SCRATCH on the exact BIND/QUERY
grammar being tested. This design is closer to true zero-shot in-context
generalization, and is honestly harder to guarantee will show anything at
all — see §4.1 and §7 throughout.

**This design supersedes/absorbs a previously-registered, never-run in-house
attempt at the same question.** `SCALE_TRANSFER_DESIGN.md` §5.5 item 2 (the
"frontier-probe transplant") already proposed splicing the from-scratch
synthetic grammar's K-cycle probe task onto a pretrained LM's own embedding
table and backbone, gated on a rung-1 validation pass before trusting a
rung-3 number. Per that design's own §5.9/§5.10 harvest notes, "§5.5 items 2
(frontier-probe transplant) and 3 (fix-effect) were NOT run" at either rung
— it was scoped, budgeted, and then dropped by cut order (§8) without ever
executing. REASONING-LINK is a materially different, more carefully adapted
realization of that same underlying idea: where the transplant design left
the embedding-table/vocabulary mismatch and the readout formula unspecified
("no existing evidence its single-token 'entity' rows behave anything like
the synthetic grammar's clean, purpose-built vocabulary"), this design
registers a concrete build-time adaptation (§4.1: ordinary period/rare-token
buffer and query markers, no reserved IDs, no vocabulary resize) and reuses
the already-validated `pred(a,h)=S_T^h·q_eff` formula (Task E /
`DELTANET_REALDATA_DESIGN.md` §5.1) rather than leaving the readout
unspecified. Cited here explicitly so a reviewer does not discover the
overlap first; the never-run transplant is superseded, not silently ignored.

### 2.1 External literature positioning (Rev 1 addition, folds in
`research/reasoning-link-litreview-2026-07-07.md`)

A dedicated research-agent literature pass (2026-07-07, verdict
**GO-WITH-REFRAME**, no outright scoop found) requires four things to be
stated explicitly rather than discovered by a reviewer:

**Novelty claim, precisely positioned.** The novel contribution here is the
**specific combination** — a K-cycle hop-recovery probe transplanted onto
already-trained, general-purpose checkpoints, plus a controlled 14M→1.31B
fixed-state scale ladder with a param-matched, mix-controlled ladder
structure, plus a causal (not merely correlational) frozen-key intervention
angle, plus the capacity-law load axis (`K/d_state`) calibrated per
`d_state` off the already-located d=64 cliff — **not** the bare claims "a
capacity cliff exists" (Based, arXiv:2402.18668, already establishes
Ω(N)-bits-of-state necessity and runs its own param-matched ladder — the
ladder+control PATTERN is theirs, cited as methodology precedent, not
reproduced as if new) or "a scale ladder with an attention control shows
something" (a standard 2026 pattern on Transformer-family looped/hop-depth
generalization work, e.g. arXiv:2601.21214/2604.07822/2603.21676/2605.26789
— none of which, per the literature pass, has been run on a DeltaNet/Mamba-
family fixed-state model; that specific combination remains open). This
design's claim to novelty rises or falls on the combination, and is stated
that way throughout, not as "a cliff exists" or "a ladder with a control."

**Okpekpe & Orvieto (arXiv:2508.19029) — the credible rival causal account,
engaged head-on.** Their central claim: "Transformers differ from SSMs not
in expressive power but in their optimization dynamics" — Mamba/Hyena MQAR
failures are gated by a narrow learning-rate window, not a hard capacity
ceiling; with a fine LR grid, Mamba solves MQAR at sequence lengths far
beyond its hidden size. A hostile reviewer can read ANY capacity-cliff-
shaped result in this program (the located d=64 cliff, Leg A's killer
prediction, Leg B's ladder) as an optimization artifact rather than a real
capacity fact. **Rebuttal, stated explicitly rather than left implicit:**
this program's causal chain does not run through a learning-rate or
optimization-hyperparameter axis at all — it runs through **forced-rank
construction and eval-time SVD truncation** (`DELTANET_RD_EXACTNESS_DESIGN.md`,
`FROZEN_BIAS_LM_DESIGN.md`'s rank-collapse corroboration §12.10.2), a
post-hoc STRUCTURAL manipulation of an already-trained matrix, landing the
recovery ceiling exactly at `k=K` regardless of how that checkpoint was
optimized. Okpekpe & Orvieto's account predicts a *training*-time,
LR-schedule-dependent cliff; this program's cliff is measured *after*
training, by truncating a fixed matrix's rank directly — the two accounts
make different, distinguishable predictions, and this program's own
evidence bears on the structural account, not the optimization-brittleness
one. Named here so REASONING-LINK's own Leg A/B results are read against
this distinction from the start, not defended reactively later.

**Frozen-QK (arXiv:2506.01115) — nearest neighbor for the frozen-bias
intervention, distinguished on three axes.** That work fully freezes the
ENTIRE Q/K projections of a standard softmax transformer at init; induction
heads still form, but at a REAL loss cost (WikiText PPL 3.07 vs. 2.78,
≈10% worse). This program's frozen-bias intervention differs on three
axes, stated explicitly: **(a) additive, not full-freeze** — a tuned-λ
blend of a frozen table into an otherwise fully-trained key projection, not
a replacement of the projection itself; **(b) loss-neutral** — rung-1's own
val-loss gate (`FROZEN_BIAS_LM_DESIGN.md` §7.2) already established equal
val-loss across arms, unlike Frozen-QK's measured 10% regression; **(c) a
recurrent fast-weight state target, not a softmax-attention target** — the
mechanism this program cares about (whether the blend improves what a
FIXED-SIZE state can retain) has no analog in Frozen-QK's unbounded-context
softmax setting.

**VLA (arXiv:2605.11196) — the geometric rival account for the capacity
cliff, engaged against §14's coherence-exoneration result.** VLA's
stabilized architecture (Kalman/variational update, Sherman-Morrison,
write-direction normalized so the recurrence Jacobian's spectral norm is 1)
pushes a sharp MQAR-recall cliff outward from their DeltaNet baseline's
collapse near `n_pairs/d_h≈0.25–0.5` to `≈1`, framing capacity as tied to
**key linear independence / state-matrix rank** — a geometric account. This
is in direct, disclosed tension with `KEY_ANCHORING_DESIGN.md` §14.12/§14.13's
coherence-dose-response result: directly injecting anchor-table coherence
up to and including doses at or above the d=64 cliff's own band left
`h4=1.0` flat at every dose, at both tested structural kinds (rank-4 and
diffuse injection) — coherence, directly manipulated, was fully exonerated
as the cliff's driver in that experiment. **The tension is engaged, not
buried:** VLA's task is single-hop MQAR recall under an architectural
stabilization; this program's is relational binding plus multi-hop
composition, and the dose-response result was a direct CAUSAL manipulation
of coherence at matched load, not a correlational geometry read. It remains
possible that VLA's rank/linear-independence account and this program's
absolute-capacity account describe the same underlying fact from different
angles (a rank-deficient key population is also, definitionally, a
capacity-constrained one at fixed `d_state`) — this program does not claim
to have resolved which framing is more fundamental, only that its own
directly-causal dose-response manipulation did not find coherence-as-such to
be the lever, and states that finding against VLA's geometric account
explicitly rather than leaving a reviewer to find the tension first.

**OpenReview R8ZbLi3oUv — convergent motivation, adjudicated non-scoop.**
"From Recall To Reasoning: Understanding the Role of Associative Memory in
Hybrid Architectures" trains hybrid attention+recurrent models at 150M/500M
on a math-reasoning curriculum and identifies in-context associative recall
as a key error factor via LLM-as-judge error categorization. Full read
remains blocked (OpenReview bot-check defeats both the forum page and the
api2 endpoint as of the 2026-07-07 pass) — this is disclosed as a residual
uncertainty, not silently treated as fully verified. Adjudication from the
richer search-snippet reconstruction: thematically convergent (independent
support for the premise that recall bottlenecks reasoning in recurrent-
family/hybrid models) but methodologically distant from either leg here —
no load/capacity axis, no causal or representation-level intervention, no
hop-depth generalization structure, no probe-based state readout, purely
correlational benchmark-level evidence. Cited as convergent motivation, not
a scoop. A full read is owed before this design's related-work section is
finalized for any publication — flagged as not build-blocking.

---

## 3. Program structure

| Leg | Question | Checkpoints (existing, zero new training) | New training? |
|---|---|---|---|
| **A** | Does the frozen-bias intervention move composition? | `/data/deltanet_rd_frozenbias_ckpts/`, 3 arms × 2 corpora × 3 seeds, final (+ optionally trajectory) checkpoints | No |
| **B** | Does composition rise with scale, tracking span_frac? | `/data/lm_rd_trackc_ckpts/{mixcontrol,wave1ext,wave2,wave3}/`, 4 rungs × up to 2 corpora × up to 3 seeds | No |
| **Phase 2** (gated) | Does the intervention move a *standard* benchmark? | None yet — new small-LM training, 2 scales, gated on Leg A showing a real effect | Yes (sketch only, §11) |

Both legs share **one** probe-generation script, **one** readout formula,
**one** pinned-CI statistical convention (`t(2,.975)=4.303`, this project's
standing formula wherever n=3 seeds), and **one** shortcut-analysis
discipline (§7). They differ only in which checkpoints they load and which
comparison they register.

---

## 4. The shared instrument

### 4.1 Build adaptation from `grammar_rd.py` — required, registered before any code is written

`grammar_rd.py::build_entity_pools()` and `_permutation_graph()` are reused
**verbatim** — the entity/relation vocabulary (GPT-2-tokenizer-verified
single-token names, the same n=107 pool `KEY_ANCHORING_DESIGN.md` already
uses) and the single-Hamiltonian-K-cycle generator need no changes; neither
depends on the reserved buffer/query token IDs.

**What cannot transfer verbatim, and the registered fix:**
`grammar_rd.py`'s clause template (`<buf...> KEY <rel> VALUE .` /
`<buf...> KEY <rel> <Q>`) uses **two reserved token IDs beyond GPT-2's
50,257-token vocabulary** — a zero-pinned-and-frozen `BUFFER` id and a
learned `<Q>` query-marker id — both of which only exist because
`model_rd.py`'s from-scratch harness builds its embedding table with
`vocab_size_base + 2` rows. The frozen-bias-LM and Track C checkpoints were
trained with a **standard 50,257-row** embedding table (`lm_pretrain_rd.py`,
general LM pretraining, no reason to reserve extra rows). Feeding a
reserved-ID token into these checkpoints is an out-of-range embedding
lookup — not merely suboptimal, an index error.

**Registered adaptation (new script `reasoning_link_probe.py`, spec, not yet
built):**
1. Replace the `BUFFER` id with the tokenizer's own **period token** (the
   same in-vocabulary token `grammar_rd.py` already uses to close each bind
   clause, `pools.period_id`) repeated `conv_size−1` times between clauses —
   an ordinary, extremely high-frequency, maximally generic token the
   pretrained model has seen in essentially unconstrained contexts, chosen
   specifically to minimize the risk that its own (unpinned, trained)
   embedding carries clause-boundary-position information (§7.6 makes this
   an explicit shortcut check, not an assumption).
2. Replace the learned `<Q>` marker with a fixed, ordinary, low-frequency
   but genuinely in-vocabulary token verified at build time to (a) be a
   single BPE token, (b) not collide with any entity/relation-verb token in
   the pool, and (c) be rare enough in the pretraining corpora that its
   pretrained embedding is close to its random-init value (candidates:
   an uncommon symbol token, e.g. `"§"` or `"‽"` if single-token under
   GPT-2 BPE — verified, not assumed, at build time; if none qualifies, a
   short fixed *phrase* ending the query clause, e.g. `" then who ?"`,
   rendered identically across every episode/arm/rung). This is a genuine,
   disclosed deviation from `grammar_rd.py`'s own construction, forced by
   evaluating pretrained-not-from-scratch checkpoints — flagged again in
   §7.6.
3. No architecture change, no retraining, no embedding-table resize. The
   checkpoint loads exactly as archived.

**Self-test obligation (Stage −1, §9):** verify (a) every entity/relation
token, the period, and the query token are single-token under GPT-2 BPE and
mutually non-overlapping (mirrors `grammar_rd.py`'s own build-time
verification, re-run against this new template, not assumed to still hold);
(b) `conv_size` for every checkpoint family (rung-1/frozen-bias: `conv_size=4`
per `FROZEN_BIAS_LM_DESIGN.md` §5; rung-2/3: **must be read from each
checkpoint's own saved config, not assumed to also be 4** — a build-time
verification item, per the standing "verify against the harness's own
printed/asserted count" rule `SCALE_TRANSFER_DESIGN.md` §5.3 already
applies to param counts).

**Registered commitment (Rev 1, attack-round-1 M4): `conv_size` parametrizes
EPISODE CONSTRUCTION itself, per checkpoint, not merely a post-hoc
verification.** `grammar_rd.py::DeltaNetRDTaskConfig.buf_len` is a property
computed as `max(1, conv_size − 1)`, and `clause_len`/`query_len`/`T_bind`
all derive from it (§4.3 below) — §7.1's position-decomposition closure
argument depends on the episode's buffer window being sized to THAT
checkpoint's own local causal-conv reach, not a fixed assumed value. The
build must therefore construct a **fresh** `DeltaNetRDTaskConfig` per
checkpoint, with `conv_size` set from that checkpoint's own loaded config —
never one shared `cfg` object (and therefore one shared `buf_len`) reused
across checkpoints whose `conv_size` could differ (rung-1/frozen-bias vs.
rung-2/rung-3, per item (b) above). **Stage −1 assertion (new item, §9):**
after loading each checkpoint and constructing its episode config, assert
`checkpoint_config["conv_size"] == episode_cfg.conv_size` before generating
a single episode for that checkpoint — a hard equality check, not a printed
value trusted by eye, closing the exact "silently mismatched buffer length"
failure mode this finding named.

**Registered commitment (Rev 2, attack-round-2 FATAL): `conv_size` also
gates which `K` values are even LEGAL for a given checkpoint, not merely
episode layout.** `model_rd.py`'s `_MIN_KERNEL_T = 128` (the hard floor below
which `chunk_delta_rule`'s backward crashes, `model_rd.py` L117) applies
identically to `lm_pretrain_rd.py`'s production kernel call
(`DeltaNetLMMixer.forward`, L831-836: `assert T >= _MIN_KERNEL_T`) — and,
unlike Wave 1's from-scratch harness, **there is no state-neutral pad token
available to extend a too-short call** (L811-813: "LM mode has no
state-neutral pad token to extend a too-short call with (unlike Wave 1's
buffer-token trick)"). Wave 1's padding trick is state-neutral only because
its BUFFER pad positions carry a HARD, externally-computed `beta=0` — the
delta rule provably writes nothing there. This design's checkpoints use a
**learned, per-token `b_proj` gate** (§0, §4.1 above): a padding token here
would receive whatever `beta` the model's own sigmoid computes for it, which
is not provably zero, so padding would genuinely write into `S_T` — a
structural feature correlated exactly with low `K` (short episodes need more
padding), i.e. a confound built into the load axis this design exists to
measure, not a cosmetic engineering gap. **Registered fix: raise the K floor,
never pad.** Since `T_bind = K × clause_len(conv_size)` (§4.3's clause-length
derivation) and `clause_len(conv_size) = max(1, conv_size − 1) + 4`
(`grammar_rd.py` L316-322, verified directly this revision), every
checkpoint's own registered K sweep must satisfy, **per that checkpoint's own
`conv_size`** (already required to be read per-checkpoint, item 7 above —
this is the SAME per-checkpoint value, reused, not a second lookup):

> `K ≥ K_min(conv_size) = ceil(_MIN_KERNEL_T / clause_len(conv_size))`

At `conv_size=4` (`clause_len=7`, verified for rung-1/frozen-bias per §0;
**assumed**, pending item 7's own per-checkpoint verification, for rung-2/3
— `SCALE_TRANSFER_DESIGN.md` states no `conv_size` for those rungs, so this
is a registered assumption, not a confirmed fact), `K_min = ceil(128/7) =
19`. §4.3 below re-derives the swept K sets against this floor; **new Stage
−1 item 9 (§9) makes this a hard, checkpoint-specific gate**, not a value
trusted once at design time — if a future rung's own `conv_size` differs,
`K_min` moves with it and the assertion recomputes automatically from the
general formula above, never from a hard-coded `19`.

### 4.2 Episode construction

Each episode: `K` entities drawn from the full 107-name entity pool (§4.5,
Rev 1: the arithmetic-impossibility finding that forced drawing from the
full pool rather than a further sub-split is recorded in §13.1 F1; Rev 3:
with no probe-training/probe-eval distinction left to keep seed-disjoint
[§4.5], every episode now draws from this same pool under the single `eval`
purpose, §4.6), bound by a
single Hamiltonian K-cycle `π` (verbatim `_permutation_graph`, never a
general random permutation — closes the periodicity-collapse trap named in
`NEXT_EXPERIMENT_DESIGN.md` §2 and `CLAUDE.md`'s standing rule). The `K`
bind clauses (`<buf> KEY_i REL VALUE_π(i) .`) are rendered at **fixed slot
positions** — this is a correction of the pre-attack draft's own premise,
not the original design (Rev 1, attack-round-1 M5; recorded in full at
§13.1): `sample_batch_rd`'s actual construction (`grammar_rd.py` L450-461)
places clause `i`'s tokens at a position that is a deterministic function of
SLOT index `i` (`item_pos = arange(K)*clause_len + buf_len + 2`), not a
per-episode shuffle — the pre-attack draft proposed adding a NEW
randomized-render-order mechanism on the (incorrect) premise that
`grammar_rd.py` renders clauses in cycle order.

**What actually closes the local-shortcut risk, verified rather than
assumed (Rev 1):** the entities occupying each slot (`entity_ids`, via a
fresh per-episode without-replacement draw, `pool_idx`) and the K-cycle
bijection over SLOT indices (`succ`, via `_permutation_graph`, also freshly
drawn per episode) are two INDEPENDENT random draws, and `succ` is
constructed from a uniformly random permutation of slot indices with no
dependency on slot position at all. This means, for any slot `i`, `succ[i]`
(the slot holding the ONE-HOP target) is uniformly distributed over the
other `K−1` slots — in particular, `P(succ[i] = i+1)` (the target sitting at
the very next RENDERED clause, the exact local shortcut §7.1 worries about)
equals `1/(K−1)`, no different from any other slot, by construction of a
uniformly random permutation. Render-position adjacency and cycle-hop
adjacency are therefore decorrelated **by the existing random-slot-population
+ random-cycle draw alone** — no additional shuffle mechanism is structurally
required. **Registered empirical check (Stage −1, §9, new item, since the
project's standing discipline is to measure this rather than trust the
algebra alone):** generate ≥10,000 episodes at each swept `K` using the
real `sample_batch_rd`, measure the empirical rate of `succ[i] ∈ {i−1, i+1}`
(render-adjacent one-hop target) against the exact combinatorial chance rate
for that `K` (`2/(K−1)` interior, `1/(K−1)` at the two wrap boundary slots),
with a bootstrapped 95% CI (this project's pinned convention). **Pass
condition:** the measured rate's CI must include the theoretical chance
value — i.e., statistically indistinguishable from chance. If any swept `K`
fails this (CI excludes chance on the high side), the randomized-clause-
order shuffle originally proposed IS added as a build item before Stage 1
launches for that `K`; if all swept `K` pass (expected, per the analysis
above), §4.2's original shuffle proposal is dropped and this measurement is
cited in its place — the correction of the premise is recorded, not hidden.

Query clause: `<buf> KEY_a REL <Q-token>`, same relation verb as every bind
clause in the episode (congruence, not verb identity, is what the bind/query
window-shape match needs — `DELTANET_REALDATA_DESIGN.md` §5.2's own
convention, carried forward unchanged). The hop count `h` **never appears in
the surface form** — it is an external parameter of the readout only (§4.4).

**Batch size — PINNED (Rev 3, attack-round-3 MINOR), not left implicit.**
Every GPU pass in this design (forward-A, forward-B, §4.4) uses
**`batch_size=16`**, verified directly against this codebase's own existing
convention: `lm_attractor_probe_rd.py` and `frozen_bias_retrofit_eval_rd.py`
both default `--batch-size` to `16`, and rung-3's own Rev 2.2 pass-cost
calibration (`SCALE_TRANSFER_DESIGN.md`) was measured at batch 16 — this
design reuses that same value rather than introducing an unmeasured one.
Stage 0/0.5 calibration (§9) validates this batch size against a real
per-cell wall-clock measurement rather than assuming it transfers unchanged
to this design's own (differently-shaped) episodes.

### 4.3 Hop depths and K sweep — stratified by `d_state`, not held at a fixed K/d fraction

Every tested `K` must satisfy `K > h_max = 4` (the single-K-cycle guard
against periodicity collapse — trivially satisfied everywhere in this
design since the smallest tested K is **20** (Rev 2, §4.3's re-derived floor,
up from Rev 1's 8), but stated as a hard config-time assertion, mirroring
`TaskEConfig.__post_init__`'s own convention, never left to informal
discipline).

Hop depths, both legs: **h ∈ {1, 2, 3, 4}**. §7.10 registers precisely what
"in-distribution" vs. "held-out" means here (it is *not* the Task
E/DELTANET_REALDATA sense of "hops seen during training," since these
checkpoints saw no BIND/QUERY training at all) — provisionally, h=1 is a
**sanity floor** (established in the induction-head/in-context-recall
literature as within reach of a pretrained fixed-state LM; must clear a
registered chance-plus-margin bar or the whole probe is invalid, §8.4), h=2
is the **first composition step** (plausibly within reach via induction-head
composition / "function vector" mechanisms already documented for
Transformer-family ICL, though not yet demonstrated for a fixed-state
DeltaNet-family model), and h=3/4 are the **registered test depths** for
H_LINK-A/B — composition this project has never demonstrated for a
never-trained-on-the-task model.

K sweep, **per `d_state`** (the absolute-capacity reading, §0, forbids
assuming one K/d fraction transfers across `d_state` values) — **re-derived
this revision (Rev 2, attack-round-2 FATAL) against §4.1's `K ≥
K_min(conv_size)` floor.** At `conv_size=4` (`K_min=19`), the ENTIRE
Rev-1 low end (`K=8`, `T_bind=56`; `K=16`, `T_bind=112`) sits below
`_MIN_KERNEL_T=128` and hard-crashes `chunk_delta_rule` — this affected BOTH
the `d=64` row (which used both 8 and 16) and the `d=128` row (which used
16), so every registered K below 19 is replaced, not merely the ones named
in the FATAL finding:

| `d_state` | K values, LEGAL (clear `K_min(conv_size)=19`) | K values, COMMITTED per leg (Phase-1 grid, Rev 3) | K values, EXTENSION (named, priced §10, not committed) | Checkpoints using this `d_state` |
|---|---|---|---|---|
| 64 | 20, 32, 40, 48 | **Leg A: 20, 32** (loads 0.3125 / 0.5); **Leg B: 32 only** (load 0.5) | 40, 48 (Leg A); 20, 40, 48 (Leg B) | Leg A (all 3 arms); Leg B 14M + 98M rungs |
| 128 | 20, 32, 64, 96 | **Leg B: 64 only** (load 0.5) | 20, 32, 96 (loads 0.15625/0.25/0.75) | Leg B 392M + 1.31B rungs |

**Rev 3 (attack-round-3 FATAL-2 — the two-forward protocol's cost forces a
per-leg, not per-`d_state`, committed set).** Rev 2's table listed one
committed K set per `d_state`, shared by whichever leg used it. §10's
FATAL-2 re-derivation (the mandatory two-forward pass doubles the per-pass
rate, 4×→8×) makes that no longer affordable at Rev 2's resolution — but
the two legs' own primary claims never needed the same K's in the first
place: Leg A's §5.3 killer prediction is structurally a **two-point**
K-contrast (one low-load, one near-cliff, §5.3), while Leg B's §6.2 primary
reading was **always** a single near-cliff K per `d_state` (`K=32` at
`d=64`, `K=64` at `d=128` — `K=20` never entered Leg B's headline at all,
§6.3). Committing per-leg rather than per-`d_state` costs neither leg's
actual claim anything, and is what makes the grid affordable again (§10).
The LEGAL column (which K's clear the `T_bind≥128` floor at
`conv_size=4`) is unchanged from Rev 2 and gates every cell regardless of
committed/extension status.

`T_bind` at every listed value (20×7=140, 32×7=224, 40×7=280, 48×7=336,
64×7=448, 96×7=672) clears `_MIN_KERNEL_T=128` with margin — verified
arithmetically here and re-verified per-checkpoint at build time by Stage −1
item 9 (§9), which is the actual gate, not this table.

**Why these specific K's (Rev 2 selection logic, still governing which K's
matter even after Rev 3's per-leg committed/extension split above):** the
`d=64` killer-prediction test (§5.3) needs exactly a low-load contrast plus
its near/past-cliff point(s) — `K=20` (replacing the invalid K=8 low-load
point), `K=32` (closest tested point below the located `x0≈0.5455` cliff
center, `0.5455×64≈34.9`, and the single primary agreement-gate K, §5.3),
and `K=48` (intentional overshoot past the cliff on the "collapsed" side,
per `KEY_ANCHORING_DESIGN.md` §11.12's own K=48 capacity-curve convention,
unchanged from Rev 1 — Rev 3: committed for Leg A through Rev 2, now a named
extension per §10's FATAL-2 re-derivation, since it was never the primary
agreement-gate K to begin with). `K=40` (a non-critical fourth resolution
point, playing the same role Rev 1's `K=16` played) remains a named
extension, unchanged in status. At `d=128`, Rev 1's own committed pair was
`{8,32}` — but `K=8` was never a valid `d=128` point at all (MAJOR-1
below), and `K=32` (load 0.25) is not d=128's near-cliff point; `K=64`
(`0.5455×128≈69.8`) is (§6.2 already used K=64 as the headline Leg-B
near-cliff comparison, which means Rev 1's own §6.2 reading depended on a K
value that was NOT actually in Rev 1's own committed budget — an
additional latent inconsistency Rev 2's fix resolved as a side effect).
`K=64` is Leg B's `d=128` primary near-cliff K (§6.2) and, Rev 3, its ONLY
committed `d=128` point — `K=20` (low-load) and `{32,96}` (mid/deep-collapse
points) are named extensions (§10), since Leg B's headline never used any
of them (§6.3).

At `d=128`, both committed and extension K's sit at loads
(0.15625/0.5/0.25/0.75) that are still, per §13.10, inside the window that
does NOT reproduce the d=64 cliff on the coherence/rank instrument — this
remains the d=128 leg's **direct, independent test of whether the
absolute-capacity reading also holds for a genuinely different observable**
(reasoning recovery, not Gram-deviation/rank), unchanged in intent from
Rev 1, only in the specific K values realizing it.

**Build-time convention for `DeltaNetRDTaskConfig.H_train`/`H_test`/`H_extra`
on never-trained checkpoints (Rev 1 minor, attack-round-1).** These field
names describe a "hops the model trained on" split in `grammar_rd.py`'s
original from-scratch harness; no such split has any referent here (§7.10 —
these checkpoints trained on zero hops of anything). The dataclass's own
`__post_init__` still requires `len(H_train) ≥ 1`, `H_train ∩ H_test = ∅`,
and the periodicity guard (no tested hop's `h % K` may collide with a
`H_train` hop's residue). Registered, semantically-inert construction: for
EVERY checkpoint/K combination, `H_extra=()` (unused — this design tests
only h∈{1,2,3,4}, never h=7/21), `H_test=(1,2,3,4)` (this design's FULL
tested-hop set — the "sanity floor" vs. "registered test depth" distinction
of §4.3 above is a DESIGN-level split, not a dataclass-level one), and
`H_train=(5,)` — a single fixed placeholder chosen because `5 < 20 =
min(K)` across the entire swept range `{20,32,40,48,64,96}` (Rev 2's
re-derived committed-plus-extension range, §4.3 — the placeholder's
validity is unaffected by the FATAL-driven K-set change, since `5` sits
below the new, higher floor too), so `5 mod K = 5` never collides with
`{1,2,3,4}`'s residues at any swept K, satisfying the dataclass's structural
invariants without asserting any false "trained-hop" semantics.

### 4.4 Readout — three options enumerated, one chosen with registered justification

**Option 1 (PRIMARY) — hooked fast-weight composition, scored directly in
`d_state` space against the target entity's own effective value (REVISED,
Rev 4 — attack-round-4 FATAL; see §13.4. Rev 3's FATAL-1/FATAL-2 fixes —
drop the trained probe, the two-forward protocol — both HOLD unchanged and
are carried forward verbatim below; only the compared-against object
reverts from Rev 3's `k_eff` back to `v_eff`.)**

> **FATAL-1 (why the Rev 2 readout died — a dimension mismatch, not a
> tuning problem; HOLDS, Rev 3 fix unchanged).** Rev 2's Option 1 scored
> `pred(a,h)` through a **trained linear probe** `W_probe: R^{d_state} →
> R^{d_state}` mapping it into the checkpoint's own **input-embedding
> space**, cosine-scored against `W_embed[value_token_id] ∈ R^{d_model}`.
> But `W_probe`'s declared codomain (`R^{d_state}`) never matched the space
> it was actually scored in (`R^{d_model}`) — and `d_state ≠ d_model` at
> **every** checkpoint this design evaluates (64 vs. 256/768 for the
> frozen-bias/14M/98M family, 128 vs. 1536/2560 for the 392M/1.31B family,
> §0/§6.1). The formula could not even be evaluated as written; every
> downstream probe-training, probe-leakage, and memorization-ceiling
> apparatus (Rev 2 §4.5) inherited the hole.

> **FATAL (Rev 4, attack-round-4 — Rev 3's `k_eff_target` choice is
> algebraically unsound, not merely a weaker convention; full record at
> §13.4).** The delta rule's per-step state update is
> `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` (this project's own
> DeltaNet kernel, `β_t` the model's learned `b_proj` sigmoid — but the
> argument below holds for **any** scalar `β_t`, not only `[0,1]`).
> **Claim, by induction on `t`: `Im(S_t) ⊆ span{v_eff_1,...,v_eff_t}` for
> every `t`, for any `β`.**
> - *Base case:* `S_0 = 0` (the recurrence's own zero-initialized state,
>   forward-A's own starting condition, §4.4 below) — trivially
>   `Im(S_0) = {0} ⊆ span{v_eff_j}`.
> - *Inductive step:* assume `Im(S_{t-1}) ⊆ span{v_eff_1,...,v_eff_{t-1}}`.
>   For any `x ∈ R^{d_state}`: `S_t x = S_{t-1}x − β_t(k_tᵀx)S_{t-1}k_t +
>   β_t(k_tᵀx)v_t`. The first two terms are `S_{t-1}` applied to some
>   vector (`x` and `k_t` respectively) — both land in `Im(S_{t-1}) ⊆
>   span{v_eff_1,...,v_eff_{t-1}}` by the inductive hypothesis; the third
>   term is a scalar multiple of `v_t = v_eff_t`. So `S_t x ∈
>   span{v_eff_1,...,v_eff_t}` for every `x`.
> - By induction, `Im(S_T) ⊆ span{v_eff_1,...,v_eff_K}` — the **full
>   value-family span**, independent of every `β_t`'s actual value.
>
> **Consequence for `pred(a,h) = S_Tʰ·q_eff_a`.** For `h=1`:
> `S_T q_eff_a ∈ Im(S_T) ⊆ span{v_eff_j}` — already value-space. For
> `h≥2`: `S_Tʰ q_eff_a = S_T(S_T^{h-1}q_eff_a)`, and `Im(S_T)` is fixed
> regardless of what vector `S_T` is applied to — so `S_Tʰ q_eff_a ∈
> Im(S_T) ⊆ span{v_eff_j}` for **every** `h≥1`, not only `h≥2`.
> **`pred(a,h)` is a `v_eff`-family object at every tested `h`; there is no
> hop at which it is ever a `k_eff`/`W_k`-family object.**
>
> **Why Rev 3's own rationale (§13.3 reason 1, quoted below) was
> backwards.** Rev 3 argued that scoring against `k_eff` "keeps every
> hop's comparison apples-to-apples" because `q_eff` and every bind-phase
> `k_eff` share the `W_k` family, and worried that scoring against `v_eff`
> would need an unestablished family-crossing at every hop. The algebra
> above shows the opposite: `pred(a,h)`'s family is set by `Im(S_T)` — a
> `v_eff`-family object by construction of the delta rule itself — not by
> which family `q_eff` belongs to before being matrix-powered (`S_T`'s
> image doesn't depend on its input's family). **`v_eff` scoring is the
> family-consistent choice at every `h`, including `h=1`; `k_eff` scoring
> was cross-family with no established referent at every `h`, including
> `h=1`.** DIRECTIVE, executed below: revert the comparison target to
> `v_eff_target`, per `DELTANET_REALDATA_DESIGN.md` §14.3's exact original
> convention.

**Fixed readout: drop the trained probe entirely; stay in `d_state` space,
score against the model's own machinery, never the embedding table (Rev 3,
HOLDS unchanged).** Run the pretrained checkpoint forward over the BIND
phase only (its own natural, learned β via `b_proj` — **not** hard-masked,
since the checkpoint cannot be retrained; this is itself part of what's
being tested, §7 throughout), capturing the per-layer final recurrent
state `S_T` via `chunk_delta_rule`'s `output_final_state=True`. Compute
`pred(a,h) = S_Tʰ @ q_eff_a` externally (plain matrix power in
`d_state`-dim space, cheap, still zero learned parameters). Score via
**absolute cosine ≥ 0.9 against the TARGET entity's own effective value,
`v_eff_target`** (REVERTED, Rev 4 — see the FATAL derivation above) —
extracted through a **new** per-layer, post-conv, pre-kernel hook on
`v_conv1d`'s output (built below), mirroring the **existing** `k_conv1d`
hook pattern (`lm_attractor_probe_rd.py::capture_raw_keys`, §0) exactly.
`v_eff`, `k_eff`, `q_eff`, and `pred(a,h)` are therefore all `R^{d_state}`
vectors produced by the checkpoint's own forward machinery — **no learned
probe, no projection into `d_model`, no dimension mismatch possible by
construction**, at `d_state=64` or `d_state=128` alike. **The probe stays
dropped (Rev 3's dimensionality fix HALF survives, Rev 4):** the part of
FATAL-1's fix that matters — never project into `d_model`, stay in
`d_state` space — is untouched by this revert; only the specific
in-family object compared against changes (`v_eff` instead of `k_eff`),
and `v_eff` is exactly as `d_state`-dimensional as `k_eff` was, so no
probe of any kind is reintroduced to make this comparison well-typed.

**The `v_conv1d` hook — a ~5-line addition, not a new observable class
(Rev 4; corrects §4.4's Rev 3 "no such observable exists" overclaim).**
`lm_pretrain_rd.py`'s `DeltaNetLMMixer` already instantiates
`self.v_conv1d = ShortConvolution(...)` at L793 (verified directly, §0)
and calls it at L839 (`v, _ = self.v_conv1d(self.v_proj(x))`) **inside the
same forward pass that already computes `k_conv1d`'s output** — Rev 3's
claim that "no `v_conv1d` hook exists anywhere in this codebase" was
accurate about the HOOK (no code currently attaches to it) but overclaimed
the absence of the underlying observable itself: the submodule is already
there, already executed, on the exact forward pass Option 1 already runs.
The fix is a direct copy of `capture_raw_keys`'s own hook-registration
pattern (`lm_attractor_probe_rd.py` L163-170), retargeted from
`blk.mixer.k_conv1d` to `blk.mixer.v_conv1d`:

```python
def make_v_hook(i):
    def hook(module, inp, out):
        v_raw = out[0] if isinstance(out, tuple) else out
        captured_v[i].append(v_raw.detach())
    return hook

v_handles = [blk.mixer.v_conv1d.register_forward_hook(make_v_hook(i))
             for i, blk in enumerate(model.blocks)]
```

registered **alongside** the existing `k_conv1d` hook on the **same**
forward-A call — both hooks fire on one pass, so capturing `k_eff_items`
AND `v_eff_items` together costs one forward, not two. This build item
adds no new GPU pass and no measurable cost (§10 confirms the budget is
unaffected).

**MINOR (Rev 5, attack-round-5) — `apply_frozen_bias_blend` touches ONLY
`k`, a disclosed design strength of the `v_eff` convention, stated
explicitly rather than left implicit.** `lm_pretrain_rd.py` L854-857
verified directly: the frozen-bias conditional wraps exactly one
reassignment, `k = apply_frozen_bias_blend(k, token_ids, ...)` — `q` and
`v` are never arguments to it, in any arm. Consequence: `v_eff_target`
(the Rev 4 scoring target) and, as of this revision, `q_eff` (via the new
`q_conv1d` hook) are **blend-invariant by construction** across all three
Leg-A arms — Arm 2's extra frozen table cannot reach either tensor
through any code path, not merely "code-path-identical" in the weaker
sense §7.9 already argued for `k_eff`. This means the readout's TARGET
(`v_eff`) and its QUERY (`q_eff`) are both immune to the exact confound
§5.2a's blend-toggle surgery grid exists to isolate for `k_eff` — a
strength of the `v_eff`/`q_conv1d` convention worth stating plainly, not
an accident to be silently relied on (full argument at §7.9).

**Exactly which target — quoting `DELTANET_REALDATA_DESIGN.md`'s own
established convention VERBATIM, zero departure (Rev 4 reverts Rev 3's
one specified departure).** That design's §5.2 pins the readout this
project already validated at production-kernel scale —
`pred(a,h) = S_Tʰ · q_eff_a` — and its §14.3 restates the scoring rule
verbatim (quoted directly, not paraphrased): *"the readout: the pinned
linear unbind `pred(a,h) = S_Tʰ · q_eff_a`, scored at eval by **absolute
cosine against the target `v_eff` at the 0.9 threshold — exact continuous
recovery, never an in-episode argmax or softmax**."* Rev 4 mirrors this
convention with **zero specified departure** — same-space (`d_state`)
scoring, absolute cosine, the 0.9 threshold, exact continuous recovery,
`v_eff` as the compared-against object — closing the one deviation Rev 3
introduced, which this revision's own FATAL finding (above) shows was
algebraically unsound.

**Why Rev 3's two "reasons for departure" don't survive scrutiny
(recorded for continuity, not because either was carelessly reasoned —
both examined the wrong family-membership fact; full record §13.4):**
1. *Representational-family consistency across `h`* — Rev 3 reasoned from
   which family `q_eff` belongs to; the FATAL derivation above shows
   `pred(a,h)`'s family is set by `Im(S_T)` (the value-family) regardless
   of `q_eff`'s own family membership before being matrix-powered — the
   premise this reason rested on was never the operative one.
2. *Mechanical minimality (no new hook)* — correct that no `v_conv1d` hook
   existed in this codebase at Rev 3's time of writing, but "avoid a new
   hook" is a build-convenience criterion, not a correctness one; the
   `v_conv1d` hook above shows the true cost is a ~5-line, zero-GPU-cost
   addition once the algebra makes the correct target clear, so this
   reason, though true as stated, was never sufficient grounds to prefer
   an unsound target over a sound one.

**Well-defined by the grammar's own structure, gathered at the CORRECT hop
index (Rev 4 — carrying `DELTANET_REALDATA_DESIGN.md` §14.2's audit-fix
forward, not re-introducing the exact bug it fixed once already).** Every
entity in a single-Hamiltonian-K-cycle episode plays **both** roles
exactly once: KEY in its own bind clause, VALUE in its predecessor's.
Concretely (`grammar_rd.py::sample_batch_rd`, §0): `key_ids = entity_ids`
(slot `i`'s KEY is the entity AT slot `i`) and
`value_ids = entity_ids.gather(1, succ)` (slot `i`'s VALUE is the entity
at slot `succ[i]`) — clause `i` states "entity at slot `i`, `π(·)`-related
to = entity at slot `succ[i]`." `v_eff_items[i]` (captured at clause `i`'s
own write position, `item_pos[i]`) is therefore the effective value
representation of the entity **at slot `succ[i]`** — not slot `i`'s own
entity.

`tgt_slot = _iterate_permutation(succ, a_slot, hops)` (already computed by
`sample_batch_rd`, §0) equals `succ^h(a_slot)` — the SLOT holding the
answer entity `π^h(a)`. **Naively gathering `v_eff_items[tgt_slot]` would
return the value written at clause `tgt_slot`, representing entity
`π(π^h(a)) = π^{h+1}(a)` — one hop PAST the intended answer** — the exact
off-by-one-hop bug `DELTANET_REALDATA_DESIGN.md` §14.2's mini-audit found
and fixed once already in this codebase ("clause `i`'s VALUE token is
entity `π(i)`, so the scored representation belonged to entity
`π^{h+1}(a)` — one hop past the queried answer `π^h(a)`"). **Rev 4's fix
carries that exact audit-fix forward by construction, not merely by
citation:**

```
prev_slot     = _iterate_permutation(succ, a_slot, hops - 1)   # slot of pi^(h-1)(a)
v_eff_target  = gather(v_eff_items, 1, prev_slot)              # value written at clause prev_slot
                                                                # = entity at slot succ[prev_slot]
                                                                # = entity at slot succ(succ^(h-1)(a_slot))
                                                                # = pi^h(a)  -- the correct target
```

equivalently, `prev_slot = inv_succ[tgt_slot]`
(`DELTANET_REALDATA_DESIGN.md` §17.2's own naming for the identical
quantity) — computed here via `_iterate_permutation(succ, a_slot,
hops-1)` rather than building a separate inverse-permutation array, since
`_iterate_permutation` is already called one line above (for `tgt_slot`)
and `succ^{-1}(succ^h(a_slot)) = succ^{h-1}(a_slot)` by the definition of
a permutation's inverse — zero new machinery, only a change of the
hop-count argument from `hops` to `hops-1`. `_iterate_permutation`'s own
`hops==0` base case (`cur = a_idx.clone()`, `grammar_rd.py` L279-280)
already returns `prev_slot = a_slot` at `h=1` — no boundary special-casing
needed.

**3-entity worked example (K=3), verifying the index arithmetic end to
end (build this as the Stage −1 self-test, §9).** Let
`succ = [1, 2, 0]` (a 3-cycle over slots `0→1→2→0`) and
`entity_ids = [A, B, C]` (slots 0, 1, 2). Then `key_ids = [A,B,C]` and
`value_ids = entity_ids[succ] = [B,C,A]` — the three bind clauses state
"A REL B", "B REL C", "C REL A" (`π(A)=B, π(B)=C, π(C)=A`). Query
`a_slot=0` (entity A):
- **h=1** (true answer `π(A)=B`): `tgt_slot = succ¹(0) = 1` (sanity-check
  only — the slot of B). `prev_slot = _iterate_permutation(succ, 0,
  hops-1=0) = 0` (the `hops==0` base case returns `a_slot` itself).
  `v_eff_target = v_eff_items[0]` = value written by clause 0 = the
  representation of entity **B** (clause 0: KEY=A, VALUE=B). **Correct: B
  = π(A).** (The naive `v_eff_items[tgt_slot=1]` would instead return C —
  `π²(A)`, one hop past the true answer.)
- **h=2** (true answer `π²(A)=π(B)=C`): `tgt_slot = succ²(0) = succ[1] =
  2` (sanity-check only — the slot of C). `prev_slot =
  _iterate_permutation(succ, 0, hops-1=1) = succ[0] = 1` (slot of B).
  `v_eff_target = v_eff_items[1]` = value written by clause 1 = the
  representation of entity **C** (clause 1: KEY=B, VALUE=C). **Correct: C
  = π²(A).** (The naive `v_eff_items[tgt_slot=2]` would instead return A —
  `π³(A)`, again one hop past.)

**PREMISE DIAGNOSTIC (iv) — cross-role identity at intermediate entities,
measured not assumed (Rev 4, attack-round-4 FATAL consequence).** The
FATAL derivation above shows `S_Tʰ q_eff_a` lands in `span{v_eff_j}` at
every `h` via repeated application of `S_T`. For `h≥2` specifically, this
composition additionally requires — beyond the family-consistency the
revert above already restores — that `S_T` continues to behave as an
associative-recall operator when handed one of its OWN outputs as a new
input: `S_T` maps a KEY vector `k_eff_j` to (to whatever extent the
checkpoint learned this) `≈v_eff_{π(j)}`; a second hop feeds `S_T` the
first hop's own output, `≈v_eff_{π(a)}`, and only continues to track
`π²(a)` to the extent that `v_eff_{π(a)} ≈ k_eff_{π(a)}` — i.e. the
INTERMEDIATE entity's value representation and its own key representation
are cross-role aligned.

**PREMISE DIAGNOSTIC (iii) — bind↔query alignment, now a genuine measured
quantity, not a near-tautology (Rev 5, attack-round-5 FATAL consequence).**
`DELTANET_REALDATA_DESIGN.md` §5.2/R2-2's premise (iii) is
`cos(q_eff_a, k_eff_a)` — does the query's own read-position
representation for entity `a` align with `a`'s own key representation from
its bind clause? That premise was built for a harness (`model_rd.py`) with
**no `q_proj` at all**, where `q_eff` IS `k_eff` re-read through the
identical `W_k`/`k_conv1d` operator at a different position — a high
score there is close to a tautology (the same weight matrix applied to
the same token in a similar local context, not an independent
comparison). Through Rev 4, this design's own premise (iii) inherited
that exact convention via the FATAL bug above (`q_eff` extracted via
`k_conv1d`), despite this checkpoint family having a real, separately
TRAINED `q_proj`/`q_conv1d` (§0). **Rev 5's fix strengthens premise
(iii)'s meaning, it does not merely correct a bug:** with `q_eff` now
genuinely extracted through the model's own `q_conv1d`, `cos(q_eff_a,
k_eff_a)` measures whether two INDEPENDENTLY-TRAINED projections (`W_q`
vs. `W_k`) of the same entity token converge to a similar representation —
a real, falsifiable structural fact about the checkpoint's own learned
geometry, not an artifact of reusing one weight matrix twice. **`h=1`
recovery structurally depends on this premise:** `pred(a,1) = S_T ·
q_eff_a` can only recover `v_eff_target` correctly if `q_eff_a` functions
as a good read-cue into the state `S_T` actually trained on
`(k_eff_a, v_eff_·)` pairs during the BIND phase — i.e., only to the
extent `q_eff_a ≈ k_eff_a`.

**Both measured identically, not assumed — and (Rev 6, attack-round-6
MAJOR) measured PER CHECKPOINT FAMILY, not only at the single Stage-0
point:** report `cos(q_eff_a, k_eff_a)` (premise iii) and
`cos(k_eff_i, v_eff_i)` (premise iv) per entity, alongside premises
(i)-(ii) (`DELTANET_REALDATA_DESIGN.md` §5.2, R2-2). Attack-round-6
identified that certifying these gates at one 2-layer/d_state=64
architecture and silently applying the verdict to 12/16/22-layer,
d_state=128 checkpoints leaves exactly the failure mode §1's own framing
note predicts (deeper models are MORE likely to compose via cross-layer
chaining, i.e. more likely to fail premise iv) unguarded at the rungs
where it is most likely. The fix costs zero new GPU passes: the premise
quantities are computed from the SAME per-layer `k_eff`/`q_eff`/`v_eff`
tensors the probe already captures at every cell, so the premise
distributions and their nulls are re-derived AT EVERY RUNG'S OWN PRIMARY
CELL (first pass per rung) and each rung's confirmatory eligibility is
gated on ITS OWN measurements per the table below — Stage 0's
measurement licenses only the architectures that share its config
(d_state=64, n_layers=2: the Leg-A arms and the 14M rung); 98M, 392M,
and 1.31B are each licensed (or demoted) by their own first-pass premise
read, applied before that rung's own unblinding.

**MAJOR-1 fix (Rev 5, attack-round-5) — one mechanical, null-relative
action-rule table replaces the reused, unjustified 0.9 bar and the
"disclosed alongside" language this design used through Rev 4, which had
no stated consequence for a cross-role comparison whose null is near
zero.** At Stage 0, for EACH premise, measure (a) the per-entity
SAME-entity cosine distribution (the premise's own quantity, above) and
(b) its NULL — the identical cosine computed over CROSS-entity shuffled
pairs drawn from the same calibration pool (entity `i`'s key vs. a
DIFFERENT, randomly chosen entity `j≠i`'s value/query). No `0.9` constant
anywhere; the gate is relative to each premise's own measured null, never
an externally invented number:

| Premise | Gate (measured at Stage 0, before unblinding) | Consequence on fail |
|---|---|---|
| **(iii) bind↔query alignment** — median same-entity `cos(q_eff_a, k_eff_a)` | Must exceed the 95th percentile of the cross-entity null `cos(q_eff_a, k_eff_{j≠a})` | `h=1`'s confirmatory status is revoked and pre-declared exploratory-only — `h=1` recovery structurally depends on this premise alone (above), so its failure is expected to produce a floor at EVERY `h`, not a surprise requiring separate diagnosis |
| **(iv) cross-role (k↔v) identity** — median same-entity `cos(k_eff_i, v_eff_i)` | Must exceed the 95th percentile of the cross-entity null `cos(k_eff_i, v_eff_{j≠i})` | `h≥2` cells are pre-declared exploratory-only (`h=1` RETAINS confirmatory status if premise (iii) alone passes, since `h=1` never compounds through premise (iv)) — READOUT-FORM-INVALID (§12) becomes the EXPECTED outcome for `h≥2`, not a surprise requiring separate diagnosis |

Both gates are evaluated and their consequences applied **before** Stage 1
launches for the Stage-0-licensed family, and **before each rung's own
unblinding** for the other families (Rev 6): a premise that fails its own
gate demotes the dependent hop-depths' confirmatory status in the
pre-registration itself, never after seeing how the headline numbers
happen to look. The same per-family scoping applies to §8.4's null-band
construction: each d_state/K combination's chance floor is measured from
its own rung's first-pass label-shuffle null (the 392M/1.31B rungs run at
K=64 with a different chance geometry than the calibration cell's K=32 —
their floors are measured there, not transplanted).

**MAJOR-2 fix (Rev 5, attack-round-5) — an absolute backstop alongside the
null-relative gate, so h=1 cannot be declared valid by a
near-zero-vs-near-zero comparison alone.** See §8.4 for the full
registration: `h=1`'s `recovered_frac@0.9` must ALSO clear a registered
absolute floor (0.10), independent of where the null band happens to sit.

**Wired into outcome interpretation, restated with the table above
replacing the free-floating threshold:** a low premise-(iv) reading is
one of the two candidate explanations for a uniform `h≥2` floor
(**regardless of whether genuine reasoning capability exists in the
checkpoint**) — the other being the mechanism-framing point of §1:
single-layer self-iteration may simply be the wrong hypothesis for how a
given checkpoint composes. **Both are disclosed together wherever a floor
result at `h≥2` is reported** (including under the READOUT-FORM-INVALID
outcome, §12), never picked as if the other explanation had been ruled
out.

**Framing adjudication and query-circularity disclosure (Rev 4,
attack-round-4 MAJOR, folded in per the attacker's own analysis —
condensed from §1's fuller statement, restated here in readout-specific
form).** Option 1 tests ONE SPECIFIC mechanism: single-layer state
self-iteration (`S_T` from ONE layer, matrix-powered `h` times against
that SAME layer's own query). This is an a priori unlikely implementation
choice for how a multi-layer, general-purpose pretrained LM composes
multi-hop information — the standard mechanistic-interpretability prior is
CROSS-LAYER chaining, not one layer repeatedly re-applying its own state.
Compounding this: `q_eff_a` (extracted from the FINAL layer via
forward-B, below) is itself the output of that layer's own conv
processing of a residual stream that ALREADY contains every earlier
layer's own computation, including whatever those earlier layers did with
their OWN states — Option 1 therefore measures the full multi-layer
stack's terminal representation, filtered through one specific (and a
priori unlikely) hypothesis about how composition is organized, never an
isolated single-layer mechanism in any clean sense. This is disclosed as a
structural property of the readout, not a defect to be engineered away —
the two-forward protocol (below) could not extract `q_eff` any other way
without abandoning the multi-layer checkpoints entirely (FATAL-2, above).
**What this does NOT invalidate:** Option 1's ARM and SCALE CONTRASTS
(§5.3, §6.2) remain valid comparisons even where its absolute recovery
level is compressed toward floor by testing the wrong mechanistic
hypothesis for a given checkpoint — if an intervention or scale change
genuinely shifts how much composition signal is expressed through
single-layer self-iteration specifically, a differential reading is
informative regardless of the absolute level; this is precisely why
§5.3's killer prediction is stated as a delta across K, never as an
absolute-recovery threshold, and why a uniform `h≥2` floor across every
arm/rung routes to READOUT-FORM-INVALID (§12) rather than REFUTE.

**The extraction problem this creates for a multi-layer production LM —
FATAL-2, and the fix.** `k_eff`/`q_eff` extraction cannot simply mirror
`model_rd.py::effective_key_window` (§0) — that function calls
`self.embed`→`self.k_proj`→`self.k_conv1d` **directly** on a short
(`query_len`, e.g. 6-token) window, entirely bypassing any sequence-length
floor, because `model_rd.py` is a **single-mixer, no-conv-boundary-assert**
harness with no multi-layer residual stream to reconstruct. The production
checkpoints this design evaluates (`lm_pretrain_rd.py::DeltaNetLMMixer`,
§0) hard-assert `T ≥ _MIN_KERNEL_T = 128` **before** the k/q/v convs run
(L831-836) — and for layer `i>0`, that layer's own input is layer `i−1`'s
**residual-stream output**, which itself requires a valid (`T≥128`)
forward through layer `i−1`. A bare short-window submodule call, run
layer-by-layer, would either (a) crash the `T≥128` assert at every layer,
or (b) if bypassed ad hoc, silently feed layer `i>0` an input that never
passed through a real multi-layer forward at all — breaking the per-layer
`S_T` framing this design's entire multi-layer claim (§6.1, `d_state=128`
rungs) depends on. Layer-0-only extraction (the tempting shortcut) is
explicitly rejected for the same reason: it would validate only the
single-layer case and silently fail to generalize to rung-2/3.

**FATAL (Rev 5, attack-round-5 — every `q_eff` extraction through Rev 4
hooked the wrong projection; full record §13.5).** Every prior mention of
`q_eff` extraction in this design — §4.4's Forward-B bullet below, the
FATAL-2 fix table (§13.3), §10's Option-1 justification, §7.9's cross-arm
confound argument, §14's checklist — said `q_eff` comes "via the same
`k_conv1d` hook at the query's own final position." That convention was
correct for `model_rd.py`'s from-scratch harness (§0) — that harness has
**no `q_proj` at all**, so reading the query through `k_conv1d`/`W_k` was
the only option, forced by the architecture, not a chosen approximation.
It is WRONG for the checkpoints this design actually evaluates:
`lm_pretrain_rd.py`'s `DeltaNetLMMixer` instantiates three UNTIED
projections/convs (`q_proj`/`k_proj`/`v_proj`, `q_conv1d`/`k_conv1d`/
`v_conv1d`, L787-793), computes all three on every forward
(`q, _ = self.q_conv1d(self.q_proj(x))` etc., L837-839), and calls
`chunk_delta_rule(q=q_bf, k=k_bf, v=v_bf, ...)` (L983) — the model's own
recurrent READ at any position runs on `q_bf`, the Q-path's own output,
never `k_bf`. Extracting `q_eff` via `k_conv1d` therefore measured a
projection (`W_k`) the checkpoint's own forward pass never actually reads
the query through — an arbitrary substitution, not a principled
approximation, discovered because the from-scratch harness's forced
convention was carried over unexamined into a checkpoint family that does
not share the constraint that forced it. **DIRECTIVE, executed below:
retarget forward-B's query-position extraction hook to `q_conv1d`**
(already instantiated at L791 and already executed on every forward at
L837 — a ~5-line hook copy, mirroring Rev 4's own `v_conv1d` hook, zero
new GPU passes). Bind-phase `k_eff` extraction (forward-A) and value
extraction (`v_eff`, forward-A) are UNCHANGED — this fix touches only the
query-position hook, nothing else. **This strengthens premise (iii)
(bind↔query alignment) rather than weakening it** — with `q_eff` now
genuinely a `W_q`-family object, `cos(q_eff_a, k_eff_a)` compares two
INDEPENDENTLY-TRAINED projections of the same entity token, a real,
falsifiable structural fact, not the near-tautology it was when both
sides were computed by the identical `W_k` matrix (see the premise
diagnostic and the unified action-rule table below).

**Fix — the two-forward protocol.** Every episode's readout is built from
**two separate full-length forward calls** through the checkpoint's own
multi-layer `DeltaNetLM.forward`, never a short submodule-level call:

- **Forward-A (bind-only, captures `S_T`).** The streamed BIND-phase
  sequence alone (`T_bind = K × clause_len(conv_size) ≥ 128`, guaranteed
  by §4.1's own `K_min(conv_size)` floor for every registered `K`), run
  through every layer with `output_final_state=True` — yields each
  layer's own final `S_T`, and, via the `k_conv1d` hook AND the new
  `v_conv1d` hook (both registered on this SAME forward call, above),
  every bind-clause's `k_eff_items` **and** `v_eff_items` (one per slot
  each — `k_eff_items` used as bind keys and as premise-(iv)'s per-entity
  key read; `v_eff_items`, gathered at `prev_slot = π^{h-1}(a)`'s slot
  [not `tgt_slot`, per the off-by-one-hop fix above], as each query's
  `v_eff_target` scoring target, and as premise-(iv)'s per-entity value
  read).
- **Forward-B (bind+query, captures `q_eff` in situ).** The BIND phase and
  the query clause **concatenated into one sequence**
  (`T_bind + query_len`, trivially `≥128` since `T_bind` alone already is,
  §4.1), run through every layer, with `q_eff` extracted via a **new**
  per-layer `q_conv1d` hook (Rev 5, attack-round-5 FATAL — retargeted from
  the erroneous `k_conv1d` hook this design used through Rev 4, see the
  FATAL box above and §13.5) at the query's own final position. This is a
  genuine, disclosed deviation from `grammar_rd.py`'s own training-time
  convention that "queries NEVER enter the streamed sequence" (§4.2) —
  justified because (i) this is an **eval-only** readout, never a training
  step, and (ii) `S_T` is taken from forward-A alone, which the appended
  query in forward-B **cannot retroactively affect** (a later token in a
  causal model cannot change an earlier state) — the state-freeze
  discipline is preserved exactly where it matters (the state that gets
  matrix-powered), even though the *query's own* extraction now
  legitimately sees the full bind context ahead of it, which is in fact
  the more faithful reading of "how would this checkpoint's own
  multi-layer processing actually represent this query," not a weaker one.
- **The `q_conv1d` hook — a ~5-line addition, the exact mirror of Rev 4's
  `v_conv1d` hook (Rev 5).** `lm_pretrain_rd.py`'s `DeltaNetLMMixer`
  already instantiates `self.q_conv1d = ShortConvolution(...)` at L791
  (verified directly, §0) and calls it at L837
  (`q, _ = self.q_conv1d(self.q_proj(x))`) — inside the exact forward pass
  Option 1 already runs as forward-B. The fix is a direct copy of
  `capture_raw_keys`'s own hook-registration pattern
  (`lm_attractor_probe_rd.py` L163-170 — the same pattern Rev 4's
  `v_conv1d` hook already copied), retargeted to `blk.mixer.q_conv1d`:

  ```python
  def make_q_hook(i):
      def hook(module, inp, out):
          q_raw = out[0] if isinstance(out, tuple) else out
          captured_q[i].append(q_raw.detach())
      return hook

  q_handles = [blk.mixer.q_conv1d.register_forward_hook(make_q_hook(i))
               for i, blk in enumerate(model.blocks)]
  ```

  registered on forward-B (not forward-A — the query token only exists in
  forward-B's sequence), read at the query clause's own final position,
  per layer. Adds no new mixer call and no measurable wall-clock delta:
  forward-B is already one of the two-forward protocol's two priced calls
  (§10) — this only adds a second hook registration and a second small
  tensor capture to a call that already runs, exactly as Rev 4's
  `v_conv1d` hook added no cost to forward-A.
- **`q_eff` is thereby defined IN SITU, honestly.** Unlike the from-scratch
  harness's raw-vector convention (query and bind key are architecturally
  guaranteed identical, `model_dn.py`), a general multi-layer pretrained
  LM's query representation is whatever ITS OWN per-layer conv processing
  of the query clause, in the full bind context, actually produces — this
  is registered as the honest, well-defined choice for every multi-layer
  checkpoint (rung-2/3, `n_layers∈{16,22}`), not an approximation of some
  other "true" `q_eff` this design could have extracted more cheaply.
- **Stage −1 causality assertion (new, §9 item 11; tolerance corrected
  Rev 6.1 — an independent audit measured the real number rather than
  trusting the "bit-identical" claim below.)** Forward-A and forward-B
  **must** produce residual streams over the shared BIND-phase prefix that
  agree to **≤`1e-6` in fp32, on CPU** — **bit-identity is not achievable
  across different-length conv calls**: forward-A's `ShortConvolution`
  call sees only the `T_bind`-length bind sequence, forward-B's sees the
  longer `T_bind+query_len` concatenation, and `torch.nn.Conv1d`'s
  internal reduction order is not guaranteed identical across two calls
  with different total lengths even over the shared causal prefix (a
  floating-point non-associativity effect, not a causality violation —
  the VALUES are correct to float precision, just not bit-for-bit). An
  independent audit measured the real CPU divergence directly: **≈`2.4e-7`
  max abs diff**, comfortably inside the `1e-6` tolerance this design's
  own items 2/5 already use elsewhere — the earlier "bit-identical (fp32,
  CPU)" framing below overclaimed a stronger guarantee than the operator
  actually provides. Tolerance-pinned (`1e-6`) is therefore the ONE
  registered CPU criterion (not "bit-identical, or 1e-6 at the bf16/GPU
  boundary" — `1e-6` applies uniformly, CPU or GPU) — the mechanical proof
  that appending the query in forward-B never leaks backward into the
  captured `S_T` is unaffected: `2.4e-7` is six orders of magnitude below
  any threshold this design conditions a decision on.
- **Cost, honestly re-derived (§10).** Every episode this readout scores
  now costs **two** full mixer forward passes, not one — the existing
  per-pass GPU-h anchors (§10) are measured **forward-pass-only** (a
  single call), so this doubles the per-pass rate for every row that runs
  Option 1, not a cost this design can absorb by construction. §10 prices
  this honestly and re-derives the committed grid under it.

**Continuous, never argmax over a codebook — satisfies the CLAUDE.md hard
rule directly**, unchanged from Rev 2.

**Justification for choosing Option 1 as primary (unchanged in substance
across Rev 3→Rev 5):** it is (a) the direct continuation of this project's
own validated readout formula, (b) the only option satisfying CLAUDE.md's
exact-continuous-recovery rule without any enumeration-based scoring, and
(c) still cheap to build correctly — forward-A reuses the existing
`k_conv1d` hook and adds the new (but trivial, ~5-line, same-pass,
zero-extra-GPU-cost) `v_conv1d` hook (Rev 4); forward-B uses a new (Rev 5,
same ~5-line, same-pass, zero-extra-GPU-cost) `q_conv1d` hook for `q_eff`
— corrected from Rev 1-4's erroneous reuse of `k_conv1d` there, §13.5.
Option 2's cost is a full model forward pass
anyway (already needed to obtain the natural next-token logits) so it
remains nearly free to also report — hence still co-registered, not
deferred.

**Option 2 (SECONDARY, co-registered, complementary not substitutive) —
fully natural, black-box next-token logit margin.** Run the *entire*
pretrained model (all layers, residual stream, tied LM head) forward over
BIND clauses + query clause as one ordinary continuous sequence, no hooks,
no truncation. **Logits are computed at the query position only** (a single
row of the LM head's output, gathering just the true-answer and `K−1`
distractor entries) — **never the full-sequence LM head over every
position**, which would materialize the standing-known 50,257-vocab logits
tensor at every token, the VRAM bottleneck `CLAUDE.md`'s own hard rule names
("The 50K vocab logits tensor is the VRAM bottleneck, not the model
activations") and this project has hit before (Rev 2, attack-round-2 minor).
Score `margin = logit(true_answer_token) − max_j(logit(distractor_j))` over
the `K−1` other in-context entities, continuous, no argmax needed for the
*metric* (only the distractor set is enumerated, the score itself is a
real-valued margin). **Confound, stated
plainly:** this reading can only cleanly realize `h` as "how many BIND
clauses chain into the query's target *as literally written in the text*,"
which is a related but distinct construct from Option 1's external
`S^h`-applied-to-a-fixed-one-hop-accumulation — it mixes hop depth with
"how much context precedes the query," which Option 1 by design does not.
Reported alongside Option 1 at every cell as a robustness/generalization
cross-check (does a fully natural forward pass echo the hooked reading),
**never used alone to license an H_LINK-A/B claim.**

**Option 3 (named, not built this phase) — patched-forward continuation.**
Feed `pred(a,h)` from Option 1 back through the block's own `o_proj` →
residual → later layers → LM head (an activation-patching-style
intervention), producing a genuinely natural logit-margin reading of the
*hooked* composition. More invasive (requires deciding which residual
stream position to patch and whether later-layer nonlinearities sensibly
consume a substituted vector) and more expensive to build correctly.
**Registered as a follow-on only if Option 1 shows a real effect worth
independently cross-validating** — not part of the Phase-1 mandatory grid.
(Option 1's own "why primary" justification now lives with its revised
description above, Rev 4.)

### 4.5 No trained probe — why the probe-leakage/memorization apparatus dissolves, and what survives (REWRITTEN, Rev 3 — attack-round-3 FATAL-1 consequence, see §13.3; target object reverted `k_eff`→`v_eff` at Rev 4 without reopening any of this, see §13.4)

**What FATAL-1's fix removes, structurally, not by choice (HOLDS unchanged
at Rev 4 — only the compared-against object's NAME changed, not its
dimensionality or its zero-parameter status).** §4.4's readout fits **zero
learned parameters** — `pred(a,h)` and every `k_eff`/`v_eff`/`q_eff` come
directly from the checkpoint's own frozen forward pass, and the cosine
comparison against the target's own `v_eff` (Rev 4) is a closed-form
computation, never a fitting step. Rev 1/Rev 2's entire probe-training
apparatus — the PRIMARY (arm-blind shared probe) and SECONDARY (per-arm
probe + Hewitt & Liang control-task null) protocols, the probe-train/
probe-eval episode-seed split, and the heldout-pool memorization-ceiling
control built to catch a FITTED model leaking eval-time information — has
**no object left to attach to**. There is no probe to leak into, no probe
capacity to worry about "under-reporting" a genuinely different-geometry
arm (attack-round-0 item 3, §13), and no probe weights whose training-pool
exposure needs a held-out-entity check. All of it is obsolete and removed
here, not merely deprioritized.

**What survives, unchanged in role, rebuilt probe-free.**

- **The 107-name entity pool, still the standard episode source.** Rev 1's
  F1 fix (§13.1) — draw every episode's `K` entities from the **full**
  107-name `train_name_ids` pool, never a further sub-split — still holds,
  for the ORIGINAL arithmetic reason (`sample_batch_rd` hard-asserts
  `N_names ≥ K`, and no swept `K` up to 96 fits a smaller sub-pool, §4.2).
  This was never actually about probe-training/probe-eval leakage in the
  first place — F1's own fix was itself an entity-pool-ARITHMETIC
  correction, and it is retained on that basis alone. There is now only
  **one** episode purpose drawing from this pool (the headline eval grid,
  §4.6's `eval` purpose, replacing the obsolete `probe_train`/`probe_eval`
  split with a single stream).
- **The label-shuffle null, rebuilt probe-free (§8.4's chance floor).**
  Generate episodes with the bind-clause `(key, value)` pairing shuffled
  across episodes (surface form, entity pool, and token statistics held
  fixed — unchanged mechanism, §8.4), then run the **identical** Rev 3
  readout (forward-A/forward-B, cosine against the shuffled episode's own
  `tgt_slot`-gathered `k_eff`) — no probe to retrain on shuffled labels,
  no separate "null probe" fitting step at all. The null is simply this
  design's one deterministic readout applied to episodes whose true
  compositional structure has been destroyed; §8.4's per-h chance-floor
  role is unchanged, only simpler to build (one fewer moving part).
- **The 106-name heldout pool.** No longer reserved for a memorization
  control (below) — available, if ever wanted, as a plain robustness
  cross-check (a second, disjoint entity source for the SAME probe-free
  readout), but not committed to any Phase-1 cell.

**Heldout-pool memorization-ceiling control — REMOVED, adjudicated, not
silently dropped.** Rev 2's MAJOR-4 fix priced this control (§10, 3.79
GPU-h) at every killer-prediction and Leg-B primary near-cliff-K cell,
specifically to catch a TRAINED PROBE's weights showing an identity-
correlated recoverability differential between arms/rungs. Adjudicated
this revision: **it no longer measures anything, and is removed.** Two
independent reasons, both verified directly against this codebase this
revision, not asserted:

1. **The mechanism it existed to catch cannot occur.** With no probe
   fitting step (above), there are no weights that could encode a
   train-pool-identity-correlated shortcut for the control to expose.
   Every episode, from either pool, is scored by the identical,
   already-frozen checkpoint machinery.
2. **Even repurposed as a bare robustness check, it would show nothing
   informative.** `build_entity_pools()`'s train/heldout split
   (`grammar_rd.py` L221-226, verified directly) is a **pure random
   shuffle** (`random.Random(seed).shuffle` over the candidate list, then
   sliced) — uncorrelated by construction with any property (pretraining
   frequency, BPE structure, embedding quality) that could make the two
   pools differ systematically. Two random subsets of the same curated
   single-token-name distribution give this design no "embedding-
   familiarity" axis to measure a differential against.

Its entire budget line (§10, 3.79 GPU-h) is freed; the **MEMORIZATION-
CONFOUND** outcome category (§8.5, old item 5) is removed with it — there
is no longer a control whose firing could demote a cell to that outcome
(§8.5 renumbered accordingly).

**Attack-round-0 item 3 dissolves (§13, recorded there and at §13.3).**
"Does the shared, arm-blind probe have enough capacity to decode a
stabilized checkpoint's geometry" presupposed a fitted probe with a
capacity ceiling. With no probe at all, there is no capacity axis for a
different-geometry arm to be under-reported through — the question does
not survive Rev 3's fix, the same way attack-round-0 item 4 stopped
applying once Rev 1 built the buffer-blank-out test it asked for (§13.1).

**§7.4 (Probe-leakage) is updated to match** — the shortcut it was written
to close no longer has a channel to occur through at all, not merely a
better-defended one; see §7.4's own Rev 3 text.

### 4.6 Episode-seed allocation — exact numeric ranges, non-collision by construction (Rev 2, attack-round-2 MAJOR-5)

**Finding fixed:** §4.5's disjoint-episode-seed-range claim (F1's own fix,
Rev 1) was stated as "a registered, non-overlapping RNG-generator-seed offset
per role — e.g. ... a disjoint offset block" — an example, not a pinned
number, and it never accounted for the OTHER seed axes already in play (3
checkpoint seeds × 2 corpora × 3 arms/4 rungs × up to 6 K values), any pair
of which could silently collide on the identical `torch.Generator` seed
without a stated allocation scheme to rule it out.

**Fixed allocation, one flat integer formula, no hashing, human-checkable by
inspection:**

```
episode_seed = PURPOSE_BASE[purpose]
             + LEG_BASE[leg]
             + condition_idx * STRIDE_CONDITION   # arm (Leg A, 0-2) or rung (Leg B, 0-3)
             + corpus_idx    * STRIDE_CORPUS       # 0-1
             + ckpt_seed_idx * STRIDE_SEED         # 0-2 (the checkpoint's OWN training seed)
             + k_idx         * STRIDE_K            # 0-5, over the registered K's {20,32,40,48,64,96}

PURPOSE_BASE = {"eval": 0, "null_shuffle": 10_000_000, "calibration": 20_000_000}
# Rev 3 (attack-round-3 FATAL-1): "probe_train"/"probe_eval"/"heldout_control"
# removed -- no probe fitting step exists to separate from scoring (§4.5); one
# flat "eval" purpose replaces the train/eval split.
LEG_BASE     = {"leg_a": 0, "leg_b": 5_000_000}
STRIDE_CONDITION = 1_000_000   # max 4 conditions used (Leg B rungs) -> uses <=3,000,000 of the 5,000,000 leg block
STRIDE_CORPUS     = 100_000    # max 2 corpora -> uses <=100,000 of the 1,000,000 condition block
STRIDE_SEED       = 10_000     # max 3 seeds  -> uses <=20,000 of the 100,000 corpus block
STRIDE_K          = 1_000      # max 6 K's    -> uses <=5,000 of the 10,000 seed block
```

**Why this cannot collide, by construction (not merely by convention):**
each stride is set larger than the maximum possible sum of every FINER
stride below it (`STRIDE_K × 6 = 6,000 < STRIDE_SEED = 10,000`;
`STRIDE_SEED × 3 = 30,000 < STRIDE_CORPUS = 100,000`;
`STRIDE_CORPUS × 2 = 200,000 < STRIDE_CONDITION = 1,000,000`;
`STRIDE_CONDITION × 4 = 4,000,000 < LEG_BASE` spacing of `5,000,000`; and
every `LEG_BASE`/`PURPOSE_BASE` combination tops out at
`5,000,000 + 4,000,000 = 9,000,000`, under the `10,000,000` spacing between
`PURPOSE_BASE` entries) — this is the standard mixed-radix/positional-
numbering guarantee, not an empirical claim requiring its own proof, but
still checked mechanically (below) per this project's "don't trust a
guarantee you can cheaply verify" convention.

**Per-purpose usage (Rev 3 — `probe_train`/`probe_eval`/`heldout_control`
removed with §4.5's fix; one flat `eval` purpose replaces the obsolete
train/eval split since there is no longer a fitting step to separate from
scoring).** `eval` uses the full `(condition, corpus, seed, K)` product for
every committed and named-extension cell in Leg A's and Leg B's grids
(§4.3/§10) — the single stream every headline and extension episode is
drawn from. `null_shuffle` is computed at Stage 0's single calibration cell
only (§9) — one fixed `(condition=off, corpus=index-0, seed=0, K=32)`
point, scored at all 4 h's, using the SAME formula (only one combination is
ever actually instantiated). `calibration` covers Stage 0 (14M mixcontrol)
and Stage 0.5 (§9, rung-3 pass-cost timing) as two disjoint sub-offsets
within the one `calibration` purpose block (Stage 0 at `condition_idx=0`;
Stage 0.5 at `condition_idx=3`, `leg_b`, `k_idx` for K=64 — no
special-casing needed, the same formula covers both).

**Stage −1 assertion (new item 10, §9): non-collision check.** Enumerate
every `(purpose, leg, condition_idx, corpus_idx, ckpt_seed_idx, k_idx)`
combination that the registered committed grid (§4.3, §10) and named
extensions actually instantiate, compute `episode_seed` for each via the
formula above, and assert `len(set(seeds)) == len(seeds)` — pure Python
arithmetic, zero-GPU, catches a construction error (e.g. a copy-pasted
stride) even though the formula is collision-free by the positional-numbering
argument above; per this project's own "run the negative test to completion,
don't just write it" discipline, this assertion is exercised, not merely
argued for.

---

## 5. LEG A — intervention → reasoning

### 5.1 Checkpoints in scope

All 20 archived training cells' **final** (step-20,000) checkpoints:
18 core (2 corpora × {off, per_token λ=0.58, global λ=0.58} × 3 seeds) +
optionally the 2 λ-mini-sweep cells (λ=0.3/0.8, per-token, openr1 only,
seed 0) as a **secondary, n=1, non-headline** λ-dependence cross-check
(mirroring the mini-sweep's own house-registered non-headline status in
`FROZEN_BIAS_LM_DESIGN.md`). A **trajectory** read (a handful of the 20
archived per-1000-step checkpoints per cell, e.g. steps
{2000,5000,10000,15000,20000}) is registered as an optional richness
extension, priced separately in §10, not required for the Phase-1 headline
(mirroring the scale-transfer harvests' own final-checkpoint-first, then
trajectory-as-bonus convention).

**Framing note (Rev 4, attack-round-4 MAJOR, folded in explicitly per §1's
adjudication):** every comparison in this section (global-vs-off,
per-token-vs-off, the §5.3 killer prediction) is read as an **arm
CONTRAST** through Option 1, never as a claim that Option 1's absolute
`recovered_frac@0.9` level is itself a direct measure of "how much
composition this checkpoint can do." Option 1 tests one a priori unlikely
mechanism (single-layer state self-iteration, §4.4); a checkpoint's
absolute Option-1 level can sit near floor at `h≥2` for reasons having
nothing to do with the frozen-bias intervention (e.g. this checkpoint
family composing via cross-layer chaining instead, or a low premise-(iv)
cross-role-identity reading, §4.4) while STILL showing a genuine,
informative arm-vs-arm delta if the intervention shifts how much signal
routes through single-layer self-iteration specifically. This is why
§5.2's CONFIRM criterion is stated as a delta's CI excluding zero, never
as an absolute-recovery threshold, and why a uniform `h≥2` floor across
every arm here routes to READOUT-FORM-INVALID (§12), not silently to
REFUTE.

### 5.2 Comparisons and pre-registered reading

Primary: **global-vs-off** and **per-token-vs-off**, each `Δ =
recovered_frac@0.9(arm) − recovered_frac@0.9(off)`, held-out hops (h=3,4),
pooled or reported per-K per §5.3, pinned CI `t(2,.975)=4.303` over 3 seeds,
per corpus (2 corpora reported separately, never pooled, matching every
prior wave's own discipline). **Rev 1 (attack-round-1 F2):** `Δ` above is
computed from the **training-effect** contrast, not the raw native forward
pass — see §5.2a, which this Δ now routes through by construction.

**Pre-registered reading (matches the task brief's own framing exactly, no
reinterpretation):** global ≥ off ≥ per-token on held-out-hop `rec@0.9`, at
matched val-loss (already established equal, §7.2's val-loss gate — reused
here as a standing control, not re-derived). CONFIRM requires the
global-vs-off delta's CI to exclude zero on the positive side (better
recovery) in at least one corpus, without per-token showing the mirror-image
positive delta in the same corpus (which would indicate a metric artifact
common to any bias arm, not a stabilization-specific effect — the exact
kind of "control doesn't distinguish the arms" pattern
`FROZEN_BIAS_LM_DESIGN.md`'s own §7.1a licensing check already worries
about and this design inherits the discipline of, not the specific test).
**Rev 1 (attack-round-1 M1) — additional CONFIRM requirement:** CONFIRM at
the killer-prediction cell (§5.3) additionally REQUIRES Option 1 (§4.4,
`S_T^h·q_eff` cosine readout against the target's own `v_eff`, Rev 4) and
Option 2 (§4.4, natural next-token
logit-margin) to **agree in direction**. Operationalized: construct
Option 2's own global-vs-off margin-delta CI the identical pinned way
(`t(2,.975)=4.303`, 3 seeds), at the SAME (K, corpus) cell. Agreement =
Option 2's CI does not exclude zero on the NEGATIVE side while Option 1's
CI excludes zero on the positive side (no direct contradiction).
Disagreement = Option 2's CI positively excludes zero on the NEGATIVE side
at a cell where Option 1's CI excludes zero on the POSITIVE side — a
directly contradictory result. **On disagreement, the cell is NOT reported
as CONFIRM** — it routes to a new, pre-registered THIRD outcome,
**READOUT-DIVERGENCE** (distinct from CONFIRM / REFUTE / probe-invalid):
either (a) `pred(a,h)=S_T^h·q_eff` is not the right composition object for
a checkpoint never trained under a hard β-mask (§13 item 2, the open
question this exact disagreement would resolve empirically), or (b) the two
readouts are tracking genuinely different mechanisms. Both are reportable
findings in their own right, never silently resolved by picking whichever
readout supports the hoped-for direction.

### 5.2a Isolating training-effect from mechanical blend-effect (Rev 1 fix, attack-round-1 F2)

`DeltaNetLMMixer.forward` (`lm_pretrain_rd.py` L854-857) applies
`apply_frozen_bias_blend` **unconditionally** whenever
`self.frozen_bias_arm != "off"` — including during the probe's OWN eval
forward pass over a BIND-phase episode, not merely at training time. The
pre-attack §5.2 comparison (arm ckpt's native forward vs. off ckpt's native
forward) therefore confounded two distinct effects: **(i)** whatever the
frozen-bias TRAINING PROCESS did to the checkpoint's learned weights (the
effect H_LINK-A is actually about), and **(ii)** the mechanical fact that
the arm checkpoint's own eval-time forward pass re-applies the blend live to
the probe's own bind-clause keys, regardless of what training did. §7.9's
pre-attack defense addressed only the hook's LOCATION (code-path identity
across arms), not this.

**The fix reuses an existing, established toggle pattern.**
`self.frozen_bias_arm` is a plain Python string instance attribute
(`lm_pretrain_rd.py` L741), not a buffer or parameter, so it can be
overwritten after `load_state_dict` without touching the checkpoint's own
weights — a one-line surgery (`for blk in model.blocks: blk.mixer.frozen_bias_arm
= "off"`). This is the same "the blend is a separable, disable-able stage"
fact `frozen_bias_retrofit_eval_rd.py`'s own `mode` dispatch already
establishes: it hooks `k_conv1d`'s output (strictly BEFORE the blend inside
`forward`, so always the pre-blend raw key regardless of what
`frozen_bias_arm` the loaded model carries) and then CHOOSES whether/how to
reapply the blend as a separate external step (`"kraw"` = no blend,
`"arm1prime"`/`"arm1double"` = blend reapplied externally) — proven
code-path-identical to the model's own live blend by that tool's own smoke
(`frozen_bias_retrofit_eval_rd.py` L223-274, `torch.equal` assertion). This
design's surgery is the same fact applied as an in-forward-pass toggle
rather than a post-hoc external reapplication.

**Registered 2×2 grid (Rev 2, attack-round-2 MAJOR-2: all 4 cells are
constructible; the 4th is DEFERRED ON BUDGET, not absent):**

| | blend ON at eval (native forward, or a retrofit-applied blend) | blend OFF at eval (surgery: `frozen_bias_arm` forced to `"off"`) |
|---|---|---|
| **off-arm checkpoint** | blend forced ON via a seed-derived frozen table (`frozen_bias_retrofit_eval_rd.py`'s Arm-1′/Arm-1″ modes, below) — **constructible, DEFERRED on budget** | = blend OFF (the off arm never trains a `frozen_bias_table` — there is nothing to blend) — the one cell, reported once |
| **arm checkpoint (global or per_token)** | pre-attack §5.2 reading, mechanically confounded per §5.2a's own F2 finding | training-effect cell, §5.2a's core fix |

**Rev 1 error, corrected here:** Rev 1's table labeled the off-arm/blend-ON
cell "not constructible — no table exists to force on; inventing one would
compare against a table the off arm never trained against." This is
factually wrong, and the tool proving it wrong already exists in this
repo: `frozen_bias_retrofit_eval_rd.py`'s `run_retrofit_measurement` (its
own `arm1prime`/`arm1double` modes) builds `table =
build_frozen_bias_table(vocab_size, d_state, seed=frozen_bias_seed)` (default
`seed=ANCHOR_INIT_SEED`) **fresh, from the seed alone**, for WHICHEVER
checkpoint path is passed in — nothing about this construction reads or
requires any training history with that table. It is already run against
non-training-with-the-table checkpoints today (that is precisely what
"retrofit" means in the tool's own name and docstring: "capture ...
`k_raw`, then apply `apply_frozen_bias_blend`" to a checkpoint's raw keys
post-hoc). Feeding an "off" checkpoint's captured `k_raw` through the
identical `apply_frozen_bias_blend` call with a seed-derived table is
code-path-identical to what the tool already does for its Arm-1′/1″
measurements — there is no "off arm never trained against this specific
table" objection, because the retrofit tool's entire point is that the
table's origin (a fixed seed, not a training run) makes it applicable to
ANY checkpoint's raw keys, trained-with-blend or not.

**Corrected registration: the 4th cell IS constructible, DEFERRED ON
BUDGET.** Cost (Rev 3 re-priced — committed killer K's now `{20,32}`, §4.3,
at the two-forward rate, §10): 2 corpora × 3 seeds (off arm only) × 2 K =
12 passes ≈3.34 GPU-h (§10 extension table, not part of the Phase-1
committed total — the margin, §10, does not accommodate it this phase). **What it would isolate, registered now so it is not
re-derived under pressure later:** whether the MECHANICAL blend effect
(§5.2a's own mechanical-effect contrast, arm-ckpt blend-ON vs. blend-OFF)
requires prior TRAINING under that blend to manifest, or whether cold-
applying the identical blend mechanism to a checkpoint that never trained
with any table produces the same (or a materially different) shift —
i.e., it decomposes "the blend mechanically helps regardless of training
history" from "the blend only helps because training adapted the
surrounding weights to expect it." **Trigger condition for running it:** if
the mechanical-effect contrast (§5.2a) is itself significant (CI excludes
zero) at the killer-prediction cell, the 4th cell is the registered
follow-up that decomposes that finding — run then, as a targeted
strengthening pass, not before, since it is only informative once there is
a mechanical effect worth decomposing.

**Pre-registered interpretation, mandatory:**
- **training-effect** (the contrast H_LINK-A's CONFIRM/REFUTE actually rides
  on, per §5.2's Δ above) = `rec@0.9`(arm ckpt, blend-OFF) −
  `rec@0.9`(off ckpt) — isolates what frozen-bias TRAINING did to the
  weights, with the mechanical blend disabled symmetrically on both sides.
- **mechanical-effect** (reported separately, alongside, never load-bearing
  for CONFIRM) = `rec@0.9`(arm ckpt, blend-ON) − `rec@0.9`(arm ckpt,
  blend-OFF) — isolates what the blend does MECHANICALLY to the SAME
  trained weights, holding training fixed.

**Stage −1 self-test (§9 item 6 — AS BUILT, Rev 6.2 reconciliation after
audit #2):** the implemented test drives the probe's OWN production
`frozen_bias_surgery()` context manager through `run_forward_b` with two
non-vacuous controls, both mutation-proven (a forced no-op surgery fails
the test — verified independently by two auditors): (a) live-gate-fires —
with `frozen_bias_arm="global"` the pipeline output differs from
`arm="off"` by more than tolerance (the blend genuinely acts); (b)
surgery-mechanics — the context manager flips and RESTORES the attribute
(including under an exception mid-scope), and forced-off output matches
the never-blended path. This is a stronger production-code-path test than
the originally drafted variant (which compared against
`frozen_bias_retrofit_eval_rd.py`'s external `"kraw"`-mode capture —
byte-for-byte `torch.equal`); the external cross-tool `kraw` equivalence
check is registered as an OPTIONAL follow-on (cheap, the retrofit tool's
`kraw` mode still exists), not a launch gate — audit #2 adjudicated the
substitution as non-vacuous and non-blocking, and this paragraph records
the reconciliation rather than silently retconning the spec.

### 5.3 The killer prediction (K-dependence)

**The single most falsifiable claim this design makes:** arm separation
(global vs. off vs. per-token) is **not uniform across K** — it is
predicted to concentrate at K∈{32,48} (near and past the located `x0≈34.9`
cliff for `d_state=64`) and be small-to-absent at **K=20** (well inside the
"capacity is not the thing that fails first" regime,
`DELTANET_REALDATA_DESIGN.md` §16.4 — **Rev 2, attack-round-2 FATAL:** this
replaces Rev 1's `K=8` low-load point, which crashes `chunk_delta_rule` at
this checkpoint family's `conv_size=4` [`T_bind=56 < _MIN_KERNEL_T=128`,
§4.1/§4.3]; `K=20` is the new floor-safe low-load contrast, load
`20/64=0.3125`, still comfortably below the cliff). **Rev 3 (attack-round-3
FATAL-2 budget consequence, §10):** `K=48` is demoted from committed to a
named, priced extension — the two-point scientific prediction still names
both `{32,48}`, but only `K=32` is **committed** for Phase-1 execution
(the point closest to the cliff center, already the sole primary
agreement-gate K per the tie-break rule below); `K=48` reports as
corroborating evidence if/when its extension is run. Operationalized:
report `Δ(K)` (the global-vs-off **training-effect** delta, §5.2a) at
every tested K separately; the **committed** killer-prediction pass
condition is `|Δ(K=32)| > |Δ(K=20)|` with `K=32`'s CI excluding zero while
`K=20`'s does not, replicated in at least one corpus, AND (Rev 1, M1)
Option 1/Option 2 directional agreement per §5.2 at `K=32`; if `K=48`'s
extension is run, `|Δ(K=48)| > |Δ(K=20)|` is checked as the same
corroborating condition, never substituting for the `K=32` committed
reading. This ties the readout **directly** to the capacity law rather
than merely re-measuring "does an intervention move some LM metric" — a
failure of this specific prediction (arm separation flat across K, or
concentrated at *low* K) would be a genuinely informative negative about
whether the capacity cliff has any behavioral teeth at all, distinct from
whether the intervention has *any* effect. **Rev 1 mandatory zero-cost
covariate (M1):** at every headline cell (this one included), also report
`S_T`'s condition number / eigenvalue spread and a within-episode cross-`a`
cosine-convergence check — see §8.3.

**Rev 2 (attack-round-2 minor M1) — multi-K conflict rule for the Option
1/Option 2 agreement gate.** With `K=32` committed and `K=48`/`K=40`
available only as named, priced extensions (Rev 3, §10), the agreement gate
still needs a single, pre-registered tie-break rather than an implicit
"whichever K looks best" choice. **Registered rule (unaffected by the Rev 3
budget demotion, since it already named `K=32` the sole primary K):** the
killer-prediction verdict (CONFIRM/REFUTE/READOUT-DIVERGENCE) is read at
**`K=32` as the single primary agreement-gate K** (the point closest to the
located `x0≈0.5455` cliff center, `0.5455×64≈34.9` — the most
externally-anchored of the swept points, §8.6's own exemption logic).
`K=48` and (if run) `K=40` report their own Option 1/Option 2 agreement and
`Δ` alongside, as corroborating or complicating evidence, but **never
override** the `K=32`-based primary verdict — a disagreement at `K=48` while
`K=32` agrees is reported as a registered open question about where exactly
the effect concentrates, not a reason to withhold or flip the primary
reading, and vice versa.

### 5.4 Statistics

3 seeds per (arm, corpus, K) cell — pinned CI formula throughout, no new
formula introduced. λ-mini-sweep cells (n=1) reported as point estimates
only, explicitly non-headline, matching `FROZEN_BIAS_LM_DESIGN.md`'s own
treatment of its own n=1 λ-sweep.

---

## 6. LEG B — scale → reasoning

### 6.1 Checkpoints in scope, stratified by `d_state`

| Rung | Params | `d_state` | Corpora × seeds | Checkpoints | Statistical power |
|---|---|---|---|---|---|
| 14M mixcontrol | 14,048,896 | 64 | 2 × 3 | 6 final | full pinned-CI (n=3) |
| 98M wave-1ext | 97,618,176 | 64 | 2 × 3 | 6 final | full pinned-CI (n=3) |
| 392M rung-2 | 391,869,440 | 128 | 2 × 3 (per §5.6/§5.10) | 6 final | full pinned-CI (n=3) |
| 1.31B rung-3 | ≈1.31B | 128 | **2 × 1** (PINNED, Rev 3 — see below), not yet harvested | 2 final (pending ≈2026-07-08) | **descriptive only, no CI** |

**Rung-3 configuration — PINNED this revision (attack-round-3 MAJOR), not
left as a harvest-time TODO.** Rev 2 left this an open confirm-at-harvest
question ("likely 1 seed × 2 corpora, or 2 seeds × 1 corpus"). Verified
directly this revision against the actual launch manifest
(`deltanet_rd/run_lm_rd_trackc_sweep.py::wave23_manifest`/
`WAVE23_SEEDS_BY_RUNG[3]`, L100/L868-869: *"rung 3 stays '2 corpora x 1
seed' as registered"*) and `STATE.md`'s own launch record ("2×1.31B scale
runs", tmux `trackc3`): **rung-3 is 2 EXTENDED-mix corpora
(`openr1-mix-ext`, `wikitext-mix-ext`) × 1 seed each — NOT 2 seeds of one
corpus.** This is not a cosmetic distinction: under "1 seed × 2 corpora,"
**both** corpora get exactly one rung-3 data point each; under "2 seeds ×
1 corpus" (the alternative Rev 2 left open), **one** corpus would have 2
rung-3 seeds and the **other would have zero** — silently truncating that
corpus's own ladder at the 392M rung, breaking §7.8's standing
within-corpus-never-pooled reporting discipline for exactly the rung that
matters most. The verified reality is the better-behaved case: every
downstream table/budget line touching rung-3 (§10, this section) is
written against **1 seed × 2 corpora**, giving each corpus a complete, if
descriptive-only (n=1), 4-rung ladder.

With only 1 seed per corpus, the pinned `t(2,.975)` formula still does not
apply (n=1, not n≥3). Rung-3's contribution to H_LINK-B is a **point
estimate plus qualitative direction only, reported separately per corpus**
(never pooled across the 2 corpora, per §7.8) — consistent with how this
project has already handled every other n=1 cell (the λ-mini-sweep, the H4
single-seed parameter-diff
check) — never a CI-based confirm/refute on its own.

### 6.2 Comparisons and pre-registered reading

Primary: `recovered_frac@0.9` at held-out hops (h=3,4), at the
`d_state`-matched near-cliff K (K=32 for the two `d=64` rungs, K=64 for the
two `d=128` rungs — the closest tested point to each `d_state`'s own
`0.5455×d` cliff-adjacent load; **Rev 2 note:** `K=64` is, this revision, a
COMMITTED d=128 point, §4.3 — Rev 1's own committed d=128 pair `{8,32}` did
not actually include `K=64` at all, a latent inconsistency this revision's
§4.3 fix also resolves), reported across the 4-rung ladder.
**Pre-registered reading:** monotone non-decreasing recovery across
14M→98M→392M→(1.31B, descriptive), and a positive rank correlation
(Spearman) between per-rung recovery and the already-measured span_frac
value at that rung (0.248/0.344/0.389/[pending]) — reusing exactly the
Spearman-ρ convention this project already applied to the Task D
rank-vs-K result (`ρ=1.0`, `STATE.md` Chapter 2 section).

**Rev 4 (attack-round-4 FATAL-ADJACENT) — the mandatory Option 1/Option 2
agreement gate, extended to Leg B.** §5.2's Rev 1 M1 gate (Option 1 and
Option 2 must agree in direction before a CONFIRM is licensed) was, before
this revision, wired ONLY to Leg A's killer-prediction cell — Leg B's own
scale-trend claim above rode on the identical `S_Tʰ·q_eff` readout, the
identical FATAL-1/FATAL (§4.4) concerns about what that readout actually
measures at `h≥2`, and the identical framing-adjudication caveat (§1) that
Option 1 tests one a priori unlikely mechanism — yet had no cross-check of
its own. **Fix, extending §5.2's gate verbatim to Leg B's own primary
cells:** at EACH of Leg B's 4 rungs, construct Option 2's own
scale-related reading (the rung's `recovered_frac@0.9`-or-margin value at
its OWN primary near-cliff K — `K=32` for the two `d=64` rungs, `K=64` for
the two `d=128` rungs, i.e. exactly the cells §6.2 already reads above) at
the SAME (K, corpus) cell, using the same pinned-CI convention where `n=3`
applies (rungs 1/2/3 — 14M/98M/392M) and a point-estimate-only reading
where it does not (rung 3 — 1.31B, per §6.1's descriptive-only status).
**Agreement/disagreement defined identically to §5.2's own rule:**
Option 2's reading does not directly contradict Option 1's own direction
at that rung (same "CI does not exclude zero on the opposite side while
the other's CI excludes zero on its side" test, or, for the descriptive
rung, "point estimates do not point in opposite directions"). **Per-rung
consequence:** a rung where Option 1/Option 2 disagree routes THAT rung's
own contribution to READOUT-DIVERGENCE (§5.2) — it is dropped from the
scale-trend reading, not silently averaged in. **Scale-claim demotion
rule (mechanical, pinned — mirrors the directive's own suggested form):**
a monotone-trend CONFIRM for H_LINK-B may be claimed ONLY over the SUBSET
of rungs where Option 1/Option 2 agree, and requires a MINIMUM of 3
agreeing rungs (of the 4-rung ladder). With fewer than 3 agreeing rungs,
H_LINK-B's scale reading is demoted to **AMBIGUOUS** (§12) — a named,
reportable outcome distinct from CONFIRM/REFUTE/READOUT-DIVERGENCE/
READOUT-FORM-INVALID — rather than a trend claim built on a minority of
the ladder's own rungs. This gate is priced at zero additional GPU cost:
Option 2's reading at each rung's already-committed primary K comes free
from the SAME forward-B pass Option 1 already runs there (§4.4), so no
new row is added to §10's budget.

### 6.3 What Leg B does NOT claim

Because `d_state` changes mid-ladder (64→128 at the 392M rung), a raw
cross-`d_state` recovery comparison at a FIXED K would conflate scale with
absolute capacity change — this design does not make that comparison. The
K=32-vs-K=64 near-cliff comparison above is **K/d-matched**, not
raw-K-matched, precisely to keep the scale claim about the same *relative*
load position at each rung's own capacity regime. A secondary, explicitly
labeled reading (raw K held fixed across all 4 rungs including the two
`d=128` ones) is reported alongside as a "what happens if you don't correct
for `d_state`" sanity contrast, never the headline. **Rev 2 (attack-round-2,
consistency fix):** this fixed-K value is **`K=20`**, not Rev 1's `K=32` —
`K=32` is a committed `d=64` point but only a named EXTENSION at `d=128`
(§4.3), so a `K=32`-fixed comparison is not affordable across all 4 rungs
under the committed Phase-1 budget; `K=20` is used here (loads 0.3125 at
`d=64` vs. 0.15625 at `d=128` — a weaker but still real "same raw K,
different relative load" contrast than `K=32` would have given). **Rev 3
(attack-round-3 FATAL-2 budget consequence, §10):** `K=20` at BOTH
`d_state`'s is itself now a named, priced extension for Leg B, not
committed — H_LINK-B's primary claim (§6.2) never used `K=20` at all, so
this demotion costs the headline nothing, but this secondary sanity check
is unfunded within the Phase-1 committed grid until its extension is run.
If the `d=128` `K=32` extension or the `K=20` extension (either `d_state`)
is ever run, the corresponding richer fixed-K contrast becomes available as
a bonus, reported alongside, not required for the design's primary claims
to stand.

### 6.4 Statistics

Pinned CI (n=3) for the three rungs with full seed coverage; descriptive-only
for rung-3, per §6.1.

---

## 7. SHORTCUT ANALYSIS (mandatory, longest section)

Every way this probe could measure something other than genuine
state-mediated composition, enumerated before any code is written.

### 7.1 Position-decomposition / attention-over-raw-context — closed by architecture, a design strength, not merely asserted

In a full-attention model, "hold K items" is trivially satisfiable via K
*positions*, each at rank 1 (`STATE.md`'s own standing hard rule, the exact
finding that killed the original Task A design). **DeltaNet-family models
close this by construction, not by a task-design trick**: the only channel
carrying information from the BIND phase to the query position is the
recurrent fast-weight state `S_T` (a fixed-size `d_state×d_state` object per
head, per layer) plus a strictly local causal-conv window (`conv_size−1`
previous tokens, §4.1 item (b) — verified at build time, not assumed equal
across rungs). Beyond that window, **there is no mechanism by which the
model can "look back" at raw bind-clause tokens** — unlike a Transformer,
there is no full self-attention over the sequence. This is exactly why
DeltaNet-family is the right testbed for this question (stated as a design
strength, not merely disclosed as a limitation): whatever composition
capability is measured here, if present, is mechanistically forced to route
through the state. **What this does NOT close:** the local conv window
itself (§4.2's randomized clause order is the mitigation for this
sub-case) and the possibility that the "state" pathway trivially memorizes
via some low-rank shortcut analogous to the rank-1-suffices-under-argmax
escape (Nichani, Lee & Bietti, ICLR 2025) — which is precisely why §4.4
registers a continuous cosine>0.9 readout, never an argmax over the K
in-context entities, as the load-bearing metric.

### 7.2 Induction-head bigram shortcuts

A one-hop query (`h=1`) is exactly the induction-head "A...B...A→B" pattern
documented extensively in the mechanistic-interpretability literature —
expected to be within reach of a pretrained LM regardless of any
state-mediated binding claim, which is why h=1 is registered as a **sanity
floor**, not evidence for H_LINK (§4.3, §8.4). The risk this section names:
a *bigram-frequency* shortcut at h=1 specifically (the model completes
"KEY REL" with whatever VALUE it has seen most often follow that bigram in
pretraining, entirely independent of the in-context binding) would inflate
h=1 without implying any real binding. **Mitigation:** every entity is
drawn fresh per episode from a pool of GPT-2-common but semantically
interchangeable single-token names (the same pool `KEY_ANCHORING_DESIGN.md`
already uses without incident), and the relation verb is shared identically
across the in-context bind clause and query — so a bigram-frequency
shortcut would require the SAME (name, relation) bigram to have a strong,
consistent completion prior in the pretraining corpus, which is diluted by
using distinct verbs across episodes (§4.2's relation-verb pool) and by
scoring exact-continuous recovery against the *specific* in-context value,
not the corpus-marginal most-likely completion. A registered self-test
(§9) constructs a hand-built adversarial episode where the bigram-prior
completion and the true in-context value **disagree**, and checks the
probe's own scoring code correctly distinguishes them — this tests the
scorer, not the model, but is a necessary precondition for the h=2–4
results to be interpretable as more than "the model ignores context and
free-associates."

### 7.3 Token-frequency confounds

Per the standing H5 discipline (`FROZEN_BIAS_LM_DESIGN.md` §12.2/§12.4:
"any apparent... effect could be driven entirely by a handful of extremely
high-frequency tokens... rather than a genuine broad-vocabulary... effect"),
this design's entity pool is drawn from `grammar_rd.py`'s already-curated
name/noun list, which was itself selected to avoid the highest-frequency
degenerate BPE fragments (verified single-token, ordinary proper
nouns/common nouns, not punctuation or sub-word pieces). Still required:
stratify `recovered_frac@0.9` by in-episode entity frequency band
(top-quartile vs. bottom-quartile by corpus unigram frequency, reusing the
exact H5 stratification pattern) as a standing diagnostic at every headline
cell, not assumed away by pool curation alone.

### 7.4 Probe-leakage (REWRITTEN, Rev 3 — dissolved by FATAL-1's fix, not merely better-defended)

Rev 1/Rev 2 addressed this shortcut by defending a **fitted** probe's
hygiene (arm-blind training, a frozen reference-condition fit, a
label-shuffle control-task null). Rev 3's §4.5 fix removes the fitting step
itself: `pred(a,h)` is scored directly against the target's own `v_eff`
(§4.4, Rev 4 — reverted from Rev 3's `k_eff`, §13.4), a closed-form cosine
comparison with **zero learned parameters**.
There is no probe weight matrix that could be fit on, peek at, or be tuned
toward any arm's or rung's eval numbers — the failure mode this section
exists to close ("you just fit a probe until it found what you wanted")
has **no mechanism left to occur through**, not merely a well-guarded one.
**What still matters, restated for the fitting-free readout:** every
threshold this design uses (the 0.9 cosine bar, the per-h chance floors,
the alignment/agreement rules) is still pre-registered in this document
**before** any checkpoint's real numbers exist (§8, §9) — the discipline
`FROZEN_BIAS_LM_DESIGN.md`'s own blind-pin was supposed to have is
inherited at the level of *thresholds*, not *probe weights*, since the
latter no longer exist to pin.

### 7.5 Periodicity collapse

Closed by construction: every tested K exceeds `h_max=4` (§4.3), and the
single-Hamiltonian-K-cycle generator (`_permutation_graph`, reused
verbatim) guarantees `π^h` is well-defined and non-degenerate for every
tested h at every tested K — the exact mechanism `NEXT_EXPERIMENT_DESIGN.md`
built to close this trap, inherited unchanged. A config-time assertion
(mirroring `TaskEConfig.__post_init__`) blocks any future config edit from
silently reintroducing a K≤h_max cell.

### 7.6 Buffer/query-marker semantic leakage — the reserved-token adaptation's own risk

§4.1's adaptation (ordinary period as buffer, an ordinary rare token or
short fixed phrase as the query marker) trades away a real guarantee the
from-scratch harness had (zero-pinned, frozen buffer embeddings with
*provably* no encodable position information) for a weaker one (a real,
unpinned, already-trained token embedding, chosen to be maximally generic).
**Named risk:** if the chosen query marker/phrase happens to carry residual
semantic content from pretraining (e.g., a phrase like `" then who ?"`
statistically co-occurring with certain topics/entities in the training
corpora), the model could exploit that correlation rather than the in-context
binding. **Mitigation, registered:** (a) verify at build time that the
candidate query-marker token/phrase's occurrence rate is not skewed toward
any entity in the pool (a frequency-co-occurrence check against the
pretraining corpora, cheap, corpus statistics already computed for the H5
stratification in §7.3); (b) run the identical grid with a **second,
independently chosen** query-marker phrase as a robustness replicate on the
calibration cell only (§9) — if the two markers disagree materially, this
is reported as a marker-specific artifact and the whole design is
reconsidered before the full grid launches, not silently averaged over.

### 7.7 Embedding-table entity-orthogonality confound

The n=107 entity pool is **not** constructed to be orthonormal in any
checkpoint's embedding space — unlike Task D/E's synthetic vectors, these
are real, pretrained word embeddings, and their mutual geometry (near-
orthogonal, correlated, or clustered) is an **empirical property of each
checkpoint**, not a design choice. This could differ across arms (frozen-bias
training could plausibly perturb the *input*-embedding-adjacent geometry,
not just the key-projection geometry the existing `span_frac`/rank
diagnostics measure) and across rungs (larger models may have more
differentiated embeddings for the same 107 words). **Mitigation:** measure
and report each checkpoint's own raw pairwise-cosine Gram statistics over
the 107-entity embedding rows (identical instrument to
`key_anchoring.py::raw_table_conditioning`, already built, zero new code) as
a **standing covariate** alongside every headline cell — if embedding-space
entity coherence itself varies systematically with arm or rung, that is
reported and controlled for (or at minimum disclosed) rather than silently
conflated with the state-mediated binding claim this design is actually
about.

### 7.8 Corpus mismatch — does the probe transfer across the checkpoint's own training corpus

The probe episodes are English-like but topically neutral (proper
names/common nouns + a transitive verb) — neither literally OpenR1-Math
nor WikiText style. **Risk:** a checkpoint trained predominantly on
math-reasoning text (`openr1-mix-ext`) might process this template
differently (better or worse) than one trained on narrative text
(`wikitext-mix-ext`) for reasons having nothing to do with the frozen-bias
intervention — exactly the kind of corpus-content confound
`DELTANET_REALDATA_DESIGN.md`'s Wave 2 found in the *opposite* direction
(reasoning-dense text was *more* truncation-damage-sensitive than narrative
text at matched K). **Mitigation:** report every comparison **within
corpus**, never pooled across corpora (already the standing convention
throughout this project, reused unchanged here) — the corpus axis is a
disclosed, uncontrolled covariate, not something this design claims to
isolate.

### 7.9 Cross-arm parameter/capacity confound

Arm 2 (per-token) carries an extra, frozen `(50257,64)` anchor table
(~12.9MB, never trained, per `FROZEN_BIAS_LM_DESIGN.md` §5) that Arm 1
(off) and Arm 2′ (global) do not carry in the same form. This is a
non-issue for a *loss*-based comparison (already handled by that design's
own artifact-matched Arm 1′/1″ construction) but is a **live question**
for this design's own probe: does the mere presence of extra frozen buffer
memory change anything about how `k_eff` (bind-phase keys, extracted via
`k_conv1d`) or `q_eff` (the query read, extracted via `q_conv1d` — Rev 5,
corrected from this section's own erroneous "conv/`W_k` stage" wording
through Rev 4, which mis-described `q_eff` as sharing `k_eff`'s hook
point; see §4.4/§13.5) are extracted at their respective conv stages?
**Answer, stated honestly: no additional control is needed beyond what
already exists** — each hook point (`k_conv1d` output for bind keys,
`q_conv1d` output for the query, both pre-kernel) is identical in shape
and code path across all three arms (verified directly from
`lm_pretrain_rd.py`, §0's reading list). **`q_eff`'s answer is in fact
stronger than `k_eff`'s (Rev 5): `apply_frozen_bias_blend` is applied to
`k` ONLY** (`lm_pretrain_rd.py` L854-857, verified directly — the
conditional wraps exactly one reassignment, `k = apply_frozen_bias_blend(k,
...)`, never touching `q` or `v`), so `q_eff` (and `v_eff`) are
**blend-invariant by construction** across all three arms, not merely
code-path-identical — the frozen table cannot reach `q_conv1d`'s output at
all, through any arm. For `k_eff`, the frozen table enters the
computation graph via the key-blend arithmetic itself, which is exactly
the quantity under test, not an artifact to be controlled away.

**Rev 1 note (attack-round-1 F2):** the answer above defends only the hook's
LOCATION (identical pre-blend code path across arms) — it does NOT, by
itself, rule out the SEPARATE mechanical confound that the live blend acts
on the probe's own eval-time forward pass for arm checkpoints. That is a
different question, closed by §5.2a's blend-toggle surgery grid, not by
this section's hook-identity argument. Both are required; neither
substitutes for the other.

### 7.10 h1-in-distribution definition ambiguity — this design's own weakest terminological point, flagged not hidden

Every prior composition study in this project (Task E, DELTANET_REALDATA
Wave A/B, F-geo-3) defines "in-distribution hop" as "a hop depth the model
was actually trained on." **These checkpoints were never trained on any
hop depth at all** — the BIND/QUERY grammar never appears in their
pretraining data. Calling h=1/2 "in-distribution" here is a looser,
literature-borrowed claim (induction-head recall is empirically common in
pretrained Transformers and increasingly documented in fixed-state/SSM-family
models too, not a property established for *these specific* checkpoints).
**This is registered as an open weak point, not resolved by this
document** — §12 item 1 returns to it as an attack-round question, and
§8.4's chance-plus-margin floor (derived from a real null distribution, not
assumed a priori) is the concrete mechanism preventing this ambiguity from
silently inflating a headline claim.

---

## 8. Metrics, thresholds, and degenerate/exclusion rules

### 8.1 Primary metric (formula corrected, Rev 3 — attack-round-3 FATAL-1; target reverted, Rev 4 — attack-round-4 FATAL, §13.4)

`recovered_frac@0.9` — fraction of eval episodes with
`cos(pred(a,h), v_eff_target) > 0.9` (§4.4: `pred(a,h) = S_Tʰ @ q_eff_a`,
`v_eff_target` gathered at `prev_slot = _iterate_permutation(succ, a_slot,
hops-1)` from forward-A — **not** `tgt_slot`, per §4.4's off-by-one-hop
fix — both terms `d_state`-dimensional, no probe, no `W_embed` in the
formula anywhere), per (arm-or-rung, corpus, K, h) cell, Option 1 readout
(§4.4), the SAME 0.9 threshold this project's entire composition-study
lineage uses (Task D/E, DELTANET_REALDATA, F-geo-3) — no new threshold
introduced without cause.

### 8.2 Secondary metrics (Rev 3 — per-arm/per-rung probe bullet removed, no probe exists)

- Option 2 next-token logit margin (§4.4), reported alongside, never
  load-bearing alone.
- The rebuilt, probe-free label-shuffle null (§4.5/§8.4), reported as the
  chance-floor cross-check at every registered h.
- Raw entity-embedding Gram statistics (§7.7), reported as a standing
  covariate.

### 8.3 Effective-rank / capacity companion (zero new instrumentation)

At every headline cell, also report `entity_subspace_effective_rank`
(identical instrument to `DELTANET_REALDATA_DESIGN.md` §16.4) on the
captured `S_T`, purely as **corroboration**, not a discriminating result on
its own (mirroring §12.2's own honest H2 caveat in
`FROZEN_BIAS_LM_DESIGN.md`: rank and the primary recovery metric are related
but not identical claims, and a rank result alone cannot substitute for a
recovery result).

**Rev 1 mandatory zero-cost covariates (attack-round-1 M1) — added to every
headline cell, not merely the killer-prediction one.** Both are computed
from the already-captured `S_T` (no new forward passes, hence zero
additional GPU cost):

1. **`S_T` condition number / eigenvalue spread** (largest singular value
   over smallest, and the full singular-value spectrum) — a direct
   structural read on whether the captured state is well-conditioned or
   dominated by a single direction.
2. **Within-episode cross-`a` cosine-convergence check** — the power-
   iteration degeneracy signature: for a fixed episode's own `S_T`, compute
   `pred(a,h) = S_T^h @ q_eff_a` for every source slot `a` in that episode
   at the SAME `h`, and measure the mean pairwise cosine similarity across
   the `K` different `a` values. If `S_T` has one dominant eigenvalue,
   `pred(a,h)` for different `a` converges toward the SAME dominant
   eigenvector as `h` grows regardless of which entity `a` actually is —
   indistinguishable from real composition by `recovered_frac@0.9` alone
   unless checked directly, since a degenerate "everything looks like the
   same answer" collapse could coincidentally score well against a small,
   correlated value-embedding neighborhood. Reported as a rising-with-h
   trend line per headline cell — a cell whose cross-`a` cosine convergence
   rises sharply with `h` while `recovered_frac@0.9` also appears to "pass"
   is flagged for manual inspection before being reported as a clean
   CONFIRM, exactly the same corroboration-not-discrimination discipline
   `entity_subspace_effective_rank` above already uses.

### 8.4 h1 sanity floor — derived from a real null, not asserted

Because §7.10 rules out assuming a fixed numeric floor (e.g. "≥0.98") the
way prior in-distribution-trained studies could, this design instead
constructs an explicit **structural chance null**: for each calibration
cell (§9), generate a matched set of episodes with the bind-clause
`(key,value)` pairing **shuffled across episodes** (surface form, entity
pool, and token statistics held fixed — only which value follows which key
is scrambled), score `recovered_frac@0.9` under the SAME readout (Rev 3:
no probe — §4.5's rebuilt, probe-free null applies this design's one
deterministic hooked-geometry readout directly), and bootstrap a 95% band
for this null. **Registered pass condition:** the
real h=1 recovery on the "off"/reference arm must exceed the null band's
upper edge by a margin at least as large as the null band's own width
(a self-relative, non-arbitrary bar, mirroring the real-floor-vs-synthetic-floor
correction this project already made once in `FROZEN_BIAS_LM_DESIGN.md`
§7.1-real, rather than repeating a fixed-threshold mistake it has already
paid for once). **Failure routes to probe-invalid (§1), not to REFUTE.**

**Rev 1 (attack-round-1 M2): the null band is constructed at EVERY tested h,
not h=1 only.** The pre-attack draft built the label-shuffle null band only
at h=1, while the design's own headline claims live at h=3/4 (§4.3) — a
chance band measured at one hop depth was never established to bound
chance performance at the OTHER, load-bearing hop depths (there is no
argument in this design, nor in the literature it cites, that a fixed-state
model's chance-level recovery is h-invariant). **Fix:** at the Stage-0
calibration cell (§9), construct the identical label-shuffle null-band
procedure independently at EVERY h∈{1,2,3,4} (same shuffled-pairing
episodes, scored at each h separately — no new episode generation beyond
what the calibration cell already runs), yielding a **per-h registered pass
floor**. The h=1 floor remains the probe-validity gate (§8.4's original
role, "failure routes to probe-invalid, not REFUTE"); the h=2/3/4 floors are
carried forward as the actual chance baselines the headline
`recovered_frac@0.9` numbers at those hops must clear before a CONFIRM/REFUTE
reading at that hop is licensed — closing the exact gap named by this
finding.

**MAJOR-2 fix (Rev 5, attack-round-5) — an absolute backstop alongside the
null-relative rule, so a near-zero-vs-near-zero comparison cannot pass by
default.** The null-band-relative pass condition above degenerates toward
"almost any recovery passes" if the measured null band itself sits near
zero — a real risk for a fixed-state model whose chance-level recovery
could plausibly be near-uniform-random across a K-entity pool.
**Registered fix:** `h=1`'s `recovered_frac@0.9` on the "off"/reference
arm must ALSO exceed a registered fixed absolute minimum, **0.10**,
independent of where the null band lands — BOTH conditions (null-relative
AND absolute) must pass; failing either routes to probe-invalid (§1), not
REFUTE. **Justification for 0.10, not an invented number:** this
project's own from-scratch, trained-on-the-exact-task exactness program
runs its h=1 guard at **1.0** (`h1_recovered_frac_at_0.9_final: 1.0`,
`KEY_ANCHORING_DESIGN.md`; `DELTANET_RD_EXACTNESS_DESIGN.md` §16.8) — but
that guard is for checkpoints TRAINED on the BIND/QUERY grammar itself,
which this design's checkpoints (§0, never-task-trained pretrained LMs)
are not, so reusing 1.0 here would repeat exactly the
"in-distribution-trained floor imported into a never-trained-on-the-task
regime" mistake §7.10 already flags as this design's own weakest
terminological point. For never-task-trained checkpoints, the defensible
low bar is instead anchored to the in-context associative-recall
capability already documented in this size class in the literature this
design's own litreview (§2.1) surfaced: Zoology (arXiv:2312.04927)
establishes fixed-size recurrent state as the associative-recall
bottleneck this whole program studies, and Based (arXiv:2402.18668,
Theorem 3.1) proves the same Ω(N)-bits-of-state necessity — both motivate
that a pretrained fixed-state LM in this size class SHOULD recover some
nontrivial fraction of simple one-hop associative bindings if the readout
has any referent at all. **0.10** is registered as that defensible low
bar — well above pure chance at the smallest committed K (`K=20`, chance
`≈1/19≈0.053`), while still low enough not to mechanically fail a
genuinely-capable-but-imperfect checkpoint. Below it, the probe is
declared invalid for that checkpoint family regardless of how the null
band happens to sit.

### 8.5 Degenerate/exclusion rules, with counts (registered categories)

1. **Premise-invalid checkpoints** (Rev 3: probe-training clause removed,
   §4.5 — no probe-training design matrix exists anymore): `k_eff`/`q_eff`
   (bind, query, or target) are near-zero-norm (`<1e-6`) or non-finite.
   Excluded from headline, reported as its own category with an exact
   count, never silently dropped.
2. **h1-floor-failed cells**: §8.4's pass condition fails for that
   (arm/rung, corpus) pair. The entire K/h grid for that cell is reported
   under "probe-invalid," not blended into CONFIRM/REFUTE, exact count
   reported.
3. **Degenerate hop-cycle configs**: `K ≤ h_max`. Hard-blocked at
   config-construction time (assertion), so the count should always be
   zero — reported as zero to make the guard's presence visible, per this
   project's own "run the negative test to completion, don't just write it"
   discipline.
4. **Marker-disagreement flag** (§7.6): if the two-query-marker robustness
   replicate disagrees beyond the registered tolerance at the calibration
   cell, the entire Phase-1 grid is blocked pending a design revision — not
   an exclusion rule applied per-cell, a hard gate before the grid launches.
   **Registered tolerance (Rev 6.1, closing a build-time gap the BUILD
   agent correctly flagged rather than silently inventing): 0.15 in
   `recovered_frac` units.** Rationale: the marker replicate exists to
   catch a probe that measures marker identity rather than binding; a
   disagreement of 0.15 would rival the arm/scale contrasts this design
   treats as meaningful (the killer prediction's separations are read
   against pinned CIs whose widths at n=3 have run 0.05-0.15 across this
   project's own waves), so any calibration-cell disagreement at that
   scale means marker choice is a same-order effect and the instrument is
   not marker-neutral. Additionally, at HARVEST any individual cell whose
   two-marker disagreement exceeds the cell's own claimed arm-contrast
   magnitude is labeled MARKER-FRAGILE and excluded from confirmatory
   tables (reported separately) — a relative, self-scaling demotion rule
   applied at the interpretation layer.
5. ~~(Rev 2, attack-round-2 MAJOR-4) MEMORIZATION-CONFOUND cells~~ —
   **REMOVED, Rev 3 (attack-round-3 FATAL-1).** This category existed to
   demote a cell when the (now-removed) heldout-pool memorization-ceiling
   control fired; with no trained probe and no heldout-pool control (§4.5),
   there is nothing left for this category to detect. Removed with
   disclosure, not silently renumbered away.

### 8.6 Multiple-comparison correction for the non-primary grid (Rev 1 minor, attack-round-1)

This design runs ONE pre-registered, externally-anchored primary test — the
§5.3 killer-prediction contrast (training-effect Δ at **K=32 vs. K=20**, Rev
3 — `K=48` is no longer committed, §10's FATAL-2 re-derivation, so the
committed primary contrast is now this two-point pair; `K=48`, if run as a
named extension, reports alongside per §5.3's unchanged tie-break rule but
was never structurally required for the primary verdict), in at least one
corpus, now also requiring Option 1/Option 2 agreement per §5.2/M1, read at
the single primary K=32 per §5.3's Rev 2 tie-break rule (unchanged by the
K=48 demotion, since that rule already named K=32 the sole primary
agreement-gate K). That cell's CI-based CONFIRM/REFUTE reading is exempt
from any multiple-comparison correction: it is a single hypothesis,
anchored to an externally-located quantity (`KEY_ANCHORING_DESIGN.md`'s own
`x0≈0.5455` cliff), not one draw from a scanned family. **Every other cell
in the grid** — every other K, h, corpus, arm/rung combination reported
across Leg A's and Leg B's full sweeps — constitutes a non-primary grid of
many simultaneous comparisons, and is reported under Benjamini-Hochberg FDR
control (`q=0.05`, this project's standard significance convention) rather
than raw per-cell CIs, so that scanning many K/h/corpus/arm cells does not
manufacture spurious "detections" through multiple comparisons alone. Which
cells counted as "the non-primary grid" for BH purposes is fixed before any
data exists (every headline cell in §5/§6 other than the single `K=32`-vs-
`K=20` killer-prediction contrast) — not chosen post-hoc after seeing which
cells look interesting.

**Rev 2 (attack-round-2 MAJOR-7) — a demoted extension K may never enter a
training-effect table without its own surgery pass.** Rev 1's trim #3
deferred `K=16` from the Leg-A **surgery-only** pass while keeping it in the
native/blend-ON grid — creating exactly the confound this finding names:
without a blend-OFF surgery reading at that K, its row cannot be attributed
to training vs. mechanical blend action, so reporting it in a
training-effect table (even a BH-corrected non-primary one) would silently
smuggle in the confounded blend-ON number. **Fix, registered as a standing
rule, not a one-off patch:** every K demoted from the committed grid to a
named extension — `K=40` (unchanged) and, Rev 3, `K=48` (§10's FATAL-2
re-derivation) — is reported **ONLY in the mechanical-effect table**
(§5.2a) until its own surgery extension (§10, priced separately) is also
run, explicitly labeled "no blend-OFF surgery run at this K" in the
interim, and is **never** included in any training-effect or
BH-corrected non-primary training-effect table before that. If a demoted
K's own surgery extension is run, its row is promoted to the
training-effect tables like any other K — but promotion requires the
surgery pass to actually exist, not merely the native reading. This closes
the specific K=16→K=40 case Rev 1 left open, extends it to K=48 (Rev 3),
and states the general rule for any future K added to either grid.

---

## 9. Staging — self-tests → calibration → full grid

**Stage −1 (self-tests, zero GPU, pure Python/CPU, mirrors
`FROZEN_BIAS_LM_DESIGN.md` §12.3.4's Stage-0.5 discipline exactly):**
1. Single-token/non-overlap verification for the adapted template (§4.1),
   against the real GPT-2 tokenizer, both checkpoint families' `conv_size`.
2. Hand-built `S`/`q` matrices with a known composition answer — verify
   `pred(a,h)=S^h@q` arithmetic and the cosine-recovery scorer reproduce a
   pinned expected value to `1e-6` (mirrors `TASK_D_PREREGISTRATION.md`'s
   own `_self_test` rigor).
3. The bigram-shortcut adversarial construction (§7.2) — verify the scorer
   distinguishes the true in-context value from the bigram-prior
   completion on a hand-built disagreement case.
4. The label-shuffle null generator (§4.5/§8.4, Rev 3: probe-free — verify
   the SHUFFLE mechanism, not a probe) — verify it actually destroys the
   true pairing (a planted-recoverable check: shuffle a hand-built K=4
   episode, assert the previously-correct answer no longer scores above
   threshold).
5. **(Rev 1, attack-round-1 M3) Buffer/query blank-out test** — carries
   forward attack-round-0's own §13 item 4/§13.1's Q4-answering finding:
   corrupt clause `j`'s tokens post-encoding (replace with a different valid
   clause's tokens, or random in-vocab tokens), verify clause `j+1`'s
   extracted `k_eff` is **unchanged to `1e-6`** (the same tolerance
   convention as item 2 above) — a concrete, checkable pass criterion,
   closing the C14-style buffer-purity gap named at attack-round-0 §13
   item 4 and never actually built there. **Registered build approach,
   keeping this a genuine Stage −1 (zero-GPU) item:** call the checkpoint's
   own `mixer.k_proj`/`mixer.k_conv1d` sub-modules DIRECTLY on a real
   checkpoint's token embeddings (bypassing `DeltaNetLM.forward`'s full
   pipeline entirely) — `k_eff` is captured strictly BEFORE the
   delta-rule kernel runs (§4.4), so this test never needs
   `chunk_delta_rule` (the CUDA-only component, per `lm_pretrain_rd.py`'s
   own `_MIN_KERNEL_T` assert) at all; it is ordinary CPU-compatible
   `nn.Linear`/short-convolution arithmetic on a loaded checkpoint's real
   weights, not a synthetic hand-built matrix like items 2-4, but still
   zero-GPU. **(Rev 3 disclosure, reconsidered against FATAL-2's own
   argument below and found NOT to have the same problem.)** This item's
   direct-submodule, layer-0-only approach looks superficially like the
   exact pattern FATAL-2 (below, §4.4) rejects for `q_eff`/`S_T`
   extraction — but the two tests check fundamentally different kinds of
   claims. FATAL-2 is a **construction/legality** question (can a valid
   multi-layer forward even be assembled for layer `i>0` without first
   passing through a real `T≥128` forward of layer `i−1`) — a bare
   submodule call cannot answer it representatively past layer 0. This
   item instead checks a **structural guarantee of the conv operator
   itself** (a short causal convolution's output at position `t` depends
   only on inputs at `[t−conv_size+1, t]`) — a property guaranteed by the
   operator's definition identically at every layer, regardless of what
   feeds it (raw token embeddings at layer 0, or a prior layer's residual
   output at layer `i>0`). Verifying it once, with real trained weights, at
   layer 0 is a representative test of the operator's structural guarantee
   at every layer — unlike FATAL-2, there is no layer-dependent construction
   step this item's approach could be silently failing to generalize past.
6. **(Rev 2, attack-round-2 MAJOR-3, supersedes Rev 1's version below —
   Rev 1's smoke was VACUOUS)** Surgery-toggle equivalence smoke, with a
   genuine negative control. **What was wrong:** Rev 1's version called
   `k_proj`→`k_conv1d` directly on the checkpoint's raw submodules, bypassing
   `DeltaNetLMMixer.forward()` entirely — but the `frozen_bias_arm` gate
   (`if self.frozen_bias_arm != "off": k = apply_frozen_bias_blend(...)`,
   `lm_pretrain_rd.py` L854) lives INSIDE `forward()`, strictly after that
   direct submodule call. A test built from the bare submodules never
   reaches the gate at all, so it produces the identical result **regardless
   of whether the surgery mechanism works or is even present** — it cannot
   fail, which is not a passed gate, per `CLAUDE.md`'s "run the negative
   test to completion" rule. **Fixed test:** reproduce `forward()`'s own
   pre-kernel sequence verbatim, up to and including the blend conditional
   — `q,k,v = q_proj/k_proj/v_proj(x)` → `q_conv1d/k_conv1d/v_conv1d` →
   `if self.frozen_bias_arm != "off": k = apply_frozen_bias_blend(...)` —
   reading the LOADED MODEL'S OWN live `self.frozen_bias_arm` attribute at
   each call (never a hand-copied parallel path that skips the conditional),
   stopping strictly before the kernel boundary (`chunk_delta_rule`,
   CUDA-only) — keeping this zero-GPU exactly as intended. **Two genuine
   negative controls, both required to pass, same tiny smoke batch, same
   checkpoint:**
   (a) **Live-gate-fires control:** on a LOADED arm checkpoint (global or
   per_token) with its native `frozen_bias_arm` left UNCHANGED (≠ `"off"`),
   the resulting blended `k` must DIFFER from the same input's `k` computed
   with `frozen_bias_arm` forced to `"off"` by more than a registered
   tolerance (mean abs diff `> 1e-3`) — proves the gate, as actually wired
   inside `forward()`, is not a no-op when `arm≠off`. This is the control
   Rev 1's version could never fail on, since it never touched the gate.
   (b) **Surgery-equivalence control:** with `frozen_bias_arm` forced to
   `"off"` post-load (the surgery), the resulting `k` must match
   `frozen_bias_retrofit_eval_rd.py`'s own `"kraw"`-mode raw-key capture to
   `1e-6` (`torch.equal`-class tolerance) on the identical input — this is
   Rev 1's ORIGINAL intended claim, now actually exercised through the real
   conditional rather than a code path that bypasses it. A test that cannot
   fail is not a passed gate; this replacement can fail on (a) if the gate
   is dead code and on (b) if the surgery's numeric path diverges from the
   retrofit tool's — closing MAJOR-3 in full.
7. **(Rev 1, attack-round-1 M4) `conv_size`-match assertion** — for every
   checkpoint family, load the checkpoint's own saved config and assert
   `checkpoint_config["conv_size"] == episode_cfg.conv_size` (§4.1) before
   any episode is generated for that checkpoint.
8. **(Rev 1, attack-round-1 M5) Render-adjacency/cycle-adjacency correlation
   measurement** — generate ≥10,000 episodes per swept `K` via the real
   `sample_batch_rd`, measure `succ[i]∈{i−1,i+1}` empirical rate against
   the exact combinatorial chance rate, bootstrap a 95% CI (§4.2). Pure
   episode-generation arithmetic, no checkpoint needed, zero-GPU.
9. **(Rev 2, attack-round-2 FATAL) `T_bind ≥ _MIN_KERNEL_T` floor assertion**
   — for every checkpoint family and EVERY registered K in that family's
   sweep (committed AND named extensions, §4.3), assert
   `K × clause_len(conv_size) ≥ _MIN_KERNEL_T = 128`, using that
   checkpoint's own verified `conv_size` (item 7's own check, reused
   immediately after it, not a second lookup) and the general formula
   `K_min(conv_size) = ceil(_MIN_KERNEL_T / clause_len(conv_size))`,
   `clause_len(conv_size) = max(1, conv_size − 1) + 4` (§4.1). At the
   currently-verified/assumed `conv_size=4` this evaluates to `K_min=19`,
   satisfied with margin by every registered K (20/32/40/48/64/96) — but the
   assertion is checked from the checkpoint's OWN loaded `conv_size`, not
   this pre-computed number, so a future rung with a different `conv_size`
   re-derives its own floor automatically rather than silently inheriting a
   stale one. Pure arithmetic on an already-loaded config, zero-GPU.
10. **(Rev 2, attack-round-2 MAJOR-5) Episode-seed non-collision assertion**
    — enumerate every `(purpose, leg, condition_idx, corpus_idx,
    ckpt_seed_idx, k_idx)` combination the registered committed grid and
    named extensions actually instantiate, compute `episode_seed` for each
    via §4.6's formula, and assert `len(set(seeds)) == len(seeds)`. The
    formula is collision-free by construction (§4.6's stride argument), but
    this assertion is run to catch a construction error, not merely
    asserted correct by the argument alone. Pure Python arithmetic,
    zero-GPU.
11. **(Rev 3, attack-round-3 FATAL-2) Two-forward causality assertion** —
    for a real checkpoint and a real episode, run forward-A (BIND-only) and
    forward-B (BIND+query concatenated) and assert their residual streams
    agree over the shared BIND-phase prefix to **≤`1e-6` in fp32, on CPU**
    (this design's standing tolerance, items 2/5 above — Rev 6.1 correction:
    bit-identity is NOT achievable across different-length conv calls, an
    independent audit measured the real CPU divergence at `≈2.4e-7`, see
    §4.4's own corrected registration; the earlier "bit-identical (fp32,
    CPU)" claim overstated what `ShortConvolution`'s reduction order
    actually guarantees across two differently-shaped calls). Mechanically
    verifies §4.4's causality argument (the appended query in forward-B
    cannot retroactively affect the `S_T` captured from forward-A) rather
    than resting on the argument alone — per this project's own "run the
    negative test to completion" discipline, this is checked directly, not
    merely asserted from the causal-mask property of the architecture.
12. **(Rev 4, attack-round-4 FATAL, hop-index self-test) The 3-entity
    worked example (§4.4) reproduced exactly as hand-built code, not just
    prose.** Build the K=3 episode by hand (`succ=[1,2,0]`,
    `entity_ids=[A,B,C]`, `value_ids=[B,C,A]`), compute `tgt_slot` and
    `prev_slot = _iterate_permutation(succ, a_slot, hops-1)` via the REAL
    `_iterate_permutation` function (not a re-implementation), and assert
    the gather `v_eff_items[prev_slot]` recovers the hand-verified correct
    entity at `h=1` (B) and `h=2` (C) — with a companion NEGATIVE check
    that the naive `v_eff_items[tgt_slot]` gather returns the WRONG,
    one-hop-past entity (C at `h=1`, A at `h=2`), confirming the test can
    actually distinguish the fixed from the buggy gather rather than
    passing either way. Pure Python/tensor index arithmetic on hand-built
    tensors, zero-GPU, zero-checkpoint — this is the "run the negative
    test to completion" instance for the off-by-one-hop bug specifically,
    per the standing house rule that this exact bug has already occurred
    once in this codebase (`DELTANET_REALDATA_DESIGN.md` §14.2).
13. **(Rev 4, attack-round-4 FATAL, `v_conv1d` hook equivalence smoke)**
    verify the new `v_conv1d` forward hook (§4.4) captures the identical
    tensor a direct `mixer.v_proj`→`mixer.v_conv1d` submodule call produces
    on the same input, to `1e-6` — the same equivalence-smoke pattern
    already used for the `k_conv1d`/surgery-toggle checks (item 6 above),
    applied to the one new observable this revision adds. CPU-compatible
    (`v_conv1d` is a `ShortConvolution`, not the CUDA-only kernel), same
    tolerance convention as items 2/5/11.
14. **(Rev 5, attack-round-5 FATAL, `q_conv1d` hook equivalence smoke,
    mirroring item 13 exactly)** verify the new `q_conv1d` forward hook
    (§4.4) captures the identical tensor a direct
    `mixer.q_proj`→`mixer.q_conv1d` submodule call produces on the same
    input, to `1e-6` — the same equivalence-smoke pattern item 13 already
    applies to `v_conv1d`, applied here to the observable this revision's
    own FATAL fix retargets `q_eff` extraction onto. CPU-compatible
    (`q_conv1d` is a `ShortConvolution`, not the CUDA-only kernel), same
    tolerance convention as items 2/5/11/13. Run on forward-B's own input
    construction (bind+query concatenated, §4.4), since `q_eff` is only
    ever read from forward-B, never forward-A.

**Gate:** Stage 0 (calibration) may not launch until all Stage −1 tests
pass — the exact "specification that has not been executed is not a passed
gate" convention this project applies everywhere.

**Stage 0 (calibration, 1 cell, blinded from any headline decision):** the
Leg-A CONTROL-arm checkpoint
`frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0` (final step
20000, under `/data/deltanet_rd_frozenbias_ckpts/` — Rev 6 fix,
attack-round-6 MINOR-2: the earlier "14M mixcontrol 'off' arm" phrase
conflated two disjoint families — "mixcontrol" is a Track-C/Leg-B
checkpoint with no arms at all; Stage 0 calibrates on the ACTUAL Leg-A
comparison family's control arm, which shares the mixcontrol
architecture d_model=256/n_layers=2/d_state=64 anyway), seed 0, one
corpus, K=32, h∈{1,2,3,4}, BOTH query markers (§7.6's robustness
replicate). Purposes (Rev 3: purpose (a)
rewritten — no probe hyperparameters exist to fix, §4.5): (a) run the
Stage −1 item 11 causality assertion for real on this checkpoint, and
confirm the two-forward protocol (§4.4) executes cleanly end to end before
touching any other checkpoint; (b) measure real per-cell wall-clock cost —
now covering BOTH forward-A and forward-B — to firm up §10's budget; (c)
run the §8.4 null-band construction for real, **at every h∈{1,2,3,4}
independently (Rev 1, attack-round-1 M2 — not h=1 only)**, and confirm the
h1 floor is achievable at all before committing to the full grid — if it is
not, this is itself a decisive, informative result (the whole readout is
uninterpretable for this checkpoint family) and the grid does not launch;
the h=2/3/4 null bands become the registered per-h chance floors §8.4 now
requires before a CONFIRM/REFUTE reading at those hops is licensed;
(d) **(Rev 4, attack-round-4 FATAL consequence; Rev 5, attack-round-5
FATAL consequence extends this to premise (iii))** measure premise
diagnostic (iv), `cos(k_eff_i, v_eff_i)` per entity `i`, AND premise
diagnostic (iii), `cos(q_eff_a, k_eff_a)` per entity `a` (now genuinely
via the new `q_conv1d` hook, §4.4/§13.5 — not the near-tautological
`k_conv1d`-only reading this design used through Rev 4), in this
calibration cell's own episode pool (§4.4), alongside premises (i)-(ii) —
each reported as a mean/median and spread, together with its own
cross-entity-shuffled null distribution (§4.4's MAJOR-1 action-rule
table); (e) **(Rev 5, attack-round-5 MAJOR-1/MAJOR-2)** apply the §4.4
action-rule table's two gates (premise iii vs. its null, premise iv vs.
its null) and the §8.4 absolute h=1 backstop (recovered_frac ≥ 0.10) —
all three gates are evaluated HERE, before Stage 1 launches, and their
consequences (demoting h=1 and/or h≥2 to exploratory-only status) are
applied to the pre-registration itself, informing (per §4.4's own wiring)
how far past `h=1` this checkpoint family's `h≥2` numbers should be
expected to track before any full-grid result is interpreted.

**Stage 0.5 (Rev 2, attack-round-2 MAJOR-6; thresholds/actions restated at
Rev 3's two-forward rates and single committed rung-3 K — rung-3 pass-cost
calibration, gates ONLY Leg B's rung-3 rows, does not block Leg A or
rungs 1/2 of Leg B):** one single-EPISODE-timing, single-checkpoint cell at
1.31B scale (one of the 2 archived rung-3 runs, `K=64` — rung-3's sole
committed K, §4.3, so the timing measured is exactly what the committed
grid will pay), timing **both** forward-A and forward-B (the two-forward
protocol's real per-episode cost, §4.4 — a single-forward timing would
re-import the exact amortization error §10's Rev 3 multiplier fix just
removed), **blinded to any h4-style recovery readout** — this cell measures
wall-clock/GPU-h cost only, so that a mini calibration wave does not become
an accidental peek at the headline composition result. **Registered because
the existing 1.31B pass-cost figure is an unmeasured extrapolation, not an
anchor:** §10's own three measured anchors are non-monotonic with scale
(0.0348 GPU-h/pass @ 14M → 0.013 @ 98M → 0.045 @ 392M, all single-forward)
— there is no justified functional form to extrapolate from, so applying
the 392M rate to rung-3 (1.31B, ≈3.3× the params) is a guess, not a
measurement. **Abort trigger (fires BEFORE the committed 25 GPU-h ceiling
is ever at risk; Rev 3 — rung-3 now has only ONE committed K, so the Rev 2
first tier's "drop the non-primary K" action has no committed object left
and moves to the extension gate):** if this cell's measured per-pass cost
exceeds **2×** the 392M two-forward baseline (`>0.18 GPU-h/pass`,
pre-sequence/contingency-multiplier — 2 × the doubled 0.09 base, the same
threshold the cell is itself conservatively priced against, §10), rung-3
keeps its single committed `K=64` cell but its `d=128` K-extension
eligibility (§10) is revoked for this phase. If the measured cost exceeds
**4×** that baseline (`>0.36 GPU-h/pass`), halt Leg B's rung-3 rows
entirely and disclose rung-3 as unevaluated within Phase-1 budget,
reporting only rungs 1/2/(392M) for H_LINK-B — mirroring the existing
Stage-0 abort rule's structure (a multiple-of-anchor trigger, fired before
the committed ceiling is at risk).

**MINOR fix (Rev 5, attack-round-5) — a registered OOM fallback, never a
crash-loop.** The abort triggers above assume the Stage 0.5 cell completes
and produces a real timing number; they say nothing about what happens if
the cell OOMs outright (a distinct failure mode from "slow but
completes," plausible at 1.31B scale with two full forward passes per
episode). **Registered fallback:** on OOM, retry the Stage 0.5 cell
**once**, at `batch_size=8` (half the pinned `batch_size=16`, §4.2) — if
that retry also OOMs, **halt Leg B's rung-3 rows and disclose**, exactly
as the `>4×` abort trigger above already does, never a further
retry-at-smaller-batch loop. This is a bounded, one-shot fallback (retry
once, then halt), not an open-ended crash-loop — consistent with this
project's own standing discipline against unattended retry loops (see
`CLAUDE.md`'s tmux/supervisor rules for the analogous long-running-job
case). If the batch-8 retry DOES complete, its measured per-pass cost
(scaled back to the `batch_size=16` rate for a like-for-like comparison
against the abort thresholds above) is used in place of the batch-16
number, and this substitution is disclosed in the harvest report rather
than silently presented as if batch 16 had been measured directly.

**Stage 1 (full grid, Rev 3 — re-derived per-leg committed K sets, §4.3/§10,
attack-round-3 FATAL-2):** Leg A committed grid — 18 core cells × 2 K
`{20,32}` × 4 h (native + surgery, both at the identical 2 K, §4.3/§8.6) —
plus Leg B committed grid — 20 checkpoints total (12 at `d_state=64`: 14M +
98M rungs, 2 corpora × 3 seeds × 2 rungs; 8 at `d_state=128`: 392M rung-2 6
+ rung-3 2, per §6.1's PINNED 1-seed×2-corpora rung-3 configuration) × the
leg's own single committed near-cliff K (`K=32` for the `d=64` rungs,
`K=64` for the `d=128` rungs, §6.1/§4.3/§6.2) × 4 h — per §10's budget.
`K=48` (both legs), `K=40` (both legs), `K=20` (Leg B, both `d_state`'s),
and `K∈{32,96}` (`d=128`, Leg B) are named, separately-priced EXTENSIONS
(§10), not part of the committed Stage 1 launch. **(Rev 2 correction,
preserved for the record: Rev 1's own Stage-1 line read "Leg B (14
checkpoints...)" — an arithmetic slip; 12 (`d=64`) + 8 (`d=128`) = 20, not
14.)**

---

## 10. Budget (RE-DERIVED, Rev 3 — attack-round-3 FATAL-1 frees the probe budget, FATAL-2 doubles the per-pass rate; full old→new reconciliation below; Rev 4 and Rev 5 each CONFIRM the total is unchanged, see the notes immediately below)

**Rev 4 budget-neutrality confirmation (attack-round-4 BUDGET item,
verified rather than merely asserted).** Reverting §4.4's comparison
target from `k_eff` to `v_eff` and adding the new `v_conv1d` hook changes
**zero** GPU-priced quantities in this section: (a) the number of mixer
forward calls per episode is unchanged (still forward-A + forward-B, the
two-forward protocol §4.4 established at Rev 3, untouched by this
revision); (b) the new hook fires on the SAME forward-A call the existing
`k_conv1d` hook already fires on — it adds a second `register_forward_hook`
call and a second small tensor capture, not a new pass, and not a
measurable wall-clock delta at this scale (the anchors below are
dominated by the mixer's own compute, not by which post-conv tensors a
hook happens to also record); (c) no probe is reintroduced (§4.5), so none
of the freed FATAL-1 budget (the 3.89 GPU-h probe-training/heldout-control
lines, below) reappears. **Every number in this section — the anchors, the
×8 multiplier, the per-row subtotals, the ≈24.20 GPU-h committed total,
the ≈0.80 GPU-h margin, the ≈38.70 GPU-h named-extension reserve — is
UNCHANGED from Rev 3 and is re-printed below rather than re-derived, per
the directive to state this explicitly.**

**Rev 5 budget-neutrality confirmation (attack-round-5 BUDGET item,
verified rather than merely asserted).** Retargeting `q_eff` extraction
from `k_conv1d` to a new `q_conv1d` hook changes **zero** GPU-priced
quantities: (a) the number of mixer forward calls per episode is
unchanged — the new hook fires on the SAME forward-B call the two-forward
protocol already runs (forward-B is where `q_eff` was always read; only
which submodule's output the hook attaches to changes), so no new pass is
added; (b) it adds a second `register_forward_hook` call and a second
small tensor capture to an already-executing call, the identical
zero-cost pattern Rev 4's `v_conv1d` hook already established for
forward-A; (c) the premise-diagnostic action-rule table, the h1 absolute
backstop, the Stage 0.5 OOM fallback, and the two new §12 precedence
rules are Stage −1/Stage 0 measurements already priced inside the
existing calibration cells, or pure interpretation/reporting rules with
no GPU cost of their own — none adds a row to the budget table below.
**Every number in this section remains UNCHANGED from Rev 3/Rev 4 —
≈24.20 GPU-h committed, ≈0.80 GPU-h margin, ≈38.70 GPU-h named-extension
reserve, 25 GPU-h ceiling — re-printed below rather than re-derived.**

**Unit cost anchor (measured, not assumed):** this project's own
retrofit-eval instrumentation on the 14M/rung-1-scale architecture costs
**≈0.0348 GPU-h per checkpoint-pass** (`FROZEN_BIAS_LM_DESIGN.md` §12.1:
1.6 GPU-h / 46 passes, forward-pass-only). The scale-transfer harvests give
two more real anchors at larger scale: **≈0.013 GPU-h/pass at 98M**
(0.08 GPU-h / ~6 passes, §5.9) and **≈0.045 GPU-h/pass at 392M**
(1.04 GPU-h / ~23 passes, §5.10) — i.e., per-pass probe cost at THIS
project's scales has not scaled up badly with params (batching/short
sequences dominate); **1.31B (rung-3) has NO measured anchor** — the three
existing anchors are non-monotonic (0.0348→0.013→0.045), so extrapolating
to rung-3 is a guess, not a measurement (§9 Stage 0.5/MAJOR-6 registers the
calibration cell that replaces the guess with a real number before the
rung-3 grid is trusted).

**Rev 3 multiplier re-derivation — the anchors do NOT amortize the
two-forward protocol; the rate doubles, honestly.** The three anchors above
were all measured on **single-forward** instrumentation (one mixer call per
batch of windows — `lm_attractor_probe_rd.py`/`frozen_bias_retrofit_eval_rd.py`
each run exactly one hooked forward per batch, verified against the tools
themselves, §0). §4.4's two-forward protocol (FATAL-2) makes every Option-1
episode cost **two** full mixer calls (forward-A bind-only + forward-B
bind+query) — nothing in the anchors' own measurement ever included a
second forward, so the honest treatment is a **2× protocol multiplier on
the per-pass rate**, not an assumption that the anchor "probably absorbs
it." (Forward-B simultaneously serves Option 2's natural logit-margin
reading — the query-position logits come from the same call — so Option 2
remains effectively free; the 2× covers both readouts, not 2× each.
Multiple queries per episode batch into forward-B's batch dimension at
`batch_size=16`, §4.2 — more sequences, not more calls, covered by the
sequence-length multiplier below.) The full per-pass rate is therefore:
anchor × 2 (longer sequences than the anchors' corpus windows, unchanged
from Rev 1/2) × 2 (standing contingency rule, `SCALE_TRANSFER_DESIGN.md`
§8.4, unchanged) × **2 (two-forward protocol, NEW)** = **anchor × 8**:
`d=64` rate `0.0348×8=0.2784` GPU-h/pass; `d=128` (392M) rate
`0.045×8=0.36` GPU-h/pass.

**Rev 3 re-derivation, committed grid (per-leg committed K's, §4.3):**

| Item | Cells | GPU-h/pass | Subtotal | Rev 2 → Rev 3 |
|---|---|---|---|---|
| Leg A native grid, COMMITTED K∈{20,32} (18 core cells × 2 K) | 36 | 0.2784 | **10.02** | 54 passes/7.52 → 36/10.02 (K=48 demoted to extension; rate doubled) |
| Leg A surgery grid, K∈{20,32} (12 arm-only cells × 2 K) | 24 | 0.2784 | **6.68** | 36 passes/5.01 → 24/6.68 (same demotion; rate doubled) |
| Leg B, `d=64` rungs, COMMITTED K=32 only (12 checkpoints × 1 K — the leg's own single primary near-cliff K, §6.2; K=20/48 were never in its headline) | 12 | 0.2784 | **3.34** | 36 passes/5.01 → 12/3.34 |
| Leg B, `d=128` rungs, COMMITTED K=64 only (8 checkpoints × 1 K — same single-primary-K logic; rung-3 = 2 checkpoints per §6.1's PINNED 1-seed×2-corpora config) | 8 | 0.36 | **2.88** | 16 passes/2.88 → 8/2.88 (half the passes at double the rate — exact coincidence of the 2× factors, disclosed as such) |
| Stage −1 self-tests (CPU only, incl. items 11-13, Rev 4 adds 12/13) | — | 0 | **0** | unchanged |
| Stage 0 calibration (1 cell × 2 markers, all 4 h scored from the same captured passes) | 2 | 0.2784 | **0.56** | 0.28 → 0.56 (rate doubled) |
| Stage 0.5: rung-3 pass-cost calibration cell (1 checkpoint × K=64, priced at 2× the 392M two-forward rate = 0.72/pass, matching its own abort threshold; now times BOTH forwards) | 1 | 0.72 | **0.72** | 0.36 → 0.72 (rate doubled) |
| ~~Probe training (2 shared reference probes + heldout-pool setup)~~ | — | — | **0 (REMOVED)** | 0.10 → 0 (FATAL-1: no probe exists to train, §4.5) |
| ~~Heldout-pool memorization-ceiling control (Rev 2 MAJOR-4)~~ | — | — | **0 (REMOVED)** | 3.79 → 0 (FATAL-1: control measures nothing without a trained probe, §4.5 adjudication) |
| **Total, Phase 1 (committed)** | | | **≈24.20 GPU-h** | Rev 2: ≈24.95 → Rev 3: ≈24.20 |

**Reconciliation, in words (old → new, every moving part):**
- **Freed by FATAL-1:** probe training (0.10) + heldout-pool control (3.79)
  = **3.89 GPU-h** removed outright — not respent elsewhere, simply gone
  with the machinery that motivated it.
- **Charged by FATAL-2:** the per-pass rate doubles (anchor×4 → anchor×8)
  on every remaining GPU row. Rev 2's grid held at Rev 2's K sets under the
  doubled rate would have cost `(7.52+5.01+5.01+2.88+0.28)×2 + 0.72 =
  42.12` GPU-h — **≈17.1 GPU-h over the 25 GPU-h ceiling**; the freed 3.89
  covers less than a quarter of that. The rate increase is therefore paid
  for by grid re-scoping, not by the freed budget alone.
- **Re-scoped:** committed K's move from per-`d_state` sets (Rev 2:
  `{20,32,48}` at d=64, `{20,64}` at d=128, shared by both legs) to
  per-leg sets (Rev 3: Leg A `{20,32}`; Leg B `K=32` (d=64) / `K=64`
  (d=128) only). What each leg's PRIMARY claim needs is preserved exactly:
  Leg A's killer prediction was always a low-load-vs-near-cliff two-point
  contrast read at K=32 (§5.3's own tie-break rule already made K=32 the
  sole primary agreement-gate K); Leg B's headline was always the single
  near-cliff K per `d_state` (§6.2). The demoted points (`K=48`, `K=40`,
  Leg-B `K=20`, `d=128` `{32,96}`) lose grid RESOLUTION — extra
  corroborating points, the §6.3 fixed-K sanity contrast — not any
  registered primary claim; each is a named, priced extension below.
- **Bottom line:** 24.95 (Rev 2) → 24.20 (Rev 3), margin 0.05 → **0.80**
  under the unchanged 25 GPU-h ceiling. The margin improves not because
  Rev 3 is cheaper in any real sense (the per-pass rate DOUBLED) but
  because the committed grid is deliberately narrower — stated plainly
  rather than presented as a savings.

**Named extensions (fully priced at the Rev 3 two-forward rates,
registered, NOT part of the Phase-1 committed total — the reserve,
disclosed rather than silently available):**

| Extension | Cells | Subtotal | Trigger to promote |
|---|---|---|---|
| Leg A native grid, K=48 only (Rev 2-committed, demoted this revision) | 18 | 5.01 GPU-h | **(Rev 4, attack-round-4 MINOR — mechanical rule, replacing "passes or trends"):** promote iff, at the committed `K=32` killer-prediction cell, in at least one corpus, EITHER (a) the training-effect `Δ`'s pinned CI (`t(2,.975)=4.303`, n=3) excludes zero on the positive side — i.e. `K=32` already CONFIRMs per §5.3 — OR (b) `Δ`'s point estimate is positive and its magnitude is at least 50% of the CI's own half-width while the CI itself still straddles zero (a "trending positive, not contradicted" reading measured against the cell's own noise floor, not an externally invented effect size). A negative or ≈zero point estimate, or a CI whose lower bound sits materially below zero, does NOT trigger the extension regardless of the upper bound. K=48 is the past-cliff corroboration point (§5.3). |
| Leg A surgery grid, K=48 only (required before K=48's native reading may enter any training-effect table, §8.6/MAJOR-7) | 12 | 3.34 GPU-h | Same trigger as above, run together with it, never the native-only reading alone |
| Leg A native grid, K=40 only | 18 | 5.01 GPU-h | Non-critical resolution point; run only after the K=48 extension |
| Leg A surgery grid, K=40 only (same §8.6 rule) | 12 | 3.34 GPU-h | Same trigger, run together |
| Leg B `d=64` rungs, K∈{20,48} (restores the low-load + past-cliff points; enables §6.3's K=20 fixed-K sanity contrast at d=64) | 24 | 6.68 GPU-h | Committed single-K ladder shows a scale trend worth bracketing in load |
| Leg B `d=64` rungs, K=40 only | 12 | 3.34 GPU-h | Only after the {20,48} extension |
| Leg B `d=128` rungs, K∈{20,32,96} (K=20 enables §6.3's fixed-K contrast at d=128) | 24 | 8.64 GPU-h | Gated on the committed K=64 pass showing a signal worth resolving (unchanged logic from Rev 1's own trim #2) |
| §5.2a's 4th 2×2-grid cell (off-ckpt, blend-ON via retrofit table, MAJOR-2), at the committed K∈{20,32} | 12 | 3.34 GPU-h | The mechanical-effect contrast (arm-ckpt blend-ON vs. blend-OFF) is itself significant at the killer-prediction cell |
| **Total named extensions (reserve)** | | **≈38.70 GPU-h** | — |

**Ceiling: 25 GPU-h** (unchanged — Rev 3's re-derivation targets fitting
under this existing ceiling, same as Rev 1/2, not raising it). **Margin:
≈0.80 GPU-h** — materially better than Rev 2's ≈0.05, for the disclosed
re-scoping reason above, and still thin enough that the abort rules below
(both Stage-0's original and Stage-0.5's) remain the real safety valve,
not this arithmetic margin. **Abort rules (two, distinct scopes; thresholds
now stated against the TWO-FORWARD per-pass rates):**
1. **Stage 0 (14M-scale):** if the calibration cell's real measured
   wall-clock cost exceeds **3×** the two-forward d=64 baseline
   (3 × 0.0696 GPU-h = the anchor's own 0.0348 doubled for two forwards,
   before the sequence/contingency multipliers), halt before Stage 1
   launches and re-price the full grid from the real number.
2. **Stage 0.5 (rung-3-scale, MAJOR-6):** if the rung-3 calibration cell's
   real measured cost exceeds **2×** the 392M two-forward baseline
   (2 × 0.09 GPU-h), rung-3 stays at its single committed K=64 cell but
   its Leg-B extension eligibility is revoked; if it exceeds **4×**, halt
   Leg B's rung-3 rows entirely and disclose (§9 gives the full trigger
   detail — the Rev 2 "drop rung-3's non-primary K" first tier is
   restated here because rung-3 now has only ONE committed K, so the
   first tier's action moves to the extension gate instead).
If either abort rule fires and a further cut is still needed, the named
extensions above are never touched (they were never committed); the next
lever is deferring Leg B's 98M rung (6 checkpoints × K=32 = 6 passes,
1.67 GPU-h — the ladder's headline question remains answerable, less
finely resolved, from the 14M/392M/1.31B points, mirroring
`SCALE_TRANSFER_DESIGN.md` §8's own "endpoints alone" cut logic) — named
here as the last-resort lever, not yet applied. (Rev 2's equivalent lever
— deferring Leg B `d=64`'s K=48 cell — no longer exists to pull: that
cell is already an extension this revision.)

---

## 11. Phase 2 (gated, sketch only — not fully designed)

**Gate condition:** Leg A's H_LINK-A killer prediction (§5.3) passes —
global-vs-off delta CI excludes zero at K near the cliff, per-token does
not mirror it, and the effect is NOT explained away by any §7 shortcut
check.

**If gated open, the confirmatory wave (sketch):** train NEW small LMs with
the frozen-bias intervention (global-vector arm only — the arm Leg A
would have shown works, per the house "standard benchmarks for publishable
claims" rule) at **2 scales** (a 14M-class and a 98M-class rung, reusing
Track C's own architecture configs so no new engineering is needed for the
model definition itself), evaluated on a **standard, published multi-hop or
associative-recall benchmark format** (candidates to be researched properly
at design time, not chosen here: options in this space include bAbI-style
synthetic multi-hop QA, a small held-out slice of a standard
associative-recall benchmark used in the SSM/fixed-state literature, or a
multi-hop subset of an existing QA benchmark rendered at a scale these
models can attempt at all — this needs its own literature-verification pass
before any threshold is chosen, per the standing "verify the claim is
novel... check the literature first" rule). **Gate criterion sketch (not
finalized):** the frozen-bias arm must move the standard benchmark's own
metric by a pre-registered margin over a val-loss-matched control, at
**both** scales, with the same corpus/seed discipline this entire program
already uses. This section is intentionally underspecified — it is a
placeholder for a full design pass, not a committed plan.

---

## 12. Pre-registered outcome table (all 4 cells named, before any data exists)

| | H_LINK-B (scale trend) present | H_LINK-B absent |
|---|---|---|
| **H_LINK-A (intervention effect) present** | **Cell 1 — the keystone confirm.** Both legs positive: geometry stabilization causally helps composition AND scale alone also helps, with the two independently-measured axes agreeing in direction. Publication value: high — connects three previously-unconnected geometric results to a genuine capability, licenses Phase 2. | **Cell 2 — intervention works, scale doesn't (yet).** The frozen-bias fix is a real, deployable lever independent of scale; scale's own span_frac climb either doesn't translate to composition or needs more than 1.31B params/1.5B tokens to show. Publication value: still high — a targeted, cheap intervention beats waiting for scale, an actionable engineering result. |
| **H_LINK-A absent** | **Cell 3 — scale alone rescues composition, the fix doesn't.** The span_frac ladder's climb is causally linked to composition, but the specific frozen-bias mechanism that moves span_frac at 14M scale isn't the lever that matters (or doesn't transfer its geometric effect into a compositional one). Publication value: moderate-high — narrows the field's account of *why* scale helps, redirects intervention-design effort away from key-bias schemes specifically. | **Cell 4 — geometry stabilization is real but functionally inert for in-context composition, at both leg's own resolutions.** The single most important negative this design can produce: it would mean the entire span_frac/coherence/rank-collapse geometric attribution program (frozen-bias, capacity-cliff, coherence-dose-response) measures something that does not causally matter for the thing this project actually cares about, redirecting the program's next phase toward a different observable or a different mechanism entirely. Publication value: high, as an honest, hard-won negative — explicitly not a "the experiment failed," a genuine finding about which geometric quantities are and are not behaviorally load-bearing. |

**Rev 1 clarification (attack-round-1 F2/M1):** "H_LINK-A (intervention
effect) present/absent" in this table now means the **training-effect**
contrast of §5.2a with Option 1/Option 2 agreement per §5.2/M1 — not the raw
native-forward (blend-ON) reading. A cell where Option 1 and Option 2
disagree directionally routes to READOUT-DIVERGENCE (§5.2), a fifth, named
outcome outside this 2×2 table, not silently folded into any of the four
cells above.

**MINOR fix (Rev 5, attack-round-5) — reporting rule for READOUT-DIVERGENCE
at Leg A's primary cell coexisting with Leg B's own cell-table verdict.**
READOUT-DIVERGENCE (Leg A's killer-prediction cell, §5.2) and the 2×2
table's H_LINK-B column (or its AMBIGUOUS/READOUT-FORM-INVALID demotion,
below) gate two DIFFERENT sub-hypotheses (H_LINK-A vs. H_LINK-B) through
the identical instrument at DIFFERENT cells — one routing to a named
outcome neither forces nor excludes any particular reading of the other.
**Registered rule:** when Leg A's killer-prediction cell routes to
READOUT-DIVERGENCE, this is reported ALONGSIDE whichever outcome Leg B's
own primary cells yield (a 2×2-table cell, AMBIGUOUS, or
READOUT-FORM-INVALID) — **both named outcomes are stated side by side in
the final report, neither overriding or suppressing the other.** A Leg-A
READOUT-DIVERGENCE does not retroactively invalidate a clean Leg-B
reading, and a clean Leg-A CONFIRM/REFUTE does not retroactively resolve a
Leg-B READOUT-DIVERGENCE/AMBIGUOUS finding at its own cells — each leg's
own outcome is reported on its own terms.

**READOUT-FORM-INVALID (Rev 4, attack-round-4 MAJOR) — a sixth named
outcome, distinct from "no capability" (Cell 4) and from
READOUT-DIVERGENCE.** **Trigger:** `h=1` clears its registered chance
floor (§8.4) — the probe itself is valid — but `h≥2` sits at floor
UNIFORMLY across EVERY arm (Leg A) and EVERY rung (Leg B), at every
tested `K`. This is the observationally-indistinguishable case the
attacker identified: a uniform `h≥2` floor is consistent with BOTH "no
checkpoint in this design's scope does multi-hop composition at all" AND
"Option 1 is testing the wrong mechanism for every checkpoint here"
(§1/§4.4's framing adjudication — single-layer state self-iteration is an
a priori unlikely implementation, and a low premise-(iv) cross-role-
identity reading, §4.4, would produce exactly this signature even in a
checkpoint that composes fine via cross-layer chaining). **Under this
outcome, Cell-4-style "stabilization is functionally inert for in-context
composition" claims are BARRED** — Cell 4 requires evidence that
composition capability is genuinely absent, not merely that ONE
mechanistic hypothesis about how it might be organized failed to show a
signal. **Resolution path:** Option 2 (the natural, mechanism-agnostic
logit-margin readout, §4.4) becomes the INTERPRETIVE INSTRUMENT of
record for `h≥2` under this outcome, and the honest report is: *"no
evidence of `S_T`-self-iteration composition in text-pretrained
checkpoints at this scale; the behavioral (Option 2) readout says
`X`"* — stating plainly which instrument the claim rests on, never
silently substituting one for the other after the fact.

**AMBIGUOUS (Rev 4, attack-round-4 FATAL-ADJACENT, §6.2) — H_LINK-B's own
demoted scale-trend reading.** When fewer than 3 of the 4 Track C ladder
rungs show Option 1/Option 2 agreement at their own primary near-cliff
cell (§6.2's new gate), H_LINK-B's scale-trend claim is reported as
AMBIGUOUS rather than forced into "present" or "absent" for the 2×2 table
above — the corresponding column is annotated "H_LINK-B: AMBIGUOUS" and
read descriptively (per-rung agreement/disagreement and point estimates),
never collapsed into Cell 1/2/3/4's binary framing.

**MINOR fix (Rev 5, attack-round-5) — precedence between
READOUT-FORM-INVALID and AMBIGUOUS when both gates could fire.**
READOUT-FORM-INVALID's trigger (h=1 clears its floor but h≥2 sits at a
uniform floor across every arm/rung) and AMBIGUOUS's trigger (fewer than
3 of 4 Track C rungs show Option 1/Option 2 agreement) are not mutually
exclusive — a uniform h≥2 floor plausibly ALSO produces disagreement or
degenerate non-agreement at most rungs' own Option 1/Option 2 check (a
floor reading is close to noise, and noise need not agree in direction
with Option 2 at any given rung). **Registered precedence rule:
READOUT-FORM-INVALID wins.** If both triggers are met simultaneously, the
outcome is reported as READOUT-FORM-INVALID, not AMBIGUOUS — it is the
MORE SPECIFIC diagnosis (it names the actual mechanism-level reason the
readout is uninformative at h≥2, whereas AMBIGUOUS only states that the
ladder's own agreement gate did not clear enough rungs, which is itself a
likely SYMPTOM of the same underlying readout-form problem, not an
independent finding). AMBIGUOUS remains the reported outcome only when
Leg B's own agreement gate fails WITHOUT a uniform h≥2 floor also being
present — some rungs show real, non-floor h≥2 signal but Option 1/Option 2
nonetheless disagree in direction at enough of them to miss the 3-of-4
bar — a genuinely narrower failure mode than READOUT-FORM-INVALID's own.

---

## 13. Attack-round-0 — minimum 6 questions, best current answers or explicit TODOs

1. **Will a checkpoint that never saw this grammar show ANY composition
   signal above the h1 chance floor at all, even at h=1?** Best current
   answer: plausible but unverified — induction-head-style one-hop recall
   is well documented in Transformers and increasingly in fixed-state/SSM
   architectures, but not yet demonstrated on THESE specific checkpoints.
   This is exactly why §8.4's floor is a *measured* null, not an assumed
   one, and why Stage 0's calibration cell (§9) is a hard, non-skippable
   gate before any grid launches — if h1 fails the floor at calibration,
   the entire design is reported as a negative result about zero-shot
   probe feasibility, not quietly patched.
2. **Is `pred(a,h)=S_T^h@q_eff` even the right composition formula for a
   model that was never trained under a hard β-mask?** TODO, unresolved:
   this is an assumption imported from a regime (hard-masked, from-scratch
   training) where it was proven to be the exactly-correct object. For a
   general LM, β fires at whatever rate/positions the model's own learned
   gate computes, so `S_T` after the BIND phase may not cleanly represent
   "K accumulated one-hop writes" at all — it could equally represent
   heavily-averaged, interference-dominated mush. Option 2 (§4.4, the fully
   natural black-box reading) exists specifically as an independent check
   on this assumption; if Option 1 and Option 2 disagree sharply, that
   disagreement is itself the finding, reported explicitly, not resolved
   by picking whichever one shows the hoped-for effect.
3. **Does the shared, arm-blind probe (Rev 1/2 §4.5 primary) have enough
   capacity to decode a stabilized (global-arm) checkpoint's geometry if
   that geometry is meaningfully DIFFERENT from the reference ("off")
   checkpoint it was trained on?** Historical best answer (Rev 1/2): a real
   unresolved tension — an arm-blind probe was the most defensible against
   probe-leakage (§7.4) but the most likely to systematically
   *under*-report a genuinely different-geometry arm, because a linear
   probe fit on one basis need not transfer perfectly to a rotated/rescaled
   one.
   **DISSOLVED at Rev 3 (attack-round-3 FATAL-1):** the trained probe this
   question was about no longer exists — §4.4's fixed readout scores in
   `d_state` space against the target's own `k_eff` at Rev 3 (Rev 4:
   `v_eff`, §13.4 — the dissolution argument is unaffected by which
   in-family object is compared, since either way there is no fitted
   basis), fitting nothing. With no fitted basis to transfer, there is no
   capacity ceiling through which a different-geometry arm could be
   under-reported; each arm's own forward machinery produces both sides of
   its own cosine comparison. Recorded at §13.3 — this question dissolves
   rather than being answered, the same way item 4 below was RESOLVED
   (rather than dissolved) by Rev 1 actually building the test it asked
   for.
4. **Is the ordinary-token buffer/query-marker adaptation (§4.1, §7.6) safe,
   or does it reopen exactly the cross-clause smearing problem the
   from-scratch harness's zero-pinned buffer was built to close?** TODO,
   partially addressed: §7.6's two-marker robustness replicate at
   calibration is the concrete check, but it only tests marker-choice
   sensitivity, not the deeper question of whether an UNPINNED buffer
   embedding (however generic) leaks any clause-boundary signal at all. A
   fully rigorous closure would require a per-item blank-out test analogous
   to `DELTANET_REALDATA_DESIGN.md`'s own C14 buffer-purity check (corrupt
   clause j, verify clause j+1's extracted `k_eff` is unchanged) — this is
   NOT yet in the design and should be added as a Stage −1 self-test before
   build, not deferred to the full grid.
   **RESOLVED at Rev 1 (attack-round-1 M3):** this exact test is now Stage
   −1 item 5 (§9), with the concrete `1e-6` tolerance this question asked
   for and a registered zero-GPU build approach (direct `k_proj`/`k_conv1d`
   submodule calls, bypassing the CUDA-only kernel). This question is the
   one attack-round-1 answered in full — recorded at §13.1.
5. **Why should the K-sweep points chosen per `d_state` (§4.3) be trusted
   to actually bracket each `d_state`'s own capacity-relevant regime, given
   that this is a NEW observable (reasoning recovery) that has never been
   measured against K for these general-purpose checkpoints?** Best current
   answer: the K choices are anchored to the ALREADY-LOCATED d=64 cliff
   (`x0=0.5455`) and the matched-K/d convention used throughout
   `KEY_ANCHORING_DESIGN.md`, which is the best available prior — but this
   design's own §6.3 explicitly disclaims that the d=128 cliff location (if
   any exists for THIS observable) is unknown, so the d=128 K choices are a
   reasonable bet, not a verified bracket. If Leg B's `d=128` results show
   no signal at any tested K, this is ambiguous between "no cliff exists
   for this observable at d=128" and "the tested K range missed it" —
   named honestly, not resolved by this document.
6. **Does the val-loss-matched, same-architecture comparison in Leg A
   actually control for everything relevant, or could the frozen bias's
   extra ~12.9MB table (Arm 2) / global vector (Arm 2′) change the
   effective optimization trajectory in a way that affects composition
   through a channel OTHER than the measured key geometry?** TODO,
   unresolved: §7.9 argues the hook point is code-identical across arms,
   but that only rules out a *measurement* artifact, not a *training
   dynamics* one (e.g., the extra frozen parameters changing gradient
   noise/effective learning rate for the rest of the model in some subtle
   way). No existing instrument in this design directly rules this out;
   flagged as an open confound a hostile reviewer would press on, with no
   current mitigation beyond "the same concern already applies to every
   claim `FROZEN_BIAS_LM_DESIGN.md` makes about span_frac, and this design
   inherits rather than solves it."

### 13.1 ATTACK-ROUND-1 (2026-07-07) — verdict NEEDS-MAJOR-REVISION

An independent adversarial pass reviewed this design before any code was
written, per house discipline (mirrors `KEY_ANCHORING_DESIGN.md`'s own
attack-round convention). Verdict: **NEEDS-MAJOR-REVISION** — 2 FATAL,
5 MAJOR, plus minors and a mandatory literature-reframe pass. Every finding
below is fixed in this revision (Rev 1); none is deferred or waved away.
Findings are recorded near-verbatim for the historical record, per house
style; resolutions are stated as landed in this text, not as intentions.

| # | Finding (attack-round-1) | Severity | Fix (Rev 1) | Location |
|---|---|---|---|---|
| F1 | Probe-eval entity pool arithmetically too small: `build_entity_pools()` yields 213 single-token names → 107 train / 106 heldout. §4.5's probe-train/probe-eval sub-split of the 107 pool (≈60/47) made the probe-eval pool 47 names, but `sample_batch_rd` draws K entities WITHOUT replacement and hard-asserts `N_names ≥ K` — the design's own K sweep needs K=48 (d=64 killer-prediction cell) and K=64/96 (d=128 rungs): construction impossibility at exactly the decisive cells | FATAL | Stopped sub-splitting the 107 pool. Probe-training AND probe-eval episodes both draw from the full 107-name pool, using disjoint EPISODE-seed ranges instead of disjoint entity sub-pools — every swept K (max 96) now fits with margin. The existing, already-disjoint 106-name heldout pool is repurposed as a strictly-stronger memorization-ceiling control (entities probe training never touched at all, not just a sub-split). Leakage argument restated explicitly: fresh per-episode random cycles + random query slots mean the (key,value) pairing an entity participates in is never a fixed fact, so raw-identity memorization cannot occur; the label-shuffle null (unchanged) and the new heldout-pool control jointly close the narrower identity-correlated-recoverability risk | §4.2, §4.5 |
| F2 | Frozen-bias blend is LIVE during the probe's own eval forward pass (`DeltaNetLMMixer.forward` applies it unconditionally for non-off arms), confounding "training changed the weights" with "the blend mechanically acts on the probe episode's own keys." §7.9 only defended the hook LOCATION, not this | FATAL | Added the mandatory 2×2 blend-toggle surgery grid to Leg A (3 of 4 cells populated; the 4th, off-ckpt-blend-forced-on, is named as not constructible). `frozen_bias_arm` is a plain attribute, overwritable post-load ("surgery") without touching weights — the same toggle fact `frozen_bias_retrofit_eval_rd.py`'s own mode dispatch already proves code-path-equal to the live blend. Pre-registered interpretation: training-effect = (arm ckpt, blend-OFF) vs. (off ckpt) now carries the headline H_LINK-A claim; mechanical-effect = (arm ckpt, blend-ON) vs. (arm ckpt, blend-OFF) reported separately, never load-bearing. New Stage −1 smoke gates the surgery mechanism. Budget re-derived: +5.01 GPU-h (36 passes, K=16 deferred as a disclosed trim) — Phase-1 total now ≈24.97 GPU-h, still under the 25 GPU-h ceiling but with a thin, disclosed margin | §5.2, §5.2a, §7.9, §9 item 6, §10 |
| M1 | CONFIRM gate never required Option-1/Option-2 agreement despite §13 item 2 naming Option 2 as the independent check; no mechanistic diagnostic (S_T conditioning, cross-a convergence) was mandatory anywhere | MAJOR | CONFIRM at the killer-prediction cell now REQUIRES Option 1/Option 2 directional agreement (defined via matching pinned CIs); disagreement routes to a new named outcome, READOUT-DIVERGENCE, reported as its own finding, not folded into CONFIRM/REFUTE/probe-invalid. `S_T` condition number/eigenvalue spread and a within-episode cross-`a` cosine-convergence check (the power-iteration degeneracy signature) added as mandatory zero-cost covariates at every headline cell | §5.2, §5.3, §8.3 |
| M2 | The measured chance-null (§8.4) was h=1-only but headline claims live at h=3/4 | MAJOR | Label-shuffle null-band construction extended to EVERY tested h∈{1,2,3,4} at the Stage-0 calibration cell (same episodes, scored at each h — no new generation cost); registered per-h pass floors; h=1 remains the probe-validity gate, h=2/3/4 become the chance baselines the headline numbers at those hops must clear | §8.4, §9 (Stage 0) |
| M3 | §13 item 4's C14-style buffer/query blank-out test was named mandatory but absent from Stage −1 (§9) | MAJOR | Added as Stage −1 item 5 with a concrete `1e-6` tolerance (matches item 2's own convention) and a registered zero-GPU build approach (direct `k_proj`/`k_conv1d` submodule calls on a real checkpoint, bypassing the CUDA-only `chunk_delta_rule` kernel entirely, since `k_eff` is captured strictly pre-kernel) — this also resolves attack-round-0's own §13 item 4 in full | §9 item 5, §13 item 4 |
| M4 | `conv_size` must parametrize episode construction per-checkpoint (`buf_len = conv_size−1`), else §7.1's position-decomposition closure could silently fail at rung-2/3 if their conv differs | MAJOR | Registered commitment: a FRESH `DeltaNetRDTaskConfig` is constructed per checkpoint with `conv_size` read from that checkpoint's own saved config — never one shared config/`buf_len` reused across checkpoints. New Stage −1 assertion (item 7) checks `checkpoint_config["conv_size"] == episode_cfg.conv_size` before any episode is generated for that checkpoint | §4.1, §4.3, §9 item 7 |
| M5 | §4.2's clause-order randomization proposal may have mischaracterized `grammar_rd.py` (render order is already fixed-slot with per-episode random entity→slot and random cycle structure — render/cycle adjacency may already be decorrelated) | MAJOR | Verified directly against `grammar_rd.py` (L450-461): clause render position IS a deterministic function of slot index (fixed-slot, not shuffled), and `succ` (the K-cycle) is drawn independent of slot position, making `P(succ[i]=i+1)=1/(K−1)` — uniform, no elevated local-shortcut risk, by construction. The proposed new shuffle mechanism is DROPPED; replaced with a Stage −1 empirical measurement (item 8: ≥10,000 episodes/K, measured render-adjacency rate vs. exact chance rate, bootstrapped CI) that adds the shuffle back only if a swept K's measured rate's CI excludes chance on the high side. §4.2's premise corrected in place, not silently patched | §4.2, §9 item 8 |
| minor | No explicit multiple-comparison statement for the non-primary grid | minor | New §8.6: Benjamini-Hochberg FDR (q=0.05) across every non-primary cell; the single killer-prediction contrast is explicitly exempt as the pre-registered, externally-anchored primary test | §8.6 |
| minor | No related-work paragraph situating `SCALE_TRANSFER_DESIGN.md` §5.5 item 2 (the never-run frontier-probe transplant this design supersedes/absorbs) | minor | New paragraph in §2 citing and distinguishing that never-executed transplant explicitly | §2 |
| minor | No build-time convention for `DeltaNetRDTaskConfig` `H_train`/`H_test` population on never-trained checkpoints | minor | Registered: `H_extra=()`, `H_test=(1,2,3,4)` (this design's full tested set), `H_train=(5,)` (a fixed, semantically-inert placeholder verified disjoint mod K from every swept K) | §4.3 |
| lit | Literature validation not yet folded in: Okpekpe & Orvieto (arXiv:2508.19029) rival optimization-brittleness account; Frozen-QK (arXiv:2506.01115) nearest-neighbor intervention; VLA (arXiv:2605.11196) geometric rival account vs. §14 coherence-exoneration; OpenReview R8ZbLi3oUv convergent motivation; novelty positioning (combination, not bare cliff/ladder claims) | (reframe, not a defect) | All five folded into new §2.1, engaged head-on rather than left for a reviewer to discover | §2.1 |

**What Rev 1 could NOT cleanly fix, disclosed rather than hidden:** the
budget margin after F2's mandatory addition is genuinely thin (≈0.03 GPU-h,
§10) — this is not a clean fix, it is a disclosed, tight fit under an
unchanged 25 GPU-h ceiling, with a named next-lever trim held in reserve if
Stage 0's real cost runs over. The OpenReview R8ZbLi3oUv full read remains
blocked (bot-check) — adjudicated non-scoop from search-snippet
reconstruction alone, with the residual uncertainty disclosed in §2.1, not
resolved. Attack-round-0's own item 3 (does the arm-blind probe under-report
a genuinely different-geometry arm) and item 6 (does the extra frozen table
change training dynamics through a channel other than key geometry) remain
open — attack-round-1 did not raise or resolve either, and this revision
does not claim to have closed them; they stay registered as open questions
for a future round, per §13 items 3/6 above, unchanged.

### 13.2 ATTACK-ROUND-2 (2026-07-07, fresh eyes) — verdict NEEDS-REVISION

An independent adversarial pass (a different reviewer from attack-round-1,
per house discipline for a second round) reviewed Rev 1 before any code was
written. Verdict: **NEEDS-REVISION** — 1 new FATAL (a whole new axis
round-1 missed entirely), 7 MAJOR, 2 minor. Every finding below is fixed in
this revision (Rev 2); none is deferred or waved away. Findings recorded
near-verbatim for the historical record, per house style; resolutions are
stated as landed in this text, not as intentions.

| # | Finding (attack-round-2) | Severity | Fix (Rev 2) | Location |
|---|---|---|---|---|
| FATAL | `T_bind = K × clause_len(conv_size)` (`clause_len=7` at `conv_size=4`, `grammar_rd.py` L316-322) crashes `chunk_delta_rule`'s hard `T ≥ _MIN_KERNEL_T=128` assert (`lm_pretrain_rd.py` L831-836, `model_rd.py` L117) at `K=8` (`T_bind=56`) and `K=16` (`T_bind=112`) — affecting BOTH the `d=64` row (used both) and the `d=128` row (used 16). Unlike Wave 1's from-scratch harness, LM-mode checkpoints have no state-neutral pad token (learned, non-hard-masked `β` means padding would genuinely write into `S_T` — a structural feature correlated with the load axis itself, i.e. a confound by construction) | FATAL | Raised the K floor everywhere instead of padding. New general formula `K_min(conv_size) = ceil(_MIN_KERNEL_T / clause_len(conv_size))` (§4.1), `=19` at `conv_size=4`. Re-derived K sweeps: `d=64` committed `{20,32,48}` (killer cells 32/48 kept, K=20 new floor-safe low-load contrast at load 0.3125), `d=128` committed `{20,64}` (the actual near-cliff point, corrected from Rev 1's erroneous `{8,32}`, MAJOR-1); `K=40` (`d=64`) and `K∈{32,96}` (`d=128`) registered as named, separately-priced extensions. New Stage −1 item 9: a per-checkpoint, per-registered-K assertion of `K ≥ K_min(conv_size)`, using each checkpoint's own verified `conv_size` (ties to item 7), never a value trusted once at design time | §4.1, §4.3, §9 item 9 |
| M1 | §10 trim #2 said "K∈{8,32}" for the `d=128` Leg-B first pass, but §4.3 defined `d=128`'s set as `{16,32,64,96}` — K=8 was never a valid `d=128` point, AND K=32 is not `d=128`'s near-cliff point (K=64 is, per §4.3's own `0.5455×128≈69.8` siting; §6.2's own headline reading already used K=64, which Rev 1's committed budget never actually funded) | MAJOR | Corrected committed `d=128` pair to `{20 (low-load), 64 (near-cliff)}`, deferring `{32,96}` as the extension — a more decisive first-pass selection than Rev 1's, not a relabeling, and one that makes §6.2's own headline claim affordable under the committed budget for the first time | §4.3, §10 |
| M2 | §5.2a's "4th cell (off-ckpt, blend forced ON) not constructible" is factually wrong: `frozen_bias_retrofit_eval_rd.py`'s Arm-1′/1″ modes already blend a seed-derived frozen table (`build_frozen_bias_table(..., seed=frozen_bias_seed)`, default `ANCHOR_INIT_SEED`) onto ANY loaded checkpoint at eval, including one that never trained with a table — no training history required, verified directly against the tool's own `run_retrofit_measurement` | MAJOR | Rewrote §5.2a's 2×2 grid: the 4th cell IS constructible, registered as DEFERRED ON BUDGET (≈2.51 GPU-h, priced in §10's extension table), with its purpose stated (decomposes whether the mechanical blend effect requires prior training-with-blend or fires cold) and a trigger condition (run if the mechanical-effect contrast is itself significant at the killer-prediction cell) | §5.2a, §10 |
| M3 | Stage −1 item 6's surgery-toggle smoke is vacuous: it calls `k_proj`→`k_conv1d` directly, bypassing `mixer.forward()` entirely — the `frozen_bias_arm` gate lives strictly inside `forward()`, so the test cannot fail regardless of whether the surgery mechanism works | MAJOR | Rewrote item 6 to reproduce `forward()`'s own pre-kernel sequence verbatim (through the blend conditional, reading the model's live `frozen_bias_arm` attribute), with two genuine negative controls: (a) live-gate-fires (arm≠off must differ from arm=off by >1e-3), (b) surgery-equivalence (arm forced off must match `"kraw"`-mode to 1e-6) — a test that can now actually fail on either axis | §9 item 6 |
| M4 | The heldout-pool memorization-ceiling control was priced as one draw per `d_state`, but H_LINK's claims ride on ARM/RUNG comparisons — an identity-memorization DIFFERENTIAL between conditions would be invisible to a single pooled draw | MAJOR | Control now run at every killer-prediction cell (Leg A: 3 arms × 2 corpora × killer-K's) and Leg B's primary near-cliff-K cells, n=1 representative-seed priced (§10, 3.79 GPU-h), with an explicit n=1 point-estimate tripwire (0.5× the headline CI half-width) and a priced n=3 escalation path; new named outcome **MEMORIZATION-CONFOUND** (§8.5 item 5) demotes CONFIRM when the differential is real | §4.5, §8.5, §10 |
| M5 | F1's episode-seed ranges were an example ("e.g., offset 0... a disjoint offset block"), not pinned numbers, and never accounted for the other seed axes (3 checkpoint seeds × 2 corpora × 3 arms/4 rungs × up to 6 K's) that could silently collide | MAJOR | New §4.6: one flat mixed-radix formula (`PURPOSE_BASE + LEG_BASE + condition_idx×STRIDE_CONDITION + corpus_idx×STRIDE_CORPUS + ckpt_seed_idx×STRIDE_SEED + k_idx×STRIDE_K`) with strides sized so no combination can collide by construction; new Stage −1 item 10 mechanically verifies `len(set(seeds))==len(seeds)` over every registered cell rather than trusting the construction argument alone | §4.6, §9 item 10 |
| M6 | The 1.31B (rung-3) pass cost used in §10 is an unmeasured extrapolation from the 392M anchor, and the three existing anchors are non-monotonic (0.0348→0.013→0.045 GPU-h/pass) — there is no justified functional form to extrapolate from | MAJOR | New Stage 0.5 (§9): a single, blinded (no h4-style readout), 1-checkpoint/1-K (K=64) rung-3 timing cell, priced at 0.36 GPU-h (2× the 392M anchor, matching its own abort threshold), BEFORE the rung-3 grid is trusted. Two-tier abort trigger: >2× anchor drops rung-3's non-primary K; >4× anchor halts Leg B's rung-3 rows entirely and discloses | §9 (Stage 0.5), §10 |
| M7 | Rev 1's trim #3 (K=16 deferred from the surgery-only pass, but retained in the native/blend-ON grid) collides with §8.6's BH-corrected reporting: without a blend-OFF surgery reading at that K, its row cannot be attributed to training vs. mechanical blend action — reporting it in any training-effect table would silently smuggle in the confounded blend-ON number | MAJOR | Standing rule registered in §8.6: any K without its own surgery pass (now `K=40`, this revision's equivalent point) is reported ONLY in the mechanical-effect table, explicitly labeled, and is NEVER promoted into a training-effect table without its surgery extension being run first — since `K=40` is now demoted to a committed-grid-excluded extension entirely (the FATAL fix's own byproduct), this also resolves the immediate case, not just the general rule | §8.6 |
| minor | §5.3's M1 agreement gate never specified a multi-K conflict rule now that `K=32`, `K=48`, and (optionally) `K=40` all report Option 1/Option 2 agreement | minor | Registered: `K=32` (closest to the located cliff center) is the single primary agreement-gate K; `K=48`/`K=40` report but never override the primary verdict | §5.3 |
| minor | §4.4 Option 2's cost was unstated — a full-sequence LM head forward risks the standing-known 50K-vocab logits-tensor VRAM bottleneck | minor | Stated explicitly: logits computed at the query position only (true-answer + K−1 distractors), never materializing the full-sequence LM head output | §4.4 |

**What Rev 2 could NOT cleanly fix, disclosed rather than hidden:** the
committed budget margin (≈0.05 GPU-h) remains genuinely thin — marginally
better than Rev 1's ≈0.03, not a comfortable cushion; the abort rules, not
this arithmetic, are the real safety valve, exactly as Rev 1 already stated
and this revision restates rather than papers over. The freed-vs-spent
reconciliation (§10) landing almost exactly at Rev 1's own total is
disclosed as a coincidence of the arithmetic, not an engineered outcome.
Attack-round-0's items 3/6 and the OpenReview R8ZbLi3oUv blocked full-read
(§13's own unresolved items, restated at the end of §13.1) remain open;
attack-round-2 did not raise or resolve either, and this revision does not
claim to have closed them.

### 13.3 ATTACK-ROUND-3 (2026-07-07, fresh eyes) — verdict NEEDS-REVISION

A third independent adversarial pass (a different reviewer from
attack-rounds 1 and 2, per house discipline) reviewed Rev 2 before any code
was written. Verdict: **NEEDS-REVISION** — 2 new FATALs, both in the shared
instrument (§4.4) itself, an axis neither prior round examined closely
(round 1 attacked the episode/probe-protocol layer, round 2 the
kernel-floor/budget layer; neither ever type-checked the readout formula or
walked the extraction path through a multi-layer checkpoint), plus 1 MAJOR
and 1 minor. **Every round-2 fix was re-verified by this pass and HOLDS** —
the two FATALs are new findings in previously-unattacked territory, not
regressions of anything round 2 fixed. Every finding below is fixed in this
revision (Rev 3); none is deferred or waved away. Findings recorded
near-verbatim for the historical record, per house style; resolutions are
stated as landed in this text, not as intentions.

| # | Finding (attack-round-3) | Severity | Fix (Rev 3) | Location |
|---|---|---|---|---|
| FATAL-1 | §4.4's Option 1 readout could not be evaluated as written: it declared a trained probe `W_probe: R^{d_state} → R^{d_state}` but cosine-scored its output against `W_embed[value_token_id] ∈ R^{d_model}` — and `d_state ≠ d_model` at EVERY checkpoint in scope (64 vs. 256/768; 128 vs. 1536/2560). A dimension mismatch, not a tuning problem; the entire §4.5 probe-training/leakage/memorization apparatus inherited the hole | FATAL | Dropped the trained probe ENTIRELY. Score in `d_state` space against the TARGET ENTITY'S OWN `k_eff` (gathered at `tgt_slot` from forward-A, through the same per-layer `k_conv1d` hook as the bind keys and `q_eff`), absolute cosine ≥ 0.9 — mirroring `DELTANET_REALDATA_DESIGN.md` §5.2/§14.3's established convention (same-space scoring, absolute cosine, 0.9 threshold, exact continuous recovery, never argmax), with one specified, doubly-justified departure: compare against `k_eff` (same `W_k` family as `q_eff` at every hop; zero new hook code — no `v_conv1d` hook exists in this codebase) rather than `v_eff`. Consequences propagated: §4.5 rewritten (probe-training protocol, probe-train/probe-eval seed split, arm-blind machinery all OBSOLETE and removed; label-shuffle null SURVIVES rebuilt probe-free as the §8.4 chance floor); heldout-pool memorization control adjudicated to measure NOTHING probe-free (no fitted weights to leak; pools are a pure random shuffle of one curated name distribution — no familiarity axis) and REMOVED with its 3.79 GPU-h line and the MEMORIZATION-CONFOUND outcome (§8.5); attack-round-0 item 3 (arm-blind probe under-reporting) DISSOLVES (recorded at §13 item 3); §10 re-derived (probe lines freed, ≈3.89 GPU-h) | §4.4, §4.5, §4.6, §7.4, §8.1, §8.2, §8.5, §10, §13 item 3 |
| FATAL-2 | Query-phase extraction was unbuildable as written for multi-layer checkpoints: `grammar_rd.py` builds queries as SEPARATE short tensors (`query_len=6`), but the production mixer hard-asserts `T ≥ _MIN_KERNEL_T=128` BEFORE the k/q/v convs (`lm_pretrain_rd.py` L831-836), and layer `i>0`'s input is layer `i−1`'s residual output — itself requiring a valid `T≥128` forward. `model_rd.py::effective_key_window`'s direct-submodule short-window precedent is a SINGLE-mixer construction that does not generalize; layer-0-only extraction would silently break the per-layer `S_T` framing at rung-2/3 (`n_layers` 16/22) | FATAL | Adopted the TWO-FORWARD protocol: forward-A = bind-only sequence (`T_bind ≥ 128` guaranteed by §4.1's K-floor) → per-layer `S_T` via the mixer's own final-state path + all bind `k_eff` (incl. the target's); forward-B = bind+query CONCATENATED (`T_bind+query_len`, trivially ≥128) → per-layer `q_eff` from the `k_conv1d` hook at the query positions. Registered: (i) a disclosed, eval-only deviation from grammar_rd's "query never enters the streamed sequence" training convention — `S_T` comes from forward-A, which the appended query cannot retroactively affect (causality); (ii) `q_eff` thereby defined IN SITU (the model's own contextual multi-layer processing of the query clause) — the honest, well-defined choice for multi-layer checkpoints; (iii) new Stage −1 item 11: forward-A/forward-B residual streams asserted IDENTICAL over the bind prefix (bit-identical fp32/CPU, or `1e-6` at the bf16 kernel boundary on GPU); (iv) cost re-derived honestly — the anchors are single-forward measurements that do NOT amortize a second pass, so the per-pass rate doubles (anchor×4 → anchor×8), §10's rows re-priced and the grid re-scoped per-leg to stay under the 25 GPU-h ceiling (old→new reconciliation printed in §10) | §4.4, §9 item 11, §9 Stage 0, §10 |
| MAJOR | The Leg-B `d=128` "8 passes"-style budget lines assumed 4 rungs × 2 corpora uniformly, but rung-3's actual configuration was left as an open "confirm at harvest time" question (§6.1: "likely 1 seed × 2 corpora, or 2 seeds × 1 corpus") — two configurations with materially different corpus coverage, silently unpinned under budget lines that priced one of them | MAJOR | PINNED from the actual launch artifacts, not guessed: `run_lm_rd_trackc_sweep.py::wave23_manifest` (L100/L868-869, "rung 3 stays '2 corpora x 1 seed' as registered") and `STATE.md`'s launch record both fix rung-3 at **2 corpora × 1 seed** (openr1-mix-ext + wikitext-mix-ext, one run each). §6.1 rewritten with the pin and its consequence (each corpus gets a complete 4-rung ladder at n=1, vs. the alternative's zero-coverage corpus); every rung-3-touching row in §10 written against this configuration. NOTE: the orchestrator's own fix directive suggested pinning to "2 seeds × 1 corpus openr1-mix-ext" — the launch artifacts contradict that suggestion, and reality wins; recorded here so the discrepancy is visible, not silently resolved | §6.1, §10 |
| minor | Batch size for GPU passes was never registered anywhere — every cost anchor implicitly assumed one, and a build-time choice could silently diverge from what the anchors measured | minor | PINNED `batch_size=16` (§4.2), verified as the shared default of both anchor-generating tools (`lm_attractor_probe_rd.py`/`frozen_bias_retrofit_eval_rd.py` `--batch-size` defaults) AND the batch rung-3's own Rev 2.2 pricing was measured at (`SCALE_TRANSFER_DESIGN.md`); Stage 0/0.5 calibration validates it on this design's own episodes | §4.2, §9 |

**What Rev 3 could NOT cleanly fix, disclosed rather than hidden:** the
committed grid is genuinely NARROWER than Rev 2's — the two-forward tax
was paid mostly in grid resolution (K=48/K=40/Leg-B K=20 demoted to
extensions), not found for free; §5.3's killer prediction executes at a
two-point contrast with its past-cliff corroboration point (K=48) gated
behind an extension, and §6.3's fixed-K sanity contrast is unfunded within
the committed grid. The improved margin (≈0.80 vs. ≈0.05 GPU-h) reflects
that narrowing, stated plainly in §10, not a real cost saving. The
adaptation of `DELTANET_REALDATA_DESIGN.md` §5.2's scoring convention
substitutes `k_eff` for `v_eff` as the comparison target — argued from
representational-family consistency and hook-mechanics in §4.4, but it IS
a departure from the letter of the validated convention, registered as
such rather than claimed identical. Attack-round-0's item 6 (frozen-table
training-dynamics channel) and the OpenReview R8ZbLi3oUv blocked full-read
remain open — this round did not raise or resolve either. Attack-round-0's
item 3 is now DISSOLVED (not answered) by FATAL-1's fix, per §13 item 3.

### 13.4 ATTACK-ROUND-4 (2026-07-07, fresh eyes) — verdict NEEDS-REVISION

A fourth independent adversarial pass (a different reviewer from
attack-rounds 1-3, per house discipline) reviewed Rev 3 before any code
was written. Verdict: **NEEDS-REVISION** — 1 new FATAL (a symbolic-algebra
proof that Rev 3's own readout-target choice is unsound), 1
FATAL-ADJACENT (a gate applied to only one of two legs that structurally
needed it), 2 MAJOR (a missing outcome category, and a framing
adjudication the design had been implicitly relying on without stating),
1 MINOR. **Every round-1/2/3 fix was re-verified by this pass and HOLDS**
— the FATAL is a new finding in previously-unattacked territory (no prior
round did the actual linear algebra on what family `S_Tʰ q_eff` lands in;
round 3 attacked the readout's *extraction mechanics* — dimensions,
multi-layer legality — never its *algebra*), not a regression of anything
round 3 fixed. Every finding below is fixed in this revision (Rev 4); none
is deferred or waved away. Findings recorded near-verbatim for the
historical record, per house style; resolutions are stated as landed in
this text, not as intentions.

**The attacker's derivation, reproduced in full — the single most
valuable analysis this design has received.** Prior rounds attacked
construction (episode arithmetic, kernel floors, seed collisions) and
mechanics (dimension typing, multi-layer extraction legality); this round
is the first to ask what the readout formula's own linear algebra actually
implies, and the answer overturns a load-bearing design choice three prior
rounds left unexamined. Restated here verbatim in substance (condensed
from the reviewer's own working):

> The delta rule updates `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`.
> **Claim: `Im(S_T) ⊆ span{v_eff_1,...,v_eff_K}`, by induction on `t`, for
> ANY scalar `β_t` (not merely `β_t ∈ [0,1]`).**
> *Base case* (`t=0`, `S_0=0`): `Im(S_0)={0} ⊆ span{v_eff_j}`, trivially.
> *Inductive step*: assume `Im(S_{t-1}) ⊆ span{v_eff_1,...,v_eff_{t-1}}`.
> For any `x`: `S_t x = S_{t-1}x − β_t(k_tᵀx)S_{t-1}k_t + β_t(k_tᵀx)v_t`.
> The first two terms are `S_{t-1}` applied to some vector — both land in
> `span{v_eff_1,...,v_eff_{t-1}}` by the inductive hypothesis; the third
> term is a scalar multiple of `v_t`. So `S_t x ∈
> span{v_eff_1,...,v_eff_t}` for every `x`. By induction,
> `Im(S_T) ⊆ span{v_eff_1,...,v_eff_K}` — independent of every `β_t`'s
> actual value; the argument never used `β_t ∈ [0,1]`, only that it is a
> scalar.
> **Consequence:** for `pred(a,h) = S_Tʰ·q_eff_a`, `h=1` gives
> `S_T q_eff_a ∈ Im(S_T) ⊆ span{v_eff_j}` directly; for `h≥2`,
> `S_Tʰ q_eff_a = S_T(S_T^{h-1}q_eff_a)` lands in `Im(S_T)` regardless of
> what vector `S_T` is applied to, since `Im(S_T)` is fixed. **So
> `pred(a,h)` is a `v_eff`-family (value-space) object at EVERY `h≥1`, with
> no `h` at which it is ever a `k_eff`/`W_k`-family object.** Rev 3's own
> stated rationale for comparing against `k_eff` — that `q_eff` and
> `k_eff` share the `W_k` family, so comparing `pred(a,h)` against `k_eff`
> "keeps every hop apples-to-apples" — examined the wrong object's family:
> `pred(a,h)`'s family is fixed by `Im(S_T)`, not by which family its
> *input* (`q_eff`) happened to belong to before `S_T` was applied.
> Rev 3's `k_eff_target` scoring was therefore cross-family with **no
> established referent connecting a value-space prediction to a
> key-space target**, at `h=1` exactly as much as at `h≥2` — the "h=1
> never needs to bridge families" argument Rev 3 used to justify treating
> `h=1` differently does not hold, because `h=1` is already in
> value-space, same as every other `h`.

| # | Finding (attack-round-4) | Severity | Fix (Rev 4) | Location |
|---|---|---|---|---|
| FATAL | Rev 3's Option 1 readout scored `pred(a,h)=S_Tʰ·q_eff_a` against the target's own `k_eff`, on the stated rationale that this keeps the comparison inside the `W_k` family at every hop. The symbolic-algebra derivation above proves `Im(S_T) ⊆ span{v_eff_j}` for any `β` (by induction on the delta-rule update), so `pred(a,h)` is a `v_eff`-family object at EVERY `h≥1`, never a `k_eff`-family one — Rev 3's scoring was cross-family with no established referent, broken at `h=1` exactly as much as at `h≥2`; Rev 3's own justification for treating `h=1` as safe was backwards | FATAL | Reverted the comparison target to `v_eff_target`, restoring `DELTANET_REALDATA_DESIGN.md` §14.3's exact original convention with zero specified departure. Three things travel with the revert, all built out: (a) the trained probe stays DROPPED — `v_eff` is exactly as `d_state`-dimensional as `k_eff` was, so Rev 3's FATAL-1 fix (never project into `d_model`) is untouched (§4.4, §4.5); (b) a new `v_conv1d` forward hook — a ~5-line copy of the existing `k_conv1d` hook pattern, correcting §4.4's Rev 3 "no such observable exists" overclaim (the `v_conv1d` submodule already exists at `lm_pretrain_rd.py` L793 and already runs on the same forward pass, only the hook was missing) — registered alongside `k_conv1d` on the SAME forward-A call, zero new GPU passes (§4.4, §10); (c) the scoring target is gathered with the CORRECT hop indexing, carrying `DELTANET_REALDATA_DESIGN.md` §14.2's audit-fix forward: `prev_slot = _iterate_permutation(succ, a_slot, hops-1)`, `v_eff_target = v_eff_items[prev_slot]` (NOT `v_eff_items[tgt_slot]`, which would repeat the exact one-hop-past bug §14.2 already found and fixed once), with a hand-worked 3-entity example and a new Stage −1 self-test (item 12) verifying the index arithmetic end to end against a deliberate negative case. Also added: PREMISE DIAGNOSTIC (iv), `cos(k_eff_i, v_eff_i)` per entity, measured at Stage 0 calibration alongside premises (i)-(iii) — makes the multi-hop compounding premise (cross-role identity at intermediate entities) MEASURED, wired into how far past `h=1` the formula's `h≥2` numbers should be trusted (§4.4, §9) | §1, §4.4, §4.5, §7.4, §8.1, §9 items 12-13, Stage 0, §13.4 |
| FATAL-ADJACENT | §5.2's Rev 1 M1 mandatory Option 1/Option 2 agreement gate was wired ONLY to Leg A's killer-prediction cell — Leg B's own scale-trend claim (§6.2) rides on the identical readout and the identical concerns about what it measures at `h≥2`, yet had no cross-check of its own | FATAL-ADJACENT | Extended the mandatory agreement gate to Leg B's own primary near-cliff cells (`K=32` at `d=64`'s two rungs, `K=64` at `d=128`'s two rungs — the exact cells §6.2 already reads as its headline). A scale-trend CONFIRM for H_LINK-B may be claimed only over the subset of the 4 ladder rungs where Option 1/Option 2 agree, and requires a minimum of 3 agreeing rungs; per-rung disagreement routes that rung to READOUT-DIVERGENCE; fewer than 3 agreeing rungs demotes the scale claim to a new named outcome, AMBIGUOUS, rather than a trend built on a minority of the ladder. Priced at zero additional GPU cost — Option 2's reading at each rung's already-committed primary K comes free from the same forward-B pass Option 1 already runs there | §6.2, §12 |
| MAJOR | §12's outcome table had no category for the specific observationally-indistinguishable failure mode where `h=1` clears the chance floor (the probe is valid) but `h≥2` sits at floor UNIFORMLY across every arm and rung — as written, this would have been silently read as Cell 4 ("stabilization is functionally inert"), when it is equally consistent with "Option 1 is testing the wrong mechanism for every checkpoint here" | MAJOR | Added READOUT-FORM-INVALID as a sixth named outcome (alongside READOUT-DIVERGENCE), distinct from Cell 4/no-capability: triggers on the uniform-floor signature above; bars Cell-4-style "functionally inert" claims under it; makes Option 2 the interpretive instrument of record for `h≥2`, with the honest report stated as "no evidence of `S_T`-self-iteration composition in text-pretrained checkpoints; the behavioral readout says X" — never silently substituting one instrument for the other | §12 |
| MAJOR | The design never stated, as an explicit adjudication, that Option 1 tests one specific (and a priori unlikely) mechanism — single-layer state self-iteration — while Option 2 is mechanism-agnostic; nor did it disclose that `q_eff` (extracted from the FINAL layer) already embeds every earlier layer's own state-reads, so Option 1 measures the full stack's terminal computation, not an isolated single-layer read. Without this stated explicitly, a reviewer would reasonably read a low absolute Option-1 level at `h≥2` as evidence against composition capability, when it may only be evidence against ONE mechanistic hypothesis for how that capability is organized | MAJOR | Folded the framing adjudication into §1 (as part of what H_LINK does NOT require Option 1 to prove in absolute terms) and into §5.1 (a scope-setting note that Leg A's comparisons are read as contrasts, never as absolute-recovery claims), with the fuller derivation and the query-circularity disclosure at §4.4. States explicitly: ARM/SCALE CONTRASTS measured through Option 1 remain valid comparisons even where its absolute level is compressed by testing the wrong mechanistic hypothesis, because the killer prediction (§5.3) and the scale-trend reading (§6.2) are both stated as CONTRASTS/deltas, never as absolute-recovery thresholds | §1, §4.4, §5.1 |
| MINOR | §10's K=48 extension trigger read "the committed K∈{20,32} killer contrast passes or trends" — not a mechanical, checkable rule | MINOR | Pinned one mechanical rule, replacing "passes or trends": promote iff, at the committed K=32 cell in at least one corpus, EITHER the training-effect Δ's pinned CI excludes zero on the positive side (K=32 already CONFIRMs), OR Δ's point estimate is positive and ≥50% of the CI's own half-width while the CI still straddles zero — a self-referential rule measured against the cell's own noise floor, inventing no new external effect-size number | §10 |

**BUDGET (verified, not merely asserted):** the FATAL fix is budget-neutral
— same number of GPU passes, the new `v_conv1d` hook fires on an
already-priced forward call, no probe is reintroduced. §10's Rev 3 total
(**≈24.20 GPU-h committed, ≈0.80 GPU-h margin, ≈38.70 GPU-h named-extension
reserve, unchanged 25 GPU-h ceiling**) is re-printed, not re-derived, with
an explicit Rev 4 confirmation note (§10).

**Straggler sweep (performed, not merely promised):** every occurrence of
`k_eff_target` as the CURRENT scoring target, every "`W_k`-family"
rationale sentence, §8.1's primary-metric formula, and every §14 checklist
line naming the comparison object were located (`grep`) and fixed to the
`v_eff` convention; historical fix-map tables at §13.1-§13.3 (this
section's own predecessors, describing what Rev 1/2/3 actually did at the
time) are left intact, per this project's own "record history, don't
retcon it" convention — where a historical record's own wording needed a
forward-pointing correction for a reader's sake (§13 item 3's dissolution
note; the §14 checklist, which is a LIVE checklist rather than a dated
historical record), a parenthetical Rev 4 note was added alongside the
original text rather than the original being silently rewritten.

**What Rev 4 could NOT cleanly fix, disclosed rather than hidden:** the
premise-(iv) diagnostic (§4.4) makes the multi-hop compounding assumption
measured rather than assumed, but it cannot be evaluated until Stage 0
actually runs — this revision registers the diagnostic and its
interpretive wiring, it does not (cannot, pre-data) resolve whether `h≥2`
composition is in fact reachable for these checkpoints. The
framing-adjudication fix (§1/§4.4/§5.1) is a stated discipline for how to
READ Option 1's absolute levels, not a fix that makes Option 1 measure the
"right" mechanism — if this checkpoint family in fact composes via
cross-layer chaining rather than single-layer self-iteration, Option 1
may still show a uniform floor at every arm/rung regardless of any true
underlying capability difference, which is exactly why READOUT-FORM-INVALID
exists as an honest outcome rather than a problem this revision claims to
have solved. Attack-round-0's item 6 (frozen-table training-dynamics
channel) and the OpenReview R8ZbLi3oUv blocked full-read remain open —
this round did not raise or resolve either.

### 13.5 ATTACK-ROUND-5 (2026-07-07, fresh eyes) — verdict NEEDS-REVISION

A fifth independent adversarial pass (a different reviewer from
attack-rounds 1-4, per house discipline) reviewed Rev 4 before any code
was written. Verdict: **NEEDS-REVISION** — 1 new FATAL (every `q_eff`
extraction in this design hooked the wrong projection), 2 MAJOR (an
unfalsifiable premise-(iv) ornament with no null and no consequence, and
a chance-floor backstop gap that lets a near-zero null band pass almost
anything), 4 MINOR. **Every Rev 1-4 fix was independently re-verified by
this pass and HOLDS** — including §13.4's own gather-indexing fix
(`prev_slot = _iterate_permutation(succ, a_slot, hops-1)`), re-derived by
hand against the 3-entity worked example and confirmed correct. The
FATAL is a new finding in previously-unattacked territory: no prior round
checked which PROJECTION `q_eff` is actually read through in the real
checkpoint code — round 3 established the two-forward EXTRACTION
MECHANICS (dimensions, multi-layer legality) and round 4 proved the
readout's own ALGEBRA (which family `pred(a,h)` lands in); this round is
the first to check the much more basic fact of which trained weight
matrix a hooked tensor actually comes from. Every finding below is fixed
in this revision (Rev 5); none is deferred or waved away. Findings
recorded near-verbatim for the historical record, per house style;
resolutions are stated as landed in this text, not as intentions.

**The attacker's finding, reproduced in substance — a source-reading
check no prior round ran.** Every occurrence of `q_eff` extraction in
this design, through Rev 4, read: "`q_eff` extracted via the **same**
`k_conv1d` hook at the query's own final position." That sentence is
true of `model_rd.py`'s own from-scratch harness — verified directly,
that harness has **no `q_proj` at all**, and its own
`effective_key_window` function (named for exactly this reason) is the
ONLY read-path available, so reading the query through `W_k` there is
forced, not chosen. It is false of `lm_pretrain_rd.py`'s
`DeltaNetLMMixer`, the class every checkpoint in Leg A and Leg B actually
is: its own header comment states plainly that "this LM block DOES use a
real, learned q_proj/q_conv1d" (L45-49) — realized as `self.q_proj =
nn.Linear(...)` and `self.q_conv1d = ShortConvolution(...)` (L787, L791),
computed on every forward (`q, _ = self.q_conv1d(self.q_proj(x))`, L837),
and fed to the kernel as `chunk_delta_rule(q=q_bf, k=k_bf, v=v_bf, ...)`
(L983) — `q_bf`, not `k_bf`. No design-level algebra was wrong here; a
convention that was FORCED in one codebase was copied into a different
codebase where it was merely CONVENIENT, and the difference was never
checked against the actual source before this round.

| # | Finding (attack-round-5) | Severity | Fix (Rev 5) | Location |
|---|---|---|---|---|
| FATAL | Every `q_eff` extraction in this design, through Rev 4, said `q_eff` comes "via the same `k_conv1d` hook at the query's own final position." This checkpoint family has a real, separately-trained `q_proj`/`q_conv1d` path that `chunk_delta_rule` actually reads through (`lm_pretrain_rd.py` L45-49 header comment, L787-793 the three untied projections/convs, L837-839 the per-token conv calls, L983 `chunk_delta_rule(q=q_bf, ...)`) — the from-scratch harness's own forced convention (no `q_proj` at all) was carried over unexamined into a checkpoint family that does not share the constraint that forced it | FATAL | Retargeted forward-B's query-position extraction to a new `q_conv1d` hook (~5-line copy of the existing `k_conv1d`/`v_conv1d` hook pattern, zero new GPU passes, mirroring Rev 4's own `v_conv1d` hook exactly). Swept every `k_conv1d` mention across the document and adjudicated each: bind-phase `k_eff` extraction (forward-A), the frozen-bias blend/surgery tooling, and the buffer-blank-out tests correctly stay `k_conv1d`/`v_conv1d` (UNCHANGED); every mention describing `q_eff` extraction (§4.4's Forward-B bullet, the new FATAL callout box, §4.4's "Justification for choosing Option 1" paragraph, §7.9's cross-arm confound argument) is corrected to `q_conv1d`. Historical fix-map tables (§13.1-§13.4) describing what earlier revisions ACTUALLY did are left intact per the "record history, don't retcon it" convention. New Stage −1 item 14 (`q_conv1d` hook equivalence smoke, mirroring item 13). Premise (iii) (bind↔query alignment) is now a genuine measured quantity as a direct consequence — see MAJOR-1 | §0, §4.4, §7.9, §9 item 14, §13.5 |
| MAJOR-1 | Premise (iv)'s "wired into outcome interpretation" text reused R2-2's 0.9 alignment bar for a cross-role comparison whose null is near-zero, with no justification, and "disclosed alongside" had no stated consequence | MAJOR | Replaced with one mechanical, null-relative action-rule table covering BOTH premise (iii) (now genuinely measured via the `q_conv1d` fix, above) and premise (iv): measure the per-entity same-entity cosine distribution AND its null (cross-entity shuffled pairs) at Stage 0; if the median same-entity cosine fails to exceed the null's 95th percentile, the dependent hop-depths are pre-declared exploratory-only BEFORE unblinding — premise (iii)'s failure demotes `h=1` itself (`h=1` depends only on premise iii); premise (iv)'s failure demotes `h≥2` only (`h=1` retains confirmatory status if premise iii alone passes) and READOUT-FORM-INVALID becomes the expected, not surprising, outcome for `h≥2`. No 0.9 magic number anywhere in the new rule | §4.4, §9 (Stage 0 items d/e) |
| MAJOR-2 | The null-band-relative h1 pass rule (§8.4) degenerates to "almost any recovery passes" if the measured null band itself sits near zero | MAJOR | Added an absolute backstop: `h=1`'s `recovered_frac@0.9` must ALSO exceed a registered fixed minimum, **0.10**, independent of the null band — both conditions required, either failing routes to probe-invalid. Justified from this project's own exactness-program h=1 guard (1.0, for TRAINED-on-task checkpoints — not reusable here per §7.10's own never-task-trained caveat) and the litreview's Zoology (arXiv:2312.04927)/Based (arXiv:2402.18668) line (fixed-state associative-recall capacity is a documented property in this size class), landing well above chance at the smallest committed K (K=20, chance≈0.053) | §8.4, §4.4 |
| MINOR (1) | Stage 0.5's abort-trigger structure assumed the rung-3 calibration cell completes and times cleanly; no registered behavior existed for an outright OOM (a distinct failure mode from "slow but completes") | minor | Registered fallback: retry once at `batch_size=8` (half the pinned 16), then halt Leg-B rung-3 rows with disclosure if the retry also OOMs — a bounded, one-shot fallback, never an open-ended crash-loop, consistent with the project's standing anti-retry-loop discipline | §9 (Stage 0.5) |
| MINOR (2) | §4.4 never stated explicitly that `apply_frozen_bias_blend` touches only `k` (verified L854-857) — a fact that, once stated, is a disclosed design strength of the `v_eff` (and now `q_eff`) scoring convention | minor | Added an explicit statement in §4.4 (and strengthened §7.9's cross-arm confound argument to match): the blend conditional wraps exactly one reassignment (`k = apply_frozen_bias_blend(k, ...)`), never `q` or `v` — so `v_eff_target` and `q_eff` are blend-invariant BY CONSTRUCTION across all three Leg-A arms, not merely code-path-identical | §4.4, §7.9 |
| MINOR (3) | §12 had no reporting rule for the case where Leg A's killer-prediction cell routes to READOUT-DIVERGENCE while Leg B's own cell-table verdict (a 2×2 cell, AMBIGUOUS, or READOUT-FORM-INVALID) is something else | minor | Added a registered reporting rule: both named outcomes are stated side by side in the final report; neither overrides or suppresses the other, since Leg A and Leg B gate different sub-hypotheses (H_LINK-A vs. H_LINK-B) through the same instrument at different cells | §12 |
| MINOR (4) | §12 had no stated precedence between READOUT-FORM-INVALID and AMBIGUOUS when both gates could fire simultaneously (a uniform `h≥2` floor plausibly also produces Option 1/Option 2 disagreement or degenerate non-agreement at most rungs) | minor | Pinned precedence: READOUT-FORM-INVALID wins — it is the more specific mechanism-level diagnosis, of which a failed AMBIGUOUS gate is a likely symptom, not an independent finding; AMBIGUOUS remains the reported outcome only when Leg B's agreement gate fails WITHOUT a uniform `h≥2` floor also present | §12 |

**BUDGET (verified, not merely asserted):** the FATAL fix is
budget-neutral — same number of GPU passes (forward-B already exists and
is already priced by the two-forward protocol; the new hook only adds a
second `register_forward_hook` call to an already-executing call), no
probe is reintroduced, and every other Rev 5 fix is a Stage −1/Stage 0
measurement already priced inside the existing calibration cells, or a
pure interpretation/reporting rule with zero GPU cost. §10's Rev 3/Rev 4
total (**≈24.20 GPU-h committed, ≈0.80 GPU-h margin, ≈38.70 GPU-h
named-extension reserve, unchanged 25 GPU-h ceiling**) is re-printed, not
re-derived, with an explicit Rev 5 confirmation note (§10).

**Straggler sweep (performed, not merely promised):** `grep -n
'k_conv1d'` located every occurrence in the document; each was read in
context and adjudicated. Bind-phase `k_eff` extraction (forward-A, the
frozen-bias blend surgery/retrofit tooling, the Stage −1 buffer-blank-out
test, the surgery-toggle smoke) correctly stays `k_conv1d` throughout —
unchanged. `q_eff` extraction — the FATAL's own target — is corrected to
`q_conv1d` at every LIVE (non-historical) location: §4.4's Forward-B
bullet, the new FATAL callout box, §4.4's "Justification for choosing
Option 1" paragraph, and §7.9's cross-arm confound argument.
`model_rd.py::effective_key_window`'s own `self.embed→self.k_proj→
self.k_conv1d` description (§4.4's FATAL-2 discussion) is an accurate
citation of the OTHER file's own (architecturally forced) convention and
is left unchanged — it is now, explicitly, the convention Rev 5 declines
to carry over. Historical fix-map tables (§13.1-§13.4) describing what
earlier revisions ACTUALLY did or found are left intact per the standing
"record history, don't retcon it" convention. `grep`'d separately:
"premise (iv)" consequence text and the "0.9 alignment bar" reuse text
(§4.4) — both located and replaced by the unified action-rule table
(MAJOR-1, above); no residual reference to the old reused-threshold
language remains outside the historical §13.1-§13.4 record.

**What Rev 5 could NOT cleanly fix, disclosed rather than hidden:** the
premise (iii)/(iv) action-rule table and the h1 absolute backstop
pre-register HOW to read a null-band or floor failure, but neither can be
evaluated until Stage 0 actually runs on a real checkpoint — this
revision registers the mechanism and its consequences, it does not
(cannot, pre-data) resolve whether premise (iii) or (iv) will in fact
clear their own gates for this checkpoint family. The FATAL fix corrects
WHICH projection `q_eff` comes from; it does not change the
framing-adjudication caveat already registered at Rev 4 (§1/§4.4/§5.1) —
Option 1 still tests one a priori unlikely mechanism (single-layer state
self-iteration), and a genuinely-correct `q_eff` extraction can still
show a uniform floor at `h≥2` if that mechanism is simply the wrong
hypothesis for how this checkpoint family composes, which is exactly why
READOUT-FORM-INVALID remains a live, expected-not-surprising outcome
rather than something this revision claims to have resolved.
Attack-round-0's item 6 (frozen-table training-dynamics channel) and the
OpenReview R8ZbLi3oUv blocked full-read remain open — this round did not
raise or resolve either.

---

## 14. Standing constraints (inherited, checked off explicitly)

- [x] Readout is continuous cosine>0.9, never argmax/nearest-neighbor over
  a codebook (§4.4, §8.1).
- [x] Single Hamiltonian K-cycle only, `_permutation_graph` reused verbatim,
  every K > h_max asserted at config time (§4.2, §4.3, §7.5).
- [x] Position-decomposition escape closed by the fixed-state-only-channel
  architecture argument, named as a design strength (§7.1) — the residual
  local-conv-window risk is separately mitigated: fixed-slot rendering plus
  an independently-drawn random K-cycle already decorrelate render-adjacency
  from cycle-adjacency by construction (verified, not merely asserted, at
  Rev 1 per §4.2/M5), with a Stage −1 empirical measurement as the
  registered check, not assumed away.
- [x] **(Rev 1)** Mechanical blend-effect isolated from training-effect via
  the §5.2a 2×2 surgery grid — Leg A's headline claim rides on the
  training-effect contrast, not a reading confounded by the blend's own
  live eval-time action (attack-round-1 F2).
- [x] **(Rev 1)** CONFIRM requires Option 1/Option 2 directional agreement
  at the killer-prediction cell; disagreement is a named, reported outcome
  (READOUT-DIVERGENCE), never silently resolved (attack-round-1 M1).
- [x] **(Rev 1)** Every K value in the sweep is constructible from its
  entity pool (the full 107-name primary pool; Rev 3: the 106-name pool's
  memorization-control role is retired with the trained probe, §4.5) with
  margin — the arithmetic-impossibility gap attack-round-1 F1 found is
  closed by construction.
- [x] **(Rev 2)** Every registered K value (20/32/40/48/64/96) also clears
  `chunk_delta_rule`'s hard `T_bind ≥ _MIN_KERNEL_T=128` floor, per
  checkpoint's own `conv_size` — a Stage −1 assertion (item 9), not an
  assumed value; the entire pre-Rev-2 low end (K=8, K=16) crashed this floor
  and is retired (attack-round-2 FATAL).
- [x] **(Rev 2)** Episode-generation seeds cannot collide across purpose
  (Rev 3 purposes: eval/null-shuffle/calibration, §4.6) or across the other
  seed axes (arm/rung, corpus, checkpoint seed, K) — a pinned, mixed-radix
  numeric formula (§4.6), mechanically verified by a Stage −1 assertion
  (item 10), not merely argued collision-free (attack-round-2 M5).
- [x] ~~(Rev 2) The heldout-pool memorization-ceiling control...~~ —
  **RETIRED at Rev 3** with the trained probe it was built to check
  (attack-round-3 FATAL-1; adjudication in §4.5): probe-free, the control
  measures nothing, and the MEMORIZATION-CONFOUND outcome is removed with
  it. Struck through rather than deleted so the checklist's history stays
  legible.
- [x] **(Rev 2)** Rung-3's (1.31B) pass cost is gated by a real,
  blinded calibration cell before its grid is trusted, with a two-tier
  abort trigger — not an unmeasured extrapolation from a non-monotonic
  anchor sequence (attack-round-2 M6).
- [x] **(Rev 3)** The readout is dimensionally well-typed at every
  checkpoint: `S_T`, `q_eff`, `v_eff_target` (Rev 4: renamed from
  `k_eff_target`, §13.4 — dimensionality claim unaffected), and `pred(a,h)`
  all live in `R^{d_state}`, produced by the checkpoint's own machinery —
  no trained probe, no projection into `d_model`, no `W_embed` in any
  scoring formula (attack-round-3 FATAL-1).
- [x] **(Rev 4)** The comparison target is family-consistent at every
  tested `h`, proved rather than argued: `Im(S_T) ⊆ span{v_eff_j}` by
  induction on the delta-rule update, for any `β`, so `pred(a,h)=S_Tʰ·q_eff`
  is a `v_eff`-family object at every `h≥1` — `v_eff_target` (not
  `k_eff_target`) is therefore the only comparison object ever in the same
  family as `pred(a,h)`, gathered at the audit-fixed `prev_slot`, never
  `tgt_slot` (attack-round-4 FATAL, §13.4).
- [x] **(Rev 4)** Leg B's own scale-trend claim (H_LINK-B) carries the
  identical mandatory Option-1/Option-2 agreement gate Leg A's killer
  prediction already had — extended to Leg B's primary near-cliff cells
  (`K=32` at `d=64`, `K=64` at `d=128`), with a named demotion outcome
  (AMBIGUOUS) when fewer than 3 of the 4 ladder rungs agree
  (attack-round-4 FATAL-ADJACENT, §6.2).
- [x] **(Rev 4)** A new outcome, READOUT-FORM-INVALID, is pre-registered
  and distinguished from REFUTE/no-capability: fires when `h=1` clears the
  chance floor but `h≥2` sits at floor uniformly across every arm and
  rung — the observationally-indistinguishable case between "no
  capability" and "wrong readout mechanism" (attack-round-4 MAJOR, §12).
- [x] **(Rev 5)** `q_eff` is extracted through the checkpoint's own real
  `q_proj`/`q_conv1d` path (a new hook, §4.4), never through `k_conv1d` —
  the from-scratch harness's forced (no-`q_proj`) convention is no longer
  carried over into a checkpoint family that has a real Q-path
  (`chunk_delta_rule(q=q_bf,...)`, `lm_pretrain_rd.py` L983); every
  `k_conv1d` mention describing `q_eff` extraction is corrected, every
  mention describing bind-phase `k_eff`/`v_eff` extraction is unchanged
  (attack-round-5 FATAL, §13.5).
- [x] **(Rev 5)** Premises (iii) (bind↔query alignment) and (iv)
  (cross-role k↔v identity) share one mechanical, null-relative
  action-rule table, measured at Stage 0 against each premise's own
  cross-entity-shuffled null — no reused, unjustified 0.9 constant;
  premise (iii) is now a genuine measured quantity (two
  independently-trained projections), not the near-tautology it was when
  `q_eff` came from `k_conv1d` (attack-round-5 MAJOR-1, §4.4).
- [x] **(Rev 5)** The h=1 sanity floor combines the null-band-relative
  rule with a registered absolute backstop (`recovered_frac@0.9 ≥ 0.10`)
  — a near-zero null band can no longer pass the probe as valid by
  default (attack-round-5 MAJOR-2, §8.4).
- [x] **(Rev 5)** Stage 0.5's rung-3 calibration cell has a registered,
  bounded OOM fallback (retry once at half batch size, then halt with
  disclosure) — never an open-ended retry/crash-loop (attack-round-5
  MINOR, §9).
- [x] **(Rev 5)** §12's outcome reporting is fully specified when multiple
  named outcomes could apply at once: a Leg-A READOUT-DIVERGENCE and a
  Leg-B cell-table verdict are reported side by side, neither overriding;
  READOUT-FORM-INVALID takes precedence over AMBIGUOUS when both gates
  would otherwise fire (attack-round-5 MINOR, §12).
- [x] **(Rev 3)** Per-layer extraction is legal at every checkpoint depth:
  the two-forward protocol (§4.4) never feeds the mixer a sub-`_MIN_KERNEL_T`
  sequence and never fabricates a layer-`i>0` input outside a real
  multi-layer forward; the forward-A/forward-B bind-prefix identity is a
  Stage −1 assertion (item 11), not an argument (attack-round-3 FATAL-2).
- [x] **(Rev 3)** Rung-3's configuration (2 corpora × 1 seed) is pinned
  from the actual launch manifest, not left to harvest-time confirmation;
  batch size (16) is pinned from the anchor tools' own measured defaults
  (attack-round-3 MAJOR/minor).
- [x] No compressing matrices to vectors — `S_T` and `pred(a,h)` stay
  `d_state`-dimensional throughout (Rev 3: the readout compares them
  directly in `d_state` space; nothing is flattened or projected out).
- [x] Same dataset/instrument across every arm in one comparison — one
  probe-generation script, one readout, shared across Leg A's 3 arms and
  Leg B's 4 rungs; only the checkpoint being loaded differs.
- [x] Calibration before sweep (§9 Stage 0), self-tests before calibration
  (§9 Stage −1), matching the mandatory house sequencing.
- [x] Pinned CI convention (`t(2,.975)=4.303`, n=3) reused unchanged; every
  n<3 cell (rung-3, λ-mini-sweep) explicitly demoted to descriptive-only,
  never silently CI'd.
- [x] Tokenization held fixed (GPT-2 BPE throughout, no byte-level
  variant bundled in) per the standing hold-the-second-axis-fixed rule.
- [x] Every design decision in this document is registered before any
  checkpoint's probe results exist — no threshold, K choice, or marker
  choice in §4–§10 is contingent on having seen a real number from any
  arm or rung.

---

## 15. PHASE 1 RESULTS — Leg A unblinded (rung-3 rows pending), 2026-07-07

**HEADLINE: the readout is PROBE-INVALID for the entire harvested grid.
`recovered_frac@0.9` (Option 1) is exactly `0.0` at every single one of
the 78 committed cells (60 Leg A + 18 Leg B rungs 0-2), at every tested
`h∈{1,2,3,4}` — 0/312 (cell,h) readings nonzero, not one. Stage 0's own
pre-registered h=1 sanity floor (§8.4/§9 purpose c/e) FAILS both its
null-relative and absolute (0.10) conditions (`gate_result_h1_probe_valid
= False`), and premises (iii)/(iv) fail their own null-relative gate at
ALL 78 cells (0/78 pass either premise). Per this design's own §8.4 rule
("failure routes to probe-invalid, not to REFUTE") and §9 Stage-0
registration ("if [the h1 floor] is not achievable... the grid does not
launch"), this is NOT a licensed REFUTE of H_LINK-A or H_LINK-B — it is
the design's own safety net correctly identifying that the `S_Tʰ·q_eff`
readout, scored via absolute cosine ≥0.9, never fires for any
never-task-trained checkpoint in this grid's scope. Separately and
importantly: the on-box launch script did NOT act on this gate — see
the discrepancy below. No CONFIRM / REFUTE / READOUT-DIVERGENCE /
READOUT-FORM-INVALID / AMBIGUOUS / Cell-1-4 reading is licensed by this
harvest. This section reports the mechanical routing, the raw numbers
behind it, and the process gap that let a probe-invalid instrument run
to completion, in full.

### 15.1 Gates (checked first, before any headline number)

| Gate | Registered rule | Measured | Result |
|---|---|---|---|
| Marker disagreement (§8.5 item 4) | abs diff ≤ 0.15 at every h, both query markers (`§`,`¶`) | 0.0 at h∈{1,2,3,4} (both markers scored identically — trivially, since both are 0.0) | PASS (vacuous — see §15.5) |
| Causality assertion (§9 item 11 / Stage 0 purpose a) | forward-A/forward-B agree ≤1e-6 fp32 CPU over shared bind prefix | `max_abs_diff = 0.0` | PASS |
| h1 null-relative pass (§8.4) | real h=1 `recovered_frac` > null_hi + null_width | real=0.0, null=[0.0,0.0] (width 0) → 0.0 > 0.0 is False | **FAIL** |
| h1 absolute backstop (§8.4 MAJOR-2) | real h=1 `recovered_frac` ≥ 0.10 | real=0.0 | **FAIL** |
| **`gate_result_h1_probe_valid`** (both above, AND) | — | — | **FAIL — probe declared invalid at Stage 0** |
| Premise (iii) bind↔query alignment (§4.4 MAJOR-1) | same-entity median > cross-entity null p95 | Stage-0 median=0.5406 vs null_p95=0.6097 | **FAIL** |
| Premise (iv) cross-role k↔v identity (§4.4 MAJOR-1) | same-entity median > cross-entity null p95 | Stage-0 median=0.0139 vs null_p95=0.1424 | **FAIL** |
| Stage-0 cost abort (§10 abort rule 1, chain script) | wall_s ≤ 3× the 0.0696 GPU-h baseline | `wall_s=1.11s`, ratio≈0.004× | PASS |

**Premise (iii)/(iv) fail everywhere, not just at Stage 0** — re-derived
per §4.4 Rev 6's own "at every rung's own primary cell" registration,
using the actual per-cell readings the raw JSONs already carry (premises
are computed once per cell, shared across all 4 h's, per
`measure_cell_all_h`):

| Family (own primary cell, seed 0, native) | premise iii median / null p95 / pass | premise iv median / null p95 / pass |
|---|---|---|
| Leg A control arm (Stage 0, licenses Leg-A arms + rung 0) | 0.5406 / 0.6097 / **False** | 0.0139 / 0.1424 / **False** |
| rung0 (14M, d64/L2) openr1 | 0.3448 / 0.4599 / **False** | 0.0793 / 0.1683 / **False** |
| rung0 (14M, d64/L2) wikitext | -0.2513 / -0.1752 / **False** | 0.0283 / 0.1184 / **False** |
| rung1 (98M, d64/L12) openr1 | -0.1408 / -0.0422 / **False** | 0.0472 / 0.2086 / **False** |
| rung1 (98M, d64/L12) wikitext | 0.6092 / 0.6898 / **False** | -0.0525 / 0.0270 / **False** |
| rung2 (392M, d128/L16) openr1 | 0.0231 / 0.1174 / **False** | 0.0431 / 0.1644 / **False** |
| rung2 (392M, d128/L16) wikitext | 0.7310 / 0.7980 / **False** | -0.0183 / 0.0835 / **False** |

Across the FULL 78-cell grid (every arm/corpus/seed/K/surgery for Leg A,
every rung/corpus/seed for Leg B), 0/78 cells pass premise (iii) and 0/78
pass premise (iv) — `premise_iii_median` ranges `[-0.3272, 0.7633]`,
`premise_iv_median` ranges `[-0.0727, 0.3435]` across the grid, i.e. this
is not a borderline near-miss at one point but a categorical failure of
both action-rule gates everywhere they were measured, at every scale from
14M to 392M params. By §4.4's own action-rule table, this means
`h1_confirmatory=False` and `h_ge2_confirmatory=False` at every single
cell — no hop depth retains confirmatory status anywhere in this harvest.

### 15.2 DISCREPANCY — the launch script did not enforce Stage 0's own registered abort gate

**`reasoning_link_chain.sh` contains exactly one Stage-0-level abort check
— the wall-clock cost ratio (§10 abort rule 1, lines ~159-176 of that
script) — and never references `marker_disagreement_flag` or
`gate_result_h1_probe_valid` anywhere (grepped directly, zero hits for
either string).** Per this design's own registration (§9 Stage 0 purpose
c: *"confirm the h1 floor is achievable at all before committing to the
full grid — if it is not, this is itself a decisive, informative result
... and the grid does not launch"*; §8.5 item 4: a marker disagreement
beyond tolerance "blocks" the grid), the h1-floor failure measured at
Stage 0 (§15.1 above) should have halted the chain before Stage 1 ever
ran. It did not — all 78 committed cells ran to completion on a probe the
design's own instrument had already flagged invalid. The marker-tolerance
gate happened to pass in this run (trivially — both markers scored
identically at 0.0, so there was nothing to disagree about) so it would
not have blocked the launch either way, but that is incidental, not a
mitigating fact: had marker disagreement been nonzero, the chain still
had no code path that would have stopped it. This is a genuine build gap
against the design's own registration, not a data-analysis judgment call
— flagged prominently per this harvest's own charter. **Realized cost of
running past this gate: ≈0.29 GPU-h (§15.7)** — small in absolute terms
(rung-3, the expensive row, was never in scope this run), but the
principle — a computed, registered gate with no enforcing code path — is
the same failure mode CLAUDE.md's own "run the negative test to
completion" and "gates without teeth" lessons already warn about, and is
recorded here as a fresh instance, not a one-off.

### 15.3 LEG A — full unblinded table

`recovered_frac@0.9` (Option 1) is `0.0` for all 3 arms (off/per_token/
global) × 2 corpora × 3 seeds × 2 K (20,32) × applicable surgery
(native, plus blend-OFF for the two bias arms) × 4 h — 60/60 cells,
240/240 (cell,h) Option-1 readings. Reproduced directly from the raw
JSONs (`per_h[h]["recovered_frac"]`), not summarized from a subset.

Because every constituent value is identically `0.0`, both §5.2a
contrasts are degenerate:

- **training-effect** (`rec@0.9`(arm ckpt, blend-OFF) − `rec@0.9`(off
  ckpt)) = `0.0 − 0.0 = 0.0` at every (arm, corpus, K, h) cell, `n=3`
  seeds, zero variance → pinned CI `[0.0000, 0.0000]` everywhere.
- **mechanical-effect** (`rec@0.9`(arm ckpt, blend-ON) − `rec@0.9`(arm
  ckpt, blend-OFF)) = `0.0 − 0.0 = 0.0` everywhere, same CI degeneracy.

No arm, corpus, K, or h shows any Option-1 separation, by construction of
the data — there is nothing to distinguish global/per_token/off on this
instrument.

### 15.4 KILLER PREDICTION VERDICT — K=32 primary (and K=20), mechanical routing

Applying `reasoning_link_probe.killer_prediction_verdict` (the design's
own pure function, §12) exactly as registered, to every (arm, corpus, h):

| arm | corpus | h | Δ(K=32) mean [95% CI] | Δ(K=20) mean [95% CI] | Opt1/Opt2 agreement | Mechanical verdict |
|---|---|---|---|---|---|---|
| global | openr1-mix-ext | 1–4 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | agree | REFUTE |
| global | wikitext-mix-ext | 1–4 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | agree | REFUTE |
| per_token | openr1-mix-ext | 1–4 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | agree | REFUTE |
| per_token | wikitext-mix-ext | 1–4 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | agree | REFUTE |

(All 16 rows — every arm × corpus × h — return the identical
`[0.0000, 0.0000]` CI and REFUTE verdict; collapsed above for space, full
per-h breakdown reproduced by the harvest script from the raw JSONs.)

**This REFUTE reading is a MECHANICAL BY-PRODUCT of `killer_prediction_
verdict`'s own logic (`ci_low > 0` fails when the CI is degenerately
`[0,0]`), not a licensed scientific verdict.** Per §15.1's gates, the
readout failed its own h1 sanity floor and both premise gates BEFORE this
point in the routing — §8.4's explicit rule ("failure routes to
probe-invalid, not to REFUTE") pre-empts this REFUTE reading. **The
correct, gate-respecting outcome is PROBE-INVALID, reported honestly as
such; the mechanical REFUTE computation is shown above only so the
routing is fully transparent, never presented as if it were the
headline finding.** The killer-prediction's own `K=32 vs K=20` structure
(is arm separation concentrated near the capacity cliff) cannot be
evaluated at all when both K's read identically zero.

### 15.5 The h=1 recall story, on its own

h=1 (in-context one-hop associative recall, needing only premise iii) is
`0.0` for every arm/corpus/K, `n=3` seeds each — the "easiest" case this
design's own literature review (Zoology, Based) predicted should show
*some* nontrivial recovery in a fixed-state LM if the readout has any
referent at all (§8.4's own justification for the 0.10 absolute floor).
It does not clear even that low bar, anywhere. Option 2 (natural
next-token logit margin, secondary/non-headline) is uniformly *negative*
at h=1 across every arm/corpus/K (range ≈ -2.6 to -4.3, meaning the
true-answer token's logit trails the best distractor's by 2.6-4.3 nats
on average) — the natural forward pass does not favor the correct
in-context entity over distractors either, at h=1, for any checkpoint in
this grid. Both instruments agree: there is no clean one-hop signal to
build a multi-hop story on top of.

### 15.6 Option 2 (secondary, non-headline) corroboration

Option 2 shows real per-seed variation (unlike Option 1's flat zero) but
no signal that would change the routing. One cell reaches BH-uncorrected
significance in the non-primary grid — per_token arm, wikitext-mix-ext
corpus, **K=20**, all four h: training-effect margin-delta CI excludes
zero on the negative side at every h (e.g. h=1: mean=-0.79, CI=[-0.96,
-0.62]) — i.e. per_token training *hurts* the natural next-token margin
at this corpus/K, consistent in sign across h=1-4. This is a single
non-primary-grid cell (K=20 is the low-load contrast point, not the
K=32 killer-prediction cell) and is reported here as an observation, not
a claim — §8.6's BH-FDR correction over the full non-primary grid was not
run for this harvest (moot: Option 1, the primary instrument, is
uniformly floor, so no CONFIRM/REFUTE claim rests on this number). No
other arm/corpus/K shows a CI excluding zero on Option 2.

**Registered covariate `cross_a_cosine_convergence` (§8.3 M1) was never
computed** — the function exists in `reasoning_link_probe.py` but is not
called anywhere in `measure_cell_all_h`/`run_cell`/`run_stage0`, so the
power-iteration-degeneracy diagnostic this design registered as
mandatory at every headline cell is absent from every JSON in this
harvest. Flagged as a second, lower-severity build gap alongside §15.2's
launch-gate gap.

**`state_condition_number_mean` (computed, present at every cell) is
uniformly enormous** — median ≈6.3e4 to 2.2e5 depending on arm/rung,
range up to 3.85e6 (leg_a global arm) — `S_T` is extremely
ill-conditioned everywhere in this grid, dominated by one or a few
directions. This is offered as corroboration only (§8.3's own discipline
— "not a discriminating result on its own"), but it is at least
*consistent with* the total-floor recovery result: a near-singular `S_T`
matrix-powered `h` times plausibly collapses toward one dominant
direction rather than any entity-specific `v_eff_target`, which would
mechanically produce exactly the kind of flat-zero recovery measured
here regardless of hop depth.

### 15.7 Realized GPU-h

The grid ran across 3 resume-safe launches (runs 6-8; runs 1-5 were
build-time crashes fixed by launch fixes 1-5/6/7, git log `6475770` down
to `8bc90ba`) after the design's own build+audit (`bb1869c`). Timestamps
below are box filesystem birth/modify times (`stat`, box is UTC,
confirmed via `date -u`/`timedatectl`), single GPU throughout
(`REASONING_LINK_GPU`, default device 3):

| Run | Birth (UTC) | Last write (UTC) | Duration | What it covered |
|---|---|---|---|---|
| run6 | 11:36:47.99 | 11:47:01.49 | 613.5s | Stage -1/0/0.5 + all 60 Leg-A cells; crashed on a Leg-B rung-0 checkpoint-path guess (fixed by launch fix 3) |
| run7 | 12:19:46.47 | 12:23:30.86 | 224.4s | Stage -1/0/0.5 rerun (idempotent) + Leg-B rung0+rung1 (Leg-A skipped, resume-safe); crashed on rung-2 FFN OOM (fixed by launch fix 7) |
| run8 | 12:25:54.29 | 12:29:11.60 | 197.3s | Stage -1/0/0.5 rerun + Leg-B rung2 (rungs 0/1 skipped, resume-safe); completed, wrote `REASONING_LINK_PHASE1_PARTIAL` |
| **Sum (runs 6-8, "realized")** | | | **1035.2s ≈ 0.2876 GPU-h** | |

Disclosed method: each run's own (birth → last-write) span is summed
directly — this excludes the idle/debugging gaps BETWEEN runs (e.g. the
~33 min between run6's crash at 11:47 and run7's birth at 12:19 was
agent debugging time, not GPU time) by construction, since only time
inside a running process is counted. Runs 1-5 (pre-launch build-time
crashes, Stage -1/CUDA-kernel failures before any real grid cell ran)
add a further ≈309s (≈0.086 GPU-h) of mostly-wasted false-start cost,
reported for completeness but excluded from the "realized" figure per
the task's own framing (runs 6-8 are the grid that actually executed).
**Grand total across all 8 runs ≈1344s ≈0.373 GPU-h — Stage 1 (this
harvest) realized ≈0.29 GPU-h against the ≈24.20 GPU-h Phase-1 ceiling
(§10), ≈1.2% utilized.** The gap is real, not a rounding artifact: every
checkpoint evaluated this run is ≤392M params (rung-3 at 1.31B, the
expensive row, is deferred), and per-cell cost is small at this scale
regardless of what the readout returns — probe-invalidity does not
change forward-pass cost.

### 15.8 LEG B rungs 0-2 (PARTIAL — rung-3 pending trackc `ALL_DONE`)

Same universal-zero pattern: `recovered_frac@0.9 = 0.0` at all 18 cells
(rung0 14M d64/L2, rung1 98M d64/L12, rung2 392M d128/L16; 2 corpora × 3
seeds each) × 4 h = 72/72 Option-1 readings, all zero. No monotone trend,
no Spearman correlation with `span_frac`, and no CONFIRM/REFUTE reading
is possible from this instrument — the same probe-invalid routing
applies to Leg B as to Leg A (§15.1/§15.4).

Option 2 (secondary) shows a monotone trend in the WRONG direction for a
"scale helps" story if taken naively: mean margin (vs. a zero baseline,
`n=3` per rung/corpus, `h=3,4`) goes from ≈-3.2 to -3.4 (rung0, 14M) to
≈-3.7 to -3.9 (rung1, 98M) to ≈-4.3 to -5.5 (rung2, 392M) — i.e. the
natural next-token margin AGAINST the correct answer gets WORSE, not
better, with scale, across both corpora. This is reported descriptively
only (Option 2 is never load-bearing alone, §5.2/§8.2) and is exactly
the kind of reading the §6.2 Rev 4 agreement gate exists to catch:
Option 1 is flat zero everywhere (no direction to agree or disagree
with), so `option_agreement` is vacuously "agree" at every rung
(`opt1_excludes_positive` is always False, so the disagreement branch
never fires) — **this vacuous agreement must not be read as validating
a scale trend**. `leg_b_scale_gate` cannot be meaningfully applied:
3-of-4 rungs "agreeing" here means 3 rungs where nothing is happening on
either instrument, not 3 rungs of genuine directional concordance.
**No H_LINK-B trend verdict (CONFIRM/REFUTE/AMBIGUOUS) is reported —
rungs 0-2 are PARTIAL, rung-3 is not yet run, and the instrument itself
is probe-invalid regardless.**

### 15.9 What this harvest does NOT show

1. **This is not evidence that geometry stabilization or scale don't
   matter for composition.** The instrument that was supposed to test
   that (Option 1, single-layer state self-iteration, §4.4's own framing
   note) never produced a usable signal on ANY checkpoint, including the
   architecturally-simplest 14M control. A uniformly-floor readout with a
   failed validity gate cannot distinguish "no capability" from "wrong
   mechanism" from "wrong threshold for this checkpoint family" — exactly
   the ambiguity §12's own READOUT-FORM-INVALID outcome was built to
   name, except this harvest sits one level more severe than that (h=1
   never cleared its OWN floor, so READOUT-FORM-INVALID's trigger — h=1
   clears, h≥2 doesn't — is not even satisfied; see §15.1).
2. **Rung-3 (1.31B) is not in this harvest at all** — deferred pending
   trackc `ALL_DONE`, per the design's own registration (§6.1). Nothing
   above should be read as a 4-rung verdict.
3. **Stage -1's self-tests passing (all 14 items + the extra gate check,
   `logs/90_reasoning_link_stage_minus1.log`, 48.6s, ALL PASSED) rules out
   an arithmetic bug in the readout's hand-built-case behavior, but does
   not rule out the readout being the wrong construct for real,
   never-task-trained checkpoints** — the self-tests exercise synthetic
   matrices with known answers, not whether real pretrained
   representations ever produce the kind of alignment Option 1 requires.
   `cos_mean` across the full grid is tightly centered near zero
   (`[-0.33, 0.25]`, `cos_std` `[0.03, 0.20]`) — reaching `|cos|>0.9`
   from this distribution is a >10σ event, i.e. the 0.9 absolute
   threshold (inherited from the task-TRAINED `KEY_ANCHORING`/exactness
   lineage, where h1=1.0 is achievable) appears simply unreachable for
   this never-task-trained checkpoint family under this specific linear
   readout — consistent with, though not proof of, a threshold/mechanism
   mismatch rather than a genuine absence of any composable structure.
4. **Options 2's own signal (§15.6/§15.8), where nonzero, is explicitly
   non-headline** (§5.2/§8.2's own standing rule) and is reported for
   completeness, not as a substitute finding.

### 15.10 Next steps (not self-launched by this harvest)

This harvest does not license a design revision, a re-threshold, or a
Phase-2 build on its own authority — those are PI/design-owner decisions.
What is recorded here, mechanically and completely, is: the gates fired
correctly (the design's own instrumentation caught its own invalidity),
the launch script did not enforce that finding before spending GPU-h
(§15.2), and the raw data, once unblinded, cannot license any of the
six named §12 outcomes. Candidate next steps for whoever picks this up
(not decided here): (a) re-examine whether an absolute-cosine≥0.9
threshold is appropriate for never-task-trained checkpoints at all
(vs. e.g. a rank/percentile-based or relative-margin readout); (b) wire
the missing gates into `reasoning_link_chain.sh` before any re-run;
(c) wire `cross_a_cosine_convergence` into `measure_cell_all_h` if a
re-run is designed; (d) decide whether rung-3 should still run once
`trackc` completes, given rungs 0-2 already show the identical
probe-invalid pattern this instrument would need to have differed from
to matter.

Archive: `experiment-runs/2026-07-07_reasoning_link_phase1/` (82 raw
result JSONs + 89 log files + the 3 exact scripts —
`reasoning_link_chain.sh`, `reasoning_link_probe.py`,
`reasoning_link_stage_minus1.py` — byte-identical to the box's own
copies, verified directly, ~1.3MB, all files ≤25MB) + SSD mirror at the
same relative path under `/Volumes/1TB_SSD/learned-representations/`.
`STATE.md` and `EXPERIMENT_LOG.md` updated.


---

## 16. THE NEXT INSTRUMENT — three paths after PROBE-INVALID, compared and sequenced (design pass, 2026-07-07)

### 16.0 What §15 licenses here, and what it doesn't

§15's verdict is PROBE-INVALID, not REFUTE: the h1 sanity floor fails both
its null-relative and absolute-0.10 conditions, and premises (iii)
(bind↔query alignment, Stage-0 median 0.5406 vs. null p95 0.6097) and (iv)
(cross-role k↔v identity, 0.0139 vs. 0.1424) fail their own action-rule
gates at every one of 78 cells, 14M-392M params, both corpora. **The
keystone question — does frozen-bias key-geometry stabilization causally
improve in-context multi-hop composition — was never actually asked of the
data; the instrument that was supposed to ask it never produced a
referent-bearing signal on any checkpoint.** This section does not attempt
to re-run the same instrument hoping for a different answer. It compares
three structurally different next moves, each of which changes something
§15 held fixed, and stress-tests a sequencing recommendation rather than
asserting one.

Two structural facts from §15 bound every option below:
1. **The failure is categorical, not marginal.** `recovered_frac@0.9` is
   exactly `0.0` at 312/312 (cell, h) readings — not "mostly zero with a
   few near-misses." Whatever is wrong is not a threshold-tuning problem.
2. **Premise (iv) failing much harder than premise (iii)** (iv's real
   values sit at 1-8% of typical premise-iii magnitudes across the grid,
   §15.1) is a specific, informative asymmetry: `k_eff` and `v_eff` (two
   projections of the SAME token, captured milliseconds apart in the same
   forward pass) are nearly uncorrelated in this checkpoint family, while
   `q_eff` and `k_eff` (different tokens, same family) are only mildly
   anisotropic. This is closer to "the k/v split itself carries little
   entity-identity structure in a never-task-trained LM" than to "the
   surface form confused the model" — a point that bears directly on how
   much any of the three paths below should expect to fix.

### 16.1 PATH (i) — PHASE-1B: natural-language surface form

**What it tests.** Whether §15's failure is a **surface-form artifact** —
the marker template's reserved-adjacent buffer/query tokens (§4.1: period
repeated `conv_size−1` times as buffer, a rare symbol or fixed phrase as
the query marker) putting the episode text far enough outside the
pretraining distribution that even a genuinely capable checkpoint could
not engage its own induction/in-context machinery — versus a **deeper
representational fact** about this checkpoint family (premise iv's
near-zero k↔v correlation) that no surface-form change can fix.

**What stays fixed, by design (per the task brief, and because changing
more than one axis at once makes any result uninterpretable per this
project's own standing "hold tokenization fixed" rule, `CLAUDE.md`):** the
single-Hamiltonian-K-cycle generator (`_permutation_graph`, §4.2), the
107-entity pool (§4.1, reused verbatim — the pool is already a fixed,
somewhat artificial set of common first names; Phase-1b changes the
SENTENCE FRAME around those names, not the entity vocabulary itself), the
three Leg-A arms and Leg-B rung ladder, the h∈{1,2,3,4} depths and K sweep,
premises (iii)/(iv) and the h1-floor gate exactly as registered (§8.4),
the `v_eff`-scored Option 1 readout and Option 2 secondary (§4.4), and the
two-forward causality protocol (§4.4/§9 item 11).

#### 16.1.1 Template design, checked against ACTUAL wikitext statistics, not assumed

The project's own local WikiText-103 corpus
(`/Volumes/1TB_SSD/learned-representations/data/text.bin`, 100MB raw
UTF-8 — the same file `CLAUDE.md`'s Data section documents) was grepped
directly this pass (read-only, no code committed, no GPU) to check what
this design assumed rather than measured. Three findings, all load-bearing
for the template choice:

| Check | Pattern | Hits / 100MB | Reading |
|---|---|---|---|
| Query marker OOD-ness | `‽` | **0** | Confirms §4.1's own prediction ("close to its random-init value") — this token is never seen in this register at all |
| Query marker OOD-ness | `§` | 72 | Rare, and essentially all legal/citation-register, not relational — not a genuine in-distribution relational cue either |
| Bind-clause shape, existing verb pool | `[Name] [gift-verb] [Name] .` (37-verb pool from `grammar_rd.py::_CANDIDATE_RELS_A/B` — "gave", "handed", "passed", "carried", "sent", ...) | 23 | Modest but real — e.g. `"Bonds passed Ruth ,"`, `"Dresden carried Huerta ,"`, `"Coulthard passed Herbert ,"`. The EXISTING bind-clause shape (ignoring buffer padding and the query marker) is not as far out-of-distribution as the marker template's total package |
| Query-clause completion shape | `[Name] [gift-verb].` (verb, sentence-final, nothing after) | 0 (5-verb sample) | Real wikitext essentially never truncates a sentence immediately after a bare transitive verb with no object — the readout's own truncation requirement (must stop before the answer token) has **no clean wikitext precedent under ANY relation-verb choice**, not just the marker's |
| Alternative relation family — succession | `[Name] succeeded [Name]`, `was succeeded by [Name]`, `[Name] replaced [Name]` | 40 + 64 + 119 = **223** | An order of magnitude more common than the gift-verb bind shape, AND directionally composable in exactly the way a K-cycle is: succession chains (e.g. lines of monarchs, officeholders) are real wikitext objects with genuine 2+-hop structure ("who succeeded the person who succeeded X" is a normal historical question) |

**Two candidate templates, both cheap enough to run in the same Stage-0
gate (§16.1.2 below):**

- **Candidate A — minimal diff.** Reuse the existing, already
  single-token-verified 107-name pool and 37-verb gift pool verbatim for
  bind clauses (`"Adam gave Bob . Carl handed Dave . ..."`); **drop the
  reserved/rare query marker entirely** and read `q_eff` off the query
  clause's own final relation-verb token via completion truncation
  (`"...Adam gave"`, no marker token at all) — this is a genuine
  simplification, not merely a substitution: it eliminates the single most
  quantifiably OOD element (`‽`: 0 hits) by construction, at **zero new
  Stage −1 verification burden** (every name/verb token is already
  verified single-token by `grammar_rd.py::build_entity_pools`).
- **Candidate B — succession family.** Replace the gift-verb pool with
  `succeeded`/`replaced` (bind: `"Adam succeeded Bob . Carl succeeded
  Dave . ..."`; query: completion-style `"...Adam succeeded"` or, as a
  named variant, an interrogative `"Who succeeded Adam ?"`). Backed by an
  order-of-magnitude stronger wikitext base rate AND a conceptually
  better match to multi-hop semantics (succession chains genuinely
  compose; gift-giving chains do not have as natural a "2-hop" reading —
  "who received what Adam originally gave, after it changed hands twice"
  is a convoluted story next to "who succeeded the person who succeeded
  X"). Requires a **new** Stage −1 single-token/non-collision check for
  `succeeded`/`replaced` under GPT-2 BPE (not yet verified anywhere in
  this codebase) — a small, disclosed build item, not a blocker.

Both candidates keep the period-repeat buffer convention unchanged (§4.1's
own argument — an ordinary, maximally generic token — is not challenged by
this pass; periods are trivially the single most common punctuation token
in the measured corpus, so this is not the risk surface Phase-1b targets).

**Worked example episode (Candidate A, K=4 for illustration; real cells
use K∈{20,32}), exactly mirroring §4.4's own 3-entity worked-example
convention:**

```
Adam gave Bob . Carl handed Dave . Ellen passed Adam . Bob offered Carl .
```

(bind clauses at fixed slot positions, cycle `succ = [1, 3, 0, 2]` meaning
Adam→Bob, Carl→Dave, Ellen→Adam, Bob→Carl — i.e. Adam gave [something]
to Bob, and separately Bob offered [something] to Carl, so hop-2 from Adam
is Carl); query at h=1: `Adam gave` (`q_eff` read at the final "gave"
token, scored against Bob's `v_eff`); at h=2: same query text, scored two
matrix-powers deep against Carl's `v_eff`. Buffer padding (`conv_size−1`
period repeats between clauses) omitted from the illustration above for
readability, present in the real render exactly as §4.1 specifies.

#### 16.1.2 Staged cost — Stage-0-gate-only, then (conditionally) the full grid

**Stage-0-gate-only first (~0.01 GPU-h).** Re-run exactly Stage 0's own
single-cell calibration procedure (§9: one checkpoint, one corpus, K=32,
h∈{1,2,3,4}, both null-band construction and premise (iii)/(iv)
measurement) — but with Candidates A and B substituted for the marker
template, and, **critically, on the WIKITEXT-mix-ext control-arm checkpoint,
not a mechanical re-use of Phase 1's own Stage-0 cell.** This is a genuine
correction to the naive form of this plan, not a cosmetic detail: Phase
1's Stage-0 calibration cell (§9) was
`frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0` — the **openr1**
corpus's control arm. OpenR1-Math is step-by-step mathematical derivation
text; it contains essentially no "Name verb Name" narrative sentences of
any kind, gift-verb or succession-family alike. Naively re-using that same
calibration cell for Phase-1b would test the natural-language premise on
exactly the checkpoint family where it has the WEAKEST possible referent
— the resulting gate reading would say more about openr1's own register
than about whether natural surface form helps. **Registered fix: Phase-1b's
Stage-0 gate cell must be drawn from the wikitext-mix-ext control arm**
(the corpus where "sentence resembling the pretraining distribution" has
an actual referent), with the openr1 cell run alongside as a **named,
expected-null contrast** — if openr1 ALSO clears the gate despite having
no natural-template precedent, that is a flag for a confound (something
generically easier about Candidates A/B, unrelated to distribution match),
not a second confirmation.

Cost basis: §15.7's own realized rate (0.2876 GPU-h across the full
78-cell grid, ≈0.0037 GPU-h/cell average) and §15.1's own single-cell
abort-rule measurement (`wall_s=1.11s ≈ 0.0003 GPU-h` for the ORIGINAL
Stage-0 cell) both put a single calibration cell an order of magnitude
under 0.01 GPU-h; running BOTH candidates × 2 corpora (4 gate cells total)
still costs ≈0.001-0.002 GPU-h at the measured rate. **~0.01 GPU-h is
registered here as a conservative estimate** (includes Stage −1's own
CPU-only re-verification — new single-token checks for Candidate B's
verbs, zero GPU — plus build/debug overhead; Phase 1 itself needed 3
resume-safe launches after 5 build-time crashes, §15.7, so a nonzero
debug tax should be priced in even for a "trivial" change).

**Full Phase-1b grid, conditional (~0.4 GPU-h at measured rates).** If the
wikitext-cell gate clears for at least one candidate: the full 78-cell grid
(60 Leg A + 18 Leg B rungs 0-2, identical shape to Phase 1, §15.3/§15.8)
re-run with the winning candidate's template. At the realized 0.0037
GPU-h/cell rate this is ≈0.29 GPU-h — essentially identical to Phase 1's
own realized cost, since template text does not change forward-pass FLOPs;
~0.4 GPU-h is the registered figure with the same build/debug margin
applied above.

#### 16.1.3 Failure modes and what a null means

1. **Both candidates fail the wikitext-cell gate identically to the marker
   template.** This is the single most informative possible outcome for
   sequencing purposes: it would mean the failure is NOT attributable to
   OOD surface form (two structurally different templates, one with real
   wikitext precedent, both fail the same way) — reinforcing premise
   (iv)'s own diagnosis (§16.0 point 2) that the k/v split itself lacks
   entity structure in this checkpoint family, a fact no zero-shot prompt
   redesign can route around. **This directly and mechanically promotes
   Path (ii) to the only remaining instrument that can engage the
   keystone** (§16.6's decision tree).
2. **Candidate A fails, Candidate B passes (or vice versa).** Would
   isolate WHICH surface-form element mattered — the marker specifically
   (A tests this cleanly, since A's only change from the original is
   marker removal) vs. the relation-verb family's compositional semantics
   (B tests this). Informative regardless of which wins.
2b. **The openr1 "expected-null" cell ALSO clears** despite having no
   natural-template referent. Named explicitly as a confound flag
   (§16.1.2) — routes to an audit of what actually changed (e.g., a
   readout-mechanics artifact common to both new templates, unrelated to
   distributional match) before any wikitext-cell pass is trusted as
   evidence for the "format was the blocker" story.
3. **h=1 clears the gate but premises (iii)/(iv) still fail their own
   action-rule gates.** Per §4.4's own action-rule table (already
   registered, not new), this demotes `h≥2` to exploratory-only even
   though the probe is no longer flatly "invalid" — h=1 becomes the
   primary, reportable result; h≥2 stays descriptive. This is a real,
   licensed outcome under the design's own existing rules, not a new
   category this section invents.

#### 16.1.4 Honest plausibility verdict — can Phase-1b show zero-shot h≥2 at all?

**No, not with confidence, and this should be stated plainly rather than
discovered by a reviewer.** Even under the MOST favorable reading of the
literature this design's own litreview surfaced
(`research/reasoning-link-litreview-2026-07-07.md`): Zoology and Based's
in-context recall results, and the credible rival account in Okpekpe &
Orvieto (arXiv:2508.19029) about optimization-gated recall in recurrent
models, are uniformly about models **trained on the task's own data
distribution** (MQAR-format training, or fine LR-grid training runs on
synthetic recall data) — not zero-shot application of a general-purpose
pretrained LM to a task-shaped prompt it has never seen a single example
of. This is the crux the task brief names, and it holds regardless of how
wikitext-native the surface form is made: **making the SENTENCES look
like wikitext does not make the TASK (extract, from one forward pass,
enough relational structure to answer a 2-4-hop query about entities never
seen together before) something wikitext pretraining teaches.** Sanford,
Hsu & Telgarsky's k-hop induction-head work (arXiv:2402.09268) establishes
hop-depth-dependent circuit depth requirements for TRANSFORMER models
performing this class of task — but again as an architectural/training
question, not a zero-shot-generalization guarantee. The general LLM
literature on zero-shot multi-hop QA (not specifically cited in this
design's litreview, flagged here as a related-work gap to close before any
publication) is consistent with genuine difficulty at 2+ hops without
chain-of-thought scaffolding, which this design's single-forward-pass,
no-decoding-loop readout structurally cannot provide.

**Consequence for how Phase-1b should be read even if it "succeeds":** a
clean Phase-1b pass most plausibly lands as **h=1 becomes the primary,
confirmatory result (in-context one-hop associative recall, a capability
this size class is documented to have some purchase on) and h≥2 stays
exploratory** (§16.1.3 item 3) — not as a demonstration that the keystone
question (multi-hop composition) has been answered zero-shot. **A
successful Stage-0 gate and full grid under Phase-1b does not, by itself,
resolve H_LINK-A/B; it removes one confound (surface form) from an
instrument that may still need Path (ii) to reach the keystone's own
h≥2 claim.** This is registered now, not after a positive h=1 result
tempts an overclaim.

---

### 16.2 PATH (ii) — PHASE-2: task familiarization (Rev 5, 2026-07-07 — post-fifth-attack; status-sync-only patch over Rev 4, recipe content unchanged)

**Rev 1 status note (superseded by Rev 2, itself superseded by Rev 3
immediately below; kept intact for the historical record, per this
document's own "record history, don't retcon it" convention, §13.5).** An
independent attack round reviewed this recipe before any code was
written, per §16.2.4's own registered prerequisite and this project's
waterfall discipline (`CLAUDE.md`). Verdict: **NEEDS-REDESIGN** — 1
FATAL, 6 MAJOR, 1 MINOR. Every finding was fixed in Rev 1 (§16.2.1-
§16.2.4); the full finding→fix trace is recorded in §16.7, per house
style.

**Rev 2 status note (superseded by Rev 3 immediately below; kept intact
for the historical record, same convention).** Rev 1's own §16.2.4
registered its next step explicitly: "Rev 1's own fixes have not yet had
their own independent audit pass... the next concrete step... is a FRESH
adversarial pass targeting Rev 1 specifically." That pass has now run —
a second independent reviewer, different from attack-round-1, per house
discipline. Verdict: **NEEDS-REVISION** — 6 new MAJOR, 2 new MINOR, **no
FATAL** (every Rev 1 fix was independently re-verified and HOLDS; this
round's findings are code-reality/buildability gaps in Rev 1's own fixes,
not design-logic errors). Every finding was fixed in Rev 2 (§16.2.1-
§16.2.4 as they read at that revision); the full finding→fix trace is
recorded in §16.9, per house style, mirroring §13.x's and §16.7's own
attack-round convention.

**Rev 3 status note (superseded by Rev 4, itself superseded by Rev 5
immediately below; kept intact for the historical record, same
convention).** Rev 2's own §16.2.4 registered its next step
explicitly: the next concrete step before build is "a THIRD, fresh
adversarial pass targeting Rev 2 specifically." That pass has now run —
a third independent reviewer, different from attack-rounds 1 and 2, per
house discipline. Verdict: **NEEDS-REVISION** — 3 new MAJOR, 4 new MINOR,
**no FATAL** (every Rev 1 AND Rev 2 fix was independently re-verified and
HOLDS; this round's findings are entirely in previously-unattacked
territory — a FALSE code citation inside Rev 2's own query-loss mechanism
[the claimed "two-forward continuation pattern" does not exist in §4.4's
own code], an undisclosed query-count assumption in Rev 2's own cost
arithmetic, a MECE gap in the pentachotomy with a concrete counter-example,
and four buildability/precision gaps). Every finding is fixed below; none
is deferred or waved away. The full finding→fix trace is recorded in
§16.10, per house style, mirroring §16.7's and §16.9's own attack-round
convention.

**Rev 4 status note (superseded by Rev 5 immediately below; kept intact
for the historical record, same convention).** Rev 3's own §16.2.4
registered a FOURTH fresh adversarial pass targeting Rev 3 specifically.
That pass has now run — a fourth independent reviewer. Verdict:
**NEEDS-REVISION** — 3 new MAJOR, 0 new MINOR, **no FATAL** (every Rev
1/2/3 fix independently re-verified and HOLDS; this round's findings are
internal-consistency bugs inside Rev 3's own fixes: a `c1`-selection
rule that could silently misroute a real PERSISTENT trajectory, an
arithmetic error in the totality narration, and a false field citation).
All fixed in Rev 4; trace in §16.12.

**Rev 5 status note (current).** Rev 4's own §16.2.4 registered a FIFTH
fresh adversarial pass targeting Rev 4 specifically. That pass has now
run — a fifth independent reviewer. Verdict: **NEEDS-REVISION** — 2
MAJOR, 1 MINOR, **no FATAL**, ALL prose/status-sync defects (stale
pre-Rev-4 narration in §16.2.4's body and §16.6 item 4, and this
preamble's own status-note chain stopping at Rev 3); **every substantive
check PASSED** (all three Rev-4 fixes verified landed byte-accurate
against source; an independent from-scratch 32-pattern re-enumeration
under the holds-anchored `c1` reproduced 1+15+1+4+11=32 and confirmed
the round-4 counter-example classifies PERSISTENT/c1=500; the
diff-scoped new-bug sweep found nothing). All fixed in Rev 5 — a
status-sync-only patch; recipe content, build tasks, cost bracket, and
all algorithms are UNCHANGED from Rev 4. Trace in §16.13. Read
§16.2.1-§16.2.4 below as Rev 5 (content-identical to Rev 4 except
status prose) — passages carrying round-3/round-4 fixes remain marked
inline (`Rev 3 fix, attack-round-3 MAJOR-R3-n`, `Rev 4 fix,
attack-round-4 MAJOR-R4-n`); unmarked passages are carried forward
unchanged.

**What it tests.** Not "does a never-task-trained checkpoint show
zero-shot multi-hop composition" (§16.1.4's own answer: implausible by
construction) but **"does the frozen-bias intervention's stabilized key
geometry help a model LEARN in-context binding, once it is actually given
task-relevant training signal"** — a learnability question, not a
zero-shot-presence one. This is the capability question §11's own
gated-Phase-2 sketch already anticipated, and it is the only one of the
three paths that reconnects H_LINK directly to the capacity law's own
empirical basis: every K/d cliff measurement in this project (`x0=0.5455`,
d=128 dissolution, coherence exoneration) was made on models **trained on
the exact BIND/QUERY task** — Zoology/Based/MQAR's own in-context-recall
successes are likewise task-trained, per §16.1.4. Path (ii) is the only
path here that puts REASONING-LINK's checkpoints in that same regime.

#### 16.2.1 Recipe sketch (Rev 5, content-identical to Rev 4 — not a committed build, a design sketch that has now survived five attack rounds (round 5's substantive checks all PASSED; its findings were status-prose only, §16.13); folds in Rev 1's FATAL-1, MAJOR-1, MAJOR-2, MAJOR-3, MAJOR-4, MAJOR-6, MINOR-1 per §16.7, Rev 2's MAJOR-NEW-1 through MAJOR-NEW-6, MINOR-NEW-1, MINOR-NEW-2 per §16.9, Rev 3's MAJOR-R3-1a, MAJOR-R3-1b, MAJOR-R3-2, MAJOR-R3-3, MINOR-R3-1, MINOR-R3-2, MINOR-R3-3, MINOR-R3-4 per §16.10, AND the round-4 audit's 3 MAJORs (c1 holds-anchor, totality count, checkpoints-field citation) per §16.12)

**Base checkpoints:** continue training from the archived Leg-A frozen-bias
checkpoints at their final (step-20,000) state — all 3 arms (`off`,
`per_token`, `global`) × 2 corpora × 3 seeds, the same 18 core cells §5.1
already scopes — rather than §11's original "train NEW small LMs from
scratch" sketch. This is a deliberate departure from §11, chosen because
continuing an ALREADY-VAL-LOSS-MATCHED set of checkpoints (§7.2's existing
gate) preserves the one clean apples-to-apples property this whole program
has fought to establish across arms; training fresh models re-opens every
val-loss-matching question from scratch. **Scope is 14M-class (rung-1)
only** — these are the only frozen-bias checkpoints that exist anywhere in
this program's archive (`FROZEN_BIAS_LM_DESIGN.md` §6.2/§8.1: rung-2, 98M,
was formally PARKED before any rung-2 cell ever launched); §16.2.3 below
re-derives cost on this basis explicitly.

**Build task registered (Rev 2 fix, attack-round-2 MAJOR-NEW-3) — no code
path exists today to load a pretrained checkpoint before training.**
Verified directly against `lm_pretrain_rd.py`: `main()` always
fresh-constructs `DeltaNetLM(...)` (L3016-3031) and passes it straight
into `train()` (L3053), which itself fresh-constructs `AdamW`
(L1795, see MAJOR-NEW-4 below) and loops `for step in range(1,
args.steps+1)` (L1822) — there is no `--init-checkpoint` flag anywhere in
the argparser (L2822-2935), and the only `load_state_dict` calls in the
whole file are inside Stage −1 self-tests (L2222, L2251) or an in-run
weight-copy for a smoke comparison (L2518), never in the real training
path. "Continue training from the archived Leg-A checkpoint" (above) is
therefore a design-level requirement with **no corresponding build
today.** Registered fix: add `--init-checkpoint <path>` to `main()`'s
argparser; immediately after `DeltaNetLM(...)` construction (L3031) and
before `train()` is called (L3053), `load_state_dict` the archived Leg-A
frozen-bias step-20,000 checkpoint's `model_state_dict` into the freshly
constructed model — the same pattern already proven correct by the
existing Stage −1 round-trip self-test ("torch.save -> fresh model ->
load_state_dict -> ...", L2208-2222), applied here to a REAL archived
checkpoint path instead of an in-run throwaway. A **new Stage −1
self-test** (mirroring that round-trip test's own assertion style) must
confirm the loaded model's first post-load forward pass exactly
reproduces the archived checkpoint's own last-recorded eval metrics
(val loss, within float round-trip tolerance) before any familiarization
step runs — closing the silent-fresh-start failure mode this finding
names.

**Dependency named explicitly (Rev 3 fix, attack-round-3 MINOR-R3-3;
field citation corrected, Rev 4 fix, attack-round-4 MAJOR-R4-3 — see
§16.12) — the archived checkpoint's own `.pt` file does NOT contain the
val-loss metrics this self-test needs to compare against.** Verified
directly: the ONLY fields `torch.save` writes at a training checkpoint
are `step`/`model_state_dict`/`config`/`corpus`/`seed`/`run_name`
(`lm_pretrain_rd.py` L1933-1935) — no `val_loss`, no `eval_windows`,
nothing metric-bearing. The self-test's own "last-recorded eval metrics"
comparison target must therefore be read from a SEPARATE source: the
ORIGINAL Leg-A training run's own results JSON — the file `main()`'s own
`--out` argument wrote (`json.dump(result, ...)`, `lm_pretrain_rd.py`
L3061-3062; `result`'s own `"checkpoints"` field — NOT `"trajectory"` as
Rev 3's text previously said. `train()` declares both as separate lists,
`trajectory, checkpoints = [], []`, L1801: `trajectory` entries (appended
L1858, at the `log_every` cadence, `step % args.log_every == 0`) carry
only `"step"`/`"loss"`/`"lr"`/`"grad_finite"`/`"skip_rate_so_far"` — no
`val_loss` key at all; `checkpoints` entries (appended L1950, at the
`ckpt_every` cadence, `step % args.ckpt_every == 0`, L1863) are built as
a `res` dict that DOES carry `"val_loss"`, L1937-1939. Both lists surface
as separate top-level fields in the assembled result JSON —
`"trajectory": trajectory, "checkpoints": checkpoints`, L1748), located
by the archived checkpoint's own `run_name`/`step`/`corpus`/`seed` fields
(all of which ARE in the `.pt`, enough to identify which results JSON and
which `checkpoints` entry to cross-reference against). This
cross-reference is a named, explicit build dependency (the results JSON
archive path must be resolvable from the checkpoint's own recorded
fields) pinned here at design time, not something discovered as a missing
input when the self-test is first run.

**Negative test registered (Rev 3 fix, attack-round-3 MINOR-R3-2)** —
the positive round-trip test above proves loading WORKS on a matching
checkpoint; it does not prove a MISMATCHED checkpoint is REJECTED rather
than silently loaded partially. A companion negative test is mandatory:
construct a deliberately-corrupted checkpoint (a `model_state_dict` with
one tensor's shape altered, or a `config` dict with a mismatched
`d_state`/`n_layers`) and assert that loading it via `--init-checkpoint`
raises/ABORTS — via `load_state_dict`'s own default `strict=True`
shape-mismatch error, or an explicit config-equality assert added at the
load site — rather than silently proceeding with a partially- or
mis-loaded model.

**What continues training, what stays frozen:** the frozen-bias table
itself and its blend mechanism (`apply_frozen_bias_blend`, λ=0.58) stay
exactly as trained — this design never re-tunes λ or re-initializes the
table, since the point is to test whether the ALREADY-STABILIZED geometry
helps learning, not to re-derive a new stabilization. Every other
parameter (backbone, k/q/v/o projections, embedding table) continues
training normally under gradient descent, same optimizer family/schedule
convention as the original Track-C/frozen-bias runs.

**Disclosed confound: optimizer restart is real (Rev 2 fix, attack-round-2
MAJOR-NEW-4) — "same optimizer family/schedule convention" is NOT the
same as "the same optimizer state," verified directly.** Checkpoints save
`model_state_dict` only — plus `step`, `config`, `corpus`, `seed`,
`run_name` (`torch.save` call, L1933-1935); there is no
`optimizer_state_dict` field anywhere. Combined with MAJOR-NEW-3's own
finding above (fresh `AdamW(model.parameters(), lr=args.lr, ...)` at
L1795, on every call to `train()`), familiarization ALWAYS begins with
**zero Adam moments** (`m_0=v_0=0`), regardless of how converged the
archived checkpoint's own optimizer state was at step 20,000. Worse:
`get_lr(step, max_lr, warmup_steps, total_steps)` (L1308-1317) is a pure
function of `step` alone, and `train()`'s own loop always starts `step`
at 1 (`for step in range(1, args.steps+1)`, L1822; `lr = get_lr(step,
args.lr, args.warmup_steps, args.steps)`, L1824) — familiarization
therefore runs a **fresh full linear-warmup + cosine-decay cycle from
`args.lr`**, stacked on top of a checkpoint whose weights were already
shaped by a full PRIOR decay down to `args.lr × min_lr_ratio` at step
20,000. This is real, and it is **uniform across all 18 cells** — every
arm, every corpus, every seed restarts identically, so it is not, by
itself, an arm-differential confound able to manufacture a false
global-vs-off separation. **But it licenses a second, registered reading
of any TRANSIENT or LATE-EMERGENT trajectory outcome (§16.2.1's
outcome hexachotomy, below): a separation that opens then closes, or one
that appears only very late, could be a symptom of shared LR-restart/
optimizer-warmup dynamics common to every arm** — a rival account
distinct from §16.2.2 confound (a)'s own new-mix-objective training-
dynamics concern. A TRANSIENT or LATE-EMERGENT finding must name BOTH
rival accounts (new-mix training dynamics, LR/optimizer restart) as live,
undifferentiated explanations, not attribute the finding to one over the
other without further evidence — the hexachotomy's own PERSISTENT-only-
counts-as-durable convention is the registered defense against both
readings simultaneously.

**Familiarization query loss — pinned explicitly (Rev 2 fix,
attack-round-2 MAJOR-NEW-1, this round's own highest-value finding): Rev 1
registered WHICH queries get trained on (h∈{1,2}, below) but never
defined what loss function scores them. Fixed here.**

**Definition (Rev 3 fix, attack-round-3 MAJOR-R3-1a — Rev 2's own
mechanism citation below was FALSE, caught by reading the real files, not
merely by re-deriving on paper).** Rev 2's text here previously claimed
this reuses "the SAME two-forward continuation pattern
`reasoning_link_probe.py`'s own eval-time extraction already uses
(§4.4)." Checked directly against `reasoning_link_probe.py` and false:
§4.4's own forward-B (`run_forward_b`, called from `measure_cell_all_h`)
runs over the BIND phase and the query window CONCATENATED into ONE
call — `main_concat = torch.cat([bind_expanded, query_flat], dim=1)`
(`reasoning_link_probe.py` L1370, inside the L1358-1404 block) — never as
a continuation over the query window ALONE with `initial_states` carried
from a prior bind-only call. `forward_body` (the probe's own shared
forward helper, L652-678) defaults `initial_states=None`, and every
caller inside the probe (`run_forward_a`, L729-742; `run_forward_b`,
L770-788) leaves that default untouched — nothing in
`reasoning_link_probe.py` ever calls `forward_body` with a non-`None`
`initial_states`. The REAL two-phase context/continuation pattern
(`initial_states` carried across two SEPARATE calls) lives in a
DIFFERENT script serving a DIFFERENT purpose: `lm_intervene_rd.py`'s
rank-truncation intervention (context forward with `initial_states=None`
returning `ctx_states`, L208; a SEPARATE, later, scored continuation
forward with `initial_states=init_states`, L230) — Rev 2 conflated the
two scripts' two genuinely different mechanisms.

**This is not merely a citation error — the continuation pattern has a
real correctness gap that makes it the wrong mechanism to repurpose for
training, verified directly, and is why it stays rejected here rather
than merely mis-cited.** `DeltaNetLMMixer.forward`'s own q/k/v
`ShortConvolution`s are called with no cache argument at all —
`q, _ = self.q_conv1d(self.q_proj(x))`, `k, _ = self.k_conv1d(self.k_proj(x))`,
`v, _ = self.v_conv1d(self.v_proj(x))` (`lm_pretrain_rd.py` L837-839,
`fla.modules.ShortConvolution`, imported L88) — a causal depthwise
convolution that, absent an explicit cache, implicitly zero-pads its OWN
call's left edge. A continuation call over the ~6-token query window
ALONE (the mechanism Rev 2 mistakenly attributed to §4.4) would therefore
corrupt every layer's q/k/v at the first `conv_size-1=3` positions of that
window (`conv_size=4` is this program's own standard config for every
constructed `DeltaNetLM` in this file, e.g. `lm_pretrain_rd.py` L2014) —
the convolution would see zero-padding where the true local context is
actually the bind phase's own trailing tokens. `DeltaNetLMMixer`'s
per-token sigmoid β-gate then reads that corrupted convolution output and
writes the corruption into the delta-rule state update itself, not merely
into a discardable local feature — a silent, structural corruption at
exactly the query positions the loss is designed to score, every
training step, every cell. This is the concrete reason continuation is
REJECTED here, not merely "not what §4.4 does."

**Corrected definition.** The query loss is scored via the SAME validated
concatenated bind+query recompute §4.4 already uses and already Stage −1
verifies for causal correctness (`causality_check`, `reasoning_link_
probe.py` L791-820, `tol=1e-6` default at L792/L820 — forward-A-only k/v
over the bind prefix must match forward-B's own k/v over that SAME prefix
inside the concatenated call; exercised live at Stage 0's own calibration
cell, `run_stage0`, L1597-1598). Per familiarization training step: form
`concat_tokens = torch.cat([token_ids, query_tokens_one_row], dim=1)` —
the exact `main_concat` construction §4.4 already builds and already
Stage −1-verifies causally safe (`sample_batch_rd`'s own `token_ids`,
`grammar_rd.py` L451-464, and `query_tokens`, `grammar_rd.py` L474-479,
unchanged); run the REAL `DeltaNetLM.forward(concat_tokens,
initial_states=None)` (the model's own class method, `lm_pretrain_rd.py`
L1178-1205 — NOT the probe's VRAM-narrowed `forward_body` helper, since
training needs the LM head's full logits, which `DeltaNetLM.forward`
already computes unconditionally at L1204); take the returned logits at
the concatenated sequence's OWN LAST position — the `<Q>`-marker position,
using §4.1's vocab-safe substitute token, never `pools.query_id`'s
reserved (>50,257) id — and score `F.cross_entropy(logits[:, -1, :],
target_ids)` against `target_ids = torch.gather(entity_ids, 1, tgt_slot)`,
the exact field `grammar_rd.py`'s own [grammar 3c] self-test (L594-616)
already proves recovers "the answer entity's token" (`answer_token =
torch.gather(bt["entity_ids"], 1, bt["tgt_slot"])`, L605) — restricted to
`hop_set=(1,2)` per the hop-split fix below, so `tgt_slot`/`hops`/
`entity_ids`/`query_tokens` are already exactly the fields this loss
needs; zero new episode-construction machinery. `initial_states=None` is
the correct, ALREADY-validated mode here (matches `forward()`'s own
docstring, "the ONLY mode training ever uses," L1180) — not a compromise
forced by rejecting continuation, the mode this design always needed. No
new causal-safety self-test is owed beyond what §4.4 already runs: this
IS §4.4's own construction, reused verbatim for a different loss
(vocab-space CE via the LM head, in place of §4.4's `d_state`-space
Option-1/Option-2 readout), not a new forward pattern.

**Explicitly REJECTED: `run_deltanet_rd.py`'s own `L_cos + NCE_LAMBDA *
L_nce` training objective** (L82-90 header comment, `cosine_loss`
L155-156, `nce_loss`/`nce_loss_fixed_m` L159-215, `loss_config` L375).
That objective scores `pred(a,h) = S_T^h · q_eff` against `v_eff_target`
by cosine/InfoNCE **in `d_state` space** — which is EXACTLY the construct
§4.4's Stage-0.5-gated readout evaluates at eval time (`pred(a,h)` scored
by absolute cosine against `v_eff_target`, §4.4 Rev 4). Training on that
objective would make familiarization directly optimize the quantity the
readout later measures — the readout would then be reporting "did SGD
succeed at the thing it was told to succeed at," not "did the intervention
help the model learn to compose," undercutting the Stage-0.5 gate's own
claimed "independent instrument-validation value" (§16.2.1's Stage-0.5
paragraph below: "if the readout construct ... is actually measuring what
this design claims, premises (iii)/(iv) SHOULD pass ... familiarization
gives the readout every reason to succeed" — that sentence is only true,
and only informative, if the training signal and the readout are measured
in DIFFERENT spaces through DIFFERENT machinery: vocab-space CE through
the LM head for training, `d_state`-space cosine through the raw
recurrent state for eval). The vocab-space CE definition above preserves
that separation by construction; `run_deltanet_rd.py`'s own objective
would not.

**Mix ratio (corpus windows : episode sequences), pinned as a build
default — reconciled with, not overriding, MINOR-1's still-open
pilot-measured fraction below.** Corpus windows (`seq_len=512`) and
episode sequences (`T_bind+query_len ≈ 224-230` tokens at K=32, `≈146` at
K=20 — always shorter, never a concatenable equal-length batch) cannot
literally share one padded tensor, so "composed into the same batch loss"
is built as **two separate forward+CE passes per training step, summed
into one scalar before one `backward()` call**: `L_total = L_corpus +
λ_fam · L_query`, `L_corpus` computed exactly as the existing corpus
training call already does (unchanged), `L_query` as defined above.
**`λ_fam = 1.0` is the pinned BUILD DEFAULT** — equal weighting, one-line
justification: it is the no-thumb-on-the-scale null, and an unequal
default would itself need a justification this design does not yet have.
This is the mechanism knob MINOR-1's own "majority original corpus,
minority BIND/QUERY" data-mix language (Data mix paragraph, below) is
realized through in a two-separate-forward-pass design: a SMALLER
`λ_fam` makes the corpus loss dominate the gradient (operationalizing
"majority corpus"), a LARGER one makes the episode loss dominate.
**MINOR-1's own pilot-measurement requirement is UNCHANGED and NOT
weakened by this pin:** `λ_fam=1.0` is the Stage-0-pilot's own SWEEP
ANCHOR/default, pinned so the loss formula and its Stage −1 self-tests are
concretely buildable and runnable now — the pilot still sweeps a small
grid of candidate `λ_fam` values (centered on this anchor) and still
reports the smallest value at which a detectable familiarization signal
appears, exactly as MINOR-1 already specifies; THAT measured value, not
the `λ_fam=1.0` anchor, is what launches the full 18-cell grid.

**Query count pinned — `n_query=2` (Rev 3 fix, attack-round-3
MAJOR-R3-1b) — Rev 2's own budget paragraph below silently assumed ONE
scored query per episode per step; the real default is `Q=K`, up to 32.**
Verified directly: `sample_batch_rd`'s own `Q = cfg.queries` (`grammar_rd.
py` L367-368 signature, L451 body) reads the `queries` property, which
returns `self.K if self.n_query is None else self.n_query` (`grammar_rd.
py` L335-336) — and `n_query: int | None = None` (L310) is the dataclass
field's own default. Every existing `DeltaNetRDTaskConfig` this design
has cited so far (`episode_config_for_checkpoint`, `reasoning_link_
probe.py` L332-343) leaves `n_query` unset, so it defaults to `Q=K`
(20 or 32, per the K-sweep paragraph above) — meaning the concatenated
recompute defined above, left unpinned, would score EVERY one of K=32
queries per episode per training step, at `main_concat`'s own per-query
row cost (`T_bind+query_len`), not the single query Rev 2's arithmetic
implicitly priced. Registered fix: familiarization's own query-loss
batches use a SEPARATE `DeltaNetRDTaskConfig` instance from the one
eval/trajectory-readout uses — `n_query=2` (`grammar_rd.py`'s existing
field, zero new code: no caller in this codebase currently sets `n_query`
to anything but its `None` default, so this is a config-value choice, not
a build task) — while the eval-time config (`episode_config_for_
checkpoint`) keeps `n_query=None` → `Q=K` UNCHANGED, preserving §4.4's own
Q=K readout convention exactly as before. **Why 2, and why this matches
the hop-split distribution without a new sampling mechanism:** with
`hop_set=(1,2)` (the hop-depth split paragraph below) and `n_query=2`,
`sample_batch_rd`'s own existing per-slot IID hop draw (`hops_pool[torch.
randint(0, len(hop_set), (B, Q), ...)]`, `grammar_rd.py` L469-470,
unchanged) draws each of the 2 query slots independently and uniformly
from `{1,2}` — NOT a guaranteed exactly-one-of-each per episode row (a
given row draws `(1,1)`, `(1,2)`, `(2,1)`, or `(2,2)` each with
probability 0.25), but in EXPECTATION exactly half of all scored queries
across a batch are h=1 and half h=2, matching the `{1,2}` hop-split
training distribution in aggregate with no new deterministic per-row
assignment mechanism to build or Stage −1-verify.

**Budget, honestly re-derived (Rev 3 fix, attack-round-3 MAJOR-R3-1b —
this re-derivation is AUDIT-FORCED, not a silent self-amendment; shown in
full, old vs. new, per this revision's own instruction).** Rev 2's own
text here computed a ~1.5× per-step multiplier from "an episode batch of
`T_bind+query_len ≈ 230` tokens (K=32) is roughly 45% of one 512-token
corpus-batch step's own token count" — that 230-token figure is exactly
ONE query row's worth of tokens, i.e. Rev 2's arithmetic silently assumed
`Q=1` without ever stating so. At the REAL unpinned default (`Q=K=32`),
the honest figure is `32×230=7,360` added tokens per corpus-batch row
against `512` corpus tokens — a `7360/512≈14.4×` addition, i.e. a
**~15× per-step cost multiplier**, which does NOT fit inside the
5-10× debug-tax margin and WOULD have blown the committed bracket had it
gone unnoticed (this is exactly what this finding caught). With
`n_query=2` now pinned instead: added tokens per corpus-batch row
`=2×(T_bind+query_len)`. At K=32 (`T_bind+query_len≈230`, the worst case
of the two committed K's, same convention Rev 2's own paragraph already
used): `2×230=460` tokens against `512` corpus tokens →
`460/512≈0.898` → a **~1.90× per-step cost multiplier** (up from Rev 2's
silently-assumed ~1.5×, but still the honest, disclosed number). At K=20
(`T_bind+query_len≈146`): `2×146=292` against `512` → `292/512≈0.57` →
a **~1.57× multiplier** (lower, consistent with K=32 being priced as the
worst case throughout this section). **Both honest figures (~1.57-1.90×)
remain comfortably inside the ALREADY-REGISTERED 5-10× debug-tax
multiplier** §16.2.3's bracket already carries as margin — so, taken on
its OWN, this fix's own re-derived query-loss multiplier does NOT, by
itself, require widening the training-cost bracket: this paragraph shows
the full worked comparison (old, silently-Q=1-assumed ~1.5× vs. new,
honestly-n_query=2-derived ~1.90× worst-case) precisely because
`CLAUDE.md`'s own "disclosure, not pretending" ceiling discipline
requires showing this arithmetic explicitly rather than re-printing the
old conclusion unexamined. **This is NOT the whole story for the
committed bracket** — §16.2.3 (Cost) separately widens the FINAL bracket
on account of a DIFFERENT finding (MAJOR-R3-3's newly-priced Stage-0.5
per-checkpoint gate cost, below), and states the combined, honestly
re-derived number there; §16.10's own BUDGET paragraph gives the full
audit-trail disclosure of both.

**Hop-depth train/eval split (Rev 1 fix, FATAL-1) — the single most
important correction this revision makes.** Familiarization trains ONLY on
h∈{1,2} queries — **exclusion by sampling, not loss-masking (Rev 2 fix,
attack-round-2 MINOR-NEW-2)**: episode batches are drawn with
`hop_set=(1,2)` passed directly to `sample_batch_rd` (`grammar_rd.py`
L367-372, L469-470), mirroring `run_deltanet_rd.py`'s own training-draw
convention exactly (`hop_set=cfg.H_train`, L689) — h∈{3,4} queries are
never SAMPLED into a familiarization batch in the first place, at any
checkpoint, any arm, any corpus, so there is no gradient to mask (the
earlier "masked loss" framing implied a loss-level suppression mechanism
that does not exist in this codebase's own convention and was never going
to be built that way). Evaluation is read at BOTH h∈{1,2} (trained — reported as a **sanity
ceiling**, confirming familiarization is actually teaching the task at
all) and h∈{3,4} (never trained on — reported as **THE H_LINK test**,
the genuine held-out generalization reading this whole path exists to
produce). This makes `DeltaNetRDTaskConfig.H_train`/`H_test` (§4.3)
**real for the first time in this program.** Under Phase 1's zero-shot
instrument, `H_train=(5,)` was registered explicitly as a
"semantically-inert placeholder" (§4.3) — it satisfied the dataclass's
own periodicity-guard invariant with no gradient ever flowing through it,
since those checkpoints "trained on zero hops of anything." Here,
`H_train=(1,2)` and `H_test=(3,4)` have a genuine referent: SGD actually
optimizes the H_train losses during familiarization. **The periodicity
guard must therefore be RE-VERIFIED for real, not assumed to still pass
vacuously** — this is exactly the collapse failure mode `CLAUDE.md`'s own
standing rule names (a general permutation's `h mod cycle_length`
periodicity silently folding "held-out" hops back into in-distribution or
trivial queries; 50-100% of nominally held-out queries collapsed across
K∈{4,8,12,16} before that fix was applied elsewhere in this codebase).
Checked at both of Phase-2's committed K's (§16.2.1, MAJOR-2, below):
`{1,2,3,4} mod 20 = {1,2,3,4}` and `{1,2,3,4} mod 32 = {1,2,3,4}` — no
residue collision between the `{1,2}` (trained) and `{3,4}` (held-out)
partitions at either K, so the guard passes for real, not merely
structurally. This re-verification is a mandatory Stage −1 assertion for
Phase-2's own build, mirroring the existing per-checkpoint K-floor
assertion convention (§4.3, §9 item 9).

**Data mix:** a blend of (a) the ORIGINAL pretraining corpus
(openr1-mix-ext or wikitext-mix-ext, matching each checkpoint's own arm)
at a majority fraction, to avoid catastrophic forgetting of general LM
capability and to keep familiarization "on top of" rather than "instead
of" the original training distribution, and (b) newly-rendered BIND/QUERY
episodes (h∈{1,2} only, per FATAL-1 above) — using whichever surface form
Path (i) validates if it has already run by this point (§16.6), or the
original marker template if Path (i) has not yet cleared a gate — at a
minority fraction, drawn from a **familiarization pool of entities/cycles
held disjoint from an eval pool** (mirroring §4.5's existing eval-purpose
seed-disjointness discipline, extended here to a genuine train/eval split
this design lacked before this revision, since Phase 1 had none). **Held
out for eval, never in the familiarization mix:** a disjoint sub-draw of
K-cycles (fresh permutations) and, if the entity pool allows it (107 names
is the hard ceiling, §4.2), a disjoint entity sub-pool — this needs its
own arithmetic check against the max committed K (now {20,32}, MAJOR-2
below) before being finalized, mirroring F1's own resolved
arithmetic-impossibility lesson (§13.1) rather than repeating it. This
composes with, and is independent of, the hop-depth split above: a
held-out eval episode is disjoint on BOTH axes simultaneously (unseen
cycle/entity draw AND, for its h=3/4 readings, an unseen hop depth).

**Rev 1 fix (MINOR-1, attack-round) — the majority/minority fraction is a
named open item, not a default picked by feel.** The split above was
originally sketched with no derivation (an assumed "majority original
corpus, minority BIND/QUERY" ratio). Registered now: the actual fraction
must be set by a **Stage-0-scale pilot measurement** (this program's own
calibration-before-sweep discipline, `CLAUDE.md`) that sweeps a small
grid of candidate fractions on ONE checkpoint/corpus/seed and reports the
**smallest fraction at which a detectable familiarization signal
appears** (e.g., h=1 sanity-ceiling accuracy on held-out cycles clears its
own chance floor) — the pilot's own measured floor pins the fraction used
in the full 18-cell grid, not an assumed round number (10%, 20%, ...).
Held open until that pilot runs; **no fraction is committed in this
revision.** **Rev 2 cross-reference:** this fraction is now known to be
operationalized as `λ_fam` (the "Familiarization query loss" paragraph
above, MAJOR-NEW-1's fix) — `λ_fam=1.0` is a pinned BUILD DEFAULT that
makes the loss formula runnable, not the fraction this paragraph still
leaves open; the pilot measurement registered here is unchanged and still
governs the value actually used in the full 18-cell grid.

**K sweep and killer-prediction re-application (Rev 1 fix, MAJOR-2).**
Familiarization and eval both use **K∈{20,32}** — Leg A's own committed
pair (§5.3), not a new K grid invented for Phase-2. This keeps Phase-2
commensurable with Phase-1's own killer-prediction structure and lets the
SAME falsifiable, externally-anchored prediction be re-applied
post-familiarization rather than replaced with an ad hoc "does training
help" reading. **The killer-prediction pass condition is re-applied
exactly as §5.3 states it**, substituting the familiarized checkpoints'
own per-trajectory-checkpoint training-effect delta for Phase-1's static
one: `|Δ(K=32)| > |Δ(K=20)|`, with K=32's CI excluding zero while K=20's
does not, checked at each trajectory readout (MAJOR-1, below) — the
near-cliff K (32, ratio 0.5 at d=64) is predicted to show any LEARNED arm
separation first and most strongly; the low-load K (20, ratio 0.3125) is
predicted to lag or stay flat, mirroring the same capacity-cliff-anchored
logic §5.3 already registers for the zero-shot instrument.

**This capacity connection is d=64-cliff-specific, not a general
K/d≈0.55 claim — registered explicitly to prevent overreach** (cites
`KEY_ANCHORING_SCALING_DRAFT.md` §15.19's own harvested verdict,
2026-07-07): that wave's own **WAVE VERDICT: AMBIGUOUS** — d=80 alone is
a clean, non-degenerate REFUTE of ratio-invariance (`x0(80)=0.6756`, CI
`[0.6620,0.6868]`, excluding the pre-registered `[0.4745,0.6165]` band
entirely), and d=96's fit is degenerate (`degenerate_frac=98.52%`, no
cliff visible anywhere in the tested `K/d∈[0.25,0.71875]` window). The
general "K/d≈0.55 is where composition-relevant capacity cliffs sit" law
does **NOT** generalize past d=64. Only the LOCATED d=64 point
(`x0=0.5455`, CI `[0.5385,0.5513]`, §12.9) has any anchoring value for
Phase-2 — precisely because Phase-2's own checkpoints ARE the d=64
architecture (Track C's rung-1 config, `d_state=64`,
`FROZEN_BIAS_LM_DESIGN.md` §6.1). K∈{20,32} is chosen for its
relationship to d=64's own measured cliff, not as one instance of a
general scaling law — a distinction Phase-2's own reporting must
preserve.

**Trajectory readout schedule and pre-registered outcome hexachotomy (Rev
1 fix, MAJOR-1/MAJOR-3, originally a trichotomy; widened to a pentachotomy,
Rev 2 fix, attack-round-2 MAJOR-NEW-6; widened again to the current
six-way hexachotomy, Rev 3 fix, attack-round-3 MAJOR-R3-2, below).**
Rather than reading arm contrasts only at the
terminal checkpoint of the familiarization budget, Phase-2 takes a full
trajectory: checkpoints at **steps {250, 500, 1000, 2500, 5000}** of the
now-fixed 5,000-step familiarization budget (superseding the prior "order
1,000-5,000 steps" open-ended framing — 5,000 is the committed ceiling,
not an estimate). The killer-prediction contrast above is computed at
EVERY checkpoint, for every arm, both corpora, all 3 seeds — not just once
at the end. This directly answers §16.2.2's confound (a) in a different
way than the Stage-0.5 gate below does: if arm separation is a training-
DYNAMICS artifact (per-token/global differing in how fast they converge
on the new mix, not in what they converge TO), a single terminal-
checkpoint readout cannot distinguish that from a genuine, durable
capability difference — a trajectory can.

**Build task registered (Rev 2 fix, attack-round-2 MAJOR-NEW-2) — this
exact cadence is not buildable with the existing flag.** Verified
directly: `--ckpt-every` (`lm_pretrain_rd.py` L2853) gates checkpointing
by `step % args.ckpt_every == 0` (L1863) — a MODULO test, which cannot
produce the non-arithmetic sequence `{250,500,1000,2500,5000}` (no single
divisor `d` fires at exactly those 5 steps and no others). **Explicitly
forbidden: the `--ckpt-every 250` workaround** — tempting because it's a
zero-code way to "cover" the 5 target steps, but it silently checkpoints
and runs the full mid-training eval pass (`val_loss_same`/`val_loss_other`
/rank stats/etc., L1863-1936) **20 times instead of 5, a 4× eval-pass
cost inflation** that would silently 4× the "negligible eval passes" cost
line §16.2.3's own cost derivation prices separately from training,
without ever showing up as a re-derived number. Registered fix: add
`--ckpt-steps` (a new argparse flag, `type=int, nargs="+"`, e.g.
`--ckpt-steps 250 500 1000 2500 5000`) accepting an explicit, sorted,
arbitrary integer list; gate checkpointing on `step in ckpt_step_set` (a
Python `set` built once from the parsed list) instead of the modulo test,
for this design's own launch only — the existing `--ckpt-every` modulo
path is UNCHANGED and stays available for every other script in this
codebase that already depends on it (its own `>2000` warning guard,
L2950-2951, is untouched). A **new Stage −1 self-test** must assert
`sorted(ckpt_step_set) == [250,500,1000,2500,5000]` and that exactly 5
checkpoint files are written per cell at the end of a real launch —
closing the exact "silently 4×'d the eval-pass budget" failure mode this
finding names. **Negative test registered (Rev 3 fix, attack-round-3
MINOR-R3-2)** — the positive assertion above proves the flag WORKS when
given the target list; it does not prove the gate actually EXCLUDES steps
OUTSIDE that list. A companion negative test is mandatory: launch a short
run with `--ckpt-steps 250 500` (a strict SUBSET of the real target list)
for a handful of steps past 750, and assert NO checkpoint file for step
750 (or any step other than 250/500) is ever written — closing the case
where `step in ckpt_step_set`'s gate is accidentally satisfied by some
OTHER condition (e.g. a stray `or` clause silently reintroducing the
modulo test alongside it) that the positive test alone cannot detect.

**Blend-off surgery restated INLINE (Rev 2 fix, attack-round-2
MAJOR-NEW-5) — not merely cited to §5.2a, since Phase-2's own blend is
"doubly live" in a way Phase-1's was not.** §5.2a fixed a real confound
for Phase-1's own (static, single-checkpoint) eval: `DeltaNetLMMixer.forward`
applies `apply_frozen_bias_blend` UNCONDITIONALLY whenever
`self.frozen_bias_arm != "off"` (`lm_pretrain_rd.py` L854-857), including
during a probe's own eval forward pass — so scoring an arm checkpoint's
native forward conflates (i) what TRAINING did to the weights with (ii)
the blend mechanically re-applying itself live at eval time. Phase-2
inherits this exact mechanism UNCHANGED, but now applies it TWICE over:
once during familiarization TRAINING itself (which correctly keeps the
blend ON — continuing to train the arm checkpoint the way it was
originally trained is the whole point, the "What continues training"
paragraph above), and again at EVERY ONE of the 5 trajectory-checkpoint
SCORING passes, not Phase-1's single terminal read — five separate
opportunities to silently skip the surgery instead of one. **Registered
requirement, stated here so it cannot be missed:** before computing
`rec@0.9` (or any Stage-0.5 gate quantity) at ANY trajectory checkpoint,
for ANY arm checkpoint, wrap the scoring forward pass in
`reasoning_link_probe.py`'s own `frozen_bias_surgery(model,
force_off=True)` context manager (L681-702 — the AS-BUILT,
mutation-tested production tool, not the raw one-line attribute flip it
wraps: `blk.mixer.frozen_bias_arm = "off"` for every block, restored on
exit even under exception). This measures what TRAINING did to the
weights (§5.2a's own "training-effect" definition, `rec@0.9`(arm ckpt,
blend-OFF) − `rec@0.9`(off ckpt)), never the blend mechanically
re-applying itself — at every checkpoint, not just the terminal one. The
mechanical-effect contrast (§5.2a, blend-ON vs. blend-OFF on the SAME
trained weights) remains available as a non-load-bearing side reading at
any checkpoint a reader wants it, exactly as §5.2a already specifies,
never substituting for the surgically-isolated training-effect reading
above.

**Six pre-registered outcomes classify each (K, corpus, seed) trajectory,
MECE for real this time (Rev 3 fix, attack-round-3 MAJOR-R3-2 — Rev 2's
own five-outcome pentachotomy was NOT actually MECE, disproven by a
direct counter-example: the pattern `holds(250)=TRUE,
holds(500)=holds(1000)=holds(2500)=FALSE, holds(5000)=TRUE` satisfies
NONE of PERSISTENT [requires a monotone TRUE run all the way to 5,000],
TRANSIENT [requires `holds(5000)=FALSE`], LATE-EMERGENT [requires FALSE
at every checkpoint before 5,000], or CONVERGED-EQUIVALENT/UNRESOLVED
[both require FALSE at EVERY checkpoint] — it maps to ZERO of the five
buckets. A sixth bucket and a total, exhaustive-by-construction precedence
rule close this below, checked against the full `2^5=32`-pattern space of
`holds(c)` truth-assignments across the 5 checkpoints, not merely the one
adversarial example — see the worked table after the outcome list.
Rev 1's original trichotomy gap (no outcome for "separation appears only
at the very last checkpoint," and the "equivalent" vs. "we cannot tell"
conflation under one CONVERGE label) was already closed in Rev 2
(LATE-EMERGENT, CONVERGED-EQUIVALENT/UNRESOLVED); this round's gap is a
distinct, newly-discovered non-exhaustiveness in Rev 2's own five-way
split, not a re-opening of Rev 1's.**

**Mechanical primitives (reused, no new CI formula — §5.4's own "pinned
CI formula throughout" instruction), defined once and applied identically
at every checkpoint `c` ∈ {250, 500, 1,000, 2,500, 5,000}:**
- `det(K,c)` — TRUE iff `Δ(K)`'s pinned 3-seed CI at checkpoint `c`
  EXCLUDES zero (the SAME CI computation §5.3's killer-prediction
  condition already reads for K=32/K=20); FALSE ("indeterminate") iff the
  CI straddles zero.
- `holds(c)` — the full killer-prediction pass condition at checkpoint
  `c`, exactly as the K-sweep paragraph above already states it:
  `det(32,c)=TRUE AND det(20,c)=FALSE AND |Δ(32,c)|>|Δ(20,c)|`.
- `det_arm(arm,c)` — TRUE iff `arm`'s own training-effect `Δ(K=32,c)` CI
  (that arm vs. off, §5.2a) excludes zero — training clearly did
  something measurable relative to off, independent of what any other arm
  did.
- `agree(c)` — TRUE iff the global-arm's and per-token-arm's own
  `Δ(K=32,c)` CIs (both already computed at every checkpoint per the
  Trajectory-readout paragraph above) OVERLAP each other — a direct
  interval comparison on two CIs already being read at that checkpoint,
  not a new statistic.

**Outcomes, checked in this fixed precedence order so exactly one
applies:**
1. **PERSISTENT (`c1` anchor corrected, Rev 4 fix, attack-round-4
   MAJOR-R4-1 — see §16.12)** — let `c1` be the FIRST checkpoint, in
   trajectory order, where `holds(c1)=TRUE`. **Tie-break (closes the
   exact boundary case attack-round-2 flagged): `c1` must be a
   NON-TERMINAL checkpoint** (`c1 ∈ {250,500,1000,2500}`) — if `holds(c)`
   is FALSE at every checkpoint before 5,000 and only becomes TRUE at
   5,000 itself, that case is NOT PERSISTENT (there is no later checkpoint
   for "continues to hold" to be non-vacuously true against); it routes to
   LATE-EMERGENT below. Given a valid non-terminal `c1`: PERSISTENT iff
   `holds(c)=TRUE` at `c1` itself (true by construction of `c1` as the
   first checkpoint where `holds` fires — Rev 3's separate
   `det(32,c1)=TRUE`-only selection step is dropped here, since
   `det(32,c)=TRUE` does NOT imply `holds(c)=TRUE` [`holds` also requires
   `det(20,c)=FALSE` AND `|Δ(32,c)|>|Δ(20,c)|`] and could misroute a real
   PERSISTENT trajectory — e.g. `det(32,250)=TRUE` but `holds(250)=FALSE`,
   `holds(500..5000)=TRUE` — to NON-MONOTONE under the old anchor) AND
   `holds(c)=TRUE` at every later checkpoint through 5,000 inclusive.
   Reads as a durable, training-stable separation, corroborated by an
   early reading — **but ONLY a valid corroboration if Stage-0.5 (below,
   now re-measured at EVERY checkpoint per Rev 3 fix MAJOR-R3-3) also
   passed at `c1` itself (Rev 3 fix, attack-round-3 MAJOR-R3-3's own named
   requirement):** if the familiarized OFF-arm's own premises (iii)/(iv)
   or h1-floor gate FAILED at `c1`, that checkpoint's arm-contrast is
   uninterpretable by §16.5 Constraint 1's own gates-must-abort rule and
   cannot serve as PERSISTENT's early corroboration even though
   `holds(c1)=TRUE` was computed — a PERSISTENT verdict must re-identify
   `c1` as the first NON-TERMINAL checkpoint, WITHIN the already-confirmed
   monotone `holds`-TRUE run through 5,000, where Stage-0.5 passed,
   skipping past any earlier `holds`-true-but-Stage-0.5-failed checkpoint
   (and re-applying the same monotone-through-5,000 condition from that
   later `c1`).

   **Rev-5-build note (BUILD agent, this commit) — resolves the open item
   §16.13 disclosed but left out of that round's tight scope: "the
   Stage-0.5 skip-past provision does not state what happens if Stage-0.5
   never passes at ANY non-terminal checkpoint within an otherwise-
   confirmed monotone holds-true run."** PINNED: when the skip-past search
   above finds NO non-terminal checkpoint within the confirmed monotone
   run whose own Stage-0.5 gate passes (i.e. Stage-0.5 FAILS at every
   candidate `c1` from `c1_raw` through the last non-terminal checkpoint
   inclusive), the trajectory is reported **UNRESOLVED-GATE** — a
   SEVENTH, named, disclosed outcome, distinct from all six outcomes
   above (never silently folded into PERSISTENT, NON-MONOTONE, or
   UNRESOLVED) and reported per-seed only, exactly like NON-MONOTONE's
   own disclosure discipline: never averaged into a headline number or a
   mean effect size alongside cleanly-classified seeds. UNRESOLVED-GATE's
   own reading: the killer-prediction condition may genuinely hold at
   every checkpoint in a monotone run, but this instrument cannot
   CERTIFY that reading as interpretable at any point in the run (every
   candidate corroborating checkpoint's own premises (iii)/(iv)/h1-floor
   failed) — a DIFFERENT epistemic status from UNRESOLVED (which means
   the best-powered terminal reading itself never clears noise) and from
   NON-MONOTONE (a genuinely unclassifiable `holds` shape): UNRESOLVED-
   GATE means the `holds` shape IS classifiable (monotone, PERSISTENT-
   shaped) but the READOUT INSTRUMENT itself failed its own validity
   check throughout. Implemented mechanically in
   `phase2_hexachotomy.classify_trajectory` (`deltanet_rd/
   phase2_hexachotomy.py`), unit-tested against this exact scenario
   (Stage-0.5 failing at every non-terminal checkpoint of a confirmed
   monotone run) as part of this build's own Stage -1 self-tests
   (`deltanet_rd/phase2_stage_minus1.py` item 5).
2. **TRANSIENT** — `holds(c)=TRUE` for at least one `c` ∈
   {250,500,1000,2500} AND `holds(5000)=FALSE`. Reads as a
   training-dynamics artifact of exactly the kind §16.2.2 confound (a) —
   and now also MAJOR-NEW-4's disclosed LR/optimizer-restart reading,
   above — warns about, not a durable capability difference.
3. **LATE-EMERGENT (new outcome, closes the Rev 1 exhaustiveness gap)** —
   `holds(5000)=TRUE` AND `holds(c)=FALSE` for every `c` ∈
   {250,500,1000,2500} (no earlier checkpoint ever satisfied the pass
   condition, and the tie-break above routes here rather than to
   PERSISTENT). Reads as a separation with NO earlier corroborating
   checkpoint: consistent with a durable effect that simply takes the
   full familiarization budget to resolve above noise, but equally
   consistent with a late-training artifact (the LR-restart cosine-decay
   tail MAJOR-NEW-4 discloses concentrates its largest per-step change
   near the end of the schedule) — reported as a qualified, NOT
   full-strength PERSISTENT finding.
4. **CONVERGED-EQUIVALENT** — `holds(c)=FALSE` for every `c` ∈
   {250,...,5000} (the pass condition never holds), AND, read at the
   TERMINAL checkpoint (5,000, the best-powered point in the trajectory):
   `det_arm(global,5000)=TRUE` AND `det_arm(per_token,5000)=TRUE`
   (**condition widened, Rev 3 fix, attack-round-3 MINOR-R3-1 — BOTH
   non-off arms must individually show a real, determinate, non-zero
   training effect vs. off, not global alone.** Rev 2's own global-only
   condition let a merely-WIDE per-token CI [`det_arm(per_token,5000)
   =FALSE`, indeterminate — not evidence of equivalence, just evidence
   that per-token's own effect could not be distinguished from zero] pass
   silently as if it were positive evidence, mislabeling asymmetric
   uncertainty as equivalence) AND `agree(5000)=TRUE` (global's and
   per-token's own effects are statistically indistinguishable from EACH
   OTHER). Reads as genuine arm-to-arm equivalence: both arms clearly
   learn the task (Stage-0.5 gate + h∈{1,2} floor, both still required,
   unchanged from Rev 1, NOW re-checked at the terminal checkpoint
   specifically per the per-checkpoint Stage-0.5 gate, MAJOR-R3-3 below)
   and BOTH individually produce a real, measurable, determinate effect,
   but that effect does not differentiate global from per-token.
5. **UNRESOLVED (closes the Rev 1 conflation gap; condition widened to
   match #4's fix, Rev 3 fix, attack-round-3 MINOR-R3-1)** —
   `holds(c)=FALSE` for every `c` ∈ {250,...,5000} (same shared
   precondition as CONVERGED-EQUIVALENT), BUT CONVERGED-EQUIVALENT's own
   three-part terminal condition does NOT hold in full — i.e.
   `det_arm(global,5000)=FALSE` OR `det_arm(per_token,5000)=FALSE` OR
   `agree(5000)=FALSE`. Even at the best-powered terminal checkpoint, at
   least one arm's own effect vs. off never clears noise (the common
   case — reads as underpowered, too few seeds/too much variance to say
   anything), OR both arms individually clear noise but disagree with
   EACH OTHER without ever tripping the specific killer-prediction
   pattern `holds` scores (a rarer sub-case — a real, determinate,
   NON-equivalent effect that simply never matched the K32-vs-K20 shape;
   report explicitly WHICH of the two sub-cases obtains when this outcome
   is read, since they are not the same finding: the first is a power
   problem, the second is a real-but-differently-shaped effect). No
   equivalence claim is licensed in either sub-case; both remain a
   DISTINCT finding from CONVERGED-EQUIVALENT that this revision refuses
   to silently fold into it.
6. **NON-MONOTONE (new outcome, Rev 3 fix, attack-round-3 MAJOR-R3-2 —
   closes the exhaustiveness gap Rev 2's five-outcome pentachotomy still
   had).** The final, catch-all branch: reached iff none of outcomes 1-5
   above fired. By construction (verified by exhaustive enumeration over
   the `2^5=32` possible `holds(c)` truth-patterns across
   `{250,500,1000,2500,5000}`, worked table below), this branch is
   reached exactly when `holds(c)=TRUE` for AT LEAST ONE `c` (ruling out
   #4/#5, which both require `holds(c)=FALSE` everywhere) AND the
   TRUE/FALSE pattern is not a monotone run of TRUEs ending at 5,000 that
   starts at a non-terminal checkpoint (ruling out #1 PERSISTENT), AND
   `holds(5000)` is not FALSE (ruling out #2 TRANSIENT, which already
   claims every `holds(5000)=FALSE` case with ≥1 earlier TRUE), AND
   `holds(5000)=TRUE` is not the ONLY true checkpoint (ruling out #3
   LATE-EMERGENT). Concretely: any trajectory where `holds` flickers —
   TRUE, then FALSE, then TRUE again, in any arrangement not already
   claimed above (the exact adversarial pattern this finding named,
   `holds(250)=T, holds(500..2500)=F, holds(5000)=T`, is the canonical
   member of this bucket). **Pinned reading: inconclusive-without-rerun.**
   This is NOT a durable finding of any kind (not PERSISTENT-strength,
   not a negative), and is NEVER averaged away into a headline number
   across seeds — every NON-MONOTONE trajectory triggers a mandatory
   seed-level disclosure (report that individual seed's own `holds(c)`
   pattern verbatim in whatever table/figure reports that cell) rather
   than being folded into a percentage or a mean effect size alongside
   seeds that DID resolve to one of outcomes 1-5. A cell whose 3 seeds
   classify `{PERSISTENT, NON-MONOTONE, TRANSIENT}` reports all three
   labels individually, never a single blended verdict.

   **Rev-5 build-audit note (FIX agent, Phase-2 build-audit round,
   2026-07-07) — reconciles the "3 seeds classify... reports all three
   labels individually" sentence above with this section's own buildable
   classification granularity, flagged by the independent build audit
   (MINOR-3).** `holds(c)` and `det(K,c)` are ALREADY defined, earlier in
   this same section (Mechanical primitives paragraph), as CI-based
   quantities computed from `Δ(K)`'s pinned **3-seed-pooled** CI at
   checkpoint `c` — not per-seed values. There is therefore exactly ONE
   `holds(c)` pattern, and exactly ONE classification, per (corpus) —
   never three independent per-seed classifications to individually label.
   `phase2_trajectory_analysis.py`'s own build-time scoping decision (its
   module docstring, disclosed explicitly for this audit's attention)
   makes this literal: ONE trajectory PER CORPUS, built from a
   3-seed-pooled `delta_ci_n3` CI at every checkpoint, for both non-off
   arms. **The registered outcome per (corpus) cell is this pooled
   classification.** The "per-seed" language above is retained as a
   DISCLOSURE requirement, not a second, competing classification
   granularity: for any pooled trajectory that is NOT cleanly
   PERSISTENT/TRANSIENT/LATE-EMERGENT (i.e. NON-MONOTONE or
   UNRESOLVED-GATE), report each seed's own observed
   recovered_frac(h=1)-vs-band pattern as a disclosure line alongside the
   pooled verdict — computed FROM the per-seed holds-relevant readings
   that already feed the pooled CI, never as a stand-alone classification
   that could disagree with it.

**Totality, checked exhaustively, not merely asserted (Rev 3 fix,
attack-round-3 MAJOR-R3-2; count corrected, Rev 4 fix, attack-round-4
MAJOR-R4-2 — see §16.12).** Every one of the `2^5=32` possible
`holds(c)` truth-assignments across `{250,500,1000,2500,5000}` maps to
exactly one of the six buckets above, by direct enumeration: the single
all-FALSE pattern routes to #4/#5 (split by the terminal `det_arm`/
`agree` reads) — note this pattern also has `holds(5000)=FALSE`, so of
the 16 total patterns with `holds(5000)=FALSE`, this one is already
claimed by #4/#5; of the remaining 31 patterns with ≥1 TRUE, the other 15
with `holds(5000)=FALSE` are ALL claimed by #2 TRANSIENT (any ≥1 TRUE
among the first four checkpoints, `holds(5000)=FALSE`, by definition —
no further split needed); of the 16 patterns with `holds(5000)=TRUE`, 1
is the all-false-before-terminal pattern (#3 LATE-EMERGENT), 4 are the
monotone-true-run-ending-at-5000 patterns for each possible non-terminal
start (`TTTTT`, `FTTTT`, `FFTTT`, `FFFTT` — #1 PERSISTENT), and the
remaining `16-1-4=11` are flicker shapes claimed by #6 NON-MONOTONE.
`1+15+1+4+11=32` — exhaustive, with no residual case. Representative
rows, including the exact adversarial pattern this finding named:

| Pattern (250,500,1000,2500,5000) | Outcome | Why |
|---|---|---|
| T,F,F,F,T | **NON-MONOTONE** | The exact attack-round-3 counter-example: fails PERSISTENT (not monotone-true to 5,000), TRANSIENT (`holds(5000)=TRUE`), LATE-EMERGENT (`holds(250)=TRUE`), CONVERGED-EQUIVALENT/UNRESOLVED (not FALSE at every checkpoint) — routes to the new sixth bucket |
| T,T,F,F,T | NON-MONOTONE | Same shape: an early TRUE run breaks, then re-emerges at the terminal checkpoint |
| T,T,T,T,T | PERSISTENT | `c1=250` (first TRUE, non-terminal), TRUE through 5,000 — subject to the Stage-0.5-at-`c1` re-check, MAJOR-R3-3 |
| F,T,T,T,T | PERSISTENT | `c1=500` |
| F,F,T,T,T | PERSISTENT | `c1=1000` |
| F,F,F,T,T | PERSISTENT | `c1=2500` |
| F,F,F,F,T | LATE-EMERGENT | Only the terminal checkpoint ever holds |
| T,F,F,F,F | TRANSIENT | `holds(250)=TRUE`, `holds(5000)=FALSE` |
| F,T,F,T,F | TRANSIENT | ≥1 TRUE among the first four, `holds(5000)=FALSE` — flicker BEFORE 5,000 is still TRANSIENT, not NON-MONOTONE (NON-MONOTONE is reserved for terminal-TRUE flicker shapes, since a FALSE terminal checkpoint is already fully claimed by TRANSIENT) |
| F,F,F,F,F (`det_arm(global,5000)=TRUE`, `det_arm(per_token,5000)=TRUE`, `agree(5000)=TRUE`) | CONVERGED-EQUIVALENT | Never holds; both arms individually show a determinate, mutually-indistinguishable training effect |
| F,F,F,F,F (`det_arm(per_token,5000)=FALSE`) | UNRESOLVED | Never holds; per_token's own effect vs. off never clears noise — MINOR-R3-1: no longer silently read as equivalence just because global's own CI is determinate |

**CONVERGED-EQUIVALENT (and UNRESOLVED) are named outcomes distinct from
Cell 4 (Rev 1 fix, MAJOR-3, carried forward unchanged in substance,
re-pointed from Rev 1's own "CONVERGE" label per MAJOR-NEW-6's split
above) — registered explicitly because the two are easy to conflate and
are NOT the same finding.** Cell 4 (§12) is "geometry stabilization is
real but functionally inert for in-context composition," diagnosed on
checkpoints that never trained on the task at all (Phase-1's zero-shot
instrument) — its natural reading is "the stabilized geometry doesn't
matter because the model was never asked to use it." CONVERGED-EQUIVALENT
is diagnosed on checkpoints that DID train on the task and DID learn it
(h∈{1,2} clear their floor, AND a real, determinate training effect
exists) — its reading is "the stabilized geometry doesn't matter even
once the model is actually asked to use it, and given every opportunity
via gradient descent to exploit whatever advantage it might confer."
Task-trained equivalence is a strictly stronger, more informative
negative than zero-shot inertness, and — per the same "publication value:
high, as an honest, hard-won negative" logic §12 already applies to Cell
4 — independently publishable on its own terms, not merely as a footnote
to Cell 4. **UNRESOLVED is a categorically different epistemic status
from both:** it is neither Cell 4's "inert" nor CONVERGED-EQUIVALENT's
"equivalent" — it is "insufficiently powered to say either," and is
reported as an open measurement question (recommending more seeds or a
longer budget), never dressed up as a negative finding of any kind.
**NON-MONOTONE (Rev 3 fix, MAJOR-R3-2) is a THIRD, equally distinct
epistemic status, not a variant of UNRESOLVED:** UNRESOLVED means the
best-powered (terminal) reading itself never clears noise; NON-MONOTONE
means the trajectory DID clear the `holds` bar at least once but in a
shape none of the three positive/negative-trajectory outcomes
(PERSISTENT/TRANSIENT/LATE-EMERGENT) can honestly claim — a genuinely
unclassifiable signal shape, not a power problem, and per its own pinned
reading above it is disclosed per-seed, never blended into a summary
statistic alongside cleanly-classified seeds.

**Stage-0.5-familiarized gate (Rev 1 fix, MAJOR-4; now PER-CHECKPOINT,
Rev 3 fix, attack-round-3 MAJOR-R3-3) — a new gate, distinct from and in
addition to Phase-1's own Stage-0/Stage −1 gates.** Before any arm
contrast AT ANY trajectory checkpoint above is trusted, premises
(iii)/(iv) and the h1 sanity floor (§8.4, the same instrument-validity
checks Phase-1's own Stage 0 registers) are RE-MEASURED on the
familiarized OFF-arm checkpoint AT THAT SAME CHECKPOINT — not once at the
end of the 5,000-step budget and treated as covering all 5 trajectory
checkpoints retroactively. **This was the actual gap (Rev 1's own text
read as one-shot):** the val-loss tolerance band immediately below
(MAJOR-6/MINOR-NEW-1) was ALREADY pinned per-checkpoint; this gate must
match that same granularity, or a checkpoint could clear its own
val-loss band while its Stage-0.5 premises/h1-floor reading is stale from
step 5,000 (or never measured at all at that checkpoint) — exactly the
kind of mismatched-granularity gate this project's own gates-must-abort
discipline exists to prevent. Registered fix: the OFF arm's own premises
(iii)/(iv) and h1 floor are measured FRESH at EVERY ONE of the 5
checkpoints `{250,500,1000,2500,5000}`, on that checkpoint's own OFF-arm
weights — reusing the SAME `measure_cell_all_h`(`compute_premises=True`)
computation §4.4 already runs at every trajectory-readout scoring pass
(`reasoning_link_probe.py`'s `run_cell`, which always sets
`compute_premises=True`), so the premises/h1-floor VALUES this gate needs
already exist as a byproduct of the scoring pass every arm's own
trajectory-readout computes at every checkpoint — this is a
re-interpretation of an already-scored per-checkpoint field as a
go/no-go gate, priced separately below only for the piece that is
genuinely new (the null-shuffle band). The mandatory go/no-go is now
per-checkpoint: if the OFF arm's own premises (iii)/(iv) or h1 floor fail
their action-rule gates (§4.4) AT CHECKPOINT `c`, checkpoint `c`'s own
arm-contrast (all 3 arms, that corpus/seed) is uninterpretable and MUST
be excluded from any reading at `c` — per §16.5 Constraint 1's own
registered hard requirement (gates-must-abort; every chain script under
this section needs an explicit abort branch, verified by a deliberate
negative test, not merely a computed-but-unread gate value, per the
paired gates-must-abort `[LEARN]` `EXPERIMENT_LOG.md`'s Phase-1 entry
already names) — applied per-checkpoint here, not once at the end of the
run.

**Rev-5 build-audit note (K coupling, MINOR-3 adjudication (ii)).** This
gate's own eval episodes run at Q=K (build-audit MAJOR-2 fix, matching
`reasoning_link_probe.episode_config_for_checkpoint`'s own Q=K eval
convention) — but `K` itself is, by construction, whatever `K` that
checkpoint's own familiarization TRAINING used to build its episodes
(`phase2_familiarization_train.K_TRAIN_DEFAULT=32`), never independently
swept over the registered `K∈{20,32}` READOUT pair the killer-prediction
contrast applies post-hoc. The Stage-0.5 gate is therefore always a
K=32-only instrument, not a second K-sweep.

**Added cost, priced (Rev 3 fix, attack-round-3 MAJOR-R3-3's own
instruction: price it, don't wave it as "small").** The premises
(iii)/(iv) and h1-floor VALUES ride for free on the already-priced
per-checkpoint trajectory-readout scoring pass (`compute_premises=True`
is `measure_cell_all_h`'s own default in `run_cell`, unchanged marginal
cost). What IS genuinely new is the null-shuffle label-permutation band
each premise's own action-rule gate needs to compare against (§4.4's
Stage-0 convention) — verified directly: `run_cell`'s regular scoring
pass does NOT compute this null band (`measure_cell_all_h`'s own
`null_seed: int | None = None` parameter is left at its default by
`run_cell`), only `run_stage0`/`run_stage0_natural` do. Registered scope
(unchanged from MAJOR-4): the gate is measured ONLY on the OFF arm's own
6 cells (2 corpora × 3 seeds — its per-checkpoint PASS/FAIL result then
gates all 3 arms sharing that corpus/seed, not 18 separate measurements)
× 5 checkpoints = **30 gate-with-null-band passes**. Priced from this
program's own REALIZED rate for exactly this class of computation:
Phase-1b's 4 stage0-natural gate cells (§16.8.5 — causality assertion +
per-h null-shuffle bands + premises/h1-floor action-rule gates, the SAME
shape of work) cost 31.85s GPU-attached total ≈ 0.0088 GPU-h, i.e.
≈0.0022 GPU-h/cell. `30 × 0.0022 ≈` **0.066 GPU-h** — a small, disclosed
ADDITIVE addition to §16.2.3's existing "negligible eval passes" line,
named explicitly rather than folded silently into that line's existing
"negligible" language. **Unlike MAJOR-R3-1b's own query-loss multiplier
above (a MULTIPLICATIVE effect the existing 5-10× margin already had room
for), this is a genuinely NEW, additive raw-cost line the original
bracket's own arithmetic (`5×0.23=1.15`, `10×1.14=11.4`, both exact, zero
slack) has no room to silently absorb** — §16.2.3 (Cost) folds this
0.066 GPU-h into the raw total BEFORE re-applying the 5-10× multiplier,
which is what widens the final committed bracket; the two findings'
effects on the bracket are NOT the same shape and are not conflated here.

**This gate also carries real instrument-validation value, disclosed here
rather than treated as pure overhead:** if the readout construct
(`S_T^h · q_eff`, §4.4) is actually measuring what this design claims,
premises (iii)/(iv) SHOULD pass on a checkpoint that has now genuinely
trained on the bind/query structure — familiarization gives the readout
every reason to succeed that Phase-1's zero-shot instrument lacked. A
PERSISTENT failure of premises (iii)/(iv) even after familiarization
indicts the readout construct itself far more decisively than Phase-1's
own zero-shot PROBE-INVALID verdict (§15) could, since that verdict could
always be attributed to "never task-trained" rather than to the
instrument. Phase-2's Stage-0.5 gate is therefore doing double duty: a
go/no-go for the arm contrast, and the most stringent test this program
has yet run of whether the readout itself is sound.

**Familiarization-mix val-loss tolerance band, blind-pinned before launch
(Rev 1 fix, MAJOR-6).** §7.2's existing val-loss-matching gate was
established on the ORIGINAL pretraining objective/corpus mix (§16.2.2
confound (a)) and has no referent on the new familiarization mix.
Registered fix: a FRESH tolerance band, `mean_ref ± 2·s_ref` (this
codebase's own house `k=2` convention, `KEY_ANCHORING_DESIGN.md` §3.6 /
`key_anchoring.derive_engaged_bands`, already reused once for exactly
this purpose by `FROZEN_BIAS_LM_DESIGN.md` §7.2's own val-loss tolerance
derivation), computed from the OFF arm's own per-seed loss ON THE NEW MIX
— and this band must be computed and committed to a `BANDS_PINNED`-style
JSON **BEFORE** the global/per_token familiarization launches, not after.
**This sequencing requirement is a direct, named guard against a real,
disclosed process failure this exact program already committed once:**
`FROZEN_BIAS_LM_DESIGN.md`'s own rung-1 wave wrote its
`BANDS_PINNED-FrozenBias.json` AFTER all 20 training cells had already
completed (`pinned_at_iso` postdates every training result on disk),
which — despite the CI arithmetic itself being numerically real and
independently re-verified — forfeited the blind's confirmatory license
and demoted the entire wave to the DESCRIPTIVE TIER; that document's own
"process finding for future waves" names "pin before launch" as a hard
sequencing rule for exactly this reason. Phase-2 registers that rule as a
build requirement here, not as a lesson to remember later.

**Gate/blind-pin granularity, pinned explicitly (Rev 2 fix,
attack-round-2 MINOR-NEW-1) — MAJOR-6's fix above under-specified WHICH
checkpoint(s) the tolerance band covers, and WHEN relative to launch,
across a now-5-checkpoint trajectory.** Registered sequencing: the OFF
arm (2 corpora × 3 seeds = 6 cells) launches ALONE first, run to the full
5,000-step familiarization budget standalone, before either intervention
arm's own familiarization run starts. From that standalone OFF-arm run,
**pin FIVE separate tolerance bands, one per trajectory checkpoint
step** — `mean_ref ± 2·s_ref` computed independently at each of
`{250,500,1000,2500,5000}` from the OFF arm's own per-seed val-loss AT
THAT STEP (same house `k=2` convention, unchanged formula, applied 5×
instead of once) — and commit all 5 to the `BANDS_PINNED`-style JSON
BEFORE the `per_token`/`global` familiarization runs launch. **Every
checkpoint reading is then gated against its OWN band, not just the
terminal one**: if a `per_token` or `global` arm's val-loss at (say) step
1,000 falls outside that step's own pinned band, THAT checkpoint's
arm-contrast is flagged uninterpretable even if the terminal (5,000-step)
reading later falls back inside its own band — a per-checkpoint gate, not
a single end-of-run gate applied retroactively to the whole trajectory.
**Scheduling note, disclosed explicitly so it is not mistaken for a cost
change:** this sequencing (OFF arm fully first, then the other two arms)
serializes what could otherwise be an 18-cell simultaneous launch into
two phases (6 cells, then 12) — this changes WALL-CLOCK scheduling only.
**Total committed GPU-h is UNCHANGED** (§16.2.3's bracket prices 18
cells' worth of compute regardless of launch order; sequencing does not
add or remove a single training step).

**Disclosure (Rev 3 fix, attack-round-3 MINOR-R3-4) — the val-loss band
has ZERO discriminating power for the OFF arm's own cells, by
construction.** Each of the 5 pinned bands (`mean_ref ± 2·s_ref`) is
computed FROM the OFF arm's own per-seed val-loss at that checkpoint —
the OFF arm's own 6 cells are therefore, barring a same-corpus
cross-seed outlier beyond the band's own 2-standard-deviation width, near
tautologically inside their own band. "OFF passed its own val-loss gate"
is never evidence of anything and must never be cited as if it
corroborated the gate's real target, which is whether `per_token`'s or
`global`'s own val-loss — an INDEPENDENT quantity, never used to
construct the band — falls inside a band built without reference to
either of them. This one-sentence disclosure exists so a future reader
cannot mistake "the OFF arm cleared its own band" for a real check.

#### 16.2.2 The new confound surface — why this needs its own attack round before build

**Training-on-task may itself differentially interact with the arms — and
that differential interaction IS the hypothesis, which is exactly why it
needs careful handling, not exclusion.** H_LINK-A's causal claim (global
arm learns to compose at least as well as off, per-token no better) is
easy to state but the val-loss-matching gate that currently makes the
arms comparable (§7.2) was established on the ORIGINAL pretraining
objective, at the ORIGINAL corpus mix — not on a mix that now includes
BIND/QUERY episodes. **The attack round this design owes before build must
resolve, at minimum:** (a) whether val-loss (or an equivalent matching
criterion) needs to be RE-established on the new mix before any
arm-contrast is trusted, since a differential in how fast each arm's loss
converges under the NEW data could itself be the causal story, confounding
"stabilized geometry helps learning" with "stabilized geometry changes
optimization dynamics on this specific new objective" (the same rival
account Okpekpe & Orvieto raise for MQAR, §2.1, re-appearing here in a
training-dynamics form); (b) whether the familiarization fraction is small
enough that this remains "continue training an LM that also sees some
task examples" rather than "train a new task-specific model with LM
pretraining as an initialization" — a framing distinction that changes
which literature this design should be positioned against; (c) whether
holding the frozen-bias table's λ fixed during familiarization (rather
than letting it also adapt) is the right choice, or whether a fixed λ
under a NEW data distribution silently becomes a worse blend than what
would have been tuned for this objective, artificially handicapping the
intervention arms relative to `off`.

**Rev 1 disposition of the three items above (attack-round, this pass;
full fix-map §16.7).** Item (a) — whether val-loss needs to be
RE-established on the new mix — is now **RESOLVED BY CONSTRUCTION**: the
Stage-0.5-familiarized gate (§16.2.1, MAJOR-4) plus the blind-pinned
val-loss tolerance band on the new mix (§16.2.1, MAJOR-6) together give
this a concrete, pre-registered mechanism, closing the open question
rather than merely re-stating it. Item (b) — whether the familiarization
fraction stays small enough to remain "continue training with some task
exposure" rather than "train a task-specific model" — is **partially
addressed**: MINOR-1 (§16.2.1) commits to measuring rather than guessing
the fraction, but the pilot measurement itself has not yet run, so this
item stays open pending that result. **Item (c) — whether holding λ
fixed during familiarization is the right choice, or silently handicaps
the intervention arms under a new data distribution — is NOT addressed
by this attack round and remains genuinely open,** disclosed here rather
than hidden: no finding in §16.7 speaks to it, and it should be the first
item a future attack round on this recipe takes up.

#### 16.2.3 Cost (Rev 1 — re-derived, attack-round MAJOR-5; supersedes the prior ungrounded "~5-15 GPU-h" placeholder. Rev 2 CONFIRMED the bracket below unchanged — see §16.9's own BUDGET paragraph for the disclosed-but-margin-covered per-step increase from MAJOR-NEW-1's query-loss forward pass. Rev 3 RE-DERIVES the bracket after TWO further audit-forced findings — see §16.10's own BUDGET paragraph: MAJOR-R3-1b corrects an undisclosed Q=1 assumption in Rev 2's own query-loss multiplier [honest figure ~1.90× worst-case, still inside margin, contributes NO widening on its own] and MAJOR-R3-3 prices the now-per-checkpoint Stage-0.5 gate's added null-band cost explicitly [≈0.066 GPU-h, NEW] — the bracket WIDENS to ≈1.48-12.06 GPU-h (from ≈1.15-11.4), driven entirely by MAJOR-R3-3's new cost line, disclosed below rather than silently re-asserted)

**Scope clarification, stated up front:** Phase-2 operates on **14M-class
(rung-1) checkpoints ONLY.** No 98M-class (rung-2) frozen-bias checkpoints
exist anywhere in this program's archive — rung-2 was formally PARKED
before any rung-2 cell ever launched (`FROZEN_BIAS_LM_DESIGN.md`
§6.2/§8.1; the VERDICT section confirms "all 20 mandatory training cells
ran to completion" at rung-1, `d_model=256, n_layers=2, d_state=64`,
`n_params=14,048,896` — zero rung-2 cells exist to continue-train from).
The prior placeholder's own text ("18 checkpoints × a modest step budget
× forward+backward at 14M-98M-class model sizes") was ambiguous enough to
read as pricing in a rung-2 leg that does not exist; this revision removes
that ambiguity.

**Raw compute-only cost, re-derived from this program's own realized rate
(not the task's placeholder estimate):** `FROZEN_BIAS_LM_DESIGN.md`'s own
rung-1 wave measured **18,175.744s summed `wall_s` across 20 training
cells × 20,000 steps = 0.04544 s/step** (18175.744 / 400,000) — a tight,
real, cross-checked realized rate (899-914s per cell, no crashes or
retries, per that document's own VERDICT section). Applying that rate to
Phase-2's own **18 core cells** (3 arms × 2 corpora × 3 seeds,
unchanged from §16.2.1's Base-checkpoints paragraph) at the registered
**1,000-5,000 step familiarization budget** (§16.2.1's trajectory
schedule, MAJOR-1 — 5,000 is now the fixed upper end, not an open-ended
"order of" estimate):

- 18 cells × 1,000 steps × 0.04544 s/step = 817.9s = **0.23 GPU-h** (lower bound)
- 18 cells × 5,000 steps × 0.04544 s/step = 4,089.6s = **1.14 GPU-h** (upper bound)

**Plus negligible eval passes:** the trajectory-readout forward passes (5
checkpoints × 18 cells × 2 K's × Option 1/Option 2, all short
forward-only jobs, no backward pass) are the same class of cost
`FROZEN_BIAS_LM_DESIGN.md`'s own 46 retrofit re-evals registered at ≈1.6
GPU-h TOTAL for a comparable-or-larger pass count — priced here as
negligible relative to the training-cell total above, not zero, but not
separately budgeted pending an exact pass count at build time. **This "5
checkpoints" count is exactly what §16.2.1's `--ckpt-steps` build task
(Rev 2, MAJOR-NEW-2) protects** — the forbidden `--ckpt-every 250`
workaround would silently inflate this to 20 checkpoints (4× the eval-pass
count this line prices), which is why that workaround is explicitly
forbidden there rather than merely discouraged.

**Plus the Stage-0.5 gate's own added null-band cost (Rev 3 fix,
attack-round-3 MAJOR-R3-3, §16.2.1) — priced, not folded into
"negligible" above:** 30 gate-with-null-band passes (OFF arm's 6 cells ×
5 checkpoints) at the realized ≈0.0022 GPU-h/cell rate §16.8.5's own
Phase-1b gate cells established ≈ **0.066 GPU-h**, a small, disclosed
addition on top of the negligible-eval-passes line above.

**Raw total: ≈0.23-1.14 GPU-h** (training) **+ ≈0.066 GPU-h** (Stage-0.5
gate null-bands) **≈ 0.30-1.21 GPU-h.** This is an order of magnitude below the
task's own placeholder bracket (~5-15 GPU-h) — the placeholder was not
merely imprecise, it was ungrounded (no derivation, no cited realized
rate). **If a larger registered bracket is retained for build/debug
margin, the multiplier must be disclosed, not silently absorbed into a
bigger round number** — this program has an exact, in-repo precedent for
how much margin a "trivial-looking" change actually needs: Path (i)'s own
Stage-0-gate-only estimate (§16.1.2) registered **~0.01 GPU-h against a
real measured-rate cost of ≈0.001-0.002 GPU-h — a 5-10× debug-tax
multiplier**, disclosed explicitly rather than picked by feel.

**Bracket re-derived and WIDENED, disclosed prominently (Rev 3 fix,
attack-round-3 MAJOR-R3-3's own added cost line; audit-forced, not a
silent self-amendment, per this revision's own instruction).** Applying
the SAME 5-10× multiplier to Phase-2's own raw **combined** total
(training + the Stage-0.5 gate's own null-band cost, `≈0.30-1.21 GPU-h`,
not the training-only `0.23-1.14 GPU-h` Rev 1/Rev 2 priced) gives
`0.296×5≈1.48` and `1.206×10≈12.06` — a registered, debug-tax-inclusive
bracket of **≈1.48-12.06 GPU-h**, REPLACING the previously-committed
**≈1.15-11.4 GPU-h** bracket (Rev 1/Rev 2). **Audit trail:** this widening
is driven entirely by MAJOR-R3-3's own newly-priced Stage-0.5
per-checkpoint gate cost (≈0.066 GPU-h raw, new this revision) —
MAJOR-R3-1b's own re-derived query-loss multiplier (§16.2.1,
`n_query=2`, ~1.90× worst-case) stays inside the existing margin on its
own and contributes nothing to this widening; the two findings' effects
on the bracket are independent and are shown separately here rather than
merged into one unexplained delta.

**Sanity context, re-stated at the new figures:** even at the
debug-tax-inclusive upper bound (≈12.06 GPU-h), this remains a small
fraction of Phase-1's own registered ≈24.20 GPU-h ceiling (§10) and of
Path (iii)'s ≈21 GPU-h mandatory grid (§16.3), while comfortably
exceeding Path (i)'s eval-only ~0.4 GPU-h full grid (§16.1.2) — expected,
since Phase-2 is continued TRAINING (forward+backward, 18 cells) rather
than a forward-only probe. The raw lower bound (≈0.30 GPU-h) is now
cheaper than Path (i)'s own full grid, a fact the old "~5-15 GPU-h"
placeholder obscured entirely.

#### 16.2.4 Gate before build (Rev 5 — FIVE attack rounds complete, verdicts recorded)

Per this project's own waterfall discipline (`CLAUDE.md`), Path (ii)
needed a dedicated attack round (fresh-eyes agent, minimum the standing
5-6 questions, addressing at minimum §16.2.2's three confound items)
BEFORE any build — registered in the prior revision as a hard
prerequisite, not a suggestion. **Four independent attack rounds have
now run, by four different reviewers.** Round 1 verdict: NEEDS-REDESIGN
— 1 FATAL, 6 MAJOR, 1 MINOR, all fixed in Rev 1 (§16.7). Round 2 verdict:
NEEDS-REVISION — 6 new MAJOR, 2 new MINOR, no FATAL (every Rev 1 fix
independently re-verified and HOLDS — round 2 found code-reality/
buildability gaps in Rev 1's OWN fixes, not design-logic errors), all
fixed in Rev 2 (§16.2.1-§16.2.3 as they read at that revision); the full
finding→fix trace is recorded in §16.9, per house style, mirroring
§16.7's own convention for round 1. Round 3 verdict: **NEEDS-REVISION** —
3 new MAJOR, 4 new MINOR, no FATAL (every Rev 1 AND Rev 2 fix
independently re-verified and HOLDS — round 3 found a FALSE code citation
inside Rev 2's own query-loss mechanism, an undisclosed cost-arithmetic
assumption, a MECE gap in the pentachotomy, and buildability/precision
gaps, not a re-opening of any prior finding), all fixed in Rev 3
(§16.2.1-§16.2.3 as they read at that revision); the full finding→fix
trace is recorded in §16.10, per house style, mirroring §16.7's and
§16.9's own convention. Round 4 verdict: **NEEDS-REVISION** — 3 new
MAJOR, 0 new MINOR, no FATAL (every Rev 1, Rev 2, AND Rev 3 fix
independently re-verified and HOLDS — round 4 found three
internal-consistency bugs inside Rev 3's OWN fixes: a `c1`-selection
rule anchored on `det(32)` instead of `holds` that could silently
misroute a real PERSISTENT trajectory to NON-MONOTONE, an arithmetic
error in the totality narration's own claimed sum, and a false field
citation — `val_loss` lives in `result["checkpoints"]`, not
`result["trajectory"]` — the same false-citation failure mode round 3
itself flagged as round 2's worst sin, reproduced inside round 3's own
fix text), all fixed in this revision, Rev 4 (§16.2.1 above); the full
finding→fix trace is recorded in §16.12, mirroring the prior rounds'
convention.

**What this gate does and does not license.** Per this project's own
standing rule that a self-audit is not a substitute for an independent
audit, and that multiple independent adversarial audit rounds catch
different bugs each round (`CLAUDE.md`, verified on Task E: round 1
caught 2 FATALs a self-audit missed, and the fix itself was audited fresh
in round 2 before being trusted — and now verified AGAIN on this exact
Phase-2 recipe across FOUR rounds: round 2 caught 6 new MAJORs round 1's
own reviewer and this document's own self-audit both missed, round 3
caught a FALSE code citation that round 2's OWN reviewer — despite that
round's stated purpose being a source-reading buildability check — still
missed, and round 4 caught the same false-citation failure mode
reproduced inside round 3's own fix text) — a FIFTH, fresh adversarial
pass targeting Rev 4 specifically HAS NOW ALSO RUN (2026-07-07, fifth
distinct reviewer). Round 5 verdict: **NEEDS-REVISION** — 0 FATAL, 2
MAJOR, 1 MINOR, ALL prose/status-sync defects in the header-bump
commit's own wake (this §16.2.4 body and §16.6 item 4 still narrated
the pre-Rev-4 state their own headers had moved past; the §16.2
preamble's status-note chain stopped at Rev 3), with **every
substantive check PASSING**: all three Rev-4 fixes verified landed
byte-accurate against the real source (line citations re-executed), an
independent from-scratch 32-pattern re-enumeration under the
holds-anchored `c1` reproduced 1+15+1+4+11=32 exactly and confirmed the
round-4 counter-example now classifies PERSISTENT/c1=500, and the
diff-scoped new-bug sweep found nothing. Those prose findings are fixed
in this revision, Rev 5 (a status-sync-only patch — recipe content,
build tasks, cost bracket, and all algorithms UNCHANGED from Rev 4);
trace in §16.13. Per the round-5 reviewer's own scoping, the remaining
step before build is that reviewer's confirmation pass on the corrected
status prose ONLY — the substantive checks (1, 2, 4) carry forward
without re-verification. Only after that confirmation (or a further
revision addressing its findings) should Phase-2 build, per §16.6's own
sequencing. **Disclosed, not treated as a stopping rule:**
findings-per-round have not yet decayed to
zero (round 1: 1 FATAL+6 MAJOR+1 MINOR; round 2: 0 FATAL+6 MAJOR+2 MINOR;
round 3: 0 FATAL+3 MAJOR+4 MINOR; round 4: 0 FATAL+3 MAJOR+0 MINOR,
all internal-consistency, none design-logic; round 5: 0 FATAL+2 MAJOR+1
MINOR, all prose/status-sync, zero design impact) — this project's `CLAUDE.md` rule
requires running the next independent round regardless of how the count
trends; a zero-finding round, not a declining count, is what licenses
build. **A specific, disclosed reason this round's own finding count
should not be over-read as "converging":** round 3's own highest-value
finding (MAJOR-R3-1a) was a FALSE CITATION, not a design gap — a
qualitatively different failure mode than rounds 1-2's own design/
buildability gaps, and a reminder that "findings per round trending down"
is not the same claim as "the remaining findings are getting less
severe" or "the next round will find nothing."

**CLEARED-FOR-BUILD (round-5 reviewer's own confirmation pass, 2026-07-07).**
The remaining gate this section's own text above registers — "the
reviewer's confirmation pass on the corrected status prose ONLY" — has now
run, on the second Rev-5 status patch (§16.13's own addendum: five residual
status-sync defects found and fixed, none substantive, same "prose only,
zero design impact" shape as every Rev-5 finding). That confirmation pass
PASSED: the five edits landed as prescribed, substantive checks 1/2/4 carry
forward unchanged (never re-opened), and no new finding was raised. Per
this section's own registered sequencing ("Only after that confirmation...
should Phase-2 build"), the gate is discharged: **Phase-2 §16.2.1 (Rev 5)
is CLEARED-FOR-BUILD.** The build ran the same day — folded in here by the
BUILD agent itself, per the round-5 reviewer's own explicit assignment of
this recording task to the build commit: files added under
`matrix-thinking/deltanet_rd/` (`phase2_familiarization_train.py`,
`phase2_gate_enforce.py`, `phase2_bands_pinned.py`, `phase2_hexachotomy.py`,
`phase2_trajectory_analysis.py`, `phase2_stage_minus1.py`,
`phase2_chain.sh`), `lm_pretrain_rd.py` extended with `--ckpt-steps`/
`--init-checkpoint` (sec 16.2.1's own two registered build tasks), and
`reasoning_link_probe.py`'s `run_cell` extended with an additive
`seed_override` parameter (sec 16.2.1's own killer-prediction-reapplication
reuse instruction). Every Stage −1 self-test this build registers (11 items,
`phase2_stage_minus1.py`, PLUS Phase-1's own unchanged 19-item suite,
re-run as a dependency-closure check) ran to completion, on CPU, via this
repo's own fla stub — build commit `1f53a68d592fd6dcc676e8dd2613793524ff1968`
(backfilled here in a small same-day follow-up commit, per this document's
own established convention of citing a commit hash only in the FIRST
subsequent revision that references it — this paragraph could not self-cite
its own hash from inside the build commit itself). This build did not
itself launch the 18-cell grid (LOCAL BUILD ONLY, per its own task scope)
— that launch, and this build's own independent audit, remain separate,
later steps.

---

### 16.3 PATH (iii) — un-park the scaling wave (`KEY_ANCHORING_SCALING_DRAFT.md` §15)

**What it tests.** `x0(d)` invariance — whether the located d=64 capacity
cliff (`x0=0.5455`, CI `[0.5385,0.5513]`) holds as a fixed `K/d_state`
ratio at `d∈{80,96}`, extending the already-closed d=128 dissolution
result (§4/`KEY_ANCHORING_DESIGN.md` §12) one step further along the
capacity-law's own generalization axis. **This is orthogonal to
REASONING-LINK entirely** — it runs on the proven `run_deltanet_rd_
exactness_sweep.py` harness, on models TRAINED on the exact BIND/QUERY
task (the capacity-law lineage's own convention throughout), and answers
a capacity-law-paper-track question, not the keystone question this
document's §2 opens with.

**Prerequisite build gaps — both already identified, both zero-GPU,
neither blocking the other paths (`KEY_ANCHORING_SCALING_DRAFT.md`
§15.18):**
- **FATAL-1:** `keyanchor_dstate_manifest()`/`_gate1_manifest()`/
  `_calibration_manifest()` hardcode seed dicts keyed to
  `K∈{68,76,84,92}` (`run_deltanet_rd_exactness_sweep.py:1888-1895`) —
  calling at the new K grids raises `KeyError` immediately. Fix: thread
  `seeds_by_k`/`gate1_seed_by_k` parameters through, or build fresh
  `keyanchor_scaling_*` manifests.
- **MAJOR-3:** `smoke_key_anchoring.py` has no `--d-state` flag and
  hardcodes `d∈{64,128}`. Fix: a new `smoke_keyanchor_scaling.py`, per the
  `smoke_keyanchor_cliff.py`/`smoke_keyanchor_dstate.py` naming
  convention already established.
- **FATAL-2 (kernel-safety gate) is ALREADY RESOLVED, not outstanding.**
  The re-run at the full registered `T∈{128,224,448}` protocol confirms
  d=80/96 PASS forward+backward with finite grads at every T (artifact
  `deltanet_rd/results/smoke_dstate_kernel_result.json`, committed); d=32
  reproduces its historical crash exactly at T=128, confirming the
  original T=256-only smoke was a false negative. Nothing further is
  needed here before launch.

Both remaining gaps are ordinary implementation work — per `CLAUDE.md`'s
Build→Audit workflow, they still need a build pass AND a separate-agent
audit before launch, not a "just patch it and go" shortcut (§16.6 makes
this an explicit sequencing step, not an assumption).

**Cost.** §15.12's own budget table: **≈21 GPU-h (20.956) mandatory-only,
at the pessimistic 2× contingency bracket** — 30 cells (12 d=80 grid + 12
d=96 grid + 6 mandatory low-K anchor cells), the load-bearing go/no-go
number. Realized-rate expectation (not the go/no-go figure) is
substantially lower: this program's own historical realized/ceiling ratios
have run 13.6%-87.0% (median well under 50%, §15.12), putting mandatory-
only-at-1× (≈10.5 GPU-h) as the reasonable single best guess. All-in worst
case (every optional Gate-1 probe and seed-contingency cell firing) is
≈36.7 GPU-h at 2×.

**Time-to-signal.** 30 mandatory cells across 6 idle GPUs (2-7, confirmed
free per `STATE.md`'s current snapshot) run in `ceil(30/6)=5` parallel
rounds at ≈1250-1550s/cell ≈ **2.2 hours** wall-clock; even the full
60-cell worst case completes in ≈4.3 hours. This is the fastest
time-to-signal of any of the three paths by a wide margin.

**Failure modes.** Structurally low-risk in the "wasted GPU-h" sense: per
§15.18's own Q4 adjudication, CONFIRM (the CI overlapping the pre-
registered `[0.4745, 0.6165]` band) is the strongly expected outcome given
§14's own coherence-exoneration headroom (doses to 0.40 flat, well above
d=64's own realized 0.098-0.200 coherence) — but a REFUTE or DIRECTIONAL-
DRIFT reading would itself be a real, informative capacity-law finding
(the ratio law breaks down beyond d=64), not a null result in the sense of
"nothing was learned." **What this path's own null would mean:** a
disconfirmed or drifting `x0(d)` invariance is a genuine revision to the
capacity-law paper track's own generality claim — independent of and
non-competing with anything REASONING-LINK concludes.

**Why "orthogonal + idle GPUs = run it now" needs one qualification.**
The uptime-metered-box discipline (`CLAUDE.md`/MEMORY: "saturate, GPUs
hot forever") correctly argues against leaving GPUs 2-7 idle while Path
(i)'s tiny gate check and Path (ii)'s attack round occupy essentially no
GPU time at all. But "now" should mean **"as soon as FATAL-1 and MAJOR-3
are fixed and independently audited"**, not literally immediately — this
project's own Build→Audit rule applies to a 2-item manifest-plumbing fix
exactly as much as it applies to a novel architecture; skipping the audit
step because the fix is "small" is precisely the kind of shortcut
`CLAUDE.md`'s own standing rules exist to prevent. The fix-then-audit
cycle is itself cheap and zero-GPU, so this qualification costs no real
calendar time against the "saturate now" framing — it just names the
actual next concrete action correctly.

---

### 16.4 Costs table — all three paths, side by side

| Path | GPU cost | Wall-clock (idle GPUs) | Build prerequisites | Engages the keystone (h≥2 composition)? | What a null tells you |
|---|---|---|---|---|---|
| (i) Stage-0-gate-only | ~0.01 GPU-h | minutes | New episode-render templates (Candidates A/B), reuse existing pool — zero new Stage −1 verification for A, one small addition for B | No — even a pass only licenses h=1 as primary | Failure isolates the blocker as NOT surface-form (premise-iv-style representational gap), mechanically promoting Path (ii) |
| (i) Full Phase-1b grid (conditional) | ~0.4 GPU-h | ~1 hour | Same as above, gated on the Stage-0 pass | No — h=1-primary result at best, h≥2 stays exploratory | A clean h1 pass with h≥2 still floor is itself informative (READOUT-FORM-INVALID-style, per §12's own existing outcome) |
| (ii) Phase-2 familiarization (Rev 5, post-fifth-attack, §16.2) | ≈0.30-1.21 GPU-h raw (training + Stage-0.5 gate null-bands); ≈1.48-12.06 GPU-h at the disclosed 5-10× debug-tax multiplier (§16.2.3 — WIDENED from Rev 1/Rev 2's ≈1.15-11.4 GPU-h by MAJOR-R3-3's newly-priced gate cost; MAJOR-R3-1b's own re-derived query-loss multiplier stays inside margin on its own, §16.10's own BUDGET paragraph) | hours (training, not eval-only) — 1,000-5,000 steps × 18 cells at the measured 0.04544 s/step rate, plus `n_query=2`-pinned query-loss forward passes | FIVE attack rounds COMPLETE (round 1: NEEDS-REDESIGN, 1 FATAL+6 MAJOR+1 MINOR, fixed in Rev 1, §16.7; round 2: NEEDS-REVISION, 6 new MAJOR+2 new MINOR, no FATAL, fixed in Rev 2, §16.9; round 3: NEEDS-REVISION, 3 new MAJOR+4 new MINOR, no FATAL, fixed in Rev 3, §16.10; round 4: NEEDS-REVISION, 3 new MAJOR+0 new MINOR, no FATAL — all internal-consistency bugs inside Rev 3's own fixes — fixed in Rev 4, §16.12; round 5: NEEDS-REVISION, 2 MAJOR+1 MINOR, no FATAL — ALL prose/status-sync, every substantive check PASSED — fixed in Rev 5, §16.13); hop-depth train/eval split present via exclusion-by-sampling (h∈{1,2} trained, h∈{3,4} held out, `hop_set=(1,2)`); query loss pinned as the CORRECTED §4.4 concatenated-recompute mechanism (vocab-space CE, MAJOR-R3-1a fixes a false continuation-pattern citation); `--ckpt-steps`/`--init-checkpoint` build tasks registered with negative tests (MAJOR-NEW-2/3, MINOR-R3-2/3); remaining gate: the round-5 reviewer's confirmation pass on Rev 5's corrected status prose only (§16.2.4) | **Yes — the only path that can** | Six named outcomes now distinguish the trajectory (§16.2.1, hexachotomy, Rev 3 MAJOR-R3-2): PERSISTENT/TRANSIENT/LATE-EMERGENT/CONVERGED-EQUIVALENT/UNRESOLVED/NON-MONOTONE — CONVERGED-EQUIVALENT (task-trained equivalence, arms never separate despite BOTH learning the task with a real determinate effect each, MINOR-R3-1) is the program's most severe finding, distinct from and stronger than Cell 4's zero-shot inertness (MAJOR-3); UNRESOLVED is a distinct, non-negative "underpowered" finding no longer silently folded into it; NON-MONOTONE is a distinct "inconclusive-without-rerun" finding, disclosed per-seed and never averaged away |
| (iii) Scaling wave, mandatory grid | ~21 GPU-h (2×); ~10.5 GPU-h realized-rate expectation | ~2.2 hours (30 cells / 6 GPUs) | FATAL-1 (manifest KeyError) + MAJOR-3 (missing `--d-state` smoke) — both zero-GPU, need build + audit | No — orthogonal, capacity-law-paper-track question | REFUTE/DIRECTIONAL-DRIFT revises the capacity law's own generality claim, unrelated to REASONING-LINK |

---

### 16.5 Design-level constraints inherited from the two Phase-1 LEARNs

**Constraint 1 — every gate needs a chain enforcement branch (gates-must-
abort).** §15.2's discrepancy and its paired `[LEARN]` (`EXPERIMENT_LOG.md`,
Phase-1 entry): `reasoning_link_chain.sh` computed `gate_result_
h1_probe_valid` into Stage 0's JSON but never read it back to decide
whether to proceed — only the unrelated wall-clock cost check was wired as
a real abort. **Registered as a hard design requirement for every chain
script built under this section, not merely a lesson to remember:** any
Phase-1b or Phase-2 launch script must implement an explicit `if`/`case`
abort branch for EVERY registered "must not launch on failure" gate
(Phase-1b: the h1 null-relative pass, the h1 absolute-0.10 backstop, and
premises (iii)/(iv)'s own action-rule checks, all already defined in §8.4/
§4.4 and now doubly load-bearing since Phase-1b's whole point is to
re-evaluate them; Phase-2: the Stage-0.5-familiarized gate — premises
(iii)/(iv) and the h1 floor re-measured on the familiarized OFF-arm
checkpoint AT EVERY TRAJECTORY CHECKPOINT (§16.2.1, MAJOR-4; per-checkpoint
granularity, MAJOR-R3-3) — and the blind-pinned val-loss tolerance
band on the new mix (§16.2.1, MAJOR-6), both now concretely registered
rather than left as "whatever the attack round names"), and each such
branch must be verified by a **deliberate
negative test** — force the gate to fail, confirm the chain actually
halts — before the chain is trusted for an unattended run. A gate computed
but not enforced is not a passed gate, per this same finding.

**Constraint 2 — instrument-vs-distribution matching is a Stage −1 design
question, never an assumption (the zero-shot-OOD-floor lesson).** §15.9
item 3 measured, after the fact, that Option 1's `cos_mean` sits in
`[-0.33,0.25]` across the whole grid — making the inherited `|cos|≥0.9`
threshold (imported unchanged from the task-TRAINED `KEY_ANCHORING`
lineage, where h1=1.0 is routinely achievable) a >10σ event for this
never-task-trained checkpoint family, "consistent with... a
threshold/mechanism mismatch" that was never checked before the full grid
ran. §16.1's own wikitext-bigram grep (§16.1.1) exists precisely because
this project already paid for the "assumed transfer instead of measured
transfer" mistake once and should not pay for a second instance of the
same shape. **Registered as a standing Stage −1 requirement for both
remaining paths:** before either Phase-1b or Phase-2 commits to a
threshold, a template, or a data-mix ratio, that choice must be measured
against the ACTUAL target distribution (the real corpus's own n-gram
statistics for a surface-form choice; the real checkpoint family's own
output/activation distribution for a threshold choice — e.g. Phase-2's own
new familiarization objective may warrant re-deriving the h1 floor's null
band rather than assuming §8.4's original marker-template-derived band
still applies), never assumed to transfer from a differently-distributed
regime (task-trained → never-task-trained, or marker-vocabulary →
natural-corpus-vocabulary) by analogy alone.

---

### 16.6 Recommended sequencing — decision tree with mechanical triggers

**The orchestrator's prior, stress-tested (four refinements below), then
adopted with those refinements folded in:**

1. **Fix Path (iii)'s two build gaps NOW, build + independently audit them
   (zero-GPU, can start immediately, does not wait on anything else).**
   Refinement over the naive "just patch and launch" reading: the audit
   step is not optional just because the fix is small (§16.3's own
   qualification) — but since both fixes are zero-GPU, this costs no real
   calendar time against the "saturate idle GPUs" goal.
2. **Launch Path (iii)'s mandatory 30-cell grid on GPUs 2-7 as soon as (1)
   clears audit — fully parallel with everything below, gated on nothing
   from Path (i) or (ii).** Reports out in ≈2.2 hours; feeds the capacity-
   law paper track independent of REASONING-LINK's own outcome.
3. **Build Path (i)'s Stage-0-gate-only check (Candidates A and B, wikitext-
   mix-ext control-arm cell as PRIMARY, openr1-mix-ext control-arm cell as
   a named expected-null contrast — §16.1.2's own correction of the naive
   "reuse Phase 1's Stage-0 cell" plan), run it (~0.01-0.02 GPU-h,
   effectively free, can run on any spare GPU including one of 2-7 before
   or interleaved with Path (iii)'s own cells).**
4. **Path (ii)'s attack rounds — FIVE COMPLETE (Revs 1-4 + the round-5
   verify pass, all 2026-07-07), all run concurrently and zero-GPU exactly
   as this step originally prescribed.** Round-1 verdict: NEEDS-REDESIGN
   (1 FATAL, 6 MAJOR, 1 MINOR), fix-map §16.7. Round-2: NEEDS-REVISION
   (6 new MAJOR, 2 new MINOR), fix-map §16.9. Round-3: NEEDS-REVISION
   (3 new MAJOR, 4 new MINOR), fix-map §16.10. Round-4: NEEDS-REVISION
   (3 new MAJOR, all internal-consistency), fix-map §16.12. Round-5:
   NEEDS-REVISION (2 MAJOR + 1 MINOR, ALL prose/status-sync; every
   substantive check PASSED — Rev-4 fixes verified landed, independent
   32-pattern re-enumeration clean, diff sweep clean), fix-map §16.13.
   This confirms the refinement over the naive "decide (ii) only after (i)
   reads out" framing was correct: all five critiques ran in parallel with
   Path (i)/(iii) at no calendar cost. **What still gates Path (ii)'s
   build:** the surface-form template is now PINNED (marker template, per
   §16.8.3's mechanical fallback — Phase 1 read out PROBE-INVALID (§15)
   and Phase-1b/Path (i)'s Stage-0 gate refused both candidates (§16.8),
   so the pre-registered marker-template fallback governs); the remaining
   gate (§16.2.4) is the round-5 reviewer's spot-check of the Rev-5
   status-sync patches (the confirmation pass on the first patch ran
   2026-07-07 and found 5 residual prose defects, all fixed in the second
   patch — §16.13's addendum).
5. **At Path (i)'s Stage-0 gate readout, branch mechanically:**
   - **PASS on the wikitext-cell (either candidate), openr1-cell stays a
     null as expected →** launch the full Phase-1b grid (~0.4 GPU-h,
     ~1 hour). This is cheap enough to just run, not worth a second gate.
     On the full-grid result: report h1 as the primary confirmatory result
     if it clears the grid-wide gate, h≥2 as exploratory-only per §16.1.4's
     own honest-plausibility verdict — **explicitly do not read a Phase-1b
     h1 pass as answering the keystone.** Fold the validated template into
     Path (ii)'s recipe (§16.2.1) — whose attack rounds already concluded
     in step 4 (Rev 1, NEEDS-REDESIGN → fixed, §16.7; Rev 2,
     NEEDS-REVISION → fixed, §16.9; Rev 3, NEEDS-REVISION → fixed, §16.10;
     Rev 4, NEEDS-REVISION → fixed, §16.12; Rev 5, NEEDS-REVISION
     prose-only → fixed, §16.13) and whose remaining gate is §16.2.4's
     current one (the round-5 reviewer's spot-check of the Rev-5
     status-sync patches) — then build+audit+launch Phase-2.
   - **PASS on the wikitext-cell but the openr1-cell ALSO passes →** flag
     the confound named in §16.1.3 item 2b before trusting the wikitext
     result; audit before running the full grid.
   - **FAIL on both candidates, at the wikitext-cell specifically →** per
     §16.1.3 item 1, this mechanically promotes Path (ii) to the sole
     remaining instrument that can engage the keystone. Do not build the
     full Phase-1b grid (nothing left to gain from it once the
     Stage-0-gate-only check has already answered "was it the format" with
     "no"). Move directly to finalizing and launching Path (ii) (whose
     attack rounds concluded in parallel in step 4, per the stress-tested
     refinement, and whose remaining gate is §16.2.4's current one — the
     round-5 reviewer's spot-check of the Rev-5 status-sync patches) —
     gauntleted next in the queue.
6. **Path (iii) reports out independently on its own ≈2.2-4.3 hour
   timeline throughout, feeding the capacity-law paper track whenever it
   completes — never gated on, and never gating, steps 3-5 above.**

**Mechanical trigger summary:**

| Trigger | Action |
|---|---|
| FATAL-1 + MAJOR-3 fixed and audited | Launch Path (iii)'s 30-cell grid immediately |
| Path (i) Stage-0 gate: wikitext-cell PASS, openr1-cell null as expected | Launch Path (i)'s full grid |
| Path (i) Stage-0 gate: wikitext-cell PASS, openr1-cell ALSO passes | Audit for a confound before trusting either reading |
| Path (i) Stage-0 gate: wikitext-cell FAIL (both candidates) | Skip Path (i)'s full grid; finalize + build + launch Path (ii) |
| Path (i) full grid: h1 clears, h≥2 stays floor | Report h1 as primary/confirmatory, h≥2 as exploratory (per §12's existing READOUT-FORM-INVALID-adjacent framing); still route to Path (ii) for the keystone itself |
| Path (ii) attack rounds 1-5 + round-5 confirmation pass complete | Marker template pinned (§16.8.3 fallback) → build → audit → launch, independent of Path (iii)'s own status; gated only on the round-5 reviewer's spot-check of the Rev-5 status-sync patches (§16.2.4) |
| Path (iii) grid complete (any outcome) | Report to the capacity-law paper track; does not change anything in this decision tree |

**Status update (2026-07-07, post-Rev-5).** Path (ii)'s attack rounds
(step 4, trigger-table row above) are **all five complete, plus the
round-5 confirmation pass** — see §16.7 (round 1: 1 FATAL+6 MAJOR+1
MINOR), §16.9 (round 2: 6 new MAJOR+2 new MINOR), §16.10 (round 3: 3
new MAJOR+4 new MINOR), §16.12 (round 4: 3 new MAJOR, all
internal-consistency), and §16.13 (round 5: 2 MAJOR+1 MINOR, ALL
prose/status-sync, every substantive check PASSED; its confirmation
pass then found 5 residual prose defects in §16.6/§16.7, fixed in the
second Rev-5 status patch — §16.13's addendum). The familiarization
surface-form template is PINNED: Phase 1 read out PROBE-INVALID (§15),
Phase-1b/Path (i)'s Stage-0 gate refused both natural candidates
(§16.8), so §16.8.3's pre-registered marker-template fallback governs —
nothing is waiting on Path (i). The live next action is: the round-5
reviewer's spot-check of the Rev-5 status-sync patches (§16.2.4), then
Phase-2 build → independent build audit → launch.

---

### 16.7 ATTACK-ROUND-1 fix-map (2026-07-07) — verdict NEEDS-REDESIGN

An independent adversarial pass reviewed §16.2 (Phase-2 task-
familiarization recipe) before any code was written, per house discipline
(mirrors `KEY_ANCHORING_DESIGN.md`'s and this document's own §13.x
attack-round convention) and per §16.2.4's own registered prerequisite.
Verdict: **NEEDS-REDESIGN** — 1 FATAL, 6 MAJOR, 1 MINOR. Every finding
below is fixed in this revision (Rev 1, §16.2.1-§16.2.4); none is
deferred or waved away. Findings are recorded near-verbatim for the
historical record, per house style; resolutions are stated as landed in
this text, not as intentions.

| # | Finding (attack-round on §16.2) | Severity | Fix (Rev 1) | Location |
|---|---|---|---|---|
| FATAL-1 | The recipe had no hop-depth train/test split at all — familiarization would have trained on and evaluated the same hop depths, making any h≥2 result unable to distinguish "learned to compose" from "memorized this specific query shape." `DeltaNetRDTaskConfig.H_train`/`H_test` (§4.3) had only ever been exercised as a semantically-inert placeholder (Phase-1 checkpoints trained on zero hops of anything); Phase-2 is the first place this convention has a real gradient referent, and the periodicity guard that makes the split non-collapsing had never been re-verified under that real referent | FATAL | Familiarize on h∈{1,2} ONLY — no gradient ever flows through an h=3/4 query loss. Eval reads h∈{1,2} as a sanity ceiling and h∈{3,4} as THE H_LINK test. Periodicity guard re-verified for real at both committed K's: `{1,2,3,4} mod 20` and `mod 32` both equal `{1,2,3,4}`, no residue collision between the trained and held-out partitions — closing the exact hop-collapse failure mode `CLAUDE.md`'s own standing rule names (50-100% collapse at K∈{4,8,12,16} before that fix) | §16.2.1 (Hop-depth train/eval split) |
| MAJOR-1 | Arm contrasts were only ever going to be read once, at the terminal checkpoint of the familiarization budget — unable to distinguish a durable capability difference from a training-dynamics artifact that opens then closes under continued training (exactly §16.2.2 confound (a)'s own concern) | MAJOR | Trajectory readout at checkpoints `{250, 500, 1000, 2500, 5000}` of the now-fixed 5,000-step budget; the killer-prediction contrast computed at EVERY checkpoint, every arm, both corpora, all 3 seeds; pre-registered PERSISTENT/TRANSIENT/CONVERGE trichotomy classifies each trajectory before any checkpoint is read | §16.2.1 (Trajectory readout schedule and pre-registered outcome trichotomy) |
| MAJOR-2 | The recipe named no committed K sweep at all for Phase-2, and the implicit capacity-cliff framing risked reading as a general K/d≈0.55 law rather than the d=64-specific result this program has actually located | MAJOR | K∈{20,32} (Leg A's committed pair) in both familiarization and eval; the killer-prediction structure re-applied exactly as §5.3 states it (`\|Δ(K=32)\|>\|Δ(K=20)\|`, K=32 CI excludes zero, K=20's does not); the capacity connection framed explicitly as d=64-cliff-specific, citing `KEY_ANCHORING_SCALING_DRAFT.md` §15.19's own harvested verdict (d=80 REFUTES ratio-invariance, d=96 degenerate/AMBIGUOUS — the general K/d law does NOT generalize past d=64; only the located `x0=0.5455` anchors Phase-2) | §16.2.1 (K sweep and killer-prediction re-application) |
| MAJOR-3 | No named outcome existed for "arms converge to statistical equivalence after task training" — this would have been silently read as Cell 4 ("stabilization is functionally inert"), when task-trained equivalence is a categorically different, stronger finding than Cell 4's zero-shot inertness | MAJOR | Added CONVERGE as a named outcome, explicitly distinct from Cell 4 (§12): CONVERGE requires both arms to have genuinely LEARNED the task (h∈{1,2} clear their sanity floor, Stage-0.5 gate passes) yet never separate on the killer-prediction contrast at any trajectory checkpoint — independently publishable on the same "honest, hard-won negative" logic §12 already grants Cell 4, not a footnote to it | §16.2.1 (CONVERGE vs. Cell 4) |
| MAJOR-4 | The val-loss-matching gate that makes the arms comparable (§7.2) was established on the original pretraining objective/mix, with no equivalent check registered for the NEW familiarization mix — any arm contrast would have been trusted with no validity gate on the familiarized checkpoints at all | MAJOR | New Stage-0.5-familiarized gate: premises (iii)/(iv) and the h1 floor re-measured on the familiarized OFF-arm checkpoint BEFORE any arm contrast is trusted; enforced abort branch required by, and verified per, §16.5 Constraint 1 (gates-must-abort, citing the paired `[LEARN]` in `EXPERIMENT_LOG.md`'s Phase-1 entry) — a deliberate negative test (force the gate to fail, confirm the chain halts) is mandatory before the chain is trusted unattended. Disclosed added value: if the readout construct is sound, premises (iii)/(iv) SHOULD pass post-familiarization — a persistent failure here indicts `S_T^h·q_eff` more decisively than Phase-1's zero-shot PROBE-INVALID verdict could | §16.2.1 (Stage-0.5-familiarized gate); §16.5 Constraint 1 |
| MAJOR-5 | §16.2.3's cost figure (~5-15 GPU-h) was a bare, underived placeholder with no cited realized rate, and its own text ("14M-98M-class model sizes") was ambiguous enough to read as pricing in a 98M-class (rung-2) leg that does not exist anywhere in this program's archive | MAJOR | Re-derived from `FROZEN_BIAS_LM_DESIGN.md`'s own realized rate (0.04544 s/step, 18,175.744s / 400,000 steps across its 20 rung-1 training cells): raw cost = 18 cells × 1,000-5,000 steps × 0.04544 s/step = 0.23-1.14 GPU-h + negligible eval passes, 14M-class ONLY (rung-2 formally PARKED, `FROZEN_BIAS_LM_DESIGN.md` §6.2/§8.1 VERDICT). A disclosed 5-10× debug-tax multiplier (Path (i)'s own precedent, §16.1.2: ~0.01 GPU-h registered against a ≈0.001-0.002 GPU-h measured-rate cost) yields a registered bracket of ≈1.15-11.4 GPU-h, replacing the old placeholder with a fully derived, provenance-disclosed number | §16.2.3 (Cost, full rewrite) |
| MAJOR-6 | No tolerance band existed for re-matching val-loss on the new familiarization mix, and this program has an exact, disclosed precedent for what happens when a blind-pin is constructed after training rather than before it (`FROZEN_BIAS_LM_DESIGN.md`'s rung-1 wave: pin written after all 20 cells completed, forfeiting confirmatory license, wave demoted to DESCRIPTIVE TIER despite numerically real CI arithmetic) | MAJOR | Fresh tolerance band, `mean_ref ± 2·s_ref` (house `k=2` convention, `KEY_ANCHORING_DESIGN.md` §3.6) from the OFF arm's own per-seed loss on the NEW mix, computed and blind-pinned to a committed JSON **BEFORE** the global/per_token familiarization launches — registering `FROZEN_BIAS_LM_DESIGN.md`'s own disclosed "pin before launch" process finding as a hard build requirement here, not a lesson to remember later | §16.2.1 (Familiarization-mix val-loss tolerance band) |
| MINOR-1 | The BIND/QUERY data-mix fraction was picked by feel (an assumed "majority original corpus, minority task episodes" split), with no derivation or measurement plan | MINOR | Registered as a named open item: the fraction must be set by a Stage-0-scale pilot measurement (smallest fraction at which a detectable familiarization signal appears), not committed in this revision — kept open pending that pilot, per this program's own calibration-before-sweep discipline | §16.2.1 (Data mix / MINOR-1 fix) |

**What Rev 1 could NOT cleanly fix, disclosed rather than hidden:**
§16.2.2 confound (c) — whether holding the frozen-bias table's λ fixed
during familiarization is the right choice, or silently handicaps the
intervention arms under a new data distribution — was not raised or
resolved by this attack round; it remains open, flagged for a future
round (§16.2.2's own Rev 1 disposition paragraph). §16.2.2 confound (b)
— the familiarization-fraction-size framing question — is only
partially addressed: MINOR-1 commits to a measurement plan, not a
measured answer. **Rev 1 itself has not yet had its own independent
audit pass** (§16.2.4) — per this project's standing rule that multiple
independent adversarial rounds catch different bugs each round, this
revision should not be read as a certification that Phase-2 is
build-ready, only that attack-round-1's own findings are now landed.

**Rev 2 status update (2026-07-07).** The "Rev 1 itself has not yet had
its own independent audit pass" caveat directly above has now been acted
on. A second independent attack round (a different reviewer from round 1)
targeted Rev 1 specifically, per §16.2.4's own registered next step;
verdict **NEEDS-REVISION**, 6 new MAJOR + 2 new MINOR, no FATAL, every
Rev 1 fix independently re-verified and HOLDS. Full finding→fix trace
recorded in §16.9 (mirroring this section's own table format), landed as
Rev 2 (§16.2.1-§16.2.4). §16.2.2 confound (c) (λ held fixed during
familiarization) remains open, unchanged — round 2 did not raise or
resolve it either. Confound (b) (familiarization-fraction framing) is now
further addressed by Rev 2's own MAJOR-NEW-1 fix (§16.9), which pins the
loss-composition MECHANISM and a build-default mix-ratio anchor (`λ_fam
=1.0`) without resolving MINOR-1's still-open PILOT-MEASURED fraction
itself — that measurement has still not run. **Rev 2 itself has not yet
had its own independent audit pass** (§16.2.4) — the same standing
caveat this paragraph just discharged for Rev 1 now applies, unchanged in
kind, to Rev 2.

**Rev 3 status update (2026-07-07).** The "Rev 2 itself has not yet had
its own independent audit pass" caveat directly above has now been acted
on. A third independent attack round (a different reviewer from rounds 1
and 2) targeted Rev 2 specifically, per §16.2.4's own registered next
step; verdict **NEEDS-REVISION**, 3 new MAJOR + 4 new MINOR, no FATAL,
every Rev 1 AND Rev 2 fix independently re-verified and HOLDS. Full
finding→fix trace recorded in §16.10 (mirroring this section's own table
format), landed as Rev 3 (§16.2.1-§16.2.4). Confound (c) (λ held fixed
during familiarization) remains open, unchanged — round 3 did not raise
or resolve it either. MINOR-1's own pilot measurement (data-mix fraction)
still has not run — unaffected by this round's findings. **Rev 3 itself
has not yet had its own independent (fourth) audit pass** (§16.2.4) — the
same standing caveat this paragraph just discharged for Rev 2 now applies,
unchanged in kind, to Rev 3.

**Rev 4 status update (2026-07-07).** The "Rev 3 itself has not yet had
its own independent (fourth) audit pass" caveat directly above has now
been acted on. A fourth independent attack round (a different reviewer
from rounds 1-3) targeted Rev 3 specifically, per §16.2.4's own
registered next step; verdict **NEEDS-REVISION**, 3 new MAJOR, no new
MINOR, no FATAL, every Rev 1, Rev 2, AND Rev 3 fix independently
re-verified and HOLDS — this round's own findings were narrow,
internal-consistency bugs inside Rev 3's own fixes themselves (the
PERSISTENT `c1` anchor, the totality-count arithmetic, and a field
citation), not newly-discovered design gaps. Full finding→fix trace
recorded in §16.12 (mirroring this section's own table format), landed
as Rev 4 (§16.2.1). Confound (c) (λ held fixed during familiarization)
remains open, unchanged — round 4 did not raise or resolve it either.
MINOR-1's own pilot measurement (data-mix fraction) still has not run —
unaffected by this round's findings. **Rev 4 itself has not yet had its
own independent (fifth) audit pass** (§16.2.4) — the same standing
caveat this paragraph just discharged for Rev 3 now applies, unchanged in
kind, to Rev 4.

**Rev 5 status update (2026-07-07).** The "Rev 4 itself has not yet had
its own independent (fifth) audit pass" caveat directly above has now
been acted on. A fifth independent attack round (a fifth distinct
reviewer) targeted Rev 4 specifically; verdict **NEEDS-REVISION**, 2
MAJOR, 1 MINOR, no FATAL — ALL prose/status-sync defects, with every
substantive check PASSING (all three Rev-4 fixes verified landed
byte-accurate against source; an independent from-scratch 32-pattern
re-enumeration reproduced 1+15+1+4+11=32 and confirmed the round-4
counter-example classifies PERSISTENT/c1=500; the diff-scoped new-bug
sweep found nothing). Findings landed as Rev 5 — a status-sync-only
patch, recipe content unchanged from Rev 4; trace in §16.13. The
reviewer's confirmation pass on the first patch then found 5 residual
prose defects (in §16.6 and this very paragraph's own chain tail),
fixed in the second Rev-5 status patch (§16.13's addendum). Confound
(c) (λ held fixed during familiarization) remains open, unchanged —
round 5 did not raise or resolve it either. MINOR-1's own pilot
measurement (data-mix fraction) still has not run — unaffected. The
remaining gate before build is the round-5 reviewer's spot-check of the
Rev-5 status-sync patches (§16.2.4).

---

### 16.8 PHASE-1B RESULT — gate null, format exonerated (2026-07-07)

**HEADLINE: Phase-1b's Stage-0-gate-only check (§16.1.2) REFUSED to
launch — both natural-language candidates (Candidate A, gift-verb bind
clauses with the reserved query marker dropped entirely; Candidate B,
succession-family verbs, an order of magnitude more common in the
project's own wikitext corpus per §16.1.1's own grep) return
`PROBE-INVALID` on the licensing WIKITEXT-mix-ext control-arm cell,
identically to §15's marker-template result. The OPENR1-mix-ext
"expected-null" contrast cells (§16.1.2/§16.1.3 item 2b) also fail, as
predicted — no confound flag fires. Per §16.1.3 item 1, four structurally
different cells (2 templates × 2 corpora) failing the SAME way is the
single most informative null this instrument can produce: the failure is
NOT attributable to surface form, and this mechanically promotes Path
(ii)/Phase-2 per §16.6's decision tree. The chain's own gate-enforcement
code (`reasoning_link_gate_enforce.py`, built specifically to close §15.2's
DISCREPANCY finding) worked exactly as designed — it read the real gate
result and refused to launch the full 78-cell grid, writing
`STAGE0_GATE_REFUSED` and exiting nonzero — a direct, verified fix of the
gap that let Phase 1 burn ≈0.29 GPU-h running a probe its own Stage-0
already knew was invalid. Realized cost: ≈0.0088 GPU-h of actual GPU time
(four gate cells), ≈0.024 GPU-h total single-launch wall-clock including
CPU-only Stage −1 self-tests — both consistent with §16.1.2's own
"~0.01 GPU-h" pre-registered estimate, no launch fixes or crashes needed
(`phase1b_run1.log`, the only run).

#### 16.8.1 Per-cell gate table (all 4 cells, unblinded, reproduced directly from the raw JSONs)

| Template | Corpus | Role | `recovered_frac`(h1) | null-relative pass | absolute (≥0.10) pass | `gate_result_h1_probe_valid` | premise (iii) median / null p95 / pass | premise (iv) median / null p95 / pass | `forward_counts` a/b | `state_condition_number_mean` |
|---|---|---|---|---|---|---|---|---|---|---|
| A (gift-verb, marker dropped) | wikitext-mix-ext | **licensing** | 0.0 | False | False | **False** | -0.5092 / -0.3637 / **False** | 0.1174 / 0.2028 / **False** | 1/1 | 272,778.7 |
| A (gift-verb, marker dropped) | openr1-mix-ext | expected-null | 0.0 | False | False | **False** | 0.3540 / 0.4492 / **False** | 0.0139 / 0.1424 / **False** | 1/1 | 51,093.7 |
| B (succession) | wikitext-mix-ext | **licensing** | 0.0 | False | False | **False** | -0.5288 / -0.4652 / **False** | 0.0250 / 0.1075 / **False** | 1/1 | 886,249.9 |
| B (succession) | openr1-mix-ext | expected-null | 0.0 | False | False | **False** | 0.3246 / 0.4398 / **False** | 0.0320 / 0.0907 / **False** | 1/1 | 74,407.1 |

Every cell: `recovered_frac`(h1) exactly `0.0`, `null=[0.0,0.0]` (zero
width — nothing to be relative to), causality check `max_abs_diff=0.0`
(forward-A/forward-B agree, `pass=true`), `marker_disagreement_flag=null`
(no marker exists in the natural-completion query window at all — moot by
construction, §16.1.1's own point). h≥2 rows are identical to h1's premise
readings within each cell (premises are computed once per cell, shared
across h, `measure_cell_all_h` convention, same as §15.1). Template B's
verb pool (`succeeded`→14131, `replaced`→6928) verified single-token,
non-colliding under the real GPT-2 tokenizer at Stage −1 item 15
(`phase1b_run1.log` line 20) before any cell ran — the new Stage −1
build item §16.1.1 disclosed as required for Candidate B landed clean.

**Wikitext-primary vs. openr1-contrast reading.** The licensing cells
(wikitext, the register with an actual natural-template referent) and the
expected-null cells (openr1, step-by-step math derivation text with no
"Name verb Name" precedent) fail by the **same margin structure** —
premise (iii) misses its null p95 by a comparable gap in both corpora
(≈0.15 abs at wikitext, ≈0.10-0.12 abs at openr1), premise (iv) clears
nowhere. Neither corpus shows a directional hint that wikitext-native
phrasing helps even partially. **No §16.1.3 item 2b confound flag fires**
— openr1 did NOT clear where wikitext also did not clear, so this is
clean "expected-null behaved as expected" (`phase1b_run1.log`'s own
per-candidate printout), not the ambiguous "openr1 also passes" case that
would have required an audit before trusting the result.

#### 16.8.2 Reading — §16.1.3 item 1, the most informative null

This is not a marginal miss. `premise_iii_pass` and `premise_iv_pass` are
`False` at all 4 cells, `gate_result_h1_probe_valid` is `False` at all 4
cells, and the absolute h=1 floor (§8.4, ≥0.10) is missed by the maximum
possible margin (`0.0`) everywhere — the exact same categorical-failure
shape §15.1 found for the marker template (0/78 cells, 0/312 (cell,h)
readings nonzero). Per §16.0 point 2 (premise (iv) failing harder than
premise (iii) — `k_eff`/`v_eff`, two projections of the same token
captured milliseconds apart in the same forward pass, are nearly
uncorrelated in this checkpoint family), this natural-language pass adds
direct, fresh evidence for that same diagnosis: dropping the OOD query
marker (Candidate A) and swapping to a wikitext-native, order-of-magnitude
more common relation-verb family with genuine compositional semantics
(Candidate B) — the two structurally different levers §16.1's own
failure-mode enumeration named — neither one moves the needle. Per
§16.1.3 item 1's own pre-registered reading: **this reinforces that the
k/v split itself lacks entity-identity structure in this checkpoint
family, a fact no zero-shot prompt redesign can route around.**

#### 16.8.3 Mechanical promotion of Phase-2 per §16.6

§16.6's own trigger table, row 4: *"Path (i) Stage-0 gate: wikitext-cell
FAIL (both candidates) → Skip Path (i)'s full grid; finalize + build +
launch Path (ii)."* That trigger has now fired for real, not
hypothetically. Per §16.1.2's own registration, the conditional ≈0.4
GPU-h full Phase-1b grid is **not launched** — there is nothing left to
gain from it once the Stage-0-gate-only check has already answered "was
it the format" with "no" (§16.6 step 5, third bullet). Path (ii) — task
familiarization (§16.2, Rev 1 landed, §16.7's fix-map: 1 FATAL, 6 MAJOR, 1
MINOR, all fixed) — is now the sole remaining instrument that can engage
the REASONING-LINK keystone question. Path (ii)'s own recipe-pinning step
(§16.2.1, "fold in the validated template if any") has no template to
fold in from this null — Path (ii) proceeds with its own registered
fallback, the original marker template (§16.2.1's own disclosed
contingency), since neither Candidate A nor B cleared a gate to supersede
it.

#### 16.8.4 Gate enforcement worked — contrast with Phase 1's §15.2 discrepancy

Phase 1's own chain (`reasoning_link_chain.sh`) computed
`gate_result_h1_probe_valid` at Stage 0 but never read it — the h1-floor
failure that should have halted the chain before Stage 1 instead let all
78 cells run to completion on an already-invalid probe (§15.2,
`≈0.29 GPU-h` spent past the gate). Phase-1b's chain
(`reasoning_link_phase1b_stage0_chain.sh`) was built specifically to close
that gap (script header, "GATE ENFORCEMENT" block): it calls
`reasoning_link_gate_enforce.py` on each wikitext cell's own JSON and acts
on its REAL subprocess exit code inside an explicit `if` guard (not
`set -e`'s default hard-stop, so BOTH candidates get a mechanical verdict
before the script decides anything, deliberately preserving the
"both fail identically" observation intact). Per `phase1b_run1.log`
lines 722-734: both `REFUSE:` lines printed, `STAGE0_GATE_REFUSED`
written, script exited nonzero, no full-grid launch attempted. This is
the first run of this design in which the registered gate had teeth at
the process boundary and those teeth were actually exercised on a real
failing result — the negative test at Stage −1 item confirmed the
enforcement script's own exit codes under the 3 negative fixtures
PLUS a subprocess-level proof (`phase1b_run1.log` line 26) before any
real cell ran.

#### 16.8.5 Realized GPU-h (box timestamps, UTC, `stat` birth/mtime, single GPU device 3)

| Segment | Start (UTC) | End (UTC) | Duration | GPU-attached? |
|---|---|---|---|---|
| Stage −1 self-tests (19 items) | 20:20:28.226 | 20:21:23.125 | 54.9s (53.5s per the script's own internal timer) | No — `CUDA_VISIBLE_DEVICES=` forced, CPU stub |
| 4 stage0-natural gate cells (A/wikitext → A/openr1 → B/wikitext → B/openr1) | 20:21:23.477 | 20:21:55.331 | 31.85s | Yes — `CUDA_VISIBLE_DEVICES=3` |
| Gate enforcement + `STAGE0_GATE_REFUSED` write | 20:21:55.331 | 20:21:56.258 | 0.93s | No — pure Python, subprocess exit-code read |
| **Single-launch wall-clock** (`phase1b_run1.log` birth → last write) | 20:20:28.223 | 20:21:56.258 | **88.04s ≈ 0.02445 GPU-h** | mixed |

**Realized GPU-device time (the 4 gate cells alone): ≈0.0088 GPU-h** —
matches this task's own pre-stated "~0.01 GPU-h realized" and §16.1.2's
own pre-registered estimate (which itself folds in the CPU-only Stage −1
re-verification and build/debug margin as a conservative envelope, so the
full 0.024 GPU-h single-launch figure is also inside that same
pre-registered ~0.01-0.02 GPU-h bracket, §16.6 step 3). One clean run,
`phase1b_run1.log`, no resume-safe relaunches and no build-time crashes —
a direct contrast with Phase 1's own 3 resume-safe launches after 5
build-time crashes (§15.7). Against the ≈24.20 GPU-h Phase-1 ceiling
(§10, this Stage-0-gate-only check drawing from the same budget line),
Phase-1b's realized cost is ≈0.1% of ceiling on top of Phase 1's own
≈1.2%, ≈1.3% cumulative utilization of the program's Phase-1 GPU budget
across both harvests.

#### 16.8.6 Discrepancies, disclosed prominently

1. **File location.** The task brief named `results/reasoning_link/` for
   the 4 stage0-natural JSONs; the box's actual path (and this design's
   own §16.1.2/chain-script convention) is
   `results/reasoning_link_phase1b/` — a sibling directory to Phase 1's
   own `results/reasoning_link/`, not a subpath of it. Pulled from, and
   archived under, the box's real path; noted here rather than silently
   renamed to match the brief.
2. **No git repo on the box path.** `/home/nvidia/chapter2/deltanet_rd`
   on `youthful-indigo-turkey` is a plain deployed directory (scp/rsync
   target), not a git checkout (`git log` → "fatal: not a git
   repository") — unlike this local repo. §15.7's commit-hash-range
   convention for identifying which fixes were live during a harvest
   does not carry over to this pull; the local working copy of
   `reasoning_link_phase1b_stage0_chain.sh` was diffed byte-for-byte
   against the box's executed copy and is IDENTICAL (no drift), so the
   committed script here is confirmed to be exactly what ran.
3. **No other discrepancies found.** `STAGE0_GATE_REFUSED` exists and is
   the expected 0-byte sentinel (a `touch`, per the chain script's own
   Step 3); all 4 JSONs' `design_ref` fields correctly cite "Rev 6
   (2026-07-07, CLEARED-FOR-BUILD)"; every gate field required by the
   task brief (h1 floor fail, premise iii/iv medians vs. nulls,
   `forward_counts`) was present and internally consistent across all 4
   raw JSONs and the console log's own (matching) numbers.

#### 16.8.7 Next steps (not self-launched by this harvest, per §15.10's own discipline)

Per §16.6's trigger table (row 4) and §16.2.4's own registered
prerequisite: Phase-2's Rev 1 (§16.2.1-§16.2.3, fix-map §16.7) needs a
**fresh, independent second audit pass** — targeting Rev 1 specifically,
not a re-read of the original sketch — before any build starts (this
project's standing multiple-independent-audit-rounds rule,
`CLAUDE.md`). That audit is zero-GPU and can run immediately, concurrently
with Path (iii)'s own independent timeline (§16.6 step 6, never gated on
this). Only after that second pass clears (or a further revision addresses
its findings) does Phase-2 build + get its own independent code audit +
launch, per this project's standing Build→Audit→Run discipline.

---

### 16.9 Rev-1 independent audit findings + Rev 2 fixes (2026-07-07)

A second independent adversarial pass reviewed Rev 1 of §16.2 (Phase-2
task-familiarization recipe, §16.2.1-§16.2.4) before any code was written,
per §16.2.4's own registered next step ("a FRESH adversarial pass
targeting Rev 1 specifically") and per house discipline (mirrors §13.x's
and §16.7's own attack-round convention; a different reviewer from
attack-round-1). Verdict: **NEEDS-REVISION** — 6 new MAJOR, 2 new MINOR,
**no FATAL**. Every Rev 1 fix (the FATAL-1 hop-depth split, MAJOR-1
trajectory schedule, MAJOR-2 K sweep, MAJOR-3 CONVERGE-vs-Cell-4
distinction, MAJOR-4 Stage-0.5 gate, MAJOR-5 cost re-derivation, MAJOR-6
val-loss tolerance band, MINOR-1 data-mix fraction) was independently
re-verified by this pass and **HOLDS** — this round's own findings are
entirely in previously-unattacked territory: whether Rev 1's design-level
fixes correspond to something BUILDABLE in the real codebase
(`lm_pretrain_rd.py`, `grammar_rd.py`, `run_deltanet_rd.py`,
`reasoning_link_probe.py`), a source-reading check round 1 did not run.
Every finding below is fixed in this revision (Rev 2, §16.2.1-§16.2.4);
none is deferred or waved away. Findings recorded near-verbatim for the
historical record, per house style; resolutions are stated as landed in
this text, not as intentions.

| # | Finding (attack-round-2 on Rev 1) | Severity | Fix (Rev 2) | Location |
|---|---|---|---|---|
| MAJOR-NEW-1 | The familiarization query loss (trains on h∈{1,2}, §16.2.1's own hop-split paragraph) had NO defined implementation — a hole at Rev-1-precedent scale (cf. FATAL-1's §4.4-Rev-2 dimension-mismatch hole). Left unspecified, the only existing candidate objective in this codebase (`run_deltanet_rd.py`'s `L_cos+λ·L_nce`, scoring `pred(a,h)=S_T^h·q_eff` against `v_eff_target` in `d_state` space) trains directly on the exact quantity §4.4's Stage-0.5 readout evaluates, which would silently void the Stage-0.5 gate's own claimed independent-validation value | MAJOR | Query loss pinned as ordinary next-token cross-entropy at the `<Q>`-position logits (reusing `DeltaNetLM.forward`'s existing `logits`/`F.cross_entropy` path, `lm_pretrain_rd.py` L1178-1205/L1835) against `target_ids=torch.gather(entity_ids,1,tgt_slot)` (`grammar_rd.py`'s own [grammar 3c] self-test proves this recovers the answer token, L594-616), via a two-forward bind→query state-continuation call mirroring §4.4's own eval-time protocol, repurposed for training. `run_deltanet_rd.py`'s `L_cos+NCE_LAMBDA*L_nce` objective (L82-90, L155-215, L375) explicitly REJECTED, with the space-separation reason stated inline. Loss composition pinned: two separate forward+CE passes per step, `L_total=L_corpus+λ_fam·L_query`, `λ_fam=1.0` build default (equal weighting, no-thumb-on-the-scale justification) — MINOR-1's own pilot-measured fraction requirement UNCHANGED, this is the pilot's sweep anchor, not its answer. Budget disclosed (~1.5× per-step upper bound), stays inside the already-registered 5-10× debug-tax margin; the 1.15-11.4 GPU-h bracket is unchanged | §16.2.1 (new "Familiarization query loss" paragraph) |
| MAJOR-NEW-2 | The registered checkpoint cadence `{250,500,1000,2500,5000}` is not buildable with the existing `--ckpt-every` flag (`lm_pretrain_rd.py` L2853), which gates on a MODULO test (`step % args.ckpt_every == 0`, L1863) — no single divisor produces exactly this 5-point set; the tempting `--ckpt-every 250` workaround silently checkpoints/evals 20 times instead of 5, a 4× eval-pass cost inflation invisible to §16.2.3's own separately-priced "negligible eval passes" line | MAJOR | Registered build task: add `--ckpt-steps` (explicit sorted int list, `nargs="+"`), gate on `step in ckpt_step_set`; `--ckpt-every`'s modulo path left unchanged for every other script that depends on it. `--ckpt-every 250` explicitly forbidden for this launch. New Stage −1 self-test asserts the parsed set matches `[250,500,1000,2500,5000]` exactly and exactly 5 checkpoint files are written per cell | §16.2.1 (Trajectory readout schedule paragraph, build task); §16.2.3 (cross-referenced) |
| MAJOR-NEW-3 | No code path anywhere in `lm_pretrain_rd.py` loads a pretrained checkpoint before training — `main()` always fresh-constructs `DeltaNetLM(...)` (L3016-3031) into `train()` (L3053), which fresh-constructs `AdamW` (L1795); the only `load_state_dict` calls in the file are inside Stage −1 self-tests (L2222, L2251) or an in-run smoke comparison (L2518), never the real training path. "Continue training from the archived Leg-A checkpoint" (§16.2.1's own Base-checkpoints paragraph) has no corresponding build | MAJOR | Registered build task: add `--init-checkpoint <path>`; immediately after model construction (L3031) and before `train()` (L3053), `load_state_dict` the archived checkpoint's `model_state_dict` into the fresh model, reusing the existing Stage −1 round-trip self-test's own proven pattern (L2208-2222). New Stage −1 self-test confirms the loaded model's first post-load forward reproduces the archived checkpoint's own last-recorded eval metrics before any familiarization step runs | §16.2.1 (Base checkpoints paragraph, build task) |
| MAJOR-NEW-4 | Optimizer restart is real and was undisclosed: checkpoints save no `optimizer_state_dict` (`torch.save` call, L1933-1935: only `model_state_dict`/`step`/`config`/`corpus`/`seed`/`run_name`), so familiarization always begins with zero Adam moments; `get_lr(step,...)` (L1308-1317) is a pure function of `step` alone and `train()`'s loop always starts `step` at 1 (L1822-1824), so familiarization also restarts a FULL warmup+cosine cycle from `args.lr` on top of a checkpoint whose weights were already shaped by a full prior decay to the step-20,000 floor | MAJOR | Explicit disclosure paragraph added: uniform across all 18 cells (not an arm-differential confound by itself), but registers the alternative reading connected to the pentachotomy — a TRANSIENT or LATE-EMERGENT trajectory outcome must name shared LR/optimizer-restart dynamics as a live, undifferentiated rival account alongside §16.2.2 confound (a)'s own new-mix training-dynamics concern, not attribute the finding to one over the other without further evidence | §16.2.1 (new disclosure paragraph, after "What continues training, what stays frozen") |
| MAJOR-NEW-5 | §5.2a's blend-off surgery was only cited by reference, not restated inline, even though Phase-2's own blend is "doubly live" in a way Phase-1's static single-checkpoint eval was not — the blend is correctly ON during familiarization TRAINING (continuing the arm as trained) but must be forced OFF at EVERY ONE of 5 trajectory-checkpoint SCORING passes, not just a single terminal read, multiplying the opportunities to silently skip the surgery and re-confound training-effect with mechanical blend re-application | MAJOR | §5.2a's surgery restated inline in §16.2.1 itself: before scoring `rec@0.9` (or any Stage-0.5 gate quantity) at ANY trajectory checkpoint, for ANY arm checkpoint, wrap the scoring forward pass in `reasoning_link_probe.py`'s own `frozen_bias_surgery(model, force_off=True)` context manager (L681-702, the as-built, mutation-tested production tool) — stated as a per-checkpoint requirement, not a one-time citation | §16.2.1 (new inline paragraph, before the pentachotomy) |
| MAJOR-NEW-6 | Rev 1's PERSISTENT/TRANSIENT/CONVERGE trichotomy was not exhaustive: no outcome existed for "separation appears only at the terminal checkpoint, absent at every earlier one" (falls outside all three categories as literally defined), and CONVERGE conflated two epistemically different findings — "arms are genuinely, measurably equivalent" and "we never had enough power to tell" — under one label | MAJOR | Pentachotomy: added LATE-EMERGENT (holds only at step 5000, absent at every earlier checkpoint) with an explicit PERSISTENT/LATE-EMERGENT tie-break for the boundary case where K=32's CI first becomes determinate exactly at the terminal checkpoint. Split CONVERGE into CONVERGED-EQUIVALENT (both arms' own training effects are individually determinate — CI excludes zero vs. off — AND overlap each other, positive evidence of equivalence) vs. UNRESOLVED (the arm's own effect vs. off never becomes determinate at any checkpoint including the terminal one — underpowered, not equivalence). Mechanical primitives (`det`, `holds`, `agree`, `det_arm`) pin exactly which checkpoints must be determinate and what "holds" means per outcome, reusing the existing pinned CI formula (§5.4) — no new statistic introduced | §16.2.1 (pentachotomy, replaces the trichotomy paragraph and its CONVERGE-vs-Cell-4 discussion) |
| MINOR-NEW-1 | MAJOR-6's val-loss tolerance-band fix (Rev 1) under-specified granularity across a now-5-checkpoint trajectory: unclear whether one band covers the whole run or is computed per checkpoint, and unclear whether the OFF arm must complete its own familiarization run before the intervention arms launch | minor | Pinned explicitly: OFF arm (6 cells) launches alone first, run to the full 5,000-step budget standalone; 5 separate `mean_ref±2·s_ref` bands are pinned, one per checkpoint step, from that standalone run, committed to `BANDS_PINNED` JSON BEFORE `per_token`/`global` launch; every checkpoint reading is gated against its OWN band, not just the terminal one. Scheduling-only change disclosed: total committed GPU-h unchanged, only wall-clock launch order changes | §16.2.1 (Familiarization-mix val-loss tolerance band paragraph, extended) |
| MINOR-NEW-2 | "The per-episode loss is masked so no gradient ever flows through an h=3 or h=4 query loss" (Rev 1's own Hop-depth split paragraph) implies a loss-masking mechanism that does not exist in this codebase's own convention and was never going to be built that way | minor | Reworded to exclusion-by-sampling: episode batches are drawn with `hop_set=(1,2)` passed directly to `sample_batch_rd` (`grammar_rd.py` L367-372, L469-470), mirroring `run_deltanet_rd.py`'s own training-draw convention exactly (`hop_set=cfg.H_train`, L689) — h∈{3,4} queries are never sampled into a batch in the first place, so there is no gradient to mask | §16.2.1 (Hop-depth train/eval split paragraph, reworded) |

**What Rev 2 could NOT cleanly fix, disclosed rather than hidden.** The
CI mechanical primitives pinning the pentachotomy (MAJOR-NEW-6) reuse the
existing CI-excludes-zero convention (§5.4) but required this revision to
make an interpretive judgment call not fully dictated by the audit
finding itself — specifically, which arm-vs-arm/arm-vs-off comparison
`agree`/`det_arm` should read — flagged here as a genuine design decision
Rev 2 made, not merely a mechanical transcription of the finding, and a
legitimate target for the next independent round to attack. §16.2.2
confound (c) (λ held fixed during familiarization) remains open,
unchanged from Rev 1 — this round did not raise or resolve it either.
MINOR-1's own pilot measurement (data-mix fraction) still has not run;
MAJOR-NEW-1's `λ_fam=1.0` build default gives the pilot a concrete anchor
but does not substitute for running it. **Rev 2 itself has not yet had
its own independent audit pass** — per this project's standing
multiple-independent-audit-rounds rule, this revision should not be read
as a certification that Phase-2 is build-ready, only that
attack-round-2's own findings are now landed (§16.2.4's own updated gate
text).

**BUDGET (verified, not merely asserted):** every Rev 2 fix is either
budget-neutral (MAJOR-NEW-2's `--ckpt-steps` build task REDUCES eval-pass
cost relative to the forbidden `--ckpt-every 250` workaround, from 20
eval passes down to the already-priced 5; MAJOR-NEW-3's
`--init-checkpoint` build task is a one-time `load_state_dict` call, no
added forward/backward passes; MAJOR-NEW-4, MAJOR-NEW-5, MAJOR-NEW-6, and
MINOR-NEW-2 are pure disclosure/interpretation-rule/reporting-mechanism
fixes with zero GPU cost; MINOR-NEW-1 changes wall-clock launch
sequencing only, not total compute) or a disclosed, bounded,
margin-covered increase (MAJOR-NEW-1's second forward pass, ~1.5×
per-step upper bound, covered by the already-registered 5-10× debug-tax
multiplier's own margin). **The committed 1.15-11.4 GPU-h bracket
(§16.2.3) is UNCHANGED, re-printed not re-derived**, per this task's own
instruction not to weaken any registered element.

---

### 16.10 Rev-2 independent audit findings + Rev 3 fixes (2026-07-07)

A third independent adversarial pass reviewed Rev 2 of §16.2 (Phase-2
task-familiarization recipe, §16.2.1-§16.2.4) before any code was written,
per §16.2.4's own registered next step ("a THIRD, fresh adversarial pass
targeting Rev 2 specifically") and per house discipline (mirrors §13.x's,
§16.7's, and §16.9's own attack-round convention; a different reviewer
from attack-rounds 1 and 2). Verdict: **NEEDS-REVISION** — 3 new MAJOR,
4 new MINOR, **no FATAL**. Every Rev 1 fix (the FATAL-1 hop-depth split,
MAJOR-1 trajectory schedule, MAJOR-2 K sweep, MAJOR-3 CONVERGE-vs-Cell-4
distinction, MAJOR-4 Stage-0.5 gate, MAJOR-5 cost re-derivation, MAJOR-6
val-loss tolerance band, MINOR-1 data-mix fraction) AND every Rev 2 fix
(MAJOR-NEW-1 through MAJOR-NEW-6, MINOR-NEW-1, MINOR-NEW-2) was
independently re-verified by this pass and **HOLDS** — this round's own
findings are entirely in previously-unattacked territory: whether Rev 2's
own MAJOR-NEW-1 fix (the query-loss mechanism) actually CITES the real
code correctly (it did not — a false mechanism claim, caught by reading
the real files, not merely re-deriving on paper — the single highest-value
finding this round produced), whether Rev 2's own cost arithmetic for that
same fix carried an undisclosed assumption, whether the pentachotomy
(MAJOR-NEW-6) is actually exhaustive (it was not, by direct
counter-example), and four buildability/precision gaps in Rev 1/Rev 2's
own registered build tasks and outcome definitions. Every finding below is
fixed in this revision (Rev 3, §16.2.1-§16.2.4); none is deferred or
waved away. Findings recorded near-verbatim for the historical record, per
house style; resolutions are stated as landed in this text, not as
intentions.

| # | Finding (attack-round-3 on Rev 2) | Severity | Fix (Rev 3) | Location |
|---|---|---|---|---|
| MAJOR-R3-1a | Rev 2's query-loss mechanism claimed to reuse "§4.4's two-forward continuation pattern" — checked directly against `reasoning_link_probe.py` and FALSE: §4.4's own `run_forward_b` (via `measure_cell_all_h`) runs forward-B over BIND+query CONCATENATED into ONE call (`main_concat = torch.cat([bind_expanded, query_flat], dim=1)`, L1358-1404), never as a continuation over the query window alone with `initial_states` carried from a prior bind-only call (`forward_body`'s own default, L652-678, is `initial_states=None`, and no probe caller ever overrides it). The REAL continuation pattern lives in a DIFFERENT script, `lm_intervene_rd.py` (context forward `initial_states=None` L208, scored continuation forward `initial_states=init_states` L230) — Rev 2 conflated the two. Continuation also has a REAL correctness gap that makes it the wrong mechanism regardless of the citation: `DeltaNetLMMixer`'s q/k/v `ShortConvolution`s (`lm_pretrain_rd.py` L837-839) carry no cache argument, so a continuation call over the ~6-token query window alone would corrupt every layer's q/k/v at the first `conv_size-1=3` positions (`conv_size=4`), and the per-token sigmoid β-gate would write that corruption into the delta-rule state update itself | MAJOR | Query-loss forward-B repinned as the SAME validated concatenated bind+query recompute §4.4 already uses and Stage −1-verifies (`causality_check`, L791-820, `tol=1e-6`, exercised at `run_stage0`, L1597-1598): `DeltaNetLM.forward(torch.cat([token_ids, query_tokens_one_row]), initial_states=None)` (the model's real class method, L1178-1205, computing full LM-head logits) scored with CE at the last position. The false continuation claim is deleted; the conv-cache correctness gap is stated inline as the reason continuation stays rejected, not merely mis-cited | §16.2.1 (Familiarization query loss, "Definition" paragraph, full rewrite) |
| MAJOR-R3-1b | Resolved TOGETHER with R3-1a: `sample_batch_rd` defaults `Q=cfg.queries=K` (`queries` property, `grammar_rd.py` L335-336, reading `n_query: int\|None=None` L310) — every `DeltaNetRDTaskConfig` this design cites leaves `n_query` unset, so the concatenated recompute fix above, left unpinned, would score all K=32 queries per episode per step; Rev 2's own ~1.5× budget multiplier silently assumed `Q=1` (`T_bind+query_len≈230` tokens treated as the WHOLE added cost, with no query count stated) — the real unpinned figure is `32×230=7,360` added tokens vs. `512` corpus tokens, a ~14.4× ("~15×") per-step multiplier that does NOT fit the 5-10× debug-tax margin and would have blown the committed bracket | MAJOR | Pinned `n_query=2` for familiarization's own query-loss batches (existing `grammar_rd.py` field, zero new code — no caller currently sets it) — one h=1 + one h=2 query IN EXPECTATION per episode (`hop_set=(1,2)`'s own existing IID per-slot draw, `grammar_rd.py` L469-470, unchanged), eval-time `Q=K` convention left UNCHANGED (`episode_config_for_checkpoint` keeps `n_query=None`). Honestly re-derived: `2×230=460` vs. `512` → ~1.90× worst-case (K=32), ~1.57× (K=20) — both inside the existing 5-10× margin ON THEIR OWN. The FINAL committed bracket is separately widened by MAJOR-R3-3's own new cost line below (≈1.48-12.06 GPU-h, replacing ≈1.15-11.4), NOT by this fix — the two effects are shown independently, disclosed as audit-forced, not silently re-asserted | §16.2.1 ("Query count pinned" + "Budget, honestly re-derived" paragraphs, new); §16.2.3 (Cost, bracket updated) |
| MAJOR-R3-2 | Rev 2's five-outcome pentachotomy was provably NOT MECE: the pattern `holds(250)=T, holds(500..2500)=F, holds(5000)=T` satisfies none of PERSISTENT, TRANSIENT, LATE-EMERGENT, CONVERGED-EQUIVALENT, or UNRESOLVED — it maps to ZERO of the five buckets, a direct counter-example, not a hypothetical gap | MAJOR | Added a sixth bucket, NON-MONOTONE (the final else-branch, reached iff none of the first five fire), pinned as "inconclusive-without-rerun," triggering mandatory per-seed disclosure, never averaged into a headline number. Totality verified by exhaustive enumeration over all `2^5=32` possible `holds(c)` truth-patterns (worked table in-doc, including the exact adversarial pattern classified as NON-MONOTONE) — every trajectory now maps to exactly one of six outcomes | §16.2.1 (pentachotomy, sixth outcome + totality proof + worked table, new) |
| MAJOR-R3-3 | The Stage-0.5 gate (premises iii/iv + h1 floor) still read as one-shot (measured once on the OFF-arm checkpoint) while the val-loss tolerance band (MAJOR-6/MINOR-NEW-1) was ALREADY pinned per-checkpoint — a checkpoint could clear its own val-loss band while its Stage-0.5 reading is stale or never measured at that checkpoint at all; PERSISTENT's own early-reading corroboration implicitly assumed Stage-0.5 validity at the early checkpoint without stating it | MAJOR | Stage-0.5 re-measured AT EACH of the 5 trajectory checkpoints on the OFF arm's own weights at that checkpoint; an arm contrast at checkpoint `c` is only interpretable if Stage-0.5 passed at `c`; PERSISTENT's `c1` must now be the first non-terminal checkpoint where BOTH `det(32,c1)=TRUE` AND Stage-0.5 passed. Added cost PRICED, not waved as "small": 30 gate-with-null-band passes (OFF arm's 6 cells × 5 checkpoints) at the §16.8.5-realized ≈0.0022 GPU-h/cell rate ≈ **0.066 GPU-h**, folded into §16.2.3's re-derived bracket | §16.2.1 (Stage-0.5-familiarized gate, per-checkpoint rewrite + priced cost paragraph; PERSISTENT outcome, Stage-0.5 cross-reference); §16.2.3 (Cost, new line) |
| MINOR-R3-1 | CONVERGED-EQUIVALENT could fire on `det_arm(global,5000)=TRUE` alone, with `det_arm(per_token,5000)=FALSE` (a merely WIDE, indeterminate per-token CI) silently passing as if it were positive evidence of equivalence — mislabeling asymmetric uncertainty as equivalence | minor | CONVERGED-EQUIVALENT now requires `det_arm(arm,c)=TRUE` for BOTH global AND per_token (plus `agree(5000)=TRUE`, unchanged); UNRESOLVED widened to the logical complement within the `holds(c)=FALSE`-everywhere precondition, with the rarer "both determinate but disagree" sub-case named explicitly and distinguished from the common "at least one indeterminate" sub-case | §16.2.1 (pentachotomy outcomes #4/#5, condition widened) |
| MINOR-R3-2 | Both new Stage −1 self-tests (`--ckpt-steps`, `--init-checkpoint`) had only POSITIVE assertions (the flag works when given correct input) — neither proved the corresponding failure mode is actually EXCLUDED (extra checkpoints not silently written; a mismatched checkpoint not silently partially loaded) | minor | Registered a negative test for each: `--ckpt-steps {250,500}` must NOT write a step-750 checkpoint (or any step beyond the given list) on a run that continues past 750; loading a deliberately-corrupted/mismatched-config checkpoint via `--init-checkpoint` must raise/ABORT (via `load_state_dict`'s own `strict=True` default or an explicit config-equality assert), not silently proceed | §16.2.1 (`--ckpt-steps` and `--init-checkpoint` build-task paragraphs, negative tests added) |
| MINOR-R3-3 | The `--init-checkpoint` Stage −1 self-test needs the archived checkpoint's own "last-recorded eval metrics" (val loss) to compare against, but the `.pt` file does not carry them — verified directly: `torch.save` writes only `step`/`model_state_dict`/`config`/`corpus`/`seed`/`run_name` (`lm_pretrain_rd.py` L1933-1935), no `val_loss` | minor | Named the dependency explicitly: the comparison target must be read from the ORIGINAL Leg-A training run's own results JSON (`main()`'s `--out` write, L3061-3062; the `"trajectory"` field's own per-checkpoint `val_loss`, L1748/L1939), located via the checkpoint's own `run_name`/`step`/`corpus`/`seed` fields — registered as an explicit build dependency, not a surprise discovered when the self-test first runs. **[CORRECTED 2026-07-07, Rev 4, attack-round-4 MAJOR-R4-3]** The field cited above is wrong — `result['trajectory']` entries (`lm_pretrain_rd.py` L1858, `log_every` cadence) carry no `val_loss` key; `result['checkpoints']` entries (L1950, appending a `res` dict built at L1937 with `val_loss` at L1939, `ckpt_every` cadence) do. §16.2.1's own dependency paragraph is corrected accordingly (§16.12); this row is otherwise left intact per the "record history, don't retcon it" convention | §16.2.1 (`--init-checkpoint` build-task paragraph, dependency paragraph added) |
| MINOR-R3-4 | Nothing disclosed that the per-checkpoint val-loss tolerance band has ZERO discriminating power for the OFF arm's own cells, since the band is computed FROM those same cells — a future reader could cite "OFF passed its own gate" as if it were evidence of anything | minor | One disclosure sentence added: the band is near-tautologically satisfied by the OFF arm's own cells by construction; "OFF passed its own val-loss gate" is never evidence and must never be cited as corroborating the gate's real target (`per_token`/`global`'s own, independently-computed val-loss) | §16.2.1 (Gate/blind-pin granularity paragraph, disclosure sentence added) |

**What Rev 3 could NOT cleanly fix, disclosed rather than hidden.**
§16.2.2 confound (c) (λ held fixed during familiarization) remains open,
unchanged from Rev 1/Rev 2 — this round did not raise or resolve it
either. MINOR-1's own pilot measurement (data-mix fraction) still has not
run — none of this round's findings touch it. The NON-MONOTONE bucket's
own "inconclusive-without-rerun" reading (MAJOR-R3-2) is itself a design
judgment call about WHAT TO DO with a flickering trajectory (disclose
per-seed, never average) rather than a fully mechanical transcription of
the finding (which only established that a sixth bucket was NEEDED, not
what its registered reading should be) — flagged here as a genuine
decision Rev 3 made, a legitimate target for the next independent round,
mirroring Rev 2's own analogous disclosure about the pentachotomy's
`agree`/`det_arm` comparison choice. **Rev 3 itself has not yet had its
own independent (fourth) audit pass** — per this project's standing
multiple-independent-audit-rounds rule, this revision should not be read
as a certification that Phase-2 is build-ready, only that
attack-round-3's own findings are now landed (§16.2.4's own updated gate
text).

**BUDGET (verified, not merely asserted):** MAJOR-R3-1a is a pure
citation/mechanism correction with zero net cost delta versus what Rev
2's own (broken) continuation idea would have cost — concatenated
recompute and continuation process the same total token count
(`T_bind+query_len`), just via one call instead of two. MAJOR-R3-1b's own
`n_query=2` re-derivation (~1.90× worst-case, up from Rev 2's
silently-assumed ~1.5×) stays inside the existing 5-10× debug-tax margin
ON ITS OWN and contributes NO widening to the final bracket. MAJOR-R3-3's
own newly-priced Stage-0.5 per-checkpoint gate cost (≈0.066 GPU-h raw) IS
new and DOES widen the final bracket once the same 5-10× multiplier is
applied to the combined raw total. MAJOR-R3-2, MINOR-R3-1, MINOR-R3-2,
MINOR-R3-3, and MINOR-R3-4 are pure disclosure/classification-rule/
build-dependency fixes with zero GPU cost. **Net effect: the committed
bracket WIDENS from ≈1.15-11.4 GPU-h to ≈1.48-12.06 GPU-h (§16.2.3),
driven entirely by MAJOR-R3-3's own new cost line** — registered as
REPLACING the prior bracket, with the audit trail (which finding, how
much, why) shown in full rather than the new number being silently
substituted for the old one.

---

### 16.11 Leg-B rung-3 rows — scale series complete (2026-07-07)

§15.8 registered Leg B rungs 0-2 as **PARTIAL**, explicitly deferred on
`results/lm_rd_trackc/wave3/ALL_DONE` (§15.9 item 2: "Rung-3 (1.31B) is
not in this harvest at all"). That sentinel never landed —
`/tmp/wave3_training_summaries.json` on box shows wave3 self-terminated
at **step 155081/183105 = 84.69% (~84.7%) of its token-matched training
budget** (`timed_out: true`), with its val-loss trajectory already
flattening by the final recorded checkpoints (openr1-mix-ext /
wikitext-mix-ext `final_val_loss` 1.162 / 5.267 at step 155081, against a
trajectory that had already slowed by step ~61,000) —
**PLATEAU-NEUTRALIZED**: the early stop does not compromise this rung's
representativeness for this probe. This entry runs the two deferred
rung-3 cells directly against the run's own final saved checkpoint
(step 155000), via `reasoning_link_probe.py --mode cell`'s `--ckpt`
override (bypassing `leg_b_ckpt_path_final`'s `ALL_DONE`-gated glob,
which would otherwise defer this rung indefinitely) — a small,
disclosed, one-off queue task (registered ~0.02 GPU-h), not a
re-opening of §15's own routing or a Phase-1 grid re-launch.

**What ran.** `reasoning_link_rung3_chain.sh` (GPU 0 only, GPUs 2-7 were
running the unrelated `keyanchor_scaling_wide` sweep, untouched): Stage
-1 self-tests (19/19 + gate checks, PASS, 61.7s, CPU-only) gating the
launch, then 2 cells — `{openr1-mix-ext, wikitext-mix-ext} × rung 3`,
K=64 (rung-3's own committed near-cliff K, `K_SWEEP`/`LEG_B_RUNG_CFG`),
hops {1,2,3,4}, surgery=native (Leg B has no frozen-bias arms), seed 0
only (§6.1's PINNED rung-3 configuration — 1 seed, not rungs 0-2's 3),
batch-size 4 (LAUNCH FIX 5/7's int32-pointer-overflow floor for
`d_model=2560`, independently re-confirmed by
`results/reasoning_link/stage05_rung3_cost_calibration.json`'s own real
measurement: `action="OK: within budget, proceed"`, `ratio_to_baseline
=0.042`). The checkpoint's `d_model=2560/d_state=128/n_layers=22/
conv_size=4` config was read from `ckpt["config"]` — the checkpoint's own
saved dict (`load_checkpoint` → `DeltaNetLM(**ckpt["config"])`) — never
from `LEG_B_RUNG_CFG`, which is used only by the path *resolvers* this
run bypassed; confirmed directly in each output JSON's own
`ckpt_config` field before trusting any number below. Both cells
completed with `forward_counts={"forward_a":1,"forward_b":1}` (the
FATAL-1 regression guard, `assert_forward_call_counts`) — no per-h
forward-loop regression, one shuffled-pairing draw scored at all 4 h's,
exactly sec 10's own pricing.

**Per-cell gate table (h=1, the headline row; full h∈{1,2,3,4} in the
archived JSONs).**

| Corpus | recovered_frac@0.9 (h=1) | premise (iii) median | (iii) null p95 | (iii) pass | premise (iv) median | (iv) null p95 | (iv) pass |
|---|---|---|---|---|---|---|---|
| openr1-mix-ext | 0.0000 | -0.0330 | -0.0173 | **False** | +0.1330 | +0.2461 | **False** |
| wikitext-mix-ext | 0.0000 | +0.0198 | +0.0335 | **False** | -0.0619 | +0.1102 | **False** |

`recovered_frac@0.9 = 0.0000` at every one of the 8 (corpus × h) readings
in this harvest (both corpora, all h∈{1,2,3,4}) — `cos_mean` stays
centered near zero throughout (`[-0.050, +0.060]`), the same
sub-threshold distribution §15.9 item 3 already characterized for rungs
0-2 (reaching `|cos|>0.9` from this spread is a >10σ event). Premise
(iii) (query↔same-entity-key alignment) and premise (iv) (key↔value
alignment) both fail their null-p95 action rule at both corpora — the
bind↔query alignment gate failure is not a small-scale artifact; it
reproduces bit-for-bit in kind (never in exact number, per-checkpoint)
at 1.31B, the largest rung this design registers.

**Verdict.** PROBE-INVALID, unchanged, now confirmed across the full
Leg-B scale ladder. Combined with §15.8's rungs 0-2 (18 cells × 4h =
72/72 zero), this rung-3 entry (2 cells × 4h = 8/8 zero) completes the
series: **80/80 Option-1 readings are `recovered_frac@0.9=0.0` across
all 4 rungs (14M → 98M → 392M → 1.31B), both corpora, every tested h**.
This is a **scale-series completion, not a new finding** — it closes
§15.9 item 2's own open item ("nothing above should be read as a 4-rung
verdict") by supplying the missing rung, and the 4-rung verdict it
licenses is the same one §15.1/§15.4/§15.8 already gave for rungs 0-2:
the instrument itself is probe-invalid, at every scale tested, so no
CONFIRM/REFUTE/AMBIGUOUS reading of the underlying H_LINK-B hypothesis
is possible from Option 1 at any rung. This does **not** reopen §15's
routing or §16.0-16.10's own path analysis — it is the disclosed
completion of a partial harvest, not a re-audit.

**Option 2 (secondary, non-headline — §5.2/§8.2's standing rule
unchanged).** The "worse with scale" margin trend §15.8 flagged (rung0
≈-3.2/-3.4 → rung1 ≈-3.7/-3.9 → rung2 ≈-4.3/-5.5, `h=3,4`) continues
monotonically at rung-3: `option2_margin_mean` at h=3,4 is
**-6.33/-6.16 (openr1-mix-ext)** and **-6.90/-6.71 (wikitext-mix-ext)**
— more negative again. Reported descriptively only, exactly as §15.8
cautions: Option 1 is flat zero at every rung, so `option_agreement` is
vacuously "agree" here too (no direction on Option 1 to agree or
disagree with) — this is **not** read as validating a monotone Option-2
trend, only as the same continuing pattern already on record.

**Realized cost.** GPU-0 wall-clock from Stage -1 completion to the
wikitext cell's own log finalizing: 61s (2 cells, checkpoint load +
one forward-A + one forward-B each) = **0.017 GPU-h**, against the
registered ~0.02 GPU-h budget (Stage -1's own 61.7s was CPU-only, GPUs
2-7 untouched throughout). No discrepancies against the registered
invocation convention, no OOM/kernel-overflow retries needed (batch=4
held on the first attempt, consistent with Stage 0.5's own prior
calibration on this exact shape).

Raw JSONs + logs archived at
`experiment-runs/2026-07-07_reasoning_link_rung3/` (repo) and mirrored
to `/Volumes/1TB_SSD/learned-representations/experiment-runs/`; the
84.7%-budget disclosure is carried inside each output JSON's own
`harvest_metadata` key (additive-only, never touching an
instrument-computed field), not merely narrated here.

---

### 16.12 Rev-3 independent audit findings + Rev 4 fixes (2026-07-07)

A fourth independent adversarial pass reviewed Rev 3 of §16.2 (Phase-2
task-familiarization recipe, §16.2.1-§16.2.4) before any code was written,
per §16.2.4's own registered next step ("a FOURTH, fresh adversarial pass
targeting Rev 3 specifically") and per house discipline (mirrors §13.x's,
§16.7's, §16.9's, and §16.10's own attack-round convention; a different
reviewer from attack-rounds 1-3). Verdict: **NEEDS-REVISION** — 3 new
MAJOR, **no new MINOR, no FATAL**. Every Rev 1 fix (the FATAL-1 hop-depth
split, MAJOR-1 trajectory schedule, MAJOR-2 K sweep, MAJOR-3
CONVERGE-vs-Cell-4 distinction, MAJOR-4 Stage-0.5 gate, MAJOR-5 cost
re-derivation, MAJOR-6 val-loss tolerance band, MINOR-1 data-mix
fraction), every Rev 2 fix (MAJOR-NEW-1 through MAJOR-NEW-6, MINOR-NEW-1,
MINOR-NEW-2), AND every Rev 3 fix (MAJOR-R3-1a, MAJOR-R3-1b, MAJOR-R3-2,
MAJOR-R3-3, MINOR-R3-1, MINOR-R3-2, MINOR-R3-4) was independently
re-verified by this pass and **HOLDS** — this round's own findings are
narrow: an internal-consistency bug in the PERSISTENT outcome's own `c1`
anchor (introduced by Rev 3's own MAJOR-R3-3 fix, which added a
Stage-0.5-at-`c1` requirement without re-checking that the underlying
`c1` SELECTION rule still implied `holds(c1)=TRUE`), an arithmetic slip
in the totality-count narration Rev 3's own MAJOR-R3-2 fix added, and a
false field citation inside Rev 3's own MINOR-R3-3 fix. Every finding
below is fixed in this revision (Rev 4, §16.2.1); the corresponding
Rev 3 fix-map row (§16.10, MINOR-R3-3) is annotated with a bracketed
correction note rather than silently rewritten, per house style. Findings
recorded near-verbatim for the historical record; resolutions are stated
as landed in this text, not as intentions.

| # | Finding (attack-round-4 on Rev 3) | Severity | Fix (Rev 4) | Location |
|---|---|---|---|---|
| MAJOR-R4-1 | PERSISTENT's own `c1` was defined as "the FIRST checkpoint where `det(32,c1)=TRUE`," then the very next sentence required `holds(c1)=TRUE` — but `det(32,c)=TRUE` does NOT imply `holds(c)=TRUE` (`holds` also requires `det(20,c)=FALSE` AND `\|Δ(32,c)\|>\|Δ(20,c)\|`). Direct counter-example: `det(32,250)=TRUE` but `holds(250)=FALSE`, `holds(500..5000)=TRUE` — a real PERSISTENT-shaped trajectory (monotone-true from 500 through 5,000) whose literal `c1`-selection rule anchors on checkpoint 250, where `holds` is FALSE, silently routing the whole trajectory to NON-MONOTONE (§16.2.1 outcome 6's own else-branch) instead of PERSISTENT — directly contradicting the section's own worked table row `F,T,T,T,T → PERSISTENT, c1=500` | MAJOR | `c1` redefined as the FIRST checkpoint where `holds(c1)=TRUE` (the `det(32)`-only selection step is dropped); the tie-break (non-terminal `c1` requirement) and the Stage-0.5 skip-past provision reworded to anchor on `holds`, not `det(32)`, throughout — closing the misrouting while leaving every other outcome (2-6) and the worked table unchanged (the table already implicitly used `holds`-based `c1`, confirmed by re-check) | §16.2.1 (PERSISTENT outcome, `c1` definition + tie-break + Stage-0.5 skip-past provision) |
| MAJOR-R4-2 | The "Totality, checked exhaustively" paragraph (Rev 3's own MAJOR-R3-2 fix) claimed "the 16 with `holds(5000)=FALSE` are ALL claimed by #2 TRANSIENT" and summed `1+16+1+4+11=32` — both wrong: the all-FALSE pattern (already claimed by #4/#5) is itself one of the 16 `holds(5000)=FALSE` patterns, leaving only 15 available for #2 TRANSIENT among the 31 patterns with ≥1 TRUE; and the stated sum `1+16+1+4+11` literally equals 33, not 32 | MAJOR | Corrected to: 1 (all-FALSE, `FFFFF` → #4/#5) + 15 (TRANSIENT, the OTHER `holds(5000)=FALSE` patterns) + 1 (`FFFFT` → LATE-EMERGENT) + 4 (PERSISTENT: `TTTTT`, `FTTTT`, `FFTTT`, `FFFTT`) + 11 (NON-MONOTONE) = 32. Worked-table rows re-verified against MAJOR-R4-1's new `holds`-based `c1` anchor — all classify identically (the table already used `holds`, not `det(32)`, per row) | §16.2.1 (Totality paragraph, count + sum corrected) |
| MAJOR-R4-3 | Rev 3's own MINOR-R3-3 fix (§16.2.1's dependency paragraph, and the corresponding §16.10 fix-map row) claimed the archived checkpoint's val loss lives in `result['trajectory']` — FALSE, verified directly against `lm_pretrain_rd.py`: `train()` declares `trajectory, checkpoints = [], []` as separate lists (L1801); `trajectory` entries (appended L1858, at the `log_every` cadence) carry only `"step"`/`"loss"`/`"lr"`/`"grad_finite"`/`"skip_rate_so_far"` — no `val_loss` key; `checkpoints` entries (appended L1950, at the `ckpt_every` cadence, L1863) are built from a `res` dict with `"val_loss"` at L1939. Both surface as separate top-level fields in the assembled result JSON, `"trajectory": trajectory, "checkpoints": checkpoints` (L1748) | MAJOR | §16.2.1's dependency paragraph corrected to cite `result['checkpoints']` (not `'trajectory'`), with the right line numbers and cadence for both lists, so the reader can tell them apart. §16.10's MINOR-R3-3 fix-map row is annotated with a bracketed `[CORRECTED 2026-07-07, Rev 4, ...]` note (house style, `STATE.md`/`EXPERIMENT_LOG.md` convention) rather than silently rewritten, per the "record history, don't retcon it" rule this document already applies to its own fix-map tables (§13.x, the `q_conv1d` FATAL fix-map row) | §16.2.1 (`--init-checkpoint` build-task paragraph, dependency paragraph corrected); §16.10 (MINOR-R3-3 row, correction note appended) |

**What Rev 4 could NOT cleanly fix, disclosed rather than hidden.**
§16.2.2 confound (c) (λ held fixed during familiarization) remains open,
unchanged from Rev 1/Rev 2/Rev 3 — this round did not raise or resolve it
either. MINOR-1's own pilot measurement (data-mix fraction) still has not
run — none of this round's findings touch it. **Rev 4 itself has not yet
had its own independent (fifth) audit pass** [SUPERSEDED 2026-07-07,
Rev 5: that fifth pass has since run — see §16.13 immediately below;
sentence kept for the historical record] — per this project's
standing multiple-independent-audit-rounds rule, this revision should not
be read as a certification that Phase-2 is build-ready, only that
attack-round-4's own findings — all three of them internal-consistency
bugs inside PRIOR rounds' own fixes, not newly-discovered design gaps —
are now landed.

**BUDGET (verified, not merely asserted):** all three Rev 4 fixes are
pure logic-correction/prose-correction fixes (a `c1`-anchor redefinition,
a totality-count arithmetic correction, and a field-citation correction)
with zero GPU cost and no change to any registered build task, op count,
or per-step forward pass. The committed ≈1.48-12.06 GPU-h bracket
(§16.2.3) is UNCHANGED, re-printed not re-derived, exactly as this task's
own instruction requires for anything this revision did not touch.

### 16.13 Round-5 independent audit findings + Rev 5 fixes (2026-07-07)

The fifth independent adversarial pass — targeting Rev 4 specifically,
per §16.2.4's own registered next step — ran 2026-07-07 (fifth distinct
reviewer). Verdict: **NEEDS-REVISION** — 0 FATAL, 2 MAJOR, 1 MINOR.

**Substantive checks: ALL PASSED.** (1) All three Rev-4 fixes verified
landed as prescribed, every line citation re-executed byte-accurate
against `lm_pretrain_rd.py` (L1748/L1801/L1858/L1863/L1933-1935/
L1937-1939/L1950/L3061-3062). (2) An independent, from-scratch,
programmatic 32-pattern totality enumeration under the holds-anchored
`c1` reproduced 1+15+1+4+11=32 exactly, matched every worked-table row,
and confirmed the round-4 counter-example (`F,T,T,T,T`) now classifies
PERSISTENT with c1=500. (3) The diff-scoped new-bug sweep of Rev 4's
own changes found nothing.

**Findings (all prose/status-sync, zero design impact):**

| # | Severity | Finding | Rev 5 fix |
|---|---|---|---|
| R5-1 | MAJOR | §16.2.4's body still narrated "Three independent attack rounds have now run" / "Rev 3's own fixes have not yet had their own independent audit pass" / "next step is a FOURTH pass targeting Rev 3" — contradicting its own Rev-4 header and the actual state (round 4 ran and produced Rev 4; round 5 ran targeting Rev 4) | Body rewritten: four rounds narrated with round-4 verdict paragraph added, round-5 verdict + scoping recorded, per-round finding-count disclosure extended through round 5 |
| R5-2 | MAJOR | §16.6 item 4's lead said "TWO COMPLETE (Rev 1; Rev 2)" — stale since Rev 2, self-contradicting its own paragraph and §16.6's trigger table | Lead rewritten: FIVE COMPLETE with per-round verdicts; template-pin status corrected to PINNED (marker template via §16.8.3's mechanical fallback); remaining gate restated as the round-5 confirmation pass only |
| R5-3 | MINOR | §16.2 preamble's status-note chain stopped at Rev 3 ("Read ... below as Rev 3" stale) | Rev-3 note marked superseded; Rev-4 and Rev-5 status notes appended mirroring the Rev-1/Rev-2 pattern |

**Scope pinned by the round-5 reviewer itself:** the three fixes above
are pure text/status-sync edits — zero GPU cost, zero change to any
build task, algorithm, or the ≈1.48-12.06 GPU-h bracket (§16.2.3,
re-printed not re-derived). Per that reviewer's own scoping, the
substantive checks (1, 2, 4 above) carry forward without
re-verification; the remaining gate before CLEARED-FOR-BUILD is that
same reviewer's confirmation pass on the corrected status prose ONLY.

**One pre-existing observation, disclosed (not a Rev-5 item):** the
round-5 reviewer noted the Stage-0.5 skip-past provision does not state
what happens if Stage-0.5 never passes at ANY non-terminal checkpoint
within an otherwise-confirmed monotone holds-true run. This ambiguity
predates Rev 4 (it is MAJOR-R3-3-era text, unchanged in kind by Rev 4's
anchor rewording) and is explicitly out of round 5's tight scope; it is
recorded here so the build round inherits it as a named open item
rather than rediscovering it.

**§16.13 addendum — round-5 confirmation pass (2026-07-07, same
reviewer).** The confirmation pass on the first Rev-5 status patch
confirmed all three prescribed fixes (R5-1/R5-2/R5-3) LANDED, then
found 5 residual status-sync defects where the patch's propagation
stopped short inside §16.6/§16.7 — 2 MAJOR (§16.6's closing
status-update paragraph still narrated post-Rev-4 state with a false
"live next action"; §16.6's trigger-table Path-(ii) row still named the
superseded fifth-audit gate) and 3 MINOR (a false verdict label in
§16.6 item 4's new parenthetical — "Path (i) and Phase-1b both read out
PROBE-INVALID" conflated Phase 1's §15 PROBE-INVALID verdict with
Phase-1b's §16.8 Stage-0 gate-null, and Path (i) IS Phase-1b; §16.6
step-5 branches still pointing at the superseded gate; §16.7's chain
tail left undischarged with no Rev-5 status update). All 5 fixed in the
second Rev-5 status patch: §16.6 status paragraph and trigger row
rewritten post-Rev-5, the verdict label corrected to the reviewer's own
prescribed sentence, both step-5 branches repointed, and §16.7 given
its Rev-5 status-update paragraph in the established chain convention.
The reviewer noted the recursive risk explicitly (fixes introducing
false citations — the same failure mode rounds 3→4 exhibited); the
corrected verdict-label sentence was taken verbatim from the reviewer's
prescription rather than re-paraphrased. Substantive checks 1/2/4
continue to carry forward; the remaining gate is the reviewer's
spot-check of these five edits, after which §16.2.4's gate reads
CLEARED-FOR-BUILD.

### 16.14 Phase-2 build-audit findings + fixes (2026-07-07)

An independent build audit of the Phase-2 familiarization BUILD (commits
`1f53a68`+`3937d0c`, discharging §16.2.4's CLEARED-FOR-BUILD gate) ran
2026-07-07 against the actually-shipped code, not the design sketch.
Verdict: **FINDINGS-REQUIRE-FIXES** — 0 FATAL, 2 MAJOR, 3 MINOR (all 4 gate
mutation-tests, run separately, PASSED 4/4). A FIX agent applied every
prescribed fix the same day, re-verified with new/extended Stage -1 items.
The two MAJORs, house table style (mirrors §16.7/§16.9/§16.10/§16.12/
§16.13's own finding→fix convention):

| # | Severity | Finding | Fix |
|---|---|---|---|
| MAJOR-1 | MAJOR | `phase2_familiarization_train.py`'s own module docstring CLAIMED EVAL-purpose episodes (the Stage-0.5 gate's premise/null-shuffle batch AND the killer-prediction arm-contrast) draw from `pools.heldout_name_ids` (`use_heldout_entities=True`) — FALSE as built: `reasoning_link_probe.measure_cell_all_h` had NO `use_heldout_entities` parameter at all and never overrode `grammar_rd.sample_batch_rd`'s own default `False`, so every eval episode silently drew from `pools.train_name_ids`, the SAME pool familiarization TRAINING itself drew from. Affected BOTH `compute_stage05_gate` and the killer-prediction readout (`phase2_trajectory_analysis.killer_prediction_readout` → `reasoning_link_probe.run_cell`) — the wave's central arm-contrast measurement | Threaded an ADDITIVE `use_heldout_entities: bool = False` param through `measure_cell_all_h` and `run_cell` (default False = byte-identical pre-existing behavior for every Phase-1 caller); set `True` at both Phase-2 eval call sites. New Stage -1 item verifies eval batches' entity ids are a subset of `heldout_name_ids`, PLUS a mutation-style negative (flag=False → ids are NOT a subset) |
| MAJOR-2 | MAJOR | `compute_stage05_gate` reused the TRAINING `episode_cfg` (`n_query=2`) instead of a design-pinned Q=K eval config — a 16x power reduction (`gate_batch_size*N_QUERY=16*2=32` vs the registered `Q=K=32`'s own `B*Q=512`) on §16.2.1's own most stringent readout-soundness test, and contradicted the "byproduct of the scoring pass" claim (the trajectory-readout path already uses Q=K via `reasoning_link_probe.episode_config_for_checkpoint`) | Added `familiarization_gate_episode_config` (Q=K, `n_query=None`), threaded into `compute_stage05_gate` as a SEPARATE eval-purpose config from training's own `n_query=2` config. New Stage -1 assertion: the gate's own episode config has `queries==K` while the training config still has `n_query=2` |

**Three MINORs, fixed in the same pass (not tabled in full here — see the
build commit's own diff for line-level detail):** MINOR-1 (resume replayed
the same early RNG stream — `gen_corpus`/`gen_episode` now seed from
`phase2_seed(..., checkpoint_step=start_step)`, so post-resume draws differ
from consumed pre-crash draws); MINOR-2 (`phase2_chain.sh` reserved 2 GPUs
but ran cells strictly serially — cells now launch 2-way parallel via
`run_cells_2way`, one per physical GPU, preserving the OFF-arm-first
barrier and resume-safety); MINOR-3 (this section's own documentation
gaps — the per-seed-labels reconciliation above, the K-coupling note
above, and this §16.14 itself).

**Scope, disclosed explicitly:** this was a build-correctness audit (does
the shipped code match what the CLEARED-FOR-BUILD design registered), not
a further design-level attack round — it carries no new finding against
the recipe itself, and does not reopen §16.2.4's gate. Every fix here is
additive/backward-compatible by construction (new parameters default to
the pre-fix behavior), verified by re-running the full Phase-2 Stage -1
suite to completion after every fix landed.

---

### 16.15 PHASE-2 RESULT — Stage-0.5 gate REFUSED at 30/30 (cell, checkpoint) readings; `per_token`/`global` launch mechanically blocked (2026-07-08)

**HEADLINE.** The OFF arm's 6 familiarization cells (2 corpora ×
3 seeds, `openr1-mix-ext` / `wikitext-mix-ext`) ran to completion —
`steps_completed=5000/5000` at all 6, `grad_finite=True` at all 606
trajectory rows (101 rows × 6 cells), no crashes, one clean single
launch (`phase2_familiarization_run1.log`, the only run). The
Stage-0.5-familiarized gate (§16.2.1, MAJOR-4/MAJOR-R3-3) then
**FAILED at every one of the 30 (cell, checkpoint) readings** —
6 cells × {250,500,1000,2500,5000} — `recovered_frac(h1)` exactly
`0.0000` at all 30, `premise_iii_pass`/`premise_iv_pass`/`probe_valid`
all `False` at all 30. Per §16.5 Constraint 1's gates-must-abort rule,
`phase2_gate_enforce.py` refused the `per_token`/`global` launch at
the terminal checkpoint for all 6 (corpus, seed) cells, wrote
`results/phase2/STAGE05_LAUNCH_GATE_REFUSED`, and the chain exited
cleanly — GPUs 0-1 idle, no partial/orphaned launch. This is the
pre-registered abort branch, not a crash: per §16.2.1's own disclosed
"double duty" framing (line ~4961), a **persistent** post-familiarization
failure of premises (iii)/(iv) "indicts the readout construct itself
far more decisively than Phase-1's own zero-shot PROBE-INVALID verdict
could." That is exactly the pattern observed. But the trajectory data
also show something the gate-only readout cannot see: `L_query` (the
training/eval objective's own vocab-space CE, deliberately built as a
DIFFERENT readout from the d_state-space gate — see §16.15.2) fell
substantially in all 6 cells, just not far enough to clear the
pre-registered <50% pin. Both facts are reported below without forcing
either into an overclaim.

#### 16.15.1 Gate table — all 30/30 (cell, checkpoint) readings FAIL

| Corpus | Seed | ck250 | ck500 | ck1000 | ck2500 | ck5000 |
|---|---|---|---|---|---|---|
| openr1-mix-ext | 0 | FAIL | FAIL | FAIL | FAIL | FAIL |
| openr1-mix-ext | 1 | FAIL | FAIL | FAIL | FAIL | FAIL |
| openr1-mix-ext | 2 | FAIL | FAIL | FAIL | FAIL | FAIL |
| wikitext-mix-ext | 0 | FAIL | FAIL | FAIL | FAIL | FAIL |
| wikitext-mix-ext | 1 | FAIL | FAIL | FAIL | FAIL | FAIL |
| wikitext-mix-ext | 2 | FAIL | FAIL | FAIL | FAIL | FAIL |

`recovered_frac(h1) = 0.0000` at every one of the 30 readings (0/512
scored queries recovered, at every checkpoint, every cell) — the
identical categorical floor §15 found at Phase 1 (0/312 (cell,h)
readings, marker template) and §16.8 found at Phase-1b (0/4 cells,
natural template). This is the THIRD instrument, at THREE structurally
different surface forms/training regimes, landing on the exact same
`0.0` floor — not a marginal miss.

**How far the two premises actually missed (terminal step-5000 values,
`median` vs. its own null-shuffle `p95`; premise passes iff
`median > p95`):**

| Cell | premise (iii) median | null p95 | pass? | premise (iv) median | null p95 | pass? |
|---|---|---|---|---|---|---|
| openr1_s0 | 0.1531 | 0.2358 | FAIL | 0.0556 | 0.1137 | FAIL |
| openr1_s1 | -0.0880 | 0.0520 | FAIL | 0.0919 | 0.2569 | FAIL |
| openr1_s2 | 0.5227 | 0.6044 | FAIL | 0.0474 | 0.1466 | FAIL |
| wikitext_s0 | -0.4130 | -0.2854 | FAIL | -0.0056 | 0.0893 | FAIL |
| wikitext_s1 | -0.3376 | -0.2649 | FAIL | -0.1773 | -0.0093 | FAIL |
| wikitext_s2 | 0.4775 | 0.5493 | FAIL | 0.1159 | 0.1876 | FAIL |

Premise (iii)'s median itself ranges from -0.41 to +0.52 across cells
(real, cell-dependent structure, not a flat zero) — but it never once
crosses its own null p95, by margins between 0.05 and 0.13 absolute.
Premise (iv) shows the same never-crosses pattern, consistent with
§16.0's standing diagnosis (`k_eff`/`v_eff`, two projections of the
same token, stay nearly uncorrelated in this checkpoint family) —
familiarization training did not change that structural fact.

#### 16.15.2 THE central question — task-level learning vs. readout validity

**Mechanical pin (pre-registered):** terminal `L_query` < 50% of its
own step-250 value.

| Cell | L_query@250 | @500 | @1000 | @2500 | @5000 | ratio(5000/250) | relative fall | pin (<50%) |
|---|---|---|---|---|---|---|---|---|
| openr1_s0 | 4.8853 | 4.4603 | 3.3657 | 3.6925 | 2.8736 | 0.5882 | -41.2% | NOT MET |
| openr1_s1 | 4.6774 | 4.7878 | 4.5467 | 4.5665 | 3.6592 | 0.7823 | -21.8% | NOT MET |
| openr1_s2 | 4.4874 | 4.6894 | 3.5333 | 2.8480 | 2.5519 | 0.5687 | -43.1% | NOT MET |
| wikitext_s0 | 4.6950 | 4.6515 | 4.6808 | 3.4403 | 3.1751 | 0.6763 | -32.4% | NOT MET |
| wikitext_s1 | 4.8407 | 4.7499 | 4.5959 | 4.3007 | 2.5948 | 0.5360 | -46.4% | NOT MET |
| wikitext_s2 | 4.6891 | 4.5396 | 4.5320 | 3.9120 | 3.2641 | 0.6961 | -30.4% | NOT MET |
| **mean** | | | | | | **0.6413** | **-35.9%** | **0/6** |

`L_corpus` at the same 5 checkpoints stays flat in every cell
(openr1 cells: ≈3.53-4.23; wikitext cells: ≈4.62-4.84, no trend) —
expected, since these checkpoints already converged on corpus
modeling during the 20,000-step frozen-bias pretraining this Phase-2
run continues from; familiarization only meaningfully moves `L_query`.
No answer-accuracy field is logged anywhere in the trajectory or
checkpoint records — `loss_corpus`/`loss_query`/`loss_total`/`lr`/
`grad_finite` are the only per-step fields the trainer writes; the
adjudication below therefore uses loss, not accuracy.

**Mechanical adjudication, run against the pre-registered rule exactly
as specified:** **0/6 cells cross the <50% pin.** The closest miss is
wikitext_s1 (ratio 0.536, a 46.4% relative fall); the largest miss is
openr1_s1 (ratio 0.782, only a 21.8% relative fall). Per the letter of
the pre-registered rule, verdict **(a) READOUT-CONSTRUCT-INVALID-
DESPITE-TASK-LEARNING is NOT mechanically licensed** — the pin the
design set for making that strong claim was not met by any cell.
At the same time, the literal condition for verdict **(b)** ("L_query
did NOT fall") **is also false** — `L_query` fell substantially and
uniformly, 21.8-46.4% (mean 35.9%), in all 6/6 cells, not "mixed
across cells" (ruling out verdict (c) as the honest description
either — this is a single uniform pattern, not a split). The
pre-registered trichotomy does not have a bucket for "fell
substantially in every cell, but short of the pin everywhere" —
disclosed here as a genuine gap in the adjudication rule rather than
silently rounded into either bucket. Applying the rule's own spirit
(the pin exists specifically to license the STRONG claim that task
learning was clean and complete while the readout stayed invalid): **since
the strong pin is not met, verdict (a) does not fire; the data are
better described as partial, incomplete task-level engagement
co-occurring with a categorically invalid geometric readout** — closer
in practical consequence to (b) FAMILIARIZATION-TRAINING-NULL (the
recipe has not been shown to fully teach the task in 5,000 steps,
so the mechanism question is not yet cleanly separable from a recipe
adequacy question) than to the strong form of (a), but explicitly
NOT the flat "model failed to learn" reading (b)'s prose implies.
**Verdict: PARTIAL-TASK-LEARNING-BELOW-PIN** (uniform across all 6
cells) — recipe-adequacy question left open, mechanism question not
cleanly resolved by this leg alone.

**A relevant dissociation, already built into the design and worth
surfacing explicitly.** `L_query` is NOT computed from the same
machinery as the Stage-0.5 gate — §16.2.1 (lines ~4275-4298) pins
`L_query` as a **vocab-space cross-entropy through the LM head**
specifically so that it is measured through DIFFERENT machinery than
the gate's **d_state-space cosine readout** (`S_T^h · q_eff`,
premises iii/iv), "explicitly REJECTED" the alternative of training on
the d_state-space objective directly for exactly this reason (would
have "made familiarization directly optimize the quantity the readout
later measures," collapsing the intended independence). That
independence is what makes the following worth reporting: `L_query`
fell substantially in 6/6 cells (in 5/6 cells the terminal value is
below `ln(K=32)=3.4657` — for reference only, not a chance floor this
trainer computes or reports — the sixth, openr1_s1, is close at 3.659),
while the d_state-space gate stayed at the exact `0.0000` floor in
30/30 readings. **This is a real dissociation between the two
already-independent readouts this design built, not a new instrument.**
It does not license the strong (a) verdict (the pin was set
deliberately strict and was not met), but it is directly relevant to
Task 2's question and is reported here rather than only in the
gate-table.

#### 16.15.3 Val-loss band cross-check

Bands (`results/phase2/BANDS_PINNED-Phase2Familiarization.json`) were
computed FROM these same 6 cells' own per-seed val-loss (`mean_ref ±
2·s_ref`, one band per checkpoint) — per §16.2.1's own MINOR-R3-4
disclosure (reproduced verbatim in the chain log), **"OFF passed its
own val-loss gate" is near-tautological and is never evidence of
anything**; it was never going to fail barring a >2σ same-corpus,
cross-seed outlier. Confirmed directly: every one of the 30 own-corpus
val-loss readings falls inside its own pinned `pass_ceiling` (e.g.
step-5000 openr1: per-seed `[2.097, 2.121, 2.244]` vs. ceiling
`2.3116`; step-5000 wikitext: per-seed `[4.388, 4.560, 4.317]` vs.
ceiling `4.6715`). Disclosed per the design's own instruction, not
cited as corroborating anything about the gate's real target
(`per_token`'s/`global`'s own val-loss, which never got measured since
the launch was refused).

**Corpus val-loss health (no divergence):** own-corpus val-loss stayed
in `[1.857, 2.421]` (openr1) / `[4.317, 4.575]` (wikitext) across all
15 own-corpus readings each; cross-corpus (other-corpus) val-loss
stayed in `[6.690, 7.468]` (openr1 cells' wikitext reading) /
`[6.361, 6.618]` (wikitext cells' openr1 reading). No NaN, no
divergence, `grad_finite=True` at all 606 trajectory rows across all
6 cells.

#### 16.15.4 Reading — the triple-null arc

Three structurally different instruments, at three different
training regimes, have now landed on the identical categorical `0.0`
`recovered_frac` floor for the `d_state`-space `S_T^h·q_eff` readout:

1. **Phase 1, marker template, zero-shot** (§15): 0/312 (cell,h)
   readings, 78 cells, 14M-392M params.
2. **Phase-1b, natural-language template, zero-shot** (§16.8):
   0/4 cells (2 candidates × 2 corpora, wikitext-primary +
   openr1-expected-null, both fail identically).
3. **Phase-2, marker template, task-FAMILIARIZED (this section)**:
   0/30 (cell, checkpoint) readings, the checkpoint family that had
   the most reason of the three to succeed — genuinely trained on the
   bind/query structure for 5,000 steps, with `L_query` showing real
   partial task engagement (§16.15.2) at the same time the geometric
   gate stayed flat.

Per §8.4/§16.0's own standing rule ("failure routes to probe-invalid,
not to REFUTE") and §16.2.1's own double-duty framing, a persistent
post-familiarization failure is the single strongest evidence this
program has produced that the `d_state`-space readout construct itself
— not the surface form (arc step 1→2), and not "never task-trained"
(arc step 2→3) — is what is invalid. This is still a PROBE-INVALID
verdict, not a REFUTE of H_LINK-A/H_LINK-B: the keystone question
(does frozen-bias key-geometry stabilization causally improve
in-context multi-hop composition) has still never been asked of the
data by an instrument shown to produce a referent-bearing signal.

#### 16.15.5 §16.6 decision-tree status — what fires next (registered options, none self-authorized)

§16.6 was written before either Path (i) or Path (ii) had reported a
result and does not contain a pre-scripted branch for "Path (ii)'s own
Stage-0.5 gate also refuses" — both instruments the decision tree
named for the REASONING-LINK keystone (Path (i)/Phase-1b, Path
(ii)/Phase-2) have now independently failed the same construct. Path
(iii) (§16.3, the scaling wave) is explicitly NOT a candidate here —
§16.3's own text names it "a capacity-law-paper-track question, not
the keystone question." No option below is self-authorized by this
harvest; this is a PI decision point for the next check-in. Registered
material already in this document that bears on the choice:

- **A different readout construct, vocab-space instead of
  `d_state`-space.** §16.2.1 already built and ran exactly this
  alternative as `L_query` (§16.15.2) — deliberately kept independent
  of the gate by construction. The dissociation observed here (partial
  vocab-space signal, zero geometric signal) is a concrete argument
  for promoting a vocab-space **behavioral contrast** (does the
  familiarized model's own next-token prediction differentiate
  correctly-bound vs. mis-bound entities, scored through the LM head,
  no `d_state` readout at all) to the PRIMARY instrument, rather than
  building a fourth zero-shot/familiarized variant of the same
  `S_T^h·q_eff` construct that has now failed identically three times.
- **Lane closure, with the triple null (§16.15.4) as the publishable
  finding.** Per §8.4/§16.0's own PROBE-INVALID-not-REFUTE routing,
  this is a legitimate, precedented outcome in this document's own
  house convention (mirrors §16.8.3's "no template to fold in" handling
  when Phase-1b also came back null) — the finding would be "three
  structurally different instruments at three training regimes all
  fail to detect `S_T^h·q_eff` structure in this checkpoint family,"
  not "the keystone hypothesis is false."
- **Extend the familiarization recipe** (more steps, a higher
  `λ_fam`, or a longer trajectory past 5,000 steps) to test whether
  §16.15.2's PARTIAL-TASK-LEARNING-BELOW-PIN verdict was a recipe
  ceiling rather than evidence about the readout — this would need its
  own calibration run and its own cost re-derivation before any sweep,
  per this program's own standing "calibration run before a big sweep
  is mandatory" rule; not run here.

No option above is ranked or selected by this harvest.

#### 16.15.6 Realized GPU-h

Box timestamps (UTC, `stat` birth/mtime, `youthful-indigo-turkey`,
2-way parallel on GPUs 0-1):

| Segment | Start (UTC) | End (UTC) | Duration |
|---|---|---|---|
| Chain launch (`97_phase2_reasoning_link_stage_minus1.log` birth) | 01:26:57 | — | — |
| `STAGE05_LAUNCH_GATE_REFUSED` written / chain log's last write | — | 01:45:28.679 | — |
| **Single-launch wall-clock, chain launch → refusal** | 01:26:57 | 01:45:28.679 | **1,111s ≈ 0.3086 hr** |

**Realized GPU-h = wall-clock × 2 GPUs ≈ 0.6172 GPU-h** (matches the
chain's own self-reported `[budget] elapsed=1110s n_gpus=2
projected_gpu_h=0.6167` to within rounding). Decomposition: summed
per-cell `wall_s` (6 cells, training loop only) = 1,967.88s ≈ **0.5466
GPU-h**; the remaining ≈0.0706 GPU-h is Stage -1 self-tests + GPU
smoke tests + the 30 gate-with-null-band passes combined — closely
matching §16.2.3's own priced estimate of ≈0.066 GPU-h for exactly the
30-gate-pass line, a clean cross-check.

**Budget vs. the ≈1.48-12.06 GPU-h bracket.** That bracket (§16.2.3)
prices the FULL 18-cell, 3-arm grid; this leg ran only the OFF arm's
6 cells (1/3 of the grid, pre-registered to run standalone first,
§16.2.1 MINOR-NEW-1's sequencing requirement), so a direct comparison
undercounts by construction. Scaling §16.2.3's own raw-cost line down
to exactly this leg (6 cells × 5,000 steps × 0.04544 s/step = 0.3787
GPU-h training + the 30-gate-pass line's own priced 0.066 GPU-h =
0.4447 GPU-h raw) and applying the SAME registered 5-10× debug-tax
multiplier implies a leg-scoped bracket of ≈2.22-4.45 GPU-h. **Realized
0.6172 GPU-h came in BELOW even that scaled-down low end** — realized/
raw ≈1.39×, well inside the pre-registered conservative margin,
consistent with one clean single launch, no crashes, no relaunches
(mirrors §16.8.5's own Phase-1b "clean run" pattern). Against the full
program's own budget lines, this is a small addition: Phase-1's
≈24.20 GPU-h ceiling (§10), Path (iii)'s ≈21 GPU-h mandatory grid
(§16.3).

#### 16.15.7 Discrepancies, disclosed prominently

1. **The pre-registered adjudication trichotomy (Task 2's own (a)/(b)/
   (c) options) does not cover the observed pattern.** All 6 cells show
   the SAME shape — substantial but sub-pin `L_query` decline — which is
   neither (a)'s strict <50% pin (0/6 met it) nor (b)'s literal "did
   not fall" (false in all 6 cells) nor (c)'s "mixed across cells"
   (it is uniform, not split). §16.15.2 names this explicitly as
   PARTIAL-TASK-LEARNING-BELOW-PIN rather than forcing it into either
   bucket — flagged here as a gap in the adjudication rule as
   registered, for whoever revises it next.
2. **A pre-launch dry run hit a real bug, resolved before the harvested
   run.** `logs/predeploy_phase2_stage_minus1_real.log` (01:10 UTC,
   ~17 minutes before chain launch) shows a Triton kernel failure
   (`ValueError: Pointer argument cannot be accessed from Triton (cpu
   tensor?)`, inside `fla`'s `layer_norm_fwd`) — a device-placement bug
   in an earlier dry-run attempt, not present in the two GPU smoke logs
   that ran immediately before launch (`predeploy_phase2_smoke_gpu_
   positive.log` / `_negative.log`, 01:25-01:26 UTC, both PASSED
   including the deliberate `PHASE2_SMOKE_FORCE_FAIL=1` negative test)
   or anywhere in the actual chain run. Pulled and archived per the
   "pull everything" instruction; disclosed here so it is not mistaken
   for a defect in the harvested result.
3. **No answer-accuracy field exists in the trainer's output** —
   `trajectory` rows carry only `loss_corpus`/`loss_query`/
   `loss_total`/`lr`/`grad_finite`. §16.15.2's adjudication is
   loss-based only, as instructed.

#### 16.15.8 Next steps (not self-launched by this harvest)

Per §16.15.5: PI decision on which registered option (different
readout construct, lane closure, or recipe extension) to pursue for
the REASONING-LINK keystone lane, at the next check-in. GPUs 0-1 are
free in the meantime. Path (iii)'s own scaling-wave track is
unaffected and continues independently (§16.3, not gated on this
result).

---

## 16.16 PATH (iv) — PHASE-2B: vocab-space behavioral-contrast instrument (Rev 2.1, round-3 verify landed, 2026-07-13 — supersedes Rev 2's own "Loader reuse" bullet and its `getsource` substring assertions; the hypothesis/confound-freedom/causal-package-framing/chain-fork/floor-pin/power-sketch/analysis-rewrite/OFF-eval-cache content below is otherwise Rev-2-content-identical, per this document's own "content-identical unless a fix touches it" convention)

This is §16.15.5's own first registered option, promoted: **a vocab-space
behavioral contrast, scored entirely through the LM head, replacing the
`d_state`-space `S_T^h·q_eff` readout that just failed for the third
structurally-different time (§16.15.4's triple-null arc) — not a fourth
variant of that same construct.** The dead readout is not needed here at
all; every piece of machinery this design reuses (the trainer, the 3-arm
checkpoint family, the 5-checkpoint trajectory schedule, the CI/hexachotomy
primitives) was already built and already ran cleanly (§16.2, Rev 5,
CLEARED-FOR-BUILD; §16.6). This section is now Rev 2.1 — still a design
sketch, not a committed build — and per this project's own waterfall
discipline (`CLAUDE.md`) it must clear a fourth, independent spot-check
round before any code changes (§16.17's own forward-pointer, below).

**Rev 1 in one paragraph (full finding→fix trace: §16.17).** Attack round 1
returned NEEDS-REVISION, 5 MAJOR + 3 MINOR, all fixed here, zero GPU spent.
The highest-value finding (MAJOR-1): Rev 0's own §16.16.3 "Build delta" had
registered the right new readout, but nothing in Rev 0 told
`phase2_trajectory_analysis.py` to actually USE it — as specced, that
module's `off_vals`/`arm_vals` still sourced the DEAD `d_state`-space
`recovered_frac` quantity (0.0 in 30/30 §16.15 readings) and its
`stage05_pass_by_c` still read the permanently-FAILED Stage-0.5 gate JSONs,
silently forcing a real monotone-holds-true trajectory into UNRESOLVED-GATE
before a single new (B)-readout number is ever computed. Fixed at §16.16.3
below with an explicit, registered rewrite of the analysis module's own
sourcing, plus a negative Stage-1 test proving the fix has teeth. The other
four MAJORs: the chain script self-refuses on reuse (fixed with a forked
`phase2b_chain.sh`, §16.16.8); the OFF-floor pin was knife-edge and
calibrated on the wrong readout (fixed with a blind, data-derived re-pin,
§16.16.6); the causal paragraph understated its own intervention (fixed at
§16.16.1); and the real-kernel smoke gate under-covered the two new arms
(fixed at §16.16.9).

**Rev 2 in one paragraph (full finding→fix trace: §16.17's own round-2
table, appended below).** Attack round 2 reviewed Rev 1 fresh and returned
NEEDS-REVISION, 3 MAJOR + 2 MINOR, all fixed here, zero GPU spent. The
highest-value finding (MAJOR-R2-1): Rev 1's own registered Stage-1
negative test proved only that `phase2_hexachotomy.classify_trajectory`
classifies a hand-built `holds_by_c` pattern correctly — a fact already
true before Rev 1 existed — never that the REWRITTEN
`build_holds_and_gate_by_checkpoint` (`phase2_trajectory_analysis.py`
L98-136) actually produces that pattern from the new readout; the test
would pass whether or not MAJOR-1's own rewrite ever landed. Fixed at
§16.16.3 below by re-registering the test one layer down: a stubbed
`eval_query_loss_heldout` feeding the REAL `delta_ci_n3`→`det`→`holds`
chain inside the rewritten function itself, plus a structural grep-level
assertion that the dead `recovered_frac`/`gate_json_path_for` sourcing is
actually gone from the module. The other two MAJORs: eval-(B)'s own
forward path was left adjudication-free against the parent document's
broadly-worded MAJOR-NEW-5 surgery rule, risking a silent habit-paste that
would revert the whole causal claim (fixed with an explicit surgery-scope
pin, §16.16.3); and the "360 passes, computed once, consumed twice" cost
claim was checked against `analyze_corpus`'s own per-arm iteration and
found to under-count by ~33% absent a cache (fixed with a registered
OFF-eval cache, both the cached (360) and honest uncached (≈480) figures
now disclosed, §16.16.8). Two MINORs: `eval_query_loss_heldout`'s own
`batch_size` is now pinned at the codebase's own `=16` convention rather
than left as an unpinned parameter; the smoke-cost line's stale ≈0.01
GPU-h figure (superseded by Rev 1's own disclosed ≈0.02-0.03 GPU-h,
MAJOR-5's three-arm suite) is corrected in the actual arithmetic, not just
narrated beside it.

**Rev 2.1 in one paragraph (full finding→fix trace: §16.17's own round-3
table, appended below).** A focused round-3 VERIFY pass on Rev 2 — checking
round-2's own 5 fixes land correctly against the real code they
cite, per §16.16.11's forward pointer — returned NEEDS-REVISION, 1 MAJOR +
1 MINOR, both surgical, zero GPU spent. **MAJOR-R3-1:** the registered
Stage-1 test (item 3(i)) stubbed `load_init_checkpoint_strict` as the
eval-model loader, but that function (`lm_pretrain_rd.py` L1803) takes an
ALREADY-CONSTRUCTED model and mutates it in place — it cannot be stubbed
to deliver a sentinel, and the actual model-construction step lives one
layer up, a separate `torch.load(...)['config']` antecedent
(`phase2_familiarization_train.py` L408→L421). Fixed at §16.16.3 below by
registering a single, new-in-this-rewrite seam, `phase2b_load_eval_model`,
that reproduces the SAME L408→L421 double-defense pattern internally and
is the ONLY thing the rewritten `killer_prediction_readout` calls to load
an eval-time model — production keeps the strict double-defense path
unconditionally, never the laxer `reasoning_link_probe.load_checkpoint`.
**MINOR-R3-1:** the `getsource` substring-absence assertions (the
`frozen_bias_surgery`/`recovered_frac`/`gate_json_path_for` checks,
§16.16.3 items and §16.16.9 item (d)) are fragile to a well-meaning
docstring or comment mentioning the same identifier in prose. Fixed by
re-pinning all four as AST-level checks (parse with `ast`, walk
`Call`/`Attribute`/`Name` nodes for the identifier, ignoring
docstrings/comments/string literals) plus a house convention that in-code
prose should cite the fix by section number rather than the literal
retired identifier where feasible.

### 16.16.1 Hypothesis and the causal logic, stated precisely

**One-sentence hypothesis:** frozen-bias key-geometry stabilization
(`arm ∈ {per_token, global}` vs. `off`) causally changes how fast and how
completely a DeltaNet LM acquires in-context multi-hop bind/query
composition during task familiarization, as measured by vocab-space
cross-entropy on held-out episodes.

**Causal logic (the keystone question, re-derived for this instrument;
REVISED Rev 1, attack-round-1 MAJOR — the treatment is the arm's ENTIRE
causal package, not a one-time pretraining-era divergence).** Rev 0's own
"exactly ONE thing at the moment familiarization begins" framing understated
the intervention. Checked directly against `lm_pretrain_rd.py`
(`DeltaNetLMMixer.forward`, L854-857): `if self.frozen_bias_arm != "off": k
= apply_frozen_bias_blend(...)` is UNCONDITIONAL on training-vs-eval mode or
on which phase (pretraining vs. familiarization) is running — it fires on
EVERY forward pass, for as long as the loaded checkpoint's own baked
`frozen_bias_arm` config field (`DeltaNetLM.config()`, L1174; see §16.16.9
Stage −1 item (c) below) reads `per_token`/`global`. The registered
intervention this design tests is therefore the arm's **entire causal
package**: (a) the ORIGINAL 20,000-step pretraining's own arm-specific
weight divergence (the blend was already active throughout that run, not
merely "baked in" as a static table afterward), PLUS (b) the SAME blend
mechanism continuing to fire on every one of the 5,000 familiarization steps
that follow — both pieces are inseparable in what these checkpoints
actually are (§16.2.1's own "Base checkpoints" paragraph: "continue training
from the archived Leg-A frozen-bias checkpoints at their final [step-20,000]
state"; that paragraph never claims the blend is switched off for
familiarization, and the code confirms it is not). Familiarization then
continues training ALL THREE arms with the IDENTICAL recipe otherwise
(§16.2.1: same corpus mix, same `λ_fam=1.0`, same H_train=(1,2)/H_test=(3,4)
split, same 5,000-step budget, same trajectory-checkpoint schedule, same
optimizer/LR schedule shape). If `per_token`'s or `global`'s own
task-learning TRAJECTORY (not a single terminal number — §16.2.1's own
trajectory-not-terminal design choice, MAJOR-1) diverges from `off`'s, at a
checkpoint where the comparison is independently certified interpretable
(§16.16.6 below), that divergence is attributable to the arm's own causal
package as a whole — pretraining-era divergence AND the persistently-applied
familiarization-time blend, taken together — not to a confound, PROVIDED the
verification in §16.16.2 holds. This IS exactly H_LINK-A's own causal claim
as registered (§16.2, "What it tests"; §16.2.1's own "Base checkpoints"
paragraph), now asked with an instrument that has actual signal (§16.15.2's
dissociation: `L_query` moved 21.8-46.4% while the geometric gate stayed at
0.0000) rather than one that has never once produced a referent-bearing
reading in three attempts (§16.15.4). **Named, not built here:** an
ISOLATED familiarization-time-only contrast — all three arms initialized
from the SAME `off` checkpoint, with `frozen_bias_arm` toggled only for the
5,000-step familiarization window — would ask a narrower, cleaner question
(does the blend help ACQUISITION specifically, independent of any
pretraining-era divergence) but is a categorically different intervention
requiring a fresh set of init checkpoints (none of which exist in this
program's archive today); registered as a distinct follow-on design, out of
scope for this Rev.

### 16.16.2 Confound-freedom, verified against the real code — not asserted

**Claim under test:** "arms differ ONLY in the frozen-bias intervention;
everything else — init checkpoints, data order, eval batches — is shared."
Checked directly against `phase2_seed` (`phase2_familiarization_train.py`
L196-222) and its own three call sites this design's new machinery adds to
(train_corpus/train_episode already existed; the new eval kinds are
registered in §16.16.3 below). **Verdict: TRUE at the level of
CONSTRUCTION (the generative protocol), FALSE at the level of REALIZATION
(the literal token sequences drawn) — a distinction this section states
explicitly because the task's own framing ("share... data order... eval
batches") elides it, and an attack round would otherwise find it first.**

1. **Init checkpoints.** Each arm's own checkpoint is the archived Leg-A
   frozen-bias step-20,000 weights FOR THAT ARM (`off`/`per_token`/`global`),
   at the SAME corpus/seed correspondence, all three already cleared §7.2's
   val-loss-matching gate (§16.2.1's "Base checkpoints" paragraph). "Shared"
   here means the PROTOCOL that produced them is identical across arms
   (same corpus, same step budget, same seed-to-seed correspondence, same
   matching gate) — not that arms load the same file, which would be
   incoherent (the arm identity IS which frozen-bias table, if any, is
   baked into that checkpoint's own pretraining).
2. **Training data order — NOT literally identical, and this is not a
   confound.** `phase2_seed(kind, arm, corpus, ckpt_seed, k,
   checkpoint_step)` is a TRUE mixed-radix seed (L196-222) with `arm_idx`
   (`_ARM_INDEX = {"off": 0, "per_token": 1, "global": 2}`, L157) as one of
   its six stacked digits. Every training-draw call —
   `gen_corpus = ...manual_seed(phase2_seed("train_corpus", arm, corpus,
   ...))` (L446) and `gen_episode = ...manual_seed(phase2_seed(
   "train_episode", arm, corpus, ...))` (L448) — therefore produces a
   DIFFERENT corpus-window order and a DIFFERENT episode draw for
   `off`/`per_token`/`global` at the SAME `(corpus, ckpt_seed)`. This is
   real and disclosed here explicitly (the task's own phrasing implied it
   might be literal). It is not a confound because (a) arm assignment does
   not select or bias which RNG stream a given arm receives — the mapping
   is a fixed, non-adaptive digit in a positional numeral system, computed
   before any result exists to condition on; (b) every arm draws from the
   IDENTICAL underlying distribution (same corpus files, same
   `sample_batch_rd` hyperparameters, same entity pools, same hop-depth
   split) — only the realized draw differs, not its distribution; (c) the
   3-`ckpt_seed` CI machinery this design reuses (`delta_ci_n3`) is already
   built to absorb seed-level noise from exactly this kind of source. Net
   effect: this is an unbiased VARIANCE contributor, not a bias source —
   but it does mean this is NOT a common-random-numbers/paired design at
   the TRAINING level, and §16.16.4's power sketch prices that honestly.
   **Changing this would touch the ALREADY-COMPLETED OFF arm's own training
   path** (whose 6 checkpoints this design reuses, §16.16.8) — out of scope
   here; registered as a standing property of the existing, audited
   trainer, not a defect this Rev fixes.
3. **Eval batches — same underlying fact, but THIS design gets to choose,
   because the eval machinery is new.** The existing `eval_killer`/
   `eval_gate_self`/`eval_gate_null` seed kinds (L159-160) are ALSO
   `arm`-keyed, so the OLD (now-dead) geometric readout never scored arms on
   identical held-out episodes either. Since §16.16.3's new frozen-checkpoint
   eval-`L_query` pass is genuinely new code, it does not inherit this
   property by default — **registered build choice: pin it to NOT inherit
   the arm-keying**, making the arm contrast a real paired comparison on
   identical held-out episodes (below).

### 16.16.3 Primary readout — adjudicated

**The choice, stated as the task requires.** (A) **Training-loop `L_query`**
— already logged, zero marginal cost, but computed on TRAIN-POOL episodes
(`use_heldout_entities=False`, `query_loss_forward`'s own default,
L263-272) with training-batch noise (`batch_size×Q=2` per step — the
`n_query=2` pin, §16.2.1 MAJOR-R3-1b) at whatever step the trajectory
logger happens to land on. (B) **A NEW frozen-checkpoint eval-`L_query`
pass** on HELD-OUT episodes, at the SAME `Q=K` convention the (now dead)
Stage-0.5 gate already used (`familiarization_gate_episode_config`,
`n_query=None → Q=K`, L239-256), run on the FROZEN `.pt` weights at each of
the 5 trajectory checkpoints, for every one of the 18 cells (6 reused OFF +
12 new).

**Recommendation: (B) is PRIMARY, (A) is corroborating.** Three reasons,
all concrete: (1) (A)'s own §16.15.2 numbers are exactly what this design
inherits as its between-seed noise floor (§16.16.4) — that noise is
partly REAL trajectory divergence and partly minibatch measurement noise
(32 queries/step vs. (B)'s `Q=K=32` per EPISODE at a full eval batch,
`batch_size×K`, an order of magnitude more queries per reading); using the
noisier one as primary would understate power further. (2) (A) is drawn
from the TRAIN pool, the exact contamination §16.2.1's own MAJOR-1 build
fix (`use_heldout_entities=True`) was written to close for every other
readout in this design — (A) alone would silently reopen it. (3) (B) is
structurally the SAME machinery `query_loss_forward` already validates
(§16.2.1's "Corrected definition" paragraph: ONE forward call over
`main_concat`, `initial_states=None`, the causally-verified, ALREADY-
Stage−1-tested mode) — reused on frozen weights instead of during the
backward pass, zero new forward-pass logic, only a new caller.

**Build delta (concrete, scoped):**
- A new function, e.g. `eval_query_loss_heldout(model, K, hop_set,
  ckpt_seed_or_pool, batch_size, device) -> float`, wrapping
  `query_loss_forward` with `use_heldout_entities=True` and a
  CALLER-supplied `hop_set` (§16.16.7 needs `hop_set=(3,4)`;
  `query_loss_forward`'s own `hop_set=H_TRAIN` is currently HARDCODED at
  the call site, L271 — this is a real, small, disclosed parameterization
  task, not a design assumption). **`batch_size` pinned to a default of
  `16` (Rev 2, attack-round-2 MINOR-R2-1)** — the codebase's own existing
  convention (`killer_prediction_readout`'s own `batch_size: int = 16`
  default and `build_holds_and_gate_by_checkpoint`'s own same default,
  `phase2_trajectory_analysis.py` L78, L99), now REGISTERED explicitly on
  this new function's own signature (`batch_size: int = 16`) rather than
  left unpinned and silently inherited from whatever a future caller
  happens to pass.
- Two new `phase2_seed` kinds, extending `_KIND_OFFSET` (currently
  `{"train_corpus":0, "train_episode":1, "eval_val":2, "eval_gate_null":3,
  "eval_gate_self":4, "eval_killer":5}`, L159-160) with
  `"eval_lquery_heldout": 6` (primary, `hop_set=(1,2)`) and
  `"eval_lquery_ood": 7` (secondary, `hop_set=(3,4)`) — `phase2_seed`'s own
  signature is UNCHANGED; this is two new dict entries, zero new digit-width
  arithmetic (kind is the outermost/last digit, no `_MAX_*` constant
  depends on how many kinds exist).
- **Pairing device (§16.16.2 item 3): every caller of these two new kinds
  passes the LITERAL STRING `"off"` as the `arm` argument, regardless of
  which arm's checkpoint is actually being scored.** `phase2_seed`'s own
  formula is untouched; this is a call-site convention (mirrors §16.2.1's
  own repeated "zero new code, a config-value choice" framing) that makes
  the resulting seed IDENTICAL across all 3 arms for a given `(corpus,
  ckpt_seed, K, checkpoint_step)` — the deliberate point of this readout is
  a paired comparison on the SAME held-out episodes, not three independent
  draws.
- **Loader — a single-seam helper, `phase2b_load_eval_model` (Rev 2.1 fix,
  round-3 verify MAJOR-R3-1; supersedes Rev 0-2's own "Loader reuse:
  `load_init_checkpoint_strict`... loads each of the 90 already-existing
  `.pt` files... no new loading code" bullet, which named a function that
  cannot deliver what a Stage-1 stub needs).** Verified directly against
  the real code: `load_init_checkpoint_strict` (`lm_pretrain_rd.py` L1803)
  takes an ALREADY-CONSTRUCTED `model` argument, mutates its weights in
  place via `load_state_dict`, and returns the ckpt dict — it does not
  build a model from a bare checkpoint path, so it cannot itself be
  stubbed to "return a sentinel" the way a loader normally would, and
  calling it for real requires an actual `.pt` file. The model is built
  one layer up, from a SEPARATE `torch.load(...)['config']` antecedent
  (`phase2_familiarization_train.py`'s own `run_familiarization_cell`,
  L408: `model = DeltaNetLM(**torch.load(init_checkpoint,
  map_location=device)["config"]).to(device)`), immediately followed
  (L421, non-resume branch) by `load_init_checkpoint_strict(model,
  init_checkpoint, device)` — construction, then strict load, the
  codebase's own established double-defense pattern (config-equality
  assert + `strict=True load_state_dict`, both inside
  `load_init_checkpoint_strict` itself, `lm_pretrain_rd.py` L1803-1820ish).
  **Fix, a new function in `phase2_trajectory_analysis.py` (the analysis
  rewrite this item's parent paragraph already registers):**
  ```python
  import lm_pretrain_rd as lpr  # same direct-import convention phase2_stage_minus1.py
                                 # already uses (L46) -- new to this module

  def phase2b_load_eval_model(ckpt_path: str, device: str) -> lpr.DeltaNetLM:
      config = torch.load(ckpt_path, map_location=device)["config"]  # mirrors
                                       # phase2_familiarization_train.py L408
      model = lpr.DeltaNetLM(**config).to(device)
      lpr.load_init_checkpoint_strict(model, ckpt_path, device)      # mirrors
                                       # L421; lm_pretrain_rd.py L1803's own
                                       # config-equality assert + strict=True
                                       # load_state_dict double-defense
      model.eval()
      return model
  ```
  This is the ONE seam — in the sense of one HELPER, with exactly TWO
  registered callers [harmonized at Rev 2.2, round-4 MINOR-R4-1: the
  original "once per (arm-or-off, checkpoint) ... no other call site"
  wording contradicted MAJOR-R2-3's own cache flow]: (caller 1) the
  REWRITTEN `killer_prediction_readout` (item 1 below) calls
  `phase2b_load_eval_model(ckpt_path, device)` once per (non-off arm,
  checkpoint) and passes the resulting `model` into
  `eval_query_loss_heldout` — its OFF half never loads a model, it reads
  `off_lquery_cache-Phase2b.json`; (caller 2) §16.16.8's chain-step-3
  OFF-eval cache builder, which loads each of the 6 reused OFF
  checkpoints through the SAME helper to populate that cache. No third
  call site in this design's own build delta constructs or loads an
  eval-time model. **Production keeps the STRICT double-defense path
  unconditionally — no silent laxening.**
  The auditor's own noted alternative, switching to
  `reasoning_link_probe.load_checkpoint` (`reasoning_link_probe.py`
  L705-713 — `DeltaNetLM(**ckpt["config"])` then a bare `load_state_dict`,
  no explicit config-equality assert; the single-layer pattern
  `killer_prediction_readout`'s OWN pre-rewrite body actually uses today,
  via `rlp.run_cell`'s internal `load_checkpoint` call), is explicitly
  REJECTED — it would silently drop the config-equality half of the
  double-defense. The Stage-1 test (item 3(i), below) stubs
  `phase2b_load_eval_model` itself — this single seam — with a sentinel
  carrying `(arm, checkpoint_step)` as plain attributes; it never stubs
  `load_init_checkpoint_strict` directly, and touches no real `.pt` file
  or GPU.

**Surgery-mode scope, pinned (Rev 2 fix, attack-round-2 MAJOR-R2-2) —
eval-(B) runs with NO surgery override; MAJOR-NEW-5's rule does not apply
here.** Checked directly against `query_loss_forward`
(`phase2_familiarization_train.py` L263-285): its own forward pass,
`logits = model(concat_tokens, initial_states=None, step=step)` (L279),
is the model's NATIVE call — no `frozen_bias_surgery` context manager
wraps it, on this path or on `eval_query_loss_heldout`'s own reuse of it
(this section's "Build delta" above: `eval_query_loss_heldout` wraps
`query_loss_forward` directly, adding no surgery of its own). This is
REQUIRED, not an oversight to close: §16.16.1's own causal-package framing
(Rev 1) registers the arm's ENTIRE causal package — pretraining-era weight
divergence AND the blend continuing to fire on every forward pass — as
the thing under test, and the blend firing during eval-(B)'s own scoring
pass IS part of that package, exactly as it is during familiarization
training itself (§16.16.1: `apply_frozen_bias_blend` is unconditional
whenever `frozen_bias_arm != "off"`, `lm_pretrain_rd.py` L854-857, on
EVERY forward pass regardless of train/eval mode). Forcing the blend off
at eval-(B) time would silently narrow the causal claim to the isolated,
un-built follow-on §16.16.1 explicitly named out of scope, while still
being reported under the whole-package framing. **§5.2a's/§16.2.1's own
MAJOR-NEW-5 rule ("before computing `rec@0.9` [or any Stage-0.5 gate
quantity] at ANY trajectory checkpoint... wrap the scoring forward pass in
`frozen_bias_surgery(model, force_off=True)`", this document's own
MAJOR-NEW-5 fix-map row) is RETIRED-GATE-ONLY for Phase-2b — it applies to
the dead `d_state`-space Stage-0.5 readout (`rec@0.9`), which this
section's own "`stage05_pass_by_c` is retired, not computed" fix (below)
already drops entirely, never to eval-(B)'s `L_query` readout.** A
builder pattern-matching on MAJOR-NEW-5's own broad "wrap scoring passes
in force_off surgery" phrasing, out of habit, could paste the wrapper onto
`eval_query_loss_heldout` and silently revert this design's own causal
claim without any test catching it (a surgery-wrapped and a
native-forward eval-(B) pass both return a plausible-looking float) —
closed here by a mandatory Stage −1 assertion (§16.16.9, new item): after
constructing `eval_query_loss_heldout`, inspect its own source (e.g.
`inspect.getsource(eval_query_loss_heldout)` plus the `query_loss_forward`
it wraps) and assert `frozen_bias_surgery` is never REFERENCED as a real
identifier in either — a mechanical, not a trusted, guarantee that eval-(B)
measures the arm's whole causal package, not a surgically-isolated slice
of it. **Assertion mechanism re-pinned at AST level (Rev 2.1 fix,
round-3 verify MINOR-R3-1) — supersedes the plain `"frozen_bias_surgery"
not in inspect.getsource(...)` substring check Rev 2 registered, which
would false-positive-fail on a well-meaning docstring or comment
mentioning the identifier in prose (exactly the kind of disclosure this
document's own house style favors) despite the code itself being
correct.** A single helper, registered once in `phase2_stage_minus1.py`
(the Stage −1 CPU-stub suite this and item 3(ii)'s own tests both live
in, §16.16.9's first bullet) and reused by every `getsource`-style
assertion in this design:
  ```python
  import ast

  def _references(func_or_module, identifier: str) -> bool:
      """True iff `identifier` appears as a real Call/Attribute/Name node
      OR as a dict-subscript key (e.g. r["per_h"][h]["recovered_frac"])
      in `func_or_module`'s own source -- never merely inside a docstring
      or comment. `ast.parse` drops comments outright, and a docstring is
      a bare Expr(Constant(str)) node -- never a Subscript slice -- so
      prose mentions stay immune. [Rev 2.2, round-4 MAJOR-R4-1: the
      Subscript clause is LOAD-BEARING -- the dead sourcing this check
      exists to detect is a dict-key string literal, which a
      Name/Attribute-only walk ignores; the round-4 auditor demonstrated
      the two-clause version passes vacuously against the pre-rewrite
      module.]"""
      tree = ast.parse(inspect.getsource(func_or_module))
      return any(
          (isinstance(n, ast.Name) and n.id == identifier)
          or (isinstance(n, ast.Attribute) and n.attr == identifier)
          or (isinstance(n, ast.Subscript)
              and isinstance(n.slice, ast.Constant)
              and n.slice.value == identifier)
          for n in ast.walk(tree)
      )
  ```
  **Empirical teeth-run, REGISTERED as a Stage −1 obligation (round-4
  MAJOR-R4-1's own fix, per the standing "run the negative test to
  completion" rule):** before the analysis rewrite lands,
  `_references(phase2_trajectory_analysis, "recovered_frac")` MUST
  return `True` against the pre-rewrite module (proving the check sees
  the dead dict-key sourcing at L116-117); after the rewrite it MUST
  return `False`. Both runs are recorded in the Stage −1 output. The
  round-4 auditor ran the two-clause version against the live module
  and got `False` pre-rewrite — the vacuous-pass this clause closes.
  (Disclosed, outside the threat model: an aliased import
  `from rlp import frozen_bias_surgery as fbs` would dodge the check —
  that is deliberate evasion, not the accidental-habit-paste class this
  assertion defends against.)
  This paragraph's own assertion becomes `assert not _references(
  eval_query_loss_heldout, "frozen_bias_surgery")` and the same check
  against `query_loss_forward` (§16.16.9 item (d), re-pinned to match).
  **House convention, registered alongside (one line, not a new
  mechanism):** in-code prose SHOULD cite a retired-quantity or
  surgery-scope fix by SECTION NUMBER (e.g. "§16.16.3's MAJOR-1 fix")
  rather than spelling out the literal identifier, where feasible — this
  keeps a future disclosure comment from tripping the same class of
  check `_references` is built to be immune to at the CALL/ATTRIBUTE/NAME
  level, but a docstring can still trivially defeat a substring check by
  accident.

**Analysis-module rewrite — registered build task (Rev 1, attack-round-1
MAJOR-1, the highest-value finding this round found).** As specced through
Rev 0, the "Build delta" above registers `eval_query_loss_heldout` but never
tells `phase2_trajectory_analysis.py` to actually call it. Verified directly
against that module (`phase2_trajectory_analysis.py`): `killer_prediction_
readout` (L77-95) still calls `rlp.run_cell(..., surgery="off", ...)` and
`build_holds_and_gate_by_checkpoint` (L98-136) still reads `off_r["per_h"]
[h]["recovered_frac"]` / `arm_r["per_h"][h]["recovered_frac"]` (L116-117)
into `off_vals`/`arm_vals` — the DEAD `d_state`-space quantity that read
exactly 0.0 in 30/30 §16.15.1 readings. The same function's own
`stage05_pass_by_c` (L108, populated L125-134) still reads a per-checkpoint
gate JSON via `gate_json_path_for`/`phase2_gate_enforce.gate_verdict`, gate
JSONs that are permanently FAIL per §16.15.1's own 30/30 REFUSED table. Left
as specced, a clean Phase-2b run would compute zero real (B)-readout numbers
and would silently route ANY monotone-holds-true trajectory into
`phase2_hexachotomy.classify_trajectory`'s own UNRESOLVED-GATE branch
(L136-152) before that trajectory is ever actually evaluated on the new
readout — burying a real PERSISTENT finding as a gate artifact of an
instrument this Phase already retired. Registered fix, three parts:

1. **`killer_prediction_readout` is replaced** (same call-site shape feeding
   `build_holds_and_gate_by_checkpoint`, new body) by a function that calls
   `eval_query_loss_heldout(model, K, hop_set=(1,2), ...)` (this section's
   own new function) on the frozen checkpoint for both `off` and `arm`,
   loading each `model` via `phase2b_load_eval_model(ckpt_path, device)`
   (this section's own "Loader" bullet, the single seam — Rev 2.1 fix,
   round-3 verify MAJOR-R3-1), and returning `L_query` floats directly. `off_vals`/`arm_vals` (the L112-117
   accumulation loop) are populated from these `L_query` floats, never from
   `recovered_frac`. `delta_ci_n3` is called with the §16.16.5 sign
   convention (`delta_ci_n3(off_vals, arm_vals)`, positive = arm's loss
   lower = arm helps) — the same reversal §16.16.5 already registers, now
   wired to the actual call site that needs it. **OFF-eval cache,
   registered (Rev 2 fix, attack-round-2 MAJOR-R2-3) — the `off`-half of
   this replacement's own two `eval_query_loss_heldout` calls reads from a
   precomputed cache rather than calling fresh every time.** Verified
   directly against the call structure this replaces `killer_prediction_
   readout` into: `analyze_corpus` (L139-142) builds `per_arm` via `{arm:
   build_holds_and_gate_by_checkpoint(ckpt_dir, arm, corpus, ...) for arm
   in ARMS_NON_OFF}` (`ARMS_NON_OFF = ("global", "per_token")`, L60) — TWO
   separate calls, each of which (per this item's own replacement,
   unchanged in this respect) independently re-scores `off` inside its own
   branch. Left as Rev 1 registered it, `off`'s own `eval_query_loss_
   heldout` calls fire once inside the `global` branch AND once again
   inside the `per_token` branch — real, silent duplication, not a
   mislabeling (§16.16.8's own corrected cost line, below). **Fix:** the
   pre-launch cache §16.16.8's "New first Python step" (item 3 below)
   writes — ALL 5 checkpoints × both K's × both hop-sets for the 6 reused
   OFF cells, not merely the `c∈{250,5000}` floor-gate pair — is promoted
   from an informal "consumed twice" convention to a literal, committed
   JSON handoff (`off_lquery_cache-Phase2b.json`) that THIS replacement
   function reads `off`'s own `L_query` values FROM, keyed on `(corpus,
   ckpt_seed, K, checkpoint_step, hop_set)`, instead of calling
   `eval_query_loss_heldout(off_model, ...)` itself. This is pure economy,
   not a new independence assumption: §16.16.2 item 3's own pairing device
   already makes `off`'s seed identical regardless of which arm branch
   asks for it, so the cached value IS the value a fresh call would
   return, deterministically.
2. **`stage05_pass_by_c` is retired, not computed.**
   `build_holds_and_gate_by_checkpoint` sets `stage05_pass_by_c = {c: True
   for c in phx.CHECKPOINTS}` unconditionally — mirroring
   `phase2_hexachotomy.totality_check`'s own `always_pass_gate = {c: True
   for c in CHECKPOINTS}` (L204) verbatim — and the `gate_json_path_for`/
   `pge.gate_verdict` call (L125-134) is DELETED, not merely bypassed, so a
   future reader cannot mistake a retained-but-dead code path for a live
   check. Rationale, stated once here: the per-checkpoint Stage-0.5 gate was
   a validity check on the OLD `d_state`-space readout; that readout is
   gone, and its replacement validity check is §16.16.6's OFF-floor gate,
   evaluated ONCE per corpus before launch, not per-checkpoint inside the
   classifier — the Stage-0.5 gate is RETIRED for Phase-2b.
3. **New Stage-1 negative test — RE-REGISTERED one layer down (Rev 2 fix,
   attack-round-2 MAJOR-R2-1; supersedes Rev 1's own version of this item,
   which only exercised `phase2_hexachotomy.classify_trajectory` on a
   hand-built `holds_by_c` dict — a fact already true before this fix
   existed, since `classify_trajectory` was never the broken piece;
   `build_holds_and_gate_by_checkpoint` itself (`phase2_trajectory_
   analysis.py` L98-136) takes no `holds_by_c` parameter at all — it
   COMPUTES `holds_by_c` internally, so a test that never calls it cannot
   prove the rewrite happened).** Two parts, both required, both in the
   SAME test:
   - **(i) Behavioral part.** Monkeypatch `eval_query_loss_heldout` (the
     seam item 1 above wires the replacement `killer_prediction_readout`
     through) to return controlled, per-(arm-or-off, checkpoint, K, seed)
     synthetic `L_query` floats from a small fixed lookup table — never a
     constant, so the reduction has real per-cell content to fold over —
     together with a stub of the replacement's own checkpoint-load call,
     **`phase2b_load_eval_model`, the single-seam loader (Rev 2.1 fix,
     round-3 verify MAJOR-R3-1; supersedes stubbing
     `load_init_checkpoint_strict` directly, which this round's own verify
     found unbuildable — that function mutates an already-constructed
     model in place rather than building one from a bare path, so it
     cannot itself deliver a sentinel; see §16.16.3's "Loader" bullet
     above for the full citation trace)** — returning a trivial sentinel
     object carrying `(arm, checkpoint_step)` as plain attributes, so the
     `eval_query_loss_heldout` stub can look up the right table entry
     without touching a real `.pt` file or GPU. Engineer the table so
     that, run through the REAL `delta_ci_n3` → `phx.det` → `phx.holds`
     chain INSIDE the rewritten `build_holds_and_gate_by_checkpoint` (not
     hand-assembled and fed to `classify_trajectory` directly): at K=32,
     `off_vals`/`arm_vals` (3 seeds each) are statistically
     indistinguishable at `c=250` (CI straddles zero, `det32=False`) and
     cleanly separated, arm lower, at `c∈{500,1000,2500,5000}` (CI
     excludes zero on the positive side, `det32=True`); at K=20, the same
     values are indistinguishable at EVERY checkpoint (`det20=False`
     throughout) — reproducing the exact `holds_by_c=
     {250:False,500:True,1000:True,2500:True,5000:True}` (`FTTTT`) pattern
     Rev 1's own version of this test named PERSISTENT, now PRODUCED by
     the real formula chain rather than asserted by construction. Assert
     the resulting classification is `PERSISTENT`, never
     `UNRESOLVED-GATE` — the same outcome Rev 1 checked, now actually
     exercising the rewritten function that is supposed to produce it.
   - **(ii) Structural part, exact assertions pinned (not left to the
     builder's judgment); re-pinned at AST level (Rev 2.1 fix, round-3
     verify MINOR-R3-1, same fragility class and same `_references`
     helper as the surgery-mode assertion, §16.16.3's "Surgery-mode
     scope" paragraph above — supersedes the plain substring checks Rev 2
     registered, which would false-positive-fail on a well-meaning
     docstring or comment mentioning either identifier in prose):**
     `not _references(phase2_trajectory_analysis, "recovered_frac")`
     (whole-module check — proves the dead `d_state`-space quantity is
     gone, not merely shadowed by an unexercised branch) AND
     `not _references(phase2_trajectory_analysis.
     build_holds_and_gate_by_checkpoint, "gate_json_path_for")`
     (function-scoped check — proves the rewritten function's own body no
     longer calls the dead gate-JSON reader, whether or not the helper
     itself is deleted elsewhere in the module). A test that passes (i)
     but fails (ii) would mean the dead sourcing was routed AROUND rather
     than removed — exactly the gap this re-registration exists to close.

### 16.16.4 Power sketch — the detectable effect size at n=3 seeds, honestly derived

**Between-seed σ proxy, from real numbers already on disk (§16.15.2's
terminal training-loop `L_query`, the 6 OFF cells):** openr1
`{2.8736, 3.6592, 2.5519}`, wikitext `{3.1751, 2.5948, 3.2641}`. Two pooling
choices, both computed directly (n−1 denominators, matching
`delta_ci_n3`'s own variance convention): pooled-within-corpus
`σ≈0.478` (average of the two corpora's own n=3 sample variances); pooled-
across-all-6 `σ≈0.427` (ignoring corpus structure). Both land in the same
`≈0.43-0.48` band — the qualitative conclusion below is insensitive to
which is used. **This proxy is disclosed as CONSERVATIVE (an
over-estimate):** it is computed from readout (A) — the noisier
training-loop numbers §16.16.3 rejected as primary specifically for
their noise — not the cleaner `Q=K` frozen-checkpoint readout (B) this
design actually registers; (B)'s own true between-seed σ is expected to be
smaller (more queries per reading, `batch_size×K≈512` vs. `≈32`) but is not
knowable until real (B) numbers exist.

**CI half-width at n=3 seeds, using `delta_ci_n3`'s own pinned formula
(`t(2,.975)=4.303`).** Assuming the ARM cell's own between-seed noise is
comparable in magnitude to, and statistically independent of, `off`'s
(a necessary assumption — no real `per_token`/`global` (B)-readout numbers
exist yet to check it against; independence follows from §16.16.2 item 2's
own finding that arm/off draw from separate RNG streams and start from
already-arm-diverged checkpoints): `Var(Δ_seed) ≈ 2σ²`. At `σ=0.48`:
`sd_Δ=0.679`, `SE=sd_Δ/√3=0.392`, `half_width=4.303×0.392≈1.69`. At
`σ=0.43`: `half_width≈4.303×(0.43×√2/√3)≈1.51`. **Detectable `|mean Δ|` at
n=3 seeds: roughly 1.5-1.7 loss units.**

**The sobering comparison, stated plainly.** The OFF arm's own FULL
5,000-step familiarization effect (mean `L_query@250 − L_query@5000`
across the 6 cells) is `4.7125 − 3.0198 ≈ 1.69` — **the SAME order of
magnitude as the minimum detectable arm-vs-arm effect.** At n=3 seeds, this
design is well-powered only for an arm effect roughly as large as the
ENTIRE observed familiarization effect itself; a more plausible, modest
arm effect (say 10-50% of the total learning effect, `≈0.17-0.85` loss
units) will very likely land in the pre-registered **UNRESOLVED** bucket
(§16.2.1's own "insufficiently powered... too few seeds/too much variance
to say anything" reading) rather than resolving cleanly either way. **This
is disclosed prominently, not softened**, because it directly qualifies
requirement 6's "no arm effect w/ adequate power" framing below (§16.16.9):
a null read from this instrument at n=3 seeds may be genuinely
underpowered, not a clean negative — the pairing device (§16.16.2 item 3)
and the larger `Q=K` per-reading sample (B) both work to shrink this
number below the pessimistic estimate above, but by how much is an
empirical question this wave itself will answer, not one this design can
resolve on paper. **No design change is proposed to fix this** — the 3
`ckpt_seed`s are fixed by the already-completed, reused Leg-A/OFF
checkpoints (§16.16.8); more seeds would mean re-running Leg-A pretraining,
out of scope for this Rev.

### 16.16.5 Arm contrast — Δ redefinition, reusing §16.2.1's machinery verbatim

**Δ, redefined.** `Δ_Lquery(arm, K, c) := L_query(off, K, c) −
L_query(arm, K, c)` (readout (B), §16.16.3) — sign chosen so **positive =
arm's loss is LOWER than off's = arm helps**, preserving the "positive=arm
better" reading convention the old `recovered_frac`-based Δ used
(`arm − off`, higher=better there). At the `delta_ci_n3(values_a,
values_b)` call site (which returns `a−b`) this means calling
`delta_ci_n3(off_vals, arm_vals)` — an INTENTIONAL, disclosed argument-order
reversal from the existing `killer_prediction_readout`'s own
`delta_ci_n3(arm_vals, off_vals)` call (§16.2.1's own convention, correct
for a higher-is-better metric) — same function, swapped arguments, one-line
reason, not a new CI formula.

**`det`/`holds`/`det_arm`/`agree` — reused verbatim, zero changes.**
`phase2_hexachotomy.py`'s own primitives operate on already-computed
`(ci_low, ci_high)` pairs and booleans (its own docstring: "does not
compute CIs itself... consumes already-computed pairs") — they are
readout-agnostic by construction and require no modification. The SAME
`K∈{32,20}` pair, the SAME per-checkpoint schedule
`{250,500,1000,2500,5000}`, and the SAME six-bucket outcome space
(PERSISTENT/TRANSIENT/LATE-EMERGENT/CONVERGED-EQUIVALENT/UNRESOLVED/
NON-MONOTONE) are reused unchanged — the totality proof (§16.2.1) is a
property of the `holds(c)` TRUTH-TABLE shape, not of what `holds(c)` is
computed FROM, so it carries over without re-verification. **Bucket labels
corrected (Rev 1, attack-round-1 MINOR-2 — a transposition, not a new
count):** Rev 0's own bare `1+15+1+4+11=32` citation, read positionally
against the six-bucket list immediately above it, transposes PERSISTENT and
CONVERGED-EQUIVALENT. Verified directly against `phase2_hexachotomy.py`'s
own `totality_check` self-test (`__main__`, L221-222, the registered
ground truth): `PERSISTENT=4, TRANSIENT=15, LATE-EMERGENT=1,
CONVERGED-EQUIVALENT=1, NON-MONOTONE=11` (`4+15+1+1+11=32`) — PERSISTENT is
the 4-pattern bucket (the monotone-true-run-ending-at-5000 patterns for
each of the 4 possible non-terminal starts, §16.2.1's own worked
enumeration), CONVERGED-EQUIVALENT is the single all-FALSE pattern, not the
reverse. The underlying `32`-pattern enumeration and the totality PROOF
itself are unaffected by this citation fix — only the per-bucket count
attached to each label, here, was transposed.

**UNRESOLVED-GATE — dropped, per the task's own instruction.** That
seventh bucket existed for one specific failure mode: a per-checkpoint
Stage-0.5 gate that could fail at every non-terminal checkpoint of an
otherwise-monotone run. There is no per-checkpoint gate in this design
(§16.16.6 replaces it with a single upfront validity check) — the bucket
has no referent here and is not carried forward. What replaces it is
§16.16.6.

### 16.16.6 The OFF-floor validity gate — replaces Stage-0.5, closes the §16.15.7 gap formally

**Why a gate is still needed, even with a live readout.** §16.15.2's own
central finding — the OFF arm's noisy readout fell 21.8-46.4% but never
cleared the strict <50% pin — means "did familiarization teach the task at
all, well enough for an arm contrast on top of it to mean anything" is
NOT free to assume even for a working readout. This design pins a
**single, upfront, per-corpus FLOOR gate**, computed from ONLY the OFF
arm's own (already-existing, reused) checkpoints at readout (B) — no
per_token/global data needed to evaluate it, so it can run BEFORE the
12-cell launch (mirroring §16.2.1's own "compute the cheap gate before the
expensive launch" sequencing, and Constraint 1's gates-must-abort
discipline, §16.5).

**Rule, exactly as specified, with the trichotomy gap now closed formally
(requirement 7), and the pin itself now BLIND rather than knife-edge (Rev 1,
attack-round-1 MAJOR-3).** Per corpus, pool the 3 `ckpt_seed`s' own
readout-(B) `L_query(off, K=32, h∈{1,2}, c)` at `c=250` and `c=5000` (mean
across seeds — the SAME per-corpus, 3-seed-pooled convention
`phase2_trajectory_analysis.py`'s own module docstring already commits to
for the hexachotomy itself, reused here, not invented fresh); compute
`ratio := L_query(c=5000) / L_query(c=250)`. **Three MECE buckets, closing
exactly the gap §16.15.7 disclosed** (the observed pattern there — a
uniform, substantial, sub-pin fall — mapped to none of the pre-registered
options):

1. **FLOOR-PASS** (`ratio ≤ FLOOR_PIN`) — reproduces or exceeds this
   wave's OWN readout-(B) baseline. That corpus's arm-contrast proceeds to
   full hexachotomy classification (§16.16.5), CONFIRMATORY tier.
2. **PARTIAL-BELOW-FLOOR** (`FLOOR_PIN < ratio < 1.00`) — the formal name
   for §16.15.7's own disclosed gap: real, substantial task engagement
   occurred, but not enough to clear the reproduction floor. That corpus's
   hexachotomy classification is still COMPUTED (there is real signal to
   contrast) but the corpus's own headline finding is DEMOTED to
   DESCRIPTIVE TIER — this project's own established demotion convention
   (`FROZEN_BIAS_LM_DESIGN.md`'s blind-pin-timing demotion;
   `KEY_ANCHORING_DESIGN.md`'s own literal "DESCRIPTIVE, not confirmed"
   verdicts) — never silently promoted to a confirmatory reading.
3. **FAMILIARIZATION-NULL** (`ratio ≥ 1.00`) — loss did not fall at all (or
   rose). That corpus's own arm-contrast is uninterpretable and MUST be
   excluded (§16.5 Constraint 1) — no hexachotomy is computed for it.

**`FLOOR_PIN`, derived BLIND from data that exists BEFORE any new cell runs
— not the fixed `≤0.80` constant Rev 0 registered.** Rev 0's own `≤0.80`
pin was knife-edge (0.018 above the observed worst readout-(A) case,
openr1_s1 at `ratio=0.7823`, against a between-seed `σ≈0.093` on THAT same
readout — a margin smaller than one seed's own noise) AND was calibrated on
readout (A) (§16.15's training-loop numbers) while the gate it feeds is
evaluated on readout (B) (this design's own new frozen-checkpoint
`Q=K` pass) — two different instruments, never cross-validated against each
other before the pin was set. **Fixed via the design's own sequencing,
mirroring `bands_pinned_frozenbias.py`/`phase2_bands_pinned.py`'s own
"pin before launch, from data that already exists" convention exactly:**
`phase2b_chain.sh`'s new first Python step (§16.16.8's chain-fork item 3)
computes readout-(B) ratios on the 6 REUSED OFF checkpoints — data that
exists before any of the 12 new cells ever launches. `FLOOR_PIN :=
mean_B(ratio) + 2·σ_B(ratio)`, per corpus, computed from those 6 cells'
own 2 corpora × 3 seeds readout-(B) ratios (the house `k=2` one-sided
convention, `KEY_ANCHORING_DESIGN.md` §3.6 / `phase2_bands_pinned.py`'s own
`K_TOLERANCE=2.0`, reused verbatim rather than inventing a fourth tolerance
convention in this document). The Rev-0 `≤0.80` number is DEMOTED to a
provisional sanity bound only — reported alongside the real pin at harvest
time (if `FLOOR_PIN` lands far from `0.80`, that discrepancy is itself
disclosed, never silently absorbed) — never used as the enforced threshold.
**Registered BEFORE launch, written to a committed
`FLOOR_PINNED-Phase2b.json`** (mirrors `BANDS_PINNED-Phase2Familiarization.
json`'s own writer/validator/blind-check pattern, `phase2_bands_pinned.py`:
sha256-hash-of-referenced-inputs tamper-evidence, a strict-precedes
timestamp assertion the chain's own step ordering proves mechanically,
never re-derived from a copy), and enforced: `phase2b_floor_gate_enforce.py`
(below) reads `FLOOR_PIN` from this JSON, never recomputes it, and the
chain refuses to proceed past this step if the file is missing or its own
tamper-evidence check fails.

**Wave-level enforcement (mirrors the old Stage-0.5 gate's own "at least
one clears" convention, §16.2, Step 4).** If NEITHER corpus reaches
FLOOR-PASS or PARTIAL-BELOW-FLOOR (i.e. both land in FAMILIARIZATION-NULL),
the chain writes `PHASE2B_FLOOR_GATE_REFUSED` (naming mirrors
`STAGE05_LAUNCH_GATE_REFUSED` exactly) and ABORTS before launching the
12-cell training grid at all — a real, mechanical, GPU-saving early exit,
not a narrated warning. If at least one corpus reaches FLOOR-PASS or
PARTIAL-BELOW-FLOOR, the 12-cell launch proceeds; each corpus's own
downstream finding is reported at the tier (CONFIRMATORY / DESCRIPTIVE /
excluded) its own floor bucket licenses.

**Gate enforcement, mirroring `phase2_gate_enforce.py` exactly.** A new
`phase2b_floor_gate_enforce.py` (pure function `floor_verdict(ratio,
floor_pin) -> str` returning one of the three bucket names — `floor_pin` is
now a REQUIRED argument, read from `FLOOR_PINNED-Phase2b.json` above, never
a hardcoded constant, and the function itself never recomputes `ratio` or
`floor_pin`) plus the SAME belt-and-suspenders proof pattern
`phase2_gate_enforce.py`'s own `_run_selftest` already established:
positive + negative fixtures at an EXAMPLE `floor_pin=0.80` (the demoted
Rev-0 sanity-bound value, reused here only as a fixture, never as the
enforced pin) — `ratio=0.75`→FLOOR-PASS, `ratio=0.90`→PARTIAL-BELOW-FLOOR,
`ratio=1.05`→FAMILIARIZATION-NULL, plus boundary cases at exactly
`ratio=floor_pin` and `ratio=1.00` — each also proven at the subprocess
exit-code level (a real abort, not a computed-but-unread value), PLUS a
fixture at a DIFFERENT `floor_pin` (e.g. `0.65`) reclassifying the SAME
`ratio=0.75` from FLOOR-PASS to PARTIAL-BELOW-FLOOR — proving the function
actually reads its `floor_pin` argument rather than silently reverting to
a baked-in `0.80` — this is the "new gate needs a negative test" obligation
the task itself names.

### 16.16.7 Secondary readout — h∈{3,4} held-out-hop generalization

Using the SAME `eval_query_loss_heldout` function with `hop_set=(3,4)`
(§16.16.3's build delta), compute `Δ_Lquery(arm, K, c)` for the held-out
hop depths at every checkpoint, both K's, both arms — reported as a
SIMPLER, standalone `det(K,c)` table (CI excludes zero, yes/no) alongside
the primary hexachotomy, NOT folded into a second parallel classification
(disproportionate for a secondary readout). **Reading:** does key-geometry
stabilization help OOD hop-depth generalization even where in-distribution
(`h∈{1,2}`) learning already matches between arms — the genuine
generalization question §16.2.1's own `H_train=(1,2)/H_test=(3,4)` split
was built to ask, now askable for real (§16.2.1's own periodicity-guard
re-verification, `{1,2,3,4} mod {20,32} = {1,2,3,4}`, already holds and
needs no re-checking).

### 16.16.8 Cells and cost

**Cells.** 12 NEW training cells (`per_token`/`global` × `openr1-mix-ext`/
`wikitext-mix-ext` × 3 seeds) — the OFF arm's own 6 cells are DONE and
REUSED, not re-run. **Verified directly, this session, against the box**
(`youthful-indigo-turkey:/home/nvidia/chapter2/deltanet_rd/results/phase2/
ckpts/`): exactly 30 `.pt` files exist (6 cells × 5 checkpoints
`{250,500,1000,2500,5000}`), matching the design's own expectation exactly,
zero missing/extra files. **sha256-pinned for reuse, mirroring the K=69
precedent** (`experiment-runs/2026-07-07_keyanchor_scaling_wide/gates/
keyanchor_scaling_wide_k69_copy_manifest.sha256` — "Pinned ONCE... against
the ORIGINAL archive... never regenerated from a copy, which would make
the check tautological"): `sha256sum *.pt` computed directly on the box
against the ORIGINAL, never-copied files, all 30 hashes captured this
session. Build task: commit this manifest as
`experiment-runs/2026-07-08_phase2_familiarization/gates/
phase2b_off_ckpts_reuse_manifest.sha256` and wire a belt-and-suspenders
check (bash-level `sha256sum -c`, mirrored by a Python-level gate function)
before Phase-2b's own eval-`L_query` pass reads any of the 6 reused OFF
cells' checkpoints — closing the exact "silently scored a corrupted or
wrong-version reused checkpoint" failure mode the K=69 precedent exists to
prevent. **Enforced with its own negative test (MINOR-1, folded in here per
the task's own instruction — not a separate item):** corrupt one byte of one
reused `.pt` file's copy in a throwaway scratch dir, confirm `sha256sum -c`
exits nonzero and the Python-level gate function returns fail, mirroring the
K=69 precedent's own "run it, don't just write it" discipline.

**Chain script — forked, not reused naively (Rev 1, attack-round-1
MAJOR-2).** Verified directly against `phase2_chain.sh`: its Step 4 ("the
TERMINAL [5,000-step] checkpoint's own gate must clear for AT LEAST ONE
[corpus,seed] cell before ANY per_token/global cell launches", L245-273)
reads `phase2_gate_enforce.py` against the OLD Stage-0.5 gate JSONs, which
are permanently FAILED per §16.15.1's own 30/30 REFUSED table — a naive
`bash phase2_chain.sh` re-run would immediately self-refuse via
`STAGE05_LAUNCH_GATE_REFUSED` (L264) before a single Phase-2b cell launches,
even though the OLD gate is exactly what §16.16.3's MAJOR-1 fix (above)
just retired. Registered build task: **`phase2b_chain.sh`, a NEW script
forked from `phase2_chain.sh`** (not a flag/branch added to the original,
which stays the historical record of the Phase-2 run that already
happened), differing in exactly these ways:
1. Step 1 (Stage −1 self-tests) and step 1.5 (real-kernel smoke, extended
   per §16.16.9's Finding-5 fix below) are REUSED verbatim.
2. Step 2's OFF-arm-first launch and step 3's `BANDS_PINNED` pin are
   DROPPED — the 6 OFF cells are DONE (§16.16.8's own verified-on-box count)
   and `BANDS_PINNED-Phase2Familiarization.json` is already reused AS-IS
   (§16.16.9). Nothing to launch or re-pin at this step.
3. **New first Python step (does triple duty for MAJOR-2, MAJOR-3, and
   Rev-2's MAJOR-R2-3):** runs `eval_query_loss_heldout` (readout (B)) on
   the 6 reused OFF checkpoints (after the sha256 reuse gate above
   clears), at ALL 5 trajectory checkpoints, BOTH K's, and BOTH hop-sets
   (primary `{1,2}`, secondary `{3,4}`) — not merely the `c∈{250,5000}`
   pair the floor gate alone needs — writing every resulting `L_query`
   float to a committed `off_lquery_cache-Phase2b.json`, keyed on
   `(corpus, ckpt_seed, K, checkpoint_step, hop_set)`. **Registered as a
   real cache, not an informal one (Rev 2 fix, attack-round-2
   MAJOR-R2-3):** this single file is CONSUMED THREE TIMES — (a) the
   OFF-floor ratio (§16.16.6, re-pinned below), (b) the primary
   hexachotomy's own `off_vals` (§16.16.3's MAJOR-1 fix, now reading the
   cache rather than calling `eval_query_loss_heldout(off_model, ...)`
   fresh inside EACH of `analyze_corpus`'s own two `ARMS_NON_OFF`
   branches), and (c) the secondary `h∈{3,4}` readout's own off-half
   (§16.16.7) — computed ONCE here, never recomputed downstream, closing
   the exact duplication a naive per-branch implementation would otherwise
   introduce (§16.16.8's "Cost" paragraph, below).
4. **OFF-floor gate (§16.16.6, re-pinned below) replaces Step 4's old
   Stage-0.5 REFUSE gate entirely** — same "real subprocess exit code,
   mechanical abort, `PHASE2B_FLOOR_GATE_REFUSED` on failure" discipline,
   different (retired) predicate.
5. The 30-checkpoint sha256 manifest is verified AT RUN TIME (enforced,
   with the negative test above) as this forked script's own first gate,
   before step 3's eval pass reads any reused checkpoint.
6. Step 5 (the 12 new `per_token`/`global` cells, `run_cells_2way` on GPUs
   0-1) is REUSED verbatim from `phase2_chain.sh` unchanged — same helper,
   same 2-way scheduling, same resume-safety and budget-check discipline.
7. Step 6 (trajectory analysis) calls the MAJOR-1-fixed
   `phase2_trajectory_analysis.py`, now covering all 18 cells (6 reused OFF
   + 12 new) rather than the original chain's 18-cells-it-just-trained.

**Cost, re-derived from §16.15's own realized rate, not a fresh estimate.**
- **Training, 12 new cells:** §16.15.6's realized `0.6172 GPU-h` for 6
  cells (same 5,000-step budget, same architecture, includes a
  proportional share of Stage−1/smoke/gate overhead — `FROZEN_BIAS_LM_
  DESIGN.md`'s own rung-1 wave measured a tight 899-914s band across ALL
  20 training cells regardless of arm, so per-cell rate is NOT
  arm-dependent) scales to `0.6172 × (12/6) = 1.234 GPU-h`.
- **New eval-`L_query` passes (both readouts, both K's, all 18 cells):**
  `18 cells × 5 checkpoints × 2 hop-sets (primary {1,2}, secondary {3,4})
  × 2 K's (32,20) = 360 passes` — **the CACHED count, now the registered
  build target (Rev 2 fix, attack-round-2 MAJOR-R2-3), not the arithmetic's
  own default outcome.** Checked directly against `analyze_corpus`'s own
  per-arm iteration (`ARMS_NON_OFF = ("global", "per_token")`,
  `phase2_trajectory_analysis.py` L60): absent the OFF-eval cache
  (§16.16.3 item 1's own Rev-2 addition; §16.16.8's "New first Python
  step" above), `off`'s own `eval_query_loss_heldout` calls fire once
  inside the `global` branch of `build_holds_and_gate_by_checkpoint` AND
  once again inside the `per_token` branch — `off`'s own 120-of-360 passes
  double to 240, for a true UNCACHED total of `240 (off, doubled) + 240
  (global + per_token, once each) = 480 passes`, a ~33% undercount of the
  bare `18 cells × 5 × 2 × 2` arithmetic. **Both figures disclosed, not
  softened:** cached (registered, mandatory build task)
  `360 × 0.0022 ≈ 0.792 GPU-h`; uncached worst case (if the cache were
  somehow not wired) `480 × 0.0022 ≈ 1.056 GPU-h`. Priced at the Stage-0.5
  gate's own realized, TWICE-cross-validated rate (`§16.2.1`'s own priced
  estimate ≈0.0022 GPU-h/pass AND `§16.15.6`'s own independent realized
  cross-check, both agreeing) — a structurally comparable one-forward-call,
  `Q=K` scoring pass. **Disclosed uncertainty, not hidden:** an OLDER,
  less-recently-validated reference rate exists in this document
  (`FROZEN_BIAS_LM_DESIGN.md`'s 46 retrofit re-evals at ≈1.6 GPU-h total ⇒
  ≈0.0348 GPU-h/pass, 16× higher) — the 0.0022 rate is preferred as
  PRIMARY because it is the more recent, independently-cross-validated
  number from THIS wave's own family (§16.15.6: "closely matching
  §16.2.1's own priced estimate... a clean cross-check"), but a build-time
  timing check on ONE real pass (cheap, minutes) is registered as a
  mandatory pre-launch calibration (`CLAUDE.md`'s own "calibration run
  before a big sweep" rule), not an assumption to build the full 360-pass
  (or, uncached, 480-pass) budget on unverified.
- **New real-kernel smoke (below, §16.16.9):** small, priced at the same
  order as the original Phase-2 smoke gate. **Corrected (Rev 2 fix,
  attack-round-2 MINOR-R2-2) — the raw total below now actually USES the
  Rev-1-disclosed ≈0.02-0.03 GPU-h figure, not the stale Rev-0 ≈0.01 GPU-h
  line Rev 1 left the arithmetic on.** MAJOR-5's fix runs this suite
  THREE times (off, per_token, global) rather than once — still the same
  order of magnitude against the already-registered 5-10× debug-tax
  margin, but the actual number now belongs in the sum below, not merely
  narrated beside it: **≈0.03 GPU-h** (the disclosed `≈0.02-0.03` range's
  own upper bound, used conservatively).
- **Raw total: `1.234 + 0.792 + 0.03 ≈ 2.06 GPU-h`** (cached, registered
  primary figure). **Uncached worst case, disclosed as a contingency, Rev
  2:** `1.234 + 1.056 + 0.03 ≈ 2.32 GPU-h` — the OFF-eval cache above is
  what keeps the real figure at the lower number, not an assumption.

**Bracket, same 5-10× debug-tax convention this document applies
everywhere:** `2.06 × 5 ≈ 10.3 GPU-h` (low) to `2.06 × 10 ≈ 20.6 GPU-h`
(high) — registered bracket **≈10.3-20.6 GPU-h** (cached, primary; Rev 2
corrects Rev 1's `≈10.2-20.4 GPU-h` — a rounding-level shift from the
smoke-line fix above, not from the cache, which by design keeps the
cached figure unchanged from Rev 1's own 360-pass arithmetic).

**TIMING-PILOT RE-DERIVATION (2026-07-08, PILOT-FORCED — the mandatory
pre-launch calibration this section itself registered, resolving
§16.16.11 item 1; disclosed, never silent).** The first launch attempt's
own timing pilot (box, real kernels, one full eval pass) measured
**13.7339 s/pass ≈ 0.003815 GPU-h/pass** — 1.73× the assumed 0.0022
reference — and the chain's enforced budget gate correctly ABORTED the
launch at a projected 26.37 GPU-h vs the then-registered 20.6 ceiling,
before any of the 12 cells spent GPU time (realized abort cost ≈0.09
GPU-h of gates+smoke). Re-derived from the MEASURED rate: evals cached
`360 × 13.7339s = 4944s ≈ 1.373 GPU-h`; raw total
`1.234 + 1.373 + 0.03 ≈ 2.64 GPU-h`; bracket
`2.64 × 5 ≈ 13.2` to `2.64 × 10 ≈ 26.4` — **registered ceiling REPLACED:
26.4 GPU-h** (chain + off_cache constants updated to match). The
expected REAL spend is ~2.6 GPU-h; the ceiling is the debug-tax-inclusive
upper bound, not the plan. This replacement follows the identical
pilot/audit-forced precedent of §16.2.3's Rev-3 widening — measured
number, shown arithmetic, named trigger — and is exactly the scenario
the "not an assumption to build the full 360-pass budget on unverified"
sentence above pre-registered the pilot to catch. **Uncached
contingency at the measured rate:** `480 × 13.7339s ≈ 1.831 GPU-h` →
raw `≈3.10` → `15.5-31.0 GPU-h` — the OFF-eval cache remains mandatory.

**Superseded pre-pilot contingency text (kept for the record):** were the
OFF-eval cache somehow not wired, the pre-pilot convention gave
`2.32 × 5 ≈ 11.6 GPU-h` to `2.32 × 10 ≈ 23.2 GPU-h` — modestly above the
pre-pilot ceiling, which is exactly why §16.16.8's OFF-eval cache (above)
is a MANDATORY, tested build task for this wave, not an optional
efficiency nicety; the budget guard (§16.16.9) is re-pinned to the CACHED
ceiling (`20.6 GPU-h`) on the assumption the cache lands as registered,
and will mechanically abort a run that overshoots it regardless of cause.
**Sanity context:** comparable in size to Phase-1's own ≈24.2 GPU-h
ceiling and Path (iii)'s ≈21 GPU-h grid, but every prior wave in this
document that used this exact methodology (raw realized-or-realized-rate-
scaled estimate × 5-10×) landed its REALIZED cost well under the
bracket's own low end — most recently §16.15.6 itself, `0.617` realized
against a leg-scoped `2.22-4.45` bracket, a ≈3.6-7.2× undershoot even of
the LOW end. No claim is made that Phase-2b will repeat that margin, only
that it is the base rate this project has observed every time it has
bracketed this way. GPUs 0-1 are free.

### 16.16.9 Gates before launch

- **Stage −1 CPU-stub suite (`phase2_stage_minus1.py`), reused, EXTENDED
  not replaced.** Already exercises `per_token`/`global` at the LOGIC level
  (L510-527: "per_token/global arms: buildable, and the OFF-only gate is
  correctly None for them") and the seed formula's own collision-freedom
  (item 9, exhaustive enumeration). New items owed: (a) the two new
  `phase2_seed` kinds' own collision-freedom, extending item 9's proof;
  (b) a NEW positive assertion — the arm-INDEPENDENCE pairing device
  (§16.16.2 item 3) actually produces IDENTICAL seeds across arms for the
  new kinds (inverse of item 9's existing `s1 != s2` collision test); (c)
  **MINOR-3 (Rev 1, attack-round-1) — a new assertion tying `--arm` to the
  init checkpoint's own baked config.** Verified directly against
  `phase2_familiarization_train.py`: `run_familiarization_cell`
  constructs the model from `torch.load(init_checkpoint, ...)["config"]`
  (L408) — that config dict includes `frozen_bias_arm` (`DeltaNetLM.
  config()`, `lm_pretrain_rd.py` L1174) — so the model's ACTUAL blend
  behavior is governed entirely by the checkpoint's own baked value, and
  the CLI `--arm` argument (used only for `run_name`, seeding, and
  bookkeeping) could silently disagree with it if a wrong `--init-
  checkpoint` path were ever passed. New Stage −1 item: immediately after
  loading `init_checkpoint`'s config (mirroring L408's own call), assert
  `config["frozen_bias_arm"] == arm` and fail loudly on mismatch — a
  cheap, mechanical defense of the chain's own by-construction safety
  (every `phase2b_chain.sh` launch already pairs the right checkpoint path
  with the right `--arm` by construction, per its own `init_ckpt` path
  templating; this assertion makes that pairing verified, not merely
  trusted); (d) **MAJOR-R2-2 (Rev 2, attack-round-2) — a new assertion
  that `eval_query_loss_heldout`'s own forward path never invokes
  surgery, per §16.16.3's "Surgery-mode scope, pinned" paragraph above;
  re-pinned at AST level (Rev 2.1 fix, round-3 verify MINOR-R3-1, same
  `_references` helper §16.16.3 registers — supersedes the plain
  substring check, fragile to a docstring/comment mentioning the
  identifier):** `not _references(eval_query_loss_heldout,
  "frozen_bias_surgery")` and the same check against `query_loss_forward`
  (the function it wraps) — a mechanical proof that eval-(B) runs the
  blend natively, never force-off, so the reported effect stays the arm's
  whole causal package
  and never silently narrows to the un-built isolated-contrast follow-on.
- **Real-kernel smoke — a genuine, previously-undiscovered gap, found
  this session, not assumed closed; EXPANDED (Rev 1, attack-round-1
  MAJOR-5) to cover BOTH new blend paths, not just one.** `phase2_smoke_
  gpu.py`'s own `SMOKE_ARM = "off"` (L80, hardcoded module constant, no
  `--arm` CLI flag exists) means the real fla/Triton kernel path has NEVER
  exercised `apply_frozen_bias_blend` (§16.16.1: applied UNCONDITIONALLY
  whenever `frozen_bias_arm != "off"`) — the exact code path all 12 of
  Phase-2b's NEW cells run. Rev 0 registered running the smoke suite
  against "at least one non-off arm" — insufficient, because `per_token`
  and `global` exercise DIFFERENT tensor operations inside `apply_frozen_
  bias_blend` (a per-token gather-lookup vs. a broadcast add — different
  kernel/memory-access shapes, not two branches of the same op), and this
  program has direct, disclosed precedent for a kernel bug that was
  branch-specific and device-placement-specific (§16.15.7 item 2: a real
  Triton `Pointer argument cannot be accessed from Triton` failure caught
  in a pre-launch dry run, invisible to the CPU-stub suite). **Registered
  build task:** extend `phase2_smoke_gpu.py` with a real `--arm
  {off,per_token,global}` flag (threading `SMOKE_ARM` through to
  checkpoint selection and model config instead of hardcoding it), and
  PARAMETERIZE the chain's own smoke step to run the FULL positive+negative
  suite three times — once per arm (`off`, `per_token`, `global`) — with
  ALL THREE required to pass before the 12-cell launch, not merely one
  non-off arm as a stand-in for both.
- **BANDS_PINNED-Phase2Familiarization.json — reused AS-IS, no re-pin.**
  This band is computed from OFF's own per-seed VAL-LOSS (not `L_query`) at
  each checkpoint, and val-loss-band validity depends on the TRAINING
  recipe (corpus mix, `λ_fam`, optimizer/LR schedule), which is COMPLETELY
  UNCHANGED for Phase-2b's 12 new cells (same trainer, same recipe, only
  the READOUT changed). Reused unchanged; disclosed explicitly so a future
  reader does not mistake "no new band" for an oversight.
- **Budget guard.** Same `phase2_chain.sh`-style mechanical wall-clock ×
  N_GPUs / 3600 check, re-pinned to the new §16.16.8 ceiling (**20.6
  GPU-h**, Rev 2's corrected cached-primary bracket high end — supersedes
  Rev 1's `20.4 GPU-h`; §16.16.8's "Cost" paragraph), same
  real-abort-not-narrated discipline (§16.5 Constraint 1).
- **OFF-floor gate enforcement.** §16.16.6's own belt-and-suspenders
  pattern, run BEFORE the 12-cell launch.
- **sha256 reuse gate.** §16.16.8's own belt-and-suspenders pattern, run
  before any reused OFF checkpoint is read by the new eval pass.

### 16.16.10 Paper stakes, both directions — qualified by §16.16.4's power sketch AND §16.16.1's causal-package framing (Rev 1, attack-round-1 MAJOR — tempered)

**Framing caveat, stated once here rather than repeated in every bullet
(§16.16.1's Rev 1 fix):** every "causal evidence" claim below is evidence
about the arm's ENTIRE causal package — pretraining-era weight divergence
PLUS the blend persistently firing throughout familiarization — not an
isolated claim about acquisition-time intervention alone (that narrower
claim would require the un-built isolated-contrast follow-on, §16.16.1).
Neither direction below should be written up, in the eventual paper, as
having decomposed which of the two components drives the effect.

- **Arm effect found (a hexachotomy PERSISTENT or LATE-EMERGENT verdict, at
  a FLOOR-PASS corpus, with the floor/CI machinery intact):** the FIRST
  causal evidence in this program that the frozen-bias arm's causal package
  (key-geometry stabilization AND its ongoing familiarization-time
  application) aids in-context task ACQUISITION, not merely a static
  zero-shot geometric property — directly answers the keystone question,
  reconnecting H_LINK to the capacity-law empirical basis (§16.2's own
  "What it tests" paragraph) for the first time with a live instrument.
- **No arm effect, WITH adequate power (CONVERGED-EQUIVALENT at a
  FLOOR-PASS corpus, both arms individually `det_arm`-positive):** the
  frozen-bias arm's causal package does not transfer to task acquisition
  even when the model is trained on the task and given every opportunity
  via SGD to exploit it — combined with the triple-null (§16.15.4), a
  coherent, publishable negative: geometry stabilization is real (Chapter
  2's own confirmed finding) but behaviorally inert for THIS composition
  task, at every instrument this program has built.
- **No arm effect, UNDERPOWERED (UNRESOLVED, or a FLOOR-gated corpus
  reporting only DESCRIPTIVE-tier numbers):** per §16.16.4, this is the
  MOST LIKELY null outcome at n=3 seeds for anything short of a huge
  effect — must be reported as an open measurement question (more seeds,
  a longer budget, or the pairing device's own realized variance-reduction
  once (B) numbers exist), NEVER dressed up as evidence the intervention
  doesn't matter. **This is a real risk to the "bounds the mechanism's
  behavioral relevance" framing** and is registered here so it cannot be
  silently elided at harvest time the way §16.15.7's trichotomy gap
  almost was.

### 16.16.11 Open items for the independent attack round (self-attack, not exhaustive; STATUS as of Rev 2.1 — none of these 5 overlapped attack-round-1's own 5 MAJOR/3 MINOR findings, attack-round-2's own 3 MAJOR/2 MINOR findings, OR round-3's own 1 MAJOR/1 MINOR verify findings, §16.17 — all 5 below remain genuinely OPEN, carried forward to round 4)

1. Is `0.0022 GPU-h/pass` really the right reference rate for the new
   eval-`L_query` pass, or does it under-price the null-shuffle-free but
   otherwise-comparable computation (§16.16.8's own disclosed 16×
   alternative)? A build-time timing pilot is registered but has not run.
   **[RESOLVED 2026-07-08, by the pilot itself: measured 13.7339 s/pass
   ≈ 0.003815 GPU-h/pass — 1.73× the 0.0022 reference, 9× BELOW the 16×
   alternative; the chain's budget gate correctly aborted the first
   launch attempt at the stale ceiling; bracket re-derived and ceiling
   replaced at 26.4 GPU-h — see §16.16.8's TIMING-PILOT RE-DERIVATION.]**
2. Is the `σ≈0.43-0.48` between-seed proxy (§16.16.4) actually conservative,
   or could readout (B)'s own between-seed variance be LARGER than
   readout (A)'s for some reason not yet considered (e.g. `Q=K=32` queries
   per episode are correlated within an episode in a way 2 independent
   training-loop queries are not)?
3. Is pooling 3 `ckpt_seed`s into ONE floor-gate ratio per corpus
   (§16.16.6) the right granularity, or should the floor gate be evaluated
   per-seed with an "at least 2/3" rule, mirroring this document's other
   "at least one/some" gate conventions more closely than a strict pooled
   mean?
4. Does the arm-independent pairing device (§16.16.2 item 3) introduce any
   NEW risk — e.g., does scoring `off`/`per_token`/`global` on the
   IDENTICAL held-out episode at a given checkpoint create any shared
   dependency the `delta_ci_n3` CI formula's own independence assumptions
   did not anticipate?
5. Should the secondary h∈{3,4} readout's own `det(K,c)` table be
   corrected for multiple comparisons (4 checkpoints × 2 K's × 2 arms = up
   to 16 tests) before being read as evidence of anything, given this
   design explicitly declines to fold it into the primary hexachotomy's
   own single pre-registered decision rule?

**Not self-launched.** Per this project's waterfall discipline, §16.16 (now
Rev 2.1) awaits a FOURTH independent round — round 3 (2026-07-13) was a
focused VERIFY pass on round-2's own 5 fixes, not a re-run of the 5
self-attack items above, so those remain open, still unaddressed, still
blocking build. Round 4 is registered as a SPOT-CHECK (per this round's
own forward-pointer, §16.17): confirm round-3's own 2 fixes (MAJOR-R3-1,
MINOR-R3-1) land as described against the real code they cite, plus take
a first pass at the 5 items above — before any build task registered
above is started. See §16.17 for the full Rev 0 → Rev 1 → Rev 2 → Rev 2.1
finding→fix trace and the round-4 forward-pointer.

### 16.17 ATTACK-ROUND fix-maps for §16.16 (round 1: 2026-07-11, round 2: 2026-07-12, round 3: 2026-07-13, round 4: 2026-07-08) — verdict: round 1 NEEDS-REVISION (fixed → Rev 1), round 2 NEEDS-REVISION (fixed → Rev 2), round 3 NEEDS-REVISION (fixed → Rev 2.1), round 4 NEEDS-REVISION (fixed → Rev 2.2, prose+assertion-mechanism only)

**Round 1** (2026-07-11). A first independent adversarial pass reviewed
Rev 0 of §16.16 (the Phase-2b vocab-space behavioral-contrast instrument,
§16.16.1-§16.16.11) before any code was written, per this project's own
waterfall discipline
(`CLAUDE.md`) and mirroring §16.7's/§16.9's own attack-round convention for
§16.2. Verdict: **NEEDS-REVISION** — 5 MAJOR, 3 MINOR, no FATAL. Every
finding below is fixed in this revision (Rev 1, §16.16.1-§16.16.11); none is
deferred or waved away. Findings recorded near-verbatim for the historical
record, per house style; resolutions are stated as landed in this text, not
as intentions. **Items verified correct by this round and NOT reopened**
(not itemized in the table below, per the same "don't re-litigate
verified-correct items" convention §16.9's own trailer uses): §16.16.2's own
confound-freedom analysis (init-checkpoint sharing, training-data-order
non-identity, eval-batch pairing device); §16.16.3's readout-(A)-vs-(B)
adjudication and `eval_query_loss_heldout`/new-seed-kind build delta as
FAR as it went (MAJOR-1 below is a gap in what CONSUMES that delta, not an
error in the delta itself); §16.16.4's power sketch arithmetic; §16.16.7's
secondary readout; §16.16.8's cost re-derivation arithmetic.

| # | Finding (attack-round-1 on §16.16 Rev 0) | Severity | Fix (Rev 1) | Location |
|---|---|---|---|---|
| MAJOR-1 | The `L_query`→hexachotomy integration was unregistered: `phase2_trajectory_analysis.py`'s own `build_holds_and_gate_by_checkpoint`/`analyze_corpus`/`killer_prediction_readout` sourced `off_vals`/`arm_vals` from `per_h[h]['recovered_frac']` (the DEAD `d_state`-space quantity, 0.0 in 30/30 §16.15.1 readings) and `stage05_pass_by_c` from the permanently-FAILED Stage-0.5 gate JSONs — as specced, a clean run would silently force a real monotone-holds-true trajectory into UNRESOLVED-GATE, burying a PERSISTENT finding, and would never compute a single real (B)-readout number at all | MAJOR | Registered the explicit analysis-module rewrite: `off_vals`/`arm_vals` now source from the NEW `eval_query_loss_heldout` outputs (`L_query` floats, paired via `delta_ci_n3(off_vals, arm_vals)`); `stage05_pass_by_c = {c: True for c in CHECKPOINTS}` unconditionally (mirroring `totality_check`'s own `always_pass_gate`, `phase2_hexachotomy.py` L204) and the dead gate-JSON read is DELETED, not bypassed — the Stage-0.5 gate is RETIRED for Phase-2b, replaced by the OFF-floor gate (§16.16.6); plus a REGISTERED negative Stage-1 test: a synthetic monotone-holds-true trajectory must classify PERSISTENT, never UNRESOLVED-GATE, under the fixed sourcing | §16.16.3 ("Analysis-module rewrite" paragraph) |
| MAJOR-2 | Chain reuse ambiguity: `phase2_chain.sh`'s Step-4 terminal Stage-0.5 REFUSE gate is permanently failed (§16.15.1's 30/30 REFUSED table) — a naive re-run of the existing chain on the reused OFF checkpoints self-refuses via `STAGE05_LAUNCH_GATE_REFUSED` before a single Phase-2b cell launches | MAJOR | Registered `phase2b_chain.sh` as its own forked script (not a flag on the original, which stays the historical Phase-2 record): drops the Stage-0.5 launch gate and the OFF-arm-training/BANDS_PINNED steps (already done, already reused); adds a new first Python step running `eval_query_loss_heldout` passes on the 6 reused OFF checkpoints (produces both the OFF-floor ratios MAJOR-3 needs AND MAJOR-1's own `off_vals`, computed once); adds the OFF-floor gate (enforced abort, MAJOR-3); verifies the 30-checkpoint sha256 manifest AT RUN TIME (enforced, with a negative test, MINOR-1); reuses Stage −1/real-kernel-smoke/`run_cells_2way`/trajectory-analysis steps verbatim; then launches the 12 new cells 2-way on GPUs 0-1 | §16.16.8 ("Chain script — forked, not reused naively" paragraph) |
| MAJOR-3 | The OFF-floor pin (`ratio ≤ 0.80`) was knife-edge — 0.018 above the observed readout-(A) worst case (openr1_s1, 0.7823) against a `σ≈0.093` between-seed spread on THAT SAME readout, a margin smaller than one seed's own noise — AND was calibrated on readout (A) while evaluated on readout (B), two different instruments never cross-validated against each other | MAJOR | Re-pinned via the design's own sequencing, mirroring `phase2_bands_pinned.py`'s "pin before launch, from data that already exists" convention: `phase2b_chain.sh`'s new first step computes readout-(B) ratios on the 6 reused OFF checkpoints (data that exists before any new cell runs); `FLOOR_PIN := mean_B(ratio) + 2·σ_B(ratio)`, per corpus (house `k=2` convention, `phase2_bands_pinned.py`'s own `K_TOLERANCE`); the Rev-0 `≤0.80` number demoted to a provisional sanity bound only, never the enforced threshold; registered BEFORE launch, written to a committed `FLOOR_PINNED-Phase2b.json` (same sha256/tamper-evidence/strict-precedes discipline as `BANDS_PINNED-Phase2Familiarization.json`), and enforced (`phase2b_floor_gate_enforce.py`'s own `floor_verdict(ratio, floor_pin)` now takes the pin as a required argument, never a hardcoded constant) | §16.16.6 ("`FLOOR_PIN`, derived BLIND from data..." paragraph) |
| MAJOR-4 | Causal-framing understatement: the frozen-bias blend fires on EVERY forward pass throughout familiarization (`lm_pretrain_rd.py` L854-857, `apply_frozen_bias_blend` unconditional whenever `frozen_bias_arm != "off"`), not merely as a one-time pretraining-era divergence baked into the init checkpoint — Rev 0's own "exactly ONE thing at the moment familiarization begins" framing implied the latter, understating what this design's intervention actually is and what it can be read as evidence for | MAJOR | Causal paragraph rewritten (§16.16.1): the registered intervention is the arm's ENTIRE causal package — (a) the ORIGINAL 20,000-step pretraining's own arm-specific weight divergence PLUS (b) the SAME blend mechanism continuing to fire on every familiarization step — this IS the registered H_LINK-A target, citing §16.2.1's own "Base checkpoints" paragraph directly; an isolated familiarization-time-only contrast (all arms initialized from the SAME `off` checkpoint, blend toggled only during familiarization) is NAMED as a distinct, out-of-scope follow-on design requiring init checkpoints that do not exist in this program's archive today, not this wave; §16.16.10's paper-stakes framing tempered with an explicit "entire causal package, not a decomposed claim" caveat | §16.16.1 ("Causal logic" paragraph, REVISED); §16.16.10 (framing caveat added) |
| MAJOR-5 | Real-kernel smoke under-covers the two new arms: Rev 0 registered running `phase2_smoke_gpu.py` against "at least one non-off arm," but `per_token` and `global` exercise DIFFERENT tensor operations inside `apply_frozen_bias_blend` (per-token gather-lookup vs. global broadcast add, not two branches of one op), and this program has direct precedent for a kernel bug that was branch-and-device-specific (§16.15.7 item 2's own Triton `Pointer argument cannot be accessed from Triton` failure, invisible to the CPU-stub suite) | MAJOR | Registered `phase2_smoke_gpu.py` parameterization (`--arm {off,per_token,global}`, threading `SMOKE_ARM` through checkpoint selection and model config) with the chain's own smoke step running the FULL positive+negative suite THREE times — once per arm, off AND per_token AND global — all three required to pass before the 12-cell launch | §16.16.9 ("Real-kernel smoke" bullet, EXPANDED) |
| MINOR-1 | The sha256 reuse gate (§16.16.8) was registered as a belt-and-suspenders check but its own negative test was left implicit, not stated as a distinct obligation | MINOR | Made explicit: corrupt one byte of a scratch-dir copy of one reused `.pt` file, confirm both `sha256sum -c` and the Python-level gate function exit/return fail — folded into §16.16.8's own sha256-gate paragraph, and into MAJOR-2's `phase2b_chain.sh` fork item 5 | §16.16.8 (sha256 reuse gate paragraph) |
| MINOR-2 | §16.16.5's totality citation (`1+15+1+4+11=32`) transposes PERSISTENT and CONVERGED-EQUIVALENT when read positionally against the six-bucket list immediately above it | MINOR | Verified directly against `phase2_hexachotomy.py`'s own `totality_check` self-test ground truth (L221-222) and corrected with explicit labels: `PERSISTENT=4, TRANSIENT=15, LATE-EMERGENT=1, CONVERGED-EQUIVALENT=1, NON-MONOTONE=11` (`4+15+1+1+11=32`) — the enumeration and proof are unaffected, only the per-bucket count attached to each label was wrong | §16.16.5 ("Bucket labels corrected" sentence) |
| MINOR-3 | No assertion ties `--arm` to the init checkpoint's own baked `frozen_bias_arm` config field — `run_familiarization_cell` constructs the model straight from the checkpoint's own saved config (`phase2_familiarization_train.py` L408; `DeltaNetLM.config()` bakes `frozen_bias_arm`, `lm_pretrain_rd.py` L1174), so the model's actual blend behavior is governed by the checkpoint, not the CLI flag, and a wrong `--init-checkpoint` path could silently disagree with `--arm` | MINOR | New Stage −1 assertion: immediately after loading the init checkpoint's config, assert `config["frozen_bias_arm"] == arm`, fail loudly on mismatch — defends the chain's own by-construction path/arm pairing mechanically rather than by trust alone | §16.16.9 (Stage −1 CPU-stub suite bullet, item (c)) |

**Round 2** (2026-07-12) reviewed Rev 1 fresh — this section's own fixes
above (§16.16.1-§16.16.11) plus the 5 self-attack items §16.16.11 listed as
still open. Verdict: **NEEDS-REVISION** — 3 MAJOR, 2 MINOR, no FATAL. Every
finding below is fixed in this revision (Rev 2, §16.16.1-§16.16.11 as
edited above); none is deferred or waved away. **Verified CORRECT and NOT
reopened this round** (per the same "don't re-litigate verified-correct
items" convention): the floor-gate DIRECTION itself (§16.16.6's
`ratio := L_query(c=5000)/L_query(c=250)`, lower=better,
`FLOOR_PIN := mean+2σ` — reviewed explicitly this round and confirmed to
correctly gate on FALLING loss, not rising); every Rev-1 fix (MAJOR-1
through MAJOR-5, MINOR-1 through MINOR-3) — checked against the real code
each cites and confirmed landed as described, not reopened. The 5
self-attack items §16.16.11 lists remain OPEN, unaddressed by this round's
own findings (below), carried forward to round 3.

| # | Finding (attack-round-2 on §16.16 Rev 1) | Severity | Fix (Rev 2) | Location |
|---|---|---|---|---|
| MAJOR-R2-1 | Rev 1's own registered Stage-1 negative test only exercised `phase2_hexachotomy.classify_trajectory` on a hand-built synthetic `holds_by_c` dict — a fact already true before Rev 1's MAJOR-1 fix ever existed, since `classify_trajectory` was never the broken piece. `build_holds_and_gate_by_checkpoint` (`phase2_trajectory_analysis.py` L98-136) takes no `holds_by_c` parameter at all — it computes `holds_by_c` internally from `killer_prediction_readout`'s own dead `recovered_frac` sourcing (L116-117) and the failed gate JSONs (L125-134) — so the registered test would pass identically whether or not MAJOR-1's rewrite ever actually landed | MAJOR | Test re-registered one layer down: stub `eval_query_loss_heldout` with controlled synthetic `L_query` floats engineered to produce the SAME monotone-holds-true `FTTTT` pattern through the REAL `delta_ci_n3`→`det`→`holds` chain running INSIDE the rewritten `build_holds_and_gate_by_checkpoint` itself (plus a stubbed checkpoint loader so no real `.pt`/GPU is touched), PLUS a structural assertion pinned exactly: `"recovered_frac" not in inspect.getsource(phase2_trajectory_analysis)` and `"gate_json_path_for" not in inspect.getsource(...build_holds_and_gate_by_checkpoint)` — proving the dead sourcing paths are actually GONE, not merely routed around | §16.16.3 (item 3, "New Stage-1 negative test") |
| MAJOR-R2-2 | The new eval-(B) readout's own forward path (`query_loss_forward`, `phase2_familiarization_train.py` L263-285) runs the frozen-bias blend NATIVELY — required by the whole-package causal framing (§16.16.1) — but the parent document's own MAJOR-NEW-5 rule ("wrap scoring passes in `frozen_bias_surgery(force_off=True)`") is worded broadly enough that a builder pattern-matching on it could paste the wrapper onto `eval_query_loss_heldout` out of habit, silently narrowing the causal claim to the un-built isolated-contrast follow-on while still reporting it under the whole-package framing | MAJOR | Explicit surgery-mode paragraph pinned: eval-(B) runs with NO surgery override (the blend is part of the measured causal package, by construction); MAJOR-NEW-5's own scope is RETIRED-GATE-ONLY — it applied to the dead `d_state`-space `rec@0.9` Stage-0.5 quantity, which §16.16.3's own MAJOR-1 fix already retires entirely, never to eval-(B)'s `L_query`; cites the native L263-285/L279 forward call directly; a new Stage −1 assertion (`"frozen_bias_surgery" not in inspect.getsource(eval_query_loss_heldout)`, same check against `query_loss_forward`) makes the no-surgery guarantee mechanical, not trusted | §16.16.3 (new "Surgery-mode scope, pinned" paragraph); §16.16.9 (new Stage −1 item (d)) |
| MAJOR-R2-3 | The cost paragraph's own "computed ONCE, consumed twice" framing (§16.16.8 item 3) contradicts the literal Rev-1 fix-1 text: `killer_prediction_readout`'s replacement calls eval for `off` INSIDE each per-arm branch, and `analyze_corpus` (`phase2_trajectory_analysis.py` L139-142) iterates 2 non-off arms (`ARMS_NON_OFF = ("global","per_token")`, L60) — so `off` is re-evaluated 2× absent a real cache; the registered `360`-pass figure undercounts the true `≈480`-pass cost by ~33% | MAJOR | Registered an explicit OFF-eval cache: the pre-launch "New first Python step" (§16.16.8) now computes `off`'s own `L_query` at ALL 5 checkpoints × both K's × both hop-sets (not just the floor gate's `c∈{250,5000}` pair) and writes it to a committed `off_lquery_cache-Phase2b.json`, keyed `(corpus, ckpt_seed, K, checkpoint_step, hop_set)`; BOTH `analyze_corpus`'s own arm branches (and the secondary h∈{3,4} readout) read `off`'s values from this cache rather than recomputing — deterministic given §16.16.2 item 3's own pairing device, so caching is pure economy, not a new independence assumption. Cost line corrected: `360` passes (cached, now the registered/mandatory build target) `≈0.792 GPU-h`, with the uncached worst case `≈480` passes `≈1.056 GPU-h` disclosed alongside, both folded into the corrected raw-total/bracket arithmetic | §16.16.3 (item 1, cache addition); §16.16.8 ("New first Python step" item 3; "Cost" paragraph) |
| MINOR-R2-1 | `eval_query_loss_heldout`'s own `batch_size` parameter (§16.16.3's "Build delta") was left unpinned — a future caller's own default would silently govern it rather than the codebase's own established convention | MINOR | Pinned explicitly to `batch_size: int = 16`, matching `killer_prediction_readout`'s and `build_holds_and_gate_by_checkpoint`'s own existing `=16` defaults (`phase2_trajectory_analysis.py` L78, L99) — now REGISTERED on the new function's own signature, not inherited | §16.16.3 ("Build delta" bullet) |
| MINOR-R2-2 | The smoke-cost line's own arithmetic still used the stale Rev-0 `≈0.01 GPU-h` figure in the "Raw total" sum, even though the SAME paragraph's own prose, two sentences earlier, disclosed MAJOR-5's three-arm suite pricing at `≈0.02-0.03 GPU-h` — the narration and the arithmetic disagreed | MINOR | Raw total corrected to actually use `≈0.03 GPU-h` (the disclosed range's own conservative upper bound) — raw total `1.234 + 0.792 + 0.03 ≈ 2.06 GPU-h` (cached), bracket `≈10.3-20.6 GPU-h`, both re-derived from the corrected figure rather than left narrated-but-unused | §16.16.8 ("Cost" paragraph, smoke bullet + "Raw total"/"Bracket" lines) |

**Round 3** (2026-07-13) executed part (a) of round-2's own forward
pointer above: verify all 5 round-2 fixes land as described against the
(still design-only) text, fresh-eyes, per this project's own waterfall
discipline (`CLAUDE.md`). All 5 confirmed landed as described, not
reopened. The pass also caught 2 NEW findings — both surgical, neither a
re-opening of a round-1/round-2 item — while checking those fixes' own
citations against the real code. Verdict: **NEEDS-REVISION** — 1 MAJOR, 1
MINOR, no FATAL. Both fixed in this revision (Rev 2.1, §16.16.3/§16.16.9
as edited above); neither deferred nor waved away. **Verified CORRECT and
NOT reopened this round:** every Rev-2 fix (MAJOR-R2-1 through MAJOR-R2-3,
MINOR-R2-1/MINOR-R2-2) — checked against the real code each cites
(`phase2_trajectory_analysis.py`, `phase2_familiarization_train.py`,
`lm_pretrain_rd.py`) and confirmed landed as described. Part (b) of
round-2's own forward pointer — the 5 self-attack items §16.16.11 lists —
was OUT OF SCOPE for this round's own surgical verify mandate; all 5
remain OPEN, unaddressed, carried forward to round 4.

| # | Finding (round-3 verify on §16.16 Rev 2) | Severity | Fix (Rev 2.1) | Location |
|---|---|---|---|---|
| MAJOR-R3-1 | The registered Stage-1 negative test (item 3(i)) stubbed `load_init_checkpoint_strict` as the replacement `killer_prediction_readout`'s own checkpoint-load call — but that function (`lm_pretrain_rd.py` L1803) takes an ALREADY-CONSTRUCTED `model` argument, mutates it in place via `load_state_dict`, and returns the ckpt dict; it does not build a model from a bare path. The actual model-construction step lives one layer up, a SEPARATE `torch.load(...)['config']` antecedent (`phase2_familiarization_train.py`'s own `run_familiarization_cell`, L408, immediately followed by `load_init_checkpoint_strict` at L421) — as specced, the stub target could neither deliver the sentinel the test needs nor avoid a real `.pt`/GPU dependency | MAJOR | Registered a single-seam loading helper, `phase2b_load_eval_model(ckpt_path, device) -> DeltaNetLM`, as part of the analysis rewrite (new function in `phase2_trajectory_analysis.py`): internally reproduces the SAME `phase2_familiarization_train.py` L408→L421 double-defense sequence (`torch.load(...)['config']` → `DeltaNetLM(**config)` → `load_init_checkpoint_strict(model, path, device)`, preserving the config-equality assert + `strict=True load_state_dict`, `lm_pretrain_rd.py` L1803, verbatim). The rewritten `killer_prediction_readout` routes ALL eval-model loading through this ONE helper; the Stage-1 test now stubs `phase2b_load_eval_model` itself — the single seam — with a sentinel carrying `(arm, checkpoint_step)` as plain attributes, no real `.pt` read, no GPU. Production keeps the strict double-defense path unconditionally — the auditor's own noted laxer alternative, `reasoning_link_probe.load_checkpoint` (`reasoning_link_probe.py` L705-713, a single-layer `load_state_dict` with no explicit config-equality assert), is explicitly REJECTED, not silently substituted | §16.16.3 ("Loader" bullet, REWRITTEN; item 1; item 3(i)) |
| MINOR-R3-1 | The `"frozen_bias_surgery" not in inspect.getsource(...)` assertion (§16.16.3's surgery-mode paragraph, §16.16.9 item (d)) and the `"recovered_frac"`/`"gate_json_path_for"` structural assertions (item 3(ii), MAJOR-R2-1) are all plain substring-absence checks against `inspect.getsource(...)` — fragile to a well-meaning docstring or comment mentioning the same identifier in prose (exactly the disclosure style this document favors), which would false-positive-fail the assertion despite the code itself being correct | MINOR | All four assertions re-pinned as AST-level checks via one shared helper, `_references(func_or_module, identifier)` (registered once in `phase2_stage_minus1.py`): parses the source with `ast.parse`, walks `Call`/`Attribute`/`Name` nodes for the identifier — matches real calls/attribute-accesses/names, never a docstring's own string-constant contents or a comment (comments are dropped by `ast.parse` outright). One-line house convention registered alongside: in-code prose SHOULD cite a fix by section number rather than the literal retired identifier, where feasible | §16.16.3 (surgery-mode paragraph; item 3(ii)); §16.16.9 (item (d)) |

**Rev 2.1 has NOT yet had its own independent audit pass — the forward
pointer (per this project's waterfall discipline, `CLAUDE.md`) is a
FOURTH round on §16.16 as it now reads (Rev 2.1, this section's own
round-3 fixes above), registered as a SPOT-CHECK rather than a full
fresh-eyes attack round — round 3's own surgical scope (2 findings, both
narrow mechanics) does not warrant re-running the full adversarial
machinery a third time in a row; build does not start until round 4
passes clean or its own findings are fixed and re-verified. Round 4's own
scope, explicitly: (a) confirm round-3's own 2 fixes (MAJOR-R3-1,
MINOR-R3-1) land as described against the (still design-only) text; (b)
the 5 self-attack items §16.16.11 lists (GPU-h reference-rate uncertainty,
readout-(B) variance-proxy conservativeness, floor-gate pooling
granularity, pairing-device CI-independence risk, secondary-readout
multiple-comparisons correction) — unaddressed by round 3, still
genuinely open, still carried forward.** No cells launched, no code
written this session; STATE.md's queue updated.

**Round-4 scoped spot-check (2026-07-08, same round-3 reviewer) →
Rev 2.2.** Verdict: NEEDS-REVISION — 1 MAJOR + 1 MINOR, both in Rev
2.1's own fresh fixes; everything else carried forward clean.

| # | Severity | Finding | Rev 2.2 fix |
|---|---|---|---|
| R4-1 | MAJOR | The re-pinned AST helper `_references` (Name/Attribute walk only) is VACUOUS for the `recovered_frac` teeth-check — the dead sourcing is a dict-key string literal (`off_r["per_h"][h]["recovered_frac"]`, an `ast.Subscript` slice `Constant`), which the walk ignores; the reviewer RAN the helper against the live pre-rewrite module and got `False` — the assertion would pass whether or not the MAJOR-1 rewrite ever lands, reproducing the exact failure class MAJOR-R2-1 was raised to kill, inside its own hardening fix | `_references` extended with the Subscript-key clause (verbatim per the reviewer's prescription); an EMPIRICAL teeth-run registered as a Stage −1 obligation (pre-rewrite → must return True — coordinator re-ran and confirmed True this session; post-rewrite → must return False); aliased-import evasion disclosed as outside the accidental-paste threat model |
| R4-2 | MINOR | The "ONE seam ... no other call site" sentence contradicted MAJOR-R2-3's own cache flow (the OFF half reads the cache; the chain-step-3 cache builder necessarily loads the 6 OFF checkpoints) | Harmonized: one HELPER, exactly TWO registered callers — the rewritten `killer_prediction_readout` (non-off arms only) and the chain-step-3 OFF-eval cache builder; "arm-or-off" dropped |

The round-4 reviewer also confirmed: no injected instructions observed
in its own session (responding to the standing injection-canary notice);
the §16.16.11 self-attack items remain open and carried forward. The
remaining gate before build is the same reviewer's confirmation of these
two Rev 2.2 fixes.
