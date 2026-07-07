# REASONING-LINK: does key-geometry stabilization causally improve in-context multi-hop composition? (Rev 5 (2026-07-07) — post-attack-5)

**Status: DESIGN, attack-round-5 complete (verdict NEEDS-REVISION,
2026-07-07, fresh-eyes independent pass), this revision (Rev 5) resolves
every finding — see §13.5 for the full attack record and fix map (§13.1
records attack-round-1, resolved by Rev 1; §13.2 records attack-round-2,
resolved by Rev 2; §13.3 records attack-round-3, resolved by Rev 3; §13.4
records attack-round-4, resolved by Rev 4). Still not built, no GPU
spent.** This document is written to survive a sixth, independent
adversarial attack round before any code is written. Nothing here is a
launch authorization. **Rev 5 headline:** attack-round-5 found that every
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

**Both measured identically, not assumed, at Stage 0 calibration (§9):**
report `cos(q_eff_a, k_eff_a)` (premise iii) and `cos(k_eff_i, v_eff_i)`
(premise iv) per entity in the calibration episode pool, alongside
premises (i)-(ii) (`DELTANET_REALDATA_DESIGN.md` §5.2, R2-2).

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
launches, per this project's standing "registered before any data exists"
discipline — a premise that fails its own gate demotes the dependent
hop-depths' confirmatory status in the pre-registration itself, never
after seeing how the headline numbers happen to look.

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
- **Stage −1 causality assertion (new, §9 item 11).** Forward-A and
  forward-B **must** produce bit-identical (fp32, CPU) or
  tolerance-pinned (`1e-6`, GPU/bf16-kernel-boundary — matching this
  design's own standing tolerance convention, §9 items 2/5) residual
  streams over the shared BIND-phase prefix — the mechanical proof that
  appending the query in forward-B never leaks backward into the captured
  `S_T`, checked, not merely argued from causality.
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

**Stage −1 self-test (new, §9 item 6):** verify the surgery toggle
reproduces `frozen_bias_retrofit_eval_rd.py`'s own `"kraw"`-mode raw-key
capture byte-for-byte (`torch.equal`) on a tiny smoke batch — forcing
`frozen_bias_arm="off"` on a LOADED arm checkpoint and running its forward
pass must produce numerically identical post-`k_conv1d` keys to that same
checkpoint's `"kraw"`-mode captured `k_raw` (both are "no blend applied"
readings of the identical model, reached by two different code paths). A
single tiny-batch smoke, not a full grid cell, gates trust in the surgery
mechanism before the real grid launches.

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
   replicate disagrees beyond a registered tolerance at the calibration
   cell, the entire Phase-1 grid is blocked pending a design revision — not
   an exclusion rule applied per-cell, a hard gate before the grid launches.
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
    are IDENTICAL over the shared BIND-phase prefix: bit-identical in fp32
    on CPU, or within `1e-6` (this design's standing tolerance, items 2/5
    above) if run at the bf16 kernel boundary on GPU. Mechanically verifies
    §4.4's causality argument (the appended query in forward-B cannot
    retroactively affect the `S_T` captured from forward-A) rather than
    resting on the argument alone — per this project's own "run the
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
14M mixcontrol "off" arm, seed 0, one corpus, K=32, h∈{1,2,3,4}, BOTH query
markers (§7.6's robustness replicate). Purposes (Rev 3: purpose (a)
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
committing further compute) but calibrated at rung-3's own scale rather
than reusing the 14M-scale threshold.

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
