# REASONING-LINK: does key-geometry stabilization causally improve in-context multi-hop composition? (Rev 1 (2026-07-07) — post-attack-1, post-litreview)

**Status: DESIGN, attack-round-1 complete (verdict NEEDS-MAJOR-REVISION,
2026-07-07), this revision (Rev 1) resolves every finding — see §13.1 for the
full attack record and fix map. Still not built, no GPU spent.** This
document is written to survive a second, independent adversarial attack round
before any code is written. Nothing here is a launch authorization.

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
shared eval-only instrument, an estimated Phase-1 budget of **≈24.97 GPU-h**
(Rev 1 — revised up from the pre-attack ≈19.96 GPU-h estimate by the
mandatory blend-toggle 2×2 surgery grid attack-round-1 F2 forced into Leg A,
§5.2a/§10; still under the 25 GPU-h ceiling, though the margin is now thin,
disclosed honestly in §10), and a gated, sketched-only Phase 2 (new training,
standard benchmark) if Leg A shows a real effect.

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
  (`d_model=2560,n_layers=22,d_state=128`, **2 runs only** — token-matched
  to rung-2 at 1.5B tok/run, ETA ≈2026-07-08, not yet harvested at draft
  time). **d_state steps 64→64→128→128 across the ladder — not constant** —
  load-bearing for §6's K-sweep stratification below.
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
  already uses. `DeltaNetLM.forward(token_ids, initial_states=...)`
  (`lm_pretrain_rd.py`) accepts/returns per-layer recurrent state, so a
  BIND-phase forward pass can be truncated and its final state captured
  without modifying the architecture.

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

### 4.2 Episode construction

Each episode: `K` entities drawn from the entity pool (§4.5, Rev 1: probe-
training and probe-eval episodes draw from the SAME 107-name pool, disjoint
only in EPISODE-seed range, not entity sub-pool — the arithmetic-impossibility
finding that forced this correction is recorded in §13.1 F1), bound by a
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

### 4.3 Hop depths and K sweep — stratified by `d_state`, not held at a fixed K/d fraction

Every tested `K` must satisfy `K > h_max = 4` (the single-K-cycle guard
against periodicity collapse — trivially satisfied everywhere in this
design since the smallest tested K is 8, but stated as a hard config-time
assertion, mirroring `TaskEConfig.__post_init__`'s own convention, never
left to informal discipline).

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
assuming one K/d fraction transfers across `d_state` values):

| `d_state` | K values (crossing K/d ≈ 0.125 / 0.25 / 0.5 / 0.75) | Checkpoints using this `d_state` |
|---|---|---|
| 64 | 8, 16, 32, 48 | Leg A (all 3 arms); Leg B 14M + 98M rungs |
| 128 | 16, 32, 64, 96 | Leg B 392M + 1.31B rungs |

The `K=32`-at-d=64 and `K=64`-at-d=128 points sit closest to the located
`x0≈0.5455` cliff center (`0.5455×64≈34.9`, `0.5455×128≈69.8` — the K=48/K=96
points intentionally overshoot past the cliff on the "collapsed" side at
d=64, per `KEY_ANCHORING_DESIGN.md` §11.12's own K=48 capacity-curve
convention, while at d=128 they sit **inside** the window §13.10 already
showed does NOT reproduce the d=64 cliff on the coherence/rank instrument —
this makes the d=128 leg a **direct, independent test of whether the
absolute-capacity reading also holds for a genuinely different observable**
(reasoning recovery, not Gram-deviation/rank), not merely a repeat of §13.10
on a new metric.

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
`H_train=(5,)` — a single fixed placeholder chosen because `5 < 8 =
min(K)` across the entire swept range `{8,16,32,48,64,96}`, so `5 mod K = 5`
never collides with `{1,2,3,4}`'s residues at any swept K, satisfying the
dataclass's structural invariants without asserting any false "trained-hop"
semantics.

### 4.4 Readout — three options enumerated, one chosen with registered justification

**Option 1 (PRIMARY) — hooked fast-weight composition + trained
embedding-space probe, cosine>0.9.** Run the pretrained checkpoint forward
over the BIND phase only (its own natural, learned β via `b_proj` — **not**
hard-masked, since the checkpoint cannot be retrained; this is itself part
of what's being tested, §7 throughout), capturing the per-layer final
recurrent state `S_T` via `chunk_delta_rule`'s `output_final_state=True`
(`DeltaNetLM.forward(..., initial_states=...)` already supports this
split, no architecture change). Extract `q_eff` at the query position via
the same non-invasive post-conv, pre-kernel hook `lm_attractor_probe_rd.py`
already uses for `k_eff` (the query span passes through the feature path
only, never re-entering the recurrence — the same split
`DELTANET_REALDATA_DESIGN.md` §5.2 and `NEXT_EXPERIMENT_DESIGN.md` §2 both
already use). Compute `pred(a,h) = S_T^h @ q_eff_a` externally (plain
matrix power in `d_state`-dim space, cheap). Score via a **trained linear
probe** `W_probe: R^{d_state} → R^{d_state}` mapping `pred(a,h)` into the
checkpoint's own input-embedding space, cosine-scored against
`W_embed[value_token_id]`, threshold **0.9** (this project's standing
convention throughout Task D/E, DeltaNet-causal-rank, DELTANET_REALDATA,
F-geo-3). **This is the direct, minimal-new-code continuation of the
already-validated `pred(a,h)=S_T^h·q_eff_a` formula** (`DELTANET_REALDATA_DESIGN.md`
§5.1), now applied eval-only to checkpoints that never trained on the task —
the probe (not the checkpoint) supplies the "does this project's own
readout convention still decode something" bridge, since a general LM's raw
key/value subspace geometry cannot be assumed a priori to align with the
embedding table the way a from-scratch, hard-β-masked model's does.
**Continuous, never argmax over a codebook — satisfies the CLAUDE.md hard
rule directly.**

**Option 2 (SECONDARY, co-registered, complementary not substitutive) —
fully natural, black-box next-token logit margin.** Run the *entire*
pretrained model (all layers, residual stream, tied LM head) forward over
BIND clauses + query clause as one ordinary continuous sequence, no hooks,
no truncation. Score `margin = logit(true_answer_token) −
max_j(logit(distractor_j))` over the `K−1` other in-context entities,
continuous, no argmax needed for the *metric* (only the distractor set is
enumerated, the score itself is a real-valued margin). **Confound, stated
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

**Justification for choosing Option 1 as primary:** it is (a) the direct
continuation of this project's own validated readout formula, (b) the only
option satisfying CLAUDE.md's exact-continuous-recovery rule without any
enumeration-based scoring, and (c) the cheapest to build correctly (reuses
existing hooks, no architecture patching). Option 2's cost is a full model
forward pass anyway (already needed to obtain `k_eff`/`q_eff` along the way)
so it is nearly free to also report — hence co-registered, not deferred.

### 4.5 Probe training protocol — closing the probe-leakage shortcut by design, not by assumption

**Rev 1 (attack-round-1 F1) — the pre-attack entity sub-split was
arithmetically impossible at the design's own decisive cells and is
replaced below; full finding recorded at §13.1.** `build_entity_pools()`
(`grammar_rd.py`, verified directly this revision) yields 213 GPT-2-BPE-
verified single-token names split 107 train / 106 heldout
(`heldout_frac=0.5`, `n_heldout=round(213×0.5)=106`). The pre-attack draft
proposed carving the 107-pool itself into a further ≈60/47 probe-train /
probe-eval sub-split — but `sample_batch_rd` draws its `K` entities WITHOUT
replacement from whichever pool it is given and hard-asserts `N_names ≥ K`
(`grammar_rd.py` L425-428). A 47-name probe-eval sub-pool cannot construct
`K=48` (d=64 killer-prediction cell, §5.3) or `K=64`/`K=96` (d=128 rungs,
§4.3) episodes at all — the sub-split made the design's own decisive cells
unbuildable.

**Corrected construction (Rev 1): no entity-level sub-split for probe-
train/probe-eval.** Both probe-TRAINING episodes and probe-EVAL (headline)
episodes are drawn from the **same, full 107-name train pool**
(`use_heldout_entities=False`), using **disjoint episode-seed ranges** (a
registered, non-overlapping RNG-generator-seed offset per role — e.g.
probe-training episodes seeded from offset `0`, probe-eval episodes from a
disjoint offset block never reused for training) rather than a disjoint
entity sub-pool. Every swept `K` (8/16/32/48 at d=64; 16/32/64/96 at d=128)
fits comfortably under 107 with margin — the arithmetic-impossibility
finding is closed by construction, not by picking smaller K values.

**Memorization-ceiling control (new purpose for the existing, already-
disjoint heldout pool — the attack's own recommended fix).** The 106-name
`heldout_name_ids` pool (`use_heldout_entities=True`) — already fully
disjoint from the 107-name train pool by `build_entity_pools`' own
construction, and never drawn from during probe training under the
corrected protocol above — is reserved for a separate diagnostic: score the
SAME frozen probe on episodes drawn entirely from this held-out pool. This
is a **strictly stronger** generalization check than the discarded entity
sub-split (it tests entities the probe's weights had zero exposure to at
ALL, under ANY permutation, rather than merely a disjoint slice of episodes
built from overlapping entities), and every swept K (max 96) fits within 106
with margin. Reported as a covariate/robustness check, not the headline.

**Why probe-leakage is still closed without the entity sub-split — stated
explicitly, not assumed (Rev 1 honesty requirement).** The residual concern
a hostile reader would raise: if probe-training and probe-eval episodes can
share entities, can the probe simply memorize "this raw vector, near token
X's own embedding, predicts value Y"? It cannot, because `Y` is not a fixed
fact about entity `X` in this design: every episode draws a FRESH random
single-Hamiltonian-K-cycle (`_permutation_graph`, resampled per episode,
§4.2) and a fresh random query source slot (`a_slot`), so the (key, value)
pairing any given entity participates in is essentially never repeated
across two different episodes even when the SAME entity token recurs — with
`K≥8`, the number of possible K-cycles is `(K−1)!/2 ≥ 2520`, making exact
episode recurrence (same entity draw AND same cycle AND same query)
vanishingly unlikely across the disjoint probe-train/probe-eval seed ranges.
A probe therefore cannot learn "entity X → value Y" as a fixed lookup; it
can only learn the general `pred(a,h) → W_embed[value]` mapping this design
exists to test. The narrower, real residual risk — could the probe instead
learn some entity-IDENTITY-correlated property of recoverability itself,
independent of the specific pairing (e.g., some tokens are systematically
"easier" regardless of role) — is exactly what the two existing controls
already close: **(a)** the label-shuffle-null memorization-ceiling in the
SECONDARY protocol below (permutes the (key,value) pairing across episodes
while holding the entity pool fixed; if recovery survives THIS shuffle, that
specifically indicates an identity-correlated shortcut, not composition —
§8.4's null-relative pass bar exists precisely to catch it), and **(b)** the
new heldout-pool memorization-ceiling control above (no degradation on
entities absent from probe training at all is the positive check that
identity-specific memorization isn't occurring). Both controls are
pre-registered and reported alongside the primary reading, never used to
retroactively rescue a headline number.

**PRIMARY protocol — one shared, arm-blind probe per `d_state`.** Train a
single linear probe on episodes drawn from a **reference condition** (the
"off"/control arm's checkpoint for Leg A; an early, still-converging rung-1
checkpoint or the same "off"-arm-style reference for Leg B, since Leg B has
no explicit "off" arm — decide at build time, registered choice: use the
14M mixcontrol final checkpoint as the shared `d_state=64` reference probe,
and the 392M rung-2 final checkpoint as the shared `d_state=128` reference
probe, since these are each leg's own largest same-`d_state` checkpoint with
no frozen-bias intervention applied). Freeze this probe; apply it
**identically** (same weights) to every arm/checkpoint/rung sharing that
`d_state`. **This is the headline reading** — any measured difference
between arms/rungs is attributable to the checkpoint's own representational
geometry, not to a probe re-fit per condition.

**SECONDARY protocol — per-arm/per-rung probes + a label-shuffle control
task (Hewitt & Liang, EMNLP 2019, "Designing and Interpreting Probes with
Control Tasks").** Fit a separate probe per arm/checkpoint on that
checkpoint's own probe-train split (giving each condition its fairest
possible chance to express whatever geometry it has), **and** fit the
identical probe procedure on a label-shuffled null (bind-clause `(key,
value)` pairing permuted across episodes, keeping surface form and token
statistics fixed) — report eval-split recovery on the null as the probe's
own **memorization ceiling**. A per-arm/per-rung headline number is treated
as informative only to the extent it clears its own null by a registered
margin; **never reported as the primary bar**, exactly the same
discipline this project already applies to its val-loss gate (a passing
gate is necessary, not sufficient, for a headline claim).

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
`S_T^h·q_eff` cosine-probe readout) and Option 2 (§4.4, natural next-token
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

**Registered 2×2 grid (3 of 4 cells populated — the 4th is not a real
cell, named rather than silently dropped):**

| | blend ON at eval (native forward) | blend OFF at eval (surgery: `frozen_bias_arm` forced to `"off"`) |
|---|---|---|
| **off-arm checkpoint** | = blend OFF (the off arm never trains a `frozen_bias_table` — there is nothing to blend) — the one cell, reported once | *not constructible — no table exists to force on; inventing one would compare against a table the off arm never trained against* |
| **arm checkpoint (global or per_token)** | pre-attack §5.2 reading, mechanically confounded per this finding | **new cell this fix adds** |

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
concentrated at K∈{32,48} (near and past the located `x0≈34.9` cliff for
`d_state=64`) and small-to-absent at K=8 (well inside the "capacity is not
the thing that fails first" regime, `DELTANET_REALDATA_DESIGN.md` §16.4).
Operationalized: report `Δ(K)` (the global-vs-off **training-effect** delta,
§5.2a) at every tested K separately; the pre-registered killer-prediction
pass condition is `|Δ(K=32 or 48)| > |Δ(K=8)|` with the larger-K delta's CI
excluding zero while K=8's does not, replicated in at least one corpus, AND
(Rev 1, M1) Option 1/Option 2 directional agreement per §5.2 at the passing
K. This ties the probe **directly** to the capacity law rather than merely
re-measuring "does an intervention move some LM metric" — a failure of this
specific prediction (arm separation flat across K, or concentrated at *low*
K) would be a genuinely informative negative about whether the capacity
cliff has any behavioral teeth at all, distinct from whether the
intervention has *any* effect. **Rev 1 mandatory zero-cost covariate (M1):**
at every headline cell (this one included), also report `S_T`'s condition
number / eigenvalue spread and a within-episode cross-`a` cosine-convergence
check — see §8.3.

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
| 1.31B rung-3 | ≈1.31B | 128 | **2 runs only**, not yet harvested | 2 final (pending ≈2026-07-08) | **descriptive only, no CI** |

**Rung-3 caveat, stated plainly and registered now, before rung-3 data
exists (mirroring the project's own standing discipline of naming a
limitation before it can be seen as post-hoc):** with only 2 runs (likely 1
seed × 2 corpora, or 2 seeds × 1 corpus — confirm against the actual launch
manifest at harvest time), the pinned `t(2,.975)` formula does not apply.
Rung-3's contribution to H_LINK-B is a **point estimate plus qualitative
direction only** — consistent with how this project has already handled
every other n=1/n=2 cell (the λ-mini-sweep, the H4 single-seed parameter-diff
check) — never a CI-based confirm/refute on its own.

### 6.2 Comparisons and pre-registered reading

Primary: `recovered_frac@0.9` at held-out hops (h=3,4), at the
`d_state`-matched near-cliff K (K=32 for the two `d=64` rungs, K=64 for the
two `d=128` rungs — the closest tested point to each `d_state`'s own
`0.5455×d` cliff-adjacent load), reported across the 4-rung ladder.
**Pre-registered reading:** monotone non-decreasing recovery across
14M→98M→392M→(1.31B, descriptive), and a positive rank correlation
(Spearman) between per-rung recovery and the already-measured span_frac
value at that rung (0.248/0.344/0.389/[pending]) — reusing exactly the
Spearman-ρ convention this project already applied to the Task D
rank-vs-K result (`ρ=1.0`, `STATE.md` Chapter 2 section).

### 6.3 What Leg B does NOT claim

Because `d_state` changes mid-ladder (64→128 at the 392M rung), a raw
cross-`d_state` recovery comparison at a FIXED K would conflate scale with
absolute capacity change — this design does not make that comparison. The
K=32-vs-K=64 near-cliff comparison above is **K/d-matched**, not
raw-K-matched, precisely to keep the scale claim about the same *relative*
load position at each rung's own capacity regime. A secondary, explicitly
labeled reading (raw K held fixed at, e.g., K=32 across all 4 rungs
including the two `d=128` ones) is reported alongside as a "what happens if
you don't correct for `d_state`" sanity contrast, never the headline.

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

### 7.4 Probe-leakage

Addressed structurally in §4.5 (arm-blind shared primary probe + label-shuffle
control-task null on the per-arm secondary reading). Restated here as the
registered decision: **the primary bar never uses a probe fit on the same
condition it is scored against.** A hostile reading of this design ("you
just fit a probe until it found what you wanted") is closed by the fact that
the headline probe is fit once, on the reference condition, frozen, and
never re-touched after seeing any arm's or rung's actual eval numbers — the
exact same "pin the reference before seeing the intervention arm" discipline
`FROZEN_BIAS_LM_DESIGN.md`'s own blind-pin was *supposed* to have (and, per
its own VERDICT section, the sequencing failure that demoted that wave to
descriptive tier). **Process commitment carried forward from that lesson:**
the probe weights (and every threshold in §8) must be written to disk and
committed **before** the full Leg A/B grid launches, not constructed after
training-cell results already exist — this design explicitly inherits that
process fix rather than repeating the sequencing mistake.

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
memory change anything about how `k_eff`/`q_eff` are extracted at the
conv/`W_k` stage this design hooks? **Answer, stated honestly: no
additional control is needed beyond what already exists** — the hook point
(`k_conv1d` output, pre-kernel) is identical in shape and code path across
all three arms (verified directly from `lm_pretrain_rd.py`, §0's reading
list); the frozen table only enters the computation graph via the
key-blend arithmetic itself, which is exactly the quantity under test, not
an artifact to be controlled away.

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

### 8.1 Primary metric

`recovered_frac@0.9` — fraction of eval episodes with
`cos(probe(pred(a,h)), W_embed[value_token]) > 0.9`, per (arm-or-rung,
corpus, K, h) cell, Option 1 readout (§4.4), the SAME threshold this
project's entire composition-study lineage uses (Task D/E,
DELTANET_REALDATA, F-geo-3) — no new threshold introduced without cause.

### 8.2 Secondary metrics

- Option 2 next-token logit margin (§4.4), reported alongside, never
  load-bearing alone.
- Per-arm/per-rung probe + label-shuffle-null delta (§4.5), reported as a
  robustness cross-check on the primary shared-probe reading.
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
is scrambled), score `recovered_frac@0.9` under the SAME probe and readout,
and bootstrap a 95% band for this null. **Registered pass condition:** the
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

### 8.5 Degenerate/exclusion rules, with counts (registered categories)

1. **Premise-invalid checkpoints**: `k_eff`/`q_eff` at bind/query positions
   are near-zero-norm (`<1e-6`), non-finite, or produce a probe-training
   design matrix with condition number `>1e6` (probe cannot be fit
   meaningfully). Excluded from headline, reported as its own category with
   an exact count, never silently dropped.
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

### 8.6 Multiple-comparison correction for the non-primary grid (Rev 1 minor, attack-round-1)

This design runs ONE pre-registered, externally-anchored primary test — the
§5.3 killer-prediction contrast (training-effect Δ at K∈{32,48} vs. K=8, in
at least one corpus, now also requiring Option 1/Option 2 agreement per
§5.2/M1). That cell's CI-based CONFIRM/REFUTE reading is exempt from any
multiple-comparison correction: it is a single hypothesis, anchored to an
externally-located quantity (`KEY_ANCHORING_DESIGN.md`'s own `x0≈0.5455`
cliff), not one draw from a scanned family. **Every other cell in the
grid** — every other K, h, corpus, arm/rung combination reported across Leg
A's and Leg B's full sweeps — constitutes a non-primary grid of many
simultaneous comparisons, and is reported under Benjamini-Hochberg FDR
control (`q=0.05`, this project's standard significance convention) rather
than raw per-cell CIs, so that scanning many K/h/corpus/arm cells does not
manufacture spurious "detections" through multiple comparisons alone. Which
cells counted as "the non-primary grid" for BH purposes is fixed before any
data exists (every headline cell in §5/§6 other than the single K∈{32,48}
killer-prediction contrast) — not chosen post-hoc after seeing which cells
look interesting.

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
4. The label-shuffle null generator (§4.5/§8.4) — verify it actually
   destroys the true pairing (a planted-recoverable check: shuffle a
   hand-built K=4 episode, assert the previously-correct answer no longer
   scores above threshold).
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
   zero-GPU.
6. **(Rev 1, attack-round-1 F2) Surgery-toggle equivalence smoke** — verify
   that forcing `frozen_bias_arm="off"` on a LOADED arm checkpoint and
   running `k_proj`→`k_conv1d` (same direct-submodule-call approach as item
   5, not a full `model(x)` forward) reproduces
   `frozen_bias_retrofit_eval_rd.py`'s own `"kraw"`-mode raw-key capture
   byte-for-byte (`torch.equal`) on a tiny smoke batch — the mechanism
   §5.2a's 2×2 grid depends on, gated here before it is trusted for the
   real grid, and (per the same direct-submodule-call approach) also
   zero-GPU.
7. **(Rev 1, attack-round-1 M4) `conv_size`-match assertion** — for every
   checkpoint family, load the checkpoint's own saved config and assert
   `checkpoint_config["conv_size"] == episode_cfg.conv_size` (§4.1) before
   any episode is generated for that checkpoint.
8. **(Rev 1, attack-round-1 M5) Render-adjacency/cycle-adjacency correlation
   measurement** — generate ≥10,000 episodes per swept `K` via the real
   `sample_batch_rd`, measure `succ[i]∈{i−1,i+1}` empirical rate against
   the exact combinatorial chance rate, bootstrap a 95% CI (§4.2). Pure
   episode-generation arithmetic, no checkpoint needed, zero-GPU.

**Gate:** Stage 0 (calibration) may not launch until all Stage −1 tests
pass — the exact "specification that has not been executed is not a passed
gate" convention this project applies everywhere.

**Stage 0 (calibration, 1 cell, blinded from any headline decision):** the
14M mixcontrol "off" arm, seed 0, one corpus, K=32, h∈{1,2,3,4}, BOTH query
markers (§7.6's robustness replicate). Purposes: (a) fix the probe's
hyperparameters (regularization strength, training-episode count) once,
before touching any other checkpoint; (b) measure real per-cell wall-clock
cost to firm up §10's budget; (c) run the §8.4 null-band construction for
real, **at every h∈{1,2,3,4} independently (Rev 1, attack-round-1 M2 — not
h=1 only)**, and confirm the h1 floor is achievable at all before committing
to the full grid — if it is not, this is itself a decisive, informative
result (the whole probe is uninterpretable for this checkpoint family) and
the grid does not launch; the h=2/3/4 null bands become the registered
per-h chance floors §8.4 now requires before a CONFIRM/REFUTE reading at
those hops is licensed.

**Stage 1 (full grid):** Leg A (18 core cells × 4 K × 4 h) + Leg B (14
checkpoints × 4 K × 4 h, per §6.1's rung-appropriate K sets), per §10's
budget.

---

## 10. Budget

**Unit cost anchor (measured, not assumed):** this project's own
retrofit-eval instrumentation on the 14M/rung-1-scale architecture costs
**≈0.0348 GPU-h per checkpoint-pass** (`FROZEN_BIAS_LM_DESIGN.md` §12.1:
1.6 GPU-h / 46 passes, forward-pass-only). The scale-transfer harvests give
two more real anchors at larger scale: **≈0.013 GPU-h/pass at 98M**
(0.08 GPU-h / ~6 passes, §5.9) and **≈0.045 GPU-h/pass at 392M**
(1.04 GPU-h / ~23 passes, §5.10) — i.e., per-pass probe cost at THIS
project's scales has not scaled up badly with params (batching/short
sequences dominate). This design's own probe sequences are longer
(`K` bind clauses instead of a handful of corpus windows) — budgeted at a
flat **2× multiplier** over the largest same-scale anchor as a conservative
per-cell estimate, plus a further **2× contingency** matching this
project's own standing "budget with a 2× contingency multiplier" rule
(`SCALE_TRANSFER_DESIGN.md` §8.4).

| Item | Cells | Anchor cost/cell | ×4 (2×seq-length × 2×contingency) | Subtotal |
|---|---|---|---|---|
| Leg A native grid (18 core cells × 4 K, `d=64` anchor — the pre-attack reading, now the "blend-ON" cell of §5.2a's 2×2) | 72 | 0.0348 | 0.1392 | 10.02 GPU-h |
| **Leg A surgery grid, NEW (Rev 1, attack-round-1 F2) — 12 arm-only cells (global+per_token, 2 corpora × 3 seeds) × 3 K {8,32,48}, `d=64` anchor** | **36** | **0.0348** | **0.1392** | **5.01 GPU-h** |
| Leg B, `d=64` rungs (12 checkpoints × 4 K) | 48 | 0.0348 (14M) / 0.013 (98M), use 0.0348 conservatively | 0.1392 | 6.68 GPU-h |
| Leg B, `d=128` rungs (8 checkpoints × 4 K, 392M anchor) | 32 | 0.045 | 0.18 | 5.76 GPU-h |
| Stage −1 self-tests | — | 0 (CPU only — including Rev 1's new items 5-8, per §9's direct-submodule-call build approach) | — | 0 |
| Stage 0 calibration (1 cell × 2 markers, now scored at all 4 h per M2) | 2 | 0.0348 | 0.1392 | 0.28 GPU-h |
| Probe training (per shared reference probe, 2 probes total + the new heldout-pool memorization-ceiling draw, §4.5 Rev 1 — covered by the same generous per-probe estimate, not separately priced) | 2 | ≈0.05 (generous, includes forward passes to generate probe-train episodes) | — | 0.10 GPU-h |
| **Total, Phase 1 (pre-trim)** | | | | **≈27.85 GPU-h at the full 4× conservative multiplier** |

**Rev 1 (attack-round-1 F2) forced a real budget increase, not merely a
re-derivation.** §5.2a's 2×2 blend-toggle surgery grid is unavoidable at
full K resolution for the training-effect contrast that now carries the
headline H_LINK-A claim (§5.2) — there is no way to compute that contrast
without the new "blend-OFF surgery" pass for the 12 arm-checkpoint cells,
at (at minimum) the K values the §5.3 killer prediction itself needs
(K=8 as the low-load contrast point, K∈{32,48} as the near-cliff decisive
points). This raises Leg A from 10.02 to a combined **15.03 GPU-h**
(10.02 native + 5.01 surgery), and the pre-trim Phase-1 total from the
pre-attack ≈22.8 to **≈27.85 GPU-h** — already re-derived with ONE
disclosed trim applied (K=16 deferred from the surgery-only pass, below),
not the full un-trimmed 48-pass/6.68 GPU-h surgery addition, which would
put the pre-trim total at **≈29.5 GPU-h**.

**Three pre-registered, disclosed trims bring it back under the 25 GPU-h
ceiling** — the first two carried over unchanged from the pre-attack
design, the third new this revision:
1. Drop the optional trajectory read (§5.1) — already excluded from the
   table above, saves nothing further since it was never counted in, but
   confirmed here as NOT part of the Phase-1 committed number.
2. Use K∈{8,32} only (2 points: inside-capacity and near-cliff) for the
   `d=128` Leg B rungs' first pass, reserving K∈{16,64} as a
   pre-registered, separately-priced extension gated on the 2-K pass
   showing a signal worth resolving further — halves the `d=128` subtotal
   to **2.88 GPU-h**.
3. **(Rev 1, new)** Defer the K=16 point from the Leg A **surgery-only**
   pass — of the 4 swept K values at d=64 (8/16/32/48), K=16 is the one not
   named in §5.3's killer-prediction pass condition (which needs exactly
   K=8 and K∈{32,48}); the ORIGINAL native/blend-ON reading is retained at
   all 4 K (preserving the full curve for the mechanically-confounded
   secondary reading), and only the NEW surgery pass's K=16 cell (12
   arm-cells × 1 K = 12 passes, 1.67 GPU-h) is deferred as a
   pre-registered, separately-priced extension — mirroring trim #2's own
   "defer the least-decisive point, not the decisive ones" logic. This
   drops the surgery grid from 48 to the **36 passes/5.01 GPU-h** already
   reflected in the table above.

**Resulting Phase-1 committed total: ≈24.97 GPU-h** (15.03 Leg A + 6.68 Leg
B d=64 + 2.88 Leg B d=128-trimmed + 0.28 calibration + 0.10 probe training),
just under the 25 GPU-h ceiling. **Stated honestly: the margin is now
≈0.03 GPU-h — far thinner than the pre-attack design's own ~25% cushion,
because F2's mandatory addition consumed nearly all of it.** This is not
papered over: if Stage 0's real measured per-pass cost comes in even
slightly above the 0.0348 GPU-h anchor, this trimmed total will exceed the
ceiling before Stage 1 launches. The abort rule below (unchanged) is the
real safety valve, not this arithmetic margin. If Stage 0's real cost
requires a further cut, the next lever to pull (named now, not
discovered under pressure) is deferring K=16 from Leg B's `d=64` rungs'
grid as well (12 checkpoints × 1 K = 12 passes, ≈1.67 GPU-h), mirroring
this same revision's own trim #3 — a fourth trim, not yet applied, held in
reserve.

**Ceiling: 25 GPU-h** (retained unchanged from the pre-attack design —
Rev 1's trims target fitting under this existing ceiling, per the task
brief's own instruction, rather than raising it). **Abort rule:**
if the Stage-0 calibration cell's real measured wall-clock cost exceeds
**3×** the 0.0348 GPU-h anchor (i.e., the sequence-length multiplier
assumption was wrong), halt before Stage 1 launches and re-price the full
grid from the real number, per this project's own standing "a calibration
run... catches convergence ceilings... before you commit a sweep's compute
to it" rule.

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
3. **Does the shared, arm-blind probe (§4.5 primary) have enough capacity
   to decode a stabilized (global-arm) checkpoint's geometry if that
   geometry is meaningfully DIFFERENT from the reference ("off") checkpoint
   it was trained on?** Best current answer: this is a real tension in the
   design — the arm-blind probe is the most defensible against
   probe-leakage (§7.4) but is also the most likely to systematically
   *under*-report a genuinely different-geometry arm's true capability,
   because a linear probe fit on one basis need not transfer perfectly to
   a rotated/rescaled one. Partially mitigated by the per-arm secondary
   reading (§4.5) with its own label-shuffle null, but this is registered
   as an unresolved tension, not a solved problem — a negative Leg-A result
   under the primary probe alone should be cross-checked against the
   secondary before being reported as REFUTE.
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
- [x] **(Rev 1)** Every K value in the sweep (8/16/32/48/64/96) is
  constructible from its entity pool (107-name primary, 106-name
  memorization-ceiling control) with margin — the arithmetic-impossibility
  gap attack-round-1 F1 found is closed by construction.
- [x] No compressing matrices to vectors — `S_T` and `pred(a,h)` stay
  `d_state`-dimensional throughout; the probe maps `d_state→d_state`, never
  through a flattened intermediate.
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
