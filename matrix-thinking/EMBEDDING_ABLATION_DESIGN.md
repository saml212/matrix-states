# Parameter-Matched Matrix-vs-Flat-vs-Flatten Embedding Ablation — Pre-Registration

**Status:** DESIGNED, NOT LAUNCHED. REV-narrow (post-audit revision).
Queue specs staged only (`matrix-thinking/embed_ablation_specs/
phase_A_probes/0640-0645` + `phase_B_seeded/0646-0669`). Nothing in this
document authorizes a GPU launch; it is a pre-registration for the
research-cascade novelty/audit gate to review first.

**Author:** builder agent, 2026-09-01. **Concurrent-edit note:** this
design does not touch `matrix-thinking/deltanet_rd/h2h_strengthen_rd.py`
or `HEAD_TO_HEAD_DEMO_DESIGN.md` (both under edit by another agent as of
this writing) and shares no code with either.

## Audit changelog

### Round 1 (F1/F2 FATAL, M1-M4 MAJOR, minor)

| Item | Fix |
|---|---|
| F1 (FATAL) harvest | Rewrote `harvest()`/`_harvest_records()`: filters to `complete is True AND steps_target>=2000 AND role!='probe' (role field OR filename)`; asserts exactly 3 valid records per `(arm,match,size)` group (0 = not run yet, fine; 1-2 or 4+ = FATAL, raises); added `--harvest-selftest` that builds the exact synthetic scenario (15 clean cells across 5 groups + 2 probe-like records + 1 `CEILING_STOP` record) and proves the verdict is byte-identical with/without the 3 extras (§8b). Probe results now write to a separate `results/probes/` subdirectory. |
| F2 (FATAL) gating | Specs split into `embed_ablation_specs/phase_A_probes/` (0640-0645, no dependencies) and `phase_B_seeded/` (0646-0669), each with its own `README.md` stating phase B is staged only after phase A's 4→**6** probe JSONs are read and `--check-admission` passes (re-deriving `--steps` if the measured rate is off). Same sentence restated in §6 below. |
| M1 (MAJOR) eval windows | `evaluate()` now reseeds a **fresh** `torch.Generator` from `corpus_fixed_seed(corpus_name)` (the `lm_pretrain_rd.py` AUDIT FIX-1 pattern) at the **start of every call** — every seed, arm, and T-leg scores identical validation windows in identical order. `--eval-batches` default raised 20→50. |
| M2 (MAJOR, RULING) init | `TARGET_STD=0.02` applied explicitly: matrix arm's `embed_u,embed_v,pos_u,pos_v` init at `std=sqrt(target_std)` (CLAUDE.md's outer-product rule); flat/flatten arms' embedding+pos tables init at `std=target_std` directly; flat/flatten's positional term now scaled by `*0.1` to match the matrix arm's own scaling (previously unscaled — an accidental asymmetry). §4.4 records both, and that `round1_vector_script.py`/`round2_matrix_script.py` had **no explicit init at all** (a historical gap this design fixes, not merely inherits). |
| M3 (MAJOR, RULING) third arm | Added the **`flatten`** arm (§2.1b, §4): the matrix arm's own outer-product embedding, flattened to a `d²` vector, resized into the flat arm's own dense backbone family, `d_model` solved to ±1% params. The pre-registered claim is now **two independent decisions**: `STRENGTHEN-01` (matrix vs flatten — embedding mechanism, finding 01) and `STRENGTHEN-04` (matrix vs flat-P — architecture at equal params, finding 04). §1 and §7 rewritten accordingly. Ledger grows from 22→30 cells, 15.622→**20.434 GPU-h**, still ≤30 GPU-h total and ≤2 GPU-h/cell. |
| M4 (MAJOR) admission | Added `check_admission()` / `--check-admission`: reads phase-A probe JSONs, asserts the last three T=1 evals are monotone non-increasing per arm, extrapolates the measured rate to the intended `--steps`, and reports/stops if that would exceed 2 GPU-h/cell. §6 and both phase READMEs state this is mandatory before phase B is staged. |
| minor | `LambdaLR`'s warmup fixed (`(step+1)/warmup`, not `step/warmup` — step 0 no longer trains at LR=0); `CEILING_STOP` now forces one more eval pass before saving so `final_evals` is never stale/empty; the empirical anchor is corrected to `exp_d16_v2`'s **actual** `CONFIG`/class-default values — `n_thinking_layers=12`, `max_len=2048` (the historical `SUMMARY.txt`'s own prose, "8 layers", was wrong — verified by reading the script's `CONFIG` dict and `MatrixThinker.__init__` default directly, not the stale summary text); `_gen_specs.py`'s dead `f"T{8 if steps > 500 else 8}"` ternary replaced with the literal `T8`; ~3.3 GB estimated host RAM per process disclosed (§5). |

### Round 2 (no FATAL, six MAJOR, minor) — REV-narrow

| Item | Fix |
|---|---|
| MJ-1 (MAJOR) probes | Every phase-A `cmd` now carries `--eval-interval 100` (was the 250 default) — at `--steps 500` that is 5 eval points (100,200,300,400,500), not 2. `check_admission()` now asserts `len(t1_seq) >= 3` per probe and FAILS LOUDLY (prints the short curve, sets `all_ok=False`) rather than silently judging monotonicity on whatever few points happened to exist. |
| MJ-2 (MAJOR) admission set | `check_admission()` now first computes the SET of `(arm,size)` found among the probe records' own fields and asserts it equals `{matrix,flat,flatten}×{S,M}` (6 combos) exactly — a probe that crashed and wrote nothing (or wrote a record missing `arm`/`size`) silently shrinks this set; that is now a loud, independent FAIL, checked before any per-probe (a)/(b) checks. |
| MJ-3 (MAJOR) flatten heads | `solve_matched_width()`'s candidate order for the flatten arm changed from `(8,4,2,1)` to `(registered_n_heads, 2, 1, 8)` — since `nn.MultiheadAttention`'s param count is head-count-invariant, trying the registered `n_heads` first can only find an equal-or-better match. Size S now lands on `n_heads=4, d_model=16` (IDENTICAL 2,452,544 params to the first pass's `n_heads=8` answer — same width, more conventional head count, `head_dim=4`). Size M is unaffected and remains disclosed as forced to `n_heads=1, d_model=25` (no multiple of 4 or 2 clears ±1% there). `head_dim` is now printed in `--selftest` for both flat-P and flatten-P. |
| MJ-4 (MAJOR) resize_in init | `FlattenThinker.resize_in` now has explicit init: `bias` zeroed, `weight ~ N(0,(1/mat_dim)²)` — derived (§4.4) so the backbone's input lands at `std ≈ TARGET_STD`, matching the matrix arm's own `M` and the flat arm's own embed output. `--selftest` now prints backbone-input std for ALL THREE arms at both sizes (target `≈0.02`) via a new `_embed()` helper each class exposes. |
| MJ-5 (MAJOR) steps coupling | `harvest()` now filters to `steps_completed == steps_target` (exact, replacing the first pass's `steps_target>=2000` threshold) and asserts a SINGLE shared `steps_target` across all valid records (raises if cells trained to different step budgets are ever mixed). `_gen_specs.py` now generates every `--steps`/eval-interval value AND every validity_check's exact-match threshold from ONE constant per phase (`STEPS_A=500`, `STEPS_B=2000`) — both READMEs state "`--steps` and the validity_check always move together." |
| MJ-6 (MAJOR) phase-A/B vcheck | Every `validity_check` (both phases) now asserts `steps_completed == STEPS_{A,B}` EXACTLY — the first pass's `>= steps - 1` slack is gone from both phases. Phase-A also now asserts `d.get('complete') is True` (not just `'T1' in final_evals`). |
| minor | `harvest()` also requires `status=="COMPLETED"` (redundant with `complete is True` given how `run_cell` sets both, but asserted explicitly, defensively) and asserts, per group, `seeds == {0,1,2}` exactly; and, across ALL valid records globally, that `corpus`/`batch_size`/`seq_len` are each a single shared value. `check_admission()` now also requires `t1_seq[-1] < t1_seq[0]` (net improvement over the whole probe, not just local non-increase). §6b gets the `ssh … mkdir -p /home/nvidia/embed_ablation` step before the `scp`. §4.2 discloses `embed+head` is 95.5%-99.0% of total params in every arm (96.6%-99.0% for the three `-P`/primary arms) and that `--lr 3e-4` is shared, untuned, and never swept for any of the three architectures. |

## 0. What this fixes

`pebble-ai-site/findings/outer-product-embedding.html` (finding 01) and
`pebble-ai-site/findings/parameter-efficiency.html` (finding 04) are the
project's two earliest published claims. Both rest on comparisons that,
on inspection of the raw evidence, are weaker than their own text admits:

- **Run 18** (`EXPERIMENT_LOG.md` line 673): matrix 2.4M params vs flat
  24.0M params (**10x asymmetric**, flat favored). The finding page itself
  flags this as "unfair on params... supporting data, not headline." This
  is also, architecturally, the historical ancestor of this design's new
  `flatten` arm (§2.1b) — same recipe, now params-matched.
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
seeds, 2 sizes, 3 arms, a total-parameter match verified by instantiating
the real `nn.Module` (not eyeballing config numbers), a hard negative test
that a mismatch fails loudly, all arms guaranteed to finish (or record an
explicit `CEILING_STOP` with a forced final eval, never silently
truncate), a metric that is not byte BPB, and two independently-decided,
narrowly-scoped claims rather than one conflated one (M3).

## 1. Hypotheses and falsifiers (TWO independent claims, audit M3)

The original single hypothesis conflated two different questions finding
01 and finding 04 actually ask. They are now separated and decided
independently — `harvest()` computes both and never merges them.

### 1a. STRENGTHEN-01 — embedding mechanism (matrix vs flatten)

**Hypothesis:** At equal total params and equal depth, keeping the
outer-product embedding **and** matrix-native downstream ops (the full
`matrix` arm) reaches lower T=1 token-BPB than keeping the **same**
outer-product embedding but flattening it into a standard dense backbone
(the `flatten` arm, §2.1b/§4). This isolates the operation family while
holding the embedding mechanism fixed — Run 18's own historical recipe,
now genuinely params-matched.

**Falsifier:** `matrix` does **not** beat the params-matched `flatten`
arm's T=1 token-BPB, by a margin exceeding the seed spread, on at least
2 of 3 seeds, at **both** registered sizes.

### 1b. STRENGTHEN-04 — architecture at equal params (matrix vs flat-P)

**Hypothesis:** At equal total params and equal depth, the full `matrix`
arm (outer-product embedding + matrix ops) reaches lower T=1 token-BPB
than a fully `flat` arm (direct embedding table + standard ops) — varying
**both** embedding and operations at once.

**Falsifier:** `matrix` does **not** beat the params-matched `flat-P`
arm's T=1 token-BPB, by a margin exceeding the seed spread, on at least
2 of 3 seeds, at **both** registered sizes.

No third outcome is defined for either claim (see §7). Neither claim
attempts to adjudicate finding 01's own open question (rank-1 structure
vs factored parameterization — the ALBERT-style-bottleneck three-way
ablation the finding page names as its own Priority 1 — the `flatten` arm
here is NOT that ablation, it shares the embedding exactly rather than
matching only its parameter count) or finding 04's compute-side claim
(a per-layer parameter-count fact, not a trainability claim — see §2.3).

## 2. Closing the reshape-equivalence loophole

`CLAUDE.md`'s hard rule: *"any d²-dim vector can be reshaped to a d×d
matrix and vice versa. Structure only matters if OPERATIONS preserve
it. Flatten = structure gone."* Three places this bites, all closed below.

### 2.1 Embedding — matrix vs flat

- **Matrix arm:** `embed_u, embed_v: Embedding(V, d)`; token embedding is
  the explicit outer product `M = u ⊗ v`, a **rank-1** `d×d` matrix built
  from `2d` free parameters. This is `matrix_thinker.py`'s
  `MatrixEmbedding`/`round2_matrix_script.py`'s embed pattern.
- **Flat arm:** `embed: Embedding(V, d_model)`, a **direct lookup table**
  with `d_model` free parameters per token, no bilinear construction, no
  rank restriction. The flat embedding is never computed as `u⊗v` and is
  never a reshape of the matrix arm's `M` — it is an independently
  parameterized embedding of a chosen width. `d_model` is solved for
  total-parameter equality (§4), not fixed at `d²` (the historical choice
  that produced Run 22's asymmetry) or forced to a particular relation to
  `2d`.

### 2.1b Embedding — matrix vs flatten (the NEW arm, audit M3)

The `flatten` arm deliberately does the opposite of §2.1's separation: it
shares the **exact same** `embed_u, embed_v, pos_u, pos_v` construction
and outer product as the `matrix` arm (same `mat_dim`, same init, §4.4)
— then flattens the resulting `(d,d)` matrix `M` to a `d²`-vector and
resizes it (`nn.Linear(d², d_model)`) into the flat arm's own dense
`VectorThinkingBlock` backbone family. This is Run 18's exact historical
recipe ("same outer-product embedding, then FLATTEN to a 256-dim vector,
standard transformer") — the difference from Run 18 is that `d_model`
(the backbone's OWN operating width, distinct from the embedding's fixed
`d²`) is solved for total-param equality (§4.2) rather than left at the
full `d²` width, which is what made Run 18 10x asymmetric. Because
`matrix` and `flatten` share their embedding bit-for-bit, any T=1 BPB gap
between them isolates the **operations**, not the embedding — this is
what makes STRENGTHEN-01 (§1a) a genuine embedding-mechanism-controlled
test, distinct from STRENGTHEN-04 (§1b), which varies embedding and
operations together.

### 2.2 Backbone operations

- **Matrix arm:** every projection (`RowThenColProjection`) computes
  `silu(A @ M) @ B` with `A, B ∈ ℝ^{d×d}`. The induced linear map on
  `vec(M)` is `(Bᵀ ⊗ A)` — a **Kronecker-restricted** subspace of the
  full `d²×d²` linear group. Not every `d×d → d×d` linear map is
  expressible this way (this is exactly finding 04's own framing).
- **Flat and flatten arms:** every layer uses `nn.MultiheadAttention` and
  `nn.Linear` on a flat vector — **full, unrestricted** linear maps
  (in_proj, out_proj, both FFN layers), the identical `VectorThinkingBlock`
  class for both arms. Both arms' linear maps are strictly more general
  than the Kronecker family the matrix arm is confined to, at any width,
  and in particular strictly more general than "the matrix arm reshaped"
  would be even at width `d²` (a full `d²×d²` Linear can express
  `(Bᵀ⊗A)` as a special case, but neither the flat nor flatten arm trains
  a weight tied to, initialized from, or reachable by any operation on
  the matrix arm's parameters — flatten's `resize_in` Linear is a freshly
  initialized, independently trained weight, not a projector derived from
  `A`/`B`).

**Net effect:** if `matrix` still wins at equal total parameters despite
`flat`/`flatten` having access to a strictly larger per-parameter
hypothesis class at every layer, that is evidence the win is about
*structure*, not about which family happens to be more constrained. If
`flat`/`flatten` wins or ties, that is evidence for the reverse — either
outcome is informative for each of the two independently-decided claims.

### 2.3 What this design does NOT re-test

Finding 04's `d⁴` vs `2d²` per-layer parameter-count claim is a **counting
fact** about two specific operator families at a **fixed `d`** — it is
true by construction (verified by instantiating both `nn.Module`s, see
`matrix-thinking/src/matrix_output_heads.py`'s own benchmark harness) and
is not something a training run confirms or falsifies. What this design
*does* inherit from finding 04 is the practical consequence: because we
solve the backbone's operating width for parameter equality rather than
fixing it at `d²` (reshape parity), the flat/flatten arms' backbone FLOP
cost lands close to the matrix arm's own (§4.2), not at the historical
8×-128× blowup — which is precisely why every registered cell fits
comfortably under the 2 GPU-h ceiling (§5).

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
  `eot_separated == True` — all three arms already consume a GPT-2-vocab
  `Embedding(vocab_size, ·)` table unmodified, so **no architecture
  change is needed** to switch off byte-level vocab. `wikitext-mix-ext`
  (not `openr1-mix-ext`) is used for **all** cells per `CLAUDE.md`'s "use
  the same dataset for all experiments in a comparison" rule.
- **Headline metric — token-BPB:** `cross_entropy_nats / ln(2)`, i.e.
  bits per GPT-2 token (NOT bits per raw byte — no byte-length claim is
  made anywhere in this design). Also reported: val loss in nats, and
  perplexity, for continuity with the historical Round-1/Round-2 numbers.
- **Both T=1 and T=8** are reported for all three arms at every checkpoint
  and at the final step, closing Run 22's actual failure mode (its flat
  arm never reached a finished T=1/T=8 pair — it died mid-run). The runner
  (`embed_ablation_rd.py`) writes `final_evals: {T1: {...}, T8: {...}}`
  unconditionally at every `--eval-interval` and forces one last eval at
  the terminal step whether that is `COMPLETED` or `CEILING_STOP` (the
  minor fix in this REV — the first pass only forced this on the
  ceiling-stop path retroactively; it is now unconditional on both exits),
  so a JSON with `complete: false` still carries a real T1/T8 pair from
  the moment it stopped — `harvest()` only accepts `complete: true` cells
  into either decision (§7), but a `CEILING_STOP` cell's curve is still
  inspectable for diagnosis.
- **Eval windows are now seed/arm/T-invariant (audit M1).** `evaluate()`
  reseeds a fresh `torch.Generator` from `corpus_fixed_seed(corpus_name)`
  — `zlib.crc32` of the corpus name, ported from `lm_pretrain_rd.py`'s own
  `corpus_fixed_seed` / AUDIT FIX-1 pattern, whose own comment reads
  "NEVER from the training seed" — at the **start of every call**. Before
  this fix, the validation generator was seeded from `args.seed + 10000`,
  so different training seeds (and the T=1 vs T=8 legs, called
  back-to-back) would score **different** random windows, adding sampling
  noise that could look like a model difference. `--eval-batches` is also
  raised from 20 to 50 (more windows per eval, tighter variance).

## 4. Architecture and parameter matching

All three arms reuse the module definitions from
`experiment-runs/8xh100-session1/round1_matrix_script.py` /
`round1_vector_script.py` / `round2_matrix_script.py` (copied
module-for-module into `matrix-thinking/src/embed_ablation_rd.py`,
single-GPU, DDP stripped since every cell fits on one H100), plus the new
`FlattenThinker` class (§2.1b) built from the same pieces — this ablation
changes the comparison's rigor, not the architecture the claims were made
about. `T` (iterative-refinement count) never changes parameter count in
any arm — the `n_layers`-deep stack is weight-shared and applied `T`
times (`_one_iteration`); only `n_layers` and the operating width set the
parameter budget. This is why the "depth vs params" tension the task
brief anticipated (Run 22's "130× per layer" problem) turns out to be
solvable at equal `n_layers` in this regime (§4.3) — the knob that trades
against total params is width, not iteration count.

### 4.1 Registered sizes

| Size | `mat_dim` (matrix/flatten) | `n_layers` | `n_heads` (matrix/flat-P) | max_len |
|---|---|---|---|---|
| S | 16 | 6 | 4 | 512 |
| M | 24 | 8 | 4 | 512 |

`n_iterations` (T) = 8 for training and for the "matrix model's iterative
T" eval leg, at both sizes (matches Round 2's T=8 convention). Eval always
reports T∈{1, 8}. The `flatten` arm's OWN `n_heads` is solved independently
(§4.3) — it is not required to equal the registered `n_heads` column above.

### 4.2 Parameter counts (verified by real `nn.Module` instantiation via
`embed_ablation_rd.py --selftest`, output in §8)

Vocab `V = 50,257` (GPT-2), `max_len = 512`.

| Size | Arm | Config | total params | ratio vs matrix | diff | embed+head share |
|---|---|---|---|---|---|---|
| S | matrix | mat_dim=16, L=6 | **2,466,562** | 1.0 | — | 98.5% |
| S | flat-P (primary, params-matched) | d_model=24, n_heads=4, head_dim=6, L=6 | **2,468,016** | 1.0006 | **0.059%** | 98.2% |
| S | flatten-P (primary, params-matched, NEW) | mat_dim=16→resize→d_model=16, n_heads=4, head_dim=4, L=6 | **2,452,544** | 0.9943 | **0.568%** | 99.0% |
| S | flat-D (disclosed control, unmatched) | d_model=32 (=2·mat_dim), L=6 | **3,309,120** | 1.3416 | 34.159% | 97.7% |
| M | matrix | mat_dim=24, L=8 | **3,755,808** | 1.0 | — | 97.0% |
| M | flat-P (primary, params-matched) | d_model=36, n_heads=4, head_dim=9, L=8 | **3,765,168** | 1.0025 | **0.249%** | 96.6% |
| M | flatten-P (primary, params-matched, NEW) | mat_dim=24→resize→d_model=25, n_heads=1, head_dim=25, L=8 | **3,770,412** | 1.0039 | **0.389%** | 98.0% |
| M | flat-D (disclosed control, unmatched) | d_model=48 (=2·mat_dim), L=8 | **5,075,520** | 1.3514 | 35.138% | 95.5% |

All four `-P` (params-matched) rows are within the ±1% gate; both `-D`
rows are outside it by construction (pre-registered as unmatched, §4.3).

**Disclosure (audit round-2, minor):** `embed+head` (the embedding
tables plus the output head, excluding the backbone/resize) is
**95.5%-99.0% of total params in every arm/size**, computed directly
from `param_breakdown()` — the "primary" `-P` arms (the ones the two
decisions are actually scored on) cluster tighter, at **96.6%-99.0%**;
only the disclosed, non-gating `flat-D` control dips to 95.5% (size M),
since its `d_model=2·mat_dim` deliberately blows up the backbone/head
relative to the params-matched arms. This is the same structural fact
§2.3/§4.2's original "key upshot" already leaned on (vocab dominates,
which is WHY total-param matching at equal depth is achievable at all in
this regime) — stated here as an explicit number rather than left
implicit, per the audit's request. It also means the backbone/resize
that actually differs between arms (RowThenCol vs full Linear vs
Linear-then-full-Linear) is a **small** fraction of what "total params"
measures — a real limitation to keep in mind when interpreting either
decision: a large T=1 BPB gap driven by embed/head differences would
still count as a win/loss under §1's rule, even though the operations
comparison (§2.2) is the part motivating the ablation. Reported, not
adjusted for — the pre-registered rule (§7) reads total-params-matched
T=1 BPB, full stop, and does not try to isolate "backbone-only" credit.

**LR is shared and untuned (audit round-2, minor disclosure):** every
cell uses `--lr 3e-4` (the script's own default, unchanged across all 3
arms, both sizes, and both `--match` conditions) — this value was never
swept or tuned for any of the three model families; it is simply the
value `round1_vector_script.py`/`round2_matrix_script.py` used
historically. A result that hinges on whether 3e-4 happens to suit one
architecture better than another is a real, disclosed confound this
design does not control for — pre-registering it here means a `DROP`
verdict driven by a bad LR for one arm is a known risk to flag in the
harvest write-up, not a surprise discovered after the fact.

**Why `flatten`'s `n_heads` can differ from `flat`'s (audit-discovered):**
`nn.MultiheadAttention`'s own parameter count is **head-count-invariant**
(`in_proj_weight` is `(3·embed_dim, embed_dim)` regardless of
`num_heads` — `num_heads` only changes how attention is computed, not how
many parameters exist), so `solve_matched_width()` (audit round-2 MJ-3)
tries the **registered `n_heads` first**, then `2`, then `1`, then `8`,
for the `flatten` arm's own independent search (the `flat` arm's own
solve is untouched — single-candidate, already verified at 0.059%/
0.249%). At size S this lands on the SAME `d_model=16` (hence the
IDENTICAL 2,452,544 params) as the first pass's `n_heads=8` search found
— because 16 is a multiple of both 4 and 8, trying the registered
`n_heads=4` first finds it too, just labeled with a smaller, more
conventional head count (`head_dim=4` instead of `head_dim=2`). At size M,
no multiple of 4 clears the ±1% tol (best achievable is 1.09% at
`d_model=24`) or of 2 either — the search falls through to `n_heads=1`
(`d_model=25`, `head_dim=25`, plain single-head attention, 0.389% — this
outcome is UNCHANGED from the first pass and remains a disclosed,
forced-by-the-integer-search oddity of size M specifically, not a choice).

### 4.3 The three pre-registered arms, and which are the verdict carriers

Per the task brief's requirement: *"pre-register BOTH a depth-matched/
params-unmatched arm and a params-matched/depth-unmatched arm."* Since
depth-matching-at-equal-params turns out to be achievable here (§4.2),
the depth-matched arm is realized as depth-matched-but-params-**unmatched**
(flat-D) rather than params-matched-but-depth-unmatched:

- **matrix — reference arm for both decisions.**
- **flat-P (params-matched, VERDICT CARRIER for STRENGTHEN-04).** Width
  solved via `solve_matched_d_model()` to land within ±1% of the matrix
  arm's total params, gated by `check_param_match()` — a >1% mismatch
  **raises `RuntimeError`** before training starts (verified in §8's
  negative test).
- **flatten-P (params-matched, VERDICT CARRIER for STRENGTHEN-01, NEW
  audit M3).** Matrix arm's own embedding, flattened + resized (§2.1b),
  width solved via `solve_matched_d_model_flatten()`. Only `--match P` is
  registered — no unmatched control is defined for this arm (the task
  brief's depth/params tension was already addressed by flat-D; adding a
  second unmatched control would not test anything new).
- **flat-D (depth-matched, `n_layers` equal), params UNMATCHED
  (secondary/disclosed control, NOT a verdict carrier for either
  decision).** `d_model = 2·mat_dim` fixed (the embedding-only "natural"
  width — same free-parameter count per token as the matrix embedding's
  `(u,v)` pair, but the backbone and head end up larger). Ratio disclosed
  (~1.34-1.35×, flat favored), never gated. Reported so a reader can see
  how much the ±1%-gated matching (vs. the old un-gated reshape-parity
  choice) actually moves the T=1 gap.

`harvest()` only ever reads `match="P"` groups when scoring
STRENGTHEN-01/04 (§7); `flat-D` is reported separately, under a
`flat-D_disclosed_not_gating` key, and is excluded from both decisions by
construction (its `match` field is `"D"`, never matched by the decision
functions' `groups.get((arm, "P", size), [])` lookups).

### 4.4 Initialization (audit M2, RULING — applied explicitly, not
inherited from the historical scripts)

`CLAUDE.md`: *"Outer-product embedding init: u,v std must be sqrt(target_
std), not target_std. Products have std=σ²."* `TARGET_STD = 0.02`
throughout (matching the CLAUDE.md example value):

- **matrix, flatten (share the same embedding construction):**
  `embed_u.weight, embed_v.weight, pos_u.weight, pos_v.weight` init at
  `std = sqrt(TARGET_STD) ≈ 0.1414`, so the **constructed** `M = u⊗v`
  entries have `std ≈ TARGET_STD = 0.02` (verified empirically in §8:
  `embed_u std=0.1412` vs the theoretical `0.1414`).
- **flat:** `embed.weight, pos.weight` init at `std = TARGET_STD = 0.02`
  directly — no factoring to compensate for, so the direct std IS the
  target (verified in §8: `flat embed std=0.0200` exactly).
- **flatten's `resize_in` (audit round-2 MJ-4, NEW):** `resize_in`
  (`nn.Linear(mat_dim², d_model)`) is the one weight with no analog in
  either other arm — a freshly-initialized Linear has no reason to land
  its OUTPUT (the tensor the backbone actually sees) anywhere near
  `TARGET_STD`. Explicit init: `bias` zeroed, `weight ~ N(0, (1/mat_dim)²)`.
  Derivation: `h_k = Σ_i W_ki · M_flat_i` over `mat_dim²` terms; with
  `W ~ N(0, (1/mat_dim)²)` and `M_flat_i` at `std ≈ TARGET_STD` (the
  matrix arm's own construction, approx-iid across the flattened entries),
  `Var(h_k) ≈ mat_dim² · (1/mat_dim)² · TARGET_STD² = TARGET_STD²`, i.e.
  `std(h_k) ≈ TARGET_STD` — so the backbone-input scale is matched across
  ALL THREE arms, not just the two that share a table. Verified empirically
  in §8's new backbone-input-std print (target `≈0.02`): size S measured
  `matrix=0.0193 flat-P=0.0208 flatten-P=0.0205`; size M measured
  `matrix=0.0207 flat-P=0.0198 flatten-P=0.0205` — all three arms, both
  sizes, land within ~15% of `TARGET_STD`, closing what would otherwise
  have been an unexamined (and, for `flatten`, previously literally
  unset) init asymmetry at the one place all three architectures'
  "first thing the backbone sees" tensors can be compared apples-to-apples.
- **Positional-term scaling parity:** the matrix arm has always scaled
  its positional outer-product contribution by `*0.1` before adding it to
  the token embedding (`M = M + pos_outer_product * 0.1`). The historical
  `round1_vector_script.py`'s flat arm added its positional term with
  **no** scaling at all — an accidental asymmetry this design fixes: both
  `VectorThinker` and `FlattenThinker` now also scale their positional
  term by `*0.1`, so the only asymmetry left between arms is the
  operation family, not an accidental init/scale mismatch (this is a
  direct instance of §2's "close the reshape-equivalence loophole"
  discipline applied to init, not just to the forward pass).
- **Historical gap disclosed:** neither `round1_matrix_script.py`,
  `round1_vector_script.py`, nor `round2_matrix_script.py` set ANY
  explicit init for these tables (confirmed by reading all three files —
  no `nn.init.*` call touches `embed_u`/`embed_v`/`pos_u`/`pos_v`/`embed`/
  `pos` anywhere in them). Every historical Run 10/11/12/13/18/22 number
  cited in §0 was produced with PyTorch's default `nn.Embedding` init
  (`N(0,1)`), not the CLAUDE.md rule. This design does not merely inherit
  that gap — it is the first script in this line to apply the rule.
  Other parameters (RowThenCol's `A,B`, the multiplicative layer's gates,
  etc.) keep their historical inits unchanged — M2's ruling was scoped to
  the outer-product embedding specifically, not a general re-init pass.

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

**Empirical anchor instead (corrected in this REV):** `exp_d16_v2_
SUMMARY.txt`'s prose ("mat_dim=16, 8 layers") is **stale/wrong** —
verified by reading `exp_d16_v2_script.py`'s actual `CONFIG` dict
(`"n_thinking_layers": 12`, `"max_len": 2048`) and the `MatrixThinker`
class's own default (`n_layers=12`, `max_len=2048`), which is what the
run actually instantiated. The correct anchor: `mat_dim=16, n_layers=12,
max_len=2048, T=8, batch=96/GPU, seq_len=512, 3000 steps, GPT-2 vocab,
2,552,788 params` took **82.0 min wall time**. Since DDP replicates the
model per GPU with no per-step cross-GPU dependency beyond the gradient
all-reduce, this 82 min is also the single-GPU wall time for that exact
per-GPU workload — i.e. **1.367 GPU-h** for that config (this numeric
anchor value is unchanged from the first pass; only its layer-count/
max_len description was wrong and is now corrected). Every cell's
`gpu_h_estimate` in §6/`embed_ablation_specs/*.json` is this anchor
scaled linearly by `(batch/96) · (steps/3000) · (params/2,552,788)` — a
linear scaling in tokens-processed and total-params, which is the right
first-order model given all three arms are embed/head-GEMM-dominated
(§4.2) rather than backbone-FLOP-dominated.

**GPU memory:** activations are `(B, L, d, d)` tensors (matrix/flatten
arms, pre-resize) or `(B, L, d_model)` (flat/flatten arms, post-resize) ×
`n_layers` × the gradient-checkpointing recompute factor
(`torch.utils.checkpoint`, wired into every `_one_iteration` method). At
`B=64, L=512, d≤36`, the largest activation tensor is on the order of
`64·512·36·36·4 bytes ≈ 212 MB` per matrix-valued activation buffer, and
checkpointing means only per-layer boundaries are retained — total GPU
memory footprint is in the **low single-digit GB**, nowhere near the
80GB H100 ceiling that `CLAUDE.md`'s other hard rules (batch=96 at
mat_dim=32 being the max, the 50K-vocab logits tensor being the VRAM
bottleneck) were written about — those rules were calibrated for models
an order of magnitude larger in vocab-projection batch×seq product than
what a single-GPU, batch=64 cell here produces. **GPU memory is not a
binding constraint for this design**; the binding constraint is
wall-clock GPU-h, addressed by the `--ceiling-gpuh` hard stop (§6).

**Host RAM (disclosed, audit minor-fix item — not previously stated):**
`load_corpus_tokens()` loads the ENTIRE `train.pt` + `val.pt` token
tensors into host RAM via `torch.load(map_location="cpu")` before any
`.to(device)` transfer, per process. This was NOT measured directly (no
box access from this sandbox — §10) and is **estimated** at **~3.3 GB
host RAM per process**: int64 GPT-2 tokens are 8 bytes each, and an
extended-mix corpus on the order of ~400M combined train+val tokens
(consistent with `SCALE_TRANSFER_DESIGN.md`'s own sizing note that these
corpora support "rung 2's 1.5B-token/run target" without excessive
epoching) gives `~400M × 8 bytes ≈ 3.2-3.3 GB`. This is per-process, so N
concurrently-running cells on the box's 8 GPUs could sum to `~3.3N GB`
host RAM (e.g. ~26 GB at full 8-way concurrency) — small relative to a
typical server's host RAM but not previously disclosed anywhere in this
design. Confirm against the first probe cell's actual process RSS before
assuming this figure; do not treat it as measured.

## 6. GPU-h ledger (≤2 GPU-h/cell, ≤30 GPU-h total — both satisfied)

30 queue specs total (grew from 22 in the first pass — audit M3 added a
third arm): **phase A** = 6 rate/admission probes (500 steps each,
`embed_ablation_specs/phase_A_probes/0640-0645`, one per {matrix, flat,
flatten} × {S, M}) + **phase B** = 24 seeded cells
(`embed_ablation_specs/phase_B_seeded/0646-0669`, 4 arm-configs {matrix,
flat-P, flat-D, flatten-P} × 2 sizes × 3 seeds).

| Cell group | n | GPU-h/cell (formula) | Subtotal |
|---|---|---|---|
| 6 phase-A probes (matrix/flat/flatten × S/M) | 6 | 0.146–0.224 | 1.112 |
| matrix_S_s{0,1,2} | 3 | 0.587 | 1.761 |
| flatp_S_s{0,1,2} | 3 | 0.587 | 1.762 |
| flatd_S_s{0,1,2} | 3 | 0.788 | 2.363 |
| flatten_S_s{0,1,2} | 3 | 0.584 | 1.751 |
| matrix_M_s{0,1,2} | 3 | 0.894 | 2.682 |
| flatp_M_s{0,1,2} | 3 | 0.896 | 2.689 |
| flatd_M_s{0,1,2} | 3 | 1.208 | 3.624 |
| flatten_M_s{0,1,2} | 3 | 0.898 | 2.693 |
| **Total** | **30** | max single cell **1.208** | **20.434 GPU-h** |

Max single-cell estimate (1.208, flatd_M) is well under the 2 GPU-h cap
(1.66× headroom); the 20.434 GPU-h total leaves ~1.47× headroom under the
30 GPU-h budget. Every `cmd` also carries `--ceiling-gpuh 2.0` (0.5 for
probes) as a hard stop, matching `005`'s own `--ceiling-gpuh` convention
— a cell that runs long writes `status: "CEILING_STOP"` (with a forced
final eval, per the minor fix) and exits cleanly rather than overrunning.

**Sequencing / gating (audit F2, FATAL fix — restated here verbatim from
`embed_ablation_specs/phase_A_probes/README.md` and `phase_B_seeded/
README.md`):** run the 6 phase-A probes first. Once all 6 land, run

```
embed_ablation_rd.py --check-admission \
    --probe-results-dir /home/nvidia/embed_ablation/results/probes \
    --intended-steps 2000 --intended-batch 64
```

**This must exit 0 before ANY `phase_B_seeded/` spec is queued.** It
checks: (0, audit round-2 MJ-2) the SET of `(arm,size)` found among the
probe records' own fields equals `{matrix,flat,flatten}×{S,M}` (6 combos)
exactly — a crashed/missing probe fails this independent of every other
check (§8c scenario 4); then per probe present: (a, tightened by audit
round-2 MJ-1) `len(t1_seq) >= 3` (FAILS LOUDLY on a 1-2-point curve, §8c
scenario 5 — not silently judged on whatever few points exist), and the
last three T=1 evals are monotone non-increasing (a coarse "is this
actually learning" check); (a2, audit round-2 minor) the LAST T1 value is
strictly less than the FIRST (net improvement over the whole probe run);
(b) the probe's measured wall time, linearly extrapolated to
`--intended-steps 2000 --intended-batch 64`, does not exceed 2.0 GPU-h/
cell. If it exits nonzero, STOP — re-derive `--steps` (updating
`_gen_specs.py`'s shared `STEPS_B` constant, audit round-2 MJ-5, so every
`phase_B_seeded/` spec's `cmd` AND its validity_check's exact-match value
move together) and re-run this generator's `gpuh()` cost model with the
measured rate before staging or queuing anything in that directory. This
document does not authorize skipping that check, and neither does either
phase's own `README.md`.

## 7. Decision rules (pre-registered, no third outcome, TWO independent
decisions per audit M3)

For **each** of the two claims (§1a STRENGTHEN-01, §1b STRENGTHEN-04)
independently, and for each registered size independently: sort each
arm's 3 seeds' T=1 token-BPB. Compute `seed_spread = max(spread_matrix,
spread_other)` (max − min within each arm, whichever is larger). Pair the
seeds (sorted order) and count a "win" per pair where `other_BPB −
matrix_BPB > seed_spread`. `size_pass = (wins ≥ 2) AND (n_pairs == 3)`.

- **STRENGTHEN** iff `size_pass` is true at **both** S and M, for that
  claim.
- **DROP** otherwise (any size failing, including ties or reversals), for
  that claim.
- **PENDING** if either size is not yet fully scored (a group has 0
  qualifying records) — never a silent partial verdict.

`STRENGTHEN-01` and `STRENGTHEN-04` are scored **completely
independently** — one can be `STRENGTHEN` while the other is `DROP` or
`PENDING`; neither's outcome constrains the other. `matrix-thinking/src/
embed_ablation_rd.py`'s `_harvest_records()` implements exactly this and
only reports `STRENGTHEN`/`DROP`/`PENDING` per claim, never a third
label, and never averages the two claims together. `flat-D` (§4.3) is
reported under `flat-D_disclosed_not_gating` for context but is excluded
from both decisions by construction (its `match` field is `"D"`, and
both `score()` lookups only read `match="P"` groups).

**F1's exact-3 invariant (FATAL fix, restated, tightened by round-2
MJ-5/MJ-6/minor):** before either decision is scored, `_harvest_records()`
filters to `status=="COMPLETED" AND complete==True AND
steps_completed==steps_target AND non-probe` (the exact-equality
threshold is round-2's MJ-5/MJ-6 fix — the first pass used a looser
`steps_target>=2000`), asserts a single shared `steps_target` and shared
`corpus`/`batch_size`/`seq_len` across ALL valid records globally, then
groups by `(arm, match, size)` and asserts every group present has
**exactly 3** records with `seeds == {0,1,2}` exactly — 0 records is fine
(not run yet, contributes `PENDING`), but 1-2 (a lost/missing seed), 4+
(a duplicate or a leaked partial/probe record), or a seed set other than
`{0,1,2}` **raises immediately**, before any BPB numbers are even read.
`--harvest-selftest` (§8b) proves this holds even when probe-labeled and
`CEILING_STOP` records are physically present in the same directory as
the 15 clean records it constructs.

## 8. Smoke test (CPU, run before any box deployment)

`embed_ablation_rd.py --selftest` does, on CPU with a scratch CPU-only
torch venv (this sandbox has no GPU and no pre-installed torch; a fresh
`venv` + `pip install torch --index-url https://download.pytorch.org/
whl/cpu` was used to actually execute this, not merely inspected by
reading), IN ORDER:

0. **Init check (audit M2):** instantiates a tiny `MatrixThinker` and
   `VectorThinker`, empirically measures `embed_u.weight.std()` /
   `embed.weight.std()`, and checks them against `sqrt(TARGET_STD)` /
   `TARGET_STD` respectively (within 15%, since a small random sample
   has sampling noise).
1. **Forward/backward/grad check, ALL THREE arms**, at two TOY configs
   (`tiny-S`: mat_dim=8, `tiny-M`: mat_dim=12, vocab=97): asserts output
   shape is `(B, L, vocab)`; loss is finite; **every** trainable
   parameter receives a gradient (`n_with_grad == n_params`, not just
   "most"); gradients are finite; and >50% of parameters get a *nonzero*
   gradient (a coarse dead-parameter check). `flatten` is new in this
   REV.
2. **Negative test:** constructs a deliberately mismatched flat model
   (`d_model=4`, ~89% off) against a real matrix model's param count, and
   confirms `check_param_match()` reports `FAIL` (not silently `PASS`) and
   that the `build_arm(match="P")`-style code path **raises**
   `RuntimeError` rather than proceeding.
3. **Registered sizes (S, M) at real GPT-2 vocab scale (50,257)** — the
   actual configuration the box will run — for **all three arms**: `flat-P`
   and `flatten-P` both solve within the ±1% gate (§4.2's numbers), and
   `flat-D` at both sizes is correctly flagged as outside the gate
   (informational, pre-registered as unmatched, not a script failure).
4. **`harvest_selftest()` (audit F1, §8b below).**

### 8a. Verbatim `--selftest` output, ROUND-2 (`DRY_RUN_BYPASS=1
/tmp/embed_ablation_venv/bin/python3 matrix-thinking/src/
embed_ablation_rd.py --selftest`, CPU, `torch==2.14.0`, re-run after
MJ-1 through MJ-6/minor)

```
==========================================================================
EMBED ABLATION SELFTEST (CPU, tiny configs)
==========================================================================

--- init std check (M2 ruling) ---
  matrix embed_u std=0.1412 target=sqrt(0.02)=0.1414
  flat embed std=0.0200 target=TARGET_STD=0.0200

--- size tiny-S: {'mat_dim': 8, 'n_layers': 2, 'n_heads': 2} ---
  [matrix/tiny-S] forward OK shape=(3, 11, 97) loss=4.6740 grads: 66/66 present, 66/66 nonzero (100%)
  flat-P/tiny-S param match: matrix=5,806 other=6,252 ratio=1.0768 diff=7.682% tol=1.0% -> FAIL  [informational at toy vocab=97; real gate is the 'registered sizes at real GPT-2 vocab' section below]
  [flat-P/tiny-S (d_model=12)] forward OK shape=(3, 11, 97) loss=4.7367 grads: 29/29 present, 29/29 nonzero (100%)
  flatten-P/tiny-S param match: matrix=5,806 other=4,784 ratio=0.8240 diff=17.602% tol=1.0% -> FAIL  [informational at toy vocab; real gate below]  n_heads=8 head_dim=1
  [flatten-P/tiny-S (d_model=8)] forward OK shape=(3, 11, 97) loss=4.8023 grads: 33/33 present, 33/33 nonzero (100%)

--- size tiny-M: {'mat_dim': 12, 'n_layers': 2, 'n_heads': 4} ---
  [matrix/tiny-M] forward OK shape=(3, 11, 97) loss=4.6294 grads: 66/66 present, 66/66 nonzero (100%)
  flat-P/tiny-M param match: matrix=11,154 other=9,872 ratio=0.8851 diff=11.494% tol=1.0% -> FAIL  [informational at toy vocab=97; real gate is the 'registered sizes at real GPT-2 vocab' section below]
  [flat-P/tiny-M (d_model=16)] forward OK shape=(3, 11, 97) loss=4.6733 grads: 29/29 present, 29/29 nonzero (100%)
  flatten-P/tiny-M param match: matrix=11,154 other=11,076 ratio=0.9930 diff=0.699% tol=1.0% -> PASS  [informational at toy vocab; real gate below]  n_heads=2 head_dim=7
  [flatten-P/tiny-M (d_model=14)] forward OK shape=(3, 11, 97) loss=4.8138 grads: 33/33 present, 33/33 nonzero (100%)

--- negative test: forced param mismatch must raise ---
  NEGATIVE param match: matrix=87,890 other=9,728 ratio=0.1107 diff=88.932% tol=1.0% -> FAIL
  negative test correctly reports FAIL for the mismatched pair (as expected)
  build_arm-style gate correctly RAISED: simulated build_arm(match='P') gate: NEGATIVE-build_arm-path param match: matrix=87,890 other=9,728 ratio=0.1107 diff=88.932% tol=1.0% -> FAIL

--- registered sizes at real GPT-2 vocab (50257), match=P ---
  size S flat-P param match: matrix=2,466,562 other=2,468,016 ratio=1.0006 diff=0.059% tol=1.0% -> PASS  (d_model=24, n_heads=4, head_dim=6)
  size S flatten-P param match: matrix=2,466,562 other=2,452,544 ratio=0.9943 diff=0.568% tol=1.0% -> PASS  (d_model=16, n_heads=4, head_dim=4)
  size S (match=D, expected UNMATCHED) param match: matrix=2,466,562 other=3,309,120 ratio=1.3416 diff=34.159% tol=1.0% -> FAIL  (d_model=32) -- D is pre-registered as unmatched, this is informational
  size S backbone-input std (target~=0.02): matrix=0.0193 flat-P=0.0208 flatten-P=0.0205
  size M flat-P param match: matrix=3,755,808 other=3,765,168 ratio=1.0025 diff=0.249% tol=1.0% -> PASS  (d_model=36, n_heads=4, head_dim=9)
  size M flatten-P param match: matrix=3,755,808 other=3,770,412 ratio=1.0039 diff=0.389% tol=1.0% -> PASS  (d_model=25, n_heads=1, head_dim=25)
  size M (match=D, expected UNMATCHED) param match: matrix=3,755,808 other=5,075,520 ratio=1.3514 diff=35.138% tol=1.0% -> FAIL  (d_model=48) -- D is pre-registered as unmatched, this is informational
  size M backbone-input std (target~=0.02): matrix=0.0207 flat-P=0.0198 flatten-P=0.0205

--- harvest-selftest: synthetic 15-clean + 2-probe + 1-CEILING_STOP dir ---
  dir A: 15 files (expect 15)
  dir B: 18 files (expect 18 = 15 + 2 probes + 1 ceiling-stop)
  [... full per-decision JSON for both dirs, omitted here for length, see §8b ...]
  STRENGTHEN-04 decision: A=STRENGTHEN B=STRENGTHEN
  STRENGTHEN-01 decision: A=PENDING B=PENDING (expected PENDING: flatten/M is missing from both dirs by construction)
  harvest-selftest: PASS -- probes + the CEILING_STOP cell provably do not change the verdict

==========================================================================
SELFTEST: ALL CHECKS PASSED
==========================================================================
```

**Round-2 headline numbers (per the coordinator's explicit report
request):** `flatten-S` (size S's flatten-P arm) now solves at
**`n_heads=4, head_dim=4, d_model=16`** — the SAME `d_model=16` and the
IDENTICAL `2,452,544` total params the first pass's `n_heads=8` search
found (verified: `other=2,452,544` in both the round-1 and round-2
transcripts), just reached via the registered-`n_heads`-first candidate
order (MJ-3) rather than starting at 8. The new backbone-input-std print
(MJ-4) confirms the `resize_in` init derivation empirically: all three
arms, both sizes, land within `0.0193`-`0.0208` against a `TARGET_STD`
of `0.02` (3.5%-9.5% off, well inside the informal 15% band this print
is meant to sanity-check, not gate).

**Flag for ruling, not assumption:** the `tiny-S`/`tiny-M` toy-vocab
param-match sub-checks report `FAIL` (or, for `flatten-P/tiny-M`, a
lucky `PASS`) at `vocab=97` — this is **expected and disclosed in the
script's own print output**, not a bug: at that scale the integer width
search space is too coarse for ±1% to be reliably achievable. This
sub-check is explicitly excluded from the pass/fail gate; the real
precision claim is the "registered sizes at real GPT-2 vocab" block,
which passes for `flat-P` and `flatten-P` at both sizes (0.059%-0.568%).

### 8b. `--harvest-selftest` (audit F1, standalone flag AND run inside
`--selftest`)

Builds two temp directories: **A** = 15 clean, `complete=True` records
across exactly 5 `(arm,match,size)` groups × 3 seeds (`matrix/P/S`,
`matrix/P/M`, `flat/P/S`, `flat/P/M`, `flatten/P/S` — `flatten/P/M`
**deliberately missing**, to also exercise the `PENDING` path for
STRENGTHEN-01 at size M); **B** = the same 15 records **plus** 3 extras:
a record with `role="probe"`, a record with no `role` field but `"probe"`
in its filename (exercising **both** detection paths F1 names), and a
`complete=False` (`CEILING_STOP`) record for the missing `flatten/P/M`
group. `harvest()` is run on both directories and the two summaries are
compared field-by-field: `STRENGTHEN-01`, `STRENGTHEN-04`, and
`flat-D_disclosed_not_gating` must be **byte-identical**, and
`n_records_valid_after_filter` must be `15` in both (only `n_records_
loaded` legitimately differs, `15` vs `18`). Verbatim result (from the
same `--selftest` run as §8a): **PASS** — `STRENGTHEN-04` computes
`STRENGTHEN` identically in both dirs (matrix consistently beats flat-P
by more than the seed spread, by construction of the synthetic BPB
values); `STRENGTHEN-01` reports `PENDING` identically in both dirs
(flatten/M is missing from both, by construction — not affected by
whether the 3 extras are present).

Run standalone: `embed_ablation_rd.py --harvest-selftest` (exit code 0
confirmed independently of the full `--selftest` run).

### 8c. `--check-admission` (audit M4, extended by round-2 MJ-1/MJ-2/
minor) — verified against 5 synthetic scenarios, not just read

**Round 1 (3 scenarios, unchanged by round 2):** (1) one monotone-
improving probe + one non-monotone (diverging) probe → **exits 1**,
correctly flags the diverging probe by name and its exact `last3` values,
while still reporting the healthy probe's own checks separately; (2) a
probe with a realistically slow measured rate (90 min for 500 steps, vs
the ~9 min the formula anchor predicts) → **exits 1** on check (b),
reporting the extrapolated `6.000 GPU-h` against the `2.0` ceiling; (3) a
normal monotone + on-rate probe → **exits 0**.

**Round 2 (2 new scenarios, per the coordinator's explicit report
request) — both actually run against synthetic JSON, verbatim below:**

**Scenario 4 — 5 of 6 required probes present (`matrix/S` missing,
simulating a crash that never wrote a result file):**

```
  ADMISSION SET FAIL: missing probe(s) for (arm,size) in [('matrix', 'S')] -- a probe crashed, never wrote its result, or wrote one with a missing/wrong arm/size field. Expected all 6 of [('flat', 'M'), ('flat', 'S'), ('flatten', 'M'), ('flatten', 'S'), ('matrix', 'M'), ('matrix', 'S')], found [('flat', 'M'), ('flat', 'S'), ('flatten', 'M'), ('flatten', 'S'), ('matrix', 'M')].
  [... each of the 5 present probes reports OK (a)/(a2)/(b) individually ...]
ADMISSION CHECK: FAIL -- STOP, do NOT stage phase B; re-derive --steps or investigate the failing arm/missing probe first
exit code: 1
```

The MJ-2 admission-SET check fails **independent of** every present
probe individually passing (a)/(a2)/(b) — this is the case F1/MJ-2 was
written for: "5 of 6 looked fine" is not admission, because the missing
6th tells you nothing about whether ITS arm/size combo would also have
passed.

**Scenario 5 — all 6 probes present, but `matrix/S`'s `training_curve`
has only a 2-point T1 sequence (simulating `--eval-interval` set too high
relative to `--steps` for that one cell):**

```
  ADMISSION SET OK: all 6 of [('flat', 'M'), ('flat', 'S'), ('flatten', 'M'), ('flatten', 'S'), ('matrix', 'M'), ('matrix', 'S')] present.
  [... the 5 other probes report OK (a)/(a2)/(b) individually ...]
  [embed_ablation_probe_matrix_S] FAIL (a): only 2 T1 eval point(s) in training_curve (need >=3 to judge monotonicity at all) -- t1_seq=[9.0, 6.2]. Check --eval-interval is set low enough relative to --steps for this probe.
  [embed_ablation_probe_matrix_S] (b) extrapolated to steps=2000 batch=64: 0.587 GPU-h (within the 2.0 GPU-h/cell ceiling)
ADMISSION CHECK: FAIL -- STOP, do NOT stage phase B; re-derive --steps or investigate the failing arm/missing probe first
exit code: 1
```

This is exactly the failure mode MJ-1's `--eval-interval 100` fix (§6,
`_gen_specs.py`) prevents in practice — every REGISTERED phase-A probe
now gets 5 eval points, not 2, so scenario 5 should never arise from a
correctly-generated spec; `check_admission()` still catches it loudly if
it ever does (e.g. a hand-edited `cmd`, or a future re-registration that
forgets the coupling).

## 6b. Deployment — exactly what must move to the box, and where

The box (`youthful-indigo-turkey` per `matrix-thinking/H100_SETUP.md`)
has **neither** `matrix-thinking/src/` nor the historical
`experiment-runs/8xh100-session1/*.py` scripts deployed by default — only
`matrix-thinking/chapter2/` and `matrix-thinking/deltanet_rd/`-derived
code is scp'd there per that doc's own "Environment on the box" section.

**Directory must exist before the `scp` (audit round-2, minor):**

```
ssh youthful-indigo-turkey "mkdir -p /home/nvidia/embed_ablation"
scp matrix-thinking/src/embed_ablation_rd.py \
    youthful-indigo-turkey:/home/nvidia/embed_ablation/embed_ablation_rd.py
```

(`youthful-indigo-turkey` per `matrix-thinking/H100_SETUP.md`'s own "the
tunnel — `ssh youthful-indigo-turkey` then works directly" convention —
`scp` does not create intermediate directories on its own, and nothing
else in this design's deployment path creates `/home/nvidia/embed_
ablation/` for it.)

| Local path | Box path | Why |
|---|---|---|
| `matrix-thinking/src/embed_ablation_rd.py` | `/home/nvidia/embed_ablation/embed_ablation_rd.py` | the entire runner; self-contained, zero import dependency on any other repo file (deliberately — `lm_pretrain_rd.py` is under concurrent edit and pulls in frozen-bias/DeltaNet machinery this ablation does not need) |
| *(nothing else)* | — | no other file is required: the corpus loader, all three model arms, the training loop, eval, harvest, and admission-check are all in the one script |

**Data already on the box, not transferred by this design:** the GPT-2
tokenized `wikitext-mix-ext` corpus must already exist at
`/data/deltanet_rd_data/wikitext103_mix_eot_extended/` (confirmed live —
see §3's citation of the completed `645_laneB_...` queue spec that trains
on this exact directory). If a fresh box or fresh data volume does NOT
have this directory, that is a hard blocker for every cell in this
design and must be flagged before phase A (the rate probes) launches —
do not assume it is there without checking `ls /data/deltanet_rd_data/
wikitext103_mix_eot_extended/meta.json` first.

**Checkpoints:** every cell's `--ckpt-dir` is under
`/data/embed_ablation_ckpts/<cell_name>/` — **never** the box's root
filesystem, per the task brief and `CLAUDE.md`'s general data-placement
discipline (mirrors `/data/fixscale_ckpts/...` in the `645_laneB_...`
spec).

**Outputs (audit F1 fix — probes now separated):** phase-B seeded cells'
`--out` is under `/home/nvidia/embed_ablation/results/<cell_name>.json`;
phase-A probe cells' `--out` is under `/home/nvidia/embed_ablation/
results/probes/<cell_name>.json` — a SEPARATE subdirectory, so `harvest()`'s
plain top-level `os.listdir(results_dir)` never even sees probe files
(the `probes/` directory entry does not end in `.json`), on top of the
explicit `role`/filename filter it also applies. The harvest step
(`--harvest --results-dir /home/nvidia/embed_ablation/results --out
.../embed_ablation_summary.json`) runs on the box against the top-level
`results/` directory, or against a copy pulled back to the repo's
`experiment-runs/` archive per that directory's own size-capped
hybrid-archive policy (`experiment-runs/README.md`) — result JSONs here
are tiny (no large Z-dumps or checkpoints), so they belong in the
committed `experiment-runs/` tree, not SSD-only.

**Python:** `/home/nvidia/tdenv/bin/python3` (the box's existing venv,
per `H100_SETUP.md` and every queue spec's `cmd` field) — already has
`torch` per that doc's `pip install torch numpy`. No new dependency is
introduced by this script (it uses only `torch`, stdlib `json`/`math`/
`argparse`/`time`/`pathlib`/`zlib`/`tempfile`).

## 9. Queue specs

30 files: `matrix-thinking/embed_ablation_specs/phase_A_probes/
0640-0645_embed_ablation_probe_*.json` (6) and `phase_B_seeded/
0646-0669_embed_ablation_*.json` (24), schema copied field-for-field from
`experiment-runs/2026-08-29_box_final_archive/queue/completed/
005_laneA_probe_K128_s0.json` (`id`, `lane`, `hypothesis`, `cmd`,
`gpu_h_estimate`, `output_dir`, `validity_check`, `notes`). IDs 0640+ sort
behind the running 1.31B K=16 grace wave (0478-0485) and the 0600-0629
recall-strengthening sweep, per the task brief. `lane: "embed-ablation"`
(a new lane tag — this campaign is independent of lanes A/B used
elsewhere). Each spec's `notes` field states the exact scp prerequisite
(§6b), the F2 gating dependency (phase-B specs only — restated verbatim
from each phase's `README.md`), and the formula basis for its
`gpu_h_estimate` (§5/§6, with the corrected anchor description); each
`cmd` includes `--ceiling-gpuh` as the hard-stop safety valve, phase-A
`cmd`s carry `--eval-interval 100` (audit round-2 MJ-1), and phase-B
`cmd`s carry `--eval-batches 50` (audit M1); every `validity_check`
(both phases, audit round-2 MJ-6) asserts `steps_completed ==
STEPS_{A,B}` EXACTLY — no `-1` slack — plus `complete is True`
(phase-A also checks `role=='probe'`; phase-B also checks
`status=='COMPLETED'` and both `T1`/`T8` present in `final_evals`) — a
cell that silently truncates (Run 22's actual failure mode) or overshoots
fails its own validity check rather than being harvested as if it had
finished. `_gen_specs.py` generates every `--steps` value and every
validity_check's exact-match threshold from ONE constant per phase
(`STEPS_A=500`, `STEPS_B=2000`, audit round-2 MJ-5) so the two can never
be hand-edited out of sync.

**Sequencing (restated from §6, and from both phases' own `README.md`):**
`phase_A_probes/0640-0645` (no dependencies, run first) → confirm real
wall time AND run `--check-admission` (§6, §8c) → only then
`phase_B_seeded/0646-0669` (matrix/flatp/flatd/flatten × S/M × 3 seeds).
Nothing in `embed_ablation_specs/` is queued for execution by this
design — staging only, per the task brief's "queue specs only"
instruction and per the coordinator's explicit "still no commit, no
launch, no staging" instruction on this revision.

## 10. Flags for the audit round (things that do not transfer cleanly —
reported, not assumed away)

- **Simplified window sampling, not `lm_pretrain_rd.py`'s document-aware
  sampler.** `embed_ablation_rd.py`'s `get_batch()` is the plain
  random-contiguous-window sampler from `round1_matrix_script.py`/
  `round2_matrix_script.py` (uniform start index, no document-boundary
  awareness), NOT `lm_pretrain_rd.py`'s AUDIT-FIX-3 machinery
  (`boundary_stats`, `train_doc_offsets`/`val_doc_offsets`-aware
  sampling). All three arms see the identical procedure and the corpus IS
  EOT-separated (so a window that crosses a document boundary still
  carries an in-band `<|endoftext|>` signal, per that file's own
  comment), so this does not introduce an asymmetry between arms — but it
  means this design does not inherit that file's own boundary-crossing-
  rate quantification. (M1's fix — `corpus_fixed_seed` for the EVAL
  generator specifically — IS ported from `lm_pretrain_rd.py`; only the
  TRAINING sampler and the boundary-aware windowing remain un-ported.)
- **`--ceiling-gpuh` is checked only at the top of the training loop, not
  inside an eval pass.** A cell that hits its ceiling mid-eval will run
  that eval to completion before stopping (bounded overshoot of at most
  one `--eval-interval`'s worth of eval batches, small relative to the 2.0
  GPU-h ceiling). The forced final eval on `CEILING_STOP` (minor fix)
  adds one more such bounded overshoot at the very end.
- **The GPU-h estimates in §5/§6/every spec's `gpu_h_estimate` are
  formula-extrapolated from one empirical anchor** (`exp_d16_v2`, a
  different-but-architecturally-related run, now correctly described as
  12 layers/max_len=2048), scaled linearly in batch×steps×params — they
  are NOT measured for this exact script, any of the three arms' exact
  widths, or this exact box. This is exactly why §6 stages 6 rate-probe
  cells ahead of the 24 seeded cells, and why `--check-admission` (M4) is
  a hard gate, not a suggestion, before phase B is staged.
- **~3.3 GB estimated (not measured) host RAM per process** for loading
  the full corpus tensors (§5) — confirm against the first probe cell's
  real process RSS.
- **This design was authored and selftested entirely off-box, on a CPU-
  only scratch venv (`torch==2.14.0`, no CUDA) in a sandbox with no
  access to the H100 box, `/data/deltanet_rd_data`, or the box's own
  `tdenv`.** The corpus-loading code (`load_corpus_tokens`) and the CUDA/
  bf16-autocast training path have been read against `lm_pretrain_rd.py`'s
  contract and written to match it, but have not been exercised against a
  real `wikitext-mix-ext` corpus file or a real GPU — that first real
  exercise IS what the phase-A probes are for. Do not treat the CPU
  selftest's "ALL CHECKS PASSED" (or the `--harvest-selftest`/
  `--check-admission` passes, which are pure-Python logic tests against
  synthetic JSON, no corpus or GPU involved) as evidence the box-side
  data loading or CUDA path is bug-free; they certify the model
  forward/backward/grad-flow, the parameter-matching arithmetic, the
  harvest filtering/grouping/decision logic, and the admission-check
  logic — not the box-specific I/O and CUDA paths.

## §9 Harvest record (2026-09-03)

- Phase A (500-step probes, 0640–0645): admission FAILED on flat_M (T=1 rising over the last three evals). Phase A′ (1000-step probes, 0670–0675, `probes_1000/`): admission FAILED on all six arms by 0.0007–0.013 bpb increases on plateaued curves (flat_M passed). Both logs archived (`stage_phase_b*.log`).
- Author override (Sam, 2026-09-03 11:04 PDT, on the agent's recommendation that the second failure was gate miscalibration, not a model pathology): phase B (0646–0669) staged and run. The pre-registered decision rules were not changed.
- Verdict: STRENGTHEN-01 DROP (matrix vs flatten never beats the flatten arm's seed spread at either size; wins 1/3 S, 0/3 M); STRENGTHEN-04 DROP (flat-P beats matrix at both sizes, 0/3, 0/3). Numbers in `experiment-runs/2026-09-03_embed_ablation/results/harvest_phaseB.json` and EXPERIMENT_LOG.md 2026-09-03 #4. Cluster 4 dropped.
