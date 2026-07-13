#!/usr/bin/env python3
"""t2a_reference_driver_v2_rd.py -- PARAM_AXIS_SCALING_DESIGN.md sec 11.11's
missing DRIVER: wires lm_recall_gap_probe_v2_rd.py's sec 11 T2-repair check
functions (check_t2a1_ceiling / check_t2a2_untrained_control /
check_t2a3_ssm_calibration / check_t1c_reference_did / check_t2b1b_key_
conditioned, plus check_t2b1_mechanism_exists) to REAL HuggingFace reference
models (sec 11.4.2's witness table: W1 RWKV7-Goose-World3-1.5B, W2
gpt2-large, C1 falcon-mamba-7b). Supersedes the QUARANTINED, pre-repair
`experiment-runs/2026-07-12_param_axis_r0/t2a_reference_driver.py` (banned at
the directory level, sec 9.1 no-read list -- see "CONTAMINATION DECLARATION"
below for exactly what was and was not read from it).

THIS FILE'S OWN SCOPE BOUNDARY (sec 11.11's pinned EXECUTION ORDER, honoured
exactly): it BUILDS the driver, fetches nothing itself (gpt2-large is fetched
separately, see DEPLOY notes in the coordinator's report), and its `--smoke`
gate is CPU-only / tiny-model / no-training-disturbance. `--pre-pass` (sec
11.7's model-free N_rows resolution, on OUR OWN corpora, no witness, no
model) IS run for real by this build session -- it disturbs nothing. `--gate`
(the real, multi-witness, multi-corpus T2a/T1c orchestration) is WIRED AND
READY but REFUSES to execute against real downloaded reference models unless
`--i-am-the-t2a-execution-agent` is passed -- mirroring mode_run's own
`--attest-job-terminated` / `--compute-verdict` refusal discipline in
lm_recall_gap_probe_v2_rd.py. **This build session never passes that flag.**
"The implementer does not review or run their own work" (CLAUDE.md) is
enforced here MECHANICALLY, not merely promised.

=============================================================================
THE AUDIT CHAIN, IN SHORT (5 independent fresh-context opus rounds, each
NOT CLEARED; rounds 1-4 EACH caught a FATAL that the PREVIOUS FIX PASS had
introduced -- which is the single most important fact about this file).
=============================================================================
  R1: 1 FATAL, 9 SERIOUS  -- bridged corpus truncated to 400K tokens => the
      sec 11.2 pools are unsatisfiable => W1 (REQUIRED) VOIDs by arithmetic.
  R2: 1 FATAL, 8 SERIOUS  -- the R1 fix loaded witnesses in bf16 and left the
      argmax in bf16 => tie-collapse biased AGAINST the rare planted value at
      a HALT bar (measured: 2.2% of decisions flip vs fp32). Also: R1's new
      5M floor sat EXACTLY on the V2^V5 degenerate point (band width ZERO).
  R3: 1 FATAL, 1 SERIOUS  -- the R2 fix added a val-size floor of 4,194,304
      while the REAL val splits are 2,300,595 and 247,349 => it VOIDed both
      bridged witnesses on the instrument's own plumbing.
  R4: 1 FATAL, 3 SERIOUS  -- the R3 fix DELETED that constant and left the
      REFERENCE => `--gate` died with NameError on every run. It survived a
      "30/30 PASS" smoke because **mode_gate's BODY had ZERO coverage**: all
      five mode_gate tests return at a refusal check. THAT is the root cause
      of the whole chain, and R4 is the round that named it.
  R5: 0 FATAL, 4 SERIOUS  -- **no fifth FATAL; the driver runs correctly**
      (the auditor executed mode_gate's happy path end-to-end). But the
      root-cause fix itself was weak, and one real driver defect remained:
      * SERIOUS-4 (DRIVER): the subset refusal used `>=` (superset), so an
        EXTRA corpus became a GATING T2a-1 leg (t2a1_gate iterates the
        operator's `corpora`, while t2a2/t2a3/t1c iterate REQUIRED_CORPORA)
        => a voiding extra corpus HALTs though both pinned corpora passed.
        FIXED: the relation is now EQUALITY -- the only one right in both
        directions (R3's `<` let subsets through; R4's `>=` let supersets).
      * SERIOUS-1 (TEST): `[6h]`'s pass condition was `INSTRUMENT_VALID is
        False` -- which is ALSO what every defect inside mode_gate produces,
        since the except-handlers convert any NameError into a fail-closed
        void. A test whose PASS condition is the state a bug produces cannot
        detect the bug; the auditor planted two defects that `[6h]` and
        `[6i]` BOTH passed. FIXED: `[6j]` drives the body with HEALTHY
        stubbed cells and asserts INSTRUMENT_VALID is **True**, that
        coverage_summary is non-empty and numerically exact, that the T1c
        advisory's arithmetic is right, that the advisory is NON-GATING, and
        (negative control) that flipping one leg flips the conjunction.
      * SERIOUS-2 (TEST): `[6i]`'s AST logic harvested EVERY function's
        locals as "module-level" (344 known names vs a true ~83) and treated
        a SUBSCRIPT store as binding its base name -- so it missed 4 of 5
        injected defects. FIXED: module bindings from TOP-LEVEL statements
        only; Subscript/Attribute stores bind NOTHING. Re-teeth-tested
        against the auditor's own planted defects: all 4 classes now CAUGHT.
      * SERIOUS-3 (TEST/OPS): `[6h]` called the REAL `load_witness_model`,
        so on the gate box (where /data/hf_cache IS populated -- a
        precondition of the gate) `--smoke` would have loaded RWKV7-1.5B +
        gpt2-large + falcon-mamba-**7B** (~19GB) into CPU RAM, on a box whose
        8 GPUs are training -- violating this file's own pinned scope
        boundary and risking an OOM kill. FIXED: the loader is MOCKED, which
        also pins the covered path (it was HF-cache-state-dependent).

=============================================================================
D5 AUDIT ROUND 3 (fresh-context opus agent, INDEPENDENT of rounds 1 and 2) --
VERDICT: NOT CLEARED. 1 FATAL, 1 SERIOUS. BOTH FIXED.
=============================================================================
Round 3 re-verified all ten round-2 fixes AS LANDED AND CORRECT -- including
the RNG-alignment property of the new n_demos retry loop, which it was asked
to attack hardest and could not break (it re-implemented the loop standalone
and MEASURED that levels 1/2/4 see byte-identical (a,b) pairs per window,
with the counterfactual -- n_demos back in the seed -- correctly failing, so
the test is not vacuous). It then found that **the round-2 fix pass had
itself introduced a new FATAL** -- the third consecutive fix pass to do so,
which is why each one is audited before it is trusted.

  * **FATAL-1(R3) -- THE VAL FLOOR I ADDED VOIDS BOTH BRIDGED WITNESSES.**
    Round 2's own SERIOUS-6 fix added `MIN_BRIDGED_VAL_TOKENS = 4 *
    N_ROWS_DEFAULT * SEQ_LEN_DEFAULT` = 4,194,304 and RAISED on it.
    **MEASURED ON THE BOX, and this is the whole finding: openr1-mix-ext's
    val split is 2,300,595 tokens and wikitext-mix-ext's is 247,349.** BOTH
    admissible corpora sit BELOW that floor -- so `build_bridged_corpus`
    would have raised on EVERY bridged (witness, corpus) pair, VOIDing W1 (a
    REQUIRED ceiling witness) and C1 before a single forward pass, driving
    `INSTRUMENT_VALID = False` and HALTing the whole program, regardless of
    any witness's actual capability. That is the IDENTICAL shape as round 1's
    FATAL (a 400K train budget) and round 2's SERIOUS-1 (a 5M floor sitting
    on the degenerate point): a guard set past the cliff, voiding the very
    thing it guards. And it was wrong IN KIND, not merely mis-tuned: (a) it
    was ASYMMETRIC -- W2 never enters `build_bridged_corpus`, so it would
    face no val floor and run happily on the same 247K split that VOIDed W1,
    which is precisely the "compare the two REQUIRED ceiling witnesses on
    different-sized corpora" sin that `mode_gate`'s own truncation refusal
    condemns; and (b) it was NOT THIS DRIVER'S GATE TO ADD -- window overlap
    on these val splits is a property of the PRE-REGISTERED instrument (our
    own rungs read the same splits, and sec 11.7's pinned pre-pass explicitly
    ladders N_rows to 8192 against them), and sec 9.6's stop rule does not
    let a driver unilaterally re-gate a pinned design property.
    **FIXED: the raise is DELETED.** The underlying statistical concern is
    real, so it is MEASURED AND REPORTED instead -- `corpus_coverage_
    provenance()` emits `val_coverage_ratio` and the implied
    `approx_bootstrap_ci_narrowing_factor` for EVERY cell, bridged AND
    unbridged (round-3 M-5), recorded in `--out` before the read
    (sec 9.6-compliant), so a T1c pass at a large coverage ratio can be
    discounted by a reader rather than silently trusted -- or silently
    VOIDed.
  * **SERIOUS-1(R3) -- THE ROLL-UP FAILED OPEN UNDER SUBSETTING.**
    `gate["t2a3"]` was created only `if "C1_falconmamba" in witnesses`, so
    `--gate --witness W1_rwkv7 W2_gpt2large` (a reasonable thing to do; it
    skips a 7B model load) produced an `INSTRUMENT_VALID` conjunction that
    NEVER SAW a pinned GATING check. Same hole on the corpus axis (every leg
    is `all(... for c in corpora)`). The auditor reproduced it:
    `INSTRUMENT_VALID: True` with T2a-3 never evaluated and one of two pinned
    corpora read. This is round-1 SERIOUS-7's "computed but never read"
    defect reborn as "never computed and never missed" -- the FALSE-PASS
    direction. **FIXED two ways:** `mode_gate` now REFUSES a subset run
    outright (`REQUIRED_WITNESSES` / `REQUIRED_CORPORA`), and every leg of
    the conjunction is built UNCONDITIONALLY over the REQUIRED sets (plus a
    `coverage_complete` leg), so even if that refusal were ever relaxed, a
    missing witness/corpus reads as False rather than vanishing.
  * Round-3 MINORs fixed: M-1 (smoke [1d]'s exception handler was labelled
    [1c] -- a copy-paste that would have mis-attributed a failure of the
    round-2 FATAL fix's own test); M-2 (`_persist` truncate-then-write is not
    atomic -- a crash mid-write destroys the partial file the incremental
    persistence exists to protect; now tmp + `os.replace`); M-3 (`pools` /
    `delta_pool` were hung on the cell and stripped only AFTER a `_persist`,
    so every intermediate write serialized megabytes of pool internals --
    they are locals now); M-4 (`expected_frac_windows_with_eos` computed
    `p*T` clipped at 1.0, an expected COUNT wearing a fraction's name, and
    hardcoded the module-default seq_len -- now the true `1-(1-p)^T`, with
    seq_len threaded through); M-5 (coverage provenance now emitted for
    unbridged W2 too, so the two REQUIRED witnesses are comparable);
    M-7 (the ENCODE side is now cached -- without it a --gate run
    re-tokenizes the FULL train splits, 344,736,296 / 418,108,652 GPT-2
    tokens, FOUR times through RWKV-World's non-fast trie tokenizer: hours of
    wall-clock that would look exactly like a hang).

=============================================================================
D5 AUDIT ROUND 2 (fresh-context opus agent, INDEPENDENT of round 1) --
VERDICT: NOT CLEARED. 1 FATAL, 8 SERIOUS. ALL FIXED; round 3 (above)
re-verified all ten as landed and correct.
=============================================================================
Round 2 confirmed round 1's FATAL-1 and SERIOUS-3/4/5/7/9 genuinely landed
(verified line-by-line against code, not against the docstring's claims) --
but found that **the round-1 fix pass introduced a NEW FATAL and set the
FATAL-1 replacement guard on a provably-broken value.** Both were verified
INDEPENDENTLY by the coordinator against the raw code/arithmetic before
being accepted (a second, conflicting audit relay claimed "CLEARED with 2
SERIOUS" and MISSED both; CLAUDE.md's contradictory-rounds rule was applied:
read the raw artifact, record the tiebreak, never split the difference).

  * **FATAL-1(R2) -- THE bf16 ARGMAX.** Round 1's SERIOUS-8 fix loaded every
    witness in bfloat16 (correct, to stop fp16 overflow on the two bf16-
    native witnesses) but `HFLogitsWrapper` returned that tensor UNCHANGED --
    and the probe module argmaxes the RAW logits
    (lm_recall_gap_probe_v2_rd.py L811/L931/L1895) while explicitly
    `.float()`ing before log_softmax (L832/L949). So a bf16 witness got a
    bf16 argmax. bf16's 8-bit significand COLLAPSES near-ties, and
    `torch.argmax` breaks a tie toward the LOWEST index -- which, because the
    planted value `b` is forced rare (V5) and non-modal (V4) while its rivals
    are common continuations, and because BPE id order tracks frequency, is
    the RIVAL. The bias is therefore DIRECTIONAL AGAINST THE WITNESS at an
    absolute 0.90/0.75 HALT bar that sec 11.4.3 forbids moving after a
    failure -- and ASYMMETRIC, since our own rungs' DeltaNetLM emits fp32.
    **MEASURED ON THE BOX: 2.2% of argmax decisions at a 0.03-logit gap flip
    between fp32 and bf16.** FIXED: `HFLogitsWrapper.forward` now upcasts to
    fp32 (`out.float()`) before the finite-check and the return, fixing every
    downstream read (argmax, log_softmax, all six arms) in one place without
    touching the frozen probe module. Smoke [1d] asserts bf16-model ->
    fp32-output.
  * **SERIOUS-1(R2) -- THE 5M FLOOR WAS THE DEGENERATE POINT.** Round 1's
    `MIN_BRIDGED_TRAIN_TOKENS = 5_000_000` was derived from the KEY side only
    (K2 4e-4 ^ K3 count>=500 => N >= 1.25M). The BINDING constraint is the
    VALUE side: V2 (`count(b) >= 500`) ^ V5 (`p_train(b) <= 1e-4`) gives an
    admissible band `count(b) in [500, 1e-4*N]`, and **V5 is NOT on any
    relaxation ladder** (only K2 is). At N = 5,000,000 that band is
    [500, 500] -- **ZERO WIDTH**. |P_val(a)| < 5 for essentially every key,
    K5 drops every key, P_key = [], and the cell VOIDs before a single plant:
    round 1's own FATAL, re-armed at round 1's own fix. Verified against the
    raw constants (V2_MIN_COUNT_B=500, V5_MAX_P_TRAIN_B=1e-4). FIXED: the
    floor is now DERIVED from the probe's own imported constants
    (`POOL_FLOOR_P_VAL * V2_MIN_COUNT_B / V5_MAX_P_TRAIN_B` = 25,000,000), so
    it cannot drift from them again; a val-side floor (absent entirely in
    round 1) is added; smoke [4-neg-b] asserts the floor is off the
    degenerate point.
  * SERIOUS-2(R2) (the floor's forced-fail test was tautological --
    `isinstance(e, RuntimeError) or True` -- and the whole bridge smoke block
    SKIPped silently, certifying by omission): FIXED -- [4-neg] now pre-seeds
    `_TEXT_CACHE` and drives the REAL `build_bridged_corpus` with no
    filesystem dependency, requiring the SPECIFIC RuntimeError; a missing W1
    tokenizer is now a hard FAIL, never a SKIP.
  * SERIOUS-3(R2) (the `dtype=` kwarg is absorbed into `**kwargs`, so a wrong
    name loads fp32 SILENTLY): the auditor's proposed fix (switch to
    `torch_dtype=`) is BACKWARDS for this box -- VERIFIED EMPIRICALLY on
    transformers 5.12.1, where `torch_dtype=` is DEPRECATED and `dtype=` is
    honored (requested bfloat16 -> realized torch.bfloat16). FIXED with the
    durable half of the finding instead: `load_witness_model` now ASSERTS the
    REALIZED parameter dtype, which catches the failure under any
    transformers version, including a future rename.
  * SERIOUS-4(R2) (the Delta-sweep ran 64 plants/point against the PINNED 2%
    drop cap -- TWO drops VOID a point): FIXED, `N_PLANTS_PER_DELTA_DEFAULT
    = 256` (>=5 drops of headroom). The cap itself is pinned and NOT touched.
  * SERIOUS-5(R2) (the n_demos levels were UNPAIRED -- `n_demos` was in the
    seed, so levels 1/2/4 drew different windows AND different (a,b) pairs,
    confounding the very within-item contrast the diagnostic exists to make):
    FIXED, one seed for all levels; [4f] asserts a single shared
    `paired_seed`.
  * SERIOUS-6(R2) (the truncation flags were still live, there was NO val
    floor, and truncation applied only to the BRIDGED witnesses while W2
    always got the full split): FIXED -- `mode_gate` REFUSES any truncation
    outright; `MIN_BRIDGED_VAL_TOKENS` added; smoke [6d] forced-fail.
  * SERIOUS-7(R2) (only T2a-1 was rolled up; T2a-2/T2a-3/T1c were computed
    and never read -- so a coordinator could green-light the instrument while
    an UNTRAINED model was passing the probe): FIXED, `results["instrument_
    gate"]` now rolls up ALL FOUR gating checks with an `INSTRUMENT_VALID`
    conjunction.
  * SERIOUS-8(R2) (the spliced `eos_id` may be OOD for W1, in a majority of
    windows, with no way to detect it): MITIGATED -- `build_bridged_corpus`
    now emits `eos_frac_val` / `expected_frac_windows_with_eos` and
    `mode_gate` carries them into each bridged cell as `bridge_provenance`,
    so a boundary-token artifact is MEASURABLE rather than latent (sec 9.6's
    stop rule forbids adding this after the read). Zero extra forwards.
  * **SERIOUS-A (coordinator, additive to the audit)** -- `mode_gate` caught
    only `VerdictGradeError`, but round 2's OWN SERIOUS-8 fix raises a plain
    `RuntimeError` on non-finite logits, and `build_bridged_corpus` /
    `load_witness_model` raise RuntimeErrors of their own; ANY of them would
    have escaped and destroyed every result already computed (including from
    an already-loaded falcon-mamba-7b), since `--out` was written ONCE at the
    very end. FIXED: `except Exception` per witness AND per cell, plus
    INCREMENTAL `--out` persistence after every cell (CLAUDE.md: "add
    try/except so one crash doesn't kill remaining configs").
  * **SERIOUS-B (coordinator, = round-2 SERIOUS-5's second half)** --
    `run_n_demos_diagnostic` hand-rolls its outer plant loop, so it does NOT
    inherit `run_t2_repaired_probe`'s pinned 2% drop-cap VOID; round 1 VOIDed
    only on LITERALLY ZERO survivors, so a level built from 4 of 64 windows
    would have silently reported an `acc_copy`. FIXED: the same pinned cap is
    enforced here, plus an absolute `min_surviving` floor; smoke [4h] is its
    forced-fail test.
  * MINORs M-1..M-10 fixed (val-side vocab-range assert, not just train;
    `add_special_tokens=False` + an assertion that the tokenizer did not
    smuggle an eos into a document's own encoding; the vacuous [0.5,inf)
    rival band removed; dead `witness_key`/`query_pos` removed; toothless
    [4e]/[4f] key-presence assertions replaced with content assertions;
    `--out` now carries commit sha + full config).

D5 AUDIT ROUND 1 (fresh-context opus agent) -- VERDICT: NOT CLEARED.
1 FATAL, 9 SERIOUS, 6 MINOR. All fixed; round 2 (above) re-verified them.
Recorded here per this repo's own convention (lm_recall_gap_probe_v2_rd.py's
own "AUDIT (2026-07-12...)" blocks) so the fix history is legible without a
diff.
=============================================================================
1 FATAL, 9 SERIOUS, 6 MINOR. All FATAL/SERIOUS addressed in this revision:
  * FATAL-1: `N_SOURCE_TOKENS_TRAIN_DEFAULT=400_000` made sec 11.2's K2(4e-4
    widest)/K3(count>=500) pool bands UNSATISFIABLE (need N >= 500/4e-4 =
    1,250,000 for even ONE token to qualify; measured n_p_key=0 at 400K,
    n_p_key=161 at the corrected scale) -- **both bridged witnesses (W1
    REQUIRED, C1) VOIDed T2 before a single plant, deterministically,
    regardless of model quality.** FIXED: bridged witnesses now default to
    the FULL train/val split (`n_source_tokens=None`); `build_bridged_
    corpus` hard-asserts `n_tokens_train >= MIN_BRIDGED_TRAIN_TOKENS`
    (5,000,000 -- see the constant's own comment for the arithmetic).
  * SERIOUS-2 (smoke certified the wrong budget; two false PASS labels;
    zero coverage of `build_bridged_corpus` itself): FIXED -- smoke now
    calls the REAL `build_bridged_corpus`/`_retokenize_documents` path with
    a forced-fail negative test at the OLD 400K budget (must VOID) and a
    positive test at a scale above `MIN_BRIDGED_TRAIN_TOKENS`; the two false
    "at W1's REAL vocab size" labels corrected to say what actually ran.
  * SERIOUS-3 (bridged corpus was a zero-EOT flat blob -- the exact form
    `lm_pretrain_rd.load_corpus` REFUSES to load, "majority of windows span
    unrelated documents"): FIXED -- `decode_corpus_to_documents` now returns
    a LIST of per-document strings (never joined into one string), and
    `_retokenize_documents` re-tokenizes EACH document separately and
    splices the WITNESS's OWN `eos_id` between them -- a genuine in-band
    boundary signal in the witness's own vocabulary, not a textual `"\\n\\n"`
    proxy. This also makes `eot_override` non-vacuous for bridged witnesses
    (it previously had almost nothing to exclude).
  * SERIOUS-4 (`n_plants=512` vs sec 11.7's pinned `n_plants(T2) = N_rows`):
    FIXED -- `N_PLANTS_DEFAULT = N_ROWS_DEFAULT`; `mode_gate` now REFUSES to
    run if `--n-plants != --n-windows`.
  * SERIOUS-5 (a witness-specific seed offset broke sec 11.4.6's byte-
    identical-plants property for W2 and T2a-2, where it was achievable --
    candidate detection is model-free, so there was no collision the
    distinct offset was protecting against): FIXED -- every witness and
    T2a-2 now use `RUNG_MATCHING_SEED_OFFSET = 424242`, IDENTICAL to
    `mode_run`'s own offset.
  * SERIOUS-6 (100K-token val truncation gave ~10x window overlap, making
    the clustered bootstrap ~3.2x too narrow -- T1c could pass at DiD~0):
    subsumed by the FATAL-1 fix (full val split by default).
  * SERIOUS-7 (`cell_void_placebo_match` / `cell_void_missing_s2_fields` /
    sec 11.7's row+candidate floors were computed but never READ -- T1c
    could pass on a VOID cell): FIXED -- `run_witness_cell` now computes
    `t1c_admissibility` explicitly and `mode_gate`'s T1c step refuses to
    call `check_t1c_reference_did` unless BOTH witness cells clear it.
    `--pre-pass` (sec 11.7's `resolve_n_rows_pre_pass`, model-free) is now
    wired AND RUN by this build session on our own corpora.
  * SERIOUS-8 (fp16 on two bf16-native witnesses + no NaN/inf guard --
    non-finite logits make `argmax` silently return 0, which reads as "the
    model cannot recall"): FIXED -- `load_witness_model` defaults to
    `torch.bfloat16`; `HFLogitsWrapper.forward` now raises `RuntimeError` on
    any non-finite output rather than silently scoring it.
  * SERIOUS-9 (`--gate` discarded `t2a1_ceiling` etc. unless `--out` was
    passed; no conjunctive T2a-1 verdict was ever assembled): FIXED --
    `--out` is now REQUIRED for `--gate`; `mode_gate` computes and prints an
    explicit `t2a1_gate_conjunction` (W1 AND W2, per corpus).
  * SERIOUS-10 (sec 11.11 build-gate item 5 -- the stratified `acc_copy`
    report, the W2 Delta-sweep, the `n_demos in {1,2,4}` diagnostic -- was
    unimplemented, and sec 9.6 forbids adding it after a T2a read): FIXED --
    `stratified_acc_copy_report` (always computed, cheap, no extra forward
    passes), `run_delta_sweep`, and `run_n_demos_diagnostic` (+ its own NEW,
    driver-local `plant_and_verify_t2_window_ndemos` hard assertion,
    explicitly flagged below as NOT part of the already-audited probe module
    and requiring its own fresh scrutiny) are all implemented and wired into
    `mode_gate` for W2 (matching sec 11.4.3's own "the W2 Delta-sweep"
    naming).
  * MINOR fixes (M-1..M-6, see inline comments at each site): the eot_
    override smoke test now does a REAL functional check (not a tautological
    signature comparison); the EOT call-site inventory below is corrected to
    five sites, not four; a vocab-range assertion guards against a
    re-tokenized id landing outside `vocab_size`; `GPT2TokenizerFast` (not
    the slow tokenizer) is used and loaded once; `mode_gate` catches
    `VerdictGradeError` per-witness so one bad cell does not kill the whole
    multi-witness run; `_TEXT_CACHE`'s bound is now documented (naturally
    <= 2 corpora x 2 splits for this design's admissible-corpora set).

=============================================================================
THE BRIDGE -- THE CENTRAL DESIGN DECISION, STATED UP FRONT FOR THE AUDITOR
=============================================================================
The reference witnesses do NOT share our GPT-2 tokenizer (W1 RWKV7-World's
own tokenizer, vocab ~65536; C1 falcon-mamba's own tokenizer, vocab ~65024)
-- EXCEPT W2 (gpt2-large), which is GPT-2-architecture and uses the
IDENTICAL GPT-2 BPE tokenizer as every corpus in this project (sec 11.4.2's
own table: W2's bridge column reads "NONE"). This is by design (sec 11.4.2:
"the reason W2 shares our tokenizer") and this driver exploits it: **W2
needs no bridge code path at all** -- it loads OUR OWN `load_corpus(...)`
output directly, byte-identical to how lm_recall_gap_probe_v2_rd.py's own
`mode_run` loads a corpus for one of our checkpoints. Every function below
therefore treats "bridge" as a per-witness boolean, not a universal code
path.

For W1/C1 (and the optional, unfetched W3 substitute), a bridge is
unavoidable. TWO designs were considered:

  (A) PER-WINDOW / PER-CALL bridging: keep windows in GPT-2 token-id space
      throughout, and make `HFLogitsWrapper.__call__(x)` (which receives a
      [B, T] GPT-2-id tensor, exactly the same call signature run_ablation_
      arm / run_t2_repaired_probe / run_did_eval use for every model) decode
      + re-tokenize + realign positions ON EVERY FORWARD CALL, returning a
      [B, T, V_ref]-shaped tensor whose T-axis still indexes GPT-2 token
      positions. REJECTED: those three orchestration functions read
      `logits.argmax(dim=-1)` at a caller-supplied GPT-2-space position
      index k (always the plant's own k0 for T2, or a candidate's own k for
      the main metric) that `HFLogitsWrapper` has NO way to discover from
      `x` alone (k0 is drawn per row by a caller `HFLogitsWrapper` never
      sees) -- correctly bridging position (A) would require either a
      character-offset alignment table rebuilt on EVERY call (expensive, and
      still ambiguous whenever a GPT-2 token's re-tokenized span does not
      align to a reference-tokenizer boundary) or a fragile external
      side-channel coupling this wrapper to the internal batching order of
      functions this file must not modify. Rejected as fragile and, worse,
      SILENTLY fragile -- exactly the "could make a model look better OR
      worse than it is" failure mode this driver must not introduce.

  (B) CORPUS-LEVEL bridging (ADOPTED): decode our OWN GPT-2-tokenized TRAIN
      and VAL splits back to raw text ONCE per (witness, corpus), re-tokenize
      that text with the witness's OWN tokenizer ONCE, and thereafter treat
      the re-tokenized id tensor AS the corpus for that witness -- feeding it
      into `build_bigram_mode_table` / `build_key_value_pools` / `run_did_
      eval` / `run_t2_repaired_probe` EXACTLY as this module already treats
      any of our own GPT-2-tokenized corpora, with `vocab_size=` set to the
      witness's own vocab size. Candidate detection, key/value pool
      construction, Delta sampling, and the plant itself are therefore
      computed NATIVELY in the witness's own token space -- there is no
      GPT-2-token-id position to realign, because no GPT-2 token id survives
      past `build_bridged_corpus`. `HFLogitsWrapper` becomes a near-one-line
      adapter because by the time it is ever called, `x` is already native
      to the model it wraps. This is the SAME pattern this repo already uses
      and has run before: `wave_neg1_hf_reference_smoke.py` (Wave -1) and
      the retired, quarantined `t2a_reference_driver.py` both decode-then-
      re-tokenize the WHOLE corpus once, never per-window. Design (B) is not
      a rediscovery, it is a continuation of established, working practice
      in this exact directory -- see the CONTAMINATION DECLARATION for what
      was and was not reused from the quarantined copy. **D5 audit round 1
      attacked this design decision directly and could not break it** (see
      "SURVIVED" items in the coordinator's audit record) -- the round-1
      FATAL/SERIOUS findings were all about the bridge's CALIBRATION
      (corpus size, EOT splicing, seeds, floors), not its architecture.

CONSEQUENCE OF (B), STATED PLAINLY: a "plant" for a bridged witness is a
(key, value) PAIR OF REFERENCE-VOCAB TOKENS chosen directly from that
witness's own re-tokenized corpus statistics -- it is NEVER a GPT-2 token
translated across the boundary. **The literal question "can a GPT-2 single
token become multi-token in the reference vocab, corrupting the plant" does
not arise for this design** (verified against code by the round-1 auditor,
independently, not merely asserted here), because no GPT-2 token ever enters
plant selection for a bridged witness.

=============================================================================
EOT / D6 -- FIXED, NOT MERELY DISCLOSED
=============================================================================
sec 10.6-D6 flagged (on the retired instrument): "`EOT_TOKEN_ID` is GPT-2's
50256 even under the reference tokenizers." **FIVE** call sites in
lm_recall_gap_probe_v2_rd.py read EOT (round-1 audit M-2: the prior revision
of this docstring said "four" and omitted one -- corrected):
`detect_candidates_and_baseline` (L602), `assign_placebo_positions` (L691,
L697), `draw_exclusive_replacement` (L722) read the module GLOBAL
`EOT_TOKEN_ID` at CALL TIME (a bare name looked up in that module's own
namespace on every call -- Python's ordinary global-lookup semantics, not
special to this file); `build_key_value_pools`'s `eot_token_id` PARAMETER
(L1448) instead defaults to that same global but BINDS IT AT DEF TIME (an
ordinary Python default-argument gotcha: re-assigning the module global after
import does NOT change an already-bound default; VERIFIED EMPIRICALLY by the
round-1 auditor, not merely asserted). Both mechanisms are fixed together,
because either alone is insufficient:
  - `eot_override(eot_id)` (below): a context manager that monkeypatches
    `lm_recall_gap_probe_v2_rd.EOT_TOKEN_ID` for its duration and restores
    the ORIGINAL value on exit (normal OR exception) -- fixes the FOUR
    call-time-lookup sites.
  - Every call into `build_key_value_pools` in this file ALSO passes
    `eot_token_id=<the witness's own eos id>` explicitly -- fixes the
    def-time-bound default, which the context manager alone cannot reach.
  `smoke()` now exercises this with a REAL functional test (round-1 M-1: the
  prior version compared a cached signature object to itself, a tautology
  that could not fail) -- a synthetic corpus is built so a SPECIFIC token
  would qualify for `P_key`, and the test confirms that token IS excluded
  when `eot_token_id=` is passed explicitly and IS NOT excluded when only
  `eot_override` (no explicit kwarg) is active.

**Residual disclosure, materially SHRUNK by the SERIOUS-3 fix above** (per-
document re-tokenization + witness-own-`eos_id` splicing means a genuine
in-band boundary token now exists in the bridged corpus at every document
seam, so `eot_override`/the explicit `eot_token_id=` argument now have real,
non-vacuous work to do). What remains open: a witness's tokenizer may itself
represent multi-byte or multi-token sequences around the spliced boundary
in a way this driver has not exhaustively verified against all three
witnesses' real tokenizers (only W1's, cached and reachable from a local
dev box, was exercised in `smoke()`) -- flagged for the execution agent to
re-confirm against C1 (falcon-mamba-7b) and W3 if ever fetched.

=============================================================================
DELTA POOL PROVENANCE
=============================================================================
sec 9.4 pins T2's plant distances to "the main metric's own empirical Delta
distribution." For a WITNESS this cannot be reused from our own rungs' pool
(Delta is a token-distance, and re-tokenization changes token boundaries)
and, independently, our own rungs' pool lives ONLY inside the QUARANTINED
`experiment-runs/2026-07-12_param_axis_r0/r0_v2_result.json`
(`all_deltas_pooled`), which this driver's author is BANNED from reading for
values (sec 9.1 no-read list). **This driver instead harvests each witness's
Delta pool FRESH, from that SAME witness's OWN T1c `run_did_eval` candidate
population** (every candidate record already carries a `"delta"` field).
Candidate detection is model-free (round-1 auditor verified this directly
against code, "SURVIVED" item), so no circularity is introduced between
T1c's `did`/`did_ci` and T2's Delta pool.

=============================================================================
CONTAMINATION DECLARATION (sec 9.1's no-read list; this driver's own)
=============================================================================
Read, driver-code-only, per this task's explicit authorization: `experiment-
runs/2026-07-12_param_axis_r0/t2a_reference_driver.py` (186 lines, in full).
It is a CLI script with NO embedded per-rung DiD/gap_true/gap_placebo/S1/S2/
acc_copy value of our own checkpoints anywhere in it -- it is driver
plumbing (argparse, an `HFLogitsWrapper` class, a `decode_val_text` helper,
a `main()` that calls the OLD, RETIRED `pick_t2_marker_tokens`/
`run_t2_planted_copy`/`check_t2b1_mechanism_exists`). Confirmed against the
design doc's own contamination ledger (sec 11.10: the three archived driver
scripts in that directory "currently carry no quarantined value" as of the
sweep that banned the directory at the directory level regardless).
**Nothing quantitative crossed over.** What WAS reused, as a PATTERN, not as
copied code: the `HFLogitsWrapper` adapter shape, the `os.environ`
offline-mode setdefaults, and the general "decode our val text, re-tokenize
with the target tokenizer" shape that file itself borrowed from the
non-quarantined `wave_neg1_hf_reference_smoke.py`. **What was deliberately
NOT reused:** (a) the OLD driver decoded a FLAT, EOT-stripped blob and split
the RE-TOKENIZED stream in half for train/val -- silently letting val-split
text leak into "train"; THIS driver decodes/re-tokenizes OUR train and OUR
val splits SEPARATELY; (b) the OLD driver reused the RETIRED single-global-
pair `pick_t2_marker_tokens`/`run_t2_planted_copy`; THIS driver wires the
REPAIRED `build_key_value_pools`/`run_t2_repaired_probe`; (c) the OLD driver
never addressed D6; THIS driver fixes it. No other file inside the banned
directory was opened by this build (not the JSON results, not
`r0_v2_run.log`, not `param_axis_r0_driver_v2.py`, not
`t2a_void_diagnosis.py`/`.json`).

Zero training. Zero execution against a real downloaded reference model by
this build session (see `--gate`'s refusal, and the coordinator's report).
`--pre-pass` and `--smoke` (both model-free or tiny-model, CPU) ARE run for
real by this build session.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pod-safe imports
sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

import torch  # noqa: E402

import lm_recall_gap_probe_v2_rd as probe  # noqa: E402 -- module object, for eot_override's monkeypatch
from lm_recall_gap_probe_v2_rd import (  # noqa: E402
    build_bigram_mode_table, build_key_value_pools, run_t2_repaired_probe,
    run_did_eval, finalize_cell, check_t2b1_mechanism_exists, check_t2b1b_key_conditioned,
    check_t2a1_ceiling, check_t2a2_untrained_control, check_t2a3_ssm_calibration,
    check_t1c_reference_did, resolve_n_rows_pre_pass, N_ROWS_DEFAULT, EVAL_MICRO_BATCH_DEFAULT,
    CANDIDATE_FLOOR_RESOLVED, ROW_FLOOR_CONTRIBUTING, VerdictGradeError, PlantContestedError,
    # sec 11.2's pinned pool constants -- imported, NEVER re-declared, so the bridged-corpus
    # size floors below are DERIVED from the same numbers build_key_value_pools actually
    # enforces (D5 round-2 SERIOUS-1: a hardcoded floor drifted onto the degenerate point).
    V2_MIN_COUNT_B, V5_MAX_P_TRAIN_B, POOL_FLOOR_P_VAL, T2_DROP_CAP_FRAC, T2_TRIPLE_MAX_TRIES,
    _combine_seed, _make_window, VOCAB_SIZE as GPT2_VOCAB_SIZE,
)
from lm_pretrain_rd import (  # noqa: E402
    DeltaNetLM, load_corpus, get_batch, CORPUS_DIRS, EOT_TOKEN_ID as GPT2_EOT,
    DEFAULT_DATA_DIR, corpus_fixed_seed,
)

VOCAB_SIZE_GPT2 = GPT2_VOCAB_SIZE   # 50257 -- imported from the audited module, not re-declared

# sec 11.4.2's pinned witness table. `bridge=False` for W2 is the load-bearing
# fact this whole file's design leans on (see module docstring).
WITNESS_SPECS = {
    "W1_rwkv7":       {"hf_repo": "RWKV/RWKV7-Goose-World3-1.5B-HF", "bridge": True,
                        "role": "T2a-1 ceiling (recurrent / delta-rule family, REQUIRED)",
                        "t2a3": False},
    "W2_gpt2large":    {"hf_repo": "gpt2-large", "bridge": False,
                         "role": "T2a-1 ceiling (attention / induction-head family, REQUIRED)",
                         "t2a3": False},
    "W3_llama32_1b":   {"hf_repo": "meta-llama/Llama-3.2-1B", "bridge": True,
                         "role": "reported; substitute for W2 ONLY if W2 unavailable -- "
                                 "NOT fetched by this build (W2 is available)",
                         "t2a3": False},
    "C1_falconmamba":  {"hf_repo": "tiiuae/falcon-mamba-7b", "bridge": True,
                         "role": "T2a-3 SSM calibration (causal-only, demoted from the 0.90 ceiling)",
                         "t2a3": True},
}

# D5 round-1 SERIOUS-5 fix: use mode_run's OWN offset for every witness AND T2a-2, not a
# witness-specific offset. Candidate detection (detect_candidates_and_baseline) is model-free,
# so two witnesses on the SAME corpus at the SAME seed get the IDENTICAL window/candidate
# population by construction -- there was never a collision a distinct offset was protecting
# against, and using one broke sec 11.4.6's byte-identical-plants property for W2 (unbridged,
# shares our tokenizer) and for T2a-2 (uses our own corpus) with no benefit.
RUNG_MATCHING_SEED_OFFSET = 424242        # lm_recall_gap_probe_v2_rd.py mode_run's own offset
T2A2_UNTRAINED_INIT_SEED = 314159         # arbitrary, fixed, documented -- not a date

# sec 11.4.2/11.4.5: ALL of these are GATING, on EACH corpus. `mode_gate` REFUSES a subset run
# (D5 round-3 SERIOUS-1 -- a narrowed conjunction is a false pass).
REQUIRED_WITNESSES = ("W1_rwkv7", "W2_gpt2large", "C1_falconmamba")
REQUIRED_CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")

# D5 round-1 FATAL-1 fix: default to the FULL train/val split (None = no cap) rather than a
# fixed token budget. `None` is threaded through decode_corpus_to_documents /
# _cached_corpus_documents / build_bridged_corpus as "decode every document."
N_SOURCE_TOKENS_TRAIN_DEFAULT = None
N_SOURCE_TOKENS_VAL_DEFAULT = None

# Declared BEFORE the floors below, which are derived from them.
SEQ_LEN_DEFAULT = 512
# D5 round-1 SERIOUS-4 fix: sec 11.7 pins n_plants(T2) = N_rows exactly.
N_PLANTS_DEFAULT = N_ROWS_DEFAULT

# =============================================================================
# THE BRIDGED-CORPUS SIZE FLOORS -- DERIVED FROM THE PROBE'S OWN PINNED CONSTANTS,
# NEVER HARDCODED. (D5 round-2 SERIOUS-1: the round-1 fix hardcoded 5,000,000,
# which is EXACTLY the degenerate point -- see the arithmetic below. A guard set
# on the cliff edge is not a guard. Deriving it from the imported constants also
# means a future change to V2/V5 in the probe module cannot silently invalidate
# this floor.)
# =============================================================================
# The BINDING constraint is the VALUE side, not the key side -- which is what the
# round-1 fix missed. build_key_value_pools requires, for every value b:
#     V2: count(b) >= V2_MIN_COUNT_B        (= 500)
#     V5: p_train(b) <= V5_MAX_P_TRAIN_B    (= 1e-4)  ==>  count(b) <= 1e-4 * N
# so the admissible value band is  count(b) in [500, 1e-4 * N].
# CRUCIALLY, unlike K2, **V5 IS NOT ON ANY RELAXATION LADDER** (probe L1541-1546
# ladders K2 only) -- so this band cannot widen itself under pressure.
#     N = 1.25M -> [500,   125]  EMPTY   (band inverted)
#     N = 5.00M -> [500,   500]  WIDTH 0 <-- the round-1 floor. Only tokens with
#                                            count EXACTLY 500 qualify; |P_val(a)|<5
#                                            for essentially every a, so K5 drops every
#                                            key, P_key = [], and the cell VOIDs before
#                                            a single plant. The round-1 FATAL, re-armed
#                                            at the round-1 fix's own new floor.
#     N = 25.0M -> [500, 2,500]  5x width  <-- what we require
#     N = 43.7M -> [500, 4,370]  8.7x      <-- the real corpora (sec 11.2's own measurement)
# Require the value band to carry at least POOL_FLOOR_P_VAL (=5) multiples of the V2
# floor, i.e. V5_MAX_P_TRAIN_B * N >= POOL_FLOOR_P_VAL * V2_MIN_COUNT_B:
MIN_BRIDGED_TRAIN_TOKENS = int(POOL_FLOOR_P_VAL * V2_MIN_COUNT_B / V5_MAX_P_TRAIN_B)   # = 25,000,000
# THE VAL SIDE IS **MEASURED AND RECORDED, NEVER GATED** -- and the round-2 fix that tried to
# gate it was itself a FATAL (D5 round-3 FATAL-1, caught by the third independent auditor and
# CONFIRMED against the raw corpora before acceptance).
#
# THE ROUND-2 MISTAKE, RECORDED SO IT IS NOT REPEATED A FOURTH TIME: round 2 added
# `MIN_BRIDGED_VAL_TOKENS = 4 * N_ROWS_DEFAULT * SEQ_LEN_DEFAULT` = 4,194,304 and RAISED on it.
# MEASURED ON THE BOX: openr1-mix-ext val = 2,300,595 tokens; wikitext-mix-ext val = 247,349.
# **Both admissible corpora are BELOW that floor** -- so the guard would have VOIDed W1 (a
# REQUIRED ceiling witness) and C1 on BOTH corpora, before a single forward pass, regardless of
# any witness's actual capability. That is the identical "guard set past the cliff" shape as the
# round-1 FATAL (a 400K train budget) and the round-2 SERIOUS-1 (a 5M floor on the degenerate
# point) -- three consecutive fix passes, three guards that VOID the thing they guard.
#
# AND THE GATE WOULD HAVE BEEN WRONG IN KIND, not merely mis-tuned:
#  (a) ASYMMETRIC. W2 (bridge=False) never enters build_bridged_corpus, so it would face NO val
#      floor and run happily on the very same 247K-token split the floor VOIDs W1 on -- exactly
#      the "compare the two REQUIRED ceiling witnesses on different-sized corpora" sin that
#      mode_gate's OWN truncation refusal condemns.
#  (b) IT IS NOT THIS DRIVER'S GATE TO ADD. Window overlap on these val splits is a property of
#      the PRE-REGISTERED instrument, not of the bridge: our own rungs read the same splits, and
#      sec 11.7's pinned pre-pass explicitly ladders N_rows over (2048, 4096, 8192) against them
#      -- i.e. the design already sanctions sampling up to 8192x513 window-tokens from a
#      247K-token val split. sec 9.6's stop rule does not let a driver unilaterally re-gate a
#      pinned design property.
#
# The underlying statistical concern is REAL (heavy overlap makes clustered_bootstrap_ci, which
# resamples over row_idx and treats rows as independent clusters, too NARROW -- a "witness looks
# BETTER than it is" false pass on T1c). So it is MEASURED and EMITTED for EVERY cell --
# bridged AND unbridged, symmetrically -- via `corpus_coverage_provenance` below, recorded in
# --out BEFORE the read (sec 9.6-compliant), exactly as the eos-density provenance is. A reader
# can then see the redundancy factor and discount the CI accordingly; nothing is silently
# hidden, and nothing capable is VOIDed by the instrument's own plumbing.


def corpus_coverage_provenance(n_tokens_val: int, n_windows: int, seq_len: int) -> dict:
    """D5 round-3 FATAL-1's replacement for the deleted val-size GATE: the
    same quantity, REPORTED instead of enforced, and computed identically for
    bridged and unbridged witnesses so the two REQUIRED ceiling witnesses are
    comparable on this axis. `val_coverage` > 1 means the sampled window
    token-slots exceed the val split, i.e. windows necessarily overlap and
    `clustered_bootstrap_ci`'s independent-cluster assumption is optimistic by
    roughly sqrt(coverage)."""
    slots = n_windows * (seq_len + 1)
    coverage = (slots / n_tokens_val) if n_tokens_val else None
    return {
        "n_tokens_val": n_tokens_val, "n_windows": n_windows, "seq_len": seq_len,
        "window_token_slots": slots,
        "val_coverage_ratio": coverage,
        "approx_bootstrap_ci_narrowing_factor": (coverage ** 0.5) if coverage and coverage > 1 else 1.0,
        "note": ("val_coverage_ratio > 1 => windows overlap => clustered_bootstrap_ci treats "
                 "non-independent rows as independent clusters, so its CI is AT MOST "
                 "sqrt(coverage) too NARROW (the ICC=1 worst case; overlapping windows share text "
                 "but not context, so the true inflation is strictly lower). This cuts BOTH ways "
                 "and the second direction is the dangerous one: it makes T1c's `did_ci[0] > 0` "
                 "and T2a-3's `KS CI excludes 0` EASIER (false PASS), and T2a-2's `KS CI INCLUDES "
                 "0` HARDER (a false FAIL of the untrained negative control = INSTRUMENT-INVALID, "
                 "HALT BY DEFECT). This is a property of the PRE-REGISTERED instrument "
                 "(our own rungs read the same val splits; sec 11.7's pre-pass ladders N_rows to "
                 "8192 against them), NOT of the bridge -- it is REPORTED here, never gated, "
                 "because gating it would VOID a capable witness on the instrument's own plumbing "
                 "(D5 round-3 FATAL-1) and would apply asymmetrically to the bridged witnesses "
                 "only. A T1c pass at a large coverage ratio should be discounted accordingly."),
    }

# 14M rung's exact architecture (PARAM_AXIS_SCALING_DESIGN.md table, "14M |
# 14,048,896 | dm256/L2/ds64"; also DeltaNetLM's own defaults, sec 11.4.2's
# T2a-2 gate) -- spelled out explicitly rather than relying on the defaults,
# so a reader does not have to cross-check lm_pretrain_rd.py to know what
# "the 14M rung's exact architecture" means here.
RUNG_14M_CONFIG = dict(d_model=256, d_state=64, n_layers=2, conv_size=4)

# sec 11.4.3 step 2's pre-registered diagnostic ladder, pinned NOW (sec 9.6's stop rule --
# cannot be added after a T2a read).
DELTA_SWEEP_TARGETS = (2, 5, 10, 20, 40, 200, 400)
N_DEMOS_VALUES = (1, 2, 4)
# D5 round-2 SERIOUS-4/SERIOUS-5: 64 (round 1's value) is below the 2%-drop-cap hair-trigger
# threshold -- at n=64 TWO dropped windows (3.1%) VOID a diagnostic point that sec 9.6's stop
# rule forbids re-running. 256 buys >=5 drops of headroom at the PINNED cap, and lowers the
# per-point binomial SE from ~6% to ~3%.
N_PLANTS_PER_DELTA_DEFAULT = 256
N_PLANTS_PER_NDEMOS_DEFAULT = 256
# The n_demos diagnostic hand-rolls its own plant loop (it cannot reuse run_t2_repaired_probe,
# which plants exactly ONE demonstration), so it does NOT inherit that function's drop-cap VOID
# for free. It enforces the SAME pinned cap itself, plus an absolute floor below which an
# acc_copy is not reportable at all.
NDEMOS_MIN_SURVIVING_PLANTS = 20


# =============================================================================
# THE BRIDGE, PART 1: the model-calling adapter. See module docstring
# "THE BRIDGE" -- this class is intentionally thin; the actual bridging work
# is corpus-level (build_bridged_corpus below), not per-call.
# =============================================================================
class HFLogitsWrapper(torch.nn.Module):
    """Adapts a transformers `PreTrainedModel` causal LM (whose `__call__`
    returns a `ModelOutput` with a `.logits` attribute of shape
    `(B, T, vocab_size)`) to lm_recall_gap_probe_v2_rd.py's `model(x) ->
    logits_tensor` calling convention -- the ONLY interface every reused
    orchestration function (`run_ablation_arm`, `run_t2_repaired_probe`'s
    intact pass, `run_did_eval`'s intact pass) assumes: `logits =
    model(chunk); logits.argmax(dim=-1)`, directly on the return value,
    never on `.logits`. By the time `x` reaches this wrapper it is ALREADY
    in the wrapped model's own vocabulary (see module docstring, "THE
    BRIDGE") -- this class has no tokenizer, no decode/re-tokenize logic,
    and no position-alignment logic, by design.

    `use_cache=False`: these are independent, full-window forward passes
    (never incremental decoding), so retaining a KV-cache-equivalent object
    across calls is pure memory overhead, worst for the largest witness
    (falcon-mamba-7b). Wrapped in a narrow TypeError fallback in case a
    trust_remote_code model class does not accept the kwarg.

    D5 round-1 SERIOUS-8 fix: a non-finite (`NaN`/`inf`) logits tensor makes
    `argmax(dim=-1)` return 0 SILENTLY, which reads downstream as "the model
    never predicts the planted value" -- indistinguishable from a genuine
    recall failure. This is the exact "witness looks worse than it is"
    failure mode the D5 audit brief asked to be constructed. Two of the
    three witnesses (RWKV7, falcon-mamba) are bf16-native; fp16 overflow on
    either is a real risk. Raises loudly instead of scoring silently."""

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self.hf_model(input_ids=x, use_cache=False).logits
        except TypeError:
            out = self.hf_model(input_ids=x).logits
        # D5 ROUND-2 FATAL-1 FIX -- THE READ IS fp32, ALWAYS.
        # The round-1 SERIOUS-8 fix (bfloat16 weights, to stop fp16 overflow on the two
        # bf16-native witnesses) introduced a NEW defect: the probe module argmaxes the RAW
        # logits (lm_recall_gap_probe_v2_rd.py L811 / L931 / L1895) while it explicitly
        # `.float()`s before log_softmax (L832 / L949). A bf16 witness therefore got a bf16
        # argmax. bf16 has an 8-bit significand (ULP ~0.125 near a top logit of ~16), so
        # near-ties COLLAPSE, and `torch.argmax` breaks a tie by returning the LOWEST index.
        # The planted value `b` is forced rare (V5: p_train(b) <= 1e-4) and non-modal (V4:
        # rank 2-50) while its rivals are common continuations -- and in BPE-family
        # vocabularies id order tracks merge rank ~ frequency, so `b` carries a HIGH id and
        # its rivals LOW ids. On a collapsed tie the RIVAL WINS. The bias is therefore
        # DIRECTIONAL AGAINST THE WITNESS, at an absolute 0.90/0.75 HALT bar that sec 11.4.3
        # forbids moving after a failure. MEASURED on this box: 2.2% of argmax decisions at a
        # 0.03-logit gap flip between fp32 and bf16. It is also ASYMMETRIC -- our own rungs'
        # DeltaNetLM emits fp32, so the witnesses would be held to the same absolute bar at
        # strictly coarser precision than the models the instrument is built to judge.
        # Upcasting here (not in the frozen probe module) fixes every downstream read --
        # argmax, log_softmax, and every arm -- in one place. ~2.1 GB fp32 transient at the
        # driver's own eval_micro_batch=16 / T=512 / V=65536, trivial on an 80GB H100.
        out = out.float()
        if not torch.isfinite(out).all():
            n_bad = int((~torch.isfinite(out)).sum().item())
            raise RuntimeError(
                f"HFLogitsWrapper: {n_bad} non-finite logit value(s) (pre-upcast dtype was the "
                f"model's own). argmax(dim=-1) would silently return 0 here and be "
                f"indistinguishable from a genuine 'model does not recall' result -- refusing to "
                f"score. D5 round-1 SERIOUS-8."
            )
        return out


# =============================================================================
# D6 FIX: the EOT_TOKEN_ID override. See module docstring "EOT / D6".
# =============================================================================
@contextlib.contextmanager
def eot_override(eot_id: int):
    """Monkeypatches `lm_recall_gap_probe_v2_rd.EOT_TOKEN_ID` (the module
    GLOBAL, looked up by bare name at CALL TIME inside `detect_candidates_
    and_baseline` / `assign_placebo_positions` / `draw_exclusive_
    replacement`) to `eot_id` for the duration of the `with` block,
    restoring the ORIGINAL value on exit -- normal return OR exception.
    Does NOT, by itself, fix `build_key_value_pools`'s own `eot_token_id`
    parameter (bound to the ORIGINAL global at function-DEFINITION time,
    per ordinary Python default-argument semantics) -- every call site in
    this file that needs a non-GPT-2 EOT passes `eot_token_id=` to that
    function EXPLICITLY, in addition to wrapping the call in this context
    manager, and `smoke()` asserts both halves of the fix independently
    with a REAL functional test (not a signature comparison)."""
    original = probe.EOT_TOKEN_ID
    probe.EOT_TOKEN_ID = eot_id
    try:
        yield
    finally:
        probe.EOT_TOKEN_ID = original


# =============================================================================
# THE BRIDGE, PART 2: corpus-level decode -> re-tokenize. See module
# docstring "THE BRIDGE" for why this is corpus-level, not per-call.
# =============================================================================
def decode_corpus_to_documents(tokens: torch.Tensor, doc_offsets: torch.Tensor, gpt2_tokenizer,
                                n_source_tokens, eot_id: int = GPT2_EOT) -> list:
    """Decodes up to `n_source_tokens` GPT-2 tokens of OUR OWN corpus back
    to raw text (`n_source_tokens=None` decodes the WHOLE split), returning
    a LIST of per-document strings -- one entry per document boundary
    `load_corpus`'s own `doc_offsets` recorded. D5 round-1 SERIOUS-3 fix:
    the prior revision JOINED documents into one `"\\n\\n"`-separated
    string and re-tokenized that single string, which (a) inserts no real
    tokenizer boundary at the seam and (b) reintroduces exactly the
    boundary-free corpus form `lm_pretrain_rd.load_corpus` REFUSES to
    accept ("zero <|endoftext|> tokens; majority of windows span unrelated
    documents"). Returning a LIST (never joined) lets the caller
    (`_retokenize_documents`) splice the WITNESS's OWN eos id between
    per-document RE-TOKENIZED id sequences -- a genuine in-band boundary in
    the witness's own vocabulary, not a textual proxy for one."""
    doc_offsets_list = doc_offsets.tolist()
    n = tokens.numel()
    chunks = []
    total = 0
    for i, start in enumerate(doc_offsets_list):
        end = doc_offsets_list[i + 1] if i + 1 < len(doc_offsets_list) else n
        doc_ids = [t for t in tokens[start:end].tolist() if t != eot_id]
        if not doc_ids:
            continue
        chunks.append(gpt2_tokenizer.decode(doc_ids))
        total += len(doc_ids)
        if n_source_tokens is not None and total >= n_source_tokens:
            break
    return chunks


_TEXT_CACHE: dict = {}
# (data_dir, corpus_name, split, n_source_tokens) -> list[str], witness-independent (the decode
# step never touches a witness's tokenizer). D5 round-1 M-6: this is NOT truly unbounded in
# practice -- this design's admissible-corpora set is exactly {"openr1-mix-ext",
# "wikitext-mix-ext"} x {"train","val"}, so a realistic --gate run populates at most 4 entries
# regardless of how many witnesses read them. Documented rather than engineered around, since
# engineering an eviction policy for a 4-entry cache would be pure overhead.


def _cached_corpus_documents(data_dir: str, corpus_name: str, split: str, gpt2_tokenizer,
                              n_source_tokens) -> list:
    """Caches decode_corpus_to_documents's output PER (corpus, split,
    budget) -- NOT per witness -- so every bridged witness re-tokenizes the
    identical source documents for a given corpus, and so a multi-witness
    `--gate` run pays the (Python-level, per-token) decode cost once per
    corpus rather than once per witness."""
    key = (data_dir, corpus_name, split, n_source_tokens)
    if key not in _TEXT_CACHE:
        train, val, meta, train_offs, val_offs = load_corpus(data_dir, corpus_name, "cpu")
        tokens, offs = (train, train_offs) if split == "train" else (val, val_offs)
        _TEXT_CACHE[key] = decode_corpus_to_documents(tokens, offs, gpt2_tokenizer, n_source_tokens)
    return _TEXT_CACHE[key]


# NO ENCODE-SIDE CACHE. D5 round-4 SERIOUS-1: round 3 added a `_RETOK_CACHE` here, reasoning
# that a --gate run otherwise re-tokenizes the full train splits FOUR times. That reasoning was
# FALSE and the cache was actively harmful. `mode_gate` visits each (witness, corpus) pair
# EXACTLY ONCE, and the four re-tokenizations -- (W1, C1) x (openr1, wikitext) -- are four
# DISTINCT required outputs (different tokenizers, different corpora), so the cache had a
# measured 0% hit rate: 8 misses, 0 hits. What it DID do was hold every re-tokenized train split
# alive for the whole run: a Python list[int] at vocab scale costs ~36 bytes/token, and the four
# splits (344,736,296 + 418,108,652 tokens, x2 witnesses) total **~55 GB of host RAM, never
# evicted** -- turning a "slow but survivable" wall-clock cost into a host-OOM kill mid-gate,
# AFTER the 7B falcon-mamba load has already been paid for. Same symptom the cache was meant to
# prevent, worse cause. The DECODE side (_TEXT_CACHE) is witness-independent and is where the
# real de-duplication already happens; the encode side simply has nothing to reuse. Progress is
# logged instead, so a long re-tokenization is legible rather than silent.


def _retokenize_documents(docs: list, ref_tokenizer, eos_id: int) -> list:
    """Re-tokenizes EACH document SEPARATELY with `ref_tokenizer` and
    splices `eos_id` between them (and after the last one) -- D5 round-1
    SERIOUS-3's actual fix. Returns a flat list[int] of token ids.

    `add_special_tokens=False` (D5 round-2 M-2): without it, a Llama-family
    tokenizer (W3) injects a `<|begin_of_text|>` per document and some
    tokenizers auto-append an EOS -- either would double or corrupt the
    boundary this function exists to create. We then ASSERT the tokenizer did
    not smuggle an `eos_id` into a document's own encoding, so the spliced
    boundary count is exactly `len(docs)` by construction rather than by
    hope."""
    ids = []
    for doc_text in docs:
        if not doc_text:
            continue
        try:
            enc = ref_tokenizer(doc_text, return_tensors=None,
                                 add_special_tokens=False)["input_ids"]
        except TypeError:   # a tokenizer that does not accept the kwarg at all
            enc = ref_tokenizer(doc_text, return_tensors=None)["input_ids"]
        if not enc:
            continue
        if eos_id in enc:
            raise RuntimeError(
                f"_retokenize_documents: the reference tokenizer emitted eos_id={eos_id} INSIDE a "
                f"document's own encoding (add_special_tokens=False was requested). The spliced "
                f"document boundary would then be ambiguous with an in-document token, and every "
                f"EOT-exclusion downstream (detect_candidates_and_baseline / "
                f"assign_placebo_positions / draw_exclusive_replacement / build_key_value_pools) "
                f"would silently mean something other than 'document boundary'. D5 round-2 M-2."
            )
        ids.extend(enc)
        ids.append(eos_id)
    return ids


def build_bridged_corpus(data_dir: str, corpus_name: str, ref_tokenizer, gpt2_tokenizer,
                          eos_id: int, vocab_size: int,
                          n_source_tokens_train, n_source_tokens_val, device: str,
                          seq_len: int = SEQ_LEN_DEFAULT) -> dict:
    """Re-tokenizes OUR train split and OUR val split SEPARATELY (never
    decode-then-split the concatenated stream) with `ref_tokenizer`,
    PER-DOCUMENT with `eos_id` spliced between documents (SERIOUS-3),
    returning int64 tensors on `device` in the reference vocab's id space.
    Hard-asserts `n_tokens_train >= MIN_BRIDGED_TRAIN_TOKENS` (FATAL-1) and
    that every re-tokenized id is `< vocab_size` (M-3 -- a tokenizer/config
    vocab-size mismatch would otherwise surface as an opaque device-side
    assert deep inside a later `scatter_reduce_`/`bincount` call instead of
    a readable message here). Returns {'train': Tensor, 'val': Tensor,
    'n_tokens_train':, 'n_tokens_val':, 'n_docs_train':, 'n_docs_val':}."""
    train_docs = _cached_corpus_documents(data_dir, corpus_name, "train", gpt2_tokenizer,
                                           n_source_tokens_train)
    val_docs = _cached_corpus_documents(data_dir, corpus_name, "val", gpt2_tokenizer,
                                         n_source_tokens_val)
    # No encode-side cache (see the block above _retokenize_documents for why round 3's was
    # deleted). Progress IS logged: a full-split re-tokenization through a non-fast trie
    # tokenizer takes a long time, and a silent long wait on an attested --gate run is
    # indistinguishable from a hang.
    repo_key = getattr(ref_tokenizer, "name_or_path", None) or repr(type(ref_tokenizer))
    t0 = time.time()
    train_enc = _retokenize_documents(train_docs, ref_tokenizer, eos_id)
    print(f"  [bridge] re-tokenized {corpus_name}/train for {repo_key}: {len(train_docs):,} docs "
          f"-> {len(train_enc):,} tokens ({time.time() - t0:.1f}s)", flush=True)
    t0 = time.time()
    val_enc = _retokenize_documents(val_docs, ref_tokenizer, eos_id)
    print(f"  [bridge] re-tokenized {corpus_name}/val for {repo_key}: {len(val_docs):,} docs "
          f"-> {len(val_enc):,} tokens ({time.time() - t0:.1f}s)", flush=True)

    if len(train_enc) < MIN_BRIDGED_TRAIN_TOKENS:
        raise RuntimeError(
            f"build_bridged_corpus({corpus_name!r}): re-tokenized TRAIN split has only "
            f"{len(train_enc):,} tokens, below MIN_BRIDGED_TRAIN_TOKENS="
            f"{MIN_BRIDGED_TRAIN_TOKENS:,}.\n"
            f"D5 round-2 SERIOUS-1 (the BINDING constraint is the VALUE side, which round 1's own "
            f"fix missed): build_key_value_pools requires count(b) in [V2_MIN_COUNT_B="
            f"{V2_MIN_COUNT_B}, V5_MAX_P_TRAIN_B*N={V5_MAX_P_TRAIN_B}*N]. V5 is NOT on any "
            f"relaxation ladder (only K2 is), so this band cannot widen under pressure. At "
            f"N={len(train_enc):,} the band is [{V2_MIN_COUNT_B}, {V5_MAX_P_TRAIN_B * len(train_enc):.0f}]"
            f" -- width {'ZERO or INVERTED' if V5_MAX_P_TRAIN_B * len(train_enc) <= V2_MIN_COUNT_B else 'too narrow'}"
            f", so |P_val(a)| < POOL_FLOOR_P_VAL={POOL_FLOOR_P_VAL} for essentially every key, K5 "
            f"drops every key, P_key=[] and the cell VOIDs before a single plant is sampled -- "
            f"REGARDLESS of the witness's actual capability.\n"
            f"DO NOT 'fix' this by setting --n-source-tokens-train to exactly this floor. Pass "
            f"None (the default) and use the FULL split (~43.7M tokens), which is what the pool "
            f"bands were calibrated on (sec 11.2)."
        )
    # NOTE: there is deliberately NO val-size RAISE here. Round 2 added one and it was a FATAL --
    # both admissible corpora's val splits (2,300,595 / 247,349 tokens, measured) sit BELOW any
    # such floor, so it VOIDed the REQUIRED W1 witness on the instrument's own plumbing. The val
    # split's coverage is MEASURED and REPORTED instead, symmetrically for every witness, via
    # corpus_coverage_provenance(). See that function's own comment block for the full record.
    train_ids = torch.tensor(train_enc, dtype=torch.int64, device=device)
    val_ids = torch.tensor(val_enc, dtype=torch.int64, device=device)
    # D5 round-2 M-1: check BOTH splits, not just train. The tokens actually fed to the model
    # come from VAL (run_did_eval / run_t2_repaired_probe sample from val_tokens) -- an
    # out-of-range val id is exactly the opaque device-side embedding assert this check exists
    # to pre-empt, and round 1's version checked only the split that never reaches the model.
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        if ids.numel() and int(ids.max()) >= vocab_size:
            raise RuntimeError(
                f"build_bridged_corpus({corpus_name!r}): re-tokenized {split_name} id "
                f"{int(ids.max())} >= vocab_size={vocab_size} -- tokenizer/config vocab-size "
                f"mismatch. Refusing rather than letting this surface as an opaque device-side "
                f"assert inside a later scatter_reduce_/bincount/embedding call."
            )
    # D5 round-2 SERIOUS-8: the spliced eos_id is a REFERENCE-vocab token whose in-band
    # frequency in that witness's OWN pretraining we cannot verify from here. Record its
    # realized density so a boundary-token artifact is MEASURABLE in the output JSON rather
    # than latent (sec 9.6's stop rule: it cannot be added after the read). Zero extra forwards.
    n_eos_val = int((val_ids == eos_id).sum())
    return {
        "train": train_ids, "val": val_ids,
        "n_tokens_train": train_ids.numel(), "n_tokens_val": val_ids.numel(),
        "n_docs_train": len(train_docs), "n_docs_val": len(val_docs),
        "eos_id": eos_id,
        "eos_frac_val": (n_eos_val / val_ids.numel()) if val_ids.numel() else None,
        # the fraction of sampled windows expected to CONTAIN >=1 spliced boundary, which is the
        # quantity that matters for "is this witness being read on text it has never seen"
        # D5 round-3 M-4: the true P(a random seq_len-window contains >=1 spliced eos) is
        # 1-(1-p)^T, NOT p*T clipped at 1 (which is an expected COUNT wearing a fraction's name).
        # seq_len is a parameter, not the module default.
        "expected_frac_windows_with_eos": (
            1.0 - (1.0 - n_eos_val / max(1, val_ids.numel())) ** seq_len),
    }


# =============================================================================
# Witness model / corpus loaders.
# =============================================================================
def load_witness_model(witness_key: str, device: str, dtype=torch.bfloat16):
    """Loads witness `witness_key`'s HF model+tokenizer (offline, from
    /data/hf_cache), wraps the model in HFLogitsWrapper. `dtype` defaults to
    bfloat16 (D5 round-1 SERIOUS-8: RWKV7 and falcon-mamba are both
    bf16-native releases; fp16 risks silent overflow -> non-finite logits ->
    HFLogitsWrapper now raises on that, but the right fix is not emitting
    them in the first place). NOT called by this build session against a
    real reference model -- exercised in `smoke()` against a tiny, freshly-
    constructed (no `from_pretrained`, no network, no cache dependency) real
    `transformers.GPT2LMHeadModel` instead."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = WITNESS_SPECS[witness_key]
    repo = spec["hf_repo"]
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code=True, dtype=dtype).to(device)
    hf_model.eval()
    # D5 round-2 SERIOUS-3: the `dtype=`/`torch_dtype=` kwarg is absorbed into **kwargs by
    # from_pretrained, so a WRONG name is SILENTLY IGNORED and the model loads fp32 (masking the
    # memory contract) with no error. The round-2 auditor recommended switching to `torch_dtype=`
    # -- that is BACKWARDS for this box's pinned env, VERIFIED EMPIRICALLY 2026-07-12: on
    # transformers 5.12.1, `torch_dtype=` emits "`torch_dtype` is deprecated! Use `dtype`
    # instead!" and `dtype=` is honored (measured: requested bfloat16 -> realized torch.bfloat16
    # on gpt2-large). So the kwarg NAME stays `dtype=`, and the durable fix is to ASSERT THE
    # REALIZED dtype rather than trust either name -- which catches the failure under ANY
    # transformers version, including a future rename.
    realized = next(hf_model.parameters()).dtype
    if realized != dtype:
        raise RuntimeError(
            f"{witness_key} ({repo}): requested dtype={dtype} but the model loaded as {realized} "
            f"-- the dtype kwarg was silently ignored (it is absorbed into **kwargs by "
            f"from_pretrained, so a wrong/renamed kwarg name never errors). Refusing to proceed "
            f"on an unverified precision contract. D5 round-2 SERIOUS-3."
        )
    wrapped = HFLogitsWrapper(hf_model).to(device)
    wrapped.eval()
    vocab_size = getattr(hf_model.config, "vocab_size", None) or len(tok)
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError(
            f"{witness_key} ({repo})'s tokenizer has no eos_token_id -- the D6 fix "
            f"(eot_override / build_key_value_pools's eot_token_id=) requires one. "
            f"Refusing to fall back to GPT-2's 50256 under a foreign tokenizer -- "
            f"that IS D6, the exact defect this driver exists to fix."
        )
    return wrapped, tok, int(vocab_size), int(eos_id)


def load_witness_corpus(witness_key: str, corpus_name: str, ref_tokenizer, gpt2_tokenizer,
                         eos_id: int, vocab_size: int, data_dir: str,
                         n_source_tokens_train, n_source_tokens_val, device: str,
                         seq_len: int = SEQ_LEN_DEFAULT) -> dict:
    """Returns {'train':, 'val':, 'bridged': bool, ...provenance}. W2
    (bridge=False) loads OUR corpus directly -- see module docstring's "THE
    BRIDGE" for why this is not a special case worth apologising for, it is
    the load-bearing fact sec 11.4.2's witness table is built around."""
    spec = WITNESS_SPECS[witness_key]
    if not spec["bridge"]:
        train, val, meta, _, _ = load_corpus(data_dir, corpus_name, device)
        return {"train": train, "val": val, "bridged": False,
                "n_tokens_train": train.numel(), "n_tokens_val": val.numel()}
    # D5 round-4 M-1: seq_len is THREADED THROUGH (round 3 added the parameter to
    # build_bridged_corpus but never passed it, so expected_frac_windows_with_eos was computed at
    # the module-default 512 while corpus_coverage used the real --seq-len -- two provenance
    # fields in the same cell disagreeing about seq_len).
    bridged = build_bridged_corpus(data_dir, corpus_name, ref_tokenizer, gpt2_tokenizer,
                                    eos_id, vocab_size, n_source_tokens_train,
                                    n_source_tokens_val, device, seq_len=seq_len)
    bridged["bridged"] = True
    return bridged


# =============================================================================
# sec 11.4.3 step 2's stratified acc_copy report -- ALWAYS computed (no extra
# forward passes; pure aggregation of fields run_t2_repaired_probe already
# emits per record), reported ALWAYS, gates NOTHING itself.
# =============================================================================
def stratified_acc_copy_report(records: list) -> dict:
    """sec 11.4.3 step 2's pre-registered diagnostic (build gate item 5,
    D5 round-1 SERIOUS-10): acc_copy stratified by rival strength
    (`max_rival_p` in [0,0.1)/[0.1,0.25)/[0.25,0.5)/[0.5,inf)), by
    `rank_b_given_a` (2-5/6-20/21-50/51+), and by `count_ab`
    (5-9/10-24/25-99/100+ -- V3's floor is count>=5, so bands start there).
    Delta-decile stratification is already emitted by `check_t2a1_ceiling`
    (`decile_accs`) and is NOT duplicated here."""
    def _bucket(key, bands):
        out = {}
        for lo, hi, label in bands:
            sub = [r for r in records if r.get(key) is not None and lo <= r[key] < hi]
            out[label] = {"n": len(sub),
                          "acc_copy": (sum(r["hit_intact"] for r in sub) / len(sub)) if sub else None}
        return out
    return {
        # sec 11.4.3 step 3's own three bands. There is deliberately NO [0.5, inf) bucket:
        # K4 caps max_rival_p <= 0.5 (probe L1491), so such a bucket could only ever hold
        # exactly-0.5 records -- round 1 carried a vacuous fourth band (D5 round-2 M-9). The
        # upper band's closing bound is inclusive of the K4 cap.
        "by_rival_strength_max_rival_p": _bucket(
            "max_rival_p", [(0, 0.1, "[0,0.1)"), (0.1, 0.25, "[0.1,0.25)"),
                            (0.25, 0.5 + 1e-12, "[0.25,0.5]")]),
        "by_rank_b_given_a": _bucket(
            "rank_b_given_a", [(2, 6, "2-5"), (6, 21, "6-20"), (21, 51, "21-50"),
                               (51, float("inf"), "51+")]),
        "by_count_ab": _bucket(
            "count_ab", [(5, 10, "5-9"), (10, 25, "10-24"), (25, 100, "25-99"),
                         (100, float("inf"), "100+")]),
    }


# =============================================================================
# sec 11.4.3 step 2's "W2 Delta-sweep" -- fixed, single-value Delta pools so
# EVERY plant in a sweep point lands at exactly that distance.
# =============================================================================
def run_delta_sweep(model, corpus_name: str, val_tokens: torch.Tensor,
                     pools: dict, counts_by_token: list, eos_id: int, vocab_size: int,
                     device: str, seq_len: int, natural_delta_pool: list,
                     n_plants_per_delta: int = N_PLANTS_PER_DELTA_DEFAULT,
                     delta_targets: tuple = DELTA_SWEEP_TARGETS,
                     eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT) -> dict:
    """sec 11.4.3 step 2's W2 Delta-sweep (build gate item 5). Runs T2 at
    FIXED, single-value Delta 'pools' -- every plant at that sweep point
    lands at EXACTLY that Delta -- at Delta in {2,5,10,20,40,median,200,400},
    reporting acc_copy(Delta). Localises whether a T2a-1 failure is a
    DISTANCE limit (accuracy falls off with Delta) or uniform (probe/
    mechanism defect, sec 11.4.3 step 3).

    D5 round-2 SERIOUS-4: `n_plants_per_delta` defaults to
    N_PLANTS_PER_DELTA_DEFAULT (256), NOT the round-1 value of 64.
    `run_t2_repaired_probe` VOIDs a cell whose plant-construction drop rate
    exceeds T2_DROP_CAP_FRAC (2%, PINNED by sec 11.2.3 point 6 -- and NOT
    something this driver may loosen). At n_plants=64 that cap is a hair
    trigger: TWO dropped windows (2/64 = 3.1%) VOID the sweep point. At 256
    it buys 5 drops of headroom, and sec 9.6's stop rule means a VOIDed
    diagnostic CANNOT be re-run after the T2a read -- losing it would forfeit
    exactly the pre-registered distance-vs-uniform localisation sec 11.4.3
    step 3 depends on. (D5 round-2 M-7: the unused `witness_key` parameter
    round 1 carried here is REMOVED rather than left as dead surface.)"""
    median_delta = sorted(natural_delta_pool)[len(natural_delta_pool) // 2] if natural_delta_pool else None
    targets = list(delta_targets)
    if median_delta is not None and median_delta not in targets:
        targets.append(median_delta)
    out = {}
    for d in targets:
        if d < 2 or d > seq_len - 6:
            out[str(d)] = {"skipped": True, "reason": f"Delta={d} outside [2, {seq_len - 6}]"}
            continue
        with eot_override(eos_id):
            t2_result = run_t2_repaired_probe(model, val_tokens, seq_len, device, corpus_name,
                                               delta_pool=[d], pools=pools,
                                               counts_by_token=counts_by_token,
                                               n_plants=n_plants_per_delta, vocab_size=vocab_size,
                                               eval_micro_batch=eval_micro_batch)
        out[str(d)] = {"void": bool(t2_result.get("void")), "void_reason": t2_result.get("void_reason"),
                        "acc_copy": t2_result.get("acc_copy"), "n_plants": t2_result.get("n_plants"),
                        "is_median": (d == median_delta)}
    return out


# =============================================================================
# sec 11.4.3 step 2's n_demos in {1,2,4} diagnostic -- "the only diagnostic
# that separates 'one-shot is too hard' from 'the model cannot copy'."
# NEW, driver-local plant construction (NOT reused from the already-audited
# lm_recall_gap_probe_v2_rd.py -- flagged for its own fresh scrutiny).
# =============================================================================
def plant_and_verify_t2_window_ndemos(orig_window: list, positions: list, a: int, b: int) -> list:
    """NEW, driver-local (explicitly NOT part of the audited probe module):
    generalizes `plant_and_verify_t2_window` (sec 11.2.3, ONE demonstration)
    to `len(positions) - 1` repeated (a,b) demonstrations before a final,
    UNdemonstrated query occurrence of `a` (`positions[-1]`) whose
    next-token target is `b`. Writes `w[p]=a` at every position in
    `positions`; `w[p+1]=b` at every DEMO position (all but the last).
    HARD-ASSERTS exact occurrence counts -- never a tolerance -- mirroring
    `plant_and_verify_t2_window`'s own discipline: a structural check
    without a forced-fail negative test is not a check (see `smoke()`
    [7c])."""
    w = list(orig_window)
    demo_positions = positions[:-1]   # positions[-1] is the QUERY -- deliberately undemonstrated
    for p in positions:
        w[p] = a
    for p in demo_positions:
        w[p + 1] = b
    a_positions = [i for i, tok in enumerate(w) if tok == a]
    b_positions = [i for i, tok in enumerate(w) if tok == b]
    expected_a = sorted(set(positions))
    expected_b = sorted({p + 1 for p in demo_positions})
    if a_positions != expected_a or b_positions != expected_b:
        raise PlantContestedError(
            f"N_DEMOS PLANT CONTESTED: expected count(a={a})=={len(expected_a)} at exactly "
            f"{expected_a} AND count(b={b})=={len(expected_b)} at exactly {expected_b} "
            f"(got a@{a_positions}, b@{b_positions}). Dropped, never scored -- same discipline "
            f"as plant_and_verify_t2_window (PARAM_AXIS_SCALING_DESIGN.md sec 11.2.3)."
        )
    return w


def run_n_demos_diagnostic(model, witness_key: str, corpus_name: str, val_tokens: torch.Tensor,
                            pools: dict, eos_id: int, device: str,
                            seq_len: int = SEQ_LEN_DEFAULT, n_demos_values: tuple = N_DEMOS_VALUES,
                            n_plants_per_level: int = N_PLANTS_PER_NDEMOS_DEFAULT, gap: int = 40,
                            eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT,
                            drop_cap_frac: float = T2_DROP_CAP_FRAC,
                            min_surviving: int = NDEMOS_MIN_SURVIVING_PLANTS) -> dict:
    """sec 11.4.3 step 2's n_demos diagnostic -- "the only diagnostic that
    separates 'one-shot is too hard' from 'the model cannot copy'". Runs the
    INTACT arm ONLY (no ablation arms -- this reads acc_copy(n_demos), not a
    causal gap) at n_demos in {1,2,4}, reusing `pools["P_key"]`/
    `pools["P_val"]` (pool construction unchanged; only the plant is new).

    D5 ROUND-2 SERIOUS-5 FIX -- THE LEVELS ARE **PAIRED**. Round 1 put
    `n_demos` INTO the seed, so levels 1/2/4 drew DIFFERENT windows AND
    DIFFERENT (a,b) pairs -- and the entire point of the read is the
    WITHIN-ITEM contrast acc_copy(1) -> acc_copy(2) -> acc_copy(4). Any
    observed difference would have confounded the number of demonstrations
    with the window draw and the plant difficulty, at n=64/level. The seed is
    now `(corpus, witness, "ndemos")` ONLY, so every level sees the SAME
    windows and the SAME (a,b) pairs; and the position construction is
    already nested (the LAST demo sits at `query_pos - gap` and the query at
    `seq_len - 8` for EVERY level), so pairing is not merely desirable, it is
    free.

    D5 ROUND-2 SERIOUS-B/coordinator FIX -- THE DROP CAP IS ENFORCED HERE.
    This function hand-rolls its outer plant loop (it cannot reuse
    `run_t2_repaired_probe`, which plants exactly ONE demonstration), so it
    does NOT inherit that function's `T2_DROP_CAP_FRAC` VOID for free. Round
    1 VOIDed only on LITERALLY ZERO survivors -- so a level where 60 of 64
    plants dropped would have silently reported an `acc_copy` off 4 windows,
    unflagged, into a diagnostic sec 11.4.3 makes load-bearing for
    interpreting a T2a-1 failure. It now enforces the SAME pinned 2% cap the
    audited probe enforces, PLUS an absolute `min_surviving` floor."""
    p_key = pools.get("P_key", [])
    p_val = pools.get("P_val", {})
    out = {}
    # ONE seed for the whole diagnostic -- NOT per level. This is the pairing fix.
    seed = _combine_seed(corpus_name, witness_key, "ndemos")
    for n_demos in n_demos_values:
        query_pos = seq_len - 8
        if query_pos - gap * n_demos < 0 or not p_key:
            out[str(n_demos)] = {"skipped": True,
                                  "reason": f"seq_len={seq_len} too small for n_demos={n_demos} at "
                                            f"gap={gap}, or empty P_key"}
            continue
        positions = [query_pos - gap * (n_demos - i) for i in range(n_demos + 1)]
        gen = torch.Generator(device=device).manual_seed(seed)          # same windows every level
        x0, y0 = get_batch(val_tokens, n_plants_per_level, seq_len, gen)
        window0 = _make_window(x0, y0)
        orig_windows = [window0[b].cpu().tolist() for b in range(window0.shape[0])]
        rng = random.Random(seed)                                        # same (a,b) draws every level
        planted, n_dropped, drop_reasons = [], 0, {}
        for w_orig in orig_windows:
            w_set = set(w_orig)
            # RETRY the (a,b) draw, exactly as the AUDITED draw_t2_triple does (probe L1611-1656,
            # "Up to 100 tries", sec 11.2.3 point 3) -- do NOT drop the window on the first
            # collision. This is not an optimization, it is a CORRECTNESS requirement the first
            # draft of this function missed and its own drop-cap safeguard then caught: the pool
            # keys are rare-but-not-absent (K2: p_train(a) <= 4e-4), so in a 512-token window
            # P(a occurs naturally) ~ 1 - (1-4e-4)^512 ~ 18%. Drawing ONCE and dropping on
            # collision produces a structural drop rate FAR above the pinned 2% cap, VOIDing the
            # diagnostic on every witness regardless of capability -- the same "the guard VOIDs
            # the thing it guards" shape as the round-1 FATAL. Retrying reduces the drop rate to
            # the residual probability that ALL `max_tries` draws collide.
            a = b = None
            for _try in range(T2_TRIPLE_MAX_TRIES):
                a_c = rng.choice(p_key)
                if a_c in w_set:
                    continue
                vals = p_val.get(a_c)
                if not vals:
                    continue
                b_c = rng.choice(vals)
                if b_c in w_set or b_c == a_c:
                    continue
                a, b = a_c, b_c
                break
            if a is None:
                n_dropped += 1
                drop_reasons["no_uncontested_pair_in_max_tries"] = \
                    drop_reasons.get("no_uncontested_pair_in_max_tries", 0) + 1
                continue
            try:
                planted.append((plant_and_verify_t2_window_ndemos(w_orig, positions, a, b), b))
            except PlantContestedError:
                n_dropped += 1
                drop_reasons["plant_contested"] = drop_reasons.get("plant_contested", 0) + 1

        drop_frac = n_dropped / n_plants_per_level if n_plants_per_level else 1.0
        base = {"n_dropped": n_dropped, "drop_frac": drop_frac, "drop_reasons": drop_reasons,
                "n_plants_requested": n_plants_per_level, "n_surviving": len(planted),
                "paired_seed": seed}
        if drop_frac > drop_cap_frac:
            out[str(n_demos)] = dict(base, void=True, void_reason=(
                f"plant construction dropped {n_dropped}/{n_plants_per_level} windows "
                f"({drop_frac:.4f} > {drop_cap_frac} cap, sec 11.2.3 point 6 -- the SAME pinned cap "
                f"run_t2_repaired_probe enforces; this hand-rolled loop does not inherit it for "
                f"free, D5 round-2)."))
            continue
        if len(planted) < min_surviving:
            out[str(n_demos)] = dict(base, void=True, void_reason=(
                f"only {len(planted)} plants survived construction, below the absolute floor "
                f"min_surviving={min_surviving} -- an acc_copy off this few windows is not "
                f"reportable into a diagnostic sec 11.4.3 makes load-bearing (D5 round-2)."))
            continue

        batch = torch.stack([torch.tensor(w[:-1], device=device, dtype=torch.int64) for w, _ in planted])
        hits = []
        with eot_override(eos_id):
            for start in range(0, batch.shape[0], eval_micro_batch):
                with torch.no_grad():
                    logits = model(batch[start:start + eval_micro_batch])
                preds = logits.argmax(dim=-1)
                for i in range(preds.shape[0]):
                    _, b_target = planted[start + i]
                    # `batch` rows are w[:-1] (length seq_len); preds[i, query_pos] is the model's
                    # prediction FOR window index query_pos+1 -- i.e. for the token that follows
                    # the query occurrence of `a` at query_pos. The plant deliberately does NOT
                    # write b at query_pos+1 (the query is UNdemonstrated), so comparing to
                    # b_target is the correct read, and matches run_t2_repaired_probe's own
                    # `pred_intact[row_idx, k0] == b` convention exactly.
                    hits.append(int(preds[i, query_pos].item() == b_target))
        out[str(n_demos)] = dict(base, void=False, acc_copy=sum(hits) / len(hits), n_plants=len(hits))
    return out


# =============================================================================
# Per-(witness, corpus) cell: T1c's DiD eval + T2's repaired probe, sharing
# ONE Delta pool. See module docstring "DELTA POOL PROVENANCE."
# =============================================================================
def run_witness_cell(model, witness_key: str, corpus_name: str, train_tokens: torch.Tensor,
                      val_tokens: torch.Tensor, eos_id: int, vocab_size: int, device: str,
                      seq_len: int = SEQ_LEN_DEFAULT, n_windows: int = N_ROWS_DEFAULT,
                      n_plants: int = N_PLANTS_DEFAULT,
                      eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT,
                      batch_size_provenance: int = 32) -> dict:
    """Runs T1c's main-metric DiD cell, THEN harvests its own candidates'
    Delta pool, THEN runs T2's repaired probe + T2b-1/T2b-1b/T2a-1(/T2a-3 if
    this witness is C1) + the sec 11.4.3 stratified report -- ONE model load
    covers all of it. Every call into the reused module that can read EOT is
    wrapped in `eot_override`; `build_key_value_pools` additionally receives
    `eot_token_id=` explicitly (D6 fix). D5 round-1 SERIOUS-7 fix: computes
    `t1c_admissibility` explicitly (`cell_void_placebo_match`,
    `cell_void_missing_s2_fields`, the sec 11.7 row/candidate floors) rather
    than leaving `check_t1c_reference_did` to trust an ungated cell. M-5:
    `VerdictGradeError` from `finalize_cell` is caught and reported as a
    VOID cell rather than propagating and killing a multi-witness run."""
    mode_next = build_bigram_mode_table(train_tokens, vocab_size, device)
    counts_by_token = torch.bincount(train_tokens, minlength=vocab_size).tolist()
    seed = corpus_fixed_seed(corpus_name) + RUNG_MATCHING_SEED_OFFSET

    with eot_override(eos_id):
        did_result = run_did_eval(model, val_tokens, batch_size_provenance, seq_len, n_windows,
                                   device, mode_next, seed, eval_micro_batch=eval_micro_batch,
                                   vocab_size=vocab_size)
    try:
        cell = finalize_cell(witness_key, corpus_name, did_result)
    except VerdictGradeError as e:
        return {"witness": witness_key, "corpus": corpus_name, "t2_void": True,
                "t2_void_reason": f"T1c cell VerdictGradeError, T2 not attempted: {e}",
                "t1c_admissibility": {"admissible": False, "reason": str(e)}}

    resolved = [r for r in did_result["records"] if r["hit_placebo_ablated"] is not None]
    n_contributing_rows = len({r["row_idx"] for r in resolved})
    t1c_admissible = (
        not cell.get("cell_void_placebo_match", True)
        and not cell.get("cell_void_missing_s2_fields", True)
        and cell.get("n_candidates_resolved", 0) >= CANDIDATE_FLOOR_RESOLVED
        and n_contributing_rows >= ROW_FLOOR_CONTRIBUTING
    )
    t1c_admissibility = {
        "admissible": t1c_admissible,
        "cell_void_placebo_match": cell.get("cell_void_placebo_match"),
        "cell_void_missing_s2_fields": cell.get("cell_void_missing_s2_fields"),
        "n_candidates_resolved": cell.get("n_candidates_resolved"),
        "candidate_floor_resolved": CANDIDATE_FLOOR_RESOLVED,
        "n_contributing_rows": n_contributing_rows,
        "row_floor_contributing": ROW_FLOOR_CONTRIBUTING,
    }
    delta_pool = [r["delta"] for r in did_result["records"]]

    out = {"witness": witness_key, "corpus": corpus_name, "cell": cell,
           "t1c_admissibility": t1c_admissibility, "delta_pool_n": len(delta_pool),
           "delta_pool": delta_pool,   # retained transiently for run_delta_sweep's median-Delta
                                        # calc; mode_gate strips it (with "pools") before --out.
           "n_candidates_total": len(did_result["records"])}
    if not delta_pool:
        out["t2_void"] = True
        out["t2_void_reason"] = "empty Delta pool from run_did_eval -- T1c's own candidate detection found nothing"
        return out

    with eot_override(eos_id):
        pools = build_key_value_pools(train_tokens, vocab_size, device, eot_token_id=eos_id)
    out["pools_summary"] = {k: pools[k] for k in
                             ("k2_band_used", "floor_pool_key", "floor_pool_val", "floor_licensed_b",
                              "n_p_key", "median_p_val", "n_licensed_b_ge2")}
    out["pools"] = pools   # retained on the returned dict for run_delta_sweep/run_n_demos_diagnostic
                            # to reuse without recomputation; NOT written verbatim to --out (see
                            # mode_gate, which strips it before json.dump -- pools are large).
    if not (pools["floor_pool_key"] and pools["floor_pool_val"] and pools["floor_licensed_b"]):
        out["t2_void"] = True
        out["t2_void_reason"] = f"sec 11.2 pool floor(s) missed: {out['pools_summary']}"
        return out

    with eot_override(eos_id):
        t2_result = run_t2_repaired_probe(model, val_tokens, seq_len, device, corpus_name, delta_pool,
                                           pools, counts_by_token, n_plants, vocab_size=vocab_size,
                                           eval_micro_batch=eval_micro_batch)
    out["t2"] = {k: v for k, v in t2_result.items() if k != "records"}
    out["t2_void"] = bool(t2_result.get("void"))
    if out["t2_void"]:
        out["t2_void_reason"] = t2_result.get("void_reason")
        return out

    t2b1 = check_t2b1_mechanism_exists(t2_result["records"])
    t2b1b = check_t2b1b_key_conditioned(t2_result["records"])
    t2a1 = check_t2a1_ceiling(t2_result["records"], t2b1, t2b1b)
    out.update({"t2b1_mechanism_exists": t2b1, "t2b1b_key_conditioned": t2b1b,
                "t2a1_ceiling": t2a1,
                "stratified_acc_copy": stratified_acc_copy_report(t2_result["records"])})
    if WITNESS_SPECS[witness_key]["t2a3"]:
        out["t2a3_ssm_calibration"] = check_t2a3_ssm_calibration(t2_result["records"], t2b1, t2b1b)
    return out


def run_t2a2_untrained_control(corpus_name: str, data_dir: str, device: str,
                                seq_len: int = SEQ_LEN_DEFAULT, n_windows: int = N_ROWS_DEFAULT,
                                n_plants: int = N_PLANTS_DEFAULT,
                                init_seed: int = T2A2_UNTRAINED_INIT_SEED) -> dict:
    """T2a-2 (sec 11.4.2, GATING, NEW): a randomly-initialised, UNTRAINED
    model of the 14M rung's exact architecture (RUNG_14M_CONFIG). Uses OUR
    OWN GPT-2-tokenized corpus DIRECTLY -- no HF model, no bridge, no
    tokenizer mismatch possible; this gate exercises the instrument's
    negative-control property, not any cross-tokenizer machinery. Uses
    RUNG_MATCHING_SEED_OFFSET (D5 SERIOUS-5) so this control's window
    population matches what a real rung would see on the same corpus."""
    train_tokens, val_tokens, meta, _, _ = load_corpus(data_dir, corpus_name, device)
    torch.manual_seed(init_seed)
    model = DeltaNetLM(VOCAB_SIZE_GPT2, **RUNG_14M_CONFIG).to(device)
    model.eval()
    mode_next = build_bigram_mode_table(train_tokens, VOCAB_SIZE_GPT2, device)
    counts_by_token = torch.bincount(train_tokens, minlength=VOCAB_SIZE_GPT2).tolist()
    seed = corpus_fixed_seed(corpus_name) + RUNG_MATCHING_SEED_OFFSET

    did_result = run_did_eval(model, val_tokens, 32, seq_len, n_windows, device, mode_next, seed,
                               vocab_size=VOCAB_SIZE_GPT2)
    delta_pool = [r["delta"] for r in did_result["records"]]
    pools = build_key_value_pools(train_tokens, VOCAB_SIZE_GPT2, device)
    result = {"corpus": corpus_name, "init_seed": init_seed, "rung_config": RUNG_14M_CONFIG,
              "delta_pool_n": len(delta_pool)}
    if not delta_pool or not (pools["floor_pool_key"] and pools["floor_pool_val"] and pools["floor_licensed_b"]):
        result["t2_void"] = True
        result["t2_void_reason"] = "empty Delta pool or sec 11.2 pool floor(s) missed on our own corpus"
        return result
    t2_result = run_t2_repaired_probe(model, val_tokens, seq_len, device, corpus_name, delta_pool,
                                       pools, counts_by_token, n_plants, vocab_size=VOCAB_SIZE_GPT2)
    if t2_result.get("void"):
        result["t2_void"] = True
        result["t2_void_reason"] = t2_result.get("void_reason")
        return result
    result["t2_void"] = False
    result["n_plants"] = t2_result.get("n_plants")
    result["t2a2_untrained_control"] = check_t2a2_untrained_control(t2_result["records"])
    return result


# =============================================================================
# sec 11.7's model-free N_rows pre-pass -- RUN FOR REAL by this build session
# (model-free, CPU-safe, disturbs nothing).
# =============================================================================
def mode_pre_pass(args) -> int:
    """sec 11.11's pinned EXECUTION ORDER step (1)'s model-free half:
    `resolve_n_rows_pre_pass` (sec 11.7) on OUR OWN corpora (GPT-2 vocab, no
    witness, no reference model, no checkpoint) to fix N_rows before any
    model loads. This build session RUNS this for real."""
    corpora = args.corpus or ["openr1-mix-ext", "wikitext-mix-ext"]
    val_by_corpus, mode_next_by_corpus = {}, {}
    for c in corpora:
        train, val, meta, _, _ = load_corpus(args.data_dir, c, args.device)
        val_by_corpus[c] = val
        mode_next_by_corpus[c] = build_bigram_mode_table(train, VOCAB_SIZE_GPT2, args.device)
    result = resolve_n_rows_pre_pass(val_by_corpus, mode_next_by_corpus, args.seq_len, args.device)
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
    if result["void"]:
        raise SystemExit(f"pre-pass VOID: {result['void_reason']}")
    print(f"\npre-pass resolved N_rows={result['n_rows']} on {corpora}.")
    return 0


# =============================================================================
# Full gate orchestration -- WIRED, NOT INVOKED by this build session.
# =============================================================================
def mode_gate(args) -> int:
    """sec 11.11's pinned EXECUTION ORDER step (2): T2a-1/T2a-2/T2a-3/T1c on
    the witnesses, across BOTH corpora, plus the sec 11.4.3 W2 Delta-sweep
    and n_demos diagnostic. REFUSES to run against real downloaded
    reference models unless `--i-am-the-t2a-execution-agent` is passed --
    this build session never passes it. Also REFUSES to run without `--out`
    (D5 SERIOUS-9: `t2a1_ceiling` etc. live inside `results['cells']`, which
    stdout deliberately omits to stay readable -- without `--out` that
    result is computed and then lost) and REFUSES if `--n-plants !=
    --n-windows` (D5 SERIOUS-4: sec 11.7 pins them equal)."""
    if not args.i_am_the_t2a_execution_agent:
        raise SystemExit(
            "REFUSING TO RUN: --i-am-the-t2a-execution-agent is required.\n"
            "This driver was built and smoke-tested by a session explicitly instructed NOT "
            "to execute T2a (CLAUDE.md: 'the implementer does not review or run their own "
            "work'; PARAM_AXIS_SCALING_DESIGN.md sec 11.11's execution order step (1) is "
            "pre-pass+smoke+D5-audit, step (2) is the T2a/T1c read -- a SEPARATE agent, after "
            "the D5 audit clears, performs step (2)). Passing this flag is that agent's own "
            "attestation, recorded in the output JSON as `execution_attested`."
        )
    if not args.out:
        raise SystemExit(
            "REFUSING TO RUN: --out is required for --gate (D5 audit SERIOUS-9). Without it, "
            "t2a1_ceiling / t2b1_mechanism_exists / t2b1b_key_conditioned / "
            "t2a3_ssm_calibration -- everything inside results['cells'] -- is computed, printed "
            "only in a stdout view that OMITS 'cells' to stay readable, and lost the moment this "
            "process exits."
        )
    if args.n_plants != args.n_windows:
        raise SystemExit(
            f"REFUSING TO RUN: --n-plants ({args.n_plants}) != --n-windows ({args.n_windows}). "
            f"PARAM_AXIS_SCALING_DESIGN.md sec 11.7 pins n_plants(T2) = N_rows exactly "
            f"(D5 audit SERIOUS-4)."
        )
    # D5 round-2 SERIOUS-6: the truncation flags exist for smoke/debug ONLY. A truncated bridged
    # corpus (a) can VOID the pools outright (SERIOUS-1's arithmetic), (b) makes the clustered
    # bootstrap anticonservative via window overlap (a FALSE PASS on T1c), and (c) shrinks ONLY
    # the bridged witnesses' corpora while W2 -- which does not go through build_bridged_corpus at
    # all -- always gets the full split, so the two REQUIRED ceiling witnesses would be compared
    # on corpora of different size and pool quality. Refused outright at the gate.
    if args.n_source_tokens_train is not None or args.n_source_tokens_val is not None:
        raise SystemExit(
            f"REFUSING TO RUN: --gate runs on the FULL train/val split. Got "
            f"--n-source-tokens-train={args.n_source_tokens_train}, "
            f"--n-source-tokens-val={args.n_source_tokens_val}.\n"
            f"D5 round-2 SERIOUS-6: truncation VOIDs the sec 11.2 pools outright below "
            f"{MIN_BRIDGED_TRAIN_TOKENS:,} train tokens, WORSENS the val split's already-"
            f">1 window-coverage ratio (making clustered_bootstrap_ci's CI even narrower than "
            f"the pre-registered design's own baseline -- see corpus_coverage_provenance), and "
            f"applies ONLY to the bridged witnesses (W2 never goes through build_bridged_corpus), "
            f"so it would silently compare the two REQUIRED ceiling witnesses on different-sized "
            f"corpora. These flags are for --smoke/debug only."
        )

    corpora = args.corpus or ["openr1-mix-ext", "wikitext-mix-ext"]
    witnesses = args.witness or ["W1_rwkv7", "W2_gpt2large", "C1_falconmamba"]
    # D5 round-3 SERIOUS-1: THE ROLL-UP FAILS OPEN UNDER SUBSETTING, so subsetting is REFUSED.
    # `gate["t2a3"]` was CONDITIONALLY created (`if "C1_falconmamba" in witnesses`), so a run
    # with `--witness W1_rwkv7 W2_gpt2large` -- an entirely reasonable thing to do, it skips a 7B
    # model load -- produced an INSTRUMENT_VALID conjunction that NEVER SAW the T2a-3 leg at all.
    # Same hole on the corpus axis: every leg is `all(... for c in corpora)`, so `--corpus X`
    # alone yields a conjunction over one corpus while sec 11.4.2/11.4.5 require BOTH. That is
    # round-1 SERIOUS-7's "computed but never read" defect reborn as "never computed and never
    # missed" -- and it is the FALSE-PASS direction. A subset run simply cannot produce a
    # verdict, so it is refused rather than silently narrowed.
    # D5 round-4 SERIOUS-2 / round-5 SERIOUS-4: the relation is EQUALITY, not subset OR superset.
    # Round 3 used `<` (PROPER subset), which let `--witness W1 W2 W3` and `--corpus openr1
    # wikitext` (all valid argparse choices, none of them the pinned sets) slip through. Round 4
    # changed it to `>=` (superset) -- which fixed those, but admitted a SUPERSET, and a superset
    # is ALSO wrong and in the HALT-BY-DEFECT direction: `t2a1_gate` iterates the operator-supplied
    # `corpora` while t2a2/t2a3/t1c iterate the pinned REQUIRED_CORPORA, so an EXTRA corpus becomes
    # a GATING T2a-1 leg -- and if that extra corpus voids, INSTRUMENT_VALID reads False even
    # though both pinned corpora passed. Equality is the only relation that is right in both
    # directions.
    # D5 round-6 M-3: DUPLICATES are refused too. `set()` collapses them, so `--corpus X X Y`
    # would pass an equality check and then run the corpus loop TWICE per witness -- a second full
    # bridged re-tokenization of a 43.7M-token split through RWKV-World's non-fast trie tokenizer
    # (hours) plus a duplicate falcon-mamba-7B cell. The verdict would be unaffected, but the cost
    # is not.
    if len(set(witnesses)) != len(witnesses) or len(set(corpora)) != len(corpora):
        raise SystemExit(
            f"REFUSING TO RUN: duplicate entries in --witness {witnesses} / --corpus {corpora}. "
            f"Each (witness, corpus) cell is evaluated once; a duplicate silently doubles a "
            f"multi-hour bridged re-tokenization and a 7B model's eval (D5 round-6 M-3)."
        )
    if set(witnesses) != set(REQUIRED_WITNESSES) or set(corpora) != set(REQUIRED_CORPORA):
        raise SystemExit(
            f"REFUSING TO RUN: --gate must run the FULL witness set and BOTH corpora.\n"
            f"Got witnesses={sorted(witnesses)}, corpora={sorted(corpora)}.\n"
            f"Required: witnesses={sorted(REQUIRED_WITNESSES)}, corpora={sorted(REQUIRED_CORPORA)}.\n"
            f"sec 11.4.2/11.4.5 make T2a-1 (W1 AND W2) AND T2a-2 AND T2a-3 (C1) AND T1c ALL "
            f"gating, on EACH corpus. A subset run cannot produce an INSTRUMENT_VALID verdict -- "
            f"and silently narrowing the conjunction to whatever was run is a FALSE PASS "
            f"(D5 round-3 SERIOUS-1)."
        )
    # D5 round-2 M-10: the gate JSON must be self-describing enough to reproduce.
    results = {
        "execution_attested": True, "corpora": corpora, "witnesses": witnesses,
        "config": {"seq_len": args.seq_len, "n_windows": args.n_windows,
                    "n_plants": args.n_plants, "eval_micro_batch": args.eval_micro_batch,
                    "device": args.device, "data_dir": args.data_dir,
                    "min_bridged_train_tokens": MIN_BRIDGED_TRAIN_TOKENS,
                    # D5 round-4 FATAL-1: there is NO val-size gate, and this key must NOT
                    # advertise one. The round-3 fix deleted the constant and left this reference
                    # behind, so EVERY --gate run died with a NameError here -- before any model
                    # load, after all five refusal checks. The smoke did not catch it because
                    # mode_gate's BODY had zero coverage (all five mode_gate tests return at a
                    # refusal), which is now fixed by smoke [6h].
                    "val_size_gate": None,
                    "val_coverage_reported_per_cell": True,
                    "seed_offset": RUNG_MATCHING_SEED_OFFSET},
        "commit_sha": _git_sha(),
        "cells": {},
    }

    def _persist():
        """D5 round-2 SERIOUS-A: write --out INCREMENTALLY. Round 1 wrote it ONCE at the very
        end, so ANY uncaught exception mid-loop destroyed every result already computed --
        including from an already-loaded falcon-mamba-7b (a ~7B model load this gate pays for
        once). CLAUDE.md's own pinned rule: 'Add try/except so one crash doesn't kill remaining
        configs.'

        D5 round-3 M-2: write ATOMICALLY (tmp + os.replace). A plain open(...,"w") truncates
        first, so a SIGKILL/host-OOM mid-write would destroy the entire partial file -- which is
        precisely the loss this incremental persistence exists to prevent."""
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, default=str)
        os.replace(tmp, args.out)

    gpt2_tok = None
    for w in witnesses:
        try:
            model, tok, vocab_size, eos_id = load_witness_model(w, args.device)
        except Exception as e:   # noqa: BLE001 -- a witness that will not even LOAD must not kill the gate
            for corpus in corpora:
                results["cells"][f"{w}/{corpus}"] = {
                    "witness": w, "corpus": corpus, "t2_void": True,
                    "t2_void_reason": f"witness failed to LOAD: {type(e).__name__}: {e}"}
            _persist()
            continue
        if WITNESS_SPECS[w]["bridge"] and gpt2_tok is None:
            gpt2_tok = _get_gpt2_tokenizer()   # M-4: loaded ONCE, fast tokenizer, reused across witnesses
        for corpus in corpora:
            # D5 round-2 SERIOUS-A: catch EVERY exception, not just VerdictGradeError. Round 1
            # caught only VerdictGradeError -- but round 2's OWN SERIOUS-8 fix now raises a plain
            # RuntimeError on non-finite logits (a real risk on the bf16 witnesses), and
            # build_bridged_corpus + load_witness_model raise RuntimeErrors of their own. Any of
            # them would have escaped the narrow catch and destroyed the whole run.
            try:
                corpus_data = load_witness_corpus(w, corpus, tok if WITNESS_SPECS[w]["bridge"] else None,
                                                   gpt2_tok, eos_id, vocab_size, args.data_dir,
                                                   args.n_source_tokens_train, args.n_source_tokens_val,
                                                   args.device, seq_len=args.seq_len)
                cell = run_witness_cell(model, w, corpus, corpus_data["train"], corpus_data["val"],
                                         eos_id, vocab_size, args.device, seq_len=args.seq_len,
                                         n_windows=args.n_windows, n_plants=args.n_plants,
                                         eval_micro_batch=args.eval_micro_batch)
                # SERIOUS-8(R2): carry the bridge's own eos-density provenance into the cell, so a
                # boundary-token artifact is MEASURABLE rather than latent (sec 9.6's stop rule
                # forbids adding it after the read).
                if corpus_data.get("bridged"):
                    cell["bridge_provenance"] = {
                        k: corpus_data.get(k) for k in
                        ("eos_id", "eos_frac_val", "expected_frac_windows_with_eos",
                         "n_tokens_train", "n_tokens_val", "n_docs_train", "n_docs_val")}
                # D5 round-3 FATAL-1's replacement for the deleted val-size GATE, and round-3
                # M-5: emitted for EVERY cell, bridged AND unbridged, so the two REQUIRED ceiling
                # witnesses are comparable on this axis instead of only the bridged one carrying
                # it. Reported, never gated -- see corpus_coverage_provenance's own comment.
                cell["corpus_coverage"] = corpus_coverage_provenance(
                    int(corpus_data["val"].numel()), args.n_windows, args.seq_len)
            except Exception as e:   # noqa: BLE001
                cell = {"witness": w, "corpus": corpus, "t2_void": True,
                        "t2_void_reason": f"{type(e).__name__}: {e}"}
                results["cells"][f"{w}/{corpus}"] = cell
                _persist()
                continue
            # D5 round-3 M-3: pull the large transient objects OFF the cell BEFORE it is
            # registered/persisted. Round 2 hung `pools` (P_val + val_meta + inverse_map --
            # thousands of entries) and `delta_pool` on the cell and stripped them only AFTER a
            # _persist(), so every intermediate write serialized megabytes of pool internals and
            # a mid-run crash could leave a bloated partial JSON. They are locals now; nothing
            # downstream needs them on the cell.
            cell_pools = cell.pop("pools", None)
            cell_delta_pool = cell.pop("delta_pool", None)
            results["cells"][f"{w}/{corpus}"] = cell
            _persist()

            # sec 11.4.3's W2 Delta-sweep + n_demos diagnostic -- computed for W2 (the design's
            # own naming: "the W2 Delta-sweep") when it clears T2 at all. natural_delta_pool is
            # THAT SAME cell's own harvested Delta pool (run_witness_cell's "delta_pool" -- the
            # same population T2's plants were already drawn from, see module docstring "DELTA
            # POOL PROVENANCE"), never a placeholder or a foreign source.
            if w == "W2_gpt2large" and not cell.get("t2_void") and cell_pools is not None:
                try:
                    results.setdefault("w2_delta_sweep", {})[corpus] = run_delta_sweep(
                        model, corpus, corpus_data["val"], cell_pools,
                        torch.bincount(corpus_data["train"], minlength=vocab_size).tolist(),
                        eos_id, vocab_size, args.device, args.seq_len,
                        natural_delta_pool=cell_delta_pool or [],
                        eval_micro_batch=args.eval_micro_batch)
                    results.setdefault("w2_n_demos", {})[corpus] = run_n_demos_diagnostic(
                        model, w, corpus, corpus_data["val"], cell_pools, eos_id, args.device,
                        seq_len=args.seq_len, eval_micro_batch=args.eval_micro_batch)
                except Exception as e:   # noqa: BLE001 -- a diagnostic must never kill the GATE
                    results.setdefault("w2_diagnostic_errors", {})[corpus] = f"{type(e).__name__}: {e}"
                _persist()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for corpus in corpora:
        try:
            results.setdefault("t2a2", {})[corpus] = run_t2a2_untrained_control(
                corpus, args.data_dir, args.device, seq_len=args.seq_len, n_windows=args.n_windows,
                n_plants=args.n_plants)
        except Exception as e:   # noqa: BLE001
            results.setdefault("t2a2", {})[corpus] = {
                "t2_void": True, "t2_void_reason": f"{type(e).__name__}: {e}"}
        _persist()

    if "W1_rwkv7" in witnesses and "W2_gpt2large" in witnesses:
        for corpus in corpora:
            c1 = results["cells"].get(f"W1_rwkv7/{corpus}", {})
            c2 = results["cells"].get(f"W2_gpt2large/{corpus}", {})
            a1 = c1.get("t1c_admissibility", {}).get("admissible")
            a2 = c2.get("t1c_admissibility", {}).get("admissible")
            if a1 and a2:
                results.setdefault("t1c", {})[corpus] = check_t1c_reference_did(c1["cell"], c2["cell"])
            else:
                results.setdefault("t1c", {})[corpus] = {
                    "passes": False, "void": True,
                    "reason": "one or both witness cells failed sec 9.2/11.7 t1c_admissibility "
                              "(D5 SERIOUS-7 -- T1c refuses to read an ungated cell)",
                    "w1_admissible": a1, "w2_admissible": a2,
                }

    # D5 round-1 SERIOUS-9 + round-2 SERIOUS-7: the FULL conjunctive instrument verdict. Round 1
    # rolled up T2a-1 ONLY -- but sec 11.4.2 makes T2a-1 **and** T2a-2 **and** T2a-3 **and** T1c
    # ALL gating. A coordinator reading `t2a1_gate_conjunction.passes == True` and green-lighting
    # the instrument while T2a-2 had FAILED (i.e. an UNTRAINED model passes the probe =>
    # sec 11.4.2: INSTRUMENT-INVALID, HALT) is exactly the false-pass this driver exists to
    # prevent -- and stdout omits `cells`, so T2a-1 was the MOST visible number in the run.
    t2a1_gate = {}
    for corpus in corpora:
        legs = []
        for w in ("W1_rwkv7", "W2_gpt2large"):
            if w in witnesses:
                legs.append(bool(results["cells"].get(f"{w}/{corpus}", {})
                                  .get("t2a1_ceiling", {}).get("passes")))
        t2a1_gate[corpus] = {"passes": bool(legs) and all(legs) and len(legs) == 2,
                              "n_required_witnesses_present": len(legs)}
    results["t2a1_gate_conjunction"] = t2a1_gate

    # D5 round-3 SERIOUS-1: EVERY leg is created UNCONDITIONALLY. Round 2 created `t2a3` only
    # `if "C1_falconmamba" in witnesses`, so a subset run produced a conjunction that never saw
    # a pinned GATING check -- a false pass by omission. `mode_gate` now also REFUSES subset runs
    # outright (above), so this is belt-and-braces: even if that refusal were ever relaxed, the
    # missing witness/corpus reads as False here rather than vanishing from the conjunction.
    # D5 round-4 SERIOUS-3: HOIST the coverage report OUT of `cells` so it reaches the surface the
    # operator actually reads. Round 3 added corpus_coverage to REPLACE the deleted val-size gate
    # -- and then attached it to `cell`, i.e. inside `results["cells"]`, the ONE key stdout
    # deliberately omits. A report the decision-maker cannot see is not a mitigation; that is
    # round-1 SERIOUS-9's "computed but never read" defect, re-committed on the very thing added
    # to replace a FATAL.
    results["coverage_summary"] = {
        k: {"val_coverage_ratio": c["corpus_coverage"]["val_coverage_ratio"],
            "ci_narrowing_upper_bound": c["corpus_coverage"]["approx_bootstrap_ci_narrowing_factor"]}
        for k, c in results["cells"].items() if "corpus_coverage" in c
    }

    # THE T1c OVERLAP ADVISORY -- REPORTED, NEVER GATING. The val splits are small enough that
    # windows overlap (measured: wikitext val_coverage_ratio ~4.25 at the pinned N_rows=2048), and
    # clustered_bootstrap_ci treats overlapping rows as independent clusters, so its CI is AT MOST
    # sqrt(coverage) too narrow. T1c gates on `did_ci[0] > 0`, which is exactly the criterion a
    # too-narrow CI can fake. This re-reads T1c's own numbers with the CI half-width inflated by
    # the reported factor, so a coordinator can SEE whether T1c passed on the strength of an
    # optimistic CI. It is ADVISORY ONLY and enters no conjunction: making it gating would VOID a
    # capable witness on the instrument's own plumbing, which is precisely round-3's FATAL.
    for corpus in REQUIRED_CORPORA:
        t1c_entry = results.get("t1c", {}).get(corpus)
        if not isinstance(t1c_entry, dict):
            continue
        adv = {}
        for wk, tag in (("W1_rwkv7", "w1"), ("W2_gpt2large", "w2")):
            wcell = results["cells"].get(f"{wk}/{corpus}", {})
            s = (wcell.get("corpus_coverage", {})
                       .get("approx_bootstrap_ci_narrowing_factor")) or 1.0
            inner = wcell.get("cell") or {}
            did, ci = inner.get("did"), (inner.get("did_ci") or [None, None])
            if did is None or ci[0] is None:
                adv[tag] = None
                continue
            lo_adj = did - (did - ci[0]) * s     # inflate the CI half-width by the reported factor
            adv[tag] = {"narrowing_factor_upper_bound": s, "did": did, "ci_lo": ci[0],
                         "ci_lo_overlap_adjusted": lo_adj, "still_excludes_zero": lo_adj > 0}
        t1c_entry["overlap_robustness_ADVISORY"] = adv

    gate = {
        "coverage_complete": (set(witnesses) >= set(REQUIRED_WITNESSES)
                               and set(corpora) >= set(REQUIRED_CORPORA)),
        "t2a1": bool(t2a1_gate) and all(v["passes"] for v in t2a1_gate.values()),
        "t2a2": all(bool(results.get("t2a2", {}).get(c, {})
                          .get("t2a2_untrained_control", {}).get("passes")) for c in REQUIRED_CORPORA),
        "t2a3": all(bool(results["cells"].get(f"C1_falconmamba/{c}", {})
                          .get("t2a3_ssm_calibration", {}).get("passes")) for c in REQUIRED_CORPORA),
        "t1c": all(bool(results.get("t1c", {}).get(c, {}).get("passes")) for c in REQUIRED_CORPORA),
    }
    # D5 round-4 M-6: enumerate the legs EXPLICITLY rather than `all(gate.values())`, which was
    # correct only because `gate["note"]` (a truthy string) happened to be inserted one line
    # later. Order-dependence in a HALT gate's own conjunction is not a property to leave to
    # line ordering.
    gate["INSTRUMENT_VALID"] = all(gate[k] for k in
                                    ("coverage_complete", "t2a1", "t2a2", "t2a3", "t1c"))
    gate["note"] = ("sec 11.4.2/11.4.5: T2a-1 (W1 AND W2, each corpus) AND T2a-2 (untrained "
                    "negative control) AND T2a-3 (SSM causal legs) AND T1c (reference DiD) are "
                    "ALL gating. Any False => INSTRUMENT-INVALID, HALT for every rung. This "
                    "roll-up is mechanical; do not hand-assemble it from the per-cell dicts.")
    results["instrument_gate"] = gate

    print(json.dumps({k: v for k, v in results.items() if k != "cells"}, indent=2, default=str))
    _persist()
    return 0


def _git_sha() -> str:
    """Best-effort commit sha for the output JSON's provenance (D5 round-2 M-10)."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__)),
                               timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _get_gpt2_tokenizer():
    from transformers import GPT2TokenizerFast   # M-4: fast, not the slow GPT2Tokenizer
    return GPT2TokenizerFast.from_pretrained("gpt2")


# =============================================================================
# gpt2-large load verification (item 2 of this build's task list). CPU-only
# by design -- see module docstring's scope boundary + the coordinator's
# report on why GPU was not used for this check.
# =============================================================================
def mode_verify_gpt2_large(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.float32)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = tok("The quick brown fox", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = HFLogitsWrapper(model)(x)
    ok = (logits.shape[0] == 1 and logits.shape[1] == x.shape[1]
          and logits.shape[2] == model.config.vocab_size)
    result = {
        "model": "gpt2-large", "n_params": n_params, "vocab_size": model.config.vocab_size,
        "logits_shape": list(logits.shape), "shape_ok": ok, "device": "cpu",
        "elapsed_s": time.time() - t0,
    }
    print(json.dumps(result, indent=2))
    assert ok, f"gpt2-large: HFLogitsWrapper output shape sanity check failed: {result}"
    assert n_params > 7 * 10 ** 8, f"gpt2-large: measured {n_params:,} params, expected ~774M"
    print(f"\ngpt2-large VERIFIED: loads, {n_params:,} params, forward pass OK (CPU, fp32).")
    return 0


# =============================================================================
# Smoke gate. CPU-only, tiny/fresh (non-`from_pretrained`) `transformers`
# classes where a real model is needed, real cached tokenizers where cheap.
# =============================================================================
def smoke(device: str = "cpu") -> int:
    print("=" * 70 + "\n  T2A_REFERENCE_DRIVER_V2_RD SMOKE GATE\n" + "=" * 70)
    n_pass, n_fail = 0, 0

    def report(name, ok, detail=""):
        nonlocal n_pass, n_fail
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

    # --- [1] HFLogitsWrapper against a REAL (tiny, untrained, no-download) transformers class ---
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=311, n_embd=16, n_layer=1, n_head=1, n_positions=32)
        tiny_hf = GPT2LMHeadModel(cfg).to(device)
        tiny_hf.eval()
        wrapped = HFLogitsWrapper(tiny_hf).to(device)
        x = torch.randint(0, 311, (3, 24), device=device)
        with torch.no_grad():
            logits = wrapped(x)
        report("[1] HFLogitsWrapper returns a raw logits tensor (not a ModelOutput) of shape (B,T,V)",
               logits.shape == (3, 24, 311), f"got {tuple(logits.shape)}")
        preds = logits.argmax(dim=-1)
        report("[1b] .argmax(dim=-1) on the wrapper's output works exactly like run_ablation_arm's "
               "own call pattern (no .logits needed downstream)",
               preds.shape == (3, 24))
    except Exception as e:  # noqa: BLE001
        report("[1] HFLogitsWrapper end-to-end vs a real transformers GPT2LMHeadModel", False,
               f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [1c] D5 SERIOUS-8 FORCED-FAIL negative test: non-finite logits must RAISE, never argmax
    #     silently to 0. ---
    try:
        class _NonFiniteStub(torch.nn.Module):
            def forward(self, input_ids=None, use_cache=None):
                B, T = input_ids.shape
                logits = torch.zeros(B, T, 50, device=input_ids.device)
                logits[0, 0, 0] = float("nan")
                class _Out:
                    pass
                o = _Out()
                o.logits = logits
                return o
        stub = HFLogitsWrapper(_NonFiniteStub())
        raised_nonfinite = False
        try:
            stub(torch.zeros(2, 4, dtype=torch.long))
        except RuntimeError:
            raised_nonfinite = True
        report("[1c] HFLogitsWrapper RAISES on non-finite logits (forced-fail negative test -- "
               "the failure mode that would make a witness look worse than it is via a silent "
               "argmax-to-0)", raised_nonfinite)
    except Exception as e:  # noqa: BLE001
        report("[1c] non-finite-logits forced-fail test", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [1d] D5 ROUND-2 FATAL-1: the READ must be fp32 even when the MODEL is bf16. A bf16
    #     argmax collapses near-ties, and argmax breaks ties toward the LOWEST index -- which is
    #     the COMMON RIVAL, not the rare planted value (high BPE id). Measured on this box: 2.2%
    #     of argmax decisions at a 0.03-logit gap flip between fp32 and bf16. Directional AGAINST
    #     the witness, at an absolute 0.90 HALT bar. ---
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg16 = GPT2Config(vocab_size=311, n_embd=16, n_layer=1, n_head=1, n_positions=32)
        bf16_model = GPT2LMHeadModel(cfg16).to(torch.bfloat16).to(device)
        bf16_model.eval()
        wrapped16 = HFLogitsWrapper(bf16_model).to(device)
        with torch.no_grad():
            out16 = wrapped16(torch.randint(0, 311, (2, 8), device=device))
        param_dtype = next(bf16_model.parameters()).dtype
        report("[1d] HFLogitsWrapper UPCASTS a bf16 model's logits to fp32 before the read "
               "(D5 round-2 FATAL-1: bf16 argmax collapses near-ties toward the LOW id = the "
               "common rival, biasing acc_copy DOWN against the witness at a HALT bar)",
               param_dtype == torch.bfloat16 and out16.dtype == torch.float32,
               f"model params={param_dtype}, wrapper output={out16.dtype} (want bfloat16 -> float32)")
    except Exception as e:  # noqa: BLE001
        report("[1d] bf16->fp32 upcast test", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [2] eot_override: positive (patched value visible mid-context) + restore-on-normal-exit ---
    original = probe.EOT_TOKEN_ID
    with eot_override(999999):
        mid_value = probe.EOT_TOKEN_ID
    report("[2] eot_override patches lm_recall_gap_probe_v2_rd.EOT_TOKEN_ID inside the `with` block",
           mid_value == 999999, f"got {mid_value}")
    report("[2b] eot_override restores the ORIGINAL value on normal exit",
           probe.EOT_TOKEN_ID == original, f"got {probe.EOT_TOKEN_ID}, expected {original}")

    # --- [2c] FORCED-FAIL negative test: restore-on-EXCEPTION, run to completion (not just written) ---
    raised = False
    try:
        with eot_override(-1):
            raise RuntimeError("deliberate -- proving restoration survives an exception")
    except RuntimeError:
        raised = True
    report("[2c] eot_override restores the ORIGINAL value even when the `with` body RAISES "
           "(forced-fail negative test, run to completion)",
           raised and probe.EOT_TOKEN_ID == original,
           f"exception_propagated={raised}, EOT_TOKEN_ID={probe.EOT_TOKEN_ID} (expected {original})")

    # --- [2d] D5 M-1 FIX: a REAL functional test, not a tautological signature comparison. Build a
    #     small synthetic corpus, take a real P_key member X from a BASELINE (default-EOT) pool
    #     build, then confirm X IS excluded from P_key when eot_token_id=X is passed EXPLICITLY,
    #     and IS NOT excluded when only eot_override(X) (no explicit kwarg) is active -- proving the
    #     def-time-binding gotcha has a real, measurable consequence. ---
    try:
        from lm_recall_gap_probe_v2_rd import build_synthetic_t2_train_corpus as _bstc
        # V_small/N reuse the EXACT calibration proven to clear the sec 11.2 floors in test [4]
        # (n_p_key=161 measured there) -- an earlier revision of this test tried a smaller,
        # independently-guessed (vocab, N) pair "for speed" and got an empty P_key (the same
        # calibration-mismatch class of bug FATAL-1 was), so this reuses the known-good scale
        # rather than re-deriving a new one.
        V_small = 20_220
        synth_small = _bstc(15_000_000, vocab_size=V_small, seed=99)
        baseline_pools = build_key_value_pools(synth_small, V_small, device)
        if baseline_pools["P_key"]:
            x_tok = baseline_pools["P_key"][0]
            with eot_override(x_tok):
                pools_explicit = build_key_value_pools(synth_small, V_small, device, eot_token_id=x_tok)
                pools_implicit_only = build_key_value_pools(synth_small, V_small, device)  # no explicit kwarg
            explicit_excludes = x_tok not in pools_explicit["P_key"]
            implicit_does_not = x_tok in pools_implicit_only["P_key"]
            report("[2d] REAL functional test: eot_token_id= passed EXPLICITLY excludes a real P_key "
                   "token; eot_override ALONE (no explicit kwarg) does NOT -- the def-time-binding "
                   "gotcha has a measurable consequence, not just a signature difference",
                   explicit_excludes and implicit_does_not,
                   f"x_tok={x_tok} excluded_when_explicit={explicit_excludes} "
                   f"still_present_when_implicit_only={implicit_does_not}")
        else:
            report("[2d] REAL functional eot_token_id test", False,
                   "baseline P_key empty -- cannot construct the test at this corpus scale")
    except Exception as e:  # noqa: BLE001
        report("[2d] REAL functional eot_token_id test", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [3] decode_corpus_to_documents: returns a LIST (never joined), one entry per document. ---
    try:
        from transformers import GPT2TokenizerFast
        gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        doc_a = gpt2_tok("The cat sat on the mat.")["input_ids"]
        doc_b = gpt2_tok("A second unrelated document about rivers.")["input_ids"]
        toks = torch.tensor(doc_a + [GPT2_EOT] + doc_b + [GPT2_EOT], dtype=torch.int64)
        offs = torch.tensor([0, len(doc_a) + 1], dtype=torch.int64)
        docs = decode_corpus_to_documents(toks, offs, gpt2_tok, n_source_tokens=10_000)
        report("[3] decode_corpus_to_documents returns a LIST of per-document strings (D5 SERIOUS-3 "
               "fix -- NOT one joined blob)",
               len(docs) == 2 and "cat" in docs[0] and "rivers" in docs[1], repr(docs))
        # --- [3b] _retokenize_documents splices eos_id BETWEEN documents' re-tokenized ids. ---
        eos_probe = 99999
        ids = _retokenize_documents(docs, gpt2_tok, eos_probe)
        n_eos = ids.count(eos_probe)
        report("[3b] _retokenize_documents splices the witness's own eos_id once PER document "
               "(genuine in-band boundary, not a text seam) -- SERIOUS-3's actual fix",
               n_eos == len(docs), f"n_eos={n_eos}, n_docs={len(docs)}")
        # --- [3c] full-split (n_source_tokens=None) decodes EVERY document, not a truncated prefix. ---
        docs_all = decode_corpus_to_documents(toks, offs, gpt2_tok, n_source_tokens=None)
        report("[3c] n_source_tokens=None decodes the FULL split (no truncation)",
               len(docs_all) == 2)
    except Exception as e:  # noqa: BLE001
        report("[3] decode_corpus_to_documents / _retokenize_documents", False,
               f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [4] D5 FATAL-1 forced-fail negative test: build_bridged_corpus MUST reject the OLD, WRONG
    #     400K-token budget (the exact scale that produced n_p_key=0 in round 1), and MUST succeed
    #     above MIN_BRIDGED_TRAIN_TOKENS. Exercises the REAL build_bridged_corpus function this
    #     time (round 1's smoke never called it at all). ---
    try:
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
        try:
            foreign_tok = AutoTokenizer.from_pretrained(
                WITNESS_SPECS["W1_rwkv7"]["hf_repo"], trust_remote_code=True)
            foreign_vocab = len(foreign_tok)
            bridge_available = True
        except Exception:
            foreign_tok, foreign_vocab, bridge_available = None, None, False

        if not bridge_available:
            # D5 round-2 SERIOUS-2: a SKIP here is not neutral -- round 1's smoke printed
            # "N PASSED, 0 FAILED" while silently running NONE of the bridge tests, which is
            # certification-by-omission of exactly the code path that carried the round-1 FATAL.
            # It is now a hard FAIL, not a SKIP.
            report("[4] full-bridge end-to-end -- W1's tokenizer is REACHABLE (required: this "
                   "smoke certifies the bridge, and a SKIP here would certify it by omission, "
                   "which is how round 1's FATAL-1 shipped)", False,
                   "W1 tokenizer not cached/reachable -- run this smoke on the H100 box "
                   "(/data/hf_cache), where it is cached.")
        else:
            eos_id = foreign_tok.eos_token_id if foreign_tok.eos_token_id is not None else 0

            # --- [4-neg] D5 round-2 SERIOUS-2: a REAL forced-fail test of build_bridged_corpus's
            #     OWN floor assertion, with NO filesystem dependency. Round 1's version was
            #     tautological twice over: `raised_fatal1 = isinstance(e, RuntimeError) or True`
            #     literally evaluates to True for ANY caught exception (so it "passed" off-box on a
            #     FileNotFoundError, never reaching the floor check), and its companion test merely
            #     asserted that two short sentences tokenize to < 5M tokens -- which never called
            #     build_bridged_corpus at all. Here we PRE-SEED _TEXT_CACHE so build_bridged_corpus
            #     runs its real code path against synthetic documents, and we require the SPECIFIC
            #     RuntimeError, not "any exception". ---
            import unittest.mock as _mock
            tiny_docs = ["a short document about cats.", "another short document about rivers."]
            seeded = {
                (("SMOKE", "smoke-corpus", "train", None)): tiny_docs,
                (("SMOKE", "smoke-corpus", "val", None)): tiny_docs,
            }
            with _mock.patch.dict(_TEXT_CACHE, seeded, clear=True):
                err = None
                try:
                    build_bridged_corpus("SMOKE", "smoke-corpus", foreign_tok, None, eos_id,
                                          foreign_vocab, None, None, device)
                except RuntimeError as e:
                    err = str(e)
                except Exception as e:  # noqa: BLE001 -- any OTHER exception is a FAILURE of this test
                    err = f"WRONG-EXCEPTION-TYPE {type(e).__name__}: {e}"
                report("[4-neg] build_bridged_corpus RAISES RuntimeError naming MIN_BRIDGED_TRAIN_TOKENS "
                       "on an under-sized bridged corpus (REAL forced-fail against the REAL function, "
                       "no filesystem dependency -- D5 round-2 SERIOUS-2: round 1's version was "
                       "tautological and never reached this code path)",
                       err is not None and "MIN_BRIDGED_TRAIN_TOKENS" in err
                       and "WRONG-EXCEPTION-TYPE" not in err,
                       (err or "NO EXCEPTION RAISED")[:150])

            # --- [4-neg-b] The SERIOUS-1 arithmetic itself: the floor must NOT sit on the
            #     V2^V5 degenerate point. At N == MIN_BRIDGED_TRAIN_TOKENS the value band
            #     [V2_MIN_COUNT_B, V5_MAX_P_TRAIN_B*N] must have at least POOL_FLOOR_P_VAL width. ---
            band_hi = V5_MAX_P_TRAIN_B * MIN_BRIDGED_TRAIN_TOKENS
            width_ratio = band_hi / V2_MIN_COUNT_B
            report("[4-neg-b] MIN_BRIDGED_TRAIN_TOKENS is NOT on the V2^V5 degenerate point: at the "
                   "floor, the value band [V2_MIN_COUNT_B, V5_MAX_P_TRAIN_B*N] carries >= "
                   "POOL_FLOOR_P_VAL multiples of the V2 floor (D5 round-2 SERIOUS-1: round 1's "
                   "5,000,000 gave the band ZERO width, re-arming the round-1 FATAL at its own fix)",
                   width_ratio >= POOL_FLOOR_P_VAL,
                   f"floor={MIN_BRIDGED_TRAIN_TOKENS:,} -> band [{V2_MIN_COUNT_B}, {band_hi:.0f}] "
                   f"= {width_ratio:.1f}x the V2 floor (need >= {POOL_FLOOR_P_VAL}x)")

            # --- [4a/4b] plumbing test at a CALIBRATED synthetic scale (vocab-magnitude-independent
            #     plumbing check; the REAL vocab-scale, REAL-corpus run happens at T2a execution
            #     time on the H100 box, not in this CPU smoke -- labels corrected per D5 SERIOUS-2,
            #     no longer claim "at W1's REAL vocab size"). ---
            V11 = 20_220
            from lm_recall_gap_probe_v2_rd import build_synthetic_t2_train_corpus
            synth_train = build_synthetic_t2_train_corpus(15_000_000, vocab_size=V11, seed=21)
            synth_val = build_synthetic_t2_train_corpus(2_000_000, vocab_size=V11, seed=22)
            report("[4-provenance] foreign tokenizer (W1's real one) reachable; used for eos_id "
                   "PROVENANCE even though the synthetic corpus below uses a CALIBRATED vocab size, "
                   "not W1's real ~65K (D5 SERIOUS-2: the prior labels claimed 'REAL vocab size' "
                   "falsely)", True, f"foreign_vocab={foreign_vocab} eos_id={eos_id}")

            with eot_override(eos_id):
                pools = build_key_value_pools(synth_train, V11, device, eot_token_id=eos_id)
            report("[4a] build_key_value_pools clears the sec 11.2 floors on a CALIBRATED synthetic "
                   "corpus (vocab=20220, matching the probe module's own smoke scale), with "
                   "eot_token_id explicitly passed",
                   pools["floor_pool_key"] and pools["floor_pool_val"] and pools["floor_licensed_b"],
                   f"n_p_key={pools['n_p_key']} vocab={V11}")

            cfg = GPT2Config(vocab_size=V11, n_embd=16, n_layer=1, n_head=1, n_positions=SEQ_LEN_DEFAULT + 8)
            tiny_witness = HFLogitsWrapper(GPT2LMHeadModel(cfg)).to(device)
            tiny_witness.eval()
            counts_by_token = torch.bincount(synth_train, minlength=V11).tolist()
            rng = random.Random(0)
            delta_pool = [rng.randint(4, 120) for _ in range(64)]

            with eot_override(eos_id):
                t2_result = run_t2_repaired_probe(tiny_witness, synth_val, seq_len=128, device=device,
                                                   corpus_name="smoke-bridge-corpus", delta_pool=delta_pool,
                                                   pools=pools, counts_by_token=counts_by_token,
                                                   n_plants=48, vocab_size=V11, eval_micro_batch=16)
            report("[4b] run_t2_repaired_probe completes end-to-end through HFLogitsWrapper wrapping a "
                   "REAL transformers.GPT2LMHeadModel, VOID=False",
                   not t2_result.get("void"), str({k: v for k, v in t2_result.items() if k != "records"}))
            if not t2_result.get("void") and t2_result.get("records"):
                t2b1 = check_t2b1_mechanism_exists(t2_result["records"])
                t2b1b = check_t2b1b_key_conditioned(t2_result["records"])
                t2a1 = check_t2a1_ceiling(t2_result["records"], t2b1, t2b1b)
                report("[4c] check_t2b1 / check_t2b1b / check_t2a1_ceiling run without error on "
                       "REAL-bridge output (an untrained tiny model correctly fails T2a-1)",
                       not t2a1["passes"], f"acc_copy={t2a1.get('acc_copy')}")
                strat = stratified_acc_copy_report(t2_result["records"])
                report("[4d] stratified_acc_copy_report (sec 11.4.3 build gate item 5) runs without "
                       "error and returns all three stratifications",
                       set(strat) == {"by_rival_strength_max_rival_p", "by_rank_b_given_a", "by_count_ab"})
                # D5 round-2 M-4: assert on CONTENT, not key-presence. Round 1's assertion was
                # `all("skipped" in v or "void" in v for v in sweep.values())` -- every returned
                # dict carries one of those keys whether it succeeded OR VOIDed, so it could not
                # fail. Require instead that a real, non-void sweep point produced a real acc_copy.
                sweep = run_delta_sweep(tiny_witness, "smoke-bridge-corpus", synth_val, pools,
                                         counts_by_token, eos_id, V11, device, seq_len=128,
                                         natural_delta_pool=delta_pool, n_plants_per_delta=32,
                                         delta_targets=(5, 20, 40))
                live_pts = [v for v in sweep.values() if not v.get("skipped") and not v.get("void")]
                report("[4e] run_delta_sweep (sec 11.4.3's W2 Delta-sweep) produces at least one "
                       "NON-VOID point carrying a real acc_copy (content assertion, not "
                       "key-presence -- D5 round-2 M-4)",
                       bool(live_pts) and all(v.get("acc_copy") is not None for v in live_pts),
                       f"{len(live_pts)}/{len(sweep)} live points; {sweep}")
                ndemos = run_n_demos_diagnostic(tiny_witness, "smoke", "smoke-bridge-corpus", synth_val,
                                                 pools, eos_id, device, seq_len=128,
                                                 n_plants_per_level=64, min_surviving=8)
                # D5 round-2 M-4 + SERIOUS-5: content assertion (a real acc_copy off a real,
                # cap-cleared sample) AND the PAIRING property -- every level must report the
                # SAME paired_seed, which is what makes acc_copy(1)->acc_copy(2)->acc_copy(4) a
                # within-item contrast rather than three unrelated draws.
                live_nd = [v for v in ndemos.values() if not v.get("skipped") and not v.get("void")]
                seeds_used = {v.get("paired_seed") for v in ndemos.values() if "paired_seed" in v}
                report("[4f] run_n_demos_diagnostic produces NON-VOID levels carrying a real "
                       "acc_copy, and every level shares ONE paired_seed (D5 round-2 SERIOUS-5: "
                       "round 1 put n_demos INTO the seed, so the levels drew different windows "
                       "and different (a,b) pairs -- confounding the very contrast the diagnostic "
                       "exists to make)",
                       bool(live_nd) and all(v.get("acc_copy") is not None for v in live_nd)
                       and len(seeds_used) == 1,
                       f"{len(live_nd)} live levels, distinct paired_seeds={len(seeds_used)}; {ndemos}")
                # --- [4h] the drop-cap safeguard has TEETH (coordinator SERIOUS-B / round-2
                #     SERIOUS-5): a min_surviving floor set above the achievable sample must VOID
                #     the level rather than silently reporting an acc_copy off a tiny sample. ---
                nd_strict = run_n_demos_diagnostic(tiny_witness, "smoke", "smoke-bridge-corpus",
                                                    synth_val, pools, eos_id, device, seq_len=128,
                                                    n_plants_per_level=64, min_surviving=10_000)
                report("[4h] run_n_demos_diagnostic VOIDs a level whose surviving-plant count is "
                       "below min_surviving (forced-fail: round 1 VOIDed only on LITERALLY ZERO "
                       "survivors, so a level built from 4 of 64 windows would have silently "
                       "reported an acc_copy into a load-bearing diagnostic)",
                       all(v.get("void") or v.get("skipped") for v in nd_strict.values())
                       and any("min_surviving" in str(v.get("void_reason", "")) for v in nd_strict.values()),
                       str({k: v.get("void_reason", v.get("reason")) for k, v in nd_strict.items()})[:160])
    except Exception as e:  # noqa: BLE001
        report("[4] full-bridge end-to-end (decode/tokenizer-level pieces)", False,
               f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [4g] FORCED-FAIL negative test for the NEW n_demos plant assertion (mirrors
    #     plant_and_verify_t2_window's own smoke discipline -- "a structural check without a
    #     forced-fail negative test is not a check"). Deliberately construct a window where `a`
    #     ALREADY occurs naturally at a position that will collide with a planted position. ---
    try:
        window = [7] * 350
        positions = [252, 292]   # n_demos=1: demo position 252, query position 292 -- both < 350
        window[100] = 42         # `a`'s value, planted NATURALLY at a position OUTSIDE `positions` --
                                  # after the plant writes w[252]=w[292]=42, count(a=42) becomes 3
                                  # (100, 252, 292) against an expected 2 ({252,292}) -- a genuine
                                  # contest the hard assertion must catch, not a no-op overwrite.
        contested = False
        try:
            plant_and_verify_t2_window_ndemos(window, positions, a=42, b=13)
        except PlantContestedError:
            contested = True
        report("[4g] plant_and_verify_t2_window_ndemos FORCED-FAIL: a natural pre-existing occurrence "
               "of `a` OUTSIDE the planted positions is caught (PlantContestedError), never silently "
               "accepted", contested)
    except Exception as e:  # noqa: BLE001
        report("[4g] n_demos plant forced-fail test", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [5] T2a-2 untrained-control plumbing: RUNG_14M_CONFIG matches DeltaNetLM's own defaults. ---
    try:
        import inspect
        sig = inspect.signature(DeltaNetLM.__init__)
        drift = {k: (sig.parameters[k].default, v) for k, v in RUNG_14M_CONFIG.items()
                 if k in sig.parameters and sig.parameters[k].default != v and k != "conv_size"}
        report("[5] RUNG_14M_CONFIG (d_model/d_state/n_layers) matches DeltaNetLM's own constructor "
               "defaults -- 'the 14M rung's exact architecture' is not a stale copy",
               not drift, f"drift={drift}" if drift else "")
    except Exception as e:  # noqa: BLE001
        report("[5] RUNG_14M_CONFIG cross-check", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [6] mode_gate's refusal gates have teeth (forced-fail, run to completion). ---
    class _FakeArgsNoAttest:
        i_am_the_t2a_execution_agent = False
    refused = False
    try:
        mode_gate(_FakeArgsNoAttest())
    except SystemExit:
        refused = True
    report("[6] mode_gate REFUSES to run without --i-am-the-t2a-execution-agent (forced-fail "
           "negative test, run to completion)", refused)

    class _FakeArgsNoOut:
        i_am_the_t2a_execution_agent = True
        out = None
        n_plants = N_PLANTS_DEFAULT
        n_windows = N_ROWS_DEFAULT
    refused_no_out = False
    try:
        mode_gate(_FakeArgsNoOut())
    except SystemExit:
        refused_no_out = True
    report("[6b] mode_gate REFUSES to run without --out (D5 SERIOUS-9 forced-fail negative test)",
           refused_no_out)

    class _FakeArgsMismatch:
        i_am_the_t2a_execution_agent = True
        out = "/tmp/unused.json"
        n_plants = 100
        n_windows = 200
    refused_mismatch = False
    try:
        mode_gate(_FakeArgsMismatch())
    except SystemExit:
        refused_mismatch = True
    report("[6c] mode_gate REFUSES to run when --n-plants != --n-windows (D5 SERIOUS-4 forced-fail "
           "negative test)", refused_mismatch)

    # --- [6d] D5 round-2 SERIOUS-6: --gate must REFUSE a truncated bridged corpus outright.
    #     The truncation flags VOID the pools (SERIOUS-1), make the bootstrap anticonservative
    #     (a FALSE PASS on T1c), and apply ONLY to the bridged witnesses -- so they would
    #     silently compare the two REQUIRED ceiling witnesses on different-sized corpora. ---
    class _FakeArgsTruncated:
        i_am_the_t2a_execution_agent = True
        out = "/tmp/unused.json"
        n_plants = N_PLANTS_DEFAULT
        n_windows = N_ROWS_DEFAULT
        n_source_tokens_train = 400_000     # the exact round-1 FATAL-1 value
        n_source_tokens_val = 100_000
    refused_trunc = False
    try:
        mode_gate(_FakeArgsTruncated())
    except SystemExit:
        refused_trunc = True
    report("[6d] mode_gate REFUSES a TRUNCATED bridged corpus (--n-source-tokens-*), which would "
           "VOID the sec 11.2 pools, make the clustered bootstrap anticonservative (a FALSE PASS "
           "on T1c), and shrink ONLY the bridged witnesses' corpora while W2 keeps the full split "
           "(D5 round-2 SERIOUS-6 forced-fail negative test)", refused_trunc)

    # --- [6e] D5 round-3 SERIOUS-1: a SUBSET run must be REFUSED. Round 2's roll-up created the
    #     t2a3 leg only `if "C1_falconmamba" in witnesses`, so `--witness W1 W2` produced an
    #     INSTRUMENT_VALID conjunction that never evaluated a pinned GATING check -- a false pass
    #     by omission. ---
    class _FakeArgsSubset:
        i_am_the_t2a_execution_agent = True
        out = "/tmp/unused.json"
        n_plants = N_PLANTS_DEFAULT
        n_windows = N_ROWS_DEFAULT
        n_source_tokens_train = None
        n_source_tokens_val = None
        witness = ["W1_rwkv7", "W2_gpt2large"]        # C1 omitted -> t2a3 would vanish
        corpus = ["openr1-mix-ext"]                    # one of two pinned corpora
    refused_subset = False
    try:
        mode_gate(_FakeArgsSubset())
    except SystemExit:
        refused_subset = True
    report("[6e] mode_gate REFUSES a SUBSET run (missing witness and/or corpus), which would have "
           "silently narrowed the INSTRUMENT_VALID conjunction and dropped the pinned GATING "
           "T2a-3 leg entirely -- a FALSE PASS by omission (D5 round-3 SERIOUS-1 forced-fail)",
           refused_subset)

    # --- [6e-b] D5 round-6 M-1: a SUPERSET must ALSO be refused, and this had NO regression test
    #     -- reverting the refusal to round-4's `>=` passed the entire suite. A superset is wrong
    #     in the HALT-BY-DEFECT direction: t2a1_gate iterates the OPERATOR's corpora while the
    #     other legs iterate REQUIRED_CORPORA, so an extra corpus becomes a GATING T2a-1 leg and a
    #     void there HALTs the program though both pinned corpora passed. ---
    def _gate_refuses(witness_list, corpus_list):
        class _A:
            i_am_the_t2a_execution_agent = True
            out = "/tmp/unused.json"
            n_plants = N_PLANTS_DEFAULT
            n_windows = N_ROWS_DEFAULT
            n_source_tokens_train = None
            n_source_tokens_val = None
            witness = witness_list
            corpus = corpus_list
        try:
            mode_gate(_A())
            return False
        except SystemExit:
            return True
        except Exception:  # noqa: BLE001 -- anything else means it got PAST the refusal
            return False

    superset_w = _gate_refuses(list(REQUIRED_WITNESSES) + ["W3_llama32_1b"], list(REQUIRED_CORPORA))
    superset_c = _gate_refuses(list(REQUIRED_WITNESSES), list(REQUIRED_CORPORA) + ["wikitext"])
    dupes = _gate_refuses(list(REQUIRED_WITNESSES),
                           [REQUIRED_CORPORA[0]] + list(REQUIRED_CORPORA))
    report("[6e-b] mode_gate REFUSES a SUPERSET (extra witness OR extra corpus) and DUPLICATES. "
           "A superset is HALT-BY-DEFECT (an extra corpus becomes a gating T2a-1 leg while the "
           "other legs read only REQUIRED_CORPORA); duplicates silently double a multi-hour "
           "re-tokenization. Round-4's `>=` admitted both and had NO regression test "
           "(D5 round-6 M-1/M-3)",
           superset_w and superset_c and dupes,
           f"extra_witness_refused={superset_w}, extra_corpus_refused={superset_c}, "
           f"duplicates_refused={dupes}")

    # --- [6f] D5 round-3 FATAL-1 REGRESSION GUARD: no val-size floor may exceed the REAL val
    #     splits. The round-2 fix added MIN_BRIDGED_VAL_TOKENS = 4,194,304 while the actual val
    #     splits are 2,300,595 (openr1) and 247,349 (wikitext) -- so it VOIDed the REQUIRED W1
    #     witness on both corpora, before a single forward pass. This test fails loudly if any
    #     future edit reintroduces a val floor above the smaller real split. ---
    REAL_VAL_TOKENS = {"openr1-mix-ext": 2_300_595, "wikitext-mix-ext": 247_349}   # measured on-box
    val_floor = globals().get("MIN_BRIDGED_VAL_TOKENS")
    report("[6f] NO val-size GATE exists that the REAL corpora cannot clear (D5 round-3 FATAL-1 "
           "regression guard: the deleted floor was 4,194,304 vs real val splits of "
           f"{REAL_VAL_TOKENS['openr1-mix-ext']:,} / {REAL_VAL_TOKENS['wikitext-mix-ext']:,} -- it "
           "VOIDed the REQUIRED W1 witness on the instrument's own plumbing)",
           val_floor is None or val_floor <= min(REAL_VAL_TOKENS.values()),
           f"MIN_BRIDGED_VAL_TOKENS={val_floor!r} (must be absent, or <= "
           f"{min(REAL_VAL_TOKENS.values()):,})")

    # --- [6h] THE ROOT-CAUSE FIX. D5 round-4 FATAL-1's real finding was not the dangling constant
    #     -- it was WHY the dangling constant shipped: **mode_gate's BODY had ZERO test coverage.**
    #     All five mode_gate tests above ([6], [6b], [6c], [6d], [6e]) are REFUSAL tests: every one
    #     returns at a SystemExit before reaching the body. So a NameError sitting in the body's
    #     very first statement survived a "30/30 PASS" smoke run, and would have killed the real
    #     --gate run after the D5 audit cleared it. FOUR consecutive fix passes each shipped a new
    #     FATAL into this function; this is the test that makes the next one visible.
    #
    #     This drives the FULL body -- past every refusal, through the witness loop, the T2a-2
    #     loop, the T1c block, the coverage summary, the advisory, and the roll-up -- with a
    #     data_dir that does not exist, so EVERY witness fails to load and EVERY cell fails closed.
    #     No model, no corpus, no GPU, no network. It asserts the run completes AND that the
    #     roll-up correctly reads INSTRUMENT_VALID=False (fail-closed), which is the only correct
    #     verdict when nothing could be evaluated. ---
    import tempfile
    import unittest.mock as _mock

    def _fake_gate_args(out_path):
        class _A:
            i_am_the_t2a_execution_agent = True
            out = out_path
            n_plants = N_PLANTS_DEFAULT
            n_windows = N_ROWS_DEFAULT
            n_source_tokens_train = None
            n_source_tokens_val = None
            witness = list(REQUIRED_WITNESSES)
            corpus = list(REQUIRED_CORPORA)
            seq_len = SEQ_LEN_DEFAULT
            eval_micro_batch = 16
            device = "cpu"
            data_dir = "/nonexistent__smoke_only"
        return _A()

    # [6h] FAIL-CLOSED path. D5 round-5 SERIOUS-3: `load_witness_model` is MOCKED, not merely
    # starved of a data_dir. The round-4 version relied on a nonexistent data_dir to make the
    # witnesses fail -- but load_witness_model runs BEFORE any corpus load, so on the real gate box
    # (where /data/hf_cache IS populated -- that is a PRECONDITION of the gate) it would have
    # loaded RWKV7-1.5B + gpt2-large + falcon-mamba-7B into CPU RAM: ~19GB and minutes of I/O,
    # inside a --smoke run, on a box whose 8 GPUs are training. That violates this file's own
    # pinned scope boundary ("--smoke is CPU-only / tiny-model / no-training-disturbance") and is
    # a real OOM-kill vector. It ALSO made the test's covered path depend on the HF cache state.
    # Mocking pins the path AND makes the test genuinely model-free everywhere.
    try:
        out_path = os.path.join(tempfile.gettempdir(), f"t2a_smoke_gate_fc_{os.getpid()}.json")
        with _mock.patch.object(sys.modules[__name__], "load_witness_model",
                                 side_effect=RuntimeError("SMOKE: witness load refused")), \
             _mock.patch.object(sys.modules[__name__], "run_t2a2_untrained_control",
                                 side_effect=RuntimeError("SMOKE: t2a2 refused")):
            rc = mode_gate(_fake_gate_args(out_path))
        with open(out_path) as f:
            gate_json = json.load(f)
        g = gate_json.get("instrument_gate", {})
        report("[6h] mode_gate's BODY runs end-to-end and FAILS CLOSED (INSTRUMENT_VALID=False) "
               "when every witness refuses to load -- model-free (loader MOCKED, D5 round-5 "
               "SERIOUS-3: the round-4 version would have loaded 19GB of real weights inside "
               "--smoke on the training box)",
               rc == 0 and g.get("INSTRUMENT_VALID") is False
               and set(g) >= {"coverage_complete", "t2a1", "t2a2", "t2a3", "t1c", "INSTRUMENT_VALID"},
               f"rc={rc}, gate={ {k: v for k, v in g.items() if k != 'note'} }")
        os.unlink(out_path)
    except Exception as e:  # noqa: BLE001
        report("[6h] mode_gate BODY end-to-end (fail-closed)", False,
               f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [6j] THE HAPPY PATH. D5 round-5 SERIOUS-1: [6h] alone is NOT a real body test, because
    #     `INSTRUMENT_VALID is False` is ALSO what every defect inside mode_gate produces (the
    #     except-handlers convert any NameError/TypeError into a fail-closed void). A test whose
    #     PASS condition is the state a bug produces cannot detect the bug. The auditor planted two
    #     defects that [6h] and [6i] both PASSED. This test drives the body with HEALTHY stubbed
    #     cells and asserts the roll-up reads INSTRUMENT_VALID=**True**, that coverage_summary is
    #     NON-EMPTY and carries the exact expected ratio, that the T1c overlap advisory's
    #     arithmetic is right on a known fixture, and -- the property nothing else asserts -- that
    #     the advisory is NON-GATING (it must not flip gate["t1c"], because gating it was round-3's
    #     FATAL). Then it flips one leg and confirms the conjunction goes False. ---
    try:
        VAL_N = 247_349                       # the real wikitext val split, measured on-box
        exp_cov = (N_ROWS_DEFAULT * (SEQ_LEN_DEFAULT + 1)) / VAL_N
        exp_s = exp_cov ** 0.5

        # D5 round-6 SERIOUS-2: the fixture is WITNESS-SPECIFIC. Round 5's `_healthy_cell` handed
        # EVERY witness a `t2a3_ssm_calibration` key, so a roll-up bug reading T2a-3 off W1's cell
        # instead of C1's was invisible. Only C1 carries it, as in the real run.
        def _healthy_cell(witness_key, corpus_name, did=0.10, ci_lo=0.02):
            c = {
                "witness": witness_key, "corpus": corpus_name, "t2_void": False,
                "cell": {"did": did, "did_ci": [ci_lo, did + 0.08]},
                "t1c_admissibility": {"admissible": True},
                "t2a1_ceiling": {"passes": True},
                "corpus_coverage": corpus_coverage_provenance(VAL_N, N_ROWS_DEFAULT, SEQ_LEN_DEFAULT),
            }
            if witness_key == "C1_falconmamba":
                c["t2a3_ssm_calibration"] = {"passes": True}
            return c

        def _fake_load_model(wk, device, dtype=None):
            return object(), object(), 50257, 50256

        def _fake_load_corpus(wk, corpus, ref_tok, gpt2_tok, eos_id, vocab, data_dir,
                               nst, nsv, device, seq_len=SEQ_LEN_DEFAULT):
            return {"train": torch.zeros(4, dtype=torch.int64),
                    "val": torch.zeros(VAL_N, dtype=torch.int64), "bridged": False}

        def _fake_t2a2(corpus, data_dir, device, **kw):
            return {"corpus": corpus, "t2_void": False,
                    "t2a2_untrained_control": {"passes": True}}

        def _run_happy(break_leg=None, break_witness=None, break_corpus=None):
            """`break_leg` in {None,'t2a1','t2a3','t2a2','t1c'} -- fails EXACTLY ONE leg, so a
            per-leg negative control can assert the OTHER legs stay True (D5 round-6 SERIOUS-2:
            round 5 flipped t2a1 uniformly across all six cells, which cannot distinguish `all`
            from `any`, nor notice a leg silently dropped from the conjunction)."""
            def _cell(model, wk, corpus, *a, **kw):
                c = _healthy_cell(wk, corpus)
                if break_leg == "t2a1" and (break_witness in (None, wk)) \
                        and (break_corpus in (None, corpus)):
                    c["t2a1_ceiling"] = {"passes": False}
                if break_leg == "t2a3" and wk == "C1_falconmamba":
                    c["t2a3_ssm_calibration"] = {"passes": False}
                if break_leg == "t1c":
                    c["t1c_admissibility"] = {"admissible": False}
                return c

            def _t2a2(corpus, data_dir, device, **kw):
                r = _fake_t2a2(corpus, data_dir, device, **kw)
                if break_leg == "t2a2":
                    r["t2a2_untrained_control"] = {"passes": False}
                return r

            op = os.path.join(tempfile.gettempdir(), f"t2a_smoke_gate_hp_{os.getpid()}.json")
            m = sys.modules[__name__]
            with _mock.patch.object(m, "load_witness_model", side_effect=_fake_load_model), \
                 _mock.patch.object(m, "load_witness_corpus", side_effect=_fake_load_corpus), \
                 _mock.patch.object(m, "run_witness_cell", side_effect=_cell), \
                 _mock.patch.object(m, "run_t2a2_untrained_control", side_effect=_t2a2), \
                 _mock.patch.object(m, "_get_gpt2_tokenizer", side_effect=lambda: None):
                mode_gate(_fake_gate_args(op))
            with open(op) as f:
                j = json.load(f)
            os.unlink(op)
            return j

        j = _run_happy()
        g = j["instrument_gate"]
        cs = j.get("coverage_summary", {})
        adv = j["t1c"]["wikitext-mix-ext"]["overlap_robustness_ADVISORY"]["w2"]
        # the pinned advisory arithmetic: lo_adj = did - (did - ci_lo) * s
        exp_lo_adj = 0.10 - (0.10 - 0.02) * exp_s

        report("[6j] mode_gate HAPPY PATH: healthy cells roll up to INSTRUMENT_VALID=True, with "
               "ALL FIVE legs true (the assertion [6h] structurally CANNOT make, since a bug also "
               "yields False -- D5 round-5 SERIOUS-1)",
               g["INSTRUMENT_VALID"] is True and all(g[k] for k in
                                                      ("coverage_complete", "t2a1", "t2a2", "t2a3", "t1c")),
               f"gate={ {k: v for k, v in g.items() if k != 'note'} }")
        report("[6j-b] coverage_summary is HOISTED, NON-EMPTY, and carries the exact expected "
               "val_coverage_ratio (D5 round-4 SERIOUS-3: the report that replaced a FATAL gate "
               "was invisible to the operator; [6h] could not see this because its cells all void)",
               bool(cs) and abs(cs["W2_gpt2large/wikitext-mix-ext"]["val_coverage_ratio"] - exp_cov) < 1e-9,
               f"n_entries={len(cs)}, ratio={cs.get('W2_gpt2large/wikitext-mix-ext', {}).get('val_coverage_ratio')}")
        report("[6j-c] the T1c overlap advisory computes lo_adj = did - (did - ci_lo)*s correctly "
               "on a known fixture, and correctly reports that an overlap-adjusted CI NO LONGER "
               "excludes zero",
               abs(adv["ci_lo_overlap_adjusted"] - exp_lo_adj) < 1e-9
               and adv["still_excludes_zero"] is False,
               f"lo_adj={adv['ci_lo_overlap_adjusted']:.6f} (expected {exp_lo_adj:.6f}), "
               f"still_excludes_zero={adv['still_excludes_zero']}")
        report("[6j-d] THE NON-GATING PROPERTY: the advisory says the adjusted CI does NOT exclude "
               "zero, yet gate['t1c'] is STILL True and INSTRUMENT_VALID is STILL True -- the "
               "advisory is ADVISORY. Gating it is exactly what made round-3's val floor a FATAL "
               "(it would VOID a capable witness on the instrument's own plumbing).",
               adv["still_excludes_zero"] is False and g["t1c"] is True
               and g["INSTRUMENT_VALID"] is True)
        # --- [6j-e] PER-LEG NEGATIVE CONTROLS (D5 round-6 SERIOUS-2). For each gating leg L,
        #     fail ONLY L and assert: gate[L] is False, INSTRUMENT_VALID is False, AND THE OTHER
        #     THREE LEGS STAY TRUE. Round 5's version flipped t2a1 uniformly across all six cells,
        #     which could not distinguish `all` from `any`, could not notice a leg dropped from the
        #     conjunction, and could not catch a leg reading the WRONG witness's cell. The auditor
        #     landed seven false-pass mutations that the old control missed; these catch them. ---
        leg_results = {}
        for leg in ("t2a1", "t2a2", "t2a3", "t1c"):
            gg = _run_happy(break_leg=leg)["instrument_gate"]
            others = [k for k in ("t2a1", "t2a2", "t2a3", "t1c") if k != leg]
            leg_results[leg] = {
                "leg_false": gg[leg] is False,
                "invalid": gg["INSTRUMENT_VALID"] is False,
                "others_still_true": all(gg[o] is True for o in others),
                "gate": {k: v for k, v in gg.items() if k != "note"},
            }
        all_legs_ok = all(r["leg_false"] and r["invalid"] and r["others_still_true"]
                          for r in leg_results.values())
        report("[6j-e] PER-LEG NEGATIVE CONTROLS: failing EXACTLY ONE of {t2a1, t2a2, t2a3, t1c} "
               "sets THAT leg False, forces INSTRUMENT_VALID False, and leaves the OTHER THREE "
               "True -- proves every pinned leg is (a) in the conjunction, (b) read off the right "
               "cell, and (c) combined with AND not OR (D5 round-6 SERIOUS-2)",
               all_legs_ok,
               "; ".join(f"{k}: leg={v['leg_false']} invalid={v['invalid']} "
                          f"others_ok={v['others_still_true']}" for k, v in leg_results.items()))

        # --- [6j-f] SINGLE-WITNESS / SINGLE-CORPUS granularity: T2a-1 must be conjunctive ACROSS
        #     witnesses AND corpora. Failing W2 on ONE corpus alone must still HALT. ---
        g_w2 = _run_happy(break_leg="t2a1", break_witness="W2_gpt2large",
                           break_corpus="wikitext-mix-ext")["instrument_gate"]
        report("[6j-f] T2a-1 is conjunctive across BOTH witnesses AND BOTH corpora: failing ONLY "
               "W2 on ONLY wikitext still drives t2a1=False and INSTRUMENT_VALID=False (an `any`-"
               "instead-of-`all` bug on either axis would pass this cell and be invisible)",
               g_w2["t2a1"] is False and g_w2["INSTRUMENT_VALID"] is False,
               f"gate={ {k: v for k, v in g_w2.items() if k != 'note'} }")
    except Exception as e:  # noqa: BLE001
        report("[6j] mode_gate HAPPY PATH (roll-up / coverage / advisory / non-gating)", False,
               f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [6i] no name in mode_gate's body can be unbound (the FATAL-1(R4) class, caught
    #     statically rather than only by execution -- [6h] catches it dynamically, this catches it
    #     even on a code path [6h] happens not to reach). ---
    try:
        import ast as _ast
        import builtins as _builtins
        _src = open(os.path.abspath(__file__)).read()
        _tree = _ast.parse(_src)
        # D5 round-5 SERIOUS-2: the FIRST version of this check was far too permissive and missed
        # 4 of 5 injected defects. Two bugs, both fixed here:
        #  (a) it harvested module bindings by walking the WHOLE tree, so EVERY OTHER FUNCTION'S
        #      LOCALS became "module-level" names (measured: 344 "known" names vs a true
        #      module-level set of 83 -- a 4x over-approximation that hid real unbound names).
        #      Module bindings must come from TOP-LEVEL statements ONLY.
        #  (b) `_names_in` walked a Store target with ast.walk, so a SUBSCRIPT store
        #      (`results["x"] = ...`) marked its BASE name (`results`) as bound -- meaning a bare
        #      typo like `resultz["x"] = ...`, bound nowhere, read as "bound". A Subscript or
        #      Attribute store binds NOTHING.
        def _bind_targets(target):
            """Names actually BOUND by a Store target. Tuple/List unpacking recurses; Subscript
            and Attribute stores bind NOTHING (they mutate an existing object)."""
            if isinstance(target, _ast.Name):
                return {target.id}
            if isinstance(target, (_ast.Tuple, _ast.List)):
                out = set()
                for el in target.elts:
                    out |= _bind_targets(el)
                return out
            if isinstance(target, _ast.Starred):
                return _bind_targets(target.value)
            return set()   # Subscript / Attribute: binds nothing

        def _bindings(nodes, recurse: bool):
            """`recurse=False` => TOP-LEVEL statements only (module scope). `recurse=True` =>
            walk into the body (function scope, incl. nested blocks)."""
            b = set()
            stream = []
            for n in nodes:
                stream.extend(_ast.walk(n) if recurse else [n])
                if not recurse:
                    # still need targets inside top-level for/with/try blocks' own headers
                    for sub in _ast.walk(n):
                        if isinstance(sub, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                            b.add(sub.name)
            for n in stream:
                if isinstance(n, _ast.Assign):
                    for t in n.targets:
                        b |= _bind_targets(t)
                elif isinstance(n, (_ast.AugAssign, _ast.AnnAssign)):
                    b |= _bind_targets(n.target)
                elif isinstance(n, (_ast.For, _ast.AsyncFor)):
                    b |= _bind_targets(n.target)
                elif isinstance(n, _ast.comprehension):
                    b |= _bind_targets(n.target)
                elif isinstance(n, (_ast.With, _ast.AsyncWith)):
                    for item in n.items:
                        if item.optional_vars is not None:
                            b |= _bind_targets(item.optional_vars)
                elif isinstance(n, _ast.ExceptHandler) and n.name:
                    b.add(n.name)
                elif isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                    b.add(n.name)
                elif isinstance(n, (_ast.Import, _ast.ImportFrom)):
                    for a in n.names:
                        b.add((a.asname or a.name).split(".")[0])
            return b

        _module_bound = _bindings(_tree.body, recurse=False)   # TOP-LEVEL ONLY
        _module_bound |= {"__file__", "__name__", "__doc__"}

        # D5 round-6 SERIOUS-1: scan EVERY function on the gate's critical path, not just
        # mode_gate. Round 5's fix mocked load_witness_model / load_witness_corpus /
        # run_witness_cell / run_t2a2_untrained_control in [6h]/[6j] -- which was REQUIRED (they
        # would otherwise load 19GB of real weights inside --smoke). But that left those four
        # bodies with ZERO coverage, static AND dynamic, while `run_witness_cell` runs for all six
        # real cells. The round-6 auditor re-injected R4's literal FATAL into each of them and this
        # check MISSED ALL FOUR. A dangling name in run_witness_cell would NameError -> be caught
        # by mode_gate's per-cell `except Exception` -> void EVERY cell -> INSTRUMENT_VALID=False
        # -> HALT-BY-DEFECT: R4's exact shape, one function over.
        _GATE_PATH_FNS = ("mode_gate", "run_witness_cell", "load_witness_model",
                           "load_witness_corpus", "run_t2a2_untrained_control",
                           "build_bridged_corpus", "run_delta_sweep", "run_n_demos_diagnostic",
                           "corpus_coverage_provenance", "stratified_acc_copy_report",
                           "_retokenize_documents", "decode_corpus_to_documents")
        _fns_by_name = {n.name: n for n in _ast.walk(_tree)
                         if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        _unbound_by_fn = {}
        for _fname in _GATE_PATH_FNS:
            _fn = _fns_by_name.get(_fname)
            if _fn is None:
                _unbound_by_fn[_fname] = ["<FUNCTION NOT FOUND -- renamed or deleted?>"]
                continue
            _local = _bindings(_fn.body, recurse=True)
            _local |= {a.arg for a in _fn.args.args}
            _local |= {a.arg for a in _fn.args.kwonlyargs}
            if _fn.args.vararg:
                _local.add(_fn.args.vararg.arg)
            if _fn.args.kwarg:
                _local.add(_fn.args.kwarg.arg)
            # nested defs/lambdas close over the ENCLOSING function's bindings, so union them in
            # (otherwise `_persist` closing over `args`/`results` false-positives).
            for _inner in _ast.walk(_fn):
                if isinstance(_inner, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Lambda)):
                    _local |= {a.arg for a in _inner.args.args}
                    _local |= {a.arg for a in _inner.args.kwonlyargs}
            _known = _module_bound | _local | set(dir(_builtins))
            _bad = sorted({n.id for n in _ast.walk(_fn)
                            if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Load)
                            and n.id not in _known})
            if _bad:
                _unbound_by_fn[_fname] = _bad
        report("[6i] STATIC check: every name loaded in EVERY gate-path function is bound "
               "(module-level / local / import / builtin). Widened from mode_gate-only per D5 "
               "round-6 SERIOUS-1: the four functions [6h]/[6j] MOCK (run_witness_cell, "
               "load_witness_model, load_witness_corpus, run_t2a2_untrained_control) had zero "
               "static AND dynamic coverage, and an R4-class dangling name in run_witness_cell "
               "would void every cell => HALT-BY-DEFECT",
               not _unbound_by_fn,
               f"unbound: {_unbound_by_fn}" if _unbound_by_fn
               else f"clean across {len(_GATE_PATH_FNS)} gate-path functions")
    except Exception as e:  # noqa: BLE001
        report("[6i] static unbound-name check on mode_gate", False,
               f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [6g] the coverage provenance that REPLACED that gate actually reports the overlap. ---
    cov = corpus_coverage_provenance(REAL_VAL_TOKENS["wikitext-mix-ext"], N_ROWS_DEFAULT, SEQ_LEN_DEFAULT)
    report("[6g] corpus_coverage_provenance REPORTS the val-window overlap (val_coverage_ratio > 1 "
           "on the real wikitext val split) instead of gating on it -- the measured-not-enforced "
           "replacement for the deleted floor",
           cov["val_coverage_ratio"] is not None and cov["val_coverage_ratio"] > 1.0
           and cov["approx_bootstrap_ci_narrowing_factor"] > 1.0,
           f"val_coverage_ratio={cov['val_coverage_ratio']:.2f}, "
           f"ci_narrowing~{cov['approx_bootstrap_ci_narrowing_factor']:.2f}x")

    print("\n" + "=" * 70)
    print(f"  T2A_REFERENCE_DRIVER_V2_RD SMOKE: {n_pass} PASSED, {n_fail} FAILED")
    print("=" * 70)
    if n_fail:
        raise SystemExit(f"{n_fail} smoke check(s) FAILED -- see [FAIL] lines above.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-gpt2-large", action="store_true")
    ap.add_argument("--pre-pass", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--i-am-the-t2a-execution-agent", action="store_true",
                     help="Required for --gate to actually run. This build session never passes it.")
    ap.add_argument("--witness", nargs="+", choices=sorted(WITNESS_SPECS),
                     default=None, help="default: W1_rwkv7 W2_gpt2large C1_falconmamba")
    ap.add_argument("--corpus", nargs="+", choices=sorted(CORPUS_DIRS) if CORPUS_DIRS else None,
                     default=None, help="default: both admissible corpora")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    ap.add_argument("--n-windows", type=int, default=N_ROWS_DEFAULT)
    ap.add_argument("--n-plants", type=int, default=N_PLANTS_DEFAULT)
    ap.add_argument("--n-source-tokens-train", type=int, default=N_SOURCE_TOKENS_TRAIN_DEFAULT)
    ap.add_argument("--n-source-tokens-val", type=int, default=N_SOURCE_TOKENS_VAL_DEFAULT)
    ap.add_argument("--eval-micro-batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        return smoke(device="cpu")
    if args.verify_gpt2_large:
        return mode_verify_gpt2_large(args)
    if args.pre_pass:
        return mode_pre_pass(args)
    if args.gate:
        return mode_gate(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
