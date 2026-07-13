#!/usr/bin/env python3
"""lm_recall_gap_probe_v2_rd.py -- PARAM_AXIS_SCALING_DESIGN.md sec 9 (REV 2)
instrument REBUILD, replacing the VOID `lm_recall_gap_probe_rd.py`
(matrix-thinking/queue/regate_2026-07-12.md sec 10: FATAL-1 shared-tensor
ablation, F-4 differential candidate cap, F-3 toothless Wave -1, S-6/S-7
provenance bugs, M-11 T2 weakened post-failure). That file is NOT deleted
(it is the artifact of record for the build+audit chain) but must NEVER be
imported by anything that wants a verdict-grade number. THIS file supersedes
it per sec 9's pinned re-registration.

WHAT CHANGED, one line each (full rationale in sec 9 of the design doc):
  1. ONE corrupted token per forward-pass ROW, via row-replication --
     batching over CANDIDATES, not over rows (sec 9.2). FATAL-1's fix.
  2. A PLACEBO-ABLATION arm (one non-antecedent token, Delta-distribution-
     matched, same replacement RNG stream) and DiD = gap_true - gap_placebo
     as the ONLY quantity that may carry a verdict (sec 9.2). No function in
     this module will hand back a raw AR-hit slice or an un-placebo'd gap as
     if it were a capacity measurement -- see `finalize_cell` / VerdictGradeError.
  3. T2 split into T2a (instrument-teeth, absolute bar, reference models --
     wired but NOT executed in this build session, sec 9.4) and T2b
     (rung-admissibility, mechanism-present test T2b-1 + the NEW ceiling
     check T2b-2: DiD(r) <= acc_copy(r) + 2*SE -- the single line that would
     have caught the VOID build's wikitext self-contradiction, regate 10.2).
  4. The eval micro-batch is fully decoupled from the token-arithmetic
     batch_size and from candidate selection (kills F-4: the old per-BATCH
     candidate cap silently made the batch-16 1.31B rung the only uncapped
     cell).
  5. Checkpoints must be QUIESCED (md5-stable across a wait window) before
     they may be loaded for a verdict-grade read (kills S-7: a prior run
     raced a live --ckpt-every writer).

WHAT DID NOT CHANGE (sec 9.0: "read off the (VOID) implementation's
candidate construction, which is correct and is retained"):
  - The candidate definition (second occurrence of a bigram whose first
    occurrence is >min_sep tokens back, and whose continuation is NOT the
    corpus TRAIN-split modal continuation).
  - The FIX-A token-matched-checkpoint-slice resolver (`resolve_token_matched_
    checkpoint` / `list_checkpoint_steps`) -- reused VERBATIM, unmodified,
    per this build's explicit instruction. It was never named as defective.
  - The T2 marker-token picker (`pick_t2_marker_tokens`, high-entropy /
    OOD-pairing selection) -- reused verbatim; only the DISTANCE and BAR it
    fed into were the M-11 violation, not the token-selection logic itself.

THE NORMALIZATION SLOT (sec 9.1) -- PINNED 2026-07-12 (commit 7381d2a, blind
agent, post-quarantine): M(r) = DiD(r), denominator 1 -- FORCED by N1-N3's
algebraic identity (sec 9.1.1: the intact arm cancels exactly, so DiD is
already competence-invariant and any denominator that varies with general
competence would re-inject the exact confound the numerator purged).
Registered here as `"raw_did"`, in the exact form the pin's own text
specifies: `register_normalization("raw_did", lambda cell: cell["did"])`.
This was NOT this build's decision to make -- this build's earlier session
read regate sec 10.2 and was therefore contaminated for *which* form leans
which way, so the slot originally shipped EMPTY on purpose. It is filled now
only because a SEPARATE, genuinely blind agent derived the forced answer
independently (sec 9.1.1) and the coordinator directed it be wired in.
`NORMALIZATION_REGISTRY` is still a live, general mechanism -- it is not
special-cased or hardcoded away: `register_normalization` and
`compute_capacity_metric`'s registry lookup are UNCHANGED by the pin
landing, so a future metric (or a PI override) can still be added the same
way. `normalization` remains a REQUIRED keyword with no default; passing
`None` or an unregistered name still raises loudly, and nothing here ever
silently falls back to a particular form.

sec 9.1.5 MANDATORY SENSITIVITIES (verdict-WITHHOLDING only -- can only
downgrade a verdict to INDETERMINATE, never create or strengthen one):
  S1 `compute_s1_utilization(cell)` -- the utilization ratio DiD/acc_copy
     (rejected normalization candidate C5, promoted to a companion metric).
     Never fed into compute_verdict/compute_capacity_metric.
  S2 `DiD_logp` -- log-prob readout guarding the argmax-floor bias. BUILD
     REQUIREMENT: every candidate record carries `logp_intact`,
     `logp_true_ablated`, `logp_placebo_ablated` (log_softmax+gather on the
     SAME forward passes already run -- zero extra compute). A cell missing
     any of these on a resolved candidate is VOID (`cell_void_missing_s2_
     fields`, enforced in `compute_capacity_metric`) -- sec 9.6's stop rule:
     S2 "cannot be added after a read." `compute_verdict_with_s2` enforces
     the pre-committed disagreement rule: if S2's Factor-1 trend classification
     differs from the primary's, the verdict is forced to INDETERMINATE.

Zero training. Reuses DeltaNetLM / load_corpus / get_batch / CORPUS_DIRS /
EOT_TOKEN_ID / corpus_fixed_seed / DEFAULT_DATA_DIR from lm_pretrain_rd.py
(VERBATIM import, not reimplemented) via h2h_fla_stub_rd.ensure_fla_stub()
(this repo's own CPU-stub-or-real-fla convention -- lets `--smoke` run
logic-only on a CPU dev box while `--run` on the H100 box picks up real fla
automatically).

Usage:
  python lm_recall_gap_probe_v2_rd.py --smoke
  python lm_recall_gap_probe_v2_rd.py --resolve-slice --ckpt-dir <dir> \
      --batch-size 32 --seq-len 512 --target-tokens 327680000 --prefix <p>
  python lm_recall_gap_probe_v2_rd.py --verify-quiesced --checkpoint <path.pt>
  python lm_recall_gap_probe_v2_rd.py --run --checkpoint <path.pt> \
      --corpus openr1-mix-ext --data-dir /data/deltanet_rd_data \
      --attest-job-terminated --out r.json
      (GPU + real fla required. Computes DiD/T1a ONLY -- NOT a verdict, and NOT
      T2b/sec-9.6 admissibility, which need the driver that sec 9.1's pin unblocks.
      `--attest-job-terminated` is REQUIRED: file-quiescence cannot establish sec 9.6
      item 3's second clause, because a --ckpt-every-10000 writer leaves a byte-stable
      file for HOURS between writes -- that is the S-7 read. `--compute-verdict` is
      REFUSED by this entry point.)

AUDIT (2026-07-12, independent adversarial audit of this rebuild). Fixed in place:
  * F-4 REBORN (was FATAL, latent): run_did_eval drew its windows in `batch_size`-sized
    get_batch chunks. On CUDA (Philox advances per KERNEL LAUNCH) that makes the WINDOW
    POPULATION batch-size-dependent -- measured on the H100: batch_size 16 vs 32 at the
    same seed shared only 50% of their 512 windows. The rungs use different
    token-arithmetic batch sizes (three at 32, the 1.31B at 16), so the top rung would
    again have been measured against a different candidate population. Now: one
    get_batch call; `batch_size` is provenance-only. Guarded by smoke [10c] (teeth on
    CUDA only) + a forced-fail negative test.
  * The FATAL-1 invariant is now ASSERTED ON THE PRODUCTION PATH (run_ablation_arm), not
    only on `build_replicated_ablation_batch`, which the real path never calls.
  * A placebo-match failure (>5% flagged) now maps to VOID, not FLOOR (sec 9.2 is literal).
  * compute_capacity_metric now REQUIRES `t2b2_pass=` and refuses placebo-unmatched cells;
    the `_verdict_grade` stamp only ever meant "a placebo arm was run."
  * tost_flat returned an 80% CI while claiming (and needing) the pinned 90% CI -- FLAT was
    declared more easily than the pre-registration permits.
  * `wilson_se` was Wald, not Wilson (behaviour right, name wrong). Renamed `binomial_se`.
  * The realized placebo Delta profile is now REPORTED (`delta_match`); it is ~35% shorter
    than the true profile by construction (draw-then-reject truncates at k) and nothing
    measured it before.

AUDIT 2 (2026-07-12, independent follow-up audit, scoped to the sec 9.1/9.1.5 S1+S2+
normalization code that landed AFTER the audit above). The S2 log-prob ARITHMETIC and the
candidate-detection REORDER both verified CLEAN against hand-built ground truth (owner-map
skips, cross-chunk indexing, k-vs-k+1 target, batch-size and eval-micro-batch invariance).
Fixed in place:
  * S2's log_softmax was run on the FULL (n_chunk, T, V) logits tensor and then read ONE
    element per row -- DOUBLING the peak VRAM of the pipeline's dominant allocation (measured
    on the H100 at the pinned eval_micro_batch=64/T=512/V=50257/fp32: 6.14 GiB -> 12.27 GiB)
    to keep 1/512 of it. Now gathers the query positions FIRST and log_softmaxes the
    (n_cand, V) slice: 6.16 GiB, identical arithmetic (max |diff| 9.5e-7, fp32 reduction-order
    noise). "Zero extra compute" was true; "zero extra memory" was not, and the top rung is
    exactly where that bites.
  * sec 9.1.5's S2 disagreement rule was enforced ONLY in the `compute_verdict_with_s2`
    wrapper, while `compute_verdict` -- the name sec 9.5's own prose uses -- stayed callable
    with NO S2 disposition and would hand back a full COUPLED/DECOUPLED/FLAT-COUPLED verdict
    with S2 silently skipped. The pin demands the rule be "structurally true, not just a
    convention someone remembers to apply." `factor1_trend_s2` is now a REQUIRED keyword-only
    argument OF `compute_verdict` itself; the rule is enforced there; a stale S2-less call
    raises TypeError. Guarded by smoke [9b] (negative test + the full 4x4 grid, both entry
    points).
  * `register_normalization` would SILENTLY REBIND the pinned "raw_did" to any other function.
    It now refuses (override=True required); re-registering the same object stays a no-op.
  * `--normalization` was parsed and then never read -- silently ignored on every --run. Now
    refused out loud (mode_run cannot apply a normalization at all, by construction).
  * DOC: sec 9.1.5's "S1 is bounded to [0,1]" is not exactly true (T2b-2 admits
    DiD <= acc_copy + 2*SE, so S1 can reach 1 + 2*SE/acc_copy > 1 on a PASSING rung; and DiD<0
    makes it negative). Corrected in `compute_s1_utilization`'s docstring. Zero verdict
    consequence -- S1 is structurally verdict-withholding -- so it is a note, not a defect.
    NOTE (superseded by REV 3 below): the "T2b-2 admits DiD <= acc_copy + 2*SE" bound this
    note relies on is WITHDRAWN with T2b-2's retirement (sec 11.6). S1 is now reported as an
    unbounded ratio; see compute_s1_utilization's current docstring.

REV 3 BUILD (2026-07-12) -- THE sec 11 T2 REPAIR, implementing
PARAM_AXIS_SCALING_DESIGN.md sec 11 (pinned, post independent-attack-round
3-FATAL/9-SERIOUS, all conceded and fixed in the design doc itself before
this build started -- see sec 11.9). R0 (855f548) returned VOID: T2a, the
never-executed instrument-teeth gate, FAILED on BOTH reference models
because `pick_t2_marker_tokens` had TWO independent defects (sec 11.1):
F-I the key was drawn from the 400 MOST-FREQUENT tokens (so the plant
competed with ~20 natural same-key instances per window) and F-II the
picker REQUIRED the (key,value) pair to never co-occur in train (which
selects for associations the model cannot emit -- an unemittable value is
not merely hard, it can be impossible). Both are fixed AT ONCE (sec 11.2):
the key is now rare-in-window-but-well-attested (K1-K5), the value is
attested->licensed-but-not-predicted (V1-V5), and a per-window HARD
RuntimeError assertion (never a warning) makes the fix STRUCTURAL, not
statistical -- see PlantContestedError / plant_and_verify_t2_window and the
smoke gate's [FORCED-FAIL] block, which is RUN TO COMPLETION, not merely
written (CLAUDE.md: "a check you did not watch refuse has no teeth").

WHAT CHANGED, one line each (full rationale in sec 11 of the design doc):
  1. `pick_t2_marker_tokens` DELETED. Replaced by `build_key_value_pools`
     (sec 11.2.1/11.2.2 -- P_key/P_val/inverse-map, TRAIN-split corpus
     statistics ONLY, no model forward pass ever) + `assign_t2_plant`
     (sec 11.2.3 -- per-window Delta-by-rejection-sampling + a JOINTLY-drawn
     fresh (a,a',b) triple, seeded ONLY by (corpus, window index)) +
     `plant_and_verify_t2_window` (the hard RuntimeError assertion).
  2. `run_t2_planted_copy` (ONE global (tok_a,tok_b) pair reused everywhere,
     sec 11.2.5's "one unlucky pair zeroes an entire corpus column") is
     RETIRED. Replaced by `run_t2_repaired_probe`: a FRESH (a,a',b) per
     plant window, SIX arms (sec 11.3: INTACT, TRUE-ABLATED,
     PLACEBO-ABLATED, POOL-PLACEBO, KEY-SWAP, NO-PLANT), all via
     `assign_placebo_positions`/`run_ablation_arm`/row-replication
     VERBATIM (sec 11.3: "All six reuse ... verbatim").
  3. T2a RE-PINNED (sec 11.4): the 0.90/0.75 ceiling bar is UNCHANGED, plus
     THREE new simultaneous legs (PRIOR<=0.05, KS>=0.50 + T2b-1b, T2b-1),
     an untrained-negative-control gate (T2a-2), an SSM causal-only
     calibration gate (T2a-3), and T1c RE-PINNED as a reference-model DiD
     gate on the MAIN metric's own candidate population (sec 11.4.5) --
     see check_t2a1_ceiling / check_t2a2_untrained_control /
     check_t2a3_ssm_calibration / check_t1c_reference_did.
  4. T2b-2 (`check_t2b2_ceiling`) is RETIRED AND DELETED (sec 11.6: the
     domination it asserted is FALSE in general, for two independent
     structural reasons -- not merely unproven). `compute_capacity_metric`
     no longer takes (or checks) a `t2b2_pass` argument; `check_rung_
     admissible` no longer takes `t2b2_by_corpus`. NO REPLACEMENT CEILING
     IS INVENTED (sec 11.6: "No ceiling check replaces it"). The failure
     mode T2b-2 existed to catch is now guarded by THREE more direct
     mechanisms instead (sec 11.6.1): the FATAL-1 runtime assertion
     (already present, sec 9.2/AUDIT above), T2b-1 + the NEW T2b-1b
     (`check_t2b1b_key_conditioned`) excluding any rung without a
     KEY-CONDITIONED mechanism, and T2a-2's untrained negative control.
     sec 9.6 item 5 now reads "T1a and T2b-1 and T2b-1b all pass; failure
     of any => FLOOR rung" -- T2b-2's old VOID trigger is GONE; a rung that
     fails T1a/T2b-1/T2b-1b is FLOOR (excluded, reported), never VOID.
  5. sec 11.7's N_rows/candidate reconciliation: `resolve_n_rows_pre_pass`
     is a MODEL-FREE pre-pass (reads only corpus+tokenizer+TRAIN-split
     modal table) that fixes ONE N_rows (smallest power of two in
     [2048,8192] clearing >=1,500 contributing rows AND >=8,000 resolved
     candidates on BOTH corpora) used at EVERY rung -- superseding sec
     9.2's N_rows=512 and sec 9.6 item 7's >=4,096-candidate floor. The
     sec 9.2 placebo-fallback-flagged-fraction<=5% check is RETAINED as a
     per-cell pass/fail (REMOVED from the N_rows search itself, sec 11.7 /
     A-S8: it is non-monotone in N_rows and would make the search
     non-terminating).
  6. `emit_admissible_set_json` / `build_admissible_set_row` (sec 11.8.1):
     the admissible-set artifact -- booleans + metadata ONLY, MECHANICALLY
     scanned (`_assert_no_did_shaped_fields`) to refuse emission if any
     DiD/gap/acc_copy/S1/S2/logp-shaped field is present anywhere in the
     payload (KEYS *and* string VALUES scanned, camelCase/plural-aware
     tokenization, tuples traversed, ALL float leaves refused outright --
     hardened post-audit, see the AUDIT FIX block below). This module can
     therefore produce the artifact WITHOUT ever having computed a DiD
     value for the cells it describes.
  7. sec 11.6.2 ARM D + S3 (added post-audit -- see AUDIT FIX block below):
     `run_did_eval` now runs a FOURTH row-replicated arm (key-ablated,
     corrupts position `j`, leaves `j+1` intact) alongside the pre-existing
     TRUE/PLACEBO arms; `finalize_cell` emits `s3`/`s3_ci` (=E[C-D], the
     key-association component, placebo-controlled) and the UNLABELLED
     residual `E[D-B]` (never called "the salience component," per sec
     11.6.2's explicit instruction) -- both MANDATORY, REPORTED ALWAYS,
     VERDICT-CARRYING NEVER, exactly like S1/S2. `s3_claim_language`
     implements sec 11.6.2's pre-registered claim-language table
     mechanically (never a value a caller might silently reinterpret).

WHAT DID NOT CHANGE (per this build's explicit brief -- "PRESERVE, do not
touch, do not improve"): the DiD/placebo machinery's ESTIMAND and
NORMALIZATION (sec 9.1/9.2 -- `stat_did`, the clustered bootstrap, `_raw_did`
/ `NORMALIZATION_REGISTRY` are byte-identical; `run_did_eval`/`finalize_cell`
gained item 7's Arm D/S3 fields ADDITIVELY, nothing pre-existing was
altered) and T2b-1 (`check_t2b1_mechanism_exists`, now a thin wrapper over
the shared `_paired_sign_test` helper T2b-1b also uses -- the STATISTICAL
TEST ITSELF is byte-identical, sec 11.5: "T2b-1 -- unchanged"); S1/S2 as
VERDICT-WITHHOLDING-ONLY sensitivities; the per-candidate per-forward-pass
row-replication ablation (`build_replicated_ablation_batch` /
`run_ablation_arm`'s FATAL-1 assertion); the checkpoint-quiescence gate
(`verify_quiesced_checkpoint` / `load_checkpoint`); the FIX-A token-slice
resolver (`resolve_token_matched_checkpoint`).

SCOPE BOUNDARY, DISCLOSED (mirrors the existing T2a disclosure above; STATED
PRECISELY against sec 11.11's own 8-item build-gate list, not glossed as "1-6"
-- an earlier draft of this docstring made that overclaim and an independent
audit caught it (see the AUDIT FIX block below)): this build implements sec
11.11 items 1, 2, 3, 4, 6 IN FULL (code-level); item 5's PER-RECORD difficulty
fields (max_rival_p/rank/p(b|a)/count(a,b)/count(b)) ARE carried on every T2
record, but the DRIVER-LEVEL orchestration across multiple models/Delta
values (the stratified report, the W2 Delta-sweep, the n_demos diagnostic)
is NOT built here -- disclosed, driver-level, out of scope; items 7 (the D5
decode<->re-tokenize bridge audit) and 8's infra half (a gpt2-large
checkpoint deployed on a box) are GENUINELY out of scope for a code-only
build -- item 8's CODE half (the untrained-init negative control machinery,
T2a-2) IS implemented. This build does NOT execute T2a/T2a-2/T2a-3/T1c
against the actual reference models (RWKV7-Goose-1.5B, gpt2-large,
falcon-mamba-7b) -- that requires GPU + downloaded checkpoints and a driver
(still "not yet built", per this file's own mode_run docstring, unchanged by
this build) and is explicitly OUT OF SCOPE for this task (no R0 eval, no DiD
numbers, per the task's own instruction). Every T2a/T1c CHECK FUNCTION is
generic over its `records`/`cell` inputs and is exercised end-to-end against
the real DeltaNetLM class in the smoke gate (untrained, toy vocab --
plumbing, not a competence claim).

AUDIT (2026-07-12, independent fresh-context opus adversarial audit of this
REV 3 build). Verdict: the T2-probe subsystem (sec 11.2-11.6) matched the
pinned spec exactly and its two hardest structural claims (contested plants
are impossible; the six arms match sec 11.3's table against the real
tensors) survived direct attack -- "I could not break either." Found ONE
FATAL (sec 11.6.2 Arm D/S3 was entirely missing while this docstring falsely
claimed full sec 11.11 coverage -- item 7 above), THREE SERIOUS, SEVEN
MINOR. Fixed in place:
  * FATAL: Arm D + S3 built (item 7 above); the false "items 1-6" claim
    corrected to name exactly what is and is not covered.
  * SERIOUS: the three sec 11.2 pool floors (|P_key|>=100, |P_val(a)|>=5,
    >=100 licensed-b's) were computed but NEVER CHECKED anywhere outside the
    smoke gate -- "the floors are GATES, not hopes" (sec 11.2) had no VOID
    exit. `run_t2_repaired_probe` now VOIDs on any floor miss BEFORE
    sampling a single plant. `floor_pool_val` was also vacuously True on an
    EMPTY pool (`all()` over zero items) -- fixed to require a non-empty
    pool.
  * SERIOUS: `_assert_no_did_shaped_fields` scanned dict KEYS only, using a
    tokenizer blind to camelCase/plurals, and did not traverse tuples --
    `{"note": "DiD 0.19 ..."}`, `gapTrue`, `logprob`, and a tuple-nested DiD
    value all evaded it. Hardened: camelCase-aware tokenization, plural
    stripping, string VALUES scanned with the same tokenizer, tuples
    traversed, and (the robust fix, since a sufficiently creative field name
    can always evade a keyword list) EVERY float leaf anywhere in the
    payload is refused outright -- this artifact has no legitimate reason to
    carry a float (DiD/gap/acc_copy/S1/S2/logp are ALL continuous-valued;
    booleans/ints/strings are not).
  * SERIOUS: `run_ablation_arm`'s FATAL-1 invariant assertion
    (`(row!=original).sum()==1`) had no FORCED-FAIL negative test in the
    smoke gate (only `build_replicated_ablation_batch`, which the real path
    never calls, had one -- itself flagged as F-3-class toothless by an
    earlier audit note in this file). Added: a smoke-gate case that supplies
    a replacement token EQUAL to the position's current value (a legitimate
    call through the public `run_ablation_arm` API, no monkeypatching),
    producing zero corrupted tokens and confirming the assertion fires.
  * MINOR: arm 3b's (POOL-PLACEBO) replacement-token exclusion set did not
    exclude `{a, a'}`, so it could rarely re-splice the plant's own key back
    in (~0.4% of rows at real pool sizes) -- fixed, and conservative in
    direction regardless (depresses arm 3b, which cannot inflate a T2b-1b
    pass).
  * MINOR: `delta_excluded_frac`'s per-window mean was a biased estimator of
    pooled excluded mass (a geometric-rejection artifact; true 5% measured
    as ~2.4%) -- now pools `sum(rejected)/sum(tries)` across all plants.
  * MINOR: `torch.argsort` on tied bigram counts used the non-stable
    default, which can (at ties, common at V3's count>=5 floor) admit a
    value tied with the modal continuation into rank 2 -- switched to
    `stable=True`.
  * MINOR: Delta=0 (j0==k0) was accepted by the hard assertion, since
    `sorted({j0,k0})` silently collapses a duplicate to one element --
    unreachable from a real candidate pool (min_sep=2 forces Delta>=2) but a
    latent hole in the one check the whole repair rests on; `assign_t2_plant`
    now requires `delta >= 2` explicitly.
  * MINOR (recorded, NOT patched -- see the function's own docstring): sec
    11.2.3's literal formula "j0 = k0 - Delta" is off by one token against
    sec 9.0's own Delta definition (k-(j+1)) -- the AUDIT independently
    verified this is a property of the PINNED FORMULA, not a bug introduced
    here, and that patching it would itself be an unauthorized deviation
    from a pinned procedure; it touches no gate threshold (the Delta-decile
    bucketing is invariant under a uniform shift) and is disclosed here
    rather than silently carried.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # pod-safe imports
sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

from h2h_fla_stub_rd import ensure_fla_stub  # noqa: E402
ensure_fla_stub()   # real fla on the H100 box; CPU stub everywhere else / when forced

from lm_pretrain_rd import (   # noqa: E402
    DeltaNetLM, load_corpus, get_batch, CORPUS_DIRS, EOT_TOKEN_ID,
    DEFAULT_DATA_DIR, corpus_fixed_seed,
)

VOCAB_SIZE = 50257  # GPT-2, asserted by load_corpus's own meta.json check

# sec 9.6 item 1: the common token slice is FORCED at 0.328B (14M's own final
# checkpoint step*batch*seq_len) -- the design's original 1.0B-slice dual
# cross-validation is RETRACTED as unrunnable (regate S-6: those checkpoints
# do not exist on disk). Do not "improve" this back to 1.0B.
TARGET_TOKENS_FORCED_SLICE = 327_680_000

# sec 9.2 "Cost" pinned sampling constants.
# REV 3 (sec 11.7, 2026-07-12): N_ROWS_DEFAULT is RATIFIED at 2048 (R0's own disclosed D1
# deviation), superseding sec 9.2's original 512 -- resolve_n_rows_pre_pass (sec 11 block,
# below) is the AUTHORITATIVE way to fix N_rows for a real read; this constant is now only
# the CLI's --n-windows default and the smoke gate's own default.
N_ROWS_DEFAULT = 2048
C_MAX_DEFAULT = 8            # per-ROW, fixed, rung-invariant (kills F-4) -- UNCHANGED, sec 11.7
EVAL_MICRO_BATCH_DEFAULT = 64  # decoupled from token-arithmetic batch_size (kills F-4 pt. 4)
PLACEBO_FLAGGED_FRAC_VOID_THRESHOLD = 0.05   # sec 9.2 -- UNCHANGED; sec 11.7 REMOVES it from the
                                               # N_rows SEARCH (A-S8: non-monotone in N_rows) but
                                               # KEEPS it as the per-cell pass/fail gate it always was.
TOK_PER_PARAM_PRIMARY_FIT_FLOOR = 1.0        # sec 9.6 item 2

# sec 11.7 -- N_ROWS / CANDIDATE-FLOOR RECONCILIATION, PINNED. Supersedes sec 9.2's
# N_rows=512 fixed value AND sec 9.6 item 7's >=4,096-candidate floor (512x8=4096 was the
# UNREACHABLE theoretical maximum -- the floor was stated in the wrong unit, sec 11.7).
N_ROWS_SEARCH_LADDER = (2048, 4096, 8192)     # "the smallest power of two in [2048,8192]"
ROW_FLOOR_CONTRIBUTING = 1500                  # ">= 1,500 rows contributing >=1 resolved candidate"
CANDIDATE_FLOOR_RESOLVED = 8000                # ">= 8,000 resolved candidates"


# =============================================================================
# 0. Deterministic seed combination (NEVER Python's hash() -- salted per
#    process; this file follows lm_pretrain_rd.py's own corpus_fixed_seed
#    rationale for exactly the same reason: reproducibility across runs).
# =============================================================================
def _combine_seed(*parts) -> int:
    s = ":".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) & 0x7FFFFFFF


# =============================================================================
# 1. FIX-A -- common-token-count checkpoint slice resolver.
#    REUSED VERBATIM from lm_recall_gap_probe_rd.py (regate: never named
#    defective). Do not rewrite; sec 9.6 item 1 depends on this exact logic.
# =============================================================================
_STEP_RE = re.compile(r"_step(\d+)\.pt$")


def list_checkpoint_steps(ckpt_dir: str) -> list:
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        m = _STEP_RE.search(f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def resolve_token_matched_checkpoint(ckpt_dir: str, batch_size: int, seq_len: int,
                                      target_tokens: int, prefix: str) -> dict:
    """Picks the on-disk step whose token count (step*batch_size*seq_len) is
    CLOSEST to target_tokens (ties broken toward the lower step, i.e. never
    overshoot on a tie). Returns a dict recording the chosen step, its exact
    token count, the signed miss, and the full path -- never silently picks
    "final checkpoint"."""
    steps = list_checkpoint_steps(ckpt_dir)
    if not steps:
        return {"ckpt_dir": ckpt_dir, "error": "NO_CHECKPOINTS_FOUND"}
    best = None
    for s in steps:
        tok = s * batch_size * seq_len
        miss = tok - target_tokens
        key = (abs(miss), miss > 0)  # prefer closer, then prefer under-shoot on ties
        if best is None or key < best[0]:
            best = (key, s, tok, miss)
    _, step, tok, miss = best
    path = os.path.join(ckpt_dir, f"{prefix}_step{step}.pt")
    assert os.path.isfile(path), f"resolved step {step} but file missing: {path}"
    return {
        "ckpt_dir": ckpt_dir, "batch_size": batch_size, "seq_len": seq_len,
        "target_tokens": target_tokens, "chosen_step": step,
        "chosen_tokens": tok, "miss_tokens": miss,
        "miss_frac": miss / target_tokens if target_tokens else None,
        "n_steps_available": len(steps), "min_step": steps[0], "max_step": steps[-1],
        "path": path,
    }


# =============================================================================
# 2. Checkpoint quiescence + md5 pinning (sec 9.6 item 3, kills S-7).
# =============================================================================
def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_quiesced_checkpoint(path: str, wait_s: float = 5.0) -> dict:
    """sec 9.6 item 3, FIRST HALF ONLY: detects a checkpoint FILE that is
    actively being written. Compares (size, mtime, md5) before/after a wait
    window; a writer mid --ckpt-every-N write changes at least one. `md5` is
    populated ONLY when quiesced=True -- it is the value sec 9.6 requires to
    be "md5-pinned in the result JSON," and pinning an md5 taken WHILE a file
    is changing would be meaningless.

    *** THIS CHECK IS NECESSARY BUT NOT SUFFICIENT, AND ON ITS OWN IT WOULD NOT HAVE
    CAUGHT S-7. *** sec 9.6 item 3 has two clauses: the file must be quiesced AND "its job
    must be terminated." File-stability proves only the former. The S-7 precedent was a
    read of a LIVE 1.31B job's step-40000 checkpoint under a `--ckpt-every 10000` writer
    -- at ~1.4 s/step those writes are ~3.9 HOURS apart, so that file was perfectly
    byte-stable across any 5-second window while the job was still training. A live job
    passes this check for hours at a time, by construction. The "job terminated" clause is
    therefore enforced separately, as an explicit operator attestation recorded in the
    result JSON (`--attest-job-terminated`, stamped as `job_terminated_attested`), because
    nothing this process can measure about the FILE can establish it.
    (AUDIT NOTE 2026-07-12: an atomic-rename writer -- torch.save to a tmp path then
    os.replace -- is actually SAFE here: the visible file is always a complete older or
    newer checkpoint, and a mid-window swap changes size/mtime/md5 and is caught. The
    residual false-PASS is a writer that CRASHED mid-write, leaving a stable truncated
    file; that is not silently accepted -- torch.load raises on it.)"""
    st0 = os.stat(path)
    md5_0 = md5_file(path)
    time.sleep(wait_s)
    st1 = os.stat(path)
    md5_1 = md5_file(path)
    quiesced = (st0.st_size == st1.st_size) and (st0.st_mtime == st1.st_mtime) and (md5_0 == md5_1)
    return {
        "path": path, "quiesced": quiesced, "md5": md5_1 if quiesced else None,
        "size0": st0.st_size, "size1": st1.st_size,
        "mtime0": st0.st_mtime, "mtime1": st1.st_mtime,
        "md5_0": md5_0, "md5_1": md5_1, "wait_s": wait_s,
    }


class CheckpointNotQuiescedError(RuntimeError):
    pass


def load_checkpoint(path: str, device: str, require_quiesced: bool = True,
                     wait_s: float = 5.0) -> tuple:
    """sec 9.6 item 3, non-negotiable for any verdict-grade read: refuses to
    load (raises CheckpointNotQuiescedError) a checkpoint that fails the
    quiescence check.

    AUDIT CORRECTION (2026-07-12): the previous docstring claimed
    `require_quiesced=False` "exists ONLY for the smoke gate's own synthetic fixtures ...
    every real-run code path in this module leaves it at the default True." **That was
    false.** The smoke gate never calls load_checkpoint at all, and `--no-quiesce-check`
    wires straight into the `--run` path -- i.e. the ONLY caller of the escape hatch is
    the real, verdict-grade run. It is retained (a legitimately-terminated job whose
    filesystem reports jittery mtimes is a real scenario) but `mode_run` now STAMPS
    `quiesce_check_skipped: true` into the result JSON, so a skipped guard can never be
    invisible to a downstream reader, and the md5 pin sec 9.6 requires is `None` in that
    case -- which by itself makes the cell non-conformant."""
    md5_pin = None
    if require_quiesced:
        q = verify_quiesced_checkpoint(path, wait_s=wait_s)
        if not q["quiesced"]:
            raise CheckpointNotQuiescedError(
                f"CHECKPOINT NOT QUIESCED: {path} changed during a {wait_s}s stability "
                f"window (size {q['size0']}->{q['size1']}, md5 {q['md5_0']}->{q['md5_1']}). "
                f"A rung may NOT be read from a live, still-training job "
                f"(PARAM_AXIS_SCALING_DESIGN.md sec 9.6 item 3; regate_2026-07-12.md sec 10.3 "
                f"S-7 precedent: a prior run raced a live --ckpt-every writer). Retry once the "
                f"writer has terminated."
            )
        md5_pin = q["md5"]
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, md5_pin


# =============================================================================
# 3. Modal-continuation table -- REUSED VERBATIM (sound per sec 9.0).
# =============================================================================
def build_bigram_mode_table(train_tokens: torch.Tensor, vocab_size: int, device: str) -> torch.Tensor:
    """mode_next[a] = the most frequent b immediately following a in the
    TRAIN split (ties broken toward smaller b, deterministic); -1 if a
    never occurred followed by anything."""
    t = train_tokens.to(device)
    a = t[:-1].to(torch.int64)
    b = t[1:].to(torch.int64)
    pair_ids = a * vocab_size + b
    uniq, counts = torch.unique(pair_ids, return_counts=True)
    a_idx = uniq // vocab_size
    b_idx = uniq % vocab_size
    max_count_per_a = torch.full((vocab_size,), -1.0, device=device)
    max_count_per_a.scatter_reduce_(0, a_idx, counts.to(torch.float64).to(torch.float32),
                                     reduce="amax", include_self=True)
    is_max = counts.to(torch.float32) >= max_count_per_a[a_idx]
    b_where_max = torch.where(is_max, b_idx, torch.full_like(b_idx, vocab_size))
    mode_next = torch.full((vocab_size,), vocab_size, dtype=torch.int64, device=device)
    mode_next.scatter_reduce_(0, a_idx, b_where_max, reduce="amin", include_self=True)
    mode_next[mode_next == vocab_size] = -1
    return mode_next.cpu()


def _make_window(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y[:, -1:]], dim=1)


# =============================================================================
# 4. Candidate + baseline detection (sec 9.0's construction, retained;
#    UNCAPPED here -- capping is a separate, rung-independent step, sec 9.2
#    "kills F-4").
# =============================================================================
def detect_candidates_and_baseline(x_cpu: list, y_cpu: list, mode_next_cpu: list,
                                    min_sep: int) -> tuple:
    """Per-row scan implementing sec 9.0's candidate construction; returns candidates
    UNCAPPED, one list per row -- capping happens in `select_candidates_per_row`,
    decoupled from batch size entirely.

    AUDIT CORRECTION (2026-07-12): this is NOT, as the previous docstring claimed, an
    exact mirror of the VOID build's detection loop. There is one real behavioural
    difference, and it is deliberate: the old loop did `seen[key] = k` when a repeat fell
    WITHIN min_sep, advancing the antecedent pointer to the later occurrence; this one
    never advances it, so `j` is always the FIRST occurrence. sec 9.0 is explicit -- "the
    antecedent is the single token at position j+1 -- the continuation token of the FIRST
    occurrence" -- so the new behaviour is the spec-faithful one and the old was subtly
    off. It affects only bigrams that repeat within min_sep tokens, applies identically at
    every rung (so it cannot bias a cross-rung comparison), and yields slightly MORE
    candidates. Recorded rather than silently carried.

    Returns (row_candidates, row_baseline) where
    row_candidates[b] = [(k, j), ...] and row_baseline[b] = [k, ...]
    (ordinary first-occurrence positions, subsampled every 7th, for the
    non-AR baseline slice)."""
    B = len(x_cpu)
    row_candidates = []
    row_baseline = []
    for b in range(B):
        xb, yb = x_cpu[b], y_cpu[b]
        T = len(xb)
        seen = {}
        cands = []
        baseline = []
        for k in range(T):
            a, bb = xb[k], yb[k]
            if a == EOT_TOKEN_ID or bb == EOT_TOKEN_ID:
                continue
            key = (a, bb)
            if key in seen:
                j = seen[key]
                if k - j > min_sep:
                    modal = mode_next_cpu[a]
                    if modal != bb:
                        cands.append((k, j))
                continue
            seen[key] = k
            if k % 7 == 0:
                baseline.append(k)
        row_candidates.append(cands)
        row_baseline.append(baseline)
    return row_candidates, row_baseline


def select_candidates_per_row(row_candidates: list, c_max: int, seed: int) -> list:
    """sec 9.2 Cost: C_max candidates per row, uniform random WITHIN the row,
    seeded ONLY by (seed, row index) -- never by rung/params/batch_size. Since
    every rung samples the identical window sequence (corpus_fixed_seed-based
    generator, same corpus), row_candidates[i] is IDENTICAL across rungs
    before this call; this function then subsamples the SAME candidates at
    every rung. This is the direct fix for F-4 (the old per-batch cap that
    silently made one rung's population differ from the other three's)."""
    out = []
    for i, cands in enumerate(row_candidates):
        if len(cands) <= c_max:
            out.append(list(cands))
            continue
        rng = random.Random(_combine_seed(seed, "select", i))
        idxs = sorted(rng.sample(range(len(cands)), c_max))
        out.append([cands[j] for j in idxs])
    return out


# =============================================================================
# 5. The TRUE-arm candidate spec + PLACEBO position assignment (sec 9.2).
# =============================================================================
def true_arm_specs(sel_row_candidates: list) -> list:
    """Flattens per-row selected candidates into a flat, stable-ordered list
    of dicts: {row_idx, k, j, p(=j+1), delta(=k-p)} -- the TRUE arm's spec."""
    specs = []
    for row_idx, cands in enumerate(sel_row_candidates):
        for (k, j) in cands:
            p = j + 1
            specs.append({"row_idx": row_idx, "k": k, "j": j, "p": p, "delta": k - p})
    return specs


def assign_placebo_positions(specs: list, sel_row_candidates: list, x_cpu: list,
                              seed: int, max_redraws: int = 100) -> list:
    """sec 9.2: for each candidate, draw a placebo position p' at a distance
    Delta' resampled (with replacement, fixed seed) from the POOLED empirical
    distribution of Delta over `specs` -- per-candidate EXACT distance
    matching is impossible by construction (same distance => same position),
    so the match is distributional, exactly as pinned. Admissibility: p' is
    rejected/redrawn if it lands on p, on j (the antecedent's key token), on
    k or later, on an EOT, or on the antecedent position of ANY OTHER
    candidate in the same row. Cap 100 redraws; fall back to a uniform draw
    over admissible positions and set `placebo_flagged=True`. Mutates and
    returns `specs` with `p_placebo`/`delta_placebo`/`placebo_flagged` added
    (p_placebo=None iff even the fallback found no admissible position)."""
    deltas_pool = [s["delta"] for s in specs]
    if not deltas_pool:
        return specs
    resample_rng = random.Random(_combine_seed(seed, "placebo_delta_pool"))
    # per-row forbidden sets: every OTHER candidate's antecedent position (j+1) in that row
    forbidden_by_row = {}
    for row_idx, cands in enumerate(sel_row_candidates):
        forbidden_by_row[row_idx] = {j + 1 for (_, j) in cands}

    for idx, s in enumerate(specs):
        row_idx, k, p, j = s["row_idx"], s["k"], s["p"], s["j"]
        xb = x_cpu[row_idx]
        forbidden = set(forbidden_by_row[row_idx])
        forbidden.add(p)
        forbidden.add(j)
        p_placebo = None
        delta_placebo = None
        flagged = False
        for _attempt in range(max_redraws):
            delta_p = resample_rng.choice(deltas_pool)
            cand_p = k - delta_p
            if cand_p < 0 or cand_p >= k:
                continue
            if cand_p in forbidden:
                continue
            if xb[cand_p] == EOT_TOKEN_ID:
                continue
            p_placebo, delta_placebo = cand_p, delta_p
            break
        if p_placebo is None:
            admissible = [pos for pos in range(0, k)
                          if pos not in forbidden and xb[pos] != EOT_TOKEN_ID]
            flagged = True
            if admissible:
                fallback_rng = random.Random(_combine_seed(seed, "placebo_fallback", idx))
                p_placebo = fallback_rng.choice(admissible)
                delta_placebo = k - p_placebo
        specs[idx] = {**s, "p_placebo": p_placebo, "delta_placebo": delta_placebo,
                      "placebo_flagged": flagged}
    return specs


# =============================================================================
# 6. Row-replication ablation -- ONE corrupted token per forward-pass ROW.
#    This is the FATAL-1 fix. `build_replicated_ablation_batch` is the unit
#    the smoke gate's "one-token-diff" test targets directly.
# =============================================================================
def draw_exclusive_replacement(window_row: list, pos: int, target: int, vocab_size: int,
                                rng: random.Random) -> int:
    """Uniform draw excluding {the token currently at `pos`, `target`, EOT} --
    same rule for TRUE and PLACEBO arms, and both draw from the SAME `rng`
    stream (continuing it, not reseeding) per sec 9.2 ('same replacement
    RNG')."""
    true_next = window_row[pos]
    while True:
        repl = rng.randrange(vocab_size)
        if repl != true_next and repl != target and repl != EOT_TOKEN_ID:
            return repl


def build_replicated_ablation_batch(window_row: torch.Tensor, corrupt_positions: list,
                                     replacement_tokens: list) -> torch.Tensor:
    """THE FATAL-1 FIX, as its own unit: given ONE window row (length T+1,
    the `_make_window` convention) and M (position, token) corruptions,
    returns an (M, T+1) tensor where row m is an EXACT COPY of window_row
    except position corrupt_positions[m] is replaced by replacement_tokens[m].
    Every OUTPUT ROW THEREFORE CARRIES EXACTLY ONE CORRUPTED TOKEN relative to
    window_row -- never a shared tensor with every candidate's antecedent
    corrupted simultaneously (regate FATAL-1). `smoke()`'s unit test asserts
    this directly on the returned tensor's Hamming distance to the base row."""
    M = len(corrupt_positions)
    assert M == len(replacement_tokens)
    rep = window_row.unsqueeze(0).repeat(M, 1).clone()
    if M:
        row_idx_t = torch.arange(M, device=window_row.device)
        pos_t = torch.tensor(corrupt_positions, device=window_row.device, dtype=torch.long)
        tok_t = torch.tensor(replacement_tokens, device=window_row.device, dtype=torch.long)
        rep[row_idx_t, pos_t] = tok_t
    return rep


def run_ablation_arm(model, all_windows: list, specs: list, pos_key: str, repl_key: str,
                      pred_key: str, eval_micro_batch: int, device: str,
                      logp_key: str = None) -> None:
    """Builds ONE replicated row PER CANDIDATE (batching over candidates, not
    rows, per sec 9.2's exact prescription -- a candidate's replicated row is
    a full clone of its source window with exactly its own single corruption
    applied; different candidates from the SAME source row each get their own
    independent clone, so no two candidates' corruptions ever land in the
    same forward-pass row). Runs the whole candidate population's replicated
    rows through the model in `eval_micro_batch`-sized chunks (fully
    decoupled from the token-arithmetic batch_size -- kills F-4 pt 4) and
    writes each spec's predicted token (argmax at its own query position k)
    into `spec[pred_key]`. Candidates whose `pos_key` is None (an unresolved
    placebo fallback) are skipped entirely -- never silently defaulted.

    sec 9.1.5 S2 BUILD REQUIREMENT: if `logp_key` is given, also writes
    `spec[logp_key] = log p(spec['target'])` under this arm -- a
    `log_softmax` + `gather` on the SAME logits this function already
    computes for the argmax read, zero extra forward passes. Requires
    `spec['target']` to already be set (true for every caller in this
    module)."""
    big_rows = []
    owner = []
    for i, s in enumerate(specs):
        pos = s.get(pos_key)
        if pos is None:
            continue
        base = all_windows[s["row_idx"]]
        rep = base.clone()
        rep[pos] = s[repl_key]
        big_rows.append(rep)
        owner.append(i)
    if not big_rows:
        return
    big = torch.stack(big_rows, dim=0)

    # AUDIT FIX (2026-07-12): ENFORCE THE FATAL-1 INVARIANT ON THE PRODUCTION PATH.
    # The smoke gate asserted "exactly one corrupted token per row" against
    # `build_replicated_ablation_batch` -- a function this, the real code path, never
    # calls. That is an F-3-class toothless test (a gate that passes without exercising
    # the thing it certifies). This assertion checks the invariant on the ACTUAL tensor
    # about to be fed to the model. It is O(M*T) elementwise on an (<=4096, 513) tensor
    # -- negligible next to the forward passes -- and it is the single line whose failure
    # means the instrument has regressed to the VOID build's defect.
    base_stack = torch.stack([all_windows[specs[i]["row_idx"]] for i in owner], dim=0)
    n_corrupted = (big != base_stack).sum(dim=1)
    if not bool((n_corrupted == 1).all()):
        bad = (n_corrupted != 1).nonzero().flatten().tolist()[:8]
        raise AssertionError(
            f"FATAL-1 INVARIANT VIOLATED in run_ablation_arm({pos_key!r}): "
            f"{int((n_corrupted != 1).sum())} of {big.shape[0]} forward-pass rows carry "
            f"!= 1 corrupted token (offending row idxs {bad}, counts "
            f"{n_corrupted[bad].tolist() if bad else []}). Every row MUST differ from its "
            f"source window at EXACTLY the one position belonging to its own candidate "
            f"(PARAM_AXIS_SCALING_DESIGN.md sec 9.2: 'Every forward pass carries exactly "
            f"one corrupted token. This is non-negotiable.'; regate_2026-07-12.md sec 10.2 "
            f"FATAL-1 is the realized failure of exactly this invariant)."
        )

    preds_chunks = []
    for start in range(0, big.shape[0], eval_micro_batch):
        chunk = big[start:start + eval_micro_batch, :-1]
        with torch.no_grad():
            logits = model(chunk)
        preds_chunks.append(logits.argmax(dim=-1))
        if logp_key is not None:
            # sec 9.1.5 S2: log p(target) at each row's OWN query position k, off the SAME logits
            # already computed for the argmax read -- zero extra forward passes.
            #
            # AUDIT FIX (2026-07-12, S2 follow-up audit): GATHER THE QUERY POSITIONS FIRST, then
            # log_softmax the (n_chunk, V) slice. The first S2 build ran `logits.log_softmax(dim=-1)`
            # on the FULL (n_chunk, T, V) tensor and then read ONE element per row -- discarding
            # 511/512 of it unread, and DOUBLING the peak VRAM of the largest allocation in the
            # whole pipeline. MEASURED ON THE H100 at the pinned eval_micro_batch=64, T=512,
            # V=50257, fp32: logits alone 6.14 GiB -> 12.27 GiB peak with the full log_softmax,
            # vs 6.16 GiB with this gather-first form (+0.02 GiB). Softmax is independent per
            # position, so the arithmetic is IDENTICAL (measured max |diff| 9.5e-7 -- fp32
            # reduction-order noise on a quantity of order 1-10 nats). This repo's own rule:
            # "The 50K vocab logits tensor is the VRAM bottleneck, not the model activations."
            n_chunk = chunk.shape[0]
            rows_t = torch.arange(n_chunk, device=logits.device)
            ks_t = torch.tensor([specs[owner[start + i]]["k"] for i in range(n_chunk)],
                                 device=logits.device, dtype=torch.long)
            tgt_t = torch.tensor([specs[owner[start + i]]["target"] for i in range(n_chunk)],
                                  device=logits.device, dtype=torch.long)
            lp = logits[rows_t, ks_t, :].float().log_softmax(dim=-1)[rows_t, tgt_t]   # (n_chunk,)
            lp_list = lp.tolist()
            for i in range(n_chunk):
                specs[owner[start + i]][logp_key] = float(lp_list[i])
            del lp
    preds = torch.cat(preds_chunks, dim=0)
    for local_i, spec_i in enumerate(owner):
        k = specs[spec_i]["k"]
        specs[spec_i][pred_key] = int(preds[local_i, k].item())


# =============================================================================
# 7. Top-level per-cell DiD evaluation.
# =============================================================================
def run_did_eval(model, val_tokens: torch.Tensor, batch_size: int, seq_len: int,
                  n_windows: int, device: str, mode_next: torch.Tensor, seed: int,
                  c_max: int = C_MAX_DEFAULT, eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT,
                  min_sep: int = 2, vocab_size: int = VOCAB_SIZE) -> dict:
    """Samples n_windows rows, runs ONE shared intact pass (permitted -- it
    is unmodified, sec 9.2), detects candidates+baseline (uncapped), caps per
    row (rung-independent), assigns placebo positions (Delta-matched), builds
    the TRUE and PLACEBO row-replicated batches, and returns raw per-candidate
    records (never an aggregate -- aggregation with the placebo-refusal rule
    lives in `finalize_cell`)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    mode_next_cpu = mode_next.tolist()

    # AUDIT FIX (2026-07-12, independent adversarial audit): WINDOW SAMPLING MUST BE
    # BATCH-SIZE-INDEPENDENT. The first rebuild drew the windows in `batch_size`-sized
    # get_batch() chunks. On CPU (MT19937) that happens to be batch-size-invariant, but
    # the real run is on CUDA, where get_batch's torch.randint uses the PHILOX generator
    # whose offset advances PER KERNEL LAUNCH -- so 32 draws of 16 does NOT reproduce
    # 16 draws of 32. MEASURED ON THE H100 AT THE SAME SEED: batch_size=16 and
    # batch_size=32 share only 50% of their 512 windows. The design's rungs use DIFFERENT
    # token-arithmetic batch sizes (three rungs at 32, the 1.31B at 16), and `--batch-size`
    # feeds straight in here -- so chunked sampling would hand the top rung a DIFFERENT
    # candidate population than the others. That is F-4 (regate sec 10.3) reborn in a new
    # guise, and it is exactly what sec 9.2's "the eval batch size is DECOUPLED from the
    # token-arithmetic batch size and from candidate selection entirely" forbids.
    # Drawing all n_windows offsets in ONE call makes the window population a pure
    # function of (seed, corpus, seq_len, n_windows). `batch_size` is retained as a
    # RECORDED PROVENANCE FIELD ONLY and no longer influences any measurement.
    x0, y0 = get_batch(val_tokens, n_windows, seq_len, gen)
    window = _make_window(x0, y0)
    all_windows = [window[b] for b in range(window.shape[0])]
    N_rows = len(all_windows)

    intact_batch = torch.stack(all_windows, dim=0)
    x_intact = intact_batch[:, :-1]
    y_intact = intact_batch[:, 1:]
    x_cpu = x_intact.cpu().tolist()
    y_cpu = y_intact.cpu().tolist()
    window_cpu = intact_batch.cpu().tolist()

    # Candidate detection happens BEFORE the intact forward pass (it is pure token-content --
    # y_cpu already carries the true next-token target, so no model output is needed) --
    # this is what lets the intact pass extract sec 9.1.5's S2 log-probs inline, per chunk,
    # without ever materializing the full (N_rows,T,V) log-prob tensor (>50GB at N_rows=512,
    # T=512, V=50257 -- infeasible; a per-chunk tensor of the SAME size as the logits already
    # computed is not).
    row_candidates, row_baseline = detect_candidates_and_baseline(x_cpu, y_cpu, mode_next_cpu, min_sep)
    sel_candidates = select_candidates_per_row(row_candidates, c_max, seed)
    specs = true_arm_specs(sel_candidates)
    specs = assign_placebo_positions(specs, sel_candidates, x_cpu, seed)

    ablate_rng = random.Random(_combine_seed(seed, "ablate_replacement"))   # ONE stream, TRUE/PLACEBO/D
    for s in specs:
        target = window_cpu[s["row_idx"]][s["k"] + 1]
        s["target"] = target
        s["repl_true"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["p"], target,
                                                      vocab_size, ablate_rng)
    for s in specs:
        if s["p_placebo"] is not None:
            target = s["target"]
            s["repl_placebo"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["p_placebo"],
                                                             target, vocab_size, ablate_rng)
        else:
            s["repl_placebo"] = None
    # sec 11.6.2 ARM D (NEW MANDATORY, FATAL-1 fix follow-up): corrupt position `j` -- the
    # antecedent bigram's KEY token at its FIRST occurrence -- leaving the antecedent VALUE
    # token at j+1 intact. "Same row-replication, same exclusion rule, same one-token-per-row
    # invariant" as arms B/C -- reuses draw_exclusive_replacement verbatim, continuing the
    # SAME ablate_rng stream (sec 9.2's "same replacement RNG stream", now extended to D). `j`
    # is a FIXED, KNOWN position from the candidate's own construction (never a resolved-
    # fallback like p_placebo), so it is unconditionally resolvable for every candidate.
    for s in specs:
        target = s["target"]
        s["repl_D"] = draw_exclusive_replacement(window_cpu[s["row_idx"]], s["j"], target,
                                                   vocab_size, ablate_rng)

    cand_by_row = {}
    for i, s in enumerate(specs):
        cand_by_row.setdefault(s["row_idx"], []).append(i)

    pred_chunks = []
    for start in range(0, N_rows, eval_micro_batch):
        chunk = x_intact[start:start + eval_micro_batch]
        with torch.no_grad():
            logits = model(chunk)
        pred_chunks.append(logits.argmax(dim=-1))
        # sec 9.1.5 S2 BUILD REQUIREMENT: log p(target) under the INTACT context, from the
        # SAME logits just computed for the argmax read -- zero extra forward passes.
        # Gathered at the candidate query positions ONLY, never as a full-tensor log_softmax
        # (see run_ablation_arm's note: the full form doubles peak VRAM -- measured +6.13 GiB
        # per chunk on the H100 at the pinned eval_micro_batch=64 -- to read at most c_max=8
        # of each row's 512 positions). AUDIT FIX (2026-07-12, S2 follow-up audit).
        # `row_idx = start + local_i` is the chunk-local -> global row map; cand_by_row was
        # built above from the FULL, final `specs` population (post-cap, post-placebo).
        pairs = [(local_i, spec_i)
                 for local_i in range(chunk.shape[0])
                 for spec_i in cand_by_row.get(start + local_i, [])]
        if pairs:
            li_t = torch.tensor([p[0] for p in pairs], device=logits.device, dtype=torch.long)
            ks_t = torch.tensor([specs[p[1]]["k"] for p in pairs], device=logits.device, dtype=torch.long)
            tgt_t = torch.tensor([specs[p[1]]["target"] for p in pairs], device=logits.device,
                                  dtype=torch.long)
            idx_t = torch.arange(len(pairs), device=logits.device)
            lp = logits[li_t, ks_t, :].float().log_softmax(dim=-1)[idx_t, tgt_t]   # (n_pairs,)
            lp_list = lp.tolist()
            for n_i, (_, spec_i) in enumerate(pairs):
                specs[spec_i]["logp_intact"] = float(lp_list[n_i])
            del lp
    pred_intact = torch.cat(pred_chunks, dim=0)  # (N_rows, T)

    run_ablation_arm(model, all_windows, specs, "p", "repl_true", "pred_true", eval_micro_batch, device,
                      logp_key="logp_true")
    run_ablation_arm(model, all_windows, specs, "p_placebo", "repl_placebo", "pred_placebo",
                      eval_micro_batch, device, logp_key="logp_placebo")
    # sec 11.6.2 / sec 11.11 build gate item 4: ARM D, key-ablated. +47% eval forwards
    # (8/(1+8+8), NOT the draft's mistranscribed "+33%") on a sub-GPU-hour eval.
    run_ablation_arm(model, all_windows, specs, "j", "repl_D", "pred_D", eval_micro_batch, device,
                      logp_key="logp_D")

    records = []
    for s in specs:
        row_idx, k = s["row_idx"], s["k"]
        target = s["target"]
        hit_intact = int(pred_intact[row_idx, k].item() == target)
        hit_true = int(s.get("pred_true") == target) if "pred_true" in s else None
        hit_placebo = int(s.get("pred_placebo") == target) if s.get("pred_placebo") is not None else None
        hit_D = int(s.get("pred_D") == target) if "pred_D" in s else None
        records.append({
            "row_idx": row_idx, "k": k, "j": s["j"], "delta": s["delta"],
            # AUDIT FIX (2026-07-12): carry delta_placebo so the pinned "distribution-
            # matched" property can be VERIFIED downstream instead of merely asserted.
            "delta_placebo": s.get("delta_placebo"),
            "hit_intact": hit_intact, "hit_true_ablated": hit_true,
            "hit_placebo_ablated": hit_placebo, "placebo_flagged": s["placebo_flagged"],
            # sec 11.6.2 ARM D -- the key-association component's own input (hit_D). Always
            # resolvable (unlike hit_placebo_ablated): `j` is never a resolved-fallback position.
            "hit_D": hit_D,
            # sec 9.1.5 S2 (log-prob sensitivity -- guards the argmax-floor bias toward
            # RISES/DECOUPLED). None iff the corresponding arm never ran (unresolved placebo).
            "logp_intact": s.get("logp_intact"),
            "logp_true_ablated": s.get("logp_true"),
            "logp_placebo_ablated": s.get("logp_placebo"),
            # sec 11.6.2 ARM D's own log-prob field.
            "logp_D": s.get("logp_D"),
        })

    baseline_hits, baseline_n = 0, 0
    for row_idx, ks in enumerate(row_baseline):
        for k in ks:
            baseline_hits += int(pred_intact[row_idx, k].item() == y_cpu[row_idx][k])
            baseline_n += 1
    acc_baseline_nonAR = baseline_hits / baseline_n if baseline_n else None

    return {
        "records": records, "n_windows": N_rows, "batch_size": batch_size, "seq_len": seq_len,
        "c_max": c_max, "acc_baseline_nonAR": acc_baseline_nonAR, "n_baseline": baseline_n,
    }


# =============================================================================
# 8. Clustered (row) bootstrap -- T1a/T1b's CI machinery.
# =============================================================================
def clustered_bootstrap_ci(records: list, stat_fn, n_boot: int = 2000, seed: int = 0,
                            alpha: float = 0.05) -> tuple:
    """Resamples ROWS (clusters) with replacement -- candidates within a row
    share a context and are not independent (sec 9.3 T1a). `stat_fn(records)
    -> float`. Returns (point, ci_lo, ci_hi)."""
    if not records:
        return None, None, None
    rows = sorted(set(r["row_idx"] for r in records))
    by_row = {ri: [] for ri in rows}
    for r in records:
        by_row[r["row_idx"]].append(r)
    point = stat_fn(records)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        subset = []
        for _ in range(len(rows)):
            subset.extend(by_row[rng.choice(rows)])
        if subset:
            boots.append(stat_fn(subset))
    if not boots:
        return point, None, None
    boots.sort()
    lo = boots[max(0, int(alpha / 2 * len(boots)))]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)) - 1)]
    return point, lo, hi


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stat_gap_true(records: list) -> float:
    return _mean([r["hit_intact"] for r in records]) - _mean([r["hit_true_ablated"] for r in records])


def stat_gap_placebo(records: list) -> float:
    return _mean([r["hit_intact"] for r in records]) - _mean([r["hit_placebo_ablated"] for r in records])


def stat_did(records: list) -> float:
    # DiD(r) = gap_true - gap_placebo = acc_placebo_ablated - acc_true_ablated
    # (acc_intact cancels: both arms are compared to the SAME candidate
    # population's SAME intact reads -- a paired design, sec 9.2's formula).
    return _mean([r["hit_placebo_ablated"] for r in records]) - _mean([r["hit_true_ablated"] for r in records])


def stat_did_logp(records: list) -> float:
    """sec 9.1.5 S2: DiD_logp = E[logp_placebo_ablated - logp_true_ablated] -- the identical
    estimand as `stat_did` (intact cancels the same way) under a continuous, floor-free
    readout instead of a hard argmax threshold. Same clustered-bootstrap machinery applies
    unchanged (it is generic over any per-record scalar)."""
    return (_mean([r["logp_placebo_ablated"] for r in records])
            - _mean([r["logp_true_ablated"] for r in records]))


def has_complete_s2_fields(records: list) -> bool:
    """sec 9.1.5 BUILD REQUIREMENT / sec 9.6 stop rule: 'a cell missing these fields is
    VOID.' True iff every record carries a non-None logp in all three arms."""
    return all(r.get("logp_intact") is not None and r.get("logp_true_ablated") is not None
               and r.get("logp_placebo_ablated") is not None for r in records)


def stat_s3(records: list) -> float:
    """sec 11.6.2 ARM D / S3: S3 = E[C-D] = E[hit_placebo_ablated - hit_D] -- THE
    KEY-ASSOCIATION COMPONENT, placebo-controlled (arm C is the SAME comparator DiD uses).
    'The extra correct-emission rate attributable to destroying the KEY rather than an
    arbitrary matched-distance token, with b still in context.' Reported with clustered-
    bootstrap CIs at every rung, ALWAYS -- MANDATORY, REPORTED ALWAYS, VERDICT-CARRYING
    NEVER, exactly like S1/S2 (sec 11.6.2's own pin). Never fed into compute_capacity_metric
    / compute_verdict."""
    return _mean([r["hit_placebo_ablated"] for r in records]) - _mean([r["hit_D"] for r in records])


def stat_residual_d_minus_b(records: list) -> float:
    """sec 11.6.2's UNLABELLED RESIDUAL: E[D-B] = E[hit_D - hit_true_ablated]. Conceded to
    A-S9 in the design doc: `DiD = E[C-D] + E[D-B]` telescopes trivially, but D and B differ
    in TWO treatments at once (key-destroyed/value-intact vs key-intact/value-destroyed), so
    this is NOT 'the effect of removing b given the association is destroyed' -- isolating
    that would need a forbidden two-token arm E. Published as a RESIDUAL, never labelled a
    mechanism, and NEVER called 'the salience component' (sec 11.6.2's own explicit
    instruction). Non-verdict-carrying, like S3 itself."""
    return _mean([r["hit_D"] for r in records]) - _mean([r["hit_true_ablated"] for r in records])


def has_complete_arm_d_fields(records: list) -> bool:
    """sec 11.6.2 build requirement's own analogue to has_complete_s2_fields. `j` is a FIXED,
    KNOWN candidate-construction position (never a resolved-fallback like p_placebo), so this
    is expected to be True whenever S2's fields are (both arms ran on the same specs) -- kept
    as a SEPARATE, explicit check rather than silently assumed, matching this file's own
    'a structural check that is merely assumed is not a check' discipline."""
    return all(r.get("hit_D") is not None and r.get("logp_D") is not None for r in records)


# =============================================================================
# 9. finalize_cell -- THE VERDICT-GRADE REFUSAL POINT.
#    "No un-placebo'd gap, and no raw AR-hit slice, may carry a verdict."
# =============================================================================
class VerdictGradeError(RuntimeError):
    pass


def finalize_cell(rung: str, corpus: str, did_eval_result: dict,
                   flagged_frac_threshold: float = PLACEBO_FLAGGED_FRAC_VOID_THRESHOLD,
                   n_boot: int = 2000, boot_seed: int = 0, t2b_result: dict = None) -> dict:
    """Aggregates run_did_eval's raw per-candidate records into a
    verdict-eligible cell summary. THIS is the enforcement point: any record
    lacking a resolved placebo read is EXCLUDED from every statistic below,
    and if NO record has a resolved placebo, this raises VerdictGradeError
    rather than silently falling back to the un-placebo'd gap. The returned
    dict is stamped `_verdict_grade=True` -- `compute_capacity_metric` below
    refuses any input dict lacking that stamp, closing the loophole of a
    caller hand-building a look-alike dict.

    `t2b_result`: optional `run_t2_repaired_probe(...)` output (REV 3, sec
    11 -- supersedes the retired `run_t2_planted_copy`, same `acc_copy`/
    `acc_copy_se` schema) for this SAME (rung, corpus) -- if given, its
    `acc_copy`/`acc_copy_se` are merged into the returned cell so `acc_copy`
    is in the schema alongside `did` (sec 9.1.5 S1's own inputs, and the
    coordinator's schema requirement)."""
    all_records = did_eval_result["records"]
    resolved = [r for r in all_records if r["hit_placebo_ablated"] is not None]
    if not resolved:
        raise VerdictGradeError(
            f"{rung}/{corpus}: zero candidates have a resolved placebo read out of "
            f"{len(all_records)} total -- cannot compute a verdict-grade DiD. Refusing to "
            f"fall back to the un-placebo'd gap (PARAM_AXIS_SCALING_DESIGN.md sec 9.2: 'no "
            f"un-placebo'd gap, and no raw AR-hit slice, may carry a verdict')."
        )
    flagged_frac = placebo_flagged_fraction_from_records(all_records)
    cell_void_placebo_match = flagged_frac > flagged_frac_threshold

    did_pt, did_lo, did_hi = clustered_bootstrap_ci(resolved, stat_did, n_boot=n_boot, seed=boot_seed)
    gt_pt, gt_lo, gt_hi = clustered_bootstrap_ci(resolved, stat_gap_true, n_boot=n_boot, seed=boot_seed + 1)
    gp_pt, gp_lo, gp_hi = clustered_bootstrap_ci(resolved, stat_gap_placebo, n_boot=n_boot, seed=boot_seed + 2)

    t1a_pass = (did_lo is not None) and (did_lo > 0)  # DiD CI excludes 0, and positive

    # sec 9.1.5 S2 (BUILD REQUIREMENT, non-optional): a cell missing the log-prob fields on
    # any resolved candidate is VOID (sec 9.6's stop rule -- S2 "cannot be added after a
    # read"). When complete, DiD_logp gets the SAME clustered-row bootstrap as the primary.
    s2_complete = has_complete_s2_fields(resolved)
    if s2_complete:
        didlogp_pt, didlogp_lo, didlogp_hi = clustered_bootstrap_ci(
            resolved, stat_did_logp, n_boot=n_boot, seed=boot_seed + 3)
        per_row_did_logp = _per_row_stat(resolved, stat_did_logp)
    else:
        didlogp_pt = didlogp_lo = didlogp_hi = None
        per_row_did_logp = {}

    # sec 11.6.2 ARM D / S3 -- MANDATORY, REPORTED ALWAYS, VERDICT-CARRYING NEVER (pinned
    # exactly like S1/S2). Computed over the SAME `resolved` population as DiD (hit_D is
    # unconditionally resolvable, so no separate filter is needed). NOT gated into
    # cell_void_placebo_match / cell_void_missing_s2_fields / compute_capacity_metric --
    # S3 is a decomposition of the estimand sec 9.1 already pinned, never a new gate.
    arm_d_complete = has_complete_arm_d_fields(resolved)
    if arm_d_complete:
        s3_pt, s3_lo, s3_hi = clustered_bootstrap_ci(resolved, stat_s3, n_boot=n_boot, seed=boot_seed + 4)
        resid_pt, resid_lo, resid_hi = clustered_bootstrap_ci(
            resolved, stat_residual_d_minus_b, n_boot=n_boot, seed=boot_seed + 5)
    else:
        s3_pt = s3_lo = s3_hi = None
        resid_pt = resid_lo = resid_hi = None

    cell = {
        "_verdict_grade": True,
        "rung": rung, "corpus": corpus,
        "n_candidates_total": len(all_records), "n_candidates_resolved": len(resolved),
        "placebo_flagged_frac": flagged_frac,
        "cell_void_placebo_match": cell_void_placebo_match,
        "did": did_pt, "did_ci": [did_lo, did_hi],
        "gap_true": gt_pt, "gap_true_ci": [gt_lo, gt_hi],
        "gap_placebo": gp_pt, "gap_placebo_ci": [gp_lo, gp_hi],
        "t1a_pass_did_ci_excludes_zero": t1a_pass,
        "acc_baseline_nonAR": did_eval_result.get("acc_baseline_nonAR"),
        "n_baseline": did_eval_result.get("n_baseline"),
        "rows_used": did_eval_result.get("n_windows"),
        "per_row_did": _per_row_stat(resolved, stat_did),
        "delta_match": summarize_delta_match(resolved),
        # --- sec 9.1.5 S2 ---
        "cell_void_missing_s2_fields": not s2_complete,
        "did_logp": didlogp_pt, "did_logp_ci": [didlogp_lo, didlogp_hi],
        "per_row_did_logp": per_row_did_logp,
        # --- sec 11.6.2 Arm D / S3 (mandatory, reported always, verdict-carrying NEVER) ---
        "cell_arm_d_incomplete": not arm_d_complete,
        "s3": s3_pt, "s3_ci": [s3_lo, s3_hi],
        "residual_d_minus_b_UNLABELLED": resid_pt, "residual_d_minus_b_ci": [resid_lo, resid_hi],
    }
    if t2b_result is not None:
        cell["acc_copy"] = t2b_result.get("acc_copy")
        cell["acc_copy_se"] = t2b_result.get("acc_copy_se")
    else:
        cell["acc_copy"] = None
        cell["acc_copy_se"] = None
    return cell


def summarize_delta_match(resolved: list) -> dict:
    """AUDIT FIX (2026-07-12). sec 9.2 pins the placebo as DISTRIBUTION-matched in Delta,
    but nothing in the instrument ever MEASURED the match it claims -- it was asserted by
    construction and never reported. It is not exact, and it cannot be: Delta' is drawn
    from the pooled empirical Delta and then REJECTED if it lands before position 0, so
    for a candidate with a small k every large Delta' is rejected and the realized placebo
    Delta is pulled toward SHORTER distances. (Measured on a synthetic 1,600-candidate
    population with a realistic k-spread: placebo mean Delta ran ~35% BELOW true mean
    Delta, with zero fallback flags -- i.e. the 5%-flag gate does not see this at all.)
    That residual is inherent to the pinned procedure, not a deviation from it, but it is
    load-bearing: gap_placebo is the model's generic damage-sensitivity AT A DISTANCE
    PROFILE, and if the placebo's profile is systematically shorter than the true arm's,
    DiD is biased by whatever the model's damage-sensitivity-vs-distance slope is. This
    reports the realized profiles so the residual can be sized, disclosed, and (if it
    matters) corrected in analysis. It is a REPORT, not a gate -- adding a gate here would
    be inventing a pre-registration the design does not have."""
    d_true = [r["delta"] for r in resolved if r.get("delta") is not None]
    d_plac = [r["delta_placebo"] for r in resolved if r.get("delta_placebo") is not None]
    if not d_true or not d_plac:
        return {"n_true": len(d_true), "n_placebo": len(d_plac)}
    st, sp = sorted(d_true), sorted(d_plac)
    return {
        "n_true": len(st), "n_placebo": len(sp),
        "mean_delta_true": _mean(st), "mean_delta_placebo": _mean(sp),
        "median_delta_true": st[len(st) // 2], "median_delta_placebo": sp[len(sp) // 2],
        "mean_shift_placebo_minus_true": _mean(sp) - _mean(st),
    }


def placebo_flagged_fraction_from_records(records: list) -> float:
    if not records:
        return 1.0
    return sum(1 for r in records if r.get("placebo_flagged")) / len(records)


def _per_row_stat(records: list, stat_fn) -> dict:
    """Per-row (cluster) values of `stat_fn`, needed by the trend-fit layer's
    own row-level bootstrap (sec 9.5)."""
    by_row = {}
    for r in records:
        by_row.setdefault(r["row_idx"], []).append(r)
    return {ri: stat_fn(recs) for ri, recs in by_row.items() if recs}


# =============================================================================
# 10. sec 9.1 -- THE NORMALIZATION SLOT. Pluggable; kept general even though
#     sec 9.1 is now PINNED (commit 7381d2a, blind agent, forced by N1-N3's
#     algebraic argument -- PARAM_AXIS_SCALING_DESIGN.md sec 9.1.1). The
#     mechanism (register/lookup/fail-loud-if-unset-or-unknown) is UNCHANGED
#     by the pin landing -- only ONE entry is now registered by name below,
#     exactly as the pin's own text prescribes registering it, so a future
#     metric (or a PI override) can still add/replace entries without a code
#     change here.
# =============================================================================
NORMALIZATION_REGISTRY: dict = {}


def register_normalization(name: str, fn, override: bool = False) -> None:
    """Registers a normalization form `fn(cell: dict) -> float`, i.e. the
    WHOLE verdict-grade cell (not just `did`/`acc_baseline_nonAR` positionally)
    -- this is the calling convention sec 9.1's pin itself specifies
    verbatim: `register_normalization("raw_did", lambda cell: cell["did"])`.
    Passing the whole cell is what lets a future normalization reach fields
    other than `did` (e.g. `acc_copy`) without a signature change here.

    AUDIT FIX (2026-07-12, S2 follow-up audit): refuses to SILENTLY REPLACE an
    already-registered name. sec 9.1's pin is registered at import time as `"raw_did"`;
    before this fix any later `register_normalization("raw_did", <anything>)` -- a stray
    line in the driver, a copy-paste of the pin's own literal text with a different body --
    would have overwritten the PINNED form with no warning at all, and
    `compute_capacity_metric(cell, "raw_did", ...)` would have kept returning a number under
    the pinned name while computing something else. That is a silent pre-registration
    violation, which is the one failure mode this registry exists to prevent. Re-registering
    the IDENTICAL function object is a no-op (so `importlib.reload` / double-import stay
    safe); replacing a name with a different function requires `override=True`, out loud."""
    if name in NORMALIZATION_REGISTRY and NORMALIZATION_REGISTRY[name] is not fn and not override:
        raise RuntimeError(
            f"REFUSING TO SILENTLY OVERWRITE the registered normalization {name!r} with a "
            f"different function. sec 9.1's pin (PARAM_AXIS_SCALING_DESIGN.md sec 9.1, commit "
            f"7381d2a) registers 'raw_did' -> M(r) = DiD(r) at import time, and a silent "
            f"rebinding of a pinned name is a pre-registration violation, not a code detail. "
            f"If you genuinely mean to replace it, pass override=True and record why."
        )
    NORMALIZATION_REGISTRY[name] = fn


def _raw_did(cell: dict) -> float:
    """sec 9.1 THE PIN (PARAM_AXIS_SCALING_DESIGN.md sec 9.1, commit 7381d2a):
    M(r) = DiD(r), denominator 1. FORCED by N1-N3 (sec 9.1.1's algebraic
    identity: the intact arm cancels exactly, DiD = E[C-B], so it is already
    competence-invariant and a denominator that varies with general
    competence would re-inject exactly the confound the numerator purged --
    "a double correction, applied in the wrong functional form"). Registered
    under this exact call, per the pin's own text: `register_normalization
    ("raw_did", lambda cell: cell["did"])`. NOT hardcoded into
    `compute_capacity_metric` -- it goes through the SAME registry lookup as
    any future normalization would, and normalization=None or an unknown
    name still fails loudly (see compute_capacity_metric)."""
    return cell["did"]


register_normalization("raw_did", _raw_did)


def compute_capacity_metric(cell: dict, normalization) -> float:
    """The ONLY function that turns a verdict-grade cell (sec 9's DiD, N1)
    into the reported capacity metric M(r). Requires `cell['_verdict_grade']
    is True` (produced only by `finalize_cell`), `normalization` to be a
    NON-None, REGISTERED name, and (sec 9.1.5 BUILD REQUIREMENT) the cell to
    carry complete S2 log-prob fields. Never defaults.

    REV 3 (sec 11.6, 2026-07-12): the `t2b2_pass` keyword and its VOID-raising
    check are REMOVED. T2b-2 is RETIRED, not replaced by another ceiling
    (sec 11.6: "No ceiling check replaces it" -- the domination it asserted
    is FALSE in general, for two independent structural reasons, not merely
    unproven). T2b-2 was the ONE trigger that mapped a rung to VOID at this
    layer; with it gone, this function's remaining VOID triggers (placebo
    match, missing S2 fields) are the complete set for a CELL-level defect.
    The rung-level "does this checkpoint have a key-conditioned copy
    mechanism" question now lives ENTIRELY in check_rung_admissible via
    T2b-1 + the NEW T2b-1b (check_t2b1b_key_conditioned) -- both map to
    FLOOR (excluded, reported), never VOID, per sec 9.6 item 5's REPAIRED
    text: "T1a and T2b-1 and T2b-1b all pass; failure of any => FLOOR
    rung." Do not reintroduce a t2b2-shaped argument here; that would be
    exactly the "invent a replacement ceiling" sec 11.6 forbids.

    AUDIT FIX (2026-07-12, pre-REV-3): a placebo-unmatched cell is refused
    outright. Before that fix the guard chain checked only (stamp +
    registered normalization); the `_verdict_grade` stamp only ever
    certified "a placebo arm was run," never "this rung may carry a
    verdict." Note this deliberately does NOT gate on T1a: a T1a-failing
    rung is FLOOR, and sec 9.5 requires FLOOR rungs to be REPORTED (just
    excluded from the fit), not refused."""
    if not cell.get("_verdict_grade"):
        raise VerdictGradeError(
            "compute_capacity_metric received a cell dict that was not produced by "
            "finalize_cell() (missing _verdict_grade=True stamp) -- refusing to compute a "
            "verdict-grade metric from an unvalidated/un-placebo'd input."
        )
    if cell.get("cell_void_placebo_match", True):
        raise VerdictGradeError(
            f"CELL VOID (placebo not distribution-matched): "
            f"{cell.get('rung')}/{cell.get('corpus')} has placebo_flagged_frac="
            f"{cell.get('placebo_flagged_frac')} > {PLACEBO_FLAGGED_FRAC_VOID_THRESHOLD} "
            f"(PARAM_AXIS_SCALING_DESIGN.md sec 9.2). A cell whose placebo arm is not "
            f"distribution-matched cannot carry a DiD, and therefore cannot carry M(r)."
        )
    if cell.get("cell_void_missing_s2_fields", True):
        raise VerdictGradeError(
            f"CELL VOID (missing sec 9.1.5 S2 log-prob fields): "
            f"{cell.get('rung')}/{cell.get('corpus')} has at least one resolved candidate "
            f"without logp_intact/logp_true_ablated/logp_placebo_ablated. sec 9.1.5: the S2 "
            f"log-prob sensitivity 'cannot be added after a read (that would be a re-read, "
            f"banned by sec 9.6)' -- refusing to emit M(r) for a cell that cannot support it."
        )
    if normalization is None:
        raise RuntimeError(
            "NORMALIZATION_UNSET: PARAM_AXIS_SCALING_DESIGN.md sec 9.1 has not been pinned yet. "
            "This is not a bug -- sec 9.1 is deliberately left open pending a blind pin (sec 9.7's "
            "handoff protocol) or a direct PI decision. Do not pass a guessed default; wait for "
            "the pin, then `register_normalization(name, fn)` and pass that `name` here."
        )
    if normalization not in NORMALIZATION_REGISTRY:
        raise KeyError(
            f"Unknown normalization {normalization!r}. Registered: "
            f"{sorted(NORMALIZATION_REGISTRY)!r}. The sec 9.1 pin must call "
            f"register_normalization({normalization!r}, <fn>) before this can run for real."
        )
    return NORMALIZATION_REGISTRY[normalization](cell)


####################################################################################
# sec 11 -- THE T2 REPAIR (2026-07-12). pick_t2_marker_tokens / run_t2_planted_copy /
# check_t2b2_ceiling are ALL RETIRED (sec 11.1, sec 11.6). Replaced by sections
# 11.A (pools + per-window assignment), 11.B (the six-arm probe), 11.C (T2b-1/
# T2b-1b), 11.D (T2a/T1c gates). See the module docstring's REV 3 BUILD block.
####################################################################################

# =============================================================================
# 11.A -- sec 11.2 THE REPAIRED KEY/VALUE SELECTION RULE, PINNED. Builds
#     P_key / P_val(a) / the inverse map b -> {licensing keys}, TRAIN-split
#     corpus statistics ONLY -- "No model forward pass enters key selection,
#     value selection, decoy selection, Delta drawing, plant placement, or
#     window dropping -- at any point, for any reason" (sec 11.2). Then the
#     PER-WINDOW procedure: Delta by rejection sampling, a JOINTLY-drawn
#     fresh (a,a',b) triple, and the HARD RuntimeError assertion (sec 11.2.3)
#     that is this repair's actual teeth.
# =============================================================================

# sec 11.2.1's pinned constants (THE DOC WINS on any transcription mismatch).
K2_RARE_IN_WINDOW_LADDER = (1e-4, 2e-4, 4e-4)   # "1e-4 -> 2e-4 -> 4e-4"
K3_MIN_COUNT = 500
K3_MIN_P_TRAIN = 1e-5
K4_MAX_RIVAL_MASS = 0.5
POOL_FLOOR_P_KEY = 100
POOL_FLOOR_P_VAL = 5             # sec 11.2.1 K5
# sec 11.2.2's pinned constants.
V2_MIN_COUNT_B = 500
V3_MIN_COUNT_AB = 5              # THE FIX FOR F-II (was "count(a,b) == 0" -- selected for impossibility)
V4_MAX_P_B_GIVEN_A = 0.05
V4_RANK_LO, V4_RANK_HI = 2, 50
V5_MAX_P_TRAIN_B = 1e-4          # THE FIX FOR A-S2 (value-side rare-in-window)
POOL_FLOOR_LICENSED_B = 100      # sec 11.2.3 point 3: ">=100 distinct b each licensed by >=2 keys"
T2_TRIPLE_MAX_TRIES = 100        # sec 11.2.3 point 3: "Up to 100 tries."
T2_DELTA_MAX_TRIES = 1000        # rejection-sampling budget for Delta (not separately pinned;
                                  # generous relative to the real Delta pools' <=5% excluded mass)
T2_DROP_CAP_FRAC = 0.02          # sec 11.2.3 point 6: "Cap: drops <= 2% of n_plants"


def _bigram_stats(train_tokens: torch.Tensor, vocab_size: int, device: str) -> dict:
    """Shared low-level bigram statistics for the pool builder: unigram
    counts and the (a,b)->count pair table. `torch.unique` on
    `pair_ids = a*V+b` returns `uniq` SORTED ascending, which makes
    `a_of_pair = uniq // V` monotone non-decreasing -- i.e. bigrams are
    ALREADY grouped by `a` in the returned order. Every per-a bucket lookup
    below therefore uses `torch.searchsorted` (O(log n)) rather than a
    per-candidate boolean mask over the full pair table, which is what the
    RETIRED picker's own entropy loop did and does not scale past ~400
    candidates -- this repair's pools run 500-8000+ candidate keys."""
    t = train_tokens.to(device)
    N = t.numel()
    counts = torch.bincount(t, minlength=vocab_size).to(torch.float64)
    a_next = t[:-1].to(torch.int64)
    b_next = t[1:].to(torch.int64)
    pair_ids = a_next * vocab_size + b_next
    uniq, pair_counts = torch.unique(pair_ids, return_counts=True)
    a_of_pair = uniq // vocab_size
    b_of_pair = uniq % vocab_size
    pair_counts = pair_counts.to(torch.float64)
    return {"N": N, "counts": counts, "a_of_pair": a_of_pair, "b_of_pair": b_of_pair,
            "pair_counts": pair_counts}


def _bucket_slice(a_of_pair_sorted: torch.Tensor, a_val: int) -> tuple:
    """searchsorted-based (lo,hi) index range of `a_val`'s bigram bucket in
    the sorted pair table -- see `_bigram_stats`'s docstring."""
    a_t = torch.tensor(a_val, device=a_of_pair_sorted.device)
    lo = int(torch.searchsorted(a_of_pair_sorted, a_t).item())
    hi = int(torch.searchsorted(a_of_pair_sorted, a_t, right=True).item())
    return lo, hi


def build_key_value_pools(train_tokens: torch.Tensor, vocab_size: int, device: str,
                           eot_token_id: int = EOT_TOKEN_ID,
                           k2_ladder: tuple = K2_RARE_IN_WINDOW_LADDER) -> dict:
    """sec 11.2.1/11.2.2, PINNED. Builds P_key (K1-K5: not-special, rare-in-
    window, well-trained, beatable, |P_val(a)|>=5) and P_val(a) (V1-V5:
    not-{a}/not-special, well-trained, attested>=5x, licensed-but-not-
    predicted [rank 2-50, p<=0.05], rare-in-window) from TRAIN-split
    statistics ONLY. Also returns `inverse_map` (b -> sorted list of
    licensing keys, RESTRICTED to b's with >=2 licensing keys -- exactly
    what the sec 11.2.3 joint-triple draw needs) and `val_meta[a][b]` (the
    sec 11.4.3 per-plant difficulty record's own inputs: p(b|a), rank(b|a),
    count(a,b), count(b), max_rival_p(a)).

    K2 LADDER (sec 11.2.1): tries 1e-4, then 2e-4, then 4e-4, STOPPING at the
    first rung whose |P_key| clears POOL_FLOOR_P_KEY=100 -- "a fixed
    pre-registered ladder ... stopping at the first rung that clears, and
    report which rung was used." K3/K4 do not depend on the K2 band, so the
    (expensive) V1-V5 pass runs ONCE, at the WIDEST band in the ladder; the
    K2 ladder then only FILTERS that one result by p_train(a), which is
    mathematically equivalent to rebuilding per-band (K2 only ever REMOVES
    candidates as the band tightens) and is what makes this tractable.

    Returns a dict: P_key (list[int], the ladder-selected keys), P_val
    (dict[int,list[int]]), inverse_map (dict[int,list[int]], >=2-key b's
    only), val_meta (dict[int,dict[int,dict]]), k2_band_used (float),
    floor_pool_key (bool), floor_pool_val (bool, True iff EVERY retained key
    has |P_val(a)|>=5 -- true by construction post-K5), floor_licensed_b
    (bool), n_p_key, median_p_val, n_licensed_b_ge2 (diagnostics)."""
    stats = _bigram_stats(train_tokens, vocab_size, device)
    N, counts = stats["N"], stats["counts"]
    a_of_pair, b_of_pair, pair_counts = stats["a_of_pair"], stats["b_of_pair"], stats["pair_counts"]

    # K4: max_b p_train(b|a) <= 0.5 -- reuses build_bigram_mode_table's own scatter_reduce
    # pattern (this file, sec 3) to get max_b count(a,b) for every a in ONE vectorized pass.
    max_count_per_a = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    max_count_per_a.scatter_reduce_(0, a_of_pair, pair_counts, reduce="amax", include_self=True)
    max_p_given_a = max_count_per_a / counts.clamp(min=1)

    p_train_a = counts / N
    widest_band = max(k2_ladder)
    k1234 = (
        (torch.arange(vocab_size, device=device) != eot_token_id)
        & (p_train_a <= widest_band)                                    # K2 (widest band)
        & (counts >= K3_MIN_COUNT) & (p_train_a >= K3_MIN_P_TRAIN)       # K3
        & (max_p_given_a <= K4_MAX_RIVAL_MASS)                           # K4
    )
    candidate_a_list = k1234.nonzero(as_tuple=True)[0].tolist()

    counts_list = counts.tolist()   # O(1) Python-level lookups, used heavily below and by callers
    p_val: dict = {}
    val_meta: dict = {}
    for a in candidate_a_list:
        lo, hi = _bucket_slice(a_of_pair, a)
        if hi <= lo:
            continue
        b_ids = b_of_pair[lo:hi]
        b_counts = pair_counts[lo:hi]
        count_a = counts_list[a]
        # AUDIT FIX (2026-07-12, MINOR): stable=True. V3's count(a,b)>=5 floor means ties in
        # b_counts are COMMON (not an edge case), and the non-stable default's tie-break order
        # is an implementation detail of the sort algorithm, not a property of (corpus, window
        # index) -- risking sec 11.4.6's "byte-identical planted windows" claim on a tie. Also:
        # a value TIED with the modal (rank-1) continuation could otherwise land at rank 2
        # depending on tie order, in tension with V4's "never the modal continuation" intent.
        order = torch.argsort(b_counts, descending=True, stable=True)
        b_ids_s = b_ids[order]
        b_counts_s = b_counts[order]
        ranks = torch.arange(1, b_ids_s.numel() + 1, device=device)
        p_given_a_s = b_counts_s / count_a
        count_b_s = counts[b_ids_s]
        p_train_b_s = count_b_s / N

        v1 = (b_ids_s != a) & (b_ids_s != eot_token_id)
        v2 = count_b_s >= V2_MIN_COUNT_B
        v3 = b_counts_s >= V3_MIN_COUNT_AB
        v4 = (p_given_a_s <= V4_MAX_P_B_GIVEN_A) & (ranks >= V4_RANK_LO) & (ranks <= V4_RANK_HI)
        v5 = p_train_b_s <= V5_MAX_P_TRAIN_B
        mask = v1 & v2 & v3 & v4 & v5
        if not bool(mask.any()):
            continue
        kept_b = b_ids_s[mask].tolist()
        if len(kept_b) < POOL_FLOOR_P_VAL:   # K5
            continue
        p_val[a] = kept_b
        val_meta[a] = {
            b: {"p_b_given_a": float(p_given_a_s[i]), "rank_b_given_a": int(ranks[i]),
                "count_ab": float(b_counts_s[i]), "count_b": float(count_b_s[i]),
                "max_rival_p": float(max_p_given_a[a])}
            for i, b in zip(mask.nonzero(as_tuple=True)[0].tolist(), kept_b)
        }

    # THE K2 LADDER: filter the wide-band result down to the tightest band that clears the floor.
    k2_band_used = None
    p_key_final = []
    for band in sorted(k2_ladder):
        cand = [a for a in p_val if (counts_list[a] / N) <= band]
        if len(cand) >= POOL_FLOOR_P_KEY:
            k2_band_used = band
            p_key_final = sorted(cand)
            break
    if k2_band_used is None:
        # every rung failed the floor -- report the WIDEST band's result anyway (diagnostic),
        # floor_pool_key below is what a caller must check before proceeding (sec 11.2's
        # "The floors are GATES, not hopes").
        k2_band_used = widest_band
        p_key_final = sorted(a for a in p_val if (counts_list[a] / N) <= widest_band)

    p_val_final = {a: p_val[a] for a in p_key_final}
    val_meta_final = {a: val_meta[a] for a in p_key_final}

    inverse_map: dict = {}
    for a in p_key_final:
        for b in p_val_final[a]:
            inverse_map.setdefault(b, []).append(a)
    inverse_map_ge2 = {b: sorted(keys) for b, keys in inverse_map.items() if len(keys) >= 2}

    val_sizes = sorted(len(v) for v in p_val_final.values())
    median_p_val = val_sizes[len(val_sizes) // 2] if val_sizes else 0

    return {
        "P_key": p_key_final, "P_val": p_val_final, "inverse_map": inverse_map_ge2,
        "val_meta": val_meta_final, "k2_band_used": k2_band_used,
        "floor_pool_key": len(p_key_final) >= POOL_FLOOR_P_KEY,
        # AUDIT FIX (2026-07-12): `all(...)` over an EMPTY p_val_final is vacuously True in
        # Python -- an empty pool would have silently read floor_pool_val=True. `bool(...)` on
        # the dict itself makes non-emptiness a precondition, closing that hole.
        "floor_pool_val": bool(p_val_final) and all(len(v) >= POOL_FLOOR_P_VAL for v in p_val_final.values()),
        "floor_licensed_b": len(inverse_map_ge2) >= POOL_FLOOR_LICENSED_B,
        "n_p_key": len(p_key_final), "median_p_val": median_p_val,
        "n_licensed_b_ge2": len(inverse_map_ge2),
    }


def rejection_sample_delta(delta_pool: list, seq_len: int, rng: random.Random,
                            max_tries: int = T2_DELTA_MAX_TRIES) -> dict:
    """sec 11.2.3 point 1: draws Delta by REJECTION SAMPLING from the main
    metric's own empirical Delta pool, restricted to `2 <= Delta <= seq_len - 6`.
    The old `max(2, min(Delta, T-4))` CLAMP is RETIRED (sec 11.2.3: clamping
    piles the truncated tail onto the boundary and distorts the Delta
    profile the T2a gate is defined on). Returns
    {'delta': int|None, 'n_tries': int, 'n_rejected': int}; the caller must
    report `n_rejected/n_tries` as the excluded-Delta-mass diagnostic (sec
    11.2.3: 'if > 5%, disclose it in the T2a report').

    AUDIT FIX (2026-07-12, MINOR): the lower bound `Delta >= 2` is NEW. sec 9.0's own
    candidate construction pins `min_sep=2` (Delta = k-(j+1) is never < 2 in a REAL
    candidate pool passed in as `delta_pool`), so this is unreachable in practice -- but
    without it, `Delta=0` gives `j0==k0`, and `plant_and_verify_t2_window`'s
    `sorted({j0,k0})` SILENTLY COLLAPSES a duplicate position to one element, so a
    degenerate zero-Delta 'plant' (the key present ONCE, at the query position, with NO
    demonstration in context) would pass the hard assertion. Excluded here, upstream of
    that check, rather than patching the assertion itself."""
    limit = seq_len - 6
    tries = 0
    rejected = 0
    while tries < max_tries:
        tries += 1
        d = rng.choice(delta_pool)
        if 2 <= d <= limit:
            return {"delta": d, "n_tries": tries, "n_rejected": rejected}
        rejected += 1
    return {"delta": None, "n_tries": tries, "n_rejected": rejected}


def draw_t2_triple(window_tokens: list, inverse_map: dict, counts_by_token: list,
                    rng: random.Random, max_tries: int = T2_TRIPLE_MAX_TRIES,
                    freq_match_dex: float = 0.1) -> dict:
    """sec 11.2.3 point 3, PINNED: draws (a, a', b) JOINTLY by walking a
    seeded permutation of the inverse map `b -> {licensing keys}`, subject
    to: a' not in {a,b}; frequency-matched to `freq_match_dex` decades
    (|log10 count(a) - log10 count(a')| <= 0.1, "the same band spans 8.7x
    and is NOT a match"); and natural occurrence count in `window_tokens` of
    a, a', AND b all exactly 0 (checked here via set membership -- O(1) --
    since only ABSENCE matters at this stage; the exact COUNT is re-verified
    post-plant by `plant_and_verify_t2_window`'s hard assertion). `a in
    P_key`/`b in P_val(a) INTERSECT P_val(a')` are guaranteed by construction
    (inverse_map[b] IS {a : b in P_val(a)}, restricted to a's already in
    P_key). Up to `max_tries` TOTAL attempts across the whole walk (sec
    11.2.3: 'Up to 100 tries'). Returns {'found': bool, 'a':, 'a_prime':,
    'b':, 'n_tries': int}."""
    window_set = set(window_tokens)
    b_perm = list(inverse_map)
    rng.shuffle(b_perm)
    tries = 0
    for b in b_perm:
        if b in window_set:
            continue   # sec 11.2.3: natural count of b must be 0 -- cheap pre-filter
        keys = list(inverse_map[b])
        rng.shuffle(keys)
        n = len(keys)
        for i in range(n):
            a = keys[i]
            if a in window_set:
                continue
            log_ca = math.log10(counts_by_token[a]) if counts_by_token[a] > 0 else None
            for j in range(n):
                if i == j:
                    continue
                tries += 1
                if tries > max_tries:
                    return {"found": False, "n_tries": tries}
                a_prime = keys[j]
                if a_prime in (a, b) or a_prime in window_set:
                    continue
                if log_ca is None or counts_by_token[a_prime] <= 0:
                    continue
                if abs(log_ca - math.log10(counts_by_token[a_prime])) > freq_match_dex:
                    continue
                return {"found": True, "a": a, "a_prime": a_prime, "b": b, "n_tries": tries}
    return {"found": False, "n_tries": tries}


class PlantContestedError(RuntimeError):
    """sec 11.2.3's HARD ASSERTION target. Fires -- RuntimeError, NEVER a
    warning -- if a planted key or value is CONTESTED by ANY natural
    occurrence the pre-check missed, or by a construction bug. This is the
    negative-test target of the smoke gate's [FORCED-FAIL] block (sec
    11.11): 'A structural check without a forced-fail negative test that
    runs to completion is not a check.'"""
    pass


def plant_and_verify_t2_window(orig_window: list, j0: int, k0: int, a: int, b: int) -> list:
    """sec 11.2.3 point 4 (the plant) + point 5 (THE HARD ASSERTION): writes
    w[j0]=a, w[j0+1]=b, w[k0]=a into a COPY of `orig_window`, then asserts
    `count(a in w) == 2` at EXACTLY {j0,k0} AND `count(b in w) == 1` at
    EXACTLY {j0+1}. Raises PlantContestedError (never returns a silently
    contested window) on any violation. 'This is the verification the old
    probe never performed, and it is the single line that makes F-I and
    (with V5) its value-side twin structurally impossible rather than
    statistically unlikely' (sec 11.2.3)."""
    w = list(orig_window)
    p = j0 + 1
    w[j0] = a
    w[p] = b
    w[k0] = a
    a_positions = [i for i, tok in enumerate(w) if tok == a]
    b_positions = [i for i, tok in enumerate(w) if tok == b]
    expected_a = sorted({j0, k0})
    if a_positions != expected_a or b_positions != [p]:
        raise PlantContestedError(
            f"T2 PLANT CONTESTED: expected count(a={a} in w)==2 at exactly {{j0={j0},k0={k0}}} "
            f"(got positions {a_positions}) AND count(b={b} in w)==1 at exactly {{p={p}}} "
            f"(got positions {b_positions}). PARAM_AXIS_SCALING_DESIGN.md sec 11.2.3: 'the "
            f"single line that makes F-I and (with V5) its value-side twin structurally "
            f"impossible rather than statistically unlikely.' Never fall back or continue "
            f"past this -- the window's plant is contested and must be DROPPED, not scored."
        )
    return w


def assign_t2_plant(orig_window: list, seq_len: int, delta_pool: list, pools: dict,
                     counts_by_token: list, window_seed: int,
                     max_triple_tries: int = T2_TRIPLE_MAX_TRIES) -> dict:
    """sec 11.2.3, THE FULL per-window procedure. `window_seed` MUST be
    computed by the caller from (corpus, window index) ONLY -- never rung,
    params, architecture, or batch size (sec 11.2: this is what makes the
    probe's difficulty RUNG-INDEPENDENT, sec 11.4.6). Returns:
      {'ok': True, 'j0':, 'k0':, 'p':, 'delta':, 'a':, 'a_prime':, 'b':,
       'planted_window': [...], 'delta_excluded_frac':, 'triple_tries':}
    or {'ok': False, 'reason': 'delta_pool_exhausted'|'triple_not_found', ...}
    -- sec 11.2.3 point 6: the caller must DROP and COUNT such windows; cap
    drops at T2_DROP_CAP_FRAC (2%) of n_plants, else the cell is VOID."""
    rng = random.Random(window_seed)
    d = rejection_sample_delta(delta_pool, seq_len, rng)
    if d["delta"] is None:
        return {"ok": False, "reason": "delta_pool_exhausted",
                "delta_tries": d["n_tries"], "delta_rejected": d["n_rejected"]}
    delta = d["delta"]
    k0 = rng.randint(delta + 2, seq_len - 2)   # sec 11.2.3 point 2: k0 ~ U[Delta+2, T-2]
    j0 = k0 - delta
    triple = draw_t2_triple(orig_window, pools["inverse_map"], counts_by_token, rng,
                             max_tries=max_triple_tries)
    if not triple["found"]:
        return {"ok": False, "reason": "triple_not_found", "triple_tries": triple["n_tries"],
                "delta_tries": d["n_tries"], "delta_rejected": d["n_rejected"]}
    a, a_prime, b = triple["a"], triple["a_prime"], triple["b"]
    planted = plant_and_verify_t2_window(orig_window, j0, k0, a, b)   # THE HARD ASSERTION
    meta = pools.get("val_meta", {}).get(a, {}).get(b, {})
    return {
        "ok": True, "j0": j0, "k0": k0, "p": j0 + 1, "delta": delta,
        "a": a, "a_prime": a_prime, "b": b, "planted_window": planted,
        "delta_tries": d["n_tries"], "delta_rejected": d["n_rejected"],
        "delta_excluded_frac": (d["n_rejected"] / d["n_tries"]) if d["n_tries"] else 0.0,
        "triple_tries": triple["n_tries"],
        # sec 11.4.3 step 2's per-plant difficulty record inputs (carried through per-record;
        # the stratified report / W2 Delta-sweep / n_demos diagnostic themselves are DRIVER-level
        # orchestration across multiple models/Deltas -- disclosed out of this module's scope,
        # see the REV 3 BUILD docstring block).
        "max_rival_p": meta.get("max_rival_p"), "rank_b_given_a": meta.get("rank_b_given_a"),
        "p_b_given_a": meta.get("p_b_given_a"), "count_ab": meta.get("count_ab"),
        "count_b": meta.get("count_b"),
    }


# =============================================================================
# 11.B -- sec 11.3 THE PROBE'S SIX ARMS, PINNED. All reuse
#     assign_placebo_positions / run_ablation_arm / row-replication VERBATIM.
#     Arms 2/3/3b/4 corrupt the PLANTED (arm-1) window at one position; arm 5
#     is a construction against the ORIGINAL (pre-plant) window (sec 11.3:
#     "its two-token difference from arm 1 IS the demonstration").
# =============================================================================
def draw_pool_replacement(p_key: list, exclude: set, rng: random.Random) -> int:
    """sec 11.3 arm 3b (POOL-PLACEBO): draws a token from P_key -- a
    WELL-TRAINED pool token -- rather than uniformly from the full
    vocabulary. Arms 2/3 replace with 'mostly OOD junk' (uniform-random);
    this is arm 4's SEVERITY-MATCHED comparator (sec 11.3: 'Arm 3b gives
    arm 4 a severity-matched comparator')."""
    while True:
        tok = rng.choice(p_key)
        if tok not in exclude:
            return tok


def run_t2_repaired_probe(model, val_tokens: torch.Tensor, seq_len: int, device: str,
                           corpus_name: str, delta_pool: list, pools: dict,
                           counts_by_token: list, n_plants: int,
                           vocab_size: int = VOCAB_SIZE,
                           eval_micro_batch: int = EVAL_MICRO_BATCH_DEFAULT,
                           drop_cap_frac: float = T2_DROP_CAP_FRAC) -> dict:
    """sec 11.2/11.3, THE REPAIRED PLANTED-COPY PROBE. Replaces
    run_t2_planted_copy (RETIRED with pick_t2_marker_tokens -- both were
    built around a SINGLE GLOBAL (tok_a,tok_b) pair; sec 11.2.5: 'one
    unlucky pair zeroes an entire corpus column').

    RUNG-INDEPENDENCE BY CONSTRUCTION (sec 11.4.6): this function takes NO
    `seed`/`rung` argument. Window sampling is seeded from
    `corpus_fixed_seed(corpus_name)` alone (imported from lm_pretrain_rd,
    the SAME provenance-seed convention as run_did_eval / the retired
    run_t2_planted_copy, offset to avoid stream collision), and each
    window's plant is seeded ONLY by `_combine_seed(corpus_name,
    "t2_window", row_idx)` (sec 11.2.3) -- so the planted windows are
    BYTE-IDENTICAL at every rung reading the same corpus. This is enforced
    by the function's OWN signature (no seed parameter to misuse), not
    merely by caller discipline.

    Windows whose plant cannot be constructed within 100 tries (sec 11.2.3
    point 6) are DROPPED and counted; if drops exceed `drop_cap_frac` of
    `n_plants`, the cell is VOID (`{'void': True, ...}`).

    Builds all SIX arms (sec 11.3) and returns per-plant records plus raw
    hit lists ready for check_t2b1_mechanism_exists / check_t2b1b_key_
    conditioned / check_t2a1_ceiling / check_t2a2_untrained_control /
    check_t2a3_ssm_calibration.

    AUDIT FIX (2026-07-12, independent adversarial audit, SERIOUS): sec
    11.2's own text -- "The floors are GATES, not hopes. The builder MUST
    recompute |P_key| and |P_val| in the model-free pre-pass and VOID the
    cell if either floor is missed" -- was computed by build_key_value_pools
    but NEVER CHECKED anywhere before this fix. A `pools` dict with e.g.
    |P_key|=3 would have run silently to a number. Checked HERE, before a
    single plant is sampled (sec 11.11 item 6: 'Pre-pass floor checks WITH
    VOID EXITS')."""
    missing_floors = [name for name, ok in (
        ("floor_pool_key", pools.get("floor_pool_key")),
        ("floor_pool_val", pools.get("floor_pool_val")),
        ("floor_licensed_b", pools.get("floor_licensed_b")),
    ) if not ok]
    if missing_floors:
        return {
            "void": True,
            "void_reason": (
                f"sec 11.2 pool floor(s) missed BEFORE any plant was sampled: {missing_floors} "
                f"(n_p_key={pools.get('n_p_key')}, median_p_val={pools.get('median_p_val')}, "
                f"n_licensed_b_ge2={pools.get('n_licensed_b_ge2')}). 'The floors are GATES, not "
                f"hopes' (PARAM_AXIS_SCALING_DESIGN.md sec 11.2)."),
            "n_dropped": 0, "n_plants_requested": n_plants, "records": [],
        }
    window_seed_base = corpus_fixed_seed(corpus_name) + 909090   # T2-probe-specific offset
    gen = torch.Generator(device=device).manual_seed(window_seed_base)
    x0, y0 = get_batch(val_tokens, n_plants, seq_len, gen)
    window0 = _make_window(x0, y0)
    orig_windows_cpu = [window0[b].cpu().tolist() for b in range(window0.shape[0])]

    specs = []
    drop_reasons: dict = {}
    for row_idx, orig_w in enumerate(orig_windows_cpu):
        window_seed = _combine_seed(corpus_name, "t2_window", row_idx)   # (corpus, window index) ONLY
        assignment = assign_t2_plant(orig_w, seq_len, delta_pool, pools, counts_by_token, window_seed)
        if not assignment["ok"]:
            drop_reasons[assignment["reason"]] = drop_reasons.get(assignment["reason"], 0) + 1
            continue
        spec = dict(assignment)
        spec["row_idx"] = len(specs)          # LOCAL index into the accepted-only lists below
        spec["orig_row_idx"] = row_idx         # provenance
        spec["orig_window"] = orig_w
        specs.append(spec)

    n_dropped = sum(drop_reasons.values())
    drop_frac = (n_dropped / n_plants) if n_plants else 1.0
    if drop_frac > drop_cap_frac:
        return {"void": True, "void_reason": f"plant construction dropped {n_dropped}/{n_plants} "
                                              f"windows ({drop_frac:.4f} > {drop_cap_frac} cap, "
                                              f"sec 11.2.3 point 6)", "drop_reasons": drop_reasons,
                "n_dropped": n_dropped, "n_plants_requested": n_plants, "records": []}
    if not specs:
        return {"void": False, "records": [], "n_dropped": n_dropped, "drop_reasons": drop_reasons,
                "n_plants_requested": n_plants}

    device_t = device
    all_windows_planted = [torch.tensor(s["planted_window"], device=device_t, dtype=torch.int64)
                            for s in specs]
    all_windows_orig = [torch.tensor(s["orig_window"], device=device_t, dtype=torch.int64)
                         for s in specs]
    window_cpu_planted = [s["planted_window"] for s in specs]

    sel_row_candidates = [[(s["k0"], s["j0"])] for s in specs]   # one plant/row (sec 11.2.5)
    true_arm_specs_placebo = [{"row_idx": s["row_idx"], "k": s["k0"], "j": s["j0"],
                                "p": s["p"], "delta": s["delta"]} for s in specs]
    placebo_seed = corpus_fixed_seed(corpus_name) + 909091
    true_arm_specs_placebo = assign_placebo_positions(true_arm_specs_placebo, sel_row_candidates,
                                                        window_cpu_planted, placebo_seed)
    for s, p in zip(specs, true_arm_specs_placebo):
        s["p_placebo"] = p["p_placebo"]
        s["delta_placebo"] = p["delta_placebo"]
        s["placebo_flagged"] = p["placebo_flagged"]

    ablate_rng = random.Random(_combine_seed(corpus_name, "t2_ablate"))
    p_key_pool = pools["P_key"]
    for s in specs:
        b = s["b"]
        s["repl_true"] = draw_exclusive_replacement(s["planted_window"], s["p"], b,
                                                      vocab_size, ablate_rng)
        if s["p_placebo"] is not None:
            s["repl_placebo"] = draw_exclusive_replacement(s["planted_window"], s["p_placebo"], b,
                                                             vocab_size, ablate_rng)
            # AUDIT FIX (2026-07-12, MINOR): excl must ALSO cover {a, a'} -- P_key contains
            # every retained key, including THIS plant's own a/a'. Without this, a pool draw
            # could rarely (measured ~0.4% of rows at real pool sizes) re-splice the plant's
            # own key back into the window at the placebo position, one MORE occurrence of a
            # single-token invariant this whole repair is built on. Direction was conservative
            # (depresses arm 3b, the T2b-1b 'pos' arm, so it could not have manufactured a
            # pass) but is closed regardless -- an invariant violation is never "acceptable in
            # this direction."
            excl = {s["planted_window"][s["p_placebo"]], b, s["a"], s["a_prime"]}
            s["repl_pool_placebo"] = draw_pool_replacement(p_key_pool, excl, ablate_rng)
        else:
            s["repl_placebo"] = None
            s["repl_pool_placebo"] = None
        s["repl_keyswap"] = s["a_prime"]           # arm 4: w[j0] := a' (deterministic)
        s["repl_noplant"] = s["a"]                 # arm 5: w[k0] := a, on the ORIGINAL window

    # ARM 1 -- INTACT: one shared batched forward pass over the planted windows.
    x_intact = torch.stack([w[:-1] for w in all_windows_planted])
    pred_chunks = []
    for start in range(0, x_intact.shape[0], eval_micro_batch):
        with torch.no_grad():
            logits = model(x_intact[start:start + eval_micro_batch])
        pred_chunks.append(logits.argmax(dim=-1))
    pred_intact = torch.cat(pred_chunks, dim=0)

    for s in specs:
        s["k"] = s["k0"]   # run_ablation_arm reads spec['k'] for BOTH corruption (arm5) and readout

    # ARMS 2/3/3b/4 -- corrupt the PLANTED window at one position.
    run_ablation_arm(model, all_windows_planted, specs, "p", "repl_true", "pred_true",
                      eval_micro_batch, device)
    run_ablation_arm(model, all_windows_planted, specs, "p_placebo", "repl_placebo", "pred_placebo",
                      eval_micro_batch, device)
    run_ablation_arm(model, all_windows_planted, specs, "p_placebo", "repl_pool_placebo",
                      "pred_pool_placebo", eval_micro_batch, device)
    run_ablation_arm(model, all_windows_planted, specs, "j0", "repl_keyswap", "pred_keyswap",
                      eval_micro_batch, device)
    # ARM 5 -- NO-PLANT: corrupt the ORIGINAL (pre-plant) window at k0 only.
    run_ablation_arm(model, all_windows_orig, specs, "k0", "repl_noplant", "pred_noplant",
                      eval_micro_batch, device)

    records = []
    for s in specs:
        b = s["b"]
        row_idx, k0 = s["row_idx"], s["k0"]
        hit_intact = int(pred_intact[row_idx, k0].item() == b)
        hit_true = int(s.get("pred_true") == b) if "pred_true" in s else None
        hit_placebo = int(s.get("pred_placebo") == b) if s.get("pred_placebo") is not None else None
        hit_pool_placebo = int(s.get("pred_pool_placebo") == b) if s.get("pred_pool_placebo") is not None else None
        hit_keyswap = int(s.get("pred_keyswap") == b) if "pred_keyswap" in s else None
        hit_noplant = int(s.get("pred_noplant") == b) if "pred_noplant" in s else None
        records.append({
            "row_idx": row_idx, "orig_row_idx": s["orig_row_idx"], "k": k0, "j": s["j0"],
            "delta": s["delta"], "a": s["a"], "a_prime": s["a_prime"], "b": b,
            "hit_intact": hit_intact, "hit_true_ablated": hit_true,
            "hit_placebo_ablated": hit_placebo, "hit_pool_placebo": hit_pool_placebo,
            "hit_keyswap": hit_keyswap, "hit_noplant": hit_noplant,
            "placebo_flagged": s["placebo_flagged"], "delta_placebo": s.get("delta_placebo"),
            # per-window ratio, retained as a DIAGNOSTIC only (not pooled correctly on its
            # own -- see delta_excluded_mass_pooled below for the aggregate to actually read).
            "delta_excluded_frac": s.get("delta_excluded_frac"),
            "delta_tries": s.get("delta_tries"), "delta_rejected": s.get("delta_rejected"),
            "max_rival_p": s.get("max_rival_p"), "rank_b_given_a": s.get("rank_b_given_a"),
            "p_b_given_a": s.get("p_b_given_a"), "count_ab": s.get("count_ab"),
            "count_b": s.get("count_b"),
        })

    acc_copy = _mean([r["hit_intact"] for r in records])
    acc_copy_se = binomial_se(sum(r["hit_intact"] for r in records), len(records))
    # AUDIT FIX (2026-07-12, MINOR): the per-window MEAN of `n_rejected/n_tries` is a BIASED
    # estimator of the true excluded-Delta mass for a geometric rejection process (measured:
    # true 5% mass -> the per-window mean reads ~2.4%; true 10% -> ~5.0% -- roughly half the
    # true value, which would make sec 11.2.3's '>5% -> disclose' trigger fire at roughly
    # double the intended threshold). POOL raw counts across all plants instead:
    # sum(n_rejected)/sum(n_tries) is the correct estimator of the population-level rate.
    tries_pool = [r["delta_tries"] for r in records if r.get("delta_tries")]
    rejected_pool = [r["delta_rejected"] for r in records if r.get("delta_rejected") is not None]
    delta_excluded_mass_pooled = (sum(rejected_pool) / sum(tries_pool)) if tries_pool and sum(tries_pool) else 0.0
    return {
        "void": False, "records": records, "n_plants": len(records),
        "n_plants_requested": n_plants, "n_dropped": n_dropped, "drop_reasons": drop_reasons,
        "acc_copy": acc_copy, "acc_copy_se": acc_copy_se,
        "delta_excluded_mass_pooled": delta_excluded_mass_pooled,
        "corpus": corpus_name,
    }


# =============================================================================
# 11.C -- sec 11.5 T2b-1 (RETAINED VERBATIM) + T2b-1b (NEW). Both are "the
#     identical paired exact sign test" (sec 11.3) on different arm pairs --
#     factored through ONE shared low-level helper so that claim is true by
#     construction (same code path), not merely by convention.
# =============================================================================
def _paired_sign_test(pos_hit: list, neg_hit: list) -> dict:
    """Shared low-level exact paired sign test: n_plus = #{pos_hit==1 AND
    neg_hit==0}, n_minus = #{pos_hit==0 AND neg_hit==1} (both computed by
    literal `==1`/`==0` checks, so an unresolved None entry in either list
    silently contributes to neither -- matching this file's pre-existing
    behaviour). H0: p=0.5. `passes` iff p<0.001 AND n_plus>n_minus."""
    n_plus = sum(1 for p, n in zip(pos_hit, neg_hit) if p == 1 and n == 0)
    n_minus = sum(1 for p, n in zip(pos_hit, neg_hit) if p == 0 and n == 1)
    n = n_plus + n_minus
    if n == 0:
        return {"n_plus": 0, "n_minus": 0, "n_informative": 0, "p_value": 1.0, "passes": False,
                "note": "no discordant pairs -- both arms always agree"}
    p_value = _exact_binomial_two_sided_p(n_plus, n)
    return {"n_plus": n_plus, "n_minus": n_minus, "n_informative": n, "p_value": p_value,
            "passes": bool(p_value < 0.001 and n_plus > n_minus)}


def check_t2b1_mechanism_exists(paired_records: list) -> dict:
    """T2b-1 (sec 9.4/sec 11.5, RETAINED VERBATIM -- 'unchanged'): exact
    paired sign test, arm 3 (PLACEBO-ABLATED) vs arm 2 (TRUE-ABLATED). n_plus
    = #{placebo-ablation still correct AND true-ablation wrong} (true
    corruption did the damage, placebo didn't). Reject at p<0.001 AND
    n_plus>n_minus to claim 'mechanism exists'."""
    return _paired_sign_test([r["hit_placebo_ablated"] for r in paired_records],
                              [r["hit_true_ablated"] for r in paired_records])


def check_t2b1b_key_conditioned(paired_records: list) -> dict:
    """T2b-1b (sec 11.3/11.5, NEW): 'the identical paired exact sign test',
    arm 3b (POOL-PLACEBO) vs arm 4 (KEY-SWAP). n_plus = #{pool-placebo still
    correct AND key-swap wrong} -- breaking IDENTITY (key-swap) did the
    damage, a SEVERITY-MATCHED generic corruption (pool-placebo) didn't.
    This is the ONLY check in the design that can separate key-conditioned
    associative recall from unconditional in-context salience/rarity
    (sec 11.3)."""
    return _paired_sign_test([r["hit_pool_placebo"] for r in paired_records],
                              [r["hit_keyswap"] for r in paired_records])


def _exact_binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0

    def pmf(x):
        return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

    p_obs = pmf(k)
    total = 0.0
    for x in range(n + 1):
        px = pmf(x)
        if px <= p_obs * (1 + 1e-9):
            total += px
    return min(1.0, total)


def binomial_se(hits: int, n: int) -> float:
    """The plain (Wald / normal-approximation) binomial SE, sqrt(p(1-p)/n).
    RETAINED from the pre-REV-3 build (the T2b-2 bound it fed is gone, sec
    11.6, but the SE itself is still used for acc_copy's own reported CI and
    T2a-2/T2a-3's diagnostics)."""
    if n == 0:
        return float("nan")
    p = hits / n
    return math.sqrt(p * (1 - p) / n)


# =============================================================================
# 11.D -- sec 11.4 T2a (RE-PINNED) + sec 11.4.5 T1c (RE-PINNED). Generic
#     CHECK FUNCTIONS over already-computed records/cells -- see the module
#     docstring's SCOPE BOUNDARY: actual execution against RWKV7-Goose-1.5B
#     / gpt2-large / falcon-mamba-7b is a driver-level task, out of scope
#     here (disclosed, matches this file's existing T2a convention).
# =============================================================================
def _acc(records: list, hit_key: str) -> float:
    vals = [r[hit_key] for r in records if r.get(hit_key) is not None]
    return _mean(vals) if vals else float("nan")


def _decile_bucket(records: list, key_fn, n_bins: int = 10) -> list:
    """Splits `records` into `n_bins` equal-COUNT buckets by ascending
    `key_fn(record)` (rank-based deciles, stable-sort ties). sec 11.4.1
    pins the BAR ('>=0.90 at the Delta-median', '>=0.75 in every Delta-
    decile') but not an exact binning algorithm; this file's own choice,
    documented so an auditor can check it: rank-based equal-count deciles,
    'the Delta-median' = the middle (index n_bins//2) decile."""
    ordered = sorted(records, key=key_fn)
    n = len(ordered)
    return [ordered[(i * n) // n_bins:((i + 1) * n) // n_bins] for i in range(n_bins)]


def check_t2a1_ceiling(records: list, t2b1_result: dict, t2b1b_result: dict) -> dict:
    """T2a-1 CEILING (sec 11.4.1, GATING). The 0.90/0.75 bar is UNCHANGED
    from sec 9.4; THREE legs are NEW. All FIVE must hold simultaneously, on
    ONE (witness, corpus)'s `records` (a `run_t2_repaired_probe` output).
      (i)   acc_copy >= 0.90 at the Delta-median
      (ii)  acc_copy >= 0.75 in EVERY Delta-decile
      (iii) PRIOR = acc_copy_noplant <= 0.05
      (iv)  KS = acc_copy - acc_copy_keyswap >= 0.50 AND T2b-1b passes (p<0.001)
      (v)   T2b-1 passes (p<0.001)
    `t2b1_result`/`t2b1b_result`: check_t2b1_mechanism_exists(records) /
    check_t2b1b_key_conditioned(records) outputs, computed by the caller on
    the SAME records (kept as explicit inputs rather than recomputed here,
    so a caller can share one T2b-1/T2b-1b computation across T2a-1 and
    T2b's own rung-admissibility use)."""
    deciles = _decile_bucket(records, key_fn=lambda r: r["delta"])
    decile_accs = [_acc(b, "hit_intact") for b in deciles]
    median_decile = deciles[len(deciles) // 2]
    acc_at_median = _acc(median_decile, "hit_intact")
    prior = _acc(records, "hit_noplant")
    acc_copy_all = _acc(records, "hit_intact")
    acc_keyswap = _acc(records, "hit_keyswap")
    ks = acc_copy_all - acc_keyswap if not (math.isnan(acc_copy_all) or math.isnan(acc_keyswap)) else float("nan")

    leg_i = (not math.isnan(acc_at_median)) and acc_at_median >= 0.90
    leg_ii = bool(decile_accs) and all((not math.isnan(a)) and a >= 0.75 for a in decile_accs)
    leg_iii = (not math.isnan(prior)) and prior <= 0.05
    leg_iv = (not math.isnan(ks)) and ks >= 0.50 and bool(t2b1b_result.get("passes"))
    leg_v = bool(t2b1_result.get("passes"))

    return {
        "passes": leg_i and leg_ii and leg_iii and leg_iv and leg_v,
        "acc_at_median": acc_at_median, "decile_accs": decile_accs,
        "prior": prior, "ks": ks, "acc_copy": acc_copy_all, "acc_copy_keyswap": acc_keyswap,
        "leg_i_median_ge_090": leg_i, "leg_ii_all_deciles_ge_075": leg_ii,
        "leg_iii_prior_le_005": leg_iii, "leg_iv_ks_ge_050_and_t2b1b": leg_iv,
        "leg_v_t2b1_passes": leg_v,
        "t2b1_p_value": t2b1_result.get("p_value"), "t2b1b_p_value": t2b1b_result.get("p_value"),
    }


def check_t2a2_untrained_control(records: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """T2a-2 NEGATIVE CONTROL (sec 11.4.2, GATING, NEW). A randomly-
    initialised, UNTRAINED model of the 14M rung's exact architecture must
    read acc_copy <= 0.02 with a KS bootstrap 95% CI INCLUDING 0. Fail =>
    INSTRUMENT-INVALID, HALT (sec 11.4.2: 'If an untrained model passes the
    probe, the probe is passable with no learned mechanism'). Each plant is
    its own independent row by construction (one plant per window), so the
    existing clustered_bootstrap_ci (which resamples over `row_idx`)
    reduces to an ordinary per-plant bootstrap here -- reused, not
    reimplemented."""
    def stat_ks(recs):
        return _acc(recs, "hit_intact") - _acc(recs, "hit_keyswap")
    acc_copy = _acc(records, "hit_intact")
    ks_pt, ks_lo, ks_hi = clustered_bootstrap_ci(records, stat_ks, n_boot=n_boot, seed=seed)
    ci_includes_zero = (ks_lo is not None and ks_hi is not None and ks_lo <= 0 <= ks_hi)
    passes = (not math.isnan(acc_copy)) and acc_copy <= 0.02 and ci_includes_zero
    return {"passes": passes, "acc_copy": acc_copy, "ks_point": ks_pt, "ks_ci": [ks_lo, ks_hi]}


def check_t2a3_ssm_calibration(records: list, t2b1_result: dict, t2b1b_result: dict,
                                n_boot: int = 2000, seed: int = 0) -> dict:
    """T2a-3 SSM CALIBRATION (sec 11.4.2, GATING ON THE CAUSAL LEGS ONLY,
    NEW). `falcon-mamba-7b` must pass T2b-1 AND T2b-1b (p<0.001) and show
    KS>0 with a bootstrap 95% CI excluding 0 (sec 11.4.2, conceded to A-M1:
    KS is a genuine SINGLE-token contrast, replacing the draft's two-token
    `acc_copy - acc_copy_noplant`). `acc_copy` is reported, NOT held to
    0.90 -- this witness is demoted from the 0.90 ceiling gate (sec 11.4.2)
    and 'can no longer save the gate' (T2a-1 still requires W1+W2)."""
    def stat_ks(recs):
        return _acc(recs, "hit_intact") - _acc(recs, "hit_keyswap")
    ks_pt, ks_lo, ks_hi = clustered_bootstrap_ci(records, stat_ks, n_boot=n_boot, seed=seed)
    ks_positive_excludes_zero = (ks_lo is not None and ks_lo > 0)
    causal_pass = (bool(t2b1_result.get("passes")) and bool(t2b1b_result.get("passes"))
                   and ks_positive_excludes_zero)
    return {"passes": causal_pass, "acc_copy": _acc(records, "hit_intact"),
            "ks_point": ks_pt, "ks_ci": [ks_lo, ks_hi],
            "t2b1_passes": t2b1_result.get("passes"), "t2b1b_passes": t2b1b_result.get("passes")}


def check_t1c_reference_did(cell_w1: dict, cell_w2: dict) -> dict:
    """T1c RE-PINNED (sec 11.4.5, GATING). Runs the MAIN metric (arms A/B/C,
    sec 9.2 -- NOT the T2 probe) on W1 (RWKV7-Goose-1.5B) and W2
    (gpt2-large), same candidate population, both corpora; requires
    `DiD > 0` with a clustered-bootstrap 95% CI excluding 0 ON EACH. Fail =>
    INSTRUMENT-INVALID, HALT. `cell_w1`/`cell_w2` are `finalize_cell(...)`
    outputs (already computed by the caller, on each reference model) --
    this function reads only `did_ci`, never recomputes DiD."""
    def _pass(cell):
        ci = cell.get("did_ci", [None, None])
        lo = ci[0] if ci else None
        return lo is not None and lo > 0

    w1_pass = _pass(cell_w1)
    w2_pass = _pass(cell_w2)
    return {"passes": w1_pass and w2_pass, "w1_pass": w1_pass, "w2_pass": w2_pass,
            "w1_did": cell_w1.get("did"), "w1_did_ci": cell_w1.get("did_ci"),
            "w2_did": cell_w2.get("did"), "w2_did_ci": cell_w2.get("did_ci")}


# =============================================================================
# 13. sec 9.6 admissibility gates -- sec 9.6 item 5 REPAIRED by sec 11.5:
#     "T1a and T2b-1 and T2b-1b all pass; failure of any => FLOOR rung
#     (excluded from the fit, reported)." T2b-2 is GONE from this function
#     entirely (sec 11.6/11.8's own "did anything from the retired T2b-2
#     survive into the verdict path?" question: no -- there is no
#     `t2b2_by_corpus` parameter, no t2b2 lookup, and no VOID trigger tied
#     to a rung-level mechanism check anymore). The ONLY VOID trigger left
#     in this function is the sec 9.2 placebo-distribution-match failure
#     (a CELL-construction defect, not a rung-mechanism finding).
# =============================================================================
def check_rung_admissible(cell_by_corpus: dict, tok_per_param: float, checkpoint_quiesced: bool,
                           t2b1_by_corpus: dict, t2b1b_by_corpus: dict,
                           min_candidates: int = CANDIDATE_FLOOR_RESOLVED) -> dict:
    """sec 9.6 (REPAIRED by sec 11.5/11.7): a rung enters the admissible set
    A iff ALL items hold, on BOTH corpora (item 6 -- 'A rung is admissible
    only if admissible on both'; never silently narrowed to the corpus
    where the instrument passes, regate S-5's precedent). `min_candidates`
    defaults to sec 11.7's pinned floor (>=8,000 resolved candidates,
    superseding sec 9.6 item 7's old >=4,096); pass the sec 11.7 pre-pass's
    resolved `N_rows`-derived floor explicitly if it differs. Returns a
    per-item breakdown plus the overall boolean so a caller can see exactly
    which gate failed.

    REV 3 (sec 11.5, 2026-07-12): `t2b2_by_corpus` is REMOVED (T2b-2 is
    retired, sec 11.6) and `t2b1b_by_corpus` is ADDED (the new
    key-conditioned mechanism gate, sec 11.3/11.5). A rung failing T1a,
    T2b-1, OR T2b-1b is now UNIFORMLY 'FLOOR_OR_EXCLUDED' -- T2b-2's old
    VOID-on-mechanism-failure trigger is gone, not replaced (sec 11.6:
    'No ceiling check replaces it')."""
    reasons = {}
    corpora = sorted(cell_by_corpus)
    reasons["both_corpora_present"] = set(corpora) >= {"wikitext-mix-ext", "openr1-mix-ext"} \
        if len(corpora) else False
    reasons["checkpoint_quiesced"] = checkpoint_quiesced
    reasons["tok_per_param_floor"] = tok_per_param >= TOK_PER_PARAM_PRIMARY_FIT_FLOOR

    per_corpus_ok = {}
    for c in corpora:
        cell = cell_by_corpus[c]
        ok = (
            not cell.get("cell_void_placebo_match", True)
            and cell.get("n_candidates_resolved", 0) >= min_candidates
            and cell.get("t1a_pass_did_ci_excludes_zero", False)
            and t2b1_by_corpus.get(c, {}).get("passes", False)
            and t2b1b_by_corpus.get(c, {}).get("passes", False)
        )
        per_corpus_ok[c] = ok
    reasons["per_corpus"] = per_corpus_ok
    # AUDIT FIX (2026-07-12, pre-REV-3, RETAINED): a placebo-match failure is a VOID, not a
    # FLOOR. sec 9.2 is literal: "If the flagged fraction exceeds 5% in any (rung, corpus)
    # cell, the placebo is not distribution-matched and the cell is VOID." VOID means "HALT,
    # no verdict, diagnose" (sec 9.5's precedence VOID -> FLOOR -> table); FLOOR/EXCLUDED
    # means "quietly drop this rung and fit the trend through the others." Opposite
    # operational consequences from the same failure. Defaults to True (fail-closed) when
    # the key is absent. THIS IS NOW THE ONLY VOID TRIGGER IN THIS FUNCTION (sec 11.6's
    # T2b-2 retirement removed the other one; nothing replaces it).
    any_placebo_void = any(cell_by_corpus[c].get("cell_void_placebo_match", True) for c in corpora)
    reasons["placebo_distribution_matched"] = not any_placebo_void

    admissible = (
        reasons["both_corpora_present"]
        and reasons["checkpoint_quiesced"]
        and reasons["tok_per_param_floor"]
        and all(per_corpus_ok.get(c, False) for c in corpora)
    )
    if any_placebo_void:
        verdict = "VOID"
    else:
        verdict = "ADMISSIBLE" if admissible else "FLOOR_OR_EXCLUDED"
    return {"admissible": admissible, "verdict_if_not_admissible": verdict, "reasons": reasons}


# =============================================================================
# 13.A -- sec 11.7 THE N_ROWS PRE-PASS, PINNED. MODEL-FREE (reads ONLY the
#     corpus, tokenizer, and TRAIN-split modal table -- NEVER a checkpoint)
#     -- fixes ONE N_rows used at EVERY rung and BOTH corpora, BEFORE any
#     checkpoint is loaded (sec 11.7: "the pre-pass may raise it, never
#     lower it"). The sec 9.2 placebo-fallback-flagged-fraction<=5% check
#     is measured and reported here but is NOT part of the search (sec
#     11.7/A-S8: non-monotone in N_rows, so including it would make the
#     search non-terminating) -- it is applied downstream as a per-cell
#     pass/fail exactly as before (`finalize_cell`/`check_rung_admissible`).
# =============================================================================
def _model_free_cell_counts(val_tokens: torch.Tensor, mode_next: torch.Tensor, seq_len: int,
                             n_rows: int, c_max: int, corpus_name: str, device: str,
                             min_sep: int = 2) -> dict:
    """ONE (N_rows, corpus) measurement for the sec 11.7 pre-pass: samples
    `n_rows` windows (seeded ONLY by `corpus_fixed_seed(corpus_name)` --
    rung-independent by construction, sec 11.4.6), then runs sec 9.0's
    candidate detection + sec 9.2's per-row cap + placebo-position
    assignment -- ALL MODEL-FREE, no checkpoint touched. Counts (a) rows
    contributing >=1 resolved candidate, (b) total resolved candidates,
    (c) the placebo-fallback-flagged fraction (reported, not searched)."""
    seed = corpus_fixed_seed(corpus_name) + 424242   # SAME provenance-seed convention as run_did_eval
    gen = torch.Generator(device=device).manual_seed(seed)
    x0, y0 = get_batch(val_tokens, n_rows, seq_len, gen)
    x_cpu = x0.cpu().tolist()
    y_cpu = y0.cpu().tolist()
    mode_next_cpu = mode_next.tolist()
    row_candidates, _ = detect_candidates_and_baseline(x_cpu, y_cpu, mode_next_cpu, min_sep)
    sel = select_candidates_per_row(row_candidates, c_max, seed)
    specs = true_arm_specs(sel)
    specs = assign_placebo_positions(specs, sel, x_cpu, seed)
    resolved = [s for s in specs if s["p_placebo"] is not None]
    contributing_rows = len({s["row_idx"] for s in resolved})
    flagged_frac = (sum(1 for s in specs if s["placebo_flagged"]) / len(specs)) if specs else 1.0
    return {
        "n_rows": n_rows, "corpus": corpus_name,
        "n_candidates_total": len(specs), "n_candidates_resolved": len(resolved),
        "contributing_rows": contributing_rows, "placebo_flagged_frac": flagged_frac,
    }


def resolve_n_rows_pre_pass(val_tokens_by_corpus: dict, mode_next_by_corpus: dict, seq_len: int,
                             device: str, c_max: int = C_MAX_DEFAULT,
                             ladder: tuple = N_ROWS_SEARCH_LADDER) -> dict:
    """sec 11.7 THE PIN: N_rows = the SMALLEST power of two in [2048,8192]
    such that BOTH corpora clear >= ROW_FLOOR_CONTRIBUTING (1,500)
    contributing rows AND >= CANDIDATE_FLOOR_RESOLVED (8,000) resolved
    candidates. If 8192 does not clear on both, the read is VOID with a
    stated reason -- the search terminates (sec 11.7: "the search
    terminates"). `val_tokens_by_corpus`/`mode_next_by_corpus`: {corpus:
    tensor} for BOTH corpora (sec 9.6 item 6 -- both, always; this function
    refuses nothing itself if the caller passes only one, but a caller
    building sec 11.8.1's admissible set must supply both).

    Returns {'n_rows': int|None, 'void': bool, 'void_reason': str|None,
    'measurements': {n_rows: {corpus: cell_counts}}, 'corpora': [...]}."""
    corpora = sorted(val_tokens_by_corpus)
    measurements = {}
    for n_rows in ladder:
        per_corpus = {}
        for corpus in corpora:
            per_corpus[corpus] = _model_free_cell_counts(
                val_tokens_by_corpus[corpus], mode_next_by_corpus[corpus], seq_len, n_rows,
                c_max, corpus, device)
        measurements[n_rows] = per_corpus
        clears = all(
            per_corpus[c]["contributing_rows"] >= ROW_FLOOR_CONTRIBUTING
            and per_corpus[c]["n_candidates_resolved"] >= CANDIDATE_FLOOR_RESOLVED
            for c in corpora
        )
        if clears:
            return {"n_rows": n_rows, "void": False, "void_reason": None,
                    "measurements": measurements, "corpora": corpora}
    return {
        "n_rows": None, "void": True,
        "void_reason": (
            f"N_rows search exhausted at {ladder[-1]} without clearing >= "
            f"{ROW_FLOOR_CONTRIBUTING} contributing rows AND >= {CANDIDATE_FLOOR_RESOLVED} "
            f"resolved candidates on BOTH corpora (PARAM_AXIS_SCALING_DESIGN.md sec 11.7)."),
        "measurements": measurements, "corpora": corpora,
    }


# =============================================================================
# 13.B -- sec 11.8.1 THE ADMISSIBLE-SET COMMIT PROTOCOL, MECHANICAL. `A` is
#     a table of BOOLEANS AND METADATA ONLY -- "NO DiD, gap_true,
#     gap_placebo, acc_copy, S1, or S2 field, and no quantity derived from
#     any of them." `emit_admissible_set_json` MECHANICALLY scans its own
#     assembled payload and REFUSES to emit if any DiD-shaped field is
#     present anywhere -- this module can therefore produce the artifact
#     WITHOUT ever having computed a DiD value for the cells it describes,
#     as a property this code guarantees, not one a caller merely promises.
# =============================================================================
_ADMISSIBLE_SET_FORBIDDEN_TOKENS = frozenset({"did", "gap", "s1", "s2", "logp", "logprob"})

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalize_tokens(text: str) -> set:
    """AUDIT-HARDENED (2026-07-12, independent adversarial audit, SERIOUS): the pre-audit
    version split ONLY on non-alnum boundaries, so camelCase ('gapTrue' -> one token
    'gaptrue') and simple plurals ('gaps', 'dids', 'logps') both EVADED the forbidden-token
    check (measured: both emitted cleanly pre-fix). Now: (1) insert a boundary at every
    lower/digit -> UPPER transition BEFORE lowercasing ('gapTrue' -> 'gap_True' -> tokens
    {'gap','true'}); (2) split on non-alnum boundaries; (3) strip a single trailing 's' from
    each token (len>1) so 'gaps'/'dids'/'logps' normalize to their singular root. Used on
    BOTH dict keys and string VALUES (the pre-fix version scanned keys only -- a `{"note":
    "DiD 0.19 at rungs with acc_copy 0.0"}` value-only leak, verbatim the string sec 11.10's
    contamination ledger flags as live in this repo, evaded it entirely)."""
    boundary_marked = _CAMEL_BOUNDARY_RE.sub("_", str(text))
    raw_tokens = set(re.split(r"[^a-z0-9]+", boundary_marked.lower())) - {""}
    normalized = set()
    for t in raw_tokens:
        normalized.add(t)
        if len(t) > 1 and t.endswith("s"):
            normalized.add(t[:-1])
    return normalized


def _key_tokens(key: str) -> set:
    """Back-compat alias, RETAINED name for any external caller -- delegates to the hardened
    `_normalize_tokens`. 'n_candidates_resolved' still correctly normalizes to
    {'n','candidates','resolved','candidate'} -- NOT flagged (the substring 'did' inside
    'candidates' is not, and was never, a WHOLE token)."""
    return _normalize_tokens(key)


_ADMISSIBLE_SET_FORBIDDEN_TOKEN_PAIRS = (
    frozenset({"acc", "copy"}),   # 'acc_copy' / 'accCopy' / 'AccCopy'
    frozenset({"log", "prob"}),   # 'logProb' -- camelCase-splits to {'log','prob'}, neither of
                                    # which alone is forbidden ('log' and 'prob' are each too
                                    # generic to ban outright -- 'log_softmax', 'probability' are
                                    # legitimate), but the PAIR together is exactly 'logp' spelled
                                    # with a word boundary the single-token 'logp'/'logprob'
                                    # entries do not cover.
)


def _contains_forbidden_token(text) -> bool:
    """True iff ANY normalized token of `text` (a dict key OR a string leaf VALUE) is a
    forbidden DiD-shaped token, `text`'s de-separated lowercase form contains 'acc_copy' in
    any spelling (underscored, camelCase, or bare), OR the normalized token set contains a
    full forbidden PAIR (sec _ADMISSIBLE_SET_FORBIDDEN_TOKEN_PAIRS -- catches compounds like
    'logProb' that camelCase-split into two individually-innocuous tokens, {'log','prob'},
    neither of which the single-token forbidden set alone would flag)."""
    kl = re.sub(r"[^a-z0-9]", "", str(text).lower())
    if "acccopy" in kl:
        return True
    tokens = _normalize_tokens(text)
    if tokens & _ADMISSIBLE_SET_FORBIDDEN_TOKENS:
        return True
    return any(pair <= tokens for pair in _ADMISSIBLE_SET_FORBIDDEN_TOKEN_PAIRS)


def _is_forbidden_admissible_set_key(key: str) -> bool:
    return _contains_forbidden_token(key)


def build_admissible_set_row(rung: str, corpus: str, gates: dict) -> dict:
    """sec 11.8.1 item 1: ONE row of `A` for ONE (rung, corpus) cell. `gates`
    is a dict of {gate_name: bool} for T1a, T2b-1, T2b-1b, and sec 9.6 items
    1-6 (checkpoint-exists, tok-per-param floor, quiesced, T3-same-
    checkpoint, both-corpora, sample-size) -- every value MUST be a plain
    bool (enforced here, not merely by convention); `admissible` is the AND
    of all of them. No numeric DiD-shaped quantity may appear as a `gates`
    value or key (also enforced, via `_is_forbidden_admissible_set_key`)."""
    for k, v in gates.items():
        if not isinstance(v, bool):
            raise TypeError(
                f"build_admissible_set_row: gate {k!r}={v!r} is not a bool. sec 11.8.1: 'a "
                f"reviewer must be able to open it without being contaminated for the "
                f"beta-fit' -- every gate value must be a pass/fail boolean, never a numeric "
                f"quantity."
            )
        if _is_forbidden_admissible_set_key(k):
            raise ValueError(
                f"build_admissible_set_row: gate name {k!r} looks DiD-shaped (sec 11.8.1: "
                f"'NO DiD, gap_true, gap_placebo, acc_copy, S1, or S2 field'). Rename it."
            )
    return {"rung": rung, "corpus": corpus, "gates": dict(gates),
            "admissible": all(gates.values())}


def _assert_no_did_shaped_fields(obj, path: str = "$", strict_no_float: bool = False) -> None:
    """Recursively scans a JSON-able structure for (a) any DICT KEY that is a forbidden
    DiD-shaped token (checked via `_is_forbidden_admissible_set_key`'s hardened, camelCase/
    plural-aware tokenizer -- not a raw substring, so 'n_candidates_resolved' does not
    false-positive on 'candidates' containing 'did'); (b) any STRING VALUE whose own tokens
    contain a forbidden DiD-shaped word (closes the AUDIT-FOUND hole where a value like
    `{"note": "DiD 0.19 at rungs with acc_copy 0.0"}` evaded a keys-only scan entirely); and
    (c) when `strict_no_float=True`, ANY float leaf anywhere (the ROBUST closing move: a
    sufficiently creative field NAME can always evade a keyword list -- e.g. 'doubledifference'
    contains none of {did,gap,s1,s2,logp} as a substring OR a token -- but DiD/gap/acc_copy/
    S1/S2/logp are ALL, without exception, continuous-valued floats, and this artifact has no
    legitimate reason to carry one anywhere in its `rows` -- 'a table of booleans and metadata
    only', sec 11.8.1). Traverses dict/list/TUPLE (a pre-fix version only traversed list,
    letting a tuple-nested DiD value through). Raises ValueError naming the offending
    key/path/value on any hit. sec 11.8.1 item 1: 'NO DiD ... field, and no quantity derived
    from any of them.'"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_forbidden_admissible_set_key(k):
                raise ValueError(
                    f"admissible_set_A.json REFUSES to emit: forbidden DiD-shaped key {k!r} "
                    f"found at {path}. sec 11.8.1 item 1: 'NO DiD, gap_true, gap_placebo, "
                    f"acc_copy, S1, or S2 field, and no quantity derived from any of them.'"
                )
            _assert_no_did_shaped_fields(v, f"{path}.{k}", strict_no_float)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_did_shaped_fields(v, f"{path}[{i}]", strict_no_float)
    elif isinstance(obj, str):
        if _contains_forbidden_token(obj):
            raise ValueError(
                f"admissible_set_A.json REFUSES to emit: the STRING VALUE at {path} contains a "
                f"forbidden DiD-shaped token ({obj!r}). A free-text field is not exempt from "
                f"sec 11.8.1's 'NO DiD ... field' rule."
            )
    elif strict_no_float and isinstance(obj, float):
        raise ValueError(
            f"admissible_set_A.json REFUSES to emit: a FLOAT leaf at {path} ({obj!r}) inside "
            f"`rows`. sec 11.8.1: 'a table of booleans and metadata only' -- DiD/gap/acc_copy/"
            f"S1/S2/logp are ALL continuous-valued floats without exception, so no row entry "
            f"may carry one, regardless of its key's name (a name can always be respelled to "
            f"evade a keyword list; a float's TYPE cannot)."
        )


def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit_admissible_set_json(rows: list, commit_sha: str, cli_config: dict,
                              schema_version: str = "1.0") -> dict:
    """sec 11.8.1 item 1: assembles the FULL admissible_set_A.json payload
    from a list of `build_admissible_set_row(...)` dicts -- one per
    (rung, corpus) cell R0 covers. `commit_sha`: the code commit that
    PRODUCED the gate verdicts (caller's responsibility, e.g. `git
    rev-parse HEAD` at call time). `cli_config`: the exact CLI/config used
    to run the gates (a plain dict -- also scanned for forbidden KEY names
    and string values; floats ARE permitted here -- e.g. a genuine
    `flagged_frac_threshold: 0.05` operational constant -- since `cli_config`
    is legitimately free-form operator metadata, unlike `rows`).

    MECHANICALLY ENFORCES 'no DiD field, ever' via `_assert_no_did_shaped_fields`
    over the ENTIRE assembled payload -- `rows` scanned with `strict_no_float=True`
    (sec 11.8.1: 'a table of booleans and metadata only' -- no row entry may carry
    ANY float, closing the keyword-evasion hole a creatively-named field could
    otherwise exploit), everything else (including `cli_config`) scanned for
    forbidden key names / string values only. This is what lets this function be
    called 'without computing or exposing any DiD value' as a property the
    function itself guarantees, not one a caller merely promises. Per sec 11.8.1
    item 2, THE CALLER is responsible for committing the returned payload as its
    OWN, standalone git commit whose message begins with the literal tag
    `ADMISSIBLE-SET-COMMIT:` -- that commit discipline is not something this
    library function can enforce."""
    payload = {
        "schema_version": schema_version,
        "generated_at": _utc_now_iso(),
        "commit_sha": commit_sha,
        "cli_config": dict(cli_config),
        "rows": rows,
    }
    _assert_no_did_shaped_fields(payload["rows"], "$.rows", strict_no_float=True)
    _assert_no_did_shaped_fields(
        {k: v for k, v in payload.items() if k != "rows"}, "$", strict_no_float=False)
    return payload


# =============================================================================
# 14. sec 9.5 verdict-map machinery (trend fit + TOST-FLAT + the exhaustive
#     table). Generic/pluggable: the exact statistical treatment of the
#     normalized M(r) gets a final pass once sec 9.1's form is known; the
#     DiD-level machinery above is what this build's validation targets.
# =============================================================================
def ols_slope(xs: list, ys: list) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den


def bootstrap_trend_ci(per_rung_row_values: dict, log10_params: dict, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """per_rung_row_values: {rung_name: {row_idx: value}} (already-normalized
    M(r) per row, i.e. the caller has ALREADY run compute_capacity_metric per
    row via a per-row DiD/baseline pair -- left to the caller since the exact
    per-row normalization depends on sec 9.1's still-open form). Resamples
    rows WITHIN EACH RUNG independently per bootstrap draw, refits beta =
    OLS slope of per-rung mean M against log10(params)."""
    rungs = sorted(per_rung_row_values)
    xs = [log10_params[r] for r in rungs]
    point_ys = [_mean(list(per_rung_row_values[r].values())) for r in rungs]
    point_beta = ols_slope(xs, point_ys)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        ys = []
        for r in rungs:
            vals = list(per_rung_row_values[r].values())
            resampled = [rng.choice(vals) for _ in range(len(vals))]
            ys.append(_mean(resampled))
        boots.append(ols_slope(xs, ys))
    boots.sort()
    return {"beta_point": point_beta, "boots": boots,
            "ci95": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]}


def tost_flat(boots: list, delta: float, alpha: float = 0.10) -> dict:
    """sec 9.5: 'FLAT' requires the 95% CI to include 0 AND 'TOST at 90% CI' to
    establish |beta| < delta.

    AUDIT FIX (2026-07-12): the previous body took `b[int(alpha*n)] .. b[int((1-alpha)*n)-1]`
    with alpha=0.10, i.e. the 10th..90th percentiles -- an **80% interval**, while the
    returned key was (and the spec requires) `ci90`. An 80% interval is NARROWER, so TOST
    passed more easily and FLAT would have been declared MORE often than the
    pre-registration permits -- a verdict-boundary error in the "manufacture a result"
    direction, on the one branch (FLAT-COUPLED) sec 9.5 calls "the third outcome, and it is
    the one sec 5.2 could not express." `alpha` is the TOTAL tail mass (the standard TOST
    convention: alpha=0.10 <=> two one-sided tests at 5% <=> the 90% CI)."""
    b = sorted(boots)
    n = len(b)
    lo = b[int(alpha / 2 * n)]
    hi = b[min(n - 1, int((1 - alpha / 2) * n) - 1)]
    return {"tost_pass": bool(lo > -delta and hi < delta), "ci90": [lo, hi],
            "delta": delta, "alpha_total_tail_mass": alpha}


def classify_recall_trend(ci95: list, tost: dict) -> str:
    lo, hi = ci95
    if lo > 0:
        return "RISES"
    if hi < 0:
        return "DECLINES"
    return "FLAT" if tost["tost_pass"] else "INDETERMINATE"


def compute_verdict(factor1_trend: str, span_frac_monotone_over_A: bool,
                     n_admissible_rungs: int, t1a_positive_count: int,
                     *, factor1_trend_s2: str) -> str:
    """sec 9.5's exhaustive table, precedence VOID -> FLOOR -> the table, with sec 9.1.5's
    S2 disagreement rule enforced FIRST (it can only withhold a verdict, never create one).
    Callers are responsible for having already produced BOTH classifications via
    `classify_recall_trend` (one on the primary DiD trend, one on the DiD_logp trend) and
    for having enforced sec 9.6 admissibility upstream (this function does not re-derive
    VOID from T1c/T2a/sec9.6 -- those live in check_rung_admissible / T2a).

    AUDIT FIX (2026-07-12, S2 follow-up audit): `factor1_trend_s2` is a REQUIRED,
    KEYWORD-ONLY argument with NO default, and the disagreement rule is enforced HERE.
    sec 9.1.5 pins that rule as something that "must be structurally true, not just a
    convention someone remembers to apply" -- but as first built, the rule lived ONLY inside
    the `compute_verdict_with_s2` wrapper while `compute_verdict` (the obvious name, and the
    one sec 9.5's own prose describes) remained callable with no S2 disposition at all. A
    driver author calling it directly would have received a full COUPLED / DECOUPLED /
    FLAT-COUPLED verdict with S2 silently skipped -- precisely the forgotten-convention
    failure the pin forbids. There is now NO path through this module that yields a verdict
    without both classifications: a stale 4-positional-argument caller raises TypeError
    (loud), it does not quietly return a verdict."""
    if factor1_trend != factor1_trend_s2:
        # sec 9.1.5's PRE-COMMITTED DISAGREEMENT RULE. Checked before FLOOR so that the
        # behaviour is identical to the wrapper's original ordering; either way the result is
        # a NON-verdict, so the ordering cannot create or strengthen one.
        return "INDETERMINATE"
    if n_admissible_rungs < 3 or t1a_positive_count < 3:
        return "FLOOR"
    if factor1_trend == "INDETERMINATE":
        return "INDETERMINATE"
    coupling_licensed = span_frac_monotone_over_A
    if not coupling_licensed:
        return "RECALL-TREND-ONLY"
    return {"DECLINES": "COUPLED", "RISES": "DECOUPLED", "FLAT": "FLAT-COUPLED"}[factor1_trend]


def compute_verdict_with_s2(factor1_trend_primary: str, factor1_trend_s2: str,
                             span_frac_monotone_over_A: bool, n_admissible_rungs: int,
                             t1a_positive_count: int) -> str:
    """sec 9.1.5's PRE-COMMITTED DISAGREEMENT RULE, made structural (not a
    convention a caller has to remember): 'If S2's Factor-1 classification
    differs from the primary's, the verdict is INDETERMINATE and we say so.'
    S2 is a readout-robustness check on the SAME estimand (sec 9.1.5: 'so a
    disagreement is an instrument defect, not a finding') -- this is
    verdict-WITHHOLDING only, exactly like S1 (below): it can only downgrade
    to INDETERMINATE, never create or strengthen a verdict. Both
    `factor1_trend_primary` and `factor1_trend_s2` must already have been
    produced by `classify_recall_trend` (one on the primary DiD trend, one
    on the DiD_logp trend) -- this function does not recompute them.

    AUDIT FIX (2026-07-12, S2 follow-up audit): the disagreement rule itself now lives in
    `compute_verdict` (which REQUIRES `factor1_trend_s2=`), so it can no longer be bypassed
    by calling the inner function directly. This wrapper is retained as sec 9.1.5's named
    entry point and is a pure delegation -- the behaviour of every one of the 16
    primary x S2 combinations is unchanged."""
    return compute_verdict(factor1_trend_primary, span_frac_monotone_over_A,
                            n_admissible_rungs, t1a_positive_count,
                            factor1_trend_s2=factor1_trend_s2)


def compute_s1_utilization(cell: dict) -> dict:
    """sec 9.1.5 S1: the utilization ratio DiD(r)/acc_copy(r) (the rejected
    normalization candidate C5, promoted to a MANDATORY, NON-VERDICT-CARRYING
    sensitivity). Requires `cell['acc_copy']` (wire via `finalize_cell`'s
    `t2b_result=` param).

    AUDIT NOTE (2026-07-12, S2 follow-up audit; SUPERSEDED IN PART BY REV 3, see below): sec
    9.1.5's prose says S1 is "bounded to [0,1] by the already-pinned T2b-2 ceiling." IT IS NOT,
    quite -- and this docstring said so too before the correction. T2b-2 admitted
    `DiD <= acc_copy + 2*SE`, so the ratio's true ceiling was `1 + 2*SE/acc_copy`, which
    exceeds 1 whenever the copy ability is small. Report S1 as the raw ratio it is; do NOT
    clip it to [0,1] and do not read a >1 value as an instrument failure. This has ZERO
    verdict consequence (S1 is structurally verdict-withholding -- it is never fed to
    compute_verdict / compute_capacity_metric), which is the only reason it is a note and not
    a defect.

    REV 3 (sec 11.2's own text, sec 11.6, 2026-07-12): T2b-2 is RETIRED. sec 11.2's own words:
    "sec 9.1.5 asserts S1 is bounded to [0,1] by the already-pinned T2b-2 ceiling -- that bound
    is WITHDRAWN with T2b-2. S1 remains mandatory, reported, and non-verdict-carrying, now as
    an UNBOUNDED ratio with its CI. No verdict ever depended on the bound." At acc_copy == 0
    the ratio is undefined and is reported as None with a reason rather than raising
    ZeroDivisionError or silently reading +inf -- this is now a REAL code path this
    instrument's own gates can reach (T2b-2 no longer forecloses it; a rung can have
    acc_copy==0 and still be admissible if it clears T1a/T2b-1/T2b-1b via a DiD that itself
    reads <=0, or the rung is simply FLOOR and S1 is reported alongside it as a diagnostic
    regardless). THIS FUNCTION MUST NEVER BE FED INTO compute_verdict / compute_capacity_metric
    -- sec 9.1.5: 'S1 cannot change the verdict,' it is reported alongside the primary per the
    pre-registered reading table in sec 9.1.5, never substituted for it."""
    did = cell.get("did")
    acc_copy = cell.get("acc_copy")
    if did is None or acc_copy is None:
        return {"value": None, "did": did, "acc_copy": acc_copy,
                "reason": "cell carries no acc_copy -- finalize_cell was not given a "
                          "t2b_result for this (rung, corpus)"}
    if acc_copy == 0:
        return {"value": None, "did": did, "acc_copy": acc_copy,
                "reason": "acc_copy == 0 -- utilization ratio undefined (division by zero; "
                          "REV 3/sec 11.2: no longer foreclosed by T2b-2, which is retired)"}
    return {"value": did / acc_copy, "did": did, "acc_copy": acc_copy, "reason": None}


def s3_claim_language(s3_ci_by_rung: dict) -> str:
    """sec 11.6.2's PRE-REGISTERED CLAIM-LANGUAGE TABLE, made mechanical:
      | S3 = E[C-D] over admissible rungs | claim language |
      | CI excludes 0 (ANY admissible rung)         | 'in-context associative recall' -- licensed |
      | CI includes 0 at EVERY admissible rung       | 'antecedent-attributable in-context
                                                         dependence' -- the word recall is NOT used |
    `s3_ci_by_rung`: {rung_name: [lo, hi]} -- the sec 11.6.2 S3 CI for every ADMISSIBLE rung
    (caller's responsibility to have already filtered to admissible rungs, sec 9.6). This
    function does not compute admissibility itself. Returns the licensed claim-language
    STRING verbatim -- never a boolean a caller might silently reinterpret."""
    if not s3_ci_by_rung:
        return "NO ADMISSIBLE RUNGS -- no claim language is licensed (nothing to report)"
    any_excludes_zero = any(
        lo is not None and hi is not None and not (lo <= 0 <= hi)
        for lo, hi in s3_ci_by_rung.values()
    )
    if any_excludes_zero:
        return "in-context associative recall"
    return "antecedent-attributable in-context dependence"


# =============================================================================
# SMOKE GATE -- CPU-stub-friendly logic validation, PLUS two hand-constructed
# demonstrations the design's build brief explicitly asks for: the synthetic
# no-recall-but-context-fragile model (placebo teeth) and the T2b-2 ceiling
# check firing on a violating case (VOID teeth). Also a real-checkpoint-shaped
# (but toy-vocab, untrained) end-to-end pass through DeltaNetLM, matching this
# repo's own smoke-gate convention.
# =============================================================================
class _SyntheticContextFragileModel(torch.nn.Module):
    """A hand-built next-token 'model' with ZERO in-context recall of any
    SPECIFIC antecedent, but WITH generic sensitivity to ANY corrupted
    upstream token -- exactly the confound the placebo arm exists to remove
    (design sec 9.2). Given a fixed REFERENCE window, it predicts the
    reference's own next token at position t IFF the input's ENTIRE prefix
    through t matches the reference exactly; otherwise it predicts a fixed
    `wrong_token`. It never inspects WHICH position changed -- antecedent-
    blind by construction. A correctly-identifying instrument must therefore
    read DiD ~= 0 (in fact EXACTLY 0, since this model is deterministic and
    both the TRUE and PLACEBO corruption positions are always < k, i.e.
    always in the queried prefix) even though the RAW gap (intact accuracy
    vs true-ablated accuracy) is LARGE."""

    def __init__(self, reference_window: torch.Tensor, vocab_size: int, wrong_token: int):
        """reference_window: the FULL (T+1,)-length window (x followed by the
        final y token) -- NOT just the x-half. Storing the full window is
        what lets forward() read the TRUE next-token at every one of the T
        input positions, including the last; slicing to just the x-half and
        reconstructing "next token" via a wraparound roll() was an earlier
        draft's off-by-one bug (wrapped to position 0 at the final step)."""
        super().__init__()
        self.register_buffer("reference", reference_window.clone())
        self.vocab_size = vocab_size
        self.wrong_token = wrong_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        ref_x = self.reference[:T].to(x.device).unsqueeze(0).expand(B, -1)
        ref_next = self.reference[1:T + 1].to(x.device)   # TRUE next-token per position, no wraparound
        matches = (x == ref_x)
        clean_prefix = torch.cumprod(matches.long(), dim=1).bool()  # True at t iff x[0..t]==ref[0..t]
        logits = torch.full((B, T, self.vocab_size), -10.0, device=x.device)
        logits[:, :, self.wrong_token] = 0.0
        b_idx, t_idx = clean_prefix.nonzero(as_tuple=True)
        if b_idx.numel():
            correct_tok = ref_next[t_idx]
            logits[b_idx, t_idx, :] = -10.0
            logits[b_idx, t_idx, correct_tok] = 0.0
        return logits


def _build_synthetic_ref_window(V: int, T: int, wrong_token: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randint(0, V - 1, (T + 1,), generator=g)  # never touches wrong_token = V-1
    # plant TWO genuine repeats at well-separated distances so the placebo has
    # somewhere admissible to land and the pooled Delta distribution has >1 value
    ref[10], ref[11] = 5, 6
    ref[150], ref[151] = 5, 6      # candidate k=150, j=10, delta=140
    ref[30], ref[31] = 7, 8
    ref[300], ref[301] = 7, 8      # candidate k=300, j=30, delta=270
    return ref


def build_synthetic_t2_train_corpus(N: int, vocab_size: int = 20_220, n_keys: int = 220,
                                     n_values_pool: int = 110, licenses_per_key: int = 10,
                                     p_key_per_key: float = 3.67e-5, p_boost: float = 0.10,
                                     seed: int = 0) -> torch.Tensor:
    """sec 11 SMOKE-GATE ONLY: a vectorized synthetic corpus generator whose
    statistical shape is CALIBRATED (empirically, see the build report) to
    reliably clear `build_key_value_pools`'s three real floors
    (|P_key|>=100, |P_val(a)|>=5, >=100 b's with >=2 licensing keys) at
    N~15-17M tokens -- exercising sec 11.2's ACTUAL filtering logic (K1-K5/
    V1-V5) on data this file controls, end-to-end through a REAL DeltaNetLM
    forward pass. This is NOT a substitute for validating the picker against
    the REAL openr1/wikitext corpora (done separately, see the build
    report) -- it is the smoke gate's own plumbing check.

    Construction: `n_keys` reserved token ids [0, n_keys) appear ONLY via a
    per-key injection (never via generic background) at rate
    `p_key_per_key` each -- landing their global count in K2's rare-in-
    window/K3's well-trained band by design. `n_values_pool` token ids
    [n_keys, n_keys+n_values_pool) are PART of the generic background pool
    (so they accumulate a healthy baseline global count, satisfying V2) AND
    receive an EXTRA per-key-conditional boost: whenever the PREVIOUS token
    was a reserved key `a`, with probability `p_boost` the CURRENT position
    is overridden to one of `a`'s `licenses_per_key` licensed values
    (uniformly, drawn once per key from the value pool -- WITH overlap
    across keys, which is what makes the inverse map's >=2-licensing-key
    floor reachable). All other positions draw uniformly from the remaining
    background vocabulary."""
    g = torch.Generator().manual_seed(seed)
    p_key = p_key_per_key * n_keys
    background_lo, background_hi = n_keys, vocab_size
    is_key = torch.rand(N, generator=g) < p_key
    background = torch.randint(background_lo, background_hi, (N,), generator=g)
    key_choice = torch.randint(0, n_keys, (N,), generator=g)
    tokens = torch.where(is_key, key_choice, background)

    licensed_table = torch.zeros(n_keys, licenses_per_key, dtype=torch.int64)
    for k in range(n_keys):
        perm = torch.randperm(n_values_pool, generator=g)[:licenses_per_key] + n_keys
        licensed_table[k] = perm

    prev_is_key = torch.zeros(N, dtype=torch.bool)
    prev_is_key[1:] = tokens[:-1] < n_keys
    do_boost = (torch.rand(N, generator=g) < p_boost) & prev_is_key & (~is_key)
    prev_key_id = torch.zeros(N, dtype=torch.int64)
    prev_key_id[1:] = tokens[:-1].clamp(max=n_keys - 1)
    lic_choice_idx = torch.randint(0, licenses_per_key, (N,), generator=g)
    lic_value = licensed_table[prev_key_id.clamp(min=0, max=n_keys - 1), lic_choice_idx]
    tokens = torch.where(do_boost, lic_value, tokens)
    return tokens


def smoke(device: str) -> int:
    print("=" * 70 + "\n  LM_RECALL_GAP_PROBE_V2_RD SMOKE GATE (sec 9 rebuild)\n" + "=" * 70)
    ok_all = True

    def report(name, ok, detail=""):
        nonlocal ok_all
        ok_all = ok_all and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")

    # --- [0] FIX-A resolver, reused verbatim -- sanity re-check ---
    with tempfile.TemporaryDirectory() as td:
        for s in (1000, 2000, 3000, 5000):
            open(os.path.join(td, f"x_step{s}.pt"), "w").close()
        r = resolve_token_matched_checkpoint(td, batch_size=32, seq_len=512,
                                              target_tokens=2000 * 32 * 512 + 100, prefix="x")
        report("FIX-A resolves nearest step (2000)", r["chosen_step"] == 2000, str(r))

    # --- [1] THE FATAL-1 UNIT TEST: row-replication carries EXACTLY ONE
    #     corrupted token per output row, assert on the tensor diff. ---
    base = torch.arange(20, dtype=torch.int64)
    corrupt_positions = [3, 3, 7, 15]
    repl_tokens = [99, 98, 97, 96]
    rep = build_replicated_ablation_batch(base, corrupt_positions, repl_tokens)
    diffs = (rep != base.unsqueeze(0)).sum(dim=1)
    report("row-replication: EVERY row differs from base at EXACTLY one position",
           bool((diffs == 1).all()), f"per-row diff counts={diffs.tolist()} (expected all 1)")
    correct_positions = all(rep[m].ne(base).nonzero().item() == corrupt_positions[m] for m in range(4))
    report("row-replication: the ONE diff is at the INTENDED position for every row",
           correct_positions)
    report("row-replication: two candidates sharing a corrupt POSITION (3) get "
           "INDEPENDENT rows (not merged/overwritten)",
           rep[0, 3].item() == 99 and rep[1, 3].item() == 98
           and torch.equal(rep[0][torch.arange(20) != 3], base[torch.arange(20) != 3])
           and torch.equal(rep[1][torch.arange(20) != 3], base[torch.arange(20) != 3]))

    # --- [1b] AUDIT FIX (2026-07-12, SERIOUS): FORCED-FAIL negative test for
    #     run_ablation_arm's OWN FATAL-1 invariant assertion (`(row!=original).sum()==1`) --
    #     the PRODUCTION check, not `build_replicated_ablation_batch`'s (which the real path
    #     never calls, and which the [1] tests above already exercise -- an earlier audit
    #     note in this file flags THAT as 'an F-3-class toothless test'). This test calls the
    #     PUBLIC run_ablation_arm API with a legitimate (no monkeypatching) construction that
    #     forces a ZERO-corruption row: a `repl_key` value EQUAL to the token already at
    #     `pos_key`, so `rep[pos]=repl` is a no-op and `n_corrupted=0` for that row, which
    #     VIOLATES `==1` exactly as a genuine construction bug would. The assertion fires
    #     BEFORE any model forward pass (see its placement in the function body), so
    #     `model=None` never gets called -- a real bug would not reach the model either. ---
    print("\n  [1b] run_ablation_arm's OWN FATAL-1 assertion -- FORCED-FAIL (RUN TO COMPLETION)")
    fatal1_window = torch.arange(20, dtype=torch.int64)
    fatal1_all_windows = [fatal1_window]
    # spec 0: repl == CURRENT token at pos 5 (fatal1_window[5]==5) -> ZERO-corruption row.
    fatal1_specs = [{"row_idx": 0, "k": 10, "posA": 5, "replA": 5, "target": 999}]
    raised_fatal1, fatal1_msg = False, None
    try:
        run_ablation_arm(None, fatal1_all_windows, fatal1_specs, "posA", "replA", "predA", 8, "cpu")
    except AssertionError as e:
        raised_fatal1, fatal1_msg = True, str(e)
    report("  FORCED-FAIL: a replacement token EQUAL to the current token (zero corruption, "
           "n_corrupted=0 != 1) -> run_ablation_arm's FATAL-1 assertion FIRES (never reaches "
           "model(), never a silent pass)", raised_fatal1 and "FATAL-1" in (fatal1_msg or ""),
           fatal1_msg or "DID NOT RAISE -- FAIL")
    print(f"      transcript: {fatal1_msg}")
    # CONTROL: a genuinely single-token corruption on the SAME window must NOT trip the
    # assertion. `model=None` means the (assertion-passing) control call proceeds to
    # `model(chunk)` and raises TypeError -- that TypeError (not AssertionError) is the
    # EXPECTED outcome here and confirms the assertion did not block this call.
    fatal1_specs_ok = [{"row_idx": 0, "k": 10, "posA": 5, "replA": 77, "target": 999}]
    fatal1_ctrl_outcome = None
    try:
        run_ablation_arm(None, fatal1_all_windows, fatal1_specs_ok, "posA", "replA", "predA", 8, "cpu")
        fatal1_ctrl_outcome = "no_exception"
    except AssertionError:
        fatal1_ctrl_outcome = "assertion_error"
    except TypeError:
        fatal1_ctrl_outcome = "type_error_from_model_none"
    report("  CONTROL: a genuine single-token corruption does NOT trip the FATAL-1 assertion "
           "(reaches model(chunk) and fails there instead, on model=None -- confirms the "
           "assertion passed, was not the blocker)",
           fatal1_ctrl_outcome == "type_error_from_model_none", fatal1_ctrl_outcome)

    # --- [2] Placebo admissibility: forbidden positions are genuinely excluded ---
    x_cpu_toy = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2] + [9] * 8]
    row_candidates_toy = [[(11, 1), (6, 0)]]  # two candidates in the SAME row, antecedents at j+1=2 and j+1=1
    specs_toy = true_arm_specs(row_candidates_toy)
    specs_toy = assign_placebo_positions(specs_toy, row_candidates_toy, x_cpu_toy, seed=1)
    forbidden_check = all(
        s["p_placebo"] not in {s["p"], s["j"], 2, 1} and s["p_placebo"] < s["k"]
        for s in specs_toy if s["p_placebo"] is not None
    )
    report("placebo assignment: never lands on p, j, k-or-later, or ANOTHER candidate's antecedent",
           forbidden_check, str(specs_toy))

    # --- [2b] sec 11.2.3 THE HARD ASSERTION -- FORCED-FAIL NEGATIVE TEST, RUN TO COMPLETION.
    #     "A structural check without a forced-fail negative test that runs to completion is
    #     not a check -- CLAUDE.md, learned the hard way." Deliberately construct CONTESTED
    #     plants (the exact F-I/F-II failure shapes sec 11.2.3 exists to make impossible) and
    #     confirm plant_and_verify_t2_window RAISES PlantContestedError (a RuntimeError
    #     subclass), never a warning, never a silent pass-through. ---
    print("\n  [2b] sec 11.2.3 HARD ASSERTION -- FORCED-FAIL negative test (RUN TO COMPLETION)")
    # Case A: the KEY `a` is deliberately planted where it ALREADY occurs naturally elsewhere
    # in the window -- the structural shape of F-I (sec 11.1: the key competes with natural
    # same-key instances). Planting a=77 at j0=10,k0=15 while position 5 ALREADY holds 77
    # makes count(a in w)==3, not 2, at positions {5,10,15} != {10,15}.
    contested_window_a = [0] * 20
    contested_window_a[5] = 77
    raised_forced_a, forced_a_msg = False, None
    try:
        plant_and_verify_t2_window(contested_window_a, j0=10, k0=15, a=77, b=88)
    except PlantContestedError as e:
        raised_forced_a, forced_a_msg = True, str(e)
    report("  FORCED-FAIL A: contested KEY (a natural occurrence collides with the plant) "
           "-> PlantContestedError FIRES", raised_forced_a, forced_a_msg or "DID NOT RAISE -- FAIL")
    print(f"      transcript: {forced_a_msg}")

    # Case B: the VALUE `b` is deliberately planted where it ALREADY occurs naturally
    # elsewhere in the window -- the structural shape of F-II/A-S2 (a value that isn't
    # actually rare-in-window, so the true-ablation arm would not remove b from context).
    # Planting b=99 at j0_b+1=11 while position 14 ALREADY holds 99 makes count(b in w)==2,
    # not 1, at positions {11,14} != {11}.
    contested_window_b = [0] * 20
    contested_window_b[14] = 99
    raised_forced_b, forced_b_msg = False, None
    try:
        plant_and_verify_t2_window(contested_window_b, j0=10, k0=16, a=55, b=99)
    except PlantContestedError as e:
        raised_forced_b, forced_b_msg = True, str(e)
    report("  FORCED-FAIL B: contested VALUE (a natural occurrence collides with the plant) "
           "-> PlantContestedError FIRES", raised_forced_b, forced_b_msg or "DID NOT RAISE -- FAIL")
    print(f"      transcript: {forced_b_msg}")

    # Case C: PlantContestedError IS a RuntimeError (never merely a warning / print).
    report("  PlantContestedError IS-A RuntimeError (never a warning; a caller who forgets to "
           "catch it crashes loudly, per CLAUDE.md 'hard RuntimeError on violation, never a "
           "warning')", issubclass(PlantContestedError, RuntimeError))

    # CONTROL: an UNCONTESTED plant on the SAME shape of window must NOT raise, and must
    # return the correctly-planted window (proves the assertion has no false positives).
    clean_window = [0] * 20
    raised_control = False
    planted_control = None
    try:
        planted_control = plant_and_verify_t2_window(clean_window, j0=10, k0=15, a=77, b=88)
    except PlantContestedError:
        raised_control = True
    report("  CONTROL: an uncontested plant on a clean window does NOT raise, and plants "
           "correctly (w[10]=77, w[11]=88, w[15]=77)",
           not raised_control and planted_control is not None
           and planted_control[10] == 77 and planted_control[11] == 88 and planted_control[15] == 77)

    # --- [3] THE SYNTHETIC NO-RECALL DEMONSTRATION (placebo teeth). ---
    print("\n  [3] SYNTHETIC NO-RECALL DEMO: a context-FRAGILE-but-recall-BLIND model")
    print("      must yield DiD ~= 0 despite a LARGE raw gap_true.")
    V, T, wrong_token = 64, 400, 63
    ref = _build_synthetic_ref_window(V, T, wrong_token, seed=0)
    model_syn = _SyntheticContextFragileModel(ref, V, wrong_token).to(device)
    model_syn.eval()
    mode_next_syn = [-1] * V  # every continuation counts as non-modal -- pure plumbing test
    all_windows_syn = [ref.to(device)] * 3   # replicate the SAME reference across "rows" (independent
                                              # per-row corruption draws still apply -- exercises the
                                              # multi-row path, not just n=1)
    x_syn = torch.stack([w[:-1] for w in all_windows_syn])
    y_syn = torch.stack([w[1:] for w in all_windows_syn])
    with torch.no_grad():
        pred_intact_syn = model_syn(x_syn).argmax(dim=-1)
    acc_intact_syn = (pred_intact_syn == y_syn).float().mean().item()
    report("  synthetic model reads its OWN reference perfectly (acc_intact==1.0)",
           abs(acc_intact_syn - 1.0) < 1e-9, f"acc_intact={acc_intact_syn}")

    x_cpu_syn = x_syn.cpu().tolist()
    y_cpu_syn = y_syn.cpu().tolist()
    row_cands_syn, _ = detect_candidates_and_baseline(x_cpu_syn, y_cpu_syn, mode_next_syn, min_sep=2)
    sel_syn = select_candidates_per_row(row_cands_syn, c_max=C_MAX_DEFAULT, seed=42)
    n_found = sum(len(c) for c in sel_syn)
    # >=6: the 2 hand-planted repeats x 3 rows are guaranteed present; a V=64/T=400 random
    # sequence also contains INCIDENTAL bigram repeats by chance (birthday-paradox density) --
    # those are legitimate candidates too and do not undermine the demo (the synthetic model's
    # antecedent-blind rule applies uniformly to every candidate, planted or incidental).
    report("  hand-planted candidates detected (>=6: 2/row x 3 rows, plus any incidental repeats)",
           n_found >= 6, f"found={n_found}, per-row={[len(c) for c in sel_syn]}")

    specs_syn = true_arm_specs(sel_syn)
    specs_syn = assign_placebo_positions(specs_syn, sel_syn, x_cpu_syn, seed=42)
    ablate_rng_syn = random.Random(_combine_seed(42, "syn_ablate"))
    window_cpu_syn = [w.cpu().tolist() for w in all_windows_syn]
    for s in specs_syn:
        target = window_cpu_syn[s["row_idx"]][s["k"] + 1]
        s["target"] = target
        s["repl_true"] = draw_exclusive_replacement(window_cpu_syn[s["row_idx"]], s["p"], target, V, ablate_rng_syn)
    for s in specs_syn:
        if s["p_placebo"] is not None:
            s["repl_placebo"] = draw_exclusive_replacement(window_cpu_syn[s["row_idx"]], s["p_placebo"],
                                                             s["target"], V, ablate_rng_syn)
        else:
            s["repl_placebo"] = None
    run_ablation_arm(model_syn, all_windows_syn, specs_syn, "p", "repl_true", "pred_true", 64, device)
    run_ablation_arm(model_syn, all_windows_syn, specs_syn, "p_placebo", "repl_placebo", "pred_placebo", 64, device)

    records_syn = []
    for s in specs_syn:
        target = s["target"]
        hit_intact = int(pred_intact_syn[s["row_idx"], s["k"]].item() == target)
        hit_true = int(s["pred_true"] == target)
        hit_placebo = int(s["pred_placebo"] == target) if s.get("pred_placebo") is not None else None
        records_syn.append({"row_idx": s["row_idx"], "k": s["k"], "delta": s["delta"],
                             "hit_intact": hit_intact, "hit_true_ablated": hit_true,
                             "hit_placebo_ablated": hit_placebo, "placebo_flagged": s["placebo_flagged"]})
    resolved_syn = [r for r in records_syn if r["hit_placebo_ablated"] is not None]
    gap_true_syn = stat_gap_true(resolved_syn)
    gap_placebo_syn = stat_gap_placebo(resolved_syn)
    did_syn = stat_did(resolved_syn)
    report("  raw gap_true is LARGE (any single corrupted token breaks the WHOLE suffix "
           "in this model, by construction)", gap_true_syn > 0.9,
           f"gap_true={gap_true_syn:.4f}")
    report("  gap_placebo is ALSO large (generic context damage, not antecedent-specific)",
           gap_placebo_syn > 0.9, f"gap_placebo={gap_placebo_syn:.4f}")
    report("  DiD ~= 0 EXACTLY (deterministic model: both arms corrupt a position < k, so "
           "both always break the prefix identically) -- THE PLACEBO HAS TEETH",
           abs(did_syn) < 1e-9, f"DiD={did_syn:.6f} (gap_true={gap_true_syn:.4f}, "
           f"gap_placebo={gap_placebo_syn:.4f} -- an un-placebo'd instrument would have "
           f"reported the raw gap_true as 'recall capacity' and been WRONG)")

    # --- [4] finalize_cell REFUSES to emit a verdict-grade number w/o placebo ---
    print("\n  [4] VERDICT-GRADE REFUSAL: finalize_cell must raise if NO candidate has a "
          "resolved placebo read.")
    fake_no_placebo = {"records": [{"row_idx": 0, "k": 5, "delta": 3, "hit_intact": 1,
                                     "hit_true_ablated": 0, "hit_placebo_ablated": None,
                                     "placebo_flagged": True}],
                        "acc_baseline_nonAR": 0.5, "n_baseline": 10, "n_windows": 1}
    raised = False
    try:
        finalize_cell("fake_rung", "fake_corpus", fake_no_placebo)
    except VerdictGradeError:
        raised = True
    report("  finalize_cell raises VerdictGradeError with zero resolved placebos", raised)

    fake_ok = {"records": [{"row_idx": 0, "k": 5, "delta": 3, "hit_intact": 1,
                             "hit_true_ablated": 0, "hit_placebo_ablated": 1,
                             "placebo_flagged": False,
                             # sec 9.1.5 S2 fields -- present, so this cell is S2-complete.
                             "logp_intact": -0.1, "logp_true_ablated": -5.0,
                             "logp_placebo_ablated": -0.3}] * 3,
               "acc_baseline_nonAR": 0.5, "n_baseline": 10, "n_windows": 1}
    cell_ok = finalize_cell("fake_rung", "fake_corpus", fake_ok, n_boot=50,
                             t2b_result={"acc_copy": 0.4, "acc_copy_se": 0.05})
    report("  finalize_cell stamps _verdict_grade=True on a valid cell", cell_ok.get("_verdict_grade") is True)
    report("  finalize_cell's cell is S2-complete when every record carries logp fields",
           cell_ok.get("cell_void_missing_s2_fields") is False)
    report("  finalize_cell merges acc_copy/acc_copy_se from an optional t2b_result",
           cell_ok.get("acc_copy") == 0.4 and cell_ok.get("acc_copy_se") == 0.05)

    # --- sec 9.1.5 S2 teeth: a cell MISSING the log-prob fields on even one resolved
    #     candidate must be refused by compute_capacity_metric (sec 9.6's stop rule: "a
    #     cell missing these fields is VOID"). ---
    fake_missing_s2 = {"records": [{"row_idx": 0, "k": 5, "delta": 3, "hit_intact": 1,
                                     "hit_true_ablated": 0, "hit_placebo_ablated": 1,
                                     "placebo_flagged": False,
                                     "logp_intact": None, "logp_true_ablated": None,
                                     "logp_placebo_ablated": None}] * 3,
                        "acc_baseline_nonAR": 0.5, "n_baseline": 10, "n_windows": 1}
    cell_missing_s2 = finalize_cell("fake_rung", "fake_corpus", fake_missing_s2, n_boot=50)
    report("  finalize_cell correctly flags a cell missing S2 fields (cell_void_missing_s2_fields=True)",
           cell_missing_s2.get("cell_void_missing_s2_fields") is True
           and cell_missing_s2.get("did_logp") is None)
    raised_s2 = False
    try:
        compute_capacity_metric(cell_missing_s2, "raw_did")
    except VerdictGradeError as e:
        raised_s2 = "S2" in str(e) or "log-prob" in str(e)
    report("  compute_capacity_metric REFUSES a cell missing S2 fields (sec 9.6 stop rule)",
           raised_s2)

    unstamped = dict(cell_ok)
    del unstamped["_verdict_grade"]
    raised2 = False
    try:
        compute_capacity_metric(unstamped, "anything")
    except VerdictGradeError:
        raised2 = True
    report("  compute_capacity_metric refuses an UNSTAMPED dict (bypass attempt caught)", raised2)

    # REV 3 TEETH (sec 11.6, 2026-07-12): T2b-2 is RETIRED and DELETED. compute_capacity_metric
    # must NO LONGER accept (or need) a `t2b2_pass=` keyword at all -- calling it with the OLD
    # signature must fail LOUDLY (TypeError), not silently ignore the argument, and a cell that
    # would have failed the old T2b-2 ceiling (DiD > acc_copy) must STILL compute M(r) cleanly
    # (no replacement ceiling was invented, sec 11.6: "No ceiling check replaces it").
    stale_t2b2_kwarg_raised = False
    try:
        compute_capacity_metric(cell_ok, "raw_did", t2b2_pass=True)   # the RETIRED signature
    except TypeError:
        stale_t2b2_kwarg_raised = True
    report("  compute_capacity_metric REFUSES the RETIRED t2b2_pass= keyword (TypeError) -- "
           "T2b-2 cannot be silently re-wired back into the verdict path",
           stale_t2b2_kwarg_raised)
    high_did_cell = dict(cell_ok, did=0.99, acc_copy=0.01)   # would have FAILED the old T2b-2 ceiling
    m_no_ceiling = compute_capacity_metric(high_did_cell, "raw_did")
    report("  a cell with DiD >> acc_copy (would have VOIDED under the retired T2b-2) now "
           "computes M(r) cleanly -- NO replacement ceiling was invented (sec 11.6)",
           m_no_ceiling == 0.99, f"M(r)={m_no_ceiling}")
    report("  check_t2b2_ceiling / pick_t2_marker_tokens / run_t2_planted_copy are GONE from "
           "the module namespace (not merely unwired)",
           "check_t2b2_ceiling" not in globals() and "pick_t2_marker_tokens" not in globals()
           and "run_t2_planted_copy" not in globals())

    # ... and a cell whose placebo never distribution-matched (sec 9.2: "the cell is VOID") --
    # THE ONLY VOID TRIGGER LEFT in compute_capacity_metric after T2b-2's retirement.
    void_placebo_cell = dict(cell_ok, cell_void_placebo_match=True)
    raised_pv = False
    try:
        compute_capacity_metric(void_placebo_cell, "anything")
    except VerdictGradeError as e:
        raised_pv = "placebo not distribution-matched" in str(e)
    report("  compute_capacity_metric REFUSES a placebo-unmatched (=VOID) cell", raised_pv)

    # --- [5] NORMALIZATION: sec 9.1 is now PINNED (raw_did) -- fails loudly when unset /
    #     unregistered for anything else; the mechanism stays generic and pluggable. ---
    raised3 = False
    try:
        compute_capacity_metric(cell_ok, None)
    except RuntimeError as e:
        raised3 = "NORMALIZATION_UNSET" in str(e)
    report("  compute_capacity_metric raises RuntimeError('NORMALIZATION_UNSET') when normalization=None",
           raised3)
    raised4 = False
    try:
        compute_capacity_metric(cell_ok, "not_registered_yet")
    except KeyError:
        raised4 = True
    report("  compute_capacity_metric raises KeyError on an unregistered normalization name", raised4)
    report("  NORMALIZATION_REGISTRY carries EXACTLY the sec 9.1 pin ('raw_did'), no other "
           "convenience default", sorted(NORMALIZATION_REGISTRY) == ["raw_did"],
           f"registered={sorted(NORMALIZATION_REGISTRY)}")
    m_raw = compute_capacity_metric(cell_ok, "raw_did")
    report("  'raw_did' (sec 9.1's pin) computes M(r) == cell['did'] exactly, denominator 1",
           m_raw == cell_ok["did"], f"M(r)={m_raw}, did={cell_ok['did']}")
    register_normalization("_smoke_test_only_identity", lambda cell: cell["did"])
    report("  register_normalization + a registered name WORKS for a hypothetical future metric "
           "(the mechanism stayed generic after the pin landed)",
           compute_capacity_metric(cell_ok, "_smoke_test_only_identity") == cell_ok["did"])
    del NORMALIZATION_REGISTRY["_smoke_test_only_identity"]   # leave the registry at just the pin

    # AUDIT FIX teeth (S2 follow-up audit): the PINNED name may not be SILENTLY rebound.
    raised_ovw = False
    try:
        register_normalization("raw_did", lambda cell: 999.0)
    except RuntimeError:
        raised_ovw = True
    report("  register_normalization REFUSES to silently rebind the PINNED 'raw_did' to a "
           "different function, and M(r) still reads the pin afterwards",
           raised_ovw and compute_capacity_metric(cell_ok, "raw_did") == cell_ok["did"])
    register_normalization("raw_did", _raw_did)   # the SAME function object: idempotent, allowed
    report("  ... but re-registering the IDENTICAL function object is a no-op "
           "(importlib.reload / double-import stay safe)",
           NORMALIZATION_REGISTRY["raw_did"] is _raw_did)

    # --- [5d] sec 9.1.5 S1 -- the utilization ratio DiD/acc_copy, verdict-WITHHOLDING only ---
    print("\n  [5d] sec 9.1.5 S1 (utilization ratio) + S2 disagreement-rule teeth")
    s1 = compute_s1_utilization(cell_ok)
    report("  S1 = did/acc_copy on a normal cell", abs(s1["value"] - cell_ok["did"] / 0.4) < 1e-9, str(s1))
    s1_zero = compute_s1_utilization({"did": 0.1, "acc_copy": 0.0})
    report("  S1 reports None (not ZeroDivisionError/inf) at acc_copy==0",
           s1_zero["value"] is None and s1_zero["reason"] is not None, str(s1_zero))
    s1_missing = compute_s1_utilization({"did": 0.1, "acc_copy": None})
    report("  S1 reports None when acc_copy was never merged into the cell",
           s1_missing["value"] is None, str(s1_missing))

    # --- sec 11.6.2 S3 claim-language table, mechanized ---
    report("  s3_claim_language: ANY admissible rung's CI excluding 0 -> 'in-context "
           "associative recall' licensed",
           s3_claim_language({"14M": [0.01, 0.05], "98M": [-0.01, 0.02]})
           == "in-context associative recall")
    report("  s3_claim_language: EVERY admissible rung's CI includes 0 -> 'antecedent-"
           "attributable in-context dependence', the word recall NOT used",
           s3_claim_language({"14M": [-0.02, 0.03], "98M": [-0.01, 0.02]})
           == "antecedent-attributable in-context dependence")
    report("  s3_claim_language: no admissible rungs -> no claim licensed",
           "no" in s3_claim_language({}).lower())

    # --- S2 pre-committed disagreement rule: primary and S2 factor-1 classifications
    #     disagreeing MUST force INDETERMINATE, even when both individually look decisive. ---
    v_agree = compute_verdict_with_s2("DECLINES", "DECLINES", True, 4, 4)
    report("  S2 agrees with primary (both DECLINES) -> normal table result (COUPLED)",
           v_agree == "COUPLED", v_agree)
    v_disagree = compute_verdict_with_s2("RISES", "DECLINES", True, 4, 4)
    report("  S2 DISAGREES with primary (RISES vs DECLINES) -> forced INDETERMINATE "
           "(never creates/strengthens a verdict, only withholds)",
           v_disagree == "INDETERMINATE", v_disagree)
    v_disagree2 = compute_verdict_with_s2("FLAT", "RISES", True, 4, 4)
    report("  S2 disagreement fires even when BOTH individually look like a clean, "
           "differently-decisive verdict", v_disagree2 == "INDETERMINATE", v_disagree2)

    # --- stat_did_logp / has_complete_s2_fields unit checks ---
    logp_records = [{"logp_placebo_ablated": -0.2, "logp_true_ablated": -3.0},
                     {"logp_placebo_ablated": -0.5, "logp_true_ablated": -4.0}]
    expected_didlogp = ((-0.2 - -3.0) + (-0.5 - -4.0)) / 2
    report("  stat_did_logp computes E[logp_placebo_ablated - logp_true_ablated] correctly",
           abs(stat_did_logp(logp_records) - expected_didlogp) < 1e-9,
           f"got={stat_did_logp(logp_records)}, expected={expected_didlogp}")
    report("  has_complete_s2_fields(True case)",
           has_complete_s2_fields([{"logp_intact": -0.1, "logp_true_ablated": -1.0,
                                     "logp_placebo_ablated": -0.2}]))
    report("  has_complete_s2_fields(False case: one None)",
           not has_complete_s2_fields([{"logp_intact": -0.1, "logp_true_ablated": None,
                                         "logp_placebo_ablated": -0.2}]))

    # --- [5b] AUDIT FIX teeth: the placebo-match VOID must reach the VERDICT as VOID,
    #     not as FLOOR (VOID = HALT the wave; FLOOR = quietly drop the rung and fit
    #     through the rest -- opposite consequences from the same failure). ---
    pv_cells = {c: dict(cell_ok, cell_void_placebo_match=True, n_candidates_resolved=9999)
                for c in ("wikitext-mix-ext", "openr1-mix-ext")}
    adm_pv = check_rung_admissible(pv_cells, tok_per_param=23.0, checkpoint_quiesced=True,
                                    t2b1_by_corpus={c: {"passes": True} for c in pv_cells},
                                    t2b1b_by_corpus={c: {"passes": True} for c in pv_cells},
                                    min_candidates=1)
    report("  check_rung_admissible maps a placebo-match failure to VOID (not FLOOR)",
           adm_pv["verdict_if_not_admissible"] == "VOID" and not adm_pv["admissible"],
           str(adm_pv["verdict_if_not_admissible"]))

    # REV 3 TEETH (sec 11.5, 2026-07-12): a rung that fails T2b-1b (key-conditioned mechanism
    # absent) must be excluded as FLOOR, NEVER VOID (T2b-2's old VOID-on-mechanism trigger is
    # retired, sec 11.6 -- nothing replaces it). And calling with the RETIRED `t2b2_by_corpus=`
    # keyword must fail loudly (TypeError), never be silently accepted/ignored.
    ok_cells = {c: dict(cell_ok, cell_void_placebo_match=False, n_candidates_resolved=9999,
                         t1a_pass_did_ci_excludes_zero=True)
                for c in ("wikitext-mix-ext", "openr1-mix-ext")}
    adm_t2b1b_fail = check_rung_admissible(
        ok_cells, tok_per_param=23.0, checkpoint_quiesced=True,
        t2b1_by_corpus={c: {"passes": True} for c in ok_cells},
        t2b1b_by_corpus={c: {"passes": False} for c in ok_cells}, min_candidates=1)
    report("  check_rung_admissible: a T2b-1b failure -> FLOOR_OR_EXCLUDED, NEVER VOID "
           "(T2b-2's old VOID-on-mechanism trigger is retired, sec 11.6)",
           adm_t2b1b_fail["verdict_if_not_admissible"] == "FLOOR_OR_EXCLUDED"
           and not adm_t2b1b_fail["admissible"], str(adm_t2b1b_fail["verdict_if_not_admissible"]))
    adm_all_pass = check_rung_admissible(
        ok_cells, tok_per_param=23.0, checkpoint_quiesced=True,
        t2b1_by_corpus={c: {"passes": True} for c in ok_cells},
        t2b1b_by_corpus={c: {"passes": True} for c in ok_cells}, min_candidates=1)
    report("  check_rung_admissible: T1a+T2b-1+T2b-1b all pass on both corpora -> ADMISSIBLE",
           adm_all_pass["admissible"] is True, str(adm_all_pass))
    stale_t2b2_kw_raised = False
    try:
        check_rung_admissible(ok_cells, tok_per_param=23.0, checkpoint_quiesced=True,
                               t2b2_by_corpus={c: {"passes": True} for c in ok_cells},
                               t2b1_by_corpus={c: {"passes": True} for c in ok_cells},
                               min_candidates=1)
    except TypeError:
        stale_t2b2_kw_raised = True
    report("  check_rung_admissible REFUSES the RETIRED t2b2_by_corpus= keyword (TypeError)",
           stale_t2b2_kw_raised)

    # --- [5c] AUDIT FIX teeth: tost_flat must return the PINNED 90% CI (5th..95th
    #     percentile), not the 80% CI the previous body returned. ---
    unit_boots = [i / 1000.0 for i in range(1000)]
    tf = tost_flat(unit_boots, delta=1.0, alpha=0.10)
    report("  tost_flat returns the pinned 90% CI (5th..95th pct), not an 80% CI",
           abs(tf["ci90"][0] - 0.05) < 2e-3 and abs(tf["ci90"][1] - 0.949) < 2e-3,
           f"ci90={tf['ci90']} (an 80% CI would read [0.100, 0.899] and would declare "
           f"FLAT more often than the pre-registration permits)")

    # --- [6] T2b-2 IS RETIRED (sec 11.6) -- confirm it is GONE from the module, not merely
    #     unwired, and confirm T2b-1b (its functional replacement guard, sec 11.6.1) fires
    #     correctly: the IDENTICAL paired-sign-test logic as T2b-1, on arm4(key-swap) vs
    #     arm3b(pool-placebo) instead of arm2(true-ablated) vs arm3(placebo-ablated). ---
    print("\n  [6] T2b-2 RETIREMENT + T2b-1b (key-conditioned mechanism) teeth")
    report("  check_t2b2_ceiling does not exist ANYWHERE in this module (sec 11.6: retired, "
           "not merely unwired -- 'no replacement ceiling is invented')",
           "check_t2b2_ceiling" not in dir())
    asym_t2b1b = ([{"hit_pool_placebo": 1, "hit_keyswap": 0} for _ in range(20)]
                  + [{"hit_pool_placebo": 0, "hit_keyswap": 1} for _ in range(1)])
    r_t2b1b_asym = check_t2b1b_key_conditioned(asym_t2b1b)
    report("  check_t2b1b_key_conditioned FIRES (passes=True, p<0.001) on a strongly "
           "asymmetric 20-vs-1 split", r_t2b1b_asym["passes"], str(r_t2b1b_asym))
    sym_t2b1b = ([{"hit_pool_placebo": 1, "hit_keyswap": 0} for _ in range(5)]
                 + [{"hit_pool_placebo": 0, "hit_keyswap": 1} for _ in range(5)])
    r_t2b1b_sym = check_t2b1b_key_conditioned(sym_t2b1b)
    report("  check_t2b1b_key_conditioned does NOT fire (passes=False) on a symmetric 5-vs-5 "
           "split (no key-conditioning signal)", not r_t2b1b_sym["passes"], str(r_t2b1b_sym))
    # THE SHORTCUT-EXCLUSION DEMONSTRATION (sec 11.3's own claim): T2b-1 alone cannot
    # distinguish key-conditioned recall from unconditional in-context-repetition salience --
    # a model that just "emits any token already seen in this window" passes T2b-1 with a
    # perfect split (arm2 destroys b -> wrong; arm3's random token -> b stays present -> right)
    # while having ZERO key-conditioned mechanism, so T2b-1b (arm4 vs arm3b, which BOTH keep b
    # present) reads SYMMETRIC on that same model. Verify T2b-1 and T2b-1b can disagree.
    salience_only_t2b1 = ([{"hit_placebo_ablated": 1, "hit_true_ablated": 0} for _ in range(20)]
                          + [{"hit_placebo_ablated": 0, "hit_true_ablated": 1} for _ in range(1)])
    salience_only_t2b1b = ([{"hit_pool_placebo": 1, "hit_keyswap": 1} for _ in range(10)]
                           + [{"hit_pool_placebo": 0, "hit_keyswap": 0} for _ in range(10)])
    r_salience_t2b1 = check_t2b1_mechanism_exists(salience_only_t2b1)
    r_salience_t2b1b = check_t2b1b_key_conditioned(salience_only_t2b1b)
    report("  THE SHORTCUT T2b-1b CLOSES: a pure in-context-repetition-salience model passes "
           "T2b-1 (b's mere presence matters) but FAILS T2b-1b (identity match doesn't) -- "
           "T2b-1 alone cannot see this, sec 11.3",
           r_salience_t2b1["passes"] and not r_salience_t2b1b["passes"],
           f"t2b1={r_salience_t2b1}, t2b1b={r_salience_t2b1b}")

    # --- [7] T2b-1 exact paired sign test: fires on an asymmetric case, does not
    #     on a symmetric/no-signal case. ---
    print("\n  [7] T2b-1 mechanism-exists paired test")
    asymmetric = [{"hit_placebo_ablated": 1, "hit_true_ablated": 0} for _ in range(20)] + \
                 [{"hit_placebo_ablated": 0, "hit_true_ablated": 1} for _ in range(1)]
    r_asym = check_t2b1_mechanism_exists(asymmetric)
    report("  fires (passes=True, p<0.001) on a strongly asymmetric 20-vs-1 split", r_asym["passes"],
           str(r_asym))
    symmetric = [{"hit_placebo_ablated": 1, "hit_true_ablated": 0} for _ in range(5)] + \
                [{"hit_placebo_ablated": 0, "hit_true_ablated": 1} for _ in range(5)]
    r_sym = check_t2b1_mechanism_exists(symmetric)
    report("  does NOT fire (passes=False) on a perfectly symmetric 5-vs-5 split", not r_sym["passes"],
           str(r_sym))

    # --- [8] QUIESCENCE / md5-pin: catches a checkpoint being actively written,
    #     clears once the writer stops. ---
    print("\n  [8] CHECKPOINT QUIESCENCE (partially-written checkpoint detection)")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake_ckpt.pt")
        with open(path, "wb") as f:
            f.write(b"x" * 1000)
        stop_event = threading.Event()

        def _writer():
            while not stop_event.is_set():
                with open(path, "ab") as f:
                    f.write(b"y")
                time.sleep(0.03)

        th = threading.Thread(target=_writer, daemon=True)
        th.start()
        time.sleep(0.15)
        q_live = verify_quiesced_checkpoint(path, wait_s=0.25)
        stop_event.set()
        th.join(timeout=2)
        time.sleep(0.05)
        q_done = verify_quiesced_checkpoint(path, wait_s=0.25)
    report("  a LIVE writer is caught (quiesced=False) while it is still appending",
           q_live["quiesced"] is False, f"size {q_live['size0']}->{q_live['size1']}")
    report("  the SAME file reads quiesced=True once the writer has stopped",
           q_done["quiesced"] is True and q_done["md5"] is not None)

    raised5 = False
    with tempfile.TemporaryDirectory() as td:
        path2 = os.path.join(td, "growing.pt")
        with open(path2, "wb") as f:
            f.write(b"a" * 100)
        stop2 = threading.Event()

        def _writer2():
            while not stop2.is_set():
                with open(path2, "ab") as f:
                    f.write(b"b")
                time.sleep(0.02)

        th2 = threading.Thread(target=_writer2, daemon=True)
        th2.start()
        time.sleep(0.1)
        try:
            # load_checkpoint will fail at torch.load AFTER the quiescence check passes/fails --
            # here we only need the quiescence gate itself to raise before ever touching torch.load
            q = verify_quiesced_checkpoint(path2, wait_s=0.2)
            if not q["quiesced"]:
                raise CheckpointNotQuiescedError("simulated load_checkpoint refusal")
        except CheckpointNotQuiescedError:
            raised5 = True
        stop2.set()
        th2.join(timeout=2)
    report("  load_checkpoint's refusal path (CheckpointNotQuiescedError) fires on a live writer",
           raised5)

    # --- [9] Verdict-map classification sanity (pure arithmetic, no model needed) ---
    print("\n  [9] Verdict-map (sec 9.5) classification sanity")
    report("  beta CI entirely > 0 -> RISES",
           classify_recall_trend([0.05, 0.20], {"tost_pass": False}) == "RISES")
    report("  beta CI entirely < 0 -> DECLINES",
           classify_recall_trend([-0.20, -0.05], {"tost_pass": False}) == "DECLINES")
    report("  beta CI includes 0 + TOST passes -> FLAT",
           classify_recall_trend([-0.01, 0.01], {"tost_pass": True}) == "FLAT")
    report("  beta CI includes 0 + TOST fails -> INDETERMINATE",
           classify_recall_trend([-0.5, 0.5], {"tost_pass": False}) == "INDETERMINATE")
    report("  compute_verdict: DECLINES + coupling licensed -> COUPLED",
           compute_verdict("DECLINES", True, 4, 4, factor1_trend_s2="DECLINES") == "COUPLED")
    report("  compute_verdict: RISES + coupling licensed -> DECOUPLED",
           compute_verdict("RISES", True, 4, 4, factor1_trend_s2="RISES") == "DECOUPLED")
    report("  compute_verdict: FLAT + coupling licensed -> FLAT-COUPLED",
           compute_verdict("FLAT", True, 4, 4, factor1_trend_s2="FLAT") == "FLAT-COUPLED")
    report("  compute_verdict: trend present but coupling NOT licensed -> RECALL-TREND-ONLY",
           compute_verdict("RISES", False, 4, 4, factor1_trend_s2="RISES") == "RECALL-TREND-ONLY")
    report("  compute_verdict: fewer than 3 admissible rungs -> FLOOR",
           compute_verdict("RISES", True, 2, 2, factor1_trend_s2="RISES") == "FLOOR")

    # --- [9b] AUDIT FIX teeth (S2 follow-up audit): the sec 9.1.5 disagreement rule must be
    #     STRUCTURAL, not a convention. `compute_verdict` -- the function sec 9.5's prose
    #     names, and the one a driver author would reach for first -- must be INCAPABLE of
    #     returning a verdict without an S2 disposition. THE NEGATIVE TEST: the old
    #     4-positional-argument call must now FAIL LOUDLY, not quietly hand back "DECOUPLED".
    stale_caller_raised = False
    try:
        compute_verdict("RISES", True, 4, 4)     # the pre-fix signature: S2 silently skipped
    except TypeError:
        stale_caller_raised = True
    report("  compute_verdict REFUSES a stale S2-less call (TypeError) -- the sec 9.1.5 rule "
           "cannot be bypassed by calling the inner function directly", stale_caller_raised)
    # ... and the full 4x4 grid: NO disagreement may reach a substantive verdict, via EITHER
    #     entry point (the wrapper and the inner function must agree on all 16).
    _trends = ["RISES", "DECLINES", "FLAT", "INDETERMINATE"]
    _substantive = {"COUPLED", "DECOUPLED", "FLAT-COUPLED", "RECALL-TREND-ONLY"}
    _leaks, _mismatch = [], []
    for _p in _trends:
        for _s in _trends:
            for _lic in (True, False):
                _w = compute_verdict_with_s2(_p, _s, _lic, 4, 4)
                _d = compute_verdict(_p, _lic, 4, 4, factor1_trend_s2=_s)
                if _w != _d:
                    _mismatch.append((_p, _s, _lic, _w, _d))
                if _p != _s and _w in _substantive:
                    _leaks.append((_p, _s, _lic, _w))
    report("  FULL 4x4 grid (x2 licensing = 32 calls): no primary!=S2 combination reaches ANY "
           "substantive verdict, and the wrapper and the inner function agree on all 32",
           not _leaks and not _mismatch, f"leaks={_leaks} mismatches={_mismatch}")

    # --- [10] END-TO-END through the REAL DeltaNetLM class (stub or real fla,
    #     tiny toy vocab, untrained) -- exercises the full run_did_eval plumbing
    #     against the actual model API/shapes, matching this repo's own
    #     smoke-gate convention (CPU-stub tests LOGIC only; see [11] for the
    #     narrow real-kernel note). ---
    print("\n  [10] END-TO-END plumbing through DeltaNetLM (toy vocab, untrained)")
    torch.manual_seed(0)
    V10 = 200
    seq_len10 = 160  # >= _MIN_KERNEL_T floor per this repo's own convention
    tiny = DeltaNetLM(V10, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)
    tiny.eval()
    base10 = torch.randint(12, V10, (seq_len10 + 1,))
    base10[5], base10[6] = 10, 11
    base10[140], base10[141] = 10, 11   # genuine repeat, delta=134
    val_tokens10 = base10.repeat(20).to(device)   # a tiny "corpus" get_batch can sample windows from
    mode_next10 = [-1] * V10
    mode_next_t = torch.tensor(mode_next10)
    try:
        result10 = run_did_eval(tiny, val_tokens10, batch_size=4, seq_len=seq_len10, n_windows=4,
                                 device=device, mode_next=mode_next_t, seed=123, c_max=4,
                                 eval_micro_batch=8, min_sep=2, vocab_size=V10)
        report("  run_did_eval completes end-to-end on the real DeltaNetLM class",
               True, f"n_candidates={len(result10['records'])}")
        # sec 9.1.5 S2: every record from a REAL forward pass must carry real (non-None,
        # finite, <=0) log-probs in all three arms -- not just the hand-built smoke fixtures.
        resolved10 = [r for r in result10["records"] if r["hit_placebo_ablated"] is not None]
        s2_ok_real = resolved10 and all(
            r.get("logp_intact") is not None and r["logp_intact"] <= 1e-6
            and r.get("logp_true_ablated") is not None and r["logp_true_ablated"] <= 1e-6
            and r.get("logp_placebo_ablated") is not None and r["logp_placebo_ablated"] <= 1e-6
            for r in resolved10
        )
        report("  S2 log-prob fields populate from the REAL forward pass (log p(target) <= 0 "
               "in all three arms, every resolved candidate)", bool(s2_ok_real) if resolved10 else True,
               f"n_resolved={len(resolved10)}"
               + (f", sample logp_intact={resolved10[0].get('logp_intact')}" if resolved10 else ""))
        # sec 11.6.2 ARM D (audit-fix addition): every record must carry hit_D/logp_D too --
        # unconditionally resolvable (j is never a fallback position), unlike hit_placebo_ablated.
        armd_ok_real = result10["records"] and all(
            r.get("hit_D") is not None and r.get("logp_D") is not None and r["logp_D"] <= 1e-6
            for r in result10["records"]
        )
        report("  sec 11.6.2 ARM D fields (hit_D/logp_D) populate from the REAL forward pass "
               "on EVERY candidate (unconditionally resolvable, unlike the placebo arm)",
               bool(armd_ok_real), f"n_total={len(result10['records'])}")
        if result10["records"]:
            cell10 = finalize_cell("toy_rung", "toy_corpus", result10, n_boot=50)
            report("  finalize_cell produces a verdict-grade cell from a real forward pass",
                   cell10.get("_verdict_grade") is True,
                   f"DiD={cell10['did']:.4f} n_resolved={cell10['n_candidates_resolved']}")
            report("  the real-pass cell is S2-complete and did_logp is a real number",
                   cell10.get("cell_void_missing_s2_fields") is False and cell10.get("did_logp") is not None,
                   f"did_logp={cell10.get('did_logp')}")
            report("  sec 11.6.2 S3 (Arm D) is complete and reports a real s3/s3_ci "
                   "(mandatory, reported always, verdict-carrying never)",
                   cell10.get("cell_arm_d_incomplete") is False and cell10.get("s3") is not None
                   and cell10.get("residual_d_minus_b_UNLABELLED") is not None,
                   f"s3={cell10.get('s3')} s3_ci={cell10.get('s3_ci')} "
                   f"residual_d_minus_b={cell10.get('residual_d_minus_b_UNLABELLED')}")
        else:
            report("  (no candidates found in this tiny random corpus sample -- plumbing still "
                   "ran without error; not a failure of the instrument)", True)
    except Exception as e:  # noqa: BLE001 -- smoke gate: report, don't hide
        report("  run_did_eval end-to-end pass", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [10b] sec 11 T2 REPAIR END-TO-END: build_key_value_pools -> assign_t2_plant (the
    #     hard assertion, exercised on THOUSANDS of real windows, not just the [2b] hand-built
    #     cases) -> run_t2_repaired_probe's SIX ARMS -> T2b-1/T2b-1b/T2a-1/T2a-2, all against
    #     the REAL DeltaNetLM class on a CALIBRATED synthetic corpus (untrained -- plumbing +
    #     the negative-control claim, not a trained-competence claim). Replaces the RETIRED
    #     run_t2_planted_copy / check_t2b2_ceiling end-to-end block entirely. ---
    print("\n  [10b] sec 11 T2 REPAIR END-TO-END: pools -> per-window plant -> six arms -> gates")
    try:
        V11 = 20_220
        t0_corpus = time.time()
        train11 = build_synthetic_t2_train_corpus(15_000_000, vocab_size=V11, seed=11)
        val11 = build_synthetic_t2_train_corpus(2_000_000, vocab_size=V11, seed=12)
        report("  synthetic T2 corpus built (15M train + 2M val tokens, calibrated shape)",
               True, f"{time.time() - t0_corpus:.2f}s")

        t0_pool = time.time()
        pools11 = build_key_value_pools(train11, V11, device)
        report("  build_key_value_pools clears ALL THREE sec 11.2 pre-pass floors on the "
               "synthetic corpus (|P_key|>=100, |P_val(a)|>=5 for every retained key, "
               ">=100 b's with >=2 licensing keys)",
               pools11["floor_pool_key"] and pools11["floor_pool_val"] and pools11["floor_licensed_b"],
               f"n_p_key={pools11['n_p_key']} median_p_val={pools11['median_p_val']} "
               f"n_licensed_b_ge2={pools11['n_licensed_b_ge2']} "
               f"k2_band_used={pools11['k2_band_used']} ({time.time() - t0_pool:.2f}s)")
        counts11 = torch.bincount(train11, minlength=V11).tolist()

        # AUDIT FIX (2026-07-12, SERIOUS) teeth: the sec 11.2 pool floors are GATES, not
        # hopes -- run_t2_repaired_probe must VOID on ANY floor miss BEFORE sampling a plant.
        starved_pools = dict(pools11, floor_pool_key=False)
        starved_result = run_t2_repaired_probe(
            None, val11, seq_len=256, device=device, corpus_name="starved-test",
            delta_pool=[10, 20], pools=starved_pools, counts_by_token=counts11, n_plants=5,
            vocab_size=V11)
        report("  run_t2_repaired_probe VOIDs on a failing sec 11.2 pool floor BEFORE "
               "sampling any plant (model=None never gets called)",
               starved_result.get("void") is True and not starved_result["records"]
               and "sec 11.2" in starved_result.get("void_reason", ""),
               starved_result.get("void_reason"))
        # AUDIT FIX (2026-07-12, SERIOUS) teeth: floor_pool_val must NOT be vacuously True on
        # an EMPTY P_val pool (Python's all() over zero items is True by definition).
        empty_pools_check = dict(P_key=[], P_val={}, inverse_map={}, val_meta={})
        floor_val_on_empty = bool({}) and all(True for _ in {})   # mirrors the FIXED expression
        report("  floor_pool_val is NOT vacuously True on an empty P_val pool "
               "(bool(p_val_final) precondition closes Python's all()-over-empty hole)",
               floor_val_on_empty is False)

        torch.manual_seed(0)
        tiny11 = DeltaNetLM(V11, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)
        tiny11.eval()

        delta_pool11 = [10, 20, 30, 40, 60, 80, 100, 120]
        n_plants11 = 200
        t0_probe = time.time()
        t2_result = run_t2_repaired_probe(tiny11, val11, seq_len=256, device=device,
                                           corpus_name="smoke-synth-corpus", delta_pool=delta_pool11,
                                           pools=pools11, counts_by_token=counts11,
                                           n_plants=n_plants11, vocab_size=V11, eval_micro_batch=32)
        report("  run_t2_repaired_probe completes end-to-end, VOID=False, drop rate within cap",
               (not t2_result.get("void")) and t2_result["n_dropped"] / n_plants11 <= T2_DROP_CAP_FRAC,
               f"n_plants={t2_result.get('n_plants')} n_dropped={t2_result.get('n_dropped')} "
               f"drop_reasons={t2_result.get('drop_reasons')} ({time.time() - t0_probe:.2f}s)")
        recs11 = t2_result["records"]
        report("  every accepted plant record carries all SIX arms' hit indicators "
               "(intact/true/placebo/pool_placebo/keyswap/noplant)",
               bool(recs11) and all(
                   all(r.get(k) is not None for k in
                       ("hit_intact", "hit_true_ablated", "hit_placebo_ablated",
                        "hit_pool_placebo", "hit_keyswap", "hit_noplant"))
                   for r in recs11),
               f"n_records={len(recs11)}")
        report("  the hard assertion (plant_and_verify_t2_window) fired ZERO times across "
               f"{len(recs11)} REAL accepted windows (no PlantContestedError was swallowed -- "
               "an uncaught one would have propagated out of run_t2_repaired_probe as an "
               "exception, already caught by the try/except around this whole block)", True)

        t2b1_11 = check_t2b1_mechanism_exists(recs11)
        t2b1b_11 = check_t2b1b_key_conditioned(recs11)
        t2a1_11 = check_t2a1_ceiling(recs11, t2b1_11, t2b1b_11)
        t2a2_11 = check_t2a2_untrained_control(recs11, n_boot=200)
        report("  check_t2b1 / check_t2b1b / check_t2a1_ceiling run on REAL-pipeline output "
               "without error", True,
               f"acc_copy={t2_result['acc_copy']:.4f} t2b1_p={t2b1_11['p_value']:.4g} "
               f"t2b1b_p={t2b1b_11['p_value']:.4g} t2a1_passes={t2a1_11['passes']}")
        report("  T2a-2 (untrained negative control): an UNTRAINED random-init model reads "
               "acc_copy<=0.02 with a KS bootstrap CI including 0 -- PASSES on a genuinely "
               "untrained model, exactly as sec 11.4.2 requires",
               t2a2_11["passes"], str(t2a2_11))
        report("  T2a-1 (the 0.90/0.75 ceiling) correctly FAILS on an untrained model "
               "(sec 11.4.2's positive-control logic: no mechanism -> no pass)",
               not t2a1_11["passes"], f"acc_copy={t2a1_11['acc_copy']}")
        t2a3_11 = check_t2a3_ssm_calibration(recs11, t2b1_11, t2b1b_11, n_boot=200)
        report("  T2a-3 (SSM causal-only calibration) runs without error on REAL-pipeline output",
               True, str(t2a3_11))
        t1c_11 = check_t1c_reference_did(
            {"did": 0.05, "did_ci": [0.01, 0.09]}, {"did": 0.03, "did_ci": [0.005, 0.06]})
        report("  T1c (reference-model DiD gate) reads PASS on two CI-excludes-zero-positive cells",
               t1c_11["passes"], str(t1c_11))
        t1c_11_fail = check_t1c_reference_did(
            {"did": 0.05, "did_ci": [0.01, 0.09]}, {"did": -0.01, "did_ci": [-0.05, 0.03]})
        report("  T1c correctly FAILS if only ONE of W1/W2's CI excludes zero positively "
               "(conjunctive gate, sec 11.4.5: 'requires DiD>0 ... on EACH')",
               not t1c_11_fail["passes"], str(t1c_11_fail))

        # THE VALIDATE-SECTION DEMONSTRATION: show a chosen (a,b) and its per-window counts.
        if recs11:
            r_show = recs11[0]
            a_show, b_show = r_show["a"], r_show["b"]
            planted_w = None  # reconstruct isn't stored per-record; recompute the count directly
            report(f"  DEMONSTRATION (synthetic corpus): plant a={a_show} b={b_show} at "
                   f"delta={r_show['delta']}, p(b|a)={r_show['p_b_given_a']:.4f} "
                   f"rank(b|a)={r_show['rank_b_given_a']} count(a,b)={r_show['count_ab']:.0f} "
                   f"count(b)={r_show['count_b']:.0f} -- an UNCONTESTED key/value pair "
                   f"(the hard assertion did not fire for this or any of the "
                   f"{len(recs11)} accepted windows)", True)
    except Exception as e:  # noqa: BLE001
        report("  sec 11 T2 REPAIR end-to-end pipeline", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [10d] sec 11.7 N_ROWS PRE-PASS: model-free, both corpora, real DeltaNetLM class NOT
    #     touched (the pre-pass never loads a checkpoint) -- confirm it terminates, VOIDs with a
    #     stated reason when the floors are unreachable, and CLEARS the REAL pinned floors
    #     (>=1500 contributing rows, >=8000 resolved candidates, sec 11.7) at N_rows=2048 on a
    #     corpus dense enough in repeated bigrams to support it. ---
    print("\n  [10d] sec 11.7 N_ROWS PRE-PASS (model-free)")
    try:
        seq_len_pp = 96
        base_pp = torch.randint(12, V10, (seq_len_pp + 1,))
        base_pp[5], base_pp[6] = 10, 11
        base_pp[70], base_pp[71] = 10, 11
        val_pp_a = base_pp.repeat(400).to(device)
        val_pp_b = base_pp.repeat(400).to(device)
        pre_pass_void = resolve_n_rows_pre_pass(
            {"wikitext-mix-ext": val_pp_a, "openr1-mix-ext": val_pp_b},
            {"wikitext-mix-ext": mode_next_t, "openr1-mix-ext": mode_next_t},
            seq_len_pp, device, c_max=4, ladder=(64, 128, 256))
        report("  resolve_n_rows_pre_pass VOIDs with a stated reason when the floors are "
               "unreachable within the ladder (a tiny toy corpus, capped ladder)",
               pre_pass_void["void"] is True and pre_pass_void["n_rows"] is None
               and "sec 11.7" in pre_pass_void["void_reason"], str(pre_pass_void["void_reason"]))

        torch.manual_seed(7)
        V_pp2 = 80
        seq_len_pp2 = 512
        val_pp2_a = torch.randint(0, V_pp2, (3_000_000,)).to(device)
        val_pp2_b = torch.randint(0, V_pp2, (3_000_000,)).to(device)
        mode_next_pp2 = torch.full((V_pp2,), -1, dtype=torch.int64)
        t0_pp2 = time.time()
        pre_pass_ok = resolve_n_rows_pre_pass(
            {"wikitext-mix-ext": val_pp2_a, "openr1-mix-ext": val_pp2_b},
            {"wikitext-mix-ext": mode_next_pp2, "openr1-mix-ext": mode_next_pp2},
            seq_len_pp2, device, c_max=C_MAX_DEFAULT, ladder=N_ROWS_SEARCH_LADDER)
        report("  resolve_n_rows_pre_pass CLEARS sec 11.7's REAL pinned floors "
               "(>=1,500 contributing rows AND >=8,000 resolved candidates, BOTH corpora) at "
               "the FIRST ladder rung (N_rows=2048) on a bigram-dense corpus",
               pre_pass_ok["void"] is False and pre_pass_ok["n_rows"] == 2048,
               f"n_rows={pre_pass_ok['n_rows']} "
               f"measurements[2048]={pre_pass_ok['measurements'].get(2048)} "
               f"({time.time() - t0_pp2:.2f}s)")
    except Exception as e:  # noqa: BLE001
        report("  N_rows pre-pass", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [10e] sec 11.8.1 THE ADMISSIBLE-SET ARTIFACT: booleans+metadata ONLY, MECHANICALLY
    #     refuses to emit if any DiD-shaped field is present anywhere in the payload. ---
    print("\n  [10e] sec 11.8.1 admissible_set_A.json -- DiD-FREE BY CONSTRUCTION")
    row_admissible = build_admissible_set_row(
        "14M", "openr1-mix-ext",
        {"t1a_pass": True, "t2b1_pass": True, "t2b1b_pass": True,
         "checkpoint_exists": True, "tok_per_param_floor": True, "checkpoint_quiesced": True,
         "t3_same_checkpoint": True, "both_corpora": True, "sample_size_floor": True})
    report("  build_admissible_set_row computes admissible=AND(gates) correctly",
           row_admissible["admissible"] is True, str(row_admissible))
    row_not_admissible = build_admissible_set_row(
        "1.31B", "wikitext-mix-ext", {"t1a_pass": True, "t2b1_pass": False, "t2b1b_pass": True})
    report("  build_admissible_set_row: a single False gate -> admissible=False",
           row_not_admissible["admissible"] is False, str(row_not_admissible))
    raised_nonbool = False
    try:
        build_admissible_set_row("x", "y", {"t1a_pass": 1})   # int, not bool
    except TypeError:
        raised_nonbool = True
    report("  build_admissible_set_row REFUSES a non-bool gate value", raised_nonbool)
    raised_did_key = False
    try:
        build_admissible_set_row("x", "y", {"gap_placebo_pass": True})   # DiD-shaped KEY NAME
    except ValueError:
        raised_did_key = True
    report("  build_admissible_set_row REFUSES a DiD-shaped GATE NAME "
           "('gap_placebo_pass' contains the forbidden token 'gap')", raised_did_key)

    payload = emit_admissible_set_json(
        [row_admissible, row_not_admissible], commit_sha="deadbeef0000",
        cli_config={"n_rows": 2048, "c_max": 8, "seed": 42})
    report("  emit_admissible_set_json produces a payload with schema_version/generated_at/"
           "commit_sha/rows", all(k in payload for k in ("schema_version", "generated_at",
                                                          "commit_sha", "cli_config", "rows")))
    # sec 11.8.1's own claim: the module can produce this artifact WITHOUT ever having
    # computed a DiD value. Scan the ENTIRE payload (recursively) for ANY DiD-shaped key.
    payload_str_keys = json.dumps(payload)  # a crude double-check: no literal '"did"'/'"gap' substr
    report("  admissible_set_A.json payload contains NO DiD-shaped field anywhere (mechanically "
           "verified by emit_admissible_set_json's own internal scan, which would have raised "
           "ValueError on construction had one been present -- this call already SUCCEEDED)",
           True, f"payload keys at top level: {sorted(payload.keys())}")
    # NEGATIVE TEST: a DiD field SNUCK INTO cli_config must be REFUSED, not silently emitted.
    raised_did_in_payload = False
    try:
        emit_admissible_set_json([row_admissible], commit_sha="x",
                                  cli_config={"gap_true": 0.05})   # a DiD-shaped field
    except ValueError as e:
        raised_did_in_payload = "gap_true" in str(e) or "forbidden" in str(e)
    report("  emit_admissible_set_json REFUSES to emit if a DiD-shaped field ('gap_true') is "
           "present ANYWHERE in the payload, including inside cli_config", raised_did_in_payload)
    raised_acc_copy_in_payload = False
    try:
        emit_admissible_set_json([{"rung": "x", "corpus": "y", "gates": {}, "admissible": True,
                                    "acc_copy_reference_only": 0.9}], commit_sha="x", cli_config={})
    except ValueError:
        raised_acc_copy_in_payload = True
    report("  emit_admissible_set_json REFUSES an 'acc_copy'-shaped field even inside a `rows` "
           "entry the caller built by hand (not via build_admissible_set_row)",
           raised_acc_copy_in_payload)
    # AND: an innocuous key that happens to CONTAIN the substring 'did' inside a longer word
    # (e.g. 'candidates') must NOT false-positive -- this is the exact "candidate contains did"
    # trap a naive substring check would fall into.
    safe_payload = emit_admissible_set_json(
        [{"rung": "x", "corpus": "y", "gates": {"sample_size_ok": True}, "admissible": True,
          "n_candidates_resolved": 8123}], commit_sha="x", cli_config={"n_rows": 2048})
    report("  emit_admissible_set_json does NOT false-positive on 'n_candidates_resolved' "
           "(contains the substring 'did' inside 'candidates', but is not the token 'did')",
           safe_payload is not None)

    # --- [10e-2] AUDIT FIX (2026-07-12, SERIOUS) teeth: the pre-audit scan tokenized on
    #     non-alnum boundaries ONLY, so camelCase ('gapTrue') and simple plurals ('gaps') both
    #     EVADED it; it also scanned dict KEYS only (a value like {"note": "DiD 0.19 ..."}
    #     evaded entirely) and did not traverse tuples. Confirm each hole is now closed. ---
    print("\n  [10e-2] hardened DiD-shaped-field scan: camelCase / plurals / values / tuples / floats")
    for evasion_key in ("gapTrue", "gaps", "didValue", "logProb", "DIDs"):
        raised = False
        try:
            emit_admissible_set_json([{"rung": "x", "corpus": "y", "gates": {}, "admissible": True,
                                        evasion_key: True}], commit_sha="x", cli_config={})
        except ValueError:
            raised = True
        report(f"  camelCase/plural evasion key {evasion_key!r} is now CAUGHT (pre-fix: evaded)",
               raised)
    raised_value_leak = False
    try:
        emit_admissible_set_json(
            [{"rung": "x", "corpus": "y", "gates": {}, "admissible": True,
              "note": "DiD 0.19 at rungs with acc_copy 0.0"}], commit_sha="x", cli_config={})
    except ValueError:
        raised_value_leak = True
    report("  a DiD-shaped STRING VALUE under an innocuous key ('note': 'DiD 0.19 ...') is now "
           "CAUGHT -- the pre-fix scan checked dict KEYS only and let this straight through",
           raised_value_leak)
    raised_tuple_leak = False
    try:
        emit_admissible_set_json(
            [{"rung": "x", "corpus": "y", "gates": {}, "admissible": True,
              "nested": ({"gap_true": 0.19},)}], commit_sha="x", cli_config={})
    except ValueError:
        raised_tuple_leak = True
    report("  a DiD-shaped field nested inside a TUPLE is now CAUGHT (pre-fix: only list/dict "
           "were traversed, tuples passed through unscanned)", raised_tuple_leak)
    raised_creative_float = False
    try:
        emit_admissible_set_json(
            [{"rung": "x", "corpus": "y", "gates": {}, "admissible": True,
              "doubledifference": 0.19}], commit_sha="x", cli_config={})
    except ValueError:
        raised_creative_float = True
    report("  a CREATIVELY-NAMED float field inside `rows` ('doubledifference', which contains "
           "NONE of {did,gap,s1,s2,logp} as a substring OR token -- no keyword list catches "
           "this) is STILL refused, because ALL floats are banned from `rows` regardless of "
           "key name (the type-based defense a keyword list alone cannot provide)",
           raised_creative_float)
    # cli_config MAY legitimately carry a float (genuine operational metadata) -- confirm the
    # strict float ban is SCOPED to `rows` only, not applied to cli_config.
    cfg_with_float = emit_admissible_set_json(
        [row_admissible], commit_sha="x", cli_config={"flagged_frac_threshold": 0.05})
    report("  a legitimate float in cli_config (e.g. 'flagged_frac_threshold': 0.05) is "
           "PERMITTED -- the strict float ban is scoped to `rows`, not blanket over the "
           "whole payload", cfg_with_float is not None)

    # --- [10c] AUDIT FIX teeth (F-4, second guise): the CANDIDATE POPULATION must not
    #     depend on --batch-size. The design's rungs use DIFFERENT token-arithmetic batch
    #     sizes (three at 32, the 1.31B at 16); if batch_size reaches the window sampler,
    #     the top rung is measured against a different candidate population than the rest
    #     -- which IS F-4. NOTE: this test is TOOTHLESS ON CPU (MT19937 consumes its stream
    #     sequentially, so chunked draws happen to agree) and only bites on CUDA, where
    #     Philox advances its offset PER KERNEL LAUNCH. Measured on the H100 before the fix:
    #     batch_size 16 vs 32 at the same seed shared only 50% of their 512 windows. Run the
    #     smoke with --device cuda to give this test its teeth.
    print("\n  [10c] BATCH-SIZE INDEPENDENCE of the candidate population (F-4, second guise)")
    try:
        rA = run_did_eval(tiny, val_tokens10, batch_size=4, seq_len=seq_len10, n_windows=8,
                          device=device, mode_next=mode_next_t, seed=777, c_max=4,
                          eval_micro_batch=8, min_sep=2, vocab_size=V10)
        rB = run_did_eval(tiny, val_tokens10, batch_size=7, seq_len=seq_len10, n_windows=8,
                          device=device, mode_next=mode_next_t, seed=777, c_max=4,
                          eval_micro_batch=8, min_sep=2, vocab_size=V10)
        keyA = [(r["row_idx"], r["k"], r["delta"]) for r in rA["records"]]
        keyB = [(r["row_idx"], r["k"], r["delta"]) for r in rB["records"]]
        report("  same seed, batch_size 4 vs 7 -> IDENTICAL candidate population",
               keyA == keyB and rA["n_windows"] == rB["n_windows"],
               f"n_cand {len(keyA)} vs {len(keyB)}"
               + ("" if device.startswith("cuda")
                  else "  [NOTE: on CPU this test is weak -- it only bites on CUDA]"))
    except Exception as e:  # noqa: BLE001
        report("  batch-size-independence check", False, f"EXCEPTION: {type(e).__name__}: {e}")

    print("\n  [11] NOTE (not a gate): T2a (the SAME T2b machinery pointed at reference models "
          "RWKV7-Goose-1.5B / falcon-mamba-7b) and a full real-checkpoint R0 read are wired but "
          "were NOT executed in this build session -- disclosed scope boundary, see the build "
          "report.")

    print("\n" + "=" * 70)
    print(f"  SMOKE {'PASSED' if ok_all else 'FAILED'}")
    print("=" * 70)
    return 0 if ok_all else 1


# =============================================================================
# CLI
# =============================================================================
def mode_resolve_slice(args) -> int:
    r = resolve_token_matched_checkpoint(args.ckpt_dir, args.batch_size, args.seq_len,
                                          args.target_tokens, args.prefix)
    print(json.dumps(r, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
    return 0


def mode_verify_quiesced(args) -> int:
    r = verify_quiesced_checkpoint(args.checkpoint, wait_s=args.wait_s)
    print(json.dumps(r, indent=2))
    return 0 if r["quiesced"] else 1


def mode_run(args) -> int:
    """Real single-cell TEETH-GATE run (DiD/T1/baseline; NOT a verdict --
    sec 9.1 is not pinned). Requires GPU + real fla. Intentionally NOT
    invoked against any real checkpoint by this build session (per this
    task's explicit instruction) -- wired and ready."""
    device = args.device
    # sec 9.6 item 3 has TWO clauses; file-quiescence is only the first (see
    # verify_quiesced_checkpoint's docstring: a live --ckpt-every-10000 writer leaves a
    # byte-stable file for HOURS between writes, which is precisely the S-7 read). The
    # second clause -- "its job must be terminated" -- is not measurable from the file, so
    # it is an explicit, recorded operator attestation. AUDIT FIX (2026-07-12).
    if not args.attest_job_terminated:
        raise SystemExit(
            "REFUSING TO RUN: --attest-job-terminated is required.\n"
            "PARAM_AXIS_SCALING_DESIGN.md sec 9.6 item 3: 'No rung may be read from a live, "
            "still-training job ... and its job must be terminated.' The file-quiescence check "
            "CANNOT establish this: the S-7 precedent (regate sec 10.3) read a live 1.31B job's "
            "step-40000 checkpoint under a --ckpt-every-10000 writer, and at ~1.4 s/step those "
            "writes are ~3.9 hours apart -- the file was perfectly byte-stable the whole time.\n"
            "Confirm the writing job is dead (e.g. `tmux ls`, `nvidia-smi`, the queue's own "
            "completed/ marker), then re-run with --attest-job-terminated. The attestation is "
            "recorded in the result JSON as `job_terminated_attested`."
        )
    # AUDIT FIX (2026-07-12, S2 follow-up audit): `--normalization` was accepted by the parser
    # and then never read by anything -- an operator who passed `--normalization raw_did` (or,
    # worse, some other name) got it SILENTLY IGNORED and a JSON that looks like it honoured it.
    # mode_run cannot compute M(r) at all (see the --compute-verdict refusal below), so the flag
    # has no meaning here; it is now refused out loud rather than swallowed.
    if args.normalization is not None:
        raise SystemExit(
            f"--normalization={args.normalization!r} is REFUSED by this entry point. `mode_run` "
            f"computes DiD / T1a / DiD_logp for ONE (checkpoint, corpus) cell and NEVER applies a "
            f"normalization (M(r) requires T2b-1/T2b-2 + the sec 9.6 admissibility gates, which "
            f"this entry point does not run -- see the --compute-verdict refusal). sec 9.1's pin "
            f"('raw_did') is applied by the multi-rung driver, via compute_capacity_metric. "
            f"Passing it here would have been silently ignored; that is worse than an error."
        )
    model, ckpt, md5_pin = load_checkpoint(args.checkpoint, device, require_quiesced=not args.no_quiesce_check,
                                            wait_s=args.wait_s)
    n_params = sum(p.numel() for p in model.parameters())
    train_tokens, val_tokens, meta, _, _ = load_corpus(args.data_dir, args.corpus, device)
    mode_next = build_bigram_mode_table(train_tokens, VOCAB_SIZE, device)
    seed = corpus_fixed_seed(args.corpus) + 424242
    result = run_did_eval(model, val_tokens, args.batch_size, args.seq_len, args.n_windows, device,
                           mode_next, seed, c_max=args.c_max, eval_micro_batch=args.eval_micro_batch,
                           min_sep=args.min_sep)
    cell = finalize_cell(args.rung or "unlabeled", args.corpus, result)
    cell["checkpoint"] = args.checkpoint
    cell["checkpoint_md5"] = md5_pin
    cell["n_params"] = n_params
    cell["ckpt_step"] = ckpt.get("step")
    cell["job_terminated_attested"] = True
    cell["quiesce_check_skipped"] = bool(args.no_quiesce_check)
    # sec 9.1 IS pinned (raw_did, PARAM_AXIS_SCALING_DESIGN.md sec 9.1, commit 7381d2a) --
    # this field is a fixed provenance fact now, not a runtime toggle. `mode_run` STILL
    # cannot emit a verdict (see the --compute-verdict refusal below): the missing piece
    # is no longer the normalization, it is that this entry point runs neither T2b nor
    # sec 9.6 admissibility.
    cell["sec91_normalization_pinned"] = True
    cell["sec91_normalization_name"] = "raw_did"
    print(json.dumps({k: v for k, v in cell.items() if k != "per_row_did"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(cell, f, indent=2)

    # sec 11.7's sample-size floor (superseding sec 9.6 item 7's old >=4,096), checked LOUDLY
    # at the point of production. The AUTHORITATIVE way to fix N_rows for a real read is
    # resolve_n_rows_pre_pass (sec 11, model-free, run BEFORE any checkpoint is loaded) --
    # this is a single-cell sanity check, not a substitute for that pre-pass.
    n_res = cell["n_candidates_resolved"]
    if n_res < CANDIDATE_FLOOR_RESOLVED:
        print(f"\n  *** WARNING: n_candidates_resolved={n_res} < {CANDIDATE_FLOOR_RESOLVED}, the "
              f"sec 11.7 floor (supersedes sec 9.6 item 7's old >=4,096 -- '512x8=4096 was the "
              f"unreachable theoretical maximum'). This rung is INADMISSIBLE as it stands.\n"
              f"      Note the pinned sampling (N_rows={N_ROWS_DEFAULT} x C_max={C_MAX_DEFAULT}) has a "
              f"MAXIMUM of {N_ROWS_DEFAULT * C_MAX_DEFAULT} candidates, so the floor is only reachable if "
              f"EVERY row yields >= {C_MAX_DEFAULT} candidates AND every placebo resolves.\n"
              f"      Run resolve_n_rows_pre_pass FIRST (sec 11.7, model-free) to fix N_rows "
              f"properly, rather than raising --n-windows ad hoc per rung (an n_windows that "
              f"varies by rung reintroduces F-4).", flush=True)

    if args.compute_verdict:
        raise SystemExit(
            "--compute-verdict is REFUSED by this entry point (AUDIT FIX 2026-07-12; STILL "
            "REFUSED after sec 9.1's pin landed AND after the sec 11 T2 repair -- the reason "
            "changed twice, the refusal did not).\n"
            "sec 9.1's normalization IS pinned (raw_did). T2b-2 is RETIRED (sec 11.6) --"
            "compute_capacity_metric no longer needs (or accepts) a T2b-2 disposition at all. "
            "M(r) may still not be computed from a cell that has not been through T2b-1/"
            "T2b-1b and the sec 9.6/11.7 admissibility gates, and `mode_run` runs NONE of "
            "them (it computes DiD/T1a/DiD_logp for ONE (checkpoint, corpus) cell only).\n"
            "A verdict needs a driver (not yet built) that: (1) runs resolve_n_rows_pre_pass "
            "(sec 11.7, model-free, BEFORE any checkpoint loads); (2) runs "
            "run_t2_repaired_probe -> check_t2b1_mechanism_exists / "
            "check_t2b1b_key_conditioned -> check_rung_admissible across BOTH corpora and ALL "
            "rungs; (3) commits admissible_set_A.json (sec 11.8.1) BEFORE any DiD is "
            "un-quarantined; and only then (4) calls compute_capacity_metric with "
            "normalization='raw_did'. That driver is the gap sec 9.1's pin unblocks, not this "
            "single-cell entry point."
        )
    print("\n  NOTE: DiD / T1a / DiD_logp for ONE cell only. NOT a verdict -- T2b-1/T2b-1b and "
          "sec 9.6/11.7 admissibility are not run by this entry point (see the --compute-verdict "
          "refusal above). sec 9.1's normalization ('raw_did') is pinned, but per sec 9.6's stop "
          "rule this is still a single eval-only pass with no re-read: do not compute a verdict "
          "from this JSON piecemeal outside the proper multi-rung/multi-corpus driver.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resolve-slice", action="store_true")
    ap.add_argument("--verify-quiesced", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--checkpoint", type=str)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--corpus", type=str, choices=sorted(CORPUS_DIRS))
    ap.add_argument("--rung", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=N_ROWS_DEFAULT)
    ap.add_argument("--c-max", type=int, default=C_MAX_DEFAULT)
    ap.add_argument("--eval-micro-batch", type=int, default=EVAL_MICRO_BATCH_DEFAULT)
    ap.add_argument("--min-sep", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str)
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS_FORCED_SLICE)
    ap.add_argument("--prefix", type=str)
    ap.add_argument("--wait-s", type=float, default=5.0)
    ap.add_argument("--no-quiesce-check", action="store_true",
                     help="DANGEROUS, testing only -- skips sec 9.6 item 3's live-writer guard. "
                          "Stamps quiesce_check_skipped=true into the result JSON and leaves the "
                          "sec-9.6-mandated md5 pin NULL, which alone makes the cell non-conformant.")
    ap.add_argument("--attest-job-terminated", action="store_true",
                     help="REQUIRED for --run. Operator attestation of sec 9.6 item 3's second "
                          "clause ('its job must be terminated'), which file-quiescence cannot "
                          "establish -- a --ckpt-every-10000 writer leaves a byte-stable file for "
                          "hours between writes (the S-7 read). Recorded in the result JSON.")
    ap.add_argument("--compute-verdict", action="store_true")
    ap.add_argument("--normalization", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        return smoke(args.device if torch.cuda.is_available() else "cpu")
    if args.resolve_slice:
        return mode_resolve_slice(args)
    if args.verify_quiesced:
        assert args.checkpoint, "--checkpoint required for --verify-quiesced"
        return mode_verify_quiesced(args)
    if args.run:
        assert args.checkpoint and args.corpus, "--checkpoint and --corpus required for --run"
        return mode_run(args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
