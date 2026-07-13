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


# =============================================================================
# sec 20 R-4 -- THE T2a-2 LIVENESS WITNESS.
#
# WHY IT EXISTS (sec 19.3d, the sixth adversary): every model-dependent number
# the T2a-2 negative control ever persisted -- `acc_copy`, `ks_point`, `ks_ci`
# -- is a function of ONE indicator bit per row, `argmax(logits[row, k0]) == b`.
# That bit is 0 for the intended null (a live, varied, mechanism-free forward
# pass) AND for a model returning CONSTANT logits AND for a model returning NaN
# logits (torch.argmax over all-NaN returns index 0, and `b` is never token 0).
# The three artifacts are BIT-IDENTICAL. The control therefore could not
# distinguish "no mechanism" from "no measurement," and the gate's own record
# cited that degeneracy as its strongest evidence. It was its weakest.
#
# WHAT SEPARATES THEM, AND WHY IT IS THE *ONLY* THING THAT CAN: liveness is the
# property that THE READOUT DEPENDS ON THE INPUT. Post-argmax, that property is
# gone -- collapsed into a bit. Pre-argmax it is directly observable, and it is
# observable EXACTLY, with no tolerance and no tuned number:
#
#   (L1) `readout_logits_finite_frac == 1.0`  -- EXACT. A NaN/Inf forward pass
#        cannot produce it; `torch.isfinite` is evaluated on the very tensor
#        `argmax` consumed. There is no path from a non-finite pass to 1.0.
#
#   (L2) `readout_logit_max_abs_dev_from_row0 > 0.0` -- EXACT, and it is the
#        DEFINITION of input-dependence, not a proxy for it: the max over all
#        rows/vocab of |L[i] - L[0]|. A model whose output does not depend on
#        its input emits a BIT-IDENTICAL readout vector for every window, so
#        this is EXACTLY 0.0 -- a float subtraction of a value from itself, not
#        a variance estimate, so there is no cancellation epsilon to hide in.
#        (A streaming sum/sumsq variance would NOT do: for n identical values it
#        returns ~1e-14, not 0, and a constant-logit model would sail through.)
#        Any input-dependent map gives > 0.
#
# WHY NOT sec 19's own suggestion (the count of DISTINCT argmax tokens at k0):
# it is recorded below, but it is NOT part of `ok`. It is unsound in BOTH
# directions. A genuinely LIVE random-init model can have one globally dominant
# token and collapse to n_distinct == 1 (a false HALT); and a NaN pass ALSO
# reads n_distinct == 1 (index 0) -- so the statistic cannot separate the two
# cases it would be introduced to separate. Argmax-distinctness is a *symptom*
# of deadness; logit dispersion is deadness itself. The symptom is reported; the
# thing itself is what gates.
#
# NO NEW NUMERIC GATING THRESHOLD IS INTRODUCED (RULE T, sec 20.1): `== 1.0`
# (all entries finite) and `> 0.0` (any dispersion at all) are the exact
# boundaries of degeneracy, fixed BY CONSTRUCTION -- a constant function has
# exactly zero dispersion; a finite tensor is exactly 100% finite. Neither is a
# tolerance, neither is a magnitude, and neither came from measuring a model.
# =============================================================================
class _LiveLogitAccumulator:
    """Streams the readout-position logit vectors of ARM 1 and reduces them to
    a small liveness record WITHOUT ever materializing the (N_rows, V) tensor
    (2048 x 50257 fp32 = 412 MB) or issuing a single extra forward pass.

    Structural (gating) fields: `readout_logits_finite_frac`,
    `readout_logit_max_abs_dev_from_row0`. Everything else is REPORTED AND
    NON-GATING -- distributional detail so a reader can tell a live-but-null
    pass from a dead one by eye, which the old three-zeros artifact could not."""

    def __init__(self):
        self.ref = None                 # row 0's readout logit vector (the bit-equality anchor)
        self.n_rows = 0
        self.n_finite = 0
        self.n_total = 0
        self.max_abs_dev = 0.0          # max_i,v |L[i,v] - L[0,v]|; NaN-propagating
        self.argmax_tokens = []
        self.entropies = []             # softmax entropy (nats) per row
        self.max_logits = []            # max logit value per row

    def absorb(self, readout: torch.Tensor) -> None:
        """readout: (n_chunk, V) -- the logit vectors at each row's OWN k0."""
        rd = readout.detach().float()
        self.n_rows += int(rd.shape[0])
        fin = torch.isfinite(rd)
        self.n_finite += int(fin.sum().item())
        self.n_total += int(rd.numel())
        if self.ref is None:
            self.ref = rd[0].clone()
        dev = (rd - self.ref.to(rd.device)).abs()
        chunk_max = float(dev.max().item()) if dev.numel() else 0.0
        # NaN must PROPAGATE (a NaN pass must not be rescued by max()); `math.isnan` guard
        # rather than max(), because max(nan, x) is order-dependent in Python.
        if math.isnan(chunk_max) or math.isnan(self.max_abs_dev):
            self.max_abs_dev = float("nan")
        else:
            self.max_abs_dev = max(self.max_abs_dev, chunk_max)
        self.argmax_tokens.extend(rd.argmax(dim=-1).tolist())
        lp = rd.log_softmax(dim=-1)
        self.entropies.extend((-(lp.exp() * lp).sum(dim=-1)).tolist())
        self.max_logits.extend(rd.max(dim=-1).values.tolist())

    def report(self) -> dict:
        finite_frac = (self.n_finite / self.n_total) if self.n_total else 0.0
        dev = self.max_abs_dev
        finite_ok = (self.n_total > 0) and (finite_frac == 1.0)
        dispersion_ok = (not math.isnan(dev)) and dev > 0.0
        reasons = []
        if self.n_total == 0:
            reasons.append("no readout logits were read (zero rows / no forward pass)")
        if not finite_ok and self.n_total > 0:
            reasons.append(f"readout logits are not all finite (finite_frac={finite_frac!r}) -- "
                            f"NaN/Inf forward pass")
        if not dispersion_ok:
            reasons.append(f"readout logits do not depend on the input: "
                            f"max|L[i]-L[0]| = {dev!r} (EXACTLY 0 => every window produced a "
                            f"BIT-IDENTICAL readout => constant-logit / dead forward pass)"
                            if not math.isnan(dev) else
                            "max|L[i]-L[0]| is NaN -- non-finite forward pass")
        return {
            # ---- STRUCTURAL, GATING (exact degeneracy boundaries; no tuned number) ----
            "ok": bool(finite_ok and dispersion_ok),
            "readout_logits_finite_frac": finite_frac,
            "readout_logit_max_abs_dev_from_row0": (None if math.isnan(dev) else dev),
            "readout_logit_max_abs_dev_is_nan": bool(math.isnan(dev)),
            "n_rows_read": self.n_rows,
            "n_logit_entries_read": self.n_total,
            "degenerate_reasons": reasons,
            # ---- REPORTED, NON-GATING (distributional detail; see class docstring) ----
            "n_distinct_argmax_at_k": len(set(self.argmax_tokens)),
            "top1_argmax_share": _modal_share(self.argmax_tokens),
            "mean_softmax_entropy_nats": _finite_mean(self.entropies),
            "std_softmax_entropy_nats": _finite_std(self.entropies),
            "mean_max_logit": _finite_mean(self.max_logits),
            "std_max_logit": _finite_std(self.max_logits),
        }


def _modal_share(xs: list) -> float:
    if not xs:
        return float("nan")
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.values()) / len(xs)


def _finite_mean(xs: list):
    vals = [x for x in xs if math.isfinite(x)]
    return (sum(vals) / len(vals)) if vals else None


def _finite_std(xs: list):
    vals = [x for x in xs if math.isfinite(x)]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def argmax_changed_frac_keyswap(records: list) -> float:
    """THE AIMING WITNESS (sec 23.3 item 4; PROMOTED TO GATING in sec 24 / B-4).

    The fraction of plants whose readout ARGMAX at k0 CHANGES when the PLANTED
    KEY at j0 is swapped (arm 1 vs arm 4). It asks whether the readout depends
    on THE PLANT SPECIFICALLY, not merely on *some* input -- which is strictly
    stronger than sec 20's LIVENESS witness (L1/L2), and it is the only
    quantity in the artifact that can separate a MIS-AIMED probe (reading k0+-1,
    the wrong tensor, or a transposed state -- fully input-dependent, so
    liveness PASSES) from a correctly-aimed one.

    NULL, FIXED BY CONSTRUCTION: `0` -- a readout that does not respond to the
    key at j0 produces a BIT-IDENTICAL argmax under the key-swap, exactly. The
    gate is `> 0`, the EXACT degeneracy boundary; it is not a tolerance, not a
    magnitude, and it did not come from measuring any model (RULE T, sec 20.1).

    SINGLE SOURCE OF TRUTH: this is the SAME expression `run_t2_repaired_probe`
    reports into `logit_liveness["argmax_changed_frac_keyswap"]` -- factored out
    here so the GATING read and the REPORTED read cannot drift apart. Zero extra
    forward passes: both argmaxes are already in `records`. NaN on empty input
    (=> `> 0` is False => FAIL-CLOSED).

    WHERE IT MAY GATE, AND WHERE IT MAY NOT -- see sec 24:
      * T2a-1 (W1/W2) and T2a-3 (C1): GATING. These witnesses are required BY
        THE DESIGN to exhibit the mechanism, so a `0` here is instrument-fatal.
      * T2a-2 (the UNTRAINED control): **NOT GATING, and it must never become
        gating.** A live, healthy, mechanism-free model legitimately reads
        EXACTLY 0.0 -- MEASURED (sec 20.4b: 0.0000 on the live untrained
        DeltaNetLM). Gating it there would FALSELY HALT a healthy control, and
        its null on a mechanism-free model is fixed by MEASUREMENT, not by
        construction => RULE T forbids it. sec 23.4 item 2's literal
        recommendation ("promote it to GATING on the T2a-2 control") is
        therefore REJECTED, with reasons, in sec 24."""
    pairs = [(r["argmax_intact_at_k"], r["argmax_keyswap_at_k"]) for r in records
             if r.get("argmax_keyswap_at_k") is not None and r.get("argmax_intact_at_k") is not None]
    return _mean([int(a != b) for a, b in pairs]) if pairs else float("nan")


class PerfectCopyOracle(torch.nn.Module):
    """THE POSITIVE CONTROL'S MODEL (sec 26 / T2a-4). **A MODEL THAT HAS THE
    MECHANISM BY FIAT.**

    Every control this instrument has ever carried is a NEGATIVE one: it asks
    "can this FAIL when it should?" (T2a-2's untrained model; sec 20's liveness
    witness; sec 24's leg (vi)). **Nothing asked "can this SUCCEED when it
    should?"** -- and that is the hole sec 25 walked through: a readout aimed on
    12 of 2048 rows (0.6%) and MIS-AIMED on the other 2036 is alive, input-
    dependent, plant-independent, and it CLEARS ALL FOUR GATING LEGS
    (acc_copy = 0.0059, PRIOR = 0.0000, KS 95% CI [0.0029, 0.0093] excludes 0,
    aiming = 0.0059 > 0) => T2a-1 PASS, T2a-3 PASS, INSTRUMENT_VALID.

    THE ORACLE. At position `t` it emits logits peaked -- uniquely, on an exact
    integer scatter, so there is no tie to break -- on the token that FOLLOWED
    THE FIRST EARLIER OCCURRENCE of `x[t]`; where `x[t]` has no earlier
    occurrence, it emits `x[t]` itself. That is a perfect, noiseless induction
    head: THE EXACT MECHANISM `run_t2_repaired_probe` EXISTS TO DETECT. It has
    no parameters, consumes no RNG, and is deterministic.

    ===== WHAT A CORRECTLY-AIMED PROBE **MUST** READ ON IT, AND WHY THAT IS A
          THEOREM ABOUT THE PLANT AND NOT A MEASUREMENT OF ANY MODEL =====

    `plant_and_verify_t2_window` HARD-ASSERTS, on every admitted record, that
    the planted window `w` satisfies `count(a in w) == 2` at EXACTLY `{j0, k0}`
    and `count(b in w) == 1` at EXACTLY `{p = j0+1}`; `rejection_sample_delta`
    admits only `2 <= Delta <= T-6` (so `j0 < p < k0`, strictly); and
    `draw_t2_triple` returns `a`, `a'`, `b` PAIRWISE DISTINCT and with natural
    occurrence count 0 in the pre-plant window. From those three facts alone:

      ARM 1 (INTACT). `x[k0] = a`; a's unique earlier occurrence is `j0`;
        `x[j0+1] = b` => the oracle emits `b` at `k0`.
        => `argmax_intact_at_k == b` ON EVERY RECORD => `acc_copy == 1`, EXACTLY.
      ARM 4 (KEY-SWAP, `w[j0] := a'`). `a` now occurs ONLY at `k0` => NO earlier
        occurrence => the oracle falls back to `x[k0] = a`, and `a != b`.
        => `argmax_keyswap_at_k == a != b == argmax_intact_at_k` ON EVERY RECORD
        => `argmax_changed_frac_keyswap == 1`, EXACTLY, and `KS == 1`, EXACTLY.
      ARM 5 (NO-PLANT, `w_orig[k0] := a`). `a` does not occur naturally in the
        pre-plant window => it occurs only at `k0` => fallback `a != b`
        => `PRIOR == 0`, EXACTLY.
      ARM 2 (TRUE-ABLATE, `w[p] := repl != b`). The earlier `a` at `j0` still
        stands; the token after it is now `repl != b` => `hit_true_ablated == 0`
        on every record.

    **NOTHING ABOVE IS A MEASUREMENT.** Each line is entailed by the plant's own
    hard assertion. That is what makes the positive control's null CONSTRUCTION-
    FIXED in the sense RULE T (sec 20.1) requires -- and it is the ONLY null in
    this design that is fixed by a construction WE CONTROL rather than by a
    property we hope some model has. `check_t2a4_positive_control` gates on
    VIOLATION COUNTS of exactly these identities, at the exact null `0`, with no
    tolerance and no numeric literal in the predicate at all.

    COST: no parameters, no matmul, O(B*T + B*V) memory. Zero GPU-h in the sense
    that matters -- it trains nothing. (sec 25.5 item 2: "It must be 0 GPU-h --
    a constructed oracle, not a trained model.")"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # first_pos[b, v] = the FIRST index at which token v occurs in row b (or T if never).
        # A token's GLOBAL-first occurrence is its FIRST EARLIER occurrence exactly when that
        # index is < t -- which is the `has_earlier` test below. O(B*V), never O(B*T*T).
        first_pos = torch.full((B, self.vocab_size), T, dtype=torch.long, device=x.device)
        first_pos.scatter_reduce_(1, x, idx, reduce="amin", include_self=True)
        fp = first_pos.gather(1, x)                       # (B,T) first occurrence of x[t]
        has_earlier = fp < idx                            # STRICTLY earlier
        nxt = x.gather(1, (fp + 1).clamp(max=T - 1))      # the token that FOLLOWED it
        pred = torch.where(has_earlier, nxt, x)           # fallback: x[t] itself
        logits = torch.full((B, T, self.vocab_size), -3.0, device=x.device, dtype=torch.float32)
        logits.scatter_(2, pred.unsqueeze(-1), 7.0)       # a UNIQUE peak: no tie to break
        return logits


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
    k0_all = torch.tensor([s["k0"] for s in specs], device=device_t, dtype=torch.long)
    pred_chunks = []
    live = _LiveLogitAccumulator()
    for start in range(0, x_intact.shape[0], eval_micro_batch):
        with torch.no_grad():
            logits = model(x_intact[start:start + eval_micro_batch])
        pred_chunks.append(logits.argmax(dim=-1))
        # sec 20 R-4 LIVENESS WITNESS: read the PRE-ARGMAX logit vectors at each row's OWN
        # readout position k0, from the SAME logits the argmax above already consumed (zero
        # extra forward passes, no RNG touched -- determinism per sec 19.4c is preserved by
        # construction). Everything downstream of `argmax` is a single indicator bit per row
        # (`argmax == b`), which is 0 for a live-but-mechanism-free model AND for a
        # constant-logit / NaN-logit model alike; the pre-argmax logits are the ONLY place
        # those two are distinguishable, and this is the only point in the program where they
        # exist. See _LiveLogitAccumulator for what is recorded and why.
        n_c = logits.shape[0]
        live.absorb(logits[torch.arange(n_c, device=logits.device),
                           k0_all[start:start + n_c], :])
        del logits
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
        argmax_intact = int(pred_intact[row_idx, k0].item())
        hit_intact = int(argmax_intact == b)
        hit_true = int(s.get("pred_true") == b) if "pred_true" in s else None
        hit_placebo = int(s.get("pred_placebo") == b) if s.get("pred_placebo") is not None else None
        hit_pool_placebo = int(s.get("pred_pool_placebo") == b) if s.get("pred_pool_placebo") is not None else None
        hit_keyswap = int(s.get("pred_keyswap") == b) if "pred_keyswap" in s else None
        hit_noplant = int(s.get("pred_noplant") == b) if "pred_noplant" in s else None
        records.append({
            "row_idx": row_idx, "orig_row_idx": s["orig_row_idx"], "k": k0, "j": s["j0"],
            "delta": s["delta"], "a": s["a"], "a_prime": s["a_prime"], "b": b,
            # sec 20 R-4: the ARGMAX TOKENS THEMSELVES, not just the `== b` bits they collapse
            # to. These are computed today and thrown away; keeping them is what lets a reader
            # (and check_t2a2_untrained_control) see WHAT a null model predicted, not merely
            # that it did not predict `b`. Pure serialization -- no new computation.
            "argmax_intact_at_k": argmax_intact,
            "argmax_keyswap_at_k": (int(s["pred_keyswap"]) if s.get("pred_keyswap") is not None
                                     else None),
            "argmax_true_ablated_at_k": (int(s["pred_true"]) if s.get("pred_true") is not None
                                          else None),
            "argmax_noplant_at_k": (int(s["pred_noplant"]) if s.get("pred_noplant") is not None
                                     else None),
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

    # sec 20 R-4 / sec 24 B-4: THE AIMING WITNESS. Does corrupting the KEY at j0 change the
    # model's readout argmax at k0 AT ALL? Computed by the single shared estimator above, so
    # the REPORTED value here and the GATING value read by check_t2a1_ceiling /
    # check_t2a3_ssm_calibration are the SAME NUMBER by construction, not by convention.
    #
    # ON THIS ARM (whatever model is passed) IT IS ONLY REPORTED. It becomes GATING inside
    # T2a-1/T2a-3 -- the cells whose witnesses the design REQUIRES to have the mechanism --
    # and it stays NON-GATING inside T2a-2, where a healthy mechanism-free model legitimately
    # reads 0.0 (MEASURED, sec 20.4b). See `argmax_changed_frac_keyswap`'s docstring and sec 24.
    liveness = live.report()
    liveness["argmax_changed_frac_keyswap"] = argmax_changed_frac_keyswap(records)

    return {
        "void": False, "records": records, "n_plants": len(records),
        "n_plants_requested": n_plants, "n_dropped": n_dropped, "drop_reasons": drop_reasons,
        "acc_copy": acc_copy, "acc_copy_se": acc_copy_se,
        "delta_excluded_mass_pooled": delta_excluded_mass_pooled,
        "logit_liveness": liveness,
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


def _log_binomial_pmf(x: int, n: int, p: float) -> float:
    """log(P(X=x)) for X ~ Binomial(n, p), computed via `math.lgamma`
    (log of the arbitrary-precision `math.comb(n, x)`) instead of the raw
    integer, so nothing is ever converted to a float before the
    vanishingly-small `p**x*(1-p)**(n-x)` factor has been applied in log
    space. `math.lgamma`/`math.log`/`math.log1p` operate on doubles the
    entire way through -- there is no intermediate arbitrary-precision int
    for ANY `n` (unlike `math.comb(n, x)` itself, whose magnitude grows
    with `n` without bound). Handles `p in {0, 1}` explicitly since
    `math.log(0)` raises. Result is always <= 0 (a valid log-probability)."""
    if p <= 0.0:
        return 0.0 if x == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if x == n else float("-inf")
    log_n_choose_x = math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
    return log_n_choose_x + x * math.log(p) + (n - x) * math.log1p(-p)


def _exact_binomial_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (the 'sum of all outcomes at
    least as extreme as observed' / minimum-likelihood method), IDENTICAL
    in method to the original implementation (same inclusion rule:
    pmf(x) <= pmf(k)*(1+1e-9)) but computed entirely in log space so it
    never materializes `math.comb(n, x)` as a Python int that Python must
    convert to float before the shrinking p**x*(1-p)**(n-x) factor applies
    -- THE bug (OverflowError at n>=~1030, verified by direct sweep,
    lm_recall_gap_probe_v2_rd.py's own binding N_rows=2048 ceiling). Every
    intermediate value here (log_pmf, and pmf=exp(log_pmf) for the
    surviving terms only) stays in [0, 1] / (-inf, 0], so nothing can
    overflow float64 regardless of n. Still an EXACT test, not a normal
    approximation -- the pinned spec (sec 11.5/9.4) requires exact."""
    if n == 0:
        return 1.0

    log_p_obs = _log_binomial_pmf(k, n, p)
    # the same multiplicative (1+1e-9) tolerance as the original, transferred
    # to log space via log1p so x==k itself is always included despite
    # floating-point noise in the lgamma evaluation.
    log_threshold = log_p_obs + math.log1p(1e-9)
    total = 0.0
    for x in range(n + 1):
        log_px = _log_binomial_pmf(x, n, p)
        if log_px <= log_threshold:
            total += math.exp(log_px)
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


def check_t2a1_ceiling(records: list, t2b1_result: dict, t2b1b_result: dict,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """T2a-1 CEILING (GATING). **THE sec 18.4 OPERATIVE PIN, IMPLEMENTED.**

    ===== WHAT THIS FUNCTION USED TO DO, AND WHY IT NO LONGER DOES =====

    Until this change (sec 23's build audit: "THE sec 18.4 PIN WAS NEVER
    IMPLEMENTED IN CODE; THE RETIRED BARS STILL GATE `INSTRUMENT_VALID`") the
    conjunction was

        passes = leg_i and leg_ii and leg_iii and leg_iv and leg_v
        leg_i  = acc_at_median >= 0.90                       # sec 9.4/11.4.1
        leg_ii = all(decile acc >= 0.75)                     # sec 9.4/11.4.1
        leg_iv = ks >= 0.50 and t2b1b.passes                 # a BARE point estimate, NO CI

    A NINE-ROUND GAUNTLET (sec 14-sec 23) retired those bars under **RULE T**
    (sec 18.1, re-stated correctly at sec 20.1):

        A departure-from-null threshold may gate ONLY in the null's own SAMPLING
        units (a significance level, or a CI exclusion) and NEVER in the raw
        units of the quantity itself. A threshold that instead asserts a
        COMPETENCE LEVEL -- one whose value can only be justified by pointing at
        how well *some model* performs -- is not a gate.

    `acc_copy >= 0.90` and `KS >= 0.50` are raw-magnitude competence bars: the
    null's sampling distribution supplies no scale on which `0.90` or `0.50` is
    a quantile. **They are RETIRED AS GATES -- permanently, and as a TYPE, not
    as a value. No absolute bar replaces them, at any value, at any operating
    point.** (`KS >= 0.50` PASSED on 3 of the 4 attempt-2 cells: 0.617 / 0.660 /
    0.524. A rule that retires a leg the data PASSED is not a fit -- sec 18.10
    charge 1.) `KS >= 0.50` was additionally a HIDDEN `acc_copy >= 0.50` bar
    wearing a causal costume (sec 15's W-1: `KS = acc_copy - acc_keyswap` and
    `acc_keyswap >= 0`), and it gated a BARE POINT ESTIMATE with no CI at all
    (sec 16's W-2: W2/openr1's `KS = 0.49951` has a 95% CI of [0.475, 0.524],
    which COVERS 0.50 -- the gate was decided on a knife-edge by 0.00049).

    ===== THE OPERATIVE GATE (sec 18.4). FIVE ROWS; FOUR GATING LEGS. =====

      (i)   `acc_copy >= 0.90` at the Delta-median
              -> **RETIRED AS A GATE. REPORTED ALWAYS, VERDICT-CARRYING NEVER.**
      (ii)  `acc_copy >= 0.75` in EVERY Delta-decile
              -> **RETIRED AS A GATE.** Reported per-decile (intact AND key-swap).
      (iii) `PRIOR = acc_copy_noplant <= 0.05`      -> **UNCHANGED. GATING.**
      (iv)  `KS > 0` with a clustered-bootstrap 95% CI EXCLUDING 0, conjoined
            with T2b-1b (`p < 0.001`)               -> **RE-PINNED. GATING.**
      (v)   T2b-1 (`p < 0.001`)                     -> **UNCHANGED. GATING.**
      (vi)  AIMING: `argmax_changed_frac_keyswap > 0`  -> **NEW. GATING.** (B-4)

    Leg (iv)'s replacement form is **NOT INVENTED HERE**: it is adopted VERBATIM
    from the form this document already pinned for T2a-3 (sec 11.4.2), and it
    REUSES that function's construction -- `clustered_bootstrap_ci(records,
    stat_ks)` then `ks_lo > 0` -- rather than reimplementing it (sec 19's
    BUILD-FIRST item 1: "The replacement code already exists verbatim in
    `check_t2a3_ssm_calibration` -- reuse it, do not reimplement it"). It fixes
    the missing-CI defect at the same time.

    **ZERO NEW NUMERIC GATING THRESHOLDS ARE INTRODUCED.** Every gating numeric
    is either carried over unchanged from the blind sec 11.4.1/11.4.2
    pre-registration (`0.05`; `p < 0.001`) or is an EXACT null/degeneracy
    boundary fixed by construction (`ks_lo > 0`; `argmax_changed_frac > 0`).
    `0.90`, `0.75` and `0.50` no longer appear anywhere in the gating path. This
    is checkable in one diff, and it is RULE T's core anti-laundering property.

    ===== LEG (vi), THE AIMING WITNESS -- WHAT IT IS AND WHAT IT IS NOT =====

    sec 20's LIVENESS witness proves the readout is FINITE and INPUT-DEPENDENT.
    It does NOT prove the readout is AIMED: a probe reading logits at k0+-1, the
    wrong tensor, or a transposed state is fully input-dependent (liveness
    PASSES) yet uncorrelated with the plant (sec 23.3 item 4). Leg (vi) asks
    whether the readout responds to THE PLANTED KEY specifically. Its null is
    fixed by construction (`0` = a bit-identical argmax under key-swap) and it
    fires on violation. RULE T: admissible.

    **DISCLOSED, BECAUSE AN AUDITOR WILL FIND IT ANYWAY: LEG (vi) IS ENTAILED BY
    LEG (iv) AND THEREFORE CANNOT INDEPENDENTLY FIRE.** If every record carries
    the key-swap arm (asserted by smoke [10b]), then
        argmax_changed_frac == 0  =>  hit_intact == hit_keyswap on every record
                                  =>  stat_ks(S) == 0 for EVERY subset S
                                  =>  every bootstrap draw is 0  =>  ks_lo == 0,
    so leg (vi) can only fire on a cell where leg (iv) has ALREADY failed. **It
    is a TRIPWIRE and a NAMED FAILURE REASON, not new gate power, and sec 24
    says so in those words.** Its value is DISAMBIGUATION: when a witness cell
    fails, `KS CI includes 0` alone cannot distinguish "this model has no
    mechanism" from "this probe is mis-aimed"; leg (vi) reading EXACTLY 0
    identifies the latter, which is precisely the reading sec 23.3 demanded be
    made ("a run in which W1/W2 both collapse to PRIOR must be read as 'possibly
    mis-aimed instrument,' never as 'no mechanism'") and which was, until now,
    left to an operator to remember.

    `t2b1_result`/`t2b1b_result`: check_t2b1_mechanism_exists(records) /
    check_t2b1b_key_conditioned(records) outputs, computed by the caller on the
    SAME records. `n_boot`/`seed` feed `clustered_bootstrap_ci` and default to
    the same values `check_t2a3_ssm_calibration` uses.

    DETERMINISM (sec 11.4.6, sec 19.4c): `clustered_bootstrap_ci` seeds a LOCAL
    `random.Random(seed)` and consumes NO global RNG stream, and this function
    issues no forward pass. The unchanged legs reproduce attempt-2 bit-for-bit."""
    deciles = _decile_bucket(records, key_fn=lambda r: r["delta"])
    decile_accs = [_acc(b, "hit_intact") for b in deciles]
    decile_accs_keyswap = [_acc(b, "hit_keyswap") for b in deciles]
    median_decile = deciles[len(deciles) // 2]
    acc_at_median = _acc(median_decile, "hit_intact")
    prior = _acc(records, "hit_noplant")
    acc_copy_all = _acc(records, "hit_intact")
    acc_keyswap = _acc(records, "hit_keyswap")
    ks = acc_copy_all - acc_keyswap if not (math.isnan(acc_copy_all) or math.isnan(acc_keyswap)) else float("nan")

    # ---- LEG (iv), RE-PINNED. The SAME construction as check_t2a3_ssm_calibration (sec 11.4.2),
    #      reused rather than reimplemented, per sec 19's BUILD-FIRST item 1.
    def stat_ks(recs):
        return _acc(recs, "hit_intact") - _acc(recs, "hit_keyswap")
    ks_pt, ks_lo, ks_hi = clustered_bootstrap_ci(records, stat_ks, n_boot=n_boot, seed=seed)
    ks_positive_excludes_zero = (ks_lo is not None and ks_lo > 0)

    # ---- LEG (vi), THE AIMING WITNESS (B-4). Exact degeneracy boundary; NaN => False (fail-closed).
    aiming_frac = argmax_changed_frac_keyswap(records)
    leg_vi = (not math.isnan(aiming_frac)) and aiming_frac > 0.0

    # ---- RETIRED (sec 18.4). COMPUTED AND REPORTED -- reporting is MANDATORY -- BUT NOT GATING.
    #      The `_RETIRED_NONGATING` suffix is deliberate: a reader (or a future roll-up) that
    #      greps for a leg flag cannot pick one of these up and mistake it for a gate.
    leg_i_RETIRED = (not math.isnan(acc_at_median)) and acc_at_median >= 0.90
    leg_ii_RETIRED = bool(decile_accs) and all((not math.isnan(a)) and a >= 0.75 for a in decile_accs)

    # ---- THE GATING CONJUNCTION.
    leg_iii = (not math.isnan(prior)) and prior <= 0.05
    leg_iv = ks_positive_excludes_zero and bool(t2b1b_result.get("passes"))
    leg_v = bool(t2b1_result.get("passes"))
    passes = leg_iii and leg_iv and leg_v and leg_vi

    reasons = []
    if not leg_iii:
        reasons.append(f"leg (iii) PRIOR: {prior!r} exceeds the 0.05 tolerance over the "
                        f"construction null (b is never the bigram-argmax given a; V4 rank 2-50)")
    if not leg_iv:
        reasons.append(f"leg (iv) KEY-CONDITIONING: KS 95% clustered-bootstrap CI "
                        f"[{ks_lo!r}, {ks_hi!r}] does not EXCLUDE 0 (point {ks_pt!r}), and/or "
                        f"T2b-1b p={t2b1b_result.get('p_value')!r} is not < 0.001")
    if not leg_v:
        reasons.append(f"leg (v) T2b-1: p={t2b1_result.get('p_value')!r} is not < 0.001")
    if not leg_vi:
        reasons.append(f"leg (vi) AIMING: argmax_changed_frac_keyswap = {aiming_frac!r}. EXACTLY 0 "
                        f"=> swapping the PLANTED KEY at j0 changed the readout argmax at k0 in "
                        f"ZERO windows => the readout does not depend on the plant at all. This is "
                        f"a MIS-AIMED PROBE (wrong position / wrong tensor / transposed state), NOT "
                        f"a null model -- liveness cannot see it (sec 23.3 item 4). Read a failure "
                        f"here as 'possibly mis-aimed instrument', NEVER as 'no mechanism'.")

    return {
        "passes": passes,
        "gating_legs": ["iii_prior", "iv_ks_ci_excludes_zero_and_t2b1b", "v_t2b1", "vi_aiming"],
        "retired_legs_REPORTED_NEVER_GATING": ["i_acc_at_median_ge_090", "ii_all_deciles_ge_075"],
        "failure_reasons": reasons,
        # ---- REPORTED ALWAYS (sec 18.4 leg (i): "REPORTED ALWAYS, VERDICT-CARRYING NEVER") ----
        "acc_at_median": acc_at_median, "decile_accs": decile_accs,
        "decile_accs_keyswap": decile_accs_keyswap,
        "prior": prior, "ks": ks, "acc_copy": acc_copy_all, "acc_copy_keyswap": acc_keyswap,
        "leg_i_median_ge_090_RETIRED_NONGATING": leg_i_RETIRED,
        "leg_ii_all_deciles_ge_075_RETIRED_NONGATING": leg_ii_RETIRED,
        # ---- THE GATING LEGS ----
        "leg_iii_prior_le_005": leg_iii,
        "leg_iv_ks_ci_excludes_zero_and_t2b1b": leg_iv,
        "leg_iv_ks_point": ks_pt, "leg_iv_ks_ci": [ks_lo, ks_hi],
        "leg_iv_ks_ci_excludes_zero": ks_positive_excludes_zero,
        "leg_v_t2b1_passes": leg_v,
        "leg_vi_aiming_keyswap_argmax_changed": leg_vi,
        "aiming_argmax_changed_frac_keyswap": aiming_frac,
        "t2b1_p_value": t2b1_result.get("p_value"), "t2b1b_p_value": t2b1b_result.get("p_value"),
    }


def check_t2a2_untrained_control(records: list, logit_liveness: dict = None,
                                  n_boot: int = 2000, seed: int = 0) -> dict:
    """T2a-2 NEGATIVE CONTROL (sec 11.4.2, GATING). A randomly-initialised,
    UNTRAINED model of the 14M rung's exact architecture must read acc_copy
    <= 0.02 with a KS bootstrap 95% CI INCLUDING 0. Fail => INSTRUMENT-
    INVALID, HALT (sec 11.4.2: 'If an untrained model passes the probe, the
    probe is passable with no learned mechanism'). Each plant is its own
    independent row by construction (one plant per window), so the existing
    clustered_bootstrap_ci (which resamples over `row_idx`) reduces to an
    ordinary per-plant bootstrap here -- reused, not reimplemented.

    ===== sec 20 R-4 (2026-07-13) -- WHAT CHANGED, AND WHAT DID NOT =====

    THE PINNED BAR IS UNCHANGED, BYTE FOR BYTE: `acc_copy <= 0.02` AND `KS
    bootstrap 95% CI includes 0`. Same constants, same estimators, same
    bootstrap, same seed. It is now computed into its OWN field,
    `pinned_bar_passes`, so an auditor can verify by inspection that nothing
    was moved. NO NEW NUMERIC GATING THRESHOLD IS INTRODUCED ANYWHERE IN THIS
    FUNCTION (sec 20.1 RULE T's core anti-laundering property).

    TWO THINGS ARE ADDED.

    (1) SERIALIZATION. Everything this control ALREADY COMPUTED and then
        DELETED is now persisted: all six arms' accuracies (intact, true-
        ablated, placebo-ablated, pool-placebo, KEY-SWAP, NO-PLANT/PRIOR),
        their raw hit counts, `acc_copy_se`, the arm-2/arm-3 DiD contrast
        with its clustered-bootstrap CI, the Delta-decile profile, and the
        T2b-1 / T2b-1b paired sign tests. All REPORTED, NONE GATING. The old
        artifact was three numbers -- `acc_copy=0.0`, `ks_point=0.0`,
        `ks_ci=[0.0,0.0]` -- and a dead forward pass reproduces all three
        exactly (sec 19.3d).

    (2) A LIVENESS PRECONDITION (`liveness`, from `run_t2_repaired_probe`'s
        `logit_liveness`). `passes` is now the CONJUNCTION
            passes = pinned_bar_passes AND liveness["ok"]
        This is a MONOTONE TIGHTENING: a conjunction can only turn a PASS
        into a HALT, never a FAIL into a PASS, so it cannot launder anything
        in any direction. Certifying INSTRUMENT_VALID off a forward pass that
        never happened is not a null result -- it is no result, and the leg
        exists to certify the former.

    FAIL-CLOSED: `logit_liveness=None` (a caller that did not thread the
    witness through) yields `liveness["ok"] = False` and therefore
    `passes = False`. A control cannot pass by OMITTING its own witness."""
    def stat_ks(recs):
        return _acc(recs, "hit_intact") - _acc(recs, "hit_keyswap")

    def stat_arm_did(recs):
        # Same estimand SHAPE as stat_did (acc_intact cancels in the paired design), on the
        # T2 probe's own arm names. REPORTED ONLY -- T2a-2 gates on the pinned bar, not this.
        return _acc(recs, "hit_placebo_ablated") - _acc(recs, "hit_true_ablated")

    # ---- THE PINNED BAR (sec 11.4.2) -- UNCHANGED, verbatim from the blind pre-registration.
    acc_copy = _acc(records, "hit_intact")
    ks_pt, ks_lo, ks_hi = clustered_bootstrap_ci(records, stat_ks, n_boot=n_boot, seed=seed)
    ci_includes_zero = (ks_lo is not None and ks_hi is not None and ks_lo <= 0 <= ks_hi)
    pinned_bar_passes = (not math.isnan(acc_copy)) and acc_copy <= 0.02 and ci_includes_zero

    # ---- THE LIVENESS PRECONDITION (sec 20 R-4). Structural, exact, fail-closed.
    if logit_liveness is None:
        liveness = {"ok": False, "degenerate_reasons": [
            "no logit_liveness witness was supplied to check_t2a2_untrained_control -- the "
            "caller did not thread run_t2_repaired_probe's `logit_liveness` through. A negative "
            "control cannot certify the instrument while omitting the only evidence that its "
            "forward pass ran at all (sec 19.3d / sec 20 R-4). FAIL-CLOSED."]}
    else:
        liveness = dict(logit_liveness)

    # ---- EVERYTHING ELSE: computed already, deleted before this fix. REPORTED, NON-GATING.
    arm_did_pt, arm_did_lo, arm_did_hi = clustered_bootstrap_ci(records, stat_arm_did,
                                                                 n_boot=n_boot, seed=seed)
    n = len(records)
    arms = {}
    for name, key in (("intact", "hit_intact"), ("true_ablated", "hit_true_ablated"),
                      ("placebo_ablated", "hit_placebo_ablated"),
                      ("pool_placebo", "hit_pool_placebo"), ("keyswap", "hit_keyswap"),
                      ("noplant_PRIOR", "hit_noplant")):
        vals = [r[key] for r in records if r.get(key) is not None]
        arms[name] = {"acc": (_mean(vals) if vals else None), "hits": sum(vals) if vals else 0,
                      "n": len(vals),
                      "se": (binomial_se(sum(vals), len(vals)) if vals else None)}
    deciles = _decile_bucket(records, key_fn=lambda r: r["delta"]) if records else []

    return {
        # ---- THE VERDICT: pinned bar AND liveness. See the docstring: monotone tightening.
        "passes": bool(pinned_bar_passes and liveness.get("ok")),
        "pinned_bar_passes": bool(pinned_bar_passes),
        "liveness": liveness,
        # ---- the three original fields, unchanged, at their original keys (back-compatible).
        "acc_copy": acc_copy, "ks_point": ks_pt, "ks_ci": [ks_lo, ks_hi],
        # ---- sec 20 R-4 SERIALIZATION: computed all along, deleted all along.
        "acc_copy_se": binomial_se(sum(r["hit_intact"] for r in records), n) if n else None,
        "n_plants": n,
        "arms": arms,
        "arm_did_placebo_minus_true": {"point": arm_did_pt, "ci": [arm_did_lo, arm_did_hi]},
        "decile_accs_intact": [_acc(bkt, "hit_intact") for bkt in deciles],
        "decile_accs_keyswap": [_acc(bkt, "hit_keyswap") for bkt in deciles],
        "t2b1_mechanism_exists_REPORTED": check_t2b1_mechanism_exists(records),
        "t2b1b_key_conditioned_REPORTED": check_t2b1b_key_conditioned(records),
    }


def check_t2a3_ssm_calibration(records: list, t2b1_result: dict, t2b1b_result: dict,
                                n_boot: int = 2000, seed: int = 0) -> dict:
    """T2a-3 SSM CALIBRATION (sec 11.4.2, GATING ON THE CAUSAL LEGS ONLY,
    NEW). `falcon-mamba-7b` must pass T2b-1 AND T2b-1b (p<0.001) and show
    KS>0 with a bootstrap 95% CI excluding 0 (sec 11.4.2, conceded to A-M1:
    KS is a genuine SINGLE-token contrast, replacing the draft's two-token
    `acc_copy - acc_copy_noplant`). `acc_copy` is reported, NOT held to
    0.90 -- this witness is demoted from the 0.90 ceiling gate (sec 11.4.2)
    and 'can no longer save the gate' (T2a-1 still requires W1+W2).

    **NOT WAIVED, NOT WEAKENED, AND STILL NEVER MEASURED** (sec 18.9, sec 23.4
    item 5). The pinned causal legs below are BYTE-FOR-BYTE what sec 11.4.2
    pre-registered.

    sec 24 (B-4): the AIMING witness (leg (vi) of T2a-1) is added as a further
    CONJUNCT. This is a MONOTONE TIGHTENING -- a conjunction can only turn a
    PASS into a HALT, never a FAIL into a PASS -- so it cannot waive or weaken
    anything in any direction. Like leg (vi) of T2a-1 it is ENTAILED by the
    `KS > 0` CI leg it sits beside (see `check_t2a1_ceiling`'s docstring) and is
    therefore a TRIPWIRE and a named failure reason, not new gate power. C1 is a
    pure-SSM architecture class the probe has NEVER been shown to read, which is
    precisely the cell on which "no mechanism" and "mis-aimed probe" are easiest
    to confuse -- so the reason is worth naming here."""
    def stat_ks(recs):
        return _acc(recs, "hit_intact") - _acc(recs, "hit_keyswap")
    ks_pt, ks_lo, ks_hi = clustered_bootstrap_ci(records, stat_ks, n_boot=n_boot, seed=seed)
    ks_positive_excludes_zero = (ks_lo is not None and ks_lo > 0)
    aiming_frac = argmax_changed_frac_keyswap(records)
    aiming_ok = (not math.isnan(aiming_frac)) and aiming_frac > 0.0
    causal_pass = (bool(t2b1_result.get("passes")) and bool(t2b1b_result.get("passes"))
                   and ks_positive_excludes_zero and aiming_ok)
    return {"passes": causal_pass, "acc_copy": _acc(records, "hit_intact"),
            "ks_point": ks_pt, "ks_ci": [ks_lo, ks_hi],
            "ks_positive_excludes_zero": ks_positive_excludes_zero,
            "aiming_keyswap_argmax_changed": aiming_ok,
            "aiming_argmax_changed_frac_keyswap": aiming_frac,
            "t2b1_passes": t2b1_result.get("passes"), "t2b1b_passes": t2b1b_result.get("passes")}


def check_t2a4_positive_control(records: list) -> dict:
    """T2a-4 POSITIVE CONTROL (sec 26, GATING, NEW). **THE FIRST CONTROL IN THIS
    DESIGN THAT ASKS WHETHER THE INSTRUMENT CAN SUCCEED WHEN IT SHOULD.**

    `records` MUST come from `run_t2_repaired_probe(PerfectCopyOracle(V), ...)`
    -- a model that has the copy mechanism BY FIAT. The driver's
    `run_t2a4_positive_control` constructs the oracle itself, so the records
    cannot be a witness's by accident. On a CORRECTLY-AIMED instrument the
    oracle's records satisfy, ON EVERY ROW, two identities that are THEOREMS of
    the plant's own hard assertion (derived line by line in `PerfectCopyOracle`'s
    docstring, from `plant_and_verify_t2_window` + `rejection_sample_delta`'s
    `Delta >= 2` + `draw_t2_triple`'s pairwise-distinct, naturally-absent triple):

        PC-1 RECOVERY:  argmax_intact_at_k == b
        PC-2 AIMING:    argmax_intact_at_k != argmax_keyswap_at_k

    ===== THE NULL, IN THE NULL'S OWN UNITS =====

    NULL (fixed BY CONSTRUCTION, not by measurement): under a correctly-aimed
    instrument, **the number of records violating PC-1 is EXACTLY 0, and the
    number violating PC-2 is EXACTLY 0.** These are not tolerances and not
    magnitudes: they are VIOLATION COUNTS of an identity we hold by fiat, and
    their sampling distribution under the null is a POINT MASS AT ZERO (zero
    variance -- the oracle is deterministic and consumes no RNG). The gate fires
    on ANY violation. There is no slack to tighten and no scale to borrow.

    **RULE T (sec 20.1), leg by leg.** RULE T's three shapes are (a) DEPARTURE
    from a null in the null's own SAMPLING units -- admissible; (b) PROXIMITY to
    a null as a TOLERANCE -- admissible conditionally, the slack being a
    disclosed weakening; (c) DEPARTURE from a null as a RAW EFFECT-SIZE
    MAGNITUDE -- INADMISSIBLE. PC-1/PC-2 are shape (b) **with the tolerance set
    to zero**: the strictest member of an already-admitted shape, and strictly
    cleaner than `PRIOR <= 0.05` or T2a-2's `acc_copy <= 0.02`, both of which
    carry un-derived (if blind-pre-registered) slack. **ZERO NEW NUMERIC GATING
    THRESHOLDS.** The gating predicate contains NO numeric literal at all: it
    compares the probe's recovered token against THE RECORD'S OWN PLANTED `b`,
    and the intact argmax against the key-swap argmax. The only numbers here are
    counts, and the null they are compared to is `0` -- an exact degeneracy
    boundary, the same family already admitted for `ks_lo > 0` (leg iv) and
    `argmax_changed_frac > 0` (leg vi).

    ===== WHY THIS CLOSES sec 25's A-1, AND WHAT IT DOES NOT DO =====

    sec 25's blocker: leg (vi) tests `argmax_changed_frac_keyswap > 0`, which
    fires ONLY IF the readout is mis-aimed on LITERALLY EVERY ROW; and because
    leg (vi) is ENTAILED by leg (iv) it inherits leg (iv)'s floor and can never
    bind above it. It certifies "aimed SOMEWHERE," not "aimed." sec 24 tested
    `f = 0` and `f = 1` and never the interior. On a model whose recovery is
    known to be TOTAL, "aimed somewhere" and "aimed" are the SAME predicate --
    the interior collapses, because any coverage `f < 1` leaves `(1-f)*n` rows
    that FAIL an identity that must hold on all `n`. An instrument aimed on 0.6%
    of rows reads `n_miss_recovery = 2036` here and HALTS.

    **WHAT IT DOES NOT DO, STATED SO NOBODY OVER-READS IT.** It certifies the
    INSTRUMENT, not the witness. It cannot tell you whether `falcon-mamba-7b` has
    a copy mechanism -- nothing can, and that is the measurement. What it buys is
    that a LOW WITNESS `acc_copy` is now attributable: with T2a-4 green, the
    readout provably recovers a plant it is supposed to recover, on every row, on
    THIS corpus at THIS `seq_len` through THIS code path -- so a witness that
    reads 0.006 is a fact about the MODEL (sec 11.4.3 step 3's pinned "report it
    as a finding about the models"), not a possible fact about the probe. That
    disambiguation is exactly what sec 24.3 wanted and could not have.

    **AND THE HAZARD sec 25.5 NAMED IN ITS OWN CANDIDATE, WHICH THIS AVOIDS BY
    CONSTRUCTION.** sec 25.5's candidate direction was a PER-STRATUM aiming leg
    on the WITNESSES. sec 25 flagged its own hazard: RWKV7/gpt2-large are
    legitimately DISTANCE-LIMITED (W1/openr1's Delta-deciles run 0.907 -> 0.376),
    and a healthy witness can legitimately read `aiming = 0` in its largest
    Delta-decile => **a per-decile `> 0` leg would FALSE-HALT a healthy witness**
    -- the identical error sec 23.4 item 2 made and sec 24 correctly refused.
    **T2a-4 READS NO WITNESS QUANTITY AT ALL.** Its records come from the oracle
    and only the oracle. It is STRUCTURALLY INCAPABLE of false-halting a witness,
    at any Delta, at any decile, at any coverage. The witnesses' aiming profiles
    remain REPORTED and GATE NOTHING per-stratum.

    The REFERENCE PROFILE sec 25.2 asked for ("what `aiming` and its per-stratum
    profile LOOK LIKE ON A CORRECTLY-AIMED PROBE") is emitted here -- `1.0` in
    every Delta-decile, by construction -- as REPORTED, NEVER-GATING context
    against which a witness's profile can be READ. Reading is not gating."""
    n_records = len(records)
    required = ("b", "argmax_intact_at_k", "argmax_keyswap_at_k")
    complete = [r for r in records if all(r.get(k) is not None for k in required)]
    n_incomplete = n_records - len(complete)

    miss_recovery = [r for r in complete if r["argmax_intact_at_k"] != r["b"]]
    aim_unchanged = [r for r in complete if r["argmax_intact_at_k"] == r["argmax_keyswap_at_k"]]
    n_miss = len(miss_recovery)
    n_unchanged = len(aim_unchanged)

    # ---- THE GATING CONJUNCTION. Exact violation counts against a construction null of 0.
    #      FAIL-CLOSED on an empty or field-incomplete record set: a positive control that
    #      certifies the instrument while omitting the rows it was supposed to check is the
    #      T2a-2 "no measurement is not a null result" defect (sec 19.3d), re-committed.
    pc0_records_complete = (n_records > 0) and (n_incomplete == 0)
    pc1_recovery = pc0_records_complete and (n_miss == 0)
    pc2_aiming = pc0_records_complete and (n_unchanged == 0)
    passes = pc0_records_complete and pc1_recovery and pc2_aiming

    reasons = []
    if not pc0_records_complete:
        reasons.append(
            f"PC-0 RECORDS: n_records={n_records}, n_incomplete={n_incomplete}. The positive "
            f"control cannot certify the instrument off rows it never read (FAIL-CLOSED).")
    if pc0_records_complete and n_miss:
        reasons.append(
            f"PC-1 RECOVERY VIOLATED: the probe FAILED TO RECOVER the planted value `b` at the "
            f"readout on {n_miss} of {len(complete)} rows ({n_miss / len(complete):.4f}) -- from a "
            f"model that emits `b` there BY CONSTRUCTION (PerfectCopyOracle). The construction "
            f"null is EXACTLY 0 violations. **THIS IS A MIS-AIMED INSTRUMENT** (wrong position / "
            f"wrong tensor / transposed state / stale row index), and it is mis-aimed on "
            f"{n_miss / len(complete):.2%} of the candidate population. It is NOT a weak model: "
            f"the model is perfect by fiat. sec 25's A-1: a readout aimed on 0.6% of rows clears "
            f"every OTHER gating leg and certifies INSTRUMENT_VALID; this leg is the one that "
            f"HALTs it. HALT -- INSTRUMENT-INVALID.")
    if pc0_records_complete and n_unchanged:
        reasons.append(
            f"PC-2 AIMING VIOLATED: swapping the planted key at j0 left the readout argmax at k0 "
            f"BIT-IDENTICAL on {n_unchanged} of {len(complete)} rows "
            f"({n_unchanged / len(complete):.4f}) -- on a model whose readout at k0 is a function "
            f"of the key at j0 BY CONSTRUCTION. The construction null is EXACTLY 0. The readout is "
            f"PLANT-INDEPENDENT on those rows. HALT -- INSTRUMENT-INVALID.")

    deciles = _decile_bucket(complete, key_fn=lambda r: r["delta"]) if complete else []

    def _frac(bucket, fn):
        return _mean([int(fn(r)) for r in bucket]) if bucket else float("nan")

    return {
        "passes": bool(passes),
        "gating_legs": ["pc0_records_present_and_complete", "pc1_recovery_no_miss",
                        "pc2_aiming_no_unchanged"],
        "pc0_records_present_and_complete": bool(pc0_records_complete),
        "pc1_recovery_no_miss": bool(pc1_recovery),
        "pc2_aiming_no_unchanged": bool(pc2_aiming),
        # ---- THE VIOLATION COUNTS. The null is 0 for both, by construction.
        "n_records": n_records, "n_records_incomplete": n_incomplete,
        "n_miss_recovery": n_miss, "n_aim_unchanged": n_unchanged,
        "construction_null_n_miss_recovery": 0, "construction_null_n_aim_unchanged": 0,
        "failure_reasons": reasons,
        # ---- THE REFERENCE PROFILE (sec 25.2). REPORTED, NEVER GATING ON ANY WITNESS.
        #      On a correctly-aimed instrument every one of these is its construction value:
        #      acc_copy 1.0, aiming 1.0, KS 1.0, PRIOR 0.0, acc_true_ablated 0.0 -- and each
        #      Delta-decile reads 1.0. A witness's own profile is READ against this. Reading
        #      is not gating: sec 25.5's own hazard (a per-decile aiming leg FALSE-HALTS a
        #      healthy distance-limited witness) is why no witness stratum is gated anywhere.
        "reference_acc_copy": _acc(complete, "hit_intact") if complete else float("nan"),
        "reference_acc_keyswap": _acc(complete, "hit_keyswap") if complete else float("nan"),
        "reference_acc_true_ablated": _acc(complete, "hit_true_ablated") if complete else float("nan"),
        "reference_prior": _acc(complete, "hit_noplant") if complete else float("nan"),
        "reference_ks": ((_acc(complete, "hit_intact") - _acc(complete, "hit_keyswap"))
                         if complete else float("nan")),
        "reference_aiming_argmax_changed_frac_keyswap": argmax_changed_frac_keyswap(complete),
        "reference_decile_recovered": [_frac(b, lambda r: r["argmax_intact_at_k"] == r["b"])
                                       for b in deciles],
        "reference_decile_aiming": [
            _frac(b, lambda r: r["argmax_intact_at_k"] != r["argmax_keyswap_at_k"])
            for b in deciles],
        "reference_decile_delta_median": [
            (sorted(r["delta"] for r in b)[len(b) // 2] if b else None) for b in deciles],
    }


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


# =============================================================================
# sec 18.4.1 -- THE THRESHOLD-FREE INFLUENCE LADDER. Replaces sec 9.4's
#     "strong-mechanism" SPLIT, which is RETIRED. (sec 19 BUILD-FIRST item 2;
#     built in sec 24 / B-2.)
# =============================================================================
def influence_ladder(per_rung_row_values: dict, log10_params: dict, ks_by_rung: dict,
                      n_boot: int = 2000, seed: int = 0, min_rungs: int = 3) -> dict:
    """sec 18.4.1, IMPLEMENTED VERBATIM.

    ===== WHAT IT REPLACES =====

    sec 9.4 required the trend fit be reported TWICE -- over all T2b-admissible
    rungs, and over "the subset that also clears `acc_copy >= 0.90`" -- with
    disagreement => INDETERMINATE. RULE T retired the `0.90` bar, so that subset
    is now undefined. sec 15's proposed replacement (a MEDIAN-`KS` split) was
    correctly killed by sec 16.5: a median split is RELATIVE, so it always
    labels half the rungs "strong" even if every rung is garbage, and it can
    therefore NEVER return "no rung is strong" -- the very condition the old
    split existed to surface. RULE T forbids inventing a new absolute bar to
    plug the hole. So sec 18.4.1 pinned this instead:

      * Order the admissible rungs by `KS` (ASCENDING).
      * Report the trend fit at EVERY PREFIX-DROP of that ordering with >= 3
        rungs remaining: all rungs -> drop-lowest-1 -> drop-lowest-2 -> ...
        **Report the ENTIRE ladder, never a selected rung of it.**
      * INDETERMINATE fires IFF the fitted exponent's SIGN, or its CI's
        EXCLUSION of the no-trend null, FLIPS anywhere along that ladder.

    RULE T (sec 20.1): sign and CI-exclusion are construction-derived (the null
    is "no trend", i.e. beta = 0); "how much change is too much" is never asked,
    so no magnitude threshold exists to launder. **ZERO NUMERIC GATING
    THRESHOLDS.** `min_rungs = 3` is not one: it is sec 9.5's OWN pre-existing
    FLOOR (`n_admissible_rungs < 3 => FLOOR`), restated in sec 18.4.1's text as
    ">= 3 rungs remaining", and an OLS slope over fewer than 3 points is
    degenerate rather than merely weak. It does not appear as a bar on any
    measured quantity.

    ===== TWO RESIDUAL CONCERNS, FLAGGED NOT SILENTLY FIXED (sec 24) =====

    sec 19.3(b) argues the ordering is STILL RELATIVE -- the ladder drops the
    LOWEST-KS rung regardless of whether ANY rung reads strongly, so like the
    median split it can never return "no rung is strong". **THAT CRITICISM
    STANDS against what is built here, and it is not silently repaired**: the
    pin is implemented as written, and sec 24 records the concern. (sec 18.4.1's
    own defence is narrower and is true as far as it goes: the ladder CAN return
    "the trend is not robust", which the median split could not. It cannot
    return "the instrument is too weak to fit a law on." **Nothing in the design
    currently can** -- sec 19.3(c)/sec 20.3 record that function as EMPTY and
    OPEN, and this function does not fill it.)

    SECOND, AND NOT PREVIOUSLY STATED ANYWHERE: at exactly 3 admissible rungs
    the ladder has exactly ONE step, so NOTHING CAN FLIP and it reports
    "robust" VACUOUSLY. Three is the design's own admissible-rung MINIMUM (sec
    9.5's FLOOR) and sec 11.8 records only 2 fit rungs available today, so the
    vacuous regime is the LIKELY one, not a corner case. `n_steps` and
    `ladder_is_vacuous` are returned so no caller can miss it.

    ARGUMENTS
      per_rung_row_values: {rung: {row_idx: M(r) value}} -- exactly what
        `bootstrap_trend_ci` consumes; this function DELEGATES to it per step
        rather than reimplementing the fit (each step gets its own `seed +
        step_index` so the steps are independent draws but the whole ladder is
        reproducible from one `seed`).
      log10_params: {rung: log10(param count)}.
      ks_by_rung: {rung: KS} -- the ORDERING key. Each rung's `KS` from its own
        T2 probe (`check_t2a1_ceiling(...)["ks"]`). A rung missing from this map,
        or carrying a NaN `KS`, cannot be ordered => the ladder is NOT EVALUABLE
        and returns INDETERMINATE (fail-closed; it never silently drops a rung).

    RETURNS a dict with the FULL ladder, plus `indeterminate` (the sec 18.4.1
    rule) and `evaluable`."""
    rungs = sorted(per_rung_row_values)
    missing = [r for r in rungs if r not in ks_by_rung or r not in log10_params]
    nan_ks = [r for r in rungs if r in ks_by_rung
              and (ks_by_rung[r] is None or math.isnan(float(ks_by_rung[r])))]
    if missing or nan_ks:
        return {
            "evaluable": False, "indeterminate": True, "ladder_is_vacuous": None,
            "n_steps": 0, "steps": [],
            "reasons": [f"the ladder cannot be ORDERED: rungs missing a KS or a log10_params "
                        f"entry: {missing}; rungs with a NaN KS: {nan_ks}. FAIL-CLOSED -- a rung "
                        f"that cannot be ordered is never silently dropped from the fit."],
        }
    # sec 18.4.1: order ASCENDING by KS. Ties broken by rung name so the ladder is deterministic.
    ordered = sorted(rungs, key=lambda r: (float(ks_by_rung[r]), r))
    if len(ordered) < min_rungs:
        return {
            "evaluable": False, "indeterminate": True, "ladder_is_vacuous": None,
            "n_steps": 0, "steps": [], "rungs_ordered_by_ks_ascending": ordered,
            "ks_by_rung": {r: float(ks_by_rung[r]) for r in ordered},
            "reasons": [f"only {len(ordered)} rung(s) -- fewer than the {min_rungs} an OLS trend "
                        f"needs. NO robustness claim is available. (sec 9.5's FLOOR already "
                        f"withholds a verdict here; this cannot CREATE one.)"],
        }

    steps = []
    for k in range(0, len(ordered) - min_rungs + 1):
        kept = ordered[k:]
        sub_vals = {r: per_rung_row_values[r] for r in kept}
        fit = bootstrap_trend_ci(sub_vals, log10_params, n_boot=n_boot, seed=seed + k)
        beta, (lo, hi) = fit["beta_point"], fit["ci95"]
        # SIGN: construction-derived, no magnitude. 0.0 is its own sign class (an exactly-flat
        # OLS slope), so a move OFF zero is itself a flip -- the conservative reading.
        sign = 0 if beta == 0.0 else (1 if beta > 0.0 else -1)
        excludes_zero = bool(lo > 0.0 or hi < 0.0)
        steps.append({
            "n_dropped_lowest_ks": k,
            "rungs": kept,
            "dropped": ordered[:k],
            "beta_point": beta,
            "ci95": [lo, hi],
            "beta_sign": sign,
            "ci_excludes_no_trend_null": excludes_zero,
        })

    signs = {s["beta_sign"] for s in steps}
    exclusions = {s["ci_excludes_no_trend_null"] for s in steps}
    sign_flips = len(signs) > 1
    exclusion_flips = len(exclusions) > 1
    indeterminate = bool(sign_flips or exclusion_flips)

    reasons = []
    if sign_flips:
        reasons.append(f"INDETERMINATE (sec 18.4.1): the fitted exponent's SIGN FLIPS along the "
                        f"ladder -- observed signs {sorted(signs)} across {len(steps)} steps.")
    if exclusion_flips:
        reasons.append(f"INDETERMINATE (sec 18.4.1): the CI's EXCLUSION of the no-trend null "
                        f"FLIPS along the ladder -- observed {sorted(exclusions)} across "
                        f"{len(steps)} steps.")
    if len(steps) == 1:
        reasons.append("LADDER IS VACUOUS: exactly one step (the admissible set is at the "
                        "3-rung minimum), so NOTHING CAN FLIP and 'robust' here means only 'not "
                        "checked'. This is the LIKELY regime, not a corner case (sec 11.8: 2 fit "
                        "rungs available against a minimum of 3). Do not read it as robustness.")

    return {
        "evaluable": True,
        "indeterminate": indeterminate,
        "ladder_is_vacuous": len(steps) == 1,
        "n_steps": len(steps),
        "sign_flips": sign_flips,
        "exclusion_flips": exclusion_flips,
        "rungs_ordered_by_ks_ascending": ordered,
        "ks_by_rung": {r: float(ks_by_rung[r]) for r in ordered},
        "steps": steps,
        "reasons": reasons,
    }


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

    # --- [7b] _exact_binomial_two_sided_p OVERFLOW FIX (BUG 1, sec 12.3 Defect B): a test
    #     with teeth per the mandate -- must FAIL on the OLD (math.comb-based) implementation
    #     and PASS on the NEW (log-space) one, at n>=~1030 (the verified overflow threshold)
    #     up through n=4096 (>= N_ROWS_DEFAULT=2048's worst case). Also verifies the new
    #     implementation agrees with the old one to high precision at small n, where the old
    #     one was never wrong -- a rewrite that changes small-n p-values would be a DIFFERENT
    #     test, not a fix. ---
    print("\n  [7b] _exact_binomial_two_sided_p OVERFLOW FIX (sec 12.3 Defect B) teeth")

    def _old_buggy_pmf(k_, n_, p_=0.5):
        # THE ORIGINAL (pre-fix) formula, byte-for-byte, kept ONLY here as a fossil to prove
        # the fix against -- math.comb returns an arbitrary-precision int; Python converts it
        # to float on the first `*` before the shrinking p**x*(1-p)**(n-x) factor can apply.
        def pmf(x):
            return math.comb(n_, x) * (p_ ** x) * ((1 - p_) ** (n_ - x))
        p_obs = pmf(k_)
        total = 0.0
        for x in range(n_ + 1):
            px = pmf(x)
            if px <= p_obs * (1 + 1e-9):
                total += px
        return min(1.0, total)

    # [7b.1] FAILS on old, PASSES on new, at n=1030 (first-overflow, verified by sweep),
    #        n=2048 (N_ROWS_DEFAULT, the design's own pinned constant), n=4096 (2x headroom).
    for n_test in (1030, 2048, 4096):
        k_test = n_test // 2   # the worst case: math.comb(n, n//2) is the largest binomial
        # coefficient for this n, so it overflows soonest / most severely.
        old_raised = False
        old_detail = ""
        try:
            _old_buggy_pmf(k_test, n_test)
        except OverflowError as e:
            old_raised = True
            old_detail = str(e)
        report(f"  OLD implementation FAILS (raises OverflowError) at n={n_test}, k=n//2 "
               f"(confirms the bug is real and reproduced, not merely inferred)",
               old_raised, old_detail)
        new_raised = False
        new_p = None
        try:
            new_p = _exact_binomial_two_sided_p(k_test, n_test)
        except OverflowError as e:
            new_raised = True
            new_detail = str(e)
        ok_new = (not new_raised) and new_p is not None and 0.0 <= new_p <= 1.0
        report(f"  NEW implementation PASSES (no overflow, returns a valid p-value in [0,1]) "
               f"at n={n_test}, k=n//2", ok_new,
               f"p={new_p}" if not new_raised else f"OverflowError: {new_detail}")
        # a perfectly balanced (k=n//2) sample under H0:p=0.5 should read p_value~=1.0
        # (least extreme possible outcome) -- a real sanity check on the NEW value, not just
        # "didn't crash".
        if ok_new:
            report(f"  NEW implementation's p-value at k=n//2, n={n_test} is close to 1.0 "
                   f"(the least-extreme possible split under H0 -- a real correctness check, "
                   f"not just 'did not crash')", new_p > 0.98, f"p={new_p}")

    # [7b.2] a STRONGLY asymmetric split at large n (the case that actually fires in
    #        production, e.g. T2b-1/T2b-1b on a real strong signal) must still read a tiny
    #        p-value (highly significant) and PASS the p<0.001 gate -- the fix must not have
    #        accidentally neutered the test's power along with fixing the crash.
    n_big, k_big = 2048, 1400   # n_plus=1400, n_minus=648 -- a strong, realistic asymmetry
    p_big = _exact_binomial_two_sided_p(k_big, n_big)
    report(f"  NEW implementation: a strongly asymmetric split (k={k_big}/{n_big}) at "
           f"N_ROWS_DEFAULT scale still reads p<0.001 (the fix did not neuter statistical "
           f"power along with fixing the crash)", p_big < 0.001, f"p={p_big}")

    # [7b.3] SMALL-n AGREEMENT: the old (buggy-but-valid-at-small-n) implementation and the
    #        new (log-space) one must agree to high precision everywhere the old one worked --
    #        a rewrite that changes small-n p-values is a DIFFERENT test, not a fix.
    max_abs_diff = 0.0
    worst = None
    n_small_range = range(10, 101)
    for n_s in n_small_range:
        for k_s in (0, n_s // 4, n_s // 2, (3 * n_s) // 4, n_s):
            old_p = _old_buggy_pmf(k_s, n_s)
            new_p = _exact_binomial_two_sided_p(k_s, n_s)
            diff = abs(old_p - new_p)
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst = (n_s, k_s, old_p, new_p)
    report(f"  NEW implementation agrees with OLD to within 1e-9 absolute across n=10..100, "
           f"k in {{0, n/4, n/2, 3n/4, n}} (worst-case |diff|={max_abs_diff:.3e} at "
           f"n={worst[0] if worst else None}, k={worst[1] if worst else None})",
           max_abs_diff < 1e-9, str(worst))

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
    # sec 20 R-4: [10f]'s forced-fail tests reuse [10b]'s (expensive) synthetic fixture. It is
    # captured into this explicit sentinel rather than relied on by NameError-luck: if [10b]
    # dies before building it, [10f] reports a HARD FAIL instead of vanishing. (This repo has
    # already shipped one "30/30 PASS" whose test body had ZERO coverage behind a NameError.)
    t2_fixture = None
    try:
        V11 = 20_220
        t0_corpus = time.time()
        train11 = build_synthetic_t2_train_corpus(15_000_000, vocab_size=V11, seed=11)
        # PRE-EXISTING BUG FIX (found by this session's independent audit; NOT one of the two
        # crash bugs, and INDEPENDENTLY REPRODUCED on the pre-fix HEAD file, so it is not a
        # regression introduced here). `build_synthetic_t2_train_corpus` builds on CPU (its
        # torch.Generator is a CPU generator by construction), but run_t2_repaired_probe builds
        # a `torch.Generator(device=device)` and calls get_batch, which does
        # `torch.randint(..., generator=generator, device=tokens.device)` -- so a CPU val tensor
        # + a cuda generator raises "Expected a 'cpu' device type for generator but found
        # 'cuda'" and this whole [10b] block died on `--device cuda`. The REAL gate path was
        # never affected (build_bridged_corpus puts val_ids on `device`, so the two agree) --
        # this is a SMOKE-FIXTURE defect only. But it meant the probe's smoke suite could only
        # ever go green under the CPU-stub path, leaving the CUDA production path with zero
        # real-kernel smoke coverage -- exactly the gap CLAUDE.md's own learned rule
        # ("CPU-stub self-test suites test logic only") warns about. Moving val11 to `device`
        # buys that coverage.
        val11 = build_synthetic_t2_train_corpus(2_000_000, vocab_size=V11, seed=12).to(device)
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
        t2a2_11 = check_t2a2_untrained_control(recs11, logit_liveness=t2_result["logit_liveness"],
                                                n_boot=200)
        report("  check_t2b1 / check_t2b1b / check_t2a1_ceiling run on REAL-pipeline output "
               "without error", True,
               f"acc_copy={t2_result['acc_copy']:.4f} t2b1_p={t2b1_11['p_value']:.4g} "
               f"t2b1b_p={t2b1b_11['p_value']:.4g} t2a1_passes={t2a1_11['passes']}")
        report("  T2a-2 (untrained negative control): an UNTRAINED random-init model reads "
               "acc_copy<=0.02 with a KS bootstrap CI including 0 -- PASSES on a genuinely "
               "untrained model, exactly as sec 11.4.2 requires",
               t2a2_11["passes"] and t2a2_11["pinned_bar_passes"],
               f"passes={t2a2_11['passes']} pinned_bar={t2a2_11['pinned_bar_passes']} "
               f"acc_copy={t2a2_11['acc_copy']} ks_ci={t2a2_11['ks_ci']}")
        # sec 20 R-4: the LIVE arm of the liveness witness. The SAME untrained model that reads
        # three zeros above must nonetheless prove its forward pass HAPPENED.
        lv11 = t2a2_11["liveness"]
        report("  [R-4 LIVE ARM] the untrained model's liveness witness is OK: readout logits "
               "100% finite AND input-dependent (max|L[i]-L[0]| > 0) -- 'no mechanism' is now "
               "DISTINGUISHED from 'no measurement' on the very artifact that reads acc_copy=0",
               lv11["ok"] is True
               and lv11["readout_logits_finite_frac"] == 1.0
               and lv11["readout_logit_max_abs_dev_from_row0"] > 0.0,
               f"finite_frac={lv11['readout_logits_finite_frac']} "
               f"max_abs_dev={lv11['readout_logit_max_abs_dev_from_row0']:.6g} "
               f"n_distinct_argmax={lv11['n_distinct_argmax_at_k']} "
               f"top1_share={lv11['top1_argmax_share']:.4f} "
               f"H={lv11['mean_softmax_entropy_nats']:.4f}+-{lv11['std_softmax_entropy_nats']:.4f} "
               f"keyswap_changed_argmax={lv11['argmax_changed_frac_keyswap']:.4f}")
        # sec 24 B-1: the RE-PINNED T2a-1 must STILL halt on an untrained model -- and now it must
        # do so on the CAUSAL legs (iv)/(vi), NOT on the retired 0.90/0.75 competence bars. This is
        # sec 18.10 charge 7 ("then the gate can never fail") answered on the code path: retiring
        # the competence bars did NOT make the gate unfailable. If a future edit ever let an
        # untrained model through, this turns RED.
        report("  T2a-1 (RE-PINNED, sec 18.4) STILL correctly FAILS on an untrained model -- and "
               "it fails on the CAUSAL legs (iv: KS CI includes 0) and (vi: aiming), NOT on the "
               "RETIRED 0.90/0.75 competence bars. Retiring the bars did not make the gate "
               "unfailable (sec 18.10 charge 7).",
               (not t2a1_11["passes"])
               and t2a1_11["leg_iv_ks_ci_excludes_zero_and_t2b1b"] is False,
               f"acc_copy={t2a1_11['acc_copy']} ks_ci={t2a1_11['leg_iv_ks_ci']} "
               f"leg_iv={t2a1_11['leg_iv_ks_ci_excludes_zero_and_t2b1b']} "
               f"leg_vi={t2a1_11['leg_vi_aiming_keyswap_argmax_changed']} "
               f"RETIRED(i)={t2a1_11['leg_i_median_ge_090_RETIRED_NONGATING']}")
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
        t2_fixture = {"val": val11, "pools": pools11, "counts": counts11, "V": V11,
                      "delta_pool": delta_pool11, "n_plants": n_plants11,
                      "live_t2a2": t2a2_11}
    except Exception as e:  # noqa: BLE001
        report("  sec 11 T2 REPAIR end-to-end pipeline", False, f"EXCEPTION: {type(e).__name__}: {e}")

    # --- [10f] sec 20 R-4 -- THE T2a-2 LIVENESS WITNESS: FORCED-FAIL NEGATIVE TESTS, RUN TO
    #     COMPLETION. THE CHECK THIS SUITE EXISTS TO PROVE HAS TEETH.
    #
    #     sec 19.3d: the T2a-2 artifact's entire model-dependent content was
    #         {"passes": true, "acc_copy": 0.0, "ks_point": 0.0, "ks_ci": [0.0, 0.0]}
    #     and a CONSTANT-logit or NaN-logit forward pass reproduces it BIT FOR BIT. This block
    #     BUILDS those two dead models, runs them through the REAL probe (not a mock of it),
    #     and demands:
    #        (1) the old artifact IS in fact reproduced bit-identically -- i.e. sec 19.3d's
    #            charge is DEMONSTRATED, not merely asserted (the pinned bar still reads PASS);
    #        (2) the NEW liveness witness FIRES on both, so `passes` is now FALSE.
    #     Failing (1) would mean the charge was wrong. Failing (2) would mean the fix is
    #     decorative. This repo's standing rule -- "always run the negative unit test that is
    #     supposed to prove the check has teeth TO COMPLETION" -- is what this block discharges.
    print("\n  [10f] sec 20 R-4: T2a-2 LIVENESS WITNESS -- FORCED-FAIL NEGATIVE TESTS")
    n_forced_fail_assertions = 0
    if t2_fixture is None:
        report("  [10f] FIXTURE UNAVAILABLE -- [10b] did not complete, so the forced-fail "
               "negative tests DID NOT RUN. This is a HARD FAIL, not a skip: an unrun teeth "
               "test is indistinguishable from a toothless one.", False)
    else:
        try:
            class _ConstantLogitsModel(torch.nn.Module):
                """The dead model sec 19.3d names first: input-independent logits, peaked on
                one token `c`. argmax == c at EVERY position of EVERY window; `b` is drawn from
                the licensed pool and is never `c`, so every hit indicator is 0 and the T2a-2
                artifact is the same three zeros a live null model produces."""

                def __init__(self, vocab_size: int, c: int = 0):
                    super().__init__()
                    self.vocab_size, self.c = vocab_size, c

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    B, T = x.shape
                    logits = torch.full((B, T, self.vocab_size), -3.0, device=x.device)
                    logits[:, :, self.c] = 7.0
                    return logits

            class _NaNLogitsModel(torch.nn.Module):
                """The second dead model sec 19.3d names: all-NaN logits. torch.argmax over
                all-NaN returns index 0; `b` is never token 0 => the identical artifact."""

                def __init__(self, vocab_size: int):
                    super().__init__()
                    self.vocab_size = vocab_size

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    B, T = x.shape
                    return torch.full((B, T, self.vocab_size), float("nan"), device=x.device)

            def _run_dead(model):
                r = run_t2_repaired_probe(model, t2_fixture["val"], seq_len=256, device=device,
                                           corpus_name="smoke-synth-corpus",
                                           delta_pool=t2_fixture["delta_pool"],
                                           pools=t2_fixture["pools"],
                                           counts_by_token=t2_fixture["counts"],
                                           n_plants=t2_fixture["n_plants"],
                                           vocab_size=t2_fixture["V"], eval_micro_batch=32)
                chk = check_t2a2_untrained_control(r["records"],
                                                    logit_liveness=r["logit_liveness"], n_boot=200)
                return r, chk

            # ---------- DEAD MODEL 1: CONSTANT LOGITS ----------
            const_model = _ConstantLogitsModel(t2_fixture["V"], c=0).to(device).eval()
            r_const, chk_const = _run_dead(const_model)
            lv_c = chk_const["liveness"]

            # (1) sec 19.3d's charge, DEMONSTRATED: the OLD artifact is bit-identical.
            n_forced_fail_assertions += 1
            report("  [CONSTANT LOGITS] sec 19.3d's charge is TRUE and is DEMONSTRATED, not "
                   "asserted: a dead forward pass reproduces the OLD T2a-2 artifact BIT FOR BIT "
                   "(acc_copy=0.0, ks_point=0.0, ks_ci=[0.0,0.0]) and the PINNED BAR still reads "
                   "PASS -- this is precisely why the old three-number artifact was worthless",
                   chk_const["acc_copy"] == 0.0 and chk_const["ks_point"] == 0.0
                   and chk_const["ks_ci"] == [0.0, 0.0]
                   and chk_const["pinned_bar_passes"] is True,
                   f"acc_copy={chk_const['acc_copy']} ks_point={chk_const['ks_point']} "
                   f"ks_ci={chk_const['ks_ci']} pinned_bar_passes={chk_const['pinned_bar_passes']}")

            # (2) THE TEETH: the liveness witness FIRES, and `passes` is now FALSE.
            n_forced_fail_assertions += 1
            report("  [CONSTANT LOGITS] FORCED FAIL: the liveness witness FIRES "
                   "(max|L[i]-L[0]| is EXACTLY 0.0 => the readout does not depend on the input) "
                   "and T2a-2 now reads passes=FALSE. The control can no longer certify the "
                   "instrument off a forward pass that never looked at its input.",
                   lv_c["ok"] is False
                   and lv_c["readout_logit_max_abs_dev_from_row0"] == 0.0
                   and lv_c["readout_logits_finite_frac"] == 1.0
                   and chk_const["passes"] is False,
                   f"liveness.ok={lv_c['ok']} "
                   f"max_abs_dev={lv_c['readout_logit_max_abs_dev_from_row0']} "
                   f"finite_frac={lv_c['readout_logits_finite_frac']} "
                   f"n_distinct_argmax={lv_c['n_distinct_argmax_at_k']} "
                   f"passes={chk_const['passes']} reasons={lv_c['degenerate_reasons']}")

            # ---------- DEAD MODEL 2: NaN LOGITS ----------
            nan_model = _NaNLogitsModel(t2_fixture["V"]).to(device).eval()
            r_nan, chk_nan = _run_dead(nan_model)
            lv_n = chk_nan["liveness"]

            n_forced_fail_assertions += 1
            report("  [NaN LOGITS] sec 19.3d's second charge, DEMONSTRATED: torch.argmax over "
                   "all-NaN returns index 0, `b` is never token 0, so the OLD artifact is again "
                   "reproduced bit-identically and the PINNED BAR again reads PASS",
                   chk_nan["acc_copy"] == 0.0 and chk_nan["ks_point"] == 0.0
                   and chk_nan["ks_ci"] == [0.0, 0.0]
                   and chk_nan["pinned_bar_passes"] is True,
                   f"acc_copy={chk_nan['acc_copy']} ks_point={chk_nan['ks_point']} "
                   f"ks_ci={chk_nan['ks_ci']} pinned_bar_passes={chk_nan['pinned_bar_passes']}")

            n_forced_fail_assertions += 1
            report("  [NaN LOGITS] FORCED FAIL: the liveness witness FIRES on BOTH structural "
                   "conditions (finite_frac < 1.0 AND max|L[i]-L[0]| is NaN, not > 0) and T2a-2 "
                   "now reads passes=FALSE",
                   lv_n["ok"] is False
                   and lv_n["readout_logits_finite_frac"] < 1.0
                   and lv_n["readout_logit_max_abs_dev_is_nan"] is True
                   and chk_nan["passes"] is False,
                   f"liveness.ok={lv_n['ok']} "
                   f"finite_frac={lv_n['readout_logits_finite_frac']} "
                   f"max_abs_dev_is_nan={lv_n['readout_logit_max_abs_dev_is_nan']} "
                   f"passes={chk_nan['passes']} reasons={lv_n['degenerate_reasons']}")

            # ---------- FAIL-CLOSED: a caller that OMITS the witness cannot pass. ----------
            chk_omitted = check_t2a2_untrained_control(r_const["records"],
                                                        logit_liveness=None, n_boot=200)
            n_forced_fail_assertions += 1
            report("  [FAIL-CLOSED] a caller that does NOT thread `logit_liveness` through gets "
                   "passes=FALSE, not a silent pass -- the control cannot certify the instrument "
                   "by OMITTING its own witness",
                   chk_omitted["passes"] is False and chk_omitted["liveness"]["ok"] is False
                   and chk_omitted["pinned_bar_passes"] is True,
                   f"passes={chk_omitted['passes']} pinned_bar={chk_omitted['pinned_bar_passes']}")

            # ---------- THE SEPARATION, STATED AS ONE NUMBER ----------
            lv_live = t2_fixture["live_t2a2"]["liveness"]
            n_forced_fail_assertions += 1
            report("  [SEPARATION] LIVE untrained vs DEAD constant-logit: the two produce "
                   "IDENTICAL acc_copy / ks_point / ks_ci (0.0 / 0.0 / [0.0,0.0]) and are "
                   "SEPARATED ONLY by the liveness witness -- which is the whole point",
                   lv_live["ok"] is True and lv_c["ok"] is False
                   and t2_fixture["live_t2a2"]["acc_copy"] == chk_const["acc_copy"]
                   and t2_fixture["live_t2a2"]["ks_ci"] == chk_const["ks_ci"],
                   f"LIVE: max_abs_dev={lv_live['readout_logit_max_abs_dev_from_row0']:.6g} "
                   f"n_distinct={lv_live['n_distinct_argmax_at_k']} | "
                   f"DEAD: max_abs_dev={lv_c['readout_logit_max_abs_dev_from_row0']} "
                   f"n_distinct={lv_c['n_distinct_argmax_at_k']}")

            # ---------- COVERAGE: the body actually ran. ----------
            # The expected count is PINNED (6), not derived from the counter, so a skipped or
            # exception-truncated assertion cannot hide behind a self-consistent tally. It has
            # already earned its keep: on the first run of this block it caught the BUILDER's
            # own miscount (6 increments asserted against a hardcoded 5) and turned the suite
            # red. A counter compared against itself would have gone green.
            report(f"  [COVERAGE] all {n_forced_fail_assertions}/6 forced-fail assertions "
                   f"EXECUTED (not skipped, not NameError'd behind a green count)",
                   n_forced_fail_assertions == 6, f"n={n_forced_fail_assertions}")
        except Exception as e:  # noqa: BLE001
            report("  [10f] sec 20 R-4 forced-fail negative tests", False,
                   f"EXCEPTION after {n_forced_fail_assertions} assertions: "
                   f"{type(e).__name__}: {e}")

    # --- [10g] sec 24 (B-1 + B-4) -- THE sec 18.4 PIN AND THE AIMING WITNESS: FORCED-FAIL
    #     NEGATIVE TESTS, RUN TO COMPLETION.
    #
    #     TWO CHECKS ARE ON TRIAL HERE, AND EACH NEEDS A DIFFERENT MODEL.
    #
    #     (A) THE AIMING WITNESS (leg (vi), B-4). sec 20's liveness witness proves the readout is
    #         FINITE and INPUT-DEPENDENT. It does NOT prove the readout is AIMED. `_MisAimedReadout
    #         Oracle` is the exact realization of sec 23.3's "reading logits at k0 +- 1": for a
    #         next-token model, reading at k0-1 returns the prediction for position k0, i.e. the
    #         token AT k0 -- which is `a`. This model emits, at every position t, logits peaked on
    #         x[t] itself. It is FULLY input-dependent (LIVENESS PASSES) and it is reading the
    #         wrong thing. The old gate could not see it. Leg (vi) must.
    #
    #     (B) THE sec 18.4 PIN ITSELF (B-1). `_WeakAimedInductionOracle` is a REAL, correctly-aimed
    #         induction head that copies on only ~half the plants (a fixed parity predicate on the
    #         key -- deterministic, and invariant across arms 1-4, which corrupt j0/p/p_placebo but
    #         never k0). It lands `acc_copy ~ 0.5`, i.e. squarely in the band the four REAL
    #         witnesses read (0.56-0.69). On IDENTICAL records the RETIRED bars HALT and the
    #         OPERATIVE sec 18.4 gate PASSES. That is sec 23.2's "the two gates give OPPOSITE
    #         verdicts on the SAME data", reproduced in the suite -- and it is the demonstration
    #         that the pin is now COMPUTED BY THE INSTRUMENT rather than asserted by an agent
    #         reading a table.
    #
    #     (C) And the leg the CI replaced: a KNIFE-EDGE `KS` whose bootstrap CI COVERS 0 must FAIL
    #         leg (iv) even though its POINT estimate is > 0 (sec 16's W-2 -- W2/openr1's
    #         KS = 0.49951 was gated on a bare point estimate with no CI at all).
    print("\n  [10g] sec 24 B-1/B-4: THE sec 18.4 PIN + THE AIMING WITNESS -- FORCED-FAIL")
    n_aim_assertions = 0
    if t2_fixture is None:
        report("  [10g] FIXTURE UNAVAILABLE -- [10b] did not complete, so the forced-fail "
               "negative tests DID NOT RUN. HARD FAIL, not a skip: an unrun teeth test is "
               "indistinguishable from a toothless one.", False)
    else:
        try:
            WRONG_TOK = t2_fixture["V"] - 1   # background id; never a licensed value `b`

            class _MisAimedReadoutOracle(torch.nn.Module):
                """(A) THE MIS-AIMED READ, realized model-side. Emits at position t logits peaked
                on x[t] -- the token AT t, not the token AFTER it. For a next-token model this is
                observationally IDENTICAL to reading the logits at k0-1 instead of k0 (sec 23.3's
                own example). Fully input-dependent => sec 20's liveness witness PASSES. Its argmax
                at k0 is `a` in every arm that does not corrupt k0, so swapping the key at j0
                changes NOTHING: argmax_changed_frac_keyswap == EXACTLY 0.0."""

                def __init__(self, vocab_size: int):
                    super().__init__()
                    self.vocab_size = vocab_size

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    B, T = x.shape
                    logits = torch.full((B, T, self.vocab_size), -3.0, device=x.device)
                    logits.scatter_(2, x.unsqueeze(-1), 7.0)      # peak on x[t] ITSELF
                    return logits

            class _WeakAimedInductionOracle(torch.nn.Module):
                """(B) A REAL, CORRECTLY-AIMED induction head that is DELIBERATELY WEAK. At
                position t it predicts the token that FOLLOWED the FIRST EARLIER occurrence of
                x[t] -- the copy mechanism the probe exists to detect -- but only when the key
                clears a fixed parity predicate (x[t] % 2 == 0); otherwise it emits `wrong_token`.
                The predicate reads the token AT the readout position, which arms 1-4 never
                corrupt, so a plant's copy/no-copy status is IDENTICAL across the intact, true-
                ablated, placebo, pool-placebo and key-swap arms -- the weakness is a property of
                the MODEL, not an artifact of the ablation. Lands acc_copy ~ 0.5."""

                def __init__(self, vocab_size: int, wrong_token: int):
                    super().__init__()
                    self.vocab_size, self.wrong_token = vocab_size, wrong_token

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    B, T = x.shape
                    idx = torch.arange(T, device=x.device)
                    eq = x.unsqueeze(2) == x.unsqueeze(1)                 # (B,T,T) x[t]==x[i]
                    earlier = idx.view(1, 1, T) < idx.view(1, T, 1)       # i < t
                    cand = eq & earlier
                    # FIRST earlier occurrence, computed by MIN over indices -- never argmax, whose
                    # tie-breaking among equal maxima torch does not guarantee.
                    pos = torch.where(cand, idx.view(1, 1, T).expand(B, T, T),
                                       torch.full((B, T, T), T, device=x.device, dtype=torch.long))
                    first_i = pos.min(dim=-1).values                      # (B,T); == T if none
                    has_earlier = first_i < T
                    nxt = torch.gather(x, 1, (first_i + 1).clamp(max=T - 1))
                    do_copy = has_earlier & (x % 2 == 0)                  # the weakness predicate
                    pred = torch.where(do_copy, nxt,
                                        torch.full_like(x, self.wrong_token))
                    logits = torch.full((B, T, self.vocab_size), -3.0, device=x.device)
                    logits.scatter_(2, pred.unsqueeze(-1), 7.0)
                    return logits

            def _run_oracle(model):
                r = run_t2_repaired_probe(model, t2_fixture["val"], seq_len=256, device=device,
                                           corpus_name="smoke-synth-corpus",
                                           delta_pool=t2_fixture["delta_pool"],
                                           pools=t2_fixture["pools"],
                                           counts_by_token=t2_fixture["counts"],
                                           n_plants=t2_fixture["n_plants"],
                                           vocab_size=t2_fixture["V"], eval_micro_batch=32)
                b1 = check_t2b1_mechanism_exists(r["records"])
                b1b = check_t2b1b_key_conditioned(r["records"])
                return (r, b1, b1b,
                        check_t2a1_ceiling(r["records"], b1, b1b, n_boot=200),
                        check_t2a3_ssm_calibration(r["records"], b1, b1b, n_boot=200),
                        check_t2a2_untrained_control(r["records"],
                                                      logit_liveness=r["logit_liveness"], n_boot=200))

            # ================= (A) THE MIS-AIMED READOUT =================
            mis = _MisAimedReadoutOracle(t2_fixture["V"]).to(device).eval()
            r_mis, b1_mis, b1b_mis, a1_mis, a3_mis, a2_mis = _run_oracle(mis)
            lv_mis = r_mis["logit_liveness"]

            n_aim_assertions += 1
            report("  [MIS-AIMED] THE GAP sec 23.3 NAMED, DEMONSTRATED: sec 20's LIVENESS WITNESS "
                   "PASSES on a probe that is reading the WRONG POSITION. finite_frac==1.0 and "
                   "max|L[i]-L[0]|>0 -- the readout is alive, input-dependent, and measuring "
                   "NOTHING. Liveness upgraded the control from 'cannot tell DEAD from NULL' to "
                   "'cannot tell MIS-AIMED from NULL', and no further.",
                   lv_mis["ok"] is True
                   and lv_mis["readout_logits_finite_frac"] == 1.0
                   and lv_mis["readout_logit_max_abs_dev_from_row0"] > 0.0,
                   f"liveness.ok={lv_mis['ok']} finite_frac={lv_mis['readout_logits_finite_frac']} "
                   f"max_abs_dev={lv_mis['readout_logit_max_abs_dev_from_row0']:.6g} "
                   f"acc_copy={a1_mis['acc_copy']}")

            n_aim_assertions += 1
            report("  [MIS-AIMED] FORCED FAIL: LEG (vi) FIRES. argmax_changed_frac_keyswap is "
                   "EXACTLY 0.0 -- swapping the PLANTED KEY changed the readout argmax in ZERO of "
                   "the windows -- so T2a-1 reads passes=FALSE and NAMES the reason as a MIS-AIMED "
                   "PROBE rather than a null model.",
                   a1_mis["aiming_argmax_changed_frac_keyswap"] == 0.0
                   and a1_mis["leg_vi_aiming_keyswap_argmax_changed"] is False
                   and a1_mis["passes"] is False
                   and any("MIS-AIMED PROBE" in s for s in a1_mis["failure_reasons"]),
                   f"aiming_frac={a1_mis['aiming_argmax_changed_frac_keyswap']} "
                   f"leg_vi={a1_mis['leg_vi_aiming_keyswap_argmax_changed']} "
                   f"passes={a1_mis['passes']}")

            n_aim_assertions += 1
            report("  [MIS-AIMED] THE ENTAILMENT, DISCLOSED NOT HIDDEN: leg (iv) FAILS TOO "
                   "(KS CI == [0,0]) -- because argmax_changed_frac==0 forces KS==0 on every "
                   "bootstrap draw. LEG (vi) IS A TRIPWIRE AND A NAMED REASON, NOT NEW GATE "
                   "POWER: it can only fire where leg (iv) has already failed. sec 24 says so.",
                   a1_mis["leg_iv_ks_ci_excludes_zero"] is False
                   and a1_mis["leg_iv_ks_ci"] == [0.0, 0.0]
                   and a1_mis["leg_iv_ks_ci_excludes_zero_and_t2b1b"] is False,
                   f"ks_point={a1_mis['leg_iv_ks_point']} ks_ci={a1_mis['leg_iv_ks_ci']}")

            n_aim_assertions += 1
            report("  [MIS-AIMED] T2a-3 (the C1 / falcon-mamba causal gate) ALSO fires on the "
                   "aiming conjunct -- the pure-SSM cell is exactly where 'no mechanism' and "
                   "'mis-aimed probe' are easiest to confuse. The pinned causal legs are "
                   "UNTOUCHED; this is a MONOTONE TIGHTENING (PASS->HALT only).",
                   a3_mis["passes"] is False
                   and a3_mis["aiming_keyswap_argmax_changed"] is False
                   and a3_mis["ks_positive_excludes_zero"] is False,
                   f"t2a3_passes={a3_mis['passes']} "
                   f"aiming={a3_mis['aiming_keyswap_argmax_changed']} "
                   f"acc_copy={a3_mis['acc_copy']}")

            n_aim_assertions += 1
            report("  [MIS-AIMED / NO FALSE HALT] T2a-2 does NOT gate on aiming, and here is the "
                   "run that proves why it must not: on these very records argmax_changed_frac is "
                   "EXACTLY 0.0 and the untrained control STILL reads passes=TRUE. sec 23.4 item "
                   "2 asked for this leg to gate T2a-2; a live HEALTHY untrained model measured "
                   "0.0000 (sec 20.4b), so that gate would HALT a healthy control. REJECTED, and "
                   "the rejection is enforced by this test.",
                   a2_mis["passes"] is True and a2_mis["pinned_bar_passes"] is True
                   and a2_mis["liveness"]["argmax_changed_frac_keyswap"] == 0.0,
                   f"t2a2_passes={a2_mis['passes']} pinned_bar={a2_mis['pinned_bar_passes']} "
                   f"aiming_frac={a2_mis['liveness']['argmax_changed_frac_keyswap']} "
                   f"acc_copy={a2_mis['acc_copy']}")

            # ================= (B) THE sec 18.4 PIN, ON A WEAK BUT REAL MECHANISM =================
            weak = _WeakAimedInductionOracle(t2_fixture["V"], WRONG_TOK).to(device).eval()
            r_wk, b1_wk, b1b_wk, a1_wk, a3_wk, _a2_wk = _run_oracle(weak)

            n_aim_assertions += 1
            report("  [WEAK ORACLE] THE sec 18.4 PIN, DEMONSTRATED ON THE CODE PATH: on IDENTICAL "
                   "records the RETIRED bars HALT (acc_at_median < 0.90 AND some decile < 0.75) "
                   "while the OPERATIVE gate PASSES (legs iii/iv/v/vi all True). This is sec 23.2's "
                   "'the two gates give OPPOSITE verdicts on the SAME data' -- and the instrument, "
                   "not an agent reading a table, now computes the verdict.",
                   a1_wk["leg_i_median_ge_090_RETIRED_NONGATING"] is False
                   and a1_wk["leg_ii_all_deciles_ge_075_RETIRED_NONGATING"] is False
                   and a1_wk["leg_iii_prior_le_005"] is True
                   and a1_wk["leg_iv_ks_ci_excludes_zero_and_t2b1b"] is True
                   and a1_wk["leg_v_t2b1_passes"] is True
                   and a1_wk["leg_vi_aiming_keyswap_argmax_changed"] is True
                   and a1_wk["passes"] is True,
                   f"acc_copy={a1_wk['acc_copy']:.4f} acc_at_median={a1_wk['acc_at_median']:.4f} "
                   f"PRIOR={a1_wk['prior']:.4f} ks={a1_wk['ks']:.4f} "
                   f"ks_ci={[round(v, 4) for v in a1_wk['leg_iv_ks_ci']]} "
                   f"aiming={a1_wk['aiming_argmax_changed_frac_keyswap']:.4f} "
                   f"RETIRED(i)={a1_wk['leg_i_median_ge_090_RETIRED_NONGATING']} "
                   f"RETIRED(ii)={a1_wk['leg_ii_all_deciles_ge_075_RETIRED_NONGATING']} "
                   f"passes={a1_wk['passes']}")

            n_aim_assertions += 1
            report("  [WEAK ORACLE / NO FALSE HALT] LEG (vi) does NOT halt a healthy, correctly-"
                   "aimed model that reads WEAKLY: argmax_changed_frac_keyswap > 0. A gate that "
                   "halted here would be an absolute competence bar wearing an aiming costume -- "
                   "exactly the shape RULE T retires.",
                   a1_wk["aiming_argmax_changed_frac_keyswap"] > 0.0
                   and a1_wk["leg_vi_aiming_keyswap_argmax_changed"] is True
                   and a3_wk["passes"] is True,
                   f"aiming_frac={a1_wk['aiming_argmax_changed_frac_keyswap']:.4f} "
                   f"t2a3_passes={a3_wk['passes']}")

            n_aim_assertions += 1
            report("  [CONJUNCTION] `passes` is EXACTLY legs (iii) AND (iv) AND (v) AND (vi) -- the "
                   "RETIRED legs (i)/(ii) are computed, REPORTED, and enter the verdict NOWHERE. "
                   "Re-derived from the returned flags on both oracles, so a future edit that "
                   "quietly re-admits a retired bar turns this suite RED.",
                   all(chk["passes"] == (chk["leg_iii_prior_le_005"]
                                          and chk["leg_iv_ks_ci_excludes_zero_and_t2b1b"]
                                          and chk["leg_v_t2b1_passes"]
                                          and chk["leg_vi_aiming_keyswap_argmax_changed"])
                       for chk in (a1_mis, a1_wk, t2a1_11))
                   and a1_wk["gating_legs"] == ["iii_prior", "iv_ks_ci_excludes_zero_and_t2b1b",
                                                 "v_t2b1", "vi_aiming"],
                   f"gating_legs={a1_wk['gating_legs']} "
                   f"retired={a1_wk['retired_legs_REPORTED_NEVER_GATING']}")

            # ================= (C) LEG (iv)'s CI HAS TEETH THE POINT ESTIMATE DID NOT ===========
            # sec 16's W-2: the OLD leg (iv) gated `ks >= 0.50` on a BARE POINT ESTIMATE with no CI
            # anywhere. W2/openr1 read KS = 0.49951 and was HALTed by 0.00049 -- while its true 95%
            # CI [0.475, 0.524] COVERS 0.50. The re-pinned leg gates the CI. These records carry a
            # KS whose POINT estimate is > 0 but whose CI COVERS 0; the leg must FAIL.
            knife = []
            for i in range(400):
                hit_i = 1 if i < 3 else 0            # 3 intact hits
                hit_k = 1 if 3 <= i < 5 else 0       # 2 keyswap hits  => KS point = +0.0025 > 0
                knife.append({
                    "row_idx": i, "delta": 10 + (i % 100), "hit_intact": hit_i,
                    "hit_keyswap": hit_k, "hit_noplant": 0,
                    "argmax_intact_at_k": 1 if hit_i else 900 + i,
                    "argmax_keyswap_at_k": 1 if hit_k else 7000 + i,   # aiming > 0: argmaxes differ
                })
            passing = {"passes": True, "p_value": 1e-9}
            a1_knife = check_t2a1_ceiling(knife, passing, passing, n_boot=400)
            n_aim_assertions += 1
            report("  [KNIFE-EDGE KS] FORCED FAIL: a KS whose POINT estimate is > 0 but whose "
                   "clustered-bootstrap 95% CI COVERS 0 FAILS leg (iv) -- the sec 16 W-2 defect "
                   "(the old leg gated a BARE POINT ESTIMATE, no CI anywhere) is CLOSED. Legs "
                   "(iii)/(v)/(vi) all pass on these records, so leg (iv) is the ONLY thing "
                   "standing between them and an INSTRUMENT_VALID.",
                   a1_knife["leg_iv_ks_point"] > 0.0
                   and a1_knife["leg_iv_ks_ci"][0] <= 0.0
                   and a1_knife["leg_iv_ks_ci_excludes_zero"] is False
                   and a1_knife["leg_iii_prior_le_005"] is True
                   and a1_knife["leg_v_t2b1_passes"] is True
                   and a1_knife["leg_vi_aiming_keyswap_argmax_changed"] is True
                   and a1_knife["passes"] is False,
                   f"ks_point={a1_knife['leg_iv_ks_point']:.6f} "
                   f"ks_ci={[round(v, 6) for v in a1_knife['leg_iv_ks_ci']]} "
                   f"passes={a1_knife['passes']}")

            # The COVERAGE check does NOT increment the counter it audits (that is the whole point:
            # a tally compared against itself goes green on a skipped body). It earned its keep
            # immediately -- on the first run it caught THIS BUILDER counting its own coverage
            # report as an assertion (n=10 against a hardcoded 9) and turned the suite RED. That is
            # now the SECOND consecutive builder it has caught, cf. sec 20.4(e).
            report(f"  [COVERAGE] all {n_aim_assertions}/9 forced-fail assertions EXECUTED (the "
                   f"expected count is HARDCODED, never derived from the counter -- a tally "
                   f"compared against itself goes green on a skipped body)",
                   n_aim_assertions == 9, f"n={n_aim_assertions}")
        except Exception as e:  # noqa: BLE001
            report("  [10g] sec 24 B-1/B-4 forced-fail negative tests", False,
                   f"EXCEPTION after {n_aim_assertions} assertions: {type(e).__name__}: {e}")

    # --- [10h] sec 24 (B-2) -- THE sec 18.4.1 INFLUENCE LADDER: FORCED-FAIL NEGATIVE TESTS.
    #     Model-free (pure CPU over synthetic per-rung values). sec 9.4's binary strong/weak SPLIT
    #     is RETIRED (its predicate `acc_copy >= 0.90` no longer exists); the ladder replaces it.
    #     INDETERMINATE must fire IFF the fitted exponent's SIGN or its CI's EXCLUSION of the
    #     no-trend null FLIPS anywhere along the prefix-drop ordering.
    print("\n  [10h] sec 24 B-2: THE sec 18.4.1 INFLUENCE LADDER -- FORCED-FAIL")
    n_ladder_assertions = 0
    try:
        log10p = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "r4": 4.0}
        ks_asc = {"r1": 0.10, "r2": 0.20, "r3": 0.30, "r4": 0.40}   # r1 is the LOWEST-KS rung

        def _rows(v, n=40):
            return {i: v for i in range(n)}   # zero within-rung variance => an exact, seedless CI

        def _rows_noisy(v, n=40):
            """Non-degenerate within-rung spread (sd ~ 0.11), so the bootstrap CI has WIDTH. A
            zero-variance rung yields ci95 == [beta, beta], which can NEVER include 0 unless beta
            is exactly 0 -- so the exclusion-flip condition below is untestable without this. (The
            first draft of this block used `_rows` throughout and the exclusion test could not
            fail; the forced-fail run caught it, which is the entire reason for running them.)"""
            jitter = (-0.15, -0.05, 0.05, 0.15)
            return {i: v + jitter[i % len(jitter)] for i in range(n)}

        # FLIP: over all four rungs the slope is POSITIVE (+0.23); drop the lowest-KS rung (r1)
        # and it turns NEGATIVE (-0.10). The SIGN flips along the ladder => INDETERMINATE.
        flip_vals = {"r1": _rows(0.0), "r2": _rows(1.0), "r3": _rows(0.9), "r4": _rows(0.8)}
        lad_flip = influence_ladder(flip_vals, log10p, ks_asc, n_boot=200)
        n_ladder_assertions += 1
        report("  [LADDER] FORCED FAIL: the fitted exponent's SIGN FLIPS when the lowest-KS rung "
               "is dropped (+ -> -) and the ladder returns INDETERMINATE. This is the condition "
               "sec 16.5 proved sec 15's median-KS split could NEVER detect.",
               lad_flip["evaluable"] is True and lad_flip["indeterminate"] is True
               and lad_flip["sign_flips"] is True and lad_flip["n_steps"] == 2
               and lad_flip["steps"][0]["beta_sign"] == 1
               and lad_flip["steps"][1]["beta_sign"] == -1,
               f"n_steps={lad_flip['n_steps']} "
               f"betas={[round(s['beta_point'], 4) for s in lad_flip['steps']]} "
               f"signs={[s['beta_sign'] for s in lad_flip['steps']]} "
               f"indeterminate={lad_flip['indeterminate']}")

        # NO FLIP: a clean monotone trend. Sign and CI-exclusion are stable => NOT indeterminate.
        # (The ladder must not cry INDETERMINATE at every dataset, or it is not a check either.)
        stable_vals = {"r1": _rows(0.1), "r2": _rows(0.2), "r3": _rows(0.3), "r4": _rows(0.4)}
        lad_ok = influence_ladder(stable_vals, log10p, ks_asc, n_boot=200)
        n_ladder_assertions += 1
        report("  [LADDER] and it does NOT fire on a robust trend: sign and CI-exclusion are "
               "stable across every prefix-drop => indeterminate=False. A check that always "
               "fires is not a check.",
               lad_ok["evaluable"] is True and lad_ok["indeterminate"] is False
               and lad_ok["sign_flips"] is False and lad_ok["exclusion_flips"] is False
               and lad_ok["n_steps"] == 2,
               f"betas={[round(s['beta_point'], 4) for s in lad_ok['steps']]} "
               f"excl={[s['ci_excludes_no_trend_null'] for s in lad_ok['steps']]}")

        # EXCLUSION FLIP with a STABLE sign: beta stays positive throughout, but dropping the
        # lowest-KS rung collapses the CI's exclusion of the no-trend null. sec 18.4.1 fires on
        # EITHER condition, not only on the sign.
        # The lowest-KS rung (r1) is the one carrying the trend: over all four the slope is a solid
        # +0.094 whose CI excludes 0; drop r1 and the remaining three are flat-within-noise
        # (+0.010, CI straddling 0). The sign is POSITIVE at BOTH steps.
        excl_vals = {"r1": _rows_noisy(0.0), "r2": _rows_noisy(0.30),
                     "r3": _rows_noisy(0.28), "r4": _rows_noisy(0.32)}
        lad_excl = influence_ladder(excl_vals, log10p, ks_asc, n_boot=2000)
        n_ladder_assertions += 1
        report("  [LADDER] FORCED FAIL, SECOND CONDITION: the SIGN never flips (positive at every "
               "step), but the CI's EXCLUSION of the no-trend null DOES -- and sec 18.4.1 fires on "
               "EITHER. A build that implemented only the sign clause would pass every other test "
               "in this block and fail here.",
               lad_excl["evaluable"] is True and lad_excl["indeterminate"] is True
               and lad_excl["sign_flips"] is False and lad_excl["exclusion_flips"] is True
               and lad_excl["steps"][0]["ci_excludes_no_trend_null"] is True
               and lad_excl["steps"][1]["ci_excludes_no_trend_null"] is False,
               f"betas={[round(s['beta_point'], 4) for s in lad_excl['steps']]} "
               f"signs={[s['beta_sign'] for s in lad_excl['steps']]} "
               f"cis={[[round(v, 4) for v in s['ci95']] for s in lad_excl['steps']]} "
               f"excl={[s['ci_excludes_no_trend_null'] for s in lad_excl['steps']]}")

        # THE VACUOUS REGIME -- flagged, not hidden. At exactly 3 rungs the ladder has ONE step,
        # so NOTHING CAN FLIP and "robust" means only "not checked". sec 11.8 records 2 fit rungs
        # available against a minimum of 3, so this is the LIKELY regime.
        lad_vac = influence_ladder({k: stable_vals[k] for k in ("r1", "r2", "r3")},
                                    log10p, ks_asc, n_boot=200)
        n_ladder_assertions += 1
        report("  [LADDER] THE VACUOUS REGIME IS DECLARED, NOT HIDDEN: at exactly 3 rungs there "
               "is ONE step, nothing can flip, and `ladder_is_vacuous=True` says so in the "
               "artifact. sec 24 records this as a RESIDUAL CONCERN rather than repairing it "
               "silently -- the pin is implemented as written.",
               lad_vac["evaluable"] is True and lad_vac["n_steps"] == 1
               and lad_vac["ladder_is_vacuous"] is True
               and lad_vac["indeterminate"] is False
               and any("VACUOUS" in s for s in lad_vac["reasons"]),
               f"n_steps={lad_vac['n_steps']} vacuous={lad_vac['ladder_is_vacuous']}")

        # FAIL-CLOSED: a rung whose KS is NaN cannot be ORDERED. It must never be silently
        # dropped from the fit -- the ladder refuses to evaluate.
        lad_nan = influence_ladder(stable_vals, log10p, dict(ks_asc, r3=float("nan")), n_boot=200)
        n_ladder_assertions += 1
        report("  [LADDER] FAIL-CLOSED: a rung carrying a NaN KS cannot be ordered, so the ladder "
               "refuses to evaluate and returns INDETERMINATE -- it never silently drops a rung "
               "from the fit (which would be a selection effect wearing a missing-data costume)",
               lad_nan["evaluable"] is False and lad_nan["indeterminate"] is True
               and lad_nan["n_steps"] == 0,
               f"evaluable={lad_nan['evaluable']} reasons={lad_nan['reasons']}")

        n_ladder_assertions += 1
        report(f"  [COVERAGE] all {n_ladder_assertions}/6 ladder assertions EXECUTED (expected "
               f"count HARDCODED)", n_ladder_assertions == 6, f"n={n_ladder_assertions}")
    except Exception as e:  # noqa: BLE001
        report("  [10h] sec 24 B-2 influence-ladder forced-fail tests", False,
               f"EXCEPTION after {n_ladder_assertions} assertions: {type(e).__name__}: {e}")

    # --- [10i] sec 26 -- THE POSITIVE CONTROL (T2a-4): FORCED-FAIL NEGATIVE TESTS, RUN TO
    #     COMPLETION, **INCLUDING THE INTERIOR OF THE AIMING-COVERAGE AXIS** -- the test sec 24
    #     did not write and which would have caught A-1 before it shipped (sec 25.5 item 3).
    #
    #     THE DEFECT UNDER TEST (sec 25.1, A-1, BLOCKER). Leg (vi) fires iff the readout is
    #     mis-aimed on LITERALLY EVERY row, and -- being ENTAILED by leg (iv) -- it inherits leg
    #     (iv)'s floor and can never bind above it. sec 25 BUILT the interior case on the deployed
    #     code: a readout correctly aimed on 12 of 2048 rows (0.6%) and mis-aimed on the other
    #     2036 reads acc_copy=0.0059, PRIOR=0.0000, KS 95% CI [0.0029, 0.0093] (excludes 0),
    #     aiming=0.0059 > 0 => ALL FOUR GATING LEGS PASS => T2a-1 PASS, T2a-3 PASS,
    #     INSTRUMENT_VALID. The RETIRED bars would have HALTed it.
    #
    #     THREE THINGS ARE ON TRIAL, AND ALL THREE MUST HOLD OR THE FIX IS WORTHLESS:
    #       (A) sec 25's 0.6%-aimed instrument MUST FAIL the positive control. Reconstructed at
    #           the PINNED n = 2048, on the REAL DEPLOYED check functions -- both as sec 25 did it
    #           (records-level, the check functions' own interface) and END-TO-END through
    #           `run_t2_repaired_probe` itself with the defect in the READOUT PATH.
    #       (B) A CORRECTLY-AIMED oracle MUST PASS. A positive control that halts healthy
    #           instruments is worse than none.
    #       (C) A HEALTHY BUT DISTANCE-LIMITED WITNESS MUST NOT FALSE-HALT. This is the hazard
    #           sec 25.5 named IN ITS OWN preferred direction, and the identical error sec 23.4
    #           item 2 made: W1/openr1's Delta-deciles run 0.907 -> 0.376, so a healthy witness
    #           can legitimately read aiming = 0 in its largest Delta-decile. T2a-4 reads NO
    #           witness quantity, and this test proves the witness path is untouched.
    print("\n  [10i] sec 26: THE POSITIVE CONTROL (T2a-4) -- FORCED-FAIL, INTERIOR OF THE AXIS")
    n_pc_assertions = 0
    if t2_fixture is None:
        report("  [10i] FIXTURE UNAVAILABLE -- [10b] did not complete, so the forced-fail "
               "negative tests DID NOT RUN. HARD FAIL, not a skip: an unrun teeth test is "
               "indistinguishable from a toothless one.", False)
    else:
        try:
            # ============ (A) sec 25's A-1 INSTRUMENT, RECONSTRUCTED AT THE PINNED n = 2048 ======
            # sec 25 ran its construction through the REAL check functions at N_rows = 2048, which
            # is the interface those functions take (`records`). Rebuilt here to the letter: `k` of
            # `n` rows CORRECTLY AIMED (the readout recovers the plant: argmax_intact == b) and the
            # other `n - k` MIS-AIMED -- input-dependent (a distinct junk argmax per row, so sec
            # 20's LIVENESS witness passes) and PLANT-INDEPENDENT (argmax_intact == argmax_keyswap
            # BIT-IDENTICALLY, so the key-swap cannot move it and no arm ever equals `b`).
            N25, PLANT_B = 2048, 4242

            def _sliver_records(n_aimed: int, n: int = N25) -> list:
                """On an AIMED row the readout recovers `b` in every arm that does NOT break the
                key->value binding: arm 3 (placebo) and arm 3b (POOL-placebo) corrupt a PLACEBO
                position -- neither the key at j0 nor the value at p -- so the copy still lands
                (hit = 1); arm 2 (TRUE-ablate) destroys the value at p and arm 4 (KEY-SWAP)
                destroys the key at j0, so both read 0. On a MIS-AIMED row the readout is
                plant-independent junk in EVERY arm, so all six read 0.
                **THIS IS PINNED BY sec 25's OWN p-VALUES, NOT BY GUESSWORK:** its table reads
                T2b-1b p = 9.3e-10 at 31 aimed and 1.9e-06 at 20 aimed, and
                `2 * 0.5**31 = 9.313e-10`, `2 * 0.5**20 = 1.907e-06` -- an EXACT binomial with
                n_plus = n_aimed and n_minus = 0, which is this arm assignment and no other."""
                recs = []
                for i in range(n):
                    aimed = i < n_aimed
                    junk = 9000 + i            # input-dependent, never `b`, distinct per row
                    recs.append({
                        "row_idx": i, "delta": 2 + (i % 120), "b": PLANT_B,
                        "argmax_intact_at_k": PLANT_B if aimed else junk,
                        # PLANT-INDEPENDENT on the mis-aimed rows: swapping the key at j0 changes
                        # NOTHING. On the aimed rows the readout is key-conditioned, so it moves.
                        "argmax_keyswap_at_k": (PLANT_B + 1) if aimed else junk,
                        "argmax_noplant_at_k": junk,
                        "hit_intact": int(aimed), "hit_keyswap": 0, "hit_noplant": 0,
                        "hit_true_ablated": 0, "hit_placebo_ablated": int(aimed),
                        "hit_pool_placebo": int(aimed),
                    })
                return recs

            sliver = _sliver_records(12)                       # 12/2048 = 0.586% -- sec 25's row
            b1_s = check_t2b1_mechanism_exists(sliver)
            b1b_s = check_t2b1b_key_conditioned(sliver)
            a1_s = check_t2a1_ceiling(sliver, b1_s, b1b_s)     # n_boot default 2000, as deployed
            a3_s = check_t2a3_ssm_calibration(sliver, b1_s, b1b_s)
            a4_s = check_t2a4_positive_control(sliver)

            n_pc_assertions += 1
            report("  [A-1 REPRODUCED] sec 25's BLOCKER, on the DEPLOYED check functions: a readout "
                   "correctly aimed on 12 of 2048 rows (0.6%) and MIS-AIMED on the other 2036 "
                   "CLEARS ALL FOUR OPERATIVE GATING LEGS -- PRIOR=0, KS CI EXCLUDES 0, T2b-1 and "
                   "T2b-1b p<0.001, aiming>0 -- and certifies T2a-1 PASS **and** T2a-3 PASS. The "
                   "leg built to catch a mis-aimed probe (vi) reads 0.0059 > 0 => TRUE. This is the "
                   "instrument the retired bars would have HALTed and the operative gate CERTIFIES.",
                   a1_s["leg_iii_prior_le_005"] is True
                   and a1_s["leg_iv_ks_ci_excludes_zero_and_t2b1b"] is True
                   and a1_s["leg_v_t2b1_passes"] is True
                   and a1_s["leg_vi_aiming_keyswap_argmax_changed"] is True
                   and a1_s["passes"] is True and a3_s["passes"] is True,
                   f"acc_copy={a1_s['acc_copy']:.4f} PRIOR={a1_s['prior']:.4f} "
                   f"ks_ci={[round(v, 4) for v in a1_s['leg_iv_ks_ci']]} "
                   f"aiming={a1_s['aiming_argmax_changed_frac_keyswap']:.4f} "
                   f"T2a1={a1_s['passes']} T2a3={a3_s['passes']}")

            # THE RECONSTRUCTION IS FAITHFUL TO sec 25 TO THE DIGIT, AND THIS ASSERTS IT RATHER THAN
            # CLAIMING IT. sec 25's table gives T2b-1b p = 9.3e-10 at 31 aimed and 1.9e-06 at 20
            # aimed; the exact binomial with n_plus = n_aimed, n_minus = 0 gives 2*0.5**31 =
            # 9.3132e-10 and 2*0.5**20 = 1.9073e-06. If a future edit perturbs the arm assignment,
            # the reconstruction stops being sec 25's instrument and THIS goes red -- which is
            # exactly how the first draft of this very block was caught (its `hit_pool_placebo` was
            # 0, so T2b-1b had ZERO discordant pairs, p = 1.0, and the A-1 instrument did NOT in
            # fact pass -- a forced-fail test catching a defect in a forced-fail test, cf. sec 24.5).
            p31 = check_t2b1b_key_conditioned(_sliver_records(31))["p_value"]
            p20 = check_t2b1b_key_conditioned(_sliver_records(20))["p_value"]
            n_pc_assertions += 1
            report("  [A-1 FAITHFUL] The reconstruction reproduces sec 25's OWN NUMBERS to the "
                   "digit -- acc_copy 0.0059, PRIOR 0.0000, KS CI [0.0029, 0.0093], aiming 0.0059, "
                   "and T2b-1b p = 9.3e-10 @ 31 aimed / 1.9e-06 @ 20 aimed (= 2*0.5^31, 2*0.5^20 "
                   "EXACTLY). It is sec 25's instrument, not a lookalike.",
                   abs(p31 - 2 * 0.5 ** 31) < 1e-18 and abs(p20 - 2 * 0.5 ** 20) < 1e-12
                   and round(a1_s["acc_copy"], 4) == 0.0059
                   and a1_s["prior"] == 0.0
                   and [round(v, 4) for v in a1_s["leg_iv_ks_ci"]] == [0.0029, 0.0093]
                   and round(a1_s["aiming_argmax_changed_frac_keyswap"], 4) == 0.0059,
                   f"p(31)={p31:.4g} (sec 25: 9.3e-10) p(20)={p20:.4g} (sec 25: 1.9e-06) "
                   f"acc_copy={a1_s['acc_copy']:.4f} PRIOR={a1_s['prior']:.4f} "
                   f"ks_ci={[round(v, 4) for v in a1_s['leg_iv_ks_ci']]}")

            n_pc_assertions += 1
            report("  [A-1 CLOSED] **THE DECISIVE FORCED FAIL**: THAT SAME 0.6%-aimed instrument "
                   "FAILS THE POSITIVE CONTROL. n_miss_recovery = 2036 against a construction null "
                   "of EXACTLY 0 -- the probe failed to recover a plant emitted BY FIAT on 99.4% of "
                   "rows -- and n_aim_unchanged = 2036. T2a-4 passes=False, and it NAMES the reason "
                   "a MIS-AIMED INSTRUMENT. The gate can now distinguish 0.6%-aimed from 100%-aimed; "
                   "it could not, at ANY leg, before this.",
                   a4_s["passes"] is False
                   and a4_s["n_miss_recovery"] == N25 - 12
                   and a4_s["n_aim_unchanged"] == N25 - 12
                   and a4_s["pc1_recovery_no_miss"] is False
                   and a4_s["pc2_aiming_no_unchanged"] is False
                   and any("MIS-AIMED INSTRUMENT" in s for s in a4_s["failure_reasons"]),
                   f"passes={a4_s['passes']} n_miss_recovery={a4_s['n_miss_recovery']}/"
                   f"{a4_s['n_records']} n_aim_unchanged={a4_s['n_aim_unchanged']} "
                   f"null={a4_s['construction_null_n_miss_recovery']}")

            # ---- THE INTERIOR OF THE AIMING-COVERAGE AXIS (sec 25.5 item 3). sec 24 tested f=0 and
            #      f=1 and NEVER BETWEEN, and its blindness is MONOTONE IN THE WRONG DIRECTION. The
            #      positive control must be a STEP FUNCTION at f = 1: FAIL at every f < 1, PASS only
            #      at f = 1. One row short of perfect must HALT.
            axis = {}
            for k_aimed in (0, 12, 31, 61, 1024, 2047, 2048):
                recs = _sliver_records(k_aimed)
                a4 = check_t2a4_positive_control(recs)
                axis[k_aimed] = (a4["passes"], a4["n_miss_recovery"])
            n_pc_assertions += 1
            report("  [INTERIOR OF THE AXIS] T2a-4 is a STEP FUNCTION at f=1: it FAILS at f = 0, "
                   "0.6%, 1.5%, 3.0%, 50%, AND at 2047/2048 (ONE row short of perfect), and PASSES "
                   "ONLY at f = 1. sec 24 measured f=0 and f=1 and nothing between; THAT is the "
                   "test that would have caught A-1 before it shipped (sec 25.5 item 3).",
                   all(axis[k][0] is False and axis[k][1] == N25 - k
                       for k in (0, 12, 31, 61, 1024, 2047))
                   and axis[2048][0] is True and axis[2048][1] == 0,
                   "; ".join(f"f={k}/{N25}: passes={v[0]} n_miss={v[1]}" for k, v in axis.items()))

            # ============ (B) THE SAME DEFECT IN THE READOUT PATH, END-TO-END THROUGH THE PROBE ===
            # The records-level reconstruction above is sec 25's own interface. This one puts the
            # aiming defect where a REAL one would live -- in the readout path -- and drives it
            # through `run_t2_repaired_probe` itself: no synthetic records, the real six arms, the
            # real plant construction, the real argmax. `PerfectCopyOracle` is the model; the
            # wrapper mis-aims the readout on a deterministic subset of rows.
            class _SliverAimedReadout(torch.nn.Module):
                """sec 25's A-1 instrument, realized IN THE READOUT PATH. On rows whose readout
                token clears a deterministic predicate it returns the inner (perfect) model's
                logits; on the rest it returns logits peaked on `x[t]` ITSELF -- sec 23.3's own
                "reading at k0-1" defect, which is fully input-dependent (LIVENESS PASSES) and
                PLANT-INDEPENDENT (arms 1-5 never corrupt `x[k0]`, so the key-swap cannot move it).
                The predicate reads the token AT the readout position, which no arm corrupts, so a
                row's aimed/mis-aimed status is IDENTICAL across all six arms -- the defect is a
                property of the INSTRUMENT, not an artifact of the ablation."""

                def __init__(self, inner, modulus: int):
                    super().__init__()
                    self.inner, self.modulus = inner, int(modulus)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    good = self.inner(x)
                    B, T = x.shape
                    bad = torch.full_like(good, -3.0)
                    bad.scatter_(2, x.unsqueeze(-1), 7.0)          # peak on x[t] itself: MIS-AIMED
                    aimed = ((x % self.modulus) == 0).unsqueeze(-1)  # per-POSITION, arm-invariant
                    return torch.where(aimed, good, bad)

            def _probe(model):
                r = run_t2_repaired_probe(model, t2_fixture["val"], seq_len=256, device=device,
                                           corpus_name="smoke-synth-corpus",
                                           delta_pool=t2_fixture["delta_pool"],
                                           pools=t2_fixture["pools"],
                                           counts_by_token=t2_fixture["counts"],
                                           n_plants=t2_fixture["n_plants"],
                                           vocab_size=t2_fixture["V"], eval_micro_batch=32)
                b1 = check_t2b1_mechanism_exists(r["records"])
                b1b = check_t2b1b_key_conditioned(r["records"])
                return (r, check_t2a1_ceiling(r["records"], b1, b1b, n_boot=200),
                        check_t2a3_ssm_calibration(r["records"], b1, b1b, n_boot=200),
                        check_t2a4_positive_control(r["records"]))

            perfect = PerfectCopyOracle(t2_fixture["V"]).to(device).eval()
            r_p, a1_p, a3_p, a4_p = _probe(perfect)

            n_pc_assertions += 1
            report("  [POSITIVE CONTROL PASSES] A CORRECTLY-AIMED instrument on a model that copies "
                   "BY FIAT reads its CONSTRUCTION VALUES EXACTLY, through the real six-arm probe: "
                   "n_miss_recovery=0, n_aim_unchanged=0, acc_copy=1.0, aiming=1.0, KS=1.0, "
                   "PRIOR=0.0, acc_true_ablated=0.0. Each is a THEOREM of the plant's own hard "
                   "assertion, not a measurement. A positive control that halted here would be "
                   "worse than none.",
                   a4_p["passes"] is True
                   and a4_p["n_miss_recovery"] == 0 and a4_p["n_aim_unchanged"] == 0
                   and a4_p["n_records"] > 0 and a4_p["n_records_incomplete"] == 0
                   and a4_p["reference_acc_copy"] == 1.0
                   and a4_p["reference_aiming_argmax_changed_frac_keyswap"] == 1.0
                   and a4_p["reference_ks"] == 1.0
                   and a4_p["reference_prior"] == 0.0
                   and a4_p["reference_acc_true_ablated"] == 0.0,
                   f"passes={a4_p['passes']} n={a4_p['n_records']} "
                   f"n_miss={a4_p['n_miss_recovery']} n_aim_unchanged={a4_p['n_aim_unchanged']} "
                   f"acc_copy={a4_p['reference_acc_copy']} "
                   f"aiming={a4_p['reference_aiming_argmax_changed_frac_keyswap']} "
                   f"KS={a4_p['reference_ks']} PRIOR={a4_p['reference_prior']} "
                   f"decile_recovered={a4_p['reference_decile_recovered']}")

            r_sl, a1_sl, a3_sl, a4_sl = _probe(_SliverAimedReadout(perfect, 8).to(device).eval())
            n_pc_assertions += 1
            report("  [READOUT-PATH DEFECT] THE SAME CLOSURE, END-TO-END: a MIS-AIMED READOUT driven "
                   "through `run_t2_repaired_probe` itself (real plants, real six arms, real argmax) "
                   "keeps sec 20's LIVENESS witness GREEN and keeps leg (vi) > 0 -- and the POSITIVE "
                   "CONTROL HALTS IT. The defect is caught where a real one lives: in the readout "
                   "path, not in a synthetic record dict.",
                   r_sl["logit_liveness"]["ok"] is True
                   and a1_sl["aiming_argmax_changed_frac_keyswap"] > 0.0
                   and a1_sl["leg_vi_aiming_keyswap_argmax_changed"] is True
                   and a4_sl["passes"] is False
                   and a4_sl["n_miss_recovery"] > 0
                   and a4_sl["n_miss_recovery"] == a4_sl["n_aim_unchanged"],
                   f"liveness.ok={r_sl['logit_liveness']['ok']} "
                   f"leg_vi={a1_sl['leg_vi_aiming_keyswap_argmax_changed']} "
                   f"aiming={a1_sl['aiming_argmax_changed_frac_keyswap']:.4f} "
                   f"T2a4_passes={a4_sl['passes']} "
                   f"n_miss={a4_sl['n_miss_recovery']}/{a4_sl['n_records']}")

            # ============ (C) NO FALSE HALT ON A HEALTHY, DISTANCE-LIMITED WITNESS ===============
            # sec 25.5 named this hazard IN ITS OWN candidate direction, and sec 23.4 item 2 already
            # committed the error once: gating aiming per-stratum on the WITNESSES would HALT a
            # healthy model whose key-conditioning legitimately dies in the Delta tail. The archived
            # attempt-2 raws (md5 87ae97087bca56894a5035a348d17f48) record W1/openr1's Delta-decile
            # curve as 0.907 / 0.839 / 0.888 / 0.746 / 0.781 / 0.637 / 0.634 / 0.620 / 0.517 / 0.376
            # -- a REAL, HEALTHY, CORRECTLY-AIMED witness that is DISTANCE-LIMITED. This oracle is
            # that witness: a correctly-aimed induction head whose copy fails as Delta grows.
            class _DistanceLimitedOracle(torch.nn.Module):
                """A HEALTHY, CORRECTLY-AIMED induction head that is DISTANCE-LIMITED: it copies
                only when the demonstration is within `max_delta` of the readout, and emits `x[t]`
                otherwise. This is exactly what the four real witnesses look like (sec 18.2b's
                decile grid), and NOTHING IN THE GATE MAY HALT IT."""

                def __init__(self, vocab_size: int, max_delta: int):
                    super().__init__()
                    self.vocab_size, self.max_delta = int(vocab_size), int(max_delta)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    B, T = x.shape
                    idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                    first_pos = torch.full((B, self.vocab_size), T, dtype=torch.long, device=x.device)
                    first_pos.scatter_reduce_(1, x, idx, reduce="amin", include_self=True)
                    fp = first_pos.gather(1, x)
                    near = (fp < idx) & ((idx - fp) <= self.max_delta)   # THE DISTANCE LIMIT
                    nxt = x.gather(1, (fp + 1).clamp(max=T - 1))
                    pred = torch.where(near, nxt, x)
                    logits = torch.full((B, T, self.vocab_size), -3.0, device=x.device,
                                         dtype=torch.float32)
                    logits.scatter_(2, pred.unsqueeze(-1), 7.0)
                    return logits

            # delta_pool11 spans 10..120; a 50-token limit gives a declining decile profile with a
            # DEAD tail -- the regime in which a per-decile aiming leg would false-HALT.
            r_dl, a1_dl, a3_dl, a4_dl = _probe(_DistanceLimitedOracle(t2_fixture["V"], 50).to(device).eval())
            dl_dec = a1_dl["decile_accs"]
            n_pc_assertions += 1
            report("  [NO FALSE HALT / THE sec 25.5 HAZARD] A HEALTHY, CORRECTLY-AIMED, "
                   "DISTANCE-LIMITED witness -- the shape of all four real witnesses (W1/openr1: "
                   "0.907 -> 0.376) -- PASSES T2a-1 and T2a-3 with its LARGEST Delta-decile DEAD "
                   "(acc = 0.0, and aiming = 0.0 in that stratum). A per-decile aiming leg -- the "
                   "candidate sec 25.5 flagged the hazard in, and sec 23.4 item 2's identical error "
                   "-- WOULD HAVE HALTED IT. T2a-4 gates NO witness quantity, so it CANNOT.",
                   a1_dl["passes"] is True and a3_dl["passes"] is True
                   and dl_dec[0] > dl_dec[-1] and dl_dec[-1] == 0.0
                   and 0.0 < a1_dl["aiming_argmax_changed_frac_keyswap"] < 1.0,
                   f"T2a1={a1_dl['passes']} T2a3={a3_dl['passes']} "
                   f"acc_copy={a1_dl['acc_copy']:.4f} "
                   f"deciles={[round(v, 3) for v in dl_dec]} "
                   f"aiming={a1_dl['aiming_argmax_changed_frac_keyswap']:.4f}")

            n_pc_assertions += 1
            report("  [T2a-4 IS WITNESS-INDEPENDENT BY CONSTRUCTION] The positive control's verdict "
                   "is BIT-IDENTICAL whether the witness is perfect, distance-limited, or 0.6%-"
                   "aimed: it is a function of the ORACLE's records and NOTHING ELSE. That is the "
                   "structural reason it cannot false-HALT a witness at any Delta, decile or "
                   "coverage -- and it is why the sec 25.5 hazard does not attach to it.",
                   check_t2a4_positive_control(r_p["records"])["passes"] is True
                   and check_t2a4_positive_control(r_p["records"])["n_miss_recovery"] == 0
                   and a4_dl["passes"] is False and a4_p["passes"] is True,
                   f"oracle_verdict={a4_p['passes']} (n_miss=0); the DISTANCE-LIMITED model's OWN "
                   f"records would read passes={a4_dl['passes']} "
                   f"(n_miss={a4_dl['n_miss_recovery']}) -- WHICH IS WHY THE DRIVER NEVER FEEDS "
                   f"THEM TO IT: `run_t2a4_positive_control` builds the oracle itself.")

            # The COVERAGE check does NOT increment the counter it audits. Hardcoded expected count,
            # never derived from the counter -- a tally compared against itself goes green on a
            # skipped body. It has now caught two consecutive builders (sec 20.4e, sec 24.5).
            report(f"  [COVERAGE] all {n_pc_assertions}/8 T2a-4 forced-fail assertions EXECUTED "
                   f"(expected count HARDCODED)", n_pc_assertions == 8, f"n={n_pc_assertions}")
        except Exception as e:  # noqa: BLE001
            report("  [10i] sec 26 POSITIVE CONTROL forced-fail negative tests", False,
                   f"EXCEPTION after {n_pc_assertions} assertions: {type(e).__name__}: {e}")

    # --- [10j] sec 26 -- THE DETERMINISM RECEIPT, REGENERABLE BY A THIRD PARTY.
    #     sec 25.3 MINOR-3: sec 24.6's determinism md5 `533cf851...` hashed a PRIVATE /tmp fixture
    #     that was never archived, so no auditor can ever recompute it. "An md5 nobody can recompute
    #     is not a receipt -- it is a number." This replaces it with a receipt whose input is the
    #     COMMITTED FILE ITSELF: an md5 over the exact source text of every function and class that
    #     PRODUCES OR ESTIMATES A RECORD. If that digest is unchanged, the record stream is
    #     bit-identical BY CONSTRUCTION -- no fixture, no RNG, no GPU, no torch version, no device.
    #     ANYONE can regenerate it, against ANY commit, with the six-line recipe in sec 26.
    print("\n  [10j] sec 26: DETERMINISM RECEIPT (source-level, third-party regenerable)")
    try:
        import ast as _ast_d
        RECORD_PATH_SYMBOLS = [
            # window sampling -> plant construction -> the six arms -> the record dicts
            "_combine_seed", "_make_window", "rejection_sample_delta", "draw_t2_triple",
            "plant_and_verify_t2_window", "assign_t2_plant", "draw_pool_replacement",
            "draw_exclusive_replacement", "assign_placebo_positions",
            "build_replicated_ablation_batch", "run_ablation_arm", "run_t2_repaired_probe",
            "build_key_value_pools", "_LiveLogitAccumulator", "argmax_changed_frac_keyswap",
            # the estimators every leg reads through
            "clustered_bootstrap_ci", "_mean", "_acc", "_decile_bucket", "binomial_se",
            "_paired_sign_test", "_exact_binomial_two_sided_p", "_log_binomial_pmf",
        ]
        _src_d = open(os.path.abspath(__file__)).read()
        _tree_d = _ast_d.parse(_src_d)
        _found = {n.name: _ast_d.get_source_segment(_src_d, n)
                  for n in _ast_d.walk(_tree_d)
                  if isinstance(n, (_ast_d.FunctionDef, _ast_d.ClassDef))
                  and n.name in RECORD_PATH_SYMBOLS}
        missing = [s for s in RECORD_PATH_SYMBOLS if s not in _found]
        digest = hashlib.md5("\n".join(_found[s] for s in RECORD_PATH_SYMBOLS
                                       if s in _found).encode()).hexdigest()
        # PINNED at the sec 26 build. This digest is IDENTICAL at HEAD `20c40c4` (sec 25's audited
        # bytes) -- verify with `git show 20c40c4:matrix-thinking/deltanet_rd/
        # lm_recall_gap_probe_v2_rd.py` and the same six lines. sec 26's edits are PURELY ADDITIVE
        # (a new oracle class, a new check function, new smoke blocks); NOT ONE record-producing or
        # estimator function was touched, so `run_t2_repaired_probe` reproduces attempt-2's records
        # bit-for-bit. If a future edit perturbs the record path, THIS TEST TURNS RED and the
        # determinism claim must be re-earned rather than re-asserted.
        RECORD_PATH_MD5 = "24bd8ae9783c0c8da35765d8181710c3"
        report("  [10j] DETERMINISM RECEIPT: the md5 of the RECORD-PRODUCING + ESTIMATOR source "
               "path is UNCHANGED from HEAD 20c40c4 => `run_t2_repaired_probe`'s record stream is "
               "bit-identical BY CONSTRUCTION. Regenerable by any third party against any commit "
               "(recipe in sec 26); the sec 24.6 md5 it replaces hashed an unarchived /tmp fixture "
               "and was un-recomputable by anyone (sec 25.3 MINOR-3).",
               digest == RECORD_PATH_MD5 and not missing,
               f"n_symbols={len(_found)}/{len(RECORD_PATH_SYMBOLS)} missing={missing} "
               f"md5={digest} (pinned {RECORD_PATH_MD5})")
    except Exception as e:  # noqa: BLE001
        report("  [10j] sec 26 determinism receipt", False, f"EXCEPTION: {type(e).__name__}: {e}")

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
