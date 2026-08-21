"""NCR REAL-LM WAVE-1 CALIBRATION RUNNER -- sec G3-B6 (build agent, 2026-07-17).
matrix-thinking/NCR_REAL_LM_DESIGN.md sec G3-B5 (COORDINATOR ADJUDICATION of
the sec G3-B4 audit): implements the TWO-ARM design + frozen attribution rule
that closes DEFECT G3-B4-1 [FATAL] (the ratified graft's read is additive/
non-bottlenecked, so a FAIL cannot be localized to the NCR head without a
backbone-only control). This is a REAL TRAINING DRIVER (step loop, LR
schedule, checkpoint/resume, ceiling-gated, blind), not a smoke -- it WRAPS
the already-audited graft (sec G3-B4 dimension 1: PASS, no defects in the
wiring itself) without reopening any of its ratified logic.

============================================================================
WHAT THIS FILE DOES NOT DO (scope discipline, per the build brief: "build
EXACTLY [sec 6.2's small Phase-1 calibration cell] spec, do not invent a
full sweep"):
============================================================================
  - No real+synthetic batch-level mixing (sec 5.2 Option 1) -- sec G3-B3's
    own build already deferred this ("out of the BUILD+SMOKE... scope"); this
    wave inherits that deferral rather than silently expanding scope. Every
    document trained/evaluated here is a pure grammar_rd synthetic Task-1
    document (disclosed, not fabricated as literal design text).
  - Task 1 / abelian construction ONLY (K=24, d_ncr=25, the audited Wave-1
    pin) -- Task 2 (non-solvable-group word problem, sec 3.2) is NOT built
    here; it is gated on the SEPARATE bridge cell (sec 6.2 Phase 0b), per
    the coordinator's own task scope.
  - The "flat-vector ablation" arm from sec 6.2's ORIGINAL Phase-1 spec is
    NOT one of this cell's two arms. Sec G3-B5 REPLACES that comparison for
    THIS calibration with (full-graft, backbone-only/read-ablated) --
    the FATAL's own required control, not sec 6.2's original arm list.
  - The teacher-force-operator / mlp-adapter / mlp-logits ablation arms
    (sec G3-B2's OTHER pre-wired flags, already smoke-verified in
    ncr_lm_wave1_smoke.py items 10-11) are FAIL-diagnosis tools for AFTER a
    result is read, not part of this calibration's own two arms.
  - This runner does NOT compute or print a WIN/PARTIAL/NULL/FAIL verdict
    anywhere (blind discipline, matching NCR_ORTHO_WRITE.md sec 6's own
    "the runner emits NO verdict" convention) -- see the BLIND DISCIPLINE
    note below.

============================================================================
TWO-ARM WIRING (sec G3-B5's own required construction)
============================================================================
Both arms use the IDENTICAL architecture (backbone + ncr_head + integ, same
param counts, BIT-IDENTICAL initial weights -- see build_two_arms() below)
and train on the EXACT SAME per-step batch (one shared data generator,
consumed once per step, its output batch object reused for both arms'
forward/backward/opt-step calls) -- "same data, same seed, same everything
else" per sec G3-B5. The ONLY difference:
  - full_graft:      o_injected = o_raw   (the real NCR read, unmodified)
  - backbone_only:    o_injected = torch.zeros_like(o_raw)   (read-ablated)
`torch.zeros_like` creates a FRESH tensor with no autograd edge back to
o_raw's own computation graph -- so in the backbone_only arm, NO gradient
reaches the write adapters or the NCR encoder at all (they stay frozen at
their shared random init for the entire run); the read literally
contributes ZERO to backbone_only's logits, verified by an EXACT (not
tolerance-based) equality assertion, both before AND after training
(assert_read_ablation_is_exact_zero, called from run_two_arm_cell). This is
the "wiring already supports it trivially" property sec G3-B5 names -- no
edit to ncr_lm_wave1_smoke.py's own NCRIntegration/ncr_lm_forward was
needed or made; this file composes the SAME audited public building blocks
(integ.extract_kv / integ.query_key / ncr_head.encode / nm.binexp_read /
integ.inject_and_logits_last) into its own read-ablatable wrapper,
ncr_lm_forward_ablatable(), leaving the audited ncr_lm_forward() untouched.

============================================================================
ATTRIBUTION FIELDS (registered in the results JSON, sec G3-B5's own
required registration "so the blind assessor applies it mechanically")
============================================================================
Two DISTINCT metrics are recorded per arm, per depth (a genuine build-time
interpretation of "recovered_frac@0.9 ... for BOTH arms", disclosed here,
not silently invented):
  (a) recovered_frac@0.9 -- the AUDITED ncr_lm_wave1_smoke.recovered_frac_at_09
      metric, computed identically for both arms on o_raw (the NCR
      pathway's own internal read, BEFORE the ablation override). For
      full_graft, o_raw IS the injected read (real gradient reaches it).
      For backbone_only, o_raw's write/encode pathway receives NO gradient
      (see above) -- so this number tracks an UNTRAINED, frozen-at-init
      NCR head's read quality throughout the run, a genuine null baseline
      for the recovery metric itself. THE PRIMARY SIGNAL (sec G3-B5:
      "recovery GAP (full-graft - backbone-only), at DEEP composition
      depth") is full_graft.recovered_frac@0.9(o_raw) minus
      backbone_only.recovered_frac@0.9(o_raw), reported at every deep-ladder
      depth AND as a single deepest-rung number (h=61).
  (b) answer_accuracy -- argmax(logits) == answer_token, computed on each
      arm's ACTUAL logits (i.e. backbone_only's logits use o_injected=0, so
      this measures whether h_q ALONE solves the task). This is what sec
      G3-B4's own attribution-rule PROSE literally names ("its answer
      accuracy is materially below the full graft's -- the NCR read is
      demonstrably load-bearing") -- the blind assessor uses THIS field,
      not (a), to evaluate the attribution rule's own precondition.
Both fields are written for in-distribution h in {1,2,3} AND the deep
ladder h in {5,12,20,29,40,61} (sec 3.1's own eval grid, re-verified sound
for K=24 by _assert_ladder_sound below -- sec 3.1's own residue check was
run at K in {15,32}, not 24, so this is NOT assumed transferred).
The frozen sec G3-B5 rule TEXT is written verbatim into the JSON
(attribution.frozen_rule_text) so a blind assessor applies it against these
raw numbers without needing this script or the design doc open side by side.

============================================================================
BLIND DISCIPLINE (sec G3-B5 build brief: "runner writes results, does not
print metric values to the launcher")
============================================================================
Precedent: NCR_ORTHO_WRITE.md sec 6 / ncr_ortho_fallback_stage0_v3.py's own
run_disc_cell/run_primary_cell -- both print per-step TRAIN LOSS to stdout
(operational liveness/divergence telemetry) but NEVER print an eval metric
(recovered_frac, cosine, accuracy) to stdout; those go ONLY into the
results JSON, unaccompanied by any verdict. This file follows the identical
split: step/loss/elapsed/status ARE printed (needed to confirm the run is
alive and not diverging, and to read Phase-0's own throughput numbers,
which are NOT part of the interpretive signal); recovered_frac@0.9,
answer_accuracy, and the attribution GAP are NEVER printed -- only written
to --out. No PASS/FAIL/WIN/PARTIAL/NULL string appears anywhere in this
file.

============================================================================
sec G3-B17 DIRECT-READ-SUPERVISION AUX LOSS (2026-07-19, coordinator-designed
fix for the sec G3-B16 WRITE-LEARNING gap: the make-or-break came back
UNINTERPRETABLE -- both arms floored at the answer-marginal after 20K steps
of CE-only INDIRECT signal to the encoder, ~32x under the standalone
free-write toy's own convergence budget, AND missing the toy's own DIRECT
cosine read-loss that let it converge)
============================================================================
NEW flag --aux-read-loss-weight (default 0.0 = OFF, reproduces the EXACT
pre-G3-B17 CE-only loss path -- no aux tensor is even constructed at that
default, see compute_arm_losses's own docstring for the byte-identical
guarantee). full_graft ARM ONLY -- backbone_only's o_raw is the untrained,
frozen-at-init null baseline sec G3-B5's attribution rule depends on;
training it toward a target would corrupt that control, so it stays
CE-only regardless of this flag's value. When weight > 0.0: total_loss =
ce_loss + weight * aux_loss, where aux_loss = mean(1 - cosine_similarity(
o_raw, target_o)) and target_o = entity_adapter(embed(answer_token))
.detach() -- the SAME re-based target recovered_frac_at_09/
cosine_and_recovered_frac use (ncr_lm_wave1_smoke.py ~512, this file's own
cosine_and_recovered_frac above), computed WITH grad enabled on the o-path
and explicitly DETACHED on the target-path (trains the encoder/read toward
the target, never the target's own entity_adapter/embed call -- see
aux_read_supervision_loss's docstring for why detach matters). See
aux_read_supervision_loss() and compute_arm_losses() below --
compute_arm_losses is the ONE shared per-arm forward+loss computation both
the training loop (run_two_arm_cell) and this build's own real-CUDA smoke
(ncr_lm_wave1_aux_smoke.py) call, so the smoke exercises the actual
training code path, not a hand-copied reimplementation that could drift
from it.

============================================================================
sec G3-B20 ORTHOGONALITY REGULARIZATION (2026-07-20, coordinator-designed
fix for the sec G3-B19-diagnosed gap: the aux read-supervision loss above
DID teach the encoder to write -- full_graft's read moved to a STABLE
cos~0.57-0.65 across every depth h=1..61 -- but that operator is
directionally-right, NOT a clean rotation, so Z^h (binexp_read's own
repeated-squaring composition) accumulates error and never clears the
~0.9 cosine bar exact recovery needs)
============================================================================
NEW flag --ortho-reg-weight (default 0.0 = OFF, reproduces the EXACT
pre-G3-B20 loss path -- no ortho tensor is even constructed at that
default, independent of --aux-read-loss-weight's own value, see
compute_arm_losses's own docstring for the byte-identical guarantee).
full_graft ARM ONLY, same rationale as --aux-read-loss-weight --
backbone_only's Z lives in a separate, frozen-at-init ncr_head instance
sec G3-B5's attribution rule depends on staying untouched. When
weight > 0.0: total_loss += weight * ortho_loss, where ortho_loss =
mean_B(||Z^T Z - I_d||_F^2) / d^2 (d=d_ncr=25, NORMALIZED -- see
ortho_regularization_loss's own docstring for the disclosed weight-balance
rationale) and Z is ncr_head.encode's own (B,d,d) output -- the SAME
tensor ncr_lm_forward_ablatable already returns, no extra forward pass.
Orthogonal matrices compose EXACTLY under powering (Q^h stays orthogonal
for any h), so this pushes the encoder's write DIRECTLY toward the regime
where binexp_read's O(log h) repeated-squaring accumulates zero error at
any depth -- complementing the aux loss above, which only pressures Z
INDIRECTLY through the read's own cosine to the target. See
ortho_regularization_loss() and compute_arm_losses() below.

============================================================================
sec G3-B31 CONTRASTIVE-AUX RE-SPEC (2026-07-29, build agent, per sec G3-B30
design + coordinator amendments A1/A2/A3, novelty-gate-cleared sec
G3-B30.1): the sec G3-B26 READ-COLLAPSE finding showed the sec G3-B17
bare-cosine aux (aux_read_supervision_loss below) has a DEGENERATE OPTIMUM
-- `1 - cos(o, target.detach())` is minimized just as well by collapsing
the ENTIRE target space to one point (entity_adapter maps every entity to
~the same vector, measured pairwise cos 0.9960) as by learning a genuinely
discriminative read; CE never penalizes this because the injected read
still reaches the decode head through `h_q`, which alone can float the
answer-marginal. sec G3-B30's fix REPLACES/AUGMENTS the aux slot with a
24-way discriminative (InfoNCE-style) term over the K=24 in-document
adapted entity targets -- under EXACT collapse all 24 candidates are
identical, so the discriminative term reads log(24) (chance) with ZERO
gradient toward that point: the degenerate optimum is removed
STRUCTURALLY, not soft-penalized (the same house preference the sec
G3-B20 ortho term already embodies for a different degenerate-optimum
class). Precedent: DELTANET_REALDATA_DESIGN.md sec 14.4 (the SAME fix
family, "option (b), in a retained-cosine form", for the SAME collapse
pathology class in a different fast-weight harness -- sec G3-B30's
internal-archive gate leg; see contrastive_read_supervision_loss()'s own
docstring for the sec A2 saddle-corrected mechanism claim).

NEW flags (--mode calibration only, full_graft arm ONLY -- same scoping
as --aux-read-loss-weight/--ortho-reg-weight above; backbone_only's o_raw
stays the untrained/read-ablated null baseline regardless):
  --aux-loss-type {cosine, contrastive, contrastive+cosine}
      default "cosine" = the EXACT sec G3-B17 pre-existing behavior
      (aux_loss = aux_read_supervision_loss(...), byte-identical call,
      see compute_arm_losses's own docstring -- this flag's default is
      NOT a new code path, it IS the old one). "contrastive": aux_loss =
      contrastive_read_supervision_loss(...) (pure 24-way InfoNCE, ALL 24
      targets detached). "contrastive+cosine" (sec G3-B30 amendment A1,
      the PRIMARY/companion-B arms' loss form): aux_loss = 0.5*L_ctr +
      0.5*L_cos (both frozen-detached-target) -- see
      compute_arm_losses's own docstring for the exact weight-composition
      ruling (aux_read_loss_weight is applied ON TOP of this 0.5/0.5
      combination, matching sec G3-B24's total-aux-gradient-scale parity,
      documented prominently there since sec G3-B30's own prose is
      ambiguous about which level the 0.5/0.5 lives at).
  --contrastive-temperature (float, default 0.07, sec G3-B30's own
      pre-registered value; sec G3-B30's disclosed fallback 0.03 is
      config-only, set this flag explicitly to use it): the softmax
      temperature T in L_ctr = -log[ exp(cos(o,T_true)/T) /
      sum_{k=1..24} exp(cos(o,T_k)/T) ].
  --freeze-entity-adapter (store_true, default OFF): freezes
      integ.entity_adapter (requires_grad_(False), excluded from the
      optimizer's param groups) in BOTH arms -- the sec G3-B28 frozen-init
      control's own measured basis (pairwise cos 0.065-0.081, no
      collapse) for sec G3-B30's PRIMARY/companion-A cells. See
      assert_entity_adapter_grad_none() for the per-step "has teeth"
      check and build_arm()/build_optimizer()'s own docstrings for the
      construction-time mechanics.

See contrastive_read_supervision_loss(), assert_entity_adapter_grad_none(),
assert_read_target_write_key_same_op(), and compute_arm_losses() below.

Run (box only -- chunk_delta_rule has no CPU path):
  python3 ncr_lm_wave1_runner.py --mode phase0-timing --device cuda \
      --out results/phase0_timing.json
  python3 ncr_lm_wave1_runner.py --mode calibration --device cuda \
      --steps 20000 --ceiling-gpuh <FROM PHASE-0, contended-priced> \
      --ckpt-dir results/ckpts --out results/wave1_calib_K24_s0.json
  python3 ncr_lm_wave1_runner.py --mode calibration --device cuda \
      --aux-loss-type contrastive+cosine --freeze-entity-adapter \
      --steps 20000 --ceiling-gpuh <...> --ckpt-dir results/ckpts \
      --out results/mob_g3b31_primary_s0.json   # sec G3-B31 PRIMARY cell
  python3 ncr_lm_wave1_runner.py --mode smoke --device cuda \
      --out results/runner_smoke.json   # this build's OWN required smoke
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The AUDITED graft module (sec G3-B4 dimension 1: PASS, no wiring defects),
# reused as a LIBRARY -- every constant/builder/function below is imported,
# never reimplemented, except the smoke-5 fix (disclosed in that file's own
# diff) and this file's own NEW read-ablation wrapper (composed from the
# audited public building blocks, see module docstring above).
import ncr_lm_wave1_smoke as graft
import ncr_models as nm                      # noqa: E402 (verbatim; binexp_read)
import grammar_rd as gr                       # noqa: E402 (verbatim; sample_batch_rd via graft helpers)
from lm_pretrain_rd import get_lr             # noqa: E402 (verbatim; the ONE reusable trainer utility --
                                                # not lm_pretrain_rd.py's own train() loop, which is
                                                # coupled to real-corpus doc_offsets/geo3 diagnostics
                                                # this synthetic-only two-arm cell does not use)

VOCAB_SIZE = graft.VOCAB_SIZE
RUNG1_BACKBONE = graft.RUNG1_BACKBONE
K_NCR, D_NCR, H_NCR = graft.K_NCR, graft.D_NCR, graft.H_NCR
NCRIntegration = graft.NCRIntegration
build_backbone = graft.build_backbone
build_ncr_head = graft.build_ncr_head
build_grammar_pools_and_cfg = graft.build_grammar_pools_and_cfg
build_task1_document = graft.build_task1_document
recovered_frac_at_09 = graft.recovered_frac_at_09

# ---------------------------------------------------------------------------
# K-SCALING PATCH R1 (NCR_KSCALING_DESIGN.md sec 5.4). RUNNER_TAG is bumped so
# a K-scaling checkpoint can never be silently resumed by, or confused with,
# the pinned K=24 wave (load_checkpoint asserts on this field).
# DEEP_LADDER is no longer a carried literal: the pinned (5,12,20,29,40,61)
# CRASHES the soundness guard at K in {12,20,28} and SILENTLY loses a rung to
# a residue collision at K in {16,24,32} (gate memo leg 1). It is replaced by
# kscaling_config's per-K residue-verified ladder.
# ---------------------------------------------------------------------------
import kscaling_config as KS                        # noqa: E402

RUNNER_TAG = "ncr_kscaling_runner_v1"
TRAIN_HOPS = tuple(KS.TRAIN_HOPS)                   # sec 3.1 Task-1 train range, verbatim
DEEP_LADDER = tuple(KS.DEEP_LADDER)                 # per-K, residue-verified (kscaling_config)
FIXED_DIST_PROBE = KS.FIXED_DIST_PROBE              # labelled fixed-effective-distance control
H_TOP = KS.H_TOP                                    # the PRIMARY readout depth for this K
PINNED_K24_LADDER = KS.REFERENCE_K24_LADDER         # kept ONLY for the smoke's negative test
CONTENDED_MULTIPLIER = 3.3                         # sec G3-B1 item 2 / sec G3-B3 launch-command note, established precedent
LOG_EVERY = 25

ATTRIBUTION_RULE_TEXT = (
    "sec G3-B5 (COORDINATOR ADJUDICATION of the sec G3-B4 audit, 2026-07-17): "
    "a Phase-1 Gate-0 in-distribution-recovery FAIL may be attributed to \"the "
    "NCR head can't train through a real LM\" ONLY IF the backbone-only arm "
    "does NOT itself solve the task in-distribution (i.e., its answer "
    "accuracy is materially below the full graft's -- the NCR read is "
    "demonstrably load-bearing). If backbone-only already solves it, the "
    "calibration is UNINTERPRETABLE and the graft must be re-bottlenecked "
    "(e.g. read-only decode / harder P=1 bottleneck) BEFORE main-wave GPU. "
    "PASS (NCR head trains AND is load-bearing) = full-graft recovers deep "
    "composition (recovered_frac@0.9 >= the sec 6.2 Gate-0 bar at deep h) "
    "AND materially exceeds backbone-only there. PRIMARY interpretable "
    "signal = the recovery GAP (full-graft - backbone-only), at DEEP "
    "composition depth."
)


# ---------------------------------------------------------------------------
# CLAUDE.md mod-K guard, ENFORCED (not merely inherited): sec 3.1's own
# residue-soundness citation was verified at K in {15,32} for this exact
# ladder -- K_NCR here is 24 (sec N2.1's Wave-1 pin), a THIRD value, so this
# is re-checked fresh, not assumed transferred.
# ---------------------------------------------------------------------------
def _assert_ladder_sound(ladder: tuple, k: int, train_hops: tuple) -> None:
    """K-SCALING PATCH R2: delegates to kscaling_config.assert_ladder_sound,
    which keeps BOTH of the pinned checks (identity residue; train-residue
    collision) and ADDS the one the pinned guard is missing -- PAIRWISE
    residue distinctness -- plus strictly-increasing depth and a matched
    squaring profile. The pinned guard passes ladders in which two rungs
    measure the SAME ground truth (it does so on the K=24 ladder of record:
    29 == 5 mod 24). The smoke's negative test proves this version REJECTS
    such a ladder."""
    KS.assert_ladder_sound(ladder, k, train_hops)


_assert_ladder_sound(DEEP_LADDER, K_NCR, TRAIN_HOPS)
assert D_NCR == K_NCR + 1, (K_NCR, D_NCR)
assert FIXED_DIST_PROBE % K_NCR == KS.FIXED_DIST_RESIDUE
assert H_TOP == DEEP_LADDER[-1] and H_TOP % K_NCR == K_NCR // 2


# ---------------------------------------------------------------------------
# IO helpers (pattern reused verbatim from matrix-thinking/ncr/run_ncr.py's
# own git_commit/atomic_write_json/stop_requested -- not imported directly
# since that module drags in unrelated heavy deps; the three functions
# themselves are tiny and self-contained).
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:
        return "UNKNOWN"


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def atomic_torch_save(path: str, obj) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def stop_requested(stop_file: str) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)


# ---------------------------------------------------------------------------
# Read-ablation forward -- composed from AUDITED building blocks ONLY
# (integ.extract_kv / integ.query_key / ncr_head.encode / nm.binexp_read /
# integ.inject_and_logits_last are the SAME functions ncr_lm_forward() in
# ncr_lm_wave1_smoke.py calls; this wrapper does NOT edit that audited
# function, so its wiring stays byte-identical to what sec G3-B4 reviewed).
# ---------------------------------------------------------------------------
def ncr_lm_forward_ablatable(backbone, ncr_head, integ: NCRIntegration, batch: dict, read_ablate: bool,
                              teacher_force: bool = False):
    """Returns (logits, o_raw, o_injected, hidden, Z, keys_v, values_v).
    read_ablate=False (full_graft): o_injected is o_raw (same tensor, real
    gradient flows through it to the write pathway).
    read_ablate=True (backbone_only): o_injected = torch.zeros_like(o_raw) --
    a FRESH tensor with NO autograd edge to o_raw's own graph, so the write
    adapters/encoder receive ZERO gradient and the read contributes EXACTLY
    zero to the logits (verified by assert_read_ablation_is_exact_zero).

    teacher_force=True (sec G3-B9 diagnostic, ported from ncr_lm_wave1_smoke.
    ncr_lm_forward's own audited teacher_force branch, smoke item 10): Z is
    built by integ.teacher_force_operator(keys_v, values_v) -- a closed-form
    least-squares fit on DETACHED key/value-adapter outputs -- instead of
    ncr_head.encode(keys_v, values_v). ncr_head's parameters therefore NEVER
    enter the autograd graph in this branch (asserted explicitly in the
    training loop, not merely assumed here). read_ablate and teacher_force
    compose orthogonally: read_ablate still controls o_injected independently
    of where Z came from (backbone_only zeros o_injected regardless).

    sec G3-B12 (ported from ncr_lm_wave1_smoke.ncr_lm_forward's own fix):
    keys_v/values_v/q_key are now extracted from RAW `input_ids` through
    `backbone.embed` + the single shared `integ.entity_adapter` (was
    contextualized `hidden` through separate key_adapter/value_adapter) --
    see graft.NCRIntegration.extract_kv/query_key docstrings for the exact
    mechanism. `hidden` is still computed (unavoidable backbone forward) and
    still used for read-injection's own tap point (RULING 2, unaffected)."""
    input_ids = batch["doc"][:, :-1]
    hidden = backbone(input_ids, return_hidden=True)
    keys_v, values_v = integ.extract_kv(input_ids, batch["key_pos"], batch["val_pos"], backbone.embed)
    if teacher_force:
        Z = integ.teacher_force_operator(keys_v, values_v)
    else:
        Z = ncr_head.encode(keys_v, values_v)
    q_key = integ.query_key(input_ids, batch["query_key_col"], backbone.embed)
    o_raw = nm.binexp_read(Z, q_key.unsqueeze(1), h=batch["hop"])["o"].squeeze(1)
    o_injected = torch.zeros_like(o_raw) if read_ablate else o_raw
    logits = integ.inject_and_logits_last(hidden, o_injected, batch["query_mark_col"], backbone.embed.weight)
    return logits, o_raw, o_injected, hidden, Z, keys_v, values_v


@torch.no_grad()
def assert_read_ablation_is_exact_zero(backbone, ncr_head, integ: NCRIntegration, batch: dict) -> float:
    """Hard EXACT-equality check (CLAUDE.md: structural correctness checks
    need exact thresholds, not tolerance): the ablated arm's logits must be
    BYTE-FOR-BYTE identical to the tied head applied to h_q alone -- proves
    the read contributes literally zero, the property sec G3-B5's whole
    control rests on."""
    logits_ablated, o_raw, o_inj, hidden, _, _, _ = ncr_lm_forward_ablatable(
        backbone, ncr_head, integ, batch, read_ablate=True)
    h_q = hidden[:, batch["query_mark_col"], :]
    logits_pure = F.linear(h_q, backbone.embed.weight)
    max_diff = (logits_ablated - logits_pure).abs().max().item()
    assert torch.equal(o_inj, torch.zeros_like(o_inj)), "o_injected is not the exact zero tensor"
    assert torch.equal(logits_ablated, logits_pure), (
        f"READ ABLATION IS NOT EXACT-ZERO: max|logits_ablated-logits_pure|={max_diff:.3e} "
        f"(expected EXACT bitwise equality) -- the backbone-only control is NOT clean, "
        f"sec G3-B5's attribution rule does not hold")
    del o_raw
    return max_diff


# ---------------------------------------------------------------------------
# mean_cos diagnostic (sec G3-B9, in response to sec G3-B8's own flagged
# instrument-ambiguity: "recovered=0 in-dist consistent with didn't-learn OR
# o_raw-recovery-instrument-mis-wired"). recovered_frac@0.9 is a THRESHOLDED
# view of read quality (cos>=0.9 or nothing); this exposes the RAW mean
# cosine alongside it so the two failure modes are visually distinguishable:
# high mean_cos with rec@0.9=0 => the threshold discarded real (sub-0.9, but
# non-trivial) signal; near-zero mean_cos => the read genuinely carries
# nothing. Re-derives the IDENTICAL target/cosine computation
# graft.recovered_frac_at_09 (ncr_lm_wave1_smoke.py, AUDITED, sec G3-B12
# RE-BASED: target = entity_adapter(RAW embed(answer_token)), o's OWN
# space -- was key_adapter(hidden at the ANSWER entity's OWN bind-clause KEY
# position), sec G3-B11 defect 3d) uses internally, byte-for-byte --
# duplicated here (not edited into that audited file, keeping its md5/audit
# status untouched, per the build brief's "do NOT disturb the audited
# two-arm path") solely to obtain the raw per-row cosine tensor that
# function computes but does not expose (it only returns the thresholded
# fraction). recovered_frac@0.9 and mean_cos below are therefore ALWAYS
# derived from the SAME cosine tensor -- guaranteed consistent by
# construction, not two independently-invented metrics that could silently
# disagree.
# ---------------------------------------------------------------------------
def cosine_and_recovered_frac(integ: NCRIntegration, embed: torch.nn.Embedding, o: torch.Tensor,
                               answer_token: torch.Tensor) -> tuple[float, float]:
    """Returns (recovered_frac@0.9, mean_cos) from ONE shared cosine tensor.
    See the module comment immediately above for why this duplicates (not
    reimplements-differently) graft.recovered_frac_at_09's own (sec G3-B12
    re-based) target. answer_token: (B,) int64, the true answer entity's OWN
    token id (batch["answer_token"]) -- embed(answer_token) is context-free,
    no position-gather through `hidden` needed any more."""
    target = integ.entity_adapter(embed(answer_token).float())
    cos = F.cosine_similarity(o, target, dim=-1)
    return (cos >= 0.9).float().mean().item(), cos.mean().item()


# ---------------------------------------------------------------------------
# sec G3-B26 discriminability instrumentation (build agent, 2026-07-29,
# dispatched by the audit's "no further training on this runner until
# eval_arm_at_hops + build_attribution carry a discriminative metric"
# BLOCKING RULE). THE FINDING this responds to: recovered_frac@0.9/mean_cos
# above are SATURATED -- they compare `o` only against the TRUE target and
# never against the 23 WRONG in-document targets, so once the trained
# entity_adapter collapses its own target space (all 24 per-document
# targets converge to pairwise cos ~0.9960, sec G3-B26), an information-free
# `o` that is literally ONE FIXED VECTOR (o pairwise cos ~1.0 across
# documents, o_var 6.95e-08, sec G3-B26) reads recovered_frac@0.9==1.0
# against EVERY document's target, indistinguishable from a genuinely
# discriminative read. This function duplicates -- byte-for-byte, not
# reimplemented differently, matching cosine_and_recovered_frac's own
# convention immediately above -- the EXACT measurement construction of the
# audit's own reference script,
# /home/nvidia/tmp_audit_g3b25/audit_readonly_discriminability.py (the
# script that FOUND the collapse), so this instrumentation is provably the
# same math the audit already trusts, not a fresh reimplementation that
# could silently diverge from it. decode_isolation_probe.py's
# read_output_stats (sec G3-B26 R1) calls this SAME function rather than
# reimplementing it a third time.
# ---------------------------------------------------------------------------
def discriminability_metrics(integ: NCRIntegration, embed: torch.nn.Embedding, o: torch.Tensor,
                              entity_ids: torch.Tensor, tgt_slot: torch.Tensor) -> dict:
    """entity_ids: (B,K) int64 token ids, the K in-document entities
    (batch["entity_ids"], audited construction in
    ncr_lm_wave1_smoke.build_task1_document). tgt_slot: (B,) int64, index
    into K identifying which of the K entities is the TRUE answer
    (batch["tgt_slot"]). K=24 for this Wave-1 task (K_NCR).

    Returns a dict with FOUR discriminability metrics (sec G3-B26 audit
    deliverable 1 (a)-(d)):
      offtarget_margin: mean_B[ cos(o, adapted TRUE target) -
        mean_{k != true}(cos(o, adapted target_k)) ] -- the audit's own
        "margin" number; ~0 means `o` cannot tell the true entity apart
        from the 23 wrong ones (sec G3-B26: measured -0.00008..+0.00027 on
        mob_g3b24_s0, i.e. ZERO discriminative signal).
      retrieval24_acc: fraction of batch items where argmax_k cos(o,
        target_k) picks the TRUE entity -- a 24-way nearest-neighbor
        retrieval accuracy IN THE SAME o/adapter space recovered_frac@0.9
        already lives in, chance = 1/24 ~= 0.0417.
      o_pairwise_cos: mean pairwise cosine of `o` ACROSS the eval batch
        (off-diagonal only) -- a collapse monitor on the READ itself; ~1.0
        means `o` is (near-)one fixed vector regardless of input document
        (sec G3-B26's mechanism finding).
      target_pairwise_cos: mean pairwise cosine of the adapted per-document
        targets (entity_adapter(embed(entity_ids))), flattened across the
        batch's B*K target vectors and subsampled to <=512 (matches the
        audit script's own subsample cap -- a MONITOR statistic, not a
        pass/fail gate, so trading exactness for O(1) cost at large
        eval_batch_size is the same tradeoff the audit script itself made)
        -- a collapse monitor on entity_adapter ITSELF, independent of `o`;
        ~1.0 means the target space has collapsed (sec G3-B26: raw GPT-2
        entity embeds pairwise cos 0.0837, trained-adapter 0.9960).
    """
    B, K = entity_ids.shape
    T = integ.entity_adapter(embed(entity_ids).float())            # (B,K,d_ncr) -- targets in o's own space
    on = F.normalize(o.float(), dim=-1)                            # (B,d_ncr)
    Tn = F.normalize(T, dim=-1)                                    # (B,K,d_ncr)
    cos_all = torch.einsum("bd,bkd->bk", on, Tn)                   # (B,K) -- o vs EVERY in-doc entity's target
    cos_true = cos_all.gather(1, tgt_slot[:, None]).squeeze(1)     # (B,)
    mask_true = F.one_hot(tgt_slot, K).bool()
    cos_off = cos_all.masked_fill(mask_true, 0.0).sum(1) / (K - 1)  # (B,) -- mean over the K-1 WRONG targets
    retrieval24_acc = (cos_all.argmax(1) == tgt_slot).float().mean().item()
    offtarget_margin = (cos_true.mean() - cos_off.mean()).item()

    pc_o = on @ on.t()                                             # (B,B)
    off_diag_o = ~torch.eye(B, dtype=torch.bool, device=on.device)
    o_pairwise_cos = pc_o[off_diag_o].mean().item() if B > 1 else float("nan")

    Tf = Tn.reshape(B * K, -1)
    n_sub = min(512, Tf.shape[0])                                  # audit script's own subsample cap
    idx = torch.randperm(Tf.shape[0], device=Tf.device)[:n_sub]
    Ts = Tf[idx]
    pc_t = Ts @ Ts.t()
    off_diag_t = ~torch.eye(Ts.shape[0], dtype=torch.bool, device=Ts.device)
    target_pairwise_cos = pc_t[off_diag_t].mean().item() if n_sub > 1 else float("nan")

    return dict(offtarget_margin=float(offtarget_margin), retrieval24_acc=float(retrieval24_acc),
                o_pairwise_cos=float(o_pairwise_cos), target_pairwise_cos=float(target_pairwise_cos))


# ---------------------------------------------------------------------------
# sec G3-B31 SAME-OP ASSERTION (build agent, 2026-07-29 -- item 5 of the
# G3-B31 build brief's "design open-question ruling": the contrastive-aux
# design (sec G3-B30) implicitly ASSUMES the read-target computation
# (entity_adapter(embed(answer_token)), used by aux_read_supervision_loss/
# discriminability_metrics/contrastive_read_supervision_loss below) and the
# write-key computation (integ.extract_kv's own keys_v =
# entity_adapter(embed(key_ids))) are the SAME tensor op applied to the
# SAME underlying entity id -- code-inspection of ncr_lm_wave1_smoke.py's
# NCRIntegration class (sec G3-B12 single-space fix) says so (ONE shared
# `entity_adapter` nn.Linear instance, both call sites read raw
# `backbone.embed` context-free), but "the docstrings say so" is not the
# same as "verified on a real batch, every launch" -- report loudly rather
# than assume, per the build brief. Two checks, both EXACT (CLAUDE.md:
# structural correctness checks need exact thresholds, never tolerance):
# (1) key_ids (extract_kv's own id source, gathered from the bind-clause
# KEY token positions) must equal batch['entity_ids'] (the canonical
# per-slot id tensor discriminability_metrics/contrastive_read_supervision_
# loss/aux_read_supervision_loss all key off of) -- if this fails, the two
# computations are reading DIFFERENT ids before entity_adapter is even
# applied, and nothing downstream can be trusted; (2)
# entity_adapter(embed(key_ids)) (write pathway, extract_kv's own
# construction) must be BIT-IDENTICAL (torch.equal, not allclose) to
# entity_adapter(embed(entity_ids)) (read-target pathway, T in
# discriminability_metrics/the contrastive loss below) -- if (1) passes but
# (2) fails, the two call sites are using different entity_adapter
# instances / embed tables / dtypes despite matching ids, which the
# NCRIntegration docstrings do not anticipate.
# ---------------------------------------------------------------------------
def assert_read_target_write_key_same_op(integ: NCRIntegration, embed: torch.nn.Embedding,
                                          batch: dict, arm_name: str) -> None:
    key_ids = torch.gather(batch["doc"][:, :-1], 1, batch["key_pos"])   # write pathway's own id source
    assert torch.equal(key_ids, batch["entity_ids"]), (
        f"SAME-OP ASSERTION FAILED (arm={arm_name}): extract_kv's key_ids (gathered from the "
        f"bind-clause KEY token positions) are NOT identical to batch['entity_ids'] (the canonical "
        f"per-slot entity ids the read-target/discriminability/contrastive computations use) -- the "
        f"read-target and write-key computations are reading DIFFERENT entity id tensors even BEFORE "
        f"entity_adapter/embed is applied. Do not assume the two paths share an op -- the divergence "
        f"starts upstream of it.")
    with torch.no_grad():
        keys_v_direct = integ.entity_adapter(embed(key_ids).float())              # write pathway (extract_kv's own construction)
        target_direct = integ.entity_adapter(embed(batch["entity_ids"]).float())  # read-target pathway (discriminability_metrics' T)
        assert torch.equal(keys_v_direct, target_direct), (
            f"SAME-OP ASSERTION FAILED (arm={arm_name}): entity_adapter(embed(key_ids)) (write-key "
            f"pathway) is NOT bit-identical to entity_adapter(embed(entity_ids)) (read-target pathway) "
            f"despite key_ids==entity_ids -- the two call sites must be using different entity_adapter "
            f"instances, different embed tables, or a dtype/precision divergence between them. Report "
            f"this loudly; do not assume the two paths coincide just because the source code looks "
            f"like it shares one nn.Linear instance.")


# ---------------------------------------------------------------------------
# sec G3-B17 direct-read-supervision aux loss (fixes the sec G3-B16-diagnosed
# WRITE-LEARNING gap: CE-only indirect signal never taught the encoder to
# write a composing operator in 20K steps; the standalone free-write toy
# that DID converge used exactly this direct cosine read-loss + 32x more
# data). See the module docstring's "sec G3-B17" section for the full
# rationale.
# ---------------------------------------------------------------------------
def aux_read_supervision_loss(integ: NCRIntegration, embed: torch.nn.Embedding, o: torch.Tensor,
                               answer_token: torch.Tensor) -> torch.Tensor:
    """Dense direct supervision on the read output `o` (binexp_read's own
    output, BEFORE any read-ablation override -- callers pass o_raw) toward
    the TRUE h-hop answer entity's re-based target -- the SAME target
    cosine_and_recovered_frac/graft.recovered_frac_at_09 use (o's own
    entity-adapter space, sec G3-B12), duplicated here (not reimplemented
    differently) with ONE deliberate change: the target is explicitly
    `.detach()`-ed. Cosine form (mean(1 - cos)), matching the standalone
    free-write toy's own converged read-loss -- NOT MSE (cosine is
    scale-invariant to o/target magnitude, matching how recovered_frac@0.9
    itself thresholds on cosine, not L2 distance -- an MSE term would also
    pressure `o`'s NORM toward the target's norm, a property binexp_read's
    own repeated-squaring composition has no reason to preserve at every h).

    Grad flow (why detach matters): `o` depends on Z (<- ncr_head.encode <-
    keys_v/values_v <- integ.entity_adapter) and q_key (<- integ.entity_adapter)
    -- so backward THROUGH o already reaches the encoder AND entity_adapter.
    `target_o` is built through the SAME entity_adapter + embed applied to
    the answer token; withOUT detach, backward would ALSO flow through the
    target's own entity_adapter/embed call, training entity_adapter to chase
    a target that is itself moving under the same update (a self-referential
    optimum -- e.g. entity_adapter could collapse all its outputs toward one
    point to trivially satisfy cos=1 everywhere). detach() removes that
    side entirely -- the ONLY path this loss can travel to reach any
    parameter is through `o`, i.e. through the read itself, which is the
    read-supervision this loss is FOR (see the aux-smoke's own sub-test (c),
    an isolated fresh-tensor proof that embed/entity_adapter get NO gradient
    via the target path). Never called for backbone_only (o_raw there is an
    untrained, read-ablated null with no gradient use for its own arm's loss
    either way -- see the module docstring's sec G3-B17 section)."""
    target_o = integ.entity_adapter(embed(answer_token).float()).detach()
    cos = F.cosine_similarity(o, target_o, dim=-1)
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# sec G3-B31 contrastive-aux loss (build agent, 2026-07-29, per sec G3-B30
# design + amendments -- the ROOT-CAUSE fix for the sec G3-B26 READ-COLLAPSE
# finding: aux_read_supervision_loss above has a degenerate optimum
# (collapsing the ENTIRE target space satisfies "1 - cos(o, target)" just
# as well as learning a discriminative read); this term REMOVES that
# optimum structurally instead of hoping SGD avoids it.
#
# sec A2 (mechanism claim, CORRECTED FORM -- do not use the overclaimed
# "gradient actively pushes away from collapse" phrasing anywhere near this
# function): at EXACT collapse (all 24 adapted targets identical), cos(o,
# T_k) is the SAME value for every k by symmetry, so the 24-way softmax
# logits are uniform, L_ctr = log(24) (chance), and dL_ctr/d(each cos) is
# ZERO -- collapse is a NON-ATTRACTING SYMMETRIC SADDLE, not a point with a
# live repelling force. The escape is the COUPLED path: starting from a
# non-collapsed point (this build's PRIMARY/companion-A cells start from
# the sec G3-B28 frozen-init entity_adapter, measured pairwise cos
# 0.065-0.081 -- i.e. training never enters the saddle's neighborhood to
# begin with), any perturbation that improves within-episode discrimination
# strictly lowers L_ctr, and (in the contrastive+cosine form) L_cos does
# not oppose that direction near a well-separated target space. This is
# the SAME correction DELTANET_REALDATA_DESIGN.md sec 14.4's own
# 2026-07-03 audit-fix note applied to the archive's own NCE term -- cite
# both when this ships (sec G3-B30.1's cite obligations).
#
# ALL 24 targets detached (sec G3-B30 design, both the true target AND the
# 23 negatives) -- matching aux_read_supervision_loss's own detach
# rationale immediately above (a self-referential optimum where
# entity_adapter chases a target moving under its own update), and
# necessary regardless of --freeze-entity-adapter: companion B
# (trainable+contrastive+cosine) is the one cell in the sec G3-B30 grid
# where entity_adapter is NOT frozen, so detach is the only thing
# preventing that self-referential collapse route there.
# ---------------------------------------------------------------------------
def contrastive_read_supervision_loss(integ: NCRIntegration, embed: torch.nn.Embedding, o: torch.Tensor,
                                       entity_ids: torch.Tensor, tgt_slot: torch.Tensor,
                                       temperature: float) -> torch.Tensor:
    """L_ctr = -log[ exp(cos(o, T_true)/T) / sum_{k=1..24} exp(cos(o, T_k)/T) ]
    over the K=24 in-document adapted entity targets (sec G3-B30 design,
    exactly). entity_ids: (B,K) int64 (batch['entity_ids']); tgt_slot: (B,)
    int64 index into K (batch['tgt_slot']); temperature: T in the formula
    above (--contrastive-temperature, default 0.07). Reuses the IDENTICAL
    cos_all construction discriminability_metrics() computes above (not
    reimplemented differently -- matches this file's own "duplicate,
    don't drift" convention, see discriminability_metrics's own module
    comment) with ONE deliberate change: T = entity_adapter(embed(
    entity_ids)) is explicitly `.detach()`-ed before the cosine (ALL 24
    targets, true AND negatives -- see the module comment above this
    function for why). `o` is NOT detached -- gradient flows from L_ctr
    through `o` back to Z/q_key/entity_adapter via the read pathway
    exactly as aux_read_supervision_loss's own gradient does; this is a
    cross_entropy over (B,24) logits = cos_all/T against tgt_slot, which
    is algebraically the loss formula above (
    CE(logits,y) = -log[exp(logits_y) / sum_k exp(logits_k)]
    with logits_k = cos(o,T_k)/T)."""
    T = integ.entity_adapter(embed(entity_ids).float()).detach()   # (B,K,d_ncr) -- ALL 24 targets detached
    on = F.normalize(o.float(), dim=-1)                            # (B,d_ncr), grad-carrying
    Tn = F.normalize(T, dim=-1)                                    # (B,K,d_ncr), detached
    cos_all = torch.einsum("bd,bkd->bk", on, Tn)                   # (B,K)
    return F.cross_entropy(cos_all / temperature, tgt_slot)


# ---------------------------------------------------------------------------
# sec G3-B20 orthogonality regularization (coordinator-designed fix for the
# sec G3-B19-diagnosed gap: the aux read-supervision loss above moved the
# encoder-written operator Z from cos~0 to a STABLE cos~0.57-0.65 across
# every depth h=1..61 -- real write-learning traction -- but Z is
# directionally-right, NOT a clean rotation, so Z^h (binexp_read's own
# repeated-squaring composition) accumulates error and the read never
# clears the ~0.9 cosine bar exact recovery needs. Orthogonal matrices
# compose EXACTLY under powering: if Q1, Q2 are orthogonal (Q^T Q = I) then
# so is Q1 Q2 -- (Q1 Q2)^T(Q1 Q2) = Q2^T(Q1^T Q1)Q2 = Q2^T Q2 = I -- so by
# induction Q^h stays orthogonal (norm/angle-preserving) for every h; a
# perfectly-orthogonal Z would compose without any depth-dependent drift.
# This term pushes each batch item's Z toward that regime DIRECTLY (a
# penalty on Z itself), independent of and additive to the aux read-loss
# above (which pressures Z only INDIRECTLY, through the read
# o=binexp_read(Z,...)'s own cosine to the target).
# ---------------------------------------------------------------------------
def ortho_regularization_loss(Z: torch.Tensor) -> torch.Tensor:
    """Z: (B,d,d) -- the encoder-written operator, ncr_head.encode's own
    output (ncr_lm_forward_ablatable's Z, the non-TF path). Called from
    compute_arm_losses on full_graft ONLY -- the arm whose Z is actually
    SGD-trained; backbone_only's Z lives in a separate, frozen-at-init
    ncr_head instance that must stay untouched (sec G3-B5's null-baseline
    control) -- see compute_arm_losses's own docstring for how the
    is_full_graft gate preserves that.

    ortho_loss = mean_B( ||Z^T Z - I_d||_F^2 ) / d^2 -- the per-batch-item
    squared-Frobenius deviation of Z from orthogonality, mean-reduced over
    the batch, then NORMALIZED by d^2 (weight-balance choice, sec G3-B20
    build brief item 4, disclosed here rather than left implicit): at
    d_ncr=25 an UNNORMALIZED ||Z^TZ-I||_F^2 sits at O(d)-O(d^2) magnitude
    for an approximately-randomly-initialized Z -- large enough to swamp
    ce_loss (O(1)-O(10) nats) and aux_loss (O(1), cosine-bounded in [0,2])
    at any --ortho-reg-weight in the same ~0.1-3.0 range
    --aux-read-loss-weight already operates in (see the sec G3-B20 smoke's
    own 3-loss magnitude report for the measured numbers this estimate is
    checked against). Dividing by d^2 rescales the term to O(1) regardless
    of d_ncr's own value, so ONE --ortho-reg-weight in that familiar range
    works without inventing a separate tiny-weight unit system (the
    alternative disclosed in the build brief -- leave it raw, pick a tiny
    weight like 0.01-0.1 instead -- was NOT taken)."""
    b, d, d2 = Z.shape
    assert d == d2, f"ortho_regularization_loss expects square (B,d,d) Z, got {tuple(Z.shape)}"
    eye = torch.eye(d, device=Z.device, dtype=Z.dtype).expand(b, d, d)
    dev = torch.matmul(Z.transpose(-1, -2), Z) - eye
    return (dev * dev).sum(dim=(-2, -1)).mean() / (d * d)


# ---------------------------------------------------------------------------
# sec G3-B31 --freeze-entity-adapter "has teeth" check (build agent,
# 2026-07-29 -- mirrors decode_isolation_probe.py's own
# assert_grad_isolation and this file's own --teacher-force-operator
# ncr_untouched assertion, both called EVERY step the relevant flag is
# active, never a construction-time-only check).
# ---------------------------------------------------------------------------
def assert_entity_adapter_grad_none(arm: dict, arm_name: str) -> None:
    """Run AFTER backward(), BEFORE opt.step(), for BOTH arms whenever
    --freeze-entity-adapter is active: every integ.entity_adapter param
    must show grad is None. For full_graft this proves requires_grad_(False)
    actually stopped gradient (not merely that the flag was passed and
    silently ignored); for backbone_only this proves the freeze layers
    cleanly on top of the PRE-EXISTING read-ablation zero-gradient property
    (assert_read_ablation_is_exact_zero) rather than conflicting with it.
    Loud AssertionError on violation -- CLAUDE.md: structural correctness
    checks need exact thresholds, never silently trusted."""
    for p in arm["integ"].entity_adapter.parameters():
        assert p.grad is None, (
            f"--freeze-entity-adapter step check FAILED for arm={arm_name}: integ.entity_adapter "
            f"received a non-None gradient despite requires_grad_(False) -- the freeze is broken")


def compute_arm_losses(arm: dict, batch: dict, read_ablate: bool, teacher_force: bool,
                        aux_read_loss_weight: float, is_full_graft: bool,
                        ortho_reg_weight: float = 0.0, aux_loss_type: str = "cosine",
                        contrastive_temperature: float = 0.07):
    """sec G3-B17: the ONE shared per-arm forward+loss computation used by
    BOTH the training loop (run_two_arm_cell) and this build's own real-CUDA
    aux-loss smoke (ncr_lm_wave1_aux_smoke.py) -- so the smoke exercises the
    EXACT training code path, not a hand-copied reimplementation that could
    silently drift from it. Returns (total_loss, ce_loss, aux_loss_or_None,
    ortho_loss_or_None, o_raw, aux_components). aux_read_loss_weight <= 0.0
    OR is_full_graft=False: aux_loss is None, the aux branch below never
    runs, so total_loss is left exactly as it was set BEFORE that branch
    (ce_loss, the identical tensor object, not a fresh one) -- no aux op is
    constructed at all.

    sec G3-B20 (orthogonality reg, threaded THE SAME WAY as the aux flag
    just above -- same full_graft-only gating, same "OFF means the branch
    never runs at all" guarantee): ortho_reg_weight <= 0.0 OR
    is_full_graft=False: ortho_loss is None and the ortho branch below never
    runs either, so total_loss is left EXACTLY as the aux branch (or the
    base ce_loss, if aux was also off) already set it. The two flags compose
    INDEPENDENTLY -- each one's OFF state is byte-identical to that flag
    never having been added, regardless of the OTHER flag's value (verified
    directly, not merely assumed, by the sec G3-B20 smoke's own sub-test
    (a)). When ON, ortho_reg_weight applies ortho_regularization_loss to Z
    (the SAME (B,d,d) operator tensor ncr_lm_forward_ablatable already
    returns above -- no extra forward pass, no extra ncr_head.encode call).

    sec G3-B31 (aux_loss_type -- selects WHICH tensor `aux_loss` is, INSIDE
    the pre-existing gate above; does not add a new gate of its own).
    "cosine" (the default): aux_loss = aux_read_supervision_loss(...), the
    EXACT sec G3-B17 call -- BYTE-IDENTICAL to the pre-G3-B31 code path,
    verified by the build's own legacy-parity smoke. "contrastive":
    aux_loss = contrastive_read_supervision_loss(...) (pure 24-way InfoNCE,
    ALL 24 targets detached). "contrastive+cosine" (sec G3-B30 amendment
    A1, the PRIMARY/companion-B loss form): aux_loss = 0.5*L_ctr + 0.5*L_cos
    (both frozen-detached-target); the two sub-losses are ALSO returned,
    as plain floats, in `aux_components` (empty dict for the other two
    types) purely for diagnostic logging -- they do not add anything to
    the training signal beyond what `aux_loss` already carries.

    WEIGHT-COMPOSITION RULING (build-time resolution of a genuine ambiguity
    in sec G3-B30's own prose, flagged prominently for audit to ratify or
    amend -- see the build report): total_loss = ce_loss +
    aux_read_loss_weight * aux_loss in ALL THREE aux_loss_type cases,
    UNCHANGED from the pre-existing formula. For "contrastive+cosine" this
    means aux_read_loss_weight is applied ON TOP OF the 0.5/0.5
    combination -- sec G3-B30's design text writes "aux = 0.5*L_ctr +
    0.5*L_cos" without pinning whether that 0.5/0.5 sits INSIDE the
    existing aux slot (this ruling) or REPLACES aux_read_loss_weight
    entirely (the other plausible reading). This ruling was chosen to
    match sec B24-PARITY: the design's own "CE + aux 0.5 + ortho 0.1
    weights held at B24 values" instruction is read as
    aux_read_loss_weight staying 0.5 (B24's own value) and being applied
    to the (0.5*L_ctr + 0.5*L_cos) combination, so the realized total aux
    gradient scale at the design's own numbers is
    0.5*(0.5*L_ctr + 0.5*L_cos) = 0.25*L_ctr + 0.25*L_cos."""
    logits, o_raw, o_inj, hidden, Z, keys_v, values_v = ncr_lm_forward_ablatable(
        arm["backbone"], arm["ncr"], arm["integ"], batch, read_ablate=read_ablate, teacher_force=teacher_force)
    ce_loss = F.cross_entropy(logits, batch["answer_token"])
    aux_loss = None
    ortho_loss = None
    aux_components: dict = {}
    total_loss = ce_loss
    if is_full_graft and aux_read_loss_weight > 0.0:
        if aux_loss_type == "cosine":
            aux_loss = aux_read_supervision_loss(arm["integ"], arm["backbone"].embed, o_raw, batch["answer_token"])
        elif aux_loss_type == "contrastive":
            aux_loss = contrastive_read_supervision_loss(
                arm["integ"], arm["backbone"].embed, o_raw, batch["entity_ids"], batch["tgt_slot"],
                contrastive_temperature)
        elif aux_loss_type == "contrastive+cosine":
            l_ctr = contrastive_read_supervision_loss(
                arm["integ"], arm["backbone"].embed, o_raw, batch["entity_ids"], batch["tgt_slot"],
                contrastive_temperature)
            l_cos = aux_read_supervision_loss(arm["integ"], arm["backbone"].embed, o_raw, batch["answer_token"])
            aux_loss = 0.5 * l_ctr + 0.5 * l_cos
            aux_components = {"l_ctr": l_ctr.item(), "l_cos": l_cos.item()}
        else:
            raise ValueError(f"unknown --aux-loss-type {aux_loss_type!r}")
        total_loss = ce_loss + aux_read_loss_weight * aux_loss
    if is_full_graft and ortho_reg_weight > 0.0:
        ortho_loss = ortho_regularization_loss(Z)
        total_loss = total_loss + ortho_reg_weight * ortho_loss
    return total_loss, ce_loss, aux_loss, ortho_loss, o_raw, aux_components


# ---------------------------------------------------------------------------
# Two-arm construction -- BIT-IDENTICAL initial weights (same seed reset
# immediately before each arm's construction, same class/order of
# submodule creation) so the ONLY difference between arms across the whole
# run is the read-ablation itself ("same everything else", sec G3-B5).
# ---------------------------------------------------------------------------
def build_arm(vocab_size_total: int, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    backbone = build_backbone(vocab_size=vocab_size_total).to(device)
    ncr_head = build_ncr_head().to(device)
    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter="linear", read_inject="add").to(device)
    return dict(backbone=backbone, ncr=ncr_head, integ=integ)


def build_two_arms(vocab_size_total: int, seed: int, device: str) -> dict:
    full_graft = build_arm(vocab_size_total, seed, device)
    backbone_only = build_arm(vocab_size_total, seed, device)     # SAME seed -> bit-identical init
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
                sum((list(m.named_parameters()) for m in full_graft.values()), []),
                sum((list(m.named_parameters()) for m in backbone_only.values()), [])):
            assert n1 == n2 and torch.equal(p1, p2), (
                f"two-arm init mismatch at {n1}/{n2} -- 'same everything else' violated at construction")
    return dict(full_graft=full_graft, backbone_only=backbone_only)


def arm_params(arm: dict) -> int:
    return sum(p.numel() for m in arm.values() for p in m.parameters())


# ---------------------------------------------------------------------------
# sec G3-B31 --freeze-entity-adapter (build agent, 2026-07-29): the sec
# G3-B28 frozen-init control's own measured basis (pairwise cos 0.065-0.081,
# no collapse -- sec G3-B30 design) for the PRIMARY/companion-A cells.
# ---------------------------------------------------------------------------
def freeze_entity_adapter_(arms: dict) -> None:
    """requires_grad_(False) on integ.entity_adapter's own params, in BOTH
    arms (full_graft AND backbone_only -- backbone_only's entity_adapter
    already receives zero EXPLICIT gradient through the pre-existing
    read-ablation architecture, ncr_lm_forward_ablatable's own docstring,
    but the build brief asks for 'BOTH arms' explicitly, a disclosed
    defensive-symmetry choice, not a no-op given the observation above).
    Idempotent, and MUST be re-called after every checkpoint restore
    (restore_arms_and_opts constructs FRESH nn.Module instances, which
    default every param back to requires_grad=True regardless of what a
    prior process had set before its own checkpoint -- calling this once
    at cold-start does not persist across a resume)."""
    for arm in arms.values():
        for p in arm["integ"].entity_adapter.parameters():
            p.requires_grad_(False)


def build_optimizer(arm: dict, lr: float, freeze_entity_adapter: bool = False) -> torch.optim.Optimizer:
    """freeze_entity_adapter=True (sec G3-B31): EXCLUDES integ.entity_adapter's
    own params from the optimizer's param groups entirely, on top of (not
    instead of) freeze_entity_adapter_()'s requires_grad_(False) -- belt-
    and-suspenders (a requires_grad=False param would already be skipped by
    AdamW's own step() since its .grad never gets set, but explicit
    exclusion by name is the build brief's own stated mechanism and keeps
    the param-group structure legible to anyone reading a checkpoint's
    opt_state without also knowing every param's requires_grad flag)."""
    integ_params = ([p for n, p in arm["integ"].named_parameters() if not n.startswith("entity_adapter.")]
                     if freeze_entity_adapter else list(arm["integ"].parameters()))
    params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + integ_params
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)     # weight_decay=0.0: lm_pretrain_rd.py
                                                                    # smoke()'s own convention, build-time
                                                                    # choice (not pinned by the design text)


# ---------------------------------------------------------------------------
# Eval -- FIXED per-h eval seeds (independent of training step and of which
# arm is being evaluated), so both arms are scored on LITERALLY the same
# held eval documents at every checkpoint.
# ---------------------------------------------------------------------------
EVAL_SEED_OFFSET = 999_983


@torch.no_grad()
def eval_arm_at_hops(arm: dict, pools, cfg, hops: tuple, batch_size: int, device: str,
                      base_seed: int, read_ablate: bool, teacher_force: bool = False) -> dict:
    backbone, ncr_head, integ = arm["backbone"], arm["ncr"], arm["integ"]
    backbone.eval(); ncr_head.eval(); integ.eval()
    out = {}
    for h in hops:
        gen = torch.Generator(device=device).manual_seed(base_seed + EVAL_SEED_OFFSET + h)
        batch = build_task1_document(cfg, pools, gen, batch_size, h, device)
        logits, o_raw, o_inj, hidden, Z, keys_v, values_v = ncr_lm_forward_ablatable(
            backbone, ncr_head, integ, batch, read_ablate=read_ablate, teacher_force=teacher_force)
        rf, mean_cos = cosine_and_recovered_frac(integ, backbone.embed, o_raw, batch["answer_token"])
        # sec G3-B26 R (BLOCKING RULE): discriminative metrics, additive alongside
        # recovered_frac@0.9/mean_cos above (neither renamed nor removed) -- see
        # discriminability_metrics's own module comment for why rec@0.9/mean_cos alone
        # cannot distinguish "read learned the true entity" from "target space collapsed".
        disc = discriminability_metrics(integ, backbone.embed, o_raw, batch["entity_ids"], batch["tgt_slot"])
        acc = (logits.argmax(dim=-1) == batch["answer_token"]).float().mean().item()
        out[f"h={h}"] = {"recovered_frac@0.9": float(rf), "mean_cos": float(mean_cos),
                          "answer_accuracy": float(acc), "n": batch_size, **disc}
    rf_vals = [v["recovered_frac@0.9"] for v in out.values()]
    cos_vals = [v["mean_cos"] for v in out.values()]
    acc_vals = [v["answer_accuracy"] for v in out.values()]
    margin_vals = [v["offtarget_margin"] for v in out.values()]
    ret24_vals = [v["retrieval24_acc"] for v in out.values()]
    out["mean_recovered_frac@0.9"] = float(sum(rf_vals) / len(rf_vals))
    out["mean_mean_cos"] = float(sum(cos_vals) / len(cos_vals))
    out["mean_answer_accuracy"] = float(sum(acc_vals) / len(acc_vals))
    out["mean_offtarget_margin"] = float(sum(margin_vals) / len(margin_vals))
    out["mean_retrieval24_acc"] = float(sum(ret24_vals) / len(ret24_vals))
    backbone.train(); ncr_head.train(); integ.train()
    return out


def eval_both_arms(arms: dict, pools, cfg, batch_size: int, device: str, seed: int,
                    teacher_force: bool = False) -> dict:
    """teacher_force (sec G3-B9) is applied to the full_graft arm ONLY --
    backbone_only's o_raw always uses the normal ncr_head.encode() path
    regardless of this flag, preserving its role as the untrained,
    frozen-at-init encoder null baseline (sec G3-B5's own definition of
    what backbone_only's o_raw metric means); swapping in a teacher-forced
    fit for backbone_only too would change that baseline's meaning without
    being asked for, and its o_raw never touches the loss either way."""
    result = {}
    for arm_name, read_ablate in (("full_graft", False), ("backbone_only", True)):
        arm = arms[arm_name]
        tf_this_arm = teacher_force and arm_name == "full_graft"
        result[arm_name] = {
            "in_dist": eval_arm_at_hops(arm, pools, cfg, TRAIN_HOPS, batch_size, device, seed, read_ablate, tf_this_arm),
            "deep": eval_arm_at_hops(arm, pools, cfg, DEEP_LADDER, batch_size, device, seed, read_ablate, tf_this_arm),
            # K-SCALING PATCH R3: the labelled fixed-effective-distance control
            # (residue 4 at every K, same squaring count as h_top). Kept in its
            # OWN block, never merged into "deep" -- it deliberately shares a
            # residue with ladder rung 1 and must never be counted as an
            # independent ladder probe. See kscaling_config's module docstring.
            "fixed_dist": eval_arm_at_hops(arm, pools, cfg, (FIXED_DIST_PROBE,), batch_size,
                                            device, seed, read_ablate, tf_this_arm),
        }
    return result


def build_attribution(eval_result: dict) -> dict:
    """Registers the frozen sec G3-B5 rule TEXT plus the raw numbers a blind
    assessor needs to apply it -- computes NO verdict (blind discipline)."""
    fg, bo = eval_result["full_graft"], eval_result["backbone_only"]
    in_dist_gap = {f"h={h}": fg["in_dist"][f"h={h}"]["recovered_frac@0.9"] - bo["in_dist"][f"h={h}"]["recovered_frac@0.9"]
                   for h in TRAIN_HOPS}
    deep_gap = {f"h={h}": fg["deep"][f"h={h}"]["recovered_frac@0.9"] - bo["deep"][f"h={h}"]["recovered_frac@0.9"]
                for h in DEEP_LADDER}
    # sec G3-B9 diagnostic addition (additive only, does not alter any
    # existing key above): the SAME gap construction but on mean_cos instead
    # of the thresholded recovered_frac@0.9, so a blind assessor can see
    # whether the read carries graded signal even when rec@0.9 floors at 0.
    in_dist_cos_gap = {f"h={h}": fg["in_dist"][f"h={h}"]["mean_cos"] - bo["in_dist"][f"h={h}"]["mean_cos"]
                        for h in TRAIN_HOPS}
    deep_cos_gap = {f"h={h}": fg["deep"][f"h={h}"]["mean_cos"] - bo["deep"][f"h={h}"]["mean_cos"]
                    for h in DEEP_LADDER}
    # sec G3-B26 discriminability addition (additive only, does not alter any
    # existing key above): the SAME gap construction, on the discriminability
    # metrics discriminability_metrics computes above. retrieval24_acc's GAP
    # is the sec G3-B26-adopted PRIMARY signal going forward (see
    # primary_signal_v2_definition below) -- it cannot read 1.0 for an
    # information-free read the way recovered_frac@0.9 can once the target
    # space collapses, because a collapsed target space makes ALL 24 targets
    # look alike, which DEGRADES (not inflates) a 24-way argmax retrieval.
    retrieval24_gap_in_dist = {f"h={h}": fg["in_dist"][f"h={h}"]["retrieval24_acc"] - bo["in_dist"][f"h={h}"]["retrieval24_acc"]
                                for h in TRAIN_HOPS}
    retrieval24_gap_deep = {f"h={h}": fg["deep"][f"h={h}"]["retrieval24_acc"] - bo["deep"][f"h={h}"]["retrieval24_acc"]
                             for h in DEEP_LADDER}
    offtarget_margin_gap_in_dist = {f"h={h}": fg["in_dist"][f"h={h}"]["offtarget_margin"] - bo["in_dist"][f"h={h}"]["offtarget_margin"]
                                     for h in TRAIN_HOPS}
    offtarget_margin_gap_deep = {f"h={h}": fg["deep"][f"h={h}"]["offtarget_margin"] - bo["deep"][f"h={h}"]["offtarget_margin"]
                                  for h in DEEP_LADDER}
    # Collapse monitors -- reported PER ARM, not gapped (a collapse in EITHER
    # arm's o/target space is diagnostic on its own; backbone_only's own
    # entity_adapter/read never trains -- sec G3-B5 -- so it is the frozen-
    # at-init control level these should be compared against, sec G3-B26's
    # own "init-adapter control: no collapse" check).
    o_pairwise_cos_in_dist = {arm_name: {f"h={h}": eval_result[arm_name]["in_dist"][f"h={h}"]["o_pairwise_cos"] for h in TRAIN_HOPS}
                               for arm_name in ("full_graft", "backbone_only")}
    o_pairwise_cos_deep = {arm_name: {f"h={h}": eval_result[arm_name]["deep"][f"h={h}"]["o_pairwise_cos"] for h in DEEP_LADDER}
                            for arm_name in ("full_graft", "backbone_only")}
    target_pairwise_cos_in_dist = {arm_name: {f"h={h}": eval_result[arm_name]["in_dist"][f"h={h}"]["target_pairwise_cos"] for h in TRAIN_HOPS}
                                    for arm_name in ("full_graft", "backbone_only")}
    target_pairwise_cos_deep = {arm_name: {f"h={h}": eval_result[arm_name]["deep"][f"h={h}"]["target_pairwise_cos"] for h in DEEP_LADDER}
                                 for arm_name in ("full_graft", "backbone_only")}
    return {
        "frozen_rule_text": ATTRIBUTION_RULE_TEXT,
        "primary_signal_definition": (
            "recovered_frac@0.9 GAP (full_graft.o_raw - backbone_only.o_raw), at DEEP composition "
            "depth (h in {5,12,20,29,40,61}). backbone_only.o_raw's write/encode pathway receives NO "
            "gradient (read-ablated -- see read_ablation_check), so it is a frozen-at-init NULL "
            "baseline for the recovery metric, not a trained comparison."
        ),
        "recovered_frac_gap_in_dist": in_dist_gap,
        "recovered_frac_gap_deep": deep_gap,
        "primary_signal_deepest_gap": deep_gap[f"h={H_TOP}"],
        "primary_signal_deepest_gap_h": H_TOP,
        "primary_signal_deepest_gap_residue": H_TOP % K_NCR,
        "mean_cos_gap_in_dist": in_dist_cos_gap,     # sec G3-B9 diagnostic addition
        "mean_cos_gap_deep": deep_cos_gap,           # sec G3-B9 diagnostic addition
        "attribution_precondition_metric": "answer_accuracy (argmax(logits)==answer_token), NOT recovered_frac@0.9 "
                                            "-- this is what the frozen rule text's own prose names",
        "answer_accuracy_in_dist": {"full_graft": {h: fg["in_dist"][h]["answer_accuracy"] for h in fg["in_dist"] if h.startswith("h=")},
                                     "backbone_only": {h: bo["in_dist"][h]["answer_accuracy"] for h in bo["in_dist"] if h.startswith("h=")}},
        "answer_accuracy_deep": {"full_graft": {h: fg["deep"][h]["answer_accuracy"] for h in fg["deep"] if h.startswith("h=")},
                                  "backbone_only": {h: bo["deep"][h]["answer_accuracy"] for h in bo["deep"] if h.startswith("h=")}},
        # sec G3-B26 additions below (additive-only; recovered_frac@0.9's own fields above are
        # UNCHANGED/kept for continuity, not superseded in-place -- see matrix-thinking/
        # NCR_REAL_LM_DESIGN.md sec G3-B26 for the saturation finding this responds to).
        "primary_signal_v2_definition": (
            "retrieval24_acc GAP (full_graft.o_raw - backbone_only.o_raw), at DEEP composition "
            f"depth (h in {list(DEEP_LADDER)}, this K's own residue-verified ladder) -- the sec "
            "G3-B26-adopted PRIMARY signal for future "
            "attribution, replacing recovered_frac@0.9's gap above (kept for continuity, not "
            "removed) because recovered_frac@0.9/mean_cos can read 1.0/~1.0 for an "
            "information-free read once the entity_adapter target space itself collapses "
            "(pairwise cos ~1.0 across all 24 in-document targets, sec G3-B26) -- retrieval24_acc "
            "is a 24-way discriminative test a collapsed target space cannot pass by accident "
            "(argmax over 24 near-identical directions reads AT CHANCE, not at 1.0)."
        ),
        "retrieval24_acc_gap_in_dist": retrieval24_gap_in_dist,
        "retrieval24_acc_gap_deep": retrieval24_gap_deep,
        "primary_signal_v2_deepest_gap": retrieval24_gap_deep[f"h={H_TOP}"],
        "primary_signal_v2_deepest_gap_h": H_TOP,
        "primary_signal_v2_deepest_gap_residue": H_TOP % K_NCR,
        "primary_signal_v2_chance": KS.CHANCE,
        "primary_signal_v2_fixed_dist_probe_h": FIXED_DIST_PROBE,
        "offtarget_margin_gap_in_dist": offtarget_margin_gap_in_dist,
        "offtarget_margin_gap_deep": offtarget_margin_gap_deep,
        "o_pairwise_cos_in_dist": o_pairwise_cos_in_dist,             # collapse monitor, per arm, NOT gapped
        "o_pairwise_cos_deep": o_pairwise_cos_deep,
        "target_pairwise_cos_in_dist": target_pairwise_cos_in_dist,   # adapter-collapse monitor, per arm, NOT gapped
        "target_pairwise_cos_deep": target_pairwise_cos_deep,
    }


# ---------------------------------------------------------------------------
# Checkpoint save/load -- TRUE step-level resume (a deliberate, disclosed
# strengthening of this repo's own toy-cell "whole-cell skip-if-COMPLETED,
# else restart from scratch" precedent, ncr_ortho_fallback_stage0_v3.py's
# run_disc_cell/run_primary_cell -- justified because this cell is priced at
# multiple contended GPU-hours, not the toy cells' ~1-3 GPU-h, and the build
# brief explicitly requires "checkpoint/resume works" as its own smoke item,
# distinct from "ceiling fires correctly"). Generator states are saved so a
# resumed run reproduces the EXACT same data stream, not merely "a" stream.
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, step: int, arms: dict, opts: dict, data_gen: torch.Generator,
                     cumulative_elapsed_s: float, cell_id: str, seed: int,
                     freeze_entity_adapter: bool = False) -> None:
    """sec G3-B26 seed trap fix: `seed` is now recorded in the checkpoint (was absent --
    build_grammar_pools_and_cfg(seed=...) is called BEFORE the checkpoint-exists check in
    run_two_arm_cell and is NOT itself saved, so a resume launched with a DIFFERENT --seed
    silently rebuilt a different entity-token vocabulary while the ckpt-restored embedding
    table still reflected the ORIGINAL seed's entity assignment -- confirmed 100% silent by
    the audit). run_two_arm_cell's resume path asserts against this field -- see its own
    seed-mismatch assert immediately after load_checkpoint().

    sec G3-B31: `freeze_entity_adapter` is recorded for the SAME reason (the
    seed trap's own pattern applied to a NEW CLI flag that changes both the
    optimizer's own param-group structure -- build_optimizer's
    freeze_entity_adapter arg -- and requires_grad state -- neither of
    which is itself saved/restored by opt_state/model state_dicts):
    resuming a --freeze-entity-adapter run WITHOUT the flag (or vice versa)
    would rebuild an optimizer with a DIFFERENT param-group shape than the
    saved opt_state was fit against. run_two_arm_cell's resume path asserts
    against this field the same way it already asserts against `seed`."""
    ckpt = {
        "runner_tag": RUNNER_TAG, "cell_id": cell_id, "step": step, "seed": seed,
        "freeze_entity_adapter": freeze_entity_adapter,
        "cumulative_elapsed_s": cumulative_elapsed_s,
        "data_gen_state": data_gen.get_state(),
    }
    for arm_name, arm in arms.items():
        ckpt[arm_name] = {
            "backbone_state": arm["backbone"].state_dict(), "backbone_config": arm["backbone"].config(),
            "ncr_state": arm["ncr"].state_dict(), "ncr_config": {"d": D_NCR, "h": H_NCR},
            "integ_state": arm["integ"].state_dict(), "integ_config": arm["integ"].config(),
            "opt_state": opts[arm_name].state_dict(),
        }
    atomic_torch_save(path, ckpt)


def load_checkpoint(path: str, device: str) -> dict | None:
    """Validity-checked resume (CLAUDE.md: 'counts as done only if the
    output parses and has the expected keys, not just file exists')."""
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device)
        assert ckpt.get("runner_tag") == RUNNER_TAG and "step" in ckpt and "full_graft" in ckpt and "backbone_only" in ckpt
        return ckpt
    except Exception as e:
        print(f"  [checkpoint] {path} failed to load/validate ({e!r}) -- treating as ABSENT, starting fresh", flush=True)
        return None


def restore_arms_and_opts(ckpt: dict, vocab_size_total: int, lr: float, device: str,
                           freeze_entity_adapter: bool = False) -> tuple[dict, dict, torch.Generator]:
    """sec G3-B31: `freeze_entity_adapter` must match the value the checkpoint's
    own opt_state was built with (run_two_arm_cell asserts this against
    ckpt['freeze_entity_adapter'] BEFORE calling here, same pattern as the
    seed-mismatch assert) -- passed through to build_optimizer so the
    restored optimizer's param groups have the SAME shape opt_state.load_
    state_dict() expects, and freeze_entity_adapter_() is re-applied to the
    freshly-constructed NCRIntegration instances below (fresh nn.Module
    construction defaults requires_grad back to True regardless of what the
    checkpointing process had set)."""
    arms, opts = {}, {}
    for arm_name in ("full_graft", "backbone_only"):
        c = ckpt[arm_name]
        backbone = graft.DeltaNetLM(**c["backbone_config"]).to(device)
        backbone.load_state_dict(c["backbone_state"])
        ncr_head = graft.els.NCREarlyLNModel(**c["ncr_config"]).to(device)
        ncr_head.load_state_dict(c["ncr_state"])
        integ_cfg = c["integ_config"]
        integ = NCRIntegration(integ_cfg["d_model"], integ_cfg["d_ncr"], integ_cfg["vocab_size"],
                                adapter=integ_cfg["adapter"], read_inject=integ_cfg["read_inject"]).to(device)
        integ.load_state_dict(c["integ_state"])
        arm = dict(backbone=backbone, ncr=ncr_head, integ=integ)
        arms[arm_name] = arm
    if freeze_entity_adapter:
        freeze_entity_adapter_(arms)     # re-apply BEFORE building optimizers, fresh nn.Module construction above defaults requires_grad back to True
    for arm_name in ("full_graft", "backbone_only"):
        opt = build_optimizer(arms[arm_name], lr, freeze_entity_adapter)
        opt.load_state_dict(ckpt[arm_name]["opt_state"])
        opts[arm_name] = opt
    data_gen = torch.Generator(device=device)
    # torch.load(map_location="cuda") in load_checkpoint moves EVERY saved
    # tensor -- including the generator's own state ByteTensor -- onto the
    # GPU; Generator.set_state() requires a CPU ByteTensor (a torch RNG state
    # is always CPU-resident regardless of the generator's device), so a raw
    # set_state(cuda_bytetensor) raises "RNG state must be a torch.ByteTensor".
    # Force it back to a CPU uint8 ByteTensor before restoring. (Caught by the
    # runner-smoke's own sub-test B, sec G3-B6 -- the resume path is exactly
    # what that smoke exists to exercise.)
    gen_state = ckpt["data_gen_state"]
    if gen_state.device.type != "cpu":
        gen_state = gen_state.cpu()
    data_gen.set_state(gen_state.to(torch.uint8))
    return arms, opts, data_gen


# ---------------------------------------------------------------------------
# Main two-arm training+eval cell.
# ---------------------------------------------------------------------------
def run_two_arm_cell(cell_id: str, steps: int, batch_size: int, eval_batch_size: int,
                      lr: float, warmup_steps: int, ceiling_gpuh: float, seed: int,
                      device: str, out_path: str, ckpt_path: str, stop_file: str,
                      ckpt_every: int, eval_every: int, teacher_force_operator: bool = False,
                      aux_read_loss_weight: float = 0.0, ortho_reg_weight: float = 0.0,
                      aux_loss_type: str = "cosine", contrastive_temperature: float = 0.07,
                      freeze_entity_adapter: bool = False) -> dict:
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"[{cell_id}] already COMPLETED -- skipping (resume-safe)", flush=True)
            return prev

    pools, cfg, pool_report = build_grammar_pools_and_cfg(seed=seed)
    vocab_size_total = pool_report["vocab_size_total"]
    pools = pools.to(device)

    ckpt = load_checkpoint(ckpt_path, device)
    if ckpt is not None:
        # sec G3-B26 seed trap fix: build_grammar_pools_and_cfg(seed=seed) above already ran
        # BEFORE this check (unavoidable -- vocab_size_total is needed either way) and is NOT
        # itself saved in the checkpoint, so a resume launched with a DIFFERENT --seed would
        # silently rebuild a DIFFERENT entity-token vocabulary while the ckpt-restored embedding
        # table still reflects the ORIGINAL seed's entity assignment -- confirmed 100% silent by
        # the audit (nothing previously asserted this). ckpt.get("seed", seed) defaults to the
        # CURRENT --seed for checkpoints saved BEFORE this patch (no "seed" key present), so old
        # checkpoints resume exactly as before (trivially seed==seed) -- only NEW checkpoints
        # (which now always carry their own seed, see save_checkpoint) are actually checked.
        ckpt_seed = ckpt.get("seed", seed)
        assert ckpt_seed == seed, (
            f"[{cell_id}] SEED MISMATCH on resume: checkpoint {ckpt_path!r} was built with "
            f"seed={ckpt_seed} but this launch passed --seed {seed}. Resuming would silently "
            f"rebuild a DIFFERENT entity pool (build_grammar_pools_and_cfg(seed=...) is not "
            f"itself saved in the checkpoint and regenerates the entity-token vocabulary fresh "
            f"from ONLY the --seed CLI arg every launch, sec G3-B26) while the ckpt-restored "
            f"embedding table still reflects the ORIGINAL seed's entity assignment -- corrupting "
            f"entity-id semantics with no other signal. Re-launch with --seed {ckpt_seed} to "
            f"resume this checkpoint, or point --ckpt-dir at a fresh path to start a new run "
            f"with seed {seed}.")
        # sec G3-B31: SAME pattern as the seed-mismatch assert immediately above, applied to
        # --freeze-entity-adapter -- see save_checkpoint's own docstring for why a mismatch here
        # would corrupt the resumed optimizer's param-group shape, not merely its semantics.
        ckpt_freeze = ckpt.get("freeze_entity_adapter", freeze_entity_adapter)
        assert ckpt_freeze == freeze_entity_adapter, (
            f"[{cell_id}] --freeze-entity-adapter MISMATCH on resume: checkpoint {ckpt_path!r} was "
            f"built with freeze_entity_adapter={ckpt_freeze} but this launch passed "
            f"freeze_entity_adapter={freeze_entity_adapter}. Resuming would rebuild an optimizer "
            f"with a DIFFERENT param-group shape than the saved opt_state was fit against (sec "
            f"G3-B31, the seed-trap fix's own pattern applied to this new flag). Re-launch with "
            f"the matching flag state, or point --ckpt-dir at a fresh path.")
        print(f"[{cell_id}] RESUMING from checkpoint at step {ckpt['step']} "
              f"(cumulative_elapsed_s={ckpt['cumulative_elapsed_s']:.0f}, seed={ckpt_seed} verified, "
              f"freeze_entity_adapter={ckpt_freeze} verified)", flush=True)
        arms, opts, data_gen = restore_arms_and_opts(ckpt, vocab_size_total, lr, device, freeze_entity_adapter)
        start_step = ckpt["step"]
        cumulative_elapsed_s = ckpt["cumulative_elapsed_s"]
    else:
        arms = build_two_arms(vocab_size_total, seed, device)
        if freeze_entity_adapter:
            freeze_entity_adapter_(arms)     # sec G3-B31: BEFORE building optimizers, see freeze_entity_adapter_'s own docstring
        opts = {name: build_optimizer(arm, lr, freeze_entity_adapter) for name, arm in arms.items()}
        data_gen = torch.Generator(device=device).manual_seed(seed + 777)
        start_step = 0
        cumulative_elapsed_s = 0.0

    n_params = {name: arm_params(arm) for name, arm in arms.items()}
    assert n_params["full_graft"] == n_params["backbone_only"], "two arms must be param-count-identical"

    rec = dict(
        cell_id=cell_id, runner_tag=RUNNER_TAG, mode="calibration",
        status="RUNNING", step=start_step, steps_target=steps,
        kscaling=KS.provenance(H_NCR, RUNG1_BACKBONE["d_model"]),
        config=dict(K=K_NCR, d_ncr=D_NCR, h_ncr=H_NCR, backbone=RUNG1_BACKBONE,
                    vocab_size_total=vocab_size_total, seed=seed, batch_size=batch_size,
                    eval_batch_size=eval_batch_size, lr=lr, warmup_steps=warmup_steps,
                    train_hops=list(TRAIN_HOPS), deep_ladder=list(DEEP_LADDER),
                    ceiling_gpuh=ceiling_gpuh, teacher_force_operator=teacher_force_operator,
                    aux_read_loss_weight=aux_read_loss_weight, ortho_reg_weight=ortho_reg_weight,
                    aux_loss_type=aux_loss_type, contrastive_temperature=contrastive_temperature,
                    freeze_entity_adapter=freeze_entity_adapter),
        params=dict(per_arm=n_params["full_graft"],
                    backbone=sum(p.numel() for p in arms["full_graft"]["backbone"].parameters()),
                    ncr_head=sum(p.numel() for p in arms["full_graft"]["ncr"].parameters()),
                    integ=sum(p.numel() for p in arms["full_graft"]["integ"].parameters())),
        git_commit=git_commit(), host=socket.gethostname(), device=device,
        torch_version=torch.__version__,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Read-ablation exact-zero check (BEFORE training) -- the property
    # sec G3-B5's whole control rests on. Loud on failure: this is not
    # something a calibration should silently proceed past.
    probe_gen = torch.Generator(device=device).manual_seed(seed + 424242)
    probe_batch = build_task1_document(cfg, pools, probe_gen, 4, TRAIN_HOPS[0], device)
    diff_pre = assert_read_ablation_is_exact_zero(
        arms["backbone_only"]["backbone"], arms["backbone_only"]["ncr"], arms["backbone_only"]["integ"], probe_batch)
    rec["read_ablation_check"] = {"pre_train_max_abs_diff": diff_pre, "pre_train_verified_exact_zero": True}
    print(f"[{cell_id}] read-ablation exact-zero check PASSED (pre-train, max_abs_diff={diff_pre:.2e})", flush=True)

    # sec G3-B31 SAME-OP ASSERTION (build brief item 5) -- BEFORE training, both arms
    # (each arm has its OWN entity_adapter instance, sec G3-B5's "same everything else"
    # construction gives them bit-identical INIT weights but they are separate nn.Module
    # objects, so this is checked per-arm, not assumed to transfer from one to the other).
    for arm_name in ("full_graft", "backbone_only"):
        assert_read_target_write_key_same_op(arms[arm_name]["integ"], arms[arm_name]["backbone"].embed,
                                              probe_batch, arm_name)
    rec["same_op_check"] = {"verified": True, "arms_checked": ["full_graft", "backbone_only"]}
    print(f"[{cell_id}] same-op assertion (read-target vs write-key) PASSED for both arms (pre-train)", flush=True)
    assert steps >= 1, "run_two_arm_cell requires steps >= 1 (use --mode phase0-timing for a zero-training-step rate probe)"

    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time() - cumulative_elapsed_s
    loss_hist = {"full_graft": [], "backbone_only": []}
    n_skipped = {"full_graft": 0, "backbone_only": 0}
    final_status = "COMPLETED"
    # sec G3-B9: counts PASSED per-step encoder-zero-grad assertions (full_graft
    # arm only, see the training loop below) -- NOT restored across a resume
    # (fresh per-process, same convention as n_skipped above), so it reports
    # checks passed IN THIS PROCESS's run only.
    teacher_force_ncr_zero_grad_checks = 0
    rec["teacher_force_check"] = {"active": teacher_force_operator, "ncr_zero_grad_checks_passed": 0}

    for step in range(start_step + 1, steps + 1):
        cur_lr = get_lr(step, max_lr=lr, warmup_steps=warmup_steps, total_steps=steps)
        idx = torch.randint(0, len(TRAIN_HOPS), (1,), generator=data_gen, device=device).item()
        hop_value = TRAIN_HOPS[idx]
        batch = build_task1_document(cfg, pools, data_gen, batch_size, hop_value, device)

        step_losses = {}
        step_aux_losses = {}   # sec G3-B17: full_graft only, populated only when aux_read_loss_weight > 0.0
        step_ortho_losses = {}  # sec G3-B20: full_graft only, populated only when ortho_reg_weight > 0.0
        step_aux_components = {}  # sec G3-B31: full_graft only, populated only for aux_loss_type=="contrastive+cosine"
        for arm_name, read_ablate in (("full_graft", False), ("backbone_only", True)):
            arm, opt = arms[arm_name], opts[arm_name]
            for g in opt.param_groups:
                g["lr"] = cur_lr
            # sec G3-B9: teacher_force applies to full_graft ONLY (see
            # eval_both_arms's own docstring for the identical rationale --
            # backbone_only's o_raw stays the untrained-encoder null baseline
            # regardless of this flag, and never touches its own loss anyway).
            tf_this_arm = teacher_force_operator and arm_name == "full_graft"
            # sec G3-B17/sec G3-B20/sec G3-B31: aux_read_loss_weight, ortho_reg_weight,
            # aux_loss_type, and contrastive_temperature all apply to full_graft ONLY,
            # same scoping/rationale as teacher_force above -- see compute_arm_losses's
            # own docstring for the byte-identical-when-OFF guarantee (each flag
            # independently, regardless of the other flags' values).
            total_loss, ce_loss, aux_loss, ortho_loss, o_raw, aux_components = compute_arm_losses(
                arm, batch, read_ablate, tf_this_arm, aux_read_loss_weight, arm_name == "full_graft",
                ortho_reg_weight, aux_loss_type, contrastive_temperature)
            opt.zero_grad()
            total_loss.backward()
            if tf_this_arm:
                # sec G3-B9 isolation proof, ported from ncr_lm_wave1_smoke.py smoke
                # item 10's construction-time check into this training loop, run
                # EVERY step this mode is active (not merely once): teacher-forcing
                # bypasses ncr_head.encode() entirely (see ncr_lm_forward_ablatable),
                # so ncr_head's parameters must receive EXACTLY zero gradient. Loud
                # AssertionError on violation -- CLAUDE.md: structural correctness
                # checks need exact thresholds, never silently trusted.
                ncr_untouched = all(p.grad is None for p in arm["ncr"].parameters())
                assert ncr_untouched, (
                    f"--teacher-force-operator step {step}: ncr_head (BindingEncoder) received a "
                    f"non-None gradient -- teacher-force isolation broken, the encoder pathway is "
                    f"not actually bypassed")
                teacher_force_ncr_zero_grad_checks += 1
            if freeze_entity_adapter:
                # sec G3-B31 "has teeth" check, EVERY step, BOTH arms -- see
                # assert_entity_adapter_grad_none's own docstring.
                assert_entity_adapter_grad_none(arm, arm_name)
            all_params = list(arm["backbone"].parameters()) + list(arm["ncr"].parameters()) + list(arm["integ"].parameters())
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in all_params)
            if finite:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
            else:
                n_skipped[arm_name] += 1
            step_losses[arm_name] = ce_loss.item()
            if aux_loss is not None:
                step_aux_losses[arm_name] = aux_loss.item()
            if ortho_loss is not None:
                step_ortho_losses[arm_name] = ortho_loss.item()
            if aux_components:
                step_aux_components[arm_name] = aux_components

        if step % LOG_EVERY == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            loss_hist["full_graft"].append([step, step_losses["full_graft"]])
            loss_hist["backbone_only"].append([step, step_losses["backbone_only"]])
            # sec G3-B17/sec G3-B20: aux_loss/ortho_loss are TRAINING losses (like CE),
            # not eval metrics -- OK to print (BLIND discipline only withholds EVAL
            # metrics, per the module docstring's BLIND DISCIPLINE section below).
            # Each is absent from the line entirely when its own weight==0.0
            # (step_aux_losses/step_ortho_losses stay empty every step in that case).
            aux_str = (f"  full_graft_aux_loss={step_aux_losses['full_graft']:.4f}"
                       if "full_graft" in step_aux_losses else "")
            ortho_str = (f"  full_graft_ortho_loss={step_ortho_losses['full_graft']:.4f}"
                         if "full_graft" in step_ortho_losses else "")
            # sec G3-B31: l_ctr/l_cos sub-loss breakdown, TRAINING losses like aux_loss
            # itself (not an eval metric) -- absent entirely unless aux_loss_type ==
            # "contrastive+cosine" (step_aux_components stays empty every step otherwise).
            comp_str = (f"  full_graft_l_ctr={step_aux_components['full_graft']['l_ctr']:.4f}"
                        f"  full_graft_l_cos={step_aux_components['full_graft']['l_cos']:.4f}"
                        if "full_graft" in step_aux_components else "")
            # BLIND: loss is operational telemetry (liveness/divergence), never an eval metric --
            # matches ncr_ortho_fallback_stage0_v3.py's own precedent.
            print(f"[{cell_id}] step {step}/{steps}  full_graft_loss={step_losses['full_graft']:.4f} "
                  f"backbone_only_loss={step_losses['backbone_only']:.4f}  lr={cur_lr:.2e}  "
                  f"{elapsed:.0f}s{aux_str}{ortho_str}{comp_str}", flush=True)

        if step % ckpt_every == 0 or step == steps:
            save_checkpoint(ckpt_path, step, arms, opts, data_gen, time.time() - t0, cell_id, seed,
                             freeze_entity_adapter)

        if step % eval_every == 0 or step == steps:
            # sec G3-B31 item 7 (integrity monitor wiring, CONFIRMED not patched --
            # target_pairwise_cos was already inside discriminability_metrics's own return
            # dict, sec G3-B26, and eval_both_arms -> eval_arm_at_hops calls it every time
            # THIS branch runs, i.e. every eval_every steps DURING training, not just at
            # the final write below): rec["attribution"]["target_pairwise_cos_in_dist"/
            # "_deep"] is live at every eval point from here, giving the sec G3-B30 band-1
            # integrity check (frozen entity_adapter must stay near the 0.065-0.081
            # frozen-init basis throughout training) something to watch, not just a
            # single post-hoc number.
            eval_result = eval_both_arms(arms, pools, cfg, eval_batch_size, device, seed,
                                          teacher_force=teacher_force_operator)
            rec["arms"] = eval_result
            rec["attribution"] = build_attribution(eval_result)
            rec["step"] = step
            rec["elapsed_s"] = time.time() - t0
            rec["loss_history"] = loss_hist
            rec["n_skipped_steps"] = n_skipped
            rec["teacher_force_check"]["ncr_zero_grad_checks_passed"] = teacher_force_ncr_zero_grad_checks
            atomic_write_json(out_path, rec)
            print(f"[{cell_id}] eval computed at step {step} -> {out_path} updated "
                  f"(values withheld from stdout, blind discipline sec G3-B6)", flush=True)

        if stop_requested(stop_file):
            save_checkpoint(ckpt_path, step, arms, opts, data_gen, time.time() - t0, cell_id, seed,
                             freeze_entity_adapter)
            print(f"[{cell_id}] STOP file detected at step {step} -- checkpoint saved, exiting", flush=True)
            sys.exit(3)

        elapsed = time.time() - t0
        if elapsed > ceiling_s:
            final_status = "ABORTED-BUDGET"
            save_checkpoint(ckpt_path, step, arms, opts, data_gen, elapsed, cell_id, seed,
                             freeze_entity_adapter)
            eval_result = eval_both_arms(arms, pools, cfg, eval_batch_size, device, seed,
                                          teacher_force=teacher_force_operator)
            rec["arms"] = eval_result
            rec["attribution"] = build_attribution(eval_result)
            rec["step"] = step
            rec["loss_history"] = loss_hist
            rec["n_skipped_steps"] = n_skipped
            rec["teacher_force_check"]["ncr_zero_grad_checks_passed"] = teacher_force_ncr_zero_grad_checks
            break

    # Post-train read-ablation re-check (paranoia: prove the invariant held
    # THROUGHOUT training, not just at init).
    diff_post = assert_read_ablation_is_exact_zero(
        arms["backbone_only"]["backbone"], arms["backbone_only"]["ncr"], arms["backbone_only"]["integ"], probe_batch)
    rec["read_ablation_check"]["post_train_max_abs_diff"] = diff_post
    rec["read_ablation_check"]["post_train_verified_exact_zero"] = True
    print(f"[{cell_id}] read-ablation exact-zero check PASSED (post-train, max_abs_diff={diff_post:.2e})", flush=True)

    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = final_status
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(out_path, rec)
    print(f"[{cell_id}] {final_status} at step {rec['step']}/{steps} in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


# ---------------------------------------------------------------------------
# Phase-0 timing probe -- measures the REAL per-step wall-clock rate of the
# exact two-arm loop above (not a synthetic/toy timing proxy), at the
# operating point the calibration cell will actually use. Reports (never
# hides -- this is operational telemetry, not the interpretive signal) the
# uncontended rate, the CONTENDED_MULTIPLIER-scaled projection, and a
# suggested --ceiling-gpuh for a given --target-steps.
# ---------------------------------------------------------------------------
def run_phase0_timing(batch_size: int, warmup_steps: int, probe_steps: int, target_steps: int,
                       lr: float, seed: int, device: str, out_path: str) -> dict:
    pools, cfg, pool_report = build_grammar_pools_and_cfg(seed=seed)
    vocab_size_total = pool_report["vocab_size_total"]
    pools = pools.to(device)
    arms = build_two_arms(vocab_size_total, seed, device)
    opts = {name: build_optimizer(arm, lr) for name, arm in arms.items()}
    data_gen = torch.Generator(device=device).manual_seed(seed + 777)

    def one_step():
        idx = torch.randint(0, len(TRAIN_HOPS), (1,), generator=data_gen, device=device).item()
        hop_value = TRAIN_HOPS[idx]
        batch = build_task1_document(cfg, pools, data_gen, batch_size, hop_value, device)
        per_arm_s = {}
        for arm_name, read_ablate in (("full_graft", False), ("backbone_only", True)):
            if device == "cuda":
                torch.cuda.synchronize()
            t_arm0 = time.time()
            arm, opt = arms[arm_name], opts[arm_name]
            logits, o_raw, o_inj, hidden, Z, keys_v, values_v = ncr_lm_forward_ablatable(
                arm["backbone"], arm["ncr"], arm["integ"], batch, read_ablate=read_ablate)
            loss = F.cross_entropy(logits, batch["answer_token"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            per_arm_s[arm_name] = time.time() - t_arm0
        return per_arm_s

    print(f"[phase0-timing] warmup: {warmup_steps} steps (untimed, CUDA kernel compile/cache)...", flush=True)
    for _ in range(warmup_steps):
        one_step()

    print(f"[phase0-timing] probe: {probe_steps} steps (timed)...", flush=True)
    per_step_full, per_step_backbone, per_step_total = [], [], []
    t_probe0 = time.time()
    for i in range(probe_steps):
        s = one_step()
        per_step_full.append(s["full_graft"])
        per_step_backbone.append(s["backbone_only"])
        per_step_total.append(s["full_graft"] + s["backbone_only"])
        if (i + 1) % 10 == 0:
            print(f"[phase0-timing] {i+1}/{probe_steps} probe steps done ({time.time()-t_probe0:.1f}s so far)", flush=True)
    wall = time.time() - t_probe0

    mean_full = sum(per_step_full) / len(per_step_full)
    mean_backbone = sum(per_step_backbone) / len(per_step_backbone)
    mean_total = sum(per_step_total) / len(per_step_total)
    doc_len = cfg.T_bind + cfg.query_len          # the fixed document length (independent of hop value)
    tokens_per_step_per_arm = batch_size * doc_len

    contended_total_s_per_step = mean_total * CONTENDED_MULTIPLIER
    projected_uncontended_s = target_steps * mean_total
    projected_contended_s = target_steps * contended_total_s_per_step
    suggested_ceiling_gpuh = round(projected_contended_s / 3600.0 * 1.15, 3)   # +15% pad, matches
                                                                                  # stage0_v3's own internal-
                                                                                  # ceiling-vs-external-backstop ratio

    rec = dict(
        runner_tag=RUNNER_TAG, mode="phase0-timing",
        kscaling=KS.provenance(H_NCR, RUNG1_BACKBONE["d_model"]),
        config=dict(K=K_NCR, d_ncr=D_NCR, backbone=RUNG1_BACKBONE, vocab_size_total=vocab_size_total,
                    seed=seed, batch_size=batch_size, warmup_steps=warmup_steps, probe_steps=probe_steps,
                    target_steps=target_steps, lr=lr, doc_len=doc_len),
        measured=dict(
            mean_s_per_step_full_graft=mean_full, mean_s_per_step_backbone_only=mean_backbone,
            mean_s_per_step_both_arms_combined=mean_total,
            tokens_per_step_per_arm=tokens_per_step_per_arm,
            tokens_per_sec_per_arm_full_graft=tokens_per_step_per_arm / mean_full,
            tokens_per_sec_per_arm_backbone_only=tokens_per_step_per_arm / mean_backbone,
            probe_wall_clock_s=wall,
        ),
        projected=dict(
            contended_multiplier=CONTENDED_MULTIPLIER,
            uncontended_s_for_target_steps=projected_uncontended_s,
            uncontended_gpuh_for_target_steps=projected_uncontended_s / 3600.0,
            contended_s_for_target_steps=projected_contended_s,
            contended_gpuh_for_target_steps=projected_contended_s / 3600.0,
            suggested_ceiling_gpuh=suggested_ceiling_gpuh,
        ),
        host=socket.gethostname(), device=device, torch_version=torch.__version__,
        git_commit=git_commit(),
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    atomic_write_json(out_path, rec)
    print("=" * 70, flush=True)
    print(f"[phase0-timing] MEASURED (uncontended): {mean_total:.4f} s/step (both arms combined), "
          f"full_graft={mean_full:.4f}s backbone_only={mean_backbone:.4f}s", flush=True)
    print(f"[phase0-timing] PROJECTED contended ({CONTENDED_MULTIPLIER}x) rate for "
          f"target_steps={target_steps}: {projected_contended_s/3600.0:.3f} GPU-h", flush=True)
    print(f"[phase0-timing] SUGGESTED --ceiling-gpuh {suggested_ceiling_gpuh} "
          f"(contended projection + 15% pad)", flush=True)
    print(f"[phase0-timing] results written to {out_path}", flush=True)
    print("=" * 70, flush=True)
    return rec


# ---------------------------------------------------------------------------
# Runner smoke -- THIS build's OWN required real-CUDA smoke of the RUNNER
# (not the graft, which sec G3-B3/B4 already smoked/audited). Drives the
# script via REAL subprocess invocations of --mode calibration (the
# faithful way the tmux supervisor will actually invoke it), not in-process
# function calls, so the resume test exercises a genuine fresh-process
# checkpoint load.
# ---------------------------------------------------------------------------
def run_runner_smoke(device: str, outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    py = sys.executable
    script = os.path.abspath(__file__)
    failures = []

    def invoke(mode, cell_id, steps, ceiling_gpuh, extra=()):
        out_path = os.path.join(outdir, f"{cell_id}.json")
        ckpt_path = os.path.join(outdir, f"{cell_id}.ckpt.pt")
        cmd = [py, script, "--mode", mode, "--device", device, "--cell-id", cell_id,
               "--steps", str(steps), "--ceiling-gpuh", str(ceiling_gpuh),
               "--batch-size", "8", "--eval-batch-size", "16", "--warmup-steps", "10",
               "--ckpt-every", "50", "--eval-every", "50",
               "--out", out_path, "--ckpt-dir", outdir, "--seed", "0", *extra]
        print(f"\n[runner-smoke] SUBPROCESS: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout[-4000:], flush=True)
        if proc.returncode not in (0,):
            print(f"[runner-smoke] subprocess stderr:\n{proc.stderr[-4000:]}", file=sys.stderr, flush=True)
        wall = time.time() - t0
        rec = None
        if os.path.exists(out_path):
            with open(out_path) as f:
                rec = json.load(f)
        return proc.returncode, rec, wall

    # --- sub-test A: a full short run (300 steps), start-to-finish ---------
    cid_a = "runner_smoke_main"
    rc_a, rec_a, wall_a = invoke("calibration", cid_a, 300, 5.0)
    ok_a = (rc_a == 0 and rec_a is not None and rec_a.get("status") == "COMPLETED"
            and rec_a.get("step") == 300 and "attribution" in rec_a and "arms" in rec_a
            and rec_a["read_ablation_check"].get("pre_train_verified_exact_zero") is True
            and rec_a["read_ablation_check"].get("post_train_verified_exact_zero") is True
            and rec_a["read_ablation_check"]["pre_train_max_abs_diff"] == 0.0
            and rec_a["read_ablation_check"]["post_train_max_abs_diff"] == 0.0
            and "recovered_frac_gap_deep" in rec_a["attribution"]
            and "answer_accuracy_in_dist" in rec_a["attribution"])
    print(f"[runner-smoke A: full 300-step run, both arms train, attribution fields present, "
          f"read-ablation exact-zero pre+post] {'PASS' if ok_a else 'FAIL'} (wall={wall_a:.1f}s)", flush=True)
    if not ok_a:
        failures.append("A:full-run")

    # --- sub-test B: checkpoint/resume -- run to step 100, kill, resume to 300 ---
    cid_b = "runner_smoke_resume"
    rc_b1, rec_b1, wall_b1 = invoke("calibration", cid_b, 100, 5.0)
    ok_b1 = rc_b1 == 0 and rec_b1 is not None and rec_b1.get("status") == "COMPLETED" and rec_b1.get("step") == 100
    # A FRESH subprocess, pointed at the same out/ckpt paths but a HIGHER
    # step target -- since out_path already reads status=COMPLETED at
    # step=100, the whole-cell skip would normally fire; delete the results
    # JSON (simulating "the supervisor is re-launched with a raised step
    # target after review") while leaving the CHECKPOINT in place, so the
    # resume path (not the whole-cell-skip path) is what gets exercised.
    out_path_b = os.path.join(outdir, f"{cid_b}.json")
    if os.path.exists(out_path_b):
        os.remove(out_path_b)
    rc_b2, rec_b2, wall_b2 = invoke("calibration", cid_b, 300, 5.0)
    ok_b2 = (rc_b2 == 0 and rec_b2 is not None and rec_b2.get("status") == "COMPLETED"
             and rec_b2.get("step") == 300)
    # resume correctness: the second invocation's own stdout must show it
    # RESUMED from step 100 (not restarted from step 0) -- re-run with
    # captured stdout to check for the "RESUMING from checkpoint at step
    # 100" line (invoke() already printed it above; re-derive from the
    # elapsed step count instead, which is a stronger, code-level check):
    # loss_history's first logged step in the second invocation's OWN run
    # should start at >100, not at 1, IF resume truly continued rather than
    # restarted (a restart would re-log step=1..300 from scratch).
    resumed_correctly = False
    if rec_b2 is not None:
        fg_hist = rec_b2.get("loss_history", {}).get("full_graft", [])
        first_logged_step = fg_hist[0][0] if fg_hist else None
        # LOG_EVERY=25 and ckpt/eval at 50 -- if truly resumed from 100, the
        # NEXT loop iteration is step 101, and the first log/ckpt/eval line
        # this SECOND process itself would hit is step 125 (next LOG_EVERY
        # multiple >100) or the final step=300 catch-all. A cold restart
        # would instead log step=1 first. So first_logged_step>100 proves resume.
        resumed_correctly = first_logged_step is not None and first_logged_step > 100
    ok_b = ok_b1 and ok_b2 and resumed_correctly
    print(f"[runner-smoke B: checkpoint/resume -- run to 100 (fresh proc), delete results JSON only, "
          f"re-invoke to 300 (fresh proc) -- must RESUME from ckpt not restart] {'PASS' if ok_b else 'FAIL'} "
          f"(resumed_correctly={resumed_correctly}, wall={wall_b1+wall_b2:.1f}s)", flush=True)
    if not ok_b:
        failures.append("B:checkpoint-resume")

    # --- sub-test C: ceiling fires correctly (tiny ceiling, expects ABORTED-BUDGET well before target) ---
    cid_c = "runner_smoke_ceiling"
    rc_c, rec_c, wall_c = invoke("calibration", cid_c, 5000, 0.0003)   # 0.0003 GPU-h ~= 1.1s
    ok_c = (rc_c == 0 and rec_c is not None and rec_c.get("status") == "ABORTED-BUDGET"
            and rec_c.get("step", 10**9) < 5000)
    print(f"[runner-smoke C: tiny --ceiling-gpuh 0.0003 must ABORT-BUDGET well before steps=5000] "
          f"{'PASS' if ok_c else 'FAIL'} (final_step={rec_c.get('step') if rec_c else None}, wall={wall_c:.1f}s)", flush=True)
    if not ok_c:
        failures.append("C:ceiling")

    # --- sub-test D: whole-cell skip-if-COMPLETED (re-invoke A's cell, must skip instantly) ---
    t0 = time.time()
    rc_d, rec_d, wall_d = invoke("calibration", cid_a, 300, 5.0)
    ok_d = rc_d == 0 and rec_d is not None and rec_d.get("status") == "COMPLETED" and wall_d < wall_a
    print(f"[runner-smoke D: re-invoking a COMPLETED cell must skip instantly, not re-train] "
          f"{'PASS' if ok_d else 'FAIL'} (wall={wall_d:.1f}s vs original {wall_a:.1f}s)", flush=True)
    if not ok_d:
        failures.append("D:whole-cell-skip")

    # --- sub-test E: --teacher-force-operator diagnostic mode (sec G3-B9) ---
    cid_e = "runner_smoke_teacher_force"
    rc_e1, rec_e1, wall_e1 = invoke("calibration", cid_e, 100, 5.0, extra=("--teacher-force-operator",))

    def _mean_cos_present(rec) -> bool:
        if rec is None or "arms" not in rec:
            return False
        try:
            return all("mean_cos" in rec["arms"][arm][band][f"h={h}"]
                       for arm in ("full_graft", "backbone_only")
                       for band, hops in (("in_dist", TRAIN_HOPS), ("deep", DEEP_LADDER))
                       for h in hops)
        except (KeyError, TypeError):
            return False

    ok_e1 = (rc_e1 == 0 and rec_e1 is not None and rec_e1.get("status") == "COMPLETED"
             and rec_e1.get("step") == 100
             and rec_e1.get("config", {}).get("teacher_force_operator") is True
             and rec_e1.get("teacher_force_check", {}).get("active") is True
             and rec_e1.get("teacher_force_check", {}).get("ncr_zero_grad_checks_passed", 0) > 0
             and _mean_cos_present(rec_e1))
    print(f"[runner-smoke E1: --teacher-force-operator, 100-step run -- COMPLETED without the "
          f"encoder-zero-grad AssertionError firing (checks_passed>0), mean_cos present at every "
          f"arm/band/h] {'PASS' if ok_e1 else 'FAIL'} "
          f"(checks_passed={rec_e1.get('teacher_force_check', {}).get('ncr_zero_grad_checks_passed') if rec_e1 else None}, "
          f"wall={wall_e1:.1f}s)", flush=True)
    if not ok_e1:
        failures.append("E1:teacher-force-run")

    # checkpoint/resume under teacher-force: delete results JSON (keep ckpt), re-invoke to a
    # higher step target with the SAME flag -- must RESUME (not restart), and the per-process
    # encoder-zero-grad check count must be > 0 again post-resume (proving the assertion re-ran
    # in the fresh process, not silently skipped).
    out_path_e = os.path.join(outdir, f"{cid_e}.json")
    if os.path.exists(out_path_e):
        os.remove(out_path_e)
    rc_e2, rec_e2, wall_e2 = invoke("calibration", cid_e, 200, 5.0, extra=("--teacher-force-operator",))
    resumed_e = False
    if rec_e2 is not None:
        fg_hist_e = rec_e2.get("loss_history", {}).get("full_graft", [])
        first_logged_step_e = fg_hist_e[0][0] if fg_hist_e else None
        resumed_e = first_logged_step_e is not None and first_logged_step_e > 100
    ok_e2 = (rc_e2 == 0 and rec_e2 is not None and rec_e2.get("status") == "COMPLETED"
             and rec_e2.get("step") == 200 and resumed_e
             and rec_e2.get("teacher_force_check", {}).get("ncr_zero_grad_checks_passed", 0) > 0
             and _mean_cos_present(rec_e2))
    print(f"[runner-smoke E2: --teacher-force-operator checkpoint/resume -- must RESUME from 100 to "
          f"200 (not restart), encoder-zero-grad checks keep passing post-resume, mean_cos still "
          f"present] {'PASS' if ok_e2 else 'FAIL'} (resumed={resumed_e}, wall={wall_e2:.1f}s)", flush=True)
    if not ok_e2:
        failures.append("E2:teacher-force-resume")

    summary = dict(runner_tag=RUNNER_TAG, mode="smoke", failures=failures,
                    sub_results=dict(A=rec_a, B_first=rec_b1, B_second=rec_b2, C=rec_c, D=rec_d,
                                      E1=rec_e1, E2=rec_e2),
                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_write_json(os.path.join(outdir, "runner_smoke_summary.json"), summary)
    print("=" * 70, flush=True)
    if failures:
        print(f"RUNNER SMOKE: {len(failures)} FAILURE(S): {failures}", file=sys.stderr, flush=True)
    else:
        print("RUNNER SMOKE: ALL 6 SUB-TESTS PASSED", flush=True)
    print("=" * 70, flush=True)
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("phase0-timing", "calibration", "smoke"))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20_000,
                     help="sec 6.2's own stated Phase-1 reduced-calibration step budget, reused directly "
                          "(disclosed transfer -- this cell's docs are shorter than sec 6.1's seq_len=512 "
                          "real-corpus operating point the number was originally token-budgeted against)")
    ap.add_argument("--ceiling-gpuh", type=float, default=None,
                     help="REQUIRED for --mode calibration (no silent default -- must be set from a "
                          "--mode phase0-timing measurement, contended-priced)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--probe-steps", type=int, default=50, help="--mode phase0-timing only")
    ap.add_argument("--target-steps", type=int, default=20_000, help="--mode phase0-timing only: "
                     "the step count to project the contended GPU-h ceiling for")
    ap.add_argument("--k", type=int, required=True,
                     help="K-SCALING PATCH R7: mandatory restatement of NCR_K. Must equal the "
                          "env var kscaling_config read; asserted in main(). Not a second source "
                          "of truth -- a tripwire against env/flag drift across 30 specs.")
    ap.add_argument("--cell-id", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--teacher-force-operator", action="store_true",
                     help="sec G3-B9 diagnostic mode (--mode calibration only): replace the "
                          "ncr_head-encoded operator Z with the closed-form least-squares "
                          "teacher-forced fit (ncr_lm_wave1_smoke.py's own audited "
                          "teacher_force_operator, smoke item 10) in BOTH the training loss and "
                          "eval, for the full_graft arm ONLY -- backbone_only is unaffected (its "
                          "read stays exact-zero and its o_raw stays the untrained-encoder null "
                          "baseline regardless of this flag). Isolates write-learning (the "
                          "encoder) from read/inject/backbone-learning: LEARNS in-distribution "
                          "under this flag => the encoder (write side) is the blocker; STAYS at "
                          "chance => the read-injection or task/loss setup is broken. The encoder "
                          "is asserted to receive EXACTLY zero gradient every step this is active "
                          "(loud AssertionError on violation, never silently trusted).")
    ap.add_argument("--aux-read-loss-weight", type=float, default=0.0,
                     help="sec G3-B17 direct-read-supervision auxiliary loss (--mode calibration "
                          "only, full_graft arm ONLY -- backbone_only stays CE-only; its o_raw is "
                          "the untrained/read-ablated null baseline sec G3-B5's attribution rule "
                          "depends on). 0.0 (default) = OFF, byte-identical to pre-G3-B17 CE-only "
                          "behavior (no aux op is constructed at all, see compute_arm_losses's own "
                          "docstring). >0.0: total_loss = ce_loss + weight * mean(1 - "
                          "cosine_similarity(o_raw, entity_adapter(embed(answer_token)).detach())) "
                          "-- dense direct supervision toward the TRUE h-hop answer entity's own "
                          "re-based target (target DETACHED: trains the encoder/read, never the "
                          "target itself, see aux_read_supervision_loss's docstring). Fixes the "
                          "sec G3-B16-diagnosed WRITE-LEARNING gap (CE-only indirect signal, 20K "
                          "steps, ~32x under the free-write toy's own convergence budget) by "
                          "matching that toy's own converged direct cosine read-loss.")
    ap.add_argument("--ortho-reg-weight", type=float, default=0.0,
                     help="sec G3-B20 orthogonality regularization on the encoder-written "
                          "operator Z (--mode calibration only, full_graft arm ONLY -- same "
                          "scoping as --aux-read-loss-weight; backbone_only's Z lives in a "
                          "separate frozen-at-init ncr_head instance that must stay untouched). "
                          "0.0 (default) = OFF, byte-identical to pre-G3-B20 behavior (no ortho "
                          "op is constructed at all, independent of --aux-read-loss-weight's own "
                          "value -- see compute_arm_losses's own docstring). >0.0: total_loss += "
                          "weight * mean_B(||Z^T Z - I_d||_F^2) / d^2 (d=d_ncr=25, NORMALIZED so "
                          "the term is O(1) at a weight in the same range --aux-read-loss-weight "
                          "already uses -- see ortho_regularization_loss's own docstring for the "
                          "disclosed weight-balance rationale). Fixes the sec G3-B19-diagnosed "
                          "gap (aux supervision alone reached a STABLE but NOT-exact cos~0.57-0.65 "
                          "read across all depths h=1..61): orthogonal matrices compose EXACTLY "
                          "under binexp_read's own repeated-squaring powering, so pushing Z toward "
                          "orthogonal should let a directionally-right write become an exact one.")
    ap.add_argument("--aux-loss-type", default="cosine", choices=("cosine", "contrastive", "contrastive+cosine"),
                     help="sec G3-B31 (contrastive-aux re-spec, per sec G3-B30 design + coordinator "
                          "amendments) -- --mode calibration only, full_graft arm ONLY, gated by "
                          "--aux-read-loss-weight > 0.0 exactly like the pre-existing aux slot (this "
                          "flag selects WHICH tensor fills that slot, it does not add a new gate). "
                          "'cosine' (default): aux_loss = aux_read_supervision_loss(...), the EXACT "
                          "sec G3-B17 call -- BYTE-IDENTICAL to pre-G3-B31 behavior, verified by this "
                          "build's own legacy-parity smoke. 'contrastive': aux_loss = "
                          "contrastive_read_supervision_loss(...) -- pure 24-way InfoNCE over the "
                          "in-document adapted entity targets, ALL 24 targets detached, fixes the "
                          "sec G3-B26 READ-COLLAPSE degenerate optimum STRUCTURALLY (at exact "
                          "collapse this term reads log(24), chance, with ZERO gradient -- a "
                          "non-attracting symmetric saddle, sec G3-B30 amendment A2's corrected "
                          "mechanism claim, NOT 'gradient pushes away from collapse'). "
                          "'contrastive+cosine' (sec G3-B30 amendment A1, the PRIMARY/companion-B "
                          "loss form): aux_loss = 0.5*L_ctr + 0.5*L_cos (both frozen-detached-target) "
                          "-- see compute_arm_losses's own docstring for the exact weight-composition "
                          "ruling (--aux-read-loss-weight is applied ON TOP of this 0.5/0.5 "
                          "combination).")
    ap.add_argument("--contrastive-temperature", type=float, default=0.07,
                     help="sec G3-B31/sec G3-B30: softmax temperature T in "
                          "contrastive_read_supervision_loss's L_ctr = -log[exp(cos(o,T_true)/T) / "
                          "sum_k exp(cos(o,T_k)/T)]. Only used when --aux-loss-type is 'contrastive' "
                          "or 'contrastive+cosine'. Default 0.07 is sec G3-B30's own pre-registered "
                          "value; its disclosed fallback 0.03 is config-only -- pass "
                          "--contrastive-temperature 0.03 explicitly to use it.")
    ap.add_argument("--freeze-entity-adapter", action="store_true",
                     help="sec G3-B31/sec G3-B30: freezes integ.entity_adapter (requires_grad_(False), "
                          "excluded from the optimizer's own param groups) in BOTH arms, for the "
                          "PRIMARY/companion-A cells -- the sec G3-B28 frozen-init control's own "
                          "measured basis (pairwise cos 0.065-0.081, no collapse). Asserted "
                          "grad-is-None on entity_adapter's own params EVERY step this is active "
                          "(assert_entity_adapter_grad_none, mirrors decode_isolation_probe.py's own "
                          "assert_grad_isolation pattern) -- loud AssertionError, never silently "
                          "trusted. Default OFF (companion-B cell, trainable+contrastive+cosine, "
                          "leaves this unset).")
    args = ap.parse_args()

    # K-SCALING PATCH R6b: --k is a MANDATORY, redundant restatement of the NCR_K
    # environment variable that kscaling_config actually reads. A spec whose env
    # and flag disagree (the single easiest way to launch 30 cells at the wrong K)
    # dies HERE -- before any GPU work, on every mode -- instead of producing
    # plausible, wrongly-labelled numbers.
    assert args.k == K_NCR, (
        f"--k {args.k} disagrees with NCR_K={os.environ.get('NCR_K')!r} "
        f"(kscaling_config resolved K_NCR={K_NCR}). Refusing to run: one of the two is a "
        f"typo and the results would be silently mislabelled.")
    print(f"[kscaling] K={K_NCR} d_ncr={D_NCR} chance={KS.CHANCE:.4f} ladder={DEEP_LADDER} "
          f"h_top={H_TOP} (residue {H_TOP % K_NCR} == K/2) fixed_dist_probe={FIXED_DIST_PROBE} "
          f"t_in={KS.t_in()} doc_left_pad={KS.doc_left_pad()}", flush=True)

    print("=" * 70)
    print(f"NCR REAL-LM WAVE-1 CALIBRATION RUNNER (sec G3-B6) mode={args.mode} device={args.device} "
          f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 70)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.mode == "phase0-timing":
        rec = run_phase0_timing(args.batch_size, args.warmup_steps, args.probe_steps, args.target_steps,
                                 args.lr, args.seed, args.device, args.out)
        return 0

    if args.mode == "smoke":
        outdir = os.path.dirname(os.path.abspath(args.out)) or "."
        return run_runner_smoke(args.device, outdir)

    # --- calibration ---
    assert args.ceiling_gpuh is not None, (
        "--ceiling-gpuh is REQUIRED for --mode calibration -- run --mode phase0-timing first and pass "
        "its suggested_ceiling_gpuh explicitly (no silent default; sec G3-B6)")
    ckpt_dir = args.ckpt_dir or os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    cell_id = args.cell_id or f"kscaling_K{K_NCR}_s{args.seed}"
    ckpt_path = os.path.join(ckpt_dir, f"{cell_id}.ckpt.pt")
    stop_file = args.stop_file or os.path.join(ckpt_dir, "STOP")

    rec = run_two_arm_cell(
        cell_id=cell_id, steps=args.steps, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, lr=args.lr, warmup_steps=args.warmup_steps,
        ceiling_gpuh=args.ceiling_gpuh, seed=args.seed, device=args.device,
        out_path=args.out, ckpt_path=ckpt_path, stop_file=stop_file,
        ckpt_every=args.ckpt_every, eval_every=args.eval_every,
        teacher_force_operator=args.teacher_force_operator,
        aux_read_loss_weight=args.aux_read_loss_weight,
        ortho_reg_weight=args.ortho_reg_weight,
        aux_loss_type=args.aux_loss_type,
        contrastive_temperature=args.contrastive_temperature,
        freeze_entity_adapter=args.freeze_entity_adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
